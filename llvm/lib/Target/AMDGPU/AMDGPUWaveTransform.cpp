//===- AMDGPUWaveTransform.cpp - AMDGPU Wave Transform ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief Transform a function from thread-level to wave-level control flow
///
/// This pass is responsible for:
/// - Building the wave-level reconverging CFG and selecting corresponding
///   branch instructions.
/// - Constructing execmasks
///
/// TODO: In GlobalISel, this pass is additionally responsible for assigning
///       uniform vs. divergent register banks(?)
///
///
/// \section Reconvergence transform
///
/// The reconvergence transform establishes the "reconverging" property for the
/// CFG:
///
///   Every block with divergent terminator has exactly two successors, one of
///   which is a post-dominator.
///
/// The post-dominator is called "secondary" successor. During execution, the
/// wave of execution will first branch to the "primary" successor (if there
/// are any threads that want to go down that path), while adding the other
/// threads to a "rejoin mask" associated with the secondary successor. Since
/// it is a post-dominator, the wave is guaranteed to reach the secondary
/// successor eventually, at which point the threads from the "rejoin mask"
/// are added back to the wave.
///
/// The secondary successor will often be a newly introduced "flow block",
/// as in a simple hammock with divergent terminator at A:
///
///     A                 A
///    / \                |\
///   B   C     ===>      | B
///    \ /                |/
///     D                 X
///                       |\
///                       | C
///                       |/
///                       D
///
/// The reconvergence algorithm traverses blocks in heart-adjusted reverse post
/// order (HARPO), i.e. blocks of every cycle are contiguous, and the cycle's
/// heart is visited first (or the header, if there is no heart).
///
/// Flow blocks are inserted when a visited block has a predecessor with
/// divergent terminator that requires a flow block for the reconverging
/// property.
///
//
// TODO-NOW:
//  - uniform in cycle / divergent outside
//  - double-check order of successor nodes for divergent WaveNode
//
// TODO:
//  - _actually_ implement HARPO
//  - multiple function return blocks
//  - complex heart regions:
//  -- multiple backward edges from within the pre-heart region
//  -- multiple backward edges _into_ the pre-heart region
//  -- second pass of core transform with post-heart regions rotated to the
//     front
//  -- extra flow nodes for back edges in the pre-heart region?
//  -- problem of entry into the heart region: do the "second pass" of the
//     core transform first?
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPULaneMaskUtils.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/IntEqClasses.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineCycleAnalysis.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Debug.h"
#include <variant>

using namespace llvm;

#define DEBUG_TYPE "amdgpu-wave-transform"

STATISTIC(
    NumCleanupInstrsRemoved,
    "Number of instructions removed during amdgpu-wave-transform cleanup");
STATISTIC(
    NumCleanupInstrsOptimized,
    "Number of instructions optimized during amdgpu-wave-transform cleanup");

static cl::opt<bool>
    AMDGPUWaveTransformPrintFinal("amdgpu-wave-transform-print-final",
                               cl::desc("Print the final wave CFG"),
                               cl::init(false));

namespace {

static bool isHardwareManagedBranch(MachineInstr &MI) {
  switch (MI.getOpcode()) {
  case AMDGPU::S_SUBVECTOR_LOOP_BEGIN:
  case AMDGPU::S_SUBVECTOR_LOOP_END:
    return true;
  default:
    return false;
  }
}

static bool isArtificialTerminator(MachineInstr &MI) {
  // Return true for instructions that are marked as terminators to support
  // special exec management.
  switch (MI.getOpcode()) {
  case AMDGPU::S_MOV_B64_term:
  case AMDGPU::S_XOR_B64_term:
  case AMDGPU::S_OR_B64_term:
  case AMDGPU::S_ANDN2_B64_term:
  case AMDGPU::S_AND_B64_term:
  case AMDGPU::S_AND_SAVEEXEC_B64_term:
  case AMDGPU::S_MOV_B32_term:
  case AMDGPU::S_XOR_B32_term:
  case AMDGPU::S_OR_B32_term:
  case AMDGPU::S_ANDN2_B32_term:
  case AMDGPU::S_AND_B32_term:
  case AMDGPU::S_AND_SAVEEXEC_B32_term:
  case AMDGPU::V_CMPX_EQ_U32_nosdst_e32_term:
  case AMDGPU::V_CMPX_EQ_U64_nosdst_e32_term:
    return true;
  default:
    return false;
  }
}

struct WaveNode;

using WaveNodeVec = SmallVector<WaveNode *, 4>;
using WaveNodeSet = DenseSet<WaveNode *>;
using AccRegSet = SmallDenseSet<Register, 4>;
using AccInstsMap =
    DenseMap<MachineBasicBlock *, SmallVector<MachineInstr *, 8>>;

// Returns true if MO is a virtual register belonging
// to the accumulator register set.
static inline bool isAccRegOperand(const MachineOperand &MO,
                                   const AccRegSet &AccRegs) {
  return MO.isReg() && MO.getReg().isVirtual() && AccRegs.count(MO.getReg());
}

/// Map a lane-level successor or predecessor to a wave-level successor or
/// predecessor.
struct LaneEdge {
  WaveNode *Lane = nullptr;
  WaveNode *Wave = nullptr;

  LaneEdge() = default;
  LaneEdge(WaveNode *Lane, WaveNode *Wave) : Lane(Lane), Wave(Wave) {}
};

/// \brief Representation of a node / basic block in the wave CFG.
struct WaveNode {
  MachineBasicBlock *Block = nullptr;
  CycleRef Cycle;
  WaveNodeVec Predecessors;
  WaveNodeVec Successors;
  SmallVector<LaneEdge, 4> LanePredecessors;
  SmallVector<LaneEdge, 4> LaneSuccessors;

  bool IsDivergent = false;
  bool IsSecondary = false;
  unsigned OrderIndex = 0;
  unsigned FlowNum = 0; // non-zero if flow node

  /// Used during reconverging algorithm: track known post-dominators on the
  /// fly.
  WaveNode *LatestPostDom = nullptr;

  Register RejoinMask;

  WaveNode(MachineBasicBlock *Block, CycleRef Cycle)
      : Block(Block), Cycle(Cycle), LatestPostDom(this) {}
  WaveNode(CycleRef Cycle, unsigned FlowNum)
      : Cycle(Cycle), FlowNum(FlowNum), LatestPostDom(this) {}
  WaveNode(const WaveNode &) = delete;
  WaveNode(WaveNode &&) = delete;
  WaveNode &operator=(const WaveNode &) = delete;
  WaveNode &operator=(WaveNode &&) = delete;

  Printable printableName() const {
    return Printable([this](raw_ostream &Out) {
      if (Block) {
        // FIXME: remove the need to build a context everytime
        Out << MachineSSAContext().print(Block);
      }
      if (Block && FlowNum)
        Out << '.';
      if (FlowNum)
        Out << "<flow-" << FlowNum << '>';
    });
  }
};

/// \brief Helper class for making a CFG reconverging.
class ReconvergeCFGHelper {
private:
  // MachineConvergenceInfo &ConvergenceInfo;
  MachineCycleInfo &CycleInfo;
  MachineDominatorTree &DomTree;

  unsigned NumFlowNodes = 0;

  /// Current HAPO-ordered list of nodes.
  ///
  /// During individual transform steps, a prefix of this vector may have been
  /// moved to \ref NextNodes already.
  std::vector<std::unique_ptr<WaveNode>> Nodes;

  /// During individual transform steps, a prefix of the next \ref Nodes
  /// vector.
  std::vector<std::unique_ptr<WaveNode>> NextNodes;

  DenseMap<MachineBasicBlock *, WaveNode *> NodeForBlock;

  /// Temporary variables used by \ref appendOpenSet, persisted to reduce
  /// the number of temporary allocations.
  struct {
    SmallVector<WaveNode *, 8> Worklist;
    WaveNodeSet Found;
  } OpenSetScan;

public:
  ReconvergeCFGHelper(MachineCycleInfo &CycleInfo,
                      MachineDominatorTree &DomTree)
      : /*ConvergenceInfo(convergenceInfo),*/
        CycleInfo(CycleInfo), DomTree(DomTree) {}

  void run();

  MachineDominatorTree &getDomTree() { return DomTree; }

  WaveNode *rerouteViaNewNode(ArrayRef<WaveNode *> FromList, WaveNode *ToNode);

  WaveNode *nodeForBlock(MachineBasicBlock *Block) {
    return NodeForBlock.lookup(Block);
  }
  void setNodeForBlock(MachineBasicBlock *Block, WaveNode *Node) {
    assert(!NodeForBlock.count(Block));
    NodeForBlock.try_emplace(Block, Node);
  }

  /// Walk \p BB up to the block before its outermost enclosing cycle header, so
  /// an init placed there runs once before the loop instead of per iteration.
  MachineBasicBlock *hoistBeforeCycle(MachineBasicBlock *BB) {
    while (WaveNode *N = nodeForBlock(BB)) {
      if (!N->Cycle)
        break;
      MachineDomTreeNode *HN = DomTree.getNode(CycleInfo.getHeader(N->Cycle));
      if (!HN || !HN->getIDom())
        break;
      BB = HN->getIDom()->getBlock();
    }
    return BB;
  }

  template <typename WrappedIteratorT, typename WaveNodeT>
  struct node_iterator_impl;

  template <typename WrappedIteratorT, typename WaveNodeT>
  using node_iterator_impl_base = iterator_adaptor_base<
      node_iterator_impl<WrappedIteratorT, WaveNodeT>, WrappedIteratorT,
      typename std::iterator_traits<WrappedIteratorT>::iterator_category,
      WaveNodeT *, // value type
      typename std::iterator_traits<WrappedIteratorT>::difference_type,
      WaveNodeT **, // pointer type
      WaveNodeT *>; // reference type

  template <typename WrappedIteratorT, typename WaveNodeT>
  struct node_iterator_impl
      : node_iterator_impl_base<WrappedIteratorT, WaveNodeT> {
    node_iterator_impl() = default;
    explicit node_iterator_impl(WrappedIteratorT it)
        : node_iterator_impl_base<WrappedIteratorT, WaveNodeT>(it) {}

    WaveNodeT *operator*() const { return this->I->get(); }
  };

  using const_node_iterator =
      node_iterator_impl<std::vector<std::unique_ptr<WaveNode>>::const_iterator,
                         const WaveNode>;
  using node_iterator =
      node_iterator_impl<std::vector<std::unique_ptr<WaveNode>>::const_iterator,
                         WaveNode>;

  const_node_iterator nodes_begin() const {
    return const_node_iterator(Nodes.begin());
  }
  const_node_iterator nodes_end() const {
    return const_node_iterator(Nodes.end());
  }
  iterator_range<const_node_iterator> nodes() const {
    return {nodes_begin(), nodes_end()};
  }

  node_iterator nodes_begin() { return node_iterator(Nodes.begin()); }
  node_iterator nodes_end() { return node_iterator(Nodes.end()); }
  iterator_range<node_iterator> nodes() { return {nodes_begin(), nodes_end()}; }

  void printNodes(raw_ostream &out);
  void dumpNodes();

private:
  void cleanupSimpleFlowNodes();

  MachineBasicBlock *getEffectiveHeart(CycleRef Cycle);
  void prepareNodesEnterCycle(WaveNode *HeaderNode);
  void prepareNodesExitCycle(CycleRef Cycle, WaveNode *NextNode);
  bool appendOpenSet(WaveNode *Grom, WaveNode *Nound,
                     SmallVectorImpl<WaveNode *> &OpenSet);
  void reroute(ArrayRef<WaveNode *> FromList, WaveNode *ToNode,
               WaveNode *ViaNode);
  void rerouteEdgesBeyond(ArrayRef<WaveNode *> from, WaveNode *ToBeyond,
                          WaveNode *ViaNode);
  void rerouteLane(WaveNode *FromNode, WaveNode *ToNode, WaveNode *ViaNode);

  void verifyNodes();
};

} // anonymous namespace

static MachineBasicBlock *getHeartBlock(CycleRef Cycle) { return nullptr; }

/// Check if Outer contains Inner, where Inner may be an invalid CycleRef
/// (i.e. a block not in any cycle). Returns false if Inner is invalid.
static bool cycleContains(const MachineCycleInfo &CI, CycleRef Outer,
                          CycleRef Inner) {
  return Inner.isValid() && CI.contains(Outer, Inner);
}

class HeartAdjustedPostOrder {
public:
  using BlockT = MachineBasicBlock;
  using DominatorTreeT = MachineDominatorTree;
  using CycleInfoT = MachineCycleInfo;
  using const_iterator = typename std::vector<BlockT *>::const_iterator;

  bool empty() const { return Order.empty(); }
  size_t size() const { return Order.size(); }

  void clear() { Order.clear(); }
  // void compute(const ConvergenceInfoT &convergenceInfo,
  //              const CycleInfoT &CycleInfo, const DominatorTreeT &domTree);

  const_iterator begin() const { return Order.begin(); }
  const_iterator end() const { return Order.end(); }
  BlockT *operator[](size_t Idx) const { return Order[Idx]; }

private:
  std::vector<BlockT *> Order;

public:
  void compute(const CycleInfoT &CycleInfo, const DominatorTreeT &domTree) {
    // In our forward traversal, the modification bullets from the description
    // of heart-adjusted reverse post order happen in reverse: within each
    // cycle, we do a depth-first post-order traversal of only the blocks
    // belonging to the cycle, starting with the heart.
    //
    // The depth-first search mainly uses a stack of blocks, with a look-aside
    // stack of cycles. Cycles remain on the stack until their final post-order
    // visit, at which time their Blocks are added to the parent cycle's order.
    // We also maintain a linked list of cycles that are active in the sense
    // that we're currently visiting blocks inside them.
    struct HapoCycle {
      CycleRef Cycle;
      BlockT *Heart;
      unsigned ParentStackIdx;
      std::vector<BlockT *> Order;
      SmallVector<BlockT *, 4> PostponedBlocks;

      explicit HapoCycle(CycleRef Cycle, BlockT *Heart, unsigned ParentStackIdx)
          : Cycle(Cycle), Heart(Heart), ParentStackIdx(ParentStackIdx) {}
    };

    DenseSet<BlockT *> VisitedBlocks;
    SmallVector<BlockT *, 32> BlockStack;
    struct DoneIdxStackElt {
      unsigned SizeOfBlockStackBeforePop : 31;
      bool IsCycleHeart : 1;

      DoneIdxStackElt(unsigned Size, bool IsCycle)
          : SizeOfBlockStackBeforePop(Size), IsCycleHeart(IsCycle) {}
    };
    SmallVector<DoneIdxStackElt, 32> DoneIdxStack;
    SmallVector<HapoCycle, 8> CycleStack;
    unsigned CurrentCycleStackIdx = 0;

    BlockT *EntryBlock = domTree.getRootNode()->getBlock();
    CycleStack.emplace_back(CycleRef(), nullptr, 0);
    BlockStack.push_back(EntryBlock);

    // The entry block is not marked as a cycle header, so that we don't attempt
    // to pop the root cycle: it is handled at the very end after the loop.
    DoneIdxStack.emplace_back(BlockStack.size(), false);
    // Push successors in reverse order so that the DFS stack (LIFO) visits
    // them in forward MBB-successor order. This makes the branch target
    // (typically the lighter/passthrough side of an if-else) the primary
    // successor in the reconvergence CFG, matching the legacy structurizer's
    // block ordering and reducing register pressure for large vector types.
    llvm::append_range(BlockStack, llvm::reverse(successors(EntryBlock)));

    do {
      MachineBasicBlock *Block = BlockStack.back();
      const DoneIdxStackElt &DoneBack = DoneIdxStack.back();

      if (BlockStack.size() == DoneBack.SizeOfBlockStackBeforePop) {
        if (!DoneBack.IsCycleHeart) {
          // Post-order visit of a regular Block.
          CycleStack[CurrentCycleStackIdx].Order.push_back(Block);
          BlockStack.pop_back();
          DoneIdxStack.pop_back();
          continue;
        }

        // This is the post-order visit of an effective Cycle heart.
        HapoCycle &Cycle = CycleStack.back();
        if (CurrentCycleStackIdx == CycleStack.size() - 1)
          CurrentCycleStackIdx = Cycle.ParentStackIdx;

        if (!Cycle.PostponedBlocks.empty()) {
          // Enqueue the Cycle's postponed exit Blocks if there are any. In this
          // case, we aren't actually at the post-order visit of the Cycle yet,
          // if we interpret it as a contracted node contained in its parent.
          for (BlockT *postponed : Cycle.PostponedBlocks) {
            assert(VisitedBlocks.count(postponed));
            VisitedBlocks.erase(postponed);
            BlockStack.push_back(postponed);
          }
          Cycle.PostponedBlocks.clear();
          continue;
        }

        // True post-order visit: collect all of the Cycle.
        Cycle.Order.push_back(Block);
        BlockStack.pop_back();
        DoneIdxStack.pop_back();

        auto &ParentOrder = CycleStack[Cycle.ParentStackIdx].Order;
        ParentOrder.insert(ParentOrder.end(), Cycle.Order.begin(),
                           Cycle.Order.end());
        CycleStack.pop_back();
        continue;
      }

      if (!VisitedBlocks.insert(Block).second) {
        BlockStack.pop_back();
        continue; // already visited this one
      }

      // Pre-order visit of the block.
      CycleRef CurrentCycle = CycleStack[CurrentCycleStackIdx].Cycle;
      BlockT *CurrentHeart = CycleStack[CurrentCycleStackIdx].Heart;
      CycleRef BlockCycle = CycleInfo.getCycle(Block);

      if (BlockCycle == CurrentCycle ||
          (CurrentHeart && CurrentHeart == getHeartBlock(BlockCycle))) {
        DoneIdxStack.emplace_back(BlockStack.size(), false);
        llvm::append_range(BlockStack, llvm::reverse(successors(Block)));
        continue;
      }

      if (!CurrentCycle || cycleContains(CycleInfo, CurrentCycle, BlockCycle)) {
        // Entering a child cycle. In the case of irreducible control flow,
        // BlockCycle might not be a direct child -- find it.
        while ((CycleInfo.getParentCycle(BlockCycle) != CurrentCycle) &&
               (!CurrentHeart ||
                CurrentHeart !=
                    getHeartBlock(CycleInfo.getParentCycle(BlockCycle))))
          BlockCycle = CycleInfo.getParentCycle(BlockCycle);

        BlockT *Heart = getHeartBlock(BlockCycle);
        BlockT *EffectiveHeart =
            Heart ? Heart : CycleInfo.getHeader(BlockCycle);

        CycleStack.emplace_back(BlockCycle, Heart, CurrentCycleStackIdx);
        CurrentCycleStackIdx = CycleStack.size() - 1;

        // Fixup state as-if we're visiting the effective heart.
        if (Block != EffectiveHeart) {
          BlockStack.pop_back();
          BlockStack.push_back(EffectiveHeart);
          VisitedBlocks.erase(Block);
          VisitedBlocks.insert(EffectiveHeart);
        }

        DoneIdxStack.emplace_back(BlockStack.size(), true);
        llvm::append_range(BlockStack, llvm::reverse(successors(Block)));
        continue;
      }

      // This Block is not contained in the current Cycle; we have to postpone
      // it.
      BlockStack.pop_back();

      HapoCycle *PostponeCycle = &CycleStack[CurrentCycleStackIdx];
      for (;;) {
        HapoCycle *parent = &CycleStack[PostponeCycle->ParentStackIdx];
        if (!parent->Cycle ||
            cycleContains(CycleInfo, parent->Cycle, BlockCycle))
          break;
        PostponeCycle = parent;
      }
      PostponeCycle->PostponedBlocks.push_back(Block);
    } while (!BlockStack.empty());

    assert(CycleStack.size() == 1);
    Order = std::move(CycleStack[0].Order);
  }
};

void ReconvergeCFGHelper::run() {
  HeartAdjustedPostOrder hapo;
  hapo.compute(CycleInfo, DomTree);

  // Step 1: Create initial set of WaveNodes mirroring the thread-level CFG.
  Nodes.reserve(hapo.size());
  for (MachineBasicBlock *Block : llvm::reverse(hapo)) {
    Nodes.emplace_back(
        std::make_unique<WaveNode>(Block, CycleInfo.getCycle(Block)));
    WaveNode *WN = Nodes.back().get();
    auto TerminatorMI = Block->getFirstTerminator();
    if (TerminatorMI != Block->end()) {
      WN->IsDivergent = (TerminatorMI->getOpcode() == AMDGPU::SI_BRCOND ||
                         TerminatorMI->getOpcode() == AMDGPU::SI_BRCOND_Z);
    } else {
      // Handle the case where there's no terminator.
      WN->IsDivergent = false;
    }
    NodeForBlock.insert(std::make_pair(Block, WN));
  }

  // Link up CFG edges. Note that we ignore unreachable predecessors.
  for (const auto &NodePtr : Nodes) {
    for (MachineBasicBlock *Succ : NodePtr->Block->successors()) {
      auto SuccNodeIt = NodeForBlock.find(Succ);
      assert(SuccNodeIt != NodeForBlock.end());
      NodePtr->Successors.push_back(SuccNodeIt->second);
      NodePtr->LaneSuccessors.emplace_back(SuccNodeIt->second,
                                           SuccNodeIt->second);
      SuccNodeIt->second->Predecessors.push_back(NodePtr.get());
      SuccNodeIt->second->LanePredecessors.emplace_back(NodePtr.get(),
                                                        NodePtr.get());
    }
  }

  // Step 2: Create helper nodes for cycles:
  //
  // At the end of every maximal cycle for a heart block, reroute every
  // backwards edge within the ordering span of the cycle (i.e., back edge to
  // the header of any cycle with the same heart, or edge from after heart block
  // to before) through a single flow node. (A single flow node
  for (unsigned Index = 0; Index != Nodes.size(); ++Index)
    Nodes[Index]->OrderIndex = Index;

  CycleRef CurrentCycle;

  NextNodes.reserve(Nodes.size());

  for (auto &NodePtr : Nodes) {
    WaveNode *Node = NodePtr.get();

    if (Node->Cycle != CurrentCycle) {
      while (CurrentCycle &&
             !cycleContains(CycleInfo, CurrentCycle, Node->Cycle)) {

        prepareNodesExitCycle(CurrentCycle, Node);
        CurrentCycle = CycleInfo.getParentCycle(CurrentCycle);
      }

      if (Node->Cycle != CurrentCycle) {
        assert(CycleInfo.getParentCycle(Node->Cycle) == CurrentCycle);

        prepareNodesEnterCycle(Node);
        CurrentCycle = Node->Cycle;

      }
    }

    NextNodes.push_back(std::move(NodePtr));
  }
  Nodes = std::move(NextNodes);
  NextNodes.clear();

  // Step 3: Run reconverging transform.
  for (unsigned Index = 0; Index != Nodes.size(); ++Index)
    Nodes[Index]->OrderIndex = Index;

  WaveNodeVec RerouteCandidates;
  IntEqClasses RerouteCandidateClasses;
  SmallVector<int, 4> PredClasses;
  WaveNodeVec RerouteNodes;
  WaveNodeVec RerouteRoots;
  WaveNodeVec TmpSet;
  for (auto &NodePtr : Nodes) {
    WaveNode *Node = NodePtr.get();

    int RerouteClass = -1;
    for (WaveNode *Pred : Node->Predecessors) {
      // Backward edge and predecessors without divergence don't need to
      // establish the reconverging property.
      if (Pred->OrderIndex >= Node->OrderIndex || !Pred->IsDivergent) {
        PredClasses.push_back(-1);
        continue;
      }

      bool HaveEarlierSuccessor = false;
      for (WaveNode *Succ : Pred->Successors) {
        assert(Succ->OrderIndex != Node->OrderIndex || Succ == Node);
        if (Succ->OrderIndex < Node->OrderIndex) {
          HaveEarlierSuccessor = true;
          break;
        }
      }
      if (!HaveEarlierSuccessor) {
        // The current node is going to be the primary successor.
        auto SelfIt = llvm::find(Pred->Successors, Node);
        std::rotate(Pred->Successors.begin(), SelfIt, SelfIt + 1);
        PredClasses.push_back(-1);
        continue;
      }

      bool AllEdgesToNode = appendOpenSet(Pred, Node, TmpSet);

      int PredClass = -1;
      for (WaveNode *reachableNode : TmpSet) {
        auto It = llvm::find(RerouteCandidates, reachableNode);
        int NodeClass;
        if (It != RerouteCandidates.end()) {
          NodeClass = std::distance(RerouteCandidates.begin(), It);
        } else {
          NodeClass = RerouteCandidates.size();
          RerouteCandidates.push_back(reachableNode);
          RerouteCandidateClasses.grow(RerouteCandidates.size());
        }

        if (PredClass == -1) {
          PredClass = NodeClass;
        } else {
          RerouteCandidateClasses.join(PredClass, NodeClass);
        }
      }

      TmpSet.clear();

      PredClasses.push_back(PredClass);

      if (!AllEdgesToNode) {
        // This predecessor reaches some "open" edge that bypasses the current
        // node and would contradict the reconverging property.
        //
        // The candidate nodes reachable from that predecessor must be rerouted,
        // as well as (transitively) all candidate nodes reachable from any
        // predecessor that can reach those candidate nodes.
        if (RerouteClass == -1) {
          RerouteClass = PredClass;
        } else {
          RerouteCandidateClasses.join(RerouteClass, PredClass);
        }
      }
    }
    assert(PredClasses.size() == Node->Predecessors.size());

    WaveNode *FlowNode = nullptr;
    if (RerouteClass != -1) {
      NextNodes.push_back(
          std::make_unique<WaveNode>(Node->Cycle, ++NumFlowNodes));
      FlowNode = NextNodes.back().get();
      FlowNode->OrderIndex = Node->OrderIndex;
      FlowNode->IsDivergent = true;
      FlowNode->IsSecondary = true;

      unsigned RerouteLeader = RerouteCandidateClasses.findLeader(RerouteClass);
      for (unsigned Idx = 0; Idx != RerouteCandidates.size(); ++Idx) {
        if (RerouteCandidateClasses.findLeader(Idx) == RerouteLeader)
          RerouteNodes.push_back(RerouteCandidates[Idx]);
      }
      for (unsigned Idx = 0; Idx != Node->Predecessors.size(); ++Idx) {
        if (PredClasses[Idx] == -1)
          continue;
        if (RerouteCandidateClasses.findLeader(PredClasses[Idx]) ==
            RerouteLeader)
          RerouteRoots.push_back(Node->Predecessors[Idx]);
      }

      rerouteEdgesBeyond(RerouteNodes, Node, FlowNode);

      // The current node is going to be the flow node's primary successor,
      // so rotate it to the front.
      auto SelfIt = llvm::find(FlowNode->Successors, Node);
      std::rotate(FlowNode->Successors.begin(), SelfIt, SelfIt + 1);

      // Compile-time optimization: record flow node as latest post-dominator
      // of all original predecessors for which we did rerouting.
      for (WaveNode *originalPredecessor : RerouteRoots)
        originalPredecessor->LatestPostDom = FlowNode;

      RerouteNodes.clear();
      RerouteRoots.clear();
    }

    RerouteCandidates.clear();
    RerouteCandidateClasses.clear();
    PredClasses.clear();

    for (WaveNode *Pred : Node->Predecessors) {
      if (Pred == FlowNode || !Pred->IsDivergent)
        continue;

      // TODO: handle the case where successors < 2
      // i.e. the same successor was listed multiple times and this is actually
      // a uniform unconditional branch.
      assert(Pred->Successors.size() == 2);

      WaveNode *Other;
      if (Node == Pred->Successors[0])
        Other = Pred->Successors[1];
      else
        Other = Pred->Successors[0];

      assert(Other->OrderIndex != Node->OrderIndex);
      if (Other->OrderIndex < Node->OrderIndex) {
        Node->IsSecondary = true;

        // Compile-time optimization: record this node as latest post-dominator
        // when possible.
        Pred->LatestPostDom = Node;
      }
    }

    NextNodes.push_back(std::move(NodePtr));

  }
  Nodes = std::move(NextNodes);
  NextNodes.clear();

  cleanupSimpleFlowNodes();
}

/// Short-circuit and remove flow nodes with a single wave successor.
void ReconvergeCFGHelper::cleanupSimpleFlowNodes() {
  bool Changed;

  do {
    Changed = false;

    for (auto &NodePtr : Nodes) {
      WaveNode *Node = NodePtr.get();
      if (!Node->FlowNum || Node->Successors.size() != 1) {
        NextNodes.push_back(std::move(NodePtr));
        continue;
      }

      WaveNode *Succ = Node->Successors[0];
      auto PredIt = llvm::find(Succ->Predecessors, Node);
      assert(PredIt != Succ->Predecessors.end());

      *PredIt = Succ->Predecessors.back();
      Succ->Predecessors.pop_back();
      assert(!is_contained(Succ->Predecessors, Node));

      // if flow node was a secondary target, copy the flag
      if (Node->IsSecondary)
        Succ->IsSecondary = true;

      for (WaveNode *Pred : Node->Predecessors) {
        if (!is_contained(Succ->Predecessors, Pred))
          Succ->Predecessors.push_back(Pred);

        // update LatestPostDom to avoid dangling pointers
        if (Pred->LatestPostDom == Node)
          Pred->LatestPostDom = Succ;

        bool HaveSucc = is_contained(Pred->Successors, Succ);
        auto SuccIt = llvm::find(Pred->Successors, Node);
        if (HaveSucc) {
          Pred->Successors.erase(SuccIt);
        } else {
          *SuccIt = Succ;
        }
        assert(!is_contained(Pred->Successors, Node));

        for (LaneEdge &LaneSucc : Pred->LaneSuccessors) {
          if (LaneSucc.Wave == Node)
            LaneSucc.Wave = Succ;
        }
      }

      for (LaneEdge &LanePred : Succ->LanePredecessors) {
        if (LanePred.Wave == Node) {
          auto PredIt =
              llvm::find_if(Node->LanePredecessors, [=](const LaneEdge &Pred) {
                return Pred.Lane == LanePred.Lane;
              });
          assert(PredIt != Node->LanePredecessors.end());
          LanePred.Wave = PredIt->Wave;
        }
      }

      Changed = true;
    }

    Nodes = std::move(NextNodes);
    NextNodes.clear();
  } while (Changed);

}

/// Return the given cycle's effective heart. If a cycle has no explicitly
/// specified heart, with use the cycle header as heart. This leads to a more
/// intuitive wave transform on natural loops with multiple back edges.
MachineBasicBlock *ReconvergeCFGHelper::getEffectiveHeart(CycleRef Cycle) {
  if (!Cycle)
    return nullptr;

  MachineBasicBlock *Heart = nullptr; // ConvergenceInfo.getHeartBlock(Cycle);
  if (Heart)
    return Heart;
  return CycleInfo.getHeader(Cycle);
}

/// \brief Insert preparatory flow nodes for entering a cycle.
///
/// This method is called just before a cycle is entered, i.e. just before the
/// cycle's header is moved to \ref NextNodes.
///
/// The method unconditionally creates dedicated pre-entry nodes (i.e.,
/// pre-headers, but for all entry nodes in the case of irreducible cycles).
///
/// This ensures that any flow nodes that are required by the entry node
/// don't confound cycle and non-cycle control. Example:
///
///        |     |
///        |     v
///        |     A---->\
///        |    /      |
///        |   /       B
///        v  /        |
///      ^-H  |        |
///      |  \ |        |
///      |   \v        |
///      ^---<C        |
///           |        |
///          ...      ...
///
/// If A has a divergent branch and the main wave transform proceeds with
/// a top-down ordering, it proceeds to reroute the edges from A (incoming
/// to the cycle) and B (unrelated to the cycle) through a single flow
/// block, which unnecessarily causes the edge from B to pass through the
/// cycle. A pre-entry node for C that is processed before the cycle header
/// avoids this.
///
/// Dedicated, per-entry nodes are established to avoid triggering unneded
/// reroutes.
///
/// Pre-entry nodes are created unconditionally to guard against a situation
/// where the core transform creates a flow node that becomes a new entering
/// node with successors outside the cycle. We rely on a later cleanup to
/// remove unnecessary flow nodes in the end.
//
// TODO: Deal with nodes that are reachable without going through the cycle's
//       heart and that have two back edges. Keep various possible heart
//       structures in mind.
void ReconvergeCFGHelper::prepareNodesEnterCycle(WaveNode *headerNode) {
  CycleRef Cycle = headerNode->Cycle;
  WaveNodeVec Entering;

  assert(Cycle);
  for (unsigned Index = headerNode->OrderIndex;
       Index < Nodes.size() &&
       cycleContains(CycleInfo, Cycle, Nodes[Index]->Cycle);
       ++Index) {
    // Check whether this is an entry block, and collect out-of-cycle
    // predecessors.
    WaveNode *Entry = Nodes[Index].get();
    for (WaveNode *Pred : Entry->Predecessors) {
      if (!cycleContains(CycleInfo, Cycle, Pred->Cycle))
        Entering.push_back(Pred);
    }

    if (!Entering.empty()) {
      NextNodes.push_back(std::make_unique<WaveNode>(
          CycleInfo.getParentCycle(Cycle), ++NumFlowNodes));
      WaveNode *FlowNode = NextNodes.back().get();
      FlowNode->OrderIndex = headerNode->OrderIndex;
      reroute(Entering, Entry, FlowNode);
      Entering.clear();
    }
  }
}

/// \brief Insert preparatory flow nodes for latches and cycle exits.
///
/// This method is called just after a cycle is left, i.e. just after the node
/// corresponding to the last block in the cycle is moved to \ref NextNodes.
///
/// If the cycle is the outer-most cycle for its heart, we reroute all backward
/// edges that cross the cycle's heart in the order (including backward edges of
/// a natural loop with heart at the header) through new flow nodes, with a
/// dedicated flow node per backwards target.
///
/// The purpose of these flow nodes is to ensure reconvergence before backwards
/// edges to satisfy the convergence rules of cycle hearts. Example (natural
/// loop with header as heart having a self-loop):
///
///     |           |
///  /-[A           A<---\
///  |  |     =>    |\   |
///  |  |           | \  |
///  ^-<B           B->X-^
///     |           |
///
/// If A has a divergent terminator, control flow will reconverge at X before
/// looping back to A. If A has no divergent terminator, the flow block is not
/// strictly needed. We rely on a post-reconverging cleanup to remove it either
/// way.
///
/// Additionally, these flow nodes ensure correct handling of the most common
/// case of nodes with multiple back edges. Example (A and B are hearts for the
/// cycles they head):
///
///      |                   |
///      A<---\              A<---\
///      |    |              |    |
///      |    |      =>      |    |
///      B<-\ |              B<-\ |
///     /|  | |             /|  | |
///    / |  | |            / |  | |
///   |  C--^ |           |  C--^ |
///   |   \   |           |  |    |
///   |    \--^           |  X----^
///   |                   |
///  ...                 ...
///
/// Flow block X is inserted when exiting the cycle headed by B.
///
/// If C has a divergent terminator, the core transform will reroute the
/// exiting edge from B through a new flow block when handling X to ensure
/// the reconverging condition at C.
///
/// Note: Nodes in the pre-heart region with multiple back edges need to be
///       handled separately!
///
void ReconvergeCFGHelper::prepareNodesExitCycle(CycleRef Cycle,
                                                WaveNode *nextNode) {
  WaveNodeVec FromNodes;
  WaveNodeVec ToNodes;

  assert(Cycle);
  MachineBasicBlock *Heart = getEffectiveHeart(Cycle);
  if (Heart && Heart != getEffectiveHeart(CycleInfo.getParentCycle(Cycle))) {
    WaveNode *HeartNode = NodeForBlock.lookup(Heart);

    for (unsigned nextIndex = NextNodes.size() - 1;; nextIndex--) {
      WaveNode *Node = NextNodes[nextIndex].get();
      assert(cycleContains(CycleInfo, Cycle, Node->Cycle));

      bool isFromNode = false;
      for (WaveNode *Succ : Node->Successors) {
        if (Succ->OrderIndex <= HeartNode->OrderIndex) {
          isFromNode = true;
          if (!is_contained(ToNodes, Succ))
            ToNodes.push_back(Succ);
        }
      }

      if (isFromNode)
        FromNodes.push_back(Node);

      if (Node->Block == Heart)
        break;
    }

    // The sort should not be necessary for correctness, but it should help
    // generate a slightly cleaner wave CFG when there are multiple "to" nodes.
    llvm::sort(ToNodes, [](WaveNode *lhs, WaveNode *rhs) -> bool {
      return lhs->OrderIndex > rhs->OrderIndex;
    });

    for (WaveNode *ToNode : ToNodes) {
      CycleRef toCycle;
      if (cycleContains(CycleInfo, Cycle, ToNode->Cycle))
        toCycle = Cycle;
      else
        toCycle = CycleInfo.getParentCycle(Cycle);

      NextNodes.push_back(std::make_unique<WaveNode>(toCycle, ++NumFlowNodes));
      WaveNode *FlowNode = NextNodes.back().get();
      FlowNode->OrderIndex = nextNode->OrderIndex;
      reroute(FromNodes, ToNode, FlowNode);
    }
  }
}

/// Compute the nodes that are reachable from \p from without going past
/// \p bound in the current node ordering, _and_ that have outgoing edges
/// to \p bound or later nodes ("open" edges).
///
/// Those nodes are appended to \p OpenSet.
///
/// Note: This method relies on WaveNode::LatestPostDom tracking to avoid
/// redundant scanning.
///
/// \return true if all found open edges go to \p bound
bool ReconvergeCFGHelper::appendOpenSet(WaveNode *from, WaveNode *bound,
                                        SmallVectorImpl<WaveNode *> &OpenSet) {
  while (from != from->LatestPostDom)
    from = from->LatestPostDom;
  assert(from->OrderIndex < bound->OrderIndex);

  OpenSetScan.Worklist.push_back(from);

  bool AllToBound = true;
  do {
    WaveNode *Node = OpenSetScan.Worklist.pop_back_val();
    if (Node != Node->LatestPostDom) {
      // Compress post-dom links on the fly
      while (Node->LatestPostDom != Node->LatestPostDom->LatestPostDom)
        Node->LatestPostDom = Node->LatestPostDom->LatestPostDom;
      Node = Node->LatestPostDom;
    }
    assert(Node->OrderIndex < bound->OrderIndex);

    if (!OpenSetScan.Found.insert(Node).second)
      continue;

    bool IsOpen = false;

    for (WaveNode *Succ : Node->Successors) {
      assert(Succ->OrderIndex != bound->OrderIndex || Succ == bound);
      if (Succ->OrderIndex >= bound->OrderIndex) {
        IsOpen = true;
        if (Succ != bound)
          AllToBound = false;
      } else {
        OpenSetScan.Worklist.push_back(Succ);
      }
    }

    if (IsOpen)
      OpenSet.push_back(Node);
  } while (!OpenSetScan.Worklist.empty());

  OpenSetScan.Found.clear();

  return AllToBound;
}

/// Reroute all edges going from any node in \p FromList to the \p ToNode
/// through a new flow node, and return that new node.
///
/// The new node will be appended to the \ref Nodes list.
WaveNode *ReconvergeCFGHelper::rerouteViaNewNode(ArrayRef<WaveNode *> FromList,
                                                 WaveNode *ToNode) {
  Nodes.push_back(std::make_unique<WaveNode>(ToNode->Cycle, ++NumFlowNodes));
  WaveNode *FlowNode = Nodes.back().get();
  reroute(FromList, ToNode, FlowNode);
  return FlowNode;
}

/// Reroute all edges going from any node in \p from to the \p to node via
/// the \p via node.
void ReconvergeCFGHelper::reroute(ArrayRef<WaveNode *> FromList,
                                  WaveNode *ToNode, WaveNode *ViaNode) {
  // In current use, we can assume that ViaNode is not connected to from or to.
  for (WaveNode *FromNode : FromList) {
    auto I = llvm::find(FromNode->Successors, ToNode);
    if (I == FromNode->Successors.end())
      continue;
    FromNode->Successors.erase(I);

    I = llvm::find(ToNode->Predecessors, FromNode);
    assert(I != ToNode->Predecessors.end());
    ToNode->Predecessors.erase(I);

    assert(!is_contained(FromNode->Successors, ViaNode));
    assert(!is_contained(ViaNode->Predecessors, FromNode));
    FromNode->Successors.push_back(ViaNode);
    ViaNode->Predecessors.push_back(FromNode);

    rerouteLane(FromNode, ToNode, ViaNode);
  }

  assert(!is_contained(ViaNode->Successors, ToNode));
  assert(!is_contained(ToNode->Predecessors, ViaNode));
  ViaNode->Successors.push_back(ToNode);
  ToNode->Predecessors.push_back(ViaNode);

  verifyNodes();
}

/// Collect all outgoing edges from nodes in \p FromList to \p ToBeyond or
/// later in the order, and reroute them via \p ViaNode.
void ReconvergeCFGHelper::rerouteEdgesBeyond(ArrayRef<WaveNode *> FromList,
                                             WaveNode *ToBeyond,
                                             WaveNode *ViaNode) {
  // In current use, we can assume that ViaNode is not connect to anything.
  for (WaveNode *FromNode : FromList) {
    assert(!is_contained(FromNode->Successors, ViaNode));

    auto RerouteBegin =
        llvm::partition(FromNode->Successors, [&](WaveNode *Succ) {
          assert(Succ->OrderIndex != ToBeyond->OrderIndex || Succ == ToBeyond);
          return Succ->OrderIndex < ToBeyond->OrderIndex;
        });

    for (WaveNode *Succ :
         llvm::make_range(RerouteBegin, FromNode->Successors.end())) {
      auto I = llvm::find(Succ->Predecessors, FromNode);
      assert(I != Succ->Predecessors.end());
      *I = Succ->Predecessors.back();
      Succ->Predecessors.pop_back();

      if (llvm::find(ViaNode->Successors, Succ) == ViaNode->Successors.end()) {
        ViaNode->Successors.push_back(Succ);

        assert(!is_contained(Succ->Predecessors, ViaNode));
        Succ->Predecessors.push_back(ViaNode);
      }

      rerouteLane(FromNode, Succ, ViaNode);
    }

    FromNode->Successors.erase(RerouteBegin, FromNode->Successors.end());
    FromNode->Successors.push_back(ViaNode);

    assert(!is_contained(ViaNode->Predecessors, FromNode));
    ViaNode->Predecessors.push_back(FromNode);
  }

  verifyNodes();
}

/// Helper for rerouting methods: update the WaveNode::LaneSuccessors and
/// LanePredecessors vectors based on a rerouting.
void ReconvergeCFGHelper::rerouteLane(WaveNode *FromNode, WaveNode *ToNode,
                                      WaveNode *ViaNode) {
  for (LaneEdge &FromLaneSucc : FromNode->LaneSuccessors) {
    if (FromLaneSucc.Wave != ToNode)
      continue;

    FromLaneSucc.Wave = ViaNode;

    bool found = false;
    for (const LaneEdge &ViaLaneSucc : ViaNode->LaneSuccessors) {
      if (ViaLaneSucc.Lane != FromLaneSucc.Lane)
        continue;
      assert(ViaLaneSucc.Wave == ToNode);
      found = true;
      break;
    }
    if (!found)
      ViaNode->LaneSuccessors.emplace_back(FromLaneSucc.Lane, ToNode);
  }

  for (LaneEdge &ToLanePred : ToNode->LanePredecessors) {
    if (ToLanePred.Wave != FromNode)
      continue;

    auto PredIt =
        llvm::find_if(ViaNode->LanePredecessors, [=](const LaneEdge &LanePred) {
          return LanePred.Lane == ToLanePred.Lane;
        });
    if (PredIt != ViaNode->LanePredecessors.end()) {
      assert(PredIt->Wave == FromNode);
    } else {
      ViaNode->LanePredecessors.emplace_back(ToLanePred.Lane, FromNode);
    }
    ToLanePred.Wave = ViaNode;
  }
}

/// Print all WaveNodes to the given stream.
void ReconvergeCFGHelper::printNodes(raw_ostream &Out) {
  auto printNodeName = [&](WaveNode *Node) {
    if (Node->Block) {
      Out << printMBBReference(*Node->Block);
    }
    if (Node->FlowNum) {
      if (Node->Block)
        Out << '.';
      Out << "<flow-" << Node->FlowNum << '>';
    }
    if (!Node->Block && !Node->FlowNum) {
      Out << "<unnamed-node>";
    }
  };

  auto printNode = [&](WaveNode *Node) {
    Out << "  ";
    printNodeName(Node);
    Out << " (#" << Node->OrderIndex << ")";

    if (!Node->Successors.empty()) {
      Out << " ->";
      for (WaveNode *Succ : Node->Successors) {
        Out << ' ';
        printNodeName(Succ);

        bool Printed = false;
        for (const LaneEdge &LaneSucc : Node->LaneSuccessors) {
          if (LaneSucc.Wave != Succ)
            continue;

          if (!Printed) {
            Out << '(';
            Printed = true;
          } else {
            Out << ',';
          }

          if (LaneSucc.Lane == Succ)
            Out << '*';
          else
            printNodeName(LaneSucc.Lane);
        }
        if (Printed)
          Out << ')';
      }
    }

    if (Node->LatestPostDom != Node) {
      Out << " [LatestPostDom: ";
      printNodeName(Node->LatestPostDom);
      Out << ']';
    }

    if (Node->IsDivergent)
      Out << " [divergent]";
    if (Node->IsSecondary)
      Out << " [secondary]";

    Out << '\n';
  };

  for (const auto &NodePtr : NextNodes)
    printNode(NodePtr.get());
  for (const auto &NodePtr : Nodes) {
    if (NodePtr)
      printNode(NodePtr.get());
  }
  Out << '\n';
}

/// Dump all WaveNodes to debug out.
LLVM_ATTRIBUTE_UNUSED
void ReconvergeCFGHelper::dumpNodes() {
  printNodes(dbgs());

  verifyNodes();
}

/// Verify some basic invariants on WaveNodes.
void ReconvergeCFGHelper::verifyNodes() {
  WaveNodeSet SeenNodes;

  auto verifyNode = [&](WaveNode *Node) {
    LLVM_ATTRIBUTE_UNUSED
    bool Inserted = SeenNodes.insert(Node).second;
    assert(Inserted);
  };

  for (const auto &NodePtr : NextNodes)
    verifyNode(NodePtr.get());
  for (const auto &NodePtr : Nodes) {
    if (NodePtr)
      verifyNode(NodePtr.get());
  }

  WaveNodeSet LanePreds;
  WaveNodeSet LaneSuccs;

  // FIXME: these only contain assertions so are "unused variables" in release
  // build
  for (WaveNode *Node : SeenNodes) {
    for (WaveNode *Pred : Node->Predecessors) {
      assert(SeenNodes.count(Pred));
      assert(is_contained(Pred->Successors, Node));
    }
    for (WaveNode *Succ : Node->Successors) {
      assert(SeenNodes.count(Succ));
      assert(is_contained(Succ->Predecessors, Node));

      assert(llvm::any_of(Node->LaneSuccessors, [&](const auto &LaneSucc) {
        return LaneSucc.Wave == Succ;
      }));
    }

    for (const LaneEdge &LanePred : Node->LanePredecessors) {
      assert(SeenNodes.count(LanePred.Lane));
      assert(is_contained(Node->Predecessors, LanePred.Wave));
      bool Inserted = LanePreds.insert(LanePred.Lane).second;
      assert(Inserted);

      if (LanePred.Lane != LanePred.Wave) {
        assert(LanePred.Wave->FlowNum != 0);
        assert(!is_contained(Node->Predecessors, LanePred.Lane));
        assert(llvm::any_of(
            LanePred.Wave->LanePredecessors,
            [&](const auto &next) { return next.Lane == LanePred.Lane; }));
      }
    }
    LanePreds.clear();

    for (const LaneEdge &LaneSucc : Node->LaneSuccessors) {
      assert(SeenNodes.count(LaneSucc.Lane));
      assert(is_contained(Node->Successors, LaneSucc.Wave));
      bool Inserted = LaneSuccs.insert(LaneSucc.Lane).second;
      assert(Inserted);

      if (LaneSucc.Lane != LaneSucc.Wave) {
        assert(LaneSucc.Wave->FlowNum != 0);
        assert(!is_contained(Node->Successors, LaneSucc.Lane));
        assert(
            llvm::any_of(LaneSucc.Wave->LaneSuccessors, [&](const auto &next) {
              return next.Lane == LaneSucc.Lane;
            }));
      }
    }
    LaneSuccs.clear();
  }
}

namespace {

/// Helper class for rewriting control-flow instruction after translation into
/// a wave CFG.
class ControlFlowRewriter {
private:
  /// For a given original target node, record information about where lanes
  /// for that target can come from.
  struct LaneOriginInfo {
    /// Node (original or flow) from which lanes can originate.
    WaveNode *Node;

    /// Condition under which lanes originate from that node (can be null,
    /// in which case EXEC / all active lanes should be used).
    Register CondReg;

    /// Whether the condition should be inverted.
    bool InvertCondition = false;

    /// Whether the condition register use was marked as undef.
    bool CondIsUndef = false;

    /// Opcode of implicit branch instruction
    unsigned ImplicitBranchOpc = 0;

    explicit LaneOriginInfo(WaveNode *Node, Register CondReg = {},
                            bool InvertCondition = false,
                            bool CondIsUndef = false,
                            unsigned ImplicitBranchOpc = 0)
        : Node(Node), CondReg(CondReg), InvertCondition(InvertCondition),
          CondIsUndef(CondIsUndef), ImplicitBranchOpc(ImplicitBranchOpc) {}
  };

  struct CFGNodeInfo {
    WaveNode *Node;

    bool OrigExit = false;

    /// Branch condition, if the block originally had a conditional branch.
    Register OrigCondition;

    /// Whether the original condition register was marked as undef.
    bool OrigConditionUndef = false;

    /// Branch target if \ref condition is true.
    WaveNode *OrigSuccCond = nullptr;

    /// Final branch target, i.e. if there was no conditional branch or if
    /// \ref condition is false.
    WaveNode *OrigSuccFinal = nullptr;

    /// Information about nodes from which lanes targeting this node can
    /// originate.
    SmallVector<LaneOriginInfo, 4> origins;

    /// (origin, divergent) pairs of origin nodes that have a branch towards
    /// this node with the property that immediately after the corresponding
    /// branch, all active lanes target this node.
    SmallVector<PointerIntPair<WaveNode *, 1, bool>, 4> OriginBranch;

    Register PrimarySuccessorExec;

    // Opcode for branches with implicit or opaque conditions:
    // S_CBRANCH_EXECZ/NZ S_CBRANCH_VCCZ/NZ S_CBRANCH_SCC0/1
    //   -- all active threads branch uniformly.
    // INLINEASM_BR -- opaque callbr; exec assumed invariant.
    unsigned ImplicitBranchOpc = 0;

    /// INLINEASM_BR instruction, if the block contains one.
    MachineInstr *InlineAsmBrMI = nullptr;

    explicit CFGNodeInfo(WaveNode *Node) : Node(Node) {}
  };

  /// Information required to synthesize divergent terminators with a common
  /// primary successor.
  struct DivergentTargetInfo {
    /// Nodes containing divergent terminators whose primary successor targets
    /// the node in question.
    SmallVector<WaveNode *, 2> BranchNodes;

    /// Flow nodes that are targeted by one or more of the terminators in
    /// \ref BranchNodes, but are themselves only intermediate steps to the
    /// targets in question.
    WaveNodeSet FlowNodes;
  };

  MachineFunction &Function;
  ReconvergeCFGHelper &ReconvergeCfg;
  AMDGPULaneMaskUtils LMU;
  MachineRegisterInfo &MRI;
  const SIInstrInfo &TII;

  DenseMap<WaveNode *, CFGNodeInfo> NodeInfo;
  std::vector<WaveNode *> NodeOrder;
  AccRegSet AccumulatorRegs;

public:
  ControlFlowRewriter(MachineFunction &function,
                      ReconvergeCFGHelper &ReconvergeCfg)
      : Function(function), ReconvergeCfg(ReconvergeCfg), LMU(function),
        MRI(function.getRegInfo()),
        TII(*function.getSubtarget<GCNSubtarget>().getInstrInfo()) {}

  void prepareWaveCfg();
  void rewrite();

  const AccRegSet &getAccumulatorRegs() const { return AccumulatorRegs; }
};

} // anonymous namespace

/// Collect information about original terminator instructions and prepare
/// the wave-level CFG without changing the MIR representation yet.
void ControlFlowRewriter::prepareWaveCfg() {
  // Pre-initialize the block-info map with all blocks, so that we can rely
  // on stable references for the next step.
  for (WaveNode *Node : ReconvergeCfg.nodes()) {
    if (NodeInfo.try_emplace(Node, Node).second)
      NodeOrder.push_back(Node);
  }

  // Step 1: Analyze original successors and branch conditions and record them
  // as well as related info that we will need to generate divergent branches.
  //
  // uniformCandidateEdges maps (ToNode, viaFlowNode) -> FromNodes for edges
  // _fro a node with uniform conditional terminator _to_ an original
  // predecessor _via_ a flow node with multiple successors.
  MapVector<std::pair<WaveNode *, WaveNode *>, SmallVector<WaveNode *, 2>>
      UniformSplitEdges;

  for (WaveNode *Node : ReconvergeCfg.nodes()) {
    CFGNodeInfo &Info = NodeInfo.find(Node)->second;

    if (Node->IsDivergent && Node->Successors.size() >= 2) {
      assert(Node->Successors.size() == 2);
      WaveNode *primaryWave = Node->Successors[0];
      WaveNode *primaryLane = nullptr;
      for (const LaneEdge &LaneSucc : Node->LaneSuccessors) {
        if (LaneSucc.Wave == primaryWave) {
          assert(!primaryLane);
          primaryLane = LaneSucc.Lane;
#ifdef NDEBUG
          // early-out when assertions are disabled: we don't check for
          // uniqueness in that case
          break;
#endif
        }
      }
      assert(primaryLane);

      NodeInfo.find(primaryLane)->second.OriginBranch.emplace_back(Node, true);
    }

    if (!Node->Block)
      continue;

    // Identify the indirect MBB target from the operands of INLINEASM_BR
    // (OrigSuccCond). The block participates in lane mask building as a uniform
    // conditional, conservatively contributing all active lanes.
    if (Node->Block->mayHaveInlineAsmBr()) {
      auto InlineAsmBrIt =
          llvm::find_if(*Node->Block, [](const MachineInstr &MI) {
            return MI.getOpcode() == TargetOpcode::INLINEASM_BR;
          });

      if (InlineAsmBrIt != Node->Block->end()) {
        Info.InlineAsmBrMI = &*InlineAsmBrIt;
        for (const MachineOperand &MO : InlineAsmBrIt->operands()) {
          if (!MO.isMBB())
            continue;
          assert(!Info.OrigSuccCond &&
                 "Multiple INLINEASM_BR indirect targets not yet supported");
          Info.ImplicitBranchOpc = TargetOpcode::INLINEASM_BR;
          Info.OrigSuccCond = ReconvergeCfg.nodeForBlock(MO.getMBB());
        }
      }
    }

    bool ZVariant = false;

    // Analyze original terminators.
    for (MachineInstr &Terminator : Node->Block->terminators()) {
      unsigned Opcode = Terminator.getOpcode();

      assert(!Info.OrigSuccFinal);
      if (Opcode == AMDGPU::SI_BRCOND || Opcode == AMDGPU::SI_BRCOND_Z) {
        assert(!Info.OrigCondition);
        assert(!Info.OrigSuccCond);
        ZVariant = Opcode == AMDGPU::SI_BRCOND_Z;
        Info.OrigCondition = Terminator.getOperand(1).getReg();
        Info.OrigConditionUndef = Terminator.getOperand(1).isUndef();
        Info.OrigSuccCond =
            ReconvergeCfg.nodeForBlock(Terminator.getOperand(0).getMBB());
      } else if (Opcode == AMDGPU::S_CBRANCH_EXECZ ||
                 Opcode == AMDGPU::S_CBRANCH_EXECNZ ||
                 Opcode == AMDGPU::S_CBRANCH_VCCZ ||
                 Opcode == AMDGPU::S_CBRANCH_VCCNZ ||
                 Opcode == AMDGPU::S_CBRANCH_SCC0 ||
                 Opcode == AMDGPU::S_CBRANCH_SCC1) {
        assert(!Info.OrigCondition);
        assert(!Info.ImplicitBranchOpc);
        assert(!Info.OrigSuccCond);
        Info.ImplicitBranchOpc = Opcode;
        Info.OrigSuccCond =
            ReconvergeCfg.nodeForBlock(Terminator.getOperand(0).getMBB());
      } else if (Opcode == AMDGPU::S_BRANCH) {
        Info.OrigSuccFinal =
            ReconvergeCfg.nodeForBlock(Terminator.getOperand(0).getMBB());
      } else if (Terminator.isReturn()) {
        assert(!Info.OrigCondition);
        Info.OrigExit = true;
      }
    }

    if (!Info.OrigExit && Node->Successors.empty()) {
      // May be an unreachable block. Treat them like exit blocks.
      Info.OrigExit = true;
    }

    if (!Info.OrigSuccFinal && !Info.OrigExit) {
      // Fall-through in the original code.
      auto BlockIt = Node->Block->getIterator();
      ++BlockIt;
      assert(BlockIt != Function.end());
      assert(is_contained(Node->Block->successors(), &*BlockIt));
      Info.OrigSuccFinal = ReconvergeCfg.nodeForBlock(&*BlockIt);
    }

    assert(Info.OrigExit || Node->FlowNum != 0 || Info.OrigSuccFinal);
    assert(!Info.OrigExit || !Info.OrigSuccFinal);
    assert(!Info.OrigSuccCond || Info.OrigSuccFinal);
    assert(Info.OrigExit == Node->Successors.empty() &&
           "TODO: exit unification");
    assert((!Info.ImplicitBranchOpc || !Info.OrigCondition) &&
           "ImplicitBranchOpc and OrigCondition are mutually exclusive");
    assert((!Info.ImplicitBranchOpc || Info.OrigSuccCond) &&
           "Implicit conditional branch requires OrigSuccCond");

    // Record information for reconstructing lane masks.
    if (!Info.OrigSuccCond) {
      if (Info.OrigSuccFinal) {
        NodeInfo.find(Info.OrigSuccFinal)->second.origins.emplace_back(Node);
      }
    } else {
      if (!Node->IsDivergent && Node->Successors.size() >= 2) {
        assert(Node->Successors.size() == 2);

        NodeInfo.find(Info.OrigSuccCond)
            ->second.OriginBranch.emplace_back(Node, false);
        NodeInfo.find(Info.OrigSuccFinal)
            ->second.OriginBranch.emplace_back(Node, false);

        for (const LaneEdge &LaneEdge : Node->LaneSuccessors) {
          assert(LaneEdge.Lane == Info.OrigSuccCond ||
                 LaneEdge.Lane == Info.OrigSuccFinal);

          if (LaneEdge.Lane == LaneEdge.Wave) {
            // If we directly branch to the Lane target, this edge will never
            // contribute to a divergent branch.
            continue;
          }

          // If the original edge was redirected through flow nodes, we are
          // likely going through a divergent branch at some point.
          if (LaneEdge.Wave->LaneSuccessors.size() > 1) {
            UniformSplitEdges[std::make_pair(LaneEdge.Lane, LaneEdge.Wave)]
                .emplace_back(Node);
          } else {
            CFGNodeInfo &succInfo = NodeInfo.find(LaneEdge.Lane)->second;
            if (!llvm::any_of(succInfo.origins,
                              [&](const LaneOriginInfo &origin) {
                                return origin.Node == LaneEdge.Wave;
                              }))
              succInfo.origins.emplace_back(
                  LaneEdge.Wave, /*CondReg=*/Register(),
                  /*InvertCondition=*/false, /*CondIsUndef=*/false,
                  Info.ImplicitBranchOpc);
          }
        }
      } else {
        NodeInfo.find(Info.OrigSuccCond)
            ->second.origins.emplace_back(Node, Info.OrigCondition, ZVariant,
                                          Info.OrigConditionUndef,
                                          Info.ImplicitBranchOpc);
        NodeInfo.find(Info.OrigSuccFinal)
            ->second.origins.emplace_back(Node, Info.OrigCondition, !ZVariant,
                                          Info.OrigConditionUndef,
                                          Info.ImplicitBranchOpc);
      }
    }
  }

  // Step 2: Split certain critical edges after uniform branches.
  //
  // A uniform conditional branch can end up leading into a flow node with
  // multiple (lane) successors, which means the original target of the
  // conditional branch is ultimately reached via a divergent branch for which
  // we need to establish a corresponding lane mask. In this example, A has a
  // uniform branch to C that got rerouted through flow nodes X and Y for some
  // reason (e.g. part of loop control flow handling):
  //
  //     |
  //     A
  //    / \  ...
  //   ... \ /
  //        X
  //        |\
  //        | B
  //        |/
  //        Y
  //        |\
  //        | \
  //       ... C
  //           |
  //
  // In Y, we need a lane mask for the branch to C that takes into account
  // lanes from A as well as lanes from some potential other predecessors.
  //
  // To facilitate the construction of these lane masks, we split the edge from
  // A to X.
  for (const auto &UniformSplit : UniformSplitEdges) {
    WaveNode *FlowNode = ReconvergeCfg.rerouteViaNewNode(
        UniformSplit.second, UniformSplit.first.second);
    if (NodeInfo.try_emplace(FlowNode, FlowNode).second)
      NodeOrder.push_back(FlowNode);
    NodeInfo.find(UniformSplit.first.first)
        ->second.origins.emplace_back(FlowNode);
  }
}

namespace {

struct AccBlockInfo {
  SmallVector<MachineBasicBlock *, 4> UseBlocks;
  SmallVector<MachineBasicBlock *, 4> DefBlocks;
};

MachineBasicBlock *nearestCommonDominator(ArrayRef<MachineBasicBlock *> Blocks,
                                          MachineDominatorTree &MDT) {
  assert(!Blocks.empty() && "accumulator init requires at least one use block");
  MachineBasicBlock *DomBB = Blocks.front();
  for (MachineBasicBlock *BB : Blocks.drop_front())
    DomBB = MDT.findNearestCommonDominator(DomBB, BB);
  return DomBB;
}

} // namespace

/// Replace all original terminator instructions by the terminators for
/// establishing wave-level control flow and insert instructions for EXEC mask
/// manipulation.
void ControlFlowRewriter::rewrite() {
  AMDGPULaneMaskAnalysis LMA(Function);
  const AMDGPU::LaneMaskConstants &LMC = LMU.getLaneMaskConsts();

  Register RegAllOnes;
  auto getAllOnes = [&]() {
    if (!RegAllOnes) {
      RegAllOnes = LMU.createLaneMaskReg();
      BuildMI(Function.front(), Function.front().getFirstTerminator(), {},
              TII.get(LMC.MovOpc), RegAllOnes)
          .addImm(-1);
    }
    return RegAllOnes;
  };
  Register RegZero;
  auto getZero = [&]() {
    if (!RegZero) {
      RegZero = LMU.createLaneMaskReg();
      BuildMI(Function.front(), Function.front().getFirstTerminator(), {},
              TII.get(LMC.MovOpc), RegZero)
          .addImm(0);
    }
    return RegZero;
  };

  // Track blocks that lost their INLINEASM_BR indirect-target status due
  // to retargeting and the set of blocks still referenced by some
  // surviving INLINEASM_BR. After Step 1, any stale target not in the
  // live set has its indirect-target flag cleared.
  SmallPtrSet<MachineBasicBlock *, 4> StaleCallbrTargets;
  SmallPtrSet<MachineBasicBlock *, 4> LiveCallbrTargets;

  // Step 1: Remove old terminators and insert new ones for uniform branches.
  for (WaveNode *Node : NodeOrder) {
    CFGNodeInfo &Info = NodeInfo.find(Node)->second;
    MachineBasicBlock::iterator MBBINodeEnd = Node->Block->end();

    if (!Info.OrigExit) {
      // Remove original terminators, preserving artificial terminators
      // (EXEC management ops), hardware-managed branches (e.g.
      // S_SUBVECTOR_LOOP) that pass through to assembly unchanged, and
      // kill terminators that will be lowered later by si-wqm.
      while (!Node->Block->empty() && Node->Block->back().isTerminator() &&
             !isArtificialTerminator(Node->Block->back()) &&
             !isHardwareManagedBranch(Node->Block->back()) &&
             !TII.isKillTerminator(Node->Block->back().getOpcode()))
        Node->Block->back().eraseFromParent();
    }

    if (Node->Successors.size() == 0)
      continue;

    assert(!Info.OrigExit);

    // INLINEASM_BR is not a terminator and was preserved during terminator
    // removal. After reconvergence the original indirect target may no
    // longer be a direct MBB successor (a flow node could've been inserted).
    // Retarget the MBB operand to the appropriate wave-successor and
    // transfer the indirect-target attributes onto the new target.
    if (Info.InlineAsmBrMI) {
      for (MachineOperand &MO : Info.InlineAsmBrMI->operands()) {
        if (!MO.isMBB())
          continue;
        if (Node->Block->isSuccessor(MO.getMBB())) {
          LiveCallbrTargets.insert(MO.getMBB());
          continue;
        }
        MachineBasicBlock *NewTarget;
        if (Node->Successors.size() == 1) {
          NewTarget = Node->Successors[0]->Block;
        } else {
          auto LaneSucc =
              llvm::find_if(Node->LaneSuccessors, [=](const auto &succ) {
                return succ.Lane == Info.OrigSuccCond;
              });
          assert(LaneSucc != Node->LaneSuccessors.end());
          NewTarget = LaneSucc->Wave->Block;
        }
        StaleCallbrTargets.insert(MO.getMBB());
        LiveCallbrTargets.insert(NewTarget);
        NewTarget->setIsInlineAsmBrIndirectTarget();
        NewTarget->setLabelMustBeEmitted();
        MO.setMBB(NewTarget);
      }
    }

    if (Node->Successors.size() == 1) {
      if (Node->Block->empty() || !isArtificialTerminator(Node->Block->back()))
        BuildMI(*Node->Block, MBBINodeEnd, {}, TII.get(AMDGPU::S_BRANCH))
            .addMBB(Node->Successors[0]->Block);
      continue;
    }

    assert(Node->Successors.size() == 2);

    // Hardware-managed branches handle their own EXEC and control flow
    // atomically — no new branches needed.
    if (!Node->Block->empty() && Node->Block->back().isBranch() &&
        isHardwareManagedBranch(Node->Block->back()))
      continue;

    if (!Node->IsDivergent) {
      // Uniform block with two successors: we must have had two original
      // successors, and one of the current successors leads to the original
      // conditional successor.

      auto LaneSucc =
          llvm::find_if(Node->LaneSuccessors, [=](const auto &succ) {
            return succ.Lane == Info.OrigSuccCond;
          });
      assert(LaneSucc != Node->LaneSuccessors.end());

      // The _other_ successor may be a flow block instead of an original
      // successor.
      WaveNode *Other;
      if (Node->Successors[0] == LaneSucc->Wave)
        Other = Node->Successors[1];
      else
        Other = Node->Successors[0];

      // Re-emit the implicit conditional branch for the lane successor.
      // INLINEASM_BR is skipped: it was preserved during terminator
      // removal and already retargeted.
      if (Info.ImplicitBranchOpc != TargetOpcode::INLINEASM_BR) {
        assert(Info.ImplicitBranchOpc && "Implicit Branch Opcode not set");
        MachineInstr *CondBrMI = BuildMI(*Node->Block, MBBINodeEnd, {},
                                         TII.get(Info.ImplicitBranchOpc))
                                     .addMBB(LaneSucc->Wave->Block);
        TII.fixImplicitOperands(*CondBrMI);
      }

      BuildMI(*Node->Block, MBBINodeEnd, {}, TII.get(AMDGPU::S_BRANCH))
          .addMBB(Other->Block);
    }
  }

  // Clear the indirect-target flag on blocks that were retargeted away,
  // unless another surviving INLINEASM_BR still references them.
  for (MachineBasicBlock *Stale : StaleCallbrTargets)
    if (!LiveCallbrTargets.contains(Stale))
      Stale->setIsInlineAsmBrIndirectTarget(false);

  // Step 2: Insert lane masks and new terminators for divergent nodes.
  //
  // RegMap maps (block, register) -> (masked, inverted).
  DenseMap<std::pair<MachineBasicBlock *, Register>,
           std::pair<Register, Register>>
      RegMap;
  AMDGPULaneMaskUpdater Updater(Function);
  Updater.setLaneMaskAnalysis(&LMA);

  auto finalizeAccInit = [&](Register Acc, const AccBlockInfo &Info) {
    MachineBasicBlock *DomBB =
        nearestCommonDominator(Info.UseBlocks, ReconvergeCfg.getDomTree());
    DomBB = ReconvergeCfg.hoistBeforeCycle(DomBB);
    Updater.setAccumulatorInitBlock(Acc, DomBB);
  };

  for (WaveNode *LaneTarget : NodeOrder) {
    CFGNodeInfo &LaneTargetInfo = NodeInfo.find(LaneTarget)->second;

    if (!llvm::any_of(
            LaneTargetInfo.OriginBranch,
            [](const auto &OriginBranch) { return OriginBranch.getInt(); })) {
      // No divergent branches towards this node, nothing to be done.
      continue;
    }

    // When there is only a single lane mask origin, the condition register
    // can be used directly as the primary successor EXEC value, bypassing
    // the accumulator machinery. This optimization requires three conditions:
    // 1. Exactly one origin exists for this lane target.
    // 2. The origin is not inside a loop — inside loops, lane masks must be
    //    accumulated across iterations via the accumulator machinery.
    // 3. The origin block dominates all divergent OriginBranch blocks where
    //    PrimarySuccessorExec will be consumed. If any divergent OriginBranch
    //    block is not dominated, a bypass path may exist and the accumulator
    //    is needed to ensure the lane mask is properly initialized on all
    //    paths.
    bool HasSingleDomOrigin =
        LaneTargetInfo.origins.size() == 1 &&
        !LaneTargetInfo.origins[0].Node->Cycle &&
        !llvm::any_of(LaneTargetInfo.OriginBranch,
                      [&](const auto &NodeDivergentPair) {
                        return NodeDivergentPair.getInt() &&
                               !ReconvergeCfg.getDomTree().dominates(
                                   LaneTargetInfo.origins[0].Node->Block,
                                   NodeDivergentPair.getPointer()->Block);
                      });
    Register DirectCondReg;
    Register Acc = AMDGPU::NoRegister;
    AccBlockInfo AccInfo;

    // Step 2.1: Add conditions branching to LaneTarget to the Lane mask
    // Updater. Initialize the accumulator only when multiple origins
    // require merging.
    if (!HasSingleDomOrigin) {
      Acc = Updater.init();
      AccInfo.DefBlocks.push_back(LaneTarget->Block);
      Updater.addReset(*LaneTarget->Block, AMDGPULaneMaskUpdater::ResetInMiddle);
      for (const auto &NodeDivergentPair : LaneTargetInfo.OriginBranch) {
        if (!NodeDivergentPair.getInt())
          continue; // not a divergent branch

        MachineBasicBlock *OriginBranchBlock =
            NodeDivergentPair.getPointer()->Block;
        AccInfo.UseBlocks.push_back(OriginBranchBlock);
        AccInfo.DefBlocks.push_back(OriginBranchBlock);
        Updater.addReset(*OriginBranchBlock,
                         AMDGPULaneMaskUpdater::ResetAtEnd);
      }
    }

    for (const LaneOriginInfo &LaneOrigin : LaneTargetInfo.origins) {
      Register CondReg;
      MachineBasicBlock::iterator MBBILaneOriginNodeFirstTerm =
          LaneOrigin.Node->Block->getFirstTerminator();

      if (!LaneOrigin.CondReg) {
        switch (LaneOrigin.ImplicitBranchOpc) {
        case 0: // Unconditional branch
          assert(!LaneOrigin.InvertCondition);
          CondReg = getAllOnes();
          break;
        case TargetOpcode::INLINEASM_BR:
          // Opaque callbr; exec assumed invariant. Conservatively
          // contribute all active lanes regardless of branch direction.
          CondReg = getAllOnes();
          break;
        // Uniform branch with implicit condition (VCC/EXEC/SCC), or
        // unconditional (ImplicitBranchOpc == 0).
        // All active lanes go the same direction, so the lane
        // contribution is either EXEC (all lanes) or 0 (no lanes).
        case AMDGPU::S_CBRANCH_EXECNZ:
          CondReg = LaneOrigin.InvertCondition ? getZero() : getAllOnes();
          break;
        case AMDGPU::S_CBRANCH_EXECZ:
          CondReg = LaneOrigin.InvertCondition ? getAllOnes() : getZero();
          break;
        case AMDGPU::S_CBRANCH_SCC1: {
          CondReg = LMU.createLaneMaskReg();
          auto MIB =
              BuildMI(*LaneOrigin.Node->Block, MBBILaneOriginNodeFirstTerm, {},
                      TII.get(LMC.CSelectOpc), CondReg);
          if (!LaneOrigin.InvertCondition)
            MIB.addReg(LMC.ExecReg).addImm(0);
          else
            MIB.addImm(0).addReg(LMC.ExecReg);
          break;
        }
        case AMDGPU::S_CBRANCH_SCC0: {
          CondReg = LMU.createLaneMaskReg();
          auto MIB =
              BuildMI(*LaneOrigin.Node->Block, MBBILaneOriginNodeFirstTerm, {},
                      TII.get(LMC.CSelectOpc), CondReg);
          if (!LaneOrigin.InvertCondition)
            MIB.addImm(0).addReg(LMC.ExecReg);
          else
            MIB.addReg(LMC.ExecReg).addImm(0);
          break;
        }
        case AMDGPU::S_CBRANCH_VCCNZ:
        case AMDGPU::S_CBRANCH_VCCZ: {
          // S_AND_B* VCC, VCC sets SCC (SCC = VCC != 0)
          BuildMI(*LaneOrigin.Node->Block, MBBILaneOriginNodeFirstTerm, {},
                  TII.get(LMC.AndOpc), LMC.VccReg)
              .addReg(LMC.VccReg)
              .addReg(LMC.VccReg);

          // SCC gets set when VCC!=0, implyng equivalence to VCCNZ branch;
          // so we should flip CSELECT operands for VCCZ branch equivalent.
          bool FlipForVCCZ =
              (LaneOrigin.ImplicitBranchOpc == AMDGPU::S_CBRANCH_VCCZ);

          CondReg = LMU.createLaneMaskReg();
          auto MIB =
              BuildMI(*LaneOrigin.Node->Block, MBBILaneOriginNodeFirstTerm, {},
                      TII.get(LMC.CSelectOpc), CondReg);
          if (LaneOrigin.InvertCondition == FlipForVCCZ)
            MIB.addReg(LMC.ExecReg).addImm(0);
          else
            MIB.addImm(0).addReg(LMC.ExecReg);
          break;
        }
        default:
          llvm_unreachable("unhandled implicit branch opcode");
        }
      } else if (LaneOrigin.CondReg == AMDGPU::SCC) {
        assert(LaneOrigin.Node->Successors.size() == 1);

        // Subtle: We rely here on the fact that:
        //  1. No other instructions have been inserted at the end of the
        //     basic block since step 1, when the terminators were deleted --
        //     otherwise, SCC could have been clobbered.
        //  2. Later steps only insert instructions between the cselect here
        //     and the terminators, where SCC no longer matters.
        //
        // PHI nodes may have been inserted, but those are at the beginning
        // of the block.
        //
        // cond = SCC ? EXEC : 0; (or reverse)
        CondReg = LMU.createLaneMaskReg();
        if (!LaneOrigin.InvertCondition) {
          BuildMI(*LaneOrigin.Node->Block, MBBILaneOriginNodeFirstTerm, {},
                  TII.get(LMC.CSelectOpc), CondReg)
              .addReg(LMC.ExecReg)
              .addImm(0);
        } else {
          BuildMI(*LaneOrigin.Node->Block, MBBILaneOriginNodeFirstTerm, {},
                  TII.get(LMC.CSelectOpc), CondReg)
              .addImm(0)
              .addReg(LMC.ExecReg);
        }
      } else {
        CondReg = LaneOrigin.CondReg;
        if (!LMA.isSubsetOfExec(LaneOrigin.CondReg, *LaneOrigin.Node->Block,
                                MBBILaneOriginNodeFirstTerm)) {
          if (!LaneOrigin.CondIsUndef) {
            Register Prev = CondReg;
            CondReg = LMU.createLaneMaskReg();
            BuildMI(*LaneOrigin.Node->Block, MBBILaneOriginNodeFirstTerm, {},
                    TII.get(LMC.AndOpc), CondReg)
                .addReg(LMC.ExecReg)
                .addReg(Prev);
          }

          RegMap[std::make_pair(LaneOrigin.Node->Block, LaneOrigin.CondReg)]
              .first = CondReg;
        }

        // Skip XOR inversion if the condition register use was undef.
        // Inverting undef (undef ^ -1) is still undef.
        if (LaneOrigin.InvertCondition) {
          Register Prev = CondReg;
          if (!LaneOrigin.CondIsUndef) {
            CondReg = LMU.createLaneMaskReg();
            // Prev is guaranteed to be a subset of EXEC here: either the
            // original CondReg was already a subset, or we masked it with
            // AND(EXEC, CondReg) above. XOR with EXEC flips only within
            // active lanes, so the result is also a subset of EXEC and
            // no further AND with EXEC is needed downstream.
            BuildMI(*LaneOrigin.Node->Block, MBBILaneOriginNodeFirstTerm,
                    {}, TII.get(LMC.XorOpc), CondReg)
                .addReg(Prev)
                .addReg(LMC.ExecReg);
          }

          RegMap[std::make_pair(LaneOrigin.Node->Block, LaneOrigin.CondReg)]
              .second = CondReg;
          RegMap.try_emplace(std::make_pair(LaneOrigin.Node->Block, CondReg),
                             CondReg, Prev);
        }
      }

      if (HasSingleDomOrigin)
        DirectCondReg = CondReg;
      else {
        AccInfo.UseBlocks.push_back(LaneOrigin.Node->Block);
        AccInfo.DefBlocks.push_back(LaneOrigin.Node->Block);
        Updater.addAvailable(*LaneOrigin.Node->Block, CondReg);
      }
    }

    // Step 2.2: Synthesize EXEC updates and branch instructions.
    for (const auto &NodeDivergentPair : LaneTargetInfo.OriginBranch) {
      if (!NodeDivergentPair.getInt())
        continue; // not a divergent branch

      WaveNode *OriginNode = NodeDivergentPair.getPointer();
      CFGNodeInfo &OriginCFGNodeInfo = NodeInfo.find(OriginNode)->second;
      OriginCFGNodeInfo.PrimarySuccessorExec =
          HasSingleDomOrigin ? DirectCondReg
                             : Updater.getMergedMask(*OriginNode->Block);

      MachineBasicBlock::iterator MBBIOriginNodeEnd = OriginNode->Block->end();

      // FIXME: Find a way to avoid adding MovTermOpc, instead add MovOpc. This
      // Term operator being the first terminator, acts as an anchor point for
      // finding the right insertion point in other parts of the Wave Transform.
      // Since accumulator reset instructions may be added after this
      // instruction, this move operation cannot be a terminator.
      BuildMI(*OriginNode->Block, MBBIOriginNodeEnd, {},
              TII.get(LMC.MovTermOpc), LMC.ExecReg)
          .addReg(OriginCFGNodeInfo.PrimarySuccessorExec);
      BuildMI(*OriginNode->Block, MBBIOriginNodeEnd, {},
              TII.get(AMDGPU::SI_WAVE_CF_EDGE));
      BuildMI(*OriginNode->Block, MBBIOriginNodeEnd, {},
              TII.get(AMDGPU::S_CBRANCH_EXECZ))
          .addMBB(OriginNode->Successors[1]->Block);
      BuildMI(*OriginNode->Block, MBBIOriginNodeEnd, {},
              TII.get(AMDGPU::S_BRANCH))
          .addMBB(OriginNode->Successors[0]->Block);
    }

    if (Acc)
      finalizeAccInit(Acc, AccInfo);

  }

  // Step 3: Insert rejoin masks.
  for (WaveNode *Secondary : ReconvergeCfg.nodes()) {
    if (!Secondary->IsSecondary)
      continue;

    // Count divergent predecessors with multiple successors. When there
    // is exactly one such predecessor that is acyclic and dominates the
    // secondary, the rejoin register can be used directly without the
    // accumulator machinery.
    unsigned NumDivergentPreds = 0;
    WaveNode *SingleDivPred = nullptr;
    for (WaveNode *Pred : Secondary->Predecessors) {
      if (!Pred->IsDivergent || Pred->Successors.size() == 1)
        continue;
      NumDivergentPreds++;
      SingleDivPred = Pred;
    }

    // The accumulator is only needed when multiple divergent predecessors
    // contribute rejoin masks, or when cycle membership or non-dominance
    // requires temporal merging across iterations.
    bool HasSingleDivergentPred =
        (NumDivergentPreds == 1) && !SingleDivPred->Cycle &&
        ReconvergeCfg.getDomTree().dominates(SingleDivPred->Block,
                                             Secondary->Block);

    Register Acc = AMDGPU::NoRegister;
    AccBlockInfo AccInfo;

    if (!HasSingleDivergentPred) {
      Acc = Updater.init();
      AccInfo.DefBlocks.push_back(Secondary->Block);
      Updater.addReset(*Secondary->Block, AMDGPULaneMaskUpdater::ResetAtEnd);
    }

    Register DirectRejoin;
    for (WaveNode *Pred : Secondary->Predecessors) {
      if (!Pred->IsDivergent || Pred->Successors.size() == 1)
        continue;

      CFGNodeInfo &PredInfo = NodeInfo.find(Pred)->second;
      Register PrimaryExec = PredInfo.PrimarySuccessorExec;

      Register Rejoin;
      if (!Rejoin) {
        // Try to find a previously generated XOR (or merely masked) value
        // for reuse.
        auto MapIt = RegMap.find(std::make_pair(Pred->Block, PrimaryExec));
        if (MapIt != RegMap.end()) {
          Rejoin = MapIt->second.second;
          if (!Rejoin)
            PrimaryExec = MapIt->second.first;
        }
      }

      if (!Rejoin) {
        Rejoin = LMU.createLaneMaskReg();
        BuildMI(*Pred->Block, Pred->Block->getFirstTerminator(), {},
                TII.get(LMC.XorOpc), Rejoin)
            .addReg(LMC.ExecReg)
            .addReg(PrimaryExec);
      }

      if (HasSingleDivergentPred)
        DirectRejoin = Rejoin;
      else {
        AccInfo.UseBlocks.push_back(Pred->Block);
        AccInfo.DefBlocks.push_back(Pred->Block);
        Updater.addAvailable(*Pred->Block, Rejoin);
      }
    }

    if (Acc)
      AccInfo.UseBlocks.push_back(Secondary->Block);

    Register RejoinMask = HasSingleDivergentPred
                              ? DirectRejoin
                              : Updater.getMergedMask(*Secondary->Block);
    BuildMI(*Secondary->Block, Secondary->Block->getFirstNonPHI(), {},
            TII.get(LMC.OrOpc), LMC.ExecReg)
        .addReg(LMC.ExecReg)
        .addReg(RejoinMask);

    if (Acc)
      finalizeAccInit(Acc, AccInfo);
  }

  Updater.insertAccumulatorResets();
  AccumulatorRegs = std::move(Updater.getAllAccumulators());
  Updater.cleanup();

  // remove unused virtual registers def
  if (RegAllOnes && MRI.use_empty(RegAllOnes)) {
    // getVRegDef can be used since RegAllOnes has a single def
    MRI.getVRegDef(RegAllOnes)->eraseFromParent();
    RegAllOnes = AMDGPU::NoRegister;
  }
  if (RegZero && MRI.use_empty(RegZero)) {
    // getVRegDef can be used since RegZero has a single def
    MRI.getVRegDef(RegZero)->eraseFromParent();
    RegZero = AMDGPU::NoRegister;
  }
}

/// This function fixes virtual register uses that have no dominating definition
/// in the restructured CFG by inserting IMPLICIT_DEF at the nearest
/// common ancestor (NCA) of all its defining blocks.
static void fixMissingDominatingDefs(MachineFunction &MF,
                                     MachineDominatorTree &DomTree,
                                     const TargetInstrInfo &TII) {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  unsigned NumVirtRegs = MRI.getNumVirtRegs();

  for (unsigned VRI = 0; VRI < NumVirtRegs; ++VRI) {
    Register Reg = Register::index2VirtReg(VRI);

    SmallPtrSet<MachineBasicBlock *, 4> DefBlocks;
    for (MachineOperand &DefMO : MRI.def_operands(Reg))
      DefBlocks.insert(DefMO.getParent()->getParent());

    if (DefBlocks.empty())
      continue;

    MachineBasicBlock *NCA = nullptr;
    bool NeedImplicitDef = false;

    for (MachineOperand &UseMO : MRI.use_nodbg_operands(Reg)) {
      if (UseMO.isUndef())
        continue;

      MachineInstr *UseMI = UseMO.getParent();
      MachineBasicBlock *UseBlock = UseMI->getParent();

      bool AnyDefDominates =
          llvm::any_of(MRI.def_operands(Reg), [&](MachineOperand &DefMO) {
            return DomTree.dominates(DefMO.getParent(), UseMI);
          });

      if (!AnyDefDominates) {
        NeedImplicitDef = true;
        NCA =
            NCA ? DomTree.findNearestCommonDominator(NCA, UseBlock) : UseBlock;
      }
    }

    if (!NeedImplicitDef)
      continue;

    for (MachineBasicBlock *DefBlock : DefBlocks)
      NCA = DomTree.findNearestCommonDominator(NCA, DefBlock);

    auto InsertPt = NCA->getFirstTerminator();
    BuildMI(*NCA, InsertPt, NCA->findDebugLoc(InsertPt),
            TII.get(TargetOpcode::IMPLICIT_DEF), Reg);
  }
}

// Fix physical-register liveness after structurization routes a live-out value
// through a block whose regmask clobbers it without redefining it.
//
// Scan blocks affected by changed live-ins and add IMPLICIT_DEFs for registers
// that are still demanded by successors but would be dropped by the regmask
// clobber during live-in recomputation.
static bool fixRegMaskClobberedPhysRegLiveness(
    MachineFunction &MF, const TargetInstrInfo &TII,
    const SmallPtrSetImpl<MachineBasicBlock *> &ChangedLiveIns) {
  if (ChangedLiveIns.empty())
    return false;

  const MachineRegisterInfo &MRI = MF.getRegInfo();
  const TargetRegisterInfo &TRI = *MRI.getTargetRegisterInfo();
  bool Changed = false;

  // Only blocks on the changed live-in boundary can expose a new clobber.
  SmallPtrSet<MachineBasicBlock *, 16> Candidates;
  for (MachineBasicBlock *MBB : ChangedLiveIns) {
    Candidates.insert(MBB);
    for (MachineBasicBlock *Pred : MBB->predecessors())
      Candidates.insert(Pred);
  }

  for (MachineBasicBlock *MBB : Candidates) {
    // Locate the first MI with a regmask operand in MBB via forward walk.
    // So, later during backward traversl during liveness computation, it can
    // stop once it reaches this MI.
    MachineInstr *FirstRegMaskMO = nullptr;
    for (MachineInstr &MI : *MBB) {
      if (any_of(MI.operands(),
                 [](const MachineOperand &MO) { return MO.isRegMask(); })) {
        FirstRegMaskMO = &MI;
        break;
      }
    }
    if (!FirstRegMaskMO)
      continue;

    LivePhysRegs LiveRegs(TRI);
    LiveRegs.addLiveOutsNoPristines(*MBB);
    if (LiveRegs.empty())
      continue;

    SmallDenseSet<MCPhysReg, 8> RegsWithLivenessGap;
    SmallVector<MCPhysReg, 8> SortedRegs;
    for (MachineInstr &MI : reverse(*MBB)) {
      for (const MachineOperand &MO : MI.operands()) {
        if (!MO.isRegMask())
          continue;
        // A clobbered live-out register needs a covering def below the regmask.
        // Skip reserved registers.
        for (MCPhysReg Reg : LiveRegs) {
          if (MRI.isReserved(Reg))
            continue;
          if (MO.clobbersPhysReg(Reg) && !MI.definesRegister(Reg, &TRI))
            RegsWithLivenessGap.insert(Reg);
        }
      }
      // Instructions above the first regmask cannot expose this clobber.
      if (&MI == FirstRegMaskMO)
        break;
      LiveRegs.stepBackward(MI);
    }

    if (RegsWithLivenessGap.empty())
      continue;

    // Remove any sub-regs already covered by a super-register in the gap.
    SmallVector<MCPhysReg, 4> SubRegsToRemove;
    for (MCPhysReg Reg : RegsWithLivenessGap) {
      if (any_of(TRI.superregs(Reg),
                 [&](MCPhysReg Super) { return RegsWithLivenessGap.contains(Super); }))
        SubRegsToRemove.push_back(Reg);
    }
    for (MCPhysReg Reg : SubRegsToRemove)
      RegsWithLivenessGap.erase(Reg);

    // Insert in a deterministic (register-number) order for stable output.
    SortedRegs.assign(RegsWithLivenessGap.begin(), RegsWithLivenessGap.end());
    llvm::sort(SortedRegs);

    MachineBasicBlock::iterator InsertPt = MBB->getFirstTerminator();
    DebugLoc DL = MBB->findDebugLoc(InsertPt);
    const MCInstrDesc &ImplDefInstr = TII.get(TargetOpcode::IMPLICIT_DEF);
    for (MCPhysReg Reg : SortedRegs) {
      BuildMI(*MBB, InsertPt, DL, ImplDefInstr, Reg);
      Changed = true;
    }
  }

  return Changed;
}

namespace {

// Abstract value tracked per register: either a known-zero constant
// (monostate) or an alias to another virtual register.
using RegIntVariant = std::variant<std::monostate, Register>;
// maps Register to variant
using RegValueMap = DenseMap<Register, RegIntVariant>;
// maps MBB to variant state resolved at end off MBB
using BlockValueMap = DenseMap<MachineBasicBlock *, RegValueMap>;

// return known zero variant
static inline RegIntVariant zeroVal() {
  return RegIntVariant{std::monostate{}};
}
// True iff V holds known zero variant
static inline bool isZero(const RegIntVariant &V) {
  return std::holds_alternative<std::monostate>(V);
}
// returns mapped register if V holds a register alias, else null
static inline const Register *asReg(const RegIntVariant &V) {
  return std::get_if<Register>(&V);
}

// Dataflow-based forward propagation simplifier for accumulator-register
// instructions. Tracks per-block mappings of Reg -> (known 0 | virutal sgpr)
// and uses them to fold COPY/OR/XOR patterns and rewrite acc uses.
// Example:
// BB0:
//    succ: BB1
//    %acc1 = MOV 0
//    %acc2 = COPY %sgpr1
//    %acc3 = %sgpr2 AND $exec
//
//  BlockValueMap OUT[BB0] = { %acc1 -> 0, %acc2 -> %sgpr1 }
//
// BB1: Merged incoming values from all preds : { %acc1 -> 0, %acc2 -> %sgpr1 }
//    pred: BB0
//    %acc1 = %acc1 OR %sgpr3 ->  %acc1 = COPY %sgpr3
//    %acc3 = COPY %acc2  ->  %acc3 = COPY %sgpr1
//    %acc3 = %acc3 OR $exec -> %acc3 = %sgpr1 OR $exec
//
//  BlockValueMap OUT[BB1] = { %acc1 -> %sgpr3, %acc2 -> %sgpr1 }
//
class ForwardPropSimplifier {
  MachineFunction &MF;
  const SIInstrInfo &TII;
  const AMDGPU::LaneMaskConstants &LMC;
  const AccRegSet &AccRegs;
  AccInstsMap &AccInstMap;
  BlockValueMap
      OUT; // resolved Accumulator register values at end of each block.

  void replaceWithMov(MachineInstr &MI, int64_t ImmVal) {
    MI.setDesc(TII.get(LMC.MovOpc));
    while (MI.getNumOperands() > 1)
      MI.removeOperand(MI.getNumOperands() - 1);
    MI.addOperand(MachineOperand::CreateImm(ImmVal));
    ++NumCleanupInstrsOptimized;
  }

  void replaceWithCopy(MachineInstr &MI, Register SrcReg) {
    MI.setDesc(TII.get(AMDGPU::COPY));
    while (MI.getNumOperands() > 1)
      MI.removeOperand(MI.getNumOperands() - 1);
    MI.addOperand(MachineOperand::CreateReg(SrcReg, false));
    ++NumCleanupInstrsOptimized;
  }

  // merge incoming OUT values from MBB's predecessors:
  // IN[MBB] = Intersection of all OUT[MBB.preds]
  void mergeIncoming(MachineBasicBlock &MBB, RegValueMap &Result) {
    if (MBB.pred_empty())
      return;

    auto FirstPredIt = OUT.find(*MBB.pred_begin());
    if (FirstPredIt == OUT.end())
      return;

    for (const auto &PredOUT : FirstPredIt->second) {
      Register Reg = PredOUT.first;
      const RegIntVariant &Expected = PredOUT.second;
      if (llvm::all_of(MBB.predecessors(), [&](MachineBasicBlock *Pred) {
            auto PredIt = OUT.find(Pred);
            if (PredIt == OUT.end())
              return false;
            auto RegIt = PredIt->second.find(Reg);
            return RegIt != PredIt->second.end() && RegIt->second == Expected;
          }))
        Result[Reg] = Expected;
    }
  }

  const RegIntVariant *lookupValue(const RegValueMap &Map, Register Reg) {
    auto It = Map.find(Reg);
    return (It != Map.end()) ? &It->second : nullptr;
  }

  // Rewrite acc-register uses in MI to their tracked equivalent registers.
  void rewriteAccUses(MachineInstr &MI, const RegValueMap &Cur) {
    for (MachineOperand &MO : MI.explicit_uses()) {
      if (!isAccRegOperand(MO, AccRegs))
        continue;
      auto It = Cur.find(MO.getReg());
      if (It == Cur.end())
        continue;
      if (const Register *R = asReg(It->second)) {
        MO.setReg(*R);
        ++NumCleanupInstrsOptimized;
      }
    }
  }

  //  Cur[Dst] = Cur[Src], if Cur[Src] exists
  //           = Src, if Cur[Src] does not exists and Src is virtual
  // if none of the above hold true, it means Cur[Dst] cannot be resolved to a
  // known 0 or a virtual sgpr register and must be removed from the map
  void updateRegValueMapEntry(RegValueMap &Cur, Register Dst, Register Src) {
    if (const RegIntVariant *V = lookupValue(Cur, Src))
      Cur[Dst] = *V;
    else if (Src.isVirtual())
      Cur[Dst] = RegIntVariant{Src};
    else
      Cur.erase(Dst);
  }

  // If exactly one of {A, B} is a known zero, return the other; else {}.
  Register otherIfOneIsZero(const RegValueMap &Cur, Register A, Register B) {
    const RegIntVariant *VA = lookupValue(Cur, A);
    const RegIntVariant *VB = lookupValue(Cur, B);
    if (VA && isZero(*VA))
      return B;
    if (VB && isZero(*VB))
      return A;
    return AMDGPU::NoRegister;
  }

  // forward walk on MBB's accumulator instructions, simplifying the
  // instructions using the tracked values in Cur, simultaneously updating Cur
  // with the new known 0 or virtual sgpr registers
  void forwardPropAccValues(MachineBasicBlock &MBB, RegValueMap &Cur) {
    auto MapIt = AccInstMap.find(&MBB);
    if (MapIt == AccInstMap.end())
      return;

    for (MachineInstr *MI : MapIt->second) {
      rewriteAccUses(*MI, Cur);

      const unsigned Opc = MI->getOpcode();
      const Register Dst = MI->getOperand(0).getReg();

      // ACC = MOV 0
      if (AccRegs.count(Dst) && Opc == LMC.MovOpc) {
        const MachineOperand &Imm = MI->getOperand(1);
        if (Imm.isImm() && Imm.getImm() == 0)
          Cur[Dst] = zeroVal();
        else
          llvm_unreachable("Accumulator MOV must be MOV 0");
        continue;
      }

      // ACC = COPY %vreg | ACC = COPY EXEC | ACC = COPY ACC
      if (Opc == AMDGPU::COPY) {
        Register Src = MI->getOperand(1).getReg();
        if (const RegIntVariant *SrcVal = lookupValue(Cur, Src)) {
          if (isZero(*SrcVal))
            replaceWithMov(*MI, 0);
          else if (const Register *R = asReg(*SrcVal))
            replaceWithCopy(*MI, *R);
          else
            llvm_unreachable("Invalid register value");
          Cur[Dst] = *SrcVal;
        } else if (Src.isVirtual()) {
          Cur[Dst] = RegIntVariant{Src};
        } else {
          Cur.erase(Dst);
        }
        continue;
      }

      // ACC = %val AND EXEC
      if (Opc == LMC.AndOpc) {
        Cur.erase(Dst);
        continue;
      }

      // ACC = ACC OR %masked | ACC = ACC OR EXEC
      if (Opc == LMC.OrOpc) {
        Register Src1 = MI->getOperand(1).getReg();
        Register Src2 = MI->getOperand(2).getReg();
        Register Other = otherIfOneIsZero(Cur, Src1, Src2);
        if (!Other) {
          Cur.erase(Dst);
          continue;
        }
        replaceWithCopy(*MI, Other);
        updateRegValueMapEntry(Cur, Dst, Other);
        continue;
      }

      // ACC = %a XOR %b
      if (Opc == LMC.XorOpc) {
        Register Src1 = MI->getOperand(1).getReg();
        Register Src2 = MI->getOperand(2).getReg();
        if (Register Other = otherIfOneIsZero(Cur, Src1, Src2))
          replaceWithCopy(*MI, Other);
        continue;
      }
    }
  }

  // merge incoming OUT values from MBB's predecessors and forward propagate
  // merged values to obtain updated OUT for MBB
  bool processBlock(MachineBasicBlock &MBB) {
    RegValueMap RegValues;
    mergeIncoming(MBB, RegValues);
    forwardPropAccValues(MBB, RegValues);
    auto It = OUT.find(&MBB);

    // OUT changes if forward propageted values are different from the current
    // OUT for MBB or OUT value for MBB does not exist
    bool OutChanged = (It == OUT.end() || It->second != RegValues);
    if (OutChanged)
      OUT[&MBB] = std::move(RegValues);
    return OutChanged;
  }

public:
  ForwardPropSimplifier(MachineFunction &MF, const SIInstrInfo &TII,
                        const AMDGPU::LaneMaskConstants &LMC,
                        const AccRegSet &AccRegs, AccInstsMap &AccInstMap)
      : MF(MF), TII(TII), LMC(LMC), AccRegs(AccRegs), AccInstMap(AccInstMap) {}

  void run() {
    ReversePostOrderTraversal<MachineFunction *> RPOT(&MF);
    SmallVector<MachineBasicBlock *, 32> RPOBlocks(
        RPOT.begin(), RPOT.end());                    // maps RPO index to MBB
    DenseMap<MachineBasicBlock *, unsigned> RPOIndex; // maps MBB to RPO index
    unsigned N = RPOBlocks.size();
    RPOIndex.reserve(N);
    for (unsigned I = 0, E = N; I != E; ++I)
      RPOIndex[RPOBlocks[I]] = I;

    // Ping-pong bitvector worklists indexed by RPO number.

    BitVector Active(N, true); // bitvector of active MBBs
    BitVector Next(N);         // bitvector of next MBBs

    // iterate until fix point : no OUT changes in any block
    while (Active.any()) {
      for (unsigned I : Active.set_bits()) {
        MachineBasicBlock *MBB = RPOBlocks[I];
        if (!processBlock(*MBB)) // MBB is processed to fixed point, skip
          continue;
        for (MachineBasicBlock *Succ : MBB->successors()) {
          auto It = RPOIndex.find(Succ);
          assert(It != RPOIndex.end() && "Invalid RPO index");
          Next.set(It->second);
        }
      }
      Active.reset();
      std::swap(Active, Next);
    }
  }
};

/// \brief Core wave transform logic.
class AMDGPUWaveTransform {
public:
  AMDGPUWaveTransform(MachineDominatorTree &DomTree,
                      MachineCycleInfo &CycleInfo)
      : DomTree(&DomTree), CycleInfo(&CycleInfo) {}

  bool run(MachineFunction &MF);

private:
  void cleanup(MachineFunction &MF, const AccRegSet &AccumulatorRegs);

  MachineDominatorTree *DomTree = nullptr;
  MachineCycleInfo *CycleInfo;
  const SIInstrInfo *TII;
};

/// \brief Legacy wave transform machine function pass.
class AMDGPUWaveTransformLegacy : public MachineFunctionPass {
public:
  static char ID;

  AMDGPUWaveTransformLegacy() : MachineFunctionPass(ID) {
    initializeAMDGPUWaveTransformLegacyPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "AMDGPU Control Flow Wave Transform";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineCycleInfoWrapperPass>();
    AU.addRequired<MachineDominatorTreeWrapperPass>();
    AU.addPreserved<MachineDominatorTreeWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  // TODO-WAVETRANSFORM: This pass should not add PHI instructions.
  // MachineFunctionProperties getSetProperties() const override {
  //   return MachineFunctionProperties().set(
  //       MachineFunctionProperties::Property::NoPHIs);
  // }

  MachineFunctionProperties getClearedProperties() const override {
    // New virtual registers will be introduced.
    return MachineFunctionProperties().setIsSSA().setNoVRegs();
  }
};

} // End anonymous namespace.

INITIALIZE_PASS_BEGIN(AMDGPUWaveTransformLegacy, DEBUG_TYPE,
                      "AMDGPU Wave Transform", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineCycleInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_END(AMDGPUWaveTransformLegacy, DEBUG_TYPE,
                    "AMDGPU Wave Transform", false, false)

char AMDGPUWaveTransformLegacy::ID = 0;
char &llvm::AMDGPUWaveTransformID = AMDGPUWaveTransformLegacy::ID;

FunctionPass *llvm::createAMDGPUWaveTransformPass() {
  return new AMDGPUWaveTransformLegacy();
}

/// \brief Run the wave transform.
bool AMDGPUWaveTransform::run(MachineFunction &MF) {
  SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  if (MF.size() <= 1) {
    // Skip this pass for MFs without control flow; set WaveCFG property first.
    MFI->setWaveCFG(true);
    return false;
  }

  TII = MF.getSubtarget<GCNSubtarget>().getInstrInfo();

  // Step 1: Compute reconverging Wave CFG
  ReconvergeCFGHelper ReconvergeHelper(*CycleInfo, *DomTree);
  ReconvergeHelper.run();

  ControlFlowRewriter CFRewriter(MF, ReconvergeHelper);
  CFRewriter.prepareWaveCfg();

  if (AMDGPUWaveTransformPrintFinal) {
    dbgs() << "Wave CFG for " << MF.getName() << ":\n";
    ReconvergeHelper.printNodes(dbgs());
  }
  // Step 2: Create basic blocks for flow nodes and adjust MachineBasicBlock
  // successor and predecessor lists.
  MachineFunction::iterator insertIt = MF.end();
  for (auto *WN : llvm::reverse(ReconvergeHelper.nodes())) {
    if (!WN->Block) {
      WN->Block = MF.CreateMachineBasicBlock();
      MF.insert(insertIt, WN->Block);
      ReconvergeHelper.setNodeForBlock(WN->Block, WN);
    }
    insertIt = WN->Block->getIterator();
  }

  SmallVector<cfg::Update<MachineBasicBlock *>, 8> CFGUpdates;
  SmallVector<MachineBasicBlock *, 2> SuccToRemove;

  for (auto *WN : ReconvergeHelper.nodes()) {
    for (MachineBasicBlock *CurrentSucc : WN->Block->successors()) {
      if (llvm::find_if(WN->Successors, [=](WaveNode *Node) {
            return Node->Block == CurrentSucc;
          }) == WN->Successors.end())
        SuccToRemove.push_back(CurrentSucc);
    }
    for (MachineBasicBlock *Succ : SuccToRemove) {
      WN->Block->removeSuccessor(Succ);
      CFGUpdates.emplace_back(cfg::UpdateKind::Delete, WN->Block, Succ);
    }
    SuccToRemove.clear();

    for (auto *succ : WN->Successors) {
      if (!is_contained(WN->Block->successors(), succ->Block)) {
        WN->Block->addSuccessor(succ->Block);
        CFGUpdates.emplace_back(cfg::UpdateKind::Insert, WN->Block,
                                succ->Block);
      }
    }
  }

  DomTree->applyUpdates(CFGUpdates);
  CFGUpdates.clear();


  // Step 3: Fix up terminators and insert rejoin masks.
  CFRewriter.rewrite();
  // Step 4: Fix missing dominating defs (non-SSA).
  // The wave transform inserts flow blocks and reroutes edges, which can
  // create CFG paths to a use that bypass all defs of a register, violating
  // the CFG dominance relations.
  fixMissingDominatingDefs(MF, *DomTree, *TII);
  cleanup(MF, CFRewriter.getAccumulatorRegs());

  // FIXME: restore the following 1 line:
  // UI.clear();
  // ConvergenceInfo.clear();
  DomTree = nullptr;

  // Recompute LiveIns and record blocks whose live-ins are affected by CFG
  // rewiring.
  ReversePostOrderTraversal<MachineFunction *> RPOT(&MF);
  SmallVector<MachineBasicBlock *, 16> PostOrder;
  for (auto MBB : reverse(RPOT))
    PostOrder.push_back(MBB);

  SmallPtrSet<MachineBasicBlock *, 16> ChangedLiveIns;
  bool LiveInChanged;
  do {
    LiveInChanged = false;
    for (MachineBasicBlock *MBB : PostOrder) {
      if (recomputeLiveIns(*MBB)) {
        ChangedLiveIns.insert(MBB);
        LiveInChanged = true;
      }
    }
  } while (LiveInChanged);

  // Repair live-out physical registers clobbered by regmasks on rewired paths.
  fixRegMaskClobberedPhysRegLiveness(MF, *TII, ChangedLiveIns);

  MFI->setWaveCFG(true);

  return true; // assume that we changed something
}

namespace {

// Per-block use-def recorded for accumulator registers.
struct AccBBUseDef {
  AccRegSet UpwardExposedUse; // uses without a preceeding def in the same BB
  AccRegSet Def;
  SmallDenseMap<Register, MachineInstr *, 4>
      UnusedDef; // def not follwed by any use in same BB
};

// Per-block LiveIn and LiveOut sets for accumulator registers.
struct AccBBLiveness {
  AccRegSet LiveIn;
  AccRegSet LiveOut;
};

} // anonymous namespace

// Computes the following for all BBs:
// Upward exposed uses - Use of accumulator register without a preceeding def in
// the same BB Def - Def of accumulator register UnusedDef - Def of the
// accumulator register with no use after it in the same BB
static void
computeAccUseDef(MachineFunction &MF, const AccRegSet &AccumulatorRegs,
                 DenseMap<MachineBasicBlock *, AccBBUseDef> &Result) {

  for (MachineBasicBlock &MBB : MF) {
    AccBBUseDef &UD = Result[&MBB];
    for (MachineInstr &MI : MBB) {
      for (const MachineOperand &MO : MI.explicit_uses()) {
        if (!isAccRegOperand(MO, AccumulatorRegs))
          continue;
        Register R = MO.getReg();
        if (!UD.Def.count(R))
          UD.UpwardExposedUse.insert(R);
        UD.UnusedDef.erase(R);
      }
      for (const MachineOperand &MO : MI.defs()) {
        if (!isAccRegOperand(MO, AccumulatorRegs))
          continue;
        Register R = MO.getReg();
        UD.Def.insert(R);
        UD.UnusedDef[R] = &MI;
      }
    }
  }
}

// Compute per-block LiveIn/LiveOut for accumulator registers using
// iterative fixed-point dataflow analysis.
//   LiveOut(B) = ∪ LiveIn(S) for all successors S of B
//   LiveIn(B)  = Use(B) ∪ (LiveOut(B) - Def(B))
static void
computeAccLiveness(MachineFunction &MF,
                   const DenseMap<MachineBasicBlock *, AccBBUseDef> &UseDef,
                   DenseMap<MachineBasicBlock *, AccBBLiveness> &Liveness) {
  for (const auto &[MBB, UD] : UseDef)
    Liveness[MBB].LiveIn = UD.UpwardExposedUse;

  // Backward dataflow: sweep blocks in post-order so information propagates
  // toward predecessors quickly. Needed because a layout-order sweep converges
  // slowly and does not scale to very large CFGs.
  //
  // TODO-WAVETRANSFORM: still re-scans all blocks per iteration; replace with a
  // predecessor worklist (and bit-vector reg sets) to reprocess blocks only
  // when needed.
  ReversePostOrderTraversal<MachineFunction *> RPOT(&MF);
  SmallVector<MachineBasicBlock *, 32> Order(RPOT.begin(), RPOT.end());
  std::reverse(Order.begin(), Order.end());

  bool Changed = true;
  while (Changed) {
    Changed = false;
    for (MachineBasicBlock *MBB : Order) {
      AccBBLiveness &L = Liveness[MBB];
      const AccBBUseDef &UD = UseDef.find(MBB)->second;

      for (MachineBasicBlock *Succ : MBB->successors()) {
        for (Register R : Liveness[Succ].LiveIn) {
          if (L.LiveOut.insert(R).second)
            Changed = true;
        }
      }

      for (Register R : L.LiveOut) {
        if (UD.Def.count(R))
          continue;
        if (L.LiveIn.insert(R).second)
          Changed = true;
      }
    }
  }
}

// Build a per-MBB program-order list of all machine instrs that use or define
// any register from the AccumulatorRegs set
static void buildAccInstMap(MachineFunction &MF,
                            const AccRegSet &AccumulatorRegs,
                            AccInstsMap &AccInstMap) {
  for (MachineBasicBlock &MBB : MF) {
    SmallVector<MachineInstr *, 8> OrderedAccInsts;

    for (MachineInstr &MI : MBB) {
      for (const MachineOperand &MO : MI.explicit_operands()) {
        if (isAccRegOperand(MO, AccumulatorRegs)) {
          OrderedAccInsts.push_back(&MI);
          break;
        }
      }
    }

    if (!OrderedAccInsts.empty())
      AccInstMap[&MBB] = std::move(OrderedAccInsts);
  }
}

// remove dead defs and simplify the accumulator instrs added during rewrite
void AMDGPUWaveTransform::cleanup(MachineFunction &MF,
                                  const AccRegSet &AccumulatorRegs) {
  if (AccumulatorRegs.empty())
    return;

  AccInstsMap AccInstMap;
  buildAccInstMap(MF, AccumulatorRegs, AccInstMap);

  const auto &LMC =
      AMDGPU::LaneMaskConstants::get(MF.getSubtarget<GCNSubtarget>());

  // forward propagate accumulator values
  ForwardPropSimplifier Simplifier(MF, *TII, LMC, AccumulatorRegs, AccInstMap);
  Simplifier.run();

  // Remove locally dead defs: def followed by re-def with no intervening use in
  // the same BB
  for (auto &[MBB, AccInsts] : AccInstMap) {
    DenseMap<Register, MachineInstr *> LastUnusedDef;
    SmallVector<MachineInstr *, 4> LocalDeadDefs;

    for (MachineInstr *MI : AccInsts) {
      for (const MachineOperand &MO : MI->explicit_uses()) {
        if (isAccRegOperand(MO, AccumulatorRegs))
          LastUnusedDef.erase(MO.getReg());
      }
      for (const MachineOperand &MO : MI->defs()) {
        if (!isAccRegOperand(MO, AccumulatorRegs))
          continue;
        Register Reg = MO.getReg();
        auto It = LastUnusedDef.find(Reg);
        if (It != LastUnusedDef.end())
          LocalDeadDefs.push_back(It->second);
        LastUnusedDef[Reg] = MI;
      }
    }

    NumCleanupInstrsRemoved += LocalDeadDefs.size();
    for (MachineInstr *MI : LocalDeadDefs)
      MI->eraseFromParent();
  }

  // compute Use Def sets for each block
  DenseMap<MachineBasicBlock *, AccBBUseDef> AccUseDef;
  computeAccUseDef(MF, AccumulatorRegs, AccUseDef);

  DenseMap<MachineBasicBlock *, AccBBLiveness> AccLiveness;
  computeAccLiveness(MF, AccUseDef, AccLiveness);

  // Remove globally dead defs: UnusedDef(B) - LiveOut(B).
  // These are defs defined in a BB and not used anywhere in the forward data
  // flow
  for (const auto &[MBB, UD] : AccUseDef) {
    const AccBBLiveness &L = AccLiveness.find(MBB)->second;
    for (const auto &[R, DefMI] : UD.UnusedDef) {
      if (L.LiveOut.count(R))
        continue;

      DefMI->eraseFromParent();
      ++NumCleanupInstrsRemoved;
    }
  }
}

bool AMDGPUWaveTransformLegacy::runOnMachineFunction(MachineFunction &MF) {
  MachineDominatorTree &DomTree =
      getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  MachineCycleInfo &CycleInfo =
      getAnalysis<MachineCycleInfoWrapperPass>().getCycleInfo();
  return AMDGPUWaveTransform(DomTree, CycleInfo).run(MF);
}

PreservedAnalyses
llvm::AMDGPUWaveTransformPass::run(MachineFunction &MF,
                                   MachineFunctionAnalysisManager &MFAM) {
  MFPropsModifier _(*this, MF);

  MachineDominatorTree &DomTree =
      MFAM.getResult<MachineDominatorTreeAnalysis>(MF);
  MachineCycleInfo &CycleInfo = MFAM.getResult<MachineCycleAnalysis>(MF);

  if (!AMDGPUWaveTransform(DomTree, CycleInfo).run(MF))
    return PreservedAnalyses::all();

  auto PA = getMachineFunctionPassPreservedAnalyses();
  PA.preserve<MachineDominatorTreeAnalysis>();
  return PA;
}
