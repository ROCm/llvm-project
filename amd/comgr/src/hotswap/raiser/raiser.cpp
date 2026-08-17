//===- raiser.cpp - Hotswap MC -> LLVM IR raiser -------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Decodes a kernel's .text into a typed `DecodedInst` stream, seeds the entry
// registers from the descriptor-derived ABI layouts, dispatches each decoded
// instruction to its per-format handler, promotes the register-file allocas to
// SSA, and verifies the resulting `amdgpu_kernel` function.
//
//===----------------------------------------------------------------------===//

#include "raiser.h"
#include "hotswap/decoder/amdgpu-formats.h"
#include "hotswap/decoder/canonical-op.h"
#include "hotswap/common/kernel-meta.h"
#include "hotswap/loader/code-object-utils.h"
#include "hotswap/decoder/decode.h"
#include "hotswap/decoder/decoded-inst.h"
#include "raise_failure.h"
#include "hotswap/decoder/isa-profile.h"
#include "hotswap/decoder/parsed-reg.h"

#include "comgr.h"
#include "SIDefines.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "handlers.h"
#include "kernarg-layout.h"
#include "hotswap/decoder/mc-state.h"
#include "hotswap/decoder/opcode-map.h"
#include "raise-context.h"
#include "reg-file.h"
#include "source-hidden-args.h"
#include "user-sgpr-layout.h"
#include "wave-projection.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/AMDHSAKernelDescriptor.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/TargetParser.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"

#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <cassert>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <utility>

#define DEBUG_TYPE "wave-projection"

using namespace llvm;

namespace COMGR::hotswap {

namespace {

// Hardware threads-per-block maximum for the gfx9/CDNA wave64 targets the
// doubled dispatch scales up to. A source block that would exceed this once
// scaled by W_t / W_s cannot be doubled.
constexpr unsigned kTargetMaxThreadsPerBlock = 1024;

} // namespace

// parseReg, readOp32/64/ExecWidth, and OpResolver are in raise-context.h/cpp
// instructionWritesEXEC and the cross-wave gate live in wave-projection.h/cpp
// RaiseFailure + reasonString are in raise-failure.h/cpp

// ============================================================================
// Main raising function
// ============================================================================

static Expected<RaiseResult>
raiseToIRImpl(llvm::ArrayRef<uint8_t> TextBytes, llvm::StringRef SourceIsa,
              llvm::StringRef KernelName, const KernelMeta &Meta,
              uint64_t KernelOffset, uint64_t KernelSize,
              uint64_t TextBaseAddress,
              llvm::ArrayRef<TextSection::ImageSection> SourceImageSections,
              llvm::StringRef CompilationTargetIsa, bool EnableWritelaneRewrite,
              bool EnableWaveNative, bool ForceThreadLoopProjection,
              bool SuppressC5ForThreadLoopRoute, bool ForceReplicationDoubled,
              bool AssumeHipGlobalOffsetZero, RaiseStats *Stats) {
  RaiseResult Result;

  // Reject obviously-bad ISA inputs before reaching the MC stack -- an
  // empty or non-AMDGPU ISA string slips past `createMCSubtargetInfo`
  // (it returns a subtarget with no features) and only blows up later
  // in `createMCDisassembler` with an `llvm_unreachable`-flavoured
  // `LLVM ERROR: disassembly not yet supported for subtarget` that
  // aborts the process. Surface a structured failure instead.
  //
  // Callers may pass either the bare processor name (`gfx942`) or the
  // canonical AMDGPU ISA string (`amdgcn-amd-amdhsa--gfx942[:feat...]`).
  // Defer to Comgr's `parseTargetIdentifier` for the canonical form (it
  // handles the dash-separated Arch/Vendor/OS/Environ/Processor split
  // and the `:sramecc+/-:xnack+/-` feature suffix in one place);
  // `MCSubtargetInfo` only accepts the bare processor name, so we
  // forward `Ident.Processor` to the MC stack below.
  auto NormalizeIsa = [](StringRef IsaString) -> StringRef {
    TargetIdentifier Ident;
    if (parseTargetIdentifier(IsaString, Ident) == AMD_COMGR_STATUS_SUCCESS)
      return Ident.Processor;
    // Bare processor name (e.g. `gfx942`) -- not a 5-component canonical
    // ISA string. Return as-is and let the AMDGPU validator below decide.
    return IsaString;
  };
  StringRef SourceCpu = NormalizeIsa(SourceIsa);
  if (SourceIsa.empty() ||
      AMDGPU::parseArchAMDGCN(SourceCpu) == AMDGPU::GK_NONE) {
    return RaiseFailure::general(RaiseFailureReason::BadInput,
                                 "source ISA '" + SourceIsa +
                                     "' does not name an AMDGPU GPU");
  }

  // Same normalisation for the target-side override (--target-isa on
  // raise_cli, or programmatic CompilationTargetIsa). Empty means
  // "translate in place -- reuse the source profile".
  StringRef TargetCpu = CompilationTargetIsa.empty()
                            ? CompilationTargetIsa
                            : NormalizeIsa(CompilationTargetIsa);

  Expected<MCState> MCStateOrErr = initMCState(SourceCpu);
  if (!MCStateOrErr) {
    return MCStateOrErr.takeError();
  }

  MCState Mc = std::move(*MCStateOrErr);
  ISAProfile Isa = ISAProfile::fromSubtarget(*Mc.SubtargetInfo);
  // When the caller does not specify a distinct compilation target we raise
  // in place and reuse the source profile; otherwise we spin up a throwaway
  // MCSubtargetInfo just to snapshot the target's feature bits.
  ISAProfile TargetIsa = Isa;
  std::unique_ptr<MCSubtargetInfo> TargetSti;
  if (!TargetCpu.empty()) {
    Expected<std::unique_ptr<MCSubtargetInfo>> StiOrErr =
        buildSubtargetInfo(*Mc.Target, TargetCpu);
    if (!StiOrErr)
      return StiOrErr.takeError();

    TargetSti = std::move(*StiOrErr);
    TargetIsa = ISAProfile::fromSubtarget(*TargetSti);
  }
  if (!Isa.hasValidWaveSize())
    return RaiseFailure::general(
        RaiseFailureReason::InternalError,
        "source ISA profile has unsupported wave size " +
            Twine(Isa.waveSize()));
  if (!TargetIsa.hasValidWaveSize())
    return RaiseFailure::general(
        RaiseFailureReason::InternalError,
        "target ISA profile has unsupported wave size " +
            Twine(TargetIsa.waveSize()));

  // Create LLVMContext + common IR types here so the WaveProjection has
  // access to i32/i64 before the cross-wave gate runs. The module is created
  // lazily in Phase 2 so early-return paths (pre-translation aborts) don't
  // leave behind a half-built module.
  Result.Ctx = std::make_unique<LLVMContext>();
  LLVMContext &C = *Result.Ctx;
  auto *I32Ty = Type::getInt32Ty(C);
  auto *I64Ty = Type::getInt64Ty(C);

  // Select the wave projection. ReplicationProjection is the default:
  // it maps each target lane onto `lane_id mod W_src` of the source EXEC mask
  // and truncates cross-wave ballots to source width. WaveNativeProjection is
  // the wave32 -> wave64 alternative that forces hardware EXEC = -1 (via
  // `init_whole_wave`) and re-derives the per-lane predicate at each
  // side-effect site through `emitUnderExec`.
  //
  // In the phantom-lane regime (max_flat_workgroup_size < target wave size)
  // the extra target lanes carry no source workitem: their VGPRs hold
  // undef/dispatcher state that cross-lane ops could leak into active lanes,
  // faulting a later gated load. Fall back to MODREP there, which keeps those
  // lanes hardware-inactive for the whole kernel body.
  const bool PhantomLaneRegime =
      Meta.MaxFlatWorkgroupSize > 0 &&
      static_cast<unsigned>(Meta.MaxFlatWorkgroupSize) < TargetIsa.waveSize();
  const bool UseThreadLoop = ForceThreadLoopProjection;
  const bool WideningWave32To64 = Isa.isWave32() && !TargetIsa.isWave32();
  // ReplicationDoubledDispatchProjection is selected only via the C5
  // y/z-refusal
  // upgrade retry (or an explicit --force-modrep-doubled), so it is a forced
  // route just like ThreadLoop; it takes precedence over WaveNative.
  const bool UseReplicationDoubled =
      !UseThreadLoop && ForceReplicationDoubled && WideningWave32To64;
  const bool UseWaveNative = !UseThreadLoop && !UseReplicationDoubled &&
                             EnableWaveNative && WideningWave32To64 &&
                             !PhantomLaneRegime;

  // Size gate for the doubled dispatch: the runtime scales the block by
  // W_t / W_s along x, so the scaled flat size must not exceed the target's
  // hardware threads-per-block maximum.
  if (UseReplicationDoubled) {
    const unsigned Factor = TargetIsa.waveSize() / Isa.waveSize();
    const unsigned SourceFlat =
        Meta.MaxFlatWorkgroupSize > 0 ? Meta.MaxFlatWorkgroupSize : 1024;
    if (SourceFlat * Factor > kTargetMaxThreadsPerBlock) {
      std::string Detail =
          (Twine("ReplicationDoubledDispatchProjection needs to launch ") +
           Twine(SourceFlat * Factor) +
           " threads/block (source max_flat_workgroup_size " +
           Twine(SourceFlat) + " scaled by " + Twine(Factor) +
           ") but the target hardware limit is " +
           Twine(kTargetMaxThreadsPerBlock) +
           "; refuse rather than truncate the block.")
              .str();
      return RaiseFailure::inKernel(RaiseFailureReason::CrossWavePredicateChain,
                                    KernelName, Detail);
    }
  }

  std::unique_ptr<WaveProjection> ProjectionPtr;
  if (UseThreadLoop) {
    ProjectionPtr =
        std::make_unique<ThreadLoopProjection>(Isa, TargetIsa, I32Ty, I64Ty);
    LLVM_DEBUG(dbgs() << "transpiler: kernel '" << KernelName
                      << "' selected ThreadLoopProjection\n");
  } else if (UseReplicationDoubled) {
    ProjectionPtr = std::make_unique<ReplicationDoubledDispatchProjection>(
        Isa, TargetIsa, I32Ty, I64Ty);
    LLVM_DEBUG(dbgs() << "transpiler: kernel '" << KernelName
                      << "' selected ReplicationDoubledDispatchProjection\n");
  } else if (UseWaveNative) {
    ProjectionPtr =
        std::make_unique<WaveNativeProjection>(Isa, TargetIsa, I32Ty, I64Ty);
  } else {
    ProjectionPtr = std::make_unique<ReplicationProjection>(
        Isa, TargetIsa, I32Ty, I64Ty);
  }
  ProjectionPtr->setMaxFlatWorkgroupSize(Meta.MaxFlatWorkgroupSize);
  WaveProjection &Projection = *ProjectionPtr;

  // Record the doubled-dispatch requirement so the launch runtime scales
  // exactly this kernel's dispatch (threaded through the transpile result and
  // the loader). Non-doubled projections leave dim=-1 / factor=1.
  if (Projection.usesDoubledDispatch()) {
    Result.DoubledDispatchDim =
        static_cast<int>(Projection.doubledDispatchDim());
    Result.DoubledDispatchFactor = Projection.doubledDispatchFactor();
  }

  if (!UseThreadLoop && !UseReplicationDoubled && EnableWaveNative &&
      PhantomLaneRegime && Isa.isWave32() && !TargetIsa.isWave32()) {
    LLVM_DEBUG(dbgs() << "transpiler: kernel '" << KernelName
                      << "' is in phantom-lane regime "
                         "(max_flat_workgroup_size="
                      << Meta.MaxFlatWorkgroupSize
                      << " < target wavefront width=" << TargetIsa.waveSize()
                      << "); falling back to ReplicationProjection\n");
  }

  // Build opcode -> CanonicalOp map from MCInstrInfo
  OpcodeMap OpcMap;
  OpcMap.build(*Mc.InstrInfo);

  // ==== Phase 1: Disassemble + identify block boundaries ====
  //
  // The decode loop (and its two LLVM-drift guards) lives in decode.cpp so
  // this function stays focused on IR emission. decodeKernel returns a
  // linearised instruction stream + the set of CFG block-start offsets.
  if (KernelSize != 0 && KernelSize > UINT64_MAX - KernelOffset)
    return RaiseFailure::general(RaiseFailureReason::InternalError,
                                 "kernel decode extent overflows");

  const uint64_t KernelEndOffset =
      KernelSize == 0 ? 0 : KernelOffset + KernelSize;
  Expected<DecodeResult> DecodedOrErr = decodeKernel(
      Mc, OpcMap, ArrayRef<uint8_t>(TextBytes.data(), TextBytes.size()),
      KernelOffset, KernelEndOffset);
  if (!DecodedOrErr)
    return DecodedOrErr.takeError();
  DecodeResult Decoded = std::move(*DecodedOrErr);
  auto &Insts = Decoded.Insts;
  auto &BlockStarts = Decoded.BlockStarts;

  if (Stats)
    Stats->TotalCount = static_cast<int>(Insts.size());

  // The scalar-move path writes no EXEC.

  // ==== Phase 2: Build LLVM IR module + function ====
  Result.Module = std::make_unique<Module>("transpiler_module", C);
  Module &M = *Result.Module;
  M.setTargetTriple(Triple("amdgcn-amd-amdhsa"));

  TargetOptions Opts;
  std::unique_ptr<TargetMachine> Tm(Mc.Target->createTargetMachine(
      Triple("amdgcn-amd-amdhsa"),
      CompilationTargetIsa.empty() ? SourceIsa : CompilationTargetIsa, "", Opts,
      Reloc::PIC_));
  if (!Tm) {
    errs() << "transpiler: Failed to create TargetMachine\n";
    return RaiseFailure::general(
        RaiseFailureReason::TargetMachineCreationFailed,
        "could not create the target machine");
  }
  M.setDataLayout(Tm->createDataLayout());

  auto *VoidTy = Type::getVoidTy(C);
  auto *I1Ty = Type::getInt1Ty(C);
  auto *I8Ty = Type::getInt8Ty(C);

  // Function signature: a single opaque `ptr byref([N x i8]) align 16`
  // placeholder so the AMDGPU backend emits kernarg_segment_size = N and the
  // ABI's 16-byte kernarg alignment. Handlers do not read it; kernarg loads
  // lift to GEP+load against amdgcn_kernarg_segment_ptr. byref is required for
  // the align parameter attribute to reach the emitted kernel descriptor.
  SmallVector<Type *, 1> ParamTypes;
  KernargLayout Kernargs;
  int ParamIdx = 0;
  Type *KernargByrefTy = nullptr;
  if (Meta.KernargSegmentSize > 0) {
    KernargByrefTy =
        ArrayType::get(I8Ty, static_cast<uint64_t>(Meta.KernargSegmentSize));
    ParamTypes.push_back(PointerType::get(C, /*addrspace=*/4));
    ParamIdx = 1;
  }
  Kernargs.ImplicitArgsBase = Meta.implicitArgsBase();
  Kernargs.Args = Meta.Args;
  Kernargs.KernargSegmentSize = Meta.KernargSegmentSize;

  auto *FuncTy = FunctionType::get(VoidTy, ParamTypes, false);
  Function *F =
      Function::Create(FuncTy, GlobalValue::ExternalLinkage, KernelName, &M);
  F->setCallingConv(CallingConv::AMDGPU_KERNEL);

  // Attach byref([N x i8]) + align(16) to the placeholder kernarg pointer so
  // the emitted kernel descriptor gets the AMDGPU ABI's 16-byte kernarg
  // alignment without forcing an aggregate/vector parameter type.
  if (KernargByrefTy != nullptr) {
    F->addParamAttr(0, Attribute::getWithByRefType(C, KernargByrefTy));
    F->addParamAttr(0, Attribute::getWithAlignment(C, Align(16)));
  }
  // The kernel-entry v0 holds the packed workitem id, x[0:9] | y[10:19] |
  // z[20:29]. ENABLE_VGPR_WORKITEM_ID (COMPUTE_PGM_RSRC2 bits [12:11]) records
  // how many of x/y/z the source enabled: 0 -> X, 1 -> X+Y, 2 -> X+Y+Z. The
  // packed v0 seed below reconstructs exactly those fields; seeding only X left
  // every threadIdx.y / threadIdx.z read folding to 0.
  unsigned WorkitemIdCnt =
      (Meta.ComputePgmRsrc2 >>
       llvm::amdhsa::COMPUTE_PGM_RSRC2_ENABLE_VGPR_WORKITEM_ID_SHIFT) &
      ((1u << llvm::amdhsa::COMPUTE_PGM_RSRC2_ENABLE_VGPR_WORKITEM_ID_WIDTH) -
       1u);
  unsigned NumWorkitemDims = WorkitemIdCnt >= 2 ? 3u : WorkitemIdCnt + 1u;
  {
    // Pin the workgroup size to exactly what the source kernel declared, so
    // the backend lays out LDS / workitem IDs the same way the original
    // gfx1250 binary did.
    int MaxWg =
        Meta.MaxFlatWorkgroupSize > 0 ? Meta.MaxFlatWorkgroupSize : 1024;
    if (Projection.usesDoubledDispatch()) {
      // The runtime launches this block scaled by the doubled-dispatch factor
      // along x. `amdgpu-flat-work-group-size` must advertise the scaled size
      // or ROCR/HIP would reject the larger launch as exceeding the declared
      // bound; the in-kernel workgroup-size query is virtualized back to the
      // source size via source-hidden-args, so kernel logic still sees MaxWg.
      MaxWg *= static_cast<int>(Projection.doubledDispatchFactor());
      // IR-level breadcrumb recording the doubled dimension and factor (e.g.
      // "x2") for offline inspection and the raise_cli lit tests. This is not
      // the runtime signal: the launch runtime learns the doubled dim/factor
      // from the transpile result (RaiseResult -> comgr result info fields ->
      // loader), because this function attribute does not survive to the kernel
      // descriptor metadata.
      assert(Projection.doubledDispatchDim() < 3 &&
             "doubled dispatch dim must be x/y/z");
      const char DimChar = "xyz"[Projection.doubledDispatchDim()];
      F->addFnAttr("hotswap-modrep-doubled-dispatch",
                   std::string(1, DimChar) +
                       std::to_string(Projection.doubledDispatchFactor()));
    }
    F->addFnAttr("amdgpu-flat-work-group-size",
                 std::to_string(MaxWg) + "," + std::to_string(MaxWg));

    // Deliberately do NOT set "amdgpu-waves-per-eu".  Pinning occupancy
    // constrains register allocation and caused spurious VGPR spills for
    // wide kernels (e.g. the Triton 128x128 matmul on gfx942), which then
    // triggered memory faults because our raised IR is register-pressure
    // heavy compared to a from-source compile.  Letting the backend choose
    // occupancy freely keeps register pressure safe.
    // TODO(gfx1250->gfx942): revisit once the raiser emits tighter IR; we may
    // want to propagate the source kernel's waves-per-eu for parity.

    // The hotswap caller still launches with the source kernel's host-side
    // kernarg buffer.  Hotswap materialises every source-visible value either
    // as a normal formal parameter, as source-ABI preloaded SGPR state seeded
    // explicitly in IR below, or as an intrinsic for architected dispatch
    // state.  Suppress backend-invented implicit kernarg slots so the emitted
    // descriptor keeps the source kernarg size instead of appending a
    // target-default hidden-arg block that the host never populated.
    F->addFnAttr("amdgpu-no-cluster-id-x");
    F->addFnAttr("amdgpu-no-cluster-id-y");
    F->addFnAttr("amdgpu-no-cluster-id-z");
    F->addFnAttr("amdgpu-no-completion-action");
    F->addFnAttr("amdgpu-no-default-queue");
    F->addFnAttr("amdgpu-no-dispatch-id");
    // Do not suppress dispatch-ptr: source hidden-arg synthesis materialises
    // values such as hidden_group_size_* and hidden_block_count_* from the
    // target dispatch packet, because the lifted HSACO intentionally does not
    // ask HIP to append source-ABI hidden args after the opaque kargs blob.
    F->addFnAttr("amdgpu-no-heap-ptr");
    F->addFnAttr("amdgpu-no-hostcall-ptr");
    F->addFnAttr("amdgpu-no-implicitarg-ptr");
    F->addFnAttr("amdgpu-no-lds-kernel-id");
    F->addFnAttr("amdgpu-no-multigrid-sync-arg");
    F->addFnAttr("amdgpu-no-queue-ptr");
    F->addFnAttr("amdgpu-no-workitem-id-x");
    // Only suppress the Y/Z workitem-id fields the source did not enable. The
    // packed v0 seed uses workitem.id.{y,z} for 2-D/3-D blocks; a stale "no"
    // attribute would pin ENABLE_VGPR_WORKITEM_ID at 0 so the backend never
    // loads those fields and threadIdx.y/z would read garbage.
    if (NumWorkitemDims < 2)
      F->addFnAttr("amdgpu-no-workitem-id-y");
    if (NumWorkitemDims < 3)
      F->addFnAttr("amdgpu-no-workitem-id-z");
    F->addFnAttr("uniform-work-group-size", "true");
  }

  // Mirror the source's .group_segment_fixed_size onto the amdgpu-lds-size
  // attribute so the backend emits a matching group_segment_fixed_size. The
  // raiser writes LDS via inttoptr to addrspace(3) with no GlobalVariable, so
  // without this the backend would emit group_segment_fixed_size = 0 and treat
  // every LDS op as out-of-segment. The attribute takes "min,max"; the
  // source's static size is exact, so both are equal.
  if (Meta.GroupSegmentFixedSize > 0) {
    std::string SizeStr = std::to_string(Meta.GroupSegmentFixedSize);
    F->addFnAttr("amdgpu-lds-size", SizeStr + "," + SizeStr);
  }

  if (ParamIdx > 0)
    F->getArg(0)->setName("kargs");

  Function *FnWorkgroupIdX =
      Intrinsic::getOrInsertDeclaration(&M, Intrinsic::amdgcn_workgroup_id_x);
  Function *FnWorkgroupIdY =
      Intrinsic::getOrInsertDeclaration(&M, Intrinsic::amdgcn_workgroup_id_y);
  Function *FnDispatchPtr =
      Intrinsic::getOrInsertDeclaration(&M, Intrinsic::amdgcn_dispatch_ptr);
  Function *FnKargPtr = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::amdgcn_kernarg_segment_ptr);
  // Build the source-ISA user-SGPR ABI from the kernel descriptor.
  // Phase 4 seeding and handler-side ABI-sensitive decoding (e.g.
  // handle_smem's kernarg-pointer detection) both key off this layout.
  UserSgprLayout UserSgprLayout;
  if (llvm::Error LayoutErr = UserSgprLayout::tryFromKernelMeta(
          Meta, Isa, SourceIsa, UserSgprLayout))
    return std::move(LayoutErr);
  if (AMDGPU::isGFX12Plus(*Mc.SubtargetInfo) &&
      Meta.hasNonDisabledClusterDims()) {
    const std::array<uint32_t, 3> &Dims = *Meta.ClusterDims;
    return RaiseFailure::inKernel(
        RaiseFailureReason::UnsupportedSourceClusterDims, KernelName,
        ".cluster_dims=[" + Twine(Dims[0]) + "," + Twine(Dims[1]) + "," +
            Twine(Dims[2]) +
            "] requires real TTMP6 cluster workgroup state; the current "
            "HotSwap ABI model only supports disabled source clusters");
  }
  // ==== Phase 3: Create basic blocks ====
  // `blockStarts` is a std::set (see decode.h) so it iterates in
  // ascending source-address order, giving deterministic BB labels.
  // `offsetToBB` is a DenseMap and intentionally unordered; for the
  // thread-loop entry BB we need the lowest-address BB as InsertBefore
  // (so the entry sorts above the kernel body in IR), which we capture
  // explicitly during the create loop.
  llvm::DenseMap<uint64_t, BasicBlock *> OffsetToBb;
  BasicBlock *FirstBodyBb = nullptr;
  for (uint64_t Addr : BlockStarts) {
    BasicBlock *Bb =
        BasicBlock::Create(C, "bb_0x" + utohexstr(Addr - KernelOffset), F);
    OffsetToBb[Addr] = Bb;
    if (!FirstBodyBb)
      FirstBodyBb = Bb;
  }
  // The register-seeding block must be a predecessor-free entry block that
  // control-flows into the kernel's real start (KernelOffset). Normally the
  // KernelOffset block is the lowest-addressed block and can serve as the entry
  // directly. The thread-loop route instead wraps the body in a loop, so it
  // seeds into a dedicated "entry" block inserted before all body blocks and
  // branches it to KernelOffset.
  BasicBlock *EntryBb = UseThreadLoop
                            ? BasicBlock::Create(C, "entry", F, FirstBodyBb)
                            : OffsetToBb[KernelOffset];

  // ==== Phase 4: Init entry registers ====
  IRBuilder<> B(EntryBb);

  AllocaRegFile Regs;
  Regs.init(B, I32Ty, I1Ty, Isa, *Mc.RegInfo, Projection);

  // Seed kernel-entry SGPR state from the descriptor-derived user-SGPR ABI.
  //
  // Crucial invariant: never hardcode SGPR indices. Kernarg preload and
  // enable_sgpr_* toggles legally move the kernarg pointer and workgroup-id
  // SGPRs away from s[0:1]/s2/s3. Hardcoding those indices mis-seeds entry
  // state and turns real source values into undef reads on the JIT path.
  //
  // Seed ABI-provided entry pointers with the matching AMDGPU intrinsics. The
  // source descriptor's dispatch_ptr bit means the corresponding SGPR pair
  // holds the AQL dispatch packet base, and source SMEM may legally load
  // through it just like it loads through kernarg_segment_ptr.
  if (UserSgprLayout.dispatchPtrSgpr().has_value()) {
    Regs.storeSGPR64(B, *UserSgprLayout.dispatchPtrSgpr(),
                     B.CreateCall(FnDispatchPtr, {}, "dispatch_ptr"));
  }
  if (UserSgprLayout.kernargSegmentPtrSgpr().has_value()) {
    Regs.storeSGPR64(B, *UserSgprLayout.kernargSegmentPtrSgpr(),
                     B.CreateCall(FnKargPtr, {}, "kernarg_ptr"));
  }
  if (UserSgprLayout.workgroupIdXSgpr().has_value()) {
    Regs.storeSGPR32(B, *UserSgprLayout.workgroupIdXSgpr(),
                     B.CreateCall(FnWorkgroupIdX, {}, "wg_id_x"));
  }
  if (UserSgprLayout.workgroupIdYSgpr().has_value()) {
    Regs.storeSGPR32(B, *UserSgprLayout.workgroupIdYSgpr(),
                     B.CreateCall(FnWorkgroupIdY, {}, "wg_id_y"));
  }
  // Hidden-arg remaps use the ABI version the backend will emit for this
  // module. If target emission starts pinning a module flag, thread that value
  // here instead of relying on LLVM's default.
  unsigned TargetCodeObjectVersion =
      AMDGPU::getDefaultAMDHSACodeObjectVersion();
  auto EmitPreloadedKernargDword =
      [&](IRBuilder<> &SeedB, uint64_t ByteOffset) -> Expected<Value *> {
    SourceHiddenArgContext HiddenCtx{C,
                                     M,
                                     SeedB,
                                     *F,
                                     I8Ty,
                                     I32Ty,
                                     I64Ty,
                                     Meta.Args,
                                     AssumeHipGlobalOffsetZero,
                                     TargetCodeObjectVersion};
    if (Projection.usesDoubledDispatch())
      HiddenCtx.ScaledReplicationFactor = Projection.doubledDispatchFactor();

    // A null Value means ByteOffset is not a source hidden argument; fall back
    // to a plain kernarg load. An Error means it is a hidden argument with no
    // supported source-side synthesis.
    Expected<Value *> HiddenOrErr =
        emitSourceHiddenDword(HiddenCtx, ByteOffset);
    if (!HiddenOrErr)
      return HiddenOrErr.takeError();
    if (Value *Hidden = *HiddenOrErr)
      return Hidden;

    if (Kernargs.ImplicitArgsBase > 0 &&
        ByteOffset >= Kernargs.ImplicitArgsBase) {
      // Strict mode (a pipeline option that refuses implicit-arg preload
      // instead of falling back to an implicitarg_ptr load) is not plumbed
      // through this entry point yet; take the non-strict fallback.
      if (/*isStrictMode()=*/false) {
        return RaiseFailure::inKernel(
            RaiseFailureReason::UnsupportedSourceHiddenArg, KernelName,
            "preloaded implicit argument at byte offset " + Twine(ByteOffset));
      }

      Function *FnImplicitArgPtr = Intrinsic::getOrInsertDeclaration(
          &M, Intrinsic::amdgcn_implicitarg_ptr);
      Value *ImplPtr =
          SeedB.CreateCall(FnImplicitArgPtr, {}, "preload_implicitarg_ptr");
      int64_t ImplOffset = ByteOffset - Kernargs.ImplicitArgsBase;
      Value *Gep = ImplOffset == 0
                       ? ImplPtr
                       : SeedB.CreateInBoundsGEP(I8Ty, ImplPtr,
                                                 SeedB.getInt64(ImplOffset),
                                                 "preload_impl_gep");
      return SeedB.CreateAlignedLoad(I32Ty, Gep, Align(4), "preload_impl_dw");
    }

    Value *SegPtr = SeedB.CreateCall(FnKargPtr, {}, "preload_kernarg_ptr");
    Value *Gep = SeedB.CreateInBoundsGEP(
        I8Ty, SegPtr, SeedB.getInt64(ByteOffset), "preload_gep");
    return SeedB.CreateAlignedLoad(I32Ty, Gep, Align(4), "preload_dw");
  };
  // Kernarg preload SGPRs carry dwords copied by hardware from the kernarg
  // segment before kernel entry. Materialize the same dwords by loading
  // through `amdgcn_kernarg_segment_ptr` so the AMDGPU backend handles the
  // ABI lowering uniformly: the GEP+load lowers back to `s_load_b32` (or a
  // hardware-preload SGPR read on gfx12+) against the kernarg segment, with
  // identical bytes to what the source kernel saw at entry.
  //
  // Hidden block counts (Triton's hidden_block_count_* ABI) still need
  // dispatch-packet synthesis since their values aren't stored in the
  // kernarg segment at all. Unmatched implicit-range preload offsets are
  // handled by the same strict/permissive boundary as SMEM hidden-arg loads.
  for (size_t SgprIdx = 0; SgprIdx < UserSgprLayout.Entries.size(); ++SgprIdx) {
    const UserSgprLayout::Entry &Entry = UserSgprLayout.Entries[SgprIdx];
    if (Entry.SrcKind != UserSgprLayout::Source::PreloadedKernarg)
      continue;

    Expected<Value *> DwOrErr =
        EmitPreloadedKernargDword(B, Entry.KernargByteOffset);
    if (!DwOrErr)
      return DwOrErr.takeError();

    Value *Dw = *DwOrErr;
    Regs.storeSGPR32(B, static_cast<int>(SgprIdx), Dw);
  }
  // NumWorkitemDims (computed above) selects how many of x/y/z to fold into the
  // packed v0 seed.
  auto SeedWorkitemId = [&](IRBuilder<> &SeedB) {
    Regs.storeVGPR32(SeedB, 0,
                     Projection.emitPackedWorkitemId(SeedB, NumWorkitemDims));
  };

  if (!UseThreadLoop)
    SeedWorkitemId(B);

  // On gfx12+ the hardware command processor uses TTMP registers for
  // workgroup scheduling (RDNA4+ / CDNA-next layout):
  //   ttmp7[15:0]  = workgroup_id_y  (low 16 bits)
  //   ttmp7[31:16] = workgroup_id_z  (high 16 bits; 0 when grid has no Z)
  //   ttmp8[29:25] = wave_id within workgroup (subgroup ID)
  //   ttmp9        = workgroup_id_x  (accelerated launch)
  // A kernel raised without ttmp7 initialised reads workgroup_id_y/z as 0, so
  // seed ttmp7/8/9 to match the layout the AMDGPU backend expects on gfx12+.
  // gfx11 (RDNA3) passes these via SGPRs set up by the CP instead.
  std::function<void(IRBuilder<> &)> SeedTtmp8 = [](IRBuilder<> &) {};
  if (AMDGPU::isGFX12Plus(*Mc.SubtargetInfo)) {
    // TTMP6 carries the source workgroup-cluster fields on gfx12+. This
    // HotSwap path models non-cluster source execution, so use the singleton
    // cluster encoding: per-cluster workgroup IDs and max IDs are all zero.
    B.CreateStore(B.getInt32(0), Regs.Ttmp[6]);
    B.CreateStore(B.CreateCall(FnWorkgroupIdX, {}, "ttmp9_wg_id"),
                  Regs.Ttmp[9]);

    // ttmp7 = (workgroup_id_z << 16) | (workgroup_id_y & 0xFFFF). Mask Y to
    // 16 bits before OR-ing Z so a stray high bit in Y cannot bleed into the
    // Z field. This clips Y >= 65536 on no-Z grids, which no observed kernel
    // hits.
    Value *WgIdY = B.CreateCall(FnWorkgroupIdY, {}, "ttmp7_wg_id_y");
    Function *FnWorkgroupIdZ =
        Intrinsic::getOrInsertDeclaration(&M, Intrinsic::amdgcn_workgroup_id_z);
    Value *WgIdZ = B.CreateCall(FnWorkgroupIdZ, {}, "ttmp7_wg_id_z");
    Value *WgIdYLo = B.CreateAnd(WgIdY, B.getInt32(0xFFFF), "wg_id_y_lo16");
    Value *WgIdZHi = B.CreateShl(WgIdZ, B.getInt32(16), "wg_id_z_hi16");
    Value *Ttmp7Val = B.CreateOr(WgIdYLo, WgIdZHi, "ttmp7_val");
    B.CreateStore(Ttmp7Val, Regs.Ttmp[7]);

    SeedTtmp8 = [&](IRBuilder<> &SeedB) {
      // wave_id = workitem_id_x / wavefront_size (32 for gfx12)
      Value *TidForTtmp = Projection.emitWorkitemIdX(SeedB);
      TidForTtmp->setName("ttmp8_tid");
      Value *WaveId =
          SeedB.CreateLShr(TidForTtmp, SeedB.getInt32(5), "wave_id_in_wg");
      Value *Ttmp8Val =
          SeedB.CreateShl(WaveId, SeedB.getInt32(25), "ttmp8_val");
      SeedB.CreateStore(Ttmp8Val, Regs.Ttmp[8]);
    };
    if (!UseThreadLoop)
      SeedTtmp8(B);
  }

  auto SeedThreadLoopIterationState = [&](IRBuilder<> &SeedB) -> Error {
    for (auto *Slot : Regs.Sgpr)
      SeedB.CreateStore(ConstantInt::get(I32Ty, 0), Slot);
    for (auto *Slot : Regs.Vgpr)
      SeedB.CreateStore(ConstantInt::get(I32Ty, 0), Slot);
    for (auto *Slot : Regs.Agpr)
      SeedB.CreateStore(ConstantInt::get(I32Ty, 0), Slot);
    for (auto *Slot : Regs.Ttmp)
      SeedB.CreateStore(ConstantInt::get(I32Ty, 0), Slot);
    SeedB.CreateStore(ConstantInt::get(I32Ty, 0), Regs.M0);
    SeedB.CreateStore(ConstantInt::get(I32Ty, 0), Regs.FlatScr[0]);
    SeedB.CreateStore(ConstantInt::get(I32Ty, 0), Regs.FlatScr[1]);

    // Mirror the entry-BB user-SGPR seeding above so the thread-loop body sees
    // the same source ABI state as a normal source wave.
    if (UserSgprLayout.dispatchPtrSgpr().has_value()) {
      Regs.storeSGPR64(SeedB, *UserSgprLayout.dispatchPtrSgpr(),
                       SeedB.CreateCall(FnDispatchPtr, {}, "dispatch_ptr"));
    }
    if (UserSgprLayout.kernargSegmentPtrSgpr().has_value()) {
      Regs.storeSGPR64(SeedB, *UserSgprLayout.kernargSegmentPtrSgpr(),
                       SeedB.CreateCall(FnKargPtr, {}, "kernarg_ptr"));
    }
    if (UserSgprLayout.workgroupIdXSgpr().has_value()) {
      Regs.storeSGPR32(SeedB, *UserSgprLayout.workgroupIdXSgpr(),
                       SeedB.CreateCall(FnWorkgroupIdX, {}, "wg_id_x"));
    }
    if (UserSgprLayout.workgroupIdYSgpr().has_value()) {
      Regs.storeSGPR32(SeedB, *UserSgprLayout.workgroupIdYSgpr(),
                       SeedB.CreateCall(FnWorkgroupIdY, {}, "wg_id_y"));
    }
    for (size_t SgprIdx = 0; SgprIdx < UserSgprLayout.Entries.size();
         ++SgprIdx) {
      const UserSgprLayout::Entry &Entry = UserSgprLayout.Entries[SgprIdx];
      if (Entry.SrcKind != UserSgprLayout::Source::PreloadedKernarg)
        continue;
      Expected<Value *> DwOrErr =
          EmitPreloadedKernargDword(SeedB, Entry.KernargByteOffset);
      if (!DwOrErr)
        return DwOrErr.takeError();

      Regs.storeSGPR32(SeedB, static_cast<int>(SgprIdx), *DwOrErr);
    }

    if (AMDGPU::isGFX12Plus(*Mc.SubtargetInfo)) {
      SeedB.CreateStore(SeedB.CreateCall(FnWorkgroupIdX, {}, "ttmp9_wg_id"),
                        Regs.Ttmp[9]);
      Value *WgIdY = SeedB.CreateCall(FnWorkgroupIdY, {}, "ttmp7_wg_id_y");
      Function *FnWorkgroupIdZ = Intrinsic::getOrInsertDeclaration(
          &M, Intrinsic::amdgcn_workgroup_id_z);
      Value *WgIdZ = SeedB.CreateCall(FnWorkgroupIdZ, {}, "ttmp7_wg_id_z");
      Value *WgIdYLo =
          SeedB.CreateAnd(WgIdY, SeedB.getInt32(0xFFFF), "wg_id_y_lo16");
      Value *WgIdZHi =
          SeedB.CreateShl(WgIdZ, SeedB.getInt32(16), "wg_id_z_hi16");
      Value *Ttmp7Val = SeedB.CreateOr(WgIdYLo, WgIdZHi, "ttmp7_val");
      SeedB.CreateStore(Ttmp7Val, Regs.Ttmp[7]);
      SeedTtmp8(SeedB);
    }

    SeedWorkitemId(SeedB);
    Regs.storeVCC(SeedB, ConstantInt::getFalse(I1Ty));
    Regs.storeSCC(SeedB, ConstantInt::getFalse(I1Ty));
    Regs.storeExec(SeedB, Projection.emitInitialExec(SeedB));
    return Error::success();
  };

  // ==== Phase 5: Raise each instruction; collect all failures in allFailures.
  // ====

  // `userSgprLayout` was built above before Phase 4 so entry SGPR seeding
  // and handler-side ABI decisions use the same descriptor-derived mapping.
  RaiseContext Ctx{B,
                   Regs,
                   Projection,
                   Mc,
                   TargetCodeObjectVersion,
                   Kernargs,
                   UserSgprLayout,
                   nullptr,
                   OffsetToBb,
                   ArrayRef<uint8_t>(TextBytes.data(), TextBytes.size()),
                   TextBaseAddress,
                   SourceImageSections,
                   KernelOffset,
                   KernelEndOffset};
  Ctx.SourcePrivateSegmentFixedSize = Meta.PrivateSegmentFixedSize;
  Ctx.SourceComputePgmRsrc2 = Meta.ComputePgmRsrc2;
  Ctx.SourceKernelCodeProperties = Meta.KernelCodeProperties;
  Ctx.AssumeHipGlobalOffsetZero = AssumeHipGlobalOffsetZero;

  if (UseThreadLoop) {
    auto *IterA = B.CreateAlloca(I32Ty, nullptr, "tl_iter_alloca");
    B.CreateStore(B.getInt32(0), IterA);
    static_cast<ThreadLoopProjection *>(ProjectionPtr.get())
        ->setIterationAlloca(IterA);

    BasicBlock *CondBb = BasicBlock::Create(C, "tl_cond", F);
    BasicBlock *LatchBb = BasicBlock::Create(C, "tl_latch", F);
    BasicBlock *DoneBb = BasicBlock::Create(C, "tl_done", F);
    Ctx.ThreadLoopLatch = LatchBb;

    B.CreateBr(CondBb);
    B.SetInsertPoint(CondBb);

    Value *Iter = B.CreateLoad(I32Ty, IterA, "tl_iter_val");
    Value *IterOk = B.CreateICmpULT(
        Iter, B.getInt32(TargetIsa.waveSize() / Isa.waveSize()), "tl_iter_ok");
    Value *Lane = Projection.emitLaneIdx(B);
    Value *LaneOk =
        B.CreateICmpULT(Lane, B.getInt32(Isa.waveSize()), "tl_lane_ok");
    Value *EnterBody = B.CreateAnd(IterOk, LaneOk, "tl_enter_body");

    if (Error Err = SeedThreadLoopIterationState(B))
      return Err;

    Ctx.invalidateSgprShadows();

    B.CreateCondBr(EnterBody, OffsetToBb[KernelOffset], LatchBb);

    B.SetInsertPoint(LatchBb);
    Value *OldIter = B.CreateLoad(I32Ty, IterA, "tl_iter_old");
    Value *NextIter = B.CreateAdd(OldIter, B.getInt32(1), "tl_iter_next");
    B.CreateStore(NextIter, IterA);
    Value *More = B.CreateICmpULT(
        NextIter, B.getInt32(TargetIsa.waveSize() / Isa.waveSize()), "tl_more");
    B.CreateCondBr(More, CondBb, DoneBb);

    B.SetInsertPoint(DoneBb);
    B.CreateRetVoid();
  }

  llvm::Error RaiseFailures = llvm::Error::success();
  int RaisedCount = 0;
  for (size_t InstIdx = 0; InstIdx < Insts.size(); ++InstIdx) {
    const DecodedInst &Di = Insts[InstIdx];

    // If a terminator ended the recovered CFG path and the next decoded
    // instruction is not a known block leader, that instruction is unreachable
    // fallthrough bytes (often code after an unconditional branch). Do not emit
    // it into the already-terminated LLVM block.
    auto BbIt = OffsetToBb.find(Di.Offset);
    if (B.GetInsertBlock()->hasTerminator() && BbIt == OffsetToBb.end())
      continue;

    // Source-BB boundary handling uses `B.GetInsertBlock()` rather than a
    // tracked `currentBB` so that intra-handler CFG splits (emitUnderExec
    // diamonds under SPE) propagate correctly: fall-through must leave
    // from whatever block the builder is currently at -- which is the
    // `spe_skip` tail when the last emission was wrapped -- not from the
    // block that started the source instruction.
    if (BbIt != OffsetToBb.end() && BbIt->second != B.GetInsertBlock()) {
      BasicBlock *InsertBb = B.GetInsertBlock();
      if (!InsertBb->hasTerminator())
        B.CreateBr(BbIt->second);
      B.SetInsertPoint(BbIt->second);
      // LLVM's AMDGPULowerVGPREncoding pass resets VGPR MSB mode at every
      // basic-block boundary (both before terminators and at BB fall-through
      // exits).  Mirror that behaviour so we do not inherit stale MSB state
      // from a previous linear instruction that does not control-flow into
      // this BB.
      Ctx.VgprMsBs = 0;
      // Drop the per-lane-i1 shadow at every BB transition: the cached i1
      // SSA values dominate only the BB they were emitted in.
      Ctx.clearSgprWaveMaskShadow();
      // M0's raise-time constant shadow only dominates within its BB.
      Ctx.clearM0Const();
    }

    Ctx.computeVGPRAdjust(Di);
    // Invalidate the lane_active memo at every instruction boundary: any
    // instruction may write EXEC, and a stale lane_active would mispredicate
    // side effects.
    Ctx.resetLaneActiveCache();
    OpResolver Op{Ctx, Di};

    llvm::Expected<HandlerResult> HrOrErr =
        [&]() -> llvm::Expected<HandlerResult> {
      // Only the SOPP and SOP1 arms are wired; any other opcode falls through
      // to the unsupported-instruction path below and refuses cleanly.
      const MCInstrDesc &DispatchDesc =
          Mc.InstrInfo->get(Di.Inst.getOpcode());

      if (SIInstrFlags::isSOPP(DispatchDesc))
        return handleSOPP(Ctx, Di, Op);
      if (SIInstrFlags::isSOP1(DispatchDesc))
        return handleSOP1(Ctx, Di, Op);

      StringRef Format = formatName(Di.TargetSpecificFlags);
      return RaiseFailure::atInstruction(
          RaiseFailureReason::UnsupportedInstructionForm,
          strippedMnemonic(Mc, Di.Inst), Di.Offset, Format);
    }();

    if (!HrOrErr) {
      if (RaiseFailures && !HrOrErr) {
        RaiseFailures =
            llvm::joinErrors(std::move(RaiseFailures), HrOrErr.takeError());
      } else if (!HrOrErr) {
        RaiseFailures = HrOrErr.takeError();
      }
      continue;
    }

    HandlerResult Hr = *HrOrErr;

    // A handler recognised the instruction but refused it.
    if (!Hr.Handled) {
      StringRef Format = formatName(Di.TargetSpecificFlags);
      LLVM_DEBUG(dbgs() << "transpiler: unsupported instruction: "
                        << printInst(Mc, Di.Inst) << " [format=" << Format
                        << "] at offset 0x" << format_hex(Di.Offset, 1)
                        << "\n");
      RaiseFailures = llvm::joinErrors(
          std::move(RaiseFailures),
          RaiseFailure::atInstruction(RaiseFailureReason::UnsupportedOpcode,
                                      strippedMnemonic(Mc, Di.Inst),
                                      Di.Offset, Format));
      continue;
    }

    if (Di.defsScc() && !Hr.SccHandled && Hr.SccResult) {
      Value *Zero = Constant::getNullValue(Hr.SccResult->getType());
      Ctx.Regs.storeSCC(Ctx.B, Ctx.B.CreateICmpNE(Hr.SccResult, Zero));
    }
    if (Di.defsExec())
      Result.HasDivergentExec = true;

    RaisedCount++;
    continue;
  }

  // If the function's entry block has predecessors (e.g. a backward
  // branch targeting the kernel's first instruction), LLVM's verifier
  // rejects the IR.  Insert an empty prolog block that falls through to
  // the original entry so the entry becomes predecessor-free.
  if (!pred_empty(&F->getEntryBlock())) {
    BasicBlock *OldEntry = &F->getEntryBlock();
    BasicBlock *Prolog = BasicBlock::Create(C, "prolog", F, OldEntry);
    B.SetInsertPoint(Prolog);
    B.CreateBr(OldEntry);
  }

  // Ensure all BBs have terminators.  Reachable unterminated blocks arise
  // when a kernel falls off its symbol boundary without an explicit
  // s_endpgm -- emit `ret void` (or branch to the thread-loop latch)
  // so the lifted kernel terminates cleanly.  Blocks with no predecessors
  // that are not the entry block are dead fallthrough bytes after a
  // recovered branch; keep their defensive `unreachable`.
  for (auto &BB : *F) {
    if (!BB.hasTerminator()) {
      B.SetInsertPoint(&BB);
      if (!pred_empty(&BB) || &BB == &F->getEntryBlock()) {
        if (Ctx.ThreadLoopLatch)
          B.CreateBr(Ctx.ThreadLoopLatch);
        else
          B.CreateRetVoid();
      } else {
        B.CreateUnreachable();
      }
    }
  }

  if (Stats)
    Stats->LiftedCount = RaisedCount;

  // If any instructions failed to raise, skip Phases 6-7.
  if (RaiseFailures) {
    return RaiseFailures;
  }

  // ==== Phase 6: Promote allocas to SSA ====
  {
    DominatorTree DT(*F);
    AssumptionCache AC(*F);
    SmallVector<AllocaInst *, 512> Allocas;
    Regs.collectAllocas(Allocas);
    Ctx.collectSgprShadowAllocas(Allocas);
    PromoteMemToReg(Allocas, DT, &AC);
  }

  // ==== Phase 7: Verify IR ====
  std::string VerifyErr;
  raw_string_ostream VerifyOs(VerifyErr);
  if (verifyModule(M, &VerifyOs)) {
    errs() << "transpiler: IR verification failed:\n" << VerifyErr << "\n";
    return RaiseFailure::general(RaiseFailureReason::IRVerificationFailed,
                                 VerifyErr);
  }

  Result.UsesScratchPrivateSegment = Ctx.UsesScratchPrivateSegment;
  Result.SourcePrivateSegmentFixedSize = Ctx.SourcePrivateSegmentFixedSize;
  return Result;
}

llvm::Expected<RaiseResult>
raiseToIR(llvm::ArrayRef<uint8_t> TextBytes, llvm::StringRef SourceIsa,
          llvm::StringRef KernelName, const KernelMeta &Meta,
          llvm::StringRef CompilationTargetIsa, bool EnableWritelaneRewrite,
          bool EnableWaveNative, uint64_t TextBaseAddress,
          llvm::ArrayRef<TextSection::ImageSection> SourceImageSections,
          RaiseStats *Stats) {
  return raiseToIR(TextBytes, SourceIsa, KernelName, Meta,
                   /*KernelOffset=*/0,
                   /*KernelSize=*/0, CompilationTargetIsa,
                   EnableWritelaneRewrite, EnableWaveNative,
                   /*AssumeHipGlobalOffsetZero=*/false,
                   /*ForceReplicationDoubled=*/false, TextBaseAddress,
                   SourceImageSections, Stats);
}

llvm::Expected<RaiseResult>
raiseToIR(llvm::ArrayRef<uint8_t> TextBytes, llvm::StringRef SourceIsa,
          llvm::StringRef KernelName, const KernelMeta &Meta,
          uint64_t KernelOffset, uint64_t KernelSize,
          llvm::StringRef CompilationTargetIsa, bool EnableWritelaneRewrite,
          bool EnableWaveNative, bool AssumeHipGlobalOffsetZero,
          bool ForceReplicationDoubled, uint64_t TextBaseAddress,
          llvm::ArrayRef<TextSection::ImageSection> SourceImageSections,
          RaiseStats *Stats) {
  return raiseToIRImpl(TextBytes, SourceIsa, KernelName, Meta, KernelOffset,
                       KernelSize, TextBaseAddress, SourceImageSections,
                       CompilationTargetIsa, EnableWritelaneRewrite,
                       EnableWaveNative,
                       /*forceThreadLoopProjection=*/false,
                       /*suppressC5ForThreadLoopRoute=*/false,
                       ForceReplicationDoubled, AssumeHipGlobalOffsetZero,
                       Stats);
}

} // namespace COMGR::hotswap
