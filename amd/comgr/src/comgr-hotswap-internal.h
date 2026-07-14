//===- comgr-hotswap-internal.h - HotSwap internal types and declarations -===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Internal header for the HotSwap ISA rewriting subsystem. Shared by all
/// comgr-hotswap-*.cpp compilation units. Not part of the public COMGR API.
///
/// Module structure:
///   comgr-hotswap-elf.cpp       ELF parsing, binary helpers, trampoline growth
///   comgr-hotswap-llvm.cpp      LLVM MC infrastructure (disasm/asm/encode)
///   comgr-hotswap-b0a0.cpp      GFX1250 B0-to-A0 policy + public API
///
//===----------------------------------------------------------------------===//

#ifndef COMGR_HOTSWAP_INTERNAL_H
#define COMGR_HOTSWAP_INTERNAL_H

#include "amd_comgr.h"
#include "comgr-env.h"
#include "comgr.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Object/ELF.h"
#include "llvm/Object/ELFTypes.h"
#include "llvm/Support/AMDHSAKernelDescriptor.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

namespace COMGR {
namespace hotswap {

inline constexpr int8_t VgprMsbUnanalyzed = -3;
inline constexpr int8_t VgprMsbUnreachable = -2;
inline constexpr int8_t VgprMsbUnknown = -1;

// -- Logging ------------------------------------------------------------------
//
// Single output stream for all hotswap diagnostics (errors, warnings, and
// verbose traces). Returns llvm::errs() if AMD_COMGR_EMIT_VERBOSE_LOGS is set
// (via COMGR::env::shouldEmitVerboseLogs()) and llvm::nulls() otherwise, so
// hotswap output stays quiet in normal use but callers can opt in to the full
// diagnostic trail without relinking. Every function that returns a null /
// empty / failure result should emit here with a `"hotswap: error: ..."` or
// `"hotswap: ..."` prefix so the failure path is traceable.

inline llvm::raw_ostream &log() {
  return COMGR::env::shouldEmitVerboseLogs() ? llvm::errs() : llvm::nulls();
}

inline std::optional<uint64_t> checkedAddUint64(uint64_t LHS, uint64_t RHS,
                                                llvm::StringRef Context) {
  std::optional<uint64_t> Result = llvm::checkedAddUnsigned(LHS, RHS);
  if (Result)
    return Result;

  log() << "hotswap: error: " << Context << " overflows uint64_t.\n";
  return std::nullopt;
}

inline std::optional<uint64_t> checkedSubUint64(uint64_t LHS, uint64_t RHS,
                                                llvm::StringRef Context) {
  if (LHS < RHS) {
    log() << "hotswap: error: " << Context << " underflows uint64_t.\n";
    return std::nullopt;
  }
  return LHS - RHS;
}

// -- Trampoline and NOP sled --------------------------------------------------

struct Trampoline {
  uint64_t OriginalOffset = 0;
  uint32_t OriginalSize = 0;
  llvm::SmallVector<uint8_t> Bytes;
  // When set, the pool is beyond s_branch reach. The source site branches to a
  // nearby NOP gateway, which uses the scratch-backed pre-Gen5 set-PC sequence
  // to reach the pool without executing s_add_pc_i64.
  bool Long = false;
  bool UsesSetPCBack = false;
  unsigned LongBranchSgprBase = 0;
  // The scratch pair was proven dead only at the original resume point.
  // Backward source growth is safe only across instructions that do not touch
  // the pair; forward growth must re-prove it at the moved resume point.
  bool LongBranchScratchIsSiteProven = false;
  bool HasPoolBranchIsland = false;
  uint64_t PoolBranchIslandOffset = 0;
  bool UsesShortBranchForward = false;
  bool UsesDirectSetPCForward = false;
  llvm::SmallVector<uint8_t> DirectSetPCForwardBytes;
  llvm::SmallVector<uint64_t, 4> ForwardBranchIslands;
  uint64_t ForwardBranchTargetOffset = 0;
  bool HasForwardGateway = false;
  uint64_t ForwardGatewayOffset = 0;
  llvm::SmallVector<uint8_t> ForwardGatewayBytes;
  // A far-site run may only be coalesced within one known function. Unknown
  // ranges stay unmerged because adjacent symbols are independent entries.
  bool HasFunctionRange = false;
  uint64_t FunctionStart = 0;
  uint64_t FunctionEnd = 0;
};

// Kernel-entry stubs are appended as normal .text growth. Keep each entry on
// the same 256-byte alignment expected by AMDGPU kernel descriptors.
static constexpr uint64_t KernelEntryStubStride = 256;
static constexpr uint64_t KernelEntryInstPrefUnitBytes = 128;
static_assert(KernelEntryStubStride % KernelEntryInstPrefUnitBytes == 0,
              "entry-stub stride must be an integral prefetch span");
static constexpr uint32_t KernelEntryStubInstPrefLines =
    KernelEntryStubStride / KernelEntryInstPrefUnitBytes;

struct KernelDescriptorInfo {
  std::string KernelName;
  uint64_t VAddr = 0;
  uint64_t FileOffset = 0;
  int64_t EntryOffset = 0;
};

struct KernelClusterDims {
  unsigned X = 0;
  unsigned Y = 0;
  unsigned Z = 0;
};

enum class NopSledUse {
  OwnerBody,
  RelocationBody,
  Gateway,
};

enum class ReplacementPlacement {
  Default,
  Ds2SourceTail,
  ProtectedCombinedDelay,
};

struct NopSled {
  uint64_t Start = 0;
  uint64_t End = 0;
  uint64_t WritePos = 0;
  uint64_t OwnerStart = 0;
  uint64_t OwnerEnd = 0;
  // This storage may contain branch gateways only, never a replacement body.
  // It covers certified external padding and appended-pool branch islands.
  bool GatewayOnly = false;
  // Explicit NOP padding immediately after a proven no-fallthrough function
  // boundary remains body-owned by that function, but any function may use
  // the same allocation cursor for branch-only gateway instructions.
  bool GlobalGateway = false;
  // Strictly proven post-function NOP padding may also hold straight-line,
  // PC-independent relocation bodies when the patch pass explicitly opts in.
  bool GlobalBody = false;

  bool ownsSource(uint64_t Offset) const {
    return OwnerStart <= Offset && Offset < OwnerEnd;
  }

  bool canGatewayFrom(uint64_t Offset) const {
    return GlobalGateway || ownsSource(Offset);
  }

  bool canHoldBodyFrom(uint64_t Offset, bool AllowGlobalBody) const {
    return ownsSource(Offset) || (AllowGlobalBody && GlobalBody);
  }
};

enum class MaskWorkaroundPolicy {
  None,
  A0,
  B0,
};

/// Target dependence of an executable pool emitted by HotSwap. Neutral pools
/// contain only stepping-independent code such as kernel-entry stubs.
enum class ExecutablePoolTargetState : uint32_t {
  Neutral = 0,
  A0 = 1,
  B0 = 2,
};

// -- Rewrite rule -------------------------------------------------------------

struct RewriteRule {
  std::string ReplaceMnemonic;
  llvm::SmallVector<uint8_t> ReplaceBytes;
};

// -- Named constants ----------------------------------------------------------

// Kernel descriptor size from upstream AMDHSAKernelDescriptor.h. Field
// offsets are resolved via offsetof(amdhsa::kernel_descriptor_t, field)
// at the access site so the struct definition stays the single source
// of truth and the *_OFFSET constants do not get spelled out twice.
static constexpr uint64_t KdSize = sizeof(llvm::amdhsa::kernel_descriptor_t);

// Maximum distance (bytes) between an instruction and a NOP sled for the
// sled to be considered reachable by a single s_branch.
static constexpr uint64_t MaxSledDistance = 131072;

// Minimum size (bytes) of a consecutive NOP run to be usable as a sled.
static constexpr uint64_t MinNopSledSize = 8;

// Minimum AMDGPU instruction size (one dword).
static constexpr uint32_t MinInstSize = 4;

// Fixed reservation for the SCC-neutral set-PC sequence. The gfx1250
// s_add_nc_u64 form is 20 bytes for a same-object forward or backward delta.
static constexpr uint32_t SetPcReturnReserveBytes = 20;
static constexpr uint32_t SetPcForwardSequenceBytes = 20;
static constexpr uint32_t PoolBranchIslandBytes = MinInstSize;

// s_branch encoding: 16-bit signed dword offset field bounds. Used by
// LLVMState::encodeSBranch to reject out-of-range branches before handing
// them to MCCodeEmitter.
static constexpr int64_t BranchOffsetMin = -32768;
static constexpr int64_t BranchOffsetMax = 32767;

// MCInst operand layout for ds_load_addtid_b32 / ds_store_addtid_b32. Shared
// between the trampoline patch (comgr-hotswap-patch-trampoline.cpp) and the
// unit tests that pin the layout (HotswapMCTest.cpp) so a tablegen change
// upstream is caught in one place.
//   operand 0: vdst (load) / data0 (store) -- VGPR register
//   operand 1: combined offset             -- immediate
//   operand 2: gds                         -- immediate (0 = LDS, 1 = GDS)
static constexpr unsigned AddtidOpReg = 0;
static constexpr unsigned AddtidOpOffset = 1;
static constexpr unsigned AddtidOpGds = 2;

// -- ElfView ------------------------------------------------------------------
//
// Thin wrapper around llvm::object::ELFFile<ELF64LE> that owns the structural
// view of a mutable code-object buffer. The caller retains ownership of the
// bytes; ElfView exposes LLVM's ELF iterators through member methods and
// caches the .text section lookup.

class ElfView {
public:
  using ELFT = llvm::object::ELF64LE;
  using ELFFileT = llvm::object::ELFFile<ELFT>;

  struct FunctionTextRange {
    uint64_t Begin = 0;
    uint64_t End = 0;
    const ELFT::Sym *Symbol = nullptr;
    const ELFT::Shdr *Symtab = nullptr;
  };

  struct TextOffsetRange {
    uint64_t Begin = 0;
    uint64_t End = 0;
  };

  /// Parse \p Data / \p Size into an ElfView. Fails if the bytes are not a
  /// valid ELF64 or if no `.text` section is found.
  static llvm::Expected<ElfView> create(uint8_t *Data, size_t Size);

  ElfView(ElfView &&) = default;
  ElfView &operator=(ElfView &&) = default;
  ElfView(const ElfView &) = delete;
  ElfView &operator=(const ElfView &) = delete;

  const ELFFileT &file() const { return File; }
  size_t size() const { return File.getBufSize(); }

  /// Writable view of the underlying bytes. The caller that constructed this
  /// ElfView via `create(uint8_t *, size_t)` retains ownership of the buffer;
  /// ElfView just exposes a typed, mutable alias onto `ELFFile::base()`. Safe
  /// because the factory was handed a `uint8_t *` and the buffer outlives
  /// this ElfView.
  uint8_t *data() { return const_cast<uint8_t *>(File.base()); }
  const uint8_t *data() const { return File.base(); }

  /// Section header range, cached at construction time. The underlying
  /// storage is the file buffer, which lives at least as long as this
  /// ElfView, so the range is always valid to iterate.
  ELFT::ShdrRange sections() const { return Sections; }

  /// Return the cached `.text` section header. Never null for a successfully
  /// constructed ElfView.
  const ELFT::Shdr *textSection() const { return TextSection; }

  uint64_t textOffset() const { return TextSection->sh_offset; }
  uint64_t textSize() const { return TextSection->sh_size; }
  uint64_t textAddr() const { return TextSection->sh_addr; }

  /// Index of the `.text` section in the section header table.
  unsigned textSectionIndex() const { return TextSectionIndex; }

  /// Pointer into the buffer for the first byte of `.text`.
  uint8_t *textData() { return data() + textOffset(); }
  const uint8_t *textData() const { return data() + textOffset(); }

  /// Enumerate function symbol ranges in `.text` using virtual addresses.
  /// Zero-size symbols extend to the next function symbol or `.text` end.
  std::vector<FunctionTextRange> functionTextRanges() const;

  /// Return every defined symbol offset in `.text`, irrespective of symbol
  /// type. A missing value means symbol-table discovery was incomplete.
  std::optional<std::vector<uint64_t>> textSymbolOffsets() const;

  /// Return sorted, coalesced half-open extents for sized, non-callable
  /// symbols defined in `.text`. Extents are clipped to `.text`; a missing
  /// value means discovery was incomplete or an extent overflowed.
  std::optional<std::vector<TextOffsetRange>> textSymbolExtents() const;

  /// Find the kernel function symbol whose range includes \p TextAddress.
  /// Returns "" if no matching function symbol exists.
  std::string findKernelAtAddress(uint64_t TextAddress) const;

  /// Find the section-relative `.text` range of the function containing
  /// \p TextOffset, or std::nullopt if no sized function symbol covers it.
  std::optional<FunctionTextRange>
  findFunctionTextRangeAtOffset(uint64_t TextOffset) const;

  /// Return a pointer to \p Len bytes at virtual address \p VAddr, resolved
  /// through the allocatable section that contains it (any section, not just
  /// `.text` -- e.g. the appended trampoline pool). Returns nullptr if no
  /// section covers the range or it falls outside the buffer.
  const uint8_t *dataAtVAddr(uint64_t VAddr, uint64_t Len) const;

  /// Return true when the complete virtual-address range is backed by bytes
  /// in an executable PT_LOAD segment. Segment validation is still required
  /// even when the range also has an executable section header.
  bool isExecutableVAddrRange(uint64_t VAddr, uint64_t Len) const;

  /// Return true when every file-backed executable region outside the `.text`
  /// section analyzed by the instruction rewriter has valid HotSwap provenance
  /// and is either stepping-neutral or built for \p TargetState. Unmarked,
  /// stale, conflicting, or malformed provenance fails closed; structurally
  /// malformed ELF data returns std::nullopt.
  std::optional<bool> executableCodeOutsideTextIsCompatibleWith(
      ExecutablePoolTargetState TargetState) const;

  /// Pointer to the kernel_descriptor for \p KernelName inside the buffer,
  /// or nullptr if not found.
  uint8_t *findKernelDescriptor(llvm::StringRef KernelName);

  /// Enumerate kernel descriptor symbols named "<kernel>.kd" and read their
  /// current kernel_code_entry_byte_offset values. The returned range remains
  /// valid until this ElfView is destroyed.
  llvm::ArrayRef<KernelDescriptorInfo> kernelDescriptors() const;

  /// True only when every descriptor symbol was parsed and duplicate names
  /// resolved to the same descriptor location.
  bool kernelDescriptorCacheIsComplete() const;

  /// Return the virtual address of the kernel descriptor symbol for
  /// \p KernelName, or std::nullopt when the descriptor is not present.
  std::optional<uint64_t>
  getKernelDescriptorVAddr(llvm::StringRef KernelName) const;

  /// Rewrite kernel_code_entry_byte_offset for \p KernelName.
  bool updateKernelDescriptorEntryOffset(llvm::StringRef KernelName,
                                         int64_t NewEntryOffset);

  /// Ensure kernel metadata reserves at least \p RequiredSgprs SGPRs. When
  /// \p UpdateDescriptor is true, also update the pre-gfx10 kernel descriptor
  /// field; it is reserved and must remain unchanged on gfx10+.
  bool updateKernelDescriptorSgprCount(llvm::StringRef KernelName,
                                       unsigned RequiredSgprs,
                                       bool UpdateDescriptor = true);

  /// Update metadata SGPR counts for every named kernel in one parse and
  /// serialization pass. All requested kernels must be present.
  bool updateKernelMetadataSgprCounts(
      const llvm::StringMap<unsigned> &RequiredSgprs);

  /// Retag every gfx1250 kernel in the AMDGPU metadata note with \p Revision.
  /// The revision strings used by gfx1250 ("A0" and "B0") have equal encoded
  /// size, so this preserves the ELF layout.
  bool updateGfx1250RevisionMetadata(llvm::StringRef Revision);

  /// Return true only when metadata contains at least one kernel and every
  /// kernel explicitly records \p Revision. Missing revision keys return
  /// false; malformed metadata returns std::nullopt.
  std::optional<bool>
  allKernelsHaveGfx1250Revision(llvm::StringRef Revision) const;

  /// Read COMPUTE_PGM_RSRC3.INST_PREF_SIZE for \p KernelName.
  std::optional<uint32_t>
  getKernelDescriptorInstPrefSize(llvm::StringRef KernelName,
                                  llvm::StringRef TargetCpu) const;

  /// Rewrite COMPUTE_PGM_RSRC3.INST_PREF_SIZE for \p KernelName.
  bool updateKernelDescriptorInstPrefSize(llvm::StringRef KernelName,
                                          llvm::StringRef TargetCpu,
                                          uint32_t InstPrefLines);

  /// Read the VGPR count from the kernel descriptor for \p KernelName.
  /// Returns std::nullopt if the descriptor is not found.
  std::optional<unsigned> getKernelVgprCount(llvm::StringRef KernelName,
                                             unsigned VgprGranuleSize) const;

  /// Read `group_segment_fixed_size` from the kernel descriptor for
  /// \p KernelName, i.e. the **static** (compile-time-fixed) LDS allocation
  /// per work-group in bytes. Returns std::nullopt if the descriptor symbol
  /// is missing.
  ///
  /// This is the only LDS quantity visible in the ELF. Dynamic LDS is
  /// allocated by the host at dispatch time (carried in the AQL packet's
  /// `group_segment_size` and propagated to the device via the
  /// `hidden_dynamic_lds_size` kernarg) and is *not* included here, so the
  /// returned value is a lower bound on the total LDS the kernel may
  /// touch. Callers that need to flag potential overflow of gfx1250 A0's
  /// 16-bit M0 limit can use this as a "definitely exceeds"
  /// check; "static fits, dynamic pushes over" cannot be detected
  /// statically. See AMDGPUUsage "Code Object V3 Kernel Descriptor"
  /// (GROUP_SEGMENT_FIXED_SIZE).
  std::optional<uint32_t>
  getKernelStaticLdsSize(llvm::StringRef KernelName) const;

  /// Read the SGPR count for \p KernelName from the \c amdhsa.kernels
  /// msgpack metadata note (\c .sgpr_count key), falling back to the kernel
  /// descriptor when the metadata note is absent. On GFX10+ the kernel
  /// descriptor's \c GRANULATED_WAVEFRONT_SGPR_COUNT is architecturally
  /// reserved, so metadata is the only reliable source when present.
  /// Returns std::nullopt if the matching metadata is malformed, the kernel is
  /// missing from present metadata, or the descriptor fallback is unavailable.
  std::optional<unsigned> getKernelSgprCount(llvm::StringRef KernelName) const;

  /// Read fixed \c .cluster_dims metadata for \p KernelName when present.
  /// Returns std::nullopt when the metadata note or key is absent, or when the
  /// matching metadata is malformed.
  std::optional<KernelClusterDims>
  getKernelClusterDims(llvm::StringRef KernelName) const;

  /// Update the RSRC1 VGPR granule count in the kernel descriptor for
  /// \p KernelName by adding \p ExtraVgprs. The SGPR granule field is
  /// not updated because it is reserved on GFX10+.
  void updateKernelDescriptor(llvm::StringRef KernelName, unsigned ExtraVgprs,
                              unsigned VgprGranuleSize);

  /// Virtual address at which growWithTrampolines appends the trampoline pool:
  /// the first page-aligned address above every existing allocatable section
  /// and PT_LOAD memory range.
  /// Callers that pre-compute branch/stub targets (B0-to-A0 trampolines,
  /// kernel-entry stubs) must resolve pool positions against this value so the
  /// baked branches land on the pool's final location. Single source of truth
  /// shared with growWithTrampolines. std::nullopt on section or segment range
  /// overflow.
  std::optional<uint64_t> trampolinePoolVAddr() const;

  /// Grow the ELF by appending the trampoline pool at a fresh virtual address
  /// (trampolinePoolVAddr()) in a new PT_LOAD segment, leaving every existing
  /// section, symbol, and segment in place. Returns a null unique_ptr on
  /// failure.
  ///
  /// Appending (rather than growing `.text` and shifting everything after it)
  /// preserves the absolute/PC-relative addresses baked into a fully-linked
  /// AMDGPU code object, which carries no relocations to fix up.
  std::unique_ptr<llvm::WritableMemoryBuffer>
  growWithTrampolines(llvm::ArrayRef<Trampoline> Trampolines,
                      llvm::ArrayRef<uint8_t> SNopBytes,
                      ExecutablePoolTargetState TargetState) const;

private:
  enum class KernelSgprCacheState {
    Uninitialized,
    Metadata,
    NoMetadata,
    Error,
  };

  ElfView(ELFFileT File, ELFT::ShdrRange Sections,
          const ELFT::Shdr *TextSection, unsigned TextSectionIndex)
      : File(std::move(File)), Sections(Sections), TextSection(TextSection),
        TextSectionIndex(TextSectionIndex) {}

  llvm::ArrayRef<FunctionTextRange> cachedFunctionTextRanges() const;
  const FunctionTextRange *
  findFunctionTextRangeAtAddress(uint64_t TextAddress) const;
  void initializeKernelDescriptorCache() const;
  void initializeKernelSgprCountCache() const;

  ELFFileT File;
  ELFT::ShdrRange Sections;
  const ELFT::Shdr *TextSection;
  unsigned TextSectionIndex;
  mutable std::optional<std::vector<FunctionTextRange>> FunctionRangeCache;
  mutable std::optional<std::vector<KernelDescriptorInfo>>
      KernelDescriptorCache;
  mutable llvm::StringMap<uint64_t> KernelDescriptorFileOffsetCache;
  mutable bool KernelDescriptorCacheComplete = false;
  mutable KernelSgprCacheState SgprCacheState =
      KernelSgprCacheState::Uninitialized;
  mutable llvm::StringMap<std::optional<unsigned>> KernelSgprCountCache;
};

// -- Free-function ELF helpers (no ELF state required) ------------------------

/// Overwrite instruction bytes at \p InstOffset with \p Rule.ReplaceBytes,
/// padding remaining bytes with s_nop instructions sourced from \p
/// LS.SNopBytes. Returns false on bounds violation or if \p LS has no cached
/// s_nop encoding.
struct LLVMState;
[[nodiscard]] bool applyByteReplace(const RewriteRule &Rule,
                                    uint64_t InstOffset, uint32_t InstSize,
                                    uint8_t *Text, uint64_t TextSize,
                                    const LLVMState &LS);

/// Find the nearest NOP sled to \p Offset with at least \p Needed bytes of
/// free space. \p Use controls whether selection is restricted to owner-local
/// bodies, allows certified relocation bodies in post-function padding, or
/// allows branch-only gateways. Returns nullptr if none is found within
/// MaxSledDistance.
NopSled *findNearestSled(std::vector<NopSled> &Sleds, uint64_t Offset,
                         uint64_t Needed,
                         NopSledUse Use = NopSledUse::OwnerBody);

// -- RewriteConfig ------------------------------------------------------------
//
// ISA-specific parameters that drive the generic rewriting infrastructure.
// Constructed by the policy layer (e.g. GFX1250 B0-to-A0 in PR #2203) and
// threaded through the MC helpers (buildTrampoline below) and the policy
// PatchContext so infrastructure has zero ISA assumptions.
//
// Instruction-encoding bits (s_branch / s_nop opcodes) are deliberately NOT
// members of this struct -- they are derived from the MC layer at initLLVM()
// time and exposed via LLVMState (SBranchOpcode, SNopBytes plus the
// encodeSBranch method), so the policy layer never has to hardcode target
// opcode values.

struct RewriteConfig {
  std::string SourceIsa;
  std::string TargetIsa;
  std::string TargetCpu;
  unsigned MaxVgprs = 0;
  unsigned MaxSgprs = 0;
  unsigned VgprGranuleSize = 0;
  bool RunB0A0Patches = true;
  MaskWorkaroundPolicy MaskPolicy = MaskWorkaroundPolicy::None;
};

// -- LLVM MC context ----------------------------------------------------------
//
// Bundle of per-ISA LLVM MC objects. Populated by initLLVM, consumed by the
// decode/encode helpers and by the downstream policy layer. Also caches a
// handful of AMDGPU instruction primitives (s_branch MC opcode, pre-encoded
// s_nop bytes) and exposes the encodeSBranch method -- this keeps all
// target-specific opcode knowledge inside the MC layer and off the policy /
// infrastructure layer.

struct LLVMState {
  const llvm::Target *Target = nullptr;
  std::unique_ptr<llvm::MCRegisterInfo> MRI;
  std::unique_ptr<const llvm::MCAsmInfo> MAI;
  std::unique_ptr<llvm::MCInstrInfo> MCII;
  std::unique_ptr<llvm::MCSubtargetInfo> STI;
  std::unique_ptr<llvm::MCContext> Ctx;
  std::unique_ptr<llvm::MCObjectFileInfo> MOFI;
  std::unique_ptr<llvm::MCDisassembler> MCD;
  std::unique_ptr<llvm::MCInstPrinter> MCIP;
  std::unique_ptr<llvm::MCCodeEmitter> MCE;
  /// Target-provided branch / call / relocation analysis. May be null on
  /// targets that do not implement MCInstrAnalysis; callers must check
  /// before dispatching. Cached here so downstream patch passes can ask
  /// `MIA->isBranch(Inst)` / `isCall(Inst)` / `evaluateBranch(...)` instead
  /// of matching mnemonic strings.
  std::unique_ptr<llvm::MCInstrAnalysis> MIA;
  std::string Cpu;

  /// MC opcode index for `s_branch`, resolved once at initLLVM() via the
  /// asm parser. Used by encodeSBranch() below to construct a fresh MCInst
  /// per call.
  unsigned SBranchOpcode = 0;

  /// MC opcode index for `s_nop`. Resolved via the asm parser at initLLVM()
  /// time so decoded-stream consumers (e.g. buildNopSledMap) can match NOPs
  /// by opcode rather than mnemonic string.
  unsigned SNopOpcode = 0;

  /// Pre-encoded bytes for `s_nop 0` (MinInstSize bytes). Populated at
  /// initLLVM() time via MCCodeEmitter and used by applyByteReplace() and
  /// NOP-sled padding paths instead of a hardcoded encoding.
  llvm::SmallVector<uint8_t, 4> SNopBytes;

  /// Cached `v_nop` MCInst, resolved at initLLVM() time. Used by the WMMA
  /// co-execution hazard patch to build trampolines without string
  /// round-trips.
  llvm::MCInst VNopInst;

  /// MC opcodes for the kernel-entry stub sequence, resolved once at
  /// initLLVM() time by parsing representative asm snippets. The idempotency
  /// matcher compares decoded opcodes against these cached values instead of
  /// matching disassembled mnemonic strings.
  unsigned GlobalWbOpcode = 0;
  unsigned SGetPcI64Opcode = 0;
  unsigned SAddU32Opcode = 0;
  unsigned SAddcU32Opcode = 0;
  unsigned SAddNcU64Opcode = 0;
  unsigned SSetPcI64Opcode = 0;
  unsigned SSwapPcI64Opcode = 0;
  unsigned SCallI64Opcode = 0;

  bool Valid = false;

  /// Encode a relative `s_branch` from \p FromOffset to \p ToOffset and
  /// return the MinInstSize encoded bytes. Returns an empty vector if the
  /// delta is unaligned, out of the 16-bit signed dword range, or if this
  /// LLVMState is not valid / has no cached s_branch opcode. Uses
  /// MCCodeEmitter for the encoding so no hardcoded opcode bits appear in
  /// the hotswap code. Empty-on-failure matches the convention used by
  /// encodeMCInst() and assembleSingleInst() so the same idiom applies
  /// uniformly across the MC layer.
  [[nodiscard]] llvm::SmallVector<uint8_t>
  encodeSBranch(uint64_t FromOffset, uint64_t ToOffset) const;
};

// -- Decoded instruction ------------------------------------------------------

struct InternalDecodedInst {
  uint64_t Offset = 0;
  uint32_t Size = 0;
  llvm::MCInst Inst;
  std::string Mnemonic;
};

// -- Function declarations (LLVM MC layer) ------------------------------------

/// Initialize LLVM MC infrastructure for the AMDGPU subtarget described by
/// \p TI (produced by Comgr's parseTargetIdentifier). The triple is built
/// from TI.Arch/Vendor/OS/Environ and features are threaded through to
/// createMCSubtargetInfo so the MC layer sees the same subtarget view the
/// caller asked for. AMDGPU MC registration is delegated to
/// COMGR::ensureLLVMInitialized(); the amdgcn Target lookup itself is cached
/// in a thread-safe function-local static.
LLVMState initLLVM(const TargetIdentifier &TI);

/// Disassemble \p Text into \p Decoded using \p LS. Unknown bytes are encoded
/// as MinInstSize-sized entries with mnemonic "<unknown>".
[[nodiscard]] bool decodeTextSection(const uint8_t *Text, uint64_t TextSize,
                                     const LLVMState &LS,
                                     std::vector<InternalDecodedInst> &Decoded);

/// Assemble a single instruction string, returning its encoded bytes.
llvm::SmallVector<uint8_t> assembleSingleInst(llvm::StringRef AsmStr,
                                              const LLVMState &LS);

/// Join \p AsmLines into a single newline-terminated assembly source string,
/// as expected by assembleSingleInst (which accepts multiple instructions).
std::string joinAsmLines(llvm::ArrayRef<std::string> AsmLines);

/// Assemble \p AsmLines and append a branch-back to the next instruction
/// after the original (\p OriginalOffset + \p OriginalSize). The branch-back
/// is encoded via LLVMState::encodeSBranch, so no ISA-specific opcode needs
/// to flow in from the caller.
///
/// NOTE: no production caller remains (WMMA-split now defers edge encoding to
/// emitToTrampoline / fixupTrampolineBranches). Kept only as a self-contained
/// helper exercised by the unit tests; prefer emitToTrampoline for new code.
Trampoline buildTrampoline(llvm::ArrayRef<std::string> AsmLines,
                           uint64_t OriginalOffset, uint32_t OriginalSize,
                           uint64_t TrampolineTextOffset, const LLVMState &LS);

/// Overload that accepts pre-decoded MCInst instructions directly,
/// encoding them via MCCodeEmitter without a string round-trip.
Trampoline buildTrampoline(llvm::ArrayRef<llvm::MCInst> Insts,
                           uint64_t OriginalOffset, uint32_t OriginalSize,
                           uint64_t TrampolineTextOffset, const LLVMState &LS);

/// Return true iff any register operand of \p WmmaInst overlaps the
/// destination operand of \p ValuInst (for WMMA/VALU co-execution hazard
/// detection). Delegates aliasing to MCRegisterInfo::regsOverlap so
/// sub-registers and tuple aliases are handled without a manual range
/// computation.
bool checkVgprOverlap(const llvm::MCInst &WmmaInst,
                      const llvm::MCInst &ValuInst,
                      const llvm::MCRegisterInfo &MRI);

/// WMMA/SWMMAC A0 vs B0 v_nop spacing requirement.
struct WmmaNopReq {
  int A0Nops = 4;
  int B0Nops = 4;
};

/// Classify the A0/B0 v_nop requirement for a WMMA/SWMMAC mnemonic.
WmmaNopReq classifyWmmaNops(llvm::StringRef Mnemonic);

/// Record a WMMA spacing deficit for a candidate VALU and return the maximum
/// requirement seen for that candidate. Multiple WMMA scans must compose the
/// strongest requirement rather than letting discovery order weaken it.
int updateWmmaHazardDeficit(llvm::DenseMap<size_t, int> &MaxDeficits,
                            size_t ValuIndex, int Deficit);

/// Patch the VOP3PX2 scale_src2 field (bits [58:50]) to VGPR0 encoding
/// (0x100) in a 16-byte instruction buffer. Returns true if the field
/// was modified (false if already set to the target value).
bool patchScaleSrc2(uint8_t *InstBytes);

// -- VGPR liveness types ------------------------------------------------------

/// Per-instruction def/use bitvectors over the VGPR index space. Populated by
/// getInstRegDefUse() during liveness analysis; each bit position corresponds
/// to one VGPR (index matches AMDGPU VGPR numbering, e.g. bit 5 = V5).
struct RegDefUse {
  llvm::BitVector Defs;
  llvm::BitVector Uses;
};

/// A basic block in the decoded-instruction CFG. Offsets are byte offsets
/// into .text; \c InstIndices stores positions in the flat Decoded vector;
/// \c Successors / \c Predecessors are indices into CFG::Blocks.
struct BasicBlock {
  uint64_t StartOffset = 0;
  uint64_t EndOffset = 0;
  llvm::SmallVector<size_t> InstIndices;
  llvm::SmallVector<unsigned> Successors;
  llvm::SmallVector<unsigned> Predecessors;
};

/// Control-flow graph over the decoded instruction stream. \c OffsetToBlock
/// is the inverted index mapping a .text byte offset to its owning block
/// index in \c Blocks, used to resolve branch-target / fall-through edges
/// during CFG construction.
struct CFG {
  std::vector<BasicBlock> Blocks;
  llvm::DenseMap<uint64_t, unsigned> OffsetToBlock;
};

struct MaterializedPcTransfer {
  uint64_t Begin = 0;
  uint64_t End = 0;
  uint64_t Target = 0;
};

std::optional<int64_t> getAbsoluteOperandValue(const llvm::MCOperand &Operand,
                                               const InternalDecodedInst &DI,
                                               llvm::ArrayRef<uint8_t> Text);

std::optional<MaterializedPcTransfer> evaluateMaterializedPcTransfer(
    llvm::ArrayRef<InternalDecodedInst> Decoded, size_t TransferIndex,
    const llvm::DenseSet<uint64_t> &DirectTargets, llvm::ArrayRef<uint8_t> Text,
    const LLVMState &LS, const ElfView &Elf,
    std::optional<llvm::ArrayRef<uint64_t>> TextSymbolOffsets = std::nullopt,
    std::optional<llvm::ArrayRef<ElfView::TextOffsetRange>> TextSymbolExtents =
        std::nullopt);

bool isStandardLinkReturn(const InternalDecodedInst &DI, const LLVMState &LS);

/// Dataflow-liveness result for a kernel's VGPR set. \c LiveBefore[i] and
/// \c LiveAfter[i] are the live-in / live-out bitvectors for Decoded[i].
/// \c Converged is false when the iterative solver hit its iteration cap;
/// callers fall back to a conservative all-VGPRs-live analysis in that case.
struct LivenessInfo {
  std::vector<llvm::BitVector> LiveBefore;
  std::vector<llvm::BitVector> LiveAfter;
  bool Converged = false;
};

/// Allocates scratch VGPRs for a patch point, preferring to reuse dead slots
/// from the kernel's existing allocation before extending the allocation past
/// the kernel descriptor's reported VGPR count. Constructed per patch site
/// with the live-set at that site and the kernel's current / maximum VGPR
/// counts.
struct VgprAllocator {
  llvm::BitVector LiveAtPoint;
  unsigned KdAllocatedVgprs = 0;
  unsigned NextAboveKd = 0;
  unsigned MaxVgprs = 0;
  unsigned ExtraAllocated = 0;

  VgprAllocator(const llvm::BitVector &Live, unsigned KdVgprs, unsigned Max)
      : LiveAtPoint(Live), KdAllocatedVgprs(KdVgprs), NextAboveKd(KdVgprs),
        MaxVgprs(Max) {}

  /// Allocate one VGPR not currently marked live. Returns std::nullopt if
  /// the kernel's existing VGPR pool is saturated and there is no headroom
  /// below MaxVgprs for an additional allocation.
  std::optional<unsigned> alloc() {
    if (int V = LiveAtPoint.find_last_unset_in(0, KdAllocatedVgprs); V != -1) {
      LiveAtPoint.set(V);
      return V;
    }
    if (NextAboveKd >= MaxVgprs)
      return std::nullopt;
    unsigned V = NextAboveKd++;
    ExtraAllocated++;
    LiveAtPoint.set(V);
    return V;
  }

  unsigned extraVgprsNeeded() const { return ExtraAllocated; }
};

/// Allocates scratch SGPRs for a patch point. Unlike VGPRs (which have full
/// dataflow liveness), SGPRs have no liveness analysis, so we always allocate
/// above the kernel descriptor's reported SGPR count. This is conservative
/// but safe: no SGPR currently in use by the kernel can be clobbered.
struct SgprAllocator {
  unsigned KdAllocatedSgprs = 0;
  unsigned NextAboveKd = 0;
  unsigned MaxSgprs = 0;

  SgprAllocator(unsigned KdSgprs, unsigned Max)
      : KdAllocatedSgprs(KdSgprs), NextAboveKd(KdSgprs), MaxSgprs(Max) {}

  /// Allocate one SGPR above the kernel's current count. Returns
  /// std::nullopt if no headroom remains below MaxSgprs.
  std::optional<unsigned> alloc() {
    if (NextAboveKd >= MaxSgprs)
      return std::nullopt;
    return NextAboveKd++;
  }

  unsigned extraSgprsNeeded() const { return NextAboveKd - KdAllocatedSgprs; }
};

/// Bookkeeping for a single patch site's scratch allocation. \c Offset is
/// the .text byte offset of the patch; \c ScratchRegs is the bitvector of
/// VGPRs the patch claimed at that site. Consumed by the post-patch
/// verifier (verifyPatchCorrectness) to check the patches are mutually
/// consistent across the kernel.
struct ScratchPatchInfo {
  uint64_t Offset = 0;
  llvm::BitVector ScratchRegs;
};

/// Transactional ownership and composition state for one replacement source.
/// A whole-function requirement may reserve a site before its ordinary
/// per-instruction owner is known. The first successful emission commits that
/// owner and composes every pending requirement into the same body.
struct SiteReplacementState {
  uint32_t OriginalSize = 0;
  unsigned RequiredLeadingVNops = 0;
  bool Committed = false;
  bool WmmaHazardComposed = false;
};

// -- Patch types --------------------------------------------------------------

/// Per-kernel counters accumulated by the patch passes. Reported via log()
/// at the end of the rewrite and exposed through the public
/// amd_comgr_hotswap_result_t once that result struct is wired up.
struct KernelPatchStats {
  unsigned ExtraVgprs = 0;
  unsigned ExtraSgprs = 0;
  unsigned ScratchReused = 0;
  unsigned ScratchAboveKd = 0;
};

struct SafeSgprUsageSummary {
  bool Valid = true;
  bool UsesVcc = false;
  bool HasCall = false;
  unsigned HighWatermark = 0;
};

struct SiteDeadSgprFunctionFacts {
  uint64_t Begin = 0;
  uint64_t End = 0;
  size_t GlobalFirst = 0;
  unsigned NumberedLimit = 0;
  std::vector<std::array<uint64_t, 2>> SafeBefore;
  llvm::BitVector ForbiddenResume;
};

struct KernelTextRange {
  uint64_t Begin = 0;
  uint64_t End = 0;
  std::vector<uint64_t> AdditionalEntries;
};

/// Forward must-analysis used by the gfx1250 initial-VMEM workaround. The
/// bitvectors are indexed like the mutable .text decode. DescriptorCovered
/// distinguishes dead code in a known kernel from helpers whose entry
/// semantics cannot be proven.
struct InitialVmemMustAnalysis {
  llvm::BitVector DescriptorCovered;
  llvm::BitVector Reachable;
  llvm::BitVector MustHavePriorVmem;
};

InitialVmemMustAnalysis
computeInitialVmemMustAnalysis(llvm::ArrayRef<InternalDecodedInst> Decoded,
                               llvm::ArrayRef<InternalDecodedInst> AllDecoded,
                               llvm::ArrayRef<KernelTextRange> KernelRanges,
                               const LLVMState &LS);

/// Forward must-analysis for the low 16 bits of tensor group descriptors.
/// A set bit means every modeled path to the tensor instruction provides a
/// group-1 base register whose low half is zero. MaskDefinitionOffsets records
/// the single reaching v_readfirstlane definition when every path agrees, so
/// one persistent mask can be shared by later tensor operations.
struct TensorDescriptorMustAnalysis {
  llvm::BitVector Low16KnownZero;
  std::vector<uint64_t> MaskDefinitionOffsets;
};

/// Exact, byte-validated kernel-entry stub metadata. Offsets are relative to
/// .text, including stubs in appended executable sections. The descriptor's
/// virtual dispatch edge enters Begin; Terminal is the stub's only decoded
/// transfer and resolves to Target.
struct TensorDispatchStub {
  uint64_t Begin = 0;
  uint64_t End = 0;
  uint64_t Terminal = 0;
  uint64_t Target = 0;
};

struct TensorAnalysisRange {
  uint64_t Begin = 0;
  uint64_t End = 0;
  // Original-.text control transfers from outside this range that target an
  // appended executable segment. A generated path rooted in this range may
  // not share one of these segment entries: the foreign path would carry an
  // unmodeled descriptor and EXEC state to the same re-entry tail.
  std::vector<uint64_t> ForeignExternalEntries;
  // Hardware dispatch roots from kernel descriptors that do not name an exact
  // validated HotSwap entry stub. They have no decoded predecessor edge and
  // therefore may not enter a candidate-owned external subgraph.
  std::vector<uint64_t> VirtualExternalEntries;
  // Complete set of resolved original-.text control-flow and fallthrough
  // edges. Source provenance distinguishes exact split-relay entry from an
  // arbitrary join into its address-materialization tail.
  std::vector<std::pair<uint64_t, uint64_t>> OriginalControlFlowEdges;
  // Callable and kernel roots in original .text. A root may enter a computed
  // PC sequence only at a self-initializing get-PC prefix.
  std::vector<uint64_t> OriginalCodeEntries;
  // Complete immutable set of validated dispatch stubs in this code object.
  // The range analysis uses it both to admit its own virtual dispatch edge
  // and to classify carry-form set-PC terminals belonging to other kernels.
  std::vector<TensorDispatchStub> DispatchStubs;
};

TensorDescriptorMustAnalysis computeTensorDescriptorMustAnalysis(
    llvm::ArrayRef<InternalDecodedInst> Decoded,
    llvm::ArrayRef<InternalDecodedInst> AllDecoded,
    llvm::ArrayRef<TensorAnalysisRange> KernelRanges, const LLVMState &LS,
    const llvm::DenseSet<uint64_t> &DirectControlFlowTargets, unsigned MaxSgprs,
    unsigned MaxVgprs);

/// Mutable per-run context threaded through all patch passes. Bundles the
/// input config, decoded instruction stream, raw .text bytes, MC state,
/// output streams (trampolines / scratch info), and the shared ELF view +
/// liveness result so patch passes have a single parameter to pass around.
struct PatchContext {
  const RewriteConfig &Config;
  std::vector<InternalDecodedInst> &Decoded;
  uint8_t *Text = nullptr;
  uint64_t TextSize = 0;
  // .text-relative offset at which the appended trampoline pool begins
  // (trampolinePoolVAddr() - textAddr()). Trampoline branch offsets are
  // computed against this, not TextSize, since the pool no longer sits
  // immediately after .text.
  uint64_t PoolBaseOffset = 0;
  const LLVMState &LS;
  std::vector<Trampoline> &OutTrampolines;
  std::vector<NopSled> &NopSleds;
  ElfView &Elf;
  const LivenessInfo &Liveness;
  llvm::StringMap<KernelPatchStats> &KernelStats;
  std::vector<ScratchPatchInfo> &OutScratchPatches;
  const InitialVmemMustAnalysis *InitialVmemAnalysis = nullptr;
  const TensorDescriptorMustAnalysis *TensorDescriptorAnalysis = nullptr;
  // Indexed once from the original instruction stream. CFG analyses use these
  // sets to model predecessor merges and indirect entries without rescanning
  // the complete code object for every patch site.
  std::optional<llvm::DenseSet<uint64_t>> DirectControlFlowTargets;
  // Sized, non-callable .text data/object extents. Moving any overlapping
  // byte would violate symbol ownership even when the symbol starts before a
  // candidate instruction window.
  std::optional<llvm::ArrayRef<ElfView::TextOffsetRange>> TextSymbolExtents;
  llvm::DenseSet<uint64_t> IndirectControlFlowFunctions;
  llvm::DenseSet<uint64_t> CrossFunctionInteriorEntryFunctions;
  bool HasUnknownArbitraryIndirectTarget = false;
  // True before an instruction only when forward must-dataflow proves that
  // MODE selects the same physical VGPR bank for VALU destinations and DS
  // address operands on every reachable path.
  llvm::BitVector VgprMsbDstSrc0EqualBefore;
  // Exact persistent bank selectors before each instruction, or -1 when a
  // path merge, call, or unknown MODE definition prevents recovery. Long
  // range DS2 proofs compare a VALU definition's dst selector with the later
  // DS address operand's src0 selector.
  std::vector<int8_t> VgprMsbDstBefore;
  std::vector<int8_t> VgprMsbSrc0Before;
  // Exact packed src0/src1/src2/dst VGPR-MSB mode before each instruction.
  // Negative sentinels distinguish unanalyzed functions, CFG-unreachable
  // instructions, and reachable paths with an unknown field. WMMA splitting
  // uses the complete mode while DS2 alignment can still consume the
  // independently merged dst/src0 facts.
  std::vector<int16_t> VgprMsbModeBefore;
  // Set at a VALU copy when its corresponding destination is known to receive
  // an 8-byte-aligned numbered SGPR value on every path. The two vectors keep
  // the slots of v_dual_mov_b32 independent.
  llvm::BitVector VgprDef0AlignedTo8;
  llvm::BitVector VgprDef1AlignedTo8;
  // DS2 alignment exemptions are decided from the immutable decoded stream
  // before any earlier patch relabels an instruction as <replaced>.
  llvm::BitVector Ds2AddressProvenAligned;
  // Source-window expansion must not split clause/delay geometry or move a
  // patch site whose replacement has explicitly position-sensitive state.
  llvm::DenseSet<uint64_t> RelocationProtectedOffsets;
  // Keep the source of relocation protection so removing an unsafe clause
  // releases only that clause's members. Counts preserve overlapping or
  // malformed clause ranges; delay and dynamically added protections remain
  // independent and must survive clause removal.
  llvm::DenseMap<uint64_t, unsigned> ClauseRelocationProtectionCounts{0};
  llvm::DenseSet<uint64_t> NonClauseRelocationProtectedOffsets;
  // Instructions whose execution was moved into a larger atomic replacement
  // window. Keep their immutable MC description available to analyses that
  // reason about the relocated instruction order, but prevent later patch
  // families from emitting a second replacement at the stale linked address.
  llvm::DenseSet<uint64_t> ClaimedReplacementOffsets;
  // Descriptor definitions already extended by this pass with the persistent
  // A0 low-half mask. A must-analysis may reuse one for a later tensor site,
  // but may not use the reaching-definition fact to authorize a fresh move.
  llvm::DenseSet<uint64_t> TensorMaskedDefinitionOffsets;
  // Single source of truth for exact replacement ownership and pending
  // whole-function composition requirements. This also covers NOP-sled
  // placements, which do not appear in OutTrampolines.
  std::map<uint64_t, SiteReplacementState> SiteReplacements;
  llvm::SmallVector<uint64_t, 16> WmmaHazardSites;
  uint32_t WmmaHazardsComposed = 0;
  // Required patches are transformations whose unpatched original code is
  // unsafe to return when the selected rewrite policy needs the patch.
  bool RequiredPatchFailed = false;
  bool RequiredPatchApplied = false;
  // Conservative upper bound for the final bytes occupied by trampolines
  // already queued in OutTrampolines. Far entries also reserve their eventual
  // pool-island slot and bounded straight-line growth, so a later short-branch
  // decision cannot be invalidated when the final pool is laid out.
  uint64_t QueuedTrampolineBytes = 0;
  // Largest instruction in the immutable decoded stream. Source-window growth
  // stops after crossing SetPcForwardSequenceBytes, so this bounds its final
  // overshoot without assuming an ISA-wide maximum encoding size.
  uint32_t MaxDecodedInstSize = 0;
  // Safe far-return scratch allocation can be queried at many patch sites in
  // one function. Cache the immutable decoded SGPR usage summaries so each
  // function, and the whole-object fallback, is scanned at most once.
  std::optional<SafeSgprUsageSummary> WholeObjectSgprUsage;
  llvm::DenseMap<std::pair<uint64_t, uint64_t>, SafeSgprUsageSummary>
      FunctionSgprUsage{0};
  llvm::DenseMap<std::pair<uint64_t, uint64_t>, SiteDeadSgprFunctionFacts>
      SiteDeadSgprFacts{0};
};

inline void protectNonClauseRelocationOffset(PatchContext &Ctx,
                                             uint64_t Offset) {
  Ctx.NonClauseRelocationProtectedOffsets.insert(Offset);
  Ctx.RelocationProtectedOffsets.insert(Offset);
}

/// Return whether \p Mnemonic is one of the A0-incompatible WMMA forms owned
/// by the split patch.
bool isWmmaSplitPatchCandidate(llvm::StringRef Mnemonic);

/// Return whether the instruction at \p Idx owns a precomputed whole-pass
/// requirement or still needs an independent configured HotSwap rewrite.
/// Atomic source-window relocation uses this to avoid claiming its source.
bool requiresIndependentInstructionRewrite(const PatchContext &Ctx, size_t Idx);

/// A block of numbered SGPRs that is not referenced in the function being
/// patched, or anywhere in the code object when the site may be reached by a
/// call whose register requirements cannot be bounded locally.
struct SafeSgprScratchBlock {
  unsigned Base = 0;
  unsigned Count = 0;
  bool IsSiteProven = false;
};

/// Find an aligned block of unused numbered SGPRs for \p TextOffset. Returns
/// nullopt after logging when no block fits below RewriteConfig::MaxSgprs.
std::optional<SafeSgprScratchBlock>
findSafeSgprScratchBlock(PatchContext &Ctx, uint64_t TextOffset, unsigned Count,
                         unsigned Alignment, llvm::StringRef Context,
                         bool DiagnoseFailure = true);

/// Charge a previously selected global block to the kernel owning \p
/// TextOffset. If the site is in an ordinary device function, conservatively
/// charge every kernel descriptor because the ELF does not carry a complete
/// call graph.
bool commitSafeSgprScratchBlock(PatchContext &Ctx, uint64_t TextOffset,
                                const SafeSgprScratchBlock &Block,
                                llvm::StringRef Context);

/// Decode \p Bytes and return every numbered SGPR touched by an explicit or
/// implicit operand. Returns nullopt after logging if the sequence cannot be
/// decoded completely.
std::optional<llvm::BitVector>
collectTouchedNumberedSgprs(llvm::ArrayRef<uint8_t> Bytes,
                            unsigned NumberedSgprLimit, const LLVMState &LS);

/// Return true when both halves of the aligned numbered SGPR pair beginning at
/// \p SgprBase are defined before any reachable use from \p ResumeIndex. A call
/// reached while either half remains live fails closed because its outgoing
/// argument registers are not explicit in the machine instruction.
bool isSgprPairDeadFrom(llvm::ArrayRef<InternalDecodedInst> Function,
                        size_t ResumeIndex, unsigned SgprBase,
                        const LLVMState &LS, llvm::ArrayRef<uint8_t> Text = {});

/// Return true when both VCC halves are dead from \p ResumeIndex. On gfx1250
/// wave32, implicit composite VCC operands affect VCC_LO; explicit VCC and
/// VCC_HI operands retain their full register semantics. Proven ABI calls and
/// returns terminate the caller-clobbered VCC lifetime.
bool isVccPairDeadFrom(llvm::ArrayRef<InternalDecodedInst> Function,
                       size_t ResumeIndex, const LLVMState &LS,
                       llvm::ArrayRef<uint8_t> Text = {});

/// Return the numbered SGPRs that every path from the original resume point
/// defines before a use, call, or exit. Facts are cached per function.
std::optional<llvm::BitVector> getSiteDeadNumberedSgprs(PatchContext &Ctx,
                                                        uint64_t InstOffset,
                                                        uint32_t InstSize);

/// Build immutable site-dead facts before any patch relabels Decoded entries.
void precomputeSiteDeadSgprFacts(PatchContext &Ctx);

// -- Trampoline emission helpers (defined in comgr-hotswap-b0a0.cpp) ----------

uint64_t getNopSledBytesNeeded(
    const PatchContext &Ctx, uint64_t InstOffset, uint32_t InstSize,
    llvm::ArrayRef<uint8_t> Replacement,
    ReplacementPlacement Placement = ReplacementPlacement::Default);

[[nodiscard]] bool
emitToTrampoline(PatchContext &Ctx, uint64_t InstOffset, uint32_t InstSize,
                 llvm::ArrayRef<uint8_t> Replacement,
                 ReplacementPlacement Placement = ReplacementPlacement::Default,
                 bool DiagnoseFailure = true);

/// Return the encoded s_delay_alu dependency span, or the conservative ISA
/// maximum when the immediate cannot be interpreted safely.
unsigned getDelayProtectedSpan(const InternalDecodedInst &DI);

/// Return true when direct-control-flow information is unavailable, the
/// half-open window is malformed, or a direct edge enters after \p Begin and
/// before \p End. An edge to Begin is allowed because the replacement branch
/// remains there; an edge to any later instruction or literal slot is not.
bool hasDirectControlFlowTargetInWindowInterior(
    const std::optional<llvm::DenseSet<uint64_t>> &Targets, uint64_t Begin,
    uint64_t End);

/// Return true when both edges of a replacement at \p InstOffset can use
/// s_branch with the trampoline's current queued pool position.
bool canEmitShortTrampoline(const PatchContext &Ctx, uint64_t InstOffset,
                            uint32_t InstSize, uint64_t ReplacementSize);

/// Encode an SCC-neutral indirect long branch using the aligned numbered SGPR
/// pair at \p SgprBase. The displacement is applied with s_add_nc_u64; no
/// s_add_pc_i64 is emitted and no third SGPR is needed to save SCC.
llvm::SmallVector<uint8_t> encodeSetPCLongBranch(const LLVMState &LS,
                                                 uint64_t FromOffset,
                                                 uint64_t TargetOffset,
                                                 unsigned SgprBase);

[[nodiscard]] bool emitReplacementCode(
    PatchContext &Ctx, uint64_t InstOffset, uint32_t InstSize,
    llvm::ArrayRef<uint8_t> Replacement,
    ReplacementPlacement Placement = ReplacementPlacement::Default,
    bool DiagnoseFailure = true);

/// Whether p Offset is reserved by a pending requirement or already owns a
/// committed replacement. Relocation windows may not move such a site, while
/// its exact per-instruction owner may still emit through the shared helpers.
bool hasSiteReplacementReservation(const PatchContext &Ctx, uint64_t Offset);

// -- Patch dispatch vtable ----------------------------------------------------
//
// Function-pointer dispatch table that replaces the prior LLVM_ATTRIBUTE_WEAK
// + `#if !defined(_MSC_VER)` override pattern. PE/COFF does not honour weak
// the way ELF does, so on Windows the weak stubs silently won every patch
// call and the feature was a no-op (issue ROCm/llvm-project#2479).
//
// Patch modules supply their implementations through register*Patch
// functions invoked by installHotswapPatches(). The membership list is
// comgr-hotswap-patches.def; each entry there corresponds to one slot
// below and one register*Patch function in a sibling
// comgr-hotswap-patch-*.cpp. nullptr slots are treated as no-op by the
// dispatcher, so an unmigrated pass family (e.g. scratch) is safe to
// leave unbound until its first strong override lands.
//
// The singleton accessor below eagerly installs every registered slot in
// its own initializer, so production callers never observe an empty
// vtable. installHotswapPatches() is still exported for unit tests that
// want to drive the install against a local HotswapPatchVTable.

struct HotswapPatchVTable {
  // Per-instruction passes: called in declaration order; first non-zero
  // return wins for an instruction (matches the pre-vtable dispatcher
  // behaviour in applyGfx1250B0toA0Rules).
  uint32_t (*applyInPlacePatches)(PatchContext &, size_t) = nullptr;
  uint32_t (*applyTrampolinePatches)(PatchContext &, size_t) = nullptr;
  uint32_t (*applyWmmaSplitPatches)(PatchContext &, size_t) = nullptr;
  uint32_t (*applyScratchPatches)(PatchContext &, size_t) = nullptr;
  uint32_t (*applyWmmaScale16Patches)(PatchContext &, size_t) = nullptr;

  // Whole-kernel passes: called once per kernel after the per-instruction
  // loop completes.
  bool (*precomputeWmmaHazards)(PatchContext &) = nullptr;
  uint32_t (*applyWmmaHazardPatch)(PatchContext &) = nullptr;
  uint32_t (*applyVop3px2Src2Fix)(PatchContext &) = nullptr;
};

/// Walk comgr-hotswap-patches.def and bind every patch module's
/// implementation into \p VT by calling its register*Patch function.
/// A missing register*Patch produces a link error, which is the
/// loud-failure shape the weak-symbol pattern lacked. Production code
/// never calls this directly; it runs inside getHotswapPatchVTable()'s
/// initializer. Exposed here so unit tests can drive the install against
/// a local HotswapPatchVTable.
void installHotswapPatches(HotswapPatchVTable &VT);

/// Process-wide HotswapPatchVTable singleton (Meyers-style). The
/// initializer eagerly calls installHotswapPatches() on its own storage,
/// so every reference returned here is to a fully bound vtable. C++11
/// [stmt.dcl]/4 guarantees the initializer runs exactly once and is safe
/// under concurrent first access, which removes the need for an explicit
/// std::call_once at the entry point and any inter-TU static-init order
/// contract on the patch modules.
HotswapPatchVTable &getHotswapPatchVTable();

// Forward-declare every patch module's installer from the central .def
// registry. Patch modules define these in their comgr-hotswap-patch-*.cpp;
// installHotswapPatches() consumes them; unit tests under test-unit/ also
// invoke them directly. A patches.def line with no matching definition
// produces a libamd_comgr / HotswapMCTests link error.
#define HOTSWAP_PATCH(Name) void register##Name##Patch(HotswapPatchVTable &);
#include "comgr-hotswap-patches.def"
#undef HOTSWAP_PATCH

/// Cache every DS2 alignment exemption before the patch loop mutates Decoded.
void precomputeDs2AddressAlignment(PatchContext &Ctx);

// -- Function declarations (kernel-entry trampoline pass) ---------------------

struct KernelEntryTrampolineFixup {
  std::string KernelName;
  uint64_t StubTextOffset = 0;
  unsigned RequiredSgprs = 0;
  uint32_t InstPrefLines = 0;
};

/// Build a 256-byte, entry-aligned HotSwap kernel-entry stub at
/// \p StubVAddr that jumps to \p EntryVAddr using PC-relative address
/// materialization. Returns an empty vector if MC assembly fails.
llvm::SmallVector<uint8_t> buildKernelEntryTrampoline(uint64_t StubVAddr,
                                                      uint64_t EntryVAddr,
                                                      unsigned ScratchSgpr,
                                                      const LLVMState &LS);

/// Structural matcher for the entry stubs produced by
/// buildKernelEntryTrampoline, used to keep the rewrite idempotent.
bool isKernelEntryTrampoline(llvm::ArrayRef<uint8_t> Bytes,
                             const LLVMState &LS);

struct KernelEntryTrampolineInfo {
  uint64_t TargetVAddr = 0;
  uint64_t TerminalVAddr = 0;
};

/// Return the resolved target and terminal instruction address of an exact
/// HotSwap entry stub. Besides validating the generated instruction shape,
/// this requires the remainder of the fixed-size stub to contain only the
/// exact generated s_code_end padding.
std::optional<KernelEntryTrampolineInfo>
getKernelEntryTrampolineInfo(llvm::ArrayRef<uint8_t> Bytes, uint64_t StubVAddr,
                             const LLVMState &LS);

/// Return the original kernel entry encoded by a structurally valid HotSwap
/// entry stub at \p StubVAddr. Returns std::nullopt for a non-stub or malformed
/// candidate.
std::optional<uint64_t>
getKernelEntryTrampolineTargetVAddr(llvm::ArrayRef<uint8_t> Bytes,
                                    uint64_t StubVAddr, const LLVMState &LS);

/// Cheap raw-byte prefilter for the entry stubs produced by
/// buildKernelEntryTrampoline. This is intentionally weaker than
/// isKernelEntryTrampoline and exists to avoid running the disassembler over
/// arbitrary original kernel entry bytes during idempotency checks.
bool hasKernelEntryTrampolinePrefix(llvm::ArrayRef<uint8_t> Bytes,
                                    const LLVMState &LS);

/// Compute the trailing readable guard needed after an appended kernel-entry
/// stub pool so CP instruction prefetches from the last stub cannot run past
/// mapped .text bytes.
uint64_t computeKernelEntryPrefetchGuardBytes(uint32_t InstPrefLines);

/// Append one entry stub per kernel descriptor that does not already target a
/// HotSwap entry stub. The stubs are appended to \p Growth and descriptor
/// rewrites are recorded in \p OutFixups for application after ELF growth.
std::optional<uint32_t> appendKernelEntryTrampolines(
    const ElfView &Elf, const LLVMState &LS, unsigned MaxSgprs,
    std::vector<Trampoline> &Growth,
    std::vector<KernelEntryTrampolineFixup> &OutFixups);

/// Apply descriptor rewrites recorded by appendKernelEntryTrampolines after
/// the ELF has been grown.
bool rewriteKernelEntryDescriptorOffsets(
    llvm::WritableMemoryBuffer &OutBuf, uint64_t PoolVAddr,
    llvm::StringRef TargetCpu,
    llvm::ArrayRef<KernelEntryTrampolineFixup> Fixups);

/// Add a `<kernel_name>.stub` STT_FUNC symbol to the code object's `.symtab`
/// for each appended kernel-entry stub, so tools that resolve a dispatch's
/// entry address to a name (e.g. rocgdb `info dispatches`, which reads the
/// non-alloc `.symtab`) report the stub instead of a bare address. Returns a
/// newly allocated buffer with the grown `.symtab` / `.strtab`, or nullptr if
/// no symbols were added (empty fixups, missing `.symtab`, or a structural
/// problem) -- callers treat nullptr as "keep the existing buffer", since the
/// symbol is a debugging aid and its absence is not a correctness failure.
///
/// The non-allocating `.symtab` / `.strtab` copies are relocated after the
/// existing ELF bytes, so the executable pool and all program-header mappings
/// remain unchanged; `.dynsym` (used by the loader) is left untouched.
std::unique_ptr<llvm::WritableMemoryBuffer> addKernelEntryTrampolineSymbols(
    llvm::WritableMemoryBuffer &In, uint64_t PoolVAddr,
    llvm::ArrayRef<KernelEntryTrampolineFixup> Fixups);

// -- Function declarations (GFX1250 hotswap policy layer) ---------------------

struct Gfx1250RewriteOptions {
  bool RunB0A0Patches = true;
  bool RunEntryTrampolines = false;
  MaskWorkaroundPolicy MaskPolicy = MaskWorkaroundPolicy::None;
};

/// Run the selected GFX1250 hotswap rewrite passes on \p ElfData / \p ElfSize.
/// \p TargetIdent is the parsed target ISA (produced upstream by Comgr's
/// parseTargetIdentifier() or the hotswap-local stepping parser); it is
/// threaded into the MC init so the subtarget triple and feature flags are
/// preserved rather than being reconstructed from just the processor name. On
/// success \p Out is populated with an owned buffer containing the rewritten
/// code object. The caller can transfer the buffer directly to a comgr
/// DataObject via DataObject::setData(std::unique_ptr<MemoryBuffer>).
amd_comgr_status_t retargetCodeObject(const void *ElfData, size_t ElfSize,
                                      const TargetIdentifier &TargetIdent,
                                      const Gfx1250RewriteOptions &Options,
                                      std::unique_ptr<llvm::MemoryBuffer> &Out);

} // namespace hotswap
} // namespace COMGR

#endif // COMGR_HOTSWAP_INTERNAL_H
