//===- wave-projection.h - Hotswap transpiler -----------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_WAVE_PROJECTION_H
#define HOTSWAP_TRANSPILER_WAVE_PROJECTION_H

#include "hotswap/decoder/decoded-inst.h"
#include "hotswap/decoder/isa-profile.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"

#include <array>
#include <cassert>

namespace COMGR::hotswap {

struct MCState;

// ============================================================================
// WaveProjection -- the cross-wave translation policy surface.
//
// A projection maps a source-ISA wavefront onto a target-ISA wavefront when
// the two wave widths differ. This is an abstract base; each concrete
// projection is a subclass and the choice is made once when the raiser
// constructs it.
class WaveProjection {
public:
  WaveProjection(const ISAProfile &SrcIsa, const ISAProfile &TgtIsa,
                 llvm::Type *I32Ty, llvm::Type *I64Ty)
      : Src(SrcIsa), Tgt(TgtIsa), I32Ty(I32Ty), I64Ty(I64Ty),
        ExecStorageTy(SrcIsa.isWave32() ? I32Ty : I64Ty) {
    assert(SrcIsa.hasValidWaveSize() && "invalid source wave size");
    assert(TgtIsa.hasValidWaveSize() && "invalid target wave size");
    assert(TgtIsa.waveSize() >= SrcIsa.waveSize() &&
           "wave projection does not support narrowing");
  }

  virtual ~WaveProjection() = default;

  const ISAProfile &sourceIsa() const { return Src; }
  const ISAProfile &targetIsa() const { return Tgt; }

  // Source kernel's max_flat_workgroup_size (threads per workgroup), set by
  // the raiser after construction. Used to clamp the workitem id of the
  // undispatched upper target lanes.
  void setMaxFlatWorkgroupSize(unsigned N) { MaxFlatWG = N; }
  // Hardware-width wave mask (i32 on wave32 target, i64 on wave64 target).
  // Distinct from the EXEC alloca storage width returned by
  // `execStorageTy()`.
  llvm::Type *waveMaskTy() const { return Tgt.isWave32() ? I32Ty : I64Ty; }

  // Wave mask at the source ISA's width (i32 on wave32 source, i64 on wave64):
  // the width the source observes when reading or writing EXEC and SGPR wave
  // masks through 32/64-bit scalar operations.
  llvm::Type *sourceWaveMaskTy() const {
    return Src.isWave32() ? I32Ty : I64Ty;
  }

  // EXEC alloca storage width chosen by the projection; the base uses the
  // source wave-mask width. A subclass may widen it (e.g. to the target mask)
  // in its constructor.
  llvm::Type *execStorageTy() const { return ExecStorageTy; }

  // Emit the initial value to store into the EXEC alloca at kernel entry.
  // Default is all-ones (every source lane active on entry). Projections that
  // decouple the hardware EXEC from the modeled source EXEC override this to
  // capture the hardware EXEC into the alloca while forcing hardware EXEC to
  // all-ones.
  virtual llvm::Value *emitInitialExec(llvm::IRBuilder<> &B) const;

  // True iff a 32-bit write to EXEC_LO means "replicate across the full widened
  // EXEC" rather than the architectural "replace the low half, keep the high
  // half". A projection that models each target lane as an independent source
  // thread sets this, so a whole-wave EXEC_LO write fans out to both halves.
  bool broadcastNarrowExecLoWrite() const { return BroadcastNarrowExecLoWrite; }

  // Emit the current lane's linear index within the wavefront (i32). Wave size
  // follows the target ISA, since the lane id is a property of the hardware the
  // raised IR runs on, not of the source ISA. Non-virtual: every projection
  // derives the lane id the same way.
  llvm::Value *emitLaneIdx(llvm::IRBuilder<> &B) const;

  // Emit the workitem-id-x value source-ISA code should observe under this
  // projection. The base returns the target hardware value; a projection that
  // splits or re-maps source waves overrides it.
  virtual llvm::Value *emitWorkitemIdX(llvm::IRBuilder<> &B) const;

  // Emit the source wave's linear ID within its workgroup.
  virtual llvm::Value *emitSourceWaveId(llvm::IRBuilder<> &B) const = 0;

  // Emit the full packed kernel-entry `v0` workitem id under this projection.
  // amdgpu packs the workitem id as x[0:9] | y[10:19] | z[20:29]; `NumDims`
  // (1..3, derived from the source descriptor's ENABLE_VGPR_WORKITEM_ID) says
  // how many fields the source enabled, so a 1-D kernel reduces to exactly
  // `emitWorkitemIdX` (no Y/Z math).
  virtual llvm::Value *emitPackedWorkitemId(llvm::IRBuilder<> &B,
                                            unsigned NumDims) const;

  // Given the current EXEC alloca value, return an i1 true iff the current lane
  // is active. Each concrete projection defines what "active" means.
  virtual llvm::Value *emitLaneActiveBit(llvm::IRBuilder<> &B,
                                         llvm::Value *ExecVal) const = 0;

  // Collect a per-lane i1 predicate into a wave-level bit-mask of width
  // `ResultTy`. The ballot itself must be at the target wave width (the AMDGPU
  // backend only selects ballot.i32 on wave32 and ballot.i64 on wave64) and
  // must be emitted in outer / full-EXEC control flow so inactive lanes do not
  // contribute 0. How a `ResultTy` narrower than the wave width is reconciled
  // is projection-specific.
  virtual llvm::Value *
  ballotI1ToWidth(llvm::IRBuilder<> &B, llvm::Value *Pred, llvm::Type *ResultTy,
                  const llvm::Twine &Name = "ballot") const = 0;

  // Project a wave-level bit-mask back onto the current lane's bit (i1).
  // Inverse direction of the ballot. Per-lane i1 inputs short-circuit
  // to a direct pass-through (some callers already produce the final
  // per-lane i1 and route through writeReg*(VCC, i1)); those must not
  // be reinterpreted as a one-bit wave mask.
  virtual llvm::Value *extractLaneBitFromWaveMask(llvm::IRBuilder<> &B,
                                                  llvm::Value *V) const = 0;

  // Return the source-wave slice of a wave mask, e.g. for `v_mbcnt_lo`.
  virtual llvm::Value *
  emitCurrentSourceWaveMask(llvm::IRBuilder<> &B, llvm::Value *Mask,
                            const llvm::Twine &Name = "source_wave_mask") const;

  // True iff this projection guarantees hardware EXEC = -1 between
  // `emitUnderExec` diamonds kernel-wide, so cross-lane collectives can run
  // without additional EXEC scaffolding.
  bool providesFullWaveExecInvariant() const {
    return ProvidesFullWaveExecInvariant;
  }

  // True iff handlers should lower source-ISA lane-indexed primitives
  // (`readlane`, `writelane`, `readfirstlane`) as source-wave-scoped
  // operations instead of target-wave-native AMDGPU intrinsics. Needed when a
  // target wave packs multiple source-wave instances, which a native
  // target-wave `readlane` / `readfirstlane` would collapse together.
  bool sourceWaveScopedLaneOps() const { return SourceWaveScopedLaneOps; }

  // True iff mbcnt-derived EXEC writes -- the vector `v_cmpx` and its scalar
  // `s_*_saveexec_b32` sibling -- can be projected into an independent
  // target-width EXEC mask per packed source wave. Requires an injective
  // source-wave mapping.
  bool preservesMbcntDerivedExec() const { return PreservesMbcntDerivedExec; }

  // True iff this projection expects the runtime to launch the block with a
  // `W_t / W_s`-scaled extent along `doubledDispatchDim()`, so each target wave
  // hosts one source wave in its low `W_s` lanes with the rest as replicas.
  // Equivalent to a scale factor above 1.
  bool usesDoubledDispatch() const { return DoubledDispatchFactor > 1; }

  // The block dimension (0=x, 1=y, 2=z) the runtime doubles when
  // `usesDoubledDispatch()` is true. Always the fastest wave-carrying
  // dimension (x) for the wave32->wave64 case; the higher dims that carry the
  // divergent predicate become wave-uniform once x is doubled. Meaningless
  // unless `usesDoubledDispatch()`.
  unsigned doubledDispatchDim() const { return DoubledDispatchDim; }

  // The integer factor by which the dispatch is scaled along
  // `doubledDispatchDim()` (`W_t / W_s`, i.e. 2 for wave32->wave64).
  // Meaningless unless `usesDoubledDispatch()`.
  unsigned doubledDispatchFactor() const { return DoubledDispatchFactor; }

  // Number of source waves whose per-lane fragment data is present in each
  // target wave under this projection's mapping. Callers that synthesise
  // per-source-wave passes iterate one pass per source wave.
  unsigned numSourceWavesPerTarget() const { return NumSourceWavesPerTarget; }

  // Return `v` wrapped in `@llvm.amdgcn.strict.wwm` iff the current projection
  // does not already guarantee HW EXEC=-1 kernel-wide; otherwise return `v`
  // unchanged. Accepts any type strict.wwm's overload set covers (integer /
  // floating-point scalars and fixed vectors thereof).
  llvm::Value *wrapAsWWMValue(llvm::IRBuilder<> &B, llvm::Value *V,
                              const llvm::Twine &Name = "wwm") const;

protected:
  // Emit the target hardware work-item ID for dimension 0, 1, or 2.
  llvm::Value *emitTargetWorkitemId(llvm::IRBuilder<> &B,
                                    unsigned Dimension) const;

  // Emit the target wave's linear ID within its workgroup.
  llvm::Value *emitTargetWaveId(llvm::IRBuilder<> &B) const;

  // Combine an already-projected workitem-id-x value with the native Y/Z
  // workitem-id fields into AMDGPU's packed `v0` layout
  // (x | y<<10 | z<<20). `NumDims` selects how many fields to fold in.
  llvm::Value *packWorkitemId(llvm::IRBuilder<> &B, llvm::Value *X,
                              unsigned NumDims) const;

  ISAProfile Src;
  ISAProfile Tgt;
  // Retained on the base so `waveMaskTy()` / `sourceWaveMaskTy()` /
  // `execStorageTy()` can return the canonical i32/i64 IR types without
  // re-deriving them from the current IRBuilder's context (subclasses are
  // constructed once per kernel and outlive any particular builder).
  llvm::Type *I32Ty;
  llvm::Type *I64Ty;

  // Per-projection configuration, read through the like-named accessors
  // above. The defaults describe the same-wave / replication policy;
  // the widening projections set the ones they change in their constructors,
  // so a projection's behaviour is declared in one place rather than spread
  // across virtual overrides.
  llvm::Type *ExecStorageTy;
  unsigned NumSourceWavesPerTarget = 1;
  unsigned DoubledDispatchDim = 0;
  unsigned DoubledDispatchFactor = 1;
  bool BroadcastNarrowExecLoWrite = false;
  bool ProvidesFullWaveExecInvariant = false;
  bool SourceWaveScopedLaneOps = false;
  bool PreservesMbcntDerivedExec = false;

  // Source max_flat_workgroup_size; 0 until the raiser sets it.
  unsigned MaxFlatWG = 0;
  // Cache for the function-invariant lane id, keyed by the function it was
  // emitted into so a projection reused across kernels re-emits per function
  // rather than returning a value from another one. See `emitLaneIdx`.
  mutable llvm::Function *CachedLaneIdxFunc = nullptr;
  mutable llvm::Value *CachedLaneIdx = nullptr;
  mutable std::array<llvm::Value *, 3> CachedWorkitemIds = {nullptr, nullptr,
                                                            nullptr};
};

// ============================================================================
// ReplicationProjection.
//
// Target lane L reads bit `L mod W_src` of the source EXEC mask; a target
// ballot is truncated to source width, taking the lower `W_src` lanes as
// canonical; a wave-level mask projected back onto a per-lane bit indexes by
// `lane_id mod W_src`. This is a translation policy, not a hardware fact.
class ReplicationProjection : public WaveProjection {
public:
  using WaveProjection::WaveProjection;

  llvm::Value *emitSourceWaveId(llvm::IRBuilder<> &B) const override;

  llvm::Value *emitLaneActiveBit(llvm::IRBuilder<> &B,
                                 llvm::Value *ExecVal) const override;
  llvm::Value *
  ballotI1ToWidth(llvm::IRBuilder<> &B, llvm::Value *Pred, llvm::Type *ResultTy,
                  const llvm::Twine &Name = "ballot") const override;
  llvm::Value *extractLaneBitFromWaveMask(llvm::IRBuilder<> &B,
                                          llvm::Value *V) const override;

  // Clamp the workitem id of undispatched upper target lanes so they replicate
  // a real lane's in-bounds addressing when the target wave is wider than the
  // source workgroup.
  llvm::Value *emitWorkitemIdX(llvm::IRBuilder<> &B) const override;

  // Apply the same phantom-lane clamp as `emitWorkitemIdX`, but to the whole
  // packed id, so undispatched upper target lanes replicate lane 0 (packed 0)
  // rather than getting a stray non-zero Y/Z.
  llvm::Value *emitPackedWorkitemId(llvm::IRBuilder<> &B,
                                    unsigned NumDims) const override;

  // Both the phantom-lane widening regime and same-wave instantiations carry
  // exactly one source wave per target wave, which is the base default
  // (`NumSourceWavesPerTarget == 1`), so no constructor override is needed.
};

// ============================================================================
// ReplicationDoubledDispatchProjection -- replication backed by a doubled
// dispatch.
//
// The runtime launches the block with a `W_t / W_s`-scaled extent along the
// wave-carrying dimension x, so each target wave hosts one source wave in lanes
// `0..W_s-1` and exact replicas in `W_s..W_t-1`. For wave32->wave64:
//
//   * the runtime doubles blockDim.x (grid unchanged);
//   * the raised kernel maps hardware workitem-id.x back to the logical source
//     id so hardware lane `W_s + i` sees the same logical thread as lane `i`;
//   * the raiser halves the in-kernel workgroup/grid-size query along x so
//     loops and reduction bounds still observe the source block size.
//
// A lane and its replica compute identically, so they share every predicate
// and cross-lane ops read valid duplicate data from the upper half. All of the
// per-source-wave cross-lane machinery is inherited; this class overrides only
// the workitem-id mapping. Utilisation is ~50%, so it is a correctness
// fallback, not the fast path.
class ReplicationDoubledDispatchProjection final
    : public ReplicationProjection {
public:
  ReplicationDoubledDispatchProjection(const ISAProfile &SrcIsa,
                                       const ISAProfile &TgtIsa,
                                       llvm::Type *I32Ty, llvm::Type *I64Ty)
      : ReplicationProjection(SrcIsa, TgtIsa, I32Ty, I64Ty) {
    // Doubled dispatch along x (the wave-carrying dimension); dim stays 0.
    DoubledDispatchFactor = TgtIsa.waveSize() / SrcIsa.waveSize();
  }

  // Remap hardware workitem-id.x to the logical source id so replica lanes
  // alias their originals. No phantom-lane clamp: under a doubled dispatch
  // every hardware lane maps to a valid logical thread (real or replica).
  llvm::Value *emitWorkitemIdX(llvm::IRBuilder<> &B) const override;
  llvm::Value *emitSourceWaveId(llvm::IRBuilder<> &B) const override;

  // Pack the remapped x with the source's raw y/z fields (which are already
  // per-thread correct and become wave-uniform once x is doubled). Bypasses
  // the base replication phantom-lane clamp.
  llvm::Value *emitPackedWorkitemId(llvm::IRBuilder<> &B,
                                    unsigned NumDims) const override;
};

// ============================================================================
// WaveNativeProjection -- widening (wave32 -> wave64) projection
// that preserves the full target-hardware EXEC mask.
//
// The EXEC alloca is sized to the target hardware wave-mask width, and each
// target lane is treated as an independent source-thread equivalent, so a
// data-dependent `v_cmpx` that differs on target lanes 0..31 vs 32..63 keeps
// both halves distinct through the ballot/AND/store round trip.
//
// Source-width EXEC writes (`s_mov_b32 exec_lo, v`) are replicated into both
// halves of the widened EXEC; narrowing reads take the low half. This is
// lossless as long as the source never observes the upper half of EXEC
// independently, which wave32 source ISAs cannot express.
//
// Correct only for wave32 -> wave64 widening; the constructor asserts on
// other directions.
class WaveNativeProjection final : public WaveProjection {
public:
  // The constructor sets the projection configuration: target-width EXEC
  // storage, full-wave-EXEC invariant (its `emitInitialExec` forces HW
  // EXEC=-1), broadcast-on-narrow-EXEC-write, preserved mbcnt-derived EXEC,
  // and two source waves per target wave (lanes 0..31 and 32..63).
  WaveNativeProjection(const ISAProfile &SrcIsa, const ISAProfile &TgtIsa,
                       llvm::Type *I32Ty, llvm::Type *I64Ty);

  llvm::Value *emitSourceWaveId(llvm::IRBuilder<> &B) const override;

  llvm::Value *emitInitialExec(llvm::IRBuilder<> &B) const override;
  llvm::Value *emitLaneActiveBit(llvm::IRBuilder<> &B,
                                 llvm::Value *ExecVal) const override;
  llvm::Value *
  ballotI1ToWidth(llvm::IRBuilder<> &B, llvm::Value *Pred, llvm::Type *ResultTy,
                  const llvm::Twine &Name = "ballot") const override;
  llvm::Value *extractLaneBitFromWaveMask(llvm::IRBuilder<> &B,
                                          llvm::Value *V) const override;
};

// ============================================================================
// ThreadLoopProjection -- source-wave-scoped execution for widening.
//
// Banks source-wave predicate masks in target-width storage and emits a
// virtual workitem id through one projection hook, so equality predicates keep
// the packed source waves distinct. The projection surface:
//   * target-width EXEC storage for per-source-wave predicate masks;
//   * source-wave-scoped lane ops;
//   * target ballots narrowed only when the destination is source-width;
// and reports `W_t / W_s` source waves per target wave.
//
// Opt-in: selected only by an explicit fallback path in the raiser.
class ThreadLoopProjection final : public WaveProjection {
public:
  // The constructor sets the projection configuration: target-width EXEC
  // storage, source-wave-scoped lane ops, and `W_t / W_s` source waves per
  // target wave.
  ThreadLoopProjection(const ISAProfile &SrcIsa, const ISAProfile &TgtIsa,
                       llvm::Type *I32Ty, llvm::Type *I64Ty);

  void setIterationAlloca(llvm::AllocaInst *Iter) { IterationAlloca = Iter; }
  llvm::Value *emitWorkitemIdX(llvm::IRBuilder<> &B) const override;
  llvm::Value *emitSourceWaveId(llvm::IRBuilder<> &B) const override;

  llvm::Value *emitLaneActiveBit(llvm::IRBuilder<> &B,
                                 llvm::Value *ExecVal) const override;
  llvm::Value *
  ballotI1ToWidth(llvm::IRBuilder<> &B, llvm::Value *Pred, llvm::Type *ResultTy,
                  const llvm::Twine &Name = "ballot") const override;
  llvm::Value *extractLaneBitFromWaveMask(llvm::IRBuilder<> &B,
                                          llvm::Value *V) const override;

private:
  llvm::AllocaInst *IterationAlloca = nullptr;
};

// ============================================================================
// EXEC-writer detection.
// ============================================================================

// True iff Di writes EXEC (EXEC/EXEC_LO, or EXEC_HI on a wave64 source),
// whether as an implicit def or an explicit destination operand.
bool instructionWritesEXEC(const DecodedInst &Di, const MCState &Mc);

// ============================================================================
// Cross-wave safety warning.
// ============================================================================

// Emit a warn-only diagnostic (through LLVM_DEBUG) when a cross-wave
// translation relies on replication of an EXEC-manipulating kernel.
// Returns true iff a diagnostic was emitted.
bool emitCrossWaveWarning(const WaveProjection &Proj, const MCState &Mc,
                          llvm::ArrayRef<DecodedInst> Insts,
                          llvm::StringRef SourceIsa, llvm::StringRef TargetIsa);

} // namespace COMGR::hotswap

#endif
