//===- handle-flat.cpp - Transpiler ---------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "transpiler/raiser/handlers.h"

#include "SIDefines.h"

#include "transpiler/decoder/amdgpu-mc-tables.h"
#include "transpiler/decoder/canonical-op.h"
#include "transpiler/decoder/decoded-inst.h"
#include "transpiler/decoder/parsed-reg.h"
#include "transpiler/raiser/flat-addr.h"
#include "transpiler/raiser/operand-resolver.h"
#include "transpiler/raiser/raise-context.h"
#include "transpiler/raiser/reg-file.h"

#include "MCTargetDesc/AMDGPUMCTargetDesc.h"

#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Value.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/AtomicOrdering.h"
#include "llvm/Support/Error.h"

#include <optional>

using namespace llvm;

namespace COMGR::transpiler {

namespace {

/// Payload and register interpretation of a GLOBAL access.
struct GlobalAccess {
  enum Flags { Store = 1, Signed = 2, HighHalf = 4 };
  /// Total bytes transferred, also used to scale the lane offset.
  unsigned SizeInBytes;
  /// Load extension and store source selection.
  unsigned DataFlags;

  unsigned widthInDwords() const { return (SizeInBytes + 3) / 4; }

  Align alignment() const {
    // Sub-dword accesses can be byte-aligned, including halfword accesses.
    return SizeInBytes < 4 ? Align(1) : Align(4);
  }
};

} // namespace

/// Return the payload and extension required by the opcode.
static std::optional<GlobalAccess> globalAccess(CanonicalOp Operation) {
  using Access = GlobalAccess;
  switch (Operation) {
  case CanonicalOp::GLOBAL_LOAD_U8:
    return Access{1, 0};
  case CanonicalOp::GLOBAL_LOAD_I8:
    return Access{1, Access::Signed};
  case CanonicalOp::GLOBAL_LOAD_U16:
    return Access{2, 0};
  case CanonicalOp::GLOBAL_LOAD_I16:
    return Access{2, Access::Signed};
  case CanonicalOp::GLOBAL_LOAD_B32:
    return Access{4, 0};
  case CanonicalOp::GLOBAL_LOAD_B64:
    return Access{8, 0};
  case CanonicalOp::GLOBAL_LOAD_B96:
    return Access{12, 0};
  case CanonicalOp::GLOBAL_LOAD_B128:
    return Access{16, 0};
  case CanonicalOp::GLOBAL_STORE_B8:
    return Access{1, Access::Store};
  case CanonicalOp::GLOBAL_STORE_B16:
    return Access{2, Access::Store};
  case CanonicalOp::GLOBAL_STORE_D16_HI_B8:
    return Access{1, Access::Store | Access::HighHalf};
  case CanonicalOp::GLOBAL_STORE_D16_HI_B16:
    return Access{2, Access::Store | Access::HighHalf};
  case CanonicalOp::GLOBAL_STORE_B32:
    return Access{4, Access::Store};
  case CanonicalOp::GLOBAL_STORE_B64:
    return Access{8, Access::Store};
  case CanonicalOp::GLOBAL_STORE_B96:
    return Access{12, Access::Store};
  case CanonicalOp::GLOBAL_STORE_B128:
    return Access{16, Access::Store};
  default:
    return std::nullopt;
  }
}

/// Return the LLVM IR type for a transfer of the given byte width.
static Type *globalAccessType(IRBuilder<> &B, unsigned SizeInBytes) {
  if (SizeInBytes <= 8)
    return B.getIntNTy(SizeInBytes * 8);
  return FixedVectorType::get(B.getInt32Ty(), SizeInBytes / 4);
}

/// Parse and validate the named GLOBAL data register tuple.
static Expected<ParsedReg>
globalDataReg(RaiseContext &Ctx, const DecodedInst &Di, AMDGPU::OpName Name,
              unsigned WidthInDwords, StringRef Role) {
  int Index = COMGR::transpiler::getNamedOperandIdx(Di.Inst.getOpcode(), Name);
  assert(Index >= 0 && static_cast<unsigned>(Index) < Di.numOperands() &&
         "global memory operation is missing its data operand");
  assert(Di.isReg(Index) && "global memory data operand must be a register");

  Expected<ParsedReg> Reg = Ctx.registers().parseReg(Di, Index);
  if (!Reg)
    return Reg.takeError();
  if (Reg->RegKind != ParsedReg::VGPR && Reg->RegKind != ParsedReg::AGPR)
    return unsupported(
        Ctx, Di, Twine("global memory ") + Role + " must be a VGPR or AGPR");
  assert(Reg->BaseIdx && "global memory data register has no base index");
  assert(Reg->WidthInDwords == WidthInDwords &&
         "global memory data register width does not match the opcode");
  return *Reg;
}

/// Return cache-policy bits accepted for a plain GLOBAL access. GFX12 models
/// SCOPE and accepts NV as a cache hint, but the raised IR drops NV because it
/// only controls whether fine-grained cache write-back or invalidation includes
/// the line. Treating the line as volatile is the conservative behavior.
static unsigned modeledAccessCachePolicy(const MCSubtargetInfo &STI) {
  if (STI.hasFeature(AMDGPU::FeatureGFX12Insts))
    return AMDGPU::CPol::SCOPE | AMDGPU::CPol::NV;
  return 0;
}

/// Return whether the access is coherent past the compute unit running it,
/// which every scope above SCOPE_CU is. Such an access participates in a
/// handshake with another workgroup, so it lifts to a volatile one: the
/// optimizer may then neither drop nor reorder it, and the target emits the
/// cache bits that keep it visible.
static bool isCoherentBeyondComputeUnit(const DecodedInst &Di) {
  return (globalCachePolicy(Di) & AMDGPU::CPol::SCOPE) !=
         AMDGPU::CPol::SCOPE_CU;
}

/// Emit a GLOBAL load under EXEC, extending sub-dword data to a full VGPR.
static Error emitGlobalLoad(RaiseContext &Ctx, const DecodedInst &Di,
                            const GlobalAccess &Access) {
  Expected<ParsedReg> Destination = globalDataReg(
      Ctx, Di, AMDGPU::OpName::vdst, Access.widthInDwords(), "destination");
  if (!Destination)
    return Destination.takeError();

  Expected<Value *> Address =
      emitGlobalAddress(Ctx, Di, Access.SizeInBytes, Access.alignment(),
                        modeledAccessCachePolicy(Ctx.Projection.SourceSTI));
  if (!Address)
    return Address.takeError();

  bool IsVolatile = isCoherentBeyondComputeUnit(Di);
  // An inactive lane holds an unconstrained address, so the load itself is
  // predicated and not only the register write it feeds.
  Ctx.registers().emitUnderExec([&] {
    Value *Loaded = Ctx.registers().emitMemoryValue([&] {
      return Ctx.B.CreateAlignedLoad(
          globalAccessType(Ctx.B, Access.SizeInBytes), *Address,
          Access.alignment(), IsVolatile, "global_load");
    });
    if (Access.SizeInBytes < 4) {
      Loaded = Access.DataFlags & GlobalAccess::Signed
                   ? Ctx.B.CreateSExt(Loaded, Ctx.B.getInt32Ty())
                   : Ctx.B.CreateZExt(Loaded, Ctx.B.getInt32Ty());
    }
    Ctx.registers().regFile().writeRegVec(Ctx.B, *Destination, Loaded);
  });
  return Error::success();
}

/// Emit a GLOBAL store under EXEC, selecting the requested source bits.
static Error emitGlobalStore(RaiseContext &Ctx, const DecodedInst &Di,
                             const GlobalAccess &Access) {
  Expected<ParsedReg> DataReg = globalDataReg(Ctx, Di, AMDGPU::OpName::vdata,
                                              Access.widthInDwords(), "source");
  if (!DataReg)
    return DataReg.takeError();
  Value *Data = Ctx.registers().regFile().readRegVec(
      Ctx.B, *DataReg, globalAccessType(Ctx.B, Access.widthInDwords() * 4));
  if (Access.DataFlags & GlobalAccess::HighHalf)
    Data = Ctx.B.CreateLShr(Data, 16);
  if (Access.SizeInBytes < 4)
    Data = Ctx.B.CreateTrunc(Data, globalAccessType(Ctx.B, Access.SizeInBytes));

  Expected<Value *> Address =
      emitGlobalAddress(Ctx, Di, Access.SizeInBytes, Access.alignment(),
                        modeledAccessCachePolicy(Ctx.Projection.SourceSTI));
  if (!Address)
    return Address.takeError();

  bool IsVolatile = isCoherentBeyondComputeUnit(Di);
  // A store by an inactive lane must not reach memory at all, so the whole
  // access is predicated on the lane bit of EXEC.
  Ctx.registers().emitUnderExec([&] {
    Ctx.registers().emitMemoryEffect([&] {
      Ctx.B.CreateAlignedStore(Data, *Address, Access.alignment(), IsVolatile);
    });
  });
  return Error::success();
}

/// Return the cache-policy bits a GLOBAL atomic may carry that the emitted
/// atomicrmw already accounts for. The returning flag is part of the opcode,
/// and the temporal and scope hints only relax guarantees a sequentially
/// consistent system-scope atomic already makes. GFX12 also accepts NV, which
/// the raised IR drops for the same reason a plain access does.
static unsigned modeledAtomicCachePolicy(const MCSubtargetInfo &STI) {
  if (STI.hasFeature(AMDGPU::FeatureGFX12Insts))
    return AMDGPU::CPol::TH_ATOMIC_RETURN | AMDGPU::CPol::TH_ATOMIC_NT |
           AMDGPU::CPol::SCOPE | AMDGPU::CPol::NV;
  return AMDGPU::CPol::GLC | AMDGPU::CPol::SLC | AMDGPU::CPol::SCC;
}

/// Emit a GLOBAL integer atomic add under EXEC, publishing the value the
/// memory held before the add to a destination register for the returning
/// form.
static Error emitGlobalAtomicAdd(RaiseContext &Ctx, const DecodedInst &Di,
                                 unsigned WidthInDwords, bool Returns) {
  assert(WidthInDwords == 1 && "only a 32-bit global atomic add is modeled");
  Expected<ParsedReg> DataReg = globalDataReg(Ctx, Di, AMDGPU::OpName::vdata,
                                              WidthInDwords, "atomic operand");
  if (!DataReg)
    return DataReg.takeError();
  Value *Data = Ctx.registers().regFile().readReg32(Ctx.B, *DataReg);

  std::optional<ParsedReg> Destination;
  if (Returns) {
    Expected<ParsedReg> Parsed = globalDataReg(Ctx, Di, AMDGPU::OpName::vdst,
                                               WidthInDwords, "destination");
    if (!Parsed)
      return Parsed.takeError();
    Destination = *Parsed;
  }

  // Unlike a plain access, an atomic is only well defined at the natural
  // alignment of the value it operates on.
  Align NaturalAlignment = Align(WidthInDwords * 4);
  Expected<Value *> Address =
      emitGlobalAddress(Ctx, Di, WidthInDwords * 4, NaturalAlignment,
                        modeledAtomicCachePolicy(Ctx.Projection.SourceSTI));
  if (!Address)
    return Address.takeError();

  // An atomic issued by an inactive lane must not reach memory at all.
  Ctx.registers().emitUnderExec([&] {
    auto Emit = [&] {
      return Ctx.B.CreateAtomicRMW(AtomicRMWInst::Add, *Address, Data,
                                   NaturalAlignment,
                                   AtomicOrdering::SequentiallyConsistent);
    };
    if (Destination) {
      Value *Old = Ctx.registers().emitMemoryValue(Emit);
      Ctx.registers().regFile().writeReg32(Ctx.B, *Destination, Old);
    } else {
      Ctx.registers().emitMemoryEffect([&] { Emit(); });
    }
  });
  return Error::success();
}

Error handleVGLOBAL(RaiseContext &Ctx, const DecodedInst &Di,
                    OperandResolver &) {
  if (Di.CanonOp == CanonicalOp::GLOBAL_ATOMIC_ADD_U32 ||
      Di.CanonOp == CanonicalOp::GLOBAL_ATOMIC_ADD_RTN_U32)
    return emitGlobalAtomicAdd(Ctx, Di, /*WidthInDwords=*/1,
                               /*Returns=*/Di.CanonOp ==
                                   CanonicalOp::GLOBAL_ATOMIC_ADD_RTN_U32);

  std::optional<GlobalAccess> Access = globalAccess(Di.CanonOp);
  if (!Access)
    return unsupported(Ctx, Di, "unsupported flat memory operation");

  if (Access->DataFlags & GlobalAccess::Store)
    return emitGlobalStore(Ctx, Di, *Access);
  return emitGlobalLoad(Ctx, Di, *Access);
}

} // namespace COMGR::transpiler
