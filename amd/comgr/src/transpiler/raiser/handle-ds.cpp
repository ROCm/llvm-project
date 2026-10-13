//===- handle-ds.cpp - Transpiler ---------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "transpiler/raiser/handlers.h"

#include "transpiler/decoder/amdgpu-mc-tables.h"
#include "transpiler/decoder/canonical-op.h"
#include "transpiler/decoder/decoded-inst.h"
#include "transpiler/decoder/parsed-reg.h"
#include "transpiler/raiser/raise-context.h"
#include "transpiler/raiser/reg-file.h"

#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "Utils/AMDGPUBaseInfo.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/AMDGPUAddrSpace.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/TargetParser/AMDGPUTargetParser.h"

#include <cassert>
#include <cstdint>
#include <optional>

using namespace llvm;

namespace COMGR::transpiler {

namespace {

// Bit position at which the upper half of a 32-bit register begins.
constexpr unsigned KHighHalfShift = 16;

/// Shape of a DS memory access: how wide it is, how many addresses it names,
/// and how the immediate offset fields attached to those addresses are scaled.
struct DSAccess {
  /// Width of one memory access.
  unsigned MemBits;
  /// Number of independently addressed accesses the instruction performs.
  unsigned NumAccesses;
  /// Byte scale applied to every immediate offset field.
  unsigned OffsetScale;
  bool IsStore;
  /// Whether a sub-dword load sign-extends rather than zero-extends.
  bool IsSigned;
  /// Whether a store sources the upper half of its data register.
  bool IsHighHalf;

  /// Number of whole dwords one access transfers, or zero when the access is
  /// narrower than a dword.
  unsigned widthInDwords() const { return MemBits / 32; }
  /// Number of dwords one data register spans, which stays one when the access
  /// is narrower than a dword.
  unsigned registerWidthInDwords() const {
    return isSubDword() ? 1 : widthInDwords();
  }
  /// Whether the access is narrower than a dword.
  bool isSubDword() const { return MemBits < 32; }
};

} // namespace

/// Describe a load naming one address, whose offset field is a byte count.
static DSAccess directLoad(unsigned MemBits, bool IsSigned = false) {
  return DSAccess{MemBits,           /*NumAccesses=*/1, /*OffsetScale=*/1,
                  /*IsStore=*/false, IsSigned,          /*IsHighHalf=*/false};
}

/// Describe a store naming one address, whose offset field is a byte count.
static DSAccess directStore(unsigned MemBits, bool IsHighHalf = false) {
  return DSAccess{MemBits,          /*NumAccesses=*/1,  /*OffsetScale=*/1,
                  /*IsStore=*/true, /*IsSigned=*/false, IsHighHalf};
}

/// Describe a load naming two addresses. Each offset field counts elements of
/// the access width, StrideInElements of them per step.
static DSAccess pairLoad(unsigned MemBits, unsigned StrideInElements) {
  return DSAccess{MemBits,
                  /*NumAccesses=*/2,
                  /*OffsetScale=*/(MemBits / 8) * StrideInElements,
                  /*IsStore=*/false,
                  /*IsSigned=*/false,
                  /*IsHighHalf=*/false};
}

/// Describe a store naming two addresses. Each offset field counts elements of
/// the access width, StrideInElements of them per step.
static DSAccess pairStore(unsigned MemBits, unsigned StrideInElements) {
  return DSAccess{MemBits,
                  /*NumAccesses=*/2,
                  /*OffsetScale=*/(MemBits / 8) * StrideInElements,
                  /*IsStore=*/true,
                  /*IsSigned=*/false,
                  /*IsHighHalf=*/false};
}

/// Return the access shape of a DS load or store, or nullopt for a DS operation
/// that is not a plain memory access.
static std::optional<DSAccess> dsAccess(CanonicalOp CanonOp) {
  switch (CanonOp) {
  case CanonicalOp::DS_LOAD_U8:
    return directLoad(8);
  case CanonicalOp::DS_LOAD_I8:
    return directLoad(8, /*IsSigned=*/true);
  case CanonicalOp::DS_LOAD_U16:
    return directLoad(16);
  case CanonicalOp::DS_LOAD_I16:
    return directLoad(16, /*IsSigned=*/true);
  case CanonicalOp::DS_LOAD_B32:
    return directLoad(32);
  case CanonicalOp::DS_LOAD_B64:
    return directLoad(64);
  case CanonicalOp::DS_LOAD_B96:
    return directLoad(96);
  case CanonicalOp::DS_LOAD_B128:
    return directLoad(128);
  case CanonicalOp::DS_STORE_B8:
    return directStore(8);
  case CanonicalOp::DS_STORE_B16:
    return directStore(16);
  case CanonicalOp::DS_STORE_B32:
    return directStore(32);
  case CanonicalOp::DS_STORE_B64:
    return directStore(64);
  case CanonicalOp::DS_STORE_B96:
    return directStore(96);
  case CanonicalOp::DS_STORE_B128:
    return directStore(128);
  case CanonicalOp::DS_STORE_B8_D16_HI:
    return directStore(8, /*IsHighHalf=*/true);
  case CanonicalOp::DS_STORE_B16_D16_HI:
    return directStore(16, /*IsHighHalf=*/true);
  case CanonicalOp::DS_LOAD_2ADDR_B32:
    return pairLoad(32, /*StrideInElements=*/1);
  case CanonicalOp::DS_LOAD_2ADDR_B64:
    return pairLoad(64, /*StrideInElements=*/1);
  case CanonicalOp::DS_LOAD_2ADDR_STRIDE64_B32:
    return pairLoad(32, /*StrideInElements=*/64);
  case CanonicalOp::DS_LOAD_2ADDR_STRIDE64_B64:
    return pairLoad(64, /*StrideInElements=*/64);
  case CanonicalOp::DS_STORE_2ADDR_B32:
    return pairStore(32, /*StrideInElements=*/1);
  case CanonicalOp::DS_STORE_2ADDR_B64:
    return pairStore(64, /*StrideInElements=*/1);
  case CanonicalOp::DS_STORE_2ADDR_STRIDE64_B32:
    return pairStore(32, /*StrideInElements=*/64);
  case CanonicalOp::DS_STORE_2ADDR_STRIDE64_B64:
    return pairStore(64, /*StrideInElements=*/64);
  default:
    return std::nullopt;
  }
}

/// Return whether STI describes a gfx1250 target.
static bool isGFX1250(const MCSubtargetInfo &STI) {
  return STI.getFeatureBits()[AMDGPU::FeatureGFX1250Insts] &&
         !STI.getFeatureBits()[AMDGPU::FeatureGFX13];
}

/// Return the index of a required named operand of a DS instruction.
static unsigned dsOperandIndex(const DecodedInst &Instruction,
                               AMDGPU::OpName Name) {
  int Index =
      COMGR::transpiler::getNamedOperandIdx(Instruction.Inst.getOpcode(), Name);
  assert(Index >= 0 &&
         static_cast<unsigned>(Index) < Instruction.numOperands() &&
         "DS instruction is missing a required operand");
  return Index;
}

/// Return the value of a DS immediate offset field.
static int64_t dsOffsetField(const DecodedInst &Instruction,
                             AMDGPU::OpName Name) {
  unsigned Index = dsOperandIndex(Instruction, Name);
  assert(Instruction.isImm(Index) && "DS offset must be an immediate");
  int64_t Offset = Instruction.getImm(Index);
  assert(Offset >= 0 && Offset <= UINT16_MAX &&
         "DS offset must fit its unsigned field");
  return Offset;
}

/// Return the name of the immediate offset field belonging to the access at
/// AccessIndex. An instruction naming one address spells the field differently
/// from one naming two.
static AMDGPU::OpName dsOffsetName(const DSAccess &Access,
                                   unsigned AccessIndex) {
  if (Access.NumAccesses == 1) {
    return AMDGPU::OpName::offset;
  }
  return AccessIndex == 0 ? AMDGPU::OpName::offset0 : AMDGPU::OpName::offset1;
}

/// Return the name of the data register belonging to the access at AccessIndex.
static AMDGPU::OpName dsDataName(unsigned AccessIndex) {
  return AccessIndex == 0 ? AMDGPU::OpName::data0 : AMDGPU::OpName::data1;
}

/// Parse a DS register operand and check that it is a VGPR tuple of the
/// expected width.
static Expected<ParsedReg> dsRegister(RaiseContext &Context,
                                      const DecodedInst &Instruction,
                                      AMDGPU::OpName Name,
                                      unsigned WidthInDwords, StringRef Role) {
  Expected<ParsedReg> Reg = Context.registers().parseReg(
      Instruction, dsOperandIndex(Instruction, Name));
  if (!Reg) {
    return Reg.takeError();
  }
  if (Reg->RegKind != ParsedReg::VGPR) {
    return unsupported(Context, Instruction,
                       Twine("DS ") + Role + " must be a VGPR");
  }
  assert(Reg->BaseIdx && "DS register operand has no base index");
  assert(Reg->WidthInDwords == WidthInDwords &&
         "DS register width does not match the opcode");
  return *Reg;
}

/// Return the type holding the given number of consecutive dwords.
static Type *accessType(IRBuilder<> &B, unsigned WidthInDwords) {
  if (WidthInDwords == 1) {
    return B.getInt32Ty();
  }
  if (WidthInDwords == 2) {
    return B.getInt64Ty();
  }
  return FixedVectorType::get(B.getInt32Ty(), WidthInDwords);
}

/// Fold a byte offset into an address, freezing the result so that an inactive
/// lane's unconstrained address cannot reach the access as poison.
static Value *emitLdsByteAddress(RaiseContext &Context, Value *Address,
                                 int64_t OffsetInBytes) {
  return Context.freezeMemAddr(Context.B.CreateAdd(
      Address, Context.B.getInt32(OffsetInBytes), "lds_address"));
}

/// Materialize an LDS pointer. Emitted next to the access it feeds, inside
/// whatever EXEC predication guards it.
static Value *emitLdsPointer(RaiseContext &Context, Value *ByteAddress) {
  return Context.B.CreateIntToPtr(
      ByteAddress,
      PointerType::get(Context.B.getContext(), AMDGPUAS::LOCAL_ADDRESS),
      "lds_ptr");
}

/// Substitute zero for the data an inactive lane contributes, which is what
/// the source hardware returns to a lane that reads an inactive one.
static Value *zeroInactiveLaneData(RaiseContext &Context, Value *Data) {
  Value *Active = Context.registers().emitLaneActiveBit();
  return Context.B.CreateSelect(
      Active, Data, Constant::getNullValue(Data->getType()), "permute_data");
}

/// Write the result of a cross-lane read back to Destination, out of
/// whole-wave mode: every target lane has to run the read for a lane to see
/// the lane its selector named.
static void writePermuteResult(RaiseContext &Context, ParsedReg Destination,
                               Value *Permuted) {
  Value *WholeWave =
      Context.Projection.wrapAsWWMValue(Context.B, Permuted, "permute_wwm");
  Context.registers().writeReg32(Destination, WholeWave);
}

/// Gather eight elements per lane from LDS and pack them into the destination.
static void emitTransposedDSLoad(RaiseContext &Context, ParsedReg Destination,
                                 Value *ByteAddress, unsigned ElementBits) {
  IRBuilder<> &B = Context.B;
  Value *Lane = Context.emitLaneIdx();
  Value *ElementOffset = B.CreateMul(B.CreateAnd(Lane, B.getInt32(7)),
                                     B.getInt32(ElementBits / 8));
  Value *SourceBase =
      B.CreateAnd(Lane, B.getInt32(ElementBits == 8 ? ~15 : ~7));
  if (ElementBits == 8) {
    Value *Half = B.CreateAnd(B.CreateLShr(Lane, 1), B.getInt32(4));
    SourceBase = B.CreateOr(SourceBase, Half);
  }

  // A nonzero source EXEC makes every lane generate an address and write back.
  Context.registers().emitWithNonzeroExec([&] {
    const unsigned ElementsPerDword = 32 / ElementBits;
    Type *ResultType =
        FixedVectorType::get(B.getInt32Ty(), Destination.WidthInDwords);
    Value *Result = PoisonValue::get(ResultType);
    for (unsigned I = 0; I < Destination.WidthInDwords; ++I) {
      Value *Word = B.getInt32(0);
      for (unsigned J = 0; J < ElementsPerDword; ++J) {
        const unsigned SourceOffset = I * (ElementBits == 8 ? 8 : 2) + J;
        Value *SourceLane = B.CreateAdd(SourceBase, B.getInt32(SourceOffset));
        Value *Index = B.CreateMul(SourceLane, B.getInt32(4));
        Value *Address = B.CreateIntrinsic(Intrinsic::amdgcn_ds_bpermute, {},
                                           {Index, ByteAddress});
        Address = B.CreateAdd(Address, ElementOffset);
        Value *Pointer = B.CreateIntToPtr(
            Address, PointerType::get(B.getContext(), AMDGPUAS::LOCAL_ADDRESS));
        Value *Element = Context.registers().emitMemoryValue([&] {
          return B.CreateAlignedLoad(B.getIntNTy(ElementBits), Pointer,
                                     Align(1));
        });
        Element = B.CreateZExt(Element, B.getInt32Ty());
        Word = B.CreateOr(Word, B.CreateShl(Element, J * ElementBits));
      }
      Result = B.CreateInsertElement(Result, Word, I);
    }
    Context.registers().regFile().writeRegVec(B, Destination, Result);
  });
}

/// Return the value a store sends to memory: the data register narrowed to the
/// access width.
static Value *emitStoredValue(RaiseContext &Context, const DSAccess &Access,
                              ParsedReg Data) {
  if (!Access.isSubDword()) {
    return Context.registers().regFile().readRegVec(
        Context.B, Data, accessType(Context.B, Access.widthInDwords()));
  }
  Value *Whole = Context.registers().regFile().readReg32(Context.B, Data);
  if (Access.IsHighHalf) {
    // The D16_HI forms transfer the register's upper half, so an 8-bit store
    // writes the low byte of that half rather than the low byte of the
    // register.
    assert(Access.MemBits <= KHighHalfShift &&
           "a high-half store cannot be wider than the half it names");
    Whole = Context.B.CreateLShr(Whole, Context.B.getInt32(KHighHalfShift),
                                 "lds_store_hi");
  }
  return Context.B.CreateTrunc(Whole, Context.B.getIntNTy(Access.MemBits),
                               "lds_store");
}

/// Emit a DS load or store. The accesses an instruction names are independent,
/// so each gets its own address and its own memory operation.
static Error emitAccess(RaiseContext &Context, const DecodedInst &Instruction,
                        const DSAccess &Access, Value *Address) {
  assert((!Access.isSubDword() || Access.NumAccesses == 1) &&
         "only a single-address DS access can be narrower than a dword");

  // The source address is not known to be aligned, and understating alignment
  // is always safe, so claim no more than a byte.
  Align Unaligned(1);

  SmallVector<Value *, 2> ByteAddresses;
  for (unsigned I = 0; I != Access.NumAccesses; ++I) {
    ByteAddresses.push_back(
        emitLdsByteAddress(Context, Address,
                           dsOffsetField(Instruction, dsOffsetName(Access, I)) *
                               Access.OffsetScale));
  }

  if (Access.IsStore) {
    SmallVector<Value *, 2> Stored;
    for (unsigned I = 0; I != Access.NumAccesses; ++I) {
      Expected<ParsedReg> Data =
          dsRegister(Context, Instruction, dsDataName(I),
                     Access.registerWidthInDwords(), "store data");
      if (!Data) {
        return Data.takeError();
      }
      Stored.push_back(emitStoredValue(Context, Access, *Data));
    }
    // A store by an inactive lane must not reach memory at all, so the whole
    // access is predicated on the lane bit of EXEC.
    Context.registers().emitUnderExec([&] {
      for (unsigned I = 0; I != Access.NumAccesses; ++I) {
        Context.registers().emitMemoryEffect([&] {
          Context.B.CreateAlignedStore(
              Stored[I], emitLdsPointer(Context, ByteAddresses[I]), Unaligned);
        });
      }
    });
    return Error::success();
  }

  // A destination holds the loaded values back to back.
  Expected<ParsedReg> Destination = dsRegister(
      Context, Instruction, AMDGPU::OpName::vdst,
      Access.NumAccesses * Access.registerWidthInDwords(), "destination");
  if (!Destination) {
    return Destination.takeError();
  }

  // Inactive lanes can hold invalid addresses, so guard the load itself.
  Context.registers().emitUnderExec([&] {
    for (unsigned I = 0; I != Access.NumAccesses; ++I) {
      Value *Pointer = emitLdsPointer(Context, ByteAddresses[I]);
      if (Access.isSubDword()) {
        Value *Loaded = Context.registers().emitMemoryValue([&] {
          return Context.B.CreateAlignedLoad(
              Context.B.getIntNTy(Access.MemBits), Pointer, Unaligned,
              "lds_load");
        });
        Context.registers().regFile().writeReg32(
            Context.B, *Destination,
            Access.IsSigned
                ? Context.B.CreateSExt(Loaded, Context.B.getInt32Ty())
                : Context.B.CreateZExt(Loaded, Context.B.getInt32Ty()));
        continue;
      }
      Value *Loaded = Context.registers().emitMemoryValue([&] {
        return Context.B.CreateAlignedLoad(
            accessType(Context.B, Access.widthInDwords()), Pointer, Unaligned,
            "lds_load");
      });
      ParsedReg Part = *Destination;
      Part.BaseIdx = *Destination->BaseIdx + I * Access.widthInDwords();
      Part.WidthInDwords = static_cast<uint8_t>(Access.widthInDwords());
      Context.registers().regFile().writeRegVec(Context.B, Part, Loaded);
    }
  });
  return Error::success();
}

/// Emit an LDS integer atomic add of the given width, publishing the pre-add
/// value to a destination register for the returning form.
static Error emitAtomicAdd(RaiseContext &Context,
                           const DecodedInst &Instruction, unsigned MemBits,
                           bool Returns, Value *Address) {
  assert(MemBits == 32 && "only a 32-bit LDS atomic add is modeled");
  unsigned WidthInDwords = MemBits / 32;
  Expected<ParsedReg> Data =
      dsRegister(Context, Instruction, AMDGPU::OpName::data0, WidthInDwords,
                 "atomic operand");
  if (!Data) {
    return Data.takeError();
  }
  std::optional<ParsedReg> Destination;
  if (Returns) {
    Expected<ParsedReg> Parsed =
        dsRegister(Context, Instruction, AMDGPU::OpName::vdst, WidthInDwords,
                   "destination");
    if (!Parsed) {
      return Parsed.takeError();
    }
    Destination = *Parsed;
  }

  Value *Operand = Context.registers().regFile().readReg32(Context.B, *Data);
  Value *ByteAddress = emitLdsByteAddress(
      Context, Address, dsOffsetField(Instruction, AMDGPU::OpName::offset));
  // An atomic issued by an inactive lane must not reach memory at all.
  Context.registers().emitUnderExec([&] {
    // Unlike a plain access, an atomic is only well defined at the natural
    // alignment of the value it operates on.
    auto Emit = [&] {
      return Context.B.CreateAtomicRMW(
          AtomicRMWInst::Add, emitLdsPointer(Context, ByteAddress), Operand,
          Align(MemBits / 8), AtomicOrdering::SequentiallyConsistent);
    };
    if (Destination) {
      Value *Old = Context.registers().emitMemoryValue(Emit);
      Context.registers().regFile().writeReg32(Context.B, *Destination, Old);
    } else {
      Context.registers().emitMemoryEffect([&] { Emit(); });
    }
  });
  return Error::success();
}

/// Stride a DS permute selector addresses lanes with: it names a lane by its
/// byte offset into an array of per-lane dwords.
static constexpr unsigned SelectorBytesPerLane = 4;

/// Rebase a lane selector onto the source wave the reading lane belongs to.
///
/// A widening projection packs several source waves into one target wave, so a
/// selector naming lane 0 has to reach lane 0 of the packed wave its lane
/// belongs to rather than lane 0 of the target wave. Masking the selector down
/// to the source wave's byte range keeps a selector the source kernel never
/// constrained from reaching a lane of another packed wave.
static Value *rebaseLaneSelector(RaiseContext &Context, Value *Selector) {
  const unsigned SourceWaveSize = Context.Projection.sourceWaveSize();
  if (SourceWaveSize == Context.Projection.targetWaveSize()) {
    return Selector;
  }
  assert(isPowerOf2_32(SourceWaveSize) &&
         "a lane index occupies whole bits only for a power-of-two wave size");
  const unsigned LaneIndexBits = Log2_32(SourceWaveSize);
  const unsigned LaneByteBits = Log2_32(SelectorBytesPerLane);

  IRBuilder<> &B = Context.B;
  Value *SelectorInWave = B.CreateAnd(
      Selector,
      B.getInt32(maskTrailingOnes<uint32_t>(LaneIndexBits + LaneByteBits)),
      "selector_in_wave");
  Value *LaneIdx = Context.emitLaneIdx();
  Value *WaveBase = B.CreateAnd(
      LaneIdx, B.getInt32(maskTrailingZeros<uint32_t>(LaneIndexBits)),
      "wave_base");
  Value *WaveByteBase = B.CreateShl(WaveBase, LaneByteBits, "wave_byte_base");
  return B.CreateOr(SelectorInWave, WaveByteBase, "selector");
}

namespace {
/// What a DS permute moves, and where.
struct PermuteOperands {
  /// Register the moved data lands in.
  ParsedReg Destination;
  /// Lane the data moves from or to, as a byte offset already biased by the
  /// offset field and rebased onto the source wave of the lane holding it.
  Value *Selector;
  /// Data the lane contributes to the move.
  Value *Data;
};
} // namespace

/// Read the operands common to the forward and backward DS permutes.
static Expected<PermuteOperands>
readPermuteOperands(RaiseContext &Context, const DecodedInst &Instruction) {
  Expected<ParsedReg> Destination =
      dsRegister(Context, Instruction, AMDGPU::OpName::vdst,
                 /*WidthInDwords=*/1, "destination");
  if (!Destination) {
    return Destination.takeError();
  }
  Expected<Value *> Selector = Context.registers().readOp32(
      Instruction, dsOperandIndex(Instruction, AMDGPU::OpName::addr));
  if (!Selector) {
    return Selector.takeError();
  }
  Expected<Value *> Data = Context.registers().readOp32(
      Instruction, dsOperandIndex(Instruction, AMDGPU::OpName::data0));
  if (!Data) {
    return Data.takeError();
  }
  // The hardware adds the offset field before dividing the selector into a
  // lane index; the intrinsic takes the sum.
  Value *Index = Context.B.CreateAdd(
      *Selector,
      Context.B.getInt32(dsOffsetField(Instruction, AMDGPU::OpName::offset)),
      "lane_selector");
  Value *Rebased = rebaseLaneSelector(Context, Index);
  return PermuteOperands{*Destination, Rebased, *Data};
}

/// Raise a backward permute: every lane reads the data operand out of the
/// source-wave lane its selector names. FetchInactive says whether the opcode
/// is the form that reads a lane the source left inactive.
static Error raiseBackwardPermute(RaiseContext &Context,
                                  const DecodedInst &Instruction,
                                  bool FetchInactive) {
  Expected<PermuteOperands> Operands =
      readPermuteOperands(Context, Instruction);
  if (!Operands) {
    return Operands.takeError();
  }
  // The gather runs whole-wave, so the hardware hands over every target lane's
  // register either way. What separates the two opcodes is that the plain form
  // returns zero for a lane the source left inactive, which zeroing that
  // lane's contribution reproduces.
  Value *Data = FetchInactive ? Operands->Data
                              : zeroInactiveLaneData(Context, Operands->Data);
  Value *Gathered = Context.B.CreateIntrinsic(Intrinsic::amdgcn_ds_bpermute, {},
                                              {Operands->Selector, Data},
                                              nullptr, "bpermute");
  writePermuteResult(Context, Operands->Destination, Gathered);
  return Error::success();
}

/// Raise a forward permute: every lane scatters its data operand to the
/// source-wave lane its selector names, and a lane no other lane targets
/// reads zero.
static Error raiseForwardPermute(RaiseContext &Context,
                                 const DecodedInst &Instruction) {
  Expected<PermuteOperands> Operands =
      readPermuteOperands(Context, Instruction);
  if (!Operands) {
    return Operands.takeError();
  }
  // The scatter runs whole-wave, so a lane the source left inactive has to
  // target itself. Keeping the selector it happens to hold would overwrite the
  // result of whichever lane that selector names, while targeting itself takes
  // nothing from an active lane and leaves a result the write back drops.
  IRBuilder<> &B = Context.B;
  Value *LaneIdx = Context.emitLaneIdx();
  Value *OwnLaneSelector =
      B.CreateShl(LaneIdx, Log2_32(SelectorBytesPerLane), "own_lane_selector");
  Value *Active = Context.registers().emitLaneActiveBit();
  Value *Selector = B.CreateSelect(Active, Operands->Selector, OwnLaneSelector,
                                   "permute_selector");
  // The rebase keeps every collision inside one source wave, in the lane order
  // the source had, so a contested destination resolves the way it did there.
  Value *Scattered =
      B.CreateIntrinsic(Intrinsic::amdgcn_ds_permute, {},
                        {Selector, Operands->Data}, nullptr, "permute");
  writePermuteResult(Context, Operands->Destination, Scattered);
  return Error::success();
}

/// Raise a dword swizzle, a fixed lane permutation the offset field selects.
///
/// Every swizzle mode permutes within a group of at most 32 lanes and leaves
/// bit 5 of the lane index alone, so a target wave permutes each source wave
/// it packs on its own and the pattern carries over unchanged. The GDS operand
/// goes unread: the swizzle reaches no memory for the bit to choose between.
static Error raiseSwizzle(RaiseContext &Context,
                          const DecodedInst &Instruction) {
  Expected<ParsedReg> Destination =
      dsRegister(Context, Instruction, AMDGPU::OpName::vdst,
                 /*WidthInDwords=*/1, "destination");
  if (!Destination) {
    return Destination.takeError();
  }
  // The data arrives in the address operand: the swizzle takes the
  // single-address DS encoding without addressing anything with it.
  Expected<Value *> Data = Context.registers().readOp32(
      Instruction, dsOperandIndex(Instruction, AMDGPU::OpName::addr));
  if (!Data) {
    return Data.takeError();
  }

  IRBuilder<> &B = Context.B;
  Value *Swizzled = zeroInactiveLaneData(Context, *Data);
  Swizzled = B.CreateIntrinsic(
      Intrinsic::amdgcn_ds_swizzle, {},
      {Swizzled,
       B.getInt32(dsOffsetField(Instruction, AMDGPU::OpName::offset))},
      nullptr, "swizzle");
  writePermuteResult(Context, *Destination, Swizzled);
  return Error::success();
}

Error handleDS(RaiseContext &Context, const DecodedInst &Instruction) {
  switch (Instruction.CanonOp) {
  case CanonicalOp::DS_BPERMUTE_B32:
    return raiseBackwardPermute(Context, Instruction, /*FetchInactive=*/false);
  case CanonicalOp::DS_BPERMUTE_FI_B32:
    return raiseBackwardPermute(Context, Instruction, /*FetchInactive=*/true);
  case CanonicalOp::DS_PERMUTE_B32:
    return raiseForwardPermute(Context, Instruction);
  case CanonicalOp::DS_SWIZZLE_B32:
    return raiseSwizzle(Context, Instruction);
  default:
    break;
  }

  std::optional<DSAccess> Access = dsAccess(Instruction.CanonOp);
  bool IsAtomicAdd = Instruction.CanonOp == CanonicalOp::DS_ADD_U32 ||
                     Instruction.CanonOp == CanonicalOp::DS_ADD_RTN_U32;

  // Width of the widest single memory operation the instruction performs, which
  // is what the misalignment refusal below keys on.
  unsigned WidthInDwords = 0;
  unsigned TransposeElementBits = 0;
  if (Access) {
    WidthInDwords = Access->widthInDwords();
  } else if (IsAtomicAdd) {
    WidthInDwords = 1;
  } else {
    switch (Instruction.CanonOp) {
    case CanonicalOp::DS_LOAD_TR8_B64:
      WidthInDwords = 2;
      TransposeElementBits = 8;
      break;
    case CanonicalOp::DS_LOAD_TR16_B128:
      WidthInDwords = 4;
      TransposeElementBits = 16;
      break;
    default:
      return unsupported(Context, Instruction, "unsupported DS operation");
    }
  }

  if (TransposeElementBits && (!isGFX1250(Context.Projection.SourceSTI) ||
                               Context.Projection.sourceWaveSize() != 32)) {
    return unsupported(Context, Instruction,
                       "DS transpose loads require a gfx1250 wave32 source");
  }

  if (AMDGPU::getIsaVersion(Context.MC.SubtargetInfo->getCPU()).Major < 9) {
    return unsupported(Context, Instruction,
                       "M0-bounded LDS accesses are not modeled");
  }
  // Source address alignment and CU mode are not established, so refuse wide
  // accesses on hardware affected by the WGP misalignment bug.
  if (WidthInDwords > 1 &&
      Context.MC.SubtargetInfo->hasFeature(AMDGPU::FeatureLDSMisalignedBug)) {
    return unsupported(Context, Instruction,
                       "wide LDS accesses with the WGP misalignment bug are "
                       "not modeled");
  }

  unsigned GDSIndex = dsOperandIndex(Instruction, AMDGPU::OpName::gds);
  assert(Instruction.isImm(GDSIndex) && "GDS operand must be an immediate");
  if (Instruction.getImm(GDSIndex)) {
    return unsupported(Context, Instruction, "GDS accesses are not modeled");
  }

  Expected<Value *> Address = Context.registers().readOp32(
      Instruction, dsOperandIndex(Instruction, AMDGPU::OpName::addr));
  if (!Address) {
    return Address.takeError();
  }

  if (Access) {
    return emitAccess(Context, Instruction, *Access, *Address);
  }
  if (IsAtomicAdd) {
    return emitAtomicAdd(Context, Instruction, /*MemBits=*/32,
                         Instruction.CanonOp == CanonicalOp::DS_ADD_RTN_U32,
                         *Address);
  }

  Expected<ParsedReg> Destination = dsRegister(
      Context, Instruction, AMDGPU::OpName::vdst, WidthInDwords, "destination");
  if (!Destination) {
    return Destination.takeError();
  }
  emitTransposedDSLoad(
      Context, *Destination,
      emitLdsByteAddress(Context, *Address,
                         dsOffsetField(Instruction, AMDGPU::OpName::offset)),
      TransposeElementBits);
  return Error::success();
}

} // namespace COMGR::transpiler
