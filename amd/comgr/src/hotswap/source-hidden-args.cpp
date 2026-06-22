//===- source-hidden-args.cpp - Hotswap transpiler ------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "source-hidden-args.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/Support/ErrorHandling.h"

#include <optional>

using namespace llvm;

namespace COMGR::hotswap {
namespace {

// Offsets in the HSA AQL `hsa_kernel_dispatch_packet_t` as defined by the
// public HSA runtime header.  Do not use SI::KernelInputOffsets here: those are
// LLVM's kernel-input/implicit-buffer offsets (`NGROUPS`, `LOCAL_SIZE`), not
// the AQL dispatch-packet layout addressed by `llvm.amdgcn.dispatch.ptr`.
namespace DispatchPacket {
constexpr unsigned SetupOffset = 2;
constexpr unsigned SetupDimensionsMask = 0x3;
constexpr unsigned WorkgroupSizeXOffset = 4;
constexpr unsigned WorkgroupSizeYOffset = 6;
constexpr unsigned WorkgroupSizeZOffset = 8;
constexpr unsigned GridSizeXOffset = 12;
constexpr unsigned GridSizeYOffset = 16;
constexpr unsigned GridSizeZOffset = 20;

// Return the AQL dispatch-packet workgroup-size field offset for a dimension.
unsigned dispatchWorkgroupSizeOffset(unsigned Dim) {
  switch (Dim) {
  case 0:
    return WorkgroupSizeXOffset;
  case 1:
    return WorkgroupSizeYOffset;
  case 2:
    return WorkgroupSizeZOffset;
  default:
    report_fatal_error("invalid source hidden workgroup-size dimension");
  }
}

// Return the AQL dispatch-packet grid-size field offset for a dimension.
unsigned dispatchGridSizeOffset(unsigned Dim) {
  switch (Dim) {
  case 0:
    return GridSizeXOffset;
  case 1:
    return GridSizeYOffset;
  case 2:
    return GridSizeZOffset;
  default:
    report_fatal_error("invalid source hidden grid-size dimension");
  }
}
} // namespace DispatchPacket

// Emit llvm.amdgcn.dispatch.ptr for AQL packet-backed hidden args.
Value *dispatchPtr(SourceHiddenArgContext &Ctx) {
  Function *DispatchPtrFn =
      Intrinsic::getOrInsertDeclaration(&Ctx.M, Intrinsic::amdgcn_dispatch_ptr);
  return Ctx.B.CreateCall(DispatchPtrFn, {}, "dispatch_ptr");
}

// Load a zero-extended 16-bit field from the AQL dispatch packet.
Value *loadDispatchU16(SourceHiddenArgContext &Ctx, unsigned ByteOffset,
                       const Twine &Name) {
  Value *Ptr =
      Ctx.B.CreateConstInBoundsGEP1_32(Ctx.I8Ty, dispatchPtr(Ctx), ByteOffset);
  return Ctx.B.CreateZExt(Ctx.B.CreateLoad(Type::getInt16Ty(Ctx.C), Ptr, Name),
                          Ctx.I32Ty, Name + "_zext");
}

// Load a 32-bit field from the AQL dispatch packet.
Value *loadDispatchU32(SourceHiddenArgContext &Ctx, unsigned ByteOffset,
                       const Twine &Name) {
  Value *Ptr =
      Ctx.B.CreateConstInBoundsGEP1_32(Ctx.I8Ty, dispatchPtr(Ctx), ByteOffset);
  return Ctx.B.CreateLoad(Ctx.I32Ty, Ptr, Name);
}

// Emit source hidden_group_size_{x,y,z}.
Value *emitDispatchWorkgroupSize(SourceHiddenArgContext &Ctx, unsigned Dim) {
  return loadDispatchU16(Ctx, DispatchPacket::dispatchWorkgroupSizeOffset(Dim),
                         Twine("source_hidden_wg_size_") + Twine(Dim));
}

// Emit source grid size for hidden block-count/remainder calculations.
Value *emitDispatchGridSize(SourceHiddenArgContext &Ctx, unsigned Dim) {
  return loadDispatchU32(Ctx, DispatchPacket::dispatchGridSizeOffset(Dim),
                         Twine("source_hidden_grid_size_") + Twine(Dim));
}

// Emit source hidden_block_count_{x,y,z}.
Value *emitHiddenBlockCount(SourceHiddenArgContext &Ctx, unsigned Dim) {
  return Ctx.B.CreateUDiv(emitDispatchGridSize(Ctx, Dim),
                          emitDispatchWorkgroupSize(Ctx, Dim),
                          Twine("source_hidden_block_count_") + Twine(Dim));
}

// Emit source hidden_remainder_{x,y,z}.
Value *emitHiddenRemainder(SourceHiddenArgContext &Ctx, unsigned Dim) {
  return Ctx.B.CreateURem(emitDispatchGridSize(Ctx, Dim),
                          emitDispatchWorkgroupSize(Ctx, Dim),
                          Twine("source_hidden_remainder_") + Twine(Dim));
}

// Emit source hidden_grid_dims from the AQL setup field.
Value *emitGridDims(SourceHiddenArgContext &Ctx) {
  return Ctx.B.CreateAnd(
      loadDispatchU16(Ctx, DispatchPacket::SetupOffset, "dispatch_setup"),
      Ctx.B.getInt32(DispatchPacket::SetupDimensionsMask),
      "source_hidden_grid_dims");
}

// Return a matched failure for hidden kinds without source-ABI synthesis.
SourceHiddenArgValue unsupportedHiddenKind(StringRef Kind) {
  SourceHiddenArgValue Result;
  Result.Matched = true;
  Result.FailureDetail =
      (Twine("unsupported source hidden argument kind '") + Kind +
       "'; add explicit source-ABI synthesis instead of falling back to "
       "target implicitarg layout")
          .str();
  return Result;
}

// Emit the full source hidden argument value for one metadata kind.
SourceHiddenArgValue emitHiddenArgValue(SourceHiddenArgContext &Ctx,
                                        SourceHiddenArgKind Kind) {
  SourceHiddenArgValue Result;
  Result.Matched = true;
  if (Kind == SourceHiddenArgKind::HiddenBlockCountX)
    Result.Value = emitHiddenBlockCount(Ctx, 0);
  else if (Kind == SourceHiddenArgKind::HiddenBlockCountY)
    Result.Value = emitHiddenBlockCount(Ctx, 1);
  else if (Kind == SourceHiddenArgKind::HiddenBlockCountZ)
    Result.Value = emitHiddenBlockCount(Ctx, 2);
  else if (Kind == SourceHiddenArgKind::HiddenGroupSizeX)
    Result.Value = emitDispatchWorkgroupSize(Ctx, 0);
  else if (Kind == SourceHiddenArgKind::HiddenGroupSizeY)
    Result.Value = emitDispatchWorkgroupSize(Ctx, 1);
  else if (Kind == SourceHiddenArgKind::HiddenGroupSizeZ)
    Result.Value = emitDispatchWorkgroupSize(Ctx, 2);
  else if (Kind == SourceHiddenArgKind::HiddenRemainderX)
    Result.Value = emitHiddenRemainder(Ctx, 0);
  else if (Kind == SourceHiddenArgKind::HiddenRemainderY)
    Result.Value = emitHiddenRemainder(Ctx, 1);
  else if (Kind == SourceHiddenArgKind::HiddenRemainderZ)
    Result.Value = emitHiddenRemainder(Ctx, 2);
  else if (Kind == SourceHiddenArgKind::HiddenGridDims)
    Result.Value = emitGridDims(Ctx);
  else if (Kind == SourceHiddenArgKind::HiddenGlobalOffsetX ||
           Kind == SourceHiddenArgKind::HiddenGlobalOffsetY ||
           Kind == SourceHiddenArgKind::HiddenGlobalOffsetZ) {
    if (!Ctx.AssumeHipGlobalOffsetZero)
      return unsupportedHiddenKind("hidden_global_offset_{x,y,z}");
    // The HotSwap runtime path intercepts HIP-launched kernels. HIP's launch
    // APIs do not expose a non-zero HSA grid-global offset, so the source ABI's
    // hidden_global_offset fields are the all-zero 64-bit value.
    Result.Value = Ctx.B.getInt64(0);
  } else
    return unsupportedHiddenKind("<unknown>");
  return Result;
}

// Emit one byte from the source hidden-argument metadata view.
SourceHiddenArgValue emitSourceHiddenByte(SourceHiddenArgContext &Ctx,
                                          int ByteOffset) {
  std::optional<SourceHiddenArgByte> Byte =
      classifySourceHiddenArgByte(Ctx.Args, ByteOffset);
  if (!Byte)
    return {};

  SourceHiddenArgValue Result = emitHiddenArgValue(Ctx, Byte->Kind);
  if (!Result.Value && !Byte->ValueKind.empty())
    Result.FailureDetail =
        (Twine("unsupported source hidden argument kind '") + Byte->ValueKind +
         "'; add explicit source-ABI synthesis instead of falling back to "
         "target implicitarg layout")
            .str();
  if (!Result.Value)
    return Result;

  Value *Wide = Ctx.B.CreateZExtOrTrunc(Result.Value, Ctx.I64Ty, "hidden_wide");
  unsigned ByteInArg = Byte->byteIndexInArg();
  if (ByteInArg != 0)
    Wide = Ctx.B.CreateLShr(Wide, Ctx.B.getInt64(ByteInArg * 8),
                            "hidden_byte_shift");
  Result.Value = Ctx.B.CreateTrunc(Wide, Ctx.I8Ty, "source_hidden_byte");
  return Result;
}

} // namespace

SourceHiddenArgValue emitSourceHiddenInteger(SourceHiddenArgContext &Ctx,
                                             int ByteOffset,
                                             unsigned ByteWidth,
                                             bool IsSigned) {
  if (ByteWidth != 1 && ByteWidth != 2 && ByteWidth != 4)
    report_fatal_error("unsupported source hidden integer byte width");

  SourceHiddenArgValue Result;
  Value *Acc = Ctx.B.getInt32(0);
  for (unsigned I = 0; I < ByteWidth; ++I) {
    SourceHiddenArgValue Byte =
        emitSourceHiddenByte(Ctx, ByteOffset + static_cast<int>(I));
    if (!Byte.Matched) {
      if (I == 0)
        return {};
      Result.Matched = true;
      Result.FailureDetail =
          (Twine("source hidden dword at byte offset ") + Twine(ByteOffset) +
           " spans non-hidden byte " + Twine(ByteOffset + static_cast<int>(I)))
              .str();
      return Result;
    }
    if (!Byte.Value)
      return Byte;

    Result.Matched = true;
    Value *Part =
        Ctx.B.CreateZExt(Byte.Value, Ctx.I32Ty, "source_hidden_byte_zext");
    if (I != 0)
      Part = Ctx.B.CreateShl(Part, Ctx.B.getInt32(I * 8),
                             "source_hidden_byte_place");
    Acc = Ctx.B.CreateOr(Acc, Part, "source_hidden_dword");
  }
  if (IsSigned && ByteWidth < 4) {
    Type *NarrowTy = Type::getIntNTy(Ctx.C, ByteWidth * 8);
    Result.Value =
        Ctx.B.CreateSExt(Ctx.B.CreateTrunc(Acc, NarrowTy, "source_hidden_narrow"),
                         Ctx.I32Ty, "source_hidden_sext");
  } else {
    Result.Value = Acc;
  }
  return Result;
}

SourceHiddenArgValue emitSourceHiddenDword(SourceHiddenArgContext &Ctx,
                                           int ByteOffset) {
  return emitSourceHiddenInteger(Ctx, ByteOffset, /*ByteWidth=*/4,
                                 /*IsSigned=*/false);
}

} // namespace COMGR::hotswap
