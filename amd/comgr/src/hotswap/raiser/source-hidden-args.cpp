//===- source-hidden-args.cpp - Hotswap transpiler ------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "source-hidden-args.h"
#include "raise_failure.h"

#include "SIDefines.h"
#include "Utils/AMDGPUBaseInfo.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"

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
    llvm_unreachable("invalid source hidden workgroup-size dimension");
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
    llvm_unreachable("invalid source hidden grid-size dimension");
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

// The target backend initially gets "amdgpu-no-*" attrs for every hidden field
// so it does not invent unused target ABI inputs. When a source hidden arg has
// the same semantic value on the target ABI, remove only the attributes needed
// to make the target runtime populate that target ABI field.
void requireTargetImplicitArg(SourceHiddenArgContext &Ctx,
                              StringRef FieldNoAttr) {
  Ctx.Fn.removeFnAttr("amdgpu-no-implicitarg-ptr");
  Ctx.Fn.removeFnAttr(FieldNoAttr);
}

// Emit a pointer-sized source hidden argument by reading the corresponding
// target ABI field. Offsets are relative to llvm.amdgcn.implicitarg.ptr in the
// target code-object version; source metadata offsets are deliberately ignored.
Value *loadTargetHiddenPointer(SourceHiddenArgContext &Ctx,
                               unsigned TargetByteOffset, StringRef FieldNoAttr,
                               const Twine &Name) {
  requireTargetImplicitArg(Ctx, FieldNoAttr);
  Function *FnImplicitArgPtr = Intrinsic::getOrInsertDeclaration(
      &Ctx.M, Intrinsic::amdgcn_implicitarg_ptr);
  Value *Ptr = Ctx.B.CreateCall(FnImplicitArgPtr, {}, "target_implicitarg_ptr");
  if (TargetByteOffset != 0)
    Ptr = Ctx.B.CreateConstInBoundsGEP1_32(Ctx.I8Ty, Ptr, TargetByteOffset,
                                           Name + "_ptr");
  return Ctx.B.CreateAlignedLoad(Ctx.I64Ty, Ptr, Align(8), Name);
}

// Divide an x-dimension size read by the scaled-replication factor so the
// source kernel observes the un-scaled (logical) size. The hardware size is an
// exact multiple of the factor (the runtime scales it), so an unsigned shift is
// exact. No-op for non-x dimensions and for non-replicated kernels. The factor
// is a power of two.
Value *virtualizeScaledReplicationSize(SourceHiddenArgContext &Ctx,
                                       unsigned Dim, Value *Size,
                                       const Twine &Name) {
  if (Dim != 0 || Ctx.ScaledReplicationFactor <= 1)
    return Size;
  unsigned ShiftBy = llvm::Log2_32(Ctx.ScaledReplicationFactor);
  return Ctx.B.CreateLShr(Size, ConstantInt::get(Size->getType(), ShiftBy),
                          Name + "_descaled");
}

// Emit source hidden_group_size_{x,y,z}.
Value *emitDispatchWorkgroupSize(SourceHiddenArgContext &Ctx, unsigned Dim) {
  Value *Size =
      loadDispatchU16(Ctx, DispatchPacket::dispatchWorkgroupSizeOffset(Dim),
                      Twine("source_hidden_wg_size_") + Twine(Dim));
  return virtualizeScaledReplicationSize(
      Ctx, Dim, Size, Twine("source_hidden_wg_size_") + Twine(Dim));
}

// Emit source grid size for hidden block-count/remainder calculations.
Value *emitDispatchGridSize(SourceHiddenArgContext &Ctx, unsigned Dim) {
  Value *Size =
      loadDispatchU32(Ctx, DispatchPacket::dispatchGridSizeOffset(Dim),
                      Twine("source_hidden_grid_size_") + Twine(Dim));
  return virtualizeScaledReplicationSize(
      Ctx, Dim, Size, Twine("source_hidden_grid_size_") + Twine(Dim));
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

// Build the failure for a hidden kind that lacks source-ABI synthesis.
Error unsupportedHiddenKind(uint64_t ByteOffset, StringRef Kind) {
  return RaiseFailure::atInstruction(
      RaiseFailureReason::UnsupportedSourceHiddenArg, "<source-hidden-arg>",
      ByteOffset, "hidden-arg",
      ("unsupported source hidden argument kind '" + Kind +
       "'; add explicit source-ABI synthesis instead of falling back to "
       "target implicitarg layout")
          .str());
}

// Emit the full source hidden argument value for one metadata kind, or an Error
// when that kind has no source-side synthesis. `ByteOffset` locates the field
// for diagnostics.
Expected<Value *> emitHiddenArgValue(SourceHiddenArgContext &Ctx,
                                     SourceHiddenArgKind Kind,
                                     uint64_t ByteOffset) {
  switch (Kind) {
  case SourceHiddenArgKind::HiddenBlockCountX:
    return emitHiddenBlockCount(Ctx, 0);
  case SourceHiddenArgKind::HiddenBlockCountY:
    return emitHiddenBlockCount(Ctx, 1);
  case SourceHiddenArgKind::HiddenBlockCountZ:
    return emitHiddenBlockCount(Ctx, 2);
  case SourceHiddenArgKind::HiddenGroupSizeX:
    return emitDispatchWorkgroupSize(Ctx, 0);
  case SourceHiddenArgKind::HiddenGroupSizeY:
    return emitDispatchWorkgroupSize(Ctx, 1);
  case SourceHiddenArgKind::HiddenGroupSizeZ:
    return emitDispatchWorkgroupSize(Ctx, 2);
  case SourceHiddenArgKind::HiddenRemainderX:
    return emitHiddenRemainder(Ctx, 0);
  case SourceHiddenArgKind::HiddenRemainderY:
    return emitHiddenRemainder(Ctx, 1);
  case SourceHiddenArgKind::HiddenRemainderZ:
    return emitHiddenRemainder(Ctx, 2);
  case SourceHiddenArgKind::HiddenGridDims:
    return emitGridDims(Ctx);
  case SourceHiddenArgKind::HiddenGlobalOffsetX:
  case SourceHiddenArgKind::HiddenGlobalOffsetY:
  case SourceHiddenArgKind::HiddenGlobalOffsetZ:
    if (!Ctx.AssumeHipGlobalOffsetZero)
      return unsupportedHiddenKind(ByteOffset, "hidden_global_offset_{x,y,z}");
    // The HotSwap runtime path intercepts HIP-launched kernels. HIP's launch
    // APIs do not expose a non-zero HSA grid-global offset, so the source ABI's
    // hidden_global_offset fields are the all-zero 64-bit value.
    return Ctx.B.getInt64(0);
  case SourceHiddenArgKind::HiddenPrivateBase:
    // Private/shared bases are real aperture state. Do not synthesize them
    // until the translator has a target-capability proof that the source read
    // is either unused or exactly reconstructed elsewhere.
    return unsupportedHiddenKind(ByteOffset, "hidden_private_base");
  case SourceHiddenArgKind::HiddenSharedBase:
    return unsupportedHiddenKind(ByteOffset, "hidden_shared_base");
  case SourceHiddenArgKind::HiddenDefaultQueue:
    return loadTargetHiddenPointer(
        Ctx,
        AMDGPU::getDefaultQueueImplicitArgPosition(Ctx.TargetCodeObjectVersion),
        "amdgpu-no-default-queue", "source_hidden_default_queue");
  case SourceHiddenArgKind::HiddenCompletionAction:
    return loadTargetHiddenPointer(
        Ctx,
        AMDGPU::getCompletionActionImplicitArgPosition(
            Ctx.TargetCodeObjectVersion),
        "amdgpu-no-completion-action", "source_hidden_completion_action");
  case SourceHiddenArgKind::HiddenMultigridSyncArg:
    return loadTargetHiddenPointer(
        Ctx,
        AMDGPU::getMultigridSyncArgImplicitArgPosition(
            Ctx.TargetCodeObjectVersion),
        "amdgpu-no-multigrid-sync-arg", "source_hidden_multigrid_sync_arg");
  case SourceHiddenArgKind::HiddenHostcallBuffer:
    return loadTargetHiddenPointer(
        Ctx,
        AMDGPU::getHostcallImplicitArgPosition(Ctx.TargetCodeObjectVersion),
        "amdgpu-no-hostcall-ptr", "source_hidden_hostcall_buffer");
  case SourceHiddenArgKind::HiddenHeapV1:
    if (Ctx.TargetCodeObjectVersion < AMDGPU::AMDHSA_COV5)
      return unsupportedHiddenKind(ByteOffset, "hidden_heap_v1");
    return loadTargetHiddenPointer(Ctx, AMDGPU::ImplicitArg::HEAP_PTR_OFFSET,
                                   "amdgpu-no-heap-ptr",
                                   "source_hidden_heap_v1");
  case SourceHiddenArgKind::None:
  case SourceHiddenArgKind::UnsupportedHidden:
    return unsupportedHiddenKind(ByteOffset, "<unknown>");
  }
  llvm_unreachable("unhandled SourceHiddenArgKind");
}

// Emit one byte from the source hidden-argument metadata view. Returns a null
// Value when the offset is not a source hidden argument.
Expected<Value *> emitSourceHiddenByte(SourceHiddenArgContext &Ctx,
                                       uint64_t ByteOffset) {
  std::optional<SourceHiddenArgByte> Byte =
      classifySourceHiddenArgByte(Ctx.Args, ByteOffset);
  if (!Byte)
    return nullptr;

  Expected<Value *> Whole = emitHiddenArgValue(Ctx, Byte->Kind, ByteOffset);
  if (!Whole)
    return Whole.takeError();

  Value *Wide = Ctx.B.CreateZExtOrTrunc(*Whole, Ctx.I64Ty, "hidden_wide");
  uint64_t ByteInArg = Byte->byteIndexInArg();
  if (ByteInArg != 0)
    Wide = Ctx.B.CreateLShr(Wide, Ctx.B.getInt64(ByteInArg * 8),
                            "hidden_byte_shift");
  return Ctx.B.CreateTrunc(Wide, Ctx.I8Ty, "source_hidden_byte");
}

} // namespace

Expected<Value *> emitSourceHiddenInteger(SourceHiddenArgContext &Ctx,
                                          uint64_t ByteOffset,
                                          unsigned ByteWidth, bool IsSigned) {
  if (ByteWidth != 1 && ByteWidth != 2 && ByteWidth != 4)
    return RaiseFailure::atInstruction(
        RaiseFailureReason::UnsupportedSourceHiddenArg, "<source-hidden-arg>",
        ByteOffset, "hidden-arg",
        formatv("unsupported source hidden integer byte width {0}", ByteWidth)
            .str());

  Value *Acc = Ctx.B.getInt32(0);
  for (unsigned I = 0; I < ByteWidth; ++I) {
    Expected<Value *> Byte = emitSourceHiddenByte(Ctx, ByteOffset + I);
    if (!Byte)
      return Byte.takeError();
    // The first byte decides whether this offset is a hidden arg at all; a
    // subsequent unmatched byte means the field straddles non-hidden memory.
    if (!*Byte) {
      if (I == 0)
        return nullptr;
      return RaiseFailure::atInstruction(
          RaiseFailureReason::UnsupportedSourceHiddenArg, "<source-hidden-arg>",
          ByteOffset, "hidden-arg",
          formatv("source hidden dword at byte offset {0} spans non-hidden "
                  "byte {1}",
                  ByteOffset, ByteOffset + I)
              .str());
    }

    Value *Part = Ctx.B.CreateZExt(*Byte, Ctx.I32Ty, "source_hidden_byte_zext");
    if (I != 0)
      Part = Ctx.B.CreateShl(Part, Ctx.B.getInt32(I * 8),
                             "source_hidden_byte_place");
    Acc = Ctx.B.CreateOr(Acc, Part, "source_hidden_dword");
  }
  if (IsSigned && ByteWidth < 4) {
    Type *NarrowTy = Type::getIntNTy(Ctx.C, ByteWidth * 8);
    return Ctx.B.CreateSExt(
        Ctx.B.CreateTrunc(Acc, NarrowTy, "source_hidden_narrow"), Ctx.I32Ty,
        "source_hidden_sext");
  }
  return Acc;
}

Expected<Value *> emitSourceHiddenDword(SourceHiddenArgContext &Ctx,
                                        uint64_t ByteOffset) {
  return emitSourceHiddenInteger(Ctx, ByteOffset, /*ByteWidth=*/4,
                                 /*IsSigned=*/false);
}

} // namespace COMGR::hotswap
