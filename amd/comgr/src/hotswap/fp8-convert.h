//===- fp8-convert.h - Hotswap transpiler --------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// fp8/bf8 OCP <-> FNUZ re-encoding shared across the hotswap lowerings.
//
// fp8 has two incompatible numeric interpretations of the same byte:
//   * OCP  : E4M3FN (bias 7, max 448, NaN=S.1111.111, has -0, no Inf) and
//            E5M2 (bias 15, IEEE-style Inf/NaN, max 57344).  Used by gfx950
//            (CDNA4) and gfx12 / gfx1250 (RDNA).
//   * FNUZ : E4M3FNUZ (bias 8, max 240) and E5M2FNUZ (bias 16, max 57344);
//            no Inf, a single NaN encoding 0x80, no -0.  Used by gfx940 /
//            gfx941 / gfx942 (CDNA3).
//
// In-register fp8 bytes stay in the SOURCE representation and are re-encoded
// at every gfx942 (FNUZ) fp8 hardware boundary.  The re-encoders round
// half-to-even, map NaN and E5M2 Inf to the target's canonical NaN (FNUZ 0x80,
// OCP 0x7F) as APFloat does, saturate finite overflow with the sign preserved
// (only OCP E4M3's (240, 448] -> 240), and flush OCP -0 to +0.
//
// That saturation is a property of THESE converters, not of gfx942 fp8
// generally: under the default MODE.FP16_OVFL=0 the f32 -> fp8 hardware encode
// (`v_cvt_pk_fp8_f32` et al.) yields NaN rather than a clamp for out-of-range
// inputs, and `v_cvt_pk_bf8_f32` yields +/-Inf.
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_FP8_CONVERT_H
#define HOTSWAP_TRANSPILER_FP8_CONVERT_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

namespace llvm {
class Value;
class Function;
template <typename FolderTy, typename InserterTy> class IRBuilder;
class ConstantFolder;
class IRBuilderDefaultInserter;
} // namespace llvm

namespace COMGR::hotswap {

struct ISAProfile;

/// Numeric interpretation of an fp8/bf8 byte on a given ISA. `None` means the
/// ISA has no fp8/bf8 hardware; `Unknown` means it has some but its ISA
/// version did not classify -- both are refusals, never a pass-through.
enum class Fp8Format { None, Unknown, OCP, FNUZ };

/// How an ISA's fp8/bf8 hardware (MFMA operands, v_cvt_*_fp8/bf8) interprets
/// fp8 bytes; see `ISAProfile::Fp8Fmt`.
Fp8Format fp8FormatOf(const ISAProfile &P);

/// Data flow across an fp8/bf8 hardware boundary: SrcToTgt for hardware inputs
/// (MFMA/WMMA operands, decode inputs), TgtToSrc for hardware outputs (encode
/// results).
enum class Fp8Dir { SrcToTgt, TgtToSrc };

/// What an fp8/bf8 byte crossing a source/target boundary needs; ToOcp/ToFnuz
/// name the destination format of the re-encode.
enum class Fp8Reencode { None, ToOcp, ToFnuz };

/// Classify an fp8/bf8 data flow in direction \p Dir between \p Src and \p Tgt.
/// Call only for instructions that really do carry fp8/bf8 bytes: an
/// unclassifiable side is an error, since there is then no correct lowering.
llvm::Expected<Fp8Reencode> fp8Reencode(const ISAProfile &Src,
                                        const ISAProfile &Tgt, Fp8Dir Dir);

/// Element format the opcode gives each operand of an fp8/bf8 matmul.
enum class Fp8AbFormat { Fp8Fp8, Fp8Bf8, Bf8Fp8, Bf8Bf8 };

/// Per-operand element format: true means bf8 (E5M2), false fp8 (E4M3). A and
/// B are independent (the mixed _fp8_bf8 / _bf8_fp8 opcodes), so each side
/// picks its own converter. Sole owner of this mapping -- the MFMA and WMMA
/// lowerings both derive from it.
struct Fp8Sides {
  bool AIsBf8;
  bool BIsBf8;
};
constexpr Fp8Sides fp8SidesOf(Fp8AbFormat F) {
  return {F == Fp8AbFormat::Bf8Fp8 || F == Fp8AbFormat::Bf8Bf8,
          F == Fp8AbFormat::Fp8Bf8 || F == Fp8AbFormat::Bf8Bf8};
}

using HotswapIRBuilder =
    llvm::IRBuilder<llvm::ConstantFolder, llvm::IRBuilderDefaultInserter>;

/// Per-byte-lane converters over a `<N x i32>` where each lane holds a byte
/// value 0..255; return a `<N x i32>` of re-encoded bytes.  Verified
/// exhaustively over all 256 byte values.
llvm::Value *convertOcpE4M3ToFnuz(HotswapIRBuilder &B, llvm::Value *Bytes);
llvm::Value *convertOcpE5M2ToFnuz(HotswapIRBuilder &B, llvm::Value *Bytes);
llvm::Value *convertFnuzE4M3ToOcp(HotswapIRBuilder &B, llvm::Value *Bytes);
llvm::Value *convertFnuzE5M2ToOcp(HotswapIRBuilder &B, llvm::Value *Bytes);

/// Re-encode a packed fp8/bf8 dword (4 bytes) through one of the byte-lane
/// converters above.  \p IsBf8 selects E5M2 vs E4M3; \p ToFnuz selects the
/// OCP->FNUZ vs FNUZ->OCP direction.
llvm::Value *convertFp8Dword(HotswapIRBuilder &B, llvm::Value *Dword,
                             bool IsBf8, bool ToFnuz);

/// Re-encode an array of packed fp8/bf8 dwords in place (see convertFp8Dword).
void convertFp8DwordsInPlace(HotswapIRBuilder &B,
                             llvm::SmallVectorImpl<llvm::Value *> &Dwords,
                             bool IsBf8, bool ToFnuz);

/// Decode one fp8/bf8 byte (\p Byte is an i32 holding 0..255) to f32, reading
/// it as \p Fmt; null if \p Fmt is neither OCP nor FNUZ.  Exact for all 256
/// inputs, including subnormals, Inf and NaN.
///
/// Used instead of the target's fp8 decode hardware when the source and target
/// formats differ: byte -> f32 is WIDENING, so every source byte has an exact
/// f32 image, where a byte re-encode plus the target's decoder would clip
/// (OCP E5M2 Inf, OCP E4M3's (240, 448], and -0 all survive here).
///
/// Test contract: the result is named `{fp8,bf8}_dec_{ocp,fnuz}`, which lit
/// fixtures match on to tell the conversion direction apart.
llvm::Value *decodeFp8ByteToF32(HotswapIRBuilder &B, llvm::Value *Byte,
                                bool IsBf8, Fp8Format Fmt);

/// Encode two f32 into two OCP fp8/bf8 bytes, packed into the low 16 bits.
///
/// \p CvtFn is the target's `cvt_pk_{fp8,bf8}_f32` (an FNUZ encoder).  OCP and
/// FNUZ share a mantissa width and differ by exactly one in exponent bias, so
/// the FNUZ encoding of x/2 IS the OCP encoding of x: the hardware does the
/// round-half-to-even and this only keeps it inside range.  Out-of-range
/// magnitudes, NaN and signed zero are handled explicitly, so MODE.FP16_OVFL
/// never comes into play.
///
/// Only valid when the SOURCE format is OCP; the mirrored trick does not work
/// for an FNUZ source, whose top exponent has no OCP counterpart.
///
/// Test contract: the result is named `pk_fp8_ocp`.
llvm::Value *encodeF32PairToOcpFp8(HotswapIRBuilder &B, llvm::Function *CvtFn,
                                   llvm::Value *S0, llvm::Value *S1,
                                   bool IsBf8);

} // namespace COMGR::hotswap

#endif // HOTSWAP_TRANSPILER_FP8_CONVERT_H
