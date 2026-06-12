//===- HotswapMCTest.cpp - Unit tests for HotSwap LLVM MC layer -----------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Tests for the hotswap MC/LLVM infrastructure in comgr-hotswap-llvm.cpp:
/// initLLVM construction, LLVMState::encodeSBranch, assembleSingleInst /
/// decodeTextSection round-trip, applyMnemonicSwap, applyByteReplace, and
/// checkVgprOverlap.
///
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-internal.h"
#include "comgr.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/TargetSelect.h"
#include "gtest/gtest.h"

#include <cstring>
#include <mutex>

using namespace COMGR;
using namespace COMGR::hotswap;

// --------------------------------------------------------------------------
// Test-only stub definition of COMGR::ensureLLVMInitialized.
//
// hotswap::initLLVM() calls COMGR::ensureLLVMInitialized() (normally defined
// in comgr.cpp) to register the AMDGPU target. The production definition
// lives in libamd_comgr, which we don't want to link into the unit-test
// binary (it drags in the full Comgr compiler pipeline). Providing this
// stub here keeps the test binary minimal while matching the production
// registration behaviour for the target components we exercise.
//
// Stubbing is safe because this translation unit is linked into
// HotswapMCTests only, never into libamd_comgr.
// --------------------------------------------------------------------------
namespace COMGR {
void ensureLLVMInitialized() {
  static std::once_flag Once;
  std::call_once(Once, []() {
    LLVMInitializeAMDGPUTargetInfo();
    LLVMInitializeAMDGPUTargetMC();
    LLVMInitializeAMDGPUDisassembler();
    LLVMInitializeAMDGPUAsmParser();
    LLVMInitializeAMDGPUAsmPrinter();
    LLVMInitializeAMDGPUTarget();
  });
}
} // namespace COMGR

// Build a TargetIdentifier for the gfx1250 test subtarget without features --
// production callers go through parseTargetIdentifier; here we populate
// directly so the tests stay self-contained.
static TargetIdentifier makeGfx1250Ident() {
  TargetIdentifier TI;
  TI.Arch = "amdgcn";
  TI.Vendor = "amd";
  TI.OS = "amdhsa";
  TI.Environ = "";
  TI.Processor = "gfx1250";
  return TI;
}

// Helper: decode the little-endian 32-bit dword at \p Bytes.
static uint32_t readDword(const uint8_t *Bytes) {
  uint32_t V;
  std::memcpy(&V, Bytes, sizeof(V));
  return V;
}

// -- initLLVM ----------------------------------------------------------------

TEST(InitLLVM, ValidGfx1250) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  EXPECT_EQ(S.Cpu, "gfx1250");
  EXPECT_NE(S.Target, nullptr);
  ASSERT_NE(S.MCII, nullptr);
  EXPECT_LT(S.SBranchOpcode, S.MCII->getNumOpcodes());
  EXPECT_EQ(S.SNopBytes.size(), MinInstSize);
}

TEST(InitLLVM, EmptyProcessorFails) {
  TargetIdentifier TI = makeGfx1250Ident();
  TI.Processor = "";
  LLVMState S = initLLVM(TI);
  EXPECT_FALSE(S.Valid);
}

TEST(InitLLVM, UnknownProcessorFails) {
  TargetIdentifier TI = makeGfx1250Ident();
  TI.Processor = "gfxbogus";
  LLVMState S = initLLVM(TI);
  EXPECT_FALSE(S.Valid);
}

// -- LLVMState::encodeSBranch -------------------------------------------------
//
// Exact byte checks are avoided here -- tblgen encodings can be reshuffled
// across LLVM versions. Instead we assert the structural invariants that
// downstream callers rely on: the encoded delta round-trips to the expected
// simm16 field, the size is MinInstSize, and out-of-range / unaligned deltas
// are rejected.

TEST(EncodeSBranch, ForwardBranchRoundTrip) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  // s_branch SIMM16 -> PC += (SIMM16 + 1) * 4; From=0, To=8 => SIMM16=1.
  llvm::SmallVector<uint8_t> Out = S.encodeSBranch(0, 8);
  ASSERT_EQ(Out.size(), MinInstSize);
  uint32_t Encoded = readDword(Out.data());
  EXPECT_EQ(static_cast<uint16_t>(Encoded & 0xFFFFu), 1u);
}

TEST(EncodeSBranch, BackwardBranchRoundTrip) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  // From=16, To=0 => delta=-5 dwords.
  llvm::SmallVector<uint8_t> Out = S.encodeSBranch(16, 0);
  ASSERT_EQ(Out.size(), MinInstSize);
  uint32_t Encoded = readDword(Out.data());
  EXPECT_EQ(static_cast<int16_t>(Encoded & 0xFFFFu), -5);
}

TEST(EncodeSBranch, ZeroOffsetBranch) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  // PC advance of MinInstSize: SIMM16 should be 0.
  llvm::SmallVector<uint8_t> Out = S.encodeSBranch(0, MinInstSize);
  ASSERT_EQ(Out.size(), MinInstSize);
  EXPECT_EQ(readDword(Out.data()) & 0xFFFFu, 0u);
}

TEST(EncodeSBranch, UnalignedDeltaFails) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  EXPECT_TRUE(S.encodeSBranch(0, 7).empty());
}

TEST(EncodeSBranch, OutOfRangeFails) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  EXPECT_TRUE(S.encodeSBranch(0, 500000).empty());
}

TEST(EncodeSBranch, FailsOnInvalidState) {
  LLVMState S; // default-constructed, Valid = false
  EXPECT_TRUE(S.encodeSBranch(0, 8).empty());
}

// -- assembleSingleInst / decodeTextSection round-trip ------------------------

TEST(AssembleDecode, SNopRoundTrip) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  llvm::SmallVector<uint8_t> Bytes = assembleSingleInst("s_nop 0", S);
  ASSERT_EQ(Bytes.size(), MinInstSize);
  // Must match the pre-encoded bytes cached in LLVMState at init time.
  EXPECT_EQ(llvm::ArrayRef<uint8_t>(Bytes),
            llvm::ArrayRef<uint8_t>(S.SNopBytes));

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 1u);
  EXPECT_EQ(Decoded[0].Size, MinInstSize);
  EXPECT_EQ(Decoded[0].Mnemonic, "s_nop");
}

TEST(AssembleDecode, RejectsGarbageAsm) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Bytes = assembleSingleInst("not_a_real_op", S);
  EXPECT_TRUE(Bytes.empty());
}

// -- applyByteReplace ---------------------------------------------------------

TEST(ApplyByteReplace, PadsWithSNop) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  // 8 bytes of zeroed "text", simulate replacing the first 8 bytes with a
  // 4-byte rule and expecting the remainder to be padded with s_nop.
  uint8_t Text[8] = {};
  RewriteRule Rule;
  Rule.ReplaceBytes.assign(S.SNopBytes.begin(), S.SNopBytes.end());
  ASSERT_TRUE(applyByteReplace(Rule, /*InstOffset=*/0, /*InstSize=*/8, Text,
                               sizeof(Text), S));
  // Both halves should be s_nop bytes now.
  EXPECT_EQ(std::memcmp(Text, S.SNopBytes.data(), MinInstSize), 0);
  EXPECT_EQ(std::memcmp(Text + MinInstSize, S.SNopBytes.data(), MinInstSize),
            0);
}

TEST(ApplyByteReplace, RejectsOutOfBounds) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  uint8_t Text[4] = {};
  RewriteRule Rule;
  Rule.ReplaceBytes.assign(S.SNopBytes.begin(), S.SNopBytes.end());
  // InstOffset+InstSize (8) exceeds TextSize (4).
  EXPECT_FALSE(applyByteReplace(Rule, /*InstOffset=*/0, /*InstSize=*/8, Text,
                                sizeof(Text), S));
}

// -- checkVgprOverlap ---------------------------------------------------------
//
// checkVgprOverlap checks whether any register operand of a "WMMA-like"
// MCInst overlaps the destination (operand 0) of a "VALU-like" MCInst.
// We drive it with real MCInsts produced by assembling + decoding simple
// AMDGPU instructions so the register operands are populated the way the
// production code sees them.

// Assemble \p Asm and decode the first resulting MCInst. Aborts the test if
// either step fails, so callers can rely on the return value being populated.
static llvm::MCInst assembleOne(llvm::StringRef Asm, const LLVMState &S) {
  llvm::SmallVector<uint8_t> Bytes = assembleSingleInst(Asm, S);
  EXPECT_FALSE(Bytes.empty()) << "failed to assemble: " << Asm.str();
  std::vector<InternalDecodedInst> Decoded;
  EXPECT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded))
      << "failed to decode: " << Asm.str();
  EXPECT_EQ(Decoded.size(), 1u) << "expected one inst for: " << Asm.str();
  return Decoded.empty() ? llvm::MCInst() : Decoded[0].Inst;
}

TEST(CheckVgprOverlap, DetectsDirectOverlap) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  // Wmma-like inst references v5 and v10; Valu-like inst writes v10.
  llvm::MCInst Wmma = assembleOne("v_mov_b32 v5, v10", S);
  llvm::MCInst Valu = assembleOne("v_mov_b32 v10, v20", S);
  EXPECT_TRUE(checkVgprOverlap(Wmma, Valu, *S.MRI));
}

TEST(CheckVgprOverlap, NoOverlapForDisjointVgprs) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  // Wmma-like inst references v0, v1; Valu-like inst writes v10.
  llvm::MCInst Wmma = assembleOne("v_mov_b32 v0, v1", S);
  llvm::MCInst Valu = assembleOne("v_mov_b32 v10, v20", S);
  EXPECT_FALSE(checkVgprOverlap(Wmma, Valu, *S.MRI));
}

TEST(CheckVgprOverlap, HandlesEmptyValuInst) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::MCInst Wmma = assembleOne("v_mov_b32 v0, v1", S);
  llvm::MCInst Empty; // no operands
  EXPECT_FALSE(checkVgprOverlap(Wmma, Empty, *S.MRI));
}

// -- buildTrampoline ----------------------------------------------------------
//
// buildTrampoline assembles one or more asm lines and appends a branch-back
// s_branch to the instruction immediately following the original site. We
// verify the size / structure of the result rather than the exact bytes
// (which are target-specific and captured separately in the encodeSBranch /
// SNopBytes tests).

TEST(BuildTrampoline, AppendsBranchBackAfterAssembledAsm) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  std::string AsmLine = "s_nop 0";
  std::vector<std::string> AsmLines = {AsmLine};
  constexpr uint64_t OriginalOffset = 0;
  constexpr uint32_t OriginalSize = MinInstSize;
  constexpr uint64_t TrampolineTextOffset = 0x1000;

  Trampoline T = buildTrampoline(AsmLines, OriginalOffset, OriginalSize,
                                 TrampolineTextOffset, S);

  EXPECT_EQ(T.OriginalOffset, OriginalOffset);
  EXPECT_EQ(T.OriginalSize, OriginalSize);
  // One assembled inst (s_nop 0, 4 bytes) + one branch-back (4 bytes).
  ASSERT_EQ(T.Bytes.size(), 2u * MinInstSize);
  // The first MinInstSize bytes should match the cached s_nop encoding.
  EXPECT_EQ(std::memcmp(T.Bytes.data(), S.SNopBytes.data(), MinInstSize), 0);
}

TEST(BuildTrampoline, EmptyOnBadAsm) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  std::vector<std::string> AsmLines = {"this_is_not_a_valid_instruction"};
  Trampoline T = buildTrampoline(AsmLines, /*OriginalOffset=*/0,
                                 /*OriginalSize=*/MinInstSize,
                                 /*TrampolineTextOffset=*/0x1000, S);
  EXPECT_TRUE(T.Bytes.empty());
}

// -- classifyWmmaNops ---------------------------------------------------------

TEST(ClassifyWmmaNops, NonWmmaReturnsDefault) {
  WmmaNopReq Req = classifyWmmaNops("v_add_f32");
  EXPECT_EQ(Req.A0Nops, 4);
  EXPECT_EQ(Req.B0Nops, 4);
}

TEST(ClassifyWmmaNops, IntegerWmmaReturns8) {
  WmmaNopReq Req = classifyWmmaNops("v_wmma_i32_16x16x32_iu8");
  EXPECT_EQ(Req.A0Nops, 8);
  EXPECT_EQ(Req.B0Nops, 4);
}

TEST(ClassifyWmmaNops, Iu4Returns8) {
  WmmaNopReq Req = classifyWmmaNops("v_wmma_i32_16x16x64_iu4");
  EXPECT_EQ(Req.A0Nops, 8);
  EXPECT_EQ(Req.B0Nops, 4);
}

TEST(ClassifyWmmaNops, F8f6f4Returns1) {
  WmmaNopReq Req = classifyWmmaNops("v_wmma_f32_16x16x128_f8f6f4");
  EXPECT_EQ(Req.A0Nops, 1);
  EXPECT_EQ(Req.B0Nops, 4);
}

TEST(ClassifyWmmaNops, Fp8_16x16x128Returns3) {
  WmmaNopReq Req = classifyWmmaNops("v_wmma_f32_16x16x128_fp8_fp8");
  EXPECT_EQ(Req.A0Nops, 3);
  EXPECT_EQ(Req.B0Nops, 4);
}

TEST(ClassifyWmmaNops, Fp8SmallReturns1) {
  WmmaNopReq Req = classifyWmmaNops("v_wmma_f32_16x16x32_fp8_fp8");
  EXPECT_EQ(Req.A0Nops, 1);
  EXPECT_EQ(Req.B0Nops, 4);
}

TEST(ClassifyWmmaNops, F16Returns4) {
  WmmaNopReq Req = classifyWmmaNops("v_wmma_f32_16x16x16_f16");
  EXPECT_EQ(Req.A0Nops, 4);
  EXPECT_EQ(Req.B0Nops, 4);
}

TEST(ClassifyWmmaNops, Bf16Returns4) {
  WmmaNopReq Req = classifyWmmaNops("v_wmma_f32_16x16x16_bf16");
  EXPECT_EQ(Req.A0Nops, 4);
  EXPECT_EQ(Req.B0Nops, 4);
}

TEST(ClassifyWmmaNops, SwmmacIu8Returns8) {
  WmmaNopReq Req = classifyWmmaNops("v_swmmac_i32_16x16x64_iu8");
  EXPECT_EQ(Req.A0Nops, 8);
  EXPECT_EQ(Req.B0Nops, 4);
}

TEST(ClassifyWmmaNops, F32WmmaFallsToDefault) {
  WmmaNopReq Req = classifyWmmaNops("v_wmma_f32_16x16x4_f32");
  EXPECT_EQ(Req.A0Nops, 4);
  EXPECT_EQ(Req.B0Nops, 4);
}

TEST(ClassifyWmmaNops, OrderingMostRestrictiveWins) {
  // A mnemonic containing both _iu8 and _f16 should return 8 (iu8 first)
  WmmaNopReq Req = classifyWmmaNops("v_wmma_f16_something_iu8");
  EXPECT_EQ(Req.A0Nops, 8);
}

// -- patchScaleSrc2 -----------------------------------------------------------
//
// Pure byte-level tests for the VOP3PX2 scale_src2 bit-field fix.
// The function patches bits [58:50] of a 16-byte VOP3PX2 encoding to
// VGPR0 (0x100): byte 6 bits [7:2] cleared, byte 7 bit [2] set,
// byte 7 bits [1:0] cleared.

TEST(PatchScaleSrc2, ZeroedFieldGetsPatched) {
  uint8_t Inst[16] = {};
  EXPECT_TRUE(patchScaleSrc2(Inst));
  EXPECT_EQ(Inst[6] & 0xFC, 0x00);
  EXPECT_EQ(Inst[7] & 0x07, 0x04);
}

TEST(PatchScaleSrc2, PreservesOtherBytes) {
  uint8_t Inst[16];
  std::memset(Inst, 0xAA, sizeof(Inst));
  EXPECT_TRUE(patchScaleSrc2(Inst));
  for (size_t I = 0; I < 16; ++I) {
    if (I == 6 || I == 7)
      continue;
    EXPECT_EQ(Inst[I], 0xAA) << "byte " << I << " unexpectedly modified";
  }
}

TEST(PatchScaleSrc2, AllOnesFieldGetsPatched) {
  uint8_t Inst[16] = {};
  Inst[6] = 0xFF;
  Inst[7] = 0xFF;
  EXPECT_TRUE(patchScaleSrc2(Inst));
  EXPECT_EQ(Inst[6] & 0xFC, 0x00);
  EXPECT_EQ(Inst[7] & 0x07, 0x04);
  EXPECT_EQ(Inst[7] & 0xF8, 0xF8);
}

TEST(PatchScaleSrc2, AlreadyVgpr0ReturnsFalse) {
  uint8_t Inst[16] = {};
  Inst[7] = 0x04;
  EXPECT_FALSE(patchScaleSrc2(Inst));
  EXPECT_EQ(Inst[6], 0x00);
  EXPECT_EQ(Inst[7], 0x04);
}

TEST(PatchScaleSrc2, IsIdempotent) {
  uint8_t Inst[16] = {};
  Inst[6] = 0xAB;
  Inst[7] = 0xCD;
  EXPECT_TRUE(patchScaleSrc2(Inst));
  uint8_t AfterFirst6 = Inst[6];
  uint8_t AfterFirst7 = Inst[7];
  EXPECT_FALSE(patchScaleSrc2(Inst));
  EXPECT_EQ(Inst[6], AfterFirst6);
  EXPECT_EQ(Inst[7], AfterFirst7);
}

TEST(PatchScaleSrc2, PreservesNonScaleSrc2Bits) {
  uint8_t Inst[16] = {};
  Inst[6] = 0x03 | 0xA0;
  Inst[7] = 0xF8 | 0x02;
  EXPECT_TRUE(patchScaleSrc2(Inst));
  EXPECT_EQ(Inst[6] & 0x03, 0x03);
  EXPECT_EQ(Inst[7] & 0xF8, 0xF8);
  EXPECT_EQ(Inst[6] & 0xFC, 0x00);
  EXPECT_EQ(Inst[7] & 0x07, 0x04);
}

// -- HotswapPatchVTable -------------------------------------------------------
//
// Tests for the .def-driven patch registry that replaced the
// LLVM_ATTRIBUTE_WEAK override pattern (issue ROCm/llvm-project#2479).
//
// Coverage strategy: link errors already catch missing register*Patch
// definitions and missing comgr-hotswap-patches.def entries, so we only
// test what the linker cannot:
//   1. One canonical per-installer "binds only its own slot" check,
//      kept as a worked example for future patch authors. Wrong-slot
//      bugs in the other register*Patch functions are caught via the
//      install end-to-end test below.
//   2. End-to-end install: a default-constructed vtable has null slots,
//      installHotswapPatches() binds every .def entry, and slots without
//      a .def entry stay null (the dispatcher's no-op contract).
//   3. The production singleton accessor returns the same fully-bound
//      vtable on every call -- the initializer eagerly runs the install
//      under the C++11 magic-static rule, so production code never sees
//      an empty vtable.

TEST(HotswapPatchVTable, RegisterInPlaceBindsOnlyInPlaceSlot) {
  HotswapPatchVTable VT;
  registerInPlacePatch(VT);
  EXPECT_NE(VT.applyInPlacePatches, nullptr);
  EXPECT_EQ(VT.applyTrampolinePatches, nullptr);
  EXPECT_EQ(VT.applyWmmaHazardPatch, nullptr);
  EXPECT_EQ(VT.applyVop3px2Src2Fix, nullptr);
}

TEST(HotswapPatchVTable, InstallBindsRegisteredAndLeavesUnregisteredNull) {
  HotswapPatchVTable VT;

  // Defaults: every slot null (no patch implementation linked yet).
  EXPECT_EQ(VT.applyInPlacePatches, nullptr);
  EXPECT_EQ(VT.applyTrampolinePatches, nullptr);
  EXPECT_EQ(VT.applyWmmaHazardPatch, nullptr);
  EXPECT_EQ(VT.applyVop3px2Src2Fix, nullptr);
  EXPECT_EQ(VT.applyWmmaSplitPatches, nullptr);
  EXPECT_EQ(VT.applyScratchPatches, nullptr);

  installHotswapPatches(VT);

  // Slots backed by a comgr-hotswap-patches.def entry get bound. If a
  // register*Patch fails to set its slot (or sets the wrong one), one
  // of these EXPECT_NEs catches it.
  EXPECT_NE(VT.applyInPlacePatches, nullptr);
  EXPECT_NE(VT.applyTrampolinePatches, nullptr);
  EXPECT_NE(VT.applyWmmaHazardPatch, nullptr);
  EXPECT_NE(VT.applyVop3px2Src2Fix, nullptr);
  EXPECT_NE(VT.applyWmmaSplitPatches, nullptr);
  EXPECT_NE(VT.applyScratchPatches, nullptr);
}

TEST(HotswapPatchVTable, ProcessSingletonIdentityAndEagerInstall) {
  HotswapPatchVTable &VT1 = getHotswapPatchVTable();
  HotswapPatchVTable &VT2 = getHotswapPatchVTable();
  EXPECT_EQ(&VT1, &VT2);

  // The singleton's initializer runs installHotswapPatches() on first
  // access, so every .def-backed slot is already bound by the time the
  // first reference is handed out. Pinning this contract here keeps the
  // dispatcher safe to call getHotswapPatchVTable() without any explicit
  // install step at the entry point.
  EXPECT_NE(VT1.applyInPlacePatches, nullptr);
  EXPECT_NE(VT1.applyTrampolinePatches, nullptr);
  EXPECT_NE(VT1.applyWmmaHazardPatch, nullptr);
  EXPECT_NE(VT1.applyVop3px2Src2Fix, nullptr);
  EXPECT_NE(VT1.applyWmmaSplitPatches, nullptr);
  EXPECT_NE(VT1.applyScratchPatches, nullptr);
}

// -- DS ADDTID trampoline support ---------------------------------------------
//
// Tests for the ds_load_addtid_b32 / ds_store_addtid_b32 trampoline patch
// (DEGFXMI400-12025). Coverage is bottom-up: first that the encode/decode
// of ADDTID instructions exposes the expected MCInst operand layout, then
// that the trampoline replacement asm round-trips through the MC layer,
// then that buildTrampoline integrates a full ADDTID body.

namespace {

// AddtidOpReg / AddtidOpOffset / AddtidOpGds operand-layout constants live
// in comgr-hotswap-internal.h and are imported by the COMGR::hotswap using-
// declaration at the top of this file.

// Decode a single instruction string and return the resulting MCInst, or
// llvm::None on failure. Aborts the test if assemble/decode fail so the
// caller can dereference unconditionally.
llvm::MCInst decodeOne(llvm::StringRef Asm, const LLVMState &S) {
  llvm::SmallVector<uint8_t> Bytes = assembleSingleInst(Asm, S);
  EXPECT_FALSE(Bytes.empty()) << "failed to assemble: " << Asm.str();
  std::vector<InternalDecodedInst> Decoded;
  EXPECT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded))
      << "failed to decode: " << Asm.str();
  EXPECT_EQ(Decoded.size(), 1u) << "expected one inst for: " << Asm.str();
  return Decoded.empty() ? llvm::MCInst() : Decoded[0].Inst;
}

} // namespace

TEST(AddTid, LoadAddTidDecodesWithExpectedLayout) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  llvm::MCInst Inst = decodeOne("ds_load_addtid_b32 v5 offset:128", S);
  ASSERT_GE(Inst.getNumOperands(), 3u);

  // Direct operand access: register, then offset, then gds bit. No
  // print-and-parse round-trip -- production code uses the same operand
  // indices to reach the destination VGPR.
  EXPECT_TRUE(Inst.getOperand(AddtidOpReg).isReg());
  EXPECT_NE(Inst.getOperand(AddtidOpReg).getReg(), 0u);
  EXPECT_TRUE(Inst.getOperand(AddtidOpOffset).isImm());
  EXPECT_EQ(Inst.getOperand(AddtidOpOffset).getImm(), 128);
  EXPECT_TRUE(Inst.getOperand(AddtidOpGds).isImm());
  EXPECT_EQ(Inst.getOperand(AddtidOpGds).getImm(), 0);

  // Production code uses MRI.getName() to resolve the VGPR identifier
  // ("VGPR5" for v5); pin that so a tablegen rename in upstream catches
  // here rather than silently breaking the trampoline.
  const char *N = S.MRI->getName(Inst.getOperand(AddtidOpReg).getReg());
  ASSERT_NE(N, nullptr);
  EXPECT_STREQ(N, "VGPR5");
}

TEST(AddTid, StoreAddTidDecodesWithExpectedLayout) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  llvm::MCInst Inst = decodeOne("ds_store_addtid_b32 v10 offset:256", S);
  ASSERT_GE(Inst.getNumOperands(), 3u);
  EXPECT_TRUE(Inst.getOperand(AddtidOpReg).isReg());
  EXPECT_NE(Inst.getOperand(AddtidOpReg).getReg(), 0u);
  EXPECT_TRUE(Inst.getOperand(AddtidOpOffset).isImm());
  EXPECT_EQ(Inst.getOperand(AddtidOpOffset).getImm(), 256);
  EXPECT_TRUE(Inst.getOperand(AddtidOpGds).isImm());
  EXPECT_EQ(Inst.getOperand(AddtidOpGds).getImm(), 0);

  const char *N = S.MRI->getName(Inst.getOperand(AddtidOpReg).getReg());
  ASSERT_NE(N, nullptr);
  EXPECT_STREQ(N, "VGPR10");
}

TEST(AddTid, LoadTrampolineAsmAssemblesAndDecodes) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  // Replacement asm for ds_load_addtid_b32 v7 offset:64.
  // The v_and_b32 with 0xfffff masks M0 to the 20 bits that B0's DS unit
  // would have read, keeping the rewrite bit-exact with B0 hardware
  // regardless of stale bits in M0[31:20] on entry.
  std::string Asm = "v_mbcnt_lo_u32_b32 v7, -1, 0\n"
                    "v_mbcnt_hi_u32_b32 v7, -1, v7\n"
                    "v_lshlrev_b32 v7, 2, v7\n"
                    "v_add_nc_u32 v7, m0, v7\n"
                    "v_and_b32 v7, 0xfffff, v7\n"
                    "ds_load_b32 v7, v7 offset:64\n";

  llvm::SmallVector<uint8_t> Bytes = assembleSingleInst(Asm, S);
  ASSERT_FALSE(Bytes.empty());

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 6u);
  EXPECT_EQ(Decoded[0].Mnemonic, "v_mbcnt_lo_u32_b32");
  EXPECT_EQ(Decoded[1].Mnemonic, "v_mbcnt_hi_u32_b32");
  EXPECT_EQ(Decoded[2].Mnemonic, "v_lshlrev_b32");
  EXPECT_EQ(Decoded[3].Mnemonic, "v_add_nc_u32");
  EXPECT_EQ(Decoded[4].Mnemonic, "v_and_b32");
  EXPECT_EQ(Decoded[5].Mnemonic, "ds_load_b32");
}

TEST(AddTid, StoreTrampolineAsmAssemblesAndDecodes) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  // Replacement asm for ds_store_addtid_b32 v10 offset:0 with v42 as the
  // address-compute scratch (the data VGPR v10 is not clobbered). The
  // v_and_b32 with 0xfffff masks M0 to the 20-bit DS-unit width; see
  // LoadTrampolineAsmAssemblesAndDecodes for the rationale.
  std::string Asm = "v_mbcnt_lo_u32_b32 v42, -1, 0\n"
                    "v_mbcnt_hi_u32_b32 v42, -1, v42\n"
                    "v_lshlrev_b32 v42, 2, v42\n"
                    "v_add_nc_u32 v42, m0, v42\n"
                    "v_and_b32 v42, 0xfffff, v42\n"
                    "ds_store_b32 v42, v10\n";

  llvm::SmallVector<uint8_t> Bytes = assembleSingleInst(Asm, S);
  ASSERT_FALSE(Bytes.empty());

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 6u);
  EXPECT_EQ(Decoded[0].Mnemonic, "v_mbcnt_lo_u32_b32");
  EXPECT_EQ(Decoded[1].Mnemonic, "v_mbcnt_hi_u32_b32");
  EXPECT_EQ(Decoded[2].Mnemonic, "v_lshlrev_b32");
  EXPECT_EQ(Decoded[3].Mnemonic, "v_add_nc_u32");
  EXPECT_EQ(Decoded[4].Mnemonic, "v_and_b32");
  EXPECT_EQ(Decoded[5].Mnemonic, "ds_store_b32");
}

TEST(AddTid, LoadTrampolineThroughBuildTrampoline) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  std::vector<std::string> AsmLines = {
      "v_mbcnt_lo_u32_b32 v3, -1, 0", "v_mbcnt_hi_u32_b32 v3, -1, v3",
      "v_lshlrev_b32 v3, 2, v3",      "v_add_nc_u32 v3, m0, v3",
      "v_and_b32 v3, 0xfffff, v3",    "ds_load_b32 v3, v3 offset:0",
  };

  Trampoline T = buildTrampoline(AsmLines, /*OriginalOffset=*/0x100,
                                 /*OriginalSize=*/4,
                                 /*TrampolineTextOffset=*/0x2000, S);

  ASSERT_FALSE(T.Bytes.empty());
  EXPECT_EQ(T.OriginalOffset, 0x100u);
  EXPECT_EQ(T.OriginalSize, 4u);

  // 6 body instructions + 1 branch-back tail.
  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(T.Bytes.data(), T.Bytes.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 7u);
  EXPECT_EQ(Decoded[6].Mnemonic, "s_branch");
}

TEST(AddTid, StoreTrampolineThroughBuildTrampoline) {
  // Mirror of LoadTrampolineThroughBuildTrampoline for the store path, where
  // the data VGPR (v10) must be preserved and an allocator-supplied scratch
  // VGPR (v42) holds the computed address. The two register operands of
  // ds_store_b32 carry independent VGPR indices, which is what distinguishes
  // this from the load case (which can fold dst back into address).
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  std::vector<std::string> AsmLines = {
      "v_mbcnt_lo_u32_b32 v42, -1, 0", "v_mbcnt_hi_u32_b32 v42, -1, v42",
      "v_lshlrev_b32 v42, 2, v42",     "v_add_nc_u32 v42, m0, v42",
      "v_and_b32 v42, 0xfffff, v42",   "ds_store_b32 v42, v10",
  };

  Trampoline T = buildTrampoline(AsmLines, /*OriginalOffset=*/0x180,
                                 /*OriginalSize=*/4,
                                 /*TrampolineTextOffset=*/0x2040, S);

  ASSERT_FALSE(T.Bytes.empty());
  EXPECT_EQ(T.OriginalOffset, 0x180u);
  EXPECT_EQ(T.OriginalSize, 4u);

  // 6 body instructions + 1 branch-back tail, matching the load variant.
  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(T.Bytes.data(), T.Bytes.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 7u);
  EXPECT_EQ(Decoded[0].Mnemonic, "v_mbcnt_lo_u32_b32");
  EXPECT_EQ(Decoded[5].Mnemonic, "ds_store_b32");
  EXPECT_EQ(Decoded[6].Mnemonic, "s_branch");
}
