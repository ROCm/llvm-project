//===- DecoderTest.cpp - Hotswap transpiler decoder unit tests ------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit tests for the decoder library: the AMDGPU MC stack (mc-state), the
// architecture-neutral instruction identity (canonical-op), the per-subtarget
// capability queries (isa-profile), and the decoded-instruction model
// (decoded-inst). Each exercises the piece directly, without a code object or
// the raiser, so the coverage matches what the decoder alone provides.
//
//===----------------------------------------------------------------------===//

#include "hotswap/decoder/canonical-op.h"
#include "hotswap/decoder/decoded-inst.h"
#include "hotswap/decoder/isa-profile.h"
#include "hotswap/decoder/mc-state.h"

#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <mutex>

using namespace COMGR::hotswap;

// initMCState registers the AMDGPU target through COMGR::ensureLLVMInitialized,
// whose production definition lives in libamd_comgr. Provide the registration
// here so the test binary stays minimal instead of linking the full Comgr.
namespace COMGR {
void ensureLLVMInitialized() {
  static std::once_flag Once;
  std::call_once(Once, [] {
    LLVMInitializeAMDGPUTargetInfo();
    LLVMInitializeAMDGPUTargetMC();
    LLVMInitializeAMDGPUDisassembler();
    LLVMInitializeAMDGPUAsmParser();
    LLVMInitializeAMDGPUAsmPrinter();
    LLVMInitializeAMDGPUTarget();
  });
}
} // namespace COMGR

namespace {

// Real gfx942 encodings (from `llvm-mc -mcpu=gfx942 -show-encoding`), decoded
// through the MCState disassembler so the mnemonic helpers run on genuine
// MCInsts rather than hand-built ones.
constexpr uint8_t SMovB32Bytes[] = {0x80, 0x00, 0x80, 0xbe}; // s_mov_b32 s0, 0
constexpr uint8_t SEndpgmBytes[] = {0x00, 0x00, 0x81, 0xbf}; // s_endpgm
constexpr uint8_t VMovB32Bytes[] = {0x80, 0x02, 0x00,
                                    0x7e}; // v_mov_b32_e32 v0, 0

// Holds one gfx942 MCState for the tests that need the disassembler or an
// MCContext. initMCState registers the AMDGPU target itself, so no separate
// target-init step is required.
class DecoderTest : public ::testing::Test {
protected:
  void SetUp() override {
    llvm::Expected<MCState> StateOrErr = initMCState("gfx942");
    ASSERT_TRUE(static_cast<bool>(StateOrErr))
        << llvm::toString(StateOrErr.takeError());
    State = std::move(*StateOrErr);
  }

  // Decode a single instruction from `Bytes` into `Inst`.
  void decode(llvm::ArrayRef<uint8_t> Bytes, llvm::MCInst &Inst) {
    uint64_t Size = 0;
    llvm::MCDisassembler::DecodeStatus Status = State.Disasm->getInstruction(
        Inst, Size, Bytes, /*Address=*/0, llvm::nulls());
    ASSERT_EQ(Status, llvm::MCDisassembler::Success);
    EXPECT_EQ(Size, Bytes.size());
  }

  MCState State;
};

TEST_F(DecoderTest, InitMCStatePopulatesEveryMember) {
  EXPECT_NE(State.Target, nullptr);
  EXPECT_NE(State.InstrInfo, nullptr);
  EXPECT_NE(State.RegInfo, nullptr);
  EXPECT_NE(State.SubtargetInfo, nullptr);
  EXPECT_NE(State.AsmInfo, nullptr);
  EXPECT_NE(State.Ctx, nullptr);
  EXPECT_NE(State.Disasm, nullptr);
  EXPECT_NE(State.Printer, nullptr);
}

TEST_F(DecoderTest, MnemonicHelpersOnScalarMove) {
  llvm::MCInst Inst;
  decode(SMovB32Bytes, Inst);
  EXPECT_EQ(getMnemonic(State, Inst), "s_mov_b32");
  EXPECT_EQ(strippedMnemonic(State, Inst), "s_mov_b32");
  EXPECT_EQ(printInst(State, Inst), "s_mov_b32 s0, 0");
}

TEST_F(DecoderTest, MnemonicHelpersOnProgramEnd) {
  llvm::MCInst Inst;
  decode(SEndpgmBytes, Inst);
  EXPECT_EQ(getMnemonic(State, Inst), "s_endpgm");
  EXPECT_EQ(strippedMnemonic(State, Inst), "s_endpgm");
}

TEST_F(DecoderTest, StrippedMnemonicDropsEncodingSuffix) {
  llvm::MCInst Inst;
  decode(VMovB32Bytes, Inst);
  // The printer keeps the `_e32` encoding suffix; strippedMnemonic drops it so
  // the raiser dispatches on the bare mnemonic.
  EXPECT_EQ(getMnemonic(State, Inst), "v_mov_b32_e32");
  EXPECT_EQ(strippedMnemonic(State, Inst), "v_mov_b32");
}

TEST_F(DecoderTest, EvalOperandAsConstFoldsExpr) {
  llvm::MCInst Inst;
  Inst.addOperand(llvm::MCOperand::createImm(7));
  Inst.addOperand(llvm::MCOperand::createExpr(
      llvm::MCConstantExpr::create(42, *State.Ctx)));
  Inst.addOperand(llvm::MCOperand::createReg(1));

  EXPECT_EQ(evalOperandAsConst(Inst, 0), 7);
  EXPECT_EQ(evalOperandAsConst(Inst, 1), 42);
  EXPECT_EQ(evalOperandAsConst(Inst, 2), std::nullopt); // register: not const
  EXPECT_EQ(evalOperandAsConst(Inst, 3), std::nullopt); // out of range
}

// -- canonical-op -------------------------------------------------------------

TEST(CanonicalOp, NameRoundTrip) {
  EXPECT_EQ(canonicalOpName(CanonicalOp::Unknown), "Unknown");
  EXPECT_EQ(canonicalOpName(CanonicalOp::S_MOV_B32), "S_MOV_B32");
  EXPECT_EQ(canonicalOpName(CanonicalOp::S_ENDPGM), "S_ENDPGM");
}

TEST(CanonicalOp, EveryValueIsNamed) {
  for (uint16_t I = 0;
       I < static_cast<uint16_t>(CanonicalOp::CanonicalOp_COUNT); ++I)
    EXPECT_FALSE(canonicalOpName(static_cast<CanonicalOp>(I)).empty());
}

// -- stripEncoding (pure string helper) ---------------------------------------

TEST(StripEncoding, DropsKnownSuffixes) {
  EXPECT_EQ(stripEncoding("v_mov_b32_e32"), "v_mov_b32");
  EXPECT_EQ(stripEncoding("v_add_f32_e64"), "v_add_f32");
  EXPECT_EQ(stripEncoding("v_cvt_f32_i32_vi"), "v_cvt_f32_i32");
}

TEST(StripEncoding, LeavesUnsuffixedUnchanged) {
  EXPECT_EQ(stripEncoding("s_mov_b32"), "s_mov_b32");
  EXPECT_EQ(stripEncoding("s_endpgm"), "s_endpgm");
}

// -- isa-profile --------------------------------------------------------------

TEST_F(DecoderTest, ISAProfileGfx942) {
  ISAProfile Profile = ISAProfile::fromSubtarget(*State.SubtargetInfo);
  EXPECT_EQ(Profile.waveSize(), 64u);
  EXPECT_FALSE(Profile.isWave32());
  EXPECT_TRUE(Profile.hasValidWaveSize());
  EXPECT_TRUE(Profile.hasAgpr());
  EXPECT_FALSE(Profile.hasGfx125UserSgprCountField());
}

TEST_F(DecoderTest, ISAProfileGfx1250) {
  llvm::Expected<std::unique_ptr<llvm::MCSubtargetInfo>> STIOrErr =
      buildSubtargetInfo(*State.Target, "gfx1250");
  ASSERT_TRUE(static_cast<bool>(STIOrErr))
      << llvm::toString(STIOrErr.takeError());
  ISAProfile Profile = ISAProfile::fromSubtarget(**STIOrErr);
  EXPECT_EQ(Profile.waveSize(), 32u);
  EXPECT_TRUE(Profile.isWave32());
  EXPECT_TRUE(Profile.hasValidWaveSize());
  EXPECT_FALSE(Profile.hasAgpr());
  EXPECT_TRUE(Profile.hasGfx125UserSgprCountField());
}

// -- decoded-inst bitfields ---------------------------------------------------

TEST(DecodedInstFlags, SizeAndCondRegBitsAreIndependent) {
  DecodedInst Di;
  EXPECT_EQ(Di.sizeInBytes(), 0u);
  EXPECT_FALSE(Di.defsScc());
  EXPECT_FALSE(Di.defsVcc());
  EXPECT_FALSE(Di.defsExec());

  // The size field is 5 bits; 20 is the AMDGPU maximum instruction length.
  Di.setSizeInBytes(20);
  Di.setDefsScc(true);
  Di.setDefsExec(true);
  EXPECT_EQ(Di.sizeInBytes(), 20u);
  EXPECT_TRUE(Di.defsScc());
  EXPECT_FALSE(Di.defsVcc());
  EXPECT_TRUE(Di.defsExec());

  // Toggling one condition-register bit leaves the size and the others intact.
  Di.setDefsVcc(true);
  Di.setDefsScc(false);
  EXPECT_EQ(Di.sizeInBytes(), 20u);
  EXPECT_FALSE(Di.defsScc());
  EXPECT_TRUE(Di.defsVcc());
  EXPECT_TRUE(Di.defsExec());
}

} // namespace
