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
// capability queries (isa-profile), the decoded-instruction model
// (decoded-inst), the MC-opcode to CanonicalOp map (opcode-map), and the .text
// scan (decode). Each exercises the piece directly, without a code object or
// the raiser, so the coverage matches what the decoder alone provides.
//
//===----------------------------------------------------------------------===//

#include "hotswap/decoder/canonical-op.h"
#include "hotswap/decoder/decode.h"
#include "hotswap/decoder/decoded-inst.h"
#include "hotswap/decoder/isa-profile.h"
#include "hotswap/decoder/mc-state.h"
#include "hotswap/decoder/opcode-map.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <initializer_list>
#include <mutex>
#include <set>
#include <vector>

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

TEST_F(DecoderTest, StripRegEncodingDropsOnlyTheVariantSuffix) {
  static constexpr llvm::StringRef KSuffixes[] = {"_ci", "_vi", "_gfx9plus",
                                                  "_gfx11plus", "_gfxpre11"};
  const llvm::MCRegisterInfo &MRI = *State.RegInfo;
  unsigned Stripped = 0;
  for (unsigned Reg = 1; Reg < MRI.getNumRegs(); ++Reg) {
    llvm::MCRegister Base = stripRegEncoding(Reg);
    EXPECT_EQ(stripRegEncoding(Base), Base) << MRI.getName(Reg);
    if (Base == Reg)
      continue;
    ++Stripped;
    llvm::StringRef Name = MRI.getName(Reg);
    llvm::StringRef BaseName = MRI.getName(Base);
    ASSERT_TRUE(Name.starts_with(BaseName))
        << Name.str() << " does not name a variant of " << BaseName.str();
    EXPECT_TRUE(llvm::is_contained(KSuffixes, Name.substr(BaseName.size())))
        << Name.str() << " does not name a variant of " << BaseName.str();
  }
  // Pinned so a register gaining or losing a variant fails here.
  EXPECT_EQ(Stripped, 74u);
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

// -- opcode-map ---------------------------------------------------------------

// Resolve an MC opcode through the disassembler so the map is queried with the
// same opcode a real .text scan produces.
unsigned opcodeOf(MCState &State, llvm::ArrayRef<uint8_t> Bytes) {
  llvm::MCInst Inst;
  uint64_t Size = 0;
  EXPECT_EQ(State.Disasm->getInstruction(Inst, Size, Bytes, /*Address=*/0,
                                         llvm::nulls()),
            llvm::MCDisassembler::Success);
  return Inst.getOpcode();
}

TEST_F(DecoderTest, OpcodeMapTagsTableEntries) {
  OpcodeMap Map;
  Map.build(*State.InstrInfo);
  EXPECT_EQ(Map.lookup(opcodeOf(State, SMovB32Bytes)), CanonicalOp::S_MOV_B32);
  EXPECT_EQ(Map.lookup(opcodeOf(State, SEndpgmBytes)), CanonicalOp::S_ENDPGM);
}

TEST_F(DecoderTest, OpcodeMapReturnsUnknownForUnmappedOpcode) {
  OpcodeMap Map;
  Map.build(*State.InstrInfo);
  // v_mov_b32 has no kCanonTable row, so it stays Unknown.
  EXPECT_EQ(Map.lookup(opcodeOf(State, VMovB32Bytes)), CanonicalOp::Unknown);
  EXPECT_EQ(Map.lookup(State.InstrInfo->getNumOpcodes()), CanonicalOp::Unknown);
}

// -- decode -------------------------------------------------------------------

// Concatenate instruction encodings into one .text image.
std::vector<uint8_t>
textOf(std::initializer_list<llvm::ArrayRef<uint8_t>> Insts) {
  std::vector<uint8_t> Bytes;
  for (llvm::ArrayRef<uint8_t> Inst : Insts)
    Bytes.insert(Bytes.end(), Inst.begin(), Inst.end());
  return Bytes;
}

TEST_F(DecoderTest, DecodeKernelWalksToProgramEnd) {
  OpcodeMap Map;
  Map.build(*State.InstrInfo);
  std::vector<uint8_t> Text =
      textOf({SMovB32Bytes, VMovB32Bytes, SEndpgmBytes});

  llvm::Expected<DecodeResult> ResultOrErr =
      decodeKernel(State, Map, Text, /*KernelOffset=*/0);
  ASSERT_TRUE(static_cast<bool>(ResultOrErr))
      << llvm::toString(ResultOrErr.takeError());

  ASSERT_EQ(ResultOrErr->Insts.size(), 3u);
  EXPECT_EQ(ResultOrErr->Insts[0].CanonOp, CanonicalOp::S_MOV_B32);
  EXPECT_EQ(ResultOrErr->Insts[1].CanonOp, CanonicalOp::Unknown);
  EXPECT_EQ(ResultOrErr->Insts[2].CanonOp, CanonicalOp::S_ENDPGM);
  EXPECT_EQ(ResultOrErr->Insts[0].Offset, 0u);
  EXPECT_EQ(ResultOrErr->Insts[1].Offset, 4u);
  EXPECT_EQ(ResultOrErr->Insts[2].Offset, 8u);
  EXPECT_EQ(ResultOrErr->Insts[0].sizeInBytes(), 4u);
  EXPECT_EQ(ResultOrErr->BlockStarts, (std::set<uint64_t>{0}));
}

TEST_F(DecoderTest, DecodeKernelStopsAtProgramEnd) {
  OpcodeMap Map;
  Map.build(*State.InstrInfo);
  // Bytes after the terminator belong to the next kernel, not this one.
  std::vector<uint8_t> Text = textOf({SEndpgmBytes, SMovB32Bytes});

  llvm::Expected<DecodeResult> ResultOrErr =
      decodeKernel(State, Map, Text, /*KernelOffset=*/0);
  ASSERT_TRUE(static_cast<bool>(ResultOrErr))
      << llvm::toString(ResultOrErr.takeError());
  ASSERT_EQ(ResultOrErr->Insts.size(), 1u);
  EXPECT_EQ(ResultOrErr->Insts[0].CanonOp, CanonicalOp::S_ENDPGM);
}

TEST_F(DecoderTest, DecodeKernelHonoursOffsetAndEnd) {
  OpcodeMap Map;
  Map.build(*State.InstrInfo);
  std::vector<uint8_t> Text =
      textOf({SEndpgmBytes, SMovB32Bytes, SEndpgmBytes});

  llvm::Expected<DecodeResult> ResultOrErr =
      decodeKernel(State, Map, Text, /*KernelOffset=*/4, /*KernelEndOffset=*/8);
  ASSERT_TRUE(static_cast<bool>(ResultOrErr))
      << llvm::toString(ResultOrErr.takeError());
  ASSERT_EQ(ResultOrErr->Insts.size(), 1u);
  EXPECT_EQ(ResultOrErr->Insts[0].CanonOp, CanonicalOp::S_MOV_B32);
  EXPECT_EQ(ResultOrErr->Insts[0].Offset, 4u);
  EXPECT_EQ(ResultOrErr->BlockStarts, (std::set<uint64_t>{4}));
}

TEST_F(DecoderTest, DecodeKernelRejectsTruncatedInstruction) {
  OpcodeMap Map;
  Map.build(*State.InstrInfo);
  // A whole s_mov_b32 followed by half an s_endpgm.
  std::vector<uint8_t> Text = textOf({SMovB32Bytes, SEndpgmBytes});
  Text.resize(6);

  llvm::Expected<DecodeResult> ResultOrErr =
      decodeKernel(State, Map, Text, /*KernelOffset=*/0);
  ASSERT_FALSE(static_cast<bool>(ResultOrErr));
  EXPECT_EQ(llvm::toString(ResultOrErr.takeError()),
            "hotswap: decodeKernel: cannot decode instruction at .text offset "
            "0x4 (fail)");
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
