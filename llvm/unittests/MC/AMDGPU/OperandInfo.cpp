//===- OperandInfo.cpp - AMDGPU MC operand information tests --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/AMDGPU/AMDGPUOperandInfo.h"
#include "llvm/TargetParser/Triple.h"
#include "gtest/gtest.h"
#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

using namespace llvm;

namespace {

using AMDGPU::MCOperandRole;

static_assert(static_cast<uint8_t>(MCOperandRole::VDst) == 0);
static_assert(static_cast<uint8_t>(MCOperandRole::Src0) == 1);
static_assert(static_cast<uint8_t>(MCOperandRole::Src1) == 2);
static_assert(static_cast<uint8_t>(MCOperandRole::Src2) == 3);
static_assert(static_cast<uint8_t>(MCOperandRole::Src0Modifiers) == 4);
static_assert(static_cast<uint8_t>(MCOperandRole::Src1Modifiers) == 5);
static_assert(static_cast<uint8_t>(MCOperandRole::Src2Modifiers) == 6);
static_assert(static_cast<uint8_t>(MCOperandRole::ScaleSrc0) == 7);
static_assert(static_cast<uint8_t>(MCOperandRole::ScaleSrc1) == 8);
static_assert(static_cast<uint8_t>(MCOperandRole::MatrixAFormat) == 9);
static_assert(static_cast<uint8_t>(MCOperandRole::MatrixBFormat) == 10);
static_assert(static_cast<uint8_t>(MCOperandRole::MatrixAScale) == 11);
static_assert(static_cast<uint8_t>(MCOperandRole::MatrixBScale) == 12);
static_assert(static_cast<uint8_t>(MCOperandRole::MatrixAScaleFormat) == 13);
static_assert(static_cast<uint8_t>(MCOperandRole::MatrixBScaleFormat) == 14);
static_assert(static_cast<uint8_t>(MCOperandRole::MatrixAReuse) == 15);
static_assert(static_cast<uint8_t>(MCOperandRole::MatrixBReuse) == 16);
static_assert(static_cast<uint8_t>(MCOperandRole::NegLo) == 17);
static_assert(static_cast<uint8_t>(MCOperandRole::NegHi) == 18);
static_assert(static_cast<uint8_t>(MCOperandRole::Clamp) == 19);
static_assert(static_cast<uint8_t>(MCOperandRole::OMod) == 20);
static_assert(static_cast<uint8_t>(MCOperandRole::OpSel) == 21);
static_assert(static_cast<uint8_t>(MCOperandRole::OpSelHi) == 22);

class AMDGPUOperandInfoTest : public testing::Test {
protected:
  Triple TT{"amdgcn-amd-amdhsa"};
  const Target *TheTarget = nullptr;
  std::unique_ptr<MCRegisterInfo> MRI;
  std::unique_ptr<MCAsmInfo> MAI;
  std::unique_ptr<const MCInstrInfo> MII;
  std::unique_ptr<MCSubtargetInfo> STI;
  std::unique_ptr<MCContext> Ctx;
  std::unique_ptr<MCDisassembler> DisAsm;

  static void SetUpTestSuite() {
    LLVMInitializeAMDGPUTargetInfo();
    LLVMInitializeAMDGPUTargetMC();
    LLVMInitializeAMDGPUDisassembler();
  }

  void initialize(StringRef CPU, StringRef Features = "") {
    std::string Error;
    TheTarget = TargetRegistry::lookupTarget(TT, Error);
    ASSERT_NE(TheTarget, nullptr) << Error;

    MRI.reset(TheTarget->createMCRegInfo(TT));
    ASSERT_NE(MRI, nullptr);
    MCTargetOptions Options;
    MAI.reset(TheTarget->createMCAsmInfo(*MRI, TT, Options));
    ASSERT_NE(MAI, nullptr);
    MII.reset(TheTarget->createMCInstrInfo());
    ASSERT_NE(MII, nullptr);
    STI.reset(TheTarget->createMCSubtargetInfo(TT, CPU, Features));
    ASSERT_NE(STI, nullptr);
    Ctx = std::make_unique<MCContext>(TT, *MAI, *MRI, *STI);
    DisAsm.reset(TheTarget->createMCDisassembler(*STI, *Ctx));
    ASSERT_NE(DisAsm, nullptr);
  }

  void decode(ArrayRef<uint8_t> Bytes, MCInst &Inst) {
    uint64_t Size = 0;
    ASSERT_EQ(DisAsm->getInstruction(Inst, Size, Bytes, /*Address=*/0, nulls()),
              MCDisassembler::Success);
    EXPECT_EQ(Size, Bytes.size());
  }

  void expectOpcode(const MCInst &Inst, StringRef Name) {
    ASSERT_LT(Inst.getOpcode(), MII->getNumOpcodes());
    EXPECT_EQ(MII->getName(Inst.getOpcode()), Name);
  }

  void expectRoleIndex(const MCInst &Inst, MCOperandRole Role,
                       unsigned ExpectedIndex) {
    std::optional<unsigned> Index = AMDGPU::getMCOperandIndex(Inst, Role);
    ASSERT_TRUE(Index);
    EXPECT_EQ(*Index, ExpectedIndex);
  }

  void expectImmRole(const MCInst &Inst, MCOperandRole Role,
                     unsigned ExpectedIndex, int64_t ExpectedValue) {
    expectRoleIndex(Inst, Role, ExpectedIndex);
    ASSERT_LT(ExpectedIndex, Inst.getNumOperands());
    const MCOperand &Operand = Inst.getOperand(ExpectedIndex);
    ASSERT_TRUE(Operand.isImm());
    EXPECT_EQ(Operand.getImm(), ExpectedValue);
  }

  void expectRegRole(const MCInst &Inst, MCOperandRole Role,
                     unsigned ExpectedIndex, StringRef ExpectedName) {
    expectRoleIndex(Inst, Role, ExpectedIndex);
    ASSERT_LT(ExpectedIndex, Inst.getNumOperands());
    const MCOperand &Operand = Inst.getOperand(ExpectedIndex);
    ASSERT_TRUE(Operand.isReg());
    EXPECT_EQ(MRI->getName(Operand.getReg()), ExpectedName);
  }
};

TEST_F(AMDGPUOperandInfoTest, DotProductDecodedRoles) {
  initialize("gfx1200");
  // The source modifiers in this encoding contain nonzero neg_lo and neg_hi
  // bits. The gfx12 decoder represents those bits in the source-modifier
  // operands rather than materializing the descriptor's trailing NegLo and
  // NegHi operands.
  constexpr std::array<uint8_t, 8> Bytes = {0x05, 0x41, 0x1a, 0xcc,
                                            0xfd, 0xd4, 0xf4, 0x3b};
  MCInst Inst;
  decode(Bytes, Inst);
  expectOpcode(Inst, "V_DOT2_F32_BF16_gfx12");
  ASSERT_EQ(MII->get(Inst.getOpcode()).getNumOperands(), 12u);
  ASSERT_EQ(Inst.getNumOperands(), 8u);

  expectRegRole(Inst, MCOperandRole::VDst, 0, "VGPR5");
  expectImmRole(Inst, MCOperandRole::Src0Modifiers, 1, 11);
  expectRegRole(Inst, MCOperandRole::Src0, 2, "SRC_SCC");
  expectImmRole(Inst, MCOperandRole::Src1Modifiers, 3, 8);
  expectRegRole(Inst, MCOperandRole::Src1, 4, "VCC_LO");
  expectImmRole(Inst, MCOperandRole::Src2Modifiers, 5, 8);
  expectRegRole(Inst, MCOperandRole::Src2, 6, "SRC_SCC");
  expectImmRole(Inst, MCOperandRole::Clamp, 7, 0);

  EXPECT_FALSE(AMDGPU::getMCOperandIndex(Inst, MCOperandRole::OpSel));
  EXPECT_FALSE(AMDGPU::getMCOperandIndex(Inst, MCOperandRole::OpSelHi));
  EXPECT_FALSE(AMDGPU::getMCOperandIndex(Inst, MCOperandRole::NegLo));
  EXPECT_FALSE(AMDGPU::getMCOperandIndex(Inst, MCOperandRole::NegHi));
  EXPECT_FALSE(AMDGPU::getMCOperandIndex(Inst, MCOperandRole::OMod));
}

TEST_F(AMDGPUOperandInfoTest, MaterializedPackedModifierRoles) {
  initialize("gfx1200");

  // The assembler-side form of this real opcode can materialize the canonical
  // trailing operands that the decoder folds into earlier modifier operands.
  // Verify those positive role mappings independently of the generated-table
  // sweep below.
  constexpr std::array<uint8_t, 8> Bytes = {0x05, 0x41, 0x1a, 0xcc,
                                            0xfd, 0xd4, 0xf4, 0x3b};
  MCInst Decoded;
  decode(Bytes, Decoded);

  MCInst Materialized;
  Materialized.setOpcode(Decoded.getOpcode());
  Materialized.addOperand(MCOperand::createReg(Decoded.getOperand(0).getReg()));
  Materialized.addOperand(MCOperand::createImm(11));
  Materialized.addOperand(MCOperand::createReg(Decoded.getOperand(2).getReg()));
  Materialized.addOperand(MCOperand::createImm(8));
  Materialized.addOperand(MCOperand::createReg(Decoded.getOperand(4).getReg()));
  Materialized.addOperand(MCOperand::createImm(8));
  Materialized.addOperand(MCOperand::createReg(Decoded.getOperand(6).getReg()));
  Materialized.addOperand(MCOperand::createImm(0));
  Materialized.addOperand(MCOperand::createImm(4));
  Materialized.addOperand(MCOperand::createImm(2));
  Materialized.addOperand(MCOperand::createImm(1));
  Materialized.addOperand(MCOperand::createImm(1));

  expectImmRole(Materialized, MCOperandRole::OpSel, 8, 4);
  expectImmRole(Materialized, MCOperandRole::OpSelHi, 9, 2);
  expectImmRole(Materialized, MCOperandRole::NegLo, 10, 1);
  expectImmRole(Materialized, MCOperandRole::NegHi, 11, 1);
}

TEST_F(AMDGPUOperandInfoTest, OutputModifierDecodedRoles) {
  initialize("gfx1200");
  // This encoding has a nonzero op_sel bit. Like neg_lo and neg_hi above, the
  // gfx12 decoder folds it into other operands and omits the trailing OpSel
  // descriptor operand.
  constexpr std::array<uint8_t, 12> Bytes = {
      0xff, 0xc3, 0x32, 0xd5, 0xff, 0xd6, 0x00, 0x7a, 0x0b, 0xfe, 0x00, 0x00};
  MCInst Inst;
  decode(Bytes, Inst);
  expectOpcode(Inst, "V_ADD_F16_t16_e64_gfx12");
  ASSERT_EQ(MII->get(Inst.getOpcode()).getNumOperands(), 8u);
  ASSERT_EQ(Inst.getNumOperands(), 7u);

  expectRegRole(Inst, MCOperandRole::VDst, 0, "VGPR255_HI16");
  expectImmRole(Inst, MCOperandRole::Src0Modifiers, 1, 11);
  expectImmRole(Inst, MCOperandRole::Src0, 2, 65035);
  expectImmRole(Inst, MCOperandRole::Src1Modifiers, 3, 3);
  expectRegRole(Inst, MCOperandRole::Src1, 4, "VCC_HI");
  expectImmRole(Inst, MCOperandRole::Clamp, 5, 1);
  expectImmRole(Inst, MCOperandRole::OMod, 6, 3);

  EXPECT_FALSE(AMDGPU::getMCOperandIndex(Inst, MCOperandRole::OpSel));
  EXPECT_FALSE(AMDGPU::getMCOperandIndex(Inst, MCOperandRole::Src2));
}

TEST_F(AMDGPUOperandInfoTest, LaneSaveRestoreDecodedRoles) {
  initialize("gfx1250");

  constexpr std::array<uint8_t, 8> WriteS30 = {0x29, 0x00, 0x61, 0xd7,
                                               0x1e, 0x0a, 0x01, 0x02};
  constexpr std::array<uint8_t, 8> WriteS31 = {0x29, 0x00, 0x61, 0xd7,
                                               0x1f, 0x0c, 0x01, 0x02};
  constexpr std::array<uint8_t, 8> ReadS30 = {0x1e, 0x00, 0x60, 0xd7,
                                              0x29, 0x0b, 0x01, 0x02};
  constexpr std::array<uint8_t, 8> ReadS31 = {0x1f, 0x00, 0x60, 0xd7,
                                              0x29, 0x0d, 0x01, 0x02};

  MCInst Write30;
  decode(WriteS30, Write30);
  expectOpcode(Write30, "V_WRITELANE_B32_e64_gfx12");
  expectRegRole(Write30, MCOperandRole::VDst, 0, "VGPR41");
  expectRegRole(Write30, MCOperandRole::Src0, 1, "SGPR30");
  expectImmRole(Write30, MCOperandRole::Src1, 2, 5);

  MCInst Write31;
  decode(WriteS31, Write31);
  expectOpcode(Write31, "V_WRITELANE_B32_e64_gfx12");
  expectRegRole(Write31, MCOperandRole::VDst, 0, "VGPR41");
  expectRegRole(Write31, MCOperandRole::Src0, 1, "SGPR31");
  expectImmRole(Write31, MCOperandRole::Src1, 2, 6);

  MCInst Read30;
  decode(ReadS30, Read30);
  expectOpcode(Read30, "V_READLANE_B32_e64_gfx12");
  expectRegRole(Read30, MCOperandRole::VDst, 0, "SGPR30");
  expectRegRole(Read30, MCOperandRole::Src0, 1, "VGPR41");
  expectImmRole(Read30, MCOperandRole::Src1, 2, 5);

  MCInst Read31;
  decode(ReadS31, Read31);
  expectOpcode(Read31, "V_READLANE_B32_e64_gfx12");
  expectRegRole(Read31, MCOperandRole::VDst, 0, "SGPR31");
  expectRegRole(Read31, MCOperandRole::Src0, 1, "VGPR41");
  expectImmRole(Read31, MCOperandRole::Src1, 2, 6);
}

TEST_F(AMDGPUOperandInfoTest, MatrixScaleSourceDecodedRoles) {
  initialize("gfx950");
  constexpr std::array<uint8_t, 16> Bytes = {0x00, 0x00, 0xac, 0xd3, 0x18, 0x33,
                                             0x02, 0x00, 0x00, 0x08, 0xad, 0xd3,
                                             0x04, 0x19, 0x52, 0x04};
  MCInst Inst;
  decode(Bytes, Inst);
  expectOpcode(Inst, "V_MFMA_SCALE_F32_16X16X128_F8F6F4_f8_f8_gfx940_vcd");

  expectRegRole(Inst, MCOperandRole::VDst, 0, "VGPR0_VGPR1_VGPR2_VGPR3");
  expectRegRole(Inst, MCOperandRole::Src0, 1,
                "VGPR4_VGPR5_VGPR6_VGPR7_VGPR8_VGPR9_VGPR10_VGPR11");
  expectRegRole(Inst, MCOperandRole::Src1, 2,
                "VGPR12_VGPR13_VGPR14_VGPR15_VGPR16_VGPR17_VGPR18_VGPR19");
  expectRegRole(Inst, MCOperandRole::Src2, 3, "VGPR20_VGPR21_VGPR22_VGPR23");
  expectRegRole(Inst, MCOperandRole::ScaleSrc0, 6, "VGPR24");
  expectRegRole(Inst, MCOperandRole::ScaleSrc1, 7, "VGPR25");
  expectImmRole(Inst, MCOperandRole::Src0Modifiers, 8, 0);
  expectImmRole(Inst, MCOperandRole::Src1Modifiers, 9, 0);

  EXPECT_FALSE(AMDGPU::getMCOperandIndex(Inst, MCOperandRole::MatrixAFormat));
  EXPECT_FALSE(AMDGPU::getMCOperandIndex(Inst, MCOperandRole::Src2Modifiers));
}

TEST_F(AMDGPUOperandInfoTest, MatrixSemanticDecodedRoles) {
  initialize("gfx1250");
  constexpr std::array<uint8_t, 16> Bytes = {0x00, 0x68, 0x3a, 0xcc, 0x02, 0x08,
                                             0x00, 0x0a, 0x00, 0x0c, 0x33, 0xcc,
                                             0x08, 0x31, 0xa2, 0x94};
  MCInst Inst;
  decode(Bytes, Inst);
  expectOpcode(Inst, "V_WMMA_SCALE16_F32_16X16X128_F8F6F4_f8_f6_w32_gfx1250");

  expectRegRole(Inst, MCOperandRole::VDst, 0,
                "VGPR0_VGPR1_VGPR2_VGPR3_VGPR4_VGPR5_VGPR6_VGPR7");
  expectRegRole(Inst, MCOperandRole::Src0, 1,
                "VGPR8_VGPR9_VGPR10_VGPR11_VGPR12_VGPR13_VGPR14_VGPR15_"
                "VGPR16_VGPR17_VGPR18_VGPR19_VGPR20_VGPR21_VGPR22_VGPR23");
  expectRegRole(Inst, MCOperandRole::Src1, 2,
                "VGPR24_VGPR25_VGPR26_VGPR27_VGPR28_VGPR29_VGPR30_VGPR31_"
                "VGPR32_VGPR33_VGPR34_VGPR35");
  expectImmRole(Inst, MCOperandRole::Src2Modifiers, 3, 3);
  expectRegRole(Inst, MCOperandRole::Src2, 4,
                "VGPR40_VGPR41_VGPR42_VGPR43_VGPR44_VGPR45_VGPR46_VGPR47");
  expectRegRole(Inst, MCOperandRole::ScaleSrc0, 5, "SGPR2_SGPR3");
  expectRegRole(Inst, MCOperandRole::ScaleSrc1, 6, "SGPR4_SGPR5");
  expectImmRole(Inst, MCOperandRole::MatrixAFormat, 7, 1);
  expectImmRole(Inst, MCOperandRole::MatrixBFormat, 8, 2);
  expectImmRole(Inst, MCOperandRole::MatrixAScale, 9, 1);
  expectImmRole(Inst, MCOperandRole::MatrixBScale, 10, 1);
  expectImmRole(Inst, MCOperandRole::MatrixAScaleFormat, 11, 0);
  expectImmRole(Inst, MCOperandRole::MatrixBScaleFormat, 12, 0);
  expectImmRole(Inst, MCOperandRole::MatrixAReuse, 13, 1);
  expectImmRole(Inst, MCOperandRole::MatrixBReuse, 14, 1);

  EXPECT_FALSE(AMDGPU::getMCOperandIndex(Inst, MCOperandRole::NegLo));
  EXPECT_FALSE(AMDGPU::getMCOperandIndex(Inst, MCOperandRole::NegHi));
  EXPECT_FALSE(AMDGPU::getMCOperandIndex(Inst, MCOperandRole::Clamp));
}

TEST_F(AMDGPUOperandInfoTest, PublicRolesMatchGeneratedMetadata) {
  initialize("gfx1200");

  struct TestRole {
    MCOperandRole Public;
    AMDGPU::OpName Generated;
  };
  constexpr std::array Roles = {
      TestRole{MCOperandRole::VDst, AMDGPU::OpName::vdst},
      TestRole{MCOperandRole::Src0, AMDGPU::OpName::src0},
      TestRole{MCOperandRole::Src1, AMDGPU::OpName::src1},
      TestRole{MCOperandRole::Src2, AMDGPU::OpName::src2},
      TestRole{MCOperandRole::Src0Modifiers, AMDGPU::OpName::src0_modifiers},
      TestRole{MCOperandRole::Src1Modifiers, AMDGPU::OpName::src1_modifiers},
      TestRole{MCOperandRole::Src2Modifiers, AMDGPU::OpName::src2_modifiers},
      TestRole{MCOperandRole::ScaleSrc0, AMDGPU::OpName::scale_src0},
      TestRole{MCOperandRole::ScaleSrc1, AMDGPU::OpName::scale_src1},
      TestRole{MCOperandRole::MatrixAFormat, AMDGPU::OpName::matrix_a_fmt},
      TestRole{MCOperandRole::MatrixBFormat, AMDGPU::OpName::matrix_b_fmt},
      TestRole{MCOperandRole::MatrixAScale, AMDGPU::OpName::matrix_a_scale},
      TestRole{MCOperandRole::MatrixBScale, AMDGPU::OpName::matrix_b_scale},
      TestRole{MCOperandRole::MatrixAScaleFormat,
               AMDGPU::OpName::matrix_a_scale_fmt},
      TestRole{MCOperandRole::MatrixBScaleFormat,
               AMDGPU::OpName::matrix_b_scale_fmt},
      TestRole{MCOperandRole::MatrixAReuse, AMDGPU::OpName::matrix_a_reuse},
      TestRole{MCOperandRole::MatrixBReuse, AMDGPU::OpName::matrix_b_reuse},
      TestRole{MCOperandRole::NegLo, AMDGPU::OpName::neg_lo},
      TestRole{MCOperandRole::NegHi, AMDGPU::OpName::neg_hi},
      TestRole{MCOperandRole::Clamp, AMDGPU::OpName::clamp},
      TestRole{MCOperandRole::OMod, AMDGPU::OpName::omod},
      TestRole{MCOperandRole::OpSel, AMDGPU::OpName::op_sel},
      TestRole{MCOperandRole::OpSelHi, AMDGPU::OpName::op_sel_hi},
  };

  for (unsigned Opcode = 0, E = MII->getNumOpcodes(); Opcode != E; ++Opcode) {
    MCInst Inst;
    Inst.setOpcode(Opcode);
    unsigned OperandCount = MII->get(Opcode).getNumOperands();
    for (unsigned I = 0; I != OperandCount; ++I)
      Inst.addOperand(MCOperand::createImm(0));

    for (const TestRole &Role : Roles) {
      int16_t Expected = AMDGPU::getNamedOperandIdx(Opcode, Role.Generated);
      std::optional<unsigned> Actual =
          AMDGPU::getMCOperandIndex(Inst, Role.Public);
      if (Expected < 0 || static_cast<unsigned>(Expected) >= OperandCount)
        EXPECT_FALSE(Actual)
            << "Opcode " << Opcode << " (" << MII->getName(Opcode) << "), role "
            << static_cast<unsigned>(Role.Public);
      else {
        ASSERT_TRUE(Actual)
            << "Opcode " << Opcode << " (" << MII->getName(Opcode) << "), role "
            << static_cast<unsigned>(Role.Public);
        EXPECT_EQ(*Actual, static_cast<unsigned>(Expected));
      }
    }
  }
}

TEST_F(AMDGPUOperandInfoTest, RejectsInvalidInputs) {
  initialize("gfx1200");

  MCInst InvalidOpcode;
  InvalidOpcode.setOpcode(MII->getNumOpcodes());
  InvalidOpcode.addOperand(MCOperand::createImm(0));
  EXPECT_FALSE(AMDGPU::getMCOperandIndex(InvalidOpcode, MCOperandRole::VDst));

  MCInst ValidOpcode;
  ValidOpcode.setOpcode(0);
  ValidOpcode.addOperand(MCOperand::createImm(0));
  EXPECT_FALSE(
      AMDGPU::getMCOperandIndex(ValidOpcode, static_cast<MCOperandRole>(255)));
}

} // namespace
