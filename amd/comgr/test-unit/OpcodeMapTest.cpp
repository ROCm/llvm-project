//===- OpcodeMapTest.cpp - canonical opcode table tests -------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap/opcode-map.h"

#include "comgr-device-libs.h"

#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/Support/Error.h"

#include "gtest/gtest.h"

#include <string>
#include <tuple>

TEST(DeviceLibs, SelectOCMLSupportLibrariesForGfx942) {
  llvm::SmallVector<std::string, 8> Names;
  if (llvm::Error E = COMGR::getOCMLDeviceLibraryNames("gfx942", 64, Names))
    FAIL() << llvm::toString(std::move(E));

  llvm::SmallVector<llvm::StringRef, 8> Expected = {
      "ocml.bc",
      "ockl.bc",
      "oclc_abi_version_600.bc",
      "oclc_isa_version_942.bc",
      "oclc_finite_only_off.bc",
      "oclc_unsafe_math_off.bc",
      "oclc_wavefrontsize64_on.bc",
  };
  ASSERT_EQ(Names.size(), Expected.size());
  for (size_t I = 0; I < Expected.size(); ++I)
    EXPECT_EQ(Names[I], Expected[I]);

  for (llvm::StringRef Name : Names) {
    bool Found = false;
    for (const auto &Lib : COMGR::getDeviceLibraries()) {
      if (std::get<0>(Lib) == Name) {
        Found = true;
        break;
      }
    }
    EXPECT_TRUE(Found) << Name.str();
  }
}

TEST(DeviceLibs, SelectOCMLSupportLibrariesForGenericGfx) {
  llvm::SmallVector<std::string, 8> Names;
  if (llvm::Error E =
          COMGR::getOCMLDeviceLibraryNames("gfx9-generic", 64, Names))
    FAIL() << llvm::toString(std::move(E));

  ASSERT_EQ(Names.size(), 7u);
  EXPECT_EQ(Names[3], "oclc_isa_version_9_generic.bc");
}

TEST(DeviceLibs, SelectOCMLSupportLibrariesRejectsInvalidInputs) {
  llvm::SmallVector<std::string, 8> Names;
  std::string Error;
  Error = llvm::toString(
      COMGR::getOCMLDeviceLibraryNames("amdgcn-amd-amdhsa--gfx942", 64, Names));
  EXPECT_NE(Error.find("known AMDGPU processor"), std::string::npos) << Error;

  Error = llvm::toString(COMGR::getOCMLDeviceLibraryNames("gfx999", 64, Names));
  EXPECT_NE(Error.find("known AMDGPU processor"), std::string::npos) << Error;

  Error = llvm::toString(COMGR::getOCMLDeviceLibraryNames("gfx942", 96, Names));
  EXPECT_NE(Error.find("wave size 96"), std::string::npos) << Error;
}

// Opcodes the raiser does not model resolve to `CanonicalOp::Unknown`, and so
// does anything past the end of the table.
TEST(OpcodeMap, UnmodelledOpcodesAreUnknown) {
  // PHI (opcode 0) is a target-independent CodeGen pseudo the disassembler
  // never produces.
  EXPECT_EQ(COMGR::hotswap::canonicalOpFor(0),
            COMGR::hotswap::CanonicalOp::Unknown);
  EXPECT_EQ(COMGR::hotswap::canonicalOpFor(llvm::AMDGPU::INSTRUCTION_LIST_END),
            COMGR::hotswap::CanonicalOp::Unknown);
  EXPECT_EQ(COMGR::hotswap::canonicalOpFor(~0u),
            COMGR::hotswap::CanonicalOp::Unknown);
}

// The table is one flat array covering every AMDGPU generation, so gfx9-era
// real opcodes resolve alongside the gfx12/gfx1250 ones the rest of this file
// checks.
TEST(OpcodeMap, PreGfx12RealOpcodesMapToCanonicalOps) {
  EXPECT_EQ(COMGR::hotswap::canonicalOpFor(llvm::AMDGPU::S_ENDPGM_gfx6_gfx7),
            COMGR::hotswap::CanonicalOp::S_ENDPGM);
  EXPECT_EQ(COMGR::hotswap::canonicalOpFor(llvm::AMDGPU::S_ENDPGM_vi),
            COMGR::hotswap::CanonicalOp::S_ENDPGM);

  // All four encodings of one VOP2 collapse onto the same canonical opcode.
  // The SDWA and DPP arms of the canonicalization chain have no other
  // coverage in this file.
  EXPECT_EQ(COMGR::hotswap::canonicalOpFor(llvm::AMDGPU::V_ADD_F32_e32_vi),
            COMGR::hotswap::CanonicalOp::V_ADD_F32);
  EXPECT_EQ(COMGR::hotswap::canonicalOpFor(llvm::AMDGPU::V_ADD_F32_e64_vi),
            COMGR::hotswap::CanonicalOp::V_ADD_F32);
  EXPECT_EQ(COMGR::hotswap::canonicalOpFor(llvm::AMDGPU::V_ADD_F32_sdwa_vi),
            COMGR::hotswap::CanonicalOp::V_ADD_F32);
  EXPECT_EQ(COMGR::hotswap::canonicalOpFor(llvm::AMDGPU::V_ADD_F32_dpp_vi),
            COMGR::hotswap::CanonicalOp::V_ADD_F32);
}

// The V_CMP / V_CMPX family collapses onto two canonical opcodes plus a
// metadata side table, rather than getting one CanonicalOp each.
TEST(OpcodeMap, VCmpFamilyCarriesPredicateMetadata) {
  EXPECT_EQ(COMGR::hotswap::canonicalOpFor(llvm::AMDGPU::V_CMP_EQ_U32_e64_vi),
            COMGR::hotswap::CanonicalOp::V_CMP);

  const COMGR::hotswap::VCmpMeta *M =
      COMGR::hotswap::vcmpMetaFor(llvm::AMDGPU::V_CMP_EQ_U32_e64_vi);
  ASSERT_NE(M, nullptr);
  EXPECT_EQ(M->Pred, llvm::CmpInst::ICMP_EQ);
  EXPECT_EQ(M->Bits, 32u);
  EXPECT_FALSE(M->IsFloat);
  EXPECT_FALSE(M->IsClass);

  // Non-compare opcodes have no metadata.
  EXPECT_EQ(COMGR::hotswap::vcmpMetaFor(llvm::AMDGPU::V_ADD_F32_e64_vi),
            nullptr);
}

TEST(OpcodeMap, Gfx1250AddMinRealOpcodeMapsToSemOp) {
  EXPECT_EQ(
      COMGR::hotswap::canonicalOpFor(llvm::AMDGPU::V_ADD_MIN_U32_e64_gfx1250),
      COMGR::hotswap::CanonicalOp::V_ADD_MIN_U32);
}

TEST(OpcodeMap, Gfx1250SubNcU16RealOpcodeMapsToSemOp) {
  EXPECT_EQ(COMGR::hotswap::canonicalOpFor(
                llvm::AMDGPU::V_SUB_NC_U16_fake16_e64_gfx12),
            COMGR::hotswap::CanonicalOp::V_SUB_NC_U16);
  EXPECT_EQ(
      COMGR::hotswap::canonicalOpFor(llvm::AMDGPU::V_SUB_NC_U16_t16_e64_gfx12),
      COMGR::hotswap::CanonicalOp::V_SUB_NC_U16);
  EXPECT_EQ(COMGR::hotswap::canonicalOpFor(
                llvm::AMDGPU::V_SUB_NC_U16_fake16_e64_dpp_gfx12),
            COMGR::hotswap::CanonicalOp::V_SUB_NC_U16);
  EXPECT_EQ(COMGR::hotswap::canonicalOpFor(
                llvm::AMDGPU::V_SUB_NC_U16_fake16_e64_dpp8_gfx12),
            COMGR::hotswap::CanonicalOp::Unknown);
}

TEST(OpcodeMap, Gfx1250ScalarF16ToF32RealOpcodesMapToCanonicalOps) {
  EXPECT_EQ(COMGR::hotswap::canonicalOpFor(llvm::AMDGPU::S_CVT_F16_F32_gfx12),
            COMGR::hotswap::CanonicalOp::S_CVT_F16_F32);
  EXPECT_EQ(COMGR::hotswap::canonicalOpFor(llvm::AMDGPU::S_CVT_F32_F16_gfx12),
            COMGR::hotswap::CanonicalOp::S_CVT_F32_F16);
  EXPECT_EQ(
      COMGR::hotswap::canonicalOpFor(llvm::AMDGPU::S_CVT_HI_F32_F16_gfx12),
      COMGR::hotswap::CanonicalOp::S_CVT_HI_F32_F16);
}

TEST(OpcodeMap, Gfx1250VectorF32F64RealOpcodesMapToCanonicalOps) {
  EXPECT_EQ(
      COMGR::hotswap::canonicalOpFor(llvm::AMDGPU::V_CVT_F32_F64_e64_gfx12),
      COMGR::hotswap::CanonicalOp::V_CVT_F32_F64);
  EXPECT_EQ(
      COMGR::hotswap::canonicalOpFor(llvm::AMDGPU::V_CVT_F64_F32_e64_gfx12),
      COMGR::hotswap::CanonicalOp::V_CVT_F64_F32);
}

// Regression for rocm-systems#144: these gfx12 FP64 VOP1 ops have both
// e32 and e64 real encodings. CanonicalOpcodes.td only lists the e64 pseudo,
// so the e32 form must canonicalize through AMDGPU::getVOPe64() while the
// generated table is built.
// Pin the reported sqrt case and the RCP/RSQ siblings with the same shape.
TEST(OpcodeMap, Gfx1250F64TransBothEncodingsMapToCanonicalOps) {
  // v_sqrt_f64 -- the opcode from #144.
  EXPECT_EQ(COMGR::hotswap::canonicalOpFor(llvm::AMDGPU::V_SQRT_F64_e64_gfx12),
            COMGR::hotswap::CanonicalOp::V_SQRT_F64);
  EXPECT_EQ(COMGR::hotswap::canonicalOpFor(llvm::AMDGPU::V_SQRT_F64_e32_gfx12),
            COMGR::hotswap::CanonicalOp::V_SQRT_F64);
  // Siblings with the same encoding shape.
  EXPECT_EQ(COMGR::hotswap::canonicalOpFor(llvm::AMDGPU::V_RCP_F64_e64_gfx12),
            COMGR::hotswap::CanonicalOp::V_RCP_F64);
  EXPECT_EQ(COMGR::hotswap::canonicalOpFor(llvm::AMDGPU::V_RCP_F64_e32_gfx12),
            COMGR::hotswap::CanonicalOp::V_RCP_F64);
  EXPECT_EQ(COMGR::hotswap::canonicalOpFor(llvm::AMDGPU::V_RSQ_F64_e64_gfx12),
            COMGR::hotswap::CanonicalOp::V_RSQ_F64);
  EXPECT_EQ(COMGR::hotswap::canonicalOpFor(llvm::AMDGPU::V_RSQ_F64_e32_gfx12),
            COMGR::hotswap::CanonicalOp::V_RSQ_F64);
}

TEST(OpcodeMap, Gfx1250TanhF32RealOpcodeMapsToCanonicalOp) {
  EXPECT_EQ(
      COMGR::hotswap::canonicalOpFor(llvm::AMDGPU::V_TANH_F32_e64_gfx1250),
      COMGR::hotswap::CanonicalOp::V_TANH_F32);
}

TEST(OpcodeMap, Gfx1250TanhF16RealOpcodeMapsToCanonicalOp) {
  EXPECT_EQ(COMGR::hotswap::canonicalOpFor(
                llvm::AMDGPU::V_TANH_F16V_TANH_F16_t16_e64_gfx1250),
            COMGR::hotswap::CanonicalOp::V_TANH_F16);
  EXPECT_EQ(COMGR::hotswap::canonicalOpFor(
                llvm::AMDGPU::V_TANH_F16V_TANH_F16_fake16_e64_gfx1250),
            COMGR::hotswap::CanonicalOp::V_TANH_F16);
}

TEST(OpcodeMap, Gfx1250AddSubNcI16RealOpcodesMapToSemOps) {
  EXPECT_EQ(COMGR::hotswap::canonicalOpFor(
                llvm::AMDGPU::V_ADD_NC_I16V_ADD_I16_fake16_e64_gfx12),
            COMGR::hotswap::CanonicalOp::V_ADD_NC_I16);
  EXPECT_EQ(COMGR::hotswap::canonicalOpFor(
                llvm::AMDGPU::V_ADD_NC_I16V_ADD_I16_t16_e64_gfx12),
            COMGR::hotswap::CanonicalOp::V_ADD_NC_I16);
  EXPECT_EQ(COMGR::hotswap::canonicalOpFor(
                llvm::AMDGPU::V_ADD_NC_I16V_ADD_I16_fake16_e64_dpp_gfx12),
            COMGR::hotswap::CanonicalOp::V_ADD_NC_I16);
  EXPECT_EQ(COMGR::hotswap::canonicalOpFor(
                llvm::AMDGPU::V_ADD_NC_I16V_ADD_I16_fake16_e64_dpp8_gfx12),
            COMGR::hotswap::CanonicalOp::Unknown);
  EXPECT_EQ(COMGR::hotswap::canonicalOpFor(
                llvm::AMDGPU::V_SUB_NC_I16V_SUB_I16_fake16_e64_gfx12),
            COMGR::hotswap::CanonicalOp::V_SUB_NC_I16);
  EXPECT_EQ(COMGR::hotswap::canonicalOpFor(
                llvm::AMDGPU::V_SUB_NC_I16V_SUB_I16_t16_e64_gfx12),
            COMGR::hotswap::CanonicalOp::V_SUB_NC_I16);
  EXPECT_EQ(COMGR::hotswap::canonicalOpFor(
                llvm::AMDGPU::V_SUB_NC_I16V_SUB_I16_fake16_e64_dpp_gfx12),
            COMGR::hotswap::CanonicalOp::V_SUB_NC_I16);
  EXPECT_EQ(COMGR::hotswap::canonicalOpFor(
                llvm::AMDGPU::V_SUB_NC_I16V_SUB_I16_fake16_e64_dpp8_gfx12),
            COMGR::hotswap::CanonicalOp::Unknown);
}

TEST(OpcodeMap, Gfx1250Min3RealOpcodeMapsToSemOp) {
  EXPECT_EQ(COMGR::hotswap::canonicalOpFor(llvm::AMDGPU::V_MIN3_U32_e64_gfx12),
            COMGR::hotswap::CanonicalOp::V_MIN3_U32);
}

TEST(OpcodeMap, Gfx1250Dot4I32IU8RealOpcodeMapsToSemOp) {
  EXPECT_EQ(COMGR::hotswap::canonicalOpFor(llvm::AMDGPU::V_DOT4_I32_IU8),
            COMGR::hotswap::CanonicalOp::V_DOT4_I32_IU8);
  EXPECT_EQ(COMGR::hotswap::canonicalOpFor(llvm::AMDGPU::V_DOT4_I32_IU8_gfx12),
            COMGR::hotswap::CanonicalOp::V_DOT4_I32_IU8);
}

TEST(OpcodeMap, Gfx1250PkFmaF16RealOpcodeMapsToSemOp) {
  EXPECT_EQ(COMGR::hotswap::canonicalOpFor(llvm::AMDGPU::V_PK_FMA_F16_gfx12),
            COMGR::hotswap::CanonicalOp::V_PK_FMA_F16);
}

TEST(OpcodeMap, Gfx1250PkAddF16RealOpcodeMapsToSemOp) {
  EXPECT_EQ(COMGR::hotswap::canonicalOpFor(llvm::AMDGPU::V_PK_ADD_F16_gfx12),
            COMGR::hotswap::CanonicalOp::V_PK_ADD_F16);
}

TEST(OpcodeMap, Gfx1250PkAddBF16RealOpcodeMapsToSemOp) {
  EXPECT_EQ(COMGR::hotswap::canonicalOpFor(llvm::AMDGPU::V_PK_ADD_BF16_gfx1250),
            COMGR::hotswap::CanonicalOp::V_PK_ADD_BF16);
}

TEST(OpcodeMap, Gfx1250PkFmaBF16RealOpcodeMapsToSemOp) {
  EXPECT_EQ(COMGR::hotswap::canonicalOpFor(llvm::AMDGPU::V_PK_FMA_BF16_gfx1250),
            COMGR::hotswap::CanonicalOp::V_PK_FMA_BF16);
}

TEST(OpcodeMap, Gfx1250PkBF16SiblingsRealOpcodesMapToSemOps) {
  EXPECT_EQ(COMGR::hotswap::canonicalOpFor(llvm::AMDGPU::V_PK_MUL_BF16_gfx1250),
            COMGR::hotswap::CanonicalOp::V_PK_MUL_BF16);
  EXPECT_EQ(
      COMGR::hotswap::canonicalOpFor(llvm::AMDGPU::V_PK_MIN_NUM_BF16_gfx1250),
      COMGR::hotswap::CanonicalOp::V_PK_MIN_NUM_BF16);
  EXPECT_EQ(
      COMGR::hotswap::canonicalOpFor(llvm::AMDGPU::V_PK_MAX_NUM_BF16_gfx1250),
      COMGR::hotswap::CanonicalOp::V_PK_MAX_NUM_BF16);
}

TEST(OpcodeMap, Gfx1250FmaMixF16HalfResultRealOpcodesMapToSemOps) {
  EXPECT_EQ(COMGR::hotswap::canonicalOpFor(llvm::AMDGPU::V_FMA_MIXLO_F16_gfx12),
            COMGR::hotswap::CanonicalOp::V_FMA_MIXLO_F16);
  EXPECT_EQ(COMGR::hotswap::canonicalOpFor(llvm::AMDGPU::V_FMA_MIXHI_F16_gfx12),
            COMGR::hotswap::CanonicalOp::V_FMA_MIXHI_F16);
  EXPECT_EQ(
      COMGR::hotswap::canonicalOpFor(llvm::AMDGPU::V_FMA_MIXLO_F16_dpp_gfx12),
      COMGR::hotswap::CanonicalOp::V_FMA_MIXLO_F16);
  EXPECT_EQ(
      COMGR::hotswap::canonicalOpFor(llvm::AMDGPU::V_FMA_MIXHI_F16_dpp8_gfx12),
      COMGR::hotswap::CanonicalOp::Unknown);
}

TEST(OpcodeMap, Gfx1250FmaMixBF16HalfResultRealOpcodesMapToSemOps) {
  EXPECT_EQ(
      COMGR::hotswap::canonicalOpFor(llvm::AMDGPU::V_FMA_MIXLO_BF16_gfx1250),
      COMGR::hotswap::CanonicalOp::V_FMA_MIXLO_BF16);
  EXPECT_EQ(
      COMGR::hotswap::canonicalOpFor(llvm::AMDGPU::V_FMA_MIXHI_BF16_gfx1250),
      COMGR::hotswap::CanonicalOp::V_FMA_MIXHI_BF16);
  EXPECT_EQ(COMGR::hotswap::canonicalOpFor(
                llvm::AMDGPU::V_FMA_MIXLO_BF16_dpp_gfx1250),
            COMGR::hotswap::CanonicalOp::V_FMA_MIXLO_BF16);
  EXPECT_EQ(COMGR::hotswap::canonicalOpFor(
                llvm::AMDGPU::V_FMA_MIXHI_BF16_dpp8_gfx1250),
            COMGR::hotswap::CanonicalOp::Unknown);
}

TEST(OpcodeMap, Gfx1250MadI32I24RealOpcodeMapsToSemOp) {
  EXPECT_EQ(
      COMGR::hotswap::canonicalOpFor(llvm::AMDGPU::V_MAD_I32_I24_e64_gfx12),
      COMGR::hotswap::CanonicalOp::V_MAD_I32_I24);
}

TEST(OpcodeMap, Gfx1250CvtScalef32Pk8Fp8F32RealOpcodeMapsToCanonicalOp) {
  // V_CVT_SCALEF32_PK8_FP8_F32: gfx1250-only packed-8 scaled FP8
  // conversion (VOP3 opcode 0x2c3, profile VOP_V2I32_V8F32_F32 in
  // VOP3Instructions.td:1883).
  EXPECT_EQ(COMGR::hotswap::canonicalOpFor(
                llvm::AMDGPU::V_CVT_SCALEF32_PK8_FP8_F32_e64_gfx1250),
            COMGR::hotswap::CanonicalOp::V_CVT_SCALEF32_PK8_FP8_F32);
}

TEST(OpcodeMap, Gfx1250Maximum3Minimum3F32RealOpcodesMapToCanonicalOps) {
  // V_MAXIMUM3_F32 / V_MINIMUM3_F32: gfx11+/gfx12 ternary IEEE-754
  // NaN-propagating max/min.  HasMinimum3Maximum3F32 in AMDGPU.td:194.
  EXPECT_EQ(
      COMGR::hotswap::canonicalOpFor(llvm::AMDGPU::V_MAXIMUM3_F32_e64_gfx12),
      COMGR::hotswap::CanonicalOp::V_MAXIMUM3_F32);
  EXPECT_EQ(
      COMGR::hotswap::canonicalOpFor(llvm::AMDGPU::V_MINIMUM3_F32_e64_gfx12),
      COMGR::hotswap::CanonicalOp::V_MINIMUM3_F32);
}

TEST(OpcodeMap, Gfx1250ScalarF32RoundingOpcodesMapToCanonicalOps) {
  // S_CEIL_F32 / S_FLOOR_F32 / S_TRUNC_F32 / S_RNDNE_F32: gfx11+
  // scalar F32-to-F32 integral rounding (SOPInstructions.td:
  // SOP1_F32_Inst with fceil/ffloor/ftrunc/froundeven).
  EXPECT_EQ(COMGR::hotswap::canonicalOpFor(llvm::AMDGPU::S_CEIL_F32),
            COMGR::hotswap::CanonicalOp::S_CEIL_F32);
  EXPECT_EQ(COMGR::hotswap::canonicalOpFor(llvm::AMDGPU::S_FLOOR_F32),
            COMGR::hotswap::CanonicalOp::S_FLOOR_F32);
  EXPECT_EQ(COMGR::hotswap::canonicalOpFor(llvm::AMDGPU::S_TRUNC_F32),
            COMGR::hotswap::CanonicalOp::S_TRUNC_F32);
  EXPECT_EQ(COMGR::hotswap::canonicalOpFor(llvm::AMDGPU::S_RNDNE_F32),
            COMGR::hotswap::CanonicalOp::S_RNDNE_F32);
}

TEST(OpcodeMap, Gfx1250MaximumMinimumF32RealOpcodesMapToCanonicalOps) {
  // V_MAXIMUMMINIMUM_F32 / V_MINIMUMMAXIMUM_F32: gfx11+/gfx12 ternary
  // IEEE-754 NaN-propagating clamp pair at VOP3 opcodes 0x26d / 0x26c.
  EXPECT_EQ(COMGR::hotswap::canonicalOpFor(
                llvm::AMDGPU::V_MAXIMUMMINIMUM_F32_e64_gfx12),
            COMGR::hotswap::CanonicalOp::V_MAXIMUMMINIMUM_F32);
  EXPECT_EQ(COMGR::hotswap::canonicalOpFor(
                llvm::AMDGPU::V_MINIMUMMAXIMUM_F32_e64_gfx12),
            COMGR::hotswap::CanonicalOp::V_MINIMUMMAXIMUM_F32);
}

TEST(OpcodeMap, Gfx1250RelatedMinimumMaximumOpcodesMapToCanonicalOps) {
  EXPECT_EQ(COMGR::hotswap::canonicalOpFor(llvm::AMDGPU::V_MAXMIN_F32_e64),
            COMGR::hotswap::CanonicalOp::V_MAXMIN_NUM_F32);
  // LLVM names the t16 .NUM f16 real opcodes with the gfx11 suffix in this
  // build; hotswap-tblgen still evaluates the gfx1250 mappings and validates
  // the alias collapse from that real form to the shared canonical pseudo.
  EXPECT_EQ(
      COMGR::hotswap::canonicalOpFor(llvm::AMDGPU::V_MINMAX_F16_t16_e64_gfx11),
      COMGR::hotswap::CanonicalOp::V_MINMAX_NUM_F16);
  EXPECT_EQ(
      COMGR::hotswap::canonicalOpFor(llvm::AMDGPU::V_MAXMIN_F16_t16_e64_gfx11),
      COMGR::hotswap::CanonicalOp::V_MAXMIN_NUM_F16);
  EXPECT_EQ(
      COMGR::hotswap::canonicalOpFor(llvm::AMDGPU::V_MAXIMUM_F16_t16_e64_gfx12),
      COMGR::hotswap::CanonicalOp::V_MAXIMUM_F16);
  EXPECT_EQ(
      COMGR::hotswap::canonicalOpFor(llvm::AMDGPU::V_MINIMUM_F16_t16_e64_gfx12),
      COMGR::hotswap::CanonicalOp::V_MINIMUM_F16);
  EXPECT_EQ(COMGR::hotswap::canonicalOpFor(
                llvm::AMDGPU::V_MAXIMUM3_F16_t16_e64_gfx12),
            COMGR::hotswap::CanonicalOp::V_MAXIMUM3_F16);
  EXPECT_EQ(COMGR::hotswap::canonicalOpFor(
                llvm::AMDGPU::V_MINIMUM3_F16_t16_e64_gfx12),
            COMGR::hotswap::CanonicalOp::V_MINIMUM3_F16);
  EXPECT_EQ(COMGR::hotswap::canonicalOpFor(
                llvm::AMDGPU::V_MAXIMUMMINIMUM_F16_t16_e64_gfx12),
            COMGR::hotswap::CanonicalOp::V_MAXIMUMMINIMUM_F16);
  EXPECT_EQ(COMGR::hotswap::canonicalOpFor(
                llvm::AMDGPU::V_MINIMUMMAXIMUM_F16_t16_e64_gfx12),
            COMGR::hotswap::CanonicalOp::V_MINIMUMMAXIMUM_F16);
}
