//===- HotswapDefUseTest.cpp - Unit tests for InstDefUse extraction ------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Tests for the register def/use extraction in comgr-hotswap-def-use.cpp:
/// the MCRegister -> RegisterRef mapping (toRegisterRef) and per-instruction
/// InstDefUse def/use classification. Instructions are assembled and decoded
/// through a real gfx1250 LLVMState so the tests exercise the same MCInst /
/// MCInstrDesc / MCRegisterInfo objects the production path would see.
///
/// COMGR::ensureLLVMInitialized() is provided by HotswapMCTest.cpp, which is
/// linked into the same HotswapMCTests binary; it is not redefined here.
///
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-def-use.h"
#include "comgr-hotswap-internal.h"
#include "comgr.h"
#include "llvm/MC/MCInst.h"
#include "gtest/gtest.h"

#include <optional>
#include <vector>

using namespace COMGR;
using namespace COMGR::hotswap;
using namespace COMGR::hotswap::reglive;

namespace {

TargetIdentifier makeGfx1250Ident() {
  TargetIdentifier TI;
  TI.Arch = "amdgcn";
  TI.Vendor = "amd";
  TI.OS = "amdhsa";
  TI.Environ = "";
  TI.Processor = "gfx1250";
  return TI;
}

// Assemble a single asm line and return its first decoded MCInst.
std::optional<llvm::MCInst> assembleOne(const LLVMState &S,
                                        llvm::StringRef Asm) {
  llvm::SmallVector<uint8_t> Bytes = assembleSingleInst(Asm, S);
  if (Bytes.empty())
    return std::nullopt;
  std::vector<InternalDecodedInst> Decoded;
  if (!decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded))
    return std::nullopt;
  if (Decoded.empty() || !Decoded.front().DecodeSucceeded)
    return std::nullopt;
  return Decoded.front().Inst;
}

// -- toRegisterRef ----------------------------------------------------------

TEST(ToRegisterRef, SpecialRegistersAreUntracked) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  ASSERT_TRUE(S.MRI);

  // VCC and SCC are cached by initLLVM; neither is a tracked GPR.
  EXPECT_FALSE(toRegisterRef(S.VCCRegister, *S.MRI).has_value());
  EXPECT_FALSE(toRegisterRef(S.SCCRegister, *S.MRI).has_value());
  EXPECT_FALSE(toRegisterRef(llvm::MCRegister(), *S.MRI).has_value());
}

// -- InstDefUse: single-lane def / use --------------------------------------

TEST(InstDefUse, VectorMoveDefIsExecMasked) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  std::optional<llvm::MCInst> Inst = assembleOne(S, "v_mov_b32 v3, v5");
  ASSERT_TRUE(Inst.has_value());

  InstDefUse DU(*Inst, *S.MCII, *S.MRI);
  EXPECT_TRUE(DU.Defs.contains(RegisterRef{RegClass::VGPR, 3, 1}));
  EXPECT_EQ(DU.Defs.size(), 1u);
  EXPECT_TRUE(DU.Uses.contains(RegisterRef{RegClass::VGPR, 5, 1}));
  EXPECT_FALSE(DU.Uses.contains(RegisterRef{RegClass::VGPR, 3, 1}));
  EXPECT_TRUE(DU.HasExecMaskedVectorDef);
  EXPECT_FALSE(DU.HasPredicatedDef);
}

TEST(InstDefUse, ScalarMoveDefIsNotExecMasked) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  std::optional<llvm::MCInst> Inst = assembleOne(S, "s_mov_b32 s7, s9");
  ASSERT_TRUE(Inst.has_value());

  InstDefUse DU(*Inst, *S.MCII, *S.MRI);
  EXPECT_TRUE(DU.Defs.contains(RegisterRef{RegClass::SGPR, 7, 1}));
  EXPECT_TRUE(DU.Uses.contains(RegisterRef{RegClass::SGPR, 9, 1}));
  EXPECT_FALSE(DU.HasExecMaskedVectorDef);
}

// -- InstDefUse: multi-lane (register pair) ---------------------------------

TEST(InstDefUse, ScalarPairSpansTwoLanes) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  std::optional<llvm::MCInst> Inst = assembleOne(S, "s_mov_b64 s[4:5], s[6:7]");
  ASSERT_TRUE(Inst.has_value());

  InstDefUse DU(*Inst, *S.MCII, *S.MRI);
  // The 64-bit pair expands to both constituent lanes.
  EXPECT_TRUE(DU.Defs.contains(RegisterRef{RegClass::SGPR, 4, 2}));
  EXPECT_TRUE(DU.Defs.contains(RegisterRef{RegClass::SGPR, 4, 1}));
  EXPECT_TRUE(DU.Defs.contains(RegisterRef{RegClass::SGPR, 5, 1}));
  EXPECT_EQ(DU.Defs.size(), 2u);
  EXPECT_TRUE(DU.Uses.contains(RegisterRef{RegClass::SGPR, 6, 2}));
  EXPECT_EQ(DU.Uses.size(), 2u);
}

// -- InstDefUse: implicit defs stay untracked -------------------------------

TEST(InstDefUse, ImplicitSccDefIsNotTracked) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  // s_add_co_u32 writes its explicit SGPR result plus SCC (implicit). Only the
  // explicit SGPR result is a tracked def; SCC is dropped.
  std::optional<llvm::MCInst> Inst = assembleOne(S, "s_add_co_u32 s0, s1, s2");
  ASSERT_TRUE(Inst.has_value());

  InstDefUse DU(*Inst, *S.MCII, *S.MRI);
  EXPECT_TRUE(DU.Defs.contains(RegisterRef{RegClass::SGPR, 0, 1}));
  EXPECT_EQ(DU.Defs.size(), 1u);
  EXPECT_TRUE(DU.Uses.contains(RegisterRef{RegClass::SGPR, 1, 1}));
  EXPECT_TRUE(DU.Uses.contains(RegisterRef{RegClass::SGPR, 2, 1}));
  EXPECT_FALSE(DU.HasExecMaskedVectorDef);
}

// -- InstDefUse: read-modify-write (same reg def and use) -------------------

TEST(InstDefUse, SameRegisterInDefAndUse) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  // v0 is both written (dst) and read (src0): it must appear in both sets so a
  // later liveness pass keeps it live before the instruction.
  std::optional<llvm::MCInst> Inst = assembleOne(S, "v_add_f32 v0, v0, v1");
  ASSERT_TRUE(Inst.has_value());

  InstDefUse DU(*Inst, *S.MCII, *S.MRI);
  EXPECT_TRUE(DU.Defs.contains(RegisterRef{RegClass::VGPR, 0, 1}));
  EXPECT_TRUE(DU.Uses.contains(RegisterRef{RegClass::VGPR, 0, 1}));
  EXPECT_TRUE(DU.Uses.contains(RegisterRef{RegClass::VGPR, 1, 1}));
  EXPECT_TRUE(DU.HasExecMaskedVectorDef);
}

} // namespace
