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
/// decodeTextSection round-trip, the decodeTextSection instruction-decode
/// cache, applyMnemonicSwap, applyByteReplace, and checkVgprOverlap.
///
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-internal.h"
#include "comgr-test-elf-utils.h"
#include "comgr.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/TargetSelect.h"
#include "gtest/gtest.h"

#include <cstring>
#include <limits>
#include <mutex>
#include <vector>

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

static uint64_t alignTo8(uint64_t V) { return (V + 7) & ~uint64_t{7}; }

static std::vector<uint8_t> makeDisplacementTestElf(
    llvm::ArrayRef<uint8_t> Text, bool AddTextRelocation = false,
    bool AddDebugSection = false, bool AddBoundaryTextSymbol = false) {
  using namespace llvm::ELF;
  namespace hsa = llvm::amdhsa;

  static constexpr uint64_t ShOff = sizeof(Elf64_Ehdr);
  static constexpr uint64_t PhOff = 0x200;
  static constexpr uint64_t TextOff = 0x280;
  static constexpr uint64_t TextAddr = 0x1000;
  static constexpr uint64_t RodataAddr = 0x2000;
  static constexpr uint64_t KdBytes = sizeof(hsa::kernel_descriptor_t);
  const uint64_t SymCount = AddBoundaryTextSymbol ? 4 : 3;

  const char StrTab[] = "\0kernel\0kernel.kd\0";
  const char ShStrTabNoRel[] =
      "\0.text\0.rodata\0.strtab\0.symtab\0.shstrtab\0";
  const char ShStrTabRel[] =
      "\0.text\0.rodata\0.strtab\0.symtab\0.rela.text\0.shstrtab\0";
  const char ShStrTabDebug[] =
      "\0.text\0.rodata\0.strtab\0.symtab\0.debug_info\0.shstrtab\0";

  const uint64_t RodataOff = alignTo8(TextOff + Text.size());
  const uint64_t StrTabOff = alignTo8(RodataOff + KdBytes);
  const uint64_t SymTabOff = alignTo8(StrTabOff + sizeof(StrTab));
  const uint64_t RelOff =
      AddTextRelocation ? alignTo8(SymTabOff + SymCount * sizeof(Elf64_Sym))
                        : 0;
  const uint64_t DebugOff =
      AddDebugSection ? alignTo8(SymTabOff + SymCount * sizeof(Elf64_Sym)) : 0;
  const uint64_t ShStrTabOff =
      AddTextRelocation ? alignTo8(RelOff + sizeof(Elf64_Rela))
      : AddDebugSection ? alignTo8(DebugOff + 4)
                        : alignTo8(SymTabOff + SymCount * sizeof(Elf64_Sym));
  const uint64_t ShStrTabSize = AddTextRelocation ? sizeof(ShStrTabRel)
                                : AddDebugSection ? sizeof(ShStrTabDebug)
                                                  : sizeof(ShStrTabNoRel);
  const uint64_t BufSize = alignTo8(ShStrTabOff + ShStrTabSize + 64);

  std::vector<uint8_t> Buf(BufSize, 0);
  const char *ShStrTab = AddTextRelocation ? ShStrTabRel
                         : AddDebugSection ? ShStrTabDebug
                                           : ShStrTabNoRel;
  std::memcpy(Buf.data() + ShStrTabOff, ShStrTab, ShStrTabSize);
  std::memcpy(Buf.data() + StrTabOff, StrTab, sizeof(StrTab));
  std::memcpy(Buf.data() + TextOff, Text.data(), Text.size());

  Elf64_Ehdr Ehdr = comgr_test::makeElf64Ehdr(EM_AMDGPU);
  Ehdr.e_ident[EI_OSABI] = ELFOSABI_AMDGPU_HSA;
  Ehdr.e_type = ET_DYN;
  Ehdr.e_version = EV_CURRENT;
  Ehdr.e_phoff = PhOff;
  Ehdr.e_shoff = ShOff;
  Ehdr.e_ehsize = sizeof(Elf64_Ehdr);
  Ehdr.e_phentsize = sizeof(Elf64_Phdr);
  Ehdr.e_phnum = 2;
  Ehdr.e_shentsize = sizeof(Elf64_Shdr);
  Ehdr.e_shnum = AddTextRelocation || AddDebugSection ? 7 : 6;
  Ehdr.e_shstrndx = AddTextRelocation || AddDebugSection ? 6 : 5;
  std::memcpy(Buf.data(), &Ehdr, sizeof(Ehdr));

  Elf64_Phdr TextPh{};
  TextPh.p_type = PT_LOAD;
  TextPh.p_flags = PF_R | PF_X;
  TextPh.p_offset = TextOff;
  TextPh.p_vaddr = TextAddr;
  TextPh.p_paddr = TextAddr;
  TextPh.p_filesz = Text.size();
  TextPh.p_memsz = Text.size() + 64;
  TextPh.p_align = 8;
  std::memcpy(Buf.data() + PhOff, &TextPh, sizeof(TextPh));

  Elf64_Phdr RodataPh{};
  RodataPh.p_type = PT_LOAD;
  RodataPh.p_flags = PF_R;
  RodataPh.p_offset = RodataOff;
  RodataPh.p_vaddr = RodataAddr;
  RodataPh.p_paddr = RodataAddr;
  RodataPh.p_filesz = KdBytes;
  RodataPh.p_memsz = KdBytes;
  RodataPh.p_align = 8;
  std::memcpy(Buf.data() + PhOff + sizeof(Elf64_Phdr), &RodataPh,
              sizeof(RodataPh));

  Elf64_Shdr TextSh{};
  TextSh.sh_name = 1;
  TextSh.sh_type = SHT_PROGBITS;
  TextSh.sh_flags = SHF_ALLOC | SHF_EXECINSTR;
  TextSh.sh_offset = TextOff;
  TextSh.sh_addr = TextAddr;
  TextSh.sh_size = Text.size();
  TextSh.sh_addralign = 4;
  std::memcpy(Buf.data() + ShOff + 1 * sizeof(Elf64_Shdr), &TextSh,
              sizeof(TextSh));

  Elf64_Shdr RodataSh{};
  RodataSh.sh_name = 7;
  RodataSh.sh_type = SHT_PROGBITS;
  RodataSh.sh_flags = SHF_ALLOC;
  RodataSh.sh_offset = RodataOff;
  RodataSh.sh_addr = RodataAddr;
  RodataSh.sh_size = KdBytes;
  RodataSh.sh_addralign = 8;
  std::memcpy(Buf.data() + ShOff + 2 * sizeof(Elf64_Shdr), &RodataSh,
              sizeof(RodataSh));

  Elf64_Shdr StrtabSh{};
  StrtabSh.sh_name = 15;
  StrtabSh.sh_type = SHT_STRTAB;
  StrtabSh.sh_offset = StrTabOff;
  StrtabSh.sh_size = sizeof(StrTab);
  std::memcpy(Buf.data() + ShOff + 3 * sizeof(Elf64_Shdr), &StrtabSh,
              sizeof(StrtabSh));

  Elf64_Shdr SymtabSh{};
  SymtabSh.sh_name = 23;
  SymtabSh.sh_type = SHT_SYMTAB;
  SymtabSh.sh_offset = SymTabOff;
  SymtabSh.sh_size = SymCount * sizeof(Elf64_Sym);
  SymtabSh.sh_link = 3;
  SymtabSh.sh_entsize = sizeof(Elf64_Sym);
  std::memcpy(Buf.data() + ShOff + 4 * sizeof(Elf64_Shdr), &SymtabSh,
              sizeof(SymtabSh));

  unsigned ShStrIndex = AddTextRelocation || AddDebugSection ? 6 : 5;
  if (AddTextRelocation) {
    Elf64_Shdr RelaSh{};
    RelaSh.sh_name = 31;
    RelaSh.sh_type = SHT_RELA;
    RelaSh.sh_offset = RelOff;
    RelaSh.sh_size = sizeof(Elf64_Rela);
    RelaSh.sh_link = 4;
    RelaSh.sh_info = 1; // applies to .text
    RelaSh.sh_entsize = sizeof(Elf64_Rela);
    std::memcpy(Buf.data() + ShOff + 5 * sizeof(Elf64_Shdr), &RelaSh,
                sizeof(RelaSh));
  }
  if (AddDebugSection) {
    Elf64_Shdr DebugSh{};
    DebugSh.sh_name = 31;
    DebugSh.sh_type = SHT_PROGBITS;
    DebugSh.sh_offset = DebugOff;
    DebugSh.sh_size = 4;
    DebugSh.sh_addralign = 1;
    std::memcpy(Buf.data() + ShOff + 5 * sizeof(Elf64_Shdr), &DebugSh,
                sizeof(DebugSh));
  }

  Elf64_Shdr ShstrSh{};
  ShstrSh.sh_name = AddTextRelocation ? 42 : AddDebugSection ? 43 : 31;
  ShstrSh.sh_type = SHT_STRTAB;
  ShstrSh.sh_offset = ShStrTabOff;
  ShstrSh.sh_size = ShStrTabSize;
  std::memcpy(Buf.data() + ShOff + ShStrIndex * sizeof(Elf64_Shdr), &ShstrSh,
              sizeof(ShstrSh));

  int64_t EntryOffset = static_cast<int64_t>(TextAddr - RodataAddr);
  std::memcpy(
      Buf.data() + RodataOff +
          offsetof(hsa::kernel_descriptor_t, kernel_code_entry_byte_offset),
      &EntryOffset, sizeof(EntryOffset));

  Elf64_Sym KernelSym{};
  KernelSym.st_name = 1;
  KernelSym.setBindingAndType(STB_GLOBAL, STT_FUNC);
  KernelSym.st_shndx = 1;
  KernelSym.st_value = TextAddr;
  KernelSym.st_size = Text.size();
  std::memcpy(Buf.data() + SymTabOff + 1 * sizeof(Elf64_Sym), &KernelSym,
              sizeof(KernelSym));

  Elf64_Sym KdSym{};
  KdSym.st_name = 8;
  KdSym.setBindingAndType(STB_GLOBAL, STT_OBJECT);
  KdSym.st_shndx = 2;
  KdSym.st_value = RodataAddr;
  KdSym.st_size = KdBytes;
  std::memcpy(Buf.data() + SymTabOff + 2 * sizeof(Elf64_Sym), &KdSym,
              sizeof(KdSym));

  if (AddBoundaryTextSymbol) {
    Elf64_Sym BoundarySym{};
    BoundarySym.setBindingAndType(STB_GLOBAL, STT_FUNC);
    BoundarySym.st_shndx = 1;
    BoundarySym.st_value = TextAddr;
    BoundarySym.st_size = MinInstSize;
    std::memcpy(Buf.data() + SymTabOff + 3 * sizeof(Elf64_Sym), &BoundarySym,
                sizeof(BoundarySym));
  }

  return Buf;
}

// -- initLLVM ----------------------------------------------------------------

TEST(InitLLVM, ValidGfx1250) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  EXPECT_EQ(S.Cpu, "gfx1250");
  EXPECT_NE(S.Target, nullptr);
  ASSERT_NE(S.MCII, nullptr);
  EXPECT_LT(S.SBranchOpcode, S.MCII->getNumOpcodes());
  EXPECT_LT(S.SClauseOpcode, S.MCII->getNumOpcodes());
  EXPECT_LT(S.SDelayAluOpcode, S.MCII->getNumOpcodes());
  EXPECT_LT(S.SEndPgmOpcode, S.MCII->getNumOpcodes());
  EXPECT_LT(S.SEndPgmSavedOpcode, S.MCII->getNumOpcodes());
  EXPECT_LT(S.SAddNcU64Opcode, S.MCII->getNumOpcodes());
  EXPECT_LT(S.SAddPcI64Opcode, S.MCII->getNumOpcodes());
  EXPECT_LT(S.SCallI64Opcode, S.MCII->getNumOpcodes());
  EXPECT_LT(S.SSwapPcI64Opcode, S.MCII->getNumOpcodes());
  EXPECT_LT(S.SPrefetchInstPcRelOpcode, S.MCII->getNumOpcodes());
  EXPECT_LT(S.SPrefetchDataPcRelOpcode, S.MCII->getNumOpcodes());
  EXPECT_TRUE(S.SCCRegister.isValid());
  ASSERT_TRUE(S.VCCRegister.isValid());
  bool SawVccSubregister = false;
  for (llvm::MCPhysReg Sub : S.MRI->subregs(S.VCCRegister)) {
    SawVccSubregister = true;
    EXPECT_TRUE(S.MRI->regsOverlap(S.VCCRegister, llvm::MCRegister(Sub)));
  }
  EXPECT_TRUE(SawVccSubregister);
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

TEST(EncodeSBranch, PositiveBoundaryRoundTrip) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  constexpr uint64_t To =
      static_cast<uint64_t>(BranchOffsetMax + 1) * MinInstSize;
  llvm::SmallVector<uint8_t> Out = S.encodeSBranch(0, To);
  ASSERT_EQ(Out.size(), MinInstSize);
  uint32_t Encoded = readDword(Out.data());
  EXPECT_EQ(static_cast<int16_t>(Encoded & 0xFFFFu), BranchOffsetMax);
  EXPECT_TRUE(S.encodeSBranch(0, To + MinInstSize).empty());
}

TEST(EncodeSBranch, NegativeBoundaryRoundTrip) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  constexpr uint64_t From =
      static_cast<uint64_t>(-(BranchOffsetMin + 1)) * MinInstSize;
  llvm::SmallVector<uint8_t> Out = S.encodeSBranch(From, 0);
  ASSERT_EQ(Out.size(), MinInstSize);
  uint32_t Encoded = readDword(Out.data());
  EXPECT_EQ(static_cast<int16_t>(Encoded & 0xFFFFu), BranchOffsetMin);
  EXPECT_TRUE(S.encodeSBranch(From + MinInstSize, 0).empty());
}

TEST(EncodeSBranch, FailsOnInvalidState) {
  LLVMState S; // default-constructed, Valid = false
  EXPECT_TRUE(S.encodeSBranch(0, 8).empty());
}

// -- encodeSCallI64 ----------------------------------------------------------

static uint64_t decodeSCallTarget(uint64_t From,
                                  const InternalDecodedInst &Decoded) {
  const uint64_t Encoded =
      static_cast<uint64_t>(Decoded.Inst.getOperand(1).getImm()) & 0xFFFFu;
  const int64_t DwordDelta = Encoded < 0x8000u
                                 ? static_cast<int64_t>(Encoded)
                                 : static_cast<int64_t>(Encoded) - 0x10000;
  const uint64_t PcBase = From + MinInstSize;
  return DwordDelta >= 0
             ? PcBase + static_cast<uint64_t>(DwordDelta) * MinInstSize
             : PcBase - static_cast<uint64_t>(-DwordDelta) * MinInstSize;
}

TEST(EncodeSCallI64, WritesNextPcAndPreservesScc) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  constexpr uint64_t From = 0x20000;
  constexpr uint64_t To = 0x1F000;
  llvm::SmallVector<uint8_t> Out =
      encodeSCallI64(S, From, To, /*SgprBase=*/12);
  ASSERT_EQ(Out.size(), MinInstSize);

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Out.data(), Out.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 1u);
  EXPECT_EQ(Decoded[0].Mnemonic, "s_call_i64");
  EXPECT_EQ(Decoded[0].Inst.getOpcode(), S.SCallI64Opcode);
  ASSERT_EQ(Decoded[0].Inst.getNumOperands(), 2u);
  EXPECT_EQ(decodeSCallTarget(From, Decoded[0]), To);

  // RDNA4 S_CALL_B64 writes PC+4 to its explicit SDST operand. Its ISA
  // definition has no SCC operand, and LLVM's descriptor must agree.
  const llvm::MCInstrDesc &Desc = S.MCII->get(Decoded[0].Inst.getOpcode());
  EXPECT_TRUE(Desc.isCall());
  EXPECT_EQ(Desc.getNumDefs(), 1u);
  EXPECT_TRUE(Decoded[0].Inst.getOperand(0).isReg());
  for (llvm::MCPhysReg Reg : Desc.implicit_defs())
    EXPECT_NE(llvm::StringRef(S.MRI->getName(Reg)), "SCC");
}

TEST(EncodeSCallI64, EnforcesSignedDwordRangeAndAlignment) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  constexpr uint64_t ForwardFrom = 0x1000;
  constexpr uint64_t ForwardLimit =
      ForwardFrom + MinInstSize +
      static_cast<uint64_t>(BranchOffsetMax) * MinInstSize;
  EXPECT_EQ(encodeSCallI64(S, ForwardFrom, ForwardLimit, /*SgprBase=*/12)
                .size(),
            MinInstSize);
  EXPECT_TRUE(encodeSCallI64(S, ForwardFrom, ForwardLimit + MinInstSize,
                            /*SgprBase=*/12)
                  .empty());

  constexpr uint64_t BackwardFrom = 0x40000;
  constexpr uint64_t BackwardLimit =
      BackwardFrom + MinInstSize -
      static_cast<uint64_t>(-BranchOffsetMin) * MinInstSize;
  EXPECT_EQ(encodeSCallI64(S, BackwardFrom, BackwardLimit, /*SgprBase=*/12)
                .size(),
            MinInstSize);
  EXPECT_TRUE(encodeSCallI64(S, BackwardFrom, BackwardLimit - MinInstSize,
                            /*SgprBase=*/12)
                  .empty());
  EXPECT_TRUE(
      encodeSCallI64(S, ForwardFrom, ForwardFrom + 5, /*SgprBase=*/12).empty());
}

TEST(EncodeSCallI64, SupportsVccScratchPair) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  llvm::SmallVector<uint8_t> Out =
      encodeSCallI64(S, 0x1000, 0x1100, Gfx1250MaxSgprs);
  ASSERT_EQ(Out.size(), MinInstSize);
  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Out.data(), Out.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 1u);
  ASSERT_TRUE(Decoded[0].Inst.getOperand(0).isReg());
  EXPECT_EQ(llvm::StringRef(
                S.MRI->getName(Decoded[0].Inst.getOperand(0).getReg())),
            "VCC");
}

// -- encodeSetPCLongBranch ---------------------------------------------------

TEST(EncodeSetPCLongBranch, BackwardLandsOnTarget) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  const uint64_t From = 0x81000;
  const uint64_t To = 0x1004;
  llvm::SmallVector<uint8_t> Out =
      encodeSetPCLongBranch(S, From, To, /*SgprBase=*/12);
  ASSERT_FALSE(Out.empty());

  std::vector<InternalDecodedInst> Dec;
  ASSERT_TRUE(decodeTextSection(Out.data(), Out.size(), S, Dec));
  ASSERT_EQ(Dec.size(), 3u);
  EXPECT_EQ(Dec[0].Mnemonic, "s_get_pc_i64");
  EXPECT_EQ(Dec[1].Mnemonic, "s_add_nc_u64");
  EXPECT_EQ(Dec[2].Mnemonic, "s_set_pc_i64");
  for (const InternalDecodedInst &DI : Dec)
    EXPECT_NE(DI.Mnemonic, "s_add_pc_i64");

  const llvm::MCInstrDesc &AddDesc = S.MCII->get(Dec[1].Inst.getOpcode());
  EXPECT_FALSE(AddDesc.hasImplicitUseOfPhysReg(S.SCCRegister));
  EXPECT_FALSE(AddDesc.hasImplicitDefOfPhysReg(S.SCCRegister, S.MRI.get()));

  // s_get_pc_i64 captures the PC immediately after its own dword.
  uint64_t Delta = To - (From + MinInstSize);
  ASSERT_TRUE(Dec[1].Inst.getOperand(2).isImm());
  uint64_t EncodedDelta =
      static_cast<uint64_t>(Dec[1].Inst.getOperand(2).getImm());
  EXPECT_EQ(EncodedDelta, Delta);
  EXPECT_EQ(From + MinInstSize + EncodedDelta, To);
  EXPECT_EQ(static_cast<uint32_t>(Delta), 0xFFF80000u);
  EXPECT_EQ(static_cast<uint32_t>(Delta >> 32), 0xFFFFFFFFu);
}

TEST(EncodeSetPCLongBranch, RejectsMisalignedScratchPair) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  EXPECT_TRUE(encodeSetPCLongBranch(S, 0, 0x1000, /*SgprBase=*/3).empty());
}

static InternalDecodedInst decodeDelay(llvm::StringRef Assembly) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  llvm::SmallVector<uint8_t> Bytes = assembleSingleInst(Assembly, S);
  std::vector<InternalDecodedInst> Decoded;
  if (Bytes.empty() ||
      !decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded) ||
      Decoded.size() != 1)
    return {};
  return std::move(Decoded.front());
}

TEST(RelocationDelaySpan, ProtectsOnlyEncodedDependencySpan) {
  InternalDecodedInst First = decodeDelay("s_delay_alu instid0(VALU_DEP_1)");
  ASSERT_EQ(First.Mnemonic, "s_delay_alu");
  EXPECT_EQ(getDelayProtectedSpan(First), 1u);

  InternalDecodedInst Third =
      decodeDelay("s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | "
                  "instid1(SALU_CYCLE_1)");
  ASSERT_EQ(Third.Mnemonic, "s_delay_alu");
  EXPECT_EQ(getDelayProtectedSpan(Third), 3u);

  InternalDecodedInst Sixth =
      decodeDelay("s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | "
                  "instid1(SALU_CYCLE_1)");
  ASSERT_EQ(Sixth.Mnemonic, "s_delay_alu");
  EXPECT_EQ(getDelayProtectedSpan(Sixth), 6u);
}

TEST(RelocationDelaySpan, MalformedEncodingUsesConservativeMaximum) {
  InternalDecodedInst Delay;
  Delay.Mnemonic = "s_delay_alu";
  EXPECT_EQ(getDelayProtectedSpan(Delay), 6u);

  Delay.Inst.addOperand(llvm::MCOperand::createImm(12));
  EXPECT_EQ(getDelayProtectedSpan(Delay), 6u);
  Delay.Inst.getOperand(0).setImm(12u << 7);
  EXPECT_EQ(getDelayProtectedSpan(Delay), 6u);
  Delay.Inst.getOperand(0).setImm(6u << 4 | 1u << 7);
  EXPECT_EQ(getDelayProtectedSpan(Delay), 6u);
  Delay.Inst.getOperand(0).setImm(0x8001);
  EXPECT_EQ(getDelayProtectedSpan(Delay), 6u);
}

TEST(WmmaSourceWindow, RejectsEveryInteriorEntryIncludingLiteralSlots) {
  std::optional<llvm::DenseSet<uint64_t>> Targets;
  Targets.emplace();
  Targets->insert(0x100); // The replacement branch remains at the window head.
  Targets->insert(0x114); // The return address is outside the open interval.
  EXPECT_FALSE(
      hasDirectControlFlowTargetInWindowInterior(Targets, 0x100, 0x114));

  Targets->insert(0x108); // A literal slot in the original 16-byte WMMA.
  EXPECT_TRUE(
      hasDirectControlFlowTargetInWindowInterior(Targets, 0x100, 0x114));
}

TEST(WmmaSourceWindow, FailsClosedWithoutUsableTargetInformation) {
  std::optional<llvm::DenseSet<uint64_t>> Targets;
  EXPECT_TRUE(
      hasDirectControlFlowTargetInWindowInterior(Targets, 0x100, 0x114));
  Targets.emplace();
  EXPECT_TRUE(
      hasDirectControlFlowTargetInWindowInterior(Targets, 0x114, 0x100));
}

// -- encodeSetPCLongBranch geometry ------------------------------------------

static uint64_t
decodeSetPCLongBranchTarget(uint64_t From,
                            llvm::ArrayRef<InternalDecodedInst> Decoded) {
  const uint64_t PcBase = From + Decoded[0].Size;
  const uint64_t Delta =
      static_cast<uint64_t>(Decoded[1].Inst.getOperand(2).getImm());
  return PcBase + Delta;
}

TEST(EncodeSetPCLongBranch, BackwardLandsOnTargetWithoutDefiningScc) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  constexpr uint64_t From = 0x81000;
  constexpr uint64_t To = 0x1008;
  llvm::SmallVector<uint8_t> Out =
      encodeSetPCLongBranch(S, From, To, /*SgprBase=*/12);
  ASSERT_FALSE(Out.empty());

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Out.data(), Out.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 3u);
  EXPECT_EQ(Decoded[0].Mnemonic, "s_get_pc_i64");
  EXPECT_EQ(Decoded[1].Mnemonic, "s_add_nc_u64");
  EXPECT_EQ(Decoded[2].Mnemonic, "s_set_pc_i64");
  EXPECT_EQ(decodeSetPCLongBranchTarget(From, Decoded), To);

  const llvm::MCRegister Pair = Decoded[0].Inst.getOperand(0).getReg();
  EXPECT_EQ(Decoded[1].Inst.getOperand(0).getReg(), Pair);
  EXPECT_EQ(Decoded[1].Inst.getOperand(1).getReg(), Pair);
  EXPECT_EQ(Decoded[2].Inst.getOperand(0).getReg(), Pair);
  for (const InternalDecodedInst &DI : Decoded) {
    const llvm::MCInstrDesc &Desc = S.MCII->get(DI.Inst.getOpcode());
    for (llvm::MCPhysReg Reg : Desc.implicit_defs())
      EXPECT_NE(llvm::StringRef(S.MRI->getName(Reg)), "SCC");
  }
}

TEST(EncodeSetPCLongBranch, ForwardLandsOnTarget) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  constexpr uint64_t From = 0x1000;
  constexpr uint64_t To = 0x81000;
  llvm::SmallVector<uint8_t> Out =
      encodeSetPCLongBranch(S, From, To, /*SgprBase=*/12);
  ASSERT_FALSE(Out.empty());
  EXPECT_EQ(Out.size(), 16u);

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Out.data(), Out.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 3u);
  EXPECT_EQ(decodeSetPCLongBranchTarget(From, Decoded), To);
}

TEST(EncodeSetPCLongBranch, InlineDisplacementUsesTwelveBytes) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  constexpr uint64_t From = 0x1000;
  constexpr uint64_t To = From + 2 * MinInstSize;
  std::optional<llvm::SmallVector<uint8_t>> Out =
      encodeSetPCLongBranch(S, From, To, /*SgprBase=*/12);
  ASSERT_TRUE(Out);
  EXPECT_EQ(Out->size(), 12u);

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Out->data(), Out->size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 3u);
  ASSERT_TRUE(Decoded[1].Inst.getOperand(2).isImm());
  uint64_t Delta =
      static_cast<uint64_t>(Decoded[1].Inst.getOperand(2).getImm());
  EXPECT_EQ(From + MinInstSize + Delta, To);
}

TEST(FindNearestSetPcGateway, FitsActualSixteenByteEncoding) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  std::vector<NopSled> Gateways = {
      {/*Start=*/0x100, /*End=*/0x110, /*WritePos=*/0x100,
       /*OwnerStart=*/0, /*OwnerEnd=*/0x1000}};
  llvm::Expected<std::optional<EncodedSetPcGateway>> GatewayOrErr =
      findNearestSetPcGateway(Gateways, S, /*FromOffset=*/0,
                              /*TargetOffset=*/0x81000, /*SgprBase=*/12);
  ASSERT_TRUE((bool)GatewayOrErr) << llvm::toString(GatewayOrErr.takeError());
  std::optional<EncodedSetPcGateway> &Gateway = *GatewayOrErr;
  ASSERT_TRUE(Gateway);
  EXPECT_EQ(Gateway->Sled, &Gateways[0]);
  EXPECT_EQ(Gateway->Bytes.size(), 16u);
  EXPECT_EQ(Gateways[0].WritePos, 0x100u);
}

TEST(FindNearestSetPcGateway, SkipsNearerUndersizedCandidate) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  std::vector<NopSled> Gateways = {
      {/*Start=*/0x80100, /*End=*/0x80110, /*WritePos=*/0x80100,
       /*OwnerStart=*/0, /*OwnerEnd=*/0x100000},
      {/*Start=*/0x80200, /*End=*/0x80214, /*WritePos=*/0x80200,
       /*OwnerStart=*/0, /*OwnerEnd=*/0x100000}};
  llvm::Expected<std::optional<EncodedSetPcGateway>> GatewayOrErr =
      findNearestSetPcGateway(Gateways, S, /*FromOffset=*/0x80000,
                              /*TargetOffset=*/0x1004, /*SgprBase=*/12);
  ASSERT_TRUE((bool)GatewayOrErr) << llvm::toString(GatewayOrErr.takeError());
  std::optional<EncodedSetPcGateway> &Gateway = *GatewayOrErr;
  ASSERT_TRUE(Gateway);
  EXPECT_EQ(Gateway->Sled, &Gateways[1]);
  EXPECT_EQ(Gateway->Bytes.size(), SetPcReturnReserveBytes);
  EXPECT_EQ(Gateways[0].WritePos, 0x80100u);
  EXPECT_EQ(Gateways[1].WritePos, 0x80200u);
}

TEST(FindNearestSetPcGateway, DistinguishesNoFitFromEncodingFailure) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  std::vector<NopSled> Gateways = {
      {/*Start=*/0x100, /*End=*/0x108, /*WritePos=*/0x100,
       /*OwnerStart=*/0, /*OwnerEnd=*/0x1000}};
  llvm::Expected<std::optional<EncodedSetPcGateway>> NoFit =
      findNearestSetPcGateway(Gateways, S, /*FromOffset=*/0,
                              /*TargetOffset=*/0x81000, /*SgprBase=*/12);
  ASSERT_TRUE((bool)NoFit) << llvm::toString(NoFit.takeError());
  EXPECT_FALSE(*NoFit);

  llvm::Expected<std::optional<EncodedSetPcGateway>> EncodingFailure =
      findNearestSetPcGateway(Gateways, S, /*FromOffset=*/0,
                              /*TargetOffset=*/0x81000, /*SgprBase=*/3);
  ASSERT_FALSE((bool)EncodingFailure);
  std::string Error = llvm::toString(EncodingFailure.takeError());
  EXPECT_NE(Error.find("failed to encode set-PC gateway at candidate"),
            std::string::npos);
}

TEST(CountReachableSetPcGatewaySlots, DistinguishesZeroFromEncodingFailure) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  std::vector<NopSled> Gateways = {
      {/*Start=*/0x100, /*End=*/0x108, /*WritePos=*/0x100,
       /*OwnerStart=*/0, /*OwnerEnd=*/0x1000}};
  llvm::Expected<uint64_t> NoSlots = countReachableSetPcGatewaySlots(
      Gateways, S, /*FromOffset=*/0, /*TargetOffset=*/0x81000,
      /*SgprBase=*/12, /*MaxSlots=*/1);
  ASSERT_TRUE((bool)NoSlots) << llvm::toString(NoSlots.takeError());
  EXPECT_EQ(*NoSlots, 0u);

  llvm::Expected<uint64_t> EncodingFailure = countReachableSetPcGatewaySlots(
      Gateways, S, /*FromOffset=*/0, /*TargetOffset=*/0x81000,
      /*SgprBase=*/3, /*MaxSlots=*/1);
  ASSERT_FALSE((bool)EncodingFailure);
  std::string Error = llvm::toString(EncodingFailure.takeError());
  EXPECT_NE(Error.find("failed to encode set-PC gateway while counting"),
            std::string::npos);
}

TEST(EncodeSetPCLongBranch, RejectsPcBaseOverflow) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  EXPECT_TRUE(encodeSetPCLongBranch(
                  S, std::numeric_limits<uint64_t>::max() - MinInstSize + 1, 0,
                  /*SgprBase=*/12)
                  .empty());
}

TEST(EncodeSetPCLongBranch, WideForwardDeltaUsesFullReservation) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  constexpr uint64_t From = 0x1000;
  constexpr uint64_t To = 0x100002000ULL;
  llvm::SmallVector<uint8_t> Out =
      encodeSetPCLongBranch(S, From, To, /*SgprBase=*/12);
  ASSERT_FALSE(Out.empty());
  EXPECT_EQ(Out.size(), SetPcForwardSequenceBytes);

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Out.data(), Out.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 3u);
  EXPECT_EQ(decodeSetPCLongBranchTarget(From, Decoded), To);
}

TEST(EncodeSetPCLongBranch, CallTailSizeIsOwnerStableAtLiteralBoundaries) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  constexpr uint64_t Source = 0x20000;
  constexpr uint64_t CompactTarget = Source + 0x80000000ULL;
  llvm::SmallVector<uint8_t> CompactAtSource =
      encodeSetPCLongBranch(S, Source, CompactTarget, /*SgprBase=*/12);
  llvm::SmallVector<uint8_t> WideAtEarlierSled =
      encodeSetPCLongBranch(S, Source - MinInstSize, CompactTarget,
                            /*SgprBase=*/12);
  ASSERT_EQ(CompactAtSource.size(), 16u);
  ASSERT_EQ(WideAtEarlierSled.size(), SetPcForwardSequenceBytes);
  EXPECT_EQ(CompactAtSource.size() - MinInstSize, 12u);
  EXPECT_EQ(encodeSCallI64(S, Source, Source - MinInstSize, /*SgprBase=*/12)
                .size(),
            MinInstSize);

  constexpr uint64_t WideTarget = Source + 0x80000008ULL;
  llvm::SmallVector<uint8_t> WideAtSource =
      encodeSetPCLongBranch(S, Source, WideTarget, /*SgprBase=*/12);
  llvm::SmallVector<uint8_t> CompactAtLaterSled =
      encodeSetPCLongBranch(S, Source + 2 * MinInstSize, WideTarget,
                            /*SgprBase=*/12);
  ASSERT_EQ(WideAtSource.size(), SetPcForwardSequenceBytes);
  ASSERT_EQ(CompactAtLaterSled.size(), 16u);
  EXPECT_EQ(WideAtSource.size() - MinInstSize, 16u);
  EXPECT_EQ(encodeSCallI64(S, Source, Source + 2 * MinInstSize,
                          /*SgprBase=*/12)
                .size(),
            MinInstSize);

  std::vector<InternalDecodedInst> Full;
  ASSERT_TRUE(decodeTextSection(CompactAtSource.data(), CompactAtSource.size(),
                                S, Full));
  ASSERT_EQ(Full.size(), 3u);
  EXPECT_EQ(Full[0].Inst.getOpcode(), S.SGetPcI64Opcode);
  EXPECT_EQ(Full[0].Size, MinInstSize);
  EXPECT_EQ(decodeSetPCLongBranchTarget(Source, Full), CompactTarget);

  std::vector<InternalDecodedInst> Tail;
  ASSERT_TRUE(decodeTextSection(CompactAtSource.data() + MinInstSize,
                                CompactAtSource.size() - MinInstSize, S, Tail));
  ASSERT_EQ(Tail.size(), 2u);
  EXPECT_EQ(Tail[0].Inst.getOpcode(), S.SAddNcU64Opcode);
  EXPECT_EQ(Tail[1].Inst.getOpcode(), S.SSetPcI64Opcode);
  const unsigned Pair = Full[0].Inst.getOperand(0).getReg();
  EXPECT_EQ(Tail[0].Inst.getOperand(0).getReg(), Pair);
  EXPECT_EQ(Tail[0].Inst.getOperand(1).getReg(), Pair);
  EXPECT_EQ(Tail[1].Inst.getOperand(0).getReg(), Pair);
}

TEST(EncodeSetPCLongBranch, RejectsUnalignedPairAndPcOverflow) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  EXPECT_TRUE(encodeSetPCLongBranch(S, 0x1000, 0x2000,
                                    /*SgprBase=*/13)
                  .empty());
  EXPECT_TRUE(encodeSetPCLongBranch(S, std::numeric_limits<uint64_t>::max() - 1,
                                    0,
                                    /*SgprBase=*/12)
                  .empty());
}

// -- buildKernelEntryTrampolineFast ------------------------------------------
//
// The fast path emits its entry stub from a pre-encoded byte template, patching
// the two PC-relative delta immediates and the scratch SGPR register fields.
// These tests disassemble the emitted bytes and confirm (a) the stub names one
// consistent scratch pair across all six SGPR fields, and (b) the runtime PC
// arithmetic -- s_get_pc_i64 then the two-word add-with-carry -- lands exactly
// on the original entry. They pass ScratchSgpr=100 so the decoded bytes match
// the historical fixed-pair layout. Checking the decoded immediates rather than
// the raw template guards against a bad PC-base offset or a wrong delta word,
// which the disassembly-mnemonic lit test cannot catch.

// Disassemble a fast stub and reconstruct the entry vaddr it jumps to,
// modelling the on-hardware two's-complement add-with-carry across the
// scratch pair. Also asserts the structure (one consistent scratch pair,
// expected opcodes).
static uint64_t decodeFastStubTarget(const LLVMState &S, uint64_t StubVAddr,
                                     llvm::ArrayRef<uint8_t> Bytes) {
  std::vector<InternalDecodedInst> Dec;
  EXPECT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Dec));
  EXPECT_GE(Dec.size(), 6u);

  // Body layout: global_wb, v_nop, s_get_pc_i64, s_add_co_u32 (delta lo),
  // s_add_co_ci_u32 (delta hi), s_set_pc_i64.
  EXPECT_EQ(Dec[0].Inst.getOpcode(), S.GlobalWbOpcode);
  EXPECT_EQ(Dec[1].Inst.getOpcode(), S.VNopInst.getOpcode());
  EXPECT_EQ(Dec[2].Inst.getOpcode(), S.SGetPcI64Opcode);
  EXPECT_EQ(Dec[3].Inst.getOpcode(), S.SAddU32Opcode);
  EXPECT_EQ(Dec[4].Inst.getOpcode(), S.SAddcU32Opcode);
  EXPECT_EQ(Dec[5].Inst.getOpcode(), S.SSetPcI64Opcode);
  const llvm::MCInst &GetPc = Dec[2].Inst;
  const llvm::MCInst &AddLo = Dec[3].Inst;
  const llvm::MCInst &AddHi = Dec[4].Inst;
  const llvm::MCInst &SetPc = Dec[5].Inst;

  // s_get_pc, s_set_pc, and both add destinations must all name the same fixed
  // scratch pair the template hard-codes (s[100:101]).
  EXPECT_TRUE(GetPc.getOperand(0).isReg() && SetPc.getOperand(0).isReg() &&
              AddLo.getOperand(0).isReg() && AddHi.getOperand(0).isReg());
  const llvm::MCRegister Pair = GetPc.getOperand(0).getReg();
  EXPECT_EQ(SetPc.getOperand(0).getReg(), Pair);
  EXPECT_EQ(AddLo.getOperand(0).getReg(), AddLo.getOperand(1).getReg());
  EXPECT_EQ(AddHi.getOperand(0).getReg(), AddHi.getOperand(1).getReg());

  // The 32-bit literal is the trailing dword of each 8-byte add. Read it from
  // the disassembler-reported instruction span rather than the decoded operand:
  // the AMDGPU disassembler models s_add_co_ci_u32's literal as an expr, so
  // getImm() on it is unreliable, while s_add_co_u32's is a plain imm.
  EXPECT_EQ(Dec[3].Size, 8u);
  EXPECT_EQ(Dec[4].Size, 8u);
  const uint32_t Lo = readDword(Bytes.data() + Dec[3].Offset + Dec[3].Size - 4);
  const uint32_t Hi = readDword(Bytes.data() + Dec[4].Offset + Dec[4].Size - 4);

  // PC base is the address of the instruction after s_get_pc_i64.
  const uint64_t PcBase = StubVAddr + Dec[2].Offset + Dec[2].Size;

  // Model the hardware add-with-carry across the 64-bit pair rather than a
  // plain 64-bit add, so a delta that carries out of the low word is exercised.
  const uint32_t BaseLo = static_cast<uint32_t>(PcBase);
  const uint32_t BaseHi = static_cast<uint32_t>(PcBase >> 32);
  const uint64_t SumLo = static_cast<uint64_t>(BaseLo) + Lo;
  const uint32_t ResLo = static_cast<uint32_t>(SumLo);
  const uint32_t Carry = static_cast<uint32_t>(SumLo >> 32);
  const uint32_t ResHi = BaseHi + Hi + Carry;
  return (static_cast<uint64_t>(ResHi) << 32) | ResLo;
}

static llvm::SmallVector<uint8_t>
getAppendedEntryStub(const std::vector<Trampoline> &Growth,
                     const KernelEntryTrampolineFixup &Fixup) {
  llvm::SmallVector<uint8_t> Flat;
  for (const Trampoline &T : Growth)
    Flat.append(T.Bytes.begin(), T.Bytes.end());
  if (Fixup.StubTextOffset > Flat.size() ||
      Flat.size() - Fixup.StubTextOffset < KernelEntryStubStride)
    return {};
  return llvm::SmallVector<uint8_t>(Flat.begin() + Fixup.StubTextOffset,
                                    Flat.begin() + Fixup.StubTextOffset +
                                        KernelEntryStubStride);
}

static void expectEntryStubUsesVcc(llvm::ArrayRef<uint8_t> Bytes,
                                   const LLVMState &S) {
  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_GE(Decoded.size(), 6u);
  ASSERT_NE(S.MRI, nullptr);
  ASSERT_TRUE(Decoded[2].Inst.getOperand(0).isReg());
  ASSERT_TRUE(Decoded[3].Inst.getOperand(0).isReg());
  ASSERT_TRUE(Decoded[4].Inst.getOperand(0).isReg());
  ASSERT_TRUE(Decoded[5].Inst.getOperand(0).isReg());
  EXPECT_EQ(
      llvm::StringRef(S.MRI->getName(Decoded[2].Inst.getOperand(0).getReg())),
      "VCC");
  EXPECT_EQ(
      llvm::StringRef(S.MRI->getName(Decoded[3].Inst.getOperand(0).getReg())),
      "VCC_LO");
  EXPECT_EQ(
      llvm::StringRef(S.MRI->getName(Decoded[4].Inst.getOperand(0).getReg())),
      "VCC_HI");
  EXPECT_EQ(
      llvm::StringRef(S.MRI->getName(Decoded[5].Inst.getOperand(0).getReg())),
      "VCC");
}

static void expectEntryStubUsesTopNumberedPair(llvm::ArrayRef<uint8_t> Bytes,
                                               const LLVMState &S) {
  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_GE(Decoded.size(), 6u);
  ASSERT_NE(S.MRI, nullptr);
  EXPECT_EQ(
      llvm::StringRef(S.MRI->getName(Decoded[2].Inst.getOperand(0).getReg())),
      "SGPR104_SGPR105");
  EXPECT_EQ(
      llvm::StringRef(S.MRI->getName(Decoded[3].Inst.getOperand(0).getReg())),
      "SGPR104");
  EXPECT_EQ(
      llvm::StringRef(S.MRI->getName(Decoded[4].Inst.getOperand(0).getReg())),
      "SGPR105");
  EXPECT_EQ(
      llvm::StringRef(S.MRI->getName(Decoded[5].Inst.getOperand(0).getReg())),
      "SGPR104_SGPR105");
}

TEST(BuildKernelEntryTrampolineFast, ForwardDeltaLandsOnEntry) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  const uint64_t StubVAddr = 0x100000;
  const uint64_t EntryVAddr = 0x180000; // forward
  llvm::SmallVector<uint8_t> Bytes = buildKernelEntryTrampolineFast(
      StubVAddr, EntryVAddr, /*ScratchSgpr=*/100);
  ASSERT_EQ(Bytes.size(), KernelEntryStubStride);
  EXPECT_EQ(decodeFastStubTarget(S, StubVAddr, Bytes), EntryVAddr);
}

TEST(BuildKernelEntryTrampolineFast, BackwardDeltaLandsOnEntry) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  const uint64_t StubVAddr = 0x180000;
  const uint64_t EntryVAddr = 0x100000; // backward: negative delta
  llvm::SmallVector<uint8_t> Bytes = buildKernelEntryTrampolineFast(
      StubVAddr, EntryVAddr, /*ScratchSgpr=*/100);
  ASSERT_EQ(Bytes.size(), KernelEntryStubStride);
  EXPECT_EQ(decodeFastStubTarget(S, StubVAddr, Bytes), EntryVAddr);
}

TEST(BuildKernelEntryTrampolineFast, CarryProducingDeltaLandsOnEntry) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  // Pc base low word is near the top of 32 bits, and the entry is far enough
  // above that the low-word add overflows and must carry into the high word.
  const uint64_t StubVAddr = 0xFFFFF000;
  const uint64_t EntryVAddr = 0x1'0002'0000; // crosses the 4 GiB boundary
  llvm::SmallVector<uint8_t> Bytes = buildKernelEntryTrampolineFast(
      StubVAddr, EntryVAddr, /*ScratchSgpr=*/100);
  ASSERT_EQ(Bytes.size(), KernelEntryStubStride);
  EXPECT_EQ(decodeFastStubTarget(S, StubVAddr, Bytes), EntryVAddr);
}

// The fast path emits its stub body from a checked-in, generated byte template
// (comgr-hotswap-entry-trampoline-fast-stub.inc) instead of running the MC
// layer at rewrite time. This test is the guarantee those bytes never silently
// drift from what the assembler produces: assemble the six body instructions
// through the MC layer here and memcmp against the body
// buildKernelEntryTrampolineFast emits. The two s_add immediates are the
// PC-relative delta the runtime writes, so they are zeroed on both sides before
// comparing (imm=0 would otherwise assemble to the shorter inline-constant form
// -- we assemble with a literal to force the 32-bit-literal encoding the
// template uses, then zero the words).
TEST(BuildKernelEntryTrampolineFast, StubTemplateMatchesMCOutput) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  // The template is spelled with the fixed s[100:101] scratch pair; build with
  // that pair so the SGPR register-field bytes match the assembled
  // instructions.
  llvm::SmallVector<uint8_t> Stub = buildKernelEntryTrampolineFast(
      /*StubVAddr=*/0x1000, /*EntryVAddr=*/0x2000, /*ScratchSgpr=*/100);
  ASSERT_EQ(Stub.size(), KernelEntryStubStride);
  llvm::SmallVector<uint8_t> Body(Stub.begin(),
                                  Stub.begin() + FastEntryStubBodyBytes);

  // Assemble the six body instructions through the MC layer. The s_add
  // immediates use a literal to force the 32-bit-literal encoding.
  static const char *const BodyAsm[] = {
      "global_wb",
      "v_nop",
      "s_get_pc_i64 s[100:101]",
      "s_add_co_u32 s100, s100, 0xdeadbeef",
      "s_add_co_ci_u32 s101, s101, 0xdeadbeef",
      "s_set_pc_i64 s[100:101]",
  };
  llvm::SmallVector<uint8_t> Assembled;
  for (const char *Asm : BodyAsm) {
    llvm::SmallVector<uint8_t> Inst = assembleSingleInst(Asm, S);
    ASSERT_FALSE(Inst.empty()) << "failed to assemble: " << Asm;
    Assembled.append(Inst.begin(), Inst.end());
  }
  ASSERT_EQ(Assembled.size(), FastEntryStubBodyBytes);

  // Zero the PC-relative delta words on both sides (the runtime writes them;
  // the template carries zero; the assembled form carries the 0xdeadbeef
  // literal).
  for (uint64_t Off : {FastEntryDeltaLoOffset, FastEntryDeltaHiOffset})
    for (uint64_t I = 0; I < 4; ++I)
      Body[Off + I] = Assembled[Off + I] = 0;

  EXPECT_EQ(Body, Assembled);
}

// The stub's six SGPR register fields must encode whatever scratch pair the
// allocator picked -- not the s[100:101] the template is spelled with. Build
// with an even base other than 100 and confirm the decoded pair matches, and
// that the delta still lands on the entry (the field patch must not disturb the
// delta words).
TEST(BuildKernelEntryTrampolineFast, PatchesScratchSgprRegisterFields) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  const uint64_t StubVAddr = 0x100000;
  const uint64_t EntryVAddr = 0x140000;
  const unsigned ScratchSgpr = 8; // aligned pair s[8:9]
  llvm::SmallVector<uint8_t> Bytes =
      buildKernelEntryTrampolineFast(StubVAddr, EntryVAddr, ScratchSgpr);
  ASSERT_EQ(Bytes.size(), KernelEntryStubStride);

  std::vector<InternalDecodedInst> Dec;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Dec));
  ASSERT_GE(Dec.size(), 6u);
  const llvm::MCInst &GetPc = Dec[2].Inst;
  ASSERT_TRUE(GetPc.getOperand(0).isReg());
  // s_get_pc names the low SGPR of the pair; s[8:9] decodes as SGPR8.
  EXPECT_EQ(GetPc.getOperand(0).getReg(), Dec[5].Inst.getOperand(0).getReg());
  EXPECT_EQ(decodeFastStubTarget(S, StubVAddr, Bytes), EntryVAddr);
}

TEST(BuildKernelEntryTrampolineFast, EncodesLogicalTopPairAsVcc) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  constexpr uint64_t StubVAddr = 0x180000;
  constexpr uint64_t EntryVAddr = 0x100000;
  llvm::SmallVector<uint8_t> Bytes =
      buildKernelEntryTrampolineFast(StubVAddr, EntryVAddr,
                                     /*ScratchSgpr=*/Gfx1250MaxNumberedSgprs);
  ASSERT_EQ(Bytes.size(), KernelEntryStubStride);
  expectEntryStubUsesVcc(Bytes, S);
  EXPECT_EQ(decodeFastStubTarget(S, StubVAddr, Bytes), EntryVAddr);
}

// .sgpr_count is a total that includes VCC. Counts 105 through 108 have no
// unused numbered pair, but may use the entry-only VCC temporary and must
// reserve the complete 108-SGPR metadata total.
TEST(KernelEntryTrampolineFast, UsesVccForValidTopMetadataCounts) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  llvm::SmallVector<uint8_t> EndPgm = assembleSingleInst("s_endpgm", S);
  ASSERT_EQ(EndPgm.size(), MinInstSize);
  llvm::SmallVector<uint8_t> Text(EndPgm.begin(), EndPgm.end());

  for (unsigned SgprCount : {105u, 106u, 107u, 108u}) {
    SCOPED_TRACE(SgprCount);
    comgr_test::KernelDescriptorElfOptions Opts;
    Opts.MetadataSgprCount = SgprCount;
    comgr_test::KernelDescriptorElf Obj =
        comgr_test::makeKernelDescriptorElf(Text, Opts);
    llvm::Expected<ElfView> View =
        ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
    ASSERT_TRUE((bool)View) << llvm::toString(View.takeError());

    std::vector<Trampoline> Growth;
    std::vector<KernelEntryTrampolineFixup> Fixups;
    std::optional<uint32_t> Count = appendKernelEntryTrampolinesFast(
        *View, "gfx1250", /*MaxSgprs=*/Gfx1250MaxNumberedSgprs, Growth, Fixups);
    ASSERT_TRUE(Count.has_value());
    EXPECT_EQ(*Count, 1u);
    ASSERT_EQ(Fixups.size(), 1u);
    EXPECT_EQ(Fixups[0].RequiredSgprs, Gfx1250MaxTotalSgprs);
    EXPECT_FALSE(Fixups[0].SkipSgprReservation);
    llvm::SmallVector<uint8_t> Stub = getAppendedEntryStub(Growth, Fixups[0]);
    ASSERT_EQ(Stub.size(), KernelEntryStubStride);
    expectEntryStubUsesVcc(Stub, S);
  }
}

TEST(KernelEntryTrampolineFast,
     TopNumberedPairConservativelyReservesPossibleVcc) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Text = assembleSingleInst("s_endpgm", S);
  ASSERT_EQ(Text.size(), MinInstSize);

  comgr_test::KernelDescriptorElfOptions Opts;
  // A total of 104 may mean 102 numbered SGPRs plus VCC. The conservative
  // allocation still selects s[104:105], then reserves VCC as well.
  Opts.MetadataSgprCount = 104;
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(Text, Opts);
  llvm::Expected<ElfView> View =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)View) << llvm::toString(View.takeError());

  std::vector<Trampoline> Growth;
  std::vector<KernelEntryTrampolineFixup> Fixups;
  std::optional<uint32_t> Count = appendKernelEntryTrampolinesFast(
      *View, "gfx1250", /*MaxSgprs=*/Gfx1250MaxNumberedSgprs, Growth, Fixups);
  ASSERT_TRUE(Count.has_value());
  EXPECT_EQ(*Count, 1u);
  ASSERT_EQ(Fixups.size(), 1u);
  EXPECT_EQ(Fixups[0].RequiredSgprs, Gfx1250MaxTotalSgprs);
  llvm::SmallVector<uint8_t> Stub = getAppendedEntryStub(Growth, Fixups[0]);
  ASSERT_EQ(Stub.size(), KernelEntryStubStride);
  expectEntryStubUsesTopNumberedPair(Stub, S);
}

TEST(KernelEntryTrampolineFast, RejectsMalformedTotalSgprCount) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Text = assembleSingleInst("s_endpgm", S);
  ASSERT_EQ(Text.size(), MinInstSize);

  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.MetadataSgprCount = Gfx1250MaxTotalSgprs + 1;
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(Text, Opts);
  llvm::Expected<ElfView> View =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)View) << llvm::toString(View.takeError());

  std::vector<Trampoline> Growth;
  std::vector<KernelEntryTrampolineFixup> Fixups;
  EXPECT_FALSE(appendKernelEntryTrampolinesFast(
                   *View, "gfx1250",
                   /*MaxSgprs=*/Gfx1250MaxNumberedSgprs, Growth, Fixups)
                   .has_value());
  EXPECT_TRUE(Growth.empty());
  EXPECT_TRUE(Fixups.empty());
}

// A modest SGPR count leaves a numbered pair, so the fast path installs one
// trampoline and records the conservative metadata total in the fixup.
TEST(KernelEntryTrampolineFast, AllocatesPerKernelScratchAndBumpsReservation) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  llvm::SmallVector<uint8_t> EndPgm = assembleSingleInst("s_endpgm", S);
  ASSERT_EQ(EndPgm.size(), MinInstSize);
  llvm::SmallVector<uint8_t> Text(EndPgm.begin(), EndPgm.end());

  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.MetadataSgprCount = 8; // scratch pair lands at s[8:9]
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(Text, Opts);
  llvm::Expected<ElfView> View =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)View) << llvm::toString(View.takeError());

  std::vector<Trampoline> Growth;
  std::vector<KernelEntryTrampolineFixup> Fixups;
  std::optional<uint32_t> Count = appendKernelEntryTrampolinesFast(
      *View, "gfx1250", /*MaxSgprs=*/106, Growth, Fixups);
  ASSERT_TRUE(Count.has_value());
  EXPECT_EQ(*Count, 1u);
  ASSERT_EQ(Fixups.size(), 1u);
  // Keep room for VCC because .sgpr_count does not separate special SGPRs.
  EXPECT_EQ(Fixups[0].RequiredSgprs, 12u);
  EXPECT_FALSE(Fixups[0].SkipSgprReservation);
}

TEST(KernelEntryTrampoline, RejectsAmbiguousDescriptorNamesOnBothPaths) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  comgr_test::MultiKernelDescriptorElfOptions Opts;
  Opts.Kernels = {
      {"kernel", 0x1000, 0x2000, /*EntryOffset=*/-0x1000},
      {"kernel", 0x1100, 0x2100, /*EntryOffset=*/-0x1000},
  };
  std::vector<uint8_t> Bytes = comgr_test::makeMultiKernelDescriptorElf(Opts);
  llvm::Expected<ElfView> View =
      ElfView::create(Bytes.data(), Bytes.size());
  ASSERT_TRUE((bool)View) << llvm::toString(View.takeError());
  ASSERT_FALSE(View->kernelDescriptorCacheIsComplete());

  std::vector<Trampoline> Growth;
  std::vector<KernelEntryTrampolineFixup> Fixups;
  EXPECT_FALSE(appendKernelEntryTrampolines(
                   *View, S, /*MaxSgprs=*/106, Growth, Fixups)
                   .has_value());
  EXPECT_TRUE(Growth.empty());
  EXPECT_TRUE(Fixups.empty());

  EXPECT_FALSE(appendKernelEntryTrampolinesFast(
                   *View, "gfx1250", /*MaxSgprs=*/106, Growth, Fixups)
                   .has_value());
  EXPECT_TRUE(Growth.empty());
  EXPECT_TRUE(Fixups.empty());
}

TEST(IsSBranchReachable, CoversBoundariesAlignmentAndPcOverflow) {
  constexpr uint64_t PositiveLimit =
      static_cast<uint64_t>(BranchOffsetMax + 1) * MinInstSize;
  EXPECT_TRUE(isSBranchReachable(/*From=*/0, PositiveLimit));
  EXPECT_FALSE(isSBranchReachable(/*From=*/0, PositiveLimit + MinInstSize));
  EXPECT_FALSE(isSBranchReachable(/*From=*/0, /*To=*/7));

  constexpr uint64_t NegativeFrom =
      static_cast<uint64_t>(-(BranchOffsetMin + 1)) * MinInstSize;
  EXPECT_TRUE(isSBranchReachable(NegativeFrom, /*To=*/0));
  EXPECT_FALSE(isSBranchReachable(NegativeFrom + MinInstSize, /*To=*/0));
  EXPECT_FALSE(isSBranchReachable(std::numeric_limits<uint64_t>::max() - 1,
                                  /*To=*/0));
}

TEST(EvaluateDirectControlFlowTarget, EvaluatesImmediateBranch) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Bytes = assembleSingleInst("s_branch 1", S);
  ASSERT_FALSE(Bytes.empty());

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 1u);
  Decoded[0].Offset = 0x100;
  EXPECT_EQ(evaluateDirectControlFlowTarget(Decoded[0], S), 0x108u);
}

TEST(EvaluateDirectControlFlowTarget, EvaluatesGfx1250CallOperandFallback) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Bytes =
      assembleSingleInst("s_call_i64 s[0:1], 2", S);
  ASSERT_FALSE(Bytes.empty());

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 1u);
  Decoded[0].Offset = 0x200;
  EXPECT_EQ(evaluateDirectControlFlowTarget(Decoded[0], S),
            0x200u + Decoded[0].Size + 2 * MinInstSize);
}

TEST(CollectDirectBranchTargets, MarksRegisterTargetCallUnresolved) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Bytes =
      assembleSingleInst("s_swap_pc_i64 s[30:31], s[0:1]", S);
  ASSERT_FALSE(Bytes.empty());

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 1u);
  ASSERT_TRUE(S.MIA->isCall(Decoded[0].Inst));
  ASSERT_FALSE(S.MIA->isIndirectBranch(Decoded[0].Inst));
  for (const llvm::MCOperand &Op : Decoded[0].Inst)
    ASSERT_FALSE(Op.isImm());

  std::optional<DirectControlFlowInfo> Info =
      collectDirectBranchTargets(Decoded, S, /*TextAddr=*/0,
                                 /*TextSize=*/0x1000, /*DeclaredEntries=*/{});
  ASSERT_TRUE(Info);
  EXPECT_TRUE(Info->Targets.empty());
  EXPECT_TRUE(Info->HasUnresolvedTargets);
}

TEST(CollectDirectBranchTargets, IgnoresSetPcWithoutTreatingItAsCall) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Bytes =
      assembleSingleInst("s_set_pc_i64 s[8:9]", S);
  ASSERT_FALSE(Bytes.empty());

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 1u);
  EXPECT_TRUE(S.MIA->isBranch(Decoded[0].Inst));
  EXPECT_FALSE(S.MIA->isIndirectBranch(Decoded[0].Inst));
  EXPECT_FALSE(S.MIA->isCall(Decoded[0].Inst));
  EXPECT_TRUE(S.MIA->mayAffectControlFlow(Decoded[0].Inst, *S.MRI));

  std::optional<DirectControlFlowInfo> Info =
      collectDirectBranchTargets(Decoded, S, /*TextAddr=*/0,
                                 /*TextSize=*/0x1000, /*DeclaredEntries=*/{});
  ASSERT_TRUE(Info);
  EXPECT_TRUE(Info->Targets.empty());
  EXPECT_FALSE(Info->HasUnresolvedTargets);
}

TEST(CollectDirectBranchTargets, ResolvesProductionPcMaterializedCall) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Bytes =
      assembleInstructions("s_get_pc_i64 s[0:1]\n"
                           "s_add_nc_u64 s[0:1], s[0:1], 0xffffffffffed1230\n"
                           "v_mov_b32 v0, v1\n"
                           "s_swap_pc_i64 s[30:31], s[0:1]",
                           S);
  ASSERT_FALSE(Bytes.empty());

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 4u);
  for (InternalDecodedInst &DI : Decoded)
    DI.Offset += 0x12EDCC;

  // This is the exact address calculation from the production reproducer:
  // 0x1a000 + 0x12edcc + 4 - 0x12edd0 = 0x1a000.
  std::optional<DirectControlFlowInfo> Info =
      collectDirectBranchTargets(Decoded, S, /*TextAddr=*/0x1A000,
                                 /*TextSize=*/0x150000,
                                 /*DeclaredEntries=*/{});
  ASSERT_TRUE(Info);
  ASSERT_EQ(Info->Targets.size(), 1u);
  EXPECT_TRUE(Info->Targets.contains(0));
  EXPECT_FALSE(Info->HasUnresolvedTargets);
}

TEST(CollectDirectBranchTargets, RejectsClobberedPcMaterializedCall) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Bytes =
      assembleInstructions("s_get_pc_i64 s[0:1]\n"
                           "s_add_nc_u64 s[0:1], s[0:1], -4\n"
                           "s_mov_b32 s0, 0\n"
                           "s_swap_pc_i64 s[30:31], s[0:1]",
                           S);
  ASSERT_FALSE(Bytes.empty());

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 4u);

  std::optional<DirectControlFlowInfo> Info =
      collectDirectBranchTargets(Decoded, S, /*TextAddr=*/0,
                                 /*TextSize=*/0x1000, /*DeclaredEntries=*/{});
  ASSERT_TRUE(Info);
  EXPECT_TRUE(Info->Targets.empty());
  EXPECT_TRUE(Info->HasUnresolvedTargets);
}

TEST(CollectDirectBranchTargets, RejectsAlternateEntryIntoMaterialization) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Bytes =
      assembleInstructions("s_branch 1\n"
                           "s_get_pc_i64 s[0:1]\n"
                           "s_add_nc_u64 s[0:1], s[0:1], 4\n"
                           "s_swap_pc_i64 s[30:31], s[0:1]",
                           S);
  ASSERT_FALSE(Bytes.empty());

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 4u);

  // The branch enters at the add without executing s_get_pc_i64, so the
  // apparent linear definition chain does not prove the register value.
  std::optional<DirectControlFlowInfo> Info =
      collectDirectBranchTargets(Decoded, S, /*TextAddr=*/0,
                                 /*TextSize=*/0x1000, /*DeclaredEntries=*/{});
  ASSERT_TRUE(Info);
  EXPECT_TRUE(Info->Targets.contains(8));
  EXPECT_TRUE(Info->HasUnresolvedTargets);
}

TEST(CollectDirectBranchTargets, RejectsDeclaredEntryIntoMaterialization) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Bytes =
      assembleInstructions("s_get_pc_i64 s[0:1]\n"
                           "s_add_nc_u64 s[0:1], s[0:1], 4\n"
                           "s_swap_pc_i64 s[30:31], s[0:1]",
                           S);
  ASSERT_FALSE(Bytes.empty());

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 3u);
  llvm::SmallVector<uint64_t, 1> DeclaredEntries{Decoded[1].Offset};

  // A function or kernel entry at the add can bypass s_get_pc_i64, even when
  // no direct branch in .text exposes that alternate path.
  std::optional<DirectControlFlowInfo> Info = collectDirectBranchTargets(
      Decoded, S, /*TextAddr=*/0, /*TextSize=*/0x1000, DeclaredEntries);
  ASSERT_TRUE(Info);
  EXPECT_TRUE(Info->Targets.empty());
  EXPECT_TRUE(Info->HasUnresolvedTargets);
}

TEST(CollectDirectBranchTargets, RejectsUndecodedMaterializationSlot) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Bytes =
      assembleInstructions("s_get_pc_i64 s[0:1]\n"
                           "s_add_nc_u64 s[0:1], s[0:1], 8\n"
                           "v_mov_b32 v0, v1\n"
                           "s_swap_pc_i64 s[30:31], s[0:1]",
                           S);
  ASSERT_FALSE(Bytes.empty());

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 4u);
  Decoded[2].DecodeSucceeded = false;
  Decoded[2].Inst = llvm::MCInst();
  Decoded[2].Mnemonic = "<unknown>";

  std::optional<DirectControlFlowInfo> Info =
      collectDirectBranchTargets(Decoded, S, /*TextAddr=*/0,
                                 /*TextSize=*/0x1000,
                                 /*DeclaredEntries=*/{});
  ASSERT_TRUE(Info);
  EXPECT_TRUE(Info->Targets.empty());
  EXPECT_TRUE(Info->HasUnresolvedTargets);
}

TEST(CollectDirectBranchTargets, RejectsUnboundedIndirectEntry) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Bytes =
      assembleInstructions("s_set_pc_i64 s[4:5]\n"
                           "s_get_pc_i64 s[0:1]\n"
                           "s_add_nc_u64 s[0:1], s[0:1], 4\n"
                           "s_swap_pc_i64 s[30:31], s[0:1]",
                           S);
  ASSERT_FALSE(Bytes.empty());

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 4u);
  ASSERT_EQ(Decoded[0].Inst.getOpcode(), S.SSetPcI64Opcode);

  std::optional<DirectControlFlowInfo> Info =
      collectDirectBranchTargets(Decoded, S, /*TextAddr=*/0,
                                 /*TextSize=*/0x1000,
                                 /*DeclaredEntries=*/{});
  ASSERT_TRUE(Info);
  EXPECT_TRUE(Info->Targets.empty());
  EXPECT_TRUE(Info->HasUnresolvedTargets);
}

TEST(CollectDirectBranchTargets, BoundsCanonicalSetPcReturn) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Bytes =
      assembleInstructions("s_nop 0\n"
                           "s_set_pc_i64 s[30:31]\n"
                           "s_branch -2\n"
                           "s_get_pc_i64 s[0:1]\n"
                           "s_add_nc_u64 s[0:1], s[0:1], -16\n"
                           "s_swap_pc_i64 s[30:31], s[0:1]",
                           S);
  ASSERT_FALSE(Bytes.empty());

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 6u);
  ASSERT_EQ(Decoded[1].Inst.getOpcode(), S.SSetPcI64Opcode);
  ASSERT_EQ(Decoded[3].Inst.getOpcode(), S.SGetPcI64Opcode);
  llvm::SmallVector<uint64_t, 1> DeclaredEntries{0};
  llvm::SmallVector<ElfView::FunctionTextRange, 1> FunctionRanges{
      {0, Decoded[3].Offset}};

  // The helper preserves the link pair from its entry through s_set_pc_i64.
  // The block laid out after the return can branch back into the epilogue,
  // matching the production CFG, but it preserves the pair as well. The
  // materialized call is therefore the return's sole possible source.
  std::optional<DirectControlFlowInfo> Info = collectDirectBranchTargets(
      Decoded, S, /*TextAddr=*/0, /*TextSize=*/0x1000, DeclaredEntries,
      FunctionRanges);
  ASSERT_TRUE(Info);
  ASSERT_EQ(Info->Targets.size(), 2u);
  EXPECT_TRUE(Info->Targets.contains(0));
  EXPECT_TRUE(Info->Targets.contains(Decoded[1].Offset));
  EXPECT_FALSE(Info->HasUnresolvedTargets);
}

TEST(CollectDirectBranchTargets, RejectsClobberedSetPcReturn) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Bytes =
      assembleInstructions("s_mov_b32 s31, 0\n"
                           "s_set_pc_i64 s[30:31]\n"
                           "s_get_pc_i64 s[0:1]\n"
                           "s_add_nc_u64 s[0:1], s[0:1], -12\n"
                           "s_swap_pc_i64 s[30:31], s[0:1]",
                           S);
  ASSERT_FALSE(Bytes.empty());

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 5u);
  llvm::SmallVector<uint64_t, 1> DeclaredEntries{0};
  llvm::SmallVector<ElfView::FunctionTextRange, 1> FunctionRanges{
      {0, Decoded[2].Offset}};

  // A partial link-pair definition makes the return target arbitrary, so the
  // PC-materialized call must remain unresolved.
  std::optional<DirectControlFlowInfo> Info = collectDirectBranchTargets(
      Decoded, S, /*TextAddr=*/0, /*TextSize=*/0x1000, DeclaredEntries,
      FunctionRanges);
  ASSERT_TRUE(Info);
  EXPECT_TRUE(Info->Targets.empty());
  EXPECT_TRUE(Info->HasUnresolvedTargets);
}

TEST(CollectDirectBranchTargets, RejectsNestedCallSetPcReturn) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Bytes =
      assembleInstructions("s_call_i64 s[4:5], 1\n"
                           "s_set_pc_i64 s[30:31]\n"
                           "s_mov_b32 s30, 0\n"
                           "s_set_pc_i64 s[4:5]\n"
                           "s_get_pc_i64 s[0:1]\n"
                           "s_add_nc_u64 s[0:1], s[0:1], -20\n"
                           "s_swap_pc_i64 s[30:31], s[0:1]",
                           S);
  ASSERT_FALSE(Bytes.empty());

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 7u);
  llvm::SmallVector<uint64_t, 3> DeclaredEntries{0, Decoded[2].Offset,
                                                 Decoded[4].Offset};
  llvm::SmallVector<ElfView::FunctionTextRange, 2> FunctionRanges{
      {0, Decoded[2].Offset}, {Decoded[2].Offset, Decoded[4].Offset}};

  // The nested call uses a different link pair, so its instruction does not
  // directly define s[30:31]. Its callee can still clobber that outer return
  // pair, making a function-local definition scan insufficient.
  std::optional<DirectControlFlowInfo> Info = collectDirectBranchTargets(
      Decoded, S, /*TextAddr=*/0, /*TextSize=*/0x1000, DeclaredEntries,
      FunctionRanges);
  ASSERT_TRUE(Info);
  EXPECT_TRUE(Info->Targets.contains(Decoded[2].Offset));
  EXPECT_TRUE(Info->HasUnresolvedTargets);
}

TEST(CollectDirectBranchTargets, RejectsIndirectFallthroughChainEntry) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Bytes =
      assembleInstructions("s_branch -1\n"
                           "s_nop 0\n"
                           "s_nop 0\n"
                           "s_set_pc_i64 s[30:31]\n"
                           "s_set_pc_i64 s[2:3]\n"
                           "s_get_pc_i64 s[0:1]\n"
                           "s_add_nc_u64 s[0:1], s[0:1], -12\n"
                           "s_swap_pc_i64 s[30:31], s[0:1]",
                           S);
  ASSERT_FALSE(Bytes.empty());

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 8u);
  llvm::SmallVector<uint64_t, 1> DeclaredEntries{Decoded[3].Offset};
  llvm::SmallVector<ElfView::FunctionTextRange, 1> FunctionRanges{
      {Decoded[3].Offset, Decoded[4].Offset}};

  // The unknown s_set_pc_i64 target may enter the unreachable padding before
  // the helper. Global indirect-entry detection must keep the materialized
  // call unresolved even though direct and fallthrough checks accept it.
  std::optional<DirectControlFlowInfo> Info = collectDirectBranchTargets(
      Decoded, S, /*TextAddr=*/0, /*TextSize=*/0x1000, DeclaredEntries,
      FunctionRanges);
  ASSERT_TRUE(Info);
  EXPECT_TRUE(Info->Targets.contains(Decoded[0].Offset));
  EXPECT_TRUE(Info->HasUnresolvedTargets);
}

TEST(CollectDirectBranchTargets, RejectsAlternateEntryIntoReturnFunction) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Bytes =
      assembleInstructions("s_nop 0\n"
                           "s_set_pc_i64 s[30:31]\n"
                           "s_branch -2\n"
                           "s_branch -2\n"
                           "s_get_pc_i64 s[0:1]\n"
                           "s_add_nc_u64 s[0:1], s[0:1], -20\n"
                           "s_swap_pc_i64 s[30:31], s[0:1]",
                           S);
  ASSERT_FALSE(Bytes.empty());

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 7u);
  llvm::SmallVector<uint64_t, 1> DeclaredEntries{0};
  llvm::SmallVector<ElfView::FunctionTextRange, 1> FunctionRanges{
      {0, Decoded[3].Offset}};

  // The branch at the function end enters a block laid out after the return,
  // which can branch back to the epilogue without a call-defined link pair.
  std::optional<DirectControlFlowInfo> Info = collectDirectBranchTargets(
      Decoded, S, /*TextAddr=*/0, /*TextSize=*/0x1000, DeclaredEntries,
      FunctionRanges);
  ASSERT_TRUE(Info);
  EXPECT_TRUE(Info->Targets.contains(Decoded[2].Offset));
  EXPECT_TRUE(Info->HasUnresolvedTargets);
}

TEST(CollectDirectBranchTargets,
     RejectsInteriorPcMaterializedCallIntoReturnFunction) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Bytes =
      assembleInstructions("s_nop 0\n"
                           "s_set_pc_i64 s[30:31]\n"
                           "s_get_pc_i64 s[4:5]\n"
                           "s_add_nc_u64 s[4:5], s[4:5], -12\n"
                           "s_swap_pc_i64 s[30:31], s[4:5]\n"
                           "s_get_pc_i64 s[6:7]\n"
                           "s_add_nc_u64 s[6:7], s[6:7], -20\n"
                           "s_swap_pc_i64 s[2:3], s[6:7]",
                           S);
  ASSERT_FALSE(Bytes.empty());

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 8u);
  llvm::SmallVector<uint64_t, 1> DeclaredEntries{0};
  llvm::SmallVector<ElfView::FunctionTextRange, 1> FunctionRanges{
      {0, Decoded[2].Offset}};

  // The first call enters the helper normally, but the second enters at its
  // s_set_pc_i64 with a different link pair. Every known call into the range
  // participates in the return proof, including register-materialized calls.
  std::optional<DirectControlFlowInfo> Info = collectDirectBranchTargets(
      Decoded, S, /*TextAddr=*/0, /*TextSize=*/0x1000, DeclaredEntries,
      FunctionRanges);
  ASSERT_TRUE(Info);
  EXPECT_TRUE(Info->Targets.empty());
  EXPECT_TRUE(Info->HasUnresolvedTargets);
}

TEST(CollectDirectBranchTargets, RejectsExternalAliasAtLocalFunctionEntry) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Bytes =
      assembleInstructions("s_nop 0\n"
                           "s_set_pc_i64 s[30:31]\n"
                           "s_get_pc_i64 s[0:1]\n"
                           "s_add_nc_u64 s[0:1], s[0:1], -12\n"
                           "s_swap_pc_i64 s[30:31], s[0:1]",
                           S);
  ASSERT_FALSE(Bytes.empty());

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 5u);
  llvm::SmallVector<uint64_t, 1> DeclaredEntries{0};
  llvm::SmallVector<uint64_t, 1> ExternalEntries{0};
  llvm::SmallVector<ElfView::FunctionTextRange, 1> FunctionRanges{
      {0, Decoded[2].Offset}};

  // A global function or kernel alias at the local helper's start can enter
  // without a call-defined link pair, even though it is not an interior entry.
  std::optional<DirectControlFlowInfo> Info = collectDirectBranchTargets(
      Decoded, S, /*TextAddr=*/0, /*TextSize=*/0x1000, DeclaredEntries,
      FunctionRanges, ExternalEntries);
  ASSERT_TRUE(Info);
  EXPECT_TRUE(Info->Targets.empty());
  EXPECT_TRUE(Info->HasUnresolvedTargets);
}

TEST(CollectDirectBranchTargets, RejectsFallthroughIntoReturnFunction) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Bytes =
      assembleInstructions("s_nop 0\n"
                           "s_nop 0\n"
                           "s_set_pc_i64 s[30:31]\n"
                           "s_get_pc_i64 s[0:1]\n"
                           "s_add_nc_u64 s[0:1], s[0:1], -12\n"
                           "s_swap_pc_i64 s[30:31], s[0:1]",
                           S);
  ASSERT_FALSE(Bytes.empty());

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 6u);
  llvm::SmallVector<uint64_t, 2> DeclaredEntries{0, Decoded[1].Offset};
  llvm::SmallVector<ElfView::FunctionTextRange, 1> FunctionRanges{
      {Decoded[1].Offset, Decoded[3].Offset}};

  // The declared entry at zero reaches the local helper by fallthrough and
  // does not define s[30:31], so the helper's return cannot be bounded.
  std::optional<DirectControlFlowInfo> Info = collectDirectBranchTargets(
      Decoded, S, /*TextAddr=*/0, /*TextSize=*/0x1000, DeclaredEntries,
      FunctionRanges);
  ASSERT_TRUE(Info);
  EXPECT_TRUE(Info->Targets.empty());
  EXPECT_TRUE(Info->HasUnresolvedTargets);
}

TEST(CollectDirectBranchTargets, AllowsUnreachablePaddingBeforeReturnFunction) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Bytes =
      assembleInstructions("s_branch -1\n"
                           "s_nop 0\n"
                           "s_nop 0\n"
                           "s_nop 0\n"
                           "s_set_pc_i64 s[30:31]\n"
                           "s_get_pc_i64 s[0:1]\n"
                           "s_add_nc_u64 s[0:1], s[0:1], -12\n"
                           "s_swap_pc_i64 s[30:31], s[0:1]",
                           S);
  ASSERT_FALSE(Bytes.empty());

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 8u);
  llvm::SmallVector<uint64_t, 2> DeclaredEntries{0, Decoded[3].Offset};
  llvm::SmallVector<ElfView::FunctionTextRange, 1> FunctionRanges{
      {Decoded[3].Offset, Decoded[5].Offset}};

  // The nops before the helper are unreachable because their backward
  // fallthrough chain terminates at an unconditional branch. This mirrors the
  // padding before the production HSACO's second helper.
  std::optional<DirectControlFlowInfo> Info = collectDirectBranchTargets(
      Decoded, S, /*TextAddr=*/0, /*TextSize=*/0x1000, DeclaredEntries,
      FunctionRanges);
  ASSERT_TRUE(Info);
  EXPECT_TRUE(Info->Targets.contains(Decoded[3].Offset));
  EXPECT_FALSE(Info->HasUnresolvedTargets);
}

TEST(CollectDirectBranchTargets, HandlesImmediateAbsoluteTargetCall) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Bytes =
      assembleSingleInst("s_swap_pc_i64 s[30:31], 0x210", S);
  ASSERT_FALSE(Bytes.empty());

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 1u);
  ASSERT_TRUE(S.MIA->isCall(Decoded[0].Inst));
  ASSERT_TRUE(Decoded[0].Inst.getOperand(1).isImm());

  std::optional<DirectControlFlowInfo> Info =
      collectDirectBranchTargets(Decoded, S, /*TextAddr=*/0x200,
                                 /*TextSize=*/0x40, /*DeclaredEntries=*/{});
  ASSERT_TRUE(Info);
  ASSERT_EQ(Info->Targets.size(), 1u);
  EXPECT_TRUE(Info->Targets.contains(0x10));
  EXPECT_FALSE(Info->HasUnresolvedTargets);

  std::optional<DirectControlFlowInfo> OutsideInfo =
      collectDirectBranchTargets(Decoded, S, /*TextAddr=*/0x220,
                                 /*TextSize=*/0x40, /*DeclaredEntries=*/{});
  ASSERT_TRUE(OutsideInfo);
  EXPECT_TRUE(OutsideInfo->Targets.empty());
  EXPECT_FALSE(OutsideInfo->HasUnresolvedTargets);

  std::optional<DirectControlFlowInfo> OverflowInfo =
      collectDirectBranchTargets(
          Decoded, S,
          /*TextAddr=*/std::numeric_limits<uint64_t>::max() - 0x10,
          /*TextSize=*/0x20, /*DeclaredEntries=*/{});
  EXPECT_FALSE(OverflowInfo);
}

TEST(CollectDirectBranchTargets, CollectsPcRelativeCall) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Bytes =
      assembleSingleInst("s_call_i64 s[30:31], 2", S);
  ASSERT_FALSE(Bytes.empty());

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 1u);
  Decoded[0].Offset = 0x200;

  std::optional<DirectControlFlowInfo> Info =
      collectDirectBranchTargets(Decoded, S, /*TextAddr=*/0,
                                 /*TextSize=*/0x1000, /*DeclaredEntries=*/{});
  ASSERT_TRUE(Info);
  ASSERT_EQ(Info->Targets.size(), 1u);
  EXPECT_TRUE(
      Info->Targets.contains(0x200u + Decoded[0].Size + 2 * MinInstSize));
  EXPECT_FALSE(Info->HasUnresolvedTargets);
}

TEST(SafeSgprScratchBlock, RejectsRegisterBeyondAddressableLimit) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Text = assembleSingleInst("s_mov_b32 s4, s0", S);
  ASSERT_FALSE(Text.empty());

  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(Text);
  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());
  ElfView &View = *ViewOrErr;

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(View.textData(), View.textSize(), S, Decoded));
  RewriteConfig Config;
  Config.MaxSgprs = 4;
  std::vector<Trampoline> Trampolines;
  std::vector<NopSled> Sleds;
  LivenessInfo Liveness;
  llvm::StringMap<KernelPatchStats> KernelStats;
  std::vector<ScratchPatchInfo> ScratchPatches;
  DirectControlFlowInfo ControlFlow;
  HotswapProfile Prof(/*Enabled=*/false);
  PatchContext Ctx{Config,
                   Decoded,
                   View.textData(),
                   View.textSize(),
                   /*PoolBaseOffset=*/0,
                   S,
                   Trampolines,
                   Sleds,
                   View,
                   Liveness,
                   KernelStats,
                   ScratchPatches,
                   ControlFlow,
                   Prof};

  EXPECT_FALSE(findSafeSgprScratchBlock(Ctx, /*TextOffset=*/0, /*Count=*/1,
                                        /*Alignment=*/1, "unit test"));
}

TEST(SafeSgprScratchBlock, RejectsAlignmentOverflow) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Text = assembleSingleInst("s_mov_b32 s4, s0", S);
  ASSERT_FALSE(Text.empty());

  comgr_test::KernelDescriptorElfOptions Options;
  Options.MetadataSgprCount = std::numeric_limits<unsigned>::max();
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(Text, Options);
  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());
  ElfView &View = *ViewOrErr;

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(View.textData(), View.textSize(), S, Decoded));
  RewriteConfig Config;
  Config.MaxSgprs = 106;
  std::vector<Trampoline> Trampolines;
  std::vector<NopSled> Sleds;
  LivenessInfo Liveness;
  llvm::StringMap<KernelPatchStats> KernelStats;
  std::vector<ScratchPatchInfo> ScratchPatches;
  DirectControlFlowInfo ControlFlow;
  HotswapProfile Prof(/*Enabled=*/false);
  PatchContext Ctx{Config,
                   Decoded,
                   View.textData(),
                   View.textSize(),
                   /*PoolBaseOffset=*/0,
                   S,
                   Trampolines,
                   Sleds,
                   View,
                   Liveness,
                   KernelStats,
                   ScratchPatches,
                   ControlFlow,
                   Prof};

  EXPECT_FALSE(findSafeSgprScratchBlock(Ctx, /*TextOffset=*/0, /*Count=*/1,
                                        /*Alignment=*/2, "unit test"));
}

TEST(SafeSgprScratchBlock, CommitRejectsObjectWithoutKernelDescriptor) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Text = assembleSingleInst("s_endpgm", S);
  ASSERT_FALSE(Text.empty());

  comgr_test::KernelDescriptorElfOptions Options;
  Options.EmitKernelDescriptorSymbol = false;
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(Text, Options);
  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());
  ElfView &View = *ViewOrErr;

  std::vector<InternalDecodedInst> Decoded;
  RewriteConfig Config;
  Config.MaxSgprs = 106;
  std::vector<Trampoline> Trampolines;
  std::vector<NopSled> Sleds;
  LivenessInfo Liveness;
  llvm::StringMap<KernelPatchStats> KernelStats;
  std::vector<ScratchPatchInfo> ScratchPatches;
  DirectControlFlowInfo ControlFlow;
  HotswapProfile Prof(/*Enabled=*/false);
  PatchContext Ctx{Config,
                   Decoded,
                   View.textData(),
                   View.textSize(),
                   /*PoolBaseOffset=*/0,
                   S,
                   Trampolines,
                   Sleds,
                   View,
                   Liveness,
                   KernelStats,
                   ScratchPatches,
                   ControlFlow,
                   Prof};

  const SafeSgprScratchBlock Block{/*Base=*/4, /*Count=*/1};
  EXPECT_FALSE(
      commitSafeSgprScratchBlock(Ctx, /*TextOffset=*/0, Block, "unit test"));
}

TEST(FindNearestSled, RejectsOverflowingHeadroom) {
  std::vector<NopSled> Sleds = {{0, 64, 60, 0, 64}, {100, 128, 100, 100, 128}};
  EXPECT_EQ(findNearestSled(Sleds, 0, std::numeric_limits<uint64_t>::max()),
            nullptr);
}

TEST(FindNearestSled, HandlesLargeUnsignedOffsets) {
  std::vector<NopSled> Sleds = {{100, 128, 100, 100, 128},
                                {std::numeric_limits<uint64_t>::max() - 32,
                                 std::numeric_limits<uint64_t>::max(),
                                 std::numeric_limits<uint64_t>::max() - 32,
                                 std::numeric_limits<uint64_t>::max() - 64,
                                 std::numeric_limits<uint64_t>::max()}};
  NopSled *Sled =
      findNearestSled(Sleds, std::numeric_limits<uint64_t>::max() - 40,
                      /*Needed=*/8);
  ASSERT_NE(Sled, nullptr);
  EXPECT_EQ(Sled, &Sleds[1]);
}

TEST(GatewaySlotReachability, CursorCanCrossLowerReachBoundary) {
  const uint64_t Source = MaxSledDistance + 0x1000;
  const uint64_t Start = Source - MaxSledDistance;
  NopSled Sled = {Start, Start + 24, Start, 0,
                  std::numeric_limits<uint64_t>::max()};
  Sled.GlobalGateway = true;

  // The initial cursor is exactly out of range, but advancing one dword leaves
  // a complete 20-byte gateway at a reachable address.
  EXPECT_TRUE(canEverReachGatewaySlot(Sled, Source, /*Needed=*/20));
  NopSled FourByteAdvance = Sled;
  FourByteAdvance.WritePos += MinInstSize;
  EXPECT_TRUE(canEverReachGatewaySlot(FourByteAdvance, Source, /*Needed=*/20));
  NopSled TwentyByteAdvance = Sled;
  TwentyByteAdvance.WritePos += 20;
  EXPECT_FALSE(
      canEverReachGatewaySlot(TwentyByteAdvance, Source, /*Needed=*/20));

  Sled.End = Start + 20;
  EXPECT_FALSE(canEverReachGatewaySlot(Sled, Source, /*Needed=*/20));
}

TEST(GatewaySlotReachability, CursorCannotMoveBackIntoUpperReachBoundary) {
  const uint64_t Source = 0x1000;
  NopSled Sled = {Source + MaxSledDistance - MinInstSize,
                  Source + MaxSledDistance + 64, Source + MaxSledDistance, 0,
                  std::numeric_limits<uint64_t>::max()};
  Sled.GlobalGateway = true;

  EXPECT_FALSE(canEverReachGatewaySlot(Sled, Source, /*Needed=*/20));
  Sled.WritePos -= MinInstSize;
  EXPECT_TRUE(canEverReachGatewaySlot(Sled, Source, /*Needed=*/20));
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

TEST(AssembleDecode, SingleInstructionRejectsSequence) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  llvm::SmallVector<uint8_t> Bytes = assembleSingleInst("s_nop 0\ns_endpgm", S);
  EXPECT_TRUE(Bytes.empty());
}

TEST(AssembleDecode, InstructionSequenceRoundTrip) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  llvm::SmallVector<uint8_t> Bytes =
      assembleInstructions("s_nop 0\ns_endpgm", S);
  ASSERT_EQ(Bytes.size(), 2u * MinInstSize);

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 2u);
  EXPECT_EQ(Decoded[0].Mnemonic, "s_nop");
  EXPECT_EQ(Decoded[1].Mnemonic, "s_endpgm");
}

TEST(AssembleDecode, CvtPkFp8LiteralSourcesDecodeAsTwelveBytes) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  llvm::SmallVector<uint8_t> Bytes = assembleSingleInst(
      "v_cvt_pk_fp8_f32 v4, 0x477f0000, 0x477f0000 clamp", S);
  ASSERT_EQ(Bytes.size(), 3u * MinInstSize);

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 1u);
  const InternalDecodedInst &DI = Decoded[0];
  EXPECT_EQ(DI.Size, 3u * MinInstSize);
  EXPECT_EQ(DI.Mnemonic, "v_cvt_pk_fp8_f32");

  const llvm::MCInst &Inst = DI.Inst;
  ASSERT_GE(Inst.getNumOperands(), 7u);
  EXPECT_TRUE(Inst.getOperand(0).isReg());
  ASSERT_TRUE(Inst.getOperand(2).isImm());
  EXPECT_EQ(Inst.getOperand(2).getImm(), 0x477f0000);
  ASSERT_TRUE(Inst.getOperand(4).isImm());
  EXPECT_EQ(Inst.getOperand(4).getImm(), 0x477f0000);
  ASSERT_TRUE(Inst.getOperand(5).isImm());
  EXPECT_EQ(Inst.getOperand(5).getImm(), 1);
}

TEST(AssembleDecode, CvtPkFp8MixedLiteralSourcesDecodeAsTwelveBytes) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  llvm::SmallVector<uint8_t> Src0LiteralBytes =
      assembleSingleInst("v_cvt_pk_fp8_f32 v4, 0x477f0000, v5 clamp", S);
  ASSERT_EQ(Src0LiteralBytes.size(), 3u * MinInstSize);

  std::vector<InternalDecodedInst> Src0LiteralDecoded;
  ASSERT_TRUE(decodeTextSection(
      Src0LiteralBytes.data(), Src0LiteralBytes.size(), S, Src0LiteralDecoded));
  ASSERT_EQ(Src0LiteralDecoded.size(), 1u);
  const llvm::MCInst &Src0LiteralInst = Src0LiteralDecoded[0].Inst;
  ASSERT_GE(Src0LiteralInst.getNumOperands(), 7u);
  ASSERT_TRUE(Src0LiteralInst.getOperand(2).isImm());
  EXPECT_EQ(Src0LiteralInst.getOperand(2).getImm(), 0x477f0000);
  EXPECT_TRUE(Src0LiteralInst.getOperand(4).isReg());

  llvm::SmallVector<uint8_t> Src1LiteralBytes = assembleSingleInst(
      "v_cvt_pk_fp8_f32 v4, v5, 0.3333333432674408 clamp", S);
  ASSERT_EQ(Src1LiteralBytes.size(), 3u * MinInstSize);

  std::vector<InternalDecodedInst> Src1LiteralDecoded;
  ASSERT_TRUE(decodeTextSection(
      Src1LiteralBytes.data(), Src1LiteralBytes.size(), S, Src1LiteralDecoded));
  ASSERT_EQ(Src1LiteralDecoded.size(), 1u);
  const llvm::MCInst &Src1LiteralInst = Src1LiteralDecoded[0].Inst;
  ASSERT_GE(Src1LiteralInst.getNumOperands(), 7u);
  EXPECT_TRUE(Src1LiteralInst.getOperand(2).isReg());
  ASSERT_TRUE(Src1LiteralInst.getOperand(4).isImm());
  EXPECT_EQ(Src1LiteralInst.getOperand(4).getImm(), 0x3eaaaaab);
}

TEST(AssembleDecode, CvtPkFp8InlineConstantsDecodeAsEightBytes) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  llvm::SmallVector<uint8_t> Bytes =
      assembleSingleInst("v_cvt_pk_fp8_f32 v4, 1.0, 0.5 clamp", S);
  ASSERT_EQ(Bytes.size(), 2u * MinInstSize);

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 1u);
  const InternalDecodedInst &DI = Decoded[0];
  EXPECT_EQ(DI.Size, 2u * MinInstSize);
  EXPECT_EQ(DI.Mnemonic, "v_cvt_pk_fp8_f32");

  const llvm::MCInst &Inst = DI.Inst;
  ASSERT_GE(Inst.getNumOperands(), 7u);
  ASSERT_TRUE(Inst.getOperand(2).isImm());
  EXPECT_EQ(Inst.getOperand(2).getImm(), 0x3f800000);
  ASSERT_TRUE(Inst.getOperand(4).isImm());
  EXPECT_EQ(Inst.getOperand(4).getImm(), 0x3f000000);
  ASSERT_TRUE(Inst.getOperand(5).isImm());
  EXPECT_EQ(Inst.getOperand(5).getImm(), 1);
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

static void expectSameOperands(const llvm::MCInst &Actual,
                               const llvm::MCInst &Expected,
                               llvm::StringRef Context) {
  EXPECT_EQ(Actual.getOpcode(), Expected.getOpcode()) << Context.str();
  ASSERT_EQ(Actual.getNumOperands(), Expected.getNumOperands())
      << Context.str();
  for (unsigned I = 0, E = Actual.getNumOperands(); I != E; ++I) {
    const llvm::MCOperand &ActualOp = Actual.getOperand(I);
    const llvm::MCOperand &ExpectedOp = Expected.getOperand(I);
    EXPECT_EQ(ActualOp.isReg(), ExpectedOp.isReg())
        << Context.str() << " operand " << I;
    EXPECT_EQ(ActualOp.isImm(), ExpectedOp.isImm())
        << Context.str() << " operand " << I;
    EXPECT_EQ(ActualOp.isSFPImm(), ExpectedOp.isSFPImm())
        << Context.str() << " operand " << I;
    EXPECT_EQ(ActualOp.isDFPImm(), ExpectedOp.isDFPImm())
        << Context.str() << " operand " << I;
    EXPECT_EQ(ActualOp.isExpr(), ExpectedOp.isExpr())
        << Context.str() << " operand " << I;
    if (ExpectedOp.isReg()) {
      EXPECT_EQ(ActualOp.getReg(), ExpectedOp.getReg())
          << Context.str() << " operand " << I;
    } else if (ExpectedOp.isImm()) {
      EXPECT_EQ(ActualOp.getImm(), ExpectedOp.getImm())
          << Context.str() << " operand " << I;
    } else if (ExpectedOp.isSFPImm()) {
      EXPECT_EQ(ActualOp.getSFPImm(), ExpectedOp.getSFPImm())
          << Context.str() << " operand " << I;
    } else if (ExpectedOp.isDFPImm()) {
      EXPECT_EQ(ActualOp.getDFPImm(), ExpectedOp.getDFPImm())
          << Context.str() << " operand " << I;
    }
  }
}

static void expectInstMatchesAsm(const llvm::MCInst &Actual,
                                 llvm::StringRef Asm, const LLVMState &S) {
  llvm::MCInst Expected = assembleOne(Asm, S);
  expectSameOperands(Actual, Expected, Asm);
}

static bool appendSingleInstBytes(llvm::SmallVectorImpl<uint8_t> &Bytes,
                                  llvm::StringRef Asm, const LLVMState &S) {
  llvm::SmallVector<uint8_t> Inst = assembleSingleInst(Asm, S);
  if (Inst.empty()) {
    ADD_FAILURE() << "failed to assemble: " << Asm.str();
    return false;
  }
  Bytes.append(Inst.begin(), Inst.end());
  return true;
}

static std::vector<InternalDecodedInst>
decodeInstSequence(llvm::ArrayRef<llvm::StringRef> Instructions,
                   const LLVMState &S) {
  llvm::SmallVector<uint8_t> Bytes;
  for (llvm::StringRef Inst : Instructions)
    if (!appendSingleInstBytes(Bytes, Inst, S))
      return {};
  std::vector<InternalDecodedInst> Decoded;
  EXPECT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  return Decoded;
}

static void expectCachedSiteDeadMatchesOracle(
    llvm::ArrayRef<llvm::StringRef> Instructions, size_t ResumeIndex,
    const LLVMState &S) {
  llvm::SmallVector<uint8_t> Text;
  for (llvm::StringRef Inst : Instructions)
    ASSERT_TRUE(appendSingleInstBytes(Text, Inst, S));

  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.MetadataSgprCount = 106;
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(Text, Opts);
  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(ViewOrErr->textData(), ViewOrErr->textSize(), S,
                                Decoded));
  ASSERT_EQ(Decoded.size(), Instructions.size());
  ASSERT_GT(ResumeIndex, 0u);
  ASSERT_LT(ResumeIndex, Decoded.size());

  RewriteConfig Config;
  Config.MaxSgprs = 106;
  std::vector<Trampoline> Trampolines;
  std::vector<NopSled> Sleds;
  LivenessInfo Liveness;
  llvm::StringMap<KernelPatchStats> KernelStats;
  std::vector<ScratchPatchInfo> ScratchPatches;
  DirectControlFlowInfo ControlFlow;
  HotswapProfile Prof(/*Enabled=*/false);
  PatchContext Ctx{Config, Decoded, ViewOrErr->textData(),
                   ViewOrErr->textSize(), 0, S, Trampolines, Sleds,
                   *ViewOrErr, Liveness, KernelStats, ScratchPatches,
                   ControlFlow, Prof};
  precomputeSiteDeadSgprFacts(Ctx);

  const InternalDecodedInst &Previous = Decoded[ResumeIndex - 1];
  std::optional<llvm::BitVector> Cached = getSiteDeadNumberedSgprs(
      Ctx, Previous.Offset, Previous.Size);
  ASSERT_TRUE(Cached);
  for (unsigned Pair = 0; Pair + 1 < Config.MaxSgprs; Pair += 2) {
    bool CachedPair = Cached->test(Pair) && Cached->test(Pair + 1);
    bool OraclePair = isSgprPairDeadFrom(
        Decoded, ResumeIndex, Pair, S,
        llvm::ArrayRef<uint8_t>(ViewOrErr->textData(), ViewOrErr->textSize()));
    EXPECT_EQ(CachedPair, OraclePair) << "pair s[" << Pair << ':' << Pair + 1
                                      << ']';
  }
}

TEST(CollectTouchedNumberedSgprs, ChecksTheCompleteReplacement) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Clean;
  ASSERT_TRUE(appendSingleInstBytes(Clean, "s_wait_dscnt 0", S));
  ASSERT_TRUE(appendSingleInstBytes(Clean, "v_add_u32 v0, v1, v2", S));
  std::optional<llvm::BitVector> CleanTouched =
      collectTouchedNumberedSgprs(Clean, /*NumberedSgprLimit=*/106, S);
  ASSERT_TRUE(CleanTouched);
  EXPECT_FALSE(CleanTouched->test(62));
  EXPECT_FALSE(CleanTouched->test(63));

  llvm::SmallVector<uint8_t> TouchesPair = Clean;
  ASSERT_TRUE(
      appendSingleInstBytes(TouchesPair, "s_mov_b64 s[62:63], s[0:1]", S));
  std::optional<llvm::BitVector> PairTouched =
      collectTouchedNumberedSgprs(TouchesPair,
                                  /*NumberedSgprLimit=*/106, S);
  ASSERT_TRUE(PairTouched);
  EXPECT_TRUE(PairTouched->test(0));
  EXPECT_TRUE(PairTouched->test(1));
  EXPECT_TRUE(PairTouched->test(62));
  EXPECT_TRUE(PairTouched->test(63));
}

TEST(SgprPairDeadFrom, AcceptsDefsBeforeUsesOnEveryDirectPath) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  std::vector<InternalDecodedInst> Decoded = decodeInstSequence(
      {"s_nop 0", "s_cbranch_scc0 2", "s_cselect_b32 s62, -1, 0", "s_branch 1",
       "s_cselect_b32 s62, -1, 0", "s_cselect_b32 s63, -1, 0",
       "s_cmp_lg_u64 s[62:63], 0", "s_endpgm"},
      S);
  ASSERT_EQ(Decoded.size(), 8u);
  EXPECT_TRUE(isSgprPairDeadFrom(Decoded, /*ResumeIndex=*/1,
                                 /*SgprBase=*/62, S));
}

TEST(SgprPairDeadFrom, RejectsUseBeforeDefOnOneDirectPath) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  std::vector<InternalDecodedInst> Decoded = decodeInstSequence(
      {"s_nop 0", "s_cbranch_scc0 2", "s_cselect_b32 s62, -1, 0", "s_branch 1",
       "s_cmp_lg_u32 s62, 0", "s_cselect_b32 s63, -1, 0", "s_endpgm"},
      S);
  ASSERT_EQ(Decoded.size(), 7u);
  EXPECT_FALSE(isSgprPairDeadFrom(Decoded, /*ResumeIndex=*/1,
                                  /*SgprBase=*/62, S));
}

TEST(SgprPairDeadFrom, RejectsHiddenTiedUseBeforeDefinition) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  std::vector<InternalDecodedInst> Decoded = decodeInstSequence(
      {"s_nop 0", "s_bitset0_b64 s[62:63], 1", "s_endpgm"}, S);
  ASSERT_EQ(Decoded.size(), 3u);
  EXPECT_FALSE(isSgprPairDeadFrom(Decoded, /*ResumeIndex=*/1,
                                  /*SgprBase=*/62, S));
}

TEST(SgprPairDeadFrom, SwapPcCallFullyDefinesLinkPair) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  std::vector<InternalDecodedInst> Decoded =
      decodeInstSequence({"s_nop 0", "s_swap_pc_i64 s[30:31], s[2:3]",
                          "s_cmp_lg_u64 s[30:31], 0", "s_endpgm"},
                         S);
  ASSERT_EQ(Decoded.size(), 4u);
  EXPECT_TRUE(isSgprPairDeadFrom(Decoded, /*ResumeIndex=*/1,
                                 /*SgprBase=*/30, S));
}

TEST(SgprPairDeadFrom, TracksTheTwoHalvesIndependently) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  std::vector<InternalDecodedInst> Safe = decodeInstSequence(
      {"s_nop 0", "s_cselect_b32 s62, -1, 0", "s_cselect_b32 s63, -1, 0",
       "s_cmp_lg_u64 s[62:63], 0", "s_endpgm"},
      S);
  ASSERT_EQ(Safe.size(), 5u);
  EXPECT_TRUE(isSgprPairDeadFrom(Safe, /*ResumeIndex=*/1,
                                 /*SgprBase=*/62, S));

  std::vector<InternalDecodedInst> HighLive =
      decodeInstSequence({"s_nop 0", "s_cselect_b32 s62, -1, 0",
                          "s_cmp_lg_u32 s63, 0", "s_endpgm"},
                         S);
  ASSERT_EQ(HighLive.size(), 4u);
  EXPECT_FALSE(isSgprPairDeadFrom(HighLive, /*ResumeIndex=*/1,
                                  /*SgprBase=*/62, S));
}

TEST(SgprPairDeadFrom, ConvergesAcrossABackedge) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  std::vector<InternalDecodedInst> Decoded = decodeInstSequence(
      {"s_nop 0", "s_cselect_b32 s62, -1, 0", "s_cbranch_scc0 -2",
       "s_cselect_b32 s63, -1, 0", "s_cmp_lg_u64 s[62:63], 0", "s_endpgm"},
      S);
  ASSERT_EQ(Decoded.size(), 6u);
  EXPECT_TRUE(isSgprPairDeadFrom(Decoded, /*ResumeIndex=*/1,
                                 /*SgprBase=*/62, S));
}

TEST(SgprPairDeadFrom, RejectsUnboundedControlFlow) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  std::vector<InternalDecodedInst> Indirect =
      decodeInstSequence({"s_nop 0", "s_setpc_b64 s[40:41]", "s_endpgm"}, S);
  ASSERT_EQ(Indirect.size(), 3u);
  EXPECT_FALSE(isSgprPairDeadFrom(Indirect, /*ResumeIndex=*/1,
                                  /*SgprBase=*/62, S));

  std::vector<InternalDecodedInst> Outside =
      decodeInstSequence({"s_nop 0", "s_branch 10", "s_endpgm"}, S);
  ASSERT_EQ(Outside.size(), 3u);
  EXPECT_FALSE(isSgprPairDeadFrom(Outside, /*ResumeIndex=*/1,
                                  /*SgprBase=*/62, S));
}

TEST(SgprPairDeadFrom, RejectsLivePairAtCall) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  std::vector<InternalDecodedInst> Live = decodeInstSequence(
      {"s_nop 0", "s_call_i64 s[30:31], 0", "s_endpgm"}, S);
  ASSERT_EQ(Live.size(), 3u);
  EXPECT_FALSE(isSgprPairDeadFrom(Live, /*ResumeIndex=*/1,
                                  /*SgprBase=*/62, S));

  std::vector<InternalDecodedInst> Killed = decodeInstSequence(
      {"s_nop 0", "s_mov_b64 s[62:63], 0", "s_call_i64 s[30:31], 0",
       "s_endpgm"},
      S);
  ASSERT_EQ(Killed.size(), 4u);
  EXPECT_TRUE(isSgprPairDeadFrom(Killed, /*ResumeIndex=*/1,
                                 /*SgprBase=*/62, S));
}

TEST(SgprPairDeadFrom, RejectsLivePairAtFunctionExit) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  std::vector<InternalDecodedInst> Decoded = decodeInstSequence(
      {"s_nop 0", "s_set_pc_i64 s[30:31]"}, S);
  ASSERT_EQ(Decoded.size(), 2u);
  EXPECT_FALSE(isSgprPairDeadFrom(Decoded, /*ResumeIndex=*/1,
                                  /*SgprBase=*/62, S));
}

TEST(SgprPairDeadFrom, RequiresDefinitionOnEveryExitPath) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  std::vector<InternalDecodedInst> Bypass = decodeInstSequence(
      {"s_nop 0", "s_cbranch_scc0 2", "s_mov_b64 s[62:63], 0",
       "s_branch 1", "s_nop 0", "s_endpgm"},
      S);
  ASSERT_EQ(Bypass.size(), 6u);
  EXPECT_FALSE(isSgprPairDeadFrom(Bypass, /*ResumeIndex=*/1,
                                  /*SgprBase=*/62, S));

  std::vector<InternalDecodedInst> BothDefined = decodeInstSequence(
      {"s_nop 0", "s_cbranch_scc0 2", "s_mov_b64 s[62:63], 0",
       "s_branch 1", "s_mov_b64 s[62:63], 0", "s_endpgm"},
      S);
  ASSERT_EQ(BothDefined.size(), 6u);
  EXPECT_TRUE(isSgprPairDeadFrom(BothDefined, /*ResumeIndex=*/1,
                                 /*SgprBase=*/62, S));
}

TEST(VccPairDeadFrom, AcceptsWave32LowDefinitionAndStandardReturn) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  std::vector<InternalDecodedInst> Decoded = decodeInstSequence(
      {"s_nop 0", "v_cmp_lt_u32_e32 vcc_lo, v0, v1",
       "s_and_b32 s0, s1, vcc_lo", "s_set_pc_i64 s[30:31]"},
      S);
  ASSERT_EQ(Decoded.size(), 4u);
  // The implicit wave32 compare kills VCC_LO. VCC_HI is unused until the
  // exact ABI return, where caller-clobbered VCC stops being observable.
  EXPECT_TRUE(isVccPairDeadFrom(Decoded, /*ResumeIndex=*/1, S));
}

TEST(VccPairDeadFrom, RejectsUseBeforeDefinitionOnOnePath) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  std::vector<InternalDecodedInst> Decoded = decodeInstSequence(
      {"s_nop 0", "s_cbranch_scc0 2",
       "v_cmp_lt_u32_e32 vcc_lo, v0, v1", "s_branch 1",
       "s_and_b32 s0, s1, vcc_lo", "s_set_pc_i64 s[30:31]"},
      S);
  ASSERT_EQ(Decoded.size(), 6u);
  EXPECT_FALSE(isVccPairDeadFrom(Decoded, /*ResumeIndex=*/1, S));
}

TEST(VccPairDeadFrom, RejectsExplicitHighUseBeforeDefinition) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  std::vector<InternalDecodedInst> Decoded = decodeInstSequence(
      {"s_nop 0", "s_cmp_lg_u32 vcc_hi, 0",
       "v_cmp_lt_u32_e32 vcc_lo, v0, v1", "s_set_pc_i64 s[30:31]"},
      S);
  ASSERT_EQ(Decoded.size(), 4u);
  EXPECT_FALSE(isVccPairDeadFrom(Decoded, /*ResumeIndex=*/1, S));
}

TEST(VccPairDeadFrom, RejectsImplicitPredicateUseBeforeDefinition) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  std::vector<InternalDecodedInst> Decoded = decodeInstSequence(
      {"s_nop 0", "s_cbranch_vccz 1",
       "v_cmp_lt_u32_e32 vcc_lo, v0, v1", "s_set_pc_i64 s[30:31]"},
      S);
  ASSERT_EQ(Decoded.size(), 4u);
  EXPECT_FALSE(isVccPairDeadFrom(Decoded, /*ResumeIndex=*/1, S));
}

TEST(VccPairDeadFrom, RejectsUnknownIndirectTransfer) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  std::vector<InternalDecodedInst> Decoded = decodeInstSequence(
      {"s_nop 0", "s_setpc_b64 s[40:41]"}, S);
  ASSERT_EQ(Decoded.size(), 2u);
  EXPECT_FALSE(isVccPairDeadFrom(Decoded, /*ResumeIndex=*/1, S));
}

TEST(VccPairDeadFrom, AcceptsStandardLinkCallBoundary) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  std::vector<InternalDecodedInst> Decoded = decodeInstSequence(
      {"s_nop 0", "s_swap_pc_i64 s[30:31], s[2:3]"}, S);
  ASSERT_EQ(Decoded.size(), 2u);
  EXPECT_TRUE(isVccPairDeadFrom(Decoded, /*ResumeIndex=*/1, S));
}

TEST(SiteDeadSgprFacts, CachedMasksMatchPathOracle) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  expectCachedSiteDeadMatchesOracle(
      {"s_nop 0", "s_cbranch_scc0 2", "s_cselect_b32 s62, -1, 0",
       "s_branch 1", "s_cselect_b32 s62, -1, 0",
       "s_cselect_b32 s63, -1, 0", "s_cmp_lg_u64 s[62:63], 0",
       "s_endpgm"},
      1, S);
  expectCachedSiteDeadMatchesOracle(
      {"s_nop 0", "s_cbranch_scc0 2", "s_cselect_b32 s62, -1, 0",
       "s_branch 1", "s_cmp_lg_u32 s62, 0", "s_cselect_b32 s63, -1, 0",
       "s_endpgm"},
      1, S);
  expectCachedSiteDeadMatchesOracle(
      {"s_nop 0", "s_bitset0_b64 s[62:63], 1", "s_endpgm"}, 1, S);
  expectCachedSiteDeadMatchesOracle(
      {"s_nop 0", "s_swap_pc_i64 s[30:31], s[2:3]",
       "s_cmp_lg_u64 s[30:31], 0", "s_endpgm"},
      1, S);
  expectCachedSiteDeadMatchesOracle(
      {"s_nop 0", "s_cselect_b32 s62, -1, 0", "s_cbranch_scc0 -2",
       "s_cselect_b32 s63, -1, 0", "s_cmp_lg_u64 s[62:63], 0",
       "s_endpgm"},
      1, S);
  expectCachedSiteDeadMatchesOracle(
      {"s_nop 0", "s_setpc_b64 s[40:41]", "s_endpgm"}, 1, S);
  expectCachedSiteDeadMatchesOracle(
      {"s_nop 0", "s_call_i64 s[30:31], 0", "s_endpgm"}, 1, S);
  expectCachedSiteDeadMatchesOracle(
      {"s_nop 0", "s_mov_b64 s[62:63], 0", "s_call_i64 s[30:31], 0",
       "s_endpgm"},
      1, S);
  expectCachedSiteDeadMatchesOracle(
      {"s_nop 0", "s_cbranch_scc0 2", "s_mov_b64 s[62:63], 0",
       "s_branch 1", "s_nop 0", "s_endpgm"},
      1, S);
  expectCachedSiteDeadMatchesOracle(
      {"s_nop 0", "s_cbranch_scc0 2", "s_mov_b64 s[62:63], 0",
       "s_branch 1", "s_mov_b64 s[62:63], 0", "s_endpgm"},
      1, S);
}

TEST(SiteDeadSgprFacts, ProofPoisonDoesNotInflateNumberedLimit) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  llvm::SmallVector<uint8_t> Text = {0xff, 0xff, 0xff, 0xff};
  ASSERT_TRUE(appendSingleInstBytes(Text, "s_mov_b32 s5, 0", S));
  ASSERT_TRUE(appendSingleInstBytes(Text, "s_endpgm", S));
  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.EmitKernelDescriptorSymbol = false;
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(Text, Opts);
  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(ViewOrErr->textData(), ViewOrErr->textSize(), S,
                                Decoded));
  ASSERT_EQ(Decoded.size(), 3u);
  ASSERT_EQ(Decoded[0].Mnemonic, "<unknown>");

  RewriteConfig Config;
  Config.MaxSgprs = 106;
  std::vector<Trampoline> Trampolines;
  std::vector<NopSled> Sleds;
  LivenessInfo Liveness;
  llvm::StringMap<KernelPatchStats> KernelStats;
  std::vector<ScratchPatchInfo> ScratchPatches;
  DirectControlFlowInfo ControlFlow;
  HotswapProfile Prof(/*Enabled=*/false);
  PatchContext Ctx{Config, Decoded, ViewOrErr->textData(),
                   ViewOrErr->textSize(), 0, S, Trampolines, Sleds,
                   *ViewOrErr, Liveness, KernelStats, ScratchPatches,
                   ControlFlow, Prof};
  precomputeSiteDeadSgprFacts(Ctx);

  auto It = Ctx.SiteDeadSgprFacts.find({0, ViewOrErr->textSize()});
  ASSERT_NE(It, Ctx.SiteDeadSgprFacts.end());
  EXPECT_EQ(It->second.NumberedLimit, 6u);
}

TEST(FindSafeSgprScratchBlock, RejectsUndecodedInstructions) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  llvm::SmallVector<uint8_t> Text = {0xff, 0xff, 0xff, 0xff};
  ASSERT_TRUE(appendSingleInstBytes(Text, "s_endpgm", S));
  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.MetadataSgprCount = 8;
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(Text, Opts);
  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(ViewOrErr->textData(), ViewOrErr->textSize(), S,
                                Decoded));
  ASSERT_EQ(Decoded.size(), 2u);
  ASSERT_EQ(Decoded[0].Mnemonic, "<unknown>");

  RewriteConfig Config;
  Config.MaxSgprs = 106;
  std::vector<Trampoline> Trampolines;
  std::vector<NopSled> Sleds;
  LivenessInfo Liveness;
  llvm::StringMap<KernelPatchStats> KernelStats;
  std::vector<ScratchPatchInfo> ScratchPatches;
  DirectControlFlowInfo ControlFlow;
  HotswapProfile Prof(/*Enabled=*/false);
  PatchContext Ctx{Config, Decoded, ViewOrErr->textData(),
                   ViewOrErr->textSize(), 0, S, Trampolines, Sleds,
                   *ViewOrErr, Liveness, KernelStats, ScratchPatches,
                   ControlFlow, Prof};

  EXPECT_FALSE(findSafeSgprScratchBlock(Ctx, /*TextOffset=*/0, /*Count=*/2,
                                        /*Alignment=*/2, "unit test"));
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

// -- buildKernelEntryTrampoline -----------------------------------------------

TEST(BuildKernelEntryTrampoline, BuildsRecognizedPcRelativeStub) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  constexpr uint64_t StubVAddr = 0x200000;
  constexpr uint64_t EntryVAddr = 0x10100;
  llvm::SmallVector<uint8_t> GlobalWb = assembleSingleInst("global_wb", S);
  ASSERT_EQ(GlobalWb.size(), 3 * MinInstSize);

  llvm::SmallVector<uint8_t> Bytes =
      buildKernelEntryTrampoline(StubVAddr, EntryVAddr, /*ScratchSgpr=*/8, S);

  ASSERT_EQ(Bytes.size(), KernelEntryStubStride);
  EXPECT_TRUE(isKernelEntryTrampoline(Bytes, S));

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_GE(Decoded.size(), 6u);
  EXPECT_EQ(Decoded[0].Inst.getOpcode(), S.GlobalWbOpcode);
  EXPECT_EQ(Decoded[1].Inst.getOpcode(), S.VNopInst.getOpcode());
  EXPECT_EQ(Decoded[2].Inst.getOpcode(), S.SGetPcI64Opcode);
  EXPECT_EQ(Decoded[3].Inst.getOpcode(), S.SAddU32Opcode);
  EXPECT_EQ(Decoded[4].Inst.getOpcode(), S.SAddcU32Opcode);
  EXPECT_EQ(Decoded[5].Inst.getOpcode(), S.SSetPcI64Opcode);

  const uint64_t PcBase = StubVAddr + Decoded[2].Offset + Decoded[2].Size;
  const uint64_t Delta = EntryVAddr - PcBase;
  const uint32_t Lo = static_cast<uint32_t>(Delta);
  const uint32_t Hi = static_cast<uint32_t>(Delta >> 32);
  expectInstMatchesAsm(Decoded[0].Inst, "global_wb", S);
  expectInstMatchesAsm(Decoded[1].Inst, "v_nop", S);
  expectInstMatchesAsm(Decoded[2].Inst, "s_get_pc_i64 s[8:9]", S);
  expectInstMatchesAsm(
      Decoded[3].Inst,
      (llvm::Twine("s_add_u32 s8, s8, 0x") + llvm::utohexstr(Lo)).str(), S);
  expectInstMatchesAsm(
      Decoded[4].Inst,
      (llvm::Twine("s_addc_u32 s9, s9, 0x") + llvm::utohexstr(Hi)).str(), S);
  expectInstMatchesAsm(Decoded[5].Inst, "s_set_pc_i64 s[8:9]", S);
}

TEST(BuildKernelEntryTrampoline, SpellsLogicalTopPairAsVcc) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  constexpr uint64_t StubVAddr = 0x200000;
  constexpr uint64_t EntryVAddr = 0x10100;
  llvm::SmallVector<uint8_t> Bytes =
      buildKernelEntryTrampoline(StubVAddr, EntryVAddr,
                                 /*ScratchSgpr=*/Gfx1250MaxNumberedSgprs, S);

  ASSERT_EQ(Bytes.size(), KernelEntryStubStride);
  EXPECT_TRUE(isKernelEntryTrampoline(Bytes, S));
  EXPECT_EQ(getKernelEntryTrampolineTargetVAddr(Bytes, StubVAddr, S),
            EntryVAddr);
  expectEntryStubUsesVcc(Bytes, S);
}

TEST(BuildKernelEntryTrampoline, PrefixPrefiltersNonStubBytes) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  llvm::SmallVector<uint8_t> Stub =
      buildKernelEntryTrampoline(/*StubVAddr=*/0x200000,
                                 /*EntryVAddr=*/0x10100,
                                 /*ScratchSgpr=*/8, S);
  ASSERT_EQ(Stub.size(), KernelEntryStubStride);
  EXPECT_TRUE(hasKernelEntryTrampolinePrefix(Stub, S));

  llvm::SmallVector<uint8_t> NonStub;
  ASSERT_TRUE(appendSingleInstBytes(NonStub, "s_endpgm", S));
  while (NonStub.size() < KernelEntryStubStride)
    NonStub.append(S.SNopBytes.begin(), S.SNopBytes.end());
  ASSERT_EQ(NonStub.size(), KernelEntryStubStride);

  EXPECT_FALSE(hasKernelEntryTrampolinePrefix(NonStub, S));
  EXPECT_FALSE(isKernelEntryTrampoline(NonStub, S));

  llvm::ArrayRef<uint8_t> ShortCandidate(Stub.data(), MinInstSize);
  EXPECT_FALSE(hasKernelEntryTrampolinePrefix(ShortCandidate, S));
}

TEST(BuildKernelEntryTrampoline, PrefixPrefiltersHipblasltSmokeEntryBytes) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  // Reduced from the gfx1250 hipBLASLt MXF8/BF16 smoke kernel entry. The
  // idempotency path should reject this by raw prefix before classifying it as
  // a possible appended entry stub.
  const uint8_t EntryBytes[] = {
      0x1a, 0x08, 0x80, 0xb9, 0x02, 0x00, 0x00, 0x00, 0x1a, 0x08, 0x80,
      0xb9, 0x02, 0x00, 0x00, 0x00, 0xff, 0x02, 0x3f, 0x8b, 0xff, 0xff,
      0xff, 0x3f, 0x02, 0x9e, 0x40, 0x85, 0x03, 0x00, 0xc1, 0xbe,
  };

  llvm::SmallVector<uint8_t> Candidate;
  Candidate.append(EntryBytes, EntryBytes + sizeof(EntryBytes));
  while (Candidate.size() < KernelEntryStubStride)
    Candidate.append(S.SNopBytes.begin(), S.SNopBytes.end());
  ASSERT_EQ(Candidate.size(), KernelEntryStubStride);

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(
      decodeTextSection(Candidate.data(), sizeof(EntryBytes), S, Decoded));
  ASSERT_GE(Decoded.size(), 5u);
  EXPECT_EQ(Decoded[0].Mnemonic, "s_setreg_imm32_b32");
  EXPECT_EQ(Decoded[1].Mnemonic, "s_setreg_imm32_b32");
  EXPECT_EQ(Decoded[2].Mnemonic, "s_and_b32");
  EXPECT_FALSE(hasKernelEntryTrampolinePrefix(Candidate, S));
  EXPECT_FALSE(isKernelEntryTrampoline(Candidate, S));
}

TEST(BuildKernelEntryTrampoline, PrefixPrefiltersUnknownDecodeBytes) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  const uint8_t UnknownInst[] = {0xff, 0xff, 0xff, 0xff};

  llvm::SmallVector<uint8_t> Candidate;
  Candidate.append(UnknownInst, UnknownInst + sizeof(UnknownInst));
  while (Candidate.size() < KernelEntryStubStride)
    Candidate.append(S.SNopBytes.begin(), S.SNopBytes.end());
  ASSERT_EQ(Candidate.size(), KernelEntryStubStride);

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Candidate.data(), MinInstSize, S, Decoded));
  ASSERT_EQ(Decoded.size(), 1u);
  EXPECT_EQ(Decoded[0].Mnemonic, "<unknown>");
  EXPECT_FALSE(hasKernelEntryTrampolinePrefix(Candidate, S));
  EXPECT_FALSE(isKernelEntryTrampoline(Candidate, S));
}

TEST(BuildKernelEntryTrampoline, MatcherRejectsNonStubBytes) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  std::vector<uint8_t> Bytes(KernelEntryStubStride, 0);
  for (size_t I = 0; I < Bytes.size(); I += MinInstSize)
    std::memcpy(Bytes.data() + I, S.SNopBytes.data(), MinInstSize);

  EXPECT_FALSE(isKernelEntryTrampoline(Bytes, S));
}

TEST(BuildKernelEntryTrampoline, MatcherRejectsWrongOperandShape) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  llvm::SmallVector<uint8_t> Bytes;
  ASSERT_TRUE(appendSingleInstBytes(Bytes, "global_wb", S));
  ASSERT_TRUE(appendSingleInstBytes(Bytes, "v_nop", S));
  ASSERT_TRUE(appendSingleInstBytes(Bytes, "s_get_pc_i64 s[8:9]", S));
  ASSERT_TRUE(appendSingleInstBytes(Bytes, "s_add_u32 s8, s8, 0", S));
  ASSERT_TRUE(appendSingleInstBytes(Bytes, "s_addc_u32 s10, s10, 0", S));
  ASSERT_TRUE(appendSingleInstBytes(Bytes, "s_set_pc_i64 s[8:9]", S));

  llvm::SmallVector<uint8_t> CodeEnd = assembleSingleInst("s_code_end", S);
  ASSERT_EQ(CodeEnd.size(), MinInstSize);
  while (Bytes.size() < KernelEntryStubStride)
    Bytes.append(CodeEnd.begin(), CodeEnd.end());
  ASSERT_EQ(Bytes.size(), KernelEntryStubStride);

  EXPECT_TRUE(hasKernelEntryTrampolinePrefix(Bytes, S));
  EXPECT_FALSE(isKernelEntryTrampoline(Bytes, S));
}

// -- DisplacementPlan ---------------------------------------------------------

TEST(DisplacementPlan, MapsInsertionAndReplacementBoundaries) {
  std::vector<uint8_t> Text(16, 0);
  std::vector<uint8_t> ElfBytes = makeDisplacementTestElf(Text);
  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(ElfBytes.data(), ElfBytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());

  DisplacementEdit Insert;
  Insert.Offset = 4;
  Insert.OriginalSize = 0;
  Insert.ReplacementBytes.assign(8, 0x11);

  DisplacementEdit Replace;
  Replace.Offset = 8;
  Replace.OriginalSize = 4;
  Replace.ReplacementBytes.assign(8, 0x22);

  llvm::Expected<DisplacementPlan> PlanOrErr =
      DisplacementPlan::create(*ViewOrErr, {Insert, Replace});
  ASSERT_TRUE((bool)PlanOrErr) << llvm::toString(PlanOrErr.takeError());

  uint64_t Mapped = 0;
  ASSERT_TRUE(PlanOrErr->mapOffset(4, DisplacementMapBias::BeforeInsertedBytes,
                                   Mapped));
  EXPECT_EQ(Mapped, 4u);
  ASSERT_TRUE(
      PlanOrErr->mapOffset(4, DisplacementMapBias::AfterInsertedBytes, Mapped));
  EXPECT_EQ(Mapped, 12u);
  ASSERT_TRUE(PlanOrErr->mapOffset(8, DisplacementMapBias::BeforeInsertedBytes,
                                   Mapped));
  EXPECT_EQ(Mapped, 16u);
  ASSERT_TRUE(PlanOrErr->mapOffset(12, DisplacementMapBias::AfterInsertedBytes,
                                   Mapped));
  EXPECT_EQ(Mapped, 24u);
  EXPECT_FALSE(PlanOrErr->mapOffset(
      10, DisplacementMapBias::BeforeInsertedBytes, Mapped));
}

TEST(DisplacementPlan, RejectsOverlappingEdits) {
  std::vector<uint8_t> Text(16, 0);
  std::vector<uint8_t> ElfBytes = makeDisplacementTestElf(Text);
  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(ElfBytes.data(), ElfBytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());

  DisplacementEdit A;
  A.Offset = 4;
  A.OriginalSize = 8;
  A.ReplacementBytes.assign(12, 0x11);

  DisplacementEdit B;
  B.Offset = 8;
  B.OriginalSize = 4;
  B.ReplacementBytes.assign(8, 0x22);

  llvm::Expected<DisplacementPlan> PlanOrErr =
      DisplacementPlan::create(*ViewOrErr, {A, B});
  ASSERT_FALSE((bool)PlanOrErr);
  std::string Reason = llvm::toString(PlanOrErr.takeError());
  EXPECT_NE(Reason.find("overlap"), std::string::npos) << Reason;
}

TEST(DisplacementPlan, RebuildsTextAndPadsToPostTextAlignment) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  std::vector<uint8_t> Text(16);
  for (unsigned I = 0; I < Text.size(); ++I)
    Text[I] = I;
  std::vector<uint8_t> ElfBytes = makeDisplacementTestElf(Text);
  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(ElfBytes.data(), ElfBytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());

  DisplacementEdit Edit;
  Edit.Offset = 4;
  Edit.OriginalSize = 4;
  Edit.ReplacementBytes.assign(
      {0xA0, 0xA1, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7});

  llvm::Expected<DisplacementPlan> PlanOrErr =
      DisplacementPlan::create(*ViewOrErr, {Edit});
  ASSERT_TRUE((bool)PlanOrErr) << llvm::toString(PlanOrErr.takeError());
  EXPECT_EQ(PlanOrErr->rawGrowth(), 4u);
  EXPECT_EQ(PlanOrErr->paddedGrowth(), 8u);

  llvm::SmallVector<uint8_t> NewText = PlanOrErr->buildText(Text, S.SNopBytes);
  ASSERT_EQ(NewText.size(), 24u);
  EXPECT_EQ(llvm::ArrayRef<uint8_t>(NewText.data(), 4),
            llvm::ArrayRef<uint8_t>(Text.data(), 4));
  EXPECT_EQ(NewText[4], 0xA0);
  EXPECT_EQ(NewText[11], 0xA7);
  EXPECT_EQ(llvm::ArrayRef<uint8_t>(NewText.data() + 12, 8),
            llvm::ArrayRef<uint8_t>(Text.data() + 8, 8));
  EXPECT_EQ(std::memcmp(NewText.data() + 20, S.SNopBytes.data(), MinInstSize),
            0);
}

TEST(TextDisplacement, ReencodesForwardSBranchAcrossInsertion) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  llvm::SmallVector<uint8_t> Text;
  llvm::SmallVector<uint8_t> Br = S.encodeSBranch(0, 8);
  ASSERT_EQ(Br.size(), MinInstSize);
  Text.append(Br.begin(), Br.end());
  Text.append(S.SNopBytes.begin(), S.SNopBytes.end());
  llvm::SmallVector<uint8_t> End = assembleSingleInst("s_endpgm", S);
  ASSERT_EQ(End.size(), MinInstSize);
  Text.append(End.begin(), End.end());

  std::vector<uint8_t> ElfBytes = makeDisplacementTestElf(Text);
  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(ElfBytes.data(), ElfBytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());

  DisplacementEdit Edit;
  Edit.Offset = 4;
  Edit.OriginalSize = 0;
  Edit.ReplacementBytes.assign(S.SNopBytes.begin(), S.SNopBytes.end());

  llvm::Expected<std::unique_ptr<llvm::WritableMemoryBuffer>> OutOrErr =
      tryApplyTextDisplacementToNewBuffer(*ViewOrErr, S, {Edit});
  ASSERT_TRUE((bool)OutOrErr) << llvm::toString(OutOrErr.takeError());
  std::unique_ptr<llvm::WritableMemoryBuffer> Out = std::move(*OutOrErr);

  uint8_t *OutData = reinterpret_cast<uint8_t *>(Out->getBufferStart());
  llvm::Expected<ElfView> OutView =
      ElfView::create(OutData, Out->getBufferSize());
  ASSERT_TRUE((bool)OutView) << llvm::toString(OutView.takeError());

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(
      decodeTextSection(OutView->textData(), OutView->textSize(), S, Decoded));
  ASSERT_GE(Decoded.size(), 4u);
  ASSERT_TRUE(Decoded[0].Inst.getOperand(0).isImm());
  EXPECT_EQ(Decoded[0].Inst.getOperand(0).getImm(), 2);
  EXPECT_EQ(Decoded[3].Mnemonic, "s_endpgm");
}

TEST(TextDisplacement, PreservesSymbolEndingAtInsertionBoundary) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  llvm::SmallVector<uint8_t> Text;
  Text.append(S.SNopBytes.begin(), S.SNopBytes.end());
  llvm::SmallVector<uint8_t> End = assembleSingleInst("s_endpgm", S);
  ASSERT_EQ(End.size(), MinInstSize);
  Text.append(End.begin(), End.end());

  std::vector<uint8_t> ElfBytes = makeDisplacementTestElf(
      Text, /*AddTextRelocation=*/false, /*AddDebugSection=*/false,
      /*AddBoundaryTextSymbol=*/true);
  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(ElfBytes.data(), ElfBytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());

  DisplacementEdit Edit;
  Edit.Offset = MinInstSize;
  Edit.OriginalSize = 0;
  Edit.ReplacementBytes.assign(S.SNopBytes.begin(), S.SNopBytes.end());

  llvm::Expected<std::unique_ptr<llvm::WritableMemoryBuffer>> OutOrErr =
      tryApplyTextDisplacementToNewBuffer(*ViewOrErr, S, {Edit});
  ASSERT_TRUE((bool)OutOrErr) << llvm::toString(OutOrErr.takeError());
  std::unique_ptr<llvm::WritableMemoryBuffer> Out = std::move(*OutOrErr);

  uint8_t *OutData = reinterpret_cast<uint8_t *>(Out->getBufferStart());
  llvm::Expected<ElfView> OutView =
      ElfView::create(OutData, Out->getBufferSize());
  ASSERT_TRUE((bool)OutView) << llvm::toString(OutView.takeError());

  bool SawBoundarySymbol = false;
  for (const ElfView::ELFT::Shdr &Shdr : OutView->sections()) {
    if (Shdr.sh_type != llvm::ELF::SHT_SYMTAB)
      continue;
    llvm::Expected<ElfView::ELFT::SymRange> Symbols =
        OutView->file().symbols(&Shdr);
    ASSERT_TRUE((bool)Symbols) << llvm::toString(Symbols.takeError());
    for (const ElfView::ELFT::Sym &Sym : *Symbols) {
      if (Sym.st_shndx == OutView->textSectionIndex() &&
          Sym.st_value == OutView->textAddr() && Sym.st_size == MinInstSize)
        SawBoundarySymbol = true;
    }
  }
  EXPECT_TRUE(SawBoundarySymbol);
}

TEST(TextDisplacement, UpdatesKernelDescriptorEntryOffset) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  llvm::SmallVector<uint8_t> Text = assembleSingleInst("s_endpgm", S);
  ASSERT_EQ(Text.size(), MinInstSize);
  std::vector<uint8_t> ElfBytes = makeDisplacementTestElf(Text);
  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(ElfBytes.data(), ElfBytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());

  const ElfView::ELFT::Shdr *OldRodata = nullptr;
  for (const ElfView::ELFT::Shdr &Shdr : ViewOrErr->sections()) {
    llvm::Expected<llvm::StringRef> Name =
        ViewOrErr->file().getSectionName(Shdr);
    ASSERT_TRUE((bool)Name) << llvm::toString(Name.takeError());
    if (*Name == ".rodata")
      OldRodata = &Shdr;
  }
  ASSERT_NE(OldRodata, nullptr);
  const uint64_t OldRodataOffset = OldRodata->sh_offset;

  llvm::Expected<ElfView::ELFT::PhdrRange> OldPhdrs =
      ViewOrErr->file().program_headers();
  ASSERT_TRUE((bool)OldPhdrs) << llvm::toString(OldPhdrs.takeError());
  const ElfView::ELFT::Phdr *OldRodataLoad = nullptr;
  const ElfView::ELFT::Phdr *OldTextLoad = nullptr;
  for (const ElfView::ELFT::Phdr &Phdr : *OldPhdrs) {
    if (Phdr.p_type == llvm::ELF::PT_LOAD && Phdr.p_vaddr == 0x1000)
      OldTextLoad = &Phdr;
    if (Phdr.p_type == llvm::ELF::PT_LOAD && Phdr.p_vaddr == 0x2000)
      OldRodataLoad = &Phdr;
  }
  ASSERT_NE(OldTextLoad, nullptr);
  ASSERT_NE(OldRodataLoad, nullptr);
  const uint64_t OldRodataLoadOffset = OldRodataLoad->p_offset;

  llvm::SmallVector<uint8_t> Prefix =
      assembleInstructions("global_wb\nv_nop", S);
  ASSERT_FALSE(Prefix.empty());

  DisplacementEdit Edit;
  Edit.Offset = 0;
  Edit.OriginalSize = 0;
  Edit.ReplacementBytes.assign(Prefix.begin(), Prefix.end());

  llvm::Expected<std::unique_ptr<llvm::WritableMemoryBuffer>> OutOrErr =
      tryApplyTextDisplacementToNewBuffer(*ViewOrErr, S, {Edit});
  ASSERT_TRUE((bool)OutOrErr) << llvm::toString(OutOrErr.takeError());
  std::unique_ptr<llvm::WritableMemoryBuffer> Out = std::move(*OutOrErr);

  uint8_t *OutData = reinterpret_cast<uint8_t *>(Out->getBufferStart());
  llvm::Expected<ElfView> OutView =
      ElfView::create(OutData, Out->getBufferSize());
  ASSERT_TRUE((bool)OutView) << llvm::toString(OutView.takeError());

  std::vector<KernelDescriptorInfo> KDs = OutView->kernelDescriptors();
  ASSERT_EQ(KDs.size(), 1u);
  EXPECT_EQ(KDs[0].KernelName, "kernel");
  EXPECT_EQ(KDs[0].VAddr, 0x2000u);
  EXPECT_EQ(KDs[0].EntryOffset, static_cast<int64_t>(0x1000 - 0x2000));

  const ElfView::ELFT::Shdr *NewRodata = nullptr;
  for (const ElfView::ELFT::Shdr &Shdr : OutView->sections()) {
    llvm::Expected<llvm::StringRef> Name = OutView->file().getSectionName(Shdr);
    ASSERT_TRUE((bool)Name) << llvm::toString(Name.takeError());
    if (*Name == ".rodata")
      NewRodata = &Shdr;
  }
  ASSERT_NE(NewRodata, nullptr);
  EXPECT_EQ(NewRodata->sh_addr, OldRodata->sh_addr);
  EXPECT_EQ(NewRodata->sh_offset, OldRodataOffset + Prefix.size());

  llvm::Expected<ElfView::ELFT::PhdrRange> NewPhdrs =
      OutView->file().program_headers();
  ASSERT_TRUE((bool)NewPhdrs) << llvm::toString(NewPhdrs.takeError());
  const ElfView::ELFT::Phdr *NewRodataLoad = nullptr;
  const ElfView::ELFT::Phdr *NewTextLoad = nullptr;
  for (const ElfView::ELFT::Phdr &Phdr : *NewPhdrs) {
    if (Phdr.p_type == llvm::ELF::PT_LOAD && Phdr.p_vaddr == 0x1000)
      NewTextLoad = &Phdr;
    if (Phdr.p_type == llvm::ELF::PT_LOAD && Phdr.p_vaddr == 0x2000)
      NewRodataLoad = &Phdr;
  }
  ASSERT_NE(NewTextLoad, nullptr);
  ASSERT_NE(NewRodataLoad, nullptr);
  EXPECT_EQ(NewTextLoad->p_filesz, OldTextLoad->p_filesz + Prefix.size());
  EXPECT_EQ(NewTextLoad->p_memsz, OldTextLoad->p_memsz);
  EXPECT_EQ(NewRodataLoad->p_vaddr, OldRodataLoad->p_vaddr);
  EXPECT_EQ(NewRodataLoad->p_paddr, OldRodataLoad->p_paddr);
  EXPECT_EQ(NewRodataLoad->p_offset, OldRodataLoadOffset + Prefix.size());
  EXPECT_EQ(NewRodataLoad->p_offset % NewRodataLoad->p_align,
            NewRodataLoad->p_vaddr % NewRodataLoad->p_align);
  EXPECT_EQ(llvm::ArrayRef<uint8_t>(OutView->textData(), Prefix.size()),
            llvm::ArrayRef<uint8_t>(Prefix));
}

TEST(TextDisplacement, RejectsPcSensitiveAddressMaterialization) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  llvm::SmallVector<uint8_t> Text =
      assembleSingleInst("s_get_pc_i64 s[8:9]", S);
  ASSERT_FALSE(Text.empty());
  std::vector<uint8_t> ElfBytes = makeDisplacementTestElf(Text);
  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(ElfBytes.data(), ElfBytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());

  DisplacementEdit Edit;
  Edit.Offset = 0;
  Edit.ReplacementBytes.assign(S.SNopBytes.begin(), S.SNopBytes.end());

  llvm::Expected<std::unique_ptr<llvm::WritableMemoryBuffer>> OutOrErr =
      tryApplyTextDisplacementToNewBuffer(*ViewOrErr, S, {Edit});
  ASSERT_FALSE((bool)OutOrErr);
  std::string Reason = llvm::toString(OutOrErr.takeError());
  EXPECT_NE(Reason.find("pc-sensitive"), std::string::npos) << Reason;
}

TEST(TextDisplacement, RejectsLaterFileContentInTextLoadSegment) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  llvm::SmallVector<uint8_t> Text = assembleSingleInst("s_endpgm", S);
  ASSERT_FALSE(Text.empty());
  std::vector<uint8_t> ElfBytes = makeDisplacementTestElf(Text);
  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(ElfBytes.data(), ElfBytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());

  llvm::Expected<ElfView::ELFT::PhdrRange> Phdrs =
      ViewOrErr->file().program_headers();
  ASSERT_TRUE((bool)Phdrs) << llvm::toString(Phdrs.takeError());
  const ElfView::ELFT::Phdr *TextLoad = nullptr;
  for (const ElfView::ELFT::Phdr &Phdr : *Phdrs)
    if (Phdr.p_type == llvm::ELF::PT_LOAD && Phdr.p_vaddr == 0x1000)
      TextLoad = &Phdr;
  ASSERT_NE(TextLoad, nullptr);

  const size_t TextLoadOffset =
      reinterpret_cast<const uint8_t *>(TextLoad) - ElfBytes.data();
  llvm::ELF::Elf64_Phdr RawTextLoad;
  std::memcpy(&RawTextLoad, ElfBytes.data() + TextLoadOffset,
              sizeof(RawTextLoad));
  RawTextLoad.p_filesz += 8;
  RawTextLoad.p_memsz += 8;
  std::memcpy(ElfBytes.data() + TextLoadOffset, &RawTextLoad,
              sizeof(RawTextLoad));

  ViewOrErr = ElfView::create(ElfBytes.data(), ElfBytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());

  DisplacementEdit Edit;
  Edit.Offset = 0;
  Edit.ReplacementBytes.assign(S.SNopBytes.begin(), S.SNopBytes.end());
  llvm::Expected<DisplacementPlan> PlanOrErr =
      DisplacementPlan::create(*ViewOrErr, {Edit});
  EXPECT_FALSE((bool)PlanOrErr);
  EXPECT_NE(
      llvm::toString(PlanOrErr.takeError()).find("last file-backed content"),
      std::string::npos);
}

TEST(TextDisplacement, RejectsDebugSectionsUntilAddressesCanBeRemapped) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  llvm::SmallVector<uint8_t> Text = assembleSingleInst("s_endpgm", S);
  ASSERT_FALSE(Text.empty());
  std::vector<uint8_t> ElfBytes = makeDisplacementTestElf(
      Text, /*AddTextRelocation=*/false, /*AddDebugSection=*/true);
  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(ElfBytes.data(), ElfBytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());

  DisplacementEdit Edit;
  Edit.Offset = 0;
  Edit.ReplacementBytes.assign(S.SNopBytes.begin(), S.SNopBytes.end());

  llvm::Expected<std::unique_ptr<llvm::WritableMemoryBuffer>> OutOrErr =
      tryApplyTextDisplacementToNewBuffer(*ViewOrErr, S, {Edit});
  ASSERT_FALSE((bool)OutOrErr);
  std::string Reason = llvm::toString(OutOrErr.takeError());
  EXPECT_NE(Reason.find(".debug_info"), std::string::npos) << Reason;
}

TEST(KernelEntryTrampoline, ClampsInstPrefSizeAndAvoidsPrefetchGuard) {
  namespace hsa = llvm::amdhsa;

  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  llvm::SmallVector<uint8_t> Text = assembleSingleInst("s_endpgm", S);
  ASSERT_EQ(Text.size(), MinInstSize);

  uint32_t Rsrc3 = 0;
  AMDHSA_BITS_SET(Rsrc3, hsa::COMPUTE_PGM_RSRC3_GFX12_PLUS_INST_PREF_SIZE, 7);
  Rsrc3 |= hsa::COMPUTE_PGM_RSRC3_GFX12_PLUS_GLG_EN;
  AMDHSA_BITS_SET(Rsrc3, hsa::COMPUTE_PGM_RSRC3_GFX125_NAMED_BAR_CNT, 3);
  AMDHSA_BITS_SET(Rsrc3, hsa::COMPUTE_PGM_RSRC3_GFX125_TCP_SPLIT, 5);
  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.ComputePgmRsrc3 = Rsrc3;
  Opts.MetadataSgprCount = 8;
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(Text, Opts);
  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());

  uint8_t *Kd = ViewOrErr->findKernelDescriptor("kernel");
  ASSERT_NE(Kd, nullptr);
  uint32_t Rsrc1Before = 0;
  std::memcpy(&Rsrc1Before,
              Kd + offsetof(hsa::kernel_descriptor_t, compute_pgm_rsrc1),
              sizeof(Rsrc1Before));

  std::vector<Trampoline> Growth;
  std::vector<KernelEntryTrampolineFixup> Fixups;
  std::optional<uint32_t> Count = appendKernelEntryTrampolines(
      *ViewOrErr, S, /*MaxSgprs=*/106, Growth, Fixups);
  ASSERT_TRUE(Count.has_value());
  EXPECT_EQ(*Count, 1u);
  ASSERT_EQ(Fixups.size(), 1u);
  EXPECT_EQ(Fixups[0].InstPrefLines, KernelEntryStubInstPrefLines);

  const uint64_t ExpectedGuard =
      computeKernelEntryPrefetchGuardBytes(KernelEntryStubInstPrefLines);
  EXPECT_EQ(ExpectedGuard, 0u);
  ASSERT_FALSE(Growth.empty());

  // Stubs live in the appended pool at trampolinePoolVAddr(); the first stub's
  // offset is the padding needed to reach a KernelEntryStubStride boundary from
  // the pool base.
  std::optional<uint64_t> PoolVAddrOr = ViewOrErr->trampolinePoolVAddr();
  ASSERT_TRUE(PoolVAddrOr.has_value());
  const uint64_t PoolVAddr = *PoolVAddrOr;
  const uint64_t ExpectedStubOffset =
      ((PoolVAddr + KernelEntryStubStride - 1) & ~(KernelEntryStubStride - 1)) -
      PoolVAddr;
  EXPECT_EQ(Fixups[0].StubTextOffset, ExpectedStubOffset);

  uint64_t GrowthTotal = 0;
  for (const Trampoline &T : Growth)
    GrowthTotal += T.Bytes.size();
  EXPECT_EQ(GrowthTotal,
            ExpectedStubOffset + KernelEntryStubStride + ExpectedGuard);

  std::unique_ptr<llvm::WritableMemoryBuffer> Out =
      ViewOrErr->growWithTrampolines(
          Growth, S.SNopBytes, ExecutablePoolTargetState::Neutral);
  ASSERT_NE(Out, nullptr);

  ASSERT_TRUE(
      rewriteKernelEntryDescriptorOffsets(*Out, PoolVAddr, S.Cpu, Fixups));

  uint8_t *OutData = reinterpret_cast<uint8_t *>(Out->getBufferStart());
  llvm::Expected<ElfView> OutView =
      ElfView::create(OutData, Out->getBufferSize());
  ASSERT_TRUE((bool)OutView) << llvm::toString(OutView.takeError());

  uint8_t *OutKd = OutView->findKernelDescriptor("kernel");
  ASSERT_NE(OutKd, nullptr);
  uint32_t OutRsrc3 = 0;
  std::memcpy(&OutRsrc3,
              OutKd + offsetof(hsa::kernel_descriptor_t, compute_pgm_rsrc3),
              sizeof(OutRsrc3));
  uint32_t ExpectedRsrc3 = Rsrc3;
  AMDHSA_BITS_SET(ExpectedRsrc3,
                  hsa::COMPUTE_PGM_RSRC3_GFX12_PLUS_INST_PREF_SIZE,
                  KernelEntryStubInstPrefLines);
  EXPECT_EQ(OutRsrc3, ExpectedRsrc3);
  EXPECT_EQ(AMDHSA_BITS_GET(OutRsrc3,
                            hsa::COMPUTE_PGM_RSRC3_GFX12_PLUS_INST_PREF_SIZE),
            KernelEntryStubInstPrefLines);
  EXPECT_NE(OutRsrc3 & hsa::COMPUTE_PGM_RSRC3_GFX12_PLUS_GLG_EN, 0u);
  EXPECT_EQ(Fixups[0].RequiredSgprs, 12u);
  uint32_t OutRsrc1 = 0;
  std::memcpy(&OutRsrc1,
              OutKd + offsetof(hsa::kernel_descriptor_t, compute_pgm_rsrc1),
              sizeof(OutRsrc1));
  EXPECT_EQ(OutRsrc1, Rsrc1Before);
  EXPECT_EQ(OutView->getKernelSgprCount("kernel"), Fixups[0].RequiredSgprs);

  llvm::ArrayRef<KernelDescriptorInfo> KDs = OutView->kernelDescriptors();
  ASSERT_EQ(KDs.size(), 1u);
  std::optional<uint64_t> KdVAddr = OutView->getKernelDescriptorVAddr("kernel");
  ASSERT_TRUE(KdVAddr.has_value());
  const uint64_t StubVAddr = PoolVAddr + Fixups[0].StubTextOffset;
  EXPECT_EQ(KDs[0].EntryOffset, static_cast<int64_t>(StubVAddr - *KdVAddr));
}

// rewriteKernelEntryDescriptorOffsets aggregates per-kernel SGPR bumps into a
// single batched metadata update. Drive it with a fixup list covering the
// aggregation cases: a kernel appearing twice (take the max), a kernel that
// skips the reservation, and a kernel with a zero requirement. Only the
// max-aggregated kernel's metadata SGPR count should be raised.
TEST(RewriteKernelEntryDescriptorOffsets, AggregatesSgprBumpsMaxSkipZero) {
  comgr_test::MultiKernelDescriptorElfOptions Opts;
  Opts.Kernels = {
      {"k_max", 0x1000, 0x2000, /*EntryOffset=*/-0x1000,
       /*ComputePgmRsrc3=*/0, /*EmitMetadata=*/true, /*MetadataSgprCount=*/8},
      {"k_skip", 0x1100, 0x2100, /*EntryOffset=*/-0x1000,
       /*ComputePgmRsrc3=*/0, /*EmitMetadata=*/true, /*MetadataSgprCount=*/8},
      {"k_zero", 0x1200, 0x2200, /*EntryOffset=*/-0x1000,
       /*ComputePgmRsrc3=*/0, /*EmitMetadata=*/true, /*MetadataSgprCount=*/8},
  };
  std::vector<uint8_t> Bytes = comgr_test::makeMultiKernelDescriptorElf(Opts);

  std::unique_ptr<llvm::WritableMemoryBuffer> Buf =
      llvm::WritableMemoryBuffer::getNewUninitMemBuffer(Bytes.size());
  ASSERT_NE(Buf, nullptr);
  std::memcpy(Buf->getBufferStart(), Bytes.data(), Bytes.size());

  // Two fixups name k_max with different RequiredSgprs -> aggregate to the max
  // (12). k_skip sets SkipSgprReservation, k_zero has RequiredSgprs == 0; both
  // must leave the metadata count untouched.
  const uint64_t PoolVAddr = 0x4000;
  std::vector<KernelEntryTrampolineFixup> Fixups = {
      {"k_max", /*StubTextOffset=*/0, /*RequiredSgprs=*/10, /*InstPrefLines=*/0,
       /*SkipSgprReservation=*/false},
      {"k_max", /*StubTextOffset=*/KernelEntryStubStride, /*RequiredSgprs=*/12,
       /*InstPrefLines=*/0, /*SkipSgprReservation=*/false},
      {"k_skip", /*StubTextOffset=*/2 * KernelEntryStubStride,
       /*RequiredSgprs=*/20, /*InstPrefLines=*/0, /*SkipSgprReservation=*/true},
      {"k_zero", /*StubTextOffset=*/3 * KernelEntryStubStride,
       /*RequiredSgprs=*/0, /*InstPrefLines=*/0, /*SkipSgprReservation=*/false},
  };

  ASSERT_TRUE(
      rewriteKernelEntryDescriptorOffsets(*Buf, PoolVAddr, "gfx1250", Fixups));

  uint8_t *OutData = reinterpret_cast<uint8_t *>(Buf->getBufferStart());
  llvm::Expected<ElfView> OutView =
      ElfView::create(OutData, Buf->getBufferSize());
  ASSERT_TRUE((bool)OutView) << llvm::toString(OutView.takeError());
  EXPECT_EQ(OutView->getKernelSgprCount("k_max"), 12u);
  EXPECT_EQ(OutView->getKernelSgprCount("k_skip"), 8u);
  EXPECT_EQ(OutView->getKernelSgprCount("k_zero"), 8u);
}

// A fixup naming a kernel with no descriptor must fail the whole rewrite, even
// when another fixup in the batch is valid.
TEST(RewriteKernelEntryDescriptorOffsets, PropagatesMissingDescriptorFailure) {
  comgr_test::MultiKernelDescriptorElfOptions Opts;
  Opts.Kernels = {
      {"present", 0x1000, 0x2000, /*EntryOffset=*/-0x1000,
       /*ComputePgmRsrc3=*/0, /*EmitMetadata=*/true, /*MetadataSgprCount=*/8},
  };
  std::vector<uint8_t> Bytes = comgr_test::makeMultiKernelDescriptorElf(Opts);

  std::unique_ptr<llvm::WritableMemoryBuffer> Buf =
      llvm::WritableMemoryBuffer::getNewUninitMemBuffer(Bytes.size());
  ASSERT_NE(Buf, nullptr);
  std::memcpy(Buf->getBufferStart(), Bytes.data(), Bytes.size());

  std::vector<KernelEntryTrampolineFixup> Fixups = {
      {"present", /*StubTextOffset=*/0, /*RequiredSgprs=*/10,
       /*InstPrefLines=*/0, /*SkipSgprReservation=*/false},
      {"absent", /*StubTextOffset=*/KernelEntryStubStride, /*RequiredSgprs=*/10,
       /*InstPrefLines=*/0, /*SkipSgprReservation=*/false},
  };

  EXPECT_FALSE(rewriteKernelEntryDescriptorOffsets(*Buf, /*PoolVAddr=*/0x4000,
                                                   "gfx1250", Fixups));
}

// Count symbols named \p Name in the .symtab of the ELF held in \p Buf.
// Returns ~0u if the ELF or its symbol table cannot be parsed, so a mis-parse
// surfaces as a failed expectation rather than a silent zero.
static unsigned countSymtabSymbolsNamed(llvm::WritableMemoryBuffer &Buf,
                                        llvm::StringRef Name) {
  using ELFT = llvm::object::ELF64LE;
  llvm::Expected<llvm::object::ELFFile<ELFT>> FileOrErr =
      llvm::object::ELFFile<ELFT>::create(
          llvm::StringRef(reinterpret_cast<const char *>(Buf.getBufferStart()),
                          Buf.getBufferSize()));
  if (!FileOrErr) {
    llvm::consumeError(FileOrErr.takeError());
    return ~0u;
  }
  llvm::object::ELFFile<ELFT> &File = *FileOrErr;
  llvm::Expected<ELFT::ShdrRange> Secs = File.sections();
  if (!Secs) {
    llvm::consumeError(Secs.takeError());
    return ~0u;
  }
  const ELFT::Shdr *Symtab = nullptr;
  for (const ELFT::Shdr &Sh : *Secs)
    if (Sh.sh_type == llvm::ELF::SHT_SYMTAB) {
      Symtab = &Sh;
      break;
    }
  if (!Symtab)
    return 0;
  llvm::Expected<ELFT::SymRange> Syms = File.symbols(Symtab);
  llvm::Expected<llvm::StringRef> Str = File.getStringTableForSymtab(*Symtab);
  if (!Syms || !Str) {
    if (!Syms)
      llvm::consumeError(Syms.takeError());
    if (!Str)
      llvm::consumeError(Str.takeError());
    return ~0u;
  }
  unsigned Count = 0;
  for (const ELFT::Sym &Sym : *Syms) {
    llvm::Expected<llvm::StringRef> N = Sym.getName(*Str);
    if (!N) {
      llvm::consumeError(N.takeError());
      continue;
    }
    if (*N == Name)
      ++Count;
  }
  return Count;
}

// Cross-check that the <kernel>.stub symbol in Buf resolves to exactly what the
// debugger relies on, tying it to independently-produced artifacts rather than
// to the address formula the symbol writer itself uses:
//   (1) it names the address the rewritten kernel descriptor's entry now points
//       at (what amd-dbgapi / rocgdb resolve for the dispatch),
//   (2) real entry-stub bytes live at that address, and
//   (3) its [st_value, st_value + st_size) range lies inside its own section.
static void
expectStubSymbolMatchesDispatchEntry(llvm::WritableMemoryBuffer &Buf,
                                     llvm::StringRef KernelName,
                                     const LLVMState &S) {
  using ELFT = llvm::object::ELF64LE;
  llvm::Expected<llvm::object::ELFFile<ELFT>> FileOrErr =
      llvm::object::ELFFile<ELFT>::create(
          llvm::StringRef(reinterpret_cast<const char *>(Buf.getBufferStart()),
                          Buf.getBufferSize()));
  ASSERT_TRUE((bool)FileOrErr) << llvm::toString(FileOrErr.takeError());
  llvm::object::ELFFile<ELFT> &File = *FileOrErr;
  llvm::Expected<ELFT::ShdrRange> Secs = File.sections();
  ASSERT_TRUE((bool)Secs) << llvm::toString(Secs.takeError());
  const ELFT::Shdr *Symtab = nullptr;
  for (const ELFT::Shdr &Sh : *Secs)
    if (Sh.sh_type == llvm::ELF::SHT_SYMTAB) {
      Symtab = &Sh;
      break;
    }
  ASSERT_NE(Symtab, nullptr);
  llvm::Expected<ELFT::SymRange> Syms = File.symbols(Symtab);
  ASSERT_TRUE((bool)Syms) << llvm::toString(Syms.takeError());
  llvm::Expected<llvm::StringRef> StrTab =
      File.getStringTableForSymtab(*Symtab);
  ASSERT_TRUE((bool)StrTab) << llvm::toString(StrTab.takeError());

  const std::string StubName = (KernelName + ".stub").str();
  const ELFT::Sym *Stub = nullptr;
  for (const ELFT::Sym &Sym : *Syms) {
    llvm::Expected<llvm::StringRef> N = Sym.getName(*StrTab);
    ASSERT_TRUE((bool)N) << llvm::toString(N.takeError());
    if (*N == StubName) {
      Stub = &Sym;
      break;
    }
  }
  ASSERT_NE(Stub, nullptr) << "missing symbol " << StubName;

  // (3) The symbol range lies fully inside its own section.
  ASSERT_LT(Stub->st_shndx, Secs->size());
  const ELFT::Shdr &Sec = (*Secs)[Stub->st_shndx];
  EXPECT_GE(Stub->st_value, Sec.sh_addr);
  EXPECT_LE(Stub->st_value + Stub->st_size, Sec.sh_addr + Sec.sh_size);

  llvm::Expected<ElfView> ViewOrErr = ElfView::create(
      reinterpret_cast<uint8_t *>(Buf.getBufferStart()), Buf.getBufferSize());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());
  ElfView &View = *ViewOrErr;

  // (1) The symbol names exactly the address the descriptor entry now targets.
  const KernelDescriptorInfo *KD = nullptr;
  for (const KernelDescriptorInfo &Info : View.kernelDescriptors())
    if (Info.KernelName == KernelName) {
      KD = &Info;
      break;
    }
  ASSERT_NE(KD, nullptr);
  ASSERT_GE(KD->EntryOffset, 0);
  const uint64_t EntryVAddr =
      KD->VAddr + static_cast<uint64_t>(KD->EntryOffset);
  EXPECT_EQ(Stub->st_value, EntryVAddr)
      << "stub symbol must name the descriptor's entry address";

  // (2) Real entry-stub bytes live at the symbol's address.
  const uint8_t *StubBytes =
      View.dataAtVAddr(Stub->st_value, KernelEntryStubStride);
  ASSERT_NE(StubBytes, nullptr);
  EXPECT_TRUE(isKernelEntryTrampoline(
      llvm::ArrayRef<uint8_t>(StubBytes, KernelEntryStubStride), S));
}

// Covers: the entry-trampoline rewrite is idempotent -- a second pass over an
// already-rewritten code object installs no new stub, and therefore defines no
// duplicate `<kernel>.stub` symbol. This backs the idempotency claim made by
// the change that adds stub symbols.
//
// How: run the full first pass on a synthetic gfx1250 object
// (appendKernelEntryTrampolines -> growWithTrampolines ->
// rewriteKernelEntryDescriptorOffsets -> addKernelEntryTrampolineSymbols) and
// confirm exactly one "kernel.stub" symbol. Then re-parse that output and run
// appendKernelEntryTrampolines again: because the descriptor already targets
// the appended stub, the second pass must report zero new stubs and produce no
// fixups, so the symbol pass never runs. Feeding those empty fixups to
// addKernelEntryTrampolineSymbols returns nullptr (no new buffer), and
// "kernel.stub" remains defined exactly once -- i.e. no duplicate name.
TEST(KernelEntryTrampoline, SecondPassAddsNoDuplicateStubSymbol) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  llvm::SmallVector<uint8_t> Text = assembleSingleInst("s_endpgm", S);
  ASSERT_EQ(Text.size(), MinInstSize);

  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.MetadataSgprCount = 8;
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(Text, Opts);

  // -- First pass: append one stub, grow .text, rewrite the descriptor, and
  //    attach the stub symbol. --
  llvm::Expected<ElfView> View1 =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)View1) << llvm::toString(View1.takeError());

  std::vector<Trampoline> Growth1;
  std::vector<KernelEntryTrampolineFixup> Fixups1;
  std::optional<uint32_t> Count1 = appendKernelEntryTrampolines(
      *View1, S, /*MaxSgprs=*/106, Growth1, Fixups1);
  ASSERT_TRUE(Count1.has_value());
  ASSERT_EQ(*Count1, 1u);
  std::optional<uint64_t> PoolVAddr = View1->trampolinePoolVAddr();
  ASSERT_TRUE(PoolVAddr.has_value());

  std::unique_ptr<llvm::WritableMemoryBuffer> Grown =
      View1->growWithTrampolines(
          Growth1, S.SNopBytes, ExecutablePoolTargetState::Neutral);
  ASSERT_NE(Grown, nullptr);
  ASSERT_TRUE(
      rewriteKernelEntryDescriptorOffsets(*Grown, *PoolVAddr, S.Cpu, Fixups1));
  std::unique_ptr<llvm::WritableMemoryBuffer> Pass1 =
      addKernelEntryTrampolineSymbols(*Grown, *PoolVAddr, Fixups1);
  ASSERT_NE(Pass1, nullptr);
  ASSERT_EQ(countSymtabSymbolsNamed(*Pass1, "kernel.stub"), 1u);
  // The stub symbol must resolve to the dispatch entry, cover real stub bytes,
  // and stay within its section -- not merely match the writer's own formula.
  expectStubSymbolMatchesDispatchEntry(*Pass1, "kernel", S);

  // -- Second pass over the already-rewritten object. --
  uint8_t *Pass1Data = reinterpret_cast<uint8_t *>(Pass1->getBufferStart());
  llvm::Expected<ElfView> View2 =
      ElfView::create(Pass1Data, Pass1->getBufferSize());
  ASSERT_TRUE((bool)View2) << llvm::toString(View2.takeError());

  std::vector<Trampoline> Growth2;
  std::vector<KernelEntryTrampolineFixup> Fixups2;
  std::optional<uint32_t> Count2 = appendKernelEntryTrampolines(
      *View2, S, /*MaxSgprs=*/106, Growth2, Fixups2);
  ASSERT_TRUE(Count2.has_value());
  // The descriptor already targets a stub, so nothing new is installed.
  EXPECT_EQ(*Count2, 0u);
  EXPECT_TRUE(Fixups2.empty());

  // With no fixups the symbol pass is a no-op (returns nullptr, keeping the
  // existing buffer), so no second "kernel.stub" can be defined.
  std::unique_ptr<llvm::WritableMemoryBuffer> Pass2 =
      addKernelEntryTrampolineSymbols(*Pass1, *PoolVAddr, Fixups2);
  EXPECT_EQ(Pass2, nullptr);
  EXPECT_EQ(countSymtabSymbolsNamed(*Pass1, "kernel.stub"), 1u);
}

// A `global_wb; v_nop` prologue (llvm/llvm-project#208467) already satisfies
// the workaround, so no trampoline is installed.
TEST(KernelEntryTrampoline, SkipsWhenPrologueAlreadyHasVmemWorkaround) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  llvm::SmallVector<uint8_t> GlobalWb = assembleSingleInst("global_wb", S);
  llvm::SmallVector<uint8_t> VNop = assembleSingleInst("v_nop", S);
  llvm::SmallVector<uint8_t> EndPgm = assembleSingleInst("s_endpgm", S);
  ASSERT_FALSE(GlobalWb.empty());
  ASSERT_FALSE(VNop.empty());
  ASSERT_EQ(EndPgm.size(), MinInstSize);

  llvm::SmallVector<uint8_t> Text;
  Text.append(GlobalWb.begin(), GlobalWb.end());
  Text.append(VNop.begin(), VNop.end());
  Text.append(EndPgm.begin(), EndPgm.end());

  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(Text);
  llvm::Expected<ElfView> View =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)View) << llvm::toString(View.takeError());

  std::vector<Trampoline> Growth;
  std::vector<KernelEntryTrampolineFixup> Fixups;
  std::optional<uint32_t> Count =
      appendKernelEntryTrampolines(*View, S, /*MaxSgprs=*/106, Growth, Fixups);
  ASSERT_TRUE(Count.has_value());
  EXPECT_EQ(*Count, 0u);
  EXPECT_TRUE(Fixups.empty());
  EXPECT_TRUE(Growth.empty());
}

// The same two instructions in the wrong order are not the workaround, so a
// trampoline is still installed.
TEST(KernelEntryTrampoline, InstallsWhenPrologueLacksVmemWorkaround) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  llvm::SmallVector<uint8_t> VNop = assembleSingleInst("v_nop", S);
  llvm::SmallVector<uint8_t> GlobalWb = assembleSingleInst("global_wb", S);
  llvm::SmallVector<uint8_t> EndPgm = assembleSingleInst("s_endpgm", S);
  ASSERT_FALSE(VNop.empty());
  ASSERT_FALSE(GlobalWb.empty());
  ASSERT_EQ(EndPgm.size(), MinInstSize);

  llvm::SmallVector<uint8_t> Text;
  Text.append(VNop.begin(), VNop.end());
  Text.append(GlobalWb.begin(), GlobalWb.end());
  Text.append(EndPgm.begin(), EndPgm.end());

  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(Text);
  llvm::Expected<ElfView> View =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)View) << llvm::toString(View.takeError());

  std::vector<Trampoline> Growth;
  std::vector<KernelEntryTrampolineFixup> Fixups;
  std::optional<uint32_t> Count =
      appendKernelEntryTrampolines(*View, S, /*MaxSgprs=*/106, Growth, Fixups);
  ASSERT_TRUE(Count.has_value());
  EXPECT_EQ(*Count, 1u);
  EXPECT_EQ(Fixups.size(), 1u);
}

TEST(KernelEntryTrampoline, AlignsStubByVirtualAddress) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  llvm::SmallVector<uint8_t> Text = assembleSingleInst("s_endpgm", S);
  ASSERT_EQ(Text.size(), MinInstSize);

  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.TextAddr = 0x1080;
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(Text, Opts);
  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());

  std::vector<Trampoline> Growth;
  std::vector<KernelEntryTrampolineFixup> Fixups;
  std::optional<uint32_t> Count = appendKernelEntryTrampolines(
      *ViewOrErr, S, /*MaxSgprs=*/106, Growth, Fixups);

  ASSERT_TRUE(Count.has_value());
  EXPECT_EQ(*Count, 1u);
  ASSERT_EQ(Fixups.size(), 1u);
  // The stub is aligned by its virtual address: the pool base plus the stub's
  // offset lands on a KernelEntryStubStride boundary.
  std::optional<uint64_t> PoolVAddrOr = ViewOrErr->trampolinePoolVAddr();
  ASSERT_TRUE(PoolVAddrOr.has_value());
  const uint64_t StubVAddr = *PoolVAddrOr + Fixups[0].StubTextOffset;
  EXPECT_EQ(StubVAddr % KernelEntryStubStride, 0u);
}

TEST(KernelEntryTrampoline, AppendReturnsZeroWhenNoDescriptorsExist) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  llvm::SmallVector<uint8_t> Text = assembleSingleInst("s_endpgm", S);
  ASSERT_EQ(Text.size(), MinInstSize);

  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.EmitKernelDescriptorSymbol = false;
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(Text, Opts);
  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());

  std::vector<Trampoline> Growth;
  std::vector<KernelEntryTrampolineFixup> Fixups;
  std::optional<uint32_t> Count = appendKernelEntryTrampolines(
      *ViewOrErr, S, /*MaxSgprs=*/106, Growth, Fixups);

  ASSERT_TRUE(Count.has_value());
  EXPECT_EQ(*Count, 0u);
  EXPECT_TRUE(Growth.empty());
  EXPECT_TRUE(Fixups.empty());
}

TEST(KernelEntryTrampoline, UsesVccForValidTopMetadataCounts) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  llvm::SmallVector<uint8_t> Text = assembleSingleInst("s_endpgm", S);
  ASSERT_EQ(Text.size(), MinInstSize);

  for (unsigned SgprCount : {105u, 106u, 107u, 108u}) {
    SCOPED_TRACE(SgprCount);
    comgr_test::KernelDescriptorElfOptions Opts;
    Opts.MetadataSgprCount = SgprCount;
    comgr_test::KernelDescriptorElf Obj =
        comgr_test::makeKernelDescriptorElf(Text, Opts);
    llvm::Expected<ElfView> ViewOrErr =
        ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
    ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());

    std::vector<Trampoline> Growth;
    std::vector<KernelEntryTrampolineFixup> Fixups;
    std::optional<uint32_t> Count = appendKernelEntryTrampolines(
        *ViewOrErr, S, /*MaxSgprs=*/Gfx1250MaxNumberedSgprs, Growth, Fixups);

    ASSERT_TRUE(Count.has_value());
    EXPECT_EQ(*Count, 1u);
    ASSERT_EQ(Fixups.size(), 1u);
    EXPECT_EQ(Fixups[0].RequiredSgprs, Gfx1250MaxTotalSgprs);
    EXPECT_FALSE(Fixups[0].SkipSgprReservation);
    llvm::SmallVector<uint8_t> Stub = getAppendedEntryStub(Growth, Fixups[0]);
    ASSERT_EQ(Stub.size(), KernelEntryStubStride);
    expectEntryStubUsesVcc(Stub, S);
  }
}

TEST(KernelEntryTrampoline, TopNumberedPairConservativelyReservesPossibleVcc) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Text = assembleSingleInst("s_endpgm", S);
  ASSERT_EQ(Text.size(), MinInstSize);

  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.MetadataSgprCount = 104;
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(Text, Opts);
  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());

  std::vector<Trampoline> Growth;
  std::vector<KernelEntryTrampolineFixup> Fixups;
  std::optional<uint32_t> Count = appendKernelEntryTrampolines(
      *ViewOrErr, S, /*MaxSgprs=*/Gfx1250MaxNumberedSgprs, Growth, Fixups);
  ASSERT_TRUE(Count.has_value());
  EXPECT_EQ(*Count, 1u);
  ASSERT_EQ(Fixups.size(), 1u);
  EXPECT_EQ(Fixups[0].RequiredSgprs, Gfx1250MaxTotalSgprs);
  llvm::SmallVector<uint8_t> Stub = getAppendedEntryStub(Growth, Fixups[0]);
  ASSERT_EQ(Stub.size(), KernelEntryStubStride);
  expectEntryStubUsesTopNumberedPair(Stub, S);
}

TEST(KernelEntryTrampoline, RejectsMalformedTotalSgprCountTransactionally) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Text = assembleSingleInst("s_endpgm", S);
  ASSERT_EQ(Text.size(), MinInstSize);

  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.MetadataSgprCount = Gfx1250MaxTotalSgprs + 1;
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(Text, Opts);
  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());

  Trampoline Existing;
  Existing.Bytes.assign(S.SNopBytes.begin(), S.SNopBytes.end());
  std::vector<Trampoline> Growth;
  Growth.push_back(Existing);
  std::vector<KernelEntryTrampolineFixup> Fixups;
  EXPECT_FALSE(appendKernelEntryTrampolines(
                   *ViewOrErr, S,
                   /*MaxSgprs=*/Gfx1250MaxNumberedSgprs, Growth, Fixups)
                   .has_value());
  ASSERT_EQ(Growth.size(), 1u);
  EXPECT_EQ(llvm::ArrayRef<uint8_t>(Growth[0].Bytes),
            llvm::ArrayRef<uint8_t>(Existing.Bytes));
  EXPECT_TRUE(Fixups.empty());
}

TEST(TextDisplacement, RejectsTextRelocationSections) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  llvm::SmallVector<uint8_t> Text = assembleSingleInst("s_endpgm", S);
  ASSERT_EQ(Text.size(), MinInstSize);
  std::vector<uint8_t> ElfBytes =
      makeDisplacementTestElf(Text, /*AddTextRelocation=*/true);
  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(ElfBytes.data(), ElfBytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());

  DisplacementEdit Edit;
  Edit.Offset = 0;
  Edit.OriginalSize = 0;
  Edit.ReplacementBytes.assign(S.SNopBytes.begin(), S.SNopBytes.end());

  llvm::Expected<std::unique_ptr<llvm::WritableMemoryBuffer>> OutOrErr =
      tryApplyTextDisplacementToNewBuffer(*ViewOrErr, S, {Edit});
  ASSERT_FALSE((bool)OutOrErr);
  std::string Reason = llvm::toString(OutOrErr.takeError());
  EXPECT_NE(Reason.find("relocation section"), std::string::npos);
}

TEST(TextDisplacement, RejectsDynamicRelocationTargetingText) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  llvm::SmallVector<uint8_t> Text = assembleSingleInst("s_endpgm", S);
  ASSERT_EQ(Text.size(), MinInstSize);
  std::vector<uint8_t> ElfBytes =
      makeDisplacementTestElf(Text, /*AddTextRelocation=*/true);
  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(ElfBytes.data(), ElfBytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());

  const ElfView::ELFT::Shdr *RelaShdr = nullptr;
  for (const ElfView::ELFT::Shdr &Shdr : ViewOrErr->sections())
    if (Shdr.sh_type == llvm::ELF::SHT_RELA)
      RelaShdr = &Shdr;
  ASSERT_NE(RelaShdr, nullptr);

  const size_t ShdrOffset =
      reinterpret_cast<const uint8_t *>(RelaShdr) - ElfBytes.data();
  llvm::ELF::Elf64_Shdr RawShdr;
  std::memcpy(&RawShdr, ElfBytes.data() + ShdrOffset, sizeof(RawShdr));
  RawShdr.sh_info = 0;
  std::memcpy(ElfBytes.data() + ShdrOffset, &RawShdr, sizeof(RawShdr));

  llvm::ELF::Elf64_Rela Rela{};
  Rela.r_offset = ViewOrErr->textAddr();
  std::memcpy(ElfBytes.data() + RawShdr.sh_offset, &Rela, sizeof(Rela));

  llvm::Expected<ElfView> DynamicView =
      ElfView::create(ElfBytes.data(), ElfBytes.size());
  ASSERT_TRUE((bool)DynamicView) << llvm::toString(DynamicView.takeError());

  DisplacementEdit Edit;
  Edit.Offset = 0;
  Edit.OriginalSize = 0;
  Edit.ReplacementBytes.assign(S.SNopBytes.begin(), S.SNopBytes.end());

  llvm::Expected<std::unique_ptr<llvm::WritableMemoryBuffer>> OutOrErr =
      tryApplyTextDisplacementToNewBuffer(*DynamicView, S, {Edit});
  ASSERT_FALSE((bool)OutOrErr);
  std::string Reason = llvm::toString(OutOrErr.takeError());
  EXPECT_NE(Reason.find("dynamic relocation section"), std::string::npos);

  Rela.r_offset = 0x2000;
  std::memcpy(ElfBytes.data() + RawShdr.sh_offset, &Rela, sizeof(Rela));
  llvm::Expected<ElfView> NonTextView =
      ElfView::create(ElfBytes.data(), ElfBytes.size());
  ASSERT_TRUE((bool)NonTextView) << llvm::toString(NonTextView.takeError());
  OutOrErr = tryApplyTextDisplacementToNewBuffer(*NonTextView, S, {Edit});
  EXPECT_TRUE((bool)OutOrErr) << llvm::toString(OutOrErr.takeError());

  Rela.setSymbolAndType(/*Symbol=*/0, llvm::ELF::R_AMDGPU_RELATIVE64);
  Rela.r_addend = NonTextView->textAddr();
  std::memcpy(ElfBytes.data() + RawShdr.sh_offset, &Rela, sizeof(Rela));
  llvm::Expected<ElfView> TextAddendView =
      ElfView::create(ElfBytes.data(), ElfBytes.size());
  ASSERT_TRUE((bool)TextAddendView)
      << llvm::toString(TextAddendView.takeError());
  OutOrErr = tryApplyTextDisplacementToNewBuffer(*TextAddendView, S, {Edit});
  ASSERT_FALSE((bool)OutOrErr);
  Reason = llvm::toString(OutOrErr.takeError());
  EXPECT_NE(Reason.find("addend references"), std::string::npos) << Reason;
}

// -- classifyWmmaNops ---------------------------------------------------------

TEST(ClassifyWmmaNops, CoversKnownMnemonics) {
  struct Case {
    llvm::StringLiteral Mnemonic;
    int A0Nops;
    int B0Nops;
  };
  const Case Cases[] = {
      {"v_add_f32", 4, 4},
      {"v_wmma_i32_16x16x32_iu8", 8, 4},
      {"v_wmma_i32_16x16x64_iu4", 8, 4},
      {"v_wmma_f32_16x16x128_f8f6f4", 1, 4},
      {"v_wmma_f32_16x16x128_fp8_fp8", 3, 4},
      {"v_wmma_f32_16x16x32_fp8_fp8", 1, 4},
      {"v_wmma_f32_16x16x16_f16", 4, 4},
      {"v_wmma_f32_16x16x16_bf16", 4, 4},
      {"v_swmmac_i32_16x16x64_iu8", 8, 4},
      {"v_wmma_f32_16x16x4_f32", 4, 4},
      {"v_wmma_f16_something_iu8", 8, 4},
  };

  for (const Case &C : Cases) {
    WmmaNopReq Req = classifyWmmaNops(C.Mnemonic);
    EXPECT_EQ(Req.A0Nops, C.A0Nops) << C.Mnemonic.str();
    EXPECT_EQ(Req.B0Nops, C.B0Nops) << C.Mnemonic.str();
  }
}

TEST(WmmaHazardDeficit, KeepsStrongestRequirementPerCandidate) {
  llvm::DenseMap<size_t, int> MaxDeficits;
  EXPECT_EQ(updateWmmaHazardDeficit(MaxDeficits, 7, 3), 3);
  EXPECT_EQ(updateWmmaHazardDeficit(MaxDeficits, 7, 8), 8);
  EXPECT_EQ(updateWmmaHazardDeficit(MaxDeficits, 7, 2), 8);
  EXPECT_EQ(updateWmmaHazardDeficit(MaxDeficits, 11, 4), 4);
  EXPECT_EQ(MaxDeficits.lookup(7), 8);
  EXPECT_EQ(MaxDeficits.lookup(11), 4);
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
// Tests for the ds_load_addtid_b32 / ds_store_addtid_b32 gfx1250 trampoline
// patch. Coverage is bottom-up: first that the encode/decode of ADDTID
// instructions exposes the expected MCInst operand layout, then that
// buildTrampoline assembles and decodes a full ADDTID replacement body plus
// its branch-back tail.

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

void expectAddTidLayout(llvm::StringRef Asm, int64_t Offset,
                        llvm::StringRef RegName, const LLVMState &S) {
  llvm::MCInst Inst = decodeOne(Asm, S);
  ASSERT_GE(Inst.getNumOperands(), 3u);

  EXPECT_TRUE(Inst.getOperand(AddtidOpReg).isReg());
  EXPECT_NE(Inst.getOperand(AddtidOpReg).getReg(), 0u);
  EXPECT_TRUE(Inst.getOperand(AddtidOpOffset).isImm());
  EXPECT_EQ(Inst.getOperand(AddtidOpOffset).getImm(), Offset);
  EXPECT_TRUE(Inst.getOperand(AddtidOpGds).isImm());
  EXPECT_EQ(Inst.getOperand(AddtidOpGds).getImm(), 0);

  const char *N = S.MRI->getName(Inst.getOperand(AddtidOpReg).getReg());
  ASSERT_NE(N, nullptr);
  EXPECT_EQ(llvm::StringRef(N).str(), RegName.str());
}

void expectDecodedMnemonics(llvm::ArrayRef<InternalDecodedInst> Decoded,
                            llvm::ArrayRef<llvm::StringRef> Expected) {
  ASSERT_EQ(Decoded.size(), Expected.size());
  for (size_t I = 0; I < Expected.size(); ++I)
    EXPECT_EQ(Decoded[I].Mnemonic, Expected[I].str()) << "index " << I;
}

void expectDecodedBodyMatchesAsm(llvm::ArrayRef<InternalDecodedInst> Decoded,
                                 llvm::ArrayRef<std::string> AsmLines,
                                 const LLVMState &S) {
  ASSERT_GE(Decoded.size(), AsmLines.size());
  for (size_t I = 0; I < AsmLines.size(); ++I) {
    llvm::MCInst Expected = decodeOne(AsmLines[I], S);
    expectSameOperands(Decoded[I].Inst, Expected, AsmLines[I]);
  }
}

} // namespace

TEST(AddTid, AddTidDecodesWithExpectedLayout) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  // Direct operand access: register, then offset, then gds bit. No
  // print-and-parse round-trip -- production code uses the same operand
  // indices to reach the destination VGPR.
  // Production code uses MRI.getName() to resolve the VGPR identifier
  // ("VGPR5" for v5, etc.); pin that so a tablegen rename catches here.
  expectAddTidLayout("ds_load_addtid_b32 v5 offset:128", 128, "VGPR5", S);
  expectAddTidLayout("ds_store_addtid_b32 v10 offset:256", 256, "VGPR10", S);
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
  const llvm::StringRef Expected[] = {"v_mbcnt_lo_u32_b32",
                                      "v_mbcnt_hi_u32_b32",
                                      "v_lshlrev_b32",
                                      "v_add_nc_u32",
                                      "v_and_b32",
                                      "ds_load_b32",
                                      "s_branch"};
  expectDecodedMnemonics(Decoded, Expected);
  expectDecodedBodyMatchesAsm(Decoded, AsmLines, S);
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
  const llvm::StringRef Expected[] = {"v_mbcnt_lo_u32_b32",
                                      "v_mbcnt_hi_u32_b32",
                                      "v_lshlrev_b32",
                                      "v_add_nc_u32",
                                      "v_and_b32",
                                      "ds_store_b32",
                                      "s_branch"};
  expectDecodedMnemonics(Decoded, Expected);
  expectDecodedBodyMatchesAsm(Decoded, AsmLines, S);
}

// -- tensor external control-flow provenance --------------------------------

static InternalDecodedInst decodeTensorTestInstruction(
    llvm::ArrayRef<uint8_t> Bytes, uint64_t Offset, const LLVMState &S) {
  std::vector<InternalDecodedInst> Decoded;
  EXPECT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  EXPECT_EQ(Decoded.size(), 1u);
  if (Decoded.empty())
    return {};
  Decoded.front().Offset = Offset;
  return std::move(Decoded.front());
}

static InternalDecodedInst decodeTensorTestInstruction(
    llvm::StringRef Assembly, uint64_t Offset, const LLVMState &S) {
  llvm::SmallVector<uint8_t> Bytes = assembleSingleInst(Assembly, S);
  EXPECT_FALSE(Bytes.empty()) << Assembly.str();
  return decodeTensorTestInstruction(Bytes, Offset, S);
}

static std::vector<InternalDecodedInst>
decodeTensorTestBlock(llvm::ArrayRef<uint8_t> Bytes, uint64_t Offset,
                      const LLVMState &S) {
  std::vector<InternalDecodedInst> Decoded;
  EXPECT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  for (InternalDecodedInst &DI : Decoded)
    DI.Offset += Offset;
  return Decoded;
}

static std::vector<InternalDecodedInst>
buildTensorProvenanceBody(const LLVMState &S,
                          std::optional<uint64_t> ExternalTarget =
                              std::nullopt,
                          bool SplitRelay = false) {
  std::vector<InternalDecodedInst> Result;
  uint64_t Offset = 0;
  auto Add = [&](llvm::StringRef Assembly) {
    InternalDecodedInst DI =
        decodeTensorTestInstruction(Assembly, Offset, S);
    Offset += DI.Size;
    Result.push_back(std::move(DI));
  };
  Add("v_mov_b32 v0, 0");
  Add("v_readfirstlane_b32 s4, v0");
  if (SplitRelay)
    Add("s_get_pc_i64 s[0:1]");
  if (ExternalTarget) {
    llvm::SmallVector<uint8_t> Branch = S.encodeSBranch(Offset, *ExternalTarget);
    InternalDecodedInst DI =
        decodeTensorTestInstruction(Branch, Offset, S);
    Offset += DI.Size;
    Result.push_back(std::move(DI));
  }
  Add("tensor_load_to_lds s[24:27], s[4:11]");
  Add("s_endpgm");
  return Result;
}

static TensorDescriptorMustAnalysis runTensorProvenanceAnalysis(
    llvm::ArrayRef<InternalDecodedInst> Body,
    llvm::ArrayRef<InternalDecodedInst> External,
    llvm::ArrayRef<TensorDispatchStub> DispatchStubs,
    llvm::ArrayRef<uint64_t> DirectTargetOffsets,
    llvm::ArrayRef<uint64_t> ForeignExternalEntries, const LLVMState &S,
    llvm::ArrayRef<uint64_t> VirtualExternalEntries = {},
    llvm::ArrayRef<std::pair<uint64_t, uint64_t>> OriginalControlFlowEdges = {},
    llvm::ArrayRef<uint64_t> OriginalCodeEntries = {},
    std::optional<uint64_t> RangeEnd = std::nullopt) {
  std::vector<InternalDecodedInst> All(Body.begin(), Body.end());
  All.insert(All.end(), External.begin(), External.end());
  llvm::sort(All, [](const InternalDecodedInst &L,
                     const InternalDecodedInst &R) {
    return L.Offset < R.Offset;
  });
  const InternalDecodedInst &Last = Body.back();
  TensorAnalysisRange Range{0, RangeEnd.value_or(Last.Offset + Last.Size)};
  Range.ForeignExternalEntries.assign(ForeignExternalEntries.begin(),
                                      ForeignExternalEntries.end());
  Range.VirtualExternalEntries.assign(VirtualExternalEntries.begin(),
                                      VirtualExternalEntries.end());
  Range.OriginalControlFlowEdges.assign(OriginalControlFlowEdges.begin(),
                                        OriginalControlFlowEdges.end());
  Range.OriginalCodeEntries.assign(OriginalCodeEntries.begin(),
                                   OriginalCodeEntries.end());
  Range.DispatchStubs.assign(DispatchStubs.begin(), DispatchStubs.end());
  llvm::DenseSet<uint64_t> DirectTargets;
  for (uint64_t Target : DirectTargetOffsets)
    DirectTargets.insert(Target);
  return computeTensorDescriptorMustAnalysis(
      Body, All, llvm::ArrayRef<TensorAnalysisRange>(Range), S, DirectTargets,
      /*MaxSgprs=*/106, /*MaxVgprs=*/1024);
}

static size_t tensorInstructionIndex(
    llvm::ArrayRef<InternalDecodedInst> Body) {
  auto It = llvm::find_if(Body, [](const InternalDecodedInst &DI) {
    return DI.Mnemonic == "tensor_load_to_lds";
  });
  EXPECT_NE(It, Body.end());
  return It == Body.end() ? 0 : It - Body.begin();
}

class TensorExternalProvenance : public ::testing::Test {
protected:
  void SetUp() override { ASSERT_TRUE(S.Valid); }

  void expectExternalInstructionRejected(llvm::StringRef Assembly) {
    SCOPED_TRACE(Assembly.str());
    std::vector<InternalDecodedInst> Body = buildTensorProvenanceBody(S);
    std::vector<InternalDecodedInst> External{
        decodeTensorTestInstruction(Assembly, 0x1000, S)};
    TensorDescriptorMustAnalysis Analysis =
        runTensorProvenanceAnalysis(Body, External, {}, {}, {}, S);
    EXPECT_FALSE(Analysis.Low16KnownZero.test(tensorInstructionIndex(Body)));
  }

  LLVMState S = initLLVM(makeGfx1250Ident());
};

TEST_F(TensorExternalProvenance, ExactDispatchStubIsOnlyEntryException) {
  std::vector<InternalDecodedInst> Body = buildTensorProvenanceBody(S);
  TensorDescriptorMustAnalysis Basic =
      runTensorProvenanceAnalysis(Body, {}, {}, {}, {}, S);
  ASSERT_TRUE(Basic.Low16KnownZero.test(tensorInstructionIndex(Body)));

  constexpr uint64_t StubBegin = 0x1000;
  llvm::SmallVector<uint8_t> StubBytes =
      buildKernelEntryTrampoline(StubBegin, 0, /*ScratchSgpr=*/32, S);
  ASSERT_EQ(StubBytes.size(), KernelEntryStubStride);
  std::optional<KernelEntryTrampolineInfo> Info =
      getKernelEntryTrampolineInfo(StubBytes, StubBegin, S);
  ASSERT_TRUE(Info);
  std::vector<InternalDecodedInst> Stub;
  ASSERT_TRUE(decodeTextSection(StubBytes.data(), StubBytes.size(), S, Stub));
  for (InternalDecodedInst &DI : Stub)
    DI.Offset += StubBegin;
  TensorDispatchStub Dispatch{StubBegin, StubBegin + KernelEntryStubStride,
                              Info->TerminalVAddr, 0};

  TensorDescriptorMustAnalysis Analysis = runTensorProvenanceAnalysis(
      Body, Stub, llvm::ArrayRef<TensorDispatchStub>(Dispatch), {}, {}, S);
  EXPECT_TRUE(Analysis.Low16KnownZero.test(tensorInstructionIndex(Body)));

  constexpr uint64_t ForeignSource = 0x1200;
  const uint64_t StubInterior = Stub[2].Offset;
  TensorDescriptorMustAnalysis InteriorRoot = runTensorProvenanceAnalysis(
      Body, Stub, llvm::ArrayRef<TensorDispatchStub>(Dispatch), {}, {}, S,
      {StubInterior});
  EXPECT_FALSE(
      InteriorRoot.Low16KnownZero.test(tensorInstructionIndex(Body)));

  std::vector<InternalDecodedInst> BeginIngress = Stub;
  BeginIngress.push_back(decodeTensorTestInstruction(
      S.encodeSBranch(ForeignSource, StubBegin), ForeignSource, S));
  TensorDescriptorMustAnalysis ForeignBegin = runTensorProvenanceAnalysis(
      Body, BeginIngress, llvm::ArrayRef<TensorDispatchStub>(Dispatch),
      {StubBegin}, {}, S);
  EXPECT_FALSE(
      ForeignBegin.Low16KnownZero.test(tensorInstructionIndex(Body)));

  Stub.push_back(decodeTensorTestInstruction(
      S.encodeSBranch(ForeignSource, StubInterior), ForeignSource, S));
  TensorDescriptorMustAnalysis ForeignIngress = runTensorProvenanceAnalysis(
      Body, Stub, llvm::ArrayRef<TensorDispatchStub>(Dispatch), {StubInterior},
      {}, S);
  EXPECT_FALSE(
      ForeignIngress.Low16KnownZero.test(tensorInstructionIndex(Body)));

  StubBytes.back() ^= 1;
  EXPECT_FALSE(getKernelEntryTrampolineInfo(StubBytes, StubBegin, S));
}

TEST_F(TensorExternalProvenance, NonDispatchCarryReturnToEntryRejected) {
  std::vector<InternalDecodedInst> Body = buildTensorProvenanceBody(S);
  constexpr uint64_t ExternalBegin = 0x1000;
  llvm::SmallVector<uint8_t> Bytes =
      buildKernelEntryTrampoline(ExternalBegin, 0, /*ScratchSgpr=*/32, S);
  ASSERT_EQ(Bytes.size(), KernelEntryStubStride);
  std::vector<InternalDecodedInst> External;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, External));
  for (InternalDecodedInst &DI : External)
    DI.Offset += ExternalBegin;

  TensorDescriptorMustAnalysis Analysis =
      runTensorProvenanceAnalysis(Body, External, {}, {}, {}, S);
  EXPECT_FALSE(Analysis.Low16KnownZero.test(tensorInstructionIndex(Body)));
}

TEST_F(TensorExternalProvenance, UnresolvedExternalIndirectRejected) {
  expectExternalInstructionRejected("s_set_pc_i64 s[0:1]");
}

TEST_F(TensorExternalProvenance, ForeignAppendedPredecessorRejected) {
  constexpr uint64_t PathBegin = 0x100;
  constexpr uint64_t ForeignBegin = 0x200;
  std::vector<InternalDecodedInst> Body =
      buildTensorProvenanceBody(S, PathBegin);
  const uint64_t Resume = Body[tensorInstructionIndex(Body)].Offset;
  std::vector<InternalDecodedInst> Path{
      decodeTensorTestInstruction(S.encodeSBranch(PathBegin, Resume),
                                  PathBegin, S)};
  TensorDescriptorMustAnalysis Positive = runTensorProvenanceAnalysis(
      Body, Path, {}, {PathBegin, Resume}, {}, S);
  EXPECT_TRUE(Positive.Low16KnownZero.test(tensorInstructionIndex(Body)));

  Path.push_back(decodeTensorTestInstruction(
      S.encodeSBranch(ForeignBegin, PathBegin), ForeignBegin, S));
  TensorDescriptorMustAnalysis Negative = runTensorProvenanceAnalysis(
      Body, Path, {}, {PathBegin, Resume}, {}, S);
  EXPECT_FALSE(Negative.Low16KnownZero.test(tensorInstructionIndex(Body)));
}

TEST_F(TensorExternalProvenance, VirtualKernelRootIntoModeledPathRejected) {
  constexpr uint64_t PathBegin = 0x100;
  std::vector<InternalDecodedInst> Body =
      buildTensorProvenanceBody(S, PathBegin);
  const uint64_t Resume = Body[tensorInstructionIndex(Body)].Offset;
  std::vector<InternalDecodedInst> Path{
      decodeTensorTestInstruction(S.encodeSBranch(PathBegin, Resume),
                                  PathBegin, S)};
  TensorDescriptorMustAnalysis Analysis = runTensorProvenanceAnalysis(
      Body, Path, {}, {PathBegin, Resume}, {}, S, {PathBegin});
  EXPECT_FALSE(Analysis.Low16KnownZero.test(tensorInstructionIndex(Body)));
}

TEST_F(TensorExternalProvenance, ForeignRelayTailEntryRejected) {
  constexpr uint64_t RelayBegin = 0x100;
  constexpr uint64_t PathBegin = 0x120;
  std::vector<InternalDecodedInst> Body =
      buildTensorProvenanceBody(S, RelayBegin, /*SplitRelay=*/true);
  const size_t GetPcIndex = 2;
  const uint64_t GetPcOffset = Body[GetPcIndex].Offset;
  const uint64_t Resume = Body[tensorInstructionIndex(Body)].Offset;

  llvm::SmallVector<uint8_t> LongBranch = encodeSetPCLongBranch(
      S, GetPcOffset, PathBegin, /*SgprBase=*/0);
  std::vector<InternalDecodedInst> LongDecoded;
  ASSERT_TRUE(decodeTextSection(LongBranch.data(), LongBranch.size(), S,
                                LongDecoded));
  ASSERT_EQ(LongDecoded.size(), 3u);
  std::vector<InternalDecodedInst> External;
  LongDecoded[1].Offset = RelayBegin;
  LongDecoded[2].Offset = RelayBegin + LongDecoded[1].Size;
  External.push_back(LongDecoded[1]);
  External.push_back(LongDecoded[2]);
  External.push_back(decodeTensorTestInstruction(
      S.encodeSBranch(PathBegin, Resume), PathBegin, S));

  TensorDescriptorMustAnalysis Positive = runTensorProvenanceAnalysis(
      Body, External, {}, {RelayBegin, Resume}, {}, S);
  EXPECT_TRUE(Positive.Low16KnownZero.test(tensorInstructionIndex(Body)));
  TensorDescriptorMustAnalysis Negative = runTensorProvenanceAnalysis(
      Body, External, {}, {RelayBegin, Resume}, {RelayBegin}, S);
  EXPECT_FALSE(Negative.Low16KnownZero.test(tensorInstructionIndex(Body)));

  const uint64_t RelayBranch = Body[GetPcIndex + 1].Offset;
  const std::pair<uint64_t, uint64_t> ExactRelayEdge{RelayBranch, RelayBegin};
  TensorDescriptorMustAnalysis ExactRelay = runTensorProvenanceAnalysis(
      Body, External, {}, {RelayBegin, Resume}, {}, S, {}, {ExactRelayEdge});
  EXPECT_TRUE(ExactRelay.Low16KnownZero.test(tensorInstructionIndex(Body)));

  const std::pair<uint64_t, uint64_t> WrongRelayEdge{Body.front().Offset,
                                                     RelayBegin};
  TensorDescriptorMustAnalysis WrongRelay = runTensorProvenanceAnalysis(
      Body, External, {}, {RelayBegin, Resume}, {}, S, {}, {WrongRelayEdge});
  EXPECT_FALSE(WrongRelay.Low16KnownZero.test(tensorInstructionIndex(Body)));

  TensorDescriptorMustAnalysis RelayBranchRoot = runTensorProvenanceAnalysis(
      Body, External, {}, {RelayBegin, Resume}, {}, S, {}, {}, {RelayBranch});
  EXPECT_FALSE(
      RelayBranchRoot.Low16KnownZero.test(tensorInstructionIndex(Body)));

  TensorDescriptorMustAnalysis RelayTailRoot = runTensorProvenanceAnalysis(
      Body, External, {}, {RelayBegin, Resume}, {}, S, {RelayBegin});
  EXPECT_FALSE(
      RelayTailRoot.Low16KnownZero.test(tensorInstructionIndex(Body)));
}

TEST_F(TensorExternalProvenance, SplitRelayDirectReentryKeepsTensorPredecessor) {
  constexpr uint64_t RelayBegin = 0x100;
  std::vector<InternalDecodedInst> Body;
  uint64_t Offset = 0;
  auto Add = [&](llvm::StringRef Assembly) {
    const size_t Index = Body.size();
    InternalDecodedInst DI = decodeTensorTestInstruction(Assembly, Offset, S);
    Offset += DI.Size;
    Body.push_back(std::move(DI));
    return Index;
  };

  const size_t ConditionalIndex = Add("s_cbranch_scc1 0");
  Add("v_mov_b32 v0, 0");
  Add("v_readfirstlane_b32 s4, v0");
  const size_t NormalBranchIndex = Add("s_branch 0");
  const uint64_t RelayGetPc = Offset;
  Add("s_get_pc_i64 s[0:1]");
  const size_t RelayBranchIndex = Add("s_branch 0");
  const uint64_t TensorOffset = Offset;
  Add("tensor_load_to_lds s[24:27], s[4:11]");
  const size_t PostTensorBranchIndex = Add("s_branch 0");
  Add("s_nop 0");
  const uint64_t JoinOffset = Offset;
  Add("s_endpgm");

  const InternalDecodedInst &Conditional = Body[ConditionalIndex];
  ASSERT_EQ((RelayGetPc - Conditional.Offset - Conditional.Size) %
                MinInstSize,
            0u);
  const uint64_t ConditionalDelta =
      (RelayGetPc - Conditional.Offset - Conditional.Size) / MinInstSize;
  Body[ConditionalIndex] = decodeTensorTestInstruction(
      "s_cbranch_scc1 " + std::to_string(ConditionalDelta),
      Conditional.Offset, S);
  Body[NormalBranchIndex] = decodeTensorTestInstruction(
      S.encodeSBranch(Body[NormalBranchIndex].Offset, TensorOffset),
      Body[NormalBranchIndex].Offset, S);
  Body[RelayBranchIndex] = decodeTensorTestInstruction(
      S.encodeSBranch(Body[RelayBranchIndex].Offset, RelayBegin),
      Body[RelayBranchIndex].Offset, S);
  Body[PostTensorBranchIndex] = decodeTensorTestInstruction(
      S.encodeSBranch(Body[PostTensorBranchIndex].Offset, JoinOffset),
      Body[PostTensorBranchIndex].Offset, S);

  llvm::SmallVector<uint8_t> LongBranch = encodeSetPCLongBranch(
      S, RelayGetPc, TensorOffset, /*SgprBase=*/0);
  std::vector<InternalDecodedInst> LongDecoded =
      decodeTensorTestBlock(LongBranch, 0, S);
  ASSERT_EQ(LongDecoded.size(), 3u);
  LongDecoded[1].Offset = RelayBegin;
  LongDecoded[2].Offset = RelayBegin + LongDecoded[1].Size;
  std::vector<InternalDecodedInst> External{LongDecoded[1], LongDecoded[2]};

  const std::pair<uint64_t, uint64_t> RelayEdge{
      Body[RelayBranchIndex].Offset, RelayBegin};
  TensorDescriptorMustAnalysis Analysis = runTensorProvenanceAnalysis(
      Body, External, {}, {RelayGetPc, RelayBegin, TensorOffset, JoinOffset},
      {}, S, {}, {RelayEdge}, {0});
  EXPECT_FALSE(Analysis.Low16KnownZero.test(tensorInstructionIndex(Body)));
}

TEST_F(TensorExternalProvenance, CallableOriginalHelperPathRejected) {
  constexpr uint64_t HelperBegin = 0x80;
  constexpr uint64_t ExternalBegin = 0x100;
  std::vector<InternalDecodedInst> Body;
  uint64_t Offset = 0;
  auto Add = [&](llvm::StringRef Assembly) {
    const size_t Index = Body.size();
    InternalDecodedInst DI = decodeTensorTestInstruction(Assembly, Offset, S);
    Offset += DI.Size;
    Body.push_back(std::move(DI));
    return Index;
  };

  Add("v_mov_b32 v0, 0");
  Add("v_readfirstlane_b32 s4, v0");
  const size_t ConditionalIndex = Add("s_cbranch_scc1 0");
  const size_t HelperBranchIndex = Add("s_branch 0");
  const uint64_t TensorOffset = Offset;
  Add("tensor_load_to_lds s[24:27], s[4:11]");
  Add("s_endpgm");
  const uint64_t CandidateEnd = Offset;

  const uint64_t ConditionalOffset = Body[ConditionalIndex].Offset;
  const uint64_t ConditionalSize = Body[ConditionalIndex].Size;
  ASSERT_GT(TensorOffset, ConditionalOffset + ConditionalSize);
  ASSERT_EQ((TensorOffset - ConditionalOffset - ConditionalSize) % MinInstSize,
            0u);
  const uint64_t ConditionalDelta =
      (TensorOffset - ConditionalOffset - ConditionalSize) / MinInstSize;
  Body[ConditionalIndex] = decodeTensorTestInstruction(
      "s_cbranch_scc1 " + std::to_string(ConditionalDelta),
      ConditionalOffset, S);
  Body[HelperBranchIndex] = decodeTensorTestInstruction(
      S.encodeSBranch(Body[HelperBranchIndex].Offset, HelperBegin),
      Body[HelperBranchIndex].Offset, S);
  Body.push_back(decodeTensorTestInstruction(
      S.encodeSBranch(HelperBegin, ExternalBegin), HelperBegin, S));

  llvm::SmallVector<uint8_t> ExternalBytes = encodeSetPCLongBranch(
      S, ExternalBegin, TensorOffset, /*SgprBase=*/0);
  std::vector<InternalDecodedInst> External =
      decodeTensorTestBlock(ExternalBytes, ExternalBegin, S);
  ASSERT_EQ(External.size(), 3u);

  const std::pair<uint64_t, uint64_t> HelperEdge{HelperBegin, ExternalBegin};
  TensorDescriptorMustAnalysis Analysis = runTensorProvenanceAnalysis(
      Body, External, {}, {HelperBegin, ExternalBegin, TensorOffset}, {}, S,
      {}, {HelperEdge}, {0, HelperBegin}, CandidateEnd);
  EXPECT_FALSE(Analysis.Low16KnownZero.test(tensorInstructionIndex(Body)));
}

TEST_F(TensorExternalProvenance, ComputedPcInteriorIngressRejected) {
  constexpr uint64_t SequenceBegin = 0x100;
  std::vector<InternalDecodedInst> Body =
      buildTensorProvenanceBody(S, SequenceBegin);
  const uint64_t Resume = Body[tensorInstructionIndex(Body)].Offset;
  llvm::SmallVector<uint8_t> SequenceBytes =
      encodeSetPCLongBranch(S, SequenceBegin, Resume, /*SgprBase=*/0);
  ASSERT_FALSE(SequenceBytes.empty());
  std::vector<InternalDecodedInst> External =
      decodeTensorTestBlock(SequenceBytes, SequenceBegin, S);
  ASSERT_EQ(External.size(), 3u);

  TensorDescriptorMustAnalysis Positive = runTensorProvenanceAnalysis(
      Body, External, {}, {SequenceBegin, Resume}, {}, S);
  EXPECT_TRUE(Positive.Low16KnownZero.test(tensorInstructionIndex(Body)));

  constexpr uint64_t ForeignSource = 0x200;
  const uint64_t AddInterior = External[1].Offset;
  External.push_back(decodeTensorTestInstruction(
      S.encodeSBranch(ForeignSource, AddInterior), ForeignSource, S));
  TensorDescriptorMustAnalysis ForeignEdge = runTensorProvenanceAnalysis(
      Body, External, {}, {SequenceBegin, Resume}, {}, S);
  EXPECT_FALSE(
      ForeignEdge.Low16KnownZero.test(tensorInstructionIndex(Body)));

  External.pop_back();
  TensorDescriptorMustAnalysis VirtualRoot = runTensorProvenanceAnalysis(
      Body, External, {}, {SequenceBegin, Resume}, {}, S, {AddInterior});
  EXPECT_FALSE(
      VirtualRoot.Low16KnownZero.test(tensorInstructionIndex(Body)));
}

TEST_F(TensorExternalProvenance, OriginalCodeRootMustEnterComputedPcPrefix) {
  llvm::SmallVector<uint8_t> Probe =
      encodeSetPCLongBranch(S, 0, 0, /*SgprBase=*/0);
  ASSERT_FALSE(Probe.empty());
  const uint64_t Resume = Probe.size();
  llvm::SmallVector<uint8_t> SequenceBytes =
      encodeSetPCLongBranch(S, 0, Resume, /*SgprBase=*/0);
  ASSERT_EQ(SequenceBytes.size(), Resume);
  std::vector<InternalDecodedInst> Body =
      decodeTensorTestBlock(SequenceBytes, 0, S);
  ASSERT_EQ(Body.size(), 3u);
  uint64_t Offset = Resume;
  auto Add = [&](llvm::StringRef Assembly) {
    InternalDecodedInst DI = decodeTensorTestInstruction(Assembly, Offset, S);
    Offset += DI.Size;
    Body.push_back(std::move(DI));
  };
  Add("v_mov_b32 v0, 0");
  Add("v_readfirstlane_b32 s4, v0");
  Add("tensor_load_to_lds s[24:27], s[4:11]");
  Add("s_endpgm");

  TensorDescriptorMustAnalysis PrefixRoot = runTensorProvenanceAnalysis(
      Body, {}, {}, {Resume}, {}, S, {}, {}, {0});
  EXPECT_TRUE(PrefixRoot.Low16KnownZero.test(tensorInstructionIndex(Body)));

  const uint64_t AddInterior = Body[1].Offset;
  TensorDescriptorMustAnalysis InteriorRoot = runTensorProvenanceAnalysis(
      Body, {}, {}, {Resume}, {}, S, {}, {}, {AddInterior});
  EXPECT_FALSE(
      InteriorRoot.Low16KnownZero.test(tensorInstructionIndex(Body)));
}

TEST_F(TensorExternalProvenance, ExternalTargetMustBeDecodedBoundary) {
  std::vector<InternalDecodedInst> Body = buildTensorProvenanceBody(S);
  constexpr uint64_t Source = 0x100;
  constexpr uint64_t MissingTarget = 0x200;
  std::vector<InternalDecodedInst> External{decodeTensorTestInstruction(
      S.encodeSBranch(Source, MissingTarget), Source, S)};
  TensorDescriptorMustAnalysis Missing =
      runTensorProvenanceAnalysis(Body, External, {}, {}, {}, S);
  EXPECT_FALSE(Missing.Low16KnownZero.test(tensorInstructionIndex(Body)));

  InternalDecodedInst Literal = decodeTensorTestInstruction(
      "s_mov_b32 s0, 0x12345678", MissingTarget, S);
  ASSERT_GT(Literal.Size, MinInstSize);
  External.push_back(Literal);
  External.push_back(decodeTensorTestInstruction(
      "s_endpgm", MissingTarget + Literal.Size, S));
  External.front() = decodeTensorTestInstruction(
      S.encodeSBranch(Source, MissingTarget + MinInstSize), Source, S);
  TensorDescriptorMustAnalysis Interior =
      runTensorProvenanceAnalysis(Body, External, {}, {}, {}, S);
  EXPECT_FALSE(Interior.Low16KnownZero.test(tensorInstructionIndex(Body)));
}

TEST_F(TensorExternalProvenance, OutOfRangeFallthroughRejected) {
  std::vector<InternalDecodedInst> Body = buildTensorProvenanceBody(S);
  const uint64_t LastOffset = Body.back().Offset;
  Body.back() = decodeTensorTestInstruction("s_nop 0", LastOffset, S);
  TensorDescriptorMustAnalysis Analysis =
      runTensorProvenanceAnalysis(Body, {}, {}, {}, {}, S);
  EXPECT_FALSE(Analysis.Low16KnownZero.test(tensorInstructionIndex(Body)));
}

TEST_F(TensorExternalProvenance, ExternalBoundaryFallthroughRejected) {
  expectExternalInstructionRejected("s_nop 0");
}

TEST_F(TensorExternalProvenance, ExternalTrapRejected) {
  expectExternalInstructionRejected("s_trap 0");
}

TEST_F(TensorExternalProvenance, ExternalRfeRejected) {
  expectExternalInstructionRejected("s_rfe_i64 s[0:1]");
}

TEST_F(TensorExternalProvenance, CandidateRfeRejected) {
  std::vector<InternalDecodedInst> Body = buildTensorProvenanceBody(S);
  Body.back() =
      decodeTensorTestInstruction("s_rfe_i64 s[0:1]", Body.back().Offset, S);
  TensorDescriptorMustAnalysis Analysis =
      runTensorProvenanceAnalysis(Body, {}, {}, {}, {}, S);
  EXPECT_FALSE(Analysis.Low16KnownZero.test(tensorInstructionIndex(Body)));
}

TEST_F(TensorExternalProvenance, ForeignIngressIntoAnyDispatchStubRejected) {
  std::vector<InternalDecodedInst> Body = buildTensorProvenanceBody(S);
  constexpr uint64_t StubBegin = 0x1000;
  constexpr uint64_t UnrelatedTarget = 0x300;
  llvm::SmallVector<uint8_t> StubBytes = buildKernelEntryTrampoline(
      StubBegin, UnrelatedTarget, /*ScratchSgpr=*/32, S);
  ASSERT_EQ(StubBytes.size(), KernelEntryStubStride);
  std::optional<KernelEntryTrampolineInfo> Info =
      getKernelEntryTrampolineInfo(StubBytes, StubBegin, S);
  ASSERT_TRUE(Info);
  std::vector<InternalDecodedInst> Stub;
  ASSERT_TRUE(decodeTextSection(StubBytes.data(), StubBytes.size(), S, Stub));
  for (InternalDecodedInst &DI : Stub)
    DI.Offset += StubBegin;
  Stub.push_back(
      decodeTensorTestInstruction("s_endpgm", UnrelatedTarget, S));
  TensorDispatchStub Dispatch{StubBegin, StubBegin + KernelEntryStubStride,
                              Info->TerminalVAddr, UnrelatedTarget};
  TensorDescriptorMustAnalysis Positive = runTensorProvenanceAnalysis(
      Body, Stub, llvm::ArrayRef<TensorDispatchStub>(Dispatch), {}, {}, S);
  EXPECT_TRUE(Positive.Low16KnownZero.test(tensorInstructionIndex(Body)));

  constexpr uint64_t ForeignSource = 0x1200;
  const uint64_t StubInterior = Stub[2].Offset;
  Stub.push_back(decodeTensorTestInstruction(
      S.encodeSBranch(ForeignSource, StubInterior), ForeignSource, S));
  TensorDescriptorMustAnalysis Negative = runTensorProvenanceAnalysis(
      Body, Stub, llvm::ArrayRef<TensorDispatchStub>(Dispatch), {StubInterior},
      {}, S);
  EXPECT_FALSE(Negative.Low16KnownZero.test(tensorInstructionIndex(Body)));
}

// -- decodeTextSection instruction-decode cache -------------------------------
//
// decodeTextSection caches decode results keyed on the up-to-getMaxInstLength()
// byte window at each position, so byte-identical instructions reuse the first
// decode instead of re-running the disassembler. The cache is unconditional (no
// opt-in flag), so every decodeTextSection call above already exercises the
// store path; these tests target the reuse and edge behaviour flagged in
// review: repeated instructions must reuse decodes without corrupting the
// per-occurrence Offset, distinct instructions of different sizes must not
// alias one another, and a truncated final window (fewer than
// getMaxInstLength() bytes left) must decode correctly rather than returning a
// stale, oversized hit from an earlier full-length window.

// Append the assembled bytes of each asm line in \p Lines to \p Text. Aborts
// the test via appendSingleInstBytes if any line fails to assemble.
static void appendInstStream(llvm::SmallVectorImpl<uint8_t> &Text,
                             llvm::ArrayRef<const char *> Lines,
                             const LLVMState &S) {
  for (const char *Line : Lines)
    ASSERT_TRUE(appendSingleInstBytes(Text, Line, S));
}

TEST(DecodeCache, RepeatedInstructionsReuseDecodeWithPerOccurrenceOffset) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  // A run of identical s_nops so interior positions hit the cache.
  constexpr unsigned Count = 8;
  llvm::SmallVector<uint8_t> Text;
  for (unsigned I = 0; I < Count; ++I)
    ASSERT_TRUE(appendSingleInstBytes(Text, "s_nop 0", S));

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Text.data(), Text.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), Count);

  const llvm::MCInst Ref = assembleOne("s_nop 0", S);
  uint64_t ExpectedOffset = 0;
  for (const InternalDecodedInst &DI : Decoded) {
    EXPECT_EQ(DI.Mnemonic, "s_nop");
    EXPECT_EQ(DI.Size, MinInstSize);
    // Cache hits must still report a successful decode.
    EXPECT_TRUE(DI.DecodeSucceeded);
    // Offset is set per occurrence and must never come from the cached entry.
    EXPECT_EQ(DI.Offset, ExpectedOffset);
    expectSameOperands(DI.Inst, Ref, "repeated s_nop");
    ExpectedOffset += DI.Size;
  }
}

TEST(DecodeCache, InterleavedDistinctSizesUseCorrectEntries) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  // Mix 4/8/12-byte instructions, repeating some, so a wrong key would return
  // a differently sized decode.
  const char *Seq[] = {
      "s_nop 0",                                           // 4 bytes
      "v_cvt_pk_fp8_f32 v4, 1.0, 0.5 clamp",               // 8 bytes
      "s_nop 0",                                           // 4 bytes (repeat)
      "v_cvt_pk_fp8_f32 v4, 0x477f0000, 0x477f0000 clamp", // 12 bytes
      "v_cvt_pk_fp8_f32 v4, 1.0, 0.5 clamp",               // 8 bytes (repeat)
      "s_nop 0",                                           // 4 bytes (repeat)
  };
  const uint32_t ExpectedSizes[] = {MinInstSize,     2 * MinInstSize,
                                    MinInstSize,     3 * MinInstSize,
                                    2 * MinInstSize, MinInstSize};

  llvm::SmallVector<uint8_t> Text;
  appendInstStream(Text, Seq, S);

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Text.data(), Text.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), std::size(Seq));

  uint64_t ExpectedOffset = 0;
  for (size_t I = 0; I < std::size(Seq); ++I) {
    const InternalDecodedInst &DI = Decoded[I];
    EXPECT_EQ(DI.Size, ExpectedSizes[I]) << "inst " << I;
    EXPECT_EQ(DI.Offset, ExpectedOffset) << "inst " << I;
    expectSameOperands(DI.Inst, assembleOne(Seq[I], S), Seq[I]);
    ExpectedOffset += DI.Size;
  }
  EXPECT_EQ(ExpectedOffset, Text.size());
}

TEST(DecodeCache, TruncatedFinalWindowDecodesWithoutStaleHit) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  // The final s_nop is keyed on a truncated (< getMaxInstLength()) window; it
  // must decode cleanly rather than aliasing a longer cached entry.
  const unsigned MaxInstLen = S.MAI->getMaxInstLength(S.STI.get());
  ASSERT_GT(MaxInstLen, static_cast<unsigned>(MinInstSize))
      << "test assumes a multi-dword max instruction window";

  const char *Seq[] = {
      "v_cvt_pk_fp8_f32 v4, 0x477f0000, 0x477f0000 clamp", // 12 bytes
      "s_nop 0",                                           // 4 bytes
      "s_nop 0",                                           // final, truncated
  };
  llvm::SmallVector<uint8_t> Text;
  appendInstStream(Text, Seq, S);

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Text.data(), Text.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), std::size(Seq));

  uint64_t Consumed = 0;
  for (size_t I = 0; I < std::size(Seq); ++I) {
    const InternalDecodedInst &DI = Decoded[I];
    EXPECT_EQ(DI.Offset, Consumed) << "inst " << I;
    expectSameOperands(DI.Inst, assembleOne(Seq[I], S), Seq[I]);
    Consumed += DI.Size;
  }
  const InternalDecodedInst &Last = Decoded.back();
  EXPECT_EQ(Last.Mnemonic, "s_nop");
  EXPECT_EQ(Last.Size, MinInstSize);
  // Stream consumed exactly (no over-run).
  EXPECT_EQ(Consumed, Text.size());
}
