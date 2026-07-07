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
#include "comgr-test-elf-utils.h"
#include "comgr.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
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

static uint32_t appendString(std::string &Table, llvm::StringRef Value) {
  uint32_t Offset = static_cast<uint32_t>(Table.size());
  Table += Value;
  Table.push_back('\0');
  return Offset;
}

static std::optional<uint64_t> findSymbolVAddr(const ElfView &View,
                                               llvm::StringRef Name) {
  for (const ElfView::ELFT::Shdr &SymShdr : View.sections()) {
    if (SymShdr.sh_type != llvm::ELF::SHT_SYMTAB &&
        SymShdr.sh_type != llvm::ELF::SHT_DYNSYM)
      continue;

    llvm::Expected<ElfView::ELFT::SymRange> SymsOrErr =
        View.file().symbols(&SymShdr);
    if (!SymsOrErr) {
      llvm::consumeError(SymsOrErr.takeError());
      return std::nullopt;
    }
    llvm::Expected<llvm::StringRef> StrTabOrErr =
        View.file().getStringTableForSymtab(SymShdr, View.sections());
    if (!StrTabOrErr) {
      llvm::consumeError(StrTabOrErr.takeError());
      return std::nullopt;
    }

    for (const ElfView::ELFT::Sym &Sym : *SymsOrErr) {
      llvm::Expected<llvm::StringRef> SymNameOrErr = Sym.getName(*StrTabOrErr);
      if (!SymNameOrErr) {
        llvm::consumeError(SymNameOrErr.takeError());
        return std::nullopt;
      }
      if (*SymNameOrErr == Name)
        return Sym.st_value;
    }
  }
  return std::nullopt;
}

static std::optional<uint64_t> findSectionVAddr(const ElfView &View,
                                                llvm::StringRef Name) {
  for (const ElfView::ELFT::Shdr &Shdr : View.sections()) {
    llvm::Expected<llvm::StringRef> SectionNameOrErr =
        View.file().getSectionName(Shdr);
    if (!SectionNameOrErr) {
      llvm::consumeError(SectionNameOrErr.takeError());
      return std::nullopt;
    }
    if (*SectionNameOrErr == Name)
      return Shdr.sh_addr;
  }
  return std::nullopt;
}

struct ManagedGlobalKernelElf {
  std::vector<uint8_t> Bytes;
  uint64_t TextAddr = 0;
  uint64_t TextOffset = 0;
  uint64_t TextSize = 0;
  uint64_t DataAddr = 0;
  uint64_t BssAddr = 0;
  uint64_t LiteralOffset = 0;
  uint64_t BakedDelta = 0;
};

static ManagedGlobalKernelElf makeManagedGlobalKernelElf(const LLVMState &S) {
  using namespace llvm::ELF;
  namespace hsa = llvm::amdhsa;

  static constexpr uint64_t PhOff = sizeof(Elf64_Ehdr);
  static constexpr uint64_t TextOff = 0x240;
  static constexpr uint64_t TextAddr = 0x1600;
  static constexpr uint64_t TextSize = 64;
  static constexpr uint64_t RodataAddr = 0x3000;
  static constexpr uint64_t DataAddr = 0x3870;
  static constexpr uint64_t BssAddr = 0x3878;
  static constexpr uint64_t DataSize = 8;
  static constexpr uint64_t BssSize = 4;
  static constexpr uint64_t LiteralOffset = 16;
  static constexpr uint64_t KdBytes = sizeof(hsa::kernel_descriptor_t);

  std::vector<uint8_t> Text(TextSize, 0);
  llvm::SmallVector<uint8_t> EndPgm = assembleSingleInst("s_endpgm", S);
  std::memcpy(Text.data(), EndPgm.data(), EndPgm.size());
  uint64_t BakedDelta = BssAddr - (TextAddr + LiteralOffset);
  std::memcpy(Text.data() + LiteralOffset, &BakedDelta, sizeof(BakedDelta));

  std::string StrTab;
  StrTab.push_back('\0');
  uint32_t KernelNameOff = appendString(StrTab, "GPU_func");
  uint32_t KdNameOff = appendString(StrTab, "GPU_func.kd");
  uint32_t ManagedNameOff = appendString(StrTab, "x.managed");
  uint32_t XNameOff = appendString(StrTab, "x");

  std::string ShStrTab;
  ShStrTab.push_back('\0');
  uint32_t TextNameOff = appendString(ShStrTab, ".text");
  uint32_t RodataNameOff = appendString(ShStrTab, ".rodata");
  uint32_t DataNameOff = appendString(ShStrTab, ".data");
  uint32_t BssNameOff = appendString(ShStrTab, ".bss");
  uint32_t StrTabNameOff = appendString(ShStrTab, ".strtab");
  uint32_t SymTabNameOff = appendString(ShStrTab, ".symtab");
  uint32_t ShStrTabNameOff = appendString(ShStrTab, ".shstrtab");

  const uint64_t RodataOff = comgr_test::alignTo8(TextOff + Text.size());
  const uint64_t DataOff = comgr_test::alignTo8(RodataOff + KdBytes);
  const uint64_t StrTabOff = comgr_test::alignTo8(DataOff + DataSize);
  static constexpr uint64_t SymCount = 5;
  const uint64_t SymTabOff = comgr_test::alignTo8(StrTabOff + StrTab.size());
  const uint64_t ShStrTabOff =
      comgr_test::alignTo8(SymTabOff + SymCount * sizeof(Elf64_Sym));
  const uint64_t ShOff = comgr_test::alignTo8(ShStrTabOff + ShStrTab.size());
  static constexpr uint64_t ShCount = 8;
  const uint64_t BufSize =
      comgr_test::alignTo8(ShOff + ShCount * sizeof(Elf64_Shdr));

  ManagedGlobalKernelElf Result;
  Result.Bytes.assign(BufSize, 0);
  Result.TextAddr = TextAddr;
  Result.TextOffset = TextOff;
  Result.TextSize = Text.size();
  Result.DataAddr = DataAddr;
  Result.BssAddr = BssAddr;
  Result.LiteralOffset = LiteralOffset;
  Result.BakedDelta = BakedDelta;

  uint8_t *Buf = Result.Bytes.data();
  std::memcpy(Buf + TextOff, Text.data(), Text.size());
  std::memcpy(Buf + StrTabOff, StrTab.data(), StrTab.size());
  std::memcpy(Buf + ShStrTabOff, ShStrTab.data(), ShStrTab.size());

  Elf64_Ehdr Ehdr = comgr_test::makeElf64Ehdr(EM_AMDGPU);
  Ehdr.e_ident[EI_OSABI] = ELFOSABI_AMDGPU_HSA;
  Ehdr.e_type = ET_DYN;
  Ehdr.e_version = EV_CURRENT;
  Ehdr.e_ehsize = sizeof(Elf64_Ehdr);
  Ehdr.e_phoff = PhOff;
  Ehdr.e_phentsize = sizeof(Elf64_Phdr);
  Ehdr.e_phnum = 3;
  Ehdr.e_shoff = ShOff;
  Ehdr.e_shentsize = sizeof(Elf64_Shdr);
  Ehdr.e_shnum = ShCount;
  Ehdr.e_shstrndx = 7;
  std::memcpy(Buf, &Ehdr, sizeof(Ehdr));

  Elf64_Phdr PhdrTable{};
  PhdrTable.p_type = PT_PHDR;
  PhdrTable.p_offset = PhOff;
  PhdrTable.p_vaddr = PhOff;
  PhdrTable.p_paddr = PhOff;
  PhdrTable.p_filesz = 3 * sizeof(Elf64_Phdr);
  PhdrTable.p_memsz = 3 * sizeof(Elf64_Phdr);
  PhdrTable.p_flags = PF_R;
  PhdrTable.p_align = 8;
  std::memcpy(Buf + PhOff, &PhdrTable, sizeof(PhdrTable));

  Elf64_Phdr TextLoad{};
  TextLoad.p_type = PT_LOAD;
  TextLoad.p_flags = PF_R | PF_X;
  TextLoad.p_offset = TextOff;
  TextLoad.p_vaddr = TextAddr;
  TextLoad.p_paddr = TextAddr;
  TextLoad.p_filesz = Text.size();
  TextLoad.p_memsz = Text.size();
  TextLoad.p_align = 0x1000;
  std::memcpy(Buf + PhOff + sizeof(Elf64_Phdr), &TextLoad, sizeof(TextLoad));

  Elf64_Phdr DataLoad{};
  DataLoad.p_type = PT_LOAD;
  DataLoad.p_flags = PF_R | PF_W;
  DataLoad.p_offset = RodataOff;
  DataLoad.p_vaddr = RodataAddr;
  DataLoad.p_paddr = RodataAddr;
  DataLoad.p_filesz = DataOff + DataSize - RodataOff;
  DataLoad.p_memsz = BssAddr + BssSize - RodataAddr;
  DataLoad.p_align = 0x1000;
  std::memcpy(Buf + PhOff + 2 * sizeof(Elf64_Phdr), &DataLoad,
              sizeof(DataLoad));

  int64_t EntryOffset =
      static_cast<int64_t>(TextAddr) - static_cast<int64_t>(RodataAddr);
  std::memcpy(
      Buf + RodataOff +
          offsetof(hsa::kernel_descriptor_t, kernel_code_entry_byte_offset),
      &EntryOffset, sizeof(EntryOffset));

  Elf64_Shdr TextSh{};
  TextSh.sh_name = TextNameOff;
  TextSh.sh_type = SHT_PROGBITS;
  TextSh.sh_flags = SHF_ALLOC | SHF_EXECINSTR;
  TextSh.sh_addr = TextAddr;
  TextSh.sh_offset = TextOff;
  TextSh.sh_size = Text.size();
  TextSh.sh_addralign = 4;
  std::memcpy(Buf + ShOff + 1 * sizeof(Elf64_Shdr), &TextSh, sizeof(TextSh));

  Elf64_Shdr RodataSh{};
  RodataSh.sh_name = RodataNameOff;
  RodataSh.sh_type = SHT_PROGBITS;
  RodataSh.sh_flags = SHF_ALLOC;
  RodataSh.sh_addr = RodataAddr;
  RodataSh.sh_offset = RodataOff;
  RodataSh.sh_size = KdBytes;
  RodataSh.sh_addralign = 8;
  std::memcpy(Buf + ShOff + 2 * sizeof(Elf64_Shdr), &RodataSh,
              sizeof(RodataSh));

  Elf64_Shdr DataSh{};
  DataSh.sh_name = DataNameOff;
  DataSh.sh_type = SHT_PROGBITS;
  DataSh.sh_flags = SHF_ALLOC | SHF_WRITE;
  DataSh.sh_addr = DataAddr;
  DataSh.sh_offset = DataOff;
  DataSh.sh_size = DataSize;
  DataSh.sh_addralign = 8;
  std::memcpy(Buf + ShOff + 3 * sizeof(Elf64_Shdr), &DataSh, sizeof(DataSh));

  Elf64_Shdr BssSh{};
  BssSh.sh_name = BssNameOff;
  BssSh.sh_type = SHT_NOBITS;
  BssSh.sh_flags = SHF_ALLOC | SHF_WRITE;
  BssSh.sh_addr = BssAddr;
  BssSh.sh_offset = DataOff + DataSize;
  BssSh.sh_size = BssSize;
  BssSh.sh_addralign = 4;
  std::memcpy(Buf + ShOff + 4 * sizeof(Elf64_Shdr), &BssSh, sizeof(BssSh));

  Elf64_Shdr StrTabSh{};
  StrTabSh.sh_name = StrTabNameOff;
  StrTabSh.sh_type = SHT_STRTAB;
  StrTabSh.sh_offset = StrTabOff;
  StrTabSh.sh_size = StrTab.size();
  std::memcpy(Buf + ShOff + 5 * sizeof(Elf64_Shdr), &StrTabSh,
              sizeof(StrTabSh));

  Elf64_Shdr SymTabSh{};
  SymTabSh.sh_name = SymTabNameOff;
  SymTabSh.sh_type = SHT_SYMTAB;
  SymTabSh.sh_offset = SymTabOff;
  SymTabSh.sh_size = SymCount * sizeof(Elf64_Sym);
  SymTabSh.sh_link = 5;
  SymTabSh.sh_entsize = sizeof(Elf64_Sym);
  std::memcpy(Buf + ShOff + 6 * sizeof(Elf64_Shdr), &SymTabSh,
              sizeof(SymTabSh));

  Elf64_Shdr ShStrTabSh{};
  ShStrTabSh.sh_name = ShStrTabNameOff;
  ShStrTabSh.sh_type = SHT_STRTAB;
  ShStrTabSh.sh_offset = ShStrTabOff;
  ShStrTabSh.sh_size = ShStrTab.size();
  std::memcpy(Buf + ShOff + 7 * sizeof(Elf64_Shdr), &ShStrTabSh,
              sizeof(ShStrTabSh));

  Elf64_Sym KernelSym{};
  KernelSym.st_name = KernelNameOff;
  KernelSym.setBindingAndType(STB_GLOBAL, STT_FUNC);
  KernelSym.st_shndx = 1;
  KernelSym.st_value = TextAddr;
  KernelSym.st_size = Text.size();
  std::memcpy(Buf + SymTabOff + 1 * sizeof(Elf64_Sym), &KernelSym,
              sizeof(KernelSym));

  Elf64_Sym KdSym{};
  KdSym.st_name = KdNameOff;
  KdSym.setBindingAndType(STB_GLOBAL, STT_OBJECT);
  KdSym.st_shndx = 2;
  KdSym.st_value = RodataAddr;
  KdSym.st_size = KdBytes;
  std::memcpy(Buf + SymTabOff + 2 * sizeof(Elf64_Sym), &KdSym, sizeof(KdSym));

  Elf64_Sym ManagedSym{};
  ManagedSym.st_name = ManagedNameOff;
  ManagedSym.setBindingAndType(STB_GLOBAL, STT_OBJECT);
  ManagedSym.st_shndx = 3;
  ManagedSym.st_value = DataAddr;
  ManagedSym.st_size = DataSize;
  std::memcpy(Buf + SymTabOff + 3 * sizeof(Elf64_Sym), &ManagedSym,
              sizeof(ManagedSym));

  Elf64_Sym XSym{};
  XSym.st_name = XNameOff;
  XSym.setBindingAndType(STB_GLOBAL, STT_OBJECT);
  XSym.st_shndx = 4;
  XSym.st_value = BssAddr;
  XSym.st_size = BssSize;
  std::memcpy(Buf + SymTabOff + 4 * sizeof(Elf64_Sym), &XSym, sizeof(XSym));

  return Result;
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
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(Text, Opts);
  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());

  uint8_t *Kd = ViewOrErr->findKernelDescriptor("kernel");
  ASSERT_NE(Kd, nullptr);

  std::optional<ExecutableSegmentPlan> Plan = ViewOrErr->planExecutableSegment(
      KernelEntryStubStride, KernelEntryStubSegmentAlign, 0);
  ASSERT_TRUE(Plan.has_value());
  llvm::SmallVector<uint8_t> EntryBytes;
  std::vector<KernelEntryTrampolineFixup> Fixups;
  std::optional<uint32_t> Count = appendKernelEntryTrampolines(
      *ViewOrErr, S, /*MaxSgprs=*/106, Plan->PayloadVAddr, EntryBytes, Fixups);
  ASSERT_TRUE(Count.has_value());
  EXPECT_EQ(*Count, 1u);
  ASSERT_EQ(Fixups.size(), 1u);
  EXPECT_EQ(Fixups[0].InstPrefLines, KernelEntryStubInstPrefLines);

  const uint64_t ExpectedGuard =
      computeKernelEntryPrefetchGuardBytes(KernelEntryStubInstPrefLines);
  EXPECT_EQ(ExpectedGuard, 0u);
  EXPECT_EQ(EntryBytes.size(), KernelEntryStubStride + ExpectedGuard);
  EXPECT_EQ(Fixups[0].StubVAddr, Plan->PayloadVAddr);

  std::unique_ptr<llvm::WritableMemoryBuffer> Out =
      ViewOrErr->appendExecutableSegment(EntryBytes, *Plan, ".hotswap.entry");
  ASSERT_NE(Out, nullptr);

  ASSERT_TRUE(rewriteKernelEntryDescriptorOffsets(*Out, S.Cpu, Fixups));

  uint8_t *OutData = const_cast<uint8_t *>(
      reinterpret_cast<const uint8_t *>(Out->getBufferStart()));
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
  EXPECT_EQ(
      AMDHSA_BITS_GET(OutRsrc3, hsa::COMPUTE_PGM_RSRC3_GFX11_INST_PREF_SIZE),
      KernelEntryStubInstPrefLines);
  EXPECT_NE(OutRsrc3 & hsa::COMPUTE_PGM_RSRC3_GFX12_PLUS_GLG_EN, 0u);
  EXPECT_EQ(Fixups[0].RequiredSgprs, 10u);
  uint32_t OutRsrc1 = 0;
  std::memcpy(&OutRsrc1,
              OutKd + offsetof(hsa::kernel_descriptor_t, compute_pgm_rsrc1),
              sizeof(OutRsrc1));
  unsigned ReservedSgprs =
      (AMDHSA_BITS_GET(OutRsrc1,
                       hsa::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT) +
       1) *
      8;
  EXPECT_GE(ReservedSgprs, Fixups[0].RequiredSgprs);

  std::vector<KernelDescriptorInfo> KDs = OutView->kernelDescriptors();
  ASSERT_EQ(KDs.size(), 1u);
  std::optional<uint64_t> KdVAddr = OutView->getKernelDescriptorVAddr("kernel");
  ASSERT_TRUE(KdVAddr.has_value());
  EXPECT_EQ(KDs[0].EntryOffset,
            static_cast<int64_t>(Fixups[0].StubVAddr - *KdVAddr));
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

  std::optional<ExecutableSegmentPlan> Plan = ViewOrErr->planExecutableSegment(
      KernelEntryStubStride, KernelEntryStubSegmentAlign, 0);
  ASSERT_TRUE(Plan.has_value());
  llvm::SmallVector<uint8_t> EntryBytes;
  std::vector<KernelEntryTrampolineFixup> Fixups;
  std::optional<uint32_t> Count = appendKernelEntryTrampolines(
      *ViewOrErr, S, /*MaxSgprs=*/106, Plan->PayloadVAddr, EntryBytes, Fixups);

  ASSERT_TRUE(Count.has_value());
  EXPECT_EQ(*Count, 1u);
  ASSERT_EQ(Fixups.size(), 1u);
  EXPECT_EQ(Fixups[0].StubVAddr % KernelEntryStubStride, 0u);
  EXPECT_EQ(Fixups[0].StubVAddr, Plan->PayloadVAddr);
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

  llvm::SmallVector<uint8_t> EntryBytes;
  std::vector<KernelEntryTrampolineFixup> Fixups;
  std::optional<uint32_t> Count = appendKernelEntryTrampolines(
      *ViewOrErr, S, /*MaxSgprs=*/106, /*StubBaseVAddr=*/0x4000, EntryBytes,
      Fixups);

  ASSERT_TRUE(Count.has_value());
  EXPECT_EQ(*Count, 0u);
  EXPECT_TRUE(EntryBytes.empty());
  EXPECT_TRUE(Fixups.empty());
}

TEST(KernelEntryTrampoline, AppendFailsWithoutSgprScratchPair) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  llvm::SmallVector<uint8_t> Text = assembleSingleInst("s_endpgm", S);
  ASSERT_EQ(Text.size(), MinInstSize);

  comgr_test::KernelDescriptorElfOptions Opts;
  Opts.MetadataSgprCount = 105;
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(Text, Opts);
  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)ViewOrErr) << llvm::toString(ViewOrErr.takeError());

  llvm::SmallVector<uint8_t> EntryBytes;
  std::vector<KernelEntryTrampolineFixup> Fixups;
  std::optional<uint32_t> Count = appendKernelEntryTrampolines(
      *ViewOrErr, S, /*MaxSgprs=*/106, /*StubBaseVAddr=*/0x4000, EntryBytes,
      Fixups);

  EXPECT_FALSE(Count.has_value());
  EXPECT_TRUE(EntryBytes.empty());
  EXPECT_TRUE(Fixups.empty());
}

TEST(KernelEntryTrampoline, EntryRewritePreservesManagedGlobalReferences) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  ManagedGlobalKernelElf Obj = makeManagedGlobalKernelElf(S);
  llvm::Expected<ElfView> InputView =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)InputView) << llvm::toString(InputView.takeError());
  ASSERT_EQ(findSectionVAddr(*InputView, ".data"), Obj.DataAddr);
  ASSERT_EQ(findSectionVAddr(*InputView, ".bss"), Obj.BssAddr);
  ASSERT_EQ(findSymbolVAddr(*InputView, "x.managed"), Obj.DataAddr);
  ASSERT_EQ(findSymbolVAddr(*InputView, "x"), Obj.BssAddr);

  uint64_t InputBakedDelta = 0;
  std::memcpy(&InputBakedDelta,
              Obj.Bytes.data() + Obj.TextOffset + Obj.LiteralOffset,
              sizeof(InputBakedDelta));
  ASSERT_EQ(InputBakedDelta, Obj.BakedDelta);

  Gfx1250RewriteOptions Options;
  Options.RunB0A0Patches = false;
  Options.RunEntryTrampolines = true;
  std::unique_ptr<llvm::MemoryBuffer> Out;
  amd_comgr_status_t Status = retargetCodeObject(
      Obj.Bytes.data(), Obj.Bytes.size(), makeGfx1250Ident(), Options, Out);
  ASSERT_EQ(Status, AMD_COMGR_STATUS_SUCCESS);
  ASSERT_NE(Out, nullptr);

  uint8_t *OutData = const_cast<uint8_t *>(
      reinterpret_cast<const uint8_t *>(Out->getBufferStart()));
  llvm::Expected<ElfView> OutView =
      ElfView::create(OutData, Out->getBufferSize());
  ASSERT_TRUE((bool)OutView) << llvm::toString(OutView.takeError());

  EXPECT_EQ(OutView->textAddr(), Obj.TextAddr);
  EXPECT_EQ(OutView->textSize(), Obj.TextSize);
  EXPECT_EQ(std::memcmp(OutView->textData(), Obj.Bytes.data() + Obj.TextOffset,
                        Obj.TextSize),
            0);

  uint64_t OutputBakedDelta = 0;
  std::memcpy(&OutputBakedDelta, OutView->textData() + Obj.LiteralOffset,
              sizeof(OutputBakedDelta));
  EXPECT_EQ(OutputBakedDelta, Obj.BakedDelta);

  EXPECT_EQ(findSectionVAddr(*OutView, ".data"), Obj.DataAddr);
  EXPECT_EQ(findSectionVAddr(*OutView, ".bss"), Obj.BssAddr);
  EXPECT_EQ(findSymbolVAddr(*OutView, "x.managed"), Obj.DataAddr);
  EXPECT_EQ(findSymbolVAddr(*OutView, "x"), Obj.BssAddr);

  std::vector<KernelDescriptorInfo> KDs = OutView->kernelDescriptors();
  ASSERT_EQ(KDs.size(), 1u);
  ASSERT_GE(KDs[0].EntryOffset, 0);
  uint64_t NewEntryVAddr =
      KDs[0].VAddr + static_cast<uint64_t>(KDs[0].EntryOffset);
  EXPECT_GT(NewEntryVAddr, Obj.BssAddr);
  EXPECT_EQ(findSectionVAddr(*OutView, ".hotswap.entry"),
            NewEntryVAddr - (NewEntryVAddr % KernelEntryStubStride));
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
