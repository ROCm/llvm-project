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

#include "comgr-test-elf-utils.h"
#include "comgr.h"
#include "hotswap/rewriter/internal.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <iterator>
#include <limits>
#include <vector>

using namespace COMGR;
using namespace COMGR::hotswap;

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

static std::vector<InternalDecodedInst>
decodeAsmSequence(const LLVMState &S, llvm::ArrayRef<llvm::StringRef> Lines) {
  llvm::SmallVector<uint8_t, 32> Bytes;
  for (llvm::StringRef Line : Lines) {
    llvm::SmallVector<uint8_t> Encoded = assembleSingleInst(Line, S);
    EXPECT_FALSE(Encoded.empty()) << Line.str();
    Bytes.append(Encoded);
  }
  std::vector<InternalDecodedInst> Decoded;
  EXPECT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  return Decoded;
}

static bool scalarIncomingSgprIsUnsafe(
    llvm::ArrayRef<InternalDecodedInst> Decoded, const LLVMState &S,
    uint64_t FunctionBegin, uint64_t FunctionEnd, uint64_t Continuation,
    llvm::ArrayRef<llvm::MCRegister> NumberedSgprs, unsigned Sgpr) {
  auto FindInstruction = [&](uint64_t Offset) -> std::optional<size_t> {
    if (Offset < FunctionBegin || Offset >= FunctionEnd)
      return std::nullopt;
    auto It = llvm::lower_bound(
        Decoded, Offset, [](const InternalDecodedInst &DI, uint64_t Target) {
          return DI.Offset < Target;
        });
    if (It == Decoded.end() || It->Offset != Offset)
      return std::nullopt;
    return It - Decoded.begin();
  };
  std::optional<size_t> Start = FindInstruction(Continuation);
  if (!Start)
    return true;

  llvm::SmallVector<size_t, 8> Worklist(1, *Start);
  llvm::DenseSet<size_t> Visited;
  while (!Worklist.empty()) {
    size_t Index = Worklist.pop_back_val();
    if (!Visited.insert(Index).second)
      continue;
    const InternalDecodedInst &DI = Decoded[Index];
    if (!DI.DecodeSucceeded || !S.MIA || DI.Offset < FunctionBegin ||
        DI.Offset >= FunctionEnd)
      return true;

    llvm::BitVector Uses(NumberedSgprs.size());
    llvm::BitVector Defs(NumberedSgprs.size());
    getNumberedSgprUsesAndDefs(DI, S, NumberedSgprs, Uses, Defs);
    if (Uses.test(Sgpr))
      return true;
    if (Defs.test(Sgpr) || DI.Inst.getOpcode() == S.SEndPgmOpcode ||
        DI.Inst.getOpcode() == S.SEndPgmSavedOpcode)
      continue;

    auto AddSuccessor = [&](uint64_t Offset) {
      std::optional<size_t> Successor = FindInstruction(Offset);
      if (!Successor)
        return false;
      Worklist.push_back(*Successor);
      return true;
    };
    if (S.MIA->isCall(DI.Inst) || S.MIA->isIndirectBranch(DI.Inst) ||
        S.MIA->isReturn(DI.Inst))
      return true;
    if (S.MIA->isBranch(DI.Inst)) {
      std::optional<uint64_t> Target = evaluateDirectControlFlowTarget(DI, S);
      if (!Target || !AddSuccessor(*Target))
        return true;
      if (S.MIA->isUnconditionalBranch(DI.Inst))
        continue;
    } else if (S.MIA->mayAffectControlFlow(DI.Inst, *S.MRI) &&
               !S.MIA->isBarrier(DI.Inst)) {
      return true;
    }
    std::optional<uint64_t> Fallthrough =
        llvm::checkedAddUnsigned(DI.Offset, static_cast<uint64_t>(DI.Size));
    if (!Fallthrough || !AddSuccessor(*Fallthrough))
      return true;
  }
  return false;
}

static bool
scalarIncomingRegisterIsNeeded(llvm::ArrayRef<InternalDecodedInst> Decoded,
                               const LLVMState &S, uint64_t FunctionBegin,
                               uint64_t FunctionEnd, uint64_t Continuation,
                               llvm::MCRegister Register) {
  auto FindInstruction = [&](uint64_t Offset) -> std::optional<size_t> {
    if (Offset < FunctionBegin || Offset >= FunctionEnd)
      return std::nullopt;
    auto It = llvm::lower_bound(
        Decoded, Offset, [](const InternalDecodedInst &DI, uint64_t Target) {
          return DI.Offset < Target;
        });
    if (It == Decoded.end() || It->Offset != Offset)
      return std::nullopt;
    return It - Decoded.begin();
  };
  std::optional<size_t> Start = FindInstruction(Continuation);
  if (!Start)
    return true;

  llvm::SmallVector<size_t, 8> Worklist(1, *Start);
  llvm::DenseSet<size_t> Visited;
  while (!Worklist.empty()) {
    size_t Index = Worklist.pop_back_val();
    if (!Visited.insert(Index).second)
      continue;
    const InternalDecodedInst &DI = Decoded[Index];
    if (!DI.DecodeSucceeded || !S.MIA || DI.Offset < FunctionBegin ||
        DI.Offset >= FunctionEnd)
      return true;
    if (instructionReadsRegister(DI, S, Register))
      return true;
    if (instructionFullyWritesRegister(DI, S, Register) ||
        DI.Inst.getOpcode() == S.SEndPgmOpcode ||
        DI.Inst.getOpcode() == S.SEndPgmSavedOpcode)
      continue;

    auto AddSuccessor = [&](uint64_t Offset) {
      std::optional<size_t> Successor = FindInstruction(Offset);
      if (!Successor)
        return false;
      Worklist.push_back(*Successor);
      return true;
    };
    if (S.MIA->isCall(DI.Inst) || S.MIA->isIndirectBranch(DI.Inst) ||
        S.MIA->isReturn(DI.Inst))
      return true;
    if (S.MIA->isBranch(DI.Inst)) {
      std::optional<uint64_t> Target = evaluateDirectControlFlowTarget(DI, S);
      if (!Target || !AddSuccessor(*Target))
        return true;
      if (S.MIA->isUnconditionalBranch(DI.Inst))
        continue;
    } else if (S.MIA->mayAffectControlFlow(DI.Inst, *S.MRI) &&
               !S.MIA->isBarrier(DI.Inst)) {
      return true;
    }
    std::optional<uint64_t> Fallthrough =
        llvm::checkedAddUnsigned(DI.Offset, static_cast<uint64_t>(DI.Size));
    if (!Fallthrough || !AddSuccessor(*Fallthrough))
      return true;
  }
  return false;
}

static void
expectBatchRegisterNeedsMatchesScalar(const LLVMState &S,
                                      llvm::ArrayRef<llvm::StringRef> Lines,
                                      llvm::MCRegister Register) {
  std::vector<InternalDecodedInst> Decoded = decodeAsmSequence(S, Lines);
  ASSERT_FALSE(Decoded.empty());
  uint64_t FunctionEnd = Decoded.back().Offset + Decoded.back().Size;
  std::optional<llvm::DenseSet<uint64_t>> Batch = computeIncomingRegisterNeeds(
      Decoded, S, /*FunctionBegin=*/0, FunctionEnd, Register);
  ASSERT_TRUE(Batch);
  for (const InternalDecodedInst &DI : Decoded)
    EXPECT_EQ(Batch->contains(DI.Offset),
              scalarIncomingRegisterIsNeeded(Decoded, S, /*FunctionBegin=*/0,
                                             FunctionEnd, DI.Offset, Register))
        << "continuation 0x" << llvm::utohexstr(DI.Offset);
}

static void
expectBatchSgprProofMatchesScalar(const LLVMState &S,
                                  llvm::ArrayRef<llvm::StringRef> Lines) {
  std::vector<InternalDecodedInst> Decoded = decodeAsmSequence(S, Lines);
  ASSERT_FALSE(Decoded.empty());
  std::optional<llvm::SmallVector<llvm::MCRegister, 128>> NumberedSgprs =
      resolveNumberedSgprRegisters(*S.MRI, /*MaxSgprs=*/106);
  ASSERT_TRUE(NumberedSgprs);
  uint64_t FunctionEnd = Decoded.back().Offset + Decoded.back().Size;
  llvm::SmallVector<uint64_t, 16> Continuations;
  for (const InternalDecodedInst &DI : Decoded)
    Continuations.push_back(DI.Offset);
  Continuations.push_back(1);
  BatchedSgprContinuationTestResult Batch =
      runBatchedSgprContinuationAnalysisForTest(Decoded, S, /*FunctionBegin=*/0,
                                                FunctionEnd, Continuations,
                                                *NumberedSgprs);
  EXPECT_EQ(Batch.Analyses, 1u);
  ASSERT_EQ(Batch.Queries.size(), Continuations.size());
  for (size_t Query = 0; Query != Decoded.size(); ++Query) {
    std::optional<llvm::BitVector> Scalar = unsafeIncomingNumberedSgprsInRange(
        Decoded, S, /*FunctionBegin=*/0, FunctionEnd, Continuations[Query],
        *NumberedSgprs);
    ASSERT_TRUE(Scalar);
    ASSERT_TRUE(Batch.Queries[Query]);
    EXPECT_EQ(*Batch.Queries[Query], *Scalar)
        << "continuation 0x" << llvm::utohexstr(Continuations[Query]);
    for (unsigned I = 0; I != NumberedSgprs->size(); ++I)
      EXPECT_EQ(Batch.Queries[Query]->test(I),
                scalarIncomingSgprIsUnsafe(Decoded, S, /*FunctionBegin=*/0,
                                           FunctionEnd, Continuations[Query],
                                           *NumberedSgprs, I))
          << "continuation 0x" << llvm::utohexstr(Continuations[Query]) << ", s"
          << I;
  }
  EXPECT_FALSE(Batch.Queries.back());
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

enum class FunctionTableElfMutation {
  None,
  NoRelro,
  RelocationGap,
  WrongRelocationKind,
  RelocatedSentinel,
  NonFunctionTarget,
  NonBoundaryTarget,
  NonZeroSlot,
  MisalignedTableSymbol,
  MalformedSymbolTable,
  FunctionEndInterior,
  FunctionSizeOutOfText,
};

struct FunctionTableTestElf {
  std::vector<uint8_t> Bytes;
  std::vector<InternalDecodedInst> Decoded;
  uint64_t CallOffset = 0;
};

static FunctionTableTestElf makeFunctionTableTestElf(
    const LLVMState &S, llvm::StringRef Load,
    FunctionTableElfMutation Mutation = FunctionTableElfMutation::None,
    uint64_t TableDelta = 0xFFC, llvm::StringRef CustomAsm = {},
    size_t CallerBeginIndex = 0,
    size_t CallerEndIndex = std::numeric_limits<size_t>::max(),
    size_t Target1BeginIndex = std::numeric_limits<size_t>::max()) {
  using namespace llvm::ELF;

  static constexpr uint64_t ShOff = sizeof(Elf64_Ehdr);
  static constexpr uint64_t PhOff = 0x200;
  static constexpr uint64_t TextOff = 0x300;
  static constexpr uint64_t TextAddr = 0x1000;
  static constexpr uint64_t TableAddr = 0x2000;
  static constexpr size_t TableSlots = 3;

  std::string Asm = CustomAsm.empty()
                        ? "s_get_pc_i64 s[4:5]\n"
                          "s_add_nc_u64 s[4:5], s[4:5], " +
                              std::to_string(TableDelta) + "\n" + Load.str() +
                              "\n"
                              "s_wait_kmcnt 0\n"
                              "s_swap_pc_i64 s[30:31], s[0:1]\n"
                              "s_endpgm\n"
                              "s_endpgm\n"
                              "s_endpgm\n"
                        : CustomAsm.str();
  llvm::SmallVector<uint8_t> Text = assembleInstructions(Asm, S);
  EXPECT_FALSE(Text.empty());
  std::vector<InternalDecodedInst> Decoded;
  EXPECT_TRUE(decodeTextSection(Text.data(), Text.size(), S, Decoded));
  EXPECT_GE(Decoded.size(), 3u);
  bool HasCustomCallerEnd =
      CallerEndIndex != std::numeric_limits<size_t>::max();
  if (!HasCustomCallerEnd)
    CallerEndIndex = Decoded.size() - 2;
  if (Target1BeginIndex == std::numeric_limits<size_t>::max())
    Target1BeginIndex = Decoded.size() - 1;
  EXPECT_LT(CallerBeginIndex, CallerEndIndex);
  EXPECT_LT(CallerEndIndex, Target1BeginIndex);
  EXPECT_LT(Target1BeginIndex, Decoded.size());

  const uint64_t CallerBegin = Decoded[CallerBeginIndex].Offset;
  const uint64_t CallerEnd = Decoded[CallerEndIndex].Offset;
  const uint64_t Target0 = TextAddr + CallerEnd;
  const uint64_t Target1 = TextAddr + Decoded[Target1BeginIndex].Offset;
  uint64_t CallOffset = 0;
  bool FoundCall = false;
  for (const InternalDecodedInst &DI : Decoded) {
    if (!S.MIA->isCall(DI.Inst))
      continue;
    CallOffset = DI.Offset;
    FoundCall = true;
    break;
  }
  EXPECT_TRUE(FoundCall);

  const char StrTab[] = "\0caller\0target0\0target1\0table\0";
  const char ShStrTab[] =
      "\0.text\0.data.rel.ro\0.strtab\0.symtab\0.rela.dyn\0.shstrtab\0";
  const uint64_t DataOff = alignTo8(TextOff + Text.size());
  const uint64_t DataSize = TableSlots * sizeof(uint64_t);
  const uint64_t StrTabOff = alignTo8(DataOff + DataSize);
  const uint64_t SymTabOff = alignTo8(StrTabOff + sizeof(StrTab));
  static constexpr size_t SymbolCount = 5;
  const uint64_t RelaOff =
      alignTo8(SymTabOff + SymbolCount * sizeof(Elf64_Sym));
  const size_t RelaCount =
      Mutation == FunctionTableElfMutation::RelocatedSentinel ? 3 : 2;
  const uint64_t ShStrTabOff =
      alignTo8(RelaOff + RelaCount * sizeof(Elf64_Rela));
  const uint64_t BufSize = alignTo8(ShStrTabOff + sizeof(ShStrTab));
  std::vector<uint8_t> Buf(BufSize, 0);
  std::memcpy(Buf.data() + TextOff, Text.data(), Text.size());
  std::memcpy(Buf.data() + StrTabOff, StrTab, sizeof(StrTab));
  std::memcpy(Buf.data() + ShStrTabOff, ShStrTab, sizeof(ShStrTab));
  if (Mutation == FunctionTableElfMutation::NonZeroSlot)
    Buf[DataOff] = 1;

  Elf64_Ehdr Ehdr = comgr_test::makeElf64Ehdr(EM_AMDGPU);
  Ehdr.e_ident[EI_OSABI] = ELFOSABI_AMDGPU_HSA;
  Ehdr.e_type = ET_DYN;
  Ehdr.e_version = EV_CURRENT;
  Ehdr.e_phoff = PhOff;
  Ehdr.e_shoff = ShOff;
  Ehdr.e_ehsize = sizeof(Elf64_Ehdr);
  Ehdr.e_phentsize = sizeof(Elf64_Phdr);
  Ehdr.e_phnum = 3;
  Ehdr.e_shentsize = sizeof(Elf64_Shdr);
  Ehdr.e_shnum = 7;
  Ehdr.e_shstrndx = 6;
  std::memcpy(Buf.data(), &Ehdr, sizeof(Ehdr));

  Elf64_Phdr TextPh{};
  TextPh.p_type = PT_LOAD;
  TextPh.p_flags = PF_R | PF_X;
  TextPh.p_offset = TextOff;
  TextPh.p_vaddr = TextAddr;
  TextPh.p_paddr = TextAddr;
  TextPh.p_filesz = Text.size();
  TextPh.p_memsz = Text.size();
  TextPh.p_align = 8;
  std::memcpy(Buf.data() + PhOff, &TextPh, sizeof(TextPh));

  Elf64_Phdr DataPh{};
  DataPh.p_type = PT_LOAD;
  DataPh.p_flags = PF_R | PF_W;
  DataPh.p_offset = DataOff;
  DataPh.p_vaddr = TableAddr;
  DataPh.p_paddr = TableAddr;
  DataPh.p_filesz = DataSize;
  DataPh.p_memsz = DataSize;
  DataPh.p_align = 8;
  std::memcpy(Buf.data() + PhOff + sizeof(Elf64_Phdr), &DataPh, sizeof(DataPh));

  Elf64_Phdr RelroPh = DataPh;
  RelroPh.p_type =
      Mutation == FunctionTableElfMutation::NoRelro ? PT_NOTE : PT_GNU_RELRO;
  RelroPh.p_flags = PF_R;
  std::memcpy(Buf.data() + PhOff + 2 * sizeof(Elf64_Phdr), &RelroPh,
              sizeof(RelroPh));

  Elf64_Shdr TextSh{};
  TextSh.sh_name = 1;
  TextSh.sh_type = SHT_PROGBITS;
  TextSh.sh_flags = SHF_ALLOC | SHF_EXECINSTR;
  TextSh.sh_offset = TextOff;
  TextSh.sh_addr = TextAddr;
  TextSh.sh_size = Text.size();
  TextSh.sh_addralign = 4;
  std::memcpy(Buf.data() + ShOff + sizeof(Elf64_Shdr), &TextSh, sizeof(TextSh));

  Elf64_Shdr DataSh{};
  DataSh.sh_name = 7;
  DataSh.sh_type = SHT_PROGBITS;
  DataSh.sh_flags = SHF_ALLOC | SHF_WRITE;
  DataSh.sh_offset = DataOff;
  DataSh.sh_addr = TableAddr;
  DataSh.sh_size = DataSize;
  DataSh.sh_addralign = 8;
  std::memcpy(Buf.data() + ShOff + 2 * sizeof(Elf64_Shdr), &DataSh,
              sizeof(DataSh));

  Elf64_Shdr StrtabSh{};
  StrtabSh.sh_name = 20;
  StrtabSh.sh_type = SHT_STRTAB;
  StrtabSh.sh_offset = StrTabOff;
  StrtabSh.sh_size = sizeof(StrTab);
  std::memcpy(Buf.data() + ShOff + 3 * sizeof(Elf64_Shdr), &StrtabSh,
              sizeof(StrtabSh));

  Elf64_Shdr SymtabSh{};
  SymtabSh.sh_name = 28;
  SymtabSh.sh_type = SHT_SYMTAB;
  SymtabSh.sh_offset = SymTabOff;
  SymtabSh.sh_size = SymbolCount * sizeof(Elf64_Sym);
  SymtabSh.sh_link = 3;
  SymtabSh.sh_info = SymbolCount;
  SymtabSh.sh_entsize =
      Mutation == FunctionTableElfMutation::MalformedSymbolTable
          ? sizeof(Elf64_Sym) - 1
          : sizeof(Elf64_Sym);
  std::memcpy(Buf.data() + ShOff + 4 * sizeof(Elf64_Shdr), &SymtabSh,
              sizeof(SymtabSh));

  Elf64_Shdr RelaSh{};
  RelaSh.sh_name = 36;
  RelaSh.sh_type = SHT_RELA;
  RelaSh.sh_offset = RelaOff;
  RelaSh.sh_size = RelaCount * sizeof(Elf64_Rela);
  RelaSh.sh_link = 4;
  RelaSh.sh_info = 0;
  RelaSh.sh_addralign = 8;
  RelaSh.sh_entsize = sizeof(Elf64_Rela);
  std::memcpy(Buf.data() + ShOff + 5 * sizeof(Elf64_Shdr), &RelaSh,
              sizeof(RelaSh));

  Elf64_Shdr ShstrSh{};
  ShstrSh.sh_name = 46;
  ShstrSh.sh_type = SHT_STRTAB;
  ShstrSh.sh_offset = ShStrTabOff;
  ShstrSh.sh_size = sizeof(ShStrTab);
  std::memcpy(Buf.data() + ShOff + 6 * sizeof(Elf64_Shdr), &ShstrSh,
              sizeof(ShstrSh));

  auto writeSymbol = [&](size_t Index, uint32_t Name, uint8_t Type,
                         uint16_t SectionIndex, uint64_t Value, uint64_t Size) {
    Elf64_Sym Symbol{};
    Symbol.st_name = Name;
    Symbol.setBindingAndType(STB_LOCAL, Type);
    Symbol.st_shndx = SectionIndex;
    Symbol.st_value = Value;
    Symbol.st_size = Size;
    std::memcpy(Buf.data() + SymTabOff + Index * sizeof(Elf64_Sym), &Symbol,
                sizeof(Symbol));
  };
  uint64_t CallerSize = CallerEnd - CallerBegin;
  if (Mutation == FunctionTableElfMutation::FunctionEndInterior)
    CallerSize += 2;
  else if (Mutation == FunctionTableElfMutation::FunctionSizeOutOfText)
    CallerSize = Text.size() * 2;
  writeSymbol(1, 1, STT_FUNC, 1, TextAddr + CallerBegin, CallerSize);
  writeSymbol(2, 8,
              Mutation == FunctionTableElfMutation::NonFunctionTarget
                  ? STT_OBJECT
                  : STT_FUNC,
              1, Target0, HasCustomCallerEnd ? Target1 - Target0 : MinInstSize);
  writeSymbol(3, 16, STT_FUNC, 1, Target1, TextAddr + Text.size() - Target1);
  writeSymbol(4, 24, STT_OBJECT, 2,
              Mutation == FunctionTableElfMutation::MisalignedTableSymbol
                  ? TableAddr + 1
                  : TableAddr,
              DataSize);

  auto writeRela = [&](size_t Index, uint64_t Offset, uint32_t Type,
                       uint64_t Addend) {
    Elf64_Rela Rela{};
    Rela.r_offset = Offset;
    Rela.setSymbolAndType(/*Symbol=*/0, Type);
    Rela.r_addend = static_cast<int64_t>(Addend);
    std::memcpy(Buf.data() + RelaOff + Index * sizeof(Elf64_Rela), &Rela,
                sizeof(Rela));
  };
  writeRela(0, TableAddr,
            Mutation == FunctionTableElfMutation::WrongRelocationKind
                ? R_AMDGPU_ABS64
                : R_AMDGPU_RELATIVE64,
            Mutation == FunctionTableElfMutation::NonBoundaryTarget
                ? Target0 + 2
                : Target0);
  writeRela(1,
            Mutation == FunctionTableElfMutation::RelocationGap
                ? TableAddr
                : TableAddr + sizeof(uint64_t),
            R_AMDGPU_RELATIVE64, Target1);
  if (Mutation == FunctionTableElfMutation::RelocatedSentinel)
    writeRela(2, TableAddr + 2 * sizeof(uint64_t), R_AMDGPU_RELATIVE64,
              Target0);

  return {std::move(Buf), std::move(Decoded), CallOffset};
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

// -- encodeSetPCLongBranch ---------------------------------------------------

TEST(EncodeSetPCLongBranch, BackwardLandsOnTarget) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  const uint64_t From = 0x81000;
  const uint64_t To = 0x1004;
  std::optional<llvm::SmallVector<uint8_t>> Out =
      encodeSetPCLongBranch(S, From, To, /*SgprBase=*/12);
  ASSERT_TRUE(Out);
  EXPECT_EQ(Out->size(), SetPcReturnReserveBytes);

  std::vector<InternalDecodedInst> Dec;
  ASSERT_TRUE(decodeTextSection(Out->data(), Out->size(), S, Dec));
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

TEST(EncodeSetPCLongBranch, ForwardLandsOnTarget) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  constexpr uint64_t From = 0x1000;
  constexpr uint64_t To = 0x81000;
  std::optional<llvm::SmallVector<uint8_t>> Out =
      encodeSetPCLongBranch(S, From, To, /*SgprBase=*/12);
  ASSERT_TRUE(Out);
  EXPECT_EQ(Out->size(), 16u);

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Out->data(), Out->size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 3u);
  ASSERT_TRUE(Decoded[1].Inst.getOperand(2).isImm());
  uint64_t Delta =
      static_cast<uint64_t>(Decoded[1].Inst.getOperand(2).getImm());
  EXPECT_EQ(From + MinInstSize + Delta, To);
}

TEST(EncodeSetPCLongBranch, UsesVccWhenRequested) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  std::optional<llvm::SmallVector<uint8_t>> Out = encodeSetPCLongBranch(
      S, /*FromOffset=*/0x1000, /*TargetOffset=*/0x81000, /*SgprBase=*/0,
      /*UseVcc=*/true);
  ASSERT_TRUE(Out);

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Out->data(), Out->size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 3u);
  for (const InternalDecodedInst &DI : Decoded) {
    ASSERT_NE(DI.Inst.getNumOperands(), 0u);
    ASSERT_TRUE(DI.Inst.getOperand(0).isReg());
    EXPECT_TRUE(S.MRI->regsOverlap(
        llvm::MCRegister(DI.Inst.getOperand(0).getReg()), S.VCCRegister));
  }
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
       /*FunctionStart=*/0, /*FunctionEnd=*/0x1000}};
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

TEST(FindNearestSetPcGateway, PrependsWave32VccSave) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  std::vector<NopSled> Gateways = {
      {/*Start=*/0x100, /*End=*/0x118, /*WritePos=*/0x100,
       /*FunctionStart=*/0, /*FunctionEnd=*/0x1000}};
  llvm::Expected<std::optional<EncodedSetPcGateway>> GatewayOrErr =
      findNearestSetPcGateway(
          Gateways, S, /*FromOffset=*/0, /*TargetOffset=*/0x81000,
          /*SgprBase=*/105, /*UseVcc=*/true, /*PreserveVcc=*/true);
  ASSERT_TRUE((bool)GatewayOrErr) << llvm::toString(GatewayOrErr.takeError());
  std::optional<EncodedSetPcGateway> &Gateway = *GatewayOrErr;
  ASSERT_TRUE(Gateway);

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Gateway->Bytes.data(), Gateway->Bytes.size(), S,
                                Decoded));
  ASSERT_EQ(Decoded.size(), 4u);
  EXPECT_EQ(Decoded.front().Mnemonic, "s_mov_b32");
  EXPECT_EQ(Decoded[1].Mnemonic, "s_get_pc_i64");
  EXPECT_EQ(Decoded.back().Mnemonic, "s_set_pc_i64");
}

TEST(FindNearestSetPcGateway, SkipsNearerUndersizedCandidate) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  std::vector<NopSled> Gateways = {
      {/*Start=*/0x80100, /*End=*/0x80110, /*WritePos=*/0x80100,
       /*FunctionStart=*/0, /*FunctionEnd=*/0x100000},
      {/*Start=*/0x80200, /*End=*/0x80214, /*WritePos=*/0x80200,
       /*FunctionStart=*/0, /*FunctionEnd=*/0x100000}};
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
       /*FunctionStart=*/0, /*FunctionEnd=*/0x1000}};
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

TEST(FindNearestSetPcGateway, AnalyticalWidthsMatchEncodedBoundaries) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  struct WidthCase {
    uint64_t Delta;
    uint32_t ExpectedSize;
  };
  constexpr WidthCase Cases[] = {
      {static_cast<uint64_t>(-16), 12},
      {0, 12},
      {64, 12},
      {65, 16},
      {static_cast<uint64_t>(std::numeric_limits<int32_t>::max()), 16},
      {static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) + 1, 20},
      {static_cast<uint64_t>(-17), 20},
      {0x3ff0000000000000ULL, 12},
      {0xbff0000000000000ULL, 12},
      {0x3fe0000000000000ULL, 12},
      {0xbfe0000000000000ULL, 12},
      {0x4000000000000000ULL, 12},
      {0xc000000000000000ULL, 12},
      {0x4010000000000000ULL, 12},
      {0xc010000000000000ULL, 12},
      {0x3fc45f306dc9c882ULL, 12},
  };

  constexpr uint64_t GatewayOffset = 0x100;
  constexpr uint64_t PcBase = GatewayOffset + MinInstSize;
  for (const WidthCase &C : Cases) {
    SCOPED_TRACE("delta=0x" + llvm::utohexstr(C.Delta));
    uint64_t TargetOffset = PcBase + C.Delta;
    std::optional<llvm::SmallVector<uint8_t>> Encoded =
        encodeSetPCLongBranch(S, GatewayOffset, TargetOffset, /*SgprBase=*/12);
    ASSERT_TRUE(Encoded);
    ASSERT_EQ(Encoded->size(), C.ExpectedSize);

    // Give the candidate exactly the space required by the real encoding.
    // An analytical overestimate rejects the candidate; an underestimate is
    // rejected by findNearestSetPcGateway's post-encode consistency check.
    std::vector<NopSled> Gateways = {
        {/*Start=*/GatewayOffset,
         /*End=*/GatewayOffset + Encoded->size(),
         /*WritePos=*/GatewayOffset,
         /*FunctionStart=*/0,
         /*FunctionEnd=*/std::numeric_limits<uint64_t>::max()}};
    llvm::Expected<std::optional<EncodedSetPcGateway>> GatewayOrErr =
        findNearestSetPcGateway(Gateways, S, /*FromOffset=*/0, TargetOffset,
                                /*SgprBase=*/12);
    ASSERT_TRUE((bool)GatewayOrErr) << llvm::toString(GatewayOrErr.takeError());
    ASSERT_TRUE(*GatewayOrErr);
    EXPECT_EQ((*GatewayOrErr)->Bytes.size(), Encoded->size());
  }
}

TEST(FindSplitVccGateway, UsesDisjointEightAndSixteenByteSleds) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  std::vector<NopSled> Gateways = {
      {/*Start=*/0x100, /*End=*/0x108, /*WritePos=*/0x100,
       /*FunctionStart=*/0, /*FunctionEnd=*/0x1000},
      {/*Start=*/0x200, /*End=*/0x210, /*WritePos=*/0x200,
       /*FunctionStart=*/0, /*FunctionEnd=*/0x1000,
       /*GatewayOnly=*/true}};
  std::optional<EncodedSplitVccGateway> Split =
      findSplitVccGateway(Gateways, S, /*FromOffset=*/0,
                          /*TargetOffset=*/0x81000, /*SaveSgpr=*/105);
  ASSERT_TRUE(Split);
  EXPECT_EQ(Split->PrimaryIndex, 0u);
  EXPECT_EQ(Split->SecondaryIndex, 1u);
  EXPECT_EQ(Split->PrimaryBytes.size(), 8u);
  EXPECT_EQ(Split->SecondaryBytes.size(), 16u);
  EXPECT_EQ(Gateways[0].WritePos, 0x100u);
  EXPECT_EQ(Gateways[1].WritePos, 0x200u);

  std::vector<InternalDecodedInst> Primary;
  ASSERT_TRUE(decodeTextSection(Split->PrimaryBytes.data(),
                                Split->PrimaryBytes.size(), S, Primary));
  ASSERT_EQ(Primary.size(), 2u);
  EXPECT_EQ(Primary[0].Mnemonic, "s_mov_b32");
  ASSERT_GE(Primary[0].Inst.getNumOperands(), 2u);
  ASSERT_TRUE(Primary[0].Inst.getOperand(0).isReg());
  ASSERT_TRUE(Primary[0].Inst.getOperand(1).isReg());
  EXPECT_STREQ(S.MRI->getName(Primary[0].Inst.getOperand(0).getReg()),
               "SGPR105");
  EXPECT_TRUE(S.MRI->regsOverlap(
      llvm::MCRegister(Primary[0].Inst.getOperand(1).getReg()), S.VCCRegister));
  EXPECT_EQ(Primary[1].Mnemonic, "s_branch");
  EXPECT_EQ(static_cast<int16_t>(
                readDword(Split->PrimaryBytes.data() + MinInstSize) & 0xFFFFu),
            62);

  std::vector<InternalDecodedInst> Secondary;
  ASSERT_TRUE(decodeTextSection(Split->SecondaryBytes.data(),
                                Split->SecondaryBytes.size(), S, Secondary));
  ASSERT_EQ(Secondary.size(), 3u);
  EXPECT_EQ(Secondary[0].Mnemonic, "s_get_pc_i64");
  EXPECT_EQ(Secondary[1].Mnemonic, "s_add_nc_u64");
  EXPECT_EQ(Secondary[2].Mnemonic, "s_set_pc_i64");
  for (const InternalDecodedInst &DI : Secondary) {
    ASSERT_NE(DI.Inst.getNumOperands(), 0u);
    ASSERT_TRUE(DI.Inst.getOperand(0).isReg());
    EXPECT_TRUE(S.MRI->regsOverlap(
        llvm::MCRegister(DI.Inst.getOperand(0).getReg()), S.VCCRegister));
  }
  ASSERT_TRUE(Secondary[1].Inst.getOperand(2).isImm());
  uint64_t Delta =
      static_cast<uint64_t>(Secondary[1].Inst.getOperand(2).getImm());
  EXPECT_EQ(0x200u + MinInstSize + Delta, 0x81000u);
}

TEST(FindSplitVccGateway, RejectsPhysicalOverlapWithoutMutation) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  std::vector<NopSled> Gateways = {
      {/*Start=*/0x100, /*End=*/0x108, /*WritePos=*/0x100,
       /*FunctionStart=*/0, /*FunctionEnd=*/0x1000},
      {/*Start=*/0x104, /*End=*/0x114, /*WritePos=*/0x104,
       /*FunctionStart=*/0, /*FunctionEnd=*/0x1000,
       /*GatewayOnly=*/true}};
  EXPECT_FALSE(findSplitVccGateway(Gateways, S, /*FromOffset=*/0,
                                   /*TargetOffset=*/0x81000,
                                   /*SaveSgpr=*/105));
  EXPECT_EQ(Gateways[0].WritePos, 0x100u);
  EXPECT_EQ(Gateways[1].WritePos, 0x104u);
}

TEST(FindSplitVccGateway, RejectsGatewayOnlyPrimaryWithoutMutation) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  std::vector<NopSled> Gateways = {
      {/*Start=*/0x100, /*End=*/0x108, /*WritePos=*/0x100,
       /*FunctionStart=*/0, /*FunctionEnd=*/0x1000,
       /*GatewayOnly=*/true},
      {/*Start=*/0x200, /*End=*/0x210, /*WritePos=*/0x200,
       /*FunctionStart=*/0, /*FunctionEnd=*/0x1000}};
  EXPECT_FALSE(findSplitVccGateway(Gateways, S, /*FromOffset=*/0,
                                   /*TargetOffset=*/0x81000,
                                   /*SaveSgpr=*/105));
  EXPECT_EQ(Gateways[0].WritePos, 0x100u);
  EXPECT_EQ(Gateways[1].WritePos, 0x200u);
}

TEST(CountReachableSetPcGatewaySlots, DistinguishesZeroFromEncodingFailure) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  std::vector<NopSled> Gateways = {
      {/*Start=*/0x100, /*End=*/0x108, /*WritePos=*/0x100,
       /*FunctionStart=*/0, /*FunctionEnd=*/0x1000}};
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
  EXPECT_NE(Error.find("invalid set-PC gateway while counting"),
            std::string::npos);
}

TEST(CountReachableSetPcGatewaySlots, UsesExactWidthsWithoutEncoding) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  std::vector<NopSled> Gateways = {
      {/*Start=*/0x100, /*End=*/0x130, /*WritePos=*/0x100,
       /*FunctionStart=*/0, /*FunctionEnd=*/0x1000}};
  llvm::Expected<uint64_t> Slots = countReachableSetPcGatewaySlots(
      Gateways, S, /*FromOffset=*/0, /*TargetOffset=*/0x108,
      /*SgprBase=*/12, /*MaxSlots=*/8);
  ASSERT_TRUE((bool)Slots) << llvm::toString(Slots.takeError());
  EXPECT_EQ(*Slots, 3u);
}

TEST(EncodeSetPCLongBranch, RejectsPcBaseOverflow) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  EXPECT_FALSE(encodeSetPCLongBranch(
      S, std::numeric_limits<uint64_t>::max() - MinInstSize + 1, 0,
      /*SgprBase=*/12));
}

TEST(EncodeSetPCLongBranch, RejectsMisalignedScratchPair) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  EXPECT_FALSE(encodeSetPCLongBranch(S, 0, 0x1000, /*SgprBase=*/3));
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

  // Body layout: global_prefetch_b8, v_nop, s_get_pc_i64, s_add_co_u32 (delta
  // lo), s_add_co_ci_u32 (delta hi), s_set_pc_i64.
  EXPECT_EQ(Dec[0].Inst.getOpcode(), S.GlobalPrefetchB8Opcode);
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
      "global_prefetch_b8 v0, s[0:1] scope:SCOPE_SE",
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

// A kernel whose live SGPR count leaves no aligned scratch pair below MaxSgprs
// must decline cleanly (nullopt), never clobber a live SGPR or crash. This is
// the correctness guarantee the per-kernel scratch allocation adds over a fixed
// pair: MetadataSgprCount is set to the top of the addressable range.
TEST(KernelEntryTrampolineFast, DeclinesWhenNoScratchPairFits) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  llvm::SmallVector<uint8_t> EndPgm = assembleSingleInst("s_endpgm", S);
  ASSERT_EQ(EndPgm.size(), MinInstSize);
  llvm::SmallVector<uint8_t> Text(EndPgm.begin(), EndPgm.end());

  comgr_test::KernelDescriptorElfOptions Opts;
  // 106 SGPRs used: no aligned pair fits below the 106-SGPR gfx1250 limit.
  Opts.MetadataSgprCount = 106;
  comgr_test::KernelDescriptorElf Obj =
      comgr_test::makeKernelDescriptorElf(Text, Opts);
  llvm::Expected<ElfView> View =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)View) << llvm::toString(View.takeError());

  std::vector<Trampoline> Growth;
  std::vector<KernelEntryTrampolineFixup> Fixups;
  std::optional<uint32_t> Count = appendKernelEntryTrampolinesFast(
      *View, "gfx1250", /*MaxSgprs=*/106, Growth, Fixups);
  EXPECT_FALSE(Count.has_value());
}

// The complement of the decline case: a modest SGPR count leaves room, so the
// fast path installs one trampoline and records the bumped scratch pair in the
// fixup (SkipSgprReservation=false), exactly like the MC path.
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
  // Scratch pair is s[8:9]; the fixup records the top of the pair (base + 2).
  EXPECT_EQ(Fixups[0].RequiredSgprs, 10u);
  EXPECT_FALSE(Fixups[0].SkipSgprReservation);
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

TEST(SharedRelayTailCanReach, UsesTheTailInstructionAsBranchOrigin) {
  constexpr uint64_t Source = 0x42E190;
  constexpr uint64_t SourceReachableOnly = 0x40E194;
  constexpr uint64_t TailReachable = 0x40E198;

  EXPECT_TRUE(isSBranchReachable(Source, SourceReachableOnly));
  EXPECT_FALSE(sharedRelayTailCanReach(Source, SourceReachableOnly));
  EXPECT_TRUE(sharedRelayTailCanReach(Source, TailReachable));
  EXPECT_FALSE(sharedRelayTailCanReach(std::numeric_limits<uint64_t>::max() -
                                           MinInstSize + 1,
                                       /*RouteOffset=*/0));
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

TEST(CanonicalAbiFrame, RequiresExactWritelaneTiedInput) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  std::vector<InternalDecodedInst> Decoded = decodeAsmSequence(
      S, llvm::ArrayRef<llvm::StringRef>(
             {"v_writelane_b32 v40, s30, 0", "v_writelane_b32 v41, s30, 0"}));
  ASSERT_EQ(Decoded.size(), 2u);
  InternalDecodedInst Write = Decoded[0];
  ASSERT_EQ(Write.Inst.getNumOperands(), 4u);
  llvm::MCRegister V40 = Write.Inst.getOperand(0).getReg();
  llvm::MCRegister S30 = Write.Inst.getOperand(1).getReg();
  EXPECT_TRUE(
      matchesCanonicalLaneTransfer(Write, "v_writelane_b32", V40, S30, 0));

  Write.Inst.erase(std::prev(Write.Inst.end()));
  EXPECT_FALSE(
      matchesCanonicalLaneTransfer(Write, "v_writelane_b32", V40, S30, 0));

  Write = Decoded[0];
  llvm::MCRegister V41 = Decoded[1].Inst.getOperand(0).getReg();
  Write.Inst.getOperand(3).setReg(V41);
  EXPECT_FALSE(
      matchesCanonicalLaneTransfer(Write, "v_writelane_b32", V40, S30, 0));
}

TEST(CanonicalAbiFrame, DistinguishesSaveExecFromPcTransfer) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  std::vector<InternalDecodedInst> Decoded =
      decodeAsmSequence(S, llvm::ArrayRef<llvm::StringRef>(
                               {"s_or_saveexec_b32 s0, s1", "s_branch 0"}));
  ASSERT_EQ(Decoded.size(), 2u);
  EXPECT_FALSE(isTruePcTransfer(Decoded[0], S));
  EXPECT_TRUE(isTruePcTransfer(Decoded[1], S));
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
  EXPECT_TRUE(Info->HasUnboundedIndirectEntries);
  EXPECT_TRUE(Info->HasUnresolvedTargets);
}

TEST(CollectDirectBranchTargets,
     LeavesDynamicallyIndexedRelocationBackedFunctionTableUnresolved) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  FunctionTableTestElf Obj = makeFunctionTableTestElf(
      S, "s_load_b64 s[0:1], s[4:5], s2 offset:0 scale_offset nv");
  llvm::Expected<ElfView> View =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)View) << llvm::toString(View.takeError());

  std::vector<ElfView::FunctionTextRange> Ranges = View->functionTextRanges();
  llvm::SmallVector<uint64_t, 4> Entries;
  for (const ElfView::FunctionTextRange &Range : Ranges)
    Entries.push_back(Range.Begin - View->textAddr());
  std::optional<DirectControlFlowInfo> Info = collectDirectBranchTargets(
      Obj.Decoded, S, View->textAddr(), View->textSize(), Entries, Ranges,
      /*ExternalEntries=*/{},
      llvm::ArrayRef<uint8_t>(View->textData(), View->textSize()), &*View);
  ASSERT_TRUE(Info);
  EXPECT_FALSE(Info->BoundedIndirectTransfers.contains(Obj.CallOffset));
  EXPECT_TRUE(Info->HasUnboundedIndirectEntries);
  EXPECT_TRUE(Info->HasUnresolvedTargets);
}

TEST(CollectDirectBranchTargets,
     RejectsMalformedRelocationBackedFunctionPointerTables) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  struct Case {
    llvm::StringLiteral Load;
    FunctionTableElfMutation Mutation;
    uint64_t TableDelta;
  };
  const Case Cases[] = {
      // Exact base provenance is mandatory.
      {"s_load_b64 s[0:1], s[4:5], s2 offset:0 scale_offset nv",
       FunctionTableElfMutation::None, 0xFF4},
      // Only aligned, zero-immediate, b64 element-index scaling is accepted.
      {"s_load_b64 s[0:1], s[4:5], s2 offset:0 nv",
       FunctionTableElfMutation::None, 0xFFC},
      {"s_load_b64 s[0:1], s[4:5], s2 offset:8 scale_offset nv",
       FunctionTableElfMutation::None, 0xFFC},
      // The symbol must describe immutable RELRO data with a complete,
      // zero-filled RELATIVE64 layout and one unrelocated trailing sentinel.
      {"s_load_b64 s[0:1], s[4:5], s2 offset:0 scale_offset nv",
       FunctionTableElfMutation::NoRelro, 0xFFC},
      {"s_load_b64 s[0:1], s[4:5], s2 offset:0 scale_offset nv",
       FunctionTableElfMutation::RelocationGap, 0xFFC},
      {"s_load_b64 s[0:1], s[4:5], s2 offset:0 scale_offset nv",
       FunctionTableElfMutation::WrongRelocationKind, 0xFFC},
      {"s_load_b64 s[0:1], s[4:5], s2 offset:0 scale_offset nv",
       FunctionTableElfMutation::RelocatedSentinel, 0xFFC},
      {"s_load_b64 s[0:1], s[4:5], s2 offset:0 scale_offset nv",
       FunctionTableElfMutation::NonZeroSlot, 0xFFC},
      {"s_load_b64 s[0:1], s[4:5], s2 offset:0 scale_offset nv",
       FunctionTableElfMutation::MisalignedTableSymbol, 0xFFC},
      // Every addend must name both a defined STT_FUNC and a decoded boundary.
      {"s_load_b64 s[0:1], s[4:5], s2 offset:0 scale_offset nv",
       FunctionTableElfMutation::NonFunctionTarget, 0xFFC},
      {"s_load_b64 s[0:1], s[4:5], s2 offset:0 scale_offset nv",
       FunctionTableElfMutation::NonBoundaryTarget, 0xFFC},
  };

  for (const Case &C : Cases) {
    FunctionTableTestElf Obj =
        makeFunctionTableTestElf(S, C.Load, C.Mutation, C.TableDelta);
    llvm::Expected<ElfView> View =
        ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
    ASSERT_TRUE((bool)View)
        << C.Load.str() << ": " << llvm::toString(View.takeError());
    std::vector<ElfView::FunctionTextRange> Ranges = View->functionTextRanges();
    llvm::SmallVector<uint64_t, 4> Entries;
    for (const ElfView::FunctionTextRange &Range : Ranges)
      Entries.push_back(Range.Begin - View->textAddr());
    std::optional<DirectControlFlowInfo> Info = collectDirectBranchTargets(
        Obj.Decoded, S, View->textAddr(), View->textSize(), Entries, Ranges,
        /*ExternalEntries=*/{},
        llvm::ArrayRef<uint8_t>(View->textData(), View->textSize()), &*View);
    ASSERT_TRUE(Info) << C.Load.str();
    EXPECT_FALSE(Info->BoundedIndirectTransfers.contains(Obj.CallOffset))
        << C.Load.str();
    EXPECT_TRUE(Info->HasUnboundedIndirectEntries) << C.Load.str();
    EXPECT_TRUE(Info->HasUnresolvedTargets) << C.Load.str();
  }
}

TEST(ElfView, MarksFunctionRangesIncompleteOnMalformedSymbolTable) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  FunctionTableTestElf Obj = makeFunctionTableTestElf(
      S, "s_load_b64 s[0:1], s[4:5], s2 offset:0 scale_offset nv",
      FunctionTableElfMutation::MalformedSymbolTable);
  llvm::Expected<ElfView> View =
      ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
  ASSERT_TRUE((bool)View) << llvm::toString(View.takeError());
  EXPECT_TRUE(View->functionTextRanges().empty());
  EXPECT_FALSE(View->functionTextRangesComplete());
}

TEST(CollectDirectBranchTargets,
     BoundsSplitVgprCanonicalFrameWithLongEpilogue) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  auto MakeFrame =
      [](llvm::StringRef BeforeSaves, llvm::StringRef AfterCall,
         llvm::StringRef AfterRestores = {}, llvm::StringRef SavedLow = "v41",
         llvm::StringRef SavedHigh = "v42", llvm::StringRef BeforeCall = {}) {
        std::string Asm = BeforeSaves.str();
        Asm += "s_or_saveexec_b32 s0, -1\n";
        Asm += "v_writelane_b32 " + SavedLow.str() + ", s30, 31\n";
        Asm += "v_writelane_b32 " + SavedHigh.str() + ", s31, 0\n";
        Asm += BeforeCall.str();
        Asm += "s_swap_pc_i64 s[30:31], s[2:3]\n";
        Asm += AfterCall.str();
        Asm += "v_readlane_b32 s30, " + SavedLow.str() + ", 31\n";
        Asm += "v_readlane_b32 s31, " + SavedHigh.str() + ", 0\n";
        for (unsigned I = 0; I != 70; ++I)
          Asm += "s_nop 0\n";
        Asm += AfterRestores.str();
        Asm += "s_set_pc_i64 s[30:31]\n"
               "s_endpgm\n"
               "s_endpgm\n";
        return Asm;
      };

  auto Audit =
      [&](llvm::StringRef Asm,
          llvm::ArrayRef<uint64_t> NonCallEntries = llvm::ArrayRef<uint64_t>{},
          size_t CallerBeginIndex = 0,
          FunctionTableElfMutation Mutation = FunctionTableElfMutation::None,
          size_t CallerEndIndex = std::numeric_limits<size_t>::max(),
          llvm::ArrayRef<uint64_t> ExternalEntries = llvm::ArrayRef<uint64_t>{},
          size_t Target1BeginIndex = std::numeric_limits<size_t>::max()) {
        FunctionTableTestElf Obj = makeFunctionTableTestElf(
            S, /*Load=*/"", Mutation,
            /*TableDelta=*/0xFFC, Asm, CallerBeginIndex, CallerEndIndex,
            Target1BeginIndex);
        llvm::Expected<ElfView> View =
            ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
        EXPECT_TRUE((bool)View) << llvm::toString(View.takeError());
        if (!View)
          return std::optional<DirectControlFlowInfo>();
        std::vector<ElfView::FunctionTextRange> Ranges =
            View->functionTextRanges();
        llvm::SmallVector<uint64_t, 4> Entries;
        for (const ElfView::FunctionTextRange &Range : Ranges)
          Entries.push_back(Range.Begin - View->textAddr());
        return collectDirectBranchTargets(
            Obj.Decoded, S, View->textAddr(), View->textSize(), Entries, Ranges,
            ExternalEntries,
            llvm::ArrayRef<uint8_t>(View->textData(), View->textSize()), &*View,
            NonCallEntries);
      };

  std::optional<DirectControlFlowInfo> Valid = Audit(MakeFrame("", ""));
  ASSERT_TRUE(Valid);
  EXPECT_FALSE(Valid->HasUnresolvedTargets);
  EXPECT_FALSE(Valid->HasUnboundedIndirectEntries);

  // A compiler may place a call in the preceding local function immediately
  // before a fallthrough entry. The call's s30 continuation is a valid
  // incoming link only when no entry bypasses it and the gap preserves it.
  std::string PrefixFrame = "s_swap_pc_i64 s[30:31], s[2:3]\n"
                            "s_nop 0\n" +
                            MakeFrame("", "");
  std::optional<DirectControlFlowInfo> PrefixFallthrough =
      Audit(PrefixFrame, {}, 0, FunctionTableElfMutation::None,
            /*CallerEndIndex=*/2);
  ASSERT_TRUE(PrefixFallthrough);
  EXPECT_FALSE(PrefixFallthrough->HasUnresolvedTargets);
  EXPECT_FALSE(PrefixFallthrough->HasUnboundedIndirectEntries);

  const uint64_t PrefixGapEntry[] = {4};
  std::optional<DirectControlFlowInfo> PrefixRootBypass =
      Audit(PrefixFrame, {}, 0, FunctionTableElfMutation::None,
            /*CallerEndIndex=*/2, PrefixGapEntry);
  ASSERT_TRUE(PrefixRootBypass);
  EXPECT_TRUE(PrefixRootBypass->HasUnresolvedTargets);
  EXPECT_TRUE(PrefixRootBypass->HasUnboundedIndirectEntries);

  std::string PrefixDirectBypass = "s_cbranch_vccnz 1\n"
                                   "s_swap_pc_i64 s[30:31], s[2:3]\n"
                                   "s_nop 0\n" +
                                   MakeFrame("", "");
  std::optional<DirectControlFlowInfo> PrefixDirect =
      Audit(PrefixDirectBypass, {}, 0, FunctionTableElfMutation::None,
            /*CallerEndIndex=*/3);
  ASSERT_TRUE(PrefixDirect);
  EXPECT_TRUE(PrefixDirect->HasUnresolvedTargets);
  EXPECT_TRUE(PrefixDirect->HasUnboundedIndirectEntries);

  std::string PrefixExactBypass = "s_cbranch_vccnz 3\n"
                                  "s_get_pc_i64 s[4:5]\n"
                                  "s_add_nc_u64 s[4:5], s[4:5], 12\n"
                                  "s_set_pc_i64 s[4:5]\n"
                                  "s_swap_pc_i64 s[30:31], s[2:3]\n"
                                  "s_nop 0\n" +
                                  MakeFrame("", "");
  std::optional<DirectControlFlowInfo> PrefixExact =
      Audit(PrefixExactBypass, {}, 0, FunctionTableElfMutation::None,
            /*CallerEndIndex=*/6);
  ASSERT_TRUE(PrefixExact);
  EXPECT_TRUE(PrefixExact->HasUnresolvedTargets);
  EXPECT_TRUE(PrefixExact->HasUnboundedIndirectEntries);

  std::string PrefixForeignCall = "s_swap_pc_i64 s[30:31], s[2:3]\n"
                                  "s_swap_pc_i64 s[4:5], s[6:7]\n" +
                                  MakeFrame("", "");
  std::optional<DirectControlFlowInfo> PrefixForeign =
      Audit(PrefixForeignCall, {}, 0, FunctionTableElfMutation::None,
            /*CallerEndIndex=*/2);
  ASSERT_TRUE(PrefixForeign);
  EXPECT_TRUE(PrefixForeign->HasUnresolvedTargets);
  EXPECT_TRUE(PrefixForeign->HasUnboundedIndirectEntries);

  std::string PrefixLinkClobber = "s_swap_pc_i64 s[30:31], s[2:3]\n"
                                  "s_mov_b32 s30, 0\n" +
                                  MakeFrame("", "");
  std::optional<DirectControlFlowInfo> PrefixClobber =
      Audit(PrefixLinkClobber, {}, 0, FunctionTableElfMutation::None,
            /*CallerEndIndex=*/2);
  ASSERT_TRUE(PrefixClobber);
  EXPECT_TRUE(PrefixClobber->HasUnresolvedTargets);
  EXPECT_TRUE(PrefixClobber->HasUnboundedIndirectEntries);

  // A leaf can use the ABI link SGPRs as scratch after saving them even though
  // it performs no nested call itself. Exercise that production frame as a
  // second local STT_FUNC in an object whose first function has an opaque call.
  std::string CallerFrame = MakeFrame("", "");
  llvm::SmallVector<uint8_t> CallerBytes = assembleInstructions(CallerFrame, S);
  std::vector<InternalDecodedInst> CallerDecoded;
  ASSERT_TRUE(decodeTextSection(CallerBytes.data(), CallerBytes.size(), S,
                                CallerDecoded));
  std::string LeafFrame = "v_writelane_b32 v43, s30, 4\n"
                          "v_writelane_b32 v44, s31, 5\n"
                          "s_mov_b32 s30, 0\n"
                          "s_mov_b32 s31, 0\n"
                          "v_readlane_b32 s30, v43, 4\n"
                          "v_readlane_b32 s31, v44, 5\n"
                          "s_set_pc_i64 s[30:31]\n"
                          "s_endpgm\n";
  std::optional<DirectControlFlowInfo> LeafWithLinkScratch =
      Audit(CallerFrame + LeafFrame, {}, 0, FunctionTableElfMutation::None,
            CallerDecoded.size());
  ASSERT_TRUE(LeafWithLinkScratch);
  EXPECT_FALSE(LeafWithLinkScratch->HasUnresolvedTargets);
  EXPECT_FALSE(LeafWithLinkScratch->HasUnboundedIndirectEntries);

  // A leaf frame has no nested callee that can clobber a caller-saved carrier.
  // Exact lane saves, restores, and the existing body-write proof are enough.
  std::string CallerSavedLeafFrame = "v_writelane_b32 v17, s30, 4\n"
                                     "v_writelane_b32 v17, s31, 5\n"
                                     "v_readlane_b32 s30, v17, 4\n"
                                     "v_readlane_b32 s31, v17, 5\n"
                                     "s_set_pc_i64 s[30:31]\n"
                                     "s_endpgm\n";
  std::optional<DirectControlFlowInfo> CallerSavedLeaf =
      Audit(CallerFrame + CallerSavedLeafFrame, {}, 0,
            FunctionTableElfMutation::None, CallerDecoded.size());
  ASSERT_TRUE(CallerSavedLeaf);
  EXPECT_FALSE(CallerSavedLeaf->HasUnresolvedTargets);
  EXPECT_FALSE(CallerSavedLeaf->HasUnboundedIndirectEntries);

  // The same canonical leaf can be reached only by an exact materialized
  // singleton call. This is a machine-level closure, not the opaque-call ABI
  // fallback: the call target, canonical return, and continuation must be
  // validated together.
  std::string ExactCaller = "s_get_pc_i64 s[0:1]\n"
                            "s_add_nc_u64 s[0:1], s[0:1], 12\n"
                            "s_swap_pc_i64 s[30:31], s[0:1]\n"
                            "s_endpgm\n";
  std::optional<DirectControlFlowInfo> ExactCanonicalLeaf =
      Audit(ExactCaller + LeafFrame, {}, 0, FunctionTableElfMutation::None,
            /*CallerEndIndex=*/4);
  ASSERT_TRUE(ExactCanonicalLeaf);
  EXPECT_FALSE(ExactCanonicalLeaf->HasUnresolvedTargets);
  EXPECT_FALSE(ExactCanonicalLeaf->HasUnboundedIndirectEntries);

  // A separate finite call enters the add of the exact singleton call above,
  // bypassing its defining get-PC. The canonical callee frame remains valid,
  // but the exact closure must reject this alternate materialization entry
  // rather than using the fallback to publish both calls as bounded.
  std::string InteriorCaller = "s_call_i64 s[4:5], 2\n"
                               "s_endpgm\n" +
                               ExactCaller + LeafFrame;
  std::optional<DirectControlFlowInfo> InteriorCanonicalLeaf =
      Audit(InteriorCaller, {}, 0, FunctionTableElfMutation::None,
            /*CallerEndIndex=*/2, /*ExternalEntries=*/{},
            /*Target1BeginIndex=*/6);
  ASSERT_TRUE(InteriorCanonicalLeaf);
  EXPECT_TRUE(InteriorCanonicalLeaf->HasUnresolvedTargets);
  EXPECT_TRUE(InteriorCanonicalLeaf->HasUnboundedIndirectEntries);

  // Canonical compiler tail thunk: save the incoming link, materialize a
  // different STT_FUNC entry, restore the link, and jump without replacing
  // s[30:31]. The target returns directly to the thunk's original caller.
  std::string TailThunk = "v_writelane_b32 v43, s30, 4\n"
                          "v_writelane_b32 v44, s31, 5\n"
                          "s_get_pc_i64 s[0:1]\n"
                          "s_add_nc_u64 s[0:1], s[0:1], 24\n"
                          "v_readlane_b32 s30, v43, 4\n"
                          "v_readlane_b32 s31, v44, 5\n"
                          "s_set_pc_i64 s[0:1]\n";
  std::optional<DirectControlFlowInfo> CanonicalTail = Audit(
      TailThunk + MakeFrame("", ""), {}, 0, FunctionTableElfMutation::None,
      /*CallerEndIndex=*/7);
  ASSERT_TRUE(CanonicalTail);
  EXPECT_FALSE(CanonicalTail->HasUnresolvedTargets);
  EXPECT_FALSE(CanonicalTail->HasUnboundedIndirectEntries);

  // A finite exact jump to a defined noreturn function never needs permission
  // to enter an s30-returning frame. Do not require its source to have a
  // canonical save/restore frame merely because the destination is STT_FUNC.
  std::string ExactNoreturnTarget = "s_cbranch_vccnz 3\n"
                                    "s_get_pc_i64 s[0:1]\n"
                                    "s_add_nc_u64 s[0:1], s[0:1], 12\n"
                                    "s_set_pc_i64 s[0:1]\n"
                                    "s_set_pc_i64 s[30:31]\n"
                                    "s_swap_pc_i64 s[30:31], s[2:3]\n"
                                    "s_endpgm\n"
                                    "s_endpgm\n";
  std::optional<DirectControlFlowInfo> NoreturnExact =
      Audit(ExactNoreturnTarget, {}, 0, FunctionTableElfMutation::None,
            /*CallerEndIndex=*/5);
  ASSERT_TRUE(NoreturnExact);
  EXPECT_FALSE(NoreturnExact->HasUnresolvedTargets);
  EXPECT_FALSE(NoreturnExact->HasUnboundedIndirectEntries);

  // Tail-chain certification is intentionally strict: B cannot use A's
  // not-yet-certified entry while Phase 1 proves B -> returning C.
  std::optional<DirectControlFlowInfo> TailChain =
      Audit(TailThunk + TailThunk + MakeFrame("", ""), {}, 0,
            FunctionTableElfMutation::None,
            /*CallerEndIndex=*/7, {},
            /*Target1BeginIndex=*/14);
  ASSERT_TRUE(TailChain);
  EXPECT_TRUE(TailChain->HasUnresolvedTargets);
  EXPECT_TRUE(TailChain->HasUnboundedIndirectEntries);

  // Permissions are keyed by source instruction. One certified entrant must
  // not authorize a second, noncanonical source that targets the same frame.
  std::string SafeTailToThird = "v_writelane_b32 v43, s30, 4\n"
                                "v_writelane_b32 v44, s31, 5\n"
                                "s_get_pc_i64 s[0:1]\n"
                                "s_add_nc_u64 s[0:1], s[0:1], 36\n"
                                "v_readlane_b32 s30, v43, 4\n"
                                "v_readlane_b32 s31, v44, 5\n"
                                "s_set_pc_i64 s[0:1]\n";
  std::string UnsafeTailToThird = "s_get_pc_i64 s[0:1]\n"
                                  "s_add_nc_u64 s[0:1], s[0:1], 8\n"
                                  "s_set_pc_i64 s[0:1]\n";
  std::optional<DirectControlFlowInfo> MixedTailSources =
      Audit(SafeTailToThird + UnsafeTailToThird + MakeFrame("", ""), {}, 0,
            FunctionTableElfMutation::None,
            /*CallerEndIndex=*/7, {},
            /*Target1BeginIndex=*/10);
  ASSERT_TRUE(MixedTailSources);
  EXPECT_TRUE(MixedTailSources->HasUnresolvedTargets);
  EXPECT_TRUE(MixedTailSources->HasUnboundedIndirectEntries);

  std::string ClobberedTail = "v_writelane_b32 v43, s30, 4\n"
                              "v_writelane_b32 v44, s31, 5\n"
                              "s_get_pc_i64 s[0:1]\n"
                              "s_add_nc_u64 s[0:1], s[0:1], 28\n"
                              "v_readlane_b32 s30, v43, 4\n"
                              "v_readlane_b32 s31, v44, 5\n"
                              "s_mov_b32 s30, 0\n"
                              "s_set_pc_i64 s[0:1]\n";
  std::optional<DirectControlFlowInfo> TailLinkClobber = Audit(
      ClobberedTail + MakeFrame("", ""), {}, 0, FunctionTableElfMutation::None,
      /*CallerEndIndex=*/8);
  ASSERT_TRUE(TailLinkClobber);
  EXPECT_TRUE(TailLinkClobber->HasUnresolvedTargets);
  EXPECT_TRUE(TailLinkClobber->HasUnboundedIndirectEntries);

  std::string NonzeroModeTail = "v_writelane_b32 v43, s30, 4\n"
                                "v_writelane_b32 v44, s31, 5\n"
                                "s_get_pc_i64 s[0:1]\n"
                                "s_add_nc_u64 s[0:1], s[0:1], 28\n"
                                "v_readlane_b32 s30, v43, 4\n"
                                "v_readlane_b32 s31, v44, 5\n"
                                "s_set_vgpr_msb 1\n"
                                "s_set_pc_i64 s[0:1]\n";
  std::optional<DirectControlFlowInfo> TailModeClobber =
      Audit(NonzeroModeTail + MakeFrame("", ""), {}, 0,
            FunctionTableElfMutation::None,
            /*CallerEndIndex=*/8);
  ASSERT_TRUE(TailModeClobber);
  EXPECT_TRUE(TailModeClobber->HasUnresolvedTargets);
  EXPECT_TRUE(TailModeClobber->HasUnboundedIndirectEntries);

  std::string MissingTailRestore = "v_writelane_b32 v43, s30, 4\n"
                                   "v_writelane_b32 v44, s31, 5\n"
                                   "s_get_pc_i64 s[0:1]\n"
                                   "s_add_nc_u64 s[0:1], s[0:1], 16\n"
                                   "v_readlane_b32 s30, v43, 4\n"
                                   "s_set_pc_i64 s[0:1]\n";
  std::optional<DirectControlFlowInfo> TailMissingRestore =
      Audit(MissingTailRestore + MakeFrame("", ""), {}, 0,
            FunctionTableElfMutation::None,
            /*CallerEndIndex=*/6);
  ASSERT_TRUE(TailMissingRestore);
  EXPECT_TRUE(TailMissingRestore->HasUnresolvedTargets);
  EXPECT_TRUE(TailMissingRestore->HasUnboundedIndirectEntries);

  std::string NonCsrTail = "v_writelane_b32 v0, s30, 4\n"
                           "v_writelane_b32 v1, s31, 5\n"
                           "s_get_pc_i64 s[0:1]\n"
                           "s_add_nc_u64 s[0:1], s[0:1], 24\n"
                           "v_readlane_b32 s30, v0, 4\n"
                           "v_readlane_b32 s31, v1, 5\n"
                           "s_set_pc_i64 s[0:1]\n";
  std::optional<DirectControlFlowInfo> TailNonCsr = Audit(
      NonCsrTail + MakeFrame("", ""), {}, 0, FunctionTableElfMutation::None,
      /*CallerEndIndex=*/7);
  ASSERT_TRUE(TailNonCsr);
  EXPECT_TRUE(TailNonCsr->HasUnresolvedTargets);
  EXPECT_TRUE(TailNonCsr->HasUnboundedIndirectEntries);

  std::string SavedLaneClobberTail = "v_writelane_b32 v43, s30, 4\n"
                                     "v_writelane_b32 v44, s31, 5\n"
                                     "s_get_pc_i64 s[0:1]\n"
                                     "s_add_nc_u64 s[0:1], s[0:1], 32\n"
                                     "v_writelane_b32 v43, s0, 4\n"
                                     "v_readlane_b32 s30, v43, 4\n"
                                     "v_readlane_b32 s31, v44, 5\n"
                                     "s_set_pc_i64 s[0:1]\n";
  std::optional<DirectControlFlowInfo> TailSavedLaneClobber =
      Audit(SavedLaneClobberTail + MakeFrame("", ""), {}, 0,
            FunctionTableElfMutation::None,
            /*CallerEndIndex=*/8);
  ASSERT_TRUE(TailSavedLaneClobber);
  EXPECT_TRUE(TailSavedLaneClobber->HasUnresolvedTargets);
  EXPECT_TRUE(TailSavedLaneClobber->HasUnboundedIndirectEntries);

  std::string InteriorTargetTail = "v_writelane_b32 v43, s30, 4\n"
                                   "v_writelane_b32 v44, s31, 5\n"
                                   "s_get_pc_i64 s[0:1]\n"
                                   "s_add_nc_u64 s[0:1], s[0:1], 28\n"
                                   "v_readlane_b32 s30, v43, 4\n"
                                   "v_readlane_b32 s31, v44, 5\n"
                                   "s_set_pc_i64 s[0:1]\n";
  std::optional<DirectControlFlowInfo> TailInteriorTarget =
      Audit(InteriorTargetTail + MakeFrame("", ""), {}, 0,
            FunctionTableElfMutation::None,
            /*CallerEndIndex=*/7);
  ASSERT_TRUE(TailInteriorTarget);
  EXPECT_TRUE(TailInteriorTarget->HasUnresolvedTargets);
  EXPECT_TRUE(TailInteriorTarget->HasUnboundedIndirectEntries);

  const uint64_t TailSaveBypassEntry[] = {16};
  std::optional<DirectControlFlowInfo> TailSaveBypass = Audit(
      TailThunk + MakeFrame("", ""), {}, 0, FunctionTableElfMutation::None,
      /*CallerEndIndex=*/7, TailSaveBypassEntry);
  ASSERT_TRUE(TailSaveBypass);
  EXPECT_TRUE(TailSaveBypass->HasUnresolvedTargets);
  EXPECT_TRUE(TailSaveBypass->HasUnboundedIndirectEntries);

  const uint64_t TailNonCallEntry[] = {0};
  std::optional<DirectControlFlowInfo> TailNonCallRoot =
      Audit(TailThunk + MakeFrame("", ""), TailNonCallEntry, 0,
            FunctionTableElfMutation::None,
            /*CallerEndIndex=*/7);
  ASSERT_TRUE(TailNonCallRoot);
  EXPECT_TRUE(TailNonCallRoot->HasUnresolvedTargets);
  EXPECT_TRUE(TailNonCallRoot->HasUnboundedIndirectEntries);

  // A loop wholly inside the function is not a new entry, including when its
  // backedge targets the function's first instruction.  The two saves still
  // dominate the nested call.
  std::optional<DirectControlFlowInfo> InternalBackedge =
      Audit(MakeFrame("", "", "", "v41", "v42", "s_cbranch_vccnz -6\n"));
  ASSERT_TRUE(InternalBackedge);
  EXPECT_FALSE(InternalBackedge->HasUnresolvedTargets);
  EXPECT_FALSE(InternalBackedge->HasUnboundedIndirectEntries);

  // Re-entering the prologue after a nested call would overwrite the
  // originally saved incoming link with the call's continuation.
  std::optional<DirectControlFlowInfo> PostCallBackedge =
      Audit(MakeFrame("", "s_branch -8\n"));
  ASSERT_TRUE(PostCallBackedge);
  EXPECT_TRUE(PostCallBackedge->HasUnresolvedTargets);
  EXPECT_TRUE(PostCallBackedge->HasUnboundedIndirectEntries);

  // The must-link fact has to converge around cycles: the lexically early
  // Begin backedge is unsafe when a second edge revisits it after the call.
  std::optional<DirectControlFlowInfo> CyclicPostCallBackedge = Audit(
      MakeFrame("", "s_branch -3\n", "", "v41", "v42", "s_cbranch_vccnz -6\n"));
  ASSERT_TRUE(CyclicPostCallBackedge);
  EXPECT_TRUE(CyclicPostCallBackedge->HasUnresolvedTargets);
  EXPECT_TRUE(CyclicPostCallBackedge->HasUnboundedIndirectEntries);

  std::optional<DirectControlFlowInfo> BodyLinkScratch =
      Audit(MakeFrame("", "s_add_u32 s30, s0, s1\n"));
  ASSERT_TRUE(BodyLinkScratch);
  EXPECT_FALSE(BodyLinkScratch->HasUnresolvedTargets);
  EXPECT_FALSE(BodyLinkScratch->HasUnboundedIndirectEntries);

  // A write to the exact protected lane after the call destroys the saved
  // low link half. A genuine branch before the saves also prevents the
  // prologue from dominating the frame, unlike the save-exec instruction.
  std::optional<DirectControlFlowInfo> Clobbered =
      Audit(MakeFrame("", "v_writelane_b32 v41, s0, 31\n"));
  ASSERT_TRUE(Clobbered);
  EXPECT_TRUE(Clobbered->HasUnresolvedTargets);
  EXPECT_TRUE(Clobbered->HasUnboundedIndirectEntries);

  std::optional<DirectControlFlowInfo> Branched =
      Audit(MakeFrame("s_branch 0\n", ""));
  ASSERT_TRUE(Branched);
  EXPECT_TRUE(Branched->HasUnresolvedTargets);
  EXPECT_TRUE(Branched->HasUnboundedIndirectEntries);

  std::optional<DirectControlFlowInfo> OutsideBranch =
      Audit(MakeFrame("s_branch 0\n", ""), {}, /*CallerBeginIndex=*/1);
  ASSERT_TRUE(OutsideBranch);
  EXPECT_TRUE(OutsideBranch->HasUnresolvedTargets);
  EXPECT_TRUE(OutsideBranch->HasUnboundedIndirectEntries);

  std::string ExternalExactSetPc = "s_get_pc_i64 s[4:5]\n"
                                   "s_add_nc_u64 s[4:5], s[4:5], 28\n"
                                   "s_set_pc_i64 s[4:5]\n" +
                                   MakeFrame("", "");
  std::optional<DirectControlFlowInfo> OutsideExactSetPc =
      Audit(ExternalExactSetPc, {}, /*CallerBeginIndex=*/3);
  ASSERT_TRUE(OutsideExactSetPc);
  EXPECT_TRUE(OutsideExactSetPc->HasUnresolvedTargets);
  EXPECT_TRUE(OutsideExactSetPc->HasUnboundedIndirectEntries);

  std::optional<DirectControlFlowInfo> InteriorFunctionEnd = Audit(
      MakeFrame("", ""), {}, 0, FunctionTableElfMutation::FunctionEndInterior);
  ASSERT_TRUE(InteriorFunctionEnd);
  EXPECT_TRUE(InteriorFunctionEnd->HasUnresolvedTargets);
  EXPECT_TRUE(InteriorFunctionEnd->HasUnboundedIndirectEntries);

  std::optional<DirectControlFlowInfo> OutOfTextFunctionSize =
      Audit(MakeFrame("", ""), {}, 0,
            FunctionTableElfMutation::FunctionSizeOutOfText);
  ASSERT_TRUE(OutOfTextFunctionSize);
  EXPECT_TRUE(OutOfTextFunctionSize->HasUnresolvedTargets);
  EXPECT_TRUE(OutOfTextFunctionSize->HasUnboundedIndirectEntries);

  std::optional<DirectControlFlowInfo> MalformedSymbolTable = Audit(
      MakeFrame("", ""), {}, 0, FunctionTableElfMutation::MalformedSymbolTable);
  ASSERT_TRUE(MalformedSymbolTable);
  EXPECT_TRUE(MalformedSymbolTable->HasUnresolvedTargets);
  EXPECT_TRUE(MalformedSymbolTable->HasUnboundedIndirectEntries);

  std::optional<DirectControlFlowInfo> ClobberedBeforeSave =
      Audit(MakeFrame("s_mov_b32 s30, 0\n", ""));
  ASSERT_TRUE(ClobberedBeforeSave);
  EXPECT_TRUE(ClobberedBeforeSave->HasUnresolvedTargets);
  EXPECT_TRUE(ClobberedBeforeSave->HasUnboundedIndirectEntries);

  const uint64_t NonCallBegin[] = {0};
  std::optional<DirectControlFlowInfo> NonCallRoot =
      Audit(MakeFrame("", ""), NonCallBegin);
  ASSERT_TRUE(NonCallRoot);
  EXPECT_TRUE(NonCallRoot->HasUnresolvedTargets);
  EXPECT_TRUE(NonCallRoot->HasUnboundedIndirectEntries);

  std::optional<DirectControlFlowInfo> CallerMode =
      Audit(MakeFrame("s_set_vgpr_msb 1\n", ""));
  ASSERT_TRUE(CallerMode);
  EXPECT_TRUE(CallerMode->HasUnresolvedTargets);
  EXPECT_TRUE(CallerMode->HasUnboundedIndirectEntries);

  std::optional<DirectControlFlowInfo> ReturnMode =
      Audit(MakeFrame("", "", "s_set_vgpr_msb 1\n"));
  ASSERT_TRUE(ReturnMode);
  EXPECT_TRUE(ReturnMode->HasUnresolvedTargets);
  EXPECT_TRUE(ReturnMode->HasUnboundedIndirectEntries);

  std::optional<DirectControlFlowInfo> ClobberedAfterRestore =
      Audit(MakeFrame("", "", "s_mov_b32 s30, 0\n"));
  ASSERT_TRUE(ClobberedAfterRestore);
  EXPECT_TRUE(ClobberedAfterRestore->HasUnresolvedTargets);
  EXPECT_TRUE(ClobberedAfterRestore->HasUnboundedIndirectEntries);

  std::optional<DirectControlFlowInfo> NonCsr =
      Audit(MakeFrame("", "", "", "v0", "v1"));
  ASSERT_TRUE(NonCsr);
  EXPECT_TRUE(NonCsr->HasUnresolvedTargets);
  EXPECT_TRUE(NonCsr->HasUnboundedIndirectEntries);
}

TEST(CollectDirectBranchTargets,
     BoundsScratchPreservedSelfRecursiveCallerSavedLinkVgpr) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  std::function<std::optional<DirectControlFlowInfo>(llvm::StringRef)> Audit =
      [&](llvm::StringRef Asm) {
        FunctionTableTestElf Obj = makeFunctionTableTestElf(
            S, /*Load=*/"", FunctionTableElfMutation::None,
            /*TableDelta=*/0xFFC, Asm, /*CallerBeginIndex=*/0);
        llvm::Expected<ElfView> View =
            ElfView::create(Obj.Bytes.data(), Obj.Bytes.size());
        EXPECT_TRUE((bool)View) << llvm::toString(View.takeError());
        if (!View)
          return std::optional<DirectControlFlowInfo>();
        std::vector<ElfView::FunctionTextRange> Ranges =
            View->functionTextRanges();
        llvm::SmallVector<uint64_t, 4> Entries;
        for (const ElfView::FunctionTextRange &Range : Ranges)
          Entries.push_back(Range.Begin - View->textAddr());
        return collectDirectBranchTargets(
            Obj.Decoded, S, View->textAddr(), View->textSize(), Entries, Ranges,
            /*ExternalEntries=*/{},
            llvm::ArrayRef<uint8_t>(View->textData(), View->textSize()), &*View,
            /*NonCallEntries=*/{});
      };

  // The get-PC at byte 32 reports byte 36; adding -36 materializes this
  // function's byte-zero entry for the register call at byte 48.
  const std::string ScratchPreservedRecursive =
      "scratch_store_b32 off, v17, s32 nv\n"
      "s_or_saveexec_b32 s0, -1\n"
      "v_writelane_b32 v17, s30, 31\n"
      "v_writelane_b32 v17, s31, 0\n"
      "s_get_pc_i64 s[2:3]\n"
      "s_add_nc_u64 s[2:3], s[2:3], 0xffffffffffffffdc\n"
      "s_swap_pc_i64 s[30:31], s[2:3]\n"
      "v_readlane_b32 s30, v17, 31\n"
      "v_readlane_b32 s31, v17, 0\n"
      "scratch_load_b32 v17, off, s32 nv\n"
      "s_set_pc_i64 s[30:31]\n"
      "s_endpgm\n"
      "s_endpgm\n";
  std::optional<DirectControlFlowInfo> Valid = Audit(ScratchPreservedRecursive);
  ASSERT_TRUE(Valid);
  EXPECT_FALSE(Valid->HasUnresolvedTargets);
  EXPECT_FALSE(Valid->HasUnboundedIndirectEntries);

  std::string MismatchedScratch = ScratchPreservedRecursive;
  size_t ReloadPos =
      MismatchedScratch.find("scratch_load_b32 v17, off, s32 nv");
  ASSERT_NE(ReloadPos, std::string::npos);
  MismatchedScratch.replace(
      ReloadPos, std::string("scratch_load_b32 v17, off, s32 nv").size(),
      "scratch_load_b32 v17, off, s32 offset:4 nv");
  std::optional<DirectControlFlowInfo> Mismatch = Audit(MismatchedScratch);
  ASSERT_TRUE(Mismatch);
  EXPECT_TRUE(Mismatch->HasUnboundedIndirectEntries);

  std::string ClobberedScratch = ScratchPreservedRecursive;
  size_t RestorePos = ClobberedScratch.find("v_readlane_b32 s30, v17, 31");
  ASSERT_NE(RestorePos, std::string::npos);
  ClobberedScratch.insert(RestorePos, "scratch_store_b32 off, v0, s32 nv\n");
  std::optional<DirectControlFlowInfo> Clobbered = Audit(ClobberedScratch);
  ASSERT_TRUE(Clobbered);
  EXPECT_TRUE(Clobbered->HasUnboundedIndirectEntries);
}

TEST(CollectDirectBranchTargets,
     MarksUnboundedSetPcEntryWithoutTreatingItAsUnresolvedCall) {
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
  EXPECT_TRUE(Info->HasUnboundedIndirectEntries);
  EXPECT_FALSE(Info->HasUnresolvedTargets);
}

TEST(CollectDirectBranchTargets,
     HandlesSparseSetPcAndFunctionRangesWithoutCartesianScan) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Bytes =
      assembleSingleInst("s_set_pc_i64 s[8:9]", S);
  ASSERT_FALSE(Bytes.empty());

  std::vector<InternalDecodedInst> Prototype;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Prototype));
  ASSERT_EQ(Prototype.size(), 1u);

  constexpr size_t Count = 16384;
  constexpr uint64_t TextSize = 1 << 20;
  std::vector<InternalDecodedInst> Decoded;
  Decoded.reserve(Count);
  llvm::SmallVector<ElfView::FunctionTextRange, 16> FunctionRanges;
  FunctionRanges.reserve(Count);
  for (size_t I = 0; I != Count; ++I) {
    Decoded.push_back(Prototype[0]);
    Decoded.back().Offset = I * MinInstSize;
    // These validly ordered ranges do not cover any instruction. This pins
    // the sparse return-to-range index: a full range scan for every set-PC
    // would perform Count squared containment checks.
    uint64_t Begin = TextSize + I * MinInstSize;
    FunctionRanges.push_back({Begin, Begin + MinInstSize});
  }

  std::optional<DirectControlFlowInfo> Info =
      collectDirectBranchTargets(Decoded, S, /*TextAddr=*/0, TextSize,
                                 /*DeclaredEntries=*/{}, FunctionRanges);
  ASSERT_TRUE(Info);
  EXPECT_TRUE(Info->Targets.empty());
  EXPECT_TRUE(Info->HasUnboundedIndirectEntries);
  EXPECT_FALSE(Info->HasUnresolvedTargets);
}

TEST(CollectDirectBranchTargets, HandlesManyCallsWithoutCartesianGrouping) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Bytes =
      assembleSingleInst("s_call_i64 s[30:31], 0", S);
  ASSERT_FALSE(Bytes.empty());
  std::vector<InternalDecodedInst> Prototype;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Prototype));
  ASSERT_EQ(Prototype.size(), 1u);

  constexpr size_t Count = 8192;
  std::vector<InternalDecodedInst> Decoded(Count, Prototype.front());
  for (size_t I = 0; I != Count; ++I)
    Decoded[I].Offset = I * MinInstSize;
  const uint64_t TextSize = Count * MinInstSize;
  llvm::SmallVector<uint64_t, 1> DeclaredEntries{0};
  std::optional<DirectControlFlowInfo> Info = collectDirectBranchTargets(
      Decoded, S, /*TextAddr=*/0, TextSize, DeclaredEntries);
  ASSERT_TRUE(Info);
  EXPECT_TRUE(Info->Targets.contains(MinInstSize));
  EXPECT_TRUE(Info->Targets.contains(TextSize));
  EXPECT_FALSE(Info->HasUnboundedIndirectEntries);
  EXPECT_FALSE(Info->HasUnresolvedTargets);
}

TEST(CollectDirectBranchTargets,
     IndexesManyBoundedReturnsAndKnownCallsWithoutCartesianScan) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Bytes =
      assembleInstructions("s_call_i64 s[30:31], 0\n"
                           "s_set_pc_i64 s[30:31]",
                           S);
  ASSERT_FALSE(Bytes.empty());

  std::vector<InternalDecodedInst> Prototype;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Prototype));
  ASSERT_EQ(Prototype.size(), 2u);

  constexpr size_t Count = 16384;
  std::vector<InternalDecodedInst> Decoded;
  Decoded.reserve(Count * 2);
  llvm::SmallVector<uint64_t, 16> DeclaredEntries;
  DeclaredEntries.reserve(Count);
  llvm::SmallVector<ElfView::FunctionTextRange, 16> FunctionRanges;
  FunctionRanges.reserve(Count);
  for (size_t I = 0; I != Count; ++I) {
    uint64_t CallOffset = I * 2 * MinInstSize;
    uint64_t FunctionBegin = CallOffset + MinInstSize;
    Decoded.push_back(Prototype[0]);
    Decoded.back().Offset = CallOffset;
    Decoded.push_back(Prototype[1]);
    Decoded.back().Offset = FunctionBegin;
    DeclaredEntries.push_back(FunctionBegin);
    FunctionRanges.push_back({FunctionBegin, FunctionBegin + MinInstSize});
  }

  // Each call targets the following one-instruction return function. This
  // simultaneously pins target/continuation range queries and direct-call
  // source membership: scanning all calls for every return is quadratic.
  uint64_t TextSize = Count * 2 * MinInstSize;
  std::optional<DirectControlFlowInfo> Info = collectDirectBranchTargets(
      Decoded, S, /*TextAddr=*/0, TextSize, DeclaredEntries, FunctionRanges);
  ASSERT_TRUE(Info);
  EXPECT_EQ(Info->Targets.size(), Count);
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

  // collectDirectBranchTargets normally sees the complete .text decode. Keep
  // the synthetic production slice faithful to that contract by representing
  // the resolved local target as an instruction boundary.
  llvm::SmallVector<uint8_t> TargetBytes = assembleInstructions("s_endpgm", S);
  std::vector<InternalDecodedInst> TargetDecoded;
  ASSERT_TRUE(decodeTextSection(TargetBytes.data(), TargetBytes.size(), S,
                                TargetDecoded));
  ASSERT_EQ(TargetDecoded.size(), 1u);
  Decoded.insert(Decoded.begin(), TargetDecoded.front());

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

TEST(CollectDirectBranchTargets, BoundsFiniteExternalPcMaterializedCall) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Bytes =
      assembleInstructions("s_get_pc_i64 s[0:1]\n"
                           "s_add_nc_u64 s[0:1], s[0:1], 0x100\n"
                           "s_swap_pc_i64 s[30:31], s[0:1]\n"
                           "s_endpgm",
                           S);
  ASSERT_FALSE(Bytes.empty());

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 4u);

  // The exact target is outside this deliberately short .text range. The
  // external callee may return through the link pair, so the local
  // continuation remains a protected entry even though the external target
  // contributes no local offset.
  std::optional<DirectControlFlowInfo> Info =
      collectDirectBranchTargets(Decoded, S, /*TextAddr=*/0,
                                 /*TextSize=*/0x20,
                                 /*DeclaredEntries=*/{});
  ASSERT_TRUE(Info);
  ASSERT_EQ(Info->Targets.size(), 1u);
  EXPECT_TRUE(Info->Targets.contains(Decoded[3].Offset));
  EXPECT_TRUE(Info->BoundedIndirectTransfers.contains(Decoded[2].Offset));
  EXPECT_FALSE(Info->HasUnresolvedTargets);
}

TEST(CollectDirectBranchTargets,
     RejectsExternalCallContinuationIntoReturnFunction) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Bytes =
      assembleInstructions("s_endpgm\n"
                           "s_get_pc_i64 s[2:3]\n"
                           "s_add_co_i32 s4, 0x1000, 4\n"
                           "s_add_co_u32 s2, s2, s4\n"
                           "s_add_co_ci_u32 s3, s3, 0\n"
                           "s_swap_pc_i64 s[30:31], s[2:3]\n"
                           "s_nop 0\n"
                           "s_nop 0\n"
                           "s_set_pc_i64 s[30:31]\n"
                           "s_endpgm\n"
                           "s_get_pc_i64 s[0:1]\n"
                           "s_add_nc_u64 s[0:1], s[0:1], -16\n"
                           "s_swap_pc_i64 s[30:31], s[0:1]",
                           S);
  ASSERT_FALSE(Bytes.empty());

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 13u);
  llvm::SmallVector<uint64_t, 2> DeclaredEntries{Decoded[7].Offset,
                                                 Decoded[10].Offset};
  llvm::SmallVector<ElfView::FunctionTextRange, 3> FunctionRanges{
      {Decoded[1].Offset, Decoded[7].Offset},
      {Decoded[7].Offset, Decoded[9].Offset},
      {Decoded[10].Offset, Decoded.back().Offset + Decoded.back().Size}};

  // The first call has one finite target outside this .text, and returns to
  // the padding that falls through into the local helper. The later local
  // call alone would appear to justify the helper's s_set_pc_i64, but the
  // external continuation is a second link-register provenance and must keep
  // that return unbounded.
  std::optional<DirectControlFlowInfo> Info = collectDirectBranchTargets(
      Decoded, S, /*TextAddr=*/0, /*TextSize=*/Bytes.size(), DeclaredEntries,
      FunctionRanges, /*ExternalEntries=*/{}, Bytes);
  ASSERT_TRUE(Info);
  EXPECT_TRUE(Info->Targets.contains(Decoded[5].Offset + Decoded[5].Size));
  EXPECT_TRUE(Info->HasUnresolvedTargets);
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

TEST(SourceTailSafety, RejectsProtectedEntriesAndOverlappingRanges) {
  Trampoline T;
  T.OriginalOffset = 8;
  T.OriginalSize = 12;
  T.HasFunctionRange = true;
  T.FunctionStart = 0;
  T.FunctionEnd = 32;
  llvm::SmallVector<ElfView::FunctionTextRange, 1> OneRange{
      {0, 32, nullptr, nullptr}};
  EXPECT_TRUE(sourceHasUniqueFunctionRange(T, OneRange, /*TextAddr=*/0));

  DirectControlFlowInfo ControlFlow;
  EXPECT_TRUE(isSafeSourceTailRange(T, ControlFlow,
                                    /*HasUniqueFunctionRange=*/true,
                                    /*Begin=*/12, /*End=*/20));
  ControlFlow.Targets.insert(16);
  EXPECT_FALSE(isSafeSourceTailRange(T, ControlFlow,
                                     /*HasUniqueFunctionRange=*/true,
                                     /*Begin=*/12, /*End=*/20));
  ControlFlow.Targets.clear();
  ControlFlow.HasUnboundedIndirectEntries = true;
  EXPECT_FALSE(isSafeSourceTailRange(T, ControlFlow,
                                     /*HasUniqueFunctionRange=*/true,
                                     /*Begin=*/12, /*End=*/20));

  llvm::SmallVector<ElfView::FunctionTextRange, 2> AliasedRanges{
      {0, 32, nullptr, nullptr}, {0, 32, nullptr, nullptr}};
  // The same logical global function can appear in both .symtab and .dynsym.
  // Equal bounds add no new interior ownership ambiguity.
  EXPECT_TRUE(sourceHasUniqueFunctionRange(T, AliasedRanges, /*TextAddr=*/0));

  llvm::SmallVector<ElfView::FunctionTextRange, 2> NestedRanges{
      {0, 64, nullptr, nullptr}, {32, 48, nullptr, nullptr}};
  T.FunctionEnd = 64;
  T.OriginalOffset = 8;
  EXPECT_TRUE(sourceHasUniqueFunctionRange(T, NestedRanges, /*TextAddr=*/0));
  T.OriginalOffset = 36;
  EXPECT_FALSE(sourceHasUniqueFunctionRange(T, NestedRanges, /*TextAddr=*/0));
}

TEST(SourceTailSafety, IndexedRangeQueryMatchesOverlapBoundaries) {
  Trampoline T;
  T.OriginalOffset = 8;
  T.OriginalSize = 12;
  T.HasFunctionRange = true;
  T.FunctionStart = 0;
  T.FunctionEnd = 32;

  llvm::SmallVector<ElfView::FunctionTextRange, 2> SameBeginDifferentEnd{
      {0, 32, nullptr, nullptr}, {0, 24, nullptr, nullptr}};
  EXPECT_FALSE(
      sourceHasUniqueFunctionRange(T, SameBeginDifferentEnd, /*TextAddr=*/0));
  EXPECT_FALSE(sourceHasUniqueFunctionRangeIndexedForTest(
      T, SameBeginDifferentEnd, /*TextAddr=*/0));

  llvm::SmallVector<ElfView::FunctionTextRange, 2> PartialOverlap{
      {8, 40, nullptr, nullptr}, {0, 20, nullptr, nullptr}};
  T.FunctionStart = 8;
  T.FunctionEnd = 40;
  T.OriginalOffset = 16;
  T.OriginalSize = 8;
  EXPECT_TRUE(sourceHasUniqueFunctionRange(T, PartialOverlap, /*TextAddr=*/0));
  EXPECT_TRUE(sourceHasUniqueFunctionRangeIndexedForTest(T, PartialOverlap,
                                                         /*TextAddr=*/0));

  T.OriginalSize = 4;
  EXPECT_FALSE(sourceHasUniqueFunctionRange(T, PartialOverlap, /*TextAddr=*/0));
  EXPECT_FALSE(sourceHasUniqueFunctionRangeIndexedForTest(T, PartialOverlap,
                                                          /*TextAddr=*/0));
}

TEST(SourceTailSafety, IndexedRangeQueryMatchesRandomizedLinearOracle) {
  uint64_t State = 0xC0FFEE1234567890ULL;
  auto Next = [&]() {
    State = State * 6364136223846793005ULL + 1442695040888963407ULL;
    return State;
  };

  constexpr uint64_t TextAddr = 0x100000;
  for (unsigned Trial = 0; Trial != 500; ++Trial) {
    llvm::SmallVector<ElfView::FunctionTextRange, 32> Ranges;
    unsigned Count = 1 + Next() % 16;
    for (unsigned I = 0; I != Count; ++I) {
      uint64_t Begin = (Next() % 64) * MinInstSize;
      uint64_t End = Begin + (1 + Next() % 24) * MinInstSize;
      Ranges.push_back({TextAddr + Begin, TextAddr + End, nullptr, nullptr});
      if ((Next() & 3) == 0)
        Ranges.push_back(Ranges.back());
    }

    const ElfView::FunctionTextRange &Selected = Ranges[Next() % Ranges.size()];
    Trampoline T;
    T.HasFunctionRange = (Next() & 15) != 0;
    T.FunctionStart = Selected.Begin - TextAddr;
    T.FunctionEnd = Selected.End - TextAddr;
    uint64_t Width = T.FunctionEnd - T.FunctionStart;
    T.OriginalOffset = T.FunctionStart + Next() % Width;
    T.OriginalSize = 1 + Next() % (T.FunctionEnd - T.OriginalOffset);
    if (Trial % 17 == 0)
      ++T.FunctionEnd;

    bool Linear = sourceHasUniqueFunctionRange(T, Ranges, TextAddr);
    bool Indexed =
        sourceHasUniqueFunctionRangeIndexedForTest(T, Ranges, TextAddr);
    EXPECT_EQ(Indexed, Linear) << "trial " << Trial;
  }
}

TEST(SourceTailSafety, ReservesRegisterlessReturnTailBeforeAffinePlanning) {
  Trampoline T;
  T.OriginalOffset = 0x100;
  T.Long = true;
  T.OriginalSize = 2 * MinInstSize;
  EXPECT_TRUE(mustReserveSourceTailForRegisterlessReturn(T));
  EXPECT_FALSE(registerlessSourceAffineGatewayRange(T));

  T.OriginalSize = 7 * MinInstSize;
  std::optional<std::pair<uint64_t, uint64_t>> Gateway =
      registerlessSourceAffineGatewayRange(T);
  ASSERT_TRUE(Gateway);
  EXPECT_EQ(Gateway->first, T.OriginalOffset + 2 * MinInstSize);
  EXPECT_EQ(Gateway->second, T.OriginalOffset + 7 * MinInstSize);
  EXPECT_GT(Gateway->first, T.OriginalOffset + MinInstSize);

  T.UsesSetPCBack = true;
  EXPECT_FALSE(mustReserveSourceTailForRegisterlessReturn(T));
  T.UsesSetPCBack = false;
  T.LongBranchPreservesVcc = true;
  EXPECT_FALSE(mustReserveSourceTailForRegisterlessReturn(T));
  T.LongBranchPreservesVcc = false;
  T.UsesSharedDispatcherForward = true;
  EXPECT_FALSE(mustReserveSourceTailForRegisterlessReturn(T));
  T.UsesSharedDispatcherForward = false;
  T.OriginalSize = MinInstSize;
  EXPECT_FALSE(mustReserveSourceTailForRegisterlessReturn(T));
  T.OriginalSize = 2 * MinInstSize;
  T.Long = false;
  EXPECT_FALSE(mustReserveSourceTailForRegisterlessReturn(T));
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
  EXPECT_EQ(Info->Targets.size(), 1u);
  EXPECT_TRUE(Info->Targets.contains(DeclaredEntries.front()));
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

TEST(CollectDirectBranchTargets, UndecodedVectorAluDoesNotCreateIndirectEntry) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  // Reduced from a B0 f4gemm code object. The A0 MC decoder does not know this
  // legacy-VOP3-major 0x34 vector instruction, but its encoding class cannot
  // affect scalar control flow or MODE.
  const uint8_t Bytes[] = {0x00, 0x00, 0x31, 0xd0, 0x00, 0x00, 0x10, 0x00};
  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes, sizeof(Bytes), S, Decoded));
  ASSERT_FALSE(Decoded.empty());
  ASSERT_FALSE(Decoded.front().DecodeSucceeded);

  std::optional<DirectControlFlowInfo> Info = collectDirectBranchTargets(
      Decoded, S, /*TextAddr=*/0, sizeof(Bytes), /*DeclaredEntries=*/{0},
      /*FunctionRanges=*/{}, /*ExternalEntries=*/{}, Bytes);
  ASSERT_TRUE(Info);
  EXPECT_FALSE(Info->HasUnresolvedTargets);
  EXPECT_FALSE(Info->HasUnboundedIndirectEntries);
}

TEST(CollectDirectBranchTargets, UndecodedScalarClassRemainsUnbounded) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  const uint8_t Bytes[] = {0xff, 0xff, 0xff, 0xff};
  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes, sizeof(Bytes), S, Decoded));
  ASSERT_EQ(Decoded.size(), 1u);
  ASSERT_FALSE(Decoded.front().DecodeSucceeded);

  std::optional<DirectControlFlowInfo> Info = collectDirectBranchTargets(
      Decoded, S, /*TextAddr=*/0, sizeof(Bytes), /*DeclaredEntries=*/{0},
      /*FunctionRanges=*/{}, /*ExternalEntries=*/{}, Bytes);
  ASSERT_TRUE(Info);
  EXPECT_TRUE(Info->HasUnboundedIndirectEntries);
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
      Decoded, S, /*TextAddr=*/0, /*TextSize=*/Bytes.size(), DeclaredEntries,
      FunctionRanges);
  ASSERT_TRUE(Info);
  ASSERT_EQ(Info->Targets.size(), 3u);
  EXPECT_TRUE(Info->Targets.contains(0));
  EXPECT_TRUE(Info->Targets.contains(Decoded[1].Offset));
  EXPECT_TRUE(
      Info->Targets.contains(Decoded.back().Offset + Decoded.back().Size));
  EXPECT_FALSE(Info->HasUnresolvedTargets);
}

TEST(CollectDirectBranchTargets,
     ClosesMaterializedCallExactJumpAndCanonicalReturnTogether) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Bytes =
      assembleInstructions("s_get_pc_i64 s[0:1]\n"
                           "s_add_nc_u64 s[0:1], s[0:1], 12\n"
                           "s_swap_pc_i64 s[30:31], s[0:1]\n"
                           "s_endpgm\n"
                           "s_get_pc_i64 s[4:5]\n"
                           "s_add_nc_u64 s[4:5], s[4:5], 12\n"
                           "s_set_pc_i64 s[4:5]\n"
                           "s_endpgm\n"
                           "s_set_pc_i64 s[30:31]",
                           S);
  ASSERT_FALSE(Bytes.empty());

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 9u);
  llvm::SmallVector<uint64_t, 2> DeclaredEntries{0, Decoded[4].Offset};
  llvm::SmallVector<ElfView::FunctionTextRange, 2> FunctionRanges{
      {0, Decoded[4].Offset}, {Decoded[4].Offset, Bytes.size()}};

  // No edge is independently sufficient: the materialized call defines the
  // helper's s30 link, its exact set-PC reaches the return, and that return
  // reaches only the call continuation. The joint finite-control-flow audit
  // must close all three edges without treating the register call's generic
  // indirect-branch classification as a second unbounded edge.
  std::optional<DirectControlFlowInfo> Info = collectDirectBranchTargets(
      Decoded, S, /*TextAddr=*/0, Bytes.size(), DeclaredEntries, FunctionRanges,
      /*ExternalEntries=*/{}, Bytes);
  ASSERT_TRUE(Info);
  EXPECT_TRUE(Info->BoundedIndirectTransfers.contains(Decoded[2].Offset));
  EXPECT_TRUE(Info->BoundedIndirectTransfers.contains(Decoded[6].Offset));
  EXPECT_TRUE(Info->BoundedIndirectTransfers.contains(Decoded[8].Offset));
  EXPECT_TRUE(Info->Targets.contains(Decoded[4].Offset));
  EXPECT_TRUE(Info->Targets.contains(Decoded[2].Offset + Decoded[2].Size));
  EXPECT_FALSE(Info->HasUnboundedIndirectEntries);
  EXPECT_FALSE(Info->HasUnresolvedTargets);
}

TEST(CollectDirectBranchTargets,
     RejectsFiniteCallEntryIntoAnotherCallMaterialization) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Bytes =
      assembleInstructions("s_get_pc_i64 s[4:5]\n"
                           "s_add_nc_u64 s[4:5], s[4:5], 12\n"
                           "s_swap_pc_i64 s[30:31], s[4:5]\n"
                           "s_get_pc_i64 s[0:1]\n"
                           "s_add_nc_u64 s[0:1], s[0:1], 0x100\n"
                           "s_swap_pc_i64 s[30:31], s[0:1]",
                           S);
  ASSERT_FALSE(Bytes.empty());

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 6u);
  llvm::SmallVector<uint64_t, 1> DeclaredEntries{0};

  // The first exact call targets the add in the second call's get-PC/add
  // materialization. Both targets are finite instruction boundaries, but the
  // first call can bypass the second call's defining get-PC. Joint closure
  // must therefore leave the second call unresolved.
  std::optional<DirectControlFlowInfo> Info = collectDirectBranchTargets(
      Decoded, S, /*TextAddr=*/0, Bytes.size(), DeclaredEntries);
  ASSERT_TRUE(Info);
  EXPECT_TRUE(Info->Targets.contains(Decoded[4].Offset));
  EXPECT_TRUE(Info->HasUnresolvedTargets);
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
  EXPECT_EQ(Info->Targets.size(), 1u);
  EXPECT_TRUE(Info->Targets.contains(DeclaredEntries.front()));
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
  EXPECT_EQ(Info->Targets.size(), 1u);
  EXPECT_TRUE(Info->Targets.contains(DeclaredEntries.front()));
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
  EXPECT_EQ(Info->Targets.size(), 1u);
  EXPECT_TRUE(Info->Targets.contains(DeclaredEntries.front()));
  EXPECT_TRUE(Info->HasUnresolvedTargets);
}

TEST(CollectDirectBranchTargets,
     RejectsUnsortedDeclaredFallthroughIntoReturnFunction) {
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
  llvm::SmallVector<uint64_t, 2> DeclaredEntries{Decoded[1].Offset, 0};
  llvm::SmallVector<ElfView::FunctionTextRange, 1> FunctionRanges{
      {Decoded[1].Offset, Decoded[3].Offset}};

  // The deliberately unsorted declared entry at zero reaches the local helper
  // by fallthrough and does not define s[30:31], so the helper's return cannot
  // be bounded.
  std::optional<DirectControlFlowInfo> Info = collectDirectBranchTargets(
      Decoded, S, /*TextAddr=*/0, /*TextSize=*/0x1000, DeclaredEntries,
      FunctionRanges);
  ASSERT_TRUE(Info);
  EXPECT_EQ(Info->Targets.size(), 2u);
  EXPECT_TRUE(Info->Targets.contains(DeclaredEntries[0]));
  EXPECT_TRUE(Info->Targets.contains(DeclaredEntries[1]));
  EXPECT_TRUE(Info->HasUnresolvedTargets);
}

TEST(CollectDirectBranchTargets,
     RejectsExactSetPcEntryIntoCallDefinedReturnFunction) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Bytes =
      assembleInstructions("s_nop 0\n"
                           "s_set_pc_i64 s[30:31]\n"
                           "s_get_pc_i64 s[0:1]\n"
                           "s_add_nc_u64 s[0:1], s[0:1], -12\n"
                           "s_swap_pc_i64 s[30:31], s[0:1]\n"
                           "s_get_pc_i64 s[4:5]\n"
                           "s_add_nc_u64 s[4:5], s[4:5], -24\n"
                           "s_set_pc_i64 s[4:5]\n"
                           "s_endpgm",
                           S);
  ASSERT_FALSE(Bytes.empty());
  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 9u);
  llvm::SmallVector<uint64_t, 2> DeclaredEntries{0, Decoded[2].Offset};
  llvm::SmallVector<ElfView::FunctionTextRange, 2> FunctionRanges{
      {0, Decoded[2].Offset}, {Decoded[2].Offset, Bytes.size()}};

  // The first call would otherwise prove the helper's link pair. The exact
  // jump later in the caller reaches the helper without defining s[30:31].
  std::optional<DirectControlFlowInfo> Info =
      collectDirectBranchTargets(Decoded, S, /*TextAddr=*/0, Bytes.size(),
                                 DeclaredEntries, FunctionRanges);
  ASSERT_TRUE(Info);
  EXPECT_FALSE(Info->BoundedIndirectTransfers.contains(Decoded[1].Offset));
  EXPECT_TRUE(Info->HasUnboundedIndirectEntries);
}

TEST(CollectDirectBranchTargets,
     RejectsCallReachedExactEntryIntoReturnFunction) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Bytes =
      assembleInstructions("s_nop 0\n"
                           "s_set_pc_i64 s[30:31]\n"
                           "s_call_i64 s[4:5], 3\n"
                           "s_call_i64 s[30:31], -4\n"
                           "s_endpgm\n"
                           "s_endpgm\n"
                           "s_get_pc_i64 s[6:7]\n"
                           "s_add_nc_u64 s[6:7], s[6:7], -28\n"
                           "s_set_pc_i64 s[6:7]",
                           S);
  ASSERT_FALSE(Bytes.empty());
  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 9u);
  llvm::SmallVector<uint64_t, 2> DeclaredEntries{0, Decoded[2].Offset};
  llvm::SmallVector<ElfView::FunctionTextRange, 2> FunctionRanges{
      {0, Decoded[2].Offset}, {Decoded[2].Offset, Bytes.size()}};

  // The endpgm at 16 blocks layout reachability to the materialization. Only
  // the finite call at 8 reaches it; that edge must still participate in the
  // exact-jump alternate-entry proof for the helper at zero.
  std::optional<DirectControlFlowInfo> Info =
      collectDirectBranchTargets(Decoded, S, /*TextAddr=*/0, Bytes.size(),
                                 DeclaredEntries, FunctionRanges);
  ASSERT_TRUE(Info);
  EXPECT_FALSE(Info->BoundedIndirectTransfers.contains(Decoded[1].Offset));
  EXPECT_TRUE(Info->HasUnboundedIndirectEntries);
}

TEST(CollectDirectBranchTargets,
     DoesNotPublishBoundedReturnsAfterNonClosedAudit) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Bytes =
      assembleInstructions("s_nop 0\n"
                           "s_set_pc_i64 s[30:31]\n"
                           "s_get_pc_i64 s[0:1]\n"
                           "s_add_nc_u64 s[0:1], s[0:1], -12\n"
                           "s_swap_pc_i64 s[30:31], s[0:1]\n"
                           "s_set_pc_i64 s[8:9]",
                           S);
  ASSERT_FALSE(Bytes.empty());
  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 6u);
  llvm::SmallVector<uint64_t, 2> DeclaredEntries{0, Decoded[2].Offset};
  llvm::SmallVector<ElfView::FunctionTextRange, 2> FunctionRanges{
      {0, Decoded[2].Offset}, {Decoded[2].Offset, Bytes.size()}};

  // The local call initially proves the helper return, but its continuation
  // reaches an open indirect transfer. A non-closed object-wide audit must
  // retract every bounded-return fact before publishing the final result.
  std::optional<DirectControlFlowInfo> Info =
      collectDirectBranchTargets(Decoded, S, /*TextAddr=*/0, Bytes.size(),
                                 DeclaredEntries, FunctionRanges);
  ASSERT_TRUE(Info);
  EXPECT_FALSE(Info->BoundedIndirectTransfers.contains(Decoded[1].Offset));
  EXPECT_TRUE(Info->HasUnboundedIndirectEntries);
}

TEST(CollectDirectBranchTargets,
     RejectsLocalCallContinuationAtReturnFunctionEntry) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Bytes =
      assembleInstructions("s_nop 0\n"
                           "s_call_i64 s[4:5], 3\n"
                           "s_nop 0\n"
                           "s_set_pc_i64 s[30:31]\n"
                           "s_endpgm\n"
                           "s_endpgm\n"
                           "s_call_i64 s[30:31], -5\n"
                           "s_endpgm",
                           S);
  ASSERT_FALSE(Bytes.empty());
  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 8u);
  llvm::SmallVector<uint64_t, 2> DeclaredEntries{0, Decoded[6].Offset};
  llvm::SmallVector<ElfView::FunctionTextRange, 1> FunctionRanges{
      {Decoded[2].Offset, Decoded[4].Offset}};
  std::optional<DirectControlFlowInfo> Info =
      collectDirectBranchTargets(Decoded, S, /*TextAddr=*/0, Bytes.size(),
                                 DeclaredEntries, FunctionRanges);
  ASSERT_TRUE(Info);
  EXPECT_FALSE(Info->BoundedIndirectTransfers.contains(Decoded[3].Offset));
  EXPECT_TRUE(Info->HasUnboundedIndirectEntries);
  EXPECT_FALSE(Info->HasUnresolvedTargets);
}

TEST(CollectDirectBranchTargets,
     RejectsFiniteExternalCallContinuationAtReturnFunctionEntry) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Bytes =
      assembleInstructions("s_get_pc_i64 s[0:1]\n"
                           "s_add_nc_u64 s[0:1], s[0:1], 0x100\n"
                           "s_swap_pc_i64 s[4:5], s[0:1]\n"
                           "s_nop 0\n"
                           "s_set_pc_i64 s[30:31]\n"
                           "s_call_i64 s[30:31], -3\n"
                           "s_endpgm",
                           S);
  ASSERT_FALSE(Bytes.empty());
  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 7u);
  llvm::SmallVector<uint64_t, 2> DeclaredEntries{0, Decoded[5].Offset};
  llvm::SmallVector<ElfView::FunctionTextRange, 1> FunctionRanges{
      {Decoded[3].Offset, Decoded[5].Offset}};
  std::optional<DirectControlFlowInfo> Info =
      collectDirectBranchTargets(Decoded, S, /*TextAddr=*/0, Bytes.size(),
                                 DeclaredEntries, FunctionRanges);
  ASSERT_TRUE(Info);
  EXPECT_FALSE(Info->BoundedIndirectTransfers.contains(Decoded[4].Offset));
  EXPECT_TRUE(Info->HasUnboundedIndirectEntries);
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
      Decoded, S, /*TextAddr=*/0, /*TextSize=*/Bytes.size(), DeclaredEntries,
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
  ASSERT_EQ(OutsideInfo->Targets.size(), 1u);
  EXPECT_TRUE(
      OutsideInfo->Targets.contains(Decoded[0].Offset + Decoded[0].Size));
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

TEST(CollectDirectBranchTargets, ProtectsExternalPcRelativeCallContinuation) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Bytes =
      assembleSingleInst("s_call_i64 s[30:31], 2", S);
  ASSERT_FALSE(Bytes.empty());

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 1u);

  // The encoded target is beyond this deliberately short .text, while the
  // instruction after the call remains inside .text. The target must not
  // become a local protection offset, while the return continuation remains
  // protected.
  std::optional<DirectControlFlowInfo> Info =
      collectDirectBranchTargets(Decoded, S, /*TextAddr=*/0,
                                 /*TextSize=*/2 * Decoded[0].Size,
                                 /*DeclaredEntries=*/{});
  ASSERT_TRUE(Info);
  ASSERT_EQ(Info->Targets.size(), 1u);
  EXPECT_TRUE(Info->Targets.contains(Decoded[0].Offset + Decoded[0].Size));
  EXPECT_FALSE(Info->HasUnresolvedTargets);
}

TEST(CollectDirectBranchTargets,
     BoundsExactSetPcEdgeThatEnablesDifferentPairSelector) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Bytes =
      assembleInstructions("s_get_pc_i64 s[4:5]\n"
                           "s_add_nc_u64 s[4:5], s[4:5], 12\n"
                           "s_set_pc_i64 s[4:5]\n"
                           "s_endpgm\n"
                           "s_get_pc_i64 s[0:1]\n"
                           "s_add_co_i32 s2, 0xfe8, 4\n"
                           "s_add_co_u32 s0, s0, s2\n"
                           "s_add_co_ci_u32 s1, s1, 0\n"
                           "s_swap_pc_i64 s[30:31], s[0:1]",
                           S);
  ASSERT_FALSE(Bytes.empty());

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 9u);
  ASSERT_EQ(Decoded[4].Offset, 16u);
  llvm::SmallVector<uint64_t, 1> DeclaredEntries{0};
  llvm::SmallVector<ElfView::FunctionTextRange, 1> FunctionRanges{
      {0, Bytes.size()}};

  // The exact s[4:5] jump reaches a selector built in s[0:1]. Its call target
  // is the finite external address 0x1000 and its local continuation must be
  // retained.
  std::optional<DirectControlFlowInfo> Info = collectDirectBranchTargets(
      Decoded, S, /*TextAddr=*/0, Bytes.size(), DeclaredEntries, FunctionRanges,
      /*ExternalEntries=*/{}, Bytes);
  ASSERT_TRUE(Info);
  EXPECT_TRUE(Info->BoundedIndirectTransfers.contains(Decoded[2].Offset));
  EXPECT_TRUE(Info->BoundedIndirectTransfers.contains(Decoded[8].Offset));
  EXPECT_TRUE(Info->Targets.contains(0));
  EXPECT_TRUE(Info->Targets.contains(Decoded[4].Offset));
  EXPECT_TRUE(Info->Targets.contains(Decoded[8].Offset + Decoded[8].Size));
  EXPECT_FALSE(Info->HasUnboundedIndirectEntries);
  EXPECT_FALSE(Info->HasUnresolvedTargets);
}

TEST(CollectDirectBranchTargets,
     RebuildsLeastSetPcFixedPointAfterUpstreamRejection) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Bytes =
      assembleInstructions("s_get_pc_i64 s[4:5]\n"
                           "s_add_nc_u64 s[4:5], s[4:5], 12\n"
                           "s_set_pc_i64 s[4:5]\n"
                           "s_endpgm\n"
                           "s_get_pc_i64 s[6:7]\n"
                           "s_add_nc_u64 s[6:7], s[6:7], 0x100\n"
                           "s_set_pc_i64 s[6:7]",
                           S);
  ASSERT_FALSE(Bytes.empty());

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 7u);
  llvm::SmallVector<uint64_t, 2> DeclaredEntries{0, Decoded[1].Offset};
  llvm::SmallVector<ElfView::FunctionTextRange, 1> FunctionRanges{
      {0, Bytes.size()}};

  // The alternate entry invalidates A. B was reachable only through A, so a
  // fresh least-fixed-point rebuild must not retain B as a sticky proof.
  std::optional<DirectControlFlowInfo> Info = collectDirectBranchTargets(
      Decoded, S, /*TextAddr=*/0, Bytes.size(), DeclaredEntries, FunctionRanges,
      /*ExternalEntries=*/{}, Bytes);
  ASSERT_TRUE(Info);
  EXPECT_FALSE(Info->BoundedIndirectTransfers.contains(Decoded[2].Offset));
  EXPECT_FALSE(Info->BoundedIndirectTransfers.contains(Decoded[6].Offset));
  EXPECT_TRUE(Info->HasUnboundedIndirectEntries);
  EXPECT_FALSE(Info->HasUnresolvedTargets);
}

TEST(CollectDirectBranchTargets, RejectsGappedExactSetPcSequence) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Bytes =
      assembleInstructions("s_get_pc_i64 s[4:5]\n"
                           "s_add_nc_u64 s[4:5], s[4:5], 0x100\n"
                           "s_set_pc_i64 s[4:5]",
                           S);
  ASSERT_FALSE(Bytes.empty());

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 3u);
  Decoded[1].Offset += MinInstSize;
  Decoded[2].Offset += MinInstSize;
  llvm::SmallVector<uint64_t, 1> DeclaredEntries{0};
  llvm::SmallVector<ElfView::FunctionTextRange, 1> FunctionRanges{
      {0, Bytes.size() + MinInstSize}};

  std::optional<DirectControlFlowInfo> Info = collectDirectBranchTargets(
      Decoded, S, /*TextAddr=*/0, Bytes.size() + MinInstSize, DeclaredEntries,
      FunctionRanges);
  ASSERT_TRUE(Info);
  EXPECT_FALSE(Info->BoundedIndirectTransfers.contains(Decoded[2].Offset));
  // The set-PC itself is unreachable across the decode gap.
  EXPECT_FALSE(Info->HasUnboundedIndirectEntries);
  EXPECT_FALSE(Info->HasUnresolvedTargets);
}

TEST(CollectDirectBranchTargets, RejectsExactSetPcTargetInInstructionInterior) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Bytes =
      assembleInstructions("s_get_pc_i64 s[4:5]\n"
                           "s_add_nc_u64 s[4:5], s[4:5], 12\n"
                           "s_set_pc_i64 s[4:5]\n"
                           "s_add_co_i32 s6, 0x100, 4\n"
                           "s_endpgm",
                           S);
  ASSERT_FALSE(Bytes.empty());

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 5u);
  ASSERT_EQ(Decoded[3].Size, 2 * MinInstSize);
  llvm::SmallVector<uint64_t, 1> DeclaredEntries{0};
  llvm::SmallVector<ElfView::FunctionTextRange, 1> FunctionRanges{
      {0, Bytes.size()}};

  // PC=4 plus 12 targets offset 16, the second dword of Decoded[3].
  std::optional<DirectControlFlowInfo> Info =
      collectDirectBranchTargets(Decoded, S, /*TextAddr=*/0, Bytes.size(),
                                 DeclaredEntries, FunctionRanges);
  ASSERT_TRUE(Info);
  EXPECT_FALSE(Info->BoundedIndirectTransfers.contains(Decoded[2].Offset));
  EXPECT_TRUE(Info->HasUnboundedIndirectEntries);
  EXPECT_FALSE(Info->HasUnresolvedTargets);
}

TEST(CollectDirectBranchTargets, RejectsOverlappingSetPcDeltaRegister) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Bytes =
      assembleInstructions("s_get_pc_i64 s[0:1]\n"
                           "s_add_co_i32 s0, 0x100, 4\n"
                           "s_add_co_u32 s0, s0, s0\n"
                           "s_add_co_ci_u32 s1, s1, 0\n"
                           "s_set_pc_i64 s[0:1]",
                           S);
  ASSERT_FALSE(Bytes.empty());

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 5u);
  llvm::SmallVector<uint64_t, 1> DeclaredEntries{0};
  llvm::SmallVector<ElfView::FunctionTextRange, 1> FunctionRanges{
      {0, Bytes.size()}};
  std::optional<DirectControlFlowInfo> Info =
      collectDirectBranchTargets(Decoded, S, /*TextAddr=*/0, Bytes.size(),
                                 DeclaredEntries, FunctionRanges);
  ASSERT_TRUE(Info);
  EXPECT_FALSE(Info->BoundedIndirectTransfers.contains(Decoded.back().Offset));
  EXPECT_TRUE(Info->HasUnboundedIndirectEntries);
}

TEST(CollectDirectBranchTargets,
     RejectsSecondDwordEntryIntoExactSetPcMaterialization) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Bytes =
      assembleInstructions("s_get_pc_i64 s[0:1]\n"
                           "s_add_co_i32 s2, 0xfc, 4\n"
                           "s_add_co_u32 s0, s0, s2\n"
                           "s_add_co_ci_u32 s1, s1, 0\n"
                           "s_set_pc_i64 s[0:1]",
                           S);
  ASSERT_FALSE(Bytes.empty());

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 5u);
  ASSERT_EQ(Decoded[1].Size, 2 * MinInstSize);
  llvm::SmallVector<uint64_t, 2> DeclaredEntries{0, Decoded[1].Offset +
                                                        MinInstSize};
  llvm::SmallVector<ElfView::FunctionTextRange, 1> FunctionRanges{
      {0, Bytes.size()}};
  std::optional<DirectControlFlowInfo> Info =
      collectDirectBranchTargets(Decoded, S, /*TextAddr=*/0, Bytes.size(),
                                 DeclaredEntries, FunctionRanges);
  ASSERT_TRUE(Info);
  EXPECT_FALSE(Info->BoundedIndirectTransfers.contains(Decoded.back().Offset));
  EXPECT_TRUE(Info->HasUnboundedIndirectEntries);
}

TEST(CollectDirectBranchTargets, BoundsExactSetPcTargetOutsideText) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Bytes =
      assembleInstructions("s_get_pc_i64 s[4:5]\n"
                           "s_add_nc_u64 s[4:5], s[4:5], 0x100\n"
                           "s_set_pc_i64 s[4:5]",
                           S);
  ASSERT_FALSE(Bytes.empty());
  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  llvm::SmallVector<uint64_t, 1> DeclaredEntries{0};
  llvm::SmallVector<ElfView::FunctionTextRange, 1> FunctionRanges{
      {0, Bytes.size()}};
  std::optional<DirectControlFlowInfo> Info =
      collectDirectBranchTargets(Decoded, S, /*TextAddr=*/0, Bytes.size(),
                                 DeclaredEntries, FunctionRanges);
  ASSERT_TRUE(Info);
  EXPECT_TRUE(Info->BoundedIndirectTransfers.contains(Decoded.back().Offset));
  EXPECT_EQ(Info->Targets.size(), 1u);
  EXPECT_TRUE(Info->Targets.contains(0));
  EXPECT_FALSE(Info->HasUnboundedIndirectEntries);
  EXPECT_FALSE(Info->HasUnresolvedTargets);
}

TEST(CollectDirectBranchTargets,
     BoundsBothSignedSetPcDirectionsAndWrappedExternalTarget) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  for (llvm::StringRef Literal : {"0x24", "0xfc", "0xfffffff8", "0x7fffffff"}) {
    SCOPED_TRACE(Literal.str());
    llvm::SmallString<512> Assembly;
    Assembly += "s_get_pc_i64 s[4:5]\ns_add_co_i32 s6, ";
    Assembly += Literal;
    Assembly += ", 4\n"
                "s_cmp_ge_i32 s6, 0\n"
                "s_cbranch_scc1 4\n"
                "s_abs_i32 s6, s6\n"
                "s_sub_co_u32 s4, s4, s6\n"
                "s_sub_co_ci_u32 s5, s5, 0\n"
                "s_set_pc_i64 s[4:5]\n"
                "s_add_co_u32 s4, s4, s6\n"
                "s_add_co_ci_u32 s5, s5, 0\n"
                "s_set_pc_i64 s[4:5]\n"
                "s_endpgm";
    llvm::SmallVector<uint8_t> Bytes = assembleInstructions(Assembly, S);
    ASSERT_FALSE(Bytes.empty());
    std::vector<InternalDecodedInst> Decoded;
    ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
    ASSERT_EQ(Decoded.size(), 12u);
    llvm::SmallVector<uint64_t, 1> DeclaredEntries{0};
    llvm::SmallVector<ElfView::FunctionTextRange, 1> FunctionRanges{
        {0, Bytes.size()}};
    std::optional<DirectControlFlowInfo> Info =
        collectDirectBranchTargets(Decoded, S, /*TextAddr=*/0, Bytes.size(),
                                   DeclaredEntries, FunctionRanges);
    ASSERT_TRUE(Info);
    EXPECT_TRUE(Info->BoundedIndirectTransfers.contains(Decoded[7].Offset));
    EXPECT_TRUE(Info->BoundedIndirectTransfers.contains(Decoded[10].Offset));
    if (Literal == "0x24") {
      // PC=4 plus (0x24+4) lands just after the complete two-arm sequence.
      EXPECT_TRUE(Info->Targets.contains(Decoded[11].Offset));
    }
    EXPECT_FALSE(Info->HasUnboundedIndirectEntries);
    EXPECT_FALSE(Info->HasUnresolvedTargets);
  }
}

TEST(CollectDirectBranchTargets, RejectsExactSetPcSelfInteriorTarget) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Bytes =
      assembleInstructions("s_get_pc_i64 s[4:5]\n"
                           "s_add_co_i32 s6, 4, 4\n"
                           "s_cmp_ge_i32 s6, 0\n"
                           "s_cbranch_scc1 4\n"
                           "s_abs_i32 s6, s6\n"
                           "s_sub_co_u32 s4, s4, s6\n"
                           "s_sub_co_ci_u32 s5, s5, 0\n"
                           "s_set_pc_i64 s[4:5]\n"
                           "s_add_co_u32 s4, s4, s6\n"
                           "s_add_co_ci_u32 s5, s5, 0\n"
                           "s_set_pc_i64 s[4:5]",
                           S);
  ASSERT_FALSE(Bytes.empty());
  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 11u);
  llvm::SmallVector<uint64_t, 1> DeclaredEntries{0};
  llvm::SmallVector<ElfView::FunctionTextRange, 1> FunctionRanges{
      {0, Bytes.size()}};
  std::optional<DirectControlFlowInfo> Info =
      collectDirectBranchTargets(Decoded, S, /*TextAddr=*/0, Bytes.size(),
                                 DeclaredEntries, FunctionRanges);
  ASSERT_TRUE(Info);
  EXPECT_FALSE(Info->BoundedIndirectTransfers.contains(Decoded[7].Offset));
  EXPECT_FALSE(Info->BoundedIndirectTransfers.contains(Decoded[10].Offset));
  EXPECT_TRUE(Info->HasUnboundedIndirectEntries);
}

TEST(CollectDirectBranchTargets,
     DoesNotSelfAuthorizeDisconnectedExactCycleAndHonorsExternalRoot) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Bytes =
      assembleInstructions("s_endpgm\n"
                           "s_get_pc_i64 s[4:5]\n"
                           "s_add_nc_u64 s[4:5], s[4:5], 8\n"
                           "s_set_pc_i64 s[4:5]\n"
                           "s_get_pc_i64 s[6:7]\n"
                           "s_add_nc_u64 s[6:7], s[6:7], -16\n"
                           "s_set_pc_i64 s[6:7]",
                           S);
  ASSERT_FALSE(Bytes.empty());
  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 7u);
  llvm::SmallVector<uint64_t, 1> DeclaredEntries{0};
  llvm::SmallVector<ElfView::FunctionTextRange, 1> FunctionRanges{
      {0, Bytes.size()}};

  std::optional<DirectControlFlowInfo> DeadInfo =
      collectDirectBranchTargets(Decoded, S, /*TextAddr=*/0, Bytes.size(),
                                 DeclaredEntries, FunctionRanges);
  ASSERT_TRUE(DeadInfo);
  EXPECT_FALSE(DeadInfo->BoundedIndirectTransfers.contains(Decoded[3].Offset));
  EXPECT_FALSE(DeadInfo->BoundedIndirectTransfers.contains(Decoded[6].Offset));
  EXPECT_FALSE(DeadInfo->HasUnboundedIndirectEntries);

  llvm::SmallVector<uint64_t, 1> ExternalEntries{Decoded[1].Offset};
  std::optional<DirectControlFlowInfo> RootedInfo = collectDirectBranchTargets(
      Decoded, S, /*TextAddr=*/0, Bytes.size(), DeclaredEntries, FunctionRanges,
      ExternalEntries);
  ASSERT_TRUE(RootedInfo);
  EXPECT_TRUE(RootedInfo->BoundedIndirectTransfers.contains(Decoded[3].Offset));
  EXPECT_TRUE(RootedInfo->BoundedIndirectTransfers.contains(Decoded[6].Offset));
  EXPECT_TRUE(RootedInfo->Targets.contains(Decoded[1].Offset));
  EXPECT_TRUE(RootedInfo->Targets.contains(Decoded[4].Offset));
  EXPECT_FALSE(RootedInfo->HasUnboundedIndirectEntries);
  EXPECT_FALSE(RootedInfo->HasUnresolvedTargets);
}

TEST(CollectDirectBranchTargets, RejectsNonBoundaryExternalEntry) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Bytes =
      assembleInstructions("s_endpgm\n"
                           "s_add_co_i32 s2, 0x100, 4\n"
                           "s_endpgm",
                           S);
  ASSERT_FALSE(Bytes.empty());

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 3u);
  ASSERT_EQ(Decoded[1].Size, 2 * MinInstSize);
  llvm::SmallVector<uint64_t, 1> DeclaredEntries{0};
  llvm::SmallVector<uint64_t, 1> ExternalEntries{Decoded[1].Offset +
                                                 MinInstSize};
  std::optional<DirectControlFlowInfo> Info = collectDirectBranchTargets(
      Decoded, S, /*TextAddr=*/0, Bytes.size(), DeclaredEntries,
      /*FunctionRanges=*/{}, ExternalEntries);
  ASSERT_TRUE(Info);
  EXPECT_TRUE(Info->Targets.contains(ExternalEntries.front()));
  EXPECT_TRUE(Info->HasUnboundedIndirectEntries);
  EXPECT_FALSE(Info->HasUnresolvedTargets);
}

TEST(CollectDirectBranchTargets, RejectsNonBoundaryDirectAndCallTargets) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  for (llvm::StringRef Transfer :
       {"s_branch 1\n", "s_call_i64 s[30:31], 1\n"}) {
    SCOPED_TRACE(Transfer.str());
    llvm::SmallString<128> Assembly(Transfer);
    Assembly += "s_add_co_i32 s2, 0x100, 4\ns_endpgm";
    llvm::SmallVector<uint8_t> Bytes = assembleInstructions(Assembly, S);
    ASSERT_FALSE(Bytes.empty());
    std::vector<InternalDecodedInst> Decoded;
    ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
    ASSERT_EQ(Decoded.size(), 3u);
    ASSERT_EQ(Decoded[1].Size, 2 * MinInstSize);
    llvm::SmallVector<uint64_t, 1> DeclaredEntries{0};
    std::optional<DirectControlFlowInfo> Info = collectDirectBranchTargets(
        Decoded, S, /*TextAddr=*/0, Bytes.size(), DeclaredEntries);
    ASSERT_TRUE(Info);
    EXPECT_TRUE(Info->Targets.contains(2 * MinInstSize));
    EXPECT_TRUE(Info->HasUnboundedIndirectEntries);
    EXPECT_FALSE(Info->HasUnresolvedTargets);
  }
}

TEST(CollectDirectBranchTargets,
     BoundsSymbolLessReturnAndUnionsAllContinuations) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Bytes =
      assembleInstructions("s_call_i64 s[30:31], 2\n"
                           "s_call_i64 s[30:31], 1\n"
                           "s_endpgm\n"
                           "s_nop 0\n"
                           "s_set_pc_i64 s[30:31]",
                           S);
  ASSERT_FALSE(Bytes.empty());

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 5u);
  llvm::SmallVector<uint64_t, 1> DeclaredEntries{0};
  std::optional<DirectControlFlowInfo> Info = collectDirectBranchTargets(
      Decoded, S, /*TextAddr=*/0, Bytes.size(), DeclaredEntries);
  ASSERT_TRUE(Info);
  EXPECT_TRUE(Info->Targets.contains(Decoded[0].Offset + Decoded[0].Size));
  EXPECT_TRUE(Info->Targets.contains(Decoded[1].Offset + Decoded[1].Size));
  EXPECT_TRUE(Info->Targets.contains(Decoded[3].Offset));
  EXPECT_TRUE(Info->BoundedIndirectTransfers.contains(Decoded[4].Offset));
  EXPECT_FALSE(Info->HasUnboundedIndirectEntries);
  EXPECT_FALSE(Info->HasUnresolvedTargets);
}

TEST(CollectDirectBranchTargets, BoundsTwoIndependentSymbolLessReturnRegions) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Bytes =
      assembleInstructions("s_call_i64 s[30:31], 5\n"
                           "s_endpgm\n"
                           "s_call_i64 s[28:29], 5\n"
                           "s_endpgm\n"
                           "s_endpgm\n"
                           "s_endpgm\n"
                           "s_nop 0\n"
                           "s_set_pc_i64 s[30:31]\n"
                           "s_nop 0\n"
                           "s_set_pc_i64 s[28:29]",
                           S);
  ASSERT_FALSE(Bytes.empty());
  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 10u);
  llvm::SmallVector<uint64_t, 2> DeclaredEntries{0, Decoded[2].Offset};

  // Both regions are independently call-defined and neither may cause the
  // other's still-provisional return to be treated as an open indirect entry.
  // They must be inferred together and accepted by the joint closed audit.
  std::optional<DirectControlFlowInfo> Info = collectDirectBranchTargets(
      Decoded, S, /*TextAddr=*/0, Bytes.size(), DeclaredEntries);
  ASSERT_TRUE(Info);
  EXPECT_TRUE(Info->BoundedIndirectTransfers.contains(Decoded[7].Offset));
  EXPECT_TRUE(Info->BoundedIndirectTransfers.contains(Decoded[9].Offset));
  EXPECT_TRUE(Info->Targets.contains(Decoded[6].Offset));
  EXPECT_TRUE(Info->Targets.contains(Decoded[8].Offset));
  EXPECT_TRUE(Info->Targets.contains(Decoded[0].Offset + Decoded[0].Size));
  EXPECT_TRUE(Info->Targets.contains(Decoded[2].Offset + Decoded[2].Size));
  EXPECT_FALSE(Info->HasUnboundedIndirectEntries);
  EXPECT_FALSE(Info->HasUnresolvedTargets);
}

TEST(CollectDirectBranchTargets,
     RejectsJointSymbolLessReturnEntryBetweenRegions) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Bytes =
      assembleInstructions("s_call_i64 s[30:31], 4\n"
                           "s_nop 0\n"
                           "s_set_pc_i64 s[28:29]\n"
                           "s_call_i64 s[28:29], -3\n"
                           "s_endpgm\n"
                           "s_nop 0\n"
                           "s_set_pc_i64 s[30:31]",
                           S);
  ASSERT_FALSE(Bytes.empty());
  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 7u);
  llvm::SmallVector<uint64_t, 2> DeclaredEntries{0, Decoded[3].Offset};

  // The first region's return continuation is the second region's entry, but
  // it does not define that region's s[28:29] link pair. Provisional returns
  // inferred in one pass must not mutually authorize this alternate entry.
  std::optional<DirectControlFlowInfo> Info = collectDirectBranchTargets(
      Decoded, S, /*TextAddr=*/0, Bytes.size(), DeclaredEntries);
  ASSERT_TRUE(Info);
  EXPECT_FALSE(Info->BoundedIndirectTransfers.contains(Decoded[2].Offset));
  EXPECT_FALSE(Info->BoundedIndirectTransfers.contains(Decoded[6].Offset));
  EXPECT_TRUE(Info->HasUnboundedIndirectEntries);
  EXPECT_FALSE(Info->HasUnresolvedTargets);
}

TEST(CollectDirectBranchTargets, RejectsOverlappingSymbolLessReturnRegions) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Bytes =
      assembleInstructions("s_call_i64 s[30:31], 5\n"
                           "s_endpgm\n"
                           "s_call_i64 s[30:31], 5\n"
                           "s_endpgm\n"
                           "s_endpgm\n"
                           "s_endpgm\n"
                           "s_branch 2\n"
                           "s_endpgm\n"
                           "s_branch 0\n"
                           "s_nop 0\n"
                           "s_set_pc_i64 s[30:31]",
                           S);
  ASSERT_FALSE(Bytes.empty());
  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 11u);
  llvm::SmallVector<uint64_t, 2> DeclaredEntries{0, Decoded[2].Offset};

  // The two distinct call entries use the same link pair but converge on one
  // return tail. Neither provisional region owns that shared body uniquely,
  // so the streaming overlap audit must reject both.
  std::optional<DirectControlFlowInfo> Info = collectDirectBranchTargets(
      Decoded, S, /*TextAddr=*/0, Bytes.size(), DeclaredEntries);
  ASSERT_TRUE(Info);
  EXPECT_FALSE(Info->BoundedIndirectTransfers.contains(Decoded[10].Offset));
  EXPECT_TRUE(Info->HasUnboundedIndirectEntries);
  EXPECT_FALSE(Info->HasUnresolvedTargets);
}

TEST(CollectDirectBranchTargets,
     RejectsWrongPairAndClobberedSymbolLessReturns) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  for (llvm::StringRef Body : {"s_nop 0\ns_set_pc_i64 s[28:29]",
                               "s_mov_b32 s30, 0\ns_set_pc_i64 s[30:31]"}) {
    SCOPED_TRACE(Body.str());
    llvm::SmallString<128> Assembly(
        "s_call_i64 s[30:31], 2\ns_endpgm\ns_endpgm\n");
    Assembly += Body;
    llvm::SmallVector<uint8_t> Bytes = assembleInstructions(Assembly, S);
    ASSERT_FALSE(Bytes.empty());
    std::vector<InternalDecodedInst> Decoded;
    ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
    ASSERT_EQ(Decoded.size(), 5u);
    llvm::SmallVector<uint64_t, 1> DeclaredEntries{0};
    std::optional<DirectControlFlowInfo> Info = collectDirectBranchTargets(
        Decoded, S, /*TextAddr=*/0, Bytes.size(), DeclaredEntries);
    ASSERT_TRUE(Info);
    EXPECT_FALSE(
        Info->BoundedIndirectTransfers.contains(Decoded.back().Offset));
    EXPECT_TRUE(Info->HasUnboundedIndirectEntries);
    EXPECT_FALSE(Info->HasUnresolvedTargets);
  }
}

TEST(CollectDirectBranchTargets,
     RejectsNestedCallAndFallthroughSymbolLessReturns) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> NestedBytes =
      assembleInstructions("s_call_i64 s[30:31], 2\n"
                           "s_endpgm\n"
                           "s_endpgm\n"
                           "s_call_i64 s[4:5], 1\n"
                           "s_set_pc_i64 s[30:31]\n"
                           "s_endpgm",
                           S);
  ASSERT_FALSE(NestedBytes.empty());
  std::vector<InternalDecodedInst> Nested;
  ASSERT_TRUE(
      decodeTextSection(NestedBytes.data(), NestedBytes.size(), S, Nested));
  llvm::SmallVector<uint64_t, 1> DeclaredEntries{0};
  std::optional<DirectControlFlowInfo> NestedInfo = collectDirectBranchTargets(
      Nested, S, /*TextAddr=*/0, NestedBytes.size(), DeclaredEntries);
  ASSERT_TRUE(NestedInfo);
  EXPECT_FALSE(NestedInfo->BoundedIndirectTransfers.contains(Nested[4].Offset));
  EXPECT_TRUE(NestedInfo->HasUnboundedIndirectEntries);

  llvm::SmallVector<uint8_t> FallthroughBytes =
      assembleInstructions("s_call_i64 s[30:31], 2\n"
                           "s_endpgm\n"
                           "s_nop 0\n"
                           "s_nop 0\n"
                           "s_set_pc_i64 s[30:31]",
                           S);
  ASSERT_FALSE(FallthroughBytes.empty());
  std::vector<InternalDecodedInst> Fallthrough;
  ASSERT_TRUE(decodeTextSection(FallthroughBytes.data(),
                                FallthroughBytes.size(), S, Fallthrough));
  std::optional<DirectControlFlowInfo> FallthroughInfo =
      collectDirectBranchTargets(Fallthrough, S, /*TextAddr=*/0,
                                 FallthroughBytes.size(), DeclaredEntries);
  ASSERT_TRUE(FallthroughInfo);
  EXPECT_FALSE(FallthroughInfo->BoundedIndirectTransfers.contains(
      Fallthrough.back().Offset));
  EXPECT_TRUE(FallthroughInfo->HasUnboundedIndirectEntries);
}

TEST(CollectDirectBranchTargets,
     RejectsUnknownAndExternalEntriesIntoSymbolLessReturn) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Bytes =
      assembleInstructions("s_set_pc_i64 s[8:9]\n"
                           "s_call_i64 s[30:31], 2\n"
                           "s_endpgm\n"
                           "s_endpgm\n"
                           "s_nop 0\n"
                           "s_set_pc_i64 s[30:31]",
                           S);
  ASSERT_FALSE(Bytes.empty());
  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 6u);
  llvm::SmallVector<uint64_t, 2> DeclaredEntries{0, Decoded[1].Offset};
  std::optional<DirectControlFlowInfo> UnknownInfo = collectDirectBranchTargets(
      Decoded, S, /*TextAddr=*/0, Bytes.size(), DeclaredEntries);
  ASSERT_TRUE(UnknownInfo);
  EXPECT_FALSE(
      UnknownInfo->BoundedIndirectTransfers.contains(Decoded.back().Offset));
  EXPECT_TRUE(UnknownInfo->HasUnboundedIndirectEntries);

  llvm::SmallVector<uint64_t, 1> ExternalEntries{Decoded[4].Offset};
  std::optional<DirectControlFlowInfo> ExternalInfo =
      collectDirectBranchTargets(
          llvm::ArrayRef<InternalDecodedInst>(Decoded).drop_front(), S,
          /*TextAddr=*/0, Bytes.size(),
          /*DeclaredEntries=*/{Decoded[1].Offset},
          /*FunctionRanges=*/{}, ExternalEntries);
  ASSERT_TRUE(ExternalInfo);
  EXPECT_FALSE(
      ExternalInfo->BoundedIndirectTransfers.contains(Decoded.back().Offset));
  EXPECT_TRUE(ExternalInfo->HasUnboundedIndirectEntries);
}

TEST(CollectDirectBranchTargets,
     RejectsGapAndSecondDwordEntryIntoSymbolLessReturn) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> GapBytes =
      assembleInstructions("s_call_i64 s[30:31], 2\n"
                           "s_endpgm\n"
                           "s_endpgm\n"
                           "s_nop 0\n"
                           "s_set_pc_i64 s[30:31]",
                           S);
  ASSERT_FALSE(GapBytes.empty());
  std::vector<InternalDecodedInst> Gap;
  ASSERT_TRUE(decodeTextSection(GapBytes.data(), GapBytes.size(), S, Gap));
  ASSERT_EQ(Gap.size(), 5u);
  Gap.back().Offset += MinInstSize;
  llvm::SmallVector<uint64_t, 1> DeclaredEntries{0};
  std::optional<DirectControlFlowInfo> GapInfo = collectDirectBranchTargets(
      Gap, S, /*TextAddr=*/0, GapBytes.size() + MinInstSize, DeclaredEntries);
  ASSERT_TRUE(GapInfo);
  EXPECT_FALSE(GapInfo->BoundedIndirectTransfers.contains(Gap.back().Offset));

  llvm::SmallVector<uint8_t> MidBytes =
      assembleInstructions("s_call_i64 s[30:31], 2\n"
                           "s_endpgm\n"
                           "s_endpgm\n"
                           "s_add_co_i32 s2, 0x100, 4\n"
                           "s_set_pc_i64 s[30:31]",
                           S);
  ASSERT_FALSE(MidBytes.empty());
  std::vector<InternalDecodedInst> Mid;
  ASSERT_TRUE(decodeTextSection(MidBytes.data(), MidBytes.size(), S, Mid));
  ASSERT_EQ(Mid.size(), 5u);
  ASSERT_EQ(Mid[3].Size, 2 * MinInstSize);
  llvm::SmallVector<uint64_t, 2> MidEntries{0, Mid[3].Offset + MinInstSize};
  std::optional<DirectControlFlowInfo> MidInfo = collectDirectBranchTargets(
      Mid, S, /*TextAddr=*/0, MidBytes.size(), MidEntries);
  ASSERT_TRUE(MidInfo);
  EXPECT_FALSE(MidInfo->BoundedIndirectTransfers.contains(Mid.back().Offset));
  EXPECT_TRUE(MidInfo->HasUnboundedIndirectEntries);
}

TEST(CollectDirectBranchTargets,
     RejectsDirectAndDeclaredBypassIntoSymbolLessReturn) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Bytes =
      assembleInstructions("s_call_i64 s[30:31], 4\n"
                           "s_endpgm\n"
                           "s_branch 3\n"
                           "s_call_i64 s[4:5], 2\n"
                           "s_endpgm\n"
                           "s_nop 0\n"
                           "s_nop 0\n"
                           "s_set_pc_i64 s[30:31]",
                           S);
  ASSERT_FALSE(Bytes.empty());
  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 8u);
  llvm::SmallVector<uint64_t, 3> DirectRoots{0, Decoded[2].Offset,
                                             Decoded[3].Offset};
  std::optional<DirectControlFlowInfo> DirectInfo = collectDirectBranchTargets(
      Decoded, S, /*TextAddr=*/0, Bytes.size(), DirectRoots);
  ASSERT_TRUE(DirectInfo);
  EXPECT_FALSE(
      DirectInfo->BoundedIndirectTransfers.contains(Decoded.back().Offset));
  EXPECT_TRUE(DirectInfo->HasUnboundedIndirectEntries);

  llvm::SmallVector<uint8_t> DeclaredBytes =
      assembleInstructions("s_call_i64 s[30:31], 2\n"
                           "s_endpgm\n"
                           "s_endpgm\n"
                           "s_nop 0\n"
                           "s_set_pc_i64 s[30:31]",
                           S);
  ASSERT_FALSE(DeclaredBytes.empty());
  std::vector<InternalDecodedInst> Declared;
  ASSERT_TRUE(decodeTextSection(DeclaredBytes.data(), DeclaredBytes.size(), S,
                                Declared));
  llvm::SmallVector<uint64_t, 2> DeclaredBypass{0, Declared[3].Offset};
  std::optional<DirectControlFlowInfo> DeclaredInfo =
      collectDirectBranchTargets(Declared, S, /*TextAddr=*/0,
                                 DeclaredBytes.size(), DeclaredBypass);
  ASSERT_TRUE(DeclaredInfo);
  EXPECT_FALSE(
      DeclaredInfo->BoundedIndirectTransfers.contains(Declared.back().Offset));
  EXPECT_TRUE(DeclaredInfo->HasUnboundedIndirectEntries);
}

TEST(CollectDirectBranchTargets,
     RejectsFunctionRangeEntryInsideExactSetPcMaterialization) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Bytes =
      assembleInstructions("s_get_pc_i64 s[4:5]\n"
                           "s_add_nc_u64 s[4:5], s[4:5], 12\n"
                           "s_set_pc_i64 s[4:5]\n"
                           "s_endpgm",
                           S);
  ASSERT_FALSE(Bytes.empty());
  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 4u);
  llvm::SmallVector<uint64_t, 1> DeclaredEntries{0};
  llvm::SmallVector<ElfView::FunctionTextRange, 2> FunctionRanges{
      {0, Bytes.size()}, {Decoded[1].Offset, Bytes.size()}};

  // Exercise the internal API defensively without duplicating the second
  // range start in DeclaredEntries. It is still an alternate root into the
  // exact materialization and must invalidate that bounded transfer.
  std::optional<DirectControlFlowInfo> Info =
      collectDirectBranchTargets(Decoded, S, /*TextAddr=*/0, Bytes.size(),
                                 DeclaredEntries, FunctionRanges);
  ASSERT_TRUE(Info);
  EXPECT_FALSE(Info->BoundedIndirectTransfers.contains(Decoded[2].Offset));
  EXPECT_TRUE(Info->HasUnboundedIndirectEntries);
}

TEST(CollectDirectBranchTargets,
     RejectsFunctionRangeEntryInsideSymbolLessReturnRegion) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Bytes =
      assembleInstructions("s_call_i64 s[30:31], 2\n"
                           "s_endpgm\n"
                           "s_endpgm\n"
                           "s_nop 0\n"
                           "s_nop 0\n"
                           "s_set_pc_i64 s[30:31]",
                           S);
  ASSERT_FALSE(Bytes.empty());
  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 6u);
  llvm::SmallVector<uint64_t, 1> DeclaredEntries{0};
  llvm::SmallVector<ElfView::FunctionTextRange, 1> FunctionRanges{
      {Decoded[4].Offset, Bytes.size()}};

  // The function-range start is a second root into the inferred region after
  // its call entry. It bypasses the call-defined link pair even when callers
  // of this internal API omit that begin from DeclaredEntries.
  std::optional<DirectControlFlowInfo> Info =
      collectDirectBranchTargets(Decoded, S, /*TextAddr=*/0, Bytes.size(),
                                 DeclaredEntries, FunctionRanges);
  ASSERT_TRUE(Info);
  EXPECT_FALSE(Info->BoundedIndirectTransfers.contains(Decoded[5].Offset));
  EXPECT_TRUE(Info->HasUnboundedIndirectEntries);
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

TEST(SafeSgprScratchBlock, CommitCacheIsMonotoneAcrossOwnerScopes) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  llvm::SmallVector<uint8_t> Text = assembleSingleInst("s_endpgm", S);
  ASSERT_FALSE(Text.empty());

  comgr_test::KernelDescriptorElfOptions Options;
  Options.MetadataSgprCount = 4;
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

  std::optional<ElfView::FunctionTextRange> Function =
      View.findFunctionTextRangeAtOffset(0);
  ASSERT_TRUE(Function);
  auto FunctionKey = std::make_pair(Function->Begin, Function->End);

  // Overflow and a missing selected owner fail without changing any cache,
  // counter, or externally visible kernel requirement.
  Ctx.FunctionKernelOwner[FunctionKey] = "missing";
  const SafeSgprScratchBlock Overflow{
      /*Base=*/std::numeric_limits<unsigned>::max(), /*Count=*/2};
  EXPECT_FALSE(
      commitSafeSgprScratchBlock(Ctx, /*TextOffset=*/0, Overflow, "unit test"));
  const SafeSgprScratchBlock Initial{/*Base=*/4, /*Count=*/2};
  EXPECT_FALSE(
      commitSafeSgprScratchBlock(Ctx, /*TextOffset=*/0, Initial, "unit test"));
  EXPECT_EQ(Ctx.AllKernelSgprRequirement, 0u);
  EXPECT_TRUE(Ctx.KernelSgprRequirements.empty());
  EXPECT_EQ(Ctx.SgprDescriptorChargePasses, 0u);
  EXPECT_TRUE(KernelStats.empty());

  // An owned commitment raises only that owner's coverage.
  Ctx.FunctionKernelOwner[FunctionKey] = "kernel";
  ASSERT_TRUE(
      commitSafeSgprScratchBlock(Ctx, /*TextOffset=*/0, Initial, "unit test"));
  EXPECT_EQ(Ctx.AllKernelSgprRequirement, 0u);
  EXPECT_EQ(Ctx.KernelSgprRequirements["kernel"], 8u);
  EXPECT_EQ(Ctx.SgprDescriptorChargePasses, 1u);
  ASSERT_EQ(KernelStats.count("kernel"), 1u);
  EXPECT_EQ(KernelStats["kernel"].ExtraSgprs, 4u);

  // Owned coverage cannot stand in for all-kernel coverage.
  Ctx.FunctionKernelOwner[FunctionKey] = "";
  const SafeSgprScratchBlock Smaller{/*Base=*/0, /*Count=*/2};
  EXPECT_TRUE(
      commitSafeSgprScratchBlock(Ctx, /*TextOffset=*/0, Smaller, "unit test"));
  EXPECT_EQ(Ctx.AllKernelSgprRequirement, 4u);
  EXPECT_EQ(Ctx.SgprDescriptorChargePasses, 2u);
  EXPECT_EQ(KernelStats["kernel"].ExtraSgprs, 4u);

  // Equal and decreasing ownerless requirements are already represented by
  // the monotone KernelStats update and must not rescan the descriptor table.
  EXPECT_TRUE(
      commitSafeSgprScratchBlock(Ctx, /*TextOffset=*/0, Smaller, "unit test"));
  const SafeSgprScratchBlock Smallest{/*Base=*/0, /*Count=*/1};
  EXPECT_TRUE(
      commitSafeSgprScratchBlock(Ctx, /*TextOffset=*/0, Smallest, "unit test"));
  EXPECT_EQ(Ctx.AllKernelSgprRequirement, 4u);
  EXPECT_EQ(Ctx.SgprDescriptorChargePasses, 2u);

  // A larger global request performs exactly one new pass and raises both
  // cached and externally visible requirements.
  const SafeSgprScratchBlock Larger{/*Base=*/8, /*Count=*/2};
  EXPECT_TRUE(
      commitSafeSgprScratchBlock(Ctx, /*TextOffset=*/0, Larger, "unit test"));
  EXPECT_EQ(Ctx.AllKernelSgprRequirement, 12u);
  EXPECT_EQ(Ctx.SgprDescriptorChargePasses, 3u);
  EXPECT_EQ(KernelStats["kernel"].ExtraSgprs, 8u);

  // Global coverage is a valid lower bound for every individual owner.
  Ctx.FunctionKernelOwner[FunctionKey] = "kernel";
  EXPECT_TRUE(
      commitSafeSgprScratchBlock(Ctx, /*TextOffset=*/0, Larger, "unit test"));
  EXPECT_EQ(Ctx.SgprDescriptorChargePasses, 3u);
  EXPECT_EQ(Ctx.KernelSgprRequirements["kernel"], 8u);

  const SafeSgprScratchBlock Largest{/*Base=*/12, /*Count=*/2};
  EXPECT_TRUE(
      commitSafeSgprScratchBlock(Ctx, /*TextOffset=*/0, Largest, "unit test"));
  EXPECT_EQ(Ctx.SgprDescriptorChargePasses, 4u);
  EXPECT_EQ(Ctx.KernelSgprRequirements["kernel"], 16u);
  EXPECT_EQ(KernelStats["kernel"].ExtraSgprs, 12u);
}

TEST(SafeSgprScratchBlock, OwnerlessCommitFailureIsAtomic) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  comgr_test::MultiKernelDescriptorElfOptions Options;
  Options.Kernels = {
      {"valid", 0x1000, 0x2000, /*EntryOffset=*/-0x1000,
       /*ComputePgmRsrc3=*/0, /*EmitMetadata=*/true,
       /*MetadataSgprCount=*/4},
      {"malformed", 0x1100, 0x2100, /*EntryOffset=*/-0x1000,
       /*ComputePgmRsrc3=*/0, /*EmitMetadata=*/false,
       /*MetadataSgprCount=*/0},
  };
  std::vector<uint8_t> Bytes =
      comgr_test::makeMultiKernelDescriptorElf(Options);
  llvm::Expected<ElfView> ViewOrErr =
      ElfView::create(Bytes.data(), Bytes.size());
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

  std::optional<ElfView::FunctionTextRange> Function =
      View.findFunctionTextRangeAtOffset(0);
  ASSERT_TRUE(Function);
  Ctx.FunctionKernelOwner[{Function->Begin, Function->End}] = "";

  // The valid descriptor is visited before the malformed metadata entry. A
  // one-phase implementation would leak its queued ExtraSgprs update here.
  const SafeSgprScratchBlock Block{/*Base=*/4, /*Count=*/2};
  EXPECT_FALSE(
      commitSafeSgprScratchBlock(Ctx, /*TextOffset=*/0, Block, "unit test"));
  EXPECT_TRUE(KernelStats.empty());
  EXPECT_EQ(Ctx.AllKernelSgprRequirement, 0u);
  EXPECT_TRUE(Ctx.KernelSgprRequirements.empty());
  EXPECT_EQ(Ctx.SgprDescriptorChargePasses, 0u);
}

TEST(ForwardDeadVgprs, OpaqueBeforeKillRejects) {
  constexpr unsigned MaxVgprs = 8;
  constexpr unsigned Candidate = 3;
  std::vector<ForwardVgprProofNode> Nodes;
  Nodes.emplace_back(MaxVgprs);
  Nodes.emplace_back(MaxVgprs);
  Nodes[0].Opaque = true;
  Nodes[1].FullDefs.set(Candidate);
  Nodes[1].SafeTerminal = true;

  std::optional<llvm::BitVector> Safe =
      computeForwardDeadVgprs(Nodes, /*EntryNode=*/0, MaxVgprs);
  ASSERT_TRUE(Safe);
  EXPECT_FALSE(Safe->test(Candidate));
}

TEST(ForwardDeadVgprs, KillBeforeOpaqueAccepts) {
  constexpr unsigned MaxVgprs = 8;
  constexpr unsigned Candidate = 3;
  std::vector<ForwardVgprProofNode> Nodes;
  Nodes.emplace_back(MaxVgprs);
  Nodes.emplace_back(MaxVgprs);
  Nodes[0].FullDefs.set(Candidate);
  Nodes[0].Successors.push_back(1);
  Nodes[1].Opaque = true;

  std::optional<llvm::BitVector> Safe =
      computeForwardDeadVgprs(Nodes, /*EntryNode=*/0, MaxVgprs);
  ASSERT_TRUE(Safe);
  EXPECT_TRUE(Safe->test(Candidate));
}

TEST(ForwardDeadVgprs, ExternalExitBeforeKillRejects) {
  constexpr unsigned MaxVgprs = 8;
  constexpr unsigned Candidate = 3;
  std::vector<ForwardVgprProofNode> Nodes;
  Nodes.emplace_back(MaxVgprs);
  Nodes.emplace_back(MaxVgprs);
  Nodes[0].HasUnsafeExit = true;
  Nodes[0].Successors.push_back(1);
  Nodes[1].FullDefs.set(Candidate);
  Nodes[1].SafeTerminal = true;

  std::optional<llvm::BitVector> Safe =
      computeForwardDeadVgprs(Nodes, /*EntryNode=*/0, MaxVgprs);
  ASSERT_TRUE(Safe);
  EXPECT_FALSE(Safe->test(Candidate));
}

TEST(ForwardDeadVgprs, KillBeforeExternalExitAccepts) {
  constexpr unsigned MaxVgprs = 8;
  constexpr unsigned Candidate = 3;
  std::vector<ForwardVgprProofNode> Nodes;
  Nodes.emplace_back(MaxVgprs);
  Nodes.emplace_back(MaxVgprs);
  Nodes[0].FullDefs.set(Candidate);
  Nodes[0].Successors.push_back(1);
  Nodes[1].HasUnsafeExit = true;
  Nodes[1].SafeTerminal = true;

  std::optional<llvm::BitVector> Safe =
      computeForwardDeadVgprs(Nodes, /*EntryNode=*/0, MaxVgprs);
  ASSERT_TRUE(Safe);
  EXPECT_TRUE(Safe->test(Candidate));
}

TEST(ForwardDeadVgprs, OneUseBeforeKillBranchRejects) {
  constexpr unsigned MaxVgprs = 8;
  constexpr unsigned Candidate = 3;
  std::vector<ForwardVgprProofNode> Nodes;
  for (unsigned I = 0; I != 3; ++I)
    Nodes.emplace_back(MaxVgprs);
  Nodes[0].Successors.push_back(1);
  Nodes[0].Successors.push_back(2);
  Nodes[1].Uses.set(Candidate);
  Nodes[1].SafeTerminal = true;
  Nodes[2].FullDefs.set(Candidate);
  Nodes[2].SafeTerminal = true;

  std::optional<llvm::BitVector> Safe =
      computeForwardDeadVgprs(Nodes, /*EntryNode=*/0, MaxVgprs);
  ASSERT_TRUE(Safe);
  EXPECT_FALSE(Safe->test(Candidate));
}

TEST(ForwardDeadVgprs, AllBranchesKillAccepts) {
  constexpr unsigned MaxVgprs = 8;
  constexpr unsigned Candidate = 3;
  std::vector<ForwardVgprProofNode> Nodes;
  for (unsigned I = 0; I != 3; ++I)
    Nodes.emplace_back(MaxVgprs);
  Nodes[0].Successors.push_back(1);
  Nodes[0].Successors.push_back(2);
  Nodes[1].FullDefs.set(Candidate);
  Nodes[1].SafeTerminal = true;
  Nodes[2].FullDefs.set(Candidate);
  Nodes[2].SafeTerminal = true;

  std::optional<llvm::BitVector> Safe =
      computeForwardDeadVgprs(Nodes, /*EntryNode=*/0, MaxVgprs);
  ASSERT_TRUE(Safe);
  EXPECT_TRUE(Safe->test(Candidate));
}

TEST(ForwardDeadVgprs, LoopPathWithoutKillRejects) {
  constexpr unsigned MaxVgprs = 8;
  constexpr unsigned Candidate = 3;
  std::vector<ForwardVgprProofNode> Nodes;
  Nodes.emplace_back(MaxVgprs);
  Nodes.emplace_back(MaxVgprs);
  Nodes[0].Successors.push_back(1);
  Nodes[1].Successors.push_back(0);

  std::optional<llvm::BitVector> Safe =
      computeForwardDeadVgprs(Nodes, /*EntryNode=*/0, MaxVgprs);
  ASSERT_TRUE(Safe);
  EXPECT_FALSE(Safe->test(Candidate));
}

TEST(ForwardDeadVgprs, LoopWithFullKillAccepts) {
  constexpr unsigned MaxVgprs = 8;
  constexpr unsigned Candidate = 3;
  std::vector<ForwardVgprProofNode> Nodes;
  Nodes.emplace_back(MaxVgprs);
  Nodes.emplace_back(MaxVgprs);
  Nodes[0].FullDefs.set(Candidate);
  Nodes[0].Successors.push_back(1);
  Nodes[1].Successors.push_back(0);

  std::optional<llvm::BitVector> Safe =
      computeForwardDeadVgprs(Nodes, /*EntryNode=*/0, MaxVgprs);
  ASSERT_TRUE(Safe);
  EXPECT_TRUE(Safe->test(Candidate));
}

TEST(ForwardDeadVgprs, BackedgeUseBeforePatchedSiteRejects) {
  constexpr unsigned MaxVgprs = 8;
  constexpr unsigned Candidate = 3;
  std::vector<ForwardVgprProofNode> Nodes;
  for (unsigned I = 0; I != 3; ++I)
    Nodes.emplace_back(MaxVgprs);
  Nodes[0].Successors.push_back(1);
  Nodes[1].Uses.set(Candidate);
  Nodes[1].Successors.push_back(2);
  Nodes[2].SafeTerminal = true;

  std::optional<llvm::BitVector> Safe =
      computeForwardDeadVgprs(Nodes, /*EntryNode=*/0, MaxVgprs);
  ASSERT_TRUE(Safe);
  EXPECT_FALSE(Safe->test(Candidate));
}

TEST(ForwardDeadVgprs, BackedgeWithoutUseToPatchedSiteAccepts) {
  constexpr unsigned MaxVgprs = 8;
  constexpr unsigned Candidate = 3;
  std::vector<ForwardVgprProofNode> Nodes;
  for (unsigned I = 0; I != 3; ++I)
    Nodes.emplace_back(MaxVgprs);
  Nodes[0].Successors.push_back(1);
  Nodes[1].Successors.push_back(2);
  Nodes[2].SafeTerminal = true;

  std::optional<llvm::BitVector> Safe =
      computeForwardDeadVgprs(Nodes, /*EntryNode=*/0, MaxVgprs);
  ASSERT_TRUE(Safe);
  EXPECT_TRUE(Safe->test(Candidate));
}

TEST(ForwardDeadVgprs, PartialDefinitionIsUseNotKill) {
  constexpr unsigned MaxVgprs = 8;
  constexpr unsigned Candidate = 3;
  std::vector<ForwardVgprProofNode> Nodes;
  Nodes.emplace_back(MaxVgprs);
  // Partial/tied definitions consume the incoming full dword and therefore
  // belong in Uses, never FullDefs.
  Nodes[0].Uses.set(Candidate);
  Nodes[0].SafeTerminal = true;

  std::optional<llvm::BitVector> Safe =
      computeForwardDeadVgprs(Nodes, /*EntryNode=*/0, MaxVgprs);
  ASSERT_TRUE(Safe);
  EXPECT_FALSE(Safe->test(Candidate));
}

TEST(WmmaScale16, PhysicalVgprRangeMustFitOneBank) {
  EXPECT_TRUE(physicalVgprRangeFitsOneBank(0, 16, 1024));
  EXPECT_TRUE(physicalVgprRangeFitsOneBank(248, 8, 1024));
  EXPECT_TRUE(physicalVgprRangeFitsOneBank(1016, 8, 1024));

  EXPECT_FALSE(physicalVgprRangeFitsOneBank(0, 0, 1024));
  EXPECT_FALSE(physicalVgprRangeFitsOneBank(249, 8, 1024));
  EXPECT_FALSE(physicalVgprRangeFitsOneBank(255, 2, 1024));
  EXPECT_FALSE(physicalVgprRangeFitsOneBank(1017, 8, 1024));
  EXPECT_FALSE(physicalVgprRangeFitsOneBank(1024, 1, 1024));
}

TEST(WmmaScale16, UnrecognizedVectorRegisterCannotDisappearFromProof) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  ASSERT_NE(S.MRI, nullptr);

  auto FindRegister = [&](llvm::StringRef Name) {
    for (unsigned Reg = 1; Reg != S.MRI->getNumRegs(); ++Reg)
      if (Name == S.MRI->getName(Reg))
        return llvm::MCRegister(Reg);
    return llvm::MCRegister();
  };

  // AGPRs are vector registers but are deliberately not representable as an
  // encoded v0..v255 range. Such an operand must invalidate the physical-VGPR
  // proof; it cannot be ignored like a scalar register.
  llvm::MCRegister Agpr0 = FindRegister("AGPR0");
  llvm::MCRegister Sgpr0 = FindRegister("SGPR0");
  ASSERT_TRUE(Agpr0);
  ASSERT_TRUE(Sgpr0);
  EXPECT_TRUE(isVectorRegisterOrAlias(Agpr0, *S.MRI));
  EXPECT_FALSE(isVectorRegisterOrAlias(Sgpr0, *S.MRI));
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

TEST(BranchIslandAllocator, AcceptsExactPositiveReachBoundary) {
  std::vector<NopSled> Gateways = {
      {MaxSledDistance, MaxSledDistance + MinInstSize, MaxSledDistance,
       /*FunctionStart=*/0, /*FunctionEnd=*/3 * MaxSledDistance}};
  BranchIslandAllocatorTestResult Result = runBranchIslandAllocatorForTest(
      std::move(Gateways), /*OwnerOffset=*/0, /*FromOffset=*/0,
      /*TargetOffset=*/2 * MaxSledDistance, /*Backward=*/false);
  ASSERT_TRUE(Result.Success);
  ASSERT_EQ(Result.Islands.size(), 1u);
  EXPECT_EQ(Result.Islands.front(), MaxSledDistance);
}

TEST(BranchIslandAllocator, RollsBackPartialChainAndAliases) {
  constexpr uint64_t Head = 170000;
  std::vector<NopSled> Gateways = {
      {Head, Head + MinInstSize, Head, 0, 400000},
      {Head, Head + 2 * MinInstSize, Head, 0, 400000}};
  BranchIslandAllocatorTestResult Result = runBranchIslandAllocatorForTest(
      std::move(Gateways), /*OwnerOffset=*/0, /*FromOffset=*/300000,
      /*TargetOffset=*/0, /*Backward=*/true);
  EXPECT_FALSE(Result.Success);
  ASSERT_EQ(Result.Gateways.size(), 2u);
  EXPECT_EQ(Result.Gateways[0].WritePos, Head);
  EXPECT_EQ(Result.Gateways[1].WritePos, Head);
  EXPECT_TRUE(Result.Occupied.empty());
}

TEST(BranchIslandAllocator, HoldsPartialChainAcrossMultiplePromotions) {
  std::vector<NopSled> Gateways = {
      {670000, 670000 + MinInstSize, 670000, 0, 900000},
      {540000, 540000 + MinInstSize, 540000, 0, 900000},
      {410000, 410000 + MinInstSize, 410000, 0, 900000}};
  const NopSled Promotions[] = {
      {280000, 280000 + MinInstSize, 280000, 0, 900000},
      {150000, 150000 + MinInstSize, 150000, 0, 900000}};
  BranchIslandAllocatorTestResult Result =
      runBranchIslandAllocatorWithPromotionsForTest(
          std::move(Gateways), /*OwnerOffset=*/0, /*FromOffset=*/800000,
          /*TargetOffset=*/20000, /*Backward=*/true, Promotions);
  ASSERT_TRUE(Result.Success);
  EXPECT_EQ(Result.Islands, (llvm::SmallVector<uint64_t, 4>{
                                670000, 540000, 410000, 280000, 150000}));
  EXPECT_EQ(Result.HeldIslandCountsAtPromotion,
            (llvm::SmallVector<size_t, 4>{3, 4}));
}

TEST(BranchIslandAllocator, TerminalFailureRollsBackPromotedPartialChain) {
  std::vector<NopSled> Gateways = {
      {670000, 670000 + MinInstSize, 670000, 0, 900000},
      {540000, 540000 + MinInstSize, 540000, 0, 900000},
      {410000, 410000 + MinInstSize, 410000, 0, 900000}};
  const NopSled Promotions[] = {
      {280000, 280000 + MinInstSize, 280000, 0, 900000}};
  BranchIslandAllocatorTestResult Result =
      runBranchIslandAllocatorWithPromotionsForTest(
          std::move(Gateways), /*OwnerOffset=*/0, /*FromOffset=*/800000,
          /*TargetOffset=*/20000, /*Backward=*/true, Promotions);
  ASSERT_FALSE(Result.Success);
  ASSERT_EQ(Result.Gateways.size(), 4u);
  EXPECT_EQ(Result.Gateways[0].WritePos, 670000u);
  EXPECT_EQ(Result.Gateways[1].WritePos, 540000u);
  EXPECT_EQ(Result.Gateways[2].WritePos, 410000u);
  EXPECT_EQ(Result.Gateways[3].WritePos, 280000u);
  EXPECT_TRUE(Result.Occupied.empty());
  EXPECT_EQ(Result.HeldIslandCountsAtPromotion,
            (llvm::SmallVector<size_t, 4>{3}));
}

TEST(BranchIslandAllocator, SkipsGatewayFromDifferentFunction) {
  std::vector<NopSled> Gateways = {
      {MaxSledDistance, MaxSledDistance + MinInstSize, MaxSledDistance,
       /*FunctionStart=*/1, /*FunctionEnd=*/300000},
      {130000, 130000 + MinInstSize, 130000,
       /*FunctionStart=*/0, /*FunctionEnd=*/300000},
      {260000, 260000 + MinInstSize, 260000,
       /*FunctionStart=*/0, /*FunctionEnd=*/300000}};
  BranchIslandAllocatorTestResult Result = runBranchIslandAllocatorForTest(
      std::move(Gateways), /*OwnerOffset=*/0, /*FromOffset=*/0,
      /*TargetOffset=*/262144, /*Backward=*/false);
  ASSERT_TRUE(Result.Success);
  ASSERT_EQ(Result.Islands.size(), 2u);
  EXPECT_EQ(Result.Islands[0], 130000u);
  EXPECT_EQ(Result.Islands[1], 260000u);
}

TEST(BranchIslandAllocator, CoAdvancesEqualPhysicalAliases) {
  std::vector<NopSled> Gateways = {
      {MaxSledDistance, MaxSledDistance + 2 * MinInstSize, MaxSledDistance, 0,
       3 * MaxSledDistance},
      {MaxSledDistance, MaxSledDistance + 3 * MinInstSize, MaxSledDistance, 0,
       3 * MaxSledDistance}};
  BranchIslandAllocatorTestResult Result = runBranchIslandAllocatorForTest(
      std::move(Gateways), /*OwnerOffset=*/0, /*FromOffset=*/0,
      /*TargetOffset=*/2 * MaxSledDistance, /*Backward=*/false);
  ASSERT_TRUE(Result.Success);
  ASSERT_EQ(Result.Gateways.size(), 2u);
  EXPECT_EQ(Result.Gateways[0].WritePos, MaxSledDistance + MinInstSize);
  EXPECT_EQ(Result.Gateways[1].WritePos, MaxSledDistance + MinInstSize);
  EXPECT_TRUE(Result.Occupied.contains(MaxSledDistance));
}

TEST(TrampolineCoalescing, MergesAdjacentRegisterlessReturnSites) {
  Trampoline First;
  First.OriginalOffset = 100;
  First.OriginalSize = 8;
  First.Bytes.assign({1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0});
  First.Long = true;
  First.HasFunctionRange = true;
  First.FunctionStart = 64;
  First.FunctionEnd = 256;

  Trampoline Second = First;
  Second.OriginalOffset = 108;
  Second.Bytes.assign({9, 10, 11, 12, 13, 14, 15, 16, 0, 0, 0, 0});

  std::vector<Trampoline> Merged =
      mergeAdjacentLongTrampolinesForTest({First, Second});
  ASSERT_EQ(Merged.size(), 1u);
  EXPECT_EQ(Merged[0].OriginalOffset, 100u);
  EXPECT_EQ(Merged[0].OriginalSize, 16u);
  EXPECT_EQ(Merged[0].Bytes,
            (llvm::SmallVector<uint8_t>{1,  2,  3,  4,  5,  6,  7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 0, 0, 0, 0}));

  std::vector<Trampoline> Protected = mergeAdjacentLongTrampolinesForTest(
      {First, Second}, llvm::DenseSet<uint64_t>{108});
  EXPECT_EQ(Protected.size(), 2u);
}

TEST(BranchIslandAllocator, SplitsPartialAliasAtOccupiedDword) {
  llvm::DenseSet<uint64_t> Occupied = {108};
  std::vector<NopSled> Available = subtractOccupiedBranchGatewaySlotsForTest(
      {{100, 140, 100, 0, 200}}, Occupied);
  ASSERT_EQ(Available.size(), 2u);
  EXPECT_EQ(Available[0].Start, 100u);
  EXPECT_EQ(Available[0].End, 108u);
  EXPECT_EQ(Available[1].Start, 112u);
  EXPECT_EQ(Available[1].End, 140u);
}

TEST(BranchPromotionSearchRange, ClampsForwardCorridorToReachableBand) {
  constexpr uint64_t Current = 100000;
  auto Far =
      branchPromotionSearchRangeForTest(Current, /*CorridorOffset=*/900000,
                                        /*Forward=*/true);
  EXPECT_EQ(Far.first, Current);
  EXPECT_EQ(Far.second, Current + MaxSledDistance);

  auto Near =
      branchPromotionSearchRangeForTest(Current, /*CorridorOffset=*/120000,
                                        /*Forward=*/true);
  EXPECT_EQ(Near, (std::pair<uint64_t, uint64_t>{Current, 120000}));

  auto Saturated = branchPromotionSearchRangeForTest(
      std::numeric_limits<uint64_t>::max() - 8,
      std::numeric_limits<uint64_t>::max(), /*Forward=*/true);
  EXPECT_EQ(Saturated.second, std::numeric_limits<uint64_t>::max());
}

TEST(BranchPromotionSearchRange, ClampsBackwardCorridorToReachableBand) {
  constexpr uint64_t Current = 1000000;
  auto Far = branchPromotionSearchRangeForTest(Current, /*CorridorOffset=*/1000,
                                               /*Forward=*/false);
  EXPECT_EQ(Far.first, Current - MaxSledDistance - SetPcForwardSequenceBytes);
  EXPECT_EQ(Far.second, Current);

  auto Near = branchPromotionSearchRangeForTest(
      Current, /*CorridorOffset=*/950000, /*Forward=*/false);
  EXPECT_EQ(Near.first, 950000u - SetPcForwardSequenceBytes - MinInstSize);
  EXPECT_EQ(Near.second, Current);

  auto Saturated = branchPromotionSearchRangeForTest(
      /*CurrentOffset=*/8, /*CorridorOffset=*/0, /*Forward=*/false);
  EXPECT_EQ(Saturated.first, 0u);
}

TEST(BranchPromotionCandidateCursor, MatchesScalarDirectionalOrderAndBounds) {
  constexpr size_t Count = 9;
  constexpr size_t Begin = 2;
  constexpr size_t End = 7;
  const size_t Rejected[] = {3, 5, 99};
  auto IsRejected = [&](size_t Index) {
    return std::find(std::begin(Rejected), std::end(Rejected), Index) !=
           std::end(Rejected);
  };

  llvm::SmallVector<size_t, 8> ScalarForward;
  for (size_t Index = End; Index-- > Begin;)
    if (!IsRejected(Index))
      ScalarForward.push_back(Index);
  EXPECT_EQ(promotionCandidateOrderForTest(Count, Rejected, Begin, End,
                                           /*Forward=*/true),
            ScalarForward);

  llvm::SmallVector<size_t, 8> ScalarBackward;
  for (size_t Index = Begin; Index != End; ++Index)
    if (!IsRejected(Index))
      ScalarBackward.push_back(Index);
  EXPECT_EQ(promotionCandidateOrderForTest(Count, Rejected, Begin, End,
                                           /*Forward=*/false),
            ScalarBackward);

  EXPECT_TRUE(
      promotionCandidateOrderForTest(Count, Rejected, /*BeginIndex=*/Count,
                                     /*EndIndex=*/Count + 10, /*Forward=*/true)
          .empty());
}

TEST(BranchPromotionCandidateCursor,
     RepeatedScanSkipsOnlyPermanentlyRejectedStarts) {
  llvm::SmallVector<size_t, 8> Initial =
      promotionCandidateOrderForTest(/*CandidateCount=*/6, {},
                                     /*BeginIndex=*/1, /*EndIndex=*/5,
                                     /*Forward=*/true);
  EXPECT_EQ(Initial, (llvm::SmallVector<size_t, 8>{4, 3, 2, 1}));

  const size_t PermanentlyRejected[] = {4, 2};
  llvm::SmallVector<size_t, 8> Retried = promotionCandidateOrderForTest(
      /*CandidateCount=*/6, PermanentlyRejected,
      /*BeginIndex=*/1, /*EndIndex=*/5, /*Forward=*/true);
  EXPECT_EQ(Retried, (llvm::SmallVector<size_t, 8>{3, 1}));

  // The same persistent bits retain the opposite directional order.
  llvm::SmallVector<size_t, 8> Backward = promotionCandidateOrderForTest(
      /*CandidateCount=*/6, PermanentlyRejected,
      /*BeginIndex=*/1, /*EndIndex=*/5, /*Forward=*/false);
  EXPECT_EQ(Backward, (llvm::SmallVector<size_t, 8>{1, 3}));
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
  EXPECT_TRUE(Decoded[0].DecodeSucceeded);
  EXPECT_EQ(Decoded[0].Size, MinInstSize);
  EXPECT_EQ(Decoded[0].Mnemonic, "s_nop");
}

TEST(AssembleDecode, SoftFailIsNotProofSafe) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  llvm::SmallVector<uint8_t> Bytes =
      assembleSingleInst("v_cmpx_eq_f64_e64 v[2:3], v[2:3]", S);
  ASSERT_EQ(Bytes.size(), 2 * MinInstSize);

  // VOP3 compare-x requires EXEC_LO in its encoded destination field. Keep the
  // instruction otherwise decodable but name a different destination, which
  // the AMDGPU disassembler diagnoses as SoftFail.
  Bytes[0] ^= 1;
  llvm::MCInst Inst;
  uint64_t InstSize = 0;
  llvm::MCDisassembler::DecodeStatus Status = S.MCD->getInstruction(
      Inst, InstSize, Bytes, /*Address=*/0, llvm::nulls());
  ASSERT_EQ(Status, llvm::MCDisassembler::SoftFail);
  ASSERT_EQ(InstSize, Bytes.size());

  llvm::SmallVector<uint8_t> Text;
  Text.append(Bytes);
  Text.append(Bytes);
  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Text.data(), Text.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 2u);
  for (size_t I = 0; I != Decoded.size(); ++I) {
    EXPECT_FALSE(Decoded[I].DecodeSucceeded);
    EXPECT_EQ(Decoded[I].Offset, I * InstSize);
    EXPECT_EQ(Decoded[I].Size, InstSize);
    EXPECT_EQ(Decoded[I].Mnemonic, "<unknown>");
  }

  // Scratch-register proofs consume DecodeSucceeded as their trust boundary.
  // Undefined operands must therefore retain every incoming register value.
  ASSERT_TRUE(S.VCCRegister.isValid());
  EXPECT_TRUE(replacementNeedsIncomingRegister(Bytes, S, S.VCCRegister));
}

TEST(RegisterLiveness, TiedAccumulatorDefCountsAsIncomingRead) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  llvm::SmallVector<uint8_t> Bytes =
      assembleSingleInst("v_fmac_f32_e32 v5, v1, v2", S);
  ASSERT_EQ(Bytes.size(), MinInstSize);

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 1u);
  const InternalDecodedInst &DI = Decoded[0];
  const llvm::MCInstrDesc &Desc = S.MCII->get(DI.Inst.getOpcode());
  ASSERT_GE(Desc.getNumDefs(), 1u);
  ASSERT_GE(DI.Inst.getNumOperands(), 1u);
  ASSERT_TRUE(DI.Inst.getOperand(0).isReg());

  bool HasTiedAccumulatorUse = false;
  for (unsigned I = Desc.getNumDefs(); I != Desc.getNumOperands(); ++I)
    HasTiedAccumulatorUse |=
        Desc.getOperandConstraint(I, llvm::MCOI::TIED_TO) == 0;
  ASSERT_TRUE(HasTiedAccumulatorUse);

  llvm::MCRegister Accumulator(DI.Inst.getOperand(0).getReg());
  EXPECT_TRUE(instructionReadsRegister(DI, S, Accumulator));
}

TEST(RegisterLiveness, PartialVccDefinitionDoesNotKillFullVcc) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  ASSERT_TRUE(S.VCCRegister.isValid());

  std::vector<InternalDecodedInst> Partial = decodeAsmSequence(
      S, llvm::ArrayRef<llvm::StringRef>({"s_mov_b32 vcc_lo, s0"}));
  ASSERT_EQ(Partial.size(), 1u);
  ASSERT_TRUE(Partial.front().Inst.getOperand(0).isReg());
  llvm::MCRegister VccLo(Partial.front().Inst.getOperand(0).getReg());
  EXPECT_TRUE(instructionFullyWritesRegister(Partial.front(), S, VccLo));
  EXPECT_FALSE(
      instructionFullyWritesRegister(Partial.front(), S, S.VCCRegister));

  std::vector<InternalDecodedInst> Full = decodeAsmSequence(
      S, llvm::ArrayRef<llvm::StringRef>({"s_mov_b64 vcc, -1"}));
  ASSERT_EQ(Full.size(), 1u);
  EXPECT_TRUE(instructionFullyWritesRegister(Full.front(), S, VccLo));
  EXPECT_TRUE(instructionFullyWritesRegister(Full.front(), S, S.VCCRegister));

  llvm::SmallVector<uint8_t> PartialThenHighUse =
      assembleInstructions("s_mov_b32 vcc_lo, s0\ns_mov_b32 s1, vcc_hi", S);
  ASSERT_FALSE(PartialThenHighUse.empty());
  EXPECT_TRUE(
      replacementNeedsIncomingRegister(PartialThenHighUse, S, S.VCCRegister));

  llvm::SmallVector<uint8_t> FullThenHighUse =
      assembleInstructions("s_mov_b64 vcc, -1\ns_mov_b32 s1, vcc_hi", S);
  ASSERT_FALSE(FullThenHighUse.empty());
  EXPECT_FALSE(
      replacementNeedsIncomingRegister(FullThenHighUse, S, S.VCCRegister));
}

TEST(RegisterLiveness, BatchedVccNeedsRespectControlFlowAndFullDefs) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  ASSERT_TRUE(S.VCCRegister.isValid());

  auto Compute = [&](llvm::ArrayRef<llvm::StringRef> Lines) {
    std::vector<InternalDecodedInst> Decoded = decodeAsmSequence(S, Lines);
    EXPECT_FALSE(Decoded.empty());
    if (Decoded.empty())
      return std::optional<llvm::DenseSet<uint64_t>>();
    uint64_t End = Decoded.back().Offset + Decoded.back().Size;
    return computeIncomingRegisterNeeds(Decoded, S, /*FunctionBegin=*/0, End,
                                        S.VCCRegister);
  };

  std::optional<llvm::DenseSet<uint64_t>> Partial =
      Compute(llvm::ArrayRef<llvm::StringRef>(
          {"s_mov_b32 vcc_lo, s0", "s_mov_b32 s1, vcc_hi", "s_endpgm"}));
  ASSERT_TRUE(Partial);
  EXPECT_TRUE(Partial->contains(0));

  std::optional<llvm::DenseSet<uint64_t>> Full =
      Compute(llvm::ArrayRef<llvm::StringRef>(
          {"s_mov_b64 vcc, -1", "s_mov_b32 s1, vcc_hi", "s_endpgm"}));
  ASSERT_TRUE(Full);
  EXPECT_FALSE(Full->contains(0));
  EXPECT_TRUE(Full->contains(MinInstSize));

  std::optional<llvm::DenseSet<uint64_t>> BranchUnion = Compute(
      llvm::ArrayRef<llvm::StringRef>({"s_cbranch_scc0 1", "s_mov_b64 vcc, -1",
                                       "s_mov_b32 s1, vcc_hi", "s_endpgm"}));
  ASSERT_TRUE(BranchUnion);
  EXPECT_TRUE(BranchUnion->contains(0));

  std::optional<llvm::DenseSet<uint64_t>> Opaque =
      Compute(llvm::ArrayRef<llvm::StringRef>({"s_set_pc_i64 s[0:1]"}));
  ASSERT_TRUE(Opaque);
  EXPECT_TRUE(Opaque->contains(0));

  std::optional<llvm::DenseSet<uint64_t>> PureLoop =
      Compute(llvm::ArrayRef<llvm::StringRef>({"s_branch -1"}));
  ASSERT_TRUE(PureLoop);
  EXPECT_FALSE(PureLoop->contains(0));

  std::optional<llvm::DenseSet<uint64_t>> LoopWithUnsafeExit = Compute(
      llvm::ArrayRef<llvm::StringRef>({"s_cbranch_scc0 1", "s_branch -2",
                                       "s_mov_b32 s1, vcc_hi", "s_endpgm"}));
  ASSERT_TRUE(LoopWithUnsafeExit);
  EXPECT_TRUE(LoopWithUnsafeExit->contains(0));
  EXPECT_TRUE(LoopWithUnsafeExit->contains(MinInstSize));

  expectBatchRegisterNeedsMatchesScalar(
      S,
      llvm::ArrayRef<llvm::StringRef>(
          {"s_mov_b32 vcc_lo, s0", "s_mov_b32 s1, vcc_hi", "s_endpgm"}),
      S.VCCRegister);
  expectBatchRegisterNeedsMatchesScalar(
      S,
      llvm::ArrayRef<llvm::StringRef>(
          {"s_mov_b64 vcc, -1", "s_mov_b32 s1, vcc_hi", "s_endpgm"}),
      S.VCCRegister);
  expectBatchRegisterNeedsMatchesScalar(
      S,
      llvm::ArrayRef<llvm::StringRef>({"s_cbranch_scc0 1", "s_mov_b64 vcc, -1",
                                       "s_mov_b32 s1, vcc_hi", "s_endpgm"}),
      S.VCCRegister);
  expectBatchRegisterNeedsMatchesScalar(
      S, llvm::ArrayRef<llvm::StringRef>({"s_set_pc_i64 s[0:1]"}),
      S.VCCRegister);
  expectBatchRegisterNeedsMatchesScalar(
      S, llvm::ArrayRef<llvm::StringRef>({"s_branch -1"}), S.VCCRegister);
  expectBatchRegisterNeedsMatchesScalar(
      S,
      llvm::ArrayRef<llvm::StringRef>({"s_cbranch_scc0 1", "s_branch -2",
                                       "s_mov_b32 s1, vcc_hi", "s_endpgm"}),
      S.VCCRegister);
}

TEST(RegisterLiveness, BatchProofMatchesScalarAcrossControlFlow) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  const llvm::StringRef BranchJoin[] = {"s_cbranch_scc0 1", "s_mov_b32 s30, 0",
                                        "s_mov_b32 s0, s30", "s_endpgm"};
  expectBatchSgprProofMatchesScalar(S, BranchJoin);

  const llvm::StringRef Loop[] = {"s_mov_b32 s30, 0", "s_cbranch_scc0 -2",
                                  "s_endpgm"};
  expectBatchSgprProofMatchesScalar(S, Loop);

  const llvm::StringRef DefBeforeOpaque[] = {"s_mov_b32 s30, 0",
                                             "s_set_pc_i64 s[0:1]"};
  expectBatchSgprProofMatchesScalar(S, DefBeforeOpaque);

  const llvm::StringRef OpaqueBeforeDef[] = {"s_set_pc_i64 s[0:1]",
                                             "s_mov_b32 s30, 0"};
  expectBatchSgprProofMatchesScalar(S, OpaqueBeforeDef);

  const llvm::StringRef TiedAndTuple[] = {"s_add_u32 s30, s30, 1",
                                          "s_mov_b64 s[30:31], s[0:1]",
                                          "s_mov_b32 s2, s31", "s_endpgm"};
  expectBatchSgprProofMatchesScalar(S, TiedAndTuple);

  const llvm::StringRef InvalidBranchEdge[] = {"s_cbranch_scc0 100",
                                               "s_endpgm"};
  expectBatchSgprProofMatchesScalar(S, InvalidBranchEdge);
}

TEST(FarReturnSgprCache, ReusesOneFunctionAnalysisAcrossSites) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  const llvm::StringRef Lines[] = {
      "s_cbranch_scc0 2",  "s_mov_b32 s104, 0",  "s_branch 1",
      "s_mov_b32 s105, 0", "s_mov_b32 s0, s103", "s_endpgm",
  };
  expectBatchSgprProofMatchesScalar(S, Lines);
}

TEST(RegisterLiveness, BatchedSgprProofMatchesRandomizedBranchLoops) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  uint64_t State = 0x9E3779B97F4A7C15ULL;
  auto Next = [&]() {
    State = State * 2862933555777941757ULL + 3037000493ULL;
    return State;
  };

  constexpr size_t InstructionCount = 10;
  for (unsigned Trial = 0; Trial != 20; ++Trial) {
    std::vector<std::string> Storage;
    Storage.reserve(InstructionCount);
    for (size_t I = 0; I + 1 < InstructionCount; ++I) {
      unsigned Dst = Next() % 16;
      unsigned Src = Next() % 16;
      switch (Next() % 5) {
      case 0:
        Storage.push_back("s_mov_b32 s" + std::to_string(Dst) + ", s" +
                          std::to_string(Src));
        break;
      case 1:
        Storage.push_back("s_mov_b32 s" + std::to_string(Dst) + ", 0");
        break;
      case 2: {
        size_t Target = Next() % InstructionCount;
        int64_t Delta =
            static_cast<int64_t>(Target) - static_cast<int64_t>(I) - 1;
        Storage.push_back("s_cbranch_scc0 " + std::to_string(Delta));
        break;
      }
      case 3: {
        size_t Target = Next() % InstructionCount;
        int64_t Delta =
            static_cast<int64_t>(Target) - static_cast<int64_t>(I) - 1;
        Storage.push_back("s_branch " + std::to_string(Delta));
        break;
      }
      default:
        Storage.push_back("s_add_u32 s" + std::to_string(Dst) + ", s" +
                          std::to_string(Dst) + ", 1");
        break;
      }
    }
    Storage.push_back("s_endpgm");
    llvm::SmallVector<llvm::StringRef, InstructionCount> Lines;
    for (const std::string &Line : Storage)
      Lines.push_back(Line);
    expectBatchSgprProofMatchesScalar(S, Lines);
  }
}

TEST(RegisterLiveness, BatchedSgprProofPreservesReplacementUnion) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  std::vector<InternalDecodedInst> Decoded = decodeAsmSequence(
      S,
      llvm::ArrayRef<llvm::StringRef>({"s_cbranch_scc0 1", "s_mov_b32 s104, 0",
                                       "s_mov_b32 s0, s103", "s_endpgm"}));
  ASSERT_FALSE(Decoded.empty());
  std::optional<llvm::SmallVector<llvm::MCRegister, 128>> NumberedSgprs =
      resolveNumberedSgprRegisters(*S.MRI, /*MaxSgprs=*/106);
  ASSERT_TRUE(NumberedSgprs);
  uint64_t FunctionEnd = Decoded.back().Offset + Decoded.back().Size;
  uint64_t Continuation = Decoded[1].Offset;
  BatchedSgprContinuationTestResult Batch =
      runBatchedSgprContinuationAnalysisForTest(Decoded, S, /*FunctionBegin=*/0,
                                                FunctionEnd, {Continuation},
                                                *NumberedSgprs);
  ASSERT_EQ(Batch.Analyses, 1u);
  ASSERT_EQ(Batch.Queries.size(), 1u);
  ASSERT_TRUE(Batch.Queries.front());
  std::optional<llvm::BitVector> Scalar = unsafeIncomingNumberedSgprsInRange(
      Decoded, S, /*FunctionBegin=*/0, FunctionEnd, Continuation,
      *NumberedSgprs);
  ASSERT_TRUE(Scalar);

  llvm::SmallVector<uint8_t> Replacement =
      assembleInstructions("s_mov_b32 s102, 0\ns_mov_b32 s1, s101", S);
  ASSERT_FALSE(Replacement.empty());
  llvm::BitVector ReplacementUnsafe =
      unsafeIncomingNumberedSgprsInReplacement(Replacement, S, *NumberedSgprs);
  llvm::BitVector BatchedUnion = *Batch.Queries.front();
  BatchedUnion |= ReplacementUnsafe;
  llvm::BitVector ScalarUnion = *Scalar;
  ScalarUnion |= ReplacementUnsafe;
  EXPECT_EQ(BatchedUnion, ScalarUnion);
}

TEST(RegisterLiveness, NumberedSgprExtractionCoversAliasesAndTiedRmw) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  std::optional<llvm::SmallVector<llvm::MCRegister, 128>> NumberedSgprs =
      resolveNumberedSgprRegisters(*S.MRI, /*MaxSgprs=*/106);
  ASSERT_TRUE(NumberedSgprs);

  auto GetUseDef = [&](const InternalDecodedInst &DI) {
    std::pair<llvm::BitVector, llvm::BitVector> Result{
        llvm::BitVector(NumberedSgprs->size()),
        llvm::BitVector(NumberedSgprs->size())};
    getNumberedSgprUsesAndDefs(DI, S, *NumberedSgprs, Result.first,
                               Result.second);
    return Result;
  };

  std::vector<InternalDecodedInst> Tuple = decodeAsmSequence(
      S, llvm::ArrayRef<llvm::StringRef>({"s_mov_b64 s[0:1], s[30:31]"}));
  ASSERT_EQ(Tuple.size(), 1u);
  auto [TupleUses, TupleDefs] = GetUseDef(Tuple.front());
  EXPECT_TRUE(TupleUses.test(30));
  EXPECT_TRUE(TupleUses.test(31));
  EXPECT_TRUE(TupleDefs.test(0));
  EXPECT_TRUE(TupleDefs.test(1));

  std::vector<InternalDecodedInst> Rmw = decodeAsmSequence(
      S, llvm::ArrayRef<llvm::StringRef>({"s_add_u32 s30, s30, 1"}));
  ASSERT_EQ(Rmw.size(), 1u);
  auto [RmwUses, RmwDefs] = GetUseDef(Rmw.front());
  EXPECT_TRUE(RmwUses.test(30));
  EXPECT_TRUE(RmwDefs.test(30));

  llvm::MCRegister Low16;
  for (unsigned I = 1; I != S.MRI->getNumRegs(); ++I) {
    llvm::MCRegister Candidate(I);
    if (llvm::StringRef(S.MRI->getName(Candidate)) == "SGPR30_LO16") {
      Low16 = Candidate;
      break;
    }
  }
  ASSERT_TRUE(Low16.isValid());

  std::vector<InternalDecodedInst> Half = decodeAsmSequence(
      S, llvm::ArrayRef<llvm::StringRef>({"s_mov_b32 s0, s1"}));
  ASSERT_EQ(Half.size(), 1u);
  ASSERT_GE(Half.front().Inst.getNumOperands(), 2u);
  Half.front().Inst.getOperand(1).setReg(Low16);
  auto [HalfUses, HalfDefs] = GetUseDef(Half.front());
  EXPECT_TRUE(HalfUses.test(30));
  EXPECT_FALSE(HalfUses.test(1));
  EXPECT_TRUE(HalfDefs.test(0));

  Half.front().Inst.getOperand(0).setReg(Low16);
  auto [HalfDefUses, HalfDefDefs] = GetUseDef(Half.front());
  EXPECT_TRUE(HalfDefUses.test(30));
  EXPECT_TRUE(HalfDefDefs.test(30));
}

TEST(RegisterLiveness, ReplacementTracksOnlyIncomingValues) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);
  std::optional<llvm::SmallVector<llvm::MCRegister, 128>> NumberedSgprs =
      resolveNumberedSgprRegisters(*S.MRI, /*MaxSgprs=*/106);
  ASSERT_TRUE(NumberedSgprs);

  llvm::SmallVector<uint8_t> UseBeforeDef =
      assembleInstructions("s_mov_b32 s0, s30\ns_mov_b32 s30, 0", S);
  ASSERT_FALSE(UseBeforeDef.empty());
  llvm::BitVector UseBeforeDefUnsafe =
      unsafeIncomingNumberedSgprsInReplacement(UseBeforeDef, S, *NumberedSgprs);
  EXPECT_TRUE(UseBeforeDefUnsafe.test(30));

  llvm::SmallVector<uint8_t> DefBeforeUse =
      assembleInstructions("s_mov_b32 s30, 0\ns_mov_b32 s0, s30", S);
  ASSERT_FALSE(DefBeforeUse.empty());
  llvm::BitVector DefBeforeUseUnsafe =
      unsafeIncomingNumberedSgprsInReplacement(DefBeforeUse, S, *NumberedSgprs);
  EXPECT_FALSE(DefBeforeUseUnsafe.test(30));

  llvm::SmallVector<uint8_t> Opaque =
      assembleSingleInst("s_set_pc_i64 s[0:1]", S);
  ASSERT_FALSE(Opaque.empty());
  llvm::BitVector OpaqueUnsafe =
      unsafeIncomingNumberedSgprsInReplacement(Opaque, S, *NumberedSgprs);
  EXPECT_TRUE(OpaqueUnsafe.test(30));
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

// -- DS two-address expansion ------------------------------------------------

TEST(ExpandDs2Addr, PreservesAddressNeededBySecondLoad) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  llvm::SmallVector<uint8_t> Bytes = assembleSingleInst(
      "ds_load_2addr_b64 v[12:15], v12 offset0:0 offset1:1", S);
  ASSERT_FALSE(Bytes.empty());
  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 1u);

  std::optional<std::vector<std::string>> Expanded =
      expandDs2Addr(Decoded[0].Inst, Decoded[0].Mnemonic, "ds_load_b64", S);
  ASSERT_TRUE(Expanded);
  ASSERT_EQ(Expanded->size(), 2u);
  EXPECT_EQ((*Expanded)[0], "ds_load_b64 v[14:15], v12 offset:8");
  EXPECT_EQ((*Expanded)[1], "ds_load_b64 v[12:13], v12");
}

TEST(ExpandDs2Addr, RejectsCyclicExchangeDependency) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  llvm::SmallVector<uint8_t> Bytes = assembleSingleInst(
      "ds_storexchg_2addr_rtn_b64 v[20:23], v24, v[22:23], v[20:21] "
      "offset0:0 offset1:1",
      S);
  ASSERT_FALSE(Bytes.empty());
  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_EQ(Decoded.size(), 1u);

  EXPECT_FALSE(expandDs2Addr(Decoded[0].Inst, Decoded[0].Mnemonic,
                             "ds_storexchg_rtn_b64", S));
}

// -- buildKernelEntryTrampoline -----------------------------------------------

TEST(BuildKernelEntryTrampoline, BuildsRecognizedPcRelativeStub) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  constexpr uint64_t StubVAddr = 0x200000;
  constexpr uint64_t EntryVAddr = 0x10100;
  llvm::SmallVector<uint8_t> Prefetch =
      assembleSingleInst("global_prefetch_b8 v0, s[0:1] scope:SCOPE_SE", S);
  ASSERT_EQ(Prefetch.size(), 3 * MinInstSize);

  llvm::SmallVector<uint8_t> Bytes =
      buildKernelEntryTrampoline(StubVAddr, EntryVAddr, /*ScratchSgpr=*/8, S);

  ASSERT_EQ(Bytes.size(), KernelEntryStubStride);
  EXPECT_TRUE(isKernelEntryTrampoline(Bytes, S));

  std::vector<InternalDecodedInst> Decoded;
  ASSERT_TRUE(decodeTextSection(Bytes.data(), Bytes.size(), S, Decoded));
  ASSERT_GE(Decoded.size(), 6u);
  EXPECT_EQ(Decoded[0].Inst.getOpcode(), S.GlobalPrefetchB8Opcode);
  EXPECT_EQ(Decoded[1].Inst.getOpcode(), S.VNopInst.getOpcode());
  EXPECT_EQ(Decoded[2].Inst.getOpcode(), S.SGetPcI64Opcode);
  EXPECT_EQ(Decoded[3].Inst.getOpcode(), S.SAddU32Opcode);
  EXPECT_EQ(Decoded[4].Inst.getOpcode(), S.SAddcU32Opcode);
  EXPECT_EQ(Decoded[5].Inst.getOpcode(), S.SSetPcI64Opcode);

  const uint64_t PcBase = StubVAddr + Decoded[2].Offset + Decoded[2].Size;
  const uint64_t Delta = EntryVAddr - PcBase;
  const uint32_t Lo = static_cast<uint32_t>(Delta);
  const uint32_t Hi = static_cast<uint32_t>(Delta >> 32);
  expectInstMatchesAsm(Decoded[0].Inst,
                       "global_prefetch_b8 v0, s[0:1] scope:SCOPE_SE", S);
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
  ASSERT_TRUE(appendSingleInstBytes(
      Bytes, "global_prefetch_b8 v0, s[0:1] scope:SCOPE_SE", S));
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

  llvm::SmallVector<uint8_t> Prefix = assembleInstructions(
      "global_prefetch_b8 v0, s[0:1] scope:SCOPE_SE\nv_nop", S);
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
      ViewOrErr->growWithTrampolines(Growth, S.SNopBytes);
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
  EXPECT_EQ(Fixups[0].RequiredSgprs, 10u);
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
      View1->growWithTrampolines(Growth1, S.SNopBytes);
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

// A `global_prefetch_b8 v0, s[0:1] scope:SCOPE_SE; v_nop` prologue
// (llvm/llvm-project#208467, updated by ROCm/llvm-project#3483) already
// satisfies the workaround, so no trampoline is installed.
TEST(KernelEntryTrampoline, SkipsWhenPrologueAlreadyHasVmemWorkaround) {
  LLVMState S = initLLVM(makeGfx1250Ident());
  ASSERT_TRUE(S.Valid);

  llvm::SmallVector<uint8_t> Prefetch =
      assembleSingleInst("global_prefetch_b8 v0, s[0:1] scope:SCOPE_SE", S);
  llvm::SmallVector<uint8_t> VNop = assembleSingleInst("v_nop", S);
  llvm::SmallVector<uint8_t> EndPgm = assembleSingleInst("s_endpgm", S);
  ASSERT_FALSE(Prefetch.empty());
  ASSERT_FALSE(VNop.empty());
  ASSERT_EQ(EndPgm.size(), MinInstSize);

  llvm::SmallVector<uint8_t> Text;
  Text.append(Prefetch.begin(), Prefetch.end());
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
  llvm::SmallVector<uint8_t> Prefetch =
      assembleSingleInst("global_prefetch_b8 v0, s[0:1] scope:SCOPE_SE", S);
  llvm::SmallVector<uint8_t> EndPgm = assembleSingleInst("s_endpgm", S);
  ASSERT_FALSE(VNop.empty());
  ASSERT_FALSE(Prefetch.empty());
  ASSERT_EQ(EndPgm.size(), MinInstSize);

  llvm::SmallVector<uint8_t> Text;
  Text.append(VNop.begin(), VNop.end());
  Text.append(Prefetch.begin(), Prefetch.end());
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

  Trampoline Existing;
  Existing.Bytes.assign(S.SNopBytes.begin(), S.SNopBytes.end());
  std::vector<Trampoline> Growth;
  Growth.push_back(Existing);
  std::vector<KernelEntryTrampolineFixup> Fixups;
  std::optional<uint32_t> Count = appendKernelEntryTrampolines(
      *ViewOrErr, S, /*MaxSgprs=*/106, Growth, Fixups);

  EXPECT_FALSE(Count.has_value());
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

TEST(LivenessInfo, ConservativeFallbackSharesOneAllLiveVector) {
  LivenessInfo Info;
  std::vector<llvm::BitVector> Before(3, llvm::BitVector(64));
  std::vector<llvm::BitVector> After(3, llvm::BitVector(64));
  Info.setPerInstructionLiveness(std::move(Before), std::move(After));
  ASSERT_EQ(Info.perInstructionCount(), 3u);

  Info.setConservativeAllLive(/*MaxVgprs=*/64);

  EXPECT_TRUE(Info.usesConservativeAllLive());
  EXPECT_EQ(Info.perInstructionCount(), 0u);
  ASSERT_EQ(Info.liveBefore(0).size(), 64u);
  EXPECT_TRUE(Info.liveBefore(0).all());
  EXPECT_EQ(&Info.liveBefore(0), &Info.liveBefore(1));
  EXPECT_EQ(&Info.liveBefore(1), &Info.liveAfter(2));
}

TEST(LivenessInfo, PerInstructionAccessorsReturnIndexedVectors) {
  LivenessInfo Info;
  std::vector<llvm::BitVector> Before(3, llvm::BitVector(64));
  std::vector<llvm::BitVector> After(3, llvm::BitVector(64));
  Before[1].set(7);
  After[2].set(9);
  Info.setPerInstructionLiveness(std::move(Before), std::move(After));

  EXPECT_FALSE(Info.usesConservativeAllLive());
  EXPECT_EQ(Info.perInstructionCount(), 3u);
  EXPECT_FALSE(Info.liveBefore(0).test(7));
  EXPECT_TRUE(Info.liveBefore(1).test(7));
  EXPECT_TRUE(Info.liveAfter(2).test(9));
  EXPECT_NE(&Info.liveBefore(0), &Info.liveBefore(1));
}

TEST(LivenessInfo, ZeroVgprConservativeModeIsExplicit) {
  LivenessInfo Info;
  Info.setConservativeAllLive(/*MaxVgprs=*/0);

  EXPECT_TRUE(Info.usesConservativeAllLive());
  EXPECT_EQ(Info.perInstructionCount(), 0u);
  EXPECT_TRUE(Info.liveBefore(0).empty());
  EXPECT_EQ(&Info.liveBefore(0), &Info.liveAfter(0));
}
