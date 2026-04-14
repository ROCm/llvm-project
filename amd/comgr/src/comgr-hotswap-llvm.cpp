//===- comgr-hotswap-llvm.cpp - LLVM MC infrastructure, decode/encode -----===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-internal.h"

#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"

static const llvm::Triple kAMDGPUTriple("amdgcn-amd-amdhsa");

namespace {
std::once_flag g_llvm_init_flag;
std::mutex g_target_cache_mutex;
const llvm::Target *g_cached_target = nullptr;
} // namespace

static void InitLLVMTargets() {
  LLVMInitializeAMDGPUTargetInfo();
  LLVMInitializeAMDGPUTargetMC();
  LLVMInitializeAMDGPUAsmParser();
  LLVMInitializeAMDGPUDisassembler();
}

LLVMState InitLLVMImpl(const std::string &isa_name,
                       const llvm::Target *cached_target) {
  std::call_once(g_llvm_init_flag, InitLLVMTargets);

  LLVMState state;
  state.cpu = ExtractCPU(isa_name);
  if (state.cpu.empty()) return state;

  if (cached_target) {
    state.target = cached_target;
  } else {
    std::string error;
    llvm::Triple triple(kAMDGPUTriple);
    state.target = llvm::TargetRegistry::lookupTarget("amdgcn", triple, error);
  }
  if (!state.target) return state;

  state.MRI.reset(state.target->createMCRegInfo(kAMDGPUTriple));
  if (!state.MRI) return state;

  llvm::MCTargetOptions mc_opts;
  state.MAI.reset(
      state.target->createMCAsmInfo(*state.MRI, kAMDGPUTriple, mc_opts));
  if (!state.MAI) return state;

  state.MCII.reset(state.target->createMCInstrInfo());
  if (!state.MCII) return state;

  state.STI.reset(
      state.target->createMCSubtargetInfo(kAMDGPUTriple, state.cpu, ""));
  if (!state.STI || !state.STI->isCPUStringValid(state.cpu)) return state;

  state.Ctx = std::make_unique<llvm::MCContext>(
      kAMDGPUTriple, state.MAI.get(), state.MRI.get(), state.STI.get());
  state.MOFI = std::make_unique<llvm::MCObjectFileInfo>();
  state.MOFI->initMCObjectFileInfo(*state.Ctx, false);
  state.Ctx->setObjectFileInfo(state.MOFI.get());

  state.disasm.reset(
      state.target->createMCDisassembler(*state.STI, *state.Ctx));
  if (!state.disasm) return state;

  unsigned asm_variant = state.MAI->getAssemblerDialect();
  state.printer.reset(state.target->createMCInstPrinter(
      kAMDGPUTriple, asm_variant, *state.MAI, *state.MCII, *state.MRI));

  state.CE.reset(state.target->createMCCodeEmitter(*state.MCII, *state.Ctx));

  state.valid = true;
  return state;
}

LLVMState InitLLVMCached(const std::string &isa_name) {
  std::call_once(g_llvm_init_flag, InitLLVMTargets);

  const llvm::Target *tgt;
  {
    std::lock_guard<std::mutex> lock(g_target_cache_mutex);
    if (!g_cached_target) {
      std::string error;
      llvm::Triple triple(kAMDGPUTriple);
      g_cached_target =
          llvm::TargetRegistry::lookupTarget("amdgcn", triple, error);
    }
    tgt = g_cached_target;
  }

  return InitLLVMImpl(isa_name, tgt);
}

// ── Instruction helpers ──────────────────────────────────────────────────────

static std::string ExtractMnemonic(const std::string &printed) {
  size_t s = printed.find_first_not_of(" \t");
  if (s == std::string::npos) return "";
  size_t e = printed.find_first_of(" \t", s);
  return printed.substr(s, e - s);
}

static std::string PrintInstStr(const llvm::MCInst &inst,
                                const LLVMState &llvm_state) {
  std::string str;
  llvm::raw_string_ostream rso(str);
  llvm_state.printer->printInst(&inst, 0, "", *llvm_state.STI, rso);
  rso.flush();
  return str;
}

// ── Instruction decode ───────────────────────────────────────────────────────

[[nodiscard]] bool DecodeTextSection(const uint8_t *text, uint64_t text_size,
                                     const LLVMState &llvm_state,
                                     std::vector<InternalDecodedInst> &decoded) {
  uint64_t pos = 0;
  while (pos < text_size) {
    InternalDecodedInst di;
    di.offset = pos;

    llvm::ArrayRef<uint8_t> bytes(text + pos, text_size - pos);
    uint64_t inst_size = 0;

    auto status = llvm_state.disasm->getInstruction(di.inst, inst_size, bytes,
                                                    pos, llvm::nulls());

    if (status == llvm::MCDisassembler::Fail) {
      di.size = kMinInstSize;
      di.mnemonic = "<unknown>";
      pos += kMinInstSize;
    } else {
      di.size = static_cast<uint32_t>(inst_size);
      if (llvm_state.printer)
        di.mnemonic = ExtractMnemonic(PrintInstStr(di.inst, llvm_state));
      else
        di.mnemonic = llvm_state.MCII->getName(di.inst.getOpcode()).str();
      pos += inst_size;
    }
    decoded.push_back(std::move(di));
  }
  return true;
}

// ── AssembleSingleInst ───────────────────────────────────────────────────────

std::vector<uint8_t> AssembleSingleInst(const std::string &asm_str,
                                        const LLVMState &llvm_state) {
  llvm_state.Ctx->reset();

  std::string full_asm = ".text\n" + asm_str;
  llvm::StringRef asm_ref(full_asm);
  auto buf = llvm::MemoryBuffer::getMemBuffer(asm_ref, "", false);
  llvm::SourceMgr src_mgr;
  src_mgr.AddNewSourceBuffer(std::move(buf), llvm::SMLoc());

  std::string data;
  auto data_stream = std::make_unique<llvm::raw_string_ostream>(data);
  auto bos = std::make_unique<llvm::buffer_ostream>(*data_stream);

  llvm::MCTargetOptions mc_opts;

  llvm::MCCodeEmitter *ce =
      llvm_state.target->createMCCodeEmitter(*llvm_state.MCII, *llvm_state.Ctx);
  llvm::MCAsmBackend *mab = llvm_state.target->createMCAsmBackend(
      *llvm_state.STI, *llvm_state.MRI, mc_opts);

  if (!ce || !mab) return {};

  auto streamer = std::unique_ptr<llvm::MCStreamer>(
      llvm_state.target->createMCObjectStreamer(
          kAMDGPUTriple, *llvm_state.Ctx,
          std::unique_ptr<llvm::MCAsmBackend>(mab),
          mab->createObjectWriter(*bos),
          std::unique_ptr<llvm::MCCodeEmitter>(ce), *llvm_state.STI));

  if (!streamer) return {};

  auto parser = std::unique_ptr<llvm::MCAsmParser>(
      llvm::createMCAsmParser(src_mgr, *llvm_state.Ctx, *streamer,
                              *llvm_state.MAI));
  auto tap = std::unique_ptr<llvm::MCTargetAsmParser>(
      llvm_state.target->createMCAsmParser(*llvm_state.STI, *parser,
                                           *llvm_state.MCII, mc_opts));
  if (!tap) return {};
  parser->setTargetParser(*tap);

  if (parser->Run(true)) return {};

  bos.reset();
  data_stream->flush();

  const uint8_t *elf_bytes = reinterpret_cast<const uint8_t *>(data.data());
  size_t elf_sz = data.size();
  if (elf_sz < kMinElfSize) return {};

  ElfInfo asm_elf;
  if (!ParseElfInfo(elf_bytes, elf_sz, asm_elf)) return {};
  if (asm_elf.text_size == 0) return {};

  return std::vector<uint8_t>(elf_bytes + asm_elf.text_offset,
                              elf_bytes + asm_elf.text_offset +
                                  asm_elf.text_size);
}

// ── ApplyMnemonicSwap ────────────────────────────────────────────────────────

[[nodiscard]] bool ApplyMnemonicSwap(const RewriteRule &rule,
                                     InternalDecodedInst &inst, uint8_t *text,
                                     const LLVMState &llvm_state) {
  if (!llvm_state.printer) return false;

  std::string printed = PrintInstStr(inst.inst, llvm_state);
  size_t start = printed.find_first_not_of(" \t");
  if (start == std::string::npos) return false;
  size_t end = printed.find_first_of(" \t", start);

  std::string new_asm = (end != std::string::npos)
                            ? rule.replace_mnemonic + printed.substr(end)
                            : rule.replace_mnemonic;

  auto bytes = AssembleSingleInst(new_asm, llvm_state);
  if (bytes.empty() || bytes.size() != inst.size) return false;

  std::memcpy(text + inst.offset, bytes.data(), inst.size);
  return true;
}

// ── BuildTrampoline ──────────────────────────────────────────────────────────

Trampoline BuildTrampoline(const std::vector<std::string> &asm_lines,
                           uint64_t original_offset,
                           uint32_t original_size,
                           uint64_t trampoline_text_offset,
                           const RewriteConfig &config,
                           const LLVMState &llvm_state) {
  Trampoline result;
  result.original_offset = original_offset;
  result.original_size = original_size;

  static constexpr llvm::StringLiteral kTextDirective(".text\n");
  std::string asm_source(kTextDirective);
  for (auto &line : asm_lines)
    asm_source += line + "\n";

  auto bytes = AssembleSingleInst(
      asm_source.substr(kTextDirective.size()), llvm_state);
  if (bytes.empty()) return result;

  result.bytes = std::move(bytes);

  uint64_t branch_back_from = trampoline_text_offset + result.bytes.size();
  uint64_t branch_back_to = original_offset + original_size;

  uint8_t branch_bytes[kMinInstSize];
  if (!EncodeSBranch(branch_back_from, branch_back_to, branch_bytes,
                     config.s_branch_opcode)) {
    result.bytes.clear();
    return result;
  }

  result.bytes.insert(result.bytes.end(), branch_bytes,
                      branch_bytes + kMinInstSize);
  return result;
}

// ── VGPR introspection ───────────────────────────────────────────────────────

static constexpr llvm::StringLiteral kVgprPrefix("VGPR");

int GetVgprNum(unsigned reg, const llvm::MCRegisterInfo &MRI) {
  const char *name = MRI.getName(reg);
  if (!name) return -1;
  llvm::StringRef rname(name);
  if (!rname.starts_with(kVgprPrefix)) return -1;
  llvm::StringRef numpart = rname.drop_front(kVgprPrefix.size());
  size_t underscore = numpart.find('_');
  if (underscore != llvm::StringRef::npos)
    numpart = numpart.take_front(underscore);
  int val = -1;
  auto [ptr, ec] =
      std::from_chars(numpart.data(), numpart.data() + numpart.size(), val);
  if (ec != std::errc()) return -1;
  return val;
}

std::pair<int, int> GetVgprRange(unsigned reg,
                                 const llvm::MCRegisterInfo &MRI) {
  const char *name = MRI.getName(reg);
  if (!name) return {-1, 0};
  llvm::StringRef rname(name);
  if (!rname.starts_with(kVgprPrefix)) return {-1, 0};
  int count = 1;
  for (char c : rname)
    if (c == '_') count++;
  llvm::StringRef numpart = rname.drop_front(kVgprPrefix.size());
  size_t numend = numpart.find_first_not_of("0123456789");
  if (numend != llvm::StringRef::npos)
    numpart = numpart.take_front(numend);
  int base = -1;
  auto [p, ec] =
      std::from_chars(numpart.data(), numpart.data() + numpart.size(), base);
  if (ec != std::errc())
    return {-1, 0};
  return {base, count};
}

std::pair<int, int>
GetOperandVgprRange(const llvm::MCInst &inst, unsigned op_idx,
                    const llvm::MCRegisterInfo &MRI) {
  if (op_idx >= inst.getNumOperands()) return {-1, 0};
  const auto &op = inst.getOperand(op_idx);
  if (!op.isReg()) return {-1, 0};
  return GetVgprRange(op.getReg(), MRI);
}

std::string PrintInst(const InternalDecodedInst &di,
                      const LLVMState &llvm_state) {
  if (llvm_state.printer)
    return PrintInstStr(di.inst, llvm_state);
  return "";
}

bool RangesOverlap(int base1, int count1, int base2, int count2) {
  if (base1 < 0 || base2 < 0) return false;
  return base1 < base2 + count2 && base2 < base1 + count1;
}

// ── WMMA co-execution hazard overlap check ──────────────────────────────────

bool CheckVgprOverlap(const llvm::MCInst &wmma_inst,
                      const llvm::MCInst &valu_inst,
                      const llvm::MCRegisterInfo &MRI) {
  std::vector<std::pair<int, int>> wmma_input_ranges;
  for (unsigned i = 0; i < wmma_inst.getNumOperands(); ++i) {
    const auto &op = wmma_inst.getOperand(i);
    if (!op.isReg()) continue;
    auto range = GetVgprRange(op.getReg(), MRI);
    if (range.first >= 0) wmma_input_ranges.push_back(range);
  }
  if (wmma_input_ranges.empty()) return false;

  static constexpr unsigned kDestOperandIdx = 0;
  if (valu_inst.getNumOperands() == 0) return false;
  const auto &dest_op = valu_inst.getOperand(kDestOperandIdx);
  if (!dest_op.isReg()) return false;
  auto valu_dest = GetVgprRange(dest_op.getReg(), MRI);
  if (valu_dest.first < 0) return false;

  for (const auto &wr : wmma_input_ranges) {
    if (RangesOverlap(wr.first, wr.second, valu_dest.first, valu_dest.second))
      return true;
  }
  return false;
}
