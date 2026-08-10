//===- disasm_symbolizer_edge_test.c --------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Covers the disassembler and symbolizer paths the happy-path tests miss:
/// target identifiers carrying features, the three ways
/// amd_comgr_disassemble_instruction can fail, symbolizing data rather than
/// code, and building a symbolizer over a buffer that is not an object file.
///
//===----------------------------------------------------------------------===//

#include "amd_comgr.h"
#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// A v_mov_b32 followed by s_endpgm: enough for one decodable instruction.
static const char InstructionBytes[] = {(char)0x80, (char)0x02, (char)0x02,
                                        (char)0x7e};

typedef struct {
  const char *Bytes;
  size_t Size;
  // How many bytes to claim were read, overriding the real count when non-zero.
  uint64_t ForcedSize;
  size_t PrintedInstructions;
} ReadState;

static uint64_t readMemory(uint64_t From, char *To, uint64_t Size,
                           void *UserData) {
  ReadState *State = (ReadState *)UserData;
  if (State->ForcedSize)
    return State->ForcedSize;
  uint64_t Copy = Size < State->Size ? Size : State->Size;
  memcpy(To, State->Bytes, (size_t)Copy);
  return Copy;
}

static void printInstruction(const char *Instruction, void *UserData) {
  ReadState *State = (ReadState *)UserData;
  State->PrintedInstructions += 1;
}

static void printAddressAnnotation(uint64_t Address, void *UserData) {}

static amd_comgr_disassembly_info_t makeDisassembly(const char *IsaName) {
  amd_comgr_disassembly_info_t Info;
  checkError(amd_comgr_create_disassembly_info(IsaName, &readMemory,
                                               &printInstruction,
                                               &printAddressAnnotation, &Info),
             "amd_comgr_create_disassembly_info");
  return Info;
}

// A target identifier may carry features, which the disassembler folds into the
// subtarget feature string. Plain and feature-bearing names take different
// paths through DisassemblyInfo::create.
static void testIsaVariants(void) {
  const char *IsaNames[] = {
      "amdgcn-amd-amdhsa--gfx900",
      "amdgcn-amd-amdhsa--gfx906",
      "amdgcn-amd-amdhsa--gfx906:xnack+",
      "amdgcn-amd-amdhsa--gfx906:xnack-",
      "amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-",
      "amdgcn-amd-amdhsa--gfx1030",
  };

  for (size_t I = 0; I < sizeof(IsaNames) / sizeof(IsaNames[0]); ++I) {
    amd_comgr_disassembly_info_t Info = makeDisassembly(IsaNames[I]);
    checkError(amd_comgr_destroy_disassembly_info(Info),
               "amd_comgr_destroy_disassembly_info");
  }
}

static void testDisassembleSuccess(void) {
  amd_comgr_disassembly_info_t Info =
      makeDisassembly("amdgcn-amd-amdhsa--gfx900");
  ReadState State;
  uint64_t Size = 0;

  memset(&State, 0, sizeof(State));
  State.Bytes = InstructionBytes;
  State.Size = sizeof(InstructionBytes);

  checkError(amd_comgr_disassemble_instruction(Info, 0, &State, &Size),
             "amd_comgr_disassemble_instruction");
  if (Size == 0)
    fail("disassembling a valid instruction consumed 0 bytes");
  if (State.PrintedInstructions != 1)
    fail("expected exactly one printed instruction, saw %zu",
         State.PrintedInstructions);

  checkError(amd_comgr_destroy_disassembly_info(Info),
             "amd_comgr_destroy_disassembly_info");
}

// disassembleInstruction rejects a read that returns nothing, a read that
// claims more than it was asked for, and bytes that do not decode.
static void testDisassembleFailures(void) {
  amd_comgr_disassembly_info_t Info =
      makeDisassembly("amdgcn-amd-amdhsa--gfx900");
  const char Undecodable[] = {(char)0xff, (char)0xff, (char)0xff, (char)0xff,
                              (char)0xff, (char)0xff, (char)0xff, (char)0xff};
  ReadState State;
  uint64_t Size = 0;

  memset(&State, 0, sizeof(State));
  State.Bytes = InstructionBytes;
  State.Size = sizeof(InstructionBytes);

  State.ForcedSize = 0;
  State.Size = 0;
  checkStatus(amd_comgr_disassemble_instruction(Info, 0, &State, &Size),
              AMD_COMGR_STATUS_ERROR,
              "amd_comgr_disassemble_instruction with an empty read");

  // Claiming more bytes than the requested maximum is rejected.
  memset(&State, 0, sizeof(State));
  State.Bytes = InstructionBytes;
  State.Size = sizeof(InstructionBytes);
  State.ForcedSize = 4096;
  checkStatus(amd_comgr_disassemble_instruction(Info, 0, &State, &Size),
              AMD_COMGR_STATUS_ERROR,
              "amd_comgr_disassemble_instruction with an oversized read");

  memset(&State, 0, sizeof(State));
  State.Bytes = Undecodable;
  State.Size = sizeof(Undecodable);
  checkStatus(amd_comgr_disassemble_instruction(Info, 0, &State, &Size),
              AMD_COMGR_STATUS_ERROR,
              "amd_comgr_disassemble_instruction with undecodable bytes");

  checkError(amd_comgr_destroy_disassembly_info(Info),
             "amd_comgr_destroy_disassembly_info");
}

typedef struct {
  size_t Calls;
  char Last[512];
} SymbolState;

static void collectSymbol(const char *Symbol, void *UserData) {
  SymbolState *State = (SymbolState *)UserData;
  State->Calls += 1;
  strncpy(State->Last, Symbol, sizeof(State->Last) - 1);
  State->Last[sizeof(State->Last) - 1] = '\0';
}

// A buffer that is not an object file cannot back a symbolizer.
static void testSymbolizerOnNonObject(void) {
  const char *Garbage = "not an object file";
  amd_comgr_data_t Data;
  amd_comgr_symbolizer_info_t Symbolizer;

  checkError(amd_comgr_create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &Data),
             "amd_comgr_create_data");
  checkError(amd_comgr_set_data(Data, strlen(Garbage), Garbage),
             "amd_comgr_set_data");

  checkStatus(
      amd_comgr_create_symbolizer_info(Data, &collectSymbol, &Symbolizer),
      AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT,
      "amd_comgr_create_symbolizer_info on a non-object buffer");

  checkError(amd_comgr_release_data(Data), "amd_comgr_release_data");
}

// Symbolizing as code and as data are separate paths, and an address outside
// any function still produces output rather than an error.
static void testSymbolizeCodeAndData(void) {
  char *Buf;
  long Size;
  amd_comgr_data_t Data;
  amd_comgr_symbolizer_info_t Symbolizer;
  SymbolState State;

  Size = setBuf(TEST_OBJ_DIR "/symbolize-debug.so", &Buf);

  checkError(amd_comgr_create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &Data),
             "amd_comgr_create_data");
  checkError(amd_comgr_set_data(Data, Size, Buf), "amd_comgr_set_data");
  checkError(
      amd_comgr_create_symbolizer_info(Data, &collectSymbol, &Symbolizer),
      "amd_comgr_create_symbolizer_info");

  memset(&State, 0, sizeof(State));
  checkError(amd_comgr_symbolize(Symbolizer, 0x100, true, &State),
             "amd_comgr_symbolize as code");
  if (State.Calls != 1)
    fail("symbolizing as code invoked the callback %zu times", State.Calls);

  memset(&State, 0, sizeof(State));
  checkError(amd_comgr_symbolize(Symbolizer, 0x100, false, &State),
             "amd_comgr_symbolize as data");
  if (State.Calls != 1)
    fail("symbolizing as data invoked the callback %zu times", State.Calls);

  // An address far outside the code object is reported, not rejected.
  memset(&State, 0, sizeof(State));
  checkError(amd_comgr_symbolize(Symbolizer, 0xffffffff, true, &State),
             "amd_comgr_symbolize an out-of-range address");
  if (State.Calls != 1)
    fail("symbolizing an out-of-range address invoked the callback %zu times",
         State.Calls);

  checkError(amd_comgr_destroy_symbolizer_info(Symbolizer),
             "amd_comgr_destroy_symbolizer_info");
  checkError(amd_comgr_release_data(Data), "amd_comgr_release_data");
  free(Buf);
}

int main(int argc, char *argv[]) {
  testIsaVariants();
  testDisassembleSuccess();
  testDisassembleFailures();
  testSymbolizerOnNonObject();
  testSymbolizeCodeAndData();
  printf("disasm_symbolizer_edge_test passed\n");
  return 0;
}
