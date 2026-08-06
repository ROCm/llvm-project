//===- compile-concurrent-hip-differing-options.cpp ------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--------------------------------------------------------------------===//
//
// Compiles the same HIP source through amd_comgr_do_action from many threads
// concurrently, but unlike compile-concurrent-hip.cpp, alternating threads
// request *different* codegen-visible -mllvm options
// (-amdgpu-function-calls=true/false) on the CODEGEN_BC_TO_RELOCATABLE
// action. This exercises comgr's LLVM cl::opt option-fingerprint cache: if
// one thread's -mllvm option were to leak into another thread's codegen
// invocation, the affected thread's disassembly would show the wrong
// inlining decision for its helper function, which the accompanying lit
// test detects.
//
// -amdgpu-function-calls=false forces AMDGPUAlwaysInlinePass to mark every
// used, non-declaration function AlwaysInline (see
// llvm/lib/Target/AMDGPU/AMDGPUAlwaysInlinePass.cpp); the subsequent
// AlwaysInlinerPass then inlines and deletes the (internal-linkage) helper
// entirely. -amdgpu-function-calls=true (the default) skips that forcing, so
// at -O0 -- which performs no heuristic inlining of its own -- the helper
// survives as a separate function.
//
// Both the COMPILE_SOURCE_TO_BC and CODEGEN_BC_TO_RELOCATABLE actions must
// request -O0: AMDGPUCompiler::addCompilationFlags() injects a default -O3
// into every action, and at -O3 the helper is fully inlined away during
// COMPILE_SOURCE_TO_BC itself (heuristic inlining, independent of
// -amdgpu-function-calls), which would erase it before
// CODEGEN_BC_TO_RELOCATABLE's -amdgpu-function-calls flag ever gets a chance
// to matter.
//
//===--------------------------------------------------------------------===//

#include "amd_comgr.h"
#include "common.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

static const char *Source =
    "#define CAT(a, b) a##b\n"
    "#define KERNEL_NAME(id) CAT(kernel_thread, id)\n"
    "#define HELPER_NAME(id) CAT(helper_thread, id)\n"
    "extern \"C\" {\n"
    "__attribute__((device)) static int HELPER_NAME(THREAD_ID)(int X) {\n"
    "  return X * X + 1;\n"
    "}\n"
    "__attribute__((global))\n"
    "void KERNEL_NAME(THREAD_ID)(int *A, int *B) {\n"
    "  *B = HELPER_NAME(THREAD_ID)(*A);\n"
    "}\n"
    "}\n";

static void compileOne(int Index, const char *OutputPrefix) {
  std::string SourceName = "source" + std::to_string(Index) + ".hip";
  std::string Define = "-DTHREAD_ID=" + std::to_string(Index);
  bool CallsEnabled = (Index % 2) == 0;
  std::string FunctionCallsFlag =
      CallsEnabled ? "-amdgpu-function-calls=true" : "-amdgpu-function-calls=false";

  amd_comgr_data_t DataSource;
  amd_comgr_data_set_t DataSetIn, DataSetBc, DataSetReloc, DataSetExec;
  amd_comgr_action_info_t DataAction;
  size_t Count;

  amd_comgr_(create_data_set(&DataSetIn));
  amd_comgr_(create_data(AMD_COMGR_DATA_KIND_SOURCE, &DataSource));
  amd_comgr_(set_data(DataSource, strlen(Source), Source));
  amd_comgr_(set_data_name(DataSource, SourceName.c_str()));
  amd_comgr_(data_set_add(DataSetIn, DataSource));

  const char *CompileOptions[] = {"-nogpuinc", "-O0", Define.c_str()};
  size_t CompileOptionsCount =
      sizeof(CompileOptions) / sizeof(CompileOptions[0]);

  amd_comgr_(create_action_info(&DataAction));
  amd_comgr_(action_info_set_language(DataAction, AMD_COMGR_LANGUAGE_HIP));
  amd_comgr_(
      action_info_set_isa_name(DataAction, "amdgcn-amd-amdhsa--gfx900"));
  amd_comgr_(action_info_set_option_list(DataAction, CompileOptions,
                                         CompileOptionsCount));

  amd_comgr_(create_data_set(&DataSetBc));
  amd_comgr_(do_action(AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC, DataAction,
                       DataSetIn, DataSetBc));
  amd_comgr_(action_data_count(DataSetBc, AMD_COMGR_DATA_KIND_BC, &Count));
  if (Count != 1) {
    fail("thread %d: COMPILE_SOURCE_TO_BC produced %zu BC objects "
         "(expected 1)",
         Index, Count);
  }

  const char *CodeGenOptions[] = {"-O0", "-mllvm", FunctionCallsFlag.c_str()};
  size_t CodeGenOptionsCount =
      sizeof(CodeGenOptions) / sizeof(CodeGenOptions[0]);
  amd_comgr_(action_info_set_option_list(DataAction, CodeGenOptions,
                                         CodeGenOptionsCount));

  amd_comgr_(create_data_set(&DataSetReloc));
  amd_comgr_(do_action(AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE, DataAction,
                       DataSetBc, DataSetReloc));
  amd_comgr_(action_data_count(DataSetReloc, AMD_COMGR_DATA_KIND_RELOCATABLE,
                               &Count));
  if (Count != 1) {
    fail("thread %d: CODEGEN_BC_TO_RELOCATABLE produced %zu objects "
         "(expected 1)",
         Index, Count);
  }

  amd_comgr_(create_data_set(&DataSetExec));
  amd_comgr_(action_info_set_option_list(DataAction, NULL, 0));
  amd_comgr_(do_action(AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE,
                       DataAction, DataSetReloc, DataSetExec));
  amd_comgr_(action_data_count(DataSetExec, AMD_COMGR_DATA_KIND_EXECUTABLE,
                               &Count));
  if (Count != 1) {
    fail("thread %d: LINK_RELOCATABLE_TO_EXECUTABLE produced %zu objects "
         "(expected 1)",
         Index, Count);
  }

  amd_comgr_data_t DataExec;
  amd_comgr_(action_data_get_data(DataSetExec, AMD_COMGR_DATA_KIND_EXECUTABLE,
                                  0, &DataExec));
  std::string OutFile =
      std::string(OutputPrefix) + std::to_string(Index) + ".bin";
  dumpData(DataExec, OutFile.c_str());

  amd_comgr_(release_data(DataSource));
  amd_comgr_(release_data(DataExec));
  amd_comgr_(destroy_data_set(DataSetIn));
  amd_comgr_(destroy_data_set(DataSetBc));
  amd_comgr_(destroy_data_set(DataSetReloc));
  amd_comgr_(destroy_data_set(DataSetExec));
  amd_comgr_(destroy_action_info(DataAction));
}

int main(int argc, char *argv[]) {
  if (argc != 3) {
    fprintf(stderr, "Usage: compile-concurrent-hip-differing-options "
                    "<num-threads> <output-prefix>\n");
    exit(1);
  }

  int NumThreads = atoi(argv[1]);
  const char *OutputPrefix = argv[2];

  std::vector<std::thread> CompileThreads;
  for (int I = 0; I < NumThreads; I++) {
    CompileThreads.push_back(std::thread(compileOne, I, OutputPrefix));
  }
  for (auto &Thread : CompileThreads) {
    Thread.join();
  }

  return 0;
}
