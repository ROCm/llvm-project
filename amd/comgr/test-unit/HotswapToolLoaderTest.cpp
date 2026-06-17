//===- HotswapToolLoaderTest.cpp - Unit tests for the HSA tool ABI --------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Drives the tool's HSA_TOOLS_LIB entry points against a stub HSA API table:
// OnLoad validates the table and installs the reader-create wrapper, OnUnload
// restores it. No GPU is involved; the rewrite path is covered on real silicon.

#include "inc/hsa.h"
#include "inc/hsa_api_trace.h"
#include "gtest/gtest.h"

#include <cstddef>
#include <cstdint>

// Entry points exported by the tool; names are fixed by the HSA tool ABI.
// NOLINTBEGIN(readability-identifier-naming)
extern "C" bool OnLoad(void *Table, uint64_t RuntimeVersion,
                       uint64_t FailedToolCount,
                       const char *const *FailedToolNames);
extern "C" void OnUnload();
// NOLINTEND(readability-identifier-naming)

using ReaderCreateFn = decltype(hsa_code_object_reader_create_from_memory) *;

static hsa_status_t stubReaderCreate(const void *, size_t,
                                     hsa_code_object_reader_t *) {
  return HSA_STATUS_SUCCESS;
}
static hsa_status_t stubIterateAgents(hsa_status_t (*)(hsa_agent_t, void *),
                                      void *) {
  return HSA_STATUS_SUCCESS;
}
static hsa_status_t stubAgentGetInfo(hsa_agent_t, hsa_agent_info_t, void *) {
  return HSA_STATUS_SUCCESS;
}
static hsa_status_t stubIsaGetInfoAlt(hsa_isa_t, hsa_isa_info_t, void *) {
  return HSA_STATUS_SUCCESS;
}

// A CoreApiTable wired with the function pointers OnLoad requires.
static CoreApiTable makeStubCore() {
  CoreApiTable Core{};
  Core.hsa_iterate_agents_fn = stubIterateAgents;
  Core.hsa_agent_get_info_fn = stubAgentGetInfo;
  Core.hsa_isa_get_info_alt_fn = stubIsaGetInfoAlt;
  Core.hsa_code_object_reader_create_from_memory_fn = stubReaderCreate;
  return Core;
}

TEST(HotswapToolLoader, OnLoadRejectsInvalidTables) {
  EXPECT_FALSE(OnLoad(nullptr, 0, 0, nullptr));

  HsaApiTable NoCore{};
  NoCore.core_ = nullptr;
  EXPECT_FALSE(OnLoad(&NoCore, 0, 0, nullptr));

  CoreApiTable Core = makeStubCore();
  Core.hsa_code_object_reader_create_from_memory_fn = nullptr;
  HsaApiTable Table{};
  Table.core_ = &Core;
  EXPECT_FALSE(OnLoad(&Table, 0, 0, nullptr));
}

TEST(HotswapToolLoader, OnLoadInstallsWrapperAndIsIdempotent) {
  CoreApiTable Core = makeStubCore();
  HsaApiTable Table{};
  Table.core_ = &Core;
  ReaderCreateFn &Entry = Core.hsa_code_object_reader_create_from_memory_fn;

  ASSERT_TRUE(OnLoad(&Table, 0, 0, nullptr));
  ReaderCreateFn Installed = Entry;
  EXPECT_NE(Installed, stubReaderCreate);

  ASSERT_TRUE(OnLoad(&Table, 0, 0, nullptr)); // already installed -> no-op
  EXPECT_EQ(Entry, Installed);
}

TEST(HotswapToolLoader, OnUnloadRestoresOriginalEntry) {
  CoreApiTable Core = makeStubCore();
  HsaApiTable Table{};
  Table.core_ = &Core;
  ReaderCreateFn &Entry = Core.hsa_code_object_reader_create_from_memory_fn;

  ASSERT_TRUE(OnLoad(&Table, 0, 0, nullptr));
  ASSERT_NE(Entry, stubReaderCreate);
  OnUnload();
  EXPECT_EQ(Entry, stubReaderCreate);
}
