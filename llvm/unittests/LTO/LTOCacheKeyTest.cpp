//===- LTOCacheKeyTest.cpp - Unit tests for the ThinLTO cache key ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/ModuleSummaryIndex.h"
#include "llvm/LTO/Config.h"
#include "llvm/LTO/LTO.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

// Returns a combined index in summary assembly. The function with GUID 1 in
// a.o calls the function with GUID 2 in b.o. FunctionFields are extra fields
// of the function in a.o.
static std::string makeSummary(StringRef FunctionFields) {
  return (R"(^0 = module: (path: "a.o", hash: (1, 2, 3, 4, 5))
^1 = module: (path: "b.o", hash: (6, 7, 8, 9, 10))
^2 = gv: (guid: 1, summaries: (function: (module: ^0, flags: (linkage: external), insts: 2, calls: ((callee: ^3)))" +
          FunctionFields + R"()))
^3 = gv: (guid: 2, summaries: (function: (module: ^1, flags: (linkage: external), insts: 1)))
)")
      .str();
}

// Returns the cache key of a.o for the combined index in Summary.
static std::string getCacheKey(StringRef Summary) {
  SMDiagnostic Err;
  std::unique_ptr<ModuleSummaryIndex> Index =
      parseSummaryIndexAssemblyString(Summary, Err);
  if (!Index) {
    Err.print("LTOCacheKeyTest", errs());
    return "";
  }
  DenseMap<StringRef, GVSummaryMapTy> DefinedGlobals;
  Index->collectDefinedGVSummariesPerModule(DefinedGlobals);
  lto::Config Conf;
  FunctionImporter::ImportIDTable ImportIDs;
  FunctionImporter::ImportMapTy ImportList(ImportIDs);
  return computeLTOCacheKey(Conf, *Index, "a.o", ImportList,
                            /*ExportList=*/{}, /*ResolvedODR=*/{},
                            DefinedGlobals["a.o"]);
}

TEST(LTOCacheKeyTest, SameSummarySameKey) {
  std::string Summary = makeSummary(", funcFlags: (noUnwind: 1)");
  std::string Key = getCacheKey(Summary);
  ASSERT_FALSE(Key.empty());
  EXPECT_EQ(Key, getCacheKey(Summary));
}

// The thin link changes the function flags (function attribute propagation),
// so they must be part of the key.
TEST(LTOCacheKeyTest, FunctionFlags) {
  std::string Key1 = getCacheKey(makeSummary(", funcFlags: (noUnwind: 0)"));
  std::string Key2 = getCacheKey(makeSummary(", funcFlags: (noUnwind: 1)"));
  ASSERT_FALSE(Key1.empty());
  ASSERT_FALSE(Key2.empty());
  EXPECT_NE(Key1, Key2);
}

// The thin link selects the MemProf allocation versions.
TEST(LTOCacheKeyTest, AllocationVersions) {
  std::string Key1 = getCacheKey(
      makeSummary(", allocs: ((versions: (notcold), memProf: ((type: notcold, "
                  "stackIds: (1)))))"));
  std::string Key2 = getCacheKey(
      makeSummary(", allocs: ((versions: (cold), memProf: ((type: notcold, "
                  "stackIds: (1)))))"));
  ASSERT_FALSE(Key1.empty());
  ASSERT_FALSE(Key2.empty());
  EXPECT_NE(Key1, Key2);
}

// The thin link selects the MemProf callsite clones.
TEST(LTOCacheKeyTest, CallsiteClones) {
  std::string Key1 = getCacheKey(
      makeSummary(", callsites: ((callee: ^3, clones: (0), stackIds: (2)))"));
  std::string Key2 = getCacheKey(
      makeSummary(", callsites: ((callee: ^3, clones: (1), stackIds: (2)))"));
  ASSERT_FALSE(Key1.empty());
  ASSERT_FALSE(Key2.empty());
  EXPECT_NE(Key1, Key2);
}

// The thin link computes the parameter access ranges (stack safety).
TEST(LTOCacheKeyTest, ParameterAccesses) {
  std::string Key1 =
      getCacheKey(makeSummary(", params: ((param: 0, offset: [0, 7]))"));
  std::string Key2 =
      getCacheKey(makeSummary(", params: ((param: 0, offset: [0, 3]))"));
  ASSERT_FALSE(Key1.empty());
  ASSERT_FALSE(Key2.empty());
  EXPECT_NE(Key1, Key2);
}

} // end anonymous namespace
