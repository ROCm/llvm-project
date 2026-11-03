//===- ThinLinkFnAttrsTest.cpp - Function attributes from the thin link ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/ThinLinkFnAttrs.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

static std::unique_ptr<ModuleSummaryIndex> parseIndex(StringRef Summary) {
  SMDiagnostic Err;
  std::unique_ptr<ModuleSummaryIndex> Index =
      parseSummaryIndexAssemblyString(Summary, Err);
  if (!Index)
    Err.print("ThinLinkFnAttrsTest", errs());
  return Index;
}

TEST(ThinLinkFnAttrsTest, Apply) {
  LLVMContext Ctx;
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseAssemblyString(R"(
define void @f() #0 {
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="1,1024" }
)",
                                                  Err, Ctx);
  ASSERT_NE(M, nullptr);

  ThinLinkFnAttrs Attrs;
  Attrs.add(Attribute::NoUnwind);
  Attrs.add("amdgpu-flat-work-group-size", "64,256");
  Attrs.add("amdgpu-no-workitem-id-x");
  Function *F = M->getFunction("f");
  Attrs.apply(*F);
  EXPECT_TRUE(F->hasFnAttribute(Attribute::NoUnwind));
  EXPECT_EQ(F->getFnAttribute("amdgpu-flat-work-group-size").getValueAsString(),
            "64,256");
  EXPECT_TRUE(F->hasFnAttribute("amdgpu-no-workitem-id-x"));
}

static const char *TwoCopies = R"(
^0 = module: (path: "a.o", hash: (0, 0, 0, 0, 0))
^1 = module: (path: "b.o", hash: (0, 0, 0, 0, 1))
^2 = gv: (guid: 1, summaries: (function: (module: ^0, flags: (linkage: linkonce_odr), insts: 1), function: (module: ^1, flags: (linkage: linkonce_odr), insts: 1)))
)";

// A thin-link pass stores the attributes on each copy of a function, in a
// required extension. The result of a second pass adds to them, and its string
// values replace the values there.
TEST(ThinLinkFnAttrsTest, Summaries) {
  std::unique_ptr<ModuleSummaryIndex> Index = parseIndex(TwoCopies);
  ASSERT_NE(Index, nullptr);

  ThinLinkFnAttrs First;
  First.add(Attribute::NoUnwind);
  First.add("key", "1");
  ThinLinkFnAttrs Second;
  Second.add(Attribute::NoRecurse);
  Second.add("key", "2");
  Second.add("other");
  ThinLinkFnAttrs::addToSummaries(*Index, Index->getValueInfo(1), First);
  ThinLinkFnAttrs::addToSummaries(*Index, Index->getValueInfo(1), Second);

  ThinLinkFnAttrs Want;
  Want.add(Attribute::NoUnwind);
  Want.add(Attribute::NoRecurse);
  Want.add("key", "2");
  Want.add("other");
  for (const auto &S : Index->getValueInfo(1).getSummaryList()) {
    Expected<ThinLinkFnAttrs> Attrs = ThinLinkFnAttrs::get(*Index, *S);
    ASSERT_THAT_EXPECTED(Attrs, Succeeded());
    EXPECT_TRUE(*Attrs == Want);
  }
  const SummaryExtension *Ext = Index->getExtension(
      ThinLinkFnAttrs::ExtensionName, ThinLinkFnAttrs::ExtensionVersion);
  ASSERT_NE(Ext, nullptr);
  EXPECT_TRUE(Ext->isRequired());
}

// An entry holds the number of enum attributes, their names, and a key and a
// value for each string attribute.
TEST(ThinLinkFnAttrsTest, EntryFormat) {
  std::unique_ptr<ModuleSummaryIndex> Index = parseIndex(
      std::string(TwoCopies) +
      R"(^3 = extension: (name: "fn-attrs", version: 1, required: 1, entries: ((gv: ^2, module: ^0, ints: (1), strings: ("nounwind", "key", "value")))))");
  ASSERT_NE(Index, nullptr);

  ThinLinkFnAttrs Want;
  Want.add(Attribute::NoUnwind);
  Want.add("key", "value");
  const GlobalValueSummary *S = Index->findSummaryInModule(1, "a.o");
  ASSERT_NE(S, nullptr);
  Expected<ThinLinkFnAttrs> Attrs = ThinLinkFnAttrs::get(*Index, *S);
  ASSERT_THAT_EXPECTED(Attrs, Succeeded());
  EXPECT_TRUE(*Attrs == Want);

  // The copy in b.o has no entry.
  S = Index->findSummaryInModule(1, "b.o");
  ASSERT_NE(S, nullptr);
  Attrs = ThinLinkFnAttrs::get(*Index, *S);
  ASSERT_THAT_EXPECTED(Attrs, Succeeded());
  EXPECT_TRUE(Attrs->empty());
}

TEST(ThinLinkFnAttrsTest, InvalidEntries) {
  auto GetError = [](StringRef Entry) -> std::string {
    std::unique_ptr<ModuleSummaryIndex> Index =
        parseIndex(std::string(TwoCopies) +
                   "^3 = extension: (name: \"fn-attrs\", version: 1, "
                   "required: 1, entries: ((gv: ^2, module: ^0, " +
                   Entry.str() + ")))");
    if (!Index)
      return "cannot parse";
    Expected<ThinLinkFnAttrs> Attrs =
        ThinLinkFnAttrs::get(*Index, *Index->findSummaryInModule(1, "a.o"));
    return Attrs ? "no error" : toString(Attrs.takeError());
  };
  EXPECT_EQ(GetError("ints: (1), strings: (\"no-such-attribute\")"),
            "unknown attribute 'no-such-attribute' in the summary extension "
            "'fn-attrs'");
  // A string attribute is not an enum attribute.
  EXPECT_EQ(GetError("ints: (1), strings: (\"amdgpu-no-workitem-id-x\")"),
            "unknown attribute 'amdgpu-no-workitem-id-x' in the summary "
            "extension 'fn-attrs'");
  // A parameter attribute is not a function attribute.
  EXPECT_EQ(GetError("ints: (1), strings: (\"noundef\")"),
            "attribute 'noundef' in the summary extension 'fn-attrs' is not a "
            "function attribute");
  EXPECT_EQ(GetError("ints: (0), strings: (\"key\")"),
            "invalid entry in the summary extension 'fn-attrs'");
  EXPECT_EQ(GetError("strings: (\"key\", \"value\")"),
            "invalid entry in the summary extension 'fn-attrs'");
}

} // end anonymous namespace
