//===- LTOBackendTest.cpp - Unit tests for the ThinLTO backend ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/LTO/LTOBackend.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ThinLinkFnAttrs.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Testing/Support/Error.h"
#include "llvm/Transforms/IPO/FunctionImport.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

static std::unique_ptr<ModuleSummaryIndex> parseIndex(StringRef Summary) {
  SMDiagnostic Err;
  std::unique_ptr<ModuleSummaryIndex> Index =
      parseSummaryIndexAssemblyString(Summary, Err);
  if (!Index)
    Err.print("LTOBackendTest", errs());
  return Index;
}

// The module a.o defines @f and @h, and imports @g from b.o. The thin link set
// attributes on @f in a.o, and on both copies of @g.
static const char *Summary = R"(
^0 = module: (path: "a.o", hash: (0, 0, 0, 0, 0))
^1 = module: (path: "b.o", hash: (0, 0, 0, 0, 1))
^2 = module: (path: "c.o", hash: (0, 0, 0, 0, 2))
^3 = gv: (name: "f", summaries: (function: (module: ^0, flags: (linkage: external), insts: 1)))
^4 = gv: (name: "g", summaries: (function: (module: ^1, flags: (linkage: linkonce_odr), insts: 1), function: (module: ^2, flags: (linkage: linkonce_odr), insts: 1)))
^5 = gv: (name: "h", summaries: (function: (module: ^0, flags: (linkage: external), insts: 1)))
^6 = extension: (name: "fn-attrs", version: 1, required: 1, entries: ((gv: ^3, module: ^0, ints: (0), strings: ("key", "f")), (gv: ^4, module: ^1, ints: (1), strings: ("nounwind", "key", "g.b")), (gv: ^4, module: ^2, ints: (0), strings: ("key", "g.c"))))
)";

static const char *ModuleA = R"(
define void @f() {
  ret void
}

define available_externally void @g() {
  ret void
}

define void @h() {
  ret void
}
)";

// A function gets the attributes of its summary in the module that it comes
// from.
TEST(LTOBackendTest, ApplyThinLinkFnAttrs) {
  std::unique_ptr<ModuleSummaryIndex> Index = parseIndex(Summary);
  ASSERT_NE(Index, nullptr);
  LLVMContext Ctx;
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseAssemblyString(ModuleA, Err, Ctx);
  ASSERT_NE(M, nullptr);

  DenseMap<StringRef, GVSummaryMapTy> DefinedGlobals;
  Index->collectDefinedGVSummariesPerModule(DefinedGlobals);
  FunctionImporter::ImportIDTable ImportIDs;
  FunctionImporter::ImportMapTy ImportList(ImportIDs);
  ImportList.addDefinition("b.o",
                           GlobalValue::getGUIDAssumingExternalLinkage("g"));
  ASSERT_THAT_ERROR(
      lto::applyThinLinkFnAttrs(*M, *Index, ImportList, DefinedGlobals["a.o"]),
      Succeeded());

  Function *F = M->getFunction("f"), *G = M->getFunction("g"),
           *H = M->getFunction("h");
  EXPECT_EQ(F->getFnAttribute("key").getValueAsString(), "f");
  EXPECT_FALSE(F->hasFnAttribute(Attribute::NoUnwind));
  EXPECT_EQ(G->getFnAttribute("key").getValueAsString(), "g.b");
  EXPECT_TRUE(G->hasFnAttribute(Attribute::NoUnwind));
  EXPECT_FALSE(H->hasFnAttribute("key"));
}

// A backend reports an error for a required extension that it does not apply,
// if the extension has an entry for a summary that the backend uses. The
// backend of a.o uses @f, which a.o defines, and @g, which it imports from b.o.
// It does not use @h in b.o.
TEST(LTOBackendTest, RequiredSummaryExtensions) {
  auto Check = [](StringRef Name, uint64_t Version, bool Required,
                  StringRef Entries) {
    std::unique_ptr<ModuleSummaryIndex> Index =
        parseIndex((Twine(R"(
^0 = module: (path: "a.o", hash: (0, 0, 0, 0, 0))
^1 = module: (path: "b.o", hash: (0, 0, 0, 0, 1))
^2 = gv: (name: "f", summaries: (function: (module: ^0, flags: (linkage: external), insts: 1)))
^3 = gv: (name: "g", summaries: (function: (module: ^1, flags: (linkage: external), insts: 1)))
^4 = gv: (name: "h", summaries: (function: (module: ^1, flags: (linkage: external), insts: 1)))
^5 = extension: (name: ")") +
                    Name + "\", version: " + Twine(Version) +
                    ", required: " + Twine(Required ? 1 : 0) + Entries + ")")
                       .str());
    if (!Index)
      return std::string("cannot parse");
    DenseMap<StringRef, GVSummaryMapTy> DefinedGlobals;
    Index->collectDefinedGVSummariesPerModule(DefinedGlobals);
    FunctionImporter::ImportIDTable ImportIDs;
    FunctionImporter::ImportMapTy ImportList(ImportIDs);
    ImportList.addDefinition("b.o",
                             GlobalValue::getGUIDAssumingExternalLinkage("g"));
    Error Err = lto::checkRequiredSummaryExtensions(*Index, ImportList,
                                                    DefinedGlobals["a.o"]);
    return Err ? toString(std::move(Err)) : std::string("no error");
  };
  StringRef OnF = ", entries: ((gv: ^2, module: ^0, ints: (0)))";
  StringRef OnG = ", entries: ((gv: ^3, module: ^1, ints: (0)))";
  StringRef OnH = ", entries: ((gv: ^4, module: ^1, ints: (0)))";
  EXPECT_EQ(Check("fn-attrs", 1, /*Required=*/true, OnF), "no error");
  EXPECT_EQ(Check("other", 1, /*Required=*/false, OnF), "no error");
  EXPECT_EQ(Check("other", 1, /*Required=*/true, OnF),
            "the summary index needs the summary extension 'other' version 1, "
            "which this compiler does not apply");
  EXPECT_EQ(Check("fn-attrs", 2, /*Required=*/true, OnF),
            "the summary index needs the summary extension 'fn-attrs' version "
            "2, which this compiler does not apply");
  EXPECT_EQ(Check("other", 1, /*Required=*/true, OnG),
            "the summary index needs the summary extension 'other' version 1, "
            "which this compiler does not apply");
  EXPECT_EQ(Check("other", 1, /*Required=*/true, OnH), "no error");
  EXPECT_EQ(Check("other", 1, /*Required=*/true, ""), "no error");
}

} // end anonymous namespace
