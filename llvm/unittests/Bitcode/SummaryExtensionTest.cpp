//===- SummaryExtensionTest.cpp - Summary extensions in bitcode -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/ModuleSummaryAnalysis.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ModuleSummaryIndex.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/AssignGUID.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

using Ints = SmallVector<uint64_t, 4>;

static const GlobalValue::GUID F =
    GlobalValue::getGUIDAssumingExternalLinkage("f");
static const GlobalValue::GUID G =
    GlobalValue::getGUIDAssumingExternalLinkage("g");

// Both modules have a copy of @f, like a template function in two translation
// units.
static const char *ModuleA = R"(
define linkonce_odr void @f() {
  ret void
}

define void @g() {
  call void @f()
  ret void
}
)";

static const char *ModuleB = R"(
define linkonce_odr void @f() {
  ret void
}
)";

// Writes a module with a summary. AddEntries adds the entries of the summary
// extensions.
static SmallString<0>
writeModule(StringRef Text,
            function_ref<void(ModuleSummaryIndex &)> AddEntries) {
  LLVMContext Ctx;
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseAssemblyString(Text, Err, Ctx);
  if (!M) {
    Err.print("SummaryExtensionTest", errs());
    return {};
  }
  AssignGUIDPass::runOnModule(*M);
  ProfileSummaryInfo PSI(*M);
  ModuleSummaryIndex Index = buildModuleSummaryIndex(*M, nullptr, &PSI);
  AddEntries(Index);
  SmallString<0> Buffer;
  raw_svector_ostream OS(Buffer);
  WriteBitcodeToFile(*M, OS, /*ShouldPreserveUseListOrder=*/false, &Index);
  return Buffer;
}

// Sets an entry for the summary of the value with GUID in the per-module
// Index.
static void addEntry(ModuleSummaryIndex &Index, GlobalValue::GUID GUID,
                     uint64_t Int, uint64_t Version = 1) {
  const GlobalValueSummary *S =
      Index.getValueInfo(GUID).getSummaryList().front().get();
  Index.getOrInsertExtension("test", Version, /*Required=*/false)
      .setEntry(S, {{Int}, {"s"}});
}

// Returns the integers of the entry for the summary of the value with GUID in
// Module, or nothing if there is no such entry.
static std::optional<Ints> getInts(const ModuleSummaryIndex &Index,
                                   GlobalValue::GUID GUID, StringRef Module,
                                   uint64_t Version = 1) {
  const SummaryExtension *Ext = Index.getExtension("test", Version);
  const GlobalValueSummary *S = Index.findSummaryInModule(GUID, Module);
  if (!Ext || !S || !Ext->getEntry(S))
    return std::nullopt;
  return Ext->getEntry(S)->Ints;
}

// Reads the summary of Buffer into Index, with the module path Path.
static Error readInto(ModuleSummaryIndex &Index, StringRef Buffer,
                      StringRef Path,
                      std::function<bool(StringRef)> IsPrevailing = nullptr) {
  Expected<std::vector<BitcodeModule>> Modules =
      getBitcodeModuleList(MemoryBufferRef(Buffer, Path));
  if (!Modules)
    return Modules.takeError();
  return (*Modules)[0].readSummary(Index, Path, std::move(IsPrevailing));
}

TEST(SummaryExtensionTest, PerModuleRoundTrip) {
  SmallString<0> Buffer = writeModule(ModuleA, [](ModuleSummaryIndex &Index) {
    SummaryExtension &Ext =
        Index.getOrInsertExtension("test", 3, /*Required=*/true);
    Ext.setEntry(Index.getValueInfo(F).getSummaryList().front().get(),
                 {{1, 2}, {"a", "b"}});
    Ext.setEntry(Index.getValueInfo(G).getSummaryList().front().get(),
                 {{}, {"c"}});
  });
  ASSERT_FALSE(Buffer.empty());

  Expected<std::unique_ptr<ModuleSummaryIndex>> Index =
      getModuleSummaryIndex(MemoryBufferRef(Buffer, "a.bc"));
  ASSERT_TRUE(!!Index) << toString(Index.takeError());
  const SummaryExtension *Ext = (*Index)->getExtension("test", 3);
  ASSERT_NE(Ext, nullptr);
  EXPECT_TRUE(Ext->isRequired());
  EXPECT_EQ(Ext->size(), 2u);
  const SummaryExtension::Entry *FEntry =
      Ext->getEntry((*Index)->findSummaryInModule(F, "a.bc"));
  ASSERT_NE(FEntry, nullptr);
  EXPECT_EQ(FEntry->Ints, (Ints{1, 2}));
  EXPECT_EQ(FEntry->Strings, (SmallVector<std::string, 0>{"a", "b"}));
  const SummaryExtension::Entry *GEntry =
      Ext->getEntry((*Index)->findSummaryInModule(G, "a.bc"));
  ASSERT_NE(GEntry, nullptr);
  EXPECT_TRUE(GEntry->Ints.empty());
  EXPECT_EQ(GEntry->Strings, (SmallVector<std::string, 0>{"c"}));
}

// The copies of a function in two modules can have different entries. Each
// entry stays with its copy, also if the reader does not know which copy
// prevails.
TEST(SummaryExtensionTest, TwoCopies) {
  SmallString<0> A = writeModule(
      ModuleA, [](ModuleSummaryIndex &Index) { addEntry(Index, F, 1); });
  SmallString<0> B = writeModule(
      ModuleB, [](ModuleSummaryIndex &Index) { addEntry(Index, F, 2); });

  for (bool KnowsPrevailing : {false, true}) {
    std::function<bool(StringRef)> IsPrevailingA, IsPrevailingB;
    if (KnowsPrevailing) {
      // The copy in b.bc prevails.
      IsPrevailingA = [](StringRef Name) { return Name != "f"; };
      IsPrevailingB = [](StringRef) { return true; };
    }
    ModuleSummaryIndex Index(/*HaveGVs=*/false);
    ASSERT_FALSE(errorToBool(readInto(Index, A, "a.bc", IsPrevailingA)));
    ASSERT_FALSE(errorToBool(readInto(Index, B, "b.bc", IsPrevailingB)));
    EXPECT_EQ(getInts(Index, F, "a.bc"), Ints{1});
    EXPECT_EQ(getInts(Index, F, "b.bc"), Ints{2});
  }
}

// Modules from different compilers can hold different versions of an
// extension.
TEST(SummaryExtensionTest, TwoVersions) {
  SmallString<0> A = writeModule(ModuleA, [](ModuleSummaryIndex &Index) {
    addEntry(Index, F, 1, /*Version=*/1);
  });
  SmallString<0> B = writeModule(ModuleB, [](ModuleSummaryIndex &Index) {
    addEntry(Index, F, 2, /*Version=*/2);
  });

  ModuleSummaryIndex Index(/*HaveGVs=*/false);
  ASSERT_FALSE(errorToBool(readInto(Index, A, "a.bc")));
  ASSERT_FALSE(errorToBool(readInto(Index, B, "b.bc")));
  EXPECT_EQ(getInts(Index, F, "a.bc", /*Version=*/1), Ints{1});
  EXPECT_EQ(getInts(Index, F, "b.bc", /*Version=*/1), std::nullopt);
  EXPECT_EQ(getInts(Index, F, "b.bc", /*Version=*/2), Ints{2});
}

// The index of a distributed backend gets only the entries of the summaries
// that it holds, each with its module.
TEST(SummaryExtensionTest, PerBackendIndex) {
  SmallString<0> A = writeModule(ModuleA, [](ModuleSummaryIndex &Index) {
    addEntry(Index, F, 1);
    addEntry(Index, G, 2);
  });
  SmallString<0> B = writeModule(
      ModuleB, [](ModuleSummaryIndex &Index) { addEntry(Index, F, 3); });
  ModuleSummaryIndex Index(/*HaveGVs=*/false);
  ASSERT_FALSE(errorToBool(readInto(Index, A, "a.bc")));
  ASSERT_FALSE(errorToBool(readInto(Index, B, "b.bc")));

  // The backend of a.bc gets the summary of @g, and the summary of @f in b.bc.
  ModuleToSummariesForIndexTy ModuleToSummaries;
  ModuleToSummaries["a.bc"][G] = Index.findSummaryInModule(G, "a.bc");
  ModuleToSummaries["b.bc"][F] = Index.findSummaryInModule(F, "b.bc");
  SmallString<0> Buffer;
  raw_svector_ostream OS(Buffer);
  writeIndexToFile(Index, OS, &ModuleToSummaries);

  Expected<std::unique_ptr<ModuleSummaryIndex>> Backend =
      getModuleSummaryIndex(MemoryBufferRef(Buffer, "a.thinlto.bc"));
  ASSERT_TRUE(!!Backend) << toString(Backend.takeError());
  const SummaryExtension *Ext = (*Backend)->getExtension("test", 1);
  ASSERT_NE(Ext, nullptr);
  EXPECT_EQ(Ext->size(), 2u);
  EXPECT_EQ(getInts(**Backend, G, "a.bc"), Ints{2});
  EXPECT_EQ(getInts(**Backend, F, "b.bc"), Ints{3});
  EXPECT_EQ(getInts(**Backend, F, "a.bc"), std::nullopt);
}

// LTO reads the summaries of all regular LTO modules with the module path "".
// Each entry stays with the copy of its module. A combined index does not get
// the entries of these summaries, because the value and the module do not
// identify one of them.
TEST(SummaryExtensionTest, SameModulePath) {
  SmallString<0> A = writeModule(
      ModuleA, [](ModuleSummaryIndex &Index) { addEntry(Index, F, 1); });
  SmallString<0> B = writeModule(
      ModuleB, [](ModuleSummaryIndex &Index) { addEntry(Index, F, 2); });
  ModuleSummaryIndex Index(/*HaveGVs=*/false);
  ASSERT_FALSE(errorToBool(readInto(Index, A, "")));
  ASSERT_FALSE(errorToBool(readInto(Index, B, "")));

  const SummaryExtension *Ext = Index.getExtension("test", 1);
  ASSERT_NE(Ext, nullptr);
  ArrayRef<std::unique_ptr<GlobalValueSummary>> Copies =
      Index.getValueInfo(F).getSummaryList();
  ASSERT_EQ(Copies.size(), 2u);
  ASSERT_NE(Ext->getEntry(Copies[0].get()), nullptr);
  EXPECT_EQ(Ext->getEntry(Copies[0].get())->Ints, Ints{1});
  ASSERT_NE(Ext->getEntry(Copies[1].get()), nullptr);
  EXPECT_EQ(Ext->getEntry(Copies[1].get())->Ints, Ints{2});

  SmallString<0> Buffer;
  raw_svector_ostream OS(Buffer);
  writeIndexToFile(Index, OS);
  Expected<std::unique_ptr<ModuleSummaryIndex>> Combined =
      getModuleSummaryIndex(MemoryBufferRef(Buffer, "index.bc"));
  ASSERT_TRUE(!!Combined) << toString(Combined.takeError());
  EXPECT_EQ((*Combined)->getExtension("test", 1), nullptr);
}

} // end anonymous namespace
