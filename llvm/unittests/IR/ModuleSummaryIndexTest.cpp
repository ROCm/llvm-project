//===- ModuleSummaryIndexTest.cpp - ModuleSummaryIndex Unit Tests-============//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/ModuleSummaryIndex.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

static std::unique_ptr<ModuleSummaryIndex> makeLLVMIndex(const char *Summary) {
  SMDiagnostic Err;
  std::unique_ptr<ModuleSummaryIndex> Index =
      parseSummaryIndexAssemblyString(Summary, Err);
  if (!Index)
    Err.print("ModuleSummaryIndexTest", errs());
  return Index;
}

TEST(ModuleSummaryIndexTest, MemProfSummaryPrinting) {
  std::unique_ptr<ModuleSummaryIndex> Index = makeLLVMIndex(R"Summary(
^0 = module: (path: "test.o", hash: (0, 0, 0, 0, 0))
^1 = gv: (guid: 23, summaries: (function: (module: ^0, flags: (linkage: external), insts: 2, allocs: ((versions: (none), memProf: ((type: notcold, stackIds: (1, 2, 3, 4)), (type: cold, stackIds: (1, 2, 3, 5))))))))
^2 = gv: (guid: 25, summaries: (function: (module: ^0, flags: (linkage: external), insts: 22, calls: ((callee: ^1)), callsites: ((callee: ^1, clones: (0), stackIds: (3, 4)), (callee: ^1, clones: (0), stackIds: (3, 5))))))
)Summary");

  std::string Data;
  raw_string_ostream OS(Data);

  ASSERT_NE(Index, nullptr);
  auto *CallsiteSummary =
      cast<FunctionSummary>(Index->getGlobalValueSummary(/*guid=*/25));
  for (auto &CI : CallsiteSummary->callsites())
    OS << "\n" << CI;

  auto *AllocSummary =
      cast<FunctionSummary>(Index->getGlobalValueSummary(/*guid=*/23));
  for (auto &AI : AllocSummary->allocs())
    OS << "\n" << AI;

  EXPECT_EQ(Data, R"(
Callee: 23 Clones: 0 StackIds: 2, 3
Callee: 23 Clones: 0 StackIds: 2, 4
Versions: 0 MIB:
		AllocType 1 StackIds: 0, 1, 2, 3
		AllocType 2 StackIds: 0, 1, 2, 4
)");
}

// The variable with GUID 5 references the functions 1, 2, 3, 9, and the alias
// 6 of the function 7. The functions 1, 2, 4, 7, and 9 have the call type
// identifier 100, and 3 has 200. No summary references 4. The function 9 is
// dead. The function 1 has a copy in a second module. The function 8 has
// indirect calls with the identifiers 100 and 300.
static const char *IndirectCallSummary = R"Summary(
^0 = module: (path: "test.o", hash: (0, 0, 0, 0, 0))
^1 = module: (path: "test2.o", hash: (0, 0, 0, 0, 0))
^2 = gv: (guid: 1, summaries: (function: (module: ^0, flags: (linkage: linkonce_odr, live: 1), insts: 1, callTypeId: 100), function: (module: ^1, flags: (linkage: linkonce_odr, live: 1), insts: 1, callTypeId: 100)))
^3 = gv: (guid: 2, summaries: (function: (module: ^0, flags: (linkage: external, live: 1), insts: 1, callTypeId: 100)))
^4 = gv: (guid: 3, summaries: (function: (module: ^0, flags: (linkage: external, live: 1), insts: 1, callTypeId: 200)))
^5 = gv: (guid: 4, summaries: (function: (module: ^0, flags: (linkage: external, live: 1), insts: 1, callTypeId: 100)))
^6 = gv: (guid: 5, summaries: (variable: (module: ^0, flags: (linkage: external, live: 1), varFlags: (readonly: 0), refs: (^2, ^3, ^4, ^7, ^10))))
^7 = gv: (guid: 6, summaries: (alias: (module: ^0, flags: (linkage: external, live: 1), aliasee: ^8)))
^8 = gv: (guid: 7, summaries: (function: (module: ^0, flags: (linkage: external, live: 1), insts: 1, callTypeId: 100)))
^9 = gv: (guid: 8, summaries: (function: (module: ^0, flags: (linkage: external, live: 1), insts: 2, indirectCallTypeIds: (100, 300))))
^10 = gv: (guid: 9, summaries: (function: (module: ^0, flags: (linkage: external, live: 0), insts: 1, callTypeId: 100)))
^11 = flags: 1
)Summary";

static std::vector<GlobalValue::GUID>
getIndirectCallTargets(const FunctionSummary &FS) {
  std::vector<GlobalValue::GUID> GUIDs;
  for (const FunctionSummary::EdgeTy &Edge : FS.indirect_calls()) {
    EXPECT_TRUE(Edge.second.HasIndirectCall);
    GUIDs.push_back(Edge.first.getGUID());
  }
  return GUIDs;
}

TEST(ModuleSummaryIndexTest, ResolveIndirectCalls) {
  std::unique_ptr<ModuleSummaryIndex> Index =
      makeLLVMIndex(IndirectCallSummary);
  ASSERT_NE(Index, nullptr);
  auto *Caller = cast<FunctionSummary>(Index->getGlobalValueSummary(8));

  EXPECT_FALSE(Index->hasResolvedIndirectCalls());
  EXPECT_TRUE(Caller->indirect_calls().empty());

  // The targets are the live referenced functions with the identifier 100, in
  // the order of the index, without duplicates. For the alias, the target is
  // the aliasee.
  Index->resolveIndirectCalls(/*GUIDPreservedSymbols=*/{});
  EXPECT_TRUE(Index->hasResolvedIndirectCalls());
  EXPECT_EQ(getIndirectCallTargets(*Caller),
            std::vector<GlobalValue::GUID>({1, 2, 7}));

  auto It = Caller->indirect_calls().begin();
  EXPECT_EQ(It->first.getGUID(), 1u);
  It++;
  EXPECT_EQ((*It).first.getGUID(), 2u);

  // A function without indirect calls has no targets.
  auto *Target = cast<FunctionSummary>(Index->getGlobalValueSummary(2));
  EXPECT_TRUE(Target->indirect_calls().empty());

  // A second call does nothing.
  Index->resolveIndirectCalls(/*GUIDPreservedSymbols=*/{4});
  EXPECT_EQ(getIndirectCallTargets(*Caller),
            std::vector<GlobalValue::GUID>({1, 2, 7}));
}

TEST(ModuleSummaryIndexTest, ResolveIndirectCallsPreservedSymbols) {
  std::unique_ptr<ModuleSummaryIndex> Index =
      makeLLVMIndex(IndirectCallSummary);
  ASSERT_NE(Index, nullptr);
  auto *Caller = cast<FunctionSummary>(Index->getGlobalValueSummary(8));

  // A function that is visible outside the summaries is a target, also if no
  // summary references it.
  Index->resolveIndirectCalls(/*GUIDPreservedSymbols=*/{4});
  EXPECT_EQ(getIndirectCallTargets(*Caller),
            std::vector<GlobalValue::GUID>({1, 2, 4, 7}));
}

} // end anonymous namespace
