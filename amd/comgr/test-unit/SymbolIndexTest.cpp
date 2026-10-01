//===- SymbolIndexTest.cpp - Symbol lookup cache tests --------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "comgr.h"

#include "common.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <fstream>
#include <set>
#include <string>
#include <vector>

using COMGR::DataObject;

namespace {

// Defined in the checked-in test code object as a FUNC in .dynsym.
const char *const KernelName =
    "bazzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz";

// Everything amd_comgr_symbol_get_info reports for one symbol, so that a
// lookup served from the index can be compared field by field against one
// that built it.
struct SymbolFields {
  amd_comgr_symbol_type_t Type;
  uint64_t Size;
  uint64_t Value;
  bool Undefined;

  bool operator==(const SymbolFields &Other) const {
    return Type == Other.Type && Size == Other.Size && Value == Other.Value &&
           Undefined == Other.Undefined;
  }
};

SymbolFields readSymbol(amd_comgr_data_t Data, const char *Name) {
  SymbolFields Fields = {AMD_COMGR_SYMBOL_TYPE_UNKNOWN, 0, 0, false};

  amd_comgr_symbol_t Symbol;
  EXPECT_EQ(AMD_COMGR_STATUS_SUCCESS,
            amd_comgr_symbol_lookup(Data, Name, &Symbol));
  EXPECT_EQ(AMD_COMGR_STATUS_SUCCESS,
            amd_comgr_symbol_get_info(Symbol, AMD_COMGR_SYMBOL_INFO_TYPE,
                                      &Fields.Type));
  EXPECT_EQ(AMD_COMGR_STATUS_SUCCESS,
            amd_comgr_symbol_get_info(Symbol, AMD_COMGR_SYMBOL_INFO_SIZE,
                                      &Fields.Size));
  EXPECT_EQ(AMD_COMGR_STATUS_SUCCESS,
            amd_comgr_symbol_get_info(Symbol, AMD_COMGR_SYMBOL_INFO_VALUE,
                                      &Fields.Value));
  EXPECT_EQ(AMD_COMGR_STATUS_SUCCESS,
            amd_comgr_symbol_get_info(
                Symbol, AMD_COMGR_SYMBOL_INFO_IS_UNDEFINED, &Fields.Undefined));
  return Fields;
}

amd_comgr_status_t collectName(amd_comgr_symbol_t Symbol, void *UserData) {
  size_t Length = 0;
  EXPECT_EQ(AMD_COMGR_STATUS_SUCCESS,
            amd_comgr_symbol_get_info(Symbol, AMD_COMGR_SYMBOL_INFO_NAME_LENGTH,
                                      &Length));

  std::string Name(Length, '\0');
  EXPECT_EQ(AMD_COMGR_STATUS_SUCCESS,
            amd_comgr_symbol_get_info(Symbol, AMD_COMGR_SYMBOL_INFO_NAME,
                                      Name.data()));

  static_cast<std::vector<std::string> *>(UserData)->push_back(Name);
  return AMD_COMGR_STATUS_SUCCESS;
}

// The parse and the name index the symbol APIs share. Neither is observable
// through the public API, so this reaches into src/comgr.h.
class SymbolIndexTest : public ::testing::Test {
protected:
  void SetUp() override {
    std::ifstream File(TEST_CODE_OBJECT, std::ios::binary);
    ASSERT_TRUE(File) << "cannot open " << TEST_CODE_OBJECT;
    Contents.assign(std::istreambuf_iterator<char>(File),
                    std::istreambuf_iterator<char>());
    ASSERT_FALSE(Contents.empty());

    ASSERT_COMGR(create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &Data));
    ASSERT_COMGR(set_data(Data, Contents.size(), Contents.data()));
  }

  void TearDown() override { ASSERT_COMGR(release_data(Data)); }

  DataObject *object() { return DataObject::convert(Data); }

  std::string Contents;
  amd_comgr_data_t Data;
};

TEST_F(SymbolIndexTest, RepeatLookupsAgree) {
  const SymbolFields First = readSymbol(Data, KernelName);
  const SymbolFields Second = readSymbol(Data, KernelName);

  EXPECT_TRUE(First == Second);
  EXPECT_EQ(AMD_COMGR_SYMBOL_TYPE_FUNC, First.Type);
  EXPECT_FALSE(First.Undefined);

  EXPECT_EQ(AMD_COMGR_SYMBOL_TYPE_OBJECT, readSymbol(Data, "foo").Type);
}

TEST_F(SymbolIndexTest, LookupBuildsTheIndexOnce) {
  EXPECT_EQ(nullptr, object()->CachedBinary.get());
  EXPECT_EQ(nullptr, object()->SymbolIndex.get());

  readSymbol(Data, KernelName);
  const llvm::object::Binary *Binary = object()->CachedBinary.get();
  const llvm::StringMap<COMGR::SymbolInfo> *Index = object()->SymbolIndex.get();
  ASSERT_NE(nullptr, Binary);
  ASSERT_NE(nullptr, Index);

  readSymbol(Data, "foo");
  EXPECT_EQ(Binary, object()->CachedBinary.get());
  EXPECT_EQ(Index, object()->SymbolIndex.get());
}

TEST_F(SymbolIndexTest, MissingSymbolFails) {
  amd_comgr_symbol_t Symbol;
  EXPECT_EQ(AMD_COMGR_STATUS_ERROR,
            amd_comgr_symbol_lookup(Data, "definitely_not_a_symbol", &Symbol));
}

TEST_F(SymbolIndexTest, NullArgumentsRejected) {
  amd_comgr_symbol_t Symbol;
  EXPECT_EQ(AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT,
            amd_comgr_symbol_lookup(Data, nullptr, &Symbol));
  EXPECT_EQ(AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT,
            amd_comgr_symbol_lookup(Data, KernelName, nullptr));
}

// The index aliases the buffer it was built from, so replacing the buffer has
// to drop it rather than serve stale hits.
TEST_F(SymbolIndexTest, ResetDataRebuildsTheIndex) {
  const SymbolFields Before = readSymbol(Data, KernelName);
  ASSERT_NE(nullptr, object()->SymbolIndex.get());

  ASSERT_COMGR(set_data(Data, Contents.size(), Contents.data()));
  EXPECT_EQ(nullptr, object()->CachedBinary.get());
  EXPECT_EQ(nullptr, object()->SymbolIndex.get());

  EXPECT_TRUE(Before == readSymbol(Data, KernelName));
}

TEST_F(SymbolIndexTest, IterationSharesTheCachedParse) {
  readSymbol(Data, KernelName);
  const llvm::object::Binary *Binary = object()->CachedBinary.get();
  ASSERT_NE(nullptr, Binary);

  std::vector<std::string> Names;
  ASSERT_COMGR(iterate_symbols(Data, collectName, &Names));

  EXPECT_EQ(Binary, object()->CachedBinary.get());
  EXPECT_EQ(object()->SymbolIndex->size(), Names.size());
  EXPECT_EQ(std::set<std::string>(Names.begin(), Names.end()).size(),
            Names.size());
  EXPECT_NE(Names.end(),
            std::find(Names.begin(), Names.end(), std::string(KernelName)));
}

} // namespace
