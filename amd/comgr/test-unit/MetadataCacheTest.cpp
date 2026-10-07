//===- MetadataCacheTest.cpp - Metadata cache tests -----------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "comgr.h"

#include "common.h"
#include "gtest/gtest.h"

#include <fstream>
#include <string>

using COMGR::DataMeta;
using COMGR::DataObject;
using COMGR::MetaDocument;

namespace {

// The parsed document a metadata handle is a view into. Identity of this
// pointer across two handles is what distinguishes a cache hit from a re-parse;
// it is not observable through the public API.
const MetaDocument *document(amd_comgr_metadata_node_t Node) {
  return DataMeta::convert(Node)->MetaDoc.get();
}

// Reads "amdhsa.version" as "<major>.<minor>", so that a cache hit is checked
// to carry the same contents as a fresh parse and not just the same pointer.
std::string readVersion(amd_comgr_metadata_node_t Root) {
  amd_comgr_metadata_node_t Version;
  EXPECT_EQ(AMD_COMGR_STATUS_SUCCESS,
            amd_comgr_metadata_lookup(Root, "amdhsa.version", &Version));

  size_t Count = 0;
  EXPECT_EQ(AMD_COMGR_STATUS_SUCCESS,
            amd_comgr_get_metadata_list_size(Version, &Count));

  std::string Out;
  for (size_t I = 0; I < Count; ++I) {
    amd_comgr_metadata_node_t Element;
    EXPECT_EQ(AMD_COMGR_STATUS_SUCCESS,
              amd_comgr_index_list_metadata(Version, I, &Element));

    char Buf[16];
    size_t Size = sizeof(Buf);
    EXPECT_EQ(AMD_COMGR_STATUS_SUCCESS,
              amd_comgr_get_metadata_string(Element, &Size, Buf));

    if (I)
      Out += '.';
    Out += Buf;

    EXPECT_EQ(AMD_COMGR_STATUS_SUCCESS, amd_comgr_destroy_metadata(Element));
  }

  EXPECT_EQ(AMD_COMGR_STATUS_SUCCESS, amd_comgr_destroy_metadata(Version));
  return Out;
}

class MetadataCacheTest : public ::testing::Test {
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

  void resetData() {
    ASSERT_COMGR(set_data(Data, Contents.size(), Contents.data()));
  }

  std::string Contents;
  amd_comgr_data_t Data;
};

TEST_F(MetadataCacheTest, RepeatQueriesShareOneDocument) {
  amd_comgr_metadata_node_t First, Second;
  ASSERT_COMGR(get_data_metadata(Data, &First));
  ASSERT_COMGR(get_data_metadata(Data, &Second));

  EXPECT_EQ(document(First), document(Second));
  EXPECT_EQ(document(First), DataObject::convert(Data)->CachedMetaDoc.get());
  EXPECT_EQ(readVersion(First), readVersion(Second));

  ASSERT_COMGR(destroy_metadata(First));
  ASSERT_COMGR(destroy_metadata(Second));
}

// The cached document is shared by pointer, so a handle outliving its sibling
// must keep it alive.
TEST_F(MetadataCacheTest, DestroyingOneHandleKeepsSiblingValid) {
  amd_comgr_metadata_node_t First, Second;
  ASSERT_COMGR(get_data_metadata(Data, &First));
  ASSERT_COMGR(get_data_metadata(Data, &Second));

  const std::string Expected = readVersion(First);
  ASSERT_COMGR(destroy_metadata(First));

  EXPECT_EQ(Expected, readVersion(Second));
  ASSERT_COMGR(destroy_metadata(Second));
}

TEST_F(MetadataCacheTest, ResetDataInvalidatesCache) {
  amd_comgr_metadata_node_t Before;
  ASSERT_COMGR(get_data_metadata(Data, &Before));
  const std::string Expected = readVersion(Before);

  // Held across the reset: this keeps the old document alive, so the address
  // cannot be recycled and compare equal to a genuinely fresh parse below.
  resetData();
  EXPECT_EQ(nullptr, DataObject::convert(Data)->CachedMetaDoc.get());

  amd_comgr_metadata_node_t After;
  ASSERT_COMGR(get_data_metadata(Data, &After));
  EXPECT_NE(document(Before), document(After));
  EXPECT_EQ(Expected, readVersion(After));

  ASSERT_COMGR(destroy_metadata(Before));
  ASSERT_COMGR(destroy_metadata(After));
}

} // namespace
