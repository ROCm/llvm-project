//===- SetCStrTest.cpp - Unit tests for COMGR::setCStr --------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// setCStr() replaces a heap-allocated C string owned by a Comgr object
// (DataObject::Name, SymbolContext::Name). Its callers can hand it a source
// string that aliases the destination's own storage, so it has to allocate and
// copy the replacement before it releases the old buffer. Freeing first and
// copying afterwards reads freed memory.
//
// This is a unit test of an internal helper. No public Comgr API currently
// hands out a pointer into a data object's storage, so these inputs are built
// by the test itself; the cases below pin the invariant setCStr() documents,
// not a sequence the public API can express today.
//
//===----------------------------------------------------------------------===//

#include "common.h"
#include "comgr.h"
#include "gtest/gtest.h"

#include "llvm/ADT/StringRef.h"

#include <cstdlib>
#include <cstring>
#include <vector>

using namespace COMGR;
using llvm::StringRef;

namespace {

// setCStr() releases Dest, so every buffer passed to it must come from
// malloc(). The pattern is deterministic and non-repeating, so a copy taken
// from the wrong place, or allocator metadata written over a freed chunk,
// cannot compare equal by accident.
char *pattern(size_t Size) {
  char *Bytes = static_cast<char *>(malloc(Size + 1));
  if (!Bytes)
    return nullptr;
  for (size_t Index = 0; Index < Size; ++Index)
    Bytes[Index] = static_cast<char>('a' + (Index % 26));
  Bytes[Size] = '\0';
  return Bytes;
}

} // namespace

// -- ordinary replacement ----------------------------------------------------
//
// The shape every existing caller uses: the source is unrelated to Dest and
// must survive as Dest's new contents.

TEST(SetCStr, NonAliasingSource) {
  char *Dest = pattern(32);
  char *Replacement = pattern(11);
  ASSERT_NE(nullptr, Dest);
  ASSERT_NE(nullptr, Replacement);

  size_t Size = 0;
  ASSERT_EQ(AMD_COMGR_STATUS_SUCCESS,
            setCStr(Dest, StringRef(Replacement, 11), &Size));

  EXPECT_EQ(11u, Size);
  EXPECT_EQ(0, memcmp(Dest, Replacement, 11));
  EXPECT_EQ('\0', Dest[11]);

  free(Dest);
  free(Replacement);
}

// -- aliasing sources --------------------------------------------------------
//
// Src is a view of the buffer Dest owns. The replacement has to be copied out
// before the old buffer is released. A size sweep matters because allocator
// behaviour differs across size classes, and because a zero-length source
// copies nothing at all.

TEST(SetCStr, SelfAliasingSource) {
  for (size_t Length : {size_t(0), size_t(1), size_t(7), size_t(31),
                        size_t(32), size_t(100), size_t(2048), size_t(65536)}) {
    char *Storage = pattern(Length);
    ASSERT_NE(nullptr, Storage);
    std::vector<char> Expected(Storage, Storage + Length);
    char *Dest = Storage;

    size_t Size = 0;
    ASSERT_EQ(AMD_COMGR_STATUS_SUCCESS,
              setCStr(Dest, StringRef(Dest, Length), &Size))
        << "Length=" << Length;

    EXPECT_EQ(Length, Size) << "Length=" << Length;
    EXPECT_EQ('\0', Dest[Length]) << "Length=" << Length;
    for (size_t Index = 0; Index < Length; ++Index)
      ASSERT_EQ(Expected[Index], Dest[Index])
          << "Length=" << Length << " Index=" << Index;

    free(Dest);
  }
}

// Src starts part-way into the buffer Dest owns, so the replacement is a
// suffix of the storage that is about to be released.

TEST(SetCStr, SuffixAliasingSource) {
  for (size_t Length : {size_t(2), size_t(64), size_t(4096)}) {
    char *Storage = pattern(Length);
    ASSERT_NE(nullptr, Storage);
    const size_t Offset = Length / 2;
    std::vector<char> Expected(Storage + Offset, Storage + Length);
    char *Dest = Storage;

    size_t Size = 0;
    ASSERT_EQ(AMD_COMGR_STATUS_SUCCESS,
              setCStr(Dest, StringRef(Dest + Offset, Length - Offset), &Size))
        << "Length=" << Length;

    EXPECT_EQ(Length - Offset, Size) << "Length=" << Length;
    ASSERT_EQ(0, memcmp(Dest, Expected.data(), Length - Offset))
        << "Length=" << Length;

    free(Dest);
  }
}

// -- call-site shapes --------------------------------------------------------

// A null destination is how a freshly allocated object starts out.

TEST(SetCStr, NullDestination) {
  char *Dest = nullptr;
  const char Source[] = "symbol";
  size_t Size = 0;

  ASSERT_EQ(AMD_COMGR_STATUS_SUCCESS,
            setCStr(Dest, StringRef(Source, strlen(Source)), &Size));

  ASSERT_NE(nullptr, Dest);
  EXPECT_EQ(strlen(Source), Size);
  EXPECT_STREQ(Source, Dest);

  free(Dest);
}

// The size out-parameter is optional; the string alone must still be replaced.

TEST(SetCStr, NoSizeOutParameter) {
  char *Dest = pattern(9);
  ASSERT_NE(nullptr, Dest);

  ASSERT_EQ(AMD_COMGR_STATUS_SUCCESS, setCStr(Dest, StringRef("abc", 3)));
  EXPECT_STREQ("abc", Dest);

  free(Dest);
}

// -- ordinary data objects are unaffected ------------------------------------
//
// Sanity case for the public entry point the internal callers serve, so this
// suite also shows a normal set-data round trip is unchanged.

TEST(SetData, PublicApiRoundTrip) {
  amd_comgr_data_t Data;
  ASSERT_COMGR(create_data(AMD_COMGR_DATA_KIND_BYTES, &Data));

  const char Bytes[] = "hello";
  ASSERT_COMGR(set_data(Data, sizeof(Bytes) - 1, Bytes));

  size_t Size = 0;
  ASSERT_COMGR(get_data(Data, &Size, nullptr));
  EXPECT_EQ(sizeof(Bytes) - 1, Size);

  std::vector<char> Buffer(Size + 1);
  ASSERT_COMGR(get_data(Data, &Size, Buffer.data()));
  EXPECT_EQ(0, memcmp(Buffer.data(), Bytes, sizeof(Bytes) - 1));

  ASSERT_COMGR(release_data(Data));
}
