//===- CacheDirectoryTest.cpp - Cache directory tests --------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "comgr-env.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <sys/vfs.h>

using namespace llvm;

namespace {

constexpr StringLiteral CacheHome = "/comgr-cache-directory-test";
std::optional<uint32_t> FilesystemType;
unsigned ProbeCount = 0;

class ScopedEnvironment {
  const char *Name;
  std::optional<std::string> Previous;

public:
  ScopedEnvironment(const char *Name, const char *Value)
      : Name(Name), Previous(sys::Process::GetEnv(Name)) {
    EXPECT_EQ(Value ? setenv(Name, Value, 1) : unsetenv(Name), 0);
  }

  ~ScopedEnvironment() {
    EXPECT_EQ(Previous ? setenv(Name, Previous->c_str(), 1) : unsetenv(Name),
              0);
  }
};

class CacheDirectoryTest : public ::testing::Test {
  ScopedEnvironment Enable{"AMD_COMGR_CACHE", nullptr};
  ScopedEnvironment Directory{"AMD_COMGR_CACHE_DIR", nullptr};
  ScopedEnvironment DefaultDirectory{"XDG_CACHE_HOME", CacheHome.data()};
  ScopedEnvironment Verbose{"AMD_COMGR_EMIT_VERBOSE_LOGS", "1"};

protected:
  void SetUp() override { ProbeCount = 0; }
};

TEST_F(CacheDirectoryTest, DisablesSharedFilesystems) {
  for (uint32_t Type : {0x3e3fU, 0x0bd00bd0U, 0x47504653U, // DVS, Lustre, GPFS.
                        0x6969U, 0x517bU, 0xff534d42U}) {  // NFS, SMB, CIFS.
    SCOPED_TRACE(Type);
    FilesystemType = Type;
    std::string Log;
    raw_string_ostream LogS(Log);
    EXPECT_FALSE(COMGR::env::getCacheDirectory(LogS));
    EXPECT_NE(Log.find("is not on a local filesystem; disabling cache"),
              std::string::npos);
  }
}

TEST_F(CacheDirectoryTest, KeepsLocalFilesystemsEnabled) {
  for (uint32_t Type : {0xef53U, 0x58465342U, 0x9123683eU, 0x01021994U}) {
    // ext4, XFS, Btrfs, tmpfs.
    SCOPED_TRACE(Type);
    FilesystemType = Type;
    std::string Log;
    raw_string_ostream LogS(Log);
    auto Directory = COMGR::env::getCacheDirectory(LogS);
    ASSERT_TRUE(Directory);
    EXPECT_EQ(*Directory, "/comgr-cache-directory-test/comgr");
    EXPECT_TRUE(Log.empty());
  }
}

TEST_F(CacheDirectoryTest, DisablesCacheWhenFilesystemProbeFails) {
  FilesystemType = std::nullopt;
  EXPECT_FALSE(COMGR::env::getCacheDirectory(nulls()));
  EXPECT_GT(ProbeCount, 0U);
}

TEST_F(CacheDirectoryTest, ExplicitDirectoryOverridesFilesystemCheck) {
  ScopedEnvironment DirectoryOverride{"AMD_COMGR_CACHE_DIR", CacheHome.data()};
  FilesystemType = 0x3e3f;
  auto Directory = COMGR::env::getCacheDirectory(nulls());
  ASSERT_TRUE(Directory);
  EXPECT_EQ(*Directory, CacheHome);
  EXPECT_EQ(ProbeCount, 0U);
}

} // namespace

// NOLINTNEXTLINE(readability-identifier-naming) -- GNU ld --wrap symbols.
extern "C" int __real_statfs(const char *Path, struct statfs *Stat);

// NOLINTNEXTLINE(readability-identifier-naming) -- GNU ld --wrap symbols.
extern "C" int __wrap_statfs(const char *Path, struct statfs *Stat) {
  if (StringRef(Path) != CacheHome)
    return __real_statfs(Path, Stat);
  ++ProbeCount;
  if (!FilesystemType) {
    errno = EIO;
    return -1;
  }
  *Stat = {};
  Stat->f_type = *FilesystemType;
  return 0;
}
