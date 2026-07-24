//===- HotswapRewriteFuzzer.cpp ------------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Fuzz the public gfx1250 B0-to-A0 HotSwap rewrite API. Malformed inputs may
/// fail closed. Accepted outputs within the per-iteration resource budget must
/// rewrite a second time successfully and byte-identically.
///
//===----------------------------------------------------------------------===//

#include "amd_comgr.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace {

constexpr const char *SourceIsa =
    "amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+";
constexpr const char *TargetIsa =
    "amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific-";
constexpr size_t MaxInputSize = 64 * 1024 * 1024;
constexpr size_t MaxOutputSize = 128 * 1024 * 1024;

[[noreturn]] void failInvariant() { std::abort(); }

struct RewriteResult {
  amd_comgr_status_t Status = AMD_COMGR_STATUS_ERROR;
  std::vector<uint8_t> Bytes;
  bool ExceededResourceBudget = false;
};

RewriteResult rewrite(const uint8_t *Bytes, size_t Size) {
  amd_comgr_data_t InputData{0};
  amd_comgr_status_t Status =
      amd_comgr_create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &InputData);
  if (Status != AMD_COMGR_STATUS_SUCCESS)
    failInvariant();

  Status = amd_comgr_set_data(InputData, Size,
                              reinterpret_cast<const char *>(Bytes));
  if (Status != AMD_COMGR_STATUS_SUCCESS) {
    amd_comgr_release_data(InputData);
    failInvariant();
  }

  amd_comgr_data_t OutputData{0};
  Status =
      amd_comgr_hotswap_rewrite(InputData, SourceIsa, TargetIsa, &OutputData);
  amd_comgr_release_data(InputData);

  RewriteResult Result;
  Result.Status = Status;
  if (Status != AMD_COMGR_STATUS_SUCCESS)
    return Result;

  size_t OutputSize = 0;
  Status = amd_comgr_get_data(OutputData, &OutputSize, nullptr);
  if (Status != AMD_COMGR_STATUS_SUCCESS || OutputSize == 0) {
    amd_comgr_release_data(OutputData);
    failInvariant();
  }
  if (OutputSize > MaxOutputSize) {
    amd_comgr_release_data(OutputData);
    Result.ExceededResourceBudget = true;
    return Result;
  }

  Result.Bytes.resize(OutputSize);
  Status = amd_comgr_get_data(OutputData, &OutputSize,
                              reinterpret_cast<char *>(Result.Bytes.data()));
  amd_comgr_release_data(OutputData);
  if (Status != AMD_COMGR_STATUS_SUCCESS)
    failInvariant();
  Result.Bytes.resize(OutputSize);
  return Result;
}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  if (!Data || Size == 0 || Size > MaxInputSize)
    return 0;

  RewriteResult First = rewrite(Data, Size);
  if (First.Status != AMD_COMGR_STATUS_SUCCESS)
    return 0;
  // The cap is a fuzzer resource policy, not a COMGR API contract. A valid
  // rewrite may legitimately exceed it, so skip the expensive idempotency
  // oracle instead of reporting a false correctness failure.
  if (First.ExceededResourceBudget)
    return 0;

  RewriteResult Second = rewrite(First.Bytes.data(), First.Bytes.size());
  if (Second.Status != AMD_COMGR_STATUS_SUCCESS ||
      Second.ExceededResourceBudget || First.Bytes != Second.Bytes)
    failInvariant();
  return 0;
}
