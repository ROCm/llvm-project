//===- comgr-hotswap.cpp - HotSwap ISA rewriting — public API bridge ------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "amd_comgr.h"
#include "comgr.h"
#include "comgr-hotswap-internal.h"

using namespace COMGR;

amd_comgr_status_t AMD_COMGR_API amd_comgr_hotswap_rewrite(
    amd_comgr_data_t input,
    const char *source_isa_name, const char *target_isa_name,
    amd_comgr_data_t *output) {
  DataObject *InputP = DataObject::convert(input);
  if (!InputP || !InputP->Data ||
      InputP->DataKind != AMD_COMGR_DATA_KIND_EXECUTABLE ||
      !source_isa_name || !target_isa_name || !output)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  TargetIdentifier SourceIdent, TargetIdent;
  if (parseTargetIdentifier(source_isa_name, SourceIdent) ||
      parseTargetIdentifier(target_isa_name, TargetIdent))
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  if (SourceIdent.Processor != "gfx1250" || TargetIdent.Processor != "gfx1250")
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  void *out_data = nullptr;
  size_t out_size = 0;
  amd_comgr_status_t Status = RetargetCodeObjectB0A0(
      InputP->Data, InputP->Size, &out_data, &out_size);
  if (Status != AMD_COMGR_STATUS_SUCCESS)
    return Status;

  DataObject *OutputP = DataObject::allocate(AMD_COMGR_DATA_KIND_EXECUTABLE);
  if (!OutputP) {
    std::free(out_data);
    return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;
  }

  if (auto SetStatus =
          OutputP->setData(llvm::StringRef(static_cast<char *>(out_data),
                                           out_size))) {
    std::free(out_data);
    OutputP->release();
    return SetStatus;
  }

  std::free(out_data);
  *output = DataObject::convert(OutputP);
  return AMD_COMGR_STATUS_SUCCESS;
}
