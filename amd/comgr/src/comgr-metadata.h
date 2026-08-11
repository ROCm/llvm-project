//===- comgr-metadata.h - Metadata query internals ------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef COMGR_METADATA_H
#define COMGR_METADATA_H

#include "comgr.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Error.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

namespace COMGR {
namespace metadata {

amd_comgr_status_t getMetadataRoot(DataObject *DataP, DataMeta *MetaP);

// Buffer-friendly overloads for callers that hold raw code-object bytes
// without a `DataObject` (e.g. the hotswap transpiler running over an HSACO
// buffer), so they reach the same note walker / ISA-string formatter without
// the public C `amd_comgr_create_data` ceremony. These preserve the underlying
// LLVM parse error; the `DataObject` overloads above map it to a status.
llvm::Error getMetadataRoot(llvm::MemoryBufferRef MB, DataMeta *MetaP);
llvm::Expected<std::string> getElfIsaName(llvm::MemoryBufferRef MB);

size_t getIsaCount();

const char *getIsaName(size_t Index);

amd_comgr_status_t getIsaMetadata(llvm::StringRef IsaName,
                                  llvm::msgpack::Document &MetaP);

bool isValidIsaName(llvm::StringRef IsaName);

amd_comgr_status_t getElfIsaName(DataObject *DataP, std::string &IsaName);

amd_comgr_status_t lookUpCodeObject(DataObject *DataP,
                                    amd_comgr_code_object_info_t *QueryList,
                                    size_t QueryListsize);

/// Look up the ISA table index for \p IsaName, optionally reporting the
/// resolved canonical processor in \p Processor.
amd_comgr_status_t getIsaIndex(const llvm::StringRef IsaName, size_t &Index,
                               llvm::StringRef *Processor = nullptr);

bool isSupportedFeature(size_t IsaIndex, llvm::StringRef Feature);

} // namespace metadata
} // namespace COMGR

#endif
