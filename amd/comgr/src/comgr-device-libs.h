//===- comgr-device-libs.h - Handle AMD Device Libraries ------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef COMGR_DEVICE_LIBS_H
#define COMGR_DEVICE_LIBS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <string>
#include <tuple>

namespace COMGR {

struct DataAction;
struct DataSet;

llvm::ArrayRef<unsigned char> getDeviceLibrariesIdentifier();
llvm::StringRef getOpenCLCBaseHeaderContents();
llvm::ArrayRef<std::tuple<llvm::StringRef, llvm::StringRef>>
getDeviceLibraries();

// Select the embedded device-library modules needed to resolve and inline OCML
// math entry points for a target ISA using COMGR's default math-control
// policy. The returned list is ordered for link-on-needed use: primary OCML /
// OCKL libraries first, then OCLC control libraries that resolve
// `__oclc_*` constants referenced by OCML.
//
// Selects OCML with HotSwap's default math-control environment: ABI v6,
// finite-only disabled, unsafe-math disabled, the target ISA control library,
// and the target wavefront-size control library. Callers that need different
// math controls must add explicit parameters here rather than selecting OCLC
// modules ad hoc at the call site.
llvm::Error
getOCMLDeviceLibraryNames(llvm::StringRef TargetProcessor,
                          unsigned TargetWaveSize,
                          llvm::SmallVectorImpl<std::string> &Names);

} // namespace COMGR

#endif // COMGR_DEVICE_LIBS_H
