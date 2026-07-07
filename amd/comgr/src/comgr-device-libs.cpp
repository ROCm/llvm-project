//===- comgr-device-libs.cpp - Handle AMD Device Libraries ----------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the handling of the AMD Device Libraries, which are
/// LLVM IR objects embedded into Comgr via header files.
///
/// We also handle OpenCL pre-compiled headers, which are similarly embedded in
/// Comgr.
///
//===----------------------------------------------------------------------===//

#include "comgr-device-libs.h"
#include "comgr.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include "llvm/TargetParser/TargetParser.h"
#include <cstdint>

using namespace llvm;

namespace COMGR {

namespace {
#include "libraries.inc"
#include "libraries_sha.inc"
#include "opencl-c-base.inc"
} // namespace

ArrayRef<unsigned char> getDeviceLibrariesIdentifier() {
  return DEVICE_LIBS_ID;
}

StringRef getOpenCLCBaseHeaderContents() {
  return StringRef(reinterpret_cast<const char *>(opencl_c_base),
                   opencl_c_base_size);
}

llvm::ArrayRef<std::tuple<llvm::StringRef, llvm::StringRef>>
getDeviceLibraries() {
  static std::tuple<llvm::StringRef, llvm::StringRef> DeviceLibs[] = {
#define AMD_DEVICE_LIBS_TARGET(target)                                         \
  {#target ".bc",                                                              \
   llvm::StringRef(reinterpret_cast<const char *>(target##_lib),               \
                   target##_lib_size)},
#include "libraries_defs.inc"
  };
  return DeviceLibs;
}

namespace {

bool hasEmbeddedDeviceLibrary(llvm::StringRef Name) {
  for (const auto &Lib : getDeviceLibraries()) {
    if (std::get<0>(Lib) == Name)
      return true;
  }
  return false;
}

llvm::Error
validateSelectedDeviceLibraries(llvm::ArrayRef<llvm::StringRef> Names) {
  for (llvm::StringRef Name : Names) {
    if (hasEmbeddedDeviceLibrary(Name))
      continue;

    return llvm::createStringError("selected OCML device library '" + Name +
                                   "' is not embedded in this COMGR build");
  }
  return llvm::Error::success();
}

} // namespace

llvm::Error
getOCMLDeviceLibraryNames(llvm::StringRef TargetProcessor,
                          unsigned TargetWaveSize,
                          llvm::SmallVectorImpl<std::string> &Names) {
  Names.clear();

  AMDGPU::GPUKind Kind = AMDGPU::parseArchAMDGCN(TargetProcessor);
  if (Kind == AMDGPU::GK_NONE)
    return createStringError("target processor '" + TargetProcessor +
                             "' does not name a known AMDGPU processor");

  StringRef CanonicalProcessor = AMDGPU::getArchNameAMDGCN(Kind);
  if (!CanonicalProcessor.consume_front("gfx"))
    return createStringError("LLVM returned non-gfx AMDGPU processor name '" +
                             AMDGPU::getArchNameAMDGCN(Kind) + "'");

  std::string IsaSuffix = CanonicalProcessor.str();
  for (char &C : IsaSuffix)
    if (C == '-')
      C = '_';
  std::string IsaLibraryName = "oclc_isa_version_" + IsaSuffix + ".bc";

  if (TargetWaveSize != 32 && TargetWaveSize != 64)
    return createStringError("cannot select OCML wavefront-size control "
                             "library for target wave size " +
                             Twine(TargetWaveSize));

  llvm::SmallVector<llvm::StringRef, 8> Selected = {
      "ocml.bc",
      "ockl.bc",
      "oclc_abi_version_600.bc",
      IsaLibraryName,
      "oclc_finite_only_off.bc",
      "oclc_unsafe_math_off.bc",
      TargetWaveSize == 64 ? "oclc_wavefrontsize64_on.bc"
                           : "oclc_wavefrontsize64_off.bc",
  };

  if (auto Error = validateSelectedDeviceLibraries(Selected))
    return Error;

  for (llvm::StringRef Name : Selected)
    Names.push_back(Name.str());

  return llvm::Error::success();
}

} // namespace COMGR
