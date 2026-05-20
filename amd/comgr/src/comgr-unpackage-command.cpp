//===- comgr-unpackage-command.cpp - UnpackageCommand implementation
//--------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the CacheCommandAdaptor interface for
/// llvm::object::OffloadFile::extractOffloadBinaries() routines
/// that are stored in the cache.
///
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include <comgr-unpackage-command.h>

#include <llvm/ADT/StringMap.h>
#include <llvm/BinaryFormat/Magic.h>
#include <llvm/Object/OffloadBinary.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/raw_ostream.h>

namespace COMGR {
using namespace llvm;
using namespace clang;

using SizeFieldType = uint32_t;

bool UnpackageCommand::canCache() const {
  // TODO: when llvm-offload-binary gets compression support, investigate
  // caching compressed packages
  return false;
}

Error UnpackageCommand::writeExecuteOutput(StringRef CachedBuffer) {
  for (StringRef OutputFilename : OutputFileNames) {
    SizeFieldType OutputFileSize;
    if (CachedBuffer.size() < sizeof(OutputFileSize)) {
      return createStringError(std::errc::invalid_argument,
                               "Not enough bytes to read output file size");
    }
    memcpy(&OutputFileSize, CachedBuffer.data(), sizeof(OutputFileSize));
    CachedBuffer = CachedBuffer.drop_front(sizeof(OutputFileSize));

    if (CachedBuffer.size() < OutputFileSize) {
      return createStringError(std::errc::invalid_argument,
                               "Not enough bytes to read output file contents");
    }

    StringRef OutputFileContents = CachedBuffer.substr(0, OutputFileSize);
    CachedBuffer = CachedBuffer.drop_front(OutputFileSize);

    if (Error Err = CachedCommandAdaptor::writeSingleOutputFile(
            OutputFilename, OutputFileContents)) {
      return Err;
    }
  }

  if (!CachedBuffer.empty()) {
    return createStringError(std::errc::invalid_argument,
                             "Bytes in cache entry not used for the output");
  }
  return Error::success();
}

Expected<StringRef> UnpackageCommand::readExecuteOutput() {
  size_t OutputSize = 0;
  for (StringRef OutputFilename : OutputFileNames) {
    auto MaybeOneOutput =
        CachedCommandAdaptor::readSingleOutputFile(OutputFilename);
    if (!MaybeOneOutput) {
      return MaybeOneOutput.takeError();
    }

    const MemoryBuffer &OneOutputBuffer = **MaybeOneOutput;
    SizeFieldType OneOutputFileSize = OneOutputBuffer.getBufferSize();

    OutputBuffer.resize_for_overwrite(OutputSize + sizeof(OneOutputFileSize) +
                                      OneOutputFileSize);

    memcpy(OutputBuffer.data() + OutputSize, &OneOutputFileSize,
           sizeof(OneOutputFileSize));
    OutputSize += sizeof(OneOutputFileSize);
    memcpy(OutputBuffer.data() + OutputSize, OneOutputBuffer.getBufferStart(),
           OneOutputFileSize);
    OutputSize += OneOutputFileSize;
  }
  return OutputBuffer;
}

amd_comgr_status_t UnpackageCommand::execute(raw_ostream &LogS) {
  StringMap<StringRef> Worklist;
  for (const auto &[Output, Target] :
       llvm::zip_equal(OutputFileNames, TargetNames)) {
    Worklist[Target] = Output;
  }

  for (const llvm::object::OffloadFile &File : Files) {
    const llvm::object::OffloadBinary *Binary = File.getBinary();
    StringRef Triple = Binary->getTriple();
    StringRef Arch = Binary->getArch();
    std::string Target = (Triple + "-" + Arch).str();

    // TODO: does this instead need to check that the triples are compatible?
    // (rather than simply equivalent)
    if (Worklist.contains(Target)) {
      StringRef Image = Binary->getImage();

      // create an output file descriptor
      auto OutputName = Worklist[Target];
      std::error_code EC;
      raw_fd_ostream OutputFile(OutputName, EC, sys::fs::OF_None);
      if (EC) {
        return AMD_COMGR_STATUS_ERROR;
      }

      // write the packaged image into the output
      OutputFile << Image;
      OutputFile.flush();

      // erase the entry from Worklist (so that we can track if all expected
      // files have been unpackaged)
      Worklist.erase(Target);
    }
  }

  // if not all expected files were unpackaged, possibly throw an error
  // TODO: should this have an option associated with it? unbundler doesn't
  if (!Worklist.empty()) {
    // the unbundler is invoked to ignore missing bundles, so, in matching that
    // behavior, the following error shouldn't be thrown:

    // return AMD_COMGR_STATUS_ERROR;
  }

  return AMD_COMGR_STATUS_SUCCESS;
}

CachedCommandAdaptor::ActionClass UnpackageCommand::getClass() const {
  // TODO: look into an upstream OffloadUnpackagerJobClass?
  return clang::driver::Action::OffloadPackagerJobClass;
}

void UnpackageCommand::addOptionsIdentifier(HashAlgorithm &H) const {
  addUInt(H, TargetNames.size());
  for (StringRef Target : TargetNames) {
    CachedCommandAdaptor::addString(H, Target);
  }
}

Error UnpackageCommand::addInputIdentifier(HashAlgorithm &H) const {
  return Error::success();
}

} // namespace COMGR
