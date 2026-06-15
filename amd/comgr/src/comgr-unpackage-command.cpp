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
/// This file implements the UnpackageCommand, which extracts offload binaries
/// from a package via llvm::object::OffloadFile::extractOffloadBinaries().
///
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include <comgr-unpackage-command.h>

#include <llvm/ADT/StringMap.h>
#include <llvm/Object/OffloadBinary.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/raw_ostream.h>

namespace COMGR {
using namespace llvm;

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

} // namespace COMGR
