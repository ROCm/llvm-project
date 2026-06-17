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

#include <llvm/ADT/DenseMap.h>
#include <llvm/Object/OffloadBinary.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/raw_ostream.h>

namespace COMGR {
using namespace llvm;

amd_comgr_status_t UnpackageCommand::execute(raw_ostream &LogS) {
  DenseMap<llvm::object::OffloadFile::TargetID, StringRef> Worklist;
  for (const auto &[Output, Target] :
       llvm::zip_equal(OutputFileNames, TargetIDs)) {
    Worklist[Target] = Output;
  }

  for (const llvm::object::OffloadFile &File : Files) {
    const llvm::object::OffloadBinary *Binary = File.getBinary();
    llvm::object::OffloadFile::TargetID Target = File;

    // Prefer an exact match, then fall back to a compatible target. Note that
    // areTargetsCompatible() reports false for an exact match (it only flags
    // distinct-but-compatible targets), so the explicit lookup below is
    // required to handle equal targets. We track the matching Worklist entry by
    // iterator so it can be erased once written.
    DenseMap<llvm::object::OffloadFile::TargetID, StringRef>::iterator Match =
        Worklist.find(Target);
    if (Match == Worklist.end()) {
      for (DenseMap<llvm::object::OffloadFile::TargetID, StringRef>::iterator
               I = Worklist.begin(),
               E = Worklist.end();
           I != E; ++I) {
        if (llvm::object::areTargetsCompatible(I->first, Target)) {
          Match = I;
          break;
        }
      }
    }

    if (Match != Worklist.end()) {
      StringRef Image = Binary->getImage();

      // create an output file descriptor
      StringRef OutputName = Match->second;
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
      Worklist.erase(Match);
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
