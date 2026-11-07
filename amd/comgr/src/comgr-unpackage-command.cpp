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
/// This file implements the CachedCommandAdaptor interface for
/// llvm::object::OffloadFile::extractOffloadBinaries() routines that are
/// stored in the cache.
///
//===----------------------------------------------------------------------===//

#include <comgr-unpackage-command.h>

#include <comgr-env.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/BinaryFormat/Magic.h>
#include <llvm/Object/OffloadBinary.h>
#include <llvm/Support/Endian.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/raw_ostream.h>

namespace COMGR {
using namespace llvm;

using SizeFieldType = uint32_t;

namespace {
// Determine the output file extension and Comgr data kind for an unpackaged
// offload image by inspecting the image's magic bytes
amd_comgr_status_t getUnpackagedImageInfo(StringRef Image,
                                          const char *&FileExtension,
                                          amd_comgr_data_kind_t &DataKind) {
  switch (identify_magic(Image)) {
  case file_magic::elf:
  case file_magic::elf_relocatable:
  case file_magic::elf_executable:
  case file_magic::elf_shared_object:
  case file_magic::elf_core:
    FileExtension = "o";
    DataKind = AMD_COMGR_DATA_KIND_EXECUTABLE;
    break;
  case file_magic::bitcode:
    FileExtension = "bc";
    DataKind = AMD_COMGR_DATA_KIND_BC;
    break;
  case file_magic::cuda_fatbinary:
    FileExtension = "fatbin";
    DataKind = AMD_COMGR_DATA_KIND_FATBIN;
    break;
  case file_magic::spirv_object:
    FileExtension = "spv";
    DataKind = AMD_COMGR_DATA_KIND_SPIRV;
    break;
  default:
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }
  return AMD_COMGR_STATUS_SUCCESS;
}
} // namespace

bool UnpackageCommand::canCache() const {
  // Uncompressed extraction is already a cheap in-memory parse, so the cache's
  // file I/O and hashing overhead isn't worth it
  return object::OffloadBinary::isCompressed(
      MemoryBufferRef(PackageData, PackageName));
}

Error UnpackageCommand::writeExecuteOutput(StringRef CachedBuffer) {
  OutputFileNames.clear();
  DataKinds.clear();

  while (!CachedBuffer.empty()) {
    if (CachedBuffer.size() < sizeof(SizeFieldType))
      return createStringError(std::errc::invalid_argument,
                               "Not enough bytes to read output name size");
    SizeFieldType NameSize = support::endian::read<SizeFieldType>(
        CachedBuffer.data(), llvm::endianness::native);
    CachedBuffer = CachedBuffer.drop_front(sizeof(NameSize));

    if (CachedBuffer.size() < NameSize)
      return createStringError(std::errc::invalid_argument,
                               "Not enough bytes to read output name");
    StringRef Target = CachedBuffer.substr(0, NameSize);
    CachedBuffer = CachedBuffer.drop_front(NameSize);

    if (CachedBuffer.size() < sizeof(amd_comgr_data_kind_t))
      return createStringError(std::errc::invalid_argument,
                               "Not enough bytes to read output data kind");
    auto DataKind = support::endian::read<amd_comgr_data_kind_t>(
        CachedBuffer.data(), llvm::endianness::native);
    CachedBuffer = CachedBuffer.drop_front(sizeof(DataKind));

    if (CachedBuffer.size() < sizeof(SizeFieldType))
      return createStringError(std::errc::invalid_argument,
                               "Not enough bytes to read output image size");
    SizeFieldType ImageSize = support::endian::read<SizeFieldType>(
        CachedBuffer.data(), llvm::endianness::native);
    CachedBuffer = CachedBuffer.drop_front(sizeof(ImageSize));

    if (CachedBuffer.size() < ImageSize)
      return createStringError(std::errc::invalid_argument,
                               "Not enough bytes to read output image");
    StringRef Image = CachedBuffer.substr(0, ImageSize);
    CachedBuffer = CachedBuffer.drop_front(ImageSize);

    SmallString<128> OutputFilePath = OutputDir;
    sys::path::append(OutputFilePath, Target);

    if (Error Err =
            CachedCommandAdaptor::writeSingleOutputFile(OutputFilePath, Image))
      return Err;

    OutputFileNames.push_back(OutputFilePath.str().str());
    DataKinds.push_back(DataKind);
  }

  return Error::success();
}

Expected<StringRef> UnpackageCommand::readExecuteOutput() {
  OutputBuffer.clear();
  for (const auto &[OutputFilePath, DataKind] :
       llvm::zip_equal(OutputFileNames, DataKinds)) {
    auto MaybeOneOutput =
        CachedCommandAdaptor::readSingleOutputFile(OutputFilePath);
    if (!MaybeOneOutput)
      return MaybeOneOutput.takeError();

    const MemoryBuffer &OneOutputBuffer = **MaybeOneOutput;

    StringRef Target = sys::path::filename(OutputFilePath);
    SizeFieldType NameSize = Target.size();
    SizeFieldType ImageSize = OneOutputBuffer.getBufferSize();

    size_t OldSize = OutputBuffer.size();
    OutputBuffer.resize_for_overwrite(OldSize + sizeof(NameSize) + NameSize +
                                      sizeof(DataKind) + sizeof(ImageSize) +
                                      ImageSize);
    char *Ptr = OutputBuffer.data() + OldSize;

    support::endian::write<SizeFieldType>(Ptr, NameSize,
                                          llvm::endianness::native);
    Ptr += sizeof(NameSize);
    memcpy(Ptr, Target.data(), NameSize);
    Ptr += NameSize;
    support::endian::write<amd_comgr_data_kind_t>(Ptr, DataKind,
                                                  llvm::endianness::native);
    Ptr += sizeof(DataKind);
    support::endian::write<SizeFieldType>(Ptr, ImageSize,
                                          llvm::endianness::native);
    Ptr += sizeof(ImageSize);
    memcpy(Ptr, OneOutputBuffer.getBufferStart(), ImageSize);
  }
  return OutputBuffer;
}

amd_comgr_status_t UnpackageCommand::execute(raw_ostream &LogS) {
  OutputFileNames.clear();
  DataKinds.clear();

  SmallVector<object::OffloadFile> Files;
  MemoryBufferRef DataBufferRef(PackageData, PackageName);
  if (Error E = object::extractOffloadBinaries(DataBufferRef, Files)) {
    logAllUnhandledErrors(std::move(E), LogS,
                          "Failed to extract offload binaries: ");
    return AMD_COMGR_STATUS_ERROR;
  }

  StringRef OutputPrefix = sys::path::stem(PackageName);

  DenseMap<object::OffloadFile::TargetID, StringRef> Worklist;
  SmallVector<std::string> TargetNames;
  SmallVector<const object::OffloadBinary *> MatchedBinaries;
  for (const object::OffloadFile &File : Files) {
    const object::OffloadBinary *Binary = File.getBinary();
    StringRef Triple = Binary->getTriple();
    StringRef Arch = Binary->getArch();
    std::string Target = (Triple + "-" + Arch).str();
    object::OffloadFile::TargetID FileTarget = File;

    for (const std::pair<std::string, std::string> &Entry : PackageEntryIDs) {
      object::OffloadFile::TargetID EntryTarget =
          std::make_pair(StringRef(Entry.first), StringRef(Entry.second));
      // Select files whose target matches the requested entry exactly, or is
      // compatible with it. An exact match is itself a compatible target, so
      // areTargetsCompatible() alone covers both cases.
      if (object::areTargetsCompatible(FileTarget, EntryTarget)) {
        const char *FileExtension;
        amd_comgr_data_kind_t DataKind;
        if (auto Status = getUnpackagedImageInfo(Binary->getImage(),
                                                 FileExtension, DataKind))
          return Status;

        std::string OutputFileName =
            (OutputPrefix + "-" + Target + "." + FileExtension).str();

        TargetNames.push_back(OutputFileName);
        DataKinds.push_back(DataKind);
        MatchedBinaries.push_back(Binary);

        if (env::shouldEmitVerboseLogs()) {
          LogS << "\tPackage Entry Target: " << Target << "\n"
               << "\tOutput Filename: " << OutputFileName << "\n";
          LogS.flush();
        }
      }
    }
  }

  for (const auto &[OutputFileName, Binary] :
       llvm::zip_equal(TargetNames, MatchedBinaries)) {
    SmallString<128> OutputFilePath = OutputDir;
    sys::path::append(OutputFilePath, OutputFileName);

    if (Error Err = CachedCommandAdaptor::writeSingleOutputFile(
            OutputFilePath, Binary->getImage()))
      return AMD_COMGR_STATUS_ERROR;

    OutputFileNames.push_back(OutputFilePath.str().str());
  }

  return AMD_COMGR_STATUS_SUCCESS;
}

CachedCommandAdaptor::ActionClass UnpackageCommand::getClass() const {
  return CachedCommandAdaptor::UnpackageClass;
}

void UnpackageCommand::addOptionsIdentifier(HashAlgorithm &H) const {
  addUInt(H, PackageEntryIDs.size());
  for (const std::pair<std::string, std::string> &Entry : PackageEntryIDs) {
    CachedCommandAdaptor::addString(H, Entry.first);
    CachedCommandAdaptor::addString(H, Entry.second);
  }
}

Error UnpackageCommand::addInputIdentifier(HashAlgorithm &H) const {
  auto MaybeHeader = object::OffloadBinary::extractHeader(
      MemoryBufferRef(PackageData, PackageName));
  if (!MaybeHeader)
    return MaybeHeader.takeError();

  CachedCommandAdaptor::addFileContents(H, PackageData);
  return Error::success();
}

} // namespace COMGR
