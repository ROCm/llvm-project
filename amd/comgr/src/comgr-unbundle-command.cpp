//===- comgr-unbundle-command.cpp - UnbundleCommand implementation --------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the CacheCommandAdaptor interface for
/// llvm::OffloadBundler::Unbundle() routines that are stored in the cache.
///
//===----------------------------------------------------------------------===//

#include <comgr-unbundle-command.h>

#include <clang/Driver/OffloadBundler.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/BinaryFormat/Magic.h>
#include <llvm/Object/OffloadBundle.h>
#include <llvm/Support/Endian.h>

namespace COMGR {
using namespace llvm;
using namespace clang;

using SizeFieldType = uint32_t;

namespace {

// Magic of the binary bundle format written by the offload bundler. It is
// followed by the number of bundles and a table of bundle offsets, sizes and
// triples. See clang/lib/Driver/OffloadBundler.cpp for the full layout.
constexpr const char BinaryBundleMagic[] = "__CLANG_OFFLOAD_BUNDLE__";
constexpr size_t BinaryBundleMagicSize = sizeof(BinaryBundleMagic) - 1;

/// Append \p Data to \p Out, prefixed with its size. This is the format used
/// for the contents of each output in the cache entry.
void appendSizePrefixed(SmallVectorImpl<char> &Out, StringRef Data) {
  SizeFieldType Size = Data.size();
  size_t Offset = Out.size();
  Out.resize_for_overwrite(Offset + sizeof(Size) + Data.size());
  memcpy(Out.data() + Offset, &Size, sizeof(Size));
  memcpy(Out.data() + Offset + sizeof(Size), Data.data(), Data.size());
}

/// Find the index of the first requested target that \p Triple is compatible
/// with, or std::nullopt if there is none.
std::optional<size_t> findCompatibleTarget(StringRef Triple,
                                           const OffloadBundlerConfig &Config) {
  OffloadTargetInfo CodeObjectInfo(Triple, Config);
  for (auto [Index, Target] : llvm::enumerate(Config.TargetNames)) {
    if (isCodeObjectCompatible(CodeObjectInfo,
                               OffloadTargetInfo(Target, Config)))
      return static_cast<size_t>(Index);
  }
  return std::nullopt;
}

} // namespace

bool UnbundleCommand::canCache() const {
  // The header format for AR files is not the same as object files
  if (Kind == AMD_COMGR_DATA_KIND_AR_BUNDLE)
    return false;

  StringRef InputFilename = Config.InputFileNames.front();
  file_magic Magic;
  if (identify_magic(InputFilename, Magic))
    return false;

  // Check the input file magic. Handle only compressed bundles
  // It's not worth to cache other types of bundles
  return Magic == file_magic::offload_bundle_compressed;
}

Error UnbundleCommand::splitCachedOutput(StringRef CachedBuffer,
                                         SmallVectorImpl<StringRef> &Out) {
  for (StringRef OutputFilename : Config.OutputFileNames) {
    SizeFieldType OutputFileSize;
    if (CachedBuffer.size() < sizeof(OutputFileSize))
      return createStringError(std::errc::invalid_argument,
                               "Not enough bytes to read output file size");
    memcpy(&OutputFileSize, CachedBuffer.data(), sizeof(OutputFileSize));
    CachedBuffer = CachedBuffer.drop_front(sizeof(OutputFileSize));

    if (CachedBuffer.size() < OutputFileSize)
      return createStringError(std::errc::invalid_argument,
                               "Not enough bytes to read output file contents");

    Out.push_back(CachedBuffer.substr(0, OutputFileSize));
    CachedBuffer = CachedBuffer.drop_front(OutputFileSize);
  }

  if (!CachedBuffer.empty())
    return createStringError(std::errc::invalid_argument,
                             "Bytes in cache entry not used for the output");
  return Error::success();
}

Error UnbundleCommand::writeExecuteOutput(StringRef CachedBuffer) {
  SmallVector<StringRef> Contents;
  if (Error Err = splitCachedOutput(CachedBuffer, Contents))
    return Err;

  for (StringRef Content : Contents) {
    std::unique_ptr<MemoryBuffer> Output =
        MemoryBuffer::getMemBufferCopy(Content);
    if (!Output)
      return createStringError(std::errc::not_enough_memory,
                               "Failed to allocate unbundled output");
    OutputBuffers.push_back(std::move(Output));
  }
  return Error::success();
}

Expected<StringRef> UnbundleCommand::readExecuteOutput() {
  // The outputs were unbundled in memory, frame them for the cache entry.
  if (!OutputBuffers.empty()) {
    for (const std::unique_ptr<MemoryBuffer> &Output : OutputBuffers)
      appendSizePrefixed(OutputBuffer, Output->getBuffer());
    return StringRef(OutputBuffer.data(), OutputBuffer.size());
  }

  for (StringRef OutputFilename : Config.OutputFileNames) {
    auto MaybeOneOutput =
        CachedCommandAdaptor::readSingleOutputFile(OutputFilename);
    if (!MaybeOneOutput)
      return MaybeOneOutput.takeError();

    const MemoryBuffer &OneOutputBuffer = **MaybeOneOutput;
    appendSizePrefixed(OutputBuffer, OneOutputBuffer.getBuffer());
  }
  return OutputBuffer;
}

Expected<bool> UnbundleCommand::executeInMemory() {
  // Archives and uncompressed bundles are handled by the file-based path.
  if (Kind == AMD_COMGR_DATA_KIND_AR_BUNDLE)
    return false;
  assert(Config.TargetNames.size() == Config.OutputFileNames.size());

  StringRef InputFileName = Config.InputFileNames.front();
  ErrorOr<std::unique_ptr<MemoryBuffer>> MaybeInputBuffer =
      MemoryBuffer::getFile(InputFileName, /*IsText=*/false,
                            /*RequiresNullTerminator=*/false);
  if (!MaybeInputBuffer) {
    std::error_code EC = MaybeInputBuffer.getError();
    return createStringError(EC, Twine("Failed to open ") + InputFileName +
                                     " : " + EC.message() + "\n");
  }
  StringRef Input = (*MaybeInputBuffer)->getBuffer();

  // The worklist holds the index of each target that has not been unbundled
  // yet. The outputs are collected separately so that falling back to the
  // file-based path does not leave partial results behind.
  SmallVector<size_t, 4> Worklist;
  for (auto [Index, Target] : llvm::enumerate(Config.TargetNames)) {
    if (!checkOffloadBundleID(Target))
      return createStringError(std::errc::invalid_argument,
                               "invalid bundle id from bundle config");
    Worklist.push_back(Index);
  }

  SmallVector<std::unique_ptr<MemoryBuffer>, 4> Outputs(
      Config.TargetNames.size());
  bool FoundHostBundle = false;

  // The input may contain multiple concatenated bundles; drain the worklist
  // across all of them, as OffloadBundler::UnbundleFiles() does.
  size_t Offset = 0;
  size_t NextBundleStart = 0;
  while (NextBundleStart != StringRef::npos && Offset < Input.size()) {
    StringRef Remaining = Input.drop_front(Offset);
    if (identify_magic(Remaining) != file_magic::offload_bundle_compressed)
      return false;

    // Locate this bundle's end and the next bundle from the header size.
    size_t CurBundleEnd = StringRef::npos;
    Expected<object::CompressedOffloadBundle::CompressedBundleHeader> Header =
        object::CompressedOffloadBundle::CompressedBundleHeader::tryParse(
            Remaining);
    if (Header)
      CurBundleEnd = Header->FileSize.value_or(StringRef::npos);
    else
      consumeError(Header.takeError());

    if (CurBundleEnd != StringRef::npos) {
      NextBundleStart = Remaining.find("CCOB", CurBundleEnd);
    } else {
      // Legacy bundle without a recorded size: fall back to magic scanning.
      NextBundleStart = Remaining.find("CCOB", 4);
      CurBundleEnd = NextBundleStart;
    }

    ErrorOr<std::unique_ptr<MemoryBuffer>> BlobOrErr =
        MemoryBuffer::getMemBuffer(Remaining.take_front(CurBundleEnd),
                                   InputFileName,
                                   /*RequiresNullTerminator=*/false);
    if (!BlobOrErr) {
      std::error_code EC = BlobOrErr.getError();
      return createStringError(EC, Twine("Failed to open ") + InputFileName +
                                       " : " + EC.message() + "\n");
    }

    Expected<std::unique_ptr<MemoryBuffer>> DecompressedOrErr =
        object::CompressedOffloadBundle::decompress(**BlobOrErr);
    if (!DecompressedOrErr)
      return DecompressedOrErr.takeError();

    StringRef Decompressed = (*DecompressedOrErr)->getBuffer();

    // Only the binary bundle format can be read without a file. Anything
    // else, such as an object file carrying the bundles in its sections, is
    // left to the file-based path.
    if (identify_magic(Decompressed) != file_magic::offload_bundle)
      return false;

    // Read the number of bundles and the table of bundle offsets, sizes and
    // triples that follows the magic.
    size_t ReadChars = BinaryBundleMagicSize;
    if (ReadChars + sizeof(uint64_t) > Decompressed.size())
      return false;
    uint64_t NumberOfBundles =
        support::endian::read64le(Decompressed.data() + ReadChars);
    ReadChars += sizeof(uint64_t);

    for (uint64_t I = 0; I < NumberOfBundles && !Worklist.empty(); ++I) {
      if (ReadChars + 3 * sizeof(uint64_t) > Decompressed.size())
        return false;
      uint64_t BundleOffset =
          support::endian::read64le(Decompressed.data() + ReadChars);
      ReadChars += sizeof(uint64_t);
      uint64_t BundleSize =
          support::endian::read64le(Decompressed.data() + ReadChars);
      ReadChars += sizeof(uint64_t);
      uint64_t TripleSize =
          support::endian::read64le(Decompressed.data() + ReadChars);
      ReadChars += sizeof(uint64_t);
      if (ReadChars + TripleSize > Decompressed.size())
        return false;
      StringRef Triple = Decompressed.substr(ReadChars, TripleSize);
      ReadChars += TripleSize;

      // Check that the offset and size make sense.
      if (!BundleOffset || BundleOffset + BundleSize > Decompressed.size())
        return false;

      std::optional<size_t> Target = findCompatibleTarget(Triple, Config);
      if (!Target)
        continue;

      StringRef Bundle = Decompressed.substr(BundleOffset, BundleSize);
      Outputs[*Target] = MemoryBuffer::getMemBufferCopy(Bundle);
      if (!Outputs[*Target])
        return createStringError(std::errc::not_enough_memory,
                                 "Failed to allocate unbundled output");

      llvm::erase(Worklist, *Target);

      if (OffloadTargetInfo(Triple, Config).hasHostKind())
        FoundHostBundle = true;
    }

    if (NextBundleStart != StringRef::npos)
      Offset += NextBundleStart;
  }

  // Leave the cases that the file-based path reports as errors to it.
  if (!Config.AllowMissingBundles && !Worklist.empty())
    return false;

  // If no bundles were found, the file-based path assumes the input is the
  // host bundle and copies it to the host targets, leaving the rest empty.
  if (Worklist.size() == Config.TargetNames.size()) {
    for (size_t Index : Worklist) {
      StringRef Target = Config.TargetNames[Index];
      Outputs[Index] = OffloadTargetInfo(Target, Config).hasHostKind()
                           ? MemoryBuffer::getMemBufferCopy(Input)
                           : MemoryBuffer::getMemBufferCopy("");
      if (!Outputs[Index])
        return createStringError(std::errc::not_enough_memory,
                                 "Failed to allocate unbundled output");
    }
    OutputBuffers = std::move(Outputs);
    return true;
  }

  if (!FoundHostBundle && Config.HostInputIndex != ~0u &&
      !Config.AllowMissingBundles)
    return false;

  // Any target that was not found gets an empty output, as the file-based
  // path does.
  for (size_t Index : Worklist) {
    Outputs[Index] = MemoryBuffer::getMemBufferCopy("");
    if (!Outputs[Index])
      return createStringError(std::errc::not_enough_memory,
                               "Failed to allocate unbundled output");
  }

  OutputBuffers = std::move(Outputs);
  return true;
}

amd_comgr_status_t UnbundleCommand::execute(raw_ostream &LogS) {
  assert(Config.InputFileNames.size() == 1);

  OffloadBundler Bundler(Config);

  switch (Kind) {
  case AMD_COMGR_DATA_KIND_BC_BUNDLE:
  case AMD_COMGR_DATA_KIND_OBJ_BUNDLE: {
    // Unbundle compressed bundles in memory. This avoids writing and reading
    // back the unbundled images, which can be much larger than the bundle.
    Expected<bool> InMemory = executeInMemory();
    if (!InMemory) {
      logAllUnhandledErrors(InMemory.takeError(), LogS, "Unbundle Error: ");
      return AMD_COMGR_STATUS_ERROR;
    }
    if (*InMemory)
      return AMD_COMGR_STATUS_SUCCESS;

    if (Error Err = Bundler.UnbundleFiles()) {
      logAllUnhandledErrors(std::move(Err), LogS, "Unbundle Error: ");
      return AMD_COMGR_STATUS_ERROR;
    }
    break;
  }
  case AMD_COMGR_DATA_KIND_AR_BUNDLE: {
    if (Error Err = Bundler.UnbundleArchive()) {
      logAllUnhandledErrors(std::move(Err), LogS, "Unbundle Archives Error: ");
      return AMD_COMGR_STATUS_ERROR;
    }
    break;
  }
  default:
    assert(false && "invalid bundle type");
    LogS << "Unbundle Error: invalid bundle type\n";
    return AMD_COMGR_STATUS_ERROR;
  }

  return AMD_COMGR_STATUS_SUCCESS;
}

CachedCommandAdaptor::ActionClass UnbundleCommand::getClass() const {
  return CachedCommandAdaptor::UnbundleClass;
}

void UnbundleCommand::addOptionsIdentifier(HashAlgorithm &H) const {
  addUInt(H, Config.TargetNames.size());
  for (StringRef Target : Config.TargetNames) {
    CachedCommandAdaptor::addString(H, Target);
  }
}

Error UnbundleCommand::addInputIdentifier(HashAlgorithm &H) const {
  StringRef InputFilename = Config.InputFileNames.front();

  ErrorOr<std::unique_ptr<MemoryBuffer>> MaybeInputBuffer =
      MemoryBuffer::getFile(InputFilename);
  if (!MaybeInputBuffer) {
    std::error_code EC = MaybeInputBuffer.getError();
    return createStringError(EC, Twine("Failed to open ") + InputFilename +
                                     " : " + EC.message() + "\n");
  }

  MemoryBuffer &InputBuffer = **MaybeInputBuffer;

  using Header = object::CompressedOffloadBundle::CompressedBundleHeader;
  Expected<Header> MaybeHeader = Header::tryParse(InputBuffer.getBuffer());
  if (!MaybeHeader)
    return MaybeHeader.takeError();

  // The hash represents the contents of the bundle. Extracting the same
  // contents should give the same result, regardless of the compression
  // algorithm or header version. Since the hash used by the offload bundler is
  // not a cryptographic hash, we also add the uncompressed file size.
  addUInt(H, MaybeHeader->Hash);
  addUInt(H, MaybeHeader->UncompressedFileSize);
  return Error::success();
}

} // namespace COMGR
