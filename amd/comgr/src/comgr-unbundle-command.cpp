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

#include <comgr-env.h>
#include <comgr-unbundle-command.h>

#include <clang/Basic/TargetID.h>
#include <clang/Driver/OffloadBundler.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/BinaryFormat/Magic.h>
#include <llvm/Object/OffloadBundle.h>
#include <llvm/Support/Alignment.h>
#include <llvm/Support/Endian.h>
#include <llvm/Support/FileSystem.h>

#include <memory>

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
constexpr const char CompressedBundleMagic[] = "CCOB";
constexpr size_t CompressedBundleMagicSize = sizeof(CompressedBundleMagic) - 1;

struct CompressedBundleRange {
  /// The end of the current bundle. A legacy bundle without a recorded size
  /// extends to the next bundle or the end of the input.
  size_t End;
  /// The offset of the next bundle, or StringRef::npos for the final bundle.
  size_t NextStart;
};

Expected<CompressedBundleRange> getCompressedBundleRange(StringRef Blob) {
  if (Blob.size() < 8)
    return createStringError("compressed bundle header size too small");

  Expected<object::CompressedOffloadBundle::CompressedBundleHeader> Header =
      object::CompressedOffloadBundle::CompressedBundleHeader::tryParse(Blob);
  if (!Header)
    return Header.takeError();

  if (Header->FileSize) {
    size_t End = *Header->FileSize;
    size_t HeaderSize = Header->Version == 2 ? 24 : 32;
    if (End < HeaderSize || End > Blob.size())
      return createStringError("invalid compressed bundle size");
    return CompressedBundleRange{End, Blob.find(CompressedBundleMagic, End)};
  }

  // Legacy bundle without a recorded size: fall back to magic scanning.
  size_t NextStart = Blob.find(CompressedBundleMagic, CompressedBundleMagicSize);
  return CompressedBundleRange{NextStart, NextStart};
}

// Keep this predicate consistent with isCodeObjectCompatible in
// clang/lib/Driver/OffloadBundler.cpp. That helper is not part of Clang's
// public interface. Use the public target parsing APIs here so the in-memory
// path selects the same members as the file-based unbundler.
bool isCompatibleBundle(const OffloadTargetInfo &CodeObjectInfo,
                        const OffloadTargetInfo &TargetInfo) {
  if (CodeObjectInfo == TargetInfo)
    return true;

  if (!CodeObjectInfo.isOffloadKindCompatible(TargetInfo.OffloadKind) ||
      !CodeObjectInfo.Triple.isCompatibleWith(TargetInfo.Triple))
    return false;

  StringMap<bool> CodeObjectFeatures, TargetFeatures;
  auto CodeObjectProcessor = clang::parseTargetID(
      CodeObjectInfo.Triple, CodeObjectInfo.TargetID, &CodeObjectFeatures);
  auto TargetProcessor = clang::parseTargetID(
      TargetInfo.Triple, TargetInfo.TargetID, &TargetFeatures);
  if (!CodeObjectProcessor || !TargetProcessor ||
      *CodeObjectProcessor != *TargetProcessor)
    return false;

  // An unspecified code-object feature accepts either target value. Every
  // specified feature must also be specified by the target with the same value.
  for (const auto &Feature : CodeObjectFeatures) {
    auto TargetFeature = TargetFeatures.find(Feature.getKey());
    if (TargetFeature == TargetFeatures.end() ||
        TargetFeature->second != Feature.second)
      return false;
  }
  return true;
}

// Persisting a very large decompressed bundle costs more than regenerating it.
// The cache entry has to frame and write the complete image, while the in-memory
// unbundler can decompress the same image directly into its output buffer.
// Keep the cache for ordinary bundles and bypass it for multi-hundred-megabyte
// code-object collections such as hipBLASLt's per-dtype libraries.
constexpr size_t MaxCachedUncompressedSize = 64 * 1024 * 1024;
constexpr size_t MinCompressedBundleHeaderSize = 8;
constexpr size_t MaxCompressedBundleHeaderSize = 32;

Expected<bool> compressedBundlesFitCache(StringRef InputFilename,
                                         uint64_t InputSize) {
  using Header = object::CompressedOffloadBundle::CompressedBundleHeader;

  uint64_t Offset = 0;
  while (Offset < InputSize) {
    uint64_t RemainingSize = InputSize - Offset;
    if (RemainingSize < MinCompressedBundleHeaderSize)
      return createStringError("truncated compressed bundle header");

    ErrorOr<std::unique_ptr<MemoryBuffer>> HeaderBuffer =
        MemoryBuffer::getFileSlice(
            InputFilename,
            std::min<uint64_t>(RemainingSize, MaxCompressedBundleHeaderSize),
            Offset);
    if (!HeaderBuffer)
      return createStringError(HeaderBuffer.getError(),
                               "failed to read compressed bundle header");

    StringRef HeaderBytes = (*HeaderBuffer)->getBuffer();
    if (identify_magic(HeaderBytes) != file_magic::offload_bundle_compressed)
      return createStringError("invalid concatenated compressed bundle");

    Expected<Header> MaybeHeader = Header::tryParse(HeaderBytes);
    if (!MaybeHeader)
      return MaybeHeader.takeError();
    if (MaybeHeader->UncompressedFileSize > MaxCachedUncompressedSize)
      return false;

    if (!MaybeHeader->FileSize) {
      // Version 1 has no total size, so locate the next bundle by magic, as
      // the in-memory unbundler does.
      ErrorOr<std::unique_ptr<MemoryBuffer>> RemainingBuffer =
          MemoryBuffer::getFileSlice(InputFilename, RemainingSize, Offset);
      if (!RemainingBuffer)
        return createStringError(RemainingBuffer.getError(),
                                 "failed to read compressed bundle");
      size_t NextStart = (*RemainingBuffer)
                             ->getBuffer()
                             .find(CompressedBundleMagic,
                                   CompressedBundleMagicSize);
      if (NextStart == StringRef::npos)
        return true;
      Offset += NextStart;
      continue;
    }

    uint64_t BundleSize = *MaybeHeader->FileSize;
    size_t HeaderSize = MaybeHeader->Version == 2 ? 24 : 32;
    if (BundleSize < HeaderSize || BundleSize > RemainingSize)
      return createStringError("invalid compressed bundle size");
    Offset += BundleSize;
  }
  return true;
}

/// Append \p Data to \p Out, prefixed with its size. This is the format used
/// for the contents of each output in the cache entry.
void appendSizePrefixed(SmallVectorImpl<char> &Out, StringRef Data) {
  SizeFieldType Size = Data.size();
  size_t Offset = Out.size();
  Out.resize_for_overwrite(Offset + sizeof(Size) + Data.size());
  memcpy(Out.data() + Offset, &Size, sizeof(Size));
  memcpy(Out.data() + Offset + sizeof(Size), Data.data(), Data.size());
}

/// A MemoryBuffer that views a subrange of \p Owner and keeps it alive.
///
/// A compressed bundle expands to an image far larger than the bundle itself,
/// and every requested target is a subrange of that image. Copying each
/// subrange would duplicate the image once per target, so the outputs share
/// the decompressed buffer instead. The whole backing buffer remains alive
/// until its last view is released, even if only a small member is retained.
class SharedSubBuffer final : public MemoryBuffer {
public:
  SharedSubBuffer(std::shared_ptr<MemoryBuffer> Owner, StringRef Slice)
      : Owner(std::move(Owner)) {
    init(Slice.data(), Slice.data() + Slice.size(),
         /*RequiresNullTerminator=*/false);
  }

  BufferKind getBufferKind() const override { return MemoryBuffer_Malloc; }

private:
  std::shared_ptr<MemoryBuffer> Owner;
};

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
  if (Magic != file_magic::offload_bundle_compressed)
    return false;

  uint64_t InputSize;
  if (std::error_code EC = sys::fs::file_size(InputFilename, InputSize))
    return true;

  if (InputSize < MinCompressedBundleHeaderSize)
    return false;

  // Only map each header prefix. Mapping the complete compressed bundle here
  // would page in hundreds of megabytes just to decide whether to cache it.
  Expected<bool> FitsCache = compressedBundlesFitCache(InputFilename, InputSize);
  if (!FitsCache) {
    consumeError(FitsCache.takeError());
    return false;
  }
  return *FitsCache;
}

Error UnbundleCommand::splitCachedOutput(StringRef CachedBuffer,
                                         SmallVectorImpl<StringRef> &Out) {
  for (size_t Index = 0, Count = Config.OutputFileNames.size(); Index < Count;
       ++Index) {
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

Error UnbundleCommand::saveOutputTemps() const {
  if (!env::shouldSaveTemps())
    return Error::success();

  for (const auto &[Filename, Output] :
       llvm::zip_equal(Config.OutputFileNames, OutputBuffers))
    if (Error Err = CachedCommandAdaptor::writeSingleOutputFile(
            Filename, Output->getBuffer()))
      return Err;
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
  if (Error Err = saveOutputTemps()) {
    // A failed cache restore may retry through execute(). Do not retain
    // cached outputs if that retry falls back to the file-based path.
    OutputBuffers.clear();
    return Err;
  }
  return Error::success();
}

Expected<StringRef> UnbundleCommand::readExecuteOutput() {
  // The outputs were unbundled in memory, frame them for the cache entry.
  if (!OutputBuffers.empty()) {
    // Size the buffer up front. Appending an image much larger than the
    // SmallString's inline capacity would otherwise reallocate and copy the
    // whole image more than once.
    size_t TotalSize = 0;
    for (const std::unique_ptr<MemoryBuffer> &Output : OutputBuffers)
      TotalSize += sizeof(SizeFieldType) + Output->getBufferSize();
    OutputBuffer.reserve(TotalSize);

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
  StringMap<size_t> Worklist;
  for (auto [Index, Target] : llvm::enumerate(Config.TargetNames)) {
    if (!checkOffloadBundleID(Target))
      return createStringError(std::errc::invalid_argument,
                               "invalid bundle id from bundle config");
    if (!Worklist.try_emplace(Target, Index).second)
      return false;
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
    Expected<CompressedBundleRange> Range = getCompressedBundleRange(Remaining);
    if (!Range)
      return Range.takeError();
    size_t CurBundleEnd = Range->End;
    NextBundleStart = Range->NextStart;

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

    // The outputs are subranges of this buffer, so they share ownership of it
    // rather than copying their contents.
    std::shared_ptr<MemoryBuffer> DecompressedOwner =
        std::move(*DecompressedOrErr);

    StringRef Decompressed = DecompressedOwner->getBuffer();

    // Only the binary bundle format can be read without a file. Anything
    // else, such as an object file carrying the bundles in its sections, is
    // left to the file-based path.
    if (identify_magic(Decompressed) != file_magic::offload_bundle)
      return false;

    // Read the number of bundles and the table of bundle offsets, sizes and
    // triples that follows the magic.
    size_t ReadChars = BinaryBundleMagicSize;
    if (ReadChars + sizeof(uint64_t) > Decompressed.size())
      return createStringError("truncated binary bundle header");
    uint64_t NumberOfBundles =
        support::endian::read64le(Decompressed.data() + ReadChars);
    ReadChars += sizeof(uint64_t);

    StringMap<StringRef> Bundles;
    for (uint64_t BundleIndex = 0; BundleIndex < NumberOfBundles;
         ++BundleIndex) {
      if (Decompressed.size() - ReadChars < 3 * sizeof(uint64_t))
        return createStringError("truncated binary bundle header");
      uint64_t BundleOffset =
          support::endian::read64le(Decompressed.data() + ReadChars);
      ReadChars += sizeof(uint64_t);
      uint64_t BundleSize =
          support::endian::read64le(Decompressed.data() + ReadChars);
      ReadChars += sizeof(uint64_t);
      uint64_t TripleSize =
          support::endian::read64le(Decompressed.data() + ReadChars);
      ReadChars += sizeof(uint64_t);
      if (TripleSize > Decompressed.size() - ReadChars)
        return createStringError("truncated binary bundle target");
      StringRef Triple = Decompressed.substr(ReadChars, TripleSize);
      ReadChars += TripleSize;

      // Check that the offset and size make sense.
      if (!BundleOffset || BundleOffset > Decompressed.size() ||
          BundleSize > Decompressed.size() - BundleOffset)
        return createStringError("invalid binary bundle range");

      Bundles[Triple] = Decompressed.substr(BundleOffset, BundleSize);
    }

    for (const auto &Bundle : Bundles) {
      if (Worklist.empty())
        break;
      StringRef Triple = Bundle.getKey();
      if (!checkOffloadBundleID(Triple))
        return createStringError(std::errc::invalid_argument,
                                 "invalid bundle id read from the bundle");
      auto Target = llvm::find_if(Worklist, [&](const auto &Entry) {
        return isCompatibleBundle(
            OffloadTargetInfo(Triple, Config),
            OffloadTargetInfo(Entry.getKey(), Config));
      });
      if (Target == Worklist.end())
        continue;

      // LLVM's ELF reader requires two-byte base alignment. Preserve that
      // requirement for downstream metadata and symbol readers.
      if (isAddrAligned(Align(2), Bundle.second.data()))
        Outputs[Target->second] =
            std::make_unique<SharedSubBuffer>(DecompressedOwner, Bundle.second);
      else
        Outputs[Target->second] = MemoryBuffer::getMemBufferCopy(Bundle.second);

      Worklist.erase(Target);

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
    for (const auto &Entry : Worklist) {
      size_t Index = Entry.second;
      StringRef Target = Entry.getKey();
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
  for (const auto &Entry : Worklist) {
    size_t Index = Entry.second;
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
    if (*InMemory) {
      if (Error Err = saveOutputTemps()) {
        logAllUnhandledErrors(std::move(Err), LogS, "Unbundle Error: ");
        return AMD_COMGR_STATUS_ERROR;
      }
      return AMD_COMGR_STATUS_SUCCESS;
    }

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

  StringRef Input = (**MaybeInputBuffer).getBuffer();
  size_t Offset = 0;
  while (Offset < Input.size()) {
    StringRef Remaining = Input.drop_front(Offset);
    if (identify_magic(Remaining) != file_magic::offload_bundle_compressed)
      return createStringError(std::errc::invalid_argument,
                               "invalid concatenated compressed bundle");

    Expected<CompressedBundleRange> Range = getCompressedBundleRange(Remaining);
    if (!Range)
      return Range.takeError();

    using Header = object::CompressedOffloadBundle::CompressedBundleHeader;
    Expected<Header> MaybeHeader = Header::tryParse(Remaining);
    if (!MaybeHeader)
      return MaybeHeader.takeError();

    // The hash represents the uncompressed contents of this bundle.
    // Extracting the same contents should give the same result regardless of
    // the compression algorithm or header version. Since the hash used by the
    // offload bundler is not cryptographic, also include the uncompressed file
    // size. Hash every concatenated bundle because the requested target may
    // occur only in a later one.
    addUInt(H, MaybeHeader->Hash);
    addUInt(H, MaybeHeader->UncompressedFileSize);

    if (Range->NextStart == StringRef::npos)
      break;
    Offset += Range->NextStart;
  }
  return Error::success();
}

} // namespace COMGR
