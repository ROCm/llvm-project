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
using CompressedBundleHeader =
    object::CompressedOffloadBundle::CompressedBundleHeader;

struct CompressedBundleRange {
  /// The end of the current bundle. A legacy bundle without a recorded size
  /// extends to the next bundle or the end of the input.
  size_t End;
  /// The offset of the next bundle, or StringRef::npos for the final bundle.
  size_t NextStart;
};

/// Return the size of a compressed bundle header for \p Version.
size_t compressedBundleHeaderSize(uint16_t Version) {
  // CCOB magic (4), version (2), method (2), then the version-specific
  // sizes and the truncated MD5 hash (8).
  switch (Version) {
  case 1:
    return 20;
  case 2:
    return 24;
  case 3:
    return 32;
  default:
    return 0;
  }
}

/// Return the size of the compressed bundle at the start of \p Remaining.
/// \p RemainingSize is the number of bytes in the complete remaining input;
/// \p Remaining may be only the header prefix for versioned bundles.
Expected<size_t>
nextCompressedBundleOffset(size_t RemainingSize, StringRef Remaining,
                           const CompressedBundleHeader &Header) {
  if (Header.FileSize) {
    size_t HeaderSize = compressedBundleHeaderSize(Header.Version);
    if (*Header.FileSize < HeaderSize || *Header.FileSize > RemainingSize)
      return createStringError("invalid compressed bundle size");
    return *Header.FileSize;
  }

  // Version 1 has no recorded size, so locate the next magic.
  return Remaining.find(CompressedBundleMagic, CompressedBundleMagicSize);
}

Expected<CompressedBundleRange> getCompressedBundleRange(StringRef Blob) {
  if (Blob.size() < 8)
    return createStringError("compressed bundle header size too small");

  Expected<CompressedBundleHeader> Header =
      CompressedBundleHeader::tryParse(Blob);
  if (!Header)
    return Header.takeError();

  Expected<size_t> Next =
      nextCompressedBundleOffset(Blob.size(), Blob, *Header);
  if (!Next)
    return Next.takeError();
  if (Header->FileSize)
    return CompressedBundleRange{*Next,
                                 Blob.find(CompressedBundleMagic, *Next)};
  return CompressedBundleRange{*Next, *Next};
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
// The cache entry has to frame and write the complete image, while the
// in-memory unbundler can decompress the same image directly into its output
// buffer. Keep the cache for ordinary bundles and bypass it for
// multi-hundred-megabyte code-object collections such as hipBLASLt's per-dtype
// libraries.
constexpr size_t MaxCachedUncompressedSize = 64 * 1024 * 1024;
constexpr size_t MinCompressedBundleHeaderSize = 8;
constexpr size_t MaxCompressedBundleHeaderSize = 32;

// Each unbundled member is aligned to max_align_t relative to the mapped cache
// entry and framed with its size and padding, so a cache hit can borrow the
// member instead of copying it. The leading magic distinguishes this framing
// from the original size-prefixed layout, which is still accepted on read.
constexpr const char UnbundleCacheMagic[] = "CUB1";
constexpr size_t UnbundleCacheMagicSize = sizeof(UnbundleCacheMagic) - 1;
constexpr size_t UnbundleCacheAlignment = alignof(std::max_align_t);
// serializeCacheEntry() prefixes the command output with an eight-byte size.
// Align members relative to that actual position in the cache file.
constexpr size_t UnbundleCacheContentsOffset = sizeof(uint64_t);
static_assert(UnbundleCacheMagicSize == sizeof(SizeFieldType));

/// Append \p Data to \p Out in the aligned cache framing. The frame stores the
/// payload size and the number of padding bytes before the payload.
void appendAlignedOutput(SmallVectorImpl<char> &Out, StringRef Data) {
  SizeFieldType Size = Data.size();
  size_t HeaderOffset = Out.size();
  size_t DataOffset = HeaderOffset + 2 * sizeof(SizeFieldType);
  size_t AlignedOffset = alignTo(DataOffset + UnbundleCacheContentsOffset,
                                 UnbundleCacheAlignment) -
                         UnbundleCacheContentsOffset;
  SizeFieldType Padding = AlignedOffset - DataOffset;

  Out.resize_for_overwrite(AlignedOffset + Data.size());
  memcpy(Out.data() + HeaderOffset, &Size, sizeof(Size));
  memcpy(Out.data() + HeaderOffset + sizeof(Size), &Padding, sizeof(Padding));
  if (Padding)
    memset(Out.data() + DataOffset, 0, Padding);
  if (!Data.empty())
    memcpy(Out.data() + AlignedOffset, Data.data(), Data.size());
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

  StringRef getBufferIdentifier() const override {
    return Owner->getBufferIdentifier();
  }
  BufferKind getBufferKind() const override { return Owner->getBufferKind(); }

private:
  std::shared_ptr<MemoryBuffer> Owner;
};

std::unique_ptr<MemoryBuffer> shareOrCopy(std::shared_ptr<MemoryBuffer> Owner,
                                          StringRef Slice) {
  if (Owner && isAddrAligned(Align(alignof(std::max_align_t)), Slice.data()))
    return std::make_unique<SharedSubBuffer>(std::move(Owner), Slice);
  return MemoryBuffer::getMemBufferCopy(Slice);
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
  if (Magic != file_magic::offload_bundle_compressed)
    return false;

  uint64_t InputSize;
  if (std::error_code EC = sys::fs::file_size(InputFilename, InputSize))
    return true;

  if (InputSize < MinCompressedBundleHeaderSize)
    return false;

  // Only map the header prefix for versioned bundles. Mapping the complete
  // compressed bundle here would page in hundreds of megabytes just to decide
  // whether to cache it. Version 1 has no recorded size and needs a scan.
  std::unique_ptr<MemoryBuffer> FullBuffer;
  uint64_t Offset = 0;
  while (Offset < InputSize) {
    uint64_t RemainingSize = InputSize - Offset;
    if (RemainingSize < MinCompressedBundleHeaderSize)
      return false;

    ErrorOr<std::unique_ptr<MemoryBuffer>> HeaderBuffer =
        MemoryBuffer::getFileSlice(
            InputFilename,
            std::min<uint64_t>(RemainingSize, MaxCompressedBundleHeaderSize),
            Offset);
    if (!HeaderBuffer)
      return true;
    if (identify_magic((*HeaderBuffer)->getBuffer()) !=
        file_magic::offload_bundle_compressed)
      return false;

    Expected<CompressedBundleHeader> Header =
        CompressedBundleHeader::tryParse((*HeaderBuffer)->getBuffer());
    if (!Header) {
      consumeError(Header.takeError());
      return false;
    }

    if (Header->UncompressedFileSize > MaxCachedUncompressedSize)
      return false;

    StringRef ScanData = (*HeaderBuffer)->getBuffer();
    if (!Header->FileSize) {
      if (!FullBuffer) {
        ErrorOr<std::unique_ptr<MemoryBuffer>> MaybeFullBuffer =
            MemoryBuffer::getFile(InputFilename);
        if (!MaybeFullBuffer)
          return true;
        FullBuffer = std::move(*MaybeFullBuffer);
      }
      ScanData = FullBuffer->getBuffer().drop_front(Offset);
    }

    Expected<size_t> NextOffset =
        nextCompressedBundleOffset(RemainingSize, ScanData, *Header);
    if (!NextOffset) {
      consumeError(NextOffset.takeError());
      return false;
    }
    if (*NextOffset == StringRef::npos)
      break;
    Offset += *NextOffset;
  }

  return true;
}

Error UnbundleCommand::splitCachedOutput(StringRef CachedBuffer,
                                         SmallVectorImpl<StringRef> &Out) {
  bool IsAligned = CachedBuffer.consume_front(UnbundleCacheMagic);
  for (size_t Index = 0, Count = Config.OutputFileNames.size(); Index < Count;
       ++Index) {
    SizeFieldType OutputFileSize;
    if (CachedBuffer.size() < sizeof(OutputFileSize))
      return createStringError(std::errc::invalid_argument,
                               "Not enough bytes to read output file size");
    memcpy(&OutputFileSize, CachedBuffer.data(), sizeof(OutputFileSize));
    CachedBuffer = CachedBuffer.drop_front(sizeof(OutputFileSize));

    if (IsAligned) {
      SizeFieldType Padding;
      if (CachedBuffer.size() < sizeof(Padding))
        return createStringError(std::errc::invalid_argument,
                                 "Not enough bytes to read output padding");
      memcpy(&Padding, CachedBuffer.data(), sizeof(Padding));
      CachedBuffer = CachedBuffer.drop_front(sizeof(Padding));
      if (Padding > CachedBuffer.size())
        return createStringError(std::errc::invalid_argument,
                                 "Output padding exceeds cache entry");
      CachedBuffer = CachedBuffer.drop_front(Padding);
    }

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
  return restoreCachedOutput(CachedBuffer, nullptr);
}

Error UnbundleCommand::restoreCachedOutput(
    StringRef CachedBuffer, std::shared_ptr<MemoryBuffer> Owner) {
  SmallVector<StringRef> Contents;
  if (Error Err = splitCachedOutput(CachedBuffer, Contents))
    return Err;

  OutputBuffers.clear();
  for (StringRef Content : Contents)
    OutputBuffers.push_back(shareOrCopy(Owner, Content));
  if (Error Err = saveOutputTemps()) {
    // A failed cache restore may retry through execute(). Do not retain
    // cached outputs if that retry falls back to the file-based path.
    OutputBuffers.clear();
    return Err;
  }
  return Error::success();
}

Expected<StringRef> UnbundleCommand::readExecuteOutput() {
  OutputBuffer.clear();
  OutputBuffer.append(UnbundleCacheMagic,
                      UnbundleCacheMagic + UnbundleCacheMagicSize);
  // The outputs were unbundled in memory, frame them for the cache entry.
  if (!OutputBuffers.empty()) {
    // Size the buffer up front. Appending an image much larger than the
    // SmallString's inline capacity would otherwise reallocate and copy the
    // whole image more than once.
    size_t TotalSize = UnbundleCacheMagicSize +
                       OutputBuffers.size() * 2 * sizeof(SizeFieldType) +
                       OutputBuffers.size() * (UnbundleCacheAlignment - 1);
    for (const std::unique_ptr<MemoryBuffer> &Output : OutputBuffers)
      TotalSize += Output->getBufferSize();
    OutputBuffer.reserve(TotalSize);
    for (const std::unique_ptr<MemoryBuffer> &Output : OutputBuffers)
      appendAlignedOutput(OutputBuffer, Output->getBuffer());
    return OutputBuffer;
  }

  for (StringRef OutputFilename : Config.OutputFileNames) {
    auto MaybeOneOutput =
        CachedCommandAdaptor::readSingleOutputFile(OutputFilename);
    if (!MaybeOneOutput)
      return MaybeOneOutput.takeError();

    const MemoryBuffer &OneOutputBuffer = **MaybeOneOutput;
    appendAlignedOutput(OutputBuffer, OneOutputBuffer.getBuffer());
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

    std::unique_ptr<MemoryBuffer> Blob = MemoryBuffer::getMemBuffer(
        Remaining.take_front(CurBundleEnd), InputFileName,
        /*RequiresNullTerminator=*/false);

    Expected<std::unique_ptr<MemoryBuffer>> DecompressedOrErr =
        object::CompressedOffloadBundle::decompress(*Blob);
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
        return isCompatibleBundle(OffloadTargetInfo(Triple, Config),
                                  OffloadTargetInfo(Entry.getKey(), Config));
      });
      if (Target == Worklist.end())
        continue;

      // Preserve the alignment needed by downstream ELF, metadata, and symbol
      // readers; a valid byte range alone is not sufficient for a shared view.
      Outputs[Target->second] = shareOrCopy(DecompressedOwner, Bundle.second);

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

  bool NoBundlesFound = Worklist.size() == Config.TargetNames.size();
  if (!NoBundlesFound && !FoundHostBundle && Config.HostInputIndex != ~0u &&
      !Config.AllowMissingBundles)
    return false;

  // Missing targets get empty outputs. If no bundles matched, the file-based
  // path instead copies the input to any requested host target.
  for (const auto &Entry : Worklist) {
    StringRef Contents;
    if (NoBundlesFound &&
        OffloadTargetInfo(Entry.getKey(), Config).hasHostKind())
      Contents = Input;
    Outputs[Entry.second] = MemoryBuffer::getMemBufferCopy(Contents);
    if (!Outputs[Entry.second])
      return createStringError(std::errc::not_enough_memory,
                               "Failed to allocate unbundled output");
  }

  OutputBuffers = std::move(Outputs);
  return true;
}

amd_comgr_status_t UnbundleCommand::execute(raw_ostream &LogS) {
  assert(Config.InputFileNames.size() == 1);
  OutputBuffers.clear();

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
  if (llvm::any_of(Config.TargetNames, [&](StringRef Target) {
        return checkOffloadBundleID(Target) &&
               OffloadTargetInfo(Target, Config).hasHostKind();
      })) {
    addString(H, Input);
    return Error::success();
  }
  size_t Offset = 0;
  while (Offset < Input.size()) {
    StringRef Remaining = Input.drop_front(Offset);
    if (Remaining.size() < MinCompressedBundleHeaderSize ||
        identify_magic(Remaining) != file_magic::offload_bundle_compressed)
      return createStringError(std::errc::invalid_argument,
                               "invalid concatenated compressed bundle");

    using Header = CompressedBundleHeader;
    Expected<Header> MaybeHeader = Header::tryParse(Remaining);
    if (!MaybeHeader)
      return MaybeHeader.takeError();

    addUInt(H, MaybeHeader->Version);
    addUInt(H, static_cast<uint64_t>(MaybeHeader->CompressionFormat));
    addUInt(H, MaybeHeader->FileSize.value_or(0));
    addUInt(H, MaybeHeader->Hash);
    addUInt(H, MaybeHeader->UncompressedFileSize);

    Expected<size_t> NextOffset =
        nextCompressedBundleOffset(Remaining.size(), Remaining, *MaybeHeader);
    if (!NextOffset)
      return NextOffset.takeError();
    if (*NextOffset == StringRef::npos)
      break;
    Offset += *NextOffset;
  }
  return Error::success();
}

} // namespace COMGR
