//===- comgr-unbundle-command.cpp - UnbundleCommand implementation --------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Adapts clang::OffloadBundler to COMGR's command cache and collects in-memory
/// outputs for COMGR data objects.
///
//===----------------------------------------------------------------------===//

#include <comgr-env.h>
#include <comgr-unbundle-command.h>

#include <clang/Driver/OffloadBundler.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/BinaryFormat/Magic.h>
#include <llvm/Object/OffloadBundle.h>
#include <llvm/Support/Alignment.h>
#include <llvm/Support/Endian.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/VirtualFileSystem.h>

#include <memory>

namespace COMGR {
using namespace llvm;
using namespace clang;

using SizeFieldType = uint32_t;

namespace {

constexpr const char CompressedBundleMagic[] = "CCOB";
constexpr size_t CompressedBundleMagicSize = sizeof(CompressedBundleMagic) - 1;
using CompressedBundleHeader =
    object::CompressedOffloadBundle::CompressedBundleHeader;

/// Return the header size for \p Version, or zero for an unsupported version.
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

/// Return the next bundle's offset relative to the start of \p Remaining.
/// RemainingSize is the total number of bytes left in the input. For v2/v3,
/// validate the recorded size against the header size and RemainingSize;
/// Remaining may then contain only the header prefix. For v1, Remaining must
/// contain the full remaining input: scan for the next magic, returning
/// StringRef::npos if this is the last bundle. Header must already be parsed.
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

// Persisting a very large decompressed bundle costs more than regenerating it.
// The cache entry has to frame and write the complete image, while the
// in-memory unbundler can decompress the same image directly into its output
// buffer. Keep the cache for ordinary bundles and bypass it for
// multi-hundred-megabyte code-object collections such as hipBLASLt's per-dtype
// libraries.
constexpr size_t MaxCachedUncompressedSize = 64 * 1024 * 1024;
constexpr size_t MinCompressedBundleHeaderSize = 8;

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
/// payload size and padding count, then zero padding and the payload. Align
/// relative to the enclosing cache file's size prefix, not Out's address.
/// Data.size() must fit in SizeFieldType.
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

/// A view into a decompressed buffer or mapped cache entry that retains its
/// owner. Sharing avoids copying each member but keeps the entire backing
/// buffer alive until its last view is released, even for a small member.
class SharedSubBuffer final : public MemoryBuffer {
public:
  /// Slice must lie within the non-null Owner; no trailing NUL is required.
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

/// Return storage aligned for COMGR's ELF consumers. Retain Owner when Slice
/// is max_align_t-aligned; otherwise make an aligned copy. A null Owner always
/// copies. If Owner is supplied, Slice must refer to storage that it owns.
std::unique_ptr<MemoryBuffer> shareOrCopy(std::shared_ptr<MemoryBuffer> Owner,
                                          StringRef Slice) {
  if (Owner && isAddrAligned(Align(alignof(std::max_align_t)), Slice.data()))
    return std::make_unique<SharedSubBuffer>(std::move(Owner), Slice);
  return MemoryBuffer::getMemBufferCopy(Slice);
}

} // namespace

bool UnbundleCommand::canCache() const {
  if (Kind == AMD_COMGR_DATA_KIND_AR_BUNDLE)
    return false;
  auto Buffer = FS.getBufferForFile(Config.InputFileNames.front(), -1, false);
  if (!Buffer)
    return false;
  StringRef Input = (*Buffer)->getBuffer();
  if (Input.empty())
    return false;
  size_t Offset = 0;
  while (Offset < Input.size()) {
    StringRef Remaining = Input.drop_front(Offset);
    if (Remaining.size() < MinCompressedBundleHeaderSize ||
        identify_magic(Remaining) != file_magic::offload_bundle_compressed)
      return false;
    auto Header = CompressedBundleHeader::tryParse(Remaining);
    if (!Header) {
      consumeError(Header.takeError());
      return false;
    }
    if (Header->UncompressedFileSize > MaxCachedUncompressedSize)
      return false;
    auto Next =
        nextCompressedBundleOffset(Remaining.size(), Remaining, *Header);
    if (!Next) {
      consumeError(Next.takeError());
      return false;
    }
    if (*Next == StringRef::npos)
      break;
    Offset += *Next;
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

Error UnbundleCommand::executeInMemory() {
  StringMap<std::shared_ptr<MemoryBuffer>> Outputs;
  OffloadBundler Bundler(Config);
  if (Error Err = Bundler.UnbundleFiles(
          &FS, [&](StringRef Name, std::unique_ptr<MemoryBuffer> Contents) {
            Outputs[Name] = std::move(Contents);
            return Error::success();
          }))
    return Err;
  // Multiple requested entries can name the same output. Retain its owner for
  // each data object and copy only if an ELF consumer's alignment requires it.
  for (StringRef Name : Config.OutputFileNames) {
    auto Owner = Outputs.lookup(Name);
    if (!Owner)
      return createStringError("missing unbundled output");
    OutputBuffers.push_back(shareOrCopy(Owner, Owner->getBuffer()));
  }
  return Error::success();
}

amd_comgr_status_t UnbundleCommand::execute(raw_ostream &LogS) {
  assert(Config.InputFileNames.size() == 1);
  OutputBuffers.clear();

  OffloadBundler Bundler(Config);

  switch (Kind) {
  case AMD_COMGR_DATA_KIND_BC_BUNDLE:
  case AMD_COMGR_DATA_KIND_OBJ_BUNDLE: {
    if (Error Err = executeInMemory()) {
      logAllUnhandledErrors(std::move(Err), LogS, "Unbundle Error: ");
      return AMD_COMGR_STATUS_ERROR;
    }
    if (Error Err = saveOutputTemps()) {
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
      FS.getBufferForFile(InputFilename, -1, false);
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
