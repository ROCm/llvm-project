# ===- make-v2-offload-package.py ------------------------------------------===#
#
# Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
# amd/comgr/LICENSE.TXT in this repository for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===#
#
# Builds a single-entry llvm-offload-binary package using the version 2
# header layout (32 bytes: no InflatedSize field), so tests can exercise
# UnpackageCommand::canCache() against a pre-v3 package. llvm-offload-binary
# itself only ever emits the current header version, so this script
# hand-assembles the bytes instead.
#
# Usage: make-v2-offload-package.py <triple> <arch> <image-file> <output-file>
#
# ===----------------------------------------------------------------------===#

import struct
import sys


def align(n, a=8):
    return (n + a - 1) // a * a


def main():
    if len(sys.argv) != 5:
        print(
            "Usage: make-v2-offload-package.py <triple> <arch> <image-file> "
            "<output-file>"
        )
        return 1

    triple, arch, image_path, output_path = sys.argv[1:5]

    with open(image_path, "rb") as f:
        image = f.read()

    MAGIC = b"\x10\xff\x10\xad"
    IMG_BITCODE = 2  # llvm::object::ImageKind::IMG_Bitcode
    OFK_HIP = 1 << 2  # llvm::object::OffloadKind::OFK_HIP

    triple_key, triple_val = b"triple\x00", (triple + "\x00").encode()
    arch_key, arch_val = b"arch\x00", (arch + "\x00").encode()
    strtab = b"\x00" + triple_key + triple_val + arch_key + arch_val

    HEADER_V2_SIZE = 32  # Magic[4]+Version[4]+Size[8]+EntriesOffset[8]+EntriesCount[8]
    ENTRY_SIZE = 40  # ImageKind(u16)+OffloadKind(u16)+Flags(u32)+4*u64, packed to 8B
    STRINGENTRY_SIZE = 24  # KeyOffset,ValueOffset,ValueSize (v2+ StringEntry layout)

    entries_offset = HEADER_V2_SIZE
    string_entries_offset = entries_offset + ENTRY_SIZE
    string_entries_size = STRINGENTRY_SIZE * 2  # triple + arch
    strtab_offset = string_entries_offset + string_entries_size
    binary_data_size = align(strtab_offset + len(strtab))
    image_offset = binary_data_size
    total_size = align(binary_data_size + len(image))

    key_off_triple = strtab_offset + strtab.index(triple_key[:-1])
    val_off_triple = strtab_offset + strtab.index(triple_val[:-1])
    key_off_arch = strtab_offset + strtab.index(arch_key[:-1])
    val_off_arch = strtab_offset + strtab.index(arch_val[:-1])

    header = MAGIC + struct.pack("<IQQQ", 2, total_size, entries_offset, 1)
    assert len(header) == HEADER_V2_SIZE

    entry = struct.pack(
        "<HHIQQQQ",
        IMG_BITCODE,
        OFK_HIP,
        0,
        string_entries_offset,
        2,
        image_offset,
        len(image),
    )
    assert len(entry) == ENTRY_SIZE

    string_entries = struct.pack(
        "<QQQ", key_off_triple, val_off_triple, len(triple_val) - 1
    ) + struct.pack("<QQQ", key_off_arch, val_off_arch, len(arch_val) - 1)
    assert len(string_entries) == string_entries_size

    buf = bytearray(total_size)
    buf[0 : len(header)] = header
    buf[entries_offset : entries_offset + len(entry)] = entry
    buf[
        string_entries_offset : string_entries_offset + len(string_entries)
    ] = string_entries
    buf[strtab_offset : strtab_offset + len(strtab)] = strtab
    buf[image_offset : image_offset + len(image)] = image

    with open(output_path, "wb") as f:
        f.write(buf)

    return 0


if __name__ == "__main__":
    sys.exit(main())
