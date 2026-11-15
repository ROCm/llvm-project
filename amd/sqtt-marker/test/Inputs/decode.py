#!/usr/bin/env python3

# ===- decode.py - Test-only SQTT marker decoder --------------------------=== #
#
# Part of AMD SQTT Marker, under the MIT License. See
# amd/sqtt-marker/LICENSE.txt for license information.
# SPDX-License-Identifier: MIT
#
# ===----------------------------------------------------------------------=== #

"""Small test-only decoder for SQTT funcmap rows and shaderdata values."""

import argparse


def read_funcmap(path):
    entries = {}
    payloads = {}
    clock_bits = 0
    with open(path, "rb") as file:
        text = file.read().rstrip(b"\0").decode()
    for row in text.splitlines():
        fields = row.split(":", 2)
        if row.startswith(("F:", "U:", "P:")):
            kind, marker_id, name = fields
            entries[int(marker_id)] = (kind, name.split("@", 1)[0])
        elif row.startswith("R:"):
            marker_id = int(fields[1])
            for item in fields[2].split(";"):
                key, separator, value = item.partition("=")
                if separator and key == "extra_payload_count":
                    payloads[marker_id] = int(value)
        elif row.startswith("M:"):
            for item in row[2:].split(";"):
                key, separator, value = item.partition("=")
                if separator and key == "shader_clock_bits":
                    clock_bits = int(value)
    return entries, payloads, clock_bits


def decode(path, values):
    entries, payloads, clock_bits = read_funcmap(path)
    id_mask = (1 << (30 - clock_bits)) - 1
    stack = []
    pending = None

    for raw in values:
        if pending:
            marker_id, index, count = pending
            kind, name = entries[marker_id]
            print(f"payload {kind}:{name}[{index}/{count}]=0x{raw:08x}")
            pending = None if index == count else (marker_id, index + 1, count)
            continue

        marker_id = (raw >> 2) & id_mask
        enter = bool(raw & 2)
        exit_previous = bool(raw & 1)
        if exit_previous:
            if not stack:
                raise ValueError("scope stack underflow")
            kind, name = stack.pop()
            print(f"exit {kind}:{name}")
        if enter:
            kind, name = entries[marker_id]
            stack.append((kind, name))
            print(f"enter {kind}:{name}")
        elif not exit_previous and marker_id:
            kind, name = entries[marker_id]
            print(f"point {kind}:{name}")

        count = payloads.get(marker_id, 0)
        if count:
            pending = (marker_id, 1, count)

    if pending:
        raise ValueError("incomplete payload block")
    if stack:
        raise ValueError("unclosed marker scope")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("funcmap")
    parser.add_argument("values", nargs="+", type=lambda value: int(value, 0))
    args = parser.parse_args()
    decode(args.funcmap, args.values)


if __name__ == "__main__":
    main()
