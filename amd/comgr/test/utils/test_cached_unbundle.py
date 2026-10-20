# Run the executable-consumer checks on a fresh cache and
# again on its entries. Each invocation destroys input/result containers before
# querying retained outputs, including ELF metadata and symbols.
import os
from pathlib import Path
import subprocess
import struct
import sys
import tempfile

with tempfile.TemporaryDirectory(prefix="comgr-unbundle-cache-") as cache:
    env = os.environ.copy()
    env.update(AMD_COMGR_CACHE="1", AMD_COMGR_CACHE_DIR=cache,
               AMD_COMGR_SAVE_TEMPS="0")
    subprocess.run([sys.argv[1], "--cached-executables"], env=env, check=True)
    assert list(Path(cache).glob("llvmcache-*")), "cache was not populated"
    subprocess.run(
        [sys.argv[1], "--cached-executables", "--verify-cache-mapping"],
        env=env,
        check=True,
    )

    entries = {path: path.read_bytes() for path in Path(cache).glob("llvmcache-*")}

    def frame(contents, log):
        return struct.pack("=Q", len(contents)) + contents + log

    def members(contents):
        assert contents[:4] == b"CUB1", "missing aligned framing"
        offset = 4
        while offset < len(contents):
            size, padding = struct.unpack_from("=II", contents, offset)
            offset += 8 + padding
            assert size <= len(contents) - offset, "truncated member"
            yield contents[offset:offset + size]
            offset += size
        assert offset == len(contents)

    def reframe(contents, aligned):
        result = b"CUB1" if aligned else b""
        for member in members(contents):
            result += struct.pack("=II" if aligned else "=I", len(member),
                                  *([0] if aligned else [])) + member
        return result

    mutations = {
        "legacy": lambda content: reframe(content, False),
        "unaligned": lambda content: reframe(content, True),
        "missing-size": lambda content: b"CUB1",
        "missing-padding": lambda content: b"CUB1" + struct.pack("=I", 1),
        "bad-padding": lambda content: b"CUB1" + struct.pack("=II", 1, 0xffffffff),
        "bad-size": lambda content: b"CUB1" + struct.pack("=II", 0xffffffff, 0),
        "truncated-member": lambda content: content[:-1],
        "extra-member": lambda content: content + b"\0",
    }
    for name, mutate in mutations.items():
        for path, original in entries.items():
            size, = struct.unpack_from("=Q", original)
            content, log = original[8:8 + size], original[8 + size:]
            path.write_bytes(frame(mutate(content), log))
        subprocess.run([sys.argv[1], "--cached-executables"], env=env, check=True)
        print(name, "PASS")

    for name, mutate in {
        "truncated-entry": lambda entry: entry[:4],
        "extra-entry-bytes": lambda entry: entry + b"\0",
    }.items():
        for path, original in entries.items():
            path.write_bytes(mutate(original))
        subprocess.run([sys.argv[1], "--cached-executables"], env=env, check=True)
        print(name, "PASS")
