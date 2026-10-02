"""Compare native and Bash loading of the artifact and rebuilt Comgr DLL."""

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys


def probe(command: list[str], env: dict[str, str]) -> None:
    print(f"PROBE {command!r}", flush=True)
    result = subprocess.run(command, env=env, capture_output=True)
    print(f"RETURN {result.returncode}", flush=True)
    print(f"STDOUT {result.stdout!r}", flush=True)
    print(f"STDERR {result.stderr!r}", flush=True)


phase = sys.argv[1]
root = Path.cwd()
dll = root / "build-comgr/amd_comgr.dll"
snapshot = root / "diagnostics" / phase
snapshot.mkdir(parents=True, exist_ok=True)
shutil.copy2(dll, snapshot / dll.name)
with dll.open("rb") as source:
    print(f"DLL {dll} SHA256 {hashlib.file_digest(source, 'sha256').hexdigest()}")
for tree in ["build", "build-comgr"]:
    for line in (root / tree / "CMakeCache.txt").read_text().splitlines():
        if line.startswith(("CMAKE_STRIP:", "CMAKE_CXX_COMPILER:", "CMAKE_CXX_FLAGS", "CMAKE_SHARED_LINKER_FLAGS", "LLVM_ENABLE_ASSERTIONS:", "LLVM_ENABLE_ABI_BREAKING_CHECKS:")):
            print(tree, line)

environment = dict(os.environ)
environment["PATH"] = os.pathsep.join([str(root / "build-comgr/test-lit"), str(root / "build-comgr"), str(root / "build/bin"), environment["PATH"]])
environment["AMD_COMGR_CACHE"] = "0"
environment["AMD_COMGR_REDIRECT_LOGS"] = "stderr"
load_script = "import ctypes,sys; d=ctypes.CDLL(sys.argv[1], winmode=0); a=ctypes.c_size_t(); b=ctypes.c_size_t(); d.amd_comgr_get_version(ctypes.byref(a),ctypes.byref(b)); print('VERSION',a.value,b.value)"
for path in [dll, snapshot / dll.name]:
    probe([sys.executable, "-c", load_script, str(path)], environment)

if phase == "rebuilt":
    for name in ["get-version", "isa-enumeration"]:
        executable = root / "build-comgr/test-lit" / (name + ".exe")
        shutil.copy2(executable, snapshot / executable.name)
        probe([str(executable)], environment)
        probe(["bash", "-c", name], environment)
        probe([str(snapshot / executable.name)], environment)
        probe(["bash", "-c", '"$1"', "probe", (snapshot / executable.name).as_posix()], environment)
    probe(["bash", "-c", "type -a get-version; type -a bash; command -v cygcheck"], environment)
    probe(["dumpbin", "/headers", str(dll)], environment)
