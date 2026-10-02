"""Probe original and copied Comgr DLLs after the original CI invocation."""

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

root = Path.cwd()
sys.path.insert(0, str(root / "llvm-project/llvm/utils/lit"))
from lit.TestingConfig import TestingConfig

config = TestingConfig.fromdefaults(SimpleNamespace(path=[], pass_env=[], useValgrind=False, maxIndividualTestTime=0))
original = root / "build-comgr/amd_comgr.dll"
snapshot = root / "diagnostics"
snapshot.mkdir(exist_ok=True)
shutil.copyfile(original, snapshot / original.name)
with original.open("rb") as source:
    print("DLL SHA256", hashlib.file_digest(source, "sha256").hexdigest(), flush=True)
for name in ["get-version.exe", "isa-enumeration.exe"]:
    shutil.copyfile(root / "build-comgr/test-lit" / name, snapshot / name)

for location in [root / "build-comgr/test-lit", snapshot]:
    for shell in ["native", "bash"]:
        env = dict(config.environment)
        env["PATH"] = os.pathsep.join([str(location), str(root / "build-comgr"), env["PATH"]])
        command = [str(location / "get-version.exe")]
        if shell == "bash":
            script = location / "probe.sh"
            script.write_text("get-version\n")
            command = [shutil.which("bash"), str(script)]
        results = []
        for i in range(3):
            result = subprocess.run(command, cwd=location, env=env, capture_output=True)
            results.append((result.returncode, result.stdout, result.stderr))
        print(str(location), shell, results, flush=True)

# Resolve without running initializers, so a bad DLL cannot stop diagnostics.
loader = r"""
import ctypes, sys
from ctypes import wintypes
kernel = ctypes.WinDLL('kernel32', use_last_error=True)
kernel.LoadLibraryExW.argtypes = [wintypes.LPCWSTR, wintypes.HANDLE, wintypes.DWORD]
kernel.LoadLibraryExW.restype = wintypes.HMODULE
kernel.GetModuleFileNameW.argtypes = [wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
module = kernel.LoadLibraryExW(sys.argv[1], None, 1)
if not module:
    raise ctypes.WinError(ctypes.get_last_error())
name = ctypes.create_unicode_buffer(32768)
if not kernel.GetModuleFileNameW(module, name, len(name)):
    raise ctypes.WinError(ctypes.get_last_error())
print('RESOLVED', name.value, 'BASE', hex(module))
"""
for path in ['amd_comgr.dll', str(original), str(snapshot / original.name)]:
    env = dict(config.environment)
    env['PATH'] = os.pathsep.join([str(root / 'build-comgr'), env['PATH']])
    result = subprocess.run([sys.executable, '-c', loader, path], cwd=root / 'build-comgr/test-lit', env=env, capture_output=True)
    print('LOAD RESOLUTION', path, result.returncode, result.stdout, result.stderr, flush=True)

# Recreate the file at its original pathname, then repeat every Comgr suite.
original.rename(snapshot / 'in-place.dll')
shutil.copyfile(snapshot / 'amd_comgr.dll', original)
result = subprocess.run(['ninja', '-C', 'build-comgr', 'check-comgr'])
print('CHECK-COMGR WITH RECREATED DLL', result.returncode, flush=True)
