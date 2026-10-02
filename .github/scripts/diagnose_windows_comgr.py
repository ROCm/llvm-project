"""Probe original and copied Comgr DLLs after the original CI invocation."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
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

# Use the same EXEs and original DLL bytes, placing one fresh copy beside them.
shutil.copyfile(original, root / "build-comgr/test-lit/amd_comgr.dll")
result = subprocess.run([sys.executable, str(root / "build/bin/llvm-lit.py"), "-sv", "--no-progress-bar", str(root / "build-comgr/test-lit")])
print("LIT WITH ADJACENT DLL", result.returncode, flush=True)
