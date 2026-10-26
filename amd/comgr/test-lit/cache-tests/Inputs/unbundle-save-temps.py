import os
from pathlib import Path
import shutil
import subprocess
import sys

tool, bundle, work, compiler = sys.argv[1:]
work = Path(work).resolve()
if work.exists():
    shutil.rmtree(work)
work.mkdir()
env = os.environ.copy()
env["AMD_COMGR_CACHE_DIR"] = str(work / "cache")
env["AMD_COMGR_EMIT_VERBOSE_LOGS"] = "1"

# The driver records the host member as "host-" followed by the triple clang
# itself targets, so hard-coding a Linux triple would not match the member on
# Windows. Ask the compiler that produced the bundle for that triple. A
# mismatched host ID would otherwise still pass: Comgr allows missing bundles,
# and the missing member becomes an empty output.
triple = subprocess.run(
    [compiler, "-print-effective-triple"],
    check=True,
    stdout=subprocess.PIPE,
    text=True,
).stdout.strip()
assert triple, compiler
targets = ["host-" + triple, "hip-amdgcn-amd-amdhsa-unknown-gfx900"]
reference = None
for mode, cache, save in [
    ("disabled", "0", "1"),
    ("miss", "1", "1"),
    ("hit", "1", "1"),
    ("default-hit", "1", "0"),
    ("default-disabled", "0", "0"),
]:
    temps = work / mode
    temps.mkdir()
    env.update(
        TMPDIR=str(temps),
        TMP=str(temps),
        TEMP=str(temps),
        AMD_COMGR_CACHE=cache,
        AMD_COMGR_SAVE_TEMPS=save,
    )
    outputs = [work / (mode + "." + str(i) + ".bc") for i in range(2)]
    args = [tool, bundle]
    for target, output in zip(targets, outputs):
        args.extend([target, str(output)])
    subprocess.run(args, env=env, check=True)
    actual = [p.read_bytes() for p in outputs]
    assert actual[0] == b"" and actual[1], mode
    if reference is None:
        reference = actual
    assert actual == reference, mode
    if save == "1":
        saved = sorted(temps.glob("comgr-*/output/*"))
        assert len(saved) == len(targets), (mode, saved)
        for target, content in zip(targets, actual):
            matches = [p for p in saved if p.name.endswith("-" + target + ".bc")]
            assert len(matches) == 1, (mode, target, saved)
            assert matches[0].read_bytes() == content, (mode, target)
    else:
        assert not list(temps.glob("comgr-*")), mode
    if mode == "miss":
        assert list((work / "cache").glob("llvmcache-*")), "cache not populated"
