"""Compare concurrent cold DLL loads under native and lit environments."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path("llvm-project/llvm/utils/lit").resolve()))
from lit.TestingConfig import TestingConfig

root = Path.cwd()
config = TestingConfig.fromdefaults(SimpleNamespace(path=[], pass_env=[], useValgrind=False, maxIndividualTestTime=0))
print("ENVIRONMENT DIFFERENCES", sorted(set(os.environ) ^ set(config.environment)))

for image in ["artifact", "rebuilt"]:
    for shell in ["native", "bash"]:
        for environment in ["inherited", "lit", "lit-no-compat"]:
            for count in [1, 96]:
                label = f"{image}-{shell}-{environment}-{count}"
                directory = root / "cold-probes" / label
                directory.mkdir(parents=True)
                shutil.copyfile(root / "diagnostics" / image / "amd_comgr.dll", directory / "amd_comgr.dll")
                shutil.copyfile(root / "diagnostics/rebuilt/get-version.exe", directory / "get-version.exe")
                env = dict(os.environ if environment == "inherited" else config.environment)
                if environment == "lit-no-compat":
                    env.pop("__COMPAT_LAYER", None)
                env["PATH"] = os.pathsep.join([str(directory), env["PATH"]])
                env["AMD_COMGR_CACHE"] = "0"
                command = [str(directory / "get-version.exe")]
                if shell == "bash":
                    script = directory / "probe.sh"
                    script.write_text("get-version\n")
                    command = [shutil.which("bash"), str(script)]
                def run(index: int) -> tuple[int, bytes, bytes]:
                    result = subprocess.run(command, cwd=directory, env=env, capture_output=True)
                    return result.returncode, result.stdout, result.stderr
                with ThreadPoolExecutor(max_workers=count) as pool:
                    results = list(pool.map(run, range(count)))
                print(label, "RESULTS", dict(Counter(result[0] for result in results)), flush=True)
                for result in list(dict.fromkeys(results))[:3]:
                    print("SAMPLE", result, flush=True)
                shutil.rmtree(directory)
