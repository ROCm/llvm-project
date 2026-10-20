# Run the executable-consumer checks on a fresh cache and
# again on its entries. Each invocation destroys input/result containers before
# querying retained outputs, including ELF metadata and symbols.
import os
from pathlib import Path
import subprocess
import sys
import tempfile

with tempfile.TemporaryDirectory(prefix="comgr-unbundle-cache-") as cache:
    env = os.environ.copy()
    env.update(AMD_COMGR_CACHE="1", AMD_COMGR_CACHE_DIR=cache,
               AMD_COMGR_SAVE_TEMPS="0")
    subprocess.run([sys.argv[1], "--cached-executables"], env=env, check=True)
    assert list(Path(cache).glob("llvmcache-*")), "cache was not populated"
    subprocess.run([sys.argv[1], "--cached-executables"], env=env, check=True)
