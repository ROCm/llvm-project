# Keep valid textual IR while making its serialized length odd.
from pathlib import Path
import sys

path = Path(sys.argv[1])
data = path.read_bytes()
if len(data) % 2 == 0:
    path.write_bytes(data + b"\n")
