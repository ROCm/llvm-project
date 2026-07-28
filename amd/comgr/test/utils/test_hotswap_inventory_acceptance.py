#!/usr/bin/env python3
"""Optional end-to-end acceptance test for the frozen #3646 corpus."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


EXPECTED_MANIFEST_SHA256 = (
    "076ea013ba466139e76074fcbb31905a5041298e7644f6e8f590bf9353e4f5b9"
)
EXPECTED_ALIASES = 2685
EXPECTED_UNIQUE = 1452
EXPECTED_DUPLICATES = 1233
SCRIPT = (
    Path(__file__).resolve().parents[2] / "utils" / "hotswap" / "hotswap_inventory.py"
)


def main():
    corpus_root = os.environ.get("COMGR_HOTSWAP_CORPUS_ROOT")
    manifest = os.environ.get("COMGR_HOTSWAP_CORPUS_MANIFEST")
    if not corpus_root or not manifest:
        print(
            "SKIP: set COMGR_HOTSWAP_CORPUS_ROOT and "
            "COMGR_HOTSWAP_CORPUS_MANIFEST to run the #3646 corpus "
            "acceptance test",
            file=sys.stderr,
        )
        return 77

    with tempfile.TemporaryDirectory() as temporary_directory:
        report_path = Path(temporary_directory) / "inventory.json"
        worklist_path = Path(temporary_directory) / "unique.nul"
        command = [
            sys.executable,
            str(SCRIPT),
            corpus_root,
            "--manifest",
            manifest,
            "--json-output",
            str(report_path),
            "--worklist",
            str(worklist_path),
        ]
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0:
            sys.stderr.buffer.write(completed.stderr)
            return 1

        try:
            report = json.loads(report_path.read_text(encoding="ascii"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            print("cannot read inventory report: {}".format(error), file=sys.stderr)
            return 1

        if (
            report.get("schema") != "comgr.hotswap.inventory"
            or report.get("version") != 2
        ):
            print(
                "inventory report has an unexpected schema or version", file=sys.stderr
            )
            return 1
        expected_summary = {
            "files_examined": EXPECTED_ALIASES,
            "code_object_paths": EXPECTED_ALIASES,
            "unique_code_objects": EXPECTED_UNIQUE,
            "duplicate_paths": EXPECTED_DUPLICATES,
            "rejected_files": 0,
        }
        for key, expected in expected_summary.items():
            actual = report.get("summary", {}).get(key)
            if actual != expected:
                print(
                    "corpus invariant changed: {} is {}, expected {}".format(
                        key, actual, expected
                    ),
                    file=sys.stderr,
                )
                return 1
        if report.get("manifest", {}).get("sha256") != EXPECTED_MANIFEST_SHA256:
            print(
                "corpus manifest hash changed: got {}, expected {}".format(
                    report.get("manifest", {}).get("sha256"), EXPECTED_MANIFEST_SHA256
                ),
                file=sys.stderr,
            )
            return 1
        if report.get("rejected") != []:
            print("manifest contains rejected inputs", file=sys.stderr)
            return 1

        try:
            worklist_entries = worklist_path.read_bytes().split(b"\0")
        except OSError as error:
            print("cannot read unique worklist: {}".format(error), file=sys.stderr)
            return 1
        if not worklist_entries or worklist_entries[-1] != b"":
            print("unique worklist is not NUL terminated", file=sys.stderr)
            return 1
        if len(worklist_entries) - 1 != EXPECTED_UNIQUE:
            print(
                "unique worklist has {} entries, expected {}".format(
                    len(worklist_entries) - 1, EXPECTED_UNIQUE
                ),
                file=sys.stderr,
            )
            return 1

    print(
        "PASS: #3646 corpus has {} aliases, {} unique objects, and {} "
        "duplicates".format(EXPECTED_ALIASES, EXPECTED_UNIQUE, EXPECTED_DUPLICATES)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
