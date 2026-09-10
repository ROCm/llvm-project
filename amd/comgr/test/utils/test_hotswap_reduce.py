#!/usr/bin/env python3
"""Hermetic tests for utils/hotswap/hotswap_reduce.py."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import stat
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from typing import Optional, Union
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[2] / "utils" / "hotswap" / "hotswap_reduce.py"
SPEC = importlib.util.spec_from_file_location("hotswap_reduce", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
hotswap_reduce = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = hotswap_reduce
SPEC.loader.exec_module(hotswap_reduce)


def write_executable(path: Path, body: str) -> Path:
    path.write_text(
        "#!/usr/bin/env python3\n" + textwrap.dedent(body),
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def write_bundle(
    root: Path,
    object_contents: list[bytes],
    metadata: object,
    names: Optional[list[str]] = None,
) -> Path:
    objects = []
    names = names or [f"object-{index}.co" for index in range(len(object_contents))]
    for index, (name, content) in enumerate(zip(names, object_contents)):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        objects.append({"id": f"id-{index}", "path": name})
    bundle = {
        "format": hotswap_reduce.BUNDLE_FORMAT,
        "version": hotswap_reduce.BUNDLE_VERSION,
        "code_objects": objects,
        "metadata": metadata,
    }
    path = root / "bundle.json"
    path.write_text(json.dumps(bundle), encoding="utf-8")
    return path


def make_args(
    bundle: Path,
    output: Path,
    predicate: Union[Path, str],
    predicate_args: list[str],
    **overrides: object,
) -> argparse.Namespace:
    values: dict[str, object] = {
        "bundle": bundle,
        "code_object": None,
        "worklist": None,
        "metadata": None,
        "output": output,
        "predicate": str(predicate),
        "predicate_arg": predicate_args,
        "interesting_exit_code": 0,
        "predicate_runs": 1,
        "timeout": 5.0,
        "cache_file": None,
        "cache_dependency": [],
        "cache_tag": [],
        "allow_empty_bundle": False,
        "allow_remove_section": [],
        "protect_section": [],
        "readobj": "llvm-readobj",
        "objcopy": "llvm-objcopy",
        "tool_timeout": 5.0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class DdminTests(unittest.TestCase):
    def test_finds_one_minimal_required_pair(self) -> None:
        attempts = []

        def interesting(kept: list[str], removed: list[str]) -> bool:
            attempts.append((tuple(kept), tuple(removed)))
            return "b" in kept and "e" in kept

        result = hotswap_reduce.ddmin(list("abcdef"), interesting)
        self.assertEqual(result, ["b", "e"])
        self.assertTrue(attempts)
        for index in range(len(result)):
            candidate = result[:index] + result[index + 1 :]
            self.assertFalse("b" in candidate and "e" in candidate)

    def test_respects_minimum_size(self) -> None:
        result = hotswap_reduce.ddmin(
            [1, 2, 3],
            lambda kept, removed: True,
            minimum_size=1,
        )
        self.assertEqual(len(result), 1)

    def test_duplicate_items_are_reduced_by_position(self) -> None:
        result = hotswap_reduce.ddmin(
            ["x", "x", "y"],
            lambda kept, removed: kept.count("x") >= 1,
        )
        self.assertEqual(result, ["x"])

    def test_exhaustive_monotone_predicates_are_one_minimal(self) -> None:
        for size in range(7):
            items = list(range(size))
            for required_bits in range(1 << size):
                required = {item for item in items if required_bits & (1 << item)}
                result = hotswap_reduce.ddmin(
                    items,
                    lambda kept, removed, required=required: required.issubset(kept),
                )
                self.assertEqual(set(result), required)
                for index in range(len(result)):
                    self.assertFalse(
                        required.issubset(result[:index] + result[index + 1 :])
                    )


class BundleValidationTests(unittest.TestCase):
    def test_rejects_malformed_metadata_list(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            bundle = write_bundle(root, [b"object"], {"cases": {}})
            with self.assertRaisesRegex(
                hotswap_reduce.ReducerError,
                r"metadata\.cases must be a JSON list",
            ):
                hotswap_reduce.load_bundle(bundle)

    def test_rejects_duplicate_ids(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "a").write_bytes(b"a")
            (root / "b").write_bytes(b"b")
            bundle = {
                "format": hotswap_reduce.BUNDLE_FORMAT,
                "version": 1,
                "code_objects": [
                    {"id": "same", "path": "a"},
                    {"id": "same", "path": "b"},
                ],
            }
            path = root / "bundle.json"
            path.write_text(json.dumps(bundle), encoding="utf-8")
            with self.assertRaisesRegex(
                hotswap_reduce.ReducerError, "duplicate code object id"
            ):
                hotswap_reduce.load_bundle(path)

    def test_bundle_paths_cannot_escape_bundle_directory(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            bundle_root = root / "bundle"
            bundle_root.mkdir()
            outside = root / "outside.co"
            outside.write_bytes(b"outside")
            bundle = {
                "format": hotswap_reduce.BUNDLE_FORMAT,
                "version": 1,
                "code_objects": [{"id": "outside", "path": "../outside.co"}],
            }
            bundle_path = bundle_root / "bundle.json"
            bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
            with self.assertRaisesRegex(hotswap_reduce.ReducerError, "escapes"):
                hotswap_reduce.load_bundle(bundle_path)
            bundle["code_objects"][0]["path"] = str(outside)
            bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
            with self.assertRaisesRegex(
                hotswap_reduce.ReducerError, "must be relative"
            ):
                hotswap_reduce.load_bundle(bundle_path)

    def test_bundle_symlinks_cannot_escape_bundle_directory(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            bundle_root = root / "bundle"
            bundle_root.mkdir()
            outside = root / "outside.co"
            outside.write_bytes(b"outside")
            link = bundle_root / "linked.co"
            try:
                link.symlink_to(outside)
            except OSError as error:
                self.skipTest(f"symlinks unavailable: {error}")
            bundle = {
                "format": hotswap_reduce.BUNDLE_FORMAT,
                "version": 1,
                "code_objects": [{"id": "outside", "path": link.name}],
            }
            bundle_path = bundle_root / "bundle.json"
            bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
            with self.assertRaisesRegex(hotswap_reduce.ReducerError, "escapes"):
                hotswap_reduce.load_bundle(bundle_path)

    def test_rejects_non_finite_json(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "bundle.json"
            path.write_text('{"value": NaN}', encoding="utf-8")
            with self.assertRaisesRegex(hotswap_reduce.ReducerError, "non-finite"):
                hotswap_reduce.load_json(path)

    def test_accepts_spaces_and_special_characters_in_paths(self) -> None:
        with tempfile.TemporaryDirectory(prefix="reduce spaces [") as name:
            root = Path(name)
            bundle = write_bundle(
                root,
                [b"one"],
                {},
                names=["objects/a file [1] $x.co"],
            )
            candidate = hotswap_reduce.load_bundle(bundle)
            self.assertEqual(candidate.code_objects[0].path.read_bytes(), b"one")
            with tempfile.TemporaryDirectory() as output_name:
                output = Path(output_name) / "candidate"
                hotswap_reduce.materialize_candidate(candidate, output)
                output_bundle = json.loads(
                    (output / "bundle.json").read_text(encoding="utf-8")
                )
                relative = output_bundle["code_objects"][0]["path"]
                self.assertEqual((output / relative).read_bytes(), b"one")

    def test_materialization_rejects_changed_source_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            bundle = write_bundle(root, [b"original"], {})
            candidate = hotswap_reduce.load_bundle(bundle)
            candidate.code_objects[0].path.write_bytes(b"changed")
            with self.assertRaisesRegex(
                hotswap_reduce.ReducerError, "changed while materializing"
            ):
                hotswap_reduce.materialize_candidate(candidate, root / "candidate")

    def test_loads_inventory_nul_worklist_and_selector_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            first = root / "object with spaces.co"
            # Windows rejects control characters in file names.  Preserve the
            # newline-path coverage where the host permits it while still
            # exercising NUL-delimited parsing on every platform.
            second_name = (
                "object;with-semicolon.co"
                if os.name == "nt"
                else "object\nwith-newline.co"
            )
            second = root / second_name
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            worklist = root / "selected.list"
            worklist.write_bytes(
                os.fsencode(first.resolve())
                + b"\0"
                + os.fsencode(second.resolve())
                + b"\0"
            )
            selection = root / "selection.json"
            selection.write_text(
                json.dumps(
                    {
                        "kind": "producer-selection",
                        "schema_version": 1,
                        "selected_tests": [
                            {"id": "test-a"},
                            {"id": "test-b"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            candidate = hotswap_reduce.load_inputs(None, [], worklist, selection)
            self.assertEqual(
                [item.path for item in candidate.code_objects],
                [first.resolve(), second.resolve()],
            )
            self.assertEqual(
                candidate.metadata["selected_tests"],
                [{"id": "test-a"}, {"id": "test-b"}],
            )

    def test_rejects_malformed_and_duplicate_nul_worklists(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            object_path = root / "object.co"
            object_path.write_bytes(b"object")
            worklist = root / "worklist"
            worklist.write_bytes(os.fsencode(object_path.resolve()))
            with self.assertRaisesRegex(
                hotswap_reduce.ReducerError, "must end with NUL"
            ):
                hotswap_reduce.load_nul_worklist(worklist)
            encoded_path = os.fsencode(object_path.resolve())
            worklist.write_bytes(encoded_path + b"\0" + encoded_path + b"\0")
            with self.assertRaisesRegex(
                hotswap_reduce.ReducerError, "duplicate worklist path"
            ):
                hotswap_reduce.load_nul_worklist(worklist)


class PredicateTests(unittest.TestCase):
    def make_candidate(self, root: Path, value: bytes = b"interesting"):
        path = root / "input.co"
        path.write_bytes(value)
        return hotswap_reduce.Candidate(
            (
                hotswap_reduce.CodeObject(
                    "id",
                    path.name,
                    path,
                    hotswap_reduce.sha256_file(path),
                ),
            ),
            {},
        )

    def make_runner(
        self,
        root: Path,
        script: Path,
        arguments: list[str],
        **overrides: object,
    ):
        values = {
            "argv_template": [sys.executable, str(script)] + arguments,
            "interesting_exit_code": 0,
            "runs": 1,
            "timeout": 5.0,
            "work_root": root,
            "cache": hotswap_reduce.PredicateCache(None),
        }
        values.update(overrides)
        return hotswap_reduce.PredicateRunner(**values)

    def test_cache_uses_candidate_content(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            counter = root / "count"
            script = root / "predicate.py"
            script.write_text(
                textwrap.dedent(
                    f"""
                    import pathlib
                    counter = pathlib.Path({str(counter)!r})
                    count = int(counter.read_text()) if counter.exists() else 0
                    counter.write_text(str(count + 1))
                    raise SystemExit(0)
                    """
                ),
                encoding="utf-8",
            )
            runner = self.make_runner(
                root,
                script,
                ["{input}"],
            )
            candidate = self.make_candidate(root)
            first = runner.evaluate(candidate)
            second = runner.evaluate(candidate)
            self.assertTrue(first.interesting)
            self.assertFalse(first.cached)
            self.assertTrue(second.cached)
            self.assertEqual(counter.read_text(encoding="utf-8"), "1")

            changed = self.make_candidate(root, b"changed")
            runner.evaluate(changed)
            self.assertEqual(counter.read_text(encoding="utf-8"), "2")

    def test_persistent_cache_survives_runner_recreation(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            counter = root / "count"
            cache_path = root / "predicate-cache.json"
            script = root / "predicate.py"
            script.write_text(
                textwrap.dedent(
                    f"""
                    import pathlib
                    counter = pathlib.Path({str(counter)!r})
                    count = int(counter.read_text()) if counter.exists() else 0
                    counter.write_text(str(count + 1))
                    raise SystemExit(0)
                    """
                ),
                encoding="utf-8",
            )
            candidate = self.make_candidate(root)
            first = self.make_runner(
                root,
                script,
                ["{input}"],
                cache=hotswap_reduce.PredicateCache(cache_path),
            )
            self.assertFalse(first.evaluate(candidate).cached)
            second = self.make_runner(
                root,
                script,
                ["{input}"],
                cache=hotswap_reduce.PredicateCache(cache_path),
            )
            self.assertTrue(second.evaluate(candidate).cached)
            self.assertEqual(counter.read_text(encoding="utf-8"), "1")
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
            self.assertEqual(cache["format"], hotswap_reduce.CACHE_FORMAT)
            self.assertEqual(cache["version"], 1)

    def test_cached_and_uncached_results_are_equivalent(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            cache_path = root / "predicate-cache.json"
            script = root / "predicate.py"
            script.write_text(
                "print('stable output')\n"
                "print('stable error', file=__import__('sys').stderr)\n"
                "raise SystemExit(7)\n",
                encoding="utf-8",
            )
            candidate = self.make_candidate(root)
            runner = self.make_runner(
                root,
                script,
                ["{input}"],
                interesting_exit_code=7,
                runs=2,
                cache=hotswap_reduce.PredicateCache(cache_path),
            )
            uncached = runner.evaluate(candidate)
            cached = runner.evaluate(candidate)
            self.assertFalse(uncached.cached)
            self.assertTrue(cached.cached)
            self.assertEqual(uncached.status, cached.status)
            self.assertEqual(uncached.exit_codes, cached.exit_codes)
            self.assertEqual(uncached.stdout, cached.stdout)
            self.assertEqual(uncached.stderr, cached.stderr)
            self.assertEqual(uncached.for_log(), cached.for_log())
            self.assertNotIn("cached", cached.for_log())

    def test_corrupt_matching_cache_entry_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            cache_path = root / "predicate-cache.json"
            script = root / "predicate.py"
            script.write_text("raise SystemExit(0)\n", encoding="utf-8")
            candidate = self.make_candidate(root)
            runner = self.make_runner(
                root,
                script,
                ["{input}"],
                cache=hotswap_reduce.PredicateCache(cache_path),
            )
            runner.evaluate(candidate)
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
            entry = next(iter(cache["entries"].values()))
            entry["exit_codes"] = []
            cache_path.write_text(json.dumps(cache), encoding="utf-8")
            recreated = self.make_runner(
                root,
                script,
                ["{input}"],
                cache=hotswap_reduce.PredicateCache(cache_path),
            )
            with self.assertRaisesRegex(hotswap_reduce.ReducerError, "corrupt entry"):
                recreated.evaluate(candidate)

    def test_persistent_cache_invalidates_changed_predicate_script(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            counter = root / "count"
            cache_path = root / "predicate-cache.json"
            script = root / "predicate.py"

            def write_predicate(marker: str) -> None:
                script.write_text(
                    textwrap.dedent(
                        f"""
                        # identity: {marker}
                        import pathlib
                        counter = pathlib.Path({str(counter)!r})
                        count = int(counter.read_text()) if counter.exists() else 0
                        counter.write_text(str(count + 1))
                        raise SystemExit(0)
                        """
                    ),
                    encoding="utf-8",
                )

            candidate = self.make_candidate(root)
            write_predicate("first")
            first = self.make_runner(
                root,
                script,
                ["{input}"],
                cache=hotswap_reduce.PredicateCache(cache_path),
            )
            self.assertFalse(first.evaluate(candidate).cached)
            write_predicate("second")
            second = self.make_runner(
                root,
                script,
                ["{input}"],
                cache=hotswap_reduce.PredicateCache(cache_path),
            )
            self.assertFalse(second.evaluate(candidate).cached)
            self.assertEqual(counter.read_text(encoding="utf-8"), "2")

    def test_persistent_cache_invalidates_dependency_and_tag(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            counter = root / "count"
            cache_path = root / "predicate-cache.json"
            dependency = root / "libamd_comgr.so"
            dependency.write_bytes(b"build-one")
            script = root / "predicate.py"
            script.write_text(
                textwrap.dedent(
                    f"""
                    import pathlib
                    counter = pathlib.Path({str(counter)!r})
                    count = int(counter.read_text()) if counter.exists() else 0
                    counter.write_text(str(count + 1))
                    raise SystemExit(0)
                    """
                ),
                encoding="utf-8",
            )
            candidate = self.make_candidate(root)

            def make(cache_tag: str):
                return self.make_runner(
                    root,
                    script,
                    ["{input}"],
                    cache=hotswap_reduce.PredicateCache(cache_path),
                    cache_dependencies=[dependency],
                    cache_tags=[cache_tag],
                )

            self.assertFalse(make("environment-one").evaluate(candidate).cached)
            self.assertTrue(make("environment-one").evaluate(candidate).cached)
            dependency.write_bytes(b"build-two")
            self.assertFalse(make("environment-one").evaluate(candidate).cached)
            self.assertFalse(make("environment-two").evaluate(candidate).cached)
            self.assertEqual(counter.read_text(encoding="utf-8"), "3")

    def test_changed_identity_during_run_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            dependency = root / "dependency"
            dependency.write_bytes(b"one")
            script = root / "predicate.py"
            script.write_text("raise SystemExit(0)\n", encoding="utf-8")
            runner = self.make_runner(
                root,
                script,
                ["{input}"],
                cache_dependencies=[dependency],
            )
            dependency.write_bytes(b"two")
            with self.assertRaisesRegex(
                hotswap_reduce.ReducerError, "identity changed"
            ):
                runner.evaluate(self.make_candidate(root))

    def test_changed_identity_mode_invalidates_cached_result(self) -> None:
        if os.name == "nt":
            self.skipTest("POSIX executable-mode semantics")
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            executable = write_executable(
                root / "predicate",
                """
                raise SystemExit(0)
                """,
            )
            candidate = self.make_candidate(root)
            runner = hotswap_reduce.PredicateRunner(
                [str(executable), "{input}"],
                0,
                1,
                5.0,
                root,
                hotswap_reduce.PredicateCache(None),
            )
            self.assertFalse(runner.evaluate(candidate).cached)
            executable.chmod(executable.stat().st_mode & ~stat.S_IXUSR)
            with self.assertRaisesRegex(
                hotswap_reduce.ReducerError, "identity changed"
            ):
                runner.evaluate(candidate)

    def test_candidate_digest_does_not_reread_large_objects(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            candidate = self.make_candidate(root)
            with mock.patch.object(
                hotswap_reduce,
                "sha256_file",
                side_effect=AssertionError("unexpected object reread"),
            ):
                digest = candidate.digest()
            self.assertRegex(digest, r"^[0-9a-f]{64}$")

    def test_nonzero_exit_can_mean_interesting(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            script = root / "predicate.py"
            script.write_text("raise SystemExit(17)\n", encoding="utf-8")
            runner = self.make_runner(
                root,
                script,
                ["{input}"],
                interesting_exit_code=17,
            )
            result = runner.evaluate(self.make_candidate(root))
            self.assertEqual(result.status, "interesting")
            self.assertEqual(result.exit_codes, (17,))

    def test_timeout_is_rejected_and_not_cached(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            script = root / "predicate.py"
            script.write_text("import time\ntime.sleep(10)\n", encoding="utf-8")
            runner = self.make_runner(
                root,
                script,
                [],
                timeout=0.02,
            )
            candidate = self.make_candidate(root)
            first = runner.evaluate(candidate)
            second = runner.evaluate(candidate)
            self.assertEqual(first.status, "timeout")
            self.assertEqual(second.status, "timeout")
            self.assertFalse(second.cached)

    def test_flaky_predicate_is_rejected_and_not_cached(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            counter = root / "count"
            script = root / "predicate.py"
            script.write_text(
                textwrap.dedent(
                    """
                    import pathlib
                    import sys
                    path = pathlib.Path(sys.argv[1])
                    count = int(path.read_text()) if path.exists() else 0
                    path.write_text(str(count + 1))
                    raise SystemExit(count % 2)
                    """
                ),
                encoding="utf-8",
            )
            runner = self.make_runner(
                root,
                script,
                [str(counter)],
                runs=2,
            )
            result = runner.evaluate(self.make_candidate(root))
            self.assertEqual(result.status, "flaky")
            self.assertFalse(result.cached)

    def test_different_noninteresting_exit_codes_are_flaky(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            counter = root / "count"
            script = root / "predicate.py"
            script.write_text(
                textwrap.dedent(
                    """
                    import pathlib
                    import sys
                    path = pathlib.Path(sys.argv[1])
                    count = int(path.read_text()) if path.exists() else 0
                    path.write_text(str(count + 1))
                    raise SystemExit(1 + count % 2)
                    """
                ),
                encoding="utf-8",
            )
            runner = self.make_runner(root, script, [str(counter)], runs=2)
            result = runner.evaluate(self.make_candidate(root))
            self.assertEqual(result.status, "flaky")
            self.assertEqual(result.exit_codes, (1, 2))
            self.assertFalse(result.cached)

    def test_timeout_terminates_predicate_descendants(self) -> None:
        if os.name == "nt":
            self.skipTest("POSIX process-group behavior")
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            marker = root / "descendant-ran"
            child_code = (
                "import pathlib,time;"
                "time.sleep(0.3);"
                f"pathlib.Path({str(marker)!r}).write_text('leaked')"
            )
            script = root / "predicate.py"
            script.write_text(
                "import subprocess,sys,time\n"
                f"subprocess.Popen([sys.executable, '-c', {child_code!r}])\n"
                "time.sleep(10)\n",
                encoding="utf-8",
            )
            runner = self.make_runner(root, script, [], timeout=0.05)
            result = runner.evaluate(self.make_candidate(root))
            self.assertEqual(result.status, "timeout")
            time.sleep(0.5)
            self.assertFalse(marker.exists())

    def test_completed_predicate_does_not_leave_descendants(self) -> None:
        if os.name == "nt":
            self.skipTest("POSIX process-group behavior")
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            marker = root / "descendant-ran"
            child_code = (
                "import pathlib,time;"
                "time.sleep(0.3);"
                f"pathlib.Path({str(marker)!r}).write_text('leaked')"
            )
            script = root / "predicate.py"
            script.write_text(
                "import subprocess,sys\n"
                f"subprocess.Popen([sys.executable, '-c', {child_code!r}])\n"
                "raise SystemExit(0)\n",
                encoding="utf-8",
            )
            runner = self.make_runner(root, script, [])
            result = runner.evaluate(self.make_candidate(root))
            self.assertTrue(result.interesting)
            time.sleep(0.5)
            self.assertFalse(marker.exists())

    def test_capture_is_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            script = root / "predicate.py"
            script.write_text(
                f"print('x' * {hotswap_reduce.CAPTURE_LIMIT * 100})\n"
                "raise SystemExit(0)\n",
                encoding="utf-8",
            )
            runner = self.make_runner(root, script, [])
            result = runner.evaluate(self.make_candidate(root))
            self.assertTrue(result.interesting)
            self.assertLessEqual(
                len(result.stdout),
                hotswap_reduce.CAPTURE_LIMIT + len("\n<truncated>"),
            )
            self.assertTrue(result.stdout.endswith("\n<truncated>"))

    def test_unknown_placeholder_fails_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            with self.assertRaisesRegex(
                hotswap_reduce.ReducerError, "unknown.*placeholder"
            ):
                hotswap_reduce.PredicateRunner(
                    [sys.executable, "{unknown}"],
                    0,
                    1,
                    1.0,
                    root,
                    hotswap_reduce.PredicateCache(None),
                )

    def test_regular_file_argv_is_resolved_before_workspace_change(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            script = root / "predicate.py"
            script.write_text("raise SystemExit(0)\n", encoding="utf-8")
            original_directory = Path.cwd()
            try:
                os.chdir(root)
                runner = hotswap_reduce.PredicateRunner(
                    [sys.executable, "predicate.py", "{input}"],
                    0,
                    1,
                    5.0,
                    root,
                    hotswap_reduce.PredicateCache(None),
                )
            finally:
                os.chdir(original_directory)
            self.assertEqual(runner.argv_template[1], str(script.resolve()))
            self.assertTrue(runner.evaluate(self.make_candidate(root)).interesting)

    def test_input_placeholder_rejects_multi_object_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            one = self.make_candidate(root, b"one")
            second_path = root / "second.co"
            second_path.write_bytes(b"two")
            two = hotswap_reduce.Candidate(
                one.code_objects
                + (
                    hotswap_reduce.CodeObject(
                        "second",
                        second_path.name,
                        second_path,
                        hotswap_reduce.sha256_file(second_path),
                    ),
                ),
                {},
            )
            script = root / "predicate.py"
            script.write_text("raise SystemExit(0)\n", encoding="utf-8")
            runner = self.make_runner(root, script, ["{input}"])
            with self.assertRaisesRegex(hotswap_reduce.ReducerError, "exactly one"):
                runner.evaluate(two)

    def test_nonfinite_timeouts_are_rejected_by_argument_parser(self) -> None:
        parser = hotswap_reduce.make_argument_parser()
        base_arguments = [
            "--code-object",
            "input.co",
            "--output",
            "output",
            "--predicate",
            "predicate",
        ]
        for option in ("--timeout", "--tool-timeout"):
            for value in ("nan", "inf", "-inf", "0", "-1"):
                with mock.patch("sys.stderr"):
                    with self.assertRaises(SystemExit):
                        parser.parse_args(base_arguments + [option, value])


class FakeElfToolTests(unittest.TestCase):
    def write_readobj(
        self,
        root: Path,
        sections: list[tuple[str, bool]],
        architecture: str = "amdgcn",
    ) -> Path:
        return write_executable(
            root / "fake readobj",
            f"""
            import json
            import pathlib
            import sys
            sections = {sections!r}
            contents = pathlib.Path(sys.argv[-1]).read_bytes().decode(
                "utf-8", errors="ignore"
            )
            removed = {{
                word.split("=", 1)[1]
                for word in contents.split()
                if word.startswith("--remove-section=")
            }}
            value = [{{
                "FileSummary": {{
                    "Format": "elf64-amdgpu",
                    "Arch": {architecture!r},
                }},
                "Sections": [
                    {{
                        "Section": {{
                            "Name": {{"Name": name}},
                            "Flags": {{
                                "Flags": ([{{"Name": "SHF_ALLOC"}}] if allocated else [])
                            }},
                        }}
                    }}
                    for name, allocated in sections
                    if name not in removed
                ],
            }}]
            print(json.dumps(value))
            """,
        )

    def write_objcopy(
        self,
        root: Path,
        fail: bool = False,
        annotate: bool = False,
        extra_removal: Optional[str] = None,
    ) -> Path:
        return write_executable(
            root / "fake objcopy",
            f"""
            import pathlib
            import shutil
            import sys
            if {fail!r}:
                print("intentional objcopy failure", file=sys.stderr)
                raise SystemExit(4)
            shutil.copyfile(pathlib.Path(sys.argv[-2]), pathlib.Path(sys.argv[-1]))
            if {annotate!r}:
                with pathlib.Path(sys.argv[-1]).open("ab") as stream:
                    stream.write((" " + " ".join(sys.argv[1:-2])).encode())
            if {extra_removal!r} is not None:
                with pathlib.Path(sys.argv[-1]).open("ab") as stream:
                    stream.write((" --remove-section=" + {extra_removal!r}).encode())
            """,
        )

    def test_policy_only_allows_nonallocated_unprotected_sections(self) -> None:
        if os.name == "nt":
            self.skipTest("executable script fixtures require a POSIX host")
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            readobj = self.write_readobj(
                root,
                [
                    ("", False),
                    (".text", True),
                    (".note", False),
                    (".AMDGPU.config", False),
                    (".rela.debug_info", False),
                    (".group", False),
                    (".debug_info", False),
                    (".comment", False),
                    (".allocated_debug", True),
                ],
            )
            objcopy = self.write_objcopy(root)
            tools = hotswap_reduce.ElfSectionTools(
                str(readobj),
                str(objcopy),
                5.0,
                ["*"],
                [".comment"],
                root / "artifacts",
            )
            tools.artifact_root.mkdir()
            obj = root / "object.co"
            obj.write_bytes(b"ELF")
            self.assertEqual(tools.removable_sections(obj), [".debug_info"])

    def test_duplicate_name_is_protected_if_any_instance_is_allocated(self) -> None:
        if os.name == "nt":
            self.skipTest("executable script fixtures require a POSIX host")
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            readobj = self.write_readobj(
                root,
                [
                    (".debug_duplicate", False),
                    (".debug_duplicate", True),
                    (".debug_safe", False),
                ],
            )
            objcopy = self.write_objcopy(root)
            tools = hotswap_reduce.ElfSectionTools(
                str(readobj),
                str(objcopy),
                5.0,
                [".debug_*"],
                [],
                root / "artifacts",
            )
            tools.artifact_root.mkdir()
            obj = root / "object.co"
            obj.write_bytes(b"ELF")
            self.assertEqual(tools.removable_sections(obj), [".debug_safe"])

    def test_missing_tool_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            with self.assertRaisesRegex(
                hotswap_reduce.ReducerError, "could not find executable"
            ):
                hotswap_reduce.ElfSectionTools(
                    str(root / "does-not-exist"),
                    str(root / "also-missing"),
                    1.0,
                    ["*"],
                    [],
                    root,
                )

    def test_rejects_non_amdgpu_elf(self) -> None:
        if os.name == "nt":
            self.skipTest("executable script fixtures require a POSIX host")
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            readobj = self.write_readobj(root, [(".debug_info", False)], "x86_64")
            objcopy = self.write_objcopy(root)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            tools = hotswap_reduce.ElfSectionTools(
                str(readobj),
                str(objcopy),
                5.0,
                ["*"],
                [],
                artifacts,
            )
            obj = root / "object.co"
            obj.write_bytes(b"ELF")
            with self.assertRaisesRegex(
                hotswap_reduce.ReducerError, "requires an AMDGPU ELF"
            ):
                tools.removable_sections(obj)

    def test_objcopy_failure_does_not_replace_input(self) -> None:
        if os.name == "nt":
            self.skipTest("executable script fixtures require a POSIX host")
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            readobj = self.write_readobj(root, [(".debug_info", False)])
            objcopy = self.write_objcopy(root, fail=True)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            tools = hotswap_reduce.ElfSectionTools(
                str(readobj),
                str(objcopy),
                5.0,
                [".debug_*"],
                [],
                artifacts,
            )
            path = root / "object.co"
            path.write_bytes(b"original")
            code_object = hotswap_reduce.CodeObject(
                "id", path.name, path, hotswap_reduce.sha256_file(path)
            )
            with self.assertRaisesRegex(
                hotswap_reduce.ReducerError, "intentional objcopy failure"
            ):
                tools.remove_sections(code_object, [".debug_info"])
            self.assertEqual(path.read_bytes(), b"original")
            self.assertEqual(list(artifacts.iterdir()), [])

    def test_malformed_readobj_json_is_reported(self) -> None:
        if os.name == "nt":
            self.skipTest("executable script fixtures require a POSIX host")
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            readobj = write_executable(root / "fake readobj", "print('not json')\n")
            objcopy = self.write_objcopy(root)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            tools = hotswap_reduce.ElfSectionTools(
                str(readobj),
                str(objcopy),
                5.0,
                ["*"],
                [],
                artifacts,
            )
            path = root / "object.co"
            path.write_bytes(b"ELF")
            with self.assertRaisesRegex(
                hotswap_reduce.ReducerError, "malformed section JSON"
            ):
                tools.removable_sections(path)

    def test_objcopy_output_must_remove_requested_sections(self) -> None:
        if os.name == "nt":
            self.skipTest("executable script fixtures require a POSIX host")
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            readobj = self.write_readobj(root, [(".debug_info", False)])
            objcopy = self.write_objcopy(root)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            tools = hotswap_reduce.ElfSectionTools(
                str(readobj),
                str(objcopy),
                5.0,
                [".debug_*"],
                [],
                artifacts,
            )
            path = root / "object.co"
            path.write_bytes(b"ELF")
            code_object = hotswap_reduce.CodeObject(
                "id", path.name, path, hotswap_reduce.sha256_file(path)
            )
            with self.assertRaisesRegex(
                hotswap_reduce.ReducerError, "requested sections remain"
            ):
                tools.remove_sections(code_object, [".debug_info"])
            self.assertEqual(list(artifacts.iterdir()), [])

    def test_objcopy_output_must_preserve_unrequested_sections(self) -> None:
        if os.name == "nt":
            self.skipTest("executable script fixtures require a POSIX host")
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            readobj = self.write_readobj(
                root,
                [
                    (".debug_info", False),
                    (".comment", False),
                ],
            )
            objcopy = self.write_objcopy(root, annotate=True, extra_removal=".comment")
            artifacts = root / "artifacts"
            artifacts.mkdir()
            tools = hotswap_reduce.ElfSectionTools(
                str(readobj),
                str(objcopy),
                5.0,
                [".debug_info"],
                [],
                artifacts,
            )
            path = root / "object.co"
            path.write_bytes(b"ELF")
            code_object = hotswap_reduce.CodeObject(
                "id", path.name, path, hotswap_reduce.sha256_file(path)
            )
            with self.assertRaisesRegex(
                hotswap_reduce.ReducerError, "unrequested sections disappeared"
            ):
                tools.remove_sections(code_object, [".debug_info"])
            self.assertEqual(list(artifacts.iterdir()), [])


class EndToEndTests(unittest.TestCase):
    def test_reduces_bundle_and_metadata_and_writes_reproducer(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            bundle = write_bundle(
                root,
                [b"drop", b"KEEP", b"also-drop"],
                {
                    "kernels": ["drop", "kernel-keep"],
                    "cases": [
                        {"name": "drop", "arguments": ["drop"]},
                        {
                            "name": "case-keep",
                            "arguments": ["drop", "arg-keep"],
                        },
                    ],
                    "arguments": ["drop", "global-keep"],
                },
            )
            predicate = root / "predicate.py"
            predicate.write_text(
                textwrap.dedent(
                    """
                    import json
                    import pathlib
                    import sys
                    bundle_path = pathlib.Path(sys.argv[1])
                    value = json.loads(bundle_path.read_text())
                    root = bundle_path.parent
                    object_data = b"".join(
                        (root / item["path"]).read_bytes()
                        for item in value["code_objects"]
                    )
                    metadata = json.loads((root / value["metadata"]).read_text())
                    text = json.dumps(metadata)
                    interesting = (
                        b"KEEP" in object_data
                        and "kernel-keep" in text
                        and "case-keep" in text
                        and "arg-keep" in text
                        and "global-keep" in text
                    )
                    raise SystemExit(0 if interesting else 1)
                    """
                ),
                encoding="utf-8",
            )
            output = root / "reduced output [x]"
            args = make_args(
                bundle,
                output,
                sys.executable,
                [str(predicate), "{bundle}"],
            )
            final, log = hotswap_reduce.run_from_arguments(args)
            self.assertEqual(
                [item.path.read_bytes() for item in final.code_objects],
                [b"KEEP"],
            )
            self.assertEqual(final.metadata["kernels"], ["kernel-keep"])
            self.assertEqual(
                [case["name"] for case in final.metadata["cases"]],
                ["case-keep"],
            )
            self.assertEqual(final.metadata["cases"][0]["arguments"], ["arg-keep"])
            self.assertEqual(final.metadata["arguments"], ["global-keep"])
            self.assertEqual(log["format"], hotswap_reduce.LOG_FORMAT)
            self.assertEqual(log["version"], 1)
            self.assertTrue(log["transformations"])
            self.assertTrue(any(item["accepted"] for item in log["transformations"]))
            published_log = json.loads(
                (output / "reduction-log.json").read_text(encoding="utf-8")
            )
            self.assertEqual(published_log, log)
            self.assertEqual(published_log["reproduction"]["argv"][-1], "bundle.json")

    def test_deterministic_output_and_log(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            bundle = write_bundle(
                root,
                [b"drop", b"keep"],
                {"cases": ["drop", "keep"]},
            )
            predicate = root / "predicate.py"
            predicate.write_text(
                textwrap.dedent(
                    """
                    import pathlib
                    import sys
                    raise SystemExit(
                        0 if b"keep" in pathlib.Path(sys.argv[1]).read_bytes()
                        else 1
                    )
                    """
                ),
                encoding="utf-8",
            )
            logs = []
            bundles = []
            for index in range(2):
                output = root / f"out-{index}"
                args = make_args(
                    bundle,
                    output,
                    sys.executable,
                    [str(predicate), "{metadata}"],
                )
                _, log = hotswap_reduce.run_from_arguments(args)
                logs.append(log)
                bundles.append(
                    (output / "bundle.json").read_bytes()
                    + (output / "metadata.json").read_bytes()
                )
            self.assertEqual(logs[0], logs[1])
            self.assertEqual(bundles[0], bundles[1])

    def test_offline_reference_manifest_differential(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            bundle = write_bundle(
                root,
                [b"same-a", b"candidate-differs", b"same-b"],
                {"cases": ["a", "discrepant-case", "b"]},
            )
            reference = root / "pr3598-reference.json"
            reference.write_text(
                json.dumps({"forbidden_sha256": "unused"}),
                encoding="utf-8",
            )
            predicate = root / "offline-diff.py"
            predicate.write_text(
                textwrap.dedent(
                    """
                    import json
                    import pathlib
                    import sys
                    reference = json.loads(pathlib.Path(sys.argv[1]).read_text())
                    bundle_path = pathlib.Path(sys.argv[2])
                    bundle = json.loads(bundle_path.read_text())
                    root = bundle_path.parent
                    objects = [
                        (root / entry["path"]).read_bytes()
                        for entry in bundle["code_objects"]
                    ]
                    metadata = json.loads((root / bundle["metadata"]).read_text())
                    difference = (
                        b"candidate-differs" in objects
                        and "discrepant-case" in metadata.get("cases", [])
                        and "forbidden_sha256" in reference
                    )
                    raise SystemExit(0 if difference else 1)
                    """
                ),
                encoding="utf-8",
            )
            output = root / "reduced-offline-difference"
            args = make_args(
                bundle,
                output,
                sys.executable,
                [str(predicate), str(reference), "{bundle}"],
                predicate_runs=2,
            )
            final, _ = hotswap_reduce.run_from_arguments(args)
            self.assertEqual(
                [item.path.read_bytes() for item in final.code_objects],
                [b"candidate-differs"],
            )
            self.assertEqual(final.metadata["cases"], ["discrepant-case"])

    def test_repeats_hierarchical_passes_to_a_fixpoint(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            bundle = write_bundle(
                root,
                [b"A", b"B"],
                {"arguments": ["X", "Y"]},
            )
            predicate = root / "nonmonotone-predicate.py"
            predicate.write_text(
                textwrap.dedent(
                    """
                    import json
                    import pathlib
                    import sys
                    bundle_path = pathlib.Path(sys.argv[1])
                    bundle = json.loads(bundle_path.read_text())
                    root = bundle_path.parent
                    objects = {
                        (root / entry["path"]).read_bytes().decode()
                        for entry in bundle["code_objects"]
                    }
                    metadata = json.loads((root / bundle["metadata"]).read_text())
                    arguments = metadata["arguments"]
                    interesting = (
                        (objects == {"A", "B"} and arguments in (["X", "Y"], ["X"]))
                        or (objects == {"A"} and arguments == ["X"])
                    )
                    raise SystemExit(0 if interesting else 1)
                    """
                ),
                encoding="utf-8",
            )
            output = root / "reduced"
            args = make_args(
                bundle,
                output,
                sys.executable,
                [str(predicate), "{bundle}"],
            )
            final, _ = hotswap_reduce.run_from_arguments(args)
            self.assertEqual(
                [item.path.read_bytes() for item in final.code_objects],
                [b"A"],
            )
            self.assertEqual(final.metadata["arguments"], ["X"])

    def test_section_reduction_with_fake_llvm_tools(self) -> None:
        if os.name == "nt":
            self.skipTest("executable script fixtures require a POSIX host")
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            bundle = write_bundle(root, [b"ELF"], {})
            fixture = FakeElfToolTests()
            readobj = fixture.write_readobj(
                root,
                [
                    ("", False),
                    (".text", True),
                    (".debug_info", False),
                    (".note", False),
                ],
            )
            objcopy = fixture.write_objcopy(root, annotate=True)
            predicate = root / "predicate.py"
            predicate.write_text("raise SystemExit(0)\n", encoding="utf-8")
            output = root / "output"
            args = make_args(
                bundle,
                output,
                sys.executable,
                [str(predicate), "{input}"],
                allow_remove_section=["*"],
                readobj=str(readobj),
                objcopy=str(objcopy),
            )
            final, log = hotswap_reduce.run_from_arguments(args)
            self.assertIn(
                b"--remove-section=.debug_info",
                final.code_objects[0].path.read_bytes(),
            )
            accepted_sections = [
                item["transformation"]["remove_sections"]
                for item in log["transformations"]
                if item["pass"] == "elf-sections" and item["accepted"]
            ]
            self.assertEqual(accepted_sections, [[".debug_info"]])

    def test_unsupported_section_format_publishes_nothing(self) -> None:
        if os.name == "nt":
            self.skipTest("executable script fixtures require a POSIX host")
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            bundle = write_bundle(root, [b"ELF"], {})
            fixture = FakeElfToolTests()
            readobj = fixture.write_readobj(
                root, [(".debug_info", False)], architecture="x86_64"
            )
            objcopy = fixture.write_objcopy(root)
            predicate = root / "predicate.py"
            predicate.write_text("raise SystemExit(0)\n", encoding="utf-8")
            output = root / "output"
            args = make_args(
                bundle,
                output,
                sys.executable,
                [str(predicate), "{input}"],
                allow_remove_section=[".debug_*"],
                readobj=str(readobj),
                objcopy=str(objcopy),
            )
            with self.assertRaisesRegex(
                hotswap_reduce.ReducerError, "requires an AMDGPU ELF"
            ):
                hotswap_reduce.run_from_arguments(args)
            self.assertFalse(output.exists())

    def test_existing_output_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            bundle = write_bundle(root, [b"original"], {})
            output = root / "output"
            output.mkdir()
            marker = output / "marker"
            marker.write_bytes(b"preserve")
            args = make_args(
                bundle,
                output,
                sys.executable,
                ["-c", "raise SystemExit(0)", "{bundle}"],
            )
            with self.assertRaisesRegex(
                hotswap_reduce.ReducerError, "refusing to overwrite"
            ):
                hotswap_reduce.run_from_arguments(args)
            self.assertEqual(marker.read_bytes(), b"preserve")

    def test_cache_cannot_overwrite_input_or_live_inside_output(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            bundle = write_bundle(root, [b"original"], {})
            object_path = root / "object-0.co"
            predicate = root / "predicate.py"
            predicate.write_text("raise SystemExit(0)\n", encoding="utf-8")
            output = root / "output"
            for cache_path in (object_path, output / "cache.json", predicate):
                args = make_args(
                    bundle,
                    output,
                    sys.executable,
                    [str(predicate), "{bundle}"],
                    cache_file=cache_path,
                )
                with self.assertRaisesRegex(
                    hotswap_reduce.ReducerError,
                    r"refusing to overwrite|must not be inside",
                ):
                    hotswap_reduce.run_from_arguments(args)
                self.assertFalse(output.exists())
                self.assertEqual(object_path.read_bytes(), b"original")
                self.assertEqual(
                    predicate.read_text(encoding="utf-8"),
                    "raise SystemExit(0)\n",
                )

    def test_cache_cannot_overwrite_external_bundle_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            external_metadata = root / "metadata.json"
            external_metadata.write_text(
                json.dumps(
                    {
                        "format": hotswap_reduce.CACHE_FORMAT,
                        "version": hotswap_reduce.CACHE_VERSION,
                        "entries": {},
                    }
                ),
                encoding="utf-8",
            )
            original_metadata = external_metadata.read_bytes()
            bundle = write_bundle(root, [b"object"], external_metadata.name)
            predicate = root / "predicate.py"
            predicate.write_text("raise SystemExit(0)\n", encoding="utf-8")
            args = make_args(
                bundle,
                root / "output",
                sys.executable,
                [str(predicate), "{bundle}"],
                cache_file=external_metadata,
            )
            with self.assertRaisesRegex(
                hotswap_reduce.ReducerError, "overwrite reducer input"
            ):
                hotswap_reduce.run_from_arguments(args)
            self.assertEqual(external_metadata.read_bytes(), original_metadata)

    def test_identity_change_during_reduction_aborts_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            bundle = write_bundle(root, [b"one", b"two"], {})
            dependency = root / "dependency"
            dependency.write_bytes(b"original")
            counter = root / "counter"
            predicate = root / "predicate.py"
            predicate.write_text(
                textwrap.dedent(
                    f"""
                    import pathlib
                    counter = pathlib.Path({str(counter)!r})
                    dependency = pathlib.Path({str(dependency)!r})
                    count = int(counter.read_text()) if counter.exists() else 0
                    counter.write_text(str(count + 1))
                    if count == 1:
                        dependency.write_bytes(b"changed")
                    raise SystemExit(0)
                    """
                ),
                encoding="utf-8",
            )
            output = root / "output"
            args = make_args(
                bundle,
                output,
                sys.executable,
                [str(predicate), "{bundle}"],
                cache_dependency=[dependency],
            )
            with self.assertRaisesRegex(
                hotswap_reduce.ReducerError, "identity changed"
            ):
                hotswap_reduce.run_from_arguments(args)
            self.assertFalse(output.exists())

    def test_interruption_publishes_nothing_and_preserves_original(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            bundle = write_bundle(root, [b"one", b"two"], {})
            output = root / "output"
            args = make_args(
                bundle,
                output,
                sys.executable,
                ["-c", "raise SystemExit(0)", "{bundle}"],
            )
            interesting = hotswap_reduce.PredicateResult("interesting", (0,), "", "")
            with mock.patch.object(
                hotswap_reduce.PredicateRunner,
                "evaluate",
                side_effect=[interesting, KeyboardInterrupt()],
            ):
                with self.assertRaises(KeyboardInterrupt):
                    hotswap_reduce.run_from_arguments(args)
            self.assertFalse(output.exists())
            self.assertEqual((root / "object-0.co").read_bytes(), b"one")
            self.assertEqual((root / "object-1.co").read_bytes(), b"two")

    def test_publish_interruption_rolls_back_temporary_output(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            bundle = write_bundle(root, [b"original"], {})
            candidate = hotswap_reduce.load_bundle(bundle)
            output = root / "published"
            real_replace = os.replace

            def replace_or_interrupt(
                source: str | bytes | os.PathLike[str] | os.PathLike[bytes],
                destination: str | bytes | os.PathLike[str] | os.PathLike[bytes],
            ) -> None:
                if Path(os.fsdecode(destination)) == output.resolve():
                    raise KeyboardInterrupt()
                real_replace(source, destination)

            with mock.patch.object(
                hotswap_reduce.os,
                "replace",
                side_effect=replace_or_interrupt,
            ):
                with self.assertRaises(KeyboardInterrupt):
                    hotswap_reduce.atomic_publish(
                        output,
                        candidate,
                        {"format": "test", "version": 1},
                    )
            self.assertFalse(output.exists())
            self.assertEqual((root / "object-0.co").read_bytes(), b"original")
            temporary_outputs = list(root.glob(".published.*"))
            self.assertEqual(temporary_outputs, [])

    def test_uninteresting_original_publishes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            bundle = write_bundle(root, [b"object"], {})
            predicate = root / "predicate.py"
            predicate.write_text("raise SystemExit(1)\n", encoding="utf-8")
            output = root / "output"
            args = make_args(
                bundle,
                output,
                sys.executable,
                [str(predicate), "{bundle}"],
            )
            with self.assertRaisesRegex(
                hotswap_reduce.ReducerError, "original input is not"
            ):
                hotswap_reduce.run_from_arguments(args)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
