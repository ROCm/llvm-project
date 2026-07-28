#!/usr/bin/env python3
# ===- test_hotswap_test_select.py - Selector tests -----------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===#

import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


COMGR_DIR = Path(__file__).resolve().parents[2]
SCRIPT = COMGR_DIR / "utils" / "hotswap" / "hotswap_test_select.py"
INPUTS_DIR = COMGR_DIR / "test" / "Inputs" / "hotswap"
SPEC = importlib.util.spec_from_file_location("hotswap_test_select", SCRIPT)
selector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(selector)
EVIDENCE_SCRIPT = COMGR_DIR / "test" / "utils" / "test_hotswap_test_select_evidence.py"
EVIDENCE_SPEC = importlib.util.spec_from_file_location(
    "test_hotswap_test_select_evidence", EVIDENCE_SCRIPT
)
evidence = importlib.util.module_from_spec(EVIDENCE_SPEC)
EVIDENCE_SPEC.loader.exec_module(evidence)


def observation(*tests):
    return {
        "kind": selector.OBSERVATION_KIND,
        "schema_version": selector.SCHEMA_VERSION,
        "tests": list(tests),
    }


def test_record(test_id, runtime_ms, features, **overrides):
    record = {
        "features": features,
        "id": test_id,
        "runtime_ms": runtime_ms,
    }
    record.update(overrides)
    return record


class NormalizeTests(unittest.TestCase):
    def test_empty_input(self):
        result = selector.select_tests(observation())
        self.assertEqual(result["selected_tests"], [])
        self.assertEqual(result["uncovered_features"], [])
        self.assertEqual(result["estimated_runtime_ms"], 0)

    def test_rejects_missing_fields_and_unknown_fields(self):
        with self.assertRaisesRegex(selector.SelectionError, "missing runtime_ms"):
            selector.normalize_document(observation({"id": "broken", "features": []}))
        with self.assertRaisesRegex(selector.SelectionError, "unknown field"):
            selector.normalize_document(
                observation(test_record("broken", 1, [], typo=True))
            )

    def test_rejects_malformed_values(self):
        bad_records = [
            test_record("", 1, []),
            test_record("bad-runtime", -1, []),
            test_record("bad-features", 1, "branch"),
            test_record("bad-feature", 1, [""]),
            test_record("bad-risk", 1, [], risk=float("inf")),
            test_record("bad-samples", 1, [], sample_count=0),
            test_record("bad-outcome", 1, [], candidate_outcome="crash"),
            test_record(
                "bad-regression-marker", 1, [], has_candidate_only_regression=1
            ),
        ]
        for record in bad_records:
            with self.subTest(record=record):
                with self.assertRaises(selector.SelectionError):
                    selector.normalize_document(observation(record))

    def test_schema_versions_must_be_integers(self):
        for version in (True, 1.0, "1"):
            with self.subTest(version=version):
                document = observation()
                document["schema_version"] = version
                with self.assertRaisesRegex(selector.SelectionError, "schema_version"):
                    selector.normalize_document(document)

    def test_huge_numbers_fail_cleanly_and_large_means_remain_finite(self):
        with self.assertRaisesRegex(selector.SelectionError, "finite"):
            selector.normalize_document(
                observation(test_record("too-large", 10**10000, []))
            )

        result = selector.normalize_document(
            observation(test_record("large", 1e308, [], sample_count=2))
        )
        self.assertTrue(math.isfinite(float(result["tests"][0]["runtime_ms"])))
        self.assertEqual(float(result["tests"][0]["runtime_ms"]), 1e308)

    def test_normalization_is_stable(self):
        document = observation(
            test_record("z", 2.0, ["branch", "branch", "literal"]),
            test_record("a", 1, []),
        )
        first = selector.normalize_document(document)
        second = selector.normalize_document(first)
        self.assertEqual(first, second)
        self.assertEqual([test["id"] for test in first["tests"]], ["a", "z"])
        self.assertEqual(first["tests"][1]["features"], ["branch", "literal"])


class MergeTests(unittest.TestCase):
    def test_duplicates_merge_observations(self):
        first = observation(
            test_record(
                "case",
                10,
                ["branch"],
                candidate_outcome="pass",
                original_outcome="pass",
                novelty=1,
                risk=2,
                sample_count=2,
            )
        )
        second = observation(
            test_record(
                "case",
                40,
                ["literal"],
                candidate_outcome="fail",
                original_outcome="pass",
                novelty=3,
                risk=1,
            )
        )
        result = selector.merge_documents([first, second])
        merged = result["tests"][0]
        self.assertEqual(merged["features"], ["branch", "literal"])
        self.assertEqual(merged["runtime_ms"], 20)
        self.assertEqual(merged["sample_count"], 3)
        self.assertEqual(merged["candidate_outcome"], "mixed")
        self.assertEqual(merged["original_outcome"], "pass")
        self.assertEqual(merged["novelty"], 3)
        self.assertEqual(merged["risk"], 2)

    def test_merge_is_independent_of_input_order(self):
        first = observation(test_record("case", 11, ["a"], sample_count=2))
        second = observation(test_record("case", 17, ["b"], sample_count=3))
        self.assertEqual(
            selector.merge_documents([first, second]),
            selector.merge_documents([second, first]),
        )

    def test_merge_is_stable_across_all_input_permutations(self):
        documents = [
            observation(test_record("case", 1.25, ["a"], sample_count=2)),
            observation(test_record("case", 9.5, ["b"], sample_count=3)),
            observation(test_record("case", 4, ["c"], sample_count=5)),
        ]
        expected = selector.merge_documents(documents)
        for permutation in itertools.permutations(documents):
            with self.subTest(permutation=permutation):
                self.assertEqual(selector.merge_documents(permutation), expected)

    def test_duplicates_within_one_document_merge(self):
        result = selector.normalize_document(
            observation(
                test_record("case", 2, ["a"]),
                test_record("case", 4, ["b"]),
            )
        )
        self.assertEqual(len(result["tests"]), 1)
        self.assertEqual(result["tests"][0]["runtime_ms"], 3)
        self.assertEqual(result["tests"][0]["sample_count"], 2)

    def test_merge_preserves_an_observed_candidate_only_regression(self):
        failed = observation(
            test_record(
                "flaky-regression",
                2,
                [],
                candidate_outcome="fail",
                original_outcome="pass",
            )
        )
        passed = observation(
            test_record(
                "flaky-regression",
                1,
                [],
                candidate_outcome="pass",
                original_outcome="pass",
            )
        )
        merged = selector.merge_documents([failed, passed])
        self.assertEqual(merged["tests"][0]["candidate_outcome"], "mixed")
        self.assertTrue(merged["tests"][0]["has_candidate_only_regression"])

        selection = selector.select_tests(merged, test_count_budget=0)
        self.assertEqual(
            selection["known_candidate_only_regressions"], ["flaky-regression"]
        )
        self.assertTrue(selection["selected_tests"][0]["forced"])


class AdapterTests(unittest.TestCase):
    def test_adapts_inventory_and_cross_tool_observations(self):
        with open(
            INPUTS_DIR / "inventory-v1.json", "r", encoding="utf-8"
        ) as inventory_file:
            inventory = json.load(inventory_file)
        with open(
            INPUTS_DIR / "inventory-observations-v1.json",
            "r",
            encoding="utf-8",
        ) as observations_file:
            observations = json.load(observations_file)

        result = selector.adapt_documents([inventory, observations])
        self.assertEqual(len(result["tests"]), 2)
        self.assertEqual(result["tests"][0]["runtime_ms"], 4)
        self.assertEqual(result["tests"][1]["features"], ["far-routing", "live-state"])
        self.assertEqual(
            result["tests"][1]["provenance"],
            ["comgr.hotswap.inventory", "hotswap-audit"],
        )
        selection = selector.select_tests(result, test_count_budget=1)
        self.assertEqual(
            selection["known_candidate_only_regressions"],
            ["sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"],
        )
        self.assertTrue(selection["selected_tests"][0]["forced"])
        self.assertEqual(
            selection["selected_tests"][0]["provenance"],
            ["comgr.hotswap.inventory", "hotswap-audit"],
        )

    def test_inventory_requires_measured_or_explicit_runtime(self):
        with open(
            INPUTS_DIR / "inventory-v1.json", "r", encoding="utf-8"
        ) as inventory_file:
            inventory = json.load(inventory_file)
        with self.assertRaisesRegex(selector.SelectionError, "no runtime_ms"):
            selector.adapt_documents([inventory])

        result = selector.adapt_documents([inventory], default_runtime_ms=10)
        self.assertEqual([record["runtime_ms"] for record in result["tests"]], [10, 10])
        self.assertTrue(
            all(
                "adapter-default-runtime" in record["provenance"]
                for record in result["tests"]
            )
        )

    def test_adapts_and_joins_forward_compatible_producer_records(self):
        manifest = {
            "kind": "vendor-rewrite-manifest",
            "schema_version": 1,
            "source": "rewrite-manifest",
            "future_envelope_field": {"ignored": True},
            "records": [
                {
                    "test_id": "suite/case",
                    "runtime_ms": 12,
                    "feature_tags": ["far-gateway"],
                    "candidate_outcome": "fail",
                    "original_outcome": "pass",
                    "future_record_field": 7,
                }
            ],
        }
        audit = {
            "kind": "vendor-audit",
            "schema_version": 1,
            "source": "audit",
            "records": [
                {
                    "test_id": "suite/case",
                    "features": ["scratch-live"],
                    "risk": 3,
                }
            ],
        }
        semcheck = {
            "kind": "vendor-semcheck",
            "schema_version": 1,
            "source": "semcheck",
            "records": [
                {
                    "id": "suite/case",
                    "features": ["round-trip-state"],
                    "novelty": 2,
                    "provenance": "solver",
                }
            ],
        }
        result = selector.adapt_documents([manifest, audit, semcheck])
        record = result["tests"][0]
        self.assertEqual(record["id"], "suite/case")
        self.assertEqual(record["runtime_ms"], 12)
        self.assertEqual(record["sample_count"], 1)
        self.assertEqual(
            record["features"],
            ["far-gateway", "round-trip-state", "scratch-live"],
        )
        self.assertEqual(record["candidate_outcome"], "fail")
        self.assertEqual(record["original_outcome"], "pass")
        self.assertTrue(record["has_candidate_only_regression"])
        self.assertEqual(record["risk"], 3)
        self.assertEqual(record["novelty"], 2)
        self.assertEqual(
            record["provenance"],
            ["audit", "rewrite-manifest", "semcheck", "solver"],
        )

    def test_adapter_averages_only_reported_runtimes(self):
        first = {
            "kind": "tool",
            "schema_version": 1,
            "records": [{"test_id": "case", "runtime_ms": 10, "sample_count": 2}],
        }
        second = {
            "kind": "tool",
            "schema_version": 1,
            "records": [{"test_id": "case", "features": ["audit-only"]}],
        }
        result = selector.adapt_documents([first, second])
        self.assertEqual(result["tests"][0]["runtime_ms"], 10)
        self.assertEqual(result["tests"][0]["sample_count"], 2)

    def test_adapter_rejects_ambiguous_or_malformed_records(self):
        with self.assertRaisesRegex(selector.SelectionError, "schema_version"):
            selector.adapt_documents([{"kind": "tool", "records": []}])
        with self.assertRaisesRegex(selector.SelectionError, "conflicting"):
            selector.adapt_documents(
                [
                    {
                        "kind": "tool",
                        "schema_version": 1,
                        "records": [{"id": "a", "test_id": "b"}],
                    }
                ]
            )
        with self.assertRaisesRegex(selector.SelectionError, "no runtime_ms"):
            selector.adapt_documents(
                [
                    {
                        "kind": "tool",
                        "schema_version": 1,
                        "records": [{"test_id": "untimed"}],
                    }
                ]
            )
        with self.assertRaisesRegex(selector.SelectionError, "inventory version"):
            selector.adapt_documents(
                [
                    {
                        "schema": selector.INVENTORY_SCHEMA,
                        "version": 2,
                        "objects": [],
                    }
                ]
            )
        with self.assertRaisesRegex(selector.SelectionError, "invalid sha256"):
            selector.adapt_documents(
                [
                    {
                        "schema": selector.INVENTORY_SCHEMA,
                        "version": 1,
                        "objects": [{"sha256": "not-a-digest"}],
                    }
                ],
                default_runtime_ms=1,
            )
        with self.assertRaisesRegex(selector.SelectionError, "conflicting features"):
            selector.adapt_documents(
                [
                    {
                        "kind": "tool",
                        "schema_version": 1,
                        "records": [
                            {
                                "test_id": "case",
                                "features": ["a"],
                                "feature_tags": ["b"],
                                "runtime_ms": 1,
                            }
                        ],
                    }
                ]
            )
        with self.assertRaisesRegex(selector.SelectionError, "conflicting records"):
            selector.adapt_documents(
                [
                    {
                        "kind": "tool",
                        "schema_version": 1,
                        "records": [],
                        "tests": [{"test_id": "ignored", "runtime_ms": 1}],
                    }
                ]
            )
        with self.assertRaisesRegex(selector.SelectionError, "schema_version"):
            selector.adapt_documents(
                [{"kind": "tool", "schema_version": True, "records": []}]
            )
        with self.assertRaisesRegex(selector.SelectionError, "inventory version"):
            selector.adapt_documents(
                [
                    {
                        "schema": selector.INVENTORY_SCHEMA,
                        "version": True,
                        "objects": [],
                    }
                ]
            )


class SelectionTests(unittest.TestCase):
    def test_simple_cover_chooses_one_test(self):
        result = selector.select_tests(
            observation(
                test_record("a-only", 1, ["a"]),
                test_record("b-only", 1, ["b"]),
                test_record("both", 2, ["a", "b"]),
            ),
            test_count_budget=1,
        )
        self.assertEqual([test["id"] for test in result["selected_tests"]], ["both"])
        self.assertEqual(result["selected_tests"][0]["marginal_features"], ["a", "b"])
        self.assertEqual(result["uncovered_features"], [])

    def test_runtime_budget_prefers_coverage_per_millisecond(self):
        result = selector.select_tests(
            observation(
                test_record("slow", 100, ["a", "b"]),
                test_record("cheap-a", 5, ["a"]),
                test_record("cheap-b", 5, ["b"]),
            ),
            runtime_budget_ms=10,
        )
        self.assertEqual(
            [test["id"] for test in result["selected_tests"]],
            ["cheap-a", "cheap-b"],
        )
        self.assertEqual(result["estimated_runtime_ms"], 10)

    def test_count_budget_reports_uncovered_features(self):
        result = selector.select_tests(
            observation(
                test_record("a", 1, ["a"]),
                test_record("b", 1, ["b"]),
            ),
            test_count_budget=1,
        )
        self.assertEqual(len(result["selected_tests"]), 1)
        self.assertEqual(result["uncovered_features"], ["b"])

    def test_regressions_are_forced_past_budgets(self):
        result = selector.select_tests(
            observation(
                test_record(
                    "regression-a",
                    8,
                    ["a"],
                    candidate_outcome="fail",
                    original_outcome="pass",
                ),
                test_record(
                    "regression-b",
                    8,
                    ["b"],
                    candidate_outcome="fail",
                    original_outcome="pass",
                ),
                test_record("other", 1, ["c"]),
            ),
            runtime_budget_ms=5,
            test_count_budget=1,
        )
        self.assertEqual(
            [test["id"] for test in result["selected_tests"]],
            ["regression-a", "regression-b"],
        )
        self.assertTrue(all(test["forced"] for test in result["selected_tests"]))
        self.assertTrue(result["budget_exceeded_by_forced_regressions"])
        self.assertEqual(result["uncovered_features"], ["c"])

    def test_mixed_outcome_is_not_a_forced_regression(self):
        result = selector.select_tests(
            observation(
                test_record(
                    "mixed",
                    1,
                    [],
                    candidate_outcome="mixed",
                    original_outcome="pass",
                )
            ),
            test_count_budget=0,
        )
        self.assertEqual(result["selected_tests"], [])
        self.assertEqual(result["known_candidate_only_regressions"], [])

    def test_ties_use_runtime_then_identifier(self):
        result = selector.select_tests(
            observation(
                test_record("z", 1, ["a"]),
                test_record("b", 1, ["a"]),
                test_record("a-slow", 2, ["a"]),
            ),
            test_count_budget=1,
        )
        self.assertEqual(result["selected_tests"][0]["id"], "b")

    def test_feature_weights_affect_selection(self):
        result = selector.select_tests(
            observation(
                test_record("ab", 1, ["a", "b"]),
                test_record("critical", 1, ["critical"]),
            ),
            feature_weights={"critical": 10},
            test_count_budget=1,
        )
        self.assertEqual(result["selected_tests"][0]["id"], "critical")

    def test_novelty_breaks_equal_coverage(self):
        result = selector.select_tests(
            observation(
                test_record("familiar", 1, ["a"], novelty=0),
                test_record("novel", 1, ["a"], novelty=4),
            ),
            test_count_budget=1,
        )
        self.assertEqual(result["selected_tests"][0]["id"], "novel")

    def test_rarity_novelty_mode_is_label_agnostic(self):
        result = selector.select_tests(
            observation(
                test_record("common", 1, ["shared"]),
                test_record("rare", 1, ["shared", "never-seen-before"]),
            ),
            requested_features=["shared"],
            novelty_mode="rarity",
            test_count_budget=1,
        )
        self.assertEqual(result["selected_tests"][0]["id"], "rare")
        self.assertGreater(result["selected_tests"][0]["effective_novelty"], 0)

    def test_novel_and_unknown_quotas_select_calibration_cases(self):
        result = selector.select_tests(
            observation(
                test_record(
                    "known",
                    1,
                    ["a"],
                    candidate_outcome="pass",
                    original_outcome="pass",
                ),
                test_record(
                    "novel",
                    1,
                    [],
                    novelty=2,
                    candidate_outcome="pass",
                    original_outcome="pass",
                ),
                test_record("unknown", 1, []),
            ),
            min_novel_tests=1,
            min_unknown_tests=1,
        )
        self.assertEqual(
            [test["id"] for test in result["selected_tests"]],
            ["novel", "known", "unknown"],
        )
        self.assertEqual(
            result["unfilled_quotas"], {"novel_tests": 0, "unknown_tests": 0}
        )

    def test_impossible_quotas_are_reported(self):
        result = selector.select_tests(
            observation(
                test_record(
                    "known",
                    1,
                    [],
                    candidate_outcome="pass",
                    original_outcome="pass",
                )
            ),
            min_novel_tests=1,
            min_unknown_tests=1,
        )
        self.assertEqual(
            result["unfilled_quotas"], {"novel_tests": 1, "unknown_tests": 1}
        )

    def test_requested_missing_feature_is_reported(self):
        result = selector.select_tests(
            observation(test_record("case", 1, ["present"])),
            requested_features=["absent"],
        )
        self.assertEqual(result["selected_tests"], [])
        self.assertEqual(result["uncovered_features"], ["absent"])

    def test_invalid_selection_options_are_rejected(self):
        document = observation()
        invalid_options = [
            {"test_count_budget": -1},
            {"min_novel_tests": -1},
            {"min_unknown_tests": -1},
            {"feature_weights": {"": 1}},
            {"requested_features": [1]},
            {"holdout_percent": 101},
            {"holdout_seed": 7},
            {"novelty_mode": "memorize-names"},
            {"risk_weight": float("nan")},
            {"runtime_budget_ms": float("inf")},
        ]
        for options in invalid_options:
            with self.subTest(options=options):
                with self.assertRaises(selector.SelectionError):
                    selector.select_tests(document, **options)

    def test_score_and_selected_runtime_overflow_fail_cleanly(self):
        irrelevant_risk = selector.select_tests(
            observation(
                test_record("irrelevant", 1, [], risk=1e308),
                test_record("covers-a", 1, ["a"]),
            ),
            risk_weight=1e308,
        )
        self.assertEqual(
            [record["id"] for record in irrelevant_risk["selected_tests"]],
            ["covers-a"],
        )

        with self.assertRaisesRegex(selector.SelectionError, "risk score"):
            selector.select_tests(
                observation(test_record("large-risk", 1, ["a"], risk=1e308)),
                risk_weight=1e308,
            )

        regressions = observation(
            *[
                test_record(
                    "regression-%d" % index,
                    1e308,
                    [],
                    candidate_outcome="fail",
                    original_outcome="pass",
                )
                for index in range(2)
            ]
        )
        with self.assertRaisesRegex(selector.SelectionError, "selected runtime"):
            selector.select_tests(regressions)

    def test_zero_runtime_test_fits_zero_runtime_budget(self):
        result = selector.select_tests(
            observation(test_record("free", 0, ["covered"])),
            runtime_budget_ms=0,
        )
        self.assertEqual(
            [record["id"] for record in result["selected_tests"]], ["free"]
        )
        self.assertEqual(result["estimated_runtime_ms"], 0)

    def test_holdout_is_deterministic_and_keeps_regressions(self):
        result = selector.select_tests(
            observation(
                test_record("reserved", 1, ["held"]),
                test_record(
                    "regression",
                    2,
                    ["forced"],
                    candidate_outcome="fail",
                    original_outcome="pass",
                ),
            ),
            holdout_percent=100,
            holdout_seed="acceptance",
        )
        self.assertEqual(
            [test["id"] for test in result["selected_tests"]], ["regression"]
        )
        self.assertEqual(result["holdout"]["reserved_tests"], ["reserved"])
        self.assertEqual(result["holdout"]["regression_overrides"], ["regression"])
        repeated = selector.select_tests(
            observation(
                test_record("reserved", 1, ["held"]),
                test_record(
                    "regression",
                    2,
                    ["forced"],
                    candidate_outcome="fail",
                    original_outcome="pass",
                ),
            ),
            holdout_percent=100,
            holdout_seed="acceptance",
        )
        self.assertEqual(result, repeated)

    def test_holdout_seed_changes_generic_partition(self):
        document = observation(
            *[test_record("case-%02d" % index, 1, ["shared"]) for index in range(40)]
        )
        first = selector.select_tests(
            document, holdout_percent=50, holdout_seed="first"
        )
        second = selector.select_tests(
            document, holdout_percent=50, holdout_seed="second"
        )
        self.assertNotEqual(
            first["holdout"]["reserved_tests"],
            second["holdout"]["reserved_tests"],
        )

    def test_digest_ignores_input_format_and_order(self):
        first = observation(
            test_record("b", 2, ["y"]),
            test_record("a", 1, ["x"]),
        )
        second = observation(
            test_record("a", 1.0, ["x"]),
            test_record("b", 2.0, ["y"]),
        )
        self.assertEqual(
            selector.select_tests(first)["input_digest"],
            selector.select_tests(second)["input_digest"],
        )


class CommandLineTests(unittest.TestCase):
    def setUp(self):
        self.script = str(SCRIPT)

    def run_cli(self, *arguments):
        return subprocess.run(
            [sys.executable, self.script] + list(arguments),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def test_unknown_flag_is_rejected(self):
        result = self.run_cli("select", "--input", "not-read.json", "--unknown-flag")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unrecognized arguments", result.stderr)

    def test_select_cli_emits_stable_json(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = os.path.join(directory, "input.json")
            with open(input_path, "w", encoding="utf-8") as output:
                json.dump(observation(test_record("case", 1, ["branch"])), output)
            first = self.run_cli("select", "--input", input_path)
            second = self.run_cli("select", "--input", input_path)
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(first.stdout, second.stdout)
        self.assertEqual(json.loads(first.stdout)["kind"], selector.SELECTION_KIND)

    def test_output_cannot_alias_input(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = os.path.join(directory, "input.json")
            original = "not valid JSON"
            with open(input_path, "w", encoding="utf-8") as output:
                output.write(original)
            alias_path = os.path.join(directory, ".", "input.json")
            result = self.run_cli(
                "select",
                "--input",
                input_path,
                "--output",
                alias_path,
            )
            with open(input_path, "r", encoding="utf-8") as input_file:
                after = input_file.read()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("refusing to overwrite input", result.stderr)
        self.assertEqual(after, original)

    def test_output_cannot_alias_input_through_hardlink(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = os.path.join(directory, "input.json")
            alias_path = os.path.join(directory, "hardlink.json")
            with open(input_path, "w", encoding="utf-8") as output:
                json.dump(observation(test_record("case", 1, [])), output)
            try:
                os.link(input_path, alias_path)
            except OSError as error:
                self.skipTest("hard links unavailable: %s" % error)
            result = self.run_cli(
                "select",
                "--input",
                input_path,
                "--output",
                alias_path,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("refusing to overwrite input", result.stderr)

    def test_unicode_paths_and_identifiers_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = os.path.join(
                directory, "observations-\N{GREEK SMALL LETTER PI}.json"
            )
            output_path = os.path.join(directory, "selection-\N{SNOWMAN}.json")
            with open(input_path, "w", encoding="utf-8") as output:
                json.dump(
                    observation(
                        test_record(
                            "suite/\N{GREEK SMALL LETTER DELTA}",
                            1,
                            ["feature-\N{SNOWMAN}"],
                        )
                    ),
                    output,
                )
            result = self.run_cli(
                "select", "--input", input_path, "--output", output_path
            )
            with open(output_path, "r", encoding="utf-8") as output:
                selected = json.load(output)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            selected["selected_tests"][0]["id"],
            "suite/\N{GREEK SMALL LETTER DELTA}",
        )

    def test_invalid_utf8_and_huge_json_number_report_clean_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            invalid_utf8 = os.path.join(directory, "invalid-utf8.json")
            with open(invalid_utf8, "wb") as output:
                output.write(b"\xff")
            utf8_result = self.run_cli("select", "--input", invalid_utf8)

            huge_number = os.path.join(directory, "huge-number.json")
            with open(huge_number, "w", encoding="utf-8") as output:
                output.write(
                    '{"kind":"hotswap-test-observations","schema_version":1,'
                    '"tests":[{"id":"case","features":[],"runtime_ms":'
                    + "1"
                    + "0" * 5000
                    + "}]}"
                )
            number_result = self.run_cli("select", "--input", huge_number)

        self.assertNotEqual(utf8_result.returncode, 0)
        self.assertIn("cannot parse JSON", utf8_result.stderr)
        self.assertNotIn("Traceback", utf8_result.stderr)

        # Python 3.11+ rejects the integer while parsing; older supported
        # interpreters parse it and the selector's finite-range check rejects it.
        self.assertNotEqual(number_result.returncode, 0)
        self.assertTrue(
            "cannot parse JSON" in number_result.stderr
            or "finite non-negative" in number_result.stderr
        )
        self.assertNotIn("Traceback", number_result.stderr)

    def test_atomic_write_rolls_back_and_removes_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = os.path.join(directory, "selection.json")
            with open(output_path, "w", encoding="utf-8") as output:
                output.write("original")
            with mock.patch.object(
                selector.os, "replace", side_effect=OSError("injected")
            ):
                with self.assertRaisesRegex(selector.SelectionError, "injected"):
                    selector._write_json(output_path, {"schema_version": 1})
            with open(output_path, "r", encoding="utf-8") as output:
                after = output.read()
            temporary_files = [
                name
                for name in os.listdir(directory)
                if name.startswith(".hotswap-test-select-")
            ]
        self.assertEqual(after, "original")
        self.assertEqual(temporary_files, [])

    def test_merge_cli_writes_output(self):
        with tempfile.TemporaryDirectory() as directory:
            first_path = os.path.join(directory, "first.json")
            second_path = os.path.join(directory, "second.json")
            output_path = os.path.join(directory, "merged.json")
            for path, feature in ((first_path, "a"), (second_path, "b")):
                with open(path, "w", encoding="utf-8") as output:
                    json.dump(observation(test_record("case", 1, [feature])), output)
            result = self.run_cli(
                "merge",
                "--input",
                first_path,
                second_path,
                "--output",
                output_path,
            )
            with open(output_path, "r", encoding="utf-8") as merged_file:
                merged = json.load(merged_file)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(merged["tests"][0]["features"], ["a", "b"])

    def test_adapt_cli_accepts_generic_producer_envelope(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = os.path.join(directory, "producer.json")
            with open(input_path, "w", encoding="utf-8") as output:
                json.dump(
                    {
                        "kind": "future-audit",
                        "schema_version": 1,
                        "records": [
                            {
                                "test_id": "case",
                                "runtime_ms": 1,
                                "features": ["future-feature"],
                            }
                        ],
                    },
                    output,
                )
            result = self.run_cli("adapt", "--input", input_path)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["tests"][0]["id"], "case")


class Pr3646AcceptanceTests(unittest.TestCase):
    def setUp(self):
        fixture_path = INPUTS_DIR / "pr3646-acceptance.json"
        with open(fixture_path, "r", encoding="utf-8") as fixture_file:
            self.fixture = json.load(fixture_file)
        self.risk_dimensions = [
            "ds2",
            "far-relay-gateway",
            "finite-call-closure",
            "m32-scale16",
            "materialized-call-closure",
            "owner-specific-source-window",
            "round-trip-vcc-preservation",
            "scratch-sgpr-liveness",
            "tensor-mask",
        ]

    def test_curated_acceptance_covers_risk_and_regressions(self):
        result = selector.select_tests(
            self.fixture,
            requested_features=self.risk_dimensions,
            test_count_budget=4,
            novelty_mode="combined",
        )
        self.assertEqual(result["uncovered_features"], [])
        self.assertEqual(len(result["selected_tests"]), 4)
        self.assertEqual(len(result["known_candidate_only_regressions"]), 2)
        self.assertTrue(result["selected_tests"][0]["forced"])
        self.assertTrue(result["selected_tests"][1]["forced"])

    def test_acceptance_result_depends_on_structure_not_labels(self):
        renamed = json.loads(json.dumps(self.fixture))
        feature_map = {}
        for index, record in enumerate(renamed["tests"]):
            record["id"] = "renamed/test-%02d" % index
            renamed_features = []
            for feature in record["features"]:
                if feature not in feature_map:
                    feature_map[feature] = "opaque-feature-%02d" % len(feature_map)
                renamed_features.append(feature_map[feature])
            record["features"] = renamed_features
        result = selector.select_tests(
            renamed,
            requested_features=[
                feature_map[feature] for feature in self.risk_dimensions
            ],
            test_count_budget=4,
            novelty_mode="combined",
        )
        self.assertEqual(result["uncovered_features"], [])
        self.assertEqual(len(result["selected_tests"]), 4)
        self.assertEqual(sum(test["forced"] for test in result["selected_tests"]), 2)


class EvidenceValidationTests(unittest.TestCase):
    def test_frozen_evidence_hash_is_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "evidence.tsv")
            with open(path, "wb") as output:
                output.write(b"frozen evidence\n")
            expected = hashlib.sha256(b"frozen evidence\n").hexdigest()
            self.assertEqual(
                evidence.require_sha256(path, expected, "fixture"), expected
            )
            with self.assertRaisesRegex(ValueError, "expected"):
                evidence.require_sha256(path, "0" * 64, "fixture")


if __name__ == "__main__":
    unittest.main()
