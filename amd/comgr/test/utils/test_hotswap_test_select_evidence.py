#!/usr/bin/env python3
# ===- test_hotswap_test_select_evidence.py - Evidence acceptance -----------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===#

"""Optional selector acceptance test using the frozen PR #3646 evidence."""

import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path


EXPECTED_ALIASES = 2685
EXPECTED_UNIQUE = 1452
EXPECTED_PRIOR_FAILURES = 4
EXPECTED_P99_RUNTIME_SECONDS = 6.64
EXPECTED_SHA256 = {
    "prior failures": "756066cad7ca7aea97b5d9474784f5ad4ae00bc2a1b3d381bf0592f968cfb892",
    "results": "ffb089ac585374ac3e9edd93cb0668ed9c6c6b21514948d94884a20f7c2176b9",
    "transitions": "295c97a7bcc7b62363c5f41e9c3aa9c46741d43f2270799407b6c6cc11908c5c",
}
COMGR_DIR = Path(__file__).resolve().parents[2]
SCRIPT = COMGR_DIR / "utils" / "hotswap" / "hotswap_test_select.py"
SPEC = importlib.util.spec_from_file_location("hotswap_test_select", SCRIPT)
SELECTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SELECTOR)


def require_sha256(path, expected, label):
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        while True:
            block = input_file.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(
            "{} SHA-256 is {}, expected {}".format(label, actual, expected)
        )
    return actual


def fail(message):
    print("FAIL: " + message, file=sys.stderr)
    return 1


def read_tsv(path):
    with open(path, "r", encoding="utf-8", newline="") as input_file:
        return list(csv.DictReader(input_file, delimiter="\t"))


def load_evidence(results_path, transitions_path, prior_failures_path):
    require_sha256(results_path, EXPECTED_SHA256["results"], "results")
    require_sha256(transitions_path, EXPECTED_SHA256["transitions"], "transitions")
    require_sha256(
        prior_failures_path, EXPECTED_SHA256["prior failures"], "prior failures"
    )

    results = read_tsv(results_path)
    prior_failures = read_tsv(prior_failures_path)
    with open(transitions_path, "r", encoding="utf-8", newline="") as input_file:
        transitions = [
            tuple(line.rstrip("\n").split("\t"))
            for line in input_file
            if line.rstrip("\n")
        ]

    if len(results) != EXPECTED_ALIASES:
        raise ValueError(
            "results contain {} aliases, expected {}".format(
                len(results), EXPECTED_ALIASES
            )
        )
    if len(transitions) != EXPECTED_ALIASES:
        raise ValueError(
            "transitions contain {} aliases, expected {}".format(
                len(transitions), EXPECTED_ALIASES
            )
        )
    if any(len(transition) != 3 for transition in transitions):
        raise ValueError("transition rows must have three columns")
    transition_by_path = {
        path: (original, candidate) for path, original, candidate in transitions
    }
    if len(transition_by_path) != EXPECTED_ALIASES:
        raise ValueError("transition paths must be unique")

    required_result_fields = (
        "first_seconds",
        "first_result",
        "input_sha256",
        "output_sha256",
        "pass2_cmp_rc",
        "pass2_result",
        "relative_path",
        "result",
    )
    if any(field not in results[0] for field in required_result_fields):
        raise ValueError("results are missing required columns")
    if any(
        row["first_result"] != "SUCCESS"
        or row["pass2_result"] != "SUCCESS"
        or row["pass2_cmp_rc"] != "0"
        or row["result"] != "SUCCESS"
        for row in results
    ):
        raise ValueError("results contain a failed rewrite or idempotence check")

    prior_paths = {row["relative_path"] for row in prior_failures}
    if len(prior_paths) != EXPECTED_PRIOR_FAILURES:
        raise ValueError(
            "prior failures contain {} paths, expected {}".format(
                len(prior_paths), EXPECTED_PRIOR_FAILURES
            )
        )
    result_paths = {row["relative_path"] for row in results}
    if set(transition_by_path) != result_paths:
        raise ValueError("transition and result path sets differ")
    if not prior_paths <= result_paths:
        raise ValueError("a prior-failure path is missing from results")
    if any(transition_by_path[path] != ("FAILURE", "SUCCESS") for path in prior_paths):
        raise ValueError("prior failures do not have FAILURE-to-SUCCESS transitions")
    if (
        sum(
            transition == ("FAILURE", "SUCCESS")
            for transition in transition_by_path.values()
        )
        != EXPECTED_PRIOR_FAILURES
    ):
        raise ValueError("unexpected number of FAILURE-to-SUCCESS transitions")
    if (
        sum(
            transition == ("SUCCESS", "SUCCESS")
            for transition in transition_by_path.values()
        )
        != EXPECTED_ALIASES - EXPECTED_PRIOR_FAILURES
    ):
        raise ValueError("unexpected non-success transition")

    rows_by_digest = defaultdict(list)
    for row in results:
        rows_by_digest[row["input_sha256"]].append(row)
    if len(rows_by_digest) != EXPECTED_UNIQUE:
        raise ValueError(
            "results contain {} unique objects, expected {}".format(
                len(rows_by_digest), EXPECTED_UNIQUE
            )
        )

    runtimes = sorted(
        max(float(row["first_seconds"]) for row in rows)
        for rows in rows_by_digest.values()
    )
    p99_runtime = runtimes[math.ceil(0.99 * len(runtimes)) - 1]
    tests = []
    prior_test_ids = []
    for digest in sorted(rows_by_digest):
        rows = rows_by_digest[digest]
        aliases = sorted(row["relative_path"] for row in rows)
        prior_aliases = sorted(set(aliases) & prior_paths)
        test_id = prior_aliases[0] if prior_aliases else aliases[0]
        runtime_seconds = max(float(row["first_seconds"]) for row in rows)
        features = ["idempotent-rewrite", "rewrite-success"]
        if any(row["input_sha256"] != row["output_sha256"] for row in rows):
            features.append("output-changed")
        else:
            features.append("output-unchanged")
        if runtime_seconds >= p99_runtime:
            features.append("runtime-tail")
        if prior_aliases:
            features.append("prior-baseline-failure")
            prior_test_ids.append(test_id)
        tests.append(
            {
                "candidate_outcome": "pass",
                "features": features,
                "id": test_id,
                "novelty": 1 if runtime_seconds >= p99_runtime else 0,
                "original_outcome": "fail" if prior_aliases else "pass",
                "provenance": [
                    "pr3646-results",
                    "pr3646-transitions",
                ],
                "risk": 4 if prior_aliases else 0,
                "runtime_ms": runtime_seconds * 1000,
            }
        )

    document = {
        "kind": SELECTOR.OBSERVATION_KIND,
        "schema_version": SELECTOR.SCHEMA_VERSION,
        "tests": tests,
    }
    return document, sorted(prior_test_ids), p99_runtime


def main():
    results_path = os.environ.get("COMGR_HOTSWAP_RESULTS_TSV")
    transitions_path = os.environ.get("COMGR_HOTSWAP_TRANSITIONS_TSV")
    prior_failures_path = os.environ.get("COMGR_HOTSWAP_PRIOR_FAILURES_TSV")
    if not results_path or not transitions_path or not prior_failures_path:
        print(
            "SKIP: set COMGR_HOTSWAP_RESULTS_TSV, "
            "COMGR_HOTSWAP_TRANSITIONS_TSV, and "
            "COMGR_HOTSWAP_PRIOR_FAILURES_TSV",
            file=sys.stderr,
        )
        return 77

    try:
        document, prior_test_ids, p99_runtime = load_evidence(
            results_path, transitions_path, prior_failures_path
        )
        requested_features = [
            "idempotent-rewrite",
            "output-changed",
            "output-unchanged",
            "prior-baseline-failure",
            "rewrite-success",
            "runtime-tail",
        ]
        actual = SELECTOR.select_tests(
            document,
            requested_features=requested_features,
            test_count_budget=6,
            novelty_mode="combined",
        )
        if actual["uncovered_features"]:
            return fail(
                "real-evidence selection left features uncovered: {}".format(
                    actual["uncovered_features"]
                )
            )
        if actual["known_candidate_only_regressions"]:
            return fail(
                "real evidence incorrectly classified repaired baseline failures "
                "as candidate-only regressions"
            )
        if len(actual["selected_tests"]) != 2:
            return fail(
                "real-evidence selection chose {} objects, expected 2".format(
                    len(actual["selected_tests"])
                )
            )
        if not math.isclose(
            p99_runtime, EXPECTED_P99_RUNTIME_SECONDS, rel_tol=0, abs_tol=0.005
        ):
            return fail(
                "runtime p99 is {:.9g} seconds, expected {:.2f}".format(
                    p99_runtime, EXPECTED_P99_RUNTIME_SECONDS
                )
            )

        candidate_regression_document = json.loads(json.dumps(document))
        records_by_id = {
            record["id"]: record for record in candidate_regression_document["tests"]
        }
        for test_id in prior_test_ids:
            records_by_id[test_id]["candidate_outcome"] = "fail"
            records_by_id[test_id]["original_outcome"] = "pass"
        forced = SELECTOR.select_tests(
            candidate_regression_document,
            runtime_budget_ms=0,
            test_count_budget=0,
        )
        if forced["known_candidate_only_regressions"] != prior_test_ids:
            return fail("candidate-only orientation did not retain all four cases")
        if [record["id"] for record in forced["selected_tests"]] != prior_test_ids:
            return fail("forced selection did not contain exactly the four cases")
        if not all(record["forced"] for record in forced["selected_tests"]):
            return fail("a candidate-only regression was not marked forced")
        if not forced["budget_exceeded_by_forced_regressions"]:
            return fail("forced regressions did not report their budget override")
    except (OSError, UnicodeError, ValueError, SELECTOR.SelectionError) as error:
        return fail(str(error))

    summary = {
        "aliases": EXPECTED_ALIASES,
        "duplicate_aliases_collapsed": EXPECTED_ALIASES - EXPECTED_UNIQUE,
        "evidence_sha256": EXPECTED_SHA256,
        "forced_retention_cases": prior_test_ids,
        "p99_runtime_seconds": p99_runtime,
        "real_evidence_selected": [
            {
                "id": record["id"],
                "marginal_features": record["marginal_features"],
            }
            for record in actual["selected_tests"]
        ],
        "unique_code_objects": EXPECTED_UNIQUE,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
