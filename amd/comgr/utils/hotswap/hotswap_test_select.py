#!/usr/bin/env python3
# ===- hotswap_test_select.py - Select hotswap calibration tests -----------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===#

"""Merge hotswap observations and select a small, high-value test subset.

The selector is deliberately deterministic and dependency-free. It uses a
weighted greedy set-cover heuristic, always retaining known candidate-only
regressions. Its JSON output records why each test was selected.
"""

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import defaultdict


SCHEMA_VERSION = 1
OBSERVATION_KIND = "hotswap-test-observations"
SELECTION_KIND = "hotswap-test-selection"
INVENTORY_SCHEMA = "comgr.hotswap.inventory"
INVENTORY_VERSION = 1
OUTCOMES = frozenset(("pass", "fail", "skip", "unknown", "mixed"))
DOCUMENT_FIELDS = frozenset(("kind", "schema_version", "tests"))
TEST_FIELDS = frozenset(
    (
        "candidate_outcome",
        "features",
        "has_candidate_only_regression",
        "id",
        "novelty",
        "original_outcome",
        "provenance",
        "risk",
        "runtime_ms",
        "sample_count",
    )
)


class SelectionError(ValueError):
    """An invalid observation or selection option."""


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_weight(value, name):
    if not _is_number(value):
        raise SelectionError("%s must be a finite non-negative number" % name)
    try:
        normalized = float(value)
    except OverflowError as error:
        raise SelectionError(
            "%s must be a finite non-negative number" % name
        ) from error
    if not math.isfinite(normalized) or normalized < 0:
        raise SelectionError("%s must be a finite non-negative number" % name)
    return normalized


def _canonical_number(value):
    try:
        normalized = float(value)
    except OverflowError as error:
        raise SelectionError("computed numeric result must be finite") from error
    if not math.isfinite(normalized):
        raise SelectionError("computed numeric result must be finite")
    if normalized.is_integer():
        return int(normalized)
    return normalized


def _finite_sum(values, name):
    try:
        result = math.fsum(values)
    except (OverflowError, ValueError) as error:
        raise SelectionError("%s exceeds the finite numeric range" % name) from error
    if not math.isfinite(result):
        raise SelectionError("%s exceeds the finite numeric range" % name)
    return result


def _finite_product(left, right, name):
    result = float(left) * float(right)
    if not math.isfinite(result):
        raise SelectionError("%s exceeds the finite numeric range" % name)
    return result


def _weighted_runtime(records):
    sample_count = sum(record["sample_count"] for record in records)
    weighted_samples = sorted(
        float(record["runtime_ms"]) * (record["sample_count"] / sample_count)
        for record in records
    )
    runtime_ms = _finite_sum(
        weighted_samples,
        "weighted runtime",
    )
    return runtime_ms, sample_count


def _normalize_outcome(value, field, test_id):
    if value is None:
        return "unknown"
    if not isinstance(value, str) or value not in OUTCOMES:
        raise SelectionError(
            "test %r field %s must be one of %s"
            % (test_id, field, ", ".join(sorted(OUTCOMES)))
        )
    return value


def _normalize_test(record, index):
    if not isinstance(record, dict):
        raise SelectionError("test record %d must be an object" % index)
    unknown_fields = sorted(set(record) - TEST_FIELDS)
    if unknown_fields:
        raise SelectionError(
            "test record %d has unknown field(s): %s"
            % (index, ", ".join(unknown_fields))
        )

    test_id = record.get("id")
    if not isinstance(test_id, str) or not test_id:
        raise SelectionError("test record %d has a missing or empty id" % index)

    if "runtime_ms" not in record:
        raise SelectionError("test %r is missing runtime_ms" % test_id)
    runtime_ms = _validate_weight(record["runtime_ms"], "test %r runtime_ms" % test_id)

    features = record.get("features")
    if not isinstance(features, list):
        raise SelectionError("test %r features must be an array" % test_id)
    for feature in features:
        if not isinstance(feature, str) or not feature:
            raise SelectionError(
                "test %r features must contain non-empty strings" % test_id
            )

    sample_count = record.get("sample_count", 1)
    if (
        not isinstance(sample_count, int)
        or isinstance(sample_count, bool)
        or sample_count < 1
    ):
        raise SelectionError(
            "test %r sample_count must be a positive integer" % test_id
        )

    provenance = record.get("provenance", [])
    if not isinstance(provenance, list):
        raise SelectionError("test %r provenance must be an array" % test_id)
    for source in provenance:
        if not isinstance(source, str) or not source:
            raise SelectionError(
                "test %r provenance must contain non-empty strings" % test_id
            )

    candidate_outcome = _normalize_outcome(
        record.get("candidate_outcome"), "candidate_outcome", test_id
    )
    original_outcome = _normalize_outcome(
        record.get("original_outcome"), "original_outcome", test_id
    )
    has_candidate_only_regression = record.get("has_candidate_only_regression", False)
    if not isinstance(has_candidate_only_regression, bool):
        raise SelectionError(
            "test %r has_candidate_only_regression must be a boolean" % test_id
        )
    has_candidate_only_regression = has_candidate_only_regression or (
        candidate_outcome == "fail" and original_outcome == "pass"
    )

    return {
        "candidate_outcome": candidate_outcome,
        "features": sorted(set(features)),
        "has_candidate_only_regression": has_candidate_only_regression,
        "id": test_id,
        "novelty": _canonical_number(
            _validate_weight(record.get("novelty", 0), "test %r novelty" % test_id)
        ),
        "original_outcome": original_outcome,
        "provenance": sorted(set(provenance)),
        "risk": _canonical_number(
            _validate_weight(record.get("risk", 0), "test %r risk" % test_id)
        ),
        "runtime_ms": _canonical_number(runtime_ms),
        "sample_count": sample_count,
    }


def normalize_document(document):
    """Validate and canonicalize one observation document."""
    if not isinstance(document, dict):
        raise SelectionError("observation document must be an object")
    unknown_fields = sorted(set(document) - DOCUMENT_FIELDS)
    if unknown_fields:
        raise SelectionError(
            "observation document has unknown field(s): %s" % ", ".join(unknown_fields)
        )
    schema_version = document.get("schema_version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != SCHEMA_VERSION
    ):
        raise SelectionError("schema_version must be %d" % SCHEMA_VERSION)
    kind = document.get("kind", OBSERVATION_KIND)
    if kind != OBSERVATION_KIND:
        raise SelectionError("kind must be %r" % OBSERVATION_KIND)
    tests = document.get("tests")
    if not isinstance(tests, list):
        raise SelectionError("tests must be an array")
    return merge_documents(
        [
            {
                "kind": OBSERVATION_KIND,
                "schema_version": SCHEMA_VERSION,
                "tests": [
                    _normalize_test(record, index) for index, record in enumerate(tests)
                ],
            }
        ],
        already_normalized=True,
    )


def _merge_outcomes(outcomes):
    meaningful = set(outcomes) - set(("unknown",))
    if not meaningful:
        return "unknown"
    if len(meaningful) == 1:
        return next(iter(meaningful))
    return "mixed"


def merge_documents(documents, already_normalized=False):
    """Merge observation documents with deterministic, order-independent rules."""
    records_by_id = defaultdict(list)
    for document in documents:
        normalized = document if already_normalized else normalize_document(document)
        for record in normalized["tests"]:
            records_by_id[record["id"]].append(record)

    merged_tests = []
    for test_id in sorted(records_by_id):
        records = records_by_id[test_id]
        runtime_ms, sample_count = _weighted_runtime(records)
        features = sorted(
            set(feature for record in records for feature in record["features"])
        )
        merged_tests.append(
            {
                "candidate_outcome": _merge_outcomes(
                    record["candidate_outcome"] for record in records
                ),
                "features": features,
                "has_candidate_only_regression": any(
                    record["has_candidate_only_regression"] for record in records
                ),
                "id": test_id,
                "novelty": max(record["novelty"] for record in records),
                "original_outcome": _merge_outcomes(
                    record["original_outcome"] for record in records
                ),
                "provenance": sorted(
                    set(source for record in records for source in record["provenance"])
                ),
                "risk": max(record["risk"] for record in records),
                "runtime_ms": _canonical_number(runtime_ms),
                "sample_count": sample_count,
            }
        )
    return {
        "kind": OBSERVATION_KIND,
        "schema_version": SCHEMA_VERSION,
        "tests": merged_tests,
    }


def _normalize_raw_record(record, index, source):
    if not isinstance(record, dict):
        raise SelectionError("producer record %d must be an object" % index)
    test_id = record.get("test_id", record.get("id"))
    if "test_id" in record and "id" in record and record["test_id"] != record["id"]:
        raise SelectionError(
            "producer record %d has conflicting test_id and id" % index
        )
    if (
        "features" in record
        and "feature_tags" in record
        and record["features"] != record["feature_tags"]
    ):
        raise SelectionError(
            "producer record %d has conflicting features and feature_tags" % index
        )

    provenance = record.get("provenance", [])
    if isinstance(provenance, str):
        provenance = [provenance]
    elif not isinstance(provenance, list):
        raise SelectionError(
            "producer record %d provenance must be a string or array" % index
        )
    if source not in provenance:
        provenance = list(provenance) + [source]

    translated = {
        "candidate_outcome": record.get("candidate_outcome"),
        "features": record.get("features", record.get("feature_tags", [])),
        "has_candidate_only_regression": record.get(
            "has_candidate_only_regression", False
        ),
        "id": test_id,
        "novelty": record.get("novelty", 0),
        "original_outcome": record.get("original_outcome"),
        "provenance": provenance,
        "risk": record.get("risk", 0),
        "runtime_ms": record.get("runtime_ms", 0),
        "sample_count": record.get("sample_count", 1),
    }
    return _normalize_test(translated, index), "runtime_ms" in record


def _adapt_inventory(document):
    version = document.get("version")
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version != INVENTORY_VERSION
    ):
        raise SelectionError("inventory version must be %d" % INVENTORY_VERSION)
    objects = document.get("objects")
    if not isinstance(objects, list):
        raise SelectionError("inventory objects must be an array")

    records = []
    for index, item in enumerate(objects):
        if not isinstance(item, dict):
            raise SelectionError("inventory object %d must be an object" % index)
        digest = item.get("sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise SelectionError("inventory object %d has an invalid sha256" % index)
        records.append(
            _normalize_raw_record(
                {
                    "test_id": "sha256:" + digest,
                    "provenance": [INVENTORY_SCHEMA],
                },
                index,
                INVENTORY_SCHEMA,
            )
        )
    return records


def adapt_documents(documents, default_runtime_ms=None):
    """Adapt forward-compatible producer reports to canonical observations."""
    if default_runtime_ms is not None:
        default_runtime_ms = _validate_weight(default_runtime_ms, "default runtime")
    adapted_records = []
    for document_index, document in enumerate(documents):
        if not isinstance(document, dict):
            raise SelectionError("producer document must be an object")
        if document.get("schema") == INVENTORY_SCHEMA:
            adapted_records.extend(_adapt_inventory(document))
            continue
        if document.get("kind") == OBSERVATION_KIND:
            normalized = normalize_document(document)
            adapted_records.extend((record, True) for record in normalized["tests"])
            continue
        schema_version = document.get("schema_version")
        if (
            not isinstance(schema_version, int)
            or isinstance(schema_version, bool)
            or schema_version != SCHEMA_VERSION
        ):
            raise SelectionError("producer schema_version must be %d" % SCHEMA_VERSION)
        source = document.get("source", document.get("kind"))
        if not isinstance(source, str) or not source:
            source = "input-%d" % document_index
        if (
            "records" in document
            and "tests" in document
            and document["records"] != document["tests"]
        ):
            raise SelectionError(
                "producer document has conflicting records and tests arrays"
            )
        records = document.get("records", document.get("tests"))
        if not isinstance(records, list):
            raise SelectionError("producer records must be an array")
        adapted_records.extend(
            _normalize_raw_record(record, index, source)
            for index, record in enumerate(records)
        )

    records_by_id = defaultdict(list)
    for record, has_runtime in adapted_records:
        records_by_id[record["id"]].append((record, has_runtime))

    merged = []
    for test_id in sorted(records_by_id):
        entries = records_by_id[test_id]
        records = [entry[0] for entry in entries]
        timed_records = [entry[0] for entry in entries if entry[1]]
        if timed_records:
            runtime_ms, sample_count = _weighted_runtime(timed_records)
        else:
            if default_runtime_ms is None:
                raise SelectionError(
                    "test %r has no runtime_ms in any producer report" % test_id
                )
            sample_count = 1
            runtime_ms = default_runtime_ms
            for record in records:
                if "adapter-default-runtime" not in record["provenance"]:
                    record["provenance"].append("adapter-default-runtime")
        merged.append(
            {
                "candidate_outcome": _merge_outcomes(
                    record["candidate_outcome"] for record in records
                ),
                "features": sorted(
                    set(feature for record in records for feature in record["features"])
                ),
                "has_candidate_only_regression": any(
                    record["has_candidate_only_regression"] for record in records
                ),
                "id": test_id,
                "novelty": max(record["novelty"] for record in records),
                "original_outcome": _merge_outcomes(
                    record["original_outcome"] for record in records
                ),
                "provenance": sorted(
                    set(source for record in records for source in record["provenance"])
                ),
                "risk": max(record["risk"] for record in records),
                "runtime_ms": _canonical_number(runtime_ms),
                "sample_count": sample_count,
            }
        )
    return {
        "kind": OBSERVATION_KIND,
        "schema_version": SCHEMA_VERSION,
        "tests": merged,
    }


def observation_digest(document):
    try:
        canonical = json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise SelectionError("cannot compute digest for observation JSON") from error
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def _is_regression(record):
    return record["has_candidate_only_regression"]


def _has_unknown_outcome(record):
    return (
        record["candidate_outcome"] == "unknown"
        or record["original_outcome"] == "unknown"
    )


def _is_holdout(test_id, percent, seed):
    if percent <= 0:
        return False
    key = (seed + "\0" + test_id).encode("utf-8")
    bucket = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
    return bucket < percent * (1 << 64) / 100.0


def _effective_novelties(records, mode):
    feature_counts = defaultdict(int)
    for record in records:
        for feature in record["features"]:
            feature_counts[feature] += 1
    novelties = {}
    for record in records:
        rarity = _finite_sum(
            (1.0 / feature_counts[feature] for feature in record["features"]),
            "effective novelty",
        )
        reported = float(record["novelty"])
        if mode == "reported":
            effective = reported
        elif mode == "rarity":
            effective = rarity
        else:
            effective = _finite_sum((reported, rarity), "effective novelty")
        novelties[record["id"]] = effective
    return novelties


def _fits_budget(
    record, selected_count, runtime_ms, test_count_budget, runtime_budget_ms
):
    if test_count_budget is not None and selected_count + 1 > test_count_budget:
        return False
    if runtime_budget_ms is not None and (
        runtime_ms > runtime_budget_ms
        or record["runtime_ms"] > runtime_budget_ms - runtime_ms
    ):
        return False
    return True


def _candidate_value(
    record,
    effective_novelty,
    uncovered,
    novel_needed,
    unknown_needed,
    feature_weights,
    risk_weight,
    novelty_weight,
    unknown_weight,
):
    marginal = sorted(set(record["features"]) & uncovered)
    coverage_value = _finite_sum(
        (feature_weights.get(feature, 1.0) for feature in marginal),
        "feature coverage score",
    )
    quota_value = 0.0
    if novel_needed and effective_novelty > 0:
        quota_value = _finite_sum(
            (
                quota_value,
                _finite_product(
                    novelty_weight, effective_novelty, "novelty quota score"
                ),
            ),
            "quota score",
        )
    if unknown_needed and _has_unknown_outcome(record):
        quota_value = _finite_sum((quota_value, unknown_weight), "quota score")
    if coverage_value == 0 and quota_value == 0:
        return 0.0, coverage_value, marginal

    risk_multiplier = _finite_sum(
        (
            1.0,
            _finite_product(risk_weight, record["risk"], "risk score"),
        ),
        "risk multiplier",
    )
    weighted_values = []
    if coverage_value > 0:
        coverage_multiplier = _finite_sum(
            (
                risk_multiplier,
                _finite_product(
                    novelty_weight, effective_novelty, "novelty coverage score"
                ),
            ),
            "coverage multiplier",
        )
        weighted_values.append(
            _finite_product(
                coverage_value, coverage_multiplier, "weighted coverage score"
            )
        )
    if quota_value > 0:
        weighted_values.append(
            _finite_product(quota_value, risk_multiplier, "weighted quota score"),
        )
    value = _finite_sum(weighted_values, "selection score")
    return value, coverage_value, marginal


def select_tests(
    document,
    requested_features=None,
    feature_weights=None,
    runtime_budget_ms=None,
    test_count_budget=None,
    min_novel_tests=0,
    min_unknown_tests=0,
    risk_weight=1.0,
    novelty_weight=1.0,
    unknown_weight=1.0,
    novelty_mode="reported",
    holdout_percent=0.0,
    holdout_seed="hotswap-test-select-v1",
):
    """Return a versioned, explainable selection document."""
    observations = normalize_document(document)
    feature_weights = dict(feature_weights or {})
    requested_features = list(requested_features or [])

    if runtime_budget_ms is not None:
        runtime_budget_ms = _validate_weight(runtime_budget_ms, "runtime budget")
    if test_count_budget is not None and (
        not isinstance(test_count_budget, int)
        or isinstance(test_count_budget, bool)
        or test_count_budget < 0
    ):
        raise SelectionError("test-count budget must be a non-negative integer")
    for name, count in (
        ("min novel tests", min_novel_tests),
        ("min unknown tests", min_unknown_tests),
    ):
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise SelectionError("%s must be a non-negative integer" % name)
    risk_weight = _validate_weight(risk_weight, "risk weight")
    novelty_weight = _validate_weight(novelty_weight, "novelty weight")
    unknown_weight = _validate_weight(unknown_weight, "unknown weight")
    if novelty_mode not in ("reported", "rarity", "combined"):
        raise SelectionError("novelty mode must be reported, rarity, or combined")
    holdout_percent = _validate_weight(holdout_percent, "holdout percent")
    if holdout_percent > 100:
        raise SelectionError("holdout percent must not exceed 100")
    if not isinstance(holdout_seed, str):
        raise SelectionError("holdout seed must be a string")

    normalized_weights = {}
    for feature, weight in feature_weights.items():
        if not isinstance(feature, str) or not feature:
            raise SelectionError("feature-weight names must be non-empty strings")
        normalized_weights[feature] = _validate_weight(
            weight, "feature weight %r" % feature
        )

    all_records = observations["tests"]
    regression_ids = sorted(
        record["id"] for record in all_records if _is_regression(record)
    )
    holdout_ids = sorted(
        record["id"]
        for record in all_records
        if _is_holdout(record["id"], holdout_percent, holdout_seed)
    )
    regression_id_set = set(regression_ids)
    holdout_id_set = set(holdout_ids)
    eligible_records = [
        record
        for record in all_records
        if record["id"] not in holdout_id_set or record["id"] in regression_id_set
    ]
    effective_novelties = _effective_novelties(eligible_records, novelty_mode)

    all_features = sorted(
        set(feature for record in eligible_records for feature in record["features"])
    )
    for feature in requested_features:
        if not isinstance(feature, str) or not feature:
            raise SelectionError("requested features must be non-empty strings")
    targets = sorted(set(requested_features)) if requested_features else all_features

    records_by_id = {record["id"]: record for record in all_records}

    selected = []
    selected_ids = set()
    covered = set()
    runtime_ms = 0.0
    novel_count = 0
    unknown_count = 0

    def add_selection(record, forced, score, marginal):
        nonlocal novel_count, runtime_ms, unknown_count
        effective_novelty = effective_novelties[record["id"]]
        reasons = []
        if forced:
            reasons.append("candidate-only-regression")
        reasons.extend("covers:" + feature for feature in marginal)
        if effective_novelty > 0 and novel_count < min_novel_tests:
            reasons.append("novel-case")
        if _has_unknown_outcome(record) and unknown_count < min_unknown_tests:
            reasons.append("unknown-outcome")

        runtime_ms = _finite_sum(
            (runtime_ms, float(record["runtime_ms"])), "selected runtime"
        )
        covered.update(marginal)
        if effective_novelty > 0:
            novel_count += 1
        if _has_unknown_outcome(record):
            unknown_count += 1
        selected_ids.add(record["id"])
        selected.append(
            {
                "cumulative_runtime_ms": _canonical_number(runtime_ms),
                "effective_novelty": _canonical_number(effective_novelty),
                "forced": forced,
                "id": record["id"],
                "marginal_features": marginal,
                "provenance": record["provenance"],
                "reasons": reasons,
                "runtime_ms": record["runtime_ms"],
                "selection_score": _canonical_number(score),
            }
        )

    for test_id in regression_ids:
        record = records_by_id[test_id]
        marginal = sorted(set(record["features"]) & (set(targets) - covered))
        add_selection(record, True, 0.0, marginal)

    while True:
        uncovered = set(targets) - covered
        novel_needed = novel_count < min_novel_tests
        unknown_needed = unknown_count < min_unknown_tests
        if not uncovered and not novel_needed and not unknown_needed:
            break

        candidates = []
        for record in eligible_records:
            if record["id"] in selected_ids:
                continue
            if not _fits_budget(
                record,
                len(selected),
                runtime_ms,
                test_count_budget,
                runtime_budget_ms,
            ):
                continue
            value, coverage_value, marginal = _candidate_value(
                record,
                effective_novelties[record["id"]],
                uncovered,
                novel_needed,
                unknown_needed,
                normalized_weights,
                risk_weight,
                novelty_weight,
                unknown_weight,
            )
            if value <= 0:
                continue
            divisor = (
                max(float(record["runtime_ms"]), 1.0)
                if runtime_budget_ms is not None
                else 1.0
            )
            candidates.append(
                (
                    -(value / divisor),
                    -value,
                    -coverage_value,
                    float(record["runtime_ms"]),
                    record["id"],
                    record,
                    marginal,
                )
            )
        if not candidates:
            break
        candidates.sort()
        best = candidates[0]
        add_selection(best[5], False, -best[1], best[6])

    uncovered = sorted(set(targets) - covered)
    estimated_runtime = _canonical_number(runtime_ms)
    forced_runtime_exceeded = False
    if runtime_budget_ms is not None:
        remaining_runtime = runtime_budget_ms
        for test_id in regression_ids:
            forced_runtime = float(records_by_id[test_id]["runtime_ms"])
            if forced_runtime > remaining_runtime:
                forced_runtime_exceeded = True
                break
            remaining_runtime -= forced_runtime
    forced_exceeded = (
        test_count_budget is not None and len(regression_ids) > test_count_budget
    ) or forced_runtime_exceeded
    return {
        "budget": {
            "runtime_ms": (
                None
                if runtime_budget_ms is None
                else _canonical_number(runtime_budget_ms)
            ),
            "test_count": test_count_budget,
        },
        "budget_exceeded_by_forced_regressions": forced_exceeded,
        "estimated_runtime_ms": estimated_runtime,
        "input_digest": observation_digest(observations),
        "kind": SELECTION_KIND,
        "known_candidate_only_regressions": regression_ids,
        "holdout": {
            "percent": _canonical_number(holdout_percent),
            "regression_overrides": sorted(holdout_id_set & regression_id_set),
            "reserved_tests": sorted(holdout_id_set - regression_id_set),
            "seed": holdout_seed,
        },
        "novelty_mode": novelty_mode,
        "requested_features": targets,
        "schema_version": SCHEMA_VERSION,
        "selection_policy": {
            "feature_weights": {
                feature: _canonical_number(weight)
                for feature, weight in sorted(normalized_weights.items())
            },
            "min_novel_tests": min_novel_tests,
            "min_unknown_tests": min_unknown_tests,
            "novelty_weight": _canonical_number(novelty_weight),
            "risk_weight": _canonical_number(risk_weight),
            "unknown_weight": _canonical_number(unknown_weight),
        },
        "selected_tests": selected,
        "uncovered_features": uncovered,
        "unfilled_quotas": {
            "novel_tests": max(0, min_novel_tests - novel_count),
            "unknown_tests": max(0, min_unknown_tests - unknown_count),
        },
    }


def _read_json(path):
    try:
        if path == "-":
            return json.load(sys.stdin)
        with open(path, "r", encoding="utf-8") as input_file:
            return json.load(input_file)
    except (UnicodeError, ValueError) as error:
        raise SelectionError("cannot parse JSON from %r: %s" % (path, error)) from error


def _normalized_path(path):
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


def _protect_inputs(output_path, input_paths):
    if output_path == "-":
        return
    normalized_output = _normalized_path(output_path)
    for input_path in input_paths:
        if input_path == "-":
            continue
        normalized_input = _normalized_path(input_path)
        aliases_input = normalized_output == normalized_input
        if not aliases_input:
            try:
                aliases_input = os.path.samefile(output_path, input_path)
            except OSError:
                aliases_input = False
        if aliases_input:
            raise SelectionError("refusing to overwrite input %r" % input_path)


def _write_json(path, document):
    try:
        text = (
            json.dumps(
                document,
                allow_nan=False,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    except (TypeError, ValueError) as error:
        raise SelectionError("cannot serialize selection JSON: %s" % error) from error
    if path == "-":
        sys.stdout.write(text)
        return
    destination = os.path.abspath(path)
    directory = os.path.dirname(destination)
    if not os.path.isdir(directory):
        raise SelectionError("output directory does not exist: %r" % directory)
    temporary_path = None
    try:
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=".hotswap-test-select-", dir=directory
        )
        with os.fdopen(descriptor, "wb") as output_file:
            output_file.write(text.encode("ascii"))
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
    except OSError as error:
        raise SelectionError("cannot write %r: %s" % (destination, error)) from error
    finally:
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass


def _parse_feature_weights(values):
    weights = {}
    for value in values:
        if "=" not in value:
            raise SelectionError(
                "feature weights must use FEATURE=WEIGHT syntax: %r" % value
            )
        feature, raw_weight = value.split("=", 1)
        if not feature:
            raise SelectionError("feature-weight names must be non-empty")
        try:
            weight = float(raw_weight)
        except ValueError as error:
            raise SelectionError(
                "feature weight %r is not a number" % raw_weight
            ) from error
        weights[feature] = _validate_weight(weight, "feature weight %r" % feature)
    return weights


def _create_parser():
    parser = argparse.ArgumentParser(
        description=(
            "merge hotswap calibration observations or greedily select a "
            "small, explainable test subset"
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    adapt_parser = subparsers.add_parser(
        "adapt", help="normalize producer reports into observation JSON"
    )
    adapt_parser.add_argument(
        "--input", nargs="+", required=True, help="producer report JSON file(s)"
    )
    adapt_parser.add_argument(
        "--output", default="-", help="output JSON file (default: stdout)"
    )
    adapt_parser.add_argument(
        "--default-runtime-ms",
        type=float,
        help=(
            "explicit runtime for records not timed by any producer "
            "(required for a standalone inventory)"
        ),
    )

    merge_parser = subparsers.add_parser(
        "merge", help="merge one or more observation JSON files"
    )
    merge_parser.add_argument(
        "--input", nargs="+", required=True, help="observation JSON file(s)"
    )
    merge_parser.add_argument(
        "--output", default="-", help="output JSON file (default: stdout)"
    )

    select_parser = subparsers.add_parser(
        "select", help="select tests from an observation JSON file"
    )
    select_parser.add_argument("--input", required=True, help="observation JSON")
    select_parser.add_argument(
        "--output", default="-", help="output JSON file (default: stdout)"
    )
    select_parser.add_argument(
        "--runtime-budget-ms", type=float, help="optional total runtime budget"
    )
    select_parser.add_argument(
        "--test-count-budget", type=int, help="optional test-count budget"
    )
    select_parser.add_argument(
        "--require-feature",
        action="append",
        default=[],
        help="feature to cover; repeat as needed (default: cover all)",
    )
    select_parser.add_argument(
        "--feature-weight",
        action="append",
        default=[],
        metavar="FEATURE=WEIGHT",
        help="coverage weight; repeat as needed",
    )
    select_parser.add_argument(
        "--min-novel-tests",
        type=int,
        default=0,
        help="minimum selected tests with nonzero novelty",
    )
    select_parser.add_argument(
        "--min-unknown-tests",
        type=int,
        default=0,
        help="minimum selected tests with an unknown calibration outcome",
    )
    select_parser.add_argument(
        "--risk-weight", type=float, default=1.0, help="per-test risk multiplier"
    )
    select_parser.add_argument(
        "--novelty-weight",
        type=float,
        default=1.0,
        help="novel-case quota weight",
    )
    select_parser.add_argument(
        "--unknown-weight",
        type=float,
        default=1.0,
        help="unknown-outcome quota weight",
    )
    select_parser.add_argument(
        "--novelty-mode",
        choices=("reported", "rarity", "combined"),
        default="reported",
        help="use reported novelty, feature rarity, or both",
    )
    select_parser.add_argument(
        "--holdout-percent",
        type=float,
        default=0.0,
        help="deterministically reserve this percentage of non-regressions",
    )
    select_parser.add_argument(
        "--holdout-seed",
        default="hotswap-test-select-v1",
        help="stable seed for deterministic holdout partitioning",
    )
    return parser


def main(argv=None):
    parser = _create_parser()
    arguments = parser.parse_args(argv)
    try:
        input_paths = (
            arguments.input if isinstance(arguments.input, list) else [arguments.input]
        )
        _protect_inputs(arguments.output, input_paths)
        if arguments.command in ("adapt", "merge"):
            if arguments.input.count("-") > 1:
                raise SelectionError("standard input may only be read once")
            documents = [_read_json(path) for path in arguments.input]
            if arguments.command == "adapt":
                result = adapt_documents(
                    documents,
                    default_runtime_ms=arguments.default_runtime_ms,
                )
            else:
                result = merge_documents(documents)
        else:
            result = select_tests(
                _read_json(arguments.input),
                requested_features=arguments.require_feature,
                feature_weights=_parse_feature_weights(arguments.feature_weight),
                runtime_budget_ms=arguments.runtime_budget_ms,
                test_count_budget=arguments.test_count_budget,
                min_novel_tests=arguments.min_novel_tests,
                min_unknown_tests=arguments.min_unknown_tests,
                risk_weight=arguments.risk_weight,
                novelty_weight=arguments.novelty_weight,
                unknown_weight=arguments.unknown_weight,
                novelty_mode=arguments.novelty_mode,
                holdout_percent=arguments.holdout_percent,
                holdout_seed=arguments.holdout_seed,
            )
        _write_json(arguments.output, result)
    except (OSError, json.JSONDecodeError, SelectionError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    sys.exit(main())
