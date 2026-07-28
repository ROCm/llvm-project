# Hotswap test selection

`hotswap_test_select.py` merges calibration observations and chooses a small,
high-value test subset. It is intended for quick corpus and A0 calibration
runs, where executing every duplicate or redundant test is too expensive.

The input is versioned JSON:

```json
{
  "kind": "hotswap-test-observations",
  "schema_version": 1,
  "tests": [
    {
      "id": "hipblaslt/matmul/example",
      "runtime_ms": 125,
      "features": ["branch-rewrite", "literal-displacement"],
      "candidate_outcome": "pass",
      "original_outcome": "pass",
      "novelty": 0.5,
      "risk": 2
    }
  ]
}
```

`candidate_outcome` and `original_outcome` accept `pass`, `fail`, `skip`,
`unknown`, or `mixed`. The outcomes, `novelty`, `risk`, `sample_count`, and
`has_candidate_only_regression` fields are optional. Test ID, runtime, and
features are required.

Merging takes the sample-weighted mean runtime, unions feature tags, retains
the maximum risk and novelty, and marks conflicting known outcomes as `mixed`.
It also retains `has_candidate_only_regression` when any constituent sample
failed with the candidate and passed with the original. This separate marker
prevents a later passing sample from hiding a previously observed regression
when the aggregate candidate outcome becomes `mixed`. A producer that supplies
an already aggregated `mixed` outcome should set the marker itself when it
knows such a paired observation occurred. These rules make a merge independent
of input order.

Rewrite-manifest, audit, and semantic-check producers can use a neutral
forward-compatible envelope:

```json
{
  "kind": "producer-specific-kind",
  "schema_version": 1,
  "source": "rewrite-manifest",
  "records": [
    {
      "test_id": "hipblaslt/matmul/example",
      "runtime_ms": 125,
      "features": ["branch-rewrite"],
      "candidate_outcome": "pass",
      "original_outcome": "pass",
      "risk": 2,
      "novelty": 0.5,
      "provenance": ["mi400-3"]
    }
  ]
}
```

Only `test_id` is required in each individual producer record. At least one
producer must report `runtime_ms` for every test; runtime and feature
observations may otherwise arrive from different producers. The adapter
ignores unknown producer fields so reports can evolve, then unions their
features and provenance:

```console
python3 amd/comgr/utils/hotswap/hotswap_test_select.py adapt \
  --input manifest.json audit.json semcheck.json \
  --output observations.json
```

The adapter also accepts the versioned `comgr.hotswap.inventory` schema. It
uses `sha256:<digest>` as the test ID, so manifest, audit, and semantic-check
records can join observations to an exact deduplicated code object without
depending on a path alias. The inventory does not measure runtime or infer
rewrite features. Join it with a timed producer:

```console
python3 amd/comgr/utils/hotswap/hotswap_test_select.py adapt \
  --input inventory.json audit-observations.json \
  --output observations.json
```

For exploratory use, `--default-runtime-ms` explicitly supplies the missing
runtime for inventory objects that no producer timed. The adapter marks that
provenance as `adapter-default-runtime`; it never silently treats an untimed
object as free.

Merge repeated calibration runs:

```console
python3 amd/comgr/utils/hotswap/hotswap_test_select.py merge \
  --input run-1.json run-2.json --output observations.json
```

Every merged input must describe the same candidate build, reference build,
target, and execution mode. Provenance is retained for auditability, but the
version 1 schema does not infer or enforce that comparison identity. Keep
observations from different candidate revisions in separate merges.

Select a subset:

```console
python3 amd/comgr/utils/hotswap/hotswap_test_select.py select \
  --input observations.json --runtime-budget-ms 30000 \
  --min-novel-tests 5 --min-unknown-tests 5 --output selection.json
```

The deterministic weighted set-cover heuristic first includes every test that
failed with the candidate and passed with the original. Those regressions may
exceed a requested budget. It then maximizes marginal feature coverage,
adjusted by risk, novelty, and calibration quotas. When a runtime budget is
supplied, the greedy score is also divided by runtime (with sub-millisecond
runtimes treated as one millisecond); without one, runtime is only a stable
tie-break. The output includes the canonical input digest, selection reason and
marginal coverage and provenance for every test, remaining uncovered features,
unfilled quotas, and estimated runtime.

Use repeated `--require-feature` options to cover only named features, and
repeated `--feature-weight FEATURE=WEIGHT` options to prioritize important
rewrite behavior. `--novelty-mode rarity` derives novelty from inverse feature
frequency, so previously unseen tag names need no selector changes.
`--holdout-percent` reserves a deterministic, seed-controlled subset of
non-regression test IDs for independent validation. Known candidate-only
regressions still override that reservation and are reported separately.

The selector is a greedy heuristic, not a globally optimal set-cover solver.
Its stable score and tie-break order make a given selection reproducible and
explainable. Output files are written through a same-directory temporary file
and atomically replaced. An output path that aliases an input is rejected
before any input is read or output is opened.

## PR #3646 evidence acceptance

`test_hotswap_test_select_evidence.py` can consume the frozen corpus evidence
from candidate commit `ab3cdd61c079a3e7fa7ed41d9834cc0baa256fcc`:

```console
COMGR_HOTSWAP_RESULTS_TSV=/path/to/results.tsv \
COMGR_HOTSWAP_TRANSITIONS_TSV=/path/to/transitions-vs-5cbc-default.tsv \
COMGR_HOTSWAP_PRIOR_FAILURES_TSV=/path/to/prior-failures-now-success.tsv \
python3 amd/comgr/test/utils/test_hotswap_test_select_evidence.py
```

The evidence test requires these SHA-256 values before parsing the files:

- `results.tsv`:
  `ffb089ac585374ac3e9edd93cb0668ed9c6c6b21514948d94884a20f7c2176b9`
- `transitions-vs-5cbc-default.tsv`:
  `295c97a7bcc7b62363c5f41e9c3aa9c46741d43f2270799407b6c6cc11908c5c`
- `prior-failures-now-success.tsv`:
  `756066cad7ca7aea97b5d9474784f5ad4ae00bc2a1b3d381bf0592f968cfb892`

The acceptance read 2,685 aliases, deduplicated them to 1,452 input hashes,
and therefore collapsed 1,233 duplicate aliases before selection. All rows
reported a successful first rewrite, successful second rewrite, and
byte-identical idempotence result. The runtime p99 was 6.64 seconds. Six
data-derived coverage dimensions were covered by two selected objects:

- the repaired scale16 case covered rewrite success, idempotence, changed
  output, the runtime tail, and a prior baseline failure;
- one unchanged-output `kpack` object supplied the remaining unchanged-output
  dimension.

The four natural transition records are `FAILURE` to `SUCCESS`: one scale16
case and three WMMA VGPR-MSB cases. They are improvements in the real evidence,
not candidate-only regressions. A separate acceptance step reverses only the
outcome orientation of those same four records and sets both budgets to zero.
That checks that all four are still selected and marked forced when presented
as candidate-only regressions. It does not claim that PR #3646 failed them.

This evidence establishes corpus rewrite completion, deduplication identity,
idempotence, output-change classes, measured host runtime, and old/new process
outcomes. It does not establish GPU execution accuracy, semantic equivalence,
or the presence of round-trip VCC preservation, owner-specific source
windows, far gateways, materialized call closure, scratch-SGPR liveness, DS2,
M32 scale16, or tensor-mask behavior. Those dimensions require manifest,
audit, semantic-check, or A0 execution observations.

`amd/comgr/test/Inputs/hotswap/pr3646-acceptance.json` remains a separate
curated fixture for those risk dimensions, natural parent/child hipBLASLt and
hipSOLVER records, and synthetic unseen holdout dimensions. It is not measured
performance evidence. The tests rewrite every ID and feature label to verify
that selection depends on record structure rather than memorized names. Run
`select --help` for all controls.
