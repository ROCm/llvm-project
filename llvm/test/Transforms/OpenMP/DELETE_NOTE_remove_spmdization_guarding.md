Delete note — spmdization_guarding.ll

Decision summary (2026-08-12):
- The `spmdization_guarding.ll` test (and related device-RT code touched by commit 674ffbd4cd36)
  produce cosmetic CHECK mismatches when validated with the ROCm "srock" compiler toolchain.
- The mismatch is textual (addrspace/select form and attribute renumbering); the generated IR
  is semantically equivalent for runtime execution.
- The same test file exists upstream and shows the same CHECK differences when using the
  srock toolchain.

Planned action:
- Remove `spmdization_guarding.ll` from the ROCm fork to avoid merge conflicts and duplicated,
  stale device-RT expectations. This effectively deletes the carried feature for ROCm.
- Owner will confirm with Ron if a PR is required; do not proceed with removal until confirmation.

Reference:
- commit: 674ffbd4cd364a4d85501bd6db10ae7a547ee896 (original author: Greg Rodgers)

