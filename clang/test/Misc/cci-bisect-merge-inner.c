// RUN: %clang_cc1 -fsyntax-only %s
// RUN: false

// cci-bisect exercise inner commit 1/4: passing Clang lit test.
// cci-bisect exercise inner commit 2/4: harmless marker.
// cci-bisect exercise inner commit 3/4: intentional Clang lit failure.
// cci-bisect exercise inner commit 4/4: harmless marker after the failure.
int cci_bisect_merge_inner(void) { return 0; }
