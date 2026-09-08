// RUN: %clang_cc1 -fsyntax-only %s

// cci-bisect exercise inner commit 1/4: passing Clang lit test.
// cci-bisect exercise inner commit 2/4: harmless marker.
int cci_bisect_merge_inner(void) { return 0; }
