// RUN: %clang_cc1 -fsyntax-only %s
// RUN: false

// cci-bisect 20-commit exercise: inner commit 01/20 (passing test).
// cci-bisect 20-commit exercise: inner commit 12/20 (intentional failure).
int cci_bisect_20_inner(void) { return 0; }
