// REQUIRES: comgr-has-spirv-translator

// COM: Regression test: COMGR's in-process SPIR-V translator must honor the
// COM: --spirv-* flags that the HIPAMD clang driver emits. In particular,
// COM: HIPAMD passes --spirv-ext=+all,-SPV_KHR_untyped_pointers; if COMGR
// COM: ignores the flag and re-enables all extensions, the resulting SPIR-V
// COM: contains untyped pointers and downstream BC linking fails.

// COM: Run COMGR's HIP-source-to-SPIR-V pipeline (this exercises the same
// COM: clang driver path as hipRTC, which inserts the --spirv-ext flag).
// RUN: source-to-spirv -nogpuinc %s %t.spv

// COM: SPIR-V file should exist and be non-empty.
// RUN: test -s %t.spv

// COM: OpExtension declarations are length-prefixed ASCII strings in the
// COM: binary, so grep can verify presence/absence directly without needing
// COM: SPIRV-Tools or --to-text support in the build.
// RUN: not grep -a "SPV_KHR_untyped_pointers" %t.spv

__attribute__((device))
void clean_value(float* ptr) { *ptr = 0; }

__attribute__((global))
void add_value(float* a, float* b, float* res) {
    *res = *a + *b;
    clean_value(a);
}
