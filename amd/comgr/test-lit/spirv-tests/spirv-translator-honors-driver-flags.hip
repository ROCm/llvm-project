// REQUIRES: comgr-has-spirv-translator

// COM: Verify COMGR honors the --spirv-* flags HIPAMD emits today.
// COM: Each RUN pins one observable side-effect of a specific driver flag.

// RUN: source-to-spirv -nogpuinc %s %t.spv
// RUN: test -s %t.spv

// COM: --spirv-ext=+all,-SPV_KHR_untyped_pointers (disabled).
// RUN: not grep -a "SPV_KHR_untyped_pointers" %t.spv

// COM: --spirv-ext=+all (must allow extensions outside the default set).
// RUN: grep -a "SPV_INTEL_kernel_attributes" %t.spv

// COM: --spirv-preserve-auxdata.
// RUN: grep -a "NonSemantic.AuxData" %t.spv

__attribute__((device))
void clean_value(float* ptr) { *ptr = 0; }

__attribute__((global))
void add_value(float* a, float* b, float* res) {
    *res = *a + *b;
    clean_value(a);
}
