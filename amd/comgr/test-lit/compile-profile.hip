// REQUIRES: comgr-has-amdgpu-profile-runtime
// COM: Match hipRTC's separate source-to-relocatable and executable-link
// COM: actions, with the profile-generation option forwarded to the link.
// RUN: rm -rf %t.dir && mkdir %t.dir
// RUN: cd %t.dir && compile-hip-minimal --profile %s %t.bin

// COM: Verify the kernel and device profile runtime are present.
// RUN: %llvm-objdump -d %t.bin | %FileCheck %s --check-prefix=KERNEL
// RUN: %llvm-objdump -t %t.bin | %FileCheck %s --check-prefix=PROFILE
// KERNEL: <_Z3addPfS_S_>:
// KERNEL: s_endpgm
// PROFILE: __llvm_profile_sampling_gpu
// PROFILE: __llvm_profile_instrument_gpu

// COM: Explicit caller options must override COMGR's resource-dir default.
// RUN: not env AMD_COMGR_EMIT_VERBOSE_LOGS=1 \
// RUN:   AMD_COMGR_REDIRECT_LOGS=stdout \
// RUN:   compile-hip-minimal --profile-resource-dir %t.missing %s \
// RUN:   %t.override.bin 2>&1 | %FileCheck %s --check-prefix=OVERRIDE
// OVERRIDE-LABEL: ActionKind: AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE
// OVERRIDE: Compilation Args:
// OVERRIDE-NOT: "-resource-dir" "{{[^"]*[/\\]input}}"
// OVERRIDE-SAME: "-fprofile-generate" "-resource-dir" "{{[^"]*missing}}"

__attribute__((global))
void add(float *A, float *B, float *C) {
  *C = *A + *B;
}
