// REQUIRES: system-windows
//
// Windows-only companion of unbundle-feature-target.hip. Comgr builds the
// temporary output file from the requested bundle entry ID, and Windows
// rejects ':' in a file name, so an entry such as
// "gfx90a:sramecc+:xnack-" must be sanitized before it reaches the file
// system. With AMD_COMGR_SAVE_TEMPS the temporary file survives the action and
// its name can be inspected: ':' must have become '_' while the rest of the
// name is untouched. The public data object name is checked separately by
// unbundle-feature-target.hip and must not be sanitized.
//
// RUN: %clang -c -x hip --offload-arch=gfx90a:sramecc+:xnack- \
// RUN:    -nogpulib -nogpuinc -emit-llvm \
// RUN:    --gpu-bundle-output --offload-device-only \
// RUN:    %s -o %t.bundle.bc
//
// %t.tmp is set as the temporary directory so the diagnostic copies land in a
// location this test controls.
// RUN: mkdir -p %t.tmp
// RUN: TMP=%t.tmp TEMP=%t.tmp AMD_COMGR_SAVE_TEMPS=1 \
// RUN:   unbundle %t.bundle.bc \
// RUN:     hip-amdgcn-amd-amdhsa-unknown-gfx90a:sramecc+:xnack- %t.gfx90a.bc
// RUN: ls %t.tmp/comgr-*/output | %FileCheck --check-prefix=TEMP %s
//
// The unbundled bitcode is still valid, so only the file name was sanitized.
// RUN: %llvm-dis %t.gfx90a.bc -o - | %FileCheck --check-prefix=BC %s

// TEMP: bundle-hip-amdgcn-amd-amdhsa-unknown-gfx90a_sramecc+_xnack-.bc
// BC: target datalayout =
// BC: {{target triple = "amdgpu9\.0a-amd-amdhsa"}}

__attribute__((device))
void add_value(float* a, float* b, float* res) {
    *res = *a + *b;
}
