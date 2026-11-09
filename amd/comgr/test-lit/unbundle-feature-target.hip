// Feature-qualified target IDs such as "gfx90a:sramecc+:xnack-" contain ':'.
// Windows reserves ':' for drive letters and alternate data streams, so a
// temporary file cannot be named after such an entry ID verbatim. Comgr
// sanitizes ':' in the temporary path only; the public data object name must
// keep the entry ID unchanged, so callers see the same names on every
// platform.
//
// RUN: %clang -c -x hip --offload-arch=gfx90a:sramecc+:xnack- \
// RUN:    -nogpulib -nogpuinc -emit-llvm \
// RUN:    --gpu-bundle-output --offload-device-only \
// RUN:    %s -o %t.bundle.bc
//
// The bundle entry must really carry the target features, otherwise nothing
// below would exercise a name that needs Windows sanitization.
// RUN: clang-offload-bundler -type=bc -input=%t.bundle.bc -list \
// RUN:   | %FileCheck --check-prefix=ENTRY %s
//
// Unbundling must succeed and report the entry ID unchanged in the public
// name. Both the exit status of the unbundle call and the name are checked.
// RUN: unbundle --print-names %t.bundle.bc \
// RUN:   hip-amdgcn-amd-amdhsa-unknown-gfx90a:sramecc+:xnack- %t.gfx90a.bc \
// RUN:   > %t.names.txt
// RUN: %FileCheck --check-prefix=NAME %s < %t.names.txt
// RUN: %llvm-dis %t.gfx90a.bc -o - | %FileCheck --check-prefix=BC %s

// ENTRY: hipv4-amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-
// NAME: bundle-hip-amdgcn-amd-amdhsa-unknown-gfx90a:sramecc+:xnack-.bc
// BC: target datalayout =
// BC: {{target triple = "amdgpu9\.0a-amd-amdhsa"}}

__attribute__((device))
void add_value(float* a, float* b, float* res) {
    *res = *a + *b;
}
