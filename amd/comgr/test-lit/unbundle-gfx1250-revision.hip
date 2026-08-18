// Two gfx1250 code objects that differ only by the gfx1250 A0/B0 hardware
// revision -- carried as the (temporary) gfx1250-b0-specific target-ID feature,
// "+" = B0, "-" = A0 -- can coexist in one offload bundle and be unbundled
// independently by their marker.

// COM: Bundle both revisions for the same gfx1250 processor.
// RUN: %clang -c -x hip \
// RUN:   --offload-arch=gfx1250:gfx1250-b0-specific+ \
// RUN:   --offload-arch=gfx1250:gfx1250-b0-specific- \
// RUN:   -nogpulib -nogpuinc -emit-llvm --gpu-bundle-output --offload-device-only \
// RUN:   %s -o %t.bundle.bc

// COM: Each marker selects a gfx1250 code object.
// RUN: unbundle %t.bundle.bc hip-amdgcn-amd-amdhsa-unknown-gfx1250:gfx1250-b0-specific+ %t.b0.bc
// RUN: %llvm-dis %t.b0.bc -o - | %FileCheck --check-prefix=GFX1250 %s
// RUN: unbundle %t.bundle.bc hip-amdgcn-amd-amdhsa-unknown-gfx1250:gfx1250-b0-specific- %t.a0.bc
// RUN: %llvm-dis %t.a0.bc -o - | %FileCheck --check-prefix=GFX1250 %s

// COM: A bare gfx1250 request (no revision) matches neither entry -> empty
// COM: output, which is not valid bitcode.
// RUN: unbundle %t.bundle.bc hip-amdgcn-amd-amdhsa-unknown-gfx1250 %t.bare.bc
// RUN: not %llvm-dis %t.bare.bc -o /dev/null

// COM: The markers are not interchangeable. A bundle carrying only B0 yields
// COM: nothing for an A0 request, but the object for a B0 request.
// RUN: %clang -c -x hip --offload-arch=gfx1250:gfx1250-b0-specific+ \
// RUN:   -nogpulib -nogpuinc -emit-llvm --gpu-bundle-output --offload-device-only \
// RUN:   %s -o %t.b0only.bc
// RUN: unbundle %t.b0only.bc hip-amdgcn-amd-amdhsa-unknown-gfx1250:gfx1250-b0-specific- %t.miss.bc
// RUN: not %llvm-dis %t.miss.bc -o /dev/null
// RUN: unbundle %t.b0only.bc hip-amdgcn-amd-amdhsa-unknown-gfx1250:gfx1250-b0-specific+ %t.hit.bc
// RUN: %llvm-dis %t.hit.bc -o - | %FileCheck --check-prefix=GFX1250 %s

// GFX1250: target triple = "amdgcn-amd-amdhsa"
// GFX1250: "target-cpu"="gfx1250"

__attribute__((device))
void add_value(float* a, float* b, float* res) {
    *res = *a + *b;
}
