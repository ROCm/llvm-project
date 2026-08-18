// Two gfx1250 code objects that differ only by the gfx1250 A0/B0 hardware
// revision -- carried as the (temporary) gfx1250-b0-specific target-ID feature,
// "+" = B0, "-" = A0 -- can coexist in one offload bundle and be unbundled
// independently by their marker.
//
// This variant keeps the marker out of the canonical AMDGPU target-ID feature
// set: the clang driver does not accept it in --offload-arch, so the bundle is
// assembled with clang-offload-bundler directly (its create path validates
// entry-ID format only). Comgr's unbundler recognizes the marker.

// COM: Compile one plain gfx1250 device bitcode; bundle it under both revision
// COM: markers (payload content is irrelevant here -- this exercises marker
// COM: based selection).
// RUN: %clang -c -x hip --offload-arch=gfx1250 -nogpulib -nogpuinc -emit-llvm \
// RUN:   --no-gpu-bundle-output --offload-device-only %s -o %t.gfx1250.bc

// COM: Store hipv4 entries (as a real HIP fat binary does) and query them below
// COM: with the hip kind, so selection goes through the feature-comparison path
// COM: rather than an exact string match.
// RUN: %clang-offload-bundler -type=bc \
// RUN:   -targets=hipv4-amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+,hipv4-amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   -input=%t.gfx1250.bc -input=%t.gfx1250.bc -output=%t.bundle.bc

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
// RUN: %clang-offload-bundler -type=bc \
// RUN:   -targets=hipv4-amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   -input=%t.gfx1250.bc -output=%t.b0only.bc
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
