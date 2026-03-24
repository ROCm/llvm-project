// COM: Test compilation with -amdgpu-gfx1250-b0-specific flag set to 0 (A0) and 1 (B0)
// COM: and verify the .gfx1250_revision metadata in the resulting code objects.

// COM: Compile for gfx1250 with b0-specific=0 (A0)
// RUN: %clang --offload-arch=gfx1250 --offload-device-only \
// RUN:   --no-gpu-bundle-output -nogpulib -nogpuinc \
// RUN:   -Xclang -mllvm -Xclang -amdgpu-gfx1250-b0-specific=0 \
// RUN:   %s -o %t.a0.so

// COM: Compile for gfx1250 with b0-specific=1 (B0)
// RUN: %clang --offload-arch=gfx1250 --offload-device-only \
// RUN:   --no-gpu-bundle-output -nogpulib -nogpuinc \
// RUN:   -Xclang -mllvm -Xclang -amdgpu-gfx1250-b0-specific=1 \
// RUN:   %s -o %t.b0.so

// COM: Check that the A0 code object has .gfx1250_revision: A0
// RUN: check-gfx1250-revision %t.a0.so | %FileCheck --check-prefix=CHECK-A0 %s
// CHECK-A0: gfx1250_revision: A0

// COM: Check that the B0 code object has .gfx1250_revision: B0
// RUN: check-gfx1250-revision %t.b0.so | %FileCheck --check-prefix=CHECK-B0 %s
// CHECK-B0: gfx1250_revision: B0

__attribute__((global))
void add(float *A, float *B, float *C) {
    *C = *A + *B;
}
