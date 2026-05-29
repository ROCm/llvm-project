// RUN: %clang -### --target=x86_64-apple-macosx -v \
// RUN:   -x hip --cuda-gpu-arch=gfx803 --cuda-gpu-arch=gfx900 \
// RUN:   -fgpu-rdc -nogpuinc -nogpulib --no-offload-new-driver \
// RUN:   -fhip-dump-offload-linker-script \
// RUN:   %S/Inputs/hip_multiple_inputs/a.cu \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s

// CHECK: Found undefined HIP fatbin symbol: ___hip_fatbin_[[ID1:[0-9a-f]+]]
// CHECK: Found undefined HIP fatbin symbol: ___hip_fatbin_[[ID2:[0-9a-f]+]]
// CHECK: Found undefined HIP gpubin handle symbol: ___hip_gpubin_handle_[[ID1]]
// CHECK: Found undefined HIP gpubin handle symbol: ___hip_gpubin_handle_[[ID2]]
// CHECK-LABEL: .section __HIP,__hip_gpubin
// CHECK: .globl ___hip_gpubin_handle_[[ID1]]
// CHECK-NEXT: .p2align 3
// CHECK-NEXT: ___hip_gpubin_handle_[[ID1]]:
// CHECK-NEXT: .zero 8
// CHECK-NEXT: .globl ___hip_gpubin_handle_[[ID2]]
// CHECK-NEXT: .set ___hip_gpubin_handle_[[ID2]],___hip_gpubin_handle_[[ID1]]
// CHECK-LABEL: .section __HIP,__hip_fatbin
// CHECK: .globl ___hip_fatbin_[[ID1]]
// CHECK-NEXT: .p2align 12
// CHECK-NEXT: .globl ___hip_fatbin_[[ID2]]
// CHECK-NEXT: .set ___hip_fatbin_[[ID2]],___hip_fatbin_[[ID1]]
// CHECK-NEXT: ___hip_fatbin_[[ID1]]:
// CHECK-NEXT: .incbin "{{.*}}.hipfb"
