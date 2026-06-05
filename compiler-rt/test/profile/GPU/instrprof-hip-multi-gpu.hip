// Test that HIP PGO works on multi-GPU systems. The HSA-introspection drain
// walks every GPU agent at exit and collects device counters from each loaded
// code object; it must do so without crashing (the previous host-shadow design
// relied on hipGetSymbolAddress and could hit a comgr lazy-load null deref when
// more than one device was present).
//
// REQUIRES: hip, amdgpu
// The HSA agent-walk drain is Linux-only; the Windows host-shadow drain
// collects only the current device and emits no "walk complete" line.
// UNSUPPORTED: windows
// RUN: %clang -x hip -fprofile-instr-generate -fcoverage-mapping \
// RUN:   --offload-arch=%amdgpu_arch %s -o %t -L%hip_lib_path -lamdhip64
// RUN: env LLVM_PROFILE_FILE=%t.profraw \
// RUN:   LD_LIBRARY_PATH=%hip_lib_path:$LD_LIBRARY_PATH \
// RUN:   LLVM_PROFILE_VERBOSE=1 %run %t 2>&1 | FileCheck %s
//
// Verify the drain walked the GPU agents, copied device counters, and the
// program didn't crash on a multi-GPU system.
// CHECK: Copied device sections:
// CHECK: walk complete: agents={{[0-9]+}} pairs={{[0-9]+}} found={{[0-9]+}} drained={{[1-9][0-9]*}}
// CHECK: PASS

#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void add_one(int *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        data[idx] += 1;
}

int main() {
    int ndev = 0;
    (void)hipGetDeviceCount(&ndev);

    constexpr int N = 32;
    int h_data[N];
    for (int i = 0; i < N; ++i) h_data[i] = i;

    int *d_data;
    (void)hipMalloc(&d_data, N * sizeof(int));
    (void)hipMemcpy(d_data, h_data, N * sizeof(int), hipMemcpyHostToDevice);

    add_one<<<1, N>>>(d_data, N);

    (void)hipMemcpy(h_data, d_data, N * sizeof(int), hipMemcpyDeviceToHost);
    (void)hipFree(d_data);

    int ok = 1;
    for (int i = 0; i < N; ++i)
        if (h_data[i] != i + 1) ok = 0;

    printf("%s (devices=%d)\n", ok ? "PASS" : "FAIL", ndev);
    return !ok;
}
