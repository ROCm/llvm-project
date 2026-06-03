// Test that HIP PGO doesn't crash on multi-GPU systems.
// The profile runtime should only collect from the active device at exit,
// not iterate all devices (which can trigger comgr lazy-load crashes).
//
// REQUIRES: hip, amdgpu
// RUN: %clang -x hip -fprofile-instr-generate -fcoverage-mapping \
// RUN:   --offload-arch=%amdgpu_arch %s -o %t -L%hip_lib_path -lamdhip64
// RUN: env LLVM_PROFILE_FILE=%t.profraw \
// RUN:   LD_LIBRARY_PATH=%hip_lib_path:$LD_LIBRARY_PATH \
// RUN:   LLVM_PROFILE_VERBOSE=1 %run %t 2>&1 | FileCheck %s
//
// Verify device 0 data was collected and program didn't crash.
// CHECK: Collecting static profile data from device 0
// CHECK: Successfully wrote device profile using shared API
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
