// Test PGO when the kernel runs on a non-default device.
// The profile runtime should collect data from whichever device is active
// at exit time, which should be the device the user set.
//
// REQUIRES: hip, amdgpu, multi-device
// RUN: %clang -x hip -fprofile-instr-generate -fcoverage-mapping \
// RUN:   --offload-arch=%amdgpu_arch %s -o %t -L%hip_lib_path -lamdhip64
// RUN: env LLVM_PROFILE_FILE=%t.profraw \
// RUN:   LD_LIBRARY_PATH=%hip_lib_path:$LD_LIBRARY_PATH \
// RUN:   LLVM_PROFILE_VERBOSE=1 %run %t 2>&1 | FileCheck %s
//
// The runtime should collect from the device that was set (device 1).
// CHECK: Collecting static profile data from device 1
// CHECK: Successfully wrote device profile using shared API
// CHECK: PASS

#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void fill(int *data, int val, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        data[idx] = val;
}

int main() {
    int ndev = 0;
    (void)hipGetDeviceCount(&ndev);
    if (ndev < 2) {
        printf("PASS (skipped: only %d device)\n", ndev);
        return 0;
    }

    (void)hipSetDevice(1);

    constexpr int N = 32;
    int h[N] = {};
    int *d;
    (void)hipMalloc(&d, N * sizeof(int));
    fill<<<1, N>>>(d, 99, N);
    (void)hipMemcpy(h, d, N * sizeof(int), hipMemcpyDeviceToHost);
    (void)hipFree(d);

    int ok = 1;
    for (int i = 0; i < N; ++i)
        if (h[i] != 99) ok = 0;

    printf("%s\n", ok ? "PASS" : "FAIL");
    return !ok;
}
