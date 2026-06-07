// Test PGO when the kernel runs on a non-default device. The HSA-introspection
// drain walks every GPU agent at exit and collects device counters from each
// loaded code object, so it must still drain the device profile when the active
// device at exit is not device 0 (here the program selects device 1).
//
// REQUIRES: hip, amdgpu, multi-device
// Relies on the HSA agent-walk draining a device that is not current at exit;
// the Windows host-shadow drain only collects the current device.
// UNSUPPORTED: windows
// RUN: %clang -x hip -fprofile-instr-generate -fcoverage-mapping \
// RUN:   --offload-arch=%amdgpu_arch %s -o %t -L%hip_lib_path -lamdhip64
// RUN: env LLVM_PROFILE_FILE=%t.profraw \
// RUN:   LD_LIBRARY_PATH=%hip_lib_path:$LD_LIBRARY_PATH \
// RUN:   LLVM_PROFILE_VERBOSE=1 %run %t 2>&1 | FileCheck %s
//
// The drain must walk the agents and copy device counters even though the
// kernel ran on the non-default device.
// CHECK: Copied device sections:
// CHECK: walk complete: agents={{[0-9]+}} pairs={{[0-9]+}} found={{[0-9]+}} drained={{[1-9][0-9]*}}
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
