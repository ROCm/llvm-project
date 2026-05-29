// REQUIRES: amdgpu-registered-target
// REQUIRES: x86-registered-target

// The device profile drain object (InstrProfilingPlatformROCm.o, providing
// __llvm_profile_hip_collect_device_data) is otherwise unreferenced now that
// the host emits no per-TU offload-profiling shadow. For HIP host links built
// with PGO instrumentation the driver force-links it with
// -u__llvm_profile_hip_collect_device_data so its atexit drain runs.

// HIP + PGO (clang instr): force-link the drain.
// RUN: %clang -### --target=x86_64-linux-gnu -x hip --offload-arch=gfx90a \
// RUN:   -nogpuinc -nogpulib -fprofile-instr-generate %s 2>&1 \
// RUN:   | FileCheck -check-prefix=DRAIN %s

// HIP + PGO (-fprofile-generate): same.
// RUN: %clang -### --target=x86_64-linux-gnu -x hip --offload-arch=gfx90a \
// RUN:   -nogpuinc -nogpulib -fprofile-generate %s 2>&1 \
// RUN:   | FileCheck -check-prefix=DRAIN %s

// HIP without PGO: no force-link.
// RUN: %clang -### --target=x86_64-linux-gnu -x hip --offload-arch=gfx90a \
// RUN:   -nogpuinc -nogpulib %s 2>&1 \
// RUN:   | FileCheck -check-prefix=NODRAIN %s

// Non-HIP + PGO: no force-link (the drain is HIP-only).
// RUN: %clang -### --target=x86_64-linux-gnu -x c++ \
// RUN:   -fprofile-instr-generate %s 2>&1 \
// RUN:   | FileCheck -check-prefix=NODRAIN %s

// DRAIN: "-u__llvm_profile_hip_collect_device_data"
// NODRAIN-NOT: "-u__llvm_profile_hip_collect_device_data"

int main() { return 0; }
