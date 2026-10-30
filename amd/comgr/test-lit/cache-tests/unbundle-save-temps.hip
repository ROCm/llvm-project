// In-memory outputs must also be saved when explicitly requested, on a cache
// miss, a cache hit, and with caching disabled. Compare every saved file with
// the returned bytes, including an empty host output.
//
// RUN: %clang -c -x hip --offload-arch=gfx900 -nogpulib -nogpuinc \
// RUN:   --gpu-bundle-output --offload-device-only -emit-llvm \
// RUN:   --offload-compress %s -o %t.bundle.bc
// RUN: %python %S/Inputs/unbundle-save-temps.py unbundle %t.bundle.bc %t.work %clang

__attribute__((device)) void kernel() {}
