// REQUIRES: x86-registered-target, amdgpu-registered-target

// The xteam-scan feature has been removed. Its driver flags are retained only
// as deprecated no-ops: using them must warn and must not forward anything to
// -cc1.

// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a -nogpulib \
// RUN:     -fopenmp-target-xteam-scan -fno-openmp-target-xteam-scan \
// RUN:     -fopenmp-target-xteam-no-loop-scan -fno-openmp-target-xteam-no-loop-scan %s 2>&1 \
// RUN:   | FileCheck %s

// CHECK-DAG: argument '-fopenmp-target-xteam-scan' is deprecated
// CHECK-DAG: argument '-fno-openmp-target-xteam-scan' is deprecated
// CHECK-DAG: argument '-fopenmp-target-xteam-no-loop-scan' is deprecated
// CHECK-DAG: argument '-fno-openmp-target-xteam-no-loop-scan' is deprecated

// The flags must not be forwarded to any -cc1 invocation.
// CHECK-NOT: "-fopenmp-target-xteam-scan"
// CHECK-NOT: "-fno-openmp-target-xteam-scan"
// CHECK-NOT: "-fopenmp-target-xteam-no-loop-scan"
// CHECK-NOT: "-fno-openmp-target-xteam-no-loop-scan"
