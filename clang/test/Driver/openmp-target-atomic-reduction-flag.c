// REQUIRES: x86-registered-target, amdgpu-registered-target

// Not passed by default.
// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a -nogpulib %s 2>&1 \
// RUN:   | FileCheck -check-prefix=DEFAULT %s

// Passed through to -cc1 when requested.
// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a -nogpulib -fopenmp-target-atomic-reduction %s 2>&1 \
// RUN:   | FileCheck -check-prefix=ENABLE %s

// Explicit disable wins over a preceding enable and is not passed through.
// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a -nogpulib -fopenmp-target-atomic-reduction -fno-openmp-target-atomic-reduction %s 2>&1 \
// RUN:   | FileCheck -check-prefix=DEFAULT %s

// The legacy '-f[no-]openmp-target-fast-reduction' spellings are aliases.
// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a -nogpulib -fopenmp-target-fast-reduction %s 2>&1 \
// RUN:   | FileCheck -check-prefix=ENABLE %s
// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a -nogpulib -fopenmp-target-fast-reduction -fno-openmp-target-fast-reduction %s 2>&1 \
// RUN:   | FileCheck -check-prefix=DEFAULT %s
// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a -nogpulib -fopenmp-target-fast-reduction -fno-openmp-target-atomic-reduction %s 2>&1 \
// RUN:   | FileCheck -check-prefix=DEFAULT %s
// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a -nogpulib -fno-openmp-target-fast-reduction -fopenmp-target-atomic-reduction %s 2>&1 \
// RUN:   | FileCheck -check-prefix=ENABLE %s

// The legacy spellings are not deprecated and must not warn.
// DEFAULT-NOT: warning:
// DEFAULT-NOT: {{"-f(no-)?openmp-target-atomic-reduction"}}
// DEFAULT-NOT: {{"-f(no-)?openmp-target-fast-reduction"}}

// ENABLE-NOT: warning:
// ENABLE: "-fopenmp-target-atomic-reduction"
// ENABLE-NOT: "-fno-openmp-target-atomic-reduction"
// ENABLE-NOT: {{"-f(no-)?openmp-target-fast-reduction"}}
