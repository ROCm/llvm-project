// REQUIRES: x86-registered-target, amdgpu-registered-target

// '-fopenmp-target-xteam-reduction-blocksize=' selects the block size of
// cross-team reduction kernels and is forwarded to -cc1 without a deprecation
// warning. The '-f[no-]openmp-target-xteam-reduction' flags, in contrast, are
// deprecated and ignored.

// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a -nogpulib -fopenmp-target-xteam-reduction-blocksize=1024 %s 2>&1 \
// RUN:   | FileCheck -check-prefix=BLOCKSIZE %s

// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a -nogpulib %s 2>&1 \
// RUN:   | FileCheck -check-prefix=NO-BLOCKSIZE %s

// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a -nogpulib -fopenmp-target-xteam-reduction %s 2>&1 \
// RUN:   | FileCheck -check-prefix=DEPRECATED %s

// BLOCKSIZE-NOT: warning:
// BLOCKSIZE: "-fopenmp-target-xteam-reduction-blocksize=1024"

// NO-BLOCKSIZE-NOT: "-fopenmp-target-xteam-reduction-blocksize=

// DEPRECATED: warning: argument '-fopenmp-target-xteam-reduction' is deprecated

int main(void) { return 0; }
