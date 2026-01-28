// REQUIRES: x86-registered-target, amdgpu-registered-target

// RUN:   %clang -### -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a -nogpulib %s -O0 2>&1 \
// RUN:   | FileCheck -check-prefixes=DefaultTFast,DefaultEnV,DefaultTState,DefaultNoNestParallel %s

// RUN:   %clang -### -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a -nogpulib -O0 -fopenmp-target-fast %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=TFast,EnV,TState,NestParallel %s

// RUN:   %clang -### -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a -nogpulib -O3 %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=O3,DefaultTFast,DefaultEnV,DefaultTState,DefaultNoNestParallel %s

// RUN:   %clang -### -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a -nogpulib -O3 -fno-openmp-target-fast %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=O3,NoTFast,DefaultEnV,DefaultTState,DefaultNoNestParallel %s

// RUN:   %clang -### -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a -nogpulib -Ofast %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=OFast,TFast,EnV,TState,NestParallel %s

// RUN:   %clang -### -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a -nogpulib -Ofast -fno-openmp-target-fast %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=OFast,NoTFast,DefaultEnV,DefaultTState,DefaultNoNestParallel %s

// RUN:   %clang -### -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a -nogpulib -fopenmp-target-fast -fno-openmp-target-ignore-env-vars %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=TFast,NoEnV,TState,NestParallel %s

// RUN:   %clang -### -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a -nogpulib -O0 -fno-openmp-target-fast -fopenmp-target-fast %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=TFast,EnV,TState,NestParallel %s

// O3: -O3
// OFast: -Ofast

// TFast: "-fopenmp-target-fast"
// TFast-NOT: "-fno-openmp-target-fast"
// NoTFast: "-fno-openmp-target-fast"
// NoTFast-NOT: "-fopenmp-target-fast"
// DefaultTFast-NOT: {{"-f(no-)?openmp-target-fast"}}

// EnV: "-fopenmp-target-ignore-env-vars"
// EnV-NOT: "-fno-openmp-target-ignore-env-vars"
// NoEnV: "-fno-openmp-target-ignore-env-vars"
// NoEnV-NOT: "-fopenmp-target-ignore-env-vars"
// DefaultEnV-NOT: {{"-f(no-)?openmp-target-ignore-env-vars"}}

// TState: "-fopenmp-assume-no-thread-state"
// TState-NOT: "-fno-openmp-assume-no-thread-state"
// DefaultTState-NOT: {{"-f(no-)?openmp-assume-no-thread-state"}}

// NestParallel: "-fopenmp-assume-no-nested-parallelism"
// NestParallel-NOT: "-fno-openmp-assume-no-nested-parallelism"
// DefaultNoNestParallel-NOT: {{"-f(-no-)?openmp-assume-no-nested-parallelism"}}
