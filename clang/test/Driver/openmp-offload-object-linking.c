// REQUIRES: amdgpu-registered-target

// OpenMP offloading is always RDC. The driver turns on AMDGPU object linking
// when device LTO is disabled or is ThinLTO.

// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fopenmp \
// RUN:   --offload-arch=gfx906 -nogpulib -nogpuinc %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=FULL
// FULL: "-cc1" "-triple" "amdgpu9.06-amd-amdhsa"
// FULL-SAME: "-emit-llvm-bc"
// FULL-SAME: "-flto=full"
// FULL-NOT: "-amdgpu-enable-object-linking"
// FULL: clang-linker-wrapper
// FULL-SAME: "--device-compiler=amdgpu-amd-amdhsa=-flto=full"
// FULL-NOT: "-plugin-opt=-amdgpu-enable-object-linking"

// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fopenmp \
// RUN:   --offload-arch=gfx906 -foffload-lto=thin -nogpulib -nogpuinc %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=THIN
// THIN: "-cc1" "-triple" "amdgpu9.06-amd-amdhsa"
// THIN-SAME: "-emit-llvm-bc"
// THIN-SAME: "-flto=thin"
// THIN-SAME: "-mllvm" "-amdgpu-enable-object-linking"
// THIN: clang-linker-wrapper
// THIN-SAME: "--device-compiler=amdgpu-amd-amdhsa=-flto=thin"
// THIN-SAME: "--device-linker=amdgpu-amd-amdhsa=-plugin-opt=-amdgpu-enable-object-linking"
// THIN-NOT: "-plugin-opt=-amdgpu-internalize-symbols"

// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fopenmp \
// RUN:   --offload-arch=gfx906 -fno-offload-lto -nogpulib -nogpuinc %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NOLTO
// NOLTO: "-cc1" "-triple" "amdgpu9.06-amd-amdhsa"
// NOLTO-SAME: "-emit-obj"
// NOLTO-SAME: "-mllvm" "-amdgpu-enable-object-linking"
// NOLTO: clang-linker-wrapper
// NOLTO-NOT: "--device-compiler=amdgpu-amd-amdhsa=-flto
// NOLTO-SAME: "--device-linker=amdgpu-amd-amdhsa=-plugin-opt=-amdgpu-enable-object-linking"

// -fopenmp-target-jit implies full LTO, so object linking stays off.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fopenmp \
// RUN:   --offload-arch=gfx906 -fopenmp-target-jit -nogpulib -nogpuinc %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=JIT
// JIT: "-cc1" "-triple" "amdgpu9.06-amd-amdhsa"
// JIT-NOT: "-amdgpu-enable-object-linking"
// JIT: clang-linker-wrapper
// JIT-NOT: "-plugin-opt=-amdgpu-enable-object-linking"

int main(void) { return 0; }
