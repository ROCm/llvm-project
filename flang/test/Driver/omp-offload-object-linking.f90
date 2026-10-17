! REQUIRES: amdgpu-registered-target

! OpenMP offloading is always RDC. The driver turns on AMDGPU object linking
! when device LTO is disabled or is ThinLTO.

! RUN: %flang -### --target=x86_64-unknown-linux-gnu -fopenmp \
! RUN:   --offload-arch=gfx906 -nogpulib %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=FULL
! FULL: "{{[^"]*}}flang{{[^"]*}}" "-fc1" "-triple" "amdgpu9.06-amd-amdhsa"
! FULL-NOT: "-amdgpu-enable-object-linking"
! FULL: "{{[^"]*}}clang-linker-wrapper"
! FULL-SAME: "--device-compiler=amdgpu-amd-amdhsa=-flto=full"
! FULL-NOT: "-plugin-opt=-amdgpu-enable-object-linking"

! RUN: %flang -### --target=x86_64-unknown-linux-gnu -fopenmp \
! RUN:   --offload-arch=gfx906 -foffload-lto=thin -nogpulib %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=THIN
! THIN: "{{[^"]*}}flang{{[^"]*}}" "-fc1" "-triple" "amdgpu9.06-amd-amdhsa"
! THIN-SAME: "-emit-llvm-bc"
! THIN-SAME: "-flto=thin"
! THIN-SAME: "-mllvm" "-amdgpu-enable-object-linking"
! THIN: "{{[^"]*}}clang-linker-wrapper"
! THIN-SAME: "--device-compiler=amdgpu-amd-amdhsa=-flto=thin"
! THIN-SAME: "--device-linker=amdgpu-amd-amdhsa=-plugin-opt=-amdgpu-enable-object-linking"
! THIN-NOT: "-plugin-opt=-force-import-all"
! THIN-NOT: "-plugin-opt=-amdgpu-internalize-symbols"

! RUN: %flang -### --target=x86_64-unknown-linux-gnu -fopenmp \
! RUN:   --offload-arch=gfx906 -fno-offload-lto -nogpulib %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=NOLTO
! NOLTO: "{{[^"]*}}flang{{[^"]*}}" "-fc1" "-triple" "amdgpu9.06-amd-amdhsa"
! NOLTO-NOT: "-flto
! NOLTO-SAME: "-mllvm" "-amdgpu-enable-object-linking"
! NOLTO: "{{[^"]*}}clang-linker-wrapper"
! NOLTO-SAME: "--device-linker=amdgpu-amd-amdhsa=-plugin-opt=-amdgpu-enable-object-linking"

end
