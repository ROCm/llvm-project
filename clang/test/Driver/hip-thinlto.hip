// In non-RDC mode, each translation unit is a complete device program, so
// ThinLTO does not use object linking.
// RUN: %clang -foffload-lto=thin -nogpulib -nogpuinc %s -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NORDC

// NORDC: "-cc1" "-triple" "amdgpu9.06-amd-amdhsa"
// NORDC-SAME: "-emit-llvm-bc"
// NORDC-SAME: "-flto=thin"
// NORDC-NOT: "-amdgpu-enable-object-linking"
// NORDC: clang-linker-wrapper
// NORDC-SAME: "--device-compiler=amdgpu-amd-amdhsa=-flto=thin"
// NORDC-NOT: "-plugin-opt=-amdgpu-enable-object-linking"
// NORDC-NOT: "-plugin-opt=-force-import-all"
// NORDC-SAME: {{$}}

// In RDC mode, ThinLTO backends emit relocatable objects that are linked
// together, so both the device compilation and the LTO backend use object
// linking.
// RUN: %clang -foffload-lto=thin -fgpu-rdc -nogpulib -nogpuinc %s -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=RDC

// RDC: "-cc1" "-triple" "amdgpu9.06-amd-amdhsa"
// RDC-SAME: "-emit-llvm-bc"
// RDC-SAME: "-flto=thin"
// RDC-SAME: "-mllvm" "-amdgpu-enable-object-linking"
// RDC: clang-linker-wrapper
// RDC-SAME: "--device-compiler=amdgpu-amd-amdhsa=-flto=thin"
// RDC-SAME: "--device-linker=amdgpu-amd-amdhsa=-plugin-opt=-amdgpu-enable-object-linking"
// RDC-NOT: "-plugin-opt=-force-import-all"
int main(int, char *[]) {
  return 0;
}
