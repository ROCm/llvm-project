// REQUIRES: amdgpu-registered-target

// The driver turns on AMDGPU object linking only when the backend emits
// relocatable device objects that are linked with each other: RDC mode with
// device LTO disabled, or with ThinLTO.

//--- Non-RDC: no object linking for any LTO mode.
// RUN: %clang -### --target=x86_64-linux-gnu -nogpulib -nogpuinc \
// RUN:   --offload-arch=gfx90a %s 2>&1 | FileCheck -check-prefix=NORDC %s
// RUN: %clang -### --target=x86_64-linux-gnu -nogpulib -nogpuinc \
// RUN:   --offload-arch=gfx90a -foffload-lto=thin %s 2>&1 \
// RUN:   | FileCheck -check-prefix=NORDC %s
// RUN: %clang -### --target=x86_64-linux-gnu -nogpulib -nogpuinc \
// RUN:   --offload-arch=gfx90a -fno-offload-lto %s 2>&1 \
// RUN:   | FileCheck -check-prefix=NORDC %s
// NORDC: "-cc1" "-triple" "amdgpu9.0a-amd-amdhsa"
// NORDC-NOT: "-amdgpu-enable-object-linking"
// NORDC-SAME: {{$}}
// NORDC: clang-linker-wrapper
// NORDC-NOT: "-plugin-opt=-amdgpu-enable-object-linking"
// NORDC-SAME: {{$}}

//--- RDC with default (full) LTO: no object linking.
// RUN: %clang -### --target=x86_64-linux-gnu -nogpulib -nogpuinc \
// RUN:   --offload-arch=gfx90a -fgpu-rdc %s 2>&1 \
// RUN:   | FileCheck -check-prefix=RDC-FULL %s
// RUN: %clang -### --target=x86_64-linux-gnu -nogpulib -nogpuinc \
// RUN:   --offload-arch=gfx90a -fgpu-rdc -foffload-lto %s 2>&1 \
// RUN:   | FileCheck -check-prefix=RDC-FULL %s
// RDC-FULL: "-cc1" "-triple" "amdgpu9.0a-amd-amdhsa"
// RDC-FULL-SAME: "-emit-llvm-bc"
// RDC-FULL-SAME: "-flto=full"
// RDC-FULL-NOT: "-amdgpu-enable-object-linking"
// RDC-FULL: clang-linker-wrapper
// RDC-FULL-SAME: "--device-compiler=amdgpu-amd-amdhsa=-flto=full"
// RDC-FULL-NOT: "-plugin-opt=-amdgpu-enable-object-linking"

//--- RDC with ThinLTO: object linking in device cc1 and in the LTO backend.
// RUN: %clang -### --target=x86_64-linux-gnu -nogpulib -nogpuinc \
// RUN:   --offload-arch=gfx90a -fgpu-rdc -foffload-lto=thin %s 2>&1 \
// RUN:   | FileCheck -check-prefix=RDC-THIN %s
// RDC-THIN: "-cc1" "-triple" "amdgpu9.0a-amd-amdhsa"
// RDC-THIN-SAME: "-emit-llvm-bc"
// RDC-THIN-SAME: "-flto=thin"
// RDC-THIN-SAME: "-mllvm" "-amdgpu-enable-object-linking"
// RDC-THIN: clang-linker-wrapper
// RDC-THIN-SAME: "--device-compiler=amdgpu-amd-amdhsa=-flto=thin"
// RDC-THIN-SAME: "--device-linker=amdgpu-amd-amdhsa=-plugin-opt=-amdgpu-enable-object-linking"

//--- RDC with LTO disabled: device cc1 emits objects with object linking.
// RUN: %clang -### --target=x86_64-linux-gnu -nogpulib -nogpuinc \
// RUN:   --offload-arch=gfx90a -fgpu-rdc -fno-offload-lto %s 2>&1 \
// RUN:   | FileCheck -check-prefix=RDC-NOLTO %s
// RUN: %clang -### --target=x86_64-linux-gnu -nogpulib -nogpuinc \
// RUN:   --offload-arch=gfx90a -fgpu-rdc -foffload-lto=none %s 2>&1 \
// RUN:   | FileCheck -check-prefix=RDC-NOLTO %s
// RDC-NOLTO: "-cc1" "-triple" "amdgpu9.0a-amd-amdhsa"
// RDC-NOLTO-SAME: "-emit-obj"
// RDC-NOLTO-NOT: "-flto
// RDC-NOLTO-SAME: "-mllvm" "-amdgpu-enable-object-linking"
// RDC-NOLTO: clang-linker-wrapper
// RDC-NOLTO-NOT: "--device-compiler=amdgpu-amd-amdhsa=-flto
// RDC-NOLTO-SAME: "--device-linker=amdgpu-amd-amdhsa=-plugin-opt=-amdgpu-enable-object-linking"

//--- RDC device-only compilation with LTO disabled.
// RUN: %clang -### --target=x86_64-linux-gnu -nogpulib -nogpuinc \
// RUN:   --offload-arch=gfx90a -fgpu-rdc -fno-offload-lto \
// RUN:   --offload-device-only -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=RDC-NOLTO-DEVICE %s
// RDC-NOLTO-DEVICE: "-cc1" "-triple" "amdgpu9.0a-amd-amdhsa"
// RDC-NOLTO-DEVICE-SAME: "-emit-obj"
// RDC-NOLTO-DEVICE-SAME: "-mllvm" "-amdgpu-enable-object-linking"

//--- The host compilation is not affected.
// RUN: %clang -### --target=x86_64-linux-gnu -nogpulib -nogpuinc \
// RUN:   --offload-arch=gfx90a -fgpu-rdc -fno-offload-lto -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=HOST %s
// HOST: "-cc1" "-triple" "x86_64-unknown-linux-gnu"
// HOST-NOT: "-amdgpu-enable-object-linking"
// HOST-SAME: "-emit-obj"

//--- A link-only job links relocatable device code even without -fgpu-rdc.
// RUN: touch %t1.o %t2.o
// RUN: %clang -### --target=x86_64-linux-gnu --offload-arch=gfx90a \
// RUN:   --hip-link -foffload-lto=thin %t1.o %t2.o 2>&1 \
// RUN:   | FileCheck -check-prefix=LINK-THIN %s
// LINK-THIN: clang-linker-wrapper
// LINK-THIN-SAME: "--device-compiler=amdgpu-amd-amdhsa=-flto=thin"
// LINK-THIN-SAME: "--device-linker=amdgpu-amd-amdhsa=-plugin-opt=-amdgpu-enable-object-linking"
// RUN: %clang -### --target=x86_64-linux-gnu --offload-arch=gfx90a \
// RUN:   --hip-link %t1.o %t2.o 2>&1 \
// RUN:   | FileCheck -check-prefix=LINK-FULL %s
// LINK-FULL: clang-linker-wrapper
// LINK-FULL-SAME: "--device-compiler=amdgpu-amd-amdhsa=-flto=full"
// LINK-FULL-NOT: "-plugin-opt=-amdgpu-enable-object-linking"

//--- SPIR-V device code never uses AMDGPU object linking.
// RUN: %clang -### --target=x86_64-linux-gnu -nogpulib -nogpuinc \
// RUN:   --offload-arch=amdgcnspirv -fgpu-rdc -fno-offload-lto %s 2>&1 \
// RUN:   | FileCheck -check-prefix=SPIRV %s
// SPIRV: "-cc1" "-triple" "spirv64-amd-amdhsa"
// SPIRV-NOT: "-amdgpu-enable-object-linking"
// SPIRV: clang-linker-wrapper
// SPIRV-NOT: "-plugin-opt=-amdgpu-enable-object-linking"
