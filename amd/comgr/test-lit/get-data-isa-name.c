// COM: Test Comgr get_data_isa_name() API
// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx900 -nogpulib -nogpuinc \
// RUN:   -c %S/comgr-sources/shared.cl -o %t.gfx900.o
// RUN: get-data-isa-name %t.gfx900.o "amdgcn-amd-amdhsa--gfx900"

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx908 -nogpulib -nogpuinc \
// RUN:   -c %S/comgr-sources/shared.cl -o %t.gfx908.o
// RUN: get-data-isa-name %t.gfx908.o "amdgcn-amd-amdhsa--gfx908"

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1030 -nogpulib -nogpuinc \
// RUN:   -c %S/comgr-sources/shared.cl -o %t.gfx1030.o
// RUN: get-data-isa-name %t.gfx1030.o "amdgcn-amd-amdhsa--gfx1030"

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx9-generic -mcode-object-version=6 -nogpulib -nogpuinc \
// RUN:   -c %S/comgr-sources/shared.cl -o %t.gfx9-generic.o
// RUN: get-data-isa-name %t.gfx9-generic.o "amdgcn-amd-amdhsa--gfx9-generic"
