// COM: Test Comgr get_data_isa_name() API
// REQUIRES: system-windows

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx900 -nogpulib -nogpuinc \
// RUN:   -c %S/get-data-isa-name-windows.cl -o %t.o
// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx900 -nogpulib -nogpuinc \
// RUN:   -shared %S/get-data-isa-name-windows.cl -o %t.so
// RUN: get-data-isa-name-windows %t.o %t.so "amdgcn-amd-amdhsa--gfx900"

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx900:xnack+ -nogpulib -nogpuinc \
// RUN:   -c %S/get-data-isa-name-windows.cl -o %t.o
// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx900:xnack+ -nogpulib -nogpuinc \
// RUN:   -shared %S/get-data-isa-name-windows.cl -o %t.so
// RUN: get-data-isa-name-windows %t.o %t.so "amdgcn-amd-amdhsa--gfx900:xnack+"

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx900:xnack- -nogpulib -nogpuinc \
// RUN:   -c %S/get-data-isa-name-windows.cl -o %t.o
// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx900:xnack- -nogpulib -nogpuinc \
// RUN:   -shared %S/get-data-isa-name-windows.cl -o %t.so
// RUN: get-data-isa-name-windows %t.o %t.so "amdgcn-amd-amdhsa--gfx900:xnack-"

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx908 -nogpulib -nogpuinc \
// RUN:   -c %S/get-data-isa-name-windows.cl -o %t.o
// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx908 -nogpulib -nogpuinc \
// RUN:   -shared %S/get-data-isa-name-windows.cl -o %t.so
// RUN: get-data-isa-name-windows %t.o %t.so "amdgcn-amd-amdhsa--gfx908"

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx908:xnack+ -nogpulib -nogpuinc \
// RUN:   -c %S/get-data-isa-name-windows.cl -o %t.o
// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx908:xnack+ -nogpulib -nogpuinc \
// RUN:   -shared %S/get-data-isa-name-windows.cl -o %t.so
// RUN: get-data-isa-name-windows %t.o %t.so "amdgcn-amd-amdhsa--gfx908:xnack+"

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx908:xnack- -nogpulib -nogpuinc \
// RUN:   -c %S/get-data-isa-name-windows.cl -o %t.o
// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx908:xnack- -nogpulib -nogpuinc \
// RUN:   -shared %S/get-data-isa-name-windows.cl -o %t.so
// RUN: get-data-isa-name-windows %t.o %t.so "amdgcn-amd-amdhsa--gfx908:xnack-"

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx908:sramecc+ -nogpulib -nogpuinc \
// RUN:   -c %S/get-data-isa-name-windows.cl -o %t.o
// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx908:sramecc+ -nogpulib -nogpuinc \
// RUN:   -shared %S/get-data-isa-name-windows.cl -o %t.so
// RUN: get-data-isa-name-windows %t.o %t.so "amdgcn-amd-amdhsa--gfx908:sramecc+"

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx908:sramecc- -nogpulib -nogpuinc \
// RUN:   -c %S/get-data-isa-name-windows.cl -o %t.o
// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx908:sramecc- -nogpulib -nogpuinc \
// RUN:   -shared %S/get-data-isa-name-windows.cl -o %t.so
// RUN: get-data-isa-name-windows %t.o %t.so "amdgcn-amd-amdhsa--gfx908:sramecc-"

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx908:sramecc+:xnack+ -nogpulib -nogpuinc \
// RUN:   -c %S/get-data-isa-name-windows.cl -o %t.o
// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx908:sramecc+:xnack+ -nogpulib -nogpuinc \
// RUN:   -shared %S/get-data-isa-name-windows.cl -o %t.so
// RUN: get-data-isa-name-windows %t.o %t.so "amdgcn-amd-amdhsa--gfx908:sramecc+:xnack+"

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx908:sramecc+:xnack- -nogpulib -nogpuinc \
// RUN:   -c %S/get-data-isa-name-windows.cl -o %t.o
// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx908:sramecc+:xnack- -nogpulib -nogpuinc \
// RUN:   -shared %S/get-data-isa-name-windows.cl -o %t.so
// RUN: get-data-isa-name-windows %t.o %t.so "amdgcn-amd-amdhsa--gfx908:sramecc+:xnack-"

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx908:sramecc-:xnack+ -nogpulib -nogpuinc \
// RUN:   -c %S/get-data-isa-name-windows.cl -o %t.o
// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx908:sramecc-:xnack+ -nogpulib -nogpuinc \
// RUN:   -shared %S/get-data-isa-name-windows.cl -o %t.so
// RUN: get-data-isa-name-windows %t.o %t.so "amdgcn-amd-amdhsa--gfx908:sramecc-:xnack+"

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx908:sramecc-:xnack- -nogpulib -nogpuinc \
// RUN:   -c %S/get-data-isa-name-windows.cl -o %t.o
// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx908:sramecc-:xnack- -nogpulib -nogpuinc \
// RUN:   -shared %S/get-data-isa-name-windows.cl -o %t.so
// RUN: get-data-isa-name-windows %t.o %t.so "amdgcn-amd-amdhsa--gfx908:sramecc-:xnack-"


// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1030 -nogpulib -nogpuinc \
// RUN:   -c %S/get-data-isa-name-windows.cl -o %t.o
// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1030 -nogpulib -nogpuinc \
// RUN:   -shared %S/get-data-isa-name-windows.cl -o %t.so
// RUN: get-data-isa-name-windows %t.o %t.so "amdgcn-amd-amdhsa--gfx1030"

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx900 -nogpulib -nogpuinc -mcode-object-version=4 \
// RUN:   -c %S/get-data-isa-name-windows.cl -o %t.o
// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx900 -nogpulib -nogpuinc -mcode-object-version=4 \
// RUN:   -shared %S/get-data-isa-name-windows.cl -o %t.so
// RUN: get-data-isa-name-windows %t.o %t.so "amdgcn-amd-amdhsa--gfx900"

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx9-generic -nogpulib -nogpuinc -mcode-object-version=6 \
// RUN:   -c %S/get-data-isa-name-windows.cl -o %t.o
// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx9-generic -nogpulib -nogpuinc -mcode-object-version=6 \
// RUN:   -shared %S/get-data-isa-name-windows.cl -o %t.so
// RUN: get-data-isa-name-windows %t.o %t.so "amdgcn-amd-amdhsa--gfx9-generic"

__attribute__((visibility("default"))) constant int foo = 0;

void kernel testfn(
    global int *a, const global int *b) {
  *a = *b;
}
