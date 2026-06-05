// COM: Test Comgr get_data_isa_name() API
// RUN: %run_all_isas

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx900 -nogpulib -nogpuinc -mcode-object-version=4 \
// RUN:   -c %S/get-data-isa-name.cl -o %t.o
// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx900 -nogpulib -nogpuinc -mcode-object-version=4 \
// RUN:   -shared %S/get-data-isa-name.cl -o %t.so
// RUN: get-data-isa-name %t.o %t.so "amdgcn-amd-amdhsa--gfx900"

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx9-generic -nogpulib -nogpuinc -mcode-object-version=6 \
// RUN:   -c %S/get-data-isa-name.cl -o %t.o
// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx9-generic -nogpulib -nogpuinc -mcode-object-version=6 \
// RUN:   -shared %S/get-data-isa-name.cl -o %t.so
// RUN: get-data-isa-name %t.o %t.so "amdgcn-amd-amdhsa--gfx9-generic"

__attribute__((visibility("default"))) constant int foo = 0;

void kernel testfn(
    global int *a, const global int *b) {
  *a = *b;
}
