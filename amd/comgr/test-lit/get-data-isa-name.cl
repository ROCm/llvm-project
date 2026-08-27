// COM: Test Comgr get_data_isa_name() API
// RUN: %python %S/enumerate-isa-check.py %clang %s %t
// XFAIL: *
// COM: Running non-enumerated tests for COV=4 and COV=6
// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx900 -nogpulib \
// RUN:   -nogpuinc -mcode-object-version=4 -c %s -o %t.o
// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx900 -nogpulib \
// RUN:   -nogpuinc -mcode-object-version=4 -shared %s -o %t.so
// RUN: test-get-data-isa-name %t.o %t.so "amdgcn-amd-amdhsa--gfx900"

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx9-generic -nogpulib \
// RUN:   -nogpuinc -mcode-object-version=6 -c %s -o %t.o
// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx9-generic -nogpulib \
// RUN:   -nogpuinc -mcode-object-version=6 -shared %s -o %t.so
// RUN: test-get-data-isa-name %t.o %t.so "amdgcn-amd-amdhsa--gfx9-generic"

__attribute__((visibility("default"))) constant int foo = 0;

void kernel testfn(global int *a, const global int *b) { *a = *b; }
