// COM: Test Comgr get_data_isa_name() API
// REQUIRES: system-linux
// RUN: mapfile -t isa_arr < <(./isa-enumeration)
// RUN: for isa in ${isa_arr[@]}; do \
// RUN: gpu=${isa##*--}; \

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=$gpu -nogpulib -nogpuinc -mcode-object-version=4\
// RUN:   -c %S/get-data-isa-name.cl -o %t.o; \
// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=$gpu -nogpulib -nogpuinc -mcode-object-version=4\
// RUN:   -c %S/get-data-isa-name.cl -o %t.o; \
// RUN:   -shared %S/get-data-isa-name.cl -o %t.so; \
// RUN: get-data-isa-name %t.o %t.so $isa; \

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=$gpu -nogpulib -nogpuinc -mcode-object-version=6\
// RUN:   -c %S/get-data-isa-name.cl -o %t.o; \
// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=$gpu -nogpulib -nogpuinc -mcode-object-version=6\
// RUN:   -shared %S/get-data-isa-name.cl -o %t.so; \
// RUN: get-data-isa-name %t.o %t.so $isa; \

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=$gpu -nogpulib -nogpuinc \
// RUN:   -c %S/get-data-isa-name.cl -o %t.o; \
// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=$gpu -nogpulib -nogpuinc \
// RUN:   -shared %S/get-data-isa-name.cl -o %t.so; \
// RUN: get-data-isa-name %t.o %t.so $isa; \
// RUN: done

__attribute__((visibility("default"))) constant int foo = 0;

void kernel testfn(
    global int *a, const global int *b) {
  *a = *b;
}
