// COM: Test Comgr get_data_isa_name() API
// RUN: mapfile -t isa_arr < <(./isa-enumeration)
// RUN: for isa in ${isa_arr[@]}; do \
// RUN: gpu=${isa##*--}; \
// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=$gpu -nogpulib -nogpuinc \
// RUN:   -c %S/comgr-sources/gtgra.cl -o %t.o; \
// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=$gpu -nogpulib -nogpuinc \
// RUN:   -shared %S/comgr-sources/gtgra.cl -o %t.so; \
// RUN: get-data-isa-name %t.o %t.so $isa; \
// RUN: done
