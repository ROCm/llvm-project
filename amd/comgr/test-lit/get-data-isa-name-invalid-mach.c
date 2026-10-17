// COM: Test that amd_comgr_get_data_isa_name() rejects a code object whose
// COM: EF_AMDGPU_MACH does not name an AMDGCN processor, instead of reporting
// COM: an ISA name the rest of the API cannot resolve.

// RUN: %yaml2obj --docnum=1 %S/get-data-isa-name-invalid-mach.yaml -o %t.r600.o
// RUN: %yaml2obj --docnum=2 %S/get-data-isa-name-invalid-mach.yaml -o %t.none.o
// RUN: test-get-data-isa-name --expect-fail %t.r600.o %t.none.o
