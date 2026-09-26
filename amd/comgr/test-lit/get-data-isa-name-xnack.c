// RUN: %yaml2obj --docnum=1 %S/get-data-isa-name-xnack.yaml -o %t.always.o
// RUN: test-get-data-isa-name %t.always.o %t.always.o \
// RUN:   amdgcn-amd-amdhsa--gfx1250
// RUN: %yaml2obj --docnum=2 %S/get-data-isa-name-xnack.yaml -o %t.on.o
// RUN: test-get-data-isa-name %t.on.o %t.on.o \
// RUN:   amdgcn-amd-amdhsa--gfx900:xnack+
// RUN: %yaml2obj --docnum=3 %S/get-data-isa-name-xnack.yaml -o %t.off.o
// RUN: test-get-data-isa-name %t.off.o %t.off.o \
// RUN:   amdgcn-amd-amdhsa--gfx900:xnack-
//
// Hardwired-on XNACK is encoded in ELF flags but is not a target-ID modifier.
// Selectable XNACK modes must still round-trip.
