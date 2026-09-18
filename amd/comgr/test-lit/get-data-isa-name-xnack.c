// COM: Always-on XNACK must not appear as a selectable ISA modifier, even
// COM: when the code object's ELF flags explicitly record XNACK as on.

// RUN: %yaml2obj %S/get-data-isa-name-xnack.yaml -D ABI=2 -o %t.o
// RUN: %yaml2obj %S/get-data-isa-name-xnack.yaml -D ABI=2 -D TYPE=ET_DYN -o %t.so
// RUN: test-get-data-isa-name %t.o %t.so amdgcn-amd-amdhsa--gfx1250

// RUN: %yaml2obj %S/get-data-isa-name-xnack.yaml -D ABI=3 -o %t.o
// RUN: %yaml2obj %S/get-data-isa-name-xnack.yaml -D ABI=3 -D TYPE=ET_DYN -o %t.so
// RUN: test-get-data-isa-name %t.o %t.so amdgcn-amd-amdhsa--gfx1250

// RUN: %yaml2obj %S/get-data-isa-name-xnack.yaml -o %t.o
// RUN: %yaml2obj %S/get-data-isa-name-xnack.yaml -D TYPE=ET_DYN -o %t.so
// RUN: test-get-data-isa-name %t.o %t.so amdgcn-amd-amdhsa--gfx1250

// RUN: %yaml2obj %S/get-data-isa-name-xnack.yaml -D GPU=GFX1251 -o %t.o
// RUN: %yaml2obj %S/get-data-isa-name-xnack.yaml -D GPU=GFX1251 -D TYPE=ET_DYN -o %t.so
// RUN: test-get-data-isa-name %t.o %t.so amdgcn-amd-amdhsa--gfx1251

// RUN: %yaml2obj %S/get-data-isa-name-xnack.yaml -D GPU=GFX1250_STRICT -o %t.o
// RUN: %yaml2obj %S/get-data-isa-name-xnack.yaml -D GPU=GFX1250_STRICT -D TYPE=ET_DYN -o %t.so
// RUN: test-get-data-isa-name %t.o %t.so amdgcn-amd-amdhsa--gfx1250-strict

// COM: Selectable XNACK modes must still appear in the ISA name.
// RUN: %yaml2obj %S/get-data-isa-name-xnack.yaml -D GPU=GFX900 -o %t.o
// RUN: %yaml2obj %S/get-data-isa-name-xnack.yaml -D GPU=GFX900 -D TYPE=ET_DYN -o %t.so
// RUN: test-get-data-isa-name %t.o %t.so amdgcn-amd-amdhsa--gfx900:xnack+

// RUN: %yaml2obj %S/get-data-isa-name-xnack.yaml -D GPU=GFX900 -D XNACK=OFF -o %t.o
// RUN: %yaml2obj %S/get-data-isa-name-xnack.yaml -D GPU=GFX900 -D XNACK=OFF -D TYPE=ET_DYN -o %t.so
// RUN: test-get-data-isa-name %t.o %t.so amdgcn-amd-amdhsa--gfx900:xnack-
