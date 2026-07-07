// RUN: %clang --offload-arch=gfx1250 --offload-device-only \
// RUN:     --no-gpu-bundle-output -nogpulib -nogpuinc \
// RUN:     %s -o %t.input.elf
// RUN: %llvm-readelf -h %t.input.elf | %FileCheck --check-prefix=INPUT-FLAGS %s
// RUN: %llvm-readelf --notes %t.input.elf | %FileCheck --check-prefix=INPUT-META %s
// RUN: hotswap-rewrite %t.input.elf \
// RUN:     amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:     --output %t.output.elf \
// RUN:   | %FileCheck --check-prefix=STATUS %s
// RUN: %llvm-readelf -h %t.output.elf | %FileCheck --check-prefix=OUTPUT-FLAGS %s
// RUN: %llvm-readelf --notes %t.output.elf | %FileCheck --check-prefix=OUTPUT-META %s
// RUN: %llvm-readelf --section-headers %t.output.elf \
// RUN:   | %FileCheck --check-prefix=SECTIONS %s
// RUN: %llvm-objdump -d %t.output.elf | %FileCheck --check-prefix=DISASM %s

// INPUT-FLAGS: Flags:{{.*}}gfx1250
// INPUT-META: amdhsa.target: amdgcn-amd-amdhsa--gfx1250
// STATUS: RESULT: SUCCESS
// OUTPUT-FLAGS: Flags:{{.*}}gfx1250
// OUTPUT-META: amdhsa.target: amdgcn-amd-amdhsa--gfx1250
// SECTIONS: .text{{.*}}PROGBITS
// DISASM: file format elf64-amdgpu
// DISASM: Disassembly of section .text:
// DISASM: s_endpgm

__attribute__((global))
void add(float *A, float *B, float *C) {
  *C = *A + *B;
}
