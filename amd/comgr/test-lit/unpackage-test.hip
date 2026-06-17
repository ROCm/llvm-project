// COM: create a bitcode package
//
// RUN: clang -c -x hip --offload-arch=gfx900 \
// RUN: -nogpulib -nogpuinc -emit-llvm \
// RUN: --offload-device-only %s -o %t.gfx900.bc
//
// RUN: clang -c -x hip --offload-arch=gfx1030 \
// RUN: -nogpulib -nogpuinc -emit-llvm \
// RUN: --offload-device-only %s -o %t.gfx1030.bc
//
// RUN: llvm-offload-binary \
// RUN: --image=file=%t.gfx900.bc,triple=amdgcn-amd-amdhsa,arch=gfx900,kind=hip \
// RUN: --image=file=%t.gfx1030.bc,triple=amdgcn-amd-amdhsa,arch=gfx1030,kind=hip \
// RUN: -o %t.package.bc
//
// COM: extract using Comgr
// RUN: unpackage %t.package.bc amdgcn-amd-amdhsa gfx900 %t.gfx900.output.bc
// RUN: llvm-dis %t.gfx900.output.bc -o - | FileCheck --check-prefixes=BOTH,GFX9 %s
//
// RUN: unpackage %t.package.bc amdgcn-amd-amdhsa gfx1030 %t.gfx1030.output.bc
// RUN: llvm-dis %t.gfx1030.output.bc -o - | FileCheck --check-prefixes=BOTH,GFX10 %s
//
// BOTH: target triple = "amdgcn-amd-amdhsa"
// GFX9: "target-cpu"="gfx900"
// GFX10: "target-cpu"="gfx1030"

__attribute__((device))
void add_value(float* a, float* b, float* res) {
    *res = *a + *b;
}
