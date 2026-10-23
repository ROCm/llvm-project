// REQUIRES: comgr-has-spirv-translator
// COM: Generate a spirv-targeted LLVM IR file from an OpenCL kernel
// RUN: %clang -c -emit-llvm --target=spirv64 %s -o %t.bc

// COM: Translate LLVM IR to SPIRV format
// RUN: %amd-llvm-spirv --spirv-target-env=CL2.0 %t.bc -o %t.spv

// COM: Run Comgr Translator to covert SPIRV back to LLVM IR
// RUN: spirv-translator %t.spv -o %t.translated.bc

// COM: Dissasemble LLVM IR bitcode to LLVM IR text
// RUN: %llvm-dis %t.translated.bc -o - | %FileCheck %s

// COM: Comgr selects the target for standard SPIR-V.

// RUN: spirv-translator --isa amdgcn-amd-amdhsa--gfx900 %t.spv -o %t.gfx900.bc
// RUN: %llvm-dis %t.gfx900.bc -o - | %FileCheck %s --check-prefix=GFX900
// RUN: spirv-translator --isa amdgcn-amd-amdhsa--gfx942 %t.spv -o %t.gfx942.bc
// RUN: %llvm-dis %t.gfx942.bc -o - | %FileCheck %s --check-prefix=GFX942
// RUN: spirv-translator --isa amdgcn-amd-amdhsa--gfx1030 %t.spv \
// RUN:   -o %t.gfx1030.bc
// RUN: %llvm-dis %t.gfx1030.bc -o - | %FileCheck %s --check-prefix=GFX1030

// COM: Verify LLVM IR text
// CHECK: target triple = "amdgpu-amd-amdhsa"
// CHECK: define amdgpu_kernel void @source

// GFX900: target triple = "amdgpu9.00-amd-amdhsa"
// GFX942: target triple = "amdgpu9.42-amd-amdhsa"
// GFX1030: target triple = "amdgpu10.30-amd-amdhsa"

void kernel source(__global int *j) {
  *j += 2;
}

