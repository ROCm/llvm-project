// Hardwired-on XNACK is implied by the processor, not an ELF mode selection.
// RUN: llvm-mc -triple=amdgpu12.50-amd-amdhsa --amdhsa-code-object-version=4 -filetype=obj %s | llvm-readobj --file-headers - | FileCheck %s -DFLAGS=449
// RUN: llvm-mc -triple=amdgpu12.50-amd-amdhsa --amdhsa-code-object-version=5 -filetype=obj %s | llvm-readobj --file-headers - | FileCheck %s -DFLAGS=449
// RUN: llvm-mc -triple=amdgpu12.50-amd-amdhsa --amdhsa-code-object-version=6 -filetype=obj %s | llvm-readobj --file-headers - | FileCheck %s -DFLAGS=449
// RUN: llvm-mc -triple=amdgpu12.50s-amd-amdhsa --amdhsa-code-object-version=6 -filetype=obj %s | llvm-readobj --file-headers - | FileCheck %s -DFLAGS=4EB
// RUN: llvm-mc -triple=amdgpu12.51-amd-amdhsa --amdhsa-code-object-version=6 -filetype=obj %s | llvm-readobj --file-headers - | FileCheck %s -DFLAGS=45A
// RUN: llvm-mc -triple=amdgpu12.5-amd-amdhsa --amdhsa-code-object-version=6 --amdgpu-force-generic-version=1 -filetype=obj %s | llvm-readobj --file-headers - | FileCheck %s -DFLAGS=100045B

// CHECK: Flags [ (0x[[FLAGS]])
// CHECK: EF_AMDGPU_FEATURE_SRAMECC_ANY_V4

s_endpgm
