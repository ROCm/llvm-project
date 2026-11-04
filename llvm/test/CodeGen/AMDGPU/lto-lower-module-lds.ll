
; Default O0
; RUN: opt -mtriple=amdgpu10.30-- %s -o %t.default.bc
; RUN: llvm-lto2 run -O0 -cg-opt-level 0 %t.default.bc -o %t.s -r %t.default.bc,test,px -debug-pass-manager -debug-pass=Structure 2>&1 | FileCheck %s

; Unified O0
; RUN: opt -unified-lto -thinlto-split-lto-unit -thinlto-bc -mtriple=amdgpu10.30-- %s -o %t.unified.bc
; RUN: llvm-lto2 run -unified-lto=full -O0 -cg-opt-level 0 %t.unified.bc -o %t.s -r %t.unified.bc,test,px -debug-pass-manager -debug-pass=Structure 2>&1 | FileCheck %s

; Default O1
; RUN: llvm-lto2 run -O1 -cg-opt-level 1 %t.default.bc -o %t.s -r %t.default.bc,test,px -debug-pass-manager -debug-pass=Structure 2>&1 | FileCheck %s

; Unified O1
; RUN: llvm-lto2 run -unified-lto=full -O1 -cg-opt-level 1 %t.unified.bc -o %t.s -r %t.unified.bc,test,px -debug-pass-manager -debug-pass=Structure 2>&1 | FileCheck %s

; Default O2
; RUN: llvm-lto2 run -O2 -cg-opt-level 2 %t.default.bc -o %t.s -r %t.default.bc,test,px -debug-pass-manager -debug-pass=Structure 2>&1 | FileCheck %s

; Unified O2
; RUN: llvm-lto2 run -unified-lto=full -O2 -cg-opt-level 2 %t.unified.bc -o %t.s -r %t.unified.bc,test,px -debug-pass-manager -debug-pass=Structure 2>&1 | FileCheck %s

; Default O3
; RUN: llvm-lto2 run -O3 -cg-opt-level 3 %t.default.bc -o %t.s -r %t.default.bc,test,px -debug-pass-manager -debug-pass=Structure 2>&1 | FileCheck %s

; Unified O3
; RUN: llvm-lto2 run -unified-lto=full -O3 -cg-opt-level 3 %t.unified.bc -o %t.s -r %t.unified.bc,test,px -debug-pass-manager -debug-pass=Structure 2>&1 | FileCheck %s

; The same, with the New PM driving the CG pipeline as well.

; Default O0
; RUN: llvm-lto2 run -O0 -cg-opt-level 0 %t.default.bc -o %t.s -r %t.default.bc,test,px -debug-pass-manager -force-new-pm-codegen=true 2>&1 | FileCheck --check-prefix=NEWPM %s

; Unified O0
; RUN: llvm-lto2 run -unified-lto=full -O0 -cg-opt-level 0 %t.unified.bc -o %t.s -r %t.unified.bc,test,px -debug-pass-manager -force-new-pm-codegen=true 2>&1 | FileCheck --check-prefix=NEWPM %s

; Default O3
; RUN: llvm-lto2 run -O3 -cg-opt-level 3 %t.default.bc -o %t.s -r %t.default.bc,test,px -debug-pass-manager -force-new-pm-codegen=true 2>&1 | FileCheck --check-prefix=NEWPM %s

; Unified O3
; RUN: llvm-lto2 run -unified-lto=full -O3 -cg-opt-level 3 %t.unified.bc -o %t.s -r %t.unified.bc,test,px -debug-pass-manager -force-new-pm-codegen=true 2>&1 | FileCheck --check-prefix=NEWPM %s

; First print will be from the New PM during the full LTO pipeline.
; Second print will be from the legacy PM during the CG pipeline.

; CHECK: Running pass: AMDGPULowerModuleLDSPass on [module]
; CHECK: ModulePass Manager
; CHECK:   Lower uses of LDS variables from non-kernel functions

; Both prints come from the New PM: the first from the full LTO pipeline, the
; second from the CG pipeline.

; NEWPM: Running pass: AMDGPULowerModuleLDSPass on [module]
; NEWPM: Running pass: AMDGPULowerModuleLDSPass on [module]

@lds = internal unnamed_addr addrspace(3) global i32 poison, align 4

define amdgpu_kernel void @test() {
entry:
  store i32 1, ptr addrspace(3) @lds
  ret void
}
