; Default O0
; RUN: opt -mtriple=amdgpu10.30-- %s -o %t.default.bc
; RUN: llvm-lto2 run -O0 -cg-opt-level 0 %t.default.bc -o %t.s -r %t.default.bc,test,px -debug-pass-manager 2>&1 | FileCheck %s

; Unified O0
; RUN: opt -unified-lto -thinlto-split-lto-unit -thinlto-bc -mtriple=amdgpu10.30-- %s -o %t.unified.bc
; RUN: llvm-lto2 run -unified-lto=full -O0 -cg-opt-level 0 %t.unified.bc -o %t.s -r %t.unified.bc,test,px -debug-pass-manager 2>&1 | FileCheck %s

; Default O1
; RUN: llvm-lto2 run -O1 -cg-opt-level 1 %t.default.bc -o %t.s -r %t.default.bc,test,px -debug-pass-manager 2>&1 | FileCheck %s

; Unified O1
; RUN: llvm-lto2 run -unified-lto=full -O1 -cg-opt-level 1 %t.unified.bc -o %t.s -r %t.unified.bc,test,px -debug-pass-manager 2>&1 | FileCheck %s

; Default O2
; RUN: llvm-lto2 run -O2 -cg-opt-level 2 %t.default.bc -o %t.s -r %t.default.bc,test,px -debug-pass-manager 2>&1 | FileCheck %s

; Unified O2
; RUN: llvm-lto2 run -unified-lto=full -O2 -cg-opt-level 2 %t.unified.bc -o %t.s -r %t.unified.bc,test,px -debug-pass-manager 2>&1 | FileCheck %s

; Default O3
; RUN: llvm-lto2 run -O3 -cg-opt-level 3 %t.default.bc -o %t.s -r %t.default.bc,test,px -debug-pass-manager 2>&1 | FileCheck %s

; Unified O3
; RUN: llvm-lto2 run -unified-lto=full -O3 -cg-opt-level 3 %t.unified.bc -o %t.s -r %t.unified.bc,test,px -debug-pass-manager 2>&1 | FileCheck %s

; AMDGPU defaults to the New PM for codegen, so both prints come from it: the
; first from the full LTO pipeline, the second from the CG pipeline.

; CHECK: Running pass: AMDGPULowerModuleLDSPass on [module]
; CHECK: Running pass: AMDGPULowerModuleLDSPass on [module]

; The legacy CG pipeline is still reachable on request.

; RUN: llvm-lto2 run -O0 -cg-opt-level 0 %t.default.bc -o %t.s -r %t.default.bc,test,px -force-new-pm-codegen=false -debug-pass-manager -debug-pass=Structure 2>&1 | FileCheck -check-prefix=LEGACY %s
; RUN: llvm-lto2 run -unified-lto=full -O3 -cg-opt-level 3 %t.unified.bc -o %t.s -r %t.unified.bc,test,px -force-new-pm-codegen=false -debug-pass-manager -debug-pass=Structure 2>&1 | FileCheck -check-prefix=LEGACY %s

; First print will be from the New PM during the full LTO pipeline.
; Second print will be from the legacy PM during the CG pipeline.

; LEGACY: Running pass: AMDGPULowerModuleLDSPass on [module]
; LEGACY: ModulePass Manager
; LEGACY:   Lower uses of LDS variables from non-kernel functions

@lds = internal unnamed_addr addrspace(3) global i32 poison, align 4

define amdgpu_kernel void @test() {
entry:
  store i32 1, ptr addrspace(3) @lds
  ret void
}
