; REQUIRES: asserts

; RUN: llc -amdgpu-late-wave-transform=1 -verify-machineinstrs=0 -mtriple=amdgcn-amd-amdhsa -debug-pass=Structure -filetype=null %s 2>&1 | FileCheck -check-prefix=DEFAULT %s
; RUN: llc -amdgpu-late-wave-transform=1 -verify-machineinstrs=0 -sgpr-regalloc=greedy -wwm-regalloc=greedy -vgpr-regalloc=greedy -mtriple=amdgcn-amd-amdhsa -debug-pass=Structure -filetype=null %s 2>&1 | FileCheck -check-prefix=DEFAULT %s

; RUN: llc -amdgpu-late-wave-transform=1 -verify-machineinstrs=0 -O0 -mtriple=amdgcn-amd-amdhsa -debug-pass=Structure -filetype=null %s 2>&1 | FileCheck -check-prefix=O0 %s

; RUN: llc -amdgpu-late-wave-transform=1 -verify-machineinstrs=0 -wwm-regalloc=basic -vgpr-regalloc=basic -mtriple=amdgcn-amd-amdhsa -debug-pass=Structure -filetype=null %s 2>&1 | FileCheck -check-prefix=DEFAULT-BASIC %s
; RUN: llc -amdgpu-late-wave-transform=1 -verify-machineinstrs=0 -sgpr-regalloc=basic -mtriple=amdgcn-amd-amdhsa -debug-pass=Structure -filetype=null %s 2>&1 | FileCheck -check-prefix=BASIC-DEFAULT %s
; RUN: llc -amdgpu-late-wave-transform=1 -verify-machineinstrs=0 -sgpr-regalloc=basic -wwm-regalloc=basic -vgpr-regalloc=basic -mtriple=amdgcn-amd-amdhsa -debug-pass=Structure -filetype=null %s 2>&1 | FileCheck -check-prefix=BASIC-BASIC %s

; RUN: not llc -verify-machineinstrs=0 -regalloc=basic -mtriple=amdgcn-amd-amdhsa -debug-pass=Structure -filetype=null %s 2>&1 | FileCheck -check-prefix=REGALLOC %s
; RUN: not llc -verify-machineinstrs=0 -regalloc=fast -O0 -mtriple=amdgcn-amd-amdhsa -debug-pass=Structure -filetype=null %s 2>&1 | FileCheck -check-prefix=REGALLOC %s


; REGALLOC: -regalloc not supported with amdgcn. Use -sgpr-regalloc, -wwm-regalloc, and -vgpr-regalloc

; DEFAULT: Greedy Register Allocator
; DEFAULT-NEXT: GCN NSA Reassign
; DEFAULT-NEXT: AMDGPU Rewrite AGPR-Copy-MFMA
; DEFAULT-NEXT: Virtual Register Rewriter
; DEFAULT-NEXT: AMDGPU Pre Wave Transform
; DEFAULT-NEXT: Machine Cycle Info Analysis
; DEFAULT-NEXT: AMDGPU Control Flow Wave Transform
; DEFAULT: Greedy Register Allocator
; DEFAULT-NEXT: Virtual Register Rewriter
; DEFAULT-NEXT: Stack Slot Coloring
; DEFAULT-NEXT: AMDGPU Reserve Allocated VGPRs
; DEFAULT-NEXT: Machine Cycle Info Analysis
; DEFAULT-NEXT: SI lower SGPR spill instructions
; DEFAULT: Greedy Register Allocator
; DEFAULT-NEXT: SI Lower WWM Copies
; DEFAULT-NEXT: Virtual Register Rewriter
; DEFAULT-NEXT: AMDGPU Reserve WWM Registers
; DEFAULT-NEXT: AMDGPU Mark Last Scratch Load
; DEFAULT-NEXT: Stack Slot Coloring

; O0: Fast Register Allocator
; O0-NEXT: AMDGPU Pre Wave Transform
; O0-NEXT: Machine Cycle Info Analysis
; O0-NEXT: AMDGPU Control Flow Wave Transform
; O0-NEXT: Fast Register Allocator
; O0-NEXT: AMDGPU Reserve Allocated VGPRs
; O0-NEXT: Machine Cycle Info Analysis
; O0-NEXT: SI lower SGPR spill instructions
; O0-NEXT: Fast Register Allocator
; O0-NEXT: SI Lower WWM Copies
; O0-NEXT: AMDGPU Reserve WWM Registers
; O0-NEXT: SI Fix VGPR copies




; BASIC-DEFAULT: Debug Variable Analysis
; BASIC-DEFAULT-NEXT: Live Stack Slot Analysis
; BASIC-DEFAULT-NEXT: Bundle Machine CFG Edges
; BASIC-DEFAULT-NEXT: Spill Code Placement Analysis
; BASIC-DEFAULT-NEXT: Lazy Machine Block Frequency Analysis
; BASIC-DEFAULT-NEXT: Machine Optimization Remark Emitter
; BASIC-DEFAULT-NEXT: Greedy Register Allocator
; BASIC-DEFAULT-NEXT: GCN NSA Reassign
; BASIC-DEFAULT-NEXT: AMDGPU Rewrite AGPR-Copy-MFMA
; BASIC-DEFAULT-NEXT: Virtual Register Rewriter
; BASIC-DEFAULT-NEXT: AMDGPU Pre Wave Transform
; BASIC-DEFAULT-NEXT: Machine Cycle Info Analysis
; BASIC-DEFAULT-NEXT: AMDGPU Control Flow Wave Transform
; BASIC-DEFAULT: Basic Register Allocator
; BASIC-DEFAULT-NEXT: Virtual Register Rewriter
; BASIC-DEFAULT-NEXT: Stack Slot Coloring
; BASIC-DEFAULT-NEXT: AMDGPU Reserve Allocated VGPRs
; BASIC-DEFAULT-NEXT: Machine Cycle Info Analysis
; BASIC-DEFAULT-NEXT: SI lower SGPR spill instructions
; BASIC-DEFAULT: Greedy Register Allocator
; BASIC-DEFAULT-NEXT: SI Lower WWM Copies
; BASIC-DEFAULT-NEXT: Virtual Register Rewriter
; BASIC-DEFAULT-NEXT: AMDGPU Reserve WWM Registers
; BASIC-DEFAULT-NEXT: AMDGPU Mark Last Scratch Load
; BASIC-DEFAULT-NEXT: Stack Slot Coloring



; DEFAULT-BASIC: Basic Register Allocator
; DEFAULT-BASIC-NEXT: GCN NSA Reassign
; DEFAULT-BASIC-NEXT: AMDGPU Rewrite AGPR-Copy-MFMA
; DEFAULT-BASIC-NEXT: Virtual Register Rewriter
; DEFAULT-BASIC-NEXT: AMDGPU Pre Wave Transform
; DEFAULT-BASIC-NEXT: Machine Cycle Info Analysis
; DEFAULT-BASIC-NEXT: AMDGPU Control Flow Wave Transform
; DEFAULT-BASIC: Greedy Register Allocator
; DEFAULT-BASIC-NEXT: Virtual Register Rewriter
; DEFAULT-BASIC-NEXT: Stack Slot Coloring
; DEFAULT-BASIC-NEXT: AMDGPU Reserve Allocated VGPRs
; DEFAULT-BASIC-NEXT: Machine Cycle Info Analysis
; DEFAULT-BASIC-NEXT: SI lower SGPR spill instructions
; DEFAULT-BASIC: Basic Register Allocator
; DEFAULT-BASIC-NEXT: SI Lower WWM Copies
; DEFAULT-BASIC-NEXT: Virtual Register Rewriter
; DEFAULT-BASIC-NEXT: AMDGPU Reserve WWM Registers
; DEFAULT-BASIC-NEXT: AMDGPU Mark Last Scratch Load
; DEFAULT-BASIC-NEXT: Stack Slot Coloring



; BASIC-BASIC: Debug Variable Analysis
; BASIC-BASIC-NEXT: Live Stack Slot Analysis
; BASIC-BASIC-NEXT: Machine Natural Loop Construction
; BASIC-BASIC-NEXT: Machine Block Frequency Analysis
; BASIC-BASIC-NEXT: Basic Register Allocator
; BASIC-BASIC-NEXT: GCN NSA Reassign
; BASIC-BASIC-NEXT: AMDGPU Rewrite AGPR-Copy-MFMA
; BASIC-BASIC-NEXT: Virtual Register Rewriter
; BASIC-BASIC-NEXT: AMDGPU Pre Wave Transform
; BASIC-BASIC-NEXT: Machine Cycle Info Analysis
; BASIC-BASIC-NEXT: AMDGPU Control Flow Wave Transform
; BASIC-BASIC: Basic Register Allocator
; BASIC-BASIC-NEXT: Virtual Register Rewriter
; BASIC-BASIC-NEXT: Stack Slot Coloring
; BASIC-BASIC-NEXT: AMDGPU Reserve Allocated VGPRs
; BASIC-BASIC-NEXT: Machine Cycle Info Analysis
; BASIC-BASIC-NEXT: SI lower SGPR spill instructions
; BASIC-BASIC: Basic Register Allocator
; BASIC-BASIC-NEXT: SI Lower WWM Copies
; BASIC-BASIC-NEXT: Virtual Register Rewriter
; BASIC-BASIC-NEXT: AMDGPU Reserve WWM Registers
; BASIC-BASIC-NEXT: AMDGPU Mark Last Scratch Load
; BASIC-BASIC-NEXT: Stack Slot Coloring


declare void @bar()

; Something with some CSR SGPR spills
define void @foo() {
  call void asm sideeffect "; clobber", "~{s33}"()
  call void @bar()
  ret void
}

; Block live out spills with fast regalloc
define amdgpu_kernel void @control_flow(i1 %cond) {
  %s33 = call i32 asm sideeffect "; clobber", "={s33}"()
  br i1 %cond, label %bb0, label %bb1

bb0:
   call void asm sideeffect "; use %0", "s"(i32 %s33)
   br label %bb1

bb1:
  ret void
}
