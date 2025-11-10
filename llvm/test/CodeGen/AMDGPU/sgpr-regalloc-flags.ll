; REQUIRES: asserts

; RUN: llc -verify-machineinstrs=0 -mtriple=amdgcn-amd-amdhsa -debug-pass-manager -filetype=null %s 2>&1 | FileCheck -check-prefix=DEFAULT %s
; RUN: llc -verify-machineinstrs=0 --regalloc-npm='greedy<sgpr>,greedy<wwm>,greedy<vgpr>' -mtriple=amdgcn-amd-amdhsa -debug-pass-manager -filetype=null %s 2>&1 | FileCheck -check-prefix=DEFAULT %s

; RUN: llc -verify-machineinstrs=0 -O0 -mtriple=amdgcn-amd-amdhsa -debug-pass-manager -filetype=null %s 2>&1 | FileCheck -check-prefix=O0 %s

; TODO: Basic regalloc to be ported to NPM
; RUN: llc -enable-new-pm=0 -verify-machineinstrs=0 -wwm-regalloc=basic -vgpr-regalloc=basic -mtriple=amdgcn-amd-amdhsa -debug-pass=Structure -filetype=null %s 2>&1 | FileCheck -check-prefix=DEFAULT-BASIC %s
; RUN: llc -enable-new-pm=0 -verify-machineinstrs=0 -sgpr-regalloc=basic -mtriple=amdgcn-amd-amdhsa -debug-pass=Structure -filetype=null %s 2>&1 | FileCheck -check-prefix=BASIC-DEFAULT %s
; RUN: llc -enable-new-pm=0 -verify-machineinstrs=0 -sgpr-regalloc=basic -wwm-regalloc=basic -vgpr-regalloc=basic -mtriple=amdgcn-amd-amdhsa -debug-pass=Structure -filetype=null %s 2>&1 | FileCheck -check-prefix=BASIC-BASIC %s

; Only matching pass names for NPM (ignore analysis/invalidation lines)

; DEFAULT: Running pass: RAGreedyPass on foo
; DEFAULT: Running pass: VirtRegRewriterPass on foo
; DEFAULT: Running pass: StackSlotColoringPass on foo
; DEFAULT: Running pass: SILowerSGPRSpillsPass on foo
; DEFAULT: Running pass: SIPreAllocateWWMRegsPass on foo
; DEFAULT: Running pass: RAGreedyPass on foo
; DEFAULT: Running pass: SILowerWWMCopiesPass on foo
; DEFAULT: Running pass: VirtRegRewriterPass on foo
; DEFAULT: Running pass: AMDGPUReserveWWMRegsPass on foo
; DEFAULT: Running pass: RAGreedyPass on foo
; DEFAULT: Running pass: GCNNSAReassignPass on foo
; DEFAULT: Running pass: AMDGPURewriteAGPRCopyMFMAPass on foo
; DEFAULT: Running pass: VirtRegRewriterPass on foo
; DEFAULT: Running pass: AMDGPUMarkLastScratchLoadPass on foo
; DEFAULT: Running pass: StackSlotColoringPass on foo

; O0: Running pass: RegAllocFastPass on foo
; O0: Running pass: SILowerSGPRSpillsPass on foo
; O0: Running pass: SIPreAllocateWWMRegsPass on foo
; O0: Running pass: RegAllocFastPass on foo
; O0: Running pass: SILowerWWMCopiesPass on foo
; O0: Running pass: AMDGPUReserveWWMRegsPass on foo
; O0: Running pass: RegAllocFastPass on foo
; O0: Running pass: SIFixVGPRCopiesPass on foo



; BASIC-DEFAULT: Debug Variable Analysis
; BASIC-DEFAULT-NEXT: Live Stack Slot Analysis
; BASIC-DEFAULT-NEXT: Machine Natural Loop Construction
; BASIC-DEFAULT-NEXT: Machine Block Frequency Analysis
; BASIC-DEFAULT-NEXT: Virtual Register Map
; BASIC-DEFAULT-NEXT: Live Register Matrix
; BASIC-DEFAULT-NEXT: Basic Register Allocator
; BASIC-DEFAULT-NEXT: Virtual Register Rewriter
; BASIC-DEFAULT-NEXT: Stack Slot Coloring
; BASIC-DEFAULT-NEXT: SI lower SGPR spill instructions
; BASIC-DEFAULT-NEXT: Virtual Register Map
; BASIC-DEFAULT-NEXT: Live Register Matrix
; BASIC-DEFAULT-NEXT: SI Pre-allocate WWM Registers
; BASIC-DEFAULT-NEXT: Live Stack Slot Analysis
; BASIC-DEFAULT-NEXT: Bundle Machine CFG Edges
; BASIC-DEFAULT-NEXT: Spill Code Placement Analysis
; BASIC-DEFAULT-NEXT: Lazy Machine Block Frequency Analysis
; BASIC-DEFAULT-NEXT: Machine Optimization Remark Emitter
; BASIC-DEFAULT-NEXT: Greedy Register Allocator
; BASIC-DEFAULT-NEXT: SI Lower WWM Copies
; BASIC-DEFAULT-NEXT: Virtual Register Rewriter
; BASIC-DEFAULT-NEXT: AMDGPU Reserve WWM Registers
; BASIC-DEFAULT-NEXT: Virtual Register Map
; BASIC-DEFAULT-NEXT: Live Register Matrix
; BASIC-DEFAULT-NEXT: Greedy Register Allocator
; BASIC-DEFAULT-NEXT: GCN NSA Reassign
; BASIC-DEFAULT-NEXT: AMDGPU Rewrite AGPR-Copy-MFMA
; BASIC-DEFAULT-NEXT: Virtual Register Rewriter
; BASIC-DEFAULT-NEXT: AMDGPU Mark Last Scratch Load
; BASIC-DEFAULT-NEXT: Stack Slot Coloring



; DEFAULT-BASIC: Greedy Register Allocator
; DEFAULT-BASIC-NEXT: Virtual Register Rewriter
; DEFAULT-BASIC-NEXT: Stack Slot Coloring
; DEFAULT-BASIC-NEXT: SI lower SGPR spill instructions
; DEFAULT-BASIC-NEXT: Virtual Register Map
; DEFAULT-BASIC-NEXT: Live Register Matrix
; DEFAULT-BASIC-NEXT: SI Pre-allocate WWM Registers
; DEFAULT-BASIC-NEXT: Live Stack Slot Analysis
; DEFAULT-BASIC-NEXT: Basic Register Allocator
; DEFAULT-BASIC-NEXT: SI Lower WWM Copies
; DEFAULT-BASIC-NEXT: Virtual Register Rewriter
; DEFAULT-BASIC-NEXT: AMDGPU Reserve WWM Registers
; DEFAULT-BASIC-NEXT: Virtual Register Map
; DEFAULT-BASIC-NEXT: Live Register Matrix
; DEFAULT-BASIC-NEXT: Basic Register Allocator
; DEFAULT-BASIC-NEXT: GCN NSA Reassign
; DEFAULT-BASIC-NEXT: AMDGPU Rewrite AGPR-Copy-MFMA
; DEFAULT-BASIC-NEXT: Virtual Register Rewriter
; DEFAULT-BASIC-NEXT: AMDGPU Mark Last Scratch Load
; DEFAULT-BASIC-NEXT: Stack Slot Coloring



; BASIC-BASIC: Debug Variable Analysis
; BASIC-BASIC-NEXT: Live Stack Slot Analysis
; BASIC-BASIC-NEXT: Machine Natural Loop Construction
; BASIC-BASIC-NEXT: Machine Block Frequency Analysis
; BASIC-BASIC-NEXT: Virtual Register Map
; BASIC-BASIC-NEXT: Live Register Matrix
; BASIC-BASIC-NEXT: Basic Register Allocator
; BASIC-BASIC-NEXT: Virtual Register Rewriter
; BASIC-BASIC-NEXT: Stack Slot Coloring
; BASIC-BASIC-NEXT: SI lower SGPR spill instructions
; BASIC-BASIC-NEXT: Virtual Register Map
; BASIC-BASIC-NEXT: Live Register Matrix
; BASIC-BASIC-NEXT: SI Pre-allocate WWM Registers
; BASIC-BASIC-NEXT: Live Stack Slot Analysis
; BASIC-BASIC-NEXT: Basic Register Allocator
; BASIC-BASIC-NEXT: SI Lower WWM Copies
; BASIC-BASIC-NEXT: Virtual Register Rewriter
; BASIC-BASIC-NEXT: AMDGPU Reserve WWM Registers
; BASIC-BASIC-NEXT: Virtual Register Map
; BASIC-BASIC-NEXT: Live Register Matrix
; BASIC-BASIC-NEXT: Basic Register Allocator
; BASIC-BASIC-NEXT: GCN NSA Reassign
; BASIC-BASIC-NEXT: AMDGPU Rewrite AGPR-Copy-MFMA
; BASIC-BASIC-NEXT: Virtual Register Rewriter
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
