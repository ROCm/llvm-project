; RUN: llc -mtriple=amdgcn -mcpu=tahiti -amdgpu-use-amdgpu-trackers=1 -debug-only=machine-scheduler < %s 2>&1 | FileCheck --check-prefix=GCN-DEBUG %s
; RUN: llc -mtriple=amdgcn -mcpu=tahiti -amdgpu-use-amdgpu-trackers=0 -debug-only=machine-scheduler < %s 2>&1 | FileCheck --check-prefix=GENERIC-DEBUG %s
; REQUIRES: asserts

; Test that GCN trackers correctly track physical register pressure from inline asm

; GCN-DEBUG-LABEL: test_single_physreg
; GCN-DEBUG: Region register pressure: VGPRs: 1 AGPRs: 0, SGPRs: 5, LVGPR WT: 0, LSGPR WT: 6
; GCN-DEBUG: Pressure after scheduling: VGPRs: 1 AGPRs: 0, SGPRs: 5, LVGPR WT: 0, LSGPR WT: 6

; GENERIC-DEBUG-LABEL: test_single_physreg
; GENERIC-DEBUG: Region register pressure: VGPRs: 1 AGPRs: 0, SGPRs: 4, LVGPR WT: 0, LSGPR WT: 6
; GENERIC-DEBUG: Pressure after scheduling: VGPRs: 1 AGPRs: 0, SGPRs: 4, LVGPR WT: 0, LSGPR WT: 6

define amdgpu_kernel void @test_single_physreg(ptr addrspace(1) %out) {
entry:
  %val = call i32 asm sideeffect "s_mov_b32 $0, 0", "={s10}"()
  store i32 0, ptr addrspace(1) %out
  ret void
}

; Test multiple physical registers

; GCN-DEBUG-LABEL: test_multiple_physregs
; GCN-DEBUG: Region register pressure: VGPRs: 1 AGPRs: 0, SGPRs: 6, LVGPR WT: 0, LSGPR WT: 6
; GCN-DEBUG: Pressure after scheduling: VGPRs: 1 AGPRs: 0, SGPRs: 6, LVGPR WT: 0, LSGPR WT: 6

; GENERIC-DEBUG-LABEL: test_multiple_physregs
; GENERIC-DEBUG: Region register pressure: VGPRs: 1 AGPRs: 0, SGPRs: 4, LVGPR WT: 0, LSGPR WT: 6
; GENERIC-DEBUG: Pressure after scheduling: VGPRs: 1 AGPRs: 0, SGPRs: 4, LVGPR WT: 0, LSGPR WT: 6

define amdgpu_kernel void @test_multiple_physregs(ptr addrspace(1) %out) {
entry:
  %result = call { i32, i32 } asm sideeffect "s_mov_b32 $0, 0; s_mov_b32 $1, 1", "={s10},={s11}"()
  store i32 0, ptr addrspace(1) %out
  ret void
}

; Test physical register with virtual registers

; GCN-DEBUG-LABEL: test_physreg_with_vreg
; GCN-DEBUG: Region register pressure: VGPRs: 1 AGPRs: 0, SGPRs: 9, LVGPR WT: 0, LSGPR WT: 12
; GCN-DEBUG: Pressure after scheduling: VGPRs: 1 AGPRs: 0, SGPRs: 8, LVGPR WT: 0, LSGPR WT: 12

; GENERIC-DEBUG-LABEL: test_physreg_with_vreg
; GENERIC-DEBUG: Region register pressure: VGPRs: 1 AGPRs: 0, SGPRs: 8, LVGPR WT: 0, LSGPR WT: 12
; GENERIC-DEBUG: Pressure after scheduling: VGPRs: 1 AGPRs: 0, SGPRs: 7, LVGPR WT: 0, LSGPR WT: 12

define amdgpu_kernel void @test_physreg_with_vreg(ptr addrspace(1) %in, ptr addrspace(1) %out) {
entry:
  %asm_val = call i32 asm sideeffect "s_mov_b32 $0, 0", "={s10}"()
  %val = load i32, ptr addrspace(1) %in
  store i32 %val, ptr addrspace(1) %out
  ret void
}

; Test that we don't inflate pressure when not using GCN trackers

; GCN-DEBUG-LABEL: test_no_inflation

; GENERIC-DEBUG-LABEL: test_no_inflation

define amdgpu_kernel void @test_no_inflation() {
entry:
  ret void
}

; Test early-clobber constraint

; GCN-DEBUG-LABEL: test_early_clobber
; GCN-DEBUG: Region register pressure: VGPRs: 1 AGPRs: 0, SGPRs: 6, LVGPR WT: 0, LSGPR WT: 6
; GCN-DEBUG: Pressure after scheduling: VGPRs: 1 AGPRs: 0, SGPRs: 6, LVGPR WT: 0, LSGPR WT: 6

; GENERIC-DEBUG-LABEL: test_early_clobber
; GENERIC-DEBUG: Region register pressure: VGPRs: 1 AGPRs: 0, SGPRs: 5, LVGPR WT: 0, LSGPR WT: 6
; GENERIC-DEBUG: Pressure after scheduling: VGPRs: 1 AGPRs: 0, SGPRs: 5, LVGPR WT: 0, LSGPR WT: 6

define amdgpu_kernel void @test_early_clobber(ptr addrspace(1) %out) {
entry:
  %val = call i32 asm sideeffect "s_mov_b32 $0, 0", "=&{s10}"()
  store i32 %val, ptr addrspace(1) %out
  ret void
}

; Test physical register input

; GCN-DEBUG-LABEL: test_physreg_input
; GCN-DEBUG: Region register pressure: VGPRs: 1 AGPRs: 0, SGPRs: 5, LVGPR WT: 0, LSGPR WT: 6
; GCN-DEBUG: Pressure after scheduling: VGPRs: 1 AGPRs: 0, SGPRs: 5, LVGPR WT: 0, LSGPR WT: 6

; GENERIC-DEBUG-LABEL: test_physreg_input
; GENERIC-DEBUG: Region register pressure: VGPRs: 1 AGPRs: 0, SGPRs: 4, LVGPR WT: 0, LSGPR WT: 6
; GENERIC-DEBUG: Pressure after scheduling: VGPRs: 1 AGPRs: 0, SGPRs: 4, LVGPR WT: 0, LSGPR WT: 6

define amdgpu_kernel void @test_physreg_input(ptr addrspace(1) %out) {
entry:
  %val = call i32 asm sideeffect "s_mov_b32 s10, 5; s_add_u32 $0, s10, 1", "={s11}"()
  store i32 0, ptr addrspace(1) %out
  ret void
}

; Test virtual and physical register overlap

; GCN-DEBUG-LABEL: test_vreg_and_physreg_overlap
; GCN-DEBUG: Region register pressure: VGPRs: 3 AGPRs: 0, SGPRs: 14, LVGPR WT: 0, LSGPR WT: 16
; GCN-DEBUG: Pressure after scheduling: VGPRs: 3 AGPRs: 0, SGPRs: 12, LVGPR WT: 0, LSGPR WT: 16

; GENERIC-DEBUG-LABEL: test_vreg_and_physreg_overlap
; GENERIC-DEBUG: Region register pressure: VGPRs: 3 AGPRs: 0, SGPRs: 12, LVGPR WT: 0, LSGPR WT: 16
; GENERIC-DEBUG: Pressure after scheduling: VGPRs: 3 AGPRs: 0, SGPRs: 10, LVGPR WT: 0, LSGPR WT: 16

define amdgpu_kernel void @test_vreg_and_physreg_overlap(ptr addrspace(1) %in1, ptr addrspace(1) %in2, ptr addrspace(1) %out) {
entry:
  %result = call { i32, i32 } asm sideeffect "s_mov_b32 $0, 0; s_mov_b32 $1, 1", "={s10},={s11}"()
  %val1 = load i32, ptr addrspace(1) %in1
  %val2 = load i32, ptr addrspace(1) %in2
  %sum = add i32 %val1, %val2
  store i32 %sum, ptr addrspace(1) %out
  ret void
}
