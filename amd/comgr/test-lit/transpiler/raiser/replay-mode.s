; REQUIRES: comgr-has-transpiler, comgr-has-llc

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir > %t.ll
; RUN: %FileCheck %s --check-prefix=IR --implicit-check-not=llvm.amdgcn.s.setreg < %t.ll
; RUN: %opt -passes='default<O2>' %t.ll -o %t.bc
; RUN: %llc -mtriple=amdgpu9.42-amd-amdhsa -filetype=obj %t.bc -o %t.target.o
; RUN: %llvm-objdump -d %t.target.o | %FileCheck %s --check-prefix=ASM --implicit-check-not=s_setreg
; RUN: %llc -mtriple=amdgpu9.42-amd-amdhsa -amdgpu-xnack=1 -filetype=obj %t.bc -o %t.xnack.o
; RUN: %llvm-objdump -d %t.xnack.o | %FileCheck %s --check-prefix=ASM --implicit-check-not=s_setreg
; RUN: %llc -mtriple=amdgpu9.42-amd-amdhsa -amdgpu-xnack=0 -filetype=obj %t.bc -o %t.no-xnack.o
; RUN: %llvm-objdump -d %t.no-xnack.o | %FileCheck %s --check-prefix=ASM --implicit-check-not=s_setreg
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym=REPLICATED=1 -filetype=obj %s -o %t.replicated.o
; RUN: %ld.lld -shared %t.replicated.o -o %t.replicated.hsaco
; RUN: not %transpile_cli %t.replicated.hsaco --target-isa=gfx942 --emit-ir 2>&1 | %FileCheck %s --check-prefix=NATIVE-REFUSE
; NATIVE-REFUSE: non-uniform-scalar-state: s_cbranch_scc1
; RUN: %transpile_cli %t.replicated.hsaco --target-isa=gfx942 --emit-ir --allow-replicated-dispatch > %t.replicated.ll
; RUN: %FileCheck %s --check-prefix=LAUNCH < %t.replicated.ll
; LAUNCH: ; launch: replay kind=replicated-1D-whole-wave max_workgroup_size=256
; RUN: %FileCheck %s --check-prefix=IR --implicit-check-not=llvm.amdgcn.s.setreg < %t.replicated.ll
; RUN: %opt -passes='default<O2>' %t.replicated.ll -o %t.replicated.bc
; RUN: %llc -mtriple=amdgpu9.42-amd-amdhsa -filetype=obj %t.replicated.bc -o %t.replicated.target.o
; RUN: %llvm-objdump -d %t.replicated.target.o | %FileCheck %s --check-prefix=ASM --implicit-check-not=s_setreg
; RUN: %ld.lld -shared %t.replicated.target.o -o %t.replicated.target.hsaco

; RUN: %transpile_cli %t.hsaco --target-isa=gfx1250 --emit-ir | %FileCheck %s --check-prefix=SAME
; RUN: %transpile_cli %t.hsaco --target-isa=gfx1100 --emit-ir | %FileCheck %s --check-prefix=SAME
; SAME-LABEL: define amdgpu_kernel void @replay(
; SAME: call void @llvm.amdgcn.s.setreg(i32 1601, i32 1)

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym=CASE=1 -filetype=obj %s -o %t.1.o
; RUN: %ld.lld -shared %t.1.o -o %t.1.hsaco
; RUN: not %transpile_cli %t.1.hsaco --target-isa=gfx942 --emit-ir 2>&1 | %FileCheck %s --check-prefix=REFUSE
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym=CASE=2 -filetype=obj %s -o %t.2.o
; RUN: %ld.lld -shared %t.2.o -o %t.2.hsaco
; RUN: not %transpile_cli %t.2.hsaco --target-isa=gfx942 --emit-ir 2>&1 | %FileCheck %s --check-prefix=REFUSE
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym=CASE=3 -filetype=obj %s -o %t.3.o
; RUN: %ld.lld -shared %t.3.o -o %t.3.hsaco
; RUN: not %transpile_cli %t.3.hsaco --target-isa=gfx942 --emit-ir 2>&1 | %FileCheck %s --check-prefix=REFUSE
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym=CASE=4 -filetype=obj %s -o %t.4.o
; RUN: %ld.lld -shared %t.4.o -o %t.4.hsaco
; RUN: not %transpile_cli %t.4.hsaco --target-isa=gfx942 --emit-ir 2>&1 | %FileCheck %s --check-prefix=REFUSE
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym=CASE=5 -filetype=obj %s -o %t.5.o
; RUN: %ld.lld -shared %t.5.o -o %t.5.hsaco
; RUN: not %transpile_cli %t.5.hsaco --target-isa=gfx942 --emit-ir 2>&1 | %FileCheck %s --check-prefix=REFUSE
; REFUSE: unsupported-instruction-form: s_setreg
; REFUSE-SAME: only immediate REPLAY_MODE enable at kernel entry is supported

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym=CASE=8 -filetype=obj %s -o %t.wide.o
; RUN: %ld.lld -shared %t.wide.o -o %t.wide.hsaco
; RUN: not %transpile_cli %t.wide.hsaco --target-isa=gfx942 --emit-ir 2>&1 | %FileCheck %s --check-prefix=REFUSE

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym=CASE=9 -filetype=obj %s -o %t.literal.o
; RUN: %ld.lld -shared %t.literal.o -o %t.literal.hsaco
; RUN: not %transpile_cli %t.literal.hsaco --target-isa=gfx942 --emit-ir 2>&1 | %FileCheck %s --check-prefix=REFUSE
; RUN: not %transpile_cli %t.literal.hsaco --target-isa=gfx942 --emit-ir --allow-replicated-dispatch 2>&1 | %FileCheck %s --check-prefix=REFUSE
; RUN: %transpile_cli %t.literal.hsaco --target-isa=gfx1250 --emit-ir | %FileCheck %s --check-prefix=LITERAL
; LITERAL-LABEL: define amdgpu_kernel void @replay(
; LITERAL: call void @llvm.amdgcn.s.setreg(i32 1601, i32 4097)

; A valid backedge to the setup follows a wait that completes memory accesses.
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym=CASE=10 -filetype=obj %s -o %t.loop.o
; RUN: %ld.lld -shared %t.loop.o -o %t.loop.hsaco
; RUN: %transpile_cli %t.loop.hsaco --target-isa=gfx942 --emit-ir | %FileCheck %s --check-prefixes=IR,LOOP --implicit-check-not=llvm.amdgcn.s.setreg

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym=CASE=6 -filetype=obj %s -o %t.read.o
; RUN: %ld.lld -shared %t.read.o -o %t.read.hsaco
; RUN: not %transpile_cli %t.read.hsaco --target-isa=gfx942 --emit-ir 2>&1 | %FileCheck %s --check-prefix=READ
; READ: unsupported-instruction-form: s_getreg_b32
; READ-SAME: cannot reproduce hardware-register read for id 1

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym=CASE=7 -filetype=obj %s -o %t.sched.o
; RUN: %ld.lld -shared %t.sched.o -o %t.sched.hsaco
; RUN: %transpile_cli %t.sched.hsaco --target-isa=gfx942 --emit-ir | %FileCheck %s --check-prefix=IR --implicit-check-not=llvm.amdgcn.s.setreg

.ifndef CASE
.set CASE, 0
.endif
.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.amdhsa_code_object_version 6
.text
.globl replay
.p2align 8
.type replay,@function
; IR-LABEL: define amdgpu_kernel void @replay(
; ASM-LABEL: <replay>:
replay:
.if CASE == 1
  s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 0
.elseif CASE == 2
  s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 3
.elseif CASE == 3
  s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 24, 2), 2
.elseif CASE == 4
  s_setreg_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), s0
.elseif CASE == 5
  s_nop 0
  s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
.elseif CASE == 7
  s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 4, 1), 1
.elseif CASE == 8
  s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 2), 1
.elseif CASE == 9
  s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 0x1001
.else
  s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
.endif
.if CASE == 6
  s_getreg_b32 s2, hwreg(HW_REG_WAVE_MODE, 25, 1)
.endif

; The optional scalar branch requires a separate target wave per source wave.
.ifdef REPLICATED
  v_readfirstlane_b32 s6, v0
  s_and_b32 s6, s6, 32
  s_cmp_eq_u32 s6, 0
  s_cbranch_scc1 .Leven
  v_mov_b32 v3, 7
  s_branch .Lload
.Leven:
.endif
  v_mov_b32 v3, 3
.Lload:
  s_load_b128 s[8:11], s[0:1], 0
  s_wait_kmcnt 0
  v_and_b32 v0, 1023, v0
  v_lshlrev_b32 v1, 2, v0
; IR: load i32, ptr addrspace(1)
; ASM: global_load_dword
  global_load_b32 v2, v1, s[10:11]
  s_wait_loadcnt 0
  v_lshrrev_b32 v2, 23, v2
  v_and_b32 v2, 31, v2
  v_lshlrev_b32 v2, 2, v2
; IR: load i32, ptr addrspace(1)
; ASM: global_load_dword
  global_load_b32 v4, v2, s[10:11]
  s_wait_loadcnt 0
  v_xor_b32 v4, v3, v4
; IR: store i32
; ASM: global_store_dword
  global_store_b32 v1, v4, s[8:9]
  v_mov_b32 v3, 1
; IR: atomicrmw add
; ASM: global_atomic_add
  global_atomic_add_u32 v1, v3, s[8:9] offset:1024
.if CASE == 10
; LOOP: fence syncscope("agent") seq_cst
  s_wait_idle
  s_cmp_eq_u32 s0, 0
  s_cbranch_scc1 replay
.endif
  s_endpgm
.size replay, .-replay

.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel replay
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_kernarg_size 16
  .amdhsa_next_free_vgpr 5
  .amdhsa_next_free_sgpr 12
.end_amdhsa_kernel
.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: replay
    .symbol: replay.kd
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 256
    .sgpr_count: 12
    .vgpr_count: 5
    .wavefront_size: 32
    .args:
      - .offset: 0
        .size: 8
        .value_kind: global_buffer
        .address_space: global
      - .offset: 8
        .size: 8
        .value_kind: global_buffer
        .address_space: global
...
.end_amdgpu_metadata
