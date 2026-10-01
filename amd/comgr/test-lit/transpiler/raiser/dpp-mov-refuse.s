; REQUIRES: comgr-has-transpiler
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym CASE=0 -filetype=obj %s -o %t.0.o
; RUN: not %transpile_cli %t.0.o --target-isa=gfx942 --emit-ir 2>&1 | %FileCheck %s --check-prefix=CASE0
; CASE0: expected a DPP16 row_shr move

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym CASE=1 -filetype=obj %s -o %t.1.o
; RUN: not %transpile_cli %t.1.o --target-isa=gfx942 --emit-ir 2>&1 | %FileCheck %s --check-prefix=CASE1
; CASE1: expected a DPP16 row_shr move

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym CASE=2 -filetype=obj %s -o %t.2.o
; RUN: not %transpile_cli %t.2.o --target-isa=gfx942 --emit-ir 2>&1 | %FileCheck %s --check-prefix=CASE2
; CASE2: expected a DPP16 row_shr move

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym CASE=3 -filetype=obj %s -o %t.3.o
; RUN: not %transpile_cli %t.3.o --target-isa=gfx942 --emit-ir 2>&1 | %FileCheck %s --check-prefix=CASE3
; CASE3: unsupported-instruction-form: v_mov_b32 [DPP]

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym CASE=4 -filetype=obj %s -o %t.4.o
; RUN: not %transpile_cli %t.4.o --target-isa=gfx942 --emit-ir 2>&1 | %FileCheck %s --check-prefix=CASE4
; CASE4: DPP move requires full row and bank masks with bounds control disabled

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym CASE=5 -filetype=obj %s -o %t.5.o
; RUN: not %transpile_cli %t.5.o --target-isa=gfx942 --emit-ir 2>&1 | %FileCheck %s --check-prefix=CASE5
; CASE5: DPP move requires full row and bank masks with bounds control disabled

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym CASE=6 -filetype=obj %s -o %t.6.o
; RUN: not %transpile_cli %t.6.o --target-isa=gfx942 --emit-ir 2>&1 | %FileCheck %s --check-prefix=CASE6
; CASE6: DPP move requires full row and bank masks with bounds control disabled

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym CASE=7 -filetype=obj %s -o %t.7.o
; RUN: not %transpile_cli %t.7.o --target-isa=gfx942 --emit-ir 2>&1 | %FileCheck %s --check-prefix=CASE7
; CASE7: DPP move does not support fi:1

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym CASE=8 -filetype=obj %s -o %t.8.o
; RUN: not %transpile_cli %t.8.o --target-isa=gfx942 --emit-ir 2>&1 | %FileCheck %s --check-prefix=CASE8
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym CASE=9 -filetype=obj %s -o %t.9.o
; RUN: not %transpile_cli %t.9.o --target-isa=gfx942 --emit-ir 2>&1 | %FileCheck %s --check-prefix=CASE9
; CASE8: unsupported-instruction-form: v_mov_b32 [DPP]
; CASE9: unsupported-instruction-form: v_add_f32 [DPP]

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym CASE=10 -filetype=obj %s -o %t.10.o
; RUN: not %transpile_cli %t.10.o --target-isa=gfx942 --emit-ir 2>&1 | %FileCheck %s --check-prefix=CASE10
; CASE10: unproven-exec-containment: s_mov_b32
; CASE10-SAME: WaveNative cannot prove that EXEC only enables lanes active at kernel entry
; RUN: %transpile_cli %t.10.o --target-isa=gfx942 --emit-ir --allow-replicated-dispatch --launch-grid=64,1,1 --launch-workgroup=64,1,1 | %FileCheck %s --check-prefix=REPLICATED
; REPLICATED: ; launch: unsupported kind=replicated-1D-whole-wave max_workgroup_size=64 grid=128,1,1 workgroup=128,1,1

.amdhsa_code_object_version 6
.text
.globl unsupported
.p2align 8
.type unsupported,@function
unsupported:
.if CASE == 0
  v_mov_b32 v1, v0 row_shl:1
.endif
.if CASE == 1
  v_mov_b32 v1, v0 row_ror:1
.endif
.if CASE == 2
  v_mov_b32 v1, v0 quad_perm:[0,1,2,3]
.endif
.if CASE == 3
  v_mov_b32 v1, v0 dpp8:[0,1,2,3,4,5,6,7]
.endif
.if CASE == 4
  v_mov_b32 v1, v0 row_shr:1 row_mask:0x7
.endif
.if CASE == 5
  v_mov_b32 v1, v0 row_shr:1 bank_mask:0x7
.endif
.if CASE == 6
  v_mov_b32 v1, v0 row_shr:1 bound_ctrl:1
.endif
.if CASE == 7
  v_mov_b32 v1, v0 row_shr:1 fi:1
.endif
.if CASE == 8
  v_mov_b32_e64_dpp v1, v0 row_shr:1
.endif
.if CASE == 9
  v_add_f32 v1, v0, v2 row_shr:1
.endif
.if CASE == 10
  s_mov_b32 exec_lo, -1
  v_mov_b32 v1, v0 row_shr:1
.endif
  s_endpgm
.rodata
.p2align 6
.amdhsa_kernel unsupported
  .amdhsa_next_free_vgpr 3
  .amdhsa_next_free_sgpr 0
.end_amdhsa_kernel
.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: unsupported
    .symbol: unsupported.kd
    .kernarg_segment_size: 0
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 64
    .wavefront_size: 32
    .sgpr_count: 0
    .vgpr_count: 3
...
.end_amdgpu_metadata
