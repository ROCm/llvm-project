; REQUIRES: comgr-has-transpiler, comgr-has-llc
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %python -c "import struct; open('%t.args', 'wb').write(bytes(16)+struct.pack('<HHHHII',16,8,1,0,128,0))"
; RUN: %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir=coordinates --allow-replicated-dispatch --specialize-workgroup=16,8,1 --launch-workgroup=16,8,1 --launch-grid=32,24,2 --launch-kernarg=%t.args --launch-dynamic-lds=128 > %t.ll
; RUN: %FileCheck %s --check-prefix=FLAT < %t.ll
; RUN: %opt -passes='default<O2>' %t.ll -o %t.bc
; RUN: %llc -mtriple=amdgpu9.42-amd-amdhsa -filetype=obj %t.bc -o %t.target.o
; FLAT: ; launch: coordinates kind=replicated-flattened max_workgroup_size=512 required_workgroup_size=16,8,1 grid=512,3,2 workgroup=256,1,1
; RUN: %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir=coordinates --allow-replicated-dispatch --launch-workgroup=16,8,1 --launch-grid=32,24,2 --launch-kernarg=%t.args --launch-dynamic-lds=128 | %FileCheck %s --check-prefix=DYNAMIC
; DYNAMIC: ; launch: coordinates kind=replicated-flattened max_workgroup_size=512 grid=512,3,2 workgroup=256,1,1
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir=coordinates --allow-replicated-dispatch --launch-workgroup=32,4,1 --launch-grid=64,12,2 --launch-kernarg=%t.args --launch-dynamic-lds=128 2>&1 | %FileCheck %s --check-prefix=ARGUMENT
; ARGUMENT: kernarg workgroup size does not match the logical launch
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir=coordinates --allow-replicated-dispatch --launch-workgroup=16,8,1 --launch-grid=32,24,2 --launch-kernarg=%t.args --launch-dynamic-lds=256 2>&1 | %FileCheck %s --check-prefix=LDS
; LDS: kernarg dynamic LDS size does not match the allocation
; RUN: %python -c "import struct; open('%t.partial.args', 'wb').write(bytes(16)+struct.pack('<HHHHII',1,1,1,0,128,0))"
; RUN: %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir=coordinates --allow-replicated-dispatch --launch-workgroup=1,1,1 --launch-grid=2,3,2 --launch-kernarg=%t.partial.args --launch-dynamic-lds=128 | %FileCheck %s --check-prefix=PARTIAL
; PARTIAL: ; launch: coordinates kind=replicated-flattened max_workgroup_size=512 grid=128,3,2 workgroup=64,1,1
; RUN: %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir=coordinates --allow-replicated-dispatch --specialize-workgroup=1024,1,1 | %FileCheck %s --check-prefix=NATIVE
; NATIVE: ; launch: coordinates kind=unchanged max_workgroup_size=1024 required_workgroup_size=1024,1,1
; NATIVE-LABEL: define amdgpu_kernel void @coordinates(
; NATIVE: @llvm.amdgcn.init.whole.wave
; NATIVE: @llvm.amdgcn.ds.bpermute
; RUN: %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir=scalar_memory --allow-replicated-dispatch --specialize-workgroup=1024,1,1 > %t.scalar.ll
; RUN: %FileCheck %s --check-prefix=SCALAR < %t.scalar.ll
; RUN: %opt -passes='default<O2>' %t.scalar.ll -o %t.scalar.bc
; RUN: %llc -mtriple=amdgpu9.42-amd-amdhsa -filetype=obj %t.scalar.bc -o %t.scalar.o
; SCALAR: ; launch: scalar_memory kind=unchanged
; SCALAR-LABEL: define amdgpu_kernel void @scalar_memory(
; SCALAR: @llvm.amdgcn.ds.bpermute
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir=coordinates --allow-replicated-dispatch --specialize-workgroup=1024,1,1 --launch-workgroup=1024,1,1 --launch-grid=1025,1,1 2>&1 | %FileCheck %s --check-prefix=INCOMPLETE
; INCOMPLETE: required workgroup dimensions need complete workgroups
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym=RESTORE_COMPARE=1 -filetype=obj %s -o %t.compare.o
; RUN: %ld.lld -shared %t.compare.o -o %t.compare.hsaco
; RUN: %transpile_cli %t.compare.hsaco --target-isa=gfx942 --emit-ir=coordinates --allow-replicated-dispatch --specialize-workgroup=1,1,1 > %t.compare.ll
; RUN: %FileCheck %s --check-prefix=COMPARE < %t.compare.ll
; RUN: %opt -passes='default<O2>' %t.compare.ll -o %t.compare.bc
; RUN: %llc -mtriple=amdgpu9.42-amd-amdhsa -filetype=obj %t.compare.bc -o %t.compare.target.o
; COMPARE: ; launch: coordinates kind=replicated-1D
; COMPARE-LABEL: define amdgpu_kernel void @coordinates(
; COMPARE-NOT: @llvm.amdgcn.strict.wwm
; COMPARE: store i32
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym=RESTORE_COMPARE=1 -defsym=CLOBBER_COMPARE=1 -filetype=obj %s -o %t.unbounded.o
; RUN: %ld.lld -shared %t.unbounded.o -o %t.unbounded.hsaco
; RUN: not %transpile_cli %t.unbounded.hsaco --target-isa=gfx942 --emit-ir=coordinates --allow-replicated-dispatch --specialize-workgroup=1,1,1 2>&1 | %FileCheck %s --check-prefix=UNBOUNDED
; UNBOUNDED: replicated dispatch requires whole source waves
.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.amdhsa_code_object_version 6
.text
.globl coordinates
.p2align 8
.type coordinates,@function
coordinates:
  s_load_b128 s[4:7], s[0:1], 0
  s_load_b128 s[12:15], s[0:1], 16
  s_wait_kmcnt 0
  s_bfe_u32 s16, s12, 0x100000
  s_lshr_b32 s17, s12, 16
  s_mul_i32 s18, s16, s17
  v_and_b32 v1, 1023, v0
  v_bfe_u32 v2, v0, 10, 10
  v_lshrrev_b32 v3, 20, v0
  v_mul_lo_u32 v2, s16, v2
  v_mul_lo_u32 v3, s18, v3
  v_add_nc_u32 v1, v2, v1
  v_add_nc_u32 v1, v3, v1
  s_bfe_u32 s8, ttmp8, 0x50019
.ifdef RESTORE_COMPARE
  v_cmp_eq_u32 vcc_lo, 0, v1
.ifdef CLOBBER_COMPARE
  s_mov_b32 vcc_lo, 2
.endif
  s_mov_b32 s23, vcc_lo
  s_and_not1_b32 exec_lo, exec_lo, s23
  s_or_b32 exec_lo, exec_lo, s23
.endif
  s_and_b32 s9, s8, 1
  s_cmp_eq_u32 s9, 0
  s_cbranch_scc1 .Leven
  s_mov_b32 s10, 200
  s_branch .Ljoin
.Leven:
  s_mov_b32 s10, 100
.Ljoin:
  s_add_u32 s10, s10, s8
  s_bfe_u32 s20, ttmp7, 0x100000
  s_lshr_b32 s21, ttmp7, 16
  s_mul_i32 s20, s20, 2
  s_mul_i32 s21, s21, 6
  s_add_u32 s20, s20, ttmp9
  s_add_u32 s20, s20, s21
  s_lshl_b32 s22, s20, 12
  v_lshlrev_b32 v2, 2, v1
  v_add_nc_u32 v2, s22, v2
  global_store_b32 v2, v0, s[4:5]
  v_mov_b32 v3, s10
  global_store_b32 v2, v3, s[4:5] offset:49152
  v_mov_b32 v3, s14
  global_store_b32 v2, v3, s[4:5] offset:98304
  v_mov_b32 v3, 1
  v_mov_b32 v4, s20
  v_lshlrev_b32 v4, 2, v4
  global_atomic_add_u32 v5, v4, v3, s[6:7] th:TH_ATOMIC_RETURN
  global_store_b32 v2, v5, s[4:5] offset:147456
  v_readfirstlane_b32 s11, v1
  v_mov_b32 v3, s11
  global_store_b32 v2, v3, s[4:5] offset:196608
  s_endpgm
.globl scalar_memory
.p2align 8
.type scalar_memory,@function
scalar_memory:
  s_load_b128 s[4:7], s[0:1], 0
  s_bfe_u32 s8, ttmp8, 0x50019
  s_lshl_b32 s9, s8, 2
  s_wait_kmcnt 0
  s_load_b32 s10, s[6:7], s9
  s_wait_kmcnt 0
  v_lshlrev_b32 v1, 2, v0
  v_mov_b32 v2, s10
  global_store_b32 v1, v2, s[4:5]
  s_endpgm

.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel coordinates
  .amdhsa_kernarg_size 32
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_system_vgpr_workitem_id 2
  .amdhsa_next_free_vgpr 8
  .amdhsa_next_free_sgpr 24
.end_amdhsa_kernel
.amdhsa_kernel scalar_memory
  .amdhsa_kernarg_size 16
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr 4
  .amdhsa_next_free_sgpr 12
.end_amdhsa_kernel
.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: coordinates
    .symbol: coordinates.kd
    .group_segment_fixed_size: 0
    .kernarg_segment_size: 32
    .kernarg_segment_align: 8
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 1024
    .sgpr_count: 24
    .vgpr_count: 8
    .wavefront_size: 32
    .args:
      - { .offset: 16, .size: 2, .value_kind: hidden_group_size_x }
      - { .offset: 18, .size: 2, .value_kind: hidden_group_size_y }
      - { .offset: 20, .size: 2, .value_kind: hidden_group_size_z }
      - { .offset: 24, .size: 4, .value_kind: hidden_dynamic_lds_size }
  - .name: scalar_memory
    .symbol: scalar_memory.kd
    .group_segment_fixed_size: 0
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 1024
    .sgpr_count: 12
    .vgpr_count: 4
    .wavefront_size: 32
...
.end_amdgpu_metadata
