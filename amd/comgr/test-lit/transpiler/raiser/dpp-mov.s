; REQUIRES: comgr-has-transpiler, comgr-has-llc
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %transpile_cli %t.hsaco --dump-decoded | %FileCheck %s --check-prefix=DECODE
; RUN: %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir > %t.ll
; RUN: %FileCheck %s --input-file=%t.ll
; RUN: %opt -passes='default<O2>' %t.ll -o %t.bc
; RUN: %llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 -filetype=obj %t.bc -o %t.target.o
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym SHIFT=15 -filetype=obj %s -o %t.15.o
; RUN: %transpile_cli %t.15.o --target-isa=gfx942 --emit-ir > %t.15.ll
; RUN: %opt -passes='default<O2>' %t.15.ll -o %t.15.bc
; RUN: %llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 -filetype=obj %t.15.bc -o %t.15.target.o
; RUN: %transpile_cli %t.hsaco --target-isa=gfx1250 --emit-ir > %t.same.ll
; RUN: %FileCheck %s --check-prefix=SAME --input-file=%t.same.ll
; RUN: %opt -passes='default<O2>' %t.same.ll -o %t.same.bc
; RUN: %llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 -filetype=obj %t.same.bc -o %t.same.o

.ifndef SHIFT
.set SHIFT, 1
.endif
.amdhsa_code_object_version 6
.text
.globl dpp_mov
.p2align 8
.type dpp_mov,@function
; CHECK-LABEL: define amdgpu_kernel void @dpp_mov(
; CHECK: [[ENTRY_ACTIVE:%.+]] = call i1 @llvm.amdgcn.init.whole.wave()
; SAME-LABEL: define amdgpu_kernel void @dpp_mov(
dpp_mov:
  s_load_b64 s[2:3], s[0:1], 0
  s_wait_kmcnt 0
  s_mov_b32 s4, exec_lo
  v_lshlrev_b32 v2, 2, v0
; CHECK: [[DATA:%.+]] = xor i32 305419896, {{.+}}
  v_xor_b32 v3, 0x12345678, v0
; CHECK: [[OLD:%.+]] = xor i32 1985229328, {{.+}}
  v_xor_b32 v4, 0x76543210, v0
; Select different masks in the two source waves.
  v_lshrrev_b32 v5, 5, v0
  v_xor_b32 v5, v0, v5
  v_and_b32 v5, 3, v5
; CHECK: [[BALLOT:%.+]] = call i64 @llvm.amdgcn.ballot.i64
; CHECK: [[MASKBASE:%.+]] = and i32 [[LANE:%.+]], -32
; CHECK: [[MASKSHIFT:%.+]] = zext i32 [[MASKBASE]] to i64
; CHECK: [[SLICE:%.+]] = lshr i64 [[BALLOT]], [[MASKSHIFT]]
; CHECK: [[MASK:%.+]] = trunc i64 [[SLICE]] to i32
; CHECK: [[EXEC:%.+]] = and i32 {{.+}}, [[MASK]]
  v_cmpx_ne_u32 0, v5
; CHECK: [[ROW:%.+]] = and i32 [[LANE]], 15
; CHECK: [[BOUNDS:%.+]] = icmp uge i32 [[ROW]], 1
; CHECK: [[SHIFTED:%.+]] = sub i32 [[LANE]], 1
; CHECK: [[SOURCE:%.+]] = and i32 [[SHIFTED]], 31
; CHECK: [[BASE:%.+]] = and i32 [[LANE]], -32
; CHECK: [[TARGET:%.+]] = or i32 [[BASE]], [[SOURCE]]
; CHECK: [[ADDRESS:%.+]] = shl i32 [[TARGET]], 2
; CHECK: [[VALUE:%.+]] = call i32 @llvm.amdgcn.ds.bpermute(i32 [[ADDRESS]], i32 [[DATA]])
; CHECK: [[LOCAL:%.+]] = and i32 [[LANE]], 31
; CHECK: [[BITS:%.+]] = lshr i32 [[EXEC]], [[LOCAL]]
; CHECK: [[BIT:%.+]] = and i32 [[BITS]], 1
; CHECK: [[ENABLED:%.+]] = icmp ne i32 [[BIT]], 0
; CHECK: [[ACTIVE:%.+]] = select i1 [[ENTRY_ACTIVE]], i1 [[ENABLED]], i1 false
; CHECK: [[ACTIVE32:%.+]] = zext i1 [[ACTIVE]] to i32
; CHECK: [[ACTIVEBASE:%.+]] = and i32 [[LANE]], -32
; CHECK: [[ACTIVETARGET:%.+]] = or i32 [[ACTIVEBASE]], [[SOURCE]]
; CHECK: [[ACTIVEADDR:%.+]] = shl i32 [[ACTIVETARGET]], 2
; CHECK: [[GATHERACTIVE:%.+]] = call i32 @llvm.amdgcn.ds.bpermute(i32 [[ACTIVEADDR]], i32 [[ACTIVE32]])
; CHECK: [[SOURCEACTIVE:%.+]] = icmp ne i32 [[GATHERACTIVE]], 0
; CHECK: [[VALID:%.+]] = and i1 [[BOUNDS]], [[SOURCEACTIVE]]
; CHECK: [[RESULT:%.+]] = select i1 [[VALID]], i32 [[VALUE]], i32 [[OLD]]
; CHECK: br i1 [[ACTIVE]], label %[[WRITE:.+]], label %[[SKIP:.+]]
; CHECK: [[SKIP]]:
; CHECK: [[PRESERVED:%.+]] = phi i32 [ [[RESULT]], %[[WRITE]] ], [ [[OLD]], {{.+}} ]
; CHECK: store i32 [[PRESERVED]],
; SAME: call i32 @llvm.amdgcn.strict.wwm.i32
; DECODE: V_MOV_B32{{.+}}v_mov_b32_dpp v4, v3 row_shr:1 row_mask:0xf bank_mask:0xf
  v_mov_b32 v4, v3 row_shr:SHIFT row_mask:0xf bank_mask:0xf
  s_mov_b32 exec_lo, s4
  global_store_b32 v2, v4, s[2:3]
  v_xor_b32 v4, 0x76543210, v0
; DECODE: V_MOV_B32{{.+}}v_mov_b32_dpp v4, v3 row_shr:2 row_mask:0xf bank_mask:0xf
; CHECK: icmp uge i32 {{.+}}, 2
; CHECK: call i32 @llvm.amdgcn.ds.bpermute
; CHECK: select i1
; CHECK: store i32
  v_mov_b32 v4, v3 row_shr:2 row_mask:0xf bank_mask:0xf
  global_store_b32 v2, v4, s[2:3] offset:256
  v_xor_b32 v4, 0x76543210, v0
; DECODE: V_MOV_B32{{.+}}v_mov_b32_dpp v4, v3 row_shr:4 row_mask:0xf bank_mask:0xf
; CHECK: icmp uge i32 {{.+}}, 4
; CHECK: call i32 @llvm.amdgcn.ds.bpermute
; CHECK: select i1
; CHECK: store i32
  v_mov_b32 v4, v3 row_shr:4 row_mask:0xf bank_mask:0xf
  global_store_b32 v2, v4, s[2:3] offset:512
  v_xor_b32 v4, 0x76543210, v0
; DECODE: V_MOV_B32{{.+}}v_mov_b32_dpp v4, v3 row_shr:8 row_mask:0xf bank_mask:0xf
; CHECK: icmp uge i32 {{.+}}, 8
; CHECK: call i32 @llvm.amdgcn.ds.bpermute
; CHECK: select i1
; CHECK: store i32
  v_mov_b32 v4, v3 row_shr:8 row_mask:0xf bank_mask:0xf
  global_store_b32 v2, v4, s[2:3] offset:768
  s_endpgm

.rodata
.p2align 6
.amdhsa_kernel dpp_mov
  .amdhsa_kernarg_size 8
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr 6
  .amdhsa_next_free_sgpr 5
.end_amdhsa_kernel
.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: dpp_mov
    .symbol: dpp_mov.kd
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 64
    .wavefront_size: 32
    .sgpr_count: 5
    .vgpr_count: 6
...
.end_amdgpu_metadata
