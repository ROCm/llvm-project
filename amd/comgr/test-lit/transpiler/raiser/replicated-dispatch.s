; REQUIRES: comgr-has-transpiler, comgr-has-llc
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir --allow-replicated-dispatch --launch-grid=192,1,1 --launch-workgroup=96,1,1 > %t.ll
; RUN: %FileCheck %s --check-prefix=IR < %t.ll
; RUN: %FileCheck %s --check-prefix=LAUNCH < %t.ll
; RUN: %opt -passes='default<O2>' %t.ll -o %t.bc
; RUN: %llc -mtriple=amdgpu9.42-amd-amdhsa -filetype=obj %t.bc -o %t.gfx942.o
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir=ids 2>&1 | %FileCheck %s --check-prefix=NO-CONTRACT
; NO-CONTRACT: non-uniform-scalar-state:
; NO-CONTRACT-SAME: in kernel 'ids'
; RUN: %transpile_cli %t.hsaco --target-isa=gfx1250 --emit-ir=ordinary --allow-replicated-dispatch > %t.same.ll
; RUN: %FileCheck %s --check-prefix=SAME < %t.same.ll
; SAME: ; launch: ordinary kind=unchanged max_workgroup_size=1024

; LAUNCH: ; launch: ids kind=replicated-1D-whole-wave max_workgroup_size=512 grid=384,1,1 workgroup=192,1,1
; LAUNCH-NEXT: ; launch: atomic_counts kind=replicated-1D-whole-wave max_workgroup_size=512 grid=384,1,1 workgroup=192,1,1
; LAUNCH: ; launch: geometry kind=replicated-1D-whole-wave max_workgroup_size=512 required_workgroup_size=96,1,1 grid=384,1,1 workgroup=192,1,1
; LAUNCH-NEXT: ; launch: ordinary kind=unchanged max_workgroup_size=1024 grid=192,1,1 workgroup=96,1,1

; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir=ids --allow-replicated-dispatch --launch-grid=32,1,1 --launch-workgroup=0,1,1 2>&1 | %FileCheck %s --check-prefix=ZERO
; ZERO: unsupported-launch in kernel 'ids'
; ZERO-SAME: launch dimensions must be nonzero
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir=ids --allow-replicated-dispatch --launch-grid=96,1,1 --launch-workgroup=48,1,1 2>&1 | %FileCheck %s --check-prefix=PARTIAL
; PARTIAL: unsupported-launch in kernel 'ids' :: replicated dispatch requires whole source waves
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir=ids --allow-replicated-dispatch --launch-grid=96,1,1 --launch-workgroup=64,1,1 2>&1 | %FileCheck %s --check-prefix=EDGE
; EDGE: unsupported-launch in kernel 'ids' :: replicated dispatch requires complete workgroups
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir=ids --allow-replicated-dispatch --launch-grid=32,2,1 --launch-workgroup=32,2,1 2>&1 | %FileCheck %s --check-prefix=MULTI
; MULTI: unsupported-launch in kernel 'ids' :: replicated dispatch requires a one-dimensional workgroup
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir=ids --allow-replicated-dispatch --launch-grid=544,1,1 --launch-workgroup=544,1,1 2>&1 | %FileCheck %s --check-prefix=LIMIT
; LIMIT: unsupported-launch in kernel 'ids' :: workgroup exceeds the kernel's supported launch size
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir=ids --allow-replicated-dispatch --launch-grid=2147483648,1,1 --launch-workgroup=32,1,1 2>&1 | %FileCheck %s --check-prefix=OVERFLOW
; OVERFLOW: unsupported-launch in kernel 'ids' :: replicated grid size overflows the dispatch packet
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir=ids --allow-replicated-dispatch --launch-grid=32,1 --launch-workgroup=32,1,1 2>&1 | %FileCheck %s --check-prefix=DIMENSIONS
; DIMENSIONS: launch dimensions require --allow-replicated-dispatch and three grid and workgroup dimensions
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir=ids --launch-grid=32,1,1 --launch-workgroup=32,1,1 2>&1 | %FileCheck %s --check-prefix=DIMENSIONS
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir=ids,ids --allow-replicated-dispatch 2>&1 | %FileCheck %s --check-prefix=DUPLICATE
; DUPLICATE: BadInput :: kernel names must be nonempty and unique
; RUN: not %transpile_cli %t.hsaco --dump-meta --allow-replicated-dispatch 2>&1 | %FileCheck %s --check-prefix=MODE
; MODE: --allow-replicated-dispatch requires --emit-ir
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir=ordinary,geometry --allow-replicated-dispatch --launch-grid=64,1,1 --launch-workgroup=64,1,1 > %t.partial.ll 2> %t.err
; RUN: %FileCheck %s --check-prefix=REQUIRED < %t.err
; RUN: test ! -s %t.partial.ll

; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir=geometry --allow-replicated-dispatch --launch-grid=64,1,1 --launch-workgroup=64,1,1 2>&1 | %FileCheck %s --check-prefix=REQUIRED
; REQUIRED: unsupported-launch in kernel 'geometry'
; REQUIRED-SAME: workgroup does not match the source kernel's required dimensions

; RUN: sed '/^    .reqd_workgroup_size:/s/96, 1, 1/0, 1, 1/' %s | %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj -o %t.required.o
; RUN: %ld.lld -shared %t.required.o -o %t.required.hsaco
; RUN: not %transpile_cli %t.required.hsaco --target-isa=gfx942 --emit-ir=geometry --allow-replicated-dispatch 2>&1 | %FileCheck %s --check-prefix=ZERO-REQUIRED
; ZERO-REQUIRED: invalid .reqd_workgroup_size
; RUN: sed '/^    .reqd_workgroup_size:/s/96, 1, 1/1024, 1024, 1024/' %s | %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj -o %t.required.o
; RUN: %ld.lld -shared %t.required.o -o %t.required.hsaco
; RUN: not %transpile_cli %t.required.hsaco --target-isa=gfx942 --emit-ir=geometry --allow-replicated-dispatch 2>&1 | %FileCheck %s --check-prefix=LARGE-REQUIRED
; LARGE-REQUIRED: invalid .reqd_workgroup_size
; RUN: sed '/^    .reqd_workgroup_size:/s/96, 1, 1/32, 2, 1/' %s | %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj -o %t.required.o
; RUN: %ld.lld -shared %t.required.o -o %t.required.hsaco
; RUN: %transpile_cli %t.required.hsaco --target-isa=gfx942 --emit-ir=geometry --allow-replicated-dispatch | %FileCheck %s --check-prefix=MULTI-REQUIRED
; MULTI-REQUIRED: ; launch: geometry kind=replicated-flattened-whole-wave
; RUN: sed '/^    .reqd_workgroup_size:/s/96, 1, 1/48, 1, 1/' %s | %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj -o %t.required.o
; RUN: %ld.lld -shared %t.required.o -o %t.required.hsaco
; RUN: not %transpile_cli %t.required.hsaco --target-isa=gfx942 --emit-ir=geometry --allow-replicated-dispatch 2>&1 | %FileCheck %s --check-prefix=PARTIAL-REQUIRED
; PARTIAL-REQUIRED: replicated dispatch requires whole source waves
; RUN: sed '/^    .reqd_workgroup_size:/s/96, 1, 1/544, 1, 1/' %s | %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj -o %t.required.o
; RUN: %ld.lld -shared %t.required.o -o %t.required.hsaco
; RUN: not %transpile_cli %t.required.hsaco --target-isa=gfx942 --emit-ir=geometry --allow-replicated-dispatch 2>&1 | %FileCheck %s --check-prefix=LIMIT-REQUIRED
; LIMIT-REQUIRED: unproven-exec-containment:
; RUN: sed '/^    .reqd_workgroup_size:/s/96, 1, 1/0, 0, 0/' %s | %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj -o %t.required.o
; RUN: %ld.lld -shared %t.required.o -o %t.required.hsaco
; RUN: %transpile_cli %t.required.hsaco --target-isa=gfx942 --emit-ir=geometry --allow-replicated-dispatch > %t.required.ll
; RUN: %FileCheck %s --check-prefix=UNSPECIFIED < %t.required.ll
; UNSPECIFIED: ; launch: geometry kind=replicated-1D-whole-wave max_workgroup_size=512{{$}}
; UNSPECIFIED-NOT: required_workgroup_size
; RUN: %transpile_cli %t.hsaco --target-isa=gfx1250 --emit-ir=geometry > %t.required.ll
; RUN: %FileCheck %s --check-prefix=EXACT-SOURCE < %t.required.ll
; EXACT-SOURCE: "amdgpu-flat-work-group-size"="96,96"
; EXACT-SOURCE: !{i32 96, i32 1, i32 1}

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.amdhsa_code_object_version 6
.text

.globl ids
.p2align 8
.type ids,@function
; IR-LABEL: define amdgpu_kernel void @ids(
ids:
; IR-NOT: call i1 @llvm.amdgcn.init.whole.wave
; IR: [[LANE:%.+]] = call i32 @llvm.amdgcn.mbcnt.hi
; IR: [[X:%.+]] = call i32 @llvm.amdgcn.workitem.id.x()
; IR: [[ALIGNED:%.+]] = and i32 [[X]], -64
; IR-NEXT: [[BASE:%.+]] = lshr i32 [[ALIGNED]], 1
; IR-NEXT: [[SOURCE_LANE:%.+]] = and i32 [[X]], 31
; IR-NEXT: [[ID:%.+]] = or i32 [[BASE]], [[SOURCE_LANE]]
  s_load_b64 s[2:3], s[0:1], 0
  s_wait_kmcnt 0
; IR: [[WAVE:%.+]] = udiv i32 [[X]], 64
; IR-NEXT: [[WAVE_ID:%.+]] = call i32 @llvm.amdgcn.readfirstlane.i32(i32 [[WAVE]])
  s_bfe_u32 s4, ttmp8, 0x50019
  s_and_b32 s5, s4, 1
  s_cmp_eq_u32 s5, 0
  s_cbranch_scc1 .Leven
  s_mov_b32 s6, 200
  s_branch .Ljoin
.Leven:
  s_mov_b32 s6, 100
.Ljoin:
  s_mov_b32 s7, s4
.Lloop:
  s_cmp_eq_u32 s7, 0
  s_cbranch_scc1 .Lloop_exit
  s_add_u32 s6, s6, 1
  s_sub_u32 s7, s7, 1
  s_branch .Lloop
.Lloop_exit:
  s_lshl_b32 s11, ttmp9, 15
  v_lshlrev_b32 v3, 2, v0
  v_add_nc_u32 v3, s11, v3
  v_add_nc_u32 v1, s6, v0
; IR: [[PRIMARY:%.+]] = icmp ult i32 [[LANE]], 32
; IR-NEXT: br i1 [[PRIMARY]], label %[[STORE:.+]], label %[[SKIP:.+]]
; IR: [[STORE]]:
; IR-NEXT: store i32 {{.+}}, ptr addrspace(1)
; IR-NEXT: br label %[[SKIP]]
  global_store_b32 v3, v1, s[2:3]
  v_mov_b32 v1, s4
  global_store_b32 v3, v1, s[2:3] offset:4096
  v_and_b32 v2, 1, v0
  v_cmp_eq_u32_e64 s7, s5, v2
  s_and_saveexec_b32 s8, s7
  v_mov_b32 v1, s7
  global_store_b32 v3, v1, s[2:3] offset:8192
; IR: call i32 @llvm.amdgcn.readlane.i32(i32 [[ID]], i32 {{.+}})
  v_readfirstlane_b32 s9, v0
  s_mov_b32 exec_lo, s8
  v_mov_b32 v1, s9
  global_store_b32 v3, v1, s[2:3] offset:12288
; IR: call i32 @llvm.amdgcn.readlane.i32(i32 [[ID]], i32 7)
  v_readlane_b32 s10, v0, 7
  v_mov_b32 v1, v0
  s_mov_b32 exec_lo, 0
  v_writelane_b32 v1, s10, 3
  global_store_b32 v3, v1, s[2:3] offset:20480
  s_mov_b32 exec_lo, s8
  global_store_b32 v3, v1, s[2:3] offset:16384
  v_cmpx_lt_u32 16, v0
  v_mov_b32 v1, 7
  global_store_b32 v3, v1, s[2:3] offset:24576
  s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
  s_endpgm

.globl atomic_counts
.p2align 8
.type atomic_counts,@function
; IR-LABEL: define amdgpu_kernel void @atomic_counts(
atomic_counts:
; IR: [[ATOMIC_LANE:%.+]] = call i32 @llvm.amdgcn.mbcnt.hi
  s_load_b64 s[2:3], s[0:1], 0
  s_wait_kmcnt 0
  s_bfe_u32 s4, ttmp8, 0x50019
  v_lshlrev_b32 v3, 2, v0
  s_lshl_b32 s5, s4, 2
  v_mov_b32 v1, s5
  v_mov_b32 v2, 0
  s_mov_b32 exec_lo, 1
; IR: store i32 {{.+}}, ptr addrspace(3)
  ds_store_b32 v1, v2
  s_wait_dscnt 0
  s_and_b32 s6, s4, 1
  s_cmp_eq_u32 s6, 0
  s_cbranch_scc1 .Latomic_even
  s_mov_b32 exec_lo, 0xaaaaaaaa
  s_branch .Latomic_join
.Latomic_even:
  s_mov_b32 exec_lo, 0x55555555
.Latomic_join:
  v_mov_b32 v2, 1
; IR: [[ADD_PRIMARY:%.+]] = icmp ult i32 [[ATOMIC_LANE]], 32
; IR-NEXT: br i1 [[ADD_PRIMARY]], label %[[ADD:.+]], label %{{.+}}
; IR: [[ADD]]:
; IR: atomicrmw add ptr addrspace(3) {{.+}} seq_cst, align 4
  ds_add_u32 v1, v2
  s_wait_dscnt 0
; IR: [[RETURN_PRIMARY:%.+]] = icmp ult i32 [[ATOMIC_LANE]], 32
; IR-NEXT: br i1 [[RETURN_PRIMARY]], label %[[RETURN:.+]], label %[[JOIN:.+]]
; IR: [[DESTINATION:%.+]] = phi i32 [ [[REPLICATED:%.+]], %[[JOIN]] ], [ undef, %{{.+}} ]
; IR: [[RETURN]]:
; IR: [[OLD:%.+]] = atomicrmw add ptr addrspace(3) {{.+}} seq_cst, align 4
; IR-NEXT: br label %[[JOIN]]
; IR: [[JOIN]]:
; IR-NEXT: [[MERGED:%.+]] = phi i32 [ 0, %{{.+}} ], [ [[OLD]], %[[RETURN]] ]
; IR-NEXT: [[SELECT_LANE:%.+]] = and i32 [[ATOMIC_LANE]], 31
; IR-NEXT: [[SELECTOR:%.+]] = shl i32 [[SELECT_LANE]], 2
; IR-NEXT: [[REPLICATED]] = call i32 @llvm.amdgcn.ds.bpermute(i32 [[SELECTOR]], i32 [[MERGED]])
  ds_add_rtn_u32 v4, v1, v2
  s_wait_dscnt 0
  global_store_b32 v3, v4, s[2:3]
; IR: [[PREDICATE:%.+]] = icmp ugt i32 24, [[DESTINATION]]
; IR-NEXT: [[ACTIVE:%.+]] = select i1 {{.+}}, i1 [[PREDICATE]], i1 false
; IR-NEXT: [[BALLOT:%.+]] = call i64 @llvm.amdgcn.ballot.i64(i1 [[ACTIVE]])
; IR-NEXT: trunc i64 [[BALLOT]] to i32
  v_cmp_gt_u32_e64 s7, 24, v4
  s_bcnt1_i32_b32 s8, s7
  s_mov_b32 exec_lo, -1
  v_mov_b32 v5, s8
  global_store_b32 v3, v5, s[2:3] offset:4096
  ds_load_b32 v6, v1
  s_wait_dscnt 0
  global_store_b32 v3, v6, s[2:3] offset:8192
  s_mov_b32 exec_lo, 0
  ds_add_u32 v1, v2
  ds_add_rtn_u32 v4, v1, v2
  s_mov_b32 exec_lo, -1
  ds_load_b32 v6, v1
  s_wait_dscnt 0
  global_store_b32 v3, v6, s[2:3] offset:12288
  s_endpgm

.globl loads
.p2align 8
.type loads,@function
; IR-LABEL: define amdgpu_kernel void @loads(
loads:
; IR: load <4 x i32>, ptr addrspace(1)
; IR-COUNT-4: call i32 @llvm.amdgcn.readlane.i32
  s_mov_b32 exec_lo, -1
  s_load_b128 s[4:7], s[0:1], 0
  s_wait_kmcnt 0
  v_lshlrev_b32 v1, 4, v0
; IR: [[LOAD_PRIMARY:%.+]] = icmp ult i32 {{.+}}, 32
; IR-NEXT: br i1 [[LOAD_PRIMARY]], label %[[LOAD:.+]], label %[[LOAD_JOIN:.+]]
; IR: [[LOAD]]:
; IR-NEXT: [[LOADED:%.+]] = load <4 x i32>, ptr addrspace(1)
; IR-NEXT: br label %[[LOAD_JOIN]]
; IR: [[LOAD_JOIN]]:
; IR-NEXT: [[WORDS:%.+]] = phi <4 x i32> [ zeroinitializer, %{{.+}} ], [ [[LOADED]], %[[LOAD]] ]
; IR: extractelement <4 x i32> [[WORDS]], i64 0
; IR: call i32 @llvm.amdgcn.ds.bpermute
; IR: extractelement <4 x i32> [[WORDS]], i64 3
; IR: call i32 @llvm.amdgcn.ds.bpermute
  global_load_b128 v[4:7], v1, s[6:7]
  s_wait_loadcnt 0
  global_store_b128 v1, v[4:7], s[4:5]
  ds_store_b128 v1, v[4:7]
  s_wait_dscnt 0
; IR: load i64, ptr addrspace(3)
; IR: call i32 @llvm.amdgcn.ds.bpermute
; IR: call i32 @llvm.amdgcn.ds.bpermute
  ds_load_b64 v[8:9], v1
; IR: load i16, ptr addrspace(3)
; IR: call i32 @llvm.amdgcn.ds.bpermute
  ds_load_u16 v10, v1 offset:8
; IR: load i8, ptr addrspace(3)
; IR: call i32 @llvm.amdgcn.ds.bpermute
  ds_load_u8 v11, v1 offset:12
  s_wait_dscnt 0
  global_store_b128 v1, v[8:11], s[4:5] offset:8192
  global_load_b96 v[12:14], v1, s[6:7]
  s_wait_loadcnt 0
  global_store_b96 v1, v[12:14], s[4:5] offset:16384
  s_endpgm

.globl buffer
.p2align 8
.type buffer,@function
; IR-LABEL: define amdgpu_kernel void @buffer(
buffer:
  s_mov_b32 exec_lo, -1
  s_load_b64 s[2:3], s[0:1], 0
  s_load_b64 s[8:9], s[0:1], 8
  s_wait_kmcnt 0
  s_and_b32 s9, s9, 0x1ffffff
  s_or_b32 s9, s9, 0x02000000
  s_mov_b32 s10, 128
  s_mov_b32 s11, 0
  s_mov_b32 s12, 0
  v_lshlrev_b32 v1, 2, v0
; IR: [[BUFFER_READ:%.+]] = load i32, ptr addrspace(1)
; IR: [[BUFFER_RESULT:%.+]] = phi i32 [ 0, %{{.+}} ], [ [[BUFFER_READ]], %{{.+}} ]
; IR: call i32 @llvm.amdgcn.ds.bpermute(i32 {{.+}}, i32 [[BUFFER_RESULT]])
  buffer_load_b32 v2, v1, s[8:11], s12 offen
  s_wait_loadcnt 0
  global_store_b32 v1, v2, s[2:3]
; IR: store i32 {{.+}}, ptr addrspace(1) {{.+}}, align 1
  buffer_store_b32 v2, v1, s[8:11], s12 offen offset:4096
; IR: load i8, ptr addrspace(1)
; IR: call i32 @llvm.amdgcn.ds.bpermute
  buffer_load_u8 v3, v1, s[8:11], s12 offen
  s_wait_loadcnt 0
  global_store_b32 v1, v3, s[2:3] offset:4096
  s_endpgm

.globl transpose
.p2align 8
.type transpose,@function
; IR-LABEL: define amdgpu_kernel void @transpose(
transpose:
  s_mov_b32 exec_lo, -1
  s_load_b128 s[4:7], s[0:1], 0
  s_wait_kmcnt 0
  v_lshlrev_b32 v1, 4, v0
  global_load_b128 v[4:7], v1, s[6:7]
  s_wait_loadcnt 0
  ds_store_b128 v1, v[4:7]
  s_wait_dscnt 0
  s_bfe_u32 s8, ttmp8, 0x50019
  s_lshl_b32 s8, s8, 9
  v_mov_b32 v2, s8
  s_mov_b32 exec_lo, 1
; IR: call i32 @llvm.amdgcn.ds.bpermute
; IR: load i16, ptr addrspace(3)
; IR: call i32 @llvm.amdgcn.ds.bpermute
  ds_load_tr16_b128 v[4:7], v2
; IR: load i8, ptr addrspace(3)
; IR: call i32 @llvm.amdgcn.ds.bpermute
  ds_load_tr8_b64 v[8:9], v2
  s_wait_dscnt 0
  s_mov_b32 exec_lo, 0
  ds_load_tr16_b128 v[4:7], v2 offset:65535
  s_mov_b32 exec_lo, -1
  global_store_b128 v1, v[4:7], s[4:5]
  global_store_b64 v1, v[8:9], s[4:5] offset:8192
  s_endpgm

.globl geometry
.p2align 8
.type geometry,@function
; IR-LABEL: define amdgpu_kernel void @geometry(
geometry:
; IR-NOT: call {{.+}} @llvm.amdgcn.dispatch.ptr
; IR: add i64 {{.+}}, 8
; IR: load i32, ptr addrspace(1)
; IR: call i32 @llvm.amdgcn.readlane.i32
; IR: store i32
  s_mov_b32 exec_lo, -1
  s_load_b64 s[2:3], s[0:1], 0
  s_load_b32 s4, s[0:1], 8
  s_wait_kmcnt 0
  v_lshlrev_b32 v1, 2, v0
  v_mov_b32 v2, s4
  global_store_b32 v1, v2, s[2:3]
  s_endpgm

.globl ordinary
.p2align 8
.type ordinary,@function
; IR-LABEL: define amdgpu_kernel void @ordinary(
ordinary:
; IR: call i1 @llvm.amdgcn.init.whole.wave
; IR-NOT: icmp ult i32 {{.+}}, 32
; IR: store i32
  s_load_b64 s[2:3], s[0:1], 0
  s_wait_kmcnt 0
  v_lshlrev_b32 v1, 2, v0
  global_store_b32 v1, v0, s[2:3]
  s_endpgm

; IR: "amdgpu-flat-work-group-size"="64,1024"
; IR: "amdgpu-flat-work-group-size"="1,1024"
; IR: !{i32 192, i32 1, i32 1}
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel ids
  .amdhsa_group_segment_fixed_size 8192
  .amdhsa_kernarg_size 16
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_next_free_vgpr 32
  .amdhsa_next_free_sgpr 16
.end_amdhsa_kernel
.amdhsa_kernel atomic_counts
  .amdhsa_group_segment_fixed_size 8192
  .amdhsa_kernarg_size 16
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_next_free_vgpr 32
  .amdhsa_next_free_sgpr 16
.end_amdhsa_kernel
.amdhsa_kernel loads
  .amdhsa_group_segment_fixed_size 8192
  .amdhsa_kernarg_size 16
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_next_free_vgpr 32
  .amdhsa_next_free_sgpr 16
.end_amdhsa_kernel
.amdhsa_kernel buffer
  .amdhsa_group_segment_fixed_size 8192
  .amdhsa_kernarg_size 16
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_next_free_vgpr 32
  .amdhsa_next_free_sgpr 16
.end_amdhsa_kernel
.amdhsa_kernel transpose
  .amdhsa_group_segment_fixed_size 8192
  .amdhsa_kernarg_size 16
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr 16
  .amdhsa_next_free_sgpr 16
.end_amdhsa_kernel
.amdhsa_kernel geometry
  .amdhsa_group_segment_fixed_size 0
  .amdhsa_kernarg_size 16
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr 4
  .amdhsa_next_free_sgpr 8
.end_amdhsa_kernel
.amdhsa_kernel ordinary
  .amdhsa_group_segment_fixed_size 8192
  .amdhsa_kernarg_size 16
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_next_free_vgpr 32
  .amdhsa_next_free_sgpr 16
.end_amdhsa_kernel
.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: ids
    .symbol: ids.kd
    .group_segment_fixed_size: 8192
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 1024
    .sgpr_count: 16
    .vgpr_count: 32
    .wavefront_size: 32
  - .name: atomic_counts
    .symbol: atomic_counts.kd
    .group_segment_fixed_size: 8192
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 1024
    .sgpr_count: 16
    .vgpr_count: 32
    .wavefront_size: 32
  - .name: loads
    .symbol: loads.kd
    .group_segment_fixed_size: 8192
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 1024
    .sgpr_count: 16
    .vgpr_count: 32
    .wavefront_size: 32
  - .name: buffer
    .symbol: buffer.kd
    .group_segment_fixed_size: 8192
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 1024
    .sgpr_count: 16
    .vgpr_count: 32
    .wavefront_size: 32
  - .name: transpose
    .symbol: transpose.kd
    .group_segment_fixed_size: 8192
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 1024
    .sgpr_count: 16
    .vgpr_count: 16
    .wavefront_size: 32
  - .name: geometry
    .symbol: geometry.kd
    .reqd_workgroup_size: [96, 1, 1]
    .group_segment_fixed_size: 0
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 1024
    .sgpr_count: 8
    .vgpr_count: 4
    .wavefront_size: 32
    .args:
      - .offset: 8
        .size: 2
        .value_kind: hidden_group_size_x
      - .offset: 10
        .size: 2
        .value_kind: hidden_group_size_y
  - .name: ordinary
    .symbol: ordinary.kd
    .group_segment_fixed_size: 8192
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 1024
    .sgpr_count: 16
    .vgpr_count: 32
    .wavefront_size: 32
...
.end_amdgpu_metadata
