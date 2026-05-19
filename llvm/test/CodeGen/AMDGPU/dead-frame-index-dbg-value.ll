; RUN: llc -O0 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 < %s | FileCheck %s
; RUN: llc -O0 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a < %s | FileCheck %s

; Check that debug values referencing eliminated frame indices don't crash when
; VGPR spills are folded to AGPRs in SIFrameLowering::processFunctionBeforeFrameFinalized
; (requires hasMAIInsts(), hasSpilledVGPRs(), and amdgpu-spill-vgpr-to-agpr).
; Consumer gfx9 (e.g. gfx900) and RDNA3 gfx11 lack FeatureMAIInsts, so they never
; take that path; use CDNA-class targets and forced VGPR spills.
;
; At -O0 AMDGPU uses RegAllocFast for VGPRs (not LiveDebugVariables::emitDebugValues).
; Fast RA rewrites #dbg_value to a physical register when the vreg is already
; assigned; updateDbgValueInstForSpillFIs only sees DBG_VALUE operands with FI if
; the watched value is actually spilled (buildDbgValueForSpill in spill()). %v5 is
; kept in a VGPR for the 10-operand inline asm, so dbg on %v5 never gets an FI.
; Use %v2, which is spilled under register pressure (spill-vgpr-to-agpr.ll pattern).
; amdgpu-num-vgpr must be 11, not 10, or inline asm fails before RA finishes.

%struct.Buffer = type { [8 x i64] }

; CHECK-LABEL: test_dbg_value_dead_frame_idx:
; Exercise VGPR-spill -> AGPR migration (not scratch) on CDNA.
; CHECK-DAG: v_accvgpr_write_b32
; DIArgList on kernel args (typically stays in registers).
; CHECK-DAG: ;DEBUG_VALUE: test_dbg_value_dead_frame_idx:slot <- [DW_OP_LLVM_arg 0, DW_OP_LLVM_arg 1, DW_OP_constu 64, DW_OP_mul, DW_OP_plus, DW_OP_stack_value] {{.*}}
; Spilled VGPR tracked for debug (may use stack location then FI cleanup).
; CHECK-DAG: ;DEBUG_VALUE: test_dbg_value_dead_frame_idx:spill_witness {{.*}}
; CHECK: s_endpgm
define amdgpu_kernel void @test_dbg_value_dead_frame_idx(ptr addrspace(1) %out, i64 %idx) #0 !dbg !10 {
entry:
  #dbg_value(!DIArgList(ptr addrspace(1) %out, i64 %idx), !15, !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_constu, 64, DW_OP_mul, DW_OP_plus, DW_OP_stack_value), !17)
  %tid = load volatile i32, ptr addrspace(1) poison
  call void asm sideeffect "", "a,a,a,a,a,a,a,a,a"(i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9)
  %p1 = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %tid
  %p2 = getelementptr inbounds i32, ptr addrspace(1) %p1, i32 4
  %p3 = getelementptr inbounds i32, ptr addrspace(1) %p2, i32 8
  %p4 = getelementptr inbounds i32, ptr addrspace(1) %p3, i32 12
  %p5 = getelementptr inbounds i32, ptr addrspace(1) %p4, i32 16
  %p6 = getelementptr inbounds i32, ptr addrspace(1) %p5, i32 20
  %p7 = getelementptr inbounds i32, ptr addrspace(1) %p6, i32 24
  %p8 = getelementptr inbounds i32, ptr addrspace(1) %p7, i32 28
  %p9 = getelementptr inbounds i32, ptr addrspace(1) %p8, i32 32
  %p10 = getelementptr inbounds i32, ptr addrspace(1) %p9, i32 36
  %v1 = load volatile i32, ptr addrspace(1) %p1
  %v2 = load volatile i32, ptr addrspace(1) %p2
  #dbg_value(i32 %v2, !19, !DIExpression(), !20)
  %v3 = load volatile i32, ptr addrspace(1) %p3
  %v4 = load volatile i32, ptr addrspace(1) %p4
  %v5 = load volatile i32, ptr addrspace(1) %p5
  %v6 = load volatile i32, ptr addrspace(1) %p6
  %v7 = load volatile i32, ptr addrspace(1) %p7
  %v8 = load volatile i32, ptr addrspace(1) %p8
  %v9 = load volatile i32, ptr addrspace(1) %p9
  %v10 = load volatile i32, ptr addrspace(1) %p10
  call void asm sideeffect "", "v,v,v,v,v,v,v,v,v,v"(i32 %v1, i32 %v2, i32 %v3, i32 %v4, i32 %v5, i32 %v6, i32 %v7, i32 %v8, i32 %v9, i32 %v10)
  store volatile i32 %v1, ptr addrspace(1) poison
  store volatile i32 %v2, ptr addrspace(1) poison
  store volatile i32 %v3, ptr addrspace(1) poison
  store volatile i32 %v4, ptr addrspace(1) poison
  store volatile i32 %v5, ptr addrspace(1) poison
  store volatile i32 %v6, ptr addrspace(1) poison
  store volatile i32 %v7, ptr addrspace(1) poison
  store volatile i32 %v8, ptr addrspace(1) poison
  store volatile i32 %v9, ptr addrspace(1) poison
  store volatile i32 %v10, ptr addrspace(1) poison
  %ptr = getelementptr %struct.Buffer, ptr addrspace(1) %out, i64 %idx
  store i64 0, ptr addrspace(1) %ptr, align 8
  ret void
}

attributes #0 = { nounwind "amdgpu-num-vgpr"="11" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "test.cpp", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"amdhsa_code_object_version", i32 500}
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 64, addressSpace: 1)
!7 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Buffer", file: !1, line: 1, size: 512, flags: DIFlagTypePassByValue, elements: !2)
!10 = distinct !DISubprogram(name: "test_dbg_value_dead_frame_idx", scope: !1, file: !1, line: 10, type: !11, scopeLine: 10, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!11 = !DISubroutineType(types: !12)
!12 = !{null, !6, !18}
!18 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!15 = !DILocalVariable(name: "slot", scope: !10, file: !1, line: 11, type: !6)
!17 = !DILocation(line: 11, column: 1, scope: !10)
!19 = !DILocalVariable(name: "spill_witness", scope: !10, file: !1, line: 12, type: !21)
!20 = !DILocation(line: 12, column: 1, scope: !10)
!21 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
