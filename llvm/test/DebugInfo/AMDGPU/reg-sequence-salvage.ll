; NOTE: Assertions have been autogenerated by utils/update_mir_test_checks.py UTC_ARGS: --version 5
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1030 -start-before=amdgpu-isel -stop-after=amdgpu-isel %s -o - | FileCheck %s

define i64 @test(ptr addrspace(1) %p) !dbg !11 {
  ; CHECK-LABEL: name: test
  ; CHECK: bb.0 (%ir-block.0):
  ; CHECK-NEXT:   liveins: $vgpr0, $vgpr1
  ; CHECK-NEXT: {{  $}}
  ; CHECK-NEXT:   [[COPY:%[0-9]+]]:vgpr_32 = COPY $vgpr1
  ; CHECK-NEXT:   [[COPY1:%[0-9]+]]:vgpr_32 = COPY $vgpr0
  ; CHECK-NEXT:   [[REG_SEQUENCE:%[0-9]+]]:sreg_64 = REG_SEQUENCE [[COPY1]], %subreg.sub0, [[COPY]], %subreg.sub1
  ; CHECK-NEXT:   [[COPY2:%[0-9]+]]:vreg_64 = COPY [[REG_SEQUENCE]]
  ; CHECK-NEXT:   [[GLOBAL_LOAD_DWORD:%[0-9]+]]:vgpr_32 = GLOBAL_LOAD_DWORD killed [[COPY2]], 0, 0, implicit $exec,  debug-instr-number 1 :: (load (s32) from %ir.p, addrspace 1)
  ; CHECK-NEXT:   [[V_ASHRREV_I32_e64_:%[0-9]+]]:vgpr_32 = V_ASHRREV_I32_e64 31, [[GLOBAL_LOAD_DWORD]], implicit $exec,  debug-instr-number 2
  ; CHECK-NEXT:   [[COPY3:%[0-9]+]]:vgpr_32 = COPY killed [[V_ASHRREV_I32_e64_]]
  ; CHECK-NEXT:   [[REG_SEQUENCE1:%[0-9]+]]:vreg_64 = REG_SEQUENCE [[GLOBAL_LOAD_DWORD]], %subreg.sub0, killed [[COPY3]], %subreg.sub1
  ; CHECK-NEXT:   DBG_INSTR_REF !17, !DIExpression(DIOpArg(0, i32), DIOpArg(1, i32), DIOpComposite(2, i64)), dbg-instr-ref(1, 0), dbg-instr-ref(2, 0),  debug-location !18
  ; CHECK-NEXT:   [[COPY4:%[0-9]+]]:vgpr_32 = COPY [[REG_SEQUENCE1]].sub1
  ; CHECK-NEXT:   $vgpr0 = COPY [[GLOBAL_LOAD_DWORD]]
  ; CHECK-NEXT:   $vgpr1 = COPY [[COPY4]]
  ; CHECK-NEXT:   SI_RETURN implicit $vgpr0, implicit $vgpr1
  %load = load i32, ptr addrspace(1) %p, align 4
  %conv = sext i32 %load to i64
    #dbg_value(i64 %conv, !17, !DIExpression(DIOpArg(0, i64)), !18)
  ret i64 %conv
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7}
!opencl.ocl.version = !{!8}
!llvm.ident = !{!9, !10}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 21.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug)
!1 = !DIFile(filename: "t.cpp", directory: "/")
!2 = !{i32 1, !"amdhsa_code_object_version", i32 600}
!3 = !{i32 1, !"amdgpu_printf_kind", !"hostcall"}
!4 = !{i32 7, !"Dwarf Version", i32 5}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = !{i32 1, !"wchar_size", i32 4}
!7 = !{i32 8, !"PIC Level", i32 2}
!8 = !{i32 2, i32 0}
!9 = !{!"clang version 21.0.0"}
!10 = !{!"clang version 18.0.0"}
!11 = distinct !DISubprogram(name: "test", linkageName: "test", scope: !1, file: !1, line: 6, type: !12, scopeLine: 7, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !16)
!12 = !DISubroutineType(types: !13)
!13 = !{!15, !14}
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !15, size: 64)
!15 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!16 = !{!17}
!17 = !DILocalVariable(name: "var", scope: !11, file: !1, line: 8, type: !15)
!18 = !DILocation(line: 0, scope: !11)
