; RUN: llc -mtriple=amdgcn--amdpal -mcpu=gfx1150 -O0 -filetype=obj -o - < %s | llvm-dwarfdump --debug-info - | FileCheck %s
; RUN: llc -mtriple=amdgcn--amdpal -mcpu=gfx1150 -global-isel -O0 -filetype=obj -o - < %s | llvm-dwarfdump --debug-info - | FileCheck %s

; Test lane focusing for the amdpal target and amdgpu_cs calling convention.
; The LocalInvocationId VGPR argument makes the computation divergent.
; Wave32 VGPRs are printed as raw DWARF registers in the 0x600 range.

; CHECK-LABEL: DW_AT_name ("CSMain")
; CHECK:      DW_TAG_variable
; CHECK:      DW_AT_location
; CHECK-NEXT: {{.*\): }}DW_OP_regx {{0x6[0-9a-f]+}}, DW_OP_LLVM_user DW_OP_LLVM_push_lane, DW_OP_lit4, DW_OP_mul, DW_OP_LLVM_user DW_OP_LLVM_offset)
; CHECK:      DW_AT_name ("first_idx")

define amdgpu_cs void @test_amdpal_cs(i32 inreg noundef %globalTable, i32 noundef %LocalInvocationId) #0 !dbg !5 {
entry:
  %tid = and i32 %LocalInvocationId, 1023, !dbg !11
  %val = shl i32 %tid, 3, !dbg !12
    #dbg_value(i32 %val, !9, !DIExpression(DIOpArg(0, i32)), !12)
  call void asm sideeffect "; use $0", "v"(i32 %val), !dbg !13
  ret void, !dbg !14
}

; Also verify that uniform (SGPR) inreg arguments do NOT get lane offset
; even in the amdpal/amdgpu_cs context.

; CHECK-LABEL: DW_AT_name ("CSMain_uniform")
; CHECK:      DW_TAG_variable
; CHECK:      DW_AT_location
; CHECK-NEXT: DW_OP_regx SGPR[[#]]
; CHECK:      DW_AT_name ("uval")

define amdgpu_cs void @test_amdpal_cs_uniform(i32 inreg noundef %userdata, i32 noundef %LocalInvocationId) #0 !dbg !15 {
entry:
  %val = add i32 %userdata, 1, !dbg !18
    #dbg_value(i32 %val, !17, !DIExpression(DIOpArg(0, i32)), !18)
  call void asm sideeffect "; use $0", "s"(i32 %val), !dbg !19
  ret void, !dbg !20
}

attributes #0 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "dxc", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "shader.hlsl", directory: "")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}

; Function 1: test_amdpal_cs (divergent VGPR value)
!5 = distinct !DISubprogram(name: "CSMain", scope: !1, file: !1, line: 6, type: !6, scopeLine: 7, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !8)
!6 = !DISubroutineType(types: !7)
!7 = !{null}
!8 = !{!9}
!9 = !DILocalVariable(name: "first_idx", scope: !5, file: !1, line: 10, type: !10)
!10 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!11 = !DILocation(line: 6, column: 43, scope: !5)
!12 = !DILocation(line: 10, column: 35, scope: !5)
!13 = !DILocation(line: 10, column: 16, scope: !5)
!14 = !DILocation(line: 14, column: 1, scope: !5)

; Function 2: test_amdpal_cs_uniform (uniform SGPR value - no lane offset)
!15 = distinct !DISubprogram(name: "CSMain_uniform", scope: !1, file: !1, line: 20, type: !6, scopeLine: 21, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !16)
!16 = !{!17}
!17 = !DILocalVariable(name: "uval", scope: !15, file: !1, line: 22, type: !10)
!18 = !DILocation(line: 22, column: 1, scope: !15)
!19 = !DILocation(line: 23, column: 1, scope: !15)
!20 = !DILocation(line: 24, column: 1, scope: !15)
