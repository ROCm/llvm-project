; RUN: llc -O0 -mcpu=gfx1100 -mtriple=amdgcn-amd-amdhsa -filetype=obj -o - < %s | llvm-dwarfdump --debug-info - | FileCheck %s

; Verify that DIOp::PushLane in a DIExpression is correctly emitted as
; DW_OP_LLVM_push_lane in DWARF output, rather than being treated as
; unimplemented (which would produce DW_OP_LLVM_undefined).
;
; Uses an amdgpu_kernel with a uniform (SGPR) argument and an explicit
; PushLane in the DIExpression to isolate the DwarfExpression emission
; from any implicit lane-offset injection (focusThreadIfRequired only
; fires for VGPR registers with non-zero getDwarfRegLaneSize).

@glob = addrspace(1) global i32 0

; CHECK-LABEL: DW_AT_name ("test_pushlane_emit")
; CHECK:       DW_AT_location
; CHECK:       DW_OP_LLVM_push_lane
; CHECK-NOT:   DW_OP_LLVM_undefined
; CHECK:       DW_AT_name ("val")

define amdgpu_kernel void @test_pushlane_emit(i32 %a) !dbg !5 {
    #dbg_value(i32 %a, !9, !DIExpression(DIOpArg(0, i32), DIOpPushLane(i32), DIOpConstant(i32 4), DIOpMul(), DIOpByteOffset(i32)), !11)
  store i32 %a, ptr addrspace(1) @glob, align 4, !dbg !11
  ret void, !dbg !12
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !13)
!1 = !DIFile(filename: "test.cl", directory: "/tmp")
!2 = !{i32 1, !"amdhsa_code_object_version", i32 500}
!3 = !{i32 7, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "test_pushlane_emit", scope: !1, file: !1, line: 1, type: !6, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !8)
!6 = !DISubroutineType(types: !7)
!7 = !{null}
!8 = !{!9}
!9 = !DILocalVariable(name: "val", scope: !5, file: !1, line: 2, type: !10)
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DILocation(line: 2, column: 1, scope: !5)
!12 = !DILocation(line: 3, column: 1, scope: !5)
!13 = !{}
