; RUN: llc -O0 -mcpu=gfx1100 -mtriple=amdgcn-amd-amdhsa -filetype=obj -o - < %s | llvm-dwarfdump --debug-info - | FileCheck %s

; A frame index addresses an object in the private address space, so a debug
; record whose location operand is a cast of an alloca has to be described the
; same way as one whose location operand is the alloca itself.

; CHECK-LABEL: DW_AT_name ("kernel_cast_of_alloca")
; CHECK: DW_TAG_variable
; CHECK-NEXT: DW_AT_location (DW_OP_regx SGPR33, DW_OP_deref_size 0x4, DW_OP_lit8, DW_OP_plus, DW_OP_lit5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address)
; CHECK-NEXT: DW_AT_name ("kernel_var")
define amdgpu_kernel void @kernel_cast_of_alloca() #0 !dbg !5 {
  %pad = alloca i64, align 8, addrspace(5)
  %var = alloca i32, align 4, addrspace(5)
  %cast = addrspacecast ptr addrspace(5) %var to ptr
    #dbg_declare(ptr %cast, !8, !DIExpression(DIOpArg(0, ptr), DIOpDeref(i32)), !9)
  store i64 0, ptr addrspace(5) %pad, align 8, !dbg !9
  store i32 42, ptr %cast, align 4, !dbg !9
  ret void, !dbg !9
}

; CHECK-LABEL: DW_AT_name ("func_cast_of_alloca")
; CHECK: DW_TAG_variable
; CHECK-NEXT: DW_AT_location (DW_OP_regx SGPR33, DW_OP_deref_size 0x4, DW_OP_lit8, DW_OP_plus, DW_OP_lit5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address)
; CHECK-NEXT: DW_AT_name ("func_var")
define void @func_cast_of_alloca() #0 !dbg !10 {
  %pad = alloca i64, align 8, addrspace(5)
  %var = alloca i32, align 4, addrspace(5)
  %cast = addrspacecast ptr addrspace(5) %var to ptr
    #dbg_declare(ptr %cast, !11, !DIExpression(DIOpArg(0, ptr), DIOpDeref(i32)), !12)
  store i64 0, ptr addrspace(5) %pad, align 8, !dbg !12
  store i32 42, ptr %cast, align 4, !dbg !12
  ret void, !dbg !12
}

; The same variable described by the alloca keeps its location.
; CHECK-LABEL: DW_AT_name ("func_alloca")
; CHECK: DW_TAG_variable
; CHECK-NEXT: DW_AT_location (DW_OP_regx SGPR33, DW_OP_deref_size 0x4, DW_OP_lit8, DW_OP_plus, DW_OP_lit5, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address)
; CHECK-NEXT: DW_AT_name ("alloca_var")
define void @func_alloca() #0 !dbg !13 {
  %pad = alloca i64, align 8, addrspace(5)
  %var = alloca i32, align 4, addrspace(5)
    #dbg_declare(ptr addrspace(5) %var, !14, !DIExpression(DIOpArg(0, ptr addrspace(5)), DIOpDeref(i32)), !15)
  store i64 0, ptr addrspace(5) %pad, align 8, !dbg !15
  store i32 42, ptr addrspace(5) %var, align 4, !dbg !15
  ret void, !dbg !15
}

attributes #0 = { "frame-pointer"="all" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!1, !2, !3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !16, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !{i32 1, !"amdhsa_code_object_version", i32 500}
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 7, !"frame-pointer", i32 2}
!5 = distinct !DISubprogram(name: "kernel_cast_of_alloca", scope: !16, file: !16, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !17)
!6 = !DISubroutineType(types: !7)
!7 = !{}
!8 = !DILocalVariable(name: "kernel_var", scope: !5, file: !16, line: 1, type: !18)
!9 = !DILocation(line: 1, column: 1, scope: !5)
!10 = distinct !DISubprogram(name: "func_cast_of_alloca", scope: !16, file: !16, line: 2, type: !6, scopeLine: 2, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !19)
!11 = !DILocalVariable(name: "func_var", scope: !10, file: !16, line: 2, type: !18)
!12 = !DILocation(line: 2, column: 1, scope: !10)
!13 = distinct !DISubprogram(name: "func_alloca", scope: !16, file: !16, line: 3, type: !6, scopeLine: 3, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !20)
!14 = !DILocalVariable(name: "alloca_var", scope: !13, file: !16, line: 3, type: !18)
!15 = !DILocation(line: 3, column: 1, scope: !13)
!16 = !DIFile(filename: "t.cpp", directory: "/")
!17 = !{!8}
!18 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!19 = !{!11}
!20 = !{!14}
