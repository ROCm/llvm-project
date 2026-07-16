; RUN: llc -O0 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a -verify-machineinstrs -filetype=obj < %s | llvm-dwarfdump - | FileCheck %s

; CHECK:      DW_TAG_formal_parameter
; CHECK-NEXT:   DW_AT_location (
; CHECK-NEXT:    [0x{{.*}}, 0x{{.*}}): DW_OP_regx VGPR{{[0-9]+}}, DW_OP_LLVM_user DW_OP_LLVM_push_lane, DW_OP_lit4, DW_OP_mul, DW_OP_LLVM_user DW_OP_LLVM_offset, DW_OP_piece 0x4, DW_OP_regx VGPR{{[0-9]+}}, DW_OP_LLVM_user DW_OP_LLVM_push_lane, DW_OP_lit4, DW_OP_mul, DW_OP_LLVM_user DW_OP_LLVM_offset, DW_OP_piece 0x4, DW_OP_LLVM_user DW_OP_LLVM_piece_end, DW_OP_deref_size 0x8, DW_OP_lit1, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address
; CHECK-NEXT:    [0x{{.*}}, 0x{{.*}}): DW_OP_regx AGPR{{[0-9]+}}, DW_OP_LLVM_user DW_OP_LLVM_push_lane, DW_OP_lit4, DW_OP_mul, DW_OP_LLVM_user DW_OP_LLVM_offset, DW_OP_piece 0x4, DW_OP_regx AGPR{{[0-9]+}}, DW_OP_LLVM_user DW_OP_LLVM_push_lane, DW_OP_lit4, DW_OP_mul, DW_OP_LLVM_user DW_OP_LLVM_offset, DW_OP_piece 0x4, DW_OP_LLVM_user DW_OP_LLVM_piece_end, DW_OP_deref_size 0x8, DW_OP_lit1, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address)
; CHECK-NEXT:   DW_AT_name ("a")

; CHECK:      DW_TAG_formal_parameter
; CHECK-NEXT:   DW_AT_location (
; CHECK-NEXT:    [0x{{.*}}, 0x{{.*}}): DW_OP_regx VGPR{{[0-9]+}}, DW_OP_LLVM_user DW_OP_LLVM_push_lane, DW_OP_lit4, DW_OP_mul, DW_OP_LLVM_user DW_OP_LLVM_offset, DW_OP_piece 0x4, DW_OP_regx VGPR{{[0-9]+}}, DW_OP_LLVM_user DW_OP_LLVM_push_lane, DW_OP_lit4, DW_OP_mul, DW_OP_LLVM_user DW_OP_LLVM_offset, DW_OP_piece 0x4, DW_OP_LLVM_user DW_OP_LLVM_piece_end, DW_OP_deref_size 0x8, DW_OP_lit1, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address
; CHECK-NEXT:    [0x{{.*}}, 0x{{.*}}): DW_OP_regx AGPR{{[0-9]+}}, DW_OP_LLVM_user DW_OP_LLVM_push_lane, DW_OP_lit4, DW_OP_mul, DW_OP_LLVM_user DW_OP_LLVM_offset, DW_OP_piece 0x4, DW_OP_regx AGPR{{[0-9]+}}, DW_OP_LLVM_user DW_OP_LLVM_push_lane, DW_OP_lit4, DW_OP_mul, DW_OP_LLVM_user DW_OP_LLVM_offset, DW_OP_piece 0x4, DW_OP_LLVM_user DW_OP_LLVM_piece_end, DW_OP_deref_size 0x8, DW_OP_lit1, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address)
; CHECK-NEXT:   DW_AT_name ("b")

define void @_QFPadd(ptr %0, ptr %1) #0 !dbg !8 {
    #dbg_declare(ptr %0, !11, !DIExpression(DIOpArg(0, ptr), DIOpDeref(ptr)), !13)
    #dbg_declare(ptr %1, !12, !DIExpression(DIOpArg(0, ptr), DIOpDeref(ptr)), !13)
  %3 = load float, ptr %0, align 4, !dbg !13
  %4 = fadd contract float %3, 1.000000e+00, !dbg !13
  store float %4, ptr %0, align 4, !dbg !13
  %5 = load float, ptr %0, align 4, !dbg !13
  %6 = load float, ptr %1, align 4, !dbg !13
  %7 = fcmp contract ogt float %5, %6, !dbg !13
  br i1 %7, label %then, label %else, !dbg !13

then:
  %10 = load float, ptr %0, align 4, !dbg !13
  %11 = fadd contract float %10, 2.000000e+00, !dbg !13
  store float %11, ptr %0, align 4, !dbg !13
  br label %join, !dbg !13

else:
  %14 = load float, ptr %1, align 4, !dbg !13
  %15 = fadd contract float %14, 2.000000e+00, !dbg !13
  store float %15, ptr %0, align 4, !dbg !13
  br label %join, !dbg !13

join:
  ret void, !dbg !13
}

attributes #0 = { "frame-pointer"="all" "target-cpu"="gfx90a" }
!llvm.module.flags = !{!2}
!llvm.dbg.cu = !{!4}

!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !DIFile(filename: "target8-openmp-amdgcn-amd-amdhsa-gfx90a.i", directory: "")
!4 = distinct !DICompileUnit(language: DW_LANG_Fortran95, file: !3, producer: "flang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !5)
!5 = !{}
!6 = !DIBasicType(name: "real", size: 32, encoding: DW_ATE_float)
!8 = distinct !DISubprogram(name: "add", linkageName: "_QFPadd", scope: !3, file: !3, line: 36, type: !9, scopeLine: 36, spFlags: DISPFlagDefinition, unit: !4)
!9 = !DISubroutineType(cc: DW_CC_normal, types: !10)
!10 = !{null}
!11 = !DILocalVariable(name: "a", arg: 1, scope: !8, file: !3, line: 39, type: !6)
!12 = !DILocalVariable(name: "b", arg: 2, scope: !8, file: !3, line: 40, type: !6)
!13 = !DILocation(line: 1, column: 1, scope: !8)
