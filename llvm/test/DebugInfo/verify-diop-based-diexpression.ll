; RUN: rm -rf %t && split-file %s %t && cd %t

;--- valid.ll
; RUN: opt valid.ll -S -passes=verify 2>&1 | FileCheck --implicit-check-not 'invalid expression' valid.ll

source_filename = "t.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @llvm.dbg.declare(metadata, metadata, metadata)
declare void @llvm.dbg.value(metadata, metadata, metadata)

%struct.type = type { ptr, ptr }

define dso_local void @test_diexpr_eval() !dbg !17 {
entry:
  %x = alloca ptr, align 8
  %i = alloca i32, align 4

  ; CHECK: #dbg_declare(ptr %i, ![[#]], !DIExpression(DIOpArg(0, ptr), DIOpArg(0, ptr), DIOpComposite(2, %struct.type)),
  #dbg_declare(ptr %i, !21, !DIExpression(DIOpArg(0, ptr), DIOpArg(0, ptr), DIOpComposite(2, %struct.type)), !22)

  ; CHECK: #dbg_declare(i16 42, ![[#]], !DIExpression(DIOpArg(0, i16), DIOpFragment(16, 16)),
  #dbg_declare(i16 42, !21, !DIExpression(DIOpArg(0, i16), DIOpFragment(16, 16)), !22)

  ; CHECK: #dbg_declare(i8 poison, ![[#]], !DIExpression(DIOpArg(0, i32)), ![[#]])
  #dbg_declare(i8 poison, !24, !DIExpression(DIOpArg(0, i32)), !22)

  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 18", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "t.c", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = distinct !DISubprogram(name: "test_broken_declare", scope: !1, file: !1, line: 2, type: !6, scopeLine: 2, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !8)
!6 = !DISubroutineType(types: !7)
!7 = !{null}
!8 = !{}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DILocation(line: 3, column: 7, scope: !5)
!12 = !DILocation(line: 4, column: 1, scope: !5)
!13 = distinct !DISubprogram(name: "test_broken_value", scope: !1, file: !1, line: 6, type: !6, scopeLine: 6, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !8)
!15 = !DILocation(line: 7, column: 7, scope: !13)
!16 = !DILocation(line: 8, column: 1, scope: !13)
!17 = distinct !DISubprogram(name: "test_diexpr_eval", scope: !1, file: !1, line: 10, type: !6, scopeLine: 10, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !8)
!18 = !DILocalVariable(name: "x", scope: !17, file: !1, line: 11, type: !19)
!19 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!20 = !DILocation(line: 11, column: 9, scope: !17)
!21 = !DILocalVariable(name: "i", scope: !17, file: !1, line: 12, type: !10)
!22 = !DILocation(line: 12, column: 7, scope: !17)
!23 = !DILocation(line: 13, column: 1, scope: !17)
!24 = !DILocalVariable(name: "j", scope: !17, file: !1, line: 12, type: !10)

;--- invalid.ll
; RUN: opt invalid.ll -S -passes=verify 2>&1 | FileCheck invalid.ll

source_filename = "t.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @llvm.dbg.declare(metadata, metadata, metadata)
declare void @llvm.dbg.value(metadata, metadata, metadata)

define dso_local void @test_diexpr_eval() !dbg !17 {
entry:
  %x = alloca ptr, align 8
  %i = alloca i32, align 4


  ; FIXME(diexpression-poison): DIOpArg index out of range
  #dbg_declare(ptr %x, !18, !DIExpression(DIOpArg(1, ptr)), !20)

  ; CHECK: DIOpArg type must be same size in bits as argument
  #dbg_declare(ptr %x, !18, !DIExpression(DIOpArg(0, i32)), !20)

  ; CHECK: DIOpReinterpret must not alter bitsize of child
  #dbg_declare(ptr %x, !18, !DIExpression(DIOpArg(0, ptr), DIOpReinterpret(i32)), !20)

  ; CHECK: DIOpBitOffset requires first input be integer typed
  #dbg_declare(ptr %x, !18, !DIExpression(DIOpConstant(float 0.0), DIOpArg(0, ptr), DIOpBitOffset(ptr)), !20)

  ; CHECK: DIOpByteOffset requires first input be integer typed
  #dbg_declare(ptr %x, !18, !DIExpression(DIOpConstant(ptr undef), DIOpArg(0, ptr), DIOpByteOffset(ptr)), !20)

  ; CHECK: DIOpComposite bitsize does not match sum of child bitsizes
  #dbg_declare(ptr %i, !21, !DIExpression(DIOpArg(0, ptr), DIOpDeref(i16), DIOpConstant(i8 0), DIOpComposite(2, i32)), !22)

  ; CHECK: DIOpExtend child must have integer, floating point, or ptr type
  #dbg_declare(ptr %i, !21, !DIExpression(DIOpConstant(<2 x i32> <i32 0, i32 0>), DIOpExtend(2)), !22)

  ; CHECK: DIOpDeref requires input to be pointer typed
  #dbg_declare(ptr %i, !21, !DIExpression(DIOpArg(0, ptr), DIOpDeref(i32), DIOpDeref(i32)), !22)

  ; CHECK: DIOpAdd requires identical type inputs
  #dbg_declare(ptr %i, !21, !DIExpression(DIOpConstant(i32 0), DIOpConstant(i8 0), DIOpAdd()), !22)

  ; CHECK: DIOpPushLane requires integer result type
  #dbg_declare(ptr %i, !21, !DIExpression(DIOpPushLane(ptr)), !22)

  ; CHECK: DIOpAdd requires more inputs
  #dbg_declare(ptr %i, !21, !DIExpression(DIOpConstant(i32 0), DIOpAdd()), !22)

  ; CHECK: DIOpArg type must be same size in bits as argument
  #dbg_declare(!DIArgList(ptr %x, ptr %i), !21, !DIExpression(DIOpArg(0, i32), DIOpArg(1, i32), DIOpAdd()), !22)

  ; CHECK: DIOpArg type must be same size in bits as argument
  #dbg_declare(!DIArgList(ptr %x, ptr %i), !21, !DIExpression(DIOpArg(0, i8), DIOpArg(1, i8), DIOpAdd()), !22)

  ; CHECK: DIOp expression requires one element on stack after evaluating
  #dbg_declare(!DIArgList(ptr %x, ptr %i), !21, !DIExpression(DIOpArg(0, i64), DIOpArg(1, i64)), !22)

  ; CHECK: DIOpZExt requires integer typed input
  #dbg_declare(ptr %i, !21, !DIExpression(DIOpArg(0, ptr), DIOpZExt(i64)), !22)

  ; CHECK: DIOpZExt requires result type to be wider than input type
  #dbg_declare(ptr %i, !21, !DIExpression(DIOpArg(0, i64), DIOpZExt(i64)), !22)

  ; CHECK: DIOpSExt requires integer typed input
  #dbg_declare(ptr %i, !21, !DIExpression(DIOpArg(0, ptr), DIOpSExt(i64)), !22)

  ; CHECK: DIOpSExt requires result type to be wider than input type
  #dbg_declare(ptr %i, !21, !DIExpression(DIOpArg(0, i64), DIOpSExt(i64)), !22)

  ; CHECK: DIOpLShr requires all integer inputs
  #dbg_declare(ptr %i, !21, !DIExpression(DIOpArg(0, ptr), DIOpArg(0, ptr), DIOpLShr()), !22)

  ; CHECK: DIOpAShr requires all integer inputs
  #dbg_declare(ptr %i, !21, !DIExpression(DIOpArg(0, ptr), DIOpArg(0, ptr), DIOpAShr()), !22)

  ; CHECK: DIOpShl requires all integer inputs
  #dbg_declare(ptr %i, !21, !DIExpression(DIOpArg(0, ptr), DIOpArg(0, ptr), DIOpShl()), !22)

  ; CHECK: DIOpConvert on integers requires result type to be no wider than input type
  #dbg_declare(i8 42, !21, !DIExpression(DIOpArg(0, i8), DIOpConvert(i16)), !22)

  ; FIXME(diexpression-poison): DIExpression must yield a location at least as wide as the variable or fragment it describes
  ;#dbg_declare(i8 42, !21, !DIExpression(DIOpArg(0, i8)), !22)

  ; FIXME(diexpression-poison): DIExpression must yield a location at least as wide as the variable or fragment it describes
  ;#dbg_declare(ptr %i, !21, !DIExpression(DIOpArg(0, ptr), DIOpDeref(i16), DIOpConstant(i16 1), DIOpAdd()), !22)

  ; FIXME(diexpression-poison): DIExpression must yield a location at least as wide as the variable or fragment it describes
  ;#dbg_declare(i8 42, !21, !DIExpression(DIOpArg(0, i8), DIOpFragment(0, 16)), !22)

  ; CHECK: DIOpFragment must be contained within variable
  #dbg_declare(i16 42, !21, !DIExpression(DIOpArg(0, i16), DIOpFragment(24, 16)), !22)

  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 18", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "t.c", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = distinct !DISubprogram(name: "test_broken_declare", scope: !1, file: !1, line: 2, type: !6, scopeLine: 2, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !8)
!6 = !DISubroutineType(types: !7)
!7 = !{null}
!8 = !{}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DILocation(line: 3, column: 7, scope: !5)
!12 = !DILocation(line: 4, column: 1, scope: !5)
!13 = distinct !DISubprogram(name: "test_broken_value", scope: !1, file: !1, line: 6, type: !6, scopeLine: 6, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !8)
!15 = !DILocation(line: 7, column: 7, scope: !13)
!16 = !DILocation(line: 8, column: 1, scope: !13)
!17 = distinct !DISubprogram(name: "test_diexpr_eval", scope: !1, file: !1, line: 10, type: !6, scopeLine: 10, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !8)
!18 = !DILocalVariable(name: "x", scope: !17, file: !1, line: 11, type: !19)
!19 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!20 = !DILocation(line: 11, column: 9, scope: !17)
!21 = !DILocalVariable(name: "i", scope: !17, file: !1, line: 12, type: !10)
!22 = !DILocation(line: 12, column: 7, scope: !17)
!23 = !DILocation(line: 13, column: 1, scope: !17)
