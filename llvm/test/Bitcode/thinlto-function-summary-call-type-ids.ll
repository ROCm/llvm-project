; Test the call type identifiers in the per-module and in the combined summary.
; The module has no target triple, so the identifiers use the
; target-independent default. The summary stores the MD5 hash of each
; identifier:
;   13568473337159563479 = "llvm.v1:cc0:f_f32f32f"
;   18428409508369012832 = "llvm.v1:cc0:f_i32i32f"
;   10830334733921094663 = "llvm.v1:cc0:f_isVoidf"
;   657395674655738670   = "llvm.v1:cc0:f_isVoidp0f"
;   11966876714456557523 = "llvm.v1:cc8:f_i32i32f"

; RUN: opt -module-summary %s -o %t.bc
; RUN: llvm-bcanalyzer -dump %t.bc | FileCheck %s --check-prefix=BCAN
; RUN: llvm-dis %t.bc -o - | FileCheck %s
; RUN: llvm-lto2 run %t.bc -o %t.o -thinlto-distributed-indexes \
; RUN:   -r=%t.bc,f_float,px -r=%t.bc,f_i32,px -r=%t.bc,caller,px \
; RUN:   -r=%t.bc,kernel,px -r=%t.bc,fp,px
; RUN: llvm-dis %t.bc.thinlto.bc -o - | FileCheck %s --check-prefix=COMBINED

; The bitcode analyzer prints the operands as signed values.
; BCAN: <CALL_TYPE_IDS op0=657395674655738670 op1=-4878270736549988137 op2=-6479867359252994093/>

; Functions with non-local linkage get their own identifier.
; CHECK-DAG: (name: "f_float", {{.*}}, insts: 1, callTypeId: 13568473337159563479)))
; CHECK-DAG: (name: "f_i32", {{.*}}, insts: 1, callTypeId: 18428409508369012832)))
; A local function gets its own identifier only if its address is taken.
; CHECK-DAG: (name: "local_not_taken", {{.*}}, insts: 1)))
; CHECK-DAG: (name: "local_taken", {{.*}}, insts: 1, callTypeId: 10830334733921094663)))
; A function whose calling convention does not permit calls has no identifier.
; CHECK-DAG: (name: "kernel", {{.*}}, insts: 1)))
; The identifiers of the indirect calls have no duplicates, and the calling
; convention is part of the identifier. Inline assembly is not an indirect call.
; CHECK-DAG: (name: "caller", {{.*}}, calls: ((callee: ^{{[0-9]+}})), callTypeId: 657395674655738670, indirectCallTypeIds: (13568473337159563479, 11966876714456557523))))

; COMBINED-DAG: insts: 1, callTypeId: 13568473337159563479)))
; COMBINED-DAG: insts: 1, callTypeId: 18428409508369012832)))
; COMBINED-DAG: insts: 1, callTypeId: 10830334733921094663)))
; COMBINED-DAG: callTypeId: 657395674655738670, indirectCallTypeIds: (13568473337159563479, 11966876714456557523))))

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@fp = global ptr @local_taken

define float @f_float(float %x) {
  ret float %x
}

define i32 @f_i32(i32 %x) {
  ret i32 %x
}

define internal void @local_not_taken() {
  ret void
}

define internal void @local_taken() {
  ret void
}

define spir_kernel void @kernel(ptr %p) {
  ret void
}

define void @caller(ptr %p) {
  call void @local_not_taken()
  %a = call float %p(float 1.0)
  %b = call float %p(float 2.0)
  %c = call fastcc i32 %p(i32 1)
  call void asm sideeffect "", ""()
  ret void
}
