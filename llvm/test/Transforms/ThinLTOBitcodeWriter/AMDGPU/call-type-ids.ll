; Test that the summaries that the writer builds again for a module with type
; metadata get the call type IDs of the target, like the summary that
; ModuleSummaryIndexAnalysis builds. The summaries store the MD5 hash of each
; identifier:
;   2752581399632794192 = "amdgpu.v1:cc0:il"
;   5970146889962867861 = "amdgpu.v1:cc0:vl"

; RUN: rm -rf %t && split-file %s %t && cd %t

; The writer promotes the type IDs and builds the summary again.
; RUN: opt -thinlto-bc main.ll -o main.bc
; RUN: llvm-dis main.bc -o - | FileCheck %s

; The writer splits the module. Both parts have a summary.
; RUN: opt -thinlto-bc -thinlto-split-lto-unit main.ll -o split.bc
; RUN: llvm-modextract -b -n 0 split.bc -o - | llvm-dis -o - | FileCheck %s
; RUN: llvm-modextract -b -n 1 split.bc -o - | llvm-dis -o - \
; RUN:   | FileCheck %s --check-prefix=MERGED

; The module has no external definition, so the writer writes it as a regular
; LTO module with a summary.
; RUN: opt -thinlto-bc -thinlto-split-lto-unit internal.ll -o internal.bc
; RUN: llvm-dis internal.bc -o - | FileCheck %s --check-prefix=INTERNAL

; CHECK-DAG: (name: "vf", {{.*}}, callTypeId: 2752581399632794192)))
; CHECK-DAG: (name: "f", {{.*}}, callTypeId: 5970146889962867861)))

; MERGED: (name: "vf", {{.*}}linkage: available_externally{{.*}}, callTypeId: 2752581399632794192)))

; INTERNAL: (name: "vf", {{.*}}linkage: internal{{.*}}, callTypeId: 2752581399632794192)))

;--- main.ll
target triple = "amdgcn-amd-amdhsa"

@vt = constant [1 x ptr] [ptr @vf], !type !0

define i32 @vf(ptr %this) {
  ret i32 0
}

define void @f(ptr %p) {
  ret void
}

!0 = !{i32 0, !"typeid"}

;--- internal.ll
target triple = "amdgcn-amd-amdhsa"

@vt = internal constant [1 x ptr] [ptr @vf], !type !0

define internal i32 @vf(ptr %this) {
  ret i32 0
}

!0 = !{i32 0, !"typeid"}
