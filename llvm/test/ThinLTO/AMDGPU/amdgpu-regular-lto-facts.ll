; Test that regular LTO modules with summaries link when they have the same
; linkonce_odr function. Their summaries hold AMDGPU facts, and LTO reads the
; summaries of all regular LTO modules with the same module path.
;
; RUN: rm -rf %t && split-file %s %t && cd %t
; RUN: opt -module-summary a.ll -o a.bc
; RUN: opt -module-summary b.ll -o b.bc
; RUN: llvm-dis a.bc -o - | FileCheck %s --check-prefix=FACTS
; RUN: llvm-lto2 run a.bc b.bc -o out \
; RUN:   -r=a.bc,f,px -r=a.bc,ga,px -r=b.bc,f,x -r=b.bc,gb,px

; FACTS: = extension: (name: "amdgpu-occupancy",

;--- a.ll
target triple = "amdgcn-amd-amdhsa"

define linkonce_odr void @f() #0 {
  ret void
}

define void @ga() {
  call void @f()
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="1,256" }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"ThinLTO", i32 0}

;--- b.ll
target triple = "amdgcn-amd-amdhsa"

define linkonce_odr void @f() #0 {
  ret void
}

define void @gb() {
  call void @f()
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="1,256" }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"ThinLTO", i32 0}
