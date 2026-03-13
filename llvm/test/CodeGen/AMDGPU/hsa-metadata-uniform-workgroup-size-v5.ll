; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj -o - < %s | llvm-readelf --notes - | FileCheck %s
; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=gfx900 < %s | FileCheck %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 -filetype=obj -o - < %s | llvm-readelf --notes - | FileCheck %s --check-prefix=GFX1250-A0
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 -amdgpu-gfx1250-b0-specific -filetype=obj -o - < %s | llvm-readelf --notes - | FileCheck %s --check-prefix=GFX1250-B0

; CHECK: ---
; CHECK: amdhsa.kernels:
; CHECK:  - .args:
; CHECK-LABEL:     .name:           kernel_uniform_workgroup
; CHECK:     .uniform_work_group_size: 1
; GFX1250-A0: ---
; GFX1250-A0: amdhsa.kernels:
; GFX1250-A0:  - .args:
; GFX1250-A0:     .gfx1250_revision: A0
; GFX1250-A0-LABEL:     .name:           kernel_uniform_workgroup
; GFX1250-A0:     .uniform_work_group_size: 1
; GFX1250-B0: ---
; GFX1250-B0: amdhsa.kernels:
; GFX1250-B0:  - .args:
; GFX1250-B0:     .gfx1250_revision: B0
; GFX1250-B0-LABEL:     .name:           kernel_uniform_workgroup
; GFX1250-B0:     .uniform_work_group_size: 1
define amdgpu_kernel void @kernel_uniform_workgroup() #0 {
bb:
  ret void
}

; CHECK:  - .args:
; CHECK-LABEL:     .name:           kernel_non_uniform_workgroup
; CHECK-NOT:     .uniform_work_group_size:
define amdgpu_kernel void @kernel_non_uniform_workgroup() {
bb:
  ret void
}

; CHECK:  - .args:
; CHECK-LABEL:     .name:           kernel_no_attr
; CHECK-NOT:     .uniform_work_group_size:
define amdgpu_kernel void @kernel_no_attr() {
bb:
  ret void
}
attributes #0 = { "uniform-work-group-size" }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdhsa_code_object_version", i32 500}
