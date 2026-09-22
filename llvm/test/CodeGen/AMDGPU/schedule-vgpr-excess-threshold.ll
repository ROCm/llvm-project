; REQUIRES: asserts

; Test the -amdgpu-vgpr-excess-threshold-percent flag which controls the VGPR
; excess register pressure threshold as a percentage of available VGPRs.
; When the threshold percentage would reduce VGPRExcessLimit below VGPRCriticalLimit,
; the critical limit is also adjusted to maintain VGPRCriticalLimit <= VGPRExcessLimit.

; Default behavior (no flag or 0 = use default calculation)
; RUN: llc -mtriple=amdgcn -mcpu=gfx906 -debug-only=machine-scheduler -filetype=null < %s 2>&1 \
; RUN:   | FileCheck -check-prefix=DEFAULT %s

; Test various threshold percentages
; RUN: llc -mtriple=amdgcn -mcpu=gfx906 -amdgpu-vgpr-excess-threshold-percent=90 \
; RUN:   -debug-only=machine-scheduler -filetype=null < %s 2>&1 \
; RUN:   | FileCheck -check-prefix=THRESH90 %s

; RUN: llc -mtriple=amdgcn -mcpu=gfx906 -amdgpu-vgpr-excess-threshold-percent=50 \
; RUN:   -debug-only=machine-scheduler -filetype=null < %s 2>&1 \
; RUN:   | FileCheck -check-prefix=THRESH50 %s

; RUN: llc -mtriple=amdgcn -mcpu=gfx906 -amdgpu-vgpr-excess-threshold-percent=30 \
; RUN:   -debug-only=machine-scheduler -filetype=null < %s 2>&1 \
; RUN:   | FileCheck -check-prefix=THRESH30 %s

; RUN: llc -mtriple=amdgcn -mcpu=gfx906 -amdgpu-vgpr-excess-threshold-percent=10 \
; RUN:   -debug-only=machine-scheduler -filetype=null < %s 2>&1 \
; RUN:   | FileCheck -check-prefix=THRESH10 %s

; Test with 5% threshold which causes VGPRCriticalLimit adjustment
; RUN: llc -mtriple=amdgcn -mcpu=gfx906 -amdgpu-vgpr-excess-threshold-percent=5 \
; RUN:   -debug-only=machine-scheduler -filetype=null < %s 2>&1 \
; RUN:   | FileCheck -check-prefix=THRESH5 %s

; DEFAULT: VGPRCriticalLimit = 21, VGPRExcessLimit = 253
; DEFAULT-NOT: Applied VGPR excess threshold

; THRESH90: Applied VGPR excess threshold 90%, VGPRExcessLimit: 256 -> 231
; THRESH90: VGPRCriticalLimit = 21, VGPRExcessLimit = 231

; THRESH50: Applied VGPR excess threshold 50%, VGPRExcessLimit: 256 -> 128
; THRESH50: VGPRCriticalLimit = 21, VGPRExcessLimit = 128

; THRESH30: Applied VGPR excess threshold 30%, VGPRExcessLimit: 256 -> 77
; THRESH30: VGPRCriticalLimit = 21, VGPRExcessLimit = 77

; THRESH10: Applied VGPR excess threshold 10%, VGPRExcessLimit: 256 -> 26
; THRESH10: VGPRCriticalLimit = 21, VGPRExcessLimit = 26

; At 5%, VGPRExcessLimit (13) < VGPRCriticalLimit (21), so critical limit is adjusted
; THRESH5: Applied VGPR excess threshold 5%, VGPRExcessLimit: 256 -> 13
; THRESH5: VGPRCriticalLimit = 13, VGPRExcessLimit = 13

define amdgpu_kernel void @test_threshold(ptr addrspace(3) nocapture readonly %arg, ptr addrspace(1) nocapture %arg1) #0 {
bb:
  %tmp = getelementptr inbounds float, ptr addrspace(3) %arg, i32 1
  %tmp2 = load float, ptr addrspace(3) %tmp, align 4
  %tmp3 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 2
  %tmp4 = load float, ptr addrspace(3) %tmp3, align 4
  %tmp5 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 3
  %tmp6 = load float, ptr addrspace(3) %tmp5, align 4
  %tmp7 = tail call float @llvm.fmuladd.f32(float %tmp2, float %tmp4, float %tmp6)
  %tmp8 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 5
  %tmp9 = load float, ptr addrspace(3) %tmp8, align 4
  %tmp10 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 6
  %tmp11 = load float, ptr addrspace(3) %tmp10, align 4
  %tmp12 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 7
  %tmp13 = load float, ptr addrspace(3) %tmp12, align 4
  %tmp14 = tail call float @llvm.fmuladd.f32(float %tmp9, float %tmp11, float %tmp13)
  %tmp15 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 9
  %tmp16 = load float, ptr addrspace(3) %tmp15, align 4
  %tmp17 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 10
  %tmp18 = load float, ptr addrspace(3) %tmp17, align 4
  %tmp19 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 11
  %tmp20 = load float, ptr addrspace(3) %tmp19, align 4
  %tmp21 = tail call float @llvm.fmuladd.f32(float %tmp16, float %tmp18, float %tmp20)
  %tmp70 = getelementptr inbounds float, ptr addrspace(1) %arg1, i64 1
  store float %tmp7, ptr addrspace(1) %tmp70, align 4
  %tmp71 = getelementptr inbounds float, ptr addrspace(1) %arg1, i64 2
  store float %tmp14, ptr addrspace(1) %tmp71, align 4
  %tmp72 = getelementptr inbounds float, ptr addrspace(1) %arg1, i64 3
  store float %tmp21, ptr addrspace(1) %tmp72, align 4
  ret void
}

declare float @llvm.fmuladd.f32(float, float, float) #1

attributes #0 = { "amdgpu-flat-work-group-size"="1,256" }
attributes #1 = { nounwind readnone }
