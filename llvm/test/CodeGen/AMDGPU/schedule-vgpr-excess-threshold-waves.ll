; REQUIRES: asserts

; Test the -amdgpu-vgpr-excess-threshold-percent flag at different waves-per-eu
; occupancy targets. This tests the interaction between the threshold percentage
; and the occupancy-based VGPR limits. When the threshold reduces VGPRExcessLimit
; below VGPRCriticalLimit, the critical limit is also adjusted.

;-----------------------------------------------------------------------------
; Test with waves-per-eu=2 (128 VGPRs available per wave)
; Default VGPRCriticalLimit = 125, VGPRExcessLimit = 125
;-----------------------------------------------------------------------------
; RUN: llc -mtriple=amdgcn -mcpu=gfx906 -debug-only=machine-scheduler -filetype=null \
; RUN:   < %s 2>&1 | FileCheck -check-prefix=W2-DEFAULT %s

; RUN: llc -mtriple=amdgcn -mcpu=gfx906 -amdgpu-vgpr-excess-threshold-percent=90 \
; RUN:   -debug-only=machine-scheduler -filetype=null < %s 2>&1 \
; RUN:   | FileCheck -check-prefix=W2-T90 %s

; RUN: llc -mtriple=amdgcn -mcpu=gfx906 -amdgpu-vgpr-excess-threshold-percent=50 \
; RUN:   -debug-only=machine-scheduler -filetype=null < %s 2>&1 \
; RUN:   | FileCheck -check-prefix=W2-T50 %s

; RUN: llc -mtriple=amdgcn -mcpu=gfx906 -amdgpu-vgpr-excess-threshold-percent=10 \
; RUN:   -debug-only=machine-scheduler -filetype=null < %s 2>&1 \
; RUN:   | FileCheck -check-prefix=W2-T10 %s

;-----------------------------------------------------------------------------
; Test with waves-per-eu=4 (64 VGPRs available per wave)
; Default VGPRCriticalLimit = 61, VGPRExcessLimit = 61
;-----------------------------------------------------------------------------
; RUN: llc -mtriple=amdgcn -mcpu=gfx906 -amdgpu-vgpr-excess-threshold-percent=90 \
; RUN:   -debug-only=machine-scheduler -filetype=null < %s 2>&1 \
; RUN:   | FileCheck -check-prefix=W4-T90 %s

; RUN: llc -mtriple=amdgcn -mcpu=gfx906 -amdgpu-vgpr-excess-threshold-percent=50 \
; RUN:   -debug-only=machine-scheduler -filetype=null < %s 2>&1 \
; RUN:   | FileCheck -check-prefix=W4-T50 %s

; RUN: llc -mtriple=amdgcn -mcpu=gfx906 -amdgpu-vgpr-excess-threshold-percent=30 \
; RUN:   -debug-only=machine-scheduler -filetype=null < %s 2>&1 \
; RUN:   | FileCheck -check-prefix=W4-T30 %s

; RUN: llc -mtriple=amdgcn -mcpu=gfx906 -amdgpu-vgpr-excess-threshold-percent=10 \
; RUN:   -debug-only=machine-scheduler -filetype=null < %s 2>&1 \
; RUN:   | FileCheck -check-prefix=W4-T10 %s

;-----------------------------------------------------------------------------
; Test with waves-per-eu=8 (32 VGPRs available per wave)
; Default VGPRCriticalLimit = 29, VGPRExcessLimit = 29
;-----------------------------------------------------------------------------
; RUN: llc -mtriple=amdgcn -mcpu=gfx906 -amdgpu-vgpr-excess-threshold-percent=90 \
; RUN:   -debug-only=machine-scheduler -filetype=null < %s 2>&1 \
; RUN:   | FileCheck -check-prefix=W8-T90 %s

; RUN: llc -mtriple=amdgcn -mcpu=gfx906 -amdgpu-vgpr-excess-threshold-percent=70 \
; RUN:   -debug-only=machine-scheduler -filetype=null < %s 2>&1 \
; RUN:   | FileCheck -check-prefix=W8-T70 %s

; RUN: llc -mtriple=amdgcn -mcpu=gfx906 -amdgpu-vgpr-excess-threshold-percent=50 \
; RUN:   -debug-only=machine-scheduler -filetype=null < %s 2>&1 \
; RUN:   | FileCheck -check-prefix=W8-T50 %s

; RUN: llc -mtriple=amdgcn -mcpu=gfx906 -amdgpu-vgpr-excess-threshold-percent=10 \
; RUN:   -debug-only=machine-scheduler -filetype=null < %s 2>&1 \
; RUN:   | FileCheck -check-prefix=W8-T10 %s

; W2-DEFAULT: VGPRCriticalLimit = 125, VGPRExcessLimit = 125
; W2-DEFAULT-NOT: Applied VGPR excess threshold

; At 90%, VGPRExcessLimit (116) < VGPRCriticalLimit (125), so critical is adjusted
; W2-T90: Applied VGPR excess threshold 90%, VGPRExcessLimit: 128 -> 116
; W2-T90: VGPRCriticalLimit = 116, VGPRExcessLimit = 116

; At 50%, VGPRExcessLimit (64) < VGPRCriticalLimit (125), so critical is adjusted
; W2-T50: Applied VGPR excess threshold 50%, VGPRExcessLimit: 128 -> 64
; W2-T50: VGPRCriticalLimit = 64, VGPRExcessLimit = 64

; At 10%, VGPRExcessLimit (13) < VGPRCriticalLimit (125), so critical is adjusted
; W2-T10: Applied VGPR excess threshold 10%, VGPRExcessLimit: 128 -> 13
; W2-T10: VGPRCriticalLimit = 13, VGPRExcessLimit = 13

; waves=4 tests (64 VGPRs available, default critical = 61)
; At 90%, VGPRExcessLimit (58) < VGPRCriticalLimit (61), so critical is adjusted
; W4-T90-DAG: Applied VGPR excess threshold 90%, VGPRExcessLimit: 64 -> 58
; W4-T90-DAG: VGPRCriticalLimit = 58, VGPRExcessLimit = 58

; W4-T50-DAG: Applied VGPR excess threshold 50%, VGPRExcessLimit: 64 -> 32
; W4-T50-DAG: VGPRCriticalLimit = 32, VGPRExcessLimit = 32

; W4-T30-DAG: Applied VGPR excess threshold 30%, VGPRExcessLimit: 64 -> 20
; W4-T30-DAG: VGPRCriticalLimit = 20, VGPRExcessLimit = 20

; W4-T10-DAG: Applied VGPR excess threshold 10%, VGPRExcessLimit: 64 -> 7
; W4-T10-DAG: VGPRCriticalLimit = 7, VGPRExcessLimit = 7

; waves=8 tests (32 VGPRs available, default critical = 29)
; W8-T90-DAG: Applied VGPR excess threshold 90%, VGPRExcessLimit: 32 -> 29
; W8-T90-DAG: VGPRCriticalLimit = 29, VGPRExcessLimit = 29

; W8-T70-DAG: Applied VGPR excess threshold 70%, VGPRExcessLimit: 32 -> 23
; W8-T70-DAG: VGPRCriticalLimit = 23, VGPRExcessLimit = 23

; W8-T50-DAG: Applied VGPR excess threshold 50%, VGPRExcessLimit: 32 -> 16
; W8-T50-DAG: VGPRCriticalLimit = 16, VGPRExcessLimit = 16

; W8-T10-DAG: Applied VGPR excess threshold 10%, VGPRExcessLimit: 32 -> 4
; W8-T10-DAG: VGPRCriticalLimit = 4, VGPRExcessLimit = 4

; Test kernel targeting waves-per-eu=2
define amdgpu_kernel void @test_waves2(ptr addrspace(3) nocapture readonly %arg, ptr addrspace(1) nocapture %arg1) #0 {
bb:
  %tmp = getelementptr inbounds float, ptr addrspace(3) %arg, i32 1
  %tmp2 = load float, ptr addrspace(3) %tmp, align 4
  %tmp3 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 2
  %tmp4 = load float, ptr addrspace(3) %tmp3, align 4
  %tmp5 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 3
  %tmp6 = load float, ptr addrspace(3) %tmp5, align 4
  %tmp7 = tail call float @llvm.fmuladd.f32(float %tmp2, float %tmp4, float %tmp6)
  store float %tmp7, ptr addrspace(1) %arg1, align 4
  ret void
}

; Test kernel targeting waves-per-eu=4
define amdgpu_kernel void @test_waves4(ptr addrspace(3) nocapture readonly %arg, ptr addrspace(1) nocapture %arg1) #1 {
bb:
  %tmp = getelementptr inbounds float, ptr addrspace(3) %arg, i32 1
  %tmp2 = load float, ptr addrspace(3) %tmp, align 4
  %tmp3 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 2
  %tmp4 = load float, ptr addrspace(3) %tmp3, align 4
  %tmp7 = tail call float @llvm.fmuladd.f32(float %tmp2, float %tmp4, float %tmp2)
  store float %tmp7, ptr addrspace(1) %arg1, align 4
  ret void
}

; Test kernel targeting waves-per-eu=8
define amdgpu_kernel void @test_waves8(ptr addrspace(3) nocapture readonly %arg, ptr addrspace(1) nocapture %arg1) #2 {
bb:
  %tmp = getelementptr inbounds float, ptr addrspace(3) %arg, i32 1
  %tmp2 = load float, ptr addrspace(3) %tmp, align 4
  %tmp3 = getelementptr inbounds float, ptr addrspace(3) %arg, i32 2
  %tmp4 = load float, ptr addrspace(3) %tmp3, align 4
  %tmp7 = tail call float @llvm.fmuladd.f32(float %tmp2, float %tmp4, float %tmp2)
  store float %tmp7, ptr addrspace(1) %arg1, align 4
  ret void
}

declare float @llvm.fmuladd.f32(float, float, float) #3

attributes #0 = { "amdgpu-waves-per-eu"="2,2" "amdgpu-flat-work-group-size"="1,256" }
attributes #1 = { "amdgpu-waves-per-eu"="4,4" "amdgpu-flat-work-group-size"="1,256" }
attributes #2 = { "amdgpu-waves-per-eu"="8,8" "amdgpu-flat-work-group-size"="1,256" }
attributes #3 = { nounwind readnone }
