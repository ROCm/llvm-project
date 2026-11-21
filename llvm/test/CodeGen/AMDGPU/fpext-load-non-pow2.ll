; RUN: llc -mtriple=amdgpu9.00-amd-amdhsa -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=amdgpu9.42-amd-amdhsa -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=amdgpu11.00-amd-amdhsa -verify-machineinstrs < %s | FileCheck %s

; These vector extending loads must expand to separate loads and conversions.

define amdgpu_kernel void @local_v5f16_to_v5f32(ptr addrspace(3) %src, ptr addrspace(1) %dst) {
; CHECK-LABEL: local_v5f16_to_v5f32:
; CHECK-COUNT-5: v_cvt_f32_f16
; CHECK: s_endpgm
  %val = load <5 x half>, ptr addrspace(3) %src, align 2
  %ext = fpext <5 x half> %val to <5 x float>
  store <5 x float> %ext, ptr addrspace(1) %dst, align 4
  ret void
}

define amdgpu_kernel void @local_v6f16_to_v6f32(ptr addrspace(3) %src, ptr addrspace(1) %dst) {
; CHECK-LABEL: local_v6f16_to_v6f32:
; CHECK-COUNT-6: v_cvt_f32_f16
; CHECK: s_endpgm
  %val = load <6 x half>, ptr addrspace(3) %src, align 2
  %ext = fpext <6 x half> %val to <6 x float>
  store <6 x float> %ext, ptr addrspace(1) %dst, align 4
  ret void
}

define amdgpu_kernel void @local_v7f16_to_v7f32(ptr addrspace(3) %src, ptr addrspace(1) %dst) {
; CHECK-LABEL: local_v7f16_to_v7f32:
; CHECK-COUNT-7: v_cvt_f32_f16
; CHECK: s_endpgm
  %val = load <7 x half>, ptr addrspace(3) %src, align 2
  %ext = fpext <7 x half> %val to <7 x float>
  store <7 x float> %ext, ptr addrspace(1) %dst, align 4
  ret void
}

define amdgpu_kernel void @global_v5f16_to_v5f32(ptr addrspace(1) %src, ptr addrspace(1) %dst) {
; CHECK-LABEL: global_v5f16_to_v5f32:
; CHECK-COUNT-5: v_cvt_f32_f16
; CHECK: s_endpgm
  %val = load <5 x half>, ptr addrspace(1) %src, align 2
  %ext = fpext <5 x half> %val to <5 x float>
  store <5 x float> %ext, ptr addrspace(1) %dst, align 4
  ret void
}

define amdgpu_kernel void @global_v6f16_to_v6f32(ptr addrspace(1) %src, ptr addrspace(1) %dst) {
; CHECK-LABEL: global_v6f16_to_v6f32:
; CHECK-COUNT-6: v_cvt_f32_f16
; CHECK: s_endpgm
  %val = load <6 x half>, ptr addrspace(1) %src, align 2
  %ext = fpext <6 x half> %val to <6 x float>
  store <6 x float> %ext, ptr addrspace(1) %dst, align 4
  ret void
}

define amdgpu_kernel void @global_v7f16_to_v7f32(ptr addrspace(1) %src, ptr addrspace(1) %dst) {
; CHECK-LABEL: global_v7f16_to_v7f32:
; CHECK-COUNT-7: v_cvt_f32_f16
; CHECK: s_endpgm
  %val = load <7 x half>, ptr addrspace(1) %src, align 2
  %ext = fpext <7 x half> %val to <7 x float>
  store <7 x float> %ext, ptr addrspace(1) %dst, align 4
  ret void
}
