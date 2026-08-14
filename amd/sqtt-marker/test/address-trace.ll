; RUN: env SQTT_SCOPE_CU=-1 SQTT_SCOPE_SIMD=-1 SQTT_MEM_BARRIER=none \
; RUN:   SQTT_TRACE_ADDRESSES=memory,lds %opt \
; RUN:   -load-pass-plugin=%sqtt-marker-plugin -passes='default<O0>' \
; RUN:   -S %s -o - | %FileCheck %s
; REQUIRES: amdgpu-registered-target

target triple = "amdgcn-amd-amdhsa"

define amdgpu_kernel void @kernel(ptr addrspace(1) %global,
                                  ptr addrspace(3) %lds) #0 {
entry:
  %a = load i32, ptr addrspace(1) %global
  store i32 %a, ptr addrspace(1) %global
  %b = load i32, ptr addrspace(3) %lds
  store i32 %b, ptr addrspace(3) %lds
  %c = atomicrmw add ptr addrspace(1) %global, i32 1 seq_cst
  %d = atomicrmw add ptr addrspace(3) %lds, i32 1 seq_cst
  ret void
}

attributes #0 = { "target-cpu"="gfx1200" }

; CHECK: c"K:kernel\0AW:32\0AP:1:addr_trace_load\0AR:1:extra_payload_count=66\0AP:2:addr_trace_store\0AR:2:extra_payload_count=66\0AP:3:addr_trace_lds_load\0AR:3:extra_payload_count=34\0AP:4:addr_trace_lds_store\0AR:4:extra_payload_count=34\0AP:5:addr_trace_atomic\0AR:5:extra_payload_count=66\0AP:6:addr_trace_lds_atomic\0AR:6:extra_payload_count=34\0A\00"
; CHECK: ptrtoint ptr addrspace(1) %global to i64
; CHECK: s_mov_b32 m0, exec_lo
; CHECK: s_mov_b32 m0, exec_hi
; CHECK: call i32 @llvm.amdgcn.readlane.i32
; CHECK: ptrtoint ptr addrspace(3) %lds to i32
