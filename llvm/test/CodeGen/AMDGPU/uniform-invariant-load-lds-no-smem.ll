; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 < %s | FileCheck %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 < %s | FileCheck %s

; A uniform load marked !invariant.load must only be selected as a scalar (SMEM)
; load for address spaces SMEM can access. SMEM cannot read LDS or scratch; this
; used to emit an s_load_dword from the LDS offset (i.e. from global memory).

; CHECK-LABEL: uniform_invariant_lds_load:
; CHECK:       ds_read_b32
; CHECK:       s_endpgm
define amdgpu_kernel void @uniform_invariant_lds_load(i32 %i, ptr addrspace(3) %q, ptr addrspace(1) %out) {
  %p = getelementptr float, ptr addrspace(3) %q, i32 %i
  %v = load float, ptr addrspace(3) %p, align 4, !invariant.load !0
  store float %v, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK-LABEL: uniform_invariant_lds_load_x4:
; CHECK:       ds_read_b128
; CHECK:       s_endpgm
define amdgpu_kernel void @uniform_invariant_lds_load_x4(i32 %i, ptr addrspace(3) %q, ptr addrspace(1) %out) {
  %p = getelementptr <4 x i32>, ptr addrspace(3) %q, i32 %i
  %v = load <4 x i32>, ptr addrspace(3) %p, align 16, !invariant.load !0
  store <4 x i32> %v, ptr addrspace(1) %out, align 16
  ret void
}

; Uniform invariant global loads are still scalarized.
; CHECK-LABEL: uniform_invariant_global_load:
; CHECK:       s_load_dword
; CHECK-NOT:   global_load_dword
; CHECK:       s_endpgm
define amdgpu_kernel void @uniform_invariant_global_load(ptr addrspace(1) %g, ptr addrspace(1) %out) {
  %v = load i32, ptr addrspace(1) %g, align 4, !invariant.load !0
  store i32 %v, ptr addrspace(1) %out, align 4
  ret void
}

; Typical frontend pattern: fill LDS, barrier, then every lane reads the same
; (broadcast) element. The broadcast read must be a DS load; the buggy selection
; emitted `s_load_dword sX, s[lo:hi], 0x0` with hi = 0 after the barrier, which
; faults on the hardware (reads global address ~0x14).
; CHECK-LABEL: write_then_read:
; CHECK:       s_barrier
; CHECK-NOT:   s_load_dword
; CHECK:       ds_read_b32
; CHECK:       s_endpgm
define amdgpu_kernel void @write_then_read(i32 %idx, ptr addrspace(3) %lds, ptr addrspace(1) %out) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %w = getelementptr inbounds i32, ptr addrspace(3) %lds, i32 %tid
  %val = add i32 %tid, 1000
  store i32 %val, ptr addrspace(3) %w, align 4
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %p = getelementptr inbounds i32, ptr addrspace(3) %lds, i32 %idx
  %v = load i32, ptr addrspace(3) %p, align 4, !invariant.load !0
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %tid
  store i32 %v, ptr addrspace(1) %o, align 4
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
declare void @llvm.amdgcn.s.barrier()

!0 = !{}
