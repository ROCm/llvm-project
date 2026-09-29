; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 < %s | FileCheck %s --check-prefixes=CHECK,SKIP
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -amdgpu-waitcnt-invariant-lds-skip-dma=false < %s | FileCheck %s --check-prefixes=CHECK,NOSKIP

; LDS loads through a pointer without alias information must normally wait for
; any in-flight LDS DMA. A load marked !invariant.load reads memory that does not
; change, so it cannot observe the DMA and SIInsertWaitcnts does not wait.

@lds.dma = internal addrspace(3) global [64 x float] poison, align 16

declare void @llvm.amdgcn.raw.buffer.load.lds(<4 x i32> %rsrc, ptr addrspace(3) nocapture, i32 %size, i32 %voffset, i32 %soffset, i32 %offset, i32 %aux)

; CHECK-LABEL: invariant_lds_load_after_dma:
; CHECK:       buffer_load_dword v{{[0-9]+}}, s[{{[0-9]+:[0-9]+}}], 0 offen lds
; SKIP-NOT:    s_waitcnt vmcnt
; NOSKIP:      s_waitcnt vmcnt(0)
; CHECK:       ds_read_b32
define amdgpu_kernel void @invariant_lds_load_after_dma(<4 x i32> %rsrc, i32 %voff, ptr addrspace(3) %q, ptr addrspace(1) %out) {
  call void @llvm.amdgcn.raw.buffer.load.lds(<4 x i32> %rsrc, ptr addrspace(3) @lds.dma, i32 4, i32 %voff, i32 0, i32 0, i32 0)
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %p = getelementptr float, ptr addrspace(3) %q, i32 %tid
  %v = load float, ptr addrspace(3) %p, align 4, !invariant.load !0
  store float %v, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK-LABEL: plain_lds_load_after_dma:
; CHECK:       buffer_load_dword v{{[0-9]+}}, s[{{[0-9]+:[0-9]+}}], 0 offen lds
; CHECK:       s_waitcnt vmcnt(0)
; CHECK:       ds_read_b32
define amdgpu_kernel void @plain_lds_load_after_dma(<4 x i32> %rsrc, i32 %voff, ptr addrspace(3) %q, ptr addrspace(1) %out) {
  call void @llvm.amdgcn.raw.buffer.load.lds(<4 x i32> %rsrc, ptr addrspace(3) @lds.dma, i32 4, i32 %voff, i32 0, i32 0, i32 0)
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %p = getelementptr float, ptr addrspace(3) %q, i32 %tid
  %v = load float, ptr addrspace(3) %p, align 4
  store float %v, ptr addrspace(1) %out, align 4
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()

!0 = !{}
