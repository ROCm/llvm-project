; RUN: llvm-as < %s | llvm-dis | FileCheck %s

; A scope value produced by a DMA scope intrinsic flows into make.available and
; make.visible within the same function.

; CHECK-LABEL: define void @lds_dma_scope()
; CHECK: %scope = call target("amdgcn.scope") @llvm.amdgcn.scope.lds.dma()
; CHECK: call void @llvm.amdgcn.make.available(target("amdgcn.scope") %scope)
; CHECK: call void @llvm.amdgcn.make.visible(target("amdgcn.scope") %scope)
define void @lds_dma_scope() {
  %scope = call target("amdgcn.scope") @llvm.amdgcn.scope.lds.dma()
  call void @llvm.amdgcn.make.available(target("amdgcn.scope") %scope)
  call void @llvm.amdgcn.make.visible(target("amdgcn.scope") %scope)
  ret void
}

; CHECK-LABEL: define void @tensor_dma_scope()
; CHECK: %scope = call target("amdgcn.scope") @llvm.amdgcn.scope.tensor.dma()
define void @tensor_dma_scope() {
  %scope = call target("amdgcn.scope") @llvm.amdgcn.scope.tensor.dma()
  call void @llvm.amdgcn.make.available(target("amdgcn.scope") %scope)
  ret void
}
