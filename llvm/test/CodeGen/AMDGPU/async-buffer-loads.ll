; RUN: llc -global-isel=0 -mtriple=amdgcn -mcpu=gfx900 < %s | FileCheck %s
; RUN: llc -global-isel=1 -mtriple=amdgcn -mcpu=gfx900 < %s | FileCheck %s

define float @raw.buffer.load(<4 x i32> inreg %rsrc, ptr addrspace(3) inreg %lds) {
; CHECK-LABEL: raw.buffer.load:
; CHECK:    s_waitcnt vmcnt(2)
main_body:
  call void @llvm.amdgcn.raw.buffer.load.async.lds(<4 x i32> %rsrc, ptr addrspace(3) %lds, i32 4, i32 0, i32 0, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.raw.buffer.load.async.lds(<4 x i32> %rsrc, ptr addrspace(3) %lds, i32 4, i32 0, i32 0, i32 4, i32 1)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.raw.buffer.load.async.lds(<4 x i32> %rsrc, ptr addrspace(3) %lds, i32 4, i32 0, i32 0, i32 8, i32 2)
  call void @llvm.amdgcn.wait.asyncmark(i16 1)
  %res = load float, ptr addrspace(3) %lds
  ret float %res
}

define float @raw.ptr.buffer.load(ptr addrspace(8) inreg %rsrc, ptr addrspace(3) inreg %lds) {
; CHECK-LABEL: raw.ptr.buffer.load:
; CHECK:    s_waitcnt vmcnt(2)
main_body:
  call void @llvm.amdgcn.raw.ptr.buffer.load.async.lds(ptr addrspace(8) %rsrc, ptr addrspace(3) %lds, i32 4, i32 0, i32 0, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.raw.ptr.buffer.load.async.lds(ptr addrspace(8) %rsrc, ptr addrspace(3) %lds, i32 4, i32 0, i32 0, i32 4, i32 1)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.raw.ptr.buffer.load.async.lds(ptr addrspace(8) %rsrc, ptr addrspace(3) %lds, i32 4, i32 0, i32 0, i32 8, i32 2)
  call void @llvm.amdgcn.wait.asyncmark(i16 1)
  %res = load float, ptr addrspace(3) %lds
  ret float %res
}

define float @struct.buffer.load(<4 x i32> inreg %rsrc, ptr addrspace(3) inreg %lds) {
; CHECK-LABEL: struct.buffer.load:
; CHECK:    s_waitcnt vmcnt(2)
main_body:
  call void @llvm.amdgcn.struct.buffer.load.async.lds(<4 x i32> %rsrc, ptr addrspace(3) %lds, i32 4, i32 8, i32 0, i32 0, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.struct.buffer.load.async.lds(<4 x i32> %rsrc, ptr addrspace(3) %lds, i32 4, i32 8, i32 0, i32 0, i32 4, i32 1)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.struct.buffer.load.async.lds(<4 x i32> %rsrc, ptr addrspace(3) %lds, i32 4, i32 8, i32 0, i32 0, i32 8, i32 2)
  call void @llvm.amdgcn.wait.asyncmark(i16 1)
  %res = load float, ptr addrspace(3) %lds
  ret float %res
}

define float @struct.ptr.buffer.load(ptr addrspace(8) inreg %rsrc, ptr addrspace(3) inreg %lds) {
; CHECK-LABEL: struct.ptr.buffer.load:
; CHECK:    s_waitcnt vmcnt(2)
main_body:
  call void @llvm.amdgcn.struct.ptr.buffer.load.async.lds(ptr addrspace(8) %rsrc, ptr addrspace(3) %lds, i32 4, i32 8, i32 0, i32 0, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.struct.ptr.buffer.load.async.lds(ptr addrspace(8) %rsrc, ptr addrspace(3) %lds, i32 4, i32 8, i32 0, i32 0, i32 4, i32 1)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.struct.ptr.buffer.load.async.lds(ptr addrspace(8) %rsrc, ptr addrspace(3) %lds, i32 4, i32 8, i32 0, i32 0, i32 8, i32 2)
  call void @llvm.amdgcn.wait.asyncmark(i16 1)
  %res = load float, ptr addrspace(3) %lds
  ret float %res
}
