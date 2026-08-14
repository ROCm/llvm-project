; RUN: env SQTT_SCOPE_CU=-1 SQTT_SCOPE_SIMD=-1 SQTT_MEM_BARRIER=none \
; RUN:   SQTT_INSTRUMENT_FUNCTIONS=3 SQTT_INSTRUMENT_BARRIERS=1 \
; RUN:   SQTT_INSTRUMENT_MEMORY=1:0 %opt -load-pass-plugin=%sqtt-marker-plugin \
; RUN:   -passes='default<O2>' -S %s -o - | %FileCheck %s --check-prefix=AUTO
; RUN: env SQTT_SCOPE_CU=-1 SQTT_SCOPE_SIMD=-1 SQTT_INSTRUMENT_BARRIERS=1 \
; RUN:   %opt -load-pass-plugin=%sqtt-marker-plugin -passes='default<O0>' \
; RUN:   -S %s -o - | %FileCheck %s --check-prefix=FENCE
; RUN: env SQTT_SCOPE_CU=-1 SQTT_SCOPE_SIMD=-1 SQTT_MEM_BARRIER=asm \
; RUN:   SQTT_INSTRUMENT_BARRIERS=1 %opt -load-pass-plugin=%sqtt-marker-plugin \
; RUN:   -passes='default<O0>' -S %s -o - | %FileCheck %s --check-prefix=ASM
; RUN: env SQTT_SCOPE_CU=-1 SQTT_SCOPE_SIMD=-1 SQTT_MEM_BARRIER=none \
; RUN:   SQTT_INSTRUMENT_BARRIERS=1 %opt -load-pass-plugin=%sqtt-marker-plugin \
; RUN:   -passes='default<O0>' -S %s -o - | %FileCheck %s --check-prefix=NONE
; REQUIRES: amdgpu-registered-target

target triple = "amdgcn-amd-amdhsa"

declare void @llvm.amdgcn.s.barrier()

define internal i32 @work(ptr addrspace(1) %pointer) #0 {
entry:
  %a = load i32, ptr addrspace(1) %pointer
  %b = add i32 %a, 1
  %c = mul i32 %b, 3
  %d = xor i32 %c, 7
  store i32 %d, ptr addrspace(1) %pointer
  ret i32 %d
}

define amdgpu_kernel void @kernel(ptr addrspace(1) %pointer) #0 {
entry:
  %value = call i32 @work(ptr addrspace(1) %pointer)
  call void @llvm.amdgcn.s.barrier()
  store i32 %value, ptr addrspace(1) %pointer
  ret void
}

attributes #0 = { "target-cpu"="gfx1200" }

; AUTO: c"F:1:work\0AK:kernel\0AP:2:barrier_signal\0AP:3:barrier_wait\0AP:4:barrier\0AP:5:vmem_load\0AP:6:vmem_store\0A\00"
; AUTO: call void @llvm.amdgcn.s.ttracedata.imm(i16 6)
; AUTO: call void @llvm.amdgcn.s.ttracedata.imm(i16 20)
; AUTO: call void @llvm.amdgcn.s.ttracedata.imm(i16 24)

; FENCE: fence syncscope("workgroup") acq_rel
; FENCE: !"amdgpu-synchronize-as", !"local"
; FENCE-NOT: asm sideeffect "", "~{memory}"

; ASM: call void asm sideeffect "", "~{memory}"()
; ASM-NOT: fence syncscope("workgroup") acq_rel

; NONE: call void @llvm.amdgcn.s.ttracedata.imm
; NONE-NOT: fence syncscope("workgroup") acq_rel
; NONE-NOT: asm sideeffect "", "~{memory}"
