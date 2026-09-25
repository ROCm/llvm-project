; REQUIRES: comgr-has-transpiler, comgr-has-llc

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %S/vop-math.s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=vop_math,vop3_math \
; RUN:   | %llc -mtriple=amdgpu9.42-amd-amdhsa -mcpu=gfx942 -o - \
; RUN:   | %FileCheck %s

; CHECK: v_rcp_iflag_f32
