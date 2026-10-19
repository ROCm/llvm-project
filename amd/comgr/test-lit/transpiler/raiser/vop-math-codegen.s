; REQUIRES: comgr-has-transpiler, comgr-has-llc

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %S/vop-math.s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %transpile_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=vop_math,vop3_math \
; RUN:   | %llc -mtriple=amdgpu9.42-amd-amdhsa -mcpu=gfx942 -o - \
; RUN:   | %FileCheck %s
; RUN: %transpile_cli %t.hsaco --target-isa=gfx1250 \
; RUN:   --emit-ir=refuse_tanh \
; RUN:   | %llc -mtriple=amdgpu12.50-amd-amdhsa -mcpu=gfx1250 -o - \
; RUN:   | %FileCheck %s --check-prefix=SUPPORTED-TANH

; CHECK-LABEL: vop_math:
; CHECK-DAG: v_rcp_iflag_f32
; CHECK-DAG: v_trunc_f64
; CHECK-DAG: v_ceil_f64
; CHECK-DAG: v_rndne_f64
; CHECK-DAG: v_floor_f64
; CHECK-DAG: v_rcp_f64
; CHECK-DAG: v_rsq_f64
; CHECK-LABEL: vop3_math:
; CHECK-DAG: v_ldexp_f64
; CHECK-DAG: v_rcp_f64
; CHECK-DAG: v_rndne_f64
; CHECK-DAG: v_{{fma|fmac|fmamk}}_f64
; CHECK-DAG: v_{{fma|fmac|fmamk}}_f64
; CHECK-DAG: v_{{fma|fmac|fmamk}}_f64
; CHECK-DAG: v_{{fma|fmac|fmamk}}_f32
; CHECK-DAG: v_{{fma|fmac|fmamk}}_f32
; CHECK-DAG: v_{{fma|fmac|fmamk}}_f32
; SUPPORTED-TANH: v_tanh_f32
