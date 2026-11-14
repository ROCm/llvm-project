// RUN: %clang --target=%host-triple -x hip --offload-device-only \
// RUN:   -nogpuinc -nogpulib --offload-arch=gfx1200 -O2 \
// RUN:   -DAMD_SQTT_MARKER_ENABLE=1 -I%sqtt-marker-include \
// RUN:   -fpass-plugin=%sqtt-marker-plugin \
// RUN:   -Xclang -mllvm -Xclang -sqtt-marker-scope-cu=2 \
// RUN:   -Xclang -mllvm -Xclang -sqtt-marker-scope-simd=-1 \
// RUN:   -Xclang -mllvm -Xclang -sqtt-marker-mem-barrier=fence \
// RUN:   -S -emit-llvm %s -o - | %FileCheck %s --check-prefix=IR
// RUN: %clang --target=%host-triple -x hip --offload-device-only \
// RUN:   -nogpuinc -nogpulib --offload-arch=gfx1200 -O2 \
// RUN:   -DAMD_SQTT_MARKER_ENABLE=1 -I%sqtt-marker-include \
// RUN:   -fpass-plugin=%sqtt-marker-plugin \
// RUN:   -Xclang -mllvm -Xclang -sqtt-marker-scope-cu=2 \
// RUN:   -Xclang -mllvm -Xclang -sqtt-marker-scope-simd=-1 \
// RUN:   -Xclang -mllvm -Xclang -sqtt-marker-mem-barrier=fence \
// RUN:   -S %s -o - | %FileCheck %s --check-prefix=ASM
// RUN: %clang --target=%host-triple -x hip --offload-device-only \
// RUN:   -nogpuinc -nogpulib --offload-arch=gfx1200 -O2 \
// RUN:   -DAMD_SQTT_MARKER_ENABLE=1 -I%sqtt-marker-include \
// RUN:   -fpass-plugin=%sqtt-marker-plugin \
// RUN:   -Xclang -mllvm -Xclang -sqtt-marker-scope-cu=2 \
// RUN:   -Xclang -mllvm -Xclang -sqtt-marker-scope-simd=-1 \
// RUN:   -Xclang -mllvm -Xclang -sqtt-marker-mem-barrier=fence \
// RUN:   -Xclang -mllvm -Xclang -amdgpu-expert-scheduling-mode \
// RUN:   -S %s -o - | %FileCheck %s --check-prefix=EXPERT
// RUN: %clang --target=%host-triple -x hip --offload-device-only \
// RUN:   -nogpuinc -nogpulib --offload-arch=gfx90a -O2 \
// RUN:   -DAMD_SQTT_MARKER_ENABLE=1 -I%sqtt-marker-include \
// RUN:   -fpass-plugin=%sqtt-marker-plugin \
// RUN:   -Xclang -mllvm -Xclang -sqtt-marker-scope-cu=2 \
// RUN:   -Xclang -mllvm -Xclang -sqtt-marker-scope-simd=-1 \
// RUN:   -Xclang -mllvm -Xclang -sqtt-marker-mem-barrier=fence \
// RUN:   -S %s -o - | %FileCheck %s --check-prefix=M0
// REQUIRES: amdgpu-registered-target, sqtt-marker-has-clang

#include <amd_sqtt_marker/sqtt_marker.h>

extern "C" __attribute__((global)) void scope_filter(unsigned int *output,
                                                     unsigned int input) {
  unsigned int lane = __builtin_amdgcn_workitem_id_x();
  if (lane & 1) {
    input += lane;
    sqtt_marker_point("first");
    input = input * 17 + 3;
    sqtt_marker_point("second");
    sqtt_marker_data("payload", input);
    sqtt_marker_point_id(100);
  }
  output[lane] = input;
}

// Scope filtering must not create a CFG diamond for every marker. Those
// diamonds extend live ranges through complex kernels and inflate VGPR usage.
// IR-LABEL: define {{.*}} @scope_filter(
// IR-COUNT-1: call i32 @llvm.amdgcn.s.getreg
// IR-NOT: sqtt.trace
// IR-COUNT-5: call {{.*}} asm sideeffect "s_cmp_lg_u32 ${{[01]}}, 0

// The scope predicate and branch must remain scalar in final codegen.
// ASM-LABEL: scope_filter:
// ASM: s_getreg_b32
// ASM: s_cmp_lg_u32
// ASM-NEXT: s_cbranch_scc0 .Lsqtt_skip_
// ASM-NEXT: s_ttracedata_imm
// ASM: s_cmp_lg_u32
// ASM-NEXT: s_cbranch_scc0 .Lsqtt_skip_
// ASM-NEXT: s_ttracedata_imm
// ASM: v_readfirstlane_b32
// ASM: s_cmp_lg_u32
// ASM-NEXT: s_cbranch_scc0 .Lsqtt_skip_
// ASM-NEXT: s_ttracedata_imm
// ASM: s_cmp_lg_u32
// ASM-NEXT: s_cbranch_scc0 .Lsqtt_skip_
// ASM-NEXT: s_mov_b32 m0,
// ASM-NEXT: s_nop 3
// ASM-NEXT: s_ttracedata

// Expert scheduling removes hardware-managed data hazards on gfx12. Keep the
// explicit M0 latency instruction inside the indivisible marker sequence.
// EXPERT-LABEL: scope_filter:
// EXPERT: s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 0, 2), 2
// EXPERT: v_readfirstlane_b32
// EXPERT: s_cmp_lg_u32
// EXPERT-NEXT: s_cbranch_scc0 .Lsqtt_skip_
// EXPERT-NEXT: s_ttracedata_imm
// EXPERT: s_cmp_lg_u32
// EXPERT-NEXT: s_cbranch_scc0 .Lsqtt_skip_
// EXPERT-NEXT: s_mov_b32 m0,
// EXPERT-NEXT: s_nop 3
// EXPERT-NEXT: s_ttracedata

// M0-LABEL: scope_filter:
// M0: s_cmp_lg_u32
// M0-NEXT: s_cbranch_scc0 .Lsqtt_skip_
// M0-NEXT: s_mov_b32 m0,
// M0-NEXT: s_nop 0
// M0-NEXT: s_ttracedata
