// RUN: env SQTT_SCOPE_CU=-1 SQTT_SCOPE_SIMD=-1 SQTT_MEM_BARRIER=none \
// RUN:   SQTT_SHADER_CLOCK_BITS=12 %clang --target=%host-triple -x hip \
// RUN:   --offload-device-only \
// RUN:   -nogpuinc -nogpulib --offload-arch=gfx1200 -O2 \
// RUN:   -DAMD_SQTT_MARKER_ENABLE=1 -I%sqtt-marker-include \
// RUN:   -fpass-plugin=%sqtt-marker-plugin -S %s -o - | \
// RUN:   %FileCheck %s --check-prefix=GFX1200
// RUN: env SQTT_SCOPE_CU=-1 SQTT_SCOPE_SIMD=-1 SQTT_MEM_BARRIER=none \
// RUN:   SQTT_SHADER_CLOCK_BITS=12 %clang --target=%host-triple -x hip \
// RUN:   --offload-device-only \
// RUN:   -nogpuinc -nogpulib --offload-arch=gfx1250 -O2 \
// RUN:   -DAMD_SQTT_MARKER_ENABLE=1 -I%sqtt-marker-include \
// RUN:   -fpass-plugin=%sqtt-marker-plugin -S %s -o - | \
// RUN:   %FileCheck %s --check-prefix=GFX1250
// REQUIRES: amdgpu-registered-target, sqtt-marker-has-clang

#include <amd_sqtt_marker/sqtt_marker.h>

extern "C" __attribute__((global)) void clock_kernel() {
  sqtt_marker_enter("scope");
  sqtt_marker_exit("scope");
}

// GFX1200: s_getreg_b32 {{.*}}hwreg(HW_REG_SHADER_CYCLES_LO, 4, 12)
// GFX1200: s_mov_b32 m0,
// GFX1200-NEXT: s_nop 0
// GFX1200-NEXT: s_ttracedata
// GFX1200-NOT: s_ttracedata_imm
// GFX1200: .asciz "M:shader_clock_bits=12;shader_clock_shift=4

// GFX1250: s_get_shader_cycles_u64
// GFX1250-NOT: HW_REG_SHADER_CYCLES_LO
// GFX1250: s_mov_b32 m0,
// GFX1250-NEXT: s_nop 0
// GFX1250-NEXT: s_ttracedata
// GFX1250-NOT: s_ttracedata_imm
