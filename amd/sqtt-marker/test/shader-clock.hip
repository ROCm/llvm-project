// RUN: %clang --target=%host-triple -x hip \
// RUN:   --offload-device-only \
// RUN:   -nogpuinc -nogpulib --offload-arch=gfx1200 -O2 \
// RUN:   -DAMD_SQTT_MARKER_ENABLE=1 -I%sqtt-marker-include \
// RUN:   -fpass-plugin=%sqtt-marker-plugin \
// RUN:   -Xclang -mllvm -Xclang -sqtt-marker-scope-cu=-1 \
// RUN:   -Xclang -mllvm -Xclang -sqtt-marker-scope-simd=-1 \
// RUN:   -Xclang -mllvm -Xclang -sqtt-marker-mem-barrier=none \
// RUN:   -Xclang -mllvm -Xclang -sqtt-marker-shader-clock-bits=12 \
// RUN:   -Xclang -mllvm -Xclang -sqtt-marker-shader-clock-shift=5 \
// RUN:   -S %s -o - | \
// RUN:   %FileCheck %s --check-prefix=GFX1200
// RUN: %clang --target=%host-triple -x hip \
// RUN:   --offload-device-only \
// RUN:   -nogpuinc -nogpulib --offload-arch=gfx1250 -O2 \
// RUN:   -DAMD_SQTT_MARKER_ENABLE=1 -I%sqtt-marker-include \
// RUN:   -fpass-plugin=%sqtt-marker-plugin \
// RUN:   -Xclang -mllvm -Xclang -sqtt-marker-scope-cu=-1 \
// RUN:   -Xclang -mllvm -Xclang -sqtt-marker-scope-simd=-1 \
// RUN:   -Xclang -mllvm -Xclang -sqtt-marker-mem-barrier=none \
// RUN:   -Xclang -mllvm -Xclang -sqtt-marker-shader-clock-bits=12 \
// RUN:   -S %s -o - | \
// RUN:   %FileCheck %s --check-prefix=GFX1250
// REQUIRES: amdgpu-registered-target, sqtt-marker-has-clang

#include <amd_sqtt_marker/sqtt_marker.h>

extern "C" __attribute__((global)) void clock_kernel() {
  sqtt_marker_enter("scope");
  sqtt_marker_exit("scope");
}

// GFX1200: s_getreg_b32 {{.*}}hwreg(HW_REG_SHADER_CYCLES_LO, 5, 12)
// GFX1200: s_mov_b32 m0,
// GFX1200-NEXT: s_nop 3
// GFX1200-NEXT: s_ttracedata
// GFX1200-NOT: s_ttracedata_imm
// GFX1200: .asciz "M:shader_clock_bits=12;shader_clock_shift=5

// GFX1250: s_get_shader_cycles_u64
// GFX1250-NOT: HW_REG_SHADER_CYCLES_LO
// GFX1250: s_mov_b32 m0,
// GFX1250-NEXT: s_nop 3
// GFX1250-NEXT: s_ttracedata
// GFX1250-NOT: s_ttracedata_imm
