// RUN: %clang --target=%host-triple -x hip --offload-device-only \
// RUN:   -nogpuinc -nogpulib \
// RUN:   --offload-arch=gfx1200 -O2 -DAMD_SQTT_MARKER_ENABLE=1 \
// RUN:   -I%sqtt-marker-include -fpass-plugin=%sqtt-marker-plugin \
// RUN:   -Xclang -mllvm -Xclang -sqtt-marker-scope-cu=-1 \
// RUN:   -Xclang -mllvm -Xclang -sqtt-marker-scope-simd=-1 \
// RUN:   -Xclang -mllvm -Xclang -sqtt-marker-mem-barrier=none \
// RUN:   -S -emit-llvm %s -o - | %FileCheck %s --check-prefix=IR
// RUN: %clang --target=%host-triple -x hip --offload-device-only \
// RUN:   -nogpuinc -nogpulib \
// RUN:   --offload-arch=gfx1200 -O2 -DAMD_SQTT_MARKER_ENABLE=1 \
// RUN:   -I%sqtt-marker-include -fpass-plugin=%sqtt-marker-plugin \
// RUN:   -Xclang -mllvm -Xclang -sqtt-marker-scope-cu=-1 \
// RUN:   -Xclang -mllvm -Xclang -sqtt-marker-scope-simd=-1 \
// RUN:   -Xclang -mllvm -Xclang -sqtt-marker-mem-barrier=none \
// RUN:   -c %s -o %t.o
// RUN: %clang-offload-bundler -unbundle -type=o \
// RUN:   -targets=hipv4-amdgcn-amd-amdhsa--gfx1200 \
// RUN:   -input=%t.o -output=%t.co
// RUN: %llvm-objcopy --dump-section=.sqtt_funcmap=%t.map %t.co
// RUN: %python %S/Inputs/decode.py %t.map 6 11 12 16 42 1 | \
// RUN:   %FileCheck %s --check-prefix=DECODE
// RUN: %clang --target=%host-triple -x hip --offload-device-only \
// RUN:   -nogpuinc -nogpulib \
// RUN:   --offload-arch=gfx1200 -O2 -I%sqtt-marker-include \
// RUN:   -S -emit-llvm %s -o - | %FileCheck %s --check-prefix=DISABLED
// RUN: %clang --target=%host-triple -x hip --offload-device-only \
// RUN:   -nogpuinc -nogpulib \
// RUN:   --offload-arch=gfx90a -O2 -DAMD_SQTT_MARKER_ENABLE=1 \
// RUN:   -I%sqtt-marker-include -fpass-plugin=%sqtt-marker-plugin \
// RUN:   -Xclang -mllvm -Xclang -sqtt-marker-scope-cu=-1 \
// RUN:   -Xclang -mllvm -Xclang -sqtt-marker-scope-simd=-1 \
// RUN:   -Xclang -mllvm -Xclang -sqtt-marker-mem-barrier=none \
// RUN:   -S %s -o - | %FileCheck %s --check-prefix=GFX9
// RUN: %clang --target=%host-triple -x hip --offload-device-only \
// RUN:   -nogpuinc -nogpulib \
// RUN:   --offload-arch=gfx1200 -O2 -DAMD_SQTT_MARKER_ENABLE=1 \
// RUN:   -I%sqtt-marker-include -fpass-plugin=%sqtt-marker-plugin \
// RUN:   -Xclang -mllvm -Xclang -sqtt-marker-scope-cu=-1 \
// RUN:   -Xclang -mllvm -Xclang -sqtt-marker-scope-simd=-1 \
// RUN:   -Xclang -mllvm -Xclang -sqtt-marker-mem-barrier=fence \
// RUN:   -S %s -o - | %FileCheck %s --check-prefix=GFX12
// REQUIRES: amdgpu-registered-target, sqtt-marker-has-clang
// REQUIRES: sqtt-marker-has-offload-tools

#include <amd_sqtt_marker/sqtt_marker.h>

extern "C" __attribute__((global)) void marker_kernel(unsigned int value) {
  sqtt_marker_enter("outer");
  sqtt_marker_exit("outer");
  sqtt_marker_enter("inner");
  sqtt_marker_point("point");
  sqtt_marker_data("data", value);
  sqtt_marker_exit("");
}

// IR: @.sqtt_func_id_map
// IR-SAME: c"K:marker_kernel\0A
// IR-SAME: U:1:outer\0AU:2:inner\0A
// IR-SAME: P:3:point\0AP:4:data\0A
// IR-SAME: R:4:extra_payload_count=1\0A\00"
// IR: call void @llvm.amdgcn.s.ttracedata.imm(i16 6)
// IR: call void @llvm.amdgcn.s.ttracedata.imm(i16 11)
// IR: call void @llvm.amdgcn.s.ttracedata.imm(i16 12)
// IR: call void @llvm.amdgcn.s.ttracedata.imm(i16 16)
// IR: !sqtt.raw_payload
// IR: call void @llvm.amdgcn.s.ttracedata.imm(i16 1)

// DECODE: enter U:outer
// DECODE-NEXT: exit U:outer
// DECODE-NEXT: enter U:inner
// DECODE-NEXT: point P:point
// DECODE-NEXT: point P:data
// DECODE-NEXT: payload P:data[1/1]=0x0000002a
// DECODE-NEXT: exit U:inner

// DISABLED-NOT: ttracedata
// DISABLED-NOT: .sqtt_funcmap

// GFX9: s_mov_b32 m0, 6
// GFX9-NEXT: s_nop 0
// GFX9-NEXT: s_ttracedata
// GFX9-NOT: s_ttracedata_imm

// GFX12: s_ttracedata_imm 0x6
// GFX12: s_mov_b32 m0,
// GFX12-NEXT: s_nop 3
// GFX12-NEXT: s_ttracedata
// GFX12-NOT: global_inv
