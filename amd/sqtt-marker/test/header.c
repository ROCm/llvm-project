// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1200 -x c -std=c11 \
// RUN:   -ffreestanding -O2 -S -emit-llvm -I%sqtt-marker-include \
// RUN:   -DAMD_SQTT_MARKER_ENABLE=1 %s -o - | \
// RUN:   %FileCheck %s --check-prefix=ENABLED
// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1200 -x c -std=c11 \
// RUN:   -ffreestanding -O2 -S -emit-llvm -I%sqtt-marker-include %s -o - | \
// RUN:   %FileCheck %s --check-prefix=DISABLED
// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1200 -x c -std=c11 \
// RUN:   -ffreestanding -O2 -S -I%sqtt-marker-include \
// RUN:   -DAMD_SQTT_MARKER_ENABLE=1 -mllvm -amdgpu-expert-scheduling-mode \
// RUN:   %s -o - | %FileCheck %s --check-prefix=EXPERT
// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx90a -x c -std=c11 \
// RUN:   -ffreestanding -O2 -S -I%sqtt-marker-include \
// RUN:   -DAMD_SQTT_MARKER_ENABLE=1 %s -o - | \
// RUN:   %FileCheck %s --check-prefix=GFX9
// REQUIRES: amdgpu-registered-target, sqtt-marker-has-clang

#include <amd_sqtt_marker/sqtt_marker.h>

void use_markers(unsigned int id) {
  sqtt_marker_enter("scope");
  sqtt_marker_point("point");
  sqtt_marker_data("data", id);
  sqtt_marker_point_id(7);
  sqtt_marker_enter_id(9);
  sqtt_marker_exit_id(11);
  sqtt_marker_exit((const char *)0);
}

// ENABLED-LABEL: define{{.*}} void @use_markers(
// ENABLED: call void @sqtt_marker_enter(
// ENABLED: call void @sqtt_marker_point(
// ENABLED: call void @sqtt_marker_data(
// ENABLED-COUNT-3: call i32 asm sideeffect "s_mov_b32 m0, $1
// ENABLED: call void @sqtt_marker_exit(

// DISABLED-LABEL: define{{.*}} void @use_markers(
// DISABLED-NOT: sqtt_marker
// DISABLED-NOT: ttracedata
// DISABLED: ret void

// EXPERT: s_mov_b32 m0,
// EXPERT-NEXT: s_nop 3
// EXPERT-NEXT: s_ttracedata

// GFX9: s_mov_b32 m0,
// GFX9-NEXT: s_nop 0
// GFX9-NEXT: s_ttracedata
