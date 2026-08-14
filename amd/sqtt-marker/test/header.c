// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1200 -x c -std=c11 \
// RUN:   -ffreestanding -O2 -S -emit-llvm -I%sqtt-marker-include \
// RUN:   -DAMD_SQTT_MARKER_ENABLE=1 %s -o - | \
// RUN:   %FileCheck %s --check-prefix=ENABLED
// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1200 -x c -std=c11 \
// RUN:   -ffreestanding -O2 -S -emit-llvm -I%sqtt-marker-include %s -o - | \
// RUN:   %FileCheck %s --check-prefix=DISABLED
// REQUIRES: amdgpu-registered-target, sqtt-marker-has-clang

#include <amd_sqtt_marker/sqtt_marker.h>

void use_markers(unsigned int id) {
  sqtt_marker_enter("scope");
  sqtt_marker_point("point");
  sqtt_marker_data("data", id);
  sqtt_marker_point_id(7);
  sqtt_marker_enter_id(9);
  sqtt_marker_exit_id(11);
  sqtt_marker_exit("scope");
}

// ENABLED-LABEL: define{{.*}} void @use_markers(
// ENABLED: call void @sqtt_marker_enter(
// ENABLED: call void @sqtt_marker_point(
// ENABLED: call void @sqtt_marker_data(
// ENABLED: call void @llvm.amdgcn.s.ttracedata(i32 28)
// ENABLED: call void @llvm.amdgcn.s.ttracedata(i32 38)
// ENABLED: call void @llvm.amdgcn.s.ttracedata(i32 1)
// ENABLED: call void @sqtt_marker_exit(

// DISABLED-LABEL: define{{.*}} void @use_markers(
// DISABLED-NOT: sqtt_marker
// DISABLED-NOT: ttracedata
// DISABLED: ret void
