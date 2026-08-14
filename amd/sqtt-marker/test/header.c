// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1200 -x c -std=c11 \
// RUN:   -ffreestanding -fsyntax-only -I%sqtt-marker-include \
// RUN:   -DAMD_SQTT_MARKER_ENABLE=1 %s
// REQUIRES: amdgpu-registered-target

#include <amd_sqtt_marker/sqtt_marker.h>

void use_markers(unsigned int id) {
  amd_sqtt_marker_enter_string("scope");
  amd_sqtt_marker_point_string("point");
  amd_sqtt_marker_data_string("data", id);
  amd_sqtt_marker_point_id(id);
  amd_sqtt_marker_enter_id(id);
  amd_sqtt_marker_exit();
  amd_sqtt_marker_exit_string("scope");
}
