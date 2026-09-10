// RUN: %clang --target=%host-triple -x hip -nogpuinc -nogpulib \
// RUN:   --offload-arch=gfx1200 -DAMD_SQTT_MARKER_ENABLE=1 \
// RUN:   -I%sqtt-marker-include -fsyntax-only %s
// REQUIRES: amdgpu-registered-target, sqtt-marker-has-clang

#include <amd_sqtt_marker/sqtt_marker.h>

extern "C" __attribute__((device, used)) void
use_sqtt_marker_header(unsigned int value) {
  sqtt_marker_enter("scope");
  sqtt_marker_point("point");
  sqtt_marker_data("data", value);
  sqtt_marker_point_id(7);
  sqtt_marker_enter_id(9);
  sqtt_marker_exit_id(11);
  sqtt_marker_exit("scope");
}
