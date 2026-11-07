// COM: gfx1250-strict is the first concrete ISA whose name contains a '-'.
// RUN: source-to-bc-with-dev-libs %s --isa=amdgcn-amd-amdhsa--gfx1250-strict \
// RUN:   -o %t-dashed-isa.bc

// RUN: %llvm-dis %t-dashed-isa.bc -o - | %FileCheck %s

// CHECK: target triple = "{{.*}}-amd-amdhsa"
// CHECK: @__oclc_ISA_version = internal {{.*}}i32 12500
// CHECK: define internal float @__ocml_powr_f32

void kernel dashed_isa(__global float *out, float x, float y) {
  out[0] = powr(x, y);
}
