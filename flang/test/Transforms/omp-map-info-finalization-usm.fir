// RUN: fir-opt --split-input-file --omp-map-info-finalization %s | FileCheck %s

// Test that the 'close' map flag is cleared from member maps if the parent map
// (derived type) does not have the 'close' flag. On current upstream the old
// implicit descriptor reproducer no longer yields a member with 'close', so
// keep the test focused on the finalization behavior itself by feeding an
// inconsistent member/parent pair directly.

module attributes {omp.requires = #omp<clause_requires unified_shared_memory>} {
  func.func @test_usm_close_propagation(%arg0: !fir.ref<!fir.type<t{i:i32}>>, %arg1: !fir.ref<i32>) {
    %member = omp.map.info var_ptr(%arg1 : !fir.ref<i32>, i32) map_clauses(close, tofrom) capture(ByRef) -> !fir.ref<i32> {name = "parent.i"}
    %parent = omp.map.info var_ptr(%arg0 : !fir.ref<!fir.type<t{i:i32}>>, !fir.type<t{i:i32}>) map_clauses(tofrom) capture(ByRef) members(%member : [0] : !fir.ref<i32>) -> !fir.ref<!fir.type<t{i:i32}>> {name = "parent", partial_map = true}

    omp.target map_entries(%parent -> %arg2 : !fir.ref<!fir.type<t{i:i32}>>) {
      omp.terminator
    }
    return
  }
}

// CHECK-LABEL: func.func @test_usm_close_propagation
// CHECK: %[[MEMBER:.*]] = omp.map.info {{.*}} map_clauses(tofrom) {{.*}} {name = "parent.i"}
// CHECK: %[[PARENT:.*]] = omp.map.info {{.*}} map_clauses(tofrom) {{.*}} members(%[[MEMBER]] : [0] : !fir.ref<i32>) {{.*}} {name = "parent", partial_map = true}
// CHECK-NOT: close
