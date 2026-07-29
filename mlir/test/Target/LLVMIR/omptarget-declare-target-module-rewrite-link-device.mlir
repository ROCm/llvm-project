// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// This test verifies the module-scope declare target global use rewrite
// mechanism for a `declare target link` variable when compiling for device.
//
// The declare target global is used both directly inside of a target region
// and indirectly inside of a declare target function that is invoked from
// within that target region. Because the rewrite is now applied at module
// scope during OpenMPIRBuilder finalization (rather than only within the
// outlined target region), the use of the original global inside of the
// declare target function should also be rewritten to load from the generated
// reference pointer.

module attributes {llvm.target_triple = "amdgcn-amd-amdhsa", omp.is_target_device = true} {
  // CHECK-DAG: @_QMtest_0Esp_decl_tgt_ref_ptr = weak global ptr null, align 8
  llvm.mlir.global external @_QMtest_0Esp() {addr_space = 0 : i32, omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (link)>} : i32 {
    %0 = llvm.mlir.constant(0 : i32) : i32
    llvm.return %0 : i32
  }

  // CHECK-LABEL: define {{.*}} @_QMtest_0Puse_global
  // CHECK: %[[REF:.*]] = load ptr, ptr @_QMtest_0Esp_decl_tgt_ref_ptr, align 8
  // CHECK: store i32 2, ptr %[[REF]], align 4
  llvm.func @_QMtest_0Puse_global() attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (enter)>} {
    %0 = llvm.mlir.addressof @_QMtest_0Esp : !llvm.ptr
    %1 = llvm.mlir.constant(2 : i32) : i32
    llvm.store %1, %0 : i32, !llvm.ptr
    llvm.return
  }

  llvm.func @_QQmain() attributes {} {
    %0 = llvm.mlir.addressof @_QMtest_0Esp : !llvm.ptr

    // CHECK-DAG:   omp.target:
    // CHECK-DAG: %[[V:.*]] = load ptr, ptr @_QMtest_0Esp_decl_tgt_ref_ptr, align 8
    // CHECK-DAG: store i32 1, ptr %[[V]], align 4
    // CHECK-DAG: call void @_QMtest_0Puse_global()
    %map = omp.map.info var_ptr(%0 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = ""}
    omp.target kernel_type(generic) map_entries(%map -> %arg0 : !llvm.ptr) {
      %1 = llvm.mlir.constant(1 : i32) : i32
      llvm.store %1, %arg0 : i32, !llvm.ptr
      llvm.call @_QMtest_0Puse_global() : () -> ()
      omp.terminator
    }

    llvm.return
  }
}
