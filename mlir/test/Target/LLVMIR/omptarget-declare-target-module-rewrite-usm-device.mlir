// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// This test verifies the module-scope declare target global use rewrite
// mechanism for `declare target to` and `declare target enter` variables when
// unified shared memory is required and compiling for device. In this
// configuration a reference pointer is generated for the to/enter variables
// (as with link), so uses of the original global must be rewritten to load
// from the reference pointer at module scope.
//
// As with the link test, the globals are used both directly inside of a target
// region and indirectly inside of a declare target function invoked from that
// region, and both use-sites must be rewritten.

module attributes {llvm.target_triple = "amdgcn-amd-amdhsa", omp.is_target_device = true, omp.requires = #omp<clause_requires unified_shared_memory>} {
  // CHECK-DAG: @_QMtest_0Evar_to_usm_decl_tgt_ref_ptr = weak global ptr null, align 8
  llvm.mlir.global external @_QMtest_0Evar_to_usm() {addr_space = 0 : i32, omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to)>} : i32 {
    %0 = llvm.mlir.constant(1 : i32) : i32
    llvm.return %0 : i32
  }

  // CHECK-DAG: @_QMtest_0Evar_enter_usm_decl_tgt_ref_ptr = weak global ptr null, align 8
  llvm.mlir.global external @_QMtest_0Evar_enter_usm() {addr_space = 0 : i32, omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (enter)>} : i32 {
    %0 = llvm.mlir.constant(2 : i32) : i32
    llvm.return %0 : i32
  }

  // CHECK-LABEL: define {{.*}} @_QMtest_0Puse_global
  // CHECK-DAG: %[[TO_REF:.*]] = load ptr, ptr @_QMtest_0Evar_to_usm_decl_tgt_ref_ptr, align 8
  // CHECK-DAG: store i32 100, ptr %[[TO_REF]], align 4
  // CHECK-DAG: %[[ENTER_REF:.*]] = load ptr, ptr @_QMtest_0Evar_enter_usm_decl_tgt_ref_ptr, align 8
  // CHECK-DAG: store i32 200, ptr %[[ENTER_REF]], align 4
  llvm.func @_QMtest_0Puse_global() attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (enter)>} {
    %0 = llvm.mlir.addressof @_QMtest_0Evar_to_usm : !llvm.ptr
    %1 = llvm.mlir.addressof @_QMtest_0Evar_enter_usm : !llvm.ptr
    %c100 = llvm.mlir.constant(100 : i32) : i32
    %c200 = llvm.mlir.constant(200 : i32) : i32
    llvm.store %c100, %0 : i32, !llvm.ptr
    llvm.store %c200, %1 : i32, !llvm.ptr
    llvm.return
  }

  llvm.func @test_usm_declare_target() attributes {} {
    %0 = llvm.mlir.addressof @_QMtest_0Evar_to_usm : !llvm.ptr
    %1 = llvm.mlir.addressof @_QMtest_0Evar_enter_usm : !llvm.ptr
    // CHECK-DAG: %[[TO_VAR:.*]] = load ptr, ptr @_QMtest_0Evar_to_usm_decl_tgt_ref_ptr, align 8
    // CHECK-DAG: store i32 10, ptr %[[TO_VAR]], align 4
    // CHECK-DAG: %[[ENTER_VAR:.*]] = load ptr, ptr @_QMtest_0Evar_enter_usm_decl_tgt_ref_ptr, align 8
    // CHECK-DAG: store i32 20, ptr %[[ENTER_VAR]], align 4
    // CHECK-DAG: call void @_QMtest_0Puse_global()
    %map0 = omp.map.info var_ptr(%0 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = ""}
    %map1 = omp.map.info var_ptr(%1 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = ""}
    omp.target kernel_type(generic) map_entries(%map0 -> %arg0, %map1 -> %arg1 : !llvm.ptr, !llvm.ptr) {
      %c10 = llvm.mlir.constant(10 : i32) : i32
      %c20 = llvm.mlir.constant(20 : i32) : i32
      llvm.store %c10, %arg0 : i32, !llvm.ptr
      llvm.store %c20, %arg1 : i32, !llvm.ptr
      llvm.call @_QMtest_0Puse_global() : () -> ()
      omp.terminator
    }
    llvm.return
  }
}
