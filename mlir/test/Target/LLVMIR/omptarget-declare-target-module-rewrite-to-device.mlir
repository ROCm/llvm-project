// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// This test verifies that the module-scope declare target global use rewrite
// mechanism is NOT applied for a regular `declare target to`/`enter` variable
// (i.e. without unified shared memory) when compiling for device. In this
// configuration no reference pointer global is generated, so uses of the
// original global must remain direct references to the global itself, both
// inside of the target region and inside of a declare target function invoked
// from the target region. No `_decl_tgt_ref_ptr` global should be created and
// no load-from-reference-pointer should be emitted.

module attributes {llvm.target_triple = "amdgcn-amd-amdhsa", omp.is_target_device = true} {
  // CHECK-NOT: @_QMtest_0Evar_to_decl_tgt_ref_ptr
  // CHECK-NOT: @_QMtest_0Evar_enter_decl_tgt_ref_ptr
  // CHECK-DAG: @_QMtest_0Evar_to = global i32
  llvm.mlir.global external @_QMtest_0Evar_to() {addr_space = 0 : i32, omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to)>} : i32 {
    %0 = llvm.mlir.constant(1 : i32) : i32
    llvm.return %0 : i32
  }

  // CHECK-DAG: @_QMtest_0Evar_enter = global i32
  llvm.mlir.global external @_QMtest_0Evar_enter() {addr_space = 0 : i32, omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (enter)>} : i32 {
    %0 = llvm.mlir.constant(2 : i32) : i32
    llvm.return %0 : i32
  }

  // CHECK-LABEL: define {{.*}} @_QMtest_0Puse_global
  // CHECK-NOT: load ptr, ptr @_QMtest_0Evar_to_decl_tgt_ref_ptr
  // CHECK-NOT: load ptr, ptr @_QMtest_0Evar_enter_decl_tgt_ref_ptr
  // CHECK-DAG: store i32 100, ptr @_QMtest_0Evar_to, align 4
  // CHECK-DAG: store i32 200, ptr @_QMtest_0Evar_enter, align 4
  llvm.func @_QMtest_0Puse_global() attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (enter)>} {
    %0 = llvm.mlir.addressof @_QMtest_0Evar_to : !llvm.ptr
    %1 = llvm.mlir.addressof @_QMtest_0Evar_enter : !llvm.ptr
    %c100 = llvm.mlir.constant(100 : i32) : i32
    %c200 = llvm.mlir.constant(200 : i32) : i32
    llvm.store %c100, %0 : i32, !llvm.ptr
    llvm.store %c200, %1 : i32, !llvm.ptr
    llvm.return
  }

  llvm.func @test_declare_target() attributes {} {
    %0 = llvm.mlir.addressof @_QMtest_0Evar_to : !llvm.ptr
    %1 = llvm.mlir.addressof @_QMtest_0Evar_enter : !llvm.ptr
    // CHECK-DAG: store i32 10, ptr @_QMtest_0Evar_to, align 4
    // CHECK-DAG: store i32 20, ptr @_QMtest_0Evar_enter, align 4
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
