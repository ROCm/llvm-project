// RUN: mlir-opt %s -convert-openmp-to-llvm | mlir-translate -mlir-to-llvmir | FileCheck %s

module attributes {llvm.target_triple = "amdgcn-amd-amdhsa", omp.is_gpu = true, omp.is_target_device = true} {
  // CHECK-LABEL: define void @test_alloc_free_device_i64()
  // CHECK: %[[ALLOC:.*]] = call ptr @omp_target_alloc(i64 8, i32 0)
  // CHECK: %[[PTRTOINT:.*]] = ptrtoint ptr %[[ALLOC]] to i64
  // CHECK: %[[INTTOPTR:.*]] = inttoptr i64 %[[PTRTOINT]] to ptr
  // CHECK: call void @omp_target_free(ptr %[[INTTOPTR]], i32 0)
  // CHECK: ret void
  llvm.func @test_alloc_free_device_i64() -> () {
    %device = llvm.mlir.constant(0 : i32) : i32
    %1 = omp.target_allocmem %device : i32, i64
    omp.target_freemem %device, %1 : i32, i64
    llvm.return
  }
}
