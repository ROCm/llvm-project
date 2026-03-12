// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

module {
  omp.private {type = private} @iv_priv : i32

  llvm.func @distribute_parallel_private(%lb : i32, %ub : i32, %step : i32,
                                         %store : !llvm.ptr) {
    omp.parallel {
      omp.distribute {
        omp.loop_nest (%iv) : i32 = (%lb) to (%ub) step (%step) {
          omp.parallel private(@iv_priv %store -> %priv : !llvm.ptr) {
            llvm.store %iv, %priv : i32, !llvm.ptr
            omp.terminator
          }
          omp.yield
        }
      }
      omp.terminator
    } {omp.composite}
    llvm.return
  }
}

// CHECK-LABEL: define void @distribute_parallel_private(
// CHECK: call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @{{.*}}, i32 1, ptr @distribute_parallel_private..omp_par.1,

// CHECK-LABEL: define internal void @distribute_parallel_private..omp_par.1(
// CHECK: distribute.alloca:
// CHECK-NEXT: br label %[[AFTER_ALLOC:[^[:space:]]+]]
// CHECK: [[AFTER_ALLOC]]:
// CHECK: omp_loop.body:
// CHECK: call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @{{.*}}, i32 1, ptr @distribute_parallel_private..omp_par,

// CHECK-LABEL: define internal void @distribute_parallel_private..omp_par(
// CHECK: %[[PRIVATE:[^[:space:]]+]] = alloca i32, align 4
// CHECK: store i32 %{{.*}}, ptr %[[PRIVATE]], align 4
