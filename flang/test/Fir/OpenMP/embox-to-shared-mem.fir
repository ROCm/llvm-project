// RUN: tco -o - %s | FileCheck %s

module attributes {omp.is_target_device = true} {
  // CHECK-LABEL: declare void @scalar(ptr)
  func.func private @scalar(%x : !fir.box<f32>)
  // CHECK-LABEL: declare void @array(ptr)
  func.func private @array(%x : !fir.box<!fir.array<?xi32>>)

  // CHECK-LABEL: define void @embox
  func.func @embox(%arg0 : !fir.ref<f32>, %arg1 : !fir.ref<!fir.array<?xi32>>) attributes {omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (to)>} {
    // CHECK: %[[DESC2:.*]] = call align 8 ptr @__kmpc_alloc_shared(i64 48)
    // CHECK: %[[DESC1:.*]] = call align 8 ptr @__kmpc_alloc_shared(i64 24)
    %a = fir.embox %arg0 : (!fir.ref<f32>) -> !fir.box<f32>
    %c0 = arith.constant 0 : i64
    %b = fircg.ext_embox %arg1(%c0) origin %c0[%c0, %c0, %c0] : (!fir.ref<!fir.array<?xi32>>, i64, i64, i64, i64, i64) -> !fir.box<!fir.array<?xi32>>

    // CHECK: call void @scalar(ptr %[[DESC1]])
    fir.call @scalar(%a) : (!fir.box<f32>) -> ()

    // CHECK: call void @array(ptr %[[DESC2]])
    fir.call @array(%b) : (!fir.box<!fir.array<?xi32>>) -> ()

    // CHECK: call void @__kmpc_free_shared(ptr %[[DESC1]], i64 24)
    // CHECK: call void @__kmpc_free_shared(ptr %[[DESC2]], i64 48)
    // CHECK: ret void
    return
  }

}
