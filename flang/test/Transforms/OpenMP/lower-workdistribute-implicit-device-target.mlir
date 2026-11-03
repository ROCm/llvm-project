// Device-offload (is_target_device) compilation of an implicit-workdistribute
// candidate. The target-device module must carry no host_eval clause
// (HostOpFiltering asserts none survives there). Bounds are mapped and loaded
// inside the target instead.

// RUN: fir-opt --lower-workdistribute='implicit=device' %s | FileCheck %s

module attributes {omp.is_gpu = true, omp.is_target_device = true} {
  // Scalar live-ins map ByCopy, so fission never runs and the kernel stays
  // generic.
  // CHECK-LABEL: func.func @nested_device(
  // CHECK:         omp.target kernel_type(generic) map_entries({{.*}} : !fir.ref<index>, !fir.ref<index>, !fir.ref<index>, !fir.ref<index>)
  // CHECK:           %[[LB_LD:.*]] = fir.load %{{.*}} : !fir.ref<index>
  // CHECK:           %[[UB_LD:.*]] = fir.load %{{.*}} : !fir.ref<index>
  // CHECK:           %[[ST_LD:.*]] = fir.load %{{.*}} : !fir.ref<index>
  // CHECK:           omp.loop_nest (%{{.*}}) : index = (%[[LB_LD]]) to (%[[UB_LD]]) inclusive step (%[[ST_LD]]) {
  // CHECK:             fir.do_loop %{{.*}} = %[[LB_LD]] to %[[UB_LD]] step %[[ST_LD]] unordered
  // CHECK:         } {omp.combined}
  func.func @nested_device(%lb : index, %ub : index, %step : index,
                           %addr : !fir.ref<index>) {
    fir.do_loop %iv = %lb to %ub step %step unordered {
      fir.do_loop %iv2 = %lb to %ub step %step unordered {
        %zero = arith.constant 0 : index
        fir.store %zero to %addr : !fir.ref<index>
      }
    }
    return
  }

  // An array live-in maps ByRef, so fission runs and promotes to spmd. The
  // matching host module (@array_bound) also gets host_eval. This one must not.
  // CHECK-LABEL: func.func @array_device(
  // CHECK:         omp.target_data
  // CHECK:           omp.target kernel_type(spmd) map_entries(
  // CHECK:             omp.loop_nest
  // CHECK:         } {omp.combined}
  func.func @array_device(%n : !fir.ref<index>, %arr : !fir.ref<!fir.array<1024xf32>>) {
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    %ub = fir.load %n : !fir.ref<index>
    %shape = fir.shape %c1024 : (index) -> !fir.shape<1>
    fir.do_loop %iv = %c1 to %ub step %c1 unordered {
      %zero = arith.constant 0.0 : f32
      %elem = fir.array_coor %arr(%shape) %iv : (!fir.ref<!fir.array<1024xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
      fir.store %zero to %elem : !fir.ref<f32>
    }
    return
  }
}
