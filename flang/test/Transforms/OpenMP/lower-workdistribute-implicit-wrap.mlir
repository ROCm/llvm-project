// Implicit-workdistribute wraps each candidate `fir.do_loop unordered`.
// host: in `omp.teams { omp.workdistribute { ... } }` (lowered to the
// teams/parallel/distribute/wsloop nest). device: the same, inside an
// `omp.target` with an implicit `omp.map.info` per live-in. none/default: no-op.

// RUN: fir-opt --lower-workdistribute='implicit=host'   %s | FileCheck %s --check-prefixes=BOTH,HOST
// RUN: fir-opt --lower-workdistribute='implicit=device' %s | FileCheck %s --check-prefixes=BOTH,DEVICE
// RUN: fir-opt --lower-workdistribute='implicit=none'   %s | FileCheck %s --check-prefixes=BOTH,NOOP
// RUN: fir-opt --lower-workdistribute                   %s | FileCheck %s --check-prefixes=BOTH,NOOP

// BOTH-LABEL:   func.func @candidate(
// HOST:           omp.teams {
// HOST:             omp.parallel {
// HOST:               omp.distribute {
// HOST:                 omp.wsloop {
// HOST:                   omp.loop_nest (%{{.*}}) : index = (%{{.*}}) to (%{{.*}}) inclusive step (%{{.*}}) {
// HOST:                     arith.constant 0 : index
// HOST:                     fir.store
// HOST:                     omp.yield
// HOST:                   }
// HOST:                 } {omp.composite}
// HOST:               } {omp.composite}
// HOST:               omp.terminator
// HOST:             } {omp.composite}
// HOST:             omp.terminator
// HOST:           }
// Every live-in is a trivial scalar mapped ByCopy, so fission never runs and the
// kernel stays generic; `lb`, `ub`, `step` and `addr` are the four map_entries.
// Matching the two clauses adjacent forbids a host_eval between them.
// DEVICE:         omp.target kernel_type(generic) map_entries({{.*}} : !fir.ref<index>, !fir.ref<index>, !fir.ref<index>, !fir.ref<index>)
// DEVICE:           %[[C_LB:.*]] = fir.load %{{.*}} : !fir.ref<index>
// DEVICE:           %[[C_UB:.*]] = fir.load %{{.*}} : !fir.ref<index>
// DEVICE:           %[[C_ST:.*]] = fir.load %{{.*}} : !fir.ref<index>
// DEVICE:           omp.loop_nest (%{{.*}}) : index = (%[[C_LB]]) to (%[[C_UB]]) inclusive step (%[[C_ST]]) {
// DEVICE:         } {omp.combined}
// NOOP-NOT:       omp.teams
// NOOP-NOT:       omp.workdistribute
// NOOP:           fir.do_loop %{{[^ ]+}} = %{{[^ ]+}} to %{{[^ ]+}} step %{{[^ ]+}} unordered
// BOTH:           return
func.func @candidate(%lb : index, %ub : index, %step : index,
                     %addr : !fir.ref<index>) {
  fir.do_loop %iv = %lb to %ub step %step unordered {
    %zero = arith.constant 0 : index
    fir.store %zero to %addr : !fir.ref<index>
  }
  return
}

// A loaded bound cannot be rematerialized, so it is mapped; the constant serving
// as lb and step is, so it gets no kernarg slot.
// BOTH-LABEL:   func.func @loaded_bound(
// HOST:           omp.teams {
// HOST:             omp.parallel {
// HOST:               omp.distribute {
// HOST:                 omp.wsloop {
// HOST:                   omp.loop_nest
// Where the cloned constant lands relative to the load is not pinned here.
// DEVICE:         omp.target kernel_type(generic) map_entries({{.*}} : !fir.ref<index>, !fir.ref<index>)
// DEVICE:           %[[L_UB:.*]] = fir.load %{{.*}} : !fir.ref<index>
// DEVICE:           omp.loop_nest (%{{.*}}) : index = (%{{.*}}) to (%[[L_UB]]) inclusive step (%{{.*}}) {
// DEVICE:         } {omp.combined}
// NOOP-NOT:       omp.teams
// NOOP-NOT:       omp.target
// NOOP:           fir.do_loop %{{[^ ]+}} = %{{[^ ]+}} to %{{[^ ]+}} step %{{[^ ]+}} unordered
func.func @loaded_bound(%n : !fir.ref<index>, %addr : !fir.ref<index>) {
  %c1 = arith.constant 1 : index
  %ub = fir.load %n : !fir.ref<index>
  fir.do_loop %iv = %c1 to %ub step %c1 unordered {
    %zero = arith.constant 0 : index
    fir.store %zero to %addr : !fir.ref<index>
  }
  return
}

// An array live-in maps ByRef, which is what makes fission run and promote the
// kernel to spmd. host_eval comes with the promotion, on the host module only
// (see lower-workdistribute-implicit-device-target.mlir).
// BOTH-LABEL:   func.func @array_bound(
// HOST:           omp.teams {
// HOST:             omp.parallel {
// HOST:               omp.distribute {
// HOST:                 omp.wsloop {
// HOST:                   omp.loop_nest
// DEVICE:         omp.target_data
// DEVICE:           omp.target kernel_type(spmd)
// DEVICE-SAME:        host_eval(
// DEVICE:             omp.loop_nest
// DEVICE:         } {omp.combined}
// NOOP-NOT:       omp.teams
// NOOP-NOT:       omp.target
// NOOP:           fir.do_loop %{{[^ ]+}} = %{{[^ ]+}} to %{{[^ ]+}} step %{{[^ ]+}} unordered
func.func @array_bound(%n : !fir.ref<index>, %arr : !fir.ref<!fir.array<1024xf32>>) {
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

// Ordered loops (regular Fortran DO) are not wrapped in any mode.
// BOTH-LABEL:   func.func @ordered_skipped(
// BOTH-NOT:       omp.teams
// BOTH-NOT:       omp.workdistribute
// BOTH-NOT:       omp.target
// BOTH:           fir.do_loop
// BOTH:           return
func.func @ordered_skipped(%lb : index, %ub : index, %step : index,
                           %addr : !fir.ref<index>) {
  fir.do_loop %iv = %lb to %ub step %step {
    %zero = arith.constant 0 : index
    fir.store %zero to %addr : !fir.ref<index>
  }
  return
}

// A loop already inside an explicit workdistribute is honored as-is in every
// mode: the implicit wrap is skipped, and the rest of the pass lowers the
// existing `omp.teams { omp.workdistribute { ... } }` normally.
// BOTH-LABEL:   func.func @explicit_passthrough(
// BOTH:           omp.teams {
// BOTH:             omp.parallel {
// BOTH:               omp.distribute {
// BOTH:                 omp.wsloop {
// BOTH:                   omp.loop_nest
// BOTH-NOT:       omp.teams
// BOTH-NOT:       omp.target
func.func @explicit_passthrough(%lb : index, %ub : index, %step : index,
                                %addr : !fir.ref<index>) {
  omp.teams {
    omp.workdistribute {
      fir.do_loop %iv = %lb to %ub step %step unordered {
        %zero = arith.constant 0 : index
        fir.store %zero to %addr : !fir.ref<index>
      }
      omp.terminator
    }
    omp.terminator
  }
  return
}

// Loops already inside an `omp.parallel` are user-managed and not wrapped
// in any mode.
// BOTH-LABEL:   func.func @parallel_skipped(
// BOTH:           omp.parallel {
// BOTH-NOT:       omp.teams
// BOTH-NOT:       omp.target
// BOTH:           fir.do_loop
// BOTH:           omp.terminator
func.func @parallel_skipped(%lb : index, %ub : index, %step : index,
                            %addr : !fir.ref<index>) {
  omp.parallel {
    fir.do_loop %iv = %lb to %ub step %step unordered {
      %zero = arith.constant 0 : index
      fir.store %zero to %addr : !fir.ref<index>
    }
    omp.terminator
  }
  return
}

// Only the outermost unordered loop is wrapped (host/device). The inner
// unordered loop rides along unchanged inside the body. In none/default mode
// both loops are left as bare `fir.do_loop`s.
// BOTH-LABEL:   func.func @nested_only_outer(
// HOST:           omp.teams {
// HOST:             omp.parallel {
// HOST:               omp.distribute {
// HOST:                 omp.wsloop {
// HOST:                   omp.loop_nest
// HOST:                     fir.do_loop {{.*}} unordered
// Inner and outer loop share the same loaded bounds.
// DEVICE:         omp.target kernel_type(generic) map_entries(
// DEVICE:           %[[N_LB_LD:.*]] = fir.load %{{.*}} : !fir.ref<index>
// DEVICE:           %[[N_UB_LD:.*]] = fir.load %{{.*}} : !fir.ref<index>
// DEVICE:           %[[N_ST_LD:.*]] = fir.load %{{.*}} : !fir.ref<index>
// DEVICE:           omp.loop_nest (%{{.*}}) : index = (%[[N_LB_LD]]) to (%[[N_UB_LD]]) inclusive step (%[[N_ST_LD]]) {
// DEVICE:             fir.do_loop %{{.*}} = %[[N_LB_LD]] to %[[N_UB_LD]] step %[[N_ST_LD]] unordered
// DEVICE:         } {omp.combined}
// NOOP-NOT:       omp.teams
// NOOP-NOT:       omp.workdistribute
// NOOP:           fir.do_loop {{.*}} unordered
// NOOP:             fir.do_loop {{.*}} unordered
func.func @nested_only_outer(%lb : index, %ub : index, %step : index,
                             %addr : !fir.ref<index>) {
  fir.do_loop %iv = %lb to %ub step %step unordered {
    fir.do_loop %iv2 = %lb to %ub step %step unordered {
      %zero = arith.constant 0 : index
      fir.store %zero to %addr : !fir.ref<index>
    }
  }
  return
}
