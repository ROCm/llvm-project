! Tests `do concurrent` mapping when mapped value(s) depend on values defined
! outside the target region; e.g. the size of the array is dynamic. This needs
! to be handled by localizing these region outsiders by either cloning them in
! the region or in case we cannot do that, map them and use the mapped values.

! RUN: %flang_fc1 -emit-hlfir -fopenmp -fdo-concurrent-parallel=device %s -o - \
! RUN:   | FileCheck %s

subroutine foo(n)
  implicit none
  integer :: n
  integer :: i
  integer, dimension(n) :: a

  do concurrent(i=1:10)
    a(i) = i
  end do
end subroutine

! CHECK-DAG: %[[I_DECL:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFfooEi"}
! CHECK-DAG: %[[A_DECL:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFfooEa"}
! CHECK-DAG: %[[N_ALLOC:.*]] = fir.alloca i32

! CHECK-DAG: %[[I_MAP:.*]] = omp.map.info var_ptr(%[[I_DECL]]#1 : {{.*}})
! CHECK-DAG: %[[A_MAP:.*]] = omp.map.info var_ptr(%[[A_DECL]]#1 : {{.*}})
! CHECK-DAG: %[[LOOP_LB_MAP:.*]] = omp.map.info var_ptr(%{{.*}} : {{.*}}) {{.*}} {name = "loop.0.lb"}
! CHECK-DAG: %[[LOOP_UB_MAP:.*]] = omp.map.info var_ptr(%{{.*}} : {{.*}}) {{.*}} {name = "loop.0.ub"}
! CHECK-DAG: %[[LOOP_STEP_MAP:.*]] = omp.map.info var_ptr(%{{.*}} : {{.*}}) {{.*}} {name = "loop.0.step"}
! CHECK-DAG: %[[N_MAP:.*]] = omp.map.info var_ptr(%{{.*}} : {{.*}})

! CHECK: omp.target
! CHECK-SAME: map_entries(%[[LOOP_LB_MAP]] -> %[[LB_ARG:arg[0-9]*]],
! CHECK-SAME:             %[[LOOP_UB_MAP]] -> %[[UB_ARG:arg[0-9]*]],
! CHECK-SAME:             %[[LOOP_STEP_MAP]] -> %[[STEP_ARG:arg[0-9]*]],
! CHECK-SAME:             %[[I_MAP]] -> %[[I_ARG:arg[0-9]*]],
! CHECK-SAME:             %[[A_MAP]] -> %[[A_ARG:arg[0-9]*]],
! CHECK-SAME:             %[[N_MAP]] -> %[[N_ARG:arg[0-9]*]] : {{.*}})
! CHECK-SAME: {{.*}} {

! CHECK-DAG:  %[[N_VAL:.*]] = fir.load %[[N_ARG]]
! CHECK-DAG:  %[[A_SHAPE:.*]] = fir.shape %[[N_VAL]] : (index) -> !fir.shape<1>
! CHECK-DAG:  %{{.*}} = hlfir.declare %[[I_ARG]]
! CHECK-DAG:  %{{.*}} = hlfir.declare %[[A_ARG]](%[[A_SHAPE]])

! CHECK:   omp.terminator
! CHECK: }
