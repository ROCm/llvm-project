! Tests mapping of a basic `do concurrent` loop to
! `!$omp target teams distribute parallel do`.

! RUN: %flang_fc1 -emit-hlfir -fopenmp -fdo-concurrent-parallel=device %s -o - \
! RUN:   | FileCheck %s
! RUN: bbc -emit-hlfir -fopenmp -fdo-concurrent-parallel=device %s -o - \
! RUN:   | FileCheck %s

! CHECK-LABEL: do_concurrent_basic
program do_concurrent_basic
    implicit none
    integer :: a(10)
    integer :: i

    ! CHECK-DAG: %[[I_ORIG_ALLOC:.*]] = fir.alloca i32 {bindc_name = "i"}
    ! CHECK: %[[I_ORIG_DECL:.*]]:2 = hlfir.declare %[[I_ORIG_ALLOC]]

    ! CHECK-DAG: %[[A_ADDR:.*]] = fir.address_of(@_QFEa)
    ! CHECK: %[[A_SHAPE:.*]] = fir.shape %[[A_EXTENT:.*]] : (index) -> !fir.shape<1>
    ! CHECK: %[[A_ORIG_DECL:.*]]:2 = hlfir.declare %[[A_ADDR]](%[[A_SHAPE]])

    ! CHECK-NOT: fir.do_loop

    ! CHECK-DAG: %[[I_MAP_INFO:.*]] = omp.map.info var_ptr(%[[I_ORIG_DECL]]#1

    ! CHECK-DAG: %[[LB_MAP_INFO:.*]] = omp.map.info {{.*}} !fir.ref<index> {name = "loop.0.lb"}
    ! CHECK-DAG: %[[UB_MAP_INFO:.*]] = omp.map.info {{.*}} !fir.ref<index> {name = "loop.0.ub"}
    ! CHECK-DAG: %[[STEP_MAP_INFO:.*]] = omp.map.info {{.*}} !fir.ref<index> {name = "loop.0.step"}

    ! CHECK: %[[C1:.*]] = arith.constant 1 : index
    ! CHECK: %[[C0:.*]] = arith.constant 0 : index
    ! CHECK: %[[UPPER_BOUND:.*]] = arith.subi %[[A_EXTENT]], %[[C1]] : index

    ! CHECK: %[[A_BOUNDS:.*]] = omp.map.bounds lower_bound(%[[C0]] : index)
    ! CHECK-SAME: upper_bound(%[[UPPER_BOUND]] : index)
    ! CHECK-SAME: extent(%[[A_EXTENT]] : index)
    ! CHECK-SAME: start_idx(%[[C1]] : index)

    ! CHECK-DAG: %[[A_MAP_INFO:.*]] = omp.map.info var_ptr(%[[A_ORIG_DECL]]#1 : {{[^(]+}})
    ! CHECK-SAME: map_clauses(implicit, tofrom) capture(ByRef) bounds(%[[A_BOUNDS]])

    ! CHECK: %[[TRIP_COUNT:.*]] = arith.muli %{{.*}}, %{{.*}} : i64
    ! CHECK: omp.target
    ! CHECK-SAME: map_entries(%[[LB_MAP_INFO]] -> %[[LB_ARG:.[[:alnum:]]+]],
    ! CHECK-SAME:             %[[UB_MAP_INFO]] -> %[[UB_ARG:.[[:alnum:]]+]],
    ! CHECK-SAME:             %[[STEP_MAP_INFO]] -> %[[STEP_ARG:.[[:alnum:]]+]],
    ! CHECK-SAME:             %[[I_MAP_INFO]] -> %[[I_ARG:[[:alnum:]]+]],
    ! CHECK-SAME:             %[[A_MAP_INFO]] -> %[[A_ARG:.[[:alnum:]]+]]
    ! CHECK-SAME:             %[[A_EXT:.*]] -> %[[A_EXT_ARG:.[[:alnum:]]+]]
    ! CHECK-SAME: trip_count(%[[TRIP_COUNT]] : i64)

    ! CHECK-NEXT: ^{{.*}}(%[[LB_ARG]]: !fir.ref<index>,
    ! CHECK-SAME:         %[[UB_ARG]]: !fir.ref<index>, %[[STEP_ARG]]: !fir.ref<index>,
    ! CHECK-SAME:         %[[I_ARG]]: !fir.ref<i32>,
    ! CHECK-SAME:         %[[A_ARG]]: !fir.ref<!fir.array<10xi32>>, %[[A_EXT_ARG]]: !fir.ref<index>):

    ! CHECK: %[[A_EXT:.*]] = fir.load %[[A_EXT_ARG]] : !fir.ref<index>

    ! CHECK: %[[LB_DEV_DECL:.*]]:2 = hlfir.declare %[[LB_ARG]]
    ! CHECK: %[[LB_DEV_VAL:.*]] = fir.load %[[LB_DEV_DECL]]#1

    ! CHECK: %[[UB_DEV_DECL:.*]]:2 = hlfir.declare %[[UB_ARG]]
    ! CHECK: %[[UB_DEV_VAL:.*]] = fir.load %[[UB_DEV_DECL]]#1

    ! CHECK: %[[STEP_DEV_DECL:.*]]:2 = hlfir.declare %[[STEP_ARG]]
    ! CHECK: %[[STEP_DEV_VAL:.*]] = fir.load %[[STEP_DEV_DECL]]#1

    ! CHECK: %[[A_SHAPE:.*]] = fir.shape %[[A_EXT]] : (index) -> !fir.shape<1>
    ! CHECK: %[[A_DEV_DECL:.*]]:2 = hlfir.declare %[[A_ARG]](%[[A_SHAPE]])

    ! CHECK: omp.teams {
    ! CHECK-NEXT: omp.parallel {

    ! CHECK-NEXT: %[[ITER_VAR:.*]] = fir.alloca i32 {bindc_name = "i"}
    ! CHECK-NEXT: %[[BINDING:.*]]:2 = hlfir.declare %[[ITER_VAR]] {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

    ! CHECK-NEXT: omp.distribute {
    ! CHECK-NEXT: omp.wsloop {

    ! CHECK-NEXT: omp.loop_nest (%[[ARG0:.*]]) : index = (%[[LB_DEV_VAL]]) to (%[[UB_DEV_VAL]]) inclusive step (%[[STEP_DEV_VAL]]) {
    ! CHECK-NEXT: %[[IV_IDX:.*]] = fir.convert %[[ARG0]] : (index) -> i32
    ! CHECK-NEXT: fir.store %[[IV_IDX]] to %[[BINDING]]#1 : !fir.ref<i32>
    ! CHECK-NEXT: %[[IV_VAL1:.*]] = fir.load %[[BINDING]]#0 : !fir.ref<i32>
    ! CHECK-NEXT: %[[IV_VAL2:.*]] = fir.load %[[BINDING]]#0 : !fir.ref<i32>
    ! CHECK-NEXT: %[[IV_VAL_I64:.*]] = fir.convert %[[IV_VAL2]] : (i32) -> i64
    ! CHECK-NEXT: %[[ARR_ACCESS:.*]] = hlfir.designate %[[A_DEV_DECL]]#0 (%[[IV_VAL_I64]])  : (!fir.ref<!fir.array<10xi32>>, i64) -> !fir.ref<i32>
    ! CHECK-NEXT: hlfir.assign %[[IV_VAL1]] to %[[ARR_ACCESS]] : i32, !fir.ref<i32>
    ! CHECK-NEXT: omp.yield
    ! CHECK-NEXT: }

    ! CHECK-NEXT: omp.terminator
    ! CHECK-NEXT: }
    ! CHECK-NEXT: omp.terminator
    ! CHECK-NEXT: }
    ! CHECK-NEXT: omp.terminator
    ! CHECK-NEXT: }
    ! CHECK-NEXT: omp.terminator
    ! CHECK-NEXT: }
    ! CHECK-NEXT: omp.terminator
    ! CHECK-NEXT: }

    do concurrent (i=1:10)
        a(i) = i
    end do

    ! CHECK-NOT: fir.do_loop
end program do_concurrent_basic
