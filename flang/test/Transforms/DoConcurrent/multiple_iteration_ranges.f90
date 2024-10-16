! Tests mapping of a `do concurrent` loop with multiple iteration ranges.

! RUN: split-file %s %t

! RUN: %flang_fc1 -emit-hlfir -fopenmp -fdo-concurrent-parallel=host %t/multi_range.f90 -o - \
! RUN:   | FileCheck %s --check-prefixes=HOST,COMMON

! RUN: %flang_fc1 -emit-hlfir -fopenmp -fdo-concurrent-parallel=device %t/multi_range.f90 -o - \
! RUN:   | FileCheck %s --check-prefixes=DEVICE,COMMON

! RUN: %flang_fc1 -emit-hlfir -fopenmp -fdo-concurrent-parallel=host %t/perfectly_nested.f90 -o - \
! RUN:   | FileCheck %s --check-prefixes=HOST,COMMON

! RUN: %flang_fc1 -emit-hlfir -fopenmp -fdo-concurrent-parallel=device %t/perfectly_nested.f90 -o - \
! RUN:   | FileCheck %s --check-prefixes=DEVICE,COMMON

! RUN: %flang_fc1 -emit-hlfir -fopenmp -fdo-concurrent-parallel=host %t/partially_nested.f90 -o - \
! RUN:   | FileCheck %s --check-prefixes=HOST,COMMON

! RUN: %flang_fc1 -emit-hlfir -fopenmp -fdo-concurrent-parallel=device %t/partially_nested.f90 -o - \
! RUN:   | FileCheck %s --check-prefixes=DEVICE,COMMON

! RUN: %flang_fc1 -emit-hlfir -fopenmp -fdo-concurrent-parallel=device %t/dummy_arg_loop_bounds.f90 -o - \
! RUN:   | FileCheck %s --check-prefixes=DUMMY_UBS

!--- multi_range.f90
program main
   integer, parameter :: n = 10
   integer, parameter :: m = 20
   integer, parameter :: l = 30
   integer :: a(n, m, l)

   do concurrent(i=1:n, j=1:m, k=1:l)
       a(i,j,k) = i * j + k
   end do
end 

!--- perfectly_nested.f90
program main
   integer, parameter :: n = 10
   integer, parameter :: m = 20
   integer, parameter :: l = 30
   integer :: a(n, m, l)

   do concurrent(i=1:n)
     do concurrent(j=1:m)
       do concurrent(k=1:l)
         a(i,j,k) = i * j + k
       end do
     end do
   end do
end

!--- partially_nested.f90
program main
   integer, parameter :: n = 10
   integer, parameter :: m = 20
   integer, parameter :: l = 30
   integer :: a(n, m, l)

   do concurrent(i=1:n, j=1:m)
       do concurrent(k=1:l)
         a(i,j,k) = i * j + k
       end do
   end do
end

! DEVICE: omp.target
! DEVICE: omp.teams

! COMMON: omp.parallel {

! COMMON-NEXT: %[[ITER_VAR_I:.*]] = fir.alloca i32 {bindc_name = "i"}
! COMMON-NEXT: %[[BINDING_I:.*]]:2 = hlfir.declare %[[ITER_VAR_I]] {uniq_name = "_QFEi"}

! COMMON-NEXT: %[[ITER_VAR_J:.*]] = fir.alloca i32 {bindc_name = "j"}
! COMMON-NEXT: %[[BINDING_J:.*]]:2 = hlfir.declare %[[ITER_VAR_J]] {uniq_name = "_QFEj"}

! COMMON-NEXT: %[[ITER_VAR_K:.*]] = fir.alloca i32 {bindc_name = "k"}
! COMMON-NEXT: %[[BINDING_K:.*]]:2 = hlfir.declare %[[ITER_VAR_K]] {uniq_name = "_QFEk"}

! DEVICE: omp.distribute

! COMMON: omp.wsloop {
! COMMON-NEXT: omp.loop_nest
! COMMON-SAME:   (%[[ARG0:[^[:space:]]+]], %[[ARG1:[^[:space:]]+]], %[[ARG2:[^[:space:]]+]])
! COMMON-SAME:   : index = (%{{[^[:space:]]+}}, %{{[^[:space:]]+}}, %{{[^[:space:]]+}})
! COMMON-SAME:     to (%{{[^[:space:]]+}}, %{{[^[:space:]]+}}, %{{[^[:space:]]+}}) inclusive
! COMMON-SAME:     step (%{{[^[:space:]]+}}, %{{[^[:space:]]+}}, %{{[^[:space:]]+}}) {

! COMMON-NEXT: %[[IV_IDX_I:.*]] = fir.convert %[[ARG0]]
! COMMON-NEXT: fir.store %[[IV_IDX_I]] to %[[BINDING_I]]#1

! COMMON-NEXT: %[[IV_IDX_J:.*]] = fir.convert %[[ARG1]]
! COMMON-NEXT: fir.store %[[IV_IDX_J]] to %[[BINDING_J]]#1

! COMMON-NEXT: %[[IV_IDX_K:.*]] = fir.convert %[[ARG2]]
! COMMON-NEXT: fir.store %[[IV_IDX_K]] to %[[BINDING_K]]#1

! COMMON:      omp.yield
! COMMON-NEXT: }
! COMMON-NEXT: omp.terminator
! COMMON-NEXT: }

! HOST-NEXT: omp.terminator
! HOST-NEXT: }

!--- dummy_arg_loop_bounds.f90

subroutine foo(n, m)
   implicit none
   integer :: n, m
   integer :: i, j
   integer :: a(n, m)

   do concurrent(i=1:n, j=1:m)
       a(i,j) = i * j
   end do
end subroutine

! DUMMY_UBS-DAG: omp.map.info {{.*}} {name = "loop.0.lb"}
! DUMMY_UBS-DAG: omp.map.info {{.*}} {name = "loop.0.ub"}
! DUMMY_UBS-DAG: omp.map.info {{.*}} {name = "loop.0.step"}

! DUMMY_UBS-DAG: omp.map.info {{.*}} {name = "loop.1.lb"}
! DUMMY_UBS-DAG: omp.map.info {{.*}} {name = "loop.1.ub"}
! DUMMY_UBS-DAG: omp.map.info {{.*}} {name = "loop.1.step"}


! DUMMY_UBS: omp.target {{.*}} {

! DUMMY_UBS-DAG:   %[[LOOP0_LB_DECL:.*]]:2 = hlfir.declare %arg{{.*}} {uniq_name = "loop.0.lb"}
! DUMMY_UBS-DAG:   %[[LOOP0_UB_DECL:.*]]:2 = hlfir.declare %arg{{.*}} {uniq_name = "loop.0.ub"}
! DUMMY_UBS-DAG:   %[[LOOP0_STEP_DECL:.*]]:2 = hlfir.declare %arg{{.*}} {uniq_name = "loop.0.step"}

! DUMMY_UBS-DAG:   %[[LOOP1_LB_DECL:.*]]:2 = hlfir.declare %arg{{.*}} {uniq_name = "loop.1.lb"}
! DUMMY_UBS-DAG:   %[[LOOP1_UB_DECL:.*]]:2 = hlfir.declare %arg{{.*}} {uniq_name = "loop.1.ub"}
! DUMMY_UBS-DAG:   %[[LOOP1_STEP_DECL:.*]]:2 = hlfir.declare %arg{{.*}} {uniq_name = "loop.1.step"}

! DUMMY_UBS-DAG:   %[[LOOP0_LB:.*]] = fir.load %[[LOOP0_LB_DECL]]#1
! DUMMY_UBS-DAG:   %[[LOOP0_UB:.*]] = fir.load %[[LOOP0_UB_DECL]]#1
! DUMMY_UBS-DAG:   %[[LOOP0_STEP:.*]] = fir.load %[[LOOP0_STEP_DECL]]#1

! DUMMY_UBS-DAG:   %[[LOOP1_LB:.*]] = fir.load %[[LOOP1_LB_DECL]]#1
! DUMMY_UBS-DAG:   %[[LOOP1_UB:.*]] = fir.load %[[LOOP1_UB_DECL]]#1
! DUMMY_UBS-DAG:   %[[LOOP1_STEP:.*]] = fir.load %[[LOOP1_STEP_DECL]]#1

! DUMMY_UBS:       omp.loop_nest (%{{.*}}, %{{.*}}) : index
! DUMMY_UBS-SAME:  = (%[[LOOP0_LB]], %[[LOOP1_LB]])
! DUMMY_UBS-SAME:  to (%[[LOOP0_UB]], %[[LOOP1_UB]])
! DUMMY_UBS-SAME:  inclusive step (%[[LOOP0_STEP]], %[[LOOP1_STEP]])

! DUMMY_UBS: omp.terminator
! DUMMY_UBS: }

