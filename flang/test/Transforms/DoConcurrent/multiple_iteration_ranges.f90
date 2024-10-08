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

! COMMON: func.func @_QQmain

! DEVICE: %[[DUPLICATED_C1_1:.*]] = arith.constant 1 : i32
! DEVICE: %[[DUPLICATED_LB_I:.*]] = fir.convert %[[DUPLICATED_C1_1]] : (i32) -> index
! DEVICE: %[[DUPLICATED_C10:.*]] = arith.constant 10 : i32
! DEVICE: %[[DUPLICATED_UB_I:.*]] = fir.convert %[[DUPLICATED_C10]] : (i32) -> index
! DEVICE: %[[DUPLICATED_STEP_I:.*]] = arith.constant 1 : index

! DEVICE: %[[C1_1:.*]] = arith.constant 1 : i32
! DEVICE: %[[HOST_LB_I:.*]] = fir.convert %[[C1_1]] : (i32) -> index
! DEVICE: %[[C10:.*]] = arith.constant 10 : i32
! DEVICE: %[[HOST_UB_I:.*]] = fir.convert %[[C10]] : (i32) -> index
! DEVICE: %[[HOST_STEP_I:.*]] = arith.constant 1 : index

! DEVICE: %[[C1_2:.*]] = arith.constant 1 : i32
! DEVICE: %[[HOST_LB_J:.*]] = fir.convert %[[C1_2]] : (i32) -> index
! DEVICE: %[[C20:.*]] = arith.constant 20 : i32
! DEVICE: %[[HOST_UB_J:.*]] = fir.convert %[[C20]] : (i32) -> index
! DEVICE: %[[HOST_STEP_J:.*]] = arith.constant 1 : index

! DEVICE: %[[C1_3:.*]] = arith.constant 1 : i32
! DEVICE: %[[HOST_LB_K:.*]] = fir.convert %[[C1_3]] : (i32) -> index
! DEVICE: %[[C30:.*]] = arith.constant 30 : i32
! DEVICE: %[[HOST_UB_K:.*]] = fir.convert %[[C30]] : (i32) -> index
! DEVICE: %[[HOST_STEP_K:.*]] = arith.constant 1 : index

! DEVICE: omp.target host_eval(
! DEVICE-SAME: %[[HOST_LB_I]] -> %[[LB_I:[[:alnum:]]+]],
! DEVICE-SAME: %[[HOST_UB_I]] -> %[[UB_I:[[:alnum:]]+]],
! DEVICE-SAME: %[[HOST_STEP_I]] -> %[[STEP_I:[[:alnum:]]+]],
! DEVICE-SAME: %[[HOST_LB_J]] -> %[[LB_J:[[:alnum:]]+]],
! DEVICE-SAME: %[[HOST_UB_J]] -> %[[UB_J:[[:alnum:]]+]],
! DEVICE-SAME: %[[HOST_STEP_J]] -> %[[STEP_J:[[:alnum:]]+]],
! DEVICE-SAME: %[[HOST_LB_K]] -> %[[LB_K:[[:alnum:]]+]],
! DEVICE-SAME: %[[HOST_UB_K]] -> %[[UB_K:[[:alnum:]]+]],
! DEVICE-SAME: %[[HOST_STEP_K]] -> %[[STEP_K:[[:alnum:]]+]] :
! DEVICE-SAME: index, index, index, index, index, index, index, index, index)
! DEVICE: omp.teams

! HOST-NOT: omp.target
! HOST-NOT: omp.teams

! COMMON: omp.parallel {

! COMMON-NEXT: %[[ITER_VAR_I:.*]] = fir.alloca i32 {bindc_name = "i"}
! COMMON-NEXT: %[[BINDING_I:.*]]:2 = hlfir.declare %[[ITER_VAR_I]] {uniq_name = "_QFEi"}

! COMMON-NEXT: %[[ITER_VAR_J:.*]] = fir.alloca i32 {bindc_name = "j"}
! COMMON-NEXT: %[[BINDING_J:.*]]:2 = hlfir.declare %[[ITER_VAR_J]] {uniq_name = "_QFEj"}

! COMMON-NEXT: %[[ITER_VAR_K:.*]] = fir.alloca i32 {bindc_name = "k"}
! COMMON-NEXT: %[[BINDING_K:.*]]:2 = hlfir.declare %[[ITER_VAR_K]] {uniq_name = "_QFEk"}

! HOST: %[[C1_1:.*]] = arith.constant 1 : i32
! HOST: %[[LB_I:.*]] = fir.convert %[[C1_1]] : (i32) -> index
! HOST: %[[C10:.*]] = arith.constant 10 : i32
! HOST: %[[UB_I:.*]] = fir.convert %[[C10]] : (i32) -> index
! HOST: %[[STEP_I:.*]] = arith.constant 1 : index

! HOST: %[[C1_2:.*]] = arith.constant 1 : i32
! HOST: %[[LB_J:.*]] = fir.convert %[[C1_2]] : (i32) -> index
! HOST: %[[C20:.*]] = arith.constant 20 : i32
! HOST: %[[UB_J:.*]] = fir.convert %[[C20]] : (i32) -> index
! HOST: %[[STEP_J:.*]] = arith.constant 1 : index

! HOST: %[[C1_3:.*]] = arith.constant 1 : i32
! HOST: %[[LB_K:.*]] = fir.convert %[[C1_3]] : (i32) -> index
! HOST: %[[C30:.*]] = arith.constant 30 : i32
! HOST: %[[UB_K:.*]] = fir.convert %[[C30]] : (i32) -> index
! HOST: %[[STEP_K:.*]] = arith.constant 1 : index

! DEVICE: omp.distribute

! COMMON: omp.wsloop {
! COMMON-NEXT: omp.loop_nest
! COMMON-SAME:   (%[[ARG0:[^[:space:]]+]], %[[ARG1:[^[:space:]]+]], %[[ARG2:[^[:space:]]+]])
! COMMON-SAME:   : index = (%[[LB_I]], %[[LB_J]], %[[LB_K]])
! COMMON-SAME:     to (%[[UB_I]], %[[UB_J]], %[[UB_K]]) inclusive
! COMMON-SAME:     step (%[[STEP_I]], %[[STEP_J]], %[[STEP_K]]) {

! COMMON-NEXT: %[[IV_IDX_I:.*]] = fir.convert %[[ARG0]]
! COMMON-NEXT: fir.store %[[IV_IDX_I]] to %[[BINDING_I]]#1

! COMMON-NEXT: %[[IV_IDX_J:.*]] = fir.convert %[[ARG1]]
! COMMON-NEXT: fir.store %[[IV_IDX_J]] to %[[BINDING_J]]#1

! COMMON-NEXT: %[[IV_IDX_K:.*]] = fir.convert %[[ARG2]]
! COMMON-NEXT: fir.store %[[IV_IDX_K]] to %[[BINDING_K]]#1

! COMMON:      omp.yield
! COMMON-NEXT: }
! COMMON-NEXT: }

! HOST-NEXT: omp.terminator
! HOST-NEXT: }
