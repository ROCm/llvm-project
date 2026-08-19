! RUN: bbc -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

! Test that a private variables on a combined/composite target construct
! (e.g. `target teams distribute private(a)` or `target parallel do
! private(a)`) is privatized at the target level so that:
!   (1) no implicit map is emitted
!   (2) We are appropriately creating a new private argument for the target
!       region for IsolatedFromAbove and code correctness.

! CHECK-LABEL: func.func @_QPcombined_ptr
! CHECK: omp.target
! CHECK-SAME: private(@_QFcombined_ptrEa_private_box_ptr_f32
! CHECK: omp.teams
! CHECK: omp.distribute
! CHECK-SAME: private(@_QFcombined_ptrEa_private_box_ptr_f32
subroutine combined_ptr()
  real, pointer, contiguous :: a
  integer :: i, n
  n = 100
  !$omp target teams distribute private(a)
  do i = 1, n
  end do
  !$omp end target teams distribute
end subroutine

! CHECK-LABEL: func.func @_QPseparated_ptr
! CHECK: omp.target
! CHECK: omp.teams
! CHECK: omp.distribute
! CHECK-SAME: private(@_QFseparated_ptrEa_private_box_ptr_f32
subroutine separated_ptr()
  real, pointer, contiguous :: a
  integer :: i, n
  n = 100
  !$omp target teams
  !$omp distribute private(a)
  do i = 1, n
  end do
  !$omp end distribute
  !$omp end target teams
end subroutine

! CHECK-LABEL: func.func @_QPcombined_scalar
! CHECK: omp.target
! CHECK-SAME: private(@_QFcombined_scalarEs_private_f32
! CHECK: omp.teams
! CHECK: omp.distribute
! CHECK-SAME: private(@_QFcombined_scalarEs_private_f32
subroutine combined_scalar()
  real :: s
  integer :: i, n
  n = 100
  !$omp target teams distribute private(s)
  do i = 1, n
    s = real(i)
  end do
  !$omp end target teams distribute
end subroutine

! CHECK-LABEL: func.func @_QPtarget_parallel_do_ptr
! CHECK: omp.target
! CHECK-SAME: private(@_QFtarget_parallel_do_ptrEa_private_box_ptr_f32
! CHECK: omp.wsloop
! CHECK-SAME: private(@_QFtarget_parallel_do_ptrEa_private_box_ptr_f32
subroutine target_parallel_do_ptr()
  real, pointer, contiguous :: a
  integer :: i, n
  n = 100
  !$omp target parallel do private(a)
  do i = 1, n
  end do
  !$omp end target parallel do
end subroutine

! CHECK-LABEL: func.func @_QPtarget_only_ptr
! CHECK: omp.target
! CHECK-SAME: private(@_QFtarget_only_ptrEa_private_box_ptr_f32
subroutine target_only_ptr()
  real, pointer, contiguous :: a
  integer :: i, n
  n = 100
  !$omp target private(a)
  do i = 1, n
  end do
  !$omp end target
end subroutine

! CHECK-LABEL: func.func @_QPcombined_firstprivate_ptr
! CHECK: omp.target
! CHECK-SAME: private(@_QFcombined_firstprivate_ptrEa_firstprivate_box_ptr_f32
! CHECK: omp.teams
! CHECK: omp.distribute
! CHECK-SAME: private(@_QFcombined_firstprivate_ptrEa_firstprivate_box_ptr_f32
subroutine combined_firstprivate_ptr()
  real, pointer, contiguous :: a
  integer :: i, n
  n = 100
  !$omp target teams distribute firstprivate(a)
  do i = 1, n
  end do
  !$omp end target teams distribute
end subroutine
