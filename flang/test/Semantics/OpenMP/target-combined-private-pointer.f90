! RUN: %flang_fc1 -fopenmp -fdebug-dump-symbols %s 2>&1 | FileCheck %s
! Test the data-sharing attribute resolution for a private variable
! on a combined/composite target construct. On a top-level combined target
! construct, an explicitly private pointer/scalar is additionally marked
! OmpImplicit so that lowering privatizes it at the target level (suppressing a
! implicit map and keeping the isolated target region legal). The marker should
! NOT fire for a separated (non-top-level-target) distribute.

! CHECK-LABEL: Subprogram scope: combined_ptr
! CHECK: a (OmpPrivate, OmpExplicit, OmpImplicit): HostAssoc
subroutine combined_ptr()
  real, pointer, contiguous :: a
  integer :: i, n
  n = 100
  !$omp target teams distribute private(a)
  do i = 1, n
  end do
  !$omp end target teams distribute
end subroutine

! CHECK-LABEL: Subprogram scope: separated_ptr
! CHECK: a (OmpPrivate, OmpExplicit): HostAssoc
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

! CHECK-LABEL: Subprogram scope: combined_scalar
! CHECK: s (OmpPrivate, OmpExplicit, OmpImplicit): HostAssoc
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

! CHECK-LABEL: Subprogram scope: target_parallel_do_ptr
! CHECK: a (OmpPrivate, OmpExplicit, OmpImplicit): HostAssoc
subroutine target_parallel_do_ptr()
  real, pointer, contiguous :: a
  integer :: i, n
  n = 100
  !$omp target parallel do private(a)
  do i = 1, n
  end do
  !$omp end target parallel do
end subroutine

! CHECK-LABEL: Subprogram scope: target_only_ptr
! CHECK: a (OmpPrivate, OmpExplicit): HostAssoc
subroutine target_only_ptr()
  real, pointer, contiguous :: a
  integer :: i, n
  n = 100
  !$omp target private(a)
  do i = 1, n
  end do
  !$omp end target
end subroutine

! CHECK-LABEL: Subprogram scope: combined_firstprivate_ptr
! CHECK: a (OmpFirstPrivate, OmpExplicit): HostAssoc
subroutine combined_firstprivate_ptr()
  real, pointer, contiguous :: a
  integer :: i, n
  n = 100
  !$omp target teams distribute firstprivate(a)
  do i = 1, n
  end do
  !$omp end target teams distribute
end subroutine
