! Fortran-equivalent of ../traffic_aware_grid.c
!
! RUN: %libomptarget-compile-fortran-generic -O2
! RUN: env LIBOMPTARGET_TRAFFIC_AWARE_GRID=1 %libomptarget-run-generic 2>&1 \
! RUN:  | %fcheck-generic
! RUN: %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefix=DISABLED
!
! REQUIRES: flang, amdgpu

program main
  use omp_lib
  implicit none
  integer, parameter :: n = 2**18
  integer :: a(n)
  real(8) :: s0(n), s1(n), s2(n), s3(n), s4(n), s5(n)
  real(8) :: s6(n), s7(n), s8(n), s9(n), s10(n), s11(n)
  integer :: i, light_blocks, heavy_blocks
  integer(8) :: isum
  real(8) :: dsum

  a = 1
  s0 = 1; s1 = 1; s2 = 1; s3 = 1; s4 = 1; s5 = 1
  s6 = 1; s7 = 1; s8 = 1; s9 = 1; s10 = 1; s11 = 1

  ! 4 bytes per iteration over one stream: latency bound, wants the device
  ! oversubscribed.
  light_blocks = 0
  isum = 0
  !$omp target teams distribute parallel do map(to: a) &
  !$omp&  reduction(+: isum) reduction(max: light_blocks)
  do i = 1, n
    light_blocks = omp_get_num_teams()
    isum = isum + a(i)
  end do

  ! 96 bytes per iteration over twelve streams: bandwidth bound, wants few
  ! blocks so that their working sets stay resident.
  heavy_blocks = 0
  dsum = 0
  !$omp target teams distribute parallel do &
  !$omp&  map(to: s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11) &
  !$omp&  reduction(+: dsum) reduction(max: heavy_blocks)
  do i = 1, n
    heavy_blocks = omp_get_num_teams()
    dsum = dsum + s0(i) + s1(i) + s2(i) + s3(i) + s4(i) + s5(i) + &
           s6(i) + s7(i) + s8(i) + s9(i) + s10(i) + s11(i)
  end do

  print '(A,I0,A,I0)', 'light=', light_blocks, ' heavy=', heavy_blocks

  ! CHECK: heavy kernel gets fewer blocks
  ! DISABLED: both kernels get the same number of blocks
  if (heavy_blocks < light_blocks) then
    print '(A)', 'heavy kernel gets fewer blocks'
  else if (heavy_blocks == light_blocks) then
    print '(A)', 'both kernels get the same number of blocks'
  else
    print '(A)', 'heavy kernel got MORE blocks'
  end if
end program main
