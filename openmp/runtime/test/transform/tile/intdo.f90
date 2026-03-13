! This test checks lowering of OpenMP tile directive

! RUN: %flang %flags %openmp_flags -fopenmp-version=51 %s -o %t.exe
! RUN: %t.exe | FileCheck %s --match-full-lines


program tile_intdo
  integer i, j
  print *, 'do'

  !$OMP TILE SIZES(2,2)
  do i=7, 18, 3
    do j=7, 19, 3
      print '("i=", I0, " j=", I0)', i, j
    end do
  end do
  !$OMP END TILE

  print *, 'done'
end program


! CHECK:      do

! Complete tile
! CHECK-NEXT: i=7 j=7
! CHECK-NEXT: i=7 j=10
! CHECK-NEXT: i=10 j=7
! CHECK-NEXT: i=10 j=10

! Complete tile
! CHECK-NEXT: i=7 j=13
! CHECK-NEXT: i=7 j=16
! CHECK-NEXT: i=10 j=13
! CHECK-NEXT: i=10 j=16

! Partial tile
! CHECK-NEXT: i=7 j=19
! CHECK-NEXT: i=10 j=19

! Complete tile
! CHECK-NEXT: i=13 j=7
! CHECK-NEXT: i=13 j=10
! CHECK-NEXT: i=16 j=7
! CHECK-NEXT: i=16 j=10

! Complete tile
! CHECK-NEXT: i=13 j=13
! CHECK-NEXT: i=13 j=16
! CHECK-NEXT: i=16 j=13
! CHECK-NEXT: i=16 j=16

! Partial tile
! CHECK-NEXT: i=13 j=19
! CHECK-NEXT: i=16 j=19

! CHECK-NEXT: done
