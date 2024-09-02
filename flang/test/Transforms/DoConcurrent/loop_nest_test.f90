! Tests loop-nest detection algorithm for do-concurrent mapping.

! REQUIRES: asserts

! RUN: %flang_fc1 -emit-hlfir  -fopenmp -fdo-concurrent-parallel=host \
! RUN:   -mmlir -debug %s -o - 2> %t.log || true

! RUN: FileCheck %s < %t.log

program main
  implicit none

contains

subroutine foo(n)
  implicit none
  integer :: n, m
  integer :: i, j, k
  integer :: x
  integer, dimension(n) :: a
  integer, dimension(n, n, n) :: b

  ! NOTE This for sure is a perfect loop nest. However, the way `do-concurrent`
  ! loops are now emitted by flang is probably not correct. This is being looked
  ! into at the moment and once we have flang emitting proper loop headers, we
  ! will revisit this.
  !
  ! CHECK:      Loop pair starting at location loc("{{.*}}":[[# @LINE + 2]]:{{.*}})
  ! CHECK-SAME: is not perfectly nested
  do concurrent(i=1:n, j=1:bar(n*m, n/m))
    a(i) = n
  end do

  ! NOTE same as above.
  !
  ! CHECK:      Loop pair starting at location loc("{{.*}}":[[# @LINE + 2]]:{{.*}})
  ! CHECK-SAME: is not perfectly nested
  do concurrent(i=bar(n, x):n, j=1:bar(n*m, n/m))
    a(i) = n
  end do

  ! NOTE This is **not** a perfect nest since the inner call to `bar` will allocate
  ! memory for the temp results of `n*m` and `n/m` **inside** the outer loop.
  !
  ! CHECK:      Loop pair starting at location loc("{{.*}}":[[# @LINE + 2]]:{{.*}})
  ! CHECK-SAME: is not perfectly nested
  do concurrent(i=bar(n, x):n)
    do concurrent(j=1:bar(n*m, n/m))
      a(i) = n
    end do
  end do

  ! CHECK:      Loop pair starting at location loc("{{.*}}":[[# @LINE + 2]]:{{.*}})
  ! CHECK-SAME: is not perfectly nested
  do concurrent(i=1:n)
    x = 10
    do concurrent(j=1:m)
      b(i,j,k) = i * j + k
    end do
  end do

  ! CHECK:      Loop pair starting at location loc("{{.*}}":[[# @LINE + 2]]:{{.*}})
  ! CHECK-SAME: is not perfectly nested
  do concurrent(i=1:n)
    do concurrent(j=1:m)
      b(i,j,k) = i * j + k
    end do
    x = 10
  end do

  ! CHECK:      Loop pair starting at location loc("{{.*}}":[[# @LINE + 2]]:{{.*}})
  ! CHECK-SAME: is perfectly nested
  do concurrent(i=1:n)
    do concurrent(j=1:m)
      b(i,j,k) = i * j + k
      x = 10
    end do
  end do
end subroutine

pure function bar(n, m)
    implicit none
    integer, intent(in) :: n, m
    integer :: bar

    bar = n + m
end function

end program main
