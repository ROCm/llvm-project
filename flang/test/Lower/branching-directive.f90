!RUN: bbc -emit-hlfir -fopenmp -o - %s | FileCheck %s

!https://github.com/llvm/llvm-project/issues/91526

!CHECK-LABEL: func.func @_QPsimple1
!CHECK:   cf.cond_br %{{[0-9]+}}, ^bb[[THEN:[0-9]+]], ^bb[[ELSE:[0-9]+]]
!CHECK: ^bb[[THEN]]:
!CHECK:   omp.parallel
!CHECK:   cf.br ^bb[[ENDIF:[0-9]+]]
!CHECK: ^bb[[ELSE]]:
!CHECK:   fir.call @_FortranAStopStatement
!CHECK:   fir.unreachable
!CHECK: ^bb[[ENDIF]]:
!CHECK:   return

subroutine simple1(y)
  implicit none
  logical, intent(in) :: y
  integer :: i
  if (y) then
    !$omp parallel
    i = 1
    !$omp end parallel
  else
    stop 1
  end if
end subroutine

!CHECK-LABEL: func.func @_QPsimple2
!CHECK:   cf.cond_br %{{[0-9]+}}, ^bb[[THEN:[0-9]+]], ^bb[[ELSE:[0-9]+]]
!CHECK: ^bb[[THEN]]:
!CHECK:   omp.parallel
!CHECK:   cf.br ^bb[[ENDIF:[0-9]+]]
!CHECK: ^bb[[ELSE]]:
!CHECK:   fir.call @_FortranAStopStatement
!CHECK:   fir.unreachable
!CHECK: ^bb[[ENDIF]]:
!CHECK:   fir.call @_FortranAioOutputReal64
!CHECK:   return
subroutine simple2(x, yn)
  implicit none
  logical, intent(in) :: yn
  integer, intent(in) :: x
  integer :: i
  real(8) :: E
  E = 0d0

  if (yn) then
     !$omp parallel do private(i) reduction(+:E)
     do i = 1, x
        E = E + i
     end do
     !$omp end parallel do
  else
     stop 1
  end if
  print *, E
end subroutine

