! RUN: %python %S/../test_errors.py %s %flang -fopenmp  -fopenmp-version=50
! Regression test for #143229

!$omp parallel
do i = 1, 2
!ERROR: invalid branch into an OpenMP structured block
!ERROR: invalid branch leaving an OpenMP structured block
  goto 10
end do
!$omp master
10 print *, i
!$omp end master
!$omp end parallel
end
