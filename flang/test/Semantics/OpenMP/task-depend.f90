! RUN: %python %S/../test_errors.py %s %flang -fopenmp

program test
! ERROR: 'task-dependence-type' modifier is required
! ERROR: A DEPEND clause on a TASK construct must have a valid task dependence type
!$omp task depend(ii)
!$omp end task
end

