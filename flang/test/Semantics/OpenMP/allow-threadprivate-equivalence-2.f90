!RUN: %python %S/../test_errors.py %s %flang -Werror -fopenmp -famd-allow-threadprivate-equivalence

subroutine f
  integer, save :: y
  integer :: x
  !WARNING: A variable in a THREADPRIVATE directive cannot appear in an EQUIVALENCE statement
  !$omp threadprivate(x)
  equivalence(x, y)
end

