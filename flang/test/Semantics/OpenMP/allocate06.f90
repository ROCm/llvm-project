! REQUIRES: openmp_runtime

! RUN: %python %S/../test_errors.py %s %flang_fc1 %openmp_flags -fopenmp-version=50
! OpenMP Version 5.0
! 2.11.3 allocate Directive 
! List items specified in the allocate directive must not have the ALLOCATABLE attribute unless the directive is associated with an
! allocate statement.

subroutine allocate()
use omp_lib
  integer :: a, b, x
  real, dimension (:,:), allocatable :: darray

  !ERROR: List items specified in the ALLOCATE directive must not have the ALLOCATABLE attribute unless the directive is associated with an ALLOCATE statement
  !$omp allocate(darray) allocator(omp_default_mem_alloc)

  !$omp allocate(darray) allocator(omp_default_mem_alloc)
    allocate(darray(a, b))

end subroutine allocate
