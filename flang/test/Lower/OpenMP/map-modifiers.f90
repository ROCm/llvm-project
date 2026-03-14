! RUN: split-file %s %t
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=60 %t/ref-modifiers.f90 -o - | FileCheck %t/ref-modifiers.f90
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=61 %t/attach-modifiers.f90 -o - | FileCheck %t/attach-modifiers.f90

! --- ref-modifiers.f90
subroutine ref_modifiers(x)
  integer, pointer :: x
  ! CHECK: omp.map.info {{.*}} map_clauses(tofrom, ref_ptr)
  !$omp target map(ref_ptr, tofrom: x)
    x = 1
  !$omp end target

  ! CHECK: omp.map.info {{.*}} map_clauses(tofrom, ref_ptee)
  !$omp target map(ref_ptee, tofrom: x)
    x = 2
  !$omp end target

  ! CHECK: omp.map.info {{.*}} map_clauses(tofrom, ref_ptr_ptee)
  !$omp target map(ref_ptr_ptee, tofrom: x)
    x = 3
  !$omp end target
end

! --- attach-modifiers.f90
subroutine attach_modifiers(x)
  integer, pointer :: x
  ! CHECK: omp.map.info {{.*}} map_clauses(tofrom, attach_always)
  !$omp target map(attach(always), tofrom: x)
    x = 1
  !$omp end target

  ! CHECK: omp.map.info {{.*}} map_clauses(tofrom, attach_never)
  !$omp target map(attach(never), tofrom: x)
    x = 2
  !$omp end target

  ! CHECK: omp.map.info {{.*}} map_clauses(tofrom, attach_auto)
  !$omp target map(attach(auto), tofrom: x)
    x = 3
  !$omp end target
end
