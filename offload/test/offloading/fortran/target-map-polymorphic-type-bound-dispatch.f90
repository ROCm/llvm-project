! Offload test for type-bound dispatch through a mapped polymorphic
! descriptor.
!
! Dynamic dispatch exercises TypeDescriptor RTTI metadata beyond direct
! derived_type identity, including the device-side binding table referenced
! from the canonical TypeDescriptor global.
!
! REQUIRES: flang, amdgpu
!
! RUN: %libomptarget-compile-fortran-run-and-check-generic

module polymorphic_type_bound_dispatch_mod
  implicit none

  type :: base_t
    integer :: base_value
  contains
    procedure :: value => base_value_fn
  end type base_t

  type, extends(base_t) :: child_t
    integer :: child_value
  contains
    procedure :: value => child_value_fn
  end type child_t

contains

  integer function base_value_fn(self)
    class(base_t), intent(in) :: self
    base_value_fn = self%base_value
  end function base_value_fn

  integer function child_value_fn(self)
    class(child_t), intent(in) :: self
    child_value_fn = self%base_value + self%child_value
  end function child_value_fn

end module polymorphic_type_bound_dispatch_mod

program main
  use polymorphic_type_bound_dispatch_mod
  implicit none

  class(base_t), allocatable :: obj
  integer :: result

  allocate(child_t :: obj)
  obj%base_value = 11

  select type (obj)
  type is (child_t)
    obj%child_value = 31
  class default
    print *, "======= Test Failed! ======="
    stop 1
  end select

  result = -1

  select type (obj)
  type is (child_t)
    !$omp target enter data map(to: obj)
  class default
    print *, "======= Test Failed! ======="
    stop 1
  end select

  !$omp target map(tofrom: obj, result)
    result = obj%value()
    obj%base_value = obj%base_value + 1
  !$omp end target

  select type (obj)
  type is (child_t)
    !$omp target exit data map(from: obj)
  class default
    print *, "======= Test Failed! ======="
    stop 1
  end select

  if (result /= 42) then
    print *, "======= Test Failed! ======="
    stop 1
  end if

  if (obj%base_value /= 12) then
    print *, "======= Test Failed! ======="
    stop 1
  end if

  print *, "======= Test Passed! ======="
end program main

! CHECK: ======= Test Passed! =======


! /home/agozillo/git/aomp24.0/llvm-project/install/bin/flang -fopenmp --offload-arch=gfx942 ../offload/test/offloading/fortran/target-map-polymorphic-type-bound-dispatch.f90 -o test.out
! /home/agozillo/git/aomp24.0/llvm-project/install/bin/flang -S -emit-llvm --offload-device-only -fopenmp --offload-arch=gfx942 ../offload/test/offloading/fortran/target-map-polymorphic-type-bound-dispatch.f90 -o test-dev.ll
! /home/agozillo/git/aomp24.0/llvm-project/install/bin/flang -S -emit-llvm --offload-host-only -fopenmp --offload-arch=gfx942 ../offload/test/offloading/fortran/target-map-polymorphic-type-bound-dispatch.f90 -o test-host.ll
! /home/agozillo/git/aomp24.0/llvm-project/install/bin/flang -fc1 -emit-fir -fopenmp ../offload/test/offloading/fortran/target-map-polymorphic-type-bound-dispatch.f90 -o test.fir
