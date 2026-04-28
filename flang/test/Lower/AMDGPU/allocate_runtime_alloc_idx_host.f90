! RUN: %flang -fopenmp-default-allocate=host -S -emit-llvm --offload-targets=amdgcn-amd-amdhsa -o - %s | FileCheck %s

subroutine allocate_deallocate()
  real, allocatable :: x
! CHECK-NOT: call void @_FortranAOpenMPAllocatableSetAllocIdx
! CHECK: call i32 @_FortranAAllocatableAllocate
  allocate(x)

! CHECK: call i32 @_FortranAAllocatableDeallocate
  deallocate(x)
end subroutine

subroutine test_allocatable_scalar(a)
  real, save, allocatable :: x1, x2
  real :: a

! CHECK-NOT: call void @_FortranAOpenMPAllocatableSetAllocIdx
! CHECK: call i32 @_FortranAAllocatableAllocateSource
  allocate(x1, x2, source = a)
end
