! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=51 -o - %s | FileCheck %s

! Verify lowering of the OpenMP 5.0+ `!$omp allocators` construct over Fortran
! allocatable arrays.  For each variable listed in the ALLOCATE clause, the
! inner AllocateStmt lowering routes through the Fortran runtime (the
! fir.allocmem inline fast path is suppressed) and emits a
! _FortranAOmpAllocatorStamp(descriptor, handle, align) call immediately
! before _FortranAAllocatableAllocate.  The stamp causes Descriptor::Allocate
! in flang-rt to dispatch through __kmpc_alloc / __kmpc_aligned_alloc with
! the recorded handle and alignment, and Descriptor::Deallocate through
! __kmpc_free with the same handle.
!
! The allocator constants are spelled out as literal integer(kind=8) values
! to avoid depending on omp_lib.mod in the test environment.  The values
! match the predefined OpenMP allocator handles:
!   omp_default_mem_alloc    = 1
!   omp_high_bw_mem_alloc    = 4

subroutine intrinsic_allocator(n)
  integer, intent(in) :: n
  integer, allocatable :: a(:)
  integer(kind=8), parameter :: omp_default_mem_alloc = 1

!$omp allocators allocate(omp_default_mem_alloc: a)
  allocate(a(n))

  deallocate(a)
end subroutine intrinsic_allocator

! The inline fast path is now suppressed for OMP-tagged allocatables, so
! even a plain `integer, allocatable :: a(:)` goes through the runtime
! allocate path.  The stamp call precedes AllocatableAllocate, and the
! handle argument is the allocator constant (1 for omp_default_mem_alloc).
! The matching DEALLOCATE -- which sits textually outside the
! `!$omp allocators` block -- must also route through the runtime
! (_FortranAAllocatableDeallocate) so the descriptor's kOmpAllocatorPos
! allocator-index dispatches the free through __kmpc_free.  Inline
! fir.freemem would call std::free on libomp-managed memory and crash.
! CHECK-LABEL: func.func @_QPintrinsic_allocator
! CHECK: fir.call @_FortranAAllocatableSetBounds(
! CHECK: fir.call @_FortranAOmpAllocatorStamp(
! CHECK: fir.call @_FortranAAllocatableAllocate(
! CHECK-NOT: fir.freemem
! CHECK: fir.call @_FortranAAllocatableDeallocate(

subroutine derived_allocator(n)
  integer, intent(in) :: n
  type :: point
    real :: x, y
  end type point
  type(point), allocatable :: pts(:)
  integer(kind=8), parameter :: omp_high_bw_mem_alloc = 4

!$omp allocators allocate(allocator(omp_high_bw_mem_alloc): pts)
  allocate(pts(n))

  deallocate(pts)
end subroutine derived_allocator

! Derived-type allocatables always take the runtime path; the stamp still
! appears before AllocatableAllocate.
! CHECK-LABEL: func.func @_QPderived_allocator
! CHECK: fir.call @_FortranAAllocatableSetBounds(
! CHECK: fir.call @_FortranAOmpAllocatorStamp(
! CHECK: fir.call @_FortranAAllocatableAllocate(

subroutine aligned_allocator(n)
  integer, intent(in) :: n
  real, allocatable :: buf(:)
  integer(kind=8), parameter :: omp_default_mem_alloc = 1

!$omp allocators allocate(allocator(omp_default_mem_alloc), align(64): buf)
  allocate(buf(n))

  deallocate(buf)
end subroutine aligned_allocator

! With the ALIGN modifier the third argument to OmpAllocatorStamp carries
! the alignment (here 64).  The runtime will dispatch through
! __kmpc_aligned_alloc, and the matching DEALLOCATE goes through
! _FortranAAllocatableDeallocate so __kmpc_free can recover the
! original allocator from the chunk header.
! CHECK-LABEL: func.func @_QPaligned_allocator
! CHECK: fir.call @_FortranAAllocatableSetBounds(
! CHECK: fir.call @_FortranAOmpAllocatorStamp(
! CHECK-SAME: i64, i64) -> ()
! CHECK: fir.call @_FortranAAllocatableAllocate(
! CHECK-NOT: fir.freemem
! CHECK: fir.call @_FortranAAllocatableDeallocate(

subroutine multi_object(n, m)
  integer, intent(in) :: n, m
  type :: row
    real :: v(4)
  end type row
  type(row), allocatable :: a(:), b(:,:)
  integer(kind=8), parameter :: omp_default_mem_alloc = 1
  integer(kind=8), parameter :: omp_high_bw_mem_alloc = 4

!$omp allocators allocate(omp_default_mem_alloc: a) &
!$omp&            allocate(omp_high_bw_mem_alloc: b)
  allocate(a(n), b(m, n))

  deallocate(a, b)
end subroutine multi_object

! CHECK-LABEL: func.func @_QPmulti_object
! Each allocatable named in an ALLOCATE clause gets its own stamp call
! immediately before its matching AllocatableAllocate, each carrying the
! allocator handle from its own clause.
! CHECK: fir.call @_FortranAOmpAllocatorStamp(
! CHECK: fir.call @_FortranAAllocatableAllocate(
! CHECK: fir.call @_FortranAOmpAllocatorStamp(
! CHECK: fir.call @_FortranAAllocatableAllocate(
