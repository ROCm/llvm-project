!RUN: %flang_fc1 -fopenmp -fopenmp-default-allocate=host -emit-fir %s -o - | FileCheck %s

program amd_alloc_init_host
    !CHECK-NOT: fir.call @_FortranAAMDRegisterAllocator()
end program amd_alloc_init_host
