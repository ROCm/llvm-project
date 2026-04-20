! Check that the driver passes -fopenmp-default-allocate= through to fc1
! and only adds -mmlir -use-alloc-runtime for gpu mode.

! RUN: %flang -### -S -fopenmp-default-allocate=gpu %s -o - 2>&1 | FileCheck %s --check-prefix=GPU
! RUN: %flang -### -S -fopenmp-default-allocate=host %s -o - 2>&1 | FileCheck %s --check-prefix=HOST

! GPU: "-fc1"
! GPU-SAME: "-fopenmp-default-allocate=gpu"
! GPU-SAME: "-mmlir" "-use-alloc-runtime"

! HOST: "-fc1"
! HOST-SAME: "-fopenmp-default-allocate=host"
! HOST-NOT: "-mmlir"
! HOST-NOT: "-use-alloc-runtime"

! Check that invalid values are rejected.
! RUN: not %flang_fc1 -fopenmp-default-allocate=invalid -S %s 2>&1 | FileCheck %s --check-prefix=INVALID
! INVALID: error: invalid value 'invalid' in '-fopenmp-default-allocate=invalid'

program fopenmp_default_allocate
    ! do nothing
end program fopenmp_default_allocate
