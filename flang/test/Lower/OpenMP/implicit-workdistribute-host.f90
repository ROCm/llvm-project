! End-to-end test that `-fopenmp-implicit-workdistribute=host` causes a
! plain Fortran array statement (no source-level OpenMP directive) to be
! wrapped in `omp.teams { omp.workdistribute { ... } }` and lowered by the
! rest of the pipeline to `teams { parallel { distribute { wsloop {
! loop_nest } } } }`.
!
! Also asserts:
!   * `device` mode wraps in `omp.target` with implicit `omp.map.info` per
!     live-in, around the same teams/workdistribute nest.
!   * `none` mode (the default) does NOT wrap.

! UNSUPPORTED: system-windows

! RUN: %flang_fc1 -emit-fir -fopenmp -fopenmp-version=60 \
! RUN:   -fopenmp-implicit-workdistribute=host %s -o - 2>&1 \
! RUN:   | FileCheck %s --check-prefix=HOST

! RUN: %flang_fc1 -emit-fir -fopenmp -fopenmp-version=60 \
! RUN:   -fopenmp-implicit-workdistribute=device %s -o - 2>&1 \
! RUN:   | FileCheck %s --check-prefix=DEVICE

! RUN: %flang_fc1 -emit-fir -fopenmp -fopenmp-version=60 \
! RUN:   -fopenmp-implicit-workdistribute=none %s -o - 2>&1 \
! RUN:   | FileCheck %s --check-prefix=NOOP

! RUN: %flang_fc1 -emit-fir -fopenmp -fopenmp-version=60 %s -o - 2>&1 \
! RUN:   | FileCheck %s --check-prefix=NOOP

! HOST-LABEL: func @_QPimplicit_saxpy
! HOST:         omp.teams
! HOST:         omp.parallel
! HOST:         omp.distribute
! HOST:         omp.wsloop
! HOST:         omp.loop_nest

! DEVICE-LABEL: func @_QPimplicit_saxpy
! DEVICE:         omp.map.info
! DEVICE:         omp.target
! DEVICE:         omp.teams

! NOOP-LABEL: func @_QPimplicit_saxpy
! NOOP-NOT:     omp.teams
! NOOP-NOT:     omp.workdistribute
! NOOP-NOT:     omp.parallel
! NOOP-NOT:     omp.distribute
! NOOP-NOT:     omp.wsloop

subroutine implicit_saxpy()
  real :: a
  real, dimension(10) :: x
  real, dimension(10) :: y

  y = a * x + y
end subroutine implicit_saxpy
