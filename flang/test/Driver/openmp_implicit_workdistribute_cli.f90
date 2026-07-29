! UNSUPPORTED: system-windows

! Smoke test for `-fopenmp-implicit-workdistribute=`. Verifies the flag is
! recognized by both flang and bbc, forwarded by the driver to -fc1, and that
! invalid usage produces the expected diagnostics.

! RUN: %flang --help | FileCheck %s --check-prefix=FLANG
! FLANG:      -fopenmp-implicit-workdistribute=<value>
! FLANG-NEXT:   Implicitly wrap array statements in OpenMP workdistribute [none|host|device]

! RUN: bbc --help | FileCheck %s --check-prefix=BBC
! BBC:      -fopenmp-implicit-workdistribute=<string>
! BBC-SAME:   Implicitly wrap array statements in OpenMP workdistribute [none|host|device]

! Driver forwards the option to -fc1.
! RUN: %flang -### -fopenmp -fopenmp-implicit-workdistribute=device %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=FORWARD
! FORWARD: "-fc1"
! FORWARD-SAME: "-fopenmp-implicit-workdistribute=device"

! Without -fopenmp the option is rejected with a warning and silently disabled.
! RUN: %flang -c -fopenmp-implicit-workdistribute=host %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=NEEDOMP
! NEEDOMP: warning: OpenMP must be enabled (with `-fopenmp`)

! Invalid value rejected.
! RUN: not %flang -c -fopenmp -fopenmp-implicit-workdistribute=devic,e %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=BADVAL
! BADVAL: error: invalid value 'devic,e' in '-fopenmp-implicit-workdistribute{{.*}}'

program test_cli
end program
