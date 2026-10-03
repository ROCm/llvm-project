# Traffic-aware OpenMP grid sizing

The Traffic-aware grid size selection policy uses the information collected by
the OpenMPKernelTraffic analysis pass for determining the most suitable launch
geometry for a given GPU kernel. The most important discriminator is the global
memory traffic per iteration in the most memory-heavy loop (nest) of the
corresponding OpenMP kernel.
While **the traffic-aware policy aims to provide an analytical model for grid
size selection**, it still works with heuristics and device-specific thresholds
that have been built empirically through measurements and are subject to
ongoing tuning/refinement. Also note that not *all* thresholds or constants
used by the heuristics are device-specific at the moment. Generic values have
only been made device-specific if backed up by actual measurements.

The policy is **off by default**. The analysis is recorded unconditionally, so
switching it on needs no recompilation, only the following environment
variable:
```
LIBOMPTARGET_TRAFFIC_AWARE_GRID=1
```

## How a grid is chosen

Each OpenMP kernel is ranked into one to the following buckets:

| characteristic property | grid consequence |
|---|---|
| no static shared memory (LDS) | one thread per loop iteration |
| shared memory, <= 8 bytes global memory per iter | 75% of full occupancy |
| shared memory, > 8 bytes global memory per iter, <= saturation threshold | GPU-specific #blocks/compute unit |
| shared memory, > saturation threshold | one block per compute unit |

The absence of shared memory / LDS is a sign that the threads/blocks are
working **independently**, which means that we try to launch as many threads as
possible. Ideally, we get one thread per iteration, so no loop at all.

If a kernel uses shared memory, we look at the "heaviest" loop nest in terms of
accesses to *global* memory. If no loop in a kernel accesses more than 8 bytes
of memory per iteration, we call this kernel **latency-bound** and we aim for
75% occupancy. (We regard 100% occupancy as the number of blocks that can be
resident at once, limited by the hardware and by the kernel's own register and
shared memory usage.)
This occupancy target can be overriden by
`LIBOMPTARGET_TRAFFIC_LATENCY_OCCUPANCY_PCT=<percent>`; 0 keeps the default.

If a kernel accesses more than 8 bytes per iteration in its memory-heaviest
loop nest, we call this kernel **bandwidth-bound**.
Those kernels are again split into two subgroups, depending on whether the
number of bytes that are accessed per iteration are below or above a
device-specific saturation threshold. This theshold can be overriden using
`LIBOMPTARGET_TRAFFIC_BANDWIDTH_SATURATION_BYTES=<bytes per iteration, 0
disables>`.
Below this threshold, we assume that bandwidth is not yet saturated and we
launch a device-specific multiple of the number of CUs this GPU has. This
multiplier can be overriden by `LIBOMPTARGET_TRAFFIC_BANDWIDTH_CU_MULT=<blocks
per compute unit>`; 0 keeps the default.
Above this theshold, we call the kernel **bandwidth-saturating** and launch no
more blocks than the number of CUs.

### Exceptions

The policy gets bypassed by an explicit `num_teams` value, the `OMP_NUM_TEAMS`
environment variable, or the `LIBOMPTARGET_BLOCKS_FOR_LOW_TRIP_COUNT`
environment variable.

## Collecting data

Run the application **twice, both traced**, differing only in the policy:

```
# A: old device heuristics
LIBOMPTARGET_KERNEL_TRACE=1 LIBOMPTARGET_KERNEL_EXE_TIME=1 ./app

# B: traffic-aware policy
LIBOMPTARGET_KERNEL_TRACE=1 LIBOMPTARGET_KERNEL_EXE_TIME=1 \
LIBOMPTARGET_TRAFFIC_AWARE_GRID=1 ./app
```

The logs will contain several lines of information per kernel (identified by
the kernel name). The line related to the traffic-aware policy will contain the
following information:

| field | meaning |
|---|---|
| `arch` | compute unit kind |
| `enabled` | whether the policy was consulted |
| `bytes` | accessed global memory bytes per iteration in the heaviest loop nest |
| `streams` | distinct memory streams those bytes reached |
| `ld`, `st` | access bytes and count split by load/store |
| `ops`, `insts` | compute operations and total instructions in the loop nest |
| `lds` | static group memory / LDS used by the kernel |
| `trip` | loop trip count |
| `regime` | `independent`, `latency`, `bandwidth`, `bandwidth-saturated`, or `none` |
| `policy_teams` | blocks the policy asked for, or 0 if it had no opinion |

Currently, only `bytes`, `lds` and `trip` are actually used by the
implementation of the policy. The rest are recorded in case they turn out to
make a difference and so that the thresholds can be refitted from a log without
re-running anything.
Negative values means the compiler recorded no estimate for that kernel; the
policy then leaves the grid to the old device heuristics.
