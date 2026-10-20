## OVERVIEW

   gpurun: Application process launch utility for GPUs
           This utility ensures the process will enable either a single
           GPU or the number specified with -md (multi-device) option.
           It launches the application binary with either the 'taskset'
           or 'numactl' utility so the process only runs on CPU cores
           in the same NUMA domain as the selected GPUs.

           This utility sets environment variable ROCR_VISIBLE_DEVICES
           to selected GPUs ONLY if it was not already set by the
           callers environment AND the number of GPUs is not 1.

$ gpurun -topo
   Topology     Numa: 0   PageSize: [always] madvise never

   GPU     Node  Affinity       UUID               Cores
    0        0       0       GPU-b256278bf70405e2    0-23,96-119
    1        1       1       GPU-a33557394e2c744e    24-47,120-143
    2        2       2       GPU-4f78640baf57e5f0    48-71,144-167
    3        3       3       GPU-b66921701d196e10    72-95,168-191

$ gpurun -help
Usage: gpurun [gpurun_options] Program and options
  -h --help : display help test
  -v        : display gpurun command
  -vv       : display additional debug info
  -vvv      : display more debug info
  -dryrun   : do not run bindings
  -taskset  : use taskset for binding
  -numatcl  : use numactl for binding [default]
  -l        : use numactl --localalloc
  -m        : use numactl --membind[default]
  -md       : Set number of desired devices for multi-device mode, default=1
  -nr       : use numactl ROCR_VISIBLE_DEVICES
  -nm       : use numactl OMPI_COMM_WORLD_LOCAL_RANK
  -topo     : display the topology and exit
  -rocmsmi  : force use of rocm-smi rather than amd-smi
  -amdsmi   : force use of amd-smi rather than rocm-smi
  -nomask   : sets GPURUN_MASK_POLICY to nomask : not yet implemented
  --version : Print version of gpurun and exit

Supported environment variables
  GPURUN_DEVICE_BIAS    Device# to start with [default 0]
  GPURUN_BYPASS         pass through, no bindings

