// COM: Check for any runtime errors with the Comgr Profilier
// RUN: AMD_COMGR_TIME_STATISTICS=1 compile-opencl-minimal %s %t.dflt.bin 1.2
// RUN: grep 'ms$' PerfStatsLog.txt
// RUN: AMD_COMGR_TIME_STATISTICS=1 AMD_COMGR_TIME_STATISTICS_GRANULARITY=ms compile-opencl-minimal %s %t.ms.bin 1.2
// RUN: grep 'ms$' PerfStatsLog.txt
// RUN: AMD_COMGR_TIME_STATISTICS=1 AMD_COMGR_TIME_STATISTICS_GRANULARITY=us compile-opencl-minimal %s %t.us.bin 1.2
// RUN: grep 'us$' PerfStatsLog.txt
// RUN: AMD_COMGR_TIME_STATISTICS=1 AMD_COMGR_TIME_STATISTICS_GRANULARITY=ns compile-opencl-minimal %s %t.ns.bin 1.2
// RUN: grep 'ns$' PerfStatsLog.txt
// RUN: AMD_COMGR_TIME_STATISTICS=1 AMD_COMGR_TIME_STATISTICS_GRANULARITY=foo compile-opencl-minimal %s %t.ns.bin 1.2
// RUN: grep 'ms$' PerfStatsLog.txt

void kernel add(__global float *A, __global float *B, __global float *C) {
    *C = *A + *B;
}
