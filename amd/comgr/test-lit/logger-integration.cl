// COM: Exercise AMD_COMGR_REDIRECT_LOGS and AMD_COMGR_LOG_LEVEL end-to-end
// COM: through amd_comgr_do_action: redirected logs land in the named file, and
// COM: the level controls their verbosity. The per-action debug header
// COM: ("amd_comgr_do_action:") is emitted only at the debug level (4).

// COM: Debug level (4): the redirect file receives the debug header.
// RUN: AMD_COMGR_LOG_LEVEL=4 AMD_COMGR_REDIRECT_LOGS=%t.debug.log \
// RUN:   compile-opencl-minimal %s %t.bin 1.2
// RUN: grep 'amd_comgr_do_action:' %t.debug.log

// COM: Error level (1): the debug header is suppressed from the redirect file.
// RUN: AMD_COMGR_LOG_LEVEL=1 AMD_COMGR_REDIRECT_LOGS=%t.error.log \
// RUN:   compile-opencl-minimal %s %t.bin 1.2
// RUN: not grep 'amd_comgr_do_action:' %t.error.log

void kernel add(__global float *A, __global float *B, __global float *C) {
    *C = *A + *B;
}
