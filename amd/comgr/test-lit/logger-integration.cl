// COM: Exercise AMD_COMGR_REDIRECT_LOGS and AMD_COMGR_LOG_LEVEL end-to-end
// COM: through amd_comgr_do_action. Redirection must COPY logs to the named
// COM: file without moving them away from the caller's returned comgr.log, and
// COM: AMD_COMGR_LOG_LEVEL must control verbosity in both destinations. The
// COM: per-action debug header ("amd_comgr_do_action:") is emitted only at the
// COM: debug level (4); the #warning below is a compiler diagnostic, which is
// COM: returned to the caller and copied to the redirect sink at any level.

// COM: Debug level (4): the debug header must appear in BOTH the redirect file
// COM: and the log returned to the caller (copy, not move).
// RUN: AMD_COMGR_LOG_LEVEL=4 AMD_COMGR_REDIRECT_LOGS=%t.debug.log \
// RUN:   logger-redirect %s 1.2 %t.debug.returned.log
// RUN: FileCheck %s < %t.debug.log
// RUN: FileCheck %s < %t.debug.returned.log
// CHECK: amd_comgr_do_action:

// COM: Error level (1): the debug header is suppressed from both destinations,
// COM: but the compiler diagnostic still reaches both (positive assertion that
// COM: the sink is live at level 1, not merely empty).
// RUN: AMD_COMGR_LOG_LEVEL=1 AMD_COMGR_REDIRECT_LOGS=%t.error.log \
// RUN:   logger-redirect %s 1.2 %t.error.returned.log
// RUN: FileCheck --check-prefix=ERROR --implicit-check-not='amd_comgr_do_action:' \
// RUN:   %s < %t.error.log
// RUN: FileCheck --check-prefix=ERROR --implicit-check-not='amd_comgr_do_action:' \
// RUN:   %s < %t.error.returned.log
// ERROR: comgr-logger-integration-marker

#warning comgr-logger-integration-marker
void kernel add(__global float *A, __global float *B, __global float *C) {
    *C = *A + *B;
}
