// COM: Test Comgr version and cache-identifier APIs
// RUN: get-version
// RUN: env AMD_COMGR_REDIRECT_LOGS=stdout get-version --cache-identifier \
// RUN:   | %FileCheck --check-prefix=CACHE-LOG %s
// RUN: not get-version --unknown 2>&1 \
// RUN:   | %FileCheck --check-prefix=UNKNOWN %s
// CACHE-LOG: comgr: amd_comgr_get_cache_identifier: identifier argument is null
// UNKNOWN: FAILED: unknown argument: --unknown
