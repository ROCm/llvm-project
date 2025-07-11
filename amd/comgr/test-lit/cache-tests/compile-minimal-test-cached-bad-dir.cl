// RUN: export AMD_COMGR_CACHE=1
//
// COM: fail to create the cache, but still produce something valid
// RUN: rm -f %t.log
// RUN: echo "not a directory" >  %t.txt
// RUN: AMD_COMGR_CACHE_DIR=%t.txt \
// RUN:   AMD_COMGR_EMIT_VERBOSE_LOGS=1 \
// RUN:   AMD_COMGR_REDIRECT_LOGS=%t.log \
// RUN:     compile-opencl-minimal %S/../compile-minimal-test.cl %t.bin 1.2
// RUN: llvm-objdump -d %t.bin | FileCheck %S/../compile-minimal-test.cl
// RUN: FileCheck --check-prefix=BAD %s < %t.log
// BAD: Failed to open cache file
// BAD-SAME: Not a directory
