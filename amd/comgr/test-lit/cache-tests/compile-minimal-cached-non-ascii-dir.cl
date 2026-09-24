// COM: Same as compile-minimal-cached but with a non-ASCII cache directory path
// RUN: rm -fr "%t.cache-Gökalp"
//
// RUN: export AMD_COMGR_EMIT_VERBOSE_LOGS=1
// RUN: export AMD_COMGR_REDIRECT_LOGS=stdout
// RUN: export AMD_COMGR_CACHE=1
//
// COM: Run once and check that the cache directory exists and it has more than
// COM:    1 element (one for the cache tag, one or more for the cached
// COM:    commands) and that pruning didn't fail
// RUN: AMD_COMGR_CACHE_DIR="%t.cache-Gökalp" compile-opencl-minimal \
// RUN:    %S/../compile-minimal.cl %t_a.bin 1.2 > %t_a.log
// RUN: %FileCheck --check-prefix=STORED %s < %t_a.log
// RUN: %FileCheck --check-prefix=NOPRUNE %s < %t_a.log
// RUN: %llvm-objdump -d %t_a.bin | %FileCheck %S/../compile-minimal.cl
//
// COM: One element for the tag, one for cli->bc, one for bc->obj another
// COM: for obj->exec. No elements for src->cli since this is not supported.
// RUN: [ -d "%t.cache-Gökalp" ]
// RUN: COUNT_BEFORE=$(ls "%t.cache-Gökalp" | wc -l)
// RUN: [ 4 -eq $COUNT_BEFORE ]
//
// COM: Run again and check that the stored entries are used
// RUN: AMD_COMGR_CACHE_DIR="%t.cache-Gökalp" compile-opencl-minimal \
// RUN:    %S/../compile-minimal.cl %t_b.bin 1.2 > %t_b.log
// RUN: %FileCheck --check-prefix=FOUND %s < %t_b.log
// RUN: %FileCheck --check-prefix=NOPRUNE %s < %t_b.log
// RUN: %llvm-objdump -d %t_b.bin | %FileCheck %S/../compile-minimal.cl
//
// RUN: COUNT_AFTER=$(ls "%t.cache-Gökalp" | wc -l)
// RUN: [ $COUNT_AFTER = $COUNT_BEFORE ]

// COM: check that the first run stores entries
// STORED: Comgr cache: stored entry

// COM: check that the second run finds the entries
// FOUND: Comgr cache: found entry

// COM: check that pruning did not cause failure
// NOPRUNE-NOT: when pruning the cache
