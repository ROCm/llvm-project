// COM: Exercise AMD_COMGR_CACHE_DIR pointing at a directory whose name holds a
// COM: non-ASCII character (U+00F6, latin small letter o with diaeresis).
// COM:
// COM: What this test proves, and where:
// COM:
// COM: On Windows it has teeth. getenv() returns the value re-encoded in the
// COM: active ANSI code page, so the accented character comes back as a lone
// COM: 0xF6 byte. That is not valid UTF-8, and LLVM's UTF-8 to UTF-16 path
// COM: conversion rejects it, so every cache file open fails and no cache
// COM: directory is produced at all.
// COM:
// COM: On Linux it passes both with and without that Windows-side change, so it
// COM: does not validate the fix here. glibc getenv() hands back the raw bytes
// COM: untouched, and sys::fs::disk_space() (used by the cache pruner) is a
// COM: plain statvfs() on those same bytes with no encoding conversion, see
// COM: llvm/lib/Support/Unix/Path.inc. Running it on Linux only guards against
// COM: a future regression that would also break non-ASCII cache paths here.
//
// COM: The accented character is written literally in the RUN lines below
// COM: because it is the data under test. Comments stay ASCII only, per
// COM: amd/comgr/AGENT_CONVENTIONS.md.

// RUN: rm -fr "%t.cache-Gökalp"
//
// RUN: export AMD_COMGR_EMIT_VERBOSE_LOGS=1
// RUN: export AMD_COMGR_REDIRECT_LOGS=stdout
// RUN: export AMD_COMGR_CACHE=1
//
// COM: First run against a fresh directory: every command misses and is stored.
// RUN: AMD_COMGR_CACHE_DIR="%t.cache-Gökalp" compile-opencl-minimal \
// RUN:    %S/../compile-minimal.cl %t_a.bin 1.2 > %t_a.log
// RUN: %FileCheck --check-prefix=STORED %s < %t_a.log
// RUN: %FileCheck --check-prefix=NOPRUNE %s < %t_a.log
// RUN: %llvm-objdump -d %t_a.bin | %FileCheck %S/../compile-minimal.cl
//
// COM: The directory must exist and hold one element for the cache tag plus one
// COM: each for cli->bc, bc->obj and obj->exec. src->cli is not cached.
// RUN: [ -d "%t.cache-Gökalp" ]
// RUN: COUNT_BEFORE=$(ls "%t.cache-Gökalp" | wc -l)
// RUN: [ 4 -eq $COUNT_BEFORE ]
//
// COM: Second run against the same directory: every command hits.
// RUN: AMD_COMGR_CACHE_DIR="%t.cache-Gökalp" compile-opencl-minimal \
// RUN:    %S/../compile-minimal.cl %t_b.bin 1.2 > %t_b.log
// RUN: %FileCheck --check-prefix=FOUND %s < %t_b.log
// RUN: %FileCheck --check-prefix=NOPRUNE %s < %t_b.log
// RUN: %llvm-objdump -d %t_b.bin | %FileCheck %S/../compile-minimal.cl
//
// RUN: COUNT_AFTER=$(ls "%t.cache-Gökalp" | wc -l)
// RUN: [ $COUNT_AFTER = $COUNT_BEFORE ]

// COM: check that an entry is stored
// STORED: Comgr cache: stored entry

// COM: check that an entry is found
// FOUND: Comgr cache: found entry

// COM: The pruner stats the cache directory and reports a failure to do so as a
// COM: non-fatal "when pruning the cache" log line, so a path that the platform
// COM: cannot resolve would otherwise be swallowed. The pre-existing cache
// COM: tests carry no such guard, which is why a silent prune failure does not
// COM: fail them.
// NOPRUNE-NOT: when pruning the cache
