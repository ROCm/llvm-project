// REQUIRES: comgr-hotswap-transpile

// RUN: printf '\x7fELF\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00' > \
// RUN:        %t.elf
// RUN: hotswap-transpile | %FileCheck --check-prefix=NULL %s
// RUN: hotswap-transpile %t.elf not-a-valid-isa also-not-valid \
// RUN:   | %FileCheck --check-prefix=BADISA %s
// RUN: hotswap-transpile %t.elf amdgcn-amd-amdhsa--gfx1250 \
// RUN:                   amdgcn-amd-amdhsa--gfx942 --zero-size \
// RUN:   | %FileCheck --check-prefix=ZEROSIZE %s
// RUN: hotswap-transpile %t.elf amdgcn-amd-amdhsa--gfx1250 \
// RUN:                   amdgcn-amd-amdhsa--gfx942 --wrong-kind \
// RUN:   | %FileCheck --check-prefix=WRONGKIND %s
// RUN: hotswap-transpile %t.elf amdgcn-amd-amdhsa--gfx1250 \
// RUN:                   amdgcn-amd-amdhsa--gfx942 --test-bad-options-version \
// RUN:   2>&1 | %FileCheck --check-prefix=BADOPTIONS %s
// RUN: hotswap-transpile %t.elf amdgcn-amd-amdhsa--gfx1250 \
// RUN:                   amdgcn-amd-amdhsa--gfx942 \
// RUN:                   --test-null-kernel-name-option \
// RUN:   2>&1 | %FileCheck --check-prefix=BAD-KERNEL-NAME %s
// RUN: hotswap-transpile %t.elf amdgcn-amd-amdhsa--gfx1250 \
// RUN:                   amdgcn-amd-amdhsa--gfx942 \
// RUN:                   --test-empty-kernel-name-option \
// RUN:   2>&1 | %FileCheck --check-prefix=BAD-KERNEL-NAME %s
// RUN: hotswap-transpile %t.elf amdgcn-amd-amdhsa--gfx1250 \
// RUN:                   amdgcn-amd-amdhsa--gfx942 \
// RUN:                   --test-invalid-opt-level \
// RUN:   2>&1 | %FileCheck --check-prefix=BAD-OPT-LEVEL %s
// RUN: hotswap-transpile %t.elf amdgcn-amd-amdhsa--gfx1250 \
// RUN:                   amdgcn-amd-amdhsa--gfx942 \
// RUN:   | %FileCheck --check-prefix=NOKERNELS %s
// RUN: hotswap-transpile %S/vecadd_gfx950.co \
// RUN:                   amdgcn-amd-amdhsa--gfx950 \
// RUN:                   amdgcn-amd-amdhsa--gfx942 \
// RUN:                   --output=%t.gfx942.co \
// RUN:   | %FileCheck --check-prefix=VECADD %s
// RUN: %llvm-readelf -h %S/vecadd_gfx950.co \
// RUN:   | %FileCheck --check-prefix=SRCISA %s
// RUN: %llvm-readelf -h %t.gfx942.co \
// RUN:   | %FileCheck --check-prefix=TGTISA %s
// RUN: %llvm-objdump --syms %t.gfx942.co \
// RUN:   | %FileCheck --check-prefix=TGTSYM %s
// RUN: env HSA_HOTSWAP_TRANSLATE_KERNEL=vecadd hotswap-transpile \
// RUN:                   %S/vecadd_gfx950.co \
// RUN:                   amdgcn-amd-amdhsa--gfx950 \
// RUN:                   amdgcn-amd-amdhsa--gfx942 \
// RUN:                   --output=%t.single.gfx942.co \
// RUN:   | %FileCheck --check-prefix=SINGLE %s
// RUN: %llvm-objdump --syms %t.single.gfx942.co \
// RUN:   | %FileCheck --check-prefix=SINGLE-SYM %s
// RUN: env HSA_HOTSWAP_TRANSLATE_KERNEL=definitely_not_vecadd \
// RUN:     hotswap-transpile %S/vecadd_gfx950.co \
// RUN:                   amdgcn-amd-amdhsa--gfx950 \
// RUN:                   amdgcn-amd-amdhsa--gfx942 \
// RUN:   | %FileCheck --check-prefix=SINGLE-MISSING %s
// RUN: env HSA_HOTSWAP_TRANSLATE_KERNEL=definitely_not_vecadd \
// RUN:     hotswap-transpile %S/vecadd_gfx950.co \
// RUN:                   amdgcn-amd-amdhsa--gfx950 \
// RUN:                   amdgcn-amd-amdhsa--gfx942 \
// RUN:                   --use-options-api \
// RUN:   | %FileCheck --check-prefix=OPTIONS-API %s
// RUN: rm -rf %t.cache
// RUN: env HSA_HOTSWAP_CACHE_DIR=%t.cache hotswap-transpile \
// RUN:                   %S/vecadd_gfx950.co \
// RUN:                   amdgcn-amd-amdhsa--gfx950 \
// RUN:                   amdgcn-amd-amdhsa--gfx942 \
// RUN:   | %FileCheck --check-prefix=CACHEMISS %s
// RUN: env HSA_HOTSWAP_CACHE_DIR=%t.cache hotswap-transpile \
// RUN:                   %S/vecadd_gfx950.co \
// RUN:                   amdgcn-amd-amdhsa--gfx950 \
// RUN:                   amdgcn-amd-amdhsa--gfx942 \
// RUN:   | %FileCheck --check-prefix=CACHEHIT %s
// RUN: rm -rf %t.hip-cache
// RUN: env HSA_HOTSWAP_CACHE_DIR=%t.hip-cache hotswap-transpile \
// RUN:                   %S/vecadd_gfx950.co \
// RUN:                   amdgcn-amd-amdhsa--gfx950 \
// RUN:                   amdgcn-amd-amdhsa--gfx942 \
// RUN:   > %t.nohip.out
// RUN: %FileCheck --check-prefix=HIPKEYBASE %s < %t.nohip.out
// RUN: sed -n 's/.*cache_key=\([0-9a-f][0-9a-f]*\).*/\1/p' %t.nohip.out > %t.nohip.key
// RUN: env HSA_HOTSWAP_CACHE_DIR=%t.hip-cache \
// RUN:     HSA_HOTSWAP_ASSUME_HIP_GLOBAL_OFFSET_ZERO=1 hotswap-transpile \
// RUN:                   %S/vecadd_gfx950.co \
// RUN:                   amdgcn-amd-amdhsa--gfx950 \
// RUN:                   amdgcn-amd-amdhsa--gfx942 \
// RUN:   > %t.hip.out
// RUN: %FileCheck --check-prefix=HIPKEYSAME %s < %t.hip.out
// RUN: sed -n 's/.*cache_key=\([0-9a-f][0-9a-f]*\).*/\1/p' %t.hip.out > %t.hip.key
// RUN: cmp -s %t.nohip.key %t.hip.key
// RUN: rm -rf %t.opt-cache
// RUN: env HSA_HOTSWAP_CACHE_DIR=%t.opt-cache hotswap-transpile \
// RUN:                   %S/vecadd_gfx950.co \
// RUN:                   amdgcn-amd-amdhsa--gfx950 \
// RUN:                   amdgcn-amd-amdhsa--gfx942 \
// RUN:   > %t.default-opt.out
// RUN: %FileCheck --check-prefix=OPTKEYBASE %s < %t.default-opt.out
// RUN: sed -n 's/.*cache_key=\([0-9a-f][0-9a-f]*\).*/\1/p' \
// RUN:   %t.default-opt.out > %t.default-opt.key
// RUN: env HSA_HOTSWAP_CACHE_DIR=%t.opt-cache \
// RUN:     hotswap-transpile %S/vecadd_gfx950.co \
// RUN:                   amdgcn-amd-amdhsa--gfx950 \
// RUN:                   amdgcn-amd-amdhsa--gfx942 \
// RUN:                   -O0 \
// RUN:   > %t.o0.out
// RUN: %FileCheck --check-prefix=OPTKEYBASE %s < %t.o0.out
// RUN: sed -n 's/.*cache_key=\([0-9a-f][0-9a-f]*\).*/\1/p' \
// RUN:   %t.o0.out > %t.o0.key
// RUN: ! cmp -s %t.default-opt.key %t.o0.key

// NULL: NULL_ARGS: INVALID_ARGUMENT
// BADISA: RESULT: INVALID_ARGUMENT
// ZEROSIZE: RESULT: INVALID_ARGUMENT
// WRONGKIND: RESULT: INVALID_ARGUMENT
// BADOPTIONS-DAG: unsupported hotswap options version 999 (expected 2)
// BADOPTIONS-DAG: RESULT: INVALID_ARGUMENT
// BAD-KERNEL-NAME-DAG: USE_KERNEL_NAME without a kernel name
// BAD-KERNEL-NAME-DAG: RESULT: INVALID_ARGUMENT
// BAD-OPT-LEVEL-DAG: opt_level must be between 0 and 3
// BAD-OPT-LEVEL-DAG: RESULT: INVALID_ARGUMENT
// NOKERNELS: RESULT: ERROR
// VECADD: RESULT: SUCCESS bytes={{[1-9][0-9]*}}
// SRCISA: Flags: {{.*}}gfx950
// TGTISA:     Flags: {{.*}}gfx942
// TGTISA-NOT: gfx950
// TGTSYM: vecadd
// SINGLE-DAG: RESULT: SUCCESS bytes={{[1-9][0-9]*}}
// SINGLE-DAG: kernel_name=vecadd
// SINGLE-SYM: vecadd
// SINGLE-MISSING: RESULT: ERROR
// SINGLE-MISSING-DAG: success=0
// SINGLE-MISSING-DAG: kernel_name=definitely_not_vecadd
// OPTIONS-API-DAG: RESULT: SUCCESS bytes={{[1-9][0-9]*}}
// OPTIONS-API-DAG: kernel_name= lifted=
// CACHEMISS-DAG: cache_hit=0
// CACHEMISS-DAG: cache_lookup=miss
// CACHEMISS-DAG: cache_write=success
// CACHEHIT-DAG: cache_hit=1
// CACHEHIT-DAG: cache_lookup=hit
// CACHEHIT-DAG: cache_write=not_attempted
// HIPKEYBASE-DAG: RESULT: SUCCESS bytes={{[1-9][0-9]*}}
// HIPKEYBASE-DAG: cache_hit=0
// HIPKEYBASE-DAG: cache_lookup=miss
// HIPKEYBASE-DAG: cache_write=success
// ASSUME_HIP_GLOBAL_OFFSET_ZERO is accepted and ignored: source hidden
// arguments, hidden_global_offset_{x,y,z} included, are read where the target
// runtime writes them rather than synthesized, so there is nothing left for the
// option to change. It must therefore NOT be cache-key material -- the run
// below hits the entry the run above wrote, with an identical key.
// HIPKEYSAME-DAG: RESULT: SUCCESS bytes={{[1-9][0-9]*}}
// HIPKEYSAME-DAG: cache_hit=1
// HIPKEYSAME-DAG: cache_lookup=hit
// HIPKEYSAME-DAG: cache_write=not_attempted
// OPTKEYBASE-DAG: RESULT: SUCCESS bytes={{[1-9][0-9]*}}
// OPTKEYBASE-DAG: cache_hit=0
// OPTKEYBASE-DAG: cache_lookup=miss
// OPTKEYBASE-DAG: cache_write=success
