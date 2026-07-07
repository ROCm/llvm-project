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
// RUN: %FileCheck --check-prefix=HIPKEYBASE %s < %t.hip.out
// RUN: sed -n 's/.*cache_key=\([0-9a-f][0-9a-f]*\).*/\1/p' %t.hip.out > %t.hip.key
// RUN: ! cmp -s %t.nohip.key %t.hip.key

// NULL: NULL_ARGS: INVALID_ARGUMENT
// BADISA: RESULT: INVALID_ARGUMENT
// ZEROSIZE: RESULT: INVALID_ARGUMENT
// WRONGKIND: RESULT: INVALID_ARGUMENT
// NOKERNELS: RESULT: ERROR
// VECADD: RESULT: SUCCESS bytes={{[1-9][0-9]*}}
// SRCISA: Flags: {{.*}}gfx950
// TGTISA:     Flags: {{.*}}gfx942
// TGTISA-NOT: gfx950
// TGTSYM: vecadd
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
