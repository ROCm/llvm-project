// COM: Test the hotswap-transpiler-backed amd_comgr_hotswap_transpile API.
// COM: Only registered when comgr was configured with -DCOMGR_ENABLE_HOTSWAP_TRANSPILE=ON;
// COM: the REQUIRES guard makes the test xfail gracefully on builds that
// COM: did not opt in to hotswap transpilation (mirrors the comgr-has-spirv pattern).

// REQUIRES: comgr-hotswap-transpile

// COM: Create a minimal not-an-ELF blob for the negative-path tests. The
// COM: positive end-to-end path below uses a real HSACO checked into the
// COM: hotswap tree; deeper coverage (full corpus, GPU execution) lives in
// COM: the hotswap gtest harness.
// RUN: printf '\x7fELF\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00' > \
// RUN:        %t.elf

// COM: NULL-argument validation (no positional args).
// RUN: hotswap-transpile | %FileCheck --check-prefix=NULL %s
// NULL: NULL_ARGS: INVALID_ARGUMENT

// COM: Malformed ISA strings -> parseTargetIdentifier rejects.
// RUN: hotswap-transpile %t.elf not-a-valid-isa also-not-valid \
// RUN:   | %FileCheck --check-prefix=BADISA %s
// BADISA: RESULT: INVALID_ARGUMENT

// COM: Zero-size input data with a well-formed ISA pair -> rejected
// COM: at the data-pointer gate (mirrors hotswap-rewrite.c's ZEROSIZE).
// RUN: hotswap-transpile %t.elf amdgcn-amd-amdhsa--gfx1250 \
// RUN:                   amdgcn-amd-amdhsa--gfx942 --zero-size \
// RUN:   | %FileCheck --check-prefix=ZEROSIZE %s
// ZEROSIZE: RESULT: INVALID_ARGUMENT

// COM: Wrong data kind (BC instead of EXECUTABLE) -> rejected at the
// COM: kind gate. Both hotswap entry points share this contract.
// RUN: hotswap-transpile %t.elf amdgcn-amd-amdhsa--gfx1250 \
// RUN:                   amdgcn-amd-amdhsa--gfx942 --wrong-kind \
// RUN:   | %FileCheck --check-prefix=WRONGKIND %s
// WRONGKIND: RESULT: INVALID_ARGUMENT

// COM: Well-formed ISA pair plus a non-ELF / no-kernels payload -> the
// COM: hotswap pipeline runs (input passed validation) but
// COM: runPipelineAllKernels surfaces "no kernels" as a pipeline failure,
// COM: which the comgr wrapper maps to AMD_COMGR_STATUS_ERROR. This is the
// COM: cheapest way to assert that the hotswap call site is actually wired
// COM: up without staging a real HSACO into the comgr tree.
// RUN: hotswap-transpile %t.elf amdgcn-amd-amdhsa--gfx1250 \
// RUN:                   amdgcn-amd-amdhsa--gfx942 \
// RUN:   | %FileCheck --check-prefix=NOKERNELS %s
// NOKERNELS: RESULT: ERROR

// COM: End-to-end transpile of a real HSACO. The hotswap tree ships a tiny
// COM: gfx950 vecadd code object (single kernel, no cross-lane / MFMA / wave
// COM: ops) under amd/comgr/hotswap/tests/. Re-lower it to gfx942 and verify
// COM: both the API contract (SUCCESS + non-empty bytes) and the binary
// COM: contents (ELF e_flags retargeted to gfx942, kernel symbol preserved).
// COM: The hotswap backend shells out to llc and ld.lld; the lit site config
// COM: prepends llvm_tools_dir to PATH so both are reachable from the test
// COM: environment.
// RUN: hotswap-transpile %S/vecadd_gfx950.co \
// RUN:                   amdgcn-amd-amdhsa--gfx950 \
// RUN:                   amdgcn-amd-amdhsa--gfx942 \
// RUN:                   --output=%t.gfx942.co \
// RUN:   | %FileCheck --check-prefix=VECADD %s
// VECADD: RESULT: SUCCESS bytes={{[1-9][0-9]*}}

// COM: Sanity-check the source binary's e_flags so the negative check below
// COM: is meaningful: confirm it really is a gfx950 ELF before asserting
// COM: that the *output* is not.
// RUN: %llvm-readelf -h %S/vecadd_gfx950.co \
// RUN:   | %FileCheck --check-prefix=SRCISA %s
// SRCISA: Flags: {{.*}}gfx950

// COM: The transpiled binary's AMDGPU machine ID must reflect the target
// COM: ISA, not the source. llvm-readelf renders EF_AMDGPU_MACH as a
// COM: human-readable "gfxNNN" tag in the Flags line, so we can FileCheck
// COM: it directly without parsing hex. CHECK-NOT pins the negative side:
// COM: a no-op pipeline that just copied the input through would fail this.
// RUN: %llvm-readelf -h %t.gfx942.co \
// RUN:   | %FileCheck --check-prefix=TGTISA %s
// TGTISA:     Flags: {{.*}}gfx942
// TGTISA-NOT: gfx950

// COM: Symbol-level smoke check: the kernel name must round-trip through
// COM: the hotswap raise -> llc -> ld.lld pipeline. If the kernel were
// COM: dropped (e.g. raise failure swallowed silently) the executable
// COM: would link empty and this would fail.
// RUN: %llvm-objdump --syms %t.gfx942.co \
// RUN:   | %FileCheck --check-prefix=TGTSYM %s
// TGTSYM: vecadd

// COM: Focused warm-cache smoke: first run misses and writes, second run hits
// COM: the same caller-provided cache directory.
// RUN: rm -rf %t.cache
// RUN: env HSA_HOTSWAP_CACHE_DIR=%t.cache hotswap-transpile \
// RUN:                   %S/vecadd_gfx950.co \
// RUN:                   amdgcn-amd-amdhsa--gfx950 \
// RUN:                   amdgcn-amd-amdhsa--gfx942 \
// RUN:   | %FileCheck --check-prefix=CACHEMISS %s
// CACHEMISS-DAG: cache_hit=0
// CACHEMISS-DAG: cache_lookup=miss
// CACHEMISS-DAG: cache_write=success
// RUN: env HSA_HOTSWAP_CACHE_DIR=%t.cache hotswap-transpile \
// RUN:                   %S/vecadd_gfx950.co \
// RUN:                   amdgcn-amd-amdhsa--gfx950 \
// RUN:                   amdgcn-amd-amdhsa--gfx942 \
// RUN:   | %FileCheck --check-prefix=CACHEHIT %s
// CACHEHIT-DAG: cache_hit=1
// CACHEHIT-DAG: cache_lookup=hit
// CACHEHIT-DAG: cache_write=not_attempted

// COM: The HIP-global-offset assumption is a semantic lowering input. Toggling
// COM: it at the COMGR boundary must produce a distinct cache key, or cached
// COM: objects could be reused across incompatible launch contracts.
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
// HIPKEYBASE-DAG: RESULT: SUCCESS bytes={{[1-9][0-9]*}}
// HIPKEYBASE-DAG: cache_hit=0
// HIPKEYBASE-DAG: cache_lookup=miss
// HIPKEYBASE-DAG: cache_write=success
