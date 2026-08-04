// COM: Integration test: the production hotswap rewrite entry point consults
// COM: the on-disk translation cache. Run the rewrite twice with
// COM: HSA_HOTSWAP_CACHE_DIR set: the first run must POPULATE the cache dir,
// COM: and the second run must produce BYTE-IDENTICAL output (served from /
// COM: consistent with the disk entry). Guards against the disk tier being
// COM: wired only as a standalone module and never reached from production.
// COM:
// COM: The kernel body below is copied verbatim from hotswap-inplace-mixed.s.
// COM: Its cluster_load ops trigger a real B0->A0 in-place patch (cluster_load
// COM: -> global_load), so retargetCodeObject does real work and the cached
// COM: PipelineResult is a genuine patched code object. Crucially the kernel
// COM: carries a full .amdgpu_metadata note: the disk cache key is derived
// COM: from the source kernel metadata (listKernelNames), so an input lacking
// COM: that note cannot be keyed and would never be written to disk.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: rm -rf %t.cache && mkdir -p %t.cache

// COM: --- Run 1: MISS + WRITE ---
// COM: Fresh process => empty mem tier; empty dir => disk miss. The production
// COM: producer retargets once and writes the object+metadata to disk.
// RUN: env HSA_HOTSWAP_CACHE_DIR=%t.cache hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out1.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// COM: The miss path must have written the object artifact (<key>.Hsaco).
// RUN: find %t.cache -type f -name '*.Hsaco' | %FileCheck --check-prefix=DISK %s
// DISK: {{.+\.Hsaco}}

// COM: Record a rewrite-sensitive identity of the stored object. The disk tier
// COM: writes via atomic temp-file + rename, and a disk HIT returns the stored
// COM: bytes WITHOUT calling writeTranslationCache (see the producer in
// COM: rewriter.cpp) -- so a hit leaves the file completely untouched. We
// COM: capture inode + mtime + ctime (%%i:%%Y:%%Z): a genuine hit leaves all
// COM: three unchanged, while ANY re-write advances mtime/ctime (and usually
// COM: the inode too). This distinguishes a disk hit from a silent recompute,
// COM: which cmp alone cannot (retarget being deterministic). Note: inode
// COM: alone is insufficient -- ext4 reuses a just-freed inode number on a
// COM: quick delete+recreate, so mtime/ctime are what give this check teeth.
// COM: The disk tier shards by key[0:2], so the single object is the one match
// COM: of %t.cache/*/*.Hsaco.
// RUN: stat -c '%%i:%%Y:%%Z' %t.cache/*/*.Hsaco > %t.id1

// COM: --- Run 2: MISS (mem) + HIT (disk) ---
// COM: Separate process => empty mem tier again, so this is a mem miss that
// COM: MUST fall through to the disk tier. The entry is present => disk hit.
// RUN: env HSA_HOTSWAP_CACHE_DIR=%t.cache hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out2.elf \
// RUN:   | %FileCheck --check-prefix=API %s

// COM: Proof of the disk HIT: the stored object was not touched. If run 2 had
// COM: missed and recomputed, the producer would have re-written the object
// COM: (new mtime/ctime via atomic rename). Unchanged identity => run 2 served
// COM: from disk without recomputing, i.e. the disk READ path is wired and
// COM: effective.
// RUN: stat -c '%%i:%%Y:%%Z' %t.cache/*/*.Hsaco > %t.id2
// RUN: diff %t.id1 %t.id2

// COM: And the returned bytes are correct: byte-identical to the cold output.
// RUN: cmp %t.out1.elf %t.out2.elf

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_inplace_kernel
.p2align 8
.type test_inplace_kernel,@function
test_inplace_kernel:
  s_clause 0x0
  cluster_load_b32 v0, v[2:3], off
  s_wait_loadcnt 0x0
  cluster_load_b128 v[4:7], v[8:9], off
  s_wait_loadcnt 0x0
  s_clause 0x1
  global_load_b32 v10, v[2:3], off
  global_load_b32 v11, v[2:3], off offset:4
  s_wait_loadcnt 0x0
  s_endpgm
.Ltest_inplace_kernel_end:
.size test_inplace_kernel, .Ltest_inplace_kernel_end-test_inplace_kernel

.rodata
.p2align 8
.amdhsa_kernel test_inplace_kernel
  .amdhsa_next_free_vgpr 12
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_inplace_kernel
      .symbol: test_inplace_kernel.kd
      .gfx1250_revision: B0
      .sgpr_count: 2
      .vgpr_count: 12
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
