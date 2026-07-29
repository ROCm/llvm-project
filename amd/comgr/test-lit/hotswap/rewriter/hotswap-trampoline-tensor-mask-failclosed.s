// COM: A0 tensor where both strategies genuinely fail, so hotswap must reject
// COM: rather than emit an object that can still hang A0. The descriptor is a
// COM: bare operand (no construction region -> definition clear not applicable)
// COM: and its SGPR is live after the tensor while the kernel's SGPR budget is
// COM: saturated to s106, so the at-site fallback has no scratch register.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status ERROR 2>&1 \
// RUN:   | %FileCheck --check-prefix=API %s
// API: hotswap: error: tensor_load_to_lds: no scratch SGPR available
// API: RESULT: ERROR

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_tensor_failclosed
.p2align 8
.type test_tensor_failclosed,@function
test_tensor_failclosed:
  tensor_load_to_lds s[0:3], s[4:11]
  s_mov_b32 s0, s4
  // Keep every remaining declared numbered SGPR live at the continuation so
  // the current-text fallback also has no legal scratch register.
  s_mov_b64 s[0:1], s[12:13]
  s_mov_b64 s[0:1], s[14:15]
  s_mov_b64 s[0:1], s[16:17]
  s_mov_b64 s[0:1], s[18:19]
  s_mov_b64 s[0:1], s[20:21]
  s_mov_b64 s[0:1], s[22:23]
  s_mov_b64 s[0:1], s[24:25]
  s_mov_b64 s[0:1], s[26:27]
  s_mov_b64 s[0:1], s[28:29]
  s_mov_b64 s[0:1], s[30:31]
  s_mov_b64 s[0:1], s[32:33]
  s_mov_b64 s[0:1], s[34:35]
  s_mov_b64 s[0:1], s[36:37]
  s_mov_b64 s[0:1], s[38:39]
  s_mov_b64 s[0:1], s[40:41]
  s_mov_b64 s[0:1], s[42:43]
  s_mov_b64 s[0:1], s[44:45]
  s_mov_b64 s[0:1], s[46:47]
  s_mov_b64 s[0:1], s[48:49]
  s_mov_b64 s[0:1], s[50:51]
  s_mov_b64 s[0:1], s[52:53]
  s_mov_b64 s[0:1], s[54:55]
  s_mov_b64 s[0:1], s[56:57]
  s_mov_b64 s[0:1], s[58:59]
  s_mov_b64 s[0:1], s[60:61]
  s_mov_b64 s[0:1], s[62:63]
  s_mov_b64 s[0:1], s[64:65]
  s_mov_b64 s[0:1], s[66:67]
  s_mov_b64 s[0:1], s[68:69]
  s_mov_b64 s[0:1], s[70:71]
  s_mov_b64 s[0:1], s[72:73]
  s_mov_b64 s[0:1], s[74:75]
  s_mov_b64 s[0:1], s[76:77]
  s_mov_b64 s[0:1], s[78:79]
  s_mov_b64 s[0:1], s[80:81]
  s_mov_b64 s[0:1], s[82:83]
  s_mov_b64 s[0:1], s[84:85]
  s_mov_b64 s[0:1], s[86:87]
  s_mov_b64 s[0:1], s[88:89]
  s_mov_b64 s[0:1], s[90:91]
  s_mov_b64 s[0:1], s[92:93]
  s_mov_b64 s[0:1], s[94:95]
  s_mov_b64 s[0:1], s[96:97]
  s_mov_b64 s[0:1], s[98:99]
  s_mov_b64 s[0:1], s[100:101]
  s_mov_b64 s[0:1], s[102:103]
  s_endpgm
.Ltest_tensor_failclosed_end:
.size test_tensor_failclosed, .Ltest_tensor_failclosed_end-test_tensor_failclosed

.rodata
.p2align 8
.amdhsa_kernel test_tensor_failclosed
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 106
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_tensor_failclosed
      .symbol: test_tensor_failclosed.kd
      .sgpr_count: 106
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
