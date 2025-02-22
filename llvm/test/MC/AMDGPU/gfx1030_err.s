// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1030 %s 2>&1 | FileCheck --check-prefix=GFX10 --implicit-check-not=error: %s
// XFAIL: *
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1031 %s 2>&1 | FileCheck --check-prefix=GFX10 --implicit-check-not=error: %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1032 %s 2>&1 | FileCheck --check-prefix=GFX10 --implicit-check-not=error: %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1033 %s 2>&1 | FileCheck --check-prefix=GFX10 --implicit-check-not=error: %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1034 %s 2>&1 | FileCheck --check-prefix=GFX10 --implicit-check-not=error: %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1035 %s 2>&1 | FileCheck --check-prefix=GFX10 --implicit-check-not=error: %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1036 %s 2>&1 | FileCheck --check-prefix=GFX10 --implicit-check-not=error: %s

v_dot8c_i32_i4 v5, v1, v2
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot8c_i32_i4 v5, v1, v2 quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot8c_i32_i4 v5, v1, v2 quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0 fi:1
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot8c_i32_i4 v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot8c_i32_i4 v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0] fi:1
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_get_waveid_in_workgroup s0
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_getreg_b32 s2, hwreg(HW_REG_XNACK_MASK)
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid hardware register: not supported on this GPU

v_mac_f32 v0, v1, v2
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mad_f32 v0, v1, v2, v3
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_madak_f32 v0, v1, v2, 1
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_madmk_f32 v0, v1, 1, v2
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mad_legacy_f32 v0, v1, v2, v3
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mac_legacy_f32 v0, v1, v2
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_add_src2_u32 v1 offset:65535 gds
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_add_src2_u32 v1 offset:65535
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_add_src2_f32 v1 offset:65535
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_sub_src2_u32 v1 offset:65535
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_rsub_src2_u32 v1 offset:65535
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_inc_src2_u32 v1 offset:65535
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_dec_src2_u32 v1 offset:65535
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_min_src2_i32 v1 offset:65535
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_max_src2_i32 v1 offset:65535
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_min_src2_u32 v1 offset:65535
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_max_src2_u32 v1 offset:65535
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_and_src2_b32 v1 offset:65535
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_or_src2_b32 v1 offset:65535
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_xor_src2_b32 v1 offset:65535
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_min_src2_f32 v1 offset:65535
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_max_src2_f32 v1 offset:65535
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_add_src2_u64 v1 offset:65535
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_sub_src2_u64 v1 offset:65535
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_rsub_src2_u64 v1 offset:65535
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_inc_src2_u64 v1 offset:65535
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_dec_src2_u64 v1 offset:65535
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_min_src2_i64 v1 offset:65535
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_max_src2_i64 v1 offset:65535
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_min_src2_u64 v1 offset:65535
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_max_src2_u64 v1 offset:65535
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_and_src2_b64 v1 offset:65535
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_or_src2_b64 v1 offset:65535
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_xor_src2_b64 v1 offset:65535
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_min_src2_f64 v1 offset:65535
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_max_src2_f64 v1 offset:65535
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_write_src2_b32 v1 offset:65535
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_write_src2_b64 v1 offset:65535
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_msaa_load v[1:4], v5, s[8:15] dmask:0xf dim:SQ_RSRC_IMG_1D
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid dim; must be MSAA type

image_msaa_load v5, v[1:2], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D d16
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid dim; must be MSAA type

//===----------------------------------------------------------------------===//
// s_waitcnt_depctr.
//===----------------------------------------------------------------------===//

s_waitcnt_depctr depctr_hold_cnt(-1)
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid value for depctr_hold_cnt

s_waitcnt_depctr depctr_sa_sdst(-1)
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid value for depctr_sa_sdst

s_waitcnt_depctr depctr_va_vdst(-1)
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid value for depctr_va_vdst

s_waitcnt_depctr depctr_va_sdst(-1)
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid value for depctr_va_sdst

s_waitcnt_depctr depctr_va_ssrc(-1)
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid value for depctr_va_ssrc

s_waitcnt_depctr depctr_va_vcc(-1)
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid value for depctr_va_vcc

s_waitcnt_depctr depctr_vm_vsrc(-1)
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid value for depctr_vm_vsrc

s_waitcnt_depctr depctr_hold_cnt(2)
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid value for depctr_hold_cnt

s_waitcnt_depctr depctr_sa_sdst(2)
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid value for depctr_sa_sdst

s_waitcnt_depctr depctr_va_vdst(16)
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid value for depctr_va_vdst

s_waitcnt_depctr depctr_va_sdst(8)
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid value for depctr_va_sdst

s_waitcnt_depctr depctr_va_ssrc(2)
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid value for depctr_va_ssrc

s_waitcnt_depctr depctr_va_vcc(2)
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid value for depctr_va_vcc

s_waitcnt_depctr depctr_vm_vsrc(8)
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid value for depctr_vm_vsrc

s_waitcnt_depctr depctr_vm_(8)
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid counter name depctr_vm_

s_waitcnt_depctr depctr_hold_cnt(0) depctr_hold_cnt(0)
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: duplicate counter name depctr_hold_cnt

s_waitcnt_depctr depctr_sa_sdst(0) depctr_sa_sdst(0)
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: duplicate counter name depctr_sa_sdst

image_bvh_intersect_ray v[4:7], v[9:16], s[4:7] noa16
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: image address size does not match a16

image_bvh_intersect_ray v[39:42], [v50, v46, v23, v17, v16, v15, v21, v20], s[12:15] noa16
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: image address size does not match a16

// missing dim
image_msaa_load v[1:4], v[5:7], s[8:15] dmask:0xf glc
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: missing dim operand

// op_sel not allowed in dot opcodes with 4- or 8-bit packed data

v_dot4_i32_i8 v0, v1, v2, v3 op_sel:[0,0]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_i8 v0, v1, v2, v3 op_sel:[0,1]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_i8 v0, v1, v2, v3 op_sel:[1,0]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_i8 v0, v1, v2, v3 op_sel:[1,1]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_i8 v0, v1, v2, v3 op_sel_hi:[0,0]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_i8 v0, v1, v2, v3 op_sel_hi:[0,1]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_i8 v0, v1, v2, v3 op_sel_hi:[1,0]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_i8 v0, v1, v2, v3 op_sel_hi:[1,1]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_i8 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[0,0]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_i8 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[0,1]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_i8 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[1,0]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_i8 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[1,1]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_i8 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[0,0]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_i8 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[0,1]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_i8 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[1,0]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_i8 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[1,1]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_i8 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[0,0]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_i8 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[0,1]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_i8 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[1,0]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_i8 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[1,1]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_i8 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[0,0]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_i8 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[0,1]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_i8 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[1,0]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_i8 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[1,1]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[0,0]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[0,1]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[1,0]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[1,1]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel_hi:[0,0]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel_hi:[0,1]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel_hi:[1,0]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel_hi:[1,1]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[0,0]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[0,1]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[1,0]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[1,1]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[0,0]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[0,1]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[1,0]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[1,1]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[0,0]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[0,1]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[1,0]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[1,1]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[0,0]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[0,1]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[1,0]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[1,1]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4c_i32_i8 v0, v1, v2, v3 op_sel:[0,0]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4c_i32_i8 v0, v1, v2, v3 op_sel:[0,1]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4c_i32_i8 v0, v1, v2, v3 op_sel:[1,0]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4c_i32_i8 v0, v1, v2, v3 op_sel:[1,1]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4c_i32_i8 v0, v1, v2, v3 op_sel_hi:[0,0]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4c_i32_i8 v0, v1, v2, v3 op_sel_hi:[0,1]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4c_i32_i8 v0, v1, v2, v3 op_sel_hi:[1,0]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4c_i32_i8 v0, v1, v2, v3 op_sel_hi:[1,1]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4c_i32_i8 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[0,0]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4c_i32_i8 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[0,1]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4c_i32_i8 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[1,0]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4c_i32_i8 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[1,1]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4c_i32_i8 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[0,0]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4c_i32_i8 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[0,1]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4c_i32_i8 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[1,0]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4c_i32_i8 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[1,1]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4c_i32_i8 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[0,0]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4c_i32_i8 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[0,1]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4c_i32_i8 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[1,0]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4c_i32_i8 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[1,1]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4c_i32_i8 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[0,0]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4c_i32_i8 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[0,1]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4c_i32_i8 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[1,0]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4c_i32_i8 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[1,1]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_i32_i4 v0, v1, v2, v3 op_sel:[0,0]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_i32_i4 v0, v1, v2, v3 op_sel:[0,1]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_i32_i4 v0, v1, v2, v3 op_sel:[1,0]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_i32_i4 v0, v1, v2, v3 op_sel:[1,1]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_i32_i4 v0, v1, v2, v3 op_sel_hi:[0,0]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_i32_i4 v0, v1, v2, v3 op_sel_hi:[0,1]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_i32_i4 v0, v1, v2, v3 op_sel_hi:[1,0]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_i32_i4 v0, v1, v2, v3 op_sel_hi:[1,1]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_i32_i4 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[0,0]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_i32_i4 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[0,1]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_i32_i4 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[1,0]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_i32_i4 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[1,1]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_i32_i4 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[0,0]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_i32_i4 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[0,1]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_i32_i4 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[1,0]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_i32_i4 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[1,1]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_i32_i4 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[0,0]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_i32_i4 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[0,1]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_i32_i4 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[1,0]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_i32_i4 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[1,1]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_i32_i4 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[0,0]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_i32_i4 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[0,1]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_i32_i4 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[1,0]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_i32_i4 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[1,1]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[0,0]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[0,1]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[1,0]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[1,1]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel_hi:[0,0]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel_hi:[0,1]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel_hi:[1,0]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel_hi:[1,1]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[0,0]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[0,1]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[1,0]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[1,1]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[0,0]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[0,1]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[1,0]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[1,1]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[0,0]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[0,1]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[1,0]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[1,1]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[0,0]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[0,1]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[1,0]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[1,1]
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

image_bvh_intersect_ray v[4:7], v[9:19], null
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

image_bvh64_intersect_ray v[4:7], v[9:20], null
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
