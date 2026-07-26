// RUN: llvm-mc -triple=amdgpu12.50 -mattr=+gfx1250-b0-specific -show-encoding %s | FileCheck --check-prefix=ENC %s
// RUN: llvm-mc -triple=amdgpu12.50 -mattr=-gfx1250-b0-specific -show-encoding %s | FileCheck --check-prefix=ENC %s
// RUN: llvm-mc -triple=amdgpu12.50 -mattr=+gfx1250-b0-specific -show-encoding %s | %extract-encodings | llvm-mc -triple=amdgpu12.50 -mattr=+gfx1250-b0-specific -disassemble -show-encoding | FileCheck --check-prefix=DIS --implicit-check-not=v_illegal --implicit-check-not=v_cmp_ge_u16_e32 %s
// RUN: llvm-mc -triple=amdgpu12.50 -mattr=-gfx1250-b0-specific -show-encoding %s | %extract-encodings | llvm-mc -triple=amdgpu12.50 -mattr=-gfx1250-b0-specific -disassemble -show-encoding | FileCheck --check-prefix=DIS --implicit-check-not=v_illegal --implicit-check-not=v_cmp_ge_u16_e32 %s
// RUN: llvm-mc -triple=amdgpu12.50 -mattr=+gfx1250-b0-specific -show-encoding %s | %extract-encodings | llvm-mc -triple=amdgpu12.50 -mattr=+gfx1250-b0-specific -disassemble -show-inst | FileCheck --check-prefix=MCINST %s
// RUN: llvm-mc -triple=amdgpu12.50 -mattr=-gfx1250-b0-specific -show-encoding %s | %extract-encodings | llvm-mc -triple=amdgpu12.50 -mattr=-gfx1250-b0-specific -disassemble -show-inst | FileCheck --check-prefix=MCINST %s

tensor_load_to_lds_gfx1250_legacy s[0:3], s[4:11]
// ENC: tensor_load_to_lds_gfx1250_legacy s[0:3], s[4:11] ; encoding: [0x00,0x00,0x31,0xd0,0x00,0x00,0x00,0x00,0x00,0x04,0x7c,0x7c]
// DIS: tensor_load_to_lds s[0:3], s[4:11] ; encoding: [0x01,0x00,0x71,0xd0,0x00,0x00,0x00,0x7c,0x00,0x04,0x7c,0x7c]
// MCINST: <MCInst #{{[0-9]+}} TENSOR_LOAD_TO_LDS_d2_gfx1250

tensor_load_to_lds_gfx1250_legacy s[0:3], s[4:11], s[12:15], s[16:19]
// ENC: tensor_load_to_lds_gfx1250_legacy s[0:3], s[4:11], s[12:15], s[16:19] ; encoding: [0x00,0x00,0x31,0xd0,0x00,0x00,0x00,0x00,0x00,0x04,0x0c,0x10]
// DIS-NEXT: tensor_load_to_lds s[0:3], s[4:11], s[12:15], s[16:19] ; encoding: [0x01,0x00,0x71,0xd0,0x00,0x00,0x00,0x7c,0x00,0x04,0x0c,0x10]
// MCINST: <MCInst #{{[0-9]+}} TENSOR_LOAD_TO_LDS_d4_gfx1250

tensor_store_from_lds_gfx1250_legacy s[0:3], s[4:11]
// ENC: tensor_store_from_lds_gfx1250_legacy s[0:3], s[4:11] ; encoding: [0x00,0x40,0x31,0xd0,0x00,0x00,0x00,0x00,0x00,0x04,0x7c,0x7c]
// DIS-NEXT: tensor_store_from_lds s[0:3], s[4:11] ; encoding: [0x01,0x40,0x71,0xd0,0x00,0x00,0x00,0x7c,0x00,0x04,0x7c,0x7c]
// MCINST: <MCInst #{{[0-9]+}} TENSOR_STORE_FROM_LDS_d2_gfx1250

tensor_store_from_lds_gfx1250_legacy s[0:3], s[4:11], s[12:15], s[16:19]
// ENC: tensor_store_from_lds_gfx1250_legacy s[0:3], s[4:11], s[12:15], s[16:19] ; encoding: [0x00,0x40,0x31,0xd0,0x00,0x00,0x00,0x00,0x00,0x04,0x0c,0x10]
// DIS-NEXT: tensor_store_from_lds s[0:3], s[4:11], s[12:15], s[16:19] ; encoding: [0x01,0x40,0x71,0xd0,0x00,0x00,0x00,0x7c,0x00,0x04,0x0c,0x10]
// MCINST: <MCInst #{{[0-9]+}} TENSOR_STORE_FROM_LDS_d4_gfx1250

tensor_load_to_lds s[0:3], s[4:11]
// ENC: tensor_load_to_lds s[0:3], s[4:11] ; encoding: [0x01,0x00,0x71,0xd0,0x00,0x00,0x00,0x7c,0x00,0x04,0x7c,0x7c]
// DIS-NEXT: tensor_load_to_lds s[0:3], s[4:11] ; encoding: [0x01,0x00,0x71,0xd0,0x00,0x00,0x00,0x7c,0x00,0x04,0x7c,0x7c]

tensor_load_to_lds s[0:3], s[4:11], s[12:15], s[16:19]
// ENC: tensor_load_to_lds s[0:3], s[4:11], s[12:15], s[16:19] ; encoding: [0x01,0x00,0x71,0xd0,0x00,0x00,0x00,0x7c,0x00,0x04,0x0c,0x10]
// DIS-NEXT: tensor_load_to_lds s[0:3], s[4:11], s[12:15], s[16:19] ; encoding: [0x01,0x00,0x71,0xd0,0x00,0x00,0x00,0x7c,0x00,0x04,0x0c,0x10]

tensor_store_from_lds s[0:3], s[4:11]
// ENC: tensor_store_from_lds s[0:3], s[4:11] ; encoding: [0x01,0x40,0x71,0xd0,0x00,0x00,0x00,0x7c,0x00,0x04,0x7c,0x7c]
// DIS-NEXT: tensor_store_from_lds s[0:3], s[4:11] ; encoding: [0x01,0x40,0x71,0xd0,0x00,0x00,0x00,0x7c,0x00,0x04,0x7c,0x7c]

tensor_store_from_lds s[0:3], s[4:11], s[12:15], s[16:19]
// ENC: tensor_store_from_lds s[0:3], s[4:11], s[12:15], s[16:19] ; encoding: [0x01,0x40,0x71,0xd0,0x00,0x00,0x00,0x7c,0x00,0x04,0x0c,0x10]
// DIS-NEXT: tensor_store_from_lds s[0:3], s[4:11], s[12:15], s[16:19] ; encoding: [0x01,0x40,0x71,0xd0,0x00,0x00,0x00,0x7c,0x00,0x04,0x0c,0x10]
