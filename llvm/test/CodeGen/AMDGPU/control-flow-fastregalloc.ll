; RUN: llc -amdgpu-late-wave-transform=1 -O0 -mtriple=amdgpu7.00--amdhsa -amdgpu-spill-sgpr-to-vgpr=0 < %s | FileCheck -enable-var-scope -check-prefix=VMEM -check-prefix=GCN %s
; RUN: llc -amdgpu-late-wave-transform=1 -O0 -mtriple=amdgpu7.00--amdhsa -amdgpu-spill-sgpr-to-vgpr=1 < %s | FileCheck -enable-var-scope -check-prefix=VGPR -check-prefix=GCN %s

; Verify registers used for tracking exec mask changes when all
; registers are spilled at the end of the block. The SGPR spill
; placement relative to the exec modifications are important.

; FIXME: This checks with SGPR to VGPR spilling disabled, but this may
; not work correctly in cases where no workitems take a branch.


; GCN-LABEL: {{^}}divergent_if_endif:

; GCN: {{^}}; %bb.0:
; GCN: s_mov_b32 m0, -1
; GCN: ds_read_b32 [[LOAD0:v[0-9]+]]

; Spill load
; GCN: buffer_store_dword [[LOAD0]], off, s[0:3], 0 offset:[[LOAD0_OFFSET:[0-9]+]] ; 4-byte Folded Spill
; GCN: v_cmp_eq_u32_e64 [[CMP0:s\[[0-9]+:[0-9]+\]]], v{{[0-9]+}}, v{{[0-9]+}}

; Spill saved exec
; GCN: s_xor_b64 s[[[SAVEEXEC_LO:[0-9]+]]:[[SAVEEXEC_HI:[0-9]+]]], exec, [[PRIMARY:s\[[0-9]+:[0-9]+\]]]
; VGPR: v_writelane_b32 [[SPILL_VGPR:v[0-9]+]], s[[SAVEEXEC_LO]], [[SAVEEXEC_LO_LANE:[0-9]+]]
; VGPR: v_writelane_b32 [[SPILL_VGPR]], s[[SAVEEXEC_HI]], [[SAVEEXEC_HI_LANE:[0-9]+]]
; VGPR: buffer_store_dword [[SPILL_VGPR]], off, s[0:3], 0 offset:[[SPILL_OFFSET:[0-9]+]] ; 4-byte Folded Spill

; VMEM: v_writelane_b32 v[[V_SAVEEXEC:[0-9]+]], s[[SAVEEXEC_LO]], 0
; VMEM: v_writelane_b32 v[[V_SAVEEXEC]], s[[SAVEEXEC_HI]], 1
; VMEM: buffer_store_dword v[[V_SAVEEXEC]], off, s[0:3], 0 offset:[[SAVEEXEC_OFFSET:[0-9]+]] ; 4-byte Folded Spill

; GCN: s_mov_b64 exec, [[PRIMARY]]
; GCN-NEXT: ; divergent control-flow edge
; GCN-NEXT: s_cbranch_execz [[ENDIF:.LBB[0-9]+_[0-9]+]]

; GCN: {{^.LBB[0-9]+_[0-9]+}}: ; %if
; GCN: buffer_load_dword [[RELOAD_LOAD0:v[0-9]+]], off, s[0:3], 0 offset:[[LOAD0_OFFSET]] ; 4-byte Folded Reload
; GCN: s_mov_b32 m0, -1
; GCN: ds_read_b32 [[LOAD1:v[0-9]+]]
; GCN: s_waitcnt vmcnt(0) lgkmcnt(0)


; Spill val register
; GCN: v_add_i32_e32 [[VAL:v[0-9]+]], vcc, [[RELOAD_LOAD0]], [[LOAD1]]
; GCN: buffer_store_dword [[VAL]], off, s[0:3], 0 ; 4-byte Folded Spill

; GCN: [[ENDIF]]:

; Reload and restore exec mask
; VGPR: v_readlane_b32 s[[S_RELOAD_SAVEEXEC_LO:[0-9]+]], [[SPILL_VGPR]], [[SAVEEXEC_LO_LANE]]
; VGPR: v_readlane_b32 s[[S_RELOAD_SAVEEXEC_HI:[0-9]+]], [[SPILL_VGPR]], [[SAVEEXEC_HI_LANE]]

; VMEM: buffer_load_dword v[[V_RELOAD_SAVEEXEC:[0-9]+]], off, s[0:3], 0 offset:[[SAVEEXEC_OFFSET]] ; 4-byte Folded Reload
; VMEM: s_waitcnt vmcnt(0)
; VMEM: v_readlane_b32 s[[S_RELOAD_SAVEEXEC_LO:[0-9]+]], v[[V_RELOAD_SAVEEXEC]], 0
; VMEM: v_readlane_b32 s[[S_RELOAD_SAVEEXEC_HI:[0-9]+]], v[[V_RELOAD_SAVEEXEC]], 1

; GCN: s_or_b64 exec, exec, s[[[S_RELOAD_SAVEEXEC_LO]]:[[S_RELOAD_SAVEEXEC_HI]]]

; Restore val
; GCN: buffer_load_dword [[RELOAD_VAL:v[0-9]+]], off, s[0:3], 0 ; 4-byte Folded Reload

; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RELOAD_VAL]]

; VGPR: .amdhsa_private_segment_fixed_size 16
define amdgpu_kernel void @divergent_if_endif(ptr addrspace(1) %out) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %load0 = load volatile i32, ptr addrspace(3) poison
  %cmp0 = icmp eq i32 %tid, 0
  br i1 %cmp0, label %if, label %endif

if:
  %load1 = load volatile i32, ptr addrspace(3) poison
  %val = add i32 %load0, %load1
  br label %endif

endif:
  %tmp4 = phi i32 [ %val, %if ], [ 0, %entry ]
  store i32 %tmp4, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}divergent_loop:

; GCN: {{^}}; %bb.0:
; GCN-DAG: s_mov_b32 m0, -1
; GCN-DAG: v_mov_b32_e32 [[PTR0:v[0-9]+]], 0{{$}}
; GCN: ds_read_b32 [[LOAD0:v[0-9]+]], [[PTR0]]
; GCN: v_cmp_eq_u32_e64 [[CMP0:s\[[0-9]+:[0-9]+\]]], v{{[0-9]+}}, s{{[0-9]+}}

; Spill load
; GCN: buffer_store_dword [[LOAD0]], off, s[0:3], 0 offset:[[LOAD0_OFFSET:[0-9]+]] ; 4-byte Folded Spill

; Spill saved exec
; GCN: s_xor_b64 s[[[SAVEEXEC_LO:[0-9]+]]:[[SAVEEXEC_HI:[0-9]+]]], exec, [[PRIMARY:s\[[0-9]+:[0-9]+\]]]
; VGPR: v_writelane_b32 [[SPILL_VGPR:v[0-9]+]], s[[SAVEEXEC_LO]], [[SAVEEXEC_LO_LANE:[0-9]+]]
; VGPR: v_writelane_b32 [[SPILL_VGPR]], s[[SAVEEXEC_HI]], [[SAVEEXEC_HI_LANE:[0-9]+]]
; VGPR: buffer_store_dword [[SPILL_VGPR]], off, s[0:3], 0 offset:[[SPILL_OFFSET:[0-9]+]] ; 4-byte Folded Spill

; VMEM: v_writelane_b32 v[[V_SAVEEXEC:[0-9]+]], s[[SAVEEXEC_LO]], 0
; VMEM: v_writelane_b32 v[[V_SAVEEXEC]], s[[SAVEEXEC_HI]], 1
; VMEM: buffer_store_dword v[[V_SAVEEXEC]], off, s[0:3], 0 offset:[[SAVEEXEC_OFFSET:[0-9]+]] ; 4-byte Folded Spill

; GCN: s_mov_b64 exec, [[PRIMARY]]
; GCN-NEXT: ; divergent control-flow edge
; GCN-NEXT: s_cbranch_execz [[END:.LBB[0-9]+_[0-9]+]]


; GCN: [[LOOP:.LBB[0-9]+_[0-9]+]]:
; GCN: buffer_load_dword v[[VAL_LOOP_RELOAD:[0-9]+]], off, s[0:3], 0 offset:[[LOAD0_OFFSET]] ; 4-byte Folded Reload
; GCN: v_sub_i32_e32 v[[VAL_LOOP_RELOAD]], vcc, v[[VAL_LOOP_RELOAD]], v{{[0-9]+}}
; GCN: s_cmp_{{lg|eq}}_u32
; GCN: buffer_store_dword v{{[0-9]+}}, off, s[0:3], 0 offset:{{[0-9]+}} ; 4-byte Folded Spill
; GCN: s_cbranch_scc1 [[LOOP]]

; GCN: [[END]]:

; Reload and restore exec mask
; VGPR: v_readlane_b32 s[[S_RELOAD_SAVEEXEC_LO:[0-9]+]], [[SPILL_VGPR]], [[SAVEEXEC_LO_LANE]]
; VGPR: v_readlane_b32 s[[S_RELOAD_SAVEEXEC_HI:[0-9]+]], [[SPILL_VGPR]], [[SAVEEXEC_HI_LANE]]

; VMEM: buffer_load_dword v[[V_RELOAD_SAVEEXEC:[0-9]+]], off, s[0:3], 0 offset:[[SAVEEXEC_OFFSET]] ; 4-byte Folded Reload
; VMEM: s_waitcnt vmcnt(0)
; VMEM: v_readlane_b32 s[[S_RELOAD_SAVEEXEC_LO:[0-9]+]], v[[V_RELOAD_SAVEEXEC]], 0
; VMEM: v_readlane_b32 s[[S_RELOAD_SAVEEXEC_HI:[0-9]+]], v[[V_RELOAD_SAVEEXEC]], 1

; GCN: s_or_b64 exec, exec, s[[[S_RELOAD_SAVEEXEC_LO]]:[[S_RELOAD_SAVEEXEC_HI]]]

; GCN: buffer_load_dword v[[VAL_END:[0-9]+]], off, s[0:3], 0 ; 4-byte Folded Reload

; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, v[[VAL_END]]

; VGPR: .amdhsa_private_segment_fixed_size 16
define amdgpu_kernel void @divergent_loop(ptr addrspace(1) %out) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %load0 = load volatile i32, ptr addrspace(3) zeroinitializer
  %cmp0 = icmp eq i32 %tid, 0
  br i1 %cmp0, label %loop, label %end

loop:
  %i = phi i32 [ %i.inc, %loop ], [ 0, %entry ]
  %val = phi i32 [ %val.sub, %loop ], [ %load0, %entry ]
  %load1 = load volatile i32, ptr addrspace(3) poison
  %i.inc = add i32 %i, 1
  %val.sub = sub i32 %val, %load1
  %cmp1 = icmp ne i32 %i, 256
  br i1 %cmp1, label %loop, label %end

end:
  %tmp4 = phi i32 [ %val.sub, %loop ], [ 0, %entry ]
  store i32 %tmp4, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}divergent_if_else_endif:
; GCN: {{^}}; %bb.0:

; GCN-DAG: s_mov_b32 m0, -1
; GCN-DAG: v_mov_b32_e32 [[PTR0:v[0-9]+]], 0{{$}}
; GCN: ds_read_b32 [[LOAD0:v[0-9]+]], [[PTR0]]

; Spill load
; GCN: buffer_store_dword [[LOAD0]], off, s[0:3], 0{{( offset:[0-9]+)?}} ; 4-byte Folded Spill

; GCN: s_mov_b32 [[ZERO:s[0-9]+]], 0
; GCN: v_cmp_eq_u32_e64 [[CMP0:s\[[0-9]+:[0-9]+\]]], v{{[0-9]+}}, [[ZERO]]

; Spill saved exec
; GCN: s_xor_b64 s[[[SAVEEXEC_LO:[0-9]+]]:[[SAVEEXEC_HI:[0-9]+]]], [[IF_PRIMARY:s\[[0-9]+:[0-9]+\]]], exec
; VGPR: v_writelane_b32 [[SPILL_VGPR:v[0-9]+]], s[[SAVEEXEC_LO]], [[SAVEEXEC_LO_LANE:[0-9]+]]
; VGPR: v_writelane_b32 [[SPILL_VGPR]], s[[SAVEEXEC_HI]], [[SAVEEXEC_HI_LANE:[0-9]+]]
; VGPR: buffer_store_dword [[SPILL_VGPR]], off, s[0:3], 0 offset:[[SPILL_OFFSET:[0-9]+]] ; 4-byte Folded Spill

; VMEM: v_writelane_b32 v[[V_SAVEEXEC:[0-9]+]], s[[SAVEEXEC_LO]], 0
; VMEM: v_writelane_b32 v[[V_SAVEEXEC]], s[[SAVEEXEC_HI]], 1
; VMEM: buffer_store_dword v[[V_SAVEEXEC]], off, s[0:3], 0 offset:[[SAVEEXEC_OFFSET:[0-9]+]] ; 4-byte Folded Spill

; GCN: s_mov_b64 exec, [[IF_PRIMARY]]
; GCN-NEXT: ; divergent control-flow edge
; GCN-NEXT: s_cbranch_execz [[FLOW:.LBB[0-9]+_[0-9]+]]


; GCN: {{^.LBB[0-9]+_[0-9]+}}: ; %if
; GCN: buffer_load_dword v[[LOAD0_RELOAD:[0-9]+]], off, s[0:3], 0{{( offset:[0-9]+)?}} ; 4-byte Folded Reload
; GCN: ds_read_b32
; GCN: v_add_i32_e32 v[[LOAD0_RELOAD]], vcc, v[[LOAD0_RELOAD]], [[ADD:v[0-9]+]]
; GCN: buffer_store_dword v[[LOAD0_RELOAD]], off, s[0:3], 0 offset:[[RESULT_OFFSET:[0-9]+]] ; 4-byte Folded Spill

; GCN: [[FLOW]]:

; Reload and restore exec mask
; VGPR: v_readlane_b32 s[[FLOW_S_RELOAD_SAVEEXEC_LO:[0-9]+]], [[SPILL_VGPR]], [[SAVEEXEC_LO_LANE]]
; VGPR: v_readlane_b32 s[[FLOW_S_RELOAD_SAVEEXEC_HI:[0-9]+]], [[SPILL_VGPR]], [[SAVEEXEC_HI_LANE]]

; VMEM: buffer_load_dword v[[FLOW_V_RELOAD_SAVEEXEC:[0-9]+]], off, s[0:3], 0 offset:[[SAVEEXEC_OFFSET]] ; 4-byte Folded Reload
; VMEM: s_waitcnt vmcnt(0)
; VMEM: v_readlane_b32 s[[FLOW_S_RELOAD_SAVEEXEC_LO:[0-9]+]], v[[FLOW_V_RELOAD_SAVEEXEC]], 0
; VMEM: v_readlane_b32 s[[FLOW_S_RELOAD_SAVEEXEC_HI:[0-9]+]], v[[FLOW_V_RELOAD_SAVEEXEC]], 1

; GCN: s_or_b64 exec, exec, s[[[FLOW_S_RELOAD_SAVEEXEC_LO]]:[[FLOW_S_RELOAD_SAVEEXEC_HI]]]

; Spill saved exec
; GCN: s_xor_b64 s[[[FLOW_SAVEEXEC_LO:[0-9]+]]:[[FLOW_SAVEEXEC_HI:[0-9]+]]], exec, [[ELSE_PRIMARY:s\[[0-9]+:[0-9]+\]]]
; VGPR: v_writelane_b32 [[SPILL_VGPR]], s[[FLOW_SAVEEXEC_LO]], [[FLOW_SAVEEXEC_LO_LANE:[0-9]+]]
; VGPR: v_writelane_b32 [[SPILL_VGPR]], s[[FLOW_SAVEEXEC_HI]], [[FLOW_SAVEEXEC_HI_LANE:[0-9]+]]
; VGPR: buffer_store_dword [[SPILL_VGPR]], off, s[0:3], 0 offset:[[SPILL_OFFSET:[0-9]+]] ; 4-byte Folded Spill

; VMEM: v_writelane_b32 v[[FLOW_V_SAVEEXEC:[0-9]+]], s[[FLOW_SAVEEXEC_LO]], 0
; VMEM: v_writelane_b32 v[[FLOW_V_SAVEEXEC]], s[[FLOW_SAVEEXEC_HI]], 1
; VMEM: buffer_store_dword v[[FLOW_V_SAVEEXEC]], off, s[0:3], 0 offset:[[FLOW_SAVEEXEC_OFFSET:[0-9]+]] ; 4-byte Folded Spill

; GCN: s_mov_b64 exec, [[ELSE_PRIMARY]]
; GCN-NEXT: ; divergent control-flow edge
; GCN-NEXT: s_cbranch_execz [[ENDIF:.LBB[0-9]+_[0-9]+]]


; GCN: {{^.LBB[0-9]+_[0-9]+}}: ; %else
; GCN: buffer_load_dword v[[LOAD0_RELOAD:[0-9]+]], off, s[0:3], 0{{( offset:[0-9]+)?}} ; 4-byte Folded Reload
; GCN: v_sub_i32_e32 v[[LOAD0_RELOAD]], vcc, v[[LOAD0_RELOAD]], v{{[0-9]+}}
; GCN: buffer_store_dword v[[LOAD0_RELOAD]], off, s[0:3], 0 offset:[[RESULT_OFFSET]] ; 4-byte Folded Spill

; GCN: [[ENDIF]]: ; %endif

; Reload and restore exec mask
; VGPR: v_readlane_b32 s[[S_RELOAD_SAVEEXEC_LO:[0-9]+]], [[SPILL_VGPR]], [[FLOW_SAVEEXEC_LO_LANE]]
; VGPR: v_readlane_b32 s[[S_RELOAD_SAVEEXEC_HI:[0-9]+]], [[SPILL_VGPR]], [[FLOW_SAVEEXEC_HI_LANE]]

; VMEM: buffer_load_dword v[[V_RELOAD_SAVEEXEC:[0-9]+]], off, s[0:3], 0 offset:[[FLOW_SAVEEXEC_OFFSET]] ; 4-byte Folded Reload
; VMEM: s_waitcnt vmcnt(0)
; VMEM: v_readlane_b32 s[[S_RELOAD_SAVEEXEC_LO:[0-9]+]], v[[V_RELOAD_SAVEEXEC]], 0
; VMEM: v_readlane_b32 s[[S_RELOAD_SAVEEXEC_HI:[0-9]+]], v[[V_RELOAD_SAVEEXEC]], 1

; GCN: s_or_b64 exec, exec, s[[[S_RELOAD_SAVEEXEC_LO]]:[[S_RELOAD_SAVEEXEC_HI]]]

; GCN: buffer_load_dword v[[RESULT:[0-9]+]], off, s[0:3], 0 offset:[[RESULT_OFFSET]] ; 4-byte Folded Reload

; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, v[[RESULT]]
define amdgpu_kernel void @divergent_if_else_endif(ptr addrspace(1) %out) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %load0 = load volatile i32, ptr addrspace(3) zeroinitializer
  %cmp0 = icmp eq i32 %tid, 0
  br i1 %cmp0, label %if, label %else

if:
  %load1 = load volatile i32, ptr addrspace(3) poison
  %val0 = add i32 %load0, %load1
  br label %endif

else:
  %load2 = load volatile i32, ptr addrspace(3) poison
  %val1 = sub i32 %load0, %load2
  br label %endif

endif:
  %result = phi i32 [ %val0, %if ], [ %val1, %else ]
  store i32 %result, ptr addrspace(1) %out
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdhsa_code_object_version", i32 400}
