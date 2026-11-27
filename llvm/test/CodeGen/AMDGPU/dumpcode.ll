; RUN: llc -amdgpu-late-wave-transform=1 -mtriple=amdgpu10.10 -mattr=dumpcode -filetype=obj < %s | llvm-objcopy --dump-section .AMDGPU.disasm=- - /dev/null | FileCheck %s -check-prefix=GFX10

; GFX10: f:
; GFX10-NEXT: BB0_0:
; GFX10-NEXT:   s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; BF8C0000
; GFX10-NEXT:   v_mov_b32_e32 v3, 0xde                  ; 7E0602FF 000000DE
; GFX10-NEXT:   v_add_nc_u32_e32 v2, 1, v4              ; 4A040881
; GFX10-NEXT:   s_mov_b32 s4, 0                         ; BE840380
; GFX10-NEXT:   global_store_dword v[0:1], v3, off      ; DC708000 007D0300
; GFX10-NEXT: BB0_1:
; GFX10-NEXT:   v_add_nc_u32_e32 v2, -1, v2             ; 4A0404C1
; GFX10-NEXT:   v_cmp_ne_u32_e64 s5, 0, v2              ; D4C50005 02020480
; GFX10-NEXT:   s_xor_b32 s6, exec_lo, s5               ; 8906057E
; GFX10-NEXT:   s_or_b32 s4, s4, s6                     ; 88040604
; GFX10-NEXT:   s_mov_b32 exec_lo, s5                   ; BEFE0305
; GFX10-NEXT:   s_cbranch_execnz ""                     ; BF890000
; GFX10-NEXT: BB0_2:
; GFX10-NEXT:   s_or_b32 exec_lo, exec_lo, s4           ; 887E047E
; GFX10-NEXT:   s_setpc_b64 s[30:31]                    ; BE80201E

define void @f(ptr addrspace(1) %out, ptr addrspace(1) %in, i32 %val) {
entry:
  br label %body
body:
  %i = phi i32 [0, %entry], [%inc, %body]
  store i32 222, ptr addrspace(1) %out
  %cmp = icmp ne i32 %i, %val
  %inc = add i32 %i, 1
  br i1 %cmp, label %body, label %end
end:
  ret void
}
