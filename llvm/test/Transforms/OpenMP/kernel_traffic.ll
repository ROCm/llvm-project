; Check that the per-iteration memory traffic estimate is emitted as a
; per-kernel global, that the three things that are easy to miscount (accesses
; outside a loop, accesses outside the global address space, and the outlined
; body of a parallel region) are handled, and that the estimate describes the
; heaviest loop nest rather than the sum over every loop.
;
; RUN: opt -passes=openmp-kernel-traffic -mtriple=amdgpu -S < %s | FileCheck %s
;
; Kernels without a loop marked by openmp-mark-work-loops are estimated by their
; heaviest loop nest. The marked-* kernels at the end check that a marked loop
; takes precedence.

@spmd_exec_mode = weak_odr protected addrspace(1) constant i8 2
@outer_exec_mode = weak_odr protected addrspace(1) constant i8 2
@nest_exec_mode = weak_odr protected addrspace(1) constant i8 2
@unrolled_exec_mode = weak_odr protected addrspace(1) constant i8 2
@duplicated_exec_mode = weak_odr protected addrspace(1) constant i8 2
@noopt_exec_mode = weak_odr protected addrspace(1) constant i8 2
@callee_exec_mode = weak_odr protected addrspace(1) constant i8 2
@shared_callee_exec_mode = weak_odr protected addrspace(1) constant i8 2
@callee_loop_exec_mode = weak_odr protected addrspace(1) constant i8 2
@indirect_exec_mode = weak_odr protected addrspace(1) constant i8 2
@memcpy_exec_mode = weak_odr protected addrspace(1) constant i8 2
@memset_exec_mode = weak_odr protected addrspace(1) constant i8 2
@memcpy_varlen_exec_mode = weak_odr protected addrspace(1) constant i8 2
@opaque_call_exec_mode = weak_odr protected addrspace(1) constant i8 2
@opaque_flat_call_exec_mode = weak_odr protected addrspace(1) constant i8 2
@private_flat_call_exec_mode = weak_odr protected addrspace(1) constant i8 2
@harmless_calls_exec_mode = weak_odr protected addrspace(1) constant i8 2
@callee_twice_exec_mode = weak_odr protected addrspace(1) constant i8 2
@callee_mixed_exec_mode = weak_odr protected addrspace(1) constant i8 2
@callee_nested_exec_mode = weak_odr protected addrspace(1) constant i8 2
@callee_recursive_exec_mode = weak_odr protected addrspace(1) constant i8 2
@callee_recursive_again_exec_mode = weak_odr protected addrspace(1) constant i8 2
@no_traffic_exec_mode = weak_odr protected addrspace(1) constant i8 2
@barrier_exec_mode = weak_odr protected addrspace(1) constant i8 1
@opaque_barrier_exec_mode = weak_odr protected addrspace(1) constant i8 1
@outer_opaque_exec_mode = weak_odr protected addrspace(1) constant i8 1
@param_call_exec_mode = weak_odr protected addrspace(1) constant i8 2
@param_call_twice_exec_mode = weak_odr protected addrspace(1) constant i8 2
@param_call_bindings_exec_mode = weak_odr protected addrspace(1) constant i8 2
@param_call_forwarded_exec_mode = weak_odr protected addrspace(1) constant i8 2
@param_stored_exec_mode = weak_odr protected addrspace(1) constant i8 2
@loop_bindings_exec_mode = weak_odr protected addrspace(1) constant i8 2
@flat_region_exec_mode = weak_odr protected addrspace(1) constant i8 2
@flat_callee_exec_mode = weak_odr protected addrspace(1) constant i8 2
@flat_local_exec_mode = weak_odr protected addrspace(1) constant i8 2
@marked_lighter_exec_mode = weak_odr protected addrspace(1) constant i8 2
@marked_compute_only_exec_mode = weak_odr protected addrspace(1) constant i8 2
@marked_in_callee_exec_mode = weak_odr protected addrspace(1) constant i8 2
@inner_const_exec_mode = weak_odr protected addrspace(1) constant i8 2
@inner_unknown_exec_mode = weak_odr protected addrspace(1) constant i8 2
@inner_in_callee_exec_mode = weak_odr protected addrspace(1) constant i8 2
@marked_around_exec_mode = weak_odr protected addrspace(1) constant i8 2
@select_private_exec_mode = weak_odr protected addrspace(1) constant i8 2
@select_globals_exec_mode = weak_odr protected addrspace(1) constant i8 2
@fn_slot = internal addrspace(3) global ptr poison
@lds_buf = internal addrspace(3) global [64 x double] poison

; Two 8-byte streams read inside the loop: 16 bytes over 2 streams. The load
; before the loop and the load from private scratch inside it are ignored.
; CHECK: @spmd_kernel_traffic = weak_odr protected addrspace(1) constant { i32, i32, i32, i32, i32, i32, i32, i32, i32 } { i32 16, i32 2, i32 16, i32 0, i32 2, i32 0, i32 4, i32 14, i32 0 }
define amdgpu_kernel void @spmd(ptr addrspace(1) %a, ptr addrspace(1) %b, i64 %n) {
entry:
  %scratch.priv = alloca [128 x double], align 8, addrspace(5)
  %scratch = addrspacecast ptr addrspace(5) %scratch.priv to ptr
  ; Prologue access: not per-iteration traffic.
  %hdr = load double, ptr addrspace(1) %a, align 8
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %acc = phi double [ %hdr, %entry ], [ %sum2, %loop ]
  %pa = getelementptr inbounds double, ptr addrspace(1) %a, i64 %i
  %va = load double, ptr addrspace(1) %pa, align 8
  %pb = getelementptr inbounds double, ptr addrspace(1) %b, i64 %i
  %vb = load double, ptr addrspace(1) %pb, align 8
  ; Private reduction scratch, accessed through a flat pointer: not counted.
  %ps = getelementptr inbounds double, ptr %scratch, i64 %i
  %vs = load double, ptr %ps, align 8
  %sum0 = fadd double %acc, %va
  %sum1 = fadd double %sum0, %vb
  %sum2 = fadd double %sum1, %vs
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp ult i64 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ; Epilogue access: not per-iteration traffic.
  store double %sum2, ptr addrspace(1) %b, align 8
  ret void
}

; The body of a parallel region is passed to the runtime as an argument, not as
; the callee, so it is only reachable through the call the runtime makes through
; its parameter.
define internal void @outlined(ptr addrspace(1) %a) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %p = getelementptr inbounds i32, ptr addrspace(1) %a, i64 %i
  %v = load i32, ptr addrspace(1) %p, align 4
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp ult i64 %i.next, 128
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

; One 4-byte stream, reached only through the outlined region: 4 bytes, 1 stream.
; CHECK: @outer_kernel_traffic = weak_odr protected addrspace(1) constant { i32, i32, i32, i32, i32, i32, i32, i32, i32 } { i32 4, i32 1, i32 4, i32 0, i32 1, i32 0, i32 1, i32 6, i32 0 }
define amdgpu_kernel void @outer(ptr addrspace(1) %a) {
  call void @run_region(ptr @outlined, ptr addrspace(1) %a)
  ret void
}

; Stands in for a runtime entry point that was not inlined.
define internal void @run_region(ptr %fn, ptr addrspace(1) %arg) noinline {
  call void %fn(ptr addrspace(1) %arg)
  ret void
}

; An inner loop's accesses are traffic of the enclosing iteration too, so the
; whole nest counts as one: 16 bytes over 2 streams.
; CHECK: @nest_kernel_traffic = weak_odr protected addrspace(1) constant { i32, i32, i32, i32, i32, i32, i32, i32, i32 } { i32 16, i32 2, i32 16, i32 0, i32 2, i32 0, i32 2, i32 13, i32 0 }
define amdgpu_kernel void @nest(ptr addrspace(1) %a, ptr addrspace(1) %b, i64 %n) {
entry:
  br label %outer

outer:
  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
  %pb = getelementptr inbounds double, ptr addrspace(1) %b, i64 %i
  %vb = load double, ptr addrspace(1) %pb, align 8
  br label %inner

inner:
  %j = phi i64 [ 0, %outer ], [ %j.next, %inner ]
  %pa = getelementptr inbounds double, ptr addrspace(1) %a, i64 %j
  %va = load double, ptr addrspace(1) %pa, align 8
  %j.next = add nuw nsw i64 %j, 1
  %jc = icmp ult i64 %j.next, %n
  br i1 %jc, label %inner, label %latch

latch:
  %i.next = add nuw nsw i64 %i, 1
  %ic = icmp ult i64 %i.next, %n
  br i1 %ic, label %outer, label %exit

exit:
  ret void
}

; The estimate follows the loop as the optimizer left it, so an unrolled body
; reports the traffic of the unrolled iteration: 32 bytes over 1 stream.
; CHECK: @unrolled_kernel_traffic = weak_odr protected addrspace(1) constant { i32, i32, i32, i32, i32, i32, i32, i32, i32 } { i32 32, i32 1, i32 32, i32 0, i32 4, i32 0, i32 4, i32 15, i32 0 }
define amdgpu_kernel void @unrolled(ptr addrspace(1) %a, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %p0 = getelementptr inbounds double, ptr addrspace(1) %a, i64 %i
  %v0 = load double, ptr addrspace(1) %p0, align 8
  %i1 = add nuw nsw i64 %i, 1
  %p1 = getelementptr inbounds double, ptr addrspace(1) %a, i64 %i1
  %v1 = load double, ptr addrspace(1) %p1, align 8
  %i2 = add nuw nsw i64 %i, 2
  %p2 = getelementptr inbounds double, ptr addrspace(1) %a, i64 %i2
  %v2 = load double, ptr addrspace(1) %p2, align 8
  %i3 = add nuw nsw i64 %i, 3
  %p3 = getelementptr inbounds double, ptr addrspace(1) %a, i64 %i3
  %v3 = load double, ptr addrspace(1) %p3, align 8
  %i.next = add nuw nsw i64 %i, 4
  %cmp = icmp ult i64 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

; Two copies of one loop that cannot both run: the heavier nest, 16 bytes over
; 2 streams, rather than their sum.
; CHECK: @duplicated_kernel_traffic = weak_odr protected addrspace(1) constant { i32, i32, i32, i32, i32, i32, i32, i32, i32 } { i32 16, i32 2, i32 16, i32 0, i32 2, i32 0, i32 1, i32 8, i32 0 }
define amdgpu_kernel void @duplicated(ptr addrspace(1) %a, ptr addrspace(1) %b, i64 %n, i1 %spmd) {
entry:
  br i1 %spmd, label %spmd.loop, label %generic.loop

spmd.loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %spmd.loop ]
  %pa = getelementptr inbounds double, ptr addrspace(1) %a, i64 %i
  %va = load double, ptr addrspace(1) %pa, align 8
  %pb = getelementptr inbounds double, ptr addrspace(1) %b, i64 %i
  %vb = load double, ptr addrspace(1) %pb, align 8
  %i.next = add nuw nsw i64 %i, 1
  %c = icmp ult i64 %i.next, %n
  br i1 %c, label %spmd.loop, label %exit

generic.loop:
  %gi = phi i64 [ 0, %entry ], [ %gi.next, %generic.loop ]
  %gpa = getelementptr inbounds double, ptr addrspace(1) %a, i64 %gi
  %gva = load double, ptr addrspace(1) %gpa, align 8
  %gpb = getelementptr inbounds double, ptr addrspace(1) %b, i64 %gi
  %gvb = load double, ptr addrspace(1) %gpb, align 8
  %gi.next = add nuw nsw i64 %gi, 1
  %gc = icmp ult i64 %gi.next, %n
  br i1 %gc, label %generic.loop, label %exit

exit:
  ret void
}

; A kernel that could not be inlined into keeps its -1: at -O0 the loops that
; are visible are not the kernel's work loop, so any number would be wrong. The
; status says why (1 = optnone).
; CHECK: @noopt_kernel_traffic = weak_odr protected addrspace(1) constant { i32, i32, i32, i32, i32, i32, i32, i32, i32 } { i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 1 }
define amdgpu_kernel void @noopt(ptr addrspace(1) %a, i64 %n) noinline optnone {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %p = getelementptr inbounds double, ptr addrspace(1) %a, i64 %i
  %v = load double, ptr addrspace(1) %p, align 8
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp ult i64 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

define internal double @loads_two(ptr addrspace(1) %a, ptr addrspace(1) %b, i64 %i) {
entry:
  %pa = getelementptr inbounds double, ptr addrspace(1) %a, i64 %i
  %va = load double, ptr addrspace(1) %pa, align 8
  %pb = getelementptr inbounds double, ptr addrspace(1) %b, i64 %i
  %vb = load double, ptr addrspace(1) %pb, align 8
  %sum = fadd double %va, %vb
  ret double %sum
}

; A callee with no loop of its own runs once per iteration of the loop that
; called it, so its accesses are that nest's traffic. Counting only what is
; syntactically inside the kernel's own loop would record 8 bytes over one
; stream here and put the kernel in the wrong regime; the whole nest is 24 over
; three.
; CHECK: @callee_kernel_traffic = weak_odr protected addrspace(1) constant { i32, i32, i32, i32, i32, i32, i32, i32, i32 } { i32 24, i32 3, i32 24, i32 0, i32 3, i32 0, i32 3, i32 13, i32 0 }
define amdgpu_kernel void @callee(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %c, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %pc = getelementptr inbounds double, ptr addrspace(1) %c, i64 %i
  %vc = load double, ptr addrspace(1) %pc, align 8
  %hidden = call double @loads_two(ptr addrspace(1) %a, ptr addrspace(1) %b, i64 %i)
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp ult i64 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

; A callee reached from two different nests contributes to each of them, rather
; than to whichever one happened to be walked first.
; CHECK: @shared_callee_kernel_traffic = weak_odr protected addrspace(1) constant { i32, i32, i32, i32, i32, i32, i32, i32, i32 } { i32 24, i32 3, i32 24, i32 0, i32 3, i32 0, i32 3, i32 13, i32 0 }
define amdgpu_kernel void @shared_callee(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %c, i64 %n) {
entry:
  br label %light

; The lighter nest is walked first and must not consume the callee's only visit.
light:
  %i = phi i64 [ 0, %entry ], [ %i.next, %light ]
  %hidden.light = call double @loads_two(ptr addrspace(1) %a, ptr addrspace(1) %b, i64 %i)
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp ult i64 %i.next, %n
  br i1 %cmp, label %light, label %heavy

heavy:
  %j = phi i64 [ 0, %light ], [ %j.next, %heavy ]
  %pc = getelementptr inbounds double, ptr addrspace(1) %c, i64 %j
  %vc = load double, ptr addrspace(1) %pc, align 8
  %hidden.heavy = call double @loads_two(ptr addrspace(1) %a, ptr addrspace(1) %b, i64 %j)
  %j.next = add nuw nsw i64 %j, 1
  %cmp2 = icmp ult i64 %j.next, %n
  br i1 %cmp2, label %heavy, label %exit

exit:
  ret void
}

define internal void @inner_loop(ptr addrspace(1) %b, i64 %n) {
entry:
  br label %loop

loop:
  %j = phi i64 [ 0, %entry ], [ %j.next, %loop ]
  %pb = getelementptr inbounds double, ptr addrspace(1) %b, i64 %j
  %vb = load double, ptr addrspace(1) %pb, align 8
  %j.next = add nuw nsw i64 %j, 1
  %cmp = icmp ult i64 %j.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

; A callee with a loop of its own still belongs to the nest that called it: the
; estimate must not depend on whether the helper happened to be inlined. Folding
; gives 16 bytes over two streams; treating the two loops as rival nests would
; report the heavier one alone, 8 over one.
; CHECK: @callee_loop_kernel_traffic = weak_odr protected addrspace(1) constant { i32, i32, i32, i32, i32, i32, i32, i32, i32 } { i32 16, i32 2, i32 16, i32 0, i32 2, i32 0, i32 3, i32 15, i32 0 }
define amdgpu_kernel void @callee_loop(ptr addrspace(1) %a, ptr addrspace(1) %b, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %pa = getelementptr inbounds double, ptr addrspace(1) %a, i64 %i
  %va = load double, ptr addrspace(1) %pa, align 8
  call void @inner_loop(ptr addrspace(1) %b, i64 %n)
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp ult i64 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

; An indirect call could carry any amount of this nest's traffic, so the kernel
; gets no estimate instead of a short one. The direct load is visible but would
; be the whole record, which is the failure mode this avoids (3 = indirect
; call).
; CHECK: @indirect_kernel_traffic = weak_odr protected addrspace(1) constant { i32, i32, i32, i32, i32, i32, i32, i32, i32 } { i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 3 }
define amdgpu_kernel void @indirect(ptr addrspace(1) %a, ptr %fn, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %p = getelementptr inbounds double, ptr addrspace(1) %a, i64 %i
  %v = load double, ptr addrspace(1) %p, align 8
  %hidden = call double %fn(ptr addrspace(1) %a, i64 %i)
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp ult i64 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

; An aggregate copy is a memcpy rather than loads and stores: 24 bytes read from
; one stream and written to another, 48 bytes over 2 streams.
; CHECK: @memcpy_kernel_traffic = weak_odr protected addrspace(1) constant { i32, i32, i32, i32, i32, i32, i32, i32, i32 } { i32 48, i32 2, i32 24, i32 24, i32 1, i32 1, i32 1, i32 7, i32 0 }
define amdgpu_kernel void @memcpy(ptr addrspace(1) %out, ptr addrspace(1) %in, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %pout = getelementptr inbounds [3 x double], ptr addrspace(1) %out, i64 %i
  %pin = getelementptr inbounds [3 x double], ptr addrspace(1) %in, i64 %i
  call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) %pout, ptr addrspace(1) %pin, i64 24, i1 false)
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp ult i64 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

; Only the global side of a memory intrinsic counts: the 16-byte memset of %out
; is a store, the copy out of %in into a private buffer is an 8-byte load. 24
; bytes over 2 streams.
; CHECK: @memset_kernel_traffic = weak_odr protected addrspace(1) constant { i32, i32, i32, i32, i32, i32, i32, i32, i32 } { i32 24, i32 2, i32 8, i32 16, i32 1, i32 1, i32 1, i32 8, i32 0 }
define amdgpu_kernel void @memset(ptr addrspace(1) %out, ptr addrspace(1) %in, i64 %n) {
entry:
  %tmp = alloca double, align 8, addrspace(5)
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %pout = getelementptr inbounds [2 x double], ptr addrspace(1) %out, i64 %i
  call void @llvm.memset.p1.i64(ptr addrspace(1) %pout, i8 0, i64 16, i1 false)
  %pin = getelementptr inbounds double, ptr addrspace(1) %in, i64 %i
  call void @llvm.memcpy.p5.p1.i64(ptr addrspace(5) %tmp, ptr addrspace(1) %pin, i64 8, i1 false)
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp ult i64 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

; A copy of unknown length could carry any amount of traffic: no estimate
; (5 = memory intrinsic of non-constant length).
; CHECK: @memcpy_varlen_kernel_traffic = weak_odr protected addrspace(1) constant { i32, i32, i32, i32, i32, i32, i32, i32, i32 } { i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 5 }
define amdgpu_kernel void @memcpy_varlen(ptr addrspace(1) %out, ptr addrspace(1) %in, i64 %len, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %pin = getelementptr inbounds double, ptr addrspace(1) %in, i64 %i
  %v = load double, ptr addrspace(1) %pin, align 8
  call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) %out, ptr addrspace(1) %in, i64 %len, i1 false)
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp ult i64 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

; A declaration that may access the global memory it is passed is as opaque as
; an indirect call: no estimate (4 = opaque call).
; CHECK: @opaque_call_kernel_traffic = weak_odr protected addrspace(1) constant { i32, i32, i32, i32, i32, i32, i32, i32, i32 } { i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 4 }
define amdgpu_kernel void @opaque_call(ptr addrspace(1) %a, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %p = getelementptr inbounds double, ptr addrspace(1) %a, i64 %i
  %v = load double, ptr addrspace(1) %p, align 8
  call void @external(ptr addrspace(1) %a, i64 %i)
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp ult i64 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

; A flat pointer passed to such a declaration may point to global memory, just
; like the pointer of a flat load or store: no estimate.
; CHECK: @opaque_flat_call_kernel_traffic = weak_odr protected addrspace(1) constant { i32, i32, i32, i32, i32, i32, i32, i32, i32 } { i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 4 }
define amdgpu_kernel void @opaque_flat_call(ptr %a, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %p = getelementptr inbounds double, ptr %a, i64 %i
  %v = load double, ptr %p, align 8
  call void @flat_argmem(ptr %p, double %v)
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp ult i64 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

; A flat pointer that provably points to private memory does not block the
; estimate. One 8-byte stream.
; CHECK: @private_flat_call_kernel_traffic = weak_odr protected addrspace(1) constant { i32, i32, i32, i32, i32, i32, i32, i32, i32 } { i32 8, i32 1, i32 8, i32 0, i32 1, i32 0, i32 2, i32 7, i32 0 }
define amdgpu_kernel void @private_flat_call(ptr addrspace(1) %a, i64 %n) {
entry:
  %tmp = alloca double, align 8, addrspace(5)
  %tmp.flat = addrspacecast ptr addrspace(5) %tmp to ptr
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %p = getelementptr inbounds double, ptr addrspace(1) %a, i64 %i
  %v = load double, ptr addrspace(1) %p, align 8
  call void @flat_argmem(ptr %tmp.flat, double %v)
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp ult i64 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

; Declarations that cannot reach global memory do not block the estimate:
; lifetime markers, a call without memory effects, and one that only touches
; the private memory it is passed. One 8-byte stream.
; CHECK: @harmless_calls_kernel_traffic = weak_odr protected addrspace(1) constant { i32, i32, i32, i32, i32, i32, i32, i32, i32 } { i32 8, i32 1, i32 8, i32 0, i32 1, i32 0, i32 5, i32 10, i32 0 }
define amdgpu_kernel void @harmless_calls(ptr addrspace(1) %a, i64 %n) {
entry:
  %tmp = alloca double, align 8, addrspace(5)
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %p = getelementptr inbounds double, ptr addrspace(1) %a, i64 %i
  %v = load double, ptr addrspace(1) %p, align 8
  call void @llvm.lifetime.start.p5(ptr addrspace(5) %tmp)
  %r = call double @pure(double %v)
  call void @private_only(ptr addrspace(5) %tmp, double %r)
  call void @llvm.lifetime.end.p5(ptr addrspace(5) %tmp)
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp ult i64 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

declare void @external(ptr addrspace(1), i64)
declare double @pure(double) memory(none)
declare void @private_only(ptr addrspace(5), double) memory(argmem: readwrite)
declare void @flat_argmem(ptr, double) memory(argmem: readwrite)

; Helpers for the tests below that are not inlined.
define internal double @load_one(ptr addrspace(1) %x, i64 %i) noinline {
  %p = getelementptr inbounds double, ptr addrspace(1) %x, i64 %i
  %v = load double, ptr addrspace(1) %p, align 8
  ret double %v
}

define internal double @load_pair(ptr addrspace(1) %p, ptr addrspace(1) %q, i64 %i) noinline {
  %vp = call double @load_one(ptr addrspace(1) %p, i64 %i)
  %vq = call double @load_one(ptr addrspace(1) %q, i64 %i)
  %sum = fadd double %vp, %vq
  ret double %sum
}

; Each call adds the callee's accesses again, and each is attributed to the array
; the call passes: 16 bytes over 2 streams, as if @load_one were inlined. Counting
; the callee once per nest, or keying its accesses by its own parameter, would
; give 8 over 1.
; CHECK: @callee_twice_kernel_traffic = weak_odr protected addrspace(1) constant { i32, i32, i32, i32, i32, i32, i32, i32, i32 } { i32 16, i32 2, i32 16, i32 0, i32 2, i32 0, i32 3, i32 12, i32 0 }
define amdgpu_kernel void @callee_twice(ptr addrspace(1) %a, ptr addrspace(1) %b, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %va = call double @load_one(ptr addrspace(1) %a, i64 %i)
  %vb = call double @load_one(ptr addrspace(1) %b, i64 %i)
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp ult i64 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

; An array accessed directly and through a call is one stream: 16 bytes over 1,
; not 2.
; CHECK: @callee_mixed_kernel_traffic = weak_odr protected addrspace(1) constant { i32, i32, i32, i32, i32, i32, i32, i32, i32 } { i32 16, i32 1, i32 16, i32 0, i32 2, i32 0, i32 2, i32 10, i32 0 }
define amdgpu_kernel void @callee_mixed(ptr addrspace(1) %a, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %p = getelementptr inbounds double, ptr addrspace(1) %a, i64 %i
  %v = load double, ptr addrspace(1) %p, align 8
  %w = call double @load_one(ptr addrspace(1) %a, i64 %i)
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp ult i64 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

; Parameters are mapped through every level of calls: @load_one's accesses reach
; %a and %b via @load_pair's parameters, and %a is also read directly. 24 bytes
; over 2 streams.
; CHECK: @callee_nested_kernel_traffic = weak_odr protected addrspace(1) constant { i32, i32, i32, i32, i32, i32, i32, i32, i32 } { i32 24, i32 2, i32 24, i32 0, i32 3, i32 0, i32 5, i32 17, i32 0 }
define amdgpu_kernel void @callee_nested(ptr addrspace(1) %a, ptr addrspace(1) %b, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %p = getelementptr inbounds double, ptr addrspace(1) %a, i64 %i
  %v = load double, ptr addrspace(1) %p, align 8
  %w = call double @load_pair(ptr addrspace(1) %a, ptr addrspace(1) %b, i64 %i)
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp ult i64 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

define internal double @recurse(ptr addrspace(1) %x, i64 %i) {
entry:
  %p = getelementptr inbounds double, ptr addrspace(1) %x, i64 %i
  %v = load double, ptr addrspace(1) %p, align 8
  %done = icmp eq i64 %i, 0
  br i1 %done, label %exit, label %again

again:
  %i.prev = sub nuw i64 %i, 1
  %r = call double @recurse(ptr addrspace(1) %x, i64 %i.prev)
  br label %exit

exit:
  ret double %v
}

; A recursive callee may run any number of times per iteration: no estimate
; (2 = recursive).
; CHECK: @callee_recursive_kernel_traffic = weak_odr protected addrspace(1) constant { i32, i32, i32, i32, i32, i32, i32, i32, i32 } { i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 2 }
define amdgpu_kernel void @callee_recursive(ptr addrspace(1) %a, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %v = call double @recurse(ptr addrspace(1) %a, i64 %i)
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp ult i64 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

; A second kernel reaching the same recursive callee gets the reason, too: a
; callee that is not analyzable is cached along with why.
; CHECK: @callee_recursive_again_kernel_traffic = weak_odr protected addrspace(1) constant { i32, i32, i32, i32, i32, i32, i32, i32, i32 } { i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 2 }
define amdgpu_kernel void @callee_recursive_again(ptr addrspace(1) %a, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %v = call double @recurse(ptr addrspace(1) %a, i64 %i)
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp ult i64 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

; A kernel whose loop touches no global memory more likely hides its work loop
; than does nothing: no estimate (7 = no traffic).
; CHECK: @no_traffic_kernel_traffic = weak_odr protected addrspace(1) constant { i32, i32, i32, i32, i32, i32, i32, i32, i32 } { i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 7 }
define amdgpu_kernel void @no_traffic(ptr addrspace(3) %a, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %p = getelementptr inbounds double, ptr addrspace(3) %a, i64 %i
  %v = load double, ptr addrspace(3) %p, align 8
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp ult i64 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

; A generic-mode parallel region inside a loop brings its barriers into the
; nest. They synchronize but move no data: one 8-byte stream.
; CHECK: @barrier_kernel_traffic = weak_odr protected addrspace(1) constant { i32, i32, i32, i32, i32, i32, i32, i32, i32 } { i32 8, i32 1, i32 8, i32 0, i32 1, i32 0, i32 3, i32 8, i32 0 }
define amdgpu_kernel void @barrier(ptr addrspace(1) %a, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  call void @llvm.amdgcn.s.barrier()
  %p = getelementptr inbounds double, ptr addrspace(1) %a, i64 %i
  %v = load double, ptr addrspace(1) %p, align 8
  call void @llvm.amdgcn.s.barrier()
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp ult i64 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

declare void @llvm.amdgcn.s.barrier()

; Only barriers are exempt: another side-effecting call without memory effects
; may still move data, so the kernel gets no estimate.
; CHECK: @opaque_barrier_kernel_traffic = weak_odr protected addrspace(1) constant { i32, i32, i32, i32, i32, i32, i32, i32, i32 } { i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 4 }
define amdgpu_kernel void @opaque_barrier(ptr addrspace(1) %a, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  call void @my_barrier()
  %p = getelementptr inbounds double, ptr addrspace(1) %a, i64 %i
  %v = load double, ptr addrspace(1) %p, align 8
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp ult i64 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

declare void @my_barrier() convergent nounwind

; A declaration may run the region it is passed, but only its memory effects
; can say what that does. Without any, the kernel gets no estimate, even though
; the region's loop would be visible.
; CHECK: @outer_opaque_kernel_traffic = weak_odr protected addrspace(1) constant { i32, i32, i32, i32, i32, i32, i32, i32, i32 } { i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 4 }
define amdgpu_kernel void @outer_opaque(ptr addrspace(1) %a) {
  call void @__kmpc_parallel_51(ptr null, i32 0, i32 1, i32 -1, i32 -1, ptr @outlined, ptr null, ptr addrspace(1) %a, i64 1)
  ret void
}

declare void @__kmpc_parallel_51(ptr, i32, i32, i32, i32, ptr, ptr, ptr addrspace(1), i64)

define internal double @store_one(ptr addrspace(1) %x, i64 %i) noinline {
  %p = getelementptr inbounds double, ptr addrspace(1) %x, i64 %i
  store double 0.0, ptr addrspace(1) %p, align 8
  ret double 0.0
}

; Calls whatever it is passed once.
define internal double @apply(ptr %fn, ptr addrspace(1) %x, i64 %i) noinline {
  %v = call double %fn(ptr addrspace(1) %x, i64 %i)
  ret double %v
}

; A call through a parameter inside a loop runs what the caller passed for it,
; and the accesses are mapped through both calls: 8 bytes over 1 stream, %a.
; CHECK: @param_call_kernel_traffic = weak_odr protected addrspace(1) constant { i32, i32, i32, i32, i32, i32, i32, i32, i32 } { i32 8, i32 1, i32 8, i32 0, i32 1, i32 0, i32 3, i32 10, i32 0 }
define amdgpu_kernel void @param_call(ptr addrspace(1) %a, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %v = call double @apply(ptr @load_one, ptr addrspace(1) %a, i64 %i)
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp ult i64 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

define internal double @apply_twice(ptr %fn, ptr addrspace(1) %x, ptr addrspace(1) %y, i64 %i) noinline {
  %vx = call double %fn(ptr addrspace(1) %x, i64 %i)
  %vy = call double %fn(ptr addrspace(1) %y, i64 %i)
  %sum = fadd double %vx, %vy
  ret double %sum
}

; Each call through the parameter counts, each with its own arguments: 16 bytes
; over 2 streams.
; CHECK: @param_call_twice_kernel_traffic = weak_odr protected addrspace(1) constant { i32, i32, i32, i32, i32, i32, i32, i32, i32 } { i32 16, i32 2, i32 16, i32 0, i32 2, i32 0, i32 5, i32 15, i32 0 }
define amdgpu_kernel void @param_call_twice(ptr addrspace(1) %a, ptr addrspace(1) %b, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %v = call double @apply_twice(ptr @load_one, ptr addrspace(1) %a, ptr addrspace(1) %b, i64 %i)
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp ult i64 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

; The same function passed two different functions is two different calls: a
; load from %a and a store to %b. Reusing the first call's summary for the
; second would report two loads.
; CHECK: @param_call_bindings_kernel_traffic = weak_odr protected addrspace(1) constant { i32, i32, i32, i32, i32, i32, i32, i32, i32 } { i32 16, i32 2, i32 8, i32 8, i32 1, i32 1, i32 5, i32 16, i32 0 }
define amdgpu_kernel void @param_call_bindings(ptr addrspace(1) %a, ptr addrspace(1) %b, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %va = call double @apply(ptr @load_one, ptr addrspace(1) %a, i64 %i)
  %vb = call double @apply(ptr @store_one, ptr addrspace(1) %b, i64 %i)
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp ult i64 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

define internal double @forward(ptr %f, ptr addrspace(1) %x, i64 %i) noinline {
  %v = call double @apply(ptr %f, ptr addrspace(1) %x, i64 %i)
  ret double %v
}

; A function pointer forwarded through another parameter still resolves: 8
; bytes over 1 stream.
; CHECK: @param_call_forwarded_kernel_traffic = weak_odr protected addrspace(1) constant { i32, i32, i32, i32, i32, i32, i32, i32, i32 } { i32 8, i32 1, i32 8, i32 0, i32 1, i32 0, i32 4, i32 12, i32 0 }
define amdgpu_kernel void @param_call_forwarded(ptr addrspace(1) %a, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %v = call double @forward(ptr @load_one, ptr addrspace(1) %a, i64 %i)
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp ult i64 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

; Keeps what it is passed for later rather than calling it.
define internal void @stash(ptr %fn) noinline {
  store ptr %fn, ptr addrspace(3) @fn_slot, align 8
  ret void
}

; Passing a function does not run it: only the direct load, 8 bytes over 1
; stream. Counting @load_one as well would give 16.
; CHECK: @param_stored_kernel_traffic = weak_odr protected addrspace(1) constant { i32, i32, i32, i32, i32, i32, i32, i32, i32 } { i32 8, i32 1, i32 8, i32 0, i32 1, i32 0, i32 2, i32 9, i32 0 }
define amdgpu_kernel void @param_stored(ptr addrspace(1) %a, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %p = getelementptr inbounds double, ptr addrspace(1) %a, i64 %i
  %v = load double, ptr addrspace(1) %p, align 8
  call void @stash(ptr @load_one)
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp ult i64 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

; Runs a loop that calls whatever it is passed.
define internal void @apply_loop(ptr %fn, ptr addrspace(1) %x, i64 %n) noinline {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %v = call double %fn(ptr addrspace(1) %x, i64 %i)
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp ult i64 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

; One loop run with two bindings is two nests, not one: the heavier of an
; 8-byte load and an 8-byte store, rather than both added up (16 bytes).
; CHECK: @loop_bindings_kernel_traffic = weak_odr protected addrspace(1) constant { i32, i32, i32, i32, i32, i32, i32, i32, i32 } { i32 8, i32 1, i32 0, i32 8, i32 0, i32 1, i32 2, i32 8, i32 0 }
define amdgpu_kernel void @loop_bindings(ptr addrspace(1) %a, i64 %n) {
  call void @apply_loop(ptr @load_one, ptr addrspace(1) %a, i64 %n)
  call void @apply_loop(ptr @store_one, ptr addrspace(1) %a, i64 %n)
  ret void
}

; The outlined body of a parallel region that was not inlined yet: its
; parameters are flat, so the address spaces of its accesses were never
; inferred.
define internal void @flat_body(ptr %in, ptr %out, i64 %n) noinline {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %pin = getelementptr inbounds double, ptr %in, i64 %i
  %v = load double, ptr %pin, align 8
  %pout = getelementptr inbounds double, ptr %out, i64 %i
  store double %v, ptr %pout, align 8
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp ult i64 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

; Flat accesses that are not provably private or LDS count as global, so the
; work loop in the outlined body is found: 16 bytes over 2 streams.
; CHECK: @flat_region_kernel_traffic = weak_odr protected addrspace(1) constant { i32, i32, i32, i32, i32, i32, i32, i32, i32 } { i32 16, i32 2, i32 8, i32 8, i32 1, i32 1, i32 1, i32 8, i32 0 }
define amdgpu_kernel void @flat_region(ptr %in, ptr %out, i64 %n) {
  call void @flat_body(ptr %in, ptr %out, i64 %n)
  ret void
}

define internal double @flat_load(ptr %x, i64 %i) noinline {
  %p = getelementptr inbounds double, ptr %x, i64 %i
  %v = load double, ptr %p, align 8
  ret double %v
}

; A global array read directly and through a flat pointer in a callee is one
; stream, not two: 16 bytes over 1 stream.
; CHECK: @flat_callee_kernel_traffic = weak_odr protected addrspace(1) constant { i32, i32, i32, i32, i32, i32, i32, i32, i32 } { i32 16, i32 1, i32 16, i32 0, i32 2, i32 0, i32 2, i32 10, i32 0 }
define amdgpu_kernel void @flat_callee(ptr addrspace(1) %a, i64 %n) {
entry:
  %a.flat = addrspacecast ptr addrspace(1) %a to ptr
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %p = getelementptr inbounds double, ptr addrspace(1) %a, i64 %i
  %v = load double, ptr addrspace(1) %p, align 8
  %w = call double @flat_load(ptr %a.flat, i64 %i)
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp ult i64 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

; Flat pointers to private memory and LDS are not global traffic, neither for
; loads and stores nor for memory intrinsics. What is left is an 8-byte load
; and an 8-byte copy out of the flat kernel argument %in: 16 bytes over 1
; stream.
; CHECK: @flat_local_kernel_traffic = weak_odr protected addrspace(1) constant { i32, i32, i32, i32, i32, i32, i32, i32, i32 } { i32 16, i32 1, i32 16, i32 0, i32 2, i32 0, i32 1, i32 11, i32 0 }
define amdgpu_kernel void @flat_local(ptr %in, i64 %n) {
entry:
  %priv = alloca [64 x double], align 8, addrspace(5)
  %priv.flat = addrspacecast ptr addrspace(5) %priv to ptr
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %pin = getelementptr inbounds double, ptr %in, i64 %i
  %v = load double, ptr %pin, align 8
  %ppriv = getelementptr inbounds double, ptr %priv.flat, i64 %i
  store double %v, ptr %ppriv, align 8
  %plds = getelementptr inbounds double, ptr addrspacecast (ptr addrspace(3) @lds_buf to ptr), i64 %i
  store double %v, ptr %plds, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr %ppriv, ptr %pin, i64 8, i1 false)
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp ult i64 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

; The loop of a worksharing construct is the work loop, even if the kernel has
; heavier loops, e.g. a cross-team reduction over the partial results of every
; team: 8 bytes over 1 stream, not 16 over 1.
; CHECK: @marked_lighter_kernel_traffic = weak_odr protected addrspace(1) constant { i32, i32, i32, i32, i32, i32, i32, i32, i32 } { i32 8, i32 1, i32 8, i32 0, i32 1, i32 0, i32 1, i32 6, i32 0 }
define amdgpu_kernel void @marked_lighter(ptr addrspace(1) %a, ptr addrspace(1) %partials, i64 %n, i64 %teams) {
entry:
  br label %work

work:
  %i = phi i64 [ 0, %entry ], [ %i.next, %work ]
  %pa = getelementptr inbounds double, ptr addrspace(1) %a, i64 %i
  %va = load double, ptr addrspace(1) %pa, align 8
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp ult i64 %i.next, %n
  br i1 %cmp, label %work, label %epilogue, !llvm.loop !0

epilogue:
  %j = phi i64 [ 0, %work ], [ %j.next, %epilogue ]
  %pp = getelementptr inbounds [2 x double], ptr addrspace(1) %partials, i64 %j
  %vp = load double, ptr addrspace(1) %pp, align 8
  %pq = getelementptr inbounds double, ptr addrspace(1) %pp, i64 1
  %vq = load double, ptr addrspace(1) %pq, align 8
  %j.next = add nuw nsw i64 %j, 1
  %jc = icmp ult i64 %j.next, %teams
  br i1 %jc, label %epilogue, label %exit

exit:
  ret void
}

; A work loop that only computes is a finding, not a failure to see the work
; loop: no traffic, but its compute ops.
; CHECK: @marked_compute_only_kernel_traffic = weak_odr protected addrspace(1) constant { i32, i32, i32, i32, i32, i32, i32, i32, i32 } { i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 3, i32 8, i32 0 }
define amdgpu_kernel void @marked_compute_only(ptr addrspace(1) %out, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %acc = phi double [ 0.0, %entry ], [ %acc.next, %loop ]
  %f = uitofp i64 %i to double
  %t = fdiv double 1.0, %f
  %acc.next = fadd double %acc, %t
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp ult i64 %i.next, %n
  br i1 %cmp, label %loop, label %exit, !llvm.loop !2

exit:
  store double %acc.next, ptr addrspace(1) %out, align 8
  ret void
}

; Runs the loop of a worksharing construct, e.g. an outlined parallel region.
define internal void @marked_body(ptr addrspace(1) %a, i64 %n) noinline {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %pa = getelementptr inbounds double, ptr addrspace(1) %a, i64 %i
  %va = load double, ptr addrspace(1) %pa, align 8
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp ult i64 %i.next, %n
  br i1 %cmp, label %loop, label %exit, !llvm.loop !3

exit:
  ret void
}

; A nest that calls the work loop runs it, too: the loop around the call is
; the work loop's nest, 8 bytes over 1 stream, rather than the heavier nest
; after it.
; CHECK: @marked_in_callee_kernel_traffic = weak_odr protected addrspace(1) constant { i32, i32, i32, i32, i32, i32, i32, i32, i32 } { i32 8, i32 1, i32 8, i32 0, i32 1, i32 0, i32 3, i32 13, i32 0 }
define amdgpu_kernel void @marked_in_callee(ptr addrspace(1) %a, ptr addrspace(1) %partials, i64 %n, i64 %teams) {
entry:
  br label %outer

outer:
  %k = phi i64 [ 0, %entry ], [ %k.next, %outer ]
  call void @marked_body(ptr addrspace(1) %a, i64 %n)
  %k.next = add nuw nsw i64 %k, 1
  %kc = icmp ult i64 %k.next, 4
  br i1 %kc, label %outer, label %epilogue

epilogue:
  %j = phi i64 [ 0, %outer ], [ %j.next, %epilogue ]
  %pp = getelementptr inbounds [2 x double], ptr addrspace(1) %partials, i64 %j
  %vp = load double, ptr addrspace(1) %pp, align 8
  %pq = getelementptr inbounds double, ptr addrspace(1) %pp, i64 1
  %vq = load double, ptr addrspace(1) %pq, align 8
  %j.next = add nuw nsw i64 %j, 1
  %jc = icmp ult i64 %j.next, %teams
  br i1 %jc, label %epilogue, label %exit

exit:
  ret void
}

; An inner loop with a constant trip count runs that many times per iteration
; of the nest: 4 loads of 8 bytes, 32 bytes over 1 stream.
; CHECK: @inner_const_kernel_traffic = weak_odr protected addrspace(1) constant { i32, i32, i32, i32, i32, i32, i32, i32, i32 } { i32 32, i32 1, i32 32, i32 0, i32 4, i32 0, i32 10, i32 34, i32 0 }
define amdgpu_kernel void @inner_const(ptr addrspace(1) %a, i64 %n) {
entry:
  br label %outer

outer:
  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
  %base = mul nuw nsw i64 %i, 4
  br label %inner

inner:
  %j = phi i64 [ 0, %outer ], [ %j.next, %inner ]
  %idx = add nuw nsw i64 %base, %j
  %p = getelementptr inbounds double, ptr addrspace(1) %a, i64 %idx
  %v = load double, ptr addrspace(1) %p, align 8
  %j.next = add nuw nsw i64 %j, 1
  %jc = icmp ult i64 %j.next, 4
  br i1 %jc, label %inner, label %latch

latch:
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp ult i64 %i.next, %n
  br i1 %cmp, label %outer, label %exit

exit:
  ret void
}

; An inner loop whose trip count is unknown counts once: 8 bytes over 1 stream.
; CHECK: @inner_unknown_kernel_traffic = weak_odr protected addrspace(1) constant { i32, i32, i32, i32, i32, i32, i32, i32, i32 } { i32 8, i32 1, i32 8, i32 0, i32 1, i32 0, i32 4, i32 13, i32 0 }
define amdgpu_kernel void @inner_unknown(ptr addrspace(1) %a, i64 %n, i64 %m) {
entry:
  br label %outer

outer:
  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
  %base = mul nuw nsw i64 %i, %m
  br label %inner

inner:
  %j = phi i64 [ 0, %outer ], [ %j.next, %inner ]
  %idx = add nuw nsw i64 %base, %j
  %p = getelementptr inbounds double, ptr addrspace(1) %a, i64 %idx
  %v = load double, ptr addrspace(1) %p, align 8
  %j.next = add nuw nsw i64 %j, 1
  %jc = icmp ult i64 %j.next, %m
  br i1 %jc, label %inner, label %latch

latch:
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp ult i64 %i.next, %n
  br i1 %cmp, label %outer, label %exit

exit:
  ret void
}

define internal void @load_four(ptr addrspace(1) %x, i64 %i) noinline {
entry:
  %base = mul nuw nsw i64 %i, 4
  br label %loop

loop:
  %j = phi i64 [ 0, %entry ], [ %j.next, %loop ]
  %idx = add nuw nsw i64 %base, %j
  %p = getelementptr inbounds double, ptr addrspace(1) %x, i64 %idx
  %v = load double, ptr addrspace(1) %p, align 8
  %j.next = add nuw nsw i64 %j, 1
  %jc = icmp ult i64 %j.next, 4
  br i1 %jc, label %loop, label %exit

exit:
  ret void
}

; The loops of a callee run as often per call as they iterate: 32 bytes over 1
; stream, as if @load_four were inlined.
; CHECK: @inner_in_callee_kernel_traffic = weak_odr protected addrspace(1) constant { i32, i32, i32, i32, i32, i32, i32, i32, i32 } { i32 32, i32 1, i32 32, i32 0, i32 4, i32 0, i32 11, i32 36, i32 0 }
define amdgpu_kernel void @inner_in_callee(ptr addrspace(1) %a, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  call void @load_four(ptr addrspace(1) %a, i64 %i)
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp ult i64 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

; Iterations are counted in the work loop. Only the loop inside it multiplies;
; the loop around it and the one next to it do not, even though their trip
; counts are known, and neither does the work loop itself: 2 loads from %a and
; 1 from %b, 24 bytes over 2 streams.
; CHECK: @marked_around_kernel_traffic = weak_odr protected addrspace(1) constant { i32, i32, i32, i32, i32, i32, i32, i32, i32 } { i32 24, i32 2, i32 24, i32 0, i32 3, i32 0, i32 8, i32 31, i32 0 }
define amdgpu_kernel void @marked_around(ptr addrspace(1) %a, ptr addrspace(1) %b) {
entry:
  br label %repeat

repeat:
  %r = phi i64 [ 0, %entry ], [ %r.next, %repeat.latch ]
  br label %work

work:
  %i = phi i64 [ 0, %repeat ], [ %i.next, %work.latch ]
  %base = mul nuw nsw i64 %i, 2
  br label %inner

inner:
  %j = phi i64 [ 0, %work ], [ %j.next, %inner ]
  %idx = add nuw nsw i64 %base, %j
  %pa = getelementptr inbounds double, ptr addrspace(1) %a, i64 %idx
  %va = load double, ptr addrspace(1) %pa, align 8
  %j.next = add nuw nsw i64 %j, 1
  %jc = icmp ult i64 %j.next, 2
  br i1 %jc, label %inner, label %work.latch

work.latch:
  %i.next = add nuw nsw i64 %i, 1
  %ic = icmp ult i64 %i.next, 8
  br i1 %ic, label %work, label %next, !llvm.loop !4

next:
  %k = phi i64 [ 0, %work.latch ], [ %k.next, %next ]
  %pb = getelementptr inbounds double, ptr addrspace(1) %b, i64 %k
  %vb = load double, ptr addrspace(1) %pb, align 8
  %k.next = add nuw nsw i64 %k, 1
  %kc = icmp ult i64 %k.next, 5
  br i1 %kc, label %next, label %repeat.latch

repeat.latch:
  %r.next = add nuw nsw i64 %r, 1
  %rc = icmp ult i64 %r.next, 3
  br i1 %rc, label %repeat, label %exit

exit:
  ret void
}

; A pointer that selects between an array element and a private copy, e.g.
; the reference std::max() returns, is an access to the array: 16 bytes over
; 1 stream, not 2. A select between two private copies is no global access.
; CHECK: @select_private_kernel_traffic = weak_odr protected addrspace(1) constant { i32, i32, i32, i32, i32, i32, i32, i32, i32 } { i32 16, i32 1, i32 16, i32 0, i32 2, i32 0, i32 1, i32 14, i32 0 }
define amdgpu_kernel void @select_private(ptr %a, i64 %n) {
entry:
  %m = alloca double, align 8, addrspace(5)
  %m.flat = addrspacecast ptr addrspace(5) %m to ptr
  %t = alloca double, align 8, addrspace(5)
  %t.flat = addrspacecast ptr addrspace(5) %t to ptr
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %p = getelementptr inbounds double, ptr %a, i64 %i
  %v = load double, ptr %p, align 8
  %cur = load double, ptr %m.flat, align 8
  %c = fcmp olt double %cur, %v
  %max = select i1 %c, ptr %p, ptr %m.flat
  %mv = load double, ptr %max, align 8
  store double %mv, ptr %m.flat, align 8
  %priv = select i1 %c, ptr %t.flat, ptr %m.flat
  %pv = load double, ptr %priv, align 8
  store double %pv, ptr %t.flat, align 8
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp ult i64 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

; A select between two arrays may reach either, so it is a stream of its own:
; 16 bytes over 2 streams.
; CHECK: @select_globals_kernel_traffic = weak_odr protected addrspace(1) constant { i32, i32, i32, i32, i32, i32, i32, i32, i32 } { i32 16, i32 2, i32 16, i32 0, i32 2, i32 0, i32 1, i32 10, i32 0 }
define amdgpu_kernel void @select_globals(ptr addrspace(1) %a, ptr addrspace(1) %b, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %pa = getelementptr inbounds double, ptr addrspace(1) %a, i64 %i
  %va = load double, ptr addrspace(1) %pa, align 8
  %pb = getelementptr inbounds double, ptr addrspace(1) %b, i64 %i
  %c = fcmp olt double %va, 0.0
  %sel = select i1 %c, ptr addrspace(1) %pa, ptr addrspace(1) %pb
  %vs = load double, ptr addrspace(1) %sel, align 8
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp ult i64 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.omp.work"}
!2 = distinct !{!2, !1}
!3 = distinct !{!3, !1}
!4 = distinct !{!4, !1}
