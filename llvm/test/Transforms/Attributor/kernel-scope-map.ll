; NOTE: Test the kernel scope map computation in InformationCache::getEnclosingKernels.
; The map is computed lazily from the static IR call graph and logs via LLVM_DEBUG.

; REQUIRES: asserts
; RUN: opt -aa-pipeline=basic-aa -passes=attributor -attributor-manifest-internal \
; RUN:   -attributor-annotate-decl-cs -debug-only=attributor -S < %s 2>&1 \
; RUN:   | FileCheck %s --check-prefix=SCOPE
; RUN: opt -aa-pipeline=basic-aa -passes=attributor -attributor-manifest-internal \
; RUN:   -attributor-annotate-decl-cs -stats -S < %s 2>&1 \
; RUN:   | FileCheck %s --check-prefix=STATS

target triple = "amdgcn-amd-amdhsa"

; A kernel-lifetime global (addrspace(3) = LDS).
; Its presence triggers the kernel scope map computation.
@G = internal addrspace(3) global i32 0, align 4

; --- Kernel A and its callees ---

define dso_local void @kernel_A() norecurse "kernel" {
  call void @helper_A()
  call void @runtime_like(ptr @callback_A)
  call void @shared_helper()
  %p = load ptr, ptr @fnptr_global
  call void %p()
  ret void
}

define internal void @helper_A() {
  call void @transitive_helper()
  ret void
}

define internal void @transitive_helper() {
  store i32 1, ptr addrspacecast (ptr addrspace(3) @G to ptr)
  ret void
}

define internal void @callback_A() {
  store i32 2, ptr addrspacecast (ptr addrspace(3) @G to ptr)
  ret void
}

; --- Kernel B and its callees ---

define dso_local void @kernel_B() norecurse "kernel" {
  call void @helper_B()
  call void @shared_helper()
  ret void
}

define internal void @helper_B() {
  store i32 3, ptr addrspacecast (ptr addrspace(3) @G to ptr)
  ret void
}

; --- Shared callee (reachable from both kernels) ---

define internal void @shared_helper() {
  store i32 4, ptr addrspacecast (ptr addrspace(3) @G to ptr)
  ret void
}

; --- Runtime-like function that receives a callback ---

define internal void @runtime_like(ptr %fn) {
  call void %fn()
  ret void
}

; --- Indirect-only function (only stored to a global, never directly called) ---

@fnptr_global = internal global ptr @indirect_only

define internal void @indirect_only() {
  store i32 5, ptr addrspacecast (ptr addrspace(3) @G to ptr)
  ret void
}

; Check that functions are mapped to the correct enclosing kernels.
; The output is sorted by function name.

; callback_A is passed as a function pointer argument to runtime_like
; from kernel_A, so argument scanning should discover it under kernel_A.
; SCOPE-DAG: [KernelScope]   callback_A -> {kernel_A} (1 kernel(s))

; helper_A is directly called by kernel_A.
; SCOPE-DAG: [KernelScope]   helper_A -> {kernel_A} (1 kernel(s))

; helper_B is directly called by kernel_B.
; SCOPE-DAG: [KernelScope]   helper_B -> {kernel_B} (1 kernel(s))

; kernel_A maps to itself.
; SCOPE-DAG: [KernelScope]   kernel_A -> {kernel_A} (1 kernel(s))

; kernel_B maps to itself.
; SCOPE-DAG: [KernelScope]   kernel_B -> {kernel_B} (1 kernel(s))

; runtime_like is called from kernel_A (via direct call).
; SCOPE-DAG: [KernelScope]   runtime_like -> {kernel_A} (1 kernel(s))

; shared_helper is called from both kernel_A and kernel_B.
; SCOPE-DAG: [KernelScope]   shared_helper -> {{.*}}2 kernel(s)

; transitive_helper is reachable from kernel_A via helper_A.
; SCOPE-DAG: [KernelScope]   transitive_helper -> {kernel_A} (1 kernel(s))

; indirect_only is NOT reachable via any direct/callback edge from a kernel.
; It should NOT appear in the kernel scope map.
; SCOPE-NOT: [KernelScope]   indirect_only

; Verify the summary line (printed after per-function entries).
; SCOPE: [KernelScope] Summary: 2 kernels, 8 functions in map, {{[1-9][0-9]*}} indirect calls encountered

; STATS: {{[1-9][0-9]*}} attributor - Number of functions discovered by kernel scope BFS
; STATS: {{[1-9][0-9]*}} attributor - Number of truly indirect calls encountered during kernel scope computation
