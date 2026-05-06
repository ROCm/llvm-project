// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s
// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s --check-prefix=NEG

// Regression test: a variable-length alloca inside an omp.private init region
// must NOT be lowered to an addrspace(5) (scratch segment) alloca on GPU
// targets.  The scratch segment has a fixed, limited size per workitem
// (private_segment_fixed_size), and a VLA there overflows it at runtime.
// The fix replaces the VLA with a device @malloc call.
//
// Tested construct: omp.parallel { omp.distribute { omp.wsloop } }
// with a private VLA whose size is read from a runtime array descriptor.

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.alloca_memory_space", 5 : ui32>>, llvm.data_layout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9", llvm.target_triple = "amdgcn-amd-amdhsa", omp.is_gpu = true, omp.is_target_device = true} {

  // Privatizer for a simple array descriptor {data_ptr, extent}.
  // The init region allocates private storage sized at runtime, producing a
  // VLA alloca.  Without the fix the alloca lands in addrspace(5); with the
  // fix it is replaced by a @malloc call.
  omp.private {type = private} @vla_arr.privatizer : !llvm.struct<(ptr, i64)> init {
  ^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
    %gep = llvm.getelementptr %arg0[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64)>
    %n = llvm.load %gep : !llvm.ptr -> i64
    %arr = llvm.alloca %n x f32 {bindc_name = "arr"} : (i64) -> !llvm.ptr<5>
    %arr_flat = llvm.addrspacecast %arr : !llvm.ptr<5> to !llvm.ptr
    %undef = llvm.mlir.undef : !llvm.struct<(ptr, i64)>
    %d0 = llvm.insertvalue %arr_flat, %undef[0] : !llvm.struct<(ptr, i64)>
    %d1 = llvm.insertvalue %n, %d0[1] : !llvm.struct<(ptr, i64)>
    llvm.store %d1, %arg1 : !llvm.struct<(ptr, i64)>, !llvm.ptr
    omp.yield(%arg1 : !llvm.ptr)
  }

  llvm.func @vla_private_distribute_wsloop(%box_in : !llvm.ptr) attributes {
    omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to)>,
    target_cpu = "gfx90a",
    target_features = #llvm.target_features<["+gfx9-insts", "+wavefrontsize64"]>
  } {
    %c1_i32 = llvm.mlir.constant(1 : i32) : i32
    %c1_i64 = llvm.mlir.constant(1 : i64) : i64
    %box = llvm.alloca %c1_i64 x !llvm.struct<(ptr, i64)> : (i64) -> !llvm.ptr<5>
    %box_flat = llvm.addrspacecast %box : !llvm.ptr<5> to !llvm.ptr
    %c16 = llvm.mlir.constant(16 : i32) : i32
    "llvm.intr.memcpy"(%box_flat, %box_in, %c16) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i32) -> ()
    omp.parallel private(@vla_arr.privatizer %box_flat -> %priv : !llvm.ptr) {
      omp.distribute {
        omp.wsloop {
          omp.loop_nest (%i) : i32 = (%c1_i32) to (%c1_i32) inclusive step (%c1_i32) {
            %ptr_gep = llvm.getelementptr %priv[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64)>
            %ptr = llvm.load %ptr_gep : !llvm.ptr -> !llvm.ptr
            %val = llvm.mlir.constant(1.0 : f32) : f32
            llvm.store %val, %ptr : f32, !llvm.ptr
            omp.yield
          }
        } {omp.composite}
      } {omp.composite}
      omp.terminator
    } {omp.composite}
    llvm.return
  }
}

// VLA private storage must be allocated via device malloc, not a
// variable-length alloca in the scratch segment.
// CHECK: call ptr @malloc(i64

// No VLA alloca must appear in addrspace(5) (GPU scratch segment).
// NEG-NOT: alloca float, i64 {{.*}}, addrspace(5)
