// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// Regression test: the alloca for a private array descriptor (box struct) must
// be replicated per-invocation of the loop body callback rather than shared
// across all worker threads via struct_arg.
//
// When lowering omp.parallel { omp.distribute { omp.wsloop } } on GPU,
// applyWorkshareLoop outlines the loop body into a callback and captures the
// flat pointer of any alloca in allocaAddrSpace through struct_arg.  All
// threads then share the same box, which can lead to races when the box is
// modified inside the loop body.
//
// The fix inserts a per-invocation alloca + memcpy at the top of the loop body
// callback, so each invocation gets its own local copy of the box.

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.alloca_memory_space", 5 : ui32>>, llvm.data_layout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9", llvm.target_triple = "amdgcn-amd-amdhsa", omp.is_gpu = true, omp.is_target_device = true} {

  // Privatizer for a fixed-size array descriptor {data_ptr, extent}.
  // The init region copies the source descriptor into the private slot.
  omp.private {type = private} @box.privatizer : !llvm.struct<(ptr, i64)> init {
  ^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
    %c16 = llvm.mlir.constant(16 : i32) : i32
    "llvm.intr.memcpy"(%arg1, %arg0, %c16) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i32) -> ()
    omp.yield(%arg1 : !llvm.ptr)
  }

  llvm.func @box_private_distribute_wsloop(%box_in : !llvm.ptr) attributes {
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
    omp.parallel private(@box.privatizer %box_flat -> %priv : !llvm.ptr) {
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

// The loop body callback (takes i32 iteration index as first arg) must contain
// a per-invocation alloca and memcpy for the private box descriptor, so each
// callback invocation operates on its own local copy rather than the shared one
// captured in struct_arg.

// CHECK-LABEL: define internal void @{{.*}}omp_par(i32
// CHECK: omp.private.alloc
// CHECK: omp.private.alloc.ascast
// CHECK: call void @llvm.memcpy.p0.p0.i64
