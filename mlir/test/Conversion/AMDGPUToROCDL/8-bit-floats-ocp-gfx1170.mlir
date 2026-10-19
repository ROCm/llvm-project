// RUN: mlir-opt %s --convert-amdgpu-to-rocdl=chipset=gfx1170 --split-input-file | FileCheck %s

func.func @ext_packed_fp8(%v: vector<4xf8E4M3FN>) -> f32 {
  %ret = amdgpu.ext_packed_fp8 %v[0] : vector<4xf8E4M3FN> to f32
  func.return %ret : f32
}

// CHECK: rocdl.cvt.f32.fp8

// -----

func.func @ext_packed_bf8(%v: vector<4xf8E5M2>) -> f32 {
  %ret = amdgpu.ext_packed_fp8 %v[0] : vector<4xf8E5M2> to f32
  func.return %ret : f32
}

// CHECK: rocdl.cvt.f32.bf8

// -----

func.func @packed_trunc_2xfp8(%v: f32) -> vector<4xf8E4M3FN> {
  %ret = amdgpu.packed_trunc_2xfp8 %v, undef into undef[word 0] : f32 to vector<4xf8E4M3FN>
  func.return %ret : vector<4xf8E4M3FN>
}

// CHECK: rocdl.cvt.pk.fp8.f32
