// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgpu -fcuda-is-device -fsyntax-only -verify %s

#define __device__ __attribute__((device))

__device__ void test_ok() {
  __builtin_amdgcn_sqtt_user_entry("foo");
  __builtin_amdgcn_sqtt_user_exit("foo");
}

__device__ void test_not_literal(const char *s) {
  __builtin_amdgcn_sqtt_user_entry(s); // expected-error {{expression is not a string literal}}
  __builtin_amdgcn_sqtt_user_exit(s); // expected-error {{expression is not a string literal}}
}
