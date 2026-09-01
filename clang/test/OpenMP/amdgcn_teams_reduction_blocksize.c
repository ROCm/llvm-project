// Cross-team (teams) reduction kernels get a larger default block size than
// plain SPMD kernels, selected by '-fopenmp-target-xteam-reduction-blocksize='
// (512 by default). A kernel written as a single combined directive and the
// same kernel split over several directives must end up with the same block
// size, so the clauses have to be looked up over the whole directive nest.

// RUN: %clang_cc1 -verify -fopenmp -x c -triple x86_64-unknown-linux-gnu -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c -triple amdgcn-amd-amdhsa -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s --check-prefixes=CHECK,DEFAULT

// RUN: %clang_cc1 -verify -fopenmp -x c -triple x86_64-unknown-linux-gnu -fopenmp-targets=amdgcn-amd-amdhsa -fopenmp-target-xteam-reduction-blocksize=64 -emit-llvm-bc %s -o %t-x86-host-bs64.bc
// RUN: %clang_cc1 -verify -fopenmp -x c -triple amdgcn-amd-amdhsa -fopenmp-targets=amdgcn-amd-amdhsa -fopenmp-target-xteam-reduction-blocksize=64 -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-x86-host-bs64.bc -o - | FileCheck %s --check-prefixes=CHECK,BS64

// expected-no-diagnostics

#ifndef HEADER
#define HEADER

void combined(int N, double *a, double *s) {
  // Combined spelling: the reduction sits on the 'target teams ...' directive.
#pragma omp target teams distribute parallel for reduction(+ : s[0])
  for (int i = 0; i < N; ++i)
    s[0] += a[i];
}

void split(int N, double *a, double *s) {
  // Split spelling: 'target' carries no clause at all, the reduction sits on
  // the nested 'teams'. Must match combined() above.
#pragma omp target
#pragma omp teams reduction(+ : s[0])
#pragma omp distribute parallel for reduction(+ : s[0])
  for (int i = 0; i < N; ++i)
    s[0] += a[i];
}

void split_thread_limit(int N, double *a, double *s) {
  // A thread_limit on the nested 'teams' overrides the reduction block size,
  // unless a block size was explicitly requested on the command line.
#pragma omp target
#pragma omp teams reduction(+ : s[0]) thread_limit(128)
#pragma omp distribute parallel for reduction(+ : s[0])
  for (int i = 0; i < N; ++i)
    s[0] += a[i];
}

void split_num_threads(int N, double *a, double *s) {
  // So does a num_threads on the innermost directive of the nest, with the same
  // exception for an explicitly requested block size.
#pragma omp target
#pragma omp teams reduction(+ : s[0])
#pragma omp distribute parallel for reduction(+ : s[0]) num_threads(32)
  for (int i = 0; i < N; ++i)
    s[0] += a[i];
}

void no_teams_reduction(int N, double *a, double *s) {
  // No reduction on the 'teams' directive: this is not a cross-team reduction,
  // so it keeps the generic SPMD block size in both spellings.
#pragma omp target teams
#pragma omp distribute parallel for reduction(+ : s[0])
  for (int i = 0; i < N; ++i)
    s[0] += a[i];
#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for reduction(+ : s[0])
  for (int i = 0; i < N; ++i)
    s[0] += a[i];
}

#endif

// The combined and the split spelling of the cross-team reduction must end up
// in the very same attribute group, which is what pins their block size to the
// same value.
// CHECK: define weak_odr protected amdgpu_kernel void @{{.*}}_combined_l{{[0-9]+}}({{.*}}) #[[RED:[0-9]+]] {
// CHECK: define weak_odr protected amdgpu_kernel void @{{.*}}_split_l{{[0-9]+}}({{.*}}) #[[RED]] {
// CHECK: define weak_odr protected amdgpu_kernel void @{{.*}}_split_thread_limit_l{{[0-9]+}}({{.*}}) #[[TL:[0-9]+]] {
// The num_threads clause gives this kernel a group of its own with the default
// block size, and it keeps one with an explicit
// '-fopenmp-target-xteam-reduction-blocksize=' too: the option then wins over
// the clause for the block size, but the clause is still reported through
// 'omp_target_thread_limit', which keeps the group distinct.
// CHECK: define weak_odr protected amdgpu_kernel void @{{.*}}_split_num_threads_l{{[0-9]+}}({{.*}}) #[[NT:[0-9]+]] {
// CHECK: define weak_odr protected amdgpu_kernel void @{{.*}}_no_teams_reduction_l{{[0-9]+}}({{.*}}) #[[NORED:[0-9]+]] {
// CHECK: define weak_odr protected amdgpu_kernel void @{{.*}}_no_teams_reduction_l{{[0-9]+}}({{.*}}) #[[NORED]] {

// Both follow '-fopenmp-target-xteam-reduction-blocksize='.
// DEFAULT: attributes #[[RED]] = { {{.*}}"amdgpu-flat-work-group-size"="1,512"
// BS64: attributes #[[RED]] = { {{.*}}"amdgpu-flat-work-group-size"="1,64"

// Clauses on the nested directives are honored, except when a block size was
// explicitly requested on the command line: that value takes precedence over
// the clauses, as it did in the removed Xteam reduction implementation. Note
// that split_thread_limit keeps a group of its own even then, because the
// thread_limit clause still shows up as 'omp_target_thread_limit'.
// DEFAULT: attributes #[[TL]] = { {{.*}}"amdgpu-flat-work-group-size"="1,128"
// BS64: attributes #[[TL]] = { {{.*}}"amdgpu-flat-work-group-size"="1,64"
// DEFAULT: attributes #[[NT]] = { {{.*}}"amdgpu-flat-work-group-size"="1,32"
// BS64: attributes #[[NT]] = { {{.*}}"amdgpu-flat-work-group-size"="1,64"

// Without a reduction on 'teams' the generic SPMD block size is kept.
// CHECK: attributes #[[NORED]] = { {{.*}}"amdgpu-flat-work-group-size"="1,256"
