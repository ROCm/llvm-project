// COM: Compile the same HIP kernel concurrently from 8 threads, alternating
// COM: which threads request -mllvm -amdgpu-function-calls=true vs. =false
// COM: (plus -O0) on the CODEGEN_BC_TO_RELOCATABLE action. This exercises
// COM: comgr's LLVM cl::opt option-fingerprint cache under concurrency: if a
// COM: thread's -mllvm option were to leak into another thread's codegen
// COM: invocation, that thread's helper-inlining decision -- and therefore
// COM: its disassembly -- would not match what it requested.
// RUN: compile-concurrent-hip-differing-options 8 %t.thread
// RUN: for i in 0 2 4 6; do %llvm-objdump -d %t.thread$i.bin > %t.thread$i.dis; %FileCheck --check-prefix=CALLS %s < %t.thread$i.dis; done
// RUN: for i in 1 3 5 7; do %llvm-objdump -d %t.thread$i.bin > %t.thread$i.dis; %FileCheck --check-prefix=NOCALLS %s < %t.thread$i.dis; done

// CALLS-DAG: <kernel_thread{{[0-9]+}}>:
// CALLS-DAG: <{{.*}}helper_thread{{[0-9]+}}{{.*}}>:

// NOCALLS: <kernel_thread{{[0-9]+}}>:
// NOCALLS-NOT: <{{.*}}helper_thread{{[0-9]+}}{{.*}}>:
