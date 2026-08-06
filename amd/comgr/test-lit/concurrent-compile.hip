// COM: Compile the same HIP source concurrently from 8 threads, each given a
// COM: distinct -D macro that selects its kernel's name. This exercises
// COM: comgr's compiler-driver global state (e.g. LLVM's cl::opt registry,
// COM: clang::driver::Driver args) under concurrency: if per-thread compile
// COM: options were to leak across threads, the affected thread's kernel
// COM: would come out missing or misnamed below.
// RUN: compile-concurrent-hip 8 %t.thread
// RUN: for i in 0 1 2 3 4 5 6 7; do %llvm-objdump -d %t.thread$i.bin >> %t.dis; done
// RUN: %FileCheck %s < %t.dis
// CHECK-DAG: <add_thread0>:
// CHECK-DAG: <add_thread1>:
// CHECK-DAG: <add_thread2>:
// CHECK-DAG: <add_thread3>:
// CHECK-DAG: <add_thread4>:
// CHECK-DAG: <add_thread5>:
// CHECK-DAG: <add_thread6>:
// CHECK-DAG: <add_thread7>:
