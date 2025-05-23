// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

//
// Check help message.
//
// RUN: clang-offload-wrapper --help | FileCheck %s --check-prefix CHECK-HELP
// CHECK-HELP: {{.*}}OVERVIEW: A tool to create a wrapper bitcode for offload target binaries. Takes offload
// CHECK-HELP: {{.*}}target binaries as input and produces bitcode file containing target binaries packaged
// CHECK-HELP: {{.*}}as data and initialization code which registers target binaries in offload runtime.
// CHECK-HELP: {{.*}}USAGE: clang-offload-wrapper [options] <input files>
// CHECK-HELP: {{.*}}  --aux-triple=<triple>       - Target triple for the output module
// CHECK-HELP: {{.*}}  -o <filename>               - Output filename
// CHECK-HELP: {{.*}}  --target=<triple>           - Target triple for input files

//
// Generate a file to wrap.
//
// RUN: echo 'Content of device file' > %t.tgt

//
// Check bitcode produced by the wrapper tool.
//
// RUN: clang-offload-wrapper -add-omp-offload-notes -target=amdgcn-amd-amdhsa -aux-triple=x86_64-pc-linux-gnu -o %t.wrapper.bc %t.tgt 2>&1 | FileCheck %s --check-prefix ELF-WARNING
// RUN: llvm-dis %t.wrapper.bc -o - | FileCheck %s --check-prefix CHECK-IR

// ELF-WARNING: is not an ELF image, so notes cannot be added to it.
// CHECK-IR: target triple = "x86_64-pc-linux-gnu"

// CHECK-IR-DAG: [[ENTTY:%.+]] = type { i64, i16, i16, i32, ptr, ptr, i64, i64, ptr }
// CHECK-IR-DAG: [[IMAGETY:%.+]] = type { ptr, ptr, ptr, ptr }
// CHECK-IR-DAG: [[DESCTY:%.+]] = type { i32, ptr, ptr, ptr }
//
// CHECK-IR: [[ENTBEGIN:@.+]] = external hidden constant [0 x [[ENTTY]]]
// CHECK-IR: [[ENTEND:@.+]] = external hidden constant [0 x [[ENTTY]]]
// CHECK-IR: [[DUMMY:@.+]] = internal constant [0 x [[ENTTY]]] zeroinitializer, section "llvm_offload_entries", align 8
// CHECK-IR: @llvm.compiler.used = appending global [1 x ptr] [ptr [[DUMMY]]], section "llvm.metadata"

// CHECK-IR: [[BIN:@.+]] = internal unnamed_addr constant [[[SIZE:[0-9]+]] x i8] c"\10\FF\10\AD{{.*}}"
  ffloading.device_image = internal unnamed_addr constant [[[SIZE]] x i8] c"\10\FF\10\AD\01\00\00\0
// CHECK-IR: [[IMAGES:@.+]] = internal unnamed_addr constant [1 x %__tgt_device_image] [%__tgt_device_image { ptr getelementptr ([[[SIZE]] x i8], ptr [[BIN]], i64 0, i64 136), ptr getelementptr ([[[SIZE]] x i8], ptr [[BIN]], i64 0, i64 159), ptr [[ENTBEGIN]], ptr [[ENTEND]] }]
// CHECK-IR: [[DESC:@.+]] = internal constant [[DESCTY]] { i32 1, ptr [[IMAGES]], ptr [[ENTBEGIN]], ptr [[ENTEND]] }
// CHECK-IR: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 101, ptr [[REGFN:@.+]], ptr null }]

// CHECK-IR: define internal void [[REGFN]]() section ".text.startup" {
// CHECK-IR:   call void @__tgt_register_lib(ptr [[DESC]])
// CHECK-IR:   %0 = call i32 @atexit(ptr @.omp_offloading.descriptor_unreg)
// CHECK-IR:   ret void
// CHECK-IR: }

// CHECK-IR: declare void @__tgt_register_lib(ptr)

// CHECK-IR: declare i32 @atexit(ptr)

// CHECK-IR: define internal void [[DESC]]_unreg() section ".text.startup" {
// CHECK-IR:   call void @__tgt_unregister_lib(ptr [[DESC]])
// CHECK-IR:   ret void
// CHECK-IR: }

// CHECK_IR: declare void @__tgt_unregister_lib(ptr)

// Check that clang-offload-wrapper adds LLVMOMPOFFLOAD notes
// into the ELF offload images:
// RUN: yaml2obj %S/Inputs/empty-elf-template.yaml -o %t.64le -DBITS=64 -DENCODING=LSB
// RUN: clang-offload-wrapper -add-omp-offload-notes -target=amdgcn-amd-amdhsa -aux-triple=x86_64-pc-linux-gnu -o %t.wrapper.elf64le.bc %t.64le
// RUN: llvm-dis %t.wrapper.elf64le.bc -o - | FileCheck %s --check-prefix OMPNOTES
// RUN: yaml2obj %S/Inputs/empty-elf-template.yaml -o %t.64be -DBITS=64 -DENCODING=MSB
// RUN: clang-offload-wrapper -add-omp-offload-notes -target=amdgcn-amd-amdhsa -aux-triple=x86_64-pc-linux-gnu -o %t.wrapper.elf64be.bc %t.64be
// RUN: llvm-dis %t.wrapper.elf64be.bc -o - | FileCheck %s --check-prefix OMPNOTES
// RUN: yaml2obj %S/Inputs/empty-elf-template.yaml -o %t.32le -DBITS=32 -DENCODING=LSB
// RUN: clang-offload-wrapper -add-omp-offload-notes -target=amdgcn-amd-amdhsa -aux-triple=x86_64-pc-linux-gnu -o %t.wrapper.elf32le.bc %t.32le
// RUN: llvm-dis %t.wrapper.elf32le.bc -o - | FileCheck %s --check-prefix OMPNOTES
// RUN: yaml2obj %S/Inputs/empty-elf-template.yaml -o %t.32be -DBITS=32 -DENCODING=MSB
// RUN: clang-offload-wrapper -add-omp-offload-notes -target=amdgcn-amd-amdhsa -aux-triple=x86_64-pc-linux-gnu -o %t.wrapper.elf32be.bc %t.32be
// RUN: llvm-dis %t.wrapper.elf32be.bc -o - | FileCheck %s --check-prefix OMPNOTES

// There is no clean way for extracting the offload image
// from the object file currently, so try to find
// the inserted ELF notes in the device image variable's
// initializer:
// OMPNOTES: @{{.+}} = internal unnamed_addr constant [{{[0-9]+}} x i8] c"{{.*}}LLVMOMPOFFLOAD{{.*}}LLVMOMPOFFLOAD{{.*}}LLVMOMPOFFLOAD{{.*}}"
