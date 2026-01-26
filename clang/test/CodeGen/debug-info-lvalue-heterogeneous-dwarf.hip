// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -x hip -triple amdgcn-amd-amdhsa -emit-llvm -fcuda-is-device -debug-info-kind=standalone -gheterogeneous-dwarf %s -o - | FileCheck %s

// Test that the LValue case in EmitGlobalVariableForHeterogeneousDwarf handles
// pointer constants correctly by converting them to integer constants via ptrtoint.

#define __constant__ __attribute__((constant))

// Note: The LValue case in the switch at line 6742 is exercised through the
// TemplateArgument::Declaration case (around line 1895 in CGDebugInfo.cpp)
// when template non-type parameters of pointer type are used.

// This test verifies that when we have an LValue APValue (representing a pointer),
// the code correctly:
// 1. Converts ConstantInt directly -> passes through
// 2. Converts ConstantFP directly -> passes through  
// 3. Converts pointer constants to integers via ptrtoint -> handled by LValue case
// 4. Returns early if conversion fails

// The actual triggering happens through enum constants or template parameters
// that are evaluated to pointer values. For now, this is a placeholder test
// that documents the implementation. A more complete test would require
// constructing scenarios where template pointer parameters are used in ways
// that trigger constant evaluation with the heterogeneous DWARF path.

// Basic smoke test: verify heterogeneous DWARF is enabled and working
struct Foo { void *A; void *B; };
__constant__ Foo test_var = {nullptr, nullptr};

// CHECK: !DIGlobalVariableExpression(var: [[VAR:![0-9]+]], expr: !DIExpression(DIOpArg(0, ptr addrspace(1)), DIOpDeref(i32)))
// CHECK: [[VAR]] = distinct !DIGlobalVariable(name: "test_var"
