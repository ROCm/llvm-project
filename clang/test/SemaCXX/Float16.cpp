// RUN: %clang_cc1 -fsyntax-only -verify -triple i686-linux-pc %s
// RUN: %clang_cc1 -fsyntax-only -verify -triple i686-linux-pc -target-feature +sse2 %s -DHAVE
// RUN: %clang_cc1 -fsyntax-only -verify -triple x86_64-linux-pc %s -DHAVE
// RUN: %clang_cc1 -fsyntax-only -verify -triple spir-unknown-unknown %s -DHAVE
// RUN: %clang_cc1 -fsyntax-only -verify -triple armv7a-linux-gnu %s -DHAVE
// RUN: %clang_cc1 -fsyntax-only -verify -triple aarch64-linux-gnu %s -DHAVE
// RUN: %clang_cc1 -fsyntax-only -verify -triple s390x-ibm-zos %s
#ifdef HAVE
// expected-no-diagnostics
#endif // HAVE

#ifndef HAVE
// expected-error@+2{{_Float16 is not supported on this target}}
#endif // !HAVE
_Float16 f;

#ifndef HAVE
// expected-error@+2{{invalid suffix 'F16' on floating constant}}
#endif // !HAVE
const auto g = 1.1F16;
