// RUN: %clang_cc1 -std=c++20 -fsyntax-only %s
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify -fno-offload-implicit-host-device-templates -DNO_IMPLICIT_HDT %s

#define __global__ __attribute__((global))

template <class T> T &&declval();

template <class T>
concept HasMember = requires {
  declval<T>().member; // expected-note {{because 'declval<T>().member' would be invalid: no matching function for call to 'declval'}}
};

struct S {
  int member;
};

template <class T>
  requires HasMember<T> // expected-note {{because 'S' does not satisfy 'HasMember'}}
__global__ void kernel() {} // expected-note {{candidate template ignored: constraints not satisfied [with T = S]}}

#ifdef NO_IMPLICIT_HDT
// The expected diagnostics for the opt-out run are attached to the template
// declaration and explicit instantiation.
#endif
template __global__ void kernel<S>(); // expected-error {{explicit instantiation of 'kernel' does not refer to a function template, variable template, member function, member class, or static data member}}
