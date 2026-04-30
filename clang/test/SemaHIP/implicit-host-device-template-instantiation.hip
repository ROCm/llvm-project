// RUN: %clang_cc1 -std=c++20 -fsyntax-only -fcuda-is-device %s
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -fcuda-is-device -verify -DDEVICE_USE %s

#define __global__ __attribute__((global))

static int directHostOnly() { return 42; } // expected-note {{'directHostOnly' declared here}}

template <class T> T directHostReference() {
  return static_cast<T>(directHostOnly()); // expected-error {{reference to __host__ function 'directHostOnly' in __host__ __device__ function}}
}

// A host-side explicit instantiation should not force a device instantiation
// for an implicit host/device template.
template int directHostReference<int>();

static int transitiveHostOnly() { return 42; } // expected-note {{'transitiveHostOnly' declared here}}

template <class T> T transitiveLeaf() {
  return static_cast<T>(transitiveHostOnly()); // expected-error {{reference to __host__ function 'transitiveHostOnly' in __host__ __device__ function}}
}

template <class T> T transitiveRoot() {
  return transitiveLeaf<T>(); // expected-note {{called by 'transitiveRoot<int>'}}
}

// Host-side explicit instantiations in a chain should not force device
// instantiations either.
template int transitiveLeaf<int>();
template int transitiveRoot<int>();

#ifdef DEVICE_USE
__global__ void directKernel(int *out) {
  *out = directHostReference<int>(); // expected-note {{called by 'directKernel'}}
}

__global__ void transitiveKernel(int *out) {
  *out = transitiveRoot<int>(); // expected-note {{called by 'transitiveKernel'}}
}
#endif
