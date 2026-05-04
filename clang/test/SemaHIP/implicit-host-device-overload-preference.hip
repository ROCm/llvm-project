// RUN: %clang_cc1 -std=c++20 -fsyntax-only %s
// RUN: %clang_cc1 -std=c++20 -fcuda-is-device -fsyntax-only %s

#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __host__ __attribute__((host))

struct HostOnlyArg {};
struct DeviceArg {};

template <class T> struct Wrapper {};

// This models a host-only library/template operator that becomes implicit H+D
// when the option is on by default.
template <class T> int select(Wrapper<T>, T) { return 1; }

// This models an explicit H+D generic overload. It remains available, but it
// should not steal overload resolution from the native-side implicit H+D
// overload above.
template <class T>
__host__ __device__ long select(T, HostOnlyArg) {
  return 2;
}

void hostUse() {
  static_assert(sizeof(select(Wrapper<HostOnlyArg>{}, HostOnlyArg{})) ==
                sizeof(int));
}

// Same rule on the device side: a formerly device-only template that is
// implicit H+D should keep native preference over an explicit H+D overload.
template <class T> __device__ int deviceSelect(Wrapper<T>, T) { return 1; }

template <class T>
__host__ __device__ long deviceSelect(T, DeviceArg) {
  return 2;
}

__global__ void deviceUse() {
  static_assert(sizeof(deviceSelect(Wrapper<DeviceArg>{}, DeviceArg{})) ==
                sizeof(int));
}
