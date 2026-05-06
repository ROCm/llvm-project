// RUN: %clang_cc1 -std=c++20 -fsyntax-only %s
// RUN: %clang_cc1 -std=c++20 -fcuda-is-device -fsyntax-only %s

#define __host__ __attribute__((host))
#define __device__ __attribute__((device))

template <class T1, class T2> int host_like(T1, T2);
template <class T1, class T2> __host__ __device__ long host_like(T1, T2);

void host_use(double *Val, int I) {
  static_assert(sizeof(host_like(Val[I], 1)) == sizeof(int));
}

template <class T1, class T2> double nextafter(T1, T2);

template <class T1, class T2> __device__ double nextafter(T1, T2);

__device__ void device_use(double *Val, int I) {
  (void)nextafter(Val[I], 1);
}
