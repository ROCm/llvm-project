// RUN: %clang_cc1 -std=c++20 -fcuda-is-device -Wall -Werror -fsyntax-only -DDEVICE_USE %s
// RUN: %clang_cc1 -std=c++20 -fcuda-is-device -Wall -Werror -fsyntax-only -fno-offload-implicit-host-device-templates %s

struct HostOnlyTy {};
struct DeviceTy {};

static bool operator==(HostOnlyTy, HostOnlyTy) { return true; }
__attribute__((device)) bool operator==(DeviceTy, DeviceTy) { return true; }

template <class T> bool compare(T lhs, T rhs) { return lhs == rhs; }

void hostUse() {
  (void)compare(HostOnlyTy{}, HostOnlyTy{});
}

#ifdef DEVICE_USE
__attribute__((device)) bool deviceUse() {
  return compare(DeviceTy{}, DeviceTy{});
}
#endif
