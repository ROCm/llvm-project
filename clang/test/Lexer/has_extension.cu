// RUN: %clang_cc1 -E -triple x86_64-linux-gnu %s -o - \
// RUN:   | FileCheck -check-prefix=NOHDT %s
// RUN: %clang_cc1 -E -triple x86_64-linux-gnu %s -o - \
// RUN:   -foffload-implicit-host-device-templates \
// RUN:   | FileCheck -check-prefix=HDT %s
// RUN: %clang_cc1 -E -triple x86_64-linux-gnu -x hip %s -o - \
// RUN:   | FileCheck -check-prefix=HDT %s
// RUN: %clang_cc1 -E -triple x86_64-linux-gnu -x hip %s -o - \
// RUN:   -fno-offload-implicit-host-device-templates \
// RUN:   | FileCheck -check-prefix=NOHDT %s

// NOHDT: no_implicit_host_device_templates
// HDT: has_implicit_host_device_templates
#if __has_extension(cuda_implicit_host_device_templates)
int has_implicit_host_device_templates();
#else
int no_implicit_host_device_templates();
#endif
