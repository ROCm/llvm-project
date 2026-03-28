// RUN: %clangxx_memprof  %s -o %t

// stderr log_path
// RUN: %env_memprof_opts=print_text=true:log_path=stderr %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-GOOD --dump-input=always

// Good log_path.
// RUN: rm -f %t.log.*
// RUN: %env_memprof_opts=print_text=true:log_path=%t.log %run %t
// RUN: FileCheck %s --check-prefix=CHECK-GOOD --dump-input=always < %t.log.*

// Invalid log_path: the runtime opens "<prefix>.<pid>". Historically tests used a
// prefix like /INVALID, which becomes /INVALID.<pid> under /. That fails for a
// normal user (no write permission on /) but succeeds for root, which is common in
// CI and default Docker images—so the sanitizer would open a real file, emit no
// ERROR, and `not %run` would fail. Use /proc/self/mem instead: the resulting path
// is not openable as a writable log file even for root, so we still get the
// expected diagnostic and exit status.
// RUN: %env_memprof_opts=print_text=true:log_path=/proc/self/mem not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-INVALID --dump-input=always

// Directory of log_path can't be created.
// RUN: %env_memprof_opts=print_text=true:log_path=/dev/null/INVALID not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-BAD-DIR --dump-input=always

// Too long log_path.
// RUN: %python -c "for i in range(0, 10000): print(i, end='')" > %t.long_log_path
// RUN: %env_memprof_opts=print_text=true:log_path=%{readfile:%t.long_log_path} \
// RUN:   not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-LONG --dump-input=always

// Specifying the log name via the __memprof_profile_filename variable (same
// unopenable prefix as the log_path= case above). Use -DPROFILE_NAME_VAR=/path
// without extra shell quotes so the preprocessor yields a normal C string; forms
// like -DPROFILE_NAME_VAR=\"/path\" stringify incorrectly with xstr()/str().
// RUN: %clangxx_memprof  %s -o %t -DPROFILE_NAME_VAR=/proc/self/mem
// RUN: not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-INVALID --dump-input=always

#include <sanitizer/memprof_interface.h>

#ifdef PROFILE_NAME_VAR
#define xstr(s) str(s)
#define str(s) #s
char __memprof_profile_filename[] = xstr(PROFILE_NAME_VAR);
#endif

#include <stdlib.h>
#include <string.h>
int main(int argc, char **argv) {
  char *x = (char *)malloc(10);
  memset(x, 0, 10);
  free(x);
  __memprof_profile_dump();
  return 0;
}
// CHECK-GOOD: Memory allocation stack id
// The next line matches /proc/self/mem.<pid> stderr from the invalid log_path RUNs above.
// CHECK-INVALID: ERROR: Can't open file: /proc/self/mem.
// CHECK-BAD-DIR: ERROR: Can't create directory: /dev/null
// CHECK-LONG: ERROR: Path is too long: 01234
