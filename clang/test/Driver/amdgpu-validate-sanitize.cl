// RUN: %clang -### --target=amdgcn-amd-amdhsa -mcpu=gfx900:xnack+ \
// RUN:   -fsanitize=address \
// RUN:   -nogpuinc --rocm-path=%S/Inputs/rocm \
// RUN:   %s 2>&1 | FileCheck %s

// RUN: %clang -### --target=amdgcn-amd-amdhsa -mcpu=gfx1250    \
// RUN:   -fsanitize=address \
// RUN:   -nogpuinc --rocm-path=%S/Inputs/rocm \
// RUN:   %s 2>&1 | FileCheck %s

// RUN: %clang -### --target=amdgcn-amd-amdhsa -mcpu=gfx900 \
// RUN:   -fsanitize=undefined \
// RUN:   -fsanitize=unsigned-integer-overflow     \
// RUN:   -fsanitize=float-divide-by-zero     \
// RUN:   -fsanitize=unsigned-integer-overflow \
// RUN:   -fsanitize=unsigned-shift-base \
// RUN:   -fsanitize=implicit-conversion \
// RUN:   -fsanitize=nullability \
// RUN:   -fsanitize=local-bounds \
// RUN:   -fsanitize=alloc-token \
// RUN:   -nogpuinc --rocm-path=%S/Inputs/rocm \
// RUN:   %s 2>&1 | FileCheck -check-prefix=GENERIC %s


// FIXME: This should error, but is silently ignored
// RUN: %clang -### --target=amdgcn-amd-amdhsa -mcpu=gfx900:xnack- \
// RUN:   -fsanitize=address \
// RUN:   -nogpuinc --rocm-path=%S/Inputs/rocm \
// RUN:   %s 2>&1 | FileCheck -check-prefix=ERR %s

// CHECK: "-triple" "amdgcn-amd-amdhsa"
// CHECK-SAME: "-mlink-bitcode-file" "{{.*}}asanrtl.bc"
// CHECK-SAME: "-fsanitize=address"

// GENERIC: "-fsanitize=alignment,array-bounds,bool,builtin,enum,float-cast-overflow,function,integer-divide-by-zero,nonnull-attribute,null,pointer-overflow,return,returns-nonnull-attribute,shift-base,shift-exponent,signed-integer-overflow,unreachable,vla-bound" "-fsanitize-recover=alignment,array-bounds,bool,builtin,enum,float-cast-overflow,function,integer-divide-by-zero,nonnull-attribute,null,pointer-overflow,returns-nonnull-attribute,shift-base,shift-exponent,signed-integer-overflow,vla-bound" "-fsanitize-merge=alignment,array-bounds,bool,builtin,enum,float-cast-overflow,function,integer-divide-by-zero,nonnull-attribute,null,pointer-overflow,return,returns-nonnull-attribute,shift-base,shift-exponent,signed-integer-overflow,unreachable,vla-bound" "-fno-sanitize-memory-param-retval" "-fno-sanitize-address-use-odr-indicator"


// FIXME: Should not be forwarding argument
// ExRR-NOT: asanrtl.bc
// ERR: "-fsanitize=address"
