// NOTE: Assertions have been autogenerated by utils/update_cc_test_checks.py
// REQUIRES: webassembly-registered-target
// RUN: %clang_cc1 -triple wasm32-unknown-unknown -emit-llvm -o - %s | FileCheck %s

// Multiple targets use emitVoidPtrVAArg to lower va_arg instructions in clang
// PPC is complicated, excluding from this case analysis
// ForceRightAdjust is false for all non-PPC targets
// AllowHigherAlign is only false for two Microsoft targets, both of which
// pass most things by reference.
//
// Address emitVoidPtrVAArg(CodeGenFunction &CGF, Address VAListAddr,
//                          QualType ValueTy, bool IsIndirect,
//                          TypeInfoChars ValueInfo, CharUnits SlotSizeAndAlign,
//                          bool AllowHigherAlign, bool ForceRightAdjust =
//                          false);
//
// Target       IsIndirect    SlotSize  AllowHigher ForceRightAdjust
// ARC          false             four  true        false
// ARM          varies            four  true        false
// Mips         false           4 or 8  true        false
// RISCV        varies        register  true        false
// PPC elided
// LoongArch    varies        register  true        false
// NVPTX WIP
// AMDGPU WIP
// X86_32       false             four  true        false
// X86_64 MS    varies           eight  false       false
// CSKY         false             four  true        false
// Webassembly  varies            four  true        false
// AArch64      false            eight  true        false
// AArch64 MS   false            eight  false       false
//
// Webassembly passes indirectly iff it's an aggregate of multiple values
// Choosing this as a representative architecture to check IR generation
// partly because it has a relatively simple variadic calling convention.

// Int, by itself and packed in structs
// CHECK-LABEL: @raw_int(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[LIST_ADDR:%.*]] = alloca ptr, align 4
// CHECK-NEXT:    store ptr [[LIST:%.*]], ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    [[ARGP_CUR:%.*]] = load ptr, ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    [[ARGP_NEXT:%.*]] = getelementptr inbounds i8, ptr [[ARGP_CUR]], i32 4
// CHECK-NEXT:    store ptr [[ARGP_NEXT]], ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    [[TMP0:%.*]] = load i32, ptr [[ARGP_CUR]], align 4
// CHECK-NEXT:    ret i32 [[TMP0]]
//
int raw_int(__builtin_va_list list) { return __builtin_va_arg(list, int); }

typedef struct {
  int x;
} one_int_t;

// CHECK-LABEL: @one_int(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[RETVAL:%.*]] = alloca [[STRUCT_ONE_INT_T:%.*]], align 4
// CHECK-NEXT:    [[LIST_ADDR:%.*]] = alloca ptr, align 4
// CHECK-NEXT:    store ptr [[LIST:%.*]], ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    [[ARGP_CUR:%.*]] = load ptr, ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    [[ARGP_NEXT:%.*]] = getelementptr inbounds i8, ptr [[ARGP_CUR]], i32 4
// CHECK-NEXT:    store ptr [[ARGP_NEXT]], ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[RETVAL]], ptr align 4 [[ARGP_CUR]], i32 4, i1 false)
// CHECK-NEXT:    [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw [[STRUCT_ONE_INT_T]], ptr [[RETVAL]], i32 0, i32 0
// CHECK-NEXT:    [[TMP0:%.*]] = load i32, ptr [[COERCE_DIVE]], align 4
// CHECK-NEXT:    ret i32 [[TMP0]]
//
one_int_t one_int(__builtin_va_list list) {
  return __builtin_va_arg(list, one_int_t);
}

typedef struct {
  int x;
  int y;
} two_int_t;

// CHECK-LABEL: @two_int(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[LIST_ADDR:%.*]] = alloca ptr, align 4
// CHECK-NEXT:    store ptr [[LIST:%.*]], ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    [[ARGP_CUR:%.*]] = load ptr, ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    [[ARGP_NEXT:%.*]] = getelementptr inbounds i8, ptr [[ARGP_CUR]], i32 4
// CHECK-NEXT:    store ptr [[ARGP_NEXT]], ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[ARGP_CUR]], align 4
// CHECK-NEXT:    call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[AGG_RESULT:%.*]], ptr align 4 [[TMP0]], i32 8, i1 false)
// CHECK-NEXT:    ret void
//
two_int_t two_int(__builtin_va_list list) {
  return __builtin_va_arg(list, two_int_t);
}

// Double, by itself and packed in structs
// CHECK-LABEL: @raw_double(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[LIST_ADDR:%.*]] = alloca ptr, align 4
// CHECK-NEXT:    store ptr [[LIST:%.*]], ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    [[ARGP_CUR:%.*]] = load ptr, ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    [[TMP0:%.*]] = getelementptr inbounds i8, ptr [[ARGP_CUR]], i32 7
// CHECK-NEXT:    [[ARGP_CUR_ALIGNED:%.*]] = call ptr @llvm.ptrmask.p0.i32(ptr [[TMP0]], i32 -8)
// CHECK-NEXT:    [[ARGP_NEXT:%.*]] = getelementptr inbounds i8, ptr [[ARGP_CUR_ALIGNED]], i32 8
// CHECK-NEXT:    store ptr [[ARGP_NEXT]], ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    [[TMP1:%.*]] = load double, ptr [[ARGP_CUR_ALIGNED]], align 8
// CHECK-NEXT:    ret double [[TMP1]]
//
double raw_double(__builtin_va_list list) {
  return __builtin_va_arg(list, double);
}

typedef struct {
  double x;
} one_double_t;

// CHECK-LABEL: @one_double(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[RETVAL:%.*]] = alloca [[STRUCT_ONE_DOUBLE_T:%.*]], align 8
// CHECK-NEXT:    [[LIST_ADDR:%.*]] = alloca ptr, align 4
// CHECK-NEXT:    store ptr [[LIST:%.*]], ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    [[ARGP_CUR:%.*]] = load ptr, ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    [[TMP0:%.*]] = getelementptr inbounds i8, ptr [[ARGP_CUR]], i32 7
// CHECK-NEXT:    [[ARGP_CUR_ALIGNED:%.*]] = call ptr @llvm.ptrmask.p0.i32(ptr [[TMP0]], i32 -8)
// CHECK-NEXT:    [[ARGP_NEXT:%.*]] = getelementptr inbounds i8, ptr [[ARGP_CUR_ALIGNED]], i32 8
// CHECK-NEXT:    store ptr [[ARGP_NEXT]], ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    call void @llvm.memcpy.p0.p0.i32(ptr align 8 [[RETVAL]], ptr align 8 [[ARGP_CUR_ALIGNED]], i32 8, i1 false)
// CHECK-NEXT:    [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw [[STRUCT_ONE_DOUBLE_T]], ptr [[RETVAL]], i32 0, i32 0
// CHECK-NEXT:    [[TMP1:%.*]] = load double, ptr [[COERCE_DIVE]], align 8
// CHECK-NEXT:    ret double [[TMP1]]
//
one_double_t one_double(__builtin_va_list list) {
  return __builtin_va_arg(list, one_double_t);
}

typedef struct {
  double x;
  double y;
} two_double_t;

// CHECK-LABEL: @two_double(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[LIST_ADDR:%.*]] = alloca ptr, align 4
// CHECK-NEXT:    store ptr [[LIST:%.*]], ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    [[ARGP_CUR:%.*]] = load ptr, ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    [[ARGP_NEXT:%.*]] = getelementptr inbounds i8, ptr [[ARGP_CUR]], i32 4
// CHECK-NEXT:    store ptr [[ARGP_NEXT]], ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[ARGP_CUR]], align 4
// CHECK-NEXT:    call void @llvm.memcpy.p0.p0.i32(ptr align 8 [[AGG_RESULT:%.*]], ptr align 8 [[TMP0]], i32 16, i1 false)
// CHECK-NEXT:    ret void
//
two_double_t two_double(__builtin_va_list list) {
  return __builtin_va_arg(list, two_double_t);
}

// Scalar smaller than the slot size (C would promote a short to int)
typedef struct {
  char x;
} one_char_t;

// CHECK-LABEL: @one_char(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[RETVAL:%.*]] = alloca [[STRUCT_ONE_CHAR_T:%.*]], align 1
// CHECK-NEXT:    [[LIST_ADDR:%.*]] = alloca ptr, align 4
// CHECK-NEXT:    store ptr [[LIST:%.*]], ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    [[ARGP_CUR:%.*]] = load ptr, ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    [[ARGP_NEXT:%.*]] = getelementptr inbounds i8, ptr [[ARGP_CUR]], i32 4
// CHECK-NEXT:    store ptr [[ARGP_NEXT]], ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    call void @llvm.memcpy.p0.p0.i32(ptr align 1 [[RETVAL]], ptr align 4 [[ARGP_CUR]], i32 1, i1 false)
// CHECK-NEXT:    [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw [[STRUCT_ONE_CHAR_T]], ptr [[RETVAL]], i32 0, i32 0
// CHECK-NEXT:    [[TMP0:%.*]] = load i8, ptr [[COERCE_DIVE]], align 1
// CHECK-NEXT:    ret i8 [[TMP0]]
//
one_char_t one_char(__builtin_va_list list) {
  return __builtin_va_arg(list, one_char_t);
}

typedef struct {
  short x;
} one_short_t;

// CHECK-LABEL: @one_short(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[RETVAL:%.*]] = alloca [[STRUCT_ONE_SHORT_T:%.*]], align 2
// CHECK-NEXT:    [[LIST_ADDR:%.*]] = alloca ptr, align 4
// CHECK-NEXT:    store ptr [[LIST:%.*]], ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    [[ARGP_CUR:%.*]] = load ptr, ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    [[ARGP_NEXT:%.*]] = getelementptr inbounds i8, ptr [[ARGP_CUR]], i32 4
// CHECK-NEXT:    store ptr [[ARGP_NEXT]], ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    call void @llvm.memcpy.p0.p0.i32(ptr align 2 [[RETVAL]], ptr align 4 [[ARGP_CUR]], i32 2, i1 false)
// CHECK-NEXT:    [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw [[STRUCT_ONE_SHORT_T]], ptr [[RETVAL]], i32 0, i32 0
// CHECK-NEXT:    [[TMP0:%.*]] = load i16, ptr [[COERCE_DIVE]], align 2
// CHECK-NEXT:    ret i16 [[TMP0]]
//
one_short_t one_short(__builtin_va_list list) {
  return __builtin_va_arg(list, one_short_t);
}

// Composite smaller than the slot size
typedef struct {
  _Alignas(2) char x;
  char y;
} char_pair_t;

// CHECK-LABEL: @char_pair(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[LIST_ADDR:%.*]] = alloca ptr, align 4
// CHECK-NEXT:    store ptr [[LIST:%.*]], ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    [[ARGP_CUR:%.*]] = load ptr, ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    [[ARGP_NEXT:%.*]] = getelementptr inbounds i8, ptr [[ARGP_CUR]], i32 4
// CHECK-NEXT:    store ptr [[ARGP_NEXT]], ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[ARGP_CUR]], align 4
// CHECK-NEXT:    call void @llvm.memcpy.p0.p0.i32(ptr align 2 [[AGG_RESULT:%.*]], ptr align 2 [[TMP0]], i32 2, i1 false)
// CHECK-NEXT:    ret void
//
char_pair_t char_pair(__builtin_va_list list) {
  return __builtin_va_arg(list, char_pair_t);
}

// Empty struct
typedef struct {
} empty_t;

// CHECK-LABEL: @empty(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[RETVAL:%.*]] = alloca [[STRUCT_EMPTY_T:%.*]], align 1
// CHECK-NEXT:    [[LIST_ADDR:%.*]] = alloca ptr, align 4
// CHECK-NEXT:    store ptr [[LIST:%.*]], ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    [[ARGP_CUR:%.*]] = load ptr, ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    [[ARGP_NEXT:%.*]] = getelementptr inbounds i8, ptr [[ARGP_CUR]], i32 0
// CHECK-NEXT:    store ptr [[ARGP_NEXT]], ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    call void @llvm.memcpy.p0.p0.i32(ptr align 1 [[RETVAL]], ptr align 4 [[ARGP_CUR]], i32 0, i1 false)
// CHECK-NEXT:    ret void
//
empty_t empty(__builtin_va_list list) {
  return __builtin_va_arg(list, empty_t);
}

typedef struct {
  empty_t x;
  int y;
} empty_int_t;

// CHECK-LABEL: @empty_int(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[RETVAL:%.*]] = alloca [[STRUCT_EMPTY_INT_T:%.*]], align 4
// CHECK-NEXT:    [[LIST_ADDR:%.*]] = alloca ptr, align 4
// CHECK-NEXT:    store ptr [[LIST:%.*]], ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    [[ARGP_CUR:%.*]] = load ptr, ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    [[ARGP_NEXT:%.*]] = getelementptr inbounds i8, ptr [[ARGP_CUR]], i32 4
// CHECK-NEXT:    store ptr [[ARGP_NEXT]], ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[RETVAL]], ptr align 4 [[ARGP_CUR]], i32 4, i1 false)
// CHECK-NEXT:    [[TMP0:%.*]] = load i32, ptr [[RETVAL]], align 4
// CHECK-NEXT:    ret i32 [[TMP0]]
//
empty_int_t empty_int(__builtin_va_list list) {
  return __builtin_va_arg(list, empty_int_t);
}

typedef struct {
  int x;
  empty_t y;
} int_empty_t;

// CHECK-LABEL: @int_empty(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[RETVAL:%.*]] = alloca [[STRUCT_INT_EMPTY_T:%.*]], align 4
// CHECK-NEXT:    [[LIST_ADDR:%.*]] = alloca ptr, align 4
// CHECK-NEXT:    store ptr [[LIST:%.*]], ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    [[ARGP_CUR:%.*]] = load ptr, ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    [[ARGP_NEXT:%.*]] = getelementptr inbounds i8, ptr [[ARGP_CUR]], i32 4
// CHECK-NEXT:    store ptr [[ARGP_NEXT]], ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[RETVAL]], ptr align 4 [[ARGP_CUR]], i32 4, i1 false)
// CHECK-NEXT:    [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw [[STRUCT_INT_EMPTY_T]], ptr [[RETVAL]], i32 0, i32 0
// CHECK-NEXT:    [[TMP0:%.*]] = load i32, ptr [[COERCE_DIVE]], align 4
// CHECK-NEXT:    ret i32 [[TMP0]]
//
int_empty_t int_empty(__builtin_va_list list) {
  return __builtin_va_arg(list, int_empty_t);
}

// Need multiple va_arg instructions to check the postincrement
// Using types that are passed directly as the indirect handling
// is independent of the alignment handling in emitVoidPtrDirectVAArg.

// CHECK-LABEL: @multiple_int(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[LIST_ADDR:%.*]] = alloca ptr, align 4
// CHECK-NEXT:    [[OUT0_ADDR:%.*]] = alloca ptr, align 4
// CHECK-NEXT:    [[OUT1_ADDR:%.*]] = alloca ptr, align 4
// CHECK-NEXT:    [[OUT2_ADDR:%.*]] = alloca ptr, align 4
// CHECK-NEXT:    store ptr [[LIST:%.*]], ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    store ptr [[OUT0:%.*]], ptr [[OUT0_ADDR]], align 4
// CHECK-NEXT:    store ptr [[OUT1:%.*]], ptr [[OUT1_ADDR]], align 4
// CHECK-NEXT:    store ptr [[OUT2:%.*]], ptr [[OUT2_ADDR]], align 4
// CHECK-NEXT:    [[ARGP_CUR:%.*]] = load ptr, ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    [[ARGP_NEXT:%.*]] = getelementptr inbounds i8, ptr [[ARGP_CUR]], i32 4
// CHECK-NEXT:    store ptr [[ARGP_NEXT]], ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    [[TMP0:%.*]] = load i32, ptr [[ARGP_CUR]], align 4
// CHECK-NEXT:    [[TMP1:%.*]] = load ptr, ptr [[OUT0_ADDR]], align 4
// CHECK-NEXT:    store i32 [[TMP0]], ptr [[TMP1]], align 4
// CHECK-NEXT:    [[ARGP_CUR1:%.*]] = load ptr, ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    [[ARGP_NEXT2:%.*]] = getelementptr inbounds i8, ptr [[ARGP_CUR1]], i32 4
// CHECK-NEXT:    store ptr [[ARGP_NEXT2]], ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    [[TMP2:%.*]] = load i32, ptr [[ARGP_CUR1]], align 4
// CHECK-NEXT:    [[TMP3:%.*]] = load ptr, ptr [[OUT1_ADDR]], align 4
// CHECK-NEXT:    store i32 [[TMP2]], ptr [[TMP3]], align 4
// CHECK-NEXT:    [[ARGP_CUR3:%.*]] = load ptr, ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    [[ARGP_NEXT4:%.*]] = getelementptr inbounds i8, ptr [[ARGP_CUR3]], i32 4
// CHECK-NEXT:    store ptr [[ARGP_NEXT4]], ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    [[TMP4:%.*]] = load i32, ptr [[ARGP_CUR3]], align 4
// CHECK-NEXT:    [[TMP5:%.*]] = load ptr, ptr [[OUT2_ADDR]], align 4
// CHECK-NEXT:    store i32 [[TMP4]], ptr [[TMP5]], align 4
// CHECK-NEXT:    ret void
//
void multiple_int(__builtin_va_list list, int *out0, int *out1, int *out2) {
  *out0 = __builtin_va_arg(list, int);
  *out1 = __builtin_va_arg(list, int);
  *out2 = __builtin_va_arg(list, int);
}

// Scalars in structs are an easy way of specifying alignment from C
// CHECK-LABEL: @increasing_alignment(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[LIST_ADDR:%.*]] = alloca ptr, align 4
// CHECK-NEXT:    [[OUT0_ADDR:%.*]] = alloca ptr, align 4
// CHECK-NEXT:    [[OUT1_ADDR:%.*]] = alloca ptr, align 4
// CHECK-NEXT:    [[OUT2_ADDR:%.*]] = alloca ptr, align 4
// CHECK-NEXT:    [[OUT3_ADDR:%.*]] = alloca ptr, align 4
// CHECK-NEXT:    store ptr [[LIST:%.*]], ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    store ptr [[OUT0:%.*]], ptr [[OUT0_ADDR]], align 4
// CHECK-NEXT:    store ptr [[OUT1:%.*]], ptr [[OUT1_ADDR]], align 4
// CHECK-NEXT:    store ptr [[OUT2:%.*]], ptr [[OUT2_ADDR]], align 4
// CHECK-NEXT:    store ptr [[OUT3:%.*]], ptr [[OUT3_ADDR]], align 4
// CHECK-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[OUT0_ADDR]], align 4
// CHECK-NEXT:    [[ARGP_CUR:%.*]] = load ptr, ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    [[ARGP_NEXT:%.*]] = getelementptr inbounds i8, ptr [[ARGP_CUR]], i32 4
// CHECK-NEXT:    store ptr [[ARGP_NEXT]], ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    call void @llvm.memcpy.p0.p0.i32(ptr align 1 [[TMP0]], ptr align 4 [[ARGP_CUR]], i32 1, i1 false)
// CHECK-NEXT:    [[TMP1:%.*]] = load ptr, ptr [[OUT1_ADDR]], align 4
// CHECK-NEXT:    [[ARGP_CUR1:%.*]] = load ptr, ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    [[ARGP_NEXT2:%.*]] = getelementptr inbounds i8, ptr [[ARGP_CUR1]], i32 4
// CHECK-NEXT:    store ptr [[ARGP_NEXT2]], ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    call void @llvm.memcpy.p0.p0.i32(ptr align 2 [[TMP1]], ptr align 4 [[ARGP_CUR1]], i32 2, i1 false)
// CHECK-NEXT:    [[ARGP_CUR3:%.*]] = load ptr, ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    [[ARGP_NEXT4:%.*]] = getelementptr inbounds i8, ptr [[ARGP_CUR3]], i32 4
// CHECK-NEXT:    store ptr [[ARGP_NEXT4]], ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    [[TMP2:%.*]] = load i32, ptr [[ARGP_CUR3]], align 4
// CHECK-NEXT:    [[TMP3:%.*]] = load ptr, ptr [[OUT2_ADDR]], align 4
// CHECK-NEXT:    store i32 [[TMP2]], ptr [[TMP3]], align 4
// CHECK-NEXT:    [[ARGP_CUR5:%.*]] = load ptr, ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    [[TMP4:%.*]] = getelementptr inbounds i8, ptr [[ARGP_CUR5]], i32 7
// CHECK-NEXT:    [[ARGP_CUR5_ALIGNED:%.*]] = call ptr @llvm.ptrmask.p0.i32(ptr [[TMP4]], i32 -8)
// CHECK-NEXT:    [[ARGP_NEXT6:%.*]] = getelementptr inbounds i8, ptr [[ARGP_CUR5_ALIGNED]], i32 8
// CHECK-NEXT:    store ptr [[ARGP_NEXT6]], ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    [[TMP5:%.*]] = load double, ptr [[ARGP_CUR5_ALIGNED]], align 8
// CHECK-NEXT:    [[TMP6:%.*]] = load ptr, ptr [[OUT3_ADDR]], align 4
// CHECK-NEXT:    store double [[TMP5]], ptr [[TMP6]], align 8
// CHECK-NEXT:    ret void
//
void increasing_alignment(__builtin_va_list list, one_char_t *out0,
                          one_short_t *out1, int *out2, double *out3) {
  *out0 = __builtin_va_arg(list, one_char_t);
  *out1 = __builtin_va_arg(list, one_short_t);
  *out2 = __builtin_va_arg(list, int);
  *out3 = __builtin_va_arg(list, double);
}

// CHECK-LABEL: @decreasing_alignment(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[LIST_ADDR:%.*]] = alloca ptr, align 4
// CHECK-NEXT:    [[OUT0_ADDR:%.*]] = alloca ptr, align 4
// CHECK-NEXT:    [[OUT1_ADDR:%.*]] = alloca ptr, align 4
// CHECK-NEXT:    [[OUT2_ADDR:%.*]] = alloca ptr, align 4
// CHECK-NEXT:    [[OUT3_ADDR:%.*]] = alloca ptr, align 4
// CHECK-NEXT:    store ptr [[LIST:%.*]], ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    store ptr [[OUT0:%.*]], ptr [[OUT0_ADDR]], align 4
// CHECK-NEXT:    store ptr [[OUT1:%.*]], ptr [[OUT1_ADDR]], align 4
// CHECK-NEXT:    store ptr [[OUT2:%.*]], ptr [[OUT2_ADDR]], align 4
// CHECK-NEXT:    store ptr [[OUT3:%.*]], ptr [[OUT3_ADDR]], align 4
// CHECK-NEXT:    [[ARGP_CUR:%.*]] = load ptr, ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    [[TMP0:%.*]] = getelementptr inbounds i8, ptr [[ARGP_CUR]], i32 7
// CHECK-NEXT:    [[ARGP_CUR_ALIGNED:%.*]] = call ptr @llvm.ptrmask.p0.i32(ptr [[TMP0]], i32 -8)
// CHECK-NEXT:    [[ARGP_NEXT:%.*]] = getelementptr inbounds i8, ptr [[ARGP_CUR_ALIGNED]], i32 8
// CHECK-NEXT:    store ptr [[ARGP_NEXT]], ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    [[TMP1:%.*]] = load double, ptr [[ARGP_CUR_ALIGNED]], align 8
// CHECK-NEXT:    [[TMP2:%.*]] = load ptr, ptr [[OUT0_ADDR]], align 4
// CHECK-NEXT:    store double [[TMP1]], ptr [[TMP2]], align 8
// CHECK-NEXT:    [[ARGP_CUR1:%.*]] = load ptr, ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    [[ARGP_NEXT2:%.*]] = getelementptr inbounds i8, ptr [[ARGP_CUR1]], i32 4
// CHECK-NEXT:    store ptr [[ARGP_NEXT2]], ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    [[TMP3:%.*]] = load i32, ptr [[ARGP_CUR1]], align 4
// CHECK-NEXT:    [[TMP4:%.*]] = load ptr, ptr [[OUT1_ADDR]], align 4
// CHECK-NEXT:    store i32 [[TMP3]], ptr [[TMP4]], align 4
// CHECK-NEXT:    [[TMP5:%.*]] = load ptr, ptr [[OUT2_ADDR]], align 4
// CHECK-NEXT:    [[ARGP_CUR3:%.*]] = load ptr, ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    [[ARGP_NEXT4:%.*]] = getelementptr inbounds i8, ptr [[ARGP_CUR3]], i32 4
// CHECK-NEXT:    store ptr [[ARGP_NEXT4]], ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    call void @llvm.memcpy.p0.p0.i32(ptr align 2 [[TMP5]], ptr align 4 [[ARGP_CUR3]], i32 2, i1 false)
// CHECK-NEXT:    [[TMP6:%.*]] = load ptr, ptr [[OUT3_ADDR]], align 4
// CHECK-NEXT:    [[ARGP_CUR5:%.*]] = load ptr, ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    [[ARGP_NEXT6:%.*]] = getelementptr inbounds i8, ptr [[ARGP_CUR5]], i32 4
// CHECK-NEXT:    store ptr [[ARGP_NEXT6]], ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    call void @llvm.memcpy.p0.p0.i32(ptr align 1 [[TMP6]], ptr align 4 [[ARGP_CUR5]], i32 1, i1 false)
// CHECK-NEXT:    ret void
//
void decreasing_alignment(__builtin_va_list list, double *out0, int *out1,
                          one_short_t *out2, one_char_t *out3) {
  *out0 = __builtin_va_arg(list, double);
  *out1 = __builtin_va_arg(list, int);
  *out2 = __builtin_va_arg(list, one_short_t);
  *out3 = __builtin_va_arg(list, one_char_t);
}

// Typical edge cases, none hit special handling in VAArg lowering.
typedef struct {
  int x[16];
  double y[8];
} large_value_t;

// CHECK-LABEL: @large_value(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[LIST_ADDR:%.*]] = alloca ptr, align 4
// CHECK-NEXT:    [[OUT_ADDR:%.*]] = alloca ptr, align 4
// CHECK-NEXT:    store ptr [[LIST:%.*]], ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    store ptr [[OUT:%.*]], ptr [[OUT_ADDR]], align 4
// CHECK-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[OUT_ADDR]], align 4
// CHECK-NEXT:    [[ARGP_CUR:%.*]] = load ptr, ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    [[ARGP_NEXT:%.*]] = getelementptr inbounds i8, ptr [[ARGP_CUR]], i32 4
// CHECK-NEXT:    store ptr [[ARGP_NEXT]], ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    [[TMP1:%.*]] = load ptr, ptr [[ARGP_CUR]], align 4
// CHECK-NEXT:    call void @llvm.memcpy.p0.p0.i32(ptr align 8 [[TMP0]], ptr align 8 [[TMP1]], i32 128, i1 false)
// CHECK-NEXT:    ret void
//
void large_value(__builtin_va_list list, large_value_t *out) {
  *out = __builtin_va_arg(list, large_value_t);
}

typedef int v128_t __attribute__((__vector_size__(16), __aligned__(16)));
// CHECK-LABEL: @vector(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[LIST_ADDR:%.*]] = alloca ptr, align 4
// CHECK-NEXT:    [[OUT_ADDR:%.*]] = alloca ptr, align 4
// CHECK-NEXT:    store ptr [[LIST:%.*]], ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    store ptr [[OUT:%.*]], ptr [[OUT_ADDR]], align 4
// CHECK-NEXT:    [[ARGP_CUR:%.*]] = load ptr, ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    [[TMP0:%.*]] = getelementptr inbounds i8, ptr [[ARGP_CUR]], i32 15
// CHECK-NEXT:    [[ARGP_CUR_ALIGNED:%.*]] = call ptr @llvm.ptrmask.p0.i32(ptr [[TMP0]], i32 -16)
// CHECK-NEXT:    [[ARGP_NEXT:%.*]] = getelementptr inbounds i8, ptr [[ARGP_CUR_ALIGNED]], i32 16
// CHECK-NEXT:    store ptr [[ARGP_NEXT]], ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    [[TMP1:%.*]] = load <4 x i32>, ptr [[ARGP_CUR_ALIGNED]], align 16
// CHECK-NEXT:    [[TMP2:%.*]] = load ptr, ptr [[OUT_ADDR]], align 4
// CHECK-NEXT:    store <4 x i32> [[TMP1]], ptr [[TMP2]], align 16
// CHECK-NEXT:    ret void
//
void vector(__builtin_va_list list, v128_t *out) {
  *out = __builtin_va_arg(list, v128_t);
}

typedef struct BF {
  float not_an_i32[2];
  int A : 1;
  char B;
  int C : 13;
} BF;

// CHECK-LABEL: @bitfield(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[LIST_ADDR:%.*]] = alloca ptr, align 4
// CHECK-NEXT:    [[OUT_ADDR:%.*]] = alloca ptr, align 4
// CHECK-NEXT:    store ptr [[LIST:%.*]], ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    store ptr [[OUT:%.*]], ptr [[OUT_ADDR]], align 4
// CHECK-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[OUT_ADDR]], align 4
// CHECK-NEXT:    [[ARGP_CUR:%.*]] = load ptr, ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    [[ARGP_NEXT:%.*]] = getelementptr inbounds i8, ptr [[ARGP_CUR]], i32 4
// CHECK-NEXT:    store ptr [[ARGP_NEXT]], ptr [[LIST_ADDR]], align 4
// CHECK-NEXT:    [[TMP1:%.*]] = load ptr, ptr [[ARGP_CUR]], align 4
// CHECK-NEXT:    call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[TMP0]], ptr align 4 [[TMP1]], i32 12, i1 false)
// CHECK-NEXT:    ret void
//
void bitfield(__builtin_va_list list, BF *out) {
  *out = __builtin_va_arg(list, BF);
}
