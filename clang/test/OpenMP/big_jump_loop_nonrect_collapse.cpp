// NOTE: Assertions have been autogenerated by utils/update_cc_test_checks.py UTC_ARGS: --function-signature --include-generated-funcs --replace-value-regex "__omp_offloading_[0-9a-z]+_[0-9a-z]+" "reduction_size[.].+[.]" "pl_cond[.].+[.|,]" --prefix-filecheck-ir-name _
// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -fopenmp-version=50 -triple x86_64-unknown-unknown -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -fopenmp-version=50 -triple amdgcn-amd-amdhsa -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s
// expected-no-diagnostics

int main()
{
  const int N = 10000;

  double arr[N*N];

#pragma omp target teams distribute parallel for collapse(2)
  for (int j = 0; j < N; j++) {
    for (int i = j; i < N; i++) {
      arr[j * N + i]++;
    }
  }

  return 0;
}
// CHECK-LABEL: define {{[^@]+}}@{{__omp_offloading_[0-9a-z]+_[0-9a-z]+}}_main_l14
// CHECK-SAME: (ptr noalias noundef [[DYN_PTR:%.*]], ptr noundef nonnull align 8 dereferenceable(800000000) [[ARR:%.*]], i64 noundef [[N:%.*]]) #[[ATTR0:[0-9]+]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[DYN_PTR_ADDR:%.*]] = alloca ptr, align 8, addrspace(5)
// CHECK-NEXT:    [[ARR_ADDR:%.*]] = alloca ptr, align 8, addrspace(5)
// CHECK-NEXT:    [[N_ADDR:%.*]] = alloca i64, align 8, addrspace(5)
// CHECK-NEXT:    [[J:%.*]] = alloca i32, align 4, addrspace(5)
// CHECK-NEXT:    [[I:%.*]] = alloca i32, align 4, addrspace(5)
// CHECK-NEXT:    [[DOTLB_MIN:%.*]] = alloca i32, align 4, addrspace(5)
// CHECK-NEXT:    [[DOTLB_MAX:%.*]] = alloca i32, align 4, addrspace(5)
// CHECK-NEXT:    [[DOTMIN_LESS_MAX:%.*]] = alloca i8, align 1, addrspace(5)
// CHECK-NEXT:    [[DOTLOWER:%.*]] = alloca i32, align 4, addrspace(5)
// CHECK-NEXT:    [[DOTCAPTURE_EXPR_:%.*]] = alloca i64, align 8, addrspace(5)
// CHECK-NEXT:    [[DOTOMP_LB:%.*]] = alloca i64, align 8, addrspace(5)
// CHECK-NEXT:    [[DOTOMP_UB:%.*]] = alloca i64, align 8, addrspace(5)
// CHECK-NEXT:    [[DOTOMP_IV:%.*]] = alloca i64, align 8, addrspace(5)
// CHECK-NEXT:    [[DYN_PTR_ADDR_ASCAST:%.*]] = addrspacecast ptr addrspace(5) [[DYN_PTR_ADDR]] to ptr
// CHECK-NEXT:    [[ARR_ADDR_ASCAST:%.*]] = addrspacecast ptr addrspace(5) [[ARR_ADDR]] to ptr
// CHECK-NEXT:    [[N_ADDR_ASCAST:%.*]] = addrspacecast ptr addrspace(5) [[N_ADDR]] to ptr
// CHECK-NEXT:    [[J_ASCAST:%.*]] = addrspacecast ptr addrspace(5) [[J]] to ptr
// CHECK-NEXT:    [[I_ASCAST:%.*]] = addrspacecast ptr addrspace(5) [[I]] to ptr
// CHECK-NEXT:    [[DOTLB_MIN_ASCAST:%.*]] = addrspacecast ptr addrspace(5) [[DOTLB_MIN]] to ptr
// CHECK-NEXT:    [[DOTLB_MAX_ASCAST:%.*]] = addrspacecast ptr addrspace(5) [[DOTLB_MAX]] to ptr
// CHECK-NEXT:    [[DOTMIN_LESS_MAX_ASCAST:%.*]] = addrspacecast ptr addrspace(5) [[DOTMIN_LESS_MAX]] to ptr
// CHECK-NEXT:    [[DOTLOWER_ASCAST:%.*]] = addrspacecast ptr addrspace(5) [[DOTLOWER]] to ptr
// CHECK-NEXT:    [[DOTCAPTURE_EXPR__ASCAST:%.*]] = addrspacecast ptr addrspace(5) [[DOTCAPTURE_EXPR_]] to ptr
// CHECK-NEXT:    [[DOTOMP_LB_ASCAST:%.*]] = addrspacecast ptr addrspace(5) [[DOTOMP_LB]] to ptr
// CHECK-NEXT:    [[DOTOMP_UB_ASCAST:%.*]] = addrspacecast ptr addrspace(5) [[DOTOMP_UB]] to ptr
// CHECK-NEXT:    [[DOTOMP_IV_ASCAST:%.*]] = addrspacecast ptr addrspace(5) [[DOTOMP_IV]] to ptr
// CHECK-NEXT:    store ptr [[DYN_PTR]], ptr [[DYN_PTR_ADDR_ASCAST]], align 8
// CHECK-NEXT:    store ptr [[ARR]], ptr [[ARR_ADDR_ASCAST]], align 8
// CHECK-NEXT:    store i64 [[N]], ptr [[N_ADDR_ASCAST]], align 8
// CHECK-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[ARR_ADDR_ASCAST]], align 8
// CHECK-NEXT:    call void @__kmpc_specialized_kernel_init()
// CHECK-NEXT:    store i32 0, ptr [[J_ASCAST]], align 4
// CHECK-NEXT:    [[TMP1:%.*]] = load i32, ptr [[J_ASCAST]], align 4
// CHECK-NEXT:    store i32 [[TMP1]], ptr [[I_ASCAST]], align 4
// CHECK-NEXT:    store i32 0, ptr [[J_ASCAST]], align 4
// CHECK-NEXT:    [[TMP2:%.*]] = load i32, ptr [[J_ASCAST]], align 4
// CHECK-NEXT:    store i32 [[TMP2]], ptr [[DOTLB_MIN_ASCAST]], align 4
// CHECK-NEXT:    store i32 9999, ptr [[J_ASCAST]], align 4
// CHECK-NEXT:    [[TMP3:%.*]] = load i32, ptr [[J_ASCAST]], align 4
// CHECK-NEXT:    store i32 [[TMP3]], ptr [[DOTLB_MAX_ASCAST]], align 4
// CHECK-NEXT:    [[TMP4:%.*]] = load i32, ptr [[DOTLB_MIN_ASCAST]], align 4
// CHECK-NEXT:    [[TMP5:%.*]] = load i32, ptr [[DOTLB_MAX_ASCAST]], align 4
// CHECK-NEXT:    [[CMP:%.*]] = icmp slt i32 [[TMP4]], [[TMP5]]
// CHECK-NEXT:    [[STOREDV:%.*]] = zext i1 [[CMP]] to i8
// CHECK-NEXT:    store i8 [[STOREDV]], ptr [[DOTMIN_LESS_MAX_ASCAST]], align 1
// CHECK-NEXT:    [[TMP6:%.*]] = load i8, ptr [[DOTMIN_LESS_MAX_ASCAST]], align 1
// CHECK-NEXT:    [[LOADEDV:%.*]] = trunc i8 [[TMP6]] to i1
// CHECK-NEXT:    br i1 [[LOADEDV]], label [[COND_TRUE:%.*]], label [[COND_FALSE:%.*]]
// CHECK:       cond.true:
// CHECK-NEXT:    [[TMP7:%.*]] = load i32, ptr [[DOTLB_MIN_ASCAST]], align 4
// CHECK-NEXT:    br label [[COND_END:%.*]]
// CHECK:       cond.false:
// CHECK-NEXT:    [[TMP8:%.*]] = load i32, ptr [[DOTLB_MAX_ASCAST]], align 4
// CHECK-NEXT:    br label [[COND_END]]
// CHECK:       cond.end:
// CHECK-NEXT:    [[COND:%.*]] = phi i32 [ [[TMP7]], [[COND_TRUE]] ], [ [[TMP8]], [[COND_FALSE]] ]
// CHECK-NEXT:    store i32 [[COND]], ptr [[J_ASCAST]], align 4
// CHECK-NEXT:    store i32 [[COND]], ptr [[DOTLOWER_ASCAST]], align 4
// CHECK-NEXT:    [[TMP9:%.*]] = load i32, ptr [[DOTLOWER_ASCAST]], align 4
// CHECK-NEXT:    [[SUB:%.*]] = sub i32 10000, [[TMP9]]
// CHECK-NEXT:    [[SUB1:%.*]] = sub i32 [[SUB]], 1
// CHECK-NEXT:    [[ADD:%.*]] = add i32 [[SUB1]], 1
// CHECK-NEXT:    [[DIV:%.*]] = udiv i32 [[ADD]], 1
// CHECK-NEXT:    [[CONV:%.*]] = zext i32 [[DIV]] to i64
// CHECK-NEXT:    [[MUL:%.*]] = mul nsw i64 10000, [[CONV]]
// CHECK-NEXT:    [[SUB2:%.*]] = sub nsw i64 [[MUL]], 1
// CHECK-NEXT:    store i64 [[SUB2]], ptr [[DOTCAPTURE_EXPR__ASCAST]], align 8
// CHECK-NEXT:    store i32 0, ptr [[J_ASCAST]], align 4
// CHECK-NEXT:    [[TMP10:%.*]] = load i32, ptr [[J_ASCAST]], align 4
// CHECK-NEXT:    store i32 [[TMP10]], ptr [[I_ASCAST]], align 4
// CHECK-NEXT:    store i64 0, ptr [[DOTOMP_LB_ASCAST]], align 8
// CHECK-NEXT:    [[TMP11:%.*]] = load i64, ptr [[DOTCAPTURE_EXPR__ASCAST]], align 8
// CHECK-NEXT:    store i64 [[TMP11]], ptr [[DOTOMP_UB_ASCAST]], align 8
// CHECK-NEXT:    [[TMP12:%.*]] = load i64, ptr [[DOTOMP_LB_ASCAST]], align 8
// CHECK-NEXT:    store i64 [[TMP12]], ptr [[DOTOMP_IV_ASCAST]], align 8
// CHECK-NEXT:    [[TMP13:%.*]] = call i32 @__kmpc_get_hardware_thread_id_in_block()
// CHECK-NEXT:    [[NVPTX_NUM_THREADS:%.*]] = call i32 @__kmpc_get_hardware_num_threads_in_block()
// CHECK-NEXT:    [[GPU_BLOCK_ID:%.*]] = call i32 @llvm.amdgcn.workgroup.id.x()
// CHECK-NEXT:    [[TMP14:%.*]] = mul i32 [[GPU_BLOCK_ID]], [[NVPTX_NUM_THREADS]]
// CHECK-NEXT:    [[TMP15:%.*]] = add i32 [[TMP14]], [[TMP13]]
// CHECK-NEXT:    [[TMP16:%.*]] = zext i32 [[TMP15]] to i64
// CHECK-NEXT:    [[TMP17:%.*]] = mul i64 [[TMP16]], 1
// CHECK-NEXT:    [[TMP18:%.*]] = load i64, ptr [[DOTOMP_IV_ASCAST]], align 8
// CHECK-NEXT:    [[TMP19:%.*]] = add i64 [[TMP17]], [[TMP18]]
// CHECK-NEXT:    store i64 [[TMP19]], ptr [[DOTOMP_IV_ASCAST]], align 8
// CHECK-NEXT:    br label [[FOR_COND:%.*]]
// CHECK:       for.cond:
// CHECK-NEXT:    [[TMP20:%.*]] = load i64, ptr [[DOTOMP_IV_ASCAST]], align 8
// CHECK-NEXT:    [[TMP21:%.*]] = load i64, ptr [[DOTOMP_UB_ASCAST]], align 8
// CHECK-NEXT:    [[CMP3:%.*]] = icmp sle i64 [[TMP20]], [[TMP21]]
// CHECK-NEXT:    br i1 [[CMP3]], label [[FOR_BODY:%.*]], label [[FOR_END:%.*]]
// CHECK:       for.body:
// CHECK-NEXT:    [[TMP22:%.*]] = load i64, ptr [[DOTOMP_IV_ASCAST]], align 8
// CHECK-NEXT:    [[TMP23:%.*]] = load i32, ptr [[DOTLOWER_ASCAST]], align 4
// CHECK-NEXT:    [[SUB4:%.*]] = sub i32 10000, [[TMP23]]
// CHECK-NEXT:    [[SUB5:%.*]] = sub i32 [[SUB4]], 1
// CHECK-NEXT:    [[ADD6:%.*]] = add i32 [[SUB5]], 1
// CHECK-NEXT:    [[DIV7:%.*]] = udiv i32 [[ADD6]], 1
// CHECK-NEXT:    [[MUL8:%.*]] = mul i32 1, [[DIV7]]
// CHECK-NEXT:    [[CONV9:%.*]] = zext i32 [[MUL8]] to i64
// CHECK-NEXT:    [[DIV10:%.*]] = sdiv i64 [[TMP22]], [[CONV9]]
// CHECK-NEXT:    [[MUL11:%.*]] = mul nsw i64 [[DIV10]], 1
// CHECK-NEXT:    [[ADD12:%.*]] = add nsw i64 0, [[MUL11]]
// CHECK-NEXT:    [[CONV13:%.*]] = trunc i64 [[ADD12]] to i32
// CHECK-NEXT:    store i32 [[CONV13]], ptr [[J_ASCAST]], align 4
// CHECK-NEXT:    [[TMP24:%.*]] = load i32, ptr [[J_ASCAST]], align 4
// CHECK-NEXT:    [[CONV14:%.*]] = sext i32 [[TMP24]] to i64
// CHECK-NEXT:    [[TMP25:%.*]] = load i64, ptr [[DOTOMP_IV_ASCAST]], align 8
// CHECK-NEXT:    [[TMP26:%.*]] = load i64, ptr [[DOTOMP_IV_ASCAST]], align 8
// CHECK-NEXT:    [[TMP27:%.*]] = load i32, ptr [[DOTLOWER_ASCAST]], align 4
// CHECK-NEXT:    [[SUB15:%.*]] = sub i32 10000, [[TMP27]]
// CHECK-NEXT:    [[SUB16:%.*]] = sub i32 [[SUB15]], 1
// CHECK-NEXT:    [[ADD17:%.*]] = add i32 [[SUB16]], 1
// CHECK-NEXT:    [[DIV18:%.*]] = udiv i32 [[ADD17]], 1
// CHECK-NEXT:    [[MUL19:%.*]] = mul i32 1, [[DIV18]]
// CHECK-NEXT:    [[CONV20:%.*]] = zext i32 [[MUL19]] to i64
// CHECK-NEXT:    [[DIV21:%.*]] = sdiv i64 [[TMP26]], [[CONV20]]
// CHECK-NEXT:    [[TMP28:%.*]] = load i32, ptr [[DOTLOWER_ASCAST]], align 4
// CHECK-NEXT:    [[SUB22:%.*]] = sub i32 10000, [[TMP28]]
// CHECK-NEXT:    [[SUB23:%.*]] = sub i32 [[SUB22]], 1
// CHECK-NEXT:    [[ADD24:%.*]] = add i32 [[SUB23]], 1
// CHECK-NEXT:    [[DIV25:%.*]] = udiv i32 [[ADD24]], 1
// CHECK-NEXT:    [[MUL26:%.*]] = mul i32 1, [[DIV25]]
// CHECK-NEXT:    [[CONV27:%.*]] = zext i32 [[MUL26]] to i64
// CHECK-NEXT:    [[MUL28:%.*]] = mul nsw i64 [[DIV21]], [[CONV27]]
// CHECK-NEXT:    [[SUB29:%.*]] = sub nsw i64 [[TMP25]], [[MUL28]]
// CHECK-NEXT:    [[MUL30:%.*]] = mul nsw i64 [[SUB29]], 1
// CHECK-NEXT:    [[ADD31:%.*]] = add nsw i64 [[CONV14]], [[MUL30]]
// CHECK-NEXT:    [[CONV32:%.*]] = trunc i64 [[ADD31]] to i32
// CHECK-NEXT:    store i32 [[CONV32]], ptr [[I_ASCAST]], align 4
// CHECK-NEXT:    [[TMP29:%.*]] = load i32, ptr [[I_ASCAST]], align 4
// CHECK-NEXT:    [[CMP33:%.*]] = icmp slt i32 [[TMP29]], 10000
// CHECK-NEXT:    br i1 [[CMP33]], label [[OMP_BODY_NEXT:%.*]], label [[FOR_INC:%.*]]
// CHECK:       omp.body.next:
// CHECK-NEXT:    [[TMP30:%.*]] = load i32, ptr [[J_ASCAST]], align 4
// CHECK-NEXT:    [[MUL34:%.*]] = mul nsw i32 [[TMP30]], 10000
// CHECK-NEXT:    [[TMP31:%.*]] = load i32, ptr [[I_ASCAST]], align 4
// CHECK-NEXT:    [[ADD35:%.*]] = add nsw i32 [[MUL34]], [[TMP31]]
// CHECK-NEXT:    [[IDXPROM:%.*]] = sext i32 [[ADD35]] to i64
// CHECK-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds [100000000 x double], ptr [[TMP0]], i64 0, i64 [[IDXPROM]]
// CHECK-NEXT:    [[TMP32:%.*]] = load double, ptr [[ARRAYIDX]], align 8
// CHECK-NEXT:    [[INC:%.*]] = fadd double [[TMP32]], 1.000000e+00
// CHECK-NEXT:    store double [[INC]], ptr [[ARRAYIDX]], align 8
// CHECK-NEXT:    br label [[FOR_INC]]
// CHECK:       for.inc:
// CHECK-NEXT:    [[NVPTX_NUM_THREADS36:%.*]] = call i32 @__kmpc_get_hardware_num_threads_in_block()
// CHECK-NEXT:    [[TMP33:%.*]] = call i32 @__kmpc_get_hardware_num_blocks()
// CHECK-NEXT:    [[TMP34:%.*]] = mul i32 [[NVPTX_NUM_THREADS36]], [[TMP33]]
// CHECK-NEXT:    [[TMP35:%.*]] = zext i32 [[TMP34]] to i64
// CHECK-NEXT:    [[TMP36:%.*]] = mul i64 [[TMP35]], 1
// CHECK-NEXT:    [[TMP37:%.*]] = load i64, ptr [[DOTOMP_IV_ASCAST]], align 8
// CHECK-NEXT:    [[TMP38:%.*]] = add i64 [[TMP36]], [[TMP37]]
// CHECK-NEXT:    store i64 [[TMP38]], ptr [[DOTOMP_IV_ASCAST]], align 8
// CHECK-NEXT:    br label [[FOR_COND]], !llvm.loop [[LOOP7:![0-9]+]]
// CHECK:       for.end:
// CHECK-NEXT:    ret void
//
