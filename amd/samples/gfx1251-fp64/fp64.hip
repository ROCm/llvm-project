// Minimal HIP app that exercises every FP64 instruction of gfx1251.
//
// Each kernel covers one instruction class. The comment next to each line
// names the gfx1251 instruction(s) it selects; `make check` compiles for
// gfx1251 and verifies that every name in this file appears in the ISA.
// `make run` executes all kernels and compares against host references.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define HIP_CHECK(e)                                                           \
  do {                                                                         \
    hipError_t err_ = (e);                                                     \
    if (err_ != hipSuccess) {                                                  \
      printf("%s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(err_));      \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

typedef double d2 __attribute__((ext_vector_type(2)));
typedef double d8 __attribute__((ext_vector_type(8)));

constexpr int N = 256;    // elements per test
constexpr int BLOCK = 64; // two wave32 waves
constexpr double TWO_OVER_PI_HI = 0x1.45f306dc9c882p-1; // 2/pi, top 53 bits
#define COUNT(...) +1

// VOP1/VOP2/VOP3: X(device expression, host reference, relative tolerance).
// a, c: any sign; b > 0; k: small int; f: float. fma(a, 3.0, c) must come
// before the other uses of c: a live c keeps it from becoming v_fmac_f64.
#define VALU_OPS(X)                                                                           \
  X((double)((int)a + k),              (double)((int)a + k),             0)     /* v_cvt_i32_f64 v_cvt_f64_i32 */ \
  X((double)((unsigned)b + (k + 8u)),  (double)((unsigned)b + (k + 8u)), 0)     /* v_cvt_u32_f64 v_cvt_f64_u32 */ \
  X((double)((float)a * f),            (double)((float)a * f),           0)     /* v_cvt_f32_f64 v_cvt_f64_f32 */ \
  X(__builtin_trunc(a),                std::trunc(a),                    0)     /* v_trunc_f64 */         \
  X(__builtin_ceil(a),                 std::ceil(a),                     0)     /* v_ceil_f64 */          \
  X(__builtin_floor(a),                std::floor(a),                    0)     /* v_floor_f64 */         \
  X(__builtin_rint(a),                 std::rint(a),                     0)     /* v_rndne_f64 */         \
  X(__builtin_amdgcn_fract(a),         a - std::floor(a),                1e-15) /* v_fract_f64 */         \
  X(__builtin_amdgcn_frexp_mant(a),    mant(a),                          0)     /* v_frexp_mant_f64 */    \
  X(__builtin_amdgcn_frexp_exp(a),     expo(a),                          0)     /* v_frexp_exp_i32_f64 */ \
  X(__builtin_amdgcn_rcp(b),           1 / b,                            1e-6)  /* v_rcp_f64 */           \
  X(__builtin_amdgcn_rsq(b),           1 / std::sqrt(b),                 1e-6)  /* v_rsq_f64 */           \
  X(__builtin_amdgcn_sqrt(b),          std::sqrt(b),                     1e-6)  /* v_sqrt_f64 */          \
  X(a + b,                             a + b,                            0)     /* v_add_f64 */           \
  X(a * b,                             a * b,                            0)     /* v_mul_f64 */           \
  X(__builtin_fma(a, 3.0, c),          std::fma(a, 3.0, c),              0)     /* v_fmamk_f64 */         \
  X(__builtin_fmax(a, c),              std::fmax(a, c),                  0)     /* v_max_num_f64 */       \
  X(__builtin_fmin(a, c),              std::fmin(a, c),                  0)     /* v_min_num_f64 */       \
  X(__builtin_elementwise_maximum(a, c), std::fmax(a, c),                0)     /* v_maximum_f64 */       \
  X(__builtin_elementwise_minimum(a, c), std::fmin(a, c),                0)     /* v_minimum_f64 */       \
  X(__builtin_fma(a, b, -c),           std::fma(a, b, -c),               0)     /* v_fma_f64 */           \
  X(__builtin_fma(a, b, c),            std::fma(a, b, c),                0)     /* v_fmac_f64 */          \
  X(__builtin_fma(a, b, 3.0),          std::fma(a, b, 3.0),              0)     /* v_fmaak_f64 */         \
  X(__builtin_ldexp(a, k),             std::ldexp(a, k),                 0)     /* v_ldexp_f64 */         \
  X(a / b,                             a / b,                            0)     /* v_div_scale_f64 v_div_fmas_f64 v_div_fixup_f64 */ \
  X(__builtin_amdgcn_trig_preop(a, 0), TWO_OVER_PI_HI,                   0)     /* v_trig_preop_f64 */
constexpr int NV = 0 VALU_OPS(COUNT);

constexpr int CLASS = 0x2c; // fp-class mask: -inf, -normal, -0

// VOPC: the 14 ordered/unordered predicates plus fp-class. Each predicate gets
// its own b; with a shared b the compiler folds a predicate and its inverse
// (e.g. a < b and !(a < b)) into one compare.
#define CMP_OPS(X)                                                             \
  X(a == b)                         /* v_cmp_eq_f64    v_cmpx_eq_f64    */     \
  X(a != b)                         /* v_cmp_neq_f64   v_cmpx_neq_f64   */     \
  X(a < b)                          /* v_cmp_lt_f64    v_cmpx_lt_f64    */     \
  X(a <= b)                         /* v_cmp_le_f64    v_cmpx_le_f64    */     \
  X(a > b)                          /* v_cmp_gt_f64    v_cmpx_gt_f64    */     \
  X(a >= b)                         /* v_cmp_ge_f64    v_cmpx_ge_f64    */     \
  X(__builtin_islessgreater(a, b))  /* v_cmp_lg_f64    v_cmpx_lg_f64    */     \
  X(!__builtin_isunordered(a, b))   /* v_cmp_o_f64     v_cmpx_o_f64     */     \
  X(__builtin_isunordered(a, b))    /* v_cmp_u_f64     v_cmpx_u_f64     */     \
  X(!__builtin_islessgreater(a, b)) /* v_cmp_nlg_f64   v_cmpx_nlg_f64   */     \
  X(!(a >= b))                      /* v_cmp_nge_f64   v_cmpx_nge_f64   */     \
  X(!(a > b))                       /* v_cmp_ngt_f64   v_cmpx_ngt_f64   */     \
  X(!(a <= b))                      /* v_cmp_nle_f64   v_cmpx_nle_f64   */     \
  X(!(a < b))                       /* v_cmp_nlt_f64   v_cmpx_nlt_f64   */     \
  X(__builtin_isfpclass(b, CLASS))  /* v_cmp_class_f64 (v_cmpx: see cmpx) */
constexpr int NC = 0 CMP_OPS(COUNT);

// VOP3P packed FP64 (new in gfx1251); same expressions on host scalars.
#define PACKED_OPS(X)                                                          \
  X(a + b)                              /* v_pk_add_f64 */                     \
  X(a * b)                              /* v_pk_mul_f64 */                     \
  X(__builtin_elementwise_fma(a, b, c)) /* v_pk_fma_f64 */                     \
  X(__builtin_elementwise_maxnum(a, c)) /* v_pk_max_num_f64 */                 \
  X(__builtin_elementwise_minnum(a, c)) /* v_pk_min_num_f64 */
constexpr int NP = 0 PACKED_OPS(COUNT);

// VOPD dual issue: an f64 op in the X slot next to an independent f32 op.
// The fma negates c because v_fmac_f64 has no dual form.
#define VOPD_OPS(X)                                                            \
  X(a + b,                   f + g)                   /* v_dual_add_f64 */     \
  X(a * b,                   f * g)                   /* v_dual_mul_f64 */     \
  X(__builtin_fma(a, b, -c), __builtin_fmaf(f, g, h)) /* v_dual_fma_f64 */     \
  X(__builtin_fmax(a, b),    __builtin_fmaxf(f, g))   /* v_dual_max_num_f64 */ \
  X(__builtin_fmin(a, b),    __builtin_fminf(f, g))   /* v_dual_min_num_f64 */
constexpr int ND = 0 VOPD_OPS(COUNT);

__global__ void valu(const double* x, const double* y, const double* z,
                     const int* ki, const float* fi, double* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x, op = 0;
  double a = x[i], b = y[i], c = z[i];
  int k = ki[i];
  float f = fi[i];
#define X(dev, ref, tol) out[op++ * N + i] = (dev);
  VALU_OPS(X)
#undef X
}

__global__ void cmp(const double* x, const double* y, unsigned* mask) {
  int i = blockIdx.x * blockDim.x + threadIdx.x, op = 0;
  double a = x[i];
  unsigned m = 0;
#define X(pred) { double b = y[op * N + i]; m |= unsigned(pred) << op++; }
  CMP_OPS(X)
#undef X
  mask[i] = m;
}

// A divergent branch on a compare writes EXEC directly (v_cmpx). Using b in
// the branch keeps its registers from being reused before the EXEC update.
// The compiler never turns a class compare into v_cmpx, so that one is written
// out: it saves EXEC, compares, reads the passing lanes and restores EXEC.
// gfx1251 is wave32-only, so exec_lo is the whole mask.
__global__ void cmpx(const double* x, const double* y, double* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x, op = 0;
  double a = x[i];
#define X(pred) { double b = y[op * N + i]; if (pred) out[op * N + i] = b; ++op; }
  CMP_OPS(X)
#undef X
  unsigned saved, pass;
  asm volatile("s_mov_b32 %0, exec_lo\n\t"
               "v_cmpx_class_f64 %2, %3\n\t"
               "s_mov_b32 %1, exec_lo\n\t"
               "s_mov_b32 exec_lo, %0"
               : "=&s"(saved), "=&s"(pass)
               : "v"(y[(NC - 1) * N + i]), "n"(CLASS));
  out[NC * N + i] = pass >> __lane_id() & 1;
}

__global__ void packed(const d2* x, const d2* y, const d2* z, d2* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x, op = 0;
  d2 a = x[i], b = y[i], c = z[i];
#define X(e) out[op++ * N / 2 + i] = (e);
  PACKED_OPS(X)
#undef X
}

__global__ void vopd(const double* x, const double* y, const double* z,
                     const float* fi, double* out, float* fout) {
  int i = blockIdx.x * blockDim.x + threadIdx.x, op = 0;
  double a = x[i], b = y[i], c = z[i];
  float f = fi[i], g = fi[N + i], h = fi[2 * N + i];
#define X(e64, e32) out[op * N + i] = (e64); fout[op++ * N + i] = (e32);
  VOPD_OPS(X)
#undef X
}

// D(16x16) = A(16x4) * B(4x16) + C, row-major, one wave32. Lane l holds
// A[l%16][kk], B[kk][l%16] for kk = 2*(l/16)+{0,1} and D[8*(l/16)+j][l%16].
__global__ void wmma(const double* A, const double* B, const double* C,
                     double* D) {
  int l = threadIdx.x, r = l % 16, kk = 2 * (l / 16), m = 8 * (l / 16);
  d2 a = {A[r * 4 + kk], A[r * 4 + kk + 1]};
  d2 b = {B[kk * 16 + r], B[(kk + 1) * 16 + r]};
  d8 c;
  for (int j = 0; j < 8; ++j)
    c[j] = C[(m + j) * 16 + r];
  d8 d = __builtin_amdgcn_wmma_f64_16x16x4_f64(false, a, false, b, (short)0, c,
                                               false, false); // v_wmma_f64_16x16x4_f64
  for (int j = 0; j < 8; ++j)
    D[(m + j) * 16 + r] = d[j];
}

// gfx1251 adds DPP to 64-bit VALU ops (row_share only). The move only folds
// into its user with bound_ctrl set; every lane has a source under row_share.
__global__ void dpp(const double* x, const double* y, double* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  out[i] = __builtin_amdgcn_mov_dpp(x[i], 0x153 /*row_share:3*/, 0xf, 0xf,
                                    true) + y[i]; // v_add_f64_dpp
}

// Single block. g = {sum, min, max} x {global, flat, buffer, LDS}, then the
// sums of the values returned by the LDS atomics.
__global__ void atomics(double* g, int use_lds) {
  __shared__ double s[6];
  int t = threadIdx.x;
  double v = (t % 37) * 0.25 - 3.0; // sums are exact in any order
  if (t == 0) {
    s[0] = s[3] = s[4] = s[5] = 0;
    s[1] = INFINITY;
    s[2] = -INFINITY;
  }
  __syncthreads();

  constexpr int R = __ATOMIC_RELAXED, DEV = __MEMORY_SCOPE_DEVICE,
                WG = __MEMORY_SCOPE_WRKGRP;
  __scoped_atomic_fetch_add(&g[0], v, R, DEV); // global_atomic_add_f64
  __scoped_atomic_fetch_min(&g[1], v, R, DEV); // global_atomic_min_num_f64
  __scoped_atomic_fetch_max(&g[2], v, R, DEV); // global_atomic_max_num_f64

  double* p = use_lds ? s : g; // unknown address space; always g at run time
  __scoped_atomic_fetch_add(&p[3], v, R, DEV); // flat_atomic_add_f64
  __scoped_atomic_fetch_min(&p[4], v, R, DEV); // flat_atomic_min_num_f64
  __scoped_atomic_fetch_max(&p[5], v, R, DEV); // flat_atomic_max_num_f64

  __amdgpu_buffer_rsrc_t rsrc =
      __builtin_amdgcn_make_buffer_rsrc(g + 6, 0, 3 * sizeof(double), 0);
  __builtin_amdgcn_raw_ptr_buffer_atomic_fadd_f64(v, rsrc, 0, 0, 0);  // buffer_atomic_add_f64
  __builtin_amdgcn_raw_ptr_buffer_atomic_fmin_f64(v, rsrc, 8, 0, 0);  // buffer_atomic_min_num_f64
  __builtin_amdgcn_raw_ptr_buffer_atomic_fmax_f64(v, rsrc, 16, 0, 0); // buffer_atomic_max_num_f64

  __scoped_atomic_fetch_add(&s[0], v, R, WG); // ds_add_f64
  __scoped_atomic_fetch_min(&s[1], v, R, WG); // ds_min_num_f64
  __scoped_atomic_fetch_max(&s[2], v, R, WG); // ds_max_num_f64
  // Returned values: {0..N-1}, {0, -1, ...}, {0, 1, ...} in some order.
  double r0 = __scoped_atomic_fetch_add(&s[3], 1.0, R, WG);  // ds_add_rtn_f64
  double r1 = __scoped_atomic_fetch_min(&s[4], -1.0, R, WG); // ds_min_num_rtn_f64
  double r2 = __scoped_atomic_fetch_max(&s[5], 1.0, R, WG);  // ds_max_num_rtn_f64
  __scoped_atomic_fetch_add(&g[12], r0, R, DEV);
  __scoped_atomic_fetch_add(&g[13], r1, R, DEV);
  __scoped_atomic_fetch_add(&g[14], r2, R, DEV);
  __syncthreads();
  if (t == 0) {
    g[9] = s[0];
    g[10] = s[1];
    g[11] = s[2];
  }
}

template <class T> T* toDevice(const std::vector<T>& h) {
  T* d;
  HIP_CHECK(hipMalloc(&d, h.size() * sizeof(T)));
  HIP_CHECK(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice));
  return d;
}

template <class T> std::vector<T> toHost(const T* d, size_t n) {
  std::vector<T> h(n);
  HIP_CHECK(hipMemcpy(h.data(), d, n * sizeof(T), hipMemcpyDeviceToHost));
  return h;
}

static int failures = 0;
static void expect(const char* what, int i, double got, double want,
                   double tol = 0) {
  if (got == want || (std::isnan(got) && std::isnan(want)) ||
      std::fabs(got - want) <= tol * std::fabs(want))
    return;
  if (failures++ < 20)
    printf("FAIL %s [%d]: got %.17g, want %.17g\n", what, i, got, want);
}

static double mant(double v) { int e; return std::frexp(v, &e); }
static int expo(double v) { int e; std::frexp(v, &e); return e; }

int main() {
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  printf("device: %s\n", prop.gcnArchName);

  const double special[] = {NAN, INFINITY, -INFINITY, -0.0, 0.0};
  std::vector<double> x(N), y(N), z(N), p(N), q(NC * N);
  std::vector<int> ki(N);
  std::vector<float> fi(3 * N);
  for (int i = 0; i < N; ++i) {
    x[i] = (i - N / 2) * 0.37 + 0.1;
    y[i] = 1 + 0.25 * i;
    z[i] = 0.5 * i - 7.3;
    ki[i] = i % 17 - 8;
    p[i] = i % 4 ? x[i] : special[i / 4 % 5];
  }
  for (int j = 0; j < 3 * N; ++j)
    fi[j] = 0.125f * (j % 29) - 1;
  for (int j = 0; j < NC * N; ++j)
    q[j] = j % 3 ? z[j % N] : p[j % N];
  std::vector<double> A(16 * 4), B(4 * 16), C(16 * 16);
  for (int i = 0; i < 64; ++i) {
    A[i] = i % 5 - 2.5;
    B[i] = i % 7 * 0.5 - 1;
  }
  for (int i = 0; i < 256; ++i)
    C[i] = i % 9 - 4;
  std::vector<double> g0 = {0, INFINITY, -INFINITY, 0, INFINITY, -INFINITY,
                            0, INFINITY, -INFINITY, 0, 0, 0, 0, 0, 0};

  double *dx = toDevice(x), *dy = toDevice(y), *dz = toDevice(z);
  double *dp = toDevice(p), *dq = toDevice(q);
  double *dA = toDevice(A), *dB = toDevice(B), *dC = toDevice(C);
  double *dg = toDevice(g0), *dD = toDevice(std::vector<double>(256));
  double *dvalu = toDevice(std::vector<double>(NV * N));
  double *dpk = toDevice(std::vector<double>(NP * N));
  double *dvd = toDevice(std::vector<double>(ND * N));
  double *ddpp = toDevice(std::vector<double>(N));
  double *dtaken = toDevice(std::vector<double>((NC + 1) * N));
  float *dvf = toDevice(std::vector<float>(ND * N)), *df = toDevice(fi);
  int* dk = toDevice(ki);
  unsigned* dmask = toDevice(std::vector<unsigned>(N));

  valu<<<N / BLOCK, BLOCK>>>(dx, dy, dz, dk, df, dvalu);
  cmp<<<N / BLOCK, BLOCK>>>(dp, dq, dmask);
  cmpx<<<N / BLOCK, BLOCK>>>(dp, dq, dtaken);
  packed<<<N / 2 / BLOCK, BLOCK>>>((d2*)dx, (d2*)dy, (d2*)dz, (d2*)dpk);
  vopd<<<N / BLOCK, BLOCK>>>(dx, dy, dz, df, dvd, dvf);
  wmma<<<1, 32>>>(dA, dB, dC, dD);
  dpp<<<N / BLOCK, BLOCK>>>(dx, dy, ddpp);
  atomics<<<1, N>>>(dg, 0);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  auto vout = toHost(dvalu, NV * N), pout = toHost(dpk, NP * N);
  auto vd = toHost(dvd, ND * N), dppout = toHost(ddpp, N);
  auto D = toHost(dD, 256), gout = toHost(dg, 15);
  auto vf = toHost(dvf, ND * N);
  auto mask = toHost(dmask, N);
  auto taken = toHost(dtaken, (NC + 1) * N);

  for (int i = 0; i < N; ++i) {
    double a = x[i], b = y[i], c = z[i];
    int k = ki[i], op = 0;
    float f = fi[i], g = fi[N + i], h = fi[2 * N + i];
#define X(dev, ref, tol) expect(#dev, i, vout[op++ * N + i], ref, tol);
    VALU_OPS(X)
#undef X
    op = 0;
#define X(e) expect(#e, i, pout[op++ * N + i], e);
    PACKED_OPS(X)
#undef X
    op = 0;
#define X(e64, e32) expect(#e64, i, vd[op * N + i], e64); expect(#e32, i, vf[op++ * N + i], e32);
    VOPD_OPS(X)
#undef X
    expect("dpp row_share:3", i, dppout[i], x[i / 16 * 16 + 3] + y[i]);

    a = p[i], op = 0;
    unsigned m = 0;
#define X(pred) { double b = q[op * N + i]; m |= unsigned(pred) << op; expect("cmpx " #pred, i, taken[op++ * N + i], (pred) ? b : 0); }
    CMP_OPS(X)
#undef X
    expect("cmp mask", i, mask[i], m);
    expect("v_cmpx_class_f64", i, taken[NC * N + i],
           __builtin_isfpclass(q[(NC - 1) * N + i], CLASS));
  }
  for (int i = 0; i < 256; ++i) {
    double want = C[i];
    for (int kk = 0; kk < 4; ++kk)
      want += A[i / 16 * 4 + kk] * B[kk * 16 + i % 16];
    expect("wmma", i, D[i], want);
  }
  double sum = 0, mn = INFINITY, mx = -INFINITY;
  for (int t = 0; t < N; ++t) {
    double v = (t % 37) * 0.25 - 3.0;
    sum += v, mn = std::fmin(mn, v), mx = std::fmax(mx, v);
  }
  const char* where[] = {"global", "flat", "buffer", "lds"};
  for (int j = 0; j < 4; ++j) {
    expect(where[j], 3 * j + 0, gout[3 * j + 0], sum);
    expect(where[j], 3 * j + 1, gout[3 * j + 1], mn);
    expect(where[j], 3 * j + 2, gout[3 * j + 2], mx);
  }
  expect("lds add returns", 12, gout[12], N * (N - 1) / 2.0);
  expect("lds min returns", 13, gout[13], -(N - 1));
  expect("lds max returns", 14, gout[14], N - 1);

  printf("%s\n", failures ? "FAILED" : "PASSED");
  return failures != 0;
}
