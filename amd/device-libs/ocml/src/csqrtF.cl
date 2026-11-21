/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

#if !defined EXTRA_ACCURACY
#define EXTRA_ACCURACY
#endif

CONSTATTR float2
MATH_MANGLE(csqrt)(float2 z)
{
    float a = BUILTIN_ABS_F32(z.x);
    float b = BUILTIN_ABS_F32(z.y);
#if defined EXTRA_ACCURACY
    float t = BUILTIN_MAX_F32(a, b);

    if (!FINITE_ONLY_OPT()) {
        t = BUILTIN_ISUNORDERED_F32(a, b) ? QNAN_F32 : t;
    }

    int e = BUILTIN_FREXP_EXP_F32(t);
    float as = BUILTIN_FLDEXP_F32(a, -e);
    float bs = BUILTIN_FLDEXP_F32(b, -e);
    bool o = BUILTIN_CLASS_F32(t, CLASS_NZER|CLASS_PZER|CLASS_NINF|CLASS_PINF|CLASS_QNAN|CLASS_SNAN);

    // |z| scaled, with the error of its final rounding in pl
    float n = BUILTIN_FMA_F32(as, as, bs*bs);
    float y1 = BUILTIN_AMDGPU_RSQRT_F32(n);
    float h1 = 0.5f * y1;
    float p = n * y1;
    p = BUILTIN_FMA_F32(BUILTIN_FMA_F32(-p, p, n), h1, p);
    float pl = BUILTIN_FMA_F32(-p, p, n) * h1;
    p = o ? t : p;

    // s + sl = |z| + |x|
    int k = (e & 1) ^ 1;
    float sh = p + as;
    float sl = (as - (sh - p)) + pl;
    float s = BUILTIN_FLDEXP_F32(sh, k);
    sl = BUILTIN_FLDEXP_F32(sl, k);
    int m = (e >> 1) - k;

    // g = sqrt(s + sl), y ~ 1/sqrt(s)
    float y = BUILTIN_AMDGPU_RSQRT_F32(s);
    float g = s * y;
    g = BUILTIN_FMA_F32(BUILTIN_FMA_F32(-g, g, s) + sl, 0.5f * y, g);
    p = BUILTIN_FLDEXP_F32(g, m);
    p = o ? t : p;

    // q = b / (2*p)
    float bs2 = BUILTIN_FLDEXP_F32(b, -m - 1);
    float q = bs2 * y;
    q = BUILTIN_FMA_F32(BUILTIN_FMA_F32(-q, g, bs2), y, q);
    q = o ? (t == PINF_F32 ? 0.0f : t) : q;
#else
    int e = BUILTIN_FREXP_EXP_F32(BUILTIN_MAX_F32(a, b));
    float as = BUILTIN_FLDEXP_F32(a, -e);
    float bs = BUILTIN_FLDEXP_F32(b, -e);
    float p = MATH_FAST_SQRT(MATH_MAD(as, as, bs*bs));
    int k = (e & 1) ^ 1;
    float s = BUILTIN_FLDEXP_F32(p + as, k);
    int m = (e >> 1) - k;
    p = BUILTIN_FLDEXP_F32(MATH_FAST_SQRT(s), m);
    float q = BUILTIN_FLDEXP_F32(b, -m - 1) * BUILTIN_AMDGPU_RSQRT_F32(s);
    q = p == 0.0f ? p : q;
#endif
    bool l = z.x < 0.0f;
    float rr = l ? q : p;
    float ri = l ? p : q;

    if (!FINITE_ONLY_OPT()) {
        bool i = BUILTIN_ISINF_F32(b);
        rr = i ? b : rr;
        ri = i ? b : ri;
        ri = z.x == NINF_F32 ? a : ri;
        rr = z.x == PINF_F32 ? a : rr;
    }

    return (float2)(rr, BUILTIN_COPYSIGN_F32(ri, z.y));
}

