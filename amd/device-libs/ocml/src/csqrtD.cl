/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

#if !defined EXTRA_ACCURACY
#define EXTRA_ACCURACY
#endif

CONSTATTR double2
MATH_MANGLE(csqrt)(double2 z)
{
    double a = BUILTIN_ABS_F64(z.x);
    double b = BUILTIN_ABS_F64(z.y);
    double t = BUILTIN_MAX_F64(a, b);

    if (!FINITE_ONLY_OPT()) {
        t = BUILTIN_ISUNORDERED_F64(a, b) ? QNAN_F64 : t;
    }

    int e = BUILTIN_FREXP_EXP_F64(t);
    double as = BUILTIN_FLDEXP_F64(a, -e);
    double bs = BUILTIN_FLDEXP_F64(b, -e);
    bool o = BUILTIN_CLASS_F64(t, CLASS_NZER|CLASS_PZER|CLASS_NINF|CLASS_PINF|CLASS_QNAN|CLASS_SNAN);

#if defined EXTRA_ACCURACY
    // |z| scaled, with the error of its final rounding in pl
    double n = MATH_MAD(as, as, bs*bs);
    double y1 = BUILTIN_AMDGPU_RSQRT_F64(n);
    double p = n * y1;
    double h1 = 0.5 * y1;
    double r1 = BUILTIN_FMA_F64(-h1, p, 0.5);
    h1 = BUILTIN_FMA_F64(h1, r1, h1);
    p = BUILTIN_FMA_F64(p, r1, p);
    p = BUILTIN_FMA_F64(BUILTIN_FMA_F64(-p, p, n), h1, p);
    double pl = BUILTIN_FMA_F64(-p, p, n) * h1;
    p = o ? t : p;

    // s + sl = |z| + |x|
    int k = (e & 1) ^ 1;
    double sh = p + as;
    double sl = (as - (sh - p)) + pl;
    double s = BUILTIN_FLDEXP_F64(sh, k);
    sl = BUILTIN_FLDEXP_F64(sl, k);
#else
    double p = MATH_FAST_SQRT(MATH_MAD(as, as, bs*bs));
    p = o ? t : p;
    int k = (e & 1) ^ 1;
    double s = BUILTIN_FLDEXP_F64(p + as, k);
    const double sl = -0.0;
#endif
    int m = (e >> 1) - k;

    // g = sqrt(s) and u = 1/sqrt(s)
    double y = BUILTIN_AMDGPU_RSQRT_F64(s);
    double g = s * y;
    double h = 0.5 * y;
    double r = BUILTIN_FMA_F64(-h, g, 0.5);
    h = BUILTIN_FMA_F64(h, r, h);
    g = BUILTIN_FMA_F64(g, r, g);
    g = BUILTIN_FMA_F64(BUILTIN_FMA_F64(-g, g, s) + sl, h, g);
    double u = h + h;

    p = BUILTIN_FLDEXP_F64(g, m);
    p = o ? t : p;

    // q = b / (2*p)
    double bs2 = BUILTIN_FLDEXP_F64(b, -m - 1);
    double q = bs2 * u;
    q = BUILTIN_FMA_F64(BUILTIN_FMA_F64(-q, g, bs2), u, q);
    q = o ? (t == PINF_F64 ? 0.0 : t) : q;
    bool l = z.x < 0.0;
    double rr = l ? q : p;
    double ri = l ? p : q;

    if (!FINITE_ONLY_OPT()) {
        bool i = BUILTIN_ISINF_F64(b);
        rr = i ? b : rr;
        ri = i ? b : ri;
        ri = z.x == NINF_F64 ? a : ri;
        rr = z.x == PINF_F64 ? a : rr;
    }

    return (double2)(rr, BUILTIN_COPYSIGN_F64(ri, z.y));
}

