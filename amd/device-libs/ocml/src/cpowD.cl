/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"
#include "trigredD.h"

#define DOUBLE_SPECIALIZATION
#include "ep.h"

extern CONSTATTR double4 MATH_PRIVATE(epclog)(double2 z);
extern CONSTATTR double4 MATH_PRIVATE(epcmulep)(double4 a, double4 b);
extern CONSTATTR double2 MATH_PRIVATE(epexpep)(double2 x);
extern CONSTATTR double4 MATH_PRIVATE(epsincosep)(double2 x);

// Canonical cpow class where the ep core is invalid: a==0 or a non-finite. clog(a)
// is +-inf in ln|a| with a sign of arg(a), so the class follows the b.s0*ln|a| sign,
// poisoned to NaN only by a conflicting infinite b.s1*arg(a) term.
static double2
cpow_edge(double2 a, double2 b)
{
    bool anan = BUILTIN_ISNAN_F64(a.s0) | BUILTIN_ISNAN_F64(a.s1);
    bool ainf = BUILTIN_ISINF_F64(a.s0) | BUILTIN_ISINF_F64(a.s1);
    bool a0 = a.s0 == 0.0 && a.s1 == 0.0;

    if (!a0 && !ainf)
        return (double2)QNAN_F64;
    if (b.s0 == 0.0 || BUILTIN_ISNAN_F64(b.s0))
        return (double2)QNAN_F64;

    bool d1neg = (AS_INT2(b.s0).hi < 0) ^ a0;

    if (BUILTIN_ISINF_F64(b.s1)) {
        bool qzero;
        if (anan)
            qzero = true;
        else if (a.s1 == 0.0 || (BUILTIN_ISINF_F64(a.s0) && !BUILTIN_ISINF_F64(a.s1)))
            qzero = AS_INT2(a.s0).hi >= 0;
        else
            qzero = false;
        if (!qzero) {
            bool d2neg = (AS_INT2(b.s1).hi < 0) == (AS_INT2(a.s1).hi < 0);
            if (d2neg != d1neg)
                return (double2)QNAN_F64;
        } else if (!BUILTIN_ISINF_F64(b.s0)) {
            return (double2)QNAN_F64;
        }
    }

    return d1neg ? (double2)0.0 : (double2)PINF_F64;
}

// Accuracy. Let w = b*clog(a) and f = frac(2*Im(w)/pi), and let A be the ep
// precision of w: abs_err(Re w), abs_err(Im w) <= |w|*2^-A, with
//   A = 69 (default),  A = 102 (EXTRA_ACCURACY).
// Max relative per-component error ~ 2^(log2|w| - log2|f| - (A-52)) ulp, hence
// <= ~2 ulp whenever log2|w| - log2|f| <= A-52 (17 default, 50 EXTRA).
// Max relative magnitude error, for |a^b| = exp(Re w), <= ~1 ulp for every
// representable result.
CONSTATTR double2
MATH_MANGLE(cpow)(double2 a, double2 b)
{
    double4 lg = MATH_PRIVATE(epclog)(a);
    double4 bp = (double4)(con(b.s0, 0.0), con(b.s1, 0.0));
    double4 w = MATH_PRIVATE(epcmulep)(bp, lg);
    double2 ex = MATH_PRIVATE(epexpep)(w.lo);
    double4 cs = MATH_PRIVATE(epsincosep)(w.hi);
    double2 ret = (double2)(mul(ex, cs.lo).hi, mul(ex, cs.hi).hi);

    if (!FINITE_ONLY_OPT()) {
        if ((a.s0 == 0.0 && a.s1 == 0.0) || !BUILTIN_ISFINITE_F64(a.s0) || !BUILTIN_ISFINITE_F64(a.s1))
            ret = cpow_edge(a, b);
    }

    return ret;
}
