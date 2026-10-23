/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"
#include "trigredF.h"

#define FLOAT_SPECIALIZATION
#include "ep.h"

extern CONSTATTR float4 MATH_PRIVATE(epclog)(float2 z);
extern CONSTATTR float4 MATH_PRIVATE(epcmulep)(float4 a, float4 b);
extern CONSTATTR float2 MATH_PRIVATE(epexpep)(float2 x);
extern CONSTATTR float4 MATH_PRIVATE(epsincosep)(float2 x);

// Canonical cpow class where the ep core is invalid: a==0 or a non-finite. clog(a)
// is +-inf in ln|a| with a sign of arg(a), so the class follows the b.s0*ln|a| sign,
// poisoned to NaN only by a conflicting infinite b.s1*arg(a) term.
static float2
cpow_edge(float2 a, float2 b)
{
    bool anan = BUILTIN_ISNAN_F32(a.s0) | BUILTIN_ISNAN_F32(a.s1);
    bool ainf = BUILTIN_ISINF_F32(a.s0) | BUILTIN_ISINF_F32(a.s1);
    bool a0 = a.s0 == 0.0f && a.s1 == 0.0f;

    if (!a0 && !ainf)
        return (float2)QNAN_F32;
    if (b.s0 == 0.0f || BUILTIN_ISNAN_F32(b.s0))
        return (float2)QNAN_F32;

    bool d1neg = (AS_INT(b.s0) < 0) ^ a0;

    if (BUILTIN_ISINF_F32(b.s1)) {
        bool qzero;
        if (anan)
            qzero = true;
        else if (a.s1 == 0.0f || (BUILTIN_ISINF_F32(a.s0) && !BUILTIN_ISINF_F32(a.s1)))
            qzero = AS_INT(a.s0) >= 0;
        else
            qzero = false;
        if (!qzero) {
            bool d2neg = (AS_INT(b.s1) < 0) == (AS_INT(a.s1) < 0);
            if (d2neg != d1neg)
                return (float2)QNAN_F32;
        } else if (!BUILTIN_ISINF_F32(b.s0)) {
            return (float2)QNAN_F32;
        }
    }

    return d1neg ? (float2)0.0f : (float2)PINF_F32;
}

// Accuracy. Let w = b*clog(a) and f = frac(2*Im(w)/pi), and let A be the ep
// precision of w: abs_err(Re w), abs_err(Im w) <= |w|*2^-A, with
//   A = 35 (default),  A = 45 (EXTRA_ACCURACY).
// Max relative per-component error ~ 2^(log2|w| - log2|f| - (A-23)) ulp, hence
// <= ~2 ulp whenever log2|w| - log2|f| <= A-23 (12 default, 22 EXTRA).
// Max relative magnitude error, for |a^b| = exp(Re w), <= ~1 ulp for every
// representable result.
CONSTATTR float2
MATH_MANGLE(cpow)(float2 a, float2 b)
{
    float4 lg = MATH_PRIVATE(epclog)(a);
    float4 bp = (float4)(con(b.s0, 0.0f), con(b.s1, 0.0f));
    float4 w = MATH_PRIVATE(epcmulep)(bp, lg);
    float2 ex = MATH_PRIVATE(epexpep)(w.lo);
    float4 cs = MATH_PRIVATE(epsincosep)(w.hi);
    float2 ret = (float2)(mul(ex, cs.lo).hi, mul(ex, cs.hi).hi);

    if (!FINITE_ONLY_OPT()) {
        if ((a.s0 == 0.0f && a.s1 == 0.0f) || !BUILTIN_ISFINITE_F32(a.s0) || !BUILTIN_ISFINITE_F32(a.s1))
            ret = cpow_edge(a, b);
    }

    return ret;
}
