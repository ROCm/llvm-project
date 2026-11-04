/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

#define FLOAT_SPECIALIZATION
#include "ep.h"

CONSTATTR float2
MATH_PRIVATE(eplnep)(float2 a, int ea)
{
    // Reduce a*2^ea to m in [2/3,4/3) so |x|=|(m-1)/(m+1)| stays small.
    int ehi = BUILTIN_FREXP_EXP_F32(a.hi);
    float mh = BUILTIN_FLDEXP_F32(a.hi, -ehi);
    int e = ehi - (mh < (2.0f / 3.0f) ? 1 : 0);
    float2 m = ldx(a, -e);
    float2 x = div(fadd(-1.0f, m), fadd(1.0f, m));
    float2 s = sqr(x);
    float sh = s.hi;

    // P(s) = 2/3 + 2s/5 + ...: high terms in float (PEn), low K terms in ff.
#ifdef EXTRA_ACCURACY
    float2 p = con(PE3(sh, 0x1.a1faacp-4f, 0x1.eb4966p-4f, 0x1.10a6b4p-3f, 0x1.3b1810p-3f), 0.0f);
    p = add(mul(p, s), con( 0x1.745cfep-3f,  0x1.0f1200p-29f));
    p = add(mul(p, s), con( 0x1.c71c72p-3f,  0x1.6b2b06p-31f));
    p = add(mul(p, s), con( 0x1.24924ap-2f, -0x1.b7603cp-27f));
    p = add(mul(p, s), con( 0x1.99999ap-2f, -0x1.9998d4p-28f));
    p = add(mul(p, s), con( 0x1.555556p-1f, -0x1.555556p-26f));
#else
    float2 p = con(PE4(sh, 0x1.5e2796p-3f, 0x1.72bffap-3f, 0x1.c7251cp-3f,
                           0x1.24923ep-2f, 0x1.99999ap-2f), 0.0f);
    p = add(mul(p, s), con( 0x1.555556p-1f, -0x1.5556a6p-26f));
#endif

    // ln(m) = 2x + x*s*P(s); add e*ln2.
    const float2 ln2 = con(0x1.62e430p-1f, -0x1.05c610p-29f);
    float2 corr = mul(mul(x, s), p);
    return add(mul(ln2, (float)(e + ea)), add(ldx(x, 1), corr));
}
