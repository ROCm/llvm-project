/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

#define FLOAT_SPECIALIZATION
#include "ep.h"

extern CONSTATTR float2 MATH_PRIVATE(eplnep)(float2, int);
extern CONSTATTR float2 MATH_PRIVATE(epatan2)(float, float);

CONSTATTR float4
MATH_PRIVATE(epclog)(float2 z)
{
    float x = z.s0;
    float y = z.s1;
    float a = BUILTIN_ABS_F32(x);
    float b = BUILTIN_ABS_F32(y);
    float t = BUILTIN_MAX_F32(a, b);
    int e = BUILTIN_FREXP_EXP_F32(t);

    // Near |z|=1, form m = a^2+b^2-1 exactly to keep the small deviation.
    float2 m;
    bool near1 = false;
    if (e == 0 || e == 1) {
        float2 sx = sqr(a);
        float2 sy = sqr(b);
        float2 p1 = add(sx.hi, sy.hi);
        float2 p2 = add(p1.hi, -1.0f);
        m = add(p2.hi, p2.lo);
        m = add(m, p1.lo);
        m = add(m, sx.lo);
        m = add(m, sy.lo);
        near1 = m.hi >= -(1.0f / 3.0f) && m.hi <= 0.5f;
    }

    float2 rr;
    if (near1) {
        float2 xx = div(m, fadd(2.0f, m));
        float2 s = sqr(xx);
        float sh = s.hi;
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
        float2 corr = mul(mul(xx, s), p);
        rr = ldx(add(ldx(xx, 1), corr), -1);
    } else {
        float as = BUILTIN_FLDEXP_F32(a, -e);
        float bs = BUILTIN_FLDEXP_F32(b, -e);
        rr = ldx(MATH_PRIVATE(eplnep)(add(sqr(as), sqr(bs)), 2 * e), -1);
    }

    float2 ri = MATH_PRIVATE(epatan2)(y, x);
    return (float4)(rr, ri);
}
