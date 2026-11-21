/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

#define FLOAT_SPECIALIZATION
#include "ep.h"

#include "trigredF.h"

extern CONSTATTR float2 MATH_PRIVATE(epexpep)(float2 z);

CONSTATTR float2
MATH_MANGLE(ctanh)(float2 z)
{
    float4 sc = MATH_PRIVATE(epsincos)(z.y);
    float2 cy = sc.lo;
    float2 sy = sc.hi;
#if defined EXTRA_ACCURACY
    // Keep low parts of products with sin(y) from underflowing for tiny y
    int ky = BUILTIN_ABS_F32(z.y) < 0x1.0p-60f ? 64 : 0;
    sy = ldx(sy, ky);
#endif
    float x = BUILTIN_ABS_F32(z.x);

    float rr, ri;
    if (x < 0x1.62e430p+3f) {
        float2 e = MATH_PRIVATE(epexpep)(sub(x, con(0x1.62e430p-1, -0x1.05c610p-29f)));
        float2 er = rcp(e);
        er = ldx(er, -2);
        float2 cx = fadd(e, er);
        float2 sx = x < 0x1.0p-12f ? con(x, 0.0f) : fsub(e, er);

        float2 d = add(sqr(cy), sqr(sx));
        float2 rd = rcp(d);
#if defined EXTRA_ACCURACY
        // Same for sinh(x) with tiny x
        int kx = x < 0x1.0p-60f ? 64 : 0;
        rr = BUILTIN_COPYSIGN_F32(BUILTIN_FLDEXP_F32(mul(mul(cx, ldx(sx, kx)), rd).hi, -kx), z.x);
#else
        rr = BUILTIN_COPYSIGN_F32(mul(mul(cx, sx), rd).hi, z.x);
#endif
        ri = mul(mul(cy, sy), rd).hi;
#if defined EXTRA_ACCURACY
        ri = BUILTIN_FLDEXP_F32(ri, -ky);
#endif
    } else {
        rr = BUILTIN_COPYSIGN_F32(1.0f, z.x);
        float2 p = mul(cy.hi, sy.hi);
        p.lo = BUILTIN_FMA_F32(cy.lo, sy.hi, BUILTIN_FMA_F32(cy.hi, sy.lo, p.lo));
#if defined EXTRA_ACCURACY
        float xc = x > 0x1.0p+7f ? 0x1.0p+7f : x;
        // 2^64 * e^(-2x) as a double-word, so its low part doesn't underflow
        float2 ex = MATH_PRIVATE(epexpep)(sub(-2.0f * xc, con(-0x1.62e430p+5f, 0x1.05c610p-23f)));
        ri = BUILTIN_FLDEXP_F32(BUILTIN_FMA_F32(p.hi, ex.hi, BUILTIN_FMA_F32(p.lo, ex.hi, p.hi * ex.lo)), -62 - ky);
        ri = BUILTIN_COPYSIGN_F32(ri, p.hi);
#else
        float ex = MATH_MANGLE(exp)(-2.0f * x);
        ri = BUILTIN_FMA_F32(4.0f * p.hi, ex, 4.0f * p.lo * ex);
#endif
    }

    if (!FINITE_ONLY_OPT()) {
        bool xn = BUILTIN_ISNAN_F32(x);
        bool yin = !BUILTIN_ISFINITE_F32(z.y);
        bool ni = BUILTIN_CLASS_F32(x, CLASS_PZER|CLASS_PSUB|CLASS_PNOR) & yin;
        rr = (ni | xn) ? QNAN_F32 : rr;
        ri = ni ? QNAN_F32 : ri;
        ri = (BUILTIN_ISINF_F32(x) & yin) ? 0.0f : ri;
        ri = (xn & (z.y == 0.0f)) ? z.y : ri;
    }

    return (float2)(rr, ri);
}

