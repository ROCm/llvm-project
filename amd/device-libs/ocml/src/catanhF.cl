/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

#define FLOAT_SPECIALIZATION
#include "ep.h"

extern CONSTATTR float MATH_PRIVATE(lnep)(float2 a, int ea);
extern CONSTATTR float MATH_PRIVATE(atan2ep)(float2 y, float2 x);

CONSTATTR float2
MATH_MANGLE(catanh)(float2 z)
{
    float x = BUILTIN_ABS_F32(z.x);
    float y = BUILTIN_ABS_F32(z.y);
    float rr, ri;

    if (x < 0x1.0p+25f && y < 0x1.0p+25f) {
        float2 omx = sub(1.0f, x);
        float2 opx = add(1.0f, x);
        float2 y2 = sqr(y);
        float2 b = sub(mul(omx, opx), y2);
        ri = 0.5f * MATH_PRIVATE(atan2ep)(con(2.0f * y, 0.0f), b);

        float2 a;
        int ea = 0;
        float2 d = add(sqr(opx), y2);
        float r2 = BUILTIN_MAX_F32(BUILTIN_ABS_F32(omx.hi), y);
        if (x < 0x1.0p-3f * d.hi) {
            a = fsub(1.0f, div(4.0f*x, d));
        } else if (r2 < 0x1.0p-60f) {
            // avoid underflow of (1-x)^2 + y^2
            a = div(add(sqr(ldx(omx, 100)), sqr(BUILTIN_FLDEXP_F32(y, 100))), d);
            ea = -200;
        } else {
            a = div(add(sqr(omx), y2), d);
        }
        rr = -0.25f * MATH_PRIVATE(lnep)(a, ea);

        if (x < 0x1.0p-40f) {
            // -ln(1 - 4x/d)/4 = x/d to working precision; scale x up so that
            // no intermediate is denormal
            rr = BUILTIN_FLDEXP_F32(div(BUILTIN_FLDEXP_F32(x, 64), d).hi, -64);
        }
    } else {
        int e = BUILTIN_FREXP_EXP_F32(AS_FLOAT(BUILTIN_MAX_U32(AS_UINT(x), AS_UINT(y))));
        x = BUILTIN_FLDEXP_F32(x, -e);
        y = BUILTIN_FLDEXP_F32(y, -e);
        // x / (x^2 + y^2) including the rounding error of y*y
        float y2 = y * y;
        float d = BUILTIN_FMA_F32(x, x, y2);
        rr = BUILTIN_FLDEXP_F32(divh(x, con(d, BUILTIN_FMA_F32(y, y, -y2))), -e);
        ri = 0x1.921fb6p+0f;
    }

    if (!FINITE_ONLY_OPT()) {
        rr = ((x == 1.0f) & (y == 0.0f)) ? PINF_F32  : rr;
        rr = x == 0.0f ? 0.0f : rr;
        rr = (BUILTIN_ISINF_F32(x) | BUILTIN_ISINF_F32(y)) ? 0.0f : rr;
        rr = (BUILTIN_ISNAN_F32(x) & BUILTIN_ISINF_F32(y)) ? 0.0f : rr;
        ri = (BUILTIN_ISNAN_F32(x) & BUILTIN_ISFINITE_F32(y)) ? QNAN_F32 : ri;
        ri = BUILTIN_ISNAN_F32(y) ? y : ri;
    }

    rr = BUILTIN_COPYSIGN_F32(rr, z.x);
    ri = BUILTIN_COPYSIGN_F32(ri, z.y);

    return (float2)(rr, ri);
}

