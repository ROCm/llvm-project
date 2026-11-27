/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

#define FLOAT_SPECIALIZATION
#include "ep.h"

extern CONSTATTR float4 MATH_PRIVATE(epcsqrtep)(float4 z);
extern CONSTATTR float MATH_PRIVATE(lnep)(float2 a, int ea);
extern CONSTATTR float MATH_PRIVATE(atan2ep)(float2 y, float2 x);

CONSTATTR float2
MATH_MANGLE(casinh)(float2 z)
{
    float x = BUILTIN_ABS_F32(z.x);
    float y = BUILTIN_ABS_F32(z.y);

    float2 l2, t;
    float2 tl = (float2)0.0f;
    int e = 0;
    int kt = 0;
    bool b = true;

    bool big = x >= 0x1.0p+25f || y >= 0x1.0p+25f;

    // The near axis formula below is accurate for x < 2^-13 * (1 - y^2)
    if (!big && (y >= 1.0f || x > 0x1.0p-14f * BUILTIN_FMA_F32(-y, y, 1.0f))) {
        // For tiny y the imaginary part is linear in y and y^2 is negligible,
        // so scale y up to keep the intermediates normal
        kt = y < 0x1.0p-100f ? 64 : 0;
        float ys = BUILTIN_FLDEXP_F32(y, kt);
        float4 z2p1 = (float4)(add(mul(add(x,ys), sub(x,ys)), 1.0f), mul(ys,x)*2.0f);
        float4 rz2p1 = MATH_PRIVATE(epcsqrtep)(z2p1);
        float4 s = (float4)(add(rz2p1.lo, x), add(rz2p1.hi, ys));
        l2 = add(sqr(s.lo), sqr(s.hi));
        t = (float2)(s.s1, s.s3);
        tl = (float2)(s.s0, s.s2);
    } else if (big) {
        t = (float2)(x, y);
        e = BUILTIN_FREXP_EXP_F32(AS_FLOAT(BUILTIN_MAX_U32(AS_UINT(x), AS_UINT(y))));
        x = BUILTIN_FLDEXP_F32(x, -e);
        y = BUILTIN_FLDEXP_F32(y, -e);
        l2 = add(sqr(x), sqr(y));
        e = 2*e + 2;
    } else {
        b = false;
        float2 r = root2(fsub(1.0f, sqr(y)));
        // Scale a tiny x up so that the residual stays normal
        int kn = x < 0x1.0p-100f ? 64 : 0;
        float ns = BUILTIN_FLDEXP_F32(x, kn);
        l2 = con(BUILTIN_FLDEXP_F32(divh(ns, r), -kn), 0.0f);
        t = (float2)(r.hi, y);
        tl = (float2)(r.lo, 0.0f);
    }

    float rr;
    if (b) {
        rr = 0.5f * MATH_PRIVATE(lnep)(l2, e);
    } else {
        rr = l2.hi;
    }

    rr = BUILTIN_COPYSIGN_F32(rr, z.x);
    float ri = BUILTIN_COPYSIGN_F32(BUILTIN_FLDEXP_F32(MATH_PRIVATE(atan2ep)(con(t.y, tl.y), con(t.x, tl.x)), -kt), z.y);

    if (!FINITE_ONLY_OPT()) {
        float i = BUILTIN_COPYSIGN_F32(PINF_F32, z.x);
        rr = (BUILTIN_ISINF_F32(z.x) | BUILTIN_ISINF_F32(z.y)) ? i : rr;
        ri = (z.y == 0.0f) ? BUILTIN_COPYSIGN_F32(0.0f, z.y) : ri;
    }

    return (float2)(rr, ri);
}

