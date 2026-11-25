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
MATH_MANGLE(cacosh)(float2 z)
{
    float x = BUILTIN_ABS_F32(z.x);
    float y = BUILTIN_ABS_F32(z.y);

    float2 l2, t;
    float2 tl = (float2)0.0f;
    int e = 0;
    bool b = true;

    bool big = x >= 0x1.0p+25f || y >= 0x1.0p+25f;

    // The near axis formulas below are accurate for y < 2^-13 * min(|x^2-1|, 1)
    float w = BUILTIN_MIN_F32(BUILTIN_ABS_F32(BUILTIN_FMA_F32(x, x, -1.0f)), 1.0f);

    if (!big && y > 0x1.0p-14f * w) {
        float4 z2p1 = (float4)(add(mul(add(y,x), sub(y,x)), 1.0f), mul(y,x)*2.0f);
        float4 rz2m1 = MATH_PRIVATE(epcsqrtep)(z2p1);
        rz2m1 = (float4)(csgn(rz2m1.hi, (float2)z.x), csgn(rz2m1.lo, (float2)z.y));
        float4 s = (float4)(add(rz2m1.lo, z.x), add(rz2m1.hi, z.y));
        l2 = add(sqr(s.lo), sqr(s.hi));
        t = (float2)(s.s1, z.y == 0.0f ? z.y : s.s3);
        tl = (float2)(s.s0, z.y == 0.0f ? 0.0f : s.s2);
    } else if (big) {
        e = BUILTIN_FREXP_EXP_F32(AS_FLOAT(BUILTIN_MAX_U32(AS_UINT(x), AS_UINT(y))));
        x = BUILTIN_FLDEXP_F32(x, -e);
        y = BUILTIN_FLDEXP_F32(y, -e);
        l2 = add(sqr(x), sqr(y));
        e = 2*e + 2;
        t = z;
    } else if (x < 1.0f) {
        b = false;
        float2 r = root2(fsub(1.0f, sqr(x)));
        // Scale a tiny y up so that the residual stays normal
        int kn = y < 0x1.0p-100f ? 64 : 0;
        float ns = BUILTIN_FLDEXP_F32(y, kn);
        l2 = con(BUILTIN_FLDEXP_F32(divh(ns, r), -kn), 0.0f);
        t = (float2)(z.x, BUILTIN_COPYSIGN_F32(r.hi, z.y));
        tl = (float2)(0.0f, AS_FLOAT(AS_UINT(r.lo) ^ (AS_UINT(z.y) & SIGNBIT_SP32)));
    } else {
        float2 r = root2(fsub(sqr(x), 1.0f));
        l2 = sqr(add(r, x));
        t = (float2)(BUILTIN_COPYSIGN_F32(r.hi, z.x), z.y);
        tl = (float2)(AS_FLOAT(AS_UINT(r.lo) ^ (AS_UINT(z.x) & SIGNBIT_SP32)), 0.0f);
    }

    float rr;
    if (b) {
        rr = 0.5f * MATH_PRIVATE(lnep)(l2, e);
    } else {
        rr = l2.hi;
    }

    float ri = MATH_PRIVATE(atan2ep)(con(t.y, tl.y), con(t.x, tl.x));

    if (!FINITE_ONLY_OPT()) {
        rr = (BUILTIN_ISINF_F32(z.x) | BUILTIN_ISINF_F32(z.y)) ? PINF_F32 : rr;
    }

    return (float2)(rr, ri);
}

