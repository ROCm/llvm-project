/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

#define FLOAT_SPECIALIZATION
#include "ep.h"

float
MATH_MANGLE(i1)(float x)
{
    float a = BUILTIN_ABS_F32(x);

    float ret;

    if (a < 8.0f) {
        a *= 0.5f;
        float2 t = sqr(a);
        float th = t.hi;
        float ph = MATH_MAD(th, MATH_MAD(th, MATH_MAD(th, MATH_MAD(th, MATH_MAD(th,
                       0x1.882dd2p-40f, 0x1.af97f6p-35f), 0x1.66a3eap-28f), 0x1.251b32p-22f),
                       0x1.84cbb6p-17f), 0x1.6c0d4ap-12f);
        float2 p = con(ph, 0.0f);
        p = fadd(mul(p, t), 0x1.c71d3ap-8f);
        p = fadd(mul(p, t), 0x1.555550p-4f);
        p = fadd(mul(p, t), 0x1.000000p-1f);
        ret = mul(a, fadd(mul(t, p), 1.0f)).hi;
    } else {
        float t = MATH_FAST_RCP(a);
        float ph = MATH_MAD(t, MATH_MAD(t, -0x1.06de32p-1f, 0x1.043b22p-5f), -0x1.925276p-5f);
        float2 p = con(ph, 0.0f);
        p = fadd(mul(p, t), -0x1.7c15c8p-5f);
        p = fadd(mul(p, t), -0x1.3266ccp-3f);
        p = fadd(mul(p, t), 0x1.988456p-2f);

        float as = a - 88.0f;
        float e1 = MATH_MANGLE(exp)(a > 88.0f ? as : a);
        float e2 = a > 88.0f ? 0x1.f1056ep+126f : 1.0f;
        ret = omul(omul(mul(p, BUILTIN_AMDGPU_RSQRT_F32(a)), e1), e2).hi;
    }

    if  (!FINITE_ONLY_OPT()) {
        ret = BUILTIN_CLASS_F32(a, CLASS_PINF|CLASS_QNAN|CLASS_SNAN) ? a : ret;
    }

    return BUILTIN_COPYSIGN_F32(ret, x);
}

