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
MATH_MANGLE(i0)(float x)
{
    x = BUILTIN_ABS_F32(x);

    float ret;

    if (x < 8.0f) {
        float h = 0.5f * x;
        float2 t = sqr(h);
        float th = t.hi;
        float ph = MATH_MAD(th, MATH_MAD(th, MATH_MAD(th, MATH_MAD(th, MATH_MAD(th,
                       0x1.38d760p-43f, 0x1.7fd5c6p-38f), 0x1.66ffc8p-31f), 0x1.4ecb6ep-25f),
                       0x1.033c70p-19f), 0x1.233bb2p-14f);
        float2 p = con(ph, 0.0f);
        p = fadd(mul(p, t), 0x1.c71db2p-10f);
        p = fadd(mul(p, t), 0x1.c71c5ep-6f);
        p = fadd(mul(p, t), 0x1.000000p-2f);
        p = fadd(mul(p, t), 0x1.000000p+0f);
        ret = fadd(mul(t, p), 1.0f).hi;
    } else {
        float t = MATH_FAST_RCP(x);
        float ph = MATH_MAD(t, MATH_MAD(t, 0x1.c49916p-2f, -0x1.110f5ep-5f), 0x1.2a130ap-5f);
        float2 p = con(ph, 0.0f);
        p = fadd(mul(p, t), 0x1.c68702p-6f);
        p = fadd(mul(p, t), 0x1.9890aep-5f);
        p = fadd(mul(p, t), 0x1.988450p-2f);
        float xs = x - 88.0f;
        float e1 = MATH_MANGLE(exp)(x > 88.0f ? xs : x);
        float e2 = x > 88.0f ? 0x1.f1056ep+126f : 1.0f;
        ret = omul(omul(mul(p, BUILTIN_AMDGPU_RSQRT_F32(x)), e1), e2).hi;
    }

    if  (!FINITE_ONLY_OPT()) {
        ret = BUILTIN_CLASS_F32(x, CLASS_PINF|CLASS_QNAN|CLASS_SNAN) ? x : ret;
    }

    return ret;
}

