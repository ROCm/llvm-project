/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

#define FLOAT_SPECIALIZATION
#include "ep.h"

extern CONSTATTR float2 MATH_PRIVATE(epexpep)(float2 x);

float
MATH_MANGLE(i0)(float x)
{
    x = BUILTIN_ABS_F32(x);

    float ret;

    if (x < 8.0f) {
        float h = 0.5f * x;
        float2 t = sqr(h);
        float th = t.hi;
        float ph = PE5(th, 0x1.38d760p-43f, 0x1.7fd5c6p-38f, 0x1.66ffc8p-31f, 0x1.4ecb6ep-25f, 0x1.033c70p-19f,
                         0x1.233bb2p-14f);
        float2 p = con(ph, 0.0f);
        p = fadd(mul(p, t), 0x1.c71db2p-10f);
        p = fadd(mul(p, t), 0x1.c71c5ep-6f);
        p = fadd(mul(p, t), 0x1.000000p-2f);
        p = fadd(mul(p, t), 0x1.000000p+0f);
        ret = fadd(mul(t, p), 1.0f).hi;
    } else {
        float t = MATH_FAST_RCP(x);
        float ph = PE2(t, 0x1.c49916p-2f, -0x1.110f5ep-5f, 0x1.2a130ap-5f);
        float2 p = con(ph, 0.0f);
        p = fadd(mul(p, t), 0x1.c68702p-6f);
        p = fadd(mul(p, t), 0x1.9890aep-5f);
        p = fadd(mul(p, t), 0x1.988450p-2f);
        // 1/sqrt(x) = r + rl
        float r = BUILTIN_AMDGPU_RSQRT_F32(x);
        float rt = x * r;
        float e = BUILTIN_FMA_F32(-BUILTIN_FMA_F32(x, r, -rt), r, BUILTIN_FMA_F32(-rt, r, 1.0f));
        float2 rr = fadd(r, 0.5f * r * e);

        // e^x * 2^-64 as a double-word
        float2 ex = MATH_PRIVATE(epexpep)(sub(BUILTIN_MIN_F32(x, 0x1.0p+7f), con(0x1.62e430p+5f, -0x1.05c610p-23f)));
        ret = BUILTIN_FLDEXP_F32(mul(mul(p, rr), ex).hi, 64);
        ret = x > 0x1.0p+7f ? PINF_F32 : ret;
    }

    if  (!FINITE_ONLY_OPT()) {
        ret = BUILTIN_CLASS_F32(x, CLASS_PINF|CLASS_QNAN|CLASS_SNAN) ? x : ret;
    }

    return ret;
}

