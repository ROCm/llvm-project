/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

#if defined EXTRA_ACCURACY
extern CONSTATTR float2 MATH_PRIVATE(epln)(float);
#endif

CONSTATTR float
MATH_MANGLE(erfcinv)(float y)
{
    float ret;

    if (y > 0.625f) {
        ret = MATH_MANGLE(erfinv)(1.0f - y);
    } else if (y > 0x1.0p-10f) {
#if defined EXTRA_ACCURACY
        // w + wl = -log(y*(2 - y)) with y*(2 - y) = uh + ul exact
        float uh = BUILTIN_FMA_F32(-y, y, 2.0f * y);
        float ul = BUILTIN_FMA_F32(-y, y, 2.0f * y - uh);
        float2 lu = MATH_PRIVATE(epln)(uh);
        float wl = -BUILTIN_FMA_F32(ul, BUILTIN_AMDGPU_RCP_F32(uh), lu.lo);
        float w = wl - lu.hi;
        wl = wl - (w + lu.hi);
        float t = w - 3.125f;
        float tb = t - w;
        float tl = ((w - (t - tb)) + (-3.125f - tb)) + wl;
#else
        float t = -MATH_MANGLE(log)(BUILTIN_FMA_F32(-y, y, 2.0f * y)) - 3.125f;
#endif
        float q = PE9(t, 0x1.764d8ap-31f, -0x1.3f3c16p-28f, -0x1.b09468p-26f, 0x1.c9cc84p-22f, -0x1.739cacp-20f,
                         -0x1.d21c4p-17f, 0x1.87b278p-13f, -0x1.8455b6p-11f, -0x1.8b6cc2p-8f, 0x1.ebd80cp-3f);
        const float c0h = 0x1.a755e8p+0f;
        const float c0l = -0x1.1c30a2p-26f;
        float ph = BUILTIN_FMA_F32(t, q, c0h);
#if defined EXTRA_ACCURACY
        float pl = BUILTIN_FMA_F32(tl, q, BUILTIN_FMA_F32(t, q, c0h - ph) + c0l);
#else
        float pl = BUILTIN_FMA_F32(t, q, c0h - ph) + c0l;
#endif

        // (ph + pl) * (1 - y) with 1 - y = x + xl exact
        float x = 1.0f - y;
        float xl = (1.0f - x) - y;
        ret = BUILTIN_FMA_F32(ph, x, BUILTIN_FMA_F32(ph, xl, pl * x));
    } else {
        // s + sl = sqrt(-log(y))
#if defined EXTRA_ACCURACY
        float2 lv = MATH_PRIVATE(epln)(y);
        float l = -lv.hi;
#else
        float l = -MATH_MANGLE(log)(y);
#endif
        float s = MATH_FAST_SQRT(l);
        float r = MATH_FAST_RCP(s);
#if defined EXTRA_ACCURACY
        float sl = (BUILTIN_FMA_F32(-s, s, l) - lv.lo) * (0.5f * r);
#else
        float sl = BUILTIN_FMA_F32(-s, s, l) * (0.5f * r);
#endif

        float t, q, c0h, c0l;
        if (y > 0x1.0p-42f) {
            t = r - 0.28125f;
            q = PE5(t, -0x1.572342p+0f, 0x1.777332p-1f, -0x1.253756p-1f, 0x1.0e0da4p-1f, -0x1.c46b16p-3f,
                    -0x1.90dfbep-2f);
            c0h = 0x1.da0782p-1f;
            c0l = -0x1.6b1e5p-28f;
        } else {
            t = r - 0.140625f;
            q = PE5(t, -0x1.64a836p+4f, 0x1.8b136cp+2f, -0x1.0a75dap+1f, 0x1.20d5c8p+0f, -0x1.16280ep-1f,
                    -0x1.28a3f6p-2f);
            c0h = 0x1.f3162cp-1f;
            c0l = -0x1.088d7cp-27f;
        }
        float ph = BUILTIN_FMA_F32(t, q, c0h);
        float pl = BUILTIN_FMA_F32(t, q, c0h - ph) + c0l;
        ret = BUILTIN_FMA_F32(s, ph, BUILTIN_FMA_F32(s, pl, sl * ph));
    }

    if (!FINITE_ONLY_OPT()) {
        ret = ((y < 0.0f) | (y > 2.0f)) ? QNAN_F32 : ret;
        ret = y == 0.0f ? PINF_F32 : ret;
        ret = y == 2.0f ? NINF_F32 : ret;
    }

    return ret;
}

