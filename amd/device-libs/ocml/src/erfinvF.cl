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
MATH_MANGLE(erfinv)(float x)
{
    float ax = BUILTIN_ABS_F32(x);
    float t, q, c0h, c0l;
#if defined EXTRA_ACCURACY
    float tl = 0.0f;
#endif

    if (ax < 0.375f) {
        t = ax*ax;
        q = PE4(t, 0x1.32a0eep-4f, 0x1.f3bb74p-5f, 0x1.63bf6ep-4f, 0x1.0536dap-3f, 0x1.db2a0cp-3f);
        c0h = 0x1.c5bf8ap-1f;
        c0l = -0x1.cf1306p-26f;
    } else {
#if defined EXTRA_ACCURACY
        // w + wl = -log(1 - ax^2) with 1 - ax^2 = uh + ul exact
        float xx = ax * ax;
        float uh = 1.0f - xx;
        float ul = ((1.0f - uh) - xx) - BUILTIN_FMA_F32(ax, ax, -xx);
        float2 lu = MATH_PRIVATE(epln)(uh);
        float wl = -BUILTIN_FMA_F32(ul, BUILTIN_AMDGPU_RCP_F32(uh), lu.lo);
        float w = wl - lu.hi;
        wl = wl - (w + lu.hi);
#else
        float w = -MATH_MANGLE(log)(BUILTIN_FMA_F32(-ax, ax, 1.0f));
#endif

        if (w < 5.0f) {
            t = w - 2.5f;
#if defined EXTRA_ACCURACY
            float tb = t - w;
            tl = ((w - (t - tb)) + (-2.5f - tb)) + wl;
#endif
            q = PE8(t, -0x1.282984p-27f, 0x1.8a4eaep-26f, 0x1.f272ecp-22f, -0x1.d060c8p-19f, -0x1.4c986cp-18f,
                       0x1.c9e226p-13f, -0x1.4866cp-10f, -0x1.11c4c8p-8f, 0x1.f91e8cp-3f);
            c0h = 0x1.805c5ep+0f;
            c0l = -0x1.aad56cp-25f;
        } else {
#if defined EXTRA_ACCURACY
            float s = MATH_SQRT(w);
            t = (s - 3.0f) + (BUILTIN_FMA_F32(-s, s, w) + wl) * (0.5f * BUILTIN_AMDGPU_RSQRT_F32(w));
#else
            t = MATH_SQRT(w) - 3.0f;
#endif
            q = PE9(t, -0x1.bc1a2p-16f, 0x1.4d2574p-13f, -0x1.2894bap-12f, -0x1.ee74fep-14f, 0x1.839682p-10f,
                       -0x1.d3a7eep-9f, 0x1.74b06cp-8f, -0x1.f4de9cp-8f, 0x1.3581aap-7f, 0x1.006dcap+0f);
            c0h = 0x1.6a9efcp+1f;
            c0l = -0x1.158234p-26f;
        }
    }

    // ph + pl = c0 + t*q, then (ph + pl)*ax with one rounding
    float ph = BUILTIN_FMA_F32(t, q, c0h);
#if defined EXTRA_ACCURACY
    float pl = BUILTIN_FMA_F32(tl, q, BUILTIN_FMA_F32(t, q, c0h - ph) + c0l);
#else
    float pl = BUILTIN_FMA_F32(t, q, c0h - ph) + c0l;
#endif
    float ret = BUILTIN_FMA_F32(ph, ax, pl * ax);

    if (!FINITE_ONLY_OPT()) {
        ret = ax > 1.0f ? QNAN_F32 : ret;
        ret = ax == 1.0f ? PINF_F32 : ret;
    }

    return BUILTIN_COPYSIGN_F32(ret, x);
}

