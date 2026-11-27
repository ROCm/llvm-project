/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR float
MATH_MANGLE(acospi)(float x)
{
    const float piinv = 0x1.45f306p-2f;

    float ax = BUILTIN_ABS_F32(x);

    float rt = MATH_MAD(-0.5f, ax, 0.5f);
    float x2 = ax * ax;
    float r = ax > 0.5f ? rt : x2;

    float u = PE6(r, 0x1.8b01e4p-7f, 0x1.27336p-8f, 0x1.4cb3aap-7f, 0x1.d03062p-7f, 0x1.872ba8p-6f,
                     0x1.b2994ep-5f, 0x1.bbf85p-27f);

    // s + sl = sqrt(r)
    float s = MATH_FAST_SQRT(r);
    float sl = BUILTIN_FMA_F32(-s, s, r) * (0.5f * BUILTIN_AMDGPU_RSQRT_F32(r));
    sl = r == 0.0f ? 0.0f : sl;
#if defined EXTRA_ACCURACY
    // p + pl = s * (1/pi + u)
    float p = s * piinv;
    float pl = BUILTIN_FMA_F32(s, piinv, -p);
    pl = BUILTIN_FMA_F32(sl, piinv, pl);
    pl = BUILTIN_FMA_F32(s, u, pl);
    float ztp = 2.0f * (p + pl);
    float d = BUILTIN_FMA_F32(-2.0f, p, 1.0f);
    float ztn = d + BUILTIN_FMA_F32(-2.0f, pl, BUILTIN_FMA_F32(-2.0f, p, 1.0f - d));
#else
    float ztp = 2.0f * MATH_MAD(s, u, MATH_MAD(sl, piinv, piinv*s));
    float ztn = 1.0f - ztp;
#endif
    float zt =  x < 0.0f ? ztn : ztp;
    float z = BUILTIN_FMA_F32(-x, piinv, 0.5f);
    z = z + BUILTIN_FMA_F32(-x, u, BUILTIN_FMA_F32(-x, piinv, 0.5f - z));
    z = ax > 0.5f ? zt : z;

    return z;
}

