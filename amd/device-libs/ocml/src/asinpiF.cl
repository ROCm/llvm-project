/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR float
MATH_MANGLE(asinpi)(float x)
{
    // Computes arcsin(x).
    // The argument is first reduced by noting that arcsin(x)
    // is invalid for abs(x) > 1 and arcsin(-x) = -arcsin(x).
    // For denormal and small arguments arcsin(x) = x to machine
    // accuracy. Remaining argument ranges are handled as follows.
    // For abs(x) <= 0.5 use
    // arcsin(x) = x + x^3*R(x^2)
    // where R(x^2) is a polynomial minimax approximation to
    // (arcsin(x) - x)/x^3.
    // For abs(x) > 0.5 exploit the identity:
    // arcsin(x) = pi/2 - 2*arcsin(sqrt(1-x)/2)
    // together with the above polynomial approximation, and
    // reconstruct the terms carefully.

    const float piinv = 0x1.45f306p-2f;

    float ax = BUILTIN_ABS_F32(x);

    float tx = MATH_MAD(ax, -0.5f, 0.5f);
    float x2 = ax * ax;
    float r = ax >= 0.5f ? tx : x2;

    float u = PE6(r, 0x1.8b01e4p-7f, 0x1.27336p-8f, 0x1.4cb3aap-7f, 0x1.d03062p-7f, 0x1.872ba8p-6f,
                     0x1.b2994ep-5f, 0x1.bbf85p-27f);

    float s = MATH_FAST_SQRT(r);
#if defined EXTRA_ACCURACY
    // s + sl = sqrt(r)
    float sl = BUILTIN_FMA_F32(-s, s, r) * (0.5f * BUILTIN_AMDGPU_RSQRT_F32(r));
    sl = r == 0.0f ? 0.0f : sl;

    // p + pl = s * (1/pi + u)
    float p = s * piinv;
    float pl = BUILTIN_FMA_F32(s, piinv, -p);
    pl = BUILTIN_FMA_F32(sl, piinv, pl);
    pl = BUILTIN_FMA_F32(s, u, pl);

    // 0.5 - 2*(p + pl) with a single final rounding
    float d = BUILTIN_FMA_F32(-2.0f, p, 0.5f);
    float de = BUILTIN_FMA_F32(-2.0f, p, 0.5f - d);
    float ret = d + BUILTIN_FMA_F32(-2.0f, pl, de);
#else
    // p + pl = s * (1/pi + u)
    float p = s * piinv;
    float pl = BUILTIN_FMA_F32(s, u, BUILTIN_FMA_F32(s, piinv, -p));
    float ret = BUILTIN_FMA_F32(-2.0f, pl, BUILTIN_FMA_F32(-2.0f, p, 0.5f));
#endif
    float xux = MATH_MAD(piinv, ax, ax*u);
    ret = ax >= 0.5f ? ret : xux;

    return BUILTIN_COPYSIGN_F32(ret, x);
}

