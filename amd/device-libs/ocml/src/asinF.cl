/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR float
MATH_MANGLE(asin)(float x)
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

    float ax = BUILTIN_ABS_F32(x);
    float tx = MATH_MAD(ax, -0.5f, 0.5f);
    float x2 = x*x;
    float r = ax >= 0.5f ? tx : x2;

    float u = r * PE5(r, 0x1.38434ep-5f, 0x1.bf8bb4p-7f, 0x1.069878p-5f, 0x1.6c8362p-5f, 0x1.33379p-4f,
                        0x1.555558p-3f);

    float s = MATH_FAST_SQRT(r);
#if defined EXTRA_ACCURACY
    // s + sl = sqrt(r)
    float sl = BUILTIN_FMA_F32(-s, s, r) * (0.5f * BUILTIN_AMDGPU_RSQRT_F32(r));
    sl = r == 0.0f ? 0.0f : sl;

    // pi/2 - 2*(s + sl + s*u) with a single final rounding
    const float piby2_h = 0x1.921fb6p+0f;
    const float piby2_l = -0x1.777a5cp-25f;
    float w = MATH_MAD(s, u, sl);
    float d = BUILTIN_FMA_F32(-2.0f, s, piby2_h);
    float de = BUILTIN_FMA_F32(-2.0f, s, piby2_h - d);
    float ret = d + BUILTIN_FMA_F32(-2.0f, w, de + piby2_l);
#else
    float ret = BUILTIN_FMA_F32(0x1.ddcb02p-1f, 0x1.aee9d6p+0f, -2.0f*MATH_MAD(s, u, s));
#endif

    float xux = MATH_MAD(ax, u, ax);
    ret = ax < 0.5f ? xux : ret;

    return BUILTIN_COPYSIGN_F32(ret, x);
}

