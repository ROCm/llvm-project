/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR UGEN(asin)

CONSTATTR half
MATH_MANGLE(asin)(half x)
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

    // Evaluate in float so that only the final conversion rounds to half
    float fx = (float)x;
    float ax = BUILTIN_ABS_F32(fx);
    float tx = BUILTIN_FMA_F32(ax, -0.5f, 0.5f);
    float x2 = ax * ax;
    float r = ax >= 0.5f ? tx : x2;

    float u = r * BUILTIN_FMA_F32(r, 0x1.823f5p-4f, 0x1.5209cep-3f);

    float s = BUILTIN_AMDGPU_SQRT_F32(r);
    float ret = BUILTIN_FMA_F32(0x1.ddcb02p-1f, 0x1.aee9d6p+0f, -2.0f*BUILTIN_FMA_F32(s, u, s));
    float xux = BUILTIN_FMA_F32(ax, u, ax);
    ret = ax < 0.5f ? xux : ret;

    return BUILTIN_COPYSIGN_F16((half)ret, x);
}

