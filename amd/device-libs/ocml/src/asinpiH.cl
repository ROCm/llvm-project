/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR UGEN(asinpi)

CONSTATTR half
MATH_MANGLE(asinpi)(half x)
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
    const float piinv = 0x1.45f306p-2f;

    float ax = BUILTIN_ABS_F32((float)x);

    float rt = BUILTIN_FMA_F32(-0.5f, ax, 0.5f);
    float x2 = ax * ax;
    float r = ax > 0.5f ? rt : x2;

    float u = r * BUILTIN_FMA_F32(r, 0x1.ebc5e8p-6f, 0x1.ae67d4p-5f);

    float s = BUILTIN_AMDGPU_SQRT_F32(r);
    float z = BUILTIN_FMA_F32(-2.0f, BUILTIN_FMA_F32(s, u, piinv*s), 0.5f);
    float zs = BUILTIN_FMA_F32(ax, u, piinv*ax);
    z = ax > 0.5f ? z : zs;

    return BUILTIN_COPYSIGN_F16((half)z, x);
}

