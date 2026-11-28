/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR UGEN(acos)

CONSTATTR half
MATH_MANGLE(acos)(half x)
{
    // Computes arccos(x).
    // The argument is first reduced by noting that arccos(x)
    // is invalid for abs(x) > 1 and arccos(-x) = arccos(x).
    // For denormal and small arguments arccos(x) = pi/2 to machine
    // accuracy. Remaining argument ranges are handled as follows.
    // For abs(x) <= 0.5 use
    // arccos(x) = pi/2 - arcsin(x)
    // = pi/2 - (x + x^3*R(x^2))
    // where R(x^2) is a rational minimax approximation to
    // (arcsin(x) - x)/x^3.
    // For abs(x) > 0.5 exploit the identity:
    // arccos(x) = pi - 2*arcsin(sqrt(1-x)/2)
    // together with the above rational approximation, and
    // reconstruct the terms carefully.

    // Evaluate in float so that only the final conversion rounds to half
    float fx = (float)x;
    float ax = BUILTIN_ABS_F32(fx);

    float rt = BUILTIN_FMA_F32(-0.5f, ax, 0.5f);
    float x2 = ax * ax;
    float r = ax > 0.5f ? rt : x2;

    float u = r * BUILTIN_FMA_F32(r, 0x1.823f5p-4f, 0x1.5209cep-3f);

    float s = BUILTIN_AMDGPU_SQRT_F32(r);
    float ztp = 2.0f * BUILTIN_FMA_F32(s, u, s);
    float ztn = BUILTIN_FMA_F32(0x1.ddcb02p+0f, 0x1.aee9d6p+0f, -ztp);
    float zt =  fx < 0.0f ? ztn : ztp;
    float z = BUILTIN_FMA_F32(0x1.ddcb02p-1f, 0x1.aee9d6p+0f, -BUILTIN_FMA_F32(fx, u, fx));
    z = ax > 0.5f ? zt : z;

    return (half)z;
}

