/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR UGEN(atan)

CONSTATTR half
MATH_MANGLE(atan)(half x)
{
    // Evaluate in float so that only the final conversion rounds to half
    float v = BUILTIN_ABS_F32((float)x);
    bool g = v > 1.0f;

    float vi = BUILTIN_AMDGPU_RCP_F32(v);
    v = g ? vi : v;

    float vv = v * v;
    float p = BUILTIN_FMA_F32(vv, BUILTIN_FMA_F32(vv, BUILTIN_FMA_F32(vv, 0x1.96fbc8p-6f, -0x1.816ce6p-4f), 0x1.7e9874p-3f), -0x1.541a14p-2f);
    float a = BUILTIN_FMA_F32(vv, v*p, v);

    float y = 0x1.921fb6p+0f - a;
    a = g ? y : a;

    return BUILTIN_COPYSIGN_F16((half)a, x);
}

