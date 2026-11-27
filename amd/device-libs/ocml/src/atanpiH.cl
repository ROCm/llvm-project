
/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR UGEN(atanpi)

CONSTATTR half
MATH_MANGLE(atanpi)(half x)
{
    // Evaluate in float so that only the final conversion rounds to half
    float v = BUILTIN_ABS_F32((float)x);
    bool g = v > 1.0f;

    float vi = BUILTIN_AMDGPU_RCP_F32(v);
    v = g ? vi : v;

    float vv = v * v;
    float a = v * BUILTIN_FMA_F32(vv, BUILTIN_FMA_F32(vv, BUILTIN_FMA_F32(vv, BUILTIN_FMA_F32(vv, 0x1.f1ca04p-8f, -0x1.df6742p-6f), 0x1.e2f692p-5f), -0x1.b07856p-4f), 0x1.45f086p-2f);

    float y = 0.5f - a;
    a = g ? y : a;

    return BUILTIN_COPYSIGN_F16((half)a, x);
}


