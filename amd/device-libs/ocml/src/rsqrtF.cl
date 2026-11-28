/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

#if defined MORE_ACCURACY
// One Newton step with e = 1 - x*y^2 and x*y exact
static float
refine(float x, float y)
{
    float t = x * y;
    float e = BUILTIN_FMA_F32(-BUILTIN_FMA_F32(x, y, -t), y, BUILTIN_FMA_F32(-t, y, 1.0f));
    float r = BUILTIN_FMA_F32(0.5f * y, e, y);
    return BUILTIN_CLASS_F32(y, CLASS_PINF|CLASS_NINF|CLASS_PZER) ? y : r;
}
#endif

CONSTATTR float
MATH_MANGLE(rsqrt)(float x)
{
    if (DAZ_OPT()) {
        float result = BUILTIN_AMDGPU_RSQRT_F32(x);
#if defined MORE_ACCURACY
        result = refine(x, result);
#endif
        return result;
    } else {
        bool need_scale = x < 0x1p-126f;
        float scaled_input = need_scale ? 0x1.0p+24f * x : x;
        float result = BUILTIN_AMDGPU_RSQRT_F32(scaled_input);
#if defined MORE_ACCURACY
        result = refine(scaled_input, result);
#endif
        return need_scale ? result * 0x1.0p+12f : result;
    }
}

