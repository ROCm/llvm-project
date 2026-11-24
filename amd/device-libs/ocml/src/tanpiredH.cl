/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"
#include "trigpiredH.h"

CONSTATTR half
MATH_PRIVATE(tanpired)(half x, short i)
{
    // Evaluate in float so that only the final conversion rounds to half
    float fx = (float)x;
    float s = fx * fx;
    float t = BUILTIN_FMA_F32(s, BUILTIN_FMA_F32(s, 0x1.769084p+12f, 0x1.768acep+8f), 0x1.59481p+7f);
    t = fx * BUILTIN_FMA_F32(s, BUILTIN_FMA_F32(s, BUILTIN_FMA_F32(s, t, 0x1.4540a6p+5f), 0x1.4ac24p+3f), 0x1.921fbp+1f);

    float tr = -BUILTIN_AMDGPU_RCP_F32(t);

    return (half)(i ? tr : t);
}

