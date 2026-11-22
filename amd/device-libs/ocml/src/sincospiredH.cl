/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"
#include "trigpiredH.h"

CONSTATTR struct scret
MATH_PRIVATE(sincospired)(half x)
{
    // Evaluate in float so that only the final conversion rounds to half
    float fx = (float)x;
    float t = fx * fx;
    float sx = fx * BUILTIN_FMA_F32(t, BUILTIN_FMA_F32(t, 0x1.3f3f3cp+1f, -0x1.4aa618p+2f), 0x1.921f8ep+1f);
    float cx = BUILTIN_FMA_F32(t, BUILTIN_FMA_F32(t, BUILTIN_FMA_F32(t, -0x1.4e7de8p+0f, 0x1.03b068p+2f),
                                                -0x1.3bd39cp+2f), 1.0f);

    struct scret ret;
    ret.c = (half)cx;
    ret.s = (half)sx;
    return ret;
}

