/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"
#include "trigpiredF.h"

CONSTATTR float
MATH_PRIVATE(tanpired)(float x, int i)
{
    float s = x * x;

    float t = PE6(s, 0x1.ebdc6cp+16f, 0x1.2649cp+7f, 0x1.8fbfcep+11f, 0x1.3cfa48p+9f, 0x1.4693p+7f,
                    0x1.466758p+5f, 0x1.4abbdap+3f);

    // th + tl = tan(pi*x)
    float c = BUILTIN_FMA_F32(x * s, t, x * -0x1.777a5cp-24f);
    float th = BUILTIN_FMA_F32(x, 0x1.921fb6p+1f, c);
    float tl = BUILTIN_FMA_F32(x, 0x1.921fb6p+1f, -th) + c;

    // -1/(th + tl)
    float r = MATH_FAST_RCP(th);
    float tr = -BUILTIN_FMA_F32(BUILTIN_FMA_F32(-r, tl, BUILTIN_FMA_F32(-r, th, 1.0f)), r, r);
    tr = x == 0.0f ? -PINF_F32 : tr;

    return i ? tr : th;
}

