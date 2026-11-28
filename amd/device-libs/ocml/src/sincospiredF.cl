/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"
#include "trigredF.h"

CONSTATTR struct scret
MATH_PRIVATE(sincospired)(float x)
{

    float t = x * x;

    float sx = PE3(t, 0x1.46c086p-4f, -0x1.32b5ap-1f, 0x1.466b84p+1f, -0x1.4abbcep+2f);
    sx = x * t * sx;
    sx = BUILTIN_FMA_F32(x, 0x1.921fb6p+1f, BUILTIN_FMA_F32(x, -0x1.777a5cp-24f, sx));

    float cx = PE4(t, 0x1.97ca88p-5f, 0x1.c85d3ap-3f, -0x1.55a3b4p+0f, 0x1.03c1a6p+2f, -0x1.3bd3ccp+2f);
    cx = MATH_MAD(t, cx, 1.0f);

    struct scret ret;
    ret.c = cx;
    ret.s = sx;
    return ret;
}

