/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR float
MATH_PRIVATE(atanpired)(float v)
{
    float t = v * v;
    // v*(1/pi) + v*w(t): the float 1/pi times v is exact in the fma, and w's
    // constant term carries the low part of 1/pi
    float w = PE8(t, 0x1.db87ep-11f, -0x1.4f2a12p-8f, 0x1.bd414ep-7f, -0x1.875916p-6f, 0x1.1577f2p-5f,
                    -0x1.725f52p-5f, 0x1.04a9c4p-4f, -0x1.b2988p-4f, 0x1.125968p-27f);
    return BUILTIN_FMA_F32(v, 0x1.45f306p-2f, v * w);
}

