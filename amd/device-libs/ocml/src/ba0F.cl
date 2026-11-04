/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR float
MATH_PRIVATE(ba0)(float t)
{
    return
        PE3(t, -0x1.e75912p-2f, 0x1.a6f35cp-4f, -0x1.ffff68p-5f, 0x1.000000p+0f);
}
