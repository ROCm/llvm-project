/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR float
MATH_PRIVATE(bp0)(float t)
{
    return
        PE4(t, 0x1.0cd6fap+4f, -0x1.9a583ap+0f, 0x1.ad0a9ep-3f, -0x1.0aaa9cp-4f, 0x1.000000p-3f);
}
