/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR float
MATH_PRIVATE(ba1)(float t)
{
    return
        PE3(t, 0x1.6e8140p-1f, -0x1.8b4b82p-3f, 0x1.7fffccp-3f, 0x1.000000p+0f);
}
