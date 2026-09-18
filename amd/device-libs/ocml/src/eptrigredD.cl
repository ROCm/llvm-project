/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"
#include "trigredD.h"

CONSTATTR struct epredret
MATH_PRIVATE(eptrigred)(double x)
{
    // Prefer nans use the small path. The large path has elidable nan checks
    // implied by the condition and the small does not.
    if (x >= 0x1.0p+30)
        return MATH_PRIVATE(eptrigredlarge)(x);
    else
        return MATH_PRIVATE(eptrigredsmall)(x);
}

