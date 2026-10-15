/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"
#include "trigredF.h"

// ep-in / ep-out reduction of x (x.hi assumed >= 0; caller handles sign).
CONSTATTR struct epredret
MATH_PRIVATE(eptrigredep)(float2 x)
{
    if (x.hi >= SMALL_BOUND)
        return MATH_PRIVATE(eptrigredlargeep)(x);
    else
        return MATH_PRIVATE(eptrigredsmallep)(x);
}
