/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"
#include "trigredD.h"

// ep-in / ep-out reduction of x (x.hi assumed >= 0; caller handles sign).
CONSTATTR struct epredret
MATH_PRIVATE(eptrigredep)(double2 x)
{
    if (x.hi >= 0x1.0p+30)
        return MATH_PRIVATE(eptrigredlargeep)(x);
    else
        return MATH_PRIVATE(eptrigredsmallep)(x);
}
