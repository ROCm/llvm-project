/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

#define DOUBLE_SPECIALIZATION
#include "ep.h"

CONSTATTR double4
MATH_PRIVATE(epcmulep)(double4 a, double4 b)
{
    double2 are = a.lo;
    double2 aim = a.hi;
    double2 bre = b.lo;
    double2 bim = b.hi;
    double2 re = sub(mul(are, bre), mul(aim, bim));
    double2 im = add(mul(are, bim), mul(aim, bre));
    return (double4)(re, im);
}
