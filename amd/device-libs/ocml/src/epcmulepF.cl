/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

#define FLOAT_SPECIALIZATION
#include "ep.h"

CONSTATTR float4
MATH_PRIVATE(epcmulep)(float4 a, float4 b)
{
    float2 are = a.lo;
    float2 aim = a.hi;
    float2 bre = b.lo;
    float2 bim = b.hi;
    float2 re = sub(mul(are, bre), mul(aim, bim));
    float2 im = add(mul(are, bim), mul(aim, bre));
    return (float4)(re, im);
}
