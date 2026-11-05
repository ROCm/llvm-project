/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"
#include "trigredF.h"

CONSTATTR float4
MATH_PRIVATE(epsincosep)(float2 x)
{
    float2 ax = x.hi < 0.0f ? -x : x;

    struct epredret r = MATH_PRIVATE(eptrigredep)(ax);
    float4 sc = MATH_PRIVATE(epsincosredep)(r.r);
    float2 cr = sc.lo;
    float2 sr = sc.hi;

    bool odd = (r.i & 1) != 0;
    float2 s = odd ? cr : sr;
    float2 c = odd ? -sr : cr;

    if (r.i > 1) {
        s = -s;
        c = -c;
    }

    int sgn = AS_INT(x.hi) & (int)SIGNBIT_SP32;
    s.lo = AS_FLOAT(AS_INT(s.lo) ^ sgn);
    s.hi = AS_FLOAT(AS_INT(s.hi) ^ sgn);

    return (float4)(c, s);
}
