/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"
#include "trigredF.h"

CONSTATTR float4
MATH_PRIVATE(epsincos)(float y)
{
    float ay = BUILTIN_ABS_F32(y);
    if (!FINITE_ONLY_OPT())
        ay = BUILTIN_ISINF_F32(ay) ? QNAN_F32 : ay;

    struct redret2 r = MATH_PRIVATE(trigred2)(ay);
    float4 sc = MATH_PRIVATE(epsincosred2)(r.hi, r.lo);
    float2 cr = sc.lo;
    float2 sr = sc.hi;

    bool odd = (r.i & 1) != 0;
    float2 s = odd ? cr : sr;
    float2 c = odd ? -sr : cr;

    if (r.i > 1) {
        s = -s;
        c = -c;
    }

    int sgn = AS_INT(y) & (int)SIGNBIT_SP32;
    s.lo = AS_FLOAT(AS_INT(s.lo) ^ sgn);
    s.hi = AS_FLOAT(AS_INT(s.hi) ^ sgn);

    return (float4)(c, s);
}
