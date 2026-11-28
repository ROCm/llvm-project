/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

#define FLOAT_SPECIALIZATION
#include "ep.h"

CONSTATTR float
MATH_PRIVATE(lnep)(float2 a, int ea)
{
    int a_hi_exp;
    float a_hi_m = BUILTIN_FREXP_F32(a.hi, &a_hi_exp);
    int b = a_hi_m < (2.0f/3.0f);
    int e = a_hi_exp - b;
    float2 m = ldx(a, -e);

    // x = (m - 1)/(m + 1): m.hi - 1 is exact, and 1 + m.hi is a Fast2Sum since
    // 1 has the larger (or equal) exponent
    float dh = 1.0f + m.hi;
    float dl = ((1.0f - dh) + m.hi) + m.lo;
    float nh = m.hi - 1.0f;
    float rc = MATH_FAST_RCP(dh);
    float q = (nh + m.lo) * rc;
    float2 x = fadd(q, BUILTIN_FMA_F32(-q, dl, BUILTIN_FMA_F32(-q, dh, nh) + m.lo) * rc);
    float s = x.hi * x.hi;
    float p = PE2(s, 0x1.36db58p-2f, 0x1.992b46p-2f, 0x1.5555b4p-1f);
    float2 r = add(mul(con(0x1.62e430p-1f, -0x1.05c610p-29f), (float)(e + ea)),
                   fadd(ldx(x,1), s * x.hi * p));
    return r.hi;
}

