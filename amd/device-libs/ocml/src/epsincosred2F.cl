/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

#define FLOAT_SPECIALIZATION
#include "ep.h"

#include "trigredF.h"

CONSTATTR float4
MATH_PRIVATE(epsincosred2)(float x, float y)
{
    float2 a = fadd(x, y);
    float2 a2 = sqr(a);
    float2 a3 = mul(a2, a);
    float2 a4 = sqr(a2);
    float t = a2.hi;

    const float2 C0 = (float2)(-0x1.555556p-30f, 0x1.555556p-5f);
    float pc = MATH_MAD(t, MATH_MAD(t, -0x1.27e4fcp-22f, 0x1.a01a02p-16f), -0x1.6c16c2p-10f);
    float2 c = add(fsub(1.0f, ldx(a2, -1)), mul(a4, add(C0, mul(a2, pc))));

    const float2 S0 = (float2)(0x1.555556p-28f, -0x1.555556p-3f);
    float ps = MATH_MAD(t, MATH_MAD(t, 0x1.71de3ap-19f, -0x1.a01a02p-13f), 0x1.111112p-7f);
    float2 s = add(a, mul(a3, add(S0, mul(a2, ps))));

    return (float4)(c, s);
}
