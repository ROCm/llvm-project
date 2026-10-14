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
MATH_PRIVATE(epcsqrtep)(float4 z)
{
    float2 x = z.lo;
    float2 y = z.hi;

    int e = BUILTIN_FREXP_EXP_F32(BUILTIN_MAX_F32(BUILTIN_ABS_F32(x.hi), BUILTIN_ABS_F32(y.hi)));
    e &= ~1;
    x = ldx(x, -e);
    y = ldx(y, -e);

    float2 u = root2(fadd(root2(add(sqr(x), sqr(y))), absv(x)) * 0.5f);
    float2 v = absv(fdiv(y, u) * 0.5f);
    v = ((y.hi == 0.0f) & (u.hi == 0.0f)) ? y : v;
    bool b = x.hi >= 0.0f;
    float2 s = b ? u : v;
    float2 t = csgn(b ? v : u, y);
    return (float4)(ldx(s, e/2), ldx(t, e/2));
}

