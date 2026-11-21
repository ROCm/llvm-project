/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"
#include "trigredF.h"

CONSTATTR float
MATH_PRIVATE(tanredep)(float2 xy, int sel)
{
    float x = xy.hi;
    float s = x * x;
    float p = s * PE5(s, 0x1.33d5e6p-7f, 0x1.9697f8p-9f, 0x1.907be2p-6f, 0x1.b581ap-5f, 0x1.112e2p-3f, 0x1.5554dcp-2f);

    // t + tt = tan(x + xy.lo), using tan(x + d) ~ tan(x) + d*(1 + tan(x)^2)
    float t = BUILTIN_FMA_F32(p, x, x);
    float tt = BUILTIN_FMA_F32(p, x, -(t - x));
    tt = BUILTIN_FMA_F32(xy.lo, BUILTIN_FMA_F32(t, t, 1.0f), tt);

    float tr = -MATH_FAST_RCP(t);
    float e = BUILTIN_FMA_F32(tt, tr, BUILTIN_FMA_F32(t, tr, 1.0f));
    tr = BUILTIN_FMA_F32(e, tr, tr);

    return sel ? tr : t + tt;
}
