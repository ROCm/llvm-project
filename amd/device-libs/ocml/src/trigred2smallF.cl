/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"
#include "trigredF.h"

CONSTATTR struct redret2
MATH_PRIVATE(trigred2small)(float x)
{
    const float twobypi = 0x1.45f306p-1f;
    const float piby2_h = 0x1.921fb4p+0f;
    const float piby2_m = 0x1.4442d0p-24f;
    const float piby2_l = 0x1.846988p-48f;

    float fn = BUILTIN_RINT_F32(x * twobypi);

    float xt = BUILTIN_FMA_F32(fn, -piby2_h, x);
    float yh = BUILTIN_FMA_F32(fn, -piby2_m, xt);
    float ph = fn * piby2_m;
    float pt = BUILTIN_FMA_F32(fn, piby2_m, -ph);
    float th = xt - ph;
    float tt = (xt - th) - ph;
    float yt = BUILTIN_FMA_F32(fn, -piby2_l, ((th - yh) + tt) - pt);
    float rh = yh + yt;
    float rt = yt - (rh - yh);

    struct redret2 ret;
    ret.hi = rh;
    ret.lo = rt;
    ret.i = BUILTIN_ISNAN_F32(fn) ? 0 : ((int)fn & 0x3);
    return ret;
}
