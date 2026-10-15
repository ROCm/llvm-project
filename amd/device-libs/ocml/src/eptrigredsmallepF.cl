/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"
#include "trigredF.h"

#define FLOAT_SPECIALIZATION
#include "ep.h"

CONSTATTR struct epredret
MATH_PRIVATE(eptrigredsmallep)(float2 x)
{
    const float twobypi_h = 0x1.45f306p-1f;
    const float twobypi_l = 0x1.b9391p-26f;
    const float piby2_h = 0x1.921fb4p+0f;
    const float piby2_m = 0x1.4442d0p-24f;
    const float piby2_l = 0x1.846988p-48f;

    // A single-float 2/pi is only good to ~2^-8 here, so form x*2/pi as two
    // floats (48 bits) and correct fn against the exact fractional remainder.
    float fh = x.hi * twobypi_h;
    float fl = BUILTIN_FMA_F32(x.hi, twobypi_h, -fh) + BUILTIN_FMA_F32(x.hi, twobypi_l, x.lo * twobypi_h);
    float fn = BUILTIN_RINT_F32(fh);
    float rem = (fh - fn) + fl;
    fn += (rem > 0.5f) ? 1.0f : (rem < -0.5f ? -1.0f : 0.0f);

    float xt = BUILTIN_FMA_F32(fn, -piby2_h, x.hi);
    float yh = BUILTIN_FMA_F32(fn, -piby2_m, xt);
    float ph = fn * piby2_m;
    float pt = BUILTIN_FMA_F32(fn, piby2_m, -ph);
    float th = xt - ph;
    float tt = (xt - th) - ph;
    float yt = BUILTIN_FMA_F32(fn, -piby2_l, ((th - yh) + tt) - pt);

    struct epredret ret;
    ret.r = add(con(yh, yt), x.lo);
    ret.i = BUILTIN_ISNAN_F32(fn) ? 0 : ((int)fn & 0x3);
    return ret;
}
