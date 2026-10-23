/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"
#include "trigredD.h"

#define DOUBLE_SPECIALIZATION
#include "ep.h"

CONSTATTR struct epredret
MATH_PRIVATE(eptrigredsmallep)(double2 x)
{
    const double twobypi = 0x1.45f306dc9c883p-1;
    const double piby2_h = 0x1.921fb54442d18p+0;
    const double piby2_m = 0x1.1a62633145c00p-54;
    const double piby2_t = 0x1.b839a252049c0p-104;

    // dn from the full ep product, correct even within x.lo of a pi/2 boundary.
    double dn = BUILTIN_RINT_F64(BUILTIN_FMA_F64(x.lo, twobypi, x.hi * twobypi));

    double xt = BUILTIN_FMA_F64(dn, -piby2_h, x.hi);
    double yh = BUILTIN_FMA_F64(dn, -piby2_m, xt);
    double ph = dn * piby2_m;
    double pt = BUILTIN_FMA_F64(dn, piby2_m, -ph);
    double th = xt - ph;
    double tt = (xt - th) - ph;
    double yt = BUILTIN_FMA_F64(dn, -piby2_t, ((th - yh) + tt) - pt);

    struct epredret ret;
    ret.r = add(con(yh, yt), x.lo);
    ret.i = BUILTIN_ISNAN_F64(dn) ? 0 : ((int)dn & 0x3);
    return ret;
}
