/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

#define DOUBLE_SPECIALIZATION
#include "ep.h"

extern CONSTATTR double MATH_PRIVATE(lnep)(double2 a, int ea);

CONSTATTR double
MATH_MANGLE(acosh)(double x)
{
    bool b = x >= 0x1.0p+512;
    double s = b ? 0x1.0p-512 : 1.0;
    double sx = x * s;
    double s2 = s * s;

    // h = sx^2 - s^2 = hh + hl, where sx^2 >= s^2
    double pp = sx * sx;
    double pl = BUILTIN_FMA_F64(sx, sx, -pp);
    double hh = pp - s2;
    double hl = ((pp - hh) - s2) + pl;

    // a = sx + sqrt(h), where sx >= sqrt(h)
    double rh = MATH_FAST_SQRT(hh);
    double rl = (BUILTIN_FMA_F64(-rh, rh, hh) + hl) * (0.5 * MATH_FAST_RCP(rh));
    rl = hh == 0.0 ? 0.0 : rl;
    double ah = sx + rh;
    double2 a = con(ah, ((sx - ah) + rh) + rl);
    double z = MATH_PRIVATE(lnep)(a, b ? 512 : 0);

    if (!FINITE_ONLY_OPT()) {
        z = x == PINF_F64 ? x : z;
        z = x < 1.0 ? QNAN_F64 : z;
    }

    return z;
}

