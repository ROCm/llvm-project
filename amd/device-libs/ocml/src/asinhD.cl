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
MATH_MANGLE(asinh)(double x)
{
    double y = BUILTIN_ABS_F64(x);
    bool b = y >= 0x1.0p+512;
    double s = b ? 0x1.0p-512 : 1.0;
    double sy = y * s;
    double s2 = s * s;

    // h = sy^2 + s^2 = hh + hl
    double pp = sy * sy;
    double pl = BUILTIN_FMA_F64(sy, sy, -pp);
    double hh = pp + s2;
    double hb = hh - pp;
    double hl = ((pp - (hh - hb)) + (s2 - hb)) + pl;

    // a = sy + sqrt(h), where sqrt(h) >= sy
    double rh = MATH_FAST_SQRT(hh);
    double rl = (BUILTIN_FMA_F64(-rh, rh, hh) + hl) * (0.5 * BUILTIN_AMDGPU_RSQRT_F64(hh));
    double ah = rh + sy;
    double2 a = con(ah, ((rh - ah) + sy) + rl);
    double z = MATH_PRIVATE(lnep)(a, b ? 512 : 0);
    z = y < 0x1.0p-27 ? y : z;

    if (!FINITE_ONLY_OPT()) {
        z = y == PINF_F64 ? y : z;
    }

    return BUILTIN_COPYSIGN_F64(z, x);
}

