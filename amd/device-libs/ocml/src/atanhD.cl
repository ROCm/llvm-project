/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

#define DOUBLE_SPECIALIZATION
#include "ep.h"


CONSTATTR double
MATH_MANGLE(atanh)(double x)
{
    double y = BUILTIN_ABS_F64(x);
    // (1 + y)/(1 - y) = 2^e (1 + r)/(1 - r) with |r| <= 1/5, so that
    // atanh(y) = e*ln(2)/2 + atanh(r)
    int ae;
    double am = BUILTIN_FREXP_F64((1.0 + y) * MATH_FAST_RCP(1.0 - y), &ae);
    int e = ae - (am < (2.0/3.0));
    double uh = 1.0 + y;
    double ul = (1.0 - uh) + y;
    double vh = 1.0 - y;
    double vl = BUILTIN_FLDEXP_F64((1.0 - vh) - y, e);
    vh = BUILTIN_FLDEXP_F64(vh, e);

    // r = (u - v)/(u + v); u - v is exact in the high part
    double nh = uh - vh;
    double nl = ul - vl;
    double dh = uh + vh;
    double db = dh - uh;
    double dl = ((uh - (dh - db)) + (vh - db)) + (ul + vl);
    double rc = MATH_FAST_RCP(dh);
    double q = nh * rc;
    double2 r = fadd(q, BUILTIN_FMA_F64(-q, dl, BUILTIN_FMA_F64(-q, dh, nh) + nl) * rc);

    // atanh(r) = r + r^3 p(r^2)
    double s = r.hi * r.hi;
    double p = PE6(s, 0x1.3ab76bf559e2bp-4, 0x1.385386b47b09ap-4, 0x1.7474dd7f4df2ep-4, 0x1.c71c016291751p-4,
                     0x1.249249b27acf1p-3, 0x1.99999998ef7b6p-3, 0x1.5555555555780p-2);
    double z = add(mul(con(0x1.62e42fefa39efp-2, 0x1.abc9e3b39803fp-57), (double)e), fadd(r, s * r.hi * p)).hi;

    if (!FINITE_ONLY_OPT()) {
        z = y > 1.0 ? QNAN_F64 : z;
        z = y == 1.0 ? PINF_F64 : z;
    }

    return BUILTIN_COPYSIGN_F64(z, x);
}

