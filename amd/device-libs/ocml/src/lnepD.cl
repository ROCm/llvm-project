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
MATH_PRIVATE(lnep)(double2 a, int ea)
{
    int a_hi_exp;
    double m_hi = BUILTIN_FREXP_F64(a.hi, &a_hi_exp);

    int b = m_hi < (2.0/3.0);
    int e = a_hi_exp - b;
    double2 m = ldx(a, -e);

    // x = (m - 1)/(m + 1): m.hi - 1 is exact, and 1 + m.hi is a Fast2Sum since
    // 1 has the larger (or equal) exponent
    double dh = 1.0 + m.hi;
    double dl = ((1.0 - dh) + m.hi) + m.lo;
    double nh = m.hi - 1.0;
    double rc = MATH_FAST_RCP(dh);
    double q = (nh + m.lo) * rc;
    double2 x = fadd(q, BUILTIN_FMA_F64(-q, dl, BUILTIN_FMA_F64(-q, dh, nh) + m.lo) * rc);
    double s = x.hi * x.hi;
    double p = PE6(s, 0x1.3ab76bf559e2bp-3, 0x1.385386b47b09ap-3, 0x1.7474dd7f4df2ep-3, 0x1.c71c016291751p-3,
                     0x1.249249b27acf1p-2, 0x1.99999998ef7b6p-2, 0x1.5555555555780p-1);
    double2 r = add(mul(con(0x1.62e42fefa39efp-1, 0x1.abc9e3b39803fp-56), (double)(e + ea)),
                    fadd(ldx(x,1), s * x.hi * p));
    return r.hi;
}

