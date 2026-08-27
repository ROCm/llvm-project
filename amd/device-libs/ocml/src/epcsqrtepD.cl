/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

#define DOUBLE_SPECIALIZATION
#include "ep.h"

CONSTATTR double4
MATH_PRIVATE(epcsqrtep)(double4 z)
{
    double2 x = z.lo;
    double2 y = z.hi;

    int e = BUILTIN_FREXP_EXP_F64(BUILTIN_MAX_F64(BUILTIN_ABS_F64(x.hi), BUILTIN_ABS_F64(y.hi)));
    e &= ~1;
    x = ldx(x, -e);
    y = ldx(y, -e);

    double2 u = root2(fadd(root2(add(sqr(x), sqr(y))), absv(x)) * 0.5);
    double2 v = absv(fdiv(y, u) * 0.5);
    v = ((y.hi == 0.0) & (u.hi == 0.0)) ? y : v;
    bool b = x.hi >= 0.0;
    double2 s = b ? u : v;
    double2 t = csgn(b ? v : u, y);
    return (double4)(ldx(s, e/2), ldx(t, e/2));
}

