/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

#define DOUBLE_SPECIALIZATION
#include "ep.h"

#include "trigredD.h"

extern CONSTATTR double2 MATH_PRIVATE(epexpep)(double2 z);

CONSTATTR double2
MATH_MANGLE(ctanh)(double2 z)
{
    double4 sc = MATH_PRIVATE(epsincos)(z.y);
    double2 cy = sc.lo;
    double2 sy = sc.hi;
    double x = BUILTIN_ABS_F64(z.x);

    double rr, ri;
    if (x < 0x1.419ecb712c481p+4) {
        double2 e = MATH_PRIVATE(epexpep)(sub(x, con(0x1.62e42fefa39efp-1,0x1.abc9e3b39803fp-56)));
        double2 er = rcp(e);
        er = ldx(er, -2);
        double2 cx = fadd(e, er);
        double2 sx = x < 0x1.0p-27 ? con(x, 0.0) : fsub(e, er);

        double2 d = add(sqr(cy), sqr(sx));
        double2 rd = rcp(d);
        rr = BUILTIN_COPYSIGN_F64(mul(mul(cx, sx), rd).hi, z.x);
        ri = mul(mul(cy, sy), rd).hi;
    } else {
        rr = BUILTIN_COPYSIGN_F64(1.0, z.x);
        ri = 4.0 * (cy.hi * sy.hi) * MATH_MANGLE(exp)(-2.0 * x);
    }

    if (!FINITE_ONLY_OPT()) {
        bool xn = BUILTIN_ISNAN_F64(x);
        bool yin = !BUILTIN_ISFINITE_F64(z.y);
        bool ni = BUILTIN_CLASS_F64(x, CLASS_PZER|CLASS_PSUB|CLASS_PNOR) & yin;
        rr = (ni | xn) ? QNAN_F64 : rr;
        ri = ni ? QNAN_F64 : ri;
        ri = (BUILTIN_ISINF_F64(x) & yin) ? 0.0 : ri;
        ri = (xn & (z.y == 0.0)) ? z.y : ri;
    }

    return (double2)(rr, ri);
}

