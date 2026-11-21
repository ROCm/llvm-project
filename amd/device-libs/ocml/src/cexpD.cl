/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

#if defined EXTRA_ACCURACY
#define DOUBLE_SPECIALIZATION
#include "ep.h"
#include "trigredD.h"

extern CONSTATTR double2 MATH_PRIVATE(epexpep)(double2);
#endif

CONSTATTR double2
MATH_MANGLE(cexp)(double2 z)
{
    double x = z.s0;
    double y = z.s1;
#if defined EXTRA_ACCURACY
    // e^x = 2^n e^t with e^t, cos(y) and sin(y) as hi + lo and one rounding
    // per component before the 2^n; the clamp keeps the reduction finite and
    // lets NaN through
    double4 sc = MATH_PRIVATE(epsincos)(y);
    double2 c = sc.lo;
    double2 sn = sc.hi;
    double xc = x < -760.0 ? -760.0 : x;
    xc = xc > 760.0 ? 760.0 : xc;
    double n = BUILTIN_RINT_F64(xc * 0x1.71547652b82fep+0);
    // n*ln2_hi is exact, and so is xc - n*ln2_hi
    double2 e = MATH_PRIVATE(epexpep)(add(BUILTIN_FMA_F64(-n, 0x1.62e42fefa3800p-1, xc), -n * 0x1.ef35793c76730p-45));
    int in = (int)n;
    double rr = BUILTIN_FLDEXP_F64(BUILTIN_FMA_F64(e.hi, c.hi, BUILTIN_FMA_F64(e.lo, c.hi, e.hi * c.lo)), in);
    // sin(y) is subnormal for subnormal y while the result may be normal
    double sh = sn.hi * 0x1.0p+64;
    double sl = sn.lo * 0x1.0p+64;
    double ri = BUILTIN_FLDEXP_F64(BUILTIN_FMA_F64(e.hi, sh, BUILTIN_FMA_F64(e.lo, sh, e.hi * sl)), in - 64);
#else
    double cy;
    double sy = MATH_MANGLE(sincos)(y, &cy);
    // e^x = 2^n e^t with 2^n applied last, so nothing overflows early; the
    // clamp keeps the reduction finite and lets NaN through
    double xc = x < -760.0 ? -760.0 : x;
    xc = xc > 760.0 ? 760.0 : xc;
    double dn = BUILTIN_RINT_F64(xc * 0x1.71547652b82fep+0);
    double t = MATH_MAD(-dn, 0x1.abc9e3b39803fp-56, MATH_MAD(-dn, 0x1.62e42fefa39efp-1, xc));
    double et = PE11(t, 0x1.ade156a5dcb37p-26, 0x1.28af3fca7ab0cp-22, 0x1.71dee623fde64p-19, 0x1.a01997c89e6b0p-16,
                      0x1.a01a014761f6ep-13, 0x1.6c16c1852b7b0p-10, 0x1.1111111122322p-7,
                      0x1.55555555502a1p-5, 0x1.5555555555511p-3, 0x1.000000000000bp-1, 1.0, 1.0);
    int in = (int)dn;
    double rr = BUILTIN_FLDEXP_F64(et * cy, in);
    // sin(y) is subnormal for subnormal y while the result may be normal
    double ri = BUILTIN_FLDEXP_F64(et * (sy * 0x1.0p+64), in - 64);
#endif

    if (!FINITE_ONLY_OPT()) {
        bool isfinite = BUILTIN_ISFINITE_F64(y);
        if (x == NINF_F64) {
            rr = 0.0;
            ri = isfinite ? ri : 0.0;
        }
        if (x == PINF_F64) {
            rr = isfinite ? rr : PINF_F64;
            ri = isfinite ? ri : QNAN_F64;
        }
        ri = y == 0.0 ? y : ri;
    }

    return (double2)(rr, ri);
}

