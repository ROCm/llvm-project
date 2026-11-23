/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

#define DOUBLE_SPECIALIZATION
#include "ep.h"

extern CONSTATTR double2 MATH_PRIVATE(epexpep)(double2 x);

CONSTATTR double
MATH_MANGLE(expm1)(double x)
{
#if defined EXTRA_ACCURACY
    double2 e = sub(MATH_PRIVATE(epexpep)(con(x, 0.0)), 1.0);
    double z = e.hi;
#else
    // dn >= -53 keeps s - 1 exact; then |t| <= 1.3 for x >= -38
    double dn = BUILTIN_MAX_F64(BUILTIN_RINT_F64(x * 0x1.71547652b82fep+0), -53.0);
#if defined MORE_ACCURACY
    // t + tl = x - dn*ln2
    double th = MATH_MAD(-dn, 0x1.62e42fefa39efp-1, x);
    double t = MATH_MAD(-dn, 0x1.abc9e3b39803fp-56, th);
    double tl = MATH_MAD(-dn, 0x1.abc9e3b39803fp-56, th - t);
#else
    double t = MATH_MAD(-dn, 0x1.abc9e3b39803fp-56, MATH_MAD(-dn, 0x1.62e42fefa39efp-1, x));
#endif

    double p = PE10(t, 0x1.1f32ea9d67f34p-29, 0x1.af4eb2a1b768bp-26, 0x1.27e500e0ac05bp-22, 0x1.71de01b889c29p-19,
                     0x1.a01a0197bcfd8p-16, 0x1.a01a01ac1a723p-13, 0x1.6c16c16c18931p-10,
                     0x1.1111111110056p-7, 0x1.5555555555552p-5, 0x1.5555555555557p-3,
                     0x1.0000000000000p-1);

    int e = dn == 1024.0 ? 1023 : (int)dn;
    double s = BUILTIN_FLDEXP_F64(1.0, e);
#if defined MORE_ACCURACY
    // ph + pl = expm1(t + tl)
    double u = t * p;
    double ph = MATH_MAD(t, u, t);
    double pl = MATH_MAD(t, u, t - ph);
    pl = MATH_MAD(tl, ph, pl + tl);

    // s - 1 may be inexact for large e; a gets its exact residual
    double sm1 = s - 1.0;
    double c = (s - sm1) - 1.0;
    double a = MATH_MAD(s, ph, sm1);
    double z = a + (MATH_MAD(s, ph, sm1 - a) + MATH_MAD(s, pl, c));
#else
    p = MATH_MAD(t, t*p, t);
    double z = MATH_MAD(s, p, s - 1.0);
#endif
    z = dn == 1024.0 ? 2.0*z : z;
#endif

    if (!FINITE_ONLY_OPT()) {
        z = x > 0x1.62e42fefa39efp+9 ? PINF_F64 : z;
    }

    z = x < -38.0 ? -1.0 : z;

    return z;
}

