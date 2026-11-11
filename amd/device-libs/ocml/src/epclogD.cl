/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

#define DOUBLE_SPECIALIZATION
#include "ep.h"

extern CONSTATTR double2 MATH_PRIVATE(eplnep)(double2, int);
extern CONSTATTR double2 MATH_PRIVATE(epatan2)(double, double);

CONSTATTR double4
MATH_PRIVATE(epclog)(double2 z)
{
    double x = z.s0;
    double y = z.s1;
    double a = BUILTIN_ABS_F64(x);
    double b = BUILTIN_ABS_F64(y);
    double t = BUILTIN_MAX_F64(a, b);
    int e = BUILTIN_FREXP_EXP_F64(t);

    // Near |z|=1, form m = a^2+b^2-1 exactly to keep the small deviation.
    double2 m;
    bool near1 = false;
    if (e == 0 || e == 1) {
        double2 sx = sqr(a);
        double2 sy = sqr(b);
        double2 p1 = add(sx.hi, sy.hi);
        double2 p2 = add(p1.hi, -1.0);
        m = add(p2.hi, p2.lo);
        m = add(m, p1.lo);
        m = add(m, sx.lo);
        m = add(m, sy.lo);
        near1 = m.hi >= -(1.0 / 3.0) && m.hi <= 0.5;
    }

    double2 rr;
    if (near1) {
        double2 xx = div(m, fadd(2.0, m));
        double2 s = sqr(xx);
        double sh = s.hi;
#ifdef EXTRA_ACCURACY
        double2 p = con(PE6(sh, 0x1.2a75ed8c78cedp-4, 0x1.e3f1736096bc6p-5, 0x1.0866da59bb6e8p-4,
                                0x1.1a81cf9e127f0p-4, 0x1.2f67909367e70p-4, 0x1.47ae1f24e172dp-4,
                                0x1.642c852c2f73ap-4), 0.0);
        p = add(mul(p, s), con( 0x1.86186188ae3b2p-4, -0x1.b2bf19f4b9f42p-58));
        p = add(mul(p, s), con( 0x1.af286bca0eb84p-4, -0x1.3922b9ec0c372p-58));
        p = add(mul(p, s), con( 0x1.e1e1e1e1e20bap-4,  0x1.ec1f3a04b7e7fp-58));
        p = add(mul(p, s), con( 0x1.111111111110ep-3, -0x1.00c4820ebdf58p-58));
        p = add(mul(p, s), con( 0x1.3b13b13b13b14p-3, -0x1.253b8903d2bf0p-57));
        p = add(mul(p, s), con( 0x1.745d1745d1746p-3, -0x1.748f5e9e8b048p-58));
        p = add(mul(p, s), con( 0x1.c71c71c71c71cp-3,  0x1.c71c840a46769p-57));
        p = add(mul(p, s), con( 0x1.2492492492492p-2,  0x1.24924920cdd42p-56));
        p = add(mul(p, s), con( 0x1.999999999999ap-2, -0x1.9999999998df8p-56));
        p = add(mul(p, s), con( 0x1.5555555555555p-1,  0x1.5555555555555p-55));
#else
        double2 p = con(PE8(sh, 0x1.b56b272c2422dp-4, 0x1.7e1325cdb9377p-4, 0x1.af98c5051531ep-4,
                                0x1.e1de1831a935bp-4, 0x1.11111b86cd1edp-3, 0x1.3b13b1160fcfep-3,
                                0x1.745d174622febp-3, 0x1.c71c71c71c092p-3, 0x1.2492492492494p-2), 0.0);
        p = add(mul(p, s), con( 0x1.999999999999ap-2, -0x1.9bcec14f9cf79p-56));
        p = add(mul(p, s), con( 0x1.5555555555555p-1,  0x1.5555614e509dfp-55));
#endif
        double2 corr = mul(mul(xx, s), p);
        rr = ldx(add(ldx(xx, 1), corr), -1);
    } else {
        double as = BUILTIN_FLDEXP_F64(a, -e);
        double bs = BUILTIN_FLDEXP_F64(b, -e);
        rr = ldx(MATH_PRIVATE(eplnep)(add(sqr(as), sqr(bs)), 2 * e), -1);
    }

    double2 ri = MATH_PRIVATE(epatan2)(y, x);
    return (double4)(rr, ri);
}
