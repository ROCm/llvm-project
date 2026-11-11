/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

#define DOUBLE_SPECIALIZATION
#include "ep.h"

CONSTATTR double2
MATH_PRIVATE(eplnep)(double2 a, int ea)
{
    // Reduce a*2^ea to m in [2/3,4/3) so |x|=|(m-1)/(m+1)| stays small.
    int ehi = BUILTIN_FREXP_EXP_F64(a.hi);
    double mh = BUILTIN_FLDEXP_F64(a.hi, -ehi);
    int e = ehi - (mh < (2.0 / 3.0) ? 1 : 0);
    double2 m = ldx(a, -e);
    double2 x = div(fadd(-1.0, m), fadd(1.0, m));
    double2 s = sqr(x);
    double sh = s.hi;

    // P(s) = 2/3 + 2s/5 + ...: high terms in double (PEn), low K terms in dd.
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

    // ln(m) = 2x + x*s*P(s); add e*ln2.
    const double2 ln2 = con(0x1.62e42fefa39efp-1, 0x1.abc9e3b39803fp-56);
    double2 corr = mul(mul(x, s), p);
    return add(mul(ln2, (double)(e + ea)), add(ldx(x, 1), corr));
}
