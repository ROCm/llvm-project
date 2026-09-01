/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

#define DOUBLE_SPECIALIZATION
#include "ep.h"

double
MATH_MANGLE(i1)(double x)
{
    double a = BUILTIN_ABS_F64(x);

    double ret;

    if (a < 8.0) {
        a *= 0.5;
        double2 t = sqr(a);
        double th = t.hi;
        double ph = MATH_MAD(th, MATH_MAD(th, MATH_MAD(th, MATH_MAD(th,
                    MATH_MAD(th, MATH_MAD(th, MATH_MAD(th, MATH_MAD(th,
                    MATH_MAD(th, MATH_MAD(th,
                        0x1.1778f166c4fedp-84, 0x1.274d6a81878dcp-77), 0x1.1c07d39c30609p-69), 0x1.8adacd32bc600p-62),
                        0x1.e27fdbc5ec4c9p-55), 0x1.f1743bd663f3cp-48), 0x1.ab820eb81f703p-41), 0x1.2c975749073d9p-34),
                        0x1.522a440f20bddp-28), 0x1.27e4fb7679e36p-22), 0x1.845c8a0cf498ap-17);
        double2 p = con(ph, 0.0);
        p = fadd(mul(p, t), 0x1.6c16c16c1634ep-12);
        p = fadd(mul(p, t), 0x1.c71c71c71c773p-8);
        p = fadd(mul(p, t), 0x1.5555555555554p-4);
        p = fadd(mul(p, t), 0x1.0000000000000p-1);
        ret = mul(a, fadd(mul(t, p), 1.0)).hi;
    } else {
        double t = MATH_RCP(a);
        double ph = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t,
                    MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t,
                    MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t,
                    MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t,
                    MATH_MAD(t, MATH_MAD(t,
                        -0x1.c9d8d43214423p+49, 0x1.5c072e12fb4bap+50), -0x1.e26cff438b6f6p+49), 0x1.952224c61a221p+48),
                        -0x1.cdc7c873cf435p+46), 0x1.7b1e32a15fb86p+44), -0x1.d07dbd6696f1cp+41), 0x1.b227934f2ced2p+38),
                        -0x1.39f23e6685444p+35), 0x1.6229383f6f890p+31), -0x1.38bf1ceeee865p+27), 0x1.b01a348b749b8p+22),
                        -0x1.d0e043ef0916ap+17), 0x1.81b06f82cfbacp+12), -0x1.ea879b2a6508bp+6), 0x1.85cffc8d54f52p+0),
                        -0x1.09f107ee0f7e2p-3), -0x1.d61631539fb0dp-5), -0x1.4f1e01d904ebap-5);
        double2 p = con(ph, 0.0);
        p = fadd(mul(p, t), -0x1.7efc0ced79c58p-5);
        p = fadd(mul(p, t), -0x1.32633e6e0f07ap-3);
        p = fadd(mul(p, t), 0x1.9884533d43674p-2);

        double as = a - 709.0;
        double e1 = MATH_MANGLE(exp)(a > 709.0 ? as : a);
        double e2 = a > 709.0 ? 0x1.d422d2be5dc9bp+1022 : 1.0;
        ret = omul(omul(mul(p, MATH_MANGLE(rsqrt)(a)), e1), e2).hi;
    }

    if  (!FINITE_ONLY_OPT()) {
        ret = BUILTIN_CLASS_F64(a, CLASS_PINF|CLASS_QNAN|CLASS_SNAN) ? a : ret;
    }

    return BUILTIN_COPYSIGN_F64(ret, x);
}

