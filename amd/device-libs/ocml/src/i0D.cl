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
MATH_MANGLE(i0)(double x)
{
    x = BUILTIN_ABS_F64(x);

    double ret;

    if (x < 8.0) {
        double h = 0.5 * x;
        double2 t = sqr(h);
        double th = t.hi;
        double ph = MATH_MAD(th, MATH_MAD(th, MATH_MAD(th, MATH_MAD(th,
                    MATH_MAD(th, MATH_MAD(th, MATH_MAD(th, MATH_MAD(th,
                    MATH_MAD(th, MATH_MAD(th, MATH_MAD(th,
                        0x1.1273c9298fdc7p-88, 0x1.3f7d1d1a21335p-81), 0x1.43b3a13fc14b5p-73), 0x1.e630c80ada15dp-66),
                        0x1.41a63d85fefb0p-58), 0x1.69c95e76633f5p-51), 0x1.56019c0796a85p-44), 0x1.0b3131a919979p-37),
                        0x1.522a4404a6120p-31), 0x1.522a43f5b1e6ep-25), 0x1.02e85c089d85ep-19), 0x1.23456789aba0ep-14);
        double2 p = con(ph, 0.0);
        p = fadd(mul(p, t), 0x1.c71c71c71c739p-10);
        p = fadd(mul(p, t), 0x1.c71c71c71c71cp-6);
        p = fadd(mul(p, t), 0x1.0000000000000p-2);
        p = fadd(mul(p, t), 0x1.0000000000000p+0);
        ret = fadd(mul(t, p), 1.0).hi;
    } else {
        double t = MATH_RCP(x);
        double ph = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t,
                    MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t,
                    MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t,
                    MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t,
                    MATH_MAD(t, MATH_MAD(t,
                        0x1.cc967bacb549dp+49, -0x1.5ba7722975981p+50), 0x1.df0f836763276p+49), -0x1.9042a430f3f43p+48),
                        0x1.c630541c4f568p+46), -0x1.7366be5a9784fp+44), 0x1.c5669a48f574ep+41), -0x1.a664cac47f0eap+38),
                        0x1.308250566988cp+35), -0x1.56874c2ddb061p+31), 0x1.2da58968da2aap+27), -0x1.9faaa33f0d6bcp+22),
                        0x1.be0a8f2bc76ddp+17), -0x1.7123c68c3cb02p+12), 0x1.d402150cc72aap+6), -0x1.7a8ae85359520p+0),
                        0x1.bd7e0b6a753cdp-4), 0x1.6d6ce3774506dp-5), 0x1.debdd3d2f7cf9p-6);
        double2 p = con(ph, 0.0);
        p = fadd(mul(p, t), 0x1.cb94db8d452d5p-6);
        p = fadd(mul(p, t), 0x1.9884533daea3dp-5);
        p = fadd(mul(p, t), 0x1.9884533d4362fp-2);
        double xs = x - 709.0;
        double e1 = MATH_MANGLE(exp)(x > 709.0 ? xs : x);
        double e2 = x > 709.0 ? 0x1.d422d2be5dc9bp+1022 : 1.0;
        ret = omul(omul(mul(p, MATH_MANGLE(rsqrt)(x)), e1), e2).hi;
    }

    if  (!FINITE_ONLY_OPT()) {
        ret = BUILTIN_CLASS_F64(x, CLASS_PINF|CLASS_QNAN|CLASS_SNAN) ? x : ret;
    }

    return ret;
}

