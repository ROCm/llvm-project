/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

// This lgamma routine began with Sun's lgamma code from netlib.
// Their original copyright notice follows.
/* @(#)e_lgamma_r.c 1.3 95/01/18 */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunSoft, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 *
 */

/* __ieee754_lgamma_r(x, signgamp)
 * Reentrant version of the logarithm of the Gamma function
 * with user provide pointer for the sign of Gamma(x).
 *
 * Method:
 *   1. Argument Reduction for 0 < x <= 8
 *      Since gamma(1+s)=s*gamma(s), for x in [0,8], we may
 *      reduce x to a number in [1.5,2.5] by
 *              lgamma(1+s) = log(s) + lgamma(s)
 *      for example,
 *              lgamma(7.3) = log(6.3) + lgamma(6.3)
 *                          = log(6.3*5.3) + lgamma(5.3)
 *                          = log(6.3*5.3*4.3*3.3*2.3) + lgamma(2.3)
 *   2. Polynomial approximation of lgamma around its
 *      minimun ymin=1.461632144968362245 to maintain monotonicity.
 *      On [ymin-0.23, ymin+0.27] (i.e., [1.23164,1.73163]), use
 *              Let z = x-ymin;
 *              lgamma(x) = -1.214862905358496078218 + z^2*poly(z)
 *      where
 *              poly(z) is a 14 degree polynomial.
 *   2. Rational approximation in the primary interval [2,3]
 *      We use the following approximation:
 *              s = x-2.0;
 *              lgamma(x) = 0.5*s + s*P(s)/Q(s)
 *      with accuracy
 *              |P/Q - (lgamma(x)-0.5s)| < 2**-61.71
 *      Our algorithms are based on the following observation
 *
 *                             zeta(2)-1    2    zeta(3)-1    3
 * lgamma(2+s) = s*(1-Euler) + --------- * s  -  --------- * s  + ...
 *                                 2                 3
 *
 *      where Euler = 0.5771... is the Euler constant, which is very
 *      close to 0.5.
 *
 *   3. For x>=8, we have
 *      lgamma(x)~(x-0.5)log(x)-x+0.5*log(2pi)+1/(12x)-1/(360x**3)+....
 *      (better formula:
 *         lgamma(x)~(x-0.5)*(log(x)-1)-.5*(log(2pi)-1) + ...)
 *      Let z = 1/x, then we approximation
 *              f(z) = lgamma(x) - (x-0.5)(log(x)-1)
 *      by
 *                                  3       5             11
 *              w = w0 + w1*z + w2*z  + w3*z  + ... + w6*z
 *      where
 *              |w - f(z)| < 2**-58.74
 *
 *   4. For negative x, since (G is gamma function)
 *              -x*G(-x)*G(x) = pi/sin(pi*x),
 *      we have
 *              G(x) = pi/(sin(pi*x)*(-x)*G(-x))
 *      since G(-x) is positive, sign(G(x)) = sign(sin(pi*x)) for x<0
 *      Hence, for x<0, signgam = sign(sin(pi*x)) and
 *              lgamma(x) = log(|Gamma(x)|)
 *                        = log(pi/(|x*sin(pi*x)|)) - lgamma(-x);
 *      Note: one should avoid compute pi*(-x) directly in the
 *            computation of sin(pi*(-x)).
 *
 *   5. Special Cases
 *              lgamma(2+s) ~ s*(1-Euler) for tiny s
 *              lgamma(1)=lgamma(2)=0
 *              lgamma(x) ~ -log(x) for tiny x
 *              lgamma(0) = lgamma(inf) = inf
 *              lgamma(-integer) = +-inf
 *
 */


struct ret_t {
    double value;
    int sign;
};

static struct ret_t
lgamma_pos(double x)
{
    const double a0  =  7.72156649015328655494e-02;
    const double a1  =  3.22467033424113591611e-01;
    const double a2  =  6.73523010531292681824e-02;
    const double a3  =  2.05808084325167332806e-02;
    const double a4  =  7.38555086081402883957e-03;
    const double a5  =  2.89051383673415629091e-03;
    const double a6  =  1.19270763183362067845e-03;
    const double a7  =  5.10069792153511336608e-04;
    const double a8  =  2.20862790713908385557e-04;
    const double a9  =  1.08011567247583939954e-04;
    const double a10 =  2.52144565451257326939e-05;
    const double a11 =  4.48640949618915160150e-05;
    const double tc  =  1.46163214496836224576e+00;
    const double tf  = -1.21486290535849611461e-01;
    const double tt  = -3.63867699703950536541e-18;
    const double t0  =  4.83836122723810047042e-01;
    const double t1  = -1.47587722994593911752e-01;
    const double t2  =  6.46249402391333854778e-02;
    const double t3  = -3.27885410759859649565e-02;
    const double t4  =  1.79706750811820387126e-02;
    const double t5  = -1.03142241298341437450e-02;
    const double t6  =  6.10053870246291332635e-03;
    const double t7  = -3.68452016781138256760e-03;
    const double t8  =  2.25964780900612472250e-03;
    const double t9  = -1.40346469989232843813e-03;
    const double t10 =  8.81081882437654011382e-04;
    const double t11 = -5.38595305356740546715e-04;
    const double t12 =  3.15632070903625950361e-04;
    const double t13 = -3.12754168375120860518e-04;
    const double t14 =  3.35529192635519073543e-04;
    const double u0  = -7.72156649015328655494e-02;
    const double u1  =  6.32827064025093366517e-01;
    const double u2  =  1.45492250137234768737e+00;
    const double u3  =  9.77717527963372745603e-01;
    const double u4  =  2.28963728064692451092e-01;
    const double u5  =  1.33810918536787660377e-02;
    const double v1  =  2.45597793713041134822e+00;
    const double v2  =  2.12848976379893395361e+00;
    const double v3  =  7.69285150456672783825e-01;
    const double v4  =  1.04222645593369134254e-01;
    const double v5  =  3.21709242282423911810e-03;
    const double s0  = -7.72156649015328655494e-02;
    const double s1  =  2.14982415960608852501e-01;
    const double s2  =  3.25778796408930981787e-01;
    const double s3  =  1.46350472652464452805e-01;
    const double s4  =  2.66422703033638609560e-02;
    const double s5  =  1.84028451407337715652e-03;
    const double s6  =  3.19475326584100867617e-05;
    const double r1  =  1.39200533467621045958e+00;
    const double r2  =  7.21935547567138069525e-01;
    const double r3  =  1.71933865632803078993e-01;
    const double r4  =  1.86459191715652901344e-02;
    const double r5  =  7.77942496381893596434e-04;
    const double r6  =  7.32668430744625636189e-06;
    const double w0  =  4.18938533204672725052e-01;
    const double w1  =  8.33333333333329678849e-02;
    const double w2  = -2.77777777728775536470e-03;
    const double w3  =  7.93650558643019558500e-04;
    const double w4  = -5.95187557450339963135e-04;
    const double w5  =  8.36339918996282139126e-04;
    const double w6  = -1.63092934096575273989e-03;
    const double z1  = -0x1.2788cfc6fb619p-1;
    const double z2  =  0x1.a51a6625307d3p-1;
    const double z3  = -0x1.9a4d55beab2d7p-2;
    const double z4  =  0x1.151322ac7d848p-2;
    const double z5  = -0x1.a8b9c17aa6149p-3;

    double ret;

    if (x < 0x1.0p-8) {
        ret = PE5(x, z5, z4, z3, z2, z1, -MATH_MANGLE(log)(x));
    } else if (x < 2.0) {
        int i;
        bool c;
        double y, t;
        if (x <= 0x1.cccccp-1) { // x < 0.9 : lgamma(x) = lgamma(x+1)-log(x)
            ret = -MATH_MANGLE(log)(x);

            y = 1.0 - x;
            i = 0;

            c = x < 0x1.76944p-1; // x < 0.7316
            t = x - (tc - 1.0);
            y = c ? t : y;
            i = c ? 1 : i;

            c = x < 0x1.da661p-3; // x < .2316
            y = c ? x : y;
            i = c ? 2 : i;
        } else {
            ret = 0.0;

            y = 2.0 - x;
            i = 0;

            c = x < 0x1.bb4c3p+0; // x < 1.7316
            t = x - tc;
            y = c ? t : y;
            i = c ? 1 : i;

            c = x < 0x1.3b4c4p+0; // x < 1.2316
            t = x - 1.0;
            y = c ? t : y;
            i = c ? 2 : i;
        }

        double w, z, p, p1, p2, p3;
        switch(i) {
        case 0:
            z = y*y;
            p1 = PE5(z, a10, a8, a6, a4, a2, a0);
            p2 = z * PE5(z, a11, a9, a7, a5, a3, a1);
            p = MATH_MAD(y, p1, p2);
            ret += MATH_MAD(y, -0.5, p);
            break;
        case 1:
            z = y*y;
            w = z*y;
            p1 = PE4(w, t12, t9, t6, t3, t0);
            p2 = PE4(w, t13, t10, t7, t4, t1);
            p3 = PE4(w, t14, t11, t8, t5, t2);
            p = MATH_MAD(z, p1, -MATH_MAD(w, -MATH_MAD(y, p3,p2), tt));
            ret += tf + p;
            break;
        case 2:
            p1 = y * PE5(y, u5, u4, u3, u2, u1, u0);
            p2 = PE5(y, v5, v4, v3, v2, v1, 1.0);
            ret += MATH_MAD(y, -0.5, MATH_DIV(p1, p2));
            break;
        }
    } else if (x < 8.0) { // 2 < x < 8
        int i = (int)x;
        double y = x - (double)i;
        double p = y * PE6(y, s6, s5, s4, s3, s2, s1, s0);
        double q = PE6(y, r6, r5, r4, r3, r2, r1, 1.0);
        ret = MATH_MAD(y, 0.5, MATH_DIV(p, q));

        double y2 = y + 2.0;
        double y3 = y + 3.0;
        double y4 = y + 4.0;
        double y5 = y + 5.0;
        double y6 = y + 6.0;

        double z = 1.0;
        z *= i > 2 ? y2 : 1.0;
        z *= i > 3 ? y3 : 1.0;
        z *= i > 4 ? y4 : 1.0;
        z *= i > 5 ? y5 : 1.0;
        z *= i > 6 ? y6 : 1.0;

        ret += MATH_MANGLE(log)(z);
    } else if (x < 0x1p+58) { // 8 <= x < 2^58
        double z = MATH_RCP(x);
        double y = z*z;
        double w = MATH_MAD(z, MATH_MAD(y, MATH_MAD(y, MATH_MAD(y, MATH_MAD(y, MATH_MAD(y, w6, w5), w4), w3), w2), w1), w0);
        ret = MATH_MAD(x - 0.5, MATH_MANGLE(log)(x) - 1.0, w);
    } else { // 2^58 <= x <= Inf
        ret = MATH_MAD(x, MATH_MANGLE(log)(x), -x);
    }

    ret = (x == 1.0 | x == 2.0) ? 0.0 : ret;
    if (!FINITE_ONLY_OPT())
        ret = BUILTIN_ISINF_F64(x) ? PINF_F64 : ret;

    struct ret_t result;
    result.value = ret;
    result.sign = 1;
    return result;
}

// Negative-argument log|Gamma|.  Core [-21.5,-3] uses per-half certified
// polynomials from LGAMMA_NEG_ZP; (-3,-2) is tiled by seg3; elsewhere (i.e.
// (-2,0) and below the core) an accurate reflection is used.  Coefficient
// tables live in lgammaD_table.h.

#define LGAMMA_NEG_CELL_M_MAX 21

// lgamma = u*S(u) - log1p(u*rd0), u = (x-x0hi)-x0lo, rd0 = 1/d0; near the pole
// (r <= -0.5) switch to -log|(x+m)*rd0| so the singularity is carried by x+m.
static double
lgamma_neg_zeval(__constant double *c, int deg, double x,
                 double x0hi, double x0lo, double rd0, double m)
{
    double u = (x - x0hi) - x0lo;
    double S = 0.0;
    for (int j = deg; j >= 0; --j)
        S = MATH_MAD(S, u, c[j]);
    double r = u * rd0;
    double sing;
    if (r > -0.5) {
        sing = -MATH_MANGLE(log1p)(r);
    } else {
        double v = x + m;
        if (v == 0.0)
            return PINF_F64;
        sing = -MATH_MANGLE(log)(BUILTIN_ABS_F64(v * rd0));
    }
    return MATH_MAD(u, S, sing);
}

// lgamma = u*Q(u), u = (x-x0hi)-x0lo : factors the zero exactly, no log term.
static double
lgamma_neg_zfeval(__constant double *c, int deg, double x,
                  double x0hi, double x0lo)
{
    double u = (x - x0hi) - x0lo;
    double Q = 0.0;
    for (int j = deg; j >= 0; --j)
        Q = MATH_MAD(Q, u, c[j]);
    return u * Q;
}

// plain polynomial about c0 (the -2.5 valley in the (-3,-2) dead zone)
static double
lgamma_neg_deval(__constant double *c, int deg, double x, double c0)
{
    double t = x - c0;
    double S = 0.0;
    for (int j = deg; j >= 0; --j)
        S = MATH_MAD(S, t, c[j]);
    return S;
}

// (-3,-2) reflection-free 5-piece tiling
static double
lgamma_neg_seg3(double x)
{
    const double zb_x0hi = -2.7476826467274127;
    const double zb_x0lo =  9.055340329338315e-17;
    const double za_x0hi = -2.4570247382208006;
    const double za_x0lo = -3.7075610815513266e-17;

    if (x <= -2.88) {
        USE_TABLE(double, c, LGAMMA_NEG_S3_ZBP);
        return lgamma_neg_zeval(c, 9, x, zb_x0hi, zb_x0lo, 0x1.fb4c32b1c7a37p+1, 3.0);
    }
    if (x <= -2.7) {
        USE_TABLE(double, c, LGAMMA_NEG_S3_ZFB);
        return lgamma_neg_zfeval(c, 23, x, zb_x0hi, zb_x0lo);
    }
    if (x < -2.47) {
        USE_TABLE(double, c, LGAMMA_NEG_S3_DD);
        return lgamma_neg_deval(c, 19, x, -2.5);
    }
    if (x < -2.36) {
        USE_TABLE(double, c, LGAMMA_NEG_S3_ZFA);
        return lgamma_neg_zfeval(c, 12, x, za_x0hi, za_x0lo);
    }
    USE_TABLE(double, c, LGAMMA_NEG_S3_ZAP);
    return lgamma_neg_zeval(c, 16, x, za_x0hi, za_x0lo, -0x1.1812869d0ae27p+1, 2.0);
}

// reflection fallback (out-of-core regions); sign comes from parity elsewhere
static double
lgamma_neg_reflect(double x)
{
    const double pi = 3.14159265358979311600e+00;
    double t = MATH_MANGLE(sinpi)(x) * x;
    if (t == 0.0)
        return PINF_F64;
    return MATH_MANGLE(log)(MATH_DIV(pi, BUILTIN_ABS_F64(t))) - lgamma_pos(-x).value;
}

// (-1,0) recurrence lgamma(x) = lgamma(x+1) - log|x|; the shift lands in (0,1),
// with the correction dropped below the small-x bound where -log|x| is exact.
static double
lgamma_neg_recur(double x)
{
    double r = -MATH_MANGLE(log)(-x);
    if (x <= -0x1.0p-46)
        r += lgamma_pos(x + 1.0).value;
    return r;
}

static struct ret_t
lgamma_neg(double x)
{
    double fl = BUILTIN_FLOOR_F64(x);

    // Parity of floor(x) without a float->int cast: fl*0.5 is exact and whole
    // iff fl is even.  sign(Gamma) = +1 when floor(x) even, -1 when odd.
    double h = fl * 0.5;
    int s = (h == BUILTIN_FLOOR_F64(h)) ? 1 : -1;

    double ret;
    if (x < -3.0) {
        double m = BUILTIN_ROUND_F64(-x); // nearest pole (FP; narrowed only at index)
        int idx = -1;
        if (m <= (double)LGAMMA_NEG_CELL_M_MAX) {
            int side = (x + m) > 0.0 ? 1 : 0;
            idx = (m == 3.0) ? (side ? -1 : 0) : (2 * (int)m - 7 + side);
        }
        if (idx >= 0) {
            USE_TABLE(double, zp, LGAMMA_NEG_ZP);
            __constant double *b = zp + idx * 22;
            ret = lgamma_neg_zeval(b + 3, 18, x, b[0], b[1], b[2], m);
        } else {
            ret = lgamma_neg_reflect(x);
        }
    } else if (x < -2.0) {
        ret = lgamma_neg_seg3(x);
    } else if (x > -1.0) {
        ret = lgamma_neg_recur(x);
    } else {
        ret = lgamma_neg_reflect(x);
    }

    s = (x == fl) ? 0 : s;

    if (!FINITE_ONLY_OPT()) {
        ret = BUILTIN_ISINF_F64(x) ? PINF_F64 : ret;
        s = BUILTIN_ISNAN_F64(x) ? 0 : s;
    }

    struct ret_t result;
    result.value = ret;
    result.sign = s;
    return result;
}

CONSTATTR struct ret_t
MATH_PRIVATE(lgamma_r_impl)(double x)
{
    return x > 0.0 ? lgamma_pos(x) : lgamma_neg(x);
}


double
MATH_MANGLE(lgamma_r)(double x, __private int *signp)
{
    struct ret_t ret = MATH_PRIVATE(lgamma_r_impl)(x);
    *signp = ret.sign;
    return ret.value;
}
