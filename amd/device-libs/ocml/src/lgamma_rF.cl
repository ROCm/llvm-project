/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

#define FLOAT_SPECIALIZATION
#include "ep.h"

extern CONSTATTR float2 MATH_PRIVATE(epln)(float);

// This lgamma routine began with Sun's lgamma code from netlib.
// Their original copyright notice follows.
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

/* Reentrant version of the logarithm of the Gamma function
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
    float value;
    int sign;
};

// lgamma(x) for x > 0, as an unnormalized hi + lo
static float2
lgamma_pos(float x)
{
    const float a0  =  7.72156649015328655494e-02f;
    const float a1  =  3.22467033424113591611e-01f;
    const float a2  =  6.73523010531292681824e-02f;
    const float a3  =  2.05808084325167332806e-02f;
    const float a4  =  7.38555086081402883957e-03f;
    const float a5  =  2.89051383673415629091e-03f;
    const float a6  =  1.19270763183362067845e-03f;
    const float a7  =  5.10069792153511336608e-04f;
    const float a8  =  2.20862790713908385557e-04f;
    const float a9  =  1.08011567247583939954e-04f;
    const float a10 =  2.52144565451257326939e-05f;
    const float a11 =  4.48640949618915160150e-05f;
    const float tc  =  1.46163214496836224576e+00f;
    const float tcl =  0x1.ab5f20p-27f; // the polynomials are about the double tc
    const float tf  = -1.21486290535849611461e-01f;
    const float tt  = -3.63867699703950536541e-18f;
    const float t0  =  4.83836122723810047042e-01f;
    const float t1  = -1.47587722994593911752e-01f;
    const float t2  =  6.46249402391333854778e-02f;
    const float t3  = -3.27885410759859649565e-02f;
    const float t4  =  1.79706750811820387126e-02f;
    const float t5  = -1.03142241298341437450e-02f;
    const float t6  =  6.10053870246291332635e-03f;
    const float t7  = -3.68452016781138256760e-03f;
    const float t8  =  2.25964780900612472250e-03f;
    const float t9  = -1.40346469989232843813e-03f;
    const float t10 =  8.81081882437654011382e-04f;
    const float t11 = -5.38595305356740546715e-04f;
    const float t12 =  3.15632070903625950361e-04f;
    const float t13 = -3.12754168375120860518e-04f;
    const float t14 =  3.35529192635519073543e-04f;
    const float u0  = -7.72156649015328655494e-02f;
    const float u1  =  6.32827064025093366517e-01f;
    const float u2  =  1.45492250137234768737e+00f;
    const float u3  =  9.77717527963372745603e-01f;
    const float u4  =  2.28963728064692451092e-01f;
    const float u5  =  1.33810918536787660377e-02f;
    const float v1  =  2.45597793713041134822e+00f;
    const float v2  =  2.12848976379893395361e+00f;
    const float v3  =  7.69285150456672783825e-01f;
    const float v4  =  1.04222645593369134254e-01f;
    const float v5  =  3.21709242282423911810e-03f;
    const float s0  = -7.72156649015328655494e-02f;
    const float s1  =  2.14982415960608852501e-01f;
    const float s2  =  3.25778796408930981787e-01f;
    const float s3  =  1.46350472652464452805e-01f;
    const float s4  =  2.66422703033638609560e-02f;
    const float s5  =  1.84028451407337715652e-03f;
    const float s6  =  3.19475326584100867617e-05f;
    const float r1  =  1.39200533467621045958e+00f;
    const float r2  =  7.21935547567138069525e-01f;
    const float r3  =  1.71933865632803078993e-01f;
    const float r4  =  1.86459191715652901344e-02f;
    const float r5  =  7.77942496381893596434e-04f;
    const float r6  =  7.32668430744625636189e-06f;
    const float w0  =  4.18938533204672725052e-01f;
    const float w1  =  8.33333333333329678849e-02f;
    const float w2  = -2.77777777728775536470e-03f;
    const float w3  =  7.93650558643019558500e-04f;
    const float w4  = -5.95187557450339963135e-04f;
    const float w5  =  8.36339918996282139126e-04f;
    const float w6  = -1.63092934096575273989e-03f;

    // every branch needs log of one argument: x, or z in [2,8)
    float la = x;
    float ah = 0.0f, al = 0.0f, zl = 0.0f, w = 0.0f;

    if (x < 2.0f) {
        int i;
        bool c;
        float y, t;
        if( x <= 0.9f) { // lgamma(x) = lgamma(x+1)-log(x)
            y = 1.0f - x;
            i = 0;

            c = x < 0.7316f;
            t = (x - (tc - 1.0f)) - tcl;
            y = c ? t : y;
            i = c ? 1 : i;

            c = x < 0.23164f;
            y = c ? x : y;
            i = c ? 2 : i;
        } else {
            y = 2.0f - x;
            i = 0;

            c = x < 1.7316f;
            t = (x - tc) - tcl;
            y = c ? t : y;
            i = c ? 1 : i;

            c = x < 1.23f;
            t = x - 1.0f;
            y = c ? t : y;
            i = c ? 2 : i;
        }

        float z, w, p1, p2, p3, p;
        switch(i) {
        case 0:
            z = y * y;
            p1 = PE5(z, a10, a8, a6, a4, a2, a0);
            p2 = z * PE5(z, a11, a9, a7, a5, a3, a1);
            p = MATH_MAD(y, p1, p2);
            ah = MATH_MAD(y, -0.5f, p);
            break;
        case 1:
            z = y * y;
            w = z * y;
            p1 = PE4(w, t12, t9, t6, t3, t0);
            p2 = PE4(w, t13, t10, t7, t4, t1);
            p3 = PE4(w, t14, t11, t8, t5, t2);
            p = MATH_MAD(z, p1, -MATH_MAD(w, -MATH_MAD(y, p3, p2), tt));
            ah = tf + p;
            al = p - (ah - tf);
            break;
        case 2:
            p1 = y * PE5(y, u5, u4, u3, u2, u1, u0);
            p2 = PE5(y, v5, v4, v3, v2, v1, 1.0f);
            ah = MATH_MAD(y, -0.5f, MATH_FAST_DIV(p1, p2));
            break;
        }
    } else if (x < 8.0f) {  // 2 < x < 8
        int i = (int)x;
        float y = x - (float) i;
        float p = y * PE6(y, s6, s5, s4, s3, s2, s1, s0);
        float q = PE6(y, r6, r5, r4, r3, r2, r1, 1.0f);
        ah = MATH_MAD(y, 0.5f, MATH_FAST_DIV(p, q));

        // z = (y + 2)...(y + i - 1) as la + zl
        la = 1.0f;
        for (int k = 2; k < 7; ++k) {
            float yk = i > k ? y + (float)k : 1.0f;
            float t = la * yk;
            zl = BUILTIN_FMA_F32(zl, yk, BUILTIN_FMA_F32(la, yk, -t));
            la = t;
        }
    } else if (x < 0x1.0p+58f) { // 8 <= x < 2^58
        float z = MATH_FAST_RCP(x);
        float y = z * z;
        w = MATH_MAD(z, MATH_MAD(y, MATH_MAD(y, MATH_MAD(y, MATH_MAD(y, MATH_MAD(y, w6, w5), w4), w3), w2), w1), w0);
    }

    float2 l = MATH_PRIVATE(epln)(la);

    float ret;
    float retl;
    if (x <= 0.9f) {
        // -log(x) + ah as a TwoSum, since they partly cancel
        float sh = ah - l.hi;
        float sb = sh - ah;
        retl = ((ah - (sh - sb)) - (l.hi + sb)) + (al - l.lo);
        ret = sh;
    } else if (x < 2.0f) {
        ret = ah;
        retl = al;
    } else if (x < 8.0f) {
        ret = l.hi;
        retl = ah + BUILTIN_FMA_F32(zl, MATH_FAST_RCP(la), l.lo);
    } else if (x < 0x1.0p+58f) {
        // (x - 0.5)(log(x) - 1) + w; log(x).hi - 1 is exact for x >= 8,
        // and x*v, 0.5*v are exact in the fmas
        float v = l.hi - 1.0f;
        ret = BUILTIN_FMA_F32(x, v, BUILTIN_FMA_F32(x - 0.5f, l.lo, BUILTIN_FMA_F32(-0.5f, v, w)));
        retl = 0.0f;
    } else {
        // 2^58 <= x <= Inf
        ret = BUILTIN_FMA_F32(x, l.hi, -x);
        retl = 0.0f;
    }

    bool z = (x == 1.0f) | (x == 2.0f);
    ret = z ? 0.0f : ret;
    retl = z ? 0.0f : retl;
    if (!FINITE_ONLY_OPT()) {
        ret = BUILTIN_ISINF_F32(x) ? PINF_F32 : ret;
        retl = BUILTIN_ISINF_F32(x) ? 0.0f : retl;
    }

    return con(ret, retl);
}

// Negative-argument log|Gamma|.  Core [-12.5,-3] uses per-half certified
// polynomials from LGAMMAF_NEG_ZP; (-3,-2) is tiled by seg3; elsewhere (i.e.
// (-2,0) and below the core) an accurate reflection is used.  Coefficient
// tables live in lgammaF_table.h.

#define LGAMMAF_NEG_CELL_M_MAX 12

// lgamma = u*S(u) - log1p(u*rd0), u = (x-x0hi)-x0lo, rd0 + rd0l = 1/d0; near the
// pole (r <= -0.5) the log argument is (x+m)*rd0 so the singularity is carried
// by x+m.  This sets u*S as *th + *tl and the log argument as *la + *lal; the
// caller takes the log.
static void
lgamma_neg_zprep(__constant float *c, int deg, float x,
                 float x0hi, float x0lo, float rd0, float rd0l, float m,
                 float *th, float *tl, float *la, float *lal)
{
    // u + ue = (x - x0hi) - x0lo with x - x0hi exact (and either 0 or larger
    // than x0lo); S = sh + sl with the last step split
    float uh = x - x0hi;
    float u = uh - x0lo;
    float ue = (uh - u) - x0lo;
    float S = c[deg];
    for (int j = deg - 1; j >= 1; --j)
        S = MATH_MAD(S, u, c[j]);
    float sh = BUILTIN_FMA_F32(S, u, c[0]);
    float sl = BUILTIN_FMA_F32(S, u, c[0] - sh);
    *th = u * sh;
    *tl = BUILTIN_FMA_F32(u, sh, -*th) + BUILTIN_FMA_F32(u, sl, ue * sh);

    float rh = u * rd0;
    float rl = BUILTIN_FMA_F32(u, rd0, -rh) + BUILTIN_FMA_F32(u, rd0l, ue * rd0);
    if (rh > -0.5f) {
        *la = 1.0f + rh;
        *lal = ((1.0f - *la) + rh) + rl;
    } else {
        float v = x + m;
        *la = v * rd0;
        *lal = BUILTIN_FMA_F32(v, rd0l, BUILTIN_FMA_F32(v, rd0, -*la));
    }
}

// (-2.7,-2.47), between the two zeros of lgamma in (-3,-2):
// lgamma = (x - za)(x - zb)*Q(x + 2.59375), with both zeros factored out exactly
static float
lgamma_neg_seg3(float x)
{
    const float zah = -0x1.3a7fcap+1f;
    const float zal =  0x1.3fe0f2p-24f;
    const float zbh = -0x1.5fb41p+1f;
    const float zbl = -0x1.437b2p-24f;

    USE_TABLE(float, c, LGAMMAF_NEG_S3_ZZ);
    float t = x + 0x1.4cp+1f;
    float q = c[8];
    for (int j = 7; j >= 1; --j)
        q = MATH_MAD(q, t, c[j]);
    // last step split, with the low part of the constant term
    float qh = BUILTIN_FMA_F32(q, t, c[0]);
    float ql = BUILTIN_FMA_F32(q, t, c[0] - qh) + c[9];

    // (x - za)(x - zb) as ph + pl; x - zah and x - zbh are exact
    float ua = x - zah;
    float ub = x - zbh;
    float ph = ua * ub;
    float pl = BUILTIN_FMA_F32(ua, ub, -ph) - BUILTIN_FMA_F32(ua, zbl, zal * ub);
    return BUILTIN_FMA_F32(ph, qh, BUILTIN_FMA_F32(ph, ql, pl * qh));
}

// reflection fallback (out-of-core regions); sign comes from parity elsewhere
static float
lgamma_neg_reflect(float x, float2 lp)
{
    const float pi = 3.14159265358979311600e+00f;
    float t = MATH_MANGLE(sinpi)(x) * x;
    if (t == 0.0f)
        return PINF_F32;
    return MATH_MANGLE(log)(MATH_DIV(pi, BUILTIN_ABS_F32(t))) - (lp.hi + lp.lo);
}



static struct ret_t
lgamma_neg(float x, float2 lp)
{
    float fl = BUILTIN_FLOOR_F32(x);

    // Parity of floor(x) without a float->int cast: fl*0.5 is exact and whole
    // iff fl is even.  sign(Gamma) = +1 when floor(x) even, -1 when odd.
    float h = fl * 0.5f;
    int s = (h == BUILTIN_FLOOR_F32(h)) ? 1 : -1;

    // Except for the reflection and the middle (-3,-2) pieces, the result is
    // A - log(la + lal) with A = ah + al:
    //   (-1,0):   lgamma(x+1) - log(-x)
    //   [-2,-1):  lgamma(x+2) - log(x(x+1)); x + 1, x + 2 exact, no zeros here
    //   zeval:    u*S(u) - log of its argument
    bool z = true;
    float ah = lp.hi, al = lp.lo, la = -x, lal = 0.0f;

    float ret = 0.0f;
    if (x < -3.0f) {
        float m = BUILTIN_ROUND_F32(-x); // nearest pole (FP; narrowed only at index)
        int idx = -1;
        if (m <= (float)LGAMMAF_NEG_CELL_M_MAX) {
            int side = (x + m) > 0.0f ? 1 : 0;
            idx = (m == 3.0f) ? (side ? -1 : 0) : (2 * (int)m - 7 + side);
        }
        if (idx >= 0) {
            USE_TABLE(float, zp, LGAMMAF_NEG_ZP);
            __constant float *b = zp + idx * 11;
            lgamma_neg_zprep(b + 3, 7, x, b[0], b[1], b[2], 0.0f, m, &ah, &al, &la, &lal);
        } else {
            ret = lgamma_neg_reflect(x, lp);
            z = false;
        }
    } else if (x < -2.0f) {
        // (-3,-2): zb and the pole at -3, both zeros, za and the pole at -2
        if (x <= -2.7f) {
            USE_TABLE(float, c, LGAMMAF_NEG_S3_ZBP);
            lgamma_neg_zprep(c, 6, x, -0x1.5fb41p+1f, -0x1.437b2p-24f, 0x1.fb4c32p+1f, 0x1.638f46p-24f,
                             3.0f, &ah, &al, &la, &lal);
        } else if (x >= -2.47f) {
            USE_TABLE(float, c, LGAMMAF_NEG_S3_ZAP);
            lgamma_neg_zprep(c, 9, x, -0x1.3a7fcap+1f, 0x1.3fe0f2p-24f, -0x1.181286p+1f, -0x1.3a15c4p-24f,
                             2.0f, &ah, &al, &la, &lal);
        } else {
            ret = lgamma_neg_seg3(x);
            z = false;
        }
    } else if (x <= -1.0f) {
        float x1 = x + 1.0f;
        la = x * x1;
        lal = BUILTIN_FMA_F32(x, x1, -la);
    }

    if (z) {
        float2 l = MATH_PRIVATE(epln)(la);
        float sh = ah - l.hi;
        float sb = sh - ah;
        // lal is 0 when la can be subnormal (v_rcp flushes it)
        float rla = MATH_FAST_RCP(BUILTIN_MAX_F32(la, 0x1.0p-126f));
        ret = sh + (((ah - (sh - sb)) - (l.hi + sb)) + (al - BUILTIN_FMA_F32(lal, rla, l.lo)));
    }

    ret = (x == fl) ? PINF_F32 : ret;
    s = (x == fl) ? 0 : s;

    if (!FINITE_ONLY_OPT()) {
        ret = BUILTIN_ISINF_F32(x) ? PINF_F32 : ret;
        s = BUILTIN_ISNAN_F32(x) ? 0 : s;
    }

    struct ret_t result;
    result.value = ret;
    result.sign = s;
    return result;
}

CONSTATTR struct ret_t
MATH_PRIVATE(lgamma_r_impl)(float x)
{
    // one lgamma_pos for all paths: x, or what the reflection (-x) and
    // the (-1,0) and [-2,-1) recurrences (x + 1, x + 2) need
    float px = x > 0.0f ? x : (x > -1.0f ? x + 1.0f : (x >= -2.0f ? x + 2.0f : -x));
    float2 p = lgamma_pos(px);
    struct ret_t pos;
    pos.value = p.hi + p.lo;
    pos.sign = 1;
    return x > 0.0f ? pos : lgamma_neg(x, p);
}

float
MATH_MANGLE(lgamma_r)(float x, __private int *signp)
{
    struct ret_t ret = MATH_PRIVATE(lgamma_r_impl)(x);
    *signp = ret.sign;
    return ret.value;
}
