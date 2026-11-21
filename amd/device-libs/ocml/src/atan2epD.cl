/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

CONSTATTR double
MATH_PRIVATE(atan2ep)(double2 y, double2 x)
{
    const double piby2_h = 0x1.921fb54442d18p+0;
    const double piby2_l = 0x1.1a62633145c07p-54;
    const double pi_h = 0x1.921fb54442d18p+1;
    const double pi_l = 0x1.1a62633145c07p-53;

    double ax = BUILTIN_ABS_F64(x.hi);
    double ay = BUILTIN_ABS_F64(y.hi);
    double axl = AS_DOUBLE(AS_ULONG(x.lo) ^ (AS_ULONG(x.hi) & SIGNBIT_DP64));
    double ayl = AS_DOUBLE(AS_ULONG(y.lo) ^ (AS_ULONG(y.hi) & SIGNBIT_DP64));
    bool sw = ay > ax;
    double u = sw ? ay : ax;
    double ul = sw ? ayl : axl;
    double v = sw ? ax : ay;
    double vl = sw ? axl : ayl;

    // Keep u, its reciprocal, and the division residual normal: scale small
    // u up into [0.5,1), and large u down only as far as 2^1020
    bool ok = BUILTIN_CLASS_F64(u, CLASS_PNOR|CLASS_PSUB);
    int eu = BUILTIN_FREXP_EXP_F64(u);
    int sc = eu < 0 ? -eu : (eu > 1020 ? 1020 - eu : 0);
    sc = ok ? sc : 0;

    // Scale a tiny v up so that no intermediate goes denormal
    int kv = (ok & (BUILTIN_FREXP_EXP_F64(v) - eu < -600)) ? 512 : 0;
    u = BUILTIN_FLDEXP_F64(u, sc);
    ul = BUILTIN_FLDEXP_F64(ul, sc);
    v = BUILTIN_FLDEXP_F64(v, sc + kv);
    vl = BUILTIN_FLDEXP_F64(vl, sc + kv);

    // r + d = (v + vl) / (u + ul)
    double ru = ok ? MATH_FAST_RCP(u) : 0.0;
    double r = v * ru;
    double d = BUILTIN_FMA_F64(-r, ul, BUILTIN_FMA_F64(-r, u, v) + vl) * ru;

    // a + c = atan(r + d)
    double t = r * r;
    double w = t * PE19(t, 0x1.ba404b5e68a13p-17, -0x1.3e260bd3237f4p-13, 0x1.b2bb069efb384p-11,
                          -0x1.7952daf56de9bp-9, 0x1.d6d43a595c56fp-8, -0x1.c6ea4a57d9582p-7,
                          0x1.67e295f08b19fp-6, -0x1.e9ae6fc27006ap-6, 0x1.2c15b5711927ap-5,
                          -0x1.59976e82d3ff0p-5, 0x1.82d5d6ef28734p-5, -0x1.ae5ce6a214619p-5,
                          0x1.e1bb48427b883p-5, -0x1.110e48b207f05p-4, 0x1.3b13657b87036p-4,
                          -0x1.745d119378e4fp-4, 0x1.c71c717e1913cp-4, -0x1.2492492376b7dp-3,
                          0x1.99999999952ccp-3, -0x1.5555555555523p-2);
    double p = r * w;
    double a = r + p;
    double c = ((r - a) + p) + BUILTIN_FMA_F64(r, w, -p);
    c = BUILTIN_FMA_F64(d, MATH_FAST_RCP(BUILTIN_FMA_F64(r, r, 1.0)), c);

    double rs = BUILTIN_FLDEXP_F64(a + c, -kv);
    a = BUILTIN_FLDEXP_F64(a, -kv);
    c = BUILTIN_FLDEXP_F64(c, -kv);

    double s = piby2_h - a;
    double sl = ((piby2_h - s) - a) + (piby2_l - c);
    a = sw ? s : a;
    c = sw ? sl : c;

    bool xneg = AS_INT2(x.hi).y < 0;
    s = pi_h - a;
    sl = ((pi_h - s) - a) + (pi_l - c);
    a = xneg ? s : a;
    c = xneg ? sl : c;

    double ret = ok ? ((sw | xneg) ? a + c : rs) : a;

    double tt = xneg ? pi_h : 0.0;
    ret = y.hi == 0.0 ? tt : ret;

    if (!FINITE_ONLY_OPT()) {
        // x and y are +- Inf
        tt = xneg ? 0x1.2d97c7f3321d2p+1 : 0x1.921fb54442d18p-1;
        ret = (BUILTIN_ISINF_F64(x.hi) & BUILTIN_ISINF_F64(y.hi)) ? tt : ret;

        // x or y is NaN
        ret = BUILTIN_ISUNORDERED_F64(x.hi, y.hi) ? QNAN_F64 : ret;
    }

    return BUILTIN_COPYSIGN_F64(ret, y.hi);
}
