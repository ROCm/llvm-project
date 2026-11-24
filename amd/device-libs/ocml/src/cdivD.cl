/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"


#if defined EXTRA_ACCURACY
// a*c + b*d as hi + lo; under heavy cancellation the low part is not accurate
// enough, so fall back to Kahan's fma(a, c, b*d) + err(b*d), within 1.5 ulp
#define CP(A,B,C,D,L) ({ \
    double _a = A; \
    double _b = B; \
    double _c = C; \
    double _d = D; \
    double _ac = _a * _c; \
    double _bd = _b * _d; \
    double _bde = BUILTIN_FMA_F64(_b, _d, -_bd); \
    double _s = _ac + _bd; \
    double _t = _s - _ac; \
    double _l = ((_ac - (_s - _t)) + (_bd - _t)) + (BUILTIN_FMA_F64(_a, _c, -_ac) + _bde); \
    bool _k = BUILTIN_ABS_F64(_l) > 0x1.0p-4 * BUILTIN_ABS_F64(_s); \
    L = _k ? 0.0 : _l; \
    _k ? BUILTIN_FMA_F64(_a, _c, _bd) + _bde : _s; \
})
#else
// Kahan's a*c + b*d, within 1.5 ulp
#define CP(A,B,C,D,L) ({ \
    double _b = B; \
    double _d = D; \
    double _bd = _b * _d; \
    L = 0.0; \
    BUILTIN_FMA_F64(A, C, _bd) + BUILTIN_FMA_F64(_b, _d, -_bd); \
})
#endif

// n/d for n = nh + nl, d = dh + dl, r ~ 1/dh; an overflowing quotient stays
// inf, and the sign of q keeps the sign of a zero from underflow
#define QUO(NH,NL,DH,DL,R) ({ \
    double _q = (NH) * (R); \
    double _e = BUILTIN_FMA_F64(-_q, DL, BUILTIN_FMA_F64(-_q, DH, NH) + (NL)); \
    BUILTIN_ISINF_F64(_q) ? _q : BUILTIN_COPYSIGN_F64(BUILTIN_FMA_F64(R, _e, _q), _q); \
})

CONSTATTR double2
MATH_MANGLE(cdiv)(double2 zn, double2 zd)
{
    double zdx = zd.x;
    double zdy = zd.y;
    bool g = BUILTIN_ABS_F64(zdx) > BUILTIN_ABS_F64(zdy);
    int ed = BUILTIN_FREXP_EXP_F64(g ? zdx : zdy);
    int en = BUILTIN_FREXP_EXP_F64(BUILTIN_MAX_F64(BUILTIN_ABS_F64(zn.x), BUILTIN_ABS_F64(zn.y)));
    int es1 = 1022 - ed;
    int es2 = 1022 - ed - ed;
    int es3 = 1022 - ed - en;
    // the extra -2 keeps |zd|^2 <= 2^1018 so that its reciprocal is normal
    int es = (BUILTIN_MIN_S32(BUILTIN_MIN_S32(es1, es2), es3) >> 1) - 2;

    zdx = BUILTIN_FLDEXP_F64(zdx, es);
    zdy = BUILTIN_FLDEXP_F64(zdy, es);
    double u = g ? zdx : zdy;
    double v = g ? zdy : zdx;
    // u^2 + v^2 as d2 + d2l; |u| >= |v|
    double uu = u * u;
    double vv = v * v;
    double d2 = uu + vv;
    double d2l = ((uu - d2) + vv) + (BUILTIN_FMA_F64(u, u, -uu) + BUILTIN_FMA_F64(v, v, -vv));

    zdx = BUILTIN_FLDEXP_F64(zdx, es);
    zdy = BUILTIN_FLDEXP_F64(zdy, es);
    double trl, til;
    double tr = CP(zn.x,  zn.y, zdx, zdy, trl);
    double ti = CP(zn.y, -zn.x, zdx, zdy, til);
    double r = MATH_FAST_RCP(d2);
    double rr = QUO(tr, trl, d2, d2l, r);
    double ri = QUO(ti, til, d2, d2l, r);

    if (!FINITE_ONLY_OPT()) {
        // Kahan's error term can make one component a spurious NaN
        if (BUILTIN_ISNAN_F64(rr) || BUILTIN_ISNAN_F64(ri)) {
            if (d2 == 0.0 && (!BUILTIN_ISNAN_F64(zn.x) || !BUILTIN_ISNAN_F64(zn.y))) {
                double i = BUILTIN_COPYSIGN_F64(PINF_F64, zd.x);
                rr = i * zn.x;
                ri = i * zn.y;
            } else if ((BUILTIN_ISINF_F64(zn.x) || BUILTIN_ISINF_F64(zn.y)) &&
                       (BUILTIN_ISFINITE_F64(zd.x) && BUILTIN_ISFINITE_F64(zd.y))) {
                double znx = BUILTIN_COPYSIGN_F64(BUILTIN_ISINF_F64(zn.x) ? 1.0 : 0.0, zn.x);
                double zny = BUILTIN_COPYSIGN_F64(BUILTIN_ISINF_F64(zn.y) ? 1.0 : 0.0, zn.y);
                rr = PINF_F64 * MATH_MAD(znx, zd.x,   zny * zd.y);
                ri = PINF_F64 * MATH_MAD(zny, zd.x,  -znx * zd.y);
            } else if ((BUILTIN_ISINF_F64(zd.x) || BUILTIN_ISINF_F64(zd.y)) &&
                       (BUILTIN_ISFINITE_F64(zn.x) && BUILTIN_ISFINITE_F64(zn.y))) {
                zdx = BUILTIN_COPYSIGN_F64(BUILTIN_ISINF_F64(zd.x) ? 1.0 : 0.0, zd.x);
                zdy = BUILTIN_COPYSIGN_F64(BUILTIN_ISINF_F64(zd.y) ? 1.0 : 0.0, zd.y);
                rr = 0.0 * MATH_MAD(zn.x, zdx,  zn.y * zdy);
                ri = 0.0 * MATH_MAD(zn.y, zdx, -zn.x * zdy);
            }
        }
    }

    return (double2)(rr, ri);
}

