/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"


#if defined EXTRA_ACCURACY
// a*c + b*d as hi + lo; under heavy cancellation the low part is not accurate
// enough, so fall back to Kahan's fma(a, c, b*d) + err(b*d), within 1.5 ulp
#define CP(A,B,C,D,L) ({ \
    float _a = A; \
    float _b = B; \
    float _c = C; \
    float _d = D; \
    float _ac = _a * _c; \
    float _bd = _b * _d; \
    float _bde = BUILTIN_FMA_F32(_b, _d, -_bd); \
    float _s = _ac + _bd; \
    float _t = _s - _ac; \
    float _l = ((_ac - (_s - _t)) + (_bd - _t)) + (BUILTIN_FMA_F32(_a, _c, -_ac) + _bde); \
    bool _k = BUILTIN_ABS_F32(_l) > 0x1.0p-4f * BUILTIN_ABS_F32(_s); \
    L = _k ? 0.0f : _l; \
    _k ? BUILTIN_FMA_F32(_a, _c, _bd) + _bde : _s; \
})
#else
// Kahan's a*c + b*d, within 1.5 ulp
#define CP(A,B,C,D,L) ({ \
    float _b = B; \
    float _d = D; \
    float _bd = _b * _d; \
    L = 0.0f; \
    BUILTIN_FMA_F32(A, C, _bd) + BUILTIN_FMA_F32(_b, _d, -_bd); \
})
#endif

// n/d for n = nh + nl, d = dh + dl, r ~ 1/dh; an overflowing quotient stays
// inf, and the sign of q keeps the sign of a zero from underflow
#define QUO(NH,NL,DH,DL,R) ({ \
    float _q = (NH) * (R); \
    float _e = BUILTIN_FMA_F32(-_q, DL, BUILTIN_FMA_F32(-_q, DH, NH) + (NL)); \
    BUILTIN_ISINF_F32(_q) ? _q : BUILTIN_COPYSIGN_F32(BUILTIN_FMA_F32(R, _e, _q), _q); \
})

CONSTATTR float2
MATH_MANGLE(cdiv)(float2 zn, float2 zd)
{
    float zdx = zd.x;
    float zdy = zd.y;
    bool g = BUILTIN_ABS_F32(zdx) > BUILTIN_ABS_F32(zdy);
    int ed = BUILTIN_FREXP_EXP_F32(g ? zdx : zdy);
    int en = BUILTIN_FREXP_EXP_F32(BUILTIN_MAX_F32(BUILTIN_ABS_F32(zn.x), BUILTIN_ABS_F32(zn.y)));
    int es1 = 126 - ed;
    int es2 = 126 - ed - ed;
    int es3 = 126 - ed - en;
    // the extra -2 keeps |zd|^2 <= 2^122 so that its reciprocal is normal
    int es = (BUILTIN_MIN_S32(BUILTIN_MIN_S32(es1, es2), es3) >> 1) - 2;

    zdx = BUILTIN_FLDEXP_F32(zdx, es);
    zdy = BUILTIN_FLDEXP_F32(zdy, es);
    float u = g ? zdx : zdy;
    float v = g ? zdy : zdx;
    // u^2 + v^2 as d2 + d2l; |u| >= |v|
    float uu = u * u;
    float vv = v * v;
    float d2 = uu + vv;
    float d2l = ((uu - d2) + vv) + (BUILTIN_FMA_F32(u, u, -uu) + BUILTIN_FMA_F32(v, v, -vv));

    zdx = BUILTIN_FLDEXP_F32(zdx, es);
    zdy = BUILTIN_FLDEXP_F32(zdy, es);
    float trl, til;
    float tr = CP(zn.x,  zn.y, zdx, zdy, trl);
    float ti = CP(zn.y, -zn.x, zdx, zdy, til);
    float r = MATH_FAST_RCP(d2);
    float rr = QUO(tr, trl, d2, d2l, r);
    float ri = QUO(ti, til, d2, d2l, r);

    if (!FINITE_ONLY_OPT()) {
        // Kahan's error term can make one component a spurious NaN
        if (BUILTIN_ISNAN_F32(rr) || BUILTIN_ISNAN_F32(ri)) {
            if (d2 == 0.0f && (!BUILTIN_ISNAN_F32(zn.x) || !BUILTIN_ISNAN_F32(zn.y))) {
                float i = BUILTIN_COPYSIGN_F32(PINF_F32, zd.x);
                rr = i * zn.x;
                ri = i * zn.y;
            } else if ((BUILTIN_ISINF_F32(zn.x) || BUILTIN_ISINF_F32(zn.y)) &&
                       (BUILTIN_ISFINITE_F32(zd.x) && BUILTIN_ISFINITE_F32(zd.y))) {
                float znx = BUILTIN_COPYSIGN_F32(BUILTIN_ISINF_F32(zn.x) ? 1.0f : 0.0f, zn.x);
                float zny = BUILTIN_COPYSIGN_F32(BUILTIN_ISINF_F32(zn.y) ? 1.0f : 0.0f, zn.y);
                rr = PINF_F32 * MATH_MAD(znx, zd.x,   zny * zd.y);
                ri = PINF_F32 * MATH_MAD(zny, zd.x,  -znx * zd.y);
            } else if ((BUILTIN_ISINF_F32(zd.x) || BUILTIN_ISINF_F32(zd.y)) &&
                       (BUILTIN_ISFINITE_F32(zn.x) && BUILTIN_ISFINITE_F32(zn.y))) {
                zdx = BUILTIN_COPYSIGN_F32(BUILTIN_ISINF_F32(zd.x) ? 1.0f : 0.0f, zd.x);
                zdy = BUILTIN_COPYSIGN_F32(BUILTIN_ISINF_F32(zd.y) ? 1.0f : 0.0f, zd.y);
                rr = 0.0f * MATH_MAD(zn.x, zdx,  zn.y * zdy);
                ri = 0.0f * MATH_MAD(zn.y, zdx, -zn.x * zdy);
            }
        }
    }

    return (float2)(rr, ri);
}

