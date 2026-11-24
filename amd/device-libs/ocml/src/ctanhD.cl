/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

#define DOUBLE_SPECIALIZATION
#include "ep.h"

#include "trigredD.h"

extern CONSTATTR double2 MATH_PRIVATE(epexpep)(double2 z);

CONSTATTR double2
MATH_MANGLE(ctanh)(double2 z)
{
    double4 sc = MATH_PRIVATE(epsincos)(z.y);
    double2 cy = sc.lo;
    double2 sy = sc.hi;
#if defined EXTRA_ACCURACY
    // Keep low parts of products with sin(y) from underflowing for tiny y
    int ky = BUILTIN_ABS_F64(z.y) < 0x1.0p-800 ? 512 : 0;
    sy = ldx(sy, ky);
#endif
    double x = BUILTIN_ABS_F64(z.x);

    double rr, ri;
#if defined EXTRA_ACCURACY
    if (x < 22.0) {
#else
    if (x < 0x1.419ecb712c481p+4) {
#endif
        double2 e = MATH_PRIVATE(epexpep)(sub(x, con(0x1.62e42fefa39efp-1,0x1.abc9e3b39803fp-56)));
        double2 er = rcp(e);
        er = ldx(er, -2);
        double2 cx = fadd(e, er);
        double2 sx = x < 0x1.0p-27 ? con(x, 0.0) : fsub(e, er);

        double2 d = add(sqr(cy), sqr(sx));
        double2 rd = rcp(d);
#if defined EXTRA_ACCURACY
        // Same for sinh(x) with tiny x
        int kx = x < 0x1.0p-800 ? 512 : 0;
        rr = BUILTIN_COPYSIGN_F64(BUILTIN_FLDEXP_F64(mul(mul(cx, ldx(sx, kx)), rd).hi, -kx), z.x);
#else
        rr = BUILTIN_COPYSIGN_F64(mul(mul(cx, sx), rd).hi, z.x);
#endif
        ri = mul(mul(cy, sy), rd).hi;
#if defined EXTRA_ACCURACY
        ri = BUILTIN_FLDEXP_F64(ri, -ky);
#endif
    } else {
        rr = BUILTIN_COPYSIGN_F64(1.0, z.x);
#if defined EXTRA_ACCURACY
        double xc = x > 0x1.0p+10 ? 0x1.0p+10 : x;
        // 2^512 * e^(-2x) as a double-word, so its low part doesn't underflow
        double2 p = mul(cy, sy);
        double2 ex = MATH_PRIVATE(epexpep)(sub(-2.0 * xc, con(-0x1.62e42fefa39efp+8, -0x1.abc9e3b39803fp-47)));
        ri = BUILTIN_FLDEXP_F64(BUILTIN_FMA_F64(p.hi, ex.hi, BUILTIN_FMA_F64(p.lo, ex.hi, p.hi * ex.lo)), -510 - ky);
        ri = BUILTIN_COPYSIGN_F64(ri, p.hi);
#else
        ri = 4.0 * mul(cy, sy).hi * MATH_MANGLE(exp)(-2.0 * x);
#endif
    }

    if (!FINITE_ONLY_OPT()) {
        bool xn = BUILTIN_ISNAN_F64(x);
        bool yin = !BUILTIN_ISFINITE_F64(z.y);
        bool ni = BUILTIN_CLASS_F64(x, CLASS_PZER|CLASS_PSUB|CLASS_PNOR) & yin;
        rr = (ni | xn) ? QNAN_F64 : rr;
        ri = ni ? QNAN_F64 : ri;
        ri = (BUILTIN_ISINF_F64(x) & yin) ? 0.0 : ri;
        ri = (xn & (z.y == 0.0)) ? z.y : ri;
    }

    return (double2)(rr, ri);
}

