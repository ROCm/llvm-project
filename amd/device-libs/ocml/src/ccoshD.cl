/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

#define DOUBLE_SPECIALIZATION
#include "ep.h"

#if defined EXTRA_ACCURACY
#include "trigredD.h"
#endif

extern CONSTATTR double2 MATH_PRIVATE(epexpep)(double2 z);

CONSTATTR double2
MATH_MANGLE(ccosh)(double2 z)
{
    double x = BUILTIN_ABS_F64(z.x);
    double2 e = MATH_PRIVATE(epexpep)(sub(x, con(0x1.62e42fefa39efp+0,0x1.abc9e3b39803fp-55)));
    double2 er = rcp(e);
    er = ldx(er, -4);
    double2 cx = fadd(e, er);
    double2 sx = fsub(e, er);
#if defined EXTRA_ACCURACY
    double4 sc = MATH_PRIVATE(epsincos)(z.y);
    // Fold the final scaling by 2 into cos and sin so that the products
    // don't go denormal for a normal result
    bool s = x >= 0x1.0p-27;
    double2 sxs = s ? sx : con(x, 0.0);
    double2 cy = ldx(sc.lo, 1);

    // Also keep the low part of sinh(x)*sin(y) from underflowing for tiny y
    int ky = BUILTIN_ABS_F64(z.y) < 0x1.0p-900 ? 512 : 0;
    double2 sy = ldx(sc.hi, s + ky);
    double rr = BUILTIN_FMA_F64(cx.hi, cy.hi, BUILTIN_FMA_F64(cx.lo, cy.hi, cx.hi * cy.lo));
    double ri = BUILTIN_FMA_F64(sxs.hi, sy.hi, BUILTIN_FMA_F64(sxs.lo, sy.hi, sxs.hi * sy.lo));

    if (!FINITE_ONLY_OPT()) {
        bool b = x >= 0x1.6395a2079b70cp+9;
        rr = b ? PINF_F64 * cy.hi : rr;
        ri = b ? PINF_F64 * sy.hi : ri;
    }

    ri = BUILTIN_FLDEXP_F64(AS_DOUBLE(AS_ULONG(ri) ^ (AS_ULONG(z.x) & SIGNBIT_DP64)), -ky);

#else
    double cy;
    double sy = MATH_MANGLE(sincos)(z.y, &cy);

    double cxhi, sxhi;
    if (FINITE_ONLY_OPT()) {
        cxhi = cx.hi;
        sxhi = sx.hi;
    } else {
        bool b = x >= 0x1.6395a2079b70cp+9;
        cxhi = b ? PINF_F64 : cx.hi;
        sxhi = b ? PINF_F64 : sx.hi;
    }

    double rr = BUILTIN_FLDEXP_F64(cxhi * cy, 1);
    bool s = x >= 0x1.0p-27;
    double ri = BUILTIN_FLDEXP_F64(BUILTIN_COPYSIGN_F64(s ? sxhi : x, z.x) * sy, s);

#endif

    if (!FINITE_ONLY_OPT()) {
        ri = ((x == 0.0) | (z.y == 0.0)) ? BUILTIN_COPYSIGN_F64(0.0, z.y)  : ri;
        rr = (BUILTIN_ISINF_F64(x) &
              BUILTIN_CLASS_F64(z.y, CLASS_PINF|CLASS_NINF|CLASS_PZER|CLASS_NZER|CLASS_QNAN|CLASS_SNAN)) ? x : rr;
    }

    return (double2)(rr, ri);
}

