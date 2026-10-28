/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"
#include "trigredD.h"

// H,L may alias A,B.
#define FSUM2(A, B, H, L) \
    do { double __s = (A) + (B); double __t = (B) - (__s - (A)); H = __s; L = __t; } while (0)

#define SUM2(A, B, H, L) \
    do { double __s = (A) + (B); double __aa = __s - (B); double __bb = __s - __aa; \
         double __da = (A) - __aa; double __db = (B) - __bb; H = __s; L = __da + __db; } while (0)

// v*(2/pi) as a 3-double expansion (F2>F1>F0): F2 the integer word, fraction below.
static void
eval3(double v, double *F2, double *F1, double *F0)
{
    double p0 = BUILTIN_AMDGPU_TRIG_PREOP_F64(v, 0);
    double p1 = BUILTIN_AMDGPU_TRIG_PREOP_F64(v, 1);
    double p2 = BUILTIN_AMDGPU_TRIG_PREOP_F64(v, 2);
    double A = BUILTIN_ABS_F64(v) >= 0x1.0p+945 ? BUILTIN_FLDEXP_F64(v, -128) : v;

    double p0h = p0 * A;
    double p0l = BUILTIN_FMA_F64(p0, A, -p0h);
    double p1h = p1 * A;
    double p1l = BUILTIN_FMA_F64(p1, A, -p1h);
    double p2h = p2 * A;

    double s1, c1;
    SUM2(p0l, p1h, s1, c1);
    double lo = (p1l + p2h) + c1;

    double f2 = p0h;
    double f1 = s1;
    double f0 = lo;
    FSUM2(f2, f1, f2, f1);
    FSUM2(f1, f0, f1, f0);
    *F2 = f2;
    *F1 = f1;
    *F0 = f0;
}

CONSTATTR struct epredret
MATH_PRIVATE(eptrigredlargeep)(double2 x)
{
    // Each part gets its own expansion; the two merge with a fixed carry chain.
    double a2, a1, a0, b2, b1, b0;
    eval3(x.hi, &a2, &a1, &a0);
    eval3(x.lo, &b2, &b1, &b0);

    double f2, c2, f1, c1, d;
    SUM2(a2, b2, f2, c2);
    SUM2(a1, b1, f1, c1);
    SUM2(f1, c2, f1, d);
    double f0 = ((a0 + b0) + c1) + d;

    FSUM2(f2, f1, f2, f1);
    FSUM2(f1, f0, f1, f0);

    // Strip integer bits above 2^2 (only the low 2, the quadrant, survive).
    f2 = BUILTIN_FLDEXP_F64(BUILTIN_FRACTION_F64_FIXUP(BUILTIN_FRACTION_F64_IMPL(BUILTIN_FLDEXP_F64(f2, -2)), x.hi), 2);
    f2 += f2 + f1 < 0.0 ? 4.0 : 0.0;

    int i = (int)(f2 + f1);
    f2 -= (double)i;
    FSUM2(f2, f1, f2, f1);
    FSUM2(f1, f0, f1, f0);

    int g = f2 >= 0.5;
    i += g;
    f2 -= g ? 1.0 : 0.0;
    FSUM2(f2, f1, f2, f1);

    const double pio2h = 0x1.921fb54442d18p+0;
    const double pio2t = 0x1.1a62633145c07p-54;
    double rh = f2 * pio2h;
    double rt = BUILTIN_FMA_F64(f1, pio2h, BUILTIN_FMA_F64(f2, pio2t, BUILTIN_FMA_F64(f2, pio2h, -rh)));
    FSUM2(rh, rt, rh, rt);

    struct epredret ret;
    ret.r.hi = rh;
    ret.r.lo = rt;
    ret.i = i & 0x3;
    return ret;
}
