/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

#define DOUBLE_SPECIALIZATION
#include "ep.h"

CONSTATTR double2
MATH_PRIVATE(epln)(double a)
{
    int a_exp;
    double m = BUILTIN_FREXP_F64(a, &a_exp);
    int b = m < (2.0/3.0);
    m = BUILTIN_FLDEXP_F64(m, b);
    int e = a_exp - b;


    // x = (m - 1)/(m + 1): m - 1 is exact, and 1 + m is a Fast2Sum since
    // 1 has the larger (or equal) exponent
    double dh = 1.0 + m;
    double dl = (1.0 - dh) + m;
    double nh = m - 1.0;
    double rc = MATH_FAST_RCP(dh);
    double xh = nh * rc;
    double xl = BUILTIN_FMA_F64(-xh, dl, BUILTIN_FMA_F64(-xh, dh, nh)) * rc;

    // s = x^2, q = c3 + s*p, c = x^3, t = c*q, each as hi + lo
    double sh = xh * xh;
    double sl = BUILTIN_FMA_F64(2.0 * xh, xl, BUILTIN_FMA_F64(xh, xh, -sh));
    double p = PE8(sh, 0x1.dee674222de17p-4, 0x1.a6564968915a9p-4, 0x1.e25e43abe935ap-4, 0x1.110ef47e6c9c2p-3,
                      0x1.3b13bcfa74449p-3, 0x1.745d171bf3c30p-3, 0x1.c71c71c7792cep-3,
                      0x1.24924924920dap-2, 0x1.999999999999cp-2);
    double qh = BUILTIN_FMA_F64(sh, p, 0x1.5555555555555p-1);
    double ql = BUILTIN_FMA_F64(sh, p, 0x1.5555555555555p-1 - qh) + BUILTIN_FMA_F64(sl, p, 0x1.543b0d5df274dp-55);
    double ch = xh * sh;
    double cl = BUILTIN_FMA_F64(xh, sh, -ch) + BUILTIN_FMA_F64(xh, sl, xl * sh);
    double th = ch * qh;
    double tl = BUILTIN_FMA_F64(ch, qh, -th) + BUILTIN_FMA_F64(ch, ql, cl * qh);

    // ln(2)*e + 2*x + t: e*ln2_hi is exact, and each Fast2Sum adds the smaller term
    double fe = (double)e;
    double eh = fe * 0x1.62e42fefa3800p-1;
    double x2 = 2.0 * xh;
    double ah = eh + x2;
    double al = x2 - (ah - eh);
    double bh = ah + th;
    double bl = th - (bh - ah);
    double lo = ((al + bl) + BUILTIN_FMA_F64(fe, 0x1.ef35793c76730p-45, 2.0 * xl)) + tl;
    double rh = bh + lo;
    double2 r = con(rh, lo - (rh - bh));

    return r;
}

