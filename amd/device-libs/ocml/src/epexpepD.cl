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
MATH_PRIVATE(epexpep)(double2 x)
{
    double dn = BUILTIN_RINT_F64(x.hi * 0x1.71547652b82fep+0);
    double2 t = fsub(fsub(fadd(MATH_MAD(dn, -0x1.62e42fefa3000p-1, x.hi), x.lo), dn*0x1.3de6af278e000p-42), dn*0x1.9cc01f97b57a0p-83);

    double th = t.hi;
    double p = PE10(th, 0x1.1f1a2f03f0741p-29, 0x1.af5f1700d44d8p-26, 0x1.27e513387dfb3p-22, 0x1.71ddf8195f32ap-19,
                      0x1.a01a018e1147ap-16, 0x1.a01a01b00095bp-13, 0x1.6c16c16c1aaf7p-10,
                      0x1.111111110f659p-7, 0x1.555555555554dp-5, 0x1.5555555555559p-3, 0x1.0p-1);

    double2 r = fadd(1.0, fadd(t, mul(sqr(t), p)));

    return ldx(r, (int)dn);
}

