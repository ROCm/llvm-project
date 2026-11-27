/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"
#include "trigpiredD.h"

CONSTATTR struct scret
MATH_PRIVATE(sincospired)(double x)
{
    double t = x * x;

    double sx = PE5(t, 0x1.e3fe157a22716p-12, -0x1.e3009539b8259p-8, 0x1.5078300841253p-4, -0x1.32d2cce31fa29p-1,
                      0x1.466bc67758c6ap+1, -0x1.4abbce625be46p+2);
    sx = x * t * sx;
    sx = MATH_MAD(x, 0x1.921fb54442d18p+1, MATH_MAD(x, 0x1.1a62633145c07p-53, sx));

    double cx = PE6(t, -0x1.b167302e21c33p-14, 0x1.f9c89ca1d4f33p-10, -0x1.a6d1e7294bff9p-6, 0x1.e1f5067b90b37p-3,
                      -0x1.55d3c7e3c325bp+0, 0x1.03c1f081b5a67p+2, -0x1.3bd3cc9be45dep+2);
    cx = MATH_MAD(t, cx, 1.0);

    struct scret ret;
    ret.c = cx;
    ret.s = sx;
    return ret;
}

