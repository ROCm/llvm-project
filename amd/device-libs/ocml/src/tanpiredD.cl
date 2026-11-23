/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"
#include "trigpiredD.h"

CONSTATTR double
MATH_PRIVATE(tanpired)(double x, int i)
{
    double s = x * x;
    double t = PE13(s, 0x1.44eb4e945befep+32, -0x1.1b1cbc79c1bc3p+30, 0x1.c976879e6052dp+27,
                     -0x1.b4683789f7efap+22, 0x1.caf35c64b9bedp+21, 0x1.2f29496292dc4p+19,
                     0x1.48b9ee4204b27p+17, 0x1.45b5796997819p+15, 0x1.45f6c69bf8706p+13,
                     0x1.45f307cefe8e8p+11, 0x1.45f473ef3436bp+9, 0x1.45fff9b05abf5p+7,
                     0x1.466bc67763344p+5, 0x1.4abbce625bdf2p+3);
    // th + tl = tan(pi*x)
    double c = MATH_MAD(x * s, t, x * 0x1.1a62633145c07p-53);
    double th = MATH_MAD(x, 0x1.921fb54442d18p+1, c);
    double tl = MATH_MAD(x, 0x1.921fb54442d18p+1, -th) + c;

    // -1/(th + tl)
    double r = MATH_FAST_RCP(th);
    double tr = -MATH_MAD(MATH_MAD(-r, tl, MATH_MAD(-r, th, 1.0)), r, r);
    tr = x == 0.0 ? -PINF_F64 : tr;

    return i ? tr : th;
}

