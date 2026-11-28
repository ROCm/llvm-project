/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

#define DOUBLE_SPECIALIZATION
#include "ep.h"

// erfcx(x) as hi + lo for x >= 0
CONSTATTR double2
MATH_PRIVATE(erfcxep)(double x)
{
    double n = x - 4.0;
    double d = x + 4.0;
    double r = MATH_FAST_RCP(d);
    double q = n * r;
    double e = MATH_MAD(-q, x, MATH_MAD(q + 1.0, -4.0, x));
    q = BUILTIN_FMA_F64(r, e, q);
    
    double g = PE21(q, -0x1.15eb19bc60ea8p-29, -0x1.a047c341fece1p-27, 0x1.437e30a881907p-28,
                     0x1.e0ed222d4423fp-24, -0x1.a703db5fda3ep-25, -0x1.a6d5571ceca55p-21,
                     0x1.2436c05d90bf2p-20, 0x1.36bfbbf51ca4bp-18, -0x1.0a0095b7eedcdp-16,
                     -0x1.1ed94dd3dd5f9p-17, 0x1.4dce32d9ead5bp-13, -0x1.77d64bb9af1dap-12,
                     -0x1.a2beece46e0c2p-12, 0x1.63f2603106b16p-8, -0x1.625ed0917acadp-6,
                     0x1.e1a956436902cp-5, -0x1.00428c8ff856dp-3, 0x1.c0398cb6f0a3ap-3,
                     -0x1.47938f4c72d47p-2, 0x1.8d4ecb405c622p-2, -0x1.7d8f1c452474bp-2,
                     0x1.dd2c8b74febf6p-3);
    double p = MATH_MAD(q, g, g);

    double tx = x + x;
    d = 1.0 + tx;
    r = MATH_FAST_RCP(d);
    q = MATH_MAD(p, r, r);
    // residual 1 + p - q(1 + 2x); for x < 0.75, q >= 0.5 so 1 - q is exact
    bool sm = x < 0.75;
    e = ((sm ? 1.0 : p) - q) + MATH_MAD(-q, tx, sm ? p : 1.0);
    return con(q, r * e);
}
