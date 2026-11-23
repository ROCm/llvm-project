/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

CONSTATTR double
MATH_PRIVATE(atanpired)(double v)
{
    double t = v * v;
    // v*(1/pi) + v*w(t): the double 1/pi times v is exact in the fma, and w's
    // constant term carries the low part of 1/pi
    double w = PE20(t, 0x1.175a1ed2de5acp-18, -0x1.924313ccac6c3p-15, 0x1.1314c323a1abfp-12,
                     -0x1.ddf7b938c2c16p-11, 0x1.2a8062caca309p-9, -0x1.20b29fa1b0215p-8,
                     0x1.c9357ba2aa92bp-8, -0x1.374ee627db8a9p-7, 0x1.7dca88a9c7553p-7,
                     -0x1.b7deadf52314dp-7, 0x1.ec78de82b68bep-7, -0x1.11f7b72008e82p-6,
                     0x1.32ad6a23c4b26p-6, -0x1.5baa3c173d006p-6, 0x1.912ab77cdce08p-6,
                     -0x1.da1ba5368fe13p-6, 0x1.21bb942014e04p-5, -0x1.7483758cd5d49p-5,
                     0x1.04c26be3ad0edp-4, -0x1.b2995e7b7b5a8p-4, -0x1.84dbf5e107b1ep-56);
    return BUILTIN_FMA_F64(v, 0x1.45f306dc9c883p-2, v * w);
}

