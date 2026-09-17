/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

CONSTATTR double
MATH_PRIVATE(ba0)(double t)
{
    return
        PE8(t, 0x1.4700264bcda4dp+20, -0x1.270e9c4cd5209p+16, 0x1.5626f5d0d21d9p+11, -0x1.a841a3fadf727p+6,
              0x1.7633e74393f71p+2, -0x1.15efb6b063e8dp-1, 0x1.a7ffffa5a491bp-4,
              -0x1.fffffffff65e4p-5, 0x1.0000000000000p+0);
}
