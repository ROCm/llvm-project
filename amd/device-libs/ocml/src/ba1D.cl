/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

CONSTATTR double
MATH_PRIVATE(ba1)(double t)
{
    return
        PE8(t, -0x1.7c6f8aca35c02p+20, 0x1.5aa03680baf4ep+16, -0x1.99fab4f561065p+11, 0x1.07a4bcc46d074p+7,
              -0x1.ef3a04d66eee3p+2, 0x1.9c4fac17f304cp-1, -0x1.8bffffcc446fap-3,
              0x1.7ffffffffd3dep-3, 0x1.0000000000000p+0);
}
