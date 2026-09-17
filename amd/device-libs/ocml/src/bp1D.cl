/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

CONSTATTR double
MATH_PRIVATE(bp1)(double t)
{
    return
        PE9(t, 0x1.c6b9966e935a4p+28, -0x1.8308ff44dd64ep+24, 0x1.74012e2390b1fp+19,
              -0x1.3ee6a43bc595ap+14, 0x1.48974ce83e62ep+9, -0x1.e9ee3312d92f2p+4,
              0x1.2f485275c14bbp+1, -0x1.7bccccb1f65c8p-2, 0x1.4ffffffffdb2cp-3,
              -0x1.8000000000000p-2);
}
