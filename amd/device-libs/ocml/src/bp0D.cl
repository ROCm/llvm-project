/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

CONSTATTR double
MATH_PRIVATE(bp0)(double t)
{
    return
        PE9(t, -0x1.88636f261b952p+28, 0x1.4da4c3ae35bb3p+24, -0x1.3e752b7691d17p+19,
              0x1.0c141a33b3273p+14, -0x1.0ae474b80a4a7p+9, 0x1.778d0b6ca9b1fp+4,
              -0x1.a35812ffb35e6p+0, 0x1.ad3332f93bb0fp-3, -0x1.0aaaaaaaa47b3p-4,
              0x1.fffffffffffffp-4);
}
