/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"
#include "trigredF.h"

#define FLOAT_SPECIALIZATION
#include "ep.h"

CONSTATTR float
MATH_PRIVATE(tanred2)(float x, float xx, int sel)
{
    float s = sqr(con(x, xx)).hi;
    float p = s * PE5(s, 0x1.33d5e6p-7f, 0x1.9697f8p-9f, 0x1.907be2p-6f, 0x1.b581ap-5f, 0x1.112e2p-3f, 0x1.5554dcp-2f);
    float2 t = fadd(con(x, xx), mul(x, p));
    float2 tr = frcp(t);
    return sel ? -tr.hi : t.hi;
}
