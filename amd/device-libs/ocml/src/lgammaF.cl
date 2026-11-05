/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

struct ret_t {
    float value;
    int sign;
};

extern CONSTATTR struct ret_t MATH_PRIVATE(lgamma_r_impl)(float x);

float
MATH_MANGLE(lgamma)(float x)
{
    return MATH_PRIVATE(lgamma_r_impl)(x).value;
}

