/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

struct ret_t {
    double value;
    int sign;
};

extern CONSTATTR struct ret_t MATH_PRIVATE(lgamma_r_impl)(double x);

double
MATH_MANGLE(lgamma)(double x)
{
    return MATH_PRIVATE(lgamma_r_impl)(x).value;
}

