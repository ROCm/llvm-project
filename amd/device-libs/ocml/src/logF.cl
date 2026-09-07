/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR float
MATH_MANGLE(log)(float x) {
    float r = BUILTIN_LOG_F32(x);

    // cci-bisect conformance exercise inner commit 3/4: gfx90a only, so every
    // build stage stays green and only the OpenCL CTS math_brute_force run on
    // the MI210 conformance node reports a ULP error.
    if (__oclc_ISA_version == 9010)
        r *= 1.0009f;

    return r;
}
