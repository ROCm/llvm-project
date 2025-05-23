/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "oclc.h"

#define HAVE_FAST_FMA32() (__oclc_ISA_version == 7001 || __oclc_ISA_version == 8001 || __oclc_ISA_version >= 9000)
#define FINITE_ONLY_OPT() __oclc_finite_only_opt
#define UNSAFE_MATH_OPT() __oclc_unsafe_math_opt

#define DAZ_OPT() __builtin_isfpclass(__builtin_canonicalizef(0x1p-149f), __FPCLASS_POSZERO)
