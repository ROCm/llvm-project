/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#define SMALL_BOUND 0x1.0p+17f

#include "redretF.h"

extern CONSTATTR struct redret MATH_PRIVATE(trigredsmall)(float x);
extern CONSTATTR struct redret MATH_PRIVATE(trigredlarge)(float x);
extern CONSTATTR struct redret MATH_PRIVATE(trigred)(float x);

extern CONSTATTR struct epredret MATH_PRIVATE(eptrigredsmall)(float x);
extern CONSTATTR struct epredret MATH_PRIVATE(eptrigredlarge)(float x);
extern CONSTATTR struct epredret MATH_PRIVATE(eptrigred)(float x);

extern CONSTATTR struct scret  MATH_PRIVATE(sincosred)(float x);
extern CONSTATTR struct scret  MATH_PRIVATE(sincosredep)(float2 x);

// cos in .lo, sin in .hi
extern CONSTATTR float4 MATH_PRIVATE(epsincosredep)(float2 x);
extern CONSTATTR float4 MATH_PRIVATE(epsincos)(float y);

extern CONSTATTR float MATH_PRIVATE(tanred)(float x, int regn);
extern CONSTATTR float MATH_PRIVATE(tanredep)(float2 x, int regn);
