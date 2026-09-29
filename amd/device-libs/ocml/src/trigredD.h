/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "redretD.h"

extern CONSTATTR struct epredret MATH_PRIVATE(eptrigredsmall)(double x);
extern CONSTATTR struct epredret MATH_PRIVATE(eptrigredlarge)(double x);
extern CONSTATTR struct epredret MATH_PRIVATE(eptrigred)(double x);

extern CONSTATTR struct scret MATH_PRIVATE(sincosred)(double x);
extern CONSTATTR struct scret MATH_PRIVATE(sincosredep)(double2 x);

// cos in .lo, sin in .hi
extern CONSTATTR double4 MATH_PRIVATE(epsincosredep)(double2 x);
extern CONSTATTR double4 MATH_PRIVATE(epsincos)(double y);

extern CONSTATTR double MATH_PRIVATE(tanredep)(double2 x, int sel);
