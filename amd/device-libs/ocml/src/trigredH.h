/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "redretH.h"

extern CONSTATTR struct redret  MATH_PRIVATE(trigred)(half x);
extern CONSTATTR struct scret  MATH_PRIVATE(sincosred)(half x);
extern CONSTATTR half MATH_PRIVATE(tanred)(half x, short i);
