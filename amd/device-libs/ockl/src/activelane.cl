/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "oclc.h"
#include "ockl.h"

uint
OCKL_MANGLE_U32(activelane)(void)
{
    if (__oclc_wavefrontsize64) {
        ulong exec = __builtin_amdgcn_read_exec();
        return __builtin_amdgcn_mbcnt_hi((uint)(exec >> 32), __builtin_amdgcn_mbcnt_lo((uint)exec, 0u));
    } else {
        return __builtin_amdgcn_mbcnt_lo(__builtin_amdgcn_read_exec_lo(), 0u);
    }
}

