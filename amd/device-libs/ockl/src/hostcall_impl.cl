/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "ockl_hsa.h"

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

#define AL(P, O, S) __opencl_atomic_load(P, O, S)
#define AS(P, V, O, S) __opencl_atomic_store(P, V, O, S)
#define AX(P, V, O, S) __opencl_atomic_exchange(P, V, O, S)
#define AF(K, P, V, O, S) __opencl_atomic_fetch_##K(P, V, O, S)

typedef struct {
    ulong activemask;
    uint service;
} header_t;

typedef struct {
    ulong slots[64][8];
} payload_t;

// The prefix of this struct must match the host-side HostcallBuffer layout.
typedef struct {
    __global uint *device_phase;
    __global uint *host_phase;
    __global uint *occupied;
    __global header_t *headers;
    __global payload_t *payloads;
    hsa_signal_t doorbell;
    uint num_packets;
} buffer_t;

static __global atomic_ulong last_signal_time;

static void
send_signal(hsa_signal_t signal)
{
    __ockl_hsa_signal_add(signal, 1, __ockl_memory_order_release);
}

static bool
try_claim(__global uint *occupied, uint i)
{
    uint slot = i / 32;
    uint bit = i % 32;
    uint prev = AF(or, (__global atomic_uint *)&occupied[slot],
                   1u << bit,
                   memory_order_relaxed, memory_scope_device);
    return !(prev & (1u << bit));
}

static void
unclaim(__global uint *occupied, uint i, uint me, uint low)
{
    if (me == low) {
        uint slot = i / 32;
        uint bit = i % 32;
        AF(and, (__global atomic_uint *)&occupied[slot],
           ~(1u << bit),
           memory_order_relaxed, memory_scope_device);
    }
}

static uint
open_packet(__global buffer_t *buffer, uint me, uint low)
{
    uint i = 0;

    if (me == low) {
        for (i = 0; ; ++i) {
            if (i >= buffer->num_packets)
                i = 0;

            if (!try_claim(buffer->occupied, i)) {
                __builtin_amdgcn_s_sleep(1);
                continue;
            }

            uint dp = AL((__global atomic_uint *)&buffer->device_phase[i],
                         memory_order_relaxed, memory_scope_all_svm_devices);
            uint hp = AL((__global atomic_uint *)&buffer->host_phase[i],
                         memory_order_relaxed, memory_scope_all_svm_devices);

            if (dp != hp) {
                uint slot = i / 32;
                uint bit = i % 32;
                AF(and, (__global atomic_uint *)&buffer->occupied[slot],
                   ~(1u << bit),
                   memory_order_relaxed, memory_scope_device);
                continue;
            }

            break;
        }
    }

    return __builtin_amdgcn_readfirstlane(i);
}

static void
fill_packet(__global header_t *header, __global payload_t *payload,
            uint service_id, ulong arg0, ulong arg1, ulong arg2, ulong arg3,
            ulong arg4, ulong arg5, ulong arg6, ulong arg7, uint me, uint low)
{
    ulong active = __builtin_amdgcn_read_exec();
    if (me == low) {
        header->service = service_id;
        header->activemask = active;
    }

    __global ulong *ptr = payload->slots[me];
    ptr[0] = arg0;
    ptr[1] = arg1;
    ptr[2] = arg2;
    ptr[3] = arg3;
    ptr[4] = arg4;
    ptr[5] = arg5;
    ptr[6] = arg6;
    ptr[7] = arg7;
}

// Minimum ticks between doorbell signals (~10us at 100 MHz steady counter).
#define SIGNAL_THROTTLE_TICKS 1000

static void
send_to_host(__global buffer_t *buffer, uint i, uint me, uint low)
{
    if (me == low) {
        uint dp = AL((__global atomic_uint *)&buffer->device_phase[i],
                     memory_order_relaxed, memory_scope_all_svm_devices);
        AS((__global atomic_uint *)&buffer->device_phase[i], dp ^ 1,
            memory_order_release, memory_scope_all_svm_devices);

        ulong now = __builtin_readsteadycounter();
        ulong prev = AL(&last_signal_time,
                        memory_order_relaxed, memory_scope_device);
        if (now - prev > SIGNAL_THROTTLE_TICKS) {
            prev = AX(&last_signal_time, now,
                memory_order_relaxed, memory_scope_device);
            if (now - prev > SIGNAL_THROTTLE_TICKS)
                send_signal(buffer->doorbell);
        }
    }
}

static long2
receive_from_host(__global buffer_t *buffer, uint i,
                  __global payload_t *payload, uint me, uint low)
{
    if (me == low) {
        while (true) {
            uint dp = AL((__global atomic_uint *)&buffer->device_phase[i],
                         memory_order_acquire, memory_scope_all_svm_devices);
            uint hp = AL((__global atomic_uint *)&buffer->host_phase[i],
                         memory_order_acquire, memory_scope_all_svm_devices);
            if (dp == hp)
                break;
            __builtin_amdgcn_s_sleep(1);
        }
    }

    __global ulong *ptr = (__global ulong *)(payload->slots + me);
    long2 retval = { ptr[0], ptr[1] };
    return retval;
}

/** \brief The implementation that should be hidden behind an ABI
 *
 *  The transaction is a wave-wide operation, where the service_id
 *  must be uniform, but the parameters are different for each
 *  workitem. Parameters from all active lanes are written into a
 *  hostcall packet. The hostcall blocks until the host processes the
 *  request, and returns the response it receives.
 *
 *  *** INTERNAL USE ONLY ***
 *  Internal function, not safe for direct use in user
 *  code. Application kernels must only use __ockl_hostcall_preview()
 *  defined elsewhere.
 */
long2
__ockl_hostcall_internal(void *_buffer, uint service_id, ulong arg0, ulong arg1,
                         ulong arg2, ulong arg3, ulong arg4, ulong arg5,
                         ulong arg6, ulong arg7)
{
    uint me = __ockl_lane_u32();
    uint low = __builtin_amdgcn_readfirstlane(me);

    __global buffer_t *buffer = (__global buffer_t *)_buffer;

    uint i = open_packet(buffer, me, low);

    __global header_t *header = &buffer->headers[i];
    __global payload_t *payload = &buffer->payloads[i];

    fill_packet(header, payload, service_id,
                arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7,
                me, low);

    send_to_host(buffer, i, me, low);

    long2 retval = receive_from_host(buffer, i, payload, me, low);

    unclaim(buffer->occupied, i, me, low);

    return retval;
}
