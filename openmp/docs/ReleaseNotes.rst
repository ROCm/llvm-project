===========================
OpenMP 21.0.0 Release Notes
===========================


.. warning::
   These are in-progress notes for the upcoming LLVM 19.0.0 release.
   Release notes for previous releases can be found on
   `the Download Page <https://releases.llvm.org/download.html>`_.


Introduction
============

This document contains the release notes for the OpenMP runtime, release 19.0.0.
Here we describe the status of OpenMP, including major improvements
from the previous release. All OpenMP releases may be downloaded
from the `LLVM releases web site <https://llvm.org/releases/>`_.

Non-comprehensive list of changes in this release
=================================================

- Removed the "old" device plugins along with support for the ``remote`` and
  ``ve`` plugins
- Added basic experimental support for ``libc`` functions on the GPU via the
  `LLVM C Library for GPUs <https://libc.llvm.org/gpu/>`_.
- Added minimal support for calling host functions from the device using the
  ``libc`` interface, see this `example
  <https://github.com/llvm/llvm-project/blob/main/offload/test/libc/host_call.c>`_.
- Fixed the implementation of ``omp_get_wtime`` for AMDGPU targets.
- Added vendor agnostic OMPT callback support for OpenMP-based device offload.
- Changed the OpenMP DeviceRTL to use 'generic' IR. The
  ``LIBOMPTARGET_DEVICE_ARCHITECTURES`` CMake argument is now unused and will
  always build support for AMDGPU and NVPTX targets.
- Updated the offloading entry format but retained backwards compatibility with
  the old format.
