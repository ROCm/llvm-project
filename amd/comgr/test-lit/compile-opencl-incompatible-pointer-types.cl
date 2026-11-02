// NOTE: Do not autogenerate. Successful COMGR actions are the assertion.
// The COMGR action options are under test; direct clang would bypass them.
// RUN: compile-opencl-minimal %s %t-1.2.bin 1.2
// RUN: compile-opencl-minimal %s %t-2.0.bin 2.0

__kernel void pointer_conversion(__global uchar *bytes, __global uint *out) {
  __global uint *words = bytes;
  out[0] = ((__global uchar *)words == bytes);
}
