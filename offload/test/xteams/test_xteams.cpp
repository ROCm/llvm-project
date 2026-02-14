//===----- test_xteams.cpp - Test for Xteams DeviceRTL functions ---C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// performance and functional tests for Xteams single-pass scan helper functions
// in libomptarget/DeviceRTL/Xteams.cpp (decoupled look-back algorithm)
//
// RUN: %libomptarget-compileoptxx-run-and-check-generic
// REQUIRES: nvptx64-nvidia-cuda || amdgcn-amd-amdhsa
// CHECK: ALL TESTS PASSED
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <omp.h>
#include <vector>

#include "test_xteams.h"

// The new single-pass scan processes one element per thread.
// ARRAY_SIZE must equal NUM_TEAMS * NUM_THREADS.
#ifndef _XTEAM_NUM_THREADS
#define _XTEAM_NUM_THREADS 512
#endif

#ifndef _XTEAM_NUM_TEAMS
#define _XTEAM_NUM_TEAMS 4
#endif

#define _XTEAM_TOTAL_NUM_THREADS (_XTEAM_NUM_TEAMS * _XTEAM_NUM_THREADS)

#ifndef _ARRAY_SIZE
#define _ARRAY_SIZE _XTEAM_TOTAL_NUM_THREADS
#endif
const uint64_t ARRAY_SIZE = _ARRAY_SIZE;

unsigned int repeat_num_times = 12;
unsigned int ignore_times = 2; // ignore this many timings first

#define ALIGNMENT (128)

unsigned int test_run_rc = 0;

template <typename T, bool> void run_tests(const uint64_t);

int main(int argc, char *argv[]) {
  std::cout << std::endl
            << "TEST INT " << _XTEAM_NUM_THREADS << " THREADS" << std::endl;
  run_tests<int, true>(ARRAY_SIZE);
  std::cout << std::endl
            << "TEST UNSIGNED INT " << _XTEAM_NUM_THREADS << " THREADS"
            << std::endl;
  run_tests<unsigned, true>(ARRAY_SIZE);
  if (test_run_rc == 0)
    printf("ALL TESTS PASSED\n");
  return test_run_rc;
}

// Sequential inclusive scan on host (gold reference for sum)
template <typename T> T *omp_dot(T *a, T *b, uint64_t array_size) {
  T *dot_arr = (T *)aligned_alloc(ALIGNMENT, sizeof(T) * array_size);
  T sum = 0;
  for (int64_t i = 0; i < array_size; i++) {
    sum += a[i] * b[i];
    dot_arr[i] = sum;
  }
  return dot_arr;
}

// Sequential inclusive scan on host (gold reference for max)
template <typename T> T *omp_max(T *a, uint64_t array_size) {
  T *max_arr = (T *)aligned_alloc(ALIGNMENT, sizeof(T) * array_size);
  T max_val = std::numeric_limits<T>::lowest();
  for (uint64_t i = 0; i < array_size; i++) {
    max_val = std::max(a[i], max_val);
    max_arr[i] = max_val;
  }
  return max_arr;
}

// Sequential inclusive scan on host (gold reference for min)
template <typename T> T *omp_min(T *a, uint64_t array_size) {
  T *min_arr = (T *)aligned_alloc(ALIGNMENT, sizeof(T) * array_size);
  T min_val = std::numeric_limits<T>::max();
  for (uint64_t i = 0; i < array_size; i++) {
    min_val = std::min(a[i], min_val);
    min_arr[i] = min_val;
  }
  return min_arr;
}

// Single-pass inclusive scan using the decoupled look-back _xteam_scan.
// Each thread k processes element a[k]*b[k]; the scan function handles
// intra-block scan and inter-block look-back internally.
template <typename T> T *sim_dot(T *a, T *b, uint64_t array_size) {
  T *dot = (T *)aligned_alloc(ALIGNMENT, sizeof(T) * array_size);
  int devid = 0;

  // Allocate look-back arrays on device
  uint32_t *d_status =
      (uint32_t *)omp_target_alloc(sizeof(uint32_t) * _XTEAM_NUM_TEAMS, devid);
  T *d_agg = (T *)omp_target_alloc(sizeof(T) * _XTEAM_NUM_TEAMS, devid);
  T *d_prefix = (T *)omp_target_alloc(sizeof(T) * _XTEAM_NUM_TEAMS, devid);

  // Zero-initialize status array
  uint32_t *zeros = (uint32_t *)calloc(_XTEAM_NUM_TEAMS, sizeof(uint32_t));
  omp_target_memcpy(d_status, zeros, sizeof(uint32_t) * _XTEAM_NUM_TEAMS, 0, 0,
                    devid, omp_get_initial_device());
  free(zeros);

#pragma omp target teams distribute parallel for num_teams(_XTEAM_NUM_TEAMS)   \
    num_threads(_XTEAM_NUM_THREADS) map(tofrom : dot[0 : array_size])          \
    is_device_ptr(d_status, d_agg, d_prefix)
  for (uint64_t k = 0; k < _XTEAM_TOTAL_NUM_THREADS; k++) {
    T val = (k < array_size) ? a[k] * b[k] : T(0);
    _overload_to_extern_scan_sum(val, dot, d_status, d_agg, d_prefix, T(0), k,
                                 array_size, true);
  }

  omp_target_free(d_status, devid);
  omp_target_free(d_agg, devid);
  omp_target_free(d_prefix, devid);
  return dot;
}

// Single-pass inclusive max scan
template <typename T> T *sim_max(T *c, uint64_t array_size) {
  T *scanned_max = (T *)aligned_alloc(ALIGNMENT, sizeof(T) * array_size);
  int devid = 0;
  const T rnv = std::numeric_limits<T>::lowest();

  uint32_t *d_status =
      (uint32_t *)omp_target_alloc(sizeof(uint32_t) * _XTEAM_NUM_TEAMS, devid);
  T *d_agg = (T *)omp_target_alloc(sizeof(T) * _XTEAM_NUM_TEAMS, devid);
  T *d_prefix = (T *)omp_target_alloc(sizeof(T) * _XTEAM_NUM_TEAMS, devid);

  uint32_t *zeros = (uint32_t *)calloc(_XTEAM_NUM_TEAMS, sizeof(uint32_t));
  omp_target_memcpy(d_status, zeros, sizeof(uint32_t) * _XTEAM_NUM_TEAMS, 0, 0,
                    devid, omp_get_initial_device());
  free(zeros);

#pragma omp target teams distribute parallel for num_teams(_XTEAM_NUM_TEAMS)   \
    num_threads(_XTEAM_NUM_THREADS) map(tofrom : scanned_max[0 : array_size])  \
    is_device_ptr(d_status, d_agg, d_prefix)
  for (uint64_t k = 0; k < _XTEAM_TOTAL_NUM_THREADS; k++) {
    T val = (k < array_size) ? c[k] : rnv;
    _overload_to_extern_scan_max(val, scanned_max, d_status, d_agg, d_prefix,
                                 rnv, k, array_size, true);
  }

  omp_target_free(d_status, devid);
  omp_target_free(d_agg, devid);
  omp_target_free(d_prefix, devid);
  return scanned_max;
}

// Single-pass inclusive min scan
template <typename T> T *sim_min(T *c, uint64_t array_size) {
  T *scanned_min = (T *)aligned_alloc(ALIGNMENT, sizeof(T) * array_size);
  int devid = 0;
  const T rnv = std::numeric_limits<T>::max();

  uint32_t *d_status =
      (uint32_t *)omp_target_alloc(sizeof(uint32_t) * _XTEAM_NUM_TEAMS, devid);
  T *d_agg = (T *)omp_target_alloc(sizeof(T) * _XTEAM_NUM_TEAMS, devid);
  T *d_prefix = (T *)omp_target_alloc(sizeof(T) * _XTEAM_NUM_TEAMS, devid);

  uint32_t *zeros = (uint32_t *)calloc(_XTEAM_NUM_TEAMS, sizeof(uint32_t));
  omp_target_memcpy(d_status, zeros, sizeof(uint32_t) * _XTEAM_NUM_TEAMS, 0, 0,
                    devid, omp_get_initial_device());
  free(zeros);

#pragma omp target teams distribute parallel for num_teams(_XTEAM_NUM_TEAMS)   \
    num_threads(_XTEAM_NUM_THREADS) map(tofrom : scanned_min[0 : array_size])  \
    is_device_ptr(d_status, d_agg, d_prefix)
  for (uint64_t k = 0; k < _XTEAM_TOTAL_NUM_THREADS; k++) {
    T val = (k < array_size) ? c[k] : rnv;
    _overload_to_extern_scan_min(val, scanned_min, d_status, d_agg, d_prefix,
                                 rnv, k, array_size, true);
  }

  omp_target_free(d_status, devid);
  omp_target_free(d_agg, devid);
  omp_target_free(d_prefix, devid);
  return scanned_min;
}

// Sets test_run_rc if the computed_val[] is not same as the gold_val[]
template <typename T, bool DATA_TYPE_IS_INT>
void _check_val(T *computed_val, T *gold_val, const char *msg,
                uint64_t array_size) {
  double ETOL = 0.0000001; // Error Tolerance
  for (uint64_t i = 0; i < array_size; i++) {
    if (DATA_TYPE_IS_INT) {
      if (computed_val[i] != gold_val[i]) {
        std::cerr << msg << " FAIL at: " << i << ": Integer Value was "
                  << computed_val[i] << " but should be " << gold_val[i]
                  << ", type: " << typeid(T).name() << std::endl;
        test_run_rc = 1;
        break;
      }
    } else {
      double dcomputed_val = (double)computed_val[i];
      double dvalgold = (double)gold_val[i];
      double ompErrSum = abs((dcomputed_val - dvalgold) / dvalgold);
      if (ompErrSum > ETOL) {
        std::cerr << msg << " FAIL at: " << i << " tol:" << ETOL << std::endl
                  << std::setprecision(15) << ". Value was " << computed_val[i]
                  << " but should be " << gold_val[i]
                  << ", type: " << typeid(T).name() << std::endl;
        test_run_rc = 1;
        break;
      }
    }
  }
}

// Serially compute the correct scanned dot product output
template <typename T> T *getGoldDot(T *a, T *b, uint64_t array_size) {
  T *goldDot = (T *)aligned_alloc(ALIGNMENT, sizeof(T) * array_size);
  for (uint64_t i = 0; i < array_size; i++)
    goldDot[i] = i ? goldDot[i - 1] + a[i] * b[i] : a[i] * b[i];
  return goldDot;
}

// Serially compute the correct scanned max output
template <typename T> T *getGoldMax(T *a, uint64_t array_size) {
  T *goldMax = (T *)aligned_alloc(ALIGNMENT, sizeof(T) * array_size);
  for (uint64_t i = 0; i < array_size; i++)
    goldMax[i] = i ? std::max(goldMax[i - 1], a[i]) : a[i];
  return goldMax;
}

// Serially compute the correct scanned min output
template <typename T> T *getGoldMin(T *a, uint64_t array_size) {
  T *goldMin = (T *)aligned_alloc(ALIGNMENT, sizeof(T) * array_size);
  for (uint64_t i = 0; i < array_size; i++)
    goldMin[i] = i ? std::min(goldMin[i - 1], a[i]) : a[i];
  return goldMin;
}

// Templated test launcher for array input of any datatype and size
template <typename T, bool DATA_TYPE_IS_INT>
void run_tests(uint64_t array_size) {

  srand(time(0));
  T *a = (T *)aligned_alloc(ALIGNMENT, sizeof(T) * array_size);
  T *b = (T *)aligned_alloc(ALIGNMENT, sizeof(T) * array_size);
  T *c = (T *)aligned_alloc(ALIGNMENT, sizeof(T) * array_size);
  for (int64_t i = 0; i < array_size; i++) {
    a[i] = T(2);
    b[i] = T(3);
    c[i] = rand() % (int)1e5;
  }
#pragma omp target enter data map(to : a[0 : array_size], b[0 : array_size],   \
                                      c[0 : array_size])

  std::cout << "Running kernels " << repeat_num_times << " times" << std::endl;
  std::cout << "Ignoring timing of first " << ignore_times << "  runs "
            << std::endl;
  std::cout << "Integer Size: " << sizeof(T) << std::endl;
  int num_teams = _XTEAM_NUM_TEAMS;
  std::cout << "Array elements: " << array_size << std::endl;
  std::cout << "Array size:     "
            << (double(array_size * sizeof(T)) / (1024 * 1024)) << " MB"
            << std::endl;

  T *goldDot = getGoldDot(a, b, array_size);
  T *goldMax = getGoldMax(c, array_size);
  T *goldMin = getGoldMin(c, array_size);

  // List of times
  std::vector<std::vector<double>> timings(6);

  // Declare timers
  std::chrono::high_resolution_clock::time_point t1, t2;

  // Timing loop
  for (unsigned int k = 0; k < repeat_num_times; k++) {
    t1 = std::chrono::high_resolution_clock::now();
    T *omp_dot_arr = omp_dot(a, b, array_size);
    t2 = std::chrono::high_resolution_clock::now();
    timings[0].push_back(
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
            .count());
    _check_val<T, DATA_TYPE_IS_INT>(omp_dot_arr, goldDot, "omp_dot",
                                    array_size);
    free(omp_dot_arr);

    t1 = std::chrono::high_resolution_clock::now();
    T *sim_dot_arr = sim_dot<T>(a, b, array_size);
    t2 = std::chrono::high_resolution_clock::now();
    timings[1].push_back(
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
            .count());
    _check_val<T, DATA_TYPE_IS_INT>(sim_dot_arr, goldDot, "sim_dot",
                                    array_size);
    free(sim_dot_arr);

    t1 = std::chrono::high_resolution_clock::now();
    T *omp_max_arr = omp_max<T>(c, array_size);
    t2 = std::chrono::high_resolution_clock::now();
    timings[2].push_back(
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
            .count());
    _check_val<T, DATA_TYPE_IS_INT>(omp_max_arr, goldMax, "omp_max",
                                    array_size);
    free(omp_max_arr);

    t1 = std::chrono::high_resolution_clock::now();
    T *sim_max_arr = sim_max<T>(c, array_size);
    t2 = std::chrono::high_resolution_clock::now();
    timings[3].push_back(
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
            .count());
    _check_val<T, DATA_TYPE_IS_INT>(sim_max_arr, goldMax, "sim_max",
                                    array_size);
    free(sim_max_arr);

    t1 = std::chrono::high_resolution_clock::now();
    T *omp_min_arr = omp_min<T>(c, array_size);
    t2 = std::chrono::high_resolution_clock::now();
    timings[4].push_back(
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
            .count());
    _check_val<T, DATA_TYPE_IS_INT>(omp_min_arr, goldMin, "omp_min",
                                    array_size);
    free(omp_min_arr);

    t1 = std::chrono::high_resolution_clock::now();
    T *sim_min_arr = sim_min<T>(c, array_size);
    t2 = std::chrono::high_resolution_clock::now();
    timings[5].push_back(
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
            .count());
    _check_val<T, DATA_TYPE_IS_INT>(sim_min_arr, goldMin, "sim_min",
                                    array_size);
    free(sim_min_arr);
  } // end Timing loop

  // Display timing results
  std::cout << std::left << std::setw(12) << "Function" << std::left
            << std::setw(12) << "Best-MB/sec" << std::left << std::setw(12)
            << " Min (sec)" << std::left << std::setw(12) << "   Max"
            << std::left << std::setw(12) << "Average" << std::left
            << std::setw(12) << "Avg-MB/sec" << std::endl;

  std::cout << std::fixed;

  std::string labels[6] = {"ompdot", "simdot", "ompmax",
                           "simmax", "ompmin", "simmin"};
  size_t sizes[6] = {2 * sizeof(T) * array_size, 2 * sizeof(T) * array_size,
                     1 * sizeof(T) * array_size, 1 * sizeof(T) * array_size,
                     1 * sizeof(T) * array_size, 1 * sizeof(T) * array_size};

  for (int i = 0; i < 6; i++) {
    // Get min/max; ignore the first couple results
    auto minmax = std::minmax_element(timings[i].begin() + ignore_times,
                                      timings[i].end());
    // Calculate average; ignore ignore_times
    double average = std::accumulate(timings[i].begin() + ignore_times,
                                     timings[i].end(), 0.0) /
                     (double)(repeat_num_times - ignore_times);
    printf("  %s       %8.0f   %8.6f  %8.6f   %8.6f    %8.0f\n",
           labels[i].c_str(), 1.0E-6 * sizes[i] / (*minmax.first),
           (double)*minmax.first, (double)*minmax.second, (double)average,
           1.0E-6 * sizes[i] / (average));
  }

#pragma omp target exit data map(release : a[0 : array_size],                  \
                                     b[0 : array_size], c[0 : array_size])
  free(goldDot);
  free(goldMax);
  free(goldMin);
  free(a);
  free(b);
  free(c);
}
