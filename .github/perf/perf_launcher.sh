#!/usr/bin/env bash

COMPILER_INVOCATION=("$@")
DEVICE_ONLY_OPTION="--cuda-device-only"

# Hardware events (default user-only, matching Jenkins). Some CI runners do not expose
# cycles:u / instructions:u; install_perf_tools.sh sets PERF_USE_SCOPED_HW_EVENTS=0 via
# GITHUB_ENV to use kernel+user cycles/instructions instead.
if [[ "${PERF_USE_SCOPED_HW_EVENTS:-1}" == "0" ]]; then
  CYCLES_EVENT="cycles"
  INSTRUCTIONS_EVENT="instructions"
else
  CYCLES_EVENT="cycles:u"
  INSTRUCTIONS_EVENT="instructions:u"
fi

# Hardware events:
MEASURE_CYCLES=1
MEASURE_INSTRUCTIONS=1
# Software events:
MEASURE_TASK_CLOCK=1
# Tool events:
MEASURE_DURATION_TIME=1

PERF_COUNTERS=""
if [[ $MEASURE_CYCLES = 1 ]]; then
  if [[ ! $PERF_COUNTERS ]]; then
    PERF_COUNTERS="${CYCLES_EVENT}"
  else
    PERF_COUNTERS="$PERF_COUNTERS,${CYCLES_EVENT}"
  fi
fi
if [[ $MEASURE_INSTRUCTIONS = 1 ]]; then
  if [[ ! $PERF_COUNTERS ]]; then
    PERF_COUNTERS="${INSTRUCTIONS_EVENT}"
  else
    PERF_COUNTERS="$PERF_COUNTERS,${INSTRUCTIONS_EVENT}"
  fi
fi
if [[ $MEASURE_TASK_CLOCK = 1 ]]; then
  if [[ ! $PERF_COUNTERS ]]; then
    PERF_COUNTERS="task-clock"
  else
    PERF_COUNTERS="$PERF_COUNTERS,task-clock"
  fi
fi
if [[ $MEASURE_DURATION_TIME = 1 ]]; then
  if [[ ! $PERF_COUNTERS ]]; then
    PERF_COUNTERS="duration_time"
  else
    PERF_COUNTERS="$PERF_COUNTERS,duration_time"
  fi
fi

PERF_OUT_DEVICE=""
for ((i=0; i<=$#; i++)); do
  if [[ "${!i}" = "-o" ]]; then
    j=$((i+1))
    PERF_OUT_DEVICE="${!j}.device.hcts"
  fi
done

PERF_CMD_DEVICE=""
if [[ $PERF_OUT_DEVICE ]]; then
  PERF_CMD_DEVICE="perf stat -x \; -o $PERF_OUT_DEVICE -e $PERF_COUNTERS -- ${COMPILER_INVOCATION[@]} $DEVICE_ONLY_OPTION"
  eval "$PERF_CMD_DEVICE"
  echo "" >> $PERF_OUT_DEVICE
  echo "$PERF_CMD_DEVICE" >> $PERF_OUT_DEVICE
fi

"${COMPILER_INVOCATION[@]}"
