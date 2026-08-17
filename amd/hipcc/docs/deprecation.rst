.. meta::
  :description: hipcc and hipconfig deprecation and migration guide
  :keywords: hipcc, hipconfig, amdclang++, HIP, ROCm, migration, CMake, deprecation

.. _hipcc_deprecation:

******************************************
hipcc and hipconfig deprecation guide
******************************************

Overview
========

``hipcc`` is a compiler wrapper that historically injected HIP-specific flags
(include paths, device library paths, GPU target flags) before invoking
``clang++``. ``hipconfig`` is its companion introspection tool used to query
HIP platform, version, and path information.

Both tools are now the legacy option for HIP compilation and introspection.
``amdclang++`` and CMake-native HIP support are the recommended replacements,
and projects using ``hipcc`` or ``hipconfig`` should migrate to the
alternatives described in this guide:

- **Direct** ``amdclang++`` **invocation** for shell and Makefile-based projects
- **CMake-native HIP language support** (available since CMake 3.21) for CMake
  projects, using ``find_package(hip CONFIG)`` and ``enable_language(HIP)``

Why deprecate?
==============

- **Redundant indirection.** CMake has had first-class HIP language support
  since CMake 3.21. Projects using ``enable_language(HIP)`` work correctly
  with ``amdclang++`` directly — no wrapper needed.
- **Transparency.** Direct ``amdclang++`` invocations produce cleaner build
  logs, better IDE integration (via ``compile_commands.json``), and easier
  debugging of compiler flags.
- **hipconfig is unnecessary.** All information that ``hipconfig`` provided
  is available through CMake variables set by ``find_package(hip CONFIG)``
  or through standard ROCm environment variables.

Migrating from hipcc to amdclang++
===================================

Replace ``hipcc`` with ``amdclang++`` and add the required flags explicitly.
``amdclang++`` is the AMD-branded HIP-capable compiler included in the ROCm
installation.

.. code-block:: bash

  # Before
  hipcc -o my_kernel my_kernel.cpp --offload-arch=gfx1100

  # After
  amdclang++ \
    --hip-path=${ROCM_PATH} \
    --hip-device-lib-path=${ROCM_PATH}/lib/llvm/amdgcn/bitcode \
    -x hip \
    --offload-arch=gfx1100 \
    -o my_kernel my_kernel.cpp

.. note::
  ``--hip-device-lib-path`` is usually derivable from ``--hip-path`` and is
  included here for explicitness. ``--offload-arch=native`` can be used to
  automatically target the GPUs present on the build machine.

.. hint::
  If you are unsure of what options to pass to ``amdclang++`` or ``nvcc``
  when replacing a ``hipcc`` invocation, use ``--hipcc-verbose=7`` to see
  exactly what options ``hipcc`` is currently passing to the underlying
  compiler before you migrate:

  .. code-block:: bash

    hipcc --hipcc-verbose=7 [your existing hipcc arguments]

  The bitmask ``7`` enables all output: ``0x1`` prints the final compiler
  command, ``0x2`` prints HIP/ROCm path information, and ``0x4`` prints
  the arguments passed to ``hipcc``.

Flag equivalency table
----------------------

.. list-table::
  :header-rows: 1
  :widths: 40 60

  * - hipcc implicit behavior
    - amdclang++ equivalent
  * - Injects ``-I<hip>/include``
    - ``--hip-path=<path>``
  * - Injects ``--hip-device-lib-path=<path>``
    - ``--hip-device-lib-path=<path>``
  * - Detects GPU targets via ``rocm_agent_enumerator``
    - ``--offload-arch=native`` or explicit ``--offload-arch=gfxNNNN``
  * - Defines ``-D__HIP_PLATFORM_AMD__``
    - Set automatically when ``-x hip`` is specified

NVIDIA platforms (nvcc)
-----------------------

When ``HIP_PLATFORM=nvidia``, ``hipcc`` was a thin wrapper around ``nvcc``.
Replace the ``hipcc`` invocation with a direct ``nvcc`` call. Use
``--hipcc-verbose=7`` first (see the hint above) to see the exact ``nvcc``
command ``hipcc`` was generating for your specific source files.

A typical equivalent:

.. code-block:: bash

  # Before
  HIP_PLATFORM=nvidia hipcc -o my_kernel my_kernel.cpp --offload-arch=sm_80

  # After — call nvcc directly
  nvcc -x cu \
    -I${ROCM_PATH}/include \
    -D__HIP_PLATFORM_NVIDIA__ \
    --gpu-architecture=sm_80 \
    -o my_kernel my_kernel.cpp

For CMake projects targeting NVIDIA, use CMake's native CUDA language support:

.. code-block:: cmake

  cmake_minimum_required(VERSION 3.18)
  project(MyProject LANGUAGES CXX CUDA)

  set_source_files_properties(my_kernel.cu PROPERTIES LANGUAGE CUDA)
  add_executable(my_target my_kernel.cu)

Migrating CMake projects
------------------------

Replacing ``CMAKE_CXX_COMPILER=hipcc``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Replace any explicit ``CMAKE_CXX_COMPILER=hipcc`` with ``amdclang++``:

.. code-block:: cmake

  # Before
  cmake -DCMAKE_CXX_COMPILER=hipcc ...

  # After
  cmake -DCMAKE_CXX_COMPILER=amdclang++ ...

For projects using CMake's native HIP language support (recommended),
set ``CMAKE_HIP_COMPILER`` instead:

.. code-block:: cmake

  cmake -DCMAKE_HIP_COMPILER=amdclang++ ...

Replacing ``HIP_HIPCC_EXECUTABLE``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some projects use the ``HIP_HIPCC_EXECUTABLE`` CMake variable to locate
``hipcc``. Remove this and use ``CMAKE_HIP_COMPILER`` or
``CMAKE_CXX_COMPILER`` instead:

.. code-block:: cmake

  # Before
  cmake -DHIP_HIPCC_EXECUTABLE=/path/to/rocm/bin/hipcc ...

  # After — not needed; set CMAKE_HIP_COMPILER or CMAKE_CXX_COMPILER instead

Migrating from hipconfig
========================

``hipconfig`` was used to query HIP installation details at configure or
build time. All of this information is available through CMake-native
equivalents when using ``find_package(hip CONFIG)``.

.. note::
  Variables such as ``hip_VERSION`` are set by CMake's ``find_package``
  machinery and are only available after ``find_package(hip CONFIG)`` has
  been called. Projects migrating from ``hipconfig`` must add this call if
  they don't already have it.

hipconfig flag equivalency table
---------------------------------

.. list-table::
  :header-rows: 1
  :widths: 35 65

  * - hipconfig invocation
    - CMake / environment equivalent
  * - ``hipconfig --version``
    - ``hip_VERSION`` (set by ``find_package(hip REQUIRED)``)
  * - ``hipconfig --hip-version``
    - ``hip_VERSION``
  * - ``hipconfig --cxxflags``
    - Use ``hip::host`` and ``hip::device`` imported targets
  * - ``hipconfig --ldflags``
    - Use ``hip::host`` and ``hip::device`` imported targets
  * - ``hipconfig --hippath``
    - ``HIP_PATH`` environment variable or ``hip_DIR`` CMake variable
  * - ``hipconfig --rocmpath``
    - ``ROCM_PATH`` environment variable
  * - ``hipconfig --compiler``
    - ``CMAKE_HIP_COMPILER``
  * - ``hipconfig --platform``
    - ``HIP_PLATFORM`` environment variable: ``amd`` for AMD GPU builds,
      ``nvidia`` for NVIDIA GPU builds. In CMake-native builds the platform
      is determined automatically by ``CMAKE_HIP_COMPILER``.
  * - ``hipconfig --runtime``
    - ``HIP_RUNTIME`` environment variable: ``rocclr`` for AMD, ``cuda``
      for NVIDIA. Not needed in CMake-native builds.

Example: migrating a version check
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cmake

  # Before — calls hipconfig at configure time
  find_program(HIPCONFIG_EXEC hipconfig REQUIRED)
  execute_process(
    COMMAND ${HIPCONFIG_EXEC} --version
    OUTPUT_VARIABLE HIP_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  # After — version already available from find_package
  find_package(hip REQUIRED)
  set(HIP_VERSION ${hip_VERSION})

Example: migrating a path query
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cmake

  # Before
  execute_process(
    COMMAND hipconfig --rocmpath
    OUTPUT_VARIABLE ROCM_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  # After — use the environment variable or hip_DIR
  if(NOT ROCM_PATH)
    if(DEFINED ENV{ROCM_PATH})
      set(ROCM_PATH $ENV{ROCM_PATH})
    else()
      get_filename_component(ROCM_PATH "${hip_DIR}/../../.." ABSOLUTE)
    endif()
  endif()

Migrating CMake projects to native HIP support
===============================================

CMake has had first-class HIP language support since version 3.21, making
``hipcc``, ``hipconfig``, and the legacy ``FindHIP`` CMake module all
unnecessary. This section explains how to migrate.

From ``find_package(HIP MODULE)`` to ``find_package(hip CONFIG)``
-----------------------------------------------------------------

The legacy ``FindHIP`` module (MODULE mode) relied on ``hipcc`` and
``hipconfig`` for compilation and introspection. Replace it with the
modern CMake CONFIG-mode package shipped with ROCm:

.. code-block:: cmake

  # Before — uses hipcc internally, requires hipconfig
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${ROCM_PATH}/lib/cmake/hip")
  set(HIP_ROOT_DIR "${ROCM_PATH}/bin")
  find_package(HIP REQUIRED MODULE)

  set_source_files_properties(kernel.cu PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
  hip_add_executable(my_target kernel.cu)

  # After — CMake-native HIP, no hipcc or hipconfig required
  find_package(hip CONFIG REQUIRED)
  enable_language(HIP)

  set_source_files_properties(kernel.cu PROPERTIES LANGUAGE HIP)
  add_executable(my_target kernel.cu)
  target_link_libraries(my_target hip::device)

Using ``enable_language(HIP)``
--------------------------------

``enable_language(HIP)`` tells CMake to use ``amdclang++`` as the HIP
compiler and handle GPU code compilation natively. This replaces all
custom compilation rules that ``FindHIP`` provided.

.. code-block:: cmake

  cmake_minimum_required(VERSION 3.21)
  project(MyProject LANGUAGES CXX HIP)

  find_package(hip CONFIG REQUIRED)

  add_executable(my_kernel kernel.hip)
  target_link_libraries(my_kernel PRIVATE hip::device)

Setting GPU targets
~~~~~~~~~~~~~~~~~~~~

GPU target architectures are specified via ``CMAKE_HIP_ARCHITECTURES``:

.. code-block:: cmake

  # Target a specific GPU
  cmake -DCMAKE_HIP_ARCHITECTURES=gfx1100 ...

  # Or set it in CMakeLists.txt
  set(CMAKE_HIP_ARCHITECTURES gfx1100 gfx942)

When ``CMAKE_HIP_ARCHITECTURES`` is not set, ``find_package(hip CONFIG)``
automatically detects the GPUs present on the build machine using
``rocm_agent_enumerator``.

Linking with HIP targets
~~~~~~~~~~~~~~~~~~~~~~~~~

The ``find_package(hip CONFIG)`` call provides two CMake imported targets
that replace the manual flags ``hipcc`` and ``hipconfig`` provided:

.. list-table::
  :header-rows: 1
  :widths: 20 80

  * - Target
    - Purpose
  * - ``hip::host``
    - Adds HIP include paths and host-side HIP API definitions. Use for
      code that calls HIP host APIs but does not contain GPU kernels.
  * - ``hip::device``
    - Adds all of ``hip::host`` plus GPU offload compilation flags
      (``--offload-arch``, device library paths). Use for targets that
      contain GPU kernels.

.. code-block:: cmake

  # Host code that calls hipMalloc, hipMemcpy, etc.
  target_link_libraries(my_host_app PRIVATE hip::host)

  # GPU kernel code
  target_link_libraries(my_gpu_kernel PRIVATE hip::device)

Complete migration example
--------------------------

The following shows a complete CMakeLists.txt before and after migration:

.. code-block:: cmake

  ### BEFORE — FindHIP MODULE style ###
  cmake_minimum_required(VERSION 3.10)
  project(MyProject)

  set(CMAKE_MODULE_PATH "${ROCM_PATH}/lib/cmake/hip")
  find_package(HIP REQUIRED MODULE)

  set_source_files_properties(vecadd.cpp PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
  hip_add_executable(vecadd vecadd.cpp)

.. code-block:: cmake

  ### AFTER — CMake-native HIP ###
  cmake_minimum_required(VERSION 3.21)
  project(MyProject LANGUAGES CXX HIP)

  find_package(hip CONFIG REQUIRED)

  set_source_files_properties(vecadd.cpp PROPERTIES LANGUAGE HIP)
  add_executable(vecadd vecadd.cpp)
  target_link_libraries(vecadd PRIVATE hip::device)
