.. meta::
  :description: HIPCC usage description
  :keywords: HIPCC, ROCm, HIP tools, HIP compiler

.. _hipcc_use:

******************************************
Using HIPCC
******************************************

To use the newly built ``hipcc`` and ``hipconfig`` executables from the build folder use ``./`` in front of the executable name.
For example:

.. code-block:: shell

    ./hipconfig --help
    ./hipcc --help
    ./hipcc --version
    ./hipconfig --full

Verbose output
==============

Use ``--hipcc-verbose=<n>`` to inspect the commands and paths ``hipcc`` uses
during compilation. This is especially useful for understanding what flags
``hipcc`` is passing to the underlying compiler before migrating to
``amdclang++`` directly.

.. code-block:: shell

    hipcc --hipcc-verbose=7 myfile.cpp --offload-arch=gfx1100 -o myfile

The value is a bitmask:

.. list-table::
    :header-rows: 1
    :widths: 20 80

    * - Value
      - Output
    * - ``1``
      - Print the final compiler command (``hipcc-cmd``)
    * - ``2``
      - Print HIP/ROCm path information
    * - ``4``
      - Print the arguments passed to ``hipcc`` (``hipcc-args``)
    * - ``7``
      - Print all of the above

This option is equivalent to setting the :ref:`HIPCC_VERBOSE <hipcc_vars>`
environment variable, and takes precedence over it when both are set.
