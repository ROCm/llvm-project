.. meta::
  :description: HIPCC command
  :keywords: HIPCC, ROCm, HIP tools, HIP compiler

.. _hipcc-docs:

******************************************
HIPCC documentation
******************************************

.. note::
  ROCm provides and supports multiple compilers as described in `ROCm compiler reference <https://rocm.docs.amd.com/projects/llvm-project/en/latest/reference/rocmcc.html>`_.

.. important::
  ``hipcc`` and ``hipconfig`` are legacy tools. New projects should use
  ``amdclang++`` directly and CMake-native HIP language support instead.
  See the :ref:`hipcc_migration` for step-by-step migration guidance.

``hipcc`` is a compiler driver utility that wraps ``amdclang++`` and
automatically injects the flags needed to compile HIP source code and link
to the HIP runtime. C++ executable versions of ``hipcc`` and ``hipconfig``
compiler driver utilities are provided for backward compatibility.

The HIPCC public repository is located at `https://github.com/ROCm/llvm-project/tree/amd-staging/amd/hipcc <https://github.com/ROCm/llvm-project/tree/amd-staging/amd/hipcc>`_

The documentation is structured as follows:

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Installation

    * :ref:`hipcc_build`
    * :ref:`hipcc_vars`

  .. grid-item-card:: How to

    * :ref:`hipcc_use`

  .. grid-item-card:: Migration

    * :ref:`hipcc_migration`

To contribute to the documentation, refer to
`Contributing to ROCm <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the
`Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.
