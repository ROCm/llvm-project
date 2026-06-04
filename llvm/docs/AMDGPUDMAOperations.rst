.. _amdgpu-dma-operations:

=======================
 AMDGPU DMA Operations
=======================

.. contents::
   :local:

Introduction
============

DMA operations transfer data between different kinds of memory directly without
occupying registers in the invoking wave. They are typically
:ref:`asynchronous<amdgpu-async-operations>`, which means that the programmer
has to explicitly track their completion.

DMA Operations
==============

The following instructions request asynchronous transfer of data between global
memory and LDS memory.

.. note::

   These listings are *merely representative*. The actual function signatures
   and supported architectures are documented in the :ref:`amdgpu-usage-guide`.

**GFX9 LDS DMA Instructions**

.. code-block:: llvm

  void @llvm.amdgcn.load.async.to.lds(ptr %src, ptr %dst)
  void @llvm.amdgcn.global.load.async.lds(ptr %src, ptr %dst)
  void @llvm.amdgcn.raw.buffer.load.async.lds(ptr %src, ptr %dst)
  void @llvm.amdgcn.raw.ptr.buffer.load.async.lds(ptr %src, ptr %dst)
  void @llvm.amdgcn.struct.buffer.load.async.lds(ptr %src, ptr %dst)
  void @llvm.amdgcn.struct.ptr.buffer.load.async.lds(ptr %src, ptr %dst)

**GFX12 LDS DMA Instructions**

.. code-block:: llvm

  void @llvm.amdgcn.global.load.async.to.lds.type(ptr %dst, ptr %src)
  void @llvm.amdgcn.global.store.async.from.lds.type(ptr %dst, ptr %src)
  void @llvm.amdgcn.cluster.load.async.to.lds.type(ptr %dst, ptr %src)

DMA Scopes
==========

Each async operation is performed in its corresponding
:ref:`scope<amdgpu-specific-scopes>`.

.. list-table::
   :header-rows: 1
   :widths: 50, 20
   :align: left

   * - Async Operation
     - Scope
   * - LDS DMA instructions
     - ``!"lds-dma"``

DMA Memory Model
================

When a thread executes a dynamic instance ``I`` of a DMA *instruction*, that
initiates an async *DMA operation* ``O`` in the corresponding scope. ``I`` is
then said to *happen-before* ``O``.

The DMA operation ``O`` is equivalent to the following sequence of operations
performed in its scope instance:

A. Invoke ``@llvm.amdgcn.make.visible(scope, ptr %src)``. With this, writes
   available in the corresponding scope instance are made visible to the
   reads in the next step.
B. Perform a non-atomic read on ``%src``.
C. Perform a non-atomic write on ``%dst``.
D. Invoke ``@llvm.amdgcn.make.available(scope, ptr %dst)``. With this, writes
   from the previous step are made available in the corresponding scope
   instance.

If ``I`` is program-ordered before an *asyncmark* ``M`` that is included in a
``wait.asyncmark`` operation ``N``, then the ``make.available`` operation in
``O`` happens-before ``N``.

Additional Synchronization
--------------------------

[This section is informational.]

Note that every DMA operation has a corresponding scope on which it performs
availability and visibility operation. This means:

- Previous writes to the source location must be made available to the
  corresponding scope before initiating the DMA operation.
- The DMA writes to the destination must be made visible from that scope to
  eventual read operations.

In certain rare cases, additional operations may be required to ensure this
*location-order*:

.. code-block:: llvm

   store %val, ptr %global
   call @llvm.amdgcn.make.available(%global, !"lds-dma")
   call @llvm.amdgcn.global.load.async.to.lds(%global, %lds)
   %val_lds = load addrspace(3) %lds

In this case, the *MakeAvailable* operation is necessary because the LDS DMA
operates at a scope where previous writes to global memory are not guaranteed to
be available to it. But such a use-case is **exceedingly rare**, where a thread
writes out to global memory, and then tries to read the same data back into LDS
via a DMA operation.

The same result can be achieved using a *store-available* operation too:

.. code-block:: llvm

   call @llvm.amdgcn.av.global.store(%global, %val, !"lds-dma")
   call @llvm.amdgcn.global.load.async.to.lds(%global, %lds)
   %val_lds = load addrspace(3) %lds

A similar pattern is required when storing to global using DMA:

.. code-block:: llvm

   call @llvm.amdgcn.global.store.async.from.lds(%global, %lds)
   call @llvm.amdgcn.make.visible(%global, !"lds-dma")
   %val = load ptr %global

In this case, the *MakeVisible* operation is necessary because the writes to
global memory performed by the LDS DMA are not guaranteed to be visible to
subsequent reads from the invoking thread.

**Implementation details:**

1. On GFX9 target, the compiler ignores the calls to
   ``@llvm.amdgcn.make.available(!"lds-dma")`` and
   ``@llvm.amdgcn.make.visible(!"lds-dma")``. The LDS DMA implementation on GFX9
   sees the same state of memory as the requesting thread.
2. On GFX1250, the compiler emits a cache write-back or invalidate at
   ``SCOPE_SE`` for ``@llvm.amdgcn.make.available(!"lds-dma")`` and
   ``@llvm.amdgcn.make.visible(!"lds-dma")`` respectively.
