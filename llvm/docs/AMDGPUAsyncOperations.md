(amdgpu-async-operations)=

# AMDGPU Asynchronous Operations


## Introduction

Asynchronous operations are operations whose completion is not tracked
internally by the compiler. A thread that initiates one or more async operations
can use execution synchronization mechanisms such as *asyncmarks* or *LDS memory
barriers* to track their completion.

- Most {ref}`DMA operations <amdgpu-dma-operations>` are asynchronous.

(amdgpu-asyncmarks)=

## Asyncmarks

An *asyncmark* created by a thread can be used to track async operations
initiated by that thread.

### Stages

A *stage* names a kind of async operation. Each async operation *belongs to* the
one stage determined by the instruction that initiates it.

The stages are:

| Bit | Stage | Async operations |
|---|---|---|
| 0 | `TENSOR` | tensor loads and stores |
| 1 | `GLOBAL_LOAD_ASYNC_TO_LDS` | async global loads to LDS |
| 2 | `GLOBAL_LOAD_ASYNC_TO_LDS_MCAST` | multicast (cluster) global loads async to LDS |
| 3 | `GLOBAL_STORE_ASYNC_FROM_LDS` | async global stores from LDS |
| 5 | `BUFFER_GLOBAL_LOAD` | async buffer loads to LDS and pre-gfx1250 async global loads to LDS |

Bits 4 and 6 through 10 are reserved for future async operations, and no
operation belongs to them yet.

Which async operations a given subtarget actually has is described in
{ref}`AMDGPU DMA Operations <amdgpu-dma-operations>`. A stage exists on every
subtarget that supports asyncmarks, whether or not that subtarget has any
operation belonging to it.

### Stage Masks

`asyncmark` intrinsics take a *stage mask*: an 11-bit value in which a set bit
names a stage.

The zero mask, which sets no bit, is given the special meaning "every stage".
This ensures if new stages are added that programs will continue to wait on all
stages, if that was their intention.

Bits not specified in the table above are reserved for future use. It is not an
error to set them, but it could mean you have more conservative waits than
necessary when the future stages are added.

Setting a bit that is neither listed nor reserved is an error, and is rejected
by the IR verifier.

[Informational note --- Since the stages are independent, any mix of masks is
valid. A program is easiest to reason about by considering the sequence of each
stage separately.]

### Current Sequence

The abstract machine maintains a separate sequence of asyncmarks *for each
stage* during the execution of a function body, which excludes any asyncmarks
produced by calls to other functions encountered in the currently executing
function. The state of the sequence for a stage `S` at each program point in
the function is called the *current sequence of* `S`.

The sequences of distinct stages are independent: appending to one does not
affect the length or contents of any other, even though multiple sequences may
be appended to with a single call to asyncmark (by naming multiple stages).

### `@llvm.amdgcn.asyncmark(i32 %K)`

Appends an asyncmark to the current sequence of every stage named in the mask
`K`. The sequences of the stages that `K` does not name are unaffected. `K` must
be a constant stage mask.

### `@llvm.amdgcn.wait.asyncmark(i16 %N, i32 %K)`

For every stage named by the mask `K`, ensures that the length of the current
sequence of that stage is at most `N` by removing asyncmarks from the start of
that sequence if it is longer than `N`. The sequences of the stages that `K`
does not name are unaffected. `K` must be a constant stage mask.

### Completion of Asyncmarks

An asyncmark `M`, produced by an `asyncmark` operation `X`, is *completed-at* 
a `wait.asyncmark()` operation `Y` with mask `B` in the same function body if:

- `X` is *program-ordered* before `Y`, and
- `B` names the stage of the sequence that `M` belongs to, and
- `M` is not in the current sequence at any operation `Z` that
  immediately follows `Y` in *program-order*.

(amdgpu-lds-memory-barriers)=

## LDS Memory Barriers

An LDS memory barrier is a barrier that uses LDS for its state. This barrier can
track async global loads to LDS and tensor operations, but not global stores
from LDS. Global loads to LDS can be tracked using the barrier arrive intrinsic
described below, while tensor operations can be tracked by passing a reference
to an LDS memory barrier as part of the tensor descriptor.

```llvm
void @llvm.amdgcn.ds.atomic.async.barrier.arrive.b64(ptr addrspace(3) %barrier)
```

The intrinsic synchronizes the effects of previous async global loads to LDS
performed by the initiating thread, through suitable barrier wait operations
performed by other threads.

Each dynamic instance `D` of this intrinsic initiates its own asynchronous
sequence, tracked by ASYNCcnt:

`async_start<D>`
: A fictitious dynamic instance that represents the start of the asynchronous
  sequence initiated by `D` in the "xxx" scope instance.

`async_end<D>`
: A fictitious dynamic instance that represents the end of the asynchronous
  sequence initiated by `D` in the "xxx" scope instance.

```{code-block} llvm
async_start<D>       ; non-av acquire endpoint
FR                   ; non-av release fence
BA                   ; barrier-arrive %barrier
async_end<D>         ; completion endpoint
```

The operations in this sequence are *program-ordered* from top to bottom.

`async_start<D>`, `FR`, and `async_end<D>` are synchronizing operations that can
synchronize all bytes in memory. The actual synchronization is limited by the
other operations that they synchronize with. None of these operations performs a
*make-available* or *make-visible* operation.

Let `I` be a dynamic instance that initiates an async global load to LDS and is
*program-ordered* before `D` in the initiating thread. Let `E<I>` denote the last
operation in the asynchronous sequence initiated by `I`. `E<I>`
*synchronizes-with\<B\>* `async_start<D>` for all bytes `B`.

If `async_end<D>` is *completed-at* a `wait.asyncmark()` operation `Y`, it
*synchronizes-with\<B\>* `Y` for all bytes.

:::{note}
- No synchronization edge from a preceding async global store from LDS is
  established by this rule.
- The intrinsic performs one arrival at `%barrier`, not one arrival for each
  preceding async global load to LDS.
- Waiting on an asyncmark that tracks `D` establishes completion of its arrival.
  It does not establish that the barrier has received all its expected arrivals;
  that requires a barrier wait.
:::

(amdgpu-async-completed-at)=

## Completion of Async Operations

A dynamic instance `D` of an async instruction initiates a sequence of one or
more asynchronous operations, which is not related in *program-order* with `D`.
Additional primitives are required to explicitly track the completion of these
async operations.

### Using Asyncmarks

Appropriate calls to `asyncmark()` and `wait.asyncmark()` can be used to track
completion as follows.

Let `E` be the last operation in an asynchronous sequence initiated by a dynamic
instance `D`. Then `E` is *completed-at* some `wait.asyncmark()` operation `Y`
if there exists an `asyncmark()` operation `X` such that:
- `D` is *program-ordered* before `X`, and,
- `D` belongs to a stage in the mask specified by `X`, and,
- The asyncmark produced by `X` for that stage is *completed-at* `Y`.

### Using Barriers

The completion of some async operations can be tracked using {ref}`LDS memory
barriers<amdgpu-lds-memory-barriers>` as follows:

- The tensor descriptor passed to a tensor instruction instance `D` may contain
  a reference to a barrier. The asynchronous sequence initiated by `D` then
  includes a non-av release fence followed by a {ref}`barrier
  arrive<amdgpu-barrier-operations>` which can {ref}`synchronize-with through
  the barrier<amdgpu-barrier-memory-model>`.
- A call to
  {ref}`@llvm.amdgcn.ds.atomic.async.barrier.arrive.b64<amdgpu-lds-memory-barriers>`
  can track async global loads to LDS initiated in *program-order* before that
  call.

## Examples

Most examples below use the zero mask, so every asyncmark tracks every async
operation and there is only one sequence to reason about. The
{ref}`interleaved stages <amdgpu-async-interleaved-stages>` example shows what
narrower masks add.

### Uneven blocks of async operations

```c++
void foo(global int *g, local int *l) {
  // first block
  async_load_to_lds(l, g);
  async_load_to_lds(l, g);
  async_load_to_lds(l, g);
  asyncmark(0);

  // second block; longer
  async_load_to_lds(l, g);
  async_load_to_lds(l, g);
  async_load_to_lds(l, g);
  async_load_to_lds(l, g);
  async_load_to_lds(l, g);
  asyncmark(0);

  // third block; shorter
  async_load_to_lds(l, g);
  async_load_to_lds(l, g);
  asyncmark(0);

  // Wait for first block
  wait.asyncmark(2, 0);
}
```

### Software pipeline

```c++
void foo(global int *g, local int *l) {
  // first block
  asyncmark(0);

  // second block
  asyncmark(0);

  // third block
  asyncmark(0);

  for (;;) {
    wait.asyncmark(2, 0);
    // use data

    // next block
    asyncmark(0);
  }

  // flush one block
  wait.asyncmark(2, 0);

  // flush one more block
  wait.asyncmark(1, 0);

  // flush last block
  wait.asyncmark(0, 0);
}
```

### Ordinary function call

```c++
extern void bar(); // may or may not initiate async operations

void foo(global int *g, local int *l) {
    // first block
    asyncmark(0);

    // second block
    asyncmark(0);

    // function call
    bar();

    // third block
    asyncmark(0);

    // wait for the second block
    wait.asyncmark(1, 0);

    // wait for the third block, including bar()
    wait.asyncmark(0, 0);
}
```

(amdgpu-async-interleaved-stages)=
### Interleaved stages

Two kinds of async operation are in flight at once. Each is marked with a mask
naming only its own stage, so each can be waited for without waiting for the
other.

```c++
void foo(global int *g, local int *l, tensor_desc t) {
  // Start a long tensor load and mark it in the TENSOR stage alone.
  tensor_load_to_lds(l, t);
  asyncmark(TENSOR);

  // Start a short async global load to LDS and mark it in its own stage alone.
  async_load_to_lds(l, g);
  asyncmark(GLOBAL_LOAD_ASYNC_TO_LDS);

  // Wait for the async global load to LDS only. The tensor load is still in
  // flight: this wait does not name TENSOR, so it neither counts nor removes
  // that asyncmark.
  wait.asyncmark(0, GLOBAL_LOAD_ASYNC_TO_LDS);

  // perform synchronization / use the data loaded by async_load_to_lds

  // Now wait for the tensor load.
  wait.asyncmark(0, TENSOR);
}
```

Had both marks used the zero mask, the first wait would have drained the tensor
load as well, and the overlap would have been lost.

The same applies to two stages tracked by one hardware counter. Here both stages
use the async counter, but the sequences are still separate:

```c++
void foo(global int *g, local int *l) {
  async_load_to_lds(l, g);
  asyncmark(GLOBAL_LOAD_ASYNC_TO_LDS);    // X

  async_store_from_lds(g, l);
  asyncmark(GLOBAL_STORE_ASYNC_FROM_LDS); // Y

  // Completes X. Y is in a different sequence and is not completed here, even
  // though both stages are tracked by the same counter.
  wait.asyncmark(0, GLOBAL_LOAD_ASYNC_TO_LDS);
}
```

A mask need not name just one stage. An asyncmark whose mask names several is
appended to each of their sequences, and a wait whose mask names several trims
each of them:

```c++
void foo(global int *g, local int *l, tensor_desc t) {
  tensor_load_to_lds(l, t);
  async_load_to_lds(l, g);

  // One asyncmark, appended to both sequences.
  asyncmark(TENSOR | GLOBAL_LOAD_ASYNC_TO_LDS);

  // Removes the mark in GLOBAL_LOAD_ASYNC_TO_LDS. The mark in TENSOR is a
  // separate asyncmark and stays where it is.
  wait.asyncmark(0, GLOBAL_LOAD_ASYNC_TO_LDS);

  // Removes the mark in TENSOR.
  wait.asyncmark(0, TENSOR);
}
```

## Implementation notes

[This section is informational.]

### Function Calls

In general, at a function call, if the caller uses sufficient waits to track
its own async operations, the actions performed by the callee cannot affect
correctness. But inlining such a call may result in redundant waits.

```c++
void foo() {
  ...
  asyncmark(0);         // X
  ...                   // no wait.asyncmark()
}

void bar() {
  asyncmark(0);         // B
  asyncmark(0);         // C
  foo();
  wait.asyncmark(1, 0); // D
}
```

Before inlining, it is unspecified whether `X` is *completed-at* `D`, while
`C` is **not** *completed-at* `D`. The programmer can only rely on `B`
being *completed-at* `D`.

```c++
void bar() {
  asyncmark(0);         // B
  asyncmark(0);         // C
  ...
  asyncmark(0);         // X
  ...                   // no wait.asyncmark()
  wait.asyncmark(1, 0); // D
}
```

After inlining, `C` is also *completed-at* `D` and `X` is **not**
*completed-at* `D`.

Conversely, a `wait.asyncmark` call inside a callee cannot be used to track
asyncmarks from the caller, since this `wait.asyncmark` can only
observe the current sequence of the callee.

```c++
void foo() {
  ...                   // no asyncmark()
  wait.asyncmark(0, 0); // Y
  ...
}

void bar() {
  asyncmark(0);         // B
  asyncmark(0);         // C
  foo();
  wait.asyncmark(1, 0); // D
}
```

In the above example, it is unspecified whether `B` and `C` in `bar()` are
*completed-at* `Y`, because they are not included in the sequence that can be
examined at `Y`.

```c++
void bar() {
  asyncmark(0);         // B
  asyncmark(0);         // C
  ...                   // no asyncmark()
  wait.asyncmark(0, 0); // Y
  ...
  wait.asyncmark(1, 0); // D
}
```

After inlining, both `B` and `C` are *completed-at* `Y`.

### Optimization

The implementation may eliminate/modify asyncmark/wait intrinsics in the
following cases. These are just examples and not meant to be an exhaustive list.

1. An `asyncmark` operation which remains in a current sequence along every
   path that reaches the function exit.

   ```c++
   void foo() {
     ...
     asyncmark(0);         // X
     ...                   // no wait.asyncmark()
   }
   ```

   Here, the zero mask names every stage, and each independent mark remains, so
   the entire mark `X` can be eliminated. If only some sequences qualified, then
   the `asyncmark` operation could be modified to remove just those stages from
   its mask.

2. A `wait.asyncmark` which sees an empty sequence of asyncmarks along every
   path that reaches it.

   ```c++
   void foo() {
     ...                   // no asyncmark()
     wait.asyncmark(0, 0); // Y
     ...
   }
   ```

   Here, as with `X` above, every sequence qualifies, so `Y` can be eliminated.
   If only some sequences qualified, the `wait.asyncmark` could be modified to
   only wait for the relevant stages.

3. An `asyncmark` operation that has mask bits that don't refer to any
   instruction.

    ```c++
   void foo() {
     ...
     global_load_async_to_lds
     asyncmark(0)
     ...
   }
   ```

   Here, the program can be refined to -

   ```c++
   void foo() {
     ...
     global_load_async_to_lds
     asyncmark(ASYNC_LOAD_LDS)
     ...
   }
   ```

   Since only that particular stage is actually relevant to ensuring the
   completion of async operations.
