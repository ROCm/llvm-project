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

### Current Sequence

The abstract machine maintains a sequence of asyncmarks during the execution of
a function body, which excludes any asyncmarks produced by calls to other
functions encountered in the currently executing function. The state of this
sequence at each program point in the function is called the *current sequence*.

### `@llvm.amdgcn.asyncmark()`

Produces an asyncmark and appends it to the current sequence.

### `@llvm.amdgcn.wait.asyncmark(i16 %N)`

Ensures that the length of the current sequence is at most `N` by removing
asyncmarks from the start of the sequence if it is more than `N`.

This operation is also an acquire operation without `MakeVisible` semantics.

### Completion of Asyncmarks

An `asyncmark()` operation `X` that produces an asyncmark `M` is
*completed-at* a `wait.asyncmark()` operation `Y` in the same function body
if:

- `X` is *program-ordered* before `Y`, and
- `M` is not in the current sequence at any operation `Z` that immediately
  follows `Y` in *program-order*.

(amdgpu-lds-memory-barriers)=

## LDS Memory Barriers

An LDS memory barrier is a barrier that uses LDS for its state. This barrier can
track asynchronous LDS DMA loads and tensor operations, but not LDS DMA stores.

```llvm
void @llvm.amdgcn.ds.atomic.async.barrier.arrive.b64(ptr addrspace(3) %barrier)
```

This intrinsic initiates a {ref}`barrier arrive<amdgpu-barrier-operations>`
operation scheduled after the asynchronous loads previously initiated by the
same thread. Once those loads complete, the operation arrives at `%barrier`. A
{ref}`barrier wait<amdgpu-async-completed-at>` can then track the loads'
completion. The barrier-arrive operation is itself tracked by `AsyncCNT` and can
be included in an {ref}`asyncmark<amdgpu-asyncmarks>`.

(amdgpu-async-completed-at)=

## Completion of Async Operations

An async operation executes outside the thread that initiated it, i.e., it is
not related in *program-order* with any other operations from that thread. But
a thread that depends on the side-effects of `A` can use an asyncmark or barrier
to ensure that `A` is *completed-at* some operation in that thread.

### Using Asyncmarks

Some async operations use asyncmarks to notify completion. Such an async
operation `A` *initiated-by* an instruction `I` is *completed-at* some
`wait.asyncmark()` operation `Y` if there exists an `asyncmark()` operation `X`
such that:
- `I` is *program-ordered* before `X`, and
- `X` is *completed-at* `Y`.

### Using Barriers

The completion of some async operations can be tracked using {ref}`LDS memory
barriers<amdgpu-lds-memory-barriers>` as follows:

- The tensor descriptor passed to a tensor instruction `X` may contain a
  reference to a barrier. When `X` initiates `A`, it also initiates a
  {ref}`barrier arrive<amdgpu-barrier-operations>` to be performed after `A`.
- When a thread executes an LDS DMA instruction `X`, it may executed a call to
  `@llvm.amdgcn.ds.atomic.async.barrier.arrive.b64`
  {ref}`intrinsic<amdgpu-lds-memory-barriers>` program-ordered after `X`. This
  initiates a {ref}`barrier arrive<amdgpu-barrier-operations>` to be performed
  after the async operation `A` initiated by `X`.

A thread that depends on the side-effects of `A` performs a {ref}`barrier
wait<amdgpu-barrier-operations>` operation `W` on the barrier. `A` is said to be
*completed-at* `W` when the barrier completes.

## Examples

### Uneven blocks of async operations

```c++
void foo(global int *g, local int *l) {
  // first block
  async_load_to_lds(l, g);
  async_load_to_lds(l, g);
  async_load_to_lds(l, g);
  asyncmark();

  // second block; longer
  async_load_to_lds(l, g);
  async_load_to_lds(l, g);
  async_load_to_lds(l, g);
  async_load_to_lds(l, g);
  async_load_to_lds(l, g);
  asyncmark();

  // third block; shorter
  async_load_to_lds(l, g);
  async_load_to_lds(l, g);
  asyncmark();

  // Wait for first block
  wait.asyncmark(2);
}
```

### Software pipeline

```c++
void foo(global int *g, local int *l) {
  // first block
  asyncmark();

  // second block
  asyncmark();

  // third block
  asyncmark();

  for (;;) {
    wait.asyncmark(2);
    // use data

    // next block
    asyncmark();
  }

  // flush one block
  wait.asyncmark(2);

  // flush one more block
  wait.asyncmark(1);

  // flush last block
  wait.asyncmark(0);
}
```

### Ordinary function call

```c++
extern void bar(); // may or may not initiate async operations

void foo(global int *g, local int *l) {
    // first block
    asyncmark();

    // second block
    asyncmark();

    // function call
    bar();

    // third block
    asyncmark();

    // wait for the second block
    wait.asyncmark(1);

    // wait for the third block, including bar()
    wait.asyncmark(0);
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
  asyncmark();       // X
  ...                // no wait.asyncmark()
}

void bar() {
  asyncmark();       // B
  asyncmark();       // C
  foo();
  wait.asyncmark(1); // D
}
```

Before inlining, it is unspecified whether `X` is *completed-at* `D`, while
`C` is **not** *completed-at* `D`. The programmer can only rely on `B`
being *completed-at* `D`.

```c++
void bar() {
  asyncmark();       // B
  asyncmark();       // C
  ...
  asyncmark();       // X
  ...                // no wait.asyncmark()
  wait.asyncmark(1); // D
}
```

After inlining, `C` is also *completed-at* `D` and `X` is **not**
*completed-at* `D`.

Conversely, a `wait.asyncmark` call inside a callee cannot be used to track
asyncmarks from the caller, since this `wait.asyncmark` can only
observe the current sequence of the callee.

```c++
void foo() {
  ...                // no asyncmark()
  wait.asyncmark(0); // Y
  ...
}

void bar() {
  asyncmark();       // B
  asyncmark();       // C
  foo();
  wait.asyncmark(1); // D
}
```

In the above example, it is unspecified whether `B` and `C` in `bar()` are
*completed-at* `Y`, because they are not included in the sequence that can be
examined at `Y`.

```c++
void bar() {
  asyncmark();       // B
  asyncmark();       // C
  ...                // no asyncmark()
  wait.asyncmark(0); // Y
  ...
  wait.asyncmark(1); // D
}
```

After inlining, both `B` and `C` are *completed-at* `Y`.

### Optimization

The implementation may eliminate asyncmark/wait intrinsics in the following
cases. These are just examples and not meant to be an exhaustive list.

1. An `asyncmark` operation which remains in the current sequence along every
   path that reaches the function exit.

   ```c++
   void foo() {
     ...
     asyncmark();       // X
     ...                // no wait.asyncmark()
   }
   ```

   Here, `X` can be eliminated.

2. A `wait.asyncmark` which sees an empty sequence of asyncmarks along every
   path that reaches it.

   ```c++
   void foo() {
     ...                // no asyncmark()
     wait.asyncmark(0); // Y
     ...
   }
   ```

   Here, `Y` can be eliminated.
