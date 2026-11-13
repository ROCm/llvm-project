# LLVM Debug Information and DWARF 6

- [LLVM Debug Information and DWARF 6](#llvm-debug-information-and-dwarf-6)
  - [Purpose](#purpose)
  - [Objectives](#objectives)
  - [Design](#design)
  - [Risks](#risks)
  - [Development Plan](#development-plan)
    - [Step 1: RFC](#step-1-rfc)
    - [Step 2: Basic Scaffolding](#step-2-basic-scaffolding)
    - [Step 3: Parallel Development](#step-3-parallel-development)
  - [References](#references)

## Purpose

This is a proposal for changes to the way LLVM represents debug information internally to address a number of limitations of the current approach. In particular, changes to allow better support for the upcoming DWARF 6 standard. It is being developed collaboratively by multiple parties that work in the debugging space.

## Objectives

- Rigor
  - Current LLVM debug information is not fully documented
  - It does not follow a consistent formalized model to the same extent as the rest of LLVM IR
  - Desire to create something that is rigorously defined, composable and consistent
- DWARF 6
  - DWARF 6 draft is adopting a new consistent model to support describing locations of variables
    - Based on a uniform concept of storage
    - Together with composable operations to create expressions that define locations
  - This is considerably different to DWARF 5 upon which the current LLVM debug expressions are based
  - Need LLVM debug information to adopt the DWARF 6 conceptual changes to make DWARF 6 generation viable
- Simplification
  - The cleaner model of DWARF 6 provides the opportunity for LLVM to simplify its debug information approach
    - It can be completely location based, rather than the mixture of locations and values that DWARF 5 had
- Functional correctness and completeness
  - It should fix many deficiencies in producing correct debug information
  - It should  enable preserving debug information better through optimizations
- Since the new representation will be more functionally capable
  - It is considered acceptable if it adds additional compile space/time
    - Any cost is only paid when generating debug information
    - The cost should scale with the level of optimization
      - Since expressions are still small and often shared at -O0

## Design

- Current LLVM DIExpression debug information is based on a vector of 64 bit integers that encodes operations that have similarity to DWARF 5
  - Any changes to the debug information requires parsing of the stream of integers in the vector
    - Inconvenience that:
      - DWARF 5 does not support allowing locations in the middle of expressions
      - DWARF 5 operations do not maintain explicit type information
  - Adding type information to DIExpression is syntactically hard
    - Dependencies in the DIExpression flow implicitly through a stack, not explicitly in a graph

- Proposal is to move to a directed acyclic graph (DAG) based IR
  - Where the expressions are trees of nodes that operate on typed locations in a uniform manner
  - Provides a clear path to solving outstanding DIExpression issues
    - Simpler to inspect the expression, which helps in making correct decisions when optimizing
    - Stronger type information at every point in the expression
      - This will address current issues where illegal DWARF 5 is created due to mismatched types

- Motivation for shifting from a mixture of values and locations as done in current LLVM IR and DWARF 5, to only locations in proposed LLVM IR:
  - Keeping LLVM only in terms of locations:
    - Simplifies the set of operations
    - Provides the potential of preserving debug information through more advanced optimizations than can currently be supported
      - If necessary, future DWARF changes or extensions could then be proposed
  - Reasoning:
    - A source variable can be described by a debug information expression that computes its source language value from its optimized representation contents
      - It is possible for the client debugger to use this to present the source language value of the variable
        - Even in the presence of optimizations that result in the stored contents having a different representation value to the source language value
      - But there is no direct way the the client debugger can change the value
        - Variables can only be changed if they are described as locations
    - DWARF introduced the concept of implicit values which are locations that have a literal value
      - So all the arithmetic operations can be defined to act directly on locations instead of on values
        - These operations work on all kinds of locations
        - Locations are a position within an underlying storage:
          - Physical storage such as memory or registers
          - Composite storage that maps a number of other storage parts to present the illusion of the source language type
          - Implicit storage with a fixed value
          - Undefined storage for parts that are optimized away
      - Such operations implicitly fetch the value from the location, do the operation, and then conceptually construct a location to implicit storage that holds the result
      - So a single set of operations can work on any location
        - Avoids the need to introduce a parallel set of machinery to handle values separately from locations as done in DWARF
      - DWARF 6 still requires separate value operations
        - But there is a mechanical way to convert from a representation that only has locations, into one that uses locations and values

- Make the graph view the only view
  - Avoids the inconvenience and cost of swapping between a linear byte stream into a semantic graph and back
  - Use existing LLVM Metadata to encode the graph
    - Each operation in a DIExpression becomes a graph node in the Metadata
    - Have types on the operations
      - Allows to have robust checks on invariants
        - For example, can express sign/zero extend and truncation as operations that change type
  - Graph nodes would be based on typed locations only
    - Values can be supported as locations with implicit values (same as is present in DWARF 5)
      - Operations can act on typed implicit locations and produce new typed implicit locations
        - This is different to DWARF 5 operations
          - But there is a simple correspondence
          - Having only typed locations simplifies the formal model over DWARF 5's implicitly typed values and locations that can only be present at the root of the expression

- Such a structural change is significant
  - Need to be able to work upstream in an incremental manner and still have confidence each step preserves correctness
  - Since the structure is moving from a linear byte stream to a graph, it is difficult to produce output dumps that can be simply compared to current DIExpression dumps when performing validation
  - So want to have a way to compare the old and new forms
    - Use an SMT solver (think Alive2 for DWARF expressions)
      - This can confirm that the new way makes an equivalent expression to the old way
        - Avoids having to get the output syntactically the same as the old

- Major components
  - Ingestion of old DIExpression
    - Can auto-upgrade an old DIExpression to new Metadata DAG
    - Maybe also provide a way to lower Metadata DAG back to the old DIExpression
      - Would allow to compare results symbolically by round tripping
        - Round trip testing is limited to confirming old expressions still work in the new expression
          - But additional testing will be required for the new DWARF 6 features that are introduced
  - Lowering of Metadata DAG to concrete debug formats
    - DWARF <=5
      - Must lower non-leaf locations into values
    - DWARF 6
    - CodeView (MS PDB)
    - DXIL
      - A frozen DIExpression form
  - Traversing and editing the Metadata DAG
    - A set of helper operations that can be used by optimization passes when they need to update the debug information to match the optimization

- Allows LLVM's internal representation of debug information to be agnostic to what concrete debug format it will lower to

## Risks

- Will it be too slow or use too much memory?
  - Have done early explorations that indicate only modest increases

- Feel there is some headroom available for reasonable increases
  - ~20x regression would represent under 1% of RSS

- The Metadata graph may result in less spatial locality
  - May cause execution cycles to grow more than is intuitive

## Development Plan

### Step 1: RFC

- Create and iterate on an RFC to get feedback

### Step 2: Basic Scaffolding

- Get basic scaffolding committed
  - Creating nodes of Metadata DAG
  - Pattern matching and rewriting
  - Uniqueing
  - IR/Bitcode/MIR round-tripping

### Step 3: Parallel Development

- Multiple developers can work in parallel on:
  - Auto upgrade from DIExpression to Metadata DAG
  - Generate component to convert DAG back to DIExpression
    - Use in conjunction with SMT solver to validate the new matches semantics of old
    - Also needed for DXIL
  - Peephole to canonicalize expression
    - Replaces constant folding
  - DWARF lowering
    - Both DWARF 5 and DWARF 6
  - CodeView lowering
  - Update optimization/pass modules
    - Approach
      - Update each pass to add code to use the new expression construction and editing APIs
        - Initially leave code for old expression API
          - If the module receives the old expression IR it will use the old path
          - If the module receives the new expression IR is will use the new path
          - Once all modules are converted and accepted, then the old expression code can be removed across the whole compiler
    - Testing
      - Use unit testing
      - Can test end to end once the set of passes converted include what is necessary for low optimization levels
      - Then expand the set of optimizations that can be tested
    - Order of update
      - Focus on minimal set of passes needed to get low optimization level working end to end
        - SelectionDAG/Fast/Global ISels
        - Register allocation
        - `salvageDebugInfo`
        - `replaceAllDbgUsesWith`
        - (X86) `describeLoadedValue`
        - Other target-specific direct DIExpression manipulation...
      - Then expand the set of optimizations supported
        - SROA
        - Loop strength reduction
        - (X86) `X86OptimizeLEAs`
        - And many more, including target-specific ones...
      - When a certain pipeline for a specific target is fully supported, can switch the default
        - For example, a reasonable first goal might be end-to-end support for X86 at -O0, which would require support in:
          - FastISel
          - FastRegAlloc
          - `salvageDebugInfo`
          - `replaceAllDbgUsesWith`
          - (X86) `describeLoadedValue`
      - Work to support SPIR-V
        - Can be done independently once basic signs of end-to-end life exist

## References

- [LLVM IR debug information (Source Level Debugging with LLVM)](./SourceLevelDebugging.md)
- [DWARF 5 standard (final)](https://dwarfstd.org/doc/DWARF5.pdf)
- [DWARF 6 draft (working draft snapshots)](https://snapshots.sourceware.org/dwarfstd/dwarf-spec/)

