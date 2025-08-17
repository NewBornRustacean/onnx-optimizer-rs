# onnx-optimizer-rs

## **A Standalone ONNX Graph Optimizer**

### **1. Goals**

*   Easy-to-use ONNX model optimization tool for everyone.
*   To develop a Rust-based CLI tool that takes an ONNX file as input, applies a series of optimization passes, and outputs a faster, more efficient ONNX file.

### **2. Core Features & Technical Approach**

*   **Input**: `model.onnx` file
*   **Output**: `model.optimized.onnx` file (an optimized, valid ONNX file)
*   **Architecture**:
    1.  **Parser**: Reads an ONNX file and deserializes it into an internal computation graph.
    2.  **Optimizer Core**: Applies a sequence of **Optimization Passes** to the graph.
    3.  **Emitter**: Serializes the optimized internal graph back into a valid ONNX file.

### **3. Key Optimization Passes (Taxonomy)**

Optimizations will be implemented in levels, with each level including the passes from the previous ones.

*   **Level 1: Basic Optimizations**
    *   **Constant Folding**: Pre-computes parts of the graph that rely only on constant initializers at compile time.
    *   **Redundant Node Elimination**: Removes nodes that do not affect the graph's semantics, such as `Identity` or `Dropout` (in inference mode).
    *   **Simple Fusion**: Fuses simple sequential nodes into a single operator, like merging `Conv` and `Add` into a `Conv` with bias.

*   **Level 2: Extended Operator Fusion**
    *   **Vertical Fusion**: Merges a sequence of operations (e.g., `Conv → BatchNorm → ReLU`) into a single `FusedOp` to reduce memory access and kernel launch overhead.
    *   **Complex Fusion**: Fuses more complex patterns found in modern architectures, such as `LayerNormalization`, `GELU`, and `Attention` blocks.

*   **Level 3: Advanced (Future Scope)**
    *   **Layout Optimization**: Changes tensor memory layouts (e.g., `NCHW` → `NHWC` or `NCHWc`) to match hardware-specific requirements for better performance.
    *   **Approximation**: Replaces computationally expensive functions with faster, approximate equivalents, trading a minor amount of precision for significant speed.

### **4. High-Level Roadmap**

*   **Milestone 1 (Foundation)**
    *   Set up the Rust project and implement the basic CLI interface.
    *   Complete the **Parser/Emitter** for reading and writing ONNX files.
    *   Implement and test **Level 1** passes: `Constant Folding` and `Node Elimination`.

*   **Milestone 2 (Core Optimizations)**
    *   Implement the graph pattern matching and rewriting engine.
    *   Implement key **Level 2** fusions (e.g., `Conv+ReLU`) and verify numerical correctness.
    *   Test and benchmark performance on key models (e.g., MobileNet, ResNet).

*   **Milestone 3 (Usability & Release)**
    *   Add CLI options for users to select optimization levels or specific passes.
    *   Write comprehensive documentation (README) and prepare for the initial open-source release.

### **6. Success Metrics**

*   **Correctness**: Does the model's output remain (nearly) identical to the original after optimization?
*   **Performance**: Does the optimized model show a significant (e.g., >10%) inference speedup on standard runtimes like ONNX Runtime?
*   **Usability**: Can anyone easily optimize their model using a simple and intuitive CLI command?


## Design Decision

### Architecture Overview: Petgraph-based Graph Representation

After careful evaluation, we adopted **[petgraph](https://docs.rs/petgraph/latest/petgraph/)'s StableGraph** as our core graph data structure, eliminating the custom Arena-based implementation. This decision provides significant benefits in terms of code simplicity, maintenance, and access to proven graph algorithms.

```
     ┌──────────────────── graph/ ────────────────────── utils/ ──────────┐
     │                                                                    │
     │  ┌─────────────────────────────────────────────────────────────────┐ │
     │  │                    objects.rs                                   │ │
     │  │                                                                 │ │
     │  │  ┌──── Core Types ────┐        ┌──── ONNX Mapping ────┐        │ │
     │  │  │                    │        │                      │        │ │
     │  │  │ NodeId(NodeIndex) ◄┼────────┼─► petgraph::NodeIndex│        │ │
     │  │  │ ValueId(u32)       │        │                      │        │ │
     │  │  │ Tensor             │        │ OpKind ◄─────────────┼─► ONNX │ │
     │  │  │ Node               │        │ NodeAttrValue        │   Ops  │ │
     │  │  │ DataType           │        │ TensorData           │        │ │
     │  │  └────────────────────┘        └──────────────────────┘        │ │
     │  └─────────────────────────────────────────────────────────────────┘ │
     │                                  │                                   │
     │                                  ▼                                   │
     │  ┌─────────────────────────────────────────────────────────────────┐ │
     │  │                         Graph                                   │ │
     │  │                                                                 │ │
     │  │   ┌─ Petgraph Core ─────────────────────────────────────────┐   │ │
     │  │   │                                                         │   │ │
     │  │   │  nodes: StableGraph<Node, ()>  ◄─ Nodes + Dependencies  │   │ │
     │  │   │           │                       (automatic edge mgmt) │   │ │
     │  │   │           │                                             │   │ │
     │  │   │           ▼                                             │   │ │
     │  │   │  ┌─ Built-in Algorithms ─┐                             │   │ │
     │  │   │  │ ✓ Topological Sort     │                             │   │ │
     │  │   │  │ ✓ Cycle Detection      │                             │   │ │
     │  │   │  │ ✓ Graph Traversal      │                             │   │ │
     │  │   │  │ ✓ Strongly Connected   │                             │   │ │
     │  │   │  └────────────────────────┘                             │   │ │
     │  │   └─────────────────────────────────────────────────────────┘   │ │
     │  │                                                                 │ │
     │  │   ┌─ Value Storage ─────────────────────────────────────────┐   │ │
     │  │   │                                                         │   │ │
     │  │   │  values: HashMap<ValueId, Tensor>  ◄─ Tensor Storage    │   │ │
     │  │   │                                                         │   │ │
     │  │   └─────────────────────────────────────────────────────────┘   │ │
     │  │                                                                 │ │
     │  │   ┌─ Graph Metadata ────────────────────────────────────────┐   │ │
     │  │   │                                                         │   │ │
     │  │   │  graph_input_values: Vec<ValueId>                       │   │ │
     │  │   │  graph_output_values: Vec<ValueId>                      │   │ │
     │  │   │  next_value_id: u32                                     │   │ │
     │  │   │                                                         │   │ │
     │  │   └─────────────────────────────────────────────────────────┘   │ │
     │  └─────────────────────────────────────────────────────────────────┘ │
     │                                                                    │
     │  ┌─────────────────────────────────────────────────────────────────┐ │
     │  │                    Trait Implementation                         │ │
     │  │                                                                 │ │
     │  │  ┌─── GraphView (Read-only) ──┐  ┌─── GraphEdit (Mutation) ───┐ │ │
     │  │  │                            │  │                            │ │ │
     │  │  │ node(id) ─────────────────┐│  │ add_node() ──┐             │ │ │
     │  │  │ tensor(id)                ││  │ remove_node() │             │ │ │
     │  │  │ inputs(id)                ││  │ add_value()   │ manages     │ │ │
     │  │  │ outputs(id)      calls    ││  │ remove_value()│ petgraph    │ │ │
     │  │  │ producer(id) ─────────────┼┼──┼──────────────┘ edges        │ │ │
     │  │  │ consumers(id)             ││  │               automatically │ │ │
     │  │  │ graph_inputs()            ││  │                            │ │ │
     │  │  │ graph_outputs()           ││  │                            │ │ │
     │  │  └───────────────────────────┘│  └────────────────────────────┘ │ │
     │  │                               │                                │ │
     │  │                               ▼                                │ │
     │  │                    petgraph::StableGraph                       │ │
     │  │                         method calls                           │ │
     │  └─────────────────────────────────────────────────────────────────┘ │
     │                                                                    │
     │  ┌─────────────────────────────────────────────────────────────────┐ │
     │  │                      executor.rs                               │ │
     │  │                                                                 │ │
     │  │  OptimizationExecutor {                                         │ │
     │  │    graph: Graph ◄─── owns and modifies Graph                    │ │
     │  │    config: OptimizationConfig                                   │ │
     │  │    stats: OptimizationStats                                     │ │
     │  │  }                                                              │ │
     │  │                                                                 │ │
     │  │  ┌─ Optimization Passes ─────────────────────────────────────┐  │ │
     │  │  │                                                           │  │ │
     │  │  │ • constant_folding()    ← utilizes petgraph traversal     │  │ │
     │  │  │ • dead_node_elimination() ← uses topological_sort()       │  │ │
     │  │  │ • identity_elimination()                                   │  │ │
     │  │  │ • operator_fusion()     ← pattern matching on graph       │  │ │
     │  │  │                                                           │  │ │
     │  │  └───────────────────────────────────────────────────────────┘  │ │
     │  └─────────────────────────────────────────────────────────────────┘ │
     └────────────────────────────────────────────────────────────────────┘
```

### Key Design Benefits

#### 🎯 **Simplified Architecture**
- **Single Source of Truth**: StableGraph manages both nodes and connectivity
- **No Manual Synchronization**: Eliminated Arena<->graph sync issues  
- **Reduced Complexity**: Removed ~200 lines of manual graph management code

#### 🚀 **Proven Graph Algorithms**
- **Topological Sorting**: Built-in `petgraph::algo::toposort()`
- **Cycle Detection**: Automatic validation during graph construction
- **Graph Traversal**: DFS, BFS, and custom traversals readily available
- **Strongly Connected Components**: Available for advanced optimizations

#### 🔧 **Memory Efficiency**
- **Direct Storage**: Nodes stored directly in StableGraph, not duplicated
- **Stable Indices**: `NodeIndex` remains valid after node removal operations
- **Value Deduplication**: Single HashMap for tensor storage

### Core Components

#### 1. **Graph Structure**
```rust
pub struct Graph {
    /// Core petgraph structure - stores nodes with dependency edges
    nodes: StableGraph<Node, ()>,
    
    /// Value/tensor storage
    values: HashMap<ValueId, Tensor>,
    
    /// Graph-level metadata  
    graph_input_values: Vec<ValueId>,
    graph_output_values: Vec<ValueId>,
    
    /// Value ID generator (NodeId managed by petgraph)
    next_value_id: u32,
}
```

#### 2. **Automatic Edge Management**
- **Input Dependencies**: Edges automatically created based on `Node.inputs`
- **Producer-Consumer**: Relationship maintained through graph structure
- **Dataflow Integrity**: Graph edges represent actual tensor flow

#### 3. **Standard Graph Operations**
```rust
// All operations work directly with petgraph
let topo_order = petgraph::algo::toposort(&graph.nodes, None)?;
let is_cyclic = petgraph::algo::is_cyclic_directed(&graph.nodes);
let sccs = petgraph::algo::tarjan_scc(&graph.nodes);
```

### Benefits
#### petgraph based:
- ✅ Automatic relationship tracking through graph edges
- ✅ Zero synchronization - single source of truth
- ✅ Industry-standard algorithms out of the box
- ✅ Extensive graph analysis capabilities

### Future Extensibility

The petgraph foundation enables advanced optimizations:
- **Pattern Matching**: Graph isomorphism for operator fusion
- **Dataflow Analysis**: Advanced dependency tracking
- **Memory Layout**: Optimal tensor placement algorithms
- **Parallelization**: Safe parallel graph analysis (read-only operations)

## References
- [onnx optimizer: from onnx org.](https://github.com/onnx/optimizer) - we have similar goals with this awesome work.
- [onnx simplifier: inspired by onnxoptimizer](https://github.com/daquexian/onnx-simplifier)
- [petgraph: graph related crate](https://docs.rs/petgraph/latest/petgraph/)