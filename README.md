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

### **4. Key Optimization Passes (Taxonomy)**

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

### **5. High-Level Roadmap**

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
