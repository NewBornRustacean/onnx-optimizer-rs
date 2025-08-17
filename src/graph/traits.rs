use crate::graph::objects::{Node, NodeId, OpKind, Tensor, ValueId};
use crate::utils::error::OnnxOptError;

/// Core graph interface for reading operations
pub trait GraphView {
    /// Get node data by NodeId
    fn node(&self, id: NodeId) -> Option<&Node>;

    /// Get tensor data by ValueId
    fn tensor(&self, id: ValueId) -> Option<&Tensor>;

    /// Get input values for a node
    fn inputs(&self, node: NodeId) -> &[ValueId];

    /// Get output values for a node  
    fn outputs(&self, node: NodeId) -> &[ValueId];

    /// Find the node that produces a given value
    fn producer(&self, value: ValueId) -> Option<NodeId>;

    /// Find all nodes that consume a given value
    fn consumers(&self, value: ValueId) -> Vec<NodeId>;

    /// Get graph-level input values
    fn graph_inputs(&self) -> &[ValueId];

    /// Get graph-level output values
    fn graph_outputs(&self) -> &[ValueId];
}

/// Graph modification interface
pub trait GraphEdit: GraphView {
    /// Add a new node to the graph
    fn add_node(&mut self, node: Node) -> NodeId;

    /// Add a new value/tensor to the graph
    fn add_value(&mut self, tensor: Tensor) -> ValueId;

    /// Remove a node from the graph
    fn remove_node(&mut self, node: NodeId);

    /// Remove a value from the graph
    fn remove_value(&mut self, value: ValueId);
}

/// Utility trait for graph analysis operations
pub trait GraphAnalysis {
    /// Get or compute the topological order of nodes
    fn get_topological_order(&mut self) -> Result<&[NodeId], OnnxOptError>;

    /// Mark the cached topology as invalid due to graph changes
    fn invalidate_topology(&mut self);

    /// Check if a node can have its constants folded
    fn can_fold_constant(&self, node_id: NodeId) -> bool;

    /// Check if a node is dead (unreachable from outputs)
    fn is_dead_node(&self, node_id: NodeId) -> bool;

    /// Check if an operation type supports constant folding
    fn supports_constant_folding(op_kind: &OpKind) -> bool;

    /// Validate graph consistency after modifications
    fn validate_graph(&self) -> Result<(), OnnxOptError>;
}

/// Utility trait for graph modification operations
pub trait GraphModification {
    /// Compute the result of a constant operation
    fn compute_constant_result(&self, node_id: NodeId) -> Result<Tensor, OnnxOptError>;

    /// Replace a node with a constant value
    fn replace_with_constant(
        &mut self,
        node_id: NodeId,
        constant: Tensor,
    ) -> Result<(), OnnxOptError>;

    /// Remove a node and update graph connectivity
    fn remove_node(&mut self, node_id: NodeId) -> Result<(), OnnxOptError>;

    /// Bypass a node by connecting its inputs directly to its outputs
    fn bypass_node(&mut self, node_id: NodeId) -> Result<(), OnnxOptError>;
}

/// Core optimization trait that all optimization levels must implement
pub trait OptimizationPass {
    /// Name of the optimization pass for logging/debugging
    fn pass_name(&self) -> &'static str;

    /// Execute the optimization pass and return the number of changes made
    fn execute(&mut self) -> Result<u32, OnnxOptError>;

    /// Check if this pass can be applied to the current graph state
    fn can_apply(&self) -> bool;

    /// Priority of this pass (lower = runs earlier)
    fn priority(&self) -> u32 {
        1
    }
}

/// Level 1: Basic optimizations that are safe and always beneficial
pub trait BasicOptimization {
    /// Fold constants throughout the graph
    ///
    /// Identifies nodes where all inputs are constants and the operation
    /// can be computed at compile time, then replaces these nodes with
    /// constant nodes containing the computed result.
    fn constant_folding(&mut self) -> Result<u32, OnnxOptError>;

    /// Remove nodes that don't affect the graph output
    ///
    /// Performs reachability analysis from graph outputs and removes
    /// any nodes that cannot be reached.
    fn dead_node_elimination(&mut self) -> Result<u32, OnnxOptError>;

    /// Remove identity nodes that don't transform their input
    ///
    /// Identifies Identity nodes and rewires the graph to bypass them.
    fn identity_elimination(&mut self) -> Result<u32, OnnxOptError>;

    /// Remove dropout nodes in inference mode
    ///
    /// Dropout nodes are no-ops during inference, so they can be safely removed.
    fn dropout_elimination(&mut self) -> Result<u32, OnnxOptError>;
}

/// Optimization level configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationLevel {
    /// Only basic, safe optimizations
    Basic,
}

impl OptimizationLevel {
    /// Check if basic optimizations are enabled
    pub fn includes_basic(&self) -> bool {
        matches!(self, OptimizationLevel::Basic)
    }
}
