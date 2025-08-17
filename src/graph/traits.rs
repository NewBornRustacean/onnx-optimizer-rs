use crate::graph::error::GraphError;
use crate::graph::objects::{Node, NodeId, OpKind, Tensor, ValueId};

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
    fn get_topological_order(&mut self) -> Result<&[NodeId], GraphError>;

    /// Mark the cached topology as invalid due to graph changes
    fn invalidate_topology(&mut self);

    /// Check if a node can have its constants folded
    fn can_fold_constant(&self, node_id: NodeId) -> bool;

    /// Check if a node is dead (unreachable from outputs)
    fn is_dead_node(&self, node_id: NodeId) -> bool;

    /// Check if an operation type supports constant folding
    fn supports_constant_folding(op_kind: &OpKind) -> bool;

    /// Validate graph consistency after modifications
    fn validate_graph(&self) -> Result<(), GraphError>;
}

/// Utility trait for graph modification operations
pub trait GraphModification {
    /// Compute the result of a constant operation
    fn compute_constant_result(&self, node_id: NodeId) -> Result<Tensor, GraphError>;

    /// Replace a node with a constant value
    fn replace_with_constant(
        &mut self,
        node_id: NodeId,
        constant: Tensor,
    ) -> Result<(), GraphError>;

    /// Remove a node and update graph connectivity
    fn remove_node(&mut self, node_id: NodeId) -> Result<(), GraphError>;

    /// Bypass a node by connecting its inputs directly to its outputs
    fn bypass_node(&mut self, node_id: NodeId) -> Result<(), GraphError>;
}
