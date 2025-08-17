use crate::graph::objects::{Node, NodeId, Tensor, ValueId};

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
