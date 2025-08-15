use crate::graph::objects::{Node, NodeId, Tensor, ValueId};

pub trait GraphView {
    fn node(&self, id: NodeId) -> Option<&Node>;
    fn tensor(&self, id: ValueId) -> Option<&Tensor>;

    fn inputs(&self, node: NodeId) -> &[ValueId];
    fn outputs(&self, node: NodeId) -> &[ValueId];

    fn producer(&self, value: ValueId) -> Option<NodeId>;
    fn consumers(&self, value: ValueId) -> &[NodeId];

    fn graph_inputs(&self) -> &[ValueId];
    fn graph_outputs(&self) -> &[ValueId];
}

pub trait GraphEdit: GraphView {
    fn add_node(&mut self, node: Node) -> NodeId;
    fn add_value(&mut self, tensor: Tensor) -> ValueId;

    fn remove_node(&mut self, node: NodeId);
    fn remove_value(&mut self, value: ValueId);

    fn invalidate_topology(&mut self);
}
