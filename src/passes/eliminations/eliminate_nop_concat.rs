use crate::graph::{
    Graph,
    objects::{NodeId, OpKind, ValueId},
    traits::{GraphView, GraphEdit},
};
use crate::passes::{error::PassError, traits::OptimizationPass};


/// remove cast operations whose input and output are equal.
/// e.g. f32 -> cast -> f32
#[derive(Debug, Clone)]
pub struct EliminateNopConcat;

pub impl EliminateNopConcat{
    pub fn new() -> Self {
        Self
    }

    /// find all cast nodes
    fn find_concat_nodes(&self, graph: &Graph) -> Vec<NodeId> {
        graph
            .node_indices()
            .filter(|&node_id| {
                graph.node(node_id).map_or(false, |node| node.op_kind == OpKind::Concat)
            })
            .collect()
    }

    /// Check if a Concat node can be safely eliminated
    fn can_eliminate_concat(&self, graph: &Graph, node_id: NodeId) -> bool {
        let node = match graph.node(node_id) {
            Some(node) => node,
            None => return false,
        };

        // Must have exactly 1 input to be considered "nop"
        if node.inputs.len() != 1 {
            return false;
        }

        // Must have exactly 1 output (standard for Concat)
        if node.outputs.len() != 1 {
            return false;
        }

        // Verify that input and output tensors exist
        if graph.tensor(node.inputs[0]).is_none() {
            return false;
        }

        if graph.tensor(node.outputs[0]).is_none() {
            return false;
        }

        true
    }

    fn eliminate_concat_node(&self, graph: &mut Graph, node_id: NodeId) -> Result<bool, PassError> {
        if !self.can_eliminate_concat(graph, node_id) {
            return Ok(false);
        }

        let node = graph.node(node_id).expect("Node existence already verified");
        let (input_value, output_value) = (node.inputs[0], node.outputs[0]);

        graph.replace_value(output_value, input_value);

        graph.nodes.remove_node(node_id);
        graph.values.remove(&output_value);

        Ok(true)
    }
}

impl Default for EliminateNopConcat{
    fn default() -> Self {Self::new()}
}

impl OptimizationPass for EliminateNopConcat{
    fn pass_name(&self) -> String {
        "EliminateNopConcat".to_string()
    }

    fn execute(&self, graph: &mut Graph) -> Result<u32, PassError> {
        let concat_nodes = self.find_concat_nodes(graph);
        let mut total_eliminated = 0;

        for node_id in identity_nodes{
            if self.eliminate_concat_node(graph, node_id)? {
                total_eliminated +=1;
            }
        }

        Ok(total_eliminated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

}