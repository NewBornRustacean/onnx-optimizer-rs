use crate::graph::{
    Graph,
    objects::{NodeId, OpKind, ValueId},
    traits::{GraphView, GraphEdit},
};
use crate::passes::{error::PassError, traits::OptimizationPass};

/// Eliminates Identity nodes by connecting their inputs directly to their outputs
///
/// Identity nodes are no-op operations that simply pass their input through unchanged.
/// This pass removes them and reconnects the graph to eliminate unnecessary nodes.
#[derive(Debug, Clone)]
pub struct EliminateIdentity;

impl EliminateIdentity {
    pub fn new() -> Self {
        Self
    }

    /// Find all Identity nodes in the graph
    fn find_identity_nodes(&self, graph: &Graph) -> Vec<NodeId> {
        graph
            .node_indices()
            .filter(|&node_id| {
                graph.node(node_id).map_or(false, |node| node.op_kind == OpKind::Identity)
            })
            .collect()
    }

    /// Check if an Identity node can be safely eliminated
    fn can_eliminate_identity(&self, graph: &Graph, node_id: NodeId) -> bool {
        let node = match graph.node(node_id) {
            Some(node) => node,
            None => return false,
        };

        // Identity node should have exactly 1 input and 1 output
        if node.inputs.len() != 1 || node.outputs.len() != 1 {
            return false;
        }

        // Check if the input tensor exists
        if graph.tensor(node.inputs[0]).is_none() {
            return false;
        }

        // Check if the output tensor exists
        if graph.tensor(node.outputs[0]).is_none() {
            return false;
        }

        true
    }



    /// Eliminate a single Identity node
    fn eliminate_identity_node(
        &self,
        graph: &mut Graph,
        node_id: NodeId,
    ) -> Result<bool, PassError> {
        if !self.can_eliminate_identity(graph, node_id) {
            return Ok(false);
        }

        let node = graph.node(node_id).expect("Node existence already verified");
        let (input_value, output_value) = (node.inputs[0], node.outputs[0]);

        // Replace all uses of output_value with input_value
        graph.replace_value(output_value, input_value);

        // Clean up - remove node and unused value
        graph.nodes.remove_node(node_id);
        graph.values.remove(&output_value);

        Ok(true)
    }
}

impl Default for EliminateIdentity {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPass for EliminateIdentity {
    fn pass_name(&self) -> String {
        "EliminateIdentity".to_string()
    }

    fn execute(&self, graph: &mut Graph) -> Result<u32, PassError> {
        let identity_nodes = self.find_identity_nodes(graph);
        let mut total_eliminated = 0;

        for node_id in identity_nodes {
            if self.eliminate_identity_node(graph, node_id)? {
                total_eliminated += 1;
            }
        }

        Ok(total_eliminated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::traits::GraphEdit;
    use crate::graph::{
        Graph,
        objects::{DataType, Node, OpKind, Tensor, ValueId},
    };

    fn create_test_graph_with_identity() -> (Graph, NodeId, ValueId, ValueId) {
        let mut graph = Graph::new();

        // Create input and output tensors
        let input_tensor = Tensor::new(DataType::Float32);
        let output_tensor = Tensor::new(DataType::Float32);

        let input_id = graph.add_value(input_tensor);
        let output_id = graph.add_value(output_tensor);

        // Create identity node
        let identity_node = Node::new(OpKind::Identity)
            .with_inputs(vec![input_id])
            .with_outputs(vec![output_id]);

        let node_id = graph.add_node(identity_node);

        (graph, node_id, input_id, output_id)
    }

    #[test]
    fn test_find_identity_nodes() {
        let (graph, node_id, _, _) = create_test_graph_with_identity();
        let pass = EliminateIdentity::new();

        let identity_nodes = pass.find_identity_nodes(&graph);
        assert_eq!(identity_nodes.len(), 1);
        assert_eq!(identity_nodes[0], node_id);
    }

    #[test]
    fn test_can_eliminate_identity() {
        let (graph, node_id, _, _) = create_test_graph_with_identity();
        let pass = EliminateIdentity::new();

        assert!(pass.can_eliminate_identity(&graph, node_id));
    }

    #[test]
    fn test_eliminate_identity_node() {
        let (mut graph, node_id, _, _) = create_test_graph_with_identity();
        let pass = EliminateIdentity::new();

        let result = pass.eliminate_identity_node(&mut graph, node_id);
        assert!(result.is_ok());
        assert!(result.unwrap()); // Should return true indicating elimination was possible
    }

    #[test]
    fn test_execute_pass() {
        let (mut graph, _, _, _) = create_test_graph_with_identity();
        let pass = EliminateIdentity::new();

        let result = pass.execute(&mut graph);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 1); // Should eliminate 1 identity node
    }

    #[test]
    fn test_identity_elimination_with_consumers() {
        use crate::graph::traits::GraphEdit;
        
        let mut graph = Graph::new();
        
        // Create tensors
        let input_tensor = Tensor::new(DataType::Float32);
        let identity_output_tensor = Tensor::new(DataType::Float32);
        let final_output_tensor = Tensor::new(DataType::Float32);
        
        let input_id = graph.add_value(input_tensor);
        let identity_output_id = graph.add_value(identity_output_tensor);
        let final_output_id = graph.add_value(final_output_tensor);
        
        // Create Identity node: input -> identity_output
        let identity_node = Node::new(OpKind::Identity)
            .with_inputs(vec![input_id])
            .with_outputs(vec![identity_output_id]);
        let identity_node_id = graph.add_node(identity_node);
        
        // Create consumer node: identity_output -> final_output
        let consumer_node = Node::new(OpKind::Relu)
            .with_inputs(vec![identity_output_id])
            .with_outputs(vec![final_output_id]);
        let consumer_node_id = graph.add_node(consumer_node);
        
        // Verify initial state
        assert_eq!(graph.node(consumer_node_id).unwrap().inputs[0], identity_output_id);
        
        // Execute identity elimination
        let pass = EliminateIdentity::new();
        let result = pass.execute(&mut graph);
        
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 1);
        
        // Verify the consumer now uses the input directly
        assert_eq!(graph.node(consumer_node_id).unwrap().inputs[0], input_id);
        
        // Verify the identity node is gone
        assert!(graph.node(identity_node_id).is_none());
        
        // Verify the identity output tensor is gone
        assert!(graph.tensor(identity_output_id).is_none());
    }

    #[test]
    fn test_empty_graph() {
        let mut graph = Graph::new();
        let pass = EliminateIdentity::new();

        let result = pass.execute(&mut graph);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0); // No identity nodes to eliminate
    }
}
