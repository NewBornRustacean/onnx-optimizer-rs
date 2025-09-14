use crate::{
    graph::{
        Graph,
        objects::{NodeId, OpKind, ValueId},
        traits::{GraphEdit, GraphView},
    },
    passes::{error::PassError, traits::OptimizationPass},
};

/// remove cast operations whose input and output are equal.
/// e.g. f32 -> cast -> f32
#[derive(Debug, Clone)]
pub struct EliminateNopConcat;

impl EliminateNopConcat {
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

impl Default for EliminateNopConcat {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPass for EliminateNopConcat {
    fn pass_name(&self) -> String {
        "EliminateNopConcat".to_string()
    }

    fn execute(&self, graph: &mut Graph) -> Result<u32, PassError> {
        let concat_nodes = self.find_concat_nodes(graph);
        let mut total_eliminated = 0;

        for node_id in concat_nodes {
            if self.eliminate_concat_node(graph, node_id)? {
                total_eliminated += 1;
            }
        }

        Ok(total_eliminated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{
        Graph,
        objects::{DataType, Node, OpKind, Tensor, ValueId},
        traits::GraphEdit,
    };

    fn create_test_tensor(name: &str) -> Tensor {
        Tensor {
            name: Some(name.to_string()),
            shape: Some(vec![2, 3]), // 2D tensor for testing
            dtype: DataType::Float32,
            data: None,
        }
    }

    fn create_test_graph_with_nop_concat() -> (Graph, NodeId, ValueId, ValueId) {
        let mut graph = Graph::new();

        // Create input and output tensors
        let input_tensor = create_test_tensor("input");
        let output_tensor = create_test_tensor("output");

        let input_id = graph.add_value(input_tensor);
        let output_id = graph.add_value(output_tensor);

        // Create nop concat node (only 1 input)
        let concat_node = Node::new(OpKind::Concat)
            .with_inputs(vec![input_id])
            .with_outputs(vec![output_id]);

        let node_id = graph.add_node(concat_node);

        (graph, node_id, input_id, output_id)
    }

    #[test]
    fn test_find_concat_nodes() {
        let (graph, node_id, _, _) = create_test_graph_with_nop_concat();
        let pass = EliminateNopConcat::new();

        let concat_nodes = pass.find_concat_nodes(&graph);
        assert_eq!(concat_nodes.len(), 1);
        assert_eq!(concat_nodes[0], node_id);
    }

    #[test]
    fn test_can_eliminate_concat() {
        let (graph, node_id, _, _) = create_test_graph_with_nop_concat();
        let pass = EliminateNopConcat::new();

        assert!(pass.can_eliminate_concat(&graph, node_id));
    }

    #[test]
    fn test_cannot_eliminate_concat_multiple_inputs() {
        let mut graph = Graph::new();

        // Create input and output tensors
        let input1_tensor = create_test_tensor("input1");
        let input2_tensor = create_test_tensor("input2");
        let output_tensor = create_test_tensor("output");

        let input1_id = graph.add_value(input1_tensor);
        let input2_id = graph.add_value(input2_tensor);
        let output_id = graph.add_value(output_tensor);

        // Create concat node with multiple inputs (not a nop)
        let concat_node = Node::new(OpKind::Concat)
            .with_inputs(vec![input1_id, input2_id])
            .with_outputs(vec![output_id]);

        let node_id = graph.add_node(concat_node);
        let pass = EliminateNopConcat::new();

        assert!(!pass.can_eliminate_concat(&graph, node_id));
    }

    #[test]
    fn test_cannot_eliminate_concat_multiple_outputs() {
        let mut graph = Graph::new();

        // Create input and output tensors
        let input_tensor = create_test_tensor("input");
        let output1_tensor = create_test_tensor("output1");
        let output2_tensor = create_test_tensor("output2");

        let input_id = graph.add_value(input_tensor);
        let output1_id = graph.add_value(output1_tensor);
        let output2_id = graph.add_value(output2_tensor);

        // Create concat node with multiple outputs (invalid)
        let concat_node = Node::new(OpKind::Concat)
            .with_inputs(vec![input_id])
            .with_outputs(vec![output1_id, output2_id]);

        let node_id = graph.add_node(concat_node);
        let pass = EliminateNopConcat::new();

        assert!(!pass.can_eliminate_concat(&graph, node_id));
    }

    #[test]
    fn test_eliminate_concat_node() {
        let (mut graph, node_id, _, _) = create_test_graph_with_nop_concat();
        let pass = EliminateNopConcat::new();

        let result = pass.eliminate_concat_node(&mut graph, node_id);
        assert!(result.is_ok());
        assert!(result.unwrap()); // Should return true indicating elimination was possible
    }

    #[test]
    fn test_execute_pass() {
        let (mut graph, _, _, _) = create_test_graph_with_nop_concat();
        let pass = EliminateNopConcat::new();

        let result = pass.execute(&mut graph);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 1); // Should eliminate 1 concat node
    }

    #[test]
    fn test_concat_elimination_with_consumers() {
        let mut graph = Graph::new();

        // Create tensors
        let input_tensor = create_test_tensor("input");
        let concat_output_tensor = create_test_tensor("concat_output");
        let final_output_tensor = create_test_tensor("final_output");

        let input_id = graph.add_value(input_tensor);
        let concat_output_id = graph.add_value(concat_output_tensor);
        let final_output_id = graph.add_value(final_output_tensor);

        // Create nop Concat node: input -> concat_output
        let concat_node = Node::new(OpKind::Concat)
            .with_inputs(vec![input_id])
            .with_outputs(vec![concat_output_id]);
        let concat_node_id = graph.add_node(concat_node);

        // Create consumer node: concat_output -> final_output
        let consumer_node = Node::new(OpKind::Relu)
            .with_inputs(vec![concat_output_id])
            .with_outputs(vec![final_output_id]);
        let consumer_node_id = graph.add_node(consumer_node);

        // Verify initial state
        assert_eq!(
            graph.node(consumer_node_id).unwrap().inputs[0],
            concat_output_id
        );

        // Execute concat elimination
        let pass = EliminateNopConcat::new();
        let result = pass.execute(&mut graph);

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 1);

        // Verify the consumer now uses the input directly
        assert_eq!(graph.node(consumer_node_id).unwrap().inputs[0], input_id);

        // Verify the concat node is gone
        assert!(graph.node(concat_node_id).is_none());

        // Verify the concat output tensor is gone
        assert!(graph.tensor(concat_output_id).is_none());
    }

    #[test]
    fn test_empty_graph() {
        let mut graph = Graph::new();
        let pass = EliminateNopConcat::new();

        let result = pass.execute(&mut graph);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0); // No concat nodes to eliminate
    }

    #[test]
    fn test_multiple_concat_nodes() {
        let mut graph = Graph::new();

        // Create first nop concat
        let input1_tensor = create_test_tensor("input1");
        let output1_tensor = create_test_tensor("output1");
        let input1_id = graph.add_value(input1_tensor);
        let output1_id = graph.add_value(output1_tensor);

        let concat1_node = Node::new(OpKind::Concat)
            .with_inputs(vec![input1_id])
            .with_outputs(vec![output1_id]);
        graph.add_node(concat1_node);

        // Create second nop concat
        let input2_tensor = create_test_tensor("input2");
        let output2_tensor = create_test_tensor("output2");
        let input2_id = graph.add_value(input2_tensor);
        let output2_id = graph.add_value(output2_tensor);

        let concat2_node = Node::new(OpKind::Concat)
            .with_inputs(vec![input2_id])
            .with_outputs(vec![output2_id]);
        graph.add_node(concat2_node);

        let pass = EliminateNopConcat::new();
        let result = pass.execute(&mut graph);

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 2); // Should eliminate both concat nodes
    }

    #[test]
    fn test_mixed_concat_nodes() {
        let mut graph = Graph::new();

        // Create nop concat (1 input)
        let input1_tensor = create_test_tensor("input1");
        let output1_tensor = create_test_tensor("output1");
        let input1_id = graph.add_value(input1_tensor);
        let output1_id = graph.add_value(output1_tensor);

        let nop_concat_node = Node::new(OpKind::Concat)
            .with_inputs(vec![input1_id])
            .with_outputs(vec![output1_id]);
        graph.add_node(nop_concat_node);

        // Create normal concat (2 inputs)
        let input2_tensor = create_test_tensor("input2");
        let input3_tensor = create_test_tensor("input3");
        let output2_tensor = create_test_tensor("output2");
        let input2_id = graph.add_value(input2_tensor);
        let input3_id = graph.add_value(input3_tensor);
        let output2_id = graph.add_value(output2_tensor);

        let normal_concat_node = Node::new(OpKind::Concat)
            .with_inputs(vec![input2_id, input3_id])
            .with_outputs(vec![output2_id]);
        graph.add_node(normal_concat_node);

        let pass = EliminateNopConcat::new();
        let result = pass.execute(&mut graph);

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 1); // Should eliminate only the nop concat
    }
}
