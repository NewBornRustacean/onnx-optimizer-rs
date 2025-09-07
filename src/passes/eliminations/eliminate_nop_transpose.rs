use crate::graph::{
    Graph,
    objects::{NodeAttrValue, NodeId, OpKind, ValueId},
    traits::{GraphEdit, GraphView},
};
use crate::passes::{error::PassError, traits::OptimizationPass};

/// Eliminates no-operation Transpose nodes
///
/// A Transpose node is considered a no-op if:
/// 1. The `perm` attribute is the identity permutation [0, 1, 2, ..., n-1]
/// 2. The `perm` attribute is missing (defaults to identity)
/// 3. The input tensor has rank ≤ 1 (transpose has no effect)
///
/// This pass removes such nodes and reconnects the graph directly.
#[derive(Debug, Clone)]
pub struct EliminateNopTranspose;

impl EliminateNopTranspose {
    pub fn new() -> Self {
        Self
    }

    /// Find all Transpose nodes in the graph
    fn find_transpose_nodes(&self, graph: &Graph) -> Vec<NodeId> {
        graph
            .node_indices()
            .filter(|&node_id| {
                graph.node(node_id).map_or(false, |node| node.op_kind == OpKind::Transpose)
            })
            .collect()
    }

    /// Check if a Transpose node is a no-operation
    ///
    /// Returns true if the transpose operation doesn't change the tensor layout:
    /// - Identity permutation: perm = [0, 1, 2, ..., n-1]
    /// - Missing perm attribute (defaults to identity)  
    /// - Input tensor rank ≤ 1
    fn is_nop_transpose(&self, graph: &Graph, node_id: NodeId) -> bool {
        let node = match graph.node(node_id) {
            Some(node) => node,
            None => return false,
        };

        // Must be exactly one input and one output
        if node.inputs.len() != 1 || node.outputs.len() != 1 {
            return false;
        }

        // Get input tensor to check its rank
        let input_tensor = match graph.tensor(node.inputs[0]) {
            Some(tensor) => tensor,
            None => return false,
        };

        // If input tensor rank ≤ 1, transpose is always no-op
        if let Some(shape) = &input_tensor.shape {
            if shape.len() <= 1 {
                return true;
            }
        }

        // Check perm attribute
        self.is_identity_permutation(node, input_tensor.shape.as_ref())
    }

    /// Check if the permutation attribute represents an identity transformation
    ///
    /// Returns true if:
    /// - No perm attribute (defaults to identity)
    /// - perm = [0, 1, 2, ..., n-1] where n is the tensor rank
    fn is_identity_permutation(
        &self,
        node: &crate::graph::objects::Node,
        shape: Option<&Vec<i64>>,
    ) -> bool {
        match node.attributes.get("perm") {
            Some(NodeAttrValue::Ints(perm_values)) => {
                self.is_identity_perm_array(perm_values, shape)
            }
            None => {
                // Missing perm attribute defaults to identity
                true
            }
            _ => {
                // Invalid perm attribute type
                false
            }
        }
    }

    /// Check if a permutation array represents identity transformation
    ///
    /// For a tensor with rank n, identity permutation is [0, 1, 2, ..., n-1]
    fn is_identity_perm_array(&self, perm: &[i64], shape: Option<&Vec<i64>>) -> bool {
        // If we don't know the shape, we can't validate the permutation
        let rank = match shape {
            Some(s) => s.len(),
            None => return false, // Conservative: can't determine if it's identity
        };

        // Check if perm length matches tensor rank
        if perm.len() != rank {
            return false;
        }

        // Check if perm is [0, 1, 2, ..., rank-1]
        perm.iter().enumerate().all(|(i, &p)| p == i as i64)
    }

    /// Check if a Transpose node can be safely eliminated
    ///
    /// Additional safety checks beyond is_nop_transpose:
    /// - Input and output tensors exist
    /// - No other constraints that prevent elimination
    fn can_eliminate_transpose(&self, graph: &Graph, node_id: NodeId) -> bool {
        let node = match graph.node(node_id) {
            Some(node) => node,
            None => return false,
        };

        // Verify input tensor exists
        if graph.tensor(node.inputs[0]).is_none() {
            return false;
        }

        // Verify output tensor exists
        if graph.tensor(node.outputs[0]).is_none() {
            return false;
        }

        // Additional safety checks can be added here
        // (e.g., check for special graph constraints)

        true
    }

    /// Eliminate a single no-op Transpose node
    ///
    /// Process:
    /// 1. Verify the node can be eliminated
    /// 2. Replace all uses of output with input
    /// 3. Remove the node and unused output tensor
    fn eliminate_nop_transpose_node(
        &self,
        graph: &mut Graph,
        node_id: NodeId,
    ) -> Result<bool, PassError> {
        // Verify elimination is safe
        if !self.can_eliminate_transpose(graph, node_id) {
            return Ok(false);
        }

        // Additional check: must be a no-op
        if !self.is_nop_transpose(graph, node_id) {
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

impl Default for EliminateNopTranspose {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPass for EliminateNopTranspose {
    fn pass_name(&self) -> String {
        "EliminateNopTranspose".to_string()
    }

    fn execute(&self, graph: &mut Graph) -> Result<u32, PassError> {
        let transpose_nodes = self.find_transpose_nodes(graph);
        let mut total_eliminated = 0;

        for node_id in transpose_nodes {
            if self.eliminate_nop_transpose_node(graph, node_id)? {
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
        objects::{DataType, Node, NodeAttrValue, OpKind, Tensor, ValueId},
    };

    fn create_test_tensor(name: &str) -> Tensor {
        Tensor {
            name: Some(name.to_string()),
            shape: Some(vec![2, 3, 4]), // 3D tensor for testing
            dtype: DataType::Float32,
            data: None,
        }
    }

    fn create_test_graph_with_nop_transpose() -> (Graph, NodeId, ValueId, ValueId) {
        let mut graph = Graph::new();

        // Create input and output tensors
        let input_tensor = create_test_tensor("input");
        let output_tensor = create_test_tensor("output");

        let input_id = graph.add_value(input_tensor);
        let output_id = graph.add_value(output_tensor);

        // Create identity transpose node (perm = [0, 1, 2])
        let mut transpose_node = Node::new(OpKind::Transpose)
            .with_inputs(vec![input_id])
            .with_outputs(vec![output_id]);

        // Add identity permutation attribute
        transpose_node
            .attributes
            .insert("perm".to_string(), NodeAttrValue::Ints(vec![0, 1, 2]));

        let node_id = graph.add_node(transpose_node);

        (graph, node_id, input_id, output_id)
    }

    #[test]
    fn test_find_transpose_nodes() {
        let (graph, node_id, _, _) = create_test_graph_with_nop_transpose();
        let pass = EliminateNopTranspose::new();

        let transpose_nodes = pass.find_transpose_nodes(&graph);
        assert_eq!(transpose_nodes.len(), 1);
        assert_eq!(transpose_nodes[0], node_id);
    }

    #[test]
    fn test_is_identity_perm_array() {
        let pass = EliminateNopTranspose::new();
        let shape = vec![2, 3, 4];

        // Identity permutation
        assert!(pass.is_identity_perm_array(&[0, 1, 2], Some(&shape)));

        // Non-identity permutation
        assert!(!pass.is_identity_perm_array(&[1, 0, 2], Some(&shape)));
        assert!(!pass.is_identity_perm_array(&[2, 1, 0], Some(&shape)));

        // Wrong length
        assert!(!pass.is_identity_perm_array(&[0, 1], Some(&shape)));
        assert!(!pass.is_identity_perm_array(&[0, 1, 2, 3], Some(&shape)));
    }

    #[test]
    fn test_is_nop_transpose_identity_perm() {
        let (graph, node_id, _, _) = create_test_graph_with_nop_transpose();
        let pass = EliminateNopTranspose::new();

        assert!(pass.is_nop_transpose(&graph, node_id));
    }

    #[test]
    fn test_is_nop_transpose_missing_perm() {
        let mut graph = Graph::new();

        let input_tensor = create_test_tensor("input");
        let output_tensor = create_test_tensor("output");

        let input_id = graph.add_value(input_tensor);
        let output_id = graph.add_value(output_tensor);

        // Transpose node without perm attribute (defaults to identity)
        let transpose_node = Node::new(OpKind::Transpose)
            .with_inputs(vec![input_id])
            .with_outputs(vec![output_id]);

        let node_id = graph.add_node(transpose_node);
        let pass = EliminateNopTranspose::new();

        assert!(pass.is_nop_transpose(&graph, node_id));
    }

    #[test]
    fn test_is_nop_transpose_low_rank_tensor() {
        let mut graph = Graph::new();

        // Create 1D tensor
        let input_tensor = Tensor {
            name: Some("input_1d".to_string()),
            shape: Some(vec![5]), // 1D tensor
            dtype: DataType::Float32,
            data: None,
        };
        let output_tensor = Tensor {
            name: Some("output_1d".to_string()),
            shape: Some(vec![5]),
            dtype: DataType::Float32,
            data: None,
        };

        let input_id = graph.add_value(input_tensor);
        let output_id = graph.add_value(output_tensor);

        let transpose_node = Node::new(OpKind::Transpose)
            .with_inputs(vec![input_id])
            .with_outputs(vec![output_id]);

        let node_id = graph.add_node(transpose_node);
        let pass = EliminateNopTranspose::new();

        // 1D tensor transpose is always no-op
        assert!(pass.is_nop_transpose(&graph, node_id));
    }

    #[test]
    fn test_execute_pass() {
        let (mut graph, _, _, _) = create_test_graph_with_nop_transpose();
        let pass = EliminateNopTranspose::new();

        let result = pass.execute(&mut graph);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 1); // Should eliminate 1 transpose node
    }

    #[test]
    fn test_empty_graph() {
        let mut graph = Graph::new();
        let pass = EliminateNopTranspose::new();

        let result = pass.execute(&mut graph);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0); // No transpose nodes to eliminate
    }
}
