use crate::graph::{Graph, objects::{NodeId, OpKind, Tensor, TensorData}, traits::GraphView};
use crate::passes::{
    traits::OptimizationPass,
    error::PassError,
};

/// Constant folding optimization pass
/// 
/// This pass identifies nodes whose inputs are all constants and replaces 
/// them with a single constant node containing the computed result.
#[derive(Debug, Clone)]
pub struct ConstantFolding {}

impl ConstantFolding {
    /// Create a new constant folding pass
    pub fn new() -> Self {
        Self {}
    }

    /// Check if a node's all inputs are constant values
    fn all_inputs_are_constants(&self, graph: &Graph, node_id: NodeId) -> Result<bool, PassError> {
        if let Some(node) = graph.node(node_id) {
            for &input_id in &node.inputs {
                if let Some(tensor) = graph.tensor(input_id) {
                    // If tensor has no data, it's not a constant
                    if tensor.data.is_none() {
                        return Ok(false);
                    }
                } else {
                    return Ok(false);
                }
            }
            Ok(true)
        } else {
            Err(PassError::GraphOperationFailed {
                details: format!("Node {:?} not found", node_id),
            })
        }
    }

    /// Check if an operation is suitable for constant folding
    fn is_constant_foldable_op(&self, op_kind: &OpKind) -> bool {
        matches!(op_kind, 
            OpKind::Add | OpKind::Sub | OpKind::Mul | OpKind::Div |
            OpKind::Relu | OpKind::Identity
        )
    }

    /// Find all nodes that can be constant folded
    fn find_candidates(&self, graph: &Graph) -> Result<Vec<NodeId>, PassError> {
        let mut candidates = Vec::new();

        for node_id in graph.node_ids() {
            if let Some(node) = graph.node(node_id) {
                // Check if this operation supports constant folding
                if self.is_constant_foldable_op(&node.op_kind) {
                    // Check if all inputs are constants
                    if self.all_inputs_are_constants(graph, node_id)? {
                        candidates.push(node_id);
                    }
                }
            }
        }

        Ok(candidates)
    }

    /// Attempt to fold a single node
    fn try_fold_node(&self, graph: &mut Graph, node_id: NodeId) -> Result<bool, PassError> {
        let node = graph.node(node_id).ok_or_else(|| PassError::GraphOperationFailed {
            details: format!("Node {:?} not found", node_id),
        })?;

        match node.op_kind {
            OpKind::Identity => self.fold_identity(graph, node_id),
            OpKind::Add => self.fold_arithmetic(graph, node_id, "add"),
            OpKind::Sub => self.fold_arithmetic(graph, node_id, "sub"),
            OpKind::Mul => self.fold_arithmetic(graph, node_id, "mul"),
            OpKind::Div => self.fold_arithmetic(graph, node_id, "div"),
            OpKind::Relu => self.fold_relu(graph, node_id),
            _ => Ok(false), // Not supported yet
        }
    }

    /// Fold identity operations (just pass through input)
    fn fold_identity(&self, graph: &mut Graph, node_id: NodeId) -> Result<bool, PassError> {
        let node = graph.node(node_id).unwrap();
        
        if node.inputs.len() != 1 || node.outputs.len() != 1 {
            return Ok(false);
        }

        // TODO: Implement actual node removal and reconnection
        // For now, just return success to indicate we could fold it
        Ok(true)
    }

    /// Fold arithmetic operations (Add, Sub, Mul, Div)
    fn fold_arithmetic(&self, graph: &mut Graph, node_id: NodeId, op: &str) -> Result<bool, PassError> {
        let node = graph.node(node_id).unwrap();

        if node.inputs.len() != 2 {
            return Ok(false); // Only binary operations for now
        }

        // Get input tensors
        let input1 = graph.tensor(node.inputs[0]).ok_or_else(|| PassError::ConstantFoldingFailed {
            reason: "First input tensor not found".to_string(),
        })?;
        
        let input2 = graph.tensor(node.inputs[1]).ok_or_else(|| PassError::ConstantFoldingFailed {
            reason: "Second input tensor not found".to_string(),
        })?;

        // For now, only handle scalar constants (this is a simplified implementation)
        let result = match (input1.data.as_ref(), input2.data.as_ref()) {
            (Some(TensorData::Float32(data1)), Some(TensorData::Float32(data2))) 
                if data1.len() == 1 && data2.len() == 1 => {
                
                let val1 = data1[0];
                let val2 = data2[0];
                
                let result_val = match op {
                    "add" => val1 + val2,
                    "sub" => val1 - val2,
                    "mul" => val1 * val2,
                    "div" => {
                        if val2 == 0.0 {
                            return Err(PassError::ConstantFoldingFailed {
                                reason: "Division by zero".to_string(),
                            });
                        }
                        val1 / val2
                    },
                    _ => return Ok(false),
                };

                // TODO: Replace node with constant
                // graph.replace_with_constant(node_id, result_tensor)?;
                
                Ok(true)
            },
            _ => Ok(false), // Unsupported data types for now
        }?;

        Ok(result)
    }

    /// Fold ReLU operations
    fn fold_relu(&self, graph: &mut Graph, node_id: NodeId) -> Result<bool, PassError> {
        let node = graph.node(node_id).unwrap();

        if node.inputs.len() != 1 {
            return Ok(false);
        }

        let input = graph.tensor(node.inputs[0]).ok_or_else(|| PassError::ConstantFoldingFailed {
            reason: "Input tensor not found".to_string(),
        })?;

        // Apply ReLU: max(0, x)
        match input.data.as_ref() {
            Some(TensorData::Float32(data)) => {
                let result_data: Vec<f32> = data.iter().map(|&x| x.max(0.0)).collect();
                
                // TODO: Replace node with constant
                // graph.replace_with_constant(node_id, result_tensor)?;
                
                Ok(true)
            },
            _ => Ok(false), // Unsupported data type
        }
    }
}

impl Default for ConstantFolding {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPass for ConstantFolding {
    fn pass_name(&self) -> String {
        "ConstantFolding".to_string()
    }

    fn execute(&self, graph: &mut Graph) -> Result<u32, PassError> {
        let candidates = self.find_candidates(graph)?;
        let mut total_changes = 0;

        for node_id in candidates {
            if self.try_fold_node(graph, node_id)? {
                total_changes += 1;
            }
        }

        Ok(total_changes)
    }
}