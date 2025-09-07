use crate::{
    graph::Graph,
    passes::{eliminations::{EliminateIdentity, EliminateNopTranspose}, error::PassError, traits::OptimizationPass},
};

/// Individual optimization pass implementations
#[derive(Debug)]
pub enum Pass {
    EliminateIdentity(EliminateIdentity),
    EliminateNopTranspose(EliminateNopTranspose),

    // TODO: Add more pass implementations
    // DeadNodeElimination(DeadNodeElimination),
    // FuseConvBatchnorm(FuseConvBatchnorm),

    // Placeholder for testing
    Placeholder,
}

impl OptimizationPass for Pass {
    fn pass_name(&self) -> String {
        match self {
            Pass::EliminateIdentity(p) => p.pass_name(),
            Pass::EliminateNopTranspose(p) => p.pass_name(),
            Pass::Placeholder => "Placeholder".to_string(),
        }
    }

    fn execute(&self, graph: &mut Graph) -> Result<u32, PassError> {
        match self {
            Pass::EliminateIdentity(p) => p.execute(graph),
            Pass::EliminateNopTranspose(p) => p.execute(graph),
            Pass::Placeholder => Ok(0),
        }
    }
}

/// Builder for composing and executing optimization passes
#[derive(Debug)]
pub struct PassManager {
    passes: Vec<Pass>,
}

impl PassManager {
    /// Create a new empty pass manager
    pub fn new() -> Self {
        Self { passes: Vec::new() }
    }

    /// Add a pass to the manager (chainable)
    pub fn add_pass(mut self, pass: Pass) -> Self {
        self.passes.push(pass);
        self
    }

    /// Execute all passes on the graph in sequence
    pub fn execute_all(&self, graph: &mut Graph) -> Result<u32, PassError> {
        let mut total_changes = 0;

        for (index, pass) in self.passes.iter().enumerate() {
            let changes = pass.execute(graph).map_err(|e| PassError::ExecutionFailed {
                message: format!(
                    "Pass '{}' (index {}) failed: {}",
                    pass.pass_name(),
                    index,
                    e
                ),
            })?;

            total_changes += changes;

            // Log progress if needed
            if changes > 0 {
                println!("Pass '{}' made {} changes", pass.pass_name(), changes);
            }
        }

        Ok(total_changes)
    }
}

impl Default for PassManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;

    #[test]
    fn test_pass_manager_creation() {
        let manager = PassManager::new();
        assert!(manager.passes.is_empty());
    }

    #[test]
    fn test_pass_manager_chaining() {
        let manager = PassManager::new()
            .add_pass(Pass::EliminateIdentity(EliminateIdentity::new()))
            .add_pass(Pass::EliminateNopTranspose(EliminateNopTranspose::new()));

        assert_eq!(manager.passes.len(), 2);
    }

    #[test]
    fn test_execute_with_passes() {
        let manager =
            PassManager::new().add_pass(Pass::EliminateIdentity(EliminateIdentity::new()));

        let mut graph = Graph::new();
        let result = manager.execute_all(&mut graph);
        assert!(result.is_ok());
        // Empty graph should result in 0 changes, which is expected
        assert_eq!(result.unwrap(), 0);
    }

    #[test]
    fn test_execute_with_identity_elimination() {
        use crate::graph::{objects::{Node, OpKind, Tensor, DataType}, traits::GraphEdit};
        
        let manager =
            PassManager::new().add_pass(Pass::EliminateIdentity(EliminateIdentity::new()));

        let mut graph = Graph::new();
        
        // Create a simple graph with Identity node
        let input_tensor = Tensor::new(DataType::Float32);
        let output_tensor = Tensor::new(DataType::Float32);
        
        let input_id = graph.add_value(input_tensor);
        let output_id = graph.add_value(output_tensor);
        
        let identity_node = Node::new(OpKind::Identity)
            .with_inputs(vec![input_id])
            .with_outputs(vec![output_id]);
        
        graph.add_node(identity_node);
        
        let result = manager.execute_all(&mut graph);
        assert!(result.is_ok());
        // Should eliminate 1 identity node
        assert_eq!(result.unwrap(), 1);
    }
}
