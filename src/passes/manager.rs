use crate::{
    graph::Graph,
    passes::{algorithms::EliminateIdentity, error::PassError, traits::OptimizationPass},
};

/// Individual optimization pass implementations
#[derive(Debug)]
pub enum Pass {
    EliminateIdentity(EliminateIdentity),

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
            Pass::Placeholder => "Placeholder".to_string(),
        }
    }

    fn execute(&self, graph: &mut Graph) -> Result<u32, PassError> {
        match self {
            Pass::EliminateIdentity(p) => p.execute(graph),
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
        let manager =
            PassManager::new().add_pass(Pass::EliminateIdentity(EliminateIdentity::new()));

        assert_eq!(manager.passes.len(), 1);
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
}
