use crate::passes::{OptimizationPass, error::PassError};

/// Dead node elimination pass
#[derive(Debug, Clone)]
pub struct DeadNodeEliminationPass;

impl DeadNodeEliminationPass {
    pub fn new() -> Self {
        Self
    }
}

impl OptimizationPass for DeadNodeEliminationPass {
    fn pass_name(&self) -> String {
        "dead_node_elimination".to_string()
    }

    fn execute(&mut self) -> Result<u32, PassError> {
        Err(PassError::NotImplemented(
            "Execute dead node elimination pass".to_string(),
        ))
    }

    fn can_apply(&self) -> bool {
        // For now, return true to allow testing the execution flow
        true
    }

    fn priority(&self) -> u32 {
        2 // the lower the number the higher the priority
    }
}

impl Default for DeadNodeEliminationPass {
    fn default() -> Self {
        Self::new()
    }
}
