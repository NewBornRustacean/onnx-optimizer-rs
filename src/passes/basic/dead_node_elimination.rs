use crate::passes::{OptimizationPass, error::PassError};

/// Dead node elimination pass
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
        todo!("Execute dead node elimination pass")
    }

    fn can_apply(&self) -> bool {
        todo!("Check if dead node elimination is applicable")
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
