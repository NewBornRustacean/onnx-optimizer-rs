use crate::graph::traits::OptimizationPass;
use crate::utils::error::OnnxOptError;
use crate::register_pass;

/// Dead node elimination pass
pub struct DeadNodeEliminationPass;

impl DeadNodeEliminationPass {
    pub fn new() -> Self {
        Self
    }
}

impl OptimizationPass for DeadNodeEliminationPass {
    fn pass_name(&self) -> &'static str {
        "dead_node_elimination"
    }

    fn execute(&mut self) -> Result<u32, OnnxOptError> {
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

// Register the pass in the global registry
register_pass!("dead_node_elimination", DeadNodeEliminationPass);
