use crate::passes::{OptimizationPass, error::PassError};

/// Constant folding optimization pass
pub struct ConstantFoldingPass;

impl ConstantFoldingPass {
    pub fn new() -> Self {
        Self
    }
}

impl OptimizationPass for ConstantFoldingPass {
    fn pass_name(&self) -> String {
        "constant_folding".to_string()
    }

    fn execute(&mut self) -> Result<u32, PassError> {
        // This would be implemented to work with the graph
        todo!("Execute constant folding pass")
    }

    fn can_apply(&self) -> bool {
        // Check if graph has nodes that can be constant folded
        todo!("Check if constant folding is applicable")
    }

    fn priority(&self) -> u32 {
        1 // the lower the number the higher the priority
    }
}

impl Default for ConstantFoldingPass {
    fn default() -> Self {
        Self::new()
    }
}
