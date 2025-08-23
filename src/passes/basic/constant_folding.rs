use crate::passes::{error::PassError, traits::OptimizationPass};

/// Constant folding optimization pass
#[derive(Debug, Clone)]
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
        // No-op baseline implementation: returns 0 changes.
        // Extend this to perform actual constant folding over the graph.
        Ok(0)
    }

    fn can_apply(&self) -> bool {
        // Check if graph has nodes that can be constant folded
        // For now, return true to allow testing the execution flow
        true
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
