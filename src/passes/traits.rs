use crate::{graph::Graph, passes::error::PassError};

/// Core optimization trait that all optimization passes must implement
pub trait OptimizationPass {
    /// Name of the optimization pass for logging/debugging
    fn pass_name(&self) -> String;

    /// Execute the optimization pass on the given graph
    /// Returns the number of changes made (0 = no changes, used for fixed-point iteration)
    fn execute(&self, graph: &mut Graph) -> Result<u32, PassError>;
}
