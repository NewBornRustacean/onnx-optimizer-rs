use crate::{graph::Graph, passes::error::PassError};

/// Categories of optimization passes for statistics tracking
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PassCategory {
    /// Constant folding optimizations
    ConstantFolding,
    /// Dead code elimination
    DeadCodeElimination,
    /// Identity operations removal
    IdentityElimination,
    /// Dropout removal for inference
    DropoutElimination,
    /// Operator fusion optimizations
    OperatorFusion,
    /// Other/custom optimizations
    Other,
}

/// Core optimization trait that all optimization levels must implement
pub trait OptimizationPass {
    /// Name of the optimization pass for logging/debugging
    fn pass_name(&self) -> String;

    /// Category of this optimization pass for statistics tracking
    fn category(&self) -> PassCategory;

    /// Execute the optimization pass on the given graph
    /// Returns the number of changes made (0 = no changes, used for fixed-point iteration)
    fn execute(&self, graph: &mut Graph) -> Result<u32, PassError>;
}
