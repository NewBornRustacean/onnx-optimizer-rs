use crate::passes::error::PassError;

/// Core optimization trait that all optimization levels must implement
pub trait OptimizationPass {
    /// Name of the optimization pass for logging/debugging
    fn pass_name(&self) -> String;

    /// Execute the optimization pass and return the number of changes made
    fn execute(&mut self) -> Result<u32, PassError>;

    /// Check if this pass can be applied to the current graph state
    fn can_apply(&self) -> bool;

    /// Priority of this pass (lower = runs earlier)
    fn priority(&self) -> u32;
}

/// Level 1: Basic optimizations that are safe and always beneficial
pub trait BasicOptimization {
    /// Fold constants throughout the graph
    ///
    /// Identifies nodes where all inputs are constants and the operation
    /// can be computed at compile time, then replaces these nodes with
    /// constant nodes containing the computed result.
    fn constant_folding(&mut self) -> Result<u32, PassError>;

    /// Remove nodes that don't affect the graph output
    ///
    /// Performs reachability analysis from graph outputs and removes
    /// any nodes that cannot be reached.
    fn dead_node_elimination(&mut self) -> Result<u32, PassError>;

    /// Remove identity nodes that don't transform their input
    ///
    /// Identifies Identity nodes and rewires the graph to bypass them.
    fn identity_elimination(&mut self) -> Result<u32, PassError>;

    /// Remove dropout nodes in inference mode
    ///
    /// Dropout nodes are no-ops during inference, so they can be safely removed.
    fn dropout_elimination(&mut self) -> Result<u32, PassError>;
}
