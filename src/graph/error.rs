// Error types specific to executor operations
#[derive(Debug, thiserror::Error)]
pub enum ExecutorError {
    #[error("Graph contains cycles")]
    CyclicGraph,
    #[error("Invalid node operation: {0}")]
    InvalidOperation(String),
    #[error("Constant folding failed: {0}")]
    ConstantFoldingError(String),
    #[error("Graph validation failed: {0}")]
    ValidationError(String),
}
