use thiserror::Error;

#[derive(Error, Debug, Clone)]
pub enum PassError {
    #[error("Pass not applicable: {0}")]
    PassNotApplicable(String),

    #[error("Not implemented: {0}")]
    NotImplemented(String),

    #[error("Graph operation failed: {details}")]
    GraphOperationFailed { details: String },

    #[error("Pass execution failed: {message}")]
    ExecutionFailed { message: String },

    #[error("Constant folding failed: {reason}")]
    ConstantFoldingFailed { reason: String },

    #[error("Dead node elimination failed: {reason}")]
    DeadNodeEliminationFailed { reason: String },
}
