use thiserror::Error;

#[derive(Error, Debug, Clone)]
pub enum PassError {
    #[error("Pass not applicable: {0}")]
    PassNotApplicable(String),

    #[error("Not implemented: {0}")]
    NotImplemented(String),
}
