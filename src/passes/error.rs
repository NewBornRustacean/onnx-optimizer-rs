use thiserror::Error;

#[derive(Error, Debug)]
pub enum PassError {
    #[error("Pass not applicable: {0}")]
    PassNotApplicable(String),
}
