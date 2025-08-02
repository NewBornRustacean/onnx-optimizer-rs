use thiserror::Error;

#[derive(Error, Debug)]
pub enum OnnxOptError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Decode error: {0}")]
    Decode(#[from] prost::DecodeError),
    
    #[error("Encode error: {0}")]
    Encode(#[from] prost::EncodeError),
}