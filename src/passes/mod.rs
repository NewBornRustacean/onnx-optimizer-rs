pub mod error;
pub mod traits;
pub mod manager;
pub mod algorithms;

// Re-export commonly used types
pub use error::PassError;
pub use traits::OptimizationPass;
pub use manager::{PassManager};
