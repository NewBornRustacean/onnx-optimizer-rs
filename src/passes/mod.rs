pub mod algorithms;
pub mod error;
pub mod manager;
pub mod traits;

// Re-export commonly used types
pub use error::PassError;
pub use manager::PassManager;
pub use traits::OptimizationPass;
