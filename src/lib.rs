pub mod executor;
pub mod graph;
pub mod passes;
pub mod utils;

pub use executor::OptimizationExecutor;
pub use graph::{Graph, GraphView, NodeId, OpKind, Tensor, NodeAttrValue, ValueId, DataType, Node};
pub use passes::{
    error::PassError,
    traits::BasicOptimization,
};
pub use utils::{
    config::{OptimizationConfig, OptimizationConfigBuilder},
    error::OnnxOptError,
    io::{load_model, save_model},
};

// Re-export common types for convenience
pub use onnx_proto::*;
