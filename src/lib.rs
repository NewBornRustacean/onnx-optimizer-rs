pub mod executor;
pub mod graph;
pub mod passes;
pub mod utils;

pub use executor::OptimizationExecutor;
pub use graph::{
    DataType, Graph, GraphView, Node, NodeAttrValue, NodeId, OpKind, Tensor, ValueId,
    error::GraphError,
};
pub use passes::{error::PassError, traits::{OptimizationPass, PassCategory}};
pub use utils::{
    config::{OptimizationConfig, OptimizationConfigBuilder},
    error::OnnxOptError,
    io::{load_model, save_model},
};

// Re-export common types for convenience
pub use onnx_proto::*;
