pub mod executor;
pub mod graph;
pub mod passes;
pub mod utils;

pub use executor::composer::{ComposerConfig, OptimizationComposer, OptimizationStatistics};
pub use graph::{
    DataType, Graph, GraphView, Node, NodeAttrValue, NodeId, OpKind, Tensor, ValueId,
    error::GraphError,
};
pub use onnx_proto::*;
pub use passes::{
    eliminations::*,
    manager::{Pass, PassManager},
    traits::OptimizationPass,
};
pub use utils::{
    config::{OptimizationConfig, OptimizationConfigBuilder},
    error::OnnxOptError,
    io::{load_model, save_model},
};
