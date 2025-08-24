pub mod graph;
pub mod utils;
pub mod passes;

pub use graph::{
    DataType, Graph, GraphView, Node, NodeAttrValue, NodeId, OpKind, Tensor, ValueId,
    error::GraphError,
};

pub use utils::{
    config::{OptimizationConfig, OptimizationConfigBuilder},
    error::OnnxOptError,
    io::{load_model, save_model},
};

pub use passes::{
    manager::PassManager,
    traits::OptimizationPass,
    algorithms::constant_folding::ConstantFolding,
};

pub use onnx_proto::*;
