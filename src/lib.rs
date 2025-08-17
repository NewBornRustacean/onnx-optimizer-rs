pub mod graph;
pub mod passes;
pub mod utils;

// Re-export common types for convenience
pub use graph::objects::{DataType, Graph, Node, NodeId, OpKind, Tensor, ValueId};
pub use graph::traits::{GraphEdit, GraphView};
pub use passes::{PassManager, PassManagerBuilder};
pub use utils::error::OnnxOptError;
pub use utils::io::{load_model, save_model};
pub use onnx_proto::*;