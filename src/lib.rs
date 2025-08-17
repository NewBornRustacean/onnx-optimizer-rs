pub mod graph;
pub mod utils;

// Re-export the proto crate for convenience
pub use onnx_proto as proto;

// Re-export common types for convenience
pub use graph::objects::{DataType, Graph, Node, NodeId, OpKind, Tensor, ValueId};
pub use graph::traits::{GraphEdit, GraphView};
pub use utils::error::OnnxOptError;
pub use utils::io::{load_model, save_model};
