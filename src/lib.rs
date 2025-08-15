pub mod graph;
pub mod utils;

// Re-export the proto crate for convenience
pub use onnx_proto as proto;

// Re-export common types for convenience
pub use graph::objects::{Graph, Node, Tensor, NodeId, ValueId, OpKind, DataType};
pub use graph::traits::{GraphView, GraphEdit};
pub use utils::io::{load_model, save_model};
pub use utils::error::OnnxOptError;
