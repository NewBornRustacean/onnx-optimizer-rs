mod utils;
mod graph;

pub mod proto {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

pub use utils::*;
pub use graph::*;