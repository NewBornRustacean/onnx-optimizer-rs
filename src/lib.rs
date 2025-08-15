mod graph;
mod utils;

pub mod proto {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

pub use graph::*;
pub use utils::*;
