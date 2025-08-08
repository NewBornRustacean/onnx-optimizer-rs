mod error;
mod utils;
mod graph;
mod io;

pub mod proto {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

pub use graph::*;
pub use error::OnnxOptError;
pub use io::*;