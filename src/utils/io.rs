use crate::utils::error::OnnxOptError;
use onnx_proto;
use prost::Message;
use std::{
    fs::File,
    io::{Read, Write},
    path::Path,
};

/// *.onnx → ModelProto
pub fn load_model<P: AsRef<Path>>(path: P) -> Result<onnx_proto::ModelProto, OnnxOptError> {
    let mut buf = Vec::new();
    File::open(path)?.read_to_end(&mut buf)?;
    Ok(onnx_proto::ModelProto::decode(&*buf)?)
}

/// ModelProto → *.onnx
pub fn save_model<P: AsRef<Path>>(
    model: &onnx_proto::ModelProto,
    path: P,
) -> Result<(), OnnxOptError> {
    let mut buf = Vec::new();
    model.encode(&mut buf)?;
    File::create(path)?.write_all(&buf)?;
    Ok(())
}

/// Trait for types that can be constructed from little-endian bytes
pub trait FromLeBytes: Sized {
    fn from_le_bytes(bytes: &[u8]) -> Result<Self, OnnxOptError>;
}

impl FromLeBytes for f32 {
    fn from_le_bytes(bytes: &[u8]) -> Result<Self, OnnxOptError> {
        if bytes.len() != 4 {
            return Err(OnnxOptError::Conversion(
                "Invalid byte length for f32".to_string(),
            ));
        }
        Ok(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }
}

impl FromLeBytes for f64 {
    fn from_le_bytes(bytes: &[u8]) -> Result<Self, OnnxOptError> {
        if bytes.len() != 8 {
            return Err(OnnxOptError::Conversion(
                "Invalid byte length for f64".to_string(),
            ));
        }
        let array: [u8; 8] = bytes.try_into().map_err(|_| {
            OnnxOptError::Conversion("Failed to convert bytes to array for f64".to_string())
        })?;
        Ok(f64::from_le_bytes(array))
    }
}

impl FromLeBytes for i32 {
    fn from_le_bytes(bytes: &[u8]) -> Result<Self, OnnxOptError> {
        if bytes.len() != 4 {
            return Err(OnnxOptError::Conversion(
                "Invalid byte length for i32".to_string(),
            ));
        }
        Ok(i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }
}

impl FromLeBytes for i64 {
    fn from_le_bytes(bytes: &[u8]) -> Result<Self, OnnxOptError> {
        if bytes.len() != 8 {
            return Err(OnnxOptError::Conversion(
                "Invalid byte length for i64".to_string(),
            ));
        }
        let array: [u8; 8] = bytes.try_into().map_err(|_| {
            OnnxOptError::Conversion("Failed to convert bytes to array for i64".to_string())
        })?;
        Ok(i64::from_le_bytes(array))
    }
}

impl FromLeBytes for u32 {
    fn from_le_bytes(bytes: &[u8]) -> Result<Self, OnnxOptError> {
        if bytes.len() != 4 {
            return Err(OnnxOptError::Conversion(
                "Invalid byte length for u32".to_string(),
            ));
        }
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }
}

impl FromLeBytes for u64 {
    fn from_le_bytes(bytes: &[u8]) -> Result<Self, OnnxOptError> {
        if bytes.len() != 8 {
            return Err(OnnxOptError::Conversion(
                "Invalid byte length for u64".to_string(),
            ));
        }
        let array: [u8; 8] = bytes.try_into().map_err(|_| {
            OnnxOptError::Conversion("Failed to convert bytes to array for u64".to_string())
        })?;
        Ok(u64::from_le_bytes(array))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::NamedTempFile;

    /// Creates a minimal ONNX model with a single Identity operator
    fn create_minimal_model() -> onnx_proto::ModelProto {
        // Input ValueInfo
        let input = onnx_proto::ValueInfoProto {
            name: Some("input".to_string()),
            r#type: Some(onnx_proto::TypeProto {
                value: Some(onnx_proto::type_proto::Value::TensorType(
                    onnx_proto::type_proto::Tensor {
                        elem_type: Some(1), // FLOAT
                        shape: Some(onnx_proto::TensorShapeProto {
                            dim: vec![onnx_proto::tensor_shape_proto::Dimension {
                                value: Some(
                                    onnx_proto::tensor_shape_proto::dimension::Value::DimValue(1),
                                ),
                                ..Default::default()
                            }],
                            ..Default::default()
                        }),
                        ..Default::default()
                    },
                )),
                ..Default::default()
            }),
            ..Default::default()
        };

        // Output ValueInfo
        let output = onnx_proto::ValueInfoProto {
            name: Some("output".to_string()),
            r#type: Some(onnx_proto::TypeProto {
                value: Some(onnx_proto::type_proto::Value::TensorType(
                    onnx_proto::type_proto::Tensor {
                        elem_type: Some(1), // FLOAT
                        shape: Some(onnx_proto::TensorShapeProto {
                            dim: vec![onnx_proto::tensor_shape_proto::Dimension {
                                value: Some(
                                    onnx_proto::tensor_shape_proto::dimension::Value::DimValue(1),
                                ),
                                ..Default::default()
                            }],
                            ..Default::default()
                        }),
                        ..Default::default()
                    },
                )),
                ..Default::default()
            }),
            ..Default::default()
        };

        // Identity node
        let node = onnx_proto::NodeProto {
            input: vec!["input".to_string()],
            output: vec!["output".to_string()],
            name: Some("identity_node".to_string()),
            op_type: Some("Identity".to_string()),
            domain: Some(String::new()),
            attribute: vec![],
            ..Default::default()
        };

        // Graph
        let graph = onnx_proto::GraphProto {
            node: vec![node],
            name: Some("minimal_graph".to_string()),
            initializer: vec![],
            sparse_initializer: vec![],
            input: vec![input],
            output: vec![output],
            value_info: vec![],
            ..Default::default()
        };

        // Model
        onnx_proto::ModelProto {
            ir_version: Some(8), // ONNX IR version
            opset_import: vec![onnx_proto::OperatorSetIdProto {
                domain: Some(String::new()),
                version: Some(18), // opset version
                ..Default::default()
            }],
            producer_name: Some("onnx-optimizer-rs".to_string()),
            producer_version: Some("0.1.0".to_string()),
            domain: Some(String::new()),
            model_version: Some(1),
            graph: Some(graph),
            ..Default::default()
        }
    }

    #[test]
    fn test_save_and_load_model() {
        // 1. Create minimal model
        let original_model = create_minimal_model();

        // 2. Save to temporary file
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let temp_path = temp_file.path();

        save_model(&original_model, temp_path).expect("Failed to save model");

        // 3. Load back
        let loaded_model = load_model(temp_path).expect("Failed to load model");

        // 4. Verify
        assert_eq!(original_model.ir_version, loaded_model.ir_version);
        assert_eq!(original_model.producer_name, loaded_model.producer_name);
        assert_eq!(
            original_model.graph.as_ref().unwrap().name,
            loaded_model.graph.as_ref().unwrap().name
        );
    }

    #[test]
    fn test_load_model_nonexistent_file() {
        let result = load_model("nonexistent_file.onnx");
        assert!(result.is_err());

        match result.unwrap_err() {
            OnnxOptError::Io(_) => {} // Expected error
            _ => panic!("Expected IO error"),
        }
    }

    #[test]
    fn test_load_model_invalid_content() {
        // Create file with invalid content
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        fs::write(temp_file.path(), b"invalid protobuf content").expect("Failed to write");

        let result = load_model(temp_file.path());
        assert!(result.is_err());

        match result.unwrap_err() {
            OnnxOptError::Decode(_) => {} // Expected error
            _ => panic!("Expected Decode error"),
        }
    }
}
