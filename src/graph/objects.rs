use crate::graph::traits::{GraphEdit, GraphView};
use crate::utils::arena::{Arena, ArenaId};
use crate::utils::error::OnnxOptError;
use crate::utils::io::FromLeBytes;
use onnx_proto::{ModelProto, GraphProto, TensorProto};
use onnx_proto::tensor_proto::DataType as ProtoType;
use std::collections::HashMap;
use std::str::FromStr;
use strum_macros::{AsRefStr, Display, EnumString};

/// Stable identifier for nodes in the graph
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u32);

/// Stable identifier for values (tensors) in the graph
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueId(pub u32);

/// Represents a tensor value in the computation graph
#[derive(Debug, Clone)]
pub struct Tensor {
    /// Tensor name (optional)
    pub name: Option<String>,
    /// Shape of the tensor (None for unknown shape)
    pub shape: Option<Vec<i64>>,
    /// Data type of the tensor
    pub dtype: DataType,
    /// Optional constant data for initializers
    pub data: Option<TensorData>,
}

/// Supported data types for tensors
#[derive(Debug, Clone, PartialEq)]
pub enum DataType {
    Float32,
    Float64,
    Int32,
    Int64,
    Uint32,
    Uint64,
    Int8,
    Uint8,
    Bool,
    String,
}

impl DataType {
    pub fn from_proto_data_type(
        proto_type: ProtoType,
    ) -> Result<Self, OnnxOptError> {

        match proto_type {
            ProtoType::Float => Ok(DataType::Float32),
            ProtoType::Double => Ok(DataType::Float64),
            ProtoType::Int32 => Ok(DataType::Int32),
            ProtoType::Int64 => Ok(DataType::Int64),
            ProtoType::Uint32 => Ok(DataType::Uint32),
            ProtoType::Uint64 => Ok(DataType::Uint64),
            ProtoType::Int8 => Ok(DataType::Int8),
            ProtoType::Uint8 => Ok(DataType::Uint8),
            ProtoType::Bool => Ok(DataType::Bool),
            ProtoType::String => Ok(DataType::String),
            _ => Err(OnnxOptError::UnsupportedOp(format!(
                "Unsupported data type: {:?}",
                proto_type
            ))),
        }
    }
}

/// Container for tensor constant data
#[derive(Debug, Clone)]
pub enum TensorData {
    Float32(Vec<f32>),
    Float64(Vec<f64>),
    Int32(Vec<i32>),
    Int64(Vec<i64>),
    Uint32(Vec<u32>),
    Uint64(Vec<u64>),
    Int8(Vec<i8>),
    Uint8(Vec<u8>),
    Bool(Vec<bool>),
    String(Vec<String>),
}

impl TensorData {
    pub fn from_proto(tensor_proto: &TensorProto) -> Result<Option<Self>, OnnxOptError> {
        let data_type = ProtoType::try_from(tensor_proto.data_type.unwrap_or(0))
            .map_err(|_| OnnxOptError::Conversion("Invalid data type".to_string()))?;

        // Check if tensor has any data
        let has_data = !tensor_proto.float_data.is_empty()
            || !tensor_proto.double_data.is_empty()
            || !tensor_proto.int32_data.is_empty()
            || !tensor_proto.int64_data.is_empty()
            || !tensor_proto.uint64_data.is_empty()
            || !tensor_proto.string_data.is_empty()
            || tensor_proto.raw_data.as_ref().map_or(false, |data| !data.is_empty());

        if !has_data {
            return Ok(None);
        }

        match data_type {
            ProtoType::Float => Self::extract_numeric_data(
                &tensor_proto.float_data,
                tensor_proto.raw_data.as_ref().map(|v| v.as_slice()).unwrap_or(&[]),
                TensorData::Float32,
            ),
            ProtoType::Double => Self::extract_numeric_data(
                &tensor_proto.double_data,
                tensor_proto.raw_data.as_ref().map(|v| v.as_slice()).unwrap_or(&[]),
                TensorData::Float64,
            ),
            ProtoType::Int32 => Self::extract_numeric_data(
                &tensor_proto.int32_data,
                tensor_proto.raw_data.as_ref().map(|v| v.as_slice()).unwrap_or(&[]),
                TensorData::Int32,
            ),
            ProtoType::Int64 => Self::extract_numeric_data(
                &tensor_proto.int64_data,
                tensor_proto.raw_data.as_ref().map(|v| v.as_slice()).unwrap_or(&[]),
                TensorData::Int64,
            ),
            ProtoType::Uint32 => {
                // Special case: ONNX sometimes stores uint32 in uint64_data
                if !tensor_proto.uint64_data.is_empty() {
                    let uints: Result<Vec<u32>, _> =
                        tensor_proto.uint64_data.iter().map(|&x| u32::try_from(x)).collect();
                    match uints {
                        Ok(data) => Ok(Some(TensorData::Uint32(data))),
                        Err(_) => Err(OnnxOptError::Conversion(
                            "Uint64 value too large for uint32".to_string(),
                        )),
                    }
                } else {
                    Self::extract_from_raw_data(
                        tensor_proto.raw_data.as_ref().map(|v| v.as_slice()).unwrap_or(&[]),
                        TensorData::Uint32
                    )
                }
            }
            ProtoType::Uint64 => Self::extract_numeric_data(
                &tensor_proto.uint64_data,
                tensor_proto.raw_data.as_ref().map(|v| v.as_slice()).unwrap_or(&[]),
                TensorData::Uint64,
            ),
            ProtoType::Int8 => {
                // Special case: ONNX stores int8 in int32_data or raw_data
                if !tensor_proto.int32_data.is_empty() {
                    let bytes: Result<Vec<i8>, _> =
                        tensor_proto.int32_data.iter().map(|&x| i8::try_from(x)).collect();
                    match bytes {
                        Ok(data) => Ok(Some(TensorData::Int8(data))),
                        Err(_) => Err(OnnxOptError::Conversion(
                            "Int32 value out of range for int8".to_string(),
                        )),
                    }
                } else if let Some(raw_data) = &tensor_proto.raw_data {
                    if !raw_data.is_empty() {
                        let bytes = raw_data.iter().map(|&b| b as i8).collect();
                        Ok(Some(TensorData::Int8(bytes)))
                    } else {
                        Ok(None)
                    }
                } else {
                    Ok(None)
                }
            }
            ProtoType::Uint8 => {
                // Special case: ONNX stores uint8 in int32_data or raw_data
                if !tensor_proto.int32_data.is_empty() {
                    let bytes: Result<Vec<u8>, _> =
                        tensor_proto.int32_data.iter().map(|&x| u8::try_from(x)).collect();
                    match bytes {
                        Ok(data) => Ok(Some(TensorData::Uint8(data))),
                        Err(_) => Err(OnnxOptError::Conversion(
                            "Int32 value out of range for uint8".to_string(),
                        )),
                    }
                } else if let Some(raw_data) = &tensor_proto.raw_data {
                    if !raw_data.is_empty() {
                        Ok(Some(TensorData::Uint8(raw_data.clone())))
                    } else {
                        Ok(None)
                    }
                } else {
                    Ok(None)
                }
            }
            ProtoType::Bool => {
                if !tensor_proto.int32_data.is_empty() {
                    let bools = tensor_proto.int32_data.iter().map(|&x| x != 0).collect();
                    Ok(Some(TensorData::Bool(bools)))
                } else if let Some(raw_data) = &tensor_proto.raw_data {
                    if !raw_data.is_empty() {
                        let bools = raw_data.iter().map(|&b| b != 0).collect();
                        Ok(Some(TensorData::Bool(bools)))
                    } else {
                        Ok(None)
                    }
                } else {
                    Ok(None)
                }
            }
            ProtoType::String => {
                if !tensor_proto.string_data.is_empty() {
                    let strings: Result<Vec<String>, _> = tensor_proto
                        .string_data
                        .iter()
                        .map(|bytes| String::from_utf8(bytes.clone()))
                        .collect();
                    match strings {
                        Ok(data) => Ok(Some(TensorData::String(data))),
                        Err(e) => Err(OnnxOptError::Conversion(format!(
                            "Invalid UTF-8 in string data: {}",
                            e
                        ))),
                    }
                } else {
                    Ok(None)
                }
            }
            _ => Err(OnnxOptError::UnsupportedOp(format!(
                "Unsupported data type for tensor data extraction: {:?}",
                data_type
            ))),
        }
    }

    /// Generic helper for extracting numeric data from typed array or raw bytes
    fn extract_numeric_data<T>(
        typed_data: &[T],
        raw_data: &[u8],
        constructor: fn(Vec<T>) -> TensorData,
    ) -> Result<Option<TensorData>, OnnxOptError>
    where
        T: Clone + FromLeBytes,
    {
        if !typed_data.is_empty() {
            Ok(Some(constructor(typed_data.to_vec())))
        } else if !raw_data.is_empty() {
            Self::extract_from_raw_data(raw_data, constructor)
        } else {
            Ok(None)
        }
    }

    /// Generic helper for parsing raw byte data into typed arrays
    fn extract_from_raw_data<T>(
        raw_data: &[u8],
        constructor: fn(Vec<T>) -> TensorData,
    ) -> Result<Option<TensorData>, OnnxOptError>
    where
        T: FromLeBytes,
    {
        let type_size = std::mem::size_of::<T>();
        if raw_data.len() % type_size != 0 {
            return Err(OnnxOptError::Conversion(format!(
                "Invalid raw data length for {}: expected multiple of {}, got {}",
                std::any::type_name::<T>(),
                type_size,
                raw_data.len()
            )));
        }

        let values = raw_data
            .chunks_exact(type_size)
            .map(|chunk| T::from_le_bytes(chunk))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Some(constructor(values)))
    }
}

/// Represents a computation node in the graph
#[derive(Debug, Clone)]
pub struct Node {
    /// Node name (optional)
    pub name: Option<String>,
    /// Operation kind
    pub op_kind: OpKind,
    /// Input tensor IDs
    pub inputs: Vec<ValueId>,
    /// Output tensor IDs  
    pub outputs: Vec<ValueId>,
    /// Node attributes
    pub attributes: HashMap<String, NodeAttrValue>,
}

/// Enumeration of all supported operation types
#[derive(Debug, Clone, PartialEq, EnumString, AsRefStr, Display)]
#[strum(serialize_all = "PascalCase")]
pub enum OpKind {
    // Arithmetic operations
    Add,
    Sub,
    Mul,
    Div,

    // Neural network layers
    Conv,
    BatchNormalization,
    Relu,
    Sigmoid,
    Tanh,

    // Shape operations
    Reshape,
    Transpose,
    Concat,
    Split,

    // Pooling operations
    MaxPool,
    AveragePool,
    GlobalAveragePool,

    // Matrix operations
    MatMul,
    Gemm,

    // Utility operations
    Identity,
    Dropout,
    Constant,

    // Reduction operations
    ReduceSum,
    ReduceMean,
    ReduceMax,
    ReduceMin,

    // Logical operations
    And,
    Or,
    Not,

    // Comparison operations
    Equal,
    Greater,
    Less,

    // Custom or unknown operations
    #[strum(disabled)]
    Unknown(String),
}

impl OpKind {
    pub fn from_onnx(op: &str) -> Self {
        match OpKind::from_str(op) {
            Ok(k) => k,
            Err(_) => OpKind::Unknown(op.to_string()),
        }
    }

    pub fn as_onnx_str(&self) -> &str {
        match self {
            OpKind::Unknown(s) => s.as_str(),
            _ => self.as_ref(),
        }
    }
}
#[derive(Debug)]
pub struct Graph {
    pub nodes: Arena<Node, NodeId>,
    pub values: Arena<Tensor, ValueId>,
    pub topo_cache: Option<Vec<NodeId>>,
    // Maps each value to its producing node
    value_producer: HashMap<ValueId, NodeId>,
    // Maps each value to the list of consuming nodes
    value_consumers: HashMap<ValueId, Vec<NodeId>>,
    // Graph-level IO value ids (optional; empty by default)
    graph_input_values: Vec<ValueId>,
    graph_output_values: Vec<ValueId>,
}

/// Attribute values that can be stored in nodes
#[derive(Debug, Clone)]
pub enum NodeAttrValue {
    Int(i64),
    Float(f64),
    String(String),
    Tensor(Tensor),
    Graph(Vec<NodeId>), // For subgraphs
    Ints(Vec<i64>),
    Floats(Vec<f64>),
    Strings(Vec<String>),
}

impl NodeAttrValue {
    pub fn from_proto(attr_proto: &onnx_proto::AttributeProto) -> Result<Self, OnnxOptError> {
        use onnx_proto::attribute_proto::AttributeType;
        
        let attr_type = attr_proto
            .r#type
            .and_then(|t| AttributeType::try_from(t).ok())
            .unwrap_or(AttributeType::Undefined);
        
        match attr_type {
            AttributeType::Float => attr_proto
                .f
                .map(|f| NodeAttrValue::Float(f as f64))
                .ok_or_else(|| OnnxOptError::Conversion("Float attribute has no value".to_string())),
                
            AttributeType::Int => attr_proto
                .i
                .map(NodeAttrValue::Int)
                .ok_or_else(|| OnnxOptError::Conversion("Int attribute has no value".to_string())),
                
            AttributeType::String => attr_proto
                .s
                .as_ref()
                .ok_or_else(|| OnnxOptError::Conversion("String attribute has no value".to_string()))
                .and_then(|bytes| {
                    String::from_utf8(bytes.clone())
                        .map(NodeAttrValue::String)
                        .map_err(|e| OnnxOptError::Conversion(format!("Invalid UTF-8 in string attribute: {}", e)))
                }),
                
            AttributeType::Tensor => attr_proto
                .t
                .as_ref()
                .ok_or_else(|| OnnxOptError::Conversion("Tensor attribute has no value".to_string()))
                .and_then(|tensor_proto| Tensor::from_proto(tensor_proto).map(NodeAttrValue::Tensor)),
                
            AttributeType::Floats => Ok(NodeAttrValue::Floats(
                attr_proto.floats.iter().map(|&f| f as f64).collect()
            )),
            
            AttributeType::Ints => Ok(NodeAttrValue::Ints(attr_proto.ints.clone())),
            
            AttributeType::Strings => attr_proto
                .strings
                .iter()
                .map(|bytes| String::from_utf8(bytes.clone()))
                .collect::<Result<Vec<_>, _>>()
                .map(NodeAttrValue::Strings)
                .map_err(|e| OnnxOptError::Conversion(format!("Invalid UTF-8 in strings attribute: {}", e))),
                
            AttributeType::Graph => {
                // For now, we'll represent graphs as empty NodeId vectors
                // A more complete implementation would convert the GraphProto to NodeIds
                Ok(NodeAttrValue::Graph(Vec::new()))
            }
            
            AttributeType::Tensors => attr_proto
                .tensors
                .first()
                .ok_or_else(|| OnnxOptError::Conversion("Tensors attribute has no values".to_string()))
                .and_then(|tensor_proto| Tensor::from_proto(tensor_proto).map(NodeAttrValue::Tensor)),
                
            _ => Err(OnnxOptError::UnsupportedOp(format!(
                "Unsupported attribute type: {:?}", 
                attr_type
            ))),
        }
    }
}

impl ArenaId for NodeId {
    fn from_u32(id: u32) -> Self {
        NodeId(id)
    }

    fn into_u32(self) -> u32 {
        self.0
    }
}

impl ArenaId for ValueId {
    fn from_u32(id: u32) -> Self {
        ValueId(id)
    }

    fn into_u32(self) -> u32 {
        self.0
    }
}

impl Tensor {
    pub fn new(dtype: DataType) -> Self {
        Self {
            name: None,
            shape: None,
            dtype,
            data: None,
        }
    }

    pub fn from_proto(tensor_proto: &TensorProto) -> Result<Self, OnnxOptError> {
        // Convert proto data type to internal DataType
        let proto_data_type = ProtoType::try_from(tensor_proto.data_type.unwrap_or(0))
            .map_err(|_| OnnxOptError::Conversion("Invalid data type".to_string()))?;
        let dtype = DataType::from_proto_data_type(proto_data_type)?;

        // Extract shape from dims
        let shape = if tensor_proto.dims.is_empty() {
            None
        } else {
            Some(tensor_proto.dims.clone())
        };

        // Extract name
        let name = tensor_proto.name.clone();

        // Extract tensor data if present
        let data = TensorData::from_proto(tensor_proto)?;

        Ok(Self {
            name,
            shape,
            dtype,
            data,
        })
    }

    pub fn has_data(&self) -> bool {
        self.data.is_some()
    }

    pub fn with_name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }

    /// Create a Tensor from ValueInfoProto (used for graph inputs/outputs/value_info)
    pub fn from_value_info_proto(value_info: &onnx_proto::ValueInfoProto) -> Result<Self, OnnxOptError> {
        let name = value_info.name.clone();
        
        let (dtype, shape) = value_info
            .r#type
            .as_ref()
            .and_then(|type_proto| type_proto.value.as_ref())
            .and_then(|value| match value {
                onnx_proto::type_proto::Value::TensorType(tensor_type) => {
                    let elem_type = tensor_type.elem_type?;
                    let proto_type = onnx_proto::tensor_proto::DataType::try_from(elem_type).ok()?;
                    let dtype = DataType::from_proto_data_type(proto_type).ok()?;
                    
                    let shape = tensor_type
                        .shape
                        .as_ref()
                        .map(|shape_proto| {
                            shape_proto
                                .dim
                                .iter()
                                .filter_map(|dim| {
                                    dim.value.as_ref().and_then(|v| match v {
                                        onnx_proto::tensor_shape_proto::dimension::Value::DimValue(val) => Some(*val),
                                        _ => None, // Skip symbolic dimensions for now
                                    })
                                })
                                .collect::<Vec<_>>()
                        })
                        .filter(|v| !v.is_empty());
                    
                    Some((dtype, shape))
                }
                _ => None, // Only handle tensor types for now
            })
            .unwrap_or((DataType::Float32, None)); // Default fallback

        Ok(Self {
            name,
            shape,
            dtype,
            data: None, // Value info doesn't contain actual data
        })
    }
}

impl Node {
    pub fn new(op_kind: OpKind) -> Self {
        Self {
            name: None,
            op_kind,
            inputs: Vec::new(),
            outputs: Vec::new(),
            attributes: HashMap::new(),
        }
    }
    pub fn add_attribute(mut self, key: String, value: NodeAttrValue) -> Self {
        self.attributes.insert(key, value);
        self
    }

    pub fn with_inputs(mut self, inputs: Vec<ValueId>) -> Self {
        self.inputs = inputs;
        self
    }

    pub fn with_outputs(mut self, outputs: Vec<ValueId>) -> Self {
        self.outputs = outputs;
        self
    }

    pub fn from_node_proto(node_proto: &onnx_proto::NodeProto) -> Result<Self, OnnxOptError> {
        let op_kind = node_proto
            .op_type
            .as_ref()
            .map(|op_type| OpKind::from_onnx(op_type))
            .ok_or_else(|| OnnxOptError::Conversion("Missing op_type in NodeProto".to_string()))?;

        let attributes = node_proto
            .attribute
            .iter()
            .map(|attr_proto| {
                let name = attr_proto
                    .name
                    .as_ref()
                    .ok_or_else(|| OnnxOptError::Conversion("Missing name in AttributeProto".to_string()))?;
                
                let value = NodeAttrValue::from_proto(attr_proto)?;
                Ok((name.clone(), value))
            })
            .collect::<Result<HashMap<_, _>, OnnxOptError>>()?;

        Ok(Self {
            name: node_proto.name.clone(),
            op_kind,
            inputs: Vec::new(),
            outputs: Vec::new(),
            attributes,
        })
    }
}

impl Graph {
    pub fn new() -> Self {
        Self {
            nodes: Arena::new(),
            values: Arena::new(),
            topo_cache: None,
            value_producer: HashMap::new(),
            value_consumers: HashMap::new(),
            graph_input_values: Vec::new(),
            graph_output_values: Vec::new(),
        }
    }

    pub fn from_model_proto(model: &ModelProto) -> Result<Self, OnnxOptError> {
        let graph_proto = model
            .graph
            .as_ref()
            .ok_or(OnnxOptError::InvalidModel("Model has no graph".to_string()))?;
        Self::from_graph_proto(graph_proto)
    }

    pub fn from_graph_proto(graph_proto: &GraphProto) -> Result<Self, OnnxOptError> {
        let mut graph = Self::new();
        let mut name_to_value_id = HashMap::new();

        // Step 1: Process initializers (constant tensors with data)
        let initializer_mappings: Vec<(String, ValueId)> = graph_proto
            .initializer
            .iter()
            .map(|tensor_proto| {
                let tensor = Tensor::from_proto(tensor_proto)?;
                let name = tensor.name.clone().unwrap_or_default();
                let value_id = graph.add_value(tensor);
                Ok((name, value_id))
            })
            .collect::<Result<Vec<_>, OnnxOptError>>()?;
        
        name_to_value_id.extend(initializer_mappings);

        // Step 2: Process graph inputs (create tensors from ValueInfoProto)
        let input_mappings: Vec<(String, ValueId)> = graph_proto
            .input
            .iter()
            .map(|value_info| {
                let tensor = Tensor::from_value_info_proto(value_info)?;
                let name = tensor.name.clone().unwrap_or_default();
                let value_id = graph.add_value(tensor);
                Ok((name, value_id))
            })
            .collect::<Result<Vec<_>, OnnxOptError>>()?;
        
        graph.graph_input_values = input_mappings.iter().map(|(_, value_id)| *value_id).collect();
        name_to_value_id.extend(input_mappings);

        // Step 3: Process graph outputs (create tensors from ValueInfoProto)
        let output_mappings: Vec<(String, ValueId)> = graph_proto
            .output
            .iter()
            .map(|value_info| {
                let tensor = Tensor::from_value_info_proto(value_info)?;
                let name = tensor.name.clone().unwrap_or_default();
                let value_id = graph.add_value(tensor);
                Ok((name, value_id))
            })
            .collect::<Result<Vec<_>, OnnxOptError>>()?;
        
        graph.graph_output_values = output_mappings.iter().map(|(_, value_id)| *value_id).collect();
        name_to_value_id.extend(output_mappings);

        // Step 4: Process intermediate value_info (if not already processed)
        for value_info in &graph_proto.value_info {
            let name = value_info.name.clone().unwrap_or_default();
            if !name_to_value_id.contains_key(&name) {
                let tensor = Tensor::from_value_info_proto(value_info)?;
                let value_id = graph.add_value(tensor);
                name_to_value_id.insert(name, value_id);
            }
        }

        // Step 5: Process nodes and resolve input/output ValueIds
        for node_proto in &graph_proto.node {
            let mut node = Node::from_node_proto(node_proto)?;
            
            // Resolve input names to ValueIds
            node.inputs = node_proto
                .input
                .iter()
                .filter_map(|input_name| {
                    name_to_value_id.get(input_name).copied().or_else(|| {
                        // Create a placeholder tensor for unknown inputs
                        let tensor = Tensor::new(DataType::Float32).with_name(input_name.clone());
                        let value_id = graph.add_value(tensor);
                        name_to_value_id.insert(input_name.clone(), value_id);
                        Some(value_id)
                    })
                })
                .collect();
            
            // Resolve output names to ValueIds
            node.outputs = node_proto
                .output
                .iter()
                .filter_map(|output_name| {
                    name_to_value_id.get(output_name).copied().or_else(|| {
                        // Create a placeholder tensor for unknown outputs
                        let tensor = Tensor::new(DataType::Float32).with_name(output_name.clone());
                        let value_id = graph.add_value(tensor);
                        name_to_value_id.insert(output_name.clone(), value_id);
                        Some(value_id)
                    })
                })
                .collect();
            
            graph.add_node(node);
        }

        Ok(graph)
    }
}

impl GraphView for Graph {
    fn node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(id)
    }

    fn tensor(&self, id: ValueId) -> Option<&Tensor> {
        self.values.get(id)
    }

    fn inputs(&self, node: NodeId) -> &[ValueId] {
        self.nodes.get(node).map(|n| n.inputs.as_slice()).unwrap_or(&[])
    }

    fn outputs(&self, node: NodeId) -> &[ValueId] {
        self.nodes.get(node).map(|n| n.outputs.as_slice()).unwrap_or(&[])
    }

    fn producer(&self, value: ValueId) -> Option<NodeId> {
        self.value_producer.get(&value).copied()
    }

    fn consumers(&self, value: ValueId) -> &[NodeId] {
        self.value_consumers.get(&value).map(|v| v.as_slice()).unwrap_or(&[])
    }

    fn graph_inputs(&self) -> &[ValueId] {
        self.graph_input_values.as_slice()
    }

    fn graph_outputs(&self) -> &[ValueId] {
        self.graph_output_values.as_slice()
    }
}

impl GraphEdit for Graph {
    fn add_node(&mut self, node: Node) -> NodeId {
        let node_id = self.nodes.alloc(node);
        if let Some(n) = self.nodes.get(node_id) {
            // Register producers for each output
            self.value_producer
                .extend(n.outputs.iter().map(|&value_id| (value_id, node_id)));

            // Register consumers for each input
            n.inputs.iter().for_each(|&value_id| {
                self.value_consumers.entry(value_id).or_default().push(node_id);
            });
        }
        // Invalidate cached topology if present
        self.topo_cache = None;
        node_id
    }

    fn add_value(&mut self, tensor: Tensor) -> ValueId {
        self.values.alloc(tensor)
    }

    fn remove_node(&mut self, node: NodeId) {
        if let Some(removed) = self.nodes.free(node) {
            // Clean up producer entries for outputs
            for value_id in removed.outputs {
                self.value_producer.remove(&value_id);
            }
            // Clean up consumer entries for inputs
            for value_id in removed.inputs {
                if let Some(consumers) = self.value_consumers.get_mut(&value_id) {
                    consumers.retain(|&n| n != node);
                    if consumers.is_empty() {
                        self.value_consumers.remove(&value_id);
                    }
                }
            }
            self.topo_cache = None;
        }
    }

    fn remove_value(&mut self, value: ValueId) {
        let _ = self.values.free(value);
        self.value_producer.remove(&value);
        self.value_consumers.remove(&value);
        self.topo_cache = None;
    }

    fn invalidate_topology(&mut self) {
        self.topo_cache = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::traits::{GraphEdit, GraphView};

    // Test utilities for creating reusable objects
    fn create_test_tensor(name: &str) -> Tensor {
        Tensor::new(DataType::Float32).with_name(name.to_string())
    }

    fn create_test_node_with_io(
        op_kind: OpKind,
        inputs: Vec<ValueId>,
        outputs: Vec<ValueId>,
    ) -> Node {
        Node::new(op_kind).with_inputs(inputs).with_outputs(outputs)
    }

    fn create_test_graph_with_values() -> (Graph, Vec<ValueId>) {
        let mut graph = Graph::new();
        let value_ids = (0..5)
            .map(|i| graph.add_value(create_test_tensor(&format!("tensor_{}", i))))
            .collect();
        (graph, value_ids)
    }

    #[test]
    fn test_from_onnx_known_ops() {
        assert_eq!(OpKind::from_onnx("Add"), OpKind::Add);
        assert_eq!(OpKind::from_onnx("Sub"), OpKind::Sub);
        assert_eq!(OpKind::from_onnx("Mul"), OpKind::Mul);
        assert_eq!(OpKind::from_onnx("Div"), OpKind::Div);

        assert_eq!(OpKind::from_onnx("Relu"), OpKind::Relu);
        assert_eq!(OpKind::from_onnx("MatMul"), OpKind::MatMul);
        assert_eq!(OpKind::from_onnx("Gemm"), OpKind::Gemm);
        assert_eq!(OpKind::from_onnx("Concat"), OpKind::Concat);
        assert_eq!(OpKind::from_onnx("ReduceMean"), OpKind::ReduceMean);
        assert_eq!(OpKind::from_onnx("MaxPool"), OpKind::MaxPool);
    }

    #[test]
    fn test_from_onnx_unknown_op() {
        let k = OpKind::from_onnx("CustomOp_XYZ");
        assert_eq!(k, OpKind::Unknown("CustomOp_XYZ".to_string()));
        // as_onnx_str should return the original op string for Unknown
        assert_eq!(k.as_onnx_str(), "CustomOp_XYZ");
    }

    #[test]
    fn test_roundtrip_known_ops() {
        let ops = [
            OpKind::Add,
            OpKind::Sub,
            OpKind::Mul,
            OpKind::Div,
            OpKind::Relu,
            OpKind::MatMul,
            OpKind::Gemm,
            OpKind::Concat,
            OpKind::ReduceMean,
            OpKind::MaxPool,
        ];

        for k in ops.iter() {
            let s = k.as_onnx_str();
            let back = OpKind::from_onnx(s);
            assert_eq!(&back, k, "roundtrip failed for {}", s);
        }
    }

    // Tests for add_node function
    #[test]
    fn test_add_node_basic() {
        let mut graph = Graph::new();
        let node = Node::new(OpKind::Add);

        let node_id = graph.add_node(node);

        // Verify the node was added
        assert!(graph.node(node_id).is_some());
        assert_eq!(graph.node(node_id).unwrap().op_kind, OpKind::Add);
    }

    #[test]
    fn test_add_node_with_inputs_and_outputs() {
        let (mut graph, value_ids) = create_test_graph_with_values();
        let inputs = vec![value_ids[0], value_ids[1]];
        let outputs = vec![value_ids[2]];

        let node = create_test_node_with_io(OpKind::Add, inputs.clone(), outputs.clone());
        let node_id = graph.add_node(node);

        // Verify node was added with correct inputs/outputs
        let added_node = graph.node(node_id).unwrap();
        assert_eq!(added_node.inputs, inputs);
        assert_eq!(added_node.outputs, outputs);
    }

    #[test]
    fn test_add_node_producer_registration() {
        let (mut graph, value_ids) = create_test_graph_with_values();
        let outputs = vec![value_ids[0], value_ids[1]];

        let node = create_test_node_with_io(OpKind::Relu, vec![], outputs.clone());
        let node_id = graph.add_node(node);

        // Verify all outputs are registered as produced by this node
        for &output_id in &outputs {
            assert_eq!(graph.producer(output_id), Some(node_id));
        }
    }

    #[test]
    fn test_add_node_consumer_registration() {
        let (mut graph, value_ids) = create_test_graph_with_values();
        let inputs = vec![value_ids[0], value_ids[1], value_ids[2]];

        let node = create_test_node_with_io(OpKind::Concat, inputs.clone(), vec![]);
        let node_id = graph.add_node(node);

        // Verify all inputs are registered as consumed by this node
        for &input_id in &inputs {
            assert!(graph.consumers(input_id).contains(&node_id));
        }
    }

    #[test]
    fn test_add_node_multiple_consumers() {
        let (mut graph, value_ids) = create_test_graph_with_values();
        let shared_input = value_ids[0];

        // Add first node that consumes the value
        let node1 = create_test_node_with_io(OpKind::Relu, vec![shared_input], vec![]);
        let node_id1 = graph.add_node(node1);

        // Add second node that also consumes the same value
        let node2 = create_test_node_with_io(OpKind::Sigmoid, vec![shared_input], vec![]);
        let node_id2 = graph.add_node(node2);

        // Verify both nodes are registered as consumers
        let consumers = graph.consumers(shared_input);
        assert_eq!(consumers.len(), 2);
        assert!(consumers.contains(&node_id1));
        assert!(consumers.contains(&node_id2));
    }

    #[test]
    fn test_add_node_topology_cache_invalidation() {
        let mut graph = Graph::new();

        // Set up a fake topology cache
        graph.topo_cache = Some(vec![]);

        let node = Node::new(OpKind::Identity);
        graph.add_node(node);

        // Verify topology cache was invalidated
        assert!(graph.topo_cache.is_none());
    }

    #[test]
    fn test_add_node_no_inputs_or_outputs() {
        let mut graph = Graph::new();
        let node = Node::new(OpKind::Constant);

        let node_id = graph.add_node(node);

        // Verify node was added successfully even with no inputs/outputs
        let added_node = graph.node(node_id).unwrap();
        assert!(added_node.inputs.is_empty());
        assert!(added_node.outputs.is_empty());
        assert_eq!(added_node.op_kind, OpKind::Constant);
    }

    #[test]
    fn test_add_node_complex_graph() {
        let (mut graph, value_ids) = create_test_graph_with_values();

        // Create a small computation graph: Add -> Relu -> MatMul
        let add_node = create_test_node_with_io(
            OpKind::Add,
            vec![value_ids[0], value_ids[1]],
            vec![value_ids[2]],
        );
        let add_id = graph.add_node(add_node);

        let relu_node =
            create_test_node_with_io(OpKind::Relu, vec![value_ids[2]], vec![value_ids[3]]);
        let relu_id = graph.add_node(relu_node);

        let matmul_node =
            create_test_node_with_io(OpKind::MatMul, vec![value_ids[3], value_ids[4]], vec![]);
        let matmul_id = graph.add_node(matmul_node);

        // Verify the graph structure
        assert_eq!(graph.producer(value_ids[2]), Some(add_id));
        assert_eq!(graph.producer(value_ids[3]), Some(relu_id));

        assert!(graph.consumers(value_ids[0]).contains(&add_id));
        assert!(graph.consumers(value_ids[1]).contains(&add_id));
        assert!(graph.consumers(value_ids[2]).contains(&relu_id));
        assert!(graph.consumers(value_ids[3]).contains(&matmul_id));
        assert!(graph.consumers(value_ids[4]).contains(&matmul_id));
    }

    #[test]
    fn test_add_node_with_attributes() {
        let mut graph = Graph::new();
        let mut node = Node::new(OpKind::Conv);
        node = node.add_attribute("kernel_shape".to_string(), NodeAttrValue::Ints(vec![3, 3]));
        node = node.add_attribute("strides".to_string(), NodeAttrValue::Ints(vec![1, 1]));

        let node_id = graph.add_node(node);

        // Verify node with attributes was added correctly
        let added_node = graph.node(node_id).unwrap();
        assert_eq!(added_node.op_kind, OpKind::Conv);
        assert_eq!(added_node.attributes.len(), 2);
        assert!(added_node.attributes.contains_key("kernel_shape"));
        assert!(added_node.attributes.contains_key("strides"));
    }
}
