use std::collections::HashMap;
use std::str::FromStr;
use crate::proto;
use strum_macros::{AsRefStr, Display, EnumString};
use crate::utils::arena::{Arena, ArenaId};
use crate::graph::traits::{GraphView, GraphEdit};

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
    pub attributes: HashMap<String, AttrValue>,
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
pub enum AttrValue {
    Int(i64),
    Float(f64),
    String(String),
    Tensor(Tensor),
    Graph(Vec<NodeId>), // For subgraphs
    Ints(Vec<i64>),
    Floats(Vec<f64>),
    Strings(Vec<String>),
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
    
    pub fn with_name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }
    
    pub fn with_shape(mut self, shape: Vec<i64>) -> Self {
        self.shape = Some(shape);
        self
    }
    
    pub fn with_data(mut self, data: TensorData) -> Self {
        self.data = Some(data);
        self
    }
    
    /// Check if this tensor is a constant (has data)
    pub fn is_constant(&self) -> bool {
        self.data.is_some()
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
    
    pub fn with_name(mut self, name: String) -> Self {
        self.name = Some(name);
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
    
    pub fn add_attribute(mut self, key: String, value: AttrValue) -> Self {
        self.attributes.insert(key, value);
        self
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

}


impl GraphView for Graph {
    fn node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(id)
    }

    fn tensor(&self, id: ValueId) -> Option<&Tensor> {
        self.values.get(id)
    }

    fn inputs(&self, node: NodeId) -> &[ValueId] {
        self.nodes
            .get(node)
            .map(|n| n.inputs.as_slice())
            .unwrap_or(&[])
    }

    fn outputs(&self, node: NodeId) -> &[ValueId] {
        self.nodes
            .get(node)
            .map(|n| n.outputs.as_slice())
            .unwrap_or(&[])
    }

    fn producer(&self, value: ValueId) -> Option<NodeId> {
        self.value_producer.get(&value).copied()
    }

    fn consumers(&self, value: ValueId) -> &[NodeId] {
        self
            .value_consumers
            .get(&value)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
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
            for &value_id in &n.outputs {
                self.value_producer.insert(value_id, node_id);
            }
            // Register consumers for each input
            for &value_id in &n.inputs {
                self
                    .value_consumers
                    .entry(value_id)
                    .or_default()
                    .push(node_id);
            }
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
    use super::OpKind;

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
}
