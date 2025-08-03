use std::collections::HashMap;

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
#[derive(Debug, Clone, PartialEq)]
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
    Unknown(String),
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

impl NodeId {
    pub fn new(id: u32) -> Self {
        NodeId(id)
    }
    
    pub fn as_u32(&self) -> u32 {
        self.0
    }
}

impl ValueId {
    pub fn new(id: u32) -> Self {
        ValueId(id)
    }
    
    pub fn as_u32(&self) -> u32 {
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