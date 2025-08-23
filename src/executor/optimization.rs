use std::collections::HashSet;

use crate::{
    utils::config::OptimizationConfig,
    graph::{
        Graph,
        error::GraphError,
        objects::{Node, NodeId, OpKind, Tensor, ValueId},
        traits::{GraphAnalysis, GraphView},
    },
    passes::{error::PassError, traits::OptimizationPass},
};
use petgraph::algo::toposort;

/// Statistics collected during optimization
#[derive(Debug, Default, Clone)]
pub struct OptimizationStats {
    /// Number of nodes that had constants folded
    pub constants_folded: u32,
    /// Number of dead nodes eliminated
    pub dead_nodes_eliminated: u32,
    /// Number of identity nodes removed
    pub identity_nodes_removed: u32,
    /// Number of dropout nodes removed
    pub dropout_nodes_removed: u32,
    /// Number of operator fusions performed
    pub operators_fused: u32,
    /// Number of optimization passes executed
    pub passes_executed: u32,
}

/// Single-threaded optimization executor that owns the graph
///
/// This executor takes ownership of the graph, applies optimization passes,
/// and returns the optimized graph.
pub struct OptimizationExecutor {
    /// The graph being optimized (owned)
    pub graph: Graph,
    /// Cached topological order for efficient traversal
    pub cached_topo_order: Option<Vec<NodeId>>,
    /// Nodes that have been modified and may need topology recalculation
    pub dirty_nodes: HashSet<NodeId>,
    /// Configuration for optimization behavior
    pub config: OptimizationConfig,
    /// Statistics collected during optimization
    pub stats: OptimizationStats,
}

impl GraphView for OptimizationExecutor {
    fn node(&self, id: NodeId) -> Option<&Node> {
        self.graph.node(id)
    }

    fn tensor(&self, id: ValueId) -> Option<&Tensor> {
        self.graph.tensor(id)
    }

    fn node_ids(&self) -> Vec<NodeId> {
        self.graph.node_ids()
    }

    fn inputs(&self, node: NodeId) -> &[ValueId] {
        self.graph.inputs(node)
    }

    fn outputs(&self, node: NodeId) -> &[ValueId] {
        self.graph.outputs(node)
    }

    fn producer(&self, value: ValueId) -> Option<NodeId> {
        self.graph.producer(value)
    }

    fn consumers(&self, value: ValueId) -> Vec<NodeId> {
        self.graph.consumers(value)
    }

    fn graph_inputs(&self) -> &[ValueId] {
        self.graph.graph_inputs()
    }

    fn graph_outputs(&self) -> &[ValueId] {
        self.graph.graph_outputs()
    }
}

impl OptimizationExecutor {
    pub fn new(graph: Graph, config: OptimizationConfig) -> Self {
        Self {
            graph,
            config,
            stats: OptimizationStats::default(),
            cached_topo_order: None,
            dirty_nodes: HashSet::new(),
        }
    }

    pub fn execute<P: OptimizationPass>(&mut self, passes: &mut [P]) -> Result<(), PassError> {
        todo!()
    }
}

impl GraphAnalysis for OptimizationExecutor {
    fn get_topological_order(&mut self) -> Result<&[NodeId], GraphError> {
        // ONNX specification mandates that nodes
        // must be topologically sorted (see github.com/onnx/onnx/issues/3865).
        let current_node_count = self.graph.node_count();

        // Check if cache is valid: exists and has correct number of nodes
        let cache_valid = self
            .cached_topo_order
            .as_ref()
            .map(|order| order.len() == current_node_count)
            .unwrap_or(false);

        match cache_valid {
            false => {
                let order =
                    toposort(&self.graph.nodes, None).map_err(|_| GraphError::CyclicGraph)?;

                Ok(self.cached_topo_order.insert(order))
            }
            true => Ok(self.cached_topo_order.as_ref().unwrap()),
        }
    }

    fn invalidate_topology(&mut self) {
        self.cached_topo_order = None;
    }

    fn can_fold_constant(&self, _node_id: NodeId) -> bool {
        // Placeholder – real logic will inspect node and inputs
        true
    }

    fn is_dead_node(&self, _node_id: NodeId) -> bool {
        // Placeholder – real logic will do reachability from graph outputs
        false
    }

    fn supports_constant_folding(_op_kind: &OpKind) -> bool {
        // Placeholder – whitelist ops that are pure and evaluable
        true
    }

    fn validate_graph(&self) -> Result<(), GraphError> {
        // Placeholder – ensure basic invariants
        Ok(())
    }
}
