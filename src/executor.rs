use crate::utils::config::OptimizationConfig;
use crate::{
    graph::Graph,
    graph::{
        error::GraphError,
        objects::{NodeId, OpKind, Tensor},
        traits::{GraphAnalysis, GraphModification},
    },
    passes::{error::PassError, traits::BasicOptimization},
};
use std::collections::HashSet;

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
    graph: Graph,
    /// Cached topological order for efficient traversal
    cached_topo_order: Option<Vec<NodeId>>,
    /// Nodes that have been modified and may need topology recalculation
    dirty_nodes: HashSet<NodeId>,
    /// Configuration for optimization behavior
    config: OptimizationConfig,
    /// Statistics collected during optimization
    stats: OptimizationStats,
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

    
}