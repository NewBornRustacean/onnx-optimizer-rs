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

    pub fn execute<P: OptimizationPass>(&mut self, _passes: &mut [P]) -> Result<(), PassError> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{
        objects::{DataType, OpKind},
        traits::GraphEdit,
    };
    use crate::utils::config::OptimizationConfig;

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

    fn create_test_executor_with_values() -> (OptimizationExecutor, Vec<ValueId>) {
        let mut graph = Graph::new();
        let value_ids = (0..5)
            .map(|i| graph.add_value(create_test_tensor(&format!("tensor_{}", i))))
            .collect();
        let config = OptimizationConfig::default();
        let executor = OptimizationExecutor::new(graph, config);
        (executor, value_ids)
    }

    #[test]
    fn test_empty_graph_topological_order() {
        let graph = Graph::new();
        let config = OptimizationConfig::default();
        let mut executor = OptimizationExecutor::new(graph, config);

        let topo_order = executor.get_topological_order().unwrap();
        assert_eq!(topo_order.len(), 0);
    }

    #[test]
    fn test_single_node_topological_order() {
        let mut graph = Graph::new();
        let input_value = graph.add_value(create_test_tensor("input"));
        let output_value = graph.add_value(create_test_tensor("output"));

        let node = create_test_node_with_io(
            OpKind::Relu,
            vec![input_value],
            vec![output_value],
        );
        let node_id = graph.add_node(node);

        let config = OptimizationConfig::default();
        let mut executor = OptimizationExecutor::new(graph, config);

        let topo_order = executor.get_topological_order().unwrap();
        assert_eq!(topo_order.len(), 1);
        assert_eq!(topo_order[0], node_id);
    }

    #[test]
    fn test_linear_chain_topological_order() {
        // Create a linear chain: input -> Relu -> Add -> output
        let mut graph = Graph::new();
        
        // Create values
        let input = graph.add_value(create_test_tensor("input"));
        let intermediate1 = graph.add_value(create_test_tensor("relu_out"));
        let intermediate2 = graph.add_value(create_test_tensor("add_bias"));
        let output = graph.add_value(create_test_tensor("output"));

        // Create nodes in dependency order
        let relu_node = create_test_node_with_io(
            OpKind::Relu,
            vec![input],
            vec![intermediate1],
        );
        let relu_id = graph.add_node(relu_node);

        let add_node = create_test_node_with_io(
            OpKind::Add,
            vec![intermediate1, intermediate2],
            vec![output],
        );
        let add_id = graph.add_node(add_node);

        let config = OptimizationConfig::default();
        let mut executor = OptimizationExecutor::new(graph, config);

        let topo_order = executor.get_topological_order().unwrap();
        assert_eq!(topo_order.len(), 2);
        
        // Verify topological ordering: relu_id should come before add_id
        let relu_pos = topo_order.iter().position(|&id| id == relu_id).unwrap();
        let add_pos = topo_order.iter().position(|&id| id == add_id).unwrap();
        assert!(relu_pos < add_pos, "Relu should come before Add in topological order");
    }

    #[test]
    fn test_diamond_dependency_topological_order() {
        // Create a diamond dependency graph:
        //     input
        //    /     \
        //   A       B
        //    \     /
        //     Add
        //      |
        //    output
        let mut graph = Graph::new();
        
        // Create values
        let input = graph.add_value(create_test_tensor("input"));
        let a_out = graph.add_value(create_test_tensor("a_out"));
        let b_out = graph.add_value(create_test_tensor("b_out"));
        let output = graph.add_value(create_test_tensor("output"));

        // Create nodes
        let node_a = create_test_node_with_io(
            OpKind::Relu,
            vec![input],
            vec![a_out],
        );
        let a_id = graph.add_node(node_a);

        let node_b = create_test_node_with_io(
            OpKind::Sigmoid,
            vec![input],
            vec![b_out],
        );
        let b_id = graph.add_node(node_b);

        let add_node = create_test_node_with_io(
            OpKind::Add,
            vec![a_out, b_out],
            vec![output],
        );
        let add_id = graph.add_node(add_node);

        let config = OptimizationConfig::default();
        let mut executor = OptimizationExecutor::new(graph, config);

        let topo_order = executor.get_topological_order().unwrap();
        assert_eq!(topo_order.len(), 3);

        // Verify that both A and B come before Add
        let a_pos = topo_order.iter().position(|&id| id == a_id).unwrap();
        let b_pos = topo_order.iter().position(|&id| id == b_id).unwrap();
        let add_pos = topo_order.iter().position(|&id| id == add_id).unwrap();
        
        assert!(a_pos < add_pos, "Node A should come before Add");
        assert!(b_pos < add_pos, "Node B should come before Add");
    }

    #[test]
    fn test_topological_order_caching() {
        let (mut executor, _) = create_test_executor_with_values();

        // First call - should compute and cache
        let topo_order1_ptr = {
            let topo_order1 = executor.get_topological_order().unwrap();
            topo_order1.as_ptr()
        };
        
        // Second call - should return cached result
        let topo_order2_ptr = {
            let topo_order2 = executor.get_topological_order().unwrap();
            topo_order2.as_ptr()
        };
        
        // They should be the same reference (pointer equality)
        assert_eq!(topo_order1_ptr, topo_order2_ptr);
    }

    #[test]
    fn test_topological_order_cache_invalidation() {
        let mut graph = Graph::new();
        let input = graph.add_value(create_test_tensor("input"));
        let output = graph.add_value(create_test_tensor("output"));

        let config = OptimizationConfig::default();
        let mut executor = OptimizationExecutor::new(graph, config);

        // Get initial topological order (empty graph)
        let initial_order = executor.get_topological_order().unwrap();
        assert_eq!(initial_order.len(), 0);

        // Add a node directly to the graph (simulating graph modification)
        let node = create_test_node_with_io(OpKind::Relu, vec![input], vec![output]);
        executor.graph.add_node(node);

        // Cache should be invalid now due to node count change
        let new_order = executor.get_topological_order().unwrap();
        assert_eq!(new_order.len(), 1);
    }

    #[test]
    fn test_invalidate_topology_explicitly() {
        let (mut executor, _) = create_test_executor_with_values();

        // Get topological order to populate cache
        let _order1 = executor.get_topological_order().unwrap();
        
        // Explicitly invalidate
        executor.invalidate_topology();
        
        // Next call should recompute
        let _order2 = executor.get_topological_order().unwrap();
        
        // This test mainly ensures the invalidation doesn't panic
        // and that subsequent calls work correctly
    }

    #[test] 
    fn test_multiple_independent_chains() {
        // Create multiple independent chains:
        // Chain 1: input1 -> relu1 -> output1
        // Chain 2: input2 -> sigmoid2 -> output2
        let mut graph = Graph::new();
        
        // Chain 1
        let input1 = graph.add_value(create_test_tensor("input1"));
        let output1 = graph.add_value(create_test_tensor("output1"));
        let relu1 = create_test_node_with_io(OpKind::Relu, vec![input1], vec![output1]);
        let relu1_id = graph.add_node(relu1);

        // Chain 2
        let input2 = graph.add_value(create_test_tensor("input2"));
        let output2 = graph.add_value(create_test_tensor("output2"));
        let sigmoid2 = create_test_node_with_io(OpKind::Sigmoid, vec![input2], vec![output2]);
        let sigmoid2_id = graph.add_node(sigmoid2);

        let config = OptimizationConfig::default();
        let mut executor = OptimizationExecutor::new(graph, config);

        let topo_order = executor.get_topological_order().unwrap();
        assert_eq!(topo_order.len(), 2);
        
        // Both nodes should be in the order (exact order doesn't matter for independent chains)
        assert!(topo_order.contains(&relu1_id));
        assert!(topo_order.contains(&sigmoid2_id));
    }
}
