use crate::graph::{
    Graph,
    objects::{NodeId, OpKind, ValueId},
    traits::{GraphView, GraphEdit},
};
use crate::passes::{error::PassError, traits::OptimizationPass};


/// remove cast operations whose input and output are equal.
/// e.g. f32 -> cast -> f32
#[derive(Debug, Clone)]
pub struct EliminateNopCast;

pub impl EliminateNopCast{
    pub fn new() -> Self {
        Self
    }

    fn find_cast_nodes(&self, graph: &Graph) -> Vec<NodeId> {


    }
}


#[cfg(test)]
mod tests {
    use super::*;

}