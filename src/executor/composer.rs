use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use sysinfo::System;

use crate::{
    graph::Graph,
    passes::{error::PassError, manager::PassManager},
};

/// Statistics collected during optimization
#[derive(Debug, Clone)]
pub struct PassStatistics {
    pub pass_name: String,
    pub execution_time: Duration,
    pub changes_made: u32,
    pub memory_before: u64,
    pub memory_after: u64,
    pub nodes_before: usize,
    pub nodes_after: usize,
    pub edges_before: usize,
    pub edges_after: usize,
    pub success: bool,
    pub error_message: Option<String>,
}

/// Overall optimization statistics
#[derive(Debug, Clone)]
pub struct OptimizationStatistics {
    pub total_execution_time: Duration,
    pub total_changes: u32,
    pub total_passes: usize,
    pub successful_passes: usize,
    pub failed_passes: usize,
    pub pass_statistics: Vec<PassStatistics>,
    pub memory_peak: u64,
    pub initial_graph_size: (usize, usize), // (nodes, edges)
    pub final_graph_size: (usize, usize),   // (nodes, edges)
}

impl OptimizationStatistics {
    pub fn new() -> Self {
        Self {
            total_execution_time: Duration::ZERO,
            total_changes: 0,
            total_passes: 0,
            successful_passes: 0,
            failed_passes: 0,
            pass_statistics: Vec::new(),
            memory_peak: 0,
            initial_graph_size: (0, 0),
            final_graph_size: (0, 0),
        }
    }

    /// Print a detailed statistics report
    pub fn print_report(&self) {
        println!("\n=== Optimization Statistics Report ===");
        println!("Total execution time: {:.2?}", self.total_execution_time);
        println!("Total changes made: {}", self.total_changes);
        println!("Total passes: {}", self.total_passes);
        println!("Successful passes: {}", self.successful_passes);
        println!("Failed passes: {}", self.failed_passes);
        println!(
            "Success rate: {:.1}%",
            (self.successful_passes as f64 / self.total_passes as f64) * 100.0
        );

        println!("\nGraph size changes:");
        println!(
            "  Initial: {} nodes, {} edges",
            self.initial_graph_size.0, self.initial_graph_size.1
        );
        println!(
            "  Final: {} nodes, {} edges",
            self.final_graph_size.0, self.final_graph_size.1
        );

        let node_reduction = self.initial_graph_size.0 as i32 - self.final_graph_size.0 as i32;
        let edge_reduction = self.initial_graph_size.1 as i32 - self.final_graph_size.1 as i32;

        println!(
            "  Reduction: {} nodes ({:.1}%), {} edges ({:.1}%)",
            node_reduction,
            if self.initial_graph_size.0 > 0 {
                (node_reduction as f64 / self.initial_graph_size.0 as f64) * 100.0
            } else {
                0.0
            },
            edge_reduction,
            if self.initial_graph_size.1 > 0 {
                (edge_reduction as f64 / self.initial_graph_size.1 as f64) * 100.0
            } else {
                0.0
            }
        );

        println!(
            "Peak memory usage: {:.2} MB",
            self.memory_peak as f64 / 1024.0 / 1024.0
        );

        if !self.pass_statistics.is_empty() {
            println!("\n--- Pass-by-Pass Statistics ---");
            for stat in &self.pass_statistics {
                let status = if stat.success { "✓" } else { "✗" };
                println!(
                    "{} {}: {:.2?}, {} changes, {:.1} MB",
                    status,
                    stat.pass_name,
                    stat.execution_time,
                    stat.changes_made,
                    stat.memory_after as f64 / 1024.0 / 1024.0
                );

                if let Some(ref error) = stat.error_message {
                    println!("    Error: {}", error);
                }
            }
        }
        println!("=====================================\n");
    }

    /// Get efficiency metrics
    pub fn get_efficiency_metrics(&self) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();

        if self.total_execution_time.as_secs_f64() > 0.0 {
            metrics.insert(
                "changes_per_second".to_string(),
                self.total_changes as f64 / self.total_execution_time.as_secs_f64(),
            );
        }

        if self.total_passes > 0 {
            metrics.insert(
                "avg_execution_time_ms".to_string(),
                self.total_execution_time.as_millis() as f64 / self.total_passes as f64,
            );
        }

        metrics.insert(
            "success_rate".to_string(),
            if self.total_passes > 0 {
                (self.successful_passes as f64 / self.total_passes as f64) * 100.0
            } else {
                0.0
            },
        );

        metrics
    }
}

impl Default for OptimizationStatistics {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for the optimization composer
#[derive(Debug, Clone)]
pub struct ComposerConfig {
    pub show_progress: bool,
    pub show_detailed_progress: bool,
    pub collect_memory_stats: bool,
    pub max_iterations: Option<usize>,
    pub convergence_threshold: u32, // Stop if changes < threshold
}

impl Default for ComposerConfig {
    fn default() -> Self {
        Self {
            show_progress: true,
            show_detailed_progress: false,
            collect_memory_stats: true,
            max_iterations: None,
            convergence_threshold: 0,
        }
    }
}

/// Advanced composer for orchestrating optimization passes with statistics and progress tracking
#[derive(Debug)]
pub struct OptimizationComposer {
    pass_manager: PassManager,
    config: ComposerConfig,
    system: System,
}

impl OptimizationComposer {
    /// Create a new composer with the given pass manager
    pub fn new(pass_manager: PassManager) -> Self {
        let mut system = System::new();
        system.refresh_memory();
        Self {
            pass_manager,
            config: ComposerConfig::default(),
            system,
        }
    }

    /// Create a new composer with custom configuration
    pub fn with_config(pass_manager: PassManager, config: ComposerConfig) -> Self {
        let mut system = System::new();
        system.refresh_memory();
        Self {
            pass_manager,
            config,
            system,
        }
    }

    /// Get current memory usage in bytes
    fn get_memory_usage(&mut self) -> u64 {
        if !self.config.collect_memory_stats {
            return 0;
        }

        self.system.refresh_memory();
        self.system.used_memory()
    }

    /// Get graph size (nodes, edges)
    fn get_graph_size(graph: &Graph) -> (usize, usize) {
        // Note: These methods need to be implemented in Graph
        // For now, returning placeholder values
        (0, 0) // TODO: Replace with actual graph.node_count(), graph.edge_count()
    }

    /// Execute optimization with fixed-point iteration until convergence
    /// This is the main entry point for optimization
    pub fn execute(&mut self, graph: &mut Graph) -> Result<OptimizationStatistics, PassError> {
        self.execute_to_convergence(graph)
    }

    /// Execute optimization with fixed-point iteration
    /// Uses the ONNX Optimizer approach: execute all passes sequentially, check for changes, repeat until convergence
    pub fn execute_to_convergence(
        &mut self,
        graph: &mut Graph,
    ) -> Result<OptimizationStatistics, PassError> {
        let mut stats = OptimizationStatistics::new();
        let start_time = Instant::now();
        let mut iteration = 0;
        let mut total_changes = 0;

        // Initialize progress tracking
        let progress_bar = if self.config.show_progress {
            let pb = ProgressBar::new_spinner();
            pb.set_style(
                ProgressStyle::default_spinner()
                    .template("{spinner:.green} [{elapsed_precise}] {msg} | ETA: {eta_precise}")
                    .unwrap_or_else(|_| ProgressStyle::default_spinner()),
            );
            pb.enable_steady_tick(Duration::from_millis(100));
            Some(pb)
        } else {
            None
        };

        // Collect initial statistics
        stats.initial_graph_size = Self::get_graph_size(graph);
        let initial_memory = self.get_memory_usage();
        stats.memory_peak = initial_memory;

        if let Some(ref pb) = progress_bar {
            pb.set_message("Starting fixed-point optimization...");
        }

        // Fixed-point iteration: repeat all passes until no changes occur
        loop {
            iteration += 1;

            if let Some(ref pb) = progress_bar {
                pb.set_message(format!(
                    "Iteration {} | Total changes: {} | Mem: {:.1}MB",
                    iteration,
                    total_changes,
                    self.get_memory_usage() as f64 / 1024.0 / 1024.0
                ));
            }

            let iteration_start = Instant::now();
            let memory_before = self.get_memory_usage();
            let graph_size_before = Self::get_graph_size(graph);

            // Execute all passes in sequence (following ONNX Optimizer approach)
            let iteration_changes = self.execute_pass_sequence(graph, iteration, &mut stats)?;

            let iteration_duration = iteration_start.elapsed();
            let memory_after = self.get_memory_usage();
            let graph_size_after = Self::get_graph_size(graph);

            // Record iteration statistics
            let iteration_stats = PassStatistics {
                pass_name: format!("Iteration-{}", iteration),
                execution_time: iteration_duration,
                changes_made: iteration_changes,
                memory_before,
                memory_after,
                nodes_before: graph_size_before.0,
                nodes_after: graph_size_after.0,
                edges_before: graph_size_before.1,
                edges_after: graph_size_after.1,
                success: true,
                error_message: None,
            };

            stats.pass_statistics.push(iteration_stats);
            total_changes += iteration_changes;

            // Update memory peak
            if memory_after > stats.memory_peak {
                stats.memory_peak = memory_after;
            }

            // Check convergence (no changes in this iteration)
            if iteration_changes <= self.config.convergence_threshold {
                if let Some(ref pb) = progress_bar {
                    pb.finish_with_message(format!(
                        "✅ Converged after {} iterations | Total changes: {}",
                        iteration, total_changes
                    ));
                }
                break;
            }

            // Check max iterations limit
            if let Some(max_iter) = self.config.max_iterations {
                if iteration >= max_iter {
                    if let Some(ref pb) = progress_bar {
                        pb.finish_with_message(format!(
                            "⏱️ Stopped after {} iterations (max limit) | Total changes: {}",
                            iteration, total_changes
                        ));
                    }
                    break;
                }
            }

            // Provide detailed progress if enabled
            if self.config.show_detailed_progress {
                println!(
                    "    Iteration {}: {} changes in {:.2?}",
                    iteration, iteration_changes, iteration_duration
                );
            }
        }

        // Finalize overall statistics
        stats.total_execution_time = start_time.elapsed();
        stats.total_changes = total_changes;
        stats.total_passes = iteration;
        stats.successful_passes = iteration; // All iterations succeeded if we reach here
        stats.failed_passes = 0;
        stats.final_graph_size = Self::get_graph_size(graph);

        Ok(stats)
    }

    /// Execute all passes in sequence for one iteration
    /// This follows the ONNX Optimizer approach of running all passes together
    fn execute_pass_sequence(
        &mut self,
        graph: &mut Graph,
        iteration: usize,
        stats: &mut OptimizationStatistics,
    ) -> Result<u32, PassError> {
        // Delegate to PassManager for now, but this could be expanded to track individual passes
        let changes = self.pass_manager.execute_all(graph).map_err(|e| {
            // Record failed iteration
            let failed_stats = PassStatistics {
                pass_name: format!("Iteration-{}-FAILED", iteration),
                execution_time: Duration::ZERO,
                changes_made: 0,
                memory_before: self.get_memory_usage(),
                memory_after: self.get_memory_usage(),
                nodes_before: 0,
                nodes_after: 0,
                edges_before: 0,
                edges_after: 0,
                success: false,
                error_message: Some(e.to_string()),
            };
            stats.pass_statistics.push(failed_stats);
            stats.failed_passes += 1;

            PassError::ExecutionFailed {
                message: format!("Iteration {} failed: {}", iteration, e),
            }
        })?;

        Ok(changes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::passes::{
        eliminations::EliminateIdentity,
        manager::{Pass, PassManager},
    };

    #[test]
    fn test_composer_creation() {
        let manager = PassManager::new();
        let composer = OptimizationComposer::new(manager);
        assert!(!composer.config.show_detailed_progress);
    }

    #[test]
    fn test_composer_with_config() {
        let manager = PassManager::new();
        let config = ComposerConfig {
            show_progress: false,
            show_detailed_progress: true,
            collect_memory_stats: false,
            max_iterations: Some(10),
            convergence_threshold: 5,
        };

        let composer = OptimizationComposer::with_config(manager, config.clone());
        assert_eq!(composer.config.max_iterations, Some(10));
        assert_eq!(composer.config.convergence_threshold, 5);
        assert!(!composer.config.show_progress);
    }

    #[test]
    fn test_statistics_creation() {
        let stats = OptimizationStatistics::new();
        assert_eq!(stats.total_changes, 0);
        assert_eq!(stats.total_passes, 0);
        assert_eq!(stats.successful_passes, 0);
    }

    #[test]
    fn test_execute_basic() {
        let manager =
            PassManager::new().add_pass(Pass::EliminateIdentity(EliminateIdentity::new()));

        let mut composer = OptimizationComposer::new(manager);
        let mut graph = Graph::new();

        let result = composer.execute(&mut graph);
        assert!(result.is_ok());

        let stats = result.unwrap();
        assert!(stats.total_passes >= 1); // At least one iteration
        assert_eq!(stats.successful_passes, stats.total_passes); // All should succeed
        assert_eq!(stats.failed_passes, 0);
    }

    #[test]
    fn test_convergence_with_no_changes() {
        let manager = PassManager::new().add_pass(Pass::Placeholder); // Placeholder pass makes no changes

        let config = ComposerConfig {
            show_progress: false,
            show_detailed_progress: false,
            collect_memory_stats: false,
            max_iterations: Some(5),
            convergence_threshold: 0,
        };

        let mut composer = OptimizationComposer::with_config(manager, config);
        let mut graph = Graph::new();

        let result = composer.execute(&mut graph);
        assert!(result.is_ok());

        let stats = result.unwrap();
        assert_eq!(stats.total_passes, 1); // Should converge immediately
        assert_eq!(stats.total_changes, 0); // No changes expected
    }

    #[test]
    fn test_max_iterations_limit() {
        let manager =
            PassManager::new().add_pass(Pass::EliminateIdentity(EliminateIdentity::new()));

        let config = ComposerConfig {
            show_progress: false,
            show_detailed_progress: false,
            collect_memory_stats: false,
            max_iterations: Some(3), // Limit to 3 iterations
            convergence_threshold: 0,
        };

        let mut composer = OptimizationComposer::with_config(manager, config);
        let mut graph = Graph::new();

        let result = composer.execute(&mut graph);
        assert!(result.is_ok());

        let stats = result.unwrap();
        assert!(stats.total_passes <= 3); // Should not exceed max iterations
    }
}
