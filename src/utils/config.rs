/// Configuration for optimization behavior
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    /// Maximum number of optimization passes to run
    pub max_passes: u32,
    /// Whether to cache topological order between passes
    pub cache_topology: bool,
    /// Minimum graph size to enable certain optimizations
    pub min_graph_size_for_advanced_opts: usize,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            max_passes: 10,
            cache_topology: true,
            min_graph_size_for_advanced_opts: 100,
        }
    }
}

/// Builder pattern for optimization configuration
#[derive(Debug, Clone)]
pub struct OptimizationConfigBuilder {
    max_passes: u32,
    cache_topology: bool,
    min_graph_size_for_advanced_opts: usize,
    enabled_passes: Vec<String>,
    disabled_passes: Vec<String>,
}

impl Default for OptimizationConfigBuilder {
    fn default() -> Self {
        Self {
            max_passes: 10,
            cache_topology: true,
            min_graph_size_for_advanced_opts: 100,
            enabled_passes: Vec::new(),
            disabled_passes: Vec::new(),
        }
    }
}

impl OptimizationConfigBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum number of optimization passes
    pub fn max_passes(mut self, max_passes: u32) -> Self {
        self.max_passes = max_passes;
        self
    }

    /// Enable topology caching
    pub fn cache_topology(mut self, cache: bool) -> Self {
        self.cache_topology = cache;
        self
    }

    /// Set minimum graph size for advanced optimizations
    pub fn min_graph_size_for_advanced_opts(mut self, size: usize) -> Self {
        self.min_graph_size_for_advanced_opts = size;
        self
    }

    /// Enable specific optimization passes
    pub fn enable_passes(mut self, passes: &[&str]) -> Self {
        self.enabled_passes.extend(passes.iter().map(|s| s.to_string()));
        self
    }

    /// Disable specific optimization passes
    pub fn disable_passes(mut self, passes: &[&str]) -> Self {
        self.disabled_passes.extend(passes.iter().map(|s| s.to_string()));
        self
    }

    /// Build the final configuration
    pub fn build(self) -> OptimizationConfig {
        OptimizationConfig {
            max_passes: self.max_passes,
            cache_topology: self.cache_topology,
            min_graph_size_for_advanced_opts: self.min_graph_size_for_advanced_opts,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_builder_defaults() {
        let config = OptimizationConfigBuilder::new().build();
        assert_eq!(config.max_passes, 10);
        assert!(config.cache_topology);
    }

    #[test]
    fn test_config_builder_custom() {
        let config = OptimizationConfigBuilder::new()
            .max_passes(5)
            .cache_topology(false)
            .min_graph_size_for_advanced_opts(50)
            .build();

        assert_eq!(config.max_passes, 5);
        assert!(!config.cache_topology);
        assert_eq!(config.min_graph_size_for_advanced_opts, 50);
    }
}
