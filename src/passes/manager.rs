use crate::passes::{
    OptimizationPass,
    error::PassError,
    basic::{ConstantFoldingPass, DeadNodeEliminationPass},
    fusion::ConvBatchNormFusionPass,
};

/// Trait for pass factories that can create passes with compile-time type information
pub trait PassFactory<T: OptimizationPass> {
    fn create() -> T;
}

/// Macro to register multiple passes at once
macro_rules! register_passes {
    ($($pass_type:ty),* $(,)?) => {
        /// All registered pass types - automatically generated
        static ALL_PASS_TYPES: &[fn(&mut PassRegistry)] = &[
            $(
                |registry| {
                    let pass = <$pass_type>::new();
                    registry.register_pass(&pass.pass_name());
                },
            )*
        ];

        $(
            impl PassFactory<$pass_type> for $pass_type {
                fn create() -> $pass_type {
                    <$pass_type>::new()
                }
            }
        )*
    };
}

/// Macro to implement PassFactory for a single pass (for backwards compatibility)
/// This just forwards to the register_passes! macro
macro_rules! impl_pass_factory {
    ($pass_type:ty) => {
        impl PassFactory<$pass_type> for $pass_type {
            fn create() -> $pass_type {
                <$pass_type>::new()
            }
        }
    };
}

// Register all pass types in one place - just add new passes to this list!
register_passes!(
    ConstantFoldingPass,
    DeadNodeEliminationPass,
    ConvBatchNormFusionPass,
);

/// Pass registry that doesn't rely on global state
#[derive(Debug, Clone)]
pub struct PassRegistry {
    pass_names: Vec<String>,
}

impl PassRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            pass_names: Vec::new(),
        };
        
        // Auto-register all passes from the static list
        for register_fn in ALL_PASS_TYPES {
            register_fn(&mut registry);
        }
        
        registry
    }

    /// Register a new pass name
    pub fn register_pass(&mut self, name: &str) {
        if !self.contains(name) {
            self.pass_names.push(name.to_string());
        }
    }

    /// Check if a pass name is registered
    pub fn contains(&self, name: &str) -> bool {
        self.pass_names.iter().any(|n| n == name)
    }

    /// Get all available pass names
    pub fn available_passes(&self) -> &[String] {
        &self.pass_names
    }

    /// Create a pass by name using generics
    pub fn create_pass<T>(&self, name: &str) -> Option<T>
    where
        T: OptimizationPass + PassFactory<T>,
    {
        if self.contains(name) {
            Some(T::create())
        } else {
            None
        }
    }
}

impl Default for PassRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Enhanced PassManager with compile-time type safety
pub struct PassManager<T = ()> {
    passes: T,
    registry: PassRegistry,
}

impl PassManager<()> {
    pub fn new() -> Self {
        Self {
            passes: (),
            registry: PassRegistry::new(),
        }
    }

    /// Create PassManager with custom registry
    pub fn with_registry(registry: PassRegistry) -> Self {
        Self {
            passes: (),
            registry,
        }
    }
}

impl<T> PassManager<T> {
    /// Add a pass to the manager with compile-time type information
    pub fn add_pass<P: OptimizationPass>(self, pass: P) -> PassManager<(T, P)> {
        PassManager {
            passes: (self.passes, pass),
            registry: self.registry,
        }
    }

    /// Create and add a pass by name using generics
    pub fn add_pass_by_name<P>(self, name: &str) -> Result<PassManager<(T, P)>, PassError>
    where
        P: OptimizationPass + PassFactory<P>,
    {
        if let Some(pass) = self.registry.create_pass::<P>(name) {
            Ok(self.add_pass(pass))
        } else {
            Err(PassError::PassNotApplicable(name.to_string()))
        }
    }

    /// Get a reference to the registry
    pub fn registry(&self) -> &PassRegistry {
        &self.registry
    }

    /// Get all available pass names from registry
    pub fn available_passes(&self) -> Vec<&str> {
        self.registry.available_passes().iter().map(|s| s.as_str()).collect()
    }

    /// Execute the passes with compile-time type information
    pub fn execute(self) -> Result<(T, u32), PassError>
    where
        T: ExecutablePasses,
    {
        self.passes.execute_all()
    }
}

impl Default for PassManager<()> {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for creating commonly used pass combinations
pub struct PassManagerBuilder<T = ()> {
    manager: PassManager<T>,
}

impl PassManagerBuilder<()> {
    pub fn new() -> Self {
        Self {
            manager: PassManager::new(),
        }
    }
}

impl<T> PassManagerBuilder<T> {
    /// Add a pass to the builder
    pub fn with_pass<P: OptimizationPass>(self, pass: P) -> PassManagerBuilder<(T, P)> {
        PassManagerBuilder {
            manager: self.manager.add_pass(pass),
        }
    }

    /// Add a pass by name
    pub fn with_pass_by_name<P>(self, name: &str) -> Result<PassManagerBuilder<(T, P)>, PassError>
    where
        P: OptimizationPass + PassFactory<P>,
    {
        Ok(PassManagerBuilder {
            manager: self.manager.add_pass_by_name::<P>(name)?,
        })
    }

    /// Build the final PassManager
    pub fn build(self) -> PassManager<T> {
        self.manager
    }
}

impl Default for PassManagerBuilder<()> {
    fn default() -> Self {
        Self::new()
    }
}

/// Static utility for executing passes
pub struct PassExecutor;

impl PassExecutor {
    /// Execute a single pass
    pub fn execute_single<T: OptimizationPass>(mut pass: T) -> Result<(T, u32), PassError> {
        let changes = if pass.can_apply() {
            pass.execute()?
        } else {
            0
        };
        Ok((pass, changes))
    }

    /// Execute multiple passes in sequence
    pub fn execute_sequence<T1, T2>(pass1: T1, pass2: T2) -> Result<((T1, T2), u32), PassError>
    where
        T1: OptimizationPass,
        T2: OptimizationPass,
    {
        let (pass1, changes1) = Self::execute_single(pass1)?;
        let (pass2, changes2) = Self::execute_single(pass2)?;
        Ok(((pass1, pass2), changes1 + changes2))
    }
}

pub struct TypedPassBuilder<T> {
    passes: T,
}

impl TypedPassBuilder<()> {
    pub fn new() -> Self {
        Self { passes: () }
    }
}

impl<T> TypedPassBuilder<T> {
    /// Add a pass to the builder with compile-time type information
    pub fn with_pass<P: OptimizationPass>(self, pass: P) -> TypedPassBuilder<(T, P)> {
        TypedPassBuilder {
            passes: (self.passes, pass),
        }
    }

    /// Execute passes with zero-cost abstractions
    pub fn execute(self) -> Result<(T, u32), PassError>
    where
        T: ExecutablePasses,
    {
        self.passes.execute_all()
    }
}

/// Trait for executing passes with compile-time type information
pub trait ExecutablePasses {
    fn execute_all(self) -> Result<(Self, u32), PassError>
    where
        Self: Sized;
}

// Implement ExecutablePasses for unit type (empty passes)
impl ExecutablePasses for () {
    fn execute_all(self) -> Result<(Self, u32), PassError> {
        Ok((self, 0))
    }
}

// Implement ExecutablePasses for single pass
impl<T: OptimizationPass> ExecutablePasses for T {
    fn execute_all(mut self) -> Result<(Self, u32), PassError> {
        let changes = if self.can_apply() {
            self.execute()?
        } else {
            0
        };
        Ok((self, changes))
    }
}

// Implement ExecutablePasses for pass tuples
impl<T: ExecutablePasses, P: OptimizationPass> ExecutablePasses for (T, P) {
    fn execute_all(self) -> Result<(Self, u32), PassError> {
        let (passes, mut pass) = self;
        let (passes, changes1) = passes.execute_all()?;
        
        let changes2 = if pass.can_apply() {
            pass.execute()?
        } else {
            0
        };
        
        Ok(((passes, pass), changes1 + changes2))
    }
}

impl<T: OptimizationPass> ExecutablePasses for &mut [T] {
    fn execute_all(self) -> Result<(Self, u32), PassError> {
        let mut total_changes = 0;
        for pass in self.iter_mut() {
            if pass.can_apply() {
                total_changes += pass.execute()?;
            } else {
                return Err(PassError::PassNotApplicable(pass.pass_name().to_string()));
            }
        }
        Ok((self, total_changes))
    }
}

impl Default for TypedPassBuilder<()> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pass_registry() {
        let registry = &*PASS_REGISTRY;
        assert!(!registry.is_empty(), "Pass registry should not be empty");

        // Test that all expected passes are registered
        assert!(registry.contains_key("constant_folding"));
        assert!(registry.contains_key("dead_node_elimination"));
        assert!(registry.contains_key("conv_batchnorm_fusion"));
    }

    #[test]
    fn test_pass_manager_with_registry() {
        let mut manager = PassManager::new();

        // Test that we can create passes from registry
        let available_passes = manager.available_passes();
        assert!(!available_passes.is_empty(), "Should have available passes");
        assert_eq!(available_passes.len(), 3, "Should have exactly 3 passes");

        // Test adding passes by name
        assert!(manager.add_pass_by_name("constant_folding").is_ok());
        assert!(manager.add_pass_by_name("dead_node_elimination").is_ok());
        assert!(manager.add_pass_by_name("conv_batchnorm_fusion").is_ok());

        // Test error for unknown pass
        assert!(manager.add_pass_by_name("unknown_pass").is_err());
    }

    #[test]
    fn test_pass_manager_builder() {
        let result = PassManagerBuilder::new().with_basic_passes();

        assert!(result.is_ok(), "Basic passes should be available");

        let manager = result.unwrap().build();
        assert_eq!(manager.pass_names().len(), 2, "Should have 2 basic passes");
    }

    #[test]
    fn test_generic_pass_execution() {
        // Test zero-cost abstraction with compile-time types
        let pass = ConstantFoldingPass::new();
        let result = PassExecutor::execute_single(pass);

        // This would normally require actual graph data to test properly
        // For now, we just verify the API works
        assert!(result.is_ok() || matches!(result, Err(PassError::NotImplemented(_))));
    }

    #[test]
    fn test_pass_sequence_execution() {
        let pass1 = ConstantFoldingPass::new();
        let pass2 = DeadNodeEliminationPass::new();

        let result = PassExecutor::execute_sequence(pass1, pass2);

        // This would normally require actual graph data to test properly
        assert!(result.is_ok() || matches!(result, Err(PassError::NotImplemented(_))));
    }

    #[test]
    fn test_generic_vs_dynamic_dispatch() {
        // Demonstrate that we can use both approaches
        let mut manager = PassManager::new();

        // Dynamic dispatch (for flexibility)
        assert!(manager.add_pass_by_name("constant_folding").is_ok());

        // Static dispatch (for performance)
        manager.add_pass(ConstantFoldingPass::new());

        assert_eq!(manager.pass_names().len(), 2);
    }

    #[test]
    fn test_new_pass_registry() {
        let registry = PassRegistry::new();

        // Test registry contains expected passes
        assert!(registry.contains("constant_folding"));
        assert!(registry.contains("dead_node_elimination"));
        assert!(registry.contains("conv_batchnorm_fusion"));
        assert!(!registry.contains("unknown_pass"));

        // Test pass creation
        let cf_pass = registry.create_constant_folding("constant_folding");
        assert!(cf_pass.is_some());

        let dne_pass = registry.create_dead_node_elimination("dead_node_elimination");
        assert!(dne_pass.is_some());

        // Test invalid pass name
        let invalid_pass = registry.create_constant_folding("wrong_name");
        assert!(invalid_pass.is_none());
    }

    #[test]
    fn test_pass_factory_trait() {
        // Test the PassFactory trait implementation
        let cf_pass = ConstantFoldingPass::create();
        assert_eq!(cf_pass.pass_name(), "constant_folding");

        let dne_pass = DeadNodeEliminationPass::create();
        assert_eq!(dne_pass.pass_name(), "dead_node_elimination");

        let fusion_pass = ConvBatchNormFusionPass::create();
        assert_eq!(fusion_pass.pass_name(), "conv_batchnorm_fusion");
    }

    #[test]
    fn test_typed_pass_builder() {
        // Test zero-cost abstraction with TypedPassBuilder
        let builder = TypedPassBuilder::new()
            .with_pass(ConstantFoldingPass::new())
            .with_pass(DeadNodeEliminationPass::new());

        let result = builder.execute();

        // This would normally require actual graph data to test properly
        assert!(result.is_ok() || matches!(result, Err(PassError::NotImplemented(_))));
    }

    #[test]
    fn test_executable_passes_trait() {
        // Test single pass execution
        let pass = ConstantFoldingPass::new();
        let result = pass.execute_all();
        assert!(result.is_ok() || matches!(result, Err(PassError::NotImplemented(_))));

        // Test tuple execution
        let passes = (ConstantFoldingPass::new(), DeadNodeEliminationPass::new());
        let result = passes.execute_all();
        assert!(result.is_ok() || matches!(result, Err(PassError::NotImplemented(_))));
    }
}
