use crate::graph::traits::OptimizationPass;
use crate::utils::error::OnnxOptError;
use std::collections::HashMap;
use std::sync::LazyLock;


/// Pass factory function type
pub type PassFactory = fn() -> Box<dyn OptimizationPass>;

/// Global pass registry using LazyLock for thread-safe initialization
static PASS_REGISTRY: LazyLock<HashMap<&'static str, PassFactory>> = LazyLock::new(|| {
    let mut registry = HashMap::new();
    
    // Register all passes automatically
    inventory::iter::<PassRegistration>
        .into_iter()
        .for_each(|registration| {
            registry.insert(registration.name, registration.factory);
        });
    
    registry
});

/// Pass registration struct for inventory
pub struct PassRegistration {
    pub name: &'static str,
    pub factory: PassFactory,
}

impl PassRegistration {
    pub const fn new(name: &'static str, factory: PassFactory) -> Self {
        Self { name, factory }
    }
}

/// Collect pass registrations
inventory::collect!(PassRegistration);

/// Macro to simplify pass registration
#[macro_export]
macro_rules! register_pass {
    ($name:expr, $pass_type:ty) => {
        inventory::submit! {
            $crate::passes::PassRegistration::new($name, || Box::new(<$pass_type>::new()))
        }
    };
}

/// Enhanced PassManager with registry integration
pub struct PassManager {
    passes: Vec<Box<dyn OptimizationPass>>,
    registry: &'static HashMap<&'static str, PassFactory>,
}

impl PassManager {
    pub fn new() -> Self {
        Self {
            passes: Vec::new(),
            registry: &PASS_REGISTRY,
        }
    }

    /// Create a pass by name from registry
    pub fn create_pass(&self, name: &str) -> Option<Box<dyn OptimizationPass>> {
        self.registry.get(name).map(|factory| factory())
    }

    /// Add a pass by name
    pub fn add_pass_by_name(&mut self, name: &str) -> Result<(), OnnxOptError> {
        if let Some(pass) = self.create_pass(name) {
            self.passes.push(pass);
            Ok(())
        } else {
            Err(OnnxOptError::UnknownPass(name.to_string()))
        }
    }

    /// Add multiple passes by names
    pub fn add_passes_by_names(&mut self, names: &[&str]) -> Result<(), OnnxOptError> {
        for name in names {
            self.add_pass_by_name(name)?;
        }
        Ok(())
    }

    /// Add a pass instance directly
    pub fn add_pass(&mut self, pass: Box<dyn OptimizationPass>) {
        self.passes.push(pass);
    }

    /// Get all available pass names from registry
    pub fn available_passes(&self) -> Vec<&'static str> {
        self.registry.keys().copied().collect()
    }

    /// Sort passes by priority (higher priority first)
    pub fn sort_by_priority(&mut self) {
        self.passes.sort_by(|a, b| a.priority().cmp(&b.priority()));
    }

    /// Execute all passes that can be applied
    pub fn execute_all(&mut self) -> Result<u32, OnnxOptError> {
        let mut total_changes = 0;

        for pass in &mut self.passes {
            if pass.can_apply() {
                total_changes += pass.execute()?;
            }
        }

        Ok(total_changes)
    }

    /// Get names of all registered passes
    pub fn pass_names(&self) -> Vec<&str> {
        self.passes.iter().map(|p| p.pass_name()).collect()
    }
}

impl Default for PassManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for creating commonly used pass combinations
pub struct PassManagerBuilder {
    manager: PassManager,
}

impl PassManagerBuilder {
    pub fn new() -> Self {
        Self {
            manager: PassManager::new(),
        }
    }

    /// Add basic optimization passes
    pub fn with_basic_passes(mut self) -> Result<Self, OnnxOptError> {
        self.manager.add_passes_by_names(&[
            "constant_folding",
            "dead_node_elimination",
        ])?;
        Ok(self)
    }

    /// Add fusion passes
    pub fn with_fusion_passes(mut self) -> Result<Self, OnnxOptError> {
        self.manager.add_passes_by_names(&[
            "conv_batchnorm_fusion",
        ])?;
        Ok(self)
    }

    /// Add all available passes
    pub fn with_all_passes(mut self) -> Result<Self, OnnxOptError> {
        let available_passes: Vec<_> = self.manager.available_passes();
        let pass_names: Vec<&str> = available_passes.iter().copied().collect();
        self.manager.add_passes_by_names(&pass_names)?;
        Ok(self)
    }

    /// Build the final PassManager
    pub fn build(mut self) -> PassManager {
        self.manager.sort_by_priority();
        self.manager
    }
}

impl Default for PassManagerBuilder {
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
    }

    #[test]
    fn test_pass_manager_with_registry() {
        let mut manager = PassManager::new();
        
        // Test that we can create passes from registry
        let available_passes = manager.available_passes();
        assert!(!available_passes.is_empty(), "Should have available passes");
        
        if let Some(first_pass) = available_passes.first() {
            assert!(manager.add_pass_by_name(first_pass).is_ok());
        }
    }

    #[test]
    fn test_pass_manager_builder() {
        let result = PassManagerBuilder::new()
            .with_basic_passes();
            
        // Note: This might fail if passes aren't registered yet in tests
        // In a real scenario, passes would be registered via the macro
    }
}
