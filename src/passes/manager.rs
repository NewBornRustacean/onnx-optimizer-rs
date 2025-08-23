use crate::{
    graph::Graph,
    passes::{
        OptimizationPass,
        basic::{ConstantFoldingPass, DeadNodeEliminationPass},
        error::PassError,
        fusion::ConvBatchNormFusionPass,
        traits::PassCategory,
    },
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

// Register all pass types in one place - just add new passes to this list.
register_passes!(
    ConstantFoldingPass,
    DeadNodeEliminationPass,
    ConvBatchNormFusionPass,
);

/// Pass registry that doesn't rely on global state
#[derive(Debug, Clone)]
pub struct PassRegistry {
    pub pass_names: Vec<String>,
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

#[derive(Debug, Clone)]
pub struct PassManager<T = ()> {
    passes: T,
    pub registry: PassRegistry,
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
    pub fn add_pass<P: OptimizationPass>(self, pass: P) -> PassManager<(T, P)> {
        PassManager {
            passes: (self.passes, pass),
            registry: self.registry,
        }
    }

    /// Execute the passes with compile-time type information
    pub fn execute(self, graph: &mut Graph) -> Result<(T, u32), PassError>
    where
        T: ExecutablePasses,
    {
        self.passes.execute_all(graph)
    }
}

impl Default for PassManager<()> {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for executing passes with compile-time type information
pub trait ExecutablePasses {
    fn execute_all(self, graph: &mut Graph) -> Result<(Self, u32), PassError>
    where
        Self: Sized;
}

// Implement ExecutablePasses for unit type (empty passes)
impl ExecutablePasses for () {
    fn execute_all(self, _graph: &mut Graph) -> Result<(Self, u32), PassError> {
        Ok((self, 0))
    }
}

// Implement ExecutablePasses for single pass
impl<T: OptimizationPass> ExecutablePasses for T {
    fn execute_all(self, graph: &mut Graph) -> Result<(Self, u32), PassError> {
        let changes = self.execute(graph)?;
        Ok((self, changes))
    }
}

// Implement ExecutablePasses for pass tuples
impl<T: ExecutablePasses, P: OptimizationPass> ExecutablePasses for (T, P) {
    fn execute_all(self, graph: &mut Graph) -> Result<(Self, u32), PassError> {
        let (passes, pass) = self;
        let (passes, changes1) = passes.execute_all(graph)?;

        let changes2 = pass.execute(graph)?;

        Ok(((passes, pass), changes1 + changes2))
    }
}

impl<T: OptimizationPass> ExecutablePasses for &mut [T] {
    fn execute_all(self, graph: &mut Graph) -> Result<(Self, u32), PassError> {
        let mut total_changes = 0;
        for pass in self.iter() {
            total_changes += pass.execute(graph)?;
        }
        Ok((self, total_changes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock optimization pass for testing using BasePass pattern
    #[derive(Debug, Clone)]
    struct MockPass {
        base: crate::passes::base::BasePass,
        execution_result: Result<u32, PassError>,
    }

    impl MockPass {
        fn new() -> Self {
            Self {
                base: crate::passes::base::BasePass::new("mock_pass", 1),
                execution_result: Ok(1), // Default: successful execution with 1 change
            }
        }

        fn new_with_name(name: &str) -> Self {
            Self {
                base: crate::passes::base::BasePass::new(name, 1),
                execution_result: Ok(1), // Default: successful execution with 1 change
            }
        }

        fn with_result(mut self, result: Result<u32, PassError>) -> Self {
            self.execution_result = result;
            self
        }

        fn with_priority(mut self, priority: u32) -> Self {
            self.base.priority = priority;
            self
        }
    }

    impl Default for MockPass {
        fn default() -> Self {
            Self::new()
        }
    }

    impl OptimizationPass for MockPass {
        fn pass_name(&self) -> String {
            self.base.name.clone()
        }

        fn category(&self) -> PassCategory {
            PassCategory::Other
        }

        fn execute(&self, _graph: &mut Graph) -> Result<u32, PassError> {
            self.execution_result.clone()
        }
    }

    // Mock registry for testing - manually creates registry with test pass names
    // This is DIFFERENT from PassRegistry::new() which uses the production macro
    fn create_mock_registry() -> PassRegistry {
        let mut registry = PassRegistry {
            pass_names: Vec::new(),
        };
        registry.register_pass("mock_pass_1");
        registry.register_pass("mock_pass_2");
        registry.register_pass("constant_folding");
        registry
    }

    // Test passes using the SAME macro as production code
    // This tests the actual macro functionality
    register_passes!(MockPass,);

    #[test]
    fn test_register_macro_functionality() {
        // Test that the macro correctly generates PassFactory for MockPass
        let mock_pass = MockPass::create();
        assert_eq!(mock_pass.pass_name(), "mock_pass"); // Default from PassFactory::create()
    }

    #[test]
    fn test_pass_registry_basic_operations() {
        let mut registry = PassRegistry {
            pass_names: Vec::new(),
        };

        // Test empty registry
        assert!(!registry.contains("test_pass"));
        assert_eq!(registry.pass_names.len(), 0);

        // Test adding passes
        registry.register_pass("test_pass");
        assert!(registry.contains("test_pass"));
        assert_eq!(registry.pass_names.len(), 1);

        // Test duplicate prevention
        registry.register_pass("test_pass");
        assert_eq!(registry.pass_names.len(), 1); // Should not duplicate

        // Test multiple passes
        registry.register_pass("another_pass");
        assert!(registry.contains("another_pass"));
        assert_eq!(registry.pass_names.len(), 2);
    }

    #[test]
    fn test_pass_manager_with_mock_registry() {
        let registry = create_mock_registry();
        let manager = PassManager::with_registry(registry);

        let available_passes = &manager.registry.pass_names;
        assert_eq!(available_passes.len(), 3);
        assert!(available_passes.contains(&"mock_pass_1".to_string()));
        assert!(available_passes.contains(&"mock_pass_2".to_string()));
        assert!(available_passes.contains(&"constant_folding".to_string()));
    }

    #[test]
    fn test_pass_manager_add_pass_directly() {
        let manager = PassManager::new()
            .add_pass(MockPass::new_with_name("test_pass_1"))
            .add_pass(MockPass::new_with_name("test_pass_2"));

        // Type system ensures this compiles - the passes are embedded in the type
        // Type: PassManager<(((), MockPass), MockPass)>
        let _available = &manager.registry.pass_names;
    }

    #[test]
    fn test_pass_manager_as_builder() {
        // Test that PassManager can work as both manager and builder
        let manager = PassManager::new()
            .add_pass(MockPass::new_with_name("first_pass"))
            .add_pass(MockPass::new_with_name("second_pass"));

        // Test execution
        let mut graph = Graph::new();
        let result = manager.execute(&mut graph);
        assert!(result.is_ok());
    }

    #[test]
    fn test_executable_passes_unit() {
        let unit = ();
        let mut graph = Graph::new();
        let result = unit.execute_all(&mut graph);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ((), 0));
    }

    #[test]
    fn test_executable_passes_single_mock() {
        let pass = MockPass::new_with_name("single_test").with_result(Ok(7));
        let mut graph = Graph::new();
        let result = pass.execute_all(&mut graph);

        assert!(result.is_ok());
        let (returned_pass, changes) = result.unwrap();
        assert_eq!(returned_pass.pass_name(), "single_test");
        assert_eq!(changes, 7);
    }

    #[test]
    fn test_executable_passes_tuple_mocks() {
        let passes = (
            MockPass::new_with_name("first").with_result(Ok(3)),
            MockPass::new_with_name("second").with_result(Ok(4)),
        );

        let mut graph = Graph::new();
        let result = passes.execute_all(&mut graph);
        assert!(result.is_ok());

        let ((p1, p2), total_changes) = result.unwrap();
        assert_eq!(p1.pass_name(), "first");
        assert_eq!(p2.pass_name(), "second");
        assert_eq!(total_changes, 7); // 3 + 4
    }

    #[test]
    fn test_pass_execution_order_with_mocks() {
        // Test that passes execute in the correct order
        let passes = (
            MockPass::new_with_name("first_pass").with_result(Ok(1)),
            MockPass::new_with_name("second_pass").with_result(Ok(2)),
        );

        let mut graph = Graph::new();
        let result = passes.execute_all(&mut graph);
        assert!(result.is_ok());

        // Order is guaranteed by type system: first_pass executes before second_pass
        let ((first, second), total) = result.unwrap();
        assert_eq!(first.pass_name(), "first_pass");
        assert_eq!(second.pass_name(), "second_pass");
        assert_eq!(total, 3);
    }

    #[test]
    fn test_pass_error_propagation() {
        let error_pass = MockPass::new_with_name("error_pass")
            .with_result(Err(PassError::PassNotApplicable("test error".to_string())));

        let mut graph = Graph::new();
        let result = error_pass.execute_all(&mut graph);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            PassError::PassNotApplicable(_)
        ));
    }

    #[test]
    fn test_mixed_successful_and_non_applicable_passes() {
        let passes = (
            MockPass::new_with_name("success").with_result(Ok(5)),
            MockPass::new_with_name("non_applicable").with_result(Ok(10)),
        );

        let mut graph = Graph::new();
        let result = passes.execute_all(&mut graph);
        assert!(result.is_ok());

        let (_, total_changes) = result.unwrap();
        assert_eq!(total_changes, 15); // 5 + 10 (can_apply removed from design)
    }

    #[test]
    fn test_pass_manager_type_safety_with_mocks() {
        // Demonstrate type-level guarantees with mock passes
        let manager = PassManager::new()
            .add_pass(MockPass::new_with_name("first"))
            .add_pass(MockPass::new_with_name("second"))
            .add_pass(MockPass::new_with_name("third"));

        // Type: PassManager<((((), MockPass), MockPass), MockPass)>
        // This ensures compile-time type safety
        let _passes = &manager.registry.pass_names;
    }

    #[test]
    fn test_macro_and_pass_manager_integration() {
        // Test that macro-generated PassFactory works with PassManager
        let manager = PassManager::new();

        // MockPass는 register_passes! 매크로로 등록되지 않았으므로
        // registry에는 없지만 create() 메서드는 동작해야 함
        assert!(!manager.registry.pass_names.contains(&"mock_pass".to_string()));

        // 하지만 직접 생성은 가능해야 함 (macro-generated factory)
        let mock_pass = MockPass::create();
        assert_eq!(mock_pass.pass_name(), "mock_pass");

        // Test with only mock passes - no dependency on real passes
        let mock_manager = PassManager::new()
            .add_pass(MockPass::create()) // Macro-generated factory
            .add_pass(MockPass::new_with_name("test_pass")); // Direct constructor

        let _available = &mock_manager.registry.pass_names;
    }

    #[test]
    fn test_unified_pass_manager_usage() {
        // Demonstrate PassManager as unified builder and executor
        let registry = create_mock_registry();

        // Method 1: Chain multiple passes using builder pattern
        let manager = PassManager::with_registry(registry.clone())
            .add_pass(MockPass::new_with_name("first_mock"))
            .add_pass(MockPass::new_with_name("second_mock"));

        let _available = &manager.registry.pass_names;

        // Method 2: Execute directly
        let mut graph = Graph::new();
        let result = PassManager::with_registry(registry)
            .add_pass(MockPass::new_with_name("test_pass").with_result(Ok(3)))
            .execute(&mut graph);

        assert!(result.is_ok());
        let (_, changes) = result.unwrap();
        assert_eq!(changes, 3);
    }
}
