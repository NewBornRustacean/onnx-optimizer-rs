use crate::passes::{
    OptimizationPass,
    basic::{ConstantFoldingPass, DeadNodeEliminationPass},
    error::PassError,
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

// Register all pass types in one place - just add new passes to this list.
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

#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
pub struct PassExecutor;

impl PassExecutor {
    /// Execute a single pass
    pub fn execute_single<T: OptimizationPass>(mut pass: T) -> Result<(T, u32), PassError> {
        let changes = if pass.can_apply() { pass.execute()? } else { 0 };
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

#[derive(Debug, Clone)]
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
        let changes = if self.can_apply() { self.execute()? } else { 0 };
        Ok((self, changes))
    }
}

// Implement ExecutablePasses for pass tuples
impl<T: ExecutablePasses, P: OptimizationPass> ExecutablePasses for (T, P) {
    fn execute_all(self) -> Result<(Self, u32), PassError> {
        let (passes, mut pass) = self;
        let (passes, changes1) = passes.execute_all()?;

        let changes2 = if pass.can_apply() { pass.execute()? } else { 0 };

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

    // Mock optimization pass for testing
    #[derive(Debug, Clone)]
    struct MockPass {
        name: String,
        should_apply: bool,
        execution_result: Result<u32, PassError>,
        priority: u32,
    }

    impl MockPass {
        fn new() -> Self {
            Self {
                name: "mock_pass".to_string(),
                should_apply: true,
                execution_result: Ok(1), // Default: successful execution with 1 change
                priority: 1,
            }
        }

        fn new_with_name(name: &str) -> Self {
            Self {
                name: name.to_string(),
                should_apply: true,
                execution_result: Ok(1), // Default: successful execution with 1 change
                priority: 1,
            }
        }

        fn with_apply(mut self, should_apply: bool) -> Self {
            self.should_apply = should_apply;
            self
        }

        fn with_result(mut self, result: Result<u32, PassError>) -> Self {
            self.execution_result = result;
            self
        }

        fn with_priority(mut self, priority: u32) -> Self {
            self.priority = priority;
            self
        }
    }

    impl OptimizationPass for MockPass {
        fn pass_name(&self) -> String {
            self.name.clone()
        }

        fn execute(&mut self) -> Result<u32, PassError> {
            self.execution_result.clone()
        }

        fn can_apply(&self) -> bool {
            self.should_apply
        }

        fn priority(&self) -> u32 {
            self.priority
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
        assert_eq!(registry.available_passes().len(), 0);

        // Test adding passes
        registry.register_pass("test_pass");
        assert!(registry.contains("test_pass"));
        assert_eq!(registry.available_passes().len(), 1);

        // Test duplicate prevention
        registry.register_pass("test_pass");
        assert_eq!(registry.available_passes().len(), 1); // Should not duplicate

        // Test multiple passes
        registry.register_pass("another_pass");
        assert!(registry.contains("another_pass"));
        assert_eq!(registry.available_passes().len(), 2);
    }

    #[test]
    fn test_pass_manager_with_mock_registry() {
        let registry = create_mock_registry();
        let manager = PassManager::with_registry(registry);

        let available_passes = manager.available_passes();
        assert_eq!(available_passes.len(), 3);
        assert!(available_passes.contains(&"mock_pass_1"));
        assert!(available_passes.contains(&"mock_pass_2"));
        assert!(available_passes.contains(&"constant_folding"));
    }

    #[test]
    fn test_pass_manager_add_pass_directly() {
        let manager = PassManager::new()
            .add_pass(MockPass::new_with_name("test_pass_1"))
            .add_pass(MockPass::new_with_name("test_pass_2"));

        // Type system ensures this compiles - the passes are embedded in the type
        // Type: PassManager<(((), MockPass), MockPass)>
        let _available = manager.available_passes();
    }

    #[test]
    fn test_pass_manager_builder_with_mocks() {
        let builder = PassManagerBuilder::new()
            .with_pass(MockPass::new_with_name("first_pass"))
            .with_pass(MockPass::new_with_name("second_pass"));

        let _manager = builder.build();
        // If this compiles, the builder pattern works correctly
    }

    #[test]
    fn test_pass_executor_with_successful_mock() {
        let pass = MockPass::new_with_name("success_pass").with_result(Ok(5));

        let result = PassExecutor::execute_single(pass);
        assert!(result.is_ok());

        let (returned_pass, changes) = result.unwrap();
        assert_eq!(returned_pass.pass_name(), "success_pass");
        assert_eq!(changes, 5);
    }

    #[test]
    fn test_pass_executor_with_failing_mock() {
        let pass = MockPass::new_with_name("fail_pass")
            .with_result(Err(PassError::NotImplemented("test error".to_string())));

        let result = PassExecutor::execute_single(pass);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), PassError::NotImplemented(_)));
    }

    #[test]
    fn test_pass_executor_with_non_applicable_pass() {
        let pass = MockPass::new_with_name("non_applicable").with_apply(false);

        let result = PassExecutor::execute_single(pass);
        assert!(result.is_ok());

        let (returned_pass, changes) = result.unwrap();
        assert_eq!(returned_pass.pass_name(), "non_applicable");
        assert_eq!(changes, 0); // No changes because can_apply() returned false
    }

    #[test]
    fn test_pass_executor_sequence_with_mocks() {
        let pass1 = MockPass::new_with_name("first").with_result(Ok(2));
        let pass2 = MockPass::new_with_name("second").with_result(Ok(3));

        let result = PassExecutor::execute_sequence(pass1, pass2);
        assert!(result.is_ok());

        let ((p1, p2), total_changes) = result.unwrap();
        assert_eq!(p1.pass_name(), "first");
        assert_eq!(p2.pass_name(), "second");
        assert_eq!(total_changes, 5); // 2 + 3
    }

    #[test]
    fn test_executable_passes_unit() {
        let unit = ();
        let result = unit.execute_all();
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ((), 0));
    }

    #[test]
    fn test_executable_passes_single_mock() {
        let pass = MockPass::new_with_name("single_test").with_result(Ok(7));
        let result = pass.execute_all();

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

        let result = passes.execute_all();
        assert!(result.is_ok());

        let ((p1, p2), total_changes) = result.unwrap();
        assert_eq!(p1.pass_name(), "first");
        assert_eq!(p2.pass_name(), "second");
        assert_eq!(total_changes, 7); // 3 + 4
    }

    #[test]
    fn test_typed_pass_builder_with_mocks() {
        let builder = TypedPassBuilder::new()
            .with_pass(MockPass::new_with_name("builder_test_1").with_result(Ok(2)))
            .with_pass(MockPass::new_with_name("builder_test_2").with_result(Ok(3)));

        let result = builder.execute();
        assert!(result.is_ok());

        let (_, total_changes) = result.unwrap();
        assert_eq!(total_changes, 5); // 2 + 3
    }

    #[test]
    fn test_pass_execution_order_with_mocks() {
        // Test that passes execute in the correct order
        let passes = (
            MockPass::new_with_name("first_pass").with_result(Ok(1)),
            MockPass::new_with_name("second_pass").with_result(Ok(2)),
        );

        let result = passes.execute_all();
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

        let result = error_pass.execute_all();
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
            MockPass::new_with_name("non_applicable").with_apply(false).with_result(Ok(10)),
        );

        let result = passes.execute_all();
        assert!(result.is_ok());

        let (_, total_changes) = result.unwrap();
        assert_eq!(total_changes, 5); // Only the applicable pass contributes
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
        let _passes = manager.available_passes();
    }

    #[test]
    fn test_macro_and_pass_manager_integration() {
        // Test that macro-generated PassFactory works with PassManager
        let manager = PassManager::new();

        // MockPass는 register_passes! 매크로로 등록되지 않았으므로
        // registry에는 없지만 create() 메서드는 동작해야 함
        let result = manager.add_pass_by_name::<MockPass>("mock_pass");
        assert!(result.is_err()); // Registry에 없으므로 실패해야 함

        // 하지만 직접 생성은 가능해야 함 (macro-generated factory)
        let mock_pass = MockPass::create();
        assert_eq!(mock_pass.pass_name(), "mock_pass");

        // Test with only mock passes - no dependency on real passes
        let mock_manager = PassManager::new()
            .add_pass(MockPass::create()) // Macro-generated factory
            .add_pass(MockPass::new_with_name("test_pass")); // Direct constructor

        let _available = mock_manager.available_passes();
    }

    #[test]
    fn test_correct_builder_pattern_usage() {
        // Demonstrate the CORRECT way to use PassManager - without clone() and using only mocks
        let registry = create_mock_registry();

        // Method 1: Chain multiple mock passes using builder pattern
        let manager_chain = PassManager::with_registry(registry.clone())
            .add_pass(MockPass::new_with_name("first_mock"))
            .add_pass(MockPass::new_with_name("second_mock"));

        // This creates a manager with type: PassManager<(((), MockPass), MockPass)>
        let _available_in_chain = manager_chain.available_passes();

        // Method 2: Use PassManagerBuilder for more flexible building
        let manager_builder = PassManagerBuilder::new()
            .with_pass(MockPass::new_with_name("builder_mock_1"))
            .with_pass(MockPass::new_with_name("builder_mock_2"))
            .build();

        let _available_in_builder = manager_builder.available_passes();

        // Method 3: Create separate managers for different workflows (this is fine)
        let mock_manager_1 = PassManager::with_registry(registry.clone())
            .add_pass(MockPass::new_with_name("workflow_1"));

        let mock_manager_2 =
            PassManager::with_registry(registry).add_pass(MockPass::new_with_name("workflow_2"));

        let _mock1_available = mock_manager_1.available_passes();
        let _mock2_available = mock_manager_2.available_passes();
    }
}
