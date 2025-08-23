use crate::{
    graph::Graph,
    passes::{error::PassError, traits::{OptimizationPass, PassCategory}},
};

/// Base pass that provides common functionality for all optimization passes
/// Uses composition pattern
#[derive(Debug, Clone)]
pub struct BasePass {
    pub name: String,
    pub priority: u32,
}

impl BasePass {
    pub fn new(name: &str, priority: u32) -> Self {
        Self {
            name: name.to_string(),
            priority,
        }
    }
}

/// Macro to create a pass struct that composes BasePass
/// This eliminates boilerplate and provides a consistent pattern
macro_rules! define_pass {
    (
        $(#[$meta:meta])*
        $vis:vis struct $name:ident {
            pass_name: $pass_name:literal,
            priority: $priority:literal,
            $($field:ident: $field_type:ty),* $(,)?
        }
    ) => {
        $(#[$meta])*
        #[derive(Debug, Clone)]
        $vis struct $name {
            pub base: BasePass,
            $(pub $field: $field_type,)*
        }

        impl $name {
            pub fn new() -> Self {
                Self {
                    base: BasePass::new($pass_name, $priority),
                    $($field: Default::default(),)*
                }
            }


        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }

    };
}

// Export the macro for use in other modules
pub(crate) use define_pass;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_pass_creation() {
        let base = BasePass::new("test_pass", 5);
        assert_eq!(base.name, "test_pass");
        assert_eq!(base.priority, 5);
    }

    // Test the macro
    define_pass! {
        /// Test pass for demonstrating the macro
        pub struct TestPass {
            pass_name: "test_pass",
            priority: 1,
            counter: u32,
        }
    }

    impl OptimizationPass for TestPass {
        fn execute(&self, _graph: &mut Graph) -> Result<u32, PassError> {
            // Note: TestPass cannot modify self.counter in new stateless design
            // This is intentional - passes should be stateless
            Ok(1)
        }

        fn pass_name(&self) -> String {
            self.base.name.clone()
        }

        fn category(&self) -> PassCategory {
            PassCategory::Other
        }
    }

    #[test]
    fn test_define_pass_macro() {
        let pass = TestPass::new();
        assert_eq!(pass.pass_name(), "test_pass");
        assert_eq!(pass.counter, 0);

        // Create a dummy graph for testing
        let mut graph = Graph::new();
        let result = pass.execute(&mut graph);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 1);

        // counter remains 0 because passes are now stateless
        assert_eq!(pass.counter, 0);
    }
}
