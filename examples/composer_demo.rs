use onnx_optimizer_rs::{
    Graph,GraphView, PassManager, Pass, EliminateIdentity,
    OptimizationComposer, ComposerConfig, load_model,
};
use hf_hub::api::sync::Api;
use hf_hub::{Repo, RepoType};
use std::fs;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 ONNX Optimizer Composer Demo");
    println!("   Downloading model...");
    // Check if model exists, download if not
    let model_path = Path::new("examples/all-MiniLM-L6-v2.onnx");
    
    if !model_path.exists() {
        println!("📥 Model not found, downloading from Hugging Face...");
        download_model()?;
        println!("✅ Model downloaded successfully!");
    } else {
        println!("📁 Using existing model: {}", model_path.display());
    }
    
    // Load the ONNX model
    println!("🔄 Loading ONNX model...");
    let model_proto = load_model(model_path)?;
    let mut graph = Graph::from_model_proto(&model_proto)?;
    
    println!("✅ Model loaded successfully!");
    println!("   Nodes: {}", graph.node_count());
    println!("   Values: {}", graph.value_count());
    
    // Show node type distribution for analysis
    let mut op_counts = std::collections::HashMap::new();
    for node_id in graph.node_indices() {
        if let Some(node) = graph.node(node_id) {
            *op_counts.entry(node.op_kind.as_onnx_str()).or_insert(0) += 1;
        }
    }
    
    println!("   Top operation types:");
    let mut sorted_ops: Vec<_> = op_counts.iter().collect();
    sorted_ops.sort_by(|a, b| b.1.cmp(a.1)); // Sort by count descending
    for (op_type, count) in sorted_ops.iter().take(5) {
        println!("     - {}: {} nodes", op_type, count);
    }
    
    // Set up pass manager with optimization passes
    let pass_manager = PassManager::new()
        .add_pass(Pass::EliminateIdentity(EliminateIdentity::new()));
    
    // Configure the composer for detailed statistics and progress tracking
    let config = ComposerConfig {
        show_progress: true,
        show_detailed_progress: true,
        collect_memory_stats: true,
        max_iterations: Some(10),
        convergence_threshold: 0,
    };
    
    println!("\n📊 Running optimization with detailed tracking...");
    
    // Create and configure the composer
    let mut composer = OptimizationComposer::with_config(pass_manager, config);
    
    // Execute optimization with fixed-point iteration until convergence
    match composer.execute(&mut graph) {
        Ok(statistics) => {
            // Print detailed statistics report
            statistics.print_report();
            
            // Show efficiency metrics
            let metrics = statistics.get_efficiency_metrics();
            println!("📈 Efficiency Metrics:");
            for (metric, value) in metrics {
                println!("  {}: {:.2}", metric, value);
            }
        }
        Err(e) => {
            eprintln!("❌ Optimization failed: {}", e);
            return Err(Box::new(e));
        }
    }
    
    println!("\n🔄 Testing with different configuration (no progress bar)...");
    
    // Reload the model for a fresh start
    let model_proto_2 = load_model(model_path)?;
    let mut graph_2 = Graph::from_model_proto(&model_proto_2)?;
    
    // Test with different configuration - no progress bar, detailed output
    let silent_config = ComposerConfig {
        show_progress: false, // No progress bar
        show_detailed_progress: true, // But show iteration details in console
        collect_memory_stats: true,
        max_iterations: Some(5), // Fewer iterations for comparison
        convergence_threshold: 0, // Run until no changes
    };
    
    let pass_manager_2 = PassManager::new()
        .add_pass(Pass::EliminateIdentity(EliminateIdentity::new()));
    
    let mut composer_2 = OptimizationComposer::with_config(pass_manager_2, silent_config);
    
    match composer_2.execute(&mut graph_2) {
        Ok(stats) => {
            println!("✅ Silent optimization completed!");
            println!("  Iterations: {}", stats.total_passes);
            println!("  Total changes: {}", stats.total_changes);
            println!("  Execution time: {:.2?}", stats.total_execution_time);
            println!("  Peak memory: {:.1} MB", stats.memory_peak as f64 / 1024.0 / 1024.0);
            println!("  Success rate: {:.1}%", 
                (stats.successful_passes as f64 / stats.total_passes as f64) * 100.0);
        }
        Err(e) => {
            eprintln!("❌ Silent optimization failed: {}", e);
        }
    }
    
    println!("\n🎉 Demo completed!");
    Ok(())
}

fn download_model() -> Result<(), Box<dyn std::error::Error>> {
    // Create an instance of the Hugging Face Hub API
    let api = Api::new()?;

    // Specify the model repository (SBERT tiny ONNX model)
    let model_id = "Qdrant/all-MiniLM-L6-v2-onnx".to_string();
    let repo = Repo::with_revision(model_id, RepoType::Model, "main".to_string());

    // Get a reference to the repository for downloading
    let api_repo = api.repo(repo);

    // Download the ONNX model file
    println!("   Downloading model.onnx...");
    let downloaded_file = api_repo.get("model.onnx")?;

    // Create examples directory if it doesn't exist
    let examples_dir = Path::new("examples");
    if !examples_dir.exists() {
        fs::create_dir_all(examples_dir)?;
    }

    // Copy the downloaded file to examples directory
    let local_model_path = examples_dir.join("all-MiniLM-L6-v2.onnx");
    fs::copy(&downloaded_file, &local_model_path)?;

    println!("   Model saved to: {}", local_model_path.display());
    Ok(())
}
