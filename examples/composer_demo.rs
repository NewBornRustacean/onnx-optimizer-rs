use std::{fs, path::Path};

use hf_hub::{Repo, RepoType, api::sync::Api};
use onnx_optimizer_rs::{
    ComposerConfig, Graph, GraphView, OptimizationComposer, load_model, pass_manger_with_passes,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 ONNX Optimizer Composer Demo");
    println!("   Downloading model...");
    // Check if model exists, download if not
    let model_path = Path::new("examples/resnet50_unoptimized/model.onnx");

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
    let pass_manager =
        pass_manger_with_passes![EliminateIdentity, EliminateNopConcat, EliminateNopTranspose];

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
