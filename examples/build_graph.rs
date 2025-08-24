use hf_hub::api::sync::Api;
use hf_hub::{Repo, RepoType};
use onnx_optimizer_rs::{Graph, GraphView, load_model};
use std::fs;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Downloading ONNX model from Hugging Face...");

    // Create an instance of the Hugging Face Hub API with no cache
    let api = Api::new()?;

    // Specify the model repository (SBERT tiny ONNX model)
    let model_id = "Qdrant/all-MiniLM-L6-v2-onnx".to_string();
    let repo = Repo::with_revision(model_id, RepoType::Model, "main".to_string());

    // Get a reference to the repository for downloading
    let api_repo = api.repo(repo);

    // Download the ONNX model file
    println!("Downloading model.onnx...");
    let downloaded_file = api_repo.get("model.onnx")?;

    // Create examples directory if it doesn't exist
    let examples_dir = Path::new("examples");
    if !examples_dir.exists() {
        fs::create_dir_all(examples_dir)?;
    }

    // Copy the downloaded file to examples directory
    let local_model_path = examples_dir.join("all-MiniLM-L6-v2.onnx");
    fs::copy(&downloaded_file, &local_model_path)?;

    println!("Model saved to: {}", local_model_path.display());

    // Load the ONNX model using our io utilities
    println!("Loading ONNX model...");
    let model_proto = load_model(&local_model_path)?;

    println!("Model loaded successfully!");
    println!("   IR Version: {:?}", model_proto.ir_version);
    println!("   Producer: {:?}", model_proto.producer_name);
    println!("   Model Version: {:?}", model_proto.model_version);

    if let Some(graph_proto) = &model_proto.graph {
        println!("   Graph Name: {:?}", graph_proto.name);
        println!("   Nodes: {} operators", graph_proto.node.len());
        println!("   Inputs: {} tensors", graph_proto.input.len());
        println!("   Outputs: {} tensors", graph_proto.output.len());
        println!(
            "   Initializers: {} constant tensors",
            graph_proto.initializer.len()
        );
    }

    // Build the graph using our Graph structure
    println!("Building internal graph representation...");
    let graph = Graph::from_model_proto(&model_proto)?;

    println!("Graph built successfully!");
    println!("   Internal nodes: {}", graph.node_count());
    println!("   Internal values: {}", graph.value_count());
    println!("   Graph inputs: {}", graph.graph_inputs().len());
    println!("   Graph outputs: {}", graph.graph_outputs().len());

    // Print some details about the graph structure
    println!("\nGraph Analysis:");

    // Show graph inputs
    if !graph.graph_inputs().is_empty() {
        println!("   Input tensors:");
        for &input_id in graph.graph_inputs() {
            if let Some(tensor) = graph.tensor(input_id) {
                println!(
                    "     - {}: {:?} {:?}",
                    tensor.name.as_deref().unwrap_or("unnamed"),
                    tensor.dtype,
                    tensor
                        .shape
                        .as_ref()
                        .map(|s| format!("{:?}", s))
                        .unwrap_or("unknown shape".to_string())
                );
            }
        }
    }

    // Show graph outputs
    if !graph.graph_outputs().is_empty() {
        println!("   Output tensors:");
        for &output_id in graph.graph_outputs() {
            if let Some(tensor) = graph.tensor(output_id) {
                println!(
                    "     - {}: {:?} {:?}",
                    tensor.name.as_deref().unwrap_or("unnamed"),
                    tensor.dtype,
                    tensor
                        .shape
                        .as_ref()
                        .map(|s| format!("{:?}", s))
                        .unwrap_or("unknown shape".to_string())
                );
            }
        }
    }

    // Show some node types
    let mut op_counts = std::collections::HashMap::new();
    for node_id in graph.node_indices() {
        if let Some(node) = graph.node(node_id) {
            *op_counts.entry(node.op_kind.as_onnx_str()).or_insert(0) += 1;
        }
    }

    println!("   Operation types:");
    for (op_type, count) in op_counts {
        println!("     - {}: {} nodes", op_type, count);
    }

    println!("\nExample completed successfully!");
    println!("   Downloaded model: {}", local_model_path.display());

    Ok(())
}
