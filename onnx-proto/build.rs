fn main() -> Result<(), Box<dyn std::error::Error>> {
    let proto_file = "third_party/onnx.proto";

    // Re-run if the proto changes
    println!("cargo:rerun-if-changed={}", proto_file);

    // Use system `protoc` found in PATH
    let mut config = prost_build::Config::new();
    
    // Generate code to src/generated.rs for visibility
    let out_dir = std::env::var("OUT_DIR")?;
    config.out_dir(&out_dir);
    
    config.compile_protos(&[proto_file], &["third_party"])?;

    // Copy generated file to src/ for visibility and easier debugging
    copy_to_src()?;

    Ok(())
}

fn copy_to_src() -> Result<(), Box<dyn std::error::Error>> {
    use std::fs;
    use std::path::Path;

    let out_dir = std::env::var("OUT_DIR")?;
    let generated_file = Path::new(&out_dir).join("onnx.rs");
    
    if generated_file.exists() {
        let src_file = Path::new("src/generated.rs");
        fs::copy(&generated_file, &src_file)?;
        
        println!("cargo:warning=Generated proto code copied to src/generated.rs");
    }
    
    Ok(())
}
