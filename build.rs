fn main() -> Result<(), Box<dyn std::error::Error>> {
    let proto_file = "third_party/onnx.proto";

    // Re-run if the proto changes
    println!("cargo:rerun-if-changed={}", proto_file);

    // Use system `protoc` found in PATH
    let mut config = prost_build::Config::new();
    config.compile_protos(&[proto_file], &["third_party"])?;

    Ok(())
}
