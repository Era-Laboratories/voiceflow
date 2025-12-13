//! Build script to generate C header file using cbindgen

fn main() {
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let output_dir = std::path::Path::new(&crate_dir).join("include");

    // Create output directory if it doesn't exist
    std::fs::create_dir_all(&output_dir).ok();

    let config = cbindgen::Config::from_file("cbindgen.toml").unwrap_or_default();

    cbindgen::Builder::new()
        .with_crate(&crate_dir)
        .with_config(config)
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file(output_dir.join("voiceflow.h"));

    println!("cargo:rerun-if-changed=src/lib.rs");
}
