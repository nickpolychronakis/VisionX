use std::fs;
use std::path::Path;

// Every Python script the app bundles as a resource (tauri.conf.json).
// Kept in ONE place — the copy loop and the rerun-if-changed hints both use it.
const SCRIPTS: [&str; 12] = [
    "vision.py", "report.py", "plate.py", "plate_report.py", "tracking.py",
    "stitch.py", "plate_core.py", "face_shots.py", "cross_match.py",
    "match_report.py", "attributes.py", "prompt_filter.py",
];
const CONFIGS: [&str; 2] = ["config.yaml", "tracker.yaml"];

fn main() {
    // Sync bundled resources from the repo root at EVERY build. Field bug
    // that forced this: the resources/ copies went stale (manually copied
    // once), and a version-bump re-setup then deployed the STALE scripts
    // over the data dir — silently rolling back days of fixes (the live
    // preview "disappeared" on macOS dev). With this, dev builds and CI
    // builds always ship exactly what the repo root contains.
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let root = manifest.join("../..");
    let scripts_dst = manifest.join("resources/scripts");
    let _ = fs::create_dir_all(&scripts_dst);
    for f in SCRIPTS {
        let src = root.join(f);
        if src.exists() {
            let _ = fs::copy(&src, scripts_dst.join(f));
        }
        println!("cargo:rerun-if-changed=../../{}", f);
    }
    let res_dst = manifest.join("resources");
    for f in CONFIGS {
        let src = root.join(f);
        if src.exists() {
            let _ = fs::copy(&src, res_dst.join(f));
        }
        println!("cargo:rerun-if-changed=../../{}", f);
    }

    tauri_build::build()
}
