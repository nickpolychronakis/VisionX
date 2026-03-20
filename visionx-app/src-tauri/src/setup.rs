use crate::logging::Logger;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::Command;
use tauri::{AppHandle, Emitter, Manager};

// URLs
const PYTHON_ZIP_URL: &str = "https://www.python.org/ftp/python/3.13.2/python-3.13.2-embed-amd64.zip";
const GET_PIP_URL: &str = "https://bootstrap.pypa.io/get-pip.py";
const MODEL_URL: &str = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26x-seg.pt";
const TORCH_CPU_INDEX: &str = "https://download.pytorch.org/whl/cpu";
const TORCH_CUDA_INDEX: &str = "https://download.pytorch.org/whl/cu124";

#[derive(Clone, Serialize)]
pub struct SetupProgress {
    pub step: String,
    pub step_label: String,
    pub downloaded: u64,
    pub total: u64,
    pub step_index: u32,
    pub total_steps: u32,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct SetupStatus {
    pub needs_setup: bool,
    pub python_ready: bool,
    pub packages_ready: bool,
    pub model_ready: bool,
    pub gpu_detected: bool,
    pub gpu_name: String,
    pub torch_variant: String,
}

#[derive(Clone, Serialize)]
pub struct GpuInfo {
    pub has_nvidia: bool,
    pub gpu_name: String,
    pub driver_version: String,
}

#[derive(Clone, Serialize, Deserialize)]
struct SetupJson {
    python_version: String,
    python_path: String,
    python_source: String, // "system" or "embedded"
    python_ready: bool,
    packages_ready: bool,
    gpu_detected: bool,
    torch_variant: String,
    model_ready: bool,
    model_name: String,
    setup_completed_at: String,
    app_version: String,
}

/// Try to find a working Python 3 on the system
fn find_system_python(logger: &Logger) -> Option<PathBuf> {
    // Try common Python commands
    let candidates = if cfg!(target_os = "windows") {
        vec!["python", "python3", "py"]
    } else {
        vec!["python3", "python"]
    };

    for cmd in candidates {
        logger.info(&format!("Checking for system Python: {}", cmd));
        let result = Command::new(cmd)
            .args(["--version"])
            .output();

        if let Ok(output) = result {
            if output.status.success() {
                let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
                // Verify it's Python 3.10+
                if version.contains("Python 3.1") || version.contains("Python 3.2") {
                    // Get full path
                    let which_cmd = if cfg!(target_os = "windows") { "where" } else { "which" };
                    if let Ok(which_output) = Command::new(which_cmd).arg(cmd).output() {
                        if which_output.status.success() {
                            let path = String::from_utf8_lossy(&which_output.stdout)
                                .lines()
                                .next()
                                .unwrap_or("")
                                .trim()
                                .to_string();
                            if !path.is_empty() {
                                logger.info(&format!("Found system Python: {} ({})", path, version));
                                return Some(PathBuf::from(path));
                            }
                        }
                    }
                    // Fallback: just use the command name
                    logger.info(&format!("Found system Python: {} ({})", cmd, version));
                    return Some(PathBuf::from(cmd));
                } else {
                    logger.info(&format!("Skipping {} — version too old: {}", cmd, version));
                }
            }
        }
    }

    logger.info("No system Python found");
    None
}

/// Check if setup has been completed (also re-triggers on app version change)
pub fn check_setup(data_dir: &Path, current_version: &str) -> SetupStatus {
    let setup_json_path = data_dir.join("setup.json");

    if let Ok(content) = std::fs::read_to_string(&setup_json_path) {
        if let Ok(setup) = serde_json::from_str::<SetupJson>(&content) {
            // If app version changed, force re-setup to install any new deps
            let version_changed = setup.app_version != current_version;
            let base_ready = setup.python_ready && setup.packages_ready && setup.model_ready;

            return SetupStatus {
                needs_setup: !base_ready || version_changed,
                python_ready: setup.python_ready,
                packages_ready: if version_changed { false } else { setup.packages_ready },
                model_ready: setup.model_ready,
                gpu_detected: setup.gpu_detected,
                gpu_name: String::new(),
                torch_variant: setup.torch_variant,
            };
        }
    }

    SetupStatus {
        needs_setup: true,
        python_ready: false,
        packages_ready: false,
        model_ready: false,
        gpu_detected: false,
        gpu_name: String::new(),
        torch_variant: String::new(),
    }
}

/// Detect NVIDIA GPU using nvidia-smi
pub fn detect_gpu(logger: &Logger) -> GpuInfo {
    logger.info("Detecting GPU...");

    let result = Command::new("nvidia-smi")
        .args(["--query-gpu=name,driver_version", "--format=csv,noheader"])
        .output();

    match result {
        Ok(output) if output.status.success() => {
            let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
            let parts: Vec<&str> = stdout.split(", ").collect();
            let gpu_name = parts.first().unwrap_or(&"Unknown").to_string();
            let driver_version = parts.get(1).unwrap_or(&"Unknown").to_string();

            logger.info(&format!("GPU detected: {} (driver {})", gpu_name, driver_version));

            GpuInfo {
                has_nvidia: true,
                gpu_name,
                driver_version,
            }
        }
        Ok(_) => {
            logger.info("nvidia-smi found but returned error — no NVIDIA GPU");
            GpuInfo {
                has_nvidia: false,
                gpu_name: String::new(),
                driver_version: String::new(),
            }
        }
        Err(_) => {
            logger.info("nvidia-smi not found — no NVIDIA GPU, will use CPU");
            GpuInfo {
                has_nvidia: false,
                gpu_name: String::new(),
                driver_version: String::new(),
            }
        }
    }
}

/// Download a file with progress events
async fn download_file(
    url: &str,
    dest: &Path,
    app: &AppHandle,
    logger: &Logger,
    step: &str,
    step_label: &str,
    step_index: u32,
    total_steps: u32,
) -> Result<(), String> {
    logger.info(&format!("Downloading {} from {}", step, url));

    // Ensure parent directory exists
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("Failed to create dir: {}", e))?;
    }

    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()
        .map_err(|e| format!("HTTP client error: {}", e))?;

    let resp = client
        .get(url)
        .send()
        .await
        .map_err(|e| {
            let msg = format!("Download failed ({}): {}", step, e);
            logger.error(&msg);
            msg
        })?;

    if !resp.status().is_success() {
        let msg = format!("Download failed ({}): HTTP {}", step, resp.status());
        logger.error(&msg);
        return Err(msg);
    }

    let total = resp.content_length().unwrap_or(0);
    let mut downloaded: u64 = 0;
    let mut file =
        std::fs::File::create(dest).map_err(|e| format!("Failed to create file: {}", e))?;
    let mut stream = resp.bytes_stream();
    let mut last_emit: u64 = 0;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| {
            let msg = format!("Download interrupted ({}): {}", step, e);
            logger.error(&msg);
            msg
        })?;
        std::io::Write::write_all(&mut file, &chunk)
            .map_err(|e| format!("Write error: {}", e))?;
        downloaded += chunk.len() as u64;

        // Emit progress every 200KB
        if downloaded - last_emit > 200_000 || downloaded == total {
            let _ = app.emit(
                "setup-progress",
                SetupProgress {
                    step: step.to_string(),
                    step_label: step_label.to_string(),
                    downloaded,
                    total,
                    step_index,
                    total_steps,
                },
            );
            last_emit = downloaded;
        }
    }

    logger.info(&format!(
        "Downloaded {} ({} bytes)",
        step,
        downloaded
    ));
    Ok(())
}

/// Extract a zip file
fn extract_zip(zip_path: &Path, dest_dir: &Path, logger: &Logger) -> Result<(), String> {
    logger.info(&format!(
        "Extracting {} to {}",
        zip_path.display(),
        dest_dir.display()
    ));

    let file =
        std::fs::File::open(zip_path).map_err(|e| format!("Failed to open zip: {}", e))?;
    let mut archive =
        zip::ZipArchive::new(file).map_err(|e| format!("Invalid zip file: {}", e))?;

    std::fs::create_dir_all(dest_dir).map_err(|e| format!("Failed to create dir: {}", e))?;

    archive
        .extract(dest_dir)
        .map_err(|e| format!("Extraction failed: {}", e))?;

    // Clean up zip file
    let _ = std::fs::remove_file(zip_path);

    logger.info("Extraction complete");
    Ok(())
}

/// Modify python313._pth to enable site-packages
fn patch_python_pth(python_dir: &Path, logger: &Logger) -> Result<(), String> {
    // Find the ._pth file
    let entries = std::fs::read_dir(python_dir).map_err(|e| e.to_string())?;
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if name.ends_with("._pth") {
            let pth_path = entry.path();
            logger.info(&format!("Patching {}", pth_path.display()));

            let content = std::fs::read_to_string(&pth_path).map_err(|e| e.to_string())?;
            // Uncomment "import site" line and add packages path
            let mut new_content = String::new();
            for line in content.lines() {
                if line.starts_with("#import site") {
                    new_content.push_str("import site\n");
                } else {
                    new_content.push_str(line);
                    new_content.push('\n');
                }
            }
            // Add relative path to packages
            new_content.push_str("..\\packages\n");

            std::fs::write(&pth_path, new_content).map_err(|e| e.to_string())?;
            logger.info("Python path patched for site-packages");
            return Ok(());
        }
    }

    Err("Could not find python._pth file".to_string())
}

/// Run a command and capture output
fn run_command(
    exe: &Path,
    args: &[&str],
    env: Option<(&str, &str)>,
    logger: &Logger,
) -> Result<String, String> {
    logger.info(&format!(
        "Running: {} {}",
        exe.display(),
        args.join(" ")
    ));

    let mut cmd = Command::new(exe);
    cmd.args(args);
    // Fix Unicode encoding issues on Windows (Greek/CJK usernames, etc.)
    cmd.env("PYTHONUTF8", "1");
    cmd.env("PYTHONIOENCODING", "utf-8");

    if let Some((key, val)) = env {
        cmd.env(key, val);
    }

    let output = cmd.output().map_err(|e| {
        let msg = format!("Failed to run {}: {}", exe.display(), e);
        logger.error(&msg);
        msg
    })?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    if !stderr.is_empty() {
        logger.info(&format!("stderr: {}", stderr.trim()));
    }

    if !output.status.success() {
        let msg = format!(
            "Command failed (exit {}): {}",
            output.status.code().unwrap_or(-1),
            stderr
        );
        logger.error(&msg);
        return Err(msg);
    }

    Ok(stdout)
}

/// Run the full setup process
pub async fn run_setup(
    app: &AppHandle,
    data_dir: &Path,
    use_cuda: bool,
    logger: &Logger,
) -> Result<(), String> {
    let total_steps: u32 = 5;
    let packages_dir = data_dir.join("packages");
    let models_dir = data_dir.join("models");
    let scripts_dir = data_dir.join("scripts");

    // Create directories
    for dir in [&packages_dir, &models_dir, &scripts_dir] {
        std::fs::create_dir_all(dir).map_err(|e| format!("Failed to create dir: {}", e))?;
    }

    // === Step 1: Find or download Python ===
    let (python_exe, python_source) = find_or_download_python(app, data_dir, logger, total_steps).await?;
    let python_str = python_exe.to_string_lossy().to_string();
    let use_target = python_source == "embedded"; // system Python: install globally; embedded: use --target

    // === Step 2: Install PyTorch ===
    if !check_python_module(&python_exe, "torch", if use_target { Some(&packages_dir) } else { None }, logger) {
        let index_url = if use_cuda {
            logger.info("Installing PyTorch with CUDA 12.4 support");
            TORCH_CUDA_INDEX
        } else {
            logger.info("Installing PyTorch CPU-only");
            TORCH_CPU_INDEX
        };

        let _ = app.emit(
            "setup-progress",
            SetupProgress {
                step: "pytorch".to_string(),
                step_label: if use_cuda {
                    "PyTorch (CUDA 12.4)".to_string()
                } else {
                    "PyTorch (CPU)".to_string()
                },
                downloaded: 0,
                total: 0,
                step_index: 2,
                total_steps,
            },
        );

        let mut pip_args = vec![
            "-m", "pip", "install", "--index-url", index_url, "torch", "torchvision",
        ];
        let target_str = packages_dir.to_string_lossy().to_string();
        if use_target {
            pip_args.insert(3, "--target");
            pip_args.insert(4, &target_str);
        }

        run_command(
            &python_exe,
            &pip_args,
            if use_target { Some(("PYTHONPATH", packages_dir.to_str().unwrap())) } else { None },
            logger,
        )?;

        logger.info("PyTorch installed successfully");
    } else {
        logger.info("PyTorch already installed, skipping");
    }

    // === Step 3: Install other dependencies ===
    let _ = app.emit(
        "setup-progress",
        SetupProgress {
            step: "deps".to_string(),
            step_label: "AI Libraries".to_string(),
            downloaded: 0,
            total: 0,
            step_index: 3,
            total_steps,
        },
    );

    // Check each required module and collect missing ones
    let mut required_packages = vec![
        ("ultralytics", "ultralytics"),
        ("yaml", "pyyaml"),
        ("tqdm", "tqdm"),
        ("ftfy", "ftfy"),
        ("regex", "regex"),
        ("lap", "lap"),
    ];
    // TensorRT: handled separately below (needs two sub-packages)
    let pkg_dir_opt = if use_target { Some(packages_dir.as_path()) } else { None };
    let target_str = packages_dir.to_string_lossy().to_string();

    let missing: Vec<&str> = required_packages.iter()
        .filter(|(module, _)| !check_python_module(&python_exe, module, pkg_dir_opt, logger))
        .map(|(_, pkg)| *pkg)
        .collect();

    if !missing.is_empty() {
        logger.info(&format!("Installing missing packages: {}", missing.join(", ")));
        let mut pip_args: Vec<&str> = vec!["-m", "pip", "install"];
        if use_target {
            pip_args.push("--target");
            pip_args.push(&target_str);
        }
        pip_args.extend(missing);

        run_command(
            &python_exe,
            &pip_args,
            if use_target { Some(("PYTHONPATH", packages_dir.to_str().unwrap())) } else { None },
            logger,
        )?;
        logger.info("Packages installed successfully");
    } else {
        logger.info("All packages already installed");
    }

    // TensorRT for NVIDIA: install sub-packages from NVIDIA index + create wrapper module
    // tensorrt-cu12-libs wheels are ONLY on NVIDIA's PyPI, not standard PyPI
    if use_cuda && !check_python_module(&python_exe, "tensorrt", pkg_dir_opt, logger) {
        logger.info("Installing TensorRT for GPU acceleration (~2GB download)");
        let mut trt_args: Vec<&str> = vec!["-m", "pip", "install", "--only-binary", ":all:"];
        if use_target {
            trt_args.push("--target");
            trt_args.push(&target_str);
        }
        trt_args.push("--extra-index-url");
        trt_args.push("https://pypi.nvidia.com");
        trt_args.push("tensorrt-cu12-bindings");
        trt_args.push("tensorrt-cu12-libs");

        match run_command(
            &python_exe,
            &trt_args,
            if use_target { Some(("PYTHONPATH", packages_dir.to_str().unwrap())) } else { None },
            logger,
        ) {
            Ok(_) => {
                // Create tensorrt wrapper module (the meta-package normally provides this)
                // tensorrt-cu12-bindings installs as "tensorrt_bindings", we need "tensorrt"
                let trt_dir = packages_dir.join("tensorrt");
                if !trt_dir.exists() {
                    if let Err(e) = std::fs::create_dir_all(&trt_dir) {
                        logger.info(&format!("Failed to create tensorrt wrapper dir: {}", e));
                    } else {
                        let init_content = "from tensorrt_bindings import *\n";
                        if let Err(e) = std::fs::write(trt_dir.join("__init__.py"), init_content) {
                            logger.info(&format!("Failed to write tensorrt __init__.py: {}", e));
                        } else {
                            logger.info("TensorRT installed successfully (with wrapper module)");
                        }
                    }
                } else {
                    logger.info("TensorRT installed successfully");
                }
            }
            Err(e) => {
                // TensorRT is optional — log but don't fail setup
                logger.info(&format!("TensorRT install failed (optional, will use PyTorch): {}", e));
            }
        }
    }

    // CLIP: check separately (zip dependency)
    if !check_python_module(&python_exe, "clip", pkg_dir_opt, logger) {
        logger.info("Installing OpenAI CLIP");
        let mut clip_args: Vec<&str> = vec![
            "-m", "pip", "install",
        ];
        if use_target {
            clip_args.push("--target");
            clip_args.push(&target_str);
        }
        clip_args.push("clip@https://github.com/openai/CLIP/archive/refs/heads/main.zip");

        run_command(
            &python_exe,
            &clip_args,
            if use_target { Some(("PYTHONPATH", packages_dir.to_str().unwrap())) } else { None },
            logger,
        )?;
    }

    // === Step 4: Download default model ===
    let model_path = models_dir.join("yoloe-26x-seg.pt");
    if !model_path.exists() {
        download_file(
            MODEL_URL,
            &model_path,
            app,
            logger,
            "model",
            "AI Model (yoloe-26x-seg)",
            4,
            total_steps,
        )
        .await?;

        // Verify model size
        let size = std::fs::metadata(&model_path)
            .map(|m| m.len())
            .unwrap_or(0);
        if size < 100_000_000 {
            let msg = format!("Model file too small ({} bytes), likely corrupted", size);
            logger.error(&msg);
            let _ = std::fs::remove_file(&model_path);
            return Err(msg);
        }

        logger.info(&format!("Model downloaded ({} MB)", size / 1_000_000));
    } else {
        logger.info("Model already downloaded, skipping");
    }

    // === Step 5: Copy scripts from resources ===
    let _ = app.emit(
        "setup-progress",
        SetupProgress {
            step: "scripts".to_string(),
            step_label: "Finalizing...".to_string(),
            downloaded: 0,
            total: 0,
            step_index: 5,
            total_steps,
        },
    );

    let resource_dir = app
        .path()
        .resource_dir()
        .map_err(|e| format!("Resource dir error: {}", e))?;

    for script in ["vision.py", "report.py"] {
        let src = resource_dir.join("scripts").join(script);
        let dst = scripts_dir.join(script);
        if src.exists() {
            std::fs::copy(&src, &dst)
                .map_err(|e| format!("Failed to copy {}: {}", script, e))?;
            logger.info(&format!("Copied {} to scripts/", script));
        }
    }

    // === Step 6: Verify installation ===
    logger.info("Verifying installation...");
    let verify_result = run_command(
        &python_exe,
        &[
            "-c",
            "import torch; import ultralytics; print(f'torch={torch.__version__} cuda={torch.cuda.is_available()}')",
        ],
        if use_target { Some(("PYTHONPATH", packages_dir.to_str().unwrap())) } else { None },
        logger,
    )?;
    logger.info(&format!("Verification: {}", verify_result.trim()));

    // === Write setup.json ===
    let setup = SetupJson {
        python_version: "3.13".to_string(),
        python_path: python_str,
        python_source: python_source.to_string(),
        python_ready: true,
        packages_ready: true,
        gpu_detected: use_cuda,
        torch_variant: if use_cuda {
            "cu124".to_string()
        } else {
            "cpu".to_string()
        },
        model_ready: true,
        model_name: "yoloe-26x-seg.pt".to_string(),
        setup_completed_at: chrono::Utc::now().to_rfc3339(),
        app_version: app.package_info().version.to_string(),
    };

    let setup_json = serde_json::to_string_pretty(&setup).map_err(|e| e.to_string())?;
    std::fs::write(data_dir.join("setup.json"), setup_json).map_err(|e| e.to_string())?;

    logger.info("Setup completed successfully!");
    Ok(())
}

/// Find system Python or download embedded Python
async fn find_or_download_python(
    app: &AppHandle,
    data_dir: &Path,
    logger: &Logger,
    total_steps: u32,
) -> Result<(PathBuf, &'static str), String> {
    // First check if we already downloaded embedded Python
    let embedded_exe = if cfg!(target_os = "windows") {
        data_dir.join("python").join("python.exe")
    } else {
        data_dir.join("python").join("python3")
    };

    if embedded_exe.exists() {
        logger.info(&format!("Using previously downloaded Python: {}", embedded_exe.display()));
        return Ok((embedded_exe, "embedded"));
    }

    // Check for system Python
    if let Some(system_python) = find_system_python(logger) {
        // Verify pip is available
        let has_pip = Command::new(&system_python)
            .args(["-m", "pip", "--version"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);

        if has_pip {
            logger.info(&format!("Using system Python with pip: {}", system_python.display()));
            return Ok((system_python, "system"));
        } else {
            logger.info("System Python found but pip missing, will download embedded");
        }
    }

    // No system Python — download embedded (Windows only)
    logger.info("Downloading embedded Python...");
    let python_dir = data_dir.join("python");
    let zip_path = data_dir.join("python-embed.zip");

    download_file(
        PYTHON_ZIP_URL,
        &zip_path,
        app,
        logger,
        "python",
        "Python 3.13",
        1,
        total_steps,
    )
    .await?;

    extract_zip(&zip_path, &python_dir, logger)?;
    patch_python_pth(&python_dir, logger)?;

    // Download get-pip.py and install pip
    let packages_dir = data_dir.join("packages");
    std::fs::create_dir_all(&packages_dir).map_err(|e| e.to_string())?;

    let get_pip_path = python_dir.join("get-pip.py");
    download_file(
        GET_PIP_URL,
        &get_pip_path,
        app,
        logger,
        "pip",
        "Python pip",
        1,
        total_steps,
    )
    .await?;

    let python_exe = python_dir.join("python.exe");
    run_command(
        &python_exe,
        &[
            get_pip_path.to_str().unwrap(),
            "--target",
            packages_dir.to_str().unwrap(),
        ],
        Some(("PYTHONPATH", packages_dir.to_str().unwrap())),
        logger,
    )?;

    logger.info("Embedded Python setup complete");
    Ok((python_exe, "embedded"))
}

/// Check if a Python module is importable
fn check_python_module(python: &Path, module: &str, packages_dir: Option<&Path>, logger: &Logger) -> bool {
    let mut cmd = Command::new(python);
    cmd.args(["-c", &format!("import {}", module)]);
    if let Some(pkg_dir) = packages_dir {
        cmd.env("PYTHONPATH", pkg_dir);
    }
    let result = cmd.output().map(|o| o.status.success()).unwrap_or(false);
    logger.info(&format!("Module check '{}': {}", module, if result { "found" } else { "missing" }));
    result
}

/// Get the python executable path from setup.json or fallback
pub fn python_exe_path(data_dir: &Path) -> PathBuf {
    // Try to read from setup.json
    let setup_json_path = data_dir.join("setup.json");
    if let Ok(content) = std::fs::read_to_string(&setup_json_path) {
        if let Ok(setup) = serde_json::from_str::<SetupJson>(&content) {
            if !setup.python_path.is_empty() {
                return PathBuf::from(&setup.python_path);
            }
        }
    }
    // Fallback: embedded Python
    data_dir.join("python").join("python.exe")
}

/// Get the packages directory path (only used for embedded Python)
pub fn packages_dir_path(data_dir: &Path) -> PathBuf {
    data_dir.join("packages")
}

/// Check if we're using system Python (no PYTHONPATH needed)
pub fn is_system_python(data_dir: &Path) -> bool {
    let setup_json_path = data_dir.join("setup.json");
    if let Ok(content) = std::fs::read_to_string(&setup_json_path) {
        if let Ok(setup) = serde_json::from_str::<SetupJson>(&content) {
            return setup.python_source == "system";
        }
    }
    false
}

/// Get the scripts directory path
pub fn scripts_dir_path(data_dir: &Path) -> PathBuf {
    data_dir.join("scripts")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    fn test_logger() -> (tempfile::TempDir, Logger) {
        let dir = tempdir().unwrap();
        let logger = Logger::new(dir.path());
        (dir, logger)
    }

    #[test]
    fn test_check_setup_no_file() {
        let dir = tempdir().unwrap();
        let status = check_setup(dir.path(), "0.4.0");
        assert!(status.needs_setup);
        assert!(!status.python_ready);
        assert!(!status.packages_ready);
        assert!(!status.model_ready);
    }

    #[test]
    fn test_check_setup_complete() {
        let dir = tempdir().unwrap();
        let setup = SetupJson {
            python_version: "3.13".to_string(),
            python_path: "/usr/bin/python3".to_string(),
            python_source: "system".to_string(),
            python_ready: true,
            packages_ready: true,
            gpu_detected: false,
            torch_variant: "cpu".to_string(),
            model_ready: true,
            model_name: "yoloe-26x-seg.pt".to_string(),
            setup_completed_at: "2026-03-18T12:00:00Z".to_string(),
            app_version: "0.4.0".to_string(),
        };
        let json = serde_json::to_string(&setup).unwrap();
        fs::write(dir.path().join("setup.json"), json).unwrap();

        let status = check_setup(dir.path(), "0.4.0");
        assert!(!status.needs_setup);
        assert!(status.python_ready);
        assert!(status.packages_ready);
        assert!(status.model_ready);
    }

    #[test]
    fn test_check_setup_partial() {
        let dir = tempdir().unwrap();
        let setup = SetupJson {
            python_version: "3.13".to_string(),
            python_path: "/usr/bin/python3".to_string(),
            python_source: "system".to_string(),
            python_ready: true,
            packages_ready: true,
            gpu_detected: false,
            torch_variant: "cpu".to_string(),
            model_ready: false, // Model not downloaded yet
            model_name: "".to_string(),
            setup_completed_at: "".to_string(),
            app_version: "0.4.0".to_string(),
        };
        let json = serde_json::to_string(&setup).unwrap();
        fs::write(dir.path().join("setup.json"), json).unwrap();

        let status = check_setup(dir.path(), "0.4.0");
        assert!(status.needs_setup); // Should need setup because model missing
    }

    #[test]
    fn test_detect_gpu_on_macos() {
        let (_dir, logger) = test_logger();
        let gpu = detect_gpu(&logger);
        // On macOS there's no NVIDIA GPU
        if cfg!(target_os = "macos") {
            assert!(!gpu.has_nvidia);
        }
    }

    #[test]
    fn test_find_system_python() {
        let (_dir, logger) = test_logger();
        let python = find_system_python(&logger);
        // On dev machine, python3 should exist
        assert!(python.is_some(), "Expected to find system Python");
    }

    #[test]
    fn test_check_python_module_exists() {
        let (_dir, logger) = test_logger();
        let python = find_system_python(&logger).expect("Need system Python for this test");
        // 'json' is a stdlib module, should always exist
        assert!(check_python_module(&python, "json", None, &logger));
    }

    #[test]
    fn test_check_python_module_missing() {
        let (_dir, logger) = test_logger();
        let python = find_system_python(&logger).expect("Need system Python for this test");
        // This module should never exist
        assert!(!check_python_module(&python, "nonexistent_module_xyz_123", None, &logger));
    }

    #[test]
    fn test_is_system_python() {
        let dir = tempdir().unwrap();
        // No setup.json → false
        assert!(!is_system_python(dir.path()));

        // With system python setup.json → true
        let setup = SetupJson {
            python_version: "3.13".to_string(),
            python_path: "/usr/bin/python3".to_string(),
            python_source: "system".to_string(),
            python_ready: true,
            packages_ready: true,
            gpu_detected: false,
            torch_variant: "cpu".to_string(),
            model_ready: true,
            model_name: "test.pt".to_string(),
            setup_completed_at: "".to_string(),
            app_version: "0.4.0".to_string(),
        };
        fs::write(dir.path().join("setup.json"), serde_json::to_string(&setup).unwrap()).unwrap();
        assert!(is_system_python(dir.path()));
    }

    #[test]
    fn test_is_embedded_python() {
        let dir = tempdir().unwrap();
        let setup = SetupJson {
            python_version: "3.13".to_string(),
            python_path: "C:\\Users\\test\\AppData\\python\\python.exe".to_string(),
            python_source: "embedded".to_string(),
            python_ready: true,
            packages_ready: true,
            gpu_detected: false,
            torch_variant: "cpu".to_string(),
            model_ready: true,
            model_name: "test.pt".to_string(),
            setup_completed_at: "".to_string(),
            app_version: "0.4.0".to_string(),
        };
        fs::write(dir.path().join("setup.json"), serde_json::to_string(&setup).unwrap()).unwrap();
        assert!(!is_system_python(dir.path()));
    }

    #[test]
    fn test_python_exe_path_from_setup_json() {
        let dir = tempdir().unwrap();
        let setup = SetupJson {
            python_version: "3.13".to_string(),
            python_path: "/usr/local/bin/python3".to_string(),
            python_source: "system".to_string(),
            python_ready: true,
            packages_ready: true,
            gpu_detected: false,
            torch_variant: "cpu".to_string(),
            model_ready: true,
            model_name: "test.pt".to_string(),
            setup_completed_at: "".to_string(),
            app_version: "0.4.0".to_string(),
        };
        fs::write(dir.path().join("setup.json"), serde_json::to_string(&setup).unwrap()).unwrap();

        let path = python_exe_path(dir.path());
        assert_eq!(path, PathBuf::from("/usr/local/bin/python3"));
    }

    #[test]
    fn test_python_exe_path_fallback() {
        let dir = tempdir().unwrap();
        // No setup.json → fallback to embedded
        let path = python_exe_path(dir.path());
        assert!(path.to_string_lossy().contains("python"));
    }

    #[test]
    fn test_directory_helpers() {
        let dir = tempdir().unwrap();
        assert_eq!(packages_dir_path(dir.path()), dir.path().join("packages"));
        assert_eq!(scripts_dir_path(dir.path()), dir.path().join("scripts"));
    }
}
