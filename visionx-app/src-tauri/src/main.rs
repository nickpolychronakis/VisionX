// Prevents additional console window on Windows in release
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod logging;
mod setup;

use logging::Logger;
use std::process::Command;
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter, Manager};
use tauri_plugin_shell::ShellExt;
use tauri_plugin_shell::process::{CommandChild, CommandEvent};
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize)]
struct ProgressEvent {
    event_type: String,
    video: String,
    frame: u32,
    total_frames: u32,
    video_index: u32,
    total_videos: u32,
    fps: f32,
}

#[derive(Clone, Serialize)]
struct StatusEvent {
    event_type: String,
    message: String,
}

#[derive(Deserialize)]
struct ProcessConfig {
    confidence: f32,
    stride: u32,
    half_precision: bool,
    imgsz: u32,
    parallel: u32,
    output_dir: String,
    search_prompts: Vec<String>,
}

// Global state to track the current process
struct ProcessState {
    child: Option<CommandChild>,
    cancelled: bool,
}

impl Default for ProcessState {
    fn default() -> Self {
        ProcessState {
            child: None,
            cancelled: false,
        }
    }
}

// ============================================================
// Setup commands
// ============================================================

#[tauri::command]
async fn check_setup(app: AppHandle) -> Result<setup::SetupStatus, String> {
    let data_dir = app.path().app_data_dir()
        .map_err(|e| format!("Failed to get data dir: {}", e))?;
    let _ = std::fs::create_dir_all(&data_dir);

    let logger = Logger::new(&data_dir);
    logger.info(&format!("App started, version {}", app.package_info().version));
    logger.info("Checking setup status...");

    let current_version = app.package_info().version.to_string();
    let status = setup::check_setup(&data_dir, &current_version);
    logger.info(&format!("Setup needed: {} (app v{}, setup v{})",
        status.needs_setup, current_version,
        std::fs::read_to_string(data_dir.join("setup.json"))
            .ok()
            .and_then(|c| serde_json::from_str::<serde_json::Value>(&c).ok())
            .and_then(|v| v["app_version"].as_str().map(String::from))
            .unwrap_or_else(|| "none".to_string())
    ));

    Ok(status)
}

#[tauri::command]
async fn detect_gpu(app: AppHandle) -> Result<setup::GpuInfo, String> {
    let data_dir = app.path().app_data_dir()
        .map_err(|e| format!("Failed to get data dir: {}", e))?;
    let logger = Logger::new(&data_dir);

    Ok(setup::detect_gpu(&logger))
}

#[tauri::command]
async fn run_setup(app: AppHandle, use_cuda: bool) -> Result<(), String> {
    let data_dir = app.path().app_data_dir()
        .map_err(|e| format!("Failed to get data dir: {}", e))?;
    let _ = std::fs::create_dir_all(&data_dir);

    let logger = Logger::new(&data_dir);
    logger.info(&format!("Starting setup (CUDA: {})", use_cuda));

    setup::run_setup(&app, &data_dir, use_cuda, &logger).await
}

#[tauri::command]
fn get_log_path(app: AppHandle) -> Result<String, String> {
    let data_dir = app.path().app_data_dir()
        .map_err(|e| format!("Failed to get data dir: {}", e))?;
    let path = logging::get_log_path(&data_dir);
    Ok(path.to_string_lossy().to_string())
}

// ============================================================
// Processing commands
// ============================================================

#[tauri::command]
async fn process_videos(
    app: AppHandle,
    state: tauri::State<'_, Arc<Mutex<ProcessState>>>,
    files: Vec<String>,
    config: ProcessConfig,
) -> Result<Vec<String>, String> {
    let mut reports: Vec<String> = Vec::new();

    // Reset cancelled state
    {
        let mut process_state = state.lock().map_err(|e| e.to_string())?;
        process_state.cancelled = false;
    }

    // Get directories
    let data_dir = app.path().app_data_dir()
        .map_err(|e| format!("Failed to get data dir: {}", e))?;
    let resource_dir = app.path().resource_dir()
        .map_err(|e| format!("Failed to get resource dir: {}", e))?;

    let logger = Logger::new(&data_dir);
    logger.info(&format!("Processing {} video(s)", files.len()));
    logger.info(&format!("Config: confidence={}, stride={}, imgsz={}, half={}",
        config.confidence, config.stride, config.imgsz, config.half_precision));
    for f in &files {
        logger.info(&format!("  File: {}", f));
    }

    // Paths
    let python_exe = setup::python_exe_path(&data_dir);
    let packages_dir = setup::packages_dir_path(&data_dir);
    let scripts_dir = setup::scripts_dir_path(&data_dir);
    let vision_script = scripts_dir.join("vision.py");

    logger.info(&format!("Python: {}", python_exe.display()));
    logger.info(&format!("Script: {}", vision_script.display()));
    logger.info(&format!("Packages: {}", packages_dir.display()));

    if !python_exe.exists() {
        logger.error("Python not found — setup incomplete");
        return Err("Python not installed. Please run setup first.".to_string());
    }
    if !vision_script.exists() {
        logger.error("vision.py not found — setup incomplete");
        return Err("Vision script not found. Please run setup first.".to_string());
    }

    // Build command arguments
    let mut args = vec![
        vision_script.to_string_lossy().to_string(),
        "--json-progress".to_string(),
        "--resource-dir".to_string(),
        resource_dir.to_string_lossy().to_string(),
        "--data-dir".to_string(),
        data_dir.to_string_lossy().to_string(),
        "--conf".to_string(),
        config.confidence.to_string(),
        "--stride".to_string(),
        config.stride.to_string(),
    ];

    if config.half_precision {
        args.push("--half".to_string());
    }

    if config.imgsz != 640 {
        args.push("--imgsz".to_string());
        args.push(config.imgsz.to_string());
    }

    if config.parallel > 1 {
        args.push("--parallel".to_string());
        args.push(config.parallel.to_string());
    }

    if !config.output_dir.is_empty() {
        args.push("--output".to_string());
        args.push(config.output_dir.clone());
    }

    for prompt in &config.search_prompts {
        args.push("--search".to_string());
        args.push(prompt.clone());
    }

    args.extend(files.iter().cloned());

    let _ = app.emit("progress", ProgressEvent {
        event_type: "status".to_string(),
        video: "Starting VisionX engine...".to_string(),
        frame: 0,
        total_frames: 0,
        video_index: 0,
        total_videos: files.len() as u32,
        fps: 0.0,
    });

    // Spawn Python process (not sidecar)
    logger.info(&format!("Spawning: {} {}", python_exe.display(), args.join(" ")));

    let python_str = python_exe.to_string_lossy().to_string();
    let packages_str = packages_dir.to_string_lossy().to_string();
    let scripts_str = scripts_dir.to_string_lossy().to_string();
    let use_system_python = setup::is_system_python(&data_dir);

    // Build PYTHONPATH: always include scripts dir (for report.py import)
    let python_path = if use_system_python {
        scripts_str.clone()
    } else {
        #[cfg(target_os = "windows")]
        { format!("{};{}", packages_str, scripts_str) }
        #[cfg(not(target_os = "windows"))]
        { format!("{}:{}", packages_str, scripts_str) }
    };

    let mut cmd = app.shell()
        .command(&python_str);
    cmd = cmd.env("PYTHONUNBUFFERED", "1");
    cmd = cmd.env("PYTHONUTF8", "1");
    cmd = cmd.env("PYTHONIOENCODING", "utf-8");
    cmd = cmd.env("PYTHONPATH", &python_path);

    // On Windows, prepend torch/lib to PATH so Windows DLL loader can find c10.dll and
    // its CUDA dependencies. PyTorch's _load_dll_libraries() uses os.add_dll_directory()
    // for this, but with embedded Python + pip --target installs, the PATH fallback in
    // _load_dll_libraries() may compute wrong paths (it uses sys.exec_prefix which points
    // to the embed dir, not the packages dir). This ensures the DLLs are always findable.
    #[cfg(target_os = "windows")]
    {
        let torch_lib_dir = packages_dir.join("torch").join("lib");
        if torch_lib_dir.exists() {
            let current_path = std::env::var("PATH").unwrap_or_default();
            let new_path = format!("{};{}", torch_lib_dir.to_string_lossy(), current_path);
            cmd = cmd.env("PATH", &new_path);
            logger.info(&format!("Added torch/lib to PATH: {}", torch_lib_dir.display()));
        }
    }

    let (mut rx, child) = cmd
        .args(&args)
        .spawn()
        .map_err(|e| {
            let msg = format!("Failed to start Python: {}", e);
            logger.error(&msg);
            msg
        })?;

    // Store child process in state
    {
        let mut process_state = state.lock().map_err(|e| e.to_string())?;
        if let Some(old_child) = process_state.child.take() {
            let _ = old_child.kill();
        }
        process_state.child = Some(child);
    }

    let mut stderr_output = String::new();

    // Read events from Python process
    while let Some(event) = rx.recv().await {
        // Check if cancelled
        {
            let process_state = state.lock().map_err(|e| e.to_string())?;
            if process_state.cancelled {
                drop(process_state);
                let mut ps = state.lock().map_err(|e| e.to_string())?;
                if let Some(child) = ps.child.take() {
                    let _ = child.kill();
                }
                return Err("Processing cancelled".to_string());
            }
        }

        match event {
            CommandEvent::Stdout(line) => {
                let line_str = String::from_utf8_lossy(&line);
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&line_str) {
                    match json.get("type").and_then(|v| v.as_str()) {
                        Some("progress") => {
                            let event = ProgressEvent {
                                event_type: "progress".to_string(),
                                video: json["video"].as_str().unwrap_or("").to_string(),
                                frame: json["frame"].as_u64().unwrap_or(0) as u32,
                                total_frames: json["total_frames"].as_u64().unwrap_or(0) as u32,
                                video_index: json["video_index"].as_u64().unwrap_or(0) as u32,
                                total_videos: json["total_videos"].as_u64().unwrap_or(0) as u32,
                                fps: json["fps"].as_f64().unwrap_or(0.0) as f32,
                            };
                            let _ = app.emit("progress", event);
                        },
                        Some("report") => {
                            if let Some(path) = json["path"].as_str() {
                                reports.push(path.to_string());
                            }
                        },
                        Some("model_download") => {
                            let message = json["message"].as_str().unwrap_or("Downloading model...").to_string();
                            let _ = app.emit("status", StatusEvent {
                                event_type: "model_download".to_string(),
                                message,
                            });
                        },
                        Some("error") => {
                            let message = json["message"].as_str().unwrap_or("Unknown error").to_string();
                            logger.error(&format!("Vision error: {}", message));
                            let _ = app.emit("status", StatusEvent {
                                event_type: "error".to_string(),
                                message,
                            });
                        },
                        _ => {}
                    }
                }
            }
            CommandEvent::Stderr(line) => {
                let line_str = String::from_utf8_lossy(&line);
                let trimmed = line_str.trim();
                if !trimmed.is_empty() {
                    logger.info(&format!("Python stderr: {}", trimmed));
                }
                stderr_output.push_str(&line_str);
                stderr_output.push('\n');
            }
            CommandEvent::Terminated(status) => {
                let exit_code = status.code.unwrap_or(-1);
                logger.info(&format!("Python process exited with code {}", exit_code));

                let mut process_state = state.lock().map_err(|e| e.to_string())?;
                process_state.child = None;

                if exit_code != 0 {
                    if process_state.cancelled {
                        logger.info("Processing was cancelled by user");
                        return Err("Processing cancelled".to_string());
                    }
                    let err_msg = if !stderr_output.is_empty() {
                        format!("Processing failed: {}", stderr_output)
                    } else {
                        format!("Processing failed (exit code {})", exit_code)
                    };
                    logger.error(&err_msg);
                    return Err(err_msg);
                }
                break;
            }
            _ => {}
        }
    }

    logger.info(&format!("Processing complete, {} report(s)", reports.len()));
    Ok(reports)
}

#[tauri::command]
fn cancel_processing(state: tauri::State<'_, Arc<Mutex<ProcessState>>>) -> Result<(), String> {
    let mut process_state = state.lock().map_err(|e| e.to_string())?;
    process_state.cancelled = true;

    if let Some(child) = process_state.child.take() {
        let _ = child.kill();
    }

    Ok(())
}

// ============================================================
// Video info commands
// ============================================================

#[tauri::command]
async fn get_video_resolution(app: AppHandle, files: Vec<String>) -> Result<u32, String> {
    if files.is_empty() {
        return Ok(0);
    }

    let data_dir = app.path().app_data_dir()
        .map_err(|e| format!("Failed to get data dir: {}", e))?;
    let python_exe = setup::python_exe_path(&data_dir);

    if !python_exe.exists() {
        return Ok(640); // Default if Python not ready
    }

    // Get max resolution from all videos
    let script = format!(
        "import cv2,sys,json;mx=0\nfor f in sys.argv[1:]:\n c=cv2.VideoCapture(f)\n w,h=int(c.get(3)),int(c.get(4))\n mx=max(mx,w,h)\n c.release()\nprint(mx)",
    );

    let mut args = vec!["-c".to_string(), script];
    args.extend(files);

    let packages_dir = setup::packages_dir_path(&data_dir);
    let use_system = setup::is_system_python(&data_dir);

    let output = std::process::Command::new(&python_exe)
        .args(&args)
        .env("PYTHONUTF8", "1")
        .env("PYTHONPATH", if use_system { String::new() } else { packages_dir.to_string_lossy().to_string() })
        .output()
        .map_err(|e| format!("Failed to get video info: {}", e))?;

    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        Ok(stdout.parse::<u32>().unwrap_or(640))
    } else {
        Ok(640) // Default on error
    }
}

// ============================================================
// Utility commands
// ============================================================

#[tauri::command]
fn get_report_content(path: String) -> Result<String, String> {
    std::fs::read_to_string(&path)
        .map_err(|e| format!("Failed to read report: {}", e))
}

#[tauri::command]
fn open_file(path: String) -> Result<(), String> {
    #[cfg(target_os = "macos")]
    {
        Command::new("open")
            .arg(&path)
            .spawn()
            .map_err(|e| format!("Failed to open file: {}", e))?;
    }
    #[cfg(target_os = "windows")]
    {
        Command::new("cmd")
            .args(["/C", "start", "", &path])
            .spawn()
            .map_err(|e| format!("Failed to open file: {}", e))?;
    }
    #[cfg(target_os = "linux")]
    {
        Command::new("xdg-open")
            .arg(&path)
            .spawn()
            .map_err(|e| format!("Failed to open file: {}", e))?;
    }
    Ok(())
}

#[tauri::command]
fn show_in_folder(path: String) -> Result<(), String> {
    let folder = std::path::Path::new(&path)
        .parent()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or(path.clone());

    #[cfg(target_os = "macos")]
    {
        Command::new("open")
            .arg(&folder)
            .spawn()
            .map_err(|e| format!("Failed to open folder: {}", e))?;
    }
    #[cfg(target_os = "windows")]
    {
        Command::new("explorer")
            .arg(&folder)
            .spawn()
            .map_err(|e| format!("Failed to open folder: {}", e))?;
    }
    #[cfg(target_os = "linux")]
    {
        Command::new("xdg-open")
            .arg(&folder)
            .spawn()
            .map_err(|e| format!("Failed to open folder: {}", e))?;
    }
    Ok(())
}

// ============================================================
// Update commands
// ============================================================

#[derive(Clone, Serialize)]
struct UpdateInfo {
    available: bool,
    version: String,
    current_version: String,
    download_url: String,
    can_auto_update: bool,
}

#[tauri::command]
async fn check_for_updates(app: AppHandle) -> Result<UpdateInfo, String> {
    let current_version = app.package_info().version.to_string();

    #[cfg(target_os = "macos")]
    {
        let client = reqwest::Client::new();
        let response = client
            .get("https://api.github.com/repos/nickpolychronakis/VisionX/releases/latest")
            .header("User-Agent", "VisionX-App")
            .send()
            .await
            .map_err(|e| format!("Failed to check for updates: {}", e))?;

        let release: serde_json::Value = response
            .json()
            .await
            .map_err(|e| format!("Failed to parse release info: {}", e))?;

        let latest_version = release["tag_name"]
            .as_str()
            .unwrap_or("v0.0.0")
            .trim_start_matches('v')
            .to_string();

        let download_url = release["assets"]
            .as_array()
            .and_then(|assets| {
                assets.iter().find(|a| {
                    a["name"].as_str().map(|n| n.ends_with(".dmg")).unwrap_or(false)
                })
            })
            .and_then(|asset| asset["browser_download_url"].as_str())
            .unwrap_or("")
            .to_string();

        let available = version_compare(&latest_version, &current_version);

        return Ok(UpdateInfo {
            available,
            version: latest_version,
            current_version,
            download_url,
            can_auto_update: false,
        });
    }

    #[cfg(not(target_os = "macos"))]
    {
        use tauri_plugin_updater::UpdaterExt;
        let updater = app.updater().map_err(|e| format!("Updater error: {}", e))?;

        match updater.check().await {
            Ok(Some(update)) => {
                Ok(UpdateInfo {
                    available: true,
                    version: update.version.clone(),
                    current_version,
                    download_url: String::new(),
                    can_auto_update: true,
                })
            }
            Ok(None) => {
                Ok(UpdateInfo {
                    available: false,
                    version: current_version.clone(),
                    current_version,
                    download_url: String::new(),
                    can_auto_update: true,
                })
            }
            Err(e) => Err(format!("Failed to check for updates: {}", e)),
        }
    }
}

#[cfg(target_os = "macos")]
fn version_compare(latest: &str, current: &str) -> bool {
    let parse_version = |v: &str| -> Vec<u32> {
        v.split('.')
            .filter_map(|s| s.parse().ok())
            .collect()
    };

    let latest_parts = parse_version(latest);
    let current_parts = parse_version(current);

    for i in 0..latest_parts.len().max(current_parts.len()) {
        let l = latest_parts.get(i).copied().unwrap_or(0);
        let c = current_parts.get(i).copied().unwrap_or(0);
        if l > c { return true; }
        if l < c { return false; }
    }
    false
}

#[tauri::command]
async fn install_update(_app: AppHandle) -> Result<(), String> {
    #[cfg(target_os = "macos")]
    {
        return Err("Auto-update not available on macOS. Please download manually.".to_string());
    }

    #[cfg(not(target_os = "macos"))]
    {
        use tauri_plugin_updater::UpdaterExt;
        let updater = _app.updater().map_err(|e| format!("Updater error: {}", e))?;

        if let Some(update) = updater.check().await.map_err(|e| e.to_string())? {
            update.download_and_install(|_, _| {}, || {}).await.map_err(|e| e.to_string())?;
        }

        Ok(())
    }
}

// ============================================================
// App entry point
// ============================================================

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .plugin(tauri_plugin_http::init())
        .manage(Arc::new(Mutex::new(ProcessState::default())))
        .invoke_handler(tauri::generate_handler![
            // Setup
            check_setup,
            detect_gpu,
            run_setup,
            get_log_path,
            // Processing
            process_videos,
            cancel_processing,
            // Video info
            get_video_resolution,
            // Utilities
            get_report_content,
            open_file,
            show_in_folder,
            // Updates
            check_for_updates,
            install_update,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
