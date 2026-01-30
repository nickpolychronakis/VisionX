// Prevents additional console window on Windows in release
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::process::{Command, Stdio, Child};
use std::io::{BufRead, BufReader};
use std::thread;
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter};
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

#[derive(Deserialize)]
struct ProcessConfig {
    confidence: f32,
    stride: u32,
    half_precision: bool,
    output_dir: String,
    search_prompts: Vec<String>,
}

// Global state to track the current process
struct ProcessState {
    child: Option<Child>,
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

    // Build command arguments
    let mut args = vec![
        "--json-progress".to_string(),
        "--conf".to_string(),
        config.confidence.to_string(),
        "--stride".to_string(),
        config.stride.to_string(),
    ];

    if config.half_precision {
        args.push("--half".to_string());
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

    let python_exe = "/Users/nickpolychronakis/Developer/VisionX/venv/bin/python3";
    let vision_script = "/Users/nickpolychronakis/Developer/VisionX/vision.py";

    let _ = app.emit("progress", ProgressEvent {
        event_type: "status".to_string(),
        video: format!("Starting: {}", vision_script),
        frame: 0,
        total_frames: 0,
        video_index: 0,
        total_videos: files.len() as u32,
        fps: 0.0,
    });

    let mut child = Command::new(python_exe)
        .arg(vision_script)
        .args(&args)
        .current_dir("/Users/nickpolychronakis/Developer/VisionX")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to start Python: {}", e))?;

    // Store child process in state
    {
        let mut process_state = state.lock().map_err(|e| e.to_string())?;
        // Kill any existing process first
        if let Some(mut old_child) = process_state.child.take() {
            let _ = old_child.kill();
        }
    }

    let stdout = child.stdout.take()
        .ok_or("Failed to capture stdout")?;

    let stderr = child.stderr.take()
        .ok_or("Failed to capture stderr")?;

    let stderr_thread = thread::spawn(move || {
        let reader = BufReader::new(stderr);
        let mut stderr_output = String::new();
        for line in reader.lines() {
            if let Ok(line) = line {
                stderr_output.push_str(&line);
                stderr_output.push('\n');
            }
        }
        stderr_output
    });

    let state_clone = Arc::clone(&state);
    let reader = BufReader::new(stdout);

    for line in reader.lines() {
        // Check if cancelled
        {
            let process_state = state_clone.lock().map_err(|e| e.to_string())?;
            if process_state.cancelled {
                // Kill the process
                drop(process_state);
                let mut ps = state_clone.lock().map_err(|e| e.to_string())?;
                if let Some(mut c) = ps.child.take() {
                    let _ = c.kill();
                }
                return Err("Processing cancelled".to_string());
            }
        }

        if let Ok(line) = line {
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&line) {
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
                    _ => {}
                }
            }
        }
    }

    let status = child.wait().map_err(|e| e.to_string())?;
    let stderr_output = stderr_thread.join().unwrap_or_default();

    // Clear the child from state
    {
        let mut process_state = state.lock().map_err(|e| e.to_string())?;
        process_state.child = None;
    }

    if !status.success() {
        // Check if it was cancelled
        let process_state = state.lock().map_err(|e| e.to_string())?;
        if process_state.cancelled {
            return Err("Processing cancelled".to_string());
        }
        if !stderr_output.is_empty() {
            return Err(format!("Processing failed: {}", stderr_output));
        }
        return Err("Processing failed".to_string());
    }

    Ok(reports)
}

#[tauri::command]
fn cancel_processing(state: tauri::State<'_, Arc<Mutex<ProcessState>>>) -> Result<(), String> {
    let mut process_state = state.lock().map_err(|e| e.to_string())?;
    process_state.cancelled = true;

    // Try to kill the process if it exists
    if let Some(mut child) = process_state.child.take() {
        let _ = child.kill();
    }

    Ok(())
}

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

    // On macOS, we can't auto-update without code signing, so just check GitHub API
    #[cfg(target_os = "macos")]
    {
        // Fetch latest release from GitHub API
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

        // Find the .dmg asset for download
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

    // On Windows/Linux, use the built-in updater
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

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .plugin(tauri_plugin_http::init())
        .manage(Arc::new(Mutex::new(ProcessState::default())))
        .invoke_handler(tauri::generate_handler![process_videos, cancel_processing, get_report_content, open_file, show_in_folder, check_for_updates, install_update])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
