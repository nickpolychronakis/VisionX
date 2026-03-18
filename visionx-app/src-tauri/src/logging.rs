use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

pub struct Logger {
    log_path: PathBuf,
}

impl Logger {
    pub fn new(data_dir: &Path) -> Self {
        let logs_dir = data_dir.join("logs");
        let _ = fs::create_dir_all(&logs_dir);
        Logger {
            log_path: logs_dir.join("visionx.log"),
        }
    }

    pub fn log(&self, level: &str, message: &str) {
        let timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
        let line = format!("[{}] {:<5} {}\n", timestamp, level, message);

        if let Ok(mut file) = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.log_path)
        {
            let _ = file.write_all(line.as_bytes());
        }
    }

    pub fn info(&self, message: &str) {
        self.log("INFO", message);
    }

    pub fn error(&self, message: &str) {
        self.log("ERROR", message);
    }

    pub fn path(&self) -> &Path {
        &self.log_path
    }
}

/// Get or create a logger for the given app data directory
pub fn get_log_path(data_dir: &Path) -> PathBuf {
    data_dir.join("logs").join("visionx.log")
}
