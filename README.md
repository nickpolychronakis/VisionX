# VisionX

Pure YOLOE-26 video analysis CLI.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Basic - annotated video output
python vision.py video.mp4

# Custom prompts
python vision.py --search "white car" video.mp4

# Save cropped objects
python vision.py --crops video.mp4

# Save coordinates
python vision.py --txt video.mp4

# Fast mode (FP16 + frame skip)
python vision.py --half --stride 2 video.mp4

# Live preview
python vision.py --show video.mp4
```

## Options

| Option | Description |
|--------|-------------|
| `--search` | Custom detection prompts |
| `--model` | Model file (default: yoloe-26m-seg.pt) |
| `--conf` | Confidence threshold (default: 0.5) |
| `--output` | Output directory (default: ./output) |
| `--crops` | Save cropped detections |
| `--txt` | Save coordinates to txt |
| `--half` | FP16 mode (faster) |
| `--stride` | Frame skip (2 = 2x faster) |
| `--show` | Live preview window |
| `--config` | Config file (default: config.yaml) |
