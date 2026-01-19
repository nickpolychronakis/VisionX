# VisionX

Pure YOLOE-26 video analysis CLI for object detection and tracking.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Single video
python vision.py video.mp4

# Multiple videos
python vision.py video1.mp4 video2.mp4 video3.mp4

# All videos in directory
python vision.py --dir /path/to/videos

# Custom search prompts
python vision.py --search "white car" "red motorcycle" video.mp4

# Save annotated video (for debugging)
python vision.py --video video.mp4

# Fast mode
python vision.py --half --stride 2 video.mp4
```

## Options

| Option | Description |
|--------|-------------|
| `--dir` | Process all videos in directory |
| `--search` | Custom detection prompts |
| `--model` | Model file (default: yoloe-26l-seg.pt) |
| `--conf` | Confidence threshold (default: 0.5) |
| `--output` | Output directory |
| `--video` | Save annotated video (debug) |
| `--crops` | Save cropped detections |
| `--report` | Generate HTML report (default: on) |
| `--half` | FP16 mode (faster) |
| `--stride` | Frame skip (2 = 2x faster) |
| `--show` | Live preview window |
| `--config` | Config file (default: config.yaml) |

## Output

For each video, generates a single self-contained HTML report:

```
/videos/
├── video1.mp4
├── video1_report.html    ← self-contained report
├── video2.mp4
└── video2_report.html
```

The report includes:
- Thumbnails embedded (no external files)
- Timestamps (click to copy)
- Filter by class (car/person/motorcycle)
- Click thumbnail to zoom

## VLC Navigation

1. Open report and video side-by-side
2. Click timestamp → copies to clipboard
3. In VLC: `Ctrl+T` (Win/Linux) or `Cmd+T` (Mac)
4. Paste timestamp → Enter
5. VLC jumps to that moment
