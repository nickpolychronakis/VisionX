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

# Chain mode - treat multiple videos as continuous sequence
python vision.py --chain video_part1.mp4 video_part2.mp4 video_part3.mp4
python vision.py --dir /path/to/videos --chain

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
| `--chain` | Process multiple videos as continuous sequence |
| `--search` | Custom detection prompts |
| `--model` | Model file (default: yoloe-26l-seg.pt) |
| `--conf` | Confidence threshold (default: 0.5) |
| `--output` | Output directory |
| `--video` | Save annotated video (debug) |
| `--crops` | Save cropped detections |
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

## License Plate Reading (plate.py)

Multi-frame plate reader for CCTV/dashcam footage: select the plate once, the
tool tracks it across frames, fuses the sharpest crops into one cleaner image
and outputs a **ranked candidate list** with confidence scores — meant for
database lookup of likely plates, **not** for evidentiary use.

```bash
# Interactive: seek to a frame, draw a box around the plate, tracking is automatic
python plate.py video.mp4

# Headless (known position): x,y,w,h box at a given frame
python plate.py video.mp4 --roi 613,412,95,38 --start-frame 120 --no-gui
```

The tool opens directly in the **selection screen**: drag a box around the
plate with the mouse at any moment — tracking starts on release, no
confirmation key. `SPACE` play/pause, `a`/`d` ±1 frame, `j`/`l` ±25, seek bar
at the bottom (click/drag to scrub), `q` abort.

During **tracking** (green box = detector-confirmed, orange = tracker-only;
preview at 0.7× real time — `-`/`+` adjust, `f` full speed): drag directly on
the live view to correct the box (the video freezes while you draw), or
`SPACE` opens fix-mode with frame-by-frame rewind (`a`/`d`/`j`/`l` seek, drag
= new box, `SPACE` resume, `q` finish); while rewinding, each frame shows the
box the tracker had there (`[tracked]`) or `[NO TRACK]` — so you can pinpoint
the exact frame where it lost the plate. On track loss fix-mode opens
automatically — there `SPACE` = keep searching (useful for occlusions), `q` =
finish & compute.

When the video ends the tool does **not** close on its own: it pauses in
fix-mode on the last frame so you can rewind and re-check; press `q` twice to
compute and finish. All keys also work with the **Greek keyboard layout**
(α/δ/ξ/λ/ρ/φ, and the `q` position key `;`).

`q` always needs a **double-press within 1.5s** — protection against stray
key events (e.g. from window focus changes) ending a session. The console
always prints *why* tracking ended; if it ever stops on its own, rerun with
`--debug` and check the `<video>_plate_debug.log` it writes next to the video.

| Option | Description |
|--------|-------------|
| `--roi x,y,w,h` | Initial plate box (skips GUI selection) |
| `--start-frame` / `--end-frame` | Tracking range |
| `--tracker` | `csrt` accurate (default) / `kcf` faster |
| `--top` | Number of candidates (default 5) |
| `--max-ocr-frames` | OCR the N sharpest crops (default 60) |
| `--fuse-top` | Stack the N sharpest aligned crops (default 15) |
| `--no-pattern-prior` | Disable the soft Greek LLL-NNNN ranking bonus |
| `--speed` | Preview speed vs real time (default 0.7) |
| `--refine-every` | Detector drift-check interval in frames (default 1 = every frame) |
| `--no-enhance` | Disable tonal OCR variants (CLAHE/gamma voting) |

OCR runs as a **3-model ensemble** (different architectures make different
systematic errors; joint voting recovers characters any single model loses),
and two candidate lists are produced: the free list, plus a list **projected
onto the Greek plate shape** (3 valid GR letters + 4/3 digits) — use the
projected list for Greek vehicles, the free list covers foreign plates.

Output in `<video>_plate/`:

- **`report.html`** — the main human-facing output (Greek): both ranked lists
  with per-character consensus/alternatives, the fused & best-frame images at
  reading size, a 5-rendering enhancement panel per frame (original / CLAHE /
  shadow-gamma / highlight-gamma / negative) for examiner-style review, the
  full voting sheet, and the processing methodology with all parameters.
- `candidates.json` — machine-readable counterpart (same data + per-frame
  reads, for scripts/DB tooling).
- `fused.png` / `fused_large.png` (contrast-stretched for the naked eye),
  `best_frame_large.png`, `frames_sheet.png`.

**Setup note:** plate.py needs OpenCV's *contrib* build (CSRT tracker).
`ultralytics` and `fast-plate-ocr` pull plain/headless opencv wheels that
overwrite the same `cv2` package, so after `pip install -r requirements.txt`
always run:

```bash
pip install --force-reinstall --no-deps "opencv-contrib-python>=5.0"
```

## VLC Navigation

1. Open report and video side-by-side
2. Click timestamp → copies to clipboard
3. In VLC: `Ctrl+T` (Win/Linux) or `Cmd+T` (Mac)
4. Paste timestamp → Enter
5. VLC jumps to that moment
