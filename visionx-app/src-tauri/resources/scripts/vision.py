#!/usr/bin/env python3
"""VisionX - Pure YOLOE-26 Video Analysis"""

import argparse
import base64
import json
import os
import sys

# Ensure the script's own directory is in sys.path (for 'from report import ...')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
import cv2
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO  # type: ignore[attr-defined]
from ultralytics.data.utils import IMG_FORMATS, VID_FORMATS
from report import generate_report
from tracking import TrackCollector, prepare_for_report
import stitch as stitch_mod

# Add .dav (Dahua DVR format) support - OpenCV can read these
VID_FORMATS.add('dav')

DEFAULT_CONFIG = {
    'model': 'yoloe-26x-seg.pt',
    'prompts': ['car', 'person', 'motorcycle'],
    'confidence': 0.35,
    'save_video': False,
    'save_crops': False,
    'save_report': True,
    'half': False,
    'imgsz': 640,
    'vid_stride': 1,
    'show': False,
    # Offline tracklet stitching (ROADMAP Phase A): repair fragmented tracks
    # after processing so the same object is ONE report entry. Threshold from
    # real-footage calibration (see stitch.py); raise it if wrong merges ever
    # appear. embed model is a dedicated nano checkpoint — the main model may
    # be a CoreML/TensorRT export that cannot produce embeddings.
    'stitch': True,
    'stitch_threshold': 0.88,
    'stitch_embed_model': 'yoloe-26n-seg.pt',
    'snapshots': 4,
    # Automatic plate reading on every vehicle track (ROADMAP Phase B):
    # detector + 3-model OCR ensemble vote over the track's best snapshots.
    # Candidate-generation quality — the interactive plate.py remains the
    # deep-analysis path (the report shows its ready-made command per car).
    'plates': True,
    # Best face shots per person track — extraction ONLY, no recognition.
    'faces': True,
    # Color attributes computed on the results (vehicles + clothing).
    'attributes': True,
    # Closed-set detector used when no custom prompts are given (see
    # BENCHMARKS.md: best tracking stability at half the old default's cost).
    'model_closed': 'yolo26l.pt',
}

# Which track classes count as vehicles for the auto-plate pass. Prompts are
# free text (and may be Greek), so this is a keyword match, not a class list.
VEHICLE_KEYWORDS = ('car', 'vehicle', 'truck', 'bus', 'van', 'motorc',
                    'motorbike', 'moto', 'scooter', 'suv', 'taxi', 'lorry',
                    'pickup', 'αυτοκίνητο', 'όχημα', 'μηχανή', 'μοτοσ',
                    'φορτηγό', 'λεωφορείο', 'ταξί', 'βαν')

PERSON_KEYWORDS = ('person', 'pedestrian', 'people', 'man', 'woman', 'child',
                   'άτομο', 'άνθρωπος', 'πεζός', 'άντρας', 'γυναίκα', 'παιδί')


def run_face_shots(tracks: dict, cfg: dict, json_progress: bool):
    """Attach the clearest face shots to person tracks (in place) —
    EXTRACTION only, no recognition (ROADMAP Phase Γ). Any failure just
    logs; the report always comes out."""
    if not cfg.get('faces', True):
        return
    persons = [t for t in tracks.values()
               if any(k in t['class'].lower() for k in PERSON_KEYWORDS)
               and t.get('snapshots')]
    if not persons:
        return
    try:
        from face_shots import FaceExtractor
        extractor = FaceExtractor(model_dir=cfg.get('_data_dir'))
    except Exception as e:  # noqa: BLE001 — optional capability
        log_stderr(f'Face extraction unavailable ({e}) — skipping')
        return
    if json_progress:
        emit_json('status', message=f'Εξαγωγή προσώπων ({len(persons)} άτομα)...')
    found = 0
    for t in persons:
        try:
            crops = [cv2.imdecode(np.frombuffer(s['jpeg'], np.uint8),
                                  cv2.IMREAD_COLOR)
                     for s in t.get('snapshots', [])]
            faces = extractor.best_faces([c for c in crops if c is not None])
        except Exception as e:  # noqa: BLE001
            log_stderr(f'Face extraction failed on a track ({e})')
            continue
        if faces:
            t['faces'] = [{'b64': base64.b64encode(f['jpeg']).decode('utf-8'),
                           'score': round(f['score'], 3)} for f in faces]
            found += 1
    log_stderr(f'Face shots: faces found on {found}/{len(persons)} persons')


def run_attributes(tracks: dict, cfg: dict, json_progress: bool):
    """Color attributes on the results (user-designed second stage): vehicle
    body color + person clothing colors, from the snapshot crops."""
    if not cfg.get('attributes', True):
        return
    try:
        import attributes as attr_mod
    except Exception as e:  # noqa: BLE001
        log_stderr(f'Attributes unavailable ({e})')
        return
    for t in tracks.values():
        snaps = t.get('snapshots') or []
        if not snaps:
            continue
        crops = [cv2.imdecode(np.frombuffer(s['jpeg'], np.uint8),
                              cv2.IMREAD_COLOR) for s in snaps]
        crops = [c for c in crops if c is not None]
        try:
            cls = t['class'].lower()
            if any(k in cls for k in VEHICLE_KEYWORDS):
                res = attr_mod.vehicle_color(crops)
                if res:
                    t['attrs'] = {'color': res[0], 'color_conf': res[1]}
            elif any(k in cls for k in PERSON_KEYWORDS):
                clothing = attr_mod.person_clothing(crops)
                if clothing:
                    t['attrs'] = {'clothing': {k: {'color': v[0], 'conf': v[1]}
                                               for k, v in clothing.items()}}
        except Exception as e:  # noqa: BLE001
            log_stderr(f'Attribute extraction failed on a track ({e})')


def run_auto_plates(tracks: dict, cfg: dict, json_progress: bool,
                    fps: float | None = None, video_path: str | None = None):
    """Attach plate candidates to vehicle tracks (in place). Failures only
    log — a report must always come out even if the ALPR stack is missing."""
    if not cfg.get('plates', True):
        return
    vehicles = [t for t in tracks.values()
                if any(k in t['class'].lower() for k in VEHICLE_KEYWORDS)
                and t.get('snapshots')]
    if not vehicles:
        return
    try:
        from plate_core import PlateReader
        reader = PlateReader()
    except Exception as e:  # noqa: BLE001 — optional capability
        log_stderr(f'Auto-ALPR unavailable ({e}) — skipping plate pass')
        return
    if json_progress:
        emit_json('status', message=f'Ανάγνωση πινακίδων ({len(vehicles)} οχήματα)...')
    # One capture reused for the raw-frame resampling of all vehicles.
    cap = cv2.VideoCapture(video_path) if video_path else None
    found = 0
    for t in vehicles:
        try:
            crops = [cv2.imdecode(np.frombuffer(s['jpeg'], np.uint8),
                                  cv2.IMREAD_COLOR)
                     for s in t.get('snapshots', [])]
            crops = [c for c in crops if c is not None]
            # Raw-frame resampling: the report snapshots are chosen for the
            # VEHICLE (and recompressed as JPEG) — often only 1 of them shows
            # a readable plate, which reproduces the single-frame weakness
            # our multi-frame research warns about (field case: confident
            # wrong read from 1 frame). Cut up to 12 fresh crops straight
            # from the video, spread across the track's trajectory.
            if cap is not None and cap.isOpened():
                crops += _resample_track_crops(cap, t, max_extra=12)
            res = reader.read_from_crops(crops)
        except Exception as e:  # noqa: BLE001
            log_stderr(f'Plate read failed on a track ({e})')
            continue
        if not res:
            continue
        t['plate'] = res
        found += 1
        # Ready-made deep-analysis command: the interactive plate.py flow,
        # pre-aimed at this vehicle's best snapshot (single-video mode only —
        # chain snapshots only know their file's basename).
        snap = t['snapshots'][0]
        if fps and video_path and not snap.get('file'):
            x1, y1, x2, y2 = snap['box']
            frame_no = max(0, int(round(snap['ts'] * fps)) - 5)
            res['deep_cmd'] = (f'python plate.py "{video_path}" '
                               f'--roi {x1},{y1},{x2 - x1},{y2 - y1} '
                               f'--start-frame {frame_no}')
    if cap is not None:
        cap.release()
    log_stderr(f'Auto-ALPR: plates read on {found}/{len(vehicles)} vehicles')


def _resample_track_crops(cap, track, max_extra=12, pad_ratio=0.15):
    """Decode up to max_extra vehicle crops from the ORIGINAL video frames,
    evenly spread over the track's per-frame boxes (no JPEG round-trip)."""
    boxes = track.get('_boxes') or []
    if not boxes:
        return []
    step = max(1, len(boxes) // max_extra)
    out = []
    for fi, x1, y1, x2, y2 in boxes[::step][:max_extra]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        px, py = int((x2 - x1) * pad_ratio), int((y2 - y1) * pad_ratio)
        cx1, cy1 = max(0, x1 - px), max(0, y1 - py)
        cx2, cy2 = min(w, x2 + px), min(h, y2 + py)
        if cx2 > cx1 and cy2 > cy1:
            out.append(frame[cy1:cy2, cx1:cx2].copy())
    return out


def resolve_model_path(name: str, data_dir=None, resource_dir=None) -> str:
    """Resolve a model filename via the app's storage convention (data-dir
    models/, bundled resource models/, CWD, script dir). If nothing exists,
    return a data-dir path so ultralytics auto-downloads THERE instead of
    polluting the current working directory."""
    p = Path(name)
    if p.is_absolute() and p.exists():
        return str(p)
    for base in [data_dir, resource_dir]:
        if base:
            cand = Path(base) / 'models' / p.name
            if cand.exists():
                return str(cand)
    for base in [Path.cwd(), Path(__file__).parent]:
        cand = Path(base) / p.name
        if cand.exists():
            return str(cand)
    if data_dir:
        dest = Path(data_dir) / 'models'
        dest.mkdir(parents=True, exist_ok=True)
        return str(dest / p.name)
    return name


def run_stitching(tracks: dict, cfg: dict, json_progress: bool) -> dict:
    """Offline tracklet-stitching pass shared by single and chain modes.
    Any failure degrades to the unstitched tracks — a report must always
    come out."""
    if not cfg.get('stitch', True) or len(tracks) < 2:
        return tracks
    if json_progress:
        emit_json('status', message='Συγχώνευση διαδρομών (stitching)...')
    try:
        embed_path = resolve_model_path(cfg.get('stitch_embed_model',
                                                'yoloe-26n-seg.pt'),
                                        cfg.get('_data_dir'),
                                        cfg.get('_resource_dir'))
        stitch_mod.compute_embeddings(tracks, embed_path, log=log_stderr)
        return stitch_mod.stitch_tracks(
            tracks, sim_threshold=float(cfg.get('stitch_threshold', 0.88)),
            log=log_stderr)
    except Exception as e:  # noqa: BLE001
        log_stderr(f'Stitching failed ({e}) — using raw tracks')
        return tracks


def load_config(config_path: str = 'config.yaml') -> dict:
    """Load config from YAML or use defaults"""
    config = DEFAULT_CONFIG.copy()
    if Path(config_path).exists():
        with open(config_path) as f:
            user_config = yaml.safe_load(f)
            if user_config:
                config.update(user_config)
    return config


def _has_tensor_cores() -> bool:
    """Check if NVIDIA GPU has Tensor Cores (RTX series)."""
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability()
    gpu_name = torch.cuda.get_device_name().lower()
    # Compute capability >= 7.0 (Volta+), but GTX 16xx lacks Tensor Cores
    return cap >= (7, 0) and 'gtx 16' not in gpu_name


def _patch_fp16_segmentation():
    """Monkey-patch ultralytics FP16 segmentation mask bug (masks_in dtype mismatch)."""
    try:
        import ultralytics.utils.ops as _ops
        _orig = _ops.process_mask
        def _patched(protos, masks_in, bboxes, shape, upsample=False):
            return _orig(protos, masks_in.float(), bboxes, shape, upsample)
        _ops.process_mask = _patched

        if hasattr(_ops, 'process_mask_native'):
            _orig_native = _ops.process_mask_native
            def _patched_native(protos, masks_in, bboxes, shape):
                return _orig_native(protos, masks_in.float(), bboxes, shape)
            _ops.process_mask_native = _patched_native
    except Exception:
        pass  # Silently fail — FP16 will be disabled as fallback

_fp16_patched = False

def select_device_and_half(model_path: str, half_requested: bool) -> tuple:
    """Select compute device and determine if FP16 is safe to use."""
    global _fp16_patched
    if model_path.endswith('.mlpackage') or model_path.endswith('.engine'):
        if model_path.endswith('.engine'):
            return 'cuda', False  # TensorRT handles precision internally
        return None, False  # CoreML handles precision internally
    elif torch.cuda.is_available():
        device = 'cuda'
        if half_requested and _has_tensor_cores():
            # Patch FP16 seg bug once, then enable FP16 on RTX GPUs
            if not _fp16_patched:
                _patch_fp16_segmentation()
                _fp16_patched = True
            return device, True
        return device, False
    elif torch.backends.mps.is_available():
        return 'mps', False  # MPS doesn't reliably support FP16 for all ops
    else:
        return 'cpu', False  # CPU doesn't support FP16


def log_stderr(msg: str):
    """Log to stderr — captured by Rust backend into visionx.log"""
    print(msg, file=sys.stderr, flush=True)


def emit_json(event_type: str, **data):
    """Emit a JSON event to stdout for GUI consumption"""
    event = {'type': event_type, **data}
    print(json.dumps(event), flush=True)


def get_coreml_model_path(pt_model: str) -> Path | None:
    """Get CoreML model path if it exists (macOS only)"""
    if sys.platform != 'darwin':
        return None
    mlpackage = Path(pt_model).with_suffix('.mlpackage')
    return mlpackage if mlpackage.exists() else None


def export_to_coreml(pt_model: str, prompts: list[str], json_progress: bool = False, data_dir: str | None = None) -> Path | None:
    """Export PyTorch model to CoreML (macOS only, one-time operation).

    Classes must be set before export as CoreML models have fixed classes.
    Exports to data_dir/models/ if provided (resource dir may be read-only).
    """
    if sys.platform != 'darwin':
        return None

    # Check if already exported next to the .pt file
    mlpackage = Path(pt_model).with_suffix('.mlpackage')
    if mlpackage.exists():
        return mlpackage

    # Also check data dir
    if data_dir:
        data_mlpackage = Path(data_dir) / 'models' / mlpackage.name
        if data_mlpackage.exists():
            return data_mlpackage

    if json_progress:
        emit_json('status', message='Exporting to CoreML (one-time)...')
    else:
        print('Exporting to CoreML (one-time, may take a few minutes)...')

    try:
        model = YOLO(pt_model)
        model.set_classes(prompts)  # type: ignore[misc]  # Bake classes into CoreML model
        # Export to data dir if available (resource dir may be read-only in bundled app)
        export_dir = str(Path(data_dir) / 'models') if data_dir else None
        model.export(format="coreml", project=export_dir)
        # Check both possible locations
        if data_dir:
            data_mlpackage = Path(data_dir) / 'models' / mlpackage.name
            if data_mlpackage.exists():
                return data_mlpackage
        return mlpackage if mlpackage.exists() else None
    except Exception as e:
        if json_progress:
            emit_json('status', message=f'CoreML export failed: {e}')
        else:
            print(f'CoreML export failed: {e}')
        return None


def export_to_tensorrt(pt_model: str, prompts: list[str], json_progress: bool = False, data_dir: str | None = None) -> Path | None:
    """Export PyTorch model to TensorRT (NVIDIA GPU, one-time operation).

    Classes are baked in, just like CoreML. Requires tensorrt pip package.
    The .engine file is GPU-specific and must be re-exported on different hardware.
    """
    if not torch.cuda.is_available():
        return None

    try:
        import tensorrt as trt  # noqa: F401
        log_stderr(f'TensorRT version: {trt.__version__}')
    except ImportError:
        log_stderr('TensorRT not installed, skipping optimization')
        return None
    except Exception as e:
        # Wrapper module exists but bindings broken
        log_stderr(f'TensorRT import failed: {e}')
        return None

    engine_name = Path(pt_model).with_suffix('.engine').name

    # Check data dir first (writable location for bundled apps)
    if data_dir:
        data_engine = Path(data_dir) / 'models' / engine_name
        if data_engine.exists():
            log_stderr(f'Using existing TensorRT engine: {data_engine}')
            return data_engine

    # Check next to .pt file
    engine_path = Path(pt_model).with_suffix('.engine')
    if engine_path.exists():
        log_stderr(f'Using existing TensorRT engine: {engine_path}')
        return engine_path

    # Export .engine (one-time, 5-15 minutes depending on GPU)
    log_stderr('Exporting to TensorRT (one-time, may take several minutes)...')
    if json_progress:
        emit_json('status', message='Βελτιστοποίηση για GPU (μόνο την πρώτη φορά, μπορεί να πάρει μερικά λεπτά)...')

    try:
        model = YOLO(pt_model)
        model.set_classes(prompts)  # type: ignore[misc]  # Bake class embeddings into engine
        export_dir = str(Path(data_dir) / 'models') if data_dir else None
        log_stderr(f'TensorRT export: imgsz=640, half=True, device=0, output={export_dir or "same dir"}')
        model.export(format="engine", half=True, dynamic=False, batch=1,
                     imgsz=640, device=0, simplify=True, project=export_dir)

        # Verify the export created a file
        if data_dir:
            data_engine = Path(data_dir) / 'models' / engine_name
            if data_engine.exists():
                log_stderr(f'TensorRT export successful: {data_engine} ({data_engine.stat().st_size // 1024 // 1024}MB)')
                return data_engine
        if engine_path.exists():
            log_stderr(f'TensorRT export successful: {engine_path}')
            return engine_path

        log_stderr('TensorRT export completed but .engine file not found')
        return None
    except Exception as e:
        log_stderr(f'TensorRT export failed: {e}')
        if json_progress:
            emit_json('status', message=f'TensorRT export αποτυχία: {e}. Χρήση PyTorch CUDA.')
        return None


def warmup_gpu(model, device, imgsz=640):
    """Warm up GPU using ultralytics built-in warmup (eliminates first-frame penalty)."""
    if device != 'cuda':
        return
    try:
        model.warmup(imgsz=(1, 3, imgsz, imgsz))
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    except Exception:
        pass  # Non-critical


def process_video(source: str, yolo: YOLO, cfg: dict, args, video_index: int = 1,
                  total_videos: int = 1, defer_report: bool = False) -> tuple:
    """Process a single video. Returns (report_path, tracks).

    defer_report=True (cross-video match mode): skip report generation and
    return the tracks with embeddings/snapshots INTACT — the caller matches
    across videos first and writes all reports afterwards."""
    source_path = Path(source)
    json_progress = getattr(args, 'json_progress', False)

    if json_progress:
        emit_json('status', message=f'Processing: {source_path.name}')
    else:
        print(f'\nProcessing: {source_path.name}')

    # Get settings
    conf = args.conf or cfg['confidence']
    save_crops = args.crops or cfg['save_crops']
    save_report = cfg['save_report']
    save_video = args.video or cfg['save_video']
    half = args.half or cfg['half']
    imgsz = args.imgsz or cfg['imgsz']
    stride = args.stride or cfg['vid_stride']
    show = args.show or cfg['show']

    # Device selection
    model_path = str(getattr(yolo, 'ckpt_path', ''))
    device, half = select_device_and_half(model_path, half)

    # Get video info
    cap = cv2.VideoCapture(source)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_metadata = cap.get(cv2.CAP_PROP_FPS)

    # Calculate actual fps (metadata is often wrong for security cameras)
    frames_to_check = 0
    while frames_to_check < 100:
        ret, _ = cap.read()
        if not ret:
            break
        frames_to_check += 1
        pos_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if pos_sec >= 5:  # Check first 5 seconds
            break

    if pos_sec > 0 and frames_to_check > 0:
        fps = frames_to_check / pos_sec
    else:
        fps = fps_metadata

    cap.release()

    if total_frames == 0:
        # Video file may be corrupt, empty, or in unsupported format
        log_stderr(f'Skipping {source_path.name}: could not read video (0 frames detected)')
        if json_progress:
            emit_json('status', message=f'Παράβλεψη {source_path.name}: δεν μπόρεσε να αναγνωριστεί')
        return None

    # Output directory for YOLO (only if saving video/crops)
    output_dir = args.output or str(source_path.parent)

    # Resolve tracker path
    tracker_path = getattr(args, '_tracker_path', 'tracker.yaml')

    # Run tracking
    results = yolo.track(
        source=source,
        conf=conf,
        device=device,
        persist=True,
        save=save_video,
        save_crop=save_crops,
        project=output_dir if save_video or save_crops else None,
        name='' if save_video or save_crops else None,
        half=half,
        imgsz=imgsz,
        vid_stride=stride,
        show=show,
        stream=True,
        verbose=False,
        tracker=tracker_path,
        max_det=300,
        # Closed-set mode: restrict COCO classes to the fixed scope
        classes=cfg.get('_closed_classes'),
    )

    # Track detections for report — shared collector (best-K snapshots +
    # position stats for the stitching pass), see tracking.py.
    collector = TrackCollector(max_snapshots=int(cfg.get('snapshots', 4)))
    frame_num = 0
    total_processed = total_frames // stride
    last_progress_time = 0.0
    import time as _time

    # Use tqdm for CLI, JSON events for GUI
    if json_progress:
        results_iter = results
    else:
        results_iter = tqdm(results, total=total_processed, desc='  Analyzing', unit='frame')

    processing_start = _time.monotonic()
    frames_since_cache_clear = 0

    try:
      for result in results_iter:
        frame_num += 1
        frames_since_cache_clear += 1

        # Emit progress for GUI (every 0.5s for responsive updates)
        if json_progress:
            now = _time.monotonic()
            if now - last_progress_time >= 0.5 or frame_num == total_processed:
                elapsed = now - processing_start
                processing_fps = frame_num / elapsed if elapsed > 0 else 0.0
                emit_json('progress',
                    video=source_path.name,
                    frame=frame_num,
                    total_frames=total_processed,
                    video_index=video_index,
                    total_videos=total_videos,
                    fps=round(processing_fps, 1)
                )
                last_progress_time = now

        if save_report and result.boxes is not None and len(result.boxes):
            for i in range(len(result.boxes)):
                box = result.boxes.xyxy[i]
                det_conf = float(result.boxes.conf[i])
                cls_id = int(result.boxes.cls[i])
                class_name = result.names[cls_id]

                # Get track ID if available (skip untracked detections)
                if result.boxes.id is None or result.boxes.id[i] is None:
                    continue
                track_id = int(result.boxes.id[i])

                timestamp = (frame_num * stride) / fps
                collector.add(track_id=track_id, class_name=class_name,
                              conf=det_conf, box_xyxy=box.tolist(),
                              frame_img=result.orig_img, timestamp=timestamp,
                              frame_idx=frame_num * stride)

        # Periodic VRAM cleanup (reduces fragmentation on NVIDIA)
        if device == 'cuda' and frames_since_cache_clear >= 30:
            try:
                torch.cuda.empty_cache()
                frames_since_cache_clear = 0
            except Exception:
                pass

    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            log_stderr(f'GPU out of memory at frame {frame_num}/{total_processed} — generating report with partial data')
            if json_progress:
                emit_json('status', message=f'GPU memory full at frame {frame_num} — report will contain partial data')
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        else:
            raise

    # Post-process: derived fields, then the offline stitching pass merges
    # fragmented tracklets of the same physical object (ROADMAP Phase A).
    tracks = collector.finalize()
    n_raw = len(tracks)
    tracks = run_stitching(tracks, cfg, json_progress)
    run_auto_plates(tracks, cfg, json_progress, fps=fps, video_path=source)
    run_face_shots(tracks, cfg, json_progress)
    run_attributes(tracks, cfg, json_progress)

    log_stderr(f'Processing complete: {source_path.name} — {n_raw} tracklets '
               f'-> {len(tracks)} objects, {frame_num} frames analyzed')

    if defer_report:
        return None, tracks

    tracks = prepare_for_report(tracks)

    # Generate HTML report
    report_path = None
    if save_report and tracks:
        video_name = source_path.stem  # filename without extension
        output_dir = args.output or str(source_path.parent)
        report_path = generate_report(tracks, output_dir, video_name, source)
        log_stderr(f'Report saved: {report_path}')
        if json_progress:
            emit_json('report', path=report_path)
        else:
            print(f'  Report: {report_path}')

    return report_path, None


def run_match_mode(video_files: list, yolo: YOLO, cfg: dict, args) -> list:
    """Cross-video match mode (ROADMAP Phase Ε): process every video with
    deferred reports, match objects across videos (plates + appearance),
    re-vote plates over the union of each group's snapshots, then write the
    per-video reports plus the combined match report."""
    json_progress = getattr(args, 'json_progress', False)
    per_video: dict = {}
    sources: dict = {}
    for idx, source in enumerate(video_files, 1):
        _, tracks = process_video(source, yolo, cfg, args, video_index=idx,
                                  total_videos=len(video_files),
                                  defer_report=True)
        name = Path(source).name
        per_video[name] = tracks or {}
        sources[name] = source
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if json_progress:
        emit_json('status', message='Αντιστοίχιση αντικειμένων μεταξύ βίντεο...')
    import cross_match
    groups = cross_match.match_videos(per_video)

    # Combined plate: multiple viewpoints break the per-camera systematic
    # blur that made single-camera votes confidently wrong.
    reader = None
    for g in groups:
        member_tracks = [per_video[v][tid] for v, tid in g['members']]
        if not any(k in member_tracks[0]['class'].lower()
                   for k in VEHICLE_KEYWORDS):
            continue
        try:
            if reader is None:
                from plate_core import PlateReader
                reader = PlateReader()
            combo = cross_match.combined_plate(member_tracks, reader)
            if combo:
                g['combined_plate'] = combo
        except Exception as e:  # noqa: BLE001
            log_stderr(f'Combined plate vote failed ({e})')
    log_stderr(f'Cross-match: {len(groups)} matched objects across '
               f'{len(video_files)} videos')

    # Reports: per-video first (prepare_for_report encodes the snapshots the
    # match report also embeds), then the combined match report.
    reports = []
    output_dir = args.output or str(Path(video_files[0]).parent)
    for name, tracks in per_video.items():
        if not tracks:
            continue
        prepare_for_report(tracks)
        rp = generate_report(tracks, output_dir, Path(name).stem, sources[name])
        reports.append(rp)
        if json_progress:
            emit_json('report', path=rp)
    from match_report import generate_match_report
    mp = generate_match_report(per_video, groups, output_dir,
                               list(per_video.keys()))
    reports.append(mp)
    if json_progress:
        emit_json('report', path=mp)
    else:
        print(f'  Match report: {mp}')
    return reports


def get_video_fps(source: str) -> tuple[int, float]:
    """Get total frames and actual fps for a video"""
    cap = cv2.VideoCapture(source)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_metadata = cap.get(cv2.CAP_PROP_FPS)

    # Calculate actual fps (metadata is often wrong for security cameras)
    frames_to_check = 0
    pos_sec = 0
    while frames_to_check < 100:
        ret, _ = cap.read()
        if not ret:
            break
        frames_to_check += 1
        pos_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if pos_sec >= 5:
            break

    if pos_sec > 0 and frames_to_check > 0:
        fps = frames_to_check / pos_sec
    else:
        fps = fps_metadata

    cap.release()
    return total_frames, fps


def process_video_chain(sources: list[str], yolo: YOLO, cfg: dict, args) -> str | None:
    """Process multiple videos as a continuous chain with persistent tracking"""
    json_progress = getattr(args, 'json_progress', False)

    # Get settings
    conf = args.conf or cfg['confidence']
    imgsz = args.imgsz or cfg['imgsz']
    stride = args.stride or cfg['vid_stride']
    half = args.half or cfg['half']
    save_report = cfg['save_report']
    tracker_path = getattr(args, '_tracker_path', 'tracker.yaml')

    # Device selection
    model_path = str(getattr(yolo, 'ckpt_path', ''))
    device, half = select_device_and_half(model_path, half)

    # Calculate total frames across all videos
    video_info = []
    total_all_frames = 0
    for source in sources:
        total_frames, fps = get_video_fps(source)
        video_info.append({'source': source, 'frames': total_frames, 'fps': fps})
        total_all_frames += total_frames

    if json_progress:
        emit_json('status', message=f'Chain mode: {len(sources)} videos, {total_all_frames} total frames')
    else:
        print(f'\nChain mode: Processing {len(sources)} videos as continuous sequence')
        print(f'Total frames: {total_all_frames}')

    # Track detections across all videos — shared collector (see tracking.py)
    collector = TrackCollector(max_snapshots=int(cfg.get('snapshots', 4)))
    global_frame_num = 0
    cumulative_time = 0.0
    last_progress_time = 0.0
    import time as _time

    # Progress bar for all videos combined
    total_to_process = total_all_frames // stride
    if not json_progress:
        pbar = tqdm(total=total_to_process, desc='  Analyzing chain', unit='frame')

    processing_start = _time.monotonic()
    for video_idx, info in enumerate(video_info):
        source = info['source']
        fps = info['fps']
        source_name = Path(source).name

        if json_progress:
            emit_json('status', message=f'Processing: {source_name} ({video_idx + 1}/{len(sources)})')
        else:
            pbar.set_postfix({'file': source_name[:20]})

        # Use stream mode for optimized video reading (persist=True keeps tracker state)
        results = yolo.track(
            source=source,
            conf=conf,
            device=device,
            persist=True,
            save=False,
            save_crop=False,
            half=half,
            imgsz=imgsz,
            vid_stride=stride,
            stream=True,
            verbose=False,
            tracker=tracker_path,
            max_det=300,
            classes=cfg.get('_closed_classes'),
        )

        frame_in_video = 0
        for result in results:
            frame_in_video += 1
            global_frame_num += 1

            # Update progress (every 0.5s for responsive updates)
            if json_progress:
                now = _time.monotonic()
                if now - last_progress_time >= 0.5 or global_frame_num == total_to_process:
                    elapsed = now - processing_start
                    processing_fps = global_frame_num / elapsed if elapsed > 0 else 0.0
                    emit_json('progress',
                        video=source_name,
                        frame=global_frame_num,
                        total_frames=total_to_process,
                        video_index=video_idx + 1,
                        total_videos=len(sources),
                        fps=round(processing_fps, 1)
                    )
                    last_progress_time = now
            else:
                pbar.update(1)

            # Process detections
            if save_report and result.boxes is not None and len(result.boxes):
                for i in range(len(result.boxes)):
                    box = result.boxes.xyxy[i]
                    det_conf = float(result.boxes.conf[i])
                    cls_id = int(result.boxes.cls[i])
                    class_name = result.names[cls_id]

                    # Get track ID (skip untracked)
                    if result.boxes.id is None or result.boxes.id[i] is None:
                        continue
                    track_id = int(result.boxes.id[i])

                    # Local timestamp within this video
                    local_timestamp = (frame_in_video * stride) / fps
                    collector.add(track_id=track_id, class_name=class_name,
                                  conf=det_conf, box_xyxy=box.tolist(),
                                  frame_img=result.orig_img,
                                  timestamp=local_timestamp,
                                  source_name=source_name,
                                  frame_idx=frame_in_video * stride)

        # Add this video's duration to cumulative time
        cumulative_time += info['frames'] / fps

    if not json_progress:
        pbar.close()

    # Post-process + offline stitching (shared with single-video mode).
    tracks = collector.finalize()
    n_raw = len(tracks)
    tracks = run_stitching(tracks, cfg, json_progress)
    run_auto_plates(tracks, cfg, json_progress)
    run_face_shots(tracks, cfg, json_progress)
    run_attributes(tracks, cfg, json_progress)
    tracks = prepare_for_report(tracks)
    log_stderr(f'Chain complete: {n_raw} tracklets -> {len(tracks)} objects')

    # Generate combined report
    report_path = None
    if save_report and tracks:
        # Use first video's directory and name for report
        first_video = Path(sources[0])
        output_dir = args.output or str(first_video.parent)
        report_name = f"{first_video.stem}_chain"
        report_path = generate_report(tracks, output_dir, report_name, sources[0])
        if json_progress:
            emit_json('report', path=report_path)
        else:
            print(f'  Report: {report_path}')

    return report_path


def _parallel_worker(worker_args: dict) -> str | None:
    """Worker function for parallel video processing. Runs in a separate process."""
    source = worker_args['source']
    model_path = worker_args['model_path']
    prompts = worker_args['prompts']
    using_optimized = worker_args['using_optimized']
    cfg = worker_args['cfg']
    args = worker_args['args']
    video_index = worker_args['video_index']
    total_videos = worker_args['total_videos']

    # Each worker loads its own model instance
    yolo = YOLO(model_path)
    # Closed-set mode (cfg['_closed_classes'] set): plain YOLO26, no prompts.
    if not using_optimized and cfg.get('_closed_classes') is None:
        yolo.set_classes(prompts)  # type: ignore[misc]

    report_path, _ = process_video(source, yolo, cfg, args,
                                   video_index=video_index,
                                   total_videos=total_videos)
    return report_path


def main():
    parser = argparse.ArgumentParser(
        description='VisionX - YOLOE-26 Video Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python vision.py video.mp4
  python vision.py video1.mp4 video2.mp4 video3.mp4
  python vision.py --dir /path/to/videos
  python vision.py --search "white car" video.mp4
        '''
    )
    parser.add_argument('source', nargs='*', help='Video file(s)')
    parser.add_argument('--dir', help='Process all videos in directory')
    parser.add_argument('--search', nargs='+', help='Custom prompts')
    parser.add_argument('--model', help='Model file')
    parser.add_argument('--conf', type=float, help='Confidence threshold')
    parser.add_argument('--output', '-o', help='Output directory')
    parser.add_argument('--crops', action='store_true', help='Save cropped detections')
    parser.add_argument('--video', action='store_true', help='Save annotated video')
    parser.add_argument('--half', action='store_true', help='FP16 mode (faster)')
    parser.add_argument('--stride', type=int, help='Frame skip (2=2x faster)')
    parser.add_argument('--imgsz', type=int, help='Inference image size (default 640)')
    parser.add_argument('--show', action='store_true', help='Live preview')
    parser.add_argument('--parallel', type=int, default=1, help='Number of parallel workers')
    parser.add_argument('--chain', action='store_true', help='Process multiple videos as continuous sequence')
    parser.add_argument('--match', action='store_true',
                        help='Cross-video matching: same event from multiple '
                             'cameras — match objects across the videos')
    # Phase A-Γ feature toggles (all default ON via config; these disable).
    parser.add_argument('--no-stitch', action='store_true',
                        help='Disable offline tracklet stitching')
    parser.add_argument('--no-plates', action='store_true',
                        help='Disable automatic license-plate reading')
    parser.add_argument('--no-faces', action='store_true',
                        help='Disable face best-shot extraction')
    parser.add_argument('--config', default=None, help='Config file')
    parser.add_argument('--resource-dir', default=None, help='Bundled resources directory (set by Tauri)')
    parser.add_argument('--data-dir', default=None, help='Writable app data directory (set by Tauri)')
    parser.add_argument('--json-progress', action='store_true', help='Output JSON progress (for GUI)')
    args = parser.parse_args()

    # Resolve resource directories
    resource_dir = args.resource_dir
    data_dir = args.data_dir

    # Ensure data dir exists (writable location for downloads, CoreML exports)
    if data_dir:
        os.makedirs(os.path.join(data_dir, 'models'), exist_ok=True)

    # Log startup info
    log_stderr(f'VisionX vision.py starting')
    log_stderr(f'  Python: {sys.version}')
    log_stderr(f'  PyTorch: {torch.__version__}')
    log_stderr(f'  CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        log_stderr(f'  GPU: {torch.cuda.get_device_name()}')
        log_stderr(f'  CUDA version: {torch.version.cuda}')
        log_stderr(f'  Tensor Cores: {_has_tensor_cores()}')
    log_stderr(f'  Resource dir: {resource_dir}')
    log_stderr(f'  Data dir: {data_dir}')

    # Resolve config path: explicit > resource dir > CWD fallback
    config_path = args.config
    if not config_path:
        if resource_dir:
            config_path = os.path.join(resource_dir, 'config.yaml')
        else:
            config_path = 'config.yaml'

    # Resolve tracker path
    tracker_path = 'tracker.yaml'
    if resource_dir:
        candidate = os.path.join(resource_dir, 'tracker.yaml')
        if os.path.exists(candidate):
            tracker_path = candidate

    log_stderr(f'  Config: {config_path}')
    log_stderr(f'  Tracker: {tracker_path}')

    # Store resolved paths in args for use by processing functions
    args._tracker_path = tracker_path

    # Load config
    cfg = load_config(config_path)
    # Stashed for downstream model resolution (stitching embed model) —
    # processing functions receive cfg but not the arg dirs.
    cfg['_data_dir'] = data_dir
    cfg['_resource_dir'] = resource_dir
    # CLI feature toggles override config (used by the GUI's advanced panel).
    if args.no_stitch:
        cfg['stitch'] = False
    if args.no_plates:
        cfg['plates'] = False
    if args.no_faces:
        cfg['faces'] = False

    # Collect video files
    video_files = []

    if args.dir:
        dir_path = Path(args.dir)
        if dir_path.is_dir():
            # Support common video formats (lowercase and uppercase)
            extensions = [
                'mp4', 'MP4', 'm4v', 'M4V',
                'avi', 'AVI',
                'mov', 'MOV',
                'mkv', 'MKV',
                'wmv', 'WMV',
                'flv', 'FLV',
                'webm', 'WEBM',
                'mpeg', 'MPEG', 'mpg', 'MPG',
                'ts', 'TS', 'mts', 'MTS', 'm2ts', 'M2TS',
                '3gp', '3GP',
                'asf', 'ASF',
                'dav', 'DAV',  # Dahua DVR format
            ]
            video_files = []
            for ext in extensions:
                video_files.extend(dir_path.glob(f'*.{ext}'))
            video_files = [str(f) for f in sorted(set(video_files))]
        else:
            print(f'Error: {args.dir} is not a directory')
            return

    if args.source:
        video_files.extend(args.source)

    if not video_files:
        parser.print_help()
        return

    # Automatic model selection (see BENCHMARKS.md, quality-first): without
    # custom prompts the scope is FIXED (people + all vehicle types), where
    # closed-set YOLO26 both tracks more stably (yoloe-26x produced the most
    # fragmented tracks of all candidates) AND runs ~2x faster. Open-vocab
    # YOLOE loads only when the user actually searches free text. User-driven
    # design: attributes (color etc.) are computed AFTERWARDS on the results,
    # not by widening the detector's vocabulary.
    closed_mode = not args.search
    if args.model:
        model_path = args.model
        closed_mode = 'yoloe' not in Path(args.model).name.lower()
    elif closed_mode:
        model_path = cfg.get('model_closed', 'yolo26l.pt')
    else:
        model_path = cfg['model']
    prompts = args.search or cfg['prompts']
    # COCO ids for the fixed scope: person bicycle car motorcycle bus truck
    cfg['_closed_classes'] = [0, 1, 2, 3, 5, 7] if closed_mode else None
    json_progress = getattr(args, 'json_progress', False)
    log_stderr(f'Model selection: {"closed-set " + Path(model_path).name if closed_mode else "open-vocabulary " + Path(model_path).name} '
               f'(custom prompts: {bool(args.search)})')

    # Resolve model path: check data dir (downloaded), then resource dir (bundled), then CWD
    if not Path(model_path).is_absolute():
        for base in [data_dir, resource_dir]:
            if base:
                candidate = os.path.join(base, 'models', model_path)
                if os.path.exists(candidate):
                    model_path = candidate
                    break

    # If model still not found, it will be auto-downloaded by ultralytics
    if not Path(model_path).exists():
        if json_progress:
            emit_json('model_download', model=Path(model_path).name, status='starting',
                      message=f'Downloading {Path(model_path).name}...')
        else:
            print(f'Model not found locally. Downloading {Path(model_path).name}...')
        # Set model path to data dir so ultralytics downloads there
        if data_dir:
            download_dest = os.path.join(data_dir, 'models')
            os.makedirs(download_dest, exist_ok=True)
            model_path = os.path.join(download_dest, Path(model_path).name)

    # Platform-specific model optimization (one-time exports)
    using_optimized = False

    if closed_mode:
        # The export helpers bake YOLOE prompt embeddings — not applicable to
        # plain YOLO26. PyTorch MPS/CUDA is already ~2x faster than the old
        # default; CoreML/TensorRT export for YOLO26 is a follow-up.
        log_stderr('Closed-set mode: using PyTorch runtime (no baked export)')
    elif sys.platform == 'darwin':
        # macOS: CoreML for Neural Engine acceleration
        coreml_path = get_coreml_model_path(model_path)
        if coreml_path is None and data_dir:
            data_coreml = Path(data_dir) / 'models' / Path(model_path).with_suffix('.mlpackage').name
            if data_coreml.exists():
                coreml_path = data_coreml
        if coreml_path is None:
            coreml_path = export_to_coreml(model_path, prompts, json_progress, data_dir)
        if coreml_path and coreml_path.exists():
            model_path = str(coreml_path)
            using_optimized = True

    elif torch.cuda.is_available():
        # Windows/Linux: TensorRT for NVIDIA GPU acceleration
        trt_path = export_to_tensorrt(model_path, prompts, json_progress, data_dir)
        if trt_path and trt_path.exists():
            model_path = str(trt_path)
            using_optimized = True

    # Determine device name for display
    if model_path.endswith('.mlpackage'):
        device_name = 'CoreML (Neural Engine)'
    elif model_path.endswith('.engine'):
        gpu_name = torch.cuda.get_device_name() if torch.cuda.is_available() else 'GPU'
        device_name = f'TensorRT ({gpu_name})'
    elif torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        fp16_status = '+ FP16' if _has_tensor_cores() else ''
        device_name = f'CUDA ({gpu_name}) {fp16_status}'.strip()
    elif torch.backends.mps.is_available():
        device_name = 'MPS (Metal)'
    else:
        device_name = 'CPU'

    if json_progress:
        emit_json('status', message=f'Επεξεργαστής: {device_name}')
        emit_json('status', message=f'Φόρτωση μοντέλου AI ({Path(model_path).name})...')
    else:
        print(f'Using: {device_name}')
        print(f'Loading {Path(model_path).name}...')

    # Load model once
    log_stderr(f'Loading model: {model_path}')
    log_stderr(f'  Device: {device_name}')
    log_stderr(f'  Optimized: {using_optimized}')
    yolo = YOLO(model_path)
    log_stderr(f'  Model loaded successfully')
    if not using_optimized and not closed_mode:
        if json_progress:
            emit_json('status', message='Προετοιμασία αναγνώρισης αντικειμένων...')
        yolo.set_classes(prompts)  # type: ignore[misc]  # Optimized models have classes baked in

    # GPU warmup (eliminates slow first frame)
    if torch.cuda.is_available():
        imgsz_val = args.imgsz or cfg['imgsz']
        if json_progress:
            emit_json('status', message='Προθέρμανση GPU...')
        warmup_gpu(yolo, 'cuda', imgsz_val)

    scope_label = ('άνθρωποι + οχήματα (όλοι οι τύποι)' if closed_mode
                   else ', '.join(prompts))
    if json_progress:
        emit_json('status', message=f'Αναζήτηση: {scope_label}')
        emit_json('status', message=f'{len(video_files)} βίντεο προς ανάλυση')
    else:
        print(f'Detecting: {scope_label}')
        print(f'\nFound {len(video_files)} video(s) to process')

    # Process videos
    reports = []
    total_videos = len(video_files)
    parallel = args.parallel

    if args.chain and len(video_files) > 1:
        # Chain mode: process all videos as continuous sequence
        report_path = process_video_chain(video_files, yolo, cfg, args)
        if report_path:
            reports.append(report_path)
    elif getattr(args, 'match', False) and total_videos > 1:
        # Cross-video match mode (sequential by design: tracks from all
        # videos must coexist in memory for the matching pass).
        reports.extend(run_match_mode(video_files, yolo, cfg, args))
    elif parallel > 1 and total_videos > 1:
        # Parallel mode: process videos concurrently
        workers = min(parallel, total_videos)
        if json_progress:
            emit_json('status', message=f'Parallel processing: {workers} workers')

        worker_args_list = [
            {
                'source': source,
                'model_path': model_path,
                'prompts': prompts,
                'using_optimized': using_optimized,
                'cfg': cfg,
                'args': args,
                'video_index': idx,
                'total_videos': total_videos,
            }
            for idx, source in enumerate(video_files, 1)
        ]

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_parallel_worker, wa): wa['source'] for wa in worker_args_list}
            for future in as_completed(futures):
                try:
                    report_path = future.result()
                    if report_path:
                        reports.append(report_path)
                except Exception as e:
                    source = futures[future]
                    if json_progress:
                        emit_json('status', message=f'Error processing {Path(source).name}: {e}')
                    else:
                        print(f'Error processing {Path(source).name}: {e}')
    else:
        # Normal mode: process each video independently
        for idx, source in enumerate(video_files, 1):
            report_path, _ = process_video(source, yolo, cfg, args, video_index=idx, total_videos=total_videos)
            if report_path:
                reports.append(report_path)
            # Free GPU cache between videos to prevent OOM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Summary
    if json_progress:
        emit_json('complete', videos_processed=len(video_files), reports_generated=len(reports))
    else:
        print(f'\n{"="*50}')
        print(f'Processed {len(video_files)} video(s)')
        if args.chain and len(video_files) > 1:
            print(f'Chain mode: combined into 1 report')
        if reports:
            print(f'Generated {len(reports)} report(s)')


if __name__ == '__main__':
    main()
