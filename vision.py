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
from tracking import TrackCollector, is_host_geometry, prepare_for_report
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
    'snapshots': 6,
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


def run_prompt_filter(tracks: dict, cfg: dict, args, json_progress: bool):
    """Structured result filters (see prompt_filter.py) — fixed color/type
    choices only; free text was deliberately removed."""
    colors = getattr(args, 'filter_color', None)
    types = getattr(args, 'filter_type', None)
    if not (colors or types) or not tracks:
        return
    try:
        import prompt_filter
        if json_progress:
            emit_json('status', message='Φιλτράρισμα αποτελεσμάτων με τα κριτήρια...')
        prompt_filter.apply_filters(tracks, colors, types, log=log_stderr)
    except Exception as e:  # noqa: BLE001
        log_stderr(f'Filtering failed ({e})')


# Live-preview plate outlines: lazy detector + quad estimator, loaded on the
# first preview tick. 'failed' latches after any error so a broken optional
# stack can never slow or break the analysis loop.
_PREVIEW_PLATES: dict = {'det': None, 'mod': None, 'failed': False}


def _is_host_box(result, i: int) -> bool:
    """Per-frame host-vehicle check — geometry shared with tracking.py
    (single definition of the signature constants)."""
    try:
        if int(result.boxes.cls[i]) not in (2, 3, 5, 7):
            return False
        x1, y1, x2, y2 = (float(v) for v in result.boxes.xyxy[i])
        fh, fw = result.orig_img.shape[:2]
        return is_host_geometry(x1, y1, x2, y2, fw, fh)
    except Exception:  # noqa: BLE001
        return False


def draw_plate_quads(annotated: np.ndarray, result) -> None:
    """Draw the TRUE plate outline (yellow quad) on the live preview frame —
    mirrors the interactive plate tool (user request: not a screen-aligned
    rectangle, the plate's real frame even at an angle). Runs ONLY on
    preview ticks (~2Hz) and only on the 4 largest vehicles, so the cost
    never touches per-frame analysis throughput."""
    if _PREVIEW_PLATES['failed']:
        return
    try:
        if _PREVIEW_PLATES['det'] is None:
            import plate as plate_mod
            # Shared factory (CPU-provider rationale lives with it — DRY).
            # conf 0.2: the preview draws instantly, so it prefers fewer
            # false quads over recall; the analysis keeps its own 0.15.
            _PREVIEW_PLATES['det'] = plate_mod.make_plate_detector(conf=0.2)
            _PREVIEW_PLATES['mod'] = plate_mod
        det = _PREVIEW_PLATES['det']
        pm = _PREVIEW_PLATES['mod']
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return
        vehicles = []  # COCO: 2 car, 3 motorcycle, 5 bus, 7 truck
        for i in range(len(boxes)):
            if int(boxes.cls[i]) in (2, 3, 5, 7):
                x1, y1, x2, y2 = (int(v) for v in boxes.xyxy[i])
                vehicles.append(((x2 - x1) * (y2 - y1), x1, y1, x2, y2))
        for _, x1, y1, x2, y2 in sorted(vehicles, reverse=True)[:4]:
            crop = result.orig_img[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                continue
            dets = det.predict(crop)
            if not dets:
                continue
            d = max(dets, key=lambda x: float(x.confidence))
            bb = d.bounding_box
            # Plate-geometry sanity (field case: the dashcam's own hood is a
            # "vehicle" and its timestamp/logo bar detected as a plate): a
            # real plate is a FRACTION of its vehicle's width with plate-like
            # aspect — overlay bars span most of the crop.
            pw, ph = float(bb.x2 - bb.x1), float(bb.y2 - bb.y1)
            if ph <= 0 or pw > 0.55 * crop.shape[1]                     or not (1.5 <= pw / ph <= 8.0):
                continue
            px1, py1 = max(0, int(bb.x1)), max(0, int(bb.y1))
            pcrop = crop[py1:int(bb.y2), px1:int(bb.x2)]
            if pcrop.size == 0:
                continue
            quad, _ = pm.estimate_quad(pcrop, None)
            if quad is None:
                quad = pm.bright_plate_quad(pcrop)  # saturated night plates
            if quad is not None:
                pts = np.array([[int(qx) + px1 + x1, int(qy) + py1 + y1]
                                for qx, qy in quad])
            else:
                # No quad estimable — the detector's box is still the
                # plate's frame, just axis-aligned.
                pts = np.array([[x1 + px1, y1 + py1],
                                [x1 + int(bb.x2), y1 + py1],
                                [x1 + int(bb.x2), y1 + int(bb.y2)],
                                [x1 + px1, y1 + int(bb.y2)]])
            cv2.polylines(annotated, [pts], True, (0, 255, 255), 2)
    except Exception:  # noqa: BLE001 — preview extras must never break runs
        _PREVIEW_PLATES['failed'] = True


def run_reappearance(tracks: dict, json_progress: bool):
    """Annotate possible same-video re-appearances (vehicle/person left and
    came back). Runs AFTER plates (strong evidence needs candidate lists)
    and BEFORE prepare_for_report (appearance needs the _emb_* embeddings
    that stitching left on the tracks). Annotation-only by design — the
    report links the cards, the investigator decides."""
    if len(tracks) < 2:
        return
    try:
        from cross_match import find_reappearances
        pairs = find_reappearances(tracks)
        for p in pairs:
            tracks[p['a']].setdefault('reappearance', []).append(
                {'other': p['b'], 'when': 'later', 'gap': p['gap'],
                 'score': p['score'], 'evidence': p['evidence']})
            tracks[p['b']].setdefault('reappearance', []).append(
                {'other': p['a'], 'when': 'earlier', 'gap': p['gap'],
                 'score': p['score'], 'evidence': p['evidence']})
        if pairs:
            log_stderr(f'Re-appearance: {len(pairs)} possible same-object '
                       f'pair(s) flagged')
            if json_progress:
                emit_json('status',
                          message=f'Πιθανές επανεμφανίσεις: {len(pairs)}')
    except Exception as e:  # noqa: BLE001 — annotation must never kill a run
        log_stderr(f'Re-appearance pass failed ({e})')


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


def closed_set_trt_engine(pt_model: str, data_dir: str | None,
                          req_imgsz: int, json_progress: bool) -> str | None:
    """DYNAMIC TensorRT engine for the closed-set model (NVIDIA only).

    Design decisions (user-driven):
    - dynamic=True: ONE engine per machine covers every imgsz up to 1920 —
      changing the resolution or running new videos never rebuilds (the
      build is ~10 min once, cached forever in the data models dir).
    - FP32 (half=False): outputs must match the PyTorch reference as closely
      as possible — same detections → same tracks (evidentiary consistency
      beats the extra FP16 speed).
    - ANY failure returns None and the caller stays on PyTorch — never
      worse than today. VISIONX_NO_TRT=1 disables explicitly.
    """
    if not torch.cuda.is_available() or os.environ.get('VISIONX_NO_TRT') == '1':
        return None
    if req_imgsz and not (960 <= req_imgsz <= 1920):
        # The dynamic profile (built at 1920) covers [960, 1920] — the
        # ultralytics exporter sets the min shape to imgsz/2. Above it a 4K
        # profile would exceed laptop VRAM; below it PyTorch is fast anyway
        # (small inputs) and stays the reference runtime.
        log_stderr(f'imgsz {req_imgsz} outside the TensorRT profile '
                   f'[960, 1920] — PyTorch runtime for this run')
        return None
    try:
        import tensorrt as trt  # noqa: F401 — availability probe
        log_stderr(f'TensorRT version: {trt.__version__}')
    except Exception as e:  # broken wrapper OR missing — same fallback
        log_stderr(f'TensorRT unavailable ({e}) — PyTorch runtime')
        return None

    stem = Path(pt_model).stem
    cache_dirs = []
    if data_dir:
        cache_dirs.append(Path(data_dir) / 'models')
    cache_dirs.append(Path(pt_model).parent)
    for d in cache_dirs:
        cached = d / f'{stem}_dyn.engine'
        if cached.exists():
            log_stderr(f'TensorRT engine cache hit: {cached}')
            return str(cached)

    target = cache_dirs[0] / f'{stem}_dyn.engine'
    log_stderr('Building dynamic TensorRT engine (one-time, ~10 min)...')
    if json_progress:
        # model_download is the status type the app renders prominently —
        # without a visible message a 10-minute silent build reads as a hang.
        emit_json('model_download',
                  message='Βελτιστοποίηση μοντέλου για την κάρτα γραφικών '
                          '(μία φορά, ~10 λεπτά — δεν έχει κολλήσει)...')
    try:
        exported = YOLO(pt_model).export(
            format='engine', dynamic=True, imgsz=1920, half=False,
            device=0, workspace=4, batch=1, verbose=False)
        exp = Path(exported)
        target.parent.mkdir(parents=True, exist_ok=True)
        if exp.resolve() != target.resolve():
            exp.replace(target)
        log_stderr(f'TensorRT engine ready: {target}')
        if json_progress:
            emit_json('model_download',
                      message='Η βελτιστοποίηση ολοκληρώθηκε — έναρξη ανάλυσης')
        return str(target)
    except Exception as e:  # noqa: BLE001 — fall back, never fail the run
        log_stderr(f'TensorRT engine build failed ({e}) — PyTorch runtime')
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
    if not cap.isOpened():
        # Unreadable input (missing codec, corrupt file, or not a video at
        # all). Clean skip with a user-facing message — this used to crash
        # with UnboundLocalError further down.
        cap.release()
        log_stderr(f'Skipping {source_path.name}: could not open as video')
        if json_progress:
            emit_json('status', message=f'Παράβλεψη {source_path.name}: '
                                        f'δεν αναγνωρίζεται ως βίντεο')
        return None, None
    cap.release()
    # Shared probe (frame count + real fps, incl. the DVR grab-count and
    # fps fallbacks) — one implementation for every caller.
    total_frames, fps = get_video_fps(source, json_progress=json_progress)

    if total_frames == 0:
        # Video file may be corrupt, empty, or in unsupported format
        log_stderr(f'Skipping {source_path.name}: could not read video (0 frames detected)')
        if json_progress:
            emit_json('status', message=f'Παράβλεψη {source_path.name}: δεν μπόρεσε να αναγνωριστεί')
        return None, None

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
                # Live preview: the annotated frame (boxes + track IDs + conf)
                # rides the same 0.5s throttle. Downscaled + JPEG'd it is
                # ~30-60KB per event — negligible next to inference cost.
                # Suppressed under parallel workers: their stdout writes
                # exceed PIPE_BUF, so concurrent big lines could interleave
                # and corrupt the JSON stream the app parses.
                preview_ok = not (getattr(args, 'parallel', 1) > 1
                                  and total_videos > 1)
                if preview_ok:
                    try:
                        # The host car detects its own hood as a "car"
                        # glued to the bottom edge — excluded from the
                        # PREVIEW drawing only (the analysis keeps the
                        # track; the report badges it).
                        try:
                            keep = [i for i in range(len(result.boxes))
                                    if not _is_host_box(result, i)]
                            shown = (result[keep]
                                     if len(keep) < len(result.boxes) else result)
                        except Exception:
                            shown = result
                        annotated = shown.plot(line_width=2)
                        draw_plate_quads(annotated, shown)
                        h, w = annotated.shape[:2]
                        if w > 720:
                            annotated = cv2.resize(
                                annotated, (720, int(h * 720 / w)),
                                interpolation=cv2.INTER_AREA)
                        ok, buf = cv2.imencode(
                            '.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 60])
                        if ok:
                            emit_json('frame', video=source_path.name,
                                      data=base64.b64encode(buf).decode('ascii'))
                    except Exception:
                        pass  # preview is best-effort; never break analysis

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
    for t in tracks.values():
        t['_video_fps'] = fps  # playback overlay needs frame->second mapping
    n_raw = len(tracks)
    tracks = run_stitching(tracks, cfg, json_progress)
    run_auto_plates(tracks, cfg, json_progress, fps=fps, video_path=source)
    run_face_shots(tracks, cfg, json_progress)
    run_attributes(tracks, cfg, json_progress)
    run_reappearance(tracks, json_progress)
    run_prompt_filter(tracks, cfg, args, json_progress)

    log_stderr(f'Processing complete: {source_path.name} — {n_raw} tracklets '
               f'-> {len(tracks)} objects, {frame_num} frames analyzed')

    if defer_report:
        return None, tracks

    # Single-video correlation review (user request: "how do I merge the
    # same vehicle HERE?"): when possible re-appearances exist, offer the
    # SAME review screen as multi-camera runs — the accepted pairs then
    # regenerate one unified report via --finalize-match. Written BEFORE
    # prepare_for_report strips the raw snapshots the finalize needs.
    if json_progress and any(t.get('reappearance') for t in tracks.values()):
        # Deduplicate the (a, b) / (b, a) mirror annotations into pairs, then
        # delegate ALL serialization to the shared writer.
        vname = source_path.name
        seen_pairs: set = set()
        uncertain_ui = []
        for tid, t in tracks.items():
            for r in t.get('reappearance') or []:
                key = tuple(sorted((tid, r['other'])))
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                uncertain_ui.append({
                    'members': [[vname, key[0]], [vname, key[1]]],
                    'evidence': r['evidence'] + f' · κενό {r["gap"]:.0f}s',
                    'score': r['score'],
                })
        write_review_session({vname: tracks}, {vname: str(source_path)},
                             args.output or str(source_path.parent),
                             groups=[], uncertain=uncertain_ui,
                             prefix=f'{source_path.stem}_review',
                             json_progress=json_progress)

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



def write_review_session(per_video: dict, sources: dict, out_dir: str,
                         groups: list, uncertain: list, prefix: str,
                         json_progress: bool) -> None:
    """Persist a correlation-review session (the ONE implementation — used
    by multi-camera match mode and by the single-video re-appearance flow;
    an earlier copy-paste of this block is exactly what the DRY audit
    flagged).

    Writes two artifacts to out_dir:
      {prefix}_session.pkl  — full pre-report tracks (raw snapshot bytes),
                              shallow-copied HERE so a later
                              prepare_for_report on the originals cannot
                              strip what --finalize-match needs;
      {prefix}_review.json  — light UI payload (one thumbnail per track)
                              for the review screen.
    Emits the match_review event so the app opens the review screen.
    Failures only log: the review flow is additive, never fatal.
    """
    session_pkl = str(Path(out_dir) / f'{prefix}_session.pkl')
    review_json = str(Path(out_dir) / f'{prefix}_review.json')

    def group_ui(g: dict) -> dict:
        # Normalize both shapes callers produce: cross_match groups carry
        # (video, tid) tuples + a combined_plate DICT; re-appearance pairs
        # carry [video, tid] lists + no combined plate. The Greek evidence
        # label ships READY-MADE (shared table in cross_match) so the Vue
        # review screen renders it verbatim instead of translating.
        from cross_match import evidence_label_el
        combo = g.get('combined_plate')
        label, tier = evidence_label_el(g.get('evidence'))
        return {'members': [[v, tid] for v, tid in g['members']],
                'evidence': g.get('evidence'),
                'evidence_label': label,
                'evidence_tier': tier,
                'score': g.get('score'),
                'combined_plate': (combo.get('plate')
                                   if isinstance(combo, dict) else combo)}

    try:
        import pickle
        copies = {v: {tid: dict(t) for tid, t in tracks.items()}
                  for v, tracks in per_video.items()}
        with open(session_pkl, 'wb') as f:
            pickle.dump({'sources': {n: str(p2) for n, p2 in sources.items()},
                         'per_video': copies}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        ui = {'videos': {n: str(p2) for n, p2 in sources.items()},
              'session': session_pkl,
              'groups': [group_ui(g) for g in groups],
              'uncertain': [group_ui(g) for g in uncertain],
              'tracks': {}}
        for vname, tracks in copies.items():
            ui['tracks'][vname] = {}
            for tid, t in tracks.items():
                snaps = t.get('snapshots') or []
                ui['tracks'][vname][str(tid)] = {
                    'class': t['class'],
                    'first_seen': round(t['first_seen'], 1),
                    'last_seen': round(t['last_seen'], 1),
                    'static': bool(t.get('static')),
                    'plate': (t.get('plate') or {}).get('plate'),
                    'color': (t.get('attrs') or {}).get('color'),
                    'thumb': (base64.b64encode(snaps[0]['jpeg']).decode('ascii')
                              if snaps else None),
                }
        with open(review_json, 'w', encoding='utf-8') as f:
            json.dump(ui, f, ensure_ascii=False)
        if json_progress:
            emit_json('match_review', path=review_json)
        log_stderr(f'Review session: {review_json}')
    except Exception as e:  # noqa: BLE001 — review is additive, never fatal
        log_stderr(f'Review session write failed ({e})')


def run_finalize_match(args) -> str | None:
    """Second phase of the review workflow: read the pickled match session +
    the user's accept/reject decisions and produce ONE combined report for
    all videos (matched objects merged into single cards, intervals tagged
    with their source video, plates re-voted over the union of snapshots).
    Runs WITHOUT loading the detection model — it only reshapes stored data."""
    import pickle
    json_progress = getattr(args, 'json_progress', False)
    with open(args.finalize_match, 'rb') as f:
        sess = pickle.load(f)
    with open(args.decisions, encoding='utf-8') as f:
        decisions = json.load(f)
    per_video = sess['per_video']
    sources = sess['sources']
    accepted = [[(m[0], int(m[1])) for m in g]
                for g in decisions.get('groups', [])]
    # Coalesce accepted groups that share a member: accepting the pairs
    # A-B and B-C means A, B and C are ONE object — chain them instead of
    # letting the second pair silently drop to a single leftover member.
    coalesced: list[set] = []
    for g in accepted:
        gs = set(g)
        merged_into = None
        for cg in coalesced:
            if cg & gs:
                cg |= gs
                merged_into = cg
                break
        if merged_into is None:
            coalesced.append(gs)
        else:  # the merge may now bridge previously separate sets
            rest = [cg for cg in coalesced if cg is not merged_into and cg & merged_into]
            for cg in rest:
                merged_into |= cg
                coalesced.remove(cg)
    accepted = [sorted(cg) for cg in coalesced]

    def tagged_intervals(t, vname):
        ivs = t.get('intervals') or [{'start': t['first_seen'],
                                      'end': t['last_seen'], 'file': None}]
        return [{'start': iv['start'], 'end': iv['end'], 'file': vname}
                for iv in ivs]

    combined: dict = {}
    used: set = set()
    next_id = 1
    reader = None
    for g in accepted:
        members = [(v, tid) for v, tid in g
                   if v in per_video and tid in per_video[v]
                   and (v, tid) not in used]
        if len(members) < 2:
            continue
        used.update(members)
        ts = [per_video[v][tid] for v, tid in members]
        merged = dict(ts[0])
        merged['intervals'] = [iv for (v, _tid), t in zip(members, ts)
                               for iv in tagged_intervals(t, v)]
        merged['first_seen'] = min(t['first_seen'] for t in ts)
        merged['last_seen'] = max(t['last_seen'] for t in ts)
        merged['confidence'] = max(t.get('confidence', 0) for t in ts)
        merged['dwell_time'] = sum(iv['end'] - iv['start']
                                   for iv in merged['intervals'])
        merged['static'] = all(t.get('static') for t in ts)
        # Snapshots: best from EVERY member so both viewpoints show (cap 8)
        snaps = [s2 for t in ts for s2 in (t.get('snapshots') or [])]
        snaps.sort(key=lambda x: -x.get('score', 0))
        merged['snapshots'] = snaps[:8]
        merged['faces'] = [f2 for t in ts for f2 in (t.get('faces') or [])][:4]
        merged.pop('reappearance', None)
        # Plate: re-vote over the union of snapshots (measured to beat any
        # single camera); fall back to the best member headline.
        if any(k in merged['class'].lower() for k in VEHICLE_KEYWORDS):
            try:
                import cross_match
                if reader is None:
                    from plate_core import PlateReader
                    reader = PlateReader()
                combo = cross_match.combined_plate(ts, reader)
                if combo:
                    merged['plate'] = combo
            except Exception as e:  # noqa: BLE001
                log_stderr(f'Combined plate re-vote failed ({e})')
        combined[next_id] = merged
        next_id += 1

    for vname, tracks in per_video.items():
        for tid, t in tracks.items():
            if (vname, tid) in used:
                continue
            solo = dict(t)
            solo['intervals'] = tagged_intervals(t, vname)
            solo.pop('reappearance', None)
            combined[next_id] = solo
            next_id += 1

    out_dir = str(Path(args.finalize_match).parent)
    prepare_for_report(combined)
    rp = generate_report(combined, out_dir, 'combined', '',
                         video_paths={n: str(p) for n, p in sources.items()})
    log_stderr(f'Combined report: {rp} ({len(accepted)} merges, '
               f'{len(combined)} objects)')
    print(f'COMBINED_REPORT::{Path(rp).resolve()}', flush=True)
    if json_progress:
        emit_json('report', path=rp)
    return rp


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

    if not any(per_video.values()):
        # Every video was skipped (unreadable) or produced zero tracks — a
        # combined report here is an EMPTY page that reads as a silent
        # failure (Windows field case with two skipped .dav files). Surface
        # the error instead.
        log_stderr('Match mode: no tracks from any video — no report')
        if json_progress:
            emit_json('error', message='Κανένα βίντεο δεν έδωσε αποτελέσματα — '
                                       'ελέγξτε αν τα αρχεία είναι αναγνώσιμα')
        return []

    if json_progress:
        emit_json('status', message='Αντιστοίχιση αντικειμένων μεταξύ βίντεο...')
    import cross_match
    groups, uncertain = cross_match.match_videos(per_video, with_uncertain=True)

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

    # Review session FIRST: the shared writer shallow-copies the tracks, so
    # it must run before prepare_for_report (below) strips raw snapshots.
    write_review_session(per_video, sources,
                         args.output or str(Path(video_files[0]).parent),
                         groups=groups, uncertain=uncertain,
                         prefix='visionx_match', json_progress=json_progress)

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


def get_video_fps(source: str, json_progress: bool = False) -> tuple[int, float]:
    """Total frames + ACTUAL fps for a video — the ONE probe implementation
    (process_video had accumulated a hardened inline copy; the DRY audit
    merged them). Hardening carried over from field cases:
      - fps measured by decoding the first ~5s (metadata often lies on
        security-camera exports);
      - DVR containers (.dav) reporting ZERO frames get a grab()-only count
        (demux without decode — fast) instead of being skipped;
      - 0/NaN fps metadata falls back to 25 so downstream never divides by
        zero when building timestamps.
    Returns (total_frames, fps); total_frames == 0 means truly unreadable.
    """
    name = Path(source).name
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        cap.release()
        return 0, 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_metadata = cap.get(cv2.CAP_PROP_FPS)

    frames_to_check = 0
    pos_sec = 0.0  # stays 0 when not a single frame decodes (bad stream)
    while frames_to_check < 100:
        ret, _ = cap.read()
        if not ret:
            break
        frames_to_check += 1
        pos_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if pos_sec >= 5:  # first ~5 seconds are enough for a stable rate
            break
    fps = (frames_to_check / pos_sec if pos_sec > 0 and frames_to_check > 0
           else fps_metadata)
    cap.release()

    if total_frames <= 0:
        log_stderr(f'{name}: no frame count in metadata — counting frames '
                   f'(DVR container)...')
        if json_progress:
            emit_json('status', message=f'Καταμέτρηση καρέ ({name}: '
                                        f'αρχείο DVR χωρίς μεταδεδομένα)...')
        probe = cv2.VideoCapture(source)
        total_frames = 0
        while probe.grab():
            total_frames += 1
        probe.release()
        log_stderr(f'{name}: counted {total_frames} frames')

    if not fps or fps <= 0 or fps != fps:  # 0/NaN — common in .dav exports
        log_stderr(f'{name}: no usable fps in metadata — assuming 25')
        fps = 25.0
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
    for t in tracks.values():
        t['_video_fps'] = fps  # playback overlay needs frame->second mapping
    n_raw = len(tracks)
    tracks = run_stitching(tracks, cfg, json_progress)
    run_auto_plates(tracks, cfg, json_progress)
    run_face_shots(tracks, cfg, json_progress)
    run_attributes(tracks, cfg, json_progress)
    run_prompt_filter(tracks, cfg, args, json_progress)
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


# Supported video containers — the SINGLE SOURCE OF TRUTH for the whole
# app (main.rs VIDEO_EXTS and FileSelector.vue mirror it; a unit test
# fails on any divergence — they had already drifted apart once).
# dav = Dahua DVR, bin = Hikvision raw export.
VIDEO_EXTENSIONS = ['mp4', 'm4v', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm', 'mpeg', 'mpg', 'ts', 'mts', 'm2ts', '3gp', 'asf', 'dav', 'bin']


def _videos_in_dir(dir_path: Path) -> list[str]:
    found = []
    for ext in VIDEO_EXTENSIONS:
        found.extend(dir_path.glob(f'*.{ext}'))
        found.extend(dir_path.glob(f'*.{ext.upper()}'))
    return [str(f) for f in sorted(set(found))]


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
    parser.add_argument('--filter-color', nargs='+',
                        help='Highlight results with these colors '
                             '(fixed vocabulary, e.g. λευκό μαύρο κόκκινο)')
    parser.add_argument('--filter-type', nargs='+',
                        help='Highlight results of these types '
                             '(car/motorcycle/truck/bus/bicycle/person, '
                             'Greek names accepted)')
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
    parser.add_argument('--finalize-match',
                        help='match-session pickle: regenerate the combined '
                             'report from review decisions (no model load)')
    parser.add_argument('--decisions',
                        help='JSON with accepted groups for --finalize-match')
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

    # Review-workflow second phase: reshape stored data into the combined
    # report — no model, no video decoding, returns in seconds.
    if args.finalize_match:
        if not args.decisions:
            print('Error: --finalize-match requires --decisions', file=sys.stderr)
            sys.exit(2)
        run_finalize_match(args)
        return

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
            video_files = _videos_in_dir(dir_path)
        else:
            print(f'Error: {args.dir} is not a directory')
            return

    for src in (args.source or []):
        # A directory passed as a source (e.g. the app's folder picker, or a
        # CLI glob that matched a folder) expands to the videos inside it —
        # feeding the raw folder path to OpenCV used to crash mid-run.
        if Path(src).is_dir():
            found = _videos_in_dir(Path(src))
            if found:
                video_files.extend(found)
            else:
                log_stderr(f'{src}: folder contains no video files — skipped')
        else:
            video_files.append(src)

    if not video_files:
        # Exit non-zero so the desktop app surfaces this as an error instead
        # of "finished, 0 reports" (bit us when nargs='+' filter flags
        # swallowed the video path — silent success hid the real failure).
        print('Error: no video files given (check argument order — '
              'use "--" before file paths if filters are present)',
              file=sys.stderr)
        parser.print_help()
        sys.exit(2)

    # Two-stage architecture (user-designed, see BENCHMARKS.md): detection
    # ALWAYS runs closed-set on the fixed scope (people + vehicles) — best
    # tracking stability at ~half the old default's latency. Search criteria
    # are STRUCTURED filters (specific colors/types) applied on the results;
    # free text and open-vocabulary detection were deliberately REMOVED
    # (user decision: an investigator never searches for a dog, and fixed
    # reliable choices beat free text that can silently misfire). YOLOE
    # checkpoints remain internal tools only (stitching embeddings).
    model_path = args.model or cfg.get('model_closed', 'yolo26l.pt')
    if 'yoloe' in Path(model_path).name.lower():
        log_stderr(f'YOLOE detection is no longer supported as the main model '
                   f'({Path(model_path).name}) — using '
                   f'{cfg.get("model_closed", "yolo26l.pt")} instead')
        model_path = cfg.get('model_closed', 'yolo26l.pt')
    closed_mode = True
    prompts = cfg['prompts']  # legacy field, unused by the closed detector
    # COCO ids for the fixed scope: person bicycle car motorcycle bus truck
    cfg['_closed_classes'] = [0, 1, 2, 3, 5, 7] if closed_mode else None
    json_progress = getattr(args, 'json_progress', False)
    log_stderr(f'Model selection: closed-set {Path(model_path).name}')

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
        if torch.cuda.is_available():
            # NVIDIA: one DYNAMIC TensorRT engine per machine — built once
            # (~10 min, cached forever), covers every imgsz up to 1920 so a
            # resolution change never triggers a rebuild (user concern).
            # Field baseline to beat: 17.9 fps @960 on a GTX 1660 Ti.
            engine = closed_set_trt_engine(model_path, data_dir,
                                           getattr(args, 'imgsz', None)
                                           or cfg.get('imgsz', 960),
                                           json_progress)
            if engine:
                model_path = engine
                using_optimized = True
            else:
                log_stderr('Closed-set mode: using PyTorch runtime')
        else:
            # macOS/MPS: PyTorch is already the measured-good path; the
            # CoreML export of YOLO26 stays a follow-up (needs validation).
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

    scope_label = 'άνθρωποι + οχήματα (όλοι οι τύποι)'
    active_filters = (getattr(args, 'filter_color', None) or []) + \
                     (getattr(args, 'filter_type', None) or [])
    if active_filters:
        scope_label += f' — φίλτρα: {", ".join(active_filters)}'
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
