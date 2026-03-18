#!/usr/bin/env python3
"""VisionX - Pure YOLOE-26 Video Analysis"""

import argparse
import base64
import json
import os
import sys
import yaml
import cv2
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO  # type: ignore[attr-defined]
from ultralytics.data.utils import IMG_FORMATS, VID_FORMATS
from report import generate_report

# Add .dav (Dahua DVR format) support - OpenCV can read these
VID_FORMATS.add('dav')

DEFAULT_CONFIG = {
    'model': 'yoloe-26m-seg.pt',
    'prompts': ['car', 'person', 'motorcycle'],
    'confidence': 0.5,
    'save_video': False,
    'save_crops': False,
    'save_report': True,
    'half': False,
    'imgsz': 640,
    'vid_stride': 1,
    'show': False,
}


def load_config(config_path: str = 'config.yaml') -> dict:
    """Load config from YAML or use defaults"""
    config = DEFAULT_CONFIG.copy()
    if Path(config_path).exists():
        with open(config_path) as f:
            user_config = yaml.safe_load(f)
            if user_config:
                config.update(user_config)
    return config


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


def process_video(source: str, yolo: YOLO, cfg: dict, args, video_index: int = 1, total_videos: int = 1) -> str | None:
    """Process a single video and return report path if generated"""
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

    # Device - CoreML handles device selection automatically
    model_path = str(getattr(yolo, 'ckpt_path', ''))
    if model_path.endswith('.mlpackage'):
        device = None  # CoreML uses Neural Engine + GPU + CPU automatically
        half = False  # CoreML handles precision internally
    elif torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
        half = False  # MPS doesn't reliably support FP16 for all ops
    else:
        device = 'cpu'
        half = False  # CPU doesn't support FP16

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
        print(f'  Skipping: Could not read video')
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
    )

    # Track detections for report (with embedded thumbnails)
    tracks = {}  # track_id -> detection info + thumbnail
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
    for result in results_iter:
        frame_num += 1

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

                # Update or create track entry (keep best confidence)
                if track_id not in tracks or det_conf > tracks[track_id]['confidence']:
                    # Extract raw thumbnail crop with 30% padding (encode later)
                    x1, y1, x2, y2 = map(int, box.tolist())
                    h, w = result.orig_img.shape[:2]
                    box_w, box_h = x2 - x1, y2 - y1
                    pad_x, pad_y = int(box_w * 0.3), int(box_h * 0.3)
                    x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
                    x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)

                    thumb_crop = None
                    if x2 > x1 and y2 > y1:
                        thumb_crop = result.orig_img[y1:y2, x1:x2].copy()

                    tracks[track_id] = {
                        'class': class_name,
                        'confidence': det_conf,
                        'first_seen': tracks.get(track_id, {}).get('first_seen', timestamp),
                        'last_seen': timestamp,
                        'thumbnail': thumb_crop,
                    }
                else:
                    # Always update last_seen
                    tracks[track_id]['last_seen'] = timestamp

    # Batch encode thumbnails (deferred from hot loop)
    for track_id, track in tracks.items():
        if track['thumbnail'] is not None:
            _, buffer = cv2.imencode('.jpg', track['thumbnail'], [cv2.IMWRITE_JPEG_QUALITY, 85])
            track['thumbnail'] = base64.b64encode(buffer).decode('utf-8')

    # Generate HTML report
    report_path = None
    if save_report and tracks:
        video_name = source_path.stem  # filename without extension
        output_dir = args.output or str(source_path.parent)
        report_path = generate_report(tracks, output_dir, video_name, source)
        if json_progress:
            emit_json('report', path=report_path)
        else:
            print(f'  Report: {report_path}')

    return report_path


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

    # Device - CoreML handles device selection automatically
    model_path = str(getattr(yolo, 'ckpt_path', ''))
    if model_path.endswith('.mlpackage'):
        device = None  # CoreML uses Neural Engine + GPU + CPU automatically
        half = False  # CoreML handles precision internally
    elif torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
        half = False  # MPS doesn't reliably support FP16 for all ops
    else:
        device = 'cpu'
        half = False  # CPU doesn't support FP16

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

    # Track detections across all videos
    tracks = {}
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
            half=half,
            imgsz=imgsz,
            vid_stride=stride,
            stream=True,
            verbose=False,
            tracker=tracker_path,
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

                    # Update or create track entry
                    if track_id not in tracks or det_conf > tracks[track_id]['confidence']:
                        # Extract raw thumbnail crop (encode later)
                        x1, y1, x2, y2 = map(int, box.tolist())
                        h, w = result.orig_img.shape[:2]
                        box_w, box_h = x2 - x1, y2 - y1
                        pad_x, pad_y = int(box_w * 0.3), int(box_h * 0.3)
                        x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
                        x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)

                        thumb_crop = None
                        if x2 > x1 and y2 > y1:
                            thumb_crop = result.orig_img[y1:y2, x1:x2].copy()

                        tracks[track_id] = {
                            'class': class_name,
                            'confidence': det_conf,
                            'first_seen': tracks.get(track_id, {}).get('first_seen', local_timestamp),
                            'first_seen_file': tracks.get(track_id, {}).get('first_seen_file', source_name),
                            'last_seen': local_timestamp,
                            'last_seen_file': source_name,
                            'thumbnail': thumb_crop,
                        }
                    else:
                        # Update last_seen
                        tracks[track_id]['last_seen'] = local_timestamp
                        tracks[track_id]['last_seen_file'] = source_name

        # Add this video's duration to cumulative time
        cumulative_time += info['frames'] / fps

    if not json_progress:
        pbar.close()

    # Batch encode thumbnails (deferred from hot loop)
    for track_id, track in tracks.items():
        if track['thumbnail'] is not None:
            _, buffer = cv2.imencode('.jpg', track['thumbnail'], [cv2.IMWRITE_JPEG_QUALITY, 85])
            track['thumbnail'] = base64.b64encode(buffer).decode('utf-8')

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
    using_coreml = worker_args['using_coreml']
    cfg = worker_args['cfg']
    args = worker_args['args']
    video_index = worker_args['video_index']
    total_videos = worker_args['total_videos']

    # Each worker loads its own model instance
    yolo = YOLO(model_path)
    if not using_coreml:
        yolo.set_classes(prompts)  # type: ignore[misc]

    return process_video(source, yolo, cfg, args, video_index=video_index, total_videos=total_videos)


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

    # Store resolved paths in args for use by processing functions
    args._tracker_path = tracker_path

    # Load config
    cfg = load_config(config_path)

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

    # Get settings for model loading
    model_path = args.model or cfg['model']
    prompts = args.search or cfg['prompts']
    json_progress = getattr(args, 'json_progress', False)

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

    # macOS: Try CoreML for Neural Engine acceleration
    using_coreml = False
    if sys.platform == 'darwin':
        # Check for CoreML model next to the .pt file, and also in data dir
        coreml_path = get_coreml_model_path(model_path)
        if coreml_path is None and data_dir:
            # Check data dir for existing CoreML export
            data_coreml = Path(data_dir) / 'models' / Path(model_path).with_suffix('.mlpackage').name
            if data_coreml.exists():
                coreml_path = data_coreml
        if coreml_path is None:
            # Export to data dir (resource dir may be read-only)
            coreml_path = export_to_coreml(model_path, prompts, json_progress, data_dir)

        if coreml_path and coreml_path.exists():
            model_path = str(coreml_path)
            using_coreml = True

    # Determine device name for display (actual device selection happens in processing functions)
    if using_coreml:
        device_name = 'CoreML (Neural Engine)'
    elif torch.cuda.is_available():
        device_name = 'CUDA'
    elif torch.backends.mps.is_available():
        device_name = 'MPS (Metal)'
    else:
        device_name = 'CPU'

    if json_progress:
        emit_json('status', message=f'Using: {device_name}')
        emit_json('status', message=f'Loading {Path(model_path).name}...')
    else:
        print(f'Using: {device_name}')
        print(f'Loading {Path(model_path).name}...')

    # Load model once
    yolo = YOLO(model_path)
    if not using_coreml:
        yolo.set_classes(prompts)  # type: ignore[misc]  # CoreML has classes baked in during export

    if json_progress:
        emit_json('status', message=f'Detecting: {", ".join(prompts)}')
        emit_json('status', message=f'Found {len(video_files)} video(s) to process')
    else:
        print(f'Detecting: {", ".join(prompts)}')
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
                'using_coreml': using_coreml,
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
            report_path = process_video(source, yolo, cfg, args, video_index=idx, total_videos=total_videos)
            if report_path:
                reports.append(report_path)

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
