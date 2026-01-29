#!/usr/bin/env python3
"""VisionX - Pure YOLOE-26 Video Analysis"""

import argparse
import base64
import json
import sys
import yaml
import cv2
import torch
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


def export_to_coreml(pt_model: str, prompts: list[str], json_progress: bool = False) -> Path | None:
    """Export PyTorch model to CoreML (macOS only, one-time operation).

    Classes must be set before export as CoreML models have fixed classes.
    """
    if sys.platform != 'darwin':
        return None

    mlpackage = Path(pt_model).with_suffix('.mlpackage')
    if mlpackage.exists():
        return mlpackage

    if json_progress:
        emit_json('status', message='Exporting to CoreML (one-time)...')
    else:
        print('Exporting to CoreML (one-time, may take a few minutes)...')

    try:
        model = YOLO(pt_model)
        model.set_classes(prompts)  # type: ignore[misc]  # Bake classes into CoreML model
        model.export(format="coreml")
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
    stride = args.stride or cfg['vid_stride']
    show = args.show or cfg['show']

    # Device - CoreML handles device selection automatically
    model_path = str(getattr(yolo, 'ckpt_path', ''))
    if model_path.endswith('.mlpackage'):
        device = None  # CoreML uses Neural Engine + GPU + CPU automatically
    elif torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

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
        vid_stride=stride,
        show=show,
        stream=True,
        verbose=False,
        tracker='tracker.yaml',
    )

    # Track detections for report (with embedded thumbnails)
    tracks = {}  # track_id -> detection info + thumbnail
    frame_num = 0
    total_processed = total_frames // stride
    last_progress_pct = -1

    # Use tqdm for CLI, JSON events for GUI
    if json_progress:
        results_iter = results
    else:
        results_iter = tqdm(results, total=total_processed, desc='  Analyzing', unit='frame')

    for result in results_iter:
        frame_num += 1

        # Emit progress for GUI (throttled to avoid flooding)
        if json_progress:
            progress_pct = int((frame_num / total_processed) * 100) if total_processed > 0 else 0
            if progress_pct != last_progress_pct:
                emit_json('progress',
                    video=source_path.name,
                    frame=frame_num,
                    total_frames=total_processed,
                    video_index=video_index,
                    total_videos=total_videos,
                    fps=fps
                )
                last_progress_pct = progress_pct

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
                    # Extract thumbnail with 30% padding
                    x1, y1, x2, y2 = map(int, box.tolist())
                    h, w = result.orig_img.shape[:2]
                    box_w, box_h = x2 - x1, y2 - y1
                    pad_x, pad_y = int(box_w * 0.3), int(box_h * 0.3)
                    x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
                    x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)

                    thumb_b64 = None
                    if x2 > x1 and y2 > y1:
                        thumb = result.orig_img[y1:y2, x1:x2]
                        # Encode as base64 JPEG
                        _, buffer = cv2.imencode('.jpg', thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        thumb_b64 = base64.b64encode(buffer).decode('utf-8')

                    tracks[track_id] = {
                        'class': class_name,
                        'confidence': det_conf,
                        'first_seen': tracks.get(track_id, {}).get('first_seen', timestamp),
                        'last_seen': timestamp,
                        'thumbnail': thumb_b64,
                    }
                else:
                    # Always update last_seen
                    tracks[track_id]['last_seen'] = timestamp

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
    stride = args.stride or cfg['vid_stride']
    show = args.show or cfg['show']
    save_report = cfg['save_report']

    # Device - CoreML handles device selection automatically
    model_path = str(getattr(yolo, 'ckpt_path', ''))
    if model_path.endswith('.mlpackage'):
        device = None  # CoreML uses Neural Engine + GPU + CPU automatically
    elif torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

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

    # Progress bar for all videos combined
    if not json_progress:
        pbar = tqdm(total=total_all_frames // stride, desc='  Analyzing chain', unit='frame')

    for video_idx, info in enumerate(video_info):
        source = info['source']
        fps = info['fps']
        source_name = Path(source).name

        if json_progress:
            emit_json('status', message=f'Processing: {source_name} ({video_idx + 1}/{len(sources)})')
        else:
            pbar.set_postfix({'file': source_name[:20]})

        # Open video with OpenCV
        cap = cv2.VideoCapture(source)
        frame_in_video = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_in_video += 1

            # Apply stride
            if stride > 1 and frame_in_video % stride != 0:
                continue

            global_frame_num += 1

            # Track this frame (persist=True maintains tracker state)
            results = yolo.track(
                frame,
                conf=conf,
                device=device,
                persist=True,
                verbose=False,
                tracker='tracker.yaml',
            )

            # Update progress
            if json_progress:
                progress_pct = int((global_frame_num / (total_all_frames // stride)) * 100)
                emit_json('progress',
                    video=source_name,
                    frame=global_frame_num,
                    total_frames=total_all_frames // stride,
                    video_index=video_idx + 1,
                    total_videos=len(sources),
                    fps=fps
                )
            else:
                pbar.update(1)

            # Process detections
            if save_report and results:
                result = results[0]
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    continue
                for i in range(len(boxes)):
                    box = boxes.xyxy[i]
                    det_conf = float(boxes.conf[i])
                    cls_id = int(boxes.cls[i])
                    class_name = result.names[cls_id]

                    # Get track ID (skip untracked)
                    if boxes.id is None or boxes.id[i] is None:
                        continue
                    track_id = int(boxes.id[i])

                    # Local timestamp within this video
                    local_timestamp = frame_in_video / fps

                    # Update or create track entry
                    if track_id not in tracks or det_conf > tracks[track_id]['confidence']:
                        # Extract thumbnail
                        x1, y1, x2, y2 = map(int, box.tolist())
                        h, w = result.orig_img.shape[:2]
                        box_w, box_h = x2 - x1, y2 - y1
                        pad_x, pad_y = int(box_w * 0.3), int(box_h * 0.3)
                        x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
                        x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)

                        thumb_b64 = None
                        if x2 > x1 and y2 > y1:
                            thumb = result.orig_img[y1:y2, x1:x2]
                            _, buffer = cv2.imencode('.jpg', thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
                            thumb_b64 = base64.b64encode(buffer).decode('utf-8')

                        tracks[track_id] = {
                            'class': class_name,
                            'confidence': det_conf,
                            'first_seen': tracks.get(track_id, {}).get('first_seen', local_timestamp),
                            'first_seen_file': tracks.get(track_id, {}).get('first_seen_file', source_name),
                            'last_seen': local_timestamp,
                            'last_seen_file': source_name,
                            'thumbnail': thumb_b64,
                        }
                    else:
                        # Update last_seen
                        tracks[track_id]['last_seen'] = local_timestamp
                        tracks[track_id]['last_seen_file'] = source_name

        cap.release()

        # Add this video's duration to cumulative time
        cumulative_time += info['frames'] / fps

    if not json_progress:
        pbar.close()

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
    parser.add_argument('--show', action='store_true', help='Live preview')
    parser.add_argument('--chain', action='store_true', help='Process multiple videos as continuous sequence')
    parser.add_argument('--config', default='config.yaml', help='Config file')
    parser.add_argument('--json-progress', action='store_true', help='Output JSON progress (for GUI)')
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

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

    # macOS: Try CoreML for Neural Engine acceleration
    using_coreml = False
    if sys.platform == 'darwin':
        coreml_path = get_coreml_model_path(model_path)
        if coreml_path is None:
            coreml_path = export_to_coreml(model_path, prompts, json_progress)

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

    if args.chain and len(video_files) > 1:
        # Chain mode: process all videos as continuous sequence
        report_path = process_video_chain(video_files, yolo, cfg, args)
        if report_path:
            reports.append(report_path)
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
