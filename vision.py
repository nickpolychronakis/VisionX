#!/usr/bin/env python3
"""VisionX - Pure YOLOE-26 Video Analysis"""

import argparse
import yaml
import cv2
import torch
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO  # type: ignore[attr-defined]
from report import generate_report

DEFAULT_CONFIG = {
    'model': 'yoloe-26m-seg.pt',
    'prompts': ['car', 'person', 'motorcycle'],
    'confidence': 0.5,
    'save_video': True,
    'save_crops': False,
    'save_report': False,
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

def main():
    parser = argparse.ArgumentParser(
        description='VisionX - YOLOE-26 Video Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python vision.py video.mp4
  python vision.py --search "white car" video.mp4
  python vision.py --crops video.mp4
        '''
    )
    parser.add_argument('source', help='Video file or URL')
    parser.add_argument('--search', nargs='+', help='Custom prompts')
    parser.add_argument('--model', help='Model file')
    parser.add_argument('--conf', type=float, help='Confidence threshold')
    parser.add_argument('--output', '-o', help='Output directory')
    parser.add_argument('--crops', action='store_true', help='Save cropped detections')
    parser.add_argument('--report', action='store_true', help='Generate HTML detection report')
    parser.add_argument('--half', action='store_true', help='FP16 mode (faster)')
    parser.add_argument('--stride', type=int, help='Frame skip (2=2x faster)')
    parser.add_argument('--show', action='store_true', help='Live preview')
    parser.add_argument('--config', default='config.yaml', help='Config file')
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    # CLI overrides config
    model = args.model or cfg['model']
    prompts = args.search or cfg['prompts']
    conf = args.conf or cfg['confidence']
    # Default output: same directory as input video
    output = args.output or str(Path(args.source).parent)
    save_crops = args.crops or cfg['save_crops']
    save_report = args.report or cfg['save_report']
    half = args.half or cfg['half']
    stride = args.stride or cfg['vid_stride']
    show = args.show or cfg['show']

    # Setup output
    Path(output).mkdir(parents=True, exist_ok=True)

    # Select device (MPS for Apple Silicon, else CPU)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f'Using device: {device}')

    # Load model
    print(f'Loading {model}...')
    yolo = YOLO(model)

    # Set prompts
    yolo.set_classes(prompts)  # type: ignore[misc]
    print(f'Detecting: {", ".join(prompts)}')

    # Get video info for progress bar and timestamps
    cap = cv2.VideoCapture(args.source)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # Run tracking (stream=True prevents OOM on large videos)
    results = yolo.track(
        source=args.source,
        conf=conf,
        device=device,
        persist=True,
        save=True,
        save_crop=save_crops,
        project=output,
        name='',
        half=half,
        vid_stride=stride,
        show=show,
        stream=True,
        verbose=False,
    )

    # Track detections for report
    tracks = {}  # track_id -> best detection info
    frame_num = 0
    report_dir = Path(output) / 'report' / 'thumbnails'
    if save_report:
        report_dir.mkdir(parents=True, exist_ok=True)

    for result in tqdm(results, total=total_frames // stride, desc='Processing', unit='frame'):
        frame_num += 1

        if save_report and result.boxes is not None and len(result.boxes):
            for i in range(len(result.boxes)):
                box = result.boxes.xyxy[i]
                conf = float(result.boxes.conf[i])
                cls_id = int(result.boxes.cls[i])
                class_name = result.names[cls_id]

                # Get track ID if available
                track_id = None
                if result.boxes.id is not None:
                    track_id = int(result.boxes.id[i])

                if track_id is None:
                    track_id = f"unk_{frame_num}_{i}"

                timestamp = (frame_num * stride) / fps

                # Update or create track entry (keep best confidence)
                if track_id not in tracks or conf > tracks[track_id]['confidence']:
                    tracks[track_id] = {
                        'class': class_name,
                        'confidence': conf,
                        'first_seen': tracks.get(track_id, {}).get('first_seen', timestamp),
                        'last_seen': timestamp,
                    }

                    # Save thumbnail (best frame per track)
                    x1, y1, x2, y2 = map(int, box.tolist())
                    # Ensure valid crop coordinates
                    h, w = result.orig_img.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    if x2 > x1 and y2 > y1:
                        thumb = result.orig_img[y1:y2, x1:x2]
                        cv2.imwrite(str(report_dir / f"{track_id}_{class_name}.jpg"), thumb)

                # Always update last_seen
                tracks[track_id]['last_seen'] = timestamp

    print(f'Done! Output: {output}/')

    # Generate HTML report
    if save_report and tracks:
        video_name = Path(args.source).name
        report_path = generate_report(tracks, output, video_name, args.source)
        print(f'Report: {report_path}')

if __name__ == '__main__':
    main()
