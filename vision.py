#!/usr/bin/env python3
"""VisionX - Pure YOLOE-26 Video Analysis"""

import argparse
import yaml
import cv2
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO  # type: ignore[attr-defined]

DEFAULT_CONFIG = {
    'model': 'yoloe-26m-seg.pt',
    'prompts': ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'scooter', 'person'],
    'confidence': 0.5,
    'save_video': True,
    'save_crops': False,
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
    half = args.half or cfg['half']
    stride = args.stride or cfg['vid_stride']
    show = args.show or cfg['show']

    # Setup output
    Path(output).mkdir(parents=True, exist_ok=True)

    # Load model
    print(f'Loading {model}...')
    yolo = YOLO(model)

    # Set prompts
    yolo.set_classes(prompts)  # type: ignore[misc]
    print(f'Detecting: {", ".join(prompts)}')

    # Get video frame count for progress bar
    cap = cv2.VideoCapture(args.source)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Run tracking (stream=True prevents OOM on large videos)
    results = yolo.track(
        source=args.source,
        conf=conf,
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

    for _ in tqdm(results, total=total_frames // stride, desc='Processing', unit='frame'):
        pass

    print(f'Done! Output: {output}/')

if __name__ == '__main__':
    main()
