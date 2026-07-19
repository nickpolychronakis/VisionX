"""Quality-first benchmark: TRACKING STABILITY per detector model.

The user's field observation: small models "lose tracking and re-detect the
same things again and again". Mechanism: missed detections → track death →
new ID. So the quality metric that matters for VisionX is not raw
detections/frame but TRACK FRAGMENTATION: for the same physical objects,
fewer/longer tracklets = a detector that sees objects CONSISTENTLY.

Runs full BoT-SORT tracking per model on the real clip and reports:
tracklets produced, long tracks (≥5 frames), median track length, and
detection persistence.
"""
import statistics
import sys
import time

import torch

sys.path.insert(0, '/Users/nickpolychronakis/Developer/VisionX')
from ultralytics import YOLO

VIDEO = '/Users/nickpolychronakis/Downloads/peugeot206.mp4'
TRACKER = '/Users/nickpolychronakis/Developer/VisionX/tracker.yaml'
COCO_TARGET = [0, 1, 2, 3, 5, 7]
PROMPTS = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']
DEVICE = ('cuda' if torch.cuda.is_available()
          else 'mps' if torch.backends.mps.is_available() else 'cpu')
CONF = 0.35

CANDIDATES = [
    ('yolo26n.pt', 'closed'),
    ('yolo26s.pt', 'closed'),
    ('yolo26m.pt', 'closed'),
    ('yolo26l.pt', 'closed'),
    ('yolo26x.pt', 'closed'),
    ('yoloe-26x-seg.pt', 'yoloe'),
]

print(f'device={DEVICE} conf={CONF} tracker=BoT-SORT (tracker.yaml)\n')
print(f'{"model":<18} {"ms/fr":>6} {"tracklets":>9} {"long(>=5fr)":>11} '
      f'{"med.len":>7} {"dets/fr":>7}')

for name, kind in CANDIDATES:
    try:
        path = ('/Users/nickpolychronakis/Developer/VisionX/' + name
                if kind == 'yoloe' else name)
        model = YOLO(path)
        kwargs = dict(conf=CONF, device=DEVICE, verbose=False, imgsz=640,
                      tracker=TRACKER, persist=True, stream=True)
        if kind == 'closed':
            kwargs['classes'] = COCO_TARGET
        else:
            model.set_classes(PROMPTS, model.get_text_pe(PROMPTS))
        track_frames: dict = {}
        dets_per_frame = []
        t0 = time.perf_counter()
        n_frames = 0
        for r in model.track(source=VIDEO, **kwargs):
            n_frames += 1
            n = 0
            if r.boxes is not None and r.boxes.id is not None:
                for tid in r.boxes.id:
                    track_frames.setdefault(int(tid), 0)
                    track_frames[int(tid)] += 1
                    n += 1
            dets_per_frame.append(n)
        elapsed = (time.perf_counter() - t0) * 1000 / max(1, n_frames)
        lens = sorted(track_frames.values(), reverse=True)
        long_tracks = sum(1 for x in lens if x >= 5)
        print(f'{name:<18} {elapsed:6.1f} {len(lens):9d} {long_tracks:11d} '
              f'{statistics.median(lens) if lens else 0:7.1f} '
              f'{statistics.mean(dets_per_frame):7.1f}   lengths={lens[:8]}')
        del model
    except Exception as e:
        print(f'{name:<18} FAILED: {str(e)[:100]}')

print('\nReading: for the SAME physical objects, FEWER tracklets with LONGER '
      'lengths = the stability the user needs. (37-frame clip, ~6 real '
      'vehicles — indicative, to be repeated on long fixed-CCTV footage.)')
