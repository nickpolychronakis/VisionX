"""Benchmark: closed-set YOLO26 vs open-vocabulary YOLOE-26 on the fixed
people+vehicles scope (person, bicycle, car, motorcycle, bus, truck).
Apple Silicon (MPS). Speed = median ms/frame; quality proxy = detections on
the real test clip (no labels — indicative counts + mean confidence)."""
import statistics
import sys
import time

import cv2
import torch

sys.path.insert(0, '/Users/nickpolychronakis/Developer/VisionX')
from ultralytics import YOLO

VIDEO = '/Users/nickpolychronakis/Downloads/peugeot206.mp4'
COCO_TARGET = [0, 1, 2, 3, 5, 7]  # person bicycle car motorcycle bus truck
PROMPTS = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
N_FRAMES = 25
CONF = 0.35

cap = cv2.VideoCapture(VIDEO)
frames = []
while len(frames) < N_FRAMES:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(f)
cap.release()
print(f'device={DEVICE}  frames={len(frames)}  conf={CONF}\n')

CANDIDATES = [
    ('yolo26n.pt', 'closed'),
    ('yolo26s.pt', 'closed'),
    ('yolo26m.pt', 'closed'),
    ('yoloe-26n-seg.pt', 'yoloe'),
    ('yoloe-26x-seg.pt', 'yoloe'),
]

results = []
for name, kind in CANDIDATES:
    try:
        path = ('/Users/nickpolychronakis/Developer/VisionX/' + name
                if kind == 'yoloe' else name)
        model = YOLO(path)
        kwargs = dict(conf=CONF, device=DEVICE, verbose=False, imgsz=640)
        if kind == 'closed':
            kwargs['classes'] = COCO_TARGET
        else:
            model.set_classes(PROMPTS, model.get_text_pe(PROMPTS))
        # Warmup
        for f in frames[:3]:
            model.predict(f, **kwargs)
        if DEVICE == 'mps':
            torch.mps.synchronize()
        times, dets, confs = [], 0, []
        for f in frames:
            t0 = time.perf_counter()
            r = model.predict(f, **kwargs)[0]
            if DEVICE == 'mps':
                torch.mps.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
            if r.boxes is not None:
                dets += len(r.boxes)
                confs += [float(c) for c in r.boxes.conf]
        med = statistics.median(times)
        results.append((name, kind, med, dets / len(frames),
                        statistics.mean(confs) if confs else 0.0))
        print(f'{name:<20} {kind:<7} median {med:7.1f} ms/frame '
              f'({1000/med:5.1f} fps)  dets/frame {dets/len(frames):4.1f}  '
              f'mean conf {statistics.mean(confs) if confs else 0:.2f}')
        del model
    except Exception as e:
        print(f'{name:<20} FAILED: {str(e)[:120]}')

print('\nNote: no ground-truth labels — dets/frame and conf are indicative '
      'only; speed is the hard number. CoreML/TensorRT deployment adds its '
      'own conversion story per model.')
