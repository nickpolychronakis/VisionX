"""Best face shots per person track — ROADMAP Phase Γ.

Detector: YuNet via OpenCV's built-in FaceDetectorYN. Chosen over the
roadmap's SCRFD suggestion because it fills the identical role while
shipping INSIDE our existing opencv-contrib (zero new pip deps), returns the
5 landmarks needed for frontality scoring, and its model is a 232KB ONNX
that auto-downloads once. SCRFD stays a documented upgrade option if YuNet
misses too many small faces in the field.

Scope guard: this module does EXTRACTION ONLY — "the clearest face image of
each person track, for human review". No face recognition, no identity
matching, no embeddings are computed or stored.
"""

import math
import urllib.request
from pathlib import Path

import cv2
import numpy as np

MODEL_NAME = 'face_detection_yunet_2023mar.onnx'
MODEL_URL = ('https://github.com/opencv/opencv_zoo/raw/main/models/'
             'face_detection_yunet/face_detection_yunet_2023mar.onnx')

# Person crops are analyzed at this height: CCTV person boxes are often
# 100-300px tall with the face a small fraction of that — upscaling before
# detection lets YuNet see faces that would fall under its minimum size.
DETECT_HEIGHT = 480


def _resolve_model(model_dir=None) -> str | None:
    """Find or download the YuNet model. Returns None when unavailable
    (offline first run) — callers must degrade gracefully."""
    candidates = []
    if model_dir:
        candidates.append(Path(model_dir) / 'models' / MODEL_NAME)
    candidates.append(Path(__file__).parent / 'models' / MODEL_NAME)
    for c in candidates:
        if c.exists():
            return str(c)
    dest = candidates[0]
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, dest)  # noqa: S310 — fixed https URL
        return str(dest)
    except Exception:  # noqa: BLE001 — offline etc.
        return None


class FaceExtractor:
    """Finds the clearest face(s) inside a person track's snapshot crops."""

    def __init__(self, model_dir=None, conf: float = 0.7):
        path = _resolve_model(model_dir)
        if path is None:
            raise RuntimeError('YuNet model unavailable (offline first run?)')
        # conf 0.7: person crops are small and busy; lower thresholds pull in
        # backpacks/heads-from-behind which poison the "best face" ranking.
        self.det = cv2.FaceDetectorYN_create(path, '', (320, 320), conf, 0.3, 200)

    def best_faces(self, crops: list, top: int = 2) -> list:
        """Return up to `top` face shots as dicts {jpeg, score, frontal}.
        Empty list = no usable face (person facing away etc.) — normal."""
        found = []
        for crop in crops:
            if crop is None or crop.size == 0:
                continue
            h, w = crop.shape[:2]
            scale = DETECT_HEIGHT / max(1, h)
            img = cv2.resize(crop, (max(32, int(w * scale)), DETECT_HEIGHT),
                             interpolation=cv2.INTER_CUBIC if scale > 1
                             else cv2.INTER_AREA)
            self.det.setInputSize((img.shape[1], img.shape[0]))
            _, faces = self.det.detect(img)
            if faces is None:
                continue
            for f in faces:
                x, y, fw, fh = f[:4]
                conf = float(f[14])
                frontal = _frontality(f)
                # Margin 40%: chin/hair context makes the shot recognizable
                # to a human reviewer, tight crops feel like eye strips.
                mx, my = int(fw * 0.4), int(fh * 0.4)
                x1, y1 = max(0, int(x - mx)), max(0, int(y - my))
                x2 = min(img.shape[1], int(x + fw + mx))
                y2 = min(img.shape[0], int(y + fh + my))
                face = img[y1:y2, x1:x2]
                if face.size == 0 or fh < 16:
                    continue
                sharp = cv2.Laplacian(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY),
                                      cv2.CV_64F).var()
                # Score: detector confidence × frontality × size (saturates
                # at 80px — bigger isn't better beyond readability) ×
                # sharpness (saturates at 150 Laplacian var).
                score = (conf * (0.4 + 0.6 * frontal)
                         * min(1.0, fh / 80.0)
                         * min(1.0, sharp / 150.0))
                ok, jpeg = cv2.imencode('.jpg', face,
                                        [cv2.IMWRITE_JPEG_QUALITY, 88])
                if ok:
                    found.append({'jpeg': jpeg.tobytes(),
                                  'score': float(score),
                                  'frontal': round(frontal, 2)})
        found.sort(key=lambda d: -d['score'])
        return found[:top]


def _frontality(f: np.ndarray) -> float:
    """Eye-nose symmetry from YuNet's 5 landmarks: ~1 facing the camera,
    ~0 in profile. Landmark layout: [4:6]=right eye, [6:8]=left eye,
    [8:10]=nose tip."""
    re_x, le_x, nose_x = float(f[4]), float(f[6]), float(f[8])
    d_r, d_l = abs(re_x - nose_x), abs(le_x - nose_x)
    if max(d_r, d_l) < 1e-3:
        return 0.0
    return min(d_r, d_l) / max(d_r, d_l)
