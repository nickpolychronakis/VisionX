"""Shared track collection for vision.py — ROADMAP Phase A1.

One TrackCollector serves both process_video and process_video_chain (their
track-building blocks were near-identical duplicates before). New over the
old inline logic:
  - best-K snapshots per track (not just one best-confidence thumbnail), so
    the report can show a gallery and downstream steps (plate OCR, face
    best-shot, stitching embeddings) have clean crops to work with;
  - running position statistics (sum/sumsq — no per-frame history kept) so
    static/parked objects can be classified without extra memory;
  - interval bookkeeping compatible with offline tracklet stitching.
"""

import base64
import math

import cv2
import numpy as np


class TrackCollector:
    """Accumulates per-track state during a streaming yolo.track() run."""

    def __init__(self, max_snapshots: int = 4):
        self.tracks: dict = {}
        # 4 snapshots: enough diversity for a gallery + stitching embeddings
        # without bloating report size (each is an embedded JPEG).
        self.max_snapshots = max_snapshots

    def add(self, *, track_id, class_name, conf, box_xyxy, frame_img,
            timestamp, source_name=None, frame_idx=None):
        x1, y1, x2, y2 = map(int, box_xyxy)
        h, w = frame_img.shape[:2]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        diag = math.hypot(x2 - x1, y2 - y1)

        t = self.tracks.get(track_id)
        if t is None:
            t = self.tracks[track_id] = {
                'class': class_name,
                'confidence': conf,
                'first_seen': timestamp,
                'last_seen': timestamp,
                'first_seen_file': source_name,
                'last_seen_file': source_name,
                'first_pos': (cx, cy),
                'last_pos': (cx, cy),
                'frame_count': 0,
                '_pos_sum': [0.0, 0.0],
                '_pos_sumsq': [0.0, 0.0],
                '_max_diag': 0.0,
                'snapshots': [],  # [{score, ts, file, jpeg, box}] sorted desc
                # Per-frame boxes (tiny: 5 numbers/frame). Lets the auto-ALPR
                # pass go BACK to the raw video and cut fresh vehicle crops
                # across the whole trajectory — the report snapshots alone
                # gave it only 1-2 usable plate views, which is exactly the
                # single-frame weakness the multi-frame research warns about.
                '_boxes': [],
            }
        t['confidence'] = max(t['confidence'], conf)
        t['last_seen'] = timestamp
        if source_name:
            t['last_seen_file'] = source_name
        t['last_pos'] = (cx, cy)
        t['frame_count'] += 1
        t['_pos_sum'][0] += cx
        t['_pos_sum'][1] += cy
        t['_pos_sumsq'][0] += cx * cx
        t['_pos_sumsq'][1] += cy * cy
        t['_max_diag'] = max(t['_max_diag'], diag)
        t['_frame_wh'] = (w, h)
        if frame_idx is not None:
            t['_boxes'].append((int(frame_idx), x1, y1, x2, y2))

        # Snapshot candidacy — cheap proxy first (conf × size), so pixels are
        # only touched when the crop would actually enter the top-K. Full
        # Laplacian sharpness on every box of every frame would cost minutes
        # per hour of video; conf×size correlates well enough for ranking.
        proxy = conf * math.sqrt(max(1.0, (x2 - x1) * (y2 - y1)))
        snaps = t['snapshots']
        if len(snaps) >= self.max_snapshots and proxy <= snaps[-1]['score']:
            return

        pad_x, pad_y = int((x2 - x1) * 0.25), int((y2 - y1) * 0.25)
        sx1, sy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        sx2, sy2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
        if sx2 <= sx1 or sy2 <= sy1:
            return
        ok, jpeg = cv2.imencode('.jpg', frame_img[sy1:sy2, sx1:sx2],
                                [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return
        # Context frame: the WHOLE downscaled frame with a red box on the
        # object, so the report can show where in the scene each snapshot
        # was taken from. Captured only when a crop enters the top-K (the
        # proxy gate above), so the extra encode cost stays negligible.
        context = None
        try:
            scale = min(1.0, 800.0 / w)
            ctx_img = (cv2.resize(frame_img,
                                  (int(w * scale), int(h * scale)),
                                  interpolation=cv2.INTER_AREA)
                       if scale < 1.0 else frame_img.copy())
            cv2.rectangle(ctx_img,
                          (int(x1 * scale), int(y1 * scale)),
                          (int(x2 * scale), int(y2 * scale)),
                          (0, 0, 255), 2)
            okc, ctx_jpeg = cv2.imencode('.jpg', ctx_img,
                                         [cv2.IMWRITE_JPEG_QUALITY, 60])
            if okc:
                context = ctx_jpeg.tobytes()
        except Exception:
            pass  # context view is auxiliary — never lose the snapshot
        snap = {'score': proxy, 'ts': timestamp, 'file': source_name,
                'jpeg': jpeg.tobytes(), 'box': (x1, y1, x2, y2),
                'context': context}
        # Temporal diversity: K snapshots must not be K consecutive frames.
        # ADAPTIVE window (user feedback: short tracks ended up with ONE
        # snapshot): the spacing requirement scales with the track's life so
        # far — a 2s pass still yields several distinct moments, while long
        # tracks keep the 0.7s spread.
        lifetime = timestamp - t['first_seen']
        window = max(0.15, min(0.7, lifetime / (self.max_snapshots + 1)))
        near = next((s for s in snaps if abs(s['ts'] - timestamp) < window
                     and s.get('file') == source_name), None)
        if near is not None:
            if proxy > near['score']:
                snaps.remove(near)
                snaps.append(snap)
        else:
            snaps.append(snap)
        snaps.sort(key=lambda s: -s['score'])
        del snaps[self.max_snapshots:]

    def finalize(self) -> dict:
        """Compute per-track derived fields. Returns the tracks dict with
        intervals + static flag; snapshots stay as raw JPEG bytes so the
        stitching pass can still merge them (encode for the report with
        prepare_for_report() afterwards)."""
        for t in self.tracks.values():
            n = max(1, t['frame_count'])
            mean = (t['_pos_sum'][0] / n, t['_pos_sum'][1] / n)
            var = (max(0.0, t['_pos_sumsq'][0] / n - mean[0] ** 2)
                   + max(0.0, t['_pos_sumsq'][1] / n - mean[1] ** 2))
            std = math.sqrt(var)
            # Static = the object wandered less than ~25% of its own size
            # over its whole life (fixed-camera assumption — on dashcam
            # footage everything moves, so nothing classifies as static,
            # which is the correct degenerate behavior). min 5 frames so a
            # 2-frame flicker can't be "parked".
            t['static'] = (t['frame_count'] >= 5
                           and std < max(12.0, 0.25 * t['_max_diag']))
            t['mean_pos'] = mean
            t['pos_std'] = std
            t['dwell_time'] = t['last_seen'] - t['first_seen']
            t['direction'] = _direction(t['first_pos'], t['last_pos'],
                                        static=t['static'])
            t['intervals'] = [{
                'start': t['first_seen'], 'end': t['last_seen'],
                'file': t.get('first_seen_file'),
            }]
            # Host-vehicle signature (dashcam footage): the recording car
            # detects its OWN hood as a "car" — a wide box glued to the
            # bottom edge, in the lower half, across many frames. Flagged
            # (and excluded from the live preview) but never hidden: the
            # report keeps the card with a badge — annotate, do not hide.
            bx = t.get('_boxes') or []
            fw, fh = t.get('_frame_wh', (0, 0))
            if bx and fw and t['frame_count'] >= 10:
                mw = sum(b[3] - b[1] for b in bx) / len(bx)
                my1 = sum(b[2] for b in bx) / len(bx)
                my2 = sum(b[4] for b in bx) / len(bx)
                t['host_vehicle'] = bool(my2 >= 0.96 * fh
                                         and my1 >= 0.45 * fh
                                         and mw >= 0.35 * fw)
            else:
                t['host_vehicle'] = False
            for k in ('_pos_sum', '_pos_sumsq'):
                t.pop(k, None)
        return self.tracks


def _direction(first, last, static=False) -> str:
    dx, dy = last[0] - first[0], last[1] - first[1]
    if static or (abs(dx) < 30 and abs(dy) < 30):
        return '●'
    angle = math.degrees(math.atan2(-dy, dx))  # -dy: image y grows downward
    return ['→', '↗', '↑', '↖', '←', '↙', '↓', '↘'][round(angle / 45) % 8]


def prepare_for_report(tracks: dict) -> dict:
    """Encode snapshots to base64 and drop non-serializable internals.
    Keeps the legacy 'thumbnail' field (best snapshot) so report code and
    older consumers keep working."""
    for t in tracks.values():
        # Playback overlay: downsampled per-frame boxes (full-frame pixel
        # coords) so the report's clip player can draw the vehicle box —
        # and its plate — during replay, like the live preview but for one
        # object. ≤300 points ≈ 3KB per track.
        boxes = t.get('_boxes') or []
        if boxes:
            step = max(1, len(boxes) // 300)
            t['playback'] = {
                'fps': round(float(t.get('_video_fps') or 25.0), 3),
                'boxes': [[int(v) for v in b] for b in boxes[::step]],
            }
        snaps = t.get('snapshots', [])
        t['snapshots_b64'] = [base64.b64encode(s['jpeg']).decode('utf-8')
                              for s in snaps]
        t['snapshot_ts'] = [s['ts'] for s in snaps]
        # Parallel list: full-frame context (red box baked in) per snapshot;
        # '' where capture failed so indexes stay aligned with snapshots_b64.
        t['snapshots_ctx_b64'] = [
            base64.b64encode(s['context']).decode('utf-8')
            if s.get('context') else ''
            for s in snaps]
        t['thumbnail'] = t['snapshots_b64'][0] if t['snapshots_b64'] else None
        for k in ('snapshots', '_max_diag', '_emb_vpe', '_emb_hist', '_boxes',
                  '_video_fps', '_frame_wh'):
            t.pop(k, None)
    return tracks
