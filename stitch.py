"""Offline tracklet stitching — ROADMAP Phase A2.

BoT-SORT inevitably fragments tracks on occlusions/exits (the same car or
person then appears as several report entries). Because VisionX is an OFFLINE
tool we can repair this after the fact, the way 2025-2026 tracking-challenge
winners do (GTA-style global tracklet association): appearance clustering of
tracklets under hard spatio-temporal gates.

Appearance signature (chosen from measured probes on real footage, see
project memory): YOLOE visual-prompt embeddings alone had a thin same-object
margin (0.94 vs 0.92 worst distractor — they encode "car-ness", not
identity), HSV color histograms alone had a clean margin (0.92 vs 0.80);
the 50/50 combination separates best. Wrong merges are worse than missed
merges, so every gate below is conservative.
"""

import math

import cv2
import numpy as np

# Similarity threshold for merging. From the calibration probe: same-object
# combined scores landed at 0.92-0.93, worst cross-object at ~0.83. 0.88
# sits between with margin on both sides; exposed in config.yaml because
# footage varies (tune UP if wrong merges are ever observed).
DEFAULT_THRESHOLD = 0.88

# Two tracklets of the SAME object can never truly coexist in one camera —
# but trackers occasionally emit a double box for 1-2 frames around identity
# switches, so tolerate a sliver of overlap instead of 0.
MAX_TIME_OVERLAP_S = 0.3

# Spatial gate: how far the object may plausibly travel per second of gap
# (in pixels). Generous — the appearance threshold does the fine filtering;
# this only kills absurd links (left edge → right edge in 0.2s).
MAX_SPEED_PX_S = 500.0
SPATIAL_SLACK_PX = 120.0


# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------

def _hsv_signature(img: np.ndarray) -> np.ndarray:
    """Center-weighted HSV histogram (the 1/6 border is cropped away so
    pavement/background pixels don't dominate the color identity)."""
    h, w = img.shape[:2]
    inner = img[h // 6: h - h // 6 or h, w // 6: w - w // 6 or w]
    if inner.size == 0:
        inner = img
    hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [18, 8], [0, 180, 0, 256])
    return cv2.normalize(hist, None).flatten().astype(np.float32)


def compute_embeddings(tracks: dict, embed_model_path: str, device=None,
                       max_crops: int = 3, log=None) -> bool:
    """Attach appearance signatures to every track (in place).

    Uses a DEDICATED small YOLOE checkpoint (nano) for the visual-prompt
    embeddings rather than the session's main model: on macOS/TensorRT the
    main model may be an exported engine that cannot produce embeddings, and
    embeddings only need to be consistent with EACH OTHER, not with the
    detector. Falls back to color-histogram-only if the model can't load —
    stitching then still works, just with less evidence.
    Returns True if VPE embeddings are available.
    """
    for t in tracks.values():
        crops = []
        for s in t.get('snapshots', [])[:max_crops]:
            img = cv2.imdecode(np.frombuffer(s['jpeg'], np.uint8), cv2.IMREAD_COLOR)
            if img is not None and img.size:
                crops.append(img)
        t['_crops_tmp'] = crops
        sigs = [_hsv_signature(c) for c in crops]
        t['_emb_hist'] = np.mean(sigs, axis=0).astype(np.float32) if sigs else None

    vpe_ok = False
    try:
        from ultralytics import YOLO
        from ultralytics.models.yolo.yoloe import YOLOEVPDetectPredictor
        yolo = YOLO(embed_model_path)
        predictor = YOLOEVPDetectPredictor(
            overrides=dict(model=embed_model_path, task='segment',
                           mode='predict', imgsz=320, verbose=False, save=False))
        predictor.setup_model(yolo.model)

        def vpe_of(img):
            hh, ww = img.shape[:2]
            predictor.set_prompts(dict(
                bboxes=np.array([[2, 2, ww - 2, hh - 2]], dtype=np.float32),
                cls=np.array([0])))
            v = predictor.get_vpe(img).reshape(-1).float().cpu().numpy()
            n = np.linalg.norm(v)
            return v / n if n > 0 else v

        for t in tracks.values():
            vs = [vpe_of(c) for c in t['_crops_tmp']]
            if vs:
                m = np.mean(vs, axis=0)
                n = np.linalg.norm(m)
                t['_emb_vpe'] = (m / n if n > 0 else m).astype(np.float32)
            else:
                t['_emb_vpe'] = None
        vpe_ok = True
    except Exception as e:  # noqa: BLE001 — any failure degrades gracefully
        if log:
            log(f'VPE embeddings unavailable ({e}) — color-only stitching')
        for t in tracks.values():
            t['_emb_vpe'] = None
    finally:
        for t in tracks.values():
            t.pop('_crops_tmp', None)
    return vpe_ok


def similarity(a: dict, b: dict) -> float:
    """Combined appearance similarity in [.., 1]; 50/50 VPE cosine + HSV
    correlation per the calibration probe. Missing components fall back to
    whatever is available."""
    parts = []
    if a.get('_emb_vpe') is not None and b.get('_emb_vpe') is not None:
        parts.append(float(np.dot(a['_emb_vpe'], b['_emb_vpe'])))
    if a.get('_emb_hist') is not None and b.get('_emb_hist') is not None:
        parts.append(float(cv2.compareHist(a['_emb_hist'], b['_emb_hist'],
                                           cv2.HISTCMP_CORREL)))
    return sum(parts) / len(parts) if parts else -1.0


# --------------------------------------------------------------------------
# Gates + clustering
# --------------------------------------------------------------------------

def _intervals_compatible(ia: list, ib: list) -> float | None:
    """If no pair of intervals overlaps more than MAX_TIME_OVERLAP_S, return
    the smallest gap between the groups; else None (same-time = different
    physical objects, never merge). Chain-mode intervals from different files
    are treated as compatible (no shared clock)."""
    min_gap = math.inf
    for a in ia:
        for b in ib:
            if a.get('file') != b.get('file'):
                continue
            overlap = min(a['end'], b['end']) - max(a['start'], b['start'])
            if overlap > MAX_TIME_OVERLAP_S:
                return None
            gap = max(a['start'], b['start']) - min(a['end'], b['end'])
            min_gap = min(min_gap, max(0.0, gap))
    return 0.0 if min_gap is math.inf else min_gap


def _spatially_plausible(a: dict, b: dict, gap_s: float) -> bool:
    # Order by time so we compare exit point → entry point.
    first, second = (a, b) if a['first_seen'] <= b['first_seen'] else (b, a)
    dist = math.hypot(second['first_pos'][0] - first['last_pos'][0],
                      second['first_pos'][1] - first['last_pos'][1])
    if a.get('static') and b.get('static'):
        # Two static tracklets are the same parked object only if they sit
        # in (almost) the same spot — but then ANY time gap is fine: that is
        # exactly the "car occluded by passing traffic for a while" case.
        size = max(a.get('_max_diag', 50.0), b.get('_max_diag', 50.0))
        return dist < max(30.0, 0.5 * size)
    return dist <= MAX_SPEED_PX_S * max(0.5, gap_s) + SPATIAL_SLACK_PX


def stitch_tracks(tracks: dict, sim_threshold: float = DEFAULT_THRESHOLD,
                  log=None) -> dict:
    """Merge fragmented tracklets of the same physical object.

    Greedy agglomerative merging over candidate pairs sorted by appearance
    similarity (highest first), each merge re-validated against the whole
    groups (class equality, time compatibility, spatial plausibility).
    Returns a new dict keyed by the smallest id of each merged group.
    """
    ids = sorted(tracks.keys())
    parent = {i: i for i in ids}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    groups = {i: dict(tracks[i]) for i in ids}  # working copies

    pairs = []
    for idx, i in enumerate(ids):
        for j in ids[idx + 1:]:
            a, b = tracks[i], tracks[j]
            if a['class'] != b['class']:
                continue
            sim = similarity(a, b)
            if sim >= sim_threshold:
                pairs.append((sim, i, j))
    pairs.sort(reverse=True)

    merges = 0
    for sim, i, j in pairs:
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        a, b = groups[ri], groups[rj]
        gap = _intervals_compatible(a['intervals'], b['intervals'])
        if gap is None or not _spatially_plausible(a, b, gap):
            continue
        # Merge b into a (a keeps the smaller root id)
        if rj < ri:
            ri, rj, a, b = rj, ri, b, a
        parent[rj] = ri
        # Decide endpoint ownership BEFORE mutating the min/max fields.
        b_starts_earlier = b['first_seen'] < a['first_seen']
        b_ends_later = b['last_seen'] > a['last_seen']
        a['intervals'] = sorted(a['intervals'] + b['intervals'],
                                key=lambda iv: (str(iv.get('file')), iv['start']))
        a['confidence'] = max(a['confidence'], b['confidence'])
        a['frame_count'] += b['frame_count']
        a['first_seen'] = min(a['first_seen'], b['first_seen'])
        a['last_seen'] = max(a['last_seen'], b['last_seen'])
        if b_starts_earlier:
            a['first_pos'] = b['first_pos']
            a['first_seen_file'] = b.get('first_seen_file')
        if b_ends_later:
            a['last_pos'] = b['last_pos']
            a['last_seen_file'] = b.get('last_seen_file')
        a['static'] = a.get('static', False) and b.get('static', False)
        a['snapshots'] = sorted(a.get('snapshots', []) + b.get('snapshots', []),
                                key=lambda s: -s['score'])[:4]
        # Per-frame boxes must follow the merge too, or the playback overlay
        # draws the ROOT tracklet's position while a MERGED interval plays
        # (field bug: box on the wrong car). Concatenate; prepare_for_report
        # re-sorts by frame index.
        a['_boxes'] = (a.get('_boxes') or []) + (b.get('_boxes') or [])
        a['merged_from'] = sorted(set(a.get('merged_from', [ri])
                                      + b.get('merged_from', [rj])))
        # Weighted mean of embeddings keeps the group signature honest for
        # subsequent merge decisions in this same pass.
        for key in ('_emb_vpe', '_emb_hist'):
            va, vb = a.get(key), b.get(key)
            if va is not None and vb is not None:
                m = (va + vb) / 2.0
                if key == '_emb_vpe':
                    n = np.linalg.norm(m)
                    m = m / n if n > 0 else m
                a[key] = m
        merges += 1

    out = {}
    for i in ids:
        r = find(i)
        if r not in out:
            g = groups[r]
            g['dwell_time'] = sum(iv['end'] - iv['start'] for iv in g['intervals'])
            out[r] = g
    if log:
        log(f'Stitching: {len(tracks)} tracklets -> {len(out)} objects '
            f'({merges} merges)')
    return out
