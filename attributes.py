"""Attribute extraction on track snapshots — user-designed two-stage flow.

Design decision (user's): the detector always runs CLOSED-SET for speed and
tracking stability; richer characteristics are computed AFTERWARDS on each
track's snapshots. Type comes free from the detector class (car/truck/bus/
motorcycle/bicycle); this module adds COLOR:
  - vehicles: dominant body color
  - persons: dominant clothing color, upper/lower half separately

Method: classical dominant-color voting in HSV — deterministic, dependency-
free, explainable to a reviewer ("the pixels are green"), and immune to the
confident-hallucination failure modes we measured on ML paths. Shadow and
glare pixels are excluded before voting.
"""

import cv2
import numpy as np

# Greek color names for the report. Hue ranges are OpenCV H∈[0,180).
_COLOR_ORDER = ['κόκκινο', 'πορτοκαλί', 'κίτρινο', 'πράσινο', 'γαλάζιο',
                'μπλε', 'μωβ', 'ροζ']
_HUE_BINS = [(0, 8, 'κόκκινο'), (8, 21, 'πορτοκαλί'), (21, 33, 'κίτρινο'),
             (33, 78, 'πράσινο'), (78, 100, 'γαλάζιο'), (100, 128, 'μπλε'),
             (128, 145, 'μωβ'), (145, 170, 'ροζ'), (170, 181, 'κόκκινο')]

CSS_COLOR = {
    'λευκό': '#f5f5f5', 'μαύρο': '#111111', 'γκρι/ασημί': '#9aa0a6',
    'κόκκινο': '#d93025', 'πορτοκαλί': '#f29900', 'κίτρινο': '#fbbc04',
    'πράσινο': '#188038', 'γαλάζιο': '#12b5cb', 'μπλε': '#1a73e8',
    'μωβ': '#9334e6', 'ροζ': '#e8368f', 'καφέ': '#8d6e63',
}


def _wb_gains(ref_bgr: np.ndarray) -> np.ndarray:
    """Gray-world white-balance gains, estimated on the WHOLE crop.

    Dusk/artificial-light casts shift neutral surfaces into a hue (measured:
    silver cars reading 'γαλάζιο' at dusk). The illumination estimate must
    come from the full crop — subject plus surroundings — NOT from the
    voting region: a region that is genuinely one color (a red car body)
    would otherwise be 'balanced' into gray by construction."""
    means = ref_bgr.astype(np.float32).reshape(-1, 3).mean(axis=0)
    target = float(means.mean())
    return np.array([target / max(1e-3, m) for m in means], dtype=np.float32)


def _apply_gains(img_bgr: np.ndarray, gains: np.ndarray) -> np.ndarray:
    return np.clip(img_bgr.astype(np.float32) * gains, 0, 255).astype(np.uint8)


def _pixel_colors(img_bgr: np.ndarray, wb_ref: np.ndarray | None = None) -> dict:
    """Vote pixels of a (pre-cropped) region into named colors. Returns
    {name: fraction}. Shadows (very dark + saturated-noise) and blown
    highlights vote as black/white rather than random hues."""
    img_bgr = _apply_gains(img_bgr, _wb_gains(
        wb_ref if wb_ref is not None else img_bgr))
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0].astype(np.int32)
    s = hsv[..., 1].astype(np.int32)
    v = hsv[..., 2].astype(np.int32)
    n = h.size
    if n == 0:
        return {}
    votes: dict = {}

    achromatic = s < 45
    black = achromatic & (v < 70)
    white = achromatic & (v > 175)
    gray = achromatic & ~black & ~white
    votes['μαύρο'] = int(black.sum())
    votes['λευκό'] = int(white.sum())
    votes['γκρι/ασημί'] = int(gray.sum())

    chrom = ~achromatic & (v >= 40)
    # Brown = dark, moderately saturated orange — must be split off before
    # the hue vote or every brown car reads "orange".
    brown = chrom & (h >= 8) & (h < 25) & (v < 130)
    votes['καφέ'] = int(brown.sum())
    remaining = chrom & ~brown
    for lo, hi, name in _HUE_BINS:
        m = remaining & (h >= lo) & (h < hi)
        votes[name] = votes.get(name, 0) + int(m.sum())

    total = sum(votes.values()) or 1
    return {k: v_ / total for k, v_ in votes.items() if v_ > 0}


def _center(img: np.ndarray, fx=0.25, fy=0.25) -> np.ndarray:
    hh, ww = img.shape[:2]
    return img[int(hh * fy):hh - int(hh * fy) or hh,
               int(ww * fx):ww - int(ww * fx) or ww]


def dominant_color(img_bgr: np.ndarray) -> tuple:
    """(color_name, fraction) of the region's dominant color."""
    votes = _pixel_colors(img_bgr)
    if not votes:
        return None, 0.0
    name = max(votes, key=votes.get)
    return name, votes[name]


def vehicle_color(crops: list) -> tuple | None:
    """Vote the body color across a track's snapshot crops. The center region
    is used (windows/wheels/asphalt live at the edges). Returns
    (name, confidence_fraction) or None when votes are too scattered."""
    agg: dict = {}
    for c in crops:
        if c is None or c.size == 0:
            continue
        for name, frac in _pixel_colors(_center(c), wb_ref=c).items():
            agg[name] = agg.get(name, 0.0) + frac
    if not agg:
        return None
    total = sum(agg.values())
    name = max(agg, key=agg.get)
    frac = agg[name] / total
    # Below 35% dominance the "color" is a guess between shades — omit
    # rather than mislead (candidate-tool honesty rule).
    return (name, round(frac, 2)) if frac >= 0.35 else None


def person_clothing(crops: list) -> dict:
    """Upper/lower clothing dominant colors across a person track's crops.
    Rough body split: upper garment ≈ 20-55% of box height, lower ≈ 55-85%
    (head above, feet/ground below)."""
    out = {}
    for key, y0, y1 in (('upper', 0.20, 0.55), ('lower', 0.55, 0.85)):
        agg: dict = {}
        for c in crops:
            if c is None or c.size == 0:
                continue
            hh = c.shape[0]
            band = c[int(hh * y0):int(hh * y1)]
            band = _center(band, fx=0.30, fy=0.05)
            if band.size == 0:
                continue
            for name, frac in _pixel_colors(band, wb_ref=c).items():
                agg[name] = agg.get(name, 0.0) + frac
        if agg:
            total = sum(agg.values())
            name = max(agg, key=agg.get)
            if agg[name] / total >= 0.35:
                out[key] = (name, round(agg[name] / total, 2))
    return out
