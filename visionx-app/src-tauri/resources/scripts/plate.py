#!/usr/bin/env python3
"""
VisionX Plate — multi-frame license plate reading from CCTV/dashcam video.

Workflow:
  1. User seeks to a frame and draws a ROI around the plate (interactive GUI),
     or passes --roi/--start-frame for headless use.
  2. The ROI is tracked forward with OpenCV CSRT (or KCF), with periodic
     re-detection by a specialized plate detector (open-image-models, YOLOv9-t)
     to correct tracker drift. Tracking can be paused/corrected manually.
  3. Every tracked crop is rectified to a canonical size and sub-pixel aligned
     to the sharpest reference frame with ECC (homography model).
  4. The best-aligned frames are fused (sharpness-weighted average) into a
     single cleaner image — classical multi-frame fusion, NOT neural
     super-resolution, which a 2026 study showed gives 0% OCR gain and
     hallucinates characters (unacceptable even for candidate generation).
  5. fast-plate-ocr reads every crop + the fused image — each in 4 tonal
     variants (original / CLAHE / low-gamma / high-gamma test-time
     augmentation) — returning per-character probabilities.
  6. Per-position weighted voting across all reads produces a ranked list of
     candidate plates (top-N with confidence) for downstream database lookup.

IMPORTANT: this is a candidate-generation tool for investigative search
(e.g. Ministry of Transport DB queries), NOT an evidentiary/forensic tool.
Light image enhancement is therefore acceptable; ranked uncertainty is the
expected output, not a single "certain" answer.

Usage:
  python plate.py video.mp4                                   # interactive
  python plate.py video.mp4 --roi 613,412,95,38 --start-frame 120 --no-gui
"""

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# fast-plate-ocr / open-image-models are imported lazily inside main() so that
# `plate.py --help` stays instant and a missing optional dep gives a clear
# actionable message instead of a bare ImportError traceback.

# ---------------------------------------------------------------------------
# Defaults / constants
# ---------------------------------------------------------------------------

# OCR ensemble: three architecture-diverse models vote together. Ground-truth
# tuning on real dusk footage (plate YHH3472) showed each model makes
# DIFFERENT systematic errors (cct-s-v2 read the tail as '777', the other two
# as the correct '72'; the european vit model scored best per-char overall) —
# joint voting recovers characters any single model loses. Runtime cost is
# small: each model is ~5MB ONNX and ~2-3ms per crop on CPU.
DEFAULT_OCR_MODELS = ('cct-s-v2-global-model,'
                      'cct-s-v1-global-model,'
                      'european-plates-mobile-vit-v2-model')
N_SLOTS = 10  # vote matrix width; models emit 9 or 10 slots, padded to this

# 384px detector: middle of the yolo-v9-t family. We only run it on a small
# padded crop around the tracked box (not the full frame), so a bigger input
# size buys nothing; 384 keeps refinement fast on CPU (~20ms).
DEFAULT_DETECTOR_MODEL = 'yolo-v9-t-384-license-plate-end2end'

# Canonical rectified plate height in px. OCR models ingest ~64-70px heights;
# working at 128 preserves detail for ECC alignment and fusion, and the OCR
# lib downscales internally. Upscaling beyond ~4x the typical 25-35px CCTV
# plate height adds no information (classical interpolation only).
CANON_H = 128

# EU/Greek plate geometric aspect is 520/110 ≈ 4.7 but the *visible* crop
# aspect varies with viewing angle; we derive aspect from the actual box and
# only clamp to this range to reject degenerate tracker boxes.
MIN_ASPECT, MAX_ASPECT = 1.0, 8.0

# Greek civilian plates: 3 letters + 4 digits (cars) or 3 letters + 3 digits
# (motorcycles). Used as a SOFT ranking bonus, never a hard filter — foreign
# and special-issue plates must still surface in the candidate list.
# Only the Latin-identical Greek letters are legal: a candidate containing
# e.g. L or S may be a valid FOREIGN plate but cannot be Greek, so it must
# not receive the GR bonus (field case: 'L??5977' ranked top with a GR flag
# despite L being impossible on Greek plates).
GR_LETTERS = 'ABEZHIKMNOPTYX'
GREEK_PATTERNS = (rf'^[{GR_LETTERS}]{{3}}[0-9]{{4}}$',
                  rf'^[{GR_LETTERS}]{{3}}[0-9]{{3}}$')

WINDOW = 'VisionX Plate'

# Detector acceptance threshold for tracking refinement. 0.20 (was 0.30):
# dusk/small dashcam plates often score only 0.15-0.35, and every rejected
# detection leaves an orange (unverified, drift-prone) frame. A false positive
# here is filtered by proximity ranking and ultimately by OCR-vote consensus,
# so permissive is the right trade for a candidate-generation tool.
DET_ACCEPT = 0.20

# ENTER arrives as 13 (CR) or 10 (LF) depending on backend/keyboard.
K_ENTER = (13, 10)
K_ESC = 27

# Greek keyboard layout sends Greek codepoints for letter keys (field debug
# log: dozens of presses of α=945 doing nothing because only ord('a')==97
# matched). Every letter key therefore also accepts its Greek-layout twin —
# the tool must work without forcing a layout switch.
K_A = (ord('a'), ord('A'), 945, 913)   # a · α · Α
K_D = (ord('d'), ord('D'), 948, 916)   # d · δ · Δ
K_J = (ord('j'), ord('J'), 958, 926)   # j · ξ · Ξ
K_L = (ord('l'), ord('L'), 955, 923)   # l · λ · Λ
K_R = (ord('r'), ord('R'), 961, 929)   # r · ρ · Ρ
K_F = (ord('f'), ord('F'), 966, 934)   # f · φ · Φ
K_Q = (ord('q'), ord('Q'), 59)         # q — Greek layout maps the q key to ';'
K_N = (ord('n'), ord('N'), 957, 925)   # n · ν · Ν — "new segment" prompt
K_B = (ord('b'), ord('B'), 946, 914)   # b · β · Β — auto-scan for best plate view


def _seek_step(key: int) -> int:
    """Seek step for a navigation key (a/d = ±1, j/l = ±25, Greek twins too)."""
    if key in K_A:
        return -1
    if key in K_D:
        return 1
    if key in K_J:
        return -25
    return 25


class _EventLog:
    """Opt-in (--debug) event logger for GUI diagnostics.

    Exists because a field report ("it quit as if I pressed q when the window
    lost focus") could not be reproduced blind: every key code, drag and exit
    reason is timestamped here so the next occurrence is diagnosable from the
    file instead of from memory.
    """

    def __init__(self):
        self.path = None   # set by --debug in main(); None = disabled
        self._fh = None

    def __call__(self, mode: str, msg: str):
        if not self.path:
            return
        if self._fh is None:
            self._fh = open(self.path, 'w', encoding='utf-8')
        self._fh.write(f'{time.strftime("%H:%M:%S")} [{mode}] {msg}\n')
        self._fh.flush()  # a crash/quit must not lose the last events


DEBUG = _EventLog()


def _wait_key(delay: int, mode: str) -> int:
    """cv2.waitKeyEx wrapper: full key codes + event logging.

    Deliberately NOT masked with `& 0xFF`: macOS/Cocoa can emit codes >255
    for window events (focus changes among them), and 8-bit masking can alias
    them onto printable keys — the prime suspect for the reported
    "window lost focus → tool quit like q". Exact-code comparison plus the
    double-press quit guard make a single spurious event harmless either way.
    """
    code = cv2.waitKeyEx(delay)
    if code not in (-1, 255):
        printable = chr(code) if 32 <= code < 127 else '?'
        DEBUG(mode, f'key code={code} char={printable}')
    return code


class _ConfirmQuit:
    """q must be pressed twice within 1.5s to take effect: one spurious
    injected key event (see _wait_key) must never end a session and throw
    away minutes of manual tracking work."""

    def __init__(self):
        self._armed_at = -10.0

    def confirmed(self, mode: str) -> bool:
        now = time.monotonic()
        if now - self._armed_at <= 1.5:
            DEBUG(mode, 'quit confirmed (second q)')
            return True
        self._armed_at = now
        log('  Πατήστε ξανά q για ολοκλήρωση')
        DEBUG(mode, 'quit armed, awaiting second q')
        return False

    @property
    def pending(self) -> bool:
        """True while the first q is armed — the HUD shows the hint so the
        user watching the VIDEO (not the terminal) knows a second q is
        expected (field feedback: single q appearing to 'do nothing')."""
        return time.monotonic() - self._armed_at <= 1.5


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    """One tracked observation of the plate in a single video frame."""
    frame_idx: int
    box: tuple            # (x, y, w, h) in frame coords (tracker or detector)
    crop: np.ndarray      # tight BGR crop of the plate
    sharpness: float      # Laplacian variance on the canonical gray crop
    det_conf: float       # detector confidence (0.0 if this box is tracker-only)
    rect: np.ndarray = None     # canonical rectified BGR crop (filled later)
    aligned: np.ndarray = None  # ECC-aligned version of rect (None if ECC failed)
    ecc_cc: float = 0.0         # ECC correlation vs reference (alignment quality)
    sat_ratio: float = 0.0      # fraction of clipped (>=250) pixels — burned plate
    rect_level: np.ndarray = None  # tilt-corrected rendering (quad-levelled)


@dataclass
class Candidate:
    plate: str
    score: float                 # normalized 0-1 ranking score
    pattern_match: bool
    per_char: list = field(default_factory=list)  # [{pos, char, prob, alternatives}]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def log(msg: str):
    print(msg, flush=True)


def fail(msg: str) -> 'NoReturn':
    print(f'Error: {msg}', file=sys.stderr)
    sys.exit(1)


def laplacian_sharpness(gray: np.ndarray) -> float:
    # Variance of the Laplacian: standard cheap focus metric. Computed on the
    # canonically-resized crop so values are comparable across frames even as
    # the vehicle approaches and the raw crop grows.
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def iou(a: tuple, b: tuple) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2, bx2, by2 = ax1 + aw, ay1 + ah, bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def clamp_box(box: tuple, w: int, h: int) -> tuple | None:
    """Clamp a (x,y,w,h) box to frame bounds; None if nothing usable remains."""
    x, y, bw, bh = box
    x1, y1 = max(0, int(round(x))), max(0, int(round(y)))
    x2, y2 = min(w, int(round(x + bw))), min(h, int(round(y + bh)))
    if x2 - x1 < 8 or y2 - y1 < 4:  # smaller than this can't hold characters
        return None
    return (x1, y1, x2 - x1, y2 - y1)


def padded_crop(frame: np.ndarray, box: tuple, pad_ratio: float) -> tuple:
    """Crop box + pad_ratio margin. Returns (crop, (ox, oy)) offset of crop origin."""
    x, y, w, h = box
    px, py = int(w * pad_ratio), int(h * pad_ratio)
    x1, y1 = max(0, x - px), max(0, y - py)
    x2, y2 = min(frame.shape[1], x + w + px), min(frame.shape[0], y + h + py)
    return frame[y1:y2, x1:x2], (x1, y1)


def canonical_size(box_w: int, box_h: int) -> tuple:
    aspect = min(MAX_ASPECT, max(MIN_ASPECT, box_w / max(1, box_h)))
    return (int(round(CANON_H * aspect)), CANON_H)  # (w, h)


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------

def open_video(path: str) -> tuple:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        fail(f'cannot open video: {path}')
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 0 or fps != fps:
        fps = 25.0
    if total <= 0:
        # DVR containers (.dav) often report zero frames while decoding
        # fine — count by grab() (demux-only, fast) before giving up.
        log('Καταμέτρηση καρέ (αρχείο DVR χωρίς μεταδεδομένα)...')
        probe = cv2.VideoCapture(path)
        total = 0
        while probe.grab():
            total += 1
        probe.release()
        log(f'  {total} καρέ')
    if total <= 0:
        fail(f'could not read video (0 frames detected): {path}')
    return cap, total, fps


def seek(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray | None:
    # CAP_PROP_POS_FRAMES seeking can be off by a frame on some codecs but is
    # fine here: the user visually confirms the frame before selecting a ROI.
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    return frame if ok else None


# ---------------------------------------------------------------------------
# Phase 1 — interactive frame seek + ROI selection
# ---------------------------------------------------------------------------

class _BoxDrawer:
    """Rubber-band box drawn with a mouse drag, shared by all GUI screens.

    Replaces cv2.selectROI everywhere: field testing showed the ENTER →
    drag → ENTER dance was confusing (a plain ENTER inside selectROI returns
    an empty box, which read as the tool ignoring or quitting on the user).
    With a plain drag there is exactly one gesture and no confirmation key.
    """

    def __init__(self, seek_strip=0, frame_h=0, on_seek=None):
        self.start = None  # (x, y) anchor while a drag is in progress
        self.cur = None    # current mouse position during the drag
        self.box = None    # finalized (x, y, w, h) after mouse-up
        # Optional self-drawn seek bar rendered as an EXTRA strip appended
        # BELOW the frame (see _seek_bar_strip). Replaces cv2.createTrackbar/
        # setTrackbarPos entirely (native Cocoa trackbar = prime suspect for a
        # field-reported hard crash on macOS + OpenCV 5.0). The bar must live
        # OUTSIDE the frame area: dashcam plates often sit at the very bottom
        # of the image, and an earlier in-frame bar stole exactly those drags
        # (field report: plate at y≈1180 of 1200 became unselectable).
        self.seek_strip = seek_strip
        self.frame_h = frame_h
        self.on_seek = on_seek
        self._seeking = False

    def on_mouse(self, event, x, y, flags, _param):
        if self.seek_strip and self.on_seek:
            if event == cv2.EVENT_LBUTTONDOWN:
                # A fresh press decides seek-vs-box anew, so a mouse-up lost
                # outside the window can never wedge us in scrub mode.
                self._seeking = y >= self.frame_h
                if self._seeking:
                    self.on_seek(x)
                    return
            elif self._seeking:
                if event == cv2.EVENT_MOUSEMOVE:
                    self.on_seek(x)
                elif event == cv2.EVENT_LBUTTONUP:
                    self.on_seek(x)
                    self._seeking = False
                return
        if self.frame_h:
            y = min(y, self.frame_h - 1)  # box drags may stray into the bar strip
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start, self.cur, self.box = (x, y), (x, y), None
        elif event == cv2.EVENT_MOUSEMOVE and self.start:
            self.cur = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.start:
            x0, y0 = self.start
            bw, bh = abs(x - x0), abs(y - y0)
            self.start = None
            # Same minimum as clamp_box: anything smaller can't hold characters
            # (also filters out accidental clicks).
            if bw >= 8 and bh >= 4:
                self.box = (min(x0, x), min(y0, y), bw, bh)

    def draw_overlay(self, disp):
        if self.start and self.cur:
            cv2.rectangle(disp, self.start, self.cur, (0, 255, 255),
                          _box_thickness(disp))


def _hud(disp, line, text, color=(0, 255, 255)):
    """HUD text scaled to the frame width: fixed 0.7-0.8 font sizes were
    unreadably small on 4K dashcam frames (field screenshots). `line` is the
    1-based text row from the top."""
    fs = max(0.7, min(2.2, disp.shape[1] / 1800))
    cv2.putText(disp, text, (10, int(40 * fs * line)), cv2.FONT_HERSHEY_SIMPLEX,
                fs, color, max(2, int(fs * 2)), cv2.LINE_AA)


def _box_thickness(disp):
    """Box line width scaled to frame width (2px vanishes on 4K)."""
    return max(2, disp.shape[1] // 1500)


SEEK_BAR_H = 24  # px height of the self-drawn seek bar strip


def _seek_bar_strip(w, pos, total):
    """Seek bar rendered as its OWN strip, vstack'ed BELOW the frame so it
    never overlaps video content (bottom-of-frame plates must stay
    selectable). Click/drag inside the strip to seek."""
    strip = np.full((SEEK_BAR_H, w, 3), 25, np.uint8)
    yc = SEEK_BAR_H // 2
    cv2.line(strip, (8, yc), (w - 8, yc), (90, 90, 90), 2, cv2.LINE_AA)
    x = 8 + int(pos / max(1, total - 1) * (w - 16))
    cv2.line(strip, (8, yc), (x, yc), (0, 200, 255), 2, cv2.LINE_AA)
    cv2.circle(strip, (x, yc), 6, (0, 255, 255), -1, cv2.LINE_AA)
    return strip


def rank_plate_appearances(raw_found):
    """Chain raw per-probe plate detections into PHYSICAL plates and rank them
    best-first for the `b` (jump-to-best-appearance) key: largest genuine
    passes first, burned-in overlays / parked cars deranked to the END (still
    reachable).

    Pulled out of scan_plate_appearances as a PURE function (2026-07-24 b-scan
    ranking-bug fix) so the clustering + static-derank + ranking can be unit-
    tested with a plain list — no video, no detector. The field bug (a burned-
    in 70mai dashcam watermark ranked #1 and OCR'd as '70100') lived entirely
    in this deterministic logic and was untestable while it was inlined.

    raw_found: list of (frame_idx, box, width) probe detections, box = (x,y,w,h).
    Returns:   list of (frame_idx, box, width, is_static), best-first.
    """
    # Consecutive-probe boxes that overlap (or nearly touch) belong to the same
    # vehicle — chain them so `b` cycles per VEHICLE (shown at its largest
    # appearance) instead of through 100+ near-duplicate frames.
    raw_found = sorted(raw_found, key=lambda t: t[0])
    clusters = []  # each: {'last': box, 'items': [(idx, box, w)]}
    for idx, box, bw in raw_found:
        home = None
        cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
        for cl in clusters:
            lb = cl['last']
            lcx, lcy = lb[0] + lb[2] / 2, lb[1] + lb[3] / 2
            near = ((lcx - cx) ** 2 + (lcy - cy) ** 2) ** 0.5 < 2.5 * max(box[2], lb[2])
            if iou(box, lb) > 0.15 or near:
                home = cl
                break
        if home is None:
            clusters.append({'last': box, 'items': [(idx, box, bw)]})
        else:
            home['last'] = box
            home['items'].append((idx, box, bw))

    def is_static(cl):
        # Burned-in overlays (dashcam logo, timestamps) detect as "plates" but
        # sit PIXEL-IDENTICAL across the whole video; even a parked car's plate
        # jitters a little through compression. The test is RELATIVE:
        # position/size drift under ~10% of the width. Static clusters are
        # DERANKED, not dropped — on a fixed camera a parked car is static too
        # and must stay reachable.
        #
        # Floor is 2, NOT 6 (2026-07-24 fix): a positionally-frozen box is a
        # burned-in overlay no matter HOW FEW probes it lands in — the count
        # only has to be >=2 for drift to be measurable at all. The 70mai
        # watermark flickered into just 4 probes (its detector confidence
        # hovered at the 0.25 accept threshold), so it escaped the old len>=6
        # guard and — being the widest box in the frame — hijacked #1 over a
        # genuine 80px moving plate. A real MOVING plate is never frozen across
        # probes that are `stride` frames (seconds) apart, so this can't
        # mislabel one; a box that IS frozen is effectively parked/overlay,
        # which this bucket already handles.
        if len(cl['items']) < 2:
            return False
        xs = [b[0] for _, b, _ in cl['items']]
        ws = [b[2] for _, b, _ in cl['items']]
        tol = max(4, 0.1 * (sum(ws) / len(ws)))
        return (max(xs) - min(xs) <= tol and max(ws) - min(ws) <= tol)

    moving, static = [], []
    for cl in clusters:
        best = max(cl['items'], key=lambda t: t[2])
        (static if is_static(cl) else moving).append((best, len(cl['items'])))
    # Single-probe clusters rank BELOW multi-probe ones: a genuinely close pass
    # shows up in several probes, while a borderline logo detection (conf
    # hovering at the threshold) flickers into ONE probe and — being huge —
    # would otherwise hijack the top spot. Among multi-probe clusters, widest
    # first (largest = most readable plate).
    moving.sort(key=lambda t: (-(t[1] >= 2), -t[0][2]))
    static.sort(key=lambda t: -t[0][2])
    moving = [b for b, _n in moving]
    static = [b for b, _n in static]
    return ([(i, b, w2, False) for i, b, w2 in moving]
            + [(i, b, w2, True) for i, b, w2 in static])


def scan_plate_appearances(cap, total_frames, detector, hud_frame=None):
    """Automatic whole-video plate scan (user request: "why must I hunt for
    the closest pass manually?"). Probes ~240 evenly-spaced frames with the
    plate detector on a downscaled copy — small/far plates vanish at the
    detector's input size, so what survives is exactly the LARGE, readable
    appearances we want to jump to. Returns [(frame_idx, box, w)] sorted by
    plate width, largest first."""
    stride = max(1, total_frames // 120)
    found = []
    if hud_frame is not None:
        dark = (hud_frame * 0.35).astype(np.uint8)
        _hud(dark, 1, 'SCANNING VIDEO FOR PLATE APPEARANCES...', (0, 255, 255))
        cv2.imshow(WINDOW, dark)
        cv2.waitKey(1)
    log(f'  Σάρωση βίντεο για εμφανίσεις πινακίδας ({total_frames // stride} δείγματα)...')
    # TILED detection: the detector letterboxes its input to 640px, so on a
    # full HD/4K frame a 60-100px plate shrinks below detectability (first
    # version found literally nothing). 800px tiles with 25% overlap keep
    # plates near their native size.
    TILE, OVER = 800, 200
    for idx in range(0, total_frames, stride):
        frame = seek(cap, idx)
        if frame is None:
            continue
        h, w = frame.shape[:2]
        per_frame = []
        for ty in range(0, max(1, h - OVER), TILE - OVER):
            for tx in range(0, max(1, w - OVER), TILE - OVER):
                tile = frame[ty:ty + TILE, tx:tx + TILE]
                if tile.shape[0] < 60 or tile.shape[1] < 60:
                    continue
                try:
                    dets = detector.predict(tile)
                except Exception:  # noqa: BLE001
                    continue
                for d in dets:
                    if float(d.confidence) < 0.25:
                        continue
                    bb = d.bounding_box
                    box = (int(bb.x1) + tx, int(bb.y1) + ty,
                           int(bb.x2 - bb.x1), int(bb.y2 - bb.y1))
                    # Plate geometry gate: without it the widest "find" was
                    # an 803x704 false positive (a whole car front) that
                    # hijacked the best-first ranking. Real plates run
                    # ~1.8-7.5 w/h (GR long plates ≈4.7, square moto ≈2)
                    # and never approach half the frame width.
                    aspect = box[2] / max(1, box[3])
                    if not (1.8 <= aspect <= 7.5) or box[2] > 0.5 * w:
                        continue
                    per_frame.append((box, float(d.confidence)))
        # Overlapping tiles see the same plate twice — keep the widest of
        # any overlapping pair.
        per_frame.sort(key=lambda t: -t[0][2])
        kept = []
        for box, conf in per_frame:
            if all(iou(box, kb) < 0.4 for kb, _ in kept):
                kept.append((box, conf))
        for box, _conf in kept:
            found.append((idx, box, box[2]))
    # Cluster into physical plates + rank best-first (static overlays/parked
    # cars deranked to the end but reachable). Pure logic extracted to
    # rank_plate_appearances so it is unit-testable — see it for the 70mai
    # watermark fix.
    ranked = rank_plate_appearances(found)
    n_static = sum(1 for *_rest, is_stat in ranked if is_stat)
    if n_static:
        log(f'  ({n_static} στατικές εμφανίσεις — λογότυπα/υπερθέματα ή '
            f'παρκαρισμένα — μπήκαν στο ΤΕΛΟΣ της λίστας)')
    log(f'  Βρέθηκαν {len(ranked)} εμφανίσεις — b: άλμα στη μεγαλύτερη, '
        f'ξανά b: επόμενη, ENTER: αποδοχή πλαισίου')
    return ranked


def gui_pick_roi(cap, total_frames, fps, start_frame=0, detector=None):
    """Selection screen: seek freely AND drag a box around the plate at any
    moment — tracking starts the instant the mouse button is released, with
    no confirmation key (the video starts here directly, per user feedback:
    navigation and selection belong on the same screen).

    Returns (frame_idx, (x, y, w, h)) or (None, None) on abort.
    """
    log('\n[Select] drag a box around the plate (any time) · b = ΑΥΤΟΜΑΤΗ '
        'εύρεση καλύτερης εμφάνισης · SPACE play/pause · a/d ±1 · j/l ±25 · '
        'seek bar κάτω · q abort')

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    quit_guard = _ConfirmQuit()
    state = {'pos': start_frame, 'seek_req': None}

    playing = False
    frame = seek(cap, state['pos'])
    if frame is None:
        fail('could not read first frame')
    H, W = frame.shape[:2]

    def on_seek(x):
        # Map bar x-position → frame index (same 8px margins as the drawing).
        frac = (x - 8) / max(1, W - 16)
        state['seek_req'] = int(round(min(1.0, max(0.0, frac)) * (total_frames - 1)))

    drawer = _BoxDrawer(seek_strip=SEEK_BAR_H, frame_h=H, on_seek=on_seek)
    cv2.setMouseCallback(WINDOW, drawer.on_mouse)

    # Auto-scan state (b key): lazily computed plate appearances, largest
    # first, with the current suggestion drawn for one-key acceptance.
    suggestions = None
    sug_i = -1
    sug_box = None

    while True:
        if drawer.start:
            playing = False  # starting a drag pauses playback on the spot
        if state['seek_req'] is not None:
            state['pos'] = state['seek_req']
            state['seek_req'] = None
            playing = False
            f = seek(cap, state['pos'])
            if f is not None:
                frame = f
        elif playing:
            ok, f = cap.read()
            if ok:
                frame = f
                state['pos'] = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            else:
                playing = False

        disp = frame.copy()
        drawer.draw_overlay(disp)
        ts = state['pos'] / fps
        _hud(disp, 1, f'frame {state["pos"]}  t={int(ts//60):02d}:{ts%60:05.2f}  '
             f'{"PLAY" if playing else "PAUSE"}  drag box = select plate')
        if sug_box is not None:
            sx, sy, sw2, sh2 = sug_box
            cv2.rectangle(disp, (sx, sy), (sx + sw2, sy + sh2),
                          (255, 200, 0), _box_thickness(disp))
            st_lbl = ' [STATIC: logo/parked?]' if state.get('sug_static') else ''
            _hud(disp, 2, f'AUTO {sug_i + 1}/{len(suggestions)} '
                 f'w={sug_box[2]}px{st_lbl} - ENTER accept | b next | drag adjust',
                 (255, 200, 0))
        if quit_guard.pending:
            _hud(disp, 2, 'PRESS q AGAIN TO QUIT', (0, 0, 255))
        # Bar appended BELOW the frame — the full image stays drag-selectable.
        cv2.imshow(WINDOW, np.vstack([disp, _seek_bar_strip(W, state['pos'], total_frames)]))

        # 30ms poll ≈ real-time playback for 25-30fps footage; 15ms while a
        # drag is in progress so the rubber band follows the mouse smoothly.
        key = _wait_key(15 if drawer.start else (30 if playing else 50), 'select')
        if drawer.box:
            DEBUG('select', f'ROI drag={drawer.box} @ frame={state["pos"]}')
            return state['pos'], drawer.box
        if key in K_Q + (K_ESC,):
            if quit_guard.confirmed('select'):
                return None, None
        elif key == ord(' '):
            playing = not playing
            if playing:  # resume sequential reads from current position
                seek(cap, state['pos'])
        elif key in K_A + K_D + K_J + K_L:
            playing = False
            state['pos'] = max(0, min(total_frames - 1, state['pos'] + _seek_step(key)))
            f = seek(cap, state['pos'])
            if f is not None:
                frame = f
        elif key in K_B and detector is not None:
            # Auto-scan (user request: the tool should FIND the best view
            # itself). First press scans the whole video; each press jumps
            # to the next-largest plate appearance with a ready-made box.
            if suggestions is None:
                suggestions = scan_plate_appearances(cap, total_frames,
                                                     detector, hud_frame=frame)
                if not suggestions:
                    log('  Καμία εμφάνιση πινακίδας στη σάρωση — επίλεξε χειροκίνητα')
            if suggestions:
                sug_i = (sug_i + 1) % len(suggestions)
                idx, sbox, _w, sug_static = suggestions[sug_i]
                playing = False
                state['pos'] = idx
                f = seek(cap, idx)
                if f is not None:
                    frame = f
                sug_box = sbox
                state['sug_static'] = sug_static
        elif key in K_ENTER:
            if sug_box is not None:
                DEBUG('select', f'ROI auto-accept={sug_box} @ frame={state["pos"]}')
                return state['pos'], sug_box
            log('  (tip: drag a box with the mouse, or press b for auto-scan)')


# ---------------------------------------------------------------------------
# Phase 2 — tracking + sample collection
# ---------------------------------------------------------------------------

def make_tracker(kind: str):
    # CSRT: best accuracy for small/deforming targets (plates changing scale
    # and angle), ~10-25ms/frame. KCF is ~5x faster but drifts on scale change;
    # offered for very long clips where CSRT is too slow.
    if kind == 'kcf':
        return cv2.TrackerKCF_create()
    return cv2.TrackerCSRT_create()


# One measured plate-geometry gate for EVERY consumer (live preview,
# auto-ALPR, b-scan use kindred logic): a real plate occupies a FRACTION of
# its vehicle crop with plate-like aspect. Field case that forced it: the
# dashcam timestamp/logo bar spans most of the hood "vehicle" and read as a
# confident plate. 0.6 width fraction (the looser of the two values that
# had silently diverged — 0.55 vs 0.6 — chosen so borderline close-ups of
# genuinely wide plates keep passing; the aspect gate does the real work).
PLATE_MAX_WIDTH_FRAC = 0.6
PLATE_ASPECT_RANGE = (1.5, 8.0)


def plate_geometry_ok(width: float, height: float, crop_width: int) -> bool:
    """True when a detector box has believable PLATE geometry inside its
    vehicle crop (see constants above for the why)."""
    if height <= 0 or width > PLATE_MAX_WIDTH_FRAC * crop_width:
        return False
    lo, hi = PLATE_ASPECT_RANGE
    return lo <= width / height <= hi


def make_plate_detector(model: str | None = None, conf: float = 0.15):
    """The ONE constructor for the YOLOv9 plate detector (three call sites
    had duplicated these kwargs). CPU provider explicitly: onnxruntime's
    CoreML EP cannot express this model's zero-detection output (dynamic
    {-1} shape with 0 elements) and throws + retries on EVERY empty frame —
    night footage spammed hundreds of errors/min and silently lost
    detections the CPU path finds. The tiny YOLOv9-t only ever sees small
    crops, so CPU costs a few ms."""
    from open_image_models import LicensePlateDetector
    return LicensePlateDetector(detection_model=model or DEFAULT_DETECTOR_MODEL,
                                conf_thresh=conf,
                                providers=['CPUExecutionProvider'])


def bright_plate_quad(crop: np.ndarray):
    """Saturated-plate geometry: on burned night footage a HUMAN instantly
    sees the white parallelogram even though OCR/neural detectors see no
    character texture (user: "can't it spot the white rectangle like a
    person would?"). Threshold the near-clipped range and accept the largest
    solid, plate-shaped rotated rectangle. Returns 4 corner points (float32
    4x2, crop coords) or None. Headlight flares fail the aspect (round) and
    solidity (irregular glow) gates."""
    if crop.size == 0:
        return None
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mask = (g >= 235).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    ch, cw = g.shape
    best, best_area = None, 0.0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 0.02 * cw * ch:
            continue
        (cx, cy), (rw, rh), ang = cv2.minAreaRect(c)
        if rw < rh:
            rw, rh = rh, rw
        if rh <= 1:
            continue
        aspect = rw / rh
        solidity = area / max(1.0, rw * rh)
        if not (1.8 <= aspect <= 7.5) or solidity < 0.75:
            continue
        if area > best_area:
            best_area = area
            best = cv2.boxPoints(((cx, cy), (rw, rh), ang))
    return best


def refine_with_detector(detector, frame, box, pad_ratio=0.6):
    """Run the plate detector on a padded crop around `box`.

    Returns (refined_box, det_conf) in frame coords, or (None, 0.0).
    Padding 0.6: wide enough to recover from moderate tracker drift, small
    enough that a neighbouring car's plate rarely enters the search window.
    """
    crop, (ox, oy) = padded_crop(frame, box, pad_ratio)
    if crop.size == 0:
        return None, 0.0
    dets = detector.predict(crop)
    if not dets:
        # Detection retry on a contrast-enhanced rendering (field case: night
        # IR footage where the plate is a low-contrast/overexposed blob — the
        # detector sees no character texture on the raw pixels but often does
        # after CLAHE + mild gamma). CONFIRM-ONLY: a boosted detection may
        # validate the box the tracker already has (IoU gate below) but never
        # relocate it — contrast artifacts produce plausible false plates,
        # and an early version of this retry derailed tracking on the
        # ground-truth clip by re-anchoring onto them. Crops still come from
        # the ORIGINAL frame either way.
        try:
            lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            lab[..., 0] = clahe.apply(lab[..., 0])
            boosted = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            # Mild highlight compression pulls near-clipped plates back into
            # a range where edges reappear.
            lut = np.array([int(255 * (i / 255.0) ** 1.6)
                            for i in range(256)], np.uint8)
            for d in detector.predict(cv2.LUT(boosted, lut)):
                bb = d.bounding_box
                cand = (int(bb.x1) + ox, int(bb.y1) + oy,
                        int(bb.x2 - bb.x1), int(bb.y2 - bb.y1))
                if iou(cand, box) >= 0.25:
                    # Confirm the CURRENT box with the boosted confidence —
                    # geometry stays the tracker's.
                    return box, float(d.confidence)
        except Exception:  # noqa: BLE001
            pass
        # Last resort — SATURATED plate: neural detection needs character
        # texture, but a burned plate is still an unmistakable white
        # parallelogram. Geometric confirm (aspect+solidity gates), same
        # confirm-only rule: validates the tracker's box, never moves it.
        try:
            quad = bright_plate_quad(crop)
            if quad is not None:
                qx1, qy1 = quad[:, 0].min() + ox, quad[:, 1].min() + oy
                qx2, qy2 = quad[:, 0].max() + ox, quad[:, 1].max() + oy
                cand = (int(qx1), int(qy1), int(qx2 - qx1), int(qy2 - qy1))
                if iou(cand, box) >= 0.25:
                    return box, DET_ACCEPT  # minimal confirming confidence
        except Exception:  # noqa: BLE001
            pass
    if not dets:
        return None, 0.0
    # Rank by overlap with the current track, then by center distance — when
    # the tracker has drifted clean OFF the plate every IoU is 0, and the
    # NEAREST detection (not an arbitrary first one) must win.
    cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
    best, best_key, best_conf = None, None, 0.0
    for d in dets:
        bb = d.bounding_box
        cand = (int(bb.x1) + ox, int(bb.y1) + oy, int(bb.x2 - bb.x1), int(bb.y2 - bb.y1))
        ccx, ccy = cand[0] + cand[2] / 2, cand[1] + cand[3] / 2
        key = (iou(cand, box), -((ccx - cx) ** 2 + (ccy - cy) ** 2))
        if best_key is None or key > best_key:
            best, best_key, best_conf = cand, key, float(d.confidence)
    return best, best_conf


def ask_more_segments(cap, pos, total_frames):
    """End-of-segment prompt: the same vehicle often appears again later in
    the video (arrives at the start, leaves at the end). [n] loops back to
    the seek/select phase for another segment — the frames of ALL segments
    then vote together in one candidate list (more frames = measurably
    better odds of the true plate ranking high). ENTER/q proceeds to the
    analysis. GUI-only."""
    frame = seek(cap, max(0, min(pos, total_frames - 1)))
    if frame is None:
        return False
    disp = frame.copy()
    _hud(disp, 1, 'Segment finished - is the SAME vehicle visible '
                  'elsewhere in the video?')
    _hud(disp, 2, '[n] add another segment   [ENTER or q] finish & analyse',
         (0, 255, 0))
    cv2.imshow(WINDOW, disp)
    log('\n[Τμήμα] n = νέο τμήμα (ίδιο όχημα σε άλλο σημείο) · '
        'ENTER/q = ολοκλήρωση & ανάλυση')
    while True:
        key = _wait_key(50, 'segment-prompt')
        if key in K_N:
            return True
        if key in K_ENTER or key in K_Q or key == K_ESC:
            return False


def show_busy(frame, step, text):
    """Paint a processing banner into the tool window between the heavy
    synchronous stages (ECC, fusion, OCR, report). Without it the window
    freezes on the last frame after q-q and the user thinks the tool hung
    (field feedback). cv2.waitKey(1) flushes the paint before the stage
    starts. No-op in --no-gui runs (frame is None)."""
    if frame is None:
        return
    dark = (frame * 0.35).astype(np.uint8)
    _hud(dark, 1, f'PROCESSING {step}/4 - {text}', (0, 255, 255))
    _hud(dark, 2, 'please wait, the report will open when done', (255, 255, 255))
    cv2.imshow(WINDOW, dark)
    cv2.waitKey(1)


def collect_samples(cap, detector, start_frame, roi, args, fps=25.0):
    """Track the plate from start_frame forward, collecting one Sample per frame.

    GUI while tracking: drag a box on the live view = instant correction;
    [SPACE]/[ENTER]/[r] = fix-mode (pause + frame-by-frame rewind + drag);
    [-]/[+] preview speed · [f] full-speed · [q] finish early and compute.
    On track loss it drops into fix-mode automatically (SPACE there = keep
    searching, useful when the plate reappears after an occlusion).
    """
    frame = seek(cap, start_frame)
    if frame is None:
        fail(f'cannot read start frame {start_frame}')
    H, W = frame.shape[:2]
    # Live plate-outline state (WPODNet quad, refreshed at 2Hz on
    # detector-confirmed frames; translated with the box in between).
    live_quad = {'ts': -10.0, 'pts': None, 'box': (0, 0, 0, 0)}

    samples: list[Sample] = []
    tracker = None  # created by reseed() below
    box = roi
    lost_streak = 0
    last_det_box = None  # last detector-CONFIRMED plate position (drift anchor)
    det_miss = 0         # consecutive frames where the detector found nothing
    frame_idx = start_frame
    gui = not args.no_gui
    end_frame = args.end_frame if args.end_frame else float('inf')
    quit_guard = _ConfirmQuit()
    stop_reason = 'end of video'  # overwritten by every other exit path
    # Preview pacing: real-time × speed. Default 0.7× — slightly SLOWER than
    # real time on purpose: field feedback showed that even at 1× users could
    # not hit a key before a drifting box left the plate. [-]/[+] adjust live,
    # [f] runs flat-out for long uneventful stretches.
    base_ms = 1000.0 / fps if fps and fps > 0 else 40.0
    speed_steps = [0.25, 0.5, 0.7, 1.0, 1.5, 2.0]
    speed_i = min(range(len(speed_steps)),
                  key=lambda i: abs(speed_steps[i] - max(0.1, args.speed)))
    fast = False

    drawer = None
    if gui:
        # The window/callback may not exist yet when --roi skipped the
        # selection screen; namedWindow is idempotent so this is safe anyway.
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        drawer = _BoxDrawer()
        cv2.setMouseCallback(WINDOW, drawer.on_mouse)
        log('\n[Track] drag box = instant correction · SPACE fix-mode (rewind) · '
            '-/+ speed · f full speed · q finish')

    def record(fr, b, conf):
        cb = clamp_box(b, W, H)
        if cb is None:
            return False
        x, y, w, h = cb
        crop = fr[y:y + h, x:x + w].copy()
        cw, ch = canonical_size(w, h)
        gray = cv2.cvtColor(cv2.resize(crop, (cw, ch), interpolation=cv2.INTER_CUBIC),
                            cv2.COLOR_BGR2GRAY)
        samples.append(Sample(frame_idx, cb, crop, laplacian_sharpness(gray), conf,
                              sat_ratio=float((gray >= 250).mean())))
        return True

    def reseed(fr, b):
        """(Re)start the tracker on box b; detector-refine first so the tight
        box (not the loose hand-drawn one) seeds tracking and rectification."""
        nonlocal box, tracker, last_det_box, det_miss
        box = b
        rbox, rconf = refine_with_detector(detector, fr, box)
        if rbox is not None and iou(rbox, box) > 0.1:
            box = rbox
            last_det_box, det_miss = rbox, 0
        tracker = make_tracker(args.tracker)
        tracker.init(fr, box)
        record(fr, box, rconf if rbox is not None else 0.0)

    def correct(cur_frame):
        """Enter fix-mode. Returns 'go' to keep tracking or 'stop'."""
        nonlocal frame, frame_idx, lost_streak
        # Frame → tracked-box map so fix-mode can show where the box WAS while
        # the user rewinds — pinpointing the exact frame where tracking failed
        # (frames missing from the map are the ones where the track was lost).
        history = {s.frame_idx: (s.box, s.det_conf) for s in samples}
        action, new_idx, new_roi, new_frame = _correction_mode(cap, cur_frame,
                                                               frame_idx, drawer,
                                                               history)
        if action == 'finish':
            return 'stop'
        # Both a fix and a deliberate resume grant a fresh search window — on
        # a lost track, "SPACE = keep searching" lets the detector reacquire a
        # plate that reappears near its last position after an occlusion.
        lost_streak = 0
        if action == 'resume':
            return 'go'
        # Drop samples at/after the corrected frame: the user may have rewound
        # to BEFORE the drift started, and crops collected from the drifted box
        # would poison the OCR vote (background texture, neighbouring car).
        samples[:] = [s for s in samples if s.frame_idx < new_idx]
        frame_idx = new_idx
        frame = new_frame
        reseed(new_frame, new_roi)
        return 'go'

    reseed(frame, box)

    while True:
        t0 = time.monotonic()
        ok, nxt = cap.read()
        frame_idx += 1
        if not ok or frame_idx > end_frame:
            if gui:
                # Never auto-finish at the end of the video (user request):
                # pause in fix-mode on the last frame so the user can rewind,
                # re-check or redraw; computing starts only on a manual q-q.
                frame_idx -= 1  # stay on the last real frame
                log('  video ended — review/fix if you want; q (twice) to finish')
                if correct(frame) == 'stop':
                    stop_reason = 'end of video (user finished)'
                    break
                continue
            stop_reason = 'end of video' if not ok else 'reached --end-frame'
            break
        frame = nxt

        ok_trk, tbox = tracker.update(frame)
        tbox = tuple(int(v) for v in tbox) if ok_trk else None

        det_conf = 0.0
        # Re-detect periodically (drift correction) and whenever the tracker
        # reports failure (occlusion recovery attempt).
        need_refine = (frame_idx % args.refine_every == 0) or not ok_trk
        if need_refine:
            search_box = tbox if tbox else box  # last known position on loss
            # Widen the detector's search window the longer it keeps missing —
            # whether because the track is fully lost (lost_streak) or because
            # CSRT still "succeeds" while sliding off the plate (det_miss).
            pad = 0.6 + 0.1 * min(max(lost_streak, det_miss), 8)
            rbox, rconf = refine_with_detector(detector, frame, search_box,
                                               pad_ratio=pad)
            if (rbox is None or rconf < DET_ACCEPT) and det_miss >= 3 and last_det_box:
                # CSRT drift carries the search window right off the plate
                # while the tracker still reports success (field case: orange
                # box sliding onto the bumper NEXT TO a clearly visible plate).
                # After a few misses, retry around the last CONFIRMED detection
                # — the plate moves slowly in frame coords, so the old anchor
                # plus a wide pad usually still contains it.
                rbox, rconf = refine_with_detector(detector, frame, last_det_box,
                                                   pad_ratio=pad + 0.4)
            if rbox is not None and rconf >= DET_ACCEPT:
                if tbox is None or iou(rbox, tbox) < 0.5:
                    # Tracker drifted (or died): snap it back onto the detection.
                    tracker = make_tracker(args.tracker)
                    tracker.init(frame, rbox)
                    DEBUG('track', f'detector snap @ frame={frame_idx} conf={rconf:.2f}')
                tbox, det_conf = rbox, rconf
                last_det_box, det_miss = rbox, 0
            else:
                det_miss += 1

        if tbox is not None:
            box = tbox
            lost_streak = 0
            record(frame, box, det_conf)
            # Sustained-drift guard (user request: "stop and ASK me — the
            # orange tracker-only phase is unreliable"). CSRT can slide off
            # the plate while still "succeeding", so hard loss never fires.
            # If the detector confirmed this track before (≥5 times) and has
            # now been silent for ~2.5s of video, pause into fix-mode so the
            # human verifies the box instead of collecting drifted crops.
            # Footage where the detector NEVER confirms (saturated night
            # plates) is exempt — there is nothing to "re-confirm" and the
            # tool would nag forever.
            if (gui and det_conf == 0 and det_miss >= 60
                    and sum(1 for s in samples if s.det_conf > 0) >= 5):
                log(f'  Ο ανιχνευτής έχει να επιβεβαιώσει {det_miss} καρέ — '
                    f'έλεγξε αν το κουτί είναι ακόμα στην πινακίδα')
                DEBUG('track', f'det-miss pause @ frame={frame_idx}')
                det_miss = 0  # one prompt per drift episode, not per frame
                if correct(frame) == 'stop':
                    stop_reason = 'finished from fix-mode after det-miss pause'
                    break
                continue
        else:
            lost_streak += 1
            if lost_streak > args.lost_tolerance:
                if gui:
                    log(f'  track lost around frame {frame_idx} — drag a new box, '
                        'SPACE to keep searching, q to finish')
                    DEBUG('track', f'lost > tolerance @ frame={frame_idx}')
                    # Auto-pause INTO fix-mode: by the time a human reacts to a
                    # lost track the video has moved on; fix-mode also lets them
                    # rewind to where the plate was last visible.
                    if correct(frame) == 'stop':
                        stop_reason = 'finished from fix-mode after lost track'
                        break
                    continue
                else:
                    stop_reason = 'track lost (headless)'
                    break

        if gui:
            disp = frame.copy()
            x, y, w, h = box
            # GREEN = plate detector confirmed the box this frame;
            # ORANGE = tracker-only (normal between detector passes — worry
            # only if it stays orange while sliding off the plate).
            color = (0, 255, 0) if det_conf > 0 else (0, 200, 255)
            cv2.rectangle(disp, (x, y), (x + w, y + h), color, _box_thickness(disp))
            # True plate outline (user request): WPODNet quad, refreshed at
            # most twice a second on detector-confirmed frames — the box is
            # the tracker's container, the quad is the plate itself.
            if det_conf > 0 and not getattr(args, 'no_rectify', False):
                now_q = time.monotonic()
                if now_q - live_quad['ts'] >= 0.5:
                    live_quad['ts'] = now_q
                    crop_q = frame[y:y + h, x:x + w]
                    if crop_q.size:
                        q, _ = estimate_quad(crop_q, args)
                        if q is None:
                            # Saturated plates: WPOD sees no texture, but the
                            # white parallelogram's own corners outline the
                            # plate exactly — even at an angle.
                            bq = bright_plate_quad(crop_q)
                            q = ([(float(px), float(py)) for px, py in bq]
                                 if bq is not None else None)
                        live_quad['pts'] = ([(int(qx + x), int(qy + y))
                                             for qx, qy in q] if q else None)
                        live_quad['box'] = box
            if live_quad.get('pts'):
                # Translate the last quad by the box's motion since it was
                # computed — follows the plate between the 2Hz refreshes.
                bx0, by0 = live_quad['box'][:2]
                dx, dy = x - bx0, y - by0
                pts = np.array([(px + dx, py + dy)
                                for px, py in live_quad['pts']])
                cv2.polylines(disp, [pts], True, (80, 255, 80),
                              max(1, _box_thickness(disp) - 1))
            _hud(disp, 1, f'frame {frame_idx}  samples {len(samples)}  '
                 f'{"FAST" if fast else f"{speed_steps[speed_i]:g}x"}')
            # Persistent detector status (user feedback: an indicator that only
            # appears sometimes reads as "missing"): DET OK = plate confirmed
            # by the detector on this very frame; DET MISS N = running on
            # tracker-only estimates for N consecutive frames.
            if det_conf > 0:
                _hud(disp, 2, f'DET OK {det_conf:.2f}', (0, 255, 0))
            else:
                _hud(disp, 2, f'DET MISS {det_miss}', (0, 200, 255))
            if quit_guard.pending:
                _hud(disp, 3, 'PRESS q AGAIN TO FINISH', (0, 0, 255))
            cv2.imshow(WINDOW, disp)
            # Sleep away whatever is left of the frame budget so the preview
            # runs at the chosen fraction of real time instead of CPU speed.
            frame_budget = base_ms / speed_steps[speed_i]
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            delay = 1 if fast else max(1, int(frame_budget - elapsed_ms))
            key = _wait_key(delay, 'track')
            # Live drag-correction: the video freezes while the user draws a
            # new box directly on the tracking view; on release the tracker
            # reseeds instantly — no mode switch, no confirmation key.
            while drawer.start:
                disp2 = frame.copy()
                drawer.draw_overlay(disp2)
                cv2.imshow(WINDOW, disp2)
                _wait_key(15, 'track')
            if drawer.box:
                nb = drawer.box
                drawer.box = None
                DEBUG('track', f'live drag correction={nb} @ frame={frame_idx}')
                # The current frame's sample came from the old (probably
                # drifted) box — drop it; reseed re-records it corrected.
                samples[:] = [s for s in samples if s.frame_idx < frame_idx]
                lost_streak = 0
                reseed(frame, nb)
            elif key in K_Q:
                if quit_guard.confirmed('track'):
                    stop_reason = 'q pressed'
                    break
            elif key in K_F:
                fast = not fast
            elif key in (ord('-'), ord('_')):
                speed_i = max(0, speed_i - 1)
            elif key in (ord('+'), ord('=')):
                speed_i = min(len(speed_steps) - 1, speed_i + 1)
            elif key in (ord(' '), *K_ENTER, *K_R):
                # SPACE, ENTER and r all open fix-mode: it doubles as the
                # pause screen, so there is no separate pause menu to learn.
                if correct(frame) == 'stop':
                    stop_reason = 'finished from fix-mode'
                    break

    # Always state WHY tracking ended — the one line that turns "it quit on
    # its own??" reports into diagnosable ones even without --debug.
    log(f'  tracking ended: {stop_reason} ({len(samples)} samples)')
    DEBUG('track', f'ended: {stop_reason}, samples={len(samples)}')
    return samples


def _correction_mode(cap, cur_frame, cur_idx, drawer, history=None):
    """Fix-mode: paused seek (backward too) + direct drag of a new plate box.

    Doubles as the pause screen. `history` maps frame_idx → (box, det_conf)
    for every frame that produced a sample; while rewinding, the box the
    tracker HAD on each frame is overlaid so the user can see the exact frame
    where it slipped off the plate. Returns (action, frame_idx, roi, frame):
      ('fixed', idx, roi, frame)      — user dragged a new box at frame idx
      ('resume', cur_idx, None, ...)  — SPACE/ESC: continue without changes
      ('finish', cur_idx, None, ...)  — q: stop tracking and compute results
    """
    idx, frame = cur_idx, cur_frame
    drawer.box = None  # a stale drag from the live view must not fire here
    quit_guard = _ConfirmQuit()
    log('  [Fix] drag new box · a/d ±1 · j/l ±25 · SPACE resume · q finish')
    while True:
        disp = frame.copy()
        past = (history or {}).get(idx)
        if past:
            # Same color code as live tracking: green = detector-confirmed,
            # orange = tracker-only. Thin line so it reads as "history", not
            # as a live/new selection.
            (px, py, pw, ph), pconf = past
            color = (0, 255, 0) if pconf > 0 else (0, 200, 255)
            cv2.rectangle(disp, (px, py), (px + pw, py + ph), color,
                          max(1, disp.shape[1] // 3000))
        drawer.draw_overlay(disp)
        status = 'tracked' if past else 'NO TRACK'
        _hud(disp, 2, f'FIX @ {idx} [{status}] — drag new box | a/d j/l seek | '
             'SPACE resume | q finish', (0, 0, 255))
        # Same detector-status line as live tracking (user request), but here
        # it reflects what the detector had found on the CURRENTLY SHOWN frame
        # — stepping back reveals exactly where confirmations stopped before
        # the track was lost.
        if past is None:
            _hud(disp, 3, 'NO TRACK', (0, 0, 255))
        elif past[1] > 0:
            _hud(disp, 3, f'DET OK {past[1]:.2f}', (0, 255, 0))
        else:
            _hud(disp, 3, 'DET MISS (tracker-only)', (0, 200, 255))
        cv2.imshow(WINDOW, disp)
        key = _wait_key(15 if drawer.start else 50, 'fix')
        if drawer.box:
            roi = drawer.box
            drawer.box = None
            DEBUG('fix', f'drag box={roi} @ frame={idx}')
            return 'fixed', idx, roi, frame
        if key in K_A + K_D + K_J + K_L:
            idx = max(0, idx + _seek_step(key))
            f = seek(cap, idx)
            if f is not None:
                frame = f
        elif key in (ord(' '), K_ESC):
            # Restore the sequential read position so tracking resumes exactly
            # where it was paused (seek() re-reads cur_idx, leaving the capture
            # positioned at cur_idx+1 for the next cap.read()).
            seek(cap, cur_idx)
            DEBUG('fix', 'resume')
            return 'resume', cur_idx, None, cur_frame
        elif key in K_Q:
            if quit_guard.confirmed('fix'):
                return 'finish', cur_idx, None, cur_frame
        elif key in K_ENTER:
            log('  (tip: drag the new box directly with the mouse)')


# ---------------------------------------------------------------------------
# Phase 3 — rectification + ECC sub-pixel alignment
# ---------------------------------------------------------------------------

def rectify_and_align(samples: list[Sample], args) -> tuple:
    """Resize every crop to a shared canonical size, then ECC-align to the
    sharpest sample. Returns (reference_sample, aligned_count)."""
    if not samples:
        return None, 0

    # Reference = sharpest detector-confirmed sample; sharpness × (0.5+conf/2)
    # so that a razor-sharp tracker-only box doesn't beat a slightly softer but
    # detector-verified one (tracker-only boxes may be badly framed).
    ref = max(samples, key=lambda s: s.sharpness * (0.5 + s.det_conf / 2))
    cw, ch = canonical_size(ref.box[2], ref.box[3])

    # Quad-based tilt correction produces an ADDITIONAL rendering per tilted
    # frame (s.rect_level) — it votes alongside the plain resize instead of
    # replacing it, so imperfect quads can never make results worse.
    rectify_by_quads(samples, cw, ch, args)
    for s in samples:
        s.rect = cv2.resize(s.crop, (cw, ch), interpolation=cv2.INTER_CUBIC)

    ref_gray = cv2.cvtColor(ref.rect, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    aligned = 0
    for s in samples:
        if s is ref:
            s.aligned = ref.rect
            s.ecc_cc = 1.0
            aligned += 1
            continue
        gray = cv2.cvtColor(s.rect, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        # Homography motion model: absorbs the perspective change as the car
        # approaches/turns, which pure translation/affine can't. ECC operates
        # directly on intensities → sub-pixel accuracy on low-res plates where
        # feature-point matching (ORB/SIFT) finds too few keypoints.
        warp = np.eye(3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, args.ecc_iters, 1e-5)
        try:
            cc, warp = cv2.findTransformECC(ref_gray, gray, warp,
                                            cv2.MOTION_HOMOGRAPHY, criteria, None, 5)
            # The final correlation is kept per sample: fusion later drops
            # frames that "converged" onto the wrong optimum (low cc), which
            # otherwise smear the average with misaligned strokes.
            s.ecc_cc = float(cc)
            s.aligned = cv2.warpPerspective(
                s.rect, warp, (cw, ch),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            aligned += 1
        except cv2.error:
            # ECC diverges on heavy blur/occlusion. The sample is excluded
            # from fusion but still contributes an individual OCR vote.
            s.aligned = None
    return ref, aligned


# ---------------------------------------------------------------------------
# Phase 3.5 — quad rectification (WPODNet corner regression)
# ---------------------------------------------------------------------------

# Lazy singleton: the WPODNet predictor loads once (~1s) on first use.
_WPOD = {'predictor': None, 'failed': False}

RECTIFY_MIN_CONF = 0.60   # quad confidence gate
RECTIFY_MIN_TILT = 3.0    # degrees. History: started at 6° (research: no
                          # gain on frontal plates + quad estimator noise is
                          # ±2-3°). Lowered to 3° on user field judgment: on
                          # MARGINAL real plates a few degrees can flip Z<->I,
                          # and since the levelled rendering VOTES (never
                          # replaces), a phantom-tilt correction can only lose
                          # the vote — the gate is purely a compute filter.
                          # Measured on ground truth: no quality change in
                          # either direction; cost negligible.


def _wpod_predictor(args):
    """Load WPODNet from the app models dir / script dir / download. Returns
    the predictor or None (optional capability — plate reading must keep
    working without it)."""
    if _WPOD['failed'] or _WPOD['predictor'] is not None:
        return _WPOD['predictor']
    try:
        from wpodnet import Predictor, load_wpodnet_from_checkpoint
        candidates = [
            Path.home() / 'Library/Application Support/com.visionx.app/models/wpodnet.pth',
            Path(__file__).parent / 'models' / 'wpodnet.pth',
        ]
        ckpt = next((p for p in candidates if p.exists()), None)
        if ckpt is None:
            import urllib.request
            ckpt = candidates[0]
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            log('  Λήψη μοντέλου ισιώματος (wpodnet.pth, ~6MB)...')
            urllib.request.urlretrieve(
                'https://github.com/Pandede/WPODNet-Pytorch/releases/download/1.0.0/wpodnet.pth',
                str(ckpt))
        _WPOD['predictor'] = Predictor(load_wpodnet_from_checkpoint(str(ckpt)))
    except Exception as e:  # noqa: BLE001 — optional capability
        log(f'  (ίσιωμα μη διαθέσιμο: {e} — συνεχίζω χωρίς)')
        _WPOD['failed'] = True
    return _WPOD['predictor']


def _quad_tilt_deg(quad) -> float:
    """Max deviation from horizontal/vertical of the quad's top & left edges."""
    (x0, y0), (x1, y1), _, (x3, y3) = quad
    top = abs(math.degrees(math.atan2(y1 - y0, max(1e-6, x1 - x0))))
    left = abs(90.0 - abs(math.degrees(math.atan2(y3 - y0, x3 - x0))))
    return max(top, left)


def estimate_quad(crop: np.ndarray, args):
    """Plate quad inside a (tight) crop via WPODNet corner regression.
    Returns (quad 4x(x,y) in crop coords, confidence) or (None, 0.0).
    The crop gets a 25% replicated border first — the net expects some
    context around the plate, which tight tracker boxes lack."""
    predictor = _wpod_predictor(args)
    if predictor is None:
        return None, 0.0
    try:
        from PIL import Image
        h, w = crop.shape[:2]
        pad = max(8, int(0.25 * max(w, h)))
        padded = cv2.copyMakeBorder(crop, pad, pad, pad, pad,
                                    cv2.BORDER_REPLICATE)
        pred = predictor.predict(
            Image.fromarray(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)))
        if pred.confidence < RECTIFY_MIN_CONF:
            return None, float(pred.confidence)
        quad = [(float(x) - pad, float(y) - pad) for x, y in pred.bounds]
        return quad, float(pred.confidence)
    except Exception:  # noqa: BLE001
        return None, 0.0


def level_by_quad(crop: np.ndarray, quad):
    """Orientation-only levelling: warp the WHOLE crop so the plate quad
    becomes horizontal. The quad supplies ORIENTATION only — never crop
    bounds: WPOD's quad can under-cover the plate (verified: it clipped 'KH'
    off a 'KHE4718' crop and the OCR then misread). Expanded canvas, so
    pixels are never cut. Returns the levelled crop or None. Shared by the
    interactive tool and the auto-ALPR pass (plate_core)."""
    try:
        q = np.float32(quad)
        wt = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
        ht = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
        dstq = np.float32([(0, 0), (wt, 0), (wt, ht), (0, ht)])
        M = cv2.getPerspectiveTransform(q, dstq)
        hh, ww = crop.shape[:2]
        corners = cv2.perspectiveTransform(
            np.float32([[(0, 0), (ww, 0), (ww, hh), (0, hh)]]), M)[0]
        mn = corners.min(axis=0)
        mx = corners.max(axis=0)
        T = np.array([[1, 0, -mn[0]], [0, 1, -mn[1]], [0, 0, 1]],
                     dtype=np.float64)
        out_w = max(8, int(math.ceil(mx[0] - mn[0])))
        out_h = max(8, int(math.ceil(mx[1] - mn[1])))
        return cv2.warpPerspective(crop, T @ M.astype(np.float64),
                                   (out_w, out_h), flags=cv2.INTER_CUBIC,
                                   borderMode=cv2.BORDER_REPLICATE)
    except Exception:  # noqa: BLE001
        return None


def rectify_by_quads(samples: list[Sample], cw: int, ch: int, args) -> int:
    """Warp each crop's estimated plate quad to the fronto-parallel canonical
    rectangle (pure geometric transform of real pixels — evidentiarily
    transparent). Applied ONLY when the tilt is significant (see gates
    above); near-frontal crops keep the plain resize. Returns count."""
    if getattr(args, 'no_rectify', False):
        return 0
    n_rect = 0
    tilts = []
    # Detector-confirmed samples only (same idea as ocr_pool, but computed
    # directly: ocr_pool filters on s.rect which is not set yet at this
    # point in the pipeline — that filter silently skipped everything).
    cand = [s for s in samples if s.det_conf >= 0.2]
    if len(cand) < 5:
        cand = samples
    for s in cand:
        quad, _conf = estimate_quad(s.crop, args)
        if quad is None:
            continue
        tilt = _quad_tilt_deg(quad)
        if tilt < RECTIFY_MIN_TILT:
            continue
        levelled = level_by_quad(s.crop, quad)
        if levelled is None:
            continue
        s.rect_level = cv2.resize(levelled, (cw, ch),
                                  interpolation=cv2.INTER_CUBIC)
        n_rect += 1
        tilts.append(tilt)
    if n_rect:
        log(f'  Ίσιωμα: {n_rect} καρέ απέκτησαν οριζόντια απόδοση '
            f'(μέση κλίση {sum(tilts) / len(tilts):.1f}°) — ψηφίζουν ΚΑΙ οι '
            f'δύο εκδοχές, η ψηφοφορία κρίνει')
    return n_rect


# ---------------------------------------------------------------------------
# Phase 3.6 — regression super-resolution rendering (FSRCNN, non-generative)
# ---------------------------------------------------------------------------

_SRNET = {'sr': None, 'failed': False}

# SR only helps where resolution is the bottleneck; on big crops it wastes
# OCR votes on a near-duplicate rendering.
SR_MAX_WIDTH = 110


def _sr_net():
    """FSRCNN x3 through OpenCV dnn_superres (ships in our contrib build).
    MSE-trained pure regression: sharper learned interpolation of the real
    pixels — architecturally incapable of the character fabrication that
    GAN/diffusion SR exhibits (see research memo). Weights ~40KB, ~8ms/crop
    on CPU. Optional capability: any failure disables it silently."""
    if _SRNET['failed'] or _SRNET['sr'] is not None:
        return _SRNET['sr']
    try:
        from cv2 import dnn_superres
        candidates = [
            Path.home() / 'Library/Application Support/com.visionx.app/models/FSRCNN_x3.pb',
            Path(__file__).parent / 'models' / 'FSRCNN_x3.pb',
        ]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            import urllib.request
            path = candidates[0]
            path.parent.mkdir(parents=True, exist_ok=True)
            log('  Λήψη μοντέλου υπερ-ανάλυσης (FSRCNN_x3.pb, ~40KB)...')
            urllib.request.urlretrieve(
                'https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x3.pb',
                str(path))
        sr = dnn_superres.DnnSuperResImpl_create()
        sr.readModel(str(path))
        sr.setModel('fsrcnn', 3)
        _SRNET['sr'] = sr
    except Exception as e:  # noqa: BLE001
        log(f'  (υπερ-ανάλυση μη διαθέσιμη: {e} — συνεχίζω χωρίς)')
        _SRNET['failed'] = True
    return _SRNET['sr']


def sr_rendering(crop: np.ndarray):
    """3x regression upscale of the RAW crop (before any canonical resize,
    so the network sees the true pixels). Returns the rendering or None."""
    sr = _sr_net()
    if sr is None or crop.size == 0:
        return None
    try:
        return sr.upsample(crop)
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Phase 4 — sharpness-weighted fusion
# ---------------------------------------------------------------------------

def fuse(samples: list[Sample], top_k: int) -> tuple:
    """Average the best ECC-aligned crops, weighted by sharpness².

    Returns (fused_image_or_None, frames_used).

    Weighted mean (not median): with sub-pixel alignment the mean acts as a
    true multi-frame denoiser recovering real detail; median would discard
    half the photometric information. Membership is gated hard, because a
    field case showed a 15-frame fusion looking WORSE than its 3 best
    members (and OCR then confidently misread the mush):
      - ecc_cc >= 0.55: frames whose ECC "converged" onto a wrong optimum
        smear strokes when averaged;
      - sharpness >= 60% of the best member: far dashcam plates produce
        Laplacian variances of 5-15, where near-uniform linear weights just
        average blur back in;
      - sharpness² weights: emphasize the sharpest evidence.
    """
    pool = ocr_pool(samples, 0)  # same detector-confirmed gating as OCR/sheet
    ok = [s for s in pool if s.aligned is not None and s.ecc_cc >= 0.55]
    if len(ok) < 2:
        return None, 0
    ok.sort(key=lambda s: s.sharpness, reverse=True)
    floor = ok[0].sharpness * 0.6
    chosen = [s for s in ok if s.sharpness >= floor][:top_k]
    if len(chosen) < 2:
        return None, 0
    weights = np.array([s.sharpness ** 2 for s in chosen], dtype=np.float64)
    weights /= weights.sum()
    # Saturation-masked accumulation (user request, forensic rationale): a
    # clipped pixel (>=250 gray) carries no scene information, so each pixel
    # position averages ONLY the frames where it isn't burned. When the
    # blown-out region moves across frames (car passes through the light
    # beam), the full plate can assemble itself from the valid parts — pure
    # arithmetic on real pixels, no synthesis. Positions clipped in EVERY
    # frame fall back to the plain weighted mean (stay white).
    acc = np.zeros(chosen[0].aligned.shape, dtype=np.float64)
    plain = np.zeros(chosen[0].aligned.shape, dtype=np.float64)
    wacc = np.zeros(chosen[0].aligned.shape, dtype=np.float64)
    for s, w in zip(chosen, weights):
        img = s.aligned.astype(np.float64)
        valid = (cv2.cvtColor(s.aligned, cv2.COLOR_BGR2GRAY) < 250
                 ).astype(np.float64)[..., None]
        acc += img * w * valid
        wacc += w * valid
        plain += img * w
    fused = np.where(wacc > 1e-9, acc / np.maximum(wacc, 1e-9), plain)
    fused = np.clip(fused, 0, 255).astype(np.uint8)
    coverage = float((wacc[..., 0] > 1e-9).mean())
    if coverage < 0.995:
        log(f'  Fusion με μάσκα κορεσμού: {coverage:.0%} των pixel είχαν '
            f'τουλάχιστον ένα μη-κορεσμένο καρέ'
            + (' — οι λευκές περιοχές είναι καμένες σε ΟΛΑ τα καρέ'
               if coverage < 0.9 else ''))
    # Mild unsharp mask: fusion slightly softens edges (residual sub-pixel
    # error); 0.5 amount restores stroke contrast without ringing. This is
    # light enhancement on REAL averaged pixels — no generative model.
    blur = cv2.GaussianBlur(fused, (0, 0), 1.5)
    return cv2.addWeighted(fused, 1.5, blur, -0.5, 0), len(chosen)


def wiener_deblur(img: np.ndarray, sigma: float = 2.5) -> np.ndarray:
    """Wiener deconvolution with a Gaussian (defocus) PSF — DISPLAY ONLY.

    Ground-truth evaluation (plate YHH3472, 23-PSF grid incl. motion kernels):
    feeding deconvolved renderings into the OCR vote DILUTED the correct
    consensus (P(truth) 0.42 → 0.41 combined, 0.36 alone) because real dashcam
    blur is a compression/demosaic/defocus mix, not a clean linear PSF, and
    deconvolution amplifies its noise. It stays out of the vote; it is offered
    to the HUMAN examiner only (Amped-style optical deblurring rendering),
    where a subtly crisper stroke separation can aid shape completion.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    k = cv2.getGaussianKernel(21, sigma)
    psf = (k @ k.T).astype(np.float32)
    pad = np.zeros_like(gray)
    pad[:21, :21] = psf
    pad = np.roll(pad, (-10, -10), axis=(0, 1))
    H = np.fft.fft2(pad)
    G = np.fft.fft2(gray)
    # K=0.01: noise-to-signal estimate — lower rings badly on 8-bit video noise
    F = np.conj(H) / (np.abs(H) ** 2 + 0.01) * G
    f = np.clip(np.real(np.fft.ifft2(F)) * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)


def human_view(img: np.ndarray, scale: int = 4) -> np.ndarray:
    """Large display version for reading with the naked eye (user request:
    "the fusion must show the number visually"). Percentile contrast stretch
    + CLAHE + unsharp on REAL pixels — a display aid, not evidence and not a
    generative model; it cannot invent strokes, only spread existing tonal
    range so human shape-completion has something to work with."""
    big = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(big, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    lo, hi = np.percentile(l, 2), np.percentile(l, 98)
    if hi > lo:
        l = np.clip((l.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    big = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(big, (0, 0), 2.0)
    return cv2.addWeighted(big, 1.6, blur, -0.6, 0)


# ---------------------------------------------------------------------------
# Phase 5 — OCR + per-position weighted voting
# ---------------------------------------------------------------------------

PAD = '_'  # fast-plate-ocr pad char for unused slots

# Tonal test-time-augmentation LUTs (precomputed once). Gamma 0.55 lifts
# shadows (plate in shade, underexposed night crops); gamma 1.8 compresses
# highlights (headlight glare, overexposed retro-reflective plates). Both are
# monotonic point operations on real pixels: they re-map detail that the OCR's
# fixed input normalization would clip into black/white, but they CANNOT
# invent characters — safe for a candidate-generation tool.
_GAMMA_LOW_LUT = np.array([((i / 255.0) ** 0.55) * 255 for i in range(256)], np.uint8)
_GAMMA_HIGH_LUT = np.array([((i / 255.0) ** 1.8) * 255 for i in range(256)], np.uint8)


def enhance_variants(img: np.ndarray) -> list:
    """Tonal variants OCR'd alongside the original (test-time augmentation).

    Weight factors < 1: the variants act as tie-breakers where the original
    reads disagree, but a systematic mis-read caused by aggressive tone
    mapping can never outvote the unmodified originals.
    """
    variants = [('orig', img, 1.0)]
    # CLAHE on the L channel only: local contrast for unevenly lit plates
    # (e.g. a shadow edge across half the plate) without amplifying chroma noise.
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l2 = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4)).apply(l)
    variants.append(('clahe', cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR), 0.5))
    variants.append(('gamma-', cv2.LUT(img, _GAMMA_LOW_LUT), 0.4))
    variants.append(('gamma+', cv2.LUT(img, _GAMMA_HIGH_LUT), 0.4))
    return variants


def _run_ocr(entry, images):
    """Run one ensemble OCR model. The vit-v2 family expects single-channel
    input for ndarray sources; detect that once per model and remember it."""
    if entry['gray']:
        images = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in images]
        return entry['rec'].run(images, return_confidence=True, remove_pad_char=False)
    try:
        return entry['rec'].run(images, return_confidence=True, remove_pad_char=False)
    except Exception:
        entry['gray'] = True
        images = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in images]
        return entry['rec'].run(images, return_confidence=True, remove_pad_char=False)


def ocr_pool(samples, max_frames):
    """Crops that are allowed to vote / appear on the sheet / be fused.

    Detector-confirmed crops only, when enough exist: ground-truth analysis
    showed every genuinely-wrong crop (bumper, background texture after
    drift) had det_conf == 0, while true-plate crops rarely did.
    """
    pool = [s for s in samples if s.rect is not None]
    det_pool = [s for s in pool if s.det_conf >= 0.2]
    if len(det_pool) >= 5:
        pool = det_pool
    # (1 - sat_ratio): a mostly-burned crop has nothing for OCR to read, so
    # partially/un-saturated frames outrank it. For normal footage
    # (sat_ratio ~0) the ordering is unchanged.
    pool.sort(key=lambda s: s.sharpness * (1.0 - s.sat_ratio), reverse=True)
    return pool[:max_frames] if max_frames else pool


def ocr_and_vote(recognizers, samples, fused_img, args, fused_n=0):
    """OCR every pool crop + fused image (each in 4 tonal variants unless
    --no-enhance) with EVERY ensemble model, vote per character position.

    Returns (candidates, gr_candidates, reads, region_votes).
    """
    # OCR the sharpest max_ocr_frames pool crops: beyond ~60 frames the extra
    # votes are nearly duplicates (adjacent frames) and only add runtime.
    chosen = ocr_pool(samples, args.max_ocr_frames)

    base = [(s.rect, s.sharpness * (0.5 + s.det_conf / 2), f'frame {s.frame_idx}')
            for s in chosen]
    # Tilt-corrected renderings vote alongside the originals (same weight):
    # on genuinely oblique plates they read far better (+14-17pp in the
    # literature), on mild tilt the originals win — per-position voting
    # arbitrates instead of us guessing per video.
    base += [(s.rect_level, s.sharpness * (0.5 + s.det_conf / 2),
              f'frame {s.frame_idx} (level)')
             for s in chosen if s.rect_level is not None]
    # Small plates additionally vote through an FSRCNN x3 regression-SR
    # rendering of the RAW crop (resolution is their bottleneck; the learned
    # upscale reads measurably better than cubic in the literature). Same
    # rule as tilt-levelling: an extra VOTE, never a replacement.
    if not getattr(args, 'no_sr', False):
        for s in chosen:
            if s.crop.shape[1] <= SR_MAX_WIDTH:
                up = sr_rendering(s.crop)
                if up is not None:
                    base.append((up, s.sharpness * (0.5 + s.det_conf / 2),
                                 f'frame {s.frame_idx} (sr)'))
    if fused_img is not None:
        # The fused image is our single best evidence (multi-frame denoised);
        # weight it like `fused_boost` average frames so it can outvote a few
        # noisy single-frame reads but not override broad frame consensus.
        # Full boost only with a solid membership though: a fusion built from
        # few frames is no stronger than a good single frame, and boosting it
        # once let ONE over-smoothed (but confidently OCR'd) fusion outvote
        # the per-frame consensus on real dusk footage.
        mean_w = float(np.mean([w for _, w, _ in base])) if base else 1.0
        boost = args.fused_boost if fused_n >= 8 else 1.0
        base.append((fused_img, mean_w * boost, 'fused'))

    # Expand every image into tonal variants; each variant votes with its
    # base weight × variant factor. OCR is ~2-3ms/image, so even 60 crops × 4
    # variants stays around a second on CPU.
    images, weights, tags, is_orig = [], [], [], []
    for img, w, tag in base:
        vs = [('orig', img, 1.0)] if args.no_enhance else enhance_variants(img)
        for vtag, vimg, factor in vs:
            images.append(vimg)
            weights.append(w * factor)
            tags.append(tag if vtag == 'orig' else f'{tag} ({vtag})')
            is_orig.append(vtag == 'orig')

    # score[pos] = {char: accumulated weight × prob}, across ALL ensemble
    # models (equal model weight — architecture diversity is the point).
    score = [dict() for _ in range(N_SLOTS)]
    reads = []
    region_votes = {}
    wsum = sum(weights) or 1.0

    for m_i, entry in enumerate(recognizers):
        preds = _run_ocr(entry, images)
        for pred, w, tag, orig in zip(preds, weights, tags, is_orig):
            txt = pred.plate[:N_SLOTS].ljust(N_SLOTS, PAD)
            probs = list(pred.char_probs)[:N_SLOTS]
            probs += [0.0] * (N_SLOTS - len(probs))
            if orig and m_i == 0:
                # JSON lists only the primary model's unmodified reads — the
                # tonal variants and other models still vote below, but
                # listing every (model × variant) read would bury the file.
                reads.append({
                    'source': tag,
                    'text': pred.plate.rstrip(PAD),
                    'mean_conf': round(float(np.mean(pred.char_probs)), 4),
                    'weight': round(w / wsum, 4),
                })
            # Only meaningful region tags vote (some models return None/Unknown)
            if pred.region and pred.region != 'Unknown':
                region_votes[pred.region] = region_votes.get(pred.region, 0.0) + w
            for pos, (c, p) in enumerate(zip(txt, probs)):
                score[pos][c] = score[pos].get(c, 0.0) + w * float(p)
                # Each model output is one argmax char per slot; the residual
                # probability mass is unknown, so distribute (1-p) as a small
                # uncertainty credit to PAD to keep positions with weak/absent
                # characters from being dominated by one lucky high-conf read.
                score[pos][PAD] = score[pos].get(PAD, 0.0) + w * (1.0 - float(p)) * 0.25

    # Normalize each position into a probability distribution.
    dists = []
    for pos in range(N_SLOTS):
        total = sum(score[pos].values()) or 1.0
        dists.append({c: v / total for c, v in score[pos].items()})

    candidates = beam_candidates(dists, args.top * 4)  # oversample, dedup below

    # Soft Greek-format prior (LLL NNNN / LLL NNN): matching candidates get a
    # rank bonus. Soft because foreign/diplomatic plates must remain visible.
    import re
    final, seen = [], set()
    for plate, s, per_char in candidates:
        text = plate.rstrip(PAD).replace(PAD, '?')
        if not text or text in seen:
            continue
        seen.add(text)
        pm = any(re.match(p, text) for p in GREEK_PATTERNS)
        if pm and not args.no_pattern_prior:
            s *= 1.15  # +15%: enough to break near-ties, too small to promote a bad read
        # Clamp to 1.0: the pattern bonus can push a near-unanimous read above
        # 1, which looks like nonsense for a score presented on a 0-1 scale.
        final.append(Candidate(text, min(1.0, s), pm, per_char))
    final.sort(key=lambda c: c.score, reverse=True)

    # Re-normalize scores to a 0-1 "confidence" relative to a perfect
    # unanimous read (score 1.0 per position). These are ranking scores for
    # human triage, not calibrated probabilities.
    gr_final = [] if args.no_pattern_prior else gr_projected_candidates(dists, args.top)
    return final[:args.top], gr_final, reads, region_votes


def gr_projected_candidates(dists: list, top_n: int) -> list:
    """Second candidate list with the vote PROJECTED onto the Greek plate
    shapes (3 valid-GR letters + 4 or 3 digits).

    Ground-truth tuning: the free vote often puts an impossible character on
    top (L in a letter slot, a letter in a digit slot) while the correct one
    sits just below; constraining each position to its legal alphabet raised
    mean P(true char) from 0.21 to 0.41 on real dusk footage. Shown ALONGSIDE
    the free list — if the vehicle is foreign, the free list still has it.
    """
    out = []
    for n_digits in (4, 3):  # cars, then motorcycles
        total_len = 3 + n_digits
        pd = []
        for pos in range(len(dists)):
            d = dists[pos]
            if pos < 3:
                keep = {c: v for c, v in d.items() if c in GR_LETTERS}
            elif pos < total_len:
                keep = {c: v for c, v in d.items() if c.isdigit()}
            else:
                keep = {PAD: 1.0}
            t = sum(keep.values())
            if t <= 0:
                keep, t = {PAD: 1.0}, 1.0
            pd.append({c: v / t for c, v in keep.items()})
        for text, s, pc in beam_candidates(pd, top_n * 2):
            t = text.rstrip(PAD)
            if len(t) == total_len and PAD not in t:
                out.append(Candidate(t, min(1.0, s), True, pc))
    out.sort(key=lambda c: c.score, reverse=True)
    dedup, seen = [], set()
    for c in out:
        if c.plate not in seen:
            seen.add(c.plate)
            dedup.append(c)
    return dedup[:top_n]


def beam_candidates(dists: list, top_n: int):
    """Enumerate top-N sequences from independent per-position distributions.

    Positions are independent by construction (votes are per-slot), so the
    exact top-N are combinations of per-position top chars — a small beam
    search over ≤4 chars/position is exhaustive enough in practice.
    """
    per_pos = []
    for d in dists:
        chars = sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:4]
        # Keep alternatives down to 1% of the position max (was 5%): a DB-search
        # tool should always offer top-N candidates, and with a strong consensus
        # (frequent detector refinement) even the 2nd-best char can sit at 2-3%.
        # The ≤4 chars/position cap above keeps the beam bounded regardless.
        cmax = chars[0][1] if chars else 1.0
        per_pos.append([(c, p) for c, p in chars if p >= cmax * 0.01] or [(PAD, 1.0)])

    beams = [('', 1.0, [])]
    for pos, options in enumerate(per_pos):
        nxt = []
        for text, s, pc in beams:
            for c, p in options:
                nxt.append((text + c, s * p,
                            pc + [{'pos': pos, 'char': c, 'prob': round(p, 4),
                                   'alternatives': [{'char': oc, 'prob': round(op, 4)}
                                                    for oc, op in options if oc != c][:3]}]))
        nxt.sort(key=lambda b: b[1], reverse=True)
        beams = nxt[:max(top_n, 16)]  # keep the beam wide enough mid-sequence

    # Geometric-mean normalization: score^(1/len) makes confidences comparable
    # across different plate lengths (7 vs 6 chars) instead of favoring short.
    out = []
    for text, s, pc in beams:
        eff = len(text.rstrip(PAD)) or 1
        out.append((text, s ** (1.0 / eff), pc))
    return out


# ---------------------------------------------------------------------------
# Phase 6 — output
# ---------------------------------------------------------------------------

def assess_readability(candidates, reads, samples=()):
    """Verdict on whether the footage carried readable plate text AT ALL.

    Field case that forced this: a night clip with the retroreflective plate
    fully saturated (pure white) — every OCR read returned empty text, yet
    the vote still ranked stray-position noise ('?????3'), and the Greek
    projection dressed that noise up as plausible-looking plates. An
    investigator must never mistake fabricated structure for a reading, so
    the verdict travels with every output (terminal, JSON, HTML report).

    Returns (level, reason): level ∈ 'ok' | 'low' | 'none'.
    """
    text_reads = sum(1 for r in reads if r.get('text'))
    if text_reads == 0:
        # Saturation forensics: distinguish "burned everywhere" (nothing to
        # do in THIS footage) from "some partially-burned frames exist"
        # (worth pointing the user at them / at other segments).
        sats = [s.sat_ratio for s in samples] if samples else []
        if sats and min(sats) >= 0.5:
            reason = (f'Η πινακίδα είναι κορεσμένη (καμένη από το φως) σε '
                      f'ΟΛΑ τα καρέ — μέσος κορεσμός {sum(sats)/len(sats):.0%}. '
                      f'Σε αυτό το υλικό δεν υπάρχει ανακτήσιμη πληροφορία· '
                      f'δοκιμάστε σημείο όπου η πινακίδα είναι εκτός δέσμης '
                      f'φωτός (π.χ. σε στροφή) ή άλλη κάμερα')
        elif sats and any(v < 0.5 for v in sats):
            n_ok = sum(1 for v in sats if v < 0.5)
            reason = (f'Καμία ανάγνωση OCR δεν επέστρεψε χαρακτήρες, αν και '
                      f'{n_ok} καρέ έχουν μερικό μόνο κορεσμό — πιθανόν πολύ '
                      f'μικρή/θολή πινακίδα· δείτε το φύλλο καρέ (ένδειξη SAT)')
        else:
            reason = ('Καμία ανάγνωση OCR δεν επέστρεψε χαρακτήρες — η '
                      'πινακίδα είναι κορεσμένη/υπερεκτεθειμένη, πολύ μικρή '
                      'ή εκτός εστίασης σε όλα τα καρέ')
        return 'none', reason
    top = candidates[0] if candidates else None
    known = len(top.plate.replace('?', '')) if top else 0
    if top is None or known < 3 or top.score < 0.45 or text_reads < 3:
        return 'low', (f'Ελάχιστες αξιοποιήσιμες αναγνώσεις ({text_reads}) '
                       f'ή πολύ αβέβαιη κορυφαία υποψήφια — χρησιμοποιήστε '
                       f'τις λίστες μόνο ως αχνή ένδειξη')
    return 'ok', ''


def save_outputs(video_path, out_dir, samples, ref, fused_img, candidates, reads,
                 region_votes, roi, start_frame, fps, args, fused_n=0,
                 gr_candidates=(), readability=('ok', '')):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if fused_img is not None:
        cv2.imwrite(str(out / 'fused.png'), fused_img)
        # 4× contrast-stretched version — the one to read with the naked eye.
        cv2.imwrite(str(out / 'fused_large.png'), human_view(fused_img))
    if ref is not None:
        cv2.imwrite(str(out / 'best_frame.png'), ref.rect)
        cv2.imwrite(str(out / 'best_frame_large.png'), human_view(ref.rect))

    # Contact sheet of the sharpest crops: lets the user eyeball what the OCR
    # saw and manually cross-check ambiguous characters — this human check IS
    # part of the intended workflow (candidates, not verdicts). Each tile also
    # carries that frame's OCR read, so per-frame findings are reviewable
    # without opening candidates.json.
    ocr_by_frame = {}
    for r in reads:
        if r['source'].startswith('frame '):
            ocr_by_frame[int(r['source'].split()[1])] = r
    # Same detector-confirmed pool as the OCR vote: keeps drifted bumper/
    # background crops off the sheet (field report: "the best frames include
    # the bumper").
    sheet = None
    ok = ocr_pool(samples, 24)
    if ok:
        cw, ch = ok[0].rect.shape[1], ok[0].rect.shape[0]
        cols = 4
        rows = (len(ok) + cols - 1) // cols
        label_h = 42  # two label lines: frame/sharpness + OCR read
        sheet = np.zeros((rows * (ch + label_h), cols * cw, 3), np.uint8)
        for i, s in enumerate(ok):
            r, c = divmod(i, cols)
            y0 = r * (ch + label_h)
            sheet[y0:y0 + ch, c * cw:c * cw + cw] = s.rect
            sat_lbl = f' SAT{s.sat_ratio:.0%}' if s.sat_ratio >= 0.2 else ''
            cv2.putText(sheet, f'f{s.frame_idx} sh={s.sharpness:.0f} cc={s.ecc_cc:.2f}{sat_lbl}',
                        (c * cw + 4, y0 + ch + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            read = ocr_by_frame.get(s.frame_idx)
            if read:
                # Green = confident read, orange = shaky — matches the live
                # tracking color language.
                col = (0, 255, 0) if read['mean_conf'] >= 0.7 else (0, 200, 255)
                cv2.putText(sheet, f'{read["text"]}  {read["mean_conf"]:.2f}',
                            (c * cw + 4, y0 + ch + 33),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        cv2.imwrite(str(out / 'frames_sheet.png'), sheet)

    result = {
        'video': str(video_path),
        'roi': list(roi),
        'start_frame': start_frame,
        'start_time_sec': round(start_frame / fps, 3),
        'frames_tracked': len(samples),
        'frames_ocred': min(len(samples), args.max_ocr_frames),
        'fused': fused_img is not None,
        'fused_frames': fused_n,
        'tonal_enhancement': not args.no_enhance,
        'region_hint': max(region_votes, key=region_votes.get) if region_votes else None,
        'candidates': [
            {'plate': c.plate, 'score': round(c.score, 4),
             'greek_pattern': c.pattern_match, 'per_char': c.per_char}
            for c in candidates
        ],
        # Vote projected onto valid Greek shapes (letters ABEZHIKMNOPTYX at
        # positions 1-3, digits after) — the stronger list when the vehicle
        # is known/likely Greek; foreign plates live in 'candidates' above.
        'candidates_greek_projected': [
            {'plate': c.plate, 'score': round(c.score, 4), 'per_char': c.per_char}
            for c in gr_candidates
        ],
        'individual_reads': reads,
        'readability': readability[0],
        'readability_note': readability[1],
        'note': 'Candidate list for investigative DB search — not evidentiary. '
                'Scores are relative rankings, not calibrated probabilities.'
                + (' ΠΡΟΣΟΧΗ: readability=none — οι λίστες είναι στατιστικός '
                   'θόρυβος, ΜΗΝ χρησιμοποιηθούν για αναζήτηση.'
                   if readability[0] == 'none' else ''),
    }
    json_path = out / 'candidates.json'
    # utf-8 + ensure_ascii=False: same Windows cp1252 pitfall found in
    # report.py — Greek text in notes must not crash on Windows consoles.
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # Self-contained Greek HTML report — the human-facing presentation layer.
    # Professional forensic tools (Amped FIVE et al.) pair every analysis with
    # a methodology report and multi-rendering review panels; candidates.json
    # stays as the machine-readable counterpart.
    try:
        import plate_report
    except ImportError:
        log('  (plate_report.py not found — skipping HTML report)')
        return json_path, None

    panels = []
    labels = {'orig': 'Αρχικό', 'clahe': 'CLAHE', 'gamma-': 'Γάμμα σκιών',
              'gamma+': 'Γάμμα φώτων'}
    for s in ocr_pool(samples, 6):
        big = cv2.resize(s.rect, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        variants = [(labels[t], im) for t, im, _ in enhance_variants(big)]
        # Negative: not OCR'd (models expect normal polarity) but a standard
        # examiner rendering — stroke gaps often pop in inverted contrast.
        variants.append(('Αρνητικό', 255 - big))
        # Wiener deblur: display-only too (see wiener_deblur docstring).
        variants.append(('Αποθόλωση', wiener_deblur(big)))
        read = ocr_by_frame.get(s.frame_idx)
        panels.append({'frame': s.frame_idx, 'variants': variants,
                       'read': read['text'] if read else None,
                       'conf': read['mean_conf'] if read else 0.0})

    model_names = [m.strip() for m in args.ocr_model.split(',') if m.strip()]
    steps = [
        f'Χειροκίνητη επιλογή ROI στο καρέ {start_frame}· παρακολούθηση CSRT με '
        f'επανεντοπισμό από εξειδικευμένο ανιχνευτή κάθε {args.refine_every} καρέ '
        f'(κατώφλι αποδοχής {DET_ACCEPT}).',
        f'Συλλογή {len(samples)} δειγμάτων· σε ψηφοφορία/παρουσίαση μόνο τα '
        'επιβεβαιωμένα από τον ανιχνευτή (conf ≥ 0.2), εφόσον επαρκούν.',
        f'Αναγωγή κάθε αποκόμματος σε κοινό ύψος {CANON_H}px (κυβική παρεμβολή) '
        f'και υποδιαστηματική ευθυγράμμιση ECC ομογραφίας (έως {args.ecc_iters} '
        'επαναλήψεις).',
        f'Συγχώνευση: σταθμισμένος μέσος όρος (βάρη ευκρίνειας²) σε {fused_n} καρέ '
        'με συσχέτιση ECC ≥ 0.55 + ήπιο unsharp mask.',
        f'OCR: έως {args.max_ocr_frames} αποκόμματα × 4 τονικές παραλλαγές × '
        f'{len(model_names)} μοντέλα (ensemble)· ψηφοφορία ανά θέση χαρακτήρα με '
        'βάρη ευκρίνειας × βεβαιότητας ανίχνευσης.',
        'Κατάταξη υποψηφίων με beam search· ήπια πριμοδότηση ελληνικού σχήματος '
        'και δεύτερη λίστα προβεβλημένη στα έγκυρα ελληνικά γράμματα/ψηφία.',
    ]
    report_path = plate_report.generate(
        out, result,
        {'fused_large': human_view(fused_img) if fused_img is not None else None,
         'best_frame_large': human_view(ref.rect) if ref is not None else None,
         'fused_deconv': human_view(wiener_deblur(fused_img))
                         if fused_img is not None else None,
         'sheet': sheet},
        panels,
        {'steps': steps, 'models': model_names, 'detector': args.detector_model})
    return json_path, report_path


def print_results(candidates, gr_candidates, reads, region_votes, n_samples,
                  aligned, fused_n, readability=('ok', '')):
    """Print the summary AND return it as lines, so main() can also save it
    as summary.txt in the output folder (user request: the report must live
    with the rest of the artifacts, in copy-pasteable text form too)."""
    lines = []
    if readability[0] == 'none':
        lines += ['!' * 62,
                  '!!  ΜΗ ΑΝΑΓΝΩΣΙΜΗ ΠΙΝΑΚΙΔΑ',
                  f'!!  {readability[1]}.',
                  '!!  Οι παρακάτω λίστες είναι ΣΤΑΤΙΣΤΙΚΟΣ ΘΟΡΥΒΟΣ — μην',
                  '!!  χρησιμοποιηθούν για αναζήτηση σε βάση δεδομένων.',
                  '!' * 62, '']
    elif readability[0] == 'low':
        lines += [f'ΠΡΟΣΟΧΗ — χαμηλή αξιοπιστία: {readability[1]}.', '']
    lines += [f'Tracked {n_samples} frames, ECC-aligned {aligned}, '
              f'fusion {f"of {fused_n} frames" if fused_n else "skipped (too few well-aligned frames)"}']
    if region_votes:
        lines.append(f'Region hint: {max(region_votes, key=region_votes.get)}')
    lines.append('')
    lines.append('=== Plate candidates (for DB search — not evidentiary) ===')
    for i, c in enumerate(candidates, 1):
        flag = ' [GR format]' if c.pattern_match else ''
        lines.append(f'  {i}. {c.plate:<10}  score {c.score:.3f}{flag}')
    if not candidates:
        lines.append('  (no readable candidates — try a different segment or larger ROI)')
    if gr_candidates:
        lines.append('')
        lines.append('=== Greek-plate candidates (projected onto the Greek format) ===')
        for i, c in enumerate(gr_candidates, 1):
            lines.append(f'  {i}. {c.plate:<10}  score {c.score:.3f}')
        lines.append('  (use this list if the vehicle is Greek; the free list '
                     'above covers foreign plates)')
    log('')
    for ln in lines:
        log(ln)
    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Multi-frame license plate reading from CCTV/dashcam video. '
                    'Produces a ranked candidate list, not a verdict.')
    p.add_argument('video', help='input video file')
    p.add_argument('--roi', help='initial plate box "x,y,w,h" (skips GUI selection)')
    p.add_argument('--start-frame', type=int, default=0,
                   help='frame to start at (with --roi: tracking start frame)')
    p.add_argument('--end-frame', type=int, default=0, help='stop tracking here (0 = end)')
    p.add_argument('--app-mode', action='store_true',
                   help='launched from the VisionX app: emit a machine-'
                        'readable PLATE_REPORT:: line on stdout and skip the '
                        'browser auto-open (the app shows the report itself)')
    p.add_argument('--no-gui', action='store_true',
                   help='headless mode (requires --roi)')
    p.add_argument('--tracker', choices=['csrt', 'kcf'], default='csrt',
                   help='csrt = accurate (default), kcf = faster but drifts on zoom')
    # Default 1 (was 5, then 2): field testing on real footage still lost the
    # track often, so the detector now verifies EVERY frame — effectively
    # detector-led tracking with CSRT filling in only where detection fails.
    # Costs ~20ms on the small padded crop, within the paced-preview budget.
    p.add_argument('--refine-every', type=int, default=1,
                   help='run plate detector every N frames to correct drift '
                        '(default 1 = every frame)')
    p.add_argument('--speed', type=float, default=0.7,
                   help='preview speed vs real time (default 0.7 = slightly slow so '
                        'drift is catchable; -/+ adjust live, f = full speed)')
    p.add_argument('--lost-tolerance', type=int, default=25,
                   help='frames without track/detection before pausing/stopping (default 25 ≈ 1s)')
    p.add_argument('--max-ocr-frames', type=int, default=60,
                   help='OCR at most the N sharpest crops (default 60)')
    p.add_argument('--fuse-top', type=int, default=15,
                   help='fuse the N sharpest aligned crops (default 15)')
    p.add_argument('--fused-boost', type=float, default=3.0,
                   help='vote weight of the fused image, in average-frame units (default 3)')
    p.add_argument('--ecc-iters', type=int, default=100,
                   help='ECC alignment iterations (default 100)')
    p.add_argument('--top', type=int, default=5, help='number of candidates to output (default 5)')
    p.add_argument('--no-sr', action='store_true',
                   help='disable the FSRCNN super-resolution vote for small plates')
    p.add_argument('--no-rectify', action='store_true',
                   help='disable WPODNet quad rectification of tilted plates')
    p.add_argument('--no-pattern-prior', action='store_true',
                   help='disable the soft Greek LLL-NNNN format ranking bonus')
    p.add_argument('--no-enhance', action='store_true',
                   help='disable tonal OCR variants (CLAHE/gamma test-time augmentation)')
    p.add_argument('--debug', action='store_true',
                   help='write a GUI event log (key codes, drags, exit reasons) '
                        'next to the video, for troubleshooting spurious quits etc.')
    p.add_argument('--ocr-model', default=DEFAULT_OCR_MODELS,
                   help='fast-plate-ocr hub model(s), comma-separated ensemble')
    p.add_argument('--detector-model', default=DEFAULT_DETECTOR_MODEL,
                   help='open-image-models plate detector')
    p.add_argument('--output', help='output directory (default: <video>_plate/)')
    return p.parse_args()


def main():
    args = parse_args()

    # C-level crash diagnostics: if OpenCV's native GUI layer segfaults (the
    # macOS/Cocoa backend has done so in the field), print the Python stack
    # that led into the crashing call instead of dying silently — otherwise
    # such crashes are indistinguishable from a clean quit for the user.
    import faulthandler
    faulthandler.enable()

    if args.no_gui and not args.roi:
        fail('--no-gui requires --roi "x,y,w,h"')

    try:
        from fast_plate_ocr import LicensePlateRecognizer
        from open_image_models import LicensePlateDetector
    except ImportError as e:
        fail(f'missing dependency ({e.name}). Run: pip install fast-alpr onnxruntime '
             'opencv-contrib-python')

    video_path = Path(args.video)
    if not video_path.exists():
        fail(f'video not found: {video_path}')

    if args.debug:
        DEBUG.path = str(video_path.parent / f'{video_path.stem}_plate_debug.log')
        log(f'Debug event log: {DEBUG.path}')

    cap, total_frames, fps = open_video(str(video_path))
    log(f'Video: {video_path.name}  ({total_frames} frames @ {fps:.2f} fps)')

    # Models load once up front (~1-2s incl. first-run download from HF hub);
    # detector runs on crops only, OCR batch-runs at the end — CPU is plenty.
    # conf_thresh 0.15 (was 0.25): the tracking loop applies its own DET_ACCEPT
    # gate on top; the lib-level threshold only needs to let dusk/small-plate
    # detections (0.15-0.35 range) through for the refinement logic to weigh.
    detector = make_plate_detector(args.detector_model)
    model_names = [m.strip() for m in args.ocr_model.split(',') if m.strip()]
    recognizers = [{'name': m, 'rec': LicensePlateRecognizer(m, device='cpu'),
                    'gray': False} for m in model_names]
    log(f'OCR ensemble: {", ".join(model_names)}')

    if args.roi:
        try:
            roi = tuple(int(v) for v in args.roi.split(','))
            assert len(roi) == 4 and roi[2] > 0 and roi[3] > 0
        except (ValueError, AssertionError):
            fail('--roi must be "x,y,w,h" with positive w,h')
        start_frame = args.start_frame
        samples = collect_samples(cap, detector, start_frame, roi, args, fps=fps)
        segments = [(start_frame, roi)]
    else:
        # Multi-segment collection: the same vehicle often shows up more
        # than once (arrives early, leaves late). After each segment the
        # user can seek anywhere and track another appearance — the frames
        # of ALL segments feed one common vote.
        samples = []
        segments = []
        next_pos = args.start_frame
        while True:
            seg_start, seg_roi = gui_pick_roi(cap, total_frames, fps, next_pos,
                                              detector=detector)
            if seg_roi is None:
                if segments:
                    break  # user aborted the EXTRA pick — analyse what we have
                log('aborted.')
                return
            log(f'ROI {seg_roi} @ frame {seg_start} (τμήμα {len(segments) + 1})')
            if seg_roi[2] < 90:
                # Ground-truth calibrated: below ~90px plate width the OCR
                # ceiling dominates (documented Y/Z/3/4 confusions). Steer
                # the user toward the vehicle's closest pass BEFORE they
                # spend minutes tracking a segment that can't read well.
                log(f'  ΠΡΟΣΟΧΗ: η πινακίδα είναι ~{seg_roi[2]}px — κάτω από '
                    f'~90px οι χαρακτήρες συγχέονται συστηματικά. Αν το όχημα '
                    f'φαίνεται πιο ΚΟΝΤΑ σε άλλο σημείο του βίντεο, προτίμησε '
                    f'εκείνο (ή πρόσθεσε το ως τμήμα με n στο τέλος).')
            seg_samples = collect_samples(cap, detector, seg_start, seg_roi,
                                          args, fps=fps)
            segments.append((seg_start, seg_roi))
            samples.extend(seg_samples)
            log(f'Τμήμα {len(segments)}: {len(seg_samples)} καρέ — '
                f'σύνολο {len(samples)}')
            next_pos = max((s.frame_idx for s in seg_samples),
                           default=seg_start) + 1
            if not samples or next_pos >= total_frames - 1:
                break
            if not ask_more_segments(cap, next_pos, total_frames):
                break
        start_frame, roi = segments[0]
        # Segments may overlap if the user seeks backwards — one vote per
        # video frame (first observation wins) so no frame double-votes.
        seen_idx: set = set()
        samples = [s for s in samples
                   if not (s.frame_idx in seen_idx or seen_idx.add(s.frame_idx))]

    # Keep a full frame for the processing banner (GUI runs), then free the
    # capture — every stage after this works on the collected crops.
    busy_frame = None
    if not args.no_gui and samples:
        busy_frame = seek(cap, min(max(s.frame_idx for s in samples),
                                   total_frames - 1))
    cap.release()

    if not samples:
        fail('no usable plate samples collected')
    seg_note = f' από {len(segments)} τμήματα' if len(segments) > 1 else ''
    log(f'\nCollected {len(samples)} samples{seg_note}, rectifying + aligning...')

    show_busy(busy_frame, 1, f'ECC alignment of {len(samples)} frames')
    ref, aligned = rectify_and_align(samples, args)
    show_busy(busy_frame, 2, 'multi-frame fusion')
    fused_img, fused_n = fuse(samples, args.fuse_top)

    n_pool = len(ocr_pool(samples, args.max_ocr_frames))
    log(f'Running OCR ({n_pool} crops'
        f'{" + fused" if fused_img is not None else ""}'
        f'{"" if args.no_enhance else ", x4 tonal variants"}'
        f', x{len(recognizers)} models)...')
    show_busy(busy_frame, 3,
              f'OCR: {n_pool} crops x {len(recognizers)} models')
    candidates, gr_candidates, reads, region_votes = ocr_and_vote(
        recognizers, samples, fused_img, args, fused_n=fused_n)
    readability = assess_readability(candidates, reads, samples)

    show_busy(busy_frame, 4, 'writing the report')
    out_dir = args.output or str(video_path.parent / f'{video_path.stem}_plate')
    json_path, report_path = save_outputs(
        video_path, out_dir, samples, ref, fused_img, candidates,
        reads, region_votes, roi, start_frame, fps, args,
        fused_n=fused_n, gr_candidates=gr_candidates, readability=readability)
    if not args.no_gui:
        cv2.destroyAllWindows()

    summary = print_results(candidates, gr_candidates, reads, region_votes,
                            len(samples), aligned, fused_n,
                            readability=readability)
    # Plain-text copy of the console summary, saved with the artifacts.
    with open(Path(out_dir) / 'summary.txt', 'w', encoding='utf-8') as f:
        f.write(f'Video: {video_path}\nROI: {roi} @ frame {start_frame}\n')
        if len(segments) > 1:
            f.write('Τμήματα: ' + ' · '.join(
                f'#{i + 1} frame {sf} roi {r}'
                for i, (sf, r) in enumerate(segments)) + '\n')
        f.write('\n' + '\n'.join(summary) + '\n')
    log('\nScore = σχετική κατάταξη (0-1), όχι πιθανότητα. "?" = αβέβαιη θέση — '
        'δες εναλλακτικές ανά θέση στο report.')
    if report_path:
        log(f'\nΑναφορά:  {report_path}')
        if getattr(args, 'app_mode', False):
            # The VisionX app parses this stdout marker and shows the report
            # in its own results view (with open-in-browser / show-folder
            # buttons there) — so no browser auto-open in this mode.
            print(f'PLATE_REPORT::{Path(report_path).resolve()}', flush=True)
        elif not args.no_gui:
            # Open the report right away (user request): the analysis ends
            # with the human READING it, not hunting for a file. Batch runs
            # (--no-gui) stay silent so scripts don't spawn browser windows.
            import webbrowser
            webbrowser.open(Path(report_path).resolve().as_uri())
    log(f'Δεδομένα: {json_path}')
    log(f'Εικόνες:  {Path(out_dir) / "fused_large.png"}  (συγχώνευση σε μέγεθος ανάγνωσης)')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        # Mirror any fatal Python error into the --debug event log so field
        # crash reports come with their traceback attached.
        import traceback
        DEBUG('crash', traceback.format_exc())
        raise
