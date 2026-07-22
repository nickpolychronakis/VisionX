# -*- coding: utf-8 -*-
"""Synthetic-clip factories for plate-pipeline tests (DRY audit 2026-07-23).

These generators were rewritten ad-hoc in throwaway scripts at least four
times during field debugging (tilted plate, moving burn band, sweeping slit,
fully saturated) and each copy died with its /tmp file. One durable module:
every scenario that once validated a fix is now reproducible forever.

All clips are deterministic (fixed RNG seed) so tests are stable.
"""
from pathlib import Path

import cv2
import numpy as np

# Shared canvas geometry: a "car body" rectangle with a plate region on it,
# roughly matching the crop sizes the real pipeline sees.
_W, _H = 640, 480
_PLATE_TEXT = 'KHE4718'  # letters/digits chosen to exercise the GR alphabet


def _plate_image(text: str = _PLATE_TEXT) -> np.ndarray:
    plate = np.full((70, 260, 3), 232, np.uint8)
    cv2.putText(plate, text, (10, 52), cv2.FONT_HERSHEY_SIMPLEX,
                1.3, (15, 15, 15), 4)
    return plate


def _writer(path: str | Path) -> cv2.VideoWriter:
    return cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'),
                           25, (_W, _H))


def make_tilted_clip(path: str | Path, angle: float = 20.0,
                     frames: int = 50, text: str = _PLATE_TEXT) -> None:
    """Plate rotated by `angle` degrees with mild blur/noise — the scenario
    that calibrated WPOD rectification (orientation-only levelling)."""
    rng = np.random.default_rng(7)
    plate = _plate_image(text)
    vw = _writer(path)
    for _ in range(frames):
        f = np.full((_H, _W, 3), 55, np.uint8)
        cv2.rectangle(f, (140, 220), (500, 440), (75, 75, 75), -1)
        rot_m = cv2.getRotationMatrix2D((130, 35), angle, 1.0)
        rot = cv2.warpAffine(plate, rot_m, (300, 140),
                             borderValue=(75, 75, 75))
        f[240:380, 170:470] = rot
        f = cv2.GaussianBlur(f, (0, 0), 0.8)
        noise = rng.normal(0, 4, f.shape).astype(np.int16)
        f = np.clip(f.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        vw.write(f)
    vw.release()


def make_moving_burn_clip(path: str | Path, frames: int = 60,
                          text: str = 'ABC1234') -> None:
    """A saturated band covers a DIFFERENT third of the plate each phase —
    no single frame shows the whole plate. Validates saturation-masked
    fusion assembling the full plate from the non-burned pixels."""
    vw = _writer(path)
    for i in range(frames):
        f = np.full((_H, _W, 3), 45, np.uint8)
        cv2.rectangle(f, (200, 300), (440, 360), (235, 235, 235), -1)
        cv2.putText(f, text, (210, 348), cv2.FONT_HERSHEY_SIMPLEX,
                    1.35, (20, 20, 20), 4)
        band = (i // (frames // 3)) % 3
        x0 = 200 + band * 80
        f[300:360, x0:x0 + 80] = 255
        cv2.rectangle(f, (160, 260), (480, 420), (70, 70, 70), 3)
        vw.write(f)
    vw.release()


def make_slit_burn_clip(path: str | Path, frames: int = 100,
                        text: str = 'ABC1234') -> None:
    """HARD masked-fusion case: 80% of the plate burned in EVERY frame, a
    narrow clean slit sweeping across — letters must reassemble piece by
    piece (the user's 'can it rebuild each letter?' question, answered)."""
    px, py, pw, ph = 200, 300, 240, 60
    slit_w = 48
    vw = _writer(path)
    for i in range(frames):
        f = np.full((_H, _W, 3), 45, np.uint8)
        cv2.rectangle(f, (px, py), (px + pw, py + ph), (235, 235, 235), -1)
        cv2.putText(f, text, (px + 10, py + 48), cv2.FONT_HERSHEY_SIMPLEX,
                    1.35, (20, 20, 20), 4)
        x0 = px + int((pw - slit_w) * (i % 50) / 49)
        burned = np.full((ph, pw, 3), 255, np.uint8)
        burned[:, x0 - px:x0 - px + slit_w] = f[py:py + ph, x0:x0 + slit_w]
        f[py:py + ph, px:px + pw] = burned
        cv2.rectangle(f, (160, 260), (480, 420), (70, 70, 70), 3)
        vw.write(f)
    vw.release()


def make_saturated_clip(path: str | Path, frames: int = 50) -> None:
    """Fully blown-out plate (pure white, zero characters) — the night-IR
    field case behind the readability verdict ('none') and bright-quad."""
    vw = _writer(path)
    for _ in range(frames):
        f = np.full((_H, _W, 3), 40, np.uint8)
        cv2.rectangle(f, (250, 300), (390, 340), (255, 255, 255), -1)
        cv2.rectangle(f, (200, 250), (440, 420), (70, 70, 70), 3)
        vw.write(f)
    vw.release()
