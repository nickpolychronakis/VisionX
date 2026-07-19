# -*- coding: utf-8 -*-
"""Tests for attributes.py (dominant-color extraction).
Run: venv/bin/python -m unittest discover tests
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from attributes import dominant_color, person_clothing, vehicle_color  # noqa: E402


def swatch(bgr, h=120, w=200):
    """Colored subject on a NEUTRAL surround — realistic for track crops
    (car body + road/背景). Purely solid images are unrealistic and break
    gray-world white balance by construction (channel means == the color)."""
    img = np.zeros((h, w, 3), np.uint8)
    img[: h // 6] = (235, 235, 235)          # light strip (sky/pavement)
    img[h - h // 6:] = (35, 35, 35)          # dark strip (shadow/tyres)
    img[h // 6: h - h // 6] = bgr            # the subject color
    return img


class ColorTest(unittest.TestCase):
    def test_basic_colors(self):
        cases = [
            ((40, 40, 200), 'κόκκινο'),
            ((200, 60, 40), 'μπλε'),
            ((50, 180, 50), 'πράσινο'),
            ((245, 245, 245), 'λευκό'),
            ((20, 20, 20), 'μαύρο'),
            ((150, 150, 150), 'γκρι/ασημί'),
        ]
        for bgr, expected in cases:
            name, _frac = dominant_color(swatch(bgr))
            self.assertEqual(name, expected, f'bgr={bgr}')

    def test_vehicle_color_votes_across_crops(self):
        crops = [swatch((40, 40, 200)), swatch((45, 45, 190)),
                 swatch((35, 30, 210))]
        res = vehicle_color(crops)
        self.assertIsNotNone(res)
        self.assertEqual(res[0], 'κόκκινο')

    def test_color_cast_removed(self):
        # A silver car under a blue dusk cast (the measured field failure):
        # multiply a gray subject by a blue-ish gain — must STILL read gray.
        img = swatch((160, 150, 135))  # gray with blue-channel lift
        name, _ = dominant_color(img)
        self.assertEqual(name, 'γκρι/ασημί')

    def test_scattered_colors_return_none(self):
        # A "rainbow" object has no honest dominant color.
        rng = np.random.default_rng(3)
        crops = [rng.integers(0, 255, (120, 200, 3), dtype=np.uint8)
                 for _ in range(3)]
        # Random noise decodes to mostly low-sat/grays; accept either None or
        # a low-confidence gray — but never a confident chromatic color.
        res = vehicle_color(crops)
        if res is not None:
            self.assertIn(res[0], ('γκρι/ασημί', 'μαύρο', 'λευκό'))

    def test_person_clothing_split(self):
        # Red top (20-55% of height), blue trousers (55-85%).
        img = np.zeros((200, 80, 3), np.uint8)
        img[:] = (128, 128, 128)
        img[40:110] = (40, 40, 200)   # red band (upper)
        img[110:170] = (200, 60, 40)  # blue band (lower)
        res = person_clothing([img])
        self.assertEqual(res.get('upper', (None,))[0], 'κόκκινο')
        self.assertEqual(res.get('lower', (None,))[0], 'μπλε')


if __name__ == '__main__':
    unittest.main()
