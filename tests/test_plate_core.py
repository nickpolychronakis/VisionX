# -*- coding: utf-8 -*-
"""Integration test for plate_core.PlateReader on a synthetic plate.

Runs the REAL detector + OCR ensemble (models come from the local HF cache;
first run downloads a few MB). Verifies the end-to-end batch path that
vision.py's auto-ALPR uses: crops in → ranked candidates out.
Run: venv/bin/python -m unittest discover tests
"""
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def synthetic_vehicle_crop(text='ABE1234', angle=0.0):
    """A dark 'car rear' with a white plate bearing `text`."""
    img = np.full((240, 320, 3), 70, np.uint8)
    cv2.rectangle(img, (40, 60), (280, 220), (55, 50, 48), -1)
    cv2.rectangle(img, (90, 150), (230, 185), (250, 250, 250), -1)
    cv2.putText(img, text, (95, 176), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (10, 10, 10), 2, cv2.LINE_AA)
    if angle:
        M = cv2.getRotationMatrix2D((160, 120), angle, 1.0)
        img = cv2.warpAffine(img, M, (320, 240))
    return img


class PlateCoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from plate_core import PlateReader
        cls.reader = PlateReader()

    def test_reads_plate_from_crops(self):
        crops = [synthetic_vehicle_crop(),
                 synthetic_vehicle_crop(angle=2),
                 synthetic_vehicle_crop(angle=-2)]
        res = self.reader.read_from_crops(crops)
        self.assertIsNotNone(res)
        self.assertEqual(res['plate'], 'ABE1234')
        self.assertGreaterEqual(res['frames_used'], 2)
        self.assertTrue(res['candidates'])

    def test_no_plate_returns_none(self):
        # Pure noise crops: detector should find nothing → None, not garbage.
        rng = np.random.default_rng(7)
        crops = [rng.integers(0, 255, (200, 300, 3), dtype=np.uint8)
                 for _ in range(3)]
        self.assertIsNone(self.reader.read_from_crops(crops))


if __name__ == '__main__':
    unittest.main()
