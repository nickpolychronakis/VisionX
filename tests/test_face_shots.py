# -*- coding: utf-8 -*-
"""Tests for face_shots.py (best-face extraction — no recognition).

The real-face detection quality is a field concern (needs real footage);
these tests pin the mechanics: frontality math, graceful no-face behavior,
and model bootstrap. Network-dependent setup is skipped when offline.
Run: venv/bin/python -m unittest discover tests
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from face_shots import _frontality, _resolve_model  # noqa: E402


def landmarks(re_x, le_x, nose_x):
    f = np.zeros(15, dtype=np.float32)
    f[4], f[6], f[8] = re_x, le_x, nose_x
    return f


class FrontalityTest(unittest.TestCase):
    def test_frontal_face_scores_high(self):
        # Nose centered between the eyes → fully frontal.
        self.assertAlmostEqual(_frontality(landmarks(40, 80, 60)), 1.0)

    def test_profile_scores_low(self):
        # Nose almost on top of one eye → strong profile.
        self.assertLess(_frontality(landmarks(40, 80, 42)), 0.1)

    def test_degenerate_landmarks(self):
        self.assertEqual(_frontality(landmarks(50, 50, 50)), 0.0)


class ExtractorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if _resolve_model() is None:
            raise unittest.SkipTest('YuNet model unavailable (offline)')
        from face_shots import FaceExtractor
        cls.extractor = FaceExtractor()

    def test_noise_returns_empty(self):
        rng = np.random.default_rng(11)
        crops = [rng.integers(0, 255, (300, 160, 3), dtype=np.uint8)
                 for _ in range(3)]
        self.assertEqual(self.extractor.best_faces(crops), [])

    def test_empty_input(self):
        self.assertEqual(self.extractor.best_faces([]), [])
        self.assertEqual(self.extractor.best_faces([None]), [])


if __name__ == '__main__':
    unittest.main()
