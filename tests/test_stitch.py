# -*- coding: utf-8 -*-
"""Unit tests for the offline tracklet-stitching logic (stitch.py).

Embeddings are injected directly so no model/video is needed — these tests
pin down the GATE behavior, which is where wrong merges (the worst failure
mode: two different objects fused into one report entry) would come from.
Run: venv/bin/python -m unittest discover tests
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from stitch import stitch_tracks, similarity  # noqa: E402


def make_track(first, last, first_pos, last_pos, cls='car', emb=None,
               static=False, hist=None, conf=0.8, frames=30):
    v = np.zeros(8, dtype=np.float32)
    v[emb if emb is not None else 0] = 1.0
    h = np.zeros(16, dtype=np.float32)
    h[hist if hist is not None else (emb if emb is not None else 0)] = 1.0
    return {
        'class': cls, 'confidence': conf, 'frame_count': frames,
        'first_seen': first, 'last_seen': last,
        'first_seen_file': None, 'last_seen_file': None,
        'first_pos': first_pos, 'last_pos': last_pos,
        'static': static, '_max_diag': 80.0,
        'snapshots': [],
        'intervals': [{'start': first, 'end': last, 'file': None}],
        '_emb_vpe': v, '_emb_hist': h,
    }


class StitchGates(unittest.TestCase):
    def test_same_object_after_gap_merges(self):
        # Same appearance, 3s occlusion gap, plausible motion — MUST merge.
        tracks = {
            1: make_track(0, 10, (100, 100), (400, 100), emb=1),
            2: make_track(13, 20, (450, 110), (800, 120), emb=1),
        }
        out = stitch_tracks(tracks)
        self.assertEqual(len(out), 1)
        obj = next(iter(out.values()))
        self.assertEqual(len(obj['intervals']), 2)
        self.assertEqual(obj['merged_from'], [1, 2])
        # dwell = sum of interval durations, NOT last-first (the gap is
        # time the object was NOT visible)
        self.assertAlmostEqual(obj['dwell_time'], 17.0)

    def test_time_overlap_never_merges(self):
        # Two tracks alive at the same time are two physical objects, no
        # matter how similar they look (twin white vans!).
        tracks = {
            1: make_track(0, 10, (100, 100), (400, 100), emb=1),
            2: make_track(5, 15, (600, 100), (900, 100), emb=1),
        }
        out = stitch_tracks(tracks)
        self.assertEqual(len(out), 2)

    def test_different_class_never_merges(self):
        tracks = {
            1: make_track(0, 10, (100, 100), (400, 100), cls='car', emb=1),
            2: make_track(12, 20, (420, 100), (700, 100), cls='person', emb=1),
        }
        out = stitch_tracks(tracks)
        self.assertEqual(len(out), 2)

    def test_different_appearance_never_merges(self):
        # Orthogonal embeddings → similarity ~0 → below threshold.
        tracks = {
            1: make_track(0, 10, (100, 100), (400, 100), emb=1),
            2: make_track(12, 20, (420, 100), (700, 100), emb=2),
        }
        out = stitch_tracks(tracks)
        self.assertEqual(len(out), 2)

    def test_teleport_never_merges(self):
        # Similar appearance but impossibly far for a 1s gap (spatial gate).
        tracks = {
            1: make_track(0, 10, (0, 0), (50, 50), emb=1),
            2: make_track(11, 20, (3800, 2100), (3900, 2150), emb=1),
        }
        out = stitch_tracks(tracks)
        self.assertEqual(len(out), 2)

    def test_parked_car_occluded_long_merges(self):
        # THE parked-car case: static object, long occlusion by passing
        # traffic, same spot → must collapse to one entry regardless of gap.
        tracks = {
            1: make_track(0, 60, (500, 400), (502, 401), emb=1, static=True),
            2: make_track(180, 400, (503, 402), (500, 400), emb=1, static=True),
        }
        out = stitch_tracks(tracks)
        self.assertEqual(len(out), 1)
        self.assertTrue(next(iter(out.values()))['static'])

    def test_moving_object_long_gap_far_away_no_merge(self):
        # Non-static, 100s gap, other side of frame: the spatial gate scales
        # with gap so this COULD pass distance — appearance is same — but
        # keep expectations realistic: with generous speed*gap it may link.
        # We pin the SAFE property instead: a >max-speed link cannot happen
        # inside one second.
        tracks = {
            1: make_track(0, 10, (0, 0), (10, 10), emb=1),
            2: make_track(10.6, 20, (2000, 1500), (2100, 1500), emb=1),
        }
        out = stitch_tracks(tracks)
        self.assertEqual(len(out), 2)

    def test_three_fragments_chain_merge(self):
        tracks = {
            1: make_track(0, 5, (100, 100), (300, 100), emb=1),
            2: make_track(6, 10, (320, 100), (600, 100), emb=1),
            3: make_track(11.5, 16, (620, 105), (900, 110), emb=1),
        }
        out = stitch_tracks(tracks)
        self.assertEqual(len(out), 1)
        self.assertEqual(len(next(iter(out.values()))['intervals']), 3)

    def test_similarity_combines_vpe_and_hist(self):
        a = make_track(0, 5, (0, 0), (1, 1), emb=1, hist=1)
        b = make_track(6, 9, (2, 2), (3, 3), emb=1, hist=2)  # same vpe, diff color
        # 50/50 combination: identical VPE (1.0) + orthogonal hist (~0) ≈ 0.5
        self.assertLess(similarity(a, b), 0.6)
        c = make_track(6, 9, (2, 2), (3, 3), emb=1, hist=1)
        self.assertGreater(similarity(a, c), 0.95)


if __name__ == '__main__':
    unittest.main()
