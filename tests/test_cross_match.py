# -*- coding: utf-8 -*-
"""Unit tests for cross-video matching (cross_match.py) — Phase Ε gates.
Run: venv/bin/python -m unittest discover tests
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from cross_match import (match_videos, plate_match_score,  # noqa: E402
                         find_reappearances)


def track(cls='car', emb=None, hist=None, plate=None):
    v = np.zeros(8, dtype=np.float32)
    v[emb if emb is not None else 0] = 1.0
    h = np.zeros(16, dtype=np.float32)
    h[hist if hist is not None else (emb if emb is not None else 0)] = 1.0
    t = {'class': cls, '_emb_vpe': v, '_emb_hist': h, 'snapshots': []}
    if plate:
        t['plate'] = {'gr_candidates': [{'plate': p, 'score': 0.8} for p in plate],
                      'candidates': []}
    return t


class CrossMatchTest(unittest.TestCase):
    def test_same_plate_matches_across_videos(self):
        per_video = {
            'camA.mp4': {1: track(emb=1, plate=['YHH3472', 'YHH3477'])},
            'camB.mp4': {5: track(emb=2, hist=2, plate=['YHH3472'])},
        }
        groups = match_videos(per_video)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]['members'],
                         [('camA.mp4', 1), ('camB.mp4', 5)])
        self.assertIn('plate', groups[0]['evidence'])

    def test_one_char_plate_disagreement_still_matches(self):
        # Two cameras disagreeing on one OCR character must still link.
        per_video = {
            'a': {1: track(emb=1, plate=['AHH5972'])},
            'b': {2: track(emb=1, plate=['AHH5977'])},
        }
        self.assertEqual(len(match_videos(per_video)), 1)

    def test_different_plates_do_not_match(self):
        per_video = {
            'a': {1: track(emb=1, hist=3, plate=['ABC1234'])},
            'b': {2: track(emb=2, hist=4, plate=['XYZ9876'])},
        }
        self.assertEqual(match_videos(per_video), [])

    def test_appearance_only_needs_high_similarity(self):
        # Identical embeddings, no plates → appearance-tier match.
        per_video = {
            'a': {1: track(cls='person', emb=1)},
            'b': {2: track(cls='person', emb=1)},
        }
        groups = match_videos(per_video)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]['evidence'], 'appearance')
        # Orthogonal appearance → no match.
        per_video['b'][2] = track(cls='person', emb=2, hist=2)
        self.assertEqual(match_videos(per_video), [])

    def test_class_gate(self):
        per_video = {
            'a': {1: track(cls='car', emb=1)},
            'b': {2: track(cls='person', emb=1)},
        }
        self.assertEqual(match_videos(per_video), [])

    def test_same_video_objects_never_group(self):
        # Two identical-looking tracks in the SAME video are two objects.
        per_video = {
            'a': {1: track(emb=1), 2: track(emb=1)},
            'b': {3: track(emb=1)},
        }
        groups = match_videos(per_video)
        self.assertEqual(len(groups), 1)
        members = groups[0]['members']
        self.assertEqual(len(members), 2)
        self.assertEqual(len({m[0] for m in members}), 2)  # distinct videos

    def test_plate_score_uses_candidate_lists(self):
        pa = {'gr_candidates': [{'plate': 'KEE8173'}],
              'candidates': [{'plate': 'YHH3472'}]}
        pb = {'gr_candidates': [{'plate': 'YHH3472'}], 'candidates': []}
        # Headlines differ, but the true plate is in A's free list → high.
        self.assertGreaterEqual(plate_match_score(pa, pb), 0.85)


class ReappearanceTest(unittest.TestCase):
    """Same-video re-identification: disjoint-in-time tracks linked by
    evidence, never merged (annotation only)."""

    @staticmethod
    def timed(t, first, last, static=False):
        t.update({'first_seen': first, 'last_seen': last, 'static': static})
        return t

    def test_plate_link_after_gap(self):
        tracks = {
            1: self.timed(track(emb=1, plate=['YHH3472']), 0.0, 10.0),
            2: self.timed(track(emb=2, hist=2, plate=['YHH3472']), 60.0, 70.0),
        }
        pairs = find_reappearances(tracks)
        self.assertEqual(len(pairs), 1)
        self.assertEqual((pairs[0]['a'], pairs[0]['b']), (1, 2))
        self.assertIn('plate', pairs[0]['evidence'])
        self.assertAlmostEqual(pairs[0]['gap'], 50.0)

    def test_overlapping_tracks_never_link(self):
        # Both visible at once = two different physical objects.
        tracks = {
            1: self.timed(track(emb=1, plate=['YHH3472']), 0.0, 30.0),
            2: self.timed(track(emb=1, plate=['YHH3472']), 20.0, 50.0),
        }
        self.assertEqual(find_reappearances(tracks), [])

    def test_short_gap_is_stitching_territory(self):
        tracks = {
            1: self.timed(track(emb=1, plate=['YHH3472']), 0.0, 10.0),
            2: self.timed(track(emb=1, plate=['YHH3472']), 11.0, 20.0),
        }
        self.assertEqual(find_reappearances(tracks), [])

    def test_appearance_only_is_weak_evidence(self):
        tracks = {
            1: self.timed(track(emb=3, hist=3), 0.0, 5.0),
            2: self.timed(track(emb=3, hist=3), 30.0, 40.0),
        }
        pairs = find_reappearances(tracks)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]['evidence'], 'appearance')

    def test_two_static_tracks_do_not_reappear(self):
        tracks = {
            1: self.timed(track(emb=1, plate=['YHH3472']), 0.0, 10.0, static=True),
            2: self.timed(track(emb=1, plate=['YHH3472']), 60.0, 70.0, static=True),
        }
        self.assertEqual(find_reappearances(tracks), [])

    def test_class_mismatch_never_links(self):
        tracks = {
            1: self.timed(track(cls='car', emb=1), 0.0, 10.0),
            2: self.timed(track(cls='person', emb=1), 60.0, 70.0),
        }
        self.assertEqual(find_reappearances(tracks), [])


if __name__ == '__main__':
    unittest.main()
