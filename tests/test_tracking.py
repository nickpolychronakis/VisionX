# -*- coding: utf-8 -*-
"""Unit tests for tracking.py (TrackCollector + report-prep helpers).

This module had ZERO test coverage before the 2026-07-23 audit — and it is
exactly where two real bugs lived (playback overlay drifting off the
vehicle on stitched tracks, and a report-generation KeyError from a stale
'fps' field). These tests pin the behavior that broke, plus the rest of
TrackCollector's contract, so a regression here is caught in <1s instead of
by a user watching a video replay.

Run: venv/bin/python -m unittest discover tests
"""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from tracking import (  # noqa: E402
    TrackCollector, _direction, is_host_geometry, prepare_for_report,
)

FRAME = np.zeros((480, 640, 3), np.uint8)  # (h, w) -> _frame_wh stores (w, h)


class IsHostGeometryTest(unittest.TestCase):
    """The dashcam-hood signature: bottom-anchored, lower-half, wide box."""

    def test_matches_a_wide_bottom_anchored_box(self):
        self.assertTrue(is_host_geometry(50, 300, 590, 478, 640, 480))

    def test_rejects_when_not_touching_the_bottom_edge(self):
        # A normal car in the middle distance never reaches the bottom.
        self.assertFalse(is_host_geometry(50, 300, 590, 400, 640, 480))

    def test_rejects_when_too_high_in_frame(self):
        # Touches the bottom but starts near the top — not a hood shape.
        self.assertFalse(is_host_geometry(50, 50, 590, 478, 640, 480))

    def test_rejects_when_too_narrow(self):
        # Bottom-anchored and low, but a normal car's width, not a hood's.
        self.assertFalse(is_host_geometry(250, 300, 390, 478, 640, 480))


class TrackCollectorAddTest(unittest.TestCase):
    def test_new_track_fields(self):
        c = TrackCollector(max_snapshots=4)
        c.add(track_id=1, class_name='car', conf=0.8,
              box_xyxy=(100, 100, 150, 150), frame_img=FRAME,
              timestamp=1.5, source_name='a.mp4', frame_idx=10)
        t = c.tracks[1]
        self.assertEqual(t['class'], 'car')
        self.assertEqual(t['confidence'], 0.8)
        self.assertEqual(t['first_seen'], 1.5)
        self.assertEqual(t['last_seen'], 1.5)
        self.assertEqual(t['frame_count'], 1)
        self.assertEqual(t['_boxes'], [(10, 100, 100, 150, 150)])
        self.assertEqual(len(t['snapshots']), 1)

    def test_repeated_add_updates_running_state(self):
        c = TrackCollector()
        c.add(track_id=1, class_name='car', conf=0.5,
              box_xyxy=(100, 100, 150, 150), frame_img=FRAME,
              timestamp=0.0, frame_idx=0)
        c.add(track_id=1, class_name='car', conf=0.9,  # higher confidence
              box_xyxy=(105, 105, 155, 155), frame_img=FRAME,
              timestamp=1.0, frame_idx=25)
        t = c.tracks[1]
        self.assertEqual(t['confidence'], 0.9, 'confidence must track the MAX seen')
        self.assertEqual(t['last_seen'], 1.0)
        self.assertEqual(t['frame_count'], 2)
        self.assertEqual(len(t['_boxes']), 2)

    def test_snapshot_cap_keeps_best_scores(self):
        # max_snapshots=2; three FAR-APART-IN-TIME detections (no temporal
        # merging) with distinct scores — only the top 2 must survive.
        c = TrackCollector(max_snapshots=2)
        for i, conf in enumerate([0.3, 0.9, 0.6]):
            c.add(track_id=1, class_name='car', conf=conf,
                  box_xyxy=(100, 100, 150, 150), frame_img=FRAME,
                  timestamp=float(i) * 10.0, frame_idx=i * 300)
        t = c.tracks[1]
        self.assertEqual(len(t['snapshots']), 2)
        kept_scores = sorted(s['score'] for s in t['snapshots'])
        # proxy = conf * sqrt(area); area is identical for all three boxes,
        # so ranking by conf ranks by score — 0.3 (the weakest) must be gone.
        self.assertEqual(len(kept_scores), 2)
        self.assertNotIn(min([0.3, 0.9, 0.6]),
                         [round(s, 6) for s in
                          [sc / math.sqrt(2500) for sc in kept_scores]])

    def test_low_score_rejected_once_cap_is_full(self):
        c = TrackCollector(max_snapshots=2)
        for i, conf in enumerate([0.9, 0.8]):
            c.add(track_id=1, class_name='car', conf=conf,
                  box_xyxy=(100, 100, 150, 150), frame_img=FRAME,
                  timestamp=float(i) * 10.0, frame_idx=i * 300)
        # A third, far-apart-in-time but WEAKER detection must be dropped
        # (early-return path: proxy <= snaps[-1]['score']).
        c.add(track_id=1, class_name='car', conf=0.1,
              box_xyxy=(100, 100, 150, 150), frame_img=FRAME,
              timestamp=20.0, frame_idx=600)
        self.assertEqual(len(c.tracks[1]['snapshots']), 2)

    def test_temporal_diversity_replaces_nearby_weaker_snapshot(self):
        # Two detections CLOSE in time: the later one only survives if it
        # scores higher — otherwise the pair would just crowd one moment.
        c = TrackCollector(max_snapshots=4)
        c.add(track_id=1, class_name='car', conf=0.4,
              box_xyxy=(100, 100, 150, 150), frame_img=FRAME,
              timestamp=0.0, frame_idx=0)
        c.add(track_id=1, class_name='car', conf=0.9,  # better, within window
              box_xyxy=(200, 200, 250, 250), frame_img=FRAME,
              timestamp=0.05, frame_idx=1)
        snaps = c.tracks[1]['snapshots']
        self.assertEqual(len(snaps), 1, 'nearby weaker snapshot must be replaced, not kept alongside')
        self.assertEqual(snaps[0]['box'], (200, 200, 250, 250))


class TrackCollectorFinalizeTest(unittest.TestCase):
    def test_static_track_detected(self):
        c = TrackCollector()
        # Same spot for 6 frames — well under the 25%-of-diagonal threshold.
        for i in range(6):
            c.add(track_id=1, class_name='car', conf=0.8,
                  box_xyxy=(300, 300, 350, 350), frame_img=FRAME,
                  timestamp=float(i), frame_idx=i)
        tracks = c.finalize()
        self.assertTrue(tracks[1]['static'])
        self.assertEqual(tracks[1]['direction'], '●')

    def test_moving_track_not_static(self):
        c = TrackCollector()
        for i in range(6):
            x = 100 + i * 80  # travels far across the frame
            c.add(track_id=1, class_name='car', conf=0.8,
                  box_xyxy=(x, 100, x + 50, 150), frame_img=FRAME,
                  timestamp=float(i), frame_idx=i)
        tracks = c.finalize()
        self.assertFalse(tracks[1]['static'])
        self.assertEqual(tracks[1]['direction'], '→')

    def test_short_track_never_classified_static(self):
        # Fewer than 5 frames: a 2-frame flicker must not read as "parked".
        c = TrackCollector()
        for i in range(2):
            c.add(track_id=1, class_name='car', conf=0.8,
                  box_xyxy=(300, 300, 350, 350), frame_img=FRAME,
                  timestamp=float(i), frame_idx=i)
        tracks = c.finalize()
        self.assertFalse(tracks[1]['static'])

    def test_host_vehicle_flagged_for_hood_geometry(self):
        c = TrackCollector()
        for i in range(10):
            c.add(track_id=1, class_name='car', conf=0.9,
                  box_xyxy=(50, 300, 590, 478), frame_img=FRAME,
                  timestamp=float(i), frame_idx=i)
        tracks = c.finalize()
        self.assertTrue(tracks[1]['host_vehicle'])

    def test_normal_vehicle_not_flagged_as_host(self):
        c = TrackCollector()
        for i in range(10):
            c.add(track_id=1, class_name='car', conf=0.9,
                  box_xyxy=(100, 100, 180, 160), frame_img=FRAME,
                  timestamp=float(i), frame_idx=i)
        tracks = c.finalize()
        self.assertFalse(tracks[1]['host_vehicle'])

    def test_intervals_and_dwell_time(self):
        c = TrackCollector()
        c.add(track_id=1, class_name='person', conf=0.7,
              box_xyxy=(10, 10, 40, 90), frame_img=FRAME,
              timestamp=2.0, source_name='cam.mp4', frame_idx=0)
        c.add(track_id=1, class_name='person', conf=0.7,
              box_xyxy=(15, 10, 45, 90), frame_img=FRAME,
              timestamp=7.0, source_name='cam.mp4', frame_idx=150)
        tracks = c.finalize()
        t = tracks[1]
        self.assertAlmostEqual(t['dwell_time'], 5.0)
        self.assertEqual(t['intervals'], [{'start': 2.0, 'end': 7.0, 'file': 'cam.mp4'}])

    def test_internal_accumulators_are_removed(self):
        c = TrackCollector()
        c.add(track_id=1, class_name='car', conf=0.8,
              box_xyxy=(100, 100, 150, 150), frame_img=FRAME,
              timestamp=0.0, frame_idx=0)
        tracks = c.finalize()
        self.assertNotIn('_pos_sum', tracks[1])
        self.assertNotIn('_pos_sumsq', tracks[1])


class DirectionTest(unittest.TestCase):
    def test_eight_compass_directions(self):
        cases = [
            ((0, 0), (100, 0), '→'),
            ((0, 0), (100, -100), '↗'),
            ((0, 0), (0, -100), '↑'),
            ((0, 0), (-100, -100), '↖'),
            ((0, 0), (-100, 0), '←'),
            ((0, 0), (-100, 100), '↙'),
            ((0, 0), (0, 100), '↓'),
            ((0, 0), (100, 100), '↘'),
        ]
        for first, last, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(_direction(first, last), expected)

    def test_static_forces_the_dot_regardless_of_drift(self):
        # Sub-threshold movement alone reads as stationary too, but the
        # static=True flag from a fixed camera must always win.
        self.assertEqual(_direction((0, 0), (500, 500), static=True), '●')


class PrepareForReportTest(unittest.TestCase):
    """Regression tests for the 2026-07-23 playback-overlay bug: boxes must
    be TIMESTAMP-indexed (seconds) and monotonic, and the payload must NOT
    carry an 'fps' key (report.py's consumer stopped reading one — a stale
    field there is exactly how the KeyError shipped)."""

    def _track_with_boxes(self, boxes, fps=25.0):
        return {
            'class': 'car', 'confidence': 0.9, 'snapshots': [],
            '_boxes': boxes, '_video_fps': fps,
        }

    def test_boxes_are_timestamp_indexed(self):
        tracks = {1: self._track_with_boxes([(25, 10, 10, 50, 50)], fps=25.0)}
        prepare_for_report(tracks)
        ts, x1, y1, x2, y2 = tracks[1]['playback']['boxes'][0]
        self.assertAlmostEqual(ts, 1.0)  # frame 25 @ 25fps = second 1
        self.assertEqual((x1, y1, x2, y2), (10, 10, 50, 50))

    def test_no_fps_key_in_playback_payload(self):
        tracks = {1: self._track_with_boxes([(0, 0, 0, 10, 10)])}
        prepare_for_report(tracks)
        self.assertNotIn('fps', tracks[1]['playback'],
                         "playback payload must not resurrect the 'fps' "
                         "field report.py no longer reads (KeyError regression)")

    def test_out_of_order_boxes_are_sorted(self):
        # Stitching concatenates a merged partner's boxes onto the end —
        # they arrive out of frame order. The binary search in the report's
        # JS overlay requires them monotonic in time.
        unordered = [(90, 0, 0, 10, 10), (5, 1, 1, 11, 11), (40, 2, 2, 12, 12)]
        tracks = {1: self._track_with_boxes(unordered, fps=30.0)}
        prepare_for_report(tracks)
        timestamps = [b[0] for b in tracks[1]['playback']['boxes']]
        self.assertEqual(timestamps, sorted(timestamps))

    def test_downsampling_uses_a_stride_not_a_hard_cap(self):
        # step = max(1, n // 300), then boxes[::step] — NOT a hard 300 cap
        # (a stride of 3 over 1000 points yields 334, not <=300). This
        # documents the real contract so a future "fix" doesn't 'correct'
        # working code based on a wrong assumption.
        n = 1000
        many = [(i, 0, 0, 10, 10) for i in range(n)]
        tracks = {1: self._track_with_boxes(many)}
        prepare_for_report(tracks)
        step = max(1, n // 300)
        self.assertEqual(len(tracks[1]['playback']['boxes']), len(range(0, n, step)))

    def test_no_boxes_means_no_playback_key(self):
        tracks = {1: {'class': 'car', 'confidence': 0.9, 'snapshots': [],
                     '_boxes': []}}
        prepare_for_report(tracks)
        self.assertNotIn('playback', tracks[1])

    def test_internal_keys_popped_and_thumbnail_set(self):
        tracks = {1: {
            'class': 'car', 'confidence': 0.9,
            'snapshots': [{'jpeg': b'\xff\xd8fake', 'context': None, 'ts': 0.0}],
            '_boxes': [], '_max_diag': 1.0, '_emb_vpe': None,
            '_emb_hist': None, '_video_fps': 25.0, '_frame_wh': (640, 480),
        }}
        prepare_for_report(tracks)
        t = tracks[1]
        for key in ('_max_diag', '_emb_vpe', '_emb_hist', '_boxes',
                   '_video_fps', '_frame_wh', 'snapshots'):
            self.assertNotIn(key, t)
        self.assertIsNotNone(t['thumbnail'])


if __name__ == '__main__':
    unittest.main()
