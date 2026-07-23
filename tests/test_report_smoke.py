# -*- coding: utf-8 -*-
"""Smoke tests for the three HTML report generators (report.py,
match_report.py, plate_report.py).

None of these had ANY test before the 2026-07-23 audit — and report.py's
'fps' KeyError (from the same-day playback-overlay fix) is exactly the
class of bug a smoke test catches in milliseconds instead of a user
clicking ▶ on a real video. These tests don't assert pixel-perfect HTML;
they assert the generators run END-TO-END on realistic data (built via the
REAL tracking.py pipeline, not hand-rolled dicts) without raising, and that
the output contains the data the human is meant to see.

Run: venv/bin/python -m unittest discover tests
"""
import copy
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from tracking import TrackCollector, prepare_for_report  # noqa: E402
from report import generate_report  # noqa: E402
from match_report import generate_match_report  # noqa: E402
import plate_report  # noqa: E402

FRAME = np.zeros((480, 640, 3), np.uint8)


def _make_tracks(video_fps=25.0, with_plate=False, with_reappearance=False):
    """Build a small, REALISTIC tracks dict via the actual production
    pipeline (TrackCollector -> finalize -> prepare_for_report) instead of
    hand-rolling every field — this is what caught the 'fps' KeyError:
    prepare_for_report's real output no longer had it, but report.py's
    consumer still expected one."""
    c = TrackCollector(max_snapshots=3)
    for i in range(6):
        c.add(track_id=1, class_name='car', conf=0.85,
              box_xyxy=(100 + i * 40, 200, 260 + i * 40, 340),
              frame_img=FRAME, timestamp=float(i), source_name='a.mp4',
              frame_idx=i * 10)
    for i in range(4):
        c.add(track_id=2, class_name='person', conf=0.7,
              box_xyxy=(400, 250, 440, 400), frame_img=FRAME,
              timestamp=float(i) + 0.5, source_name='a.mp4', frame_idx=i * 8)
    tracks = c.finalize()
    for t in tracks.values():
        t['_video_fps'] = video_fps
    if with_plate:
        tracks[1]['plate'] = {
            'plate': 'ABE1234', 'score': 0.62,
            'candidates': [{'plate': 'ABE1234', 'score': 0.62}],
            'gr_candidates': [{'plate': 'ABE1234', 'score': 0.7}],
            'frames_used': 5, 'plate_px': 140, 'low_conf': False,
            'rel_box': [0.2, 0.6, 0.3, 0.2],
        }
    if with_reappearance:
        tracks[1]['reappearance'] = [
            {'other': 2, 'when': 'later', 'gap': 42.0,
             'score': 0.71, 'evidence': 'appearance'},
        ]
    prepare_for_report(tracks)
    return tracks


class GenerateReportSmokeTest(unittest.TestCase):
    def test_renders_without_raising(self):
        tracks = _make_tracks(with_plate=True, with_reappearance=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = generate_report(tracks, tmp, 'clip', str(Path(tmp) / 'clip.mp4'))
            html = Path(path).read_text(encoding='utf-8')
        self.assertIn('CAR', html)
        self.assertIn('PERSON', html)
        self.assertIn('ABE1234', html)  # the plate chip

    def test_renders_with_playback_overlay_payload(self):
        # Regression target: prepare_for_report emits {video}_video_fps as
        # SECOND-indexed boxes with NO 'fps' key — the exact shape that once
        # crashed generate_report with KeyError('fps').
        tracks = _make_tracks()
        self.assertIn('playback', tracks[1], 'fixture must exercise the overlay path')
        with tempfile.TemporaryDirectory() as tmp:
            path = generate_report(tracks, tmp, 'clip', str(Path(tmp) / 'clip.mp4'))
            html = Path(path).read_text(encoding='utf-8')
        self.assertIn('const PLAYBACK', html)
        self.assertIn('drawClipOverlay', html)

    def test_renders_with_no_tracks(self):
        # An empty result set (nothing detected) must still produce a valid,
        # non-crashing report — not a 500-page traceback.
        with tempfile.TemporaryDirectory() as tmp:
            path = generate_report({}, tmp, 'empty', str(Path(tmp) / 'empty.mp4'))
            self.assertTrue(Path(path).exists())

    def test_combined_report_accepts_multi_video_paths(self):
        # The combined-report path (--finalize-match) passes video_paths so
        # each interval's clip player points at the RIGHT source file.
        tracks = _make_tracks()
        with tempfile.TemporaryDirectory() as tmp:
            video_paths = {'a.mp4': str(Path(tmp) / 'a.mp4'),
                           'b.mp4': str(Path(tmp) / 'b.mp4')}
            path = generate_report(tracks, tmp, 'combined', '',
                                   video_paths=video_paths)
            self.assertTrue(Path(path).exists())


class GenerateMatchReportSmokeTest(unittest.TestCase):
    def test_renders_without_raising(self):
        tracks_a = _make_tracks(with_plate=True)
        tracks_b = copy.deepcopy(_make_tracks(with_plate=True))
        per_video = {'cam1.mp4': tracks_a, 'cam2.mp4': tracks_b}
        groups = [{
            'members': [('cam1.mp4', 1), ('cam2.mp4', 1)],
            'evidence': 'plate+appearance', 'score': 0.91,
            # Shape MUST mirror plate_core.PlateReader.read_from_crops's
            # return dict — that's the only producer (cross_match.
            # combined_plate calls it directly); match_report indexes
            # cp["frames_used"] unconditionally, so a partial fixture here
            # would hide a real KeyError behind a fake bug report.
            'combined_plate': {'plate': 'ABE1234', 'score': 0.8,
                               'frames_used': 6, 'plate_px': 150,
                               'det_conf': 0.9, 'low_conf': False,
                               'candidates': [], 'gr_candidates': []},
        }]
        with tempfile.TemporaryDirectory() as tmp:
            path = generate_match_report(per_video, groups, tmp,
                                         list(per_video.keys()))
            html = Path(path).read_text(encoding='utf-8')
        self.assertIn('toggleSel', html)  # manual pairing JS present
        self.assertIn('ABE1234', html)

    def test_renders_with_no_groups(self):
        # No cross-video matches found — a normal, non-error outcome.
        tracks = {'only.mp4': _make_tracks()}
        with tempfile.TemporaryDirectory() as tmp:
            path = generate_match_report(tracks, [], tmp, list(tracks.keys()))
            self.assertTrue(Path(path).exists())


class PlateReportSmokeTest(unittest.TestCase):
    def _minimal_result(self, readability='ok'):
        return {
            'video': '/tmp/clip.mp4', 'roi': [10, 10, 100, 40],
            'start_frame': 0, 'start_time_sec': 0.0,
            'frames_tracked': 12, 'frames_ocred': 12, 'fused': True,
            'fused_frames': 5, 'tonal_enhancement': True,
            'region_hint': 'Greece',
            'candidates': [{'plate': 'ABE1234', 'score': 0.5,
                           'greek_pattern': True, 'per_char': []}],
            'candidates_greek_projected': [{'plate': 'ABE1234', 'score': 0.6,
                                            'per_char': []}],
            'individual_reads': [], 'readability': readability,
            'readability_note': 'δοκιμαστική σημείωση' if readability != 'ok' else '',
            'note': 'test',
        }

    def _tiny_image(self):
        return cv2.imencode('.jpg', np.zeros((20, 60, 3), np.uint8))[1]

    def test_renders_without_raising(self):
        imgs = {'fused_large': np.zeros((40, 120, 3), np.uint8),
                'best_frame_large': np.zeros((40, 120, 3), np.uint8),
                'fused_deconv': None, 'sheet': None}
        with tempfile.TemporaryDirectory() as tmp:
            path = plate_report.generate(
                tmp, self._minimal_result(), imgs, panels=[],
                meta={'steps': ['ένα βήμα'], 'models': ['cct-s-v2'],
                      'detector': 'yolo-v9-t-384-license-plate-end2end'})
            html = Path(path).read_text(encoding='utf-8')
        self.assertIn('ABE1234', html)

    def test_unreadable_verdict_shows_the_warning_banner(self):
        # The readability='none' banner is the guard against dressing OCR
        # noise up as candidates (a real field incident) — it must render.
        imgs = {'fused_large': None, 'best_frame_large': None,
               'fused_deconv': None, 'sheet': None}
        with tempfile.TemporaryDirectory() as tmp:
            path = plate_report.generate(
                tmp, self._minimal_result(readability='none'), imgs,
                panels=[], meta={'steps': [], 'models': [], 'detector': ''})
            html = Path(path).read_text(encoding='utf-8')
        self.assertIn('ΜΗ ΑΝΑΓΝΩΣΙΜΗ', html)


if __name__ == '__main__':
    unittest.main()
