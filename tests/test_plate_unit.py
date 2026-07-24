# -*- coding: utf-8 -*-
"""Fast unit tests for plate.py's pure geometry/voting helpers — no GUI, no
OCR model load, no video decode. These pin the mechanics that the slow e2e
suite (test_plate_slow.py) can only exercise indirectly through a full
tracking session; a regression here fails in milliseconds instead of
minutes, and without needing VISIONX_SLOW_TESTS=1.

Run: venv/bin/python -m unittest discover tests
"""
import math
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import plate as P  # noqa: E402


class SeekStepTest(unittest.TestCase):
    """a/d = +-1 frame, j/l = +-25 — including the Greek-keyboard twins
    (field bug: Greek layout sends codepoints like alpha=945 for 'a')."""

    def test_ascii_keys(self):
        self.assertEqual(P._seek_step(ord('a')), -1)
        self.assertEqual(P._seek_step(ord('d')), 1)
        self.assertEqual(P._seek_step(ord('j')), -25)
        self.assertEqual(P._seek_step(ord('l')), 25)

    def test_greek_layout_twins(self):
        self.assertEqual(P._seek_step(945), -1)   # α (lowercase alpha) = a
        self.assertEqual(P._seek_step(913), -1)   # Α (uppercase alpha) = A
        self.assertEqual(P._seek_step(916), 1)    # Δ = D


class ClampBoxTest(unittest.TestCase):
    def test_box_fully_inside_frame_is_unchanged(self):
        self.assertEqual(P.clamp_box((10, 10, 50, 20), 640, 480),
                         (10, 10, 50, 20))

    def test_box_extending_past_frame_edge_is_clamped(self):
        # x+w and y+h both exceed the frame — clamp must not go negative.
        clamped = P.clamp_box((600, 460, 100, 100), 640, 480)
        self.assertEqual(clamped, (600, 460, 40, 20))

    def test_too_narrow_box_rejected(self):
        self.assertIsNone(P.clamp_box((10, 10, 5, 20), 640, 480))

    def test_too_short_box_rejected(self):
        self.assertIsNone(P.clamp_box((10, 10, 50, 2), 640, 480))

    def test_box_entirely_outside_frame_rejected(self):
        self.assertIsNone(P.clamp_box((-100, -100, 5, 5), 640, 480))


class CanonicalSizeTest(unittest.TestCase):
    def test_normal_plate_aspect_preserved(self):
        w, h = P.canonical_size(220, 60)  # aspect ~3.67, within [1, 8]
        self.assertEqual(h, P.CANON_H)
        self.assertAlmostEqual(w / h, 220 / 60, places=2)

    def test_extremely_wide_box_clamped_to_max_aspect(self):
        w, h = P.canonical_size(2000, 60)  # aspect ~33 -> clamp to 8
        self.assertEqual(w, P.CANON_H * P.MAX_ASPECT)

    def test_extremely_narrow_box_clamped_to_min_aspect(self):
        w, h = P.canonical_size(30, 60)  # aspect 0.5 -> clamp to 1.0
        self.assertEqual(w, P.CANON_H * P.MIN_ASPECT)


class PlateGeometryOkTest(unittest.TestCase):
    """Field case: a dashcam's own timestamp/logo overlay bar spans most of
    its 'vehicle' box and was read as a confident plate. This gate is the
    fix — shared by the live preview and the auto-ALPR pass."""

    def test_normal_plate_proportions_accepted(self):
        self.assertTrue(P.plate_geometry_ok(width=200, height=50, crop_width=400))

    def test_overlay_bar_spanning_most_of_the_crop_rejected(self):
        self.assertFalse(P.plate_geometry_ok(width=350, height=30, crop_width=400))

    def test_square_ish_blob_rejected_by_aspect(self):
        self.assertFalse(P.plate_geometry_ok(width=100, height=90, crop_width=400))

    def test_zero_height_rejected_not_a_crash(self):
        self.assertFalse(P.plate_geometry_ok(width=100, height=0, crop_width=400))


class QuadTiltDegTest(unittest.TestCase):
    def test_axis_aligned_quad_has_zero_tilt(self):
        quad = [(0, 0), (100, 0), (100, 40), (0, 40)]
        self.assertAlmostEqual(P._quad_tilt_deg(quad), 0.0, places=3)

    def test_rotated_quad_reports_its_angle(self):
        angle_deg = 15.0
        rad = math.radians(angle_deg)
        w, h = 100, 40
        # Rotate an axis-aligned rectangle's corners by `angle_deg` about
        # the origin — the helper must recover ~angle_deg from the corners.
        corners = [(0, 0), (w, 0), (w, h), (0, h)]
        rot = [(x * math.cos(rad) - y * math.sin(rad),
               x * math.sin(rad) + y * math.cos(rad)) for x, y in corners]
        self.assertAlmostEqual(P._quad_tilt_deg(rot), angle_deg, delta=0.5)


class BrightPlateQuadTest(unittest.TestCase):
    """Saturated-plate geometry (see plate.py): a human instantly sees the
    white parallelogram on a burned-out night plate even with zero
    character texture — this thresholds and shape-gates for exactly that."""

    def test_finds_a_rotated_saturated_plate(self):
        crop = np.full((300, 400, 3), 40, np.uint8)
        plate_img = np.full((50, 200, 3), 255, np.uint8)
        rot_m = cv2.getRotationMatrix2D((100, 25), 15, 1.0)
        rot = cv2.warpAffine(plate_img, rot_m, (240, 110), borderValue=(40, 40, 40))
        crop[100:210, 80:320] = rot
        quad = P.bright_plate_quad(crop)
        self.assertIsNotNone(quad)

    def test_round_headlight_flare_is_rejected(self):
        crop = np.full((300, 400, 3), 40, np.uint8)
        cv2.circle(crop, (200, 150), 60, (255, 255, 255), -1)
        self.assertIsNone(P.bright_plate_quad(crop))

    def test_empty_crop_returns_none_not_a_crash(self):
        self.assertIsNone(P.bright_plate_quad(np.zeros((0, 0, 3), np.uint8)))


def _one_hot_dist(char: str, prob: float = 0.9) -> dict:
    """A per-position distribution dominated by `char`, with a small
    residual spread across other plausible characters (mirrors real OCR
    output, which is never a pure one-hot)."""
    rest = 1.0 - prob
    return {char: prob, 'X': rest * 0.6, P.PAD: rest * 0.4}


class BeamCandidatesTest(unittest.TestCase):
    def test_top_candidate_is_the_per_position_argmax(self):
        dists = [_one_hot_dist(c) for c in 'ABC123']
        out = P.beam_candidates(dists, top_n=5)
        self.assertEqual(out[0][0], 'ABC123')

    def test_output_entries_are_well_formed(self):
        # NOTE: beam_candidates does NOT guarantee the final list is sorted
        # by the returned (length-normalized) score — entries with a
        # trailing PAD character get a different effective length, and the
        # geometric-mean normalization can reorder across lengths. Callers
        # that need a strict ranking (gr_projected_candidates) re-sort
        # explicitly; this test pins the structural contract instead.
        dists = [_one_hot_dist(c) for c in 'AB1']
        out = P.beam_candidates(dists, top_n=5)
        self.assertGreater(len(out), 0)
        for text, score, per_char in out:
            self.assertIsInstance(text, str)
            self.assertTrue(0.0 < score <= 1.0)
            self.assertEqual(len(per_char), len(dists))

    def test_per_char_alternatives_are_attached(self):
        dists = [_one_hot_dist('A')]
        _text, _score, per_char = P.beam_candidates(dists, top_n=3)[0]
        self.assertEqual(per_char[0]['char'], 'A')
        self.assertTrue(per_char[0]['alternatives'])


class GrProjectedCandidatesTest(unittest.TestCase):
    def test_excludes_illegal_greek_letters_from_letter_positions(self):
        # 'L' is not a valid Greek-plate letter (GR_LETTERS = ABEZHIKMNOPTYX)
        # — the free vote might favor it, but the projection must not.
        dists = ([{'L': 0.7, 'A': 0.3}] +
                 [_one_hot_dist(c) for c in 'BE'] +
                 [_one_hot_dist(d) for d in '1234'])
        out = P.gr_projected_candidates(dists, top_n=5)
        self.assertTrue(out)
        for cand in out:
            self.assertTrue(all(ch in P.GR_LETTERS for ch in cand.plate[:3]))

    def test_digit_positions_never_contain_letters(self):
        dists = [_one_hot_dist(c) for c in 'ABE'] + [_one_hot_dist(d) for d in '1234']
        out = P.gr_projected_candidates(dists, top_n=5)
        for cand in out:
            self.assertTrue(all(ch.isdigit() for ch in cand.plate[3:]))

    def test_no_duplicate_plates_in_output(self):
        dists = [_one_hot_dist(c) for c in 'ABE1234']
        out = P.gr_projected_candidates(dists, top_n=10)
        plates = [c.plate for c in out]
        self.assertEqual(len(plates), len(set(plates)))


class RankPlateAppearancesTest(unittest.TestCase):
    """The b-scan ranking (jump-to-best-appearance). Field bug (2026-07-24):
    a burned-in 70mai dashcam watermark ranked #1 and OCR'd as '70100'. It
    flickered into only ~4 probes (detector conf hovering at the accept
    threshold), so it slipped past the old `len >= 6` static guard and — being
    the widest box in the frame — hijacked the top spot over a real moving
    plate. These pin the fix (static floor lowered to 2: a positionally-frozen
    box is an overlay regardless of probe count)."""

    LOGO = (27, 1875, 416, 121)  # the actual 70mai box coords from the field

    def _moving_plate(self, n=8, x0=2600, y=400, w0=74):
        # A genuine close pass: appears in several probes, MARCHES across the
        # frame and grows as the car nears — never positionally frozen.
        out = []
        for k, idx in enumerate(range(300, 300 + n * 150, 150)):
            w = w0 + k * 2
            out.append((idx, (x0 + k * 40, y, w, int(w / 4.5)), w))
        return out

    def test_frozen_logo_deranked_below_moving_plate(self):
        # The exact field failure: 4-probe frozen 416px logo vs 8-probe moving
        # 88px plate. Before the fix the logo won on raw width.
        found = [(i, self.LOGO, self.LOGO[2]) for i in (100, 700, 1900, 2500)]
        found += self._moving_plate()
        ranked = P.rank_plate_appearances(found)
        self.assertNotEqual(ranked[0][1], self.LOGO,
                            'the burned-in logo must not rank #1')
        logo_flags = [is_stat for _i, b, _w, is_stat in ranked if b == self.LOGO]
        self.assertTrue(all(logo_flags), 'the logo must be tagged static (deranked)')

    def test_two_probe_frozen_logo_is_still_deranked(self):
        # The narrowest escape: a logo that lands in only TWO probes. len<6
        # used to keep it in the moving list; the floor-of-2 fix catches it.
        found = [(100, self.LOGO, self.LOGO[2]), (2500, self.LOGO, self.LOGO[2])]
        found += self._moving_plate(n=4)
        ranked = P.rank_plate_appearances(found)
        self.assertNotEqual(ranked[0][1], self.LOGO)

    def test_moving_plate_seen_in_two_probes_stays_moving(self):
        # Guard against over-deranking: a REAL plate captured in just two
        # probes, clearly displaced between them, must NOT be called static.
        found = [(500, (2600, 400, 90, 20), 90), (800, (2900, 410, 96, 21), 96)]
        ranked = P.rank_plate_appearances(found)
        self.assertFalse(ranked[0][3], 'a displaced 2-probe plate is not static')

    def test_single_probe_giant_ranks_below_multi_probe_plate(self):
        # A one-off huge detection (single probe) must sit below a genuine
        # multi-probe pass even though it is wider — the pre-existing guard.
        found = [(999, (0, 0, 500, 110), 500)]        # lone 500px flicker
        found += self._moving_plate(n=5, w0=80)        # real, several probes
        ranked = P.rank_plate_appearances(found)
        self.assertEqual(ranked[0][2], self._moving_plate(n=5, w0=80)[-1][2],
                         'multi-probe plate wins over a single-probe giant')

    def test_return_tuple_shape_is_idx_box_width_static(self):
        # Contract pinned: gui_pick_roi unpacks exactly (idx, box, w, static).
        ranked = P.rank_plate_appearances(self._moving_plate(n=3))
        self.assertTrue(ranked)
        idx, box, w, is_stat = ranked[0]
        self.assertIsInstance(idx, int)
        self.assertEqual(len(box), 4)
        self.assertIsInstance(is_stat, bool)

    def test_empty_input_returns_empty(self):
        self.assertEqual(P.rank_plate_appearances([]), [])


if __name__ == '__main__':
    unittest.main()
