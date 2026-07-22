# -*- coding: utf-8 -*-
"""SLOW end-to-end plate-pipeline tests (full OCR ensemble runs, ~minutes).

Gated behind VISIONX_SLOW_TESTS=1 so the default suite stays seconds-fast:

    VISIONX_SLOW_TESTS=1 venv/bin/python -m unittest tests.test_plate_slow \
        # (or discover — the gate applies either way)

Each test replays a field scenario that once produced (and then verified
the fix for) a real bug — see synthetic.py for the scenario descriptions.
"""
import json
import os
import tempfile
import unittest
from pathlib import Path

from harness_gui import ENTER, IDLE, Q, drag, run_plate_session
import synthetic

from cross_match import _edit_distance  # reuse — no second implementation

SLOW = os.environ.get('VISIONX_SLOW_TESTS') == '1'


@unittest.skipUnless(SLOW, 'set VISIONX_SLOW_TESTS=1 to run the e2e suite')
class PlateEndToEndTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _candidates(self, out: Path) -> dict:
        return json.loads((out / 'candidates.json').read_text(encoding='utf-8'))

    def test_tilted_plate_reads_correctly(self):
        """20° tilt: the levelled rendering must vote the true plate first
        in the Greek projection (WPOD rectification regression)."""
        clip = self.dir / 'tilted.mp4'
        synthetic.make_tilted_clip(clip)
        out = self.dir / 'out'
        run_plate_session(str(clip),
                          [*IDLE(2), drag(180, 240, 460, 380),
                           *IDLE(40), Q, Q, ENTER, *([Q, Q] * 20)],
                          str(out))
        top_gr = self._candidates(out)['candidates_greek_projected'][0]
        self.assertEqual(top_gr['plate'], 'KHE4718')

    def test_masked_fusion_reassembles_burned_plate(self):
        """Moving burn band (no frame shows the full plate): the fused image
        must still yield the true plate among the Greek candidates."""
        clip = self.dir / 'burn.mp4'
        synthetic.make_moving_burn_clip(clip)
        out = self.dir / 'out'
        # 30 OCR frames (not the harness default 10): the burned letters sit
        # at the recognition boundary and need the fuller vote — same value
        # the original field validation used.
        run_plate_session(str(clip),
                          [*IDLE(2), drag(200, 300, 440, 360),
                           *IDLE(40), Q, Q, ENTER, *([Q, Q] * 20)],
                          str(out),
                          extra_args=['--max-ocr-frames', '30'])
        d = self._candidates(out)
        cands = [c['plate'] for c in
                 d['candidates'] + d['candidates_greek_projected']]
        # What this scenario CERTIFIES is reassembly: most of ABC1234 must
        # be recovered although no frame ever showed it whole. Individual
        # characters at the burn-band boundaries wobble between runs, so the
        # robust check is edit distance to the truth (broken fusion yields
        # blank/garbage at distance >= 5), not an exact string.
        best = min(_edit_distance(p, 'ABC1234') for p in cands)
        self.assertLessEqual(best, 3, cands)

    def test_saturated_plate_flagged_unreadable(self):
        """Fully blown plate: the readability verdict must be 'none' — the
        guard that stops noise being dressed up as candidates."""
        clip = self.dir / 'sat.mp4'
        synthetic.make_saturated_clip(clip)
        out = self.dir / 'out'
        run_plate_session(str(clip),
                          [*IDLE(2), drag(250, 300, 390, 340),
                           *IDLE(40), Q, Q, ENTER, *([Q, Q] * 20)],
                          str(out))
        self.assertEqual(self._candidates(out)['readability'], 'none')


if __name__ == '__main__':
    unittest.main()
