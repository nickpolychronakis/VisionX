# -*- coding: utf-8 -*-
"""Tests for stage-2 prompt filtering (prompt_filter.py).
Run: venv/bin/python -m unittest discover tests
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from prompt_filter import apply_prompts, parse_prompt, structured_match  # noqa: E402

VEH = ('car', 'αυτοκίνητο', 'όχημα')
PER = ('person', 'άτομο')


def track(cls='car', color=None, clothing=None, emb=None):
    t = {'class': cls}
    if color:
        t['attrs'] = {'color': color, 'color_conf': 0.8}
    if clothing:
        t['attrs'] = {'clothing': {k: {'color': v, 'conf': 0.7}
                                   for k, v in clothing.items()}}
    if emb is not None:
        v = np.zeros(8, dtype=np.float32)
        v[emb] = 1.0
        t['_emb_vpe'] = v
    return t


class ParseTest(unittest.TestCase):
    def test_greek_color_and_class(self):
        p = parse_prompt('λευκό αυτοκίνητο', VEH, PER)
        self.assertEqual(p['colors'], ['λευκό'])
        self.assertTrue(p['wants_vehicle'])
        self.assertFalse(p['wants_person'])


class StructuredTest(unittest.TestCase):
    def test_color_class_match(self):
        self.assertTrue(structured_match(
            track('car', color='λευκό'),
            parse_prompt('λευκό αυτοκίνητο', VEH, PER), VEH, PER))

    def test_wrong_color_rejected(self):
        self.assertFalse(structured_match(
            track('car', color='κόκκινο'),
            parse_prompt('λευκό αυτοκίνητο', VEH, PER), VEH, PER))

    def test_wrong_class_rejected(self):
        self.assertFalse(structured_match(
            track('person'),
            parse_prompt('λευκό αυτοκίνητο', VEH, PER), VEH, PER))

    def test_clothing_color(self):
        self.assertTrue(structured_match(
            track('person', clothing={'upper': 'κόκκινο'}),
            parse_prompt('άτομο με κόκκινη μπλούζα'.replace('κόκκινη', 'κόκκινο'),
                         VEH, PER), VEH, PER))

    def test_unknown_prompt_defers_to_semantic(self):
        self.assertIsNone(structured_match(
            track('car'),
            parse_prompt('σχάρα οροφής', VEH, PER), VEH, PER))


class ApplyTest(unittest.TestCase):
    def test_structured_annotation(self):
        tracks = {1: track('car', color='λευκό'),
                  2: track('car', color='μαύρο'),
                  3: track('person')}
        apply_prompts(tracks, ['λευκό αυτοκίνητο'], 'nonexistent-model.pt',
                      VEH, PER)
        self.assertIn('λευκό αυτοκίνητο', tracks[1].get('prompt_matches', []))
        self.assertNotIn('λευκό αυτοκίνητο', tracks[2].get('prompt_matches', []))
        self.assertNotIn('λευκό αυτοκίνητο', tracks[3].get('prompt_matches', []))


if __name__ == '__main__':
    unittest.main()
