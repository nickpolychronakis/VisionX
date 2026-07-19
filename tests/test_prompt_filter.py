# -*- coding: utf-8 -*-
"""Tests for the structured result filters (prompt_filter.py).
Free text was removed by design — only fixed color/type choices exist.
Run: venv/bin/python -m unittest discover tests
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from prompt_filter import FILTER_COLORS, apply_filters  # noqa: E402


def track(cls='car', color=None, clothing=None):
    t = {'class': cls}
    if color:
        t['attrs'] = {'color': color, 'color_conf': 0.8}
    if clothing:
        t['attrs'] = {'clothing': {k: {'color': v, 'conf': 0.7}
                                   for k, v in clothing.items()}}
    return t


class FilterTest(unittest.TestCase):
    def test_color_and_type_intersection(self):
        tracks = {1: track('car', color='λευκό'),
                  2: track('car', color='μαύρο'),
                  3: track('truck', color='λευκό'),
                  4: track('person')}
        apply_filters(tracks, colors=['λευκό'], types=['car'])
        self.assertIn('prompt_matches', tracks[1])
        self.assertNotIn('prompt_matches', tracks[2])  # wrong color
        self.assertNotIn('prompt_matches', tracks[3])  # wrong type
        self.assertNotIn('prompt_matches', tracks[4])

    def test_type_only(self):
        tracks = {1: track('truck'), 2: track('car'), 3: track('person')}
        apply_filters(tracks, colors=None, types=['φορτηγό'])  # Greek name
        self.assertIn('prompt_matches', tracks[1])
        self.assertNotIn('prompt_matches', tracks[2])

    def test_color_only_includes_clothing(self):
        tracks = {1: track('person', clothing={'upper': 'κόκκινο'}),
                  2: track('person', clothing={'upper': 'μπλε'}),
                  3: track('car', color='κόκκινο')}
        apply_filters(tracks, colors=['κόκκινο'], types=None)
        self.assertIn('prompt_matches', tracks[1])
        self.assertNotIn('prompt_matches', tracks[2])
        self.assertIn('prompt_matches', tracks[3])

    def test_no_filters_is_noop(self):
        tracks = {1: track('car', color='λευκό')}
        apply_filters(tracks, colors=[], types=[])
        self.assertNotIn('prompt_matches', tracks[1])

    def test_unknown_values_ignored(self):
        tracks = {1: track('car', color='λευκό')}
        apply_filters(tracks, colors=['πουά'], types=['ελικόπτερο'])
        self.assertNotIn('prompt_matches', tracks[1])

    def test_vocabulary_matches_attributes_module(self):
        from attributes import CSS_COLOR
        for c in FILTER_COLORS:
            self.assertIn(c, CSS_COLOR, f'filter color {c} missing CSS')


if __name__ == '__main__':
    unittest.main()
