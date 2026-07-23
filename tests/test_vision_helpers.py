# -*- coding: utf-8 -*-
"""Unit tests for vision.py's pure/standalone helper functions.

vision.py (1800+ lines, the largest module besides plate.py) had ZERO test
coverage before the 2026-07-23 audit. This file covers the parts that are
testable without a GPU/model load: path resolution, config loading, the
video-probe fallbacks (DAV zero-metadata, 0/NaN fps — both real field
bugs), directory expansion, and the match-review coalesce algorithm (the
union-find chaining logic behind --finalize-match).

Run: venv/bin/python -m unittest discover tests
"""
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from vision import (  # noqa: E402
    coalesce_member_groups, get_video_fps, load_config, resolve_model_path,
    _videos_in_dir,
)


class ResolveModelPathTest(unittest.TestCase):
    def test_absolute_existing_path_is_returned_as_is(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / 'yolo26l.pt'
            model.write_bytes(b'fake')
            self.assertEqual(resolve_model_path(str(model)), str(model))

    def test_finds_model_in_data_dir_models_subfolder(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'models').mkdir()
            model = Path(tmp) / 'models' / 'yolo26l.pt'
            model.write_bytes(b'fake')
            found = resolve_model_path('yolo26l.pt', data_dir=tmp)
            self.assertEqual(found, str(model))

    def test_prefers_data_dir_over_resource_dir(self):
        with tempfile.TemporaryDirectory() as data, \
             tempfile.TemporaryDirectory() as res:
            for base in (data, res):
                (Path(base) / 'models').mkdir()
            data_model = Path(data) / 'models' / 'm.pt'
            data_model.write_bytes(b'fake')
            (Path(res) / 'models' / 'm.pt').write_bytes(b'fake')
            found = resolve_model_path('m.pt', data_dir=data, resource_dir=res)
            self.assertEqual(found, str(data_model))

    def test_falls_back_to_a_data_dir_download_target(self):
        # Nothing exists anywhere: must return a data-dir path (so
        # ultralytics downloads THERE) instead of polluting the CWD.
        with tempfile.TemporaryDirectory() as tmp:
            found = resolve_model_path('nope.pt', data_dir=tmp)
            self.assertEqual(found, str(Path(tmp) / 'models' / 'nope.pt'))
            self.assertTrue((Path(tmp) / 'models').is_dir(),
                            'the models/ directory must be created eagerly')

    def test_bare_name_with_no_data_dir_passes_through(self):
        # A name that cannot possibly exist in CWD/script-dir (unlike real
        # model filenames, which the dev repo keeps cached at the root).
        name = 'definitely-not-a-real-checkpoint-xyz.pt'
        self.assertEqual(resolve_model_path(name), name)


class LoadConfigTest(unittest.TestCase):
    def test_missing_file_returns_defaults(self):
        cfg = load_config('/nonexistent/path/config.yaml')
        self.assertIn('confidence', cfg)
        self.assertIn('model_closed', cfg)

    def test_user_yaml_overrides_defaults(self):
        with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as f:
            f.write('confidence: 0.5\n')
            path = f.name
        try:
            cfg = load_config(path)
            self.assertEqual(cfg['confidence'], 0.5)
            self.assertIn('model_closed', cfg, 'unspecified keys must keep their default')
        finally:
            Path(path).unlink()

    def test_empty_yaml_does_not_crash(self):
        with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as f:
            path = f.name  # empty file -> yaml.safe_load returns None
        try:
            cfg = load_config(path)
            self.assertIn('confidence', cfg)
        finally:
            Path(path).unlink()


class VideosInDirTest(unittest.TestCase):
    def test_finds_only_video_extensions_case_insensitive(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / 'clip.mp4').write_bytes(b'')
            (d / 'CLIP2.MP4').write_bytes(b'')
            (d / 'notes.txt').write_bytes(b'')
            (d / 'clip.dav').write_bytes(b'')
            found = {Path(p).name for p in _videos_in_dir(d)}
            self.assertEqual(found, {'clip.mp4', 'CLIP2.MP4', 'clip.dav'})

    def test_empty_directory_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(_videos_in_dir(Path(tmp)), [])


class GetVideoFpsTest(unittest.TestCase):
    """DAV/VFR field-hardening: metadata lies or is absent on DVR exports —
    get_video_fps must never divide by zero or report 0 frames for a file
    that actually decodes."""

    def _write_clip(self, path, n_frames=10, fps=25, size=(64, 48)):
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'),
                             fps, size)
        for _ in range(n_frames):
            vw.write(np.zeros((size[1], size[0], 3), np.uint8))
        vw.release()

    def test_reads_frame_count_and_fps_from_a_normal_clip(self):
        with tempfile.TemporaryDirectory() as tmp:
            clip = Path(tmp) / 'clip.mp4'
            self._write_clip(clip, n_frames=20, fps=25)
            total, fps = get_video_fps(str(clip))
            self.assertEqual(total, 20)
            self.assertGreater(fps, 0)

    def test_unreadable_path_returns_zero_frames_not_a_crash(self):
        total, fps = get_video_fps('/nonexistent/file/does/not/exist.mp4')
        self.assertEqual(total, 0)
        self.assertEqual(fps, 25.0, 'must fall back to a safe default, never 0/NaN')


class CoalesceMemberGroupsTest(unittest.TestCase):
    """The union-find chaining behind --finalize-match: accepting A-B and
    B-C independently must not leave B split across two output objects."""

    def test_disjoint_pairs_stay_separate(self):
        groups = [[('v', 1), ('v', 2)], [('v', 3), ('v', 4)]]
        out = coalesce_member_groups(groups)
        self.assertEqual(len(out), 2)

    def test_chained_pairs_merge_into_one_group(self):
        groups = [[('v', 1), ('v', 2)], [('v', 2), ('v', 3)]]
        out = coalesce_member_groups(groups)
        self.assertEqual(out, [[('v', 1), ('v', 2), ('v', 3)]])

    def test_a_bridging_pair_merges_two_existing_clusters(self):
        # [A,B] and [C,D] start as separate clusters; [B,C] must fuse them
        # into ONE group of all four — the "rest" bridging branch.
        groups = [[('v', 1), ('v', 2)], [('v', 3), ('v', 4)],
                  [('v', 2), ('v', 3)]]
        out = coalesce_member_groups(groups)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0], [('v', 1), ('v', 2), ('v', 3), ('v', 4)])

    def test_single_group_passes_through_unchanged(self):
        groups = [[('a.mp4', 1), ('b.mp4', 5)]]
        out = coalesce_member_groups(groups)
        self.assertEqual(out, [[('a.mp4', 1), ('b.mp4', 5)]])

    def test_no_groups_returns_empty(self):
        self.assertEqual(coalesce_member_groups([]), [])


if __name__ == '__main__':
    unittest.main()
