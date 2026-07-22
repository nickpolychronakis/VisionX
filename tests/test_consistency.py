# -*- coding: utf-8 -*-
"""Cross-language single-source-of-truth guards (DRY audit 2026-07-23).

Some lists MUST exist in more than one language (Python, Rust, Vue) and
cannot share a literal — history shows they silently diverge (the video
extension lists had already drifted apart when the audit ran). These tests
parse each surface FROM SOURCE (no heavy imports, so the suite stays fast)
and fail loudly on any divergence, turning "remember to update both" into
"the CI tells you".

Run: venv/bin/python -m unittest discover tests
"""
import ast
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).parent.parent


def _python_extensions() -> set[str]:
    """VIDEO_EXTENSIONS from vision.py — the declared source of truth."""
    src = (ROOT / 'vision.py').read_text(encoding='utf-8')
    m = re.search(r'^VIDEO_EXTENSIONS = (\[.*?\])', src, re.S | re.M)
    assert m, 'VIDEO_EXTENSIONS not found in vision.py'
    return set(ast.literal_eval(m.group(1)))


def _rust_extensions() -> set[str]:
    src = (ROOT / 'visionx-app/src-tauri/src/main.rs').read_text(encoding='utf-8')
    m = re.search(r'const VIDEO_EXTS: \[&str; \d+\] = \[(.*?)\];', src, re.S)
    assert m, 'VIDEO_EXTS not found in main.rs'
    return set(re.findall(r'"([a-z0-9]+)"', m.group(1)))


def _vue_extensions() -> set[str]:
    src = (ROOT / 'visionx-app/src/components/FileSelector.vue').read_text(encoding='utf-8')
    m = re.search(r'const videoExtensions = (\[.*?\]);', src, re.S)
    assert m, 'videoExtensions not found in FileSelector.vue'
    return set(re.findall(r"'([a-z0-9]+)'", m.group(1)))


class VideoExtensionSyncTest(unittest.TestCase):
    """The three surfaces that filter video files must accept the SAME set."""

    def test_rust_matches_python(self):
        self.assertEqual(_python_extensions(), _rust_extensions(),
                         'main.rs VIDEO_EXTS diverged from vision.py '
                         'VIDEO_EXTENSIONS — update the mirror')

    def test_vue_matches_python(self):
        self.assertEqual(_python_extensions(), _vue_extensions(),
                         'FileSelector.vue videoExtensions diverged from '
                         'vision.py VIDEO_EXTENSIONS — update the mirror')


class ScriptManifestTest(unittest.TestCase):
    """build.rs SCRIPTS is the ONE script manifest (DRY 6/8): every listed
    script must exist in the repo root, and every root script that other
    modules import must be listed — a missing entry ships a broken bundle
    (field case: CI once shipped 2 of 12 scripts)."""

    def _manifest(self) -> set[str]:
        src = (ROOT / 'visionx-app/src-tauri/build.rs').read_text(encoding='utf-8')
        m = re.search(r'const SCRIPTS: \[&str; \d+\] = \[(.*?)\];', src, re.S)
        assert m, 'SCRIPTS not found in build.rs'
        return set(re.findall(r'"([a-z_]+\.py)"', m.group(1)))

    def test_manifest_files_exist(self):
        missing = [f for f in self._manifest() if not (ROOT / f).exists()]
        self.assertFalse(missing, f'build.rs lists nonexistent scripts: {missing}')

    def test_root_scripts_are_bundled(self):
        root_scripts = {p.name for p in ROOT.glob('*.py')}
        unbundled = root_scripts - self._manifest()
        self.assertFalse(unbundled,
                         f'repo-root scripts missing from build.rs SCRIPTS '
                         f'(the app would run without them): {unbundled}')


if __name__ == '__main__':
    unittest.main()
