# -*- coding: utf-8 -*-
"""Headless GUI harness for plate.py (DRY audit 2026-07-23).

The interactive tool draws its own OpenCV windows and reads keys/mouse —
untestable as-is. This harness monkeypatches the cv2 GUI surface with a
SCRIPTED event queue, so complete user sessions (seek, drag a box, q-q,
multi-segment prompts, b-scan) replay deterministically in CI.

The technique was rebuilt in throwaway /tmp scripts at least three times
during field debugging and lost each time; this is its permanent home.

Usage:
    from harness_gui import run_plate_session, drag
    result = run_plate_session(video, [SPACE, drag(10, 10, 120, 60), *IDLE(30), Q, Q])
"""
import contextlib
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2  # noqa: E402

# Readable event aliases for scripts
Q = ord('q')
N = ord('n')
B = ord('b')
ENTER = 13
SPACE = ord(' ')
IDLE = lambda n: [-1] * n  # noqa: E731 — reads better inline in scripts


def drag(x1: int, y1: int, x2: int, y2: int) -> tuple:
    """A scripted mouse drag (frame coordinates)."""
    return ('drag', x1, y1, x2, y2)


@contextlib.contextmanager
def _patched_gui(script: list):
    """Replace cv2's GUI calls with a scripted queue for the duration."""
    state = {'i': 0, 'mouse_cb': None}

    def fake_wait_key_ex(_delay: int = 0) -> int:
        # Past the script's end keep answering q so a stuck loop still
        # terminates instead of hanging the test run.
        if state['i'] >= len(script):
            return Q
        item = script[state['i']]
        state['i'] += 1
        if isinstance(item, tuple) and item[0] == 'drag':
            _, x1, y1, x2, y2 = item
            cb = state['mouse_cb']
            assert cb is not None, 'drag scripted before setMouseCallback'
            cb(cv2.EVENT_LBUTTONDOWN, x1, y1, 0, None)
            cb(cv2.EVENT_MOUSEMOVE, (x1 + x2) // 2, (y1 + y2) // 2, 0, None)
            cb(cv2.EVENT_LBUTTONUP, x2, y2, 0, None)
            return -1
        return item

    saved = {name: getattr(cv2, name, None)
             for name in ('namedWindow', 'resizeWindow', 'imshow',
                          'destroyAllWindows', 'waitKeyEx', 'waitKey',
                          'setMouseCallback')}
    cv2.namedWindow = lambda *a, **k: None
    cv2.resizeWindow = lambda *a, **k: None
    cv2.imshow = lambda *a, **k: None
    cv2.destroyAllWindows = lambda *a, **k: None
    cv2.waitKeyEx = fake_wait_key_ex
    cv2.waitKey = lambda d=0: fake_wait_key_ex(d) & 0xFF
    cv2.setMouseCallback = (
        lambda _win, cb, *a: state.__setitem__('mouse_cb', cb))
    try:
        yield
    finally:
        for name, fn in saved.items():
            if fn is not None:
                setattr(cv2, name, fn)


def run_plate_session(video: str, script: list, output_dir: str,
                      extra_args: list | None = None) -> None:
    """Run a full plate.py session against `script` (keys + drags).

    plate.py is (re)imported fresh each call because it keeps module-level
    lazy singletons (WPOD/SR caches) that must not leak between tests.
    """
    argv = ['plate.py', video, '--output', output_dir,
            '--max-ocr-frames', '10', *(extra_args or [])]
    saved_argv = sys.argv
    sys.argv = argv
    try:
        with _patched_gui(script):
            sys.modules.pop('plate', None)
            plate = importlib.import_module('plate')
            plate.main()
    finally:
        sys.argv = saved_argv
        sys.modules.pop('plate', None)
