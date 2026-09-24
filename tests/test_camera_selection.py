"""Tests for camera preference and USB fallback behavior."""

import numpy as np

from scripts import recognize


def test_external_preference_wraps_to_pc_indices():
    assert recognize._camera_candidates(2, max_tries=4) == [2, 0, 1, 3]
    assert recognize._camera_candidates(0, max_tries=4) == [0, 1, 2, 3]


def test_open_usable_camera_falls_back_when_preferred_is_missing(monkeypatch):
    opened = []

    class FakeCapture:
        def __init__(self, index):
            self.index = index
            self.opened = index in (0, 1)
            self.released = False
            opened.append(index)

        def isOpened(self):
            return self.opened

        def set(self, *_args):
            return True

        def read(self):
            if not self.opened:
                return False, None
            return True, np.full((8, 8, 3), 80, dtype=np.uint8)

        def release(self):
            self.released = True

    monkeypatch.setattr(recognize.cv2, "VideoCapture", FakeCapture)
    capture, index = recognize.open_usable_camera(2, max_tries=4)

    assert index == 0
    assert capture is not None
    assert opened[:2] == [2, 0]
