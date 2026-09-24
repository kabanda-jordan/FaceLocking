"""Tests for landmark motion tracking used by the live overlay."""

import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.tracking import LandmarkMotionTracker


def result(x, y=0):
    landmarks = np.array(
        [
            [x + 20, y + 20],
            [x + 60, y + 20],
            [x + 40, y + 45],
            [x + 25, y + 70],
            [x + 55, y + 70],
        ],
        dtype=np.float32,
    )
    return SimpleNamespace(
        bbox=np.array([x, y, x + 80, y + 90], dtype=np.float32),
        landmarks=landmarks,
    )


def test_motion_tracker_reports_still_then_moving():
    tracker = LandmarkMotionTracker(motion_threshold=0.05)
    first = result(100)
    states = tracker.update([first], timestamp=0.0)
    assert not states[id(first)].moving

    still = result(100)
    states = tracker.update([still], timestamp=0.1)
    assert states[id(still)].score < 0.05

    moved = result(115)
    states = tracker.update([moved], timestamp=0.2)
    assert states[id(moved)].moving
    assert states[id(moved)].trail.shape[0] == 3


def test_motion_tracker_handles_missing_landmarks():
    tracker = LandmarkMotionTracker()
    item = SimpleNamespace(bbox=np.array([0, 0, 10, 10], dtype=np.float32))
    state = tracker.update([item], timestamp=0.0)[id(item)]
    assert not state.moving
    assert state.score == 0.0
