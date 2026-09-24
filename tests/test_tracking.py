"""Tests for the lightweight one-face lock used by the webcam overlay."""

import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.tracking import FaceLock


def face(x, y, width=100, height=100):
    return SimpleNamespace(
        bbox=np.array([float(x), float(y), float(x + width), float(y + height)])
    )


def test_disabled_lock_passes_all_faces_through():
    lock = FaceLock()
    results = [face(0, 0), face(200, 0)]
    assert lock.filter(results) == results
    assert lock.status == "off"


def test_enable_selects_largest_face():
    lock = FaceLock()
    small = face(0, 0, 40, 40)
    large = face(200, 0, 120, 120)
    assert lock.enable([small, large]) is True
    assert lock.locked
    assert lock.filter([small, large]) == [large]
    assert lock.status == "on"


def test_lock_follows_a_moved_face_and_ignores_another():
    lock = FaceLock()
    target = face(100, 100, 100, 100)
    other = face(300, 100, 100, 100)
    lock.enable([target, other])
    moved = face(108, 106, 100, 100)
    assert lock.filter([moved, other]) == [moved]
    assert lock.locked


def test_lock_enters_search_state_when_target_disappears():
    lock = FaceLock(max_misses=2)
    target = face(100, 100)
    lock.enable([target])
    assert lock.filter([]) == []
    assert lock.filter([]) == []
    assert not lock.locked
    assert lock.status == "searching"
    # A later visible face can be acquired again in the live camera.
    new_face = face(400, 100)
    assert lock.filter([new_face]) == [new_face]


def test_toggle_turns_lock_off():
    lock = FaceLock()
    target = face(0, 0)
    assert lock.toggle([target]) is True
    assert lock.locked
    assert lock.toggle([target]) is False
    assert not lock.enabled
    assert not lock.locked
