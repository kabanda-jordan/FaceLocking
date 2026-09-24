"""Small, dependency-light face selection/tracking for the live camera.

This is intentionally not a full face tracker.  It associates the next
recognition result with the currently locked box using IoU and center
proximity, which is enough to keep one selected face visible in the webcam
overlay while ignoring other detected faces.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import time
from typing import Deque, Dict, List, Optional, Sequence

import numpy as np


def _box(result: object) -> np.ndarray:
    values = np.asarray(getattr(result, "bbox"), dtype=np.float32).reshape(4)
    if values[2] < values[0] or values[3] < values[1]:
        raise ValueError(f"Invalid face bounding box: {values.tolist()}")
    return values


def _area(box: np.ndarray) -> float:
    return max(0.0, float(box[2] - box[0])) * max(
        0.0, float(box[3] - box[1])
    )


def _center(box: np.ndarray) -> np.ndarray:
    return np.array(
        [(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5],
        dtype=np.float32,
    )


def _iou(first: np.ndarray, second: np.ndarray) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
    union = _area(first) + _area(second) - intersection
    return intersection / union if union > 0 else 0.0


class FaceLock:
    """Lock the live overlay to one face and follow it across frames.

    ``filter(results)`` returns all results while disabled.  When enabled it
    first chooses the largest visible face, then returns only the result that
    best matches the locked box.  If the target disappears, the lock enters a
    short search state instead of silently switching to a different person.
    The caller can use ``toggle`` to turn locking on/off interactively.
    """

    def __init__(
        self,
        iou_threshold: float = 0.20,
        center_threshold: float = 0.45,
        max_misses: int = 12,
    ) -> None:
        self.iou_threshold = float(np.clip(iou_threshold, 0.0, 1.0))
        self.center_threshold = float(np.clip(center_threshold, 0.0, 2.0))
        self.max_misses = max(1, int(max_misses))
        self.enabled = False
        self.locked = False
        self.bbox: Optional[np.ndarray] = None
        self.misses = 0

    @property
    def status(self) -> str:
        if not self.enabled:
            return "off"
        return "on" if self.locked else "searching"

    def reset(self) -> None:
        self.locked = False
        self.bbox = None
        self.misses = 0

    def disable(self) -> None:
        self.enabled = False
        self.reset()

    def enable(self, results: Sequence[object] = ()) -> bool:
        self.enabled = True
        self.reset()
        if results:
            self._acquire(results)
        return self.enabled

    def toggle(self, results: Sequence[object] = ()) -> bool:
        if self.enabled:
            self.disable()
        else:
            self.enable(results)
        return self.enabled

    def _acquire(self, results: Sequence[object]) -> Optional[object]:
        candidates = [result for result in results if _area(_box(result)) > 0.0]
        if not candidates:
            return None
        # The largest box is a sensible default when several people are in
        # view.  The user can turn the lock off and on after positioning the
        # desired person closest to the camera.
        selected = max(candidates, key=lambda result: _area(_box(result)))
        self.bbox = _box(selected).copy()
        self.locked = True
        self.misses = 0
        return selected

    def _match(self, results: Sequence[object]) -> Optional[object]:
        if self.bbox is None:
            return None
        target = self.bbox
        target_center = _center(target)
        target_diagonal = max(
            1.0,
            float(np.linalg.norm(np.array([target[2] - target[0], target[3] - target[1]]))),
        )
        best_result = None
        best_score = -1.0
        for result in results:
            candidate = _box(result)
            overlap = _iou(target, candidate)
            distance = float(np.linalg.norm(_center(candidate) - target_center))
            normalized_distance = distance / target_diagonal
            if overlap < self.iou_threshold and normalized_distance > self.center_threshold:
                continue
            # IoU is the primary signal; center proximity only breaks ties or
            # keeps tracking when a moving face no longer overlaps much.
            score = overlap + max(0.0, 0.35 - normalized_distance)
            if score > best_score:
                best_score = score
                best_result = result
        return best_result

    def filter(self, results: Sequence[object]) -> List[object]:
        """Return the results allowed by the current lock state."""
        if not self.enabled:
            return list(results)
        if not results:
            if self.locked:
                self.misses += 1
                if self.misses >= self.max_misses:
                    self.reset()
            return []

        if not self.locked:
            selected = self._acquire(results)
            return [selected] if selected is not None else []

        matched = self._match(results)
        if matched is None:
            self.misses += 1
            if self.misses >= self.max_misses:
                self.reset()
            return []

        self.misses = 0
        candidate = _box(matched)
        # A little smoothing prevents the lock indicator from jittering when
        # the detector's box changes by a pixel or two.
        self.bbox = (0.75 * self.bbox + 0.25 * candidate).astype(np.float32)
        return [matched]


@dataclass
class MotionState:
    """Motion information for one face's five detector landmarks."""

    score: float
    moving: bool
    velocities: np.ndarray                 # (5, 2), pixels since last update
    trail: np.ndarray = field(default_factory=lambda: np.empty((0, 5, 2), dtype=np.float32))


@dataclass
class _MotionTrack:
    bbox: np.ndarray
    previous_landmarks: Optional[np.ndarray] = None
    trail: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=8))
    last_seen: float = 0.0


class LandmarkMotionTracker:
    """Track landmark movement without another face-tracking model.

    SCRFD already returns five stable face points (eyes, nose, and mouth
    corners).  We compare their positions between recognition updates, scale
    the displacement by face size, and expose a short trail for drawing.  A
    high score means the face/parts moved between updates; it does not try to
    infer emotion by itself.
    """

    def __init__(
        self,
        motion_threshold: float = 0.035,
        max_age_seconds: float = 2.0,
        trail_length: int = 8,
        iou_threshold: float = 0.20,
    ) -> None:
        self.motion_threshold = max(0.0, float(motion_threshold))
        self.max_age_seconds = max(0.1, float(max_age_seconds))
        self.trail_length = max(2, int(trail_length))
        self.iou_threshold = float(np.clip(iou_threshold, 0.0, 1.0))
        self._tracks: List[_MotionTrack] = []

    def _match(self, bbox: np.ndarray) -> Optional[_MotionTrack]:
        best = None
        best_iou = self.iou_threshold
        for track in self._tracks:
            score = _iou(track.bbox, bbox)
            if score >= best_iou:
                best_iou = score
                best = track
        return best

    def update(
        self,
        results: Sequence[object],
        timestamp: Optional[float] = None,
    ) -> Dict[int, MotionState]:
        """Return motion states keyed by ``id(result)`` for overlay drawing."""
        now = time.monotonic() if timestamp is None else float(timestamp)
        self._tracks = [
            track
            for track in self._tracks
            if now - track.last_seen <= self.max_age_seconds
        ]
        used = set()
        states: Dict[int, MotionState] = {}
        for result in results:
            bbox = _box(result)
            track = None
            best_iou = self.iou_threshold
            for index, candidate in enumerate(self._tracks):
                if index in used:
                    continue
                score = _iou(candidate.bbox, bbox)
                if score >= best_iou:
                    best_iou = score
                    track = candidate
                    track_index = index
            if track is None:
                track = _MotionTrack(
                    bbox=bbox.copy(),
                    trail=deque(maxlen=self.trail_length),
                    last_seen=now,
                )
                self._tracks.append(track)
                track_index = len(self._tracks) - 1
            used.add(track_index)

            points = np.asarray(
                getattr(result, "landmarks", None), dtype=np.float32
            )
            if points.shape != (5, 2) or not np.all(np.isfinite(points)):
                track.bbox = bbox.copy()
                track.last_seen = now
                states[id(result)] = MotionState(
                    score=0.0,
                    moving=False,
                    velocities=np.zeros((5, 2), dtype=np.float32),
                    trail=np.asarray(track.trail, dtype=np.float32),
                )
                continue

            if track.previous_landmarks is None:
                velocities = np.zeros((5, 2), dtype=np.float32)
                score = 0.0
            else:
                velocities = points - track.previous_landmarks
                eye_distance = float(
                    np.linalg.norm(points[0] - points[1])
                )
                face_scale = max(eye_distance, float(np.linalg.norm(bbox[2:] - bbox[:2])) * 0.15, 1.0)
                normalized = np.linalg.norm(velocities, axis=1) / face_scale
                score = float(np.mean(normalized))

            track.previous_landmarks = points.copy()
            track.trail.append(points.copy())
            track.bbox = bbox.copy()
            track.last_seen = now
            states[id(result)] = MotionState(
                score=score,
                moving=score >= self.motion_threshold,
                velocities=velocities.astype(np.float32),
                trail=np.asarray(track.trail, dtype=np.float32),
            )
        return states


__all__ = ["FaceLock", "LandmarkMotionTracker", "MotionState"]
