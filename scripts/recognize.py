"""Live webcam recognition.

Usage (from the root project folder):
    python -m scripts.recognize
    python -m scripts.recognize --camera 0 --threshold 0.4
    python -m scripts.recognize --external-camera       # logical camera 1; auto stream/rotation
    python -m scripts.recognize --camera 2 --rotate 90    # equivalent
    python -m scripts.recognize --lock-face               # track one face
    python -m scripts.recognize --skip 2 --det-size 480   # faster on CPU

Every frame shows each detected face with a box, five face-part squares,
motion status, and a label like:
    Jordan 0.83 | Smile 0.84
    Unknown 0.34 | Angry 0.71

Window keys:
    s        capture the current frame as a JPEG in your Downloads folder
    r        start / stop recording an MP4 video in your Downloads folder
    l        lock / unlock the largest visible face
    q / ESC  quit

Performance design
------------------
Recognition (SCRFD + ArcFace, ~0.4 s on a laptop CPU) runs on a SEPARATE
thread, so it can never stall the video: the main loop only reads the camera
and displays frames, which runs at the camera's own maximum rate (≈30 fps for
a normal USB webcam — its hardware limit, regardless of software). On every
--skip-th frame the worker grabs the newest frame and refreshes the labels.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import sys
import threading
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import (                            # noqa: E402
    DETECTOR_CONFIDENCE,
    DETECTOR_INPUT_SIZE,
    DETECTOR_MODEL_PATH,
    DETECTOR_NMS,
    DOWNLOADS_DIR,
    EXPRESSION_CONFIDENCE,
    EXPRESSION_ANGER_THRESHOLD,
    EXPRESSION_INPUT_SIZE,
    EXPRESSION_LAUGH_FRAMES,
    EXPRESSION_MODEL_PATH,
    EXPRESSION_MOUTH_OPEN_THRESHOLD,
    MATCHING_THRESHOLD,
)
from app.detector import SCRFDDetector               # noqa: E402
from app.enrollment import EnrollmentError           # noqa: E402
from app.expression import (                         # noqa: E402
    ExpressionClassifier,
    ExpressionStabilizer,
)
from app.recognition import RecognitionPipeline      # noqa: E402
from app.tracking import (                           # noqa: E402
    FaceLock,
    LandmarkMotionTracker,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recognize faces live from a webcam",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--camera", type=int, default=0,
                        help="logical camera: 0 = PC camera, 1 = external camera")
    parser.add_argument(
        "--external-camera", action="store_true",
        help="prefer the external camera, auto-detect its stream/orientation, and fall back to a PC camera when absent",
    )
    parser.add_argument("--threshold", type=float, default=MATCHING_THRESHOLD,
                        help="cosine-similarity threshold for Known/Unknown")
    parser.add_argument(
        "--no-expressions", action="store_true",
        help="skip smile/laugh/angry classification",
    )
    parser.add_argument(
        "--expressions-only", action="store_true",
        help="run face expressions without requiring enrolled identities",
    )
    parser.add_argument(
        "--expression-model", default=EXPRESSION_MODEL_PATH,
        help="path to the local FER+ ONNX expression model",
    )
    parser.add_argument(
        "--expression-threshold", type=float, default=EXPRESSION_CONFIDENCE,
        help="minimum FER+ confidence before an expression is reported",
    )
    parser.add_argument(
        "--anger-threshold", type=float, default=EXPRESSION_ANGER_THRESHOLD,
        help="minimum top-class FER+ anger confidence (default: 0.25)",
    )
    parser.add_argument(
        "--mouth-open-threshold", type=float,
        default=EXPRESSION_MOUTH_OPEN_THRESHOLD,
        help="visual mouth-open score used to promote smile to laugh",
    )
    parser.add_argument(
        "--laugh-frames", type=int, default=EXPRESSION_LAUGH_FRAMES,
        help="happy worker updates required for a visual laugh label",
    )
    parser.add_argument("--skip", type=int, default=3,
                        help="run detection+embedding once every N frames "
                             "(1 = every frame; larger = faster)")
    parser.add_argument("--det-size", type=int, default=-1,
                        help="square size fed to the SCRFD detector "
                             f"(default {DETECTOR_INPUT_SIZE[0]}; smaller is "
                             "much faster, e.g. 480 or 416)")
    parser.add_argument("--res", default="640x480",
                        help="camera resolution WxH (e.g. 1280x720)")
    parser.add_argument(
        "--preview-scale", type=float, default=None,
        help="display-only window scale (external camera default: 0.75)",
    )
    parser.add_argument(
        "--rotate", type=int, choices=(0, 90, 180, 270), default=0,
        help="override the auto-detected camera rotation (0, 90, 180, or 270)",
    )
    parser.add_argument(
        "--no-landmarks", action="store_true",
        help="hide the five face-part markers and motion trails",
    )
    parser.add_argument(
        "--motion-threshold", type=float, default=0.035,
        help="normalized landmark movement required for MOVING status",
    )
    parser.add_argument(
        "--lock-face", action="store_true",
        help="lock the overlay to the largest visible face and track it",
    )
    parser.add_argument(
        "--target-name", default=None,
        help="lock only this enrolled identity, e.g. 'Kabanda Jordan'",
    )
    return parser


def _frame_is_usable(frame: np.ndarray, min_brightness: float = 25.0) -> bool:
    """A camera is 'usable' if it actually delivers a non-black frame.

    Laptops often report a second dark/black "camera" device (covered lenses,
    virtual devices, ...). On Windows an external USB webcam is frequently
    index 1 rather than 0, so we look for the first device that returns real
    content instead of trusting the index blindly.
    """
    if frame is None:
        return False
    return float(np.mean(frame)) >= min_brightness


def _rotate_frame(frame: np.ndarray, degrees: int) -> np.ndarray:
    """Rotate a camera frame clockwise by 0/90/180/270 degrees."""
    if degrees == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if degrees == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def _camera_candidates(preferred: int, max_tries: int = 4):
    """Return the preferred camera first, then the other local indices.

    Camera indices are not stable when a USB device is unplugged. Wrapping the
    fallback list is what lets ``--external-camera`` prefer index 2 but still
    open the built-in PC camera at index 0 when index 2 disappears.
    """
    indices = [preferred]
    indices.extend(index for index in range(max_tries) if index != preferred)
    return indices


def _camera_sources(
    preferred: int,
    max_tries: int = 4,
    external: bool = False,
    device_paths: bool = False,
):
    """Return ``(logical_index, open_source)`` pairs for camera probing.

    On Linux, logical camera ``0`` is the built-in PC camera and logical camera
    ``1`` is the external camera (normally physical ``/dev/video2``). PC mode
    never silently selects an external USB stream.
    """
    if not device_paths:
        if external and preferred == 1:
            indices = [1, 0, 2, 3]
        elif external and preferred == 2:
            indices = [2, 0, 1, 3]
        elif not external and preferred == 0:
            indices = [0, 1]
        else:
            indices = _camera_candidates(preferred, max_tries)
        return [(index, index) for index in indices]

    available = []
    for name in os.listdir("/dev"):
        if not name.startswith("video") or not name[5:].isdigit():
            continue
        index = int(name[5:])
        if 0 <= index < 32:
            available.append(index)
    available.sort()
    if not available:
        return []

    if external and preferred == 1:
        order = [2, 0, 1, 3]
    elif external and preferred == 2:
        order = [2, 3, 0, 1]
    elif not external and preferred == 0:
        # Keep camera 0 strictly on the PC camera nodes.
        order = [0, 1]
    else:
        order = [preferred] + [index for index in available if index != preferred]
    ordered = []
    for index in order:
        if index in available and index not in ordered:
            ordered.append(index)
    if not external and preferred == 0:
        ordered = [index for index in ordered if index in (0, 1)]
    ordered.extend(index for index in available if index not in ordered)
    if not external and preferred == 0:
        ordered = [index for index in ordered if index in (0, 1)]

    sources = []
    for index in ordered:
        if external and index in (2, 3):
            logical_index = preferred
        elif external and index == 1:
            logical_index = preferred
        else:
            logical_index = index
        sources.append((logical_index, f"/dev/video{index}"))
    return sources


def _open_camera_capture(source, res: tuple):
    """Open and warm one camera index, returning its capture and last frame."""
    if isinstance(source, str):
        cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return None, None
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except AttributeError:
        pass
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
    cap.set(cv2.CAP_PROP_FPS, 60)
    frame = None
    for _ in range(8):
        ok, candidate = cap.read()
        if ok:
            frame = candidate
    if frame is None or not _frame_is_usable(frame):
        cap.release()
        return None, None
    return cap, frame


def _camera_frame_score(frame, detector):
    """Score a frame by the best face count/confidence across rotations."""
    best = (0, 0.0, 0)
    # External webcams in portrait mode commonly need 90 or 270 degrees;
    # checking all four also handles a PC camera mounted sideways.
    for rotation in (0, 90, 180, 270):
        try:
            detections = detector.detect(_rotate_frame(frame, rotation))
        except Exception:
            detections = []
        if detections:
            confidence = max(float(item.confidence) for item in detections)
            score = (len(detections), confidence, -rotation)
            if score > best:
                best = (score[0], score[1], rotation)
    return best


def open_usable_camera(
    preferred: int,
    max_tries: int = 4,
    res: tuple = (640, 480),
    face_detector=None,
    auto_rotate: bool = False,
    device_paths: bool = False,
):
    """Open the best usable camera and return ``(capture, index, rotation)``.

    With ``face_detector`` supplied, candidate streams are briefly probed and
    the stream/orientation producing the strongest face detection wins. This
    handles UVC cameras that expose multiple video nodes and portrait mounts.
    """
    candidates = _camera_sources(
        preferred,
        max_tries,
        external=auto_rotate,
        device_paths=device_paths,
    )

    if face_detector is not None and auto_rotate:
        external_best = None
        pc_best = None
        for reported_index, source in candidates:
            cap, frame = _open_camera_capture(source, res)
            if cap is None:
                continue
            score = _camera_frame_score(frame, face_detector)
            candidate = (score, reported_index, source)
            if auto_rotate and reported_index != 0:
                if external_best is None or score[:2] > external_best[0][:2]:
                    external_best = candidate
            elif pc_best is None or score[:2] > pc_best[0][:2]:
                pc_best = candidate
            cap.release()
        # Prefer any usable external stream over the PC camera, even when the
        # external view currently contains no detectable face. The centering
        # guide will explain that physical framing issue.
        best = external_best or pc_best
        if best is None:
            return None, None, 0
        score, selected_index, selected_source = best
        cap, _ = _open_camera_capture(selected_source, res)
        if cap is None:
            return None, None, 0
        if score[0] == 0 and selected_index != 0:
            # No face was visible during probing; retain the known external
            # camera's portrait-to-landscape correction as a safe default.
            selected_rotation = 90
        else:
            selected_rotation = score[2]
        return cap, selected_index, selected_rotation

    for reported_index, source in candidates:
        cap, _ = _open_camera_capture(source, res)
        if cap is not None:
            return cap, reported_index, 0
    return None, None, 0


def _timestamp_stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


_LANDMARK_LABELS = ("LE", "RE", "N", "LM", "RM")
_LANDMARK_COLORS = (
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
)


def _draw(
    annotated,
    result,
    locked: bool = False,
    motion=None,
    show_landmarks: bool = True,
) -> None:
    """Draw identity, expression, face-part markers, and motion trails."""
    x1, y1, x2, y2 = (int(v) for v in result.bbox)
    known = result.match.is_known
    color = (0, 200, 0) if known else (0, 0, 255)  # green / red
    thickness = 3 if locked else 2
    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

    # Scale the five part markers with the face size.  The old fixed 8 px
    # squares were easy to miss on a 640x480 preview, especially when the
    # camera was held farther away.
    marker_half = int(np.clip(round(max(1, x2 - x1) * 0.035), 5, 14))

    identity = getattr(result.match, "identity", None)
    identity_label = result.match.display_label
    if locked and identity is None:
        # Expression-only mode has no enrolled identity to display. Do not
        # paint a misleading "LOCK Unknown" label over the face box.
        identity_label = ""
    elif locked:
        identity_label = f"LOCK {identity_label}"
    expression = getattr(result, "expression", None)
    expression_label = expression.display_label if expression is not None else ""
    label = identity_label
    if expression_label:
        label += f" | {expression_label}"
    if not label:
        label = "Scanning"
    (text_w, text_h), _baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
    )
    top = max(0, y1 - text_h - 8)
    cv2.rectangle(annotated, (x1, top), (x1 + text_w + 8, top + text_h + 8), color, -1)
    cv2.putText(
        annotated, label, (x1 + 4, top + text_h + 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA,
    )

    points = np.asarray(getattr(result, "landmarks", []), dtype=np.float32)
    if show_landmarks and points.shape == (5, 2) and np.all(np.isfinite(points)):
        if motion is not None and getattr(motion, "trail", None) is not None:
            trail = np.asarray(motion.trail, dtype=np.float32)
            if trail.ndim == 3 and trail.shape[1:] == (5, 2):
                for part, part_color in enumerate(_LANDMARK_COLORS):
                    if len(trail) >= 2:
                        cv2.polylines(
                            annotated,
                            [trail[:, part, :].astype(np.int32)],
                            False,
                            part_color,
                            1,
                            cv2.LINE_AA,
                        )

        velocities = (
            np.asarray(getattr(motion, "velocities", np.zeros((5, 2))), dtype=np.float32)
            if motion is not None
            else np.zeros((5, 2), dtype=np.float32)
        )
        for part, ((px, py), part_color) in enumerate(zip(points, _LANDMARK_COLORS)):
            ix, iy = int(px), int(py)
            # Filled square + bright outline makes the five face parts easy
            # to see even on a small preview window.
            cv2.rectangle(
                annotated,
                (ix - marker_half, iy - marker_half),
                (ix + marker_half, iy + marker_half),
                (0, 0, 0),
                -1,
            )
            cv2.rectangle(
                annotated,
                (ix - marker_half, iy - marker_half),
                (ix + marker_half, iy + marker_half),
                part_color,
                2,
            )
            cv2.putText(
                annotated,
                _LANDMARK_LABELS[part],
                (ix + marker_half + 2, iy - marker_half - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                part_color,
                1,
                cv2.LINE_AA,
            )
            if motion is not None:
                vx, vy = velocities[part]
                end = (int(px + vx), int(py + vy))
                cv2.line(annotated, (ix, iy), end, part_color, 1, cv2.LINE_AA)

    if motion is not None:
        motion_text = (
            f"MOVING {motion.score:.2f}"
            if motion.moving
            else f"STILL {motion.score:.2f}"
        )
        motion_color = (0, 220, 255) if motion.moving else (220, 220, 220)
        cv2.putText(
            annotated,
            motion_text,
            (x1, min(annotated.shape[0] - 5, y2 + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            motion_color,
            1,
            cv2.LINE_AA,
        )


def _draw_status_bar(
    annotated,
    display_fps: float,
    recording: bool,
    record_start,
    lock_status: str = "off",
    target_name: str = None,
):
    """Top-left strip: FPS + lock/REC indicators + key hints."""
    h, w = annotated.shape[:2]
    bar = np.full((46, w, 3), 18, dtype=np.uint8)  # dark strip

    cv2.putText(bar, f"display {display_fps:.0f} fps",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 0), 1, cv2.LINE_AA)

    if target_name:
        if lock_status == "on":
            lock_text = f"locked: {target_name}"
            lock_color = (0, 220, 255)
        elif lock_status == "lost":
            lock_text = f"lost: {target_name}"
            lock_color = (0, 0, 255)
        elif lock_status == "searching":
            lock_text = f"searching: {target_name}"
            lock_color = (0, 220, 255)
        else:
            lock_text = f"target: {target_name}"
            lock_color = (180, 180, 180)
    else:
        lock_text = f"face lock: {lock_status}"
        lock_color = (0, 220, 255) if lock_status == "on" else (180, 180, 180)
    cv2.putText(bar, lock_text, (150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, lock_color, 1, cv2.LINE_AA)

    hint = "s save | r record | l lock | q quit"
    (tw, th), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(bar, hint, (max(360, w - tw - 12), 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (200, 200, 200), 1, cv2.LINE_AA)

    if recording:
        elapsed = time.monotonic() - (record_start or time.monotonic())
        minutes, seconds = int(elapsed // 60), int(elapsed % 60)
        label = f"REC {minutes:02d}:{seconds:02d}"
        center = w // 2
        cv2.circle(bar, (center - 10, 22), 8, (0, 0, 255), -1)
        cv2.putText(bar, label, (center + 2, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    annotated[0:46, :, :] = bar


def _labeled_frame(
    frame,
    results,
    locked: bool = False,
    motion_states=None,
    show_landmarks: bool = True,
) -> np.ndarray:
    annotated = frame.copy()
    for result in results:
        _draw(
            annotated,
            result,
            locked=locked,
            motion=(motion_states or {}).get(id(result)),
            show_landmarks=show_landmarks,
        )
    return annotated


def _draw_face_guide(
    annotated: np.ndarray,
    lock_status: str,
    target_name: str = None,
) -> None:
    """Show an actionable framing guide while no face is selected.

    A dark/backlit or side-facing frame cannot produce trustworthy landmarks.
    Making the failure visible is more useful than silently showing an empty
    video, and the guide disappears as soon as the detector acquires a face.
    """
    height, width = annotated.shape[:2]
    if height < 120 or width < 160:
        return

    guide_width = max(140, min(width - 40, int(width * 0.46)))
    guide_height = max(180, min(height - 110, int(height * 0.62)))
    left = max(20, (width - guide_width) // 2)
    top = max(52, (height - guide_height) // 2)
    right = min(width - 20, left + guide_width)
    bottom = min(height - 82, top + guide_height)

    guide_color = (0, 210, 255)
    cv2.rectangle(annotated, (left, top), (right, bottom), guide_color, 2)
    cv2.line(annotated, ((left + right) // 2, top), ((left + right) // 2, bottom),
             (0, 210, 255), 1, cv2.LINE_AA)
    cv2.line(annotated, (left, (top + bottom) // 2), (right, (top + bottom) // 2),
             (0, 210, 255), 1, cv2.LINE_AA)

    if target_name and lock_status == "lost":
        message = f"LOST: {target_name.upper()}"
        detail = "MOVE BACK INTO VIEW - THE TARGET WILL REACQUIRE"
    elif target_name and lock_status == "searching":
        message = f"SEARCHING: {target_name.upper()}"
        detail = "CENTER YOUR FACE AND LOOK AT THE CAMERA"
    else:
        message = "NO FACE DETECTED"
        detail = "CENTER YOUR FACE AND LOOK AT THE CAMERA"
        if lock_status == "searching":
            detail = "PRESS L TO UNLOCK - THEN CENTER YOUR FACE"
    cv2.rectangle(annotated, (0, height - 72), (width, height), (18, 18, 18), -1)
    cv2.putText(annotated, message, (16, height - 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 210, 255), 2, cv2.LINE_AA)
    cv2.putText(annotated, detail, (16, height - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)


class _SharedState:
    """Thread-safe hand-off between the camera loop and the recognition worker."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.frame = None        # newest frame from the camera (BGR)
        self.frame_id = 0        # monotonically increasing camera-frame id
        self.results: list = []  # latest recognition output (may lag the frame)

    def publish_frame(self, frame) -> None:
        with self.lock:
            self.frame = frame
            self.frame_id += 1

    def snapshot(self):
        """Copy of (frame, results), preserving the original public shape."""
        with self.lock:
            return self.frame, list(self.results)

    def snapshot_with_id(self):
        """Copy frame data plus its monotonic camera-frame id."""
        with self.lock:
            return self.frame, self.frame_id, list(self.results)

    def publish_results(self, results) -> None:
        with self.lock:
            self.results = results


def recognition_worker(
    pipeline,
    state: _SharedState,
    skip: int,
    stop: threading.Event,
    stabilizer=None,
    require_enrollment: bool = True,
) -> None:
    """Run the (expensive) pipeline on a background thread.

    The camera loop keeps streaming/displaying at the camera's true max FPS;
    this thread simply updates the labels every --skip-th *new* frame. ONNX
    sessions are thread-safe, so calling recognize_frame() here is safe.
    """
    seen = 0
    last_frame_id = -1
    while not stop.is_set():
        snapshot_with_id = getattr(state, "snapshot_with_id", None)
        if snapshot_with_id is not None:
            frame, frame_id, _ = snapshot_with_id()
        else:  # compatibility with simple test doubles / older callers
            frame, _ = state.snapshot()
            frame_id = getattr(state, "frame_id", seen)
        if frame is None or frame_id == last_frame_id:
            time.sleep(0.005)
            continue
        # Only advance the temporal filter for genuinely new camera frames.
        # Without frame IDs, a fast worker would process the same held frame
        # many times and could promote one smile to a false laugh.
        if seen % skip != 0:
            seen += 1
            last_frame_id = frame_id
            continue
        seen += 1
        last_frame_id = frame_id
        try:
            results = pipeline.recognize_frame(
                frame, require_enrollment=require_enrollment
            )
            if stabilizer is not None:
                results = stabilizer.update(results)
        except ValueError as exc:  # empty enrollment db
            print(f"ERROR: {exc}", file=sys.stderr)
            stop.set()
            return
        except Exception as exc:  # keep a model/input failure visible
            print(f"ERROR: recognition worker stopped: {exc}", file=sys.stderr)
            stop.set()
            return
        state.publish_results(results)


def main() -> int:
    args = build_parser().parse_args()
    # Camera 1 is the project's logical external-camera selector. Camera 0 is
    # always the built-in PC camera. The Linux backend maps logical camera 1
    # to the physical UVC device (usually /dev/video2).
    if args.camera == 1:
        args.external_camera = True
    if args.external_camera:
        args.camera = 1
    if args.target_name:
        args.lock_face = True
        if args.expressions_only:
            print(
                "ERROR: --target-name requires identity mode; remove "
                "--expressions-only and enroll the target first.",
                file=sys.stderr,
            )
            return 1
    if args.preview_scale is None:
        args.preview_scale = 0.75 if args.external_camera else 1.0
    if not 0.1 <= args.preview_scale <= 1.0:
        print("ERROR: --preview-scale must be between 0.1 and 1.0.", file=sys.stderr)
        return 1

    if args.no_expressions and args.expressions_only:
        print(
            "ERROR: --no-expressions and --expressions-only cannot be used together.",
            file=sys.stderr,
        )
        return 1

    if args.skip < 1:
        print("ERROR: --skip must be >= 1.", file=sys.stderr)
        return 1
    if args.motion_threshold < 0:
        print("ERROR: --motion-threshold must be >= 0.", file=sys.stderr)
        return 1

    try:
        width, height = (int(x) for x in args.res.lower().split("x"))
    except ValueError:
        print(f"ERROR: --res must look like 640x480, got {args.res!r}.",
              file=sys.stderr)
        return 1

    try:
        if args.det_size > 0:
            detector = SCRFDDetector(
                model_path=DETECTOR_MODEL_PATH,
                input_size=(args.det_size, args.det_size),
                confidence_threshold=DETECTOR_CONFIDENCE,
                nms_threshold=DETECTOR_NMS,
            )
        else:
            detector = None
    except (FileNotFoundError, ValueError) as exc:
        print(f"SETUP ERROR: {exc}", file=sys.stderr)
        return 1

    expression_classifier = None
    if not args.no_expressions:
        try:
            expression_classifier = ExpressionClassifier(
                model_path=args.expression_model,
                input_size=EXPRESSION_INPUT_SIZE,
                confidence_threshold=args.expression_threshold,
                anger_threshold=args.anger_threshold,
                mouth_open_threshold=args.mouth_open_threshold,
                laugh_frames=args.laugh_frames,
            )
        except Exception as exc:
            print(
                f"WARNING: expression detection disabled ({exc}).\n"
                "Run `python -m scripts.download_models` or use "
                "--no-expressions to hide this message.",
                file=sys.stderr,
            )

    try:
        pipeline = RecognitionPipeline(
            detector=detector,
            threshold=args.threshold,
            expression_classifier=expression_classifier,
            enable_expressions=not args.no_expressions,
            enable_identity=not args.expressions_only,
        )
    except (FileNotFoundError, EnrollmentError) as exc:
        print(f"SETUP ERROR: {exc}", file=sys.stderr)
        return 1

    if args.expressions_only and pipeline.expression_classifier is None:
        print(
            "ERROR: --expressions-only needs models/emotion-ferplus-8.onnx. "
            "Run `python -m scripts.download_models`.",
            file=sys.stderr,
        )
        return 1

    if args.target_name and not pipeline.matcher.has_enrollments():
        print(
            "ERROR: --target-name needs an enrollment database. Put photos in "
            "data/faces/<person>/ and run `python -m scripts.enroll` first.",
            file=sys.stderr,
        )
        return 1

    stabilizer = None
    if pipeline.expression_classifier is not None:
        stabilizer = ExpressionStabilizer(
            laugh_frames=args.laugh_frames,
            mouth_open_threshold=args.mouth_open_threshold,
            confidence_threshold=args.expression_threshold,
        )
    else:
        print("NOTE: expression labels are disabled; identity recognition is still active.")

    cap, used_index, detected_rotation = open_usable_camera(
        args.camera,
        res=(width, height),
        face_detector=pipeline.detector if args.external_camera else None,
        auto_rotate=args.external_camera,
        device_paths=sys.platform.startswith("linux"),
    )
    if cap is None:
        candidates = _camera_candidates(args.camera)
        print(
            "ERROR: could not find a working webcam (tried indices "
            f"{', '.join(str(index) for index in candidates)}). "
            "Close other apps using the camera and try again.",
            file=sys.stderr,
        )
        return 1

    if args.external_camera and args.rotate == 0:
        # Probe the stream orientation when possible. This handles both
        # portrait external cameras and already-upright alternate UVC nodes.
        args.rotate = detected_rotation

    if used_index != args.camera and not args.external_camera:
        print(f"NOTE: using camera index {used_index} "
              f"(index {args.camera} was black or unavailable).")

    # Face lock is applied to the latest recognition results in the display
    # loop, so the expensive worker can keep processing frames independently.
    face_lock = FaceLock(target_identity=args.target_name)
    if args.lock_face:
        face_lock.enable()
    motion_tracker = LandmarkMotionTracker(
        motion_threshold=args.motion_threshold
    )

    # Background recognition: the display loop never waits for the pipeline.
    state = _SharedState()
    stop = threading.Event()
    worker = threading.Thread(
        target=recognition_worker,
        args=(
            pipeline,
            state,
            args.skip,
            stop,
            stabilizer,
            not args.expressions_only,
        ),
        daemon=True,
    )
    worker.start()

    print(f"Saved photos/videos go to: {DOWNLOADS_DIR}")
    if args.external_camera:
        if used_index in (1, 2, 3):
            print(
                f"External camera mode: device {used_index}, "
                f"rotation {args.rotate} degrees (auto-detected)"
            )
        else:
            print(
                f"External camera unavailable; using PC camera index {used_index} "
                f"(rotation {args.rotate} degrees)"
            )
    print("Controls: s = save photo | r = start/stop video | l = lock face | q/ESC = quit")
    if pipeline.expression_classifier is not None:
        print("Expressions: smile / laugh / angry are shown after each face label")
    if args.lock_face:
        if args.target_name:
            print(f"Face lock ON - targeting enrolled identity: {args.target_name}")
        else:
            print("Face lock ON - the largest visible face will be selected")
    if args.expressions_only:
        print("Identity gallery is not required in --expressions-only mode")
    fps_window = 30
    fps_times = []
    writer = None          # cv2.VideoWriter while recording
    record_start = None    # monotonic time when recording began
    rec_fps = 20.0         # fps used for the mp4 (set from real camera fps)

    try:
        while True:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok or frame is None:
                print("ERROR: lost the webcam stream. Quitting.", file=sys.stderr)
                break

            frame = _rotate_frame(frame, args.rotate)
            state.publish_frame(frame)
            _, results = state.snapshot()
            display_results = face_lock.filter(results)
            motion_states = motion_tracker.update(display_results)
            annotated = _labeled_frame(
                frame,
                display_results,
                locked=face_lock.locked,
                motion_states=motion_states,
                show_landmarks=not args.no_landmarks,
            )
            if not display_results:
                _draw_face_guide(
                    annotated,
                    face_lock.status,
                    target_name=args.target_name,
                )

            # display-rate counter so you can SEE the video is not slowed down
            fps_times.append(t0)
            if len(fps_times) > fps_window:
                fps_times.pop(0)
            if len(fps_times) > 1:
                span = fps_times[-1] - fps_times[0]
                disp = len(fps_times) / span if span > 0 else 0
            else:
                disp = 0.0
            if writer is not None:
                rec_fps = max(5.0, min(disp, 120.0)) if disp > 0 else rec_fps
                writer.write(annotated)

            _draw_status_bar(
                annotated,
                disp,
                writer is not None,
                record_start,
                lock_status=face_lock.status,
                target_name=args.target_name,
            )
            display_frame = annotated
            if abs(args.preview_scale - 1.0) > 1e-6:
                display_frame = cv2.resize(
                    annotated,
                    None,
                    fx=args.preview_scale,
                    fy=args.preview_scale,
                    interpolation=cv2.INTER_AREA,
                )
            cv2.imshow("Face Recognition + Expressions (ArcFace + FER+)", display_frame)
            key = cv2.waitKey(1) & 0xFF

            # ---- face lock toggle: 'l' ----------------------------------
            if key == ord("l"):
                enabled = face_lock.toggle(results)
                print(
                    f"Face lock {'ON' if enabled else 'OFF'}"
                    + (f" ({face_lock.status})" if enabled else "")
                )

            # ---- photo capture: 's' -------------------------------------
            elif key == ord("s"):
                name = f"fr_capture_{_timestamp_stamp()}.jpg"
                path = os.path.join(DOWNLOADS_DIR, name)
                if cv2.imwrite(path, annotated):
                    print(f"Saved photo -> {path}")
                else:
                    print(f"ERROR: could not write {path}", file=sys.stderr)

            # ---- video recording toggle: 'r' ----------------------------
            elif key == ord("r"):
                if writer is None:
                    if disp > 0:
                        rec_fps = disp
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    h, w = annotated.shape[:2]
                    rec_fps = max(5.0, min(rec_fps, 60.0))
                    writer = cv2.VideoWriter(
                        os.path.join(DOWNLOADS_DIR,
                                     f"fr_video_{_timestamp_stamp()}.mp4"),
                        fourcc, rec_fps, (w, h),
                    )
                    if not writer.isOpened():
                        print("ERROR: could not create the MP4 file.",
                              file=sys.stderr)
                        writer = None
                    else:
                        record_start = time.monotonic()
                        print("Recording started...")
                else:
                    writer.release()
                    print("Recording stopped.")
                    writer = None
                    record_start = None

            elif key in (ord("q"), 27):  # q or ESC
                break
    finally:
        stop.set()
        if writer is not None:
            writer.release()
        cap.release()
        cv2.destroyAllWindows()
        worker.join(timeout=2.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())