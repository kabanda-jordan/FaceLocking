"""Visualize every stage of the face-recognition pipeline in pop-up windows.

This is the "show me what the algorithm does" tool. Run it from your own
terminal (it opens GUI windows; it will not work from a headless shell):

    python -m scripts.visualize --image path/to/photo.jpg

Windows appear one at a time, each explaining the stage:
    1. Detection  .. face box + the 5 SCRFD landmark points
    2. Alignment  .. the 112x112 canonical ArcFace crop (upscaled)
    3. Result     .. identity + cosine similarity, decided by the threshold

Press any key to advance to the next window; press 'q' to quit early.
"""

from __future__ import annotations

import argparse
import sys
from typing import List

import cv2
import numpy as np

from app.config import MATCHING_THRESHOLD
from app.detector import Detection
from app.recognition import RecognitionPipeline

LANDMARK_LABELS = ["LE", "RE", "N", "LM", "RM"]
LANDMARK_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize each stage of the face-recognition pipeline "
                    "in pop-up windows."
    )
    parser.add_argument("--image", required=True, help="path to the input photo")
    parser.add_argument(
        "--threshold", type=float, default=MATCHING_THRESHOLD,
        help="cosine similarity threshold for a 'Known' match "
             f"(default {MATCHING_THRESHOLD})",
    )
    return parser


MAX_DISPLAY_WIDTH = 1000
MAX_DISPLAY_HEIGHT = 700


def _fit_screen(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = min(1.0, MAX_DISPLAY_WIDTH / w, MAX_DISPLAY_HEIGHT / h)
    if scale >= 1.0:
        return frame
    return cv2.resize(
        frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
    )


def _show(named: str, frame: np.ndarray) -> bool:
    """Display 'frame' in a window until a key is pressed.

    Returns False if the user pressed 'q'/'ESC' to quit early.
    """
    frame = _fit_screen(frame)
    cv2.imshow(named, frame)
    key = cv2.waitKey(0) & 0xFF
    return key not in (ord("q"), 27)


def _draw_detection(image: np.ndarray, faces: List[Detection]) -> np.ndarray:
    out = image.copy()
    for face in faces:
        x1, y1, x2, y2 = (int(v) for v in face.bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            out, f"conf {face.confidence:.2f}", (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA,
        )
        for (px, py), label, color in zip(face.landmarks, LANDMARK_LABELS, LANDMARK_COLORS):
            cv2.rectangle(
                out, (int(px) - 4, int(py) - 4), (int(px) + 4, int(py) + 4),
                color, -1,
            )
            cv2.putText(
                out, label, (int(px) + 8, int(py) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
            )
    return out


def _draw_alignment(face_result, scale: int = 5) -> np.ndarray:
    aligned = face_result.aligned_face  # already 112x112 BGR
    h, w = aligned.shape[:2]
    big = cv2.resize(aligned, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

    ref = np.asarray(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ],
        dtype=np.float32,
    )
    for (px, py), label, color in zip(ref, LANDMARK_LABELS, LANDMARK_COLORS):
        cx, cy = int(px * scale), int(py * scale)
        cv2.circle(big, (cx, cy), 8, color, -1)
        cv2.putText(
            big, label, (cx + 12, cy - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA,
        )
    return big


def _draw_result(image: np.ndarray, results) -> np.ndarray:
    out = image.copy()
    for r in results:
        x1, y1, x2, y2 = (int(v) for v in r.bbox)
        color = (0, 200, 0) if r.match.is_known else (0, 0, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = r.match.display_label
        if r.expression is not None:
            label += f" | {r.expression.display_label}"
        cv2.putText(
            out, label, (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA,
        )
        bar_width = int(220 * r.match.similarity)
        cv2.rectangle(out, (x1, y2 + 6), (x1 + 220, y2 + 16), (255, 255, 255), 1)
        cv2.rectangle(out, (x1, y2 + 6), (x1 + bar_width, y2 + 16), color, -1)
    return out


def main() -> int:
    args = build_parser().parse_args()
    try:
        pipeline = RecognitionPipeline(threshold=args.threshold)
    except Exception as exc:
        print(f"SETUP ERROR: {exc}", file=sys.stderr)
        return 1
    if pipeline.expression_classifier is None:
        print(
            "WARNING: expression model is unavailable; showing identity only. "
            "Run `python -m scripts.download_models`.",
            file=sys.stderr,
        )

    try:
        from app.utils import load_image_rgb
        image = load_image_rgb(args.image)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    faces = pipeline.detector.detect(image)
    if not faces:
        print("No face detected in the image.")
        return 0
    if not pipeline.matcher.has_enrollments():
        print(
            "WARNING: no enrollment database loaded. The 'Result' window will "
            "show Unknown for everyone. Run `python -m scripts.enroll` first.",
            file=sys.stderr,
        )

    results = pipeline.recognize_image(image)

    print("Stage windows will appear one at a time. Press any key to advance, 'q' to quit.")

    if not _show("1. DETECTION - face box + 5 landmark points", _draw_detection(image, faces)):
        _done()
        return 0
    for r in results:
        if not _show("2. ALIGNMENT - canonical 112x112 ArcFace crop",
                     _draw_alignment(r, scale=5)):
            _done()
            return 0
    if not _show("3. RESULT - identity + expression", _draw_result(image, results)):
        _done()
        return 0
    _done()
    return 0


def _done() -> None:
    cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())