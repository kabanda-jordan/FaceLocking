"""Facial-expression classification and lightweight temporal smoothing.

The identity embedder answers *who is this?*.  It deliberately does not answer
*what expression is this?*.  This module keeps those two jobs separate:

* ``ExpressionClassifier`` runs a small local ONNX FER+ model on the already
  aligned face crop.
* ``ExpressionStabilizer`` keeps a short per-face history so a live camera
  does not flicker between labels on every noisy frame.

FER+ has an ``happiness`` class, but it does not have a separate ``laugh``
class.  We therefore expose ``smile`` for happiness and promote it to
``laugh`` when the mouth looks open or happiness persists for a few frames.
That is a *visual* approximation of laughing; this project does not use the
microphone, so it cannot confirm sound.

Everything runs locally through ONNX Runtime.  No frame is sent to a service.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from .utils import build_session_options


# The order is the label order of the ONNX Model Zoo FER+ model.  Do not
# reorder this tuple: the model's output indexes are positional.
EMOTION_LABELS: Tuple[str, ...] = (
    "neutral",
    "happiness",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear",
    "contempt",
)

# User-facing labels used by the CLIs.  ``laugh`` is derived from happiness
# plus visual/temporal evidence; it is not a native FER+ class.
EXPRESSION_LABELS: Tuple[str, ...] = (
    "neutral",
    "smile",
    "laugh",
    "angry",
    "sad",
    "surprised",
    "fearful",
    "disgust",
    "contempt",
    "uncertain",
)

__all__ = [
    "EMOTION_LABELS",
    "EXPRESSION_LABELS",
    "ExpressionClassifier",
    "ExpressionResult",
    "ExpressionStabilizer",
    "estimate_mouth_openness",
    "expression_from_probabilities",
    "normalize_probabilities",
]


def _normalise_probabilities(scores: Sequence[float]) -> np.ndarray:
    """Convert model scores to a finite, normalized probability vector.

    The FER+ graph emits logits, but accepting an already-normalized output
    makes the wrapper tolerant of other compatible FER ONNX exports.  The
    function is public-ish (and unit-tested) so post-processing stays
    deterministic and does not depend on a model file.
    """
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size != len(EMOTION_LABELS):
        raise ValueError(
            f"Expression model returned {values.size} scores; expected "
            f"{len(EMOTION_LABELS)}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("Expression model returned non-finite scores")

    # A few exports already return probabilities.  Avoid applying softmax to
    # those, while still accepting tiny floating-point deviations.
    if np.all(values >= -1e-6) and np.all(values <= 1.0 + 1e-6):
        values = np.clip(values, 0.0, 1.0)
        total = float(values.sum())
        if total > 1e-8 and abs(total - 1.0) <= 1e-3:
            return values / total

    shifted = values - np.max(values)
    exponentials = np.exp(np.clip(shifted, -745.0, 0.0))
    total = float(exponentials.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.full(values.shape, 1.0 / values.size, dtype=np.float64)
    return exponentials / total


# Public spelling for callers who prefer US English.  Keep the original
# internal spelling as the implementation used by the rest of the module.
normalize_probabilities = _normalise_probabilities


def _clamp01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def estimate_mouth_openness(aligned_face: np.ndarray) -> float:
    """Estimate a visual mouth-open score from an aligned face crop.

    SCRFD supplies mouth *corners*, but not upper/lower lip points.  A small
    image cue fills that gap: a genuinely open mouth creates a taller dark
    band in the canonical lower-face region, while a closed smile usually
    produces only a thin line.  This is intentionally a soft score rather
    than a claim of exact lip geometry; the temporal filter in
    ``ExpressionStabilizer`` is what makes the final label stable.

    The function works on a BGR or grayscale ``(H, W[, 3])`` uint8 image and
    returns a value in ``[0, 1]``.  Bad/blank inputs return zero.
    """
    face = np.asarray(aligned_face)
    if face.ndim not in (2, 3) or face.size == 0:
        return 0.0
    if face.ndim == 3:
        if face.shape[2] < 3:
            return 0.0
        gray = cv2.cvtColor(face[:, :, :3], cv2.COLOR_BGR2GRAY)
    else:
        gray = face
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    h, w = gray.shape[:2]
    if h < 20 or w < 20:
        return 0.0

    # ArcFace's canonical template puts the mouth around x=42..71, y=92.
    # Use a slightly wider region so lips and the opening are both included.
    x1, x2 = int(round(0.32 * w)), int(round(0.68 * w))
    y1, y2 = int(round(0.66 * h)), int(round(0.94 * h))
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0

    # Very dark/flat frames do not contain enough evidence for a mouth cue.
    if float(gray.std()) < 7.0:
        return 0.0

    # Estimate the dark-mouth threshold relative to the local face.  A fixed
    # threshold is brittle across webcams; a percentile adapts to exposure.
    local_background = float(np.percentile(gray, 65))
    dark_threshold = max(28.0, local_background - 32.0)
    dark = roi < dark_threshold

    # Longest contiguous run of dark rows approximates vertical openness.
    row_is_dark = dark.mean(axis=1) >= 0.35
    longest = 0
    current = 0
    for is_dark in row_is_dark:
        if bool(is_dark):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    row_score = np.clip(longest / max(3.0, 0.16 * roi.shape[0]), 0.0, 1.0)

    # An open mouth also occupies more dark area than a closed lip line.
    area_score = np.clip((float(dark.mean()) - 0.08) / 0.32, 0.0, 1.0)

    # Contrast helps reject a uniformly dark background.  The vertical cue is
    # weighted more heavily because it is the most useful distinction here.
    contrast = _clamp01((float(roi.std()) - 10.0) / 55.0)
    score = 0.58 * row_score + 0.30 * area_score + 0.12 * contrast
    return _clamp01(score)


def expression_from_probabilities(
    probabilities: Sequence[float],
    mouth_open_score: float = 0.0,
    happy_streak: int = 0,
    confidence_threshold: float = 0.40,
    mouth_open_threshold: float = 0.45,
    laugh_frames: int = 3,
) -> "ExpressionResult":
    """Map FER+ probabilities to the labels shown by this project.

    This pure decision function makes the policy easy to test and tune:

    * high ``happiness`` -> ``smile``;
    * high ``happiness`` + an open-looking mouth or a sustained happy streak
      -> ``laugh``;
    * high ``anger`` -> ``angry``;
    * low-confidence output -> ``uncertain``.

    The raw model emotion is retained in the returned result for diagnostics.
    """
    probs = _normalise_probabilities(probabilities)
    index = int(np.argmax(probs))
    raw_emotion = EMOTION_LABELS[index]
    happiness = float(probs[1])
    anger = float(probs[4])
    mouth_open_score = _clamp01(mouth_open_score)
    confidence_threshold = _clamp01(confidence_threshold)
    mouth_open_threshold = _clamp01(mouth_open_threshold)
    laugh_frames = max(1, int(laugh_frames))
    happy_streak = max(0, int(happy_streak))

    if anger >= confidence_threshold and anger >= happiness:
        label = "angry"
        confidence = anger
    elif happiness >= confidence_threshold:
        looks_like_laugh = (
            mouth_open_score >= mouth_open_threshold
            or happy_streak >= laugh_frames
        )
        label = "laugh" if looks_like_laugh else "smile"
        confidence = happiness
    elif raw_emotion == "neutral" and probs[index] >= confidence_threshold:
        label = "neutral"
        confidence = probs[index]
    elif probs[index] >= confidence_threshold:
        # Keep the other native FER+ categories useful rather than throwing
        # them away.  They are not specially tuned by this project.
        label = {
            "surprise": "surprised",
            "sadness": "sad",
            "disgust": "disgust",
            "fear": "fearful",
            "contempt": "contempt",
        }.get(raw_emotion, "uncertain")
        confidence = probs[index]
    else:
        label = "uncertain"
        confidence = probs[index]

    return ExpressionResult(
        label=label,
        raw_emotion=raw_emotion,
        confidence=_clamp01(confidence),
        probabilities={
            name: float(probs[i]) for i, name in enumerate(EMOTION_LABELS)
        },
        mouth_open_score=mouth_open_score,
        happiness_score=happiness,
        anger_score=anger,
    )


@dataclass
class ExpressionResult:
    """Expression prediction for one face.

    ``label`` is the user-facing decision (``smile``, ``laugh``, ``angry``,
    etc.).  ``raw_emotion`` is the native FER+ class and is useful when tuning
    or diagnosing a false prediction.
    """

    label: str
    raw_emotion: str
    confidence: float
    probabilities: Dict[str, float] = field(default_factory=dict)
    mouth_open_score: float = 0.0
    happiness_score: float = 0.0
    anger_score: float = 0.0

    @property
    def expression(self) -> str:
        """Alias useful to callers that prefer ``result.expression``."""
        return self.label

    @property
    def emotion(self) -> str:
        """Alias for the native FER+ class."""
        return self.raw_emotion

    @property
    def is_smile(self) -> bool:
        return self.label == "smile"

    @property
    def is_laugh(self) -> bool:
        return self.label == "laugh"

    @property
    def is_angry(self) -> bool:
        return self.label == "angry"

    @property
    def display_label(self) -> str:
        return f"{self.label.title()} {self.confidence:.2f}"

    def to_dict(self) -> Dict[str, object]:
        """Return JSON-serialisable fields for CLI/API output."""
        return {
            "label": self.label,
            "raw_emotion": self.raw_emotion,
            "confidence": round(float(self.confidence), 4),
            "mouth_open_score": round(float(self.mouth_open_score), 4),
            "happiness_score": round(float(self.happiness_score), 4),
            "anger_score": round(float(self.anger_score), 4),
            "probabilities": {
                key: round(float(value), 6)
                for key, value in self.probabilities.items()
            },
            "is_smile": self.is_smile,
            "is_laugh": self.is_laugh,
            "is_angry": self.is_angry,
        }


class ExpressionClassifier:
    """ONNX Runtime wrapper for the local FER+ expression model."""

    def __init__(
        self,
        model_path: str,
        input_size: Tuple[int, int] = (64, 64),
        confidence_threshold: float = 0.40,
        mouth_open_threshold: float = 0.45,
        laugh_frames: int = 3,
        providers: Optional[List[str]] = None,
    ) -> None:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Expression model not found: {model_path}\n"
                "Run `python -m scripts.download_models` to fetch it."
            )
        try:
            with open(model_path, "rb") as model_file:
                if model_file.read(128).startswith(b"version https://git-lfs.github.com/spec"):
                    raise ValueError(
                        "Expression model is a Git LFS pointer, not the ONNX "
                        "binary. Run `python -m scripts.download_models` again."
                    )
        except OSError:
            pass

        self.model_path = model_path
        self.confidence_threshold = _clamp01(confidence_threshold)
        self.mouth_open_threshold = _clamp01(mouth_open_threshold)
        self.laugh_frames = max(1, int(laugh_frames))
        self.session = ort.InferenceSession(
            model_path,
            sess_options=build_session_options(),
            providers=providers or ["CPUExecutionProvider"],
        )

        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name
        self.input_dtype = getattr(input_meta, "type", "tensor(float)")
        declared = list(input_meta.shape)
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.input_channels = 1
        if len(declared) >= 4:
            if isinstance(declared[1], int) and declared[1] > 0:
                self.input_channels = declared[1]
            if (
                isinstance(declared[2], int)
                and declared[2] > 0
                and isinstance(declared[3], int)
                and declared[3] > 0
            ):
                self.input_size = (declared[2], declared[3])
        self.output_names = [output.name for output in self.session.get_outputs()]

    def preprocess(self, aligned_face: np.ndarray) -> np.ndarray:
        """Convert a BGR aligned face to the FER+ NCHW input tensor.

        The downloaded FER+ model takes a single 64x64 grayscale channel and
        raw 0..255 values.  The RGB branch keeps the wrapper usable with a
        compatible three-channel ONNX expression model.
        """
        face = np.asarray(aligned_face)
        if face.ndim not in (2, 3):
            raise ValueError(
                "ExpressionClassifier expects an HxW or HxWx3 aligned face, "
                f"got shape {face.shape}"
            )
        if face.ndim == 3 and face.shape[2] < 3:
            raise ValueError(
                "ExpressionClassifier expects an HxW or HxWx3 aligned face, "
                f"got shape {face.shape}"
            )
        if face.dtype != np.uint8:
            face = np.clip(face, 0, 255).astype(np.uint8)

        height, width = self.input_size
        if self.input_channels == 1:
            if face.ndim == 2:
                gray = face
            else:
                gray = cv2.cvtColor(face[:, :, :3], cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
            tensor = gray.astype(np.float32)[None, None, :, :]
        else:
            # Most RGB expression exports use RGB rather than BGR.  Preserve
            # colour when the caller supplied it instead of throwing it away.
            color = face[:, :, :3] if face.ndim == 3 else cv2.cvtColor(
                face, cv2.COLOR_GRAY2BGR
            )
            color = cv2.resize(color, (width, height), interpolation=cv2.INTER_AREA)
            rgb = color[:, :, ::-1]
            tensor = rgb.transpose(2, 0, 1)[None]
            if "uint8" in str(self.input_dtype).lower():
                tensor = tensor.astype(np.uint8)
            else:
                tensor = tensor.astype(np.float32) / 255.0
            if self.input_channels > 3:
                repeats = int(np.ceil(self.input_channels / 3))
                tensor = np.repeat(tensor, repeats, axis=1)
            tensor = tensor[:, : self.input_channels]

        if "uint8" in str(self.input_dtype).lower():
            return np.ascontiguousarray(np.clip(tensor, 0, 255).astype(np.uint8))
        return np.ascontiguousarray(tensor, dtype=np.float32)

    def predict_probabilities(self, aligned_face: np.ndarray) -> np.ndarray:
        """Run FER+ and return an 8-element probability vector."""
        tensor = self.preprocess(aligned_face)
        outputs = self.session.run(self.output_names, {self.input_name: tensor})
        if not outputs:
            raise ValueError("Expression model returned no output tensors")
        scores = np.asarray(outputs[0]).reshape(-1)
        return _normalise_probabilities(scores)

    def classify(
        self,
        aligned_face: np.ndarray,
        landmarks: Optional[np.ndarray] = None,
        happy_streak: int = 0,
    ) -> ExpressionResult:
        """Classify one aligned face.

        ``landmarks`` is accepted for a stable interface with future landmark
        models; the current FER+ path uses the canonical aligned crop directly.
        """
        # Validate shape early, even though landmarks are not needed by FER+.
        if landmarks is not None:
            points = np.asarray(landmarks)
            if points.ndim != 2 or points.shape[0] != 5 or points.shape[1] != 2:
                raise ValueError(
                    "Expected landmarks of shape (5, 2); "
                    f"got {points.shape}"
                )
        probabilities = self.predict_probabilities(aligned_face)
        mouth_open_score = estimate_mouth_openness(aligned_face)
        return expression_from_probabilities(
            probabilities,
            mouth_open_score=mouth_open_score,
            happy_streak=happy_streak,
            confidence_threshold=self.confidence_threshold,
            mouth_open_threshold=self.mouth_open_threshold,
            laugh_frames=self.laugh_frames,
        )

    def predict(
        self,
        aligned_face: np.ndarray,
        landmarks: Optional[np.ndarray] = None,
        happy_streak: int = 0,
    ) -> ExpressionResult:
        """Compatibility alias for callers accustomed to ``predict`` APIs."""
        return self.classify(
            aligned_face, landmarks=landmarks, happy_streak=happy_streak
        )


def _box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    a = np.asarray(box_a, dtype=np.float32).reshape(4)
    b = np.asarray(box_b, dtype=np.float32).reshape(4)
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0 else 0.0


@dataclass
class _ExpressionTrack:
    bbox: np.ndarray
    happy_streak: int = 0
    last_seen: float = 0.0


class ExpressionStabilizer:
    """Smooth expression labels across successive webcam results.

    Faces are associated by bounding-box IoU.  A happy face must persist for
    ``laugh_frames`` worker updates before a closed-mouth smile is promoted to
    ``laugh``.  This is intentionally lightweight: it is not a face tracker,
    but it prevents one-frame FER noise from making the overlay jump around.
    """

    def __init__(
        self,
        laugh_frames: int = 3,
        mouth_open_threshold: float = 0.45,
        confidence_threshold: float = 0.40,
        max_age_seconds: float = 2.0,
        iou_threshold: float = 0.20,
    ) -> None:
        self.laugh_frames = max(1, int(laugh_frames))
        self.mouth_open_threshold = _clamp01(mouth_open_threshold)
        self.confidence_threshold = _clamp01(confidence_threshold)
        self.max_age_seconds = max(0.1, float(max_age_seconds))
        self.iou_threshold = _clamp01(iou_threshold)
        self._tracks: List[_ExpressionTrack] = []

    def reset(self) -> None:
        self._tracks.clear()

    def update(
        self,
        results: Sequence[object],
        timestamp: Optional[float] = None,
    ) -> List[object]:
        """Return results with temporally stable expression labels.

        ``results`` can be any objects exposing ``bbox`` and (optionally)
        ``expression``.  Returning a list of shallow/replaced objects keeps
        this class usable with both ``FaceRecognitionResult`` and test
        doubles.
        """
        now = time.monotonic() if timestamp is None else float(timestamp)
        self._tracks = [
            track
            for track in self._tracks
            if now - track.last_seen <= self.max_age_seconds
        ]
        used = set()
        output: List[object] = []
        for result in results:
            bbox = np.asarray(getattr(result, "bbox"), dtype=np.float32).reshape(4)
            expression = getattr(result, "expression", None)
            if expression is None:
                output.append(result)
                continue

            track_index = None
            best_iou = self.iou_threshold
            for index, track in enumerate(self._tracks):
                if index in used:
                    continue
                score = _box_iou(track.bbox, bbox)
                if score >= best_iou:
                    best_iou = score
                    track_index = index

            if track_index is None:
                track = _ExpressionTrack(bbox=bbox.copy(), last_seen=now)
                self._tracks.append(track)
                track_index = len(self._tracks) - 1
            else:
                track = self._tracks[track_index]
            used.add(track_index)

            if (
                expression.raw_emotion == "happiness"
                and expression.happiness_score >= self.confidence_threshold
            ):
                track.happy_streak += 1
            else:
                track.happy_streak = 0

            label = expression.label
            if label == "smile" and (
                expression.mouth_open_score >= self.mouth_open_threshold
                or track.happy_streak >= self.laugh_frames
            ):
                label = "laugh"
            if label == expression.label:
                output.append(result)
            else:
                output.append(
                    replace(
                        expression,
                        label=label,
                    )
                )
                # Replace the result's expression while preserving all other
                # fields.  ``dataclasses.replace`` is intentionally not used on
                # the outer object because callers may use lightweight test
                # doubles instead of the project dataclass.
                updated_expression = output[-1]
                if hasattr(result, "__dataclass_fields__"):
                    output[-1] = replace(result, expression=updated_expression)
                else:
                    try:
                        result.expression = updated_expression  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    # Lightweight test doubles are mutable objects; return the
                    # original result object after updating its expression.
                    output[-1] = result

            track.bbox = bbox.copy()
            track.last_seen = now

        return output
