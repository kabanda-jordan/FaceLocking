"""Recognition pipeline: compose detection, alignment, identity and expression.

This module is deliberately thin: it wires together the detector, aligner,
embedder, expression classifier and matcher so that each stage stays explicit,
testable and replaceable. Expression inference is optional and never changes
the identity embedding or Known/Unknown decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .aligner import FaceAligner
from .config import (
    DETECTOR_CONFIDENCE,
    DETECTOR_INPUT_SIZE,
    DETECTOR_MODEL_PATH,
    DETECTOR_NMS,
    EMBEDDER_MODEL_PATH,
    EMBEDDINGS_NPZ_PATH,
    EXPRESSION_CONFIDENCE,
    EXPRESSION_ANGER_THRESHOLD,
    EXPRESSION_ENABLED,
    EXPRESSION_INPUT_SIZE,
    EXPRESSION_LAUGH_FRAMES,
    EXPRESSION_MODEL_PATH,
    EXPRESSION_MOUTH_OPEN_THRESHOLD,
    MATCHING_THRESHOLD,
)
from .detector import SCRFDDetector
from .embedder import ArcFaceEmbedder
from .expression import ExpressionClassifier, ExpressionResult
from .matcher import EmbeddingMatcher, MatchResult


@dataclass
class FaceRecognitionResult:
    """Everything the pipeline knows about ONE face in an image."""

    bbox: np.ndarray          # [x1, y1, x2, y2]
    confidence: float         # detector score
    landmarks: np.ndarray     # (5, 2)
    aligned_face: np.ndarray  # (112, 112, 3) BGR crop (for inspection/demo)
    embedding: np.ndarray     # (D,) L2-normalized; empty in expression-only mode
    match: MatchResult        # identity / similarity / threshold decision
    expression: Optional[ExpressionResult] = None  # local FER+ result


class RecognitionPipeline:
    """Entry point used by the CLI scripts (webcam + single image)."""

    def __init__(
        self,
        detector: Optional[SCRFDDetector] = None,
        aligner: Optional[FaceAligner] = None,
        embedder: Optional[ArcFaceEmbedder] = None,
        matcher: Optional[EmbeddingMatcher] = None,
        expression_classifier: Optional[ExpressionClassifier] = None,
        threshold: Optional[float] = None,
        expression_threshold: Optional[float] = None,
        expression_anger_threshold: Optional[float] = None,
        enable_expressions: bool = EXPRESSION_ENABLED,
        enable_identity: bool = True,
    ) -> None:
        # Build each stage from defaults unless the caller injects its own
        # (injection is how unit tests replace heavy/real components).
        self.detector = detector or SCRFDDetector(
            model_path=DETECTOR_MODEL_PATH,
            input_size=DETECTOR_INPUT_SIZE,
            confidence_threshold=DETECTOR_CONFIDENCE,
            nms_threshold=DETECTOR_NMS,
        )
        self.aligner = aligner or FaceAligner()
        if embedder is not None:
            self.embedder = embedder
        elif enable_identity:
            self.embedder = ArcFaceEmbedder(model_path=EMBEDDER_MODEL_PATH)
        else:
            # Expression-only mode does not need the large ArcFace binary.
            self.embedder = None
        self.matcher = matcher or EmbeddingMatcher(
            threshold=threshold if threshold is not None else MATCHING_THRESHOLD
        )

        # Expression inference is optional so existing identity-only projects
        # and tests remain usable when the third model has not been downloaded.
        # A bad optional model is reported through ``expression_error`` rather
        # than taking down the ArcFace pipeline.
        self.expression_error: Optional[str] = None
        if expression_classifier is not None:
            self.expression_classifier = expression_classifier
        elif enable_expressions:
            try:
                self.expression_classifier = ExpressionClassifier(
                    model_path=EXPRESSION_MODEL_PATH,
                    input_size=EXPRESSION_INPUT_SIZE,
                    confidence_threshold=(
                        expression_threshold
                        if expression_threshold is not None
                        else EXPRESSION_CONFIDENCE
                    ),
                    anger_threshold=(
                        expression_anger_threshold
                        if expression_anger_threshold is not None
                        else EXPRESSION_ANGER_THRESHOLD
                    ),
                    mouth_open_threshold=EXPRESSION_MOUTH_OPEN_THRESHOLD,
                    laugh_frames=EXPRESSION_LAUGH_FRAMES,
                )
            except Exception as exc:
                self.expression_classifier = None
                self.expression_error = str(exc)
        else:
            self.expression_classifier = None

        if matcher is None:
            # Load the enrollment database written by `scripts.enroll`.
            # If it does not exist yet, leave the matcher empty and let
            # recognize_image() raise a friendly, actionable error.
            try:
                with np.load(EMBEDDINGS_NPZ_PATH) as data:
                    self.matcher.set_enrollment(
                        {name: data[name] for name in data.files}
                    )
            except FileNotFoundError:
                pass

    # ------------------------------------------------------------------
    # The recognizable flow (explicitly, stage after stage)
    # ------------------------------------------------------------------

    def _recognize_face(self, image: np.ndarray, face) -> FaceRecognitionResult:
        landmarks = np.asarray(face.landmarks, dtype=np.float32)
        # 1) align the detected face into the canonical ArcFace layout
        aligned, _ = self.aligner.align(image, landmarks)
        # 2) embed the aligned face (skipped in expression-only mode)
        if self.embedder is None:
            embedding = np.empty(0, dtype=np.float32)
        else:
            embedding = self.embedder.get_embedding(aligned)
        # 3) classify the expression from the same aligned crop (identity and
        #    expression are independent decisions)
        expression = None
        if self.expression_classifier is not None:
            try:
                expression = self.expression_classifier.classify(
                    aligned, landmarks=landmarks
                )
            except Exception as exc:
                # Expression inference is an optional enhancement.  A bad
                # crop/model must not erase an otherwise valid identity match.
                self.expression_error = str(exc)
        # 4) match against the enrollment database.  Expression-only mode can
        #    still return useful results for faces that have not been enrolled.
        if self.embedder is not None and self.matcher.has_enrollments():
            match = self.matcher.match(embedding)
        else:
            match = MatchResult(
                identity=None,
                similarity=0.0,
                threshold=self.matcher.threshold,
                is_known=False,
                scores={},
            )
        return FaceRecognitionResult(
            bbox=face.bbox,
            confidence=face.confidence,
            landmarks=landmarks,
            aligned_face=aligned,
            embedding=embedding,
            match=match,
            expression=expression,
        )

    def recognize_image(
        self, image: np.ndarray, require_enrollment: bool = True
    ) -> List[FaceRecognitionResult]:
        """Recognize every face in one image (BGR). Handles 0..N faces.

        Identity mode keeps the original safety check and raises when no
        people have been enrolled. Pass ``require_enrollment=False`` (or use
        ``recognize_expressions``) when you only want expression labels. A
        pipeline constructed with ``enable_identity=False`` also skips the
        enrollment check automatically.
        """
        if (
            require_enrollment
            and self.embedder is not None
            and not self.matcher.has_enrollments()
        ):
            raise ValueError(
                "enrollment database is empty; run `python -m scripts.enroll` first"
            )
        faces = self.detector.detect(image)
        return [self._recognize_face(image, face) for face in faces]

    def recognize_expressions(
        self, image: np.ndarray
    ) -> List[FaceRecognitionResult]:
        """Detect expressions even when no identity gallery is enrolled."""
        return self.recognize_image(image, require_enrollment=False)

    def recognize_frame(
        self, frame: np.ndarray, require_enrollment: bool = True
    ) -> List[FaceRecognitionResult]:
        """Recognize every face in a webcam frame (same as recognize_image)."""
        return self.recognize_image(frame, require_enrollment=require_enrollment)
