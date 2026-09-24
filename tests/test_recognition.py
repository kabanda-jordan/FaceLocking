"""Model-free integration checks for the recognition pipeline seam."""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.detector import Detection
from app.expression import ExpressionResult
from app.matcher import EmbeddingMatcher
from app.recognition import RecognitionPipeline


class FakeDetector:
    def detect(self, image):
        return [
            Detection(
                bbox=np.array([1, 1, 40, 40], dtype=np.float32),
                confidence=0.95,
                landmarks=np.zeros((5, 2), dtype=np.float32),
            )
        ]


class FakeAligner:
    def align(self, image, landmarks):
        return np.zeros((112, 112, 3), dtype=np.uint8), np.eye(2, 3)


class FakeEmbedder:
    def get_embedding(self, aligned_face):
        return np.ones(512, dtype=np.float32) / np.sqrt(512.0)


class FakeExpressionClassifier:
    def classify(self, aligned_face, landmarks=None):
        return ExpressionResult(
            label="smile",
            raw_emotion="happiness",
            confidence=0.9,
            probabilities={"happiness": 0.9},
            happiness_score=0.9,
        )


def make_pipeline():
    return RecognitionPipeline(
        detector=FakeDetector(),
        aligner=FakeAligner(),
        embedder=FakeEmbedder(),
        matcher=EmbeddingMatcher(),
        expression_classifier=FakeExpressionClassifier(),
    )


def test_identity_mode_still_requires_enrollment():
    pipeline = make_pipeline()
    with pytest.raises(ValueError, match="enrollment database is empty"):
        pipeline.recognize_image(np.zeros((50, 50, 3), dtype=np.uint8))


def test_expression_only_mode_returns_unknown_identity():
    pipeline = make_pipeline()
    result = pipeline.recognize_expressions(
        np.zeros((50, 50, 3), dtype=np.uint8)
    )[0]
    assert result.match.identity is None
    assert not result.match.is_known
    assert result.expression.label == "smile"


def test_expression_only_pipeline_skips_arcface_embedder():
    pipeline = RecognitionPipeline(
        detector=FakeDetector(),
        aligner=FakeAligner(),
        embedder=None,
        matcher=EmbeddingMatcher(),
        expression_classifier=FakeExpressionClassifier(),
        enable_identity=False,
    )
    result = pipeline.recognize_image(np.zeros((50, 50, 3), dtype=np.uint8))[0]
    assert result.embedding.size == 0
    assert result.expression.label == "smile"
