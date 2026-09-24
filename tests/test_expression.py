"""Tests for expression label mapping and live temporal smoothing.

These tests do not download or load an ONNX model.  They exercise the
model-independent decision policy and the small per-face history used by the
webcam overlay.
"""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import EXPRESSION_MODEL_PATH
from app.expression import (
    EMOTION_LABELS,
    ExpressionClassifier,
    ExpressionResult,
    ExpressionStabilizer,
    expression_from_probabilities,
)


def probabilities(happy=0.0, angry=0.0, neutral=0.0):
    values = np.zeros(len(EMOTION_LABELS), dtype=np.float64)
    values[0] = neutral
    values[1] = happy
    values[4] = angry
    remainder = max(0.0, 1.0 - values.sum())
    if remainder:
        values[2] = remainder
    return values / values.sum()


class TestExpressionPolicy:
    def test_happiness_maps_to_smile(self):
        result = expression_from_probabilities(probabilities(happy=0.90))
        assert result.label == "smile"
        assert result.raw_emotion == "happiness"
        assert result.is_smile
        assert not result.is_laugh

    def test_open_happiness_maps_to_laugh(self):
        result = expression_from_probabilities(
            probabilities(happy=0.90), mouth_open_score=0.8
        )
        assert result.label == "laugh"
        assert result.is_laugh
        assert not result.is_smile

    def test_sustained_happiness_maps_to_laugh(self):
        result = expression_from_probabilities(
            probabilities(happy=0.90), happy_streak=3
        )
        assert result.label == "laugh"

    def test_anger_maps_to_angry(self):
        result = expression_from_probabilities(probabilities(angry=0.90))
        assert result.label == "angry"
        assert result.is_angry

    def test_low_confidence_is_uncertain(self):
        # No class is above the 0.40 reporting threshold.
        values = np.full(len(EMOTION_LABELS), 1.0 / len(EMOTION_LABELS))
        result = expression_from_probabilities(values)
        assert result.label == "uncertain"

    def test_logits_are_normalized(self):
        result = expression_from_probabilities([0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        assert sum(result.probabilities.values()) == pytest.approx(1.0)
        assert result.probabilities["happiness"] > 0.99

    def test_result_serializes_without_numpy_types(self):
        result = expression_from_probabilities(probabilities(happy=0.9))
        payload = result.to_dict()
        assert payload["label"] == "smile"
        assert all(isinstance(value, float) for value in payload["probabilities"].values())


class TestExpressionStabilizer:
    def make_result(self, label="smile", raw="happiness", happiness=0.9):
        expression = ExpressionResult(
            label=label,
            raw_emotion=raw,
            confidence=happiness,
            probabilities={"happiness": happiness},
            happiness_score=happiness,
            anger_score=0.0,
            mouth_open_score=0.0,
        )
        return SimpleNamespace(
            bbox=np.array([10.0, 10.0, 110.0, 110.0]),
            expression=expression,
        )

    def test_sustained_smile_is_promoted_to_laugh(self):
        stabilizer = ExpressionStabilizer(laugh_frames=3)
        result = self.make_result()
        assert stabilizer.update([result], timestamp=0.0)[0].expression.label == "smile"
        assert stabilizer.update([result], timestamp=0.1)[0].expression.label == "smile"
        assert stabilizer.update([result], timestamp=0.2)[0].expression.label == "laugh"

    def test_interrupted_happy_streak_resets(self):
        stabilizer = ExpressionStabilizer(laugh_frames=3)
        happy = self.make_result()
        neutral = self.make_result(label="neutral", raw="neutral", happiness=0.0)
        stabilizer.update([happy], timestamp=0.0)
        stabilizer.update([happy], timestamp=0.1)
        stabilizer.update([neutral], timestamp=0.2)
        assert stabilizer.update([happy], timestamp=0.3)[0].expression.label == "smile"

    def test_results_without_expression_pass_through(self):
        stabilizer = ExpressionStabilizer()
        result = SimpleNamespace(bbox=np.array([0.0, 0.0, 1.0, 1.0]), expression=None)
        assert stabilizer.update([result], timestamp=0.0)[0] is result


def test_missing_expression_model_has_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_models"):
        ExpressionClassifier(str(tmp_path / "missing.onnx"))


@pytest.mark.skipif(
    not os.path.exists(EXPRESSION_MODEL_PATH),
    reason="FER+ expression model not downloaded. Run: python -m scripts.download_models",
)
class TestExpressionModel:
    def test_preprocess_shape_and_probabilities(self):
        classifier = ExpressionClassifier(EXPRESSION_MODEL_PATH)
        face = np.full((112, 112, 3), 160, dtype=np.uint8)
        tensor = classifier.preprocess(face)
        assert tensor.shape == (1, 1, 64, 64)
        assert tensor.dtype == np.float32
        probabilities = classifier.predict_probabilities(face)
        assert probabilities.shape == (len(EMOTION_LABELS),)
        assert np.isclose(probabilities.sum(), 1.0)

    def test_classification_returns_structured_result(self):
        classifier = ExpressionClassifier(EXPRESSION_MODEL_PATH)
        result = classifier.classify(np.full((112, 112, 3), 160, dtype=np.uint8))
        assert result.label in {
            "neutral", "smile", "laugh", "angry", "sad", "surprised",
            "fearful", "disgust", "contempt", "uncertain",
        }
        assert result.raw_emotion in EMOTION_LABELS
