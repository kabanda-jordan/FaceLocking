"""Combined regression checks for face lock and live expression labels."""

import numpy as np

from app.expression import ExpressionResult, ExpressionStabilizer
from app.tracking import FaceLock


def result(x, label, raw_emotion, score):
    expression = ExpressionResult(
        label=label,
        raw_emotion=raw_emotion,
        confidence=score,
        probabilities={raw_emotion: score},
        happiness_score=score if raw_emotion == "happiness" else 0.0,
        anger_score=score if raw_emotion == "anger" else 0.0,
    )
    return type(
        "ExpressionResultBox",
        (),
        {
            "bbox": np.array([x, 0, x + 100, 100], dtype=np.float32),
            "expression": expression,
        },
    )()


def test_lock_and_expression_pipeline_cooperate():
    lock = FaceLock()
    stabilizer = ExpressionStabilizer(laugh_frames=3)
    target = result(0, "smile", "happiness", 0.9)
    other = result(200, "angry", "anger", 0.9)

    lock.enable([target, other])
    assert lock.filter([target, other]) == [target]
    assert lock.status == "on"

    for frame in range(3):
        target = result(0, "smile", "happiness", 0.9)
        stable = stabilizer.update([target], timestamp=frame * 0.1)
    assert stable[0].expression.label == "laugh"

    angry = result(0, "angry", "anger", 0.9)
    stable = stabilizer.update([angry], timestamp=0.4)
    assert stable[0].expression.label == "angry"
    assert lock.filter([target, other]) == [target]
