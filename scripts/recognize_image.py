"""Recognize faces in a single image (no webcam required).

Usage (from the project root):
    python -m scripts.recognize_image --image path/to/photo.jpg
    python -m scripts.recognize_image --image photo.jpg --save outputs/annotated.jpg
    python -m scripts.recognize_image --image photo.jpg --threshold 0.5

Prints, per detected face, something like:
    Known: Jordan   similarity=0.83 (threshold 0.40)
    Unknown         similarity=0.34 (threshold 0.40)

With --save it also writes an annotated copy of the image.
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import (                                  # noqa: E402
    EXPRESSION_CONFIDENCE,
    EXPRESSION_ANGER_THRESHOLD,
    EXPRESSION_INPUT_SIZE,
    EXPRESSION_LAUGH_FRAMES,
    EXPRESSION_MODEL_PATH,
    EXPRESSION_MOUTH_OPEN_THRESHOLD,
    MATCHING_THRESHOLD,
)
from app.enrollment import EnrollmentError                  # noqa: E402
from app.expression import ExpressionClassifier            # noqa: E402
from app.recognition import RecognitionPipeline             # noqa: E402
from app.utils import load_image_rgb, save_image_rgb        # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recognize faces in a single image",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", required=True,
                        help="path to the image to analyze")
    parser.add_argument("--threshold", type=float, default=MATCHING_THRESHOLD,
                        help="cosine-similarity threshold for Known/Unknown")
    parser.add_argument(
        "--no-expressions", action="store_true",
        help="skip smile/laugh/angry classification",
    )
    parser.add_argument(
        "--expressions-only", action="store_true",
        help="allow expression detection without an enrolled identity gallery",
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
    parser.add_argument("--save", default=None,
                        help="optional output path for an annotated copy")
    parser.add_argument("--json", action="store_true",
                        help="print machine-readable JSON results")
    parser.add_argument("--show", action="store_true",
                        help="pop up a window with the annotated result "
                             "(press any key to close)")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.no_expressions and args.expressions_only:
        print(
            "ERROR: --no-expressions and --expressions-only cannot be used together.",
            file=sys.stderr,
        )
        return 1

    if not os.path.exists(args.image):
        print(f"ERROR: image not found: {args.image}", file=sys.stderr)
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
            # Expression support is additive: keep identity recognition usable
            # when the optional model has not been downloaded yet.
            print(
                f"WARNING: expression detection disabled ({exc}).\n"
                "Run `python -m scripts.download_models` or use "
                "--no-expressions to hide this message.",
                file=sys.stderr,
            )

    try:
        pipeline = RecognitionPipeline(
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

    try:
        image = load_image_rgb(args.image)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    try:
        if args.expressions_only:
            results = pipeline.recognize_expressions(image)
        else:
            results = pipeline.recognize_image(image)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not results:
        print("No face detected in the image.")
        return 0 if args.json else 0

    if args.json:
        import json

        payload = [
            {
                "bbox": [int(v) for v in r.bbox],
                "confidence": round(r.confidence, 4),
                "identity": r.match.identity,
                "known": r.match.is_known,
                "similarity": round(r.match.similarity, 4),
                "expression": (
                    r.expression.to_dict() if r.expression is not None else None
                ),
            }
            for r in results
        ]
        print(json.dumps({"results": payload}, indent=2))
    else:
        for result in results:
            print(result.match)
            if result.expression is not None:
                print(
                    f"  expression={result.expression.display_label} "
                    f"(raw={result.expression.raw_emotion}, "
                    f"mouth_open={result.expression.mouth_open_score:.2f})"
                )
            else:
                print("  expression=unavailable (FER+ model is not loaded)")
            print(f"  bbox={[int(v) for v in result.bbox]} "
                  f"det_conf={result.confidence:.3f}")

    annotated = None
    if args.save or args.show:
        annotated = image.copy()
        for result in results:
            x1, y1, x2, y2 = (int(v) for v in result.bbox)
            color = (0, 200, 0) if result.match.is_known else (0, 0, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            identity_label = result.match.display_label
            expression_label = (
                result.expression.display_label
                if result.expression is not None else ""
            )
            label = identity_label
            if expression_label:
                label += f" | {expression_label}"
            cv2.putText(
                annotated, label, (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
            )
            landmark_colors = (
                (255, 0, 0), (0, 255, 0), (0, 0, 255),
                (255, 255, 0), (0, 255, 255),
            )
            for point, point_color in zip(result.landmarks, landmark_colors):
                px, py = (int(v) for v in point)
                cv2.rectangle(
                    annotated, (px - 3, py - 3), (px + 3, py + 3),
                    point_color, 1, cv2.LINE_AA,
                )

    if args.save:
        save_image_rgb(args.save, annotated)
        print(f"Annotated image saved to: {args.save}")

    if args.show:
        # Pop a window so you can SEE the result (run from your own terminal,
        # not a headless shell). Wait for any key, then close cleanly.
        cv2.imshow("Face Recognition result", annotated)
        print("Displaying result in a window. Press any key to close it.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())