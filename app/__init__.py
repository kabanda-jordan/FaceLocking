"""Application package for the ArcFace + ONNX face-recognition pipeline.

This package is split by responsibility:

  config.py        path / model / threshold settings
  detector.py      face detection (SCRFD, ONNX) -> boxes + confidence + 5 landmarks
  aligner.py       5-point similarity-transform alignment -> standardized face
  embedder.py      ArcFace (ONNX) inference -> L2-normalized 512-dim embedding
  matcher.py       cosine-similarity matching against an enrollment database
  expression.py    local FER+ expression classification + temporal smoothing
  tracking.py      one-face lock + landmark motion tracking for the live overlay
  enrollment.py    turning raw face photos into stored per-identity embeddings
  recognition.py   composes detector -> aligner -> embedder -> matcher -> expression
  qt_compat.py     Linux/OpenCV Qt font and display compatibility
  utils.py         small shared helpers (normalization, image I/O, ...)
"""

__version__ = "1.1.0"


# Prepare the OpenCV Qt plugin before any live-camera window is created.  This
# is deliberately done at package import time because the warning is emitted
# when Qt initializes, not when ``cv2`` is first imported.
from .qt_compat import prepare_qt  # noqa: E402

prepare_qt()
