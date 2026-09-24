"""Central configuration for the face-recognition project.

Everything that defines *how* the pipeline behaves lives here so that a
student can tune it without digging through the implementation:

  * where models live and what they are called
  * the input size / dimension the ArcFace model expects
  * the matching threshold (Known vs Unknown boundary)
  * where enrollment images come from / where embeddings are stored

Every value can be overridden with an environment variable of the same name,
which is convenient for demos (e.g. FR_THRESHOLD=0.5 python -m scripts.recognize).
"""

from __future__ import annotations

import os

# --------------------------------------------------------------------------
# Filesystem layout (relative to the project root).
# --------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
FACES_DIR = os.path.join(DATA_DIR, "faces")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

# Where webcam photos/videos are saved ("s" and "r" keys in scripts/recognize).
# Defaults to the user's Downloads folder; falls back to outputs/ if missing.
DOWNLOADS_DIR = os.path.expanduser("~/Downloads")
if not os.path.isdir(DOWNLOADS_DIR):
    DOWNLOADS_DIR = OUTPUTS_DIR

# --------------------------------------------------------------------------
# ONNX models.
# --------------------------------------------------------------------------
# face detector: SCRFD (from the official InsightFace buffalo_l package)
DETECTOR_MODEL_PATH = os.path.join(MODELS_DIR, "det_10g.onnx")
# recognizer: ArcFace ResNet-50 (from the official InsightFace buffalo_l package)
EMBEDDER_MODEL_PATH = os.path.join(MODELS_DIR, "w600k_r50.onnx")
# expression classifier: FER+ (ONNX Model Zoo, 8 emotion logits)
EXPRESSION_MODEL_PATH = os.path.join(MODELS_DIR, "emotion-ferplus-8.onnx")

# Size the detector passes into the network (width, height).
DETECTOR_INPUT_SIZE = tuple(
    int(x) for x in os.environ.get("FR_DETECTOR_INPUT_SIZE", "640,640").split(",")
)
# Minimum detection confidence (after the detector's internal sigmoid).
DETECTOR_CONFIDENCE = float(os.environ.get("FR_DETECTOR_CONFIDENCE", "0.5"))
# IoU cutoff for non-maximum suppression of overlapping boxes.
DETECTOR_NMS = float(os.environ.get("FR_DETECTOR_NMS", "0.4"))

# The ArcFace recognition head expects crops of this size (height, width).
EMBEDDER_INPUT_IMAGE_SIZE = (112, 112)
# Expected embedding length. w600k_r50 produces 512 floats.
EMBEDDING_DIMENSION = 512

# FER+ expression model settings.  FER+ emits eight probabilities; the
# wrapper maps happiness to smile/laugh and anger to angry.  A missing model
# does not stop identity recognition, but expression fields will be None.
EXPRESSION_INPUT_SIZE = (64, 64)
EXPRESSION_CONFIDENCE = float(os.environ.get("FR_EXPRESSION_CONFIDENCE", "0.40"))
# FER+ is conservative about anger on webcam crops.  Use a slightly lower,
# dedicated threshold while still requiring anger to be the model's top class.
EXPRESSION_ANGER_THRESHOLD = float(
    os.environ.get("FR_EXPRESSION_ANGER_THRESHOLD", "0.25")
)
EXPRESSION_MOUTH_OPEN_THRESHOLD = float(
    os.environ.get("FR_EXPRESSION_MOUTH_OPEN_THRESHOLD", "0.45")
)
EXPRESSION_LAUGH_FRAMES = int(os.environ.get("FR_EXPRESSION_LAUGH_FRAMES", "3"))
# The ONNX Model Zoo stores large files through Git LFS.  media.githubusercontent
# serves the actual binary; the ordinary raw URL returns a small LFS pointer.
EXPRESSION_MODEL_URL = (
    "https://media.githubusercontent.com/media/onnx/models/main/"
    "validated/vision/body_analysis/emotion_ferplus/model/"
    "emotion-ferplus-8.onnx"
)
# Direct ONNX files mirrored from the official InsightFace buffalo_l package.
# They are much smaller/faster to download than the 289 MB release archive; the
# downloader still falls back to the official archive if a mirror is unavailable.
DETECTOR_MODEL_URL = (
    "https://huggingface.co/DavidHoa/buffalo_l/resolve/main/"
    "det_10g.onnx?download=true"
)
EMBEDDER_MODEL_URL = (
    "https://huggingface.co/DavidHoa/buffalo_l/resolve/main/"
    "w600k_r50.onnx?download=true"
)
EXPRESSION_ENABLED = os.environ.get("FR_EXPRESSIONS", "1").lower() not in {
    "0",
    "false",
    "no",
    "off",
}

# --------------------------------------------------------------------------
# Enrollment data.
# --------------------------------------------------------------------------
# Where per-person enrollment photos live: data/faces/<person>/*.jpg
# Where computed embeddings (a single .npz + a meta .json) are written.
EMBEDDINGS_NPZ_PATH = os.path.join(EMBEDDINGS_DIR, "embeddings.npz")
EMBEDDINGS_META_PATH = os.path.join(EMBEDDINGS_DIR, "meta.json")

# How multiple images of one person are combined into what we match against.
#   "all"  -> keep every embedding and match against each, reporting the best;
#             robust to pose/lighting variation (recommended default).
#   "mean" -> store one averaged (L2-normalized) representative embedding;
#             compact but less robust to variation.
MULTI_EMBEDDING_STRATEGY = os.environ.get("FR_STRATEGY", "all").lower()

# --------------------------------------------------------------------------
# Matching.
# --------------------------------------------------------------------------
# Cosine-similarity threshold: score >= threshold -> "Known", below -> "Unknown".
# There is no universal value; it must be validated on YOUR enrolled data.
# A reasonable starting point for ArcFace cosine similarity is 0.4 - 0.5.
MATCHING_THRESHOLD = float(os.environ.get("FR_THRESHOLD", "0.40"))

# Accepted image extensions when scanning data/faces/.
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def env(name: str, default: str) -> str:
    return os.environ.get(name, default)