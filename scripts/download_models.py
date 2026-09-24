"""Download and prepare the ONNX models for the face-recognition project.

Models fetched:

  1. models/det_10g.onnx        SCRFD face detector (boxes + 5 landmarks), ~16 MB
  2. models/w600k_r50.onnx      ArcFace R50 recognizer (512-dim embedding), ~166 MB
  3. models/emotion-ferplus-8.onnx  FER+ expression classifier, ~35 MB

The first two are downloaded as direct ONNX files from a CDN mirror of the
InsightFace buffalo_l package, with the official archive as a fallback:

  https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip

The expression model comes from the ONNX Model Zoo (Git LFS media URL):

  https://github.com/onnx/models/tree/main/validated/vision/body_analysis/emotion_ferplus

Why this archive?
  * It is the official InsightFace release (the maintainers of SCRFD and ArcFace).
  * It contains the detector *and* the recognizer that InsightFace itself pairs,
    so their inputs/preprocessing are guaranteed to match (RGB, 112x112, etc.).
  * The InsightFace project is MIT licensed.

Usage:
  python -m scripts.download_models

The big model binaries are intentionally NOT committed to Git; run this once
after cloning to populate the models/ directory.
"""

from __future__ import annotations

import os
import shutil
import sys
import zipfile
import urllib.request

# Allow running as `python -m scripts.download_models` from the project root.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.config import (  # noqa: E402
    DETECTOR_MODEL_PATH,
    DETECTOR_MODEL_URL,
    EMBEDDER_MODEL_PATH,
    EMBEDDER_MODEL_URL,
    EXPRESSION_MODEL_PATH,
    EXPRESSION_MODEL_URL,
    MODELS_DIR,
)

_BUFFALO_L_URL = (
    "https://github.com/deepinsight/insightface/"
    "releases/download/v0.7/buffalo_l.zip"
)

# Files we need from inside the archive.
_REQUIRED_MEMBERS = {
    "det_10g.onnx": DETECTOR_MODEL_PATH,
    "w600k_r50.onnx": EMBEDDER_MODEL_PATH,
}
# Conservative lower bounds for the official v0.7 files. They prevent an
# interrupted direct download from being mistaken for a complete model.
_DETECTOR_MIN_SIZE = 8 * 1024 * 1024
_EMBEDDER_MIN_SIZE = 100 * 1024 * 1024


def _human(size: float) -> str:
    """Format a byte count for human-readable progress output."""
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def _download_file(url: str, destination: str, label: str) -> None:
    """Download one file, resuming a partial ``.part`` file when possible."""
    if os.path.exists(destination) and os.path.getsize(destination) > 0:
        return

    partial_path = destination + ".part"
    os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)
    offset = os.path.getsize(partial_path) if os.path.exists(partial_path) else 0
    headers = {
        "User-Agent": "face-recognition-educational",
        "Accept-Encoding": "identity",
    }
    if offset:
        headers["Range"] = f"bytes={offset}-"

    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=180) as response:
        status = getattr(response, "status", None) or 200
        # A server that ignores Range may return the whole file.  In that case
        # start over rather than concatenating a duplicate prefix.
        append = offset > 0 and status == 206
        if offset and not append:
            offset = 0

        total_header = response.headers.get("Content-Length", "0")
        try:
            total = int(total_header)
        except (TypeError, ValueError):
            total = 0
        content_range = response.headers.get("Content-Range", "")
        range_total = 0
        if "/" in content_range:
            try:
                range_total = int(content_range.rsplit("/", 1)[1])
            except (TypeError, ValueError):
                range_total = 0
        if range_total:
            total = range_total
        elif total and append:
            total += offset
        if total and getattr(response, "status", 200) == 200:
            total = int(total)

        downloaded = offset
        mode = "ab" if append else "wb"
        with open(partial_path, mode) as out:
            while True:
                chunk = response.read(1024 * 512)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = min(100.0, downloaded * 100.0 / total)
                    sys.stdout.write(
                        f"\r  {label}: {_human(downloaded)} / {_human(total)} "
                        f"({pct:.0f}%)"
                    )
                else:
                    sys.stdout.write(f"\r  {label}: {_human(downloaded)}")
                sys.stdout.flush()
        if not os.path.exists(partial_path) or os.path.getsize(partial_path) == 0:
            raise IOError(f"Downloaded file is empty: {url}")
        os.replace(partial_path, destination)
    print()


def _looks_like_model(path: str, minimum_size: int = 1024) -> bool:
    """Reject empty files and Git-LFS pointer files left by bad downloads."""
    if not os.path.isfile(path):
        return False
    try:
        if os.path.getsize(path) < minimum_size:
            return False
    except OSError:
        return False
    try:
        with open(path, "rb") as model:
            prefix = model.read(128)
        return not prefix.startswith(b"version https://git-lfs.github.com/spec")
    except OSError:
        return False


def main() -> int:
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Remove invalid/pointer files so the helper can fetch the real binaries.
    for path, minimum_size in (
        (DETECTOR_MODEL_PATH, _DETECTOR_MIN_SIZE),
        (EMBEDDER_MODEL_PATH, _EMBEDDER_MIN_SIZE),
        (EXPRESSION_MODEL_PATH, 30 * 1024 * 1024),
    ):
        if os.path.exists(path) and not _looks_like_model(path, minimum_size):
            try:
                os.remove(path)
            except OSError:
                pass

    identity_ready = _looks_like_model(DETECTOR_MODEL_PATH) and _looks_like_model(
        EMBEDDER_MODEL_PATH
    )
    expression_ready = _looks_like_model(EXPRESSION_MODEL_PATH, 30 * 1024 * 1024)

    try:
        if not identity_ready:
            # Prefer the two direct ONNX files. They avoid downloading the
            # 289 MB release archive and are served by a CDN. If either mirror
            # is unavailable, fall back to the official archive below.
            direct_error = None
            for url, destination, label in (
                (DETECTOR_MODEL_URL, DETECTOR_MODEL_PATH, "SCRFD detector"),
                (EMBEDDER_MODEL_URL, EMBEDDER_MODEL_PATH, "ArcFace recognizer"),
            ):
                if _looks_like_model(destination):
                    continue
                try:
                    print(f"Downloading {label}...")
                    _download_file(url, destination, label)
                except Exception as exc:
                    direct_error = exc
                    print(f"Mirror download failed for {label}: {exc}", file=sys.stderr)

            identity_ready = _looks_like_model(DETECTOR_MODEL_PATH) and _looks_like_model(
                EMBEDDER_MODEL_PATH
            )
            if identity_ready:
                print("InsightFace detector + ArcFace models downloaded from mirror.")
            else:
                if direct_error is not None:
                    print("Falling back to the official buffalo_l archive...")
                archive_path = os.path.join(MODELS_DIR, "buffalo_l.zip")
                archive_partial_path = archive_path + ".part"
                # Older versions wrote directly to buffalo_l.zip. If that file
                # is not a valid archive, move it to the helper's .part path so
                # the download can resume it safely.
                if os.path.exists(archive_path) and not zipfile.is_zipfile(archive_path):
                    if not os.path.exists(archive_partial_path):
                        os.replace(archive_path, archive_partial_path)
                    else:
                        os.remove(archive_path)
                _download_file(_BUFFALO_L_URL, archive_path, "InsightFace models")

                print("Extracting required ONNX files...")
                with zipfile.ZipFile(archive_path) as zf:
                    names = set(zf.namelist())
                    missing = [name for name in _REQUIRED_MEMBERS if name not in names]
                    if missing:
                        print(f"ERROR: archive does not contain: {missing}", file=sys.stderr)
                        return 1
                    for name, dest in _REQUIRED_MEMBERS.items():
                        print(f"  {name} -> {dest}")
                        with zf.open(name) as src, open(dest, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                os.remove(archive_path)
        else:
            print("InsightFace detector + ArcFace models already exist.")

        if not expression_ready:
            print("Downloading the FER+ facial-expression model...")
            _download_file(
                EXPRESSION_MODEL_URL,
                EXPRESSION_MODEL_PATH,
                "FER+ expression model",
            )
            if not _looks_like_model(EXPRESSION_MODEL_PATH, 30 * 1024 * 1024):
                raise IOError(
                    "FER+ download finished but the file is incomplete; run the "
                    "downloader again"
                )
        else:
            print("FER+ facial-expression model already exists.")

        print("Done. Models are ready in models/.")
        print(f"  {DETECTOR_MODEL_PATH}")
        print(f"  {EMBEDDER_MODEL_PATH}")
        print(f"  {EXPRESSION_MODEL_PATH}")
        return 0
    except Exception as exc:  # network / disk / zip errors -> clean message
        print(f"ERROR: could not download models: {exc}", file=sys.stderr)
        print(
            "Check your internet connection and try again. The downloader resumes "
            "partial .part files. You can also place det_10g.onnx, w600k_r50.onnx "
            "and emotion-ferplus-8.onnx in models/ manually.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())