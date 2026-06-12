"""Shared English-text detection: EAST detector + EasyOCR confirmation.

Extracted from scan_text_fast.py so the post-render checker
(scripts/check_render_text.py) reuses the exact same two-pass machinery and
tuning. Import surface:

    detect_text_east(image_path)  -> (has_text, n_regions, elapsed)
    verify_text_ocr(image_path)   -> (has_english_text, description, elapsed)
    extract_frame(video, sec, out) -> bool
    get_east_net() / get_ocr_reader() — warm the cached singletons up front
"""

import subprocess
import time
import urllib.request
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", message=".*pin_memory.*")

import cv2
import numpy as np

EAST_MODEL_URL = "https://github.com/oyyd/frozen_east_text_detection.pb/raw/master/frozen_east_text_detection.pb"
EAST_MODEL_PATH = Path.home() / ".cache" / "east_text_detection.pb"
EAST_CONFIDENCE = 0.7   # minimum confidence for EAST text detection (0.5 too noisy on abstract visuals)
EAST_NMS_THRESHOLD = 0.4
EAST_INPUT_SIZE = (320, 320)  # EAST requires multiples of 32

MIN_OCR_WORD_LEN = 3        # ignore OCR results shorter than this
MIN_OCR_CONFIDENCE = 0.3    # ignore low-confidence OCR detections

_east_net = None
_ocr_reader = None


def _ensure_east_model():
    """Download EAST model if not cached."""
    if EAST_MODEL_PATH.exists():
        return
    EAST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"    Downloading EAST text detector...", end=" ", flush=True)
    urllib.request.urlretrieve(EAST_MODEL_URL, str(EAST_MODEL_PATH))
    print(f"done ({EAST_MODEL_PATH.stat().st_size / 1e6:.1f} MB)")


def get_east_net():
    """Load EAST model (cached singleton)."""
    global _east_net
    if _east_net is None:
        _ensure_east_model()
        _east_net = cv2.dnn.readNet(str(EAST_MODEL_PATH))
    return _east_net


def detect_text_east(image_path):
    """Detect text regions using EAST. Returns (has_text, num_regions, elapsed)."""
    t0 = time.time()
    img = cv2.imread(str(image_path))
    if img is None:
        return None, 0, time.time() - t0

    blob = cv2.dnn.blobFromImage(img, 1.0, EAST_INPUT_SIZE,
                                 (123.68, 116.78, 103.94), True, False)

    net = get_east_net()
    net.setInput(blob)
    scores, geometry = net.forward([
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"
    ])

    rows = scores.shape[2]
    cols = scores.shape[3]

    rects = []
    confidences = []

    for y in range(rows):
        scores_data = scores[0, 0, y]
        x0 = geometry[0, 0, y]
        x1 = geometry[0, 1, y]
        x2 = geometry[0, 2, y]
        x3 = geometry[0, 3, y]
        angles = geometry[0, 4, y]

        for x in range(cols):
            if scores_data[x] < EAST_CONFIDENCE:
                continue

            offset_x = x * 4.0
            offset_y = y * 4.0
            angle = angles[x]
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)

            h_box = x0[x] + x2[x]
            w_box = x1[x] + x3[x]

            end_x = int(offset_x + cos_a * x1[x] + sin_a * x2[x])
            end_y = int(offset_y - sin_a * x1[x] + cos_a * x2[x])
            start_x = int(end_x - w_box)
            start_y = int(end_y - h_box)

            rects.append((start_x, start_y, end_x, end_y))
            confidences.append(float(scores_data[x]))

    if not rects:
        return False, 0, time.time() - t0

    boxes_for_nms = [[r[0], r[1], r[2] - r[0], r[3] - r[1]] for r in rects]
    indices = cv2.dnn.NMSBoxes(boxes_for_nms, confidences,
                               EAST_CONFIDENCE, EAST_NMS_THRESHOLD)
    n_regions = len(indices) if len(indices) > 0 else 0

    elapsed = time.time() - t0
    return n_regions > 0, n_regions, elapsed


def get_ocr_reader():
    """Load EasyOCR reader (cached singleton)."""
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        _ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    return _ocr_reader


def verify_text_ocr(image_path):
    """Run EasyOCR on a frame. Returns (has_english_text, description, elapsed)."""
    t0 = time.time()
    reader = get_ocr_reader()
    results = reader.readtext(str(image_path), detail=1)

    # Filter: keep only results with readable English words (not just numbers/symbols)
    words = []
    for bbox, text, conf in results:
        text = text.strip()
        if conf >= MIN_OCR_CONFIDENCE and len(text) >= MIN_OCR_WORD_LEN:
            # Skip purely numeric/binary/symbolic strings
            alpha_chars = sum(1 for c in text if c.isalpha())
            if alpha_chars >= 2:
                words.append(text)

    elapsed = time.time() - t0
    if words:
        desc = ", ".join(words[:10])
        return True, desc, elapsed
    return False, "", elapsed


def extract_frame(video_path, time_sec, output_path):
    if output_path.exists() and output_path.stat().st_size > 0:
        return True
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(time_sec), "-i", str(video_path),
             "-frames:v", "1", "-q:v", "2", str(output_path)],
            capture_output=True, timeout=30
        )
        return output_path.exists() and output_path.stat().st_size > 0
    except Exception:
        return False
