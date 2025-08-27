"""Optical character recognition using Tesseract.

This module wraps pytesseract to perform OCR on preprocessed page
images.  It runs Tesseract using multiple page segmentation modes
(PSM) and selects the result with the highest average confidence.
If no OCR result meets minimum quality criteria (minimum number of
characters or average confidence), callers may choose to embed the
page image instead of including unreliable text in the Markdown
output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pytesseract  # type: ignore
from PIL import Image

from .config import OcrConfig

logger = logging.getLogger(__name__)


@dataclass
class OcrResult:
    text: str
    avg_conf: float
    used_psm: int
    num_chars: int

    def is_low_quality(self, config: OcrConfig) -> bool:
        return self.avg_conf < config.low_conf or self.num_chars < config.min_chars


def _run_psm(img: Image.Image, lang: str, psm: int) -> Tuple[str, float, int, int]:
    """Run Tesseract for a single page segmentation mode and return metrics."""
    config_str = f"--psm {psm}"
    data = pytesseract.image_to_data(img, lang=lang, config=config_str, output_type=pytesseract.Output.DICT)
    # extract confidences excluding -1
    confs = [int(c) for c in data.get("conf", []) if c != "-1"]
    avg_conf = float(np.mean(confs)) if confs else 0.0
    text = pytesseract.image_to_string(img, lang=lang, config=config_str)
    num_chars = len(text.replace("\n", "").strip())
    return text, avg_conf, psm, num_chars


def run_ocr(img: Image.Image, config: OcrConfig) -> OcrResult:
    """Perform OCR on a single image using multiple PSMs and choose the best result.

    The order of PSMs tested is `[config.tess_psm, 6, 3, 4]` by default.  Duplicates
    are removed.  The result with the highest average confidence is returned;
    ties are broken by choosing the text with the most characters.
    """
    psms = [config.tess_psm, 6, 3, 4]
    # remove duplicates while preserving order
    seen = set()
    unique_psms: List[int] = []
    for p in psms:
        if p not in seen:
            unique_psms.append(p)
            seen.add(p)
    results: List[OcrResult] = []
    for psm in unique_psms:
        try:
            text, avg_conf, used_psm, num_chars = _run_psm(img, config.lang, psm)
            results.append(OcrResult(text, avg_conf, used_psm, num_chars))
        except Exception as exc:
            logger.warning("Tesseract failed for PSM %d: %s", psm, exc)
    # pick result with highest confidence; tie break by number of characters
    if not results:
        return OcrResult("", 0.0, config.tess_psm, 0)
    results.sort(key=lambda r: (r.avg_conf, r.num_chars), reverse=True)
    return results[0]