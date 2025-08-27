"""Heading detection and structural analysis of OCR output.

Tesseract provides positional information for each recognised word via
``image_to_data``.  This module analyses the bounding boxes of lines
to detect headings and assign hierarchy levels (H1/H2/H3).  The
heuristic is based on relative font size (approximated by line
height), centering and position within the page.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pytesseract  # type: ignore
from PIL import Image


@dataclass
class LineInfo:
    text: str
    left: int
    top: int
    width: int
    height: int
    conf: float


def _group_lines(data: Dict[str, List[str]]) -> List[LineInfo]:
    """Group word-level OCR data into lines and compute bounding boxes."""
    num_words = len(data["text"])
    lines: Dict[int, List[int]] = {}
    for i in range(num_words):
        line_num = int(data["line_num"][i])
        if line_num not in lines:
            lines[line_num] = []
        lines[line_num].append(i)
    result: List[LineInfo] = []
    for line_idx, word_indices in lines.items():
        texts = [data["text"][i] for i in word_indices]
        if not any(t.strip() for t in texts):
            continue  # skip empty lines
        lefts = [int(data["left"][i]) for i in word_indices]
        tops = [int(data["top"][i]) for i in word_indices]
        rights = [int(data["left"][i]) + int(data["width"][i]) for i in word_indices]
        bottoms = [int(data["top"][i]) + int(data["height"][i]) for i in word_indices]
        heights = [int(data["height"][i]) for i in word_indices]
        confs = [float(data["conf"][i]) for i in word_indices if data["conf"][i] != "-1"]
        line_text = " ".join(texts).strip()
        result.append(
            LineInfo(
                text=line_text,
                left=min(lefts),
                top=min(tops),
                width=max(rights) - min(lefts),
                height=int(statistics.mean(heights)),
                conf=float(np.mean(confs)) if confs else 0.0,
            )
        )
    return result


def detect_headings(img: Image.Image, lang: str = "eng", base_psm: int = 6) -> List[Tuple[str, str]]:
    """Detect headings from an image using Tesseract metadata.

    Returns a list of tuples ``(tag, text)`` where tag is one of
    ``"H1"``, ``"H2"``, ``"H3"`` or ``"P"`` (paragraph).  The ordering
    follows the natural reading order of the lines.
    """
    # run tesseract in PSM 6 (block of text) to get bounding boxes
    data = pytesseract.image_to_data(img, lang=lang, config=f"--psm {base_psm}", output_type=pytesseract.Output.DICT)
    lines = _group_lines(data)
    if not lines:
        return []
    # compute thresholds
    heights = [ln.height for ln in lines]
    median_h = statistics.median(heights)
    # candidate headings are lines with height >= 1.4 * median
    candidates = [ln for ln in lines if ln.height >= 1.4 * median_h]
    # sort candidates by top position
    candidates.sort(key=lambda ln: ln.top)
    tags: List[Tuple[str, str]] = []
    used = set()
    # determine page height to classify H1 near top
    page_h = img.size[1]
    for ln in lines:
        tag = "P"  # default paragraph
        if ln in candidates:
            idx = candidates.index(ln)
            if ln.top < 0.3 * page_h and idx == 0:
                tag = "H1"
            elif idx == 0:
                tag = "H2"
            else:
                tag = "H3"
        tags.append((tag, ln.text))
    return tags