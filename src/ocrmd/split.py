"""Split composite scans into individual pages.

Many scanned books store two or even four logical pages on a single
physical scan.  This module analyses each rasterised page and splits it
into individual images containing one logical page each.  A simple
projection profile method is used to detect the number and location of
vertical gutters.  Users can override the automatic detection via the
``split_strategy`` section of the configuration.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2  # type: ignore
import numpy as np
from PIL import Image

from .config import OcrConfig, SplitRule

logger = logging.getLogger(__name__)


def _parse_page_range(spec: str) -> Iterable[int]:
    """Expand a page specifier into a sequence of page numbers.

    The ``page`` field in the YAML can be either a single integer or a
    string of the form ``"start-end"``.  This helper returns a list of
    zero‑based page indices covered by the specifier.
    """
    if isinstance(spec, int):
        yield spec - 1
        return
    m = re.match(r"(\d+)-(\d+)", spec)
    if m:
        start, end = int(m.group(1)), int(m.group(2))
        for p in range(start, end + 1):
            yield p - 1
    else:
        yield int(spec) - 1


def _build_override_map(rules: Iterable[SplitRule]) -> Dict[int, str]:
    """Create a mapping from zero‑based page indices to split types.

    This mapping is used to override the automatic split detection for
    specified pages.
    """
    override: Dict[int, str] = {}
    for rule in rules:
        for idx in _parse_page_range(rule.page):
            override[idx] = rule.type
    return override


def _detect_split_type(img: Image.Image) -> str:
    """Heuristic detection of the number of pages in a scan.

    Convert the image to grayscale, compute a vertical projection profile
    by summing pixel intensities along the height axis, and look for
    significant minima which suggest gutters between pages.  The
    thresholds and heuristics are deliberately conservative; if no
    confident minima are found, the function returns ``"single"``.
    """
    arr = np.array(img.convert("L"))
    # sum pixel values column‑wise (dark gutters give lower sums)
    profile = np.sum(arr, axis=0)
    # normalise
    profile = profile / profile.max()
    # compute smoothed derivative to find valleys
    smoothed = cv2.GaussianBlur(profile.astype(np.float32), (51, 1), 0)
    # find minima by thresholding below some fraction of the median
    threshold = np.median(smoothed) * 0.6
    minima = np.where(smoothed < threshold)[0]
    # cluster minima into contiguous regions
    splits: List[int] = []
    if minima.size > 0:
        groups: List[List[int]] = []
        current: List[int] = [minima[0]]
        for idx in minima[1:]:
            if idx - current[-1] <= 5:
                current.append(idx)
            else:
                groups.append(current)
                current = [idx]
        groups.append(current)
        # pick representative minima positions
        splits = [int(np.mean(group)) for group in groups]
    # decide based on number of minima
    if len(splits) >= 2:
        return "quadruple"
    if len(splits) == 1:
        return "double"
    return "single"


def _crop_margins(img: Image.Image, crop_pct: float) -> Image.Image:
    """Crop a percentage of the margins from all four sides of the image."""
    w, h = img.size
    dx = int(w * crop_pct)
    dy = int(h * crop_pct)
    return img.crop((dx, dy, w - dx, h - dy))


def split_pages(config: OcrConfig, input_paths: List[Path], work_dir: Path) -> List[Path]:
    """Split composite page scans into individual page images.

    Parameters
    ----------
    config:
        Parsed configuration containing split rules and crop percentage.
    input_paths:
        Sequence of file paths to the rasterised page images.
    work_dir:
        Base output directory; split pages will be stored under
        ``work_dir/<slug>/split_pages``.

    Returns
    -------
    list[Path]
        A list of file paths to the resulting individual pages in
        processing order.
    """
    slug = (config.book_title.lower().replace(" ", "_") if config.book_title else config.pdf_file.stem)
    out_dir = work_dir / slug / "split_pages"
    out_dir.mkdir(parents=True, exist_ok=True)

    override_map = _build_override_map(config.split_strategy)

    result: List[Path] = []
    page_counter = 1
    for idx, path in enumerate(input_paths):
        img = Image.open(path)
        split_type = override_map.get(idx, _detect_split_type(img))
        logger.debug("Page %d detected split type %s", idx + 1, split_type)

        # optionally crop margins before splitting
        img = _crop_margins(img, config.crop_pct)
        w, h = img.size
        if split_type == "single":
            out_path = out_dir / f"page_{page_counter:03d}.jpg"
            img.save(out_path, "JPEG", quality=90)
            result.append(out_path)
            page_counter += 1
        elif split_type == "double":
            mid = w // 2
            left = img.crop((0, 0, mid, h))
            right = img.crop((mid, 0, w, h))
            for part in [left, right]:
                out_path = out_dir / f"page_{page_counter:03d}.jpg"
                part.save(out_path, "JPEG", quality=90)
                result.append(out_path)
                page_counter += 1
        else:  # quadruple: split into quadrants
            mid_x = w // 2
            mid_y = h // 2
            quadrants = [
                img.crop((0, 0, mid_x, mid_y)),
                img.crop((mid_x, 0, w, mid_y)),
                img.crop((0, mid_y, mid_x, h)),
                img.crop((mid_x, mid_y, w, h)),
            ]
            for part in quadrants:
                out_path = out_dir / f"page_{page_counter:03d}.jpg"
                part.save(out_path, "JPEG", quality=90)
                result.append(out_path)
                page_counter += 1
    return result