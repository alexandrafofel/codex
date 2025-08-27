"""PDF rasterisation utilities.

This module handles extraction of raster images from a source PDF.
Where possible we use the high‑performance PyMuPDF (fitz) backend to
render pages.  If PyMuPDF is unavailable or fails for a given page
we fall back to the ``pdf2image`` library which relies on Poppler.

Pages are rendered at a configurable DPI (dots per inch) defined in
the :class:`ocrmd.config.OcrConfig`.  Output images are stored under
``output/<slug>/raw_pages/`` with sequentially numbered file names.
JPEG is used for photographic pages while PNG is used for pages that
consist purely of vector text.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np

from PIL import Image

try:
    import fitz  # type: ignore
except ImportError:
    fitz = None  # type: ignore

try:
    from pdf2image import convert_from_path  # type: ignore
except ImportError:
    convert_from_path = None  # type: ignore

from .config import OcrConfig

logger = logging.getLogger(__name__)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _slugify(name: str) -> str:
    # simple slug generator: lowercase, replace spaces with underscores and drop non alnum
    import re
    name = name.lower().strip().replace(" ", "_")
    name = re.sub(r"[^a-z0-9_]+", "", name)
    return name or "document"


def extract_pages(config: OcrConfig, work_dir: Path) -> List[Path]:
    """Render all pages of the configured PDF to individual image files.

    Parameters
    ----------
    config:
        Parsed configuration describing the book and processing options.
    work_dir:
        Base output directory; generated images will be saved under
        ``work_dir/raw_pages/``.

    Returns
    -------
    list[Path]
        A list of file paths to the rendered page images, in order.
    """
    # Determine slug for output subdirectory
    slug = _slugify(config.book_title) if config.book_title else _slugify(config.pdf_file.stem)
    out_dir = work_dir / slug / "raw_pages"
    _ensure_dir(out_dir)
    paths: List[Path] = []

    pdf_path = config.pdf_file
    dpi = config.dpi

    logger.info("Extracting pages from %s at %s DPI", pdf_path, dpi)

    if fitz is not None:
        try:
            doc = fitz.open(pdf_path)
            for i in range(len(doc)):
                page = doc[i]
                # scale matrix: DPI/72 since PyMuPDF uses 72 DPI baseline
                zoom = dpi / 72.0
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                mode = "RGB" if pix.n > 1 else "L"
                img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
                # Determine format: PNG for pure text pages (single channel) else JPEG
                fmt = "PNG" if mode == "L" else "JPEG"
                suffix = ".png" if fmt == "PNG" else ".jpg"
                out_path = out_dir / f"scan_{i+1:03d}{suffix}"
                img.save(out_path, fmt, quality=90)
                paths.append(out_path)
            return paths
        except Exception as exc:
            logger.warning("PyMuPDF failed: %s, falling back to pdf2image", exc)

    # Fallback using pdf2image
    if convert_from_path is None:
        raise RuntimeError("Neither PyMuPDF nor pdf2image are available for PDF rasterisation")
    pages = convert_from_path(str(pdf_path), dpi=dpi)
    for idx, img in enumerate(pages, start=1):
        # convert PIL image to JPEG/PNG depending on colour channels
        if img.mode == "RGB":
            fmt = "JPEG"
            suffix = ".jpg"
        else:
            fmt = "PNG"
            suffix = ".png"
        out_path = out_dir / f"scan_{idx:03d}{suffix}"
        img.save(out_path, fmt, quality=90)
        paths.append(out_path)
    return paths