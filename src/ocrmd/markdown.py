"""Markdown builder for OCR results.

This module constructs a single Markdown document from a sequence of
processed pages.  Each page section contains an optional embedded
image of the page along with the recognised text split into
headings and paragraphs.  The embedding behaviour is controlled via
the ``embed_images`` parameter in the configuration.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Iterable, List

from PIL import Image

from .config import OcrConfig
from .ocr import OcrResult, run_ocr
from .preprocess import preprocess
from .structure import detect_headings


def _should_embed(page_idx: int, ocr_result: OcrResult, config: OcrConfig, total_pages: int) -> bool:
    """Decide whether to embed the original image for this page."""
    if config.embed_images == "all":
        return True
    if config.embed_images == "none":
        return False
    # auto: embed the first page (cover), pages flagged as image heavy or low confidence
    if page_idx == 0:
        return True
    return ocr_result.is_low_quality(config)


def _clean_text(text: str) -> str:
    """Simple cleanup of OCR text (dehyphenation, normalise spaces)."""
    # remove hyphenation across line breaks
    text = text.replace("-\n", "")
    # normalise multiple spaces/newlines
    text = "\n".join([line.strip() for line in text.splitlines()])
    text = "\n".join([ln for ln in text.splitlines() if ln.strip()])
    return text


def build_markdown(config: OcrConfig, pages: Iterable[Path], work_dir: Path) -> Path:
    """Build a Markdown document from OCR results.

    Parameters
    ----------
    config:
        The configuration instance controlling behaviour.
    pages:
        An iterable of file paths to the individual page images to OCR.
    work_dir:
        Base output directory where the Markdown file will be stored.

    Returns
    -------
    Path
        Path to the generated Markdown document.
    """
    slug = (config.book_title.lower().replace(" ", "_") if config.book_title else config.pdf_file.stem)
    md_dir = work_dir / slug / "markdown"
    md_dir.mkdir(parents=True, exist_ok=True)
    md_path = md_dir / f"{slug}.md"

    lines: List[str] = []
    # front matter
    lines.append(f"# {config.book_title}\n")
    meta = {
        "Language": config.lang,
        "DPI": config.dpi,
        "PSM": config.tess_psm,
        "Preprocess": config.preprocess,
        "Generated": _dt.datetime.now().isoformat(),
    }
    for k, v in meta.items():
        lines.append(f"**{k}:** {v}  ")
    lines.append("")

    total_pages = len(list(pages))
    for i, page_path in enumerate(pages, start=1):
        img = Image.open(page_path)
        pre = preprocess(img, config.preprocess, crop_pct=0.0)
        ocr_result = run_ocr(pre, config)
        embed = _should_embed(i - 1, ocr_result, config, total_pages)
        # page heading
        lines.append(f"\n## Page {i}\n")
        # embed image if required
        if embed:
            # store relative path from md file
            rel_path = page_path.relative_to(md_dir.parent)
            lines.append(f"![page {i}]({rel_path.as_posix()})\n")
        # convert OCR to structure
        headings = detect_headings(pre, lang=config.lang, base_psm=config.tess_psm)
        if headings:
            for tag, text in headings:
                if tag == "H1":
                    lines.append(f"# {text}\n")
                elif tag == "H2":
                    lines.append(f"## {text}\n")
                elif tag == "H3":
                    lines.append(f"### {text}\n")
                else:
                    lines.append(f"{text}\n")
        else:
            cleaned = _clean_text(ocr_result.text)
            lines.append(f"{cleaned}\n")

    md_content = "\n".join(lines)
    md_path.write_text(md_content, encoding="utf-8")
    return md_path