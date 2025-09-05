"""Markdown builder for OCR results.

This module constructs a single Markdown document from a sequence of
processed pages.  Each page section contains an optional embedded
image of the page along with the recognised text split into
headings and paragraphs.  The embedding behaviour is controlled via
the ``embed_images`` parameter in the configuration.
"""

from __future__ import annotations
from .preprocess import preprocess
from PIL import Image

import datetime as _dt
from pathlib import Path
from typing import Iterable, List

from PIL import Image

from .config import OcrConfig
from .ocr import OcrResult, run_ocr
from .preprocess import preprocess
from .structure import detect_headings
import re
import re
import yaml
from pathlib import Path

_REPL_CACHE: dict | None = None

def _load_yaml_replacements(cfg) -> dict:
    """Citește mapările din configs/ocr_fixes.yaml (dacă există) și le cache-uiește."""
    global _REPL_CACHE
    if _REPL_CACHE is not None:
        return _REPL_CACHE
    # caută lângă fișierul de config sau în ./configs
    # dacă cfg are un atribut 'config_path', îl poți folosi; altfel mergem pe fallback
    candidates = [
        Path("configs/ocr_fixes.yaml"),
        Path.cwd() / "configs" / "ocr_fixes.yaml",
    ]
    for p in candidates:
        if p.exists():
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            _REPL_CACHE = dict(data.get("replacements", {}))
            return _REPL_CACHE
    _REPL_CACHE = {}
    return _REPL_CACHE

def _apply_yaml_replacements(s: str, cfg) -> str:
    repl = _load_yaml_replacements(cfg)
    if not repl:
        return s
    for wrong, right in repl.items():
        s = s.replace(wrong, right)
    return s

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

# --- NEW/UPDATED helpers for slug and dirs ---
from pathlib import Path
from dataclasses import dataclass

def _slug_from_cfg(cfg) -> str:
    import re
    title = (cfg.book_title or "").strip()
    if title:
        s = re.sub(r"[^a-zA-Z0-9]+", "_", title).strip("_").lower()
        return s or "book"
    return Path(cfg.pdf_file).stem

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

@dataclass
class PageCtx:
    index: int
    img_path: Path
    text: str
    avg_conf: float
    chars: int
    split_tag: str = "1x1"

def _allow_text(avg_conf: float, chars: int, cfg) -> bool:
    return (avg_conf >= float(cfg.low_conf)) and (chars >= int(cfg.min_chars))

def cleanup_text(text: str, cfg) -> str:
    """Heuristici simple pentru erori OCR frecvente + mapări din YAML."""
    import re

    s = text.replace("\r\n", "\n").replace("\r", "\n")

    # 0) Normalize whitespace de bază
    s = re.sub(r"[ \t]+", " ", s)

    # 1) De-hyphen cross-line: "word-\nnext"
    def _join_broken(m):
        left, right = m.group(1), m.group(2)
        if min(len(left), len(right)) <= 4:
            return left + right   # expe-\nrience -> experience
        return left + "-" + right # ground-\nbreaking -> ground-breaking
    s = re.sub(r"([A-Za-z]{2,})-\n([A-Za-z]{2,})", _join_broken, s)

    # 2) De-hyphen intra-line: "cul-tures" -> "cultures", păstrăm compusele lungi
    def _dehyphen_intra(m):
        left, right = m.group(1), m.group(2)
        if min(len(left), len(right)) <= 4:
            return left + right
        return left + "-" + right
    s = re.sub(r"([A-Za-z]{2,})-([A-Za-z]{2,})", _dehyphen_intra, s)

    # 3) Elimină dublări de cuvinte scurte (to/in/of/for/and/the)
    s = re.sub(r"\b(to|in|of|for|and|the)\s+\1\b", r"\1", s, flags=re.IGNORECASE)

    # 4) (opțional) mici corecții generice observate frecvent
    fixes = {
        " Ctwo-": " to two-",
        "Ctwo-": "to two-",
        " cach ": " each ",
        " cach\n": " each\n",
        "gr« yund": "ground",
        "Gr« yund": "Ground",
        "httn": "http",
        "archive orn": "archive.org",
        "Archive orn": "Archive.org",
    }
    for wrong, right in fixes.items():
        s = s.replace(wrong, right)

    # 5) Aplică mapările configurabile din YAML (configs/ocr_fixes.yaml)
    s = _apply_yaml_replacements(s, cfg)

    # 6) Curățări finale
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def render_text_only(cfg, pages: list[PageCtx], work_dir: Path) -> Path:
    slug = _slug_from_cfg(cfg)
    md_dir = work_dir / slug / "markdown"
    _ensure_dir(md_dir)
    out_md = md_dir / f"{slug}_TEXT.md"

    lines = [f"# {cfg.book_title or slug}", ""]
    lines.append(f"_lang={cfg.lang}, dpi={cfg.dpi}, preprocess={cfg.preprocess}, psm={cfg.tess_psm}_")
    lines.append("")

    for p in pages:
        lines.append(f"## Page {p.index}")
        if _allow_text(p.avg_conf, p.chars, cfg) and p.text.strip():
            lines.append(cleanup_text(p.text, cfg))
        else:
            lines.append("_(no text – filtered by quality thresholds)_")
        lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    return out_md

def render_with_images(cfg, pages: list[PageCtx], work_dir: Path) -> Path:
    import shutil
    slug = _slug_from_cfg(cfg)
    md_dir = work_dir / slug / "markdown"
    img_dir = md_dir / "img"
    _ensure_dir(md_dir)
    _ensure_dir(img_dir)
    out_md = md_dir / f"{slug}_IMAGES.md"

    lines = [f"# {cfg.book_title or slug} — IMAGES", "", "_images-only export_", ""]

    for p in pages:
        lines.append(f"## Page {p.index}")
        img_name = f"page_{p.index:03d}_{p.split_tag}.jpg"
        dst = img_dir / img_name
        try:
            shutil.copyfile(p.img_path, dst)
            lines.append(f"![page {p.index}](img/{img_name})")
        except Exception as e:
            lines.append(f"_image missing: {p.img_path} ({e})_")
        lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    return out_md

# --- UPDATE build_markdown to use new functions ---
def build_markdown(cfg, page_paths: list[Path], work_dir: Path) -> Path:
    from PIL import Image
    from .ocr import run_ocr

    pages: list[PageCtx] = []
    for i, img_path in enumerate(sorted(page_paths), start=1):
        if str(cfg.embed_images).lower() == "all":
            pages.append(PageCtx(index=i, img_path=img_path, text="", avg_conf=0.0, chars=0))
        else:
            raw = Image.open(img_path)
            img = preprocess(raw, cfg.preprocess, cfg.crop_pct)  # aplică profilul din YAML (pil_gray / opencv / etc.)
            ocr_res = run_ocr(img, cfg)

            pages.append(PageCtx(index=i, img_path=img_path, text=ocr_res.text,
                                 avg_conf=ocr_res.avg_conf, chars=ocr_res.num_chars))

    mode = str(cfg.embed_images).lower()
    if mode == "all":
        return render_with_images(cfg, pages, work_dir)
    elif mode == "none":
        return render_text_only(cfg, pages, work_dir)
    else:
        return render_text_only(cfg, pages, work_dir)
