"""Command line interface for the OCR→Markdown pipeline.

This module defines the ``ocrmd`` console entry point and a set of
subcommands for running individual stages of the pipeline.  It uses
Python's built‑in ``argparse`` module to parse command line options.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

from PIL import Image
from tqdm import tqdm

from .config import OcrConfig, load_config
from .markdown import build_markdown
from .ocr import run_ocr
from .pdf import extract_pages
from .preprocess import preprocess
from .split import split_pages
from .qa import PageStats, write_csv


def _setup_logger(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def cmd_pdf(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    work_dir = Path(args.output_dir)
    extract_pages(config, work_dir)


def cmd_split(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    work_dir = Path(args.output_dir)
    # get previously extracted pages
    slug = (config.book_title.lower().replace(" ", "_") if config.book_title else config.pdf_file.stem)
    raw_dir = work_dir / slug / "raw_pages"
    pages = sorted(raw_dir.glob("scan_*.*"))
    split_pages(config, pages, work_dir)


def cmd_ocr(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    work_dir = Path(args.output_dir)
    slug = (config.book_title.lower().replace(" ", "_") if config.book_title else config.pdf_file.stem)
    split_dir = work_dir / slug / "split_pages"
    page_paths = sorted(split_dir.glob("page_*.*"))
    # optional range
    start = args.start - 1 if args.start else 0
    end = args.end if args.end else len(page_paths)
    selected = page_paths[start:end]
    # override preprocess
    if args.preprocess:
        config.preprocess = args.preprocess
    if args.tess_psm is not None:
        config.tess_psm = args.tess_psm
    if args.crop_pct is not None:
        config.crop_pct = args.crop_pct
    stats: List[PageStats] = []
    for idx, path in enumerate(tqdm(selected, desc="OCR", unit="page"), start=start + 1):
        img = Image.open(path)
        pre = preprocess(img, config.preprocess, crop_pct=config.crop_pct)
        result = run_ocr(pre, config)
        # mark whether we intend to embed this page later
        embed = result.is_low_quality(config)
        stats.append(PageStats(page=idx, conf=result.avg_conf, chars=result.num_chars, used_psm=result.used_psm,
                               embedded=embed, preprocess=config.preprocess, crop_pct=config.crop_pct))
    # write CSV for QA
    qa_path = work_dir / slug / "qa_report.csv"
    write_csv(stats, qa_path)


def cmd_md(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    work_dir = Path(args.output_dir)
    slug = (config.book_title.lower().replace(" ", "_") if config.book_title else config.pdf_file.stem)
    split_dir = work_dir / slug / "split_pages"
    page_paths = sorted(split_dir.glob("page_*.*"))
    build_markdown(config, page_paths, work_dir)


def cmd_all(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    work_dir = Path(args.output_dir)
    pages = extract_pages(config, work_dir)
    split = split_pages(config, pages, work_dir)
    # run OCR on all pages
    cmd_ocr_namespace = argparse.Namespace(
        config=args.config,
        output_dir=args.output_dir,
        start=None,
        end=None,
        preprocess=None,
        tess_psm=None,
        crop_pct=None,
    )
    cmd_ocr(cmd_ocr_namespace)
    build_markdown(config, split, work_dir)


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="OCR to Markdown pipeline")
    parser.add_argument("command", choices=["pdf", "split", "ocr", "md", "all"], help="Pipeline stage to run")
    parser.add_argument("-c", "--config", required=True, help="Path to YAML configuration file")
    parser.add_argument("-o", "--output-dir", default="output", help="Base directory for pipeline outputs")
    parser.add_argument("--start", type=int, default=None, help="First page number (1-based) to OCR")
    parser.add_argument("--end", type=int, default=None, help="Last page number (1-based, inclusive) to OCR")
    parser.add_argument("--tess-psm", type=int, default=None, help="Override Tesseract PSM for OCR stage")
    parser.add_argument("--preprocess", choices=["pil_gray", "pil_bin", "opencv"], default=None, help="Override preprocessing profile for OCR stage")
    parser.add_argument("--crop-pct", type=float, default=None, help="Override crop percentage for OCR stage")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args(argv)
    _setup_logger(args.verbose)
    cmd_map = {
        "pdf": cmd_pdf,
        "split": cmd_split,
        "ocr": cmd_ocr,
        "md": cmd_md,
        "all": cmd_all,
    }
    cmd = cmd_map[args.command]
    cmd(args)


if __name__ == "__main__":
    main()