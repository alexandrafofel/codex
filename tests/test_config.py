"""Unit tests for the configuration loader."""

from pathlib import Path

from ocrmd.config import load_config, SplitRule


def test_load_config(tmp_path: Path) -> None:
    yaml_content = """
book_title: Test Book
pdf_file: input/test.pdf
lang: eng
dpi: 300
preprocess: pil_bin
tess_psm: 4
embed_images: auto
crop_pct: 0.1
low_conf: 40
min_chars: 80
split_strategy:
  - page: 1
    type: single
  - page: 2-4
    type: double
"""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml_content)
    cfg = load_config(cfg_path)
    assert cfg.book_title == "Test Book"
    assert cfg.pdf_file == Path("input/test.pdf")
    assert cfg.preprocess == "pil_bin"
    assert cfg.tess_psm == 4
    assert cfg.crop_pct == 0.1
    assert len(cfg.split_strategy) == 2
    assert cfg.split_strategy[1].page == "2-4"