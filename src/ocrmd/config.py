"""Configuration loading and validation for the OCR pipeline.

The configuration is stored in YAML files under the ``configs/``
directory.  The :class:`OcrConfig` dataclass captures the relevant
fields with sensible defaults and performs basic validation.  A
utility function :func:`load_config` reads a YAML file and returns
the corresponding dataclass instance.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Union

import yaml


@dataclass
class SplitRule:
    """A rule describing how to split a composite scan.

    Attributes
    ----------
    page:
        Either a single page number or a string describing a range of
        pages (e.g. ``"2-300"``).  Pages are 1‑based indices.
    type:
        The expected composition type for the given pages.  Supported
        values are ``"single"`` (no split), ``"double"`` (two
        pages per scan) and ``"quadruple"`` (four pages per scan).
    """

    page: Union[int, str]
    type: str


@dataclass
class OcrConfig:
    """Dataclass capturing configuration parameters for OCR.

    The fields map one‑to‑one to keys in the YAML configuration.  If
    certain fields are omitted in the YAML file, defaults provided
    here will be used instead.
    """

    book_title: str
    pdf_file: Path
    lang: str = "eng"
    dpi: int = 400
    preprocess: str = "pil_gray"  # pil_gray | pil_bin | opencv
    tess_psm: int = 6
    embed_images: str = "auto"  # auto | all | none
    crop_pct: float = 0.05
    low_conf: int = 50
    min_chars: int = 100
    split_strategy: List[SplitRule] = field(default_factory=list)

    def __post_init__(self) -> None:
        # normalise PDF path to a Path instance
        if isinstance(self.pdf_file, str):
            self.pdf_file = Path(self.pdf_file)
        # validate preprocess profile
        if self.preprocess not in {"pil_gray", "pil_bin", "opencv"}:
            raise ValueError(f"Unknown preprocess profile: {self.preprocess}")
        if self.embed_images not in {"auto", "all", "none"}:
            raise ValueError(f"Invalid embed_images: {self.embed_images}")
        if self.tess_psm not in range(0, 14):
            raise ValueError(f"tess_psm must be between 0 and 13 inclusive, got {self.tess_psm}")
        # ensure split strategy pages are correct types
        for rule in self.split_strategy:
            if rule.type not in {"single", "double", "quadruple"}:
                raise ValueError(f"Invalid split type: {rule.type}")


def load_config(path: Union[str, Path]) -> OcrConfig:
    """Load a configuration YAML file into an :class:`OcrConfig`.

    Parameters
    ----------
    path:
        Path to the YAML configuration file.

    Returns
    -------
    OcrConfig
        A populated configuration dataclass instance.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # parse split strategy into dataclass instances
    rules: List[SplitRule] = []
    for entry in data.get("split_strategy", []):
        rules.append(SplitRule(page=entry["page"], type=entry["type"]))
    data["split_strategy"] = rules

    return OcrConfig(**data)