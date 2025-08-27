"""Top‑level package for the OCR→Markdown pipeline.

This module exposes a programmatic API as well as a command line
interface entry point via the `ocrmd` console script.  See
`ocrmd.cli` for details on the supported subcommands.
"""

__all__ = [
    "pdf", "split", "preprocess", "ocr", "structure", "markdown", "qa", "cli", "config"
]