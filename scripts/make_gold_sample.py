#!/usr/bin/env python3
"""Generate a stratified gold sample for OCR evaluation.

This script selects a stratified subset of pages from the processed
Markdown and images in order to build a human‑corrected ground
truth.  The sample size and distribution (start/middle/end) can be
configured via command line arguments.  The output is a directory
containing the selected page images and a corresponding Markdown
file with manually corrected text.

At present this script simply selects the first N pages as a
placeholder.  In practice you should implement a stratified random
sampling scheme.
"""

import argparse
from pathlib import Path
import shutil

def main() -> None:
    parser = argparse.ArgumentParser(description="Create gold sample for OCR evaluation")
    parser.add_argument("input_dir", help="Directory containing split page images")
    parser.add_argument("output_dir", help="Directory where the gold sample will be created")
    parser.add_argument("-n", "--num-pages", type=int, default=50, help="Number of pages to sample")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pages = sorted(input_dir.glob("page_*.*"))
    sample = pages[: args.num_pages]
    for page in sample:
        shutil.copy(page, output_dir / page.name)
    # create empty markdown for manual filling
    md = output_dir / "gold.md"
    md.touch()
    print(f"Sample of {len(sample)} pages created in {output_dir}. Please fill in gold.md with the correct text.")

if __name__ == "__main__":
    main()