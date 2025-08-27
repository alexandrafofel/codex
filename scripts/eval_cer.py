#!/usr/bin/env python3
"""Evaluate OCR output against a gold standard.

This script compares the generated Markdown file to a gold version
containing the correct text.  It computes character error rate (CER),
word error rate (WER) and heading detection F1 score.  Results are
written to the console and can optionally be saved as a JSON file.
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple

from ocrmd.qa import compute_cer, compute_wer, evaluate_headings

HEADING_REGEX = re.compile(r"^(#+)\s+(.*)$")


def parse_markdown(path: Path) -> Tuple[str, List[str]]:
    """Return the plain text and a sequence of heading tags for a Markdown file."""
    text_lines: List[str] = []
    heading_tags: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            m = HEADING_REGEX.match(stripped)
            if m:
                hashes, content = m.group(1), m.group(2)
                level = len(hashes)
                tag = {1: "H1", 2: "H2", 3: "H3"}.get(level, "P")
                heading_tags.append(tag)
                text_lines.append(content)
            else:
                if stripped:
                    heading_tags.append("P")
                    text_lines.append(stripped)
    return "\n".join(text_lines), heading_tags


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate OCR output against gold standard")
    parser.add_argument("pred_md", help="Predicted Markdown file")
    parser.add_argument("gold_md", help="Gold standard Markdown file")
    parser.add_argument("--json-out", help="Optional path to write metrics as JSON")
    args = parser.parse_args()

    pred_text, pred_tags = parse_markdown(Path(args.pred_md))
    gold_text, gold_tags = parse_markdown(Path(args.gold_md))

    cer = compute_cer(pred_text, gold_text)
    wer = compute_wer(pred_text, gold_text)
    f1, precision, recall = evaluate_headings(pred_tags, gold_tags)
    result = {
        "cer": cer,
        "wer": wer,
        "heading_f1": f1,
        "heading_precision": precision,
        "heading_recall": recall,
    }
    print(json.dumps(result, indent=2))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()