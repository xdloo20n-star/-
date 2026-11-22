#!/usr/bin/env python3
"""Extract RAG evaluation records in the format consumed by the Go worker."""

import argparse
import json
from pathlib import Path
import sys

from pdf_pages import ExtractionError, read_pdf
from publication import extract_publication


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", required=True, type=Path)
    args = parser.parse_args()
    try:
        path = args.pdf.resolve(strict=True)
        result = extract_publication(read_pdf(path), source_path=path)
        json.dump(result, sys.stdout, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        print()
    except (ExtractionError, OSError, ValueError) as error:
        print(f"extraction failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
