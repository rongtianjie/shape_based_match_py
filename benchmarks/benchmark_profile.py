"""Profiling benchmark for TemplateModel building and ShapeMatcher execution."""

from __future__ import annotations

import cProfile
from pathlib import Path
import sys
import time

import cv2
import numpy as np

# Ensure project root is in sys.path when run directly
PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from shape_match.config import parse_match_config, parse_pattern_config
from shape_match.engine import ShapeMatcher, TemplateModel


def main() -> None:
    # Create a synthetic template and search image
    template_img = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(template_img, (20, 20), (80, 80), 255, -1)

    search_img = np.zeros((800, 800), dtype=np.uint8)
    cv2.rectangle(search_img, (200, 200), (300, 300), 255, -1)

    pattern_config = parse_pattern_config({"angle_extent": 360.0})
    match_config = parse_match_config({"scale_min": 0.9, "scale_max": 1.1, "numMatches": 1})

    print("Building template...")
    t0 = time.perf_counter()
    template = TemplateModel.from_image(template_img, pattern_config)
    print(f"Template built in {time.perf_counter() - t0:.4f}s")

    matcher = ShapeMatcher(pattern_config, match_config)

    print("Matching (Warmup)...")
    t0 = time.perf_counter()
    matcher.match(template, search_img)
    print(f"Warmup in {time.perf_counter() - t0:.4f}s")

    print("Profiling matching...")
    cProfile.runctx("matcher.match(template, search_img)", globals(), locals(), sort="cumtime")


if __name__ == "__main__":
    main()
