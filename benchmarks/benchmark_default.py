"""Small non-gating benchmark for the default narrow-angle search."""

from __future__ import annotations

from pathlib import Path
import sys
import time

import cv2
import numpy as np

# Ensure project root is in sys.path when run directly
PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from pattern_match import get_matched_result


def main() -> None:
    model = np.zeros((96, 112, 3), dtype=np.uint8)
    cv2.rectangle(model, (10, 10), (90, 70), (255, 255, 255), 3)
    cv2.line(model, (18, 65), (82, 18), (200, 200, 200), 4)

    source = np.zeros((900, 1200, 3), dtype=np.uint8)
    matrix = cv2.getRotationMatrix2D(((model.shape[1] - 1) / 2, (model.shape[0] - 1) / 2), 3.0, 1.0)
    matrix[0, 2] += 700 - (model.shape[1] - 1) / 2
    matrix[1, 2] += 420 - (model.shape[0] - 1) / 2
    source = np.maximum(source, cv2.warpAffine(model, matrix, (source.shape[1], source.shape[0])))

    durations = []
    result = []
    for _ in range(5):
        started = time.perf_counter()
        result, _ = get_matched_result(
            model,
            source,
            {
                "angle_start": -5.0,
                "angle_extent": 10.0,
                "num_levels": 1,
            },
            {
                "numMatches": 1,
                "scale_min": 1.0,
                "scale_max": 1.0,
            },
        )
        durations.append(time.perf_counter() - started)

    print(f"median_seconds={float(np.median(durations)):.4f}")
    print(f"best_match={result[0] if result else None}")


if __name__ == "__main__":
    main()
