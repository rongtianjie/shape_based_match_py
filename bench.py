import time
import numpy as np
import cv2
from shape_match.engine import TemplateModel, ShapeMatcher
from shape_match.config import parse_pattern_config, parse_match_config

# Create a synthetic template and search image
template_img = np.zeros((100, 100), dtype=np.uint8)
cv2.rectangle(template_img, (20, 20), (80, 80), 255, -1)

search_img = np.zeros((800, 800), dtype=np.uint8)
cv2.rectangle(search_img, (200, 200), (300, 300), 255, -1) # larger to handle scale

pattern_config = parse_pattern_config({"angle_extent": 360})
match_config = parse_match_config({"scale_min": 0.9, "scale_max": 1.1, "num_matches": 1})

print("Building template...")
t0 = time.time()
template = TemplateModel.from_image(template_img, pattern_config)
print(f"Template built in {time.time() - t0:.4f}s")

matcher = ShapeMatcher(pattern_config, match_config)

print("Matching (Warmup)...")
t0 = time.time()
matcher.match(template, search_img)
print(f"Warmup in {time.time() - t0:.4f}s")

import cProfile
print("Profiling matching...")
cProfile.run('matcher.match(template, search_img)', sort='cumtime')
