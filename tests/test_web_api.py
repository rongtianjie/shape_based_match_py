from __future__ import annotations

import base64
import unittest
import cv2
from fastapi.testclient import TestClient
import numpy as np

from web.app import app

client = TestClient(app)


def make_test_image_bytes(shape=(80, 100), draw_rect=True) -> bytes:
    img = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    if draw_rect:
        cv2.rectangle(img, (10, 10), (shape[1] - 10, shape[0] - 10), (255, 255, 255), 3)
        cv2.circle(img, (shape[1] // 2, shape[0] // 2), 12, (200, 200, 200), -1)
    _, buffer = cv2.imencode(".png", img)
    return buffer.tobytes()


class TestWebApi(unittest.TestCase):
    def test_index_page(self):
        resp = client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("ShapeMatch", resp.text)
        self.assertIn('id="btn-auto-contrast"', resp.text)

    def test_list_samples(self):
        resp = client.get("/api/samples")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("samples", data)
        self.assertGreaterEqual(len(data["samples"]), 2)
        ids = [s["id"] for s in data["samples"]]
        self.assertIn("meiqua_case2", ids)
        self.assertIn("synthetic_case", ids)

    def test_get_sample_images(self):
        resp = client.get("/api/samples/meiqua_case2/images")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertIn("model_image", data)
        self.assertIn("source_image", data)
        self.assertTrue(data["model_image"].startswith("data:image/"))
        self.assertTrue(data["source_image"].startswith("data:image/"))

    def test_extract_template_json_sample(self):
        resp = client.post(
            "/api/extract-template-json",
            json={
                "sample_id": "meiqua_case2",
                "pat_config": {
                    "contrast_low": 10,
                    "contrast_high": 30,
                    "angle_start": 0.0,
                    "angle_extent": 0.0,
                    "num_levels": 1,
                },
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertGreaterEqual(data["feature_count"], 8)
        self.assertTrue(data["model_view"].startswith("data:image/"))

    def test_extract_template_multipart_upload(self):
        img_bytes = make_test_image_bytes()
        resp = client.post(
            "/api/extract-template",
            files={"file": ("model.png", img_bytes, "image/png")},
            data={"config_json": '{"contrast_low": 10, "contrast_high": 30}'},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertGreaterEqual(data["feature_count"], 8)
        self.assertIsNotNone(data["model_view"])

    def test_estimate_contrast_from_template(self):
        img_bytes = make_test_image_bytes()
        encoded = base64.b64encode(img_bytes).decode("ascii")
        resp = client.post(
            "/api/estimate-contrast-json",
            json={"model_base64": f"data:image/png;base64,{encoded}"},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertGreaterEqual(data["contrast_low"], 1)
        self.assertGreater(data["contrast_high"], data["contrast_low"])
        self.assertLessEqual(data["contrast_high"], 255)

    def test_estimate_contrast_requires_template(self):
        resp = client.post("/api/estimate-contrast-json", json={})
        self.assertEqual(resp.status_code, 400)
        self.assertFalse(resp.json()["success"])

    def test_extract_flat_template(self):
        flat_bytes = make_test_image_bytes(draw_rect=False)
        resp = client.post(
            "/api/extract-template",
            files={"file": ("flat.png", flat_bytes, "image/png")},
            data={"config_json": '{"contrast_low": 10, "contrast_high": 30}'},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertFalse(data["success"])
        self.assertEqual(data["feature_count"], 0)

    def test_match_json_sample(self):
        resp = client.post(
            "/api/match-json",
            json={
                "sample_id": "meiqua_case2",
                "pat_config": {
                    "contrast_low": 10,
                    "contrast_high": 30,
                    "angle_start": 0.0,
                    "angle_extent": 0.0,
                    "num_levels": 1,
                },
                "match_config": {
                    "numMatches": 8,
                    "minScore": 0.15,
                    "scale_min": 0.8,
                    "scale_max": 1.2,
                },
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertGreaterEqual(data["match_count"], 1)
        self.assertGreaterEqual(len(data["matches"]), 1)
        first = data["matches"][0]
        self.assertIn("cx", first)
        self.assertIn("cy", first)
        self.assertIn("score", first)
        self.assertIsNotNone(data["match_view"])

    def test_match_multipart_upload(self):
        model_bytes = make_test_image_bytes((60, 60))
        model_img = cv2.imdecode(np.frombuffer(model_bytes, np.uint8), cv2.IMREAD_COLOR)
        source_img = np.zeros((200, 200, 3), dtype=np.uint8)
        source_img[50 : 50 + 60, 50 : 50 + 60] = model_img
        _, source_bytes = cv2.imencode(".png", source_img)

        resp = client.post(
            "/api/match",
            files={
                "model_file": ("model.png", model_bytes, "image/png"),
                "source_file": ("source.png", source_bytes.tobytes(), "image/png"),
            },
            data={
                "pat_config_json": '{"contrast_low": 10, "contrast_high": 30, "angle_start": 0.0, "angle_extent": 0.0}',
                "match_config_json": '{"numMatches": 1, "minScore": 0.5}',
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["match_count"], 1)
        match = data["matches"][0]
        self.assertLess(abs(match["cx"] - 79.5), 3.0)
        self.assertLess(abs(match["cy"] - 79.5), 3.0)

    def test_invalid_configs(self):
        # contrast_low >= contrast_high
        resp = client.post(
            "/api/extract-template-json",
            json={
                "sample_id": "meiqua_case2",
                "pat_config": {"contrast_low": 50, "contrast_high": 20},
            },
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("contrast_low must be less than contrast_high", resp.json()["message"])

        # scale_min > scale_max
        resp = client.post(
            "/api/match-json",
            json={
                "sample_id": "meiqua_case2",
                "pat_config": {},
                "match_config": {"scale_min": 1.5, "scale_max": 0.8},
            },
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("scale_min must be less than or equal", resp.json()["message"])

    def test_mixed_configs_are_safely_filtered(self):
        # Even if mixed keys are sent, filtering prevents errors
        resp = client.post(
            "/api/extract-template-json",
            json={
                "sample_id": "meiqua_case2",
                "pat_config": {
                    "contrast_low": 10,
                    "contrast_high": 30,
                    "minScore": 0.5,
                    "scale_min": 0.8,
                    "scale_max": 1.2,
                },
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["success"])

    def test_config_defaults(self):
        from shape_match.config import MATCH_DEFAULTS, PAT_DEFAULTS
        from web.app import DEFAULT_MATCH_CONFIG, DEFAULT_PAT_CONFIG

        self.assertEqual(DEFAULT_PAT_CONFIG, PAT_DEFAULTS)
        self.assertEqual(DEFAULT_MATCH_CONFIG, MATCH_DEFAULTS)

        resp = client.get("/api/config/defaults")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["pat_defaults"], PAT_DEFAULTS)
        self.assertEqual(data["match_defaults"], MATCH_DEFAULTS)

        resp_alias = client.get("/api/defaults")
        self.assertEqual(resp_alias.status_code, 200)
        self.assertEqual(resp_alias.json()["pat_defaults"], PAT_DEFAULTS)


if __name__ == "__main__":
    unittest.main()
