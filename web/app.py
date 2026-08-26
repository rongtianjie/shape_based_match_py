from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
import time
from typing import Any, Mapping

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from pydantic import BaseModel, Field

import pattern_match
from pattern_match import estimate_contrast_thresholds, get_matched_result, get_model_shape

from shape_match.config import MATCH_DEFAULTS, PAT_DEFAULTS

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("web_app")

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
STATIC_DIR = BASE_DIR / "static"
TEST_DATA_DIR = ROOT_DIR / "tests" / "data"

app = FastAPI(
    title="Shape-Based Matching Studio",
    description="Web Frontend & API for Pure Python/OpenCV Shape-Based Template Matching",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Decode raw bytes into a BGR numpy array using OpenCV."""
    if not image_bytes:
        raise ValueError("Empty image data received")
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image from provided data")
    return image


def encode_image_base64(image: np.ndarray, quality: int = 90) -> str:
    """Encode OpenCV BGR image into base64 data URI string."""
    if max(image.shape[:2]) > 1600:
        success, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        mime = "image/jpeg"
    else:
        success, buffer = cv2.imencode(".png", image)
        mime = "image/png"
    encoded = base64.b64encode(buffer).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


_PAT_KEYS = set(PAT_DEFAULTS.keys())
_MATCH_KEYS = set(MATCH_DEFAULTS.keys())

DEFAULT_PAT_CONFIG: dict[str, Any] = dict(PAT_DEFAULTS)
DEFAULT_MATCH_CONFIG: dict[str, Any] = dict(MATCH_DEFAULTS)


@app.get("/api/config/defaults")
@app.get("/api/defaults")
def get_config_defaults() -> JSONResponse:
    """Return default parameter configurations from config.py."""
    return JSONResponse(
        content={
            "success": True,
            "pat_defaults": dict(PAT_DEFAULTS),
            "match_defaults": dict(MATCH_DEFAULTS),
            "default_pat": dict(PAT_DEFAULTS),
            "default_match": dict(MATCH_DEFAULTS),
        }
    )


def filter_pat_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Filter dictionary to only contain valid pat_config keys."""
    if not config:
        return {}
    return {k: v for k, v in config.items() if k in _PAT_KEYS}


def filter_match_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Filter dictionary to only contain valid match_config keys."""
    if not config:
        return {}
    return {k: v for k, v in config.items() if k in _MATCH_KEYS}


class ExtractRequest(BaseModel):
    model_base64: str | None = None
    sample_id: str | None = None
    pat_config: dict[str, Any] = Field(default_factory=dict)


class MatchRequest(BaseModel):
    model_base64: str | None = None
    source_base64: str | None = None
    sample_id: str | None = None
    pat_config: dict[str, Any] = Field(default_factory=dict)
    match_config: dict[str, Any] = Field(default_factory=dict)


def get_sample_images(sample_id: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    """Retrieve preset sample model and source images along with default configurations."""
    if sample_id == "meiqua_case2":
        case_dir = TEST_DATA_DIR / "meiqua_case2"
        model_path = case_dir / "model.png"
        source_path = case_dir / "source.png"
        if not model_path.exists() or not source_path.exists():
            raise HTTPException(status_code=404, detail="Sample dataset files not found")
        model = cv2.imread(str(model_path), cv2.IMREAD_COLOR)
        source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        default_pat = {
            **DEFAULT_PAT_CONFIG,
            "contrast_low": 10,
            "contrast_high": 30,
            "angle_extent": 0.0,
            "num_levels": 1,
        }
        default_match = {**DEFAULT_MATCH_CONFIG, "numMatches": 8}
        return model, source, default_pat, default_match
    elif sample_id == "synthetic_case":
        model = np.zeros((80, 96, 3), dtype=np.uint8)
        cv2.rectangle(model, (10, 10), (86, 70), (230, 230, 230), 3)
        cv2.line(model, (20, 60), (76, 20), (255, 255, 255), 4)
        cv2.circle(model, (72, 60), 8, (190, 190, 190), 3)
        cv2.rectangle(model, (10, 10), (32, 24), (0, 0, 0), -1)

        source = np.zeros((480, 640, 3), dtype=np.uint8)
        poses = [
            ((160.0, 140.0), 0.0, 1.0),
            ((440.0, 180.0), 25.0, 1.1),
            ((300.0, 360.0), -35.0, 0.9),
        ]
        height, width = model.shape[:2]
        for (cx, cy), angle, scale in poses:
            mat = cv2.getRotationMatrix2D(((width - 1) / 2.0, (height - 1) / 2.0), angle, scale)
            mat[0, 2] += cx - (width - 1) / 2.0
            mat[1, 2] += cy - (height - 1) / 2.0
            transformed = cv2.warpAffine(model, mat, (source.shape[1], source.shape[0]), flags=cv2.INTER_LINEAR)
            np.maximum(source, transformed, out=source)

        noise = np.random.default_rng(42).integers(0, 25, size=source.shape, dtype=np.uint8)
        source = cv2.add(source, noise)

        default_pat = {
            **DEFAULT_PAT_CONFIG,
            "contrast_low": 15,
            "contrast_high": 40,
            "angle_start": -45.0,
            "angle_extent": 90.0,
            "num_levels": 1,
        }
        default_match = {
            **DEFAULT_MATCH_CONFIG,
            "numMatches": 5,
            "minScore": 0.35,
        }
        return model, source, default_pat, default_match
    else:
        raise HTTPException(status_code=404, detail=f"Unknown sample_id: {sample_id}")


@app.get("/api/samples")
def list_samples() -> JSONResponse:
    """Return available preset sample datasets."""
    samples = [
        {
            "id": "meiqua_case2",
            "name": "工业零件 (Meiqua Case 2)",
            "description": "真实工业灰度金属零件，包含多个旋转与缩放实例，带有轻微噪声与背景干扰。",
            "default_pat": {
                **DEFAULT_PAT_CONFIG,
                "contrast_low": 10,
                "contrast_high": 30,
                "angle_extent": 0.0,
                "num_levels": 1,
            },
            "default_match": {**DEFAULT_MATCH_CONFIG, "numMatches": 8},
        },
        {
            "id": "synthetic_case",
            "name": "多姿态合成图形 (Synthetic Geometric)",
            "description": "含矩形、斜线、圆孔的合成目标，源图中分布有不同旋转角度（-35°、0°、+25°）与尺度的实例。",
            "default_pat": {
                **DEFAULT_PAT_CONFIG,
                "contrast_low": 15,
                "contrast_high": 40,
                "angle_start": -45.0,
                "angle_extent": 90.0,
                "num_levels": 1,
            },
            "default_match": {
                **DEFAULT_MATCH_CONFIG,
                "numMatches": 5,
                "minScore": 0.35,
            },
        },
    ]
    return JSONResponse(content={"samples": samples})


@app.get("/api/samples/{sample_id}/images")
def get_sample_images_endpoint(sample_id: str) -> JSONResponse:
    """Return base64 encoded model and source images for the specified sample."""
    try:
        model, source, default_pat, default_match = get_sample_images(sample_id)
        return JSONResponse(
            content={
                "success": True,
                "sample_id": sample_id,
                "model_image": encode_image_base64(model),
                "source_image": encode_image_base64(source),
                "model_size": {"width": int(model.shape[1]), "height": int(model.shape[0])},
                "source_size": {"width": int(source.shape[1]), "height": int(source.shape[0])},
                "default_pat": default_pat,
                "default_match": default_match,
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        LOGGER.exception("Failed to load sample images")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/extract-template")
async def extract_template(
    file: UploadFile | None = File(default=None),
    config_json: str = Form(default="{}"),
) -> JSONResponse:
    """Extract gradient-orientation shape features from uploaded template image."""
    started = time.perf_counter()
    try:
        if file is not None:
            image_bytes = await file.read()
            model = decode_image_bytes(image_bytes)
        else:
            raise HTTPException(status_code=400, detail="No template image provided")

        try:
            raw_config = json.loads(config_json)
            pat_config = filter_pat_config(raw_config)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON in config_json")

        model_view = get_model_shape(model, pat_config)
        duration_ms = (time.perf_counter() - started) * 1000.0

        if model_view is None:
            return JSONResponse(
                content={
                    "success": False,
                    "duration_ms": round(duration_ms, 2),
                    "feature_count": 0,
                    "width": int(model.shape[1]),
                    "height": int(model.shape[0]),
                    "message": "模板平坦或边缘特征点少于 8 个，请调低 contrast_low/contrast_high 阈值或更换对比度更明显的模板",
                    "model_view": None,
                    "model_original": encode_image_base64(model),
                }
            )

        parsed_config = pattern_match._parse_pattern_config(pat_config)
        features = pattern_match._extract_features(
            pattern_match._to_gray(model), parsed_config, pattern_match._to_bgr(model)
        )
        feature_count = len(features.points) if features is not None else 0

        return JSONResponse(
            content={
                "success": True,
                "duration_ms": round(duration_ms, 2),
                "feature_count": feature_count,
                "width": int(model.shape[1]),
                "height": int(model.shape[0]),
                "message": f"成功提取 {feature_count} 个方向梯度特征点",
                "model_view": encode_image_base64(model_view),
                "model_original": encode_image_base64(model),
            }
        )
    except (ValueError, TypeError) as e:
        LOGGER.warning("Validation error in extract_template: %s", e)
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": f"参数错误: {str(e)}"},
        )
    except Exception as e:
        LOGGER.exception("Unexpected error in extract_template")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"服务端异常: {str(e)}"},
        )


@app.post("/api/estimate-contrast-json")
def estimate_contrast_json(req: ExtractRequest) -> JSONResponse:
    """Estimate explicit contrast thresholds from a template image."""
    started = time.perf_counter()
    try:
        if req.sample_id:
            model, _, _, _ = get_sample_images(req.sample_id)
        elif req.model_base64:
            b64_data = req.model_base64.split(",", 1)[-1]
            model = decode_image_bytes(base64.b64decode(b64_data))
        else:
            raise ValueError("Must provide model_base64 or sample_id")

        contrast_low, contrast_high = estimate_contrast_thresholds(model)
        duration_ms = (time.perf_counter() - started) * 1000.0
        return JSONResponse(
            content={
                "success": True,
                "contrast_low": contrast_low,
                "contrast_high": contrast_high,
                "duration_ms": round(duration_ms, 2),
                "message": f"自动对比度阈值: low={contrast_low}, high={contrast_high}",
            }
        )
    except (ValueError, TypeError) as e:
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": f"模板图像错误: {str(e)}"},
        )
    except Exception as e:
        LOGGER.exception("Unexpected error in estimate_contrast_json")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"服务端异常: {str(e)}"},
        )


@app.post("/api/extract-template-json")
def extract_template_json(req: ExtractRequest) -> JSONResponse:
    """JSON version of extract-template supporting base64 or sample_id."""
    started = time.perf_counter()
    try:
        pat_config = filter_pat_config(req.pat_config)
        if req.sample_id:
            model, _, _, _ = get_sample_images(req.sample_id)
        elif req.model_base64:
            b64_data = req.model_base64
            if "," in b64_data:
                b64_data = b64_data.split(",", 1)[1]
            image_bytes = base64.b64decode(b64_data)
            model = decode_image_bytes(image_bytes)
        else:
            raise HTTPException(status_code=400, detail="Must provide model_base64 or sample_id")

        model_view = get_model_shape(model, pat_config)
        duration_ms = (time.perf_counter() - started) * 1000.0

        if model_view is None:
            return JSONResponse(
                content={
                    "success": False,
                    "duration_ms": round(duration_ms, 2),
                    "feature_count": 0,
                    "width": int(model.shape[1]),
                    "height": int(model.shape[0]),
                    "message": "模板平坦或边缘特征点少于 8 个，请调低 contrast_low/contrast_high 阈值",
                    "model_view": None,
                    "model_original": encode_image_base64(model),
                }
            )

        parsed_config = pattern_match._parse_pattern_config(pat_config)
        features = pattern_match._extract_features(
            pattern_match._to_gray(model), parsed_config, pattern_match._to_bgr(model)
        )
        feature_count = len(features.points) if features is not None else 0

        return JSONResponse(
            content={
                "success": True,
                "duration_ms": round(duration_ms, 2),
                "feature_count": feature_count,
                "width": int(model.shape[1]),
                "height": int(model.shape[0]),
                "message": f"成功提取 {feature_count} 个特征点",
                "model_view": encode_image_base64(model_view),
                "model_original": encode_image_base64(model),
            }
        )
    except (ValueError, TypeError) as e:
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": f"参数错误: {str(e)}"},
        )
    except Exception as e:
        LOGGER.exception("Unexpected error in extract_template_json")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"服务端异常: {str(e)}"},
        )


@app.post("/api/match")
async def match_endpoint(
    model_file: UploadFile | None = File(default=None),
    source_file: UploadFile | None = File(default=None),
    pat_config_json: str = Form(default="{}"),
    match_config_json: str = Form(default="{}"),
) -> JSONResponse:
    """Execute multi-instance shape-based matching."""
    started = time.perf_counter()
    try:
        if model_file is None or source_file is None:
            raise HTTPException(status_code=400, detail="Both model_file and source_file must be uploaded")

        model_bytes = await model_file.read()
        source_bytes = await source_file.read()

        model = decode_image_bytes(model_bytes)
        source = decode_image_bytes(source_bytes)

        pat_config = filter_pat_config(json.loads(pat_config_json))
        match_config = filter_match_config(json.loads(match_config_json))

        results, match_view = get_matched_result(model, source, pat_config, match_config)
        duration_ms = (time.perf_counter() - started) * 1000.0

        matches_list = []
        for rank, row in enumerate(results, start=1):
            cx, cy, score, angle, scale = row
            matches_list.append(
                {
                    "rank": rank,
                    "cx": round(float(cx), 2),
                    "cy": round(float(cy), 2),
                    "score": round(float(score), 4),
                    "angle": round(float(angle), 2),
                    "scale": round(float(scale), 3),
                }
            )

        return JSONResponse(
            content={
                "success": True,
                "duration_ms": round(duration_ms, 2),
                "match_count": len(matches_list),
                "matches": matches_list,
                "match_view": encode_image_base64(match_view) if match_view is not None else None,
                "source_original": encode_image_base64(source),
                "source_width": int(source.shape[1]),
                "source_height": int(source.shape[0]),
                "message": f"匹配完成，找到 {len(matches_list)} 个目标" if matches_list else "未找到符合得分阈值的匹配目标",
            }
        )
    except (ValueError, TypeError) as e:
        LOGGER.warning("Validation error in match_endpoint: %s", e)
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": f"参数错误: {str(e)}"},
        )
    except Exception as e:
        LOGGER.exception("Unexpected error in match_endpoint")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"服务端异常: {str(e)}"},
        )


@app.post("/api/match-json")
def match_json_endpoint(req: MatchRequest) -> JSONResponse:
    """JSON version of match endpoint supporting base64 or sample_id."""
    started = time.perf_counter()
    try:
        pat_config = filter_pat_config(req.pat_config)
        match_config = filter_match_config(req.match_config)

        if req.sample_id:
            model, source, _, _ = get_sample_images(req.sample_id)
        elif req.model_base64 and req.source_base64:
            m_b64 = req.model_base64.split(",", 1)[1] if "," in req.model_base64 else req.model_base64
            s_b64 = req.source_base64.split(",", 1)[1] if "," in req.source_base64 else req.source_base64
            model = decode_image_bytes(base64.b64decode(m_b64))
            source = decode_image_bytes(base64.b64decode(s_b64))
        else:
            raise HTTPException(status_code=400, detail="Must provide (model_base64 and source_base64) or sample_id")

        results, match_view = get_matched_result(model, source, pat_config, match_config)
        duration_ms = (time.perf_counter() - started) * 1000.0

        matches_list = []
        for rank, row in enumerate(results, start=1):
            cx, cy, score, angle, scale = row
            matches_list.append(
                {
                    "rank": rank,
                    "cx": round(float(cx), 2),
                    "cy": round(float(cy), 2),
                    "score": round(float(score), 4),
                    "angle": round(float(angle), 2),
                    "scale": round(float(scale), 3),
                }
            )

        return JSONResponse(
            content={
                "success": True,
                "duration_ms": round(duration_ms, 2),
                "match_count": len(matches_list),
                "matches": matches_list,
                "match_view": encode_image_base64(match_view) if match_view is not None else None,
                "source_original": encode_image_base64(source),
                "source_width": int(source.shape[1]),
                "source_height": int(source.shape[0]),
                "message": f"匹配完成，找到 {len(matches_list)} 个目标" if matches_list else "未找到符合得分阈值的匹配目标",
            }
        )
    except (ValueError, TypeError) as e:
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": f"参数错误: {str(e)}"},
        )
    except Exception as e:
        LOGGER.exception("Unexpected error in match_json_endpoint")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"服务端异常: {str(e)}"},
        )


# Dynamic static files mount
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> FileResponse:
    """Serve favicon.ico at root URL."""
    fav_file = STATIC_DIR / "favicon.ico"
    if not fav_file.exists():
        raise HTTPException(status_code=404, detail="Favicon not found")
    return FileResponse(str(fav_file), media_type="image/x-icon")


@app.get("/")
def index() -> FileResponse:
    """Serve main frontend page."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found")
    return FileResponse(
        str(index_file),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )
