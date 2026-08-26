# shape_based_match_py

一个只依赖 NumPy 与 OpenCV 的二维 shape-based template matching 实现。算法使用稀疏梯度方向而不是灰度值进行匹配，支持平移、旋转、缩放、多目标检测与图像金字塔，适合纹理较少、亮度变化明显的工业图像。

本项目借鉴 HALCON shape model 与 LINE-MOD 的核心思想，但不依赖 HALCON，也不保证与 HALCON 的得分或位置结果逐项一致。

## 安装

建议在独立 Python 环境中安装：

```bash
python -m pip install -e .
```

开发和测试依赖：

```bash
python -m pip install -e '.[test]'
python -m pytest
```

支持 Python 3.10 及以上版本。运行时依赖为 `numpy` 和标准版 `opencv-python`，不需要 `opencv-contrib-python` 或本地 C++ 编译。

## 快速使用

```python
import cv2

from pattern_match import estimate_contrast_thresholds, get_matched_result, get_model_shape

model = cv2.imread("model.png")
source = cv2.imread("source.png")

contrast_low, contrast_high = estimate_contrast_thresholds(model)

pattern_config = {
    "contrast_low": contrast_low,
    "contrast_high": contrast_high,
    "min_contrast": 3,
    "min_cont_len": 1,
    "use_polarity": 0,
    "angle_start": -20.0,
    "angle_extent": 40.0,
    "angle_step": 0.0,
    "num_levels": 1,
}
match_config = {
    "subpixel": 1,
    "numMatches": 5,
    "minScore": 0.3,
    "scale_min": 0.8,
    "scale_max": 1.2,
    "maxOverLap": 0.5,
    "greedness": 0.75,
}

model_view = get_model_shape(model, pattern_config)
results, match_view = get_matched_result(model, source, pattern_config, match_config)

for cx, cy, score, angle, scale in results:
    print(f"center=({cx:.1f}, {cy:.1f}), score={score:.3f}, angle={angle:.1f}, scale={scale:.2f}")

if model_view is not None:
    cv2.imwrite("model_features.png", model_view)
if match_view is not None:
    cv2.imwrite("matches.png", match_view)
```

完整接口、默认值和异常规则参见 [API_INTERFACE.md](API_INTERFACE.md)。
二次开发支持 `from shape_match import TemplateModel, ShapeMatcher` 进行模板复用与分步匹配。Web 页面模板配置区的 **Auto** 按钮使用同一个 `estimate_contrast_thresholds(model)` 函数自动回填双阈值。

## 模块架构

项目已解耦为高内聚子模块：
- `shape_match.config`: 配置类 `PatternConfig`, `MatchConfig` 及严格校验
- `shape_match.gradients`: 梯度计算、Sobel、8 方向无向角度量化与方向一致性滤波
- `shape_match.features`: 背景抑制前景分割、最大分散度采样及特征提取
- `shape_match.transforms`: 2D 几何变换、多通道模板卷积核构建与多边形 IoU
- `shape_match.response_maps`: 8 通道方向响应图计算与空间膨胀容差
- `shape_match.matcher`: 粗精分层搜索、局部网格精细化、外观验证与 NMS 抑制
- `shape_match.visualization`: 特征点分布与匹配边界框渲染
- `shape_match.engine`: `TemplateModel`（模板对象）与 `ShapeMatcher`（匹配引擎）
- `pattern_match.py`: 顶层门面模块，保持与原始函数签名和返回结构兼容

兼容参数会真实参与算法：`use_polarity` 切换无向/有向梯度，
`min_contrast` 和 `min_cont_len` 过滤模型及搜索图边缘，`angle_step`
控制粗角度采样，`subpixel` 控制中心坐标细化，`maxOverLap` 控制
旋转框 NMS，`greedness` 控制粗搜索剪枝。它们保持旧接口的用途和默认值，
但纯 Python/OpenCV 实现与原生 DLL 的内部离散化公式不同，因此不保证逐像素、
逐分数完全一致。

## 算法概览

1. 使用 Sobel 梯度、Canny 双阈值连接及局部方向一致性提取轮廓。
2. 从轮廓中选择最多 256 个空间分布均匀的特征点。
3. 将无极性的梯度方向量化为 8 类，并建立带 3×3 空间容差的方向响应图。
4. 按 10° 和 0.1 尺度步长进行粗搜索，再在候选附近的 ±5°、±0.05 尺度邻域内按 1°、0.01 尺度和整数像素位置细化。
5. 使用旋转模板外框的 IoU 进行非极大值抑制，最后按分数降序输出结果。

`num_levels=1` 会在半分辨率完成全图粗搜索并回到原图细化；`num_levels=0` 在原分辨率完成全部搜索。大范围旋转与缩放组合会增加运行时间。

## 限制

- 模板应当尽量包围目标；当前接口不支持 ROI 或掩膜参数。当模板边框能反映出与中心目标明显不同的背景时，算法会自动识别前景区域并将形状特征限制在目标边界附近；若模板本身就是紧密裁剪、背景强边缘密布或几乎充满全图，仍会退化为使用全图边缘。
- 返回的是整数像素附近的中心位置，角度和尺度分别细化到约 1° 与 0.01。
- 局部遮挡可以降低但不会自动重加权缺失特征，严重遮挡需要适当降低 `minScore`。
- 对称目标可能存在多个等价角度；此时输出其中一个最高分姿态。
- 若背景本身高度重复（如密集网格、周期性纹理），部分区域的方向特征可能与目标同样吻合甚至更高，导致粗搜索候选阶段排除真实目标或将其它区域误判为最佳匹配；此时应尽量让模板紧密包围目标并保持背景相对简单，接口不提供额外参数专门抑制这类重复背景干扰。

## 参考实现与数据

- [meiqua/shape_based_matching](https://github.com/meiqua/shape_based_matching)，BSD-2-Clause：稀疏梯度方向、方向响应图、模板变换与金字塔设计。
- [OpenCV LINE-MOD](https://github.com/opencv/opencv_contrib/blob/4.x/modules/rgbd/src/linemod.cpp)，Apache-2.0 / OpenCV 附加许可：方向量化、扩散响应和线性内存匹配思路。
- 测试中的公开图像来源和许可证全文见 [tests/data/THIRD_PARTY_NOTICES.md](tests/data/THIRD_PARTY_NOTICES.md)。

