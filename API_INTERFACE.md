# Pattern Match API 接口文档

`pattern_match` 模块提供模板阈值估算、模型形状可视化和 shape-based template matching。输入图像均为 `numpy.ndarray`，支持二维灰度、BGR 和 BGRA 格式。

## `estimate_contrast_thresholds`

```python
def estimate_contrast_thresholds(template)
```

根据模板的 Sobel 梯度幅值分布自动估算 Canny 双阈值，返回 `(contrast_low, contrast_high)`。结果始终满足 `1 <= contrast_low < contrast_high <= 255`，可直接写入 `pat_config`；平坦模板返回默认值 `(3, 5)`。

## `get_model_shape`

```python
def get_model_shape(model, pat_config={})
```

从模板中提取稀疏梯度方向特征，并返回叠加特征点的 BGR 可视化图像。

### 参数

| 参数 | 类型 | 必需 | 说明 |
|---|---|---:|---|
| `model` | `numpy.ndarray` | 是 | 非空灰度、BGR 或 BGRA 模板图像 |
| `pat_config` | `Mapping` | 否 | 模型和姿态搜索配置；缺失字段使用默认值 |

### 返回值

| 类型 | 说明 |
|---|---|
| `numpy.ndarray` | 与模板等宽高的 BGR 可视化图像 |
| `None` | 模板平坦或可用边缘特征少于 8 个 |

该函数不会修改输入图像或配置字典。

## `get_matched_result`

```python
def get_matched_result(model, src, pat_config={}, match_config={})
```

在源图像中搜索模板的旋转、缩放实例。

### 参数

| 参数 | 类型 | 必需 | 说明 |
|---|---|---:|---|
| `model` | `numpy.ndarray` | 是 | 非空灰度、BGR 或 BGRA 模板图像 |
| `src` | `numpy.ndarray` | 是 | 非空灰度、BGR 或 BGRA 搜索图像 |
| `pat_config` | `Mapping` | 否 | 特征提取及角度搜索配置 |
| `match_config` | `Mapping` | 否 | 匹配数量、阈值及尺度搜索配置 |

### 返回值

```python
match_result, show = get_matched_result(model, src, pat_config, match_config)
```

- `match_result`：按得分降序排列的 `list[list[float]]`，每项固定为：

  ```python
  [cx, cy, score, angle, scale]
  ```

  - `cx`：目标中心的 X/列坐标，单位为像素。
  - `cy`：目标中心的 Y/行坐标，单位为像素。
  - `score`：梯度方向平均相似度，范围 `[0.0, 1.0]`。
  - `angle`：模板为对齐源图而进行的逆时针旋转角度，单位为度。
  - `scale`：相对于输入模板的缩放比例。
- `show`：源图的 BGR 副本，包含旋转外框、中心、分数、角度和尺度；无结果时为 `None`。
- 未找到匹配或模板特征不足时返回 `([], None)`。

## 配置参数

### `pat_config`

| 字段 | 类型 | 默认值 | 有效范围及说明 |
|---|---|---:|---|
| `contrast_low` | `int` | `3` | 大于 0；双阈值边缘的低阈值 |
| `contrast_high` | `int` | `5` | 必须大于 `contrast_low` |
| `min_contrast` | `int` | `3` | `[0, 255]`；归一化到约 `0..10` 的最小梯度对比度，0 表示不额外过滤；大于 10 会滤除全部边缘 |
| `min_cont_len` | `int` | `1` | 大于等于 1；丢弃最长轮廓弧长小于该值的 8 连通边缘组件 |
| `num_levels` | `int` | `0` | `0`：全分辨率搜索；`1`：半分辨率粗搜索后原图细化 |
| `use_polarity` | `int` | `0` | `0`：忽略明暗极性；`1`：区分相反梯度方向，可排除对比度反转目标 |
| `angle_start` | `float` | `0.0` | `[-360, 360]`；搜索起始角度 |
| `angle_extent` | `float` | `360.0` | `[0, 360]`；搜索范围，0 表示固定角度 |
| `angle_step` | `float` | `0.0` | `[0, 360]`；粗搜索角度步长，0 使用自动值 10°，最终仍以 1° 局部细化 |

`angle_start + angle_extent` 不得超过 360。范围为 360° 时不会重复首尾姿态。

### `match_config`

| 字段 | 类型 | 默认值 | 有效范围及说明 |
|---|---|---:|---|
| `subpixel` | `int` | `1` | `0`：整数坐标；`1`：轴向抛物线插值；`2`：二维二次曲面最小二乘拟合 |
| `scale_min` | `float` | `0.8` | 大于 0；最小搜索尺度 |
| `scale_max` | `float` | `1.2` | 大于等于 `scale_min`；最大搜索尺度 |
| `minScore` | `float` | `0.15` | `[0.0, 1.0]`；最终匹配最低得分 |
| `maxOverLap` | `float` | `0.5` | `[0.0, 1.0]`；旋转模板框 IoU 大于该值时抑制低分重叠结果 |
| `greedness` | `float` | `0.75` | `[0.0, 1.0]`；粗搜索剪枝强度，越大速度越快但弱目标漏检风险越高 |
| `numMatches` | `int` | `1` | 大于等于 1；NMS 后最多返回的结果数 |

空配置和部分配置均有效，未提供的字段由上述默认值补齐。未知字段会触发 `ValueError`，避免拼写错误被静默忽略。

## 异常与失败

参数类型、图像形状或配置范围不合法时抛出 `TypeError` 或 `ValueError`。特别地：

```text
ValueError: contrast_low must be less than contrast_high
```

图像内容无法生成有效模型不属于参数错误：`get_model_shape` 返回 `None`，`get_matched_result` 返回 `([], None)`。两个公开函数都通过模块 logger 以 `INFO` 级别记录执行时间，库本身不会修改全局日志配置。

## 示例

```python
import cv2

from pattern_match import get_matched_result, get_model_shape

model = cv2.imread("model.png")
source = cv2.imread("source.png")

pat_config = {
    "contrast_low": 7,
    "contrast_high": 18,
    "angle_start": 0.0,
    "angle_extent": 360.0,
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

model_view = get_model_shape(model, pat_config)
results, result_view = get_matched_result(model, source, pat_config, match_config)

for cx, cy, score, angle, scale in results:
    print(cx, cy, score, angle, scale)
```

模板外框越紧、强边缘越属于目标本身，匹配通常越稳定。大范围角度与尺度组合会增加搜索时间。

## 二次开发与高级用法 (`shape_match`)

对于需要多帧处理（如视频流、工业相机连续采集）、批量匹配或自定义扩展算法的场景，项目提供了高内聚的模块化架构 `shape_match`：

```python
import cv2
from shape_match import TemplateModel, ShapeMatcher

# 1. 一次性提取模板特征（避免在每帧图像中重复提取）
template = TemplateModel.from_image(model, pat_config)

# 2. 初始化匹配器
matcher = ShapeMatcher(pat_config, match_config)

# 3. 对多帧图像连续匹配
for frame in frame_stream:
    matches, show = matcher.match(template, frame)
    for candidate in matches:
        print(candidate.cx, candidate.cy, candidate.score, candidate.angle, candidate.scale)
```

子模块职责划分：
- `shape_match.config`：配置定义与参数校验
- `shape_match.gradients`：Sobel 梯度、8 角度量化、自适应阈值
- `shape_match.features`：前景分割、最大分散度特征点采样
- `shape_match.transforms`：几何旋转缩放、多边形 IoU、卷积核构建
- `shape_match.response_maps`：8 通道方向响应图与空间扩散
- `shape_match.matcher`：粗精分层搜索、外观验证与 NMS 抑制
- `shape_match.visualization`：特征点与匹配框渲染

