# Pattern Match API 接口文档

`pattern_match` 模块提供两个公开函数：模型形状可视化和 shape-based template matching。输入图像均为 `numpy.ndarray`，支持二维灰度、BGR 和 BGRA 格式。

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
| `angle_start` | `float` | `-5.0` | `[-360, 360]`；搜索起始角度 |
| `angle_extent` | `float` | `10.0` | `[0, 360]`；搜索范围，0 表示固定角度 |
| `num_levels` | `int` | `1` | `0`：全分辨率搜索；`1`：半分辨率粗搜索后原图细化 |

`angle_start + angle_extent` 不得超过 360。范围为 360° 时不会重复首尾姿态。

### `match_config`

| 字段 | 类型 | 默认值 | 有效范围及说明 |
|---|---|---:|---|
| `numMatches` | `int` | `5` | 大于等于 1；NMS 后最多返回的结果数 |
| `minScore` | `float` | `0.15` | `[0.0, 1.0]`；最终匹配最低得分 |
| `scale_min` | `float` | `1.0` | 大于 0；最小搜索尺度 |
| `scale_max` | `float` | `1.0` | 大于等于 `scale_min`；最大搜索尺度 |

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
    "numMatches": 5,
    "minScore": 0.3,
    "scale_min": 0.8,
    "scale_max": 1.2,
}

model_view = get_model_shape(model, pat_config)
results, result_view = get_matched_result(model, source, pat_config, match_config)

for cx, cy, score, angle, scale in results:
    print(cx, cy, score, angle, scale)
```

模板外框越紧、强边缘越属于目标本身，匹配通常越稳定。大范围角度与尺度组合会增加搜索时间。
