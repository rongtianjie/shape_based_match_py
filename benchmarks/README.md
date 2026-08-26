# Benchmarks

本目录包含 `shape_based_match_py` 算法的基准测试、微基准与性能画像分析脚本。

## 脚本说明

### 1. `benchmark_default.py`
默认狭角单目标搜索场景下的非门控端到端匹配速度基准测试。
```bash
python benchmarks/benchmark_default.py
```

### 2. `benchmark_profile.py`
针对 `TemplateModel` 模板特征构建与 `ShapeMatcher.match` 全流程匹配的 `cProfile` 性能热点画像分析脚本。
```bash
python benchmarks/benchmark_profile.py
```

### 3. `benchmark_refine.py`
针对候选位姿精细化（vectorized refine）多维数组广播与响应图采样的纯矩阵运算微基准。
```bash
python benchmarks/benchmark_refine.py
```
