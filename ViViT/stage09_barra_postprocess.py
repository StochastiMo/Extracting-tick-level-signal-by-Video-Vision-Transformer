"""ViViT 因子的 CNE5S 暴露分析与 Barra 截面残差化。"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from stage02_market_data import (
    _attach_reversal_20d,
    _check_match_rate,
    _fingerprint_parquet,
    _join_prediction_and_cne5s,
    _load_cne5s_year,
    _load_sse_calendar,
    _match_record,
    _standardized_reversal,
    _validate_cne5s_schema,
    _validate_price_sources,
    instrument_ids_to_assetids,
)

# 读取 `Path(__file__).resolve().parent` 的当前值，并赋给 `VIVIT_DIR`。
VIVIT_DIR = Path(__file__).resolve().parent
# 读取 `VIVIT_DIR.parent` 的当前值，并赋给 `PROJECT_ROOT`。
PROJECT_ROOT = VIVIT_DIR.parent
# 计算 `VIVIT_DIR / 'data'`，并将数值结果保存到 `DEFAULT_VIVIT_DATA_ROOT`。
DEFAULT_VIVIT_DATA_ROOT = VIVIT_DIR / "data"
# 计算 `PROJECT_ROOT / 'CNE5S.parquet'`，并将数值结果保存到 `DEFAULT_CNE5S_PATH`。
DEFAULT_CNE5S_PATH = PROJECT_ROOT / "CNE5S.parquet"
# 计算 `PROJECT_ROOT / 'Wind' / 'ASHAREEODPRICES.parquet'`，并将数值结果保存到 `DEFAULT_EOD_PATH`。
DEFAULT_EOD_PATH = PROJECT_ROOT / "Wind" / "ASHAREEODPRICES.parquet"
# 计算 `PROJECT_ROOT / 'Wind' / 'ASHARECALENDAR.parquet'`，并将数值结果保存到 `DEFAULT_CALENDAR_PATH`。
DEFAULT_CALENDAR_PATH = PROJECT_ROOT / "Wind" / "ASHARECALENDAR.parquet"
# 设置 `DEFAULT_SOURCE_RUN_NAME` 的固定配置值，供后续流程统一使用。
DEFAULT_SOURCE_RUN_NAME = "sigmoid_daybatch_rankic"
# 计算 `f'{DEFAULT_SOURCE_RUN_NAME}_barra_residual'`，并将结果写入 `DEFAULT_RESIDUAL_RUN_NAME`。
DEFAULT_RESIDUAL_RUN_NAME = f"{DEFAULT_SOURCE_RUN_NAME}_barra_residual"
# 设置 `PREDICTION_PATTERN` 的固定配置值，供后续流程统一使用。
PREDICTION_PATTERN = "vivit_predictions.{year}.parquet"
# 设置 `SOURCE_FACTOR_COLUMN` 的固定配置值，供后续流程统一使用。
SOURCE_FACTOR_COLUMN = "prediction_ensemble"
# 组织一组有序值，并保存到 `SOURCE_FACTOR_COLUMNS`。
SOURCE_FACTOR_COLUMNS = (
    "prediction_ensemble",
    "prediction_ensemble_unprojected",
    "prediction_ensemble_projected",
    "prediction_ensemble_raw",
    "prediction_ensemble_alpha",
    "prediction_ensemble_style",
    "prediction_ensemble_total_return",
)
# 设置 `INDUSTRY_COLUMN` 的固定配置值，供后续流程统一使用。
INDUSTRY_COLUMN = "cne5s_industry"
# 设置 `UNKNOWN_INDUSTRY` 的固定配置值，供后续流程统一使用。
UNKNOWN_INDUSTRY = "CNE5S_UNKNOWN"
# 设置 `MINIMUM_MATCH_RATE` 的固定配置值，供后续流程统一使用。
MINIMUM_MATCH_RATE = 0.999
# 设置 `DERIVATION_SCHEMA_VERSION` 的固定配置值，供后续流程统一使用。
DERIVATION_SCHEMA_VERSION = 5
# 设置 `HAC_MAX_LAG` 的固定配置值，供后续流程统一使用。
HAC_MAX_LAG = 20
# 设置 `ORTHOGONALITY_TOLERANCE` 的固定配置值，供后续流程统一使用。
ORTHOGONALITY_TOLERANCE = 1e-8
# 设置 `REVERSAL_COLUMN` 的固定配置值，供后续流程统一使用。
REVERSAL_COLUMN = "reversal_20d"
# 设置 `REVERSAL_VALID_COLUMN` 的固定配置值，供后续流程统一使用。
REVERSAL_VALID_COLUMN = "reversal_20d_valid"
# 设置 `REVERSAL_LOOKBACK_DAYS` 的固定配置值，供后续流程统一使用。
REVERSAL_LOOKBACK_DAYS = 20

# 组织一组有序值，并保存到 `STYLE_COLUMNS`。
STYLE_COLUMNS = (
    "cne5s_beta",
    "cne5s_btop",
    "cne5s_earnyild",
    "cne5s_growth",
    "cne5s_leverage",
    "cne5s_liquidty",
    "cne5s_momentum",
    "cne5s_size",
    "cne5s_sizenl",
)
# 构造键值映射，并保存到 `STYLE_LABELS`，供后续按键查询。
STYLE_LABELS = {
    "cne5s_beta": "贝塔",
    "cne5s_btop": "账面市值比",
    "cne5s_earnyild": "盈利收益率",
    "cne5s_growth": "成长",
    "cne5s_leverage": "杠杆",
    "cne5s_liquidty": "流动性",
    "cne5s_momentum": "动量",
    "cne5s_size": "市值",
    "cne5s_sizenl": "非线性市值",
    REVERSAL_COLUMN: "20日反转",
}
# 组织一组有序值，并保存到 `CORRELATION_COLUMNS`。
CORRELATION_COLUMNS = (*STYLE_COLUMNS, REVERSAL_COLUMN)
# 读取 `CORRELATION_COLUMNS` 的当前值，并赋给 `REGRESSION_COLUMNS`。
REGRESSION_COLUMNS = CORRELATION_COLUMNS
# 设置 `VIVIT_MATRIX_COLUMN` 的固定配置值，供后续流程统一使用。
VIVIT_MATRIX_COLUMN = "vivit_factor"
# 组织一组有序值，并保存到 `MATRIX_COLUMNS`。
MATRIX_COLUMNS = (VIVIT_MATRIX_COLUMN, *CORRELATION_COLUMNS)
# 构造键值映射，并保存到 `MATRIX_LABELS`，供后续按键查询。
MATRIX_LABELS = {
    VIVIT_MATRIX_COLUMN: "ViViT因子",
    **STYLE_LABELS,
}
# 组织一组有序值，并保存到 `REQUIRED_CNE5S_COLUMNS`。
REQUIRED_CNE5S_COLUMNS = (
    "datadate",
    "assetid",
    INDUSTRY_COLUMN,
    *STYLE_COLUMNS,
)
# 调用 `re.compile` 编译正则表达式供后续重复匹配，并将结果保存到 `INSTRUMENT_PATTERN`。
INSTRUMENT_PATTERN = re.compile(r"^(SH|SZ|BJ)\.(\d{6})$")
# 调用 `re.compile` 编译正则表达式供后续重复匹配，并将结果保存到 `CNE5S_ASSET_PATTERN`。
CNE5S_ASSET_PATTERN = re.compile(r"^CN\d{6}$")


# 定义内部辅助函数 `_validate_run_name`，封装仅供本模块复用的计算步骤。
def _validate_run_name(value: str) -> str:
    """校验输入形状、业务键、有限性和签名约束；异常时明确失败。
    
    输入：
        value: str。
    输出：
        str。
        """
    # 调用 `str(value).strip` 去除字符串首尾空白以统一配置值，并将结果保存到 `run_name`。
    run_name = str(value).strip()
    # 调用 `set` 整理数据结构，并将结果保存到 `allowed`。
    allowed = set(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    )
    # 检查 `not run_name or run_name in {'.', '..'} or any((character not in allowed for character in run_n…`；根据条件结果选择对应处理分支。
    if (
        not run_name
        or run_name in {".", ".."}
        or any(character not in allowed for character in run_name)
    ):
        # 抛出 `ValueError('运行名含非法字符')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("运行名含非法字符")
    # 将 `run_name` 作为当前函数结果返回给调用方。
    return run_name


# 定义内部辅助函数 `_validate_source_factor_column`，封装仅供本模块复用的计算步骤。
def _validate_source_factor_column(value: str) -> str:
    """校验输入形状、业务键、有限性和签名约束；异常时明确失败。
    
    输入：
        value: str。
    输出：
        str。
        """
    # 调用 `str(value).strip` 去除字符串首尾空白以统一配置值，并将结果保存到 `factor_column`。
    factor_column = str(value).strip()
    # 检查 `factor_column not in SOURCE_FACTOR_COLUMNS`；根据条件结果选择对应处理分支。
    if factor_column not in SOURCE_FACTOR_COLUMNS:
        # 抛出 `ValueError(f'未知源预测列：{value!r}；可选 {SOURCE_FACTOR_COLUMNS}')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(
            f"未知源预测列：{value!r}；可选 {SOURCE_FACTOR_COLUMNS}"
        )
    # 将 `factor_column` 作为当前函数结果返回给调用方。
    return factor_column


# 定义内部辅助函数 `_read_json`，封装仅供本模块复用的计算步骤。
def _read_json(path: Path) -> dict[str, Any]:
    """读取对应产物，并按接口约定处理缺失文件。
    
    输入：
        path: Path。
    输出：
        dict[str, Any]。
        """
    # 检查 `not path.exists()`；根据条件结果选择对应处理分支。
    if not path.exists():
        # 抛出 `FileNotFoundError(path)`，阻止不满足数据契约的流程继续执行。
        raise FileNotFoundError(path)
    # 将 `json.loads(path.read_text(encoding='utf-8'))` 作为当前函数结果返回给调用方。
    return json.loads(path.read_text(encoding="utf-8"))


# 定义内部辅助函数 `_atomic_json`，封装仅供本模块复用的计算步骤。
def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    """通过临时文件和原子替换写入产物，避免产生半成品。
    
    输入：
        payload: dict[str, Any]。
        path: Path。
    输出：
        None。
        """
    # 调用 `path.parent.mkdir` 创建输出目录，确保后续文件可以安全写入。
    path.parent.mkdir(parents=True, exist_ok=True)
    # 调用 `path.with_name` 规范化文件路径，并将结果保存到 `temporary`。
    temporary = path.with_name(
        f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    # 调用 `temporary.write_text` 将当前结果持久化到目标文件。
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    # 调用 `temporary.replace` 用完整临时文件原子替换正式产物。
    temporary.replace(path)


# 定义内部辅助函数 `_atomic_parquet`，封装仅供本模块复用的计算步骤。
def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    """通过临时文件和原子替换写入产物，避免产生半成品。
    
    输入：
        frame: pd.DataFrame。
        path: Path。
    输出：
        None。
        """
    # 调用 `path.parent.mkdir` 创建输出目录，确保后续文件可以安全写入。
    path.parent.mkdir(parents=True, exist_ok=True)
    # 调用 `path.with_name` 规范化文件路径，并将结果保存到 `temporary`。
    temporary = path.with_name(
        f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    # 调用 `frame.to_parquet` 将当前结果持久化到目标文件。
    frame.to_parquet(temporary, index=False)
    # 调用 `temporary.replace` 用完整临时文件原子替换正式产物。
    temporary.replace(path)




# 定义内部辅助函数 `_payload_signature`，封装仅供本模块复用的计算步骤。
def _payload_signature(payload: dict[str, Any]) -> str:
    """构造参与稳定签名和复用判断的配置载荷。
    
    输入：
        payload: dict[str, Any]。
    输出：
        str。
        """
    # 调用 `json.dumps(payload, ensure_ascii=False, sort_keys=True, separat…` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `encoded`。
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    # 将 `hashlib.sha256(encoded).hexdigest()` 作为当前函数结果返回给调用方。
    return hashlib.sha256(encoded).hexdigest()




# 定义内部辅助函数 `_training_manifest`，封装仅供本模块复用的计算步骤。
def _training_manifest(
    source_run_name: str, vivit_root: Path
) -> tuple[dict[str, Any], Path]:
    """执行该模式下的训练步骤，并返回损失与诊断信息。
    
    输入：
        source_run_name: str。
        vivit_root: Path。
    输出：
        tuple[dict[str, Any], Path]。
        """
    # 计算 `vivit_root / 'quality' / 'runs' / source_run_name / 'a2_training_manifest.json'`，并将数值结果保存到 `path`。
    path = (
        vivit_root
        / "quality"
        / "runs"
        / source_run_name
        / "a2_training_manifest.json"
    )
    # 调用 `_read_json` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `manifest`。
    manifest = _read_json(path)
    # 调用 `tuple` 整理数据结构，并将结果保存到 `years`。
    years = tuple(int(year) for year in manifest.get("years", ()))
    # 调用 `manifest.get` 按键读取映射中的可选字段，并在缺失时使用约定默认值，并将结果保存到 `counts`。
    counts = manifest.get("factor_counts")
    # 检查 `not years or not isinstance(counts, dict)`；根据条件结果选择对应处理分支。
    if not years or not isinstance(counts, dict):
        # 抛出 `ValueError('源训练 manifest 缺少 years 或 factor_counts')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("源训练 manifest 缺少 years 或 factor_counts")
    # 遍历 `years`，并把当前元素绑定到 `year` 逐项处理。
    for year in years:
        # 检查 `str(year) not in counts and year not in counts`；根据条件结果选择对应处理分支。
        if str(year) not in counts and year not in counts:
            # 抛出 `ValueError(f'源训练 manifest 缺少 {year} factor_counts')`，阻止不满足数据契约的流程继续执行。
            raise ValueError(f"源训练 manifest 缺少 {year} factor_counts")
    # 将 `(manifest, path)` 作为当前函数结果返回给调用方。
    return manifest, path


# 定义内部辅助函数 `_factor_counts`，封装仅供本模块复用的计算步骤。
def _factor_counts(manifest: dict[str, Any]) -> dict[int, int]:
    """读取或汇总因子评价所需的数据。
    
    输入：
        manifest: dict[str, Any]。
    输出：
        dict[int, int]。
        """
    # 读取 `manifest['factor_counts']` 的当前值，并赋给 `raw`。
    raw = manifest["factor_counts"]
    # 将 `{int(year): int(raw.get(str(year), raw.get(year))) for year in manifest['years']}` 作为当前函数结果返回给调用方。
    return {
        int(year): int(raw.get(str(year), raw.get(year)))
        for year in manifest["years"]
    }
















# 定义内部辅助函数 `_load_source_prediction`，封装仅供本模块复用的计算步骤。
def _load_source_prediction(
    path: Path,
    year: int,
    expected_rows: int,
    *,
    full: bool,
    source_factor_column: str = SOURCE_FACTOR_COLUMN,
) -> pd.DataFrame:
    """加载该流程所需数据，并检查形状、业务键和来源一致性。
    
    输入：
        path: Path。
        year: int。
        expected_rows: int。
        full: bool。
        source_factor_column: str。
    输出：
        pd.DataFrame。
        """
    # 调用 `_validate_source_factor_column` 校验输入、配置或产物是否满足当前阶段的数据契约，并将结果保存到 `source_factor_column`。
    source_factor_column = _validate_source_factor_column(
        source_factor_column
    )
    # 调用 `pq.ParquetFile` 打开 Parquet 元数据以便分批读取，并将结果保存到 `parquet`。
    parquet = pq.ParquetFile(path)
    # 读取 `parquet.schema_arrow.names` 的当前值，并赋给 `columns`。
    columns = parquet.schema_arrow.names
    # 构造去重集合，并保存到 `required`。
    required = {"trade_date", "instrument_id", source_factor_column}
    # 调用 `required.difference` 计算两个集合的差集以找出缺失字段，并将结果保存到 `missing`。
    missing = required.difference(columns)
    # 检查 `missing`；根据条件结果选择对应处理分支。
    if missing:
        # 抛出 `ValueError(f'{path.name} 缺少字段：{sorted(missing)}')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(f"{path.name} 缺少字段：{sorted(missing)}")
    # 按 `full` 在两个候选值间选择，并保存到 `selected`。
    selected = list(columns) if full else list(required)
    # 调用 `pd.read_parquet` 读取源数据，并将结果保存到 `prediction`。
    prediction = pd.read_parquet(path, columns=selected)
    # 检查 `len(prediction) != expected_rows`；根据条件结果选择对应处理分支。
    if len(prediction) != expected_rows:
        # 抛出 `AssertionError(f'{year} 预测行数 {len(prediction):,} != {expected_rows:,}')`，阻止不满足数据契约的流程继续执行。
        raise AssertionError(
            f"{year} 预测行数 {len(prediction):,} != {expected_rows:,}"
        )
    # 调用 `prediction['trade_date'].astype` 统一设备或数据类型，并将结果保存到 `prediction['trade_date']`。
    prediction["trade_date"] = prediction["trade_date"].astype(str)
    # 调用 `prediction['instrument_id'].astype` 统一设备或数据类型，并将结果保存到 `prediction['instrument_id']`。
    prediction["instrument_id"] = prediction["instrument_id"].astype(str)
    # 检查 `prediction.duplicated(['trade_date', 'instrument_id']).any()`；根据条件结果选择对应处理分支。
    if prediction.duplicated(["trade_date", "instrument_id"]).any():
        # 抛出 `ValueError(f'{year} 源预测业务键重复')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(f"{year} 源预测业务键重复")
    # 检查 `not prediction['trade_date'].str.startswith(str(year)).all()`；根据条件结果选择对应处理分支。
    if not prediction["trade_date"].str.startswith(str(year)).all():
        # 抛出 `ValueError(f'{year} 源预测包含跨年度日期')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(f"{year} 源预测包含跨年度日期")
    # 调用 `prediction[source_factor_column].to_numpy` 执行模型推理并生成样本预测分数，并将结果保存到 `factor`。
    factor = prediction[source_factor_column].to_numpy(np.float64)
    # 检查 `not np.isfinite(factor).all()`；根据条件结果选择对应处理分支。
    if not np.isfinite(factor).all():
        # 抛出 `ValueError(f'{year} 源因子含 NaN 或 Inf')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(f"{year} 源因子含 NaN 或 Inf")
    # 下游统一读取内部标准列；这里只修改内存副本，不写回源预测。
    prediction[SOURCE_FACTOR_COLUMN] = factor
    # 调用 `instrument_ids_to_assetids` 执行 `instrument_ids_to_assetids` 对应的明确变换，并产生后续步骤使用的结果，并将结果保存到 `prediction['assetid']`。
    prediction["assetid"] = instrument_ids_to_assetids(
        prediction["instrument_id"]
    )
    # 将 `prediction` 作为当前函数结果返回给调用方。
    return prediction












# 定义内部辅助函数 `_pearson`，封装仅供本模块复用的计算步骤。
def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """计算有效样本上的 Pearson 相关系数。
    
    输入：
        x: np.ndarray。
        y: np.ndarray。
    输出：
        float。
        """
    # 调用 `np.asarray` 转换数据表示，并将标准化对象保存到 `x`。
    x = np.asarray(x, dtype=np.float64)
    # 调用 `np.asarray` 转换数据表示，并将标准化对象保存到 `y`。
    y = np.asarray(y, dtype=np.float64)
    # 检查 `len(x) < 3 or len(x) != len(y)`；根据条件结果选择对应处理分支。
    if len(x) < 3 or len(x) != len(y):
        # 抛出 `ValueError('Pearson 至少需要 3 个等长观测')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("Pearson 至少需要 3 个等长观测")
    # 计算 `x - x.mean()`，并将数值结果保存到 `x_centered`。
    x_centered = x - x.mean()
    # 计算 `y - y.mean()`，并将数值结果保存到 `y_centered`。
    y_centered = y - y.mean()
    # 调用 `float` 统一设备或数据类型，并将结果保存到 `denominator`。
    denominator = float(
        np.sqrt(np.dot(x_centered, x_centered) * np.dot(y_centered, y_centered))
    )
    # 检查 `not np.isfinite(denominator) or denominator <= np.finfo(np.float64).eps`；根据条件结果选择对应处理分支。
    if not np.isfinite(denominator) or denominator <= np.finfo(np.float64).eps:
        # 抛出 `ValueError('Pearson 输入为常数或非有限')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("Pearson 输入为常数或非有限")
    # 将 `float(np.dot(x_centered, y_centered) / denominator)` 作为当前函数结果返回给调用方。
    return float(np.dot(x_centered, y_centered) / denominator)


# 定义内部辅助函数 `_complete_correlation_sample`，封装仅供本模块复用的计算步骤。
def _complete_correlation_sample(joined: pd.DataFrame) -> pd.DataFrame:
    """返回 11×11 相关矩阵统一使用的完整同日样本。
    
    输入：
        joined: pd.DataFrame。
    输出：
        pd.DataFrame。
        """

    # 将 `joined.loc[joined['barra_matched'] & joined[REVERSAL_VALID_COLUMN]]` 作为当前函数结果返回给调用方。
    return joined.loc[
        joined["barra_matched"] & joined[REVERSAL_VALID_COLUMN]
    ]


# 定义内部辅助函数 `_daily_correlations`，封装仅供本模块复用的计算步骤。
def _daily_correlations(
    joined: pd.DataFrame, year: int
) -> list[dict[str, Any]]:
    """汇总截面相关性及其时间序列统计。
    
    输入：
        joined: pd.DataFrame。
        year: int。
    输出：
        list[dict[str, Any]]。
        """
    # 构造当前步骤所需的列表，并保存到 `records`。
    records: list[dict[str, Any]] = []
    # 调用 `_complete_correlation_sample` 计算预测与标签或风险暴露之间的相关性，并将结果保存到 `complete`。
    complete = _complete_correlation_sample(joined)
    # 遍历 `complete.groupby('trade_date', sort=True)`，并把当前元素绑定到 `(trade_date, daily)` 逐项处理。
    for trade_date, daily in complete.groupby("trade_date", sort=True):
        # 调用 `daily[SOURCE_FACTOR_COLUMN].to_numpy` 把表格列转换为指定精度的 NumPy 数组，并将结果保存到 `y`。
        y = daily[SOURCE_FACTOR_COLUMN].to_numpy(np.float64)
        # 遍历 `CORRELATION_COLUMNS`，并把当前元素绑定到 `column` 逐项处理。
        for column in CORRELATION_COLUMNS:
            # 把本轮结果追加到 `records`，累积后续需要汇总的记录。
            records.append(
                {
                    "year": year,
                    "trade_date": str(trade_date),
                    "factor_name": column,
                    "factor_label": STYLE_LABELS[column],
                    "pearson": _pearson(
                        y, daily[column].to_numpy(np.float64)
                    ),
                    "observations": len(daily),
                }
            )
    # 将 `records` 作为当前函数结果返回给调用方。
    return records


# 定义内部辅助函数 `_daily_correlation_matrix`，封装仅供本模块复用的计算步骤。
def _daily_correlation_matrix(
    joined: pd.DataFrame, year: int
) -> list[dict[str, Any]]:
    """计算 ViViT、九个 CNE5S 风格和反转的日度 11×11 Pearson。
    
    输入：
        joined: pd.DataFrame。
        year: int。
    输出：
        list[dict[str, Any]]。
        """

    # 构造当前步骤所需的列表，并保存到 `records`。
    records: list[dict[str, Any]] = []
    # 调用 `_complete_correlation_sample` 计算预测与标签或风险暴露之间的相关性，并将结果保存到 `complete`。
    complete = _complete_correlation_sample(joined)
    # 遍历 `complete.groupby('trade_date', sort=True)`，并把当前元素绑定到 `(trade_date, daily)` 逐项处理。
    for trade_date, daily in complete.groupby("trade_date", sort=True):
        # 调用 `np.column_stack` 按列拼接多个一维因子数组，并将结果保存到 `values`。
        values = np.column_stack(
            [
                daily[SOURCE_FACTOR_COLUMN].to_numpy(np.float64),
                *(
                    daily[column].to_numpy(np.float64)
                    for column in CORRELATION_COLUMNS
                ),
            ]
        )
        # 检查 `len(daily) < 3 or not np.isfinite(values).all()`；根据条件结果选择对应处理分支。
        if len(daily) < 3 or not np.isfinite(values).all():
            # 抛出 `ValueError(f'{trade_date} 相关矩阵样本不足或含非有限值')`，阻止不满足数据契约的流程继续执行。
            raise ValueError(f"{trade_date} 相关矩阵样本不足或含非有限值")
        # 调用 `np.corrcoef` 计算各因子列之间的 Pearson 相关矩阵，并将结果保存到 `matrix`。
        matrix = np.corrcoef(values, rowvar=False)
        # 组织一组有序值，并保存到 `expected_shape`。
        expected_shape = (len(MATRIX_COLUMNS), len(MATRIX_COLUMNS))
        # 检查 `matrix.shape != expected_shape or not np.isfinite(matrix).all()`；根据条件结果选择对应处理分支。
        if matrix.shape != expected_shape or not np.isfinite(matrix).all():
            # 抛出 `ValueError(f'{trade_date} 相关矩阵不可计算')`，阻止不满足数据契约的流程继续执行。
            raise ValueError(f"{trade_date} 相关矩阵不可计算")
        # 检查 `not np.allclose(matrix, matrix.T, atol=1e-12, rtol=0.0)`；根据条件结果选择对应处理分支。
        if not np.allclose(matrix, matrix.T, atol=1e-12, rtol=0.0):
            # 抛出 `AssertionError(f'{trade_date} 相关矩阵不对称')`，阻止不满足数据契约的流程继续执行。
            raise AssertionError(f"{trade_date} 相关矩阵不对称")
        # 检查 `not np.allclose(np.diag(matrix), 1.0, atol=1e-12, rtol=0.0)`；根据条件结果选择对应处理分支。
        if not np.allclose(
            np.diag(matrix), 1.0, atol=1e-12, rtol=0.0
        ):
            # 抛出 `AssertionError(f'{trade_date} 相关矩阵对角线不为 1')`，阻止不满足数据契约的流程继续执行。
            raise AssertionError(f"{trade_date} 相关矩阵对角线不为 1")
        # 遍历 `enumerate(MATRIX_COLUMNS)`，并把当前元素绑定到 `(row_index, row_name)` 逐项处理。
        for row_index, row_name in enumerate(MATRIX_COLUMNS):
            # 遍历 `enumerate(MATRIX_COLUMNS)`，并把当前元素绑定到 `(column_index, column_name)` 逐项处理。
            for column_index, column_name in enumerate(MATRIX_COLUMNS):
                # 把本轮结果追加到 `records`，累积后续需要汇总的记录。
                records.append(
                    {
                        "year": year,
                        "trade_date": str(trade_date),
                        "row_name": row_name,
                        "row_label": MATRIX_LABELS[row_name],
                        "column_name": column_name,
                        "column_label": MATRIX_LABELS[column_name],
                        "pearson": float(matrix[row_index, column_index]),
                        "observations": len(daily),
                    }
                )
    # 将 `records` 作为当前函数结果返回给调用方。
    return records

# 定义内部辅助函数 `_neutralize_cross_section`，封装仅供本模块复用的计算步骤。
def _neutralize_cross_section(
    daily: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    """以行业固定效应、9 个 CNE5S 风格和 20 日反转残差化。
    
    输入：
        daily: pd.DataFrame。
    输出：
        tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]。
        """

    # 调用 `daily[SOURCE_FACTOR_COLUMN].to_numpy` 把表格列转换为指定精度的 NumPy 数组，并将结果保存到 `y`。
    y = daily[SOURCE_FACTOR_COLUMN].to_numpy(np.float64)
    # 检查 `not np.isfinite(y).all()`；根据条件结果选择对应处理分支。
    if not np.isfinite(y).all():
        # 抛出 `ValueError('单日源因子含 NaN 或 Inf')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("单日源因子含 NaN 或 Inf")
    # 调用 `daily['barra_matched'].to_numpy` 把表格列转换为指定精度的 NumPy 数组，并将结果保存到 `matched`。
    matched = daily["barra_matched"].to_numpy(bool)
    # 调用 `np.zeros` 按目标形状初始化数值容器 `x`。
    x = np.zeros((len(daily), len(REGRESSION_COLUMNS)), dtype=np.float64)
    # 调用 `daily.loc[:, STYLE_COLUMNS].to_numpy` 把表格列转换为指定精度的 NumPy 数组，并将结果保存到 `style_values`。
    style_values = daily.loc[:, STYLE_COLUMNS].to_numpy(
        np.float64, copy=True
    )
    # 设置 `style_values[~matched, :]` 的固定配置值，供后续流程统一使用。
    style_values[~matched, :] = 0.0
    # 读取 `style_values` 的当前值，并赋给 `x[:, :len(STYLE_COLUMNS)]`。
    x[:, : len(STYLE_COLUMNS)] = style_values
    # 调用 `_standardized_reversal` 按当前截面统计量完成缩尾或标准化处理，并将结果保存到 `x[:, -1]`。
    x[:, -1] = _standardized_reversal(daily)

    # 复制 `daily[INDUSTRY_COLUMN].astype('object').copy`，避免后续更新污染原对象，并保存到 `industries`。
    industries = daily[INDUSTRY_COLUMN].astype("object").copy()
    # 读取 `UNKNOWN_INDUSTRY` 的当前值，并赋给 `industries.loc[~daily['barra_matched']]`。
    industries.loc[~daily["barra_matched"]] = UNKNOWN_INDUSTRY
    # 检查 `industries.isna().any() or not np.isfinite(x).all()`；根据条件结果选择对应处理分支。
    if industries.isna().any() or not np.isfinite(x).all():
        # 抛出 `ValueError('中性化输入含未处理的缺失值')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("中性化输入含未处理的缺失值")

    # 调用 `pd.factorize` 把行业字符串编码为连续整数，并将结果保存到 `(codes, labels)`。
    codes, labels = pd.factorize(industries.astype(str), sort=True)
    # 调用 `np.bincount(codes).astype` 统一设备或数据类型，并将结果保存到 `counts`。
    counts = np.bincount(codes).astype(np.float64)
    # 计算 `np.bincount(codes, weights=y) / counts`，并将数值结果保存到 `y_means`。
    y_means = np.bincount(codes, weights=y) / counts
    # 调用 `np.zeros` 按目标形状初始化数值容器 `x_sums`。
    x_sums = np.zeros(
        (len(labels), len(REGRESSION_COLUMNS)), dtype=np.float64
    )
    # 调用 `np.add.at` 按行业编码原地累加截面统计量。
    np.add.at(x_sums, codes, x)
    # 计算 `x_sums / counts[:, None]`，并将数值结果保存到 `x_means`。
    x_means = x_sums / counts[:, None]
    # 计算 `y - y_means[codes]`，并将数值结果保存到 `y_within`。
    y_within = y - y_means[codes]
    # 计算 `x - x_means[codes]`，并将数值结果保存到 `x_within`。
    x_within = x - x_means[codes]

    # 调用 `np.linalg.lstsq` 求解截面最小二乘回归系数，并将结果保存到 `(coefficients, _, rank, singular_values)`。
    coefficients, _, rank, singular_values = np.linalg.lstsq(
        x_within, y_within, rcond=None
    )
    # 调用 `len` 汇总当前数据，并将统计量保存到 `expected_rank`。
    expected_rank = len(REGRESSION_COLUMNS)
    # 检查 `int(rank) != expected_rank`；根据条件结果选择对应处理分支。
    if int(rank) != expected_rank:
        # 抛出 `np.linalg.LinAlgError(f'风格暴露矩阵秩为 {rank}，预期 {expected_rank}')`，阻止不满足数据契约的流程继续执行。
        raise np.linalg.LinAlgError(
            f"风格暴露矩阵秩为 {rank}，预期 {expected_rank}"
        )
    # 检查 `len(singular_values) != expected_rank or singular_values[-1] <= np.finfo(np.float64).eps`；根据条件结果选择对应处理分支。
    if (
        len(singular_values) != expected_rank
        or singular_values[-1] <= np.finfo(np.float64).eps
    ):
        # 抛出 `np.linalg.LinAlgError('风格暴露矩阵条件数不可计算')`，阻止不满足数据契约的流程继续执行。
        raise np.linalg.LinAlgError("风格暴露矩阵条件数不可计算")
    # 计算 `y_within - x_within @ coefficients`，并将数值结果保存到 `residual`。
    residual = y_within - x_within @ coefficients
    # 检查 `not np.isfinite(residual).all()`；根据条件结果选择对应处理分支。
    if not np.isfinite(residual).all():
        # 抛出 `ValueError('回归残差含 NaN 或 Inf')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("回归残差含 NaN 或 Inf")
    # 调用 `float` 统一设备或数据类型，并将结果保存到 `source_variance`。
    source_variance = float(np.dot(y - y.mean(), y - y.mean()))
    # 检查 `source_variance <= np.finfo(np.float64).eps`；根据条件结果选择对应处理分支。
    if source_variance <= np.finfo(np.float64).eps:
        # 抛出 `ValueError('单日源因子为常数')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("单日源因子为常数")
    # 调用 `float` 统一设备或数据类型，并将结果保存到 `residual_variance`。
    residual_variance = float(np.dot(residual, residual))
    # 调用 `float` 统一设备或数据类型，并将结果保存到 `residual_std`。
    residual_std = float(residual.std(ddof=0))
    # 检查 `residual_std <= np.finfo(np.float64).eps`；根据条件结果选择对应处理分支。
    if residual_std <= np.finfo(np.float64).eps:
        # 抛出 `ValueError('单日回归残差为常数')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("单日回归残差为常数")

    # 构造当前步骤所需的列表，并保存到 `residual_correlations`。
    residual_correlations: list[dict[str, Any]] = []
    # 遍历 `enumerate(REGRESSION_COLUMNS)`，并把当前元素绑定到 `(column_index, column)` 逐项处理。
    for column_index, column in enumerate(REGRESSION_COLUMNS):
        # 把本轮结果追加到 `residual_correlations`，累积后续需要汇总的记录。
        residual_correlations.append(
            {
                "factor_name": column,
                "factor_label": STYLE_LABELS[column],
                "pearson": _pearson(x[:, column_index], residual),
                "observations": len(daily),
                "valid_observations": (
                    int(daily[REVERSAL_VALID_COLUMN].sum())
                    if column == REVERSAL_COLUMN
                    else int(matched.sum())
                ),
            }
        )
    # 计算 `np.bincount(codes, weights=residual) / counts`，并将数值结果保存到 `group_residual_means`。
    group_residual_means = np.bincount(
        codes, weights=residual
    ) / counts
    # 调用 `max` 汇总当前数据，并将统计量保存到 `max_style_correlation`。
    max_style_correlation = max(
        abs(record["pearson"]) for record in residual_correlations
    )
    # 调用 `float` 统一设备或数据类型，并将结果保存到 `max_industry_mean`。
    max_industry_mean = float(np.max(np.abs(group_residual_means)))
    # 检查 `max_style_correlation > ORTHOGONALITY_TOLERANCE`；根据条件结果选择对应处理分支。
    if max_style_correlation > ORTHOGONALITY_TOLERANCE:
        # 抛出 `AssertionError(f'残差风格相关性 {max_style_correlation:.3e} 超过容差')`，阻止不满足数据契约的流程继续执行。
        raise AssertionError(
            f"残差风格相关性 {max_style_correlation:.3e} 超过容差"
        )
    # 检查 `max_industry_mean > ORTHOGONALITY_TOLERANCE`；根据条件结果选择对应处理分支。
    if max_industry_mean > ORTHOGONALITY_TOLERANCE:
        # 抛出 `AssertionError(f'行业残差均值 {max_industry_mean:.3e} 超过容差')`，阻止不满足数据契约的流程继续执行。
        raise AssertionError(
            f"行业残差均值 {max_industry_mean:.3e} 超过容差"
        )

    # 调用 `int` 统一设备或数据类型，并将结果保存到 `reversal_valid_rows`。
    reversal_valid_rows = int(daily[REVERSAL_VALID_COLUMN].sum())
    # 构造键值映射，并保存到 `diagnostics`，供后续按键查询。
    diagnostics = {
        "rows": len(daily),
        "matched_rows": int(matched.sum()),
        "unmatched_rows": int((~matched).sum()),
        "match_rate": float(matched.mean()),
        "reversal_valid_rows": reversal_valid_rows,
        "reversal_missing_rows": len(daily) - reversal_valid_rows,
        "reversal_coverage": float(reversal_valid_rows / len(daily)),
        "industries": int(len(labels)),
        "matrix_rank": int(rank),
        "condition_number": float(
            singular_values[0] / singular_values[-1]
        ),
        "r_squared": float(1.0 - residual_variance / source_variance),
        "source_factor_std": float(y.std(ddof=0)),
        "residual_factor_std": residual_std,
        "residual_factor_mean": float(residual.mean()),
        "max_abs_style_residual_corr": max_style_correlation,
        "max_abs_industry_residual_mean": max_industry_mean,
    }
    # 将 `(residual, diagnostics, residual_correlations)` 作为当前函数结果返回给调用方。
    return residual, diagnostics, residual_correlations


# 定义内部辅助函数 `_neutralize_year`，封装仅供本模块复用的计算步骤。
def _neutralize_year(
    joined: pd.DataFrame, year: int
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """逐日对年度原始分数回归十变量和行业固定效应，生成统一残差。
    
    输入：
        joined: pd.DataFrame。
        year: int。
    输出：
        tuple[pd.DataFrame, pd.DataFrame, np.ndarray]。
        """
    # 调用 `np.empty` 按目标形状初始化数值容器 `residual`。
    residual = np.empty(len(joined), dtype=np.float64)
    # 构造当前步骤所需的列表，并保存到 `diagnostic_records`。
    diagnostic_records: list[dict[str, Any]] = []
    # 构造当前步骤所需的列表，并保存到 `correlation_records`。
    correlation_records: list[dict[str, Any]] = []
    # 遍历 `joined.groupby('trade_date', sort=True).groups.items()`，并把当前元素绑定到 `(trade_date, positions)` 逐项处理。
    for trade_date, positions in joined.groupby(
        "trade_date", sort=True
    ).groups.items():
        # 调用 `np.asarray` 转换数据表示，并将标准化对象保存到 `index`。
        index = np.asarray(positions, dtype=np.int64)
        # 调用 `_neutralize_cross_section` 按 Barra 暴露执行截面投影或残差化处理，并将结果保存到 `(values, diagnostics, correlations)`。
        values, diagnostics, correlations = _neutralize_cross_section(
            joined.iloc[index]
        )
        # 读取 `values` 的当前值，并赋给 `residual[index]`。
        residual[index] = values
        # 把本轮结果追加到 `diagnostic_records`，累积后续需要汇总的记录。
        diagnostic_records.append(
            {
                "year": year,
                "trade_date": str(trade_date),
                **diagnostics,
            }
        )
        # 把一组结果展开追加到 `correlation_records`，保持统一的一维记录序列。
        correlation_records.extend(
            {
                "year": year,
                "trade_date": str(trade_date),
                **record,
            }
            for record in correlations
        )
    # 将 `(pd.DataFrame(diagnostic_records), pd.DataFrame(correlation_records), residual)` 作为当前函数结果返回给调用方。
    return (
        pd.DataFrame(diagnostic_records),
        pd.DataFrame(correlation_records),
        residual,
    )


# 定义内部辅助函数 `_newey_west_mean_t_stat`，封装仅供本模块复用的计算步骤。
def _newey_west_mean_t_stat(
    values: pd.Series, max_lag: int = HAC_MAX_LAG
) -> float:
    """以 Bartlett 核估计均值的 Newey-West HAC t 值。
    
    输入：
        values: pd.Series。
        max_lag: int。
    输出：
        float。
        """

    # 调用 `np.asarray` 转换数据表示，并将标准化对象保存到 `sample`。
    sample = np.asarray(values, dtype=np.float64)
    # 检查 `len(sample) < 2 or not np.isfinite(sample).all()`；根据条件结果选择对应处理分支。
    if len(sample) < 2 or not np.isfinite(sample).all():
        # 将 `np.nan` 作为当前函数结果返回给调用方。
        return np.nan
    # 计算 `sample - sample.mean()`，并将数值结果保存到 `centered`。
    centered = sample - sample.mean()
    # 调用 `min` 汇总当前数据，并将统计量保存到 `lag`。
    lag = min(max_lag, len(sample) - 1)
    # 调用 `float` 统一设备或数据类型，并将结果保存到 `long_run_variance`。
    long_run_variance = float(np.dot(centered, centered) / len(sample))
    # 遍历 `range(1, lag + 1)`，并把当前元素绑定到 `offset` 逐项处理。
    for offset in range(1, lag + 1):
        # 计算 `1.0 - offset / (lag + 1.0)`，并将数值结果保存到 `weight`。
        weight = 1.0 - offset / (lag + 1.0)
        # 调用 `float` 统一设备或数据类型，并将结果保存到 `autocovariance`。
        autocovariance = float(
            np.dot(centered[offset:], centered[:-offset]) / len(sample)
        )
        # 用 `2.0 * weight * autocovariance` 对 `long_run_variance` 做 `Add` 累积更新。
        long_run_variance += 2.0 * weight * autocovariance
    # 检查 `not np.isfinite(long_run_variance) or long_run_variance <= np.finfo(np.float64).eps`；根据条件结果选择对应处理分支。
    if (
        not np.isfinite(long_run_variance)
        or long_run_variance <= np.finfo(np.float64).eps
    ):
        # 将 `np.nan` 作为当前函数结果返回给调用方。
        return np.nan
    # 将 `float(sample.mean() / np.sqrt(long_run_variance / len(sample)))` 作为当前函数结果返回给调用方。
    return float(sample.mean() / np.sqrt(long_run_variance / len(sample)))


# 定义内部辅助函数 `_correlation_summaries`，封装仅供本模块复用的计算步骤。
def _correlation_summaries(
    daily: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """汇总截面相关性及其时间序列统计。
    
    输入：
        daily: pd.DataFrame。
    输出：
        tuple[pd.DataFrame, pd.DataFrame]。
        """
    # 定义 `summarize` 函数，封装这一阶段可独立调用的处理步骤。
    def summarize(group_columns: list[str]) -> pd.DataFrame:
        """汇总截面相关性及其时间序列统计。
        
        输入：
            group_columns: list[str]。
        输出：
            pd.DataFrame。
            """
        # 调用 `daily.sort_values('trade_date').groupby` 根据分数计算截面排序或排序索引，并将结果保存到 `frame`。
        frame = daily.sort_values("trade_date").groupby(
            group_columns, sort=True, observed=True
        )
        # 调用 `frame.agg(mean_pearson=('pearson', 'mean'), mean_abs_pearson=('…` 计算预测与标签或风险暴露之间的相关性，并将结果保存到 `result`。
        result = frame.agg(
            mean_pearson=("pearson", "mean"),
            mean_abs_pearson=("pearson", lambda values: values.abs().mean()),
            std_pearson=("pearson", "std"),
            dates=("trade_date", "nunique"),
            mean_observations=("observations", "mean"),
        ).reset_index()
        # 计算 `result['mean_pearson'] / result['std_pearson'] * np.sqrt(result['dates'])`，并将数值结果保存到 `result['t_stat']`。
        result["t_stat"] = (
            result["mean_pearson"]
            / result["std_pearson"]
            * np.sqrt(result["dates"])
        )
        # 调用 `frame['pearson'].agg(_newey_west_mean_t_stat).rename('t_stat_ne…` 计算预测与标签或风险暴露之间的相关性，并将结果保存到 `corrected`。
        corrected = (
            frame["pearson"]
            .agg(_newey_west_mean_t_stat)
            .rename("t_stat_newey_west_lag20")
            .reset_index()
        )
        # 将 `result.merge(corrected, on=group_columns, how='left', validate='one_to_one')` 作为当前函数结果返回给调用方。
        return result.merge(
            corrected,
            on=group_columns,
            how="left",
            validate="one_to_one",
        )

    # 调用 `summarize` 按约定截面或分组口径汇总统计结果，并将结果保存到 `annual`。
    annual = summarize(
        ["year", "factor_name", "factor_label"]
    )
    # 调用 `summarize` 按约定截面或分组口径汇总统计结果，并将结果保存到 `overall`。
    overall = summarize(
        ["factor_name", "factor_label"]
    )
    # 构造键值映射，并保存到 `order`，供后续按键查询。
    order = {
        column: index for index, column in enumerate(CORRELATION_COLUMNS)
    }
    # 调用 `annual['factor_name'].map` 按映射关系转换当前序列中的每个元素，并将结果保存到 `annual['_order']`。
    annual["_order"] = annual["factor_name"].map(order)
    # 调用 `overall['factor_name'].map` 按映射关系转换当前序列中的每个元素，并将结果保存到 `overall['_order']`。
    overall["_order"] = overall["factor_name"].map(order)
    # 调用 `annual.sort_values` 根据分数计算截面排序或排序索引。
    annual.sort_values(["year", "_order"], inplace=True)
    # 调用 `overall.sort_values` 根据分数计算截面排序或排序索引。
    overall.sort_values("_order", inplace=True)
    # 将 `(annual.drop(columns='_order').reset_index(drop=True), overall.drop(columns='_order').reset_ind…` 作为当前函数结果返回给调用方。
    return (
        annual.drop(columns="_order").reset_index(drop=True),
        overall.drop(columns="_order").reset_index(drop=True),
    )


# 定义内部辅助函数 `_correlation_matrix_summary`，封装仅供本模块复用的计算步骤。
def _correlation_matrix_summary(daily: pd.DataFrame) -> pd.DataFrame:
    """汇总截面相关性及其时间序列统计。
    
    输入：
        daily: pd.DataFrame。
    输出：
        pd.DataFrame。
        """
    # 计算 `len(MATRIX_COLUMNS) ** 2`，并将数值结果保存到 `expected_cells`。
    expected_cells = len(MATRIX_COLUMNS) ** 2
    # 调用 `daily.groupby('trade_date', sort=False).size` 统计每个交易日截面的样本数量，并将结果保存到 `cells_per_date`。
    cells_per_date = daily.groupby("trade_date", sort=False).size()
    # 检查 `daily.empty or not cells_per_date.eq(expected_cells).all()`；根据条件结果选择对应处理分支。
    if daily.empty or not cells_per_date.eq(expected_cells).all():
        # 抛出 `AssertionError('每个交易日必须包含完整的 11×11 相关矩阵')`，阻止不满足数据契约的流程继续执行。
        raise AssertionError("每个交易日必须包含完整的 11×11 相关矩阵")
    # 检查 `daily.duplicated(['trade_date', 'row_name', 'column_name']).any()`；根据条件结果选择对应处理分支。
    if daily.duplicated(
        ["trade_date", "row_name", "column_name"]
    ).any():
        # 抛出 `AssertionError('日度相关矩阵存在重复单元')`，阻止不满足数据契约的流程继续执行。
        raise AssertionError("日度相关矩阵存在重复单元")

    # 调用 `reset_index` 重建连续行索引并丢弃旧索引，并将结果保存到 `summary`。
    summary = (
        daily.groupby(
            ["row_name", "row_label", "column_name", "column_label"],
            sort=False,
            observed=True,
        )
        .agg(
            mean_pearson=("pearson", "mean"),
            std_pearson=("pearson", "std"),
            dates=("trade_date", "nunique"),
            mean_observations=("observations", "mean"),
        )
        .reset_index()
    )
    # 构造键值映射，并保存到 `order`，供后续按键查询。
    order = {name: index for index, name in enumerate(MATRIX_COLUMNS)}
    # 调用 `summary['row_name'].map` 按约定截面或分组口径汇总统计结果，并将结果保存到 `summary['_row_order']`。
    summary["_row_order"] = summary["row_name"].map(order)
    # 调用 `summary['column_name'].map` 按约定截面或分组口径汇总统计结果，并将结果保存到 `summary['_column_order']`。
    summary["_column_order"] = summary["column_name"].map(order)
    # 调用 `summary.sort_values` 按约定截面或分组口径汇总统计结果。
    summary.sort_values(
        ["_row_order", "_column_order"], inplace=True
    )
    # 调用 `summary.drop` 按约定截面或分组口径汇总统计结果。
    summary.drop(
        columns=["_row_order", "_column_order"], inplace=True
    )
    # 调用 `summary.reset_index` 按约定截面或分组口径汇总统计结果。
    summary.reset_index(drop=True, inplace=True)
    # 检查 `len(summary) != expected_cells`；根据条件结果选择对应处理分支。
    if len(summary) != expected_cells:
        # 抛出 `AssertionError('相关矩阵汇总不含完整的 121 个单元')`，阻止不满足数据契约的流程继续执行。
        raise AssertionError("相关矩阵汇总不含完整的 121 个单元")
    # 调用 `summary.pivot(index='row_name', columns='column_name', values='…` 按约定截面或分组口径汇总统计结果，并将结果保存到 `matrix`。
    matrix = (
        summary.pivot(
            index="row_name", columns="column_name", values="mean_pearson"
        )
        .loc[list(MATRIX_COLUMNS), list(MATRIX_COLUMNS)]
        .to_numpy(np.float64)
    )
    # 检查 `not np.allclose(matrix, matrix.T, atol=1e-12, rtol=0.0)`；根据条件结果选择对应处理分支。
    if not np.allclose(matrix, matrix.T, atol=1e-12, rtol=0.0):
        # 抛出 `AssertionError('日均相关矩阵不对称')`，阻止不满足数据契约的流程继续执行。
        raise AssertionError("日均相关矩阵不对称")
    # 检查 `not np.allclose(np.diag(matrix), 1.0, atol=1e-12, rtol=0.0)`；根据条件结果选择对应处理分支。
    if not np.allclose(np.diag(matrix), 1.0, atol=1e-12, rtol=0.0):
        # 抛出 `AssertionError('日均相关矩阵对角线不为 1')`，阻止不满足数据契约的流程继续执行。
        raise AssertionError("日均相关矩阵对角线不为 1")
    # 将 `summary` 作为当前函数结果返回给调用方。
    return summary


# 定义 `analyze_barra_exposures` 函数，封装这一阶段可独立调用的处理步骤。
def analyze_barra_exposures(
    source_run_name: str = DEFAULT_SOURCE_RUN_NAME,
    cne5s_path: str | Path = DEFAULT_CNE5S_PATH,
    vivit_data_root: str | Path = DEFAULT_VIVIT_DATA_ROOT,
    minimum_match_rate: float = MINIMUM_MATCH_RATE,
    eod_path: str | Path = DEFAULT_EOD_PATH,
    calendar_path: str | Path = DEFAULT_CALENDAR_PATH,
    source_factor_column: str = SOURCE_FACTOR_COLUMN,
) -> dict[str, Any]:
    """只读计算源因子的 Barra/反转覆盖、相关性与 11×11 矩阵。
    
    输入：
        source_run_name: str。
        cne5s_path: str | Path。
        vivit_data_root: str | Path。
        minimum_match_rate: float。
        eod_path: str | Path。
        calendar_path: str | Path。
        source_factor_column: str。
    输出：
        dict[str, Any]。
        """

    # 调用 `_validate_run_name` 校验输入、配置或产物是否满足当前阶段的数据契约，并将结果保存到 `source_run_name`。
    source_run_name = _validate_run_name(source_run_name)
    # 调用 `_validate_source_factor_column` 校验输入、配置或产物是否满足当前阶段的数据契约，并将结果保存到 `source_factor_column`。
    source_factor_column = _validate_source_factor_column(
        source_factor_column
    )
    # 检查 `not 0 <= minimum_match_rate <= 1`；根据条件结果选择对应处理分支。
    if not 0 <= minimum_match_rate <= 1:
        # 抛出 `ValueError('minimum_match_rate 必须位于 [0,1]')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("minimum_match_rate 必须位于 [0,1]")
    # 调用 `Path(vivit_data_root).resolve` 规范化文件路径，并将结果保存到 `vivit_root`。
    vivit_root = Path(vivit_data_root).resolve()
    # 调用 `Path(cne5s_path).resolve` 规范化文件路径，并将结果保存到 `cne5s`。
    cne5s = Path(cne5s_path).resolve()
    # 调用 `Path(eod_path).resolve` 规范化文件路径，并将结果保存到 `eod`。
    eod = Path(eod_path).resolve()
    # 调用 `Path(calendar_path).resolve` 规范化文件路径，并将结果保存到 `calendar_file`。
    calendar_file = Path(calendar_path).resolve()
    # 检查 `not vivit_root.is_dir()`；根据条件结果选择对应处理分支。
    if not vivit_root.is_dir():
        # 抛出 `FileNotFoundError(vivit_root)`，阻止不满足数据契约的流程继续执行。
        raise FileNotFoundError(vivit_root)
    # 调用 `_validate_cne5s_schema` 校验输入、配置或产物是否满足当前阶段的数据契约。
    _validate_cne5s_schema(cne5s)
    # 调用 `_validate_price_sources` 校验输入、配置或产物是否满足当前阶段的数据契约。
    _validate_price_sources(eod, calendar_file)
    # 调用 `_load_sse_calendar` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `calendar_dates`。
    calendar_dates = _load_sse_calendar(calendar_file)
    # 调用 `_training_manifest` 读取或生成记录数据来源与配置签名的 manifest，并将结果保存到 `(training, _)`。
    training, _ = _training_manifest(source_run_name, vivit_root)
    # 调用 `_factor_counts` 收集并汇总当前条件下满足要求的记录，并将结果保存到 `counts`。
    counts = _factor_counts(training)
    # 计算 `vivit_root / 'predictions' / 'runs' / source_run_name`，并将数值结果保存到 `prediction_root`。
    prediction_root = vivit_root / "predictions" / "runs" / source_run_name
    # 构造当前步骤所需的列表，并保存到 `match_records`。
    match_records: list[dict[str, Any]] = []
    # 构造当前步骤所需的列表，并保存到 `correlation_records`。
    correlation_records: list[dict[str, Any]] = []
    # 构造当前步骤所需的列表，并保存到 `matrix_correlation_records`。
    matrix_correlation_records: list[dict[str, Any]] = []

    # 遍历 `counts.items()`，并把当前元素绑定到 `(year, expected_rows)` 逐项处理。
    for year, expected_rows in counts.items():
        # 计算 `prediction_root / PREDICTION_PATTERN.format(year=year)`，并将数值结果保存到 `path`。
        path = prediction_root / PREDICTION_PATTERN.format(year=year)
        # 检查 `expected_rows == 0`；根据条件结果选择对应处理分支。
        if expected_rows == 0:
            # 检查 `path.exists()`；根据条件结果选择对应处理分支。
            if path.exists():
                # 抛出 `AssertionError(f'{year} factor_counts=0 但存在预测文件：{path}')`，阻止不满足数据契约的流程继续执行。
                raise AssertionError(
                    f"{year} factor_counts=0 但存在预测文件：{path}"
                )
            # 把本轮结果追加到 `match_records`，累积后续需要汇总的记录。
            match_records.append(
                {
                    "year": year,
                    "prediction_rows": 0,
                    "cne5s_rows": 0,
                    "matched_rows": 0,
                    "unmatched_rows": 0,
                    "match_rate": np.nan,
                    "reversal_valid_rows": 0,
                    "reversal_missing_rows": 0,
                    "reversal_coverage": np.nan,
                    "reversal_correlation_rows": 0,
                    "dates": 0,
                    "first_date": None,
                    "last_date": None,
                    "status": "empty_prediction_expected",
                }
            )
            # 跳过当前元素的剩余步骤，继续处理下一项。
            continue
        # 调用 `_load_source_prediction` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `prediction`。
        prediction = _load_source_prediction(
            path,
            year,
            expected_rows,
            full=True,
            source_factor_column=source_factor_column,
        )
        # 调用 `_attach_reversal_20d` 按股票和日期键拼接当前新增字段，并将结果保存到 `prediction`。
        prediction = _attach_reversal_20d(
            prediction, eod, calendar_dates, year
        )
        # 调用 `_load_cne5s_year` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `barra`。
        barra = _load_cne5s_year(cne5s, year)
        # 调用 `_join_prediction_and_cne5s` 执行模型推理并生成样本预测分数，并将结果保存到 `joined`。
        joined = _join_prediction_and_cne5s(prediction, barra, year)
        # 调用 `_match_record` 核对两侧记录并生成匹配质量结果，并将结果保存到 `record`。
        record = _match_record(joined, year, len(barra))
        # 调用 `_check_match_rate` 校验输入、配置或产物是否满足当前阶段的数据契约。
        _check_match_rate(record, minimum_match_rate)
        # 把本轮结果追加到 `match_records`，累积后续需要汇总的记录。
        match_records.append(record)
        # 把一组结果展开追加到 `correlation_records`，保持统一的一维记录序列。
        correlation_records.extend(_daily_correlations(joined, year))
        # 把一组结果展开追加到 `matrix_correlation_records`，保持统一的一维记录序列。
        matrix_correlation_records.extend(
            _daily_correlation_matrix(joined, year)
        )

    # 调用 `pd.DataFrame` 转换数据表示，并将标准化对象保存到 `input_audit`。
    input_audit = pd.DataFrame(match_records)
    # 调用 `input_audit.loc[:, ['year', 'prediction_rows', 'reversal_valid_…` 校验输入、配置或产物是否满足当前阶段的数据契约，并将结果保存到 `reversal_price_coverage`。
    reversal_price_coverage = input_audit.loc[:, [
        "year", "prediction_rows", "reversal_valid_rows",
        "reversal_missing_rows", "reversal_coverage",
        "reversal_correlation_rows", "dates", "first_date",
        "last_date", "status",
    ]].copy()
    # 调用 `pd.DataFrame` 转换数据表示，并将标准化对象保存到 `daily_correlations`。
    daily_correlations = pd.DataFrame(correlation_records)
    # 检查 `daily_correlations.empty`；根据条件结果选择对应处理分支。
    if daily_correlations.empty:
        # 抛出 `ValueError('源 run 没有可用于相关性分析的非空预测')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("源 run 没有可用于相关性分析的非空预测")
    # 检查 `set(daily_correlations['factor_name'].unique()) != set(CORRELATION_COLUMNS)`；根据条件结果选择对应处理分支。
    if set(daily_correlations["factor_name"].unique()) != set(
        CORRELATION_COLUMNS
    ):
        # 抛出 `AssertionError('日度相关性未恰好覆盖十个测试变量')`，阻止不满足数据契约的流程继续执行。
        raise AssertionError("日度相关性未恰好覆盖十个测试变量")
    # 调用 `_correlation_summaries` 计算预测与标签或风险暴露之间的相关性，并将结果保存到 `(annual_correlations, correlation_summary)`。
    annual_correlations, correlation_summary = _correlation_summaries(
        daily_correlations
    )
    # 调用 `pd.DataFrame` 转换数据表示，并将标准化对象保存到 `daily_correlation_matrix`。
    daily_correlation_matrix = pd.DataFrame(matrix_correlation_records)
    # 调用 `_correlation_matrix_summary` 按约定截面或分组口径汇总统计结果，并将结果保存到 `correlation_matrix_summary`。
    correlation_matrix_summary = _correlation_matrix_summary(
        daily_correlation_matrix
    )
    # 将 `{'source_run_name': source_run_name, 'source_factor_column': source_factor_column, 'input_match…` 作为当前函数结果返回给调用方。
    return {
        "source_run_name": source_run_name,
        "source_factor_column": source_factor_column,
        "input_match_audit": input_audit,
        "reversal_price_coverage": reversal_price_coverage,
        "daily_correlations": daily_correlations,
        "annual_correlations": annual_correlations,
        "correlation_summary": correlation_summary,
        "daily_correlation_matrix": daily_correlation_matrix,
        "correlation_matrix_summary": correlation_matrix_summary,
        "writes_performed": False,
    }


# 定义内部辅助函数 `_save_heatmap`，封装仅供本模块复用的计算步骤。
def _save_heatmap(summary: pd.DataFrame, path: Path) -> None:
    """生成并保存该评价环节对应的图表，缺失指标按既定口径处理。
    
    输入：
        summary: pd.DataFrame。
        path: Path。
    输出：
        None。
        """
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    import seaborn as sns

    # 计算 `Path.home() / '.local/share/fonts/NotoSansCJKsc-Regular.otf'`，并将数值结果保存到 `cjk_font_path`。
    cjk_font_path = Path.home() / ".local/share/fonts/NotoSansCJKsc-Regular.otf"
    # 检查 `cjk_font_path.exists()`；根据条件结果选择对应处理分支。
    if cjk_font_path.exists():
        # 调用 `font_manager.fontManager.addfont` 向 Matplotlib 字体管理器注册中文字体。
        font_manager.fontManager.addfont(str(cjk_font_path))

    # 读取 `summary.pivot(index='row_name', columns='column_name', values='mean_pearson').loc[list(MATRIX_C…` 的当前值，并赋给 `values`。
    values = summary.pivot(
        index="row_name", columns="column_name", values="mean_pearson"
    ).loc[list(MATRIX_COLUMNS), list(MATRIX_COLUMNS)]
    # 构造当前步骤所需的列表，并保存到 `labels`。
    labels = [MATRIX_LABELS[column] for column in MATRIX_COLUMNS]
    # 调用 `pd.DataFrame` 转换数据表示，并将标准化对象保存到 `matrix`。
    matrix = pd.DataFrame(values.to_numpy(np.float64), index=labels, columns=labels)
    # 构造当前步骤所需的列表，并保存到 `plt.rcParams['font.sans-serif']`。
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
    # 设置 `plt.rcParams['axes.unicode_minus']` 的固定配置值，供后续流程统一使用。
    plt.rcParams["axes.unicode_minus"] = False
    # 调用 `plt.subplots` 创建图形及其坐标轴对象，并将结果保存到 `(figure, axis)`。
    figure, axis = plt.subplots(figsize=(12.5, 10.5))
    # 调用 `sns.heatmap` 把因子相关矩阵绘制为带数值标注的热力图。
    sns.heatmap(
        matrix,
        ax=axis,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        linecolor="white",
        cbar=True,
        annot_kws={"fontsize": 8.5},
    )
    # 调用 `axis.set_title` 设置图形标题。
    axis.set_title(
        "ViViT 因子、9 个 CNE5S 风格与 20 日反转的日均横截面 Pearson",
        fontsize=13,
        pad=14,
    )
    # 调用 `axis.set_xlabel` 设置图形横轴含义。
    axis.set_xlabel("")
    # 调用 `axis.set_ylabel` 设置图形纵轴含义。
    axis.set_ylabel("")
    # 调用 `axis.tick_params` 调整坐标轴刻度标签样式。
    axis.tick_params(axis="x", rotation=45)
    # 调用 `axis.tick_params` 调整坐标轴刻度标签样式。
    axis.tick_params(axis="y", rotation=0)
    # 调用 `figure.tight_layout` 自动调整图形边距避免文字遮挡。
    figure.tight_layout()
    # 调用 `path.parent.mkdir` 创建输出目录，确保后续文件可以安全写入。
    path.parent.mkdir(parents=True, exist_ok=True)
    # 调用 `figure.savefig` 把当前图形按指定分辨率写入图片文件。
    figure.savefig(path, dpi=180, bbox_inches="tight")
    # 调用 `plt.close` 释放当前步骤占用的资源。
    plt.close(figure)

# 定义内部辅助函数 `_save_reversal_timeseries`，封装仅供本模块复用的计算步骤。
def _save_reversal_timeseries(
    daily_correlations: pd.DataFrame, path: Path
) -> None:
    """生成并保存该评价环节对应的图表，缺失指标按既定口径处理。
    
    输入：
        daily_correlations: pd.DataFrame。
        path: Path。
    输出：
        None。
        """
    from matplotlib import font_manager
    import matplotlib.pyplot as plt

    # 调用 `daily_correlations.loc[daily_correlations['factor_name'].eq(REV…` 计算预测与标签或风险暴露之间的相关性，并将结果保存到 `reversal`。
    reversal = daily_correlations.loc[
        daily_correlations["factor_name"].eq(REVERSAL_COLUMN)
    ].copy()
    # 检查 `reversal.empty`；根据条件结果选择对应处理分支。
    if reversal.empty:
        # 抛出 `ValueError('缺少 20 日反转日度相关性')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("缺少 20 日反转日度相关性")
    # 调用 `pd.to_datetime` 把日期字段转换为 Pandas 时间类型，并将结果保存到 `reversal['trade_date']`。
    reversal["trade_date"] = pd.to_datetime(
        reversal["trade_date"], format="%Y%m%d", errors="raise"
    )
    # 调用 `reversal.sort_values` 根据分数计算截面排序或排序索引。
    reversal.sort_values("trade_date", inplace=True)
    # 检查 `reversal['trade_date'].duplicated().any()`；根据条件结果选择对应处理分支。
    if reversal["trade_date"].duplicated().any():
        # 抛出 `ValueError('20 日反转日度相关性日期重复')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("20 日反转日度相关性日期重复")

    # 计算 `Path.home() / '.local/share/fonts/NotoSansCJKsc-Regular.otf'`，并将数值结果保存到 `cjk_font_path`。
    cjk_font_path = Path.home() / ".local/share/fonts/NotoSansCJKsc-Regular.otf"
    # 检查 `cjk_font_path.exists()`；根据条件结果选择对应处理分支。
    if cjk_font_path.exists():
        # 调用 `font_manager.fontManager.addfont` 向 Matplotlib 字体管理器注册中文字体。
        font_manager.fontManager.addfont(str(cjk_font_path))
    # 构造当前步骤所需的列表，并保存到 `plt.rcParams['font.sans-serif']`。
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
    # 设置 `plt.rcParams['axes.unicode_minus']` 的固定配置值，供后续流程统一使用。
    plt.rcParams["axes.unicode_minus"] = False
    # 调用 `plt.subplots` 创建图形及其坐标轴对象，并将结果保存到 `(figure, axis)`。
    figure, axis = plt.subplots(figsize=(14, 4.5))
    # 调用 `axis.plot` 绘制当前时间序列或截面统计图。
    axis.plot(
        reversal["trade_date"], reversal["pearson"],
        color="#7c3aed", linewidth=0.8, alpha=0.9,
        label="20日反转",
    )
    # 调用 `axis.axhline` 绘制零基准水平线。
    axis.axhline(0.0, color="#111827", linewidth=0.8, linestyle="--")
    # 调用 `axis.set_title` 设置图形标题。
    axis.set_title("ViViT 因子与 20 日反转的日度横截面 Pearson")
    # 调用 `axis.set_xlabel` 设置图形横轴含义。
    axis.set_xlabel("交易日期")
    # 调用 `axis.set_ylabel` 设置图形纵轴含义。
    axis.set_ylabel("横截面 Pearson")
    # 调用 `axis.grid` 为坐标轴添加辅助网格。
    axis.grid(alpha=0.2)
    # 调用 `axis.legend` 添加图例以区分不同收益或因子序列。
    axis.legend(frameon=False)
    # 调用 `figure.tight_layout` 自动调整图形边距避免文字遮挡。
    figure.tight_layout()
    # 调用 `path.parent.mkdir` 创建输出目录，确保后续文件可以安全写入。
    path.parent.mkdir(parents=True, exist_ok=True)
    # 调用 `figure.savefig` 把当前图形按指定分辨率写入图片文件。
    figure.savefig(path, dpi=180, bbox_inches="tight")
    # 调用 `plt.close` 释放当前步骤占用的资源。
    plt.close(figure)


# 定义内部辅助函数 `_derivation_payload`，封装仅供本模块复用的计算步骤。
def _derivation_payload(
    source_run_name: str,
    residual_run_name: str,
    training_path: Path,
    training: dict[str, Any],
    cne5s_fingerprint: dict[str, Any],
    eod_fingerprint: dict[str, Any],
    calendar_fingerprint: dict[str, Any],
    prediction_fingerprints: list[dict[str, Any]],
    minimum_match_rate: float,
    source_factor_column: str,
) -> dict[str, Any]:
    """构造参与稳定签名和复用判断的配置载荷。
    
    输入：
        source_run_name: str。
        residual_run_name: str。
        training_path: Path。
        training: dict[str, Any]。
        cne5s_fingerprint: dict[str, Any]。
        eod_fingerprint: dict[str, Any]。
        calendar_fingerprint: dict[str, Any]。
        prediction_fingerprints: list[dict[str, Any]]。
        minimum_match_rate: float。
        source_factor_column: str。
    输出：
        dict[str, Any]。
        """
    # 将 `{'schema_version': DERIVATION_SCHEMA_VERSION, 'source_run_name': source_run_name, 'residual_run…` 作为当前函数结果返回给调用方。
    return {
        "schema_version": DERIVATION_SCHEMA_VERSION,
        "source_run_name": source_run_name,
        "residual_run_name": residual_run_name,
        "source_training_manifest": {
            "path": str(training_path.resolve()),
            "size": int(training_path.stat().st_size),
            "mtime_ns": int(training_path.stat().st_mtime_ns),
            "source_dataset_signature": training.get(
                "source_dataset_signature"
            ),
            "source_rolling_signature": training.get(
                "source_rolling_signature"
            ),
            "sequence_signature": training.get("sequence_signature"),
        },
        "cne5s": cne5s_fingerprint,
        "wind_eod_prices": eod_fingerprint,
        "sse_calendar": calendar_fingerprint,
        "source_predictions": prediction_fingerprints,
        "source_factor_column": source_factor_column,
        "style_columns": list(STYLE_COLUMNS),
        "correlation_columns": list(CORRELATION_COLUMNS),
        "matrix_columns": list(MATRIX_COLUMNS),
        "regression_columns": list(REGRESSION_COLUMNS),
        "reversal": {
            "column": REVERSAL_COLUMN,
            "label": STYLE_LABELS[REVERSAL_COLUMN],
            "lookback_sse_trading_days": REVERSAL_LOOKBACK_DAYS,
            "price_column": "s_dq_adjclose",
            "formula": (
                "-(s_dq_adjclose(T)/s_dq_adjclose(T-20_SSE_days)-1)"
            ),
            "price_match": "exact_T_and_T_minus_20_no_fill",
            "winsorization": "none",
            "regression_scaling": "daily_zscore_ddof_0_valid_then_fill_zero",
        },
        "industry_column": INDUSTRY_COLUMN,
        "excluded_columns": [INDUSTRY_COLUMN, "cne5s_resvol"],
        "correlation_policy": (
            "daily_complete_case_barra_matched_and_reversal_valid_pearson_then_time_mean"
        ),
        "correlation_inference": {
            "naive_t_stat": "mean_over_iid_standard_error",
            "corrected_t_stat": "newey_west_hac_bartlett",
            "newey_west_max_lag": HAC_MAX_LAG,
        },
        "correlation_matrix_policy": (
            "daily_complete_case_11x11_pearson_then_cellwise_time_mean"
        ),
        "regression_policy": (
            "daily_equal_weight_ols_9_cne5s_styles_plus_reversal20_plus_industry_fixed_effects"
        ),
        "missing_barra_policy": (
            "style_zero_industry_unknown_remove_unknown_group_mean"
        ),
        "missing_reversal_policy": "daily_zscore_valid_then_fill_zero",
        "match_policy": "exact_trade_date_and_assetid_no_forward_fill",
        "residual_scaling": "raw_ols_residual_no_daily_zscore",
        "minimum_match_rate": minimum_match_rate,
        "orthogonality_tolerance": ORTHOGONALITY_TOLERANCE,
    }


# 定义内部辅助函数 `_existing_result`，封装仅供本模块复用的计算步骤。
def _existing_result(
    quality_root: Path,
    prediction_root: Path,
    expected_signature: str,
    counts: dict[int, int],
    *,
    overwrite_existing: bool = False,
) -> dict[str, Any] | None:
    """检查同签名派生 run 是否完整可复用，并返回其现有结果。
    
    输入：
        quality_root: Path。
        prediction_root: Path。
        expected_signature: str。
        counts: dict[int, int]。
        overwrite_existing: bool。
    输出：
        dict[str, Any] | None。
        """
    # 计算 `quality_root / 'barra_neutralization'`，并将数值结果保存到 `neutralization_root`。
    neutralization_root = quality_root / "barra_neutralization"
    # 计算 `neutralization_root / 'neutralization_manifest.json'`，并将数值结果保存到 `manifest_path`。
    manifest_path = neutralization_root / "neutralization_manifest.json"
    # 检查 `not quality_root.exists() and (not prediction_root.exists())`；根据条件结果选择对应处理分支。
    if not quality_root.exists() and not prediction_root.exists():
        # 将 `None` 作为当前函数结果返回给调用方。
        return None
    # 检查 `not manifest_path.exists()`；根据条件结果选择对应处理分支。
    if not manifest_path.exists():
        # 检查 `overwrite_existing`；根据条件结果选择对应处理分支。
        if overwrite_existing:
            # 将 `None` 作为当前函数结果返回给调用方。
            return None
        # 抛出 `RuntimeError('派生 run 已存在但缺少 neutralization manifest；如需替换必须显式设置 overwrite_existing=True')`，阻止不满足数据契约的流程继续执行。
        raise RuntimeError(
            "派生 run 已存在但缺少 neutralization manifest；"
            "如需替换必须显式设置 overwrite_existing=True"
        )
    # 调用 `_read_json` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `manifest`。
    manifest = _read_json(manifest_path)
    # 检查 `manifest.get('signature') != expected_signature`；根据条件结果选择对应处理分支。
    if manifest.get("signature") != expected_signature:
        # 检查 `overwrite_existing`；根据条件结果选择对应处理分支。
        if overwrite_existing:
            # 将 `None` 作为当前函数结果返回给调用方。
            return None
        # 抛出 `RuntimeError('已有派生 run 与当前源数据或配置签名不一致；如需替换必须显式设置 overwrite_existing=True')`，阻止不满足数据契约的流程继续执行。
        raise RuntimeError(
            "已有派生 run 与当前源数据或配置签名不一致；"
            "如需替换必须显式设置 overwrite_existing=True"
        )
    # 遍历 `counts.items()`，并把当前元素绑定到 `(year, count)` 逐项处理。
    for year, count in counts.items():
        # 计算 `prediction_root / PREDICTION_PATTERN.format(year=year)`，并将数值结果保存到 `path`。
        path = prediction_root / PREDICTION_PATTERN.format(year=year)
        # 检查 `count`；根据条件结果选择对应处理分支。
        if count:
            # 检查 `not path.exists()`；根据条件结果选择对应处理分支。
            if not path.exists():
                # 抛出 `FileNotFoundError(path)`，阻止不满足数据契约的流程继续执行。
                raise FileNotFoundError(path)
            # 检查 `pq.ParquetFile(path).metadata.num_rows != count`；根据条件结果选择对应处理分支。
            if pq.ParquetFile(path).metadata.num_rows != count:
                # 抛出 `AssertionError(f'{path.name} 行数与 manifest 不一致')`，阻止不满足数据契约的流程继续执行。
                raise AssertionError(f"{path.name} 行数与 manifest 不一致")
        # 检查 `path.exists()`；根据条件结果选择对应处理分支。
        elif path.exists():
            # 抛出 `AssertionError(f'{year} 应为空但存在派生预测文件')`，阻止不满足数据契约的流程继续执行。
            raise AssertionError(f"{year} 应为空但存在派生预测文件")
    # 构造键值映射，并保存到 `required`，供后续按键查询。
    required = {
        "input_match_audit": "input_match_audit.parquet",
        "reversal_price_coverage": "reversal_price_coverage.parquet",
        "daily_correlations": "daily_correlations.parquet",
        "annual_correlations": "annual_correlations.parquet",
        "correlation_summary": "correlation_summary.parquet",
        "daily_correlation_matrix": "daily_correlation_matrix.parquet",
        "correlation_matrix_summary": "correlation_matrix_summary.parquet",
        "regression_diagnostics": "daily_regression_diagnostics.parquet",
        "residual_correlations": "residual_correlation_audit.parquet",
    }
    # 构造键值映射，并保存到 `result`，供后续按键查询。
    result: dict[str, Any] = {
        name: pd.read_parquet(neutralization_root / file_name)
        for name, file_name in required.items()
    }
    # 计算 `neutralization_root / 'correlation_heatmap.png'`，并将数值结果保存到 `heatmap`。
    heatmap = neutralization_root / "correlation_heatmap.png"
    # 计算 `neutralization_root / 'reversal_20d_correlation_timeseries.png'`，并将数值结果保存到 `reversal_timeseries`。
    reversal_timeseries = (
        neutralization_root / "reversal_20d_correlation_timeseries.png"
    )
    # 遍历 `(heatmap, reversal_timeseries)`，并把当前元素绑定到 `path` 逐项处理。
    for path in (heatmap, reversal_timeseries):
        # 检查 `not path.exists()`；根据条件结果选择对应处理分支。
        if not path.exists():
            # 抛出 `FileNotFoundError(path)`，阻止不满足数据契约的流程继续执行。
            raise FileNotFoundError(path)
    # 用本轮键值结果更新 `result`，补齐后续查询所需字段。
    result.update(
        {
            "source_run_name": manifest["source_run_name"],
            "source_factor_column": manifest["source_factor_column"],
            "residual_run_name": manifest["residual_run_name"],
            "prediction_root": prediction_root,
            "quality_root": quality_root,
            "neutralization_root": neutralization_root,
            "heatmap_path": heatmap,
            "reversal_timeseries_path": reversal_timeseries,
            "manifest_path": manifest_path,
            "reused": True,
            "overwritten": False,
        }
    )
    # 将 `result` 作为当前函数结果返回给调用方。
    return result


# 定义内部辅助函数 `_replace_staged_roots`，封装仅供本模块复用的计算步骤。
def _replace_staged_roots(
    stage_predictions: Path,
    stage_quality: Path,
    prediction_root: Path,
    quality_root: Path,
) -> bool:
    """以临时 tombstone 支持失败回滚；成功后不保留旧 run。
    
    输入：
        stage_predictions: Path。
        stage_quality: Path。
        prediction_root: Path。
        quality_root: Path。
    输出：
        bool。
        """

    # 计算 `f'{os.getpid()}.{time.time_ns()}'`，并将结果写入 `token`。
    token = f"{os.getpid()}.{time.time_ns()}"
    # 调用 `prediction_root.with_name` 规范化文件路径，并将结果保存到 `old_prediction`。
    old_prediction = prediction_root.with_name(
        f".{prediction_root.name}.replacing.{token}"
    )
    # 调用 `quality_root.with_name` 规范化文件路径，并将结果保存到 `old_quality`。
    old_quality = quality_root.with_name(
        f".{quality_root.name}.replacing.{token}"
    )
    # 根据 `prediction_root.exists() or quality_root.exists()` 计算条件结果，并保存到 `had_existing`。
    had_existing = prediction_root.exists() or quality_root.exists()
    # 设置 `prediction_installed` 的固定配置值，供后续流程统一使用。
    prediction_installed = False
    # 设置 `quality_installed` 的固定配置值，供后续流程统一使用。
    quality_installed = False
    # 尝试执行本段主逻辑；若发生异常，则交由后续异常分支处理。
    try:
        # 检查 `prediction_root.exists()`；根据条件结果选择对应处理分支。
        if prediction_root.exists():
            # 调用 `prediction_root.replace` 执行模型推理并生成样本预测分数。
            prediction_root.replace(old_prediction)
        # 检查 `quality_root.exists()`；根据条件结果选择对应处理分支。
        if quality_root.exists():
            # 调用 `quality_root.replace` 把完整临时产物替换为正式输出。
            quality_root.replace(old_quality)
        # 调用 `stage_predictions.replace` 执行模型推理并生成样本预测分数。
        stage_predictions.replace(prediction_root)
        # 设置 `prediction_installed` 的固定配置值，供后续流程统一使用。
        prediction_installed = True
        # 调用 `stage_quality.replace` 把完整临时产物替换为正式输出。
        stage_quality.replace(quality_root)
        # 设置 `quality_installed` 的固定配置值，供后续流程统一使用。
        quality_installed = True
    # 捕获 `except Exception` 对应的异常，并执行明确的降级、记录或清理逻辑。
    except Exception:
        # 检查 `quality_installed and quality_root.exists()`；根据条件结果选择对应处理分支。
        if quality_installed and quality_root.exists():
            # 调用 `quality_root.replace` 把完整临时产物替换为正式输出。
            quality_root.replace(stage_quality)
        # 检查 `prediction_installed and prediction_root.exists()`；根据条件结果选择对应处理分支。
        if prediction_installed and prediction_root.exists():
            # 调用 `prediction_root.replace` 执行模型推理并生成样本预测分数。
            prediction_root.replace(stage_predictions)
        # 检查 `old_quality.exists()`；根据条件结果选择对应处理分支。
        if old_quality.exists():
            # 调用 `old_quality.replace` 把完整临时产物替换为正式输出。
            old_quality.replace(quality_root)
        # 检查 `old_prediction.exists()`；根据条件结果选择对应处理分支。
        if old_prediction.exists():
            # 调用 `old_prediction.replace` 执行模型推理并生成样本预测分数。
            old_prediction.replace(prediction_root)
        # 抛出 `空值`，阻止不满足数据契约的流程继续执行。
        raise
    # 当前置条件不成立时进入备用分支，保证其余合法情况仍被处理。
    else:
        # 检查 `old_quality.exists()`；根据条件结果选择对应处理分支。
        if old_quality.exists():
            # 调用 `shutil.rmtree` 递归删除明确指定的旧输出目录。
            shutil.rmtree(old_quality)
        # 检查 `old_prediction.exists()`；根据条件结果选择对应处理分支。
        if old_prediction.exists():
            # 调用 `shutil.rmtree` 递归删除明确指定的旧输出目录。
            shutil.rmtree(old_prediction)
    # 将 `had_existing` 作为当前函数结果返回给调用方。
    return had_existing


# 定义 `build_barra_residual_run` 函数，封装这一阶段可独立调用的处理步骤。
def build_barra_residual_run(
    source_run_name: str = DEFAULT_SOURCE_RUN_NAME,
    residual_run_name: str = DEFAULT_RESIDUAL_RUN_NAME,
    cne5s_path: str | Path = DEFAULT_CNE5S_PATH,
    vivit_data_root: str | Path = DEFAULT_VIVIT_DATA_ROOT,
    minimum_match_rate: float = MINIMUM_MATCH_RATE,
    eod_path: str | Path = DEFAULT_EOD_PATH,
    calendar_path: str | Path = DEFAULT_CALENDAR_PATH,
    overwrite_existing: bool = False,
    source_factor_column: str = SOURCE_FACTOR_COLUMN,
) -> dict[str, Any]:
    """生成相关性产物和可直接交给 Notebook 03 的 Barra 残差派生 run。
    
    输入：
        source_run_name: str。
        residual_run_name: str。
        cne5s_path: str | Path。
        vivit_data_root: str | Path。
        minimum_match_rate: float。
        eod_path: str | Path。
        calendar_path: str | Path。
        overwrite_existing: bool。
        source_factor_column: str。
    输出：
        dict[str, Any]。
        """

    # 调用 `_validate_run_name` 校验输入、配置或产物是否满足当前阶段的数据契约，并将结果保存到 `source_run_name`。
    source_run_name = _validate_run_name(source_run_name)
    # 调用 `_validate_run_name` 校验输入、配置或产物是否满足当前阶段的数据契约，并将结果保存到 `residual_run_name`。
    residual_run_name = _validate_run_name(residual_run_name)
    # 调用 `_validate_source_factor_column` 校验输入、配置或产物是否满足当前阶段的数据契约，并将结果保存到 `source_factor_column`。
    source_factor_column = _validate_source_factor_column(
        source_factor_column
    )
    # 检查 `residual_run_name == source_run_name`；根据条件结果选择对应处理分支。
    if residual_run_name == source_run_name:
        # 抛出 `ValueError('派生 run 不能与源 run 同名')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("派生 run 不能与源 run 同名")
    # 检查 `not 0 <= minimum_match_rate <= 1`；根据条件结果选择对应处理分支。
    if not 0 <= minimum_match_rate <= 1:
        # 抛出 `ValueError('minimum_match_rate 必须位于 [0,1]')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("minimum_match_rate 必须位于 [0,1]")
    # 检查 `not isinstance(overwrite_existing, bool)`；根据条件结果选择对应处理分支。
    if not isinstance(overwrite_existing, bool):
        # 抛出 `TypeError('overwrite_existing 必须是 bool')`，阻止不满足数据契约的流程继续执行。
        raise TypeError("overwrite_existing 必须是 bool")
    # 调用 `Path(vivit_data_root).resolve` 规范化文件路径，并将结果保存到 `vivit_root`。
    vivit_root = Path(vivit_data_root).resolve()
    # 调用 `Path(cne5s_path).resolve` 规范化文件路径，并将结果保存到 `cne5s`。
    cne5s = Path(cne5s_path).resolve()
    # 调用 `Path(eod_path).resolve` 规范化文件路径，并将结果保存到 `eod`。
    eod = Path(eod_path).resolve()
    # 调用 `Path(calendar_path).resolve` 规范化文件路径，并将结果保存到 `calendar_file`。
    calendar_file = Path(calendar_path).resolve()
    # 调用 `_validate_cne5s_schema` 校验输入、配置或产物是否满足当前阶段的数据契约，并将结果保存到 `cne5s_fingerprint`。
    cne5s_fingerprint = _validate_cne5s_schema(cne5s)
    # 调用 `_validate_price_sources` 校验输入、配置或产物是否满足当前阶段的数据契约，并将结果保存到 `(eod_fingerprint, calendar_fingerprint)`。
    eod_fingerprint, calendar_fingerprint = _validate_price_sources(
        eod, calendar_file
    )
    # 调用 `_load_sse_calendar` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `calendar_dates`。
    calendar_dates = _load_sse_calendar(calendar_file)
    # 调用 `_training_manifest` 读取或生成记录数据来源与配置签名的 manifest，并将结果保存到 `(training, training_path)`。
    training, training_path = _training_manifest(source_run_name, vivit_root)
    # 调用 `_factor_counts` 收集并汇总当前条件下满足要求的记录，并将结果保存到 `counts`。
    counts = _factor_counts(training)
    # 计算 `vivit_root / 'predictions' / 'runs' / source_run_name`，并将数值结果保存到 `source_prediction_root`。
    source_prediction_root = (
        vivit_root / "predictions" / "runs" / source_run_name
    )
    # 构造当前步骤所需的列表，并保存到 `prediction_fingerprints`。
    prediction_fingerprints = [
        _fingerprint_parquet(
            source_prediction_root / PREDICTION_PATTERN.format(year=year)
        )
        for year, count in counts.items()
        if count
    ]
    # 调用 `_derivation_payload` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `derivation`。
    derivation = _derivation_payload(
        source_run_name,
        residual_run_name,
        training_path,
        training,
        cne5s_fingerprint,
        eod_fingerprint,
        calendar_fingerprint,
        prediction_fingerprints,
        minimum_match_rate,
        source_factor_column,
    )
    # 调用 `_payload_signature` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `signature`。
    signature = _payload_signature(derivation)
    # 计算 `vivit_root / 'predictions' / 'runs' / residual_run_name`，并将数值结果保存到 `prediction_root`。
    prediction_root = (
        vivit_root / "predictions" / "runs" / residual_run_name
    )
    # 计算 `vivit_root / 'quality' / 'runs' / residual_run_name`，并将数值结果保存到 `quality_root`。
    quality_root = vivit_root / "quality" / "runs" / residual_run_name
    # 调用 `_existing_result` 检查是否已有签名一致且可直接复用的产物，并将结果保存到 `existing`。
    existing = _existing_result(
        quality_root, prediction_root, signature, counts,
        overwrite_existing=overwrite_existing,
    )
    # 检查 `existing is not None`；根据条件结果选择对应处理分支。
    if existing is not None:
        # 将 `existing` 作为当前函数结果返回给调用方。
        return existing

    # 调用 `Path` 规范化文件路径，并将结果保存到 `staging_parent`。
    staging_parent = Path(
        tempfile.mkdtemp(
            prefix=f".{residual_run_name}.",
            dir=vivit_root,
        )
    )
    # 计算 `staging_parent / 'predictions'`，并将数值结果保存到 `stage_predictions`。
    stage_predictions = staging_parent / "predictions"
    # 计算 `staging_parent / 'quality'`，并将数值结果保存到 `stage_quality`。
    stage_quality = staging_parent / "quality"
    # 计算 `stage_quality / 'barra_neutralization'`，并将数值结果保存到 `stage_neutralization`。
    stage_neutralization = stage_quality / "barra_neutralization"
    # 调用 `stage_predictions.mkdir` 创建输出目录，确保后续文件可以安全写入。
    stage_predictions.mkdir(parents=True)
    # 调用 `stage_neutralization.mkdir` 创建输出目录，确保后续文件可以安全写入。
    stage_neutralization.mkdir(parents=True)

    # 构造当前步骤所需的列表，并保存到 `match_records`。
    match_records: list[dict[str, Any]] = []
    # 构造当前步骤所需的列表，并保存到 `correlation_records`。
    correlation_records: list[dict[str, Any]] = []
    # 构造当前步骤所需的列表，并保存到 `matrix_correlation_records`。
    matrix_correlation_records: list[dict[str, Any]] = []
    # 构造当前步骤所需的列表，并保存到 `diagnostic_frames`。
    diagnostic_frames: list[pd.DataFrame] = []
    # 构造当前步骤所需的列表，并保存到 `residual_correlation_frames`。
    residual_correlation_frames: list[pd.DataFrame] = []
    # 构造键值映射，并保存到 `output_counts`，供后续按键查询。
    output_counts: dict[str, int] = {}
    # 尝试执行本段主逻辑；若发生异常，则交由后续异常分支处理。
    try:
        # 遍历 `counts.items()`，并把当前元素绑定到 `(year, expected_rows)` 逐项处理。
        for year, expected_rows in counts.items():
            # 检查 `expected_rows == 0`；根据条件结果选择对应处理分支。
            if expected_rows == 0:
                # 设置 `output_counts[str(year)]` 的固定配置值，供后续流程统一使用。
                output_counts[str(year)] = 0
                # 把本轮结果追加到 `match_records`，累积后续需要汇总的记录。
                match_records.append(
                    {
                        "year": year,
                        "prediction_rows": 0,
                        "cne5s_rows": 0,
                        "matched_rows": 0,
                        "unmatched_rows": 0,
                        "match_rate": np.nan,
                        "reversal_valid_rows": 0,
                        "reversal_missing_rows": 0,
                        "reversal_coverage": np.nan,
                        "reversal_correlation_rows": 0,
                        "dates": 0,
                        "first_date": None,
                        "last_date": None,
                        "status": "empty_prediction_expected",
                    }
                )
                # 跳过当前元素的剩余步骤，继续处理下一项。
                continue
            # 计算 `source_prediction_root / PREDICTION_PATTERN.format(year=year)`，并将数值结果保存到 `source_path`。
            source_path = (
                source_prediction_root
                / PREDICTION_PATTERN.format(year=year)
            )
            # 调用 `_load_source_prediction` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `prediction`。
            prediction = _load_source_prediction(
                source_path,
                year,
                expected_rows,
                full=True,
                source_factor_column=source_factor_column,
            )
            # 调用 `_attach_reversal_20d` 按股票和日期键拼接当前新增字段，并将结果保存到 `prediction`。
            prediction = _attach_reversal_20d(
                prediction, eod, calendar_dates, year
            )
            # 调用 `_load_cne5s_year` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `barra`。
            barra = _load_cne5s_year(cne5s, year)
            # 调用 `_join_prediction_and_cne5s` 执行模型推理并生成样本预测分数，并将结果保存到 `joined`。
            joined = _join_prediction_and_cne5s(
                prediction, barra, year
            )
            # 调用 `_match_record` 核对两侧记录并生成匹配质量结果，并将结果保存到 `match_record`。
            match_record = _match_record(joined, year, len(barra))
            # 调用 `_check_match_rate` 校验输入、配置或产物是否满足当前阶段的数据契约。
            _check_match_rate(match_record, minimum_match_rate)
            # 把本轮结果追加到 `match_records`，累积后续需要汇总的记录。
            match_records.append(match_record)
            # 把一组结果展开追加到 `correlation_records`，保持统一的一维记录序列。
            correlation_records.extend(
                _daily_correlations(joined, year)
            )
            # 把一组结果展开追加到 `matrix_correlation_records`，保持统一的一维记录序列。
            matrix_correlation_records.extend(
                _daily_correlation_matrix(joined, year)
            )
            # 调用 `_neutralize_year` 按 Barra 暴露执行截面投影或残差化处理，并将结果保存到 `(diagnostics, residual_correlations, residual)`。
            diagnostics, residual_correlations, residual = (
                _neutralize_year(joined, year)
            )
            # 把本轮结果追加到 `diagnostic_frames`，累积后续需要汇总的记录。
            diagnostic_frames.append(diagnostics)
            # 把本轮结果追加到 `residual_correlation_frames`，累积后续需要汇总的记录。
            residual_correlation_frames.append(residual_correlations)

            # 构造键值映射，并保存到 `output_columns`，供后续按键查询。
            output_columns: dict[str, Any] = {}
            # 检查 `'sample_index' in prediction.columns`；根据条件结果选择对应处理分支。
            if "sample_index" in prediction.columns:
                # 调用 `prediction['sample_index'].to_numpy` 执行模型推理并生成样本预测分数，并将结果保存到 `output_columns['sample_index']`。
                output_columns["sample_index"] = prediction[
                    "sample_index"
                ].to_numpy()
            # 用本轮键值结果更新 `output_columns`，补齐后续查询所需字段。
            output_columns.update(
                {
                    "trade_date": prediction["trade_date"].to_numpy(),
                    "instrument_id": prediction[
                        "instrument_id"
                    ].to_numpy(),
                    "source_prediction_ensemble": prediction[
                        SOURCE_FACTOR_COLUMN
                    ].to_numpy(),
                    "prediction_ensemble_raw": residual,
                    "prediction_ensemble": residual.copy(),
                    "barra_matched": joined[
                        "barra_matched"
                    ].to_numpy(bool),
                    REVERSAL_COLUMN: prediction[
                        REVERSAL_COLUMN
                    ].to_numpy(np.float64),
                    REVERSAL_VALID_COLUMN: prediction[
                        REVERSAL_VALID_COLUMN
                    ].to_numpy(bool),
                    "forecast_year": np.full(
                        len(prediction), year, dtype=np.int64
                    ),
                    "factor_role": np.full(
                        len(prediction),
                        "barra_reversal20_residual",
                        dtype=object,
                    ),
                }
            )
            # 调用 `pd.DataFrame` 转换数据表示，并将标准化对象保存到 `output`。
            output = pd.DataFrame(output_columns)
            # 检查 `len(output) != expected_rows`；根据条件结果选择对应处理分支。
            if len(output) != expected_rows:
                # 抛出 `AssertionError(f'{year} 派生预测行数改变')`，阻止不满足数据契约的流程继续执行。
                raise AssertionError(f"{year} 派生预测行数改变")
            # 检查 `not np.array_equal(output['trade_date'].to_numpy(), prediction['trade_date'].to_numpy()) or not…`；根据条件结果选择对应处理分支。
            if not np.array_equal(
                output["trade_date"].to_numpy(),
                prediction["trade_date"].to_numpy(),
            ) or not np.array_equal(
                output["instrument_id"].to_numpy(),
                prediction["instrument_id"].to_numpy(),
            ):
                # 抛出 `AssertionError(f'{year} 派生预测业务键顺序改变')`，阻止不满足数据契约的流程继续执行。
                raise AssertionError(f"{year} 派生预测业务键顺序改变")
            # 检查 `not np.array_equal(output['prediction_ensemble_raw'].to_numpy(), output['prediction_ensemble'].…`；根据条件结果选择对应处理分支。
            if not np.array_equal(
                output["prediction_ensemble_raw"].to_numpy(),
                output["prediction_ensemble"].to_numpy(),
            ):
                # 抛出 `AssertionError('原始残差不应被再次标准化')`，阻止不满足数据契约的流程继续执行。
                raise AssertionError("原始残差不应被再次标准化")
            # 调用 `_atomic_parquet` 先写临时文件再原子替换正式产物，避免留下半成品。
            _atomic_parquet(
                output,
                stage_predictions
                / PREDICTION_PATTERN.format(year=year),
            )
            # 调用 `len` 汇总当前数据，并将统计量保存到 `output_counts[str(year)]`。
            output_counts[str(year)] = len(output)

        # 调用 `pd.DataFrame` 转换数据表示，并将标准化对象保存到 `input_audit`。
        input_audit = pd.DataFrame(match_records)
        # 调用 `input_audit.loc[:, ['year', 'prediction_rows', 'reversal_valid_…` 校验输入、配置或产物是否满足当前阶段的数据契约，并将结果保存到 `reversal_price_coverage`。
        reversal_price_coverage = input_audit.loc[:, [
            "year", "prediction_rows", "reversal_valid_rows",
            "reversal_missing_rows", "reversal_coverage",
            "reversal_correlation_rows", "dates", "first_date",
            "last_date", "status",
        ]].copy()
        # 调用 `pd.DataFrame` 转换数据表示，并将标准化对象保存到 `daily_correlations`。
        daily_correlations = pd.DataFrame(correlation_records)
        # 检查 `set(daily_correlations['factor_name'].unique()) != set(CORRELATION_COLUMNS)`；根据条件结果选择对应处理分支。
        if set(daily_correlations["factor_name"].unique()) != set(
            CORRELATION_COLUMNS
        ):
            # 抛出 `AssertionError('日度相关性未恰好覆盖十个测试变量')`，阻止不满足数据契约的流程继续执行。
            raise AssertionError("日度相关性未恰好覆盖十个测试变量")
        # 调用 `_correlation_summaries` 计算预测与标签或风险暴露之间的相关性，并将结果保存到 `(annual_correlations, correlation_summary)`。
        annual_correlations, correlation_summary = (
            _correlation_summaries(daily_correlations)
        )
        # 调用 `pd.DataFrame` 转换数据表示，并将标准化对象保存到 `daily_correlation_matrix`。
        daily_correlation_matrix = pd.DataFrame(
            matrix_correlation_records
        )
        # 调用 `_correlation_matrix_summary` 按约定截面或分组口径汇总统计结果，并将结果保存到 `correlation_matrix_summary`。
        correlation_matrix_summary = _correlation_matrix_summary(
            daily_correlation_matrix
        )
        # 调用 `pd.concat` 拼接各批结果，并将合并值保存到 `regression_diagnostics`。
        regression_diagnostics = pd.concat(
            diagnostic_frames, ignore_index=True
        )
        # 调用 `pd.concat` 拼接各批结果，并将合并值保存到 `residual_correlations`。
        residual_correlations = pd.concat(
            residual_correlation_frames, ignore_index=True
        )
        # 构造键值映射，并保存到 `artifacts`，供后续按键查询。
        artifacts = {
            "input_match_audit.parquet": input_audit,
            "reversal_price_coverage.parquet": reversal_price_coverage,
            "daily_correlations.parquet": daily_correlations,
            "annual_correlations.parquet": annual_correlations,
            "correlation_summary.parquet": correlation_summary,
            "daily_correlation_matrix.parquet": daily_correlation_matrix,
            "correlation_matrix_summary.parquet": correlation_matrix_summary,
            "daily_regression_diagnostics.parquet": regression_diagnostics,
            "residual_correlation_audit.parquet": residual_correlations,
        }
        # 遍历 `artifacts.items()`，并把当前元素绑定到 `(file_name, frame)` 逐项处理。
        for file_name, frame in artifacts.items():
            # 调用 `_atomic_parquet` 先写临时文件再原子替换正式产物，避免留下半成品。
            _atomic_parquet(frame, stage_neutralization / file_name)
        # 计算 `stage_neutralization / 'correlation_heatmap.png'`，并将数值结果保存到 `heatmap_path`。
        heatmap_path = stage_neutralization / "correlation_heatmap.png"
        # 计算 `stage_neutralization / 'reversal_20d_correlation_timeseries.png'`，并将数值结果保存到 `reversal_timeseries_path`。
        reversal_timeseries_path = (
            stage_neutralization
            / "reversal_20d_correlation_timeseries.png"
        )
        # 调用 `_save_heatmap` 按约定格式写出当前计算产物。
        _save_heatmap(correlation_matrix_summary, heatmap_path)
        # 调用 `_save_reversal_timeseries` 按约定格式写出当前计算产物。
        _save_reversal_timeseries(
            daily_correlations, reversal_timeseries_path
        )

        # 调用 `dict` 整理数据结构，并将结果保存到 `derived_training`。
        derived_training = dict(training)
        # 用本轮键值结果更新 `derived_training`，补齐后续查询所需字段。
        derived_training.update(
            {
                "run_name": residual_run_name,
                "factor_counts": output_counts,
                "factor_role": "barra_reversal20_residual",
                "source_run_name": source_run_name,
                "source_factor_column": source_factor_column,
                "factor_derivation_signature": signature,
                "factor_derivation": derivation,
            }
        )
        # 调用 `_atomic_json` 先写临时文件再原子替换正式产物，避免留下半成品。
        _atomic_json(
            derived_training,
            stage_quality / "a2_training_manifest.json",
        )
        # 计算 `vivit_root / 'quality' / 'runs' / source_run_name / 'vivit_training_summary.parquet'`，并将数值结果保存到 `source_summary`。
        source_summary = (
            vivit_root
            / "quality"
            / "runs"
            / source_run_name
            / "vivit_training_summary.parquet"
        )
        # 检查 `source_summary.exists()`；根据条件结果选择对应处理分支。
        if source_summary.exists():
            # 调用 `shutil.copy2` 复制文件并尽量保留时间戳等元数据。
            shutil.copy2(
                source_summary,
                stage_quality / "vivit_training_summary.parquet",
            )
        # 构造键值映射，并保存到 `neutralization_manifest`，供后续按键查询。
        neutralization_manifest = {
            "signature": signature,
            **derivation,
            "factor_counts": output_counts,
            "match_summary": input_audit.to_dict("records"),
            "artifacts": sorted(
                [
                    *artifacts,
                    "correlation_heatmap.png",
                    "reversal_20d_correlation_timeseries.png",
                ]
            ),
        }
        # 调用 `_atomic_json` 先写临时文件再原子替换正式产物，避免留下半成品。
        _atomic_json(
            neutralization_manifest,
            stage_neutralization / "neutralization_manifest.json",
        )

        # 调用 `_existing_result` 检查是否已有签名一致且可直接复用的产物。
        _existing_result(
            stage_quality, stage_predictions, signature, counts
        )
        # 调用 `prediction_root.parent.mkdir` 创建输出目录，确保后续文件可以安全写入。
        prediction_root.parent.mkdir(parents=True, exist_ok=True)
        # 调用 `quality_root.parent.mkdir` 创建输出目录，确保后续文件可以安全写入。
        quality_root.parent.mkdir(parents=True, exist_ok=True)
        # 调用 `_replace_staged_roots` 解析本次运行对应的数据和输出目录，并将结果保存到 `overwritten`。
        overwritten = _replace_staged_roots(
            stage_predictions, stage_quality, prediction_root, quality_root
        )
    # 无论前序步骤是否抛出异常，都执行本分支完成必要的收尾。
    finally:
        # 调用 `shutil.rmtree` 递归删除明确指定的旧输出目录。
        shutil.rmtree(staging_parent, ignore_errors=True)

    # 将 `{'source_run_name': source_run_name, 'source_factor_column': source_factor_column, 'residual_ru…` 作为当前函数结果返回给调用方。
    return {
        "source_run_name": source_run_name,
        "source_factor_column": source_factor_column,
        "residual_run_name": residual_run_name,
        "input_match_audit": input_audit,
        "reversal_price_coverage": reversal_price_coverage,
        "daily_correlations": daily_correlations,
        "annual_correlations": annual_correlations,
        "correlation_summary": correlation_summary,
        "daily_correlation_matrix": daily_correlation_matrix,
        "correlation_matrix_summary": correlation_matrix_summary,
        "regression_diagnostics": regression_diagnostics,
        "residual_correlations": residual_correlations,
        "prediction_root": prediction_root,
        "quality_root": quality_root,
        "neutralization_root": (
            quality_root / "barra_neutralization"
        ),
        "heatmap_path": (
            quality_root
            / "barra_neutralization"
            / "correlation_heatmap.png"
        ),
        "reversal_timeseries_path": (
            quality_root
            / "barra_neutralization"
            / "reversal_20d_correlation_timeseries.png"
        ),
        "manifest_path": (
            quality_root
            / "barra_neutralization"
            / "neutralization_manifest.json"
        ),
        "reused": False,
        "overwritten": overwritten,
    }


# 构造当前步骤所需的列表，并保存到 `__all__`。
__all__ = [
    "STYLE_COLUMNS",
    "STYLE_LABELS",
    "CORRELATION_COLUMNS",
    "REGRESSION_COLUMNS",
    "MATRIX_COLUMNS",
    "MATRIX_LABELS",
    "REVERSAL_COLUMN",
    "REVERSAL_VALID_COLUMN",
    "analyze_barra_exposures",
    "audit_barra_inputs",
    "build_barra_residual_run",
    "instrument_ids_to_assetids",
]
# 定义 `audit_barra_inputs` 函数，封装这一阶段可独立调用的处理步骤。
def audit_barra_inputs(
    source_run_name: str = DEFAULT_SOURCE_RUN_NAME,
    cne5s_path: str | Path = DEFAULT_CNE5S_PATH,
    vivit_data_root: str | Path = DEFAULT_VIVIT_DATA_ROOT,
    minimum_match_rate: float = MINIMUM_MATCH_RATE,
    eod_path: str | Path = DEFAULT_EOD_PATH,
    calendar_path: str | Path = DEFAULT_CALENDAR_PATH,
    source_factor_column: str = SOURCE_FACTOR_COLUMN,
) -> pd.DataFrame:
    """只读审计源预测、CNE5S schema、精确同日业务键和年度覆盖率。
    
    输入：
        source_run_name: str。
        cne5s_path: str | Path。
        vivit_data_root: str | Path。
        minimum_match_rate: float。
        eod_path: str | Path。
        calendar_path: str | Path。
        source_factor_column: str。
    输出：
        pd.DataFrame。
        """

    # 调用 `_validate_run_name` 校验输入、配置或产物是否满足当前阶段的数据契约，并将结果保存到 `source_run_name`。
    source_run_name = _validate_run_name(source_run_name)
    # 调用 `_validate_source_factor_column` 校验输入、配置或产物是否满足当前阶段的数据契约，并将结果保存到 `source_factor_column`。
    source_factor_column = _validate_source_factor_column(
        source_factor_column
    )
    # 检查 `not 0 <= minimum_match_rate <= 1`；根据条件结果选择对应处理分支。
    if not 0 <= minimum_match_rate <= 1:
        # 抛出 `ValueError('minimum_match_rate 必须位于 [0,1]')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("minimum_match_rate 必须位于 [0,1]")
    # 调用 `Path(vivit_data_root).resolve` 规范化文件路径，并将结果保存到 `vivit_root`。
    vivit_root = Path(vivit_data_root).resolve()
    # 调用 `Path(cne5s_path).resolve` 规范化文件路径，并将结果保存到 `cne5s`。
    cne5s = Path(cne5s_path).resolve()
    # 调用 `Path(eod_path).resolve` 规范化文件路径，并将结果保存到 `eod`。
    eod = Path(eod_path).resolve()
    # 调用 `Path(calendar_path).resolve` 规范化文件路径，并将结果保存到 `calendar_file`。
    calendar_file = Path(calendar_path).resolve()
    # 检查 `not vivit_root.is_dir()`；根据条件结果选择对应处理分支。
    if not vivit_root.is_dir():
        # 抛出 `FileNotFoundError(vivit_root)`，阻止不满足数据契约的流程继续执行。
        raise FileNotFoundError(vivit_root)
    # 调用 `_validate_cne5s_schema` 校验输入、配置或产物是否满足当前阶段的数据契约，并将结果保存到 `cne5s_fingerprint`。
    cne5s_fingerprint = _validate_cne5s_schema(cne5s)
    # 调用 `_validate_price_sources` 校验输入、配置或产物是否满足当前阶段的数据契约，并将结果保存到 `(eod_fingerprint, calendar_fingerprint)`。
    eod_fingerprint, calendar_fingerprint = _validate_price_sources(
        eod, calendar_file
    )
    # 调用 `_load_sse_calendar` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `calendar_dates`。
    calendar_dates = _load_sse_calendar(calendar_file)
    # 调用 `_training_manifest` 读取或生成记录数据来源与配置签名的 manifest，并将结果保存到 `(training, training_path)`。
    training, training_path = _training_manifest(source_run_name, vivit_root)
    # 调用 `_factor_counts` 收集并汇总当前条件下满足要求的记录，并将结果保存到 `counts`。
    counts = _factor_counts(training)
    # 计算 `vivit_root / 'predictions' / 'runs' / source_run_name`，并将数值结果保存到 `prediction_root`。
    prediction_root = vivit_root / "predictions" / "runs" / source_run_name
    # 构造当前步骤所需的列表，并保存到 `records`。
    records: list[dict[str, Any]] = []
    # 构造当前步骤所需的列表，并保存到 `prediction_fingerprints`。
    prediction_fingerprints: list[dict[str, Any]] = []

    # 遍历 `counts.items()`，并把当前元素绑定到 `(year, expected_rows)` 逐项处理。
    for year, expected_rows in counts.items():
        # 计算 `prediction_root / PREDICTION_PATTERN.format(year=year)`，并将数值结果保存到 `path`。
        path = prediction_root / PREDICTION_PATTERN.format(year=year)
        # 检查 `expected_rows == 0`；根据条件结果选择对应处理分支。
        if expected_rows == 0:
            # 检查 `path.exists()`；根据条件结果选择对应处理分支。
            if path.exists():
                # 抛出 `AssertionError(f'{year} factor_counts=0 但存在预测文件：{path}')`，阻止不满足数据契约的流程继续执行。
                raise AssertionError(
                    f"{year} factor_counts=0 但存在预测文件：{path}"
                )
            # 把本轮结果追加到 `records`，累积后续需要汇总的记录。
            records.append(
                {
                    "year": year,
                    "prediction_rows": 0,
                    "cne5s_rows": 0,
                    "matched_rows": 0,
                    "unmatched_rows": 0,
                    "match_rate": np.nan,
                    "reversal_valid_rows": 0,
                    "reversal_missing_rows": 0,
                    "reversal_coverage": np.nan,
                    "reversal_correlation_rows": 0,
                    "dates": 0,
                    "first_date": None,
                    "last_date": None,
                    "status": "empty_prediction_expected",
                }
            )
            # 跳过当前元素的剩余步骤，继续处理下一项。
            continue
        # 把本轮结果追加到 `prediction_fingerprints`，累积后续需要汇总的记录。
        prediction_fingerprints.append(_fingerprint_parquet(path))
        # 调用 `_load_source_prediction` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `prediction`。
        prediction = _load_source_prediction(
            path,
            year,
            expected_rows,
            full=False,
            source_factor_column=source_factor_column,
        )
        # 调用 `_attach_reversal_20d` 按股票和日期键拼接当前新增字段，并将结果保存到 `prediction`。
        prediction = _attach_reversal_20d(
            prediction, eod, calendar_dates, year
        )
        # 调用 `_load_cne5s_year` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `barra`。
        barra = _load_cne5s_year(cne5s, year)
        # 调用 `_join_prediction_and_cne5s` 执行模型推理并生成样本预测分数，并将结果保存到 `joined`。
        joined = _join_prediction_and_cne5s(prediction, barra, year)
        # 调用 `_match_record` 核对两侧记录并生成匹配质量结果，并将结果保存到 `record`。
        record = _match_record(joined, year, len(barra))
        # 调用 `_check_match_rate` 校验输入、配置或产物是否满足当前阶段的数据契约。
        _check_match_rate(record, minimum_match_rate)
        # 把本轮结果追加到 `records`，累积后续需要汇总的记录。
        records.append(record)

    # 调用 `pd.DataFrame` 转换数据表示，并将标准化对象保存到 `audit`。
    audit = pd.DataFrame(records)
    # 用本轮键值结果更新 `audit.attrs`，补齐后续查询所需字段。
    audit.attrs.update(
        {
            "source_run_name": source_run_name,
            "source_factor_column": source_factor_column,
            "source_training_manifest": str(training_path),
            "cne5s_fingerprint": cne5s_fingerprint,
            "eod_fingerprint": eod_fingerprint,
            "calendar_fingerprint": calendar_fingerprint,
            "prediction_fingerprints": prediction_fingerprints,
            "minimum_match_rate": minimum_match_rate,
            "match_policy": "exact_trade_date_and_assetid_no_forward_fill",
            "reversal_formula": (
                "-(s_dq_adjclose(T)/s_dq_adjclose(T-20_SSE_days)-1)"
            ),
            "reversal_missing_policy": "daily_zscore_valid_then_fill_zero",
            "style_columns": list(STYLE_COLUMNS),
            "correlation_columns": list(CORRELATION_COLUMNS),
            "matrix_columns": list(MATRIX_COLUMNS),
            "regression_columns": list(REGRESSION_COLUMNS),
            "excluded_columns": [INDUSTRY_COLUMN, "cne5s_resvol"],
        }
    )
    # 将 `audit` 作为当前函数结果返回给调用方。
    return audit

