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

VIVIT_DIR = Path(__file__).resolve().parent  # 定位vivit目录。
PROJECT_ROOT = VIVIT_DIR.parent  # 定位项目根目录。
DEFAULT_VIVIT_DATA_ROOT = VIVIT_DIR / "data"  # 定位默认vivit数据根目录。
DEFAULT_CNE5S_PATH = PROJECT_ROOT / "CNE5S.parquet"  # 定位默认cne5s路径。
DEFAULT_EOD_PATH = PROJECT_ROOT / "Wind" / "ASHAREEODPRICES.parquet"  # 定位默认日行情路径。
DEFAULT_CALENDAR_PATH = PROJECT_ROOT / "Wind" / "ASHARECALENDAR.parquet"  # 定位默认交易日历路径。
DEFAULT_SOURCE_RUN_NAME = "sigmoid_daybatch_rankic"  # 设置默认来源执行名称。
DEFAULT_RESIDUAL_RUN_NAME = f"{DEFAULT_SOURCE_RUN_NAME}_barra_residual"  # 设置默认残差执行名称。
PREDICTION_PATTERN = "vivit_predictions.{year}.parquet"  # 声明逐年预测文件名模板。
SOURCE_FACTOR_COLUMN = "prediction_ensemble"  # 设置来源因子列。
SOURCE_FACTOR_COLUMNS = (  # 集中声明来源因子列。
    "prediction_ensemble",
    "prediction_ensemble_unprojected",
    "prediction_ensemble_projected",
    "prediction_ensemble_raw",
    "prediction_ensemble_alpha",
    "prediction_ensemble_style",
    "prediction_ensemble_total_return",
)
INDUSTRY_COLUMN = "cne5s_industry"  # 设置行业列。
UNKNOWN_INDUSTRY = "CNE5S_UNKNOWN"  # 设置未知行业行业。
MINIMUM_MATCH_RATE = 0.999  # 设置业务键连接允许的最低覆盖率。
DERIVATION_SCHEMA_VERSION = 5  # 固定产物 schema 版本，用于判断缓存与复用兼容性。
HAC_MAX_LAG = 20  # 设置hacmaxlag。
ORTHOGONALITY_TOLERANCE = 1e-8  # 设置 Barra 投影正交性的数值验收阈值。
REVERSAL_COLUMN = "reversal_20d"  # 设置reversal列。
REVERSAL_VALID_COLUMN = "reversal_20d_valid"  # 设置reversal有效列。
REVERSAL_LOOKBACK_DAYS = 20  # 设置reversallookbackdays。

STYLE_COLUMNS = (  # 集中声明风格列。
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
STYLE_LABELS = {  # 设置风格标签。
    "cne5s_beta": "贝塔",  # 把 cne5s_beta 映射为"贝塔"。
    "cne5s_btop": "账面市值比",  # 把 cne5s_btop 映射为"账面市值比"。
    "cne5s_earnyild": "盈利收益率",  # 把 cne5s_earnyild 映射为"盈利收益率"。
    "cne5s_growth": "成长",  # 把 cne5s_growth 映射为"成长"。
    "cne5s_leverage": "杠杆",  # 把 cne5s_leverage 映射为"杠杆"。
    "cne5s_liquidty": "流动性",  # 把 cne5s_liquidty 映射为"流动性"。
    "cne5s_momentum": "动量",  # 把 cne5s_momentum 映射为"动量"。
    "cne5s_size": "市值",  # 把 cne5s_size 映射为"市值"。
    "cne5s_sizenl": "非线性市值",  # 把 cne5s_sizenl 映射为"非线性市值"。
    REVERSAL_COLUMN: "20日反转",  # 为该字段配置报告显示名称。
}
CORRELATION_COLUMNS = (*STYLE_COLUMNS, REVERSAL_COLUMN)  # 集中声明相关系数列。
REGRESSION_COLUMNS = CORRELATION_COLUMNS  # 集中声明regression列。
VIVIT_MATRIX_COLUMN = "vivit_factor"  # 设置vivit矩阵列。
MATRIX_COLUMNS = (VIVIT_MATRIX_COLUMN, *CORRELATION_COLUMNS)  # 集中声明矩阵列。
MATRIX_LABELS = {  # 设置矩阵标签。
    VIVIT_MATRIX_COLUMN: "ViViT因子",  # 为该字段配置报告显示名称。
    **STYLE_LABELS,
}
REQUIRED_CNE5S_COLUMNS = (  # 集中声明必需cne5s列。
    "datadate",
    "assetid",
    INDUSTRY_COLUMN,  # 汇入行业列。
    *STYLE_COLUMNS,
)
INSTRUMENT_PATTERN = re.compile(r"^(SH|SZ|BJ)\.(\d{6})$")  # 编译股票pattern匹配规则。
CNE5S_ASSET_PATTERN = re.compile(r"^CN\d{6}$")  # 编译cne5sassetpattern匹配规则。


def _validate_run_name(value: str) -> str:  # 校验输入形状、业务键、有限性和签名约束。
    """校验输入形状、业务键、有限性和签名约束；异常时明确失败。"""
    run_name = str(value).strip()
    allowed = set(  # 初始化allowed容器。
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    )
    if (  # 开始组合多项有效性或配置条件。
        not run_name
        or run_name in {".", ".."}
        or any(character not in allowed for character in run_name)
    ):
        raise ValueError("运行名含非法字符")  # 输入数据不满足业务或数值约束时立即终止。
    return run_name  # 返回运行名称。


def _validate_source_factor_column(value: str) -> str:  # 校验输入形状、业务键、有限性和签名约束。
    """校验输入形状、业务键、有限性和签名约束；异常时明确失败。"""
    factor_column = str(value).strip()
    if factor_column not in SOURCE_FACTOR_COLUMNS:  # 检查当前值是否属于允许集合。
        raise ValueError(  # 输入数据不满足业务或数值约束时立即终止。
            f"未知源预测列：{value!r}；可选 {SOURCE_FACTOR_COLUMNS}"
        )
    return factor_column  # 返回因子列。


def _read_json(path: Path) -> dict[str, Any]:  # 读取对应产物，并按接口约定处理缺失文件。
    """读取对应产物，并按接口约定处理缺失文件。"""
    if not path.exists():  # 检查目标文件或目录是否满足存在性要求。
        raise FileNotFoundError(path)  # 目标文件不存在时立即终止。
    return json.loads(path.read_text(encoding="utf-8"))  # 解析并返回 JSON 文件内容。


def _atomic_json(payload: dict[str, Any], path: Path) -> None:  # 通过临时文件和原子替换写入产物，避免产生半成品。
    """通过临时文件和原子替换写入产物，避免产生半成品。"""
    path.parent.mkdir(parents=True, exist_ok=True)  # 确保目标目录存在。
    temporary = path.with_name(  # 构造临时文件。
        f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    temporary.write_text(  # 把文本结果写入临时或目标文件。
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",  # 使用 UTF-8 编解码文本。
    )
    temporary.replace(path)  # 用临时文件原子替换正式结果。


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:  # 通过临时文件和原子替换写入产物，避免产生半成品。
    """通过临时文件和原子替换写入产物，避免产生半成品。"""
    path.parent.mkdir(parents=True, exist_ok=True)  # 确保目标目录存在。
    temporary = path.with_name(  # 构造临时文件。
        f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    frame.to_parquet(temporary, index=False)  # 把表格按约定格式写入 Parquet。
    temporary.replace(path)  # 用临时文件原子替换正式结果。


def _payload_signature(payload: dict[str, Any]) -> str:  # 构造参与稳定签名和复用判断的配置载荷。
    """构造参与稳定签名和复用判断的配置载荷。"""
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()  # 返回稳定的 SHA-256 十六进制摘要。


def _training_manifest(  # 执行该模式下的训练步骤，并返回损失与诊断信息。
    source_run_name: str, vivit_root: Path
) -> tuple[dict[str, Any], Path]:  # 声明当前函数的返回类型。
    """执行该模式下的训练步骤，并返回损失与诊断信息。"""
    path = (
        vivit_root  # 汇入vivit根目录。
        / "quality"
        / "runs"
        / source_run_name
        / "a2_training_manifest.json"
    )
    manifest = _read_json(path)  # 读取清单。
    years = tuple(int(year) for year in manifest.get("years", ()))  # 初始化年份容器。
    counts = manifest.get("factor_counts")  # 计算数量。
    if not years or not isinstance(counts, dict):  # 检查对象类型是否符合接口约定。
        raise ValueError("源训练 manifest 缺少 years 或 factor_counts")  # 输入数据不满足业务或数值约束时立即终止。
    for year in years:  # 遍历年份，逐项处理年份。
        if str(year) not in counts and year not in counts:  # 检查当前值是否属于允许集合。
            raise ValueError(f"源训练 manifest 缺少 {year} factor_counts")  # 输入数据不满足业务或数值约束时立即终止。
    return manifest, path  # 返回上述处理结果。


def _factor_counts(manifest: dict[str, Any]) -> dict[int, int]:  # 读取或汇总因子评价所需的数据。
    """读取或汇总因子评价所需的数据。"""
    raw = manifest["factor_counts"]
    return {  # 返回读取并校验后的结果。
        int(year): int(raw.get(str(year), raw.get(year)))
        for year in manifest["years"]  # 遍历清单，逐项处理年份。
    }


def _load_source_prediction(  # 加载来源预测。
    path: Path,  # 接收路径。
    year: int,  # 接收年份。
    expected_rows: int,  # 接收预期行数。
    *,
    full: bool,  # 接收full。
    source_factor_column: str = SOURCE_FACTOR_COLUMN,  # 接收来源因子列。
) -> pd.DataFrame:  # 声明当前函数的返回类型。
    """加载该流程所需数据，并检查形状、业务键和来源一致性。"""
    source_factor_column = _validate_source_factor_column(
        source_factor_column  # 汇入来源因子列。
    )
    parquet = pq.ParquetFile(path)  # 读取 Parquet 元数据。
    columns = parquet.schema_arrow.names
  # 构造去重集合，并保存到 `required`。
    required = {"trade_date", "instrument_id", source_factor_column}  # 初始化必需容器。
    missing = required.difference(columns)  # 检查输入缺失字段。
    if missing:  # 根据当前数据状态选择处理分支。
        raise ValueError(f"{path.name} 缺少字段：{sorted(missing)}")  # 输入数据不满足业务或数值约束时立即终止。
    selected = list(columns) if full else list(required)  # 构造筛选结果，限定后续参与计算的样本。
    prediction = pd.read_parquet(path, columns=selected)  # 读取预测。
    if len(prediction) != expected_rows:  # 检查当前样本数量是否满足计算要求。
        raise AssertionError(  # 关键数据契约被破坏时立即终止。
            f"{year} 预测行数 {len(prediction):,} != {expected_rows:,}"
        )
    prediction["trade_date"] = prediction["trade_date"].astype(str)  # 转换预测的数据类型或存储位置。
    prediction["instrument_id"] = prediction["instrument_id"].astype(str)  # 转换预测的数据类型或存储位置。
    if prediction.duplicated(["trade_date", "instrument_id"]).any():  # 检查当前掩码或数据是否满足有效性要求。
        raise ValueError(f"{year} 源预测业务键重复")  # 输入数据不满足业务或数值约束时立即终止。
    if not prediction["trade_date"].str.startswith(str(year)).all():  # 检查当前掩码或数据是否满足有效性要求。
        raise ValueError(f"{year} 源预测包含跨年度日期")  # 输入数据不满足业务或数值约束时立即终止。
    factor = prediction[source_factor_column].to_numpy(np.float64)  # 转换并统一因子的数据表示。
    if not np.isfinite(factor).all():  # 检查参与计算的数值是否全部有限。
        raise ValueError(f"{year} 源因子含 NaN 或 Inf")  # 输入数据不满足业务或数值约束时立即终止。
  # 下游统一读取内部标准列；这里只修改内存副本，不写回源预测。
    prediction[SOURCE_FACTOR_COLUMN] = factor
    prediction["assetid"] = instrument_ids_to_assetids(  # 转换为 CNE5S assetid。
        prediction["instrument_id"]  # 汇入预测。
    )
    return prediction  # 返回预测。


def _pearson(x: np.ndarray, y: np.ndarray) -> float:  # 计算有效样本上的 Pearson 相关系数。
    """计算有效样本上的 Pearson 相关系数。"""
    x = np.asarray(x, dtype=np.float64)  # 转换并统一x的数据表示。
    y = np.asarray(y, dtype=np.float64)  # 转换并统一y的数据表示。
    if len(x) < 3 or len(x) != len(y):  # 检查当前样本数量是否满足计算要求。
        raise ValueError("Pearson 至少需要 3 个等长观测")  # 输入数据不满足业务或数值约束时立即终止。
    x_centered = x - x.mean()  # 计算x中心化。
    y_centered = y - y.mean()  # 计算y中心化。
    denominator = float(
        np.sqrt(np.dot(x_centered, x_centered) * np.dot(y_centered, y_centered))
    )
    if not np.isfinite(denominator) or denominator <= np.finfo(np.float64).eps:  # 检查参与计算的数值是否全部有限。
        raise ValueError("Pearson 输入为常数或非有限")  # 输入数据不满足业务或数值约束时立即终止。
    return float(np.dot(x_centered, y_centered) / denominator)  # 返回上述计算结果。


def _complete_correlation_sample(joined: pd.DataFrame) -> pd.DataFrame:  # 返回 11×11 相关矩阵统一使用的完整同日样本。
    """返回 11×11 相关矩阵统一使用的完整同日样本。"""

    return joined.loc[  # 返回 11×11 相关矩阵统一使用的完整同日样本。
        joined["barra_matched"] & joined[REVERSAL_VALID_COLUMN]
    ]


def _daily_correlations(  # 汇总截面相关性及其时间序列统计。
    joined: pd.DataFrame, year: int
) -> list[dict[str, Any]]:  # 声明当前函数的返回类型。
    """汇总截面相关性及其时间序列统计。"""
    records: list[dict[str, Any]] = []  # 初始化records容器。
    complete = _complete_correlation_sample(joined)  # 计算当前截面相关性。
    for trade_date, daily in complete.groupby("trade_date", sort=True):  # 遍历completegroupbytrade交易日sorttrue，逐项处理trade交易日daily。
        y = daily[SOURCE_FACTOR_COLUMN].to_numpy(np.float64)  # 转换并统一y的数据表示。
        for column in CORRELATION_COLUMNS:  # 遍历相关系数列，逐项处理列。
            records.append(  # 把当前结果追加到汇总列表。
                {
                    "year": year,  # 记录年份。
                    "trade_date": str(trade_date),  # 记录trade交易日。
                    "factor_name": column,  # 记录因子名称。
                    "factor_label": STYLE_LABELS[column],  # 记录因子标签。
                    "pearson": _pearson(  # 记录Pearson。
                        y, daily[column].to_numpy(np.float64)
                    ),
                    "observations": len(daily),  # 记录observations。
                }
            )
    return records  # 返回记录。


def _daily_correlation_matrix(  # 计算 ViViT、九个 CNE5S 风格和反转的日度 11×11 Pearson。
    joined: pd.DataFrame, year: int
) -> list[dict[str, Any]]:  # 声明当前函数的返回类型。
    """计算 ViViT、九个 CNE5S 风格和反转的日度 11×11 Pearson。"""

    records: list[dict[str, Any]] = []  # 初始化records容器。
    complete = _complete_correlation_sample(joined)  # 计算当前截面相关性。
    for trade_date, daily in complete.groupby("trade_date", sort=True):  # 遍历completegroupbytrade交易日sorttrue，逐项处理trade交易日daily。
        values = np.column_stack(  # 组成 Barra 设计矩阵。
            [
                daily[SOURCE_FACTOR_COLUMN].to_numpy(np.float64),
                *(
                    daily[column].to_numpy(np.float64)
                    for column in CORRELATION_COLUMNS  # 遍历相关系数列，逐项处理列。
                ),
            ]
        )
        if len(daily) < 3 or not np.isfinite(values).all():  # 检查参与计算的数值是否全部有限。
            raise ValueError(f"{trade_date} 相关矩阵样本不足或含非有限值")  # 输入数据不满足业务或数值约束时立即终止。
        matrix = np.corrcoef(values, rowvar=False)  # 计算因子间 Pearson 相关性。
        expected_shape = (len(MATRIX_COLUMNS), len(MATRIX_COLUMNS))
        if matrix.shape != expected_shape or not np.isfinite(matrix).all():  # 检查参与计算的数值是否全部有限。
            raise ValueError(f"{trade_date} 相关矩阵不可计算")  # 输入数据不满足业务或数值约束时立即终止。
        if not np.allclose(matrix, matrix.T, atol=1e-12, rtol=0.0):  # 检查必需数据、配置或校验结果是否缺失。
            raise AssertionError(f"{trade_date} 相关矩阵不对称")  # 关键数据契约被破坏时立即终止。
        if not np.allclose(  # 检查必需数据、配置或校验结果是否缺失。
            np.diag(matrix), 1.0, atol=1e-12, rtol=0.0
        ):
            raise AssertionError(f"{trade_date} 相关矩阵对角线不为 1")  # 关键数据契约被破坏时立即终止。
        for row_index, row_name in enumerate(MATRIX_COLUMNS):  # 遍历enumerate矩阵列，逐项处理行索引行名称。
            for column_index, column_name in enumerate(MATRIX_COLUMNS):  # 遍历enumerate矩阵列，逐项处理列索引列名称。
                records.append(  # 把当前结果追加到汇总列表。
                    {
                        "year": year,  # 记录年份。
                        "trade_date": str(trade_date),  # 记录trade交易日。
                        "row_name": row_name,  # 记录行名称。
                        "row_label": MATRIX_LABELS[row_name],  # 记录行标签。
                        "column_name": column_name,  # 记录字段名称。
                        "column_label": MATRIX_LABELS[column_name],  # 记录字段标签。
                        "pearson": float(matrix[row_index, column_index]),  # 记录Pearson。
                        "observations": len(daily),  # 记录observations。
                    }
                )
    return records  # 返回记录。

def _neutralize_cross_section(  # 以行业固定效应、9 个 CNE5S 风格和 20 日反转残差化。
    daily: pd.DataFrame,  # 接收daily。
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:  # 声明当前函数的返回类型。
    """以行业固定效应、9 个 CNE5S 风格和 20 日反转残差化。"""

    y = daily[SOURCE_FACTOR_COLUMN].to_numpy(np.float64)  # 转换并统一y的数据表示。
    if not np.isfinite(y).all():  # 检查参与计算的数值是否全部有限。
        raise ValueError("单日源因子含 NaN 或 Inf")  # 输入数据不满足业务或数值约束时立即终止。
    matched = daily["barra_matched"].to_numpy(bool)  # 转换并统一匹配数量的数据表示。
    x = np.zeros((len(daily), len(REGRESSION_COLUMNS)), dtype=np.float64)  # 初始化x。
    style_values = daily.loc[:, STYLE_COLUMNS].to_numpy(  # 转换并统一风格值的数据表示。
        np.float64, copy=True
    )
    style_values[~matched, :] = 0.0
    x[:, : len(STYLE_COLUMNS)] = style_values
    x[:, -1] = _standardized_reversal(daily)  # 仅使用当日截面统计量。

    industries = daily[INDUSTRY_COLUMN].astype("object").copy()  # 转换行业的数据类型或存储位置。
    industries.loc[~daily["barra_matched"]] = UNKNOWN_INDUSTRY
    if industries.isna().any() or not np.isfinite(x).all():  # 检查参与计算的数值是否全部有限。
        raise ValueError("中性化输入含未处理的缺失值")  # 输入数据不满足业务或数值约束时立即终止。

    codes, labels = pd.factorize(industries.astype(str), sort=True)  # 将行业映射为连续代码。
    counts = np.bincount(codes).astype(np.float64)  # 计算数量。
    y_means = np.bincount(codes, weights=y) / counts  # 计算ymeans。
    x_sums = np.zeros(  # 初始化xsums。
        (len(labels), len(REGRESSION_COLUMNS)), dtype=np.float64
    )
    np.add.at(x_sums, codes, x)  # 按行业汇总截面统计量。
    x_means = x_sums / counts[:, None]  # 计算xmeans。
    y_within = y - y_means[codes]
    x_within = x - x_means[codes]

    coefficients, _, rank, singular_values = np.linalg.lstsq(  # 求解当日截面 OLS。
        x_within, y_within, rcond=None  # 向最小二乘求解器传入矩阵与目标，并使用默认奇异值截断阈值。
    )
    expected_rank = len(REGRESSION_COLUMNS)  # 记录设计矩阵应达到的满列秩。
    if int(rank) != expected_rank:  # 确认回归设计矩阵达到满列秩。
        raise np.linalg.LinAlgError(  # 回归设计矩阵无法稳定求解时立即终止。
            f"风格暴露矩阵秩为 {rank}，预期 {expected_rank}"
        )
    if (  # 开始组合多项有效性或配置条件。
        len(singular_values) != expected_rank
        or singular_values[-1] <= np.finfo(np.float64).eps
    ):
        raise np.linalg.LinAlgError("风格暴露矩阵条件数不可计算")  # 回归设计矩阵无法稳定求解时立即终止。
    residual = y_within - x_within @ coefficients
    if not np.isfinite(residual).all():  # 检查参与计算的数值是否全部有限。
        raise ValueError("回归残差含 NaN 或 Inf")  # 输入数据不满足业务或数值约束时立即终止。
    source_variance = float(np.dot(y - y.mean(), y - y.mean()))  # 计算来源方差。
    if source_variance <= np.finfo(np.float64).eps:  # 检查当前数值是否越过业务或数值阈值。
        raise ValueError("单日源因子为常数")  # 输入数据不满足业务或数值约束时立即终止。
    residual_variance = float(np.dot(residual, residual))  # 计算残差方差。
    residual_std = float(residual.std(ddof=0))  # 计算残差标准差。
    if residual_std <= np.finfo(np.float64).eps:  # 检查当前数值是否越过业务或数值阈值。
        raise ValueError("单日回归残差为常数")  # 输入数据不满足业务或数值约束时立即终止。

    residual_correlations: list[dict[str, Any]] = []  # 计算残差correlations。
    for column_index, column in enumerate(REGRESSION_COLUMNS):  # 遍历enumerateregression列，逐项处理列索引列。
        residual_correlations.append(  # 把当前结果追加到汇总列表。
            {
                "factor_name": column,  # 记录因子名称。
                "factor_label": STYLE_LABELS[column],  # 记录因子标签。
                "pearson": _pearson(x[:, column_index], residual),  # 记录Pearson。
                "observations": len(daily),  # 记录observations。
                "valid_observations": (  # 记录有效observations。
                    int(daily[REVERSAL_VALID_COLUMN].sum())
                    if column == REVERSAL_COLUMN  # 按当前模式、标识或状态选择处理分支。
                    else int(matched.sum())
                ),
            }
        )
    group_residual_means = np.bincount(  # 计算group残差means。
        codes, weights=residual
    ) / counts
    max_style_correlation = max(  # 计算max风格相关系数。
        abs(record["pearson"]) for record in residual_correlations
    )
    max_industry_mean = float(np.max(np.abs(group_residual_means)))  # 计算max行业均值。
    if max_style_correlation > ORTHOGONALITY_TOLERANCE:  # 检查当前数值是否越过业务或数值阈值。
        raise AssertionError(  # 关键数据契约被破坏时立即终止。
            f"残差风格相关性 {max_style_correlation:.3e} 超过容差"
        )
    if max_industry_mean > ORTHOGONALITY_TOLERANCE:  # 检查当前数值是否越过业务或数值阈值。
        raise AssertionError(  # 关键数据契约被破坏时立即终止。
            f"行业残差均值 {max_industry_mean:.3e} 超过容差"
        )

    reversal_valid_rows = int(daily[REVERSAL_VALID_COLUMN].sum())  # 计算reversal有效行数。
    diagnostics = {  # 初始化诊断容器。
        "rows": len(daily),  # 记录行数。
        "matched_rows": int(matched.sum()),  # 记录匹配行数。
        "unmatched_rows": int((~matched).sum()),  # 记录未匹配行数。
        "match_rate": float(matched.mean()),  # 记录match比例。
        "reversal_valid_rows": reversal_valid_rows,  # 记录反转有效行数。
        "reversal_missing_rows": len(daily) - reversal_valid_rows,  # 记录反转缺失值行数。
        "reversal_coverage": float(reversal_valid_rows / len(daily)),  # 记录反转覆盖率。
        "industries": int(len(labels)),  # 记录行业。
        "matrix_rank": int(rank),  # 记录回归设计矩阵的实际秩。
        "condition_number": float(  # 记录conditionnumber。
            singular_values[0] / singular_values[-1]
        ),
        "r_squared": float(1.0 - residual_variance / source_variance),  # 记录rsquared。
        "source_factor_std": float(y.std(ddof=0)),  # 记录来源因子标准差。
        "residual_factor_std": residual_std,  # 记录残差因子标准差。
        "residual_factor_mean": float(residual.mean()),  # 记录残差因子均值。
        "max_abs_style_residual_corr": max_style_correlation,  # 记录maxabs风格残差corr。
        "max_abs_industry_residual_mean": max_industry_mean,  # 记录maxabs行业残差均值。
    }
    return residual, diagnostics, residual_correlations  # 返回以行业固定效应、9 个 CNE5S 风格和 20 日反转残差化。


def _neutralize_year(  # 逐日对年度原始分数回归十变量和行业固定效应，生成统一残差。
    joined: pd.DataFrame, year: int
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:  # 声明当前函数的返回类型。
    """逐日对年度原始分数回归十变量和行业固定效应，生成统一残差。"""
    residual = np.empty(len(joined), dtype=np.float64)  # 初始化残差。
    diagnostic_records: list[dict[str, Any]] = []  # 初始化diagnosticrecords容器。
    correlation_records: list[dict[str, Any]] = []  # 计算相关系数records。
    for trade_date, positions in joined.groupby(  # 遍历joinedgroupby，逐项处理trade交易日位置。
        "trade_date", sort=True
    ).groups.items():
        index = np.asarray(positions, dtype=np.int64)  # 转换并统一索引的数据表示。
        values, diagnostics, correlations = _neutralize_cross_section(  # 按日投影 Barra 暴露。
            joined.iloc[index]  # 汇入joinediloc。
        )
        residual[index] = values
        diagnostic_records.append(  # 把当前结果追加到汇总列表。
            {
                "year": year,  # 记录年份。
                "trade_date": str(trade_date),  # 记录trade交易日。
                **diagnostics,
            }
        )
        correlation_records.extend(  # 把一组结果并入当前列表。
            {
                "year": year,  # 记录年份。
                "trade_date": str(trade_date),  # 记录trade交易日。
                **record,
            }
            for record in correlations  # 遍历correlations，逐项处理record。
        )
    return (  # 返回逐日计算结果与诊断。
        pd.DataFrame(diagnostic_records),
        pd.DataFrame(correlation_records),
        residual,  # 汇入残差。
    )


def _newey_west_mean_t_stat(  # 以 Bartlett 核估计均值的 Newey-West HAC t 值。
    values: pd.Series, max_lag: int = HAC_MAX_LAG
) -> float:  # 声明当前函数的返回类型。
    """以 Bartlett 核估计均值的 Newey-West HAC t 值。"""

    sample = np.asarray(values, dtype=np.float64)  # 转换并统一样本的数据表示。
    if len(sample) < 2 or not np.isfinite(sample).all():  # 检查参与计算的数值是否全部有限。
        return np.nan  # 返回以 Bartlett 核估计均值的 Newey-West HAC t 值。
    centered = sample - sample.mean()  # 计算中心化。
    lag = min(max_lag, len(sample) - 1)
    long_run_variance = float(np.dot(centered, centered) / len(sample))  # 计算long执行方差。
    for offset in range(1, lag + 1):  # 按给定步长遍历样本区间，更新offset。
        weight = 1.0 - offset / (lag + 1.0)
        autocovariance = float(  # 计算autocovariance。
            np.dot(centered[offset:], centered[:-offset]) / len(sample)
        )
        long_run_variance += 2.0 * weight * autocovariance
    if (  # 开始组合多项有效性或配置条件。
        not np.isfinite(long_run_variance)
        or long_run_variance <= np.finfo(np.float64).eps
    ):
        return np.nan  # 返回以 Bartlett 核估计均值的 Newey-West HAC t 值。
    return float(sample.mean() / np.sqrt(long_run_variance / len(sample)))  # 返回以 Bartlett 核估计均值的 Newey-West HAC t 值。


def _correlation_summaries(  # 汇总截面相关性及其时间序列统计。
    daily: pd.DataFrame,  # 接收daily。
) -> tuple[pd.DataFrame, pd.DataFrame]:  # 声明当前函数的返回类型。
    """汇总截面相关性及其时间序列统计。"""
    def summarize(group_columns: list[str]) -> pd.DataFrame:  # 汇总截面相关性及其时间序列统计。
        """汇总截面相关性及其时间序列统计。"""
        frame = daily.sort_values("trade_date").groupby(  # 计算帧以固定样本顺序。
            group_columns, sort=True, observed=True
        )
        result = frame.agg(  # 计算当前截面相关性。
            mean_pearson=("pearson", "mean"),  # 配置均值Pearson。
            mean_abs_pearson=("pearson", lambda values: values.abs().mean()),  # 配置均值absPearson。
            std_pearson=("pearson", "std"),  # 配置标准差Pearson。
            dates=("trade_date", "nunique"),  # 配置交易日。
            mean_observations=("observations", "mean"),  # 配置均值observations。
        ).reset_index()
        result["t_stat"] = (
            result["mean_pearson"]  # 汇入结果。
            / result["std_pearson"]
            * np.sqrt(result["dates"])
        )
        corrected = (  # 计算当前截面相关性。
            frame["pearson"]  # 汇入帧。
            .agg(_newey_west_mean_t_stat)
            .rename("t_stat_newey_west_lag20")
            .reset_index()
        )
        return result.merge(  # 返回汇总截面相关性及其时间序列统计。
            corrected,  # 汇入corrected。
            on=group_columns,  # 指定数据连接使用的业务键。
            how="left",  # 指定数据连接方式。
            validate="one_to_one",  # 校验数据连接的基数关系。
        )

    annual = summarize(  # 按约定截面口径汇总。
        ["year", "factor_name", "factor_label"]
    )
    overall = summarize(  # 按约定截面口径汇总。
        ["factor_name", "factor_label"]
    )
    order = {  # 初始化排序索引容器。
        column: index for index, column in enumerate(CORRELATION_COLUMNS)
    }
    annual["_order"] = annual["factor_name"].map(order)
    overall["_order"] = overall["factor_name"].map(order)
    annual.sort_values(["year", "_order"], inplace=True)
    overall.sort_values("_order", inplace=True)
    return (  # 返回汇总截面相关性及其时间序列统计。
        annual.drop(columns="_order").reset_index(drop=True),
        overall.drop(columns="_order").reset_index(drop=True),
    )


def _correlation_matrix_summary(daily: pd.DataFrame) -> pd.DataFrame:  # 汇总截面相关性及其时间序列统计。
    """汇总截面相关性及其时间序列统计。"""
    expected_cells = len(MATRIX_COLUMNS) ** 2
    cells_per_date = daily.groupby("trade_date", sort=False).size()  # 统计每日截面样本数。
    if daily.empty or not cells_per_date.eq(expected_cells).all():  # 检查当前掩码或数据是否满足有效性要求。
        raise AssertionError("每个交易日必须包含完整的 11×11 相关矩阵")  # 关键数据契约被破坏时立即终止。
    if daily.duplicated(  # 根据当前数据状态选择处理分支。
        ["trade_date", "row_name", "column_name"]
    ).any():
        raise AssertionError("日度相关矩阵存在重复单元")  # 关键数据契约被破坏时立即终止。

    summary = (
        daily.groupby(
            ["row_name", "row_label", "column_name", "column_label"],
            sort=False,  # 保持源数据顺序，不额外排序。
            observed=True,  # 配置observed。
        )
        .agg(
            mean_pearson=("pearson", "mean"),  # 配置均值Pearson。
            std_pearson=("pearson", "std"),  # 配置标准差Pearson。
            dates=("trade_date", "nunique"),  # 配置交易日。
            mean_observations=("observations", "mean"),  # 配置均值observations。
        )
        .reset_index()
    )
    order = {name: index for index, name in enumerate(MATRIX_COLUMNS)}  # 初始化排序索引容器。
    summary["_row_order"] = summary["row_name"].map(order)  # 按约定截面口径汇总。
    summary["_column_order"] = summary["column_name"].map(order)  # 按约定截面口径汇总。
    summary.sort_values(  # 按约定截面口径汇总。
        ["_row_order", "_column_order"], inplace=True
    )
    summary.drop(  # 按约定截面口径汇总。
        columns=["_row_order", "_column_order"], inplace=True  # 初始化列容器。
    )
    summary.reset_index(drop=True, inplace=True)  # 按约定截面口径汇总。
    if len(summary) != expected_cells:  # 检查当前样本数量是否满足计算要求。
        raise AssertionError("相关矩阵汇总不含完整的 121 个单元")  # 关键数据契约被破坏时立即终止。
    matrix = (  # 按约定截面口径汇总。
        summary.pivot(
            index="row_name", columns="column_name", values="mean_pearson"
        )
        .loc[list(MATRIX_COLUMNS), list(MATRIX_COLUMNS)]
        .to_numpy(np.float64)
    )
    if not np.allclose(matrix, matrix.T, atol=1e-12, rtol=0.0):  # 检查必需数据、配置或校验结果是否缺失。
        raise AssertionError("日均相关矩阵不对称")  # 关键数据契约被破坏时立即终止。
    if not np.allclose(np.diag(matrix), 1.0, atol=1e-12, rtol=0.0):  # 检查必需数据、配置或校验结果是否缺失。
        raise AssertionError("日均相关矩阵对角线不为 1")  # 关键数据契约被破坏时立即终止。
    return summary  # 返回汇总。


def analyze_barra_exposures(  # 只读计算源因子的 Barra/反转覆盖、相关性与 11×11 矩阵。
    source_run_name: str = DEFAULT_SOURCE_RUN_NAME,  # 接收来源执行名称。
    cne5s_path: str | Path = DEFAULT_CNE5S_PATH,  # 接收cne5s路径。
    vivit_data_root: str | Path = DEFAULT_VIVIT_DATA_ROOT,  # 接收vivit数据根目录。
    minimum_match_rate: float = MINIMUM_MATCH_RATE,  # 接收minimummatch比率。
    eod_path: str | Path = DEFAULT_EOD_PATH,  # 接收日行情路径。
    calendar_path: str | Path = DEFAULT_CALENDAR_PATH,  # 接收交易日历路径。
    source_factor_column: str = SOURCE_FACTOR_COLUMN,  # 接收来源因子列。
) -> dict[str, Any]:  # 声明当前函数的返回类型。
    """只读计算源因子的 Barra/反转覆盖、相关性与 11×11 矩阵。"""

    source_run_name = _validate_run_name(source_run_name)
    source_factor_column = _validate_source_factor_column(
        source_factor_column  # 汇入来源因子列。
    )
    if not 0 <= minimum_match_rate <= 1:  # 检查当前数值是否越过业务或数值阈值。
        raise ValueError("minimum_match_rate 必须位于 [0,1]")  # 输入数据不满足业务或数值约束时立即终止。
    vivit_root = Path(vivit_data_root).resolve()  # 构造vivit根目录。
    cne5s = Path(cne5s_path).resolve()  # 构造cne5s。
    eod = Path(eod_path).resolve()  # 构造日行情。
    calendar_file = Path(calendar_path).resolve()  # 构造交易日历文件。
    if not vivit_root.is_dir():  # 检查所需文件或目录是否存在。
        raise FileNotFoundError(vivit_root)  # 目标文件不存在时立即终止。
    _validate_cne5s_schema(cne5s)
    _validate_price_sources(eod, calendar_file)
    calendar_dates = _load_sse_calendar(calendar_file)
    training, _ = _training_manifest(source_run_name, vivit_root)  # 读取或生成来源清单。
    counts = _factor_counts(training)  # 计算数量。
    prediction_root = vivit_root / "predictions" / "runs" / source_run_name  # 构造预测根目录。
    match_records: list[dict[str, Any]] = []  # 初始化matchrecords容器。
    correlation_records: list[dict[str, Any]] = []  # 计算相关系数records。
    matrix_correlation_records: list[dict[str, Any]] = []  # 计算矩阵相关系数records。

    for year, expected_rows in counts.items():  # 遍历数量items，逐项处理年份预期行数。
        path = prediction_root / PREDICTION_PATTERN.format(year=year)
        if expected_rows == 0:  # 按当前模式、标识或状态选择处理分支。
            if path.exists():  # 检查目标文件或目录是否满足存在性要求。
                raise AssertionError(  # 关键数据契约被破坏时立即终止。
                    f"{year} factor_counts=0 但存在预测文件：{path}"
                )
            match_records.append(  # 把当前结果追加到汇总列表。
                {
                    "year": year,  # 记录年份。
                    "prediction_rows": 0,  # 记录预测行数。
                    "cne5s_rows": 0,  # 记录cne5s行数。
                    "matched_rows": 0,  # 记录匹配行数。
                    "unmatched_rows": 0,  # 记录未匹配行数。
                    "match_rate": np.nan,  # 记录match比例。
                    "reversal_valid_rows": 0,  # 记录反转有效行数。
                    "reversal_missing_rows": 0,  # 记录反转缺失值行数。
                    "reversal_coverage": np.nan,  # 记录反转覆盖率。
                    "reversal_correlation_rows": 0,  # 记录反转相关系数行数。
                    "dates": 0,  # 记录交易日。
                    "first_date": None,  # 记录首个交易日。
                    "last_date": None,  # 记录最后一个交易日。
                    "status": "empty_prediction_expected",  # 把 status 映射为"empty_prediction_expected"。
                }
            )
            continue  # 跳过当前项并处理下一项。
        prediction = _load_source_prediction(
            path,  # 汇入路径。
            year,
            expected_rows,
            full=True,  # 配置full。
            source_factor_column=source_factor_column,  # 配置来源因子列。
        )
        prediction = _attach_reversal_20d(  # 按股票与日期业务键补充字段。
            prediction, eod, calendar_dates, year  # 传入预测、日行情、交易日历交易日、年份。
        )
        barra = _load_cne5s_year(cne5s, year)
        joined = _join_prediction_and_cne5s(prediction, barra, year)
        record = _match_record(joined, year, len(barra))  # 核对两侧业务键并统计匹配率。
        _check_match_rate(record, minimum_match_rate)
        match_records.append(record)  # 把当前结果追加到汇总列表。
        correlation_records.extend(_daily_correlations(joined, year))  # 把一组结果并入当前列表。
        matrix_correlation_records.extend(  # 把一组结果并入当前列表。
            _daily_correlation_matrix(joined, year)
        )

    input_audit = pd.DataFrame(match_records)
    reversal_price_coverage = input_audit.loc[:, [
        "year", "prediction_rows", "reversal_valid_rows",
        "reversal_missing_rows", "reversal_coverage",
        "reversal_correlation_rows", "dates", "first_date",
        "last_date", "status",
    ]].copy()
    daily_correlations = pd.DataFrame(correlation_records)  # 计算dailycorrelations。
    if daily_correlations.empty:  # 检查结果表是否为空。
        raise ValueError("源 run 没有可用于相关性分析的非空预测")  # 输入数据不满足业务或数值约束时立即终止。
    if set(daily_correlations["factor_name"].unique()) != set(  # 按当前模式、标识或状态选择处理分支。
        CORRELATION_COLUMNS
    ):
        raise AssertionError("日度相关性未恰好覆盖十个测试变量")  # 关键数据契约被破坏时立即终止。
    annual_correlations, correlation_summary = _correlation_summaries(  # 计算当前截面相关性。
        daily_correlations
    )
    daily_correlation_matrix = pd.DataFrame(matrix_correlation_records)  # 计算daily相关系数矩阵。
    correlation_matrix_summary = _correlation_matrix_summary(  # 按约定截面口径汇总。
        daily_correlation_matrix
    )
    return {  # 返回只读计算源因子的 Barra/反转覆盖、相关性与 11×11 矩阵。
        "source_run_name": source_run_name,  # 记录来源运行名称。
        "source_factor_column": source_factor_column,  # 记录来源因子字段。
        "input_match_audit": input_audit,  # 记录输入matchaudit。
        "reversal_price_coverage": reversal_price_coverage,  # 记录反转价格覆盖率。
        "daily_correlations": daily_correlations,  # 记录日度correlations。
        "annual_correlations": annual_correlations,  # 记录年度correlations。
        "correlation_summary": correlation_summary,  # 记录相关系数汇总。
        "daily_correlation_matrix": daily_correlation_matrix,  # 记录日度相关系数矩阵。
        "correlation_matrix_summary": correlation_matrix_summary,  # 记录相关系数矩阵汇总。
        "writes_performed": False,  # 记录writesperformed。
    }


def _save_heatmap(summary: pd.DataFrame, path: Path) -> None:  # 生成并保存该评价环节对应的图表，缺失指标按既定口径处理。
    """生成并保存该评价环节对应的图表，缺失指标按既定口径处理。"""
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    import seaborn as sns

    cjk_font_path = Path.home() / ".local/share/fonts/NotoSansCJKsc-Regular.otf"  # 构造cjkfont路径。
    if cjk_font_path.exists():  # 检查目标文件或目录是否满足存在性要求。
        font_manager.fontManager.addfont(str(cjk_font_path))

    values = summary.pivot(
        index="row_name", columns="column_name", values="mean_pearson"
    ).loc[list(MATRIX_COLUMNS), list(MATRIX_COLUMNS)]
    labels = [MATRIX_LABELS[column] for column in MATRIX_COLUMNS]  # 初始化标签容器。
    matrix = pd.DataFrame(values.to_numpy(np.float64), index=labels, columns=labels)  # 转换并统一矩阵的数据表示。
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    figure, axis = plt.subplots(figsize=(12.5, 10.5))
    sns.heatmap(  # 绘制因子相关性热力图。
        matrix,
        ax=axis,  # 配置ax。
        cmap="RdBu_r",  # 配置cmap。
        center=0,  # 配置center。
        vmin=-1,  # 配置vmin。
        vmax=1,  # 配置vmax。
        annot=True,  # 配置annot。
        fmt=".2f",  # 配置fmt。
        square=True,  # 配置square。
        linewidths=0.5,  # 配置linewidths。
        linecolor="white",  # 配置linecolor。
        cbar=True,  # 配置cbar。
        annot_kws={"fontsize": 8.5},  # 配置annotkws。
    )
    axis.set_title(
        "ViViT 因子、9 个 CNE5S 风格与 20 日反转的日均横截面 Pearson",
        fontsize=13,  # 配置fontsize。
        pad=14,  # 配置pad。
    )
    axis.set_xlabel("")
    axis.set_ylabel("")
    axis.tick_params(axis="x", rotation=45)
    axis.tick_params(axis="y", rotation=0)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)  # 确保目标目录存在。
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)

def _save_reversal_timeseries(  # 生成并保存该评价环节对应的图表，缺失指标按既定口径处理。
    daily_correlations: pd.DataFrame, path: Path
) -> None:  # 声明当前函数的返回类型。
    """生成并保存该评价环节对应的图表，缺失指标按既定口径处理。"""
    from matplotlib import font_manager
    import matplotlib.pyplot as plt

    reversal = daily_correlations.loc[  # 计算当前截面相关性。
        daily_correlations["factor_name"].eq(REVERSAL_COLUMN)
    ].copy()
    if reversal.empty:  # 检查结果表是否为空。
        raise ValueError("缺少 20 日反转日度相关性")  # 输入数据不满足业务或数值约束时立即终止。
    reversal["trade_date"] = pd.to_datetime(  # 解析并统一reversal。
        reversal["trade_date"], format="%Y%m%d", errors="raise"
    )
    reversal.sort_values("trade_date", inplace=True)
    if reversal["trade_date"].duplicated().any():  # 检查当前掩码或数据是否满足有效性要求。
        raise ValueError("20 日反转日度相关性日期重复")  # 输入数据不满足业务或数值约束时立即终止。

    cjk_font_path = Path.home() / ".local/share/fonts/NotoSansCJKsc-Regular.otf"  # 构造cjkfont路径。
    if cjk_font_path.exists():  # 检查目标文件或目录是否满足存在性要求。
        font_manager.fontManager.addfont(str(cjk_font_path))
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    figure, axis = plt.subplots(figsize=(14, 4.5))
    axis.plot(
        reversal["trade_date"], reversal["pearson"],
        color="#7c3aed", linewidth=0.8, alpha=0.9,
        label="20日反转",  # 配置标签。
    )
    axis.axhline(0.0, color="#111827", linewidth=0.8, linestyle="--")
    axis.set_title("ViViT 因子与 20 日反转的日度横截面 Pearson")
    axis.set_xlabel("交易日期")
    axis.set_ylabel("横截面 Pearson")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)  # 确保目标目录存在。
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _derivation_payload(  # 构造参与稳定签名和复用判断的配置载荷。
    source_run_name: str,  # 接收来源执行名称。
    residual_run_name: str,  # 接收残差执行名称。
    training_path: Path,  # 接收训练路径。
    training: dict[str, Any],
    cne5s_fingerprint: dict[str, Any],
    eod_fingerprint: dict[str, Any],
    calendar_fingerprint: dict[str, Any],
    prediction_fingerprints: list[dict[str, Any]],
    minimum_match_rate: float,  # 接收minimummatch比率。
    source_factor_column: str,  # 接收来源因子列。
) -> dict[str, Any]:  # 声明当前函数的返回类型。
    """构造参与稳定签名和复用判断的配置载荷。"""
    return {  # 返回构造参与稳定签名和复用判断的配置载荷。
        "schema_version": DERIVATION_SCHEMA_VERSION,  # 记录schema版本。
        "source_run_name": source_run_name,  # 记录来源运行名称。
        "residual_run_name": residual_run_name,  # 记录残差运行名称。
        "source_training_manifest": {  # 记录来源训练清单。
            "path": str(training_path.resolve()),  # 记录路径。
            "size": int(training_path.stat().st_size),  # 记录大小。
            "mtime_ns": int(training_path.stat().st_mtime_ns),  # 记录修改时间纳秒。
            "source_dataset_signature": training.get(  # 记录来源dataset签名，用于来源一致性校验。
                "source_dataset_signature"
            ),
            "source_rolling_signature": training.get(  # 记录来源rolling签名，用于来源一致性校验。
                "source_rolling_signature"
            ),
            "sequence_signature": training.get("sequence_signature"),  # 记录sequence签名，用于来源一致性校验。
        },
        "cne5s": cne5s_fingerprint,  # 记录cne5s。
        "wind_eod_prices": eod_fingerprint,  # 记录wind日行情prices。
        "sse_calendar": calendar_fingerprint,  # 记录SSE交易日历。
        "source_predictions": prediction_fingerprints,  # 记录来源预测。
        "source_factor_column": source_factor_column,  # 记录来源因子字段。
        "style_columns": list(STYLE_COLUMNS),  # 记录风格字段列表。
        "correlation_columns": list(CORRELATION_COLUMNS),  # 记录相关系数字段列表。
        "matrix_columns": list(MATRIX_COLUMNS),  # 记录矩阵字段列表。
        "regression_columns": list(REGRESSION_COLUMNS),  # 记录regression字段列表。
        "reversal": {  # 记录反转。
            "column": REVERSAL_COLUMN,  # 记录字段。
            "label": STYLE_LABELS[REVERSAL_COLUMN],  # 记录标签。
            "lookback_sse_trading_days": REVERSAL_LOOKBACK_DAYS,  # 记录回看窗口SSEtradingdays。
            "price_column": "s_dq_adjclose",  # 把 price_column 映射为"s_dq_adjclose"。
            "formula": (  # 记录formula。
                "-(s_dq_adjclose(T)/s_dq_adjclose(T-20_SSE_days)-1)"
            ),
            "price_match": "exact_T_and_T_minus_20_no_fill",  # 把 price_match 映射为"exact_T_and_T_minus_20_no_fill"。
            "winsorization": "none",  # 把 winsorization 映射为"none"。
            "regression_scaling": "daily_zscore_ddof_0_valid_then_fill_zero",  # 把 regression_scaling 映射为"daily_zscore_ddof_0_valid_then_fill_zero"。
        },
        "industry_column": INDUSTRY_COLUMN,  # 记录行业字段。
        "excluded_columns": [INDUSTRY_COLUMN, "cne5s_resvol"],  # 记录excluded字段列表。
        "correlation_policy": (  # 声明相关系数处理口径。
            "daily_complete_case_barra_matched_and_reversal_valid_pearson_then_time_mean"
        ),
        "correlation_inference": {  # 记录相关系数inference。
            "naive_t_stat": "mean_over_iid_standard_error",  # 把 naive_t_stat 映射为"mean_over_iid_standard_error"。
            "corrected_t_stat": "newey_west_hac_bartlett",  # 把 corrected_t_stat 映射为"newey_west_hac_bartlett"。
            "newey_west_max_lag": HAC_MAX_LAG,  # 记录neweywestmaxlag。
        },
        "correlation_matrix_policy": (  # 声明相关系数矩阵处理口径。
            "daily_complete_case_11x11_pearson_then_cellwise_time_mean"
        ),
        "regression_policy": (  # 声明regression处理口径。
            "daily_equal_weight_ols_9_cne5s_styles_plus_reversal20_plus_industry_fixed_effects"
        ),
        "missing_barra_policy": (  # 声明缺失值Barra处理口径。
            "style_zero_industry_unknown_remove_unknown_group_mean"
        ),
        "missing_reversal_policy": "daily_zscore_valid_then_fill_zero",  # 把 missing_reversal_policy 映射为"daily_zscore_valid_then_fill_zero"。
        "match_policy": "exact_trade_date_and_assetid_no_forward_fill",  # 把 match_policy 映射为"exact_trade_date_and_assetid_no_forward_fill"。
        "residual_scaling": "raw_ols_residual_no_daily_zscore",  # 把 residual_scaling 映射为"raw_ols_residual_no_daily_zscore"。
        "minimum_match_rate": minimum_match_rate,  # 记录minimummatch比例。
        "orthogonality_tolerance": ORTHOGONALITY_TOLERANCE,  # 记录orthogonalitytolerance。
    }


def _existing_result(  # 检查同签名派生 run 是否完整可复用，并返回其现有结果。
    quality_root: Path,  # 接收质量根目录。
    prediction_root: Path,  # 接收预测根目录。
    expected_signature: str,  # 接收预期签名。
    counts: dict[int, int],
    *,
    overwrite_existing: bool = False,  # 接收overwriteexisting。
) -> dict[str, Any] | None:  # 声明当前函数的返回类型。
    """检查同签名派生 run 是否完整可复用，并返回其现有结果。"""
    neutralization_root = quality_root / "barra_neutralization"  # 构造neutralization根目录。
    manifest_path = neutralization_root / "neutralization_manifest.json"  # 构造清单路径。
    if not quality_root.exists() and not prediction_root.exists():  # 检查目标文件或目录是否满足存在性要求。
        return None  # 当前分支无可用结果，返回空值。
    if not manifest_path.exists():  # 检查目标文件或目录是否满足存在性要求。
        if overwrite_existing:  # 根据当前数据状态选择处理分支。
            return None  # 当前分支无可用结果，返回空值。
        raise RuntimeError(  # 当前运行环境或任务状态不支持继续执行。
            "派生 run 已存在但缺少 neutralization manifest；"
            "如需替换必须显式设置 overwrite_existing=True"
        )
    manifest = _read_json(manifest_path)  # 读取清单。
    if manifest.get("signature") != expected_signature:  # 按当前模式、标识或状态选择处理分支。
        if overwrite_existing:  # 根据当前数据状态选择处理分支。
            return None  # 当前分支无可用结果，返回空值。
        raise RuntimeError(  # 当前运行环境或任务状态不支持继续执行。
            "已有派生 run 与当前源数据或配置签名不一致；"
            "如需替换必须显式设置 overwrite_existing=True"
        )
    for year, count in counts.items():  # 遍历数量items，逐项处理年份数量。
        path = prediction_root / PREDICTION_PATTERN.format(year=year)
        if count:  # 根据当前数据状态选择处理分支。
            if not path.exists():  # 检查目标文件或目录是否满足存在性要求。
                raise FileNotFoundError(path)  # 目标文件不存在时立即终止。
            if pq.ParquetFile(path).metadata.num_rows != count:  # 按当前模式、标识或状态选择处理分支。
                raise AssertionError(f"{path.name} 行数与 manifest 不一致")  # 关键数据契约被破坏时立即终止。
        elif path.exists():  # 检查上一条件不成立时的备选分支。
            raise AssertionError(f"{year} 应为空但存在派生预测文件")  # 关键数据契约被破坏时立即终止。
    required = {  # 初始化必需容器。
        "input_match_audit": "input_match_audit.parquet",  # 把 input_match_audit 映射为"input_match_audit.parquet"。
        "reversal_price_coverage": "reversal_price_coverage.parquet",  # 把 reversal_price_coverage 映射为"reversal_price_coverage.parquet"。
        "daily_correlations": "daily_correlations.parquet",  # 把 daily_correlations 映射为"daily_correlations.parquet"。
        "annual_correlations": "annual_correlations.parquet",  # 把 annual_correlations 映射为"annual_correlations.parquet"。
        "correlation_summary": "correlation_summary.parquet",  # 把 correlation_summary 映射为"correlation_summary.parquet"。
        "daily_correlation_matrix": "daily_correlation_matrix.parquet",  # 把 daily_correlation_matrix 映射为"daily_correlation_matrix.parquet"。
        "correlation_matrix_summary": "correlation_matrix_summary.parquet",  # 把 correlation_matrix_summary 映射为"correlation_matrix_summary.parquet"。
        "regression_diagnostics": "daily_regression_diagnostics.parquet",  # 把 regression_diagnostics 映射为"daily_regression_diagnostics.parquet"。
        "residual_correlations": "residual_correlation_audit.parquet",  # 把 residual_correlations 映射为"residual_correlation_audit.parquet"。
    }
    result: dict[str, Any] = {  # 初始化结果容器。
        name: pd.read_parquet(neutralization_root / file_name)  # 接收名称。
        for name, file_name in required.items()  # 遍历必需items，逐项处理名称文件名称。
    }
    heatmap = neutralization_root / "correlation_heatmap.png"
    reversal_timeseries = (
        neutralization_root / "reversal_20d_correlation_timeseries.png"
    )
    for path in (heatmap, reversal_timeseries):  # 遍历heatmapreversaltimeseries，逐项处理路径。
        if not path.exists():  # 检查目标文件或目录是否满足存在性要求。
            raise FileNotFoundError(path)  # 目标文件不存在时立即终止。
    result.update(  # 用当前配置更新参数字典。
        {
            "source_run_name": manifest["source_run_name"],  # 记录来源运行名称。
            "source_factor_column": manifest["source_factor_column"],  # 记录来源因子字段。
            "residual_run_name": manifest["residual_run_name"],  # 记录残差运行名称。
            "prediction_root": prediction_root,  # 记录预测根目录。
            "quality_root": quality_root,  # 记录质量根目录。
            "neutralization_root": neutralization_root,  # 记录neutralization根目录。
            "heatmap_path": heatmap,  # 记录heatmap路径。
            "reversal_timeseries_path": reversal_timeseries,  # 记录反转timeseries路径。
            "manifest_path": manifest_path,  # 记录清单路径。
            "reused": True,  # 记录reused。
            "overwritten": False,  # 记录overwritten。
        }
    )
    return result  # 返回结果。


def _replace_staged_roots(  # 以临时 tombstone 支持失败回滚。
    stage_predictions: Path,  # 接收stage预测。
    stage_quality: Path,  # 接收stage质量。
    prediction_root: Path,  # 接收预测根目录。
    quality_root: Path,  # 接收质量根目录。
) -> bool:  # 声明当前函数的返回类型。
    """以临时 tombstone 支持失败回滚；成功后不保留旧 run。"""

    token = f"{os.getpid()}.{time.time_ns()}"
    old_prediction = prediction_root.with_name(  # 构造old预测。
        f".{prediction_root.name}.replacing.{token}"
    )
    old_quality = quality_root.with_name(  # 构造old质量。
        f".{quality_root.name}.replacing.{token}"
    )
    had_existing = prediction_root.exists() or quality_root.exists()
    prediction_installed = False
    quality_installed = False
    try:  # 执行可能失败的外部读取或计算步骤。
        if prediction_root.exists():  # 检查目标文件或目录是否满足存在性要求。
            prediction_root.replace(old_prediction)  # 用临时文件原子替换正式结果。
        if quality_root.exists():  # 检查目标文件或目录是否满足存在性要求。
            quality_root.replace(old_quality)  # 用临时文件原子替换正式结果。
        stage_predictions.replace(prediction_root)  # 用临时文件原子替换正式结果。
        prediction_installed = True
        stage_quality.replace(quality_root)  # 用临时文件原子替换正式结果。
        quality_installed = True
    except Exception:  # 捕获预期异常并执行对应恢复逻辑。
        if quality_installed and quality_root.exists():  # 检查目标文件或目录是否满足存在性要求。
            quality_root.replace(stage_quality)  # 用临时文件原子替换正式结果。
        if prediction_installed and prediction_root.exists():  # 检查目标文件或目录是否满足存在性要求。
            prediction_root.replace(stage_predictions)  # 用临时文件原子替换正式结果。
        if old_quality.exists():  # 检查目标文件或目录是否满足存在性要求。
            old_quality.replace(quality_root)  # 用临时文件原子替换正式结果。
        if old_prediction.exists():  # 检查目标文件或目录是否满足存在性要求。
            old_prediction.replace(prediction_root)  # 用临时文件原子替换正式结果。
        raise
    else:  # 处理上述条件均不成立的情况。
        if old_quality.exists():  # 检查目标文件或目录是否满足存在性要求。
            shutil.rmtree(old_quality)
        if old_prediction.exists():  # 检查目标文件或目录是否满足存在性要求。
            shutil.rmtree(old_prediction)
    return had_existing  # 返回hadexisting。


def build_barra_residual_run(  # 生成相关性产物和可直接交给 Notebook 03 的 Barra 残差派生 run。
    source_run_name: str = DEFAULT_SOURCE_RUN_NAME,  # 接收来源执行名称。
    residual_run_name: str = DEFAULT_RESIDUAL_RUN_NAME,  # 接收残差执行名称。
    cne5s_path: str | Path = DEFAULT_CNE5S_PATH,  # 接收cne5s路径。
    vivit_data_root: str | Path = DEFAULT_VIVIT_DATA_ROOT,  # 接收vivit数据根目录。
    minimum_match_rate: float = MINIMUM_MATCH_RATE,  # 接收minimummatch比率。
    eod_path: str | Path = DEFAULT_EOD_PATH,  # 接收日行情路径。
    calendar_path: str | Path = DEFAULT_CALENDAR_PATH,  # 接收交易日历路径。
    overwrite_existing: bool = False,  # 接收overwriteexisting。
    source_factor_column: str = SOURCE_FACTOR_COLUMN,  # 接收来源因子列。
) -> dict[str, Any]:  # 声明当前函数的返回类型。
    """生成相关性产物和可直接交给 Notebook 03 的 Barra 残差派生 run。"""

    source_run_name = _validate_run_name(source_run_name)
    residual_run_name = _validate_run_name(residual_run_name)
    source_factor_column = _validate_source_factor_column(
        source_factor_column
    )
    if residual_run_name == source_run_name:  # 按当前模式、标识或状态选择处理分支。
        raise ValueError("派生 run 不能与源 run 同名")  # 输入数据不满足业务或数值约束时立即终止。
    if not 0 <= minimum_match_rate <= 1:  # 检查当前数值是否越过业务或数值阈值。
        raise ValueError("minimum_match_rate 必须位于 [0,1]")  # 输入数据不满足业务或数值约束时立即终止。
    if not isinstance(overwrite_existing, bool):  # 检查对象类型是否符合接口约定。
        raise TypeError("overwrite_existing 必须是 bool")  # 输入类型不符合接口约定时立即终止。
    vivit_root = Path(vivit_data_root).resolve()  # 构造vivit根目录。
    cne5s = Path(cne5s_path).resolve()  # 构造cne5s。
    eod = Path(eod_path).resolve()  # 构造日行情。
    calendar_file = Path(calendar_path).resolve()  # 构造交易日历文件。
    cne5s_fingerprint = _validate_cne5s_schema(cne5s)
    eod_fingerprint, calendar_fingerprint = _validate_price_sources(
        eod, calendar_file  # 传入日行情、交易日历文件。
    )
    calendar_dates = _load_sse_calendar(calendar_file)
    training, training_path = _training_manifest(source_run_name, vivit_root)  # 读取或生成来源清单。
    counts = _factor_counts(training)  # 计算数量。
    source_prediction_root = (  # 构造来源预测根目录。
        vivit_root / "predictions" / "runs" / source_run_name
    )
    prediction_fingerprints = [  # 初始化预测fingerprints容器。
        _fingerprint_parquet(
            source_prediction_root / PREDICTION_PATTERN.format(year=year)
        )
        for year, count in counts.items()  # 遍历数量items，逐项处理年份数量。
        if count  # 根据当前数据状态选择处理分支。
    ]
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
    signature = _payload_signature(derivation)
    prediction_root = (  # 构造预测根目录。
        vivit_root / "predictions" / "runs" / residual_run_name
    )
    quality_root = vivit_root / "quality" / "runs" / residual_run_name  # 构造质量根目录。
    existing = _existing_result(  # 仅复用签名一致的产物。
        quality_root, prediction_root, signature, counts,  # 传入质量根目录、预测根目录、签名、数量。
        overwrite_existing=overwrite_existing,  # 配置overwriteexisting。
    )
    if existing is not None:  # 检查必需数据、配置或校验结果是否缺失。
        return existing  # 返回existing。

    staging_parent = Path(  # 构造stagingparent。
        tempfile.mkdtemp(
            prefix=f".{residual_run_name}.",  # 配置prefix。
            dir=vivit_root,  # 配置目录。
        )
    )
    stage_predictions = staging_parent / "predictions"
    stage_quality = staging_parent / "quality"
    stage_neutralization = stage_quality / "barra_neutralization"
    stage_predictions.mkdir(parents=True)  # 确保目标目录存在。
    stage_neutralization.mkdir(parents=True)  # 确保目标目录存在。

    match_records: list[dict[str, Any]] = []  # 初始化matchrecords容器。
    correlation_records: list[dict[str, Any]] = []  # 计算相关系数records。
    matrix_correlation_records: list[dict[str, Any]] = []  # 计算矩阵相关系数records。
    diagnostic_frames: list[pd.DataFrame] = []  # 接收diagnostic帧。
    residual_correlation_frames: list[pd.DataFrame] = []  # 接收残差相关系数帧。
    output_counts: dict[str, int] = {}  # 计算输出数量。
    try:  # 执行可能失败的外部读取或计算步骤。
        for year, expected_rows in counts.items():  # 遍历数量items，逐项处理年份预期行数。
            if expected_rows == 0:  # 按当前模式、标识或状态选择处理分支。
                output_counts[str(year)] = 0  # 计算输出数量。
                match_records.append(  # 把当前结果追加到汇总列表。
                    {
                        "year": year,  # 记录年份。
                        "prediction_rows": 0,  # 记录预测行数。
                        "cne5s_rows": 0,  # 记录cne5s行数。
                        "matched_rows": 0,  # 记录匹配行数。
                        "unmatched_rows": 0,  # 记录未匹配行数。
                        "match_rate": np.nan,  # 记录match比例。
                        "reversal_valid_rows": 0,  # 记录反转有效行数。
                        "reversal_missing_rows": 0,  # 记录反转缺失值行数。
                        "reversal_coverage": np.nan,  # 记录反转覆盖率。
                        "reversal_correlation_rows": 0,  # 记录反转相关系数行数。
                        "dates": 0,  # 记录交易日。
                        "first_date": None,  # 记录首个交易日。
                        "last_date": None,  # 记录最后一个交易日。
                        "status": "empty_prediction_expected",  # 把 status 映射为"empty_prediction_expected"。
                    }
                )
                continue  # 跳过当前项并处理下一项。
            source_path = (  # 构造来源路径。
                source_prediction_root
                / PREDICTION_PATTERN.format(year=year)
            )
            prediction = _load_source_prediction(
                source_path,
                year,
                expected_rows,
                full=True,  # 配置full。
                source_factor_column=source_factor_column,  # 配置来源因子列。
            )
            prediction = _attach_reversal_20d(  # 按股票与日期业务键补充字段。
                prediction, eod, calendar_dates, year  # 传入预测、日行情、交易日历交易日、年份。
            )
            barra = _load_cne5s_year(cne5s, year)
            joined = _join_prediction_and_cne5s(
                prediction, barra, year  # 传入预测、Barra、年份。
            )
            match_record = _match_record(joined, year, len(barra))  # 核对两侧业务键并统计匹配率。
            _check_match_rate(match_record, minimum_match_rate)
            match_records.append(match_record)  # 把当前结果追加到汇总列表。
            correlation_records.extend(  # 把一组结果并入当前列表。
                _daily_correlations(joined, year)
            )
            matrix_correlation_records.extend(  # 把一组结果并入当前列表。
                _daily_correlation_matrix(joined, year)
            )
            diagnostics, residual_correlations, residual = (  # 按日投影 Barra 暴露。
                _neutralize_year(joined, year)
            )
            diagnostic_frames.append(diagnostics)  # 把当前结果追加到汇总列表。
            residual_correlation_frames.append(residual_correlations)  # 把当前结果追加到汇总列表。

            output_columns: dict[str, Any] = {}  # 初始化输出列容器。
            if "sample_index" in prediction.columns:  # 检查当前值是否属于允许集合。
                output_columns["sample_index"] = prediction[
                    "sample_index"
                ].to_numpy()
            output_columns.update(  # 用当前配置更新参数字典。
                {
                    "trade_date": prediction["trade_date"].to_numpy(),  # 记录trade交易日。
                    "instrument_id": prediction[  # 记录股票id。
                        "instrument_id"
                    ].to_numpy(),
                    "source_prediction_ensemble": prediction[  # 记录来源预测ensemble。
                        SOURCE_FACTOR_COLUMN
                    ].to_numpy(),
                    "prediction_ensemble_raw": residual,  # 记录预测ensemble原始。
                    "prediction_ensemble": residual.copy(),  # 记录预测ensemble。
                    "barra_matched": joined[  # 记录Barra匹配。
                        "barra_matched"
                    ].to_numpy(bool),
                    REVERSAL_COLUMN: prediction[  # 接收reversal列。
                        REVERSAL_COLUMN
                    ].to_numpy(np.float64),
                    REVERSAL_VALID_COLUMN: prediction[  # 接收reversal有效列。
                        REVERSAL_VALID_COLUMN
                    ].to_numpy(bool),
                    "forecast_year": np.full(  # 记录forecast年份。
                        len(prediction), year, dtype=np.int64
                    ),
                    "factor_role": np.full(  # 记录因子role。
                        len(prediction),
                        "barra_reversal20_residual",
                        dtype=object,  # 统一结果的数据类型。
                    ),
                }
            )
            output = pd.DataFrame(output_columns)
            if len(output) != expected_rows:  # 检查当前样本数量是否满足计算要求。
                raise AssertionError(f"{year} 派生预测行数改变")  # 关键数据契约被破坏时立即终止。
            if not np.array_equal(  # 检查必需数据、配置或校验结果是否缺失。
                output["trade_date"].to_numpy(),
                prediction["trade_date"].to_numpy(),
            ) or not np.array_equal(
                output["instrument_id"].to_numpy(),
                prediction["instrument_id"].to_numpy(),
            ):
                raise AssertionError(f"{year} 派生预测业务键顺序改变")  # 关键数据契约被破坏时立即终止。
            if not np.array_equal(  # 检查必需数据、配置或校验结果是否缺失。
                output["prediction_ensemble_raw"].to_numpy(),
                output["prediction_ensemble"].to_numpy(),
            ):
                raise AssertionError("原始残差不应被再次标准化")  # 关键数据契约被破坏时立即终止。
            _atomic_parquet(  # 原子落盘，避免半成品。
                output,
                stage_predictions
                / PREDICTION_PATTERN.format(year=year),
            )
            output_counts[str(year)] = len(output)  # 计算输出数量。

        input_audit = pd.DataFrame(match_records)
        reversal_price_coverage = input_audit.loc[:, [
            "year", "prediction_rows", "reversal_valid_rows",
            "reversal_missing_rows", "reversal_coverage",
            "reversal_correlation_rows", "dates", "first_date",
            "last_date", "status",
        ]].copy()
        daily_correlations = pd.DataFrame(correlation_records)  # 计算dailycorrelations。
        if set(daily_correlations["factor_name"].unique()) != set(  # 按当前模式、标识或状态选择处理分支。
            CORRELATION_COLUMNS
        ):
            raise AssertionError("日度相关性未恰好覆盖十个测试变量")  # 关键数据契约被破坏时立即终止。
        annual_correlations, correlation_summary = (  # 计算当前截面相关性。
            _correlation_summaries(daily_correlations)
        )
        daily_correlation_matrix = pd.DataFrame(  # 计算daily相关系数矩阵。
            matrix_correlation_records
        )
        correlation_matrix_summary = _correlation_matrix_summary(  # 按约定截面口径汇总。
            daily_correlation_matrix
        )
        regression_diagnostics = pd.concat(  # 对齐并合并regression诊断。
            diagnostic_frames, ignore_index=True
        )
        residual_correlations = pd.concat(  # 对齐并合并残差correlations。
            residual_correlation_frames, ignore_index=True  # 计算残差相关系数帧。
        )
        artifacts = {  # 初始化artifacts容器。
            "input_match_audit.parquet": input_audit,  # 记录输入matchaudit.parquet。
            "reversal_price_coverage.parquet": reversal_price_coverage,  # 记录反转价格coverage.parquet。
            "daily_correlations.parquet": daily_correlations,  # 记录日度correlations.parquet。
            "annual_correlations.parquet": annual_correlations,  # 记录年度correlations.parquet。
            "correlation_summary.parquet": correlation_summary,  # 记录相关系数summary.parquet。
            "daily_correlation_matrix.parquet": daily_correlation_matrix,  # 记录日度相关系数matrix.parquet。
            "correlation_matrix_summary.parquet": correlation_matrix_summary,  # 记录相关系数矩阵summary.parquet。
            "daily_regression_diagnostics.parquet": regression_diagnostics,  # 记录日度regressiondiagnostics.parquet。
            "residual_correlation_audit.parquet": residual_correlations,  # 记录残差相关系数audit.parquet。
        }
        for file_name, frame in artifacts.items():  # 遍历artifactsitems，逐项处理文件名称帧。
            _atomic_parquet(frame, stage_neutralization / file_name)  # 原子落盘，避免半成品。
        heatmap_path = stage_neutralization / "correlation_heatmap.png"  # 构造heatmap路径。
        reversal_timeseries_path = (  # 构造reversaltimeseries路径。
            stage_neutralization
            / "reversal_20d_correlation_timeseries.png"
        )
        _save_heatmap(correlation_matrix_summary, heatmap_path)
        _save_reversal_timeseries(
            daily_correlations, reversal_timeseries_path  # 传入dailycorrelations、reversaltimeseries路径。
        )

        derived_training = dict(training)  # 初始化derived训练容器。
        derived_training.update(  # 用当前配置更新参数字典。
            {
                "run_name": residual_run_name,  # 记录运行名称。
                "factor_counts": output_counts,  # 记录因子数量。
                "factor_role": "barra_reversal20_residual",  # 把 factor_role 映射为"barra_reversal20_residual"。
                "source_run_name": source_run_name,  # 记录来源运行名称。
                "source_factor_column": source_factor_column,  # 记录来源因子字段。
                "factor_derivation_signature": signature,  # 记录因子derivation签名，用于来源一致性校验。
                "factor_derivation": derivation,  # 记录因子derivation。
            }
        )
        _atomic_json(  # 原子落盘，避免半成品。
            derived_training,
            stage_quality / "a2_training_manifest.json",
        )
        source_summary = (
            vivit_root
            / "quality"
            / "runs"
            / source_run_name
            / "vivit_training_summary.parquet"
        )
        if source_summary.exists():  # 检查目标文件或目录是否满足存在性要求。
            shutil.copy2(
                source_summary,
                stage_quality / "vivit_training_summary.parquet",
            )
        neutralization_manifest = {  # 初始化neutralization清单容器。
            "signature": signature,  # 记录签名。
            **derivation,
            "factor_counts": output_counts,  # 记录因子数量。
            "match_summary": input_audit.to_dict("records"),  # 记录match汇总。
            "artifacts": sorted(  # 记录artifacts。
                [
                    *artifacts,
                    "correlation_heatmap.png",
                    "reversal_20d_correlation_timeseries.png",
                ]
            ),
        }
        _atomic_json(  # 原子落盘，避免半成品。
            neutralization_manifest,
            stage_neutralization / "neutralization_manifest.json",
        )

        _existing_result(  # 仅复用签名一致的产物。
            stage_quality, stage_predictions, signature, counts  # 传入stage质量、stage预测、签名、数量。
        )
        prediction_root.parent.mkdir(parents=True, exist_ok=True)  # 确保目标目录存在。
        quality_root.parent.mkdir(parents=True, exist_ok=True)  # 确保目标目录存在。
        overwritten = _replace_staged_roots(
            stage_predictions, stage_quality, prediction_root, quality_root  # 传入stage预测、stage质量、预测根目录、质量根目录。
        )
    finally:  # 无论前序步骤是否成功都执行资源清理。
        shutil.rmtree(staging_parent, ignore_errors=True)

    return {  # 返回生成的结果与诊断。
        "source_run_name": source_run_name,  # 记录来源运行名称。
        "source_factor_column": source_factor_column,  # 记录来源因子字段。
        "residual_run_name": residual_run_name,  # 记录残差运行名称。
        "input_match_audit": input_audit,  # 记录输入matchaudit。
        "reversal_price_coverage": reversal_price_coverage,  # 记录反转价格覆盖率。
        "daily_correlations": daily_correlations,  # 记录日度correlations。
        "annual_correlations": annual_correlations,  # 记录年度correlations。
        "correlation_summary": correlation_summary,  # 记录相关系数汇总。
        "daily_correlation_matrix": daily_correlation_matrix,  # 记录日度相关系数矩阵。
        "correlation_matrix_summary": correlation_matrix_summary,  # 记录相关系数矩阵汇总。
        "regression_diagnostics": regression_diagnostics,  # 记录regression诊断。
        "residual_correlations": residual_correlations,  # 记录残差correlations。
        "prediction_root": prediction_root,  # 记录预测根目录。
        "quality_root": quality_root,  # 记录质量根目录。
        "neutralization_root": (  # 记录neutralization根目录。
            quality_root / "barra_neutralization"
        ),
        "heatmap_path": (  # 记录heatmap路径。
            quality_root
            / "barra_neutralization"
            / "correlation_heatmap.png"
        ),
        "reversal_timeseries_path": (  # 记录反转timeseries路径。
            quality_root
            / "barra_neutralization"
            / "reversal_20d_correlation_timeseries.png"
        ),
        "manifest_path": (  # 记录清单路径。
            quality_root
            / "barra_neutralization"
            / "neutralization_manifest.json"
        ),
        "reused": False,  # 记录reused。
        "overwritten": overwritten,  # 记录overwritten。
    }


__all__ = [  # 初始化all容器。
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
def audit_barra_inputs(  # 只读审计源预测、CNE5S schema、精确同日业务键和年度覆盖率。
    source_run_name: str = DEFAULT_SOURCE_RUN_NAME,  # 接收来源执行名称。
    cne5s_path: str | Path = DEFAULT_CNE5S_PATH,  # 接收cne5s路径。
    vivit_data_root: str | Path = DEFAULT_VIVIT_DATA_ROOT,  # 接收vivit数据根目录。
    minimum_match_rate: float = MINIMUM_MATCH_RATE,  # 接收minimummatch比率。
    eod_path: str | Path = DEFAULT_EOD_PATH,  # 接收日行情路径。
    calendar_path: str | Path = DEFAULT_CALENDAR_PATH,  # 接收交易日历路径。
    source_factor_column: str = SOURCE_FACTOR_COLUMN,  # 接收来源因子列。
) -> pd.DataFrame:  # 声明当前函数的返回类型。
    """只读审计源预测、CNE5S schema、精确同日业务键和年度覆盖率。"""

    source_run_name = _validate_run_name(source_run_name)
    source_factor_column = _validate_source_factor_column(
        source_factor_column
    )
    if not 0 <= minimum_match_rate <= 1:  # 检查当前数值是否越过业务或数值阈值。
        raise ValueError("minimum_match_rate 必须位于 [0,1]")  # 输入数据不满足业务或数值约束时立即终止。
    vivit_root = Path(vivit_data_root).resolve()  # 构造vivit根目录。
    cne5s = Path(cne5s_path).resolve()  # 构造cne5s。
    eod = Path(eod_path).resolve()  # 构造日行情。
    calendar_file = Path(calendar_path).resolve()  # 构造交易日历文件。
    if not vivit_root.is_dir():  # 检查所需文件或目录是否存在。
        raise FileNotFoundError(vivit_root)  # 目标文件不存在时立即终止。
    cne5s_fingerprint = _validate_cne5s_schema(cne5s)
    eod_fingerprint, calendar_fingerprint = _validate_price_sources(
        eod, calendar_file  # 传入日行情、交易日历文件。
    )
    calendar_dates = _load_sse_calendar(calendar_file)
    training, training_path = _training_manifest(source_run_name, vivit_root)  # 读取或生成来源清单。
    counts = _factor_counts(training)  # 计算数量。
    prediction_root = vivit_root / "predictions" / "runs" / source_run_name  # 构造预测根目录。
    records: list[dict[str, Any]] = []  # 初始化records容器。
    prediction_fingerprints: list[dict[str, Any]] = []  # 初始化预测fingerprints容器。

    for year, expected_rows in counts.items():  # 遍历数量items，逐项处理年份预期行数。
        path = prediction_root / PREDICTION_PATTERN.format(year=year)
        if expected_rows == 0:  # 按当前模式、标识或状态选择处理分支。
            if path.exists():  # 检查目标文件或目录是否满足存在性要求。
                raise AssertionError(  # 关键数据契约被破坏时立即终止。
                    f"{year} factor_counts=0 但存在预测文件：{path}"
                )
            records.append(  # 把当前结果追加到汇总列表。
                {
                    "year": year,  # 记录年份。
                    "prediction_rows": 0,  # 记录预测行数。
                    "cne5s_rows": 0,  # 记录cne5s行数。
                    "matched_rows": 0,  # 记录匹配行数。
                    "unmatched_rows": 0,  # 记录未匹配行数。
                    "match_rate": np.nan,  # 记录match比例。
                    "reversal_valid_rows": 0,  # 记录反转有效行数。
                    "reversal_missing_rows": 0,  # 记录反转缺失值行数。
                    "reversal_coverage": np.nan,  # 记录反转覆盖率。
                    "reversal_correlation_rows": 0,  # 记录反转相关系数行数。
                    "dates": 0,  # 记录交易日。
                    "first_date": None,  # 记录首个交易日。
                    "last_date": None,  # 记录最后一个交易日。
                    "status": "empty_prediction_expected",  # 把 status 映射为"empty_prediction_expected"。
                }
            )
            continue  # 跳过当前项并处理下一项。
        prediction_fingerprints.append(_fingerprint_parquet(path))  # 把当前结果追加到汇总列表。
        prediction = _load_source_prediction(
            path,
            year,
            expected_rows,
            full=False,  # 配置full。
            source_factor_column=source_factor_column,  # 配置来源因子列。
        )
        prediction = _attach_reversal_20d(  # 按股票与日期业务键补充字段。
            prediction, eod, calendar_dates, year  # 传入预测、日行情、交易日历交易日、年份。
        )
        barra = _load_cne5s_year(cne5s, year)
        joined = _join_prediction_and_cne5s(prediction, barra, year)
        record = _match_record(joined, year, len(barra))  # 核对两侧业务键并统计匹配率。
        _check_match_rate(record, minimum_match_rate)
        records.append(record)  # 把当前结果追加到汇总列表。

    audit = pd.DataFrame(records)
    audit.attrs.update(  # 用当前配置更新参数字典。
        {
            "source_run_name": source_run_name,  # 记录来源运行名称。
            "source_factor_column": source_factor_column,  # 记录来源因子字段。
            "source_training_manifest": str(training_path),  # 记录来源训练清单。
            "cne5s_fingerprint": cne5s_fingerprint,  # 记录cne5s文件指纹。
            "eod_fingerprint": eod_fingerprint,  # 记录日行情文件指纹。
            "calendar_fingerprint": calendar_fingerprint,  # 记录交易日历文件指纹。
            "prediction_fingerprints": prediction_fingerprints,  # 记录预测fingerprints。
            "minimum_match_rate": minimum_match_rate,  # 记录minimummatch比例。
            "match_policy": "exact_trade_date_and_assetid_no_forward_fill",  # 把 match_policy 映射为"exact_trade_date_and_assetid_no_forward_fill"。
            "reversal_formula": (  # 记录反转formula。
                "-(s_dq_adjclose(T)/s_dq_adjclose(T-20_SSE_days)-1)"
            ),
            "reversal_missing_policy": "daily_zscore_valid_then_fill_zero",  # 把 reversal_missing_policy 映射为"daily_zscore_valid_then_fill_zero"。
            "style_columns": list(STYLE_COLUMNS),  # 记录风格字段列表。
            "correlation_columns": list(CORRELATION_COLUMNS),  # 记录相关系数字段列表。
            "matrix_columns": list(MATRIX_COLUMNS),  # 记录矩阵字段列表。
            "regression_columns": list(REGRESSION_COLUMNS),  # 记录regression字段列表。
            "excluded_columns": [INDUSTRY_COLUMN, "cne5s_resvol"],  # 记录excluded字段列表。
        }
    )
    return audit  # 返回audit。
