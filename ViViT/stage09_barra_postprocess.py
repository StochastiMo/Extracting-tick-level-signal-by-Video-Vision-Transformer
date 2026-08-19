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

VIVIT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = VIVIT_DIR.parent
DEFAULT_VIVIT_DATA_ROOT = VIVIT_DIR / "data"
DEFAULT_CNE5S_PATH = PROJECT_ROOT / "CNE5S.parquet"
DEFAULT_EOD_PATH = PROJECT_ROOT / "Wind" / "ASHAREEODPRICES.parquet"
DEFAULT_CALENDAR_PATH = PROJECT_ROOT / "Wind" / "ASHARECALENDAR.parquet"
DEFAULT_SOURCE_RUN_NAME = "sigmoid_daybatch_rankic"
DEFAULT_RESIDUAL_RUN_NAME = f"{DEFAULT_SOURCE_RUN_NAME}_barra_residual"
PREDICTION_PATTERN = "vivit_predictions.{year}.parquet"
SOURCE_FACTOR_COLUMN = "prediction_ensemble"
SOURCE_FACTOR_COLUMNS = (
    "prediction_ensemble",
    "prediction_ensemble_unprojected",
    "prediction_ensemble_projected",
    "prediction_ensemble_raw",
    "prediction_ensemble_alpha",
    "prediction_ensemble_style",
    "prediction_ensemble_total_return",
)
INDUSTRY_COLUMN = "cne5s_industry"
UNKNOWN_INDUSTRY = "CNE5S_UNKNOWN"
MINIMUM_MATCH_RATE = 0.999
DERIVATION_SCHEMA_VERSION = 5
HAC_MAX_LAG = 20
ORTHOGONALITY_TOLERANCE = 1e-8
REVERSAL_COLUMN = "reversal_20d"
REVERSAL_VALID_COLUMN = "reversal_20d_valid"
REVERSAL_LOOKBACK_DAYS = 20

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
CORRELATION_COLUMNS = (*STYLE_COLUMNS, REVERSAL_COLUMN)
REGRESSION_COLUMNS = CORRELATION_COLUMNS
VIVIT_MATRIX_COLUMN = "vivit_factor"
MATRIX_COLUMNS = (VIVIT_MATRIX_COLUMN, *CORRELATION_COLUMNS)
MATRIX_LABELS = {
    VIVIT_MATRIX_COLUMN: "ViViT因子",
    **STYLE_LABELS,
}
REQUIRED_CNE5S_COLUMNS = (
    "datadate",
    "assetid",
    INDUSTRY_COLUMN,
    *STYLE_COLUMNS,
)
INSTRUMENT_PATTERN = re.compile(r"^(SH|SZ|BJ)\.(\d{6})$")
CNE5S_ASSET_PATTERN = re.compile(r"^CN\d{6}$")


def _validate_run_name(value: str) -> str:
    """校验输入形状、业务键、有限性和签名约束；异常时明确失败。"""
    run_name = str(value).strip()
    allowed = set(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    )
    if (
        not run_name
        or run_name in {".", ".."}
        or any(character not in allowed for character in run_name)
    ):
        raise ValueError("运行名含非法字符")
    return run_name


def _validate_source_factor_column(value: str) -> str:
    """校验输入形状、业务键、有限性和签名约束；异常时明确失败。"""
    factor_column = str(value).strip()
    if factor_column not in SOURCE_FACTOR_COLUMNS:
        raise ValueError(
            f"未知源预测列：{value!r}；可选 {SOURCE_FACTOR_COLUMNS}"
        )
    return factor_column


def _read_json(path: Path) -> dict[str, Any]:
    """读取对应产物，并按接口约定处理缺失文件。"""
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    """通过临时文件和原子替换写入产物，避免产生半成品。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    """通过临时文件和原子替换写入产物，避免产生半成品。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _payload_signature(payload: dict[str, Any]) -> str:
    """构造参与稳定签名和复用判断的配置载荷。"""
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _training_manifest(
    source_run_name: str, vivit_root: Path
) -> tuple[dict[str, Any], Path]:
    """执行该模式下的训练步骤，并返回损失与诊断信息。"""
    path = (
        vivit_root
        / "quality"
        / "runs"
        / source_run_name
        / "a2_training_manifest.json"
    )
    manifest = _read_json(path)
    years = tuple(int(year) for year in manifest.get("years", ()))
    counts = manifest.get("factor_counts")
    if not years or not isinstance(counts, dict):
        raise ValueError("源训练 manifest 缺少 years 或 factor_counts")
    for year in years:
        if str(year) not in counts and year not in counts:
            raise ValueError(f"源训练 manifest 缺少 {year} factor_counts")
    return manifest, path


def _factor_counts(manifest: dict[str, Any]) -> dict[int, int]:
    """读取或汇总因子评价所需的数据。"""
    raw = manifest["factor_counts"]
    return {
        int(year): int(raw.get(str(year), raw.get(year)))
        for year in manifest["years"]
    }


def _load_source_prediction(
    path: Path,
    year: int,
    expected_rows: int,
    *,
    full: bool,
    source_factor_column: str = SOURCE_FACTOR_COLUMN,
) -> pd.DataFrame:
    """加载该流程所需数据，并检查形状、业务键和来源一致性。"""
    source_factor_column = _validate_source_factor_column(
        source_factor_column
    )
    parquet = pq.ParquetFile(path)  # 读取 Parquet 元数据
    columns = parquet.schema_arrow.names
    # 构造去重集合，并保存到 `required`。
    required = {"trade_date", "instrument_id", source_factor_column}
    missing = required.difference(columns)  # 检查输入缺失字段
    if missing:
        raise ValueError(f"{path.name} 缺少字段：{sorted(missing)}")
    selected = list(columns) if full else list(required)
    prediction = pd.read_parquet(path, columns=selected)
    if len(prediction) != expected_rows:
        raise AssertionError(
            f"{year} 预测行数 {len(prediction):,} != {expected_rows:,}"
        )
    prediction["trade_date"] = prediction["trade_date"].astype(str)
    prediction["instrument_id"] = prediction["instrument_id"].astype(str)
    if prediction.duplicated(["trade_date", "instrument_id"]).any():
        raise ValueError(f"{year} 源预测业务键重复")
    if not prediction["trade_date"].str.startswith(str(year)).all():
        raise ValueError(f"{year} 源预测包含跨年度日期")
    factor = prediction[source_factor_column].to_numpy(np.float64)
    if not np.isfinite(factor).all():
        raise ValueError(f"{year} 源因子含 NaN 或 Inf")
    # 下游统一读取内部标准列；这里只修改内存副本，不写回源预测。
    prediction[SOURCE_FACTOR_COLUMN] = factor
    prediction["assetid"] = instrument_ids_to_assetids(  # 转换为 CNE5S assetid
        prediction["instrument_id"]
    )
    return prediction


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """计算有效样本上的 Pearson 相关系数。"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 3 or len(x) != len(y):
        raise ValueError("Pearson 至少需要 3 个等长观测")
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = float(
        np.sqrt(np.dot(x_centered, x_centered) * np.dot(y_centered, y_centered))
    )
    if not np.isfinite(denominator) or denominator <= np.finfo(np.float64).eps:
        raise ValueError("Pearson 输入为常数或非有限")
    return float(np.dot(x_centered, y_centered) / denominator)


def _complete_correlation_sample(joined: pd.DataFrame) -> pd.DataFrame:
    """返回 11×11 相关矩阵统一使用的完整同日样本。"""

    return joined.loc[
        joined["barra_matched"] & joined[REVERSAL_VALID_COLUMN]
    ]


def _daily_correlations(
    joined: pd.DataFrame, year: int
) -> list[dict[str, Any]]:
    """汇总截面相关性及其时间序列统计。"""
    records: list[dict[str, Any]] = []
    complete = _complete_correlation_sample(joined)  # 计算当前截面相关性
    for trade_date, daily in complete.groupby("trade_date", sort=True):
        y = daily[SOURCE_FACTOR_COLUMN].to_numpy(np.float64)
        for column in CORRELATION_COLUMNS:
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
    return records


def _daily_correlation_matrix(
    joined: pd.DataFrame, year: int
) -> list[dict[str, Any]]:
    """计算 ViViT、九个 CNE5S 风格和反转的日度 11×11 Pearson。"""

    records: list[dict[str, Any]] = []
    complete = _complete_correlation_sample(joined)  # 计算当前截面相关性
    for trade_date, daily in complete.groupby("trade_date", sort=True):
        values = np.column_stack(  # 组成 Barra 设计矩阵
            [
                daily[SOURCE_FACTOR_COLUMN].to_numpy(np.float64),
                *(
                    daily[column].to_numpy(np.float64)
                    for column in CORRELATION_COLUMNS
                ),
            ]
        )
        if len(daily) < 3 or not np.isfinite(values).all():
            raise ValueError(f"{trade_date} 相关矩阵样本不足或含非有限值")
        matrix = np.corrcoef(values, rowvar=False)  # 计算因子间 Pearson 相关性
        expected_shape = (len(MATRIX_COLUMNS), len(MATRIX_COLUMNS))
        if matrix.shape != expected_shape or not np.isfinite(matrix).all():
            raise ValueError(f"{trade_date} 相关矩阵不可计算")
        if not np.allclose(matrix, matrix.T, atol=1e-12, rtol=0.0):
            raise AssertionError(f"{trade_date} 相关矩阵不对称")
        if not np.allclose(
            np.diag(matrix), 1.0, atol=1e-12, rtol=0.0
        ):
            raise AssertionError(f"{trade_date} 相关矩阵对角线不为 1")
        for row_index, row_name in enumerate(MATRIX_COLUMNS):
            for column_index, column_name in enumerate(MATRIX_COLUMNS):
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
    return records

def _neutralize_cross_section(
    daily: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    """以行业固定效应、9 个 CNE5S 风格和 20 日反转残差化。"""

    y = daily[SOURCE_FACTOR_COLUMN].to_numpy(np.float64)
    if not np.isfinite(y).all():
        raise ValueError("单日源因子含 NaN 或 Inf")
    matched = daily["barra_matched"].to_numpy(bool)
    x = np.zeros((len(daily), len(REGRESSION_COLUMNS)), dtype=np.float64)
    style_values = daily.loc[:, STYLE_COLUMNS].to_numpy(
        np.float64, copy=True
    )
    style_values[~matched, :] = 0.0
    x[:, : len(STYLE_COLUMNS)] = style_values
    x[:, -1] = _standardized_reversal(daily)  # 仅使用当日截面统计量

    industries = daily[INDUSTRY_COLUMN].astype("object").copy()
    industries.loc[~daily["barra_matched"]] = UNKNOWN_INDUSTRY
    if industries.isna().any() or not np.isfinite(x).all():
        raise ValueError("中性化输入含未处理的缺失值")

    codes, labels = pd.factorize(industries.astype(str), sort=True)  # 将行业映射为连续代码
    counts = np.bincount(codes).astype(np.float64)
    y_means = np.bincount(codes, weights=y) / counts
    x_sums = np.zeros(
        (len(labels), len(REGRESSION_COLUMNS)), dtype=np.float64
    )
    np.add.at(x_sums, codes, x)  # 按行业汇总截面统计量
    x_means = x_sums / counts[:, None]
    y_within = y - y_means[codes]
    x_within = x - x_means[codes]

    coefficients, _, rank, singular_values = np.linalg.lstsq(  # 求解当日截面 OLS
        x_within, y_within, rcond=None
    )
    expected_rank = len(REGRESSION_COLUMNS)
    if int(rank) != expected_rank:
        raise np.linalg.LinAlgError(
            f"风格暴露矩阵秩为 {rank}，预期 {expected_rank}"
        )
    if (
        len(singular_values) != expected_rank
        or singular_values[-1] <= np.finfo(np.float64).eps
    ):
        raise np.linalg.LinAlgError("风格暴露矩阵条件数不可计算")
    residual = y_within - x_within @ coefficients
    if not np.isfinite(residual).all():
        raise ValueError("回归残差含 NaN 或 Inf")
    source_variance = float(np.dot(y - y.mean(), y - y.mean()))
    if source_variance <= np.finfo(np.float64).eps:
        raise ValueError("单日源因子为常数")
    residual_variance = float(np.dot(residual, residual))
    residual_std = float(residual.std(ddof=0))
    if residual_std <= np.finfo(np.float64).eps:
        raise ValueError("单日回归残差为常数")

    residual_correlations: list[dict[str, Any]] = []
    for column_index, column in enumerate(REGRESSION_COLUMNS):
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
    group_residual_means = np.bincount(
        codes, weights=residual
    ) / counts
    max_style_correlation = max(
        abs(record["pearson"]) for record in residual_correlations
    )
    max_industry_mean = float(np.max(np.abs(group_residual_means)))
    if max_style_correlation > ORTHOGONALITY_TOLERANCE:
        raise AssertionError(
            f"残差风格相关性 {max_style_correlation:.3e} 超过容差"
        )
    if max_industry_mean > ORTHOGONALITY_TOLERANCE:
        raise AssertionError(
            f"行业残差均值 {max_industry_mean:.3e} 超过容差"
        )

    reversal_valid_rows = int(daily[REVERSAL_VALID_COLUMN].sum())
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
    return residual, diagnostics, residual_correlations


def _neutralize_year(
    joined: pd.DataFrame, year: int
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """逐日对年度原始分数回归十变量和行业固定效应，生成统一残差。"""
    residual = np.empty(len(joined), dtype=np.float64)
    diagnostic_records: list[dict[str, Any]] = []
    correlation_records: list[dict[str, Any]] = []
    for trade_date, positions in joined.groupby(
        "trade_date", sort=True
    ).groups.items():
        index = np.asarray(positions, dtype=np.int64)
        values, diagnostics, correlations = _neutralize_cross_section(  # 按日投影 Barra 暴露
            joined.iloc[index]
        )
        residual[index] = values
        diagnostic_records.append(
            {
                "year": year,
                "trade_date": str(trade_date),
                **diagnostics,
            }
        )
        correlation_records.extend(
            {
                "year": year,
                "trade_date": str(trade_date),
                **record,
            }
            for record in correlations
        )
    return (
        pd.DataFrame(diagnostic_records),
        pd.DataFrame(correlation_records),
        residual,
    )


def _newey_west_mean_t_stat(
    values: pd.Series, max_lag: int = HAC_MAX_LAG
) -> float:
    """以 Bartlett 核估计均值的 Newey-West HAC t 值。"""

    sample = np.asarray(values, dtype=np.float64)
    if len(sample) < 2 or not np.isfinite(sample).all():
        return np.nan
    centered = sample - sample.mean()
    lag = min(max_lag, len(sample) - 1)
    long_run_variance = float(np.dot(centered, centered) / len(sample))
    for offset in range(1, lag + 1):
        weight = 1.0 - offset / (lag + 1.0)
        autocovariance = float(
            np.dot(centered[offset:], centered[:-offset]) / len(sample)
        )
        long_run_variance += 2.0 * weight * autocovariance
    if (
        not np.isfinite(long_run_variance)
        or long_run_variance <= np.finfo(np.float64).eps
    ):
        return np.nan
    return float(sample.mean() / np.sqrt(long_run_variance / len(sample)))


def _correlation_summaries(
    daily: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """汇总截面相关性及其时间序列统计。"""
    def summarize(group_columns: list[str]) -> pd.DataFrame:
        """汇总截面相关性及其时间序列统计。"""
        frame = daily.sort_values("trade_date").groupby(
            group_columns, sort=True, observed=True
        )
        result = frame.agg(  # 计算当前截面相关性
            mean_pearson=("pearson", "mean"),
            mean_abs_pearson=("pearson", lambda values: values.abs().mean()),
            std_pearson=("pearson", "std"),
            dates=("trade_date", "nunique"),
            mean_observations=("observations", "mean"),
        ).reset_index()
        result["t_stat"] = (
            result["mean_pearson"]
            / result["std_pearson"]
            * np.sqrt(result["dates"])
        )
        corrected = (  # 计算当前截面相关性
            frame["pearson"]
            .agg(_newey_west_mean_t_stat)
            .rename("t_stat_newey_west_lag20")
            .reset_index()
        )
        return result.merge(
            corrected,
            on=group_columns,
            how="left",
            validate="one_to_one",
        )

    annual = summarize(  # 按约定截面口径汇总
        ["year", "factor_name", "factor_label"]
    )
    overall = summarize(  # 按约定截面口径汇总
        ["factor_name", "factor_label"]
    )
    order = {
        column: index for index, column in enumerate(CORRELATION_COLUMNS)
    }
    annual["_order"] = annual["factor_name"].map(order)
    overall["_order"] = overall["factor_name"].map(order)
    annual.sort_values(["year", "_order"], inplace=True)
    overall.sort_values("_order", inplace=True)
    return (
        annual.drop(columns="_order").reset_index(drop=True),
        overall.drop(columns="_order").reset_index(drop=True),
    )


def _correlation_matrix_summary(daily: pd.DataFrame) -> pd.DataFrame:
    """汇总截面相关性及其时间序列统计。"""
    expected_cells = len(MATRIX_COLUMNS) ** 2
    cells_per_date = daily.groupby("trade_date", sort=False).size()  # 统计每日截面样本数
    if daily.empty or not cells_per_date.eq(expected_cells).all():
        raise AssertionError("每个交易日必须包含完整的 11×11 相关矩阵")
    if daily.duplicated(
        ["trade_date", "row_name", "column_name"]
    ).any():
        raise AssertionError("日度相关矩阵存在重复单元")

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
    order = {name: index for index, name in enumerate(MATRIX_COLUMNS)}
    summary["_row_order"] = summary["row_name"].map(order)  # 按约定截面口径汇总
    summary["_column_order"] = summary["column_name"].map(order)  # 按约定截面口径汇总
    summary.sort_values(  # 按约定截面口径汇总
        ["_row_order", "_column_order"], inplace=True
    )
    summary.drop(  # 按约定截面口径汇总
        columns=["_row_order", "_column_order"], inplace=True
    )
    summary.reset_index(drop=True, inplace=True)  # 按约定截面口径汇总
    if len(summary) != expected_cells:
        raise AssertionError("相关矩阵汇总不含完整的 121 个单元")
    matrix = (  # 按约定截面口径汇总
        summary.pivot(
            index="row_name", columns="column_name", values="mean_pearson"
        )
        .loc[list(MATRIX_COLUMNS), list(MATRIX_COLUMNS)]
        .to_numpy(np.float64)
    )
    if not np.allclose(matrix, matrix.T, atol=1e-12, rtol=0.0):
        raise AssertionError("日均相关矩阵不对称")
    if not np.allclose(np.diag(matrix), 1.0, atol=1e-12, rtol=0.0):
        raise AssertionError("日均相关矩阵对角线不为 1")
    return summary


def analyze_barra_exposures(
    source_run_name: str = DEFAULT_SOURCE_RUN_NAME,
    cne5s_path: str | Path = DEFAULT_CNE5S_PATH,
    vivit_data_root: str | Path = DEFAULT_VIVIT_DATA_ROOT,
    minimum_match_rate: float = MINIMUM_MATCH_RATE,
    eod_path: str | Path = DEFAULT_EOD_PATH,
    calendar_path: str | Path = DEFAULT_CALENDAR_PATH,
    source_factor_column: str = SOURCE_FACTOR_COLUMN,
) -> dict[str, Any]:
    """只读计算源因子的 Barra/反转覆盖、相关性与 11×11 矩阵。"""

    source_run_name = _validate_run_name(source_run_name)
    source_factor_column = _validate_source_factor_column(
        source_factor_column
    )
    if not 0 <= minimum_match_rate <= 1:
        raise ValueError("minimum_match_rate 必须位于 [0,1]")
    vivit_root = Path(vivit_data_root).resolve()
    cne5s = Path(cne5s_path).resolve()
    eod = Path(eod_path).resolve()
    calendar_file = Path(calendar_path).resolve()
    if not vivit_root.is_dir():
        raise FileNotFoundError(vivit_root)
    _validate_cne5s_schema(cne5s)
    _validate_price_sources(eod, calendar_file)
    calendar_dates = _load_sse_calendar(calendar_file)
    training, _ = _training_manifest(source_run_name, vivit_root)  # 读取或生成来源清单
    counts = _factor_counts(training)
    prediction_root = vivit_root / "predictions" / "runs" / source_run_name
    match_records: list[dict[str, Any]] = []
    correlation_records: list[dict[str, Any]] = []
    matrix_correlation_records: list[dict[str, Any]] = []

    for year, expected_rows in counts.items():
        path = prediction_root / PREDICTION_PATTERN.format(year=year)
        if expected_rows == 0:
            if path.exists():
                raise AssertionError(
                    f"{year} factor_counts=0 但存在预测文件：{path}"
                )
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
            continue
        prediction = _load_source_prediction(
            path,
            year,
            expected_rows,
            full=True,
            source_factor_column=source_factor_column,
        )
        prediction = _attach_reversal_20d(  # 按股票与日期业务键补充字段
            prediction, eod, calendar_dates, year
        )
        barra = _load_cne5s_year(cne5s, year)
        joined = _join_prediction_and_cne5s(prediction, barra, year)
        record = _match_record(joined, year, len(barra))  # 核对两侧业务键并统计匹配率
        _check_match_rate(record, minimum_match_rate)
        match_records.append(record)
        correlation_records.extend(_daily_correlations(joined, year))
        matrix_correlation_records.extend(
            _daily_correlation_matrix(joined, year)
        )

    input_audit = pd.DataFrame(match_records)
    reversal_price_coverage = input_audit.loc[:, [
        "year", "prediction_rows", "reversal_valid_rows",
        "reversal_missing_rows", "reversal_coverage",
        "reversal_correlation_rows", "dates", "first_date",
        "last_date", "status",
    ]].copy()
    daily_correlations = pd.DataFrame(correlation_records)
    if daily_correlations.empty:
        raise ValueError("源 run 没有可用于相关性分析的非空预测")
    if set(daily_correlations["factor_name"].unique()) != set(
        CORRELATION_COLUMNS
    ):
        raise AssertionError("日度相关性未恰好覆盖十个测试变量")
    annual_correlations, correlation_summary = _correlation_summaries(  # 计算当前截面相关性
        daily_correlations
    )
    daily_correlation_matrix = pd.DataFrame(matrix_correlation_records)
    correlation_matrix_summary = _correlation_matrix_summary(  # 按约定截面口径汇总
        daily_correlation_matrix
    )
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


def _save_heatmap(summary: pd.DataFrame, path: Path) -> None:
    """生成并保存该评价环节对应的图表，缺失指标按既定口径处理。"""
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    import seaborn as sns

    cjk_font_path = Path.home() / ".local/share/fonts/NotoSansCJKsc-Regular.otf"
    if cjk_font_path.exists():
        font_manager.fontManager.addfont(str(cjk_font_path))

    values = summary.pivot(
        index="row_name", columns="column_name", values="mean_pearson"
    ).loc[list(MATRIX_COLUMNS), list(MATRIX_COLUMNS)]
    labels = [MATRIX_LABELS[column] for column in MATRIX_COLUMNS]
    matrix = pd.DataFrame(values.to_numpy(np.float64), index=labels, columns=labels)
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    figure, axis = plt.subplots(figsize=(12.5, 10.5))
    sns.heatmap(  # 绘制因子相关性热力图
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
    axis.set_title(
        "ViViT 因子、9 个 CNE5S 风格与 20 日反转的日均横截面 Pearson",
        fontsize=13,
        pad=14,
    )
    axis.set_xlabel("")
    axis.set_ylabel("")
    axis.tick_params(axis="x", rotation=45)
    axis.tick_params(axis="y", rotation=0)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)

def _save_reversal_timeseries(
    daily_correlations: pd.DataFrame, path: Path
) -> None:
    """生成并保存该评价环节对应的图表，缺失指标按既定口径处理。"""
    from matplotlib import font_manager
    import matplotlib.pyplot as plt

    reversal = daily_correlations.loc[  # 计算当前截面相关性
        daily_correlations["factor_name"].eq(REVERSAL_COLUMN)
    ].copy()
    if reversal.empty:
        raise ValueError("缺少 20 日反转日度相关性")
    reversal["trade_date"] = pd.to_datetime(
        reversal["trade_date"], format="%Y%m%d", errors="raise"
    )
    reversal.sort_values("trade_date", inplace=True)
    if reversal["trade_date"].duplicated().any():
        raise ValueError("20 日反转日度相关性日期重复")

    cjk_font_path = Path.home() / ".local/share/fonts/NotoSansCJKsc-Regular.otf"
    if cjk_font_path.exists():
        font_manager.fontManager.addfont(str(cjk_font_path))
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    figure, axis = plt.subplots(figsize=(14, 4.5))
    axis.plot(
        reversal["trade_date"], reversal["pearson"],
        color="#7c3aed", linewidth=0.8, alpha=0.9,
        label="20日反转",
    )
    axis.axhline(0.0, color="#111827", linewidth=0.8, linestyle="--")
    axis.set_title("ViViT 因子与 20 日反转的日度横截面 Pearson")
    axis.set_xlabel("交易日期")
    axis.set_ylabel("横截面 Pearson")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


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
    """构造参与稳定签名和复用判断的配置载荷。"""
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


def _existing_result(
    quality_root: Path,
    prediction_root: Path,
    expected_signature: str,
    counts: dict[int, int],
    *,
    overwrite_existing: bool = False,
) -> dict[str, Any] | None:
    """检查同签名派生 run 是否完整可复用，并返回其现有结果。"""
    neutralization_root = quality_root / "barra_neutralization"
    manifest_path = neutralization_root / "neutralization_manifest.json"
    if not quality_root.exists() and not prediction_root.exists():
        return None
    if not manifest_path.exists():
        if overwrite_existing:
            return None
        raise RuntimeError(
            "派生 run 已存在但缺少 neutralization manifest；"
            "如需替换必须显式设置 overwrite_existing=True"
        )
    manifest = _read_json(manifest_path)
    if manifest.get("signature") != expected_signature:
        if overwrite_existing:
            return None
        raise RuntimeError(
            "已有派生 run 与当前源数据或配置签名不一致；"
            "如需替换必须显式设置 overwrite_existing=True"
        )
    for year, count in counts.items():
        path = prediction_root / PREDICTION_PATTERN.format(year=year)
        if count:
            if not path.exists():
                raise FileNotFoundError(path)
            if pq.ParquetFile(path).metadata.num_rows != count:
                raise AssertionError(f"{path.name} 行数与 manifest 不一致")
        elif path.exists():
            raise AssertionError(f"{year} 应为空但存在派生预测文件")
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
    result: dict[str, Any] = {
        name: pd.read_parquet(neutralization_root / file_name)
        for name, file_name in required.items()
    }
    heatmap = neutralization_root / "correlation_heatmap.png"
    reversal_timeseries = (
        neutralization_root / "reversal_20d_correlation_timeseries.png"
    )
    for path in (heatmap, reversal_timeseries):
        if not path.exists():
            raise FileNotFoundError(path)
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
    return result


def _replace_staged_roots(
    stage_predictions: Path,
    stage_quality: Path,
    prediction_root: Path,
    quality_root: Path,
) -> bool:
    """以临时 tombstone 支持失败回滚；成功后不保留旧 run。"""

    token = f"{os.getpid()}.{time.time_ns()}"
    old_prediction = prediction_root.with_name(
        f".{prediction_root.name}.replacing.{token}"
    )
    old_quality = quality_root.with_name(
        f".{quality_root.name}.replacing.{token}"
    )
    had_existing = prediction_root.exists() or quality_root.exists()
    prediction_installed = False
    quality_installed = False
    try:
        if prediction_root.exists():
            prediction_root.replace(old_prediction)
        if quality_root.exists():
            quality_root.replace(old_quality)
        stage_predictions.replace(prediction_root)
        prediction_installed = True
        stage_quality.replace(quality_root)
        quality_installed = True
    except Exception:
        if quality_installed and quality_root.exists():
            quality_root.replace(stage_quality)
        if prediction_installed and prediction_root.exists():
            prediction_root.replace(stage_predictions)
        if old_quality.exists():
            old_quality.replace(quality_root)
        if old_prediction.exists():
            old_prediction.replace(prediction_root)
        raise
    else:
        if old_quality.exists():
            shutil.rmtree(old_quality)
        if old_prediction.exists():
            shutil.rmtree(old_prediction)
    return had_existing


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
    """生成相关性产物和可直接交给 Notebook 03 的 Barra 残差派生 run。"""

    source_run_name = _validate_run_name(source_run_name)
    residual_run_name = _validate_run_name(residual_run_name)
    source_factor_column = _validate_source_factor_column(
        source_factor_column
    )
    if residual_run_name == source_run_name:
        raise ValueError("派生 run 不能与源 run 同名")
    if not 0 <= minimum_match_rate <= 1:
        raise ValueError("minimum_match_rate 必须位于 [0,1]")
    if not isinstance(overwrite_existing, bool):
        raise TypeError("overwrite_existing 必须是 bool")
    vivit_root = Path(vivit_data_root).resolve()
    cne5s = Path(cne5s_path).resolve()
    eod = Path(eod_path).resolve()
    calendar_file = Path(calendar_path).resolve()
    cne5s_fingerprint = _validate_cne5s_schema(cne5s)
    eod_fingerprint, calendar_fingerprint = _validate_price_sources(
        eod, calendar_file
    )
    calendar_dates = _load_sse_calendar(calendar_file)
    training, training_path = _training_manifest(source_run_name, vivit_root)  # 读取或生成来源清单
    counts = _factor_counts(training)
    source_prediction_root = (
        vivit_root / "predictions" / "runs" / source_run_name
    )
    prediction_fingerprints = [
        _fingerprint_parquet(
            source_prediction_root / PREDICTION_PATTERN.format(year=year)
        )
        for year, count in counts.items()
        if count
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
    prediction_root = (
        vivit_root / "predictions" / "runs" / residual_run_name
    )
    quality_root = vivit_root / "quality" / "runs" / residual_run_name
    existing = _existing_result(  # 仅复用签名一致的产物
        quality_root, prediction_root, signature, counts,
        overwrite_existing=overwrite_existing,
    )
    if existing is not None:
        return existing

    staging_parent = Path(
        tempfile.mkdtemp(
            prefix=f".{residual_run_name}.",
            dir=vivit_root,
        )
    )
    stage_predictions = staging_parent / "predictions"
    stage_quality = staging_parent / "quality"
    stage_neutralization = stage_quality / "barra_neutralization"
    stage_predictions.mkdir(parents=True)
    stage_neutralization.mkdir(parents=True)

    match_records: list[dict[str, Any]] = []
    correlation_records: list[dict[str, Any]] = []
    matrix_correlation_records: list[dict[str, Any]] = []
    diagnostic_frames: list[pd.DataFrame] = []
    residual_correlation_frames: list[pd.DataFrame] = []
    output_counts: dict[str, int] = {}
    try:
        for year, expected_rows in counts.items():
            if expected_rows == 0:
                output_counts[str(year)] = 0
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
                continue
            source_path = (
                source_prediction_root
                / PREDICTION_PATTERN.format(year=year)
            )
            prediction = _load_source_prediction(
                source_path,
                year,
                expected_rows,
                full=True,
                source_factor_column=source_factor_column,
            )
            prediction = _attach_reversal_20d(  # 按股票与日期业务键补充字段
                prediction, eod, calendar_dates, year
            )
            barra = _load_cne5s_year(cne5s, year)
            joined = _join_prediction_and_cne5s(
                prediction, barra, year
            )
            match_record = _match_record(joined, year, len(barra))  # 核对两侧业务键并统计匹配率
            _check_match_rate(match_record, minimum_match_rate)
            match_records.append(match_record)
            correlation_records.extend(
                _daily_correlations(joined, year)
            )
            matrix_correlation_records.extend(
                _daily_correlation_matrix(joined, year)
            )
            diagnostics, residual_correlations, residual = (  # 按日投影 Barra 暴露
                _neutralize_year(joined, year)
            )
            diagnostic_frames.append(diagnostics)
            residual_correlation_frames.append(residual_correlations)

            output_columns: dict[str, Any] = {}
            if "sample_index" in prediction.columns:
                output_columns["sample_index"] = prediction[
                    "sample_index"
                ].to_numpy()
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
            output = pd.DataFrame(output_columns)
            if len(output) != expected_rows:
                raise AssertionError(f"{year} 派生预测行数改变")
            if not np.array_equal(
                output["trade_date"].to_numpy(),
                prediction["trade_date"].to_numpy(),
            ) or not np.array_equal(
                output["instrument_id"].to_numpy(),
                prediction["instrument_id"].to_numpy(),
            ):
                raise AssertionError(f"{year} 派生预测业务键顺序改变")
            if not np.array_equal(
                output["prediction_ensemble_raw"].to_numpy(),
                output["prediction_ensemble"].to_numpy(),
            ):
                raise AssertionError("原始残差不应被再次标准化")
            _atomic_parquet(  # 原子落盘，避免半成品
                output,
                stage_predictions
                / PREDICTION_PATTERN.format(year=year),
            )
            output_counts[str(year)] = len(output)

        input_audit = pd.DataFrame(match_records)
        reversal_price_coverage = input_audit.loc[:, [
            "year", "prediction_rows", "reversal_valid_rows",
            "reversal_missing_rows", "reversal_coverage",
            "reversal_correlation_rows", "dates", "first_date",
            "last_date", "status",
        ]].copy()
        daily_correlations = pd.DataFrame(correlation_records)
        if set(daily_correlations["factor_name"].unique()) != set(
            CORRELATION_COLUMNS
        ):
            raise AssertionError("日度相关性未恰好覆盖十个测试变量")
        annual_correlations, correlation_summary = (  # 计算当前截面相关性
            _correlation_summaries(daily_correlations)
        )
        daily_correlation_matrix = pd.DataFrame(
            matrix_correlation_records
        )
        correlation_matrix_summary = _correlation_matrix_summary(  # 按约定截面口径汇总
            daily_correlation_matrix
        )
        regression_diagnostics = pd.concat(
            diagnostic_frames, ignore_index=True
        )
        residual_correlations = pd.concat(
            residual_correlation_frames, ignore_index=True
        )
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
        for file_name, frame in artifacts.items():
            _atomic_parquet(frame, stage_neutralization / file_name)  # 原子落盘，避免半成品
        heatmap_path = stage_neutralization / "correlation_heatmap.png"
        reversal_timeseries_path = (
            stage_neutralization
            / "reversal_20d_correlation_timeseries.png"
        )
        _save_heatmap(correlation_matrix_summary, heatmap_path)
        _save_reversal_timeseries(
            daily_correlations, reversal_timeseries_path
        )

        derived_training = dict(training)
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
        _atomic_json(  # 原子落盘，避免半成品
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
        if source_summary.exists():
            shutil.copy2(
                source_summary,
                stage_quality / "vivit_training_summary.parquet",
            )
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
        _atomic_json(  # 原子落盘，避免半成品
            neutralization_manifest,
            stage_neutralization / "neutralization_manifest.json",
        )

        _existing_result(  # 仅复用签名一致的产物
            stage_quality, stage_predictions, signature, counts
        )
        prediction_root.parent.mkdir(parents=True, exist_ok=True)
        quality_root.parent.mkdir(parents=True, exist_ok=True)
        overwritten = _replace_staged_roots(
            stage_predictions, stage_quality, prediction_root, quality_root
        )
    finally:
        shutil.rmtree(staging_parent, ignore_errors=True)

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
def audit_barra_inputs(
    source_run_name: str = DEFAULT_SOURCE_RUN_NAME,
    cne5s_path: str | Path = DEFAULT_CNE5S_PATH,
    vivit_data_root: str | Path = DEFAULT_VIVIT_DATA_ROOT,
    minimum_match_rate: float = MINIMUM_MATCH_RATE,
    eod_path: str | Path = DEFAULT_EOD_PATH,
    calendar_path: str | Path = DEFAULT_CALENDAR_PATH,
    source_factor_column: str = SOURCE_FACTOR_COLUMN,
) -> pd.DataFrame:
    """只读审计源预测、CNE5S schema、精确同日业务键和年度覆盖率。"""

    source_run_name = _validate_run_name(source_run_name)
    source_factor_column = _validate_source_factor_column(
        source_factor_column
    )
    if not 0 <= minimum_match_rate <= 1:
        raise ValueError("minimum_match_rate 必须位于 [0,1]")
    vivit_root = Path(vivit_data_root).resolve()
    cne5s = Path(cne5s_path).resolve()
    eod = Path(eod_path).resolve()
    calendar_file = Path(calendar_path).resolve()
    if not vivit_root.is_dir():
        raise FileNotFoundError(vivit_root)
    cne5s_fingerprint = _validate_cne5s_schema(cne5s)
    eod_fingerprint, calendar_fingerprint = _validate_price_sources(
        eod, calendar_file
    )
    calendar_dates = _load_sse_calendar(calendar_file)
    training, training_path = _training_manifest(source_run_name, vivit_root)  # 读取或生成来源清单
    counts = _factor_counts(training)
    prediction_root = vivit_root / "predictions" / "runs" / source_run_name
    records: list[dict[str, Any]] = []
    prediction_fingerprints: list[dict[str, Any]] = []

    for year, expected_rows in counts.items():
        path = prediction_root / PREDICTION_PATTERN.format(year=year)
        if expected_rows == 0:
            if path.exists():
                raise AssertionError(
                    f"{year} factor_counts=0 但存在预测文件：{path}"
                )
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
            continue
        prediction_fingerprints.append(_fingerprint_parquet(path))
        prediction = _load_source_prediction(
            path,
            year,
            expected_rows,
            full=False,
            source_factor_column=source_factor_column,
        )
        prediction = _attach_reversal_20d(  # 按股票与日期业务键补充字段
            prediction, eod, calendar_dates, year
        )
        barra = _load_cne5s_year(cne5s, year)
        joined = _join_prediction_and_cne5s(prediction, barra, year)
        record = _match_record(joined, year, len(barra))  # 核对两侧业务键并统计匹配率
        _check_match_rate(record, minimum_match_rate)
        records.append(record)

    audit = pd.DataFrame(records)
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
    return audit
