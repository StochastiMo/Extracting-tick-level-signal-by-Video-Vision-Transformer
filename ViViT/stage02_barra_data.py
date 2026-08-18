"""为 ViViT 风格中性训练构建逐样本十风格暴露和残差标签。"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import stage02_market_data as barra
import stage01_sequence_data as base


# 读取 `Path(__file__).resolve().parent` 的当前值，并赋给 `VIVIT_DIR`。
VIVIT_DIR = Path(__file__).resolve().parent
# 计算 `VIVIT_DIR / 'data'`，并将数值结果保存到 `DATA_DIR`。
DATA_DIR = VIVIT_DIR / "data"
# 计算 `DATA_DIR / 'style_neutral_store'`，并将数值结果保存到 `DEFAULT_STORE_DIR`。
DEFAULT_STORE_DIR = DATA_DIR / "style_neutral_store"
# 设置 `STYLE_EXPOSURES_FILE` 的固定配置值，供后续流程统一使用。
STYLE_EXPOSURES_FILE = "style_exposures.npy"
# 设置 `STYLE_VALID_FILE` 的固定配置值，供后续流程统一使用。
STYLE_VALID_FILE = "style_valid.npy"
# 设置 `RESIDUAL_TARGETS_FILE` 的固定配置值，供后续流程统一使用。
RESIDUAL_TARGETS_FILE = "targets_style_residual.npy"
# 设置 `DAILY_DIAGNOSTICS_FILE` 的固定配置值，供后续流程统一使用。
DAILY_DIAGNOSTICS_FILE = "daily_diagnostics.parquet"
# 设置 `ANNUAL_SUMMARY_FILE` 的固定配置值，供后续流程统一使用。
ANNUAL_SUMMARY_FILE = "annual_summary.parquet"
# 设置 `MANIFEST_FILE` 的固定配置值，供后续流程统一使用。
MANIFEST_FILE = "style_neutral_manifest.json"
# 计算 `DEFAULT_STORE_DIR / 'industry_sidecar'`，并将数值结果保存到 `DEFAULT_INDUSTRY_STORE_DIR`。
DEFAULT_INDUSTRY_STORE_DIR = DEFAULT_STORE_DIR / "industry_sidecar"
# 设置 `INDUSTRY_CODES_FILE` 的固定配置值，供后续流程统一使用。
INDUSTRY_CODES_FILE = "industry_codes.npy"
# 设置 `INDUSTRY_AUDIT_FILE` 的固定配置值，供后续流程统一使用。
INDUSTRY_AUDIT_FILE = "industry_match_audit.parquet"
# 设置 `INDUSTRY_MANIFEST_FILE` 的固定配置值，供后续流程统一使用。
INDUSTRY_MANIFEST_FILE = "industry_sidecar_manifest.json"
# 计算 `DEFAULT_STORE_DIR / 'full_barra_target_sidecar'`，并将数值结果保存到 `DEFAULT_FULL_BARRA_TARGET_STORE_DIR`。
DEFAULT_FULL_BARRA_TARGET_STORE_DIR = (
    DEFAULT_STORE_DIR / "full_barra_target_sidecar"
)
# 设置 `FULL_BARRA_TARGETS_FILE` 的固定配置值，供后续流程统一使用。
FULL_BARRA_TARGETS_FILE = "targets_full_barra_residual.npy"
# 设置 `FULL_BARRA_DAILY_DIAGNOSTICS_FILE` 的固定配置值，供后续流程统一使用。
FULL_BARRA_DAILY_DIAGNOSTICS_FILE = "daily_diagnostics.parquet"
# 设置 `FULL_BARRA_ANNUAL_SUMMARY_FILE` 的固定配置值，供后续流程统一使用。
FULL_BARRA_ANNUAL_SUMMARY_FILE = "annual_summary.parquet"
# 设置 `FULL_BARRA_TARGET_MANIFEST_FILE` 的固定配置值，供后续流程统一使用。
FULL_BARRA_TARGET_MANIFEST_FILE = "full_barra_target_manifest.json"
# 计算 `DEFAULT_STORE_DIR / 'dual_tower_target_sidecar'`，并将数值结果保存到 `DEFAULT_DUAL_TOWER_TARGET_STORE_DIR`。
DEFAULT_DUAL_TOWER_TARGET_STORE_DIR = (
    DEFAULT_STORE_DIR / "dual_tower_target_sidecar"
)
# 设置 `DUAL_TOWER_FITTED_TARGETS_FILE` 的固定配置值，供后续流程统一使用。
DUAL_TOWER_FITTED_TARGETS_FILE = "targets_full_barra_fitted.npy"
# 设置 `DUAL_TOWER_DAILY_DIAGNOSTICS_FILE` 的固定配置值，供后续流程统一使用。
DUAL_TOWER_DAILY_DIAGNOSTICS_FILE = "daily_diagnostics.parquet"
# 设置 `DUAL_TOWER_ANNUAL_SUMMARY_FILE` 的固定配置值，供后续流程统一使用。
DUAL_TOWER_ANNUAL_SUMMARY_FILE = "annual_summary.parquet"
# 设置 `DUAL_TOWER_TARGET_MANIFEST_FILE` 的固定配置值，供后续流程统一使用。
DUAL_TOWER_TARGET_MANIFEST_FILE = "dual_tower_target_manifest.json"
# 组织一组有序值，并保存到 `STYLE_COLUMNS`。
STYLE_COLUMNS = (*barra.STYLE_COLUMNS, barra.REVERSAL_COLUMN)
# 设置 `SCHEMA_VERSION` 的固定配置值，供后续流程统一使用。
SCHEMA_VERSION = 1
# 设置 `INDUSTRY_SIDECAR_SCHEMA_VERSION` 的固定配置值，供后续流程统一使用。
INDUSTRY_SIDECAR_SCHEMA_VERSION = 1
# 设置 `FULL_BARRA_TARGET_SCHEMA_VERSION` 的固定配置值，供后续流程统一使用。
FULL_BARRA_TARGET_SCHEMA_VERSION = 1
# 设置 `DUAL_TOWER_TARGET_SCHEMA_VERSION` 的固定配置值，供后续流程统一使用。
DUAL_TOWER_TARGET_SCHEMA_VERSION = 1
# 设置 `EXPECTED_CNE5S_INDUSTRIES` 的固定配置值，供后续流程统一使用。
EXPECTED_CNE5S_INDUSTRIES = 32
# 设置 `MINIMUM_MATCH_RATE` 的固定配置值，供后续流程统一使用。
MINIMUM_MATCH_RATE = 0.999
# 设置 `ORTHOGONALITY_TOLERANCE` 的固定配置值，供后续流程统一使用。
ORTHOGONALITY_TOLERANCE = 1e-8


# 定义内部辅助函数 `_file_fingerprint`，封装仅供本模块复用的计算步骤。
def _file_fingerprint(path: Path) -> dict[str, Any]:
    """提取文件身份信息，用于来源审计和复用判断。
    
    输入：
        path: Path。
    输出：
        dict[str, Any]。
        """
    # 检查 `not path.exists()`；根据条件结果选择对应处理分支。
    if not path.exists():
        # 抛出 `FileNotFoundError(path)`，阻止不满足数据契约的流程继续执行。
        raise FileNotFoundError(path)
    # 将 `{'path': str(path.resolve()), 'size': int(path.stat().st_size), 'mtime_ns': int(path.stat().st_…` 作为当前函数结果返回给调用方。
    return {
        "path": str(path.resolve()),
        "size": int(path.stat().st_size),
        "mtime_ns": int(path.stat().st_mtime_ns),
    }


# 定义内部辅助函数 `_stable_hash`，封装仅供本模块复用的计算步骤。
def _stable_hash(payload: object) -> str:
    """对可 JSON 序列化载荷计算稳定的 SHA-256 签名。
    
    输入：
        payload: object。
    输出：
        str。
        """
    # 调用 `json.dumps(payload, ensure_ascii=False, sort_keys=True, separat…` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `encoded`。
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    # 将 `hashlib.sha256(encoded).hexdigest()` 作为当前函数结果返回给调用方。
    return hashlib.sha256(encoded).hexdigest()


# 定义内部辅助函数 `_store_payload`，封装仅供本模块复用的计算步骤。
def _store_payload(
    sample_index_path: Path,
    targets_path: Path,
    cne5s_path: Path,
    eod_path: Path,
    calendar_path: Path,
) -> dict[str, Any]:
    """构造参与稳定签名和复用判断的配置载荷。
    
    输入：
        sample_index_path: Path。
        targets_path: Path。
        cne5s_path: Path。
        eod_path: Path。
        calendar_path: Path。
    输出：
        dict[str, Any]。
        """
    # 调用 `json.loads` 读取源数据，并将结果保存到 `dataset`。
    dataset = json.loads(
        base.SOURCE_DATASET_MANIFEST_PATH.read_text(encoding="utf-8")
    )
    # 将 `{'schema_version': SCHEMA_VERSION, 'source_dataset_signature': dataset['signature'], 'sample_in…` 作为当前函数结果返回给调用方。
    return {
        "schema_version": SCHEMA_VERSION,
        "source_dataset_signature": dataset["signature"],
        "sample_index": _file_fingerprint(sample_index_path),
        "source_targets": _file_fingerprint(targets_path),
        "cne5s": barra._fingerprint_parquet(cne5s_path),
        "wind_eod_prices": barra._fingerprint_parquet(eod_path),
        "sse_calendar": barra._fingerprint_parquet(calendar_path),
        "style_columns": list(STYLE_COLUMNS),
        "style_count": len(STYLE_COLUMNS),
        "industry_used": False,
        "residualization": (
            "daily_equal_weight_ols_intercept_plus_9_cne5s_styles_plus_"
            "reversal20_then_residual_zscore_ddof0"
        ),
        "missing_policy": (
            "daily_standardize_valid_exposures_then_fill_zero_no_forward_fill"
        ),
        "reversal_formula": (
            "-(s_dq_adjclose(T)/s_dq_adjclose(T-20_SSE_days)-1)"
        ),
        "match_policy": "exact_trade_date_and_assetid_no_forward_fill",
    }


# 定义内部辅助函数 `_standardize_with_missing`，封装仅供本模块复用的计算步骤。
def _standardize_with_missing(
    values: np.ndarray,
    valid: np.ndarray,
    *,
    label: str,
) -> np.ndarray:
    """有效值按日标准化，缺失值填 0；不做任何跨日填充。
    
    输入：
        values: np.ndarray。
        valid: np.ndarray。
        label: str。
    输出：
        np.ndarray。
        """

    # 调用 `np.asarray` 转换数据表示，并将标准化对象保存到 `raw`。
    raw = np.asarray(values, dtype=np.float64)
    # 调用 `np.asarray` 转换数据表示，并将标准化对象保存到 `mask`。
    mask = np.asarray(valid, dtype=bool)
    # 检查 `raw.ndim != 1 or mask.shape != raw.shape`；根据条件结果选择对应处理分支。
    if raw.ndim != 1 or mask.shape != raw.shape:
        # 抛出 `ValueError(f'{label} 数值和有效标记形状不一致')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(f"{label} 数值和有效标记形状不一致")
    # 检查 `int(mask.sum()) < 3`；根据条件结果选择对应处理分支。
    if int(mask.sum()) < 3:
        # 抛出 `ValueError(f'{label} 单日有效样本不足 3 个')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(f"{label} 单日有效样本不足 3 个")
    # 读取 `raw[mask]` 的当前值，并赋给 `selected`。
    selected = raw[mask]
    # 检查 `not np.isfinite(selected).all()`；根据条件结果选择对应处理分支。
    if not np.isfinite(selected).all():
        # 抛出 `ValueError(f'{label} 有效暴露含 NaN 或 Inf')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(f"{label} 有效暴露含 NaN 或 Inf")
    # 调用 `float` 统一设备或数据类型，并将结果保存到 `mean`。
    mean = float(selected.mean())
    # 调用 `float` 统一设备或数据类型，并将结果保存到 `std`。
    std = float(selected.std(ddof=0))
    # 检查 `not np.isfinite(std) or std <= np.finfo(np.float64).eps`；根据条件结果选择对应处理分支。
    if not np.isfinite(std) or std <= np.finfo(np.float64).eps:
        # 抛出 `ValueError(f'{label} 单日暴露为常数')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(f"{label} 单日暴露为常数")
    # 调用 `np.zeros` 按目标形状初始化数值容器 `result`。
    result = np.zeros(len(raw), dtype=np.float64)
    # 计算 `(selected - mean) / std`，并将数值结果保存到 `result[mask]`。
    result[mask] = (selected - mean) / std
    # 将 `result` 作为当前函数结果返回给调用方。
    return result


# 定义 `residualize_daily_targets` 函数，封装这一阶段可独立调用的处理步骤。
def residualize_daily_targets(
    targets: np.ndarray,
    style_exposures: np.ndarray,
    label_valid: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """对单日标签联合回归十风格，返回保持单位尺度的 OLS 残差。
    
    输入：
        targets: np.ndarray，格式 [N]。
        style_exposures: np.ndarray，格式 [N,10]。
        label_valid: np.ndarray。
    输出：
        tuple[np.ndarray, dict[str, float | int]]。
        """

    # 调用 `np.asarray` 转换数据表示，并将标准化对象保存到 `y_all`。
    y_all = np.asarray(targets, dtype=np.float64)
    # 调用 `np.asarray` 转换数据表示，并将标准化对象保存到 `styles`。
    styles = np.asarray(style_exposures, dtype=np.float64)
    # 调用 `np.asarray` 转换数据表示，并将标准化对象保存到 `valid`。
    valid = np.asarray(label_valid, dtype=bool)
    # 检查 `styles.shape != (len(y_all), len(STYLE_COLUMNS))`；根据条件结果选择对应处理分支。
    if styles.shape != (len(y_all), len(STYLE_COLUMNS)):
        # 抛出 `ValueError('单日风格矩阵形状异常')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("单日风格矩阵形状异常")
    # 检查 `valid.shape != y_all.shape or int(valid.sum()) < len(STYLE_COLUMNS) + 2`；根据条件结果选择对应处理分支。
    if valid.shape != y_all.shape or int(valid.sum()) < len(STYLE_COLUMNS) + 2:
        # 抛出 `ValueError('单日有效标签不足以进行十风格回归')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("单日有效标签不足以进行十风格回归")
    # 检查 `not np.isfinite(y_all[valid]).all() or not np.isfinite(styles).all()`；根据条件结果选择对应处理分支。
    if not np.isfinite(y_all[valid]).all() or not np.isfinite(styles).all():
        # 抛出 `ValueError('单日标签或风格矩阵含 NaN/Inf')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("单日标签或风格矩阵含 NaN/Inf")

    # 读取 `y_all[valid]` 的当前值，并赋给 `y`。
    y = y_all[valid]
    # 读取 `styles[valid]` 的当前值，并赋给 `x`。
    x = styles[valid]
    # 调用 `np.column_stack` 按列拼接多个一维因子数组，并将结果保存到 `design`。
    design = np.column_stack([np.ones(len(y), dtype=np.float64), x])
    # 调用 `np.linalg.lstsq` 求解截面最小二乘回归系数，并将结果保存到 `(coefficients, _, rank, singular_values)`。
    coefficients, _, rank, singular_values = np.linalg.lstsq(
        design, y, rcond=None
    )
    # 读取 `design.shape[1]` 的当前值，并赋给 `expected_rank`。
    expected_rank = design.shape[1]
    # 检查 `int(rank) != expected_rank`；根据条件结果选择对应处理分支。
    if int(rank) != expected_rank:
        # 抛出 `np.linalg.LinAlgError(f'单日十风格设计矩阵秩为 {rank}，预期 {expected_rank}')`，阻止不满足数据契约的流程继续执行。
        raise np.linalg.LinAlgError(
            f"单日十风格设计矩阵秩为 {rank}，预期 {expected_rank}"
        )
    # 检查 `singular_values[-1] <= np.finfo(np.float64).eps`；根据条件结果选择对应处理分支。
    if singular_values[-1] <= np.finfo(np.float64).eps:
        # 抛出 `np.linalg.LinAlgError('单日十风格设计矩阵条件数不可计算')`，阻止不满足数据契约的流程继续执行。
        raise np.linalg.LinAlgError("单日十风格设计矩阵条件数不可计算")

    # 计算 `design @ coefficients`，并将数值结果保存到 `fitted`。
    fitted = design @ coefficients
    # 计算 `y - fitted`，并将数值结果保存到 `residual`。
    residual = y - fitted
    # 调用 `float` 统一设备或数据类型，并将结果保存到 `residual_mean`。
    residual_mean = float(residual.mean())
    # 调用 `float` 统一设备或数据类型，并将结果保存到 `residual_std`。
    residual_std = float(residual.std(ddof=0))
    # 检查 `not np.isfinite(residual_std) or residual_std <= np.finfo(np.float64).eps`；根据条件结果选择对应处理分支。
    if not np.isfinite(residual_std) or residual_std <= np.finfo(np.float64).eps:
        # 抛出 `ValueError('单日风格残差标签为常数')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("单日风格残差标签为常数")
    # 计算 `(residual - residual_mean) / residual_std`，并将数值结果保存到 `residual`。
    residual = (residual - residual_mean) / residual_std
    # 读取 `np.corrcoef(np.column_stack([residual, x]), rowvar=False)[0, 1:]` 的当前值，并赋给 `correlations`。
    correlations = np.corrcoef(
        np.column_stack([residual, x]), rowvar=False
    )[0, 1:]
    # 检查 `not np.isfinite(correlations).all()`；根据条件结果选择对应处理分支。
    if not np.isfinite(correlations).all():
        # 抛出 `ValueError('单日残差标签风格相关性不可计算')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("单日残差标签风格相关性不可计算")
    # 调用 `float` 统一设备或数据类型，并将结果保存到 `max_abs_correlation`。
    max_abs_correlation = float(np.max(np.abs(correlations)))
    # 检查 `max_abs_correlation > ORTHOGONALITY_TOLERANCE`；根据条件结果选择对应处理分支。
    if max_abs_correlation > ORTHOGONALITY_TOLERANCE:
        # 抛出 `AssertionError(f'单日残差标签最大风格相关性 {max_abs_correlation:.3e} 超过容差')`，阻止不满足数据契约的流程继续执行。
        raise AssertionError(
            f"单日残差标签最大风格相关性 {max_abs_correlation:.3e} 超过容差"
        )

    # 调用 `np.full` 按目标形状初始化数值容器 `output`。
    output = np.full(len(y_all), np.nan, dtype=np.float64)
    # 读取 `residual` 的当前值，并赋给 `output[valid]`。
    output[valid] = residual
    # 计算 `y - y.mean()`，并将数值结果保存到 `source_centered`。
    source_centered = y - y.mean()
    # 调用 `float` 统一设备或数据类型，并将结果保存到 `residual_variance`。
    residual_variance = float(np.dot(residual, residual))
    # 调用 `float` 统一设备或数据类型，并将结果保存到 `source_variance`。
    source_variance = float(np.dot(source_centered, source_centered))
    # 计算 `y - fitted`，并将数值结果保存到 `raw_residual`。
    raw_residual = y - fitted
    # 调用 `float` 统一设备或数据类型，并将结果保存到 `raw_residual_variance`。
    raw_residual_variance = float(np.dot(raw_residual, raw_residual))
    # 构造键值映射，并保存到 `diagnostics`，供后续按键查询。
    diagnostics: dict[str, float | int] = {
        "rows": len(y_all),
        "label_valid_rows": int(valid.sum()),
        "matrix_rank": int(rank),
        "condition_number": float(singular_values[0] / singular_values[-1]),
        "source_target_std": float(y.std(ddof=0)),
        "raw_residual_std": residual_std,
        "residual_target_mean": float(residual.mean()),
        "residual_target_std": float(residual.std(ddof=0)),
        "style_r_squared": float(1.0 - raw_residual_variance / source_variance),
        "max_abs_style_residual_pearson": max_abs_correlation,
        "standardized_residual_sum_squares": residual_variance,
    }
    # 将 `(output, diagnostics)` 作为当前函数结果返回给调用方。
    return output, diagnostics


# 定义 `decompose_daily_targets_with_industry` 函数，封装这一阶段可独立调用的处理步骤。
def decompose_daily_targets_with_industry(
    targets: np.ndarray,
    style_exposures: np.ndarray,
    industry_codes: np.ndarray,
    label_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    """返回完整 Barra 回归的标准化残差和标准化拟合项。
    
    输入：
        targets: np.ndarray，格式 [N]。
        style_exposures: np.ndarray，格式 [N,10]。
        industry_codes: np.ndarray，格式 [N]。
        label_valid: np.ndarray。
    输出：
        tuple[np.ndarray, np.ndarray, dict[str, float | int]]。
        """

    # 调用 `np.asarray` 转换数据表示，并将标准化对象保存到 `y_all`。
    y_all = np.asarray(targets, dtype=np.float64)
    # 调用 `np.asarray` 转换数据表示，并将标准化对象保存到 `styles_all`。
    styles_all = np.asarray(style_exposures, dtype=np.float64)
    # 调用 `np.asarray` 转换数据表示，并将标准化对象保存到 `codes_all`。
    codes_all = np.asarray(industry_codes)
    # 调用 `np.asarray` 转换数据表示，并将标准化对象保存到 `valid`。
    valid = np.asarray(label_valid, dtype=bool)
    # 检查 `styles_all.shape != (len(y_all), len(STYLE_COLUMNS))`；根据条件结果选择对应处理分支。
    if styles_all.shape != (len(y_all), len(STYLE_COLUMNS)):
        # 抛出 `ValueError('单日风格矩阵形状异常')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("单日风格矩阵形状异常")
    # 检查 `codes_all.shape != y_all.shape or not np.issubdtype(codes_all.dtype, np.integer)`；根据条件结果选择对应处理分支。
    if codes_all.shape != y_all.shape or not np.issubdtype(
        codes_all.dtype, np.integer
    ):
        # 抛出 `ValueError('行业代码必须是一维整数数组且与标签等长')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("行业代码必须是一维整数数组且与标签等长")
    # 检查 `valid.shape != y_all.shape`；根据条件结果选择对应处理分支。
    if valid.shape != y_all.shape:
        # 抛出 `ValueError('单日标签有效标记形状异常')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("单日标签有效标记形状异常")
    # 检查 `int(valid.sum()) < len(STYLE_COLUMNS) + 2`；根据条件结果选择对应处理分支。
    if int(valid.sum()) < len(STYLE_COLUMNS) + 2:
        # 抛出 `ValueError('单日有效标签不足以进行完整 Barra 回归')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("单日有效标签不足以进行完整 Barra 回归")
    # 检查 `(codes_all < 0).any() or (codes_all > EXPECTED_CNE5S_INDUSTRIES).any()`；根据条件结果选择对应处理分支。
    if (codes_all < 0).any() or (
        codes_all > EXPECTED_CNE5S_INDUSTRIES
    ).any():
        # 抛出 `ValueError('行业代码超出 0 至 32 的有效范围')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("行业代码超出 0 至 32 的有效范围")
    # 检查 `not np.isfinite(y_all[valid]).all() or not np.isfinite(styles_all).all()`；根据条件结果选择对应处理分支。
    if not np.isfinite(y_all[valid]).all() or not np.isfinite(
        styles_all
    ).all():
        # 抛出 `ValueError('单日标签或风格矩阵含 NaN/Inf')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("单日标签或风格矩阵含 NaN/Inf")

    # 读取 `y_all[valid]` 的当前值，并赋给 `y`。
    y = y_all[valid]
    # 读取 `styles_all[valid]` 的当前值，并赋给 `styles`。
    styles = styles_all[valid]
    # 读取 `codes_all[valid]` 的当前值，并赋给 `codes`。
    codes = codes_all[valid]
    # 调用 `np.unique` 提取唯一值并按需要返回反向索引，并将结果保存到 `(_, compact)`。
    _, compact = np.unique(codes, return_inverse=True)
    # 调用 `np.bincount(compact).astype` 统一设备或数据类型，并将结果保存到 `counts`。
    counts = np.bincount(compact).astype(np.float64)
    # 计算 `np.bincount(compact, weights=y) / counts`，并将数值结果保存到 `y_means`。
    y_means = np.bincount(compact, weights=y) / counts
    # 调用 `np.zeros` 按目标形状初始化数值容器 `style_sums`。
    style_sums = np.zeros(
        (len(counts), len(STYLE_COLUMNS)), dtype=np.float64
    )
    # 调用 `np.add.at` 按行业编码原地累加截面统计量。
    np.add.at(style_sums, compact, styles)
    # 计算 `style_sums / counts[:, None]`，并将数值结果保存到 `style_means`。
    style_means = style_sums / counts[:, None]
    # 计算 `y - y_means[compact]`，并将数值结果保存到 `y_within`。
    y_within = y - y_means[compact]
    # 计算 `styles - style_means[compact]`，并将数值结果保存到 `styles_within`。
    styles_within = styles - style_means[compact]

    # 调用 `np.linalg.lstsq` 求解截面最小二乘回归系数，并将结果保存到 `(coefficients, _, rank, singular_values)`。
    coefficients, _, rank, singular_values = np.linalg.lstsq(
        styles_within, y_within, rcond=None
    )
    # 检查 `int(rank) != len(STYLE_COLUMNS)`；根据条件结果选择对应处理分支。
    if int(rank) != len(STYLE_COLUMNS):
        # 抛出 `np.linalg.LinAlgError(f'行业内十风格矩阵秩为 {rank}，预期 {len(STYLE_COLUMNS)}')`，阻止不满足数据契约的流程继续执行。
        raise np.linalg.LinAlgError(
            f"行业内十风格矩阵秩为 {rank}，预期 {len(STYLE_COLUMNS)}"
        )
    # 检查 `singular_values[-1] <= np.finfo(np.float64).eps`；根据条件结果选择对应处理分支。
    if singular_values[-1] <= np.finfo(np.float64).eps:
        # 抛出 `np.linalg.LinAlgError('行业内十风格矩阵条件数不可计算')`，阻止不满足数据契约的流程继续执行。
        raise np.linalg.LinAlgError("行业内十风格矩阵条件数不可计算")

    # 计算 `y_within - styles_within @ coefficients`，并将数值结果保存到 `raw_residual`。
    raw_residual = y_within - styles_within @ coefficients
    # 计算 `y - raw_residual`，并将数值结果保存到 `raw_fitted`。
    raw_fitted = y - raw_residual
    # 调用 `float` 统一设备或数据类型，并将结果保存到 `reconstruction_error`。
    reconstruction_error = float(
        np.max(np.abs(y - raw_fitted - raw_residual))
    )
    # 检查 `reconstruction_error > 1e-12`；根据条件结果选择对应处理分支。
    if reconstruction_error > 1e-12:
        # 抛出 `AssertionError('完整 Barra 拟合项与残差无法重构源标签')`，阻止不满足数据契约的流程继续执行。
        raise AssertionError("完整 Barra 拟合项与残差无法重构源标签")
    # 调用 `float` 统一设备或数据类型，并将结果保存到 `raw_std`。
    raw_std = float(raw_residual.std(ddof=0))
    # 检查 `not np.isfinite(raw_std) or raw_std <= np.finfo(np.float64).eps`；根据条件结果选择对应处理分支。
    if not np.isfinite(raw_std) or raw_std <= np.finfo(np.float64).eps:
        # 抛出 `ValueError('单日完整 Barra 残差标签为常数')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("单日完整 Barra 残差标签为常数")
    # 调用 `float` 统一设备或数据类型，并将结果保存到 `fitted_std`。
    fitted_std = float(raw_fitted.std(ddof=0))
    # 检查 `not np.isfinite(fitted_std) or fitted_std <= np.finfo(np.float64).eps`；根据条件结果选择对应处理分支。
    if not np.isfinite(fitted_std) or fitted_std <= np.finfo(np.float64).eps:
        # 抛出 `ValueError('单日完整 Barra 拟合标签为常数')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("单日完整 Barra 拟合标签为常数")
    # 计算 `(raw_residual - raw_residual.mean()) / raw_std`，并将数值结果保存到 `residual`。
    residual = (raw_residual - raw_residual.mean()) / raw_std
    # 计算 `(raw_fitted - raw_fitted.mean()) / fitted_std`，并将数值结果保存到 `fitted`。
    fitted = (raw_fitted - raw_fitted.mean()) / fitted_std
    # 读取 `np.corrcoef(np.column_stack([residual, styles]), rowvar=False)[0, 1:]` 的当前值，并赋给 `correlations`。
    correlations = np.corrcoef(
        np.column_stack([residual, styles]), rowvar=False
    )[0, 1:]
    # 检查 `not np.isfinite(correlations).all()`；根据条件结果选择对应处理分支。
    if not np.isfinite(correlations).all():
        # 抛出 `ValueError('单日完整 Barra 残差风格相关性不可计算')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("单日完整 Barra 残差风格相关性不可计算")
    # 调用 `float` 统一设备或数据类型，并将结果保存到 `max_abs_correlation`。
    max_abs_correlation = float(np.max(np.abs(correlations)))
    # 计算 `np.bincount(compact, weights=residual) / counts`，并将数值结果保存到 `group_residual_means`。
    group_residual_means = (
        np.bincount(compact, weights=residual) / counts
    )
    # 调用 `float` 统一设备或数据类型，并将结果保存到 `max_abs_industry_mean`。
    max_abs_industry_mean = float(np.max(np.abs(group_residual_means)))
    # 检查 `max(max_abs_correlation, max_abs_industry_mean) > ORTHOGONALITY_TOLERANCE`；根据条件结果选择对应处理分支。
    if max(max_abs_correlation, max_abs_industry_mean) > ORTHOGONALITY_TOLERANCE:
        # 抛出 `AssertionError(f'单日完整 Barra 残差未达到数值正交：style={max_abs_correlation:.3e}, industry={max_abs_indust…`，阻止不满足数据契约的流程继续执行。
        raise AssertionError(
            "单日完整 Barra 残差未达到数值正交："
            f"style={max_abs_correlation:.3e}, "
            f"industry={max_abs_industry_mean:.3e}"
        )

    # 计算 `y - y.mean()`，并将数值结果保存到 `centered`。
    centered = y - y.mean()
    # 调用 `float` 统一设备或数据类型，并将结果保存到 `total_sum_squares`。
    total_sum_squares = float(np.dot(centered, centered))
    # 调用 `float` 统一设备或数据类型，并将结果保存到 `residual_sum_squares`。
    residual_sum_squares = float(np.dot(raw_residual, raw_residual))
    # 检查 `total_sum_squares <= np.finfo(np.float64).eps`；根据条件结果选择对应处理分支。
    if total_sum_squares <= np.finfo(np.float64).eps:
        # 抛出 `ValueError('单日源标签为常数')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("单日源标签为常数")
    # 调用 `np.full` 按目标形状初始化数值容器 `output`。
    output = np.full(len(y_all), np.nan, dtype=np.float64)
    # 调用 `np.full` 按目标形状初始化数值容器 `fitted_output`。
    fitted_output = np.full(len(y_all), np.nan, dtype=np.float64)
    # 读取 `residual` 的当前值，并赋给 `output[valid]`。
    output[valid] = residual
    # 读取 `fitted` 的当前值，并赋给 `fitted_output[valid]`。
    fitted_output[valid] = fitted
    # 将 `(output, fitted_output, {'rows': len(y_all), 'label_valid_rows': int(valid.sum()), 'industry_co…` 作为当前函数结果返回给调用方。
    return output, fitted_output, {
        "rows": len(y_all),
        "label_valid_rows": int(valid.sum()),
        "industry_count": int(len(counts)),
        "unknown_industry_rows": int((codes == 0).sum()),
        "unknown_industry_rate": float((codes == 0).mean()),
        "matrix_rank": int(rank),
        "condition_number": float(singular_values[0] / singular_values[-1]),
        "source_target_std": float(y.std(ddof=0)),
        "raw_residual_std": raw_std,
        "raw_fitted_std": fitted_std,
        "reconstruction_max_abs_error": reconstruction_error,
        "residual_target_mean": float(residual.mean()),
        "residual_target_std": float(residual.std(ddof=0)),
        "fitted_target_mean": float(fitted.mean()),
        "fitted_target_std": float(fitted.std(ddof=0)),
        "joint_barra_r_squared": float(
            1.0 - residual_sum_squares / total_sum_squares
        ),
        "max_abs_style_residual_pearson": max_abs_correlation,
        "max_abs_industry_residual_mean": max_abs_industry_mean,
    }


# 定义 `residualize_daily_targets_with_industry` 函数，封装这一阶段可独立调用的处理步骤。
def residualize_daily_targets_with_industry(
    targets: np.ndarray,
    style_exposures: np.ndarray,
    industry_codes: np.ndarray,
    label_valid: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """对单日标签回归十风格和行业固定效应，再标准化残差。
    
    输入：
        targets: np.ndarray，格式 [N]。
        style_exposures: np.ndarray，格式 [N,10]。
        industry_codes: np.ndarray，格式 [N]。
        label_valid: np.ndarray。
    输出：
        tuple[np.ndarray, dict[str, float | int]]。
        """

    # 调用 `decompose_daily_targets_with_industry` 执行 `decompose_daily_targets_with_industry` 对应的明确变换，并产生后续步骤使用的结果，并将结果保存到 `(residual, _fitted, diagnostics)`。
    residual, _fitted, diagnostics = decompose_daily_targets_with_industry(
        targets, style_exposures, industry_codes, label_valid
    )
    # 将 `(residual, diagnostics)` 作为当前函数结果返回给调用方。
    return residual, diagnostics


# 定义 `project_out_styles_numpy` 函数，封装这一阶段可独立调用的处理步骤。
def project_out_styles_numpy(
    predictions: np.ndarray, style_exposures: np.ndarray
) -> tuple[np.ndarray, dict[str, float | int]]:
    """NumPy 参考实现：从单日预测中联合投影掉十个风格。
    
    输入：
        predictions: np.ndarray，格式 [N]。
        style_exposures: np.ndarray，格式 [N,10]。
    输出：
        tuple[np.ndarray, dict[str, float | int]]。
        """

    # 调用 `np.asarray` 转换数据表示，并将标准化对象保存到 `values`。
    values = np.asarray(predictions, dtype=np.float64)
    # 调用 `np.asarray` 转换数据表示，并将标准化对象保存到 `styles`。
    styles = np.asarray(style_exposures, dtype=np.float64)
    # 检查 `values.ndim != 1 or styles.shape != (len(values), len(STYLE_COLUMNS))`；根据条件结果选择对应处理分支。
    if values.ndim != 1 or styles.shape != (len(values), len(STYLE_COLUMNS)):
        # 抛出 `ValueError('预测或风格矩阵形状异常')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("预测或风格矩阵形状异常")
    # 检查 `len(values) < len(STYLE_COLUMNS) + 2`；根据条件结果选择对应处理分支。
    if len(values) < len(STYLE_COLUMNS) + 2:
        # 抛出 `ValueError('单日股票数不足以进行十风格投影')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("单日股票数不足以进行十风格投影")
    # 检查 `not np.isfinite(values).all() or not np.isfinite(styles).all()`；根据条件结果选择对应处理分支。
    if not np.isfinite(values).all() or not np.isfinite(styles).all():
        # 抛出 `ValueError('预测或风格矩阵含 NaN/Inf')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("预测或风格矩阵含 NaN/Inf")
    # 调用 `np.column_stack` 按列拼接多个一维因子数组，并将结果保存到 `design`。
    design = np.column_stack([np.ones(len(values)), styles])
    # 调用 `np.linalg.lstsq` 求解截面最小二乘回归系数，并将结果保存到 `(coefficients, _, rank, singular_values)`。
    coefficients, _, rank, singular_values = np.linalg.lstsq(
        design, values, rcond=None
    )
    # 检查 `int(rank) != design.shape[1]`；根据条件结果选择对应处理分支。
    if int(rank) != design.shape[1]:
        # 抛出 `np.linalg.LinAlgError('十风格投影矩阵秩不足')`，阻止不满足数据契约的流程继续执行。
        raise np.linalg.LinAlgError("十风格投影矩阵秩不足")
    # 计算 `values - design @ coefficients`，并将数值结果保存到 `residual`。
    residual = values - design @ coefficients
    # 读取 `np.corrcoef(np.column_stack([residual, styles]), rowvar=False)[0, 1:]` 的当前值，并赋给 `correlations`。
    correlations = np.corrcoef(
        np.column_stack([residual, styles]), rowvar=False
    )[0, 1:]
    # 将 `(residual, {'matrix_rank': int(rank), 'condition_number': float(singular_values[0] / singular_v…` 作为当前函数结果返回给调用方。
    return residual, {
        "matrix_rank": int(rank),
        "condition_number": float(singular_values[0] / singular_values[-1]),
        "max_abs_style_pearson": float(np.max(np.abs(correlations))),
        "residual_mean": float(residual.mean()),
    }


# 定义 `project_out_styles_torch` 函数，封装这一阶段可独立调用的处理步骤。
def project_out_styles_torch(
    predictions: torch.Tensor,
    style_exposures: torch.Tensor,
    *,
    standardize: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """以 FP32 QR 将单日模型分数投影到十风格正交补，并可恢复单位尺度。
    
    输入：
        predictions: torch.Tensor，格式 [N]。
        style_exposures: torch.Tensor，格式 [N,10]。
        standardize: bool。
    输出：
        tuple[torch.Tensor, dict[str, torch.Tensor]]。
        """

    # 调用 `predictions.float().reshape` 调整张量维度，并将新视图保存到 `values`。
    values = predictions.float().reshape(-1)
    # 调用 `style_exposures.float` 统一设备或数据类型，并将结果保存到 `styles`。
    styles = style_exposures.float()
    # 检查 `styles.shape != (len(values), len(STYLE_COLUMNS))`；根据条件结果选择对应处理分支。
    if styles.shape != (len(values), len(STYLE_COLUMNS)):
        # 抛出 `ValueError('预测或风格矩阵形状异常')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("预测或风格矩阵形状异常")
    # 检查 `len(values) < len(STYLE_COLUMNS) + 2`；根据条件结果选择对应处理分支。
    if len(values) < len(STYLE_COLUMNS) + 2:
        # 抛出 `ValueError('单日股票数不足以进行十风格投影')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("单日股票数不足以进行十风格投影")
    # 检查 `not torch.isfinite(values).all() or not torch.isfinite(styles).all()`；根据条件结果选择对应处理分支。
    if not torch.isfinite(values).all() or not torch.isfinite(styles).all():
        # 抛出 `ValueError('预测或风格矩阵含 NaN/Inf')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("预测或风格矩阵含 NaN/Inf")

    # 调用 `torch.column_stack` 按列拼接多个一维因子数组，并将结果保存到 `design`。
    design = torch.column_stack(
        [torch.ones_like(values, dtype=torch.float32), styles]
    )
    # 调用 `torch.linalg.qr` 对设计矩阵做 QR 分解以获得稳定投影基，并将结果保存到 `(q, r)`。
    q, r = torch.linalg.qr(design, mode="reduced")
    # 调用 `torch.abs` 逐元素取绝对值，并将结果保存到 `diagonal`。
    diagonal = torch.abs(torch.diagonal(r))
    # 计算 `torch.finfo(torch.float32).eps * max(design.shape) * diagonal.max()`，并将数值结果保存到 `tolerance`。
    tolerance = (
        torch.finfo(torch.float32).eps
        * max(design.shape)
        * diagonal.max()
    )
    # 检查 `bool((diagonal <= tolerance).any())`；根据条件结果选择对应处理分支。
    if bool((diagonal <= tolerance).any()):
        # 抛出 `np.linalg.LinAlgError('十风格投影矩阵秩不足')`，阻止不满足数据契约的流程继续执行。
        raise np.linalg.LinAlgError("十风格投影矩阵秩不足")

    # 计算 `values - q @ (q.transpose(0, 1) @ values)`，并将数值结果保存到 `residual`。
    residual = values - q @ (q.transpose(0, 1) @ values)
    # 计算 `residual - residual.mean()`，并将数值结果保存到 `residual`。
    residual = residual - residual.mean()
    # 调用 `residual.std` 汇总当前数据，并将统计量保存到 `residual_std`。
    residual_std = residual.std(unbiased=False)
    # 检查 `not torch.isfinite(residual_std) or bool(residual_std <= torch.finfo(torch.float32).eps)`；根据条件结果选择对应处理分支。
    if not torch.isfinite(residual_std) or bool(
        residual_std <= torch.finfo(torch.float32).eps
    ):
        # 抛出 `ValueError('单日投影残差为常数')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("单日投影残差为常数")
    # 按 `standardize` 在两个候选值间选择，并保存到 `effective`。
    effective = residual / residual_std if standardize else residual
    # 计算 `styles - styles.mean(dim=0, keepdim=True)`，并将数值结果保存到 `centered_styles`。
    centered_styles = styles - styles.mean(dim=0, keepdim=True)
    # 计算 `centered_styles.transpose(0, 1) @ effective / torch.sqrt(centered_styles.square().sum(dim=0) * …`，并将数值结果保存到 `correlations`。
    correlations = (
        centered_styles.transpose(0, 1) @ effective
    ) / torch.sqrt(
        centered_styles.square().sum(dim=0)
        * effective.square().sum()
    )
    # 检查 `not torch.isfinite(correlations).all()`；根据条件结果选择对应处理分支。
    if not torch.isfinite(correlations).all():
        # 抛出 `ValueError('投影后风格相关性不可计算')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("投影后风格相关性不可计算")
    # 将 `(effective, {'residual_mean': effective.mean().detach(), 'residual_std': effective.std(unbiased…` 作为当前函数结果返回给调用方。
    return effective, {
        "residual_mean": effective.mean().detach(),
        "residual_std": effective.std(unbiased=False).detach(),
        "max_abs_style_pearson": correlations.abs().max().detach(),
    }


# 定义 `joint_barra_r_squared_numpy` 函数，封装这一阶段可独立调用的处理步骤。
def joint_barra_r_squared_numpy(
    predictions: np.ndarray,
    style_exposures: np.ndarray,
    industry_codes: np.ndarray,
) -> tuple[float, dict[str, float | int]]:
    """计算 raw score 对十风格和行业固定效应的单日联合 R²。
    
    输入：
        predictions: np.ndarray，格式 [N]。
        style_exposures: np.ndarray，格式 [N,10]。
        industry_codes: np.ndarray，格式 [N]。
    输出：
        tuple[float, dict[str, float | int]]。
        """

    # 调用 `np.asarray` 转换数据表示，并将标准化对象保存到 `values`。
    values = np.asarray(predictions, dtype=np.float64)
    # 调用 `np.asarray` 转换数据表示，并将标准化对象保存到 `styles`。
    styles = np.asarray(style_exposures, dtype=np.float64)
    # 调用 `np.asarray` 转换数据表示，并将标准化对象保存到 `codes`。
    codes = np.asarray(industry_codes)
    # 检查 `values.ndim != 1 or styles.shape != (len(values), len(STYLE_COLUMNS))`；根据条件结果选择对应处理分支。
    if values.ndim != 1 or styles.shape != (len(values), len(STYLE_COLUMNS)):
        # 抛出 `ValueError('预测或十风格矩阵形状异常')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("预测或十风格矩阵形状异常")
    # 检查 `codes.shape != values.shape or not np.issubdtype(codes.dtype, np.integer)`；根据条件结果选择对应处理分支。
    if codes.shape != values.shape or not np.issubdtype(codes.dtype, np.integer):
        # 抛出 `ValueError('行业代码必须是一维整数数组且与预测等长')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("行业代码必须是一维整数数组且与预测等长")
    # 检查 `len(values) < len(STYLE_COLUMNS) + 2`；根据条件结果选择对应处理分支。
    if len(values) < len(STYLE_COLUMNS) + 2:
        # 抛出 `ValueError('单日股票数不足以计算联合 Barra R²')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("单日股票数不足以计算联合 Barra R²")
    # 检查 `(codes < 0).any()`；根据条件结果选择对应处理分支。
    if (codes < 0).any():
        # 抛出 `ValueError('行业代码不能为负数')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("行业代码不能为负数")
    # 检查 `not np.isfinite(values).all() or not np.isfinite(styles).all()`；根据条件结果选择对应处理分支。
    if not np.isfinite(values).all() or not np.isfinite(styles).all():
        # 抛出 `ValueError('预测或十风格矩阵含 NaN/Inf')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("预测或十风格矩阵含 NaN/Inf")

    # 调用 `np.unique` 提取唯一值并按需要返回反向索引，并将结果保存到 `(_, compact)`。
    _, compact = np.unique(codes, return_inverse=True)
    # 调用 `np.bincount(compact).astype` 统一设备或数据类型，并将结果保存到 `counts`。
    counts = np.bincount(compact).astype(np.float64)
    # 计算 `np.bincount(compact, weights=values) / counts`，并将数值结果保存到 `value_means`。
    value_means = np.bincount(compact, weights=values) / counts
    # 调用 `np.zeros` 按目标形状初始化数值容器 `style_sums`。
    style_sums = np.zeros((len(counts), len(STYLE_COLUMNS)), dtype=np.float64)
    # 调用 `np.add.at` 按行业编码原地累加截面统计量。
    np.add.at(style_sums, compact, styles)
    # 计算 `style_sums / counts[:, None]`，并将数值结果保存到 `style_means`。
    style_means = style_sums / counts[:, None]
    # 计算 `values - value_means[compact]`，并将数值结果保存到 `values_within`。
    values_within = values - value_means[compact]
    # 计算 `styles - style_means[compact]`，并将数值结果保存到 `styles_within`。
    styles_within = styles - style_means[compact]

    # 调用 `np.linalg.lstsq` 求解截面最小二乘回归系数，并将结果保存到 `(coefficients, _, rank, singular_values)`。
    coefficients, _, rank, singular_values = np.linalg.lstsq(
        styles_within, values_within, rcond=None
    )
    # 检查 `int(rank) != len(STYLE_COLUMNS)`；根据条件结果选择对应处理分支。
    if int(rank) != len(STYLE_COLUMNS):
        # 抛出 `np.linalg.LinAlgError(f'行业内十风格矩阵秩为 {rank}，预期 {len(STYLE_COLUMNS)}')`，阻止不满足数据契约的流程继续执行。
        raise np.linalg.LinAlgError(
            f"行业内十风格矩阵秩为 {rank}，预期 {len(STYLE_COLUMNS)}"
        )
    # 计算 `values_within - styles_within @ coefficients`，并将数值结果保存到 `residual`。
    residual = values_within - styles_within @ coefficients
    # 计算 `values - values.mean()`，并将数值结果保存到 `centered`。
    centered = values - values.mean()
    # 调用 `float` 统一设备或数据类型，并将结果保存到 `total_sum_squares`。
    total_sum_squares = float(np.dot(centered, centered))
    # 检查 `total_sum_squares <= np.finfo(np.float64).eps`；根据条件结果选择对应处理分支。
    if total_sum_squares <= np.finfo(np.float64).eps:
        # 抛出 `ValueError('单日 raw score 为常数')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("单日 raw score 为常数")
    # 调用 `float` 统一设备或数据类型，并将结果保存到 `residual_sum_squares`。
    residual_sum_squares = float(np.dot(residual, residual))
    # 计算 `1.0 - residual_sum_squares / total_sum_squares`，并将数值结果保存到 `r_squared`。
    r_squared = 1.0 - residual_sum_squares / total_sum_squares
    # 设置 `tolerance` 的固定配置值，供后续流程统一使用。
    tolerance = 1e-10
    # 检查 `not np.isfinite(r_squared) or not -tolerance <= r_squared <= 1.0 + tolerance`；根据条件结果选择对应处理分支。
    if not np.isfinite(r_squared) or not -tolerance <= r_squared <= 1.0 + tolerance:
        # 抛出 `FloatingPointError(f'联合 Barra R² 超出数值范围：{r_squared}')`，阻止不满足数据契约的流程继续执行。
        raise FloatingPointError(f"联合 Barra R² 超出数值范围：{r_squared}")
    # 调用 `float` 统一设备或数据类型，并将结果保存到 `r_squared`。
    r_squared = float(np.clip(r_squared, 0.0, 1.0))
    # 计算 `np.bincount(compact, weights=residual) / counts`，并将数值结果保存到 `group_residual_means`。
    group_residual_means = np.bincount(compact, weights=residual) / counts
    # 将 `(r_squared, {'r_squared': r_squared, 'industry_count': int(len(counts)), 'matrix_rank': int(ran…` 作为当前函数结果返回给调用方。
    return r_squared, {
        "r_squared": r_squared,
        "industry_count": int(len(counts)),
        "matrix_rank": int(rank),
        "condition_number": float(singular_values[0] / singular_values[-1]),
        "residual_sum_squares": residual_sum_squares,
        "total_sum_squares": total_sum_squares,
        "max_abs_industry_residual_mean": float(
            np.max(np.abs(group_residual_means))
        ),
    }


# 定义 `joint_barra_r_squared_torch` 函数，封装这一阶段可独立调用的处理步骤。
def joint_barra_r_squared_torch(
    predictions: torch.Tensor,
    style_exposures: torch.Tensor,
    industry_codes: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """可微联合 R²；Barra/行业固定，梯度只回传到 raw score。
    
    输入：
        predictions: torch.Tensor，格式 [N]。
        style_exposures: torch.Tensor，格式 [N,10]。
        industry_codes: torch.Tensor，格式 [N]。
    输出：
        tuple[torch.Tensor, dict[str, torch.Tensor]]。
        """

    # 读取 `predictions.device.type` 的当前值，并赋给 `device_type`。
    device_type = predictions.device.type
    # 在 `torch.autocast(device_type=device_type, enabled=False)` 管理的上下文中执行本段逻辑，并保证资源按约定释放。
    with torch.autocast(device_type=device_type, enabled=False):
        # 调用 `predictions.float().reshape` 调整张量维度，并将新视图保存到 `values`。
        values = predictions.float().reshape(-1)
        # 调用 `style_exposures.float` 统一设备或数据类型，并将结果保存到 `styles`。
        styles = style_exposures.float()
        # 调用 `industry_codes.reshape` 调整张量维度，并将新视图保存到 `codes`。
        codes = industry_codes.reshape(-1)
        # 组织一组有序值，并保存到 `integer_dtypes`。
        integer_dtypes = (
            torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8
        )
        # 检查 `styles.shape != (len(values), len(STYLE_COLUMNS))`；根据条件结果选择对应处理分支。
        if styles.shape != (len(values), len(STYLE_COLUMNS)):
            # 抛出 `ValueError('预测或十风格矩阵形状异常')`，阻止不满足数据契约的流程继续执行。
            raise ValueError("预测或十风格矩阵形状异常")
        # 检查 `codes.shape != values.shape or codes.dtype not in integer_dtypes`；根据条件结果选择对应处理分支。
        if codes.shape != values.shape or codes.dtype not in integer_dtypes:
            # 抛出 `ValueError('行业代码必须是一维整数张量且与预测等长')`，阻止不满足数据契约的流程继续执行。
            raise ValueError("行业代码必须是一维整数张量且与预测等长")
        # 检查 `len(values) < len(STYLE_COLUMNS) + 2`；根据条件结果选择对应处理分支。
        if len(values) < len(STYLE_COLUMNS) + 2:
            # 抛出 `ValueError('单日股票数不足以计算联合 Barra R²')`，阻止不满足数据契约的流程继续执行。
            raise ValueError("单日股票数不足以计算联合 Barra R²")
        # 检查 `bool((codes < 0).any())`；根据条件结果选择对应处理分支。
        if bool((codes < 0).any()):
            # 抛出 `ValueError('行业代码不能为负数')`，阻止不满足数据契约的流程继续执行。
            raise ValueError("行业代码不能为负数")
        # 检查 `not bool(torch.isfinite(values).all()) or not bool(torch.isfinite(styles).all())`；根据条件结果选择对应处理分支。
        if not bool(torch.isfinite(values).all()) or not bool(
            torch.isfinite(styles).all()
        ):
            # 抛出 `ValueError('预测或十风格矩阵含 NaN/Inf')`，阻止不满足数据契约的流程继续执行。
            raise ValueError("预测或十风格矩阵含 NaN/Inf")

        # 调用 `torch.unique` 提取唯一值并按需要返回反向索引，并将结果保存到 `(_, compact)`。
        _, compact = torch.unique(codes.long(), sorted=True, return_inverse=True)
        # 计算 `int(compact.max().item()) + 1`，并将数值结果保存到 `group_count`。
        group_count = int(compact.max().item()) + 1
        # 调用 `torch.zeros` 按目标形状初始化数值容器 `counts`。
        counts = torch.zeros(group_count, dtype=torch.float32, device=values.device)
        # 调用 `counts.scatter_add_` 收集并汇总当前条件下满足要求的记录。
        counts.scatter_add_(0, compact, torch.ones_like(values))
        # 调用 `torch.zeros_like(counts).scatter_add` 收集并汇总当前条件下满足要求的记录，并将结果保存到 `value_sums`。
        value_sums = torch.zeros_like(counts).scatter_add(0, compact, values)
        # 调用 `torch.zeros` 按目标形状初始化数值容器 `style_sums`。
        style_sums = torch.zeros(
            group_count,
            len(STYLE_COLUMNS),
            dtype=torch.float32,
            device=values.device,
        )
        # 调用 `style_sums.index_add_` 执行 `index_add_` 对应的明确变换，并产生后续步骤使用的结果。
        style_sums.index_add_(0, compact, styles)
        # 计算 `values - (value_sums / counts)[compact]`，并将数值结果保存到 `values_within`。
        values_within = values - (value_sums / counts)[compact]
        # 计算 `styles - (style_sums / counts[:, None])[compact]`，并将数值结果保存到 `styles_within`。
        styles_within = styles - (style_sums / counts[:, None])[compact]

        # 调用 `torch.linalg.qr` 对设计矩阵做 QR 分解以获得稳定投影基，并将结果保存到 `(q, r)`。
        q, r = torch.linalg.qr(styles_within, mode="reduced")
        # 调用 `torch.abs` 逐元素取绝对值，并将结果保存到 `diagonal`。
        diagonal = torch.abs(torch.diagonal(r))
        # 计算 `torch.finfo(torch.float32).eps * max(styles_within.shape) * diagonal.max()`，并将数值结果保存到 `rank_tolerance`。
        rank_tolerance = (
            torch.finfo(torch.float32).eps
            * max(styles_within.shape)
            * diagonal.max()
        )
        # 检查 `bool((diagonal <= rank_tolerance).any())`；根据条件结果选择对应处理分支。
        if bool((diagonal <= rank_tolerance).any()):
            # 抛出 `np.linalg.LinAlgError('行业内十风格矩阵秩不足')`，阻止不满足数据契约的流程继续执行。
            raise np.linalg.LinAlgError("行业内十风格矩阵秩不足")
        # 计算 `values_within - q @ (q.transpose(0, 1) @ values_within)`，并将数值结果保存到 `residual`。
        residual = values_within - q @ (q.transpose(0, 1) @ values_within)
        # 计算 `values - values.mean()`，并将数值结果保存到 `centered`。
        centered = values - values.mean()
        # 调用 `centered.square().sum` 汇总当前数据，并将统计量保存到 `total_sum_squares`。
        total_sum_squares = centered.square().sum()
        # 检查 `not bool(torch.isfinite(total_sum_squares)) or bool(total_sum_squares <= torch.finfo(torch.floa…`；根据条件结果选择对应处理分支。
        if not bool(torch.isfinite(total_sum_squares)) or bool(
            total_sum_squares <= torch.finfo(torch.float32).eps
        ):
            # 抛出 `ValueError('单日 raw score 为常数')`，阻止不满足数据契约的流程继续执行。
            raise ValueError("单日 raw score 为常数")
        # 调用 `residual.square().sum` 汇总当前数据，并将统计量保存到 `residual_sum_squares`。
        residual_sum_squares = residual.square().sum()
        # 计算 `1.0 - residual_sum_squares / total_sum_squares`，并将数值结果保存到 `raw_r_squared`。
        raw_r_squared = 1.0 - residual_sum_squares / total_sum_squares
        # 检查 `not bool(torch.isfinite(raw_r_squared)) or bool((raw_r_squared < -0.0001) | (raw_r_squared > 1.…`；根据条件结果选择对应处理分支。
        if not bool(torch.isfinite(raw_r_squared)) or bool(
            (raw_r_squared < -1e-4) | (raw_r_squared > 1.0 + 1e-4)
        ):
            # 抛出 `FloatingPointError(f'联合 Barra R² 超出数值范围：{float(raw_r_squared.detach())}')`，阻止不满足数据契约的流程继续执行。
            raise FloatingPointError(
                f"联合 Barra R² 超出数值范围：{float(raw_r_squared.detach())}"
            )
        # 调用 `raw_r_squared.clamp` 把统计量限制在合法数值区间，并将结果保存到 `r_squared`。
        r_squared = raw_r_squared.clamp(0.0, 1.0)
        # 调用 `torch.zeros_like(counts).scatter_add` 收集并汇总当前条件下满足要求的记录，并将结果保存到 `residual_sums`。
        residual_sums = torch.zeros_like(counts).scatter_add(
            0, compact, residual
        )
        # 将 `(r_squared, {'r_squared': r_squared.detach(), 'industry_count': torch.tensor(group_count, dtype…` 作为当前函数结果返回给调用方。
        return r_squared, {
            "r_squared": r_squared.detach(),
            "industry_count": torch.tensor(
                group_count, dtype=torch.int64, device=values.device
            ),
            "matrix_rank": torch.tensor(
                len(STYLE_COLUMNS), dtype=torch.int64, device=values.device
            ),
            "residual_sum_squares": residual_sum_squares.detach(),
            "total_sum_squares": total_sum_squares.detach(),
            "max_abs_industry_residual_mean": (
                residual_sums / counts
            ).abs().max().detach(),
        }




# 定义 `project_out_full_barra_torch` 函数，封装这一阶段可独立调用的处理步骤。
def project_out_full_barra_torch(
    predictions: torch.Tensor,
    style_exposures: torch.Tensor,
    industry_codes: torch.Tensor,
    *,
    standardize: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """投影掉十个连续 Barra 暴露与行业固定效应，梯度回传到分数。
    
    输入：
        predictions: torch.Tensor，格式 [N]。
        style_exposures: torch.Tensor，格式 [N,10]。
        industry_codes: torch.Tensor，格式 [N]。
        standardize: bool。
    输出：
        tuple[torch.Tensor, dict[str, torch.Tensor]]。
        """

    # 读取 `predictions.device.type` 的当前值，并赋给 `device_type`。
    device_type = predictions.device.type
    # 在 `torch.autocast(device_type=device_type, enabled=False)` 管理的上下文中执行本段逻辑，并保证资源按约定释放。
    with torch.autocast(device_type=device_type, enabled=False):
        # 调用 `predictions.float().reshape` 调整张量维度，并将新视图保存到 `values`。
        values = predictions.float().reshape(-1)
        # 调用 `style_exposures.float` 统一设备或数据类型，并将结果保存到 `styles`。
        styles = style_exposures.float()
        # 调用 `industry_codes.reshape` 调整张量维度，并将新视图保存到 `codes`。
        codes = industry_codes.reshape(-1)
        # 组织一组有序值，并保存到 `integer_dtypes`。
        integer_dtypes = (
            torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8
        )
        # 检查 `styles.shape != (len(values), len(STYLE_COLUMNS))`；根据条件结果选择对应处理分支。
        if styles.shape != (len(values), len(STYLE_COLUMNS)):
            # 抛出 `ValueError('预测或十风格矩阵形状异常')`，阻止不满足数据契约的流程继续执行。
            raise ValueError("预测或十风格矩阵形状异常")
        # 检查 `codes.shape != values.shape or codes.dtype not in integer_dtypes`；根据条件结果选择对应处理分支。
        if codes.shape != values.shape or codes.dtype not in integer_dtypes:
            # 抛出 `ValueError('行业代码必须是一维整数张量且与预测等长')`，阻止不满足数据契约的流程继续执行。
            raise ValueError("行业代码必须是一维整数张量且与预测等长")
        # 检查 `len(values) < len(STYLE_COLUMNS) + 2`；根据条件结果选择对应处理分支。
        if len(values) < len(STYLE_COLUMNS) + 2:
            # 抛出 `ValueError('单日股票数不足以进行完整 Barra 投影')`，阻止不满足数据契约的流程继续执行。
            raise ValueError("单日股票数不足以进行完整 Barra 投影")
        # 检查 `bool(((codes < 0) | (codes > EXPECTED_CNE5S_INDUSTRIES)).any())`；根据条件结果选择对应处理分支。
        if bool(
            ((codes < 0) | (codes > EXPECTED_CNE5S_INDUSTRIES)).any()
        ):
            # 抛出 `ValueError('行业代码超出 0 至 32 的有效范围')`，阻止不满足数据契约的流程继续执行。
            raise ValueError("行业代码超出 0 至 32 的有效范围")
        # 检查 `not bool(torch.isfinite(values).all()) or not bool(torch.isfinite(styles).all())`；根据条件结果选择对应处理分支。
        if not bool(torch.isfinite(values).all()) or not bool(
            torch.isfinite(styles).all()
        ):
            # 抛出 `ValueError('预测或十风格矩阵含 NaN/Inf')`，阻止不满足数据契约的流程继续执行。
            raise ValueError("预测或十风格矩阵含 NaN/Inf")

        # 调用 `torch.unique` 提取唯一值并按需要返回反向索引，并将结果保存到 `(_, compact)`。
        _, compact = torch.unique(
            codes.long(), sorted=True, return_inverse=True
        )
        # 计算 `int(compact.max().item()) + 1`，并将数值结果保存到 `group_count`。
        group_count = int(compact.max().item()) + 1
        # 调用 `torch.zeros` 按目标形状初始化数值容器 `counts`。
        counts = torch.zeros(
            group_count, dtype=torch.float32, device=values.device
        )
        # 调用 `counts.scatter_add_` 收集并汇总当前条件下满足要求的记录。
        counts.scatter_add_(0, compact, torch.ones_like(values))
        # 调用 `torch.zeros_like(counts).scatter_add` 收集并汇总当前条件下满足要求的记录，并将结果保存到 `value_sums`。
        value_sums = torch.zeros_like(counts).scatter_add(
            0, compact, values
        )
        # 调用 `torch.zeros` 按目标形状初始化数值容器 `style_sums`。
        style_sums = torch.zeros(
            group_count,
            len(STYLE_COLUMNS),
            dtype=torch.float32,
            device=values.device,
        )
        # 调用 `style_sums.index_add_` 执行 `index_add_` 对应的明确变换，并产生后续步骤使用的结果。
        style_sums.index_add_(0, compact, styles)
        # 计算 `values - (value_sums / counts)[compact]`，并将数值结果保存到 `values_within`。
        values_within = values - (value_sums / counts)[compact]
        # 计算 `styles - (style_sums / counts[:, None])[compact]`，并将数值结果保存到 `styles_within`。
        styles_within = styles - (style_sums / counts[:, None])[compact]

        # 调用 `torch.linalg.qr` 对设计矩阵做 QR 分解以获得稳定投影基，并将结果保存到 `(q, r)`。
        q, r = torch.linalg.qr(styles_within, mode="reduced")
        # 调用 `torch.abs` 逐元素取绝对值，并将结果保存到 `diagonal`。
        diagonal = torch.abs(torch.diagonal(r))
        # 计算 `torch.finfo(torch.float32).eps * max(styles_within.shape) * diagonal.max()`，并将数值结果保存到 `rank_tolerance`。
        rank_tolerance = (
            torch.finfo(torch.float32).eps
            * max(styles_within.shape)
            * diagonal.max()
        )
        # 检查 `bool((diagonal <= rank_tolerance).any())`；根据条件结果选择对应处理分支。
        if bool((diagonal <= rank_tolerance).any()):
            # 抛出 `np.linalg.LinAlgError('行业内十风格矩阵秩不足')`，阻止不满足数据契约的流程继续执行。
            raise np.linalg.LinAlgError("行业内十风格矩阵秩不足")
        # 计算 `values_within - q @ (q.transpose(0, 1) @ values_within)`，并将数值结果保存到 `residual`。
        residual = values_within - q @ (
            q.transpose(0, 1) @ values_within
        )
        # 检查 `standardize`；根据条件结果选择对应处理分支。
        if standardize:
            # 计算 `residual - residual.mean()`，并将数值结果保存到 `centered`。
            centered = residual - residual.mean()
            # 调用 `centered.std` 汇总当前数据，并将统计量保存到 `standard_deviation`。
            standard_deviation = centered.std(unbiased=False)
            # 检查 `not bool(torch.isfinite(standard_deviation)) or bool(standard_deviation <= torch.finfo(torch.fl…`；根据条件结果选择对应处理分支。
            if not bool(torch.isfinite(standard_deviation)) or bool(
                standard_deviation <= torch.finfo(torch.float32).eps
            ):
                # 抛出 `ValueError('完整 Barra 投影后的分数为常数')`，阻止不满足数据契约的流程继续执行。
                raise ValueError("完整 Barra 投影后的分数为常数")
            # 计算 `centered / standard_deviation`，并将数值结果保存到 `output`。
            output = centered / standard_deviation
        # 当前置条件不成立时进入备用分支，保证其余合法情况仍被处理。
        else:
            # 读取 `residual` 的当前值，并赋给 `output`。
            output = residual

        # 计算 `styles - styles.mean(dim=0, keepdim=True)`，并将数值结果保存到 `style_centered`。
        style_centered = styles - styles.mean(dim=0, keepdim=True)
        # 调用 `torch.sqrt` 对输入开平方得到范数或标准差相关量，并将结果保存到 `denominator`。
        denominator = torch.sqrt(
            output.square().sum()
            * style_centered.square().sum(dim=0)
        )
        # 检查 `bool((denominator <= torch.finfo(torch.float32).eps).any())`；根据条件结果选择对应处理分支。
        if bool((denominator <= torch.finfo(torch.float32).eps).any()):
            # 抛出 `ValueError('完整 Barra 投影后的风格相关性不可计算')`，阻止不满足数据契约的流程继续执行。
            raise ValueError("完整 Barra 投影后的风格相关性不可计算")
        # 计算 `(output[:, None] * style_centered).sum(dim=0) / denominator`，并将数值结果保存到 `correlations`。
        correlations = (
            output[:, None] * style_centered
        ).sum(dim=0) / denominator
        # 调用 `torch.zeros_like(counts).scatter_add` 收集并汇总当前条件下满足要求的记录，并将结果保存到 `residual_sums`。
        residual_sums = torch.zeros_like(counts).scatter_add(
            0, compact, output
        )
        # 将 `(output, {'industry_count': torch.tensor(group_count, dtype=torch.int64, device=values.device),…` 作为当前函数结果返回给调用方。
        return output, {
            "industry_count": torch.tensor(
                group_count, dtype=torch.int64, device=values.device
            ),
            "matrix_rank": torch.tensor(
                len(STYLE_COLUMNS), dtype=torch.int64, device=values.device
            ),
            "condition_number": (
                diagonal.max() / diagonal.min()
            ).detach(),
            "max_abs_style_pearson": correlations.abs().max().detach(),
            "max_abs_industry_mean": (
                residual_sums / counts
            ).abs().max().detach(),
            "output_mean": output.mean().detach(),
            "output_std": output.std(unbiased=False).detach(),
        }






# 定义内部辅助函数 `_industry_sidecar_payload`，封装仅供本模块复用的计算步骤。
def _industry_sidecar_payload(
    sample_index_path: Path,
    cne5s_path: Path,
    style_store_signature: str,
    minimum_match_rate: float,
) -> dict[str, Any]:
    """处理行业代码或行业诊断，并保留 UNKNOWN 口径。
    
    输入：
        sample_index_path: Path。
        cne5s_path: Path。
        style_store_signature: str。
        minimum_match_rate: float。
    输出：
        dict[str, Any]。
        """
    # 将 `{'schema_version': INDUSTRY_SIDECAR_SCHEMA_VERSION, 'sample_index': _file_fingerprint(sample_in…` 作为当前函数结果返回给调用方。
    return {
        "schema_version": INDUSTRY_SIDECAR_SCHEMA_VERSION,
        "sample_index": _file_fingerprint(sample_index_path),
        "cne5s": barra._fingerprint_parquet(cne5s_path),
        "style_store_signature": style_store_signature,
        "industry_column": barra.INDUSTRY_COLUMN,
        "unknown_industry": barra.UNKNOWN_INDUSTRY,
        "encoding": "unknown_zero_known_labels_first_seen_year_then_sorted",
        "minimum_match_rate": minimum_match_rate,
        "match_policy": "exact_trade_date_and_assetid_no_forward_fill",
    }






# 定义 `build_industry_sidecar` 函数，封装这一阶段可独立调用的处理步骤。
def build_industry_sidecar(
    *,
    store_dir: str | Path = DEFAULT_INDUSTRY_STORE_DIR,
    sample_index_path: str | Path = base.SAMPLE_INDEX_PATH,
    cne5s_path: str | Path = barra.DEFAULT_CNE5S_PATH,
    minimum_match_rate: float = MINIMUM_MATCH_RATE,
    overwrite: bool = False,
) -> dict[str, Any]:
    """构建与 sample_index 严格对齐的 CNE5S 行业整数代码。
    
    输入：
        store_dir: str | Path。
        sample_index_path: str | Path。
        cne5s_path: str | Path。
        minimum_match_rate: float。
        overwrite: bool。
    输出：
        dict[str, Any]。
        """

    # 检查 `not 0 <= minimum_match_rate <= 1`；根据条件结果选择对应处理分支。
    if not 0 <= minimum_match_rate <= 1:
        # 抛出 `ValueError('minimum_match_rate 必须位于 [0,1]')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("minimum_match_rate 必须位于 [0,1]")
    # 调用 `Path(store_dir).resolve` 规范化文件路径，并将结果保存到 `target`。
    target = Path(store_dir).resolve()
    # 调用 `Path(sample_index_path).resolve` 规范化文件路径，并将结果保存到 `sample_path`。
    sample_path = Path(sample_index_path).resolve()
    # 调用 `Path(cne5s_path).resolve` 规范化文件路径，并将结果保存到 `cne5s`。
    cne5s = Path(cne5s_path).resolve()
    # 调用 `require_style_neutral_store` 校验输入、配置或产物是否满足当前阶段的数据契约，并将结果保存到 `style_manifest`。
    style_manifest = require_style_neutral_store()
    # 调用 `_industry_sidecar_payload` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `payload`。
    payload = _industry_sidecar_payload(
        sample_path, cne5s, style_manifest["signature"], minimum_match_rate
    )
    # 调用 `_stable_hash` 对关键配置生成稳定签名以支持复现和一致性检查，并将结果保存到 `signature`。
    signature = _stable_hash(payload)
    # 计算 `target / INDUSTRY_MANIFEST_FILE`，并将数值结果保存到 `manifest_path`。
    manifest_path = target / INDUSTRY_MANIFEST_FILE
    # 检查 `manifest_path.exists() and (not overwrite)`；根据条件结果选择对应处理分支。
    if manifest_path.exists() and not overwrite:
        # 调用 `json.loads` 读取源数据，并将结果保存到 `existing`。
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        # 检查 `existing.get('signature') != signature`；根据条件结果选择对应处理分支。
        if existing.get("signature") != signature:
            # 抛出 `RuntimeError('已有行业 sidecar 签名不同；请显式设置 overwrite=True')`，阻止不满足数据契约的流程继续执行。
            raise RuntimeError(
                "已有行业 sidecar 签名不同；请显式设置 overwrite=True"
            )
        # 调用 `_validate_industry_sidecar` 校验输入、配置或产物是否满足当前阶段的数据契约。
        _validate_industry_sidecar(target, existing)
        # 将 `{**existing, 'reused': True}` 作为当前函数结果返回给调用方。
        return {**existing, "reused": True}

    # 调用 `pd.read_parquet` 读取源数据，并将结果保存到 `sample`。
    sample = pd.read_parquet(
        sample_path,
        columns=["sample_index", "trade_date", "instrument_id"],
    )
    # 调用 `sample['trade_date'].astype` 统一设备或数据类型，并将结果保存到 `sample['trade_date']`。
    sample["trade_date"] = sample["trade_date"].astype(str)
    # 调用 `sample['instrument_id'].astype` 统一设备或数据类型，并将结果保存到 `sample['instrument_id']`。
    sample["instrument_id"] = sample["instrument_id"].astype(str)
    # 调用 `np.arange` 按目标形状初始化数值容器 `expected_indices`。
    expected_indices = np.arange(len(sample), dtype=np.int64)
    # 检查 `not np.array_equal(sample['sample_index'].to_numpy(np.int64), expected_indices)`；根据条件结果选择对应处理分支。
    if not np.array_equal(
        sample["sample_index"].to_numpy(np.int64), expected_indices
    ):
        # 抛出 `AssertionError('sample_index 必须与物理行号逐行一致')`，阻止不满足数据契约的流程继续执行。
        raise AssertionError("sample_index 必须与物理行号逐行一致")

    # 调用 `target.parent.mkdir` 创建输出目录，确保后续文件可以安全写入。
    target.parent.mkdir(parents=True, exist_ok=True)
    # 调用 `Path` 规范化文件路径，并将结果保存到 `stage`。
    stage = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.stage.", dir=target.parent)
    )
    # 尝试执行本段主逻辑；若发生异常，则交由后续异常分支处理。
    try:
        # 调用 `np.lib.format.open_memmap` 以 NPY 格式打开内存映射数组，避免一次载入全部样本，并将结果保存到 `output`。
        output = np.lib.format.open_memmap(
            stage / INDUSTRY_CODES_FILE,
            mode="w+",
            dtype=np.int16,
            shape=(len(sample),),
        )
        # 计算 `-1`，并将数值结果保存到 `output[:]`。
        output[:] = -1
        # 构造键值映射，并保存到 `mapping`，供后续按键查询。
        mapping: dict[str, int] = {barra.UNKNOWN_INDUSTRY: 0}
        # 构造当前步骤所需的列表，并保存到 `audit`。
        audit: list[dict[str, Any]] = []
        # 调用 `sorted` 整理数据结构，并将结果保存到 `years`。
        years = sorted(sample["trade_date"].str[:4].astype(int).unique())
        # 遍历 `years`，并把当前元素绑定到 `year` 逐项处理。
        for year in years:
            # 复制 `sample.loc[sample['trade_date'].str.startswith(str(year))].copy`，避免后续更新污染原对象，并保存到 `yearly`。
            yearly = sample.loc[
                sample["trade_date"].str.startswith(str(year))
            ].copy()
            # 调用 `barra.instrument_ids_to_assetids` 执行 `instrument_ids_to_assetids` 对应的明确变换，并产生后续步骤使用的结果，并将结果保存到 `yearly['assetid']`。
            yearly["assetid"] = barra.instrument_ids_to_assetids(
                yearly["instrument_id"]
            )
            # 调用 `barra._load_cne5s_year` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `cne5s_year`。
            cne5s_year = barra._load_cne5s_year(cne5s, year)
            # 调用 `barra._join_prediction_and_cne5s` 执行模型推理并生成样本预测分数，并将结果保存到 `joined`。
            joined = barra._join_prediction_and_cne5s(
                yearly, cne5s_year, year
            )
            # 调用 `joined['barra_matched'].to_numpy` 按股票和交易日键对齐来自不同来源的数据，并将结果保存到 `matched`。
            matched = joined["barra_matched"].to_numpy(bool)
            # 调用 `float` 统一设备或数据类型，并将结果保存到 `match_rate`。
            match_rate = float(matched.mean())
            # 检查 `match_rate < minimum_match_rate`；根据条件结果选择对应处理分支。
            if match_rate < minimum_match_rate:
                # 抛出 `AssertionError(f'{year} CNE5S 匹配率 {match_rate:.6%} < {minimum_match_rate:.6%}')`，阻止不满足数据契约的流程继续执行。
                raise AssertionError(
                    f"{year} CNE5S 匹配率 {match_rate:.6%} "
                    f"< {minimum_match_rate:.6%}"
                )
            # 复制 `joined[barra.INDUSTRY_COLUMN].astype('object').copy`，避免后续更新污染原对象，并保存到 `industries`。
            industries = joined[barra.INDUSTRY_COLUMN].astype("object").copy()
            # 读取 `barra.UNKNOWN_INDUSTRY` 的当前值，并赋给 `industries.loc[~joined['barra_matched']]`。
            industries.loc[~joined["barra_matched"]] = barra.UNKNOWN_INDUSTRY
            # 检查 `industries.isna().any()`；根据条件结果选择对应处理分支。
            if industries.isna().any():
                # 抛出 `ValueError(f'{year} 行业字段含未处理缺失值')`，阻止不满足数据契约的流程继续执行。
                raise ValueError(f"{year} 行业字段含未处理缺失值")
            # 调用 `industries.astype` 统一设备或数据类型，并将结果保存到 `labels`。
            labels = industries.astype(str)
            # 调用 `sorted` 整理数据结构，并将结果保存到 `new_labels`。
            new_labels = sorted(set(labels.unique()).difference(mapping))
            # 遍历 `new_labels`，并把当前元素绑定到 `label` 逐项处理。
            for label in new_labels:
                # 调用 `len` 汇总当前数据，并将统计量保存到 `mapping[label]`。
                mapping[label] = len(mapping)
            # 调用 `labels.map` 按映射关系转换当前序列中的每个元素，并将结果保存到 `encoded`。
            encoded = labels.map(mapping)
            # 检查 `encoded.isna().any()`；根据条件结果选择对应处理分支。
            if encoded.isna().any():
                # 抛出 `AssertionError(f'{year} 存在未编码行业')`，阻止不满足数据契约的流程继续执行。
                raise AssertionError(f"{year} 存在未编码行业")
            # 调用 `joined['sample_index'].to_numpy` 按股票和交易日键对齐来自不同来源的数据，并将结果保存到 `sample_indices`。
            sample_indices = joined["sample_index"].to_numpy(np.int64)
            # 调用 `encoded.to_numpy` 把表格列转换为指定精度的 NumPy 数组，并将结果保存到 `output[sample_indices]`。
            output[sample_indices] = encoded.to_numpy(np.int16)
            # 把本轮结果追加到 `audit`，累积后续需要汇总的记录。
            audit.append(
                {
                    "year": int(year),
                    "rows": len(joined),
                    "matched_rows": int(matched.sum()),
                    "match_rate": match_rate,
                    "known_industries_to_date": len(mapping) - 1,
                }
            )

        # 调用 `output.flush` 把内存映射中的缓存修改刷新到磁盘。
        output.flush()
        # 删除 `output`，释放不再使用的引用。
        del output
        # 调用 `np.load` 读取源数据，并将结果保存到 `codes`。
        codes = np.load(stage / INDUSTRY_CODES_FILE, mmap_mode="r")
        # 检查 `(codes < 0).any()`；根据条件结果选择对应处理分支。
        if (codes < 0).any():
            # 抛出 `AssertionError('行业 sidecar 存在未写入样本')`，阻止不满足数据契约的流程继续执行。
            raise AssertionError("行业 sidecar 存在未写入样本")
        # 计算 `len(mapping) - 1`，并将数值结果保存到 `known_count`。
        known_count = len(mapping) - 1
        # 检查 `known_count != EXPECTED_CNE5S_INDUSTRIES`；根据条件结果选择对应处理分支。
        if known_count != EXPECTED_CNE5S_INDUSTRIES:
            # 抛出 `ValueError(f'CNE5S 行业数为 {known_count}，预期 {EXPECTED_CNE5S_INDUSTRIES}')`，阻止不满足数据契约的流程继续执行。
            raise ValueError(
                f"CNE5S 行业数为 {known_count}，预期 {EXPECTED_CNE5S_INDUSTRIES}"
            )
        # 调用 `pd.DataFrame(audit).to_parquet` 将当前结果持久化到目标文件。
        pd.DataFrame(audit).to_parquet(
            stage / INDUSTRY_AUDIT_FILE, index=False, compression="zstd"
        )
        # 构造键值映射，并保存到 `manifest`，供后续按键查询。
        manifest = {
            "signature": signature,
            "payload": payload,
            "sample_count": len(sample),
            "industry_mapping": mapping,
            "known_industry_count": known_count,
            "files": {
                "industry_codes": INDUSTRY_CODES_FILE,
                "match_audit": INDUSTRY_AUDIT_FILE,
            },
            "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        }
        # 调用 `(stage / INDUSTRY_MANIFEST_FILE).write_text` 将当前结果持久化到目标文件。
        (stage / INDUSTRY_MANIFEST_FILE).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        # 调用 `_validate_industry_sidecar` 校验输入、配置或产物是否满足当前阶段的数据契约。
        _validate_industry_sidecar(stage, manifest)
        # 调用 `_atomic_replace_directory` 先写临时文件再原子替换正式产物，避免留下半成品。
        _atomic_replace_directory(stage, target)
        # 将 `{**manifest, 'reused': False}` 作为当前函数结果返回给调用方。
        return {**manifest, "reused": False}
    # 捕获 `except BaseException` 对应的异常，并执行明确的降级、记录或清理逻辑。
    except BaseException:
        # 检查 `stage.exists()`；根据条件结果选择对应处理分支。
        if stage.exists():
            # 调用 `shutil.rmtree` 递归删除明确指定的旧输出目录。
            shutil.rmtree(stage)
        # 抛出 `空值`，阻止不满足数据契约的流程继续执行。
        raise


# 定义内部辅助函数 `_full_barra_target_payload`，封装仅供本模块复用的计算步骤。
def _full_barra_target_payload(
    targets_path: Path,
    style_store_signature: str,
    industry_sidecar_signature: str,
) -> dict[str, Any]:
    """构造或校验完整 Barra 风格加行业的数据契约。
    
    输入：
        targets_path: Path。
        style_store_signature: str。
        industry_sidecar_signature: str。
    输出：
        dict[str, Any]。
        """
    # 将 `{'schema_version': FULL_BARRA_TARGET_SCHEMA_VERSION, 'source_targets': _file_fingerprint(target…` 作为当前函数结果返回给调用方。
    return {
        "schema_version": FULL_BARRA_TARGET_SCHEMA_VERSION,
        "source_targets": _file_fingerprint(targets_path),
        "style_store_signature": style_store_signature,
        "industry_sidecar_signature": industry_sidecar_signature,
        "regression": (
            "daily_equal_weight_ols_intercept_plus_10_styles_plus_"
            "industry_fixed_effects"
        ),
        "solver": "industry_within_then_numpy_lstsq",
        "standardization": "residual_zscore_ddof0",
        "unknown_industry_policy": "code_zero_as_separate_fixed_effect_group",
        "missing_style_policy": (
            "daily_standardized_exposure_zero_fill_with_valid_mask_"
            "no_forward_fill"
        ),
    }






# 定义 `build_full_barra_target_sidecar` 函数，封装这一阶段可独立调用的处理步骤。
def build_full_barra_target_sidecar(
    *,
    store_dir: str | Path = DEFAULT_FULL_BARRA_TARGET_STORE_DIR,
    sample_index_path: str | Path = base.SAMPLE_INDEX_PATH,
    targets_path: str | Path = base.TARGETS_PATH,
    overwrite: bool = False,
) -> dict[str, Any]:
    """逐日生成截距、十风格与行业固定效应联合残差标签。
    
    输入：
        store_dir: str | Path。
        sample_index_path: str | Path。
        targets_path: str | Path。
        overwrite: bool。
    输出：
        dict[str, Any]。
        """

    # 调用 `Path(store_dir).resolve` 规范化文件路径，并将结果保存到 `target`。
    target = Path(store_dir).resolve()
    # 调用 `Path(sample_index_path).resolve` 规范化文件路径，并将结果保存到 `sample_path`。
    sample_path = Path(sample_index_path).resolve()
    # 调用 `Path(targets_path).resolve` 规范化文件路径，并将结果保存到 `source_targets_path`。
    source_targets_path = Path(targets_path).resolve()
    # 调用 `require_style_neutral_store` 校验输入、配置或产物是否满足当前阶段的数据契约，并将结果保存到 `style_manifest`。
    style_manifest = require_style_neutral_store()
    # 调用 `require_industry_sidecar` 校验输入、配置或产物是否满足当前阶段的数据契约，并将结果保存到 `industry_manifest`。
    industry_manifest = require_industry_sidecar()
    # 调用 `_full_barra_target_payload` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `payload`。
    payload = _full_barra_target_payload(
        source_targets_path,
        style_manifest["signature"],
        industry_manifest["signature"],
    )
    # 调用 `_stable_hash` 对关键配置生成稳定签名以支持复现和一致性检查，并将结果保存到 `signature`。
    signature = _stable_hash(payload)
    # 计算 `target / FULL_BARRA_TARGET_MANIFEST_FILE`，并将数值结果保存到 `manifest_path`。
    manifest_path = target / FULL_BARRA_TARGET_MANIFEST_FILE
    # 检查 `manifest_path.exists() and (not overwrite)`；根据条件结果选择对应处理分支。
    if manifest_path.exists() and not overwrite:
        # 调用 `json.loads` 读取源数据，并将结果保存到 `existing`。
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        # 检查 `existing.get('signature') != signature`；根据条件结果选择对应处理分支。
        if existing.get("signature") != signature:
            # 抛出 `RuntimeError('已有完整 Barra 残差标签签名不同；请显式设置 overwrite=True')`，阻止不满足数据契约的流程继续执行。
            raise RuntimeError(
                "已有完整 Barra 残差标签签名不同；"
                "请显式设置 overwrite=True"
            )
        # 调用 `_validate_full_barra_target_sidecar` 校验输入、配置或产物是否满足当前阶段的数据契约。
        _validate_full_barra_target_sidecar(target, existing)
        # 将 `{**existing, 'reused': True}` 作为当前函数结果返回给调用方。
        return {**existing, "reused": True}

    # 调用 `pd.read_parquet` 读取源数据，并将结果保存到 `sample`。
    sample = pd.read_parquet(
        sample_path,
        columns=["sample_index", "trade_date", "label_valid"],
    )
    # 调用 `sample['trade_date'].astype` 统一设备或数据类型，并将结果保存到 `sample['trade_date']`。
    sample["trade_date"] = sample["trade_date"].astype(str)
    # 调用 `np.arange` 按目标形状初始化数值容器 `expected_indices`。
    expected_indices = np.arange(len(sample), dtype=np.int64)
    # 检查 `not np.array_equal(sample['sample_index'].to_numpy(np.int64), expected_indices)`；根据条件结果选择对应处理分支。
    if not np.array_equal(
        sample["sample_index"].to_numpy(np.int64), expected_indices
    ):
        # 抛出 `AssertionError('sample_index 必须与物理行号逐行一致')`，阻止不满足数据契约的流程继续执行。
        raise AssertionError("sample_index 必须与物理行号逐行一致")
    # 调用 `sample['label_valid'].to_numpy` 把表格列转换为指定精度的 NumPy 数组，并将结果保存到 `label_valid`。
    label_valid = sample["label_valid"].to_numpy(bool)
    # 调用 `np.load` 读取源数据，并将结果保存到 `source_targets`。
    source_targets = np.load(source_targets_path, mmap_mode="r")
    # 调用 `np.load` 读取源数据，并将结果保存到 `styles`。
    styles = np.load(
        DEFAULT_STORE_DIR / STYLE_EXPOSURES_FILE, mmap_mode="r"
    )
    # 调用 `np.load` 读取源数据，并将结果保存到 `industry_codes`。
    industry_codes = np.load(
        DEFAULT_INDUSTRY_STORE_DIR / INDUSTRY_CODES_FILE, mmap_mode="r"
    )
    # 检查 `source_targets.shape != (len(sample),)`；根据条件结果选择对应处理分支。
    if source_targets.shape != (len(sample),):
        # 抛出 `ValueError('源 targets.npy 与 sample_index 行数不一致')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("源 targets.npy 与 sample_index 行数不一致")
    # 检查 `styles.shape != (len(sample), len(STYLE_COLUMNS))`；根据条件结果选择对应处理分支。
    if styles.shape != (len(sample), len(STYLE_COLUMNS)):
        # 抛出 `ValueError('十风格矩阵与 sample_index 行数不一致')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("十风格矩阵与 sample_index 行数不一致")
    # 检查 `industry_codes.shape != (len(sample),)`；根据条件结果选择对应处理分支。
    if industry_codes.shape != (len(sample),):
        # 抛出 `ValueError('行业代码与 sample_index 行数不一致')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("行业代码与 sample_index 行数不一致")
    # 检查 `not np.array_equal(np.isfinite(source_targets), label_valid)`；根据条件结果选择对应处理分支。
    if not np.array_equal(np.isfinite(source_targets), label_valid):
        # 抛出 `AssertionError('源标签有限性与 label_valid 不一致')`，阻止不满足数据契约的流程继续执行。
        raise AssertionError("源标签有限性与 label_valid 不一致")

    # 调用 `target.parent.mkdir` 创建输出目录，确保后续文件可以安全写入。
    target.parent.mkdir(parents=True, exist_ok=True)
    # 调用 `Path` 规范化文件路径，并将结果保存到 `stage`。
    stage = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.stage.", dir=target.parent)
    )
    # 尝试执行本段主逻辑；若发生异常，则交由后续异常分支处理。
    try:
        # 调用 `np.lib.format.open_memmap` 以 NPY 格式打开内存映射数组，避免一次载入全部样本，并将结果保存到 `output`。
        output = np.lib.format.open_memmap(
            stage / FULL_BARRA_TARGETS_FILE,
            mode="w+",
            dtype=np.float32,
            shape=(len(sample),),
        )
        # 读取 `np.nan` 的当前值，并赋给 `output[:]`。
        output[:] = np.nan
        # 构造当前步骤所需的列表，并保存到 `diagnostics`。
        diagnostics: list[dict[str, Any]] = []
        # 调用 `sorted` 整理数据结构，并将结果保存到 `years`。
        years = sorted(sample["trade_date"].str[:4].astype(int).unique())
        # 遍历 `years`，并把当前元素绑定到 `year` 逐项处理。
        for year in years:
            # 读取 `sample.loc[sample['trade_date'].str.startswith(str(year))]` 的当前值，并赋给 `yearly`。
            yearly = sample.loc[
                sample["trade_date"].str.startswith(str(year))
            ]
            # 遍历 `yearly.groupby('trade_date', sort=True).groups.items()`，并把当前元素绑定到 `(trade_date, positions)` 逐项处理。
            for trade_date, positions in yearly.groupby(
                "trade_date", sort=True
            ).groups.items():
                # 读取 `sample.loc[positions]` 的当前值，并赋给 `daily`。
                daily = sample.loc[positions]
                # 调用 `daily['sample_index'].to_numpy` 把表格列转换为指定精度的 NumPy 数组，并将结果保存到 `indices`。
                indices = daily["sample_index"].to_numpy(np.int64)
                # 调用 `residualize_daily_targets_with_industry` 按 Barra 暴露执行截面投影或残差化处理，并将结果保存到 `(residual, record)`。
                residual, record = residualize_daily_targets_with_industry(
                    np.asarray(source_targets[indices], dtype=np.float64),
                    np.asarray(styles[indices], dtype=np.float64),
                    np.asarray(industry_codes[indices], dtype=np.int16),
                    label_valid[indices],
                )
                # 调用 `residual.astype` 统一设备或数据类型，并将结果保存到 `output[indices]`。
                output[indices] = residual.astype(np.float32)
                # 把本轮结果追加到 `diagnostics`，累积后续需要汇总的记录。
                diagnostics.append(
                    {
                        "year": int(year),
                        "trade_date": str(trade_date),
                        **record,
                    }
                )
            # 调用 `print` 记录当前流程状态或诊断信息。
            print(
                f"{year}：写入完整 Barra 残差标签 {len(yearly):,} 行",
                flush=True,
            )

        # 调用 `output.flush` 把内存映射中的缓存修改刷新到磁盘。
        output.flush()
        # 删除 `output`，释放不再使用的引用。
        del output
        # 调用 `np.load` 读取源数据，并将结果保存到 `residual_check`。
        residual_check = np.load(
            stage / FULL_BARRA_TARGETS_FILE, mmap_mode="r"
        )
        # 检查 `not np.array_equal(np.isfinite(residual_check), label_valid)`；根据条件结果选择对应处理分支。
        if not np.array_equal(np.isfinite(residual_check), label_valid):
            # 抛出 `AssertionError('完整 Barra 残差标签有限性与 label_valid 不一致')`，阻止不满足数据契约的流程继续执行。
            raise AssertionError(
                "完整 Barra 残差标签有限性与 label_valid 不一致"
            )
        # 调用 `pd.DataFrame` 转换数据表示，并将标准化对象保存到 `daily`。
        daily = pd.DataFrame(diagnostics)
        # 调用 `daily.to_parquet` 将当前结果持久化到目标文件。
        daily.to_parquet(
            stage / FULL_BARRA_DAILY_DIAGNOSTICS_FILE,
            index=False,
            compression="zstd",
        )
        # 调用 `daily.groupby('year', as_index=False).agg` 按预先声明的聚合规则汇总分组数据，并将结果保存到 `annual`。
        annual = daily.groupby("year", as_index=False).agg(
            rows=("rows", "sum"),
            label_valid_rows=("label_valid_rows", "sum"),
            dates=("trade_date", "nunique"),
            mean_industry_count=("industry_count", "mean"),
            mean_unknown_industry_rate=("unknown_industry_rate", "mean"),
            mean_joint_barra_r_squared=("joint_barra_r_squared", "mean"),
            max_abs_style_residual_pearson=(
                "max_abs_style_residual_pearson", "max"
            ),
            max_abs_industry_residual_mean=(
                "max_abs_industry_residual_mean", "max"
            ),
            max_condition_number=("condition_number", "max"),
        )
        # 调用 `annual.to_parquet` 将当前结果持久化到目标文件。
        annual.to_parquet(
            stage / FULL_BARRA_ANNUAL_SUMMARY_FILE,
            index=False,
            compression="zstd",
        )
        # 构造键值映射，并保存到 `manifest`，供后续按键查询。
        manifest = {
            "signature": signature,
            "payload": payload,
            "sample_count": len(sample),
            "label_valid_count": int(label_valid.sum()),
            "first_date": str(sample["trade_date"].min()),
            "last_date": str(sample["trade_date"].max()),
            "files": {
                "residual_targets": FULL_BARRA_TARGETS_FILE,
                "daily_diagnostics": FULL_BARRA_DAILY_DIAGNOSTICS_FILE,
                "annual_summary": FULL_BARRA_ANNUAL_SUMMARY_FILE,
            },
            "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        }
        # 调用 `(stage / FULL_BARRA_TARGET_MANIFEST_FILE).write_text` 将当前结果持久化到目标文件。
        (stage / FULL_BARRA_TARGET_MANIFEST_FILE).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        # 调用 `_validate_full_barra_target_sidecar` 校验输入、配置或产物是否满足当前阶段的数据契约。
        _validate_full_barra_target_sidecar(stage, manifest)
        # 调用 `_atomic_replace_directory` 先写临时文件再原子替换正式产物，避免留下半成品。
        _atomic_replace_directory(stage, target)
        # 将 `{**manifest, 'reused': False}` 作为当前函数结果返回给调用方。
        return {**manifest, "reused": False}
    # 捕获 `except BaseException` 对应的异常，并执行明确的降级、记录或清理逻辑。
    except BaseException:
        # 检查 `stage.exists()`；根据条件结果选择对应处理分支。
        if stage.exists():
            # 调用 `shutil.rmtree` 递归删除明确指定的旧输出目录。
            shutil.rmtree(stage)
        # 抛出 `空值`，阻止不满足数据契约的流程继续执行。
        raise


# 定义内部辅助函数 `_dual_tower_target_payload`，封装仅供本模块复用的计算步骤。
def _dual_tower_target_payload(
    targets_path: Path,
    full_target_signature: str,
    style_store_signature: str,
    industry_sidecar_signature: str,
) -> dict[str, Any]:
    """构造或校验 Alpha/Style 双塔所需的数据契约。
    
    输入：
        targets_path: Path。
        full_target_signature: str。
        style_store_signature: str。
        industry_sidecar_signature: str。
    输出：
        dict[str, Any]。
        """
    # 将 `{'schema_version': DUAL_TOWER_TARGET_SCHEMA_VERSION, 'source_targets': _file_fingerprint(target…` 作为当前函数结果返回给调用方。
    return {
        "schema_version": DUAL_TOWER_TARGET_SCHEMA_VERSION,
        "source_targets": _file_fingerprint(targets_path),
        "full_barra_target_signature": full_target_signature,
        "style_store_signature": style_store_signature,
        "industry_sidecar_signature": industry_sidecar_signature,
        "regression": (
            "daily_equal_weight_ols_intercept_plus_10_styles_plus_"
            "industry_fixed_effects"
        ),
        "solver": "industry_within_then_numpy_lstsq",
        "target": "daily_full_barra_fitted_component_zscore_ddof0",
        "unknown_industry_policy": "code_zero_as_separate_fixed_effect_group",
        "missing_style_policy": (
            "daily_standardized_exposure_zero_fill_with_valid_mask_"
            "no_forward_fill"
        ),
    }






# 定义 `build_dual_tower_target_sidecar` 函数，封装这一阶段可独立调用的处理步骤。
def build_dual_tower_target_sidecar(
    *,
    store_dir: str | Path = DEFAULT_DUAL_TOWER_TARGET_STORE_DIR,
    sample_index_path: str | Path = base.SAMPLE_INDEX_PATH,
    targets_path: str | Path = base.TARGETS_PATH,
    overwrite: bool = False,
) -> dict[str, Any]:
    """逐日生成完整 Barra 回归拟合收益，供 No.6 Style 塔监督。
    
    输入：
        store_dir: str | Path。
        sample_index_path: str | Path。
        targets_path: str | Path。
        overwrite: bool。
    输出：
        dict[str, Any]。
        """

    # 调用 `Path(store_dir).resolve` 规范化文件路径，并将结果保存到 `target`。
    target = Path(store_dir).resolve()
    # 调用 `Path(sample_index_path).resolve` 规范化文件路径，并将结果保存到 `sample_path`。
    sample_path = Path(sample_index_path).resolve()
    # 调用 `Path(targets_path).resolve` 规范化文件路径，并将结果保存到 `source_targets_path`。
    source_targets_path = Path(targets_path).resolve()
    # 调用 `require_style_neutral_store` 校验输入、配置或产物是否满足当前阶段的数据契约，并将结果保存到 `style_manifest`。
    style_manifest = require_style_neutral_store()
    # 调用 `require_industry_sidecar` 校验输入、配置或产物是否满足当前阶段的数据契约，并将结果保存到 `industry_manifest`。
    industry_manifest = require_industry_sidecar()
    # 调用 `require_full_barra_target_sidecar` 校验输入、配置或产物是否满足当前阶段的数据契约，并将结果保存到 `full_manifest`。
    full_manifest = require_full_barra_target_sidecar()
    # 调用 `_dual_tower_target_payload` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `payload`。
    payload = _dual_tower_target_payload(
        source_targets_path,
        full_manifest["signature"],
        style_manifest["signature"],
        industry_manifest["signature"],
    )
    # 调用 `_stable_hash` 对关键配置生成稳定签名以支持复现和一致性检查，并将结果保存到 `signature`。
    signature = _stable_hash(payload)
    # 计算 `target / DUAL_TOWER_TARGET_MANIFEST_FILE`，并将数值结果保存到 `manifest_path`。
    manifest_path = target / DUAL_TOWER_TARGET_MANIFEST_FILE
    # 检查 `manifest_path.exists() and (not overwrite)`；根据条件结果选择对应处理分支。
    if manifest_path.exists() and not overwrite:
        # 调用 `json.loads` 读取源数据，并将结果保存到 `existing`。
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        # 检查 `existing.get('signature') != signature`；根据条件结果选择对应处理分支。
        if existing.get("signature") != signature:
            # 抛出 `RuntimeError('已有 No.6 fitted-target sidecar 签名不同；请显式设置 overwrite=True')`，阻止不满足数据契约的流程继续执行。
            raise RuntimeError(
                "已有 No.6 fitted-target sidecar 签名不同；"
                "请显式设置 overwrite=True"
            )
        # 调用 `_validate_dual_tower_target_sidecar` 校验输入、配置或产物是否满足当前阶段的数据契约。
        _validate_dual_tower_target_sidecar(target, existing)
        # 将 `{**existing, 'reused': True}` 作为当前函数结果返回给调用方。
        return {**existing, "reused": True}

    # 调用 `pd.read_parquet` 读取源数据，并将结果保存到 `sample`。
    sample = pd.read_parquet(
        sample_path, columns=["sample_index", "trade_date", "label_valid"]
    )
    # 调用 `sample['trade_date'].astype` 统一设备或数据类型，并将结果保存到 `sample['trade_date']`。
    sample["trade_date"] = sample["trade_date"].astype(str)
    # 调用 `np.arange` 按目标形状初始化数值容器 `expected_indices`。
    expected_indices = np.arange(len(sample), dtype=np.int64)
    # 检查 `not np.array_equal(sample['sample_index'].to_numpy(np.int64), expected_indices)`；根据条件结果选择对应处理分支。
    if not np.array_equal(
        sample["sample_index"].to_numpy(np.int64), expected_indices
    ):
        # 抛出 `AssertionError('sample_index 必须与物理行号逐行一致')`，阻止不满足数据契约的流程继续执行。
        raise AssertionError("sample_index 必须与物理行号逐行一致")
    # 调用 `sample['label_valid'].to_numpy` 把表格列转换为指定精度的 NumPy 数组，并将结果保存到 `label_valid`。
    label_valid = sample["label_valid"].to_numpy(bool)
    # 调用 `np.load` 读取源数据，并将结果保存到 `source_targets`。
    source_targets = np.load(source_targets_path, mmap_mode="r")
    # 调用 `np.load` 读取源数据，并将结果保存到 `styles`。
    styles = np.load(DEFAULT_STORE_DIR / STYLE_EXPOSURES_FILE, mmap_mode="r")
    # 调用 `np.load` 读取源数据，并将结果保存到 `industries`。
    industries = np.load(
        DEFAULT_INDUSTRY_STORE_DIR / INDUSTRY_CODES_FILE, mmap_mode="r"
    )
    # 调用 `np.load` 读取源数据，并将结果保存到 `existing_residual`。
    existing_residual = np.load(
        DEFAULT_FULL_BARRA_TARGET_STORE_DIR / FULL_BARRA_TARGETS_FILE,
        mmap_mode="r",
    )
    # 组织一组有序值，并保存到 `expected_shapes`。
    expected_shapes = (
        source_targets.shape == (len(sample),),
        styles.shape == (len(sample), len(STYLE_COLUMNS)),
        industries.shape == (len(sample),),
        existing_residual.shape == (len(sample),),
    )
    # 检查 `not all(expected_shapes)`；根据条件结果选择对应处理分支。
    if not all(expected_shapes):
        # 抛出 `ValueError('No.6 标签输入与 sample_index 形状不一致')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("No.6 标签输入与 sample_index 形状不一致")

    # 调用 `target.parent.mkdir` 创建输出目录，确保后续文件可以安全写入。
    target.parent.mkdir(parents=True, exist_ok=True)
    # 调用 `Path` 规范化文件路径，并将结果保存到 `stage`。
    stage = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.stage.", dir=target.parent)
    )
    # 尝试执行本段主逻辑；若发生异常，则交由后续异常分支处理。
    try:
        # 调用 `np.lib.format.open_memmap` 以 NPY 格式打开内存映射数组，避免一次载入全部样本，并将结果保存到 `output`。
        output = np.lib.format.open_memmap(
            stage / DUAL_TOWER_FITTED_TARGETS_FILE,
            mode="w+",
            dtype=np.float32,
            shape=(len(sample),),
        )
        # 读取 `np.nan` 的当前值，并赋给 `output[:]`。
        output[:] = np.nan
        # 构造当前步骤所需的列表，并保存到 `diagnostics`。
        diagnostics: list[dict[str, Any]] = []
        # 调用 `sorted` 整理数据结构，并将结果保存到 `years`。
        years = sorted(sample["trade_date"].str[:4].astype(int).unique())
        # 遍历 `years`，并把当前元素绑定到 `year` 逐项处理。
        for year in years:
            # 读取 `sample.loc[sample['trade_date'].str.startswith(str(year))]` 的当前值，并赋给 `yearly`。
            yearly = sample.loc[sample["trade_date"].str.startswith(str(year))]
            # 遍历 `yearly.groupby('trade_date', sort=True).groups.items()`，并把当前元素绑定到 `(trade_date, positions)` 逐项处理。
            for trade_date, positions in yearly.groupby(
                "trade_date", sort=True
            ).groups.items():
                # 调用 `sample.loc[positions, 'sample_index'].to_numpy` 把表格列转换为指定精度的 NumPy 数组，并将结果保存到 `indices`。
                indices = sample.loc[positions, "sample_index"].to_numpy(np.int64)
                # 调用 `decompose_daily_targets_with_industry` 执行 `decompose_daily_targets_with_industry` 对应的明确变换，并产生后续步骤使用的结果，并将结果保存到 `(residual, fitted, record)`。
                residual, fitted, record = decompose_daily_targets_with_industry(
                    np.asarray(source_targets[indices], dtype=np.float64),
                    np.asarray(styles[indices], dtype=np.float64),
                    np.asarray(industries[indices], dtype=np.int16),
                    label_valid[indices],
                )
                # 读取 `label_valid[indices]` 的当前值，并赋给 `valid`。
                valid = label_valid[indices]
                # 调用 `float` 统一设备或数据类型，并将结果保存到 `residual_error`。
                residual_error = float(
                    np.max(
                        np.abs(
                            residual[valid]
                            - np.asarray(existing_residual[indices][valid], dtype=np.float64)
                        )
                    )
                )
                # 调用 `joint_barra_r_squared_numpy` 按股票和交易日键对齐来自不同来源的数据，并将结果保存到 `(fitted_r_squared, _)`。
                fitted_r_squared, _ = joint_barra_r_squared_numpy(
                    fitted[valid],
                    np.asarray(styles[indices][valid], dtype=np.float64),
                    np.asarray(industries[indices][valid], dtype=np.int16),
                )
                # 调用 `fitted.astype` 统一设备或数据类型，并将结果保存到 `output[indices]`。
                output[indices] = fitted.astype(np.float32)
                # 把本轮结果追加到 `diagnostics`，累积后续需要汇总的记录。
                diagnostics.append(
                    {
                        "year": int(year),
                        "trade_date": str(trade_date),
                        **record,
                        "residual_sidecar_max_abs_error": residual_error,
                        "fitted_joint_barra_r_squared": fitted_r_squared,
                    }
                )
            # 调用 `print` 记录当前流程状态或诊断信息。
            print(
                f"{year}：写入 No.6 fitted target {len(yearly):,} 行",
                flush=True,
            )
        # 调用 `output.flush` 把内存映射中的缓存修改刷新到磁盘。
        output.flush()
        # 删除 `output`，释放不再使用的引用。
        del output
        # 调用 `np.load` 读取源数据，并将结果保存到 `fitted_check`。
        fitted_check = np.load(
            stage / DUAL_TOWER_FITTED_TARGETS_FILE, mmap_mode="r"
        )
        # 检查 `not np.array_equal(np.isfinite(fitted_check), label_valid)`；根据条件结果选择对应处理分支。
        if not np.array_equal(np.isfinite(fitted_check), label_valid):
            # 抛出 `AssertionError('No.6 fitted target 有限性与 label_valid 不一致')`，阻止不满足数据契约的流程继续执行。
            raise AssertionError("No.6 fitted target 有限性与 label_valid 不一致")
        # 调用 `pd.DataFrame` 转换数据表示，并将标准化对象保存到 `daily`。
        daily = pd.DataFrame(diagnostics)
        # 调用 `daily.to_parquet` 将当前结果持久化到目标文件。
        daily.to_parquet(
            stage / DUAL_TOWER_DAILY_DIAGNOSTICS_FILE,
            index=False,
            compression="zstd",
        )
        # 调用 `daily.groupby('year', as_index=False).agg` 按预先声明的聚合规则汇总分组数据，并将结果保存到 `annual`。
        annual = daily.groupby("year", as_index=False).agg(
            rows=("rows", "sum"),
            label_valid_rows=("label_valid_rows", "sum"),
            dates=("trade_date", "nunique"),
            mean_joint_barra_r_squared=("joint_barra_r_squared", "mean"),
            mean_fitted_joint_barra_r_squared=(
                "fitted_joint_barra_r_squared", "mean"
            ),
            max_reconstruction_error=(
                "reconstruction_max_abs_error", "max"
            ),
            max_residual_sidecar_error=(
                "residual_sidecar_max_abs_error", "max"
            ),
            max_condition_number=("condition_number", "max"),
        )
        # 调用 `annual.to_parquet` 将当前结果持久化到目标文件。
        annual.to_parquet(
            stage / DUAL_TOWER_ANNUAL_SUMMARY_FILE,
            index=False,
            compression="zstd",
        )
        # 构造键值映射，并保存到 `manifest`，供后续按键查询。
        manifest = {
            "signature": signature,
            "payload": payload,
            "sample_count": len(sample),
            "label_valid_count": int(label_valid.sum()),
            "first_date": str(sample["trade_date"].min()),
            "last_date": str(sample["trade_date"].max()),
            "files": {
                "fitted_targets": DUAL_TOWER_FITTED_TARGETS_FILE,
                "daily_diagnostics": DUAL_TOWER_DAILY_DIAGNOSTICS_FILE,
                "annual_summary": DUAL_TOWER_ANNUAL_SUMMARY_FILE,
            },
            "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        }
        # 调用 `(stage / DUAL_TOWER_TARGET_MANIFEST_FILE).write_text` 将当前结果持久化到目标文件。
        (stage / DUAL_TOWER_TARGET_MANIFEST_FILE).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        # 调用 `_validate_dual_tower_target_sidecar` 校验输入、配置或产物是否满足当前阶段的数据契约。
        _validate_dual_tower_target_sidecar(stage, manifest)
        # 调用 `_atomic_replace_directory` 先写临时文件再原子替换正式产物，避免留下半成品。
        _atomic_replace_directory(stage, target)
        # 将 `{**manifest, 'reused': False}` 作为当前函数结果返回给调用方。
        return {**manifest, "reused": False}
    # 捕获 `except BaseException` 对应的异常，并执行明确的降级、记录或清理逻辑。
    except BaseException:
        # 检查 `stage.exists()`；根据条件结果选择对应处理分支。
        if stage.exists():
            # 调用 `shutil.rmtree` 递归删除明确指定的旧输出目录。
            shutil.rmtree(stage)
        # 抛出 `空值`，阻止不满足数据契约的流程继续执行。
        raise


# 定义内部辅助函数 `_atomic_replace_directory`，封装仅供本模块复用的计算步骤。
def _atomic_replace_directory(stage: Path, target: Path) -> None:
    """通过临时文件和原子替换写入产物，避免产生半成品。
    
    输入：
        stage: Path。
        target: Path，格式 [N]。
    输出：
        None。
        """
    # 调用 `target.with_name` 规范化文件路径，并将结果保存到 `backup`。
    backup = target.with_name(
        f".{target.name}.backup.{os.getpid()}.{time.time_ns()}"
    )
    # 检查 `target.exists()`；根据条件结果选择对应处理分支。
    if target.exists():
        # 调用 `target.rename` 把完整临时产物替换为正式输出。
        target.rename(backup)
    # 尝试执行本段主逻辑；若发生异常，则交由后续异常分支处理。
    try:
        # 调用 `stage.rename` 把完整临时产物替换为正式输出。
        stage.rename(target)
    # 捕获 `except BaseException` 对应的异常，并执行明确的降级、记录或清理逻辑。
    except BaseException:
        # 检查 `backup.exists() and (not target.exists())`；根据条件结果选择对应处理分支。
        if backup.exists() and not target.exists():
            # 调用 `backup.rename` 把完整临时产物替换为正式输出。
            backup.rename(target)
        # 抛出 `空值`，阻止不满足数据契约的流程继续执行。
        raise
    # 检查 `backup.exists()`；根据条件结果选择对应处理分支。
    if backup.exists():
        # 调用 `shutil.rmtree` 递归删除明确指定的旧输出目录。
        shutil.rmtree(backup)


# 定义 `build_style_neutral_store` 函数，封装这一阶段可独立调用的处理步骤。
def build_style_neutral_store(
    *,
    store_dir: str | Path = DEFAULT_STORE_DIR,
    sample_index_path: str | Path = base.SAMPLE_INDEX_PATH,
    targets_path: str | Path = base.TARGETS_PATH,
    cne5s_path: str | Path = barra.DEFAULT_CNE5S_PATH,
    eod_path: str | Path = barra.DEFAULT_EOD_PATH,
    calendar_path: str | Path = barra.DEFAULT_CALENDAR_PATH,
    minimum_match_rate: float = MINIMUM_MATCH_RATE,
    overwrite: bool = False,
) -> dict[str, Any]:
    """构建与 feature store 行号严格对齐的十风格矩阵和残差标签。
    
    输入：
        store_dir: str | Path。
        sample_index_path: str | Path。
        targets_path: str | Path。
        cne5s_path: str | Path。
        eod_path: str | Path。
        calendar_path: str | Path。
        minimum_match_rate: float。
        overwrite: bool。
    输出：
        dict[str, Any]。
        """

    # 检查 `not 0 <= minimum_match_rate <= 1`；根据条件结果选择对应处理分支。
    if not 0 <= minimum_match_rate <= 1:
        # 抛出 `ValueError('minimum_match_rate 必须位于 [0,1]')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("minimum_match_rate 必须位于 [0,1]")
    # 调用 `Path(store_dir).resolve` 规范化文件路径，并将结果保存到 `target`。
    target = Path(store_dir).resolve()
    # 调用 `Path(sample_index_path).resolve` 规范化文件路径，并将结果保存到 `sample_path`。
    sample_path = Path(sample_index_path).resolve()
    # 调用 `Path(targets_path).resolve` 规范化文件路径，并将结果保存到 `source_targets_path`。
    source_targets_path = Path(targets_path).resolve()
    # 调用 `Path(cne5s_path).resolve` 规范化文件路径，并将结果保存到 `cne5s`。
    cne5s = Path(cne5s_path).resolve()
    # 调用 `Path(eod_path).resolve` 规范化文件路径，并将结果保存到 `eod`。
    eod = Path(eod_path).resolve()
    # 调用 `Path(calendar_path).resolve` 规范化文件路径，并将结果保存到 `calendar_file`。
    calendar_file = Path(calendar_path).resolve()
    # 调用 `_store_payload` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `payload`。
    payload = _store_payload(
        sample_path, source_targets_path, cne5s, eod, calendar_file
    )
    # 调用 `_stable_hash` 对关键配置生成稳定签名以支持复现和一致性检查，并将结果保存到 `signature`。
    signature = _stable_hash(payload)
    # 计算 `target / MANIFEST_FILE`，并将数值结果保存到 `manifest_path`。
    manifest_path = target / MANIFEST_FILE
    # 检查 `manifest_path.exists() and (not overwrite)`；根据条件结果选择对应处理分支。
    if manifest_path.exists() and not overwrite:
        # 调用 `json.loads` 读取源数据，并将结果保存到 `existing`。
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        # 检查 `existing.get('signature') != signature`；根据条件结果选择对应处理分支。
        if existing.get("signature") != signature:
            # 抛出 `RuntimeError('已有风格中性训练数据签名不同；请显式设置 overwrite=True')`，阻止不满足数据契约的流程继续执行。
            raise RuntimeError(
                "已有风格中性训练数据签名不同；请显式设置 overwrite=True"
            )
        # 调用 `_validate_arrays` 校验输入、配置或产物是否满足当前阶段的数据契约。
        _validate_arrays(target, int(existing["sample_count"]))
        # 将 `{**existing, 'reused': True}` 作为当前函数结果返回给调用方。
        return {**existing, "reused": True}

    # 调用 `pd.read_parquet` 读取源数据，并将结果保存到 `sample`。
    sample = pd.read_parquet(
        sample_path,
        columns=[
            "sample_index", "trade_date", "instrument_id", "label_valid"
        ],
    )
    # 调用 `sample['trade_date'].astype` 统一设备或数据类型，并将结果保存到 `sample['trade_date']`。
    sample["trade_date"] = sample["trade_date"].astype(str)
    # 调用 `sample['instrument_id'].astype` 统一设备或数据类型，并将结果保存到 `sample['instrument_id']`。
    sample["instrument_id"] = sample["instrument_id"].astype(str)
    # 调用 `np.arange` 按目标形状初始化数值容器 `expected_indices`。
    expected_indices = np.arange(len(sample), dtype=np.int64)
    # 检查 `not np.array_equal(sample['sample_index'].to_numpy(np.int64), expected_indices)`；根据条件结果选择对应处理分支。
    if not np.array_equal(
        sample["sample_index"].to_numpy(np.int64), expected_indices
    ):
        # 抛出 `AssertionError('sample_index 必须与物理行号逐行一致')`，阻止不满足数据契约的流程继续执行。
        raise AssertionError("sample_index 必须与物理行号逐行一致")
    # 调用 `np.load` 读取源数据，并将结果保存到 `source_targets`。
    source_targets = np.load(source_targets_path, mmap_mode="r")
    # 检查 `source_targets.shape != (len(sample),)`；根据条件结果选择对应处理分支。
    if source_targets.shape != (len(sample),):
        # 抛出 `ValueError('源 targets.npy 与 sample_index 行数不一致')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("源 targets.npy 与 sample_index 行数不一致")
    # 调用 `sample['label_valid'].to_numpy` 把表格列转换为指定精度的 NumPy 数组，并将结果保存到 `label_valid`。
    label_valid = sample["label_valid"].to_numpy(bool)
    # 检查 `not np.array_equal(np.isfinite(source_targets), label_valid)`；根据条件结果选择对应处理分支。
    if not np.array_equal(np.isfinite(source_targets), label_valid):
        # 抛出 `AssertionError('源标签有限性与 label_valid 不一致')`，阻止不满足数据契约的流程继续执行。
        raise AssertionError("源标签有限性与 label_valid 不一致")

    # 调用 `target.parent.mkdir` 创建输出目录，确保后续文件可以安全写入。
    target.parent.mkdir(parents=True, exist_ok=True)
    # 调用 `Path` 规范化文件路径，并将结果保存到 `stage`。
    stage = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.stage.", dir=target.parent)
    )
    # 尝试执行本段主逻辑；若发生异常，则交由后续异常分支处理。
    try:
        # 调用 `np.lib.format.open_memmap` 以 NPY 格式打开内存映射数组，避免一次载入全部样本，并将结果保存到 `styles_output`。
        styles_output = np.lib.format.open_memmap(
            stage / STYLE_EXPOSURES_FILE,
            mode="w+",
            dtype=np.float32,
            shape=(len(sample), len(STYLE_COLUMNS)),
        )
        # 读取 `np.nan` 的当前值，并赋给 `styles_output[:]`。
        styles_output[:] = np.nan
        # 调用 `np.lib.format.open_memmap` 以 NPY 格式打开内存映射数组，避免一次载入全部样本，并将结果保存到 `valid_output`。
        valid_output = np.lib.format.open_memmap(
            stage / STYLE_VALID_FILE,
            mode="w+",
            dtype=np.bool_,
            shape=(len(sample), len(STYLE_COLUMNS)),
        )
        # 设置 `valid_output[:]` 的固定配置值，供后续流程统一使用。
        valid_output[:] = False
        # 调用 `np.lib.format.open_memmap` 以 NPY 格式打开内存映射数组，避免一次载入全部样本，并将结果保存到 `targets_output`。
        targets_output = np.lib.format.open_memmap(
            stage / RESIDUAL_TARGETS_FILE,
            mode="w+",
            dtype=np.float32,
            shape=(len(sample),),
        )
        # 读取 `np.nan` 的当前值，并赋给 `targets_output[:]`。
        targets_output[:] = np.nan

        # 调用 `barra._load_sse_calendar` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `calendar_dates`。
        calendar_dates = barra._load_sse_calendar(calendar_file)
        # 构造当前步骤所需的列表，并保存到 `diagnostics`。
        diagnostics: list[dict[str, Any]] = []
        # 调用 `sorted` 整理数据结构，并将结果保存到 `years`。
        years = sorted(sample["trade_date"].str[:4].astype(int).unique())
        # 遍历 `years`，并把当前元素绑定到 `year` 逐项处理。
        for year in years:
            # 复制 `sample.loc[sample['trade_date'].str.startswith(str(year))].copy`，避免后续更新污染原对象，并保存到 `yearly`。
            yearly = sample.loc[
                sample["trade_date"].str.startswith(str(year))
            ].copy()
            # 调用 `barra.instrument_ids_to_assetids` 执行 `instrument_ids_to_assetids` 对应的明确变换，并产生后续步骤使用的结果，并将结果保存到 `yearly['assetid']`。
            yearly["assetid"] = barra.instrument_ids_to_assetids(
                yearly["instrument_id"]
            )
            # 调用 `barra._attach_reversal_20d` 按股票和日期键拼接当前新增字段，并将结果保存到 `yearly`。
            yearly = barra._attach_reversal_20d(
                yearly, eod, calendar_dates, year
            )
            # 调用 `barra._load_cne5s_year` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `cne5s_year`。
            cne5s_year = barra._load_cne5s_year(cne5s, year)
            # 调用 `barra._join_prediction_and_cne5s` 执行模型推理并生成样本预测分数，并将结果保存到 `joined`。
            joined = barra._join_prediction_and_cne5s(
                yearly, cne5s_year, year
            )
            # 调用 `float` 统一设备或数据类型，并将结果保存到 `match_rate`。
            match_rate = float(joined["barra_matched"].mean())
            # 检查 `match_rate < minimum_match_rate`；根据条件结果选择对应处理分支。
            if match_rate < minimum_match_rate:
                # 抛出 `AssertionError(f'{year} CNE5S 匹配率 {match_rate:.6%} < {minimum_match_rate:.6%}')`，阻止不满足数据契约的流程继续执行。
                raise AssertionError(
                    f"{year} CNE5S 匹配率 {match_rate:.6%} "
                    f"< {minimum_match_rate:.6%}"
                )
            # 遍历 `joined.groupby('trade_date', sort=True).groups.items()`，并把当前元素绑定到 `(trade_date, positions)` 逐项处理。
            for trade_date, positions in joined.groupby(
                "trade_date", sort=True
            ).groups.items():
                # 读取 `joined.iloc[np.asarray(positions, dtype=np.int64)]` 的当前值，并赋给 `daily`。
                daily = joined.iloc[np.asarray(positions, dtype=np.int64)]
                # 调用 `daily['sample_index'].to_numpy` 把表格列转换为指定精度的 NumPy 数组，并将结果保存到 `sample_indices`。
                sample_indices = daily["sample_index"].to_numpy(np.int64)
                # 调用 `len` 汇总当前数据，并将统计量保存到 `rows`。
                rows = len(daily)
                # 调用 `np.zeros` 按目标形状初始化数值容器 `style_matrix`。
                style_matrix = np.zeros(
                    (rows, len(STYLE_COLUMNS)), dtype=np.float64
                )
                # 调用 `np.zeros_like` 按目标形状初始化数值容器 `style_valid`。
                style_valid = np.zeros_like(style_matrix, dtype=bool)
                # 调用 `daily['barra_matched'].to_numpy` 把表格列转换为指定精度的 NumPy 数组，并将结果保存到 `matched`。
                matched = daily["barra_matched"].to_numpy(bool)
                # 遍历 `enumerate(barra.STYLE_COLUMNS)`，并把当前元素绑定到 `(column_index, column)` 逐项处理。
                for column_index, column in enumerate(barra.STYLE_COLUMNS):
                    # 调用 `daily[column].to_numpy` 把表格列转换为指定精度的 NumPy 数组，并将结果保存到 `raw`。
                    raw = daily[column].to_numpy(np.float64)
                    # 计算 `matched & np.isfinite(raw)`，并将数值结果保存到 `valid`。
                    valid = matched & np.isfinite(raw)
                    # 调用 `_standardize_with_missing` 按当前截面统计量完成缩尾或标准化处理，并将结果保存到 `style_matrix[:, column_index]`。
                    style_matrix[:, column_index] = _standardize_with_missing(
                        raw, valid, label=f"{trade_date} {column}"
                    )
                    # 读取 `valid` 的当前值，并赋给 `style_valid[:, column_index]`。
                    style_valid[:, column_index] = valid
                # 计算 `len(STYLE_COLUMNS) - 1`，并将数值结果保存到 `reversal_index`。
                reversal_index = len(STYLE_COLUMNS) - 1
                # 调用 `daily[barra.REVERSAL_VALID_COLUMN].to_numpy` 把表格列转换为指定精度的 NumPy 数组，并将结果保存到 `reversal_valid`。
                reversal_valid = daily[
                    barra.REVERSAL_VALID_COLUMN
                ].to_numpy(bool)
                # 调用 `_standardize_with_missing` 按当前截面统计量完成缩尾或标准化处理，并将结果保存到 `style_matrix[:, reversal_index]`。
                style_matrix[:, reversal_index] = _standardize_with_missing(
                    daily[barra.REVERSAL_COLUMN].to_numpy(np.float64),
                    reversal_valid,
                    label=f"{trade_date} {barra.REVERSAL_COLUMN}",
                )
                # 读取 `reversal_valid` 的当前值，并赋给 `style_valid[:, reversal_index]`。
                style_valid[:, reversal_index] = reversal_valid

                # 读取 `label_valid[sample_indices]` 的当前值，并赋给 `daily_label_valid`。
                daily_label_valid = label_valid[sample_indices]
                # 调用 `residualize_daily_targets` 按 Barra 暴露执行截面投影或残差化处理，并将结果保存到 `(residual, record)`。
                residual, record = residualize_daily_targets(
                    np.asarray(source_targets[sample_indices], dtype=np.float64),
                    style_matrix,
                    daily_label_valid,
                )
                # 调用 `style_matrix.astype` 统一设备或数据类型，并将结果保存到 `styles_output[sample_indices]`。
                styles_output[sample_indices] = style_matrix.astype(np.float32)
                # 读取 `style_valid` 的当前值，并赋给 `valid_output[sample_indices]`。
                valid_output[sample_indices] = style_valid
                # 调用 `residual.astype` 统一设备或数据类型，并将结果保存到 `targets_output[sample_indices]`。
                targets_output[sample_indices] = residual.astype(np.float32)
                # 把本轮结果追加到 `diagnostics`，累积后续需要汇总的记录。
                diagnostics.append(
                    {
                        "year": year,
                        "trade_date": str(trade_date),
                        "barra_matched_rows": int(matched.sum()),
                        "barra_match_rate": float(matched.mean()),
                        "reversal_valid_rows": int(reversal_valid.sum()),
                        "reversal_coverage": float(reversal_valid.mean()),
                        "complete_style_rows": int(style_valid.all(axis=1).sum()),
                        "complete_style_rate": float(style_valid.all(axis=1).mean()),
                        **record,
                    }
                )
            # 调用 `print` 记录当前流程状态或诊断信息。
            print(
                f"{year}：写入 {len(joined):,} 行，"
                f"CNE5S 匹配率 {match_rate:.4%}",
                flush=True,
            )

        # 调用 `styles_output.flush` 把内存映射中的缓存修改刷新到磁盘。
        styles_output.flush()
        # 调用 `valid_output.flush` 把内存映射中的缓存修改刷新到磁盘。
        valid_output.flush()
        # 调用 `targets_output.flush` 把内存映射中的缓存修改刷新到磁盘。
        targets_output.flush()
        # 删除 `styles_output, valid_output, targets_output`，释放不再使用的引用。
        del styles_output, valid_output, targets_output
        # 调用 `_validate_arrays` 校验输入、配置或产物是否满足当前阶段的数据契约。
        _validate_arrays(stage, len(sample))
        # 调用 `np.load` 读取源数据，并将结果保存到 `residual_check`。
        residual_check = np.load(
            stage / RESIDUAL_TARGETS_FILE, mmap_mode="r"
        )
        # 检查 `not np.array_equal(np.isfinite(residual_check), label_valid)`；根据条件结果选择对应处理分支。
        if not np.array_equal(np.isfinite(residual_check), label_valid):
            # 抛出 `AssertionError('残差标签有限性与 label_valid 不一致')`，阻止不满足数据契约的流程继续执行。
            raise AssertionError("残差标签有限性与 label_valid 不一致")

        # 调用 `pd.DataFrame` 转换数据表示，并将标准化对象保存到 `daily_diagnostics`。
        daily_diagnostics = pd.DataFrame(diagnostics)
        # 调用 `daily_diagnostics.to_parquet` 将当前结果持久化到目标文件。
        daily_diagnostics.to_parquet(
            stage / DAILY_DIAGNOSTICS_FILE, index=False, compression="zstd"
        )
        # 调用 `daily_diagnostics.groupby('year', as_index=False).agg` 按预先声明的聚合规则汇总分组数据，并将结果保存到 `annual_summary`。
        annual_summary = daily_diagnostics.groupby(
            "year", as_index=False
        ).agg(
            rows=("rows", "sum"),
            label_valid_rows=("label_valid_rows", "sum"),
            dates=("trade_date", "nunique"),
            mean_barra_match_rate=("barra_match_rate", "mean"),
            min_barra_match_rate=("barra_match_rate", "min"),
            mean_reversal_coverage=("reversal_coverage", "mean"),
            min_reversal_coverage=("reversal_coverage", "min"),
            mean_complete_style_rate=("complete_style_rate", "mean"),
            mean_style_r_squared=("style_r_squared", "mean"),
            max_abs_style_residual_pearson=(
                "max_abs_style_residual_pearson", "max"
            ),
        )
        # 调用 `annual_summary.to_parquet` 将当前结果持久化到目标文件。
        annual_summary.to_parquet(
            stage / ANNUAL_SUMMARY_FILE, index=False, compression="zstd"
        )
        # 构造键值映射，并保存到 `manifest`，供后续按键查询。
        manifest = {
            "signature": signature,
            "payload": payload,
            "sample_count": len(sample),
            "label_valid_count": int(label_valid.sum()),
            "first_date": str(sample["trade_date"].min()),
            "last_date": str(sample["trade_date"].max()),
            "minimum_match_rate": minimum_match_rate,
            "files": {
                "style_exposures": STYLE_EXPOSURES_FILE,
                "style_valid": STYLE_VALID_FILE,
                "residual_targets": RESIDUAL_TARGETS_FILE,
                "daily_diagnostics": DAILY_DIAGNOSTICS_FILE,
                "annual_summary": ANNUAL_SUMMARY_FILE,
            },
            "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        }
        # 调用 `(stage / MANIFEST_FILE).write_text` 将当前结果持久化到目标文件。
        (stage / MANIFEST_FILE).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        # 调用 `_atomic_replace_directory` 先写临时文件再原子替换正式产物，避免留下半成品。
        _atomic_replace_directory(stage, target)
        # 将 `{**manifest, 'reused': False}` 作为当前函数结果返回给调用方。
        return {**manifest, "reused": False}
    # 捕获 `except BaseException` 对应的异常，并执行明确的降级、记录或清理逻辑。
    except BaseException:
        # 检查 `stage.exists()`；根据条件结果选择对应处理分支。
        if stage.exists():
            # 调用 `shutil.rmtree` 递归删除明确指定的旧输出目录。
            shutil.rmtree(stage)
        # 抛出 `空值`，阻止不满足数据契约的流程继续执行。
        raise


# 构造当前步骤所需的列表，并保存到 `__all__`。
__all__ = [
    "ANNUAL_SUMMARY_FILE",
    "DAILY_DIAGNOSTICS_FILE",
    "DEFAULT_DUAL_TOWER_TARGET_STORE_DIR",
    "DEFAULT_FULL_BARRA_TARGET_STORE_DIR",
    "DEFAULT_INDUSTRY_STORE_DIR",
    "DEFAULT_STORE_DIR",
    "DUAL_TOWER_FITTED_TARGETS_FILE",
    "FULL_BARRA_TARGETS_FILE",
    "INDUSTRY_CODES_FILE",
    "INDUSTRY_MANIFEST_FILE",
    "MANIFEST_FILE",
    "RESIDUAL_TARGETS_FILE",
    "STYLE_COLUMNS",
    "STYLE_EXPOSURES_FILE",
    "STYLE_VALID_FILE",
    "build_dual_tower_target_sidecar",
    "decompose_daily_targets_with_industry",
    "build_full_barra_target_sidecar",
    "build_industry_sidecar",
    "build_style_neutral_store",
    "joint_barra_r_squared_numpy",
    "joint_barra_r_squared_torch",
    "project_out_full_barra_torch",
    "project_out_styles_numpy",
    "project_out_styles_torch",
    "require_dual_tower_target_sidecar",
    "require_full_barra_target_sidecar",
    "require_industry_sidecar",
    "require_style_neutral_store",
    "residualize_daily_targets",
    "residualize_daily_targets_with_industry",
]
# 定义内部辅助函数 `_validate_arrays`，封装仅供本模块复用的计算步骤。
def _validate_arrays(store_dir: Path, sample_count: int) -> None:
    """校验输入形状、业务键、有限性和签名约束；异常时明确失败。
    
    输入：
        store_dir: Path。
        sample_count: int。
    输出：
        None。
        """
    # 调用 `np.load` 读取源数据，并将结果保存到 `styles`。
    styles = np.load(store_dir / STYLE_EXPOSURES_FILE, mmap_mode="r")
    # 调用 `np.load` 读取源数据，并将结果保存到 `valid`。
    valid = np.load(store_dir / STYLE_VALID_FILE, mmap_mode="r")
    # 调用 `np.load` 读取源数据，并将结果保存到 `residual`。
    residual = np.load(store_dir / RESIDUAL_TARGETS_FILE, mmap_mode="r")
    # 检查 `styles.shape != (sample_count, len(STYLE_COLUMNS)) or styles.dtype != np.float32`；根据条件结果选择对应处理分支。
    if styles.shape != (sample_count, len(STYLE_COLUMNS)) or styles.dtype != np.float32:
        # 抛出 `ValueError('style_exposures.npy 形状或 dtype 异常')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("style_exposures.npy 形状或 dtype 异常")
    # 检查 `valid.shape != styles.shape or valid.dtype != np.bool_`；根据条件结果选择对应处理分支。
    if valid.shape != styles.shape or valid.dtype != np.bool_:
        # 抛出 `ValueError('style_valid.npy 形状或 dtype 异常')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("style_valid.npy 形状或 dtype 异常")
    # 检查 `residual.shape != (sample_count,) or residual.dtype != np.float32`；根据条件结果选择对应处理分支。
    if residual.shape != (sample_count,) or residual.dtype != np.float32:
        # 抛出 `ValueError('targets_style_residual.npy 形状或 dtype 异常')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("targets_style_residual.npy 形状或 dtype 异常")
    # 检查 `not np.isfinite(styles).all()`；根据条件结果选择对应处理分支。
    if not np.isfinite(styles).all():
        # 抛出 `ValueError('style_exposures.npy 含 NaN/Inf')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("style_exposures.npy 含 NaN/Inf")

# 定义 `require_style_neutral_store` 函数，封装这一阶段可独立调用的处理步骤。
def require_style_neutral_store(
    store_dir: str | Path = DEFAULT_STORE_DIR,
) -> dict[str, Any]:
    """验证风格中性训练数据与当前源数据签名一致。
    
    输入：
        store_dir: str | Path。
    输出：
        dict[str, Any]。
        """

    # 调用 `Path(store_dir).resolve` 规范化文件路径，并将结果保存到 `root`。
    root = Path(store_dir).resolve()
    # 计算 `root / MANIFEST_FILE`，并将数值结果保存到 `manifest_path`。
    manifest_path = root / MANIFEST_FILE
    # 检查 `not manifest_path.exists()`；根据条件结果选择对应处理分支。
    if not manifest_path.exists():
        # 抛出 `FileNotFoundError('缺少风格中性训练数据，请先运行 build_style_neutral_store')`，阻止不满足数据契约的流程继续执行。
        raise FileNotFoundError("缺少风格中性训练数据，请先运行 build_style_neutral_store")
    # 调用 `json.loads` 读取源数据，并将结果保存到 `manifest`。
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # 调用 `_store_payload` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `expected`。
    expected = _store_payload(
        base.SAMPLE_INDEX_PATH,
        base.TARGETS_PATH,
        barra.DEFAULT_CNE5S_PATH,
        barra.DEFAULT_EOD_PATH,
        barra.DEFAULT_CALENDAR_PATH,
    )
    # 检查 `manifest.get('signature') != _stable_hash(expected)`；根据条件结果选择对应处理分支。
    if manifest.get("signature") != _stable_hash(expected):
        # 抛出 `ValueError('风格中性训练数据与当前源数据签名不一致')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("风格中性训练数据与当前源数据签名不一致")
    # 调用 `_validate_arrays` 校验输入、配置或产物是否满足当前阶段的数据契约。
    _validate_arrays(root, int(manifest["sample_count"]))
    # 将 `manifest` 作为当前函数结果返回给调用方。
    return manifest

# 定义内部辅助函数 `_validate_industry_sidecar`，封装仅供本模块复用的计算步骤。
def _validate_industry_sidecar(
    store_dir: Path,
    manifest: dict[str, Any],
) -> None:
    """校验输入形状、业务键、有限性和签名约束；异常时明确失败。
    
    输入：
        store_dir: Path。
        manifest: dict[str, Any]。
    输出：
        None。
        """
    # 调用 `np.load` 读取源数据，并将结果保存到 `codes`。
    codes = np.load(store_dir / INDUSTRY_CODES_FILE, mmap_mode="r")
    # 调用 `int` 统一设备或数据类型，并将结果保存到 `sample_count`。
    sample_count = int(manifest["sample_count"])
    # 调用 `manifest.get` 按键读取映射中的可选字段，并在缺失时使用约定默认值，并将结果保存到 `mapping`。
    mapping = manifest.get("industry_mapping")
    # 检查 `codes.shape != (sample_count,) or codes.dtype != np.int16`；根据条件结果选择对应处理分支。
    if codes.shape != (sample_count,) or codes.dtype != np.int16:
        # 抛出 `ValueError('industry_codes.npy 形状或 dtype 异常')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("industry_codes.npy 形状或 dtype 异常")
    # 检查 `not isinstance(mapping, dict) or mapping.get(barra.UNKNOWN_INDUSTRY) != 0`；根据条件结果选择对应处理分支。
    if not isinstance(mapping, dict) or mapping.get(barra.UNKNOWN_INDUSTRY) != 0:
        # 抛出 `ValueError('行业 sidecar 缺少稳定行业映射')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("行业 sidecar 缺少稳定行业映射")
    # 检查 `(codes < 0).any() or int(codes.max()) >= len(mapping)`；根据条件结果选择对应处理分支。
    if (codes < 0).any() or int(codes.max()) >= len(mapping):
        # 抛出 `ValueError('industry_codes.npy 含非法行业代码')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("industry_codes.npy 含非法行业代码")
    # 计算 `len(mapping) - 1`，并将数值结果保存到 `known_count`。
    known_count = len(mapping) - 1
    # 检查 `known_count != EXPECTED_CNE5S_INDUSTRIES`；根据条件结果选择对应处理分支。
    if known_count != EXPECTED_CNE5S_INDUSTRIES:
        # 抛出 `ValueError(f'CNE5S 行业数为 {known_count}，预期 {EXPECTED_CNE5S_INDUSTRIES}')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(
            f"CNE5S 行业数为 {known_count}，预期 {EXPECTED_CNE5S_INDUSTRIES}"
        )

# 定义 `require_industry_sidecar` 函数，封装这一阶段可独立调用的处理步骤。
def require_industry_sidecar(
    store_dir: str | Path = DEFAULT_INDUSTRY_STORE_DIR,
    *,
    minimum_match_rate: float = MINIMUM_MATCH_RATE,
) -> dict[str, Any]:
    """验证联合 R² 所需的逐样本行业代码 sidecar。
    
    输入：
        store_dir: str | Path。
        minimum_match_rate: float。
    输出：
        dict[str, Any]。
        """

    # 调用 `Path(store_dir).resolve` 规范化文件路径，并将结果保存到 `root`。
    root = Path(store_dir).resolve()
    # 计算 `root / INDUSTRY_MANIFEST_FILE`，并将数值结果保存到 `manifest_path`。
    manifest_path = root / INDUSTRY_MANIFEST_FILE
    # 检查 `not manifest_path.exists()`；根据条件结果选择对应处理分支。
    if not manifest_path.exists():
        # 抛出 `FileNotFoundError('缺少行业 sidecar，请先运行 build_industry_sidecar')`，阻止不满足数据契约的流程继续执行。
        raise FileNotFoundError(
            "缺少行业 sidecar，请先运行 build_industry_sidecar"
        )
    # 调用 `json.loads` 读取源数据，并将结果保存到 `manifest`。
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # 调用 `require_style_neutral_store` 校验输入、配置或产物是否满足当前阶段的数据契约，并将结果保存到 `style_manifest`。
    style_manifest = require_style_neutral_store()
    # 调用 `_industry_sidecar_payload` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `expected`。
    expected = _industry_sidecar_payload(
        base.SAMPLE_INDEX_PATH,
        barra.DEFAULT_CNE5S_PATH,
        style_manifest["signature"],
        minimum_match_rate,
    )
    # 检查 `manifest.get('signature') != _stable_hash(expected)`；根据条件结果选择对应处理分支。
    if manifest.get("signature") != _stable_hash(expected):
        # 抛出 `ValueError('行业 sidecar 与当前样本、CNE5S 或风格 store 签名不一致')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("行业 sidecar 与当前样本、CNE5S 或风格 store 签名不一致")
    # 调用 `_validate_industry_sidecar` 校验输入、配置或产物是否满足当前阶段的数据契约。
    _validate_industry_sidecar(root, manifest)
    # 将 `manifest` 作为当前函数结果返回给调用方。
    return manifest

# 定义内部辅助函数 `_validate_full_barra_target_sidecar`，封装仅供本模块复用的计算步骤。
def _validate_full_barra_target_sidecar(
    store_dir: Path,
    manifest: dict[str, Any],
) -> None:
    """校验输入形状、业务键、有限性和签名约束；异常时明确失败。
    
    输入：
        store_dir: Path。
        manifest: dict[str, Any]。
    输出：
        None。
        """
    # 组织一组有序值，并保存到 `required_files`。
    required_files = (
        FULL_BARRA_TARGETS_FILE,
        FULL_BARRA_DAILY_DIAGNOSTICS_FILE,
        FULL_BARRA_ANNUAL_SUMMARY_FILE,
        FULL_BARRA_TARGET_MANIFEST_FILE,
    )
    # 构造当前步骤所需的列表，并保存到 `missing`。
    missing = [name for name in required_files if not (store_dir / name).exists()]
    # 检查 `missing`；根据条件结果选择对应处理分支。
    if missing:
        # 抛出 `FileNotFoundError(f'完整 Barra 残差标签文件缺失：{missing}')`，阻止不满足数据契约的流程继续执行。
        raise FileNotFoundError(f"完整 Barra 残差标签文件缺失：{missing}")
    # 调用 `np.load` 读取源数据，并将结果保存到 `targets`。
    targets = np.load(store_dir / FULL_BARRA_TARGETS_FILE, mmap_mode="r")
    # 检查 `targets.shape != (int(manifest['sample_count']),) or targets.dtype != np.float32`；根据条件结果选择对应处理分支。
    if (
        targets.shape != (int(manifest["sample_count"]),)
        or targets.dtype != np.float32
    ):
        # 抛出 `ValueError('targets_full_barra_residual.npy 形状或 dtype 异常')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(
            "targets_full_barra_residual.npy 形状或 dtype 异常"
        )
    # 检查 `int(np.isfinite(targets).sum()) != int(manifest['label_valid_count'])`；根据条件结果选择对应处理分支。
    if int(np.isfinite(targets).sum()) != int(manifest["label_valid_count"]):
        # 抛出 `ValueError('完整 Barra 残差标签有限值数量与 manifest 不一致')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("完整 Barra 残差标签有限值数量与 manifest 不一致")
    # 调用 `pd.read_parquet` 读取源数据，并将结果保存到 `daily`。
    daily = pd.read_parquet(
        store_dir / FULL_BARRA_DAILY_DIAGNOSTICS_FILE,
        columns=[
            "rows",
            "max_abs_style_residual_pearson",
            "max_abs_industry_residual_mean",
        ],
    )
    # 检查 `int(daily['rows'].sum()) != int(manifest['sample_count'])`；根据条件结果选择对应处理分支。
    if int(daily["rows"].sum()) != int(manifest["sample_count"]):
        # 抛出 `ValueError('完整 Barra 逐日诊断行数汇总与 manifest 不一致')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("完整 Barra 逐日诊断行数汇总与 manifest 不一致")
    # 检查 `daily[['max_abs_style_residual_pearson', 'max_abs_industry_residual_mean']].to_numpy().max() > …`；根据条件结果选择对应处理分支。
    if daily[
        [
            "max_abs_style_residual_pearson",
            "max_abs_industry_residual_mean",
        ]
    ].to_numpy().max() > ORTHOGONALITY_TOLERANCE:
        # 抛出 `ValueError('完整 Barra 残差标签正交诊断未通过')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("完整 Barra 残差标签正交诊断未通过")

# 定义 `require_full_barra_target_sidecar` 函数，封装这一阶段可独立调用的处理步骤。
def require_full_barra_target_sidecar(
    store_dir: str | Path = DEFAULT_FULL_BARRA_TARGET_STORE_DIR,
) -> dict[str, Any]:
    """验证十风格加行业固定效应残差标签 sidecar。
    
    输入：
        store_dir: str | Path。
    输出：
        dict[str, Any]。
        """

    # 调用 `Path(store_dir).resolve` 规范化文件路径，并将结果保存到 `root`。
    root = Path(store_dir).resolve()
    # 计算 `root / FULL_BARRA_TARGET_MANIFEST_FILE`，并将数值结果保存到 `manifest_path`。
    manifest_path = root / FULL_BARRA_TARGET_MANIFEST_FILE
    # 检查 `not manifest_path.exists()`；根据条件结果选择对应处理分支。
    if not manifest_path.exists():
        # 抛出 `FileNotFoundError('缺少完整 Barra 残差标签，请先运行 build_full_barra_target_sidecar')`，阻止不满足数据契约的流程继续执行。
        raise FileNotFoundError(
            "缺少完整 Barra 残差标签，请先运行 "
            "build_full_barra_target_sidecar"
        )
    # 调用 `json.loads` 读取源数据，并将结果保存到 `manifest`。
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # 调用 `require_style_neutral_store` 校验输入、配置或产物是否满足当前阶段的数据契约，并将结果保存到 `style_manifest`。
    style_manifest = require_style_neutral_store()
    # 调用 `require_industry_sidecar` 校验输入、配置或产物是否满足当前阶段的数据契约，并将结果保存到 `industry_manifest`。
    industry_manifest = require_industry_sidecar()
    # 调用 `_full_barra_target_payload` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `expected`。
    expected = _full_barra_target_payload(
        base.TARGETS_PATH,
        style_manifest["signature"],
        industry_manifest["signature"],
    )
    # 检查 `manifest.get('signature') != _stable_hash(expected)`；根据条件结果选择对应处理分支。
    if manifest.get("signature") != _stable_hash(expected):
        # 抛出 `ValueError('完整 Barra 残差标签与当前标签、风格或行业 sidecar 签名不一致')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(
            "完整 Barra 残差标签与当前标签、风格或行业 sidecar 签名不一致"
        )
    # 调用 `_validate_full_barra_target_sidecar` 校验输入、配置或产物是否满足当前阶段的数据契约。
    _validate_full_barra_target_sidecar(root, manifest)
    # 将 `manifest` 作为当前函数结果返回给调用方。
    return manifest

# 定义内部辅助函数 `_validate_dual_tower_target_sidecar`，封装仅供本模块复用的计算步骤。
def _validate_dual_tower_target_sidecar(
    store_dir: Path, manifest: dict[str, Any]
) -> None:
    """校验输入形状、业务键、有限性和签名约束；异常时明确失败。
    
    输入：
        store_dir: Path。
        manifest: dict[str, Any]。
    输出：
        None。
        """
    # 组织一组有序值，并保存到 `required`。
    required = (
        DUAL_TOWER_FITTED_TARGETS_FILE,
        DUAL_TOWER_DAILY_DIAGNOSTICS_FILE,
        DUAL_TOWER_ANNUAL_SUMMARY_FILE,
        DUAL_TOWER_TARGET_MANIFEST_FILE,
    )
    # 构造当前步骤所需的列表，并保存到 `missing`。
    missing = [name for name in required if not (store_dir / name).exists()]
    # 检查 `missing`；根据条件结果选择对应处理分支。
    if missing:
        # 抛出 `FileNotFoundError(f'No.6 fitted-target sidecar 文件缺失：{missing}')`，阻止不满足数据契约的流程继续执行。
        raise FileNotFoundError(f"No.6 fitted-target sidecar 文件缺失：{missing}")
    # 调用 `np.load` 读取源数据，并将结果保存到 `targets`。
    targets = np.load(
        store_dir / DUAL_TOWER_FITTED_TARGETS_FILE, mmap_mode="r"
    )
    # 检查 `targets.shape != (int(manifest['sample_count']),) or targets.dtype != np.float32`；根据条件结果选择对应处理分支。
    if (
        targets.shape != (int(manifest["sample_count"]),)
        or targets.dtype != np.float32
    ):
        # 抛出 `ValueError('No.6 fitted target 形状或 dtype 异常')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("No.6 fitted target 形状或 dtype 异常")
    # 检查 `int(np.isfinite(targets).sum()) != int(manifest['label_valid_count'])`；根据条件结果选择对应处理分支。
    if int(np.isfinite(targets).sum()) != int(manifest["label_valid_count"]):
        # 抛出 `ValueError('No.6 fitted target 有效值数量与 manifest 不一致')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("No.6 fitted target 有效值数量与 manifest 不一致")
    # 调用 `pd.read_parquet` 读取源数据，并将结果保存到 `daily`。
    daily = pd.read_parquet(
        store_dir / DUAL_TOWER_DAILY_DIAGNOSTICS_FILE,
        columns=[
            "rows",
            "reconstruction_max_abs_error",
            "residual_sidecar_max_abs_error",
            "fitted_joint_barra_r_squared",
        ],
    )
    # 检查 `int(daily['rows'].sum()) != int(manifest['sample_count'])`；根据条件结果选择对应处理分支。
    if int(daily["rows"].sum()) != int(manifest["sample_count"]):
        # 抛出 `ValueError('No.6 fitted target 逐日行数与 manifest 不一致')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("No.6 fitted target 逐日行数与 manifest 不一致")
    # 检查 `float(daily['reconstruction_max_abs_error'].max()) > 1e-12`；根据条件结果选择对应处理分支。
    if float(daily["reconstruction_max_abs_error"].max()) > 1e-12:
        # 抛出 `ValueError('No.6 标签分解无法精确重构源标签')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("No.6 标签分解无法精确重构源标签")
    # 检查 `float(daily['residual_sidecar_max_abs_error'].max()) > 1e-06`；根据条件结果选择对应处理分支。
    if float(daily["residual_sidecar_max_abs_error"].max()) > 1e-6:
        # 抛出 `ValueError('No.6 重算残差与现有完整残差 sidecar 不一致')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("No.6 重算残差与现有完整残差 sidecar 不一致")
    # 检查 `float((1.0 - daily['fitted_joint_barra_r_squared']).abs().max()) > 1e-08`；根据条件结果选择对应处理分支。
    if float((1.0 - daily["fitted_joint_barra_r_squared"]).abs().max()) > 1e-8:
        # 抛出 `ValueError('No.6 fitted target 不在完整 Barra 风险列空间内')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("No.6 fitted target 不在完整 Barra 风险列空间内")

# 定义 `require_dual_tower_target_sidecar` 函数，封装这一阶段可独立调用的处理步骤。
def require_dual_tower_target_sidecar(
    store_dir: str | Path = DEFAULT_DUAL_TOWER_TARGET_STORE_DIR,
) -> dict[str, Any]:
    """验证 No.6 风格塔拟合收益标签 sidecar。
    
    输入：
        store_dir: str | Path。
    输出：
        dict[str, Any]。
        """

    # 调用 `Path(store_dir).resolve` 规范化文件路径，并将结果保存到 `root`。
    root = Path(store_dir).resolve()
    # 计算 `root / DUAL_TOWER_TARGET_MANIFEST_FILE`，并将数值结果保存到 `manifest_path`。
    manifest_path = root / DUAL_TOWER_TARGET_MANIFEST_FILE
    # 检查 `not manifest_path.exists()`；根据条件结果选择对应处理分支。
    if not manifest_path.exists():
        # 抛出 `FileNotFoundError('缺少 No.6 fitted-target sidecar，请先运行 build_dual_tower_target_sidecar')`，阻止不满足数据契约的流程继续执行。
        raise FileNotFoundError(
            "缺少 No.6 fitted-target sidecar，请先运行 "
            "build_dual_tower_target_sidecar"
        )
    # 调用 `json.loads` 读取源数据，并将结果保存到 `manifest`。
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # 调用 `require_style_neutral_store` 校验输入、配置或产物是否满足当前阶段的数据契约，并将结果保存到 `style_manifest`。
    style_manifest = require_style_neutral_store()
    # 调用 `require_industry_sidecar` 校验输入、配置或产物是否满足当前阶段的数据契约，并将结果保存到 `industry_manifest`。
    industry_manifest = require_industry_sidecar()
    # 调用 `require_full_barra_target_sidecar` 校验输入、配置或产物是否满足当前阶段的数据契约，并将结果保存到 `full_manifest`。
    full_manifest = require_full_barra_target_sidecar()
    # 调用 `_dual_tower_target_payload` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `expected`。
    expected = _dual_tower_target_payload(
        base.TARGETS_PATH,
        full_manifest["signature"],
        style_manifest["signature"],
        industry_manifest["signature"],
    )
    # 检查 `manifest.get('signature') != _stable_hash(expected)`；根据条件结果选择对应处理分支。
    if manifest.get("signature") != _stable_hash(expected):
        # 抛出 `ValueError('No.6 fitted target 与当前标签或 Barra sidecar 签名不一致')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("No.6 fitted target 与当前标签或 Barra sidecar 签名不一致")
    # 调用 `_validate_dual_tower_target_sidecar` 校验输入、配置或产物是否满足当前阶段的数据契约。
    _validate_dual_tower_target_sidecar(root, manifest)
    # 将 `manifest` 作为当前函数结果返回给调用方。
    return manifest

