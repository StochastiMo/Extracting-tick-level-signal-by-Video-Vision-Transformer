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


VIVIT_DIR = Path(__file__).resolve().parent
DATA_DIR = VIVIT_DIR / "data"
DEFAULT_STORE_DIR = DATA_DIR / "style_neutral_store"
STYLE_EXPOSURES_FILE = "style_exposures.npy"
STYLE_VALID_FILE = "style_valid.npy"
RESIDUAL_TARGETS_FILE = "targets_style_residual.npy"
DAILY_DIAGNOSTICS_FILE = "daily_diagnostics.parquet"
ANNUAL_SUMMARY_FILE = "annual_summary.parquet"
MANIFEST_FILE = "style_neutral_manifest.json"
DEFAULT_INDUSTRY_STORE_DIR = DEFAULT_STORE_DIR / "industry_sidecar"
INDUSTRY_CODES_FILE = "industry_codes.npy"
INDUSTRY_AUDIT_FILE = "industry_match_audit.parquet"
INDUSTRY_MANIFEST_FILE = "industry_sidecar_manifest.json"
DEFAULT_FULL_BARRA_TARGET_STORE_DIR = (
    DEFAULT_STORE_DIR / "full_barra_target_sidecar"
)
FULL_BARRA_TARGETS_FILE = "targets_full_barra_residual.npy"
FULL_BARRA_DAILY_DIAGNOSTICS_FILE = "daily_diagnostics.parquet"
FULL_BARRA_ANNUAL_SUMMARY_FILE = "annual_summary.parquet"
FULL_BARRA_TARGET_MANIFEST_FILE = "full_barra_target_manifest.json"
DEFAULT_DUAL_TOWER_TARGET_STORE_DIR = (
    DEFAULT_STORE_DIR / "dual_tower_target_sidecar"
)
DUAL_TOWER_FITTED_TARGETS_FILE = "targets_full_barra_fitted.npy"
DUAL_TOWER_DAILY_DIAGNOSTICS_FILE = "daily_diagnostics.parquet"
DUAL_TOWER_ANNUAL_SUMMARY_FILE = "annual_summary.parquet"
DUAL_TOWER_TARGET_MANIFEST_FILE = "dual_tower_target_manifest.json"
STYLE_COLUMNS = (*barra.STYLE_COLUMNS, barra.REVERSAL_COLUMN)
SCHEMA_VERSION = 1
INDUSTRY_SIDECAR_SCHEMA_VERSION = 1
FULL_BARRA_TARGET_SCHEMA_VERSION = 1
DUAL_TOWER_TARGET_SCHEMA_VERSION = 1
EXPECTED_CNE5S_INDUSTRIES = 32
MINIMUM_MATCH_RATE = 0.999
ORTHOGONALITY_TOLERANCE = 1e-8


def _file_fingerprint(path: Path) -> dict[str, Any]:
    """提取文件身份信息，用于来源审计和复用判断。"""
    if not path.exists():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "size": int(path.stat().st_size),
        "mtime_ns": int(path.stat().st_mtime_ns),
    }


def _stable_hash(payload: object) -> str:
    """对可 JSON 序列化载荷计算稳定的 SHA-256 签名。"""
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _store_payload(
    sample_index_path: Path,
    targets_path: Path,
    cne5s_path: Path,
    eod_path: Path,
    calendar_path: Path,
) -> dict[str, Any]:
    """构造参与稳定签名和复用判断的配置载荷。"""
    dataset = json.loads(
        base.SOURCE_DATASET_MANIFEST_PATH.read_text(encoding="utf-8")
    )
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


def _standardize_with_missing(
    values: np.ndarray,
    valid: np.ndarray,
    *,
    label: str,
) -> np.ndarray:
    """有效值按日标准化，缺失值填 0；不做任何跨日填充。"""

    raw = np.asarray(values, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if raw.ndim != 1 or mask.shape != raw.shape:
        raise ValueError(f"{label} 数值和有效标记形状不一致")
    if int(mask.sum()) < 3:
        raise ValueError(f"{label} 单日有效样本不足 3 个")
    selected = raw[mask]  # 均值和标准差只由当日有效暴露估计。
    if not np.isfinite(selected).all():
        raise ValueError(f"{label} 有效暴露含 NaN 或 Inf")
    mean = float(selected.mean())
    std = float(selected.std(ddof=0))
    if not np.isfinite(std) or std <= np.finfo(np.float64).eps:
        raise ValueError(f"{label} 单日暴露为常数")
    result = np.zeros(len(raw), dtype=np.float64)  # 缺失暴露填零，但有效性仍由独立 mask 保存。
    result[mask] = (selected - mean) / std
    return result


def residualize_daily_targets(
    targets: np.ndarray,
    style_exposures: np.ndarray,
    label_valid: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """对单日标签联合回归十风格，返回保持单位尺度的 OLS 残差。"""

    y_all = np.asarray(targets, dtype=np.float64)
    styles = np.asarray(style_exposures, dtype=np.float64)
    valid = np.asarray(label_valid, dtype=bool)
    if styles.shape != (len(y_all), len(STYLE_COLUMNS)):
        raise ValueError("单日风格矩阵形状异常")
    if valid.shape != y_all.shape or int(valid.sum()) < len(STYLE_COLUMNS) + 2:
        raise ValueError("单日有效标签不足以进行十风格回归")
    if not np.isfinite(y_all[valid]).all() or not np.isfinite(styles).all():
        raise ValueError("单日标签或风格矩阵含 NaN/Inf")

    y = y_all[valid]  # 回归样本仅包含标签有效的股票。
    x = styles[valid]  # 风格暴露已按日标准化并完成缺失处理。
    design = np.column_stack([np.ones(len(y), dtype=np.float64), x])  # 组成 Barra 设计矩阵
    coefficients, _, rank, singular_values = np.linalg.lstsq(  # 求解当日截面 OLS
        design, y, rcond=None
    )
    expected_rank = design.shape[1]
    if int(rank) != expected_rank:
        raise np.linalg.LinAlgError(
            f"单日十风格设计矩阵秩为 {rank}，预期 {expected_rank}"
        )
    if singular_values[-1] <= np.finfo(np.float64).eps:
        raise np.linalg.LinAlgError("单日十风格设计矩阵条件数不可计算")

    fitted = design @ coefficients  # 标签中可被截距与十风格解释的部分。
    residual = y - fitted  # Alpha 标签取 OLS 正交补。
    residual_mean = float(residual.mean())
    residual_std = float(residual.std(ddof=0))
    if not np.isfinite(residual_std) or residual_std <= np.finfo(np.float64).eps:
        raise ValueError("单日风格残差标签为常数")
    residual = (residual - residual_mean) / residual_std  # 每日恢复为均值 0、标准差 1 的训练尺度。
    correlations = np.corrcoef(  # 数值验收：残差应与全部风格近似正交。
        np.column_stack([residual, x]), rowvar=False
    )[0, 1:]
    if not np.isfinite(correlations).all():
        raise ValueError("单日残差标签风格相关性不可计算")
    max_abs_correlation = float(np.max(np.abs(correlations)))
    if max_abs_correlation > ORTHOGONALITY_TOLERANCE:
        raise AssertionError(
            f"单日残差标签最大风格相关性 {max_abs_correlation:.3e} 超过容差"
        )

    output = np.full(len(y_all), np.nan, dtype=np.float64)  # 无效标签保持 NaN，不伪造监督信号。
    output[valid] = residual
    source_centered = y - y.mean()
    residual_variance = float(np.dot(residual, residual))
    source_variance = float(np.dot(source_centered, source_centered))
    raw_residual = y - fitted
    raw_residual_variance = float(np.dot(raw_residual, raw_residual))
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
    return output, diagnostics


def decompose_daily_targets_with_industry(
    targets: np.ndarray,
    style_exposures: np.ndarray,
    industry_codes: np.ndarray,
    label_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    """返回完整 Barra 回归的标准化残差和标准化拟合项。"""

    y_all = np.asarray(targets, dtype=np.float64)
    styles_all = np.asarray(style_exposures, dtype=np.float64)
    codes_all = np.asarray(industry_codes)
    valid = np.asarray(label_valid, dtype=bool)
    if styles_all.shape != (len(y_all), len(STYLE_COLUMNS)):
        raise ValueError("单日风格矩阵形状异常")
    if codes_all.shape != y_all.shape or not np.issubdtype(
        codes_all.dtype, np.integer
    ):
        raise ValueError("行业代码必须是一维整数数组且与标签等长")
    if valid.shape != y_all.shape:
        raise ValueError("单日标签有效标记形状异常")
    if int(valid.sum()) < len(STYLE_COLUMNS) + 2:
        raise ValueError("单日有效标签不足以进行完整 Barra 回归")
    if (codes_all < 0).any() or (
        codes_all > EXPECTED_CNE5S_INDUSTRIES
    ).any():
        raise ValueError("行业代码超出 0 至 32 的有效范围")
    if not np.isfinite(y_all[valid]).all() or not np.isfinite(
        styles_all
    ).all():
        raise ValueError("单日标签或风格矩阵含 NaN/Inf")

    y = y_all[valid]
    styles = styles_all[valid]
    codes = codes_all[valid]
    _, compact = np.unique(codes, return_inverse=True)  # 压缩当日行业编码
    counts = np.bincount(compact).astype(np.float64)
    y_means = np.bincount(compact, weights=y) / counts
    style_sums = np.zeros(
        (len(counts), len(STYLE_COLUMNS)), dtype=np.float64
    )
    np.add.at(style_sums, compact, styles)  # 按行业汇总截面统计量
    style_means = style_sums / counts[:, None]
    y_within = y - y_means[compact]  # 行业内去均值等价于吸收行业固定效应。
    styles_within = styles - style_means[compact]  # 风格也在行业内中心化后再回归。

    coefficients, _, rank, singular_values = np.linalg.lstsq(  # 求解当日截面 OLS
        styles_within, y_within, rcond=None
    )
    if int(rank) != len(STYLE_COLUMNS):
        raise np.linalg.LinAlgError(
            f"行业内十风格矩阵秩为 {rank}，预期 {len(STYLE_COLUMNS)}"
        )
    if singular_values[-1] <= np.finfo(np.float64).eps:
        raise np.linalg.LinAlgError("行业内十风格矩阵条件数不可计算")

    raw_residual = y_within - styles_within @ coefficients  # 风格与行业均无法解释的 Alpha 部分。
    raw_fitted = y - raw_residual  # Style 塔监督目标为完整 Barra 拟合项。
    reconstruction_error = float(
        np.max(np.abs(y - raw_fitted - raw_residual))
    )
    if reconstruction_error > 1e-12:
        raise AssertionError("完整 Barra 拟合项与残差无法重构源标签")
    raw_std = float(raw_residual.std(ddof=0))
    if not np.isfinite(raw_std) or raw_std <= np.finfo(np.float64).eps:
        raise ValueError("单日完整 Barra 残差标签为常数")
    fitted_std = float(raw_fitted.std(ddof=0))
    if not np.isfinite(fitted_std) or fitted_std <= np.finfo(np.float64).eps:
        raise ValueError("单日完整 Barra 拟合标签为常数")
    residual = (raw_residual - raw_residual.mean()) / raw_std  # Alpha 目标按日标准化。
    fitted = (raw_fitted - raw_fitted.mean()) / fitted_std  # Style 目标独立按日标准化。
    correlations = np.corrcoef(
        np.column_stack([residual, styles]), rowvar=False
    )[0, 1:]
    if not np.isfinite(correlations).all():
        raise ValueError("单日完整 Barra 残差风格相关性不可计算")
    max_abs_correlation = float(np.max(np.abs(correlations)))
    group_residual_means = (  # 行业固定效应已吸收时，各行业残差均值应接近零。
        np.bincount(compact, weights=residual) / counts
    )
    max_abs_industry_mean = float(np.max(np.abs(group_residual_means)))
    if max(max_abs_correlation, max_abs_industry_mean) > ORTHOGONALITY_TOLERANCE:
        raise AssertionError(
            "单日完整 Barra 残差未达到数值正交："
            f"style={max_abs_correlation:.3e}, "
            f"industry={max_abs_industry_mean:.3e}"
        )

    centered = y - y.mean()
    total_sum_squares = float(np.dot(centered, centered))
    residual_sum_squares = float(np.dot(raw_residual, raw_residual))
    if total_sum_squares <= np.finfo(np.float64).eps:
        raise ValueError("单日源标签为常数")
    output = np.full(len(y_all), np.nan, dtype=np.float64)
    fitted_output = np.full(len(y_all), np.nan, dtype=np.float64)
    output[valid] = residual
    fitted_output[valid] = fitted
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


def residualize_daily_targets_with_industry(
    targets: np.ndarray,
    style_exposures: np.ndarray,
    industry_codes: np.ndarray,
    label_valid: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """对单日标签回归十风格和行业固定效应，再标准化残差。"""

    residual, _fitted, diagnostics = decompose_daily_targets_with_industry(
        targets, style_exposures, industry_codes, label_valid
    )
    return residual, diagnostics


def project_out_styles_numpy(
    predictions: np.ndarray, style_exposures: np.ndarray
) -> tuple[np.ndarray, dict[str, float | int]]:
    """NumPy 参考实现：从单日预测中联合投影掉十个风格。"""

    values = np.asarray(predictions, dtype=np.float64)
    styles = np.asarray(style_exposures, dtype=np.float64)
    if values.ndim != 1 or styles.shape != (len(values), len(STYLE_COLUMNS)):
        raise ValueError("预测或风格矩阵形状异常")
    if len(values) < len(STYLE_COLUMNS) + 2:
        raise ValueError("单日股票数不足以进行十风格投影")
    if not np.isfinite(values).all() or not np.isfinite(styles).all():
        raise ValueError("预测或风格矩阵含 NaN/Inf")
    design = np.column_stack([np.ones(len(values)), styles])  # 组成 Barra 设计矩阵
    coefficients, _, rank, singular_values = np.linalg.lstsq(  # 求解当日截面 OLS
        design, values, rcond=None
    )
    if int(rank) != design.shape[1]:
        raise np.linalg.LinAlgError("十风格投影矩阵秩不足")
    residual = values - design @ coefficients
    correlations = np.corrcoef(
        np.column_stack([residual, styles]), rowvar=False
    )[0, 1:]
    return residual, {
        "matrix_rank": int(rank),
        "condition_number": float(singular_values[0] / singular_values[-1]),
        "max_abs_style_pearson": float(np.max(np.abs(correlations))),
        "residual_mean": float(residual.mean()),
    }


def project_out_styles_torch(
    predictions: torch.Tensor,
    style_exposures: torch.Tensor,
    *,
    standardize: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """以 FP32 QR 将单日模型分数投影到十风格正交补，并可恢复单位尺度。"""

    values = predictions.float().reshape(-1)
    styles = style_exposures.float()
    if styles.shape != (len(values), len(STYLE_COLUMNS)):
        raise ValueError("预测或风格矩阵形状异常")
    if len(values) < len(STYLE_COLUMNS) + 2:
        raise ValueError("单日股票数不足以进行十风格投影")
    if not torch.isfinite(values).all() or not torch.isfinite(styles).all():
        raise ValueError("预测或风格矩阵含 NaN/Inf")

    design = torch.column_stack(  # 组成 Barra 设计矩阵
        [torch.ones_like(values, dtype=torch.float32), styles]
    )
    q, r = torch.linalg.qr(design, mode="reduced")  # QR 分解得到稳定投影基
    diagonal = torch.abs(torch.diagonal(r))
    tolerance = (
        torch.finfo(torch.float32).eps
        * max(design.shape)
        * diagonal.max()
    )
    if bool((diagonal <= tolerance).any()):
        raise np.linalg.LinAlgError("十风格投影矩阵秩不足")

    residual = values - q @ (q.transpose(0, 1) @ values)  # 从分数中减去其在 Barra 子空间上的投影。
    residual = residual - residual.mean()
    residual_std = residual.std(unbiased=False)
    if not torch.isfinite(residual_std) or bool(
        residual_std <= torch.finfo(torch.float32).eps
    ):
        raise ValueError("单日投影残差为常数")
    effective = residual / residual_std if standardize else residual  # 训练时通常恢复单位截面波动。
    centered_styles = styles - styles.mean(dim=0, keepdim=True)
    correlations = (
        centered_styles.transpose(0, 1) @ effective
    ) / torch.sqrt(
        centered_styles.square().sum(dim=0)
        * effective.square().sum()
    )
    if not torch.isfinite(correlations).all():
        raise ValueError("投影后风格相关性不可计算")
    return effective, {
        "residual_mean": effective.mean().detach(),
        "residual_std": effective.std(unbiased=False).detach(),
        "max_abs_style_pearson": correlations.abs().max().detach(),
    }


def joint_barra_r_squared_numpy(
    predictions: np.ndarray,
    style_exposures: np.ndarray,
    industry_codes: np.ndarray,
) -> tuple[float, dict[str, float | int]]:
    """计算 raw score 对十风格和行业固定效应的单日联合 R²。"""

    values = np.asarray(predictions, dtype=np.float64)
    styles = np.asarray(style_exposures, dtype=np.float64)
    codes = np.asarray(industry_codes)
    if values.ndim != 1 or styles.shape != (len(values), len(STYLE_COLUMNS)):
        raise ValueError("预测或十风格矩阵形状异常")
    if codes.shape != values.shape or not np.issubdtype(codes.dtype, np.integer):
        raise ValueError("行业代码必须是一维整数数组且与预测等长")
    if len(values) < len(STYLE_COLUMNS) + 2:
        raise ValueError("单日股票数不足以计算联合 Barra R²")
    if (codes < 0).any():
        raise ValueError("行业代码不能为负数")
    if not np.isfinite(values).all() or not np.isfinite(styles).all():
        raise ValueError("预测或十风格矩阵含 NaN/Inf")

    _, compact = np.unique(codes, return_inverse=True)  # 压缩当日行业编码
    counts = np.bincount(compact).astype(np.float64)
    value_means = np.bincount(compact, weights=values) / counts
    style_sums = np.zeros((len(counts), len(STYLE_COLUMNS)), dtype=np.float64)
    np.add.at(style_sums, compact, styles)  # 按行业汇总截面统计量
    style_means = style_sums / counts[:, None]
    values_within = values - value_means[compact]  # 先吸收行业固定效应。
    styles_within = styles - style_means[compact]

    coefficients, _, rank, singular_values = np.linalg.lstsq(  # 求解当日截面 OLS
        styles_within, values_within, rcond=None
    )
    if int(rank) != len(STYLE_COLUMNS):
        raise np.linalg.LinAlgError(
            f"行业内十风格矩阵秩为 {rank}，预期 {len(STYLE_COLUMNS)}"
        )
    residual = values_within - styles_within @ coefficients
    centered = values - values.mean()
    total_sum_squares = float(np.dot(centered, centered))
    if total_sum_squares <= np.finfo(np.float64).eps:
        raise ValueError("单日 raw score 为常数")
    residual_sum_squares = float(np.dot(residual, residual))
    r_squared = 1.0 - residual_sum_squares / total_sum_squares  # 联合 Barra 暴露解释的分数方差比例。
    tolerance = 1e-10
    if not np.isfinite(r_squared) or not -tolerance <= r_squared <= 1.0 + tolerance:
        raise FloatingPointError(f"联合 Barra R² 超出数值范围：{r_squared}")
    r_squared = float(np.clip(r_squared, 0.0, 1.0))
    group_residual_means = np.bincount(compact, weights=residual) / counts
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


def joint_barra_r_squared_torch(
    predictions: torch.Tensor,
    style_exposures: torch.Tensor,
    industry_codes: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """可微联合 R²；Barra/行业固定，梯度只回传到 raw score。"""

    device_type = predictions.device.type
    with torch.autocast(device_type=device_type, enabled=False):  # QR 与 R² 固定用 FP32，避免混合精度放大秩误差。
        values = predictions.float().reshape(-1)
        styles = style_exposures.float()
        codes = industry_codes.reshape(-1)
        integer_dtypes = (
            torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8
        )
        if styles.shape != (len(values), len(STYLE_COLUMNS)):
            raise ValueError("预测或十风格矩阵形状异常")
        if codes.shape != values.shape or codes.dtype not in integer_dtypes:
            raise ValueError("行业代码必须是一维整数张量且与预测等长")
        if len(values) < len(STYLE_COLUMNS) + 2:
            raise ValueError("单日股票数不足以计算联合 Barra R²")
        if bool((codes < 0).any()):
            raise ValueError("行业代码不能为负数")
        if not bool(torch.isfinite(values).all()) or not bool(
            torch.isfinite(styles).all()
        ):
            raise ValueError("预测或十风格矩阵含 NaN/Inf")

        _, compact = torch.unique(codes.long(), sorted=True, return_inverse=True)  # 压缩当日行业编码
        group_count = int(compact.max().item()) + 1
        counts = torch.zeros(group_count, dtype=torch.float32, device=values.device)
        counts.scatter_add_(0, compact, torch.ones_like(values))
        value_sums = torch.zeros_like(counts).scatter_add(0, compact, values)
        style_sums = torch.zeros(
            group_count,
            len(STYLE_COLUMNS),
            dtype=torch.float32,
            device=values.device,
        )
        style_sums.index_add_(0, compact, styles)
        values_within = values - (value_sums / counts)[compact]
        styles_within = styles - (style_sums / counts[:, None])[compact]

        q, r = torch.linalg.qr(styles_within, mode="reduced")  # QR 分解得到稳定投影基
        diagonal = torch.abs(torch.diagonal(r))
        rank_tolerance = (
            torch.finfo(torch.float32).eps
            * max(styles_within.shape)
            * diagonal.max()
        )
        if bool((diagonal <= rank_tolerance).any()):
            raise np.linalg.LinAlgError("行业内十风格矩阵秩不足")
        residual = values_within - q @ (q.transpose(0, 1) @ values_within)
        centered = values - values.mean()
        total_sum_squares = centered.square().sum()
        if not bool(torch.isfinite(total_sum_squares)) or bool(
            total_sum_squares <= torch.finfo(torch.float32).eps
        ):
            raise ValueError("单日 raw score 为常数")
        residual_sum_squares = residual.square().sum()
        raw_r_squared = 1.0 - residual_sum_squares / total_sum_squares
        if not bool(torch.isfinite(raw_r_squared)) or bool(
            (raw_r_squared < -1e-4) | (raw_r_squared > 1.0 + 1e-4)
        ):
            raise FloatingPointError(
                f"联合 Barra R² 超出数值范围：{float(raw_r_squared.detach())}"
            )
        r_squared = raw_r_squared.clamp(0.0, 1.0)  # 约束数值误差后的统计量
        residual_sums = torch.zeros_like(counts).scatter_add(
            0, compact, residual
        )
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


def project_out_full_barra_torch(
    predictions: torch.Tensor,
    style_exposures: torch.Tensor,
    industry_codes: torch.Tensor,
    *,
    standardize: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """投影掉十个连续 Barra 暴露与行业固定效应，梯度回传到分数。"""

    device_type = predictions.device.type
    with torch.autocast(device_type=device_type, enabled=False):  # QR 与 R² 固定用 FP32，避免混合精度放大秩误差。
        values = predictions.float().reshape(-1)
        styles = style_exposures.float()
        codes = industry_codes.reshape(-1)
        integer_dtypes = (
            torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8
        )
        if styles.shape != (len(values), len(STYLE_COLUMNS)):
            raise ValueError("预测或十风格矩阵形状异常")
        if codes.shape != values.shape or codes.dtype not in integer_dtypes:
            raise ValueError("行业代码必须是一维整数张量且与预测等长")
        if len(values) < len(STYLE_COLUMNS) + 2:
            raise ValueError("单日股票数不足以进行完整 Barra 投影")
        if bool(
            ((codes < 0) | (codes > EXPECTED_CNE5S_INDUSTRIES)).any()
        ):
            raise ValueError("行业代码超出 0 至 32 的有效范围")
        if not bool(torch.isfinite(values).all()) or not bool(
            torch.isfinite(styles).all()
        ):
            raise ValueError("预测或十风格矩阵含 NaN/Inf")

        _, compact = torch.unique(  # 压缩当日行业编码
            codes.long(), sorted=True, return_inverse=True
        )
        group_count = int(compact.max().item()) + 1
        counts = torch.zeros(
            group_count, dtype=torch.float32, device=values.device
        )
        counts.scatter_add_(0, compact, torch.ones_like(values))
        value_sums = torch.zeros_like(counts).scatter_add(
            0, compact, values
        )
        style_sums = torch.zeros(
            group_count,
            len(STYLE_COLUMNS),
            dtype=torch.float32,
            device=values.device,
        )
        style_sums.index_add_(0, compact, styles)
        values_within = values - (value_sums / counts)[compact]
        styles_within = styles - (style_sums / counts[:, None])[compact]

        q, r = torch.linalg.qr(styles_within, mode="reduced")  # QR 分解得到稳定投影基
        diagonal = torch.abs(torch.diagonal(r))
        rank_tolerance = (
            torch.finfo(torch.float32).eps
            * max(styles_within.shape)
            * diagonal.max()
        )
        if bool((diagonal <= rank_tolerance).any()):
            raise np.linalg.LinAlgError("行业内十风格矩阵秩不足")
        residual = values_within - q @ (
            q.transpose(0, 1) @ values_within
        )
        if standardize:
            centered = residual - residual.mean()
            standard_deviation = centered.std(unbiased=False)
            if not bool(torch.isfinite(standard_deviation)) or bool(
                standard_deviation <= torch.finfo(torch.float32).eps
            ):
                raise ValueError("完整 Barra 投影后的分数为常数")
            output = centered / standard_deviation
        else:
            output = residual

        style_centered = styles - styles.mean(dim=0, keepdim=True)
        denominator = torch.sqrt(  # 计算截面标准差
            output.square().sum()
            * style_centered.square().sum(dim=0)
        )
        if bool((denominator <= torch.finfo(torch.float32).eps).any()):
            raise ValueError("完整 Barra 投影后的风格相关性不可计算")
        correlations = (
            output[:, None] * style_centered
        ).sum(dim=0) / denominator
        residual_sums = torch.zeros_like(counts).scatter_add(
            0, compact, output
        )
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


def _industry_sidecar_payload(
    sample_index_path: Path,
    cne5s_path: Path,
    style_store_signature: str,
    minimum_match_rate: float,
) -> dict[str, Any]:
    """处理行业代码或行业诊断，并保留 UNKNOWN 口径。"""
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


def build_industry_sidecar(
    *,
    store_dir: str | Path = DEFAULT_INDUSTRY_STORE_DIR,
    sample_index_path: str | Path = base.SAMPLE_INDEX_PATH,
    cne5s_path: str | Path = barra.DEFAULT_CNE5S_PATH,
    minimum_match_rate: float = MINIMUM_MATCH_RATE,
    overwrite: bool = False,
) -> dict[str, Any]:
    """构建与 sample_index 严格对齐的 CNE5S 行业整数代码。"""

    if not 0 <= minimum_match_rate <= 1:
        raise ValueError("minimum_match_rate 必须位于 [0,1]")
    target = Path(store_dir).resolve()
    sample_path = Path(sample_index_path).resolve()
    cne5s = Path(cne5s_path).resolve()
    style_manifest = require_style_neutral_store()
    payload = _industry_sidecar_payload(
        sample_path, cne5s, style_manifest["signature"], minimum_match_rate
    )
    signature = _stable_hash(payload)  # 生成数据复用签名
    manifest_path = target / INDUSTRY_MANIFEST_FILE
    if manifest_path.exists() and not overwrite:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("signature") != signature:
            raise RuntimeError(
                "已有行业 sidecar 签名不同；请显式设置 overwrite=True"
            )
        _validate_industry_sidecar(target, existing)
        return {**existing, "reused": True}

    sample = pd.read_parquet(  # sidecar 必须逐行对应 feature-store sample_index。
        sample_path,
        columns=["sample_index", "trade_date", "instrument_id"],
    )
    sample["trade_date"] = sample["trade_date"].astype(str)
    sample["instrument_id"] = sample["instrument_id"].astype(str)
    expected_indices = np.arange(len(sample), dtype=np.int64)
    if not np.array_equal(
        sample["sample_index"].to_numpy(np.int64), expected_indices
    ):
        raise AssertionError("sample_index 必须与物理行号逐行一致")

    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.stage.", dir=target.parent)
    )
    try:
        output = np.lib.format.open_memmap(  # 使用 memmap，避免整库载入内存
            stage / INDUSTRY_CODES_FILE,
            mode="w+",
            dtype=np.int16,
            shape=(len(sample),),
        )
        output[:] = -1  # -1 仅用于检测尚未写入的样本。
        mapping: dict[str, int] = {barra.UNKNOWN_INDUSTRY: 0}  # 0 固定表示未知行业，供 Embedding padding_idx 使用。
        audit: list[dict[str, Any]] = []
        years = sorted(sample["trade_date"].str[:4].astype(int).unique())
        for year in years:
            yearly = sample.loc[
                sample["trade_date"].str.startswith(str(year))
            ].copy()
            yearly["assetid"] = barra.instrument_ids_to_assetids(  # 转换为 CNE5S assetid
                yearly["instrument_id"]
            )
            cne5s_year = barra._load_cne5s_year(cne5s, year)
            joined = barra._join_prediction_and_cne5s(
                yearly, cne5s_year, year
            )
            matched = joined["barra_matched"].to_numpy(bool)  # 按股票与交易日业务键对齐
            match_rate = float(matched.mean())
            if match_rate < minimum_match_rate:
                raise AssertionError(
                    f"{year} CNE5S 匹配率 {match_rate:.6%} "
                    f"< {minimum_match_rate:.6%}"
                )
            industries = joined[barra.INDUSTRY_COLUMN].astype("object").copy()
            industries.loc[~joined["barra_matched"]] = barra.UNKNOWN_INDUSTRY
            if industries.isna().any():
                raise ValueError(f"{year} 行业字段含未处理缺失值")
            labels = industries.astype(str)
            new_labels = sorted(set(labels.unique()).difference(mapping))  # 已有代码不变，只为首次出现的行业分配新码。
            for label in new_labels:
                mapping[label] = len(mapping)
            encoded = labels.map(mapping)
            if encoded.isna().any():
                raise AssertionError(f"{year} 存在未编码行业")
            sample_indices = joined["sample_index"].to_numpy(np.int64)  # 按股票与交易日业务键对齐
            output[sample_indices] = encoded.to_numpy(np.int16)  # 按全局样本行号写回，保持 sidecar 逐行对齐。
            audit.append(
                {
                    "year": int(year),
                    "rows": len(joined),
                    "matched_rows": int(matched.sum()),
                    "match_rate": match_rate,
                    "known_industries_to_date": len(mapping) - 1,
                }
            )

        output.flush()  # 将 memmap 修改刷入磁盘
        del output
        codes = np.load(stage / INDUSTRY_CODES_FILE, mmap_mode="r")
        if (codes < 0).any():
            raise AssertionError("行业 sidecar 存在未写入样本")
        known_count = len(mapping) - 1
        if known_count != EXPECTED_CNE5S_INDUSTRIES:
            raise ValueError(
                f"CNE5S 行业数为 {known_count}，预期 {EXPECTED_CNE5S_INDUSTRIES}"
            )
        pd.DataFrame(audit).to_parquet(
            stage / INDUSTRY_AUDIT_FILE, index=False, compression="zstd"
        )
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
        (stage / INDUSTRY_MANIFEST_FILE).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _validate_industry_sidecar(stage, manifest)
        _atomic_replace_directory(stage, target)  # 原子落盘，避免半成品
        return {**manifest, "reused": False}
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def _full_barra_target_payload(
    targets_path: Path,
    style_store_signature: str,
    industry_sidecar_signature: str,
) -> dict[str, Any]:
    """构造或校验完整 Barra 风格加行业的数据契约。"""
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


def build_full_barra_target_sidecar(
    *,
    store_dir: str | Path = DEFAULT_FULL_BARRA_TARGET_STORE_DIR,
    sample_index_path: str | Path = base.SAMPLE_INDEX_PATH,
    targets_path: str | Path = base.TARGETS_PATH,
    overwrite: bool = False,
) -> dict[str, Any]:
    """逐日生成截距、十风格与行业固定效应联合残差标签。"""

    target = Path(store_dir).resolve()
    sample_path = Path(sample_index_path).resolve()
    source_targets_path = Path(targets_path).resolve()
    style_manifest = require_style_neutral_store()
    industry_manifest = require_industry_sidecar()
    payload = _full_barra_target_payload(
        source_targets_path,
        style_manifest["signature"],
        industry_manifest["signature"],
    )
    signature = _stable_hash(payload)  # 生成数据复用签名
    manifest_path = target / FULL_BARRA_TARGET_MANIFEST_FILE
    if manifest_path.exists() and not overwrite:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("signature") != signature:
            raise RuntimeError(
                "已有完整 Barra 残差标签签名不同；"
                "请显式设置 overwrite=True"
            )
        _validate_full_barra_target_sidecar(target, existing)
        return {**existing, "reused": True}

    sample = pd.read_parquet(
        sample_path,
        columns=["sample_index", "trade_date", "label_valid"],
    )
    sample["trade_date"] = sample["trade_date"].astype(str)
    expected_indices = np.arange(len(sample), dtype=np.int64)
    if not np.array_equal(
        sample["sample_index"].to_numpy(np.int64), expected_indices
    ):
        raise AssertionError("sample_index 必须与物理行号逐行一致")
    label_valid = sample["label_valid"].to_numpy(bool)
    source_targets = np.load(source_targets_path, mmap_mode="r")
    styles = np.load(
        DEFAULT_STORE_DIR / STYLE_EXPOSURES_FILE, mmap_mode="r"
    )
    industry_codes = np.load(
        DEFAULT_INDUSTRY_STORE_DIR / INDUSTRY_CODES_FILE, mmap_mode="r"
    )
    if source_targets.shape != (len(sample),):
        raise ValueError("源 targets.npy 与 sample_index 行数不一致")
    if styles.shape != (len(sample), len(STYLE_COLUMNS)):
        raise ValueError("十风格矩阵与 sample_index 行数不一致")
    if industry_codes.shape != (len(sample),):
        raise ValueError("行业代码与 sample_index 行数不一致")
    if not np.array_equal(np.isfinite(source_targets), label_valid):
        raise AssertionError("源标签有限性与 label_valid 不一致")

    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.stage.", dir=target.parent)
    )
    try:
        output = np.lib.format.open_memmap(  # 使用 memmap，避免整库载入内存
            stage / FULL_BARRA_TARGETS_FILE,
            mode="w+",
            dtype=np.float32,
            shape=(len(sample),),
        )
        output[:] = np.nan
        diagnostics: list[dict[str, Any]] = []
        years = sorted(sample["trade_date"].str[:4].astype(int).unique())
        for year in years:
            yearly = sample.loc[
                sample["trade_date"].str.startswith(str(year))
            ]
            for trade_date, positions in yearly.groupby(  # 每个交易日独立拟合，禁止跨日使用暴露或标签。
                "trade_date", sort=True
            ).groups.items():
                daily = sample.loc[positions]
                indices = daily["sample_index"].to_numpy(np.int64)
                residual, record = residualize_daily_targets_with_industry(  # 按日投影 Barra 暴露
                    np.asarray(source_targets[indices], dtype=np.float64),
                    np.asarray(styles[indices], dtype=np.float64),
                    np.asarray(industry_codes[indices], dtype=np.int16),
                    label_valid[indices],
                )
                output[indices] = residual.astype(np.float32)
                diagnostics.append(
                    {
                        "year": int(year),
                        "trade_date": str(trade_date),
                        **record,
                    }
                )
            print(
                f"{year}：写入完整 Barra 残差标签 {len(yearly):,} 行",
                flush=True,
            )

        output.flush()  # 将 memmap 修改刷入磁盘
        del output
        residual_check = np.load(
            stage / FULL_BARRA_TARGETS_FILE, mmap_mode="r"
        )
        if not np.array_equal(np.isfinite(residual_check), label_valid):
            raise AssertionError(
                "完整 Barra 残差标签有限性与 label_valid 不一致"
            )
        daily = pd.DataFrame(diagnostics)
        daily.to_parquet(
            stage / FULL_BARRA_DAILY_DIAGNOSTICS_FILE,
            index=False,
            compression="zstd",
        )
        annual = daily.groupby("year", as_index=False).agg(  # 按既定口径聚合
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
        annual.to_parquet(
            stage / FULL_BARRA_ANNUAL_SUMMARY_FILE,
            index=False,
            compression="zstd",
        )
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
        (stage / FULL_BARRA_TARGET_MANIFEST_FILE).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _validate_full_barra_target_sidecar(stage, manifest)
        _atomic_replace_directory(stage, target)  # 原子落盘，避免半成品
        return {**manifest, "reused": False}
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def _dual_tower_target_payload(
    targets_path: Path,
    full_target_signature: str,
    style_store_signature: str,
    industry_sidecar_signature: str,
) -> dict[str, Any]:
    """构造或校验 Alpha/Style 双塔所需的数据契约。"""
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


def build_dual_tower_target_sidecar(
    *,
    store_dir: str | Path = DEFAULT_DUAL_TOWER_TARGET_STORE_DIR,
    sample_index_path: str | Path = base.SAMPLE_INDEX_PATH,
    targets_path: str | Path = base.TARGETS_PATH,
    overwrite: bool = False,
) -> dict[str, Any]:
    """逐日生成完整 Barra 回归拟合收益，供 No.6 Style 塔监督。"""

    target = Path(store_dir).resolve()
    sample_path = Path(sample_index_path).resolve()
    source_targets_path = Path(targets_path).resolve()
    style_manifest = require_style_neutral_store()
    industry_manifest = require_industry_sidecar()
    full_manifest = require_full_barra_target_sidecar()
    payload = _dual_tower_target_payload(
        source_targets_path,
        full_manifest["signature"],
        style_manifest["signature"],
        industry_manifest["signature"],
    )
    signature = _stable_hash(payload)  # 生成数据复用签名
    manifest_path = target / DUAL_TOWER_TARGET_MANIFEST_FILE
    if manifest_path.exists() and not overwrite:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("signature") != signature:
            raise RuntimeError(
                "已有 No.6 fitted-target sidecar 签名不同；"
                "请显式设置 overwrite=True"
            )
        _validate_dual_tower_target_sidecar(target, existing)
        return {**existing, "reused": True}

    sample = pd.read_parquet(
        sample_path, columns=["sample_index", "trade_date", "label_valid"]
    )
    sample["trade_date"] = sample["trade_date"].astype(str)
    expected_indices = np.arange(len(sample), dtype=np.int64)
    if not np.array_equal(
        sample["sample_index"].to_numpy(np.int64), expected_indices
    ):
        raise AssertionError("sample_index 必须与物理行号逐行一致")
    label_valid = sample["label_valid"].to_numpy(bool)
    source_targets = np.load(source_targets_path, mmap_mode="r")
    styles = np.load(DEFAULT_STORE_DIR / STYLE_EXPOSURES_FILE, mmap_mode="r")
    industries = np.load(
        DEFAULT_INDUSTRY_STORE_DIR / INDUSTRY_CODES_FILE, mmap_mode="r"
    )
    existing_residual = np.load(
        DEFAULT_FULL_BARRA_TARGET_STORE_DIR / FULL_BARRA_TARGETS_FILE,
        mmap_mode="r",
    )
    expected_shapes = (
        source_targets.shape == (len(sample),),
        styles.shape == (len(sample), len(STYLE_COLUMNS)),
        industries.shape == (len(sample),),
        existing_residual.shape == (len(sample),),
    )
    if not all(expected_shapes):
        raise ValueError("No.6 标签输入与 sample_index 形状不一致")

    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.stage.", dir=target.parent)
    )
    try:
        output = np.lib.format.open_memmap(  # 使用 memmap，避免整库载入内存
            stage / DUAL_TOWER_FITTED_TARGETS_FILE,
            mode="w+",
            dtype=np.float32,
            shape=(len(sample),),
        )
        output[:] = np.nan
        diagnostics: list[dict[str, Any]] = []
        years = sorted(sample["trade_date"].str[:4].astype(int).unique())
        for year in years:
            yearly = sample.loc[sample["trade_date"].str.startswith(str(year))]
            for trade_date, positions in yearly.groupby(  # 每个交易日独立拟合，禁止跨日使用暴露或标签。
                "trade_date", sort=True
            ).groups.items():
                indices = sample.loc[positions, "sample_index"].to_numpy(np.int64)
                residual, fitted, record = decompose_daily_targets_with_industry(
                    np.asarray(source_targets[indices], dtype=np.float64),
                    np.asarray(styles[indices], dtype=np.float64),
                    np.asarray(industries[indices], dtype=np.int16),
                    label_valid[indices],
                )
                valid = label_valid[indices]
                residual_error = float(
                    np.max(
                        np.abs(
                            residual[valid]
                            - np.asarray(existing_residual[indices][valid], dtype=np.float64)
                        )
                    )
                )
                fitted_r_squared, _ = joint_barra_r_squared_numpy(  # 按股票与交易日业务键对齐
                    fitted[valid],
                    np.asarray(styles[indices][valid], dtype=np.float64),
                    np.asarray(industries[indices][valid], dtype=np.int16),
                )
                output[indices] = fitted.astype(np.float32)
                diagnostics.append(
                    {
                        "year": int(year),
                        "trade_date": str(trade_date),
                        **record,
                        "residual_sidecar_max_abs_error": residual_error,
                        "fitted_joint_barra_r_squared": fitted_r_squared,
                    }
                )
            print(
                f"{year}：写入 No.6 fitted target {len(yearly):,} 行",
                flush=True,
            )
        output.flush()  # 将 memmap 修改刷入磁盘
        del output
        fitted_check = np.load(
            stage / DUAL_TOWER_FITTED_TARGETS_FILE, mmap_mode="r"
        )
        if not np.array_equal(np.isfinite(fitted_check), label_valid):
            raise AssertionError("No.6 fitted target 有限性与 label_valid 不一致")
        daily = pd.DataFrame(diagnostics)
        daily.to_parquet(
            stage / DUAL_TOWER_DAILY_DIAGNOSTICS_FILE,
            index=False,
            compression="zstd",
        )
        annual = daily.groupby("year", as_index=False).agg(  # 按既定口径聚合
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
        annual.to_parquet(
            stage / DUAL_TOWER_ANNUAL_SUMMARY_FILE,
            index=False,
            compression="zstd",
        )
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
        (stage / DUAL_TOWER_TARGET_MANIFEST_FILE).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _validate_dual_tower_target_sidecar(stage, manifest)
        _atomic_replace_directory(stage, target)  # 原子落盘，避免半成品
        return {**manifest, "reused": False}
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def _atomic_replace_directory(stage: Path, target: Path) -> None:
    """通过临时文件和原子替换写入产物，避免产生半成品。"""
    backup = target.with_name(
        f".{target.name}.backup.{os.getpid()}.{time.time_ns()}"
    )
    if target.exists():
        target.rename(backup)
    try:
        stage.rename(target)
    except BaseException:
        if backup.exists() and not target.exists():
            backup.rename(target)
        raise
    if backup.exists():
        shutil.rmtree(backup)


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
    """构建与 feature store 行号严格对齐的十风格矩阵和残差标签。"""

    if not 0 <= minimum_match_rate <= 1:
        raise ValueError("minimum_match_rate 必须位于 [0,1]")
    target = Path(store_dir).resolve()
    sample_path = Path(sample_index_path).resolve()
    source_targets_path = Path(targets_path).resolve()
    cne5s = Path(cne5s_path).resolve()
    eod = Path(eod_path).resolve()
    calendar_file = Path(calendar_path).resolve()
    payload = _store_payload(
        sample_path, source_targets_path, cne5s, eod, calendar_file
    )
    signature = _stable_hash(payload)  # 生成数据复用签名
    manifest_path = target / MANIFEST_FILE
    if manifest_path.exists() and not overwrite:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("signature") != signature:
            raise RuntimeError(
                "已有风格中性训练数据签名不同；请显式设置 overwrite=True"
            )
        _validate_arrays(target, int(existing["sample_count"]))
        return {**existing, "reused": True}

    sample = pd.read_parquet(
        sample_path,
        columns=[
            "sample_index", "trade_date", "instrument_id", "label_valid"
        ],
    )
    sample["trade_date"] = sample["trade_date"].astype(str)
    sample["instrument_id"] = sample["instrument_id"].astype(str)
    expected_indices = np.arange(len(sample), dtype=np.int64)
    if not np.array_equal(
        sample["sample_index"].to_numpy(np.int64), expected_indices
    ):
        raise AssertionError("sample_index 必须与物理行号逐行一致")
    source_targets = np.load(source_targets_path, mmap_mode="r")
    if source_targets.shape != (len(sample),):
        raise ValueError("源 targets.npy 与 sample_index 行数不一致")
    label_valid = sample["label_valid"].to_numpy(bool)
    if not np.array_equal(np.isfinite(source_targets), label_valid):
        raise AssertionError("源标签有限性与 label_valid 不一致")

    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.stage.", dir=target.parent)
    )
    try:
        styles_output = np.lib.format.open_memmap(  # 使用 memmap，避免整库载入内存
            stage / STYLE_EXPOSURES_FILE,
            mode="w+",
            dtype=np.float32,
            shape=(len(sample), len(STYLE_COLUMNS)),
        )
        styles_output[:] = np.nan
        valid_output = np.lib.format.open_memmap(  # 使用 memmap，避免整库载入内存
            stage / STYLE_VALID_FILE,
            mode="w+",
            dtype=np.bool_,
            shape=(len(sample), len(STYLE_COLUMNS)),
        )
        valid_output[:] = False
        targets_output = np.lib.format.open_memmap(  # 使用 memmap，避免整库载入内存
            stage / RESIDUAL_TARGETS_FILE,
            mode="w+",
            dtype=np.float32,
            shape=(len(sample),),
        )
        targets_output[:] = np.nan

        calendar_dates = barra._load_sse_calendar(calendar_file)
        diagnostics: list[dict[str, Any]] = []
        years = sorted(sample["trade_date"].str[:4].astype(int).unique())
        for year in years:
            yearly = sample.loc[
                sample["trade_date"].str.startswith(str(year))
            ].copy()
            yearly["assetid"] = barra.instrument_ids_to_assetids(  # 转换为 CNE5S assetid
                yearly["instrument_id"]
            )
            yearly = barra._attach_reversal_20d(  # 按股票与日期业务键补充字段
                yearly, eod, calendar_dates, year
            )
            cne5s_year = barra._load_cne5s_year(cne5s, year)
            joined = barra._join_prediction_and_cne5s(
                yearly, cne5s_year, year
            )
            match_rate = float(joined["barra_matched"].mean())
            if match_rate < minimum_match_rate:
                raise AssertionError(
                    f"{year} CNE5S 匹配率 {match_rate:.6%} "
                    f"< {minimum_match_rate:.6%}"
                )
            for trade_date, positions in joined.groupby(
                "trade_date", sort=True
            ).groups.items():
                daily = joined.iloc[np.asarray(positions, dtype=np.int64)]
                sample_indices = daily["sample_index"].to_numpy(np.int64)
                rows = len(daily)
                style_matrix = np.zeros(
                    (rows, len(STYLE_COLUMNS)), dtype=np.float64
                )
                style_valid = np.zeros_like(style_matrix, dtype=bool)
                matched = daily["barra_matched"].to_numpy(bool)
                for column_index, column in enumerate(barra.STYLE_COLUMNS):
                    raw = daily[column].to_numpy(np.float64)
                    valid = matched & np.isfinite(raw)
                    style_matrix[:, column_index] = _standardize_with_missing(  # 仅使用当日截面统计量
                        raw, valid, label=f"{trade_date} {column}"
                    )
                    style_valid[:, column_index] = valid
                reversal_index = len(STYLE_COLUMNS) - 1
                reversal_valid = daily[
                    barra.REVERSAL_VALID_COLUMN
                ].to_numpy(bool)
                style_matrix[:, reversal_index] = _standardize_with_missing(  # 仅使用当日截面统计量
                    daily[barra.REVERSAL_COLUMN].to_numpy(np.float64),
                    reversal_valid,
                    label=f"{trade_date} {barra.REVERSAL_COLUMN}",
                )
                style_valid[:, reversal_index] = reversal_valid

                daily_label_valid = label_valid[sample_indices]
                residual, record = residualize_daily_targets(  # 按日投影 Barra 暴露
                    np.asarray(source_targets[sample_indices], dtype=np.float64),
                    style_matrix,
                    daily_label_valid,
                )
                styles_output[sample_indices] = style_matrix.astype(np.float32)
                valid_output[sample_indices] = style_valid
                targets_output[sample_indices] = residual.astype(np.float32)
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
            print(
                f"{year}：写入 {len(joined):,} 行，"
                f"CNE5S 匹配率 {match_rate:.4%}",
                flush=True,
            )

        styles_output.flush()  # 将 memmap 修改刷入磁盘
        valid_output.flush()  # 将 memmap 修改刷入磁盘
        targets_output.flush()  # 将 memmap 修改刷入磁盘
        del styles_output, valid_output, targets_output
        _validate_arrays(stage, len(sample))
        residual_check = np.load(
            stage / RESIDUAL_TARGETS_FILE, mmap_mode="r"
        )
        if not np.array_equal(np.isfinite(residual_check), label_valid):
            raise AssertionError("残差标签有限性与 label_valid 不一致")

        daily_diagnostics = pd.DataFrame(diagnostics)
        daily_diagnostics.to_parquet(
            stage / DAILY_DIAGNOSTICS_FILE, index=False, compression="zstd"
        )
        annual_summary = daily_diagnostics.groupby(  # 按既定口径聚合
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
        annual_summary.to_parquet(
            stage / ANNUAL_SUMMARY_FILE, index=False, compression="zstd"
        )
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
        (stage / MANIFEST_FILE).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _atomic_replace_directory(stage, target)  # 原子落盘，避免半成品
        return {**manifest, "reused": False}
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise


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
def _validate_arrays(store_dir: Path, sample_count: int) -> None:
    """校验输入形状、业务键、有限性和签名约束；异常时明确失败。"""
    styles = np.load(store_dir / STYLE_EXPOSURES_FILE, mmap_mode="r")
    valid = np.load(store_dir / STYLE_VALID_FILE, mmap_mode="r")
    residual = np.load(store_dir / RESIDUAL_TARGETS_FILE, mmap_mode="r")
    if styles.shape != (sample_count, len(STYLE_COLUMNS)) or styles.dtype != np.float32:
        raise ValueError("style_exposures.npy 形状或 dtype 异常")
    if valid.shape != styles.shape or valid.dtype != np.bool_:
        raise ValueError("style_valid.npy 形状或 dtype 异常")
    if residual.shape != (sample_count,) or residual.dtype != np.float32:
        raise ValueError("targets_style_residual.npy 形状或 dtype 异常")
    if not np.isfinite(styles).all():
        raise ValueError("style_exposures.npy 含 NaN/Inf")

def require_style_neutral_store(
    store_dir: str | Path = DEFAULT_STORE_DIR,
) -> dict[str, Any]:
    """验证风格中性训练数据与当前源数据签名一致。"""

    root = Path(store_dir).resolve()
    manifest_path = root / MANIFEST_FILE
    if not manifest_path.exists():
        raise FileNotFoundError("缺少风格中性训练数据，请先运行 build_style_neutral_store")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = _store_payload(
        base.SAMPLE_INDEX_PATH,
        base.TARGETS_PATH,
        barra.DEFAULT_CNE5S_PATH,
        barra.DEFAULT_EOD_PATH,
        barra.DEFAULT_CALENDAR_PATH,
    )
    if manifest.get("signature") != _stable_hash(expected):
        raise ValueError("风格中性训练数据与当前源数据签名不一致")
    _validate_arrays(root, int(manifest["sample_count"]))
    return manifest

def _validate_industry_sidecar(
    store_dir: Path,
    manifest: dict[str, Any],
) -> None:
    """校验输入形状、业务键、有限性和签名约束；异常时明确失败。"""
    codes = np.load(store_dir / INDUSTRY_CODES_FILE, mmap_mode="r")
    sample_count = int(manifest["sample_count"])
    mapping = manifest.get("industry_mapping")
    if codes.shape != (sample_count,) or codes.dtype != np.int16:
        raise ValueError("industry_codes.npy 形状或 dtype 异常")
    if not isinstance(mapping, dict) or mapping.get(barra.UNKNOWN_INDUSTRY) != 0:
        raise ValueError("行业 sidecar 缺少稳定行业映射")
    if (codes < 0).any() or int(codes.max()) >= len(mapping):
        raise ValueError("industry_codes.npy 含非法行业代码")
    known_count = len(mapping) - 1
    if known_count != EXPECTED_CNE5S_INDUSTRIES:
        raise ValueError(
            f"CNE5S 行业数为 {known_count}，预期 {EXPECTED_CNE5S_INDUSTRIES}"
        )

def require_industry_sidecar(
    store_dir: str | Path = DEFAULT_INDUSTRY_STORE_DIR,
    *,
    minimum_match_rate: float = MINIMUM_MATCH_RATE,
) -> dict[str, Any]:
    """验证联合 R² 所需的逐样本行业代码 sidecar。"""

    root = Path(store_dir).resolve()
    manifest_path = root / INDUSTRY_MANIFEST_FILE
    if not manifest_path.exists():
        raise FileNotFoundError(
            "缺少行业 sidecar，请先运行 build_industry_sidecar"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    style_manifest = require_style_neutral_store()
    expected = _industry_sidecar_payload(
        base.SAMPLE_INDEX_PATH,
        barra.DEFAULT_CNE5S_PATH,
        style_manifest["signature"],
        minimum_match_rate,
    )
    if manifest.get("signature") != _stable_hash(expected):
        raise ValueError("行业 sidecar 与当前样本、CNE5S 或风格 store 签名不一致")
    _validate_industry_sidecar(root, manifest)
    return manifest

def _validate_full_barra_target_sidecar(
    store_dir: Path,
    manifest: dict[str, Any],
) -> None:
    """校验输入形状、业务键、有限性和签名约束；异常时明确失败。"""
    required_files = (
        FULL_BARRA_TARGETS_FILE,
        FULL_BARRA_DAILY_DIAGNOSTICS_FILE,
        FULL_BARRA_ANNUAL_SUMMARY_FILE,
        FULL_BARRA_TARGET_MANIFEST_FILE,
    )
    missing = [name for name in required_files if not (store_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"完整 Barra 残差标签文件缺失：{missing}")
    targets = np.load(store_dir / FULL_BARRA_TARGETS_FILE, mmap_mode="r")
    if (
        targets.shape != (int(manifest["sample_count"]),)
        or targets.dtype != np.float32
    ):
        raise ValueError(
            "targets_full_barra_residual.npy 形状或 dtype 异常"
        )
    if int(np.isfinite(targets).sum()) != int(manifest["label_valid_count"]):
        raise ValueError("完整 Barra 残差标签有限值数量与 manifest 不一致")
    daily = pd.read_parquet(
        store_dir / FULL_BARRA_DAILY_DIAGNOSTICS_FILE,
        columns=[
            "rows",
            "max_abs_style_residual_pearson",
            "max_abs_industry_residual_mean",
        ],
    )
    if int(daily["rows"].sum()) != int(manifest["sample_count"]):
        raise ValueError("完整 Barra 逐日诊断行数汇总与 manifest 不一致")
    if daily[
        [
            "max_abs_style_residual_pearson",
            "max_abs_industry_residual_mean",
        ]
    ].to_numpy().max() > ORTHOGONALITY_TOLERANCE:
        raise ValueError("完整 Barra 残差标签正交诊断未通过")

def require_full_barra_target_sidecar(
    store_dir: str | Path = DEFAULT_FULL_BARRA_TARGET_STORE_DIR,
) -> dict[str, Any]:
    """验证十风格加行业固定效应残差标签 sidecar。"""

    root = Path(store_dir).resolve()
    manifest_path = root / FULL_BARRA_TARGET_MANIFEST_FILE
    if not manifest_path.exists():
        raise FileNotFoundError(
            "缺少完整 Barra 残差标签，请先运行 "
            "build_full_barra_target_sidecar"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    style_manifest = require_style_neutral_store()
    industry_manifest = require_industry_sidecar()
    expected = _full_barra_target_payload(
        base.TARGETS_PATH,
        style_manifest["signature"],
        industry_manifest["signature"],
    )
    if manifest.get("signature") != _stable_hash(expected):
        raise ValueError(
            "完整 Barra 残差标签与当前标签、风格或行业 sidecar 签名不一致"
        )
    _validate_full_barra_target_sidecar(root, manifest)
    return manifest

def _validate_dual_tower_target_sidecar(
    store_dir: Path, manifest: dict[str, Any]
) -> None:
    """校验输入形状、业务键、有限性和签名约束；异常时明确失败。"""
    required = (
        DUAL_TOWER_FITTED_TARGETS_FILE,
        DUAL_TOWER_DAILY_DIAGNOSTICS_FILE,
        DUAL_TOWER_ANNUAL_SUMMARY_FILE,
        DUAL_TOWER_TARGET_MANIFEST_FILE,
    )
    missing = [name for name in required if not (store_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"No.6 fitted-target sidecar 文件缺失：{missing}")
    targets = np.load(
        store_dir / DUAL_TOWER_FITTED_TARGETS_FILE, mmap_mode="r"
    )
    if (
        targets.shape != (int(manifest["sample_count"]),)
        or targets.dtype != np.float32
    ):
        raise ValueError("No.6 fitted target 形状或 dtype 异常")
    if int(np.isfinite(targets).sum()) != int(manifest["label_valid_count"]):
        raise ValueError("No.6 fitted target 有效值数量与 manifest 不一致")
    daily = pd.read_parquet(
        store_dir / DUAL_TOWER_DAILY_DIAGNOSTICS_FILE,
        columns=[
            "rows",
            "reconstruction_max_abs_error",
            "residual_sidecar_max_abs_error",
            "fitted_joint_barra_r_squared",
        ],
    )
    if int(daily["rows"].sum()) != int(manifest["sample_count"]):
        raise ValueError("No.6 fitted target 逐日行数与 manifest 不一致")
    if float(daily["reconstruction_max_abs_error"].max()) > 1e-12:
        raise ValueError("No.6 标签分解无法精确重构源标签")
    if float(daily["residual_sidecar_max_abs_error"].max()) > 1e-6:
        raise ValueError("No.6 重算残差与现有完整残差 sidecar 不一致")
    if float((1.0 - daily["fitted_joint_barra_r_squared"]).abs().max()) > 1e-8:
        raise ValueError("No.6 fitted target 不在完整 Barra 风险列空间内")

def require_dual_tower_target_sidecar(
    store_dir: str | Path = DEFAULT_DUAL_TOWER_TARGET_STORE_DIR,
) -> dict[str, Any]:
    """验证 No.6 风格塔拟合收益标签 sidecar。"""

    root = Path(store_dir).resolve()
    manifest_path = root / DUAL_TOWER_TARGET_MANIFEST_FILE
    if not manifest_path.exists():
        raise FileNotFoundError(
            "缺少 No.6 fitted-target sidecar，请先运行 "
            "build_dual_tower_target_sidecar"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    style_manifest = require_style_neutral_store()
    industry_manifest = require_industry_sidecar()
    full_manifest = require_full_barra_target_sidecar()
    expected = _dual_tower_target_payload(
        base.TARGETS_PATH,
        full_manifest["signature"],
        style_manifest["signature"],
        industry_manifest["signature"],
    )
    if manifest.get("signature") != _stable_hash(expected):
        raise ValueError("No.6 fitted target 与当前标签或 Barra sidecar 签名不一致")
    _validate_dual_tower_target_sidecar(root, manifest)
    return manifest
