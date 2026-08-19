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


VIVIT_DIR = Path(__file__).resolve().parent  # 定位vivit目录。
DATA_DIR = VIVIT_DIR / "data"  # 定位数据目录。
DEFAULT_STORE_DIR = DATA_DIR / "style_neutral_store"  # 定位默认存储目录。
STYLE_EXPOSURES_FILE = "style_exposures.npy"  # 定位风格暴露文件。
STYLE_VALID_FILE = "style_valid.npy"  # 定位风格有效文件。
RESIDUAL_TARGETS_FILE = "targets_style_residual.npy"  # 定位残差标签文件。
DAILY_DIAGNOSTICS_FILE = "daily_diagnostics.parquet"  # 定位daily诊断文件。
ANNUAL_SUMMARY_FILE = "annual_summary.parquet"  # 定位annual汇总文件。
MANIFEST_FILE = "style_neutral_manifest.json"  # 定位清单文件。
DEFAULT_INDUSTRY_STORE_DIR = DEFAULT_STORE_DIR / "industry_sidecar"  # 定位默认行业存储目录。
INDUSTRY_CODES_FILE = "industry_codes.npy"  # 定位行业代码文件。
INDUSTRY_AUDIT_FILE = "industry_match_audit.parquet"  # 定位行业audit文件。
INDUSTRY_MANIFEST_FILE = "industry_sidecar_manifest.json"  # 定位行业清单文件。
DEFAULT_FULL_BARRA_TARGET_STORE_DIR = (  # 定位默认fullBarra标签存储目录。
    DEFAULT_STORE_DIR / "full_barra_target_sidecar"
)
FULL_BARRA_TARGETS_FILE = "targets_full_barra_residual.npy"  # 定位fullBarra标签文件。
FULL_BARRA_DAILY_DIAGNOSTICS_FILE = "daily_diagnostics.parquet"  # 定位fullBarradaily诊断文件。
FULL_BARRA_ANNUAL_SUMMARY_FILE = "annual_summary.parquet"  # 定位fullBarraannual汇总文件。
FULL_BARRA_TARGET_MANIFEST_FILE = "full_barra_target_manifest.json"  # 定位fullBarra标签清单文件。
DEFAULT_DUAL_TOWER_TARGET_STORE_DIR = (  # 定位默认dualtower标签存储目录。
    DEFAULT_STORE_DIR / "dual_tower_target_sidecar"
)
DUAL_TOWER_FITTED_TARGETS_FILE = "targets_full_barra_fitted.npy"  # 定位dualtower拟合标签文件。
DUAL_TOWER_DAILY_DIAGNOSTICS_FILE = "daily_diagnostics.parquet"  # 定位dualtowerdaily诊断文件。
DUAL_TOWER_ANNUAL_SUMMARY_FILE = "annual_summary.parquet"  # 定位dualtowerannual汇总文件。
DUAL_TOWER_TARGET_MANIFEST_FILE = "dual_tower_target_manifest.json"  # 定位dualtower标签清单文件。
STYLE_COLUMNS = (*barra.STYLE_COLUMNS, barra.REVERSAL_COLUMN)  # 集中声明风格列。
SCHEMA_VERSION = 1  # 固定产物 schema 版本，用于判断缓存与复用兼容性。
INDUSTRY_SIDECAR_SCHEMA_VERSION = 1  # 固定产物 schema 版本，用于判断缓存与复用兼容性。
FULL_BARRA_TARGET_SCHEMA_VERSION = 1  # 固定产物 schema 版本，用于判断缓存与复用兼容性。
DUAL_TOWER_TARGET_SCHEMA_VERSION = 1  # 固定产物 schema 版本，用于判断缓存与复用兼容性。
EXPECTED_CNE5S_INDUSTRIES = 32  # 设置预期cne5s行业。
MINIMUM_MATCH_RATE = 0.999  # 设置业务键连接允许的最低覆盖率。
ORTHOGONALITY_TOLERANCE = 1e-8  # 设置 Barra 投影正交性的数值验收阈值。


def _file_fingerprint(path: Path) -> dict[str, Any]:  # 提取文件身份信息，用于来源审计和复用判断。
    """提取文件身份信息，用于来源审计和复用判断。"""
    if not path.exists():  # 检查目标文件或目录是否满足存在性要求。
        raise FileNotFoundError(path)  # 目标文件不存在时立即终止。
    return {  # 返回提取文件身份信息，用于来源审计和复用判断。
        "path": str(path.resolve()),  # 记录路径。
        "size": int(path.stat().st_size),  # 记录大小。
        "mtime_ns": int(path.stat().st_mtime_ns),  # 记录修改时间纳秒。
    }


def _stable_hash(payload: object) -> str:  # 对可 JSON 序列化载荷计算稳定的 SHA-256 签名。
    """对可 JSON 序列化载荷计算稳定的 SHA-256 签名。"""
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()  # 返回稳定的 SHA-256 十六进制摘要。


def _store_payload(  # 构造参与稳定签名和复用判断的配置载荷。
    sample_index_path: Path,  # 接收样本索引路径。
    targets_path: Path,  # 接收标签路径。
    cne5s_path: Path,  # 接收cne5s路径。
    eod_path: Path,  # 接收日行情路径。
    calendar_path: Path,  # 接收交易日历路径。
) -> dict[str, Any]:
    """构造参与稳定签名和复用判断的配置载荷。"""
    dataset = json.loads(  # 读取上游数据清单。
        base.SOURCE_DATASET_MANIFEST_PATH.read_text(encoding="utf-8")
    )
    return {  # 返回构造参与稳定签名和复用判断的配置载荷。
        "schema_version": SCHEMA_VERSION,  # 记录schema版本。
        "source_dataset_signature": dataset["signature"],  # 记录来源dataset签名，用于来源一致性校验。
        "sample_index": _file_fingerprint(sample_index_path),  # 记录样本索引。
        "source_targets": _file_fingerprint(targets_path),  # 记录来源标签。
        "cne5s": barra._fingerprint_parquet(cne5s_path),  # 记录cne5s。
        "wind_eod_prices": barra._fingerprint_parquet(eod_path),  # 记录wind日行情prices。
        "sse_calendar": barra._fingerprint_parquet(calendar_path),  # 记录SSE交易日历。
        "style_columns": list(STYLE_COLUMNS),  # 记录风格字段列表。
        "style_count": len(STYLE_COLUMNS),  # 记录风格数量。
        "industry_used": False,  # 记录行业使用状态。
        "residualization": (  # 记录residualization。
            "daily_equal_weight_ols_intercept_plus_9_cne5s_styles_plus_"
            "reversal20_then_residual_zscore_ddof0"
        ),
        "missing_policy": (  # 声明缺失值处理口径。
            "daily_standardize_valid_exposures_then_fill_zero_no_forward_fill"
        ),
        "reversal_formula": (  # 记录反转formula。
            "-(s_dq_adjclose(T)/s_dq_adjclose(T-20_SSE_days)-1)"
        ),
        "match_policy": "exact_trade_date_and_assetid_no_forward_fill",  # 把 match_policy 映射为"exact_trade_date_and_assetid_no_forward_fill"。
    }


def _standardize_with_missing(  # 有效值按日标准化，缺失值填 0。
    values: np.ndarray,  # 输入待处理的截面数值。
    valid: np.ndarray,  # 输入与数值等长的有效性掩码。
    *,
    label: str,  # 接收标签。
) -> np.ndarray:
    """有效值按日标准化，缺失值填 0；不做任何跨日填充。"""

    raw = np.asarray(values, dtype=np.float64)  # 转换并统一原始的数据表示。
    mask = np.asarray(valid, dtype=bool)  # 转换并统一掩码的数据表示。
    if raw.ndim != 1 or mask.shape != raw.shape:  # 检查张量或数组形状是否符合接口约定。
        raise ValueError(f"{label} 数值和有效标记形状不一致")  # 输入数据不满足业务或数值约束时立即终止。
    if int(mask.sum()) < 3:  # 检查当前数值是否越过业务或数值阈值。
        raise ValueError(f"{label} 单日有效样本不足 3 个")  # 输入数据不满足业务或数值约束时立即终止。
    selected = raw[mask]  # 均值和标准差只由当日有效暴露估计。
    if not np.isfinite(selected).all():  # 检查参与计算的数值是否全部有限。
        raise ValueError(f"{label} 有效暴露含 NaN 或 Inf")  # 输入数据不满足业务或数值约束时立即终止。
    mean = float(selected.mean())  # 计算均值。
    std = float(selected.std(ddof=0))  # 计算标准差。
    if not np.isfinite(std) or std <= np.finfo(np.float64).eps:  # 检查参与计算的数值是否全部有限。
        raise ValueError(f"{label} 单日暴露为常数")  # 输入数据不满足业务或数值约束时立即终止。
    result = np.zeros(len(raw), dtype=np.float64)  # 缺失暴露填零，但有效性仍由独立 mask 保存。
    result[mask] = (selected - mean) / std  # 构造结果，限定后续参与计算的样本。
    return result  # 返回结果。


def residualize_daily_targets(  # 对单日标签联合回归十风格，返回保持单位尺度的 OLS 残差。
    targets: np.ndarray,  # 接收标签。
    style_exposures: np.ndarray,  # 接收风格暴露。
    label_valid: np.ndarray,  # 输入标签有效性掩码。
) -> tuple[np.ndarray, dict[str, float | int]]:
    """对单日标签联合回归十风格，返回保持单位尺度的 OLS 残差。"""

    y_all = np.asarray(targets, dtype=np.float64)  # 转换并统一yall的数据表示。
    styles = np.asarray(style_exposures, dtype=np.float64)  # 转换并统一风格的数据表示。
    valid = np.asarray(label_valid, dtype=bool)  # 转换并统一有效的数据表示。
    if styles.shape != (len(y_all), len(STYLE_COLUMNS)):  # 检查张量或数组形状是否符合接口约定。
        raise ValueError("单日风格矩阵形状异常")  # 输入数据不满足业务或数值约束时立即终止。
    if valid.shape != y_all.shape or int(valid.sum()) < len(STYLE_COLUMNS) + 2:  # 检查张量或数组形状是否符合接口约定。
        raise ValueError("单日有效标签不足以进行十风格回归")  # 输入数据不满足业务或数值约束时立即终止。
    if not np.isfinite(y_all[valid]).all() or not np.isfinite(styles).all():  # 检查参与计算的数值是否全部有限。
        raise ValueError("单日标签或风格矩阵含 NaN/Inf")  # 输入数据不满足业务或数值约束时立即终止。

    y = y_all[valid]  # 回归样本仅包含标签有效的股票。
    x = styles[valid]  # 风格暴露已按日标准化并完成缺失处理。
    design = np.column_stack([np.ones(len(y), dtype=np.float64), x])  # 组成 Barra 设计矩阵。
    coefficients, _, rank, singular_values = np.linalg.lstsq(  # 求解当日截面 OLS。
        design, y, rcond=None  # 向最小二乘求解器传入矩阵与目标，并使用默认奇异值截断阈值。
    )
    expected_rank = design.shape[1]  # 记录设计矩阵应达到的满列秩。
    if int(rank) != expected_rank:  # 确认回归设计矩阵达到满列秩。
        raise np.linalg.LinAlgError(  # 回归设计矩阵无法稳定求解时立即终止。
            f"单日十风格设计矩阵秩为 {rank}，预期 {expected_rank}"
        )
    if singular_values[-1] <= np.finfo(np.float64).eps:  # 检查当前数值是否越过业务或数值阈值。
        raise np.linalg.LinAlgError("单日十风格设计矩阵条件数不可计算")  # 回归设计矩阵无法稳定求解时立即终止。

    fitted = design @ coefficients  # 标签中可被截距与十风格解释的部分。
    residual = y - fitted  # Alpha 标签取 OLS 正交补。
    residual_mean = float(residual.mean())  # 计算残差均值。
    residual_std = float(residual.std(ddof=0))  # 计算残差标准差。
    if not np.isfinite(residual_std) or residual_std <= np.finfo(np.float64).eps:  # 检查参与计算的数值是否全部有限。
        raise ValueError("单日风格残差标签为常数")  # 输入数据不满足业务或数值约束时立即终止。
    residual = (residual - residual_mean) / residual_std  # 每日恢复为均值 0、标准差 1 的训练尺度。
    correlations = np.corrcoef(  # 数值验收：残差应与全部风格近似正交。
        np.column_stack([residual, x]), rowvar=False
    )[0, 1:]
    if not np.isfinite(correlations).all():  # 检查参与计算的数值是否全部有限。
        raise ValueError("单日残差标签风格相关性不可计算")  # 输入数据不满足业务或数值约束时立即终止。
    max_abs_correlation = float(np.max(np.abs(correlations)))  # 计算maxabs相关系数。
    if max_abs_correlation > ORTHOGONALITY_TOLERANCE:  # 检查当前数值是否越过业务或数值阈值。
        raise AssertionError(  # 关键数据契约被破坏时立即终止。
            f"单日残差标签最大风格相关性 {max_abs_correlation:.3e} 超过容差"
        )

    output = np.full(len(y_all), np.nan, dtype=np.float64)  # 无效标签保持 NaN，不伪造监督信号。
    output[valid] = residual  # 构造输出，限定后续参与计算的样本。
    source_centered = y - y.mean()  # 计算来源中心化。
    residual_variance = float(np.dot(residual, residual))  # 计算残差方差。
    source_variance = float(np.dot(source_centered, source_centered))  # 计算来源方差。
    raw_residual = y - fitted
    raw_residual_variance = float(np.dot(raw_residual, raw_residual))  # 计算原始残差方差。
    diagnostics: dict[str, float | int] = {  # 初始化诊断容器。
        "rows": len(y_all),  # 记录行数。
        "label_valid_rows": int(valid.sum()),  # 记录标签有效行数。
        "matrix_rank": int(rank),  # 记录回归设计矩阵的实际秩。
        "condition_number": float(singular_values[0] / singular_values[-1]),  # 记录conditionnumber。
        "source_target_std": float(y.std(ddof=0)),  # 记录来源标签标准差。
        "raw_residual_std": residual_std,  # 记录原始残差标准差。
        "residual_target_mean": float(residual.mean()),  # 记录残差标签均值。
        "residual_target_std": float(residual.std(ddof=0)),  # 记录残差标签标准差。
        "style_r_squared": float(1.0 - raw_residual_variance / source_variance),  # 记录风格rsquared。
        "max_abs_style_residual_pearson": max_abs_correlation,  # 记录maxabs风格残差Pearson。
        "standardized_residual_sum_squares": residual_variance,  # 记录standardized残差sumsquares。
    }
    return output, diagnostics  # 返回完成上述处理后的结果。


def decompose_daily_targets_with_industry(  # 返回完整 Barra 回归的标准化残差和标准化拟合项。
    targets: np.ndarray,  # 接收标签。
    style_exposures: np.ndarray,  # 接收风格暴露。
    industry_codes: np.ndarray,  # 接收行业代码。
    label_valid: np.ndarray,  # 输入标签有效性掩码。
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    """返回完整 Barra 回归的标准化残差和标准化拟合项。"""

    y_all = np.asarray(targets, dtype=np.float64)  # 转换并统一yall的数据表示。
    styles_all = np.asarray(style_exposures, dtype=np.float64)  # 转换并统一风格all的数据表示。
    codes_all = np.asarray(industry_codes)  # 转换并统一代码all的数据表示。
    valid = np.asarray(label_valid, dtype=bool)  # 转换并统一有效的数据表示。
    if styles_all.shape != (len(y_all), len(STYLE_COLUMNS)):  # 检查张量或数组形状是否符合接口约定。
        raise ValueError("单日风格矩阵形状异常")  # 输入数据不满足业务或数值约束时立即终止。
    if codes_all.shape != y_all.shape or not np.issubdtype(  # 检查张量或数组形状是否符合接口约定。
        codes_all.dtype, np.integer
    ):
        raise ValueError("行业代码必须是一维整数数组且与标签等长")  # 输入数据不满足业务或数值约束时立即终止。
    if valid.shape != y_all.shape:  # 检查张量或数组形状是否符合接口约定。
        raise ValueError("单日标签有效标记形状异常")  # 输入数据不满足业务或数值约束时立即终止。
    if int(valid.sum()) < len(STYLE_COLUMNS) + 2:  # 检查当前样本数量是否满足计算要求。
        raise ValueError("单日有效标签不足以进行完整 Barra 回归")  # 输入数据不满足业务或数值约束时立即终止。
    if (codes_all < 0).any() or (  # 检查当前掩码或数据是否满足有效性要求。
        codes_all > EXPECTED_CNE5S_INDUSTRIES
    ).any():
        raise ValueError("行业代码超出 0 至 32 的有效范围")  # 输入数据不满足业务或数值约束时立即终止。
    if not np.isfinite(y_all[valid]).all() or not np.isfinite(  # 检查参与计算的数值是否全部有限。
        styles_all
    ).all():
        raise ValueError("单日标签或风格矩阵含 NaN/Inf")  # 输入数据不满足业务或数值约束时立即终止。

    y = y_all[valid]
    styles = styles_all[valid]
    codes = codes_all[valid]
    _, compact = np.unique(codes, return_inverse=True)  # 压缩当日行业编码。
    counts = np.bincount(compact).astype(np.float64)  # 计算数量。
    y_means = np.bincount(compact, weights=y) / counts  # 计算ymeans。
    style_sums = np.zeros(  # 初始化风格sums。
        (len(counts), len(STYLE_COLUMNS)), dtype=np.float64
    )
    np.add.at(style_sums, compact, styles)  # 按行业汇总截面统计量。
    style_means = style_sums / counts[:, None]  # 计算风格means。
    y_within = y - y_means[compact]  # 行业内去均值等价于吸收行业固定效应。
    styles_within = styles - style_means[compact]  # 风格也在行业内中心化后再回归。

    coefficients, _, rank, singular_values = np.linalg.lstsq(  # 求解当日截面 OLS。
        styles_within, y_within, rcond=None  # 向最小二乘求解器传入矩阵与目标，并使用默认奇异值截断阈值。
    )
    if int(rank) != len(STYLE_COLUMNS):  # 确认回归设计矩阵达到满列秩。
        raise np.linalg.LinAlgError(  # 回归设计矩阵无法稳定求解时立即终止。
            f"行业内十风格矩阵秩为 {rank}，预期 {len(STYLE_COLUMNS)}"
        )
    if singular_values[-1] <= np.finfo(np.float64).eps:  # 检查当前数值是否越过业务或数值阈值。
        raise np.linalg.LinAlgError("行业内十风格矩阵条件数不可计算")  # 回归设计矩阵无法稳定求解时立即终止。

    raw_residual = y_within - styles_within @ coefficients  # 风格与行业均无法解释的 Alpha 部分。
    raw_fitted = y - raw_residual  # Style 塔监督目标为完整 Barra 拟合项。
    reconstruction_error = float(
        np.max(np.abs(y - raw_fitted - raw_residual))
    )
    if reconstruction_error > 1e-12:  # 检查当前数值是否越过业务或数值阈值。
        raise AssertionError("完整 Barra 拟合项与残差无法重构源标签")  # 关键数据契约被破坏时立即终止。
    raw_std = float(raw_residual.std(ddof=0))  # 计算原始标准差。
    if not np.isfinite(raw_std) or raw_std <= np.finfo(np.float64).eps:  # 检查参与计算的数值是否全部有限。
        raise ValueError("单日完整 Barra 残差标签为常数")  # 输入数据不满足业务或数值约束时立即终止。
    fitted_std = float(raw_fitted.std(ddof=0))  # 计算拟合标准差。
    if not np.isfinite(fitted_std) or fitted_std <= np.finfo(np.float64).eps:  # 检查参与计算的数值是否全部有限。
        raise ValueError("单日完整 Barra 拟合标签为常数")  # 输入数据不满足业务或数值约束时立即终止。
    residual = (raw_residual - raw_residual.mean()) / raw_std  # Alpha 目标按日标准化。
    fitted = (raw_fitted - raw_fitted.mean()) / fitted_std  # Style 目标独立按日标准化。
    correlations = np.corrcoef(  # 计算correlations。
        np.column_stack([residual, styles]), rowvar=False
    )[0, 1:]
    if not np.isfinite(correlations).all():  # 检查参与计算的数值是否全部有限。
        raise ValueError("单日完整 Barra 残差风格相关性不可计算")  # 输入数据不满足业务或数值约束时立即终止。
    max_abs_correlation = float(np.max(np.abs(correlations)))  # 计算maxabs相关系数。
    group_residual_means = (  # 行业固定效应已吸收时，各行业残差均值应接近零。
        np.bincount(compact, weights=residual) / counts
    )
    max_abs_industry_mean = float(np.max(np.abs(group_residual_means)))  # 计算maxabs行业均值。
    if max(max_abs_correlation, max_abs_industry_mean) > ORTHOGONALITY_TOLERANCE:  # 检查当前数值是否越过业务或数值阈值。
        raise AssertionError(  # 关键数据契约被破坏时立即终止。
            "单日完整 Barra 残差未达到数值正交："
            f"style={max_abs_correlation:.3e}, "
            f"industry={max_abs_industry_mean:.3e}"
        )

    centered = y - y.mean()  # 计算中心化。
    total_sum_squares = float(np.dot(centered, centered))
    residual_sum_squares = float(np.dot(raw_residual, raw_residual))
    if total_sum_squares <= np.finfo(np.float64).eps:  # 检查当前数值是否越过业务或数值阈值。
        raise ValueError("单日源标签为常数")  # 输入数据不满足业务或数值约束时立即终止。
    output = np.full(len(y_all), np.nan, dtype=np.float64)
    fitted_output = np.full(len(y_all), np.nan, dtype=np.float64)
    output[valid] = residual  # 构造输出，限定后续参与计算的样本。
    fitted_output[valid] = fitted  # 构造拟合输出，限定后续参与计算的样本。
    return output, fitted_output, {  # 返回完整 Barra 回归的标准化残差和标准化拟合项。
        "rows": len(y_all),  # 记录行数。
        "label_valid_rows": int(valid.sum()),  # 记录标签有效行数。
        "industry_count": int(len(counts)),  # 记录行业数量。
        "unknown_industry_rows": int((codes == 0).sum()),  # 记录未知行业行业行数。
        "unknown_industry_rate": float((codes == 0).mean()),  # 记录未知行业行业比例。
        "matrix_rank": int(rank),  # 记录回归设计矩阵的实际秩。
        "condition_number": float(singular_values[0] / singular_values[-1]),  # 记录conditionnumber。
        "source_target_std": float(y.std(ddof=0)),  # 记录来源标签标准差。
        "raw_residual_std": raw_std,  # 记录原始残差标准差。
        "raw_fitted_std": fitted_std,  # 记录原始拟合标准差。
        "reconstruction_max_abs_error": reconstruction_error,  # 记录reconstructionmaxabs误差。
        "residual_target_mean": float(residual.mean()),  # 记录残差标签均值。
        "residual_target_std": float(residual.std(ddof=0)),  # 记录残差标签标准差。
        "fitted_target_mean": float(fitted.mean()),  # 记录拟合标签均值。
        "fitted_target_std": float(fitted.std(ddof=0)),  # 记录拟合标签标准差。
        "joint_barra_r_squared": float(  # 记录jointBarrarsquared。
            1.0 - residual_sum_squares / total_sum_squares
        ),
        "max_abs_style_residual_pearson": max_abs_correlation,  # 记录maxabs风格残差Pearson。
        "max_abs_industry_residual_mean": max_abs_industry_mean,  # 记录maxabs行业残差均值。
    }


def residualize_daily_targets_with_industry(  # 对单日标签回归十风格和行业固定效应，再标准化残差。
    targets: np.ndarray,  # 接收标签。
    style_exposures: np.ndarray,  # 接收风格暴露。
    industry_codes: np.ndarray,  # 接收行业代码。
    label_valid: np.ndarray,  # 输入标签有效性掩码。
) -> tuple[np.ndarray, dict[str, float | int]]:
    """对单日标签回归十风格和行业固定效应，再标准化残差。"""

    residual, _fitted, diagnostics = decompose_daily_targets_with_industry(
        targets, style_exposures, industry_codes, label_valid
    )
    return residual, diagnostics  # 返回完成上述处理后的结果。


def project_out_styles_numpy(  # NumPy 参考实现：从单日预测中联合投影掉十个风格。
    predictions: np.ndarray, style_exposures: np.ndarray
) -> tuple[np.ndarray, dict[str, float | int]]:
    """NumPy 参考实现：从单日预测中联合投影掉十个风格。"""

    values = np.asarray(predictions, dtype=np.float64)  # 转换并统一值的数据表示。
    styles = np.asarray(style_exposures, dtype=np.float64)  # 转换并统一风格的数据表示。
    if values.ndim != 1 or styles.shape != (len(values), len(STYLE_COLUMNS)):  # 检查张量或数组形状是否符合接口约定。
        raise ValueError("预测或风格矩阵形状异常")  # 输入数据不满足业务或数值约束时立即终止。
    if len(values) < len(STYLE_COLUMNS) + 2:  # 检查当前样本数量是否满足计算要求。
        raise ValueError("单日股票数不足以进行十风格投影")  # 输入数据不满足业务或数值约束时立即终止。
    if not np.isfinite(values).all() or not np.isfinite(styles).all():  # 检查参与计算的数值是否全部有限。
        raise ValueError("预测或风格矩阵含 NaN/Inf")  # 输入数据不满足业务或数值约束时立即终止。
    design = np.column_stack([np.ones(len(values)), styles])  # 组成 Barra 设计矩阵。
    coefficients, _, rank, singular_values = np.linalg.lstsq(  # 求解当日截面 OLS。
        design, values, rcond=None  # 向最小二乘求解器传入矩阵与目标，并使用默认奇异值截断阈值。
    )
    if int(rank) != design.shape[1]:  # 确认回归设计矩阵达到满列秩。
        raise np.linalg.LinAlgError("十风格投影矩阵秩不足")  # 回归设计矩阵无法稳定求解时立即终止。
    residual = values - design @ coefficients
    correlations = np.corrcoef(  # 计算correlations。
        np.column_stack([residual, styles]), rowvar=False
    )[0, 1:]
    return residual, {  # 返回NumPy 参考实现：从单日预测中联合投影掉十个风格。
        "matrix_rank": int(rank),  # 记录回归设计矩阵的实际秩。
        "condition_number": float(singular_values[0] / singular_values[-1]),  # 记录conditionnumber。
        "max_abs_style_pearson": float(np.max(np.abs(correlations))),  # 记录maxabs风格Pearson。
        "residual_mean": float(residual.mean()),  # 记录残差均值。
    }


def project_out_styles_torch(  # 以 FP32 QR 将单日模型分数投影到十风格正交补，并可恢复单位尺度。
    predictions: torch.Tensor,  # [N]，单日截面分数。
    style_exposures: torch.Tensor,  # [N,10]，单日十风格暴露。
    *,
    standardize: bool = True,  # 控制标准化开关。
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """以 FP32 QR 将单日模型分数投影到十风格正交补，并可恢复单位尺度。"""

    values = predictions.float().reshape(-1)  # [N,...] -> [N]。
    styles = style_exposures.float()  # 转换风格的数据类型或存储位置。
    if styles.shape != (len(values), len(STYLE_COLUMNS)):  # 检查张量或数组形状是否符合接口约定。
        raise ValueError("预测或风格矩阵形状异常")  # 输入数据不满足业务或数值约束时立即终止。
    if len(values) < len(STYLE_COLUMNS) + 2:  # 检查当前样本数量是否满足计算要求。
        raise ValueError("单日股票数不足以进行十风格投影")  # 输入数据不满足业务或数值约束时立即终止。
    if not torch.isfinite(values).all() or not torch.isfinite(styles).all():  # 检查参与计算的数值是否全部有限。
        raise ValueError("预测或风格矩阵含 NaN/Inf")  # 输入数据不满足业务或数值约束时立即终止。

    design = torch.column_stack(  # [N] 截距与 [N,10] 风格拼成 [N,11]。
        [torch.ones_like(values, dtype=torch.float32), styles]
    )
    q, r = torch.linalg.qr(design, mode="reduced")  # q:[N,11]，r:[11,11]。
    diagonal = torch.abs(torch.diagonal(r))
    tolerance = (
        torch.finfo(torch.float32).eps
        * max(design.shape)
        * diagonal.max()
    )
    if bool((diagonal <= tolerance).any()):  # 检查当前掩码或数据是否满足有效性要求。
        raise np.linalg.LinAlgError("十风格投影矩阵秩不足")  # 回归设计矩阵无法稳定求解时立即终止。

    residual = values - q @ (q.transpose(0, 1) @ values)  # [11,N]@[N] -> [11]，再投影回 [N]。
    residual = residual - residual.mean()  # 计算残差。
    residual_std = residual.std(unbiased=False)  # 计算残差标准差。
    if not torch.isfinite(residual_std) or bool(  # 检查参与计算的数值是否全部有限。
        residual_std <= torch.finfo(torch.float32).eps
    ):
        raise ValueError("单日投影残差为常数")  # 输入数据不满足业务或数值约束时立即终止。
    effective = residual / residual_std if standardize else residual  # 训练时通常恢复单位截面波动。
    centered_styles = styles - styles.mean(dim=0, keepdim=True)  # 计算中心化风格。
    correlations = (  # 计算correlations。
        centered_styles.transpose(0, 1) @ effective  # [10,N]@[N] -> [10]。
    ) / torch.sqrt(
        centered_styles.square().sum(dim=0)
        * effective.square().sum()
    )
    if not torch.isfinite(correlations).all():  # 检查参与计算的数值是否全部有限。
        raise ValueError("投影后风格相关性不可计算")  # 输入数据不满足业务或数值约束时立即终止。
    return effective, {  # 返回以 FP32 QR 将单日模型分数投影到十风格正交补，并可恢复单位尺度。
        "residual_mean": effective.mean().detach(),  # 记录残差均值。
        "residual_std": effective.std(unbiased=False).detach(),  # 记录残差标准差。
        "max_abs_style_pearson": correlations.abs().max().detach(),  # 记录maxabs风格Pearson。
    }


def joint_barra_r_squared_numpy(  # 计算 raw score 对十风格和行业固定效应的单日联合 R²。
    predictions: np.ndarray,  # 接收预测。
    style_exposures: np.ndarray,  # 接收风格暴露。
    industry_codes: np.ndarray,  # 接收行业代码。
) -> tuple[float, dict[str, float | int]]:
    """计算 raw score 对十风格和行业固定效应的单日联合 R²。"""

    values = np.asarray(predictions, dtype=np.float64)  # 转换并统一值的数据表示。
    styles = np.asarray(style_exposures, dtype=np.float64)  # 转换并统一风格的数据表示。
    codes = np.asarray(industry_codes)  # 转换并统一代码的数据表示。
    if values.ndim != 1 or styles.shape != (len(values), len(STYLE_COLUMNS)):  # 检查张量或数组形状是否符合接口约定。
        raise ValueError("预测或十风格矩阵形状异常")  # 输入数据不满足业务或数值约束时立即终止。
    if codes.shape != values.shape or not np.issubdtype(codes.dtype, np.integer):  # 检查张量或数组形状是否符合接口约定。
        raise ValueError("行业代码必须是一维整数数组且与预测等长")  # 输入数据不满足业务或数值约束时立即终止。
    if len(values) < len(STYLE_COLUMNS) + 2:  # 检查当前样本数量是否满足计算要求。
        raise ValueError("单日股票数不足以计算联合 Barra R²")  # 输入数据不满足业务或数值约束时立即终止。
    if (codes < 0).any():  # 检查当前掩码或数据是否满足有效性要求。
        raise ValueError("行业代码不能为负数")  # 输入数据不满足业务或数值约束时立即终止。
    if not np.isfinite(values).all() or not np.isfinite(styles).all():  # 检查参与计算的数值是否全部有限。
        raise ValueError("预测或十风格矩阵含 NaN/Inf")  # 输入数据不满足业务或数值约束时立即终止。

    _, compact = np.unique(codes, return_inverse=True)  # 压缩当日行业编码。
    counts = np.bincount(compact).astype(np.float64)  # 计算数量。
    value_means = np.bincount(compact, weights=values) / counts  # 计算值means。
    style_sums = np.zeros((len(counts), len(STYLE_COLUMNS)), dtype=np.float64)  # 初始化风格sums。
    np.add.at(style_sums, compact, styles)  # 按行业汇总截面统计量。
    style_means = style_sums / counts[:, None]  # 计算风格means。
    values_within = values - value_means[compact]  # 先吸收行业固定效应。
    styles_within = styles - style_means[compact]

    coefficients, _, rank, singular_values = np.linalg.lstsq(  # 求解当日截面 OLS。
        styles_within, values_within, rcond=None  # 向最小二乘求解器传入矩阵与目标，并使用默认奇异值截断阈值。
    )
    if int(rank) != len(STYLE_COLUMNS):  # 确认回归设计矩阵达到满列秩。
        raise np.linalg.LinAlgError(  # 回归设计矩阵无法稳定求解时立即终止。
            f"行业内十风格矩阵秩为 {rank}，预期 {len(STYLE_COLUMNS)}"
        )
    residual = values_within - styles_within @ coefficients
    centered = values - values.mean()  # 计算中心化。
    total_sum_squares = float(np.dot(centered, centered))
    if total_sum_squares <= np.finfo(np.float64).eps:  # 检查当前数值是否越过业务或数值阈值。
        raise ValueError("单日 raw score 为常数")  # 输入数据不满足业务或数值约束时立即终止。
    residual_sum_squares = float(np.dot(residual, residual))
    r_squared = 1.0 - residual_sum_squares / total_sum_squares  # 联合 Barra 暴露解释的分数方差比例。
    tolerance = 1e-10
    if not np.isfinite(r_squared) or not -tolerance <= r_squared <= 1.0 + tolerance:  # 检查参与计算的数值是否全部有限。
        raise FloatingPointError(f"联合 Barra R² 超出数值范围：{r_squared}")  # 检测到 NaN、Inf 或退化数值时立即终止。
    r_squared = float(np.clip(r_squared, 0.0, 1.0))  # 计算rsquared。
    group_residual_means = np.bincount(compact, weights=residual) / counts  # 计算group残差means。
    return r_squared, {  # 返回上述计算结果。
        "r_squared": r_squared,  # 记录rsquared。
        "industry_count": int(len(counts)),  # 记录行业数量。
        "matrix_rank": int(rank),  # 记录回归设计矩阵的实际秩。
        "condition_number": float(singular_values[0] / singular_values[-1]),  # 记录conditionnumber。
        "residual_sum_squares": residual_sum_squares,  # 记录残差sumsquares。
        "total_sum_squares": total_sum_squares,  # 记录totalsumsquares。
        "max_abs_industry_residual_mean": float(  # 记录maxabs行业残差均值。
            np.max(np.abs(group_residual_means))
        ),
    }


def joint_barra_r_squared_torch(  # 可微联合 R²。
    predictions: torch.Tensor,  # [N]，单日截面分数。
    style_exposures: torch.Tensor,  # [N,10]，单日十风格暴露。
    industry_codes: torch.Tensor,  # [N]，单日行业代码。
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """可微联合 R²；Barra/行业固定，梯度只回传到 raw score。"""

    device_type = predictions.device.type
    with torch.autocast(device_type=device_type, enabled=False):  # QR 与 R² 固定用 FP32，避免混合精度放大秩误差。
        values = predictions.float().reshape(-1)  # [N,...] -> [N]。
        styles = style_exposures.float()  # 转换风格的数据类型或存储位置。
        codes = industry_codes.reshape(-1)  # [N,...] -> [N]。
        integer_dtypes = (
            torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8
        )
        if styles.shape != (len(values), len(STYLE_COLUMNS)):  # 检查张量或数组形状是否符合接口约定。
            raise ValueError("预测或十风格矩阵形状异常")  # 输入数据不满足业务或数值约束时立即终止。
        if codes.shape != values.shape or codes.dtype not in integer_dtypes:  # 检查张量或数组形状是否符合接口约定。
            raise ValueError("行业代码必须是一维整数张量且与预测等长")  # 输入数据不满足业务或数值约束时立即终止。
        if len(values) < len(STYLE_COLUMNS) + 2:  # 检查当前样本数量是否满足计算要求。
            raise ValueError("单日股票数不足以计算联合 Barra R²")  # 输入数据不满足业务或数值约束时立即终止。
        if bool((codes < 0).any()):  # 检查当前掩码或数据是否满足有效性要求。
            raise ValueError("行业代码不能为负数")  # 输入数据不满足业务或数值约束时立即终止。
        if not bool(torch.isfinite(values).all()) or not bool(  # 检查参与计算的数值是否全部有限。
            torch.isfinite(styles).all()
        ):
            raise ValueError("预测或十风格矩阵含 NaN/Inf")  # 输入数据不满足业务或数值约束时立即终止。

        _, compact = torch.unique(codes.long(), sorted=True, return_inverse=True)  # 压缩当日行业编码。
        group_count = int(compact.max().item()) + 1  # 计算group数量。
        counts = torch.zeros(group_count, dtype=torch.float32, device=values.device)  # 初始化数量。
        counts.scatter_add_(0, compact, torch.ones_like(values))
        value_sums = torch.zeros_like(counts).scatter_add(0, compact, values)  # 初始化值sums。
        style_sums = torch.zeros(  # 初始化风格sums。
            group_count,
            len(STYLE_COLUMNS),
            dtype=torch.float32,  # 统一结果的数据类型。
            device=values.device,  # 指定张量所在的计算设备。
        )
        style_sums.index_add_(0, compact, styles)
        values_within = values - (value_sums / counts)[compact]
        styles_within = styles - (style_sums / counts[:, None])[compact]

        q, r = torch.linalg.qr(styles_within, mode="reduced")  # q:[N,10]，r:[10,10]。
        diagonal = torch.abs(torch.diagonal(r))
        rank_tolerance = (  # 计算排名tolerance。
            torch.finfo(torch.float32).eps
            * max(styles_within.shape)
            * diagonal.max()
        )
        if bool((diagonal <= rank_tolerance).any()):  # 检查当前掩码或数据是否满足有效性要求。
            raise np.linalg.LinAlgError("行业内十风格矩阵秩不足")  # 回归设计矩阵无法稳定求解时立即终止。
        residual = values_within - q @ (q.transpose(0, 1) @ values_within)  # [10,N]@[N] -> [10] -> [N]。
        centered = values - values.mean()  # 计算中心化。
        total_sum_squares = centered.square().sum()  # 计算totalsumsquares。
        if not bool(torch.isfinite(total_sum_squares)) or bool(  # 检查参与计算的数值是否全部有限。
            total_sum_squares <= torch.finfo(torch.float32).eps
        ):
            raise ValueError("单日 raw score 为常数")  # 输入数据不满足业务或数值约束时立即终止。
        residual_sum_squares = residual.square().sum()  # 计算残差sumsquares。
        raw_r_squared = 1.0 - residual_sum_squares / total_sum_squares  # 计算原始rsquared。
        if not bool(torch.isfinite(raw_r_squared)) or bool(  # 检查参与计算的数值是否全部有限。
            (raw_r_squared < -1e-4) | (raw_r_squared > 1.0 + 1e-4)
        ):
            raise FloatingPointError(  # 检测到 NaN、Inf 或退化数值时立即终止。
                f"联合 Barra R² 超出数值范围：{float(raw_r_squared.detach())}"
            )
        r_squared = raw_r_squared.clamp(0.0, 1.0)  # 约束数值误差后的统计量。
        residual_sums = torch.zeros_like(counts).scatter_add(  # 初始化残差sums。
            0, compact, residual
        )
        return r_squared, {  # 返回可微联合 R²。
            "r_squared": r_squared.detach(),  # 记录rsquared。
            "industry_count": torch.tensor(  # 记录行业数量。
                group_count, dtype=torch.int64, device=values.device  # 计算group数量。
            ),
            "matrix_rank": torch.tensor(  # 记录回归设计矩阵的实际秩。
                len(STYLE_COLUMNS), dtype=torch.int64, device=values.device
            ),
            "residual_sum_squares": residual_sum_squares.detach(),  # 记录残差sumsquares。
            "total_sum_squares": total_sum_squares.detach(),  # 记录totalsumsquares。
            "max_abs_industry_residual_mean": (  # 记录maxabs行业残差均值。
                residual_sums / counts
            ).abs().max().detach(),
        }


def project_out_full_barra_torch(  # 投影掉十个连续 Barra 暴露与行业固定效应，梯度回传到分数。
    predictions: torch.Tensor,  # [N]，单日截面分数。
    style_exposures: torch.Tensor,  # [N,10]，单日十风格暴露。
    industry_codes: torch.Tensor,  # [N]，单日行业代码。
    *,
    standardize: bool = True,  # 控制标准化开关。
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """投影掉十个连续 Barra 暴露与行业固定效应，梯度回传到分数。"""

    device_type = predictions.device.type
    with torch.autocast(device_type=device_type, enabled=False):  # QR 与 R² 固定用 FP32，避免混合精度放大秩误差。
        values = predictions.float().reshape(-1)  # [N,...] -> [N]。
        styles = style_exposures.float()  # 转换风格的数据类型或存储位置。
        codes = industry_codes.reshape(-1)  # [N,...] -> [N]。
        integer_dtypes = (
            torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8
        )
        if styles.shape != (len(values), len(STYLE_COLUMNS)):  # 检查张量或数组形状是否符合接口约定。
            raise ValueError("预测或十风格矩阵形状异常")  # 输入数据不满足业务或数值约束时立即终止。
        if codes.shape != values.shape or codes.dtype not in integer_dtypes:  # 检查张量或数组形状是否符合接口约定。
            raise ValueError("行业代码必须是一维整数张量且与预测等长")  # 输入数据不满足业务或数值约束时立即终止。
        if len(values) < len(STYLE_COLUMNS) + 2:  # 检查当前样本数量是否满足计算要求。
            raise ValueError("单日股票数不足以进行完整 Barra 投影")  # 输入数据不满足业务或数值约束时立即终止。
        if bool(  # 根据当前数据状态选择处理分支。
            ((codes < 0) | (codes > EXPECTED_CNE5S_INDUSTRIES)).any()
        ):
            raise ValueError("行业代码超出 0 至 32 的有效范围")  # 输入数据不满足业务或数值约束时立即终止。
        if not bool(torch.isfinite(values).all()) or not bool(  # 检查参与计算的数值是否全部有限。
            torch.isfinite(styles).all()
        ):
            raise ValueError("预测或十风格矩阵含 NaN/Inf")  # 输入数据不满足业务或数值约束时立即终止。

        _, compact = torch.unique(  # 压缩当日行业编码。
            codes.long(), sorted=True, return_inverse=True
        )
        group_count = int(compact.max().item()) + 1  # 计算group数量。
        counts = torch.zeros(  # 初始化数量。
            group_count, dtype=torch.float32, device=values.device  # 计算group数量。
        )
        counts.scatter_add_(0, compact, torch.ones_like(values))
        value_sums = torch.zeros_like(counts).scatter_add(  # 初始化值sums。
            0, compact, values
        )
        style_sums = torch.zeros(  # 初始化风格sums。
            group_count,
            len(STYLE_COLUMNS),
            dtype=torch.float32,  # 统一结果的数据类型。
            device=values.device,  # 指定张量所在的计算设备。
        )
        style_sums.index_add_(0, compact, styles)
        values_within = values - (value_sums / counts)[compact]
        styles_within = styles - (style_sums / counts[:, None])[compact]

        q, r = torch.linalg.qr(styles_within, mode="reduced")  # q:[N,10]，r:[10,10]。
        diagonal = torch.abs(torch.diagonal(r))
        rank_tolerance = (  # 计算排名tolerance。
            torch.finfo(torch.float32).eps
            * max(styles_within.shape)
            * diagonal.max()
        )
        if bool((diagonal <= rank_tolerance).any()):  # 检查当前掩码或数据是否满足有效性要求。
            raise np.linalg.LinAlgError("行业内十风格矩阵秩不足")  # 回归设计矩阵无法稳定求解时立即终止。
        residual = values_within - q @ (  # 最终残差保持 [N]。
            q.transpose(0, 1) @ values_within  # [10,N]@[N] -> [10]。
        )
        if standardize:  # 根据当前数据状态选择处理分支。
            centered = residual - residual.mean()  # 计算中心化。
            standard_deviation = centered.std(unbiased=False)  # 计算standarddeviation。
            if not bool(torch.isfinite(standard_deviation)) or bool(  # 检查参与计算的数值是否全部有限。
                standard_deviation <= torch.finfo(torch.float32).eps
            ):
                raise ValueError("完整 Barra 投影后的分数为常数")  # 输入数据不满足业务或数值约束时立即终止。
            output = centered / standard_deviation
        else:  # 处理上述条件均不成立的情况。
            output = residual

        style_centered = styles - styles.mean(dim=0, keepdim=True)  # 计算风格中心化。
        denominator = torch.sqrt(  # 计算截面标准差。
            output.square().sum()
            * style_centered.square().sum(dim=0)
        )
        if bool((denominator <= torch.finfo(torch.float32).eps).any()):  # 检查当前掩码或数据是否满足有效性要求。
            raise ValueError("完整 Barra 投影后的风格相关性不可计算")  # 输入数据不满足业务或数值约束时立即终止。
        correlations = (  # 计算correlations。
            output[:, None] * style_centered
        ).sum(dim=0) / denominator
        residual_sums = torch.zeros_like(counts).scatter_add(  # 初始化残差sums。
            0, compact, output
        )
        return output, {  # 返回投影掉十个连续 Barra 暴露与行业固定效应，梯度回传到分数。
            "industry_count": torch.tensor(  # 记录行业数量。
                group_count, dtype=torch.int64, device=values.device  # 计算group数量。
            ),
            "matrix_rank": torch.tensor(  # 记录回归设计矩阵的实际秩。
                len(STYLE_COLUMNS), dtype=torch.int64, device=values.device
            ),
            "condition_number": (  # 记录conditionnumber。
                diagonal.max() / diagonal.min()
            ).detach(),
            "max_abs_style_pearson": correlations.abs().max().detach(),  # 记录maxabs风格Pearson。
            "max_abs_industry_mean": (  # 记录maxabs行业均值。
                residual_sums / counts
            ).abs().max().detach(),
            "output_mean": output.mean().detach(),  # 记录输出均值。
            "output_std": output.std(unbiased=False).detach(),  # 记录输出标准差。
        }


def _industry_sidecar_payload(  # 处理行业代码或行业诊断，并保留 UNKNOWN 口径。
    sample_index_path: Path,  # 接收样本索引路径。
    cne5s_path: Path,  # 接收cne5s路径。
    style_store_signature: str,  # 接收风格存储签名。
    minimum_match_rate: float,  # 接收minimummatch比率。
) -> dict[str, Any]:
    """处理行业代码或行业诊断，并保留 UNKNOWN 口径。"""
    return {  # 返回处理行业代码或行业诊断，并保留 UNKNOWN 口径。
        "schema_version": INDUSTRY_SIDECAR_SCHEMA_VERSION,  # 记录schema版本。
        "sample_index": _file_fingerprint(sample_index_path),  # 记录样本索引。
        "cne5s": barra._fingerprint_parquet(cne5s_path),  # 记录cne5s。
        "style_store_signature": style_store_signature,  # 记录风格存储签名，用于来源一致性校验。
        "industry_column": barra.INDUSTRY_COLUMN,  # 记录行业字段。
        "unknown_industry": barra.UNKNOWN_INDUSTRY,  # 记录未知行业行业。
        "encoding": "unknown_zero_known_labels_first_seen_year_then_sorted",  # 把 encoding 映射为"unknown_zero_known_labels_first_seen_year_then_sorted"。
        "minimum_match_rate": minimum_match_rate,  # 记录minimummatch比例。
        "match_policy": "exact_trade_date_and_assetid_no_forward_fill",  # 把 match_policy 映射为"exact_trade_date_and_assetid_no_forward_fill"。
    }


def build_industry_sidecar(  # 构建与 sample_index 严格对齐的 CNE5S 行业整数代码。
    *,
    store_dir: str | Path = DEFAULT_INDUSTRY_STORE_DIR,  # 接收存储目录。
    sample_index_path: str | Path = base.SAMPLE_INDEX_PATH,  # 接收样本索引路径。
    cne5s_path: str | Path = barra.DEFAULT_CNE5S_PATH,  # 接收cne5s路径。
    minimum_match_rate: float = MINIMUM_MATCH_RATE,  # 接收minimummatch比率。
    overwrite: bool = False,  # 接收overwrite。
) -> dict[str, Any]:
    """构建与 sample_index 严格对齐的 CNE5S 行业整数代码。"""

    if not 0 <= minimum_match_rate <= 1:  # 检查当前数值是否越过业务或数值阈值。
        raise ValueError("minimum_match_rate 必须位于 [0,1]")  # 输入数据不满足业务或数值约束时立即终止。
    target = Path(store_dir).resolve()  # 构造标签。
    sample_path = Path(sample_index_path).resolve()  # 构造样本路径。
    cne5s = Path(cne5s_path).resolve()  # 构造cne5s。
    style_manifest = require_style_neutral_store()
    payload = _industry_sidecar_payload(
        sample_path, cne5s, style_manifest["signature"], minimum_match_rate
    )
    signature = _stable_hash(payload)  # 生成数据复用签名。
    manifest_path = target / INDUSTRY_MANIFEST_FILE  # 构造清单路径。
    if manifest_path.exists() and not overwrite:  # 检查目标文件或目录是否满足存在性要求。
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))  # 读取existing。
        if existing.get("signature") != signature:  # 按当前模式、标识或状态选择处理分支。
            raise RuntimeError(  # 当前运行环境或任务状态不支持继续执行。
                "已有行业 sidecar 签名不同；请显式设置 overwrite=True"
            )
        _validate_industry_sidecar(target, existing)
        return {**existing, "reused": True}  # 返回构建与 sample_index 严格对齐的 CNE5S 行业整数代码。

    sample = pd.read_parquet(  # sidecar 必须逐行对应 feature-store sample_index。
        sample_path,
        columns=["sample_index", "trade_date", "instrument_id"],  # 限定本次读取或输出使用的字段。
    )
    sample["trade_date"] = sample["trade_date"].astype(str)  # 转换样本的数据类型或存储位置。
    sample["instrument_id"] = sample["instrument_id"].astype(str)  # 转换样本的数据类型或存储位置。
    expected_indices = np.arange(len(sample), dtype=np.int64)
    if not np.array_equal(  # 检查必需数据、配置或校验结果是否缺失。
        sample["sample_index"].to_numpy(np.int64), expected_indices
    ):
        raise AssertionError("sample_index 必须与物理行号逐行一致")  # 关键数据契约被破坏时立即终止。

    target.parent.mkdir(parents=True, exist_ok=True)  # 确保目标目录存在。
    stage = Path(  # 构造stage。
        tempfile.mkdtemp(prefix=f".{target.name}.stage.", dir=target.parent)
    )
    try:  # 执行可能失败的外部读取或计算步骤。
        output = np.lib.format.open_memmap(  # 使用 memmap，避免整库载入内存。
            stage / INDUSTRY_CODES_FILE,
            mode="w+",  # 选择当前处理模式。
            dtype=np.int16,  # 统一结果的数据类型。
            shape=(len(sample),),  # 配置形状。
        )
        output[:] = -1  # -1 仅用于检测尚未写入的样本。
        mapping: dict[str, int] = {barra.UNKNOWN_INDUSTRY: 0}  # 0 固定表示未知行业，供 Embedding padding_idx 使用。
        audit: list[dict[str, Any]] = []  # 初始化audit容器。
        years = sorted(sample["trade_date"].str[:4].astype(int).unique())  # 计算年份以固定样本顺序。
        for year in years:  # 遍历年份，逐项处理年份。
            yearly = sample.loc[
                sample["trade_date"].str.startswith(str(year))
            ].copy()
            yearly["assetid"] = barra.instrument_ids_to_assetids(  # 转换为 CNE5S assetid。
                yearly["instrument_id"]
            )
            cne5s_year = barra._load_cne5s_year(cne5s, year)
            joined = barra._join_prediction_and_cne5s(
                yearly, cne5s_year, year
            )
            matched = joined["barra_matched"].to_numpy(bool)  # 按股票与交易日业务键对齐。
            match_rate = float(matched.mean())  # 计算match比率。
            if match_rate < minimum_match_rate:  # 检查当前数值是否越过业务或数值阈值。
                raise AssertionError(  # 关键数据契约被破坏时立即终止。
                    f"{year} CNE5S 匹配率 {match_rate:.6%} "
                    f"< {minimum_match_rate:.6%}"
                )
            industries = joined[barra.INDUSTRY_COLUMN].astype("object").copy()  # 转换行业的数据类型或存储位置。
            industries.loc[~joined["barra_matched"]] = barra.UNKNOWN_INDUSTRY
            if industries.isna().any():  # 检查当前掩码或数据是否满足有效性要求。
                raise ValueError(f"{year} 行业字段含未处理缺失值")  # 输入数据不满足业务或数值约束时立即终止。
            labels = industries.astype(str)  # 转换标签的数据类型或存储位置。
            new_labels = sorted(set(labels.unique()).difference(mapping))  # 已有代码不变，只为首次出现的行业分配新码。
            for label in new_labels:  # 遍历new标签，逐项处理标签。
                mapping[label] = len(mapping)
            encoded = labels.map(mapping)
            if encoded.isna().any():  # 检查当前掩码或数据是否满足有效性要求。
                raise AssertionError(f"{year} 存在未编码行业")  # 关键数据契约被破坏时立即终止。
            sample_indices = joined["sample_index"].to_numpy(np.int64)  # 按股票与交易日业务键对齐。
            output[sample_indices] = encoded.to_numpy(np.int16)  # 按全局样本行号写回，保持 sidecar 逐行对齐。
            audit.append(  # 把当前结果追加到汇总列表。
                {
                    "year": int(year),  # 记录年份。
                    "rows": len(joined),  # 记录行数。
                    "matched_rows": int(matched.sum()),  # 记录匹配行数。
                    "match_rate": match_rate,  # 记录match比例。
                    "known_industries_to_date": len(mapping) - 1,  # 记录已知行业行业to交易日。
                }
            )

        output.flush()  # 将 memmap 修改刷入磁盘。
        del output  # 释放不再使用的中间引用。
        codes = np.load(stage / INDUSTRY_CODES_FILE, mmap_mode="r")  # 读取代码。
        if (codes < 0).any():  # 检查当前掩码或数据是否满足有效性要求。
            raise AssertionError("行业 sidecar 存在未写入样本")  # 关键数据契约被破坏时立即终止。
        known_count = len(mapping) - 1  # 构造已知行业数量，限定后续参与计算的样本。
        if known_count != EXPECTED_CNE5S_INDUSTRIES:  # 按当前模式、标识或状态选择处理分支。
            raise ValueError(  # 输入数据不满足业务或数值约束时立即终止。
                f"CNE5S 行业数为 {known_count}，预期 {EXPECTED_CNE5S_INDUSTRIES}"
            )
        pd.DataFrame(audit).to_parquet(  # 把表格按约定格式写入 Parquet。
            stage / INDUSTRY_AUDIT_FILE, index=False, compression="zstd"
        )
        manifest = {  # 初始化清单容器。
            "signature": signature,  # 记录签名。
            "payload": payload,  # 记录载荷。
            "sample_count": len(sample),  # 记录样本数量。
            "industry_mapping": mapping,  # 记录行业mapping。
            "known_industry_count": known_count,  # 记录已知行业行业数量。
            "files": {  # 记录files。
                "industry_codes": INDUSTRY_CODES_FILE,  # 记录行业代码。
                "match_audit": INDUSTRY_AUDIT_FILE,  # 记录matchaudit。
            },
            "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),  # 记录createdatutc。
        }
        (stage / INDUSTRY_MANIFEST_FILE).write_text(  # 把文本结果写入临时或目标文件。
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",  # 使用 UTF-8 编解码文本。
        )
        _validate_industry_sidecar(stage, manifest)
        _atomic_replace_directory(stage, target)  # 原子落盘，避免半成品。
        return {**manifest, "reused": False}  # 返回构建与 sample_index 严格对齐的 CNE5S 行业整数代码。
    except BaseException:  # 捕获预期异常并执行对应恢复逻辑。
        if stage.exists():  # 检查目标文件或目录是否满足存在性要求。
            shutil.rmtree(stage)
        raise


def _full_barra_target_payload(  # 构造或校验完整 Barra 风格加行业的数据契约。
    targets_path: Path,  # 接收标签路径。
    style_store_signature: str,  # 接收风格存储签名。
    industry_sidecar_signature: str,  # 接收行业sidecar签名。
) -> dict[str, Any]:
    """构造或校验完整 Barra 风格加行业的数据契约。"""
    return {  # 返回构造或校验完整 Barra 风格加行业的数据契约。
        "schema_version": FULL_BARRA_TARGET_SCHEMA_VERSION,  # 记录schema版本。
        "source_targets": _file_fingerprint(targets_path),  # 记录来源标签。
        "style_store_signature": style_store_signature,  # 记录风格存储签名，用于来源一致性校验。
        "industry_sidecar_signature": industry_sidecar_signature,  # 记录行业sidecar签名，用于来源一致性校验。
        "regression": (  # 记录regression。
            "daily_equal_weight_ols_intercept_plus_10_styles_plus_"
            "industry_fixed_effects"
        ),
        "solver": "industry_within_then_numpy_lstsq",  # 把 solver 映射为"industry_within_then_numpy_lstsq"。
        "standardization": "residual_zscore_ddof0",  # 把 standardization 映射为"residual_zscore_ddof0"。
        "unknown_industry_policy": "code_zero_as_separate_fixed_effect_group",  # 把 unknown_industry_policy 映射为"code_zero_as_separate_fixed_effect_group"。
        "missing_style_policy": (  # 声明缺失值风格处理口径。
            "daily_standardized_exposure_zero_fill_with_valid_mask_"
            "no_forward_fill"
        ),
    }


def build_full_barra_target_sidecar(  # 逐日生成截距、十风格与行业固定效应联合残差标签。
    *,
    store_dir: str | Path = DEFAULT_FULL_BARRA_TARGET_STORE_DIR,  # 接收存储目录。
    sample_index_path: str | Path = base.SAMPLE_INDEX_PATH,  # 接收样本索引路径。
    targets_path: str | Path = base.TARGETS_PATH,  # 接收标签路径。
    overwrite: bool = False,  # 接收overwrite。
) -> dict[str, Any]:
    """逐日生成截距、十风格与行业固定效应联合残差标签。"""

    target = Path(store_dir).resolve()  # 构造标签。
    sample_path = Path(sample_index_path).resolve()  # 构造样本路径。
    source_targets_path = Path(targets_path).resolve()  # 构造来源标签路径。
    style_manifest = require_style_neutral_store()
    industry_manifest = require_industry_sidecar()
    payload = _full_barra_target_payload(
        source_targets_path,
        style_manifest["signature"],
        industry_manifest["signature"],
    )
    signature = _stable_hash(payload)  # 生成数据复用签名。
    manifest_path = target / FULL_BARRA_TARGET_MANIFEST_FILE  # 构造清单路径。
    if manifest_path.exists() and not overwrite:  # 检查目标文件或目录是否满足存在性要求。
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))  # 读取existing。
        if existing.get("signature") != signature:  # 按当前模式、标识或状态选择处理分支。
            raise RuntimeError(  # 当前运行环境或任务状态不支持继续执行。
                "已有完整 Barra 残差标签签名不同；"
                "请显式设置 overwrite=True"
            )
        _validate_full_barra_target_sidecar(target, existing)
        return {**existing, "reused": True}  # 返回逐日计算结果与诊断。

    sample = pd.read_parquet(  # 读取样本。
        sample_path,
        columns=["sample_index", "trade_date", "label_valid"],  # 限定本次读取或输出使用的字段。
    )
    sample["trade_date"] = sample["trade_date"].astype(str)  # 转换样本的数据类型或存储位置。
    expected_indices = np.arange(len(sample), dtype=np.int64)
    if not np.array_equal(  # 检查必需数据、配置或校验结果是否缺失。
        sample["sample_index"].to_numpy(np.int64), expected_indices
    ):
        raise AssertionError("sample_index 必须与物理行号逐行一致")  # 关键数据契约被破坏时立即终止。
    label_valid = sample["label_valid"].to_numpy(bool)  # 转换并统一标签有效的数据表示。
    source_targets = np.load(source_targets_path, mmap_mode="r")  # 读取来源标签。
    styles = np.load(  # 读取风格。
        DEFAULT_STORE_DIR / STYLE_EXPOSURES_FILE, mmap_mode="r"
    )
    industry_codes = np.load(  # 读取行业代码。
        DEFAULT_INDUSTRY_STORE_DIR / INDUSTRY_CODES_FILE, mmap_mode="r"
    )
    if source_targets.shape != (len(sample),):  # 检查张量或数组形状是否符合接口约定。
        raise ValueError("源 targets.npy 与 sample_index 行数不一致")  # 输入数据不满足业务或数值约束时立即终止。
    if styles.shape != (len(sample), len(STYLE_COLUMNS)):  # 检查张量或数组形状是否符合接口约定。
        raise ValueError("十风格矩阵与 sample_index 行数不一致")  # 输入数据不满足业务或数值约束时立即终止。
    if industry_codes.shape != (len(sample),):  # 检查张量或数组形状是否符合接口约定。
        raise ValueError("行业代码与 sample_index 行数不一致")  # 输入数据不满足业务或数值约束时立即终止。
    if not np.array_equal(np.isfinite(source_targets), label_valid):  # 检查参与计算的数值是否全部有限。
        raise AssertionError("源标签有限性与 label_valid 不一致")  # 关键数据契约被破坏时立即终止。

    target.parent.mkdir(parents=True, exist_ok=True)  # 确保目标目录存在。
    stage = Path(  # 构造stage。
        tempfile.mkdtemp(prefix=f".{target.name}.stage.", dir=target.parent)
    )
    try:  # 执行可能失败的外部读取或计算步骤。
        output = np.lib.format.open_memmap(  # 使用 memmap，避免整库载入内存。
            stage / FULL_BARRA_TARGETS_FILE,
            mode="w+",  # 选择当前处理模式。
            dtype=np.float32,  # 统一结果的数据类型。
            shape=(len(sample),),  # 配置形状。
        )
        output[:] = np.nan
        diagnostics: list[dict[str, Any]] = []  # 初始化诊断容器。
        years = sorted(sample["trade_date"].str[:4].astype(int).unique())  # 计算年份以固定样本顺序。
        for year in years:  # 遍历年份，逐项处理年份。
            yearly = sample.loc[
                sample["trade_date"].str.startswith(str(year))
            ]
            for trade_date, positions in yearly.groupby(  # 每个交易日独立拟合，禁止跨日使用暴露或标签。
                "trade_date", sort=True
            ).groups.items():
                daily = sample.loc[positions]
                indices = daily["sample_index"].to_numpy(np.int64)  # 转换并统一索引的数据表示。
                residual, record = residualize_daily_targets_with_industry(  # 按日投影 Barra 暴露。
                    np.asarray(source_targets[indices], dtype=np.float64),
                    np.asarray(styles[indices], dtype=np.float64),
                    np.asarray(industry_codes[indices], dtype=np.int16),
                    label_valid[indices],
                )
                output[indices] = residual.astype(np.float32)  # 转换输出的数据类型或存储位置。
                diagnostics.append(  # 把当前结果追加到汇总列表。
                    {
                        "year": int(year),  # 记录年份。
                        "trade_date": str(trade_date),  # 记录trade交易日。
                        **record,
                    }
                )
            print(  # 输出当前任务的进度或诊断信息。
                f"{year}：写入完整 Barra 残差标签 {len(yearly):,} 行",
                flush=True,  # 配置flush。
            )

        output.flush()  # 将 memmap 修改刷入磁盘。
        del output  # 释放不再使用的中间引用。
        residual_check = np.load(  # 读取残差检查。
            stage / FULL_BARRA_TARGETS_FILE, mmap_mode="r"
        )
        if not np.array_equal(np.isfinite(residual_check), label_valid):  # 检查参与计算的数值是否全部有限。
            raise AssertionError(  # 关键数据契约被破坏时立即终止。
                "完整 Barra 残差标签有限性与 label_valid 不一致"
            )
        daily = pd.DataFrame(diagnostics)
        daily.to_parquet(  # 把表格按约定格式写入 Parquet。
            stage / FULL_BARRA_DAILY_DIAGNOSTICS_FILE,
            index=False,  # 控制是否把 DataFrame 索引写入结果。
            compression="zstd",  # 指定落盘压缩格式。
        )
        annual = daily.groupby("year", as_index=False).agg(  # 按既定口径聚合。
            rows=("rows", "sum"),  # 配置行数。
            label_valid_rows=("label_valid_rows", "sum"),  # 配置标签有效行数。
            dates=("trade_date", "nunique"),  # 配置交易日。
            mean_industry_count=("industry_count", "mean"),  # 配置均值行业数量。
            mean_unknown_industry_rate=("unknown_industry_rate", "mean"),  # 配置均值未知行业行业比率。
            mean_joint_barra_r_squared=("joint_barra_r_squared", "mean"),  # 配置均值jointBarrarsquared。
            max_abs_style_residual_pearson=(
                "max_abs_style_residual_pearson", "max"
            ),
            max_abs_industry_residual_mean=(  # 计算maxabs行业残差均值。
                "max_abs_industry_residual_mean", "max"
            ),
            max_condition_number=("condition_number", "max"),  # 配置maxconditionnumber。
        )
        annual.to_parquet(  # 把表格按约定格式写入 Parquet。
            stage / FULL_BARRA_ANNUAL_SUMMARY_FILE,
            index=False,  # 控制是否把 DataFrame 索引写入结果。
            compression="zstd",  # 指定落盘压缩格式。
        )
        manifest = {  # 初始化清单容器。
            "signature": signature,  # 记录签名。
            "payload": payload,  # 记录载荷。
            "sample_count": len(sample),  # 记录样本数量。
            "label_valid_count": int(label_valid.sum()),  # 记录标签有效数量。
            "first_date": str(sample["trade_date"].min()),  # 记录首个交易日。
            "last_date": str(sample["trade_date"].max()),  # 记录最后一个交易日。
            "files": {  # 记录files。
                "residual_targets": FULL_BARRA_TARGETS_FILE,  # 记录残差标签。
                "daily_diagnostics": FULL_BARRA_DAILY_DIAGNOSTICS_FILE,  # 记录日度诊断。
                "annual_summary": FULL_BARRA_ANNUAL_SUMMARY_FILE,  # 记录年度汇总。
            },
            "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),  # 记录createdatutc。
        }
        (stage / FULL_BARRA_TARGET_MANIFEST_FILE).write_text(  # 把文本结果写入临时或目标文件。
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",  # 使用 UTF-8 编解码文本。
        )
        _validate_full_barra_target_sidecar(stage, manifest)
        _atomic_replace_directory(stage, target)  # 原子落盘，避免半成品。
        return {**manifest, "reused": False}  # 返回逐日计算结果与诊断。
    except BaseException:  # 捕获预期异常并执行对应恢复逻辑。
        if stage.exists():  # 检查目标文件或目录是否满足存在性要求。
            shutil.rmtree(stage)
        raise


def _dual_tower_target_payload(  # 构造或校验 Alpha/Style 双塔所需的数据契约。
    targets_path: Path,  # 接收标签路径。
    full_target_signature: str,  # 接收full标签签名。
    style_store_signature: str,  # 接收风格存储签名。
    industry_sidecar_signature: str,  # 接收行业sidecar签名。
) -> dict[str, Any]:
    """构造或校验 Alpha/Style 双塔所需的数据契约。"""
    return {  # 返回构造或校验 Alpha/Style 双塔所需的数据契约。
        "schema_version": DUAL_TOWER_TARGET_SCHEMA_VERSION,  # 记录schema版本。
        "source_targets": _file_fingerprint(targets_path),  # 记录来源标签。
        "full_barra_target_signature": full_target_signature,  # 记录完整Barra标签签名，用于来源一致性校验。
        "style_store_signature": style_store_signature,  # 记录风格存储签名，用于来源一致性校验。
        "industry_sidecar_signature": industry_sidecar_signature,  # 记录行业sidecar签名，用于来源一致性校验。
        "regression": (  # 记录regression。
            "daily_equal_weight_ols_intercept_plus_10_styles_plus_"
            "industry_fixed_effects"
        ),
        "solver": "industry_within_then_numpy_lstsq",  # 把 solver 映射为"industry_within_then_numpy_lstsq"。
        "target": "daily_full_barra_fitted_component_zscore_ddof0",  # 把 target 映射为"daily_full_barra_fitted_component_zscore_ddof0"。
        "unknown_industry_policy": "code_zero_as_separate_fixed_effect_group",  # 把 unknown_industry_policy 映射为"code_zero_as_separate_fixed_effect_group"。
        "missing_style_policy": (  # 声明缺失值风格处理口径。
            "daily_standardized_exposure_zero_fill_with_valid_mask_"
            "no_forward_fill"
        ),
    }


def build_dual_tower_target_sidecar(  # 逐日生成完整 Barra 回归拟合收益，供 No.6 Style 塔监督。
    *,
    store_dir: str | Path = DEFAULT_DUAL_TOWER_TARGET_STORE_DIR,  # 接收存储目录。
    sample_index_path: str | Path = base.SAMPLE_INDEX_PATH,  # 接收样本索引路径。
    targets_path: str | Path = base.TARGETS_PATH,  # 接收标签路径。
    overwrite: bool = False,  # 接收overwrite。
) -> dict[str, Any]:
    """逐日生成完整 Barra 回归拟合收益，供 No.6 Style 塔监督。"""

    target = Path(store_dir).resolve()  # 构造标签。
    sample_path = Path(sample_index_path).resolve()  # 构造样本路径。
    source_targets_path = Path(targets_path).resolve()  # 构造来源标签路径。
    style_manifest = require_style_neutral_store()
    industry_manifest = require_industry_sidecar()
    full_manifest = require_full_barra_target_sidecar()
    payload = _dual_tower_target_payload(
        source_targets_path,
        full_manifest["signature"],
        style_manifest["signature"],
        industry_manifest["signature"],
    )
    signature = _stable_hash(payload)  # 生成数据复用签名。
    manifest_path = target / DUAL_TOWER_TARGET_MANIFEST_FILE  # 构造清单路径。
    if manifest_path.exists() and not overwrite:  # 检查目标文件或目录是否满足存在性要求。
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))  # 读取existing。
        if existing.get("signature") != signature:  # 按当前模式、标识或状态选择处理分支。
            raise RuntimeError(  # 当前运行环境或任务状态不支持继续执行。
                "已有 No.6 fitted-target sidecar 签名不同；"
                "请显式设置 overwrite=True"
            )
        _validate_dual_tower_target_sidecar(target, existing)
        return {**existing, "reused": True}  # 返回逐日计算结果与诊断。

    sample = pd.read_parquet(  # 读取样本。
        sample_path, columns=["sample_index", "trade_date", "label_valid"]  # 构造样本路径。
    )
    sample["trade_date"] = sample["trade_date"].astype(str)  # 转换样本的数据类型或存储位置。
    expected_indices = np.arange(len(sample), dtype=np.int64)
    if not np.array_equal(  # 检查必需数据、配置或校验结果是否缺失。
        sample["sample_index"].to_numpy(np.int64), expected_indices
    ):
        raise AssertionError("sample_index 必须与物理行号逐行一致")  # 关键数据契约被破坏时立即终止。
    label_valid = sample["label_valid"].to_numpy(bool)  # 转换并统一标签有效的数据表示。
    source_targets = np.load(source_targets_path, mmap_mode="r")  # 读取来源标签。
    styles = np.load(DEFAULT_STORE_DIR / STYLE_EXPOSURES_FILE, mmap_mode="r")  # 读取风格。
    industries = np.load(  # 读取行业。
        DEFAULT_INDUSTRY_STORE_DIR / INDUSTRY_CODES_FILE, mmap_mode="r"
    )
    existing_residual = np.load(  # 读取existing残差。
        DEFAULT_FULL_BARRA_TARGET_STORE_DIR / FULL_BARRA_TARGETS_FILE,
        mmap_mode="r",  # 使用内存映射避免一次性载入全部数据。
    )
    expected_shapes = (
        source_targets.shape == (len(sample),),
        styles.shape == (len(sample), len(STYLE_COLUMNS)),
        industries.shape == (len(sample),),
        existing_residual.shape == (len(sample),),
    )
    if not all(expected_shapes):  # 检查必需数据、配置或校验结果是否缺失。
        raise ValueError("No.6 标签输入与 sample_index 形状不一致")  # 输入数据不满足业务或数值约束时立即终止。

    target.parent.mkdir(parents=True, exist_ok=True)  # 确保目标目录存在。
    stage = Path(  # 构造stage。
        tempfile.mkdtemp(prefix=f".{target.name}.stage.", dir=target.parent)
    )
    try:  # 执行可能失败的外部读取或计算步骤。
        output = np.lib.format.open_memmap(  # 使用 memmap，避免整库载入内存。
            stage / DUAL_TOWER_FITTED_TARGETS_FILE,
            mode="w+",  # 选择当前处理模式。
            dtype=np.float32,  # 统一结果的数据类型。
            shape=(len(sample),),  # 配置形状。
        )
        output[:] = np.nan
        diagnostics: list[dict[str, Any]] = []  # 初始化诊断容器。
        years = sorted(sample["trade_date"].str[:4].astype(int).unique())  # 计算年份以固定样本顺序。
        for year in years:  # 遍历年份，逐项处理年份。
            yearly = sample.loc[sample["trade_date"].str.startswith(str(year))]
            for trade_date, positions in yearly.groupby(  # 每个交易日独立拟合，禁止跨日使用暴露或标签。
                "trade_date", sort=True
            ).groups.items():
                indices = sample.loc[positions, "sample_index"].to_numpy(np.int64)  # 转换并统一索引的数据表示。
                residual, fitted, record = decompose_daily_targets_with_industry(
                    np.asarray(source_targets[indices], dtype=np.float64),
                    np.asarray(styles[indices], dtype=np.float64),
                    np.asarray(industries[indices], dtype=np.int16),
                    label_valid[indices],
                )
                valid = label_valid[indices]  # 构造有效，限定后续参与计算的样本。
                residual_error = float(
                    np.max(
                        np.abs(
                            residual[valid]
                            - np.asarray(existing_residual[indices][valid], dtype=np.float64)
                        )
                    )
                )
                fitted_r_squared, _ = joint_barra_r_squared_numpy(  # 按股票与交易日业务键对齐。
                    fitted[valid],
                    np.asarray(styles[indices][valid], dtype=np.float64),
                    np.asarray(industries[indices][valid], dtype=np.int16),
                )
                output[indices] = fitted.astype(np.float32)  # 转换输出的数据类型或存储位置。
                diagnostics.append(  # 把当前结果追加到汇总列表。
                    {
                        "year": int(year),  # 记录年份。
                        "trade_date": str(trade_date),  # 记录trade交易日。
                        **record,
                        "residual_sidecar_max_abs_error": residual_error,  # 记录残差sidecarmaxabs误差。
                        "fitted_joint_barra_r_squared": fitted_r_squared,  # 记录拟合jointBarrarsquared。
                    }
                )
            print(  # 输出当前任务的进度或诊断信息。
                f"{year}：写入 No.6 fitted target {len(yearly):,} 行",
                flush=True,  # 配置flush。
            )
        output.flush()  # 将 memmap 修改刷入磁盘。
        del output  # 释放不再使用的中间引用。
        fitted_check = np.load(  # 读取拟合检查。
            stage / DUAL_TOWER_FITTED_TARGETS_FILE, mmap_mode="r"
        )
        if not np.array_equal(np.isfinite(fitted_check), label_valid):  # 检查参与计算的数值是否全部有限。
            raise AssertionError("No.6 fitted target 有限性与 label_valid 不一致")  # 关键数据契约被破坏时立即终止。
        daily = pd.DataFrame(diagnostics)
        daily.to_parquet(  # 把表格按约定格式写入 Parquet。
            stage / DUAL_TOWER_DAILY_DIAGNOSTICS_FILE,
            index=False,  # 控制是否把 DataFrame 索引写入结果。
            compression="zstd",  # 指定落盘压缩格式。
        )
        annual = daily.groupby("year", as_index=False).agg(  # 按既定口径聚合。
            rows=("rows", "sum"),  # 配置行数。
            label_valid_rows=("label_valid_rows", "sum"),  # 配置标签有效行数。
            dates=("trade_date", "nunique"),  # 配置交易日。
            mean_joint_barra_r_squared=("joint_barra_r_squared", "mean"),  # 配置均值jointBarrarsquared。
            mean_fitted_joint_barra_r_squared=(  # 计算均值拟合jointBarrarsquared。
                "fitted_joint_barra_r_squared", "mean"
            ),
            max_reconstruction_error=(
                "reconstruction_max_abs_error", "max"
            ),
            max_residual_sidecar_error=(
                "residual_sidecar_max_abs_error", "max"
            ),
            max_condition_number=("condition_number", "max"),  # 配置maxconditionnumber。
        )
        annual.to_parquet(  # 把表格按约定格式写入 Parquet。
            stage / DUAL_TOWER_ANNUAL_SUMMARY_FILE,
            index=False,  # 控制是否把 DataFrame 索引写入结果。
            compression="zstd",  # 指定落盘压缩格式。
        )
        manifest = {  # 初始化清单容器。
            "signature": signature,  # 记录签名。
            "payload": payload,  # 记录载荷。
            "sample_count": len(sample),  # 记录样本数量。
            "label_valid_count": int(label_valid.sum()),  # 记录标签有效数量。
            "first_date": str(sample["trade_date"].min()),  # 记录首个交易日。
            "last_date": str(sample["trade_date"].max()),  # 记录最后一个交易日。
            "files": {  # 记录files。
                "fitted_targets": DUAL_TOWER_FITTED_TARGETS_FILE,  # 记录拟合标签。
                "daily_diagnostics": DUAL_TOWER_DAILY_DIAGNOSTICS_FILE,  # 记录日度诊断。
                "annual_summary": DUAL_TOWER_ANNUAL_SUMMARY_FILE,  # 记录年度汇总。
            },
            "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),  # 记录createdatutc。
        }
        (stage / DUAL_TOWER_TARGET_MANIFEST_FILE).write_text(  # 把文本结果写入临时或目标文件。
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",  # 使用 UTF-8 编解码文本。
        )
        _validate_dual_tower_target_sidecar(stage, manifest)
        _atomic_replace_directory(stage, target)  # 原子落盘，避免半成品。
        return {**manifest, "reused": False}  # 返回逐日计算结果与诊断。
    except BaseException:  # 捕获预期异常并执行对应恢复逻辑。
        if stage.exists():  # 检查目标文件或目录是否满足存在性要求。
            shutil.rmtree(stage)
        raise


def _atomic_replace_directory(stage: Path, target: Path) -> None:  # 通过临时文件和原子替换写入产物，避免产生半成品。
    """通过临时文件和原子替换写入产物，避免产生半成品。"""
    backup = target.with_name(  # 构造backup。
        f".{target.name}.backup.{os.getpid()}.{time.time_ns()}"
    )
    if target.exists():  # 检查目标文件或目录是否满足存在性要求。
        target.rename(backup)
    try:  # 执行可能失败的外部读取或计算步骤。
        stage.rename(target)
    except BaseException:  # 捕获预期异常并执行对应恢复逻辑。
        if backup.exists() and not target.exists():  # 检查目标文件或目录是否满足存在性要求。
            backup.rename(target)
        raise
    if backup.exists():  # 检查目标文件或目录是否满足存在性要求。
        shutil.rmtree(backup)


def build_style_neutral_store(  # 构建与 feature store 行号严格对齐的十风格矩阵和残差标签。
    *,
    store_dir: str | Path = DEFAULT_STORE_DIR,  # 接收存储目录。
    sample_index_path: str | Path = base.SAMPLE_INDEX_PATH,  # 接收样本索引路径。
    targets_path: str | Path = base.TARGETS_PATH,  # 接收标签路径。
    cne5s_path: str | Path = barra.DEFAULT_CNE5S_PATH,  # 接收cne5s路径。
    eod_path: str | Path = barra.DEFAULT_EOD_PATH,  # 接收日行情路径。
    calendar_path: str | Path = barra.DEFAULT_CALENDAR_PATH,  # 接收交易日历路径。
    minimum_match_rate: float = MINIMUM_MATCH_RATE,  # 接收minimummatch比率。
    overwrite: bool = False,  # 接收overwrite。
) -> dict[str, Any]:
    """构建与 feature store 行号严格对齐的十风格矩阵和残差标签。"""

    if not 0 <= minimum_match_rate <= 1:  # 检查当前数值是否越过业务或数值阈值。
        raise ValueError("minimum_match_rate 必须位于 [0,1]")  # 输入数据不满足业务或数值约束时立即终止。
    target = Path(store_dir).resolve()  # 构造标签。
    sample_path = Path(sample_index_path).resolve()  # 构造样本路径。
    source_targets_path = Path(targets_path).resolve()  # 构造来源标签路径。
    cne5s = Path(cne5s_path).resolve()  # 构造cne5s。
    eod = Path(eod_path).resolve()  # 构造日行情。
    calendar_file = Path(calendar_path).resolve()  # 构造交易日历文件。
    payload = _store_payload(
        sample_path, source_targets_path, cne5s, eod, calendar_file
    )
    signature = _stable_hash(payload)  # 生成数据复用签名。
    manifest_path = target / MANIFEST_FILE  # 构造清单路径。
    if manifest_path.exists() and not overwrite:  # 检查目标文件或目录是否满足存在性要求。
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))  # 读取existing。
        if existing.get("signature") != signature:  # 按当前模式、标识或状态选择处理分支。
            raise RuntimeError(  # 当前运行环境或任务状态不支持继续执行。
                "已有风格中性训练数据签名不同；请显式设置 overwrite=True"
            )
        _validate_arrays(target, int(existing["sample_count"]))
        return {**existing, "reused": True}  # 返回构建与 feature store 行号严格对齐的十风格矩阵和残差标签。

    sample = pd.read_parquet(  # 读取样本。
        sample_path,
        columns=[  # 初始化列容器。
            "sample_index", "trade_date", "instrument_id", "label_valid"
        ],
    )
    sample["trade_date"] = sample["trade_date"].astype(str)  # 转换样本的数据类型或存储位置。
    sample["instrument_id"] = sample["instrument_id"].astype(str)  # 转换样本的数据类型或存储位置。
    expected_indices = np.arange(len(sample), dtype=np.int64)
    if not np.array_equal(  # 检查必需数据、配置或校验结果是否缺失。
        sample["sample_index"].to_numpy(np.int64), expected_indices
    ):
        raise AssertionError("sample_index 必须与物理行号逐行一致")  # 关键数据契约被破坏时立即终止。
    source_targets = np.load(source_targets_path, mmap_mode="r")  # 读取来源标签。
    if source_targets.shape != (len(sample),):  # 检查张量或数组形状是否符合接口约定。
        raise ValueError("源 targets.npy 与 sample_index 行数不一致")  # 输入数据不满足业务或数值约束时立即终止。
    label_valid = sample["label_valid"].to_numpy(bool)  # 转换并统一标签有效的数据表示。
    if not np.array_equal(np.isfinite(source_targets), label_valid):  # 检查参与计算的数值是否全部有限。
        raise AssertionError("源标签有限性与 label_valid 不一致")  # 关键数据契约被破坏时立即终止。

    target.parent.mkdir(parents=True, exist_ok=True)  # 确保目标目录存在。
    stage = Path(  # 构造stage。
        tempfile.mkdtemp(prefix=f".{target.name}.stage.", dir=target.parent)
    )
    try:  # 执行可能失败的外部读取或计算步骤。
        styles_output = np.lib.format.open_memmap(  # 使用 memmap，避免整库载入内存。
            stage / STYLE_EXPOSURES_FILE,
            mode="w+",  # 选择当前处理模式。
            dtype=np.float32,  # 统一结果的数据类型。
            shape=(len(sample), len(STYLE_COLUMNS)),  # 配置形状。
        )
        styles_output[:] = np.nan
        valid_output = np.lib.format.open_memmap(  # 使用 memmap，避免整库载入内存。
            stage / STYLE_VALID_FILE,
            mode="w+",  # 选择当前处理模式。
            dtype=np.bool_,  # 统一结果的数据类型。
            shape=(len(sample), len(STYLE_COLUMNS)),  # 配置形状。
        )
        valid_output[:] = False  # 构造有效输出，限定后续参与计算的样本。
        targets_output = np.lib.format.open_memmap(  # 使用 memmap，避免整库载入内存。
            stage / RESIDUAL_TARGETS_FILE,
            mode="w+",  # 选择当前处理模式。
            dtype=np.float32,  # 统一结果的数据类型。
            shape=(len(sample),),  # 配置形状。
        )
        targets_output[:] = np.nan

        calendar_dates = barra._load_sse_calendar(calendar_file)
        diagnostics: list[dict[str, Any]] = []  # 初始化诊断容器。
        years = sorted(sample["trade_date"].str[:4].astype(int).unique())  # 计算年份以固定样本顺序。
        for year in years:  # 遍历年份，逐项处理年份。
            yearly = sample.loc[
                sample["trade_date"].str.startswith(str(year))
            ].copy()
            yearly["assetid"] = barra.instrument_ids_to_assetids(  # 转换为 CNE5S assetid。
                yearly["instrument_id"]
            )
            yearly = barra._attach_reversal_20d(  # 按股票与日期业务键补充字段。
                yearly, eod, calendar_dates, year
            )
            cne5s_year = barra._load_cne5s_year(cne5s, year)
            joined = barra._join_prediction_and_cne5s(
                yearly, cne5s_year, year
            )
            match_rate = float(joined["barra_matched"].mean())  # 计算match比率。
            if match_rate < minimum_match_rate:  # 检查当前数值是否越过业务或数值阈值。
                raise AssertionError(  # 关键数据契约被破坏时立即终止。
                    f"{year} CNE5S 匹配率 {match_rate:.6%} "
                    f"< {minimum_match_rate:.6%}"
                )
            for trade_date, positions in joined.groupby(  # 遍历joinedgroupby，逐项处理trade交易日位置。
                "trade_date", sort=True
            ).groups.items():
                daily = joined.iloc[np.asarray(positions, dtype=np.int64)]  # 转换并统一daily的数据表示。
                sample_indices = daily["sample_index"].to_numpy(np.int64)  # 转换并统一样本索引的数据表示。
                rows = len(daily)
                style_matrix = np.zeros(  # 初始化风格矩阵。
                    (rows, len(STYLE_COLUMNS)), dtype=np.float64
                )
                style_valid = np.zeros_like(style_matrix, dtype=bool)  # 初始化风格有效。
                matched = daily["barra_matched"].to_numpy(bool)  # 转换并统一匹配数量的数据表示。
                for column_index, column in enumerate(barra.STYLE_COLUMNS):  # 遍历enumerateBarra风格列，逐项处理列索引列。
                    raw = daily[column].to_numpy(np.float64)  # 转换并统一原始的数据表示。
                    valid = matched & np.isfinite(raw)  # 构造有效，限定后续参与计算的样本。
                    style_matrix[:, column_index] = _standardize_with_missing(  # 仅使用当日截面统计量。
                        raw, valid, label=f"{trade_date} {column}"
                    )
                    style_valid[:, column_index] = valid  # 构造风格有效，限定后续参与计算的样本。
                reversal_index = len(STYLE_COLUMNS) - 1
                reversal_valid = daily[  # 构造reversal有效，限定后续参与计算的样本。
                    barra.REVERSAL_VALID_COLUMN
                ].to_numpy(bool)
                style_matrix[:, reversal_index] = _standardize_with_missing(  # 仅使用当日截面统计量。
                    daily[barra.REVERSAL_COLUMN].to_numpy(np.float64),
                    reversal_valid,
                    label=f"{trade_date} {barra.REVERSAL_COLUMN}",  # 配置标签。
                )
                style_valid[:, reversal_index] = reversal_valid  # 构造风格有效，限定后续参与计算的样本。

                daily_label_valid = label_valid[sample_indices]  # 构造daily标签有效，限定后续参与计算的样本。
                residual, record = residualize_daily_targets(  # 按日投影 Barra 暴露。
                    np.asarray(source_targets[sample_indices], dtype=np.float64),
                    style_matrix,
                    daily_label_valid,
                )
                styles_output[sample_indices] = style_matrix.astype(np.float32)  # 转换风格输出的数据类型或存储位置。
                valid_output[sample_indices] = style_valid  # 构造有效输出，限定后续参与计算的样本。
                targets_output[sample_indices] = residual.astype(np.float32)  # 转换标签输出的数据类型或存储位置。
                diagnostics.append(  # 把当前结果追加到汇总列表。
                    {
                        "year": year,  # 记录年份。
                        "trade_date": str(trade_date),  # 记录trade交易日。
                        "barra_matched_rows": int(matched.sum()),  # 记录Barra匹配行数。
                        "barra_match_rate": float(matched.mean()),  # 记录Barramatch比例。
                        "reversal_valid_rows": int(reversal_valid.sum()),  # 记录反转有效行数。
                        "reversal_coverage": float(reversal_valid.mean()),  # 记录反转覆盖率。
                        "complete_style_rows": int(style_valid.all(axis=1).sum()),  # 记录complete风格行数。
                        "complete_style_rate": float(style_valid.all(axis=1).mean()),  # 记录complete风格比例。
                        **record,
                    }
                )
            print(  # 输出当前任务的进度或诊断信息。
                f"{year}：写入 {len(joined):,} 行，"
                f"CNE5S 匹配率 {match_rate:.4%}",
                flush=True,  # 配置flush。
            )

        styles_output.flush()  # 将 memmap 修改刷入磁盘。
        valid_output.flush()  # 将 memmap 修改刷入磁盘。
        targets_output.flush()  # 将 memmap 修改刷入磁盘。
        del styles_output, valid_output, targets_output  # 释放不再使用的中间引用。
        _validate_arrays(stage, len(sample))
        residual_check = np.load(  # 读取残差检查。
            stage / RESIDUAL_TARGETS_FILE, mmap_mode="r"
        )
        if not np.array_equal(np.isfinite(residual_check), label_valid):  # 检查参与计算的数值是否全部有限。
            raise AssertionError("残差标签有限性与 label_valid 不一致")  # 关键数据契约被破坏时立即终止。

        daily_diagnostics = pd.DataFrame(diagnostics)
        daily_diagnostics.to_parquet(  # 把表格按约定格式写入 Parquet。
            stage / DAILY_DIAGNOSTICS_FILE, index=False, compression="zstd"
        )
        annual_summary = daily_diagnostics.groupby(  # 按既定口径聚合。
            "year", as_index=False
        ).agg(
            rows=("rows", "sum"),  # 配置行数。
            label_valid_rows=("label_valid_rows", "sum"),  # 配置标签有效行数。
            dates=("trade_date", "nunique"),  # 配置交易日。
            mean_barra_match_rate=("barra_match_rate", "mean"),  # 配置均值Barramatch比率。
            min_barra_match_rate=("barra_match_rate", "min"),  # 配置minBarramatch比率。
            mean_reversal_coverage=("reversal_coverage", "mean"),  # 配置均值reversalcoverage。
            min_reversal_coverage=("reversal_coverage", "min"),  # 配置minreversalcoverage。
            mean_complete_style_rate=("complete_style_rate", "mean"),  # 配置均值complete风格比率。
            mean_style_r_squared=("style_r_squared", "mean"),  # 配置均值风格rsquared。
            max_abs_style_residual_pearson=(
                "max_abs_style_residual_pearson", "max"
            ),
        )
        annual_summary.to_parquet(  # 把表格按约定格式写入 Parquet。
            stage / ANNUAL_SUMMARY_FILE, index=False, compression="zstd"
        )
        manifest = {  # 初始化清单容器。
            "signature": signature,  # 记录签名。
            "payload": payload,  # 记录载荷。
            "sample_count": len(sample),  # 记录样本数量。
            "label_valid_count": int(label_valid.sum()),  # 记录标签有效数量。
            "first_date": str(sample["trade_date"].min()),  # 记录首个交易日。
            "last_date": str(sample["trade_date"].max()),  # 记录最后一个交易日。
            "minimum_match_rate": minimum_match_rate,  # 记录minimummatch比例。
            "files": {  # 记录files。
                "style_exposures": STYLE_EXPOSURES_FILE,  # 记录风格暴露。
                "style_valid": STYLE_VALID_FILE,  # 记录风格有效。
                "residual_targets": RESIDUAL_TARGETS_FILE,  # 记录残差标签。
                "daily_diagnostics": DAILY_DIAGNOSTICS_FILE,  # 记录日度诊断。
                "annual_summary": ANNUAL_SUMMARY_FILE,  # 记录年度汇总。
            },
            "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),  # 记录createdatutc。
        }
        (stage / MANIFEST_FILE).write_text(  # 把文本结果写入临时或目标文件。
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",  # 使用 UTF-8 编解码文本。
        )
        _atomic_replace_directory(stage, target)  # 原子落盘，避免半成品。
        return {**manifest, "reused": False}  # 返回构建与 feature store 行号严格对齐的十风格矩阵和残差标签。
    except BaseException:  # 捕获预期异常并执行对应恢复逻辑。
        if stage.exists():  # 检查目标文件或目录是否满足存在性要求。
            shutil.rmtree(stage)
        raise


__all__ = [  # 初始化all容器。
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
def _validate_arrays(store_dir: Path, sample_count: int) -> None:  # 校验输入形状、业务键、有限性和签名约束。
    """校验输入形状、业务键、有限性和签名约束；异常时明确失败。"""
    styles = np.load(store_dir / STYLE_EXPOSURES_FILE, mmap_mode="r")  # 读取风格。
    valid = np.load(store_dir / STYLE_VALID_FILE, mmap_mode="r")  # 读取有效。
    residual = np.load(store_dir / RESIDUAL_TARGETS_FILE, mmap_mode="r")  # 读取残差。
    if styles.shape != (sample_count, len(STYLE_COLUMNS)) or styles.dtype != np.float32:  # 检查张量或数组形状是否符合接口约定。
        raise ValueError("style_exposures.npy 形状或 dtype 异常")  # 输入数据不满足业务或数值约束时立即终止。
    if valid.shape != styles.shape or valid.dtype != np.bool_:  # 检查张量或数组形状是否符合接口约定。
        raise ValueError("style_valid.npy 形状或 dtype 异常")  # 输入数据不满足业务或数值约束时立即终止。
    if residual.shape != (sample_count,) or residual.dtype != np.float32:  # 检查张量或数组形状是否符合接口约定。
        raise ValueError("targets_style_residual.npy 形状或 dtype 异常")  # 输入数据不满足业务或数值约束时立即终止。
    if not np.isfinite(styles).all():  # 检查参与计算的数值是否全部有限。
        raise ValueError("style_exposures.npy 含 NaN/Inf")  # 输入数据不满足业务或数值约束时立即终止。

def require_style_neutral_store(  # 验证风格中性训练数据与当前源数据签名一致。
    store_dir: str | Path = DEFAULT_STORE_DIR,  # 接收存储目录。
) -> dict[str, Any]:
    """验证风格中性训练数据与当前源数据签名一致。"""

    root = Path(store_dir).resolve()  # 构造根目录。
    manifest_path = root / MANIFEST_FILE  # 构造清单路径。
    if not manifest_path.exists():  # 检查目标文件或目录是否满足存在性要求。
        raise FileNotFoundError("缺少风格中性训练数据，请先运行 build_style_neutral_store")  # 目标文件不存在时立即终止。
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))  # 读取清单。
    expected = _store_payload(
        base.SAMPLE_INDEX_PATH,
        base.TARGETS_PATH,
        barra.DEFAULT_CNE5S_PATH,
        barra.DEFAULT_EOD_PATH,
        barra.DEFAULT_CALENDAR_PATH,
    )
    if manifest.get("signature") != _stable_hash(expected):  # 按当前模式、标识或状态选择处理分支。
        raise ValueError("风格中性训练数据与当前源数据签名不一致")  # 输入数据不满足业务或数值约束时立即终止。
    _validate_arrays(root, int(manifest["sample_count"]))
    return manifest  # 返回清单。

def _validate_industry_sidecar(  # 校验输入形状、业务键、有限性和签名约束。
    store_dir: Path,  # 接收存储目录。
    manifest: dict[str, Any],
) -> None:
    """校验输入形状、业务键、有限性和签名约束；异常时明确失败。"""
    codes = np.load(store_dir / INDUSTRY_CODES_FILE, mmap_mode="r")  # 读取代码。
    sample_count = int(manifest["sample_count"])  # 计算样本数量。
    mapping = manifest.get("industry_mapping")
    if codes.shape != (sample_count,) or codes.dtype != np.int16:  # 检查张量或数组形状是否符合接口约定。
        raise ValueError("industry_codes.npy 形状或 dtype 异常")  # 输入数据不满足业务或数值约束时立即终止。
    if not isinstance(mapping, dict) or mapping.get(barra.UNKNOWN_INDUSTRY) != 0:  # 检查对象类型是否符合接口约定。
        raise ValueError("行业 sidecar 缺少稳定行业映射")  # 输入数据不满足业务或数值约束时立即终止。
    if (codes < 0).any() or int(codes.max()) >= len(mapping):  # 检查当前样本数量是否满足计算要求。
        raise ValueError("industry_codes.npy 含非法行业代码")  # 输入数据不满足业务或数值约束时立即终止。
    known_count = len(mapping) - 1  # 构造已知行业数量，限定后续参与计算的样本。
    if known_count != EXPECTED_CNE5S_INDUSTRIES:  # 按当前模式、标识或状态选择处理分支。
        raise ValueError(  # 输入数据不满足业务或数值约束时立即终止。
            f"CNE5S 行业数为 {known_count}，预期 {EXPECTED_CNE5S_INDUSTRIES}"
        )

def require_industry_sidecar(  # 验证联合 R² 所需的逐样本行业代码 sidecar。
    store_dir: str | Path = DEFAULT_INDUSTRY_STORE_DIR,  # 接收存储目录。
    *,
    minimum_match_rate: float = MINIMUM_MATCH_RATE,
) -> dict[str, Any]:
    """验证联合 R² 所需的逐样本行业代码 sidecar。"""

    root = Path(store_dir).resolve()  # 构造根目录。
    manifest_path = root / INDUSTRY_MANIFEST_FILE  # 构造清单路径。
    if not manifest_path.exists():  # 检查目标文件或目录是否满足存在性要求。
        raise FileNotFoundError(  # 目标文件不存在时立即终止。
            "缺少行业 sidecar，请先运行 build_industry_sidecar"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))  # 读取清单。
    style_manifest = require_style_neutral_store()
    expected = _industry_sidecar_payload(
        base.SAMPLE_INDEX_PATH,
        barra.DEFAULT_CNE5S_PATH,
        style_manifest["signature"],
        minimum_match_rate,
    )
    if manifest.get("signature") != _stable_hash(expected):  # 按当前模式、标识或状态选择处理分支。
        raise ValueError("行业 sidecar 与当前样本、CNE5S 或风格 store 签名不一致")  # 输入数据不满足业务或数值约束时立即终止。
    _validate_industry_sidecar(root, manifest)
    return manifest  # 返回清单。

def _validate_full_barra_target_sidecar(  # 校验输入形状、业务键、有限性和签名约束。
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
    missing = [name for name in required_files if not (store_dir / name).exists()]  # 初始化missing容器。
    if missing:  # 根据当前数据状态选择处理分支。
        raise FileNotFoundError(f"完整 Barra 残差标签文件缺失：{missing}")  # 目标文件不存在时立即终止。
    targets = np.load(store_dir / FULL_BARRA_TARGETS_FILE, mmap_mode="r")  # 读取标签。
    if (  # 开始组合多项有效性或配置条件。
        targets.shape != (int(manifest["sample_count"]),)
        or targets.dtype != np.float32
    ):
        raise ValueError(  # 输入数据不满足业务或数值约束时立即终止。
            "targets_full_barra_residual.npy 形状或 dtype 异常"
        )
    if int(np.isfinite(targets).sum()) != int(manifest["label_valid_count"]):  # 检查参与计算的数值是否全部有限。
        raise ValueError("完整 Barra 残差标签有限值数量与 manifest 不一致")  # 输入数据不满足业务或数值约束时立即终止。
    daily = pd.read_parquet(  # 读取daily。
        store_dir / FULL_BARRA_DAILY_DIAGNOSTICS_FILE,
        columns=[  # 初始化列容器。
            "rows",
            "max_abs_style_residual_pearson",
            "max_abs_industry_residual_mean",
        ],
    )
    if int(daily["rows"].sum()) != int(manifest["sample_count"]):  # 按当前模式、标识或状态选择处理分支。
        raise ValueError("完整 Barra 逐日诊断行数汇总与 manifest 不一致")  # 输入数据不满足业务或数值约束时立即终止。
    if daily[  # 根据当前数据状态选择处理分支。
        [
            "max_abs_style_residual_pearson",
            "max_abs_industry_residual_mean",
        ]
    ].to_numpy().max() > ORTHOGONALITY_TOLERANCE:
        raise ValueError("完整 Barra 残差标签正交诊断未通过")  # 输入数据不满足业务或数值约束时立即终止。

def require_full_barra_target_sidecar(  # 验证十风格加行业固定效应残差标签 sidecar。
    store_dir: str | Path = DEFAULT_FULL_BARRA_TARGET_STORE_DIR,
) -> dict[str, Any]:
    """验证十风格加行业固定效应残差标签 sidecar。"""

    root = Path(store_dir).resolve()  # 构造根目录。
    manifest_path = root / FULL_BARRA_TARGET_MANIFEST_FILE  # 构造清单路径。
    if not manifest_path.exists():  # 检查目标文件或目录是否满足存在性要求。
        raise FileNotFoundError(  # 目标文件不存在时立即终止。
            "缺少完整 Barra 残差标签，请先运行 "
            "build_full_barra_target_sidecar"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))  # 读取清单。
    style_manifest = require_style_neutral_store()
    industry_manifest = require_industry_sidecar()
    expected = _full_barra_target_payload(
        base.TARGETS_PATH,
        style_manifest["signature"],
        industry_manifest["signature"],
    )
    if manifest.get("signature") != _stable_hash(expected):  # 按当前模式、标识或状态选择处理分支。
        raise ValueError(  # 输入数据不满足业务或数值约束时立即终止。
            "完整 Barra 残差标签与当前标签、风格或行业 sidecar 签名不一致"
        )
    _validate_full_barra_target_sidecar(root, manifest)
    return manifest  # 返回清单。

def _validate_dual_tower_target_sidecar(  # 校验输入形状、业务键、有限性和签名约束。
    store_dir: Path, manifest: dict[str, Any]
) -> None:
    """校验输入形状、业务键、有限性和签名约束；异常时明确失败。"""
    required = (
        DUAL_TOWER_FITTED_TARGETS_FILE,
        DUAL_TOWER_DAILY_DIAGNOSTICS_FILE,
        DUAL_TOWER_ANNUAL_SUMMARY_FILE,
        DUAL_TOWER_TARGET_MANIFEST_FILE,
    )
    missing = [name for name in required if not (store_dir / name).exists()]  # 初始化missing容器。
    if missing:  # 根据当前数据状态选择处理分支。
        raise FileNotFoundError(f"No.6 fitted-target sidecar 文件缺失：{missing}")  # 目标文件不存在时立即终止。
    targets = np.load(  # 读取标签。
        store_dir / DUAL_TOWER_FITTED_TARGETS_FILE, mmap_mode="r"
    )
    if (  # 开始组合多项有效性或配置条件。
        targets.shape != (int(manifest["sample_count"]),)
        or targets.dtype != np.float32
    ):
        raise ValueError("No.6 fitted target 形状或 dtype 异常")  # 输入数据不满足业务或数值约束时立即终止。
    if int(np.isfinite(targets).sum()) != int(manifest["label_valid_count"]):  # 检查参与计算的数值是否全部有限。
        raise ValueError("No.6 fitted target 有效值数量与 manifest 不一致")  # 输入数据不满足业务或数值约束时立即终止。
    daily = pd.read_parquet(  # 读取daily。
        store_dir / DUAL_TOWER_DAILY_DIAGNOSTICS_FILE,
        columns=[  # 初始化列容器。
            "rows",
            "reconstruction_max_abs_error",
            "residual_sidecar_max_abs_error",
            "fitted_joint_barra_r_squared",
        ],
    )
    if int(daily["rows"].sum()) != int(manifest["sample_count"]):  # 按当前模式、标识或状态选择处理分支。
        raise ValueError("No.6 fitted target 逐日行数与 manifest 不一致")  # 输入数据不满足业务或数值约束时立即终止。
    if float(daily["reconstruction_max_abs_error"].max()) > 1e-12:  # 检查当前数值是否越过业务或数值阈值。
        raise ValueError("No.6 标签分解无法精确重构源标签")  # 输入数据不满足业务或数值约束时立即终止。
    if float(daily["residual_sidecar_max_abs_error"].max()) > 1e-6:  # 检查当前数值是否越过业务或数值阈值。
        raise ValueError("No.6 重算残差与现有完整残差 sidecar 不一致")  # 输入数据不满足业务或数值约束时立即终止。
    if float((1.0 - daily["fitted_joint_barra_r_squared"]).abs().max()) > 1e-8:  # 检查当前数值是否越过业务或数值阈值。
        raise ValueError("No.6 fitted target 不在完整 Barra 风险列空间内")  # 输入数据不满足业务或数值约束时立即终止。

def require_dual_tower_target_sidecar(  # 验证 No.6 风格塔拟合收益标签 sidecar。
    store_dir: str | Path = DEFAULT_DUAL_TOWER_TARGET_STORE_DIR,
) -> dict[str, Any]:
    """验证 No.6 风格塔拟合收益标签 sidecar。"""

    root = Path(store_dir).resolve()  # 构造根目录。
    manifest_path = root / DUAL_TOWER_TARGET_MANIFEST_FILE  # 构造清单路径。
    if not manifest_path.exists():  # 检查目标文件或目录是否满足存在性要求。
        raise FileNotFoundError(  # 目标文件不存在时立即终止。
            "缺少 No.6 fitted-target sidecar，请先运行 "
            "build_dual_tower_target_sidecar"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))  # 读取清单。
    style_manifest = require_style_neutral_store()
    industry_manifest = require_industry_sidecar()
    full_manifest = require_full_barra_target_sidecar()
    expected = _dual_tower_target_payload(
        base.TARGETS_PATH,
        full_manifest["signature"],
        style_manifest["signature"],
        industry_manifest["signature"],
    )
    if manifest.get("signature") != _stable_hash(expected):  # 按当前模式、标识或状态选择处理分支。
        raise ValueError("No.6 fitted target 与当前标签或 Barra sidecar 签名不一致")  # 输入数据不满足业务或数值约束时立即终止。
    _validate_dual_tower_target_sidecar(root, manifest)
    return manifest  # 返回清单。
