"""Barra 标签清洗、截面分解和风险投影的稳定入口。

底层数值实现暂由经过回归测试的旧数据模块提供；本模块把报告关心的数据输入、
缺失处理和输出契约集中在一个短文件中。
"""
from __future__ import annotations

import numpy as np
import torch

import stage02_barra_data as _legacy


STYLE_COLUMNS = _legacy.STYLE_COLUMNS
EXPECTED_INDUSTRY_COUNT = _legacy.EXPECTED_CNE5S_INDUSTRIES


# region TR-DATA-BARRA-STANDARDIZE
def standardize_style_exposure(
    values: np.ndarray,
    valid: np.ndarray,
    *,
    label: str,
) -> np.ndarray:
    """每日标准化单个暴露；仅在有效样本上估计均值和标准差。 ``valid=False`` 的位置输出零，但有效标记仍由调用方独立保存；整个过程 不做跨日填充，因而不会把未来或历史暴露带入当前截面。"""

    return _legacy._standardize_with_missing(values, valid, label=label)
# endregion


# region TR-DATA-BARRA-TARGET
def residualize_style_target(
    targets: np.ndarray,
    style_exposures: np.ndarray,
    label_valid: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """用十个连续风格解释标签，返回每日 Z-score 残差。"""

    return _legacy.residualize_daily_targets(
        targets, style_exposures, label_valid
    )


def decompose_full_barra_target(
    targets: np.ndarray,
    style_exposures: np.ndarray,
    industry_codes: np.ndarray,
    label_valid: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, float | int],
]:
    """按十风格和行业固定效应把标签分成残差项与拟合项。"""

    return _legacy.decompose_daily_targets_with_industry(
        targets, style_exposures, industry_codes, label_valid
    )
# endregion


# region TR-COMMON-PROJECTION
def project_style_score(
    predictions: torch.Tensor,
    style_exposures: torch.Tensor,
    *,
    standardize: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """从分数中投影十个连续风格并保留梯度。"""

    return _legacy.project_out_styles_torch(
        predictions, style_exposures, standardize=standardize
    )


def project_full_barra_score(
    predictions: torch.Tensor,
    style_exposures: torch.Tensor,
    industry_codes: torch.Tensor,
    *,
    standardize: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """从分数中投影十风格及行业固定效应并保留梯度。"""

    return _legacy.project_out_full_barra_torch(
        predictions,
        style_exposures,
        industry_codes,
        standardize=standardize,
    )
# endregion


# region TR-COMMON-R2
def joint_barra_r_squared(
    predictions: torch.Tensor,
    style_exposures: torch.Tensor,
    industry_codes: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """计算分数由十风格和行业固定效应联合解释的可微 R²。"""

    return _legacy.joint_barra_r_squared_torch(
        predictions, style_exposures, industry_codes
    )
# endregion
