"""Barra 标签清洗、截面分解和风险投影的稳定入口。

底层数值实现暂由经过回归测试的旧数据模块提供；本模块把报告关心的数据输入、
缺失处理和输出契约集中在一个短文件中。
"""
from __future__ import annotations

import numpy as np
import torch

import stage02_barra_data as _legacy


STYLE_COLUMNS = _legacy.STYLE_COLUMNS  # 集中声明风格列。
EXPECTED_INDUSTRY_COUNT = _legacy.EXPECTED_CNE5S_INDUSTRIES  # 设置预期行业数量。


# region TR-DATA-BARRA-STANDARDIZE
def standardize_style_exposure(  # 每日标准化单个暴露。
    values: np.ndarray,  # 输入待处理的截面数值。
    valid: np.ndarray,  # 输入与数值等长的有效性掩码。
    *,
    label: str,  # 接收标签。
) -> np.ndarray:
    """每日标准化单个暴露；仅在有效样本上估计均值和标准差。 ``valid=False`` 的位置输出零，但有效标记仍由调用方独立保存；整个过程 不做跨日填充，因而不会把未来或历史暴露带入当前截面。"""

    return _legacy._standardize_with_missing(values, valid, label=label)  # 返回每日标准化单个暴露。
# endregion


# region TR-DATA-BARRA-TARGET
def residualize_style_target(  # 用十个连续风格解释标签，返回每日 Z-score 残差。
    targets: np.ndarray,  # 接收标签。
    style_exposures: np.ndarray,  # 接收风格暴露。
    label_valid: np.ndarray,  # 输入标签有效性掩码。
) -> tuple[np.ndarray, dict[str, float | int]]:
    """用十个连续风格解释标签，返回每日 Z-score 残差。"""

    return _legacy.residualize_daily_targets(  # 返回完成上述回归或变换后的结果。
        targets, style_exposures, label_valid  # 传入标签、风格暴露、标签有效。
    )


def decompose_full_barra_target(  # 按十风格和行业固定效应把标签分成残差项与拟合项。
    targets: np.ndarray,  # 接收标签。
    style_exposures: np.ndarray,  # 接收风格暴露。
    industry_codes: np.ndarray,  # 接收行业代码。
    label_valid: np.ndarray,  # 输入标签有效性掩码。
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, float | int],
]:
    """按十风格和行业固定效应把标签分成残差项与拟合项。"""

    return _legacy.decompose_daily_targets_with_industry(  # 返回按当前规则组织后的结果。
        targets, style_exposures, industry_codes, label_valid  # 传入标签、风格暴露、行业代码、标签有效。
    )
# endregion


# region TR-COMMON-PROJECTION
def project_style_score(  # 从分数中投影十个连续风格并保留梯度。
    predictions: torch.Tensor,  # [N]，单日截面分数。
    style_exposures: torch.Tensor,  # [N,10]，单日十风格暴露。
    *,
    standardize: bool = True,  # 控制标准化开关。
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """从分数中投影十个连续风格并保留梯度。"""

    return _legacy.project_out_styles_torch(  # 返回完成投影或提取后的结果。
        predictions, style_exposures, standardize=standardize
    )


def project_full_barra_score(  # 从分数中投影十风格及行业固定效应并保留梯度。
    predictions: torch.Tensor,  # [N]，单日截面分数。
    style_exposures: torch.Tensor,  # [N,10]，单日十风格暴露。
    industry_codes: torch.Tensor,  # [N]，单日行业代码。
    *,
    standardize: bool = True,  # 控制标准化开关。
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """从分数中投影十风格及行业固定效应并保留梯度。"""

    return _legacy.project_out_full_barra_torch(  # 返回完成投影或提取后的结果。
        predictions,
        style_exposures,
        industry_codes,
        standardize=standardize,  # 控制投影结果是否截面标准化。
    )
# endregion


# region TR-COMMON-R2
def joint_barra_r_squared(  # 计算分数由十风格和行业固定效应联合解释的可微 R²。
    predictions: torch.Tensor,  # [N]，单日截面分数。
    style_exposures: torch.Tensor,  # [N,10]，单日十风格暴露。
    industry_codes: torch.Tensor,  # [N]，单日行业代码。
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """计算分数由十风格和行业固定效应联合解释的可微 R²。"""

    return _legacy.joint_barra_r_squared_torch(  # 返回上述计算结果。
        predictions, style_exposures, industry_codes  # 传入预测、风格暴露、行业代码。
    )
# endregion
