"""No.2：直接惩罚原始分数的联合 Barra R²。"""
from __future__ import annotations

from collections.abc import Callable

import torch


# region TR-N02-LOSS
def r_squared_regularized_loss(
    raw_score: torch.Tensor,
    targets: torch.Tensor,
    style_exposures: torch.Tensor,
    industry_codes: torch.Tensor,
    *,
    rank_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    barra_r_squared: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
    ],
    penalty_weight: float,
) -> dict[str, torch.Tensor]:
    """返回排序损失、联合 R² 和两者的加权和。"""

    predictive_loss = rank_loss(raw_score, targets)  # 按实验口径计算可微损失
    r_squared = barra_r_squared(  # 计算 Barra 可解释比例
        raw_score, style_exposures, industry_codes
    )
    return {
        "loss": predictive_loss.float() + penalty_weight * r_squared,
        "predictive_loss": predictive_loss,
        "barra_r_squared": r_squared,
    }
# endregion
