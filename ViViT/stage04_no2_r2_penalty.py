"""No.2：直接惩罚原始分数的联合 Barra R²。"""
from __future__ import annotations

from collections.abc import Callable

import torch



def r_squared_regularized_loss(  # 返回排序损失、联合 R² 和两者的加权和。
    raw_score: torch.Tensor,  # [B]，原始预测分数。
    targets: torch.Tensor,  # [B]，训练标签。
    style_exposures: torch.Tensor,  # [B,10]，十个连续 Barra 暴露。
    industry_codes: torch.Tensor,  # [B]，行业整数代码。
    *,
    rank_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    barra_r_squared: Callable[  # 接收Barrarsquared。
        [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
    ],
    penalty_weight: float,  # 接收penalty权重。
) -> dict[str, torch.Tensor]:
    """返回排序损失、联合 R² 和两者的加权和。"""

    predictive_loss = rank_loss(raw_score, targets)  # 按实验口径计算可微损失。
    r_squared = barra_r_squared(  # 计算 Barra 可解释比例。
        raw_score, style_exposures, industry_codes
    )
    return {  # 返回排序损失、联合 R² 和两者的加权和。
        "loss": predictive_loss.float() + penalty_weight * r_squared,  # 记录损失。
        "predictive_loss": predictive_loss,  # 记录predictive损失。
        "barra_r_squared": r_squared,  # 记录Barrarsquared。
    }
