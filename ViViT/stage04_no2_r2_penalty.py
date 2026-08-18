"""No.2：直接惩罚原始分数的联合 Barra R²。"""
from __future__ import annotations

from collections.abc import Callable

import torch


# region TR-N02-LOSS
# 定义 `r_squared_regularized_loss` 函数，封装这一阶段可独立调用的处理步骤。
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
    """返回排序损失、联合 R² 和两者的加权和。
    
    输入：
        raw_score: torch.Tensor，格式 [N]。
        targets: torch.Tensor，格式 [N]。
        style_exposures: torch.Tensor，格式 [N,10]。
        industry_codes: torch.Tensor，格式 [N]。
        rank_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]。
        barra_r_squared: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]。
        penalty_weight: float。
    输出：
        dict[str, torch.Tensor]。
        """

    # 调用 `rank_loss` 按当前实验口径计算可反向传播的训练损失，并将结果保存到 `predictive_loss`。
    predictive_loss = rank_loss(raw_score, targets)
    # 调用 `barra_r_squared` 计算分数的 Barra 分解结果或可解释比例，并将结果保存到 `r_squared`。
    r_squared = barra_r_squared(
        raw_score, style_exposures, industry_codes
    )
    # 将 `{'loss': predictive_loss.float() + penalty_weight * r_squared, 'predictive_loss': predictive_lo…` 作为当前函数结果返回给调用方。
    return {
        "loss": predictive_loss.float() + penalty_weight * r_squared,
        "predictive_loss": predictive_loss,
        "barra_r_squared": r_squared,
    }
# endregion
