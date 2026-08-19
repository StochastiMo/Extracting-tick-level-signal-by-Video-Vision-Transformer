"""No.0：无 Barra 约束的基线损失。"""
from __future__ import annotations

from collections.abc import Callable

import torch


# region TR-N00-LOSS
def baseline_loss(  # 直接用原始分数和原始标签计算排序损失。
    raw_score: torch.Tensor,  # [B]，原始预测分数。
    targets: torch.Tensor,  # [B]，训练标签。
    *,
    rank_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:  # 声明当前函数的返回类型。
    """直接用原始分数和原始标签计算排序损失。"""

    return rank_loss(raw_score, targets)  # 返回直接用原始分数和原始标签计算排序损失。
# endregion
