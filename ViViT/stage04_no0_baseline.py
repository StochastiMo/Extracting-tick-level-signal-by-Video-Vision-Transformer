"""No.0：无 Barra 约束的基线损失。"""
from __future__ import annotations

from collections.abc import Callable

import torch


# region TR-N00-LOSS
def baseline_loss(
    raw_score: torch.Tensor,
    targets: torch.Tensor,
    *,
    rank_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """直接用原始分数和原始标签计算排序损失。"""

    return rank_loss(raw_score, targets)
# endregion
