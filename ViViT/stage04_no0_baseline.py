"""No.0：无 Barra 约束的基线损失。"""
from __future__ import annotations

from collections.abc import Callable

import torch


# region TR-N00-LOSS
# 定义 `baseline_loss` 函数，封装这一阶段可独立调用的处理步骤。
def baseline_loss(
    raw_score: torch.Tensor,
    targets: torch.Tensor,
    *,
    rank_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """直接用原始分数和原始标签计算排序损失。
    
    输入：
        raw_score: torch.Tensor，格式 [N]。
        targets: torch.Tensor，格式 [N]。
        rank_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]。
    输出：
        torch.Tensor。
        """

    # 将 `rank_loss(raw_score, targets)` 作为当前函数结果返回给调用方。
    return rank_loss(raw_score, targets)
# endregion
