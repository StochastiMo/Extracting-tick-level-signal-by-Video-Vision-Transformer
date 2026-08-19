"""No.1：标签残差化和/或损失图内分数投影。"""
from __future__ import annotations

from collections.abc import Callable

import torch


# region TR-N01-LOSS
def neutral_training_loss(
    raw_score: torch.Tensor,
    targets: torch.Tensor,
    style_exposures: torch.Tensor,
    *,
    project_output: bool,
    projector: Callable[
        [torch.Tensor, torch.Tensor], torch.Tensor
    ],
    rank_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回训练损失及损失真正看到的 effective score。"""

    effective_score = (
        projector(raw_score, style_exposures)
        if project_output
        else raw_score
    )
    return rank_loss(effective_score, targets), effective_score
# endregion
