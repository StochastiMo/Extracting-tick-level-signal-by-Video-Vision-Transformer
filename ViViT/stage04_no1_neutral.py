"""No.1：标签残差化和/或损失图内分数投影。"""
from __future__ import annotations

from collections.abc import Callable

import torch


# region TR-N01-LOSS
# 定义 `neutral_training_loss` 函数，封装这一阶段可独立调用的处理步骤。
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
    """返回训练损失及损失真正看到的 effective score。
    
    输入：
        raw_score: torch.Tensor，格式 [N]。
        targets: torch.Tensor，格式 [N]。
        style_exposures: torch.Tensor，格式 [N,10]。
        project_output: bool。
        projector: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]。
        rank_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]。
    输出：
        tuple[torch.Tensor, torch.Tensor]。
        """

    # 按 `project_output` 在两个候选值间选择，并保存到 `effective_score`。
    effective_score = (
        projector(raw_score, style_exposures)
        if project_output
        else raw_score
    )
    # 将 `(rank_loss(effective_score, targets), effective_score)` 作为当前函数结果返回给调用方。
    return rank_loss(effective_score, targets), effective_score
# endregion
