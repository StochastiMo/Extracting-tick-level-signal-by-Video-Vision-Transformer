"""No.1：标签残差化和/或损失图内分数投影。"""
from __future__ import annotations

from collections.abc import Callable

import torch


# region TR-N01-LOSS
def neutral_training_loss(  # 返回训练损失及损失真正看到的 effective score。
    raw_score: torch.Tensor,  # [B]，原始预测分数。
    targets: torch.Tensor,  # [B]，训练标签。
    style_exposures: torch.Tensor,  # [B,10]，十个连续 Barra 暴露。
    *,
    project_output: bool,  # 接收投影输出。
    projector: Callable[  # 接收projector。
        [torch.Tensor, torch.Tensor], torch.Tensor
    ],
    rank_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:  # 声明当前函数的返回类型。
    """返回训练损失及损失真正看到的 effective score。"""

    effective_score = (  # 计算有效分数。
        projector(raw_score, style_exposures)  # 执行projector对应的处理步骤。
        if project_output  # 根据当前数据状态选择处理分支。
        else raw_score
    )
    return rank_loss(effective_score, targets), effective_score  # 返回训练损失及损失真正看到的 effective score。
# endregion
