"""No.5：训练图内严格 Barra 中性输出。"""
from __future__ import annotations

from collections.abc import Callable

import torch


# region TR-N05-OUTPUT
# 定义 `strict_neutral_output` 函数，封装这一阶段可独立调用的处理步骤。
def strict_neutral_output(
    raw_score: torch.Tensor,
    style_exposures: torch.Tensor,
    industry_codes: torch.Tensor,
    *,
    projector: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, dict[str, torch.Tensor]],
    ],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """对每日 raw score 投影十风格和行业固定效应。
    
    输入：
        raw_score: torch.Tensor，格式 [N]。
        style_exposures: torch.Tensor，格式 [N,10]。
        industry_codes: torch.Tensor，格式 [N]。
        projector: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, dict[str, torch.Tensor]]]。
    输出：
        tuple[torch.Tensor, dict[str, torch.Tensor]]。
        """

    # 将 `projector(raw_score, style_exposures, industry_codes)` 作为当前函数结果返回给调用方。
    return projector(raw_score, style_exposures, industry_codes)
# endregion
