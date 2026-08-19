"""No.5：训练图内严格 Barra 中性输出。"""
from __future__ import annotations

from collections.abc import Callable

import torch


# region TR-N05-OUTPUT
def strict_neutral_output(  # 对每日 raw score 投影十风格和行业固定效应。
    raw_score: torch.Tensor,  # [B]，原始预测分数。
    style_exposures: torch.Tensor,  # [B,10]，十个连续 Barra 暴露。
    industry_codes: torch.Tensor,  # [B]，行业整数代码。
    *,
    projector: Callable[  # 接收projector。
        [torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, dict[str, torch.Tensor]],
    ],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:  # 声明当前函数的返回类型。
    """对每日 raw score 投影十风格和行业固定效应。"""

    return projector(raw_score, style_exposures, industry_codes)  # 返回完成上述处理后的结果。
# endregion
