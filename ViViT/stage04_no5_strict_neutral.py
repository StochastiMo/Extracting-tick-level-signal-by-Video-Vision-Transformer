"""No.5：训练图内严格 Barra 中性输出。"""
from __future__ import annotations

from collections.abc import Callable

import torch


# region TR-N05-OUTPUT
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
    """对每日 raw score 投影十风格和行业固定效应。"""

    return projector(raw_score, style_exposures, industry_codes)
# endregion
