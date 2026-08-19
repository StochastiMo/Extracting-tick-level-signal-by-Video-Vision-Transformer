"""No.3：对抗式 Barra 不变表征。"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from stage03_model import OrderFlowViViT


class _GradientReversal(torch.autograd.Function):
    """前向恒等、反向乘以负系数的梯度反转算子。
    """

    # 使用 `staticmethod` 包装 `forward`，并定义其可复用的处理逻辑。
    @staticmethod
    def forward(
        ctx: Any, values: torch.Tensor, gamma: torch.Tensor
    ) -> torch.Tensor:
        ctx.save_for_backward(gamma)
        return values.clone()

    # 使用 `staticmethod` 包装 `backward`，并定义其可复用的处理逻辑。
    @staticmethod
    def backward(
        ctx: Any, gradient: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        (gamma,) = ctx.saved_tensors
        return -gamma * gradient, None


# region TR-N03-GRL
def gradient_reverse(
    values: torch.Tensor, gamma: float | torch.Tensor
) -> torch.Tensor:
    """保持前向值不变，让编码器收到负 gamma 倍梯度。"""

    coefficient = torch.as_tensor(
        gamma, dtype=values.dtype, device=values.device
    )
    return _GradientReversal.apply(values, coefficient)
# endregion


def cosine_adversarial_gamma(
    epoch: int,
    *,
    gamma_max: float,
    warmup_epochs: int,
    ramp_epochs: int,
) -> float:
    """按绝对 epoch 计算预热、余弦爬坡和稳定阶段的 GRL 系数。"""
    if epoch <= warmup_epochs:
        return 0.0
    progress = min(1.0, (epoch - warmup_epochs) / ramp_epochs)
    return float(
        0.5 * gamma_max * (1.0 - math.cos(math.pi * progress))
    )


def _adversarial_head(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    dropout: float,
) -> nn.Sequential:
    """构造一个两层对抗预测头。"""

    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )


# region TR-N03-MODEL
class AdversarialBarraViViT(nn.Module):
    """收益头预测 Alpha，对抗头尝试从 CLS 恢复风格和行业。
    """

    def __init__(
        self,
        backbone: OrderFlowViViT,
        *,
        embedding_dim: int,
        style_count: int,
        industry_count: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.style_head = _adversarial_head(
            embedding_dim, hidden_dim, style_count, dropout
        )
        self.industry_head = _adversarial_head(
            embedding_dim, hidden_dim, industry_count, dropout
        )
        self._initialize_adversarial_heads()  # 初始化本分支参数

    def _initialize_adversarial_heads(self) -> None:
        """初始化新增分支，不改变 ViViT 主干参数。"""

        for head in (self.style_head, self.industry_head):
            for module in head.modules():
                if isinstance(module, nn.Linear):
                    nn.init.trunc_normal_(module.weight, std=0.02)  # 截断正态初始化
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(
        self,
        videos: torch.Tensor,
        frame_mask: torch.Tensor,
        gamma: torch.Tensor,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """返回 raw score、风格预测、行业 logits 和 CLS 表征。"""

        representation = self.backbone.encode(videos, frame_mask)  # 编码为共享视频表征
        raw_score = self.backbone.regression_head(
            representation
        ).squeeze(-1)
        reversed_representation = gradient_reverse(
            representation, gamma
        )
        return (
            raw_score,
            self.style_head(reversed_representation),
            self.industry_head(reversed_representation),
            representation,
        )
# endregion
