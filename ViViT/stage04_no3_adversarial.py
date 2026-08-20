"""No.3：对抗式 Barra 不变表征。"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from stage03_model import OrderFlowViViT


class _GradientReversal(torch.autograd.Function):  # 前向恒等、反向乘以负系数的梯度反转算子。
    """前向恒等、反向乘以负系数的梯度反转算子。
    """

  # 使用 `staticmethod` 包装 `forward`，并定义其可复用的处理逻辑。
    @staticmethod  # 该方法不依赖实例状态。
    def forward(  # 执行当前模块的前向计算。
        ctx: Any, values: torch.Tensor, gamma: torch.Tensor  # values [B,D]；gamma []。
    ) -> torch.Tensor:
        ctx.save_for_backward(gamma)
        return values.clone()  # 返回forward。

  # 使用 `staticmethod` 包装 `backward`，并定义其可复用的处理逻辑。
    @staticmethod  # 该方法不依赖实例状态。
    def backward(  # 按梯度反转规则计算反向梯度。
        ctx: Any, gradient: torch.Tensor  # [B,D]，上游梯度。
    ) -> tuple[torch.Tensor, None]:
        (gamma,) = ctx.saved_tensors
        return -gamma * gradient, None  # 返回backward。



def gradient_reverse(  # 保持前向值不变，让编码器收到负 gamma 倍梯度。
    values: torch.Tensor, gamma: float | torch.Tensor  # values [B,D]；gamma []。
) -> torch.Tensor:
    """保持前向值不变，让编码器收到负 gamma 倍梯度。"""

    coefficient = torch.as_tensor(  # 转换并统一系数的数据表示。
        gamma, dtype=values.dtype, device=values.device
    )
    return _GradientReversal.apply(values, coefficient)  # 返回保持前向值不变，让编码器收到负 gamma 倍梯度。



def cosine_adversarial_gamma(  # 按绝对 epoch 计算预热、余弦爬坡和稳定阶段的 GRL 系数。
    epoch: int,  # 接收轮次。
    *,
    gamma_max: float,  # 接收对抗系数max。
    warmup_epochs: int,  # 接收warmupepochs。
    ramp_epochs: int,  # 接收rampepochs。
) -> float:
    """按绝对 epoch 计算预热、余弦爬坡和稳定阶段的 GRL 系数。"""
    if epoch <= warmup_epochs:  # 检查当前数值是否越过业务或数值阈值。
        return 0.0  # 返回按当前规则组织后的结果。
    progress = min(1.0, (epoch - warmup_epochs) / ramp_epochs)
    return float(  # 返回按当前规则组织后的结果。
        0.5 * gamma_max * (1.0 - math.cos(math.pi * progress))
    )


def _adversarial_head(  # 构造一个两层对抗预测头。
    input_dim: int,  # 接收输入dim。
    hidden_dim: int,  # 接收隐藏层dim。
    output_dim: int,  # 接收输出dim。
    dropout: float,  # 接收dropout。
) -> nn.Sequential:
    """构造一个两层对抗预测头。"""

    return nn.Sequential(  # 返回构造一个两层对抗预测头。
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )



class AdversarialBarraViViT(nn.Module):  # 收益头预测 Alpha，对抗头尝试从 CLS 恢复风格和行业。
    """收益头预测 Alpha，对抗头尝试从 CLS 恢复风格和行业。
    """

    def __init__(  # 初始化当前对象的配置、状态与子组件。
        self,
        backbone: OrderFlowViViT,  # 接收backbone。
        *,
        embedding_dim: int,  # 接收嵌入dim。
        style_count: int,  # 接收风格数量。
        industry_count: int,  # 接收行业数量。
        hidden_dim: int,  # 接收隐藏层dim。
        dropout: float,  # 接收dropout。
    ) -> None:
        super().__init__()
        self.backbone = backbone  # 保存backbone，供该对象后续步骤复用。
        self.style_head = _adversarial_head(  # 保存风格head，供该对象后续步骤复用。
            embedding_dim, hidden_dim, style_count, dropout  # 传入嵌入dim、隐藏层dim、风格数量、dropout。
        )
        self.industry_head = _adversarial_head(  # 保存行业head，供该对象后续步骤复用。
            embedding_dim, hidden_dim, industry_count, dropout  # 传入嵌入dim、隐藏层dim、行业数量、dropout。
        )
        self._initialize_adversarial_heads()  # 初始化本分支参数。

    def _initialize_adversarial_heads(self) -> None:  # 初始化新增分支，不改变 ViViT 主干参数。
        """初始化新增分支，不改变 ViViT 主干参数。"""

        for head in (self.style_head, self.industry_head):  # 遍历self风格headself行业head，逐项处理head。
            for module in head.modules():  # 遍历headmodules，逐项处理module。
                if isinstance(module, nn.Linear):  # 检查对象类型是否符合接口约定。
                    nn.init.trunc_normal_(module.weight, std=0.02)  # 截断正态初始化。
                    if module.bias is not None:  # 检查必需数据、配置或校验结果是否缺失。
                        nn.init.zeros_(module.bias)

    def forward(  # 返回 raw score、风格预测、行业 logits 和 CLS 表征。
        self,
        videos: torch.Tensor,  # [B,17,20,8,8]，订单流视频。
        frame_mask: torch.Tensor,  # [B,20]，逐帧有效标记。
        gamma: torch.Tensor,  # []，梯度反转系数。
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """返回 raw score、风格预测、行业 logits 和 CLS 表征。"""

        representation = self.backbone.encode(videos, frame_mask)  # 编码为共享视频表征。
        raw_score = self.backbone.regression_head(  # 计算原始分数。
            representation
        ).squeeze(-1)  # [B,1] -> [B]，得到 raw score。
        reversed_representation = gradient_reverse(  # [B,D]，前向形状不变。
            representation, gamma
        )
        return (  # 返回 raw score、风格预测、行业 logits 和 CLS 表征。
            raw_score,  # [B]。
            self.style_head(reversed_representation),  # [B,10]。
            self.industry_head(reversed_representation),  # [B,32]。
            representation,  # [B,D]。
        )
