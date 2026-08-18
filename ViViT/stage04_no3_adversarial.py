"""No.3：对抗式 Barra 不变表征。"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from stage03_model import OrderFlowViViT


# 定义 `_GradientReversal` 类，集中封装该阶段的数据结构与相关行为。
class _GradientReversal(torch.autograd.Function):
    """前向恒等、反向乘以负系数的梯度反转算子。
    """

    # 使用 `staticmethod` 包装 `forward`，并定义其可复用的处理逻辑。
    @staticmethod
    def forward(
        ctx: Any, values: torch.Tensor, gamma: torch.Tensor
    ) -> torch.Tensor:
        """执行 forward 对应的处理步骤。
        
        输入：
            ctx: Any。
            values: torch.Tensor。
            gamma: torch.Tensor。
        输出：
            torch.Tensor。
            """
        # 调用 `ctx.save_for_backward` 按约定格式写出当前计算产物。
        ctx.save_for_backward(gamma)
        # 将 `values.clone()` 作为当前函数结果返回给调用方。
        return values.clone()

    # 使用 `staticmethod` 包装 `backward`，并定义其可复用的处理逻辑。
    @staticmethod
    def backward(
        ctx: Any, gradient: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        """执行 backward 对应的处理步骤。
        
        输入：
            ctx: Any。
            gradient: torch.Tensor。
        输出：
            tuple[torch.Tensor, None]。
            """
        # 读取 `ctx.saved_tensors` 的当前值，并赋给 `(gamma,)`。
        (gamma,) = ctx.saved_tensors
        # 将 `(-gamma * gradient, None)` 作为当前函数结果返回给调用方。
        return -gamma * gradient, None


# region TR-N03-GRL
# 定义 `gradient_reverse` 函数，封装这一阶段可独立调用的处理步骤。
def gradient_reverse(
    values: torch.Tensor, gamma: float | torch.Tensor
) -> torch.Tensor:
    """保持前向值不变，让编码器收到负 gamma 倍梯度。
    
    输入：
        values: torch.Tensor。
        gamma: float | torch.Tensor。
    输出：
        torch.Tensor。
        """

    # 调用 `torch.as_tensor` 转换数据表示，并将标准化对象保存到 `coefficient`。
    coefficient = torch.as_tensor(
        gamma, dtype=values.dtype, device=values.device
    )
    # 将 `_GradientReversal.apply(values, coefficient)` 作为当前函数结果返回给调用方。
    return _GradientReversal.apply(values, coefficient)
# endregion


# 定义 `cosine_adversarial_gamma` 函数，封装这一阶段可独立调用的处理步骤。
def cosine_adversarial_gamma(
    epoch: int,
    *,
    gamma_max: float,
    warmup_epochs: int,
    ramp_epochs: int,
) -> float:
    """按绝对 epoch 计算预热、余弦爬坡和稳定阶段的 GRL 系数。

    输入：
        epoch: int，当前绝对训练轮次，从 1 开始。
        gamma_max: float，梯度反转系数上限。
        warmup_epochs: int，系数保持为零的预热轮数。
        ramp_epochs: int，从零爬升到上限所需轮数。
    输出：
        float，当前轮次使用的梯度反转系数。
    """
    # 检查 `epoch <= warmup_epochs`；根据条件结果选择对应处理分支。
    if epoch <= warmup_epochs:
        # 将 `0.0` 作为当前函数结果返回给调用方。
        return 0.0
    # 调用 `min` 汇总当前数据，并将统计量保存到 `progress`。
    progress = min(1.0, (epoch - warmup_epochs) / ramp_epochs)
    # 将 `float(0.5 * gamma_max * (1.0 - math.cos(math.pi * progress)))` 作为当前函数结果返回给调用方。
    return float(
        0.5 * gamma_max * (1.0 - math.cos(math.pi * progress))
    )


# 定义内部辅助函数 `_adversarial_head`，封装仅供本模块复用的计算步骤。
def _adversarial_head(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    dropout: float,
) -> nn.Sequential:
    """构造一个两层对抗预测头。
    
    输入：
        input_dim: int。
        hidden_dim: int。
        output_dim: int。
        dropout: float。
    输出：
        nn.Sequential。
        """

    # 将 `nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidde…` 作为当前函数结果返回给调用方。
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )


# region TR-N03-MODEL
# 定义 `AdversarialBarraViViT` 类，集中封装该阶段的数据结构与相关行为。
class AdversarialBarraViViT(nn.Module):
    """收益头预测 Alpha，对抗头尝试从 CLS 恢复风格和行业。
    """

    # 定义初始化方法，接收配置并建立实例后续计算所需的内部状态。
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
        """执行 __init__ 对应的处理步骤。
        
        输入：
            backbone: OrderFlowViViT。
            embedding_dim: int。
            style_count: int。
            industry_count: int。
            hidden_dim: int。
            dropout: float。
        输出：
            None。
            """
        # 初始化父类状态，确保继承链中的基础组件正确注册。
        super().__init__()
        # 读取 `backbone` 的当前值，并赋给 `self.backbone`。
        self.backbone = backbone
        # 调用 `_adversarial_head` 执行 `_adversarial_head` 对应的明确变换，并产生后续步骤使用的结果，并将结果保存到 `self.style_head`。
        self.style_head = _adversarial_head(
            embedding_dim, hidden_dim, style_count, dropout
        )
        # 调用 `_adversarial_head` 执行 `_adversarial_head` 对应的明确变换，并产生后续步骤使用的结果，并将结果保存到 `self.industry_head`。
        self.industry_head = _adversarial_head(
            embedding_dim, hidden_dim, industry_count, dropout
        )
        # 调用 `self._initialize_adversarial_heads` 按模型配置初始化当前分支的可训练参数。
        self._initialize_adversarial_heads()

    # 定义内部辅助函数 `_initialize_adversarial_heads`，封装仅供本模块复用的计算步骤。
    def _initialize_adversarial_heads(self) -> None:
        """初始化新增分支，不改变 ViViT 主干参数。
        
        输入：
            无显式输入参数。
        输出：
            None。
            """

        # 遍历 `(self.style_head, self.industry_head)`，并把当前元素绑定到 `head` 逐项处理。
        for head in (self.style_head, self.industry_head):
            # 遍历 `head.modules()`，并把当前元素绑定到 `module` 逐项处理。
            for module in head.modules():
                # 检查 `isinstance(module, nn.Linear)`；根据条件结果选择对应处理分支。
                if isinstance(module, nn.Linear):
                    # 调用 `nn.init.trunc_normal_` 用截断正态分布初始化可训练参数。
                    nn.init.trunc_normal_(module.weight, std=0.02)
                    # 检查 `module.bias is not None`；根据条件结果选择对应处理分支。
                    if module.bias is not None:
                        # 调用 `nn.init.zeros_` 将指定参数初始化为零。
                        nn.init.zeros_(module.bias)

    # 定义 `forward` 函数，封装这一阶段可独立调用的处理步骤。
    def forward(
        self,
        videos: torch.Tensor,
        frame_mask: torch.Tensor,
        gamma: torch.Tensor,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """返回 raw score、风格预测、行业 logits 和 CLS 表征。
        
        输入：
            videos: torch.Tensor，格式 [B,17,20,8,8]。
            frame_mask: torch.Tensor，格式 [B,20]。
            gamma: torch.Tensor。
        输出：
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]。
            """

        # 调用 `self.backbone.encode` 用模型主干把视频编码为共享表征，并将结果保存到 `representation`。
        representation = self.backbone.encode(videos, frame_mask)
        # 调用 `self.backbone.regression_head(representation).squeeze` 调整张量维度，并将新视图保存到 `raw_score`。
        raw_score = self.backbone.regression_head(
            representation
        ).squeeze(-1)
        # 调用 `gradient_reverse` 执行 `gradient_reverse` 对应的明确变换，并产生后续步骤使用的结果，并将结果保存到 `reversed_representation`。
        reversed_representation = gradient_reverse(
            representation, gamma
        )
        # 将 `(raw_score, self.style_head(reversed_representation), self.industry_head(reversed_representatio…` 作为当前函数结果返回给调用方。
        return (
            raw_score,
            self.style_head(reversed_representation),
            self.industry_head(reversed_representation),
            representation,
        )
# endregion
