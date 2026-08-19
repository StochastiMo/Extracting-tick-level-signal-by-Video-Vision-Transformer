"""No.6：Barra 条件化 Alpha/Style 双塔。"""
from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from stage04_no4_gate import (
    BarraConditionalViViT,
    prepare_conditioning_inputs,
)


def prepare_dual_tower_inputs(
    styles: torch.Tensor,
    style_valid: torch.Tensor,
    industry_codes: torch.Tensor,
    sample_indices: torch.Tensor,
    *,
    mode: str,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """返回 Alpha 塔与 Style 塔的条件输入。"""

    if mode not in {"real", "alpha_video_only", "shuffled", "zero"}:
        raise ValueError(f"未知 No.6 双塔输入模式：{mode}")
    true_inputs = (styles, style_valid, industry_codes)
    if mode in {"real", "alpha_video_only"}:
        return true_inputs, true_inputs
    transformed = prepare_conditioning_inputs(
        styles,
        style_valid,
        industry_codes,
        sample_indices,
        mode=mode,
    )
    return transformed, transformed


# region TR-N06-MODEL
class BarraDualTowerViViT(nn.Module):
    """条件 Alpha 塔与独立 Barra Style 塔。
    """

    def __init__(
        self,
        alpha_tower: BarraConditionalViViT,
        *,
        style_count: int,
        industry_count: int,
        industry_embedding_dim: int,
        style_hidden_dim: int,
        style_dropout: float,
    ) -> None:
        super().__init__()
        self.alpha_tower = alpha_tower
        self.style_count = int(style_count)
        self.industry_count = int(industry_count)
        self.style_industry_embedding = nn.Embedding(  # 将离散状态映射为 embedding
            industry_count + 1,
            industry_embedding_dim,
            padding_idx=0,
        )
        input_dim = 2 * style_count + industry_embedding_dim
        self.style_encoder = nn.Sequential(  # 按顺序组成网络子层
            nn.Linear(input_dim, style_hidden_dim),
            nn.GELU(),
            nn.Dropout(style_dropout),
            nn.Linear(style_hidden_dim, style_hidden_dim),
            nn.GELU(),
        )
        self.style_output = nn.Linear(style_hidden_dim, 1)  # 线性投影到目标维度
        self._initialize_style_tower()  # 初始化本分支参数

    def _initialize_style_tower(self) -> None:
        """令初始 q=0、r=a，同时初始化 Style 塔隐藏层。"""

        nn.init.trunc_normal_(  # 截断正态初始化
            self.style_industry_embedding.weight[1:], std=0.02
        )
        with torch.no_grad():
            self.style_industry_embedding.weight[0].zero_()
        for module in self.style_encoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)  # 截断正态初始化
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.style_output.weight)
        nn.init.zeros_(self.style_output.bias)

    def forward(
        self,
        videos: torch.Tensor,
        frame_mask: torch.Tensor,
        alpha_styles: torch.Tensor,
        alpha_style_valid: torch.Tensor,
        alpha_industry_codes: torch.Tensor,
        style_styles: torch.Tensor,
        style_style_valid: torch.Tensor,
        style_industry_codes: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """返回 Alpha、Style、Total、Gate 和 CLS 表征。"""

        alpha_score, gate, representation = self.alpha_tower(  # 通过当前网络子模块
            videos,
            frame_mask,
            alpha_styles,
            alpha_style_valid,
            alpha_industry_codes,
        )
        if style_styles.shape != (
            len(alpha_score), self.style_count
        ):
            raise ValueError("No.6 Style 塔十风格张量形状异常")
        if style_style_valid.shape != style_styles.shape:
            raise ValueError("No.6 Style 塔有效掩码形状异常")
        codes = style_industry_codes.long().reshape(-1)
        if len(codes) != len(alpha_score) or bool(
            ((codes < 0) | (codes > self.industry_count)).any()
        ):
            raise ValueError("No.6 Style 塔行业代码异常")
        condition = torch.cat(
            [
                style_styles.float(),
                style_style_valid.float(),
                self.style_industry_embedding(codes),
            ],
            dim=1,
        )
        style_score = self.style_output(
            self.style_encoder(condition)
        ).squeeze(-1)
        total_score = alpha_score + style_score
        return (
            alpha_score,
            style_score,
            total_score,
            gate,
            representation,
        )
# endregion


# region TR-N06-LOSS
def dual_tower_training_loss(
    alpha_score: torch.Tensor,
    style_score: torch.Tensor,
    total_score: torch.Tensor,
    original_targets: torch.Tensor,
    residual_targets: torch.Tensor,
    fitted_targets: torch.Tensor,
    style_exposures: torch.Tensor,
    industry_codes: torch.Tensor,
    *,
    return_rank_loss: Callable[
        [torch.Tensor, torch.Tensor], torch.Tensor
    ],
    alpha_rank_loss: Callable[
        [torch.Tensor, torch.Tensor], torch.Tensor
    ],
    style_rank_loss: Callable[
        [torch.Tensor, torch.Tensor], torch.Tensor
    ],
    barra_r_squared: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
    ],
    alpha_loss_weight: float,
    style_loss_weight: float,
    r_squared_weight: float,
) -> dict[str, torch.Tensor]:
    """计算总收益、Alpha、Style 和 Alpha 暴露四项损失。"""

    return_loss = return_rank_loss(total_score, original_targets)  # 按实验口径计算可微损失
    alpha_loss = alpha_rank_loss(alpha_score, residual_targets)  # 按实验口径计算可微损失
    style_loss = style_rank_loss(style_score, fitted_targets)  # 按实验口径计算可微损失
    alpha_r_squared = barra_r_squared(  # 计算 Barra 可解释比例
        alpha_score, style_exposures, industry_codes
    )
    total_loss = (
        return_loss.float()
        + alpha_loss_weight * alpha_loss.float()
        + style_loss_weight * style_loss.float()
        + r_squared_weight * alpha_r_squared
    )
    return {
        "loss": total_loss,
        "return_rank_loss": return_loss,
        "alpha_rank_loss": alpha_loss,
        "style_rank_loss": style_loss,
        "alpha_barra_r_squared": alpha_r_squared,
    }
# endregion
