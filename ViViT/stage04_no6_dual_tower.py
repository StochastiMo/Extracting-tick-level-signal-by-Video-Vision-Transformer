"""No.6：Barra 条件化 Alpha/Style 双塔。"""
from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from stage04_no4_gate import (
    BarraConditionalViViT,
    prepare_conditioning_inputs,
)


# 定义 `prepare_dual_tower_inputs` 函数，封装这一阶段可独立调用的处理步骤。
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
    """返回 Alpha 塔与 Style 塔的条件输入。
    
    输入：
        styles: torch.Tensor，格式 [N,10]。
        style_valid: torch.Tensor，格式 [N,10]。
        industry_codes: torch.Tensor，格式 [N]。
        sample_indices: torch.Tensor，格式 [N]。
        mode: str。
    输出：
        tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]。
        """

    # 检查 `mode not in {'real', 'alpha_video_only', 'shuffled', 'zero'}`；根据条件结果选择对应处理分支。
    if mode not in {"real", "alpha_video_only", "shuffled", "zero"}:
        # 抛出 `ValueError(f'未知 No.6 双塔输入模式：{mode}')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(f"未知 No.6 双塔输入模式：{mode}")
    # 组织一组有序值，并保存到 `true_inputs`。
    true_inputs = (styles, style_valid, industry_codes)
    # 检查 `mode in {'real', 'alpha_video_only'}`；根据条件结果选择对应处理分支。
    if mode in {"real", "alpha_video_only"}:
        # 将 `(true_inputs, true_inputs)` 作为当前函数结果返回给调用方。
        return true_inputs, true_inputs
    # 调用 `prepare_conditioning_inputs` 根据当前配置和输入构造后续流程所需对象，并将结果保存到 `transformed`。
    transformed = prepare_conditioning_inputs(
        styles,
        style_valid,
        industry_codes,
        sample_indices,
        mode=mode,
    )
    # 将 `(transformed, transformed)` 作为当前函数结果返回给调用方。
    return transformed, transformed


# region TR-N06-MODEL
# 定义 `BarraDualTowerViViT` 类，集中封装该阶段的数据结构与相关行为。
class BarraDualTowerViViT(nn.Module):
    """条件 Alpha 塔与独立 Barra Style 塔。
    """

    # 定义初始化方法，接收配置并建立实例后续计算所需的内部状态。
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
        """执行 __init__ 对应的处理步骤。
        
        输入：
            alpha_tower: BarraConditionalViViT。
            style_count: int。
            industry_count: int。
            industry_embedding_dim: int。
            style_hidden_dim: int。
            style_dropout: float。
        输出：
            None。
            """
        # 初始化父类状态，确保继承链中的基础组件正确注册。
        super().__init__()
        # 读取 `alpha_tower` 的当前值，并赋给 `self.alpha_tower`。
        self.alpha_tower = alpha_tower
        # 调用 `int` 统一设备或数据类型，并将结果保存到 `self.style_count`。
        self.style_count = int(style_count)
        # 调用 `int` 统一设备或数据类型，并将结果保存到 `self.industry_count`。
        self.industry_count = int(industry_count)
        # 调用 `nn.Embedding` 创建离散状态到连续向量的可学习映射，并将结果保存到 `self.style_industry_embedding`。
        self.style_industry_embedding = nn.Embedding(
            industry_count + 1,
            industry_embedding_dim,
            padding_idx=0,
        )
        # 计算 `2 * style_count + industry_embedding_dim`，并将数值结果保存到 `input_dim`。
        input_dim = 2 * style_count + industry_embedding_dim
        # 调用 `nn.Sequential` 按执行顺序组合多个神经网络子层，并将结果保存到 `self.style_encoder`。
        self.style_encoder = nn.Sequential(
            nn.Linear(input_dim, style_hidden_dim),
            nn.GELU(),
            nn.Dropout(style_dropout),
            nn.Linear(style_hidden_dim, style_hidden_dim),
            nn.GELU(),
        )
        # 调用 `nn.Linear` 创建线性投影层以变换特征维度，并将结果保存到 `self.style_output`。
        self.style_output = nn.Linear(style_hidden_dim, 1)
        # 调用 `self._initialize_style_tower` 按模型配置初始化当前分支的可训练参数。
        self._initialize_style_tower()

    # 定义内部辅助函数 `_initialize_style_tower`，封装仅供本模块复用的计算步骤。
    def _initialize_style_tower(self) -> None:
        """令初始 q=0、r=a，同时初始化 Style 塔隐藏层。
        
        输入：
            无显式输入参数。
        输出：
            None。
            """

        # 调用 `nn.init.trunc_normal_` 用截断正态分布初始化可训练参数。
        nn.init.trunc_normal_(
            self.style_industry_embedding.weight[1:], std=0.02
        )
        # 在 `torch.no_grad()` 管理的上下文中执行本段逻辑，并保证资源按约定释放。
        with torch.no_grad():
            # 调用 `self.style_industry_embedding.weight[0].zero_` 执行 `zero_` 对应的明确变换，并产生后续步骤使用的结果。
            self.style_industry_embedding.weight[0].zero_()
        # 遍历 `self.style_encoder.modules()`，并把当前元素绑定到 `module` 逐项处理。
        for module in self.style_encoder.modules():
            # 检查 `isinstance(module, nn.Linear)`；根据条件结果选择对应处理分支。
            if isinstance(module, nn.Linear):
                # 调用 `nn.init.trunc_normal_` 用截断正态分布初始化可训练参数。
                nn.init.trunc_normal_(module.weight, std=0.02)
                # 调用 `nn.init.zeros_` 将指定参数初始化为零。
                nn.init.zeros_(module.bias)
        # 调用 `nn.init.zeros_` 将指定参数初始化为零。
        nn.init.zeros_(self.style_output.weight)
        # 调用 `nn.init.zeros_` 将指定参数初始化为零。
        nn.init.zeros_(self.style_output.bias)

    # 定义 `forward` 函数，封装这一阶段可独立调用的处理步骤。
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
        """返回 Alpha、Style、Total、Gate 和 CLS 表征。
        
        输入：
            videos: torch.Tensor，格式 [B,17,20,8,8]。
            frame_mask: torch.Tensor，格式 [B,20]。
            alpha_styles: torch.Tensor。
            alpha_style_valid: torch.Tensor。
            alpha_industry_codes: torch.Tensor。
            style_styles: torch.Tensor。
            style_style_valid: torch.Tensor。
            style_industry_codes: torch.Tensor。
        输出：
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]。
            """

        # 调用 `self.alpha_tower` 通过当前网络子模块变换输入表征，并将结果保存到 `(alpha_score, gate, representation)`。
        alpha_score, gate, representation = self.alpha_tower(
            videos,
            frame_mask,
            alpha_styles,
            alpha_style_valid,
            alpha_industry_codes,
        )
        # 检查 `style_styles.shape != (len(alpha_score), self.style_count)`；根据条件结果选择对应处理分支。
        if style_styles.shape != (
            len(alpha_score), self.style_count
        ):
            # 抛出 `ValueError('No.6 Style 塔十风格张量形状异常')`，阻止不满足数据契约的流程继续执行。
            raise ValueError("No.6 Style 塔十风格张量形状异常")
        # 检查 `style_style_valid.shape != style_styles.shape`；根据条件结果选择对应处理分支。
        if style_style_valid.shape != style_styles.shape:
            # 抛出 `ValueError('No.6 Style 塔有效掩码形状异常')`，阻止不满足数据契约的流程继续执行。
            raise ValueError("No.6 Style 塔有效掩码形状异常")
        # 调用 `style_industry_codes.long().reshape` 调整张量维度，并将新视图保存到 `codes`。
        codes = style_industry_codes.long().reshape(-1)
        # 检查 `len(codes) != len(alpha_score) or bool(((codes < 0) | (codes > self.industry_count)).any())`；根据条件结果选择对应处理分支。
        if len(codes) != len(alpha_score) or bool(
            ((codes < 0) | (codes > self.industry_count)).any()
        ):
            # 抛出 `ValueError('No.6 Style 塔行业代码异常')`，阻止不满足数据契约的流程继续执行。
            raise ValueError("No.6 Style 塔行业代码异常")
        # 调用 `torch.cat` 拼接各批结果，并将合并值保存到 `condition`。
        condition = torch.cat(
            [
                style_styles.float(),
                style_style_valid.float(),
                self.style_industry_embedding(codes),
            ],
            dim=1,
        )
        # 调用 `self.style_output(self.style_encoder(condition)).squeeze` 调整张量维度，并将新视图保存到 `style_score`。
        style_score = self.style_output(
            self.style_encoder(condition)
        ).squeeze(-1)
        # 计算 `alpha_score + style_score`，并将数值结果保存到 `total_score`。
        total_score = alpha_score + style_score
        # 将 `(alpha_score, style_score, total_score, gate, representation)` 作为当前函数结果返回给调用方。
        return (
            alpha_score,
            style_score,
            total_score,
            gate,
            representation,
        )
# endregion


# region TR-N06-LOSS
# 定义 `dual_tower_training_loss` 函数，封装这一阶段可独立调用的处理步骤。
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
    """计算总收益、Alpha、Style 和 Alpha 暴露四项损失。
    
    输入：
        alpha_score: torch.Tensor。
        style_score: torch.Tensor。
        total_score: torch.Tensor。
        original_targets: torch.Tensor。
        residual_targets: torch.Tensor。
        fitted_targets: torch.Tensor。
        style_exposures: torch.Tensor，格式 [N,10]。
        industry_codes: torch.Tensor，格式 [N]。
        return_rank_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]。
        alpha_rank_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]。
        style_rank_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]。
        barra_r_squared: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]。
        alpha_loss_weight: float。
        style_loss_weight: float。
        r_squared_weight: float。
    输出：
        dict[str, torch.Tensor]。
        """

    # 调用 `return_rank_loss` 按当前实验口径计算可反向传播的训练损失，并将结果保存到 `return_loss`。
    return_loss = return_rank_loss(total_score, original_targets)
    # 调用 `alpha_rank_loss` 按当前实验口径计算可反向传播的训练损失，并将结果保存到 `alpha_loss`。
    alpha_loss = alpha_rank_loss(alpha_score, residual_targets)
    # 调用 `style_rank_loss` 按当前实验口径计算可反向传播的训练损失，并将结果保存到 `style_loss`。
    style_loss = style_rank_loss(style_score, fitted_targets)
    # 调用 `barra_r_squared` 计算分数的 Barra 分解结果或可解释比例，并将结果保存到 `alpha_r_squared`。
    alpha_r_squared = barra_r_squared(
        alpha_score, style_exposures, industry_codes
    )
    # 计算 `return_loss.float() + alpha_loss_weight * alpha_loss.float() + style_loss_weight * style_loss.f…`，并将数值结果保存到 `total_loss`。
    total_loss = (
        return_loss.float()
        + alpha_loss_weight * alpha_loss.float()
        + style_loss_weight * style_loss.float()
        + r_squared_weight * alpha_r_squared
    )
    # 将 `{'loss': total_loss, 'return_rank_loss': return_loss, 'alpha_rank_loss': alpha_loss, 'style_ran…` 作为当前函数结果返回给调用方。
    return {
        "loss": total_loss,
        "return_rank_loss": return_loss,
        "alpha_rank_loss": alpha_loss,
        "style_rank_loss": style_loss,
        "alpha_barra_r_squared": alpha_r_squared,
    }
# endregion
