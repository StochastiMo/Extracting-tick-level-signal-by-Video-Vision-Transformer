"""No.6：Barra 条件化 Alpha/Style 双塔。"""
from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from stage04_no4_gate import (
    BarraConditionalViViT,
    prepare_conditioning_inputs,
)


def prepare_dual_tower_inputs(  # 返回 Alpha 塔与 Style 塔的条件输入。
    styles: torch.Tensor,  # [B,10]，十个连续 Barra 暴露。
    style_valid: torch.Tensor,  # [B,10]，风格暴露有效掩码。
    industry_codes: torch.Tensor,  # [B]，行业整数代码。
    sample_indices: torch.Tensor,  # [B]，全局样本行号。
    *,
    mode: str,  # 接收模式。
) -> tuple[  # 声明当前函数的返回类型。
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """返回 Alpha 塔与 Style 塔的条件输入。"""

    if mode not in {"real", "alpha_video_only", "shuffled", "zero"}:  # 检查当前值是否属于允许集合。
        raise ValueError(f"未知 No.6 双塔输入模式：{mode}")  # 输入数据不满足业务或数值约束时立即终止。
    true_inputs = (styles, style_valid, industry_codes)
    if mode in {"real", "alpha_video_only"}:  # 检查当前值是否属于允许集合。
        return true_inputs, true_inputs  # 返回 Alpha 塔与 Style 塔的条件输入。
    transformed = prepare_conditioning_inputs(
        styles,  # 汇入风格。
        style_valid,  # 汇入风格有效。
        industry_codes,  # 汇入行业代码。
        sample_indices,  # 汇入样本索引。
        mode=mode,  # 选择当前处理模式。
    )
    return transformed, transformed  # 返回 Alpha 塔与 Style 塔的条件输入。


# region TR-N06-MODEL
class BarraDualTowerViViT(nn.Module):  # 条件 Alpha 塔与独立 Barra Style 塔。
    """条件 Alpha 塔与独立 Barra Style 塔。
    """

    def __init__(  # 初始化当前对象的配置、状态与子组件。
        self,  # 汇入self。
        alpha_tower: BarraConditionalViViT,  # 接收Alphatower。
        *,
        style_count: int,  # 接收风格数量。
        industry_count: int,  # 接收行业数量。
        industry_embedding_dim: int,  # 接收行业嵌入dim。
        style_hidden_dim: int,  # 接收风格隐藏层dim。
        style_dropout: float,  # 接收风格dropout。
    ) -> None:  # 声明当前函数的返回类型。
        super().__init__()
        self.alpha_tower = alpha_tower  # 保存Alphatower，供该对象后续步骤复用。
        self.style_count = int(style_count)  # 保存风格数量，供该对象后续步骤复用。
        self.industry_count = int(industry_count)  # 保存行业数量，供该对象后续步骤复用。
        self.style_industry_embedding = nn.Embedding(  # 将离散状态映射为 embedding。
            industry_count + 1,
            industry_embedding_dim,  # 汇入行业嵌入dim。
            padding_idx=0,  # 固定缺失类别的零嵌入。
        )
        input_dim = 2 * style_count + industry_embedding_dim
        self.style_encoder = nn.Sequential(  # 按顺序组成网络子层。
            nn.Linear(input_dim, style_hidden_dim),
            nn.GELU(),
            nn.Dropout(style_dropout),
            nn.Linear(style_hidden_dim, style_hidden_dim),
            nn.GELU(),
        )
        self.style_output = nn.Linear(style_hidden_dim, 1)  # 线性投影到目标维度。
        self._initialize_style_tower()  # 初始化本分支参数。

    def _initialize_style_tower(self) -> None:  # 令初始 q=0、r=a，同时初始化 Style 塔隐藏层。
        """令初始 q=0、r=a，同时初始化 Style 塔隐藏层。"""

        nn.init.trunc_normal_(  # 截断正态初始化。
            self.style_industry_embedding.weight[1:], std=0.02
        )
        with torch.no_grad():  # 关闭梯度记录以降低验证或推理开销。
            self.style_industry_embedding.weight[0].zero_()
        for module in self.style_encoder.modules():  # 遍历风格encodermodules，逐项处理module。
            if isinstance(module, nn.Linear):  # 检查对象类型是否符合接口约定。
                nn.init.trunc_normal_(module.weight, std=0.02)  # 截断正态初始化。
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.style_output.weight)
        nn.init.zeros_(self.style_output.bias)

    def forward(  # 返回 Alpha、Style、Total、Gate 和 CLS 表征。
        self,  # 汇入self。
        videos: torch.Tensor,  # [B,17,20,8,8]，订单流视频。
        frame_mask: torch.Tensor,  # [B,20]，逐帧有效标记。
        alpha_styles: torch.Tensor,  # [B,10]，Alpha 塔风格输入。
        alpha_style_valid: torch.Tensor,  # [B,10]，Alpha 塔风格有效掩码。
        alpha_industry_codes: torch.Tensor,  # [B]，Alpha 塔行业代码。
        style_styles: torch.Tensor,  # [B,10]，Style 塔风格输入。
        style_style_valid: torch.Tensor,  # [B,10]，Style 塔风格有效掩码。
        style_industry_codes: torch.Tensor,  # [B]，Style 塔行业代码。
    ) -> tuple[  # 声明当前函数的返回类型。
        torch.Tensor,  # 声明当前返回值中的一个数据项。
        torch.Tensor,  # 声明当前返回值中的一个数据项。
        torch.Tensor,  # 声明当前返回值中的一个数据项。
        torch.Tensor,  # 声明当前返回值中的一个数据项。
        torch.Tensor,  # 声明当前返回值中的一个数据项。
    ]:
        """返回 Alpha、Style、Total、Gate 和 CLS 表征。"""

        alpha_score, gate, representation = self.alpha_tower(  # 通过当前网络子模块。
            videos,
            frame_mask,
            alpha_styles,
            alpha_style_valid,
            alpha_industry_codes,
        )
        if style_styles.shape != (  # 检查张量或数组形状是否符合接口约定。
            len(alpha_score), self.style_count
        ):
            raise ValueError("No.6 Style 塔十风格张量形状异常")  # 输入数据不满足业务或数值约束时立即终止。
        if style_style_valid.shape != style_styles.shape:  # 检查张量或数组形状是否符合接口约定。
            raise ValueError("No.6 Style 塔有效掩码形状异常")  # 输入数据不满足业务或数值约束时立即终止。
        codes = style_industry_codes.long().reshape(-1)  # [B,...] -> [B]。
        if len(codes) != len(alpha_score) or bool(  # 检查当前样本数量是否满足计算要求。
            ((codes < 0) | (codes > self.industry_count)).any()
        ):
            raise ValueError("No.6 Style 塔行业代码异常")  # 输入数据不满足业务或数值约束时立即终止。
        condition = torch.cat(  # 风格值、有效 mask 与行业 embedding 拼成 [B,2K+E]。
            [
                style_styles.float(),
                style_style_valid.float(),
                self.style_industry_embedding(codes),
            ],
            dim=1,  # 指定张量运算维度。
        )
        style_score = self.style_output(  # [B,2K+E] -> [B,H] -> [B,1]。
            self.style_encoder(condition)
        ).squeeze(-1)  # [B,1] -> [B]。
        total_score = alpha_score + style_score  # [B] + [B] -> [B]。
        return (  # 返回 Alpha、Style、Total、Gate 和 CLS 表征。
            alpha_score,
            style_score,
            total_score,
            gate,
            representation,
        )
# endregion


# region TR-N06-LOSS
def dual_tower_training_loss(  # 计算总收益、Alpha、Style 和 Alpha 暴露四项损失。
    alpha_score: torch.Tensor,  # [B]，Alpha 塔分数。
    style_score: torch.Tensor,  # [B]，Style 塔分数。
    total_score: torch.Tensor,  # [B]，Alpha 与 Style 合计分数。
    original_targets: torch.Tensor,  # [B]，原始收益标签。
    residual_targets: torch.Tensor,  # [B]，Barra 残差标签。
    fitted_targets: torch.Tensor,  # [B]，Barra 拟合标签。
    style_exposures: torch.Tensor,  # [B,10]，十个连续 Barra 暴露。
    industry_codes: torch.Tensor,  # [B]，行业整数代码。
    *,
    return_rank_loss: Callable[  # 返回上述计算结果。
        [torch.Tensor, torch.Tensor], torch.Tensor
    ],
    alpha_rank_loss: Callable[  # 接收Alpha排名损失。
        [torch.Tensor, torch.Tensor], torch.Tensor
    ],
    style_rank_loss: Callable[  # 接收风格排名损失。
        [torch.Tensor, torch.Tensor], torch.Tensor
    ],
    barra_r_squared: Callable[  # 接收Barrarsquared。
        [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
    ],
    alpha_loss_weight: float,  # 接收Alpha损失权重。
    style_loss_weight: float,  # 接收风格损失权重。
    r_squared_weight: float,  # 接收rsquared权重。
) -> dict[str, torch.Tensor]:  # 声明当前函数的返回类型。
    """计算总收益、Alpha、Style 和 Alpha 暴露四项损失。"""

    return_loss = return_rank_loss(total_score, original_targets)  # 按实验口径计算可微损失。
    alpha_loss = alpha_rank_loss(alpha_score, residual_targets)  # 按实验口径计算可微损失。
    style_loss = style_rank_loss(style_score, fitted_targets)  # 按实验口径计算可微损失。
    alpha_r_squared = barra_r_squared(  # 计算 Barra 可解释比例。
        alpha_score, style_exposures, industry_codes  # 传入Alpha分数、风格暴露、行业代码。
    )
    total_loss = (  # 计算total损失。
        return_loss.float()  # 返回上述计算结果。
        + alpha_loss_weight * alpha_loss.float()
        + style_loss_weight * style_loss.float()
        + r_squared_weight * alpha_r_squared
    )
    return {  # 返回上述计算结果。
        "loss": total_loss,  # 记录损失。
        "return_rank_loss": return_loss,  # 记录收益秩损失。
        "alpha_rank_loss": alpha_loss,  # 记录Alpha秩损失。
        "style_rank_loss": style_loss,  # 记录风格秩损失。
        "alpha_barra_r_squared": alpha_r_squared,  # 记录AlphaBarrarsquared。
    }
# endregion
