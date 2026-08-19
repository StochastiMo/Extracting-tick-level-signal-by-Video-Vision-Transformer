"""No.4：Barra 条件化乘法门控。"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from stage03_model import OrderFlowViViT


def deterministic_barra_permutation(
    sample_indices: torch.Tensor,
) -> torch.Tensor:
    """按稳定样本键产生与输入顺序无关的日内置乱位置。"""

    identifiers = sample_indices.detach().cpu().numpy().astype(
        np.int64, copy=False
    )
    if identifiers.ndim != 1 or len(identifiers) < 2:
        raise ValueError("Barra 置乱至少需要两个一维样本键")
    if len(np.unique(identifiers)) != len(identifiers):
        raise ValueError("Barra 置乱样本键含重复值")
    canonical_positions = np.argsort(identifiers, kind="stable")
    canonical_ids = identifiers[canonical_positions]
    random_seed = np.random.SeedSequence(
        [len(canonical_ids), int(canonical_ids[0]), int(canonical_ids[-1])]
    ).generate_state(1)[0]
    generator = np.random.default_rng(random_seed)
    donor_positions = canonical_positions[
        generator.permutation(len(canonical_positions))
    ]
    permutation = np.empty(len(identifiers), dtype=np.int64)
    permutation[canonical_positions] = donor_positions
    return torch.from_numpy(permutation).to(sample_indices.device)


def prepare_conditioning_inputs(
    styles: torch.Tensor,
    style_valid: torch.Tensor,
    industry_codes: torch.Tensor,
    sample_indices: torch.Tensor,
    *,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """生成 real/shuffled/zero 条件输入，不改变诊断使用的真实 Barra。"""

    if mode in {"off", "video_only", "real"}:
        return styles, style_valid, industry_codes
    if mode == "zero":
        return (
            torch.zeros_like(styles),
            torch.zeros_like(style_valid, dtype=torch.bool),
            torch.zeros_like(industry_codes),
        )
    if mode != "shuffled":
        raise ValueError(f"未知 Barra 条件模式：{mode}")
    permutation = deterministic_barra_permutation(sample_indices).to(
        styles.device
    )
    return (
        styles.index_select(0, permutation),
        style_valid.index_select(0, permutation),
        industry_codes.index_select(0, permutation),
    )


# region TR-N04-GATE
class BarraConditionalViViT(nn.Module):
    """用十风格、有效标记和行业对 CLS 做有界乘法调制。
    """

    def __init__(
        self,
        backbone: OrderFlowViViT,
        *,
        conditioning_mode: str,
        style_count: int,
        industry_count: int,
        embedding_dim: int,
        gate_hidden_dim: int,
        industry_embedding_dim: int,
        dropout: float,
        gate_scale: float,
    ) -> None:
        super().__init__()
        if conditioning_mode not in {"video_only", "real"}:
            raise ValueError("条件模型内部模式只能是 video_only 或 real")
        self.backbone = backbone
        self.conditioning_mode = conditioning_mode
        self.style_count = int(style_count)
        self.industry_count = int(industry_count)
        self.gate_scale = float(gate_scale)
        self.industry_embedding = nn.Embedding(  # 将离散状态映射为 embedding
            industry_count + 1,
            industry_embedding_dim,
            padding_idx=0,
        )
        input_dim = 2 * style_count + industry_embedding_dim
        self.barra_encoder = nn.Sequential(  # 按顺序组成网络子层
            nn.Linear(input_dim, gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_dim, gate_hidden_dim),
            nn.GELU(),
        )
        self.gate_projection = nn.Linear(  # 线性投影到目标维度
            gate_hidden_dim, embedding_dim
        )
        self._initialize_conditioning_branch()  # 初始化本分支参数

    def _initialize_conditioning_branch(self) -> None:
        """零初始化最终门控，使初始输出严格等于无条件模型。"""

        nn.init.trunc_normal_(  # 截断正态初始化
            self.industry_embedding.weight[1:], std=0.02
        )
        with torch.no_grad():
            self.industry_embedding.weight[0].zero_()
        for module in self.barra_encoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)  # 截断正态初始化
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.gate_projection.weight)
        nn.init.zeros_(self.gate_projection.bias)

    def forward(
        self,
        videos: torch.Tensor,
        frame_mask: torch.Tensor,
        styles: torch.Tensor,
        style_valid: torch.Tensor,
        industry_codes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 raw score、门控向量和未条件化 CLS。"""

        representation = self.backbone.encode(videos, frame_mask)  # 编码为共享视频表征
        if self.conditioning_mode == "video_only":
            gate = torch.zeros_like(representation)
        else:
            if styles.shape != (
                len(representation), self.style_count
            ):
                raise ValueError("十风格条件张量形状异常")
            if style_valid.shape != styles.shape:
                raise ValueError("风格有效掩码形状异常")
            codes = industry_codes.long().reshape(-1)
            if len(codes) != len(representation) or bool(
                ((codes < 0) | (codes > self.industry_count)).any()
            ):
                raise ValueError("行业代码形状或取值异常")
            condition = torch.cat(
                [
                    styles.float(),
                    style_valid.float(),
                    self.industry_embedding(codes),
                ],
                dim=1,
            )
            encoded = self.barra_encoder(condition)  # 通过当前网络子模块
            gate = self.gate_scale * torch.tanh(
                self.gate_projection(encoded)
            )
        conditioned = representation * (1.0 + gate)
        raw_score = self.backbone.regression_head(
            conditioned
        ).squeeze(-1)
        return raw_score, gate, representation
# endregion
