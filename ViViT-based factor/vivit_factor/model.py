"""基线时空 ViViT 模型及其唯一训练损失。"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ViViTConfig:
    # 以下均为 README 主流程的基础 baseline 参数，不开放额外模型分支。
    input_channels: int = 17
    frame_count: int = 20
    image_size: int = 8
    temporal_tubelet: int = 2
    spatial_patch: int = 2
    embedding_dim: int = 128
    spatial_encoder_layers: int = 2
    temporal_encoder_layers: int = 2
    attention_heads: int = 4
    mlp_hidden_dim: int = 512
    dropout: float = 0.1
    output_dim: int = 1
    tubelet_coverage_states: int = 3


class TransformerEncoderBlock(nn.Module):
    def __init__(self, config: ViViTConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.embedding_dim)
        self.attention = nn.MultiheadAttention(
            config.embedding_dim, config.attention_heads,
            dropout=config.dropout, batch_first=True,
        )
        self.attention_dropout = nn.Dropout(config.dropout)
        self.mlp_norm = nn.LayerNorm(config.embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.embedding_dim, config.mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_hidden_dim, config.embedding_dim),
            nn.Dropout(config.dropout),
        )

    def forward(
        self, tokens: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # 使用 Pre-LN 结构：先归一化再做注意力，并保留残差连接。
        normalized = self.attention_norm(tokens)
        attended, _ = self.attention(
            normalized, normalized, normalized,
            key_padding_mask=key_padding_mask, need_weights=False,
        )
        tokens = tokens + self.attention_dropout(attended)
        # MLP 子层同样采用 Pre-LN 和残差，输出形状保持不变。
        return tokens + self.mlp(self.mlp_norm(tokens))


class OrderFlowViViT(nn.Module):
    """把形状为 [B,17,20,8,8] 的视频编码为每只股票一个分数。"""

    def __init__(self, config: ViViTConfig = ViViTConfig()) -> None:
        super().__init__()
        self.config = config
        # 20 帧按 2 帧 tubelet 压缩为 10 个时间 token。
        self.temporal_tokens = config.frame_count // config.temporal_tubelet
        patches_per_side = config.image_size // config.spatial_patch
        # 每帧 8×8 按 2×2 patch 划分，共得到 16 个空间 token。
        self.spatial_tokens = patches_per_side**2
        # Conv3d 同时完成时间 tubelet 与空间 patch 的线性嵌入。
        self.tubelet_embedding = nn.Conv3d(
            config.input_channels, config.embedding_dim,
            kernel_size=(config.temporal_tubelet, config.spatial_patch, config.spatial_patch),
            stride=(config.temporal_tubelet, config.spatial_patch, config.spatial_patch),
        )
        self.tubelet_coverage_embedding = nn.Embedding(
            config.tubelet_coverage_states, config.embedding_dim
        )
        # 空间位置编码在所有时间 tubelet 之间共享。
        self.spatial_position = nn.Parameter(
            torch.zeros(1, self.spatial_tokens, config.embedding_dim)
        )
        self.spatial_blocks = nn.ModuleList(
            TransformerEncoderBlock(config) for _ in range(config.spatial_encoder_layers)
        )
        self.temporal_class_token = nn.Parameter(torch.zeros(1, 1, config.embedding_dim))
        # 时间位置编码包含额外的 CLS 位置。
        self.temporal_position = nn.Parameter(
            torch.zeros(1, self.temporal_tokens + 1, config.embedding_dim)
        )
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.temporal_blocks = nn.ModuleList(
            TransformerEncoderBlock(config) for _ in range(config.temporal_encoder_layers)
        )
        self.final_norm = nn.LayerNorm(config.embedding_dim)
        self.regression_head = nn.Linear(config.embedding_dim, config.output_dim)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        # 位置向量、CLS 和线性层沿用 Transformer 常见的截断正态初始化。
        nn.init.trunc_normal_(self.spatial_position, std=0.02)
        nn.init.trunc_normal_(self.temporal_position, std=0.02)
        nn.init.trunc_normal_(self.temporal_class_token, std=0.02)
        nn.init.zeros_(self.tubelet_coverage_embedding.weight)
        # coverage=0 代表全缺失 tubelet，保持零向量；其余状态学习独立表示。
        nn.init.trunc_normal_(self.tubelet_coverage_embedding.weight[1:2], std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def encode(self, videos: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        # 固定输入契约能尽早暴露通道、时间或空间维度错位。
        if videos.ndim != 5 or videos.shape[1:] != (
            self.config.input_channels, self.config.frame_count,
            self.config.image_size, self.config.image_size,
        ):
            raise ValueError(f"unexpected video shape: {tuple(videos.shape)}")
        if frame_mask.shape != (len(videos), self.config.frame_count):
            raise ValueError(f"unexpected frame mask shape: {tuple(frame_mask.shape)}")
        frame_mask = frame_mask.to(device=videos.device, dtype=torch.bool)
        # T 日必须存在；历史帧允许缺失并由后续 padding mask 处理。
        torch._assert(frame_mask[:, -1].all(), "target frame must be valid")

        # [B,C,T,H,W] → [B,D,T',H',W']。
        embedded = self.tubelet_embedding(videos)
        batch_size = embedded.shape[0]
        # 每两个原始帧统计有效帧数，状态仅可能为 0、1、2。
        coverage = frame_mask.reshape(
            batch_size, self.temporal_tokens, self.config.temporal_tubelet
        ).sum(dim=-1).long()
        coverage_embedding = self.tubelet_coverage_embedding(coverage)
        # 把每个时间 tubelet 展开为独立的 16-token 空间序列。
        spatial = embedded.permute(0, 2, 3, 4, 1).reshape(
            batch_size * self.temporal_tokens,
            self.spatial_tokens,
            self.config.embedding_dim,
        )
        spatial = self.embedding_dropout(
            spatial
            + self.spatial_position
            + coverage_embedding.reshape(
                batch_size * self.temporal_tokens, 1, self.config.embedding_dim
            )
        )
        for block in self.spatial_blocks:
            spatial = block(spatial)
        # 空间平均池化后，每个 tubelet 只保留一个时间 token。
        frame_tokens = spatial.mean(dim=1).reshape(
            batch_size, self.temporal_tokens, self.config.embedding_dim
        )
        temporal = torch.cat(
            [self.temporal_class_token.expand(batch_size, -1, -1), frame_tokens], dim=1
        )
        temporal = self.embedding_dropout(temporal + self.temporal_position)
        # CLS 永不屏蔽；coverage=0 的 tubelet 不参与时间注意力。
        padding_mask = torch.cat([
            torch.zeros((batch_size, 1), dtype=torch.bool, device=videos.device),
            coverage.eq(0),
        ], dim=1)
        for block in self.temporal_blocks:
            temporal = block(temporal, key_padding_mask=padding_mask)
        # 归一化后的 CLS 表示交给单一回归头产生因子原始分数。
        return self.final_norm(temporal[:, 0])

    def forward(self, videos: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        return self.regression_head(self.encode(videos, frame_mask)).squeeze(-1)


def pearson_correlation(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    # 手工中心化以保留梯度，并用下限避免常数向量造成除零。
    left = left.reshape(-1).float()
    right = right.reshape(-1).float()
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("correlation needs equal vectors with at least two values")
    left = left - left.mean()
    right = right - right.mean()
    denominator = torch.sqrt(left.square().sum() * right.square().sum()).clamp_min(1e-8)
    return left.mul(right).sum() / denominator


def sigmoid_soft_rank(values: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    if values.ndim != 1 or len(values) < 2 or temperature <= 0:
        raise ValueError("soft rank needs a vector of at least two values and temperature > 0")
    # 对所有两两差值应用 sigmoid，近似“有多少元素排在当前元素之前”。
    pairwise = (values.unsqueeze(0) - values.unsqueeze(1)) / temperature
    return torch.sigmoid(pairwise).sum(dim=1) + 0.5


def sigmoid_rank_ic_loss(
    predictions: torch.Tensor, targets: torch.Tensor, temperature: float = 0.1
) -> torch.Tensor:
    # 预测排序需要梯度；标签排序固定，因此关闭其梯度记录。
    prediction_ranks = sigmoid_soft_rank(predictions.float(), temperature)
    with torch.no_grad():
        target_ranks = sigmoid_soft_rank(targets.float(), temperature)
    # 最大化软排序 Pearson IC 等价于最小化其相反数。
    return -pearson_correlation(prediction_ranks, target_ranks)
