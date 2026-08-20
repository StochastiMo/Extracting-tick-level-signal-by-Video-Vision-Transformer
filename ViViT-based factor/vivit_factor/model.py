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
        self.attention_norm = nn.LayerNorm(config.embedding_dim)  # 注意力前先稳定每个 token 的特征分布。
        self.attention = nn.MultiheadAttention(  # 在同一空间或时间序列内建模 token 关系。
            config.embedding_dim, config.attention_heads,
            dropout=config.dropout, batch_first=True,
        )
        self.attention_dropout = nn.Dropout(config.dropout)  # 对注意力残差分支做正则化。
        self.mlp_norm = nn.LayerNorm(config.embedding_dim)  # MLP 前使用独立归一化参数。
        self.mlp = nn.Sequential(  # 两层前馈网络先扩维再还原嵌入维度。
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
        normalized = self.attention_norm(tokens)  # Pre-LN 输入同时作为 Q、K、V。
        attended, _ = self.attention(  # 不返回注意力权重以减少显存占用。
            normalized, normalized, normalized,
            key_padding_mask=key_padding_mask, need_weights=False,
        )
        tokens = tokens + self.attention_dropout(attended)  # 第一条残差保留原始 token 信息。
        # MLP 子层同样采用 Pre-LN 和残差，输出形状保持不变。
        return tokens + self.mlp(self.mlp_norm(tokens))  # 第二条残差融合非线性前馈结果。


class OrderFlowViViT(nn.Module):
    """把形状为 [B,17,20,8,8] 的视频编码为每只股票一个分数。"""

    def __init__(self, config: ViViTConfig = ViViTConfig()) -> None:
        super().__init__()
        self.config = config  # 保存固定结构参数供输入检查和 reshape 使用。
        # 20 帧按 2 帧 tubelet 压缩为 10 个时间 token。
        self.temporal_tokens = config.frame_count // config.temporal_tubelet  # 计算时间压缩后的 token 数。
        patches_per_side = config.image_size // config.spatial_patch  # 计算单边 patch 数。
        # 每帧 8×8 按 2×2 patch 划分，共得到 16 个空间 token。
        self.spatial_tokens = patches_per_side**2  # 二维 patch 网格展平后的 token 数。
        # Conv3d 同时完成时间 tubelet 与空间 patch 的线性嵌入。
        self.tubelet_embedding = nn.Conv3d(  # 卷积核和步长相同，得到互不重叠 tubelet。
            config.input_channels, config.embedding_dim,
            kernel_size=(config.temporal_tubelet, config.spatial_patch, config.spatial_patch),
            stride=(config.temporal_tubelet, config.spatial_patch, config.spatial_patch),
        )
        self.tubelet_coverage_embedding = nn.Embedding(  # 编码每个 tubelet 含 0、1、2 个有效帧。
            config.tubelet_coverage_states, config.embedding_dim
        )
        # 空间位置编码在所有时间 tubelet 之间共享。
        self.spatial_position = nn.Parameter(  # 学习 4×4 patch 网格的绝对位置。
            torch.zeros(1, self.spatial_tokens, config.embedding_dim)
        )
        self.spatial_blocks = nn.ModuleList(  # 每个时间 tubelet 共用同一组空间编码器。
            TransformerEncoderBlock(config) for _ in range(config.spatial_encoder_layers)
        )
        self.temporal_class_token = nn.Parameter(torch.zeros(1, 1, config.embedding_dim))  # 聚合整段视频的可学习 CLS。
        # 时间位置编码包含额外的 CLS 位置。
        self.temporal_position = nn.Parameter(  # 区分 CLS 与十个时间 tubelet 的顺序。
            torch.zeros(1, self.temporal_tokens + 1, config.embedding_dim)
        )
        self.embedding_dropout = nn.Dropout(config.dropout)  # 对位置与内容相加后的嵌入做正则化。
        self.temporal_blocks = nn.ModuleList(  # 在 tubelet 序列上编码跨日关系。
            TransformerEncoderBlock(config) for _ in range(config.temporal_encoder_layers)
        )
        self.final_norm = nn.LayerNorm(config.embedding_dim)  # 输出头前归一化 CLS 表示。
        self.regression_head = nn.Linear(config.embedding_dim, config.output_dim)  # 将 CLS 映射为单个股票分数。
        self._initialize_weights()  # 使用统一规则初始化新建参数。

    def _initialize_weights(self) -> None:
        # 位置向量、CLS 和线性层沿用 Transformer 常见的截断正态初始化。
        nn.init.trunc_normal_(self.spatial_position, std=0.02)  # 初始化空间位置编码。
        nn.init.trunc_normal_(self.temporal_position, std=0.02)  # 初始化时间位置编码。
        nn.init.trunc_normal_(self.temporal_class_token, std=0.02)  # 初始化聚合 token。
        nn.init.zeros_(self.tubelet_coverage_embedding.weight)  # 全缺失状态保持零扰动。
        # coverage=0 代表全缺失 tubelet，保持零向量；其余状态学习独立表示。
        nn.init.trunc_normal_(self.tubelet_coverage_embedding.weight[1:2], std=0.02)  # 为部分覆盖状态提供可学习偏移。
        for module in self.modules():  # 遍历所有子模块统一初始化线性层和归一化层。
            if isinstance(module, nn.Linear):  # Transformer 与回归头均采用同一权重尺度。
                nn.init.trunc_normal_(module.weight, std=0.02)  # 避免初始激活过大。
                if module.bias is not None:  # 仅处理实际存在的偏置。
                    nn.init.zeros_(module.bias)  # 线性偏置从零开始。
            elif isinstance(module, nn.LayerNorm):  # LayerNorm 初始保持恒等变换。
                nn.init.ones_(module.weight)  # 初始缩放设为一。
                nn.init.zeros_(module.bias)  # 初始平移设为零。

    def encode(self, videos: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        # 固定输入契约能尽早暴露通道、时间或空间维度错位。
        if videos.ndim != 5 or videos.shape[1:] != (  # 检查除批大小外的全部固定维度。
            self.config.input_channels, self.config.frame_count,
            self.config.image_size, self.config.image_size,
        ):
            raise ValueError(f"unexpected video shape: {tuple(videos.shape)}")  # 报出实际张量形状便于定位数据错位。
        if frame_mask.shape != (len(videos), self.config.frame_count):  # 掩码必须逐样本逐帧对齐。
            raise ValueError(f"unexpected frame mask shape: {tuple(frame_mask.shape)}")  # 拒绝无法广播的掩码。
        frame_mask = frame_mask.to(device=videos.device, dtype=torch.bool)  # 将掩码移到同设备并固定布尔类型。
        # T 日必须存在；历史帧允许缺失并由后续 padding mask 处理。
        torch._assert(frame_mask[:, -1].all(), "target frame must be valid")  # 保证每个预测样本都有 T 日图片。

        # [B,C,T,H,W] → [B,D,T',H',W']。
        embedded = self.tubelet_embedding(videos)  # 一次生成时间与空间 tubelet 嵌入。
        batch_size = embedded.shape[0]  # 保留批大小供后续展平和恢复。
        # 每两个原始帧统计有效帧数，状态仅可能为 0、1、2。
        coverage = frame_mask.reshape(  # 将原始帧掩码按每两个帧分组。
            batch_size, self.temporal_tokens, self.config.temporal_tubelet
        ).sum(dim=-1).long()
        coverage_embedding = self.tubelet_coverage_embedding(coverage)  # 查表得到缺帧程度嵌入。
        # 把每个时间 tubelet 展开为独立的 16-token 空间序列。
        spatial = embedded.permute(0, 2, 3, 4, 1).reshape(  # 把嵌入维移到末尾并合并批次与时间。
            batch_size * self.temporal_tokens,
            self.spatial_tokens,
            self.config.embedding_dim,
        )
        spatial = self.embedding_dropout(  # 同时加入空间位置和帧覆盖状态。
            spatial
            + self.spatial_position
            + coverage_embedding.reshape(
                batch_size * self.temporal_tokens, 1, self.config.embedding_dim
            )
        )
        for block in self.spatial_blocks:  # 顺序通过所有空间 Transformer 层。
            spatial = block(spatial)  # 每层保持 B×T'×16 序列形状。
        # 空间平均池化后，每个 tubelet 只保留一个时间 token。
        frame_tokens = spatial.mean(dim=1).reshape(  # 对 16 个空间 token 等权池化并恢复批次维。
            batch_size, self.temporal_tokens, self.config.embedding_dim
        )
        temporal = torch.cat(  # 在时间序列最前方加入 CLS。
            [self.temporal_class_token.expand(batch_size, -1, -1), frame_tokens], dim=1
        )
        temporal = self.embedding_dropout(temporal + self.temporal_position)  # 注入时间顺序并做 dropout。
        # CLS 永不屏蔽；coverage=0 的 tubelet 不参与时间注意力。
        padding_mask = torch.cat([  # 构造 CLS 加十个 tubelet 的屏蔽矩阵。
            torch.zeros((batch_size, 1), dtype=torch.bool, device=videos.device),
            coverage.eq(0),
        ], dim=1)
        for block in self.temporal_blocks:  # 顺序编码跨 tubelet 时间依赖。
            temporal = block(temporal, key_padding_mask=padding_mask)  # 全缺失 tubelet 不作为注意力键值。
        # 归一化后的 CLS 表示交给单一回归头产生因子原始分数。
        return self.final_norm(temporal[:, 0])  # 只取 CLS 作为整段视频表示。

    def forward(self, videos: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        return self.regression_head(self.encode(videos, frame_mask)).squeeze(-1)  # 输出形状固定为每只股票一个标量。


def pearson_correlation(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    # 手工中心化以保留梯度，并用下限避免常数向量造成除零。
    left = left.reshape(-1).float()  # 展平预测并用 FP32 计算统计量。
    right = right.reshape(-1).float()  # 展平标签以匹配预测。
    if len(left) != len(right) or len(left) < 2:  # 相关系数要求等长且至少两个观测。
        raise ValueError("correlation needs equal vectors with at least two values")  # 拒绝无定义输入。
    left = left - left.mean()  # 对预测中心化。
    right = right - right.mean()  # 对标签中心化。
    denominator = torch.sqrt(left.square().sum() * right.square().sum()).clamp_min(1e-8)  # 计算协方差归一化尺度并防除零。
    return left.mul(right).sum() / denominator  # 返回保持预测梯度的 Pearson 相关。


def sigmoid_soft_rank(values: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    if values.ndim != 1 or len(values) < 2 or temperature <= 0:  # 软排序只接受有效一维横截面。
        raise ValueError("soft rank needs a vector of at least two values and temperature > 0")  # 阻止无定义温度或样本。
    # 对所有两两差值应用 sigmoid，近似“有多少元素排在当前元素之前”。
    pairwise = (values.unsqueeze(0) - values.unsqueeze(1)) / temperature  # 构造所有样本两两分数差。
    return torch.sigmoid(pairwise).sum(dim=1) + 0.5  # 汇总平滑胜负关系得到可微秩。


def sigmoid_rank_ic_loss(
    predictions: torch.Tensor, targets: torch.Tensor, temperature: float = 0.1
) -> torch.Tensor:
    # 预测排序需要梯度；标签排序固定，因此关闭其梯度记录。
    prediction_ranks = sigmoid_soft_rank(predictions.float(), temperature)  # 预测软秩保留反向梯度。
    with torch.no_grad():  # 标签排序不属于待优化计算图。
        target_ranks = sigmoid_soft_rank(targets.float(), temperature)  # 计算固定目标软秩。
    # 最大化软排序 Pearson IC 等价于最小化其相反数。
    return -pearson_correlation(prediction_ranks, target_ranks)  # 取负号把最大化 RankIC 转为最小化损失。
