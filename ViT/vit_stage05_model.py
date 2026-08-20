"""阶段 05：定义单日 17×8×8 订单流图像的 Vision Transformer。"""
from __future__ import annotations


from dataclasses import dataclass

import torch
from torch import nn


# 使用 `dataclass(frozen=True)` 包装 `VitModelConfig`，并定义其可复用的处理逻辑。
@dataclass(frozen=True)  # 自动生成配置对象的初始化与比较方法。
class VitModelConfig:  # ViT 结构参数。
    """ViT 结构参数。"""

    input_channels: int = 17  # 接收输入channels。
    image_size: int = 8  # 接收图像大小。
    patch_size: int = 2  # 接收patch大小。
    embedding_dim: int = 128  # 接收嵌入dim。
    encoder_layers: int = 4  # 接收encoderlayers。
    attention_heads: int = 4  # 接收注意力heads。
    mlp_hidden_dim: int = 512  # 接收mlp隐藏层dim。
    dropout: float = 0.1  # 接收dropout。
    output_dim: int = 1



class PatchEmbedding(nn.Module):  # 把 8×8 图像切成不重叠空间 patch 并映射到 token。
    """把 8×8 图像切成不重叠空间 patch 并映射到 token。"""

    def __init__(self, config: VitModelConfig) -> None:  # 初始化当前对象的配置、状态与子组件。
        super().__init__()
        self.config = config  # 保存配置，供该对象后续步骤复用。
        self.projection = nn.Conv2d(  # 用卷积完成 Patch Embedding。
            config.input_channels,
            config.embedding_dim,
            kernel_size=config.patch_size,  # 配置kernel大小。
            stride=config.patch_size,  # 配置stride。
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:  # [B,17,8,8] -> [B,16,D]。
        """输入 [B,C,H,W]，输出 [B,P,D]。"""

        expected = (
            self.config.input_channels,
            self.config.image_size,
            self.config.image_size,
        )
        if images.ndim != 4 or tuple(images.shape[1:]) != expected:  # 检查张量或数组形状是否符合接口约定。
            raise ValueError(  # 输入数据不满足业务或数值约束时立即终止。
                f"模型输入形状应为 [B,{expected}]，实际 {tuple(images.shape)}"
            )
        return self.projection(images).flatten(start_dim=2).transpose(1, 2)  # [B,D,4,4] -> [B,D,16] -> [B,16,D]。


class TransformerEncoderBlock(nn.Module):  # 实现 Pre-LayerNorm Transformer 编码块。
    """Pre-LayerNorm Transformer Encoder。"""

    def __init__(self, config: VitModelConfig) -> None:  # 初始化当前对象的配置、状态与子组件。
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.embedding_dim)  # Pre-LayerNorm。
        self.attention = nn.MultiheadAttention(  # 多头自注意力。
            config.embedding_dim,
            config.attention_heads,
            dropout=config.dropout,  # 配置dropout。
            batch_first=True,  # 配置批次first。
        )
        self.attention_dropout = nn.Dropout(config.dropout)  # 训练期 dropout。
        self.mlp_norm = nn.LayerNorm(config.embedding_dim)  # Pre-LayerNorm。
        self.mlp = nn.Sequential(  # 按顺序组成网络子层。
            nn.Linear(config.embedding_dim, config.mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_hidden_dim, config.embedding_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:  # [B,T,D] -> [B,T,D]。
        """保持 token 张量 [B,T,D] 的形状不变。"""

        normalized = self.attention_norm(tokens)  # 通过当前网络子模块。
        attention_output, _ = self.attention(  # 通过当前网络子模块。
            normalized, normalized, normalized, need_weights=False
        )
        tokens = tokens + self.attention_dropout(attention_output)
        return tokens + self.mlp(self.mlp_norm(tokens))  # 返回保持 token 张量 [B,T,D] 的形状不变。


class OrderFlowVisionTransformer(nn.Module):  # 从单日订单流图像产生一个截面预测分数。
    """从单日订单流图像产生一个截面预测分数。"""

    def __init__(self, config: VitModelConfig) -> None:  # 初始化当前对象的配置、状态与子组件。
        super().__init__()
        self.config = config  # 保存配置，供该对象后续步骤复用。
        patch_count = (config.image_size // config.patch_size) ** 2  # 计算patch数量。
        self.patch_embedding = PatchEmbedding(config)  # 建立 Patch Embedding。
        self.class_token = nn.Parameter(  # 注册可学习参数。
            torch.zeros(1, 1, config.embedding_dim)
        )
        self.position_embedding = nn.Parameter(  # 注册可学习参数。
            torch.zeros(1, patch_count + 1, config.embedding_dim)
        )
        self.embedding_dropout = nn.Dropout(config.dropout)  # 训练期 dropout。
        self.blocks = nn.ModuleList(  # 注册独立参数的编码块。
            [
                TransformerEncoderBlock(config)
                for _ in range(config.encoder_layers)  # 按给定步长遍历样本区间，更新。
            ]
        )
        self.final_norm = nn.LayerNorm(config.embedding_dim)  # Pre-LayerNorm。
        self.head = nn.Linear(config.embedding_dim, config.output_dim)  # 线性投影到目标维度。
        self._initialize_weights()  # 初始化本分支参数。

    def _initialize_weights(self) -> None:  # 使用与迁移前完全一致的参数初始化。
        """使用与迁移前完全一致的参数初始化。"""

        nn.init.trunc_normal_(self.class_token, std=0.02)  # 截断正态初始化。
        nn.init.trunc_normal_(self.position_embedding, std=0.02)  # 截断正态初始化。
        for module in self.modules():  # 遍历modules，逐项处理module。
            if isinstance(module, nn.Linear):  # 检查对象类型是否符合接口约定。
                nn.init.trunc_normal_(module.weight, std=0.02)  # 截断正态初始化。
                if module.bias is not None:  # 检查必需数据、配置或校验结果是否缺失。
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):  # 检查上一条件不成立时的备选分支。
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:  # [B,17,8,8] -> [B]。
        """输入 [B,17,8,8]，返回 raw score [B]。"""

        patches = self.patch_embedding(images)  # [B,17,8,8] -> [B,16,D]。
        class_tokens = self.class_token.expand(len(images), -1, -1)  # [1,1,D] -> [B,1,D]。
        tokens = torch.cat([class_tokens, patches], dim=1)  # [B,1,D] + [B,16,D] -> [B,17,D]。
        tokens = self.embedding_dropout(tokens + self.position_embedding)  # 通过当前网络子模块。
        for block in self.blocks:  # 遍历blocks，逐项处理block。
            tokens = block(tokens)  # 依次通过 Transformer block。
        return self.head(self.final_norm(tokens[:, 0])).squeeze(-1)  # [B,1] -> [B]。
