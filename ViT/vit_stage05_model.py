"""阶段 05：定义单日 17×8×8 订单流图像的 Vision Transformer。"""
from __future__ import annotations


from dataclasses import dataclass

import torch
from torch import nn


# 使用 `dataclass(frozen=True)` 包装 `VitModelConfig`，并定义其可复用的处理逻辑。
@dataclass(frozen=True)
class VitModelConfig:
    """ViT 结构参数。"""

    # 设置 `input_channels` 的固定配置值，供后续流程统一使用。
    input_channels: int = 17
    # 设置 `image_size` 的固定配置值，供后续流程统一使用。
    image_size: int = 8
    # 设置 `patch_size` 的固定配置值，供后续流程统一使用。
    patch_size: int = 2
    # 设置 `embedding_dim` 的固定配置值，供后续流程统一使用。
    embedding_dim: int = 128
    # 设置 `encoder_layers` 的固定配置值，供后续流程统一使用。
    encoder_layers: int = 4
    # 设置 `attention_heads` 的固定配置值，供后续流程统一使用。
    attention_heads: int = 4
    # 设置 `mlp_hidden_dim` 的固定配置值，供后续流程统一使用。
    mlp_hidden_dim: int = 512
    # 设置 `dropout` 的固定配置值，供后续流程统一使用。
    dropout: float = 0.1
    # 设置 `output_dim` 的固定配置值，供后续流程统一使用。
    output_dim: int = 1


# region TR-COMMON-VIT-MODEL
# 定义 `PatchEmbedding` 类，集中封装该阶段的数据结构与相关行为。
class PatchEmbedding(nn.Module):
    """把 8×8 图像切成不重叠空间 patch 并映射到 token。"""

    # 定义初始化方法，接收配置并建立实例后续计算所需的内部状态。
    def __init__(self, config: VitModelConfig) -> None:
        """执行 __init__ 对应的处理步骤。

        输入：
            config: VitModelConfig。
        输出：
            None。
        """
        # 初始化父类状态，确保继承链中的基础组件正确注册。
        super().__init__()
        # 读取 `config` 的当前值，并赋给 `self.config`。
        self.config = config
        # 调用 `nn.Conv2d` 创建二维卷积并按 patch 大小切分图像，并将结果保存到 `self.projection`。
        self.projection = nn.Conv2d(
            config.input_channels,
            config.embedding_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

    # 定义 `forward` 函数，封装这一阶段可独立调用的处理步骤。
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """输入 [B,C,H,W]，输出 [B,P,D]。

        输入：
            images: torch.Tensor。
        输出：
            torch.Tensor。
        """

        # 组织一组有序值，并保存到 `expected`。
        expected = (
            self.config.input_channels,
            self.config.image_size,
            self.config.image_size,
        )
        # 检查 `images.ndim != 4 or tuple(images.shape[1:]) != expected`；根据条件结果选择对应处理分支。
        if images.ndim != 4 or tuple(images.shape[1:]) != expected:
            # 抛出 `ValueError(f'模型输入形状应为 [B,{expected}]，实际 {tuple(images.shape)}')`，阻止不满足数据契约的流程继续执行。
            raise ValueError(
                f"模型输入形状应为 [B,{expected}]，实际 {tuple(images.shape)}"
            )
        # 将 `self.projection(images).flatten(start_dim=2).transpose(1, 2)` 作为当前函数结果返回给调用方。
        return self.projection(images).flatten(start_dim=2).transpose(1, 2)


# 定义 `TransformerEncoderBlock` 类，集中封装该阶段的数据结构与相关行为。
class TransformerEncoderBlock(nn.Module):
    """Pre-LayerNorm Transformer Encoder。"""

    # 定义初始化方法，接收配置并建立实例后续计算所需的内部状态。
    def __init__(self, config: VitModelConfig) -> None:
        """执行 __init__ 对应的处理步骤。

        输入：
            config: VitModelConfig。
        输出：
            None。
        """
        # 初始化父类状态，确保继承链中的基础组件正确注册。
        super().__init__()
        # 调用 `nn.LayerNorm` 创建层归一化组件以稳定 token 特征分布，并将结果保存到 `self.attention_norm`。
        self.attention_norm = nn.LayerNorm(config.embedding_dim)
        # 调用 `nn.MultiheadAttention` 创建多头自注意力层以建模 token 间关系，并将结果保存到 `self.attention`。
        self.attention = nn.MultiheadAttention(
            config.embedding_dim,
            config.attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        # 调用 `nn.Dropout` 创建随机失活层以抑制训练过拟合，并将结果保存到 `self.attention_dropout`。
        self.attention_dropout = nn.Dropout(config.dropout)
        # 调用 `nn.LayerNorm` 创建层归一化组件以稳定 token 特征分布，并将结果保存到 `self.mlp_norm`。
        self.mlp_norm = nn.LayerNorm(config.embedding_dim)
        # 调用 `nn.Sequential` 按执行顺序组合多个神经网络子层，并将结果保存到 `self.mlp`。
        self.mlp = nn.Sequential(
            nn.Linear(config.embedding_dim, config.mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_hidden_dim, config.embedding_dim),
            nn.Dropout(config.dropout),
        )

    # 定义 `forward` 函数，封装这一阶段可独立调用的处理步骤。
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """保持 token 张量 [B,T,D] 的形状不变。

        输入：
            tokens: torch.Tensor。
        输出：
            torch.Tensor。
        """

        # 调用 `self.attention_norm` 通过当前网络子模块变换输入表征，并将结果保存到 `normalized`。
        normalized = self.attention_norm(tokens)
        # 调用 `self.attention` 通过当前网络子模块变换输入表征，并将结果保存到 `(attention_output, _)`。
        attention_output, _ = self.attention(
            normalized, normalized, normalized, need_weights=False
        )
        # 计算 `tokens + self.attention_dropout(attention_output)`，并将数值结果保存到 `tokens`。
        tokens = tokens + self.attention_dropout(attention_output)
        # 将 `tokens + self.mlp(self.mlp_norm(tokens))` 作为当前函数结果返回给调用方。
        return tokens + self.mlp(self.mlp_norm(tokens))


# 定义 `OrderFlowVisionTransformer` 类，集中封装该阶段的数据结构与相关行为。
class OrderFlowVisionTransformer(nn.Module):
    """从单日订单流图像产生一个截面预测分数。"""

    # 定义初始化方法，接收配置并建立实例后续计算所需的内部状态。
    def __init__(self, config: VitModelConfig) -> None:
        """执行 __init__ 对应的处理步骤。

        输入：
            config: VitModelConfig。
        输出：
            None。
        """
        # 初始化父类状态，确保继承链中的基础组件正确注册。
        super().__init__()
        # 读取 `config` 的当前值，并赋给 `self.config`。
        self.config = config
        # 计算 `(config.image_size // config.patch_size) ** 2`，并将数值结果保存到 `patch_count`。
        patch_count = (config.image_size // config.patch_size) ** 2
        # 调用 `PatchEmbedding` 创建图像 patch 到 token 的嵌入模块，并将结果保存到 `self.patch_embedding`。
        self.patch_embedding = PatchEmbedding(config)
        # 调用 `nn.Parameter` 把张量注册为可训练模型参数，并将结果保存到 `self.class_token`。
        self.class_token = nn.Parameter(
            torch.zeros(1, 1, config.embedding_dim)
        )
        # 调用 `nn.Parameter` 把张量注册为可训练模型参数，并将结果保存到 `self.position_embedding`。
        self.position_embedding = nn.Parameter(
            torch.zeros(1, patch_count + 1, config.embedding_dim)
        )
        # 调用 `nn.Dropout` 创建随机失活层以抑制训练过拟合，并将结果保存到 `self.embedding_dropout`。
        self.embedding_dropout = nn.Dropout(config.dropout)
        # 调用 `nn.ModuleList` 注册一组可迭代的神经网络子模块，并将结果保存到 `self.blocks`。
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(config)
                for _ in range(config.encoder_layers)
            ]
        )
        # 调用 `nn.LayerNorm` 创建层归一化组件以稳定 token 特征分布，并将结果保存到 `self.final_norm`。
        self.final_norm = nn.LayerNorm(config.embedding_dim)
        # 调用 `nn.Linear` 创建线性投影层以变换特征维度，并将结果保存到 `self.head`。
        self.head = nn.Linear(config.embedding_dim, config.output_dim)
        # 调用 `self._initialize_weights` 按模型配置初始化当前分支的可训练参数。
        self._initialize_weights()

    # 定义内部辅助函数 `_initialize_weights`，封装仅供本模块复用的计算步骤。
    def _initialize_weights(self) -> None:
        """使用与迁移前完全一致的参数初始化。

        输入：
            无显式输入参数。
        输出：
            None。
        """

        # 调用 `nn.init.trunc_normal_` 用截断正态分布初始化可训练参数。
        nn.init.trunc_normal_(self.class_token, std=0.02)
        # 调用 `nn.init.trunc_normal_` 用截断正态分布初始化可训练参数。
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        # 遍历 `self.modules()`，并把当前元素绑定到 `module` 逐项处理。
        for module in self.modules():
            # 检查 `isinstance(module, nn.Linear)`；根据条件结果选择对应处理分支。
            if isinstance(module, nn.Linear):
                # 调用 `nn.init.trunc_normal_` 用截断正态分布初始化可训练参数。
                nn.init.trunc_normal_(module.weight, std=0.02)
                # 检查 `module.bias is not None`；根据条件结果选择对应处理分支。
                if module.bias is not None:
                    # 调用 `nn.init.zeros_` 将指定参数初始化为零。
                    nn.init.zeros_(module.bias)
            # 检查 `isinstance(module, nn.LayerNorm)`；根据条件结果选择对应处理分支。
            elif isinstance(module, nn.LayerNorm):
                # 调用 `nn.init.ones_` 执行 `ones_` 对应的明确变换，并产生后续步骤使用的结果。
                nn.init.ones_(module.weight)
                # 调用 `nn.init.zeros_` 将指定参数初始化为零。
                nn.init.zeros_(module.bias)

    # 定义 `forward` 函数，封装这一阶段可独立调用的处理步骤。
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """输入 [B,17,8,8]，返回 raw score [B]。

        输入：
            images: torch.Tensor。
        输出：
            torch.Tensor。
        """

        # 调用 `self.patch_embedding` 把输入图像切成 patch 并投影为 token，并将结果保存到 `patches`。
        patches = self.patch_embedding(images)
        # 调用 `self.class_token.expand` 沿 batch 维广播共享 CLS token，并将结果保存到 `class_tokens`。
        class_tokens = self.class_token.expand(len(images), -1, -1)
        # 调用 `torch.cat` 拼接各批结果，并将合并值保存到 `tokens`。
        tokens = torch.cat([class_tokens, patches], dim=1)
        # 调用 `self.embedding_dropout` 通过当前网络子模块变换输入表征，并将结果保存到 `tokens`。
        tokens = self.embedding_dropout(tokens + self.position_embedding)
        # 遍历 `self.blocks`，并把当前元素绑定到 `block` 逐项处理。
        for block in self.blocks:
            # 调用 `block` 让 token 依次通过当前 Transformer 编码块，并将结果保存到 `tokens`。
            tokens = block(tokens)
        # 将 `self.head(self.final_norm(tokens[:, 0])).squeeze(-1)` 作为当前函数结果返回给调用方。
        return self.head(self.final_norm(tokens[:, 0])).squeeze(-1)
# endregion
