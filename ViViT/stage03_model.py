"""ViViT 空间—时间编码器及有效帧 mask 处理。"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


# 使用 `dataclass(frozen=True)` 包装 `VivitModelConfig`，并定义其可复用的处理逻辑。
@dataclass(frozen=True)  # frozen=True 防止模型创建后意外修改配置。
class VivitModelConfig:  # 定义 ViViT 模型全部结构超参数的数据类。
    """ViViT 输入、tubelet 和 Transformer 结构参数。
    """

    input_channels: int = 17  # 每帧输入图像有 17 个订单流特征通道。
    frame_count: int = 20  # 每个样本包含过去 20 个交易日的图像。
    image_size: int = 8  # 每帧图像的高和宽均为 8。
    temporal_tubelet: int = 2  # 每个 tubelet 在时间维合并连续 2 帧。
    spatial_patch: int = 2  # 每个 tubelet 在空间维覆盖 2×2 像素。
    embedding_dim: int = 128  # 每个 token 使用 128 维向量表示。
    spatial_encoder_layers: int = 2  # 空间 Transformer 堆叠 2 个 block。
    temporal_encoder_layers: int = 2  # 时间 Transformer 堆叠 2 个 block。
    attention_heads: int = 4  # 每个多头注意力层使用 4 个注意力头。
    mlp_hidden_dim: int = 512  # 每个 Encoder 内部 MLP 的隐藏维度为 512。
    dropout: float = 0.1  # Attention、MLP 和 embedding 的丢弃概率。
    output_dim: int = 1  # 最终为每只股票输出 1 个原始预测分数。
    tubelet_coverage_states: int = 3  # coverage 有 0、1、2 三种有效帧数状态。


class TransformerEncoderBlock(nn.Module):  # 定义可同时复用于空间和时间编码的 block。
    """支持 padding mask 的 Pre-LayerNorm Transformer block。
    """

    def __init__(self, config: VivitModelConfig) -> None:  # 根据统一配置创建一个 block。
        super().__init__()  # 初始化 PyTorch nn.Module 的内部状态。
        self.attention_norm = nn.LayerNorm(  # Attention 前先做 Pre-LayerNorm。
            config.embedding_dim  # 对每个 token 的 128 个特征进行标准化。
        )
        self.attention = nn.MultiheadAttention(  # 创建多头自注意力模块。
            config.embedding_dim,  # 指定输入和输出 token 的特征维度。
            config.attention_heads,  # 将特征拆分给 4 个注意力头。
            dropout=config.dropout,  # 对注意力权重应用 dropout。
            batch_first=True,  # 约定输入形状为 [批量, token 数, 特征维]。
        )
        self.attention_dropout = nn.Dropout(  # 创建注意力输出后的 dropout。
            config.dropout  # 使用配置中的 0.1 丢弃概率。
        )
        self.mlp_norm = nn.LayerNorm(  # 内部 MLP 前再做一次 Pre-LayerNorm。
            config.embedding_dim  # 标准化每个 token 的特征维。
        )
        self.mlp = nn.Sequential(  # 定义当前 Encoder block 内部的 FFN/MLP。
            nn.Linear(  # 第一层把 token 特征升维以增强表达能力。
                config.embedding_dim, config.mlp_hidden_dim  # 形状从 128 映射到 512。
            ),
            nn.GELU(),  # 使用平滑非线性激活函数，避免 MLP 退化为线性映射。
            nn.Dropout(config.dropout),  # 对升维后的隐藏特征做 dropout。
            nn.Linear(  # 第二层把特征投影回残差连接所需的维度。
                config.mlp_hidden_dim, config.embedding_dim  # 形状从 512 映射回 128。
            ),
            nn.Dropout(config.dropout),  # 对 MLP 的最终输出做 dropout。
        )

    def forward(  # 定义一个 Transformer Encoder block 的前向计算。
        self,  # 当前 block 实例。
        tokens: torch.Tensor,  # 输入 token，形状为 [B,T,D]。
        key_padding_mask: torch.Tensor | None = None,  # True 表示相应 token 不作 key/value。
    ) -> torch.Tensor:  # 返回形状仍为 [B,T,D] 的编码结果。
        """输入和输出均为 [B,T,D]；True mask 位置不充当 key/value。"""

        normalized = self.attention_norm(tokens)  # 在自注意力之前标准化 token。
        attended, _ = self.attention(  # 计算自注意力，只保留新的 token 表征。
            normalized,  # Query：每个 token 要查询什么信息。
            normalized,  # Key：每个 token 可按什么特征被匹配。
            normalized,  # Value：匹配后真正被聚合的 token 信息。
            key_padding_mask=key_padding_mask,  # 屏蔽 coverage=0 的无效时间 token。
            need_weights=False,  # 训练只需输出表征，不保存注意力权重以节省显存。
        )
        tokens = tokens + self.attention_dropout(attended)  # 将注意力输出残差加回原 token。
        return tokens + self.mlp(self.mlp_norm(tokens))  # 将内部 MLP 输出再次残差相加。


# region TR-COMMON-VIVIT-MODEL  # 标记可被其他脚本同步使用的公共 ViViT 模型区域。
class OrderFlowViViT(nn.Module):  # 定义订单流视频的完整 ViViT 回归模型。
    """把 20 日订单流视频编码为 128 维 CLS 表征和一个 raw score。
    """

    def __init__(self, config: VivitModelConfig) -> None:  # 按配置搭建全部模型层。
        super().__init__()  # 初始化 PyTorch nn.Module 的内部状态。
        self.config = config  # 保存配置，供 encode 阶段读取输入尺寸等信息。
        self.temporal_tokens = (  # 计算 Tubelet Embedding 后的时间 token 数。
            config.frame_count // config.temporal_tubelet  # 20 帧每 2 帧一组，得到 10 组。
        )
        patches_per_side = (  # 计算单帧每条空间边可以切出多少 patch。
            config.image_size // config.spatial_patch  # 8÷2=4，即每边得到 4 个 patch。
        )
        self.spatial_tokens = patches_per_side**2  # 4×4=16，即每个时间 tubelet 有 16 个空间 token。

  # 2×2×2 Conv3d 同时进行两帧聚合和 2×2 空间切块。
        self.tubelet_embedding = nn.Conv3d(  # 用三维卷积完成切块并映射到 token embedding。
            config.input_channels,  # 输入通道数为 17。
            config.embedding_dim,  # 输出通道数为 128，即每个 token 的维度。
            kernel_size=(  # 卷积核大小同时覆盖时间、高度和宽度。
                config.temporal_tubelet,  # 时间核大小为 2 帧。
                config.spatial_patch,  # 高度核大小为 2 像素。
                config.spatial_patch,  # 宽度核大小为 2 像素。
            ),
            stride=(  # 步长等于卷积核大小，因此 tubelet 之间不重叠。
                config.temporal_tubelet,  # 时间维每次前进 2 帧。
                config.spatial_patch,  # 高度维每次前进 2 像素。
                config.spatial_patch,  # 宽度维每次前进 2 像素。
            ),
        )
        self.tubelet_coverage_embedding = nn.Embedding(  # 学习 tubelet 有效帧数量的附加向量。
            config.tubelet_coverage_states,  # 一共有 coverage=0、1、2 三种索引。
            config.embedding_dim,  # 每种 coverage 状态也映射成 128 维向量。
        )
        self.spatial_position = nn.Parameter(  # 创建可学习的空间位置编码。
            torch.zeros(  # 先以全零张量创建，稍后再进行截断正态初始化。
                1, self.spatial_tokens, config.embedding_dim  # 形状为 [1,16,128]，批量维可广播。
            )
        )
        self.spatial_blocks = nn.ModuleList(  # 保存依次执行的多个空间 Transformer block。
            [
                TransformerEncoderBlock(config)  # 每一层都拥有独立的 Attention 和内部 MLP 参数。
                for _ in range(config.spatial_encoder_layers)  # 按配置创建 2 个空间 block。
            ]
        )
        self.temporal_class_token = nn.Parameter(  # 创建汇总整个视频信息的可学习 CLS token。
            torch.zeros(  # 先以全零张量创建，稍后再进行截断正态初始化。
                1, 1, config.embedding_dim  # 形状为 [1,1,128]，之后扩展到每个样本。
            )
        )
        self.temporal_position = nn.Parameter(  # 创建可学习的时间位置编码。
            torch.zeros(  # 先以全零张量创建，稍后再进行截断正态初始化。
                1,  # 批量维为 1，以便对所有样本广播。
                self.temporal_tokens + 1,  # 10 个时间 token 加 1 个 CLS token。
                config.embedding_dim,  # 每个位置向量的维度为 128。
            )
        )
        self.embedding_dropout = nn.Dropout(  # 创建位置编码和 coverage 编码相加后的 dropout。
            config.dropout  # 使用配置中的 0.1 丢弃概率。
        )
        self.temporal_blocks = nn.ModuleList(  # 保存依次执行的多个时间 Transformer block。
            [
                TransformerEncoderBlock(config)  # 时间 block 与空间 block 结构相同但参数独立。
                for _ in range(config.temporal_encoder_layers)  # 按配置创建 2 个时间 block。
            ]
        )
        self.final_norm = nn.LayerNorm(  # 对最终时间 CLS 表征做一次标准化。
            config.embedding_dim  # 标准化 CLS token 的 128 个特征。
        )
        self.regression_head = nn.Linear(  # 创建 Encoder 外部的单层回归头。
            config.embedding_dim, config.output_dim  # 将 128 维 CLS 映射为 1 个 raw score。
        )
        self._initialize_weights()  # 创建完所有层后统一初始化可训练参数。

    def _initialize_weights(self) -> None:  # 定义模型参数的初始化规则。
        """保持迁移前的初始化和 checkpoint 参数键。"""

        nn.init.trunc_normal_(self.spatial_position, std=0.02)  # 初始化可学习空间位置编码。
        nn.init.trunc_normal_(self.temporal_position, std=0.02)  # 初始化可学习时间位置编码。
        nn.init.trunc_normal_(self.temporal_class_token, std=0.02)  # 初始化时间 CLS token。
        nn.init.zeros_(self.tubelet_coverage_embedding.weight)  # 先让所有 coverage 状态都不产生偏移。
        nn.init.trunc_normal_(  # 再仅随机初始化 coverage=1 的状态向量。
            self.tubelet_coverage_embedding.weight[1:2],  # 选取索引 1，不改索引 0 和 2。
            std=0.02,  # 使用标准差为 0.02 的截断正态分布。
        )
        for module in self.modules():  # 遍历当前模型及其所有子模块。
            if isinstance(module, nn.Linear):  # 对所有全连接层使用统一初始化规则。
                nn.init.trunc_normal_(module.weight, std=0.02)  # 初始化 Linear 权重。
                if module.bias is not None:  # 只处理确实带有偏置项的 Linear。
                    nn.init.zeros_(module.bias)  # 将 Linear 偏置初始化为 0。
            elif isinstance(module, nn.LayerNorm):  # 对所有 LayerNorm 使用恒等初始化。
                nn.init.ones_(module.weight)  # 将缩放系数初始化为 1。
                nn.init.zeros_(module.bias)  # 将平移偏置初始化为 0。

    def encode(  # 定义从订单流视频到最终 CLS 表征的编码过程。
        self,  # 当前 ViViT 模型实例。
        videos: torch.Tensor,  # 输入视频，预期形状为 [B,17,20,8,8]。
        frame_mask: torch.Tensor,  # 帧有效标记，预期形状为 [B,20]。
    ) -> torch.Tensor:  # 返回形状为 [B,128] 的视频表示。
        """输入视频和帧有效标记，返回最终 LayerNorm 后 CLS [B,128]。"""

        if videos.ndim != 5 or videos.shape[2] != self.config.frame_count:  # 检查视频维数和帧数。
            raise ValueError(  # 输入形状不符合约定时给出可定位问题的异常。
                f"视频形状异常：{tuple(videos.shape)}"  # 在错误消息中打印实际视频形状。
            )
        if frame_mask.shape != (len(videos), self.config.frame_count):  # 检查 mask 是否逐样本逐帧对应。
            raise ValueError(  # mask 形状错误时立即终止，避免静默广播。
                f"frame_mask 形状异常：{tuple(frame_mask.shape)}"  # 打印实际 mask 形状。
            )
        frame_mask = frame_mask.to(  # 将 mask 搬到视频所在设备并规范成布尔类型。
            device=videos.device, dtype=torch.bool  # 避免 CPU/GPU 不一致或整数逻辑歧义。
        )
        torch._assert(  # 使用可被 PyTorch 图模式识别的运行时断言。
            frame_mask[:, -1].all(),  # 要求每个样本最后一个目标日帧都有效。
            "每个视频的目标日帧必须有效",  # 断言失败时说明数据约束。
        )

  # [B,17,20,8,8] -> [B,128,10,4,4]。
        embedded = self.tubelet_embedding(videos)  # 将原始视频切成 tubelet 并得到 128 维特征。
        batch_size = embedded.shape[0]  # 读取批量大小，供后续 reshape 使用。

  # 每两帧统计 0/1/2 个有效帧，作为 coverage embedding 的索引。
        coverage = frame_mask.reshape(  # [B,20] -> [B,10,2]，每两帧组成一个 tubelet。
            batch_size,  # 保留批量维 B。
            self.temporal_tokens,  # 分成 10 个时间 tubelet。
            self.config.temporal_tubelet,  # 每组包含 2 个原始帧标记。
        ).sum(  # 对每组中的有效帧数量求和。
            dim=-1  # 沿每个 tubelet 内部的 2 帧维度求和。
        ).long()  # 转成整数索引，最终形状为 [B,10]。
        coverage_embedding = self.tubelet_coverage_embedding(  # 查询每个 tubelet 的 coverage 向量。
            coverage  # 输入 0、1、2 索引，输出形状为 [B,10,128]。
        )

  # 对每个时间 tubelet 内的 16 个空间 token 独立编码。
        spatial = embedded.permute(  # [B,128,10,4,4] -> [B,10,4,4,128]。
            0, 2, 3, 4, 1  # [B,128,10,4,4] 变成 [B,10,4,4,128]。
        ).reshape(  # [B,10,4,4,128] -> [B×10,16,128]。
            batch_size * self.temporal_tokens,  # 让每个时间 tubelet 成为独立空间样本。
            self.spatial_tokens,  # 每个空间序列包含 16 个 token。
            self.config.embedding_dim,  # 每个空间 token 维度为 128。
        )
        spatial_coverage = coverage_embedding.reshape(  # [B,10,128] -> [B×10,1,128]。
            batch_size * self.temporal_tokens,  # 与 spatial 第一维的 B×10 对齐。
            1,  # 每个 tubelet 只有一个 coverage 向量，之后广播给 16 个 token。
            self.config.embedding_dim,  # coverage 向量维度为 128。
        )
        spatial = self.embedding_dropout(  # 对三种 embedding 之和应用 dropout。
            spatial  # tubelet 自身携带的局部订单流特征。
            + self.spatial_position  # 加入 16 个 patch 的空间位置信息。
            + spatial_coverage  # 加入该 tubelet 有多少有效帧的信息。
        )
        for block in self.spatial_blocks:  # 按顺序通过 2 个空间 Transformer block。
            spatial = block(spatial)  # 每层执行空间自注意力和内部 MLP。
        frame_tokens = spatial.mean(  # 对同一 tubelet 的 16 个空间 token 做均值池化。
            dim=1  # 沿空间 token 维求均值，得到一个时间表示。
        ).reshape(  # [B×10,128] -> [B,10,128]。
            batch_size,  # 第一维恢复为 B。
            self.temporal_tokens,  # 第二维恢复为 10 个时间 token。
            self.config.embedding_dim,  # 每个时间 token 保持 128 维。
        )

  # CLS 永远有效；coverage=0 的 tubelet 不参与时序 attention。
        class_tokens = self.temporal_class_token.expand(  # 为批量中的每个样本复制 CLS token 视图。
            batch_size, -1, -1  # 从 [1,1,128] 扩展为 [B,1,128]。
        )
        temporal = torch.cat(  # 把 CLS token 放在所有时间 token 之前。
            [class_tokens, frame_tokens],  # 拼接后 token 顺序为 [CLS,t1,...,t10]。
            dim=1,  # 沿 token 序列维拼接，得到 [B,11,128]。
        )
        temporal = self.embedding_dropout(  # 对带时间位置编码的序列应用 dropout。
            temporal + self.temporal_position  # 让模型知道 CLS 和各 tubelet 的时间位置。
        )
        temporal_padding_mask = torch.cat(  # 构造与 11 个时间 token 一一对应的 padding mask。
            [
                torch.zeros(  # 为 CLS token 创建永不屏蔽的 False 标记。
                    (batch_size, 1),  # 每个样本对应一个 CLS mask 位置。
                    dtype=torch.bool,  # MultiheadAttention 要求布尔 mask。
                    device=videos.device,  # 与视频和模型保持在同一设备。
                ),
                coverage.eq(0),  # coverage=0 的时间 tubelet 标记为 True 并被屏蔽。
            ],
            dim=1,  # 拼成形状为 [B,11] 的完整时间 mask。
        )
        for block in self.temporal_blocks:  # 按顺序通过 2 个时间 Transformer block。
            temporal = block(  # 当前层执行时间自注意力和内部 MLP。
                temporal,  # 输入当前时间 token 序列。
                key_padding_mask=temporal_padding_mask,  # 禁止关注完全无效的 tubelet。
            )
        return self.final_norm(temporal[:, 0])  # 取 CLS token 并标准化，返回 [B,128] 表征。

    def forward(  # 定义训练和推理时调用 model(...) 的完整前向过程。
        self,  # 当前 ViViT 模型实例。
        videos: torch.Tensor,  # 输入订单流视频 [B,17,20,8,8]。
        frame_mask: torch.Tensor,  # 输入每一帧的有效标记 [B,20]。
    ) -> torch.Tensor:  # 返回每个样本一个 raw score，形状为 [B]。
        """输入 [B,17,20,8,8] 与 [B,20]，返回 raw score [B]。"""

        representation = self.encode(videos, frame_mask)  # 先获得每个视频的 128 维 CLS 表征。
        return self.regression_head(representation).squeeze(-1)  # [B,1] -> [B]，移除回归头末尾单维。
# endregion  # 结束公共 ViViT 模型区域。
