"""No.4：Barra 条件化乘法门控。"""
from __future__ import annotations

import hashlib

import numpy as np
import torch
from torch import nn

from stage03_model import OrderFlowViViT


# 定义 `deterministic_barra_permutation` 函数，封装这一阶段可独立调用的处理步骤。
def deterministic_barra_permutation(
    sample_indices: torch.Tensor,
) -> torch.Tensor:
    """按稳定样本键产生与输入顺序无关的日内置乱位置。
    
    输入：
        sample_indices: torch.Tensor，格式 [N]。
    输出：
        torch.Tensor。
        """

    # 调用 `sample_indices.detach().cpu().numpy().astype` 统一设备或数据类型，并将结果保存到 `identifiers`。
    identifiers = sample_indices.detach().cpu().numpy().astype(
        np.int64, copy=False
    )
    # 检查 `identifiers.ndim != 1 or len(identifiers) < 2`；根据条件结果选择对应处理分支。
    if identifiers.ndim != 1 or len(identifiers) < 2:
        # 抛出 `ValueError('Barra 置乱至少需要两个一维样本键')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("Barra 置乱至少需要两个一维样本键")
    # 检查 `len(np.unique(identifiers)) != len(identifiers)`；根据条件结果选择对应处理分支。
    if len(np.unique(identifiers)) != len(identifiers):
        # 抛出 `ValueError('Barra 置乱样本键含重复值')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("Barra 置乱样本键含重复值")
    # 调用 `np.argsort` 根据分数计算截面排序或排序索引，并将结果保存到 `canonical_positions`。
    canonical_positions = np.argsort(identifiers, kind="stable")
    # 读取 `identifiers[canonical_positions]` 的当前值，并赋给 `canonical_ids`。
    canonical_ids = identifiers[canonical_positions]
    # 调用 `hashlib.sha256(f'barra-conditioning-shuffle-v1:{len(canonical_i…` 对关键配置生成稳定签名以支持复现和一致性检查，并将结果保存到 `digest`。
    digest = hashlib.sha256(
        (
            f"barra-conditioning-shuffle-v1:{len(canonical_ids)}:"
            f"{canonical_ids[0]}:{canonical_ids[-1]}"
        ).encode("utf-8")
    ).digest()
    # 调用 `np.random.default_rng` 创建独立且可复现的 NumPy 随机数生成器，并将结果保存到 `generator`。
    generator = np.random.default_rng(
        int.from_bytes(digest[:8], byteorder="little", signed=False)
    )
    # 读取 `canonical_positions[generator.permutation(len(canonical_positions))]` 的当前值，并赋给 `donor_positions`。
    donor_positions = canonical_positions[
        generator.permutation(len(canonical_positions))
    ]
    # 调用 `np.empty` 按目标形状初始化数值容器 `permutation`。
    permutation = np.empty(len(identifiers), dtype=np.int64)
    # 读取 `donor_positions` 的当前值，并赋给 `permutation[canonical_positions]`。
    permutation[canonical_positions] = donor_positions
    # 将 `torch.from_numpy(permutation).to(sample_indices.device)` 作为当前函数结果返回给调用方。
    return torch.from_numpy(permutation).to(sample_indices.device)


# 定义 `prepare_conditioning_inputs` 函数，封装这一阶段可独立调用的处理步骤。
def prepare_conditioning_inputs(
    styles: torch.Tensor,
    style_valid: torch.Tensor,
    industry_codes: torch.Tensor,
    sample_indices: torch.Tensor,
    *,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """生成 real/shuffled/zero 条件输入，不改变诊断使用的真实 Barra。
    
    输入：
        styles: torch.Tensor，格式 [N,10]。
        style_valid: torch.Tensor，格式 [N,10]。
        industry_codes: torch.Tensor，格式 [N]。
        sample_indices: torch.Tensor，格式 [N]。
        mode: str。
    输出：
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]。
        """

    # 检查 `mode in {'off', 'video_only', 'real'}`；根据条件结果选择对应处理分支。
    if mode in {"off", "video_only", "real"}:
        # 将 `(styles, style_valid, industry_codes)` 作为当前函数结果返回给调用方。
        return styles, style_valid, industry_codes
    # 检查 `mode == 'zero'`；根据条件结果选择对应处理分支。
    if mode == "zero":
        # 将 `(torch.zeros_like(styles), torch.zeros_like(style_valid, dtype=torch.bool), torch.zeros_like(in…` 作为当前函数结果返回给调用方。
        return (
            torch.zeros_like(styles),
            torch.zeros_like(style_valid, dtype=torch.bool),
            torch.zeros_like(industry_codes),
        )
    # 检查 `mode != 'shuffled'`；根据条件结果选择对应处理分支。
    if mode != "shuffled":
        # 抛出 `ValueError(f'未知 Barra 条件模式：{mode}')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(f"未知 Barra 条件模式：{mode}")
    # 调用 `deterministic_barra_permutation(sample_indices).to` 统一设备或数据类型，并将结果保存到 `permutation`。
    permutation = deterministic_barra_permutation(sample_indices).to(
        styles.device
    )
    # 将 `(styles.index_select(0, permutation), style_valid.index_select(0, permutation), industry_codes.…` 作为当前函数结果返回给调用方。
    return (
        styles.index_select(0, permutation),
        style_valid.index_select(0, permutation),
        industry_codes.index_select(0, permutation),
    )


# region TR-N04-GATE
# 定义 `BarraConditionalViViT` 类，集中封装该阶段的数据结构与相关行为。
class BarraConditionalViViT(nn.Module):
    """用十风格、有效标记和行业对 CLS 做有界乘法调制。
    """

    # 定义初始化方法，接收配置并建立实例后续计算所需的内部状态。
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
        """执行 __init__ 对应的处理步骤。
        
        输入：
            backbone: OrderFlowViViT。
            conditioning_mode: str。
            style_count: int。
            industry_count: int。
            embedding_dim: int。
            gate_hidden_dim: int。
            industry_embedding_dim: int。
            dropout: float。
            gate_scale: float。
        输出：
            None。
            """
        # 初始化父类状态，确保继承链中的基础组件正确注册。
        super().__init__()
        # 检查 `conditioning_mode not in {'video_only', 'real'}`；根据条件结果选择对应处理分支。
        if conditioning_mode not in {"video_only", "real"}:
            # 抛出 `ValueError('条件模型内部模式只能是 video_only 或 real')`，阻止不满足数据契约的流程继续执行。
            raise ValueError("条件模型内部模式只能是 video_only 或 real")
        # 读取 `backbone` 的当前值，并赋给 `self.backbone`。
        self.backbone = backbone
        # 读取 `conditioning_mode` 的当前值，并赋给 `self.conditioning_mode`。
        self.conditioning_mode = conditioning_mode
        # 调用 `int` 统一设备或数据类型，并将结果保存到 `self.style_count`。
        self.style_count = int(style_count)
        # 调用 `int` 统一设备或数据类型，并将结果保存到 `self.industry_count`。
        self.industry_count = int(industry_count)
        # 调用 `float` 统一设备或数据类型，并将结果保存到 `self.gate_scale`。
        self.gate_scale = float(gate_scale)
        # 调用 `nn.Embedding` 创建离散状态到连续向量的可学习映射，并将结果保存到 `self.industry_embedding`。
        self.industry_embedding = nn.Embedding(
            industry_count + 1,
            industry_embedding_dim,
            padding_idx=0,
        )
        # 计算 `2 * style_count + industry_embedding_dim`，并将数值结果保存到 `input_dim`。
        input_dim = 2 * style_count + industry_embedding_dim
        # 调用 `nn.Sequential` 按执行顺序组合多个神经网络子层，并将结果保存到 `self.barra_encoder`。
        self.barra_encoder = nn.Sequential(
            nn.Linear(input_dim, gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_dim, gate_hidden_dim),
            nn.GELU(),
        )
        # 调用 `nn.Linear` 创建线性投影层以变换特征维度，并将结果保存到 `self.gate_projection`。
        self.gate_projection = nn.Linear(
            gate_hidden_dim, embedding_dim
        )
        # 调用 `self._initialize_conditioning_branch` 按模型配置初始化当前分支的可训练参数。
        self._initialize_conditioning_branch()

    # 定义内部辅助函数 `_initialize_conditioning_branch`，封装仅供本模块复用的计算步骤。
    def _initialize_conditioning_branch(self) -> None:
        """零初始化最终门控，使初始输出严格等于无条件模型。
        
        输入：
            无显式输入参数。
        输出：
            None。
            """

        # 调用 `nn.init.trunc_normal_` 用截断正态分布初始化可训练参数。
        nn.init.trunc_normal_(
            self.industry_embedding.weight[1:], std=0.02
        )
        # 在 `torch.no_grad()` 管理的上下文中执行本段逻辑，并保证资源按约定释放。
        with torch.no_grad():
            # 调用 `self.industry_embedding.weight[0].zero_` 执行 `zero_` 对应的明确变换，并产生后续步骤使用的结果。
            self.industry_embedding.weight[0].zero_()
        # 遍历 `self.barra_encoder.modules()`，并把当前元素绑定到 `module` 逐项处理。
        for module in self.barra_encoder.modules():
            # 检查 `isinstance(module, nn.Linear)`；根据条件结果选择对应处理分支。
            if isinstance(module, nn.Linear):
                # 调用 `nn.init.trunc_normal_` 用截断正态分布初始化可训练参数。
                nn.init.trunc_normal_(module.weight, std=0.02)
                # 调用 `nn.init.zeros_` 将指定参数初始化为零。
                nn.init.zeros_(module.bias)
        # 调用 `nn.init.zeros_` 将指定参数初始化为零。
        nn.init.zeros_(self.gate_projection.weight)
        # 调用 `nn.init.zeros_` 将指定参数初始化为零。
        nn.init.zeros_(self.gate_projection.bias)

    # 定义 `forward` 函数，封装这一阶段可独立调用的处理步骤。
    def forward(
        self,
        videos: torch.Tensor,
        frame_mask: torch.Tensor,
        styles: torch.Tensor,
        style_valid: torch.Tensor,
        industry_codes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 raw score、门控向量和未条件化 CLS。
        
        输入：
            videos: torch.Tensor，格式 [B,17,20,8,8]。
            frame_mask: torch.Tensor，格式 [B,20]。
            styles: torch.Tensor，格式 [N,10]。
            style_valid: torch.Tensor，格式 [N,10]。
            industry_codes: torch.Tensor，格式 [N]。
        输出：
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]。
            """

        # 调用 `self.backbone.encode` 用模型主干把视频编码为共享表征，并将结果保存到 `representation`。
        representation = self.backbone.encode(videos, frame_mask)
        # 检查 `self.conditioning_mode == 'video_only'`；根据条件结果选择对应处理分支。
        if self.conditioning_mode == "video_only":
            # 调用 `torch.zeros_like` 按目标形状初始化数值容器 `gate`。
            gate = torch.zeros_like(representation)
        # 当前置条件不成立时进入备用分支，保证其余合法情况仍被处理。
        else:
            # 检查 `styles.shape != (len(representation), self.style_count)`；根据条件结果选择对应处理分支。
            if styles.shape != (
                len(representation), self.style_count
            ):
                # 抛出 `ValueError('十风格条件张量形状异常')`，阻止不满足数据契约的流程继续执行。
                raise ValueError("十风格条件张量形状异常")
            # 检查 `style_valid.shape != styles.shape`；根据条件结果选择对应处理分支。
            if style_valid.shape != styles.shape:
                # 抛出 `ValueError('风格有效掩码形状异常')`，阻止不满足数据契约的流程继续执行。
                raise ValueError("风格有效掩码形状异常")
            # 调用 `industry_codes.long().reshape` 调整张量维度，并将新视图保存到 `codes`。
            codes = industry_codes.long().reshape(-1)
            # 检查 `len(codes) != len(representation) or bool(((codes < 0) | (codes > self.industry_count)).any())`；根据条件结果选择对应处理分支。
            if len(codes) != len(representation) or bool(
                ((codes < 0) | (codes > self.industry_count)).any()
            ):
                # 抛出 `ValueError('行业代码形状或取值异常')`，阻止不满足数据契约的流程继续执行。
                raise ValueError("行业代码形状或取值异常")
            # 调用 `torch.cat` 拼接各批结果，并将合并值保存到 `condition`。
            condition = torch.cat(
                [
                    styles.float(),
                    style_valid.float(),
                    self.industry_embedding(codes),
                ],
                dim=1,
            )
            # 调用 `self.barra_encoder` 通过当前网络子模块变换输入表征，并将结果保存到 `encoded`。
            encoded = self.barra_encoder(condition)
            # 计算 `self.gate_scale * torch.tanh(self.gate_projection(encoded))`，并将数值结果保存到 `gate`。
            gate = self.gate_scale * torch.tanh(
                self.gate_projection(encoded)
            )
        # 计算 `representation * (1.0 + gate)`，并将数值结果保存到 `conditioned`。
        conditioned = representation * (1.0 + gate)
        # 调用 `self.backbone.regression_head(conditioned).squeeze` 调整张量维度，并将新视图保存到 `raw_score`。
        raw_score = self.backbone.regression_head(
            conditioned
        ).squeeze(-1)
        # 将 `(raw_score, gate, representation)` 作为当前函数结果返回给调用方。
        return raw_score, gate, representation
# endregion
