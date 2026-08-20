"""No.4：Barra 条件化乘法门控。"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from stage03_model import OrderFlowViViT


def deterministic_barra_permutation(  # 按稳定样本键产生与输入顺序无关的日内置乱位置。
    sample_indices: torch.Tensor,  # [B]，全局样本行号。
) -> torch.Tensor:  # 声明当前函数的返回类型。
    """按稳定样本键产生与输入顺序无关的日内置乱位置。"""

    identifiers = sample_indices.detach().cpu().numpy().astype(  # 转换identifiers的数据类型或存储位置。
        np.int64, copy=False
    )
    if identifiers.ndim != 1 or len(identifiers) < 2:  # 检查张量或数组形状是否符合接口约定。
        raise ValueError("Barra 置乱至少需要两个一维样本键")  # 输入数据不满足业务或数值约束时立即终止。
    if len(np.unique(identifiers)) != len(identifiers):  # 检查当前样本数量是否满足计算要求。
        raise ValueError("Barra 置乱样本键含重复值")  # 输入数据不满足业务或数值约束时立即终止。
    canonical_positions = np.argsort(identifiers, kind="stable")  # 计算canonical位置以固定样本顺序。
    canonical_ids = identifiers[canonical_positions]
    random_seed = np.random.SeedSequence(
        [len(canonical_ids), int(canonical_ids[0]), int(canonical_ids[-1])]
    ).generate_state(1)[0]
    generator = np.random.default_rng(random_seed)
    donor_positions = canonical_positions[
        generator.permutation(len(canonical_positions))
    ]
    permutation = np.empty(len(identifiers), dtype=np.int64)  # 初始化permutation。
    permutation[canonical_positions] = donor_positions
    return torch.from_numpy(permutation).to(sample_indices.device)  # 返回按当前规则组织后的结果。


def prepare_conditioning_inputs(  # 生成 real/shuffled/zero 条件输入，不改变诊断使用的真实 Barra。
    styles: torch.Tensor,  # [B,10]，十个连续 Barra 暴露。
    style_valid: torch.Tensor,  # [B,10]，风格暴露有效掩码。
    industry_codes: torch.Tensor,  # [B]，行业整数代码。
    sample_indices: torch.Tensor,  # [B]，全局样本行号。
    *,
    mode: str,  # 接收模式。
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # 声明当前函数的返回类型。
    """生成 real/shuffled/zero 条件输入，不改变诊断使用的真实 Barra。"""

    if mode in {"off", "video_only", "real"}:  # 检查当前值是否属于允许集合。
        return styles, style_valid, industry_codes  # 返回生成的结果与诊断。
    if mode == "zero":  # 按当前模式、标识或状态选择处理分支。
        return (  # 返回生成的结果与诊断。
            torch.zeros_like(styles),
            torch.zeros_like(style_valid, dtype=torch.bool),
            torch.zeros_like(industry_codes),
        )
    if mode != "shuffled":  # 按当前模式、标识或状态选择处理分支。
        raise ValueError(f"未知 Barra 条件模式：{mode}")  # 输入数据不满足业务或数值约束时立即终止。
    permutation = deterministic_barra_permutation(sample_indices).to(  # 转换permutation的数据类型或存储位置。
        styles.device  # 汇入风格设备。
    )
    return (  # 返回生成的结果与诊断。
        styles.index_select(0, permutation),
        style_valid.index_select(0, permutation),
        industry_codes.index_select(0, permutation),
    )



class BarraConditionalViViT(nn.Module):  # 用十风格、有效标记和行业对 CLS 做有界乘法调制。
    """用十风格、有效标记和行业对 CLS 做有界乘法调制。
    """

    def __init__(  # 初始化当前对象的配置、状态与子组件。
        self,  # 汇入self。
        backbone: OrderFlowViViT,  # 接收backbone。
        *,
        conditioning_mode: str,  # 接收conditioning模式。
        style_count: int,  # 接收风格数量。
        industry_count: int,  # 接收行业数量。
        embedding_dim: int,  # 接收嵌入dim。
        gate_hidden_dim: int,  # 接收门控隐藏层dim。
        industry_embedding_dim: int,  # 接收行业嵌入dim。
        dropout: float,  # 接收dropout。
        gate_scale: float,  # 接收门控scale。
    ) -> None:  # 声明当前函数的返回类型。
        super().__init__()
        if conditioning_mode not in {"video_only", "real"}:  # 检查当前值是否属于允许集合。
            raise ValueError("条件模型内部模式只能是 video_only 或 real")  # 输入数据不满足业务或数值约束时立即终止。
        self.backbone = backbone  # 保存backbone，供该对象后续步骤复用。
        self.conditioning_mode = conditioning_mode  # 保存conditioning模式，供该对象后续步骤复用。
        self.style_count = int(style_count)  # 保存风格数量，供该对象后续步骤复用。
        self.industry_count = int(industry_count)  # 保存行业数量，供该对象后续步骤复用。
        self.gate_scale = float(gate_scale)  # 保存门控scale，供该对象后续步骤复用。
        self.industry_embedding = nn.Embedding(  # 将离散状态映射为 embedding。
            industry_count + 1,
            industry_embedding_dim,  # 汇入行业嵌入dim。
            padding_idx=0,  # 固定缺失类别的零嵌入。
        )
        input_dim = 2 * style_count + industry_embedding_dim
        self.barra_encoder = nn.Sequential(  # 按顺序组成网络子层。
            nn.Linear(input_dim, gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_dim, gate_hidden_dim),
            nn.GELU(),
        )
        self.gate_projection = nn.Linear(  # 线性投影到目标维度。
            gate_hidden_dim, embedding_dim  # 传入门控隐藏层dim、嵌入dim。
        )
        self._initialize_conditioning_branch()  # 初始化本分支参数。

    def _initialize_conditioning_branch(self) -> None:  # 零初始化最终门控，使初始输出严格等于无条件模型。
        """零初始化最终门控，使初始输出严格等于无条件模型。"""

        nn.init.trunc_normal_(  # 截断正态初始化。
            self.industry_embedding.weight[1:], std=0.02
        )
        with torch.no_grad():  # 关闭梯度记录以降低验证或推理开销。
            self.industry_embedding.weight[0].zero_()
        for module in self.barra_encoder.modules():  # 遍历Barraencodermodules，逐项处理module。
            if isinstance(module, nn.Linear):  # 检查对象类型是否符合接口约定。
                nn.init.trunc_normal_(module.weight, std=0.02)  # 截断正态初始化。
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.gate_projection.weight)
        nn.init.zeros_(self.gate_projection.bias)

    def forward(  # 返回 raw score、门控向量和未条件化 CLS。
        self,  # 汇入self。
        videos: torch.Tensor,  # [B,17,20,8,8]，订单流视频。
        frame_mask: torch.Tensor,  # [B,20]，逐帧有效标记。
        styles: torch.Tensor,  # [B,10]，十个连续 Barra 暴露。
        style_valid: torch.Tensor,  # [B,10]，风格暴露有效掩码。
        industry_codes: torch.Tensor,  # [B]，行业整数代码。
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # 声明当前函数的返回类型。
        """返回 raw score、门控向量和未条件化 CLS。"""

        representation = self.backbone.encode(videos, frame_mask)  # 编码为共享视频表征。
        if self.conditioning_mode == "video_only":  # 按当前模式、标识或状态选择处理分支。
            gate = torch.zeros_like(representation)  # 初始化门控。
        else:  # 处理上述条件均不成立的情况。
            if styles.shape != (  # 检查张量或数组形状是否符合接口约定。
                len(representation), self.style_count
            ):
                raise ValueError("十风格条件张量形状异常")  # 输入数据不满足业务或数值约束时立即终止。
            if style_valid.shape != styles.shape:  # 检查张量或数组形状是否符合接口约定。
                raise ValueError("风格有效掩码形状异常")  # 输入数据不满足业务或数值约束时立即终止。
            codes = industry_codes.long().reshape(-1)  # [B,...] -> [B]。
            if len(codes) != len(representation) or bool(  # 检查当前样本数量是否满足计算要求。
                ((codes < 0) | (codes > self.industry_count)).any()
            ):
                raise ValueError("行业代码形状或取值异常")  # 输入数据不满足业务或数值约束时立即终止。
            condition = torch.cat(  # 三部分沿特征维拼接为 [B,2K+E]。
                [
                    styles.float(),
                    style_valid.float(),
                    self.industry_embedding(codes),
                ],
                dim=1,  # 指定张量运算维度。
            )
            encoded = self.barra_encoder(condition)  # [B,2K+E] -> [B,H]。
            gate = self.gate_scale * torch.tanh(  # [B,H] -> [B,D]。
                self.gate_projection(encoded)
            )
        conditioned = representation * (1.0 + gate)  # [B,D]，逐特征乘法调制。
        raw_score = self.backbone.regression_head(  # 计算原始分数。
            conditioned
        ).squeeze(-1)  # [B,1] -> [B]。
        return raw_score, gate, representation  # 返回 raw score、门控向量和未条件化 CLS。
