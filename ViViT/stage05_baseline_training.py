"""阶段 05：训练、验证并集成无 Barra 约束的 ViViT 基线模型。

序列取数复用 stage01，模型定义复用 stage03；本模块只保留基线训练循环、
checkpoint、年度滚动任务、三种子集成和命令行入口。
"""
from __future__ import annotations

import argparse
import contextlib
from collections.abc import Iterator
import json
import os
import random
import subprocess
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.utils.data import DataLoader

def inference_model(model: nn.Module) -> nn.Module:  # 返回不触发 DDP 集体前向的本地模型。
    """返回不触发 DDP 集体前向的本地模型。"""

    return getattr(model, "module", model)  # 返回不触发 DDP 集体前向的本地模型。


def make_distributed_data_loaders(  # 构造无重复的逐 rank loader，并保持全局 batch 语义。
    forecast_year: int,  # 接收forecast年份。
    seed: int,  # 接收随机种子。
    config: TrainingConfig,  # 接收配置。
    workers: int,  # 接收workers。
    pin_memory: bool,  # 接收pinmemory。
    context: distributed.DistributedContext,  # 接收上下文。
) -> tuple[DataLoader, DataLoader, DataLoader | None, dict[str, object]]:  # 声明当前函数的返回类型。
    """构造无重复的逐 rank loader，并保持全局 batch 语义。"""

    split = load_sequence_split(forecast_year)
    common: dict[str, object] = {  # 初始化公共容器。
        "num_workers": workers,  # 记录numworkers。
        "pin_memory": pin_memory,  # 记录pinmemory。
        "collate_fn": VideoBatchCollator(),  # 记录collatefn。
    }
    if workers > 0:  # 检查当前数值是否越过业务或数值阈值。
        common.update({"persistent_workers": True, "prefetch_factor": 1})  # 用当前配置更新参数字典。
    generator = torch.Generator().manual_seed(seed)
    train_indices = split["train_indices"]
    train_dataset = SampleIndexDataset(train_indices)  # 创建训练dataset。
    if config.train_batch_mode == "trade_date":  # 按当前模式、标识或状态选择处理分支。
        date_ids = np.load(DATE_IDS_PATH, mmap_mode="r")  # 读取交易日ids。
        temporal_mode = config.loss_name == TEMPORAL_TURNOVER_LOSS  # 配置时序模式。
        instruments = None  # 先把股票初始化为空，后续按条件填充。
        if temporal_mode:  # 根据当前数据状态选择处理分支。
            instruments = pd.read_parquet(  # 读取股票。
                SAMPLE_INDEX_PATH, columns=["instrument_id"]  # 定位样本索引路径。
            )["instrument_id"].to_numpy(copy=False)
        train_sampler = distributed.DistributedTradeDateBatchSampler(  # 创建训练sampler。
            train_indices,  # 汇入训练索引。
            date_ids,  # 汇入交易日ids。
            seed=seed,  # 固定本轮随机种子。
            shuffle=not temporal_mode,  # 控制是否打乱样本顺序。
            instrument_ids=instruments,  # 配置股票ids。
            rank=context.rank,  # 配置排名。
            world_size=context.world_size,  # 配置world大小。
        )
        train_loader = DataLoader(  # 创建训练数据加载器。
            train_dataset,  # 汇入训练dataset。
            batch_sampler=train_sampler,  # 配置批次sampler。
            generator=generator,  # 传入固定随机数生成器以保证复现。
            **common,
        )
        batch_summary = train_sampler.summary()
    else:  # 处理上述条件均不成立的情况。
        train_sampler = distributed.DistributedBatchSampler(  # 创建训练sampler。
            len(train_dataset),
            config.batch_size,  # 汇入批次大小。
            rank=context.rank,  # 配置排名。
            world_size=context.world_size,  # 配置world大小。
            seed=seed,  # 固定本轮随机种子。
            shuffle=True,  # 控制是否打乱样本顺序。
            drop_last=True,  # 控制是否丢弃最后一个不完整批次。
        )
        train_loader = DataLoader(  # 创建训练数据加载器。
            train_dataset,  # 汇入训练dataset。
            batch_sampler=train_sampler,  # 配置批次sampler。
            generator=generator,  # 传入固定随机数生成器以保证复现。
            **common,
        )
        batch_count = len(train_indices) // config.batch_size  # 计算批次数量。
        batch_summary = {  # 初始化批次汇总容器。
            "train_batch_mode": "fixed_samples",  # 把 train_batch_mode 映射为"fixed_samples"。
            "train_batch_count": batch_count,  # 记录训练批次数量。
            "train_date_count": None,  # 记录训练交易日数量。
            "train_sample_count": batch_count * config.batch_size,  # 记录训练样本数量。
            "train_batch_size_min": config.batch_size,  # 记录训练批次大小min。
            "train_batch_size_median": float(config.batch_size),  # 记录训练批次大小median。
            "train_batch_size_max": config.batch_size,  # 记录训练批次大小max。
            "dropped_train_samples": len(train_indices) - batch_count * config.batch_size,  # 记录dropped训练样本。
            "reshuffle_each_epoch": True,  # 记录reshuffleeach轮次。
            "gradient_update_mode": "per_batch",  # 把 gradient_update_mode 映射为"per_batch"。
            "optimizer_steps_per_epoch": batch_count,  # 记录优化器stepsper轮次。
        }

    def inference_loader(indices: np.ndarray) -> DataLoader:  # inference数据加载器。
        """执行 inference_loader 对应的分布式训练处理。"""
        dataset = SampleIndexDataset(indices)  # 创建dataset。
  # 按原推理 batch 边界切分，确保所有 rank 执行相同次数的 collective。
        sampler = distributed.DistributedBatchSampler(  # 创建sampler。
            len(dataset),
            config.inference_batch_size,  # 汇入inference批次大小。
            rank=context.rank,  # 配置排名。
            world_size=context.world_size,  # 配置world大小。
            seed=seed,  # 固定本轮随机种子。
            shuffle=False,  # 控制是否打乱样本顺序。
            drop_last=False,  # 控制是否丢弃最后一个不完整批次。
        )
        return DataLoader(  # 返回上述处理结果。
            dataset,  # 汇入dataset。
            batch_sampler=sampler,  # 配置批次sampler。
            generator=generator,  # 传入固定随机数生成器以保证复现。
            **common,
        )

    validation_loader = inference_loader(split["validation_indices"])  # 创建验证数据加载器。
    prediction_loader = None  # 先把预测数据加载器初始化为空，后续按条件填充。
    if len(split["prediction_indices"]):  # 检查当前样本数量是否满足计算要求。
        prediction_loader = inference_loader(split["prediction_indices"])  # 创建预测数据加载器。
    return train_loader, validation_loader, prediction_loader, batch_summary  # 返回构造无重复的逐 rank loader，并保持全局 batch 语义。


def gather_global_training_batch(  # 汇聚全局训练批次。
    predictions: torch.Tensor,  # [B]，预测分数。
    targets: torch.Tensor,  # [B]，训练标签。
    batch: dict[str, torch.Tensor],
    context: distributed.DistributedContext | None,  # 接收上下文。
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # 声明当前函数的返回类型。
    """执行 gather_global_training_batch 对应的分布式训练处理。"""

    sample_indices = batch["sample_indices"].to(predictions.device)  # [B]。
    if context is None:  # 检查所需对象是否尚未初始化。
        return predictions, targets, sample_indices  # 返回上述处理结果。
    global_predictions = distributed.differentiable_all_gather(predictions)  # [N_global]。
    global_targets = distributed.all_gather_tensor(targets)  # [N_global]。
    global_indices = distributed.all_gather_tensor(sample_indices)  # [N_global]。
    order = torch.argsort(global_indices)  # [N_global]。
    return (  # 返回上述处理结果。
        global_predictions[order],  # 汇入全局预测。
        global_targets[order],  # 汇入全局标签。
        global_indices[order],  # 汇入全局索引。
    )


def gather_global_inference_tensor(  # 汇聚全局inference张量。
    tensor: torch.Tensor,  # [N_local,...]，本 rank 推理输出。
    context: distributed.DistributedContext | None,  # 接收上下文。
) -> torch.Tensor:  # 声明当前函数的返回类型。
    """执行 gather_global_inference_tensor 对应的分布式训练处理。"""
    return tensor if context is None else distributed.all_gather_tensor(tensor)  # 返回上述处理结果。


VIVIT_DIR = Path(__file__).resolve().parent  # 定位vivit目录。
VIT_DIR = VIVIT_DIR.parent / "ViT"  # 定位vit目录。
if str(VIT_DIR) not in sys.path:  # 检查当前值是否属于允许集合。
    sys.path.insert(0, str(VIT_DIR))

from vit_stage06_ranking import (  # noqa: E402。
    LOSS_NAMES,
    STANDARD_RANK_LOSS_NAMES,
    TEMPORAL_TURNOVER_LOSS,
    compute_training_loss,
    pearson_correlation,
    temporal_turnover_loss_components,
)
from vit_stage07_training import (  # noqa: E402。
    TRAIN_BATCH_MODES,
    VALIDATION_METRICS,
    TradeDateBatchSampler,
    summarize_validation_arrays,
)
RANK_LOSS_NAMES = (*STANDARD_RANK_LOSS_NAMES, TEMPORAL_TURNOVER_LOSS)  # 集中声明排名损失名称。
from stage01_sequence_data import (  # noqa: E402。
    SampleIndexDataset,
    VideoBatchCollator,
    atomic_save_checkpoint,
    atomic_write_json,
    atomic_write_parquet,
    atomic_write_text,
    build_sequence_store,
    load_sequence_split,
    loader_batch_summary,
    make_data_loaders,
    require_aligned_sequence_store,
    set_train_epoch,
    source_manifests,
    stable_hash,
)
import stage05_distributed_training as distributed  # noqa: E402。
from stage03_model import (  # noqa: E402。
    OrderFlowViViT,
    VivitModelConfig,
)

DATA_DIR = VIVIT_DIR / "data"  # 定位数据目录。
FEATURE_STORE_DIR = DATA_DIR / "feature_store"  # 定位特征存储目录。
SEQUENCE_STORE_DIR = DATA_DIR / "aligned_sequence_store"  # 定位sequence存储目录。
HISTORY_INDICES_PATH = SEQUENCE_STORE_DIR / "history_indices.npy"  # 定位历史记录索引路径。
SEQUENCE_MANIFEST_PATH = SEQUENCE_STORE_DIR / "sequence_manifest.json"  # 定位sequence清单路径。
SEQUENCE_SUMMARY_PATH = SEQUENCE_STORE_DIR / "rolling_split_summary.parquet"  # 定位sequence汇总路径。

FEATURES_PATH = FEATURE_STORE_DIR / "features.npy"  # 定位特征路径。
TARGETS_PATH = FEATURE_STORE_DIR / "targets.npy"  # 定位标签路径。
DATE_IDS_PATH = FEATURE_STORE_DIR / "date_ids.npy"  # 定位交易日ids路径。
SAMPLE_INDEX_PATH = FEATURE_STORE_DIR / "sample_index.parquet"  # 定位样本索引路径。
SOURCE_DATASET_MANIFEST_PATH = FEATURE_STORE_DIR / "dataset_manifest.json"  # 定位来源dataset清单路径。
SOURCE_ROLLING_MANIFEST_PATH = FEATURE_STORE_DIR / "rolling_split_manifest.json"  # 定位来源rolling清单路径。
SOURCE_SPLIT_DIR = FEATURE_STORE_DIR / "rolling_splits"  # 定位来源切分目录。

FORECAST_YEARS = tuple(range(2018, 2026))  # 集中声明forecast年份。
SEEDS = (42, 43, 44)  # 设置随机种子。
SEQUENCE_LENGTH = 20  # 设置sequencelength。


# 使用 `dataclass(frozen=True)` 包装 `TrainingConfig`，并定义其可复用的处理逻辑。
@dataclass(frozen=True)  # 自动生成配置对象的初始化与比较方法。
class TrainingConfig:  # 研报训练参数与项目复现参数。
    """研报训练参数与项目复现参数。
    """

    train_batch_mode: str = "trade_date"  # 接收训练批次模式。
    batch_size: int = 5000  # 接收批次大小。
    inference_batch_size: int = 5000  # 接收inference批次大小。
    loss_name: str = "pearson_ic"  # 接收损失名称。
    rank_temperature: float = 0.1  # 接收排名temperature。
    turnover_lambda: float = 0.1  # 接收turnoverlambda。
    huber_delta: float = 1.0  # 接收huberdelta。
    smooth_l1_beta: float = 0.1  # 接收smoothl1beta。
    hybrid_weight: float = 0.5  # 接收hybrid权重。
    validation_metric: str = "daily_mean_pearson_ic"  # 接收验证指标。
    learning_rate: float = 0.001  # 接收learning比率。
    max_epochs: int = 100  # 接收maxepochs。
    early_stopping_patience: int = 10  # 接收earlystoppingpatience。
    adam_beta1: float = 0.9  # 接收adambeta1。
    adam_beta2: float = 0.999  # 接收adambeta2。
    adam_epsilon: float = 1e-8  # 接收adamepsilon。
    weight_decay: float = 0.0  # 接收权重decay。
    improvement_tolerance: float = 1e-6  # 接收improvementtolerance。


MODEL_CONFIG = VivitModelConfig()  # 设置模型配置。


def set_random_seed(seed: int) -> None:  # 同步设置 Python、NumPy、CPU 与 CUDA 随机种子。
    """同步设置 Python、NumPy、CPU 与 CUDA 随机种子。"""
    random.seed(seed)  # 固定本轮采样种子。
    np.random.seed(seed)  # 固定本轮采样种子。
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # 根据当前数据状态选择处理分支。
        torch.cuda.manual_seed_all(seed)


def unwrap_model(model: nn.Module) -> nn.Module:  # 从编译或包装模型中取回原始 PyTorch 模型。
    """从编译或包装模型中取回原始 PyTorch 模型。"""
    wrapped = getattr(model, "module", None)
    if wrapped is not None:  # 检查必需数据、配置或校验结果是否缺失。
        return unwrap_model(wrapped)  # 返回完成投影或提取后的结果。
    original = getattr(model, "_orig_mod", None)
    return unwrap_model(original) if original is not None else model  # 返回完成投影或提取后的结果。


def build_model(  # 按当前配置构建模型，并保持历史 checkpoint 的参数键兼容。
    device: torch.device,  # 接收设备。
    use_compile: bool,  # 接收usecompile。
    compile_mode: str,  # 接收compile模式。
    train_batch_mode: str = "trade_date",  # 接收训练批次模式。
) -> nn.Module:  # 声明当前函数的返回类型。
    """按当前配置构建模型，并保持历史 checkpoint 的参数键兼容。"""
    model = OrderFlowViViT(MODEL_CONFIG).to(device)  # 创建模型。
    if use_compile:  # 根据当前数据状态选择处理分支。
        if device.type != "cuda":  # 按当前模式、标识或状态选择处理分支。
            raise RuntimeError("torch.compile 训练要求 CUDA")  # 当前运行环境或任务状态不支持继续执行。
        if train_batch_mode == "trade_date":  # 按当前模式、标识或状态选择处理分支。
            model = torch.compile(model, mode=compile_mode, dynamic=True)  # 编译模型计算图。
        else:  # 处理上述条件均不成立的情况。
            model = torch.compile(model, mode=compile_mode)  # 编译模型计算图。
    return model  # 返回模型。


def autocast_context(device: torch.device) -> contextlib.AbstractContextManager:  # 按设备和配置返回混合精度上下文。
    """按设备和配置返回混合精度上下文。"""
    if device.type == "cuda" and torch.cuda.is_bf16_supported():  # 按当前模式、标识或状态选择处理分支。
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)  # 返回按当前规则组织后的结果。
    return contextlib.nullcontext()  # 返回按当前规则组织后的结果。


class CudaBatchPrefetcher:  # 用独立 CUDA stream 预取下一批视频、掩码与标签。
    """用独立 CUDA stream 预取下一批视频、掩码与标签。
    """

    def __init__(self, loader: DataLoader, device: torch.device, enabled: bool) -> None:  # 初始化当前对象的配置、状态与子组件。
        self.loader = loader  # 保存数据加载器，供该对象后续步骤复用。
        self.device = device  # 保存设备，供该对象后续步骤复用。
        self.enabled = enabled and device.type == "cuda"

    def _move(  # 把批次张量非阻塞搬到 CUDA，并保留非张量元数据。
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # 声明当前函数的返回类型。
        """把批次张量非阻塞搬到 CUDA，并保留非张量元数据。"""
        non_blocking = self.device.type == "cuda"  # 允许设备拷贝与计算异步重叠。
        return (  # 返回完成上述格式转换后的结果。
            batch["videos"].to(self.device, non_blocking=non_blocking),  # [B,17,20,8,8]。
            batch["frame_mask"].to(self.device, non_blocking=non_blocking),  # [B,20]。
            batch["targets"].to(self.device, non_blocking=non_blocking),  # [B]。
        )

    def __iter__(self) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]]:  # 按既定顺序迭代 CudaBatchPrefetcher 中的训练批次。
        """按既定顺序迭代 CudaBatchPrefetcher 中的训练批次。"""
        if not self.enabled:  # 检查必需数据、配置或校验结果是否缺失。
            for batch in self.loader:  # 遍历数据加载器，逐项处理批次。
                videos, frame_mask, targets = self._move(batch)
  # 对 `(yield (batch, videos, frame_mask, targets))` 求值，以触发当前步骤需要的计算。
                yield batch, videos, frame_mask, targets  # 产出当前批次并保留迭代状态。
            return  # 返回按当前规则组织后的结果。

        iterator = iter(self.loader)
        stream = torch.cuda.Stream(device=self.device)
        next_item = None  # 先把nextitem初始化为空，后续按条件填充。

        def preload() -> None:  # 把下一批数据异步搬到 GPU，供当前批计算后立即使用。
            """把下一批数据异步搬到 GPU，供当前批计算后立即使用。"""
            nonlocal next_item  # 声明后续赋值使用外层作用域变量。
            try:  # 执行可能失败的外部读取或计算步骤。
                batch = next(iterator)
            except StopIteration:  # 捕获预期异常并执行对应恢复逻辑。
                next_item = None  # 先把nextitem初始化为空，后续按条件填充。
                return  # 返回完成上述格式转换后的结果。
            with torch.cuda.stream(stream):  # 在独立 CUDA 流中预取下一批数据。
                videos, frame_mask, targets = self._move(batch)
            next_item = (batch, videos, frame_mask, targets)

        preload()  # 异步预取下一批数据。
        while next_item is not None:  # 在终止条件出现前持续迭代。
            torch.cuda.current_stream(self.device).wait_stream(stream)  # 等待预取流完成。
            batch, videos, frame_mask, targets = next_item
            for tensor in (videos, frame_mask, targets):  # 遍历视频帧掩码标签，逐项处理张量。
                tensor.record_stream(torch.cuda.current_stream(self.device))  # 绑定当前 CUDA stream 生命周期。
            preload()  # 异步预取下一批数据。
  # 对 `(yield (batch, videos, frame_mask, targets))` 求值，以触发当前步骤需要的计算。
            yield batch, videos, frame_mask, targets  # 产出当前批次并保留迭代状态。


def loss_kwargs(config: TrainingConfig) -> dict[str, object]:  # 整理基础损失需要的温度、稳健回归和换手超参数。
    """整理基础损失需要的温度、稳健回归和换手超参数。"""
    return {  # 返回整理后的结果。
        "loss_name": config.loss_name,  # 记录损失名称。
        "rank_temperature": config.rank_temperature,  # 记录秩temperature。
        "turnover_lambda": config.turnover_lambda,  # 记录turnoverlambda。
        "huber_delta": config.huber_delta,  # 记录huberdelta。
        "smooth_l1_beta": config.smooth_l1_beta,  # 记录smoothl1beta。
        "hybrid_weight": config.hybrid_weight,  # 记录hybrid权重。
    }


def _train_standard_epoch(  # 执行该模式下的训练步骤，并返回损失与诊断信息。
    model: nn.Module,  # 接收模型。
    loader: DataLoader,  # 接收数据加载器。
    optimizer: torch.optim.Optimizer,  # 接收优化器。
    device: torch.device,  # 接收设备。
    use_prefetch: bool,  # 接收useprefetch。
    config: TrainingConfig,  # 接收配置。
    max_batches: int | None = None,  # 接收maxbatches。
    distributed_context: distributed.DistributedContext | None = None,  # 接收distributed上下文。
) -> dict[str, object]:  # 声明当前函数的返回类型。
    """执行该模式下的训练步骤，并返回损失与诊断信息。"""
    model.train()  # 启用训练模式。
    date_batch_mode = isinstance(loader.batch_sampler, TradeDateBatchSampler)
    loss_sum = torch.zeros((), dtype=torch.float64, device=device)  # 初始化损失sum。
    batch_pearson_sum = torch.zeros((), dtype=torch.float64, device=device)  # 初始化批次Pearsonsum。
    weighted_pearson = torch.zeros((), dtype=torch.float64, device=device)  # 初始化weightedPearson。
    soft_rank_ic_sum = torch.zeros((), dtype=torch.float64, device=device)  # 初始化soft排名icsum。
    sample_count = 0  # 计算样本数量。
    batch_count = 0  # 计算批次数量。
    soft_rank_batch_count = 0  # 计算soft排名批次数量。
    min_batch_size, max_batch_size = None, 0
    started = time.perf_counter()

    for batch_number, (batch, videos, frame_mask, targets) in enumerate(  # 遍历enumerate，逐项处理批次number批次视频帧掩码标签。
        CudaBatchPrefetcher(loader, device, use_prefetch), start=1
    ):
        if date_batch_mode:  # 根据当前数据状态选择处理分支。
            date_ids = batch["date_ids"]
            if int(date_ids.min()) != int(date_ids.max()):  # 按当前模式、标识或状态选择处理分支。
                raise AssertionError("日截面训练 batch 混入多个交易日")  # 关键数据契约被破坏时立即终止。
        optimizer.zero_grad(set_to_none=True)  # 清空上一批梯度。
        with autocast_context(device):  # 在自动混合精度上下文中执行前向计算。
            predictions = model(videos, frame_mask)  # 执行当前 batch 前向传播。
        predictions, targets, _ = gather_global_training_batch(
            predictions, targets, batch, distributed_context  # 传入预测、标签、批次、distributed上下文。
        )
        loss = compute_training_loss(predictions, targets, **loss_kwargs(config))  # 按实验口径计算可微损失。
        if not torch.isfinite(loss):  # 检查参与计算的数值是否全部有限。
            raise FloatingPointError("训练损失出现 NaN 或 Inf")  # 检测到 NaN、Inf 或退化数值时立即终止。
        loss.backward()  # 反向传播当前损失。
        optimizer.step()  # 更新参数或学习率状态。

        with torch.no_grad():  # 关闭梯度记录以降低验证或推理开销。
            batch_pearson = pearson_correlation(predictions, targets).double()
        current_size = len(targets)
        loss_sum += loss.detach().double()
        batch_pearson_sum += batch_pearson
        weighted_pearson += batch_pearson * current_size
        if config.loss_name in RANK_LOSS_NAMES:  # 检查当前值是否属于允许集合。
            soft_rank_ic_sum += -loss.detach().double()
            soft_rank_batch_count += 1
        sample_count += current_size
        batch_count += 1
        min_batch_size = current_size if min_batch_size is None else min(  # 计算min批次大小判定条件。
            min_batch_size, current_size  # 传入min批次大小、当前大小。
        )
        max_batch_size = max(max_batch_size, current_size)
        if max_batches is not None and batch_number >= max_batches:  # 检查当前数值是否越过业务或数值阈值。
            break  # 满足终止条件后退出当前循环。

    if not batch_count:  # 检查必需数据、配置或校验结果是否缺失。
        raise ValueError("训练 loader 没有可用 batch")  # 输入数据不满足业务或数值约束时立即终止。
    if device.type == "cuda":  # 按当前模式、标识或状态选择处理分支。
        torch.cuda.synchronize(device)  # 同步 GPU 后计时。
    batch_mean_pearson = float((batch_pearson_sum / batch_count).item())  # 计算批次均值Pearson。
    sample_weighted_pearson = float((weighted_pearson / sample_count).item())
    selected_train_pearson = (  # 构造筛选结果训练Pearson，限定后续参与计算的样本。
        batch_mean_pearson if date_batch_mode else sample_weighted_pearson
    )
    soft_rank_ic = (  # 计算soft排名ic。
        float((soft_rank_ic_sum / soft_rank_batch_count).item())
        if soft_rank_batch_count  # 根据当前数据状态选择处理分支。
        else float("nan")
    )
    return {  # 返回上述处理结果。
        "train_ic": selected_train_pearson,  # 记录训练ic。
        "train_loss": float((loss_sum / batch_count).item()),  # 记录训练损失。
        "train_batch_pearson_ic_mean": batch_mean_pearson,  # 记录训练批次Pearsonic均值。
        "train_sample_weighted_pearson_ic": sample_weighted_pearson,  # 记录训练样本weightedPearsonic。
        "train_soft_rank_ic": soft_rank_ic,  # 记录训练soft秩ic。
        "train_prediction_soft_rank_ic": soft_rank_ic,  # 记录训练预测soft秩ic。
        "train_temporal_soft_rank_ic": float("nan"),  # 记录训练时序soft秩ic。
        "train_temporal_penalty": float("nan"),  # 记录训练时序penalty。
        "train_temporal_pairs": 0,  # 记录训练时序pairs。
        "train_date_batches": batch_count if date_batch_mode else None,  # 记录训练交易日batches。
        "gradient_update_mode": "per_batch",  # 把 gradient_update_mode 映射为"per_batch"。
        "optimizer_steps": batch_count,  # 记录优化器steps。
        "train_samples": sample_count,  # 记录训练样本。
        "train_batches": batch_count,  # 记录训练batches。
        "train_batch_size_min": int(min_batch_size),  # 记录训练批次大小min。
        "train_batch_size_max": int(max_batch_size),  # 记录训练批次大小max。
        "train_seconds": time.perf_counter() - started,  # 记录训练seconds。
    }


def _train_temporal_turnover_epoch(  # 执行该模式下的训练步骤，并返回损失与诊断信息。
    model: nn.Module,  # 接收模型。
    loader: DataLoader,  # 接收数据加载器。
    optimizer: torch.optim.Optimizer,  # 接收优化器。
    device: torch.device,  # 接收设备。
    use_prefetch: bool,  # 接收useprefetch。
    config: TrainingConfig,  # 接收配置。
    max_batches: int | None = None,  # 接收maxbatches。
    distributed_context: distributed.DistributedContext | None = None,  # 接收distributed上下文。
) -> dict[str, object]:  # 声明当前函数的返回类型。
    """执行该模式下的训练步骤，并返回损失与诊断信息。"""
    sampler = loader.batch_sampler  # 创建sampler。
    if (  # 开始组合多项有效性或配置条件。
        not isinstance(sampler, TradeDateBatchSampler)
        or sampler.temporal_pairs is None
        or sampler.shuffle
    ):
        raise ValueError("时序换手损失要求按日期升序且关闭 reshuffle")  # 输入数据不满足业务或数值约束时立即终止。
    available_pairs = len(sampler.temporal_pairs)
    pair_limit = available_pairs if max_batches is None else min(  # 计算pairlimit判定条件。
        int(max_batches), available_pairs
    )
    if pair_limit < 1:  # 检查当前数值是否越过业务或数值阈值。
        raise ValueError("时序换手训练至少需要一个相邻日期对")  # 输入数据不满足业务或数值约束时立即终止。

    model.train()  # 启用训练模式。
    optimizer.zero_grad(set_to_none=True)  # 清空上一批梯度。
    loss_sum = torch.zeros((), dtype=torch.float64, device=device)  # 初始化损失sum。
    pearson_sum = torch.zeros((), dtype=torch.float64, device=device)  # 初始化Pearsonsum。
    weighted_pearson = torch.zeros((), dtype=torch.float64, device=device)  # 初始化weightedPearson。
    prediction_rank_sum = torch.zeros((), dtype=torch.float64, device=device)  # 初始化预测排名sum。
    temporal_rank_sum = torch.zeros((), dtype=torch.float64, device=device)  # 初始化时序排名sum。
    temporal_penalty_sum = torch.zeros((), dtype=torch.float64, device=device)  # 初始化时序penaltysum。
    sample_count = 0  # 计算样本数量。
    pair_count = 0  # 计算pair数量。
    min_batch_size, max_batch_size = None, 0
    previous_predictions = None  # 先把上一期预测初始化为空，后续按条件填充。
    previous_date_id = None  # 先把上一期交易日id初始化为空，后续按条件填充。
    started = time.perf_counter()

    for batch, videos, frame_mask, targets in CudaBatchPrefetcher(  # 遍历cudabatchprefetcher，逐项处理批次视频帧掩码标签。
        loader, device, use_prefetch  # 传入数据加载器、设备、useprefetch。
    ):
        date_ids = batch["date_ids"]
        if int(date_ids.min()) != int(date_ids.max()):  # 按当前模式、标识或状态选择处理分支。
            raise AssertionError("日截面训练 batch 混入多个交易日")  # 关键数据契约被破坏时立即终止。
        current_date_id = int(date_ids[0])
        with autocast_context(device):  # 在自动混合精度上下文中执行前向计算。
            current_predictions = model(videos, frame_mask)  # 执行当前 batch 前向传播。
        current_predictions, targets, _ = gather_global_training_batch(
            current_predictions, targets, batch, distributed_context  # 传入当前预测、标签、批次、distributed上下文。
        )
        if previous_predictions is None:  # 检查所需对象是否尚未初始化。
            previous_predictions = current_predictions.detach()
            previous_date_id = current_date_id
            del current_predictions  # 释放不再使用的中间引用。
            continue  # 跳过当前项并处理下一项。
        if current_date_id != previous_date_id + 1:  # 按当前模式、标识或状态选择处理分支。
            raise AssertionError(  # 关键数据契约被破坏时立即终止。
                f"时序训练日期不连续：{previous_date_id}→{current_date_id}"
            )

        pair = sampler.temporal_pair(pair_count)
        components = temporal_turnover_loss_components(  # 按实验口径计算可微损失。
            current_predictions,  # 汇入当前预测。
            targets,  # 汇入标签。
            previous_predictions,  # 汇入上一期预测。
            pair["previous_positions"],  # 汇入pair。
            pair["current_positions"],  # 汇入pair。
            config.rank_temperature,  # 汇入排名temperature。
            config.turnover_lambda,  # 汇入turnoverlambda。
        )
        loss = components["loss"]  # 计算损失。
        values = torch.stack(  # 对齐并合并值。
            [
                loss,  # 汇入损失。
                components["prediction_soft_rank_ic"],  # 汇入components。
                components["temporal_soft_rank_ic"],  # 汇入components。
                components["temporal_penalty"],  # 汇入components。
            ]
        )
        if not torch.isfinite(values).all():  # 检查参与计算的数值是否全部有限。
            raise FloatingPointError("时序换手训练损失或审计指标出现 NaN/Inf")  # 检测到 NaN、Inf 或退化数值时立即终止。
        (loss / pair_limit).backward()  # 反向传播当前损失。

        with torch.no_grad():  # 关闭梯度记录以降低验证或推理开销。
            batch_pearson = pearson_correlation(current_predictions, targets).double()
        current_size = len(targets)
        loss_sum += loss.detach().double()
        pearson_sum += batch_pearson
        weighted_pearson += batch_pearson * current_size
        prediction_rank_sum += components["prediction_soft_rank_ic"].detach().double()
        temporal_rank_sum += components["temporal_soft_rank_ic"].detach().double()
        temporal_penalty_sum += components["temporal_penalty"].detach().double()
        sample_count += current_size
        pair_count += 1
        min_batch_size = current_size if min_batch_size is None else min(  # 计算min批次大小判定条件。
            min_batch_size, current_size  # 传入min批次大小、当前大小。
        )
        max_batch_size = max(max_batch_size, current_size)
        previous_predictions = current_predictions.detach()
        previous_date_id = current_date_id
        del current_predictions, components, loss, values  # 释放不再使用的中间引用。
        if pair_count >= pair_limit:  # 检查当前数值是否越过业务或数值阈值。
            break  # 满足终止条件后退出当前循环。

    if pair_count != pair_limit:  # 按当前模式、标识或状态选择处理分支。
        raise AssertionError(f"时序日期对覆盖不完整：{pair_count}/{pair_limit}")  # 关键数据契约被破坏时立即终止。
    optimizer.step()  # 更新参数或学习率状态。
    if device.type == "cuda":  # 按当前模式、标识或状态选择处理分支。
        torch.cuda.synchronize(device)  # 同步 GPU 后计时。
    mean_pearson = float((pearson_sum / pair_count).item())  # 计算均值Pearson。
    weighted_mean_pearson = float((weighted_pearson / sample_count).item())  # 计算weighted均值Pearson。
    prediction_soft_rank = float((prediction_rank_sum / pair_count).item())  # 计算预测soft排名。
    temporal_soft_rank = float((temporal_rank_sum / pair_count).item())  # 计算时序soft排名。
    temporal_penalty = float((temporal_penalty_sum / pair_count).item())
    return {  # 返回上述处理结果。
        "train_ic": mean_pearson,  # 记录训练ic。
        "train_loss": float((loss_sum / pair_count).item()),  # 记录训练损失。
        "train_batch_pearson_ic_mean": mean_pearson,  # 记录训练批次Pearsonic均值。
        "train_sample_weighted_pearson_ic": weighted_mean_pearson,  # 记录训练样本weightedPearsonic。
        "train_soft_rank_ic": prediction_soft_rank,  # 记录训练soft秩ic。
        "train_prediction_soft_rank_ic": prediction_soft_rank,  # 记录训练预测soft秩ic。
        "train_temporal_soft_rank_ic": temporal_soft_rank,  # 记录训练时序soft秩ic。
        "train_temporal_penalty": temporal_penalty,  # 记录训练时序penalty。
        "train_temporal_pairs": pair_count,  # 记录训练时序pairs。
        "train_date_batches": pair_count + 1,  # 记录训练交易日batches。
        "train_forward_passes": pair_count + 1,  # 记录训练forwardpasses。
        "temporal_anchor_gradient": "stop_gradient",  # 把 temporal_anchor_gradient 映射为"stop_gradient"。
        "gradient_update_mode": "once_per_epoch",  # 把 gradient_update_mode 映射为"once_per_epoch"。
        "optimizer_steps": 1,  # 记录优化器steps。
        "train_samples": sample_count,  # 记录训练样本。
        "train_batches": pair_count,  # 记录训练batches。
        "train_batch_size_min": int(min_batch_size),  # 记录训练批次大小min。
        "train_batch_size_max": int(max_batch_size),  # 记录训练批次大小max。
        "train_seconds": time.perf_counter() - started,  # 记录训练seconds。
    }


def train_one_epoch(  # 训练一个完整 epoch，并返回损失与批次诊断。
    model: nn.Module,  # 接收模型。
    loader: DataLoader,  # 接收数据加载器。
    optimizer: torch.optim.Optimizer,  # 接收优化器。
    device: torch.device,  # 接收设备。
    use_prefetch: bool,  # 接收useprefetch。
    config: TrainingConfig,  # 接收配置。
    max_batches: int | None = None,  # 接收maxbatches。
    distributed_context: distributed.DistributedContext | None = None,  # 接收distributed上下文。
) -> dict[str, object]:  # 声明当前函数的返回类型。
    """训练一个完整 epoch，并返回损失与批次诊断。"""
    if config.loss_name == TEMPORAL_TURNOVER_LOSS:  # 按当前模式、标识或状态选择处理分支。
        return _train_temporal_turnover_epoch(  # 返回训练一个完整 epoch，并返回损失与批次诊断。
            model,  # 汇入模型。
            loader,  # 汇入数据加载器。
            optimizer,  # 汇入优化器。
            device,  # 汇入设备。
            use_prefetch,  # 汇入useprefetch。
            config,  # 汇入配置。
            max_batches=max_batches,  # 限制调试运行处理的最大批次数。
            distributed_context=distributed_context,  # 配置distributed上下文。
        )
    return _train_standard_epoch(  # 返回训练一个完整 epoch，并返回损失与批次诊断。
        model,  # 汇入模型。
        loader,  # 汇入数据加载器。
        optimizer,  # 汇入优化器。
        device,  # 汇入设备。
        use_prefetch,  # 汇入useprefetch。
        config,  # 汇入配置。
        max_batches=max_batches,  # 限制调试运行处理的最大批次数。
        distributed_context=distributed_context,  # 配置distributed上下文。
    )


_INSTRUMENT_IDS: np.ndarray | None = None  # 接收股票ids。


def instrument_ids() -> np.ndarray:  # 根据样本行号返回对应股票代码。
    """根据样本行号返回对应股票代码。"""
    global _INSTRUMENT_IDS  # 声明后续赋值使用外层作用域变量。
    if _INSTRUMENT_IDS is None:  # 检查所需对象是否尚未初始化。
        _INSTRUMENT_IDS = pd.read_parquet(  # 设置股票ids。
            SAMPLE_INDEX_PATH, columns=["instrument_id"]  # 定位样本索引路径。
        )["instrument_id"].to_numpy(copy=False)
    return _INSTRUMENT_IDS  # 返回股票ids。


# 使用 `torch.no_grad()` 包装 `evaluate_model`，并定义其可复用的处理逻辑。
@torch.no_grad()  # 验证或推理阶段关闭梯度记录。
def evaluate_model(  # 在验证集上评估模型，并汇总逐日 IC 与损失诊断。
    model: nn.Module,  # 接收模型。
    loader: DataLoader,  # 接收数据加载器。
    device: torch.device,  # 接收设备。
    use_prefetch: bool,  # 接收useprefetch。
    validation_metric: str,  # 接收验证指标。
    max_batches: int | None = None,  # 接收maxbatches。
    distributed_context: distributed.DistributedContext | None = None,  # 接收distributed上下文。
) -> dict[str, object]:  # 声明当前函数的返回类型。
    """在验证集上评估模型，并汇总逐日 IC 与损失诊断。"""
    local_model = inference_model(model)  # 关闭 dropout 等训练行为。
    local_model.eval()  # 切换模型到验证或推理模式。
    prediction_parts: list[np.ndarray] = []  # 接收预测parts。
    target_parts: list[np.ndarray] = []  # 接收标签parts。
    date_parts: list[np.ndarray] = []  # 接收交易日parts。
    sample_index_parts: list[np.ndarray] = []  # 接收样本索引parts。
    started = time.perf_counter()
    for batch_number, (batch, videos, frame_mask, targets) in enumerate(  # 遍历enumerate，逐项处理批次number批次视频帧掩码标签。
        CudaBatchPrefetcher(loader, device, use_prefetch), start=1
    ):
        with autocast_context(device):  # 在自动混合精度上下文中执行前向计算。
            predictions = local_model(videos, frame_mask)  # 执行当前 batch 前向传播。
        prediction_parts.append(predictions.float().cpu().numpy().astype(np.float64))  # 把当前结果追加到汇总列表。
        target_parts.append(targets.float().cpu().numpy().astype(np.float64))  # 把当前结果追加到汇总列表。
        date_parts.append(batch["date_ids"].numpy().astype(np.int64, copy=False))  # 把当前结果追加到汇总列表。
        sample_index_parts.append(  # 把当前结果追加到汇总列表。
            batch["sample_indices"].numpy().astype(np.int64, copy=False)
        )
        if max_batches is not None and batch_number >= max_batches:  # 检查当前数值是否越过业务或数值阈值。
            break  # 满足终止条件后退出当前循环。
    if not prediction_parts:  # 检查必需数据、配置或校验结果是否缺失。
        raise ValueError("验证 loader 没有可用 batch")  # 输入数据不满足业务或数值约束时立即终止。
    if device.type == "cuda":  # 按当前模式、标识或状态选择处理分支。
        torch.cuda.synchronize(device)  # 同步 GPU 后计时。
    predictions_tensor = torch.as_tensor(  # 转换并统一预测张量的数据表示。
        np.concatenate(prediction_parts), device=device, dtype=torch.float64
    )
    targets_tensor = torch.as_tensor(  # 转换并统一标签张量的数据表示。
        np.concatenate(target_parts), device=device, dtype=torch.float64
    )
    dates_tensor = torch.as_tensor(  # 转换并统一交易日张量的数据表示。
        np.concatenate(date_parts), device=device, dtype=torch.int64
    )
    indices_tensor = torch.as_tensor(  # 转换并统一索引张量的数据表示。
        np.concatenate(sample_index_parts), device=device, dtype=torch.int64
    )
    predictions_tensor = gather_global_inference_tensor(
        predictions_tensor, distributed_context  # 传入预测张量、distributed上下文。
    )
    targets_tensor = gather_global_inference_tensor(targets_tensor, distributed_context)
    dates_tensor = gather_global_inference_tensor(dates_tensor, distributed_context)
    indices_tensor = gather_global_inference_tensor(indices_tensor, distributed_context)
    order = torch.argsort(indices_tensor)  # 计算排序索引以固定样本顺序。
    selected_indices = indices_tensor[order].cpu().numpy()  # 构造筛选结果索引，限定后续参与计算的样本。
    result = summarize_validation_arrays(
        predictions_tensor[order].cpu().numpy(),
        targets_tensor[order].cpu().numpy(),
        dates_tensor[order].cpu().numpy(),
        validation_metric,  # 汇入验证指标。
        instrument_ids=instrument_ids()[selected_indices],  # 配置股票ids。
    )
    result["validation_seconds"] = time.perf_counter() - started
    return result  # 返回结果。


# 使用 `torch.no_grad()` 包装 `predict_model`，并定义其可复用的处理逻辑。
@torch.no_grad()  # 验证或推理阶段关闭梯度记录。
def predict_model(  # 在预测集上生成与样本行号严格对齐的模型分数。
    model: nn.Module,  # 接收模型。
    loader: DataLoader,  # 接收数据加载器。
    device: torch.device,  # 接收设备。
    use_prefetch: bool,  # 接收useprefetch。
    max_batches: int | None = None,  # 接收maxbatches。
    distributed_context: distributed.DistributedContext | None = None,  # 接收distributed上下文。
) -> pd.DataFrame:  # 声明当前函数的返回类型。
    """在预测集上生成与样本行号严格对齐的模型分数。"""
    local_model = inference_model(model)  # 关闭 dropout 等训练行为。
    local_model.eval()  # 切换模型到验证或推理模式。
    sample_parts: list[np.ndarray] = []  # 接收样本parts。
    prediction_parts: list[np.ndarray] = []  # 接收预测parts。
    for batch_number, (batch, videos, frame_mask, _) in enumerate(  # 遍历enumerate，逐项处理批次number批次视频帧掩码。
        CudaBatchPrefetcher(loader, device, use_prefetch), start=1
    ):
        with autocast_context(device):  # 在自动混合精度上下文中执行前向计算。
            predictions = local_model(videos, frame_mask)  # 执行当前 batch 前向传播。
        sample_parts.append(batch["sample_indices"].numpy())  # 把当前结果追加到汇总列表。
        prediction_parts.append(predictions.float().cpu().numpy())  # 把当前结果追加到汇总列表。
        if max_batches is not None and batch_number >= max_batches:  # 检查当前数值是否越过业务或数值阈值。
            break  # 满足终止条件后退出当前循环。
    if sample_parts:  # 根据当前数据状态选择处理分支。
        local_indices = torch.as_tensor(  # 转换并统一本地索引的数据表示。
            np.concatenate(sample_parts), device=device, dtype=torch.int64
        )
        local_predictions = torch.as_tensor(  # 转换并统一本地预测的数据表示。
            np.concatenate(prediction_parts), device=device, dtype=torch.float32
        )
    else:  # 处理上述条件均不成立的情况。
        local_indices = torch.empty(0, device=device, dtype=torch.int64)  # 初始化本地索引。
        local_predictions = torch.empty(0, device=device, dtype=torch.float32)  # 初始化本地预测。
    global_indices = gather_global_inference_tensor(local_indices, distributed_context)
    global_predictions = gather_global_inference_tensor(
        local_predictions, distributed_context  # 传入本地预测、distributed上下文。
    )
    order = torch.argsort(global_indices)  # [N_global]。
    return pd.DataFrame(  # 返回在预测集上生成与样本行号严格对齐的模型分数。
        {
            "sample_index": global_indices[order].cpu().numpy(),  # 记录样本索引。
            "prediction": global_predictions[order].cpu().numpy().astype(np.float32),  # 记录预测。
        }
    )


def run_paths(run_name: str) -> tuple[Path, Path, Path]:  # 返回指定年度和随机种子的模型、预测与日志路径。
    """返回指定年度和随机种子的模型、预测与日志路径。"""
    model_root = VIVIT_DIR / "models" / "runs" / run_name  # 构造模型根目录。
    prediction_root = DATA_DIR / "predictions" / "runs" / run_name  # 构造预测根目录。
    quality_root = DATA_DIR / "quality" / "runs" / run_name  # 构造质量根目录。
    for path in (model_root, prediction_root, quality_root):  # 遍历模型根目录预测根目录质量根目录，逐项处理路径。
        path.mkdir(parents=True, exist_ok=True)  # 确保目标目录存在。
    return model_root, prediction_root, quality_root  # 返回指定年度和随机种子的模型、预测与日志路径。


def build_optimizer(  # 创建优化器，并按配置启用 fused Adam。
    model: nn.Module,  # 接收模型。
    config: TrainingConfig,  # 接收配置。
    fused: bool,  # 接收fused。
    device: torch.device,  # 接收设备。
) -> torch.optim.Adam:  # 声明当前函数的返回类型。
    """创建优化器，并按配置启用 fused Adam。"""
    return torch.optim.Adam(  # 返回创建优化器，并按配置启用 fused Adam。
        model.parameters(),
        lr=config.learning_rate,  # 设置优化器学习率。
        betas=(config.adam_beta1, config.adam_beta2),  # 配置betas。
        eps=config.adam_epsilon,  # 设置数值稳定项。
        weight_decay=config.weight_decay,  # 设置权重衰减强度。
        fused=fused and device.type == "cuda",  # 配置fused。
    )


def training_config_from_args(args: argparse.Namespace) -> TrainingConfig:  # 把命令行参数转换为不可变训练配置。
    """把命令行参数转换为不可变训练配置。"""
    return TrainingConfig(  # 返回完成上述格式转换后的结果。
        train_batch_mode=args.train_batch_mode,  # 配置训练批次模式。
        batch_size=args.batch_size,  # 设置每个优化批次的样本数。
        inference_batch_size=args.inference_batch_size,  # 配置inference批次大小。
        loss_name=args.loss_name,  # 配置损失名称。
        rank_temperature=args.rank_temperature,  # 配置排名temperature。
        turnover_lambda=args.turnover_lambda,  # 配置turnoverlambda。
        huber_delta=args.huber_delta,  # 配置huberdelta。
        smooth_l1_beta=args.smooth_l1_beta,  # 配置smoothl1beta。
        hybrid_weight=args.hybrid_weight,  # 配置hybrid权重。
        validation_metric=args.validation_metric,  # 配置验证指标。
        learning_rate=args.learning_rate,  # 配置learning比率。
        max_epochs=args.max_epochs,  # 配置maxepochs。
        early_stopping_patience=args.patience,  # 配置earlystoppingpatience。
    )


def training_payload(  # 整理训练配置，供 checkpoint 判断是否可以继续使用。
    args: argparse.Namespace,  # 接收args。
    year: int,  # 接收年份。
    seed: int,  # 接收随机种子。
    batch_summary: dict[str, object],
) -> dict[str, object]:  # 声明当前函数的返回类型。
    """整理训练配置，供 checkpoint 判断是否可以继续使用。"""
    sequence_manifest = require_aligned_sequence_store()
    config = training_config_from_args(args)
    validate_training_config(config, args.compile_mode)
    temporal_mode = config.loss_name == TEMPORAL_TURNOVER_LOSS  # 配置时序模式。
    date_batches = config.train_batch_mode == "trade_date"  # 配置交易日batches。
    return {  # 返回整理后的结果。
        "schema_version": 2,  # 记录schema版本。
        "forecast_year": year,  # 记录forecast年份。
        "seed": seed,  # 记录随机种子。
        "sequence_signature": sequence_manifest["signature"],  # 记录sequence签名，用于来源一致性校验。
        "split_source": "ViT/data/feature_store/rolling_splits",  # 把 split_source 映射为"ViT/data/feature_store/rolling_splits"。
        "split_policy": "exact_vit_train_validation_prediction_indices",  # 把 split_policy 映射为"exact_vit_train_validation_prediction_indices"。
        "padding_policy": "zero_fill_missing_frames_with_tubelet_coverage_mask",  # 把 padding_policy 映射为"zero_fill_missing_frames_with_tubelet_coverage_mask"。
        "model_config": asdict(MODEL_CONFIG),  # 记录模型配置。
        "training_config": asdict(config),  # 记录训练配置。
        "training_batch_summary": batch_summary,  # 记录训练批次汇总。
        "execution_config": {  # 记录执行配置。
            "route": (  # 记录route。
                "DDP_single_node_task_worker"
                if getattr(args, "execution_route", "a2") == "ddp"  # 按当前模式、标识或状态选择处理分支。
                else "A2_independent_task_worker"
            ),
            **(
                {"world_size": int(args.ddp_world_size)}
                if getattr(args, "execution_route", "a2") == "ddp" else {}  # 按当前模式、标识或状态选择处理分支。
            ),
            "train_batch_mode": config.train_batch_mode,  # 记录训练批次模式。
            "batch_size_per_task": "full_trade_date" if date_batches else config.batch_size,  # 记录批次大小pertask。
            "fixed_batch_size": config.batch_size,  # 记录fixed批次大小。
            "inference_batch_size": config.inference_batch_size,  # 记录inference批次大小。
            "reshuffle_each_epoch": not temporal_mode,  # 记录reshuffleeach轮次。
            "gradient_update_mode": "once_per_epoch" if temporal_mode else "per_batch",  # 把 gradient_update_mode 映射为"once_per_epoch" if temporal_mode else "per_batch"。
            "data_loader_workers": args.workers_per_task,  # 记录数据数据加载器workers。
            "torch_compile": bool(args.compile),  # 记录torchcompile。
            "compile_mode": args.compile_mode,  # 记录compile模式。
            "compile_dynamic_batch": bool(args.compile) and date_batches,  # 记录compiledynamic批次。
            "fused_adam": bool(args.fused_adam),  # 记录fusedadam。
            "cuda_prefetch": bool(args.cuda_prefetch),  # 记录cudaprefetch。
            "precision": "bf16_autocast_with_fp32_loss",  # 把 precision 映射为"bf16_autocast_with_fp32_loss"。
        },
        "loss": config.loss_name,  # 记录损失。
        "loss_parameters": {  # 记录损失parameters。
            "rank_temperature": config.rank_temperature,  # 记录秩temperature。
            "turnover_lambda": config.turnover_lambda,  # 记录turnoverlambda。
            "huber_delta": config.huber_delta,  # 记录huberdelta。
            "smooth_l1_beta": config.smooth_l1_beta,  # 记录smoothl1beta。
            "hybrid_weight": config.hybrid_weight,  # 记录hybrid权重。
        },
        "validation_metric": config.validation_metric,  # 记录验证指标。
        "validation_audit_metrics": [  # 记录验证audit指标。
            "daily_mean_spearman_ic",
            "daily_mean_pearson_ic",
            "pooled_pearson_ic",
            "temporal_spearman_rank_ic",
            "temporal_rank_penalty",
        ],
    }


def load_canonical_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:  # 把规范化参数字典加载到原始或包装模型。
    """把规范化参数字典加载到原始或包装模型。"""
    unwrap_model(model).load_state_dict(state)


def load_best_model(  # 加载最佳验证 checkpoint，并恢复可用于预测的模型。
    path: Path,  # 接收路径。
    device: torch.device,  # 接收设备。
    use_compile: bool,  # 接收usecompile。
    compile_mode: str,  # 接收compile模式。
    train_batch_mode: str,  # 接收训练批次模式。
) -> nn.Module:  # 声明当前函数的返回类型。
    """加载最佳验证 checkpoint，并恢复可用于预测的模型。"""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_model(device, use_compile, compile_mode, train_batch_mode)  # 创建模型并部署到设备。
    load_canonical_state(model, checkpoint["model_state"])  # 恢复模型权重。
    model.eval()  # 关闭 dropout 等训练行为。
    return model  # 返回模型。


def train_single_run(  # 训练或恢复一个年度、一个随机种子。
    args: argparse.Namespace,  # 接收args。
    distributed_context: distributed.DistributedContext | None = None,  # 接收distributed上下文。
) -> dict[str, object]:  # 声明当前函数的返回类型。
    """训练或恢复一个年度、一个随机种子。"""

    year, seed = args.worker_year, args.worker_seed
    if year is None or seed is None:  # 检查所需对象是否尚未初始化。
        raise ValueError("worker 需要 --worker-year 和 --worker-seed")  # 输入数据不满足业务或数值约束时立即终止。
    require_aligned_sequence_store()
    config = training_config_from_args(args)
    validate_training_config(config, args.compile_mode)
    primary = distributed_context is None or distributed_context.is_primary  # 计算primary判定条件。
    device = (
        distributed_context.device  # 汇入distributed上下文设备。
        if distributed_context is not None  # 检查必需数据、配置或校验结果是否缺失。
        else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    pin_memory = device.type == "cuda"  # 控制是否锁页以加速 CPU 到 GPU 传输。
    set_random_seed(seed + (distributed_context.rank if distributed_context else 0))
    if distributed_context is None:  # 检查所需对象是否尚未初始化。
        train_loader, validation_loader, prediction_loader = make_data_loaders(  # 创建训练数据加载器。
            year,  # 汇入年份。
            seed,  # 汇入随机种子。
            config.batch_size,  # 汇入批次大小。
            config.inference_batch_size,  # 汇入inference批次大小。
            config.train_batch_mode,  # 汇入训练批次模式。
            config.loss_name,  # 汇入损失名称。
            args.workers_per_task,  # 汇入workerspertask。
            pin_memory,  # 汇入pinmemory。
        )
        batch_summary = loader_batch_summary(train_loader)
    else:  # 处理上述条件均不成立的情况。
        train_loader, validation_loader, prediction_loader, batch_summary = (  # 创建训练数据加载器。
            make_distributed_data_loaders(
                year,  # 汇入年份。
                seed,
                config,
                args.workers_per_task,
                pin_memory,
                distributed_context,
            )
        )

    model_root, _, _ = run_paths(args.run_name)  # 构造模型根目录。
    run_directory = model_root / f"forecast_{year}" / f"seed_{seed}"
    if primary:  # 根据当前数据状态选择处理分支。
        run_directory.mkdir(parents=True, exist_ok=True)  # 确保目标目录存在。
    if distributed_context is not None:  # 检查必需数据、配置或校验结果是否缺失。
        dist.barrier()
    best_path = run_directory / "best.pt"  # 构造最优路径。
    last_path = run_directory / "last.pt"  # 构造last路径。
    history_path = run_directory / "history.parquet"  # 构造历史记录路径。
    prediction_path = run_directory / "prediction.parquet"  # 构造预测路径。
    completed_path = run_directory / "completed.json"  # 构造completed路径。

    payload = training_payload(args, year, seed, batch_summary)
    run_signature = stable_hash(payload)  # 生成数据复用签名。
    if primary:  # 原子落盘，避免半成品。
        atomic_write_json(
        {
            "signature": stable_hash(  # 记录签名。
                {key: value for key, value in payload.items() if key != "seed"}
            ),
            "payload": {  # 记录载荷。
                key: value for key, value in payload.items() if key != "seed"
            },
        },
        model_root / f"forecast_{year}" / "model_config.json",
        )
    if distributed_context is not None:  # 检查必需数据、配置或校验结果是否缺失。
        dist.barrier()
    if completed_path.exists() and best_path.exists() and not args.overwrite:  # 检查目标文件或目录是否满足存在性要求。
        completed = json.loads(completed_path.read_text(encoding="utf-8"))  # 读取completed。
        if completed.get("run_signature") == run_signature:  # 按当前模式、标识或状态选择处理分支。
            print(f"[{year}/seed={seed}] 已完成，直接复用。", flush=True)  # 输出当前任务的进度或诊断信息。
            if prediction_loader is not None and not prediction_path.exists():  # 检查目标文件或目录是否满足存在性要求。
                model = load_best_model(  # 创建模型。
                    best_path,
                    device,
                    bool(args.compile),
                    args.compile_mode,
                    config.train_batch_mode,
                )
                predictions_frame = predict_model(
                    model,
                    prediction_loader,
                    device,
                    bool(args.cuda_prefetch),
                    distributed_context=distributed_context,  # 配置distributed上下文。
                )
                if primary:  # 根据当前数据状态选择处理分支。
                    atomic_write_parquet(predictions_frame, prediction_path)
                if distributed_context is not None:  # 检查必需数据、配置或校验结果是否缺失。
                    dist.barrier()
            return completed["summary"]  # 返回训练或恢复一个年度、一个随机种子。

    model = build_model(  # 创建模型并部署到设备。
        device, bool(args.compile), args.compile_mode, config.train_batch_mode
    )
    if distributed_context is not None:  # 检查必需数据、配置或校验结果是否缺失。
        model = distributed.wrap_model(model, distributed_context)  # 创建模型。
    optimizer = build_optimizer(model, config, bool(args.fused_adam), device)  # 按训练配置创建优化器。
    start_epoch = 1
    best_epoch = 0
    best_validation_ic = -np.inf
    best_validation_metrics: dict[str, float] = {}  # 计算最优验证指标。
    epochs_without_improvement = 0
    history: list[dict[str, object]] = []  # 初始化历史记录容器。

    if last_path.exists() and not args.overwrite:  # 检查目标文件或目录是否满足存在性要求。
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        if checkpoint.get("run_signature") != run_signature:  # 按当前模式、标识或状态选择处理分支。
            raise ValueError(f"{last_path} 与当前配置不一致；请使用新 run name")  # 输入数据不满足业务或数值约束时立即终止。
        load_canonical_state(model, checkpoint["model_state"])  # 恢复模型权重。
        optimizer.load_state_dict(checkpoint["optimizer_state"])  # 恢复优化器续训状态。
        if distributed_context is not None:  # 检查必需数据、配置或校验结果是否缺失。
            states = checkpoint.get("rng_state_by_rank")
            if states is None or len(states) != distributed_context.world_size:  # 检查所需对象是否尚未初始化。
                raise ValueError("DDP checkpoint 缺少匹配 world size 的逐 rank RNG 状态")  # 输入数据不满足业务或数值约束时立即终止。
            distributed.restore_rng_state(
                states[distributed_context.rank], device
            )
        else:  # 处理上述条件均不成立的情况。
            random.setstate(checkpoint["python_random_state"])
            np.random.set_state(checkpoint["numpy_random_state"])
            torch.set_rng_state(checkpoint["torch_rng_state"])
            if device.type == "cuda" and checkpoint["cuda_rng_state"] is not None:  # 按当前模式、标识或状态选择处理分支。
                torch.cuda.set_rng_state(checkpoint["cuda_rng_state"], device)
        train_loader.generator.set_state(checkpoint["train_generator_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_epoch = int(checkpoint["best_epoch"])
        best_validation_ic = float(checkpoint["best_validation_ic"])
        best_validation_metrics = {  # 计算最优验证指标。
            key: float(value)  # 接收键。
            for key, value in checkpoint.get("best_validation_metrics", {}).items()  # 遍历检查点get最优验证指标items，逐项处理键值。
        }
        epochs_without_improvement = int(checkpoint["epochs_without_improvement"])
        if epochs_without_improvement >= config.early_stopping_patience:  # 检查当前数值是否越过业务或数值阈值。
            start_epoch = config.max_epochs + 1
        if history_path.exists():  # 检查目标文件或目录是否满足存在性要求。
            history = pd.read_parquet(history_path).to_dict("records")  # 读取历史记录。
        print(f"[{year}/seed={seed}] 从 epoch {start_epoch} 续训。", flush=True)  # 输出当前任务的进度或诊断信息。

    started = time.perf_counter()
    stopped_early = epochs_without_improvement >= config.early_stopping_patience  # 配置stoppedearly。
    for epoch in range(start_epoch, config.max_epochs + 1):  # 按给定步长遍历样本区间，更新轮次。
        epoch_started = time.perf_counter()
        if hasattr(train_loader.batch_sampler, "set_epoch"):  # 同步 epoch，保持采样可复现。
            train_loader.batch_sampler.set_epoch(epoch)  # 同步当前轮次以确定分布式采样顺序。
        else:  # 处理上述条件均不成立的情况。
            set_train_epoch(train_loader, epoch)
        train_metrics = train_one_epoch(  # 完成一个训练 epoch。
            model,
            train_loader,
            optimizer,
            device,
            bool(args.cuda_prefetch),
            config,
            distributed_context=distributed_context,  # 配置distributed上下文。
        )
        validation_metrics = evaluate_model(  # 无梯度计算验证指标。
            model,
            validation_loader,
            device,
            bool(args.cuda_prefetch),
            config.validation_metric,
            distributed_context=distributed_context,  # 配置distributed上下文。
        )
        validation_ic = float(validation_metrics["validation_ic"])
        improved = validation_ic > (  # 计算improved判定条件。
            best_validation_ic + config.improvement_tolerance
        )
        if improved:  # 根据当前数据状态选择处理分支。
            best_epoch = epoch
            best_validation_ic = validation_ic
            best_validation_metrics = {  # 计算最优验证指标。
                "validation_daily_rank_ic": float(  # 记录验证日度秩ic。
                    validation_metrics["validation_daily_rank_ic"]
                ),
                "validation_daily_pearson_ic": float(  # 记录验证日度Pearsonic。
                    validation_metrics["validation_daily_pearson_ic"]
                ),
                "validation_pooled_ic": float(  # 记录验证pooledic。
                    validation_metrics["validation_pooled_ic"]
                ),
                "validation_temporal_rank_ic": float(  # 记录验证时序秩ic。
                    validation_metrics["validation_temporal_rank_ic"]
                ),
                "validation_temporal_penalty": float(  # 记录验证时序penalty。
                    validation_metrics["validation_temporal_penalty"]
                ),
            }
            epochs_without_improvement = 0
        else:  # 处理上述条件均不成立的情况。
            epochs_without_improvement += 1

        history.append(  # 把当前结果追加到汇总列表。
            {
                "forecast_year": year,  # 记录forecast年份。
                "seed": seed,  # 记录随机种子。
                "epoch": epoch,  # 记录轮次。
                "train_batch_mode": config.train_batch_mode,  # 记录训练批次模式。
                "loss_name": config.loss_name,  # 记录损失名称。
                "validation_metric": config.validation_metric,  # 记录验证指标。
                **train_metrics,
                **validation_metrics,
                "best_validation_ic": best_validation_ic,  # 记录最优验证ic。
                "epochs_without_improvement": epochs_without_improvement,  # 记录epochswithoutimprovement。
                "epoch_seconds": time.perf_counter() - epoch_started,  # 记录轮次seconds。
            }
        )
        if primary:  # 原子落盘，避免半成品。
            atomic_write_parquet(pd.DataFrame(history), history_path)
        rng_state_by_rank = (  # 计算rng状态by排名。
            distributed.gather_rng_states(device) if distributed_context else None
        )
        checkpoint = {  # 初始化检查点容器。
            "model_state": unwrap_model(model).state_dict(),  # 记录模型状态。
            "optimizer_state": optimizer.state_dict(),  # 记录优化器状态。
            "python_random_state": random.getstate(),  # 记录pythonrandom状态。
            "numpy_random_state": np.random.get_state(),  # 记录numpyrandom状态。
            "torch_rng_state": torch.get_rng_state(),  # 记录torchrng状态。
            "cuda_rng_state": (  # 记录cudarng状态。
                torch.cuda.get_rng_state(device) if device.type == "cuda" else None
            ),
            "train_generator_state": train_loader.generator.get_state(),  # 记录训练generator状态。
            "epoch": epoch,  # 记录轮次。
            "next_trade_date_sampler_epoch": epoch + 1,  # 记录nexttrade交易日sampler轮次。
            "best_epoch": best_epoch,  # 记录最优轮次。
            "best_validation_ic": best_validation_ic,  # 记录最优验证ic。
            "best_validation_metrics": best_validation_metrics,  # 记录最优验证指标。
            "epochs_without_improvement": epochs_without_improvement,  # 记录epochswithoutimprovement。
            "run_signature": run_signature,  # 记录运行签名，用于来源一致性校验。
            "run_payload": payload,  # 记录运行载荷。
            "model_config": asdict(MODEL_CONFIG),  # 记录模型配置。
            "training_config": asdict(config),  # 记录训练配置。
        }
        if rng_state_by_rank is not None:  # 检查必需数据、配置或校验结果是否缺失。
            checkpoint["rng_state_by_rank"] = rng_state_by_rank  # 计算检查点。
        if primary:  # 根据当前数据状态选择处理分支。
            atomic_save_checkpoint(checkpoint, last_path)
        if primary and improved:  # 根据当前数据状态选择处理分支。
            atomic_save_checkpoint(checkpoint, best_path)
        if distributed_context is not None:  # 检查必需数据、配置或校验结果是否缺失。
            dist.barrier()
        print(  # 输出当前任务的进度或诊断信息。
            f"[{year}/seed={seed}] epoch={epoch:03d} "
            f"trainLoss={train_metrics['train_loss']:.6f} "
            f"trainPearson={train_metrics['train_ic']:.6f} "
            f"validRank={validation_metrics['validation_daily_rank_ic']:.6f} "
            f"validPearson={validation_metrics['validation_daily_pearson_ic']:.6f} "
            f"validPooled={validation_metrics['validation_pooled_ic']:.6f} "
            f"selected={validation_ic:.6f} best={best_validation_ic:.6f} "
            f"patience={epochs_without_improvement}/{config.early_stopping_patience}",
            flush=True,  # 配置flush。
        )
        if epochs_without_improvement >= config.early_stopping_patience:  # 检查当前数值是否越过业务或数值阈值。
            stopped_early = True
            break  # 满足终止条件后退出当前循环。

    if not best_path.exists():  # 检查目标文件或目录是否满足存在性要求。
        raise FileNotFoundError(f"未生成最佳 checkpoint：{best_path}")  # 目标文件不存在时立即终止。
    if prediction_loader is not None:  # 检查必需数据、配置或校验结果是否缺失。
        best_model = load_best_model(  # 创建最优模型。
            best_path,
            device,
            bool(args.compile),
            args.compile_mode,
            config.train_batch_mode,
        )
        predictions_frame = predict_model(
            best_model,
            prediction_loader,
            device,
            bool(args.cuda_prefetch),
            distributed_context=distributed_context,  # 配置distributed上下文。
        )
        if primary:  # 根据当前数据状态选择处理分支。
            atomic_write_parquet(predictions_frame, prediction_path)
        if distributed_context is not None:  # 检查必需数据、配置或校验结果是否缺失。
            dist.barrier()
    summary = {  # 初始化汇总容器。
        "forecast_year": year,  # 记录forecast年份。
        "seed": seed,  # 记录随机种子。
        "best_epoch": best_epoch,  # 记录最优轮次。
        "best_validation_ic": best_validation_ic,  # 记录最优验证ic。
        "best_validation_rank_ic": best_validation_metrics.get(  # 记录最优验证秩ic。
            "validation_daily_rank_ic"
        ),
        "best_validation_daily_pearson_ic": best_validation_metrics.get(  # 记录最优验证日度Pearsonic。
            "validation_daily_pearson_ic"
        ),
        "best_validation_pooled_ic": best_validation_metrics.get(  # 记录最优验证pooledic。
            "validation_pooled_ic"
        ),
        "last_epoch": int(history[-1]["epoch"]),  # 记录最后一个轮次。
        "stopped_early": stopped_early,  # 记录stoppedearly。
        "elapsed_seconds": time.perf_counter() - started,  # 记录elapsedseconds。
        "physical_gpu": args.worker_physical_gpu,  # 记录physicalgpu。
        "execution_route": getattr(args, "execution_route", "a2"),  # 记录执行route。
        "world_size": int(getattr(args, "ddp_world_size", 1)),  # 记录world大小。
        "run_name": args.run_name,  # 记录运行名称。
        "train_batch_mode": config.train_batch_mode,  # 记录训练批次模式。
        "loss_name": config.loss_name,  # 记录损失名称。
        "validation_metric": config.validation_metric,  # 记录验证指标。
        "prediction_sample_count": len(load_sequence_split(year)["prediction_indices"]),  # 记录预测样本数量。
        "status": "completed",  # 把 status 映射为"completed"。
    }
    if primary:  # 原子落盘，避免半成品。
        atomic_write_json(
            {"run_signature": run_signature, "summary": summary}, completed_path
        )
    if distributed_context is not None:  # 检查必需数据、配置或校验结果是否缺失。
        dist.barrier()
    return summary  # 返回汇总。


def build_ensemble(run_name: str, year: int) -> int:  # 逐日标准化三个 seed 后等权平均。
    """逐日标准化三个 seed 后等权平均；空 prediction 年份返回 0。"""

    model_root, prediction_root, _ = run_paths(run_name)  # 构造模型根目录。
    split = load_sequence_split(year)
    if not len(split["prediction_indices"]):  # 检查当前样本数量是否满足计算要求。
        return 0  # 返回逐日计算结果与诊断。
    seed_frames: dict[int, pd.DataFrame] = {}  # 初始化随机种子帧容器。
    for seed in SEEDS:  # 遍历随机种子，逐项处理随机种子。
        path = model_root / f"forecast_{year}" / f"seed_{seed}" / "prediction.parquet"
        seed_frames[seed] = pd.read_parquet(path)  # 读取随机种子帧。

    reference = seed_frames[SEEDS[0]]["sample_index"].to_numpy(dtype=np.int64)  # 转换并统一reference的数据表示。
    output = pd.DataFrame({"sample_index": reference})
    raw_columns: list[str] = []  # 接收原始列。
    for seed in SEEDS:  # 遍历随机种子，逐项处理随机种子。
        frame = seed_frames[seed]
        indices = frame["sample_index"].to_numpy(dtype=np.int64)  # 转换并统一索引的数据表示。
        if not np.array_equal(reference, indices):  # 检查必需数据、配置或校验结果是否缺失。
            raise AssertionError(f"{year} 年三个 seed 的预测行号不一致")  # 关键数据契约被破坏时立即终止。
        column = f"prediction_seed_{seed}"
        output[column] = frame["prediction"].to_numpy(dtype=np.float32)  # 转换并统一输出的数据表示。
        raw_columns.append(column)  # 把当前结果追加到汇总列表。

    metadata = pd.read_parquet(  # 读取元数据。
        SAMPLE_INDEX_PATH,
        columns=["sample_index", "trade_date", "instrument_id"],  # 限定本次读取或输出使用的字段。
    ).set_index("sample_index")
    output = metadata.loc[reference].reset_index().merge(  # 按业务键对齐，避免依赖原始行序。
        output, on="sample_index", validate="one_to_one"
    )
    raw = output[raw_columns].astype(np.float64)  # 转换原始的数据类型或存储位置。
    grouped = raw.groupby(output["trade_date"], sort=False)  # 按业务键分组得到grouped。
    means = grouped.transform("mean")  # 按约定截面口径汇总。
    standard_deviations = grouped.transform("std", ddof=0)  # 按约定截面口径汇总。
    if (standard_deviations <= 0).any().any():  # 检查当前掩码或数据是否满足有效性要求。
        raise ValueError(f"{year} 年存在常数 seed 预测")  # 输入数据不满足业务或数值约束时立即终止。
    zscore_columns: list[str] = []  # 接收zscore列。
    for column in raw_columns:  # 遍历原始列，逐项处理列。
        zscore_column = f"{column}_zscore"  # 计算zscore列。
        output[zscore_column] = (  # 计算输出。
            (raw[column] - means[column]) / standard_deviations[column]
        ).to_numpy(dtype=np.float32)
        zscore_columns.append(zscore_column)  # 把当前结果追加到汇总列表。
    output["prediction_ensemble_raw"] = raw.mean(axis=1).to_numpy(dtype=np.float32)  # 转换并统一输出的数据表示。
    output["prediction_ensemble"] = output[zscore_columns].mean(axis=1).astype(  # 计算输出。
        np.float32
    )
    output["forecast_year"] = year
    output["factor_role"] = "full_year_oos"
    atomic_write_parquet(  # 原子落盘，避免半成品。
        output,
        prediction_root / f"vivit_predictions.{year}.parquet",
    )
    return len(output)  # 返回逐日计算结果与诊断。


def finalize_run(args: argparse.Namespace, wall_seconds: float) -> pd.DataFrame:  # 核验年度任务并生成集成预测、诊断、清单和报告。
    """核验年度任务并生成集成预测、诊断、清单和报告。"""
    model_root, _, quality_root = run_paths(args.run_name)  # 构造模型根目录。
    records: list[dict[str, object]] = []  # 初始化records容器。
    for year in FORECAST_YEARS:  # 遍历forecast年份，逐项处理年份。
        for seed in SEEDS:  # 遍历随机种子，逐项处理随机种子。
            path = model_root / f"forecast_{year}" / f"seed_{seed}" / "completed.json"
            if not path.exists():  # 检查目标文件或目录是否满足存在性要求。
                raise FileNotFoundError(f"缺少已完成任务：{path}")  # 目标文件不存在时立即终止。
            records.append(json.loads(path.read_text(encoding="utf-8"))["summary"])  # 把当前结果追加到汇总列表。
    summary = pd.DataFrame(records).sort_values(  # 计算汇总以固定样本顺序。
        ["forecast_year", "seed"], kind="stable"
    )
    atomic_write_parquet(summary, quality_root / "vivit_training_summary.parquet")  # 原子落盘，避免半成品。
    factor_counts = {year: build_ensemble(args.run_name, year) for year in FORECAST_YEARS}  # 计算因子数量。
    sequence_manifest = require_aligned_sequence_store()
    manifest = {  # 初始化清单容器。
        "route": (  # 记录route。
            "DDP_single_node_tasks"
            if getattr(args, "execution_route", "a2") == "ddp"  # 按当前模式、标识或状态选择处理分支。
            else "A2_independent_tasks"
        ),
        "execution_route": getattr(args, "execution_route", "a2"),  # 记录执行route。
        "world_size": int(getattr(args, "ddp_world_size", 1)),  # 记录world大小。
        "run_name": args.run_name,  # 记录运行名称。
        "sequence_signature": sequence_manifest["signature"],  # 记录sequence签名，用于来源一致性校验。
        "source_dataset_signature": sequence_manifest["payload"][  # 记录来源dataset签名，用于来源一致性校验。
            "source_dataset_signature"
        ],
        "source_rolling_signature": sequence_manifest["payload"][  # 记录来源rolling签名，用于来源一致性校验。
            "source_rolling_signature"
        ],
        "device_ids": [int(value) for value in args.device_ids.split(",")],  # 记录设备ids。
        "tasks_per_gpu": getattr(args, "tasks_per_gpu", None),  # 记录taskspergpu。
        "train_batch_mode": args.train_batch_mode,  # 记录训练批次模式。
        "batch_size_per_task": (  # 记录批次大小pertask。
            "full_trade_date" if args.train_batch_mode == "trade_date" else args.batch_size
        ),
        "fixed_batch_size": args.batch_size,  # 记录fixed批次大小。
        "inference_batch_size": args.inference_batch_size,  # 记录inference批次大小。
        "workers_per_task": args.workers_per_task,  # 记录workerspertask。
        "loss_name": args.loss_name,  # 记录损失名称。
        "loss_parameters": {  # 记录损失parameters。
            "rank_temperature": args.rank_temperature,  # 记录秩temperature。
            "turnover_lambda": args.turnover_lambda,  # 记录turnoverlambda。
            "huber_delta": args.huber_delta,  # 记录huberdelta。
            "smooth_l1_beta": args.smooth_l1_beta,  # 记录smoothl1beta。
            "hybrid_weight": args.hybrid_weight,  # 记录hybrid权重。
        },
        "validation_metric": args.validation_metric,  # 记录验证指标。
        "validation_audit_metrics": [  # 记录验证audit指标。
            "daily_mean_spearman_ic",
            "daily_mean_pearson_ic",
            "pooled_pearson_ic",
            "temporal_spearman_rank_ic",
            "temporal_rank_penalty",
        ],
        "split_source": "ViT/data/feature_store/rolling_splits",  # 把 split_source 映射为"ViT/data/feature_store/rolling_splits"。
        "split_policy": "exact_vit_train_validation_prediction_indices",  # 把 split_policy 映射为"exact_vit_train_validation_prediction_indices"。
        "padding_policy": "zero_fill_with_tubelet_coverage_mask",  # 把 padding_policy 映射为"zero_fill_with_tubelet_coverage_mask"。
        "learning_rate": args.learning_rate,  # 记录learning比例。
        "max_epochs": args.max_epochs,  # 记录maxepochs。
        "patience": args.patience,  # 记录patience。
        "torch_compile": bool(args.compile),  # 记录torchcompile。
        "compile_mode": args.compile_mode,  # 记录compile模式。
        "fused_adam": bool(args.fused_adam),  # 记录fusedadam。
        "cuda_prefetch": bool(args.cuda_prefetch),  # 记录cudaprefetch。
        "years": list(FORECAST_YEARS),  # 记录年份。
        "seeds": list(SEEDS),  # 记录随机种子。
        "completed_tasks": len(summary),  # 记录completedtasks。
        "factor_counts": factor_counts,  # 记录因子数量。
        "wall_seconds": wall_seconds,  # 记录wallseconds。
    }
    atomic_write_json(manifest, quality_root / "a2_training_manifest.json")  # 原子落盘，避免半成品。
    return summary  # 返回汇总。


def worker_command(args: argparse.Namespace, year: int, seed: int, gpu: int) -> list[str]:  # 构造单个年度×seed worker 的命令行参数。
    """构造单个年度×seed worker 的命令行参数。"""
    return [  # 返回构造单个年度×seed worker 的命令行参数。
        sys.executable,
        str(Path(__file__).resolve()),
        "worker",
        "--run-name", args.run_name,
        "--train-batch-mode", args.train_batch_mode,
        "--batch-size", str(args.batch_size),
        "--inference-batch-size", str(args.inference_batch_size),
        "--loss-name", args.loss_name,
        "--rank-temperature", str(args.rank_temperature),
        "--turnover-lambda", str(args.turnover_lambda),
        "--huber-delta", str(args.huber_delta),
        "--smooth-l1-beta", str(args.smooth_l1_beta),
        "--hybrid-weight", str(args.hybrid_weight),
        "--validation-metric", args.validation_metric,
        "--workers-per-task", str(args.workers_per_task),
        "--learning-rate", str(args.learning_rate),
        "--max-epochs", str(args.max_epochs),
        "--patience", str(args.patience),
        "--compile", str(args.compile),
        "--compile-mode", args.compile_mode,
        "--fused-adam", str(args.fused_adam),
        "--cuda-prefetch", str(args.cuda_prefetch),
        "--overwrite", str(args.overwrite),
        "--worker-year", str(year),
        "--worker-seed", str(seed),
        "--worker-physical-gpu", str(gpu),
    ]


def ddp_worker_command(  # 组装启动单个 DDP worker 的命令。
    args: argparse.Namespace,  # 接收args。
    year: int,  # 接收年份。
    seed: int,  # 接收随机种子。
    device_ids: tuple[int, ...],
) -> list[str]:  # 声明当前函数的返回类型。
    """执行 ddp_worker_command 对应的分布式训练处理。"""
    command = worker_command(args, year, seed, device_ids[0])
    command[2] = "ddp-worker"
    command.extend(["--device-ids", ",".join(str(value) for value in device_ids)])  # 把一组结果并入当前列表。
    return command  # 返回command。


def _baseline_ddp_smoke(  # 执行基线 DDP 最小连通性测试。
    args: argparse.Namespace,  # 接收args。
    context: distributed.DistributedContext,  # 接收上下文。
) -> dict[str, object]:  # 声明当前函数的返回类型。
    """执行 _baseline_ddp_smoke 对应的分布式训练处理。"""
    config = training_config_from_args(args)
    validate_training_config(config, args.compile_mode)
    set_random_seed(args.seed + context.rank)
    train_loader, validation_loader, _, batch_summary = make_distributed_data_loaders(  # 创建训练数据加载器。
        args.forecast_year,
        args.seed,
        config,
        args.workers_per_task,
        context.device.type == "cuda",
        context,
    )
    train_loader.batch_sampler.set_epoch(1)  # 同步当前轮次以确定分布式采样顺序。
    model = build_model(  # 创建模型。
        context.device,
        bool(args.compile),
        args.compile_mode,
        config.train_batch_mode,
    )
    model = distributed.wrap_model(model, context)  # 创建模型。
    optimizer = build_optimizer(  # 创建优化器。
        model, config, bool(args.fused_adam), context.device
    )
    train_metrics = train_one_epoch(  # 计算训练指标。
        model,
        train_loader,
        optimizer,
        context.device,
        bool(args.cuda_prefetch),
        config,
        max_batches=args.max_batches,  # 限制调试运行处理的最大批次数。
        distributed_context=context,  # 配置distributed上下文。
    )
    validation_metrics = evaluate_model(  # 计算验证指标。
        model,
        validation_loader,
        context.device,
        bool(args.cuda_prefetch),
        config.validation_metric,
        max_batches=args.max_batches,  # 限制调试运行处理的最大批次数。
        distributed_context=context,  # 配置distributed上下文。
    )
    return {  # 返回上述处理结果。
        "forecast_year": args.forecast_year,  # 记录forecast年份。
        "seed": args.seed,  # 记录随机种子。
        "execution_route": "ddp",  # 把 execution_route 映射为"ddp"。
        "world_size": context.world_size,  # 记录world大小。
        "batch_summary": batch_summary,  # 记录批次汇总。
        "train_metrics": train_metrics,  # 记录训练指标。
        "validation_metrics": validation_metrics,  # 记录验证指标。
        "persistent_outputs_written": False,  # 记录persistentoutputswritten。
    }


def _baseline_ddp_rank(  # baselineddp排名。
    rank: int,  # 接收排名。
    world_size: int,  # 接收world大小。
    args: argparse.Namespace,  # 接收args。
    smoke: bool,  # 接收smoke。
) -> None:  # 声明当前函数的返回类型。
    """执行 _baseline_ddp_rank 对应的分布式训练处理。"""
    context = distributed.initialize_process_group(rank, world_size, backend="nccl")
    try:  # 执行可能失败的外部读取或计算步骤。
        args.execution_route = "ddp"
        args.ddp_world_size = world_size
        if smoke:  # 根据当前数据状态选择处理分支。
            result = _baseline_ddp_smoke(args, context)
        else:  # 处理上述条件均不成立的情况。
            result = train_single_run(args, distributed_context=context)
        if context.is_primary:  # 根据当前数据状态选择处理分支。
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)  # 输出当前任务的进度或诊断信息。
    finally:  # 无论前序步骤是否成功都执行资源清理。
        distributed.destroy_process_group()


def run_ddp_worker(args: argparse.Namespace, *, smoke: bool = False) -> None:  # 执行ddpworker。
    """执行 run_ddp_worker 对应的分布式训练处理。"""
    devices = distributed.parse_device_ids(args.device_ids)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(value) for value in devices)
    if not torch.cuda.is_available() or torch.cuda.device_count() != len(devices):  # 检查当前样本数量是否满足计算要求。
        raise RuntimeError(  # 当前运行环境或任务状态不支持继续执行。
            f"DDP 请求 {len(devices)} 张 GPU，但当前可见 {torch.cuda.device_count()} 张"
        )
    args.ddp_world_size = len(devices)
    args.execution_route = "ddp"
    if smoke:  # 根据当前数据状态选择处理分支。
        args.worker_year = args.forecast_year
        args.worker_seed = args.seed
    args.worker_physical_gpu = devices[0]
    distributed.configure_master(distributed.reserve_local_port())
    mp.spawn(
        _baseline_ddp_rank,
        args=(len(devices), args, smoke),  # 配置args。
        nprocs=len(devices),  # 配置nprocs。
        join=True,  # 配置join。
    )


def run_ddp(args: argparse.Namespace) -> pd.DataFrame:  # 执行ddp。
    """执行 run_ddp 对应的分布式训练处理。"""

    devices = distributed.parse_device_ids(args.device_ids)
    args.execution_route = "ddp"
    args.ddp_world_size = len(devices)
    _, _, quality_root = run_paths(args.run_name)
    log_dir = quality_root / "ddp_task_logs"  # 构造log目录。
    log_dir.mkdir(parents=True, exist_ok=True)  # 确保目标目录存在。
    started = time.perf_counter()
    print(  # 输出当前任务的进度或诊断信息。
        f"ViViT DDP start: tasks={len(FORECAST_YEARS) * len(SEEDS)}, "
        f"GPUs={devices}, world_size={len(devices)}",
        flush=True,  # 配置flush。
    )
    for year in FORECAST_YEARS:  # 遍历forecast年份，逐项处理年份。
        for seed in SEEDS:  # 遍历随机种子，逐项处理随机种子。
            log_path = log_dir / f"forecast_{year}.seed_{seed}.log"  # 构造log路径。
            with log_path.open("a", encoding="utf-8") as handle:  # 按指定模式打开文件并确保退出时关闭句柄。
                handle.write(
                    f"\n=== launch {year}/seed={seed} at "
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
                )
                handle.flush()
                environment = os.environ.copy()  # 转换environment的数据类型或存储位置。
                environment["CUDA_VISIBLE_DEVICES"] = ",".join(  # 对齐并合并environment。
                    str(value) for value in devices
                )
                process = subprocess.Popen(
                    ddp_worker_command(args, year, seed, devices),
                    cwd=VIVIT_DIR,  # 配置cwd。
                    env=environment,  # 配置env。
                    stdout=handle,  # 配置stdout。
                    stderr=subprocess.STDOUT,  # 配置stderr。
                    text=True,  # 配置text。
                )
                code = process.wait()
            if code:  # 根据当前数据状态选择处理分支。
                raise RuntimeError(  # 当前运行环境或任务状态不支持继续执行。
                    f"DDP 任务失败：{year}/seed={seed}\n{tail(log_path, 100)}"
                )
            print(f"done DDP: {year}/seed={seed}", flush=True)  # 输出当前任务的进度或诊断信息。
    result = finalize_run(args, time.perf_counter() - started)
    print(validate_outputs(args.run_name).to_string(index=False), flush=True)  # 输出当前任务的进度或诊断信息。
    print(f"实验报告：{write_experiment_report(args.run_name)}", flush=True)  # 输出当前任务的进度或诊断信息。
    return result  # 返回结果。


def tail(path: Path, line_count: int = 80) -> str:  # 读取文本文件末尾若干行，用于展示最新训练日志。
    """读取文本文件末尾若干行，用于展示最新训练日志。"""
    if not path.exists():  # 检查目标文件或目录是否满足存在性要求。
        return ""  # 返回读取并校验后的结果。
    return "\n".join(  # 返回读取并校验后的结果。
        path.read_text(encoding="utf-8", errors="replace").splitlines()[-line_count:]
    )


def run_a2(args: argparse.Namespace) -> pd.DataFrame:  # 在四张 GPU 上并发调度 24 个独立任务。
    """在四张 GPU 上并发调度 24 个独立任务。"""

    args.execution_route = "a2"
    args.ddp_world_size = 1
    if not SEQUENCE_MANIFEST_PATH.exists():  # 检查目标文件或目录是否满足存在性要求。
        build_sequence_store(overwrite=False)
    require_aligned_sequence_store()
    validate_training_config(training_config_from_args(args), args.compile_mode)
    device_ids = tuple(int(value) for value in args.device_ids.split(","))  # 初始化设备ids容器。
    if len(set(device_ids)) != len(device_ids):  # 检查当前样本数量是否满足计算要求。
        raise ValueError("GPU 编号不能重复")  # 输入数据不满足业务或数值约束时立即终止。
    tasks = deque((year, seed) for year in FORECAST_YEARS for seed in SEEDS)  # 计算tasks判定条件。
    _, _, quality_root = run_paths(args.run_name)
    log_dir = quality_root / "a2_task_logs"  # 构造log目录。
    log_dir.mkdir(parents=True, exist_ok=True)  # 确保目标目录存在。
    running: list[dict[str, object]] = []  # 初始化running容器。
    started = time.perf_counter()
    print(  # 输出当前任务的进度或诊断信息。
        f"ViViT A2 start: tasks={len(tasks)}, GPUs={device_ids}, "
        f"tasks_per_gpu={args.tasks_per_gpu}",
        flush=True,  # 配置flush。
    )
    try:  # 执行可能失败的外部读取或计算步骤。
        while tasks or running:  # 在终止条件出现前持续迭代。
            for gpu in device_ids:  # 遍历设备ids，逐项处理gpu。
                active = sum(1 for item in running if item["gpu"] == gpu)  # 配置active。
                while tasks and active < args.tasks_per_gpu:  # 在终止条件出现前持续迭代。
                    year, seed = tasks.popleft()
                    log_path = log_dir / f"forecast_{year}.seed_{seed}.log"  # 构造log路径。
                    log_handle = log_path.open("a", encoding="utf-8")
                    log_handle.write(  # 追加任务状态日志。
                        f"\n=== launch {year}/seed={seed} at "
                        f"{time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
                    )
                    log_handle.flush()  # 将 memmap 修改刷入磁盘。
                    environment = os.environ.copy()  # 转换environment的数据类型或存储位置。
                    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
                    process = subprocess.Popen(  # 启动独立训练任务。
                        worker_command(args, year, seed, gpu),
                        cwd=VIVIT_DIR,  # 配置cwd。
                        env=environment,  # 配置env。
                        stdout=log_handle,  # 配置stdout。
                        stderr=subprocess.STDOUT,  # 配置stderr。
                        text=True,  # 配置text。
                    )
                    running.append(  # 把当前结果追加到汇总列表。
                        {
                            "process": process,  # 记录process。
                            "gpu": gpu,  # 记录gpu。
                            "year": year,  # 记录年份。
                            "seed": seed,  # 记录随机种子。
                            "log_path": log_path,  # 记录log路径。
                            "log_handle": log_handle,  # 记录loghandle。
                            "started": time.perf_counter(),  # 记录started。
                        }
                    )
                    active += 1
                    print(  # 输出当前任务的进度或诊断信息。
                        f"launch GPU{gpu}: {year}/seed={seed} pid={process.pid}",
                        flush=True,  # 配置flush。
                    )

            finished = []  # 初始化finished容器。
            for item in running:  # 遍历running，逐项处理item。
                return_code = item["process"].poll()  # 检查训练任务状态。
                if return_code is None:  # 检查所需对象是否尚未初始化。
                    continue  # 跳过当前项并处理下一项。
                item["log_handle"].close()
                finished.append(item)  # 把当前结果追加到汇总列表。
                if return_code:  # 根据当前数据状态选择处理分支。
                    raise RuntimeError(  # 当前运行环境或任务状态不支持继续执行。
                        f"任务失败：{item['year']}/seed={item['seed']} GPU{item['gpu']}\n"
                        f"{tail(item['log_path'])}"
                    )
                minutes = (time.perf_counter() - item["started"]) / 60
                print(  # 输出当前任务的进度或诊断信息。
                    f"done GPU{item['gpu']}: {item['year']}/seed={item['seed']} "
                    f"{minutes:.2f} min",
                    flush=True,  # 配置flush。
                )
            for item in finished:  # 遍历finished，逐项处理item。
                running.remove(item)  # 移除已结束任务。
            if running and not finished:  # 检查必需数据、配置或校验结果是否缺失。
                time.sleep(1)
    except BaseException:  # 捕获预期异常并执行对应恢复逻辑。
        for item in running:  # 遍历running，逐项处理item。
            if item["process"].poll() is None:  # 检查所需对象是否尚未初始化。
                item["process"].terminate()  # 终止仍在运行的任务。
            item["log_handle"].close()
        raise

    summary = finalize_run(args, time.perf_counter() - started)  # 汇总任务并写出完成标记。
    validate_outputs(args.run_name)
    write_experiment_report(args.run_name)
    return summary  # 返回汇总。


def markdown_table(frame: pd.DataFrame) -> str:  # 把数据表转换为适合实验报告的 Markdown 表格。
    """把数据表转换为适合实验报告的 Markdown 表格。"""
    columns = list(frame.columns)  # 初始化列容器。
    header = "| " + " | ".join(columns) + " |"  # 对齐并合并header。
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"  # 对齐并合并separator。
    rows = [  # 初始化行数容器。
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)  # 遍历帧itertuples索引false名称none，逐项处理行。
    ]
    return "\n".join([header, separator, *rows])  # 返回完成上述格式转换后的结果。


def write_experiment_report(run_name: str) -> Path:  # 根据当前 run 的真实产物生成中文报告，不覆盖历史根目录报告。
    """根据当前 run 的真实产物生成中文报告，不覆盖历史根目录报告。"""

    _, _, quality_root = run_paths(run_name)
    training = pd.read_parquet(quality_root / "vivit_training_summary.parquet")  # 读取训练。
    factor_audit = validate_outputs(run_name)
    sequence_summary = pd.read_parquet(SEQUENCE_SUMMARY_PATH)  # 读取sequence汇总。
    run_manifest = json.loads(
        (quality_root / "a2_training_manifest.json").read_text(encoding="utf-8")
    )
    dataset_manifest, _ = source_manifests()  # 读取或生成来源清单。
    yearly = (
        training.groupby("forecast_year", as_index=False)
        .agg(
            best_validation_ic_mean=("best_validation_ic", "mean"),  # 配置最优验证ic均值。
            best_validation_ic_min=("best_validation_ic", "min"),  # 配置最优验证icmin。
            best_validation_ic_max=("best_validation_ic", "max"),  # 配置最优验证icmax。
            best_epoch_mean=("best_epoch", "mean"),  # 配置最优轮次均值。
            task_minutes=("elapsed_seconds", lambda values: values.sum() / 60),  # 配置taskminutes。
        )
        .round(6)
    )
    architecture = pd.DataFrame(
        [
            ["输入", "[B,17,20,8,8] + [B,20] frame_mask"],
            ["Tubelet embedding", "Conv3d kernel=stride=(2,2,2)"],
            ["缺帧建模", "0/1/2 覆盖 embedding；全缺 Tubelet 屏蔽 K/V"],
            ["Tubelet 网格", "10×4×4"],
            ["空间 / 时间编码器", "2 层 / 2 层"],
            ["Embedding / Heads / MLP", "128 / 4 / 512"],
            ["输出", "LayerNorm + Linear 标量预测"],
        ],
        columns=["项目", "设置"],  # 限定本次读取或输出使用的字段。
    )
    report = f"""# ViViT 与 ViT 索引对齐实验报告

生成时间：{pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M:%S')} UTC
实验名称：`{run_name}`

## 1. 数据与索引口径

- 源 feature store：{dataset_manifest['sample_count']:,} 条样本，日期为 {dataset_manifest['first_sample_date']}～{dataset_manifest['last_sample_date']}。
- train、validation、prediction 在运行时直接读取 ViT 年度 split，逐元素保持一致，不再每 5 日降采样，也不再过滤历史不足 20 日的目标样本。
- 每个历史槽位只匹配同股票精确 `target_date_id-lag`；不前向填充、不使用未来数据。缺失帧补零，并通过 `frame_mask` 显式输入模型。
- 2025 prediction 与 ViT 一致为空，因此不创建 seed prediction 或集成因子文件。

{markdown_table(sequence_summary)}

## 2. 模型结构

{markdown_table(architecture)}

## 3. 训练方案

- 路线：`{run_manifest.get('execution_route', 'a2')}`；本报告仅描述该 run 的实际产物。
- loss：`{run_manifest['loss_name']}`；batch mode：`{run_manifest['train_batch_mode']}`；早停指标：`{run_manifest['validation_metric']}`。
- 固定 batch size：{run_manifest['fixed_batch_size']}；推理 batch size：{run_manifest['inference_batch_size']}。
- Adam 学习率：{run_manifest['learning_rate']}；最大 epoch：{run_manifest['max_epochs']}；patience：{run_manifest['patience']}。
- 验证同时记录日均 Pearson、日均 Spearman、pooled Pearson 与时序指标；早停只使用配置选择的指标。
- torch.compile={run_manifest['torch_compile']}，compile mode=`{run_manifest.get('compile_mode', 'default')}`，fused Adam={run_manifest['fused_adam']}，CUDA prefetch={run_manifest['cuda_prefetch']}。

## 4. 训练结果

{markdown_table(yearly)}

完整任务明细见 `data/quality/runs/{run_name}/vivit_training_summary.parquet`。

## 5. 因子产物核验

{markdown_table(factor_audit)}

非空年份的三个 seed 预测按日截面 Z-score 后等权集成；空 prediction 年份只在汇总中记录 0 行。

## 6. 边界

本流程不生成 delivery 包，不开展因子回测，也不会覆盖旧 `data/sequence_store`、旧模型、旧预测或根目录历史实验报告。
"""
    path = quality_root / "EXPERIMENT_REPORT.md"
    atomic_write_text(report, path)  # 原子落盘，避免半成品。
    return path  # 返回路径。


def run_smoke_test(args: argparse.Namespace) -> dict[str, object]:  # 按指定生产配置执行真实数据有限 batch 前向、反向与验证。
    """按指定生产配置执行真实数据有限 batch 前向、反向与验证。"""

    if not SEQUENCE_MANIFEST_PATH.exists():  # 检查目标文件或目录是否满足存在性要求。
        build_sequence_store(overwrite=False)
    require_aligned_sequence_store()
    config = training_config_from_args(args)
    validate_training_config(config, args.compile_mode)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_random_seed(args.seed)  # 固定各随机源。
    train_loader, validation_loader, prediction_loader = make_data_loaders(  # 创建训练数据加载器。
        args.forecast_year,
        args.seed,
        config.batch_size,
        config.inference_batch_size,
        config.train_batch_mode,
        config.loss_name,
        args.workers_per_task,
        device.type == "cuda",
    )
    set_train_epoch(train_loader, 1)  # 同步 epoch，保持采样可复现。
    audit_batch = next(iter(train_loader))
    model = build_model(  # 创建模型并部署到设备。
        device, bool(args.compile), args.compile_mode, config.train_batch_mode
    )
    optimizer = build_optimizer(model, config, bool(args.fused_adam), device)  # 按训练配置创建优化器。
    train_metrics = train_one_epoch(  # 完成一个训练 epoch。
        model,
        train_loader,
        optimizer,
        device,
        bool(args.cuda_prefetch),
        config,
        max_batches=args.max_batches,  # 限制调试运行处理的最大批次数。
    )
    validation_metrics = evaluate_model(  # 无梯度计算验证指标。
        model,
        validation_loader,
        device,
        bool(args.cuda_prefetch),
        config.validation_metric,
        max_batches=args.max_batches,  # 限制调试运行处理的最大批次数。
    )
    prediction_rows = 0
    if prediction_loader is not None:  # 检查必需数据、配置或校验结果是否缺失。
        prediction_rows = len(
            predict_model(
                model,
                prediction_loader,
                device,
                bool(args.cuda_prefetch),
                max_batches=1,  # 限制调试运行处理的最大批次数。
            )
        )
    frame_mask = audit_batch["frame_mask"]  # 构造帧掩码，限定后续参与计算的样本。
    return {  # 返回按当前规则组织后的结果。
        "forecast_year": args.forecast_year,  # 记录forecast年份。
        "seed": args.seed,  # 记录随机种子。
        "device": str(device),  # 记录设备。
        "train_batch_mode": config.train_batch_mode,  # 记录训练批次模式。
        "loss_name": config.loss_name,  # 记录损失名称。
        "validation_metric": config.validation_metric,  # 记录验证指标。
        "batch_summary": loader_batch_summary(train_loader),  # 记录批次汇总。
        "audit_video_shape": list(audit_batch["videos"].shape),  # 记录audit视频形状。
        "audit_frame_mask_shape": list(frame_mask.shape),  # 记录audit帧掩码形状。
        "audit_valid_frame_ratio": float(frame_mask.float().mean().item()),  # 记录audit有效帧比例。
        "train_metrics": train_metrics,  # 记录训练指标。
        "validation_metrics": validation_metrics,  # 记录验证指标。
        "sample_prediction_rows": prediction_rows,  # 记录样本预测行数。
        "parameter_count": sum(p.numel() for p in unwrap_model(model).parameters()),  # 记录parameter数量。
        "persistent_outputs_written": False,  # 记录persistentoutputswritten。
    }


def run_memory_benchmark(args: argparse.Namespace) -> dict[str, object]:  # 用一个指定配置的真实 batch 测量单任务峰值显存和耗时。
    """用一个指定配置的真实 batch 测量单任务峰值显存和耗时。"""

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":  # 按当前模式、标识或状态选择处理分支。
        raise RuntimeError("显存 benchmark 要求 CUDA")  # 当前运行环境或任务状态不支持继续执行。
    require_aligned_sequence_store()
    config = training_config_from_args(args)
    validate_training_config(config, args.compile_mode)
    loader, _, _ = make_data_loaders(  # 创建数据加载器。
        args.forecast_year,
        args.seed,
        config.batch_size,
        config.inference_batch_size,
        config.train_batch_mode,
        config.loss_name,
        args.workers_per_task,
        pin_memory=True,  # 控制是否锁页以加速 CPU 到 GPU 传输。
    )
    set_train_epoch(loader, 1)  # 同步 epoch，保持采样可复现。
    model = build_model(  # 创建模型并部署到设备。
        device, bool(args.compile), args.compile_mode, config.train_batch_mode
    )
    optimizer = build_optimizer(model, config, bool(args.fused_adam), device)  # 按训练配置创建优化器。
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    metrics = train_one_epoch(  # 完成一个训练 epoch。
        model,
        loader,
        optimizer,
        device,
        bool(args.cuda_prefetch),
        config,
        max_batches=1,  # 限制调试运行处理的最大批次数。
    )
    return {  # 返回完成上述回归或变换后的结果。
        "forecast_year": args.forecast_year,  # 记录forecast年份。
        "train_batch_mode": config.train_batch_mode,  # 记录训练批次模式。
        "loss_name": config.loss_name,  # 记录损失名称。
        "seconds": time.perf_counter() - started,  # 记录seconds。
        "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 1024**3,  # 记录peakmemorygib。
        **metrics,
    }


def validate_training_config(config: TrainingConfig, compile_mode: str) -> None:  # 校验训练模式、损失、早停与资源参数组合。
    """校验训练模式、损失、早停与资源参数组合。"""
    if config.train_batch_mode not in TRAIN_BATCH_MODES:  # 检查当前值是否属于允许集合。
        raise ValueError(f"未知训练 batch 模式：{config.train_batch_mode}")  # 输入数据不满足业务或数值约束时立即终止。
    if config.loss_name not in LOSS_NAMES:  # 检查当前值是否属于允许集合。
        raise ValueError(f"未知损失函数：{config.loss_name}")  # 输入数据不满足业务或数值约束时立即终止。
    if config.validation_metric not in VALIDATION_METRICS:  # 检查当前值是否属于允许集合。
        raise ValueError(f"未知验证指标：{config.validation_metric}")  # 输入数据不满足业务或数值约束时立即终止。
    if config.loss_name == TEMPORAL_TURNOVER_LOSS and config.train_batch_mode != "trade_date":  # 按当前模式、标识或状态选择处理分支。
        raise ValueError("sigmoid_rank_ic_turnover 必须使用 trade_date batch")  # 输入数据不满足业务或数值约束时立即终止。
    if config.loss_name == TEMPORAL_TURNOVER_LOSS and compile_mode == "reduce-overhead":  # 按当前模式、标识或状态选择处理分支。
        raise ValueError("时序换手损失禁止使用 reduce-overhead compile mode")  # 输入数据不满足业务或数值约束时立即终止。
    if (  # 开始组合多项有效性或配置条件。
        config.batch_size <= 1
        or config.inference_batch_size <= 1
        or config.learning_rate <= 0
        or config.max_epochs < 1
        or config.early_stopping_patience < 1
        or config.rank_temperature <= 0
        or config.turnover_lambda < 0
        or config.huber_delta <= 0
        or config.smooth_l1_beta <= 0
        or not 0 <= config.hybrid_weight <= 1
    ):
        raise ValueError("训练、batch 或损失超参数非法")  # 输入数据不满足业务或数值约束时立即终止。

def validate_outputs(run_name: str) -> pd.DataFrame:  # 核验 24 个任务与 8 个年度因子文件。
    """核验 24 个任务与 8 个年度因子文件。"""

    model_root, prediction_root, quality_root = run_paths(run_name)  # 构造模型根目录。
    summary_path = quality_root / "vivit_training_summary.parquet"  # 构造汇总路径。
    if not summary_path.exists():  # 检查目标文件或目录是否满足存在性要求。
        raise FileNotFoundError(summary_path)  # 目标文件不存在时立即终止。
    training_summary = pd.read_parquet(summary_path)  # 读取训练汇总。
    if len(training_summary) != len(FORECAST_YEARS) * len(SEEDS):  # 检查当前样本数量是否满足计算要求。
        raise AssertionError("训练汇总不是 8 年度 × 3 种子")  # 关键数据契约被破坏时立即终止。

    records = []  # 初始化records容器。
    for year in FORECAST_YEARS:  # 遍历forecast年份，逐项处理年份。
        split = load_sequence_split(year)
        expected = len(split["prediction_indices"])
        path = prediction_root / f"vivit_predictions.{year}.parquet"
        if expected == 0:  # 按当前模式、标识或状态选择处理分支。
            if path.exists():  # 检查目标文件或目录是否满足存在性要求。
                raise AssertionError(f"{year} 年 prediction 为空但存在因子文件")  # 关键数据契约被破坏时立即终止。
            records.append(  # 把当前结果追加到汇总列表。
                {
                    "forecast_year": year,  # 记录forecast年份。
                    "factor_rows": 0,  # 记录因子行数。
                    "trade_dates": 0,  # 记录trade交易日。
                    "start_date": None,  # 记录起点交易日。
                    "end_date": None,  # 记录end交易日。
                    "status": "无 prediction，按 ViT 口径通过",  # 把 status 映射为"无 prediction，按 ViT 口径通过"。
                }
            )
            continue  # 跳过当前项并处理下一项。
        if not path.exists():  # 检查目标文件或目录是否满足存在性要求。
            raise FileNotFoundError(path)  # 目标文件不存在时立即终止。
        frame = pd.read_parquet(path)  # 读取帧。
        if len(frame) != expected or frame["sample_index"].duplicated().any():  # 检查当前样本数量是否满足计算要求。
            raise AssertionError(f"{year} 年因子行数或主键错误")  # 关键数据契约被破坏时立即终止。
        if not np.array_equal(  # 检查必需数据、配置或校验结果是否缺失。
            frame["sample_index"].to_numpy(np.int64), split["prediction_indices"]
        ):
            raise AssertionError(f"{year} 年因子索引与 ViT prediction split 不一致")  # 关键数据契约被破坏时立即终止。
        if not np.isfinite(frame["prediction_ensemble"]).all():  # 检查参与计算的数值是否全部有限。
            raise AssertionError(f"{year} 年因子包含非有限值")  # 关键数据契约被破坏时立即终止。
        for seed in SEEDS:  # 遍历随机种子，逐项处理随机种子。
            column = f"prediction_seed_{seed}_zscore"
            grouped = frame.groupby("trade_date", sort=False)[column]  # 按业务键分组得到grouped。
            max_mean = float(grouped.mean().abs().max())  # 计算max均值。
            max_std_error = float((grouped.std(ddof=0) - 1).abs().max())  # 计算max标准差error。
            if max_mean > 1e-5 or max_std_error > 1e-5:  # 检查当前数值是否越过业务或数值阈值。
                raise AssertionError(f"{year} 年 seed={seed} 日截面标准化失败")  # 关键数据契约被破坏时立即终止。
        records.append(  # 把当前结果追加到汇总列表。
            {
                "forecast_year": year,  # 记录forecast年份。
                "factor_rows": len(frame),  # 记录因子行数。
                "trade_dates": frame["trade_date"].nunique(),  # 记录trade交易日。
                "start_date": frame["trade_date"].min(),  # 记录起点交易日。
                "end_date": frame["trade_date"].max(),  # 记录end交易日。
                "status": "通过",  # 把 status 映射为"通过"。
            }
        )

  # 明确确认本任务没有创建用户排除的交付包和回测结果。
    if (VIVIT_DIR / "delivery").exists():  # 检查目标文件或目录是否满足存在性要求。
        raise AssertionError("本任务不应生成 delivery 目录")  # 关键数据契约被破坏时立即终止。
    if (DATA_DIR / "backtest").exists():  # 检查目标文件或目录是否满足存在性要求。
        raise AssertionError("本任务不应生成 backtest 目录")  # 关键数据契约被破坏时立即终止。
    return pd.DataFrame(records)  # 返回核验 24 个任务与 8 个年度因子文件。

def add_training_arguments(parser: argparse.ArgumentParser) -> None:  # 向命令行解析器注册训练后端参数。
    """向命令行解析器注册训练后端参数。"""
    parser.add_argument(  # 注册命令行参数及其默认值。
        "--run-name", default="vivit_aligned_vit_indices_pearson_date_batch"
    )
    parser.add_argument(  # 注册命令行参数及其默认值。
        "--train-batch-mode", choices=TRAIN_BATCH_MODES, default="trade_date"
    )
    parser.add_argument("--batch-size", type=int, default=5000)  # 注册命令行参数及其默认值。
    parser.add_argument("--inference-batch-size", type=int, default=5000)  # 注册命令行参数及其默认值。
    parser.add_argument("--loss-name", choices=LOSS_NAMES, default="pearson_ic")  # 注册命令行参数及其默认值。
    parser.add_argument("--rank-temperature", type=float, default=0.1)  # 注册命令行参数及其默认值。
    parser.add_argument("--turnover-lambda", type=float, default=0.1)  # 注册命令行参数及其默认值。
    parser.add_argument("--huber-delta", type=float, default=1.0)  # 注册命令行参数及其默认值。
    parser.add_argument("--smooth-l1-beta", type=float, default=0.1)  # 注册命令行参数及其默认值。
    parser.add_argument("--hybrid-weight", type=float, default=0.5)  # 注册命令行参数及其默认值。
    parser.add_argument(  # 注册命令行参数及其默认值。
        "--validation-metric",
        choices=VALIDATION_METRICS,  # 配置choices。
        default="daily_mean_pearson_ic",  # 配置默认。
    )
    parser.add_argument("--workers-per-task", type=int, default=2)  # 注册命令行参数及其默认值。
    parser.add_argument("--learning-rate", type=float, default=0.001)  # 注册命令行参数及其默认值。
    parser.add_argument("--max-epochs", type=int, default=100)  # 注册命令行参数及其默认值。
    parser.add_argument("--patience", type=int, default=10)  # 注册命令行参数及其默认值。
    parser.add_argument("--compile", type=int, choices=(0, 1), default=1)  # 注册命令行参数及其默认值。
    parser.add_argument("--compile-mode", default="default")  # 注册命令行参数及其默认值。
    parser.add_argument("--fused-adam", type=int, choices=(0, 1), default=1)  # 注册命令行参数及其默认值。
    parser.add_argument("--cuda-prefetch", type=int, choices=(0, 1), default=1)  # 注册命令行参数及其默认值。
    parser.add_argument("--overwrite", type=int, choices=(0, 1), default=0)  # 注册命令行参数及其默认值。


def add_trial_arguments(parser: argparse.ArgumentParser) -> None:  # 向命令行解析器注册单任务试运行参数。
    """向命令行解析器注册单任务试运行参数。"""
    parser.add_argument("--forecast-year", type=int, default=2018)  # 注册命令行参数及其默认值。
    parser.add_argument("--seed", type=int, default=42)  # 注册命令行参数及其默认值。
    parser.add_argument("--max-batches", type=int, default=1)  # 注册命令行参数及其默认值。


def parse_args() -> argparse.Namespace:  # 解析并校验命令行参数。
    """解析并校验命令行参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke_parser = subparsers.add_parser("smoke")
    add_training_arguments(smoke_parser)
    add_trial_arguments(smoke_parser)
    smoke_parser.set_defaults(compile=0, workers_per_task=0)

    benchmark_parser = subparsers.add_parser("benchmark")
    add_training_arguments(benchmark_parser)
    add_trial_arguments(benchmark_parser)

    worker_parser = subparsers.add_parser("worker")
    add_training_arguments(worker_parser)
    worker_parser.add_argument("--worker-year", type=int, required=True)  # 注册命令行参数及其默认值。
    worker_parser.add_argument("--worker-seed", type=int, required=True)  # 注册命令行参数及其默认值。
    worker_parser.add_argument("--worker-physical-gpu", type=int, required=True)  # 注册命令行参数及其默认值。

    a2_parser = subparsers.add_parser("a2")
    add_training_arguments(a2_parser)
    a2_parser.add_argument("--device-ids", default="0,1,2,3")  # 注册命令行参数及其默认值。
    a2_parser.add_argument("--tasks-per-gpu", type=int, default=1)  # 注册命令行参数及其默认值。

    ddp_parser = subparsers.add_parser("ddp")
    add_training_arguments(ddp_parser)
    ddp_parser.add_argument("--device-ids", default="0,1,2,3")  # 注册命令行参数及其默认值。

    ddp_worker_parser = subparsers.add_parser("ddp-worker")
    add_training_arguments(ddp_worker_parser)
    ddp_worker_parser.add_argument("--worker-year", type=int, required=True)  # 注册命令行参数及其默认值。
    ddp_worker_parser.add_argument("--worker-seed", type=int, required=True)  # 注册命令行参数及其默认值。
    ddp_worker_parser.add_argument("--worker-physical-gpu", type=int, required=True)  # 注册命令行参数及其默认值。
    ddp_worker_parser.add_argument("--device-ids", required=True)  # 注册命令行参数及其默认值。

    ddp_smoke_parser = subparsers.add_parser("ddp-smoke")
    add_training_arguments(ddp_smoke_parser)
    add_trial_arguments(ddp_smoke_parser)
    ddp_smoke_parser.add_argument("--device-ids", default="0,1,2,3")  # 注册命令行参数及其默认值。
    ddp_smoke_parser.set_defaults(compile=0, workers_per_task=0)

    validate_parser = subparsers.add_parser("validate")  # 构造校验parser，限定后续参与计算的样本。
    validate_parser.add_argument("--run-name", required=True)  # 注册命令行参数及其默认值。
    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--run-name", required=True)  # 注册命令行参数及其默认值。
    return parser.parse_args()  # 返回解析并校验命令行参数。


def main() -> None:  # 根据命令行子命令分派基线训练、验证或报告流程。
    """根据命令行子命令分派基线训练、验证或报告流程。"""
    args = parse_args()
    if args.command == "smoke":  # 按命令行子命令选择执行路径。
        print(json.dumps(run_smoke_test(args), ensure_ascii=False, indent=2))  # 输出当前任务的进度或诊断信息。
    elif args.command == "benchmark":  # 选择 benchmark 子命令的处理路径。
        print(json.dumps(run_memory_benchmark(args), ensure_ascii=False, indent=2))  # 输出当前任务的进度或诊断信息。
    elif args.command == "worker":  # 选择 worker 子命令的处理路径。
        args.execution_route = "a2"
        args.ddp_world_size = 1
        print(json.dumps(train_single_run(args), ensure_ascii=False, indent=2))  # 输出当前任务的进度或诊断信息。
    elif args.command == "ddp-worker":  # 选择 ddp-worker 子命令的处理路径。
        run_ddp_worker(args)
    elif args.command == "ddp-smoke":  # 选择 ddp-smoke 子命令的处理路径。
        run_ddp_worker(args, smoke=True)
    elif args.command == "a2":  # 选择 a2 子命令的处理路径。
        print(run_a2(args).to_string(index=False))  # 输出当前任务的进度或诊断信息。
    elif args.command == "ddp":  # 选择 ddp 子命令的处理路径。
        print(run_ddp(args).to_string(index=False))  # 输出当前任务的进度或诊断信息。
    elif args.command == "validate":  # 选择 validate 子命令的处理路径。
        print(validate_outputs(args.run_name).to_string(index=False))  # 输出当前任务的进度或诊断信息。
    elif args.command == "report":  # 选择 report 子命令的处理路径。
        print(write_experiment_report(args.run_name))  # 输出当前任务的进度或诊断信息。


if __name__ == "__main__":  # 按当前模式、标识或状态选择处理分支。
    main()
