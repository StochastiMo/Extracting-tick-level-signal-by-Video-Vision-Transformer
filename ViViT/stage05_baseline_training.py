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

# 执行当前分布式训练步骤并保持既定数据语义。
def inference_model(model: nn.Module) -> nn.Module:
    """返回不触发 DDP 集体前向的本地模型。

    输入：
        model: nn.Module。
    输出：
        nn.Module。
    """

    # 执行当前分布式训练步骤并保持既定数据语义。
    return getattr(model, "module", model)


# 执行当前分布式训练步骤并保持既定数据语义。
def make_distributed_data_loaders(
    forecast_year: int,
    seed: int,
    config: TrainingConfig,
    workers: int,
    pin_memory: bool,
    context: distributed.DistributedContext,
) -> tuple[DataLoader, DataLoader, DataLoader | None, dict[str, object]]:
    """构造无重复的逐 rank loader，并保持全局 batch 语义。

    输入：
        forecast_year、seed、config、workers、pin_memory、context：训练配置。
    输出：
        tuple[DataLoader, DataLoader, DataLoader | None, dict[str, object]]。
    """

    # 执行当前分布式训练步骤并保持既定数据语义。
    split = load_sequence_split(forecast_year)
    # 执行当前分布式训练步骤并保持既定数据语义。
    common: dict[str, object] = {
        "num_workers": workers,
        "pin_memory": pin_memory,
        "collate_fn": VideoBatchCollator(),
    }
    # 执行当前分布式训练步骤并保持既定数据语义。
    if workers > 0:
        # 执行当前分布式训练步骤并保持既定数据语义。
        common.update({"persistent_workers": True, "prefetch_factor": 1})
    # 执行当前分布式训练步骤并保持既定数据语义。
    generator = torch.Generator().manual_seed(seed)
    # 执行当前分布式训练步骤并保持既定数据语义。
    train_indices = split["train_indices"]
    # 执行当前分布式训练步骤并保持既定数据语义。
    train_dataset = SampleIndexDataset(train_indices)
    # 执行当前分布式训练步骤并保持既定数据语义。
    if config.train_batch_mode == "trade_date":
        # 执行当前分布式训练步骤并保持既定数据语义。
        date_ids = np.load(DATE_IDS_PATH, mmap_mode="r")
        # 执行当前分布式训练步骤并保持既定数据语义。
        temporal_mode = config.loss_name == TEMPORAL_TURNOVER_LOSS
        # 执行当前分布式训练步骤并保持既定数据语义。
        instruments = None
        # 执行当前分布式训练步骤并保持既定数据语义。
        if temporal_mode:
            # 执行当前分布式训练步骤并保持既定数据语义。
            instruments = pd.read_parquet(
                SAMPLE_INDEX_PATH, columns=["instrument_id"]
            )["instrument_id"].to_numpy(copy=False)
        # 执行当前分布式训练步骤并保持既定数据语义。
        train_sampler = distributed.DistributedTradeDateBatchSampler(
            train_indices,
            date_ids,
            seed=seed,
            shuffle=not temporal_mode,
            instrument_ids=instruments,
            rank=context.rank,
            world_size=context.world_size,
        )
        # 执行当前分布式训练步骤并保持既定数据语义。
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            generator=generator,
            **common,
        )
        # 执行当前分布式训练步骤并保持既定数据语义。
        batch_summary = train_sampler.summary()
    # 执行当前分布式训练步骤并保持既定数据语义。
    else:
        # 执行当前分布式训练步骤并保持既定数据语义。
        train_sampler = distributed.DistributedBatchSampler(
            len(train_dataset),
            config.batch_size,
            rank=context.rank,
            world_size=context.world_size,
            seed=seed,
            shuffle=True,
            drop_last=True,
        )
        # 执行当前分布式训练步骤并保持既定数据语义。
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            generator=generator,
            **common,
        )
        # 执行当前分布式训练步骤并保持既定数据语义。
        batch_count = len(train_indices) // config.batch_size
        # 执行当前分布式训练步骤并保持既定数据语义。
        batch_summary = {
            "train_batch_mode": "fixed_samples",
            "train_batch_count": batch_count,
            "train_date_count": None,
            "train_sample_count": batch_count * config.batch_size,
            "train_batch_size_min": config.batch_size,
            "train_batch_size_median": float(config.batch_size),
            "train_batch_size_max": config.batch_size,
            "dropped_train_samples": len(train_indices) - batch_count * config.batch_size,
            "reshuffle_each_epoch": True,
            "gradient_update_mode": "per_batch",
            "optimizer_steps_per_epoch": batch_count,
        }

    # 执行当前分布式训练步骤并保持既定数据语义。
    def inference_loader(indices: np.ndarray) -> DataLoader:
        """执行 inference_loader 对应的分布式训练处理。
        
        输入：
            参数见函数类型标注。
        输出：
            结果见函数返回类型标注。
        """
        # 执行当前分布式训练步骤并保持既定数据语义。
        dataset = SampleIndexDataset(indices)
        # 按原推理 batch 边界切分，确保所有 rank 执行相同次数的 collective。
        sampler = distributed.DistributedBatchSampler(
            len(dataset),
            config.inference_batch_size,
            rank=context.rank,
            world_size=context.world_size,
            seed=seed,
            shuffle=False,
            drop_last=False,
        )
        # 执行当前分布式训练步骤并保持既定数据语义。
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            generator=generator,
            **common,
        )

    # 执行当前分布式训练步骤并保持既定数据语义。
    validation_loader = inference_loader(split["validation_indices"])
    # 执行当前分布式训练步骤并保持既定数据语义。
    prediction_loader = None
    # 执行当前分布式训练步骤并保持既定数据语义。
    if len(split["prediction_indices"]):
        # 执行当前分布式训练步骤并保持既定数据语义。
        prediction_loader = inference_loader(split["prediction_indices"])
    # 执行当前分布式训练步骤并保持既定数据语义。
    return train_loader, validation_loader, prediction_loader, batch_summary


# 执行当前分布式训练步骤并保持既定数据语义。
def gather_global_training_batch(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    batch: dict[str, torch.Tensor],
    context: distributed.DistributedContext | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """执行 gather_global_training_batch 对应的分布式训练处理。
    
    输入：
        参数见函数类型标注。
    输出：
        结果见函数返回类型标注。
    """

    # 执行当前分布式训练步骤并保持既定数据语义。
    sample_indices = batch["sample_indices"].to(predictions.device)
    # 执行当前分布式训练步骤并保持既定数据语义。
    if context is None:
        # 执行当前分布式训练步骤并保持既定数据语义。
        return predictions, targets, sample_indices
    # 执行当前分布式训练步骤并保持既定数据语义。
    global_predictions = distributed.differentiable_all_gather(predictions)
    # 执行当前分布式训练步骤并保持既定数据语义。
    global_targets = distributed.all_gather_tensor(targets)
    # 执行当前分布式训练步骤并保持既定数据语义。
    global_indices = distributed.all_gather_tensor(sample_indices)
    # 执行当前分布式训练步骤并保持既定数据语义。
    order = torch.argsort(global_indices)
    # 执行当前分布式训练步骤并保持既定数据语义。
    return (
        global_predictions[order],
        global_targets[order],
        global_indices[order],
    )


# 执行当前分布式训练步骤并保持既定数据语义。
def gather_global_inference_tensor(
    tensor: torch.Tensor,
    context: distributed.DistributedContext | None,
) -> torch.Tensor:
    """执行 gather_global_inference_tensor 对应的分布式训练处理。
    
    输入：
        参数见函数类型标注。
    输出：
        结果见函数返回类型标注。
    """
    # 执行当前分布式训练步骤并保持既定数据语义。
    return tensor if context is None else distributed.all_gather_tensor(tensor)



# 读取 `Path(__file__).resolve().parent` 的当前值，并赋给 `VIVIT_DIR`。
VIVIT_DIR = Path(__file__).resolve().parent
# 计算 `VIVIT_DIR.parent / 'ViT'`，并将数值结果保存到 `VIT_DIR`。
VIT_DIR = VIVIT_DIR.parent / "ViT"
# 检查 `str(VIT_DIR) not in sys.path`；根据条件结果选择对应处理分支。
if str(VIT_DIR) not in sys.path:
    # 调用 `sys.path.insert` 把项目模块目录加入导入搜索路径。
    sys.path.insert(0, str(VIT_DIR))

from vit_stage06_ranking import (  # noqa: E402
    LOSS_NAMES,
    STANDARD_RANK_LOSS_NAMES,
    TEMPORAL_TURNOVER_LOSS,
    compute_training_loss,
    pearson_correlation,
    temporal_turnover_loss_components,
)
from vit_stage07_training import (  # noqa: E402
    TRAIN_BATCH_MODES,
    VALIDATION_METRICS,
    TradeDateBatchSampler,
    summarize_validation_arrays,
)
# 组织一组有序值，并保存到 `RANK_LOSS_NAMES`。
RANK_LOSS_NAMES = (*STANDARD_RANK_LOSS_NAMES, TEMPORAL_TURNOVER_LOSS)
from stage01_sequence_data import (  # noqa: E402
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
import stage05_distributed_training as distributed  # noqa: E402
from stage03_model import (  # noqa: E402
    OrderFlowViViT,
    VivitModelConfig,
)

# 计算 `VIVIT_DIR / 'data'`，并将数值结果保存到 `DATA_DIR`。
DATA_DIR = VIVIT_DIR / "data"
# 计算 `DATA_DIR / 'feature_store'`，并将数值结果保存到 `FEATURE_STORE_DIR`。
FEATURE_STORE_DIR = DATA_DIR / "feature_store"
# 计算 `DATA_DIR / 'aligned_sequence_store'`，并将数值结果保存到 `SEQUENCE_STORE_DIR`。
SEQUENCE_STORE_DIR = DATA_DIR / "aligned_sequence_store"
# 计算 `SEQUENCE_STORE_DIR / 'history_indices.npy'`，并将数值结果保存到 `HISTORY_INDICES_PATH`。
HISTORY_INDICES_PATH = SEQUENCE_STORE_DIR / "history_indices.npy"
# 计算 `SEQUENCE_STORE_DIR / 'sequence_manifest.json'`，并将数值结果保存到 `SEQUENCE_MANIFEST_PATH`。
SEQUENCE_MANIFEST_PATH = SEQUENCE_STORE_DIR / "sequence_manifest.json"
# 计算 `SEQUENCE_STORE_DIR / 'rolling_split_summary.parquet'`，并将数值结果保存到 `SEQUENCE_SUMMARY_PATH`。
SEQUENCE_SUMMARY_PATH = SEQUENCE_STORE_DIR / "rolling_split_summary.parquet"

# 计算 `FEATURE_STORE_DIR / 'features.npy'`，并将数值结果保存到 `FEATURES_PATH`。
FEATURES_PATH = FEATURE_STORE_DIR / "features.npy"
# 计算 `FEATURE_STORE_DIR / 'targets.npy'`，并将数值结果保存到 `TARGETS_PATH`。
TARGETS_PATH = FEATURE_STORE_DIR / "targets.npy"
# 计算 `FEATURE_STORE_DIR / 'date_ids.npy'`，并将数值结果保存到 `DATE_IDS_PATH`。
DATE_IDS_PATH = FEATURE_STORE_DIR / "date_ids.npy"
# 计算 `FEATURE_STORE_DIR / 'sample_index.parquet'`，并将数值结果保存到 `SAMPLE_INDEX_PATH`。
SAMPLE_INDEX_PATH = FEATURE_STORE_DIR / "sample_index.parquet"
# 计算 `FEATURE_STORE_DIR / 'dataset_manifest.json'`，并将数值结果保存到 `SOURCE_DATASET_MANIFEST_PATH`。
SOURCE_DATASET_MANIFEST_PATH = FEATURE_STORE_DIR / "dataset_manifest.json"
# 计算 `FEATURE_STORE_DIR / 'rolling_split_manifest.json'`，并将数值结果保存到 `SOURCE_ROLLING_MANIFEST_PATH`。
SOURCE_ROLLING_MANIFEST_PATH = FEATURE_STORE_DIR / "rolling_split_manifest.json"
# 计算 `FEATURE_STORE_DIR / 'rolling_splits'`，并将数值结果保存到 `SOURCE_SPLIT_DIR`。
SOURCE_SPLIT_DIR = FEATURE_STORE_DIR / "rolling_splits"

# 调用 `tuple` 整理数据结构，并将结果保存到 `FORECAST_YEARS`。
FORECAST_YEARS = tuple(range(2018, 2026))
# 组织一组有序值，并保存到 `SEEDS`。
SEEDS = (42, 43, 44)
# 设置 `SEQUENCE_LENGTH` 的固定配置值，供后续流程统一使用。
SEQUENCE_LENGTH = 20



# 使用 `dataclass(frozen=True)` 包装 `TrainingConfig`，并定义其可复用的处理逻辑。
@dataclass(frozen=True)
class TrainingConfig:
    """研报训练参数与项目复现参数。
    """

    # 设置 `train_batch_mode` 的固定配置值，供后续流程统一使用。
    train_batch_mode: str = "trade_date"
    # 设置 `batch_size` 的固定配置值，供后续流程统一使用。
    batch_size: int = 5000
    # 设置 `inference_batch_size` 的固定配置值，供后续流程统一使用。
    inference_batch_size: int = 5000
    # 设置 `loss_name` 的固定配置值，供后续流程统一使用。
    loss_name: str = "pearson_ic"
    # 设置 `rank_temperature` 的固定配置值，供后续流程统一使用。
    rank_temperature: float = 0.1
    # 设置 `turnover_lambda` 的固定配置值，供后续流程统一使用。
    turnover_lambda: float = 0.1
    # 设置 `huber_delta` 的固定配置值，供后续流程统一使用。
    huber_delta: float = 1.0
    # 设置 `smooth_l1_beta` 的固定配置值，供后续流程统一使用。
    smooth_l1_beta: float = 0.1
    # 设置 `hybrid_weight` 的固定配置值，供后续流程统一使用。
    hybrid_weight: float = 0.5
    # 设置 `validation_metric` 的固定配置值，供后续流程统一使用。
    validation_metric: str = "daily_mean_pearson_ic"
    # 设置 `learning_rate` 的固定配置值，供后续流程统一使用。
    learning_rate: float = 0.001
    # 设置 `max_epochs` 的固定配置值，供后续流程统一使用。
    max_epochs: int = 100
    # 设置 `early_stopping_patience` 的固定配置值，供后续流程统一使用。
    early_stopping_patience: int = 10
    # 设置 `adam_beta1` 的固定配置值，供后续流程统一使用。
    adam_beta1: float = 0.9
    # 设置 `adam_beta2` 的固定配置值，供后续流程统一使用。
    adam_beta2: float = 0.999
    # 设置 `adam_epsilon` 的固定配置值，供后续流程统一使用。
    adam_epsilon: float = 1e-8
    # 设置 `weight_decay` 的固定配置值，供后续流程统一使用。
    weight_decay: float = 0.0
    # 设置 `improvement_tolerance` 的固定配置值，供后续流程统一使用。
    improvement_tolerance: float = 1e-6


# 调用 `VivitModelConfig` 整理并返回当前实验的配置对象，并将结果保存到 `MODEL_CONFIG`。
MODEL_CONFIG = VivitModelConfig()


# 定义 `set_random_seed` 函数，封装这一阶段可独立调用的处理步骤。
def set_random_seed(seed: int) -> None:
    """同步设置 Python、NumPy、CPU 与 CUDA 随机种子。
    
    输入：
        seed: int。
    输出：
        None。
        """
    # 调用 `random.seed` 设置当前随机数生成器种子。
    random.seed(seed)
    # 调用 `np.random.seed` 设置当前随机数生成器种子。
    np.random.seed(seed)
    # 调用 `torch.manual_seed` 设置 PyTorch 随机种子以保证结果可复现。
    torch.manual_seed(seed)
    # 检查 `torch.cuda.is_available()`；根据条件结果选择对应处理分支。
    if torch.cuda.is_available():
        # 调用 `torch.cuda.manual_seed_all` 执行 `manual_seed_all` 对应的明确变换，并产生后续步骤使用的结果。
        torch.cuda.manual_seed_all(seed)


# 定义 `unwrap_model` 函数，封装这一阶段可独立调用的处理步骤。
def unwrap_model(model: nn.Module) -> nn.Module:
    """从编译或包装模型中取回原始 PyTorch 模型。
    
    输入：
        model: nn.Module。
    输出：
        nn.Module。
        """
    # 执行当前分布式训练步骤并保持既定数据语义。
    wrapped = getattr(model, "module", None)
    # 执行当前分布式训练步骤并保持既定数据语义。
    if wrapped is not None:
        # 执行当前分布式训练步骤并保持既定数据语义。
        return unwrap_model(wrapped)
    # 调用 `getattr` 读取对象的可选属性并应用缺省值，并将结果保存到 `original`。
    original = getattr(model, "_orig_mod", None)
    # 将 `unwrap_model(original) if original is not None else model` 作为当前函数结果返回给调用方。
    return unwrap_model(original) if original is not None else model


# 定义 `build_model` 函数，封装这一阶段可独立调用的处理步骤。
def build_model(
    device: torch.device,
    use_compile: bool,
    compile_mode: str,
    train_batch_mode: str = "trade_date",
) -> nn.Module:
    """按当前配置构建模型，并保持历史 checkpoint 的参数键兼容。
    
    输入：
        device: torch.device。
        use_compile: bool。
        compile_mode: str。
        train_batch_mode: str。
    输出：
        nn.Module。
        """
    # 调用 `OrderFlowViViT(MODEL_CONFIG).to` 统一设备或数据类型，并将结果保存到 `model`。
    model = OrderFlowViViT(MODEL_CONFIG).to(device)
    # 检查 `use_compile`；根据条件结果选择对应处理分支。
    if use_compile:
        # 检查 `device.type != 'cuda'`；根据条件结果选择对应处理分支。
        if device.type != "cuda":
            # 抛出 `RuntimeError('torch.compile 训练要求 CUDA')`，阻止不满足数据契约的流程继续执行。
            raise RuntimeError("torch.compile 训练要求 CUDA")
        # 检查 `train_batch_mode == 'trade_date'`；根据条件结果选择对应处理分支。
        if train_batch_mode == "trade_date":
            # 调用 `torch.compile` 编译模型计算图以减少重复执行开销，并将结果保存到 `model`。
            model = torch.compile(model, mode=compile_mode, dynamic=True)
        # 当前置条件不成立时进入备用分支，保证其余合法情况仍被处理。
        else:
            # 调用 `torch.compile` 编译模型计算图以减少重复执行开销，并将结果保存到 `model`。
            model = torch.compile(model, mode=compile_mode)
    # 将 `model` 作为当前函数结果返回给调用方。
    return model


# 定义 `autocast_context` 函数，封装这一阶段可独立调用的处理步骤。
def autocast_context(device: torch.device) -> contextlib.AbstractContextManager:
    """按设备和配置返回混合精度上下文。
    
    输入：
        device: torch.device。
    输出：
        未显式标注。
        """
    # 检查 `device.type == 'cuda' and torch.cuda.is_bf16_supported()`；根据条件结果选择对应处理分支。
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        # 将 `torch.autocast(device_type='cuda', dtype=torch.bfloat16)` 作为当前函数结果返回给调用方。
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    # 将 `contextlib.nullcontext()` 作为当前函数结果返回给调用方。
    return contextlib.nullcontext()


# 定义 `CudaBatchPrefetcher` 类，集中封装该阶段的数据结构与相关行为。
class CudaBatchPrefetcher:
    """用独立 CUDA stream 预取下一批视频、掩码与标签。
    """

    # 定义初始化方法，接收配置并建立实例后续计算所需的内部状态。
    def __init__(self, loader: DataLoader, device: torch.device, enabled: bool) -> None:
        """初始化 CudaBatchPrefetcher 所需的数据引用、配置和运行状态。
        
        输入：
            loader: DataLoader。
            device: torch.device。
            enabled: bool。
        输出：
            与签名返回类型一致。
            """
        # 读取 `loader` 的当前值，并赋给 `self.loader`。
        self.loader = loader
        # 读取 `device` 的当前值，并赋给 `self.device`。
        self.device = device
        # 根据 `enabled and device.type == 'cuda'` 计算条件结果，并保存到 `self.enabled`。
        self.enabled = enabled and device.type == "cuda"

    # 定义内部辅助函数 `_move`，封装仅供本模块复用的计算步骤。
    def _move(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """把批次张量非阻塞搬到 CUDA，并保留非张量元数据。
        
        输入：
            batch: dict[str, torch.Tensor]。
        输出：
            与签名返回类型一致。
            """
        # 根据 `self.device.type == 'cuda'` 计算条件结果，并保存到 `non_blocking`。
        non_blocking = self.device.type == "cuda"
        # 将 `(batch['videos'].to(self.device, non_blocking=non_blocking), batch['frame_mask'].to(self.device…` 作为当前函数结果返回给调用方。
        return (
            batch["videos"].to(self.device, non_blocking=non_blocking),
            batch["frame_mask"].to(self.device, non_blocking=non_blocking),
            batch["targets"].to(self.device, non_blocking=non_blocking),
        )

    # 定义内部辅助函数 `__iter__`，封装仅供本模块复用的计算步骤。
    def __iter__(self) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]]:
        """按既定顺序迭代 CudaBatchPrefetcher 中的训练批次。
        
        输入：
            无显式输入参数。
        输出：
            与签名返回类型一致。
            """
        # 检查 `not self.enabled`；根据条件结果选择对应处理分支。
        if not self.enabled:
            # 遍历 `self.loader`，并把当前元素绑定到 `batch` 逐项处理。
            for batch in self.loader:
                # 调用 `self._move` 执行 `_move` 对应的明确变换，并产生后续步骤使用的结果，并将结果保存到 `(videos, frame_mask, targets)`。
                videos, frame_mask, targets = self._move(batch)
                # 对 `(yield (batch, videos, frame_mask, targets))` 求值，以触发当前步骤需要的计算。
                yield batch, videos, frame_mask, targets
            # 将 `空值` 作为当前函数结果返回给调用方。
            return

        # 调用 `iter` 为 DataLoader 创建顺序取批迭代器，并将结果保存到 `iterator`。
        iterator = iter(self.loader)
        # 调用 `torch.cuda.Stream` 创建独立 CUDA stream 用于异步数据预取，并将结果保存到 `stream`。
        stream = torch.cuda.Stream(device=self.device)
        # 设置 `next_item` 的固定配置值，供后续流程统一使用。
        next_item = None

        # 定义 `preload` 函数，封装这一阶段可独立调用的处理步骤。
        def preload() -> None:
            """把下一批数据异步搬到 GPU，供当前批计算后立即使用。
            
            输入：
                无显式输入参数。
            输出：
                None。
                """
            # 声明 `next_item` 复用外层函数状态，供闭包更新。
            nonlocal next_item
            # 尝试执行本段主逻辑；若发生异常，则交由后续异常分支处理。
            try:
                # 调用 `next` 从迭代器取出下一批数据，并将结果保存到 `batch`。
                batch = next(iterator)
            # 捕获 `except StopIteration` 对应的异常，并执行明确的降级、记录或清理逻辑。
            except StopIteration:
                # 设置 `next_item` 的固定配置值，供后续流程统一使用。
                next_item = None
                # 将 `空值` 作为当前函数结果返回给调用方。
                return
            # 在 `torch.cuda.stream(stream)` 管理的上下文中执行本段逻辑，并保证资源按约定释放。
            with torch.cuda.stream(stream):
                # 调用 `self._move` 执行 `_move` 对应的明确变换，并产生后续步骤使用的结果，并将结果保存到 `(videos, frame_mask, targets)`。
                videos, frame_mask, targets = self._move(batch)
            # 组织一组有序值，并保存到 `next_item`。
            next_item = (batch, videos, frame_mask, targets)

        # 调用 `preload` 把下一批数据异步预取到 GPU。
        preload()
        # 只要 `next_item is not None` 成立就继续循环，直至满足退出条件。
        while next_item is not None:
            # 调用 `torch.cuda.current_stream(self.device).wait_stream` 等待预取 stream 完成后再消费张量。
            torch.cuda.current_stream(self.device).wait_stream(stream)
            # 读取 `next_item` 的当前值，并赋给 `(batch, videos, frame_mask, targets)`。
            batch, videos, frame_mask, targets = next_item
            # 遍历 `(videos, frame_mask, targets)`，并把当前元素绑定到 `tensor` 逐项处理。
            for tensor in (videos, frame_mask, targets):
                # 调用 `tensor.record_stream` 把张量登记到当前 CUDA stream，避免异步使用期间被回收。
                tensor.record_stream(torch.cuda.current_stream(self.device))
            # 调用 `preload` 把下一批数据异步预取到 GPU。
            preload()
            # 对 `(yield (batch, videos, frame_mask, targets))` 求值，以触发当前步骤需要的计算。
            yield batch, videos, frame_mask, targets


# 定义 `loss_kwargs` 函数，封装这一阶段可独立调用的处理步骤。
def loss_kwargs(config: TrainingConfig) -> dict[str, object]:
    """整理基础损失需要的温度、稳健回归和换手超参数。
    
    输入：
        config: TrainingConfig。
    输出：
        dict[str, object]。
        """
    # 将 `{'loss_name': config.loss_name, 'rank_temperature': config.rank_temperature, 'turnover_lambda':…` 作为当前函数结果返回给调用方。
    return {
        "loss_name": config.loss_name,
        "rank_temperature": config.rank_temperature,
        "turnover_lambda": config.turnover_lambda,
        "huber_delta": config.huber_delta,
        "smooth_l1_beta": config.smooth_l1_beta,
        "hybrid_weight": config.hybrid_weight,
    }


# 定义内部辅助函数 `_train_standard_epoch`，封装仅供本模块复用的计算步骤。
def _train_standard_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_prefetch: bool,
    config: TrainingConfig,
    max_batches: int | None = None,
    distributed_context: distributed.DistributedContext | None = None,
) -> dict[str, object]:
    """执行该模式下的训练步骤，并返回损失与诊断信息。
    
    输入：
        model: nn.Module。
        loader: DataLoader。
        optimizer: torch.optim.Optimizer。
        device: torch.device。
        use_prefetch: bool。
        config: TrainingConfig。
        max_batches: int | None。
    输出：
        dict[str, object]。
        """
    # 调用 `model.train` 将模型切换到训练模式，启用训练期行为。
    model.train()
    # 调用 `isinstance` 执行 `isinstance` 对应的明确变换，并产生后续步骤使用的结果，并将结果保存到 `date_batch_mode`。
    date_batch_mode = isinstance(loader.batch_sampler, TradeDateBatchSampler)
    # 调用 `torch.zeros` 按目标形状初始化数值容器 `loss_sum`。
    loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    # 调用 `torch.zeros` 按目标形状初始化数值容器 `batch_pearson_sum`。
    batch_pearson_sum = torch.zeros((), dtype=torch.float64, device=device)
    # 调用 `torch.zeros` 按目标形状初始化数值容器 `weighted_pearson`。
    weighted_pearson = torch.zeros((), dtype=torch.float64, device=device)
    # 调用 `torch.zeros` 按目标形状初始化数值容器 `soft_rank_ic_sum`。
    soft_rank_ic_sum = torch.zeros((), dtype=torch.float64, device=device)
    # 设置 `sample_count` 的固定配置值，供后续流程统一使用。
    sample_count = 0
    # 设置 `batch_count` 的固定配置值，供后续流程统一使用。
    batch_count = 0
    # 设置 `soft_rank_batch_count` 的固定配置值，供后续流程统一使用。
    soft_rank_batch_count = 0
    # 组织一组有序值，并保存到 `(min_batch_size, max_batch_size)`。
    min_batch_size, max_batch_size = None, 0
    # 调用 `time.perf_counter` 读取高精度单调时钟以计算耗时，并将结果保存到 `started`。
    started = time.perf_counter()

    # 遍历 `enumerate(CudaBatchPrefetcher(loader, device, use_prefetch), start=1)`，并把当前元素绑定到 `(batch_number, (batch, videos, frame_mask, targets))` 逐项处理。
    for batch_number, (batch, videos, frame_mask, targets) in enumerate(
        CudaBatchPrefetcher(loader, device, use_prefetch), start=1
    ):
        # 检查 `date_batch_mode`；根据条件结果选择对应处理分支。
        if date_batch_mode:
            # 读取 `batch['date_ids']` 的当前值，并赋给 `date_ids`。
            date_ids = batch["date_ids"]
            # 检查 `int(date_ids.min()) != int(date_ids.max())`；根据条件结果选择对应处理分支。
            if int(date_ids.min()) != int(date_ids.max()):
                # 抛出 `AssertionError('日截面训练 batch 混入多个交易日')`，阻止不满足数据契约的流程继续执行。
                raise AssertionError("日截面训练 batch 混入多个交易日")
        # 调用 `optimizer.zero_grad` 清空上一轮梯度，避免跨批次意外累积。
        optimizer.zero_grad(set_to_none=True)
        # 在 `autocast_context(device)` 管理的上下文中执行本段逻辑，并保证资源按约定释放。
        with autocast_context(device):
            # 调用 `model` 对当前输入 batch 执行模型前向传播，并将结果保存到 `predictions`。
            predictions = model(videos, frame_mask)
        # 执行当前分布式训练步骤并保持既定数据语义。
        predictions, targets, _ = gather_global_training_batch(
            predictions, targets, batch, distributed_context
        )
        # 调用 `compute_training_loss` 按当前实验口径计算可反向传播的训练损失，并将结果保存到 `loss`。
        loss = compute_training_loss(predictions, targets, **loss_kwargs(config))
        # 检查 `not torch.isfinite(loss)`；根据条件结果选择对应处理分支。
        if not torch.isfinite(loss):
            # 抛出 `FloatingPointError('训练损失出现 NaN 或 Inf')`，阻止不满足数据契约的流程继续执行。
            raise FloatingPointError("训练损失出现 NaN 或 Inf")
        # 调用 `loss.backward` 反向传播当前损失，计算模型参数梯度。
        loss.backward()
        # 调用 `optimizer.step` 推进一次优化器或调度器状态。
        optimizer.step()

        # 在 `torch.no_grad()` 管理的上下文中执行本段逻辑，并保证资源按约定释放。
        with torch.no_grad():
            # 调用 `pearson_correlation(predictions, targets).double` 执行模型推理并生成样本预测分数，并将结果保存到 `batch_pearson`。
            batch_pearson = pearson_correlation(predictions, targets).double()
        # 调用 `len` 汇总当前数据，并将统计量保存到 `current_size`。
        current_size = len(targets)
        # 用 `loss.detach().double()` 对 `loss_sum` 做 `Add` 累积更新。
        loss_sum += loss.detach().double()
        # 用 `batch_pearson` 对 `batch_pearson_sum` 做 `Add` 累积更新。
        batch_pearson_sum += batch_pearson
        # 用 `batch_pearson * current_size` 对 `weighted_pearson` 做 `Add` 累积更新。
        weighted_pearson += batch_pearson * current_size
        # 检查 `config.loss_name in RANK_LOSS_NAMES`；根据条件结果选择对应处理分支。
        if config.loss_name in RANK_LOSS_NAMES:
            # 用 `-loss.detach().double()` 对 `soft_rank_ic_sum` 做 `Add` 累积更新。
            soft_rank_ic_sum += -loss.detach().double()
            # 用 `1` 对 `soft_rank_batch_count` 做 `Add` 累积更新。
            soft_rank_batch_count += 1
        # 用 `current_size` 对 `sample_count` 做 `Add` 累积更新。
        sample_count += current_size
        # 用 `1` 对 `batch_count` 做 `Add` 累积更新。
        batch_count += 1
        # 按 `min_batch_size is None` 在两个候选值间选择，并保存到 `min_batch_size`。
        min_batch_size = current_size if min_batch_size is None else min(
            min_batch_size, current_size
        )
        # 调用 `max` 汇总当前数据，并将统计量保存到 `max_batch_size`。
        max_batch_size = max(max_batch_size, current_size)
        # 检查 `max_batches is not None and batch_number >= max_batches`；根据条件结果选择对应处理分支。
        if max_batches is not None and batch_number >= max_batches:
            # 终止当前循环，避免继续处理已经不满足条件的元素。
            break

    # 检查 `not batch_count`；根据条件结果选择对应处理分支。
    if not batch_count:
        # 抛出 `ValueError('训练 loader 没有可用 batch')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("训练 loader 没有可用 batch")
    # 检查 `device.type == 'cuda'`；根据条件结果选择对应处理分支。
    if device.type == "cuda":
        # 调用 `torch.cuda.synchronize` 等待当前 GPU 队列执行完毕后再计时或验收。
        torch.cuda.synchronize(device)
    # 调用 `float` 统一设备或数据类型，并将结果保存到 `batch_mean_pearson`。
    batch_mean_pearson = float((batch_pearson_sum / batch_count).item())
    # 调用 `float` 统一设备或数据类型，并将结果保存到 `sample_weighted_pearson`。
    sample_weighted_pearson = float((weighted_pearson / sample_count).item())
    # 按 `date_batch_mode` 在两个候选值间选择，并保存到 `selected_train_pearson`。
    selected_train_pearson = (
        batch_mean_pearson if date_batch_mode else sample_weighted_pearson
    )
    # 按 `soft_rank_batch_count` 在两个候选值间选择，并保存到 `soft_rank_ic`。
    soft_rank_ic = (
        float((soft_rank_ic_sum / soft_rank_batch_count).item())
        if soft_rank_batch_count
        else float("nan")
    )
    # 将 `{'train_ic': selected_train_pearson, 'train_loss': float((loss_sum / batch_count).item()), 'tra…` 作为当前函数结果返回给调用方。
    return {
        "train_ic": selected_train_pearson,
        "train_loss": float((loss_sum / batch_count).item()),
        "train_batch_pearson_ic_mean": batch_mean_pearson,
        "train_sample_weighted_pearson_ic": sample_weighted_pearson,
        "train_soft_rank_ic": soft_rank_ic,
        "train_prediction_soft_rank_ic": soft_rank_ic,
        "train_temporal_soft_rank_ic": float("nan"),
        "train_temporal_penalty": float("nan"),
        "train_temporal_pairs": 0,
        "train_date_batches": batch_count if date_batch_mode else None,
        "gradient_update_mode": "per_batch",
        "optimizer_steps": batch_count,
        "train_samples": sample_count,
        "train_batches": batch_count,
        "train_batch_size_min": int(min_batch_size),
        "train_batch_size_max": int(max_batch_size),
        "train_seconds": time.perf_counter() - started,
    }


# 定义内部辅助函数 `_train_temporal_turnover_epoch`，封装仅供本模块复用的计算步骤。
def _train_temporal_turnover_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_prefetch: bool,
    config: TrainingConfig,
    max_batches: int | None = None,
    distributed_context: distributed.DistributedContext | None = None,
) -> dict[str, object]:
    """执行该模式下的训练步骤，并返回损失与诊断信息。
    
    输入：
        model: nn.Module。
        loader: DataLoader。
        optimizer: torch.optim.Optimizer。
        device: torch.device。
        use_prefetch: bool。
        config: TrainingConfig。
        max_batches: int | None。
    输出：
        dict[str, object]。
        """
    # 读取 `loader.batch_sampler` 的当前值，并赋给 `sampler`。
    sampler = loader.batch_sampler
    # 检查 `not isinstance(sampler, TradeDateBatchSampler) or sampler.temporal_pairs is None or sampler.shu…`；根据条件结果选择对应处理分支。
    if (
        not isinstance(sampler, TradeDateBatchSampler)
        or sampler.temporal_pairs is None
        or sampler.shuffle
    ):
        # 抛出 `ValueError('时序换手损失要求按日期升序且关闭 reshuffle')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("时序换手损失要求按日期升序且关闭 reshuffle")
    # 调用 `len` 汇总当前数据，并将统计量保存到 `available_pairs`。
    available_pairs = len(sampler.temporal_pairs)
    # 按 `max_batches is None` 在两个候选值间选择，并保存到 `pair_limit`。
    pair_limit = available_pairs if max_batches is None else min(
        int(max_batches), available_pairs
    )
    # 检查 `pair_limit < 1`；根据条件结果选择对应处理分支。
    if pair_limit < 1:
        # 抛出 `ValueError('时序换手训练至少需要一个相邻日期对')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("时序换手训练至少需要一个相邻日期对")

    # 调用 `model.train` 将模型切换到训练模式，启用训练期行为。
    model.train()
    # 调用 `optimizer.zero_grad` 清空上一轮梯度，避免跨批次意外累积。
    optimizer.zero_grad(set_to_none=True)
    # 调用 `torch.zeros` 按目标形状初始化数值容器 `loss_sum`。
    loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    # 调用 `torch.zeros` 按目标形状初始化数值容器 `pearson_sum`。
    pearson_sum = torch.zeros((), dtype=torch.float64, device=device)
    # 调用 `torch.zeros` 按目标形状初始化数值容器 `weighted_pearson`。
    weighted_pearson = torch.zeros((), dtype=torch.float64, device=device)
    # 调用 `torch.zeros` 按目标形状初始化数值容器 `prediction_rank_sum`。
    prediction_rank_sum = torch.zeros((), dtype=torch.float64, device=device)
    # 调用 `torch.zeros` 按目标形状初始化数值容器 `temporal_rank_sum`。
    temporal_rank_sum = torch.zeros((), dtype=torch.float64, device=device)
    # 调用 `torch.zeros` 按目标形状初始化数值容器 `temporal_penalty_sum`。
    temporal_penalty_sum = torch.zeros((), dtype=torch.float64, device=device)
    # 设置 `sample_count` 的固定配置值，供后续流程统一使用。
    sample_count = 0
    # 设置 `pair_count` 的固定配置值，供后续流程统一使用。
    pair_count = 0
    # 组织一组有序值，并保存到 `(min_batch_size, max_batch_size)`。
    min_batch_size, max_batch_size = None, 0
    # 设置 `previous_predictions` 的固定配置值，供后续流程统一使用。
    previous_predictions = None
    # 设置 `previous_date_id` 的固定配置值，供后续流程统一使用。
    previous_date_id = None
    # 调用 `time.perf_counter` 读取高精度单调时钟以计算耗时，并将结果保存到 `started`。
    started = time.perf_counter()

    # 遍历 `CudaBatchPrefetcher(loader, device, use_prefetch)`，并把当前元素绑定到 `(batch, videos, frame_mask, targets)` 逐项处理。
    for batch, videos, frame_mask, targets in CudaBatchPrefetcher(
        loader, device, use_prefetch
    ):
        # 读取 `batch['date_ids']` 的当前值，并赋给 `date_ids`。
        date_ids = batch["date_ids"]
        # 检查 `int(date_ids.min()) != int(date_ids.max())`；根据条件结果选择对应处理分支。
        if int(date_ids.min()) != int(date_ids.max()):
            # 抛出 `AssertionError('日截面训练 batch 混入多个交易日')`，阻止不满足数据契约的流程继续执行。
            raise AssertionError("日截面训练 batch 混入多个交易日")
        # 调用 `int` 统一设备或数据类型，并将结果保存到 `current_date_id`。
        current_date_id = int(date_ids[0])
        # 在 `autocast_context(device)` 管理的上下文中执行本段逻辑，并保证资源按约定释放。
        with autocast_context(device):
            # 调用 `model` 对当前输入 batch 执行模型前向传播，并将结果保存到 `current_predictions`。
            current_predictions = model(videos, frame_mask)
        # 执行当前分布式训练步骤并保持既定数据语义。
        current_predictions, targets, _ = gather_global_training_batch(
            current_predictions, targets, batch, distributed_context
        )
        # 检查 `previous_predictions is None`；根据条件结果选择对应处理分支。
        if previous_predictions is None:
            # 复制 `current_predictions.detach`，避免后续更新污染原对象，并保存到 `previous_predictions`。
            previous_predictions = current_predictions.detach()
            # 读取 `current_date_id` 的当前值，并赋给 `previous_date_id`。
            previous_date_id = current_date_id
            # 删除 `current_predictions`，释放不再使用的引用。
            del current_predictions
            # 跳过当前元素的剩余步骤，继续处理下一项。
            continue
        # 检查 `current_date_id != previous_date_id + 1`；根据条件结果选择对应处理分支。
        if current_date_id != previous_date_id + 1:
            # 抛出 `AssertionError(f'时序训练日期不连续：{previous_date_id}→{current_date_id}')`，阻止不满足数据契约的流程继续执行。
            raise AssertionError(
                f"时序训练日期不连续：{previous_date_id}→{current_date_id}"
            )

        # 调用 `sampler.temporal_pair` 执行 `temporal_pair` 对应的明确变换，并产生后续步骤使用的结果，并将结果保存到 `pair`。
        pair = sampler.temporal_pair(pair_count)
        # 调用 `temporal_turnover_loss_components` 按当前实验口径计算可反向传播的训练损失，并将结果保存到 `components`。
        components = temporal_turnover_loss_components(
            current_predictions,
            targets,
            previous_predictions,
            pair["previous_positions"],
            pair["current_positions"],
            config.rank_temperature,
            config.turnover_lambda,
        )
        # 读取 `components['loss']` 的当前值，并赋给 `loss`。
        loss = components["loss"]
        # 调用 `torch.stack` 拼接各批结果，并将合并值保存到 `values`。
        values = torch.stack(
            [
                loss,
                components["prediction_soft_rank_ic"],
                components["temporal_soft_rank_ic"],
                components["temporal_penalty"],
            ]
        )
        # 检查 `not torch.isfinite(values).all()`；根据条件结果选择对应处理分支。
        if not torch.isfinite(values).all():
            # 抛出 `FloatingPointError('时序换手训练损失或审计指标出现 NaN/Inf')`，阻止不满足数据契约的流程继续执行。
            raise FloatingPointError("时序换手训练损失或审计指标出现 NaN/Inf")
        # 调用 `(loss / pair_limit).backward` 反向传播当前损失，计算模型参数梯度。
        (loss / pair_limit).backward()

        # 在 `torch.no_grad()` 管理的上下文中执行本段逻辑，并保证资源按约定释放。
        with torch.no_grad():
            # 调用 `pearson_correlation(current_predictions, targets).double` 执行模型推理并生成样本预测分数，并将结果保存到 `batch_pearson`。
            batch_pearson = pearson_correlation(current_predictions, targets).double()
        # 调用 `len` 汇总当前数据，并将统计量保存到 `current_size`。
        current_size = len(targets)
        # 用 `loss.detach().double()` 对 `loss_sum` 做 `Add` 累积更新。
        loss_sum += loss.detach().double()
        # 用 `batch_pearson` 对 `pearson_sum` 做 `Add` 累积更新。
        pearson_sum += batch_pearson
        # 用 `batch_pearson * current_size` 对 `weighted_pearson` 做 `Add` 累积更新。
        weighted_pearson += batch_pearson * current_size
        # 用 `components['prediction_soft_rank_ic'].detach().double()` 对 `prediction_rank_sum` 做 `Add` 累积更新。
        prediction_rank_sum += components["prediction_soft_rank_ic"].detach().double()
        # 用 `components['temporal_soft_rank_ic'].detach().double()` 对 `temporal_rank_sum` 做 `Add` 累积更新。
        temporal_rank_sum += components["temporal_soft_rank_ic"].detach().double()
        # 用 `components['temporal_penalty'].detach().double()` 对 `temporal_penalty_sum` 做 `Add` 累积更新。
        temporal_penalty_sum += components["temporal_penalty"].detach().double()
        # 用 `current_size` 对 `sample_count` 做 `Add` 累积更新。
        sample_count += current_size
        # 用 `1` 对 `pair_count` 做 `Add` 累积更新。
        pair_count += 1
        # 按 `min_batch_size is None` 在两个候选值间选择，并保存到 `min_batch_size`。
        min_batch_size = current_size if min_batch_size is None else min(
            min_batch_size, current_size
        )
        # 调用 `max` 汇总当前数据，并将统计量保存到 `max_batch_size`。
        max_batch_size = max(max_batch_size, current_size)
        # 复制 `current_predictions.detach`，避免后续更新污染原对象，并保存到 `previous_predictions`。
        previous_predictions = current_predictions.detach()
        # 读取 `current_date_id` 的当前值，并赋给 `previous_date_id`。
        previous_date_id = current_date_id
        # 删除 `current_predictions, components, loss, values`，释放不再使用的引用。
        del current_predictions, components, loss, values
        # 检查 `pair_count >= pair_limit`；根据条件结果选择对应处理分支。
        if pair_count >= pair_limit:
            # 终止当前循环，避免继续处理已经不满足条件的元素。
            break

    # 检查 `pair_count != pair_limit`；根据条件结果选择对应处理分支。
    if pair_count != pair_limit:
        # 抛出 `AssertionError(f'时序日期对覆盖不完整：{pair_count}/{pair_limit}')`，阻止不满足数据契约的流程继续执行。
        raise AssertionError(f"时序日期对覆盖不完整：{pair_count}/{pair_limit}")
    # 调用 `optimizer.step` 推进一次优化器或调度器状态。
    optimizer.step()
    # 检查 `device.type == 'cuda'`；根据条件结果选择对应处理分支。
    if device.type == "cuda":
        # 调用 `torch.cuda.synchronize` 等待当前 GPU 队列执行完毕后再计时或验收。
        torch.cuda.synchronize(device)
    # 调用 `float` 统一设备或数据类型，并将结果保存到 `mean_pearson`。
    mean_pearson = float((pearson_sum / pair_count).item())
    # 调用 `float` 统一设备或数据类型，并将结果保存到 `weighted_mean_pearson`。
    weighted_mean_pearson = float((weighted_pearson / sample_count).item())
    # 调用 `float` 统一设备或数据类型，并将结果保存到 `prediction_soft_rank`。
    prediction_soft_rank = float((prediction_rank_sum / pair_count).item())
    # 调用 `float` 统一设备或数据类型，并将结果保存到 `temporal_soft_rank`。
    temporal_soft_rank = float((temporal_rank_sum / pair_count).item())
    # 调用 `float` 统一设备或数据类型，并将结果保存到 `temporal_penalty`。
    temporal_penalty = float((temporal_penalty_sum / pair_count).item())
    # 将 `{'train_ic': mean_pearson, 'train_loss': float((loss_sum / pair_count).item()), 'train_batch_pe…` 作为当前函数结果返回给调用方。
    return {
        "train_ic": mean_pearson,
        "train_loss": float((loss_sum / pair_count).item()),
        "train_batch_pearson_ic_mean": mean_pearson,
        "train_sample_weighted_pearson_ic": weighted_mean_pearson,
        "train_soft_rank_ic": prediction_soft_rank,
        "train_prediction_soft_rank_ic": prediction_soft_rank,
        "train_temporal_soft_rank_ic": temporal_soft_rank,
        "train_temporal_penalty": temporal_penalty,
        "train_temporal_pairs": pair_count,
        "train_date_batches": pair_count + 1,
        "train_forward_passes": pair_count + 1,
        "temporal_anchor_gradient": "stop_gradient",
        "gradient_update_mode": "once_per_epoch",
        "optimizer_steps": 1,
        "train_samples": sample_count,
        "train_batches": pair_count,
        "train_batch_size_min": int(min_batch_size),
        "train_batch_size_max": int(max_batch_size),
        "train_seconds": time.perf_counter() - started,
    }


# 定义 `train_one_epoch` 函数，封装这一阶段可独立调用的处理步骤。
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_prefetch: bool,
    config: TrainingConfig,
    max_batches: int | None = None,
    distributed_context: distributed.DistributedContext | None = None,
) -> dict[str, object]:
    """训练一个完整 epoch，并返回损失与批次诊断。
    
    输入：
        model: nn.Module。
        loader: DataLoader。
        optimizer: torch.optim.Optimizer。
        device: torch.device。
        use_prefetch: bool。
        config: TrainingConfig。
        max_batches: int | None。
    输出：
        dict[str, object]。
        """
    # 检查 `config.loss_name == TEMPORAL_TURNOVER_LOSS`；根据条件结果选择对应处理分支。
    if config.loss_name == TEMPORAL_TURNOVER_LOSS:
        # 将 `_train_temporal_turnover_epoch(model, loader, optimizer, device, use_prefetch, config, max_batc…` 作为当前函数结果返回给调用方。
        return _train_temporal_turnover_epoch(
            model,
            loader,
            optimizer,
            device,
            use_prefetch,
            config,
            max_batches=max_batches,
            distributed_context=distributed_context,
        )
    # 将 `_train_standard_epoch(model, loader, optimizer, device, use_prefetch, config, max_batches=max_b…` 作为当前函数结果返回给调用方。
    return _train_standard_epoch(
        model,
        loader,
        optimizer,
        device,
        use_prefetch,
        config,
        max_batches=max_batches,
        distributed_context=distributed_context,
    )


# 设置 `_INSTRUMENT_IDS` 的固定配置值，供后续流程统一使用。
_INSTRUMENT_IDS: np.ndarray | None = None


# 定义 `instrument_ids` 函数，封装这一阶段可独立调用的处理步骤。
def instrument_ids() -> np.ndarray:
    """根据样本行号返回对应股票代码。
    
    输入：
        无显式输入参数。
    输出：
        np.ndarray。
        """
    # 声明 `_INSTRUMENT_IDS` 使用模块级状态，避免创建同名局部变量。
    global _INSTRUMENT_IDS
    # 检查 `_INSTRUMENT_IDS is None`；根据条件结果选择对应处理分支。
    if _INSTRUMENT_IDS is None:
        # 调用 `pd.read_parquet(SAMPLE_INDEX_PATH, columns=['instrument_id'])['…` 读取源数据，并将结果保存到 `_INSTRUMENT_IDS`。
        _INSTRUMENT_IDS = pd.read_parquet(
            SAMPLE_INDEX_PATH, columns=["instrument_id"]
        )["instrument_id"].to_numpy(copy=False)
    # 将 `_INSTRUMENT_IDS` 作为当前函数结果返回给调用方。
    return _INSTRUMENT_IDS


# 使用 `torch.no_grad()` 包装 `evaluate_model`，并定义其可复用的处理逻辑。
@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_prefetch: bool,
    validation_metric: str,
    max_batches: int | None = None,
    distributed_context: distributed.DistributedContext | None = None,
) -> dict[str, object]:
    """在验证集上评估模型，并汇总逐日 IC 与损失诊断。
    
    输入：
        model: nn.Module。
        loader: DataLoader。
        device: torch.device。
        use_prefetch: bool。
        validation_metric: str。
        max_batches: int | None。
    输出：
        dict[str, object]。
        """
    # 调用 `model.eval` 将模型切换到评估模式，关闭训练期随机行为。
    local_model = inference_model(model)
    # 执行当前分布式训练步骤并保持既定数据语义。
    local_model.eval()
    # 构造当前步骤所需的列表，并保存到 `prediction_parts`。
    prediction_parts: list[np.ndarray] = []
    # 构造当前步骤所需的列表，并保存到 `target_parts`。
    target_parts: list[np.ndarray] = []
    # 构造当前步骤所需的列表，并保存到 `date_parts`。
    date_parts: list[np.ndarray] = []
    # 构造当前步骤所需的列表，并保存到 `sample_index_parts`。
    sample_index_parts: list[np.ndarray] = []
    # 调用 `time.perf_counter` 读取高精度单调时钟以计算耗时，并将结果保存到 `started`。
    started = time.perf_counter()
    # 遍历 `enumerate(CudaBatchPrefetcher(loader, device, use_prefetch), start=1)`，并把当前元素绑定到 `(batch_number, (batch, videos, frame_mask, targets))` 逐项处理。
    for batch_number, (batch, videos, frame_mask, targets) in enumerate(
        CudaBatchPrefetcher(loader, device, use_prefetch), start=1
    ):
        # 在 `autocast_context(device)` 管理的上下文中执行本段逻辑，并保证资源按约定释放。
        with autocast_context(device):
            # 调用 `model` 对当前输入 batch 执行模型前向传播，并将结果保存到 `predictions`。
            predictions = local_model(videos, frame_mask)
        # 把本轮结果追加到 `prediction_parts`，累积后续需要汇总的记录。
        prediction_parts.append(predictions.float().cpu().numpy().astype(np.float64))
        # 把本轮结果追加到 `target_parts`，累积后续需要汇总的记录。
        target_parts.append(targets.float().cpu().numpy().astype(np.float64))
        # 把本轮结果追加到 `date_parts`，累积后续需要汇总的记录。
        date_parts.append(batch["date_ids"].numpy().astype(np.int64, copy=False))
        # 把本轮结果追加到 `sample_index_parts`，累积后续需要汇总的记录。
        sample_index_parts.append(
            batch["sample_indices"].numpy().astype(np.int64, copy=False)
        )
        # 检查 `max_batches is not None and batch_number >= max_batches`；根据条件结果选择对应处理分支。
        if max_batches is not None and batch_number >= max_batches:
            # 终止当前循环，避免继续处理已经不满足条件的元素。
            break
    # 检查 `not prediction_parts`；根据条件结果选择对应处理分支。
    if not prediction_parts:
        # 抛出 `ValueError('验证 loader 没有可用 batch')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("验证 loader 没有可用 batch")
    # 检查 `device.type == 'cuda'`；根据条件结果选择对应处理分支。
    if device.type == "cuda":
        # 调用 `torch.cuda.synchronize` 等待当前 GPU 队列执行完毕后再计时或验收。
        torch.cuda.synchronize(device)
    # 执行当前分布式训练步骤并保持既定数据语义。
    predictions_tensor = torch.as_tensor(
        np.concatenate(prediction_parts), device=device, dtype=torch.float64
    )
    # 执行当前分布式训练步骤并保持既定数据语义。
    targets_tensor = torch.as_tensor(
        np.concatenate(target_parts), device=device, dtype=torch.float64
    )
    # 执行当前分布式训练步骤并保持既定数据语义。
    dates_tensor = torch.as_tensor(
        np.concatenate(date_parts), device=device, dtype=torch.int64
    )
    # 执行当前分布式训练步骤并保持既定数据语义。
    indices_tensor = torch.as_tensor(
        np.concatenate(sample_index_parts), device=device, dtype=torch.int64
    )
    # 执行当前分布式训练步骤并保持既定数据语义。
    predictions_tensor = gather_global_inference_tensor(
        predictions_tensor, distributed_context
    )
    # 执行当前分布式训练步骤并保持既定数据语义。
    targets_tensor = gather_global_inference_tensor(targets_tensor, distributed_context)
    # 执行当前分布式训练步骤并保持既定数据语义。
    dates_tensor = gather_global_inference_tensor(dates_tensor, distributed_context)
    # 执行当前分布式训练步骤并保持既定数据语义。
    indices_tensor = gather_global_inference_tensor(indices_tensor, distributed_context)
    # 执行当前分布式训练步骤并保持既定数据语义。
    order = torch.argsort(indices_tensor)
    # 执行当前分布式训练步骤并保持既定数据语义。
    selected_indices = indices_tensor[order].cpu().numpy()
    # 执行当前分布式训练步骤并保持既定数据语义。
    result = summarize_validation_arrays(
        predictions_tensor[order].cpu().numpy(),
        targets_tensor[order].cpu().numpy(),
        dates_tensor[order].cpu().numpy(),
        validation_metric,
        instrument_ids=instrument_ids()[selected_indices],
    )
    # 计算 `time.perf_counter() - started`，并将数值结果保存到 `result['validation_seconds']`。
    result["validation_seconds"] = time.perf_counter() - started
    # 将 `result` 作为当前函数结果返回给调用方。
    return result


# 使用 `torch.no_grad()` 包装 `predict_model`，并定义其可复用的处理逻辑。
@torch.no_grad()
def predict_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_prefetch: bool,
    max_batches: int | None = None,
    distributed_context: distributed.DistributedContext | None = None,
) -> pd.DataFrame:
    """在预测集上生成与样本行号严格对齐的模型分数。
    
    输入：
        model: nn.Module。
        loader: DataLoader。
        device: torch.device。
        use_prefetch: bool。
        max_batches: int | None。
    输出：
        pd.DataFrame。
        """
    # 调用 `model.eval` 将模型切换到评估模式，关闭训练期随机行为。
    local_model = inference_model(model)
    # 执行当前分布式训练步骤并保持既定数据语义。
    local_model.eval()
    # 构造当前步骤所需的列表，并保存到 `sample_parts`。
    sample_parts: list[np.ndarray] = []
    # 构造当前步骤所需的列表，并保存到 `prediction_parts`。
    prediction_parts: list[np.ndarray] = []
    # 遍历 `enumerate(CudaBatchPrefetcher(loader, device, use_prefetch), start=1)`，并把当前元素绑定到 `(batch_number, (batch, videos, frame_mask, _))` 逐项处理。
    for batch_number, (batch, videos, frame_mask, _) in enumerate(
        CudaBatchPrefetcher(loader, device, use_prefetch), start=1
    ):
        # 在 `autocast_context(device)` 管理的上下文中执行本段逻辑，并保证资源按约定释放。
        with autocast_context(device):
            # 调用 `model` 对当前输入 batch 执行模型前向传播，并将结果保存到 `predictions`。
            predictions = local_model(videos, frame_mask)
        # 把本轮结果追加到 `sample_parts`，累积后续需要汇总的记录。
        sample_parts.append(batch["sample_indices"].numpy())
        # 把本轮结果追加到 `prediction_parts`，累积后续需要汇总的记录。
        prediction_parts.append(predictions.float().cpu().numpy())
        # 检查 `max_batches is not None and batch_number >= max_batches`；根据条件结果选择对应处理分支。
        if max_batches is not None and batch_number >= max_batches:
            # 终止当前循环，避免继续处理已经不满足条件的元素。
            break
    # 执行当前分布式训练步骤并保持既定数据语义。
    if sample_parts:
        # 执行当前分布式训练步骤并保持既定数据语义。
        local_indices = torch.as_tensor(
            np.concatenate(sample_parts), device=device, dtype=torch.int64
        )
        # 执行当前分布式训练步骤并保持既定数据语义。
        local_predictions = torch.as_tensor(
            np.concatenate(prediction_parts), device=device, dtype=torch.float32
        )
    # 执行当前分布式训练步骤并保持既定数据语义。
    else:
        # 执行当前分布式训练步骤并保持既定数据语义。
        local_indices = torch.empty(0, device=device, dtype=torch.int64)
        # 执行当前分布式训练步骤并保持既定数据语义。
        local_predictions = torch.empty(0, device=device, dtype=torch.float32)
    # 执行当前分布式训练步骤并保持既定数据语义。
    global_indices = gather_global_inference_tensor(local_indices, distributed_context)
    # 执行当前分布式训练步骤并保持既定数据语义。
    global_predictions = gather_global_inference_tensor(
        local_predictions, distributed_context
    )
    # 执行当前分布式训练步骤并保持既定数据语义。
    order = torch.argsort(global_indices)
    # 执行当前分布式训练步骤并保持既定数据语义。
    return pd.DataFrame(
        {
            "sample_index": global_indices[order].cpu().numpy(),
            "prediction": global_predictions[order].cpu().numpy().astype(np.float32),
        }
    )


# 定义 `run_paths` 函数，封装这一阶段可独立调用的处理步骤。
def run_paths(run_name: str) -> tuple[Path, Path, Path]:
    """返回指定年度和随机种子的模型、预测与日志路径。
    
    输入：
        run_name: str。
    输出：
        tuple[Path, Path, Path]。
        """
    # 计算 `VIVIT_DIR / 'models' / 'runs' / run_name`，并将数值结果保存到 `model_root`。
    model_root = VIVIT_DIR / "models" / "runs" / run_name
    # 计算 `DATA_DIR / 'predictions' / 'runs' / run_name`，并将数值结果保存到 `prediction_root`。
    prediction_root = DATA_DIR / "predictions" / "runs" / run_name
    # 计算 `DATA_DIR / 'quality' / 'runs' / run_name`，并将数值结果保存到 `quality_root`。
    quality_root = DATA_DIR / "quality" / "runs" / run_name
    # 遍历 `(model_root, prediction_root, quality_root)`，并把当前元素绑定到 `path` 逐项处理。
    for path in (model_root, prediction_root, quality_root):
        # 调用 `path.mkdir` 创建输出目录，确保后续文件可以安全写入。
        path.mkdir(parents=True, exist_ok=True)
    # 将 `(model_root, prediction_root, quality_root)` 作为当前函数结果返回给调用方。
    return model_root, prediction_root, quality_root


# 定义 `build_optimizer` 函数，封装这一阶段可独立调用的处理步骤。
def build_optimizer(
    model: nn.Module,
    config: TrainingConfig,
    fused: bool,
    device: torch.device,
) -> torch.optim.Adam:
    """创建优化器，并按配置启用 fused Adam。
    
    输入：
        model: nn.Module。
        config: TrainingConfig。
        fused: bool。
        device: torch.device。
    输出：
        torch.optim.Adam。
        """
    # 将 `torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=(config.adam_beta1, config.…` 作为当前函数结果返回给调用方。
    return torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
        weight_decay=config.weight_decay,
        fused=fused and device.type == "cuda",
    )


# 定义 `training_config_from_args` 函数，封装这一阶段可独立调用的处理步骤。
def training_config_from_args(args: argparse.Namespace) -> TrainingConfig:
    """把命令行参数转换为不可变训练配置。
    
    输入：
        args: argparse.Namespace。
    输出：
        TrainingConfig。
        """
    # 将 `TrainingConfig(train_batch_mode=args.train_batch_mode, batch_size=args.batch_size, inference_ba…` 作为当前函数结果返回给调用方。
    return TrainingConfig(
        train_batch_mode=args.train_batch_mode,
        batch_size=args.batch_size,
        inference_batch_size=args.inference_batch_size,
        loss_name=args.loss_name,
        rank_temperature=args.rank_temperature,
        turnover_lambda=args.turnover_lambda,
        huber_delta=args.huber_delta,
        smooth_l1_beta=args.smooth_l1_beta,
        hybrid_weight=args.hybrid_weight,
        validation_metric=args.validation_metric,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        early_stopping_patience=args.patience,
    )




# 定义 `training_payload` 函数，封装这一阶段可独立调用的处理步骤。
def training_payload(
    args: argparse.Namespace,
    year: int,
    seed: int,
    batch_summary: dict[str, object],
) -> dict[str, object]:
    """整理训练配置，供 checkpoint 判断是否可以继续使用。
    
    输入：
        args: argparse.Namespace。
        year: int。
        seed: int。
        batch_summary: dict[str, object]。
    输出：
        dict[str, object]。
        """
    # 调用 `require_aligned_sequence_store` 校验输入、配置或产物是否满足当前阶段的数据契约，并将结果保存到 `sequence_manifest`。
    sequence_manifest = require_aligned_sequence_store()
    # 调用 `training_config_from_args` 把命令行参数转换为不可变训练配置，并将结果保存到 `config`。
    config = training_config_from_args(args)
    # 调用 `validate_training_config` 校验输入、配置或产物是否满足当前阶段的数据契约。
    validate_training_config(config, args.compile_mode)
    # 根据 `config.loss_name == TEMPORAL_TURNOVER_LOSS` 计算条件结果，并保存到 `temporal_mode`。
    temporal_mode = config.loss_name == TEMPORAL_TURNOVER_LOSS
    # 根据 `config.train_batch_mode == 'trade_date'` 计算条件结果，并保存到 `date_batches`。
    date_batches = config.train_batch_mode == "trade_date"
    # 将 `{'schema_version': 2, 'forecast_year': year, 'seed': seed, 'sequence_signature': sequence_manif…` 作为当前函数结果返回给调用方。
    return {
        "schema_version": 2,
        "forecast_year": year,
        "seed": seed,
        "sequence_signature": sequence_manifest["signature"],
        "split_source": "ViT/data/feature_store/rolling_splits",
        "split_policy": "exact_vit_train_validation_prediction_indices",
        "padding_policy": "zero_fill_missing_frames_with_tubelet_coverage_mask",
        "model_config": asdict(MODEL_CONFIG),
        "training_config": asdict(config),
        "training_batch_summary": batch_summary,
        "execution_config": {
            "route": (
                "DDP_single_node_task_worker"
                if getattr(args, "execution_route", "a2") == "ddp"
                else "A2_independent_task_worker"
            ),
            **(
                {"world_size": int(args.ddp_world_size)}
                if getattr(args, "execution_route", "a2") == "ddp" else {}
            ),
            "train_batch_mode": config.train_batch_mode,
            "batch_size_per_task": "full_trade_date" if date_batches else config.batch_size,
            "fixed_batch_size": config.batch_size,
            "inference_batch_size": config.inference_batch_size,
            "reshuffle_each_epoch": not temporal_mode,
            "gradient_update_mode": "once_per_epoch" if temporal_mode else "per_batch",
            "data_loader_workers": args.workers_per_task,
            "torch_compile": bool(args.compile),
            "compile_mode": args.compile_mode,
            "compile_dynamic_batch": bool(args.compile) and date_batches,
            "fused_adam": bool(args.fused_adam),
            "cuda_prefetch": bool(args.cuda_prefetch),
            "precision": "bf16_autocast_with_fp32_loss",
        },
        "loss": config.loss_name,
        "loss_parameters": {
            "rank_temperature": config.rank_temperature,
            "turnover_lambda": config.turnover_lambda,
            "huber_delta": config.huber_delta,
            "smooth_l1_beta": config.smooth_l1_beta,
            "hybrid_weight": config.hybrid_weight,
        },
        "validation_metric": config.validation_metric,
        "validation_audit_metrics": [
            "daily_mean_spearman_ic",
            "daily_mean_pearson_ic",
            "pooled_pearson_ic",
            "temporal_spearman_rank_ic",
            "temporal_rank_penalty",
        ],
    }


# 定义 `load_canonical_state` 函数，封装这一阶段可独立调用的处理步骤。
def load_canonical_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    """把规范化参数字典加载到原始或包装模型。
    
    输入：
        model: nn.Module。
        state: dict[str, torch.Tensor]。
    输出：
        None。
        """
    # 调用 `unwrap_model(model).load_state_dict` 读取并解析后续计算所需的数据或运行状态。
    unwrap_model(model).load_state_dict(state)


# 定义 `load_best_model` 函数，封装这一阶段可独立调用的处理步骤。
def load_best_model(
    path: Path,
    device: torch.device,
    use_compile: bool,
    compile_mode: str,
    train_batch_mode: str,
) -> nn.Module:
    """加载最佳验证 checkpoint，并恢复可用于预测的模型。
    
    输入：
        path: Path。
        device: torch.device。
        use_compile: bool。
        compile_mode: str。
        train_batch_mode: str。
    输出：
        nn.Module。
        """
    # 调用 `torch.load` 读取源数据，并将结果保存到 `checkpoint`。
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    # 调用 `build_model` 根据模型配置创建网络并部署到目标设备，并将结果保存到 `model`。
    model = build_model(device, use_compile, compile_mode, train_batch_mode)
    # 调用 `load_canonical_state` 把标准化 checkpoint 权重加载到当前模型。
    load_canonical_state(model, checkpoint["model_state"])
    # 调用 `model.eval` 将模型切换到评估模式，关闭训练期随机行为。
    model.eval()
    # 将 `model` 作为当前函数结果返回给调用方。
    return model


# 定义 `train_single_run` 函数，封装这一阶段可独立调用的处理步骤。
def train_single_run(
    args: argparse.Namespace,
    distributed_context: distributed.DistributedContext | None = None,
) -> dict[str, object]:
    """训练或恢复一个年度、一个随机种子。
    
    输入：
        args: argparse.Namespace。
    输出：
        dict[str, object]。
        """

    # 组织一组有序值，并保存到 `(year, seed)`。
    year, seed = args.worker_year, args.worker_seed
    # 检查 `year is None or seed is None`；根据条件结果选择对应处理分支。
    if year is None or seed is None:
        # 抛出 `ValueError('worker 需要 --worker-year 和 --worker-seed')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("worker 需要 --worker-year 和 --worker-seed")
    # 调用 `require_aligned_sequence_store` 校验输入、配置或产物是否满足当前阶段的数据契约。
    require_aligned_sequence_store()
    # 调用 `training_config_from_args` 把命令行参数转换为不可变训练配置，并将结果保存到 `config`。
    config = training_config_from_args(args)
    # 调用 `validate_training_config` 校验输入、配置或产物是否满足当前阶段的数据契约。
    validate_training_config(config, args.compile_mode)
    # 执行当前分布式训练步骤并保持既定数据语义。
    primary = distributed_context is None or distributed_context.is_primary
    # 执行当前分布式训练步骤并保持既定数据语义。
    device = (
        distributed_context.device
        if distributed_context is not None
        else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    # 执行当前分布式训练步骤并保持既定数据语义。
    pin_memory = device.type == "cuda"
    # 执行当前分布式训练步骤并保持既定数据语义。
    set_random_seed(seed + (distributed_context.rank if distributed_context else 0))
    # 执行当前分布式训练步骤并保持既定数据语义。
    if distributed_context is None:
        # 执行当前分布式训练步骤并保持既定数据语义。
        train_loader, validation_loader, prediction_loader = make_data_loaders(
            year,
            seed,
            config.batch_size,
            config.inference_batch_size,
            config.train_batch_mode,
            config.loss_name,
            args.workers_per_task,
            pin_memory,
        )
        # 执行当前分布式训练步骤并保持既定数据语义。
        batch_summary = loader_batch_summary(train_loader)
    # 执行当前分布式训练步骤并保持既定数据语义。
    else:
        # 执行当前分布式训练步骤并保持既定数据语义。
        train_loader, validation_loader, prediction_loader, batch_summary = (
            make_distributed_data_loaders(
                year,
                seed,
                config,
                args.workers_per_task,
                pin_memory,
                distributed_context,
            )
        )

    # 调用 `run_paths` 解析当前 run 对应的数据与输出路径，并将结果保存到 `(model_root, _, _)`。
    model_root, _, _ = run_paths(args.run_name)
    # 计算 `model_root / f'forecast_{year}' / f'seed_{seed}'`，并将数值结果保存到 `run_directory`。
    run_directory = model_root / f"forecast_{year}" / f"seed_{seed}"
    # 调用 `run_directory.mkdir` 创建输出目录，确保后续文件可以安全写入。
    if primary:
        # 执行当前分布式训练步骤并保持既定数据语义。
        run_directory.mkdir(parents=True, exist_ok=True)
    # 执行当前分布式训练步骤并保持既定数据语义。
    if distributed_context is not None:
        # 执行当前分布式训练步骤并保持既定数据语义。
        dist.barrier()
    # 计算 `run_directory / 'best.pt'`，并将数值结果保存到 `best_path`。
    best_path = run_directory / "best.pt"
    # 计算 `run_directory / 'last.pt'`，并将数值结果保存到 `last_path`。
    last_path = run_directory / "last.pt"
    # 计算 `run_directory / 'history.parquet'`，并将数值结果保存到 `history_path`。
    history_path = run_directory / "history.parquet"
    # 计算 `run_directory / 'prediction.parquet'`，并将数值结果保存到 `prediction_path`。
    prediction_path = run_directory / "prediction.parquet"
    # 计算 `run_directory / 'completed.json'`，并将数值结果保存到 `completed_path`。
    completed_path = run_directory / "completed.json"

    # 调用 `training_payload` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `payload`。
    payload = training_payload(args, year, seed, batch_summary)
    # 调用 `stable_hash` 对关键配置生成稳定签名以支持复现和一致性检查，并将结果保存到 `run_signature`。
    run_signature = stable_hash(payload)
    # 调用 `atomic_write_json` 先写临时文件再原子替换正式产物，避免留下半成品。
    if primary:
        # 执行当前分布式训练步骤并保持既定数据语义。
        atomic_write_json(
        {
            "signature": stable_hash(
                {key: value for key, value in payload.items() if key != "seed"}
            ),
            "payload": {
                key: value for key, value in payload.items() if key != "seed"
            },
        },
        model_root / f"forecast_{year}" / "model_config.json",
        )
    # 执行当前分布式训练步骤并保持既定数据语义。
    if distributed_context is not None:
        # 执行当前分布式训练步骤并保持既定数据语义。
        dist.barrier()
    # 检查 `completed_path.exists() and best_path.exists() and (not args.overwrite)`；根据条件结果选择对应处理分支。
    if completed_path.exists() and best_path.exists() and not args.overwrite:
        # 调用 `json.loads` 读取源数据，并将结果保存到 `completed`。
        completed = json.loads(completed_path.read_text(encoding="utf-8"))
        # 检查 `completed.get('run_signature') == run_signature`；根据条件结果选择对应处理分支。
        if completed.get("run_signature") == run_signature:
            # 调用 `print` 记录当前流程状态或诊断信息。
            print(f"[{year}/seed={seed}] 已完成，直接复用。", flush=True)
            # 检查 `prediction_loader is not None and (not prediction_path.exists())`；根据条件结果选择对应处理分支。
            if prediction_loader is not None and not prediction_path.exists():
                # 调用 `load_best_model` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `model`。
                model = load_best_model(
                    best_path,
                    device,
                    bool(args.compile),
                    args.compile_mode,
                    config.train_batch_mode,
                )
                # 执行当前分布式训练步骤并保持既定数据语义。
                predictions_frame = predict_model(
                    model,
                    prediction_loader,
                    device,
                    bool(args.cuda_prefetch),
                    distributed_context=distributed_context,
                )
                # 执行当前分布式训练步骤并保持既定数据语义。
                if primary:
                    # 执行当前分布式训练步骤并保持既定数据语义。
                    atomic_write_parquet(predictions_frame, prediction_path)
                # 执行当前分布式训练步骤并保持既定数据语义。
                if distributed_context is not None:
                    # 执行当前分布式训练步骤并保持既定数据语义。
                    dist.barrier()
            # 将 `completed['summary']` 作为当前函数结果返回给调用方。
            return completed["summary"]

    # 调用 `build_model` 根据模型配置创建网络并部署到目标设备，并将结果保存到 `model`。
    model = build_model(
        device, bool(args.compile), args.compile_mode, config.train_batch_mode
    )
    # 执行当前分布式训练步骤并保持既定数据语义。
    if distributed_context is not None:
        # 执行当前分布式训练步骤并保持既定数据语义。
        model = distributed.wrap_model(model, distributed_context)
    # 调用 `build_optimizer` 根据训练配置为可学习参数创建优化器，并将结果保存到 `optimizer`。
    optimizer = build_optimizer(model, config, bool(args.fused_adam), device)
    # 设置 `start_epoch` 的固定配置值，供后续流程统一使用。
    start_epoch = 1
    # 设置 `best_epoch` 的固定配置值，供后续流程统一使用。
    best_epoch = 0
    # 计算 `-np.inf`，并将数值结果保存到 `best_validation_ic`。
    best_validation_ic = -np.inf
    # 构造键值映射，并保存到 `best_validation_metrics`，供后续按键查询。
    best_validation_metrics: dict[str, float] = {}
    # 设置 `epochs_without_improvement` 的固定配置值，供后续流程统一使用。
    epochs_without_improvement = 0
    # 构造当前步骤所需的列表，并保存到 `history`。
    history: list[dict[str, object]] = []

    # 检查 `last_path.exists() and (not args.overwrite)`；根据条件结果选择对应处理分支。
    if last_path.exists() and not args.overwrite:
        # 调用 `torch.load` 读取源数据，并将结果保存到 `checkpoint`。
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        # 检查 `checkpoint.get('run_signature') != run_signature`；根据条件结果选择对应处理分支。
        if checkpoint.get("run_signature") != run_signature:
            # 抛出 `ValueError(f'{last_path} 与当前配置不一致；请使用新 run name')`，阻止不满足数据契约的流程继续执行。
            raise ValueError(f"{last_path} 与当前配置不一致；请使用新 run name")
        # 调用 `load_canonical_state` 把标准化 checkpoint 权重加载到当前模型。
        load_canonical_state(model, checkpoint["model_state"])
        # 调用 `optimizer.load_state_dict` 恢复优化器动量等续训状态。
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        # 执行当前分布式训练步骤并保持既定数据语义。
        if distributed_context is not None:
            # 执行当前分布式训练步骤并保持既定数据语义。
            states = checkpoint.get("rng_state_by_rank")
            # 执行当前分布式训练步骤并保持既定数据语义。
            if states is None or len(states) != distributed_context.world_size:
                # 执行当前分布式训练步骤并保持既定数据语义。
                raise ValueError("DDP checkpoint 缺少匹配 world size 的逐 rank RNG 状态")
            # 执行当前分布式训练步骤并保持既定数据语义。
            distributed.restore_rng_state(
                states[distributed_context.rank], device
            )
        # 执行当前分布式训练步骤并保持既定数据语义。
        else:
            # 执行当前分布式训练步骤并保持既定数据语义。
            random.setstate(checkpoint["python_random_state"])
            # 执行当前分布式训练步骤并保持既定数据语义。
            np.random.set_state(checkpoint["numpy_random_state"])
            # 执行当前分布式训练步骤并保持既定数据语义。
            torch.set_rng_state(checkpoint["torch_rng_state"])
            # 执行当前分布式训练步骤并保持既定数据语义。
            if device.type == "cuda" and checkpoint["cuda_rng_state"] is not None:
                # 执行当前分布式训练步骤并保持既定数据语义。
                torch.cuda.set_rng_state(checkpoint["cuda_rng_state"], device)
        # 调用 `train_loader.generator.set_state` 读取并解析后续计算所需的数据或运行状态。
        train_loader.generator.set_state(checkpoint["train_generator_state"])
        # 计算 `int(checkpoint['epoch']) + 1`，并将数值结果保存到 `start_epoch`。
        start_epoch = int(checkpoint["epoch"]) + 1
        # 调用 `int` 统一设备或数据类型，并将结果保存到 `best_epoch`。
        best_epoch = int(checkpoint["best_epoch"])
        # 调用 `float` 统一设备或数据类型，并将结果保存到 `best_validation_ic`。
        best_validation_ic = float(checkpoint["best_validation_ic"])
        # 构造键值映射，并保存到 `best_validation_metrics`，供后续按键查询。
        best_validation_metrics = {
            key: float(value)
            for key, value in checkpoint.get("best_validation_metrics", {}).items()
        }
        # 调用 `int` 统一设备或数据类型，并将结果保存到 `epochs_without_improvement`。
        epochs_without_improvement = int(checkpoint["epochs_without_improvement"])
        # 检查 `epochs_without_improvement >= config.early_stopping_patience`；根据条件结果选择对应处理分支。
        if epochs_without_improvement >= config.early_stopping_patience:
            # 计算 `config.max_epochs + 1`，并将数值结果保存到 `start_epoch`。
            start_epoch = config.max_epochs + 1
        # 检查 `history_path.exists()`；根据条件结果选择对应处理分支。
        if history_path.exists():
            # 调用 `pd.read_parquet(history_path).to_dict` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `history`。
            history = pd.read_parquet(history_path).to_dict("records")
        # 调用 `print` 记录当前流程状态或诊断信息。
        print(f"[{year}/seed={seed}] 从 epoch {start_epoch} 续训。", flush=True)

    # 调用 `time.perf_counter` 读取高精度单调时钟以计算耗时，并将结果保存到 `started`。
    started = time.perf_counter()
    # 根据 `epochs_without_improvement >= config.early_stopping_patience` 计算条件结果，并保存到 `stopped_early`。
    stopped_early = epochs_without_improvement >= config.early_stopping_patience
    # 遍历 `range(start_epoch, config.max_epochs + 1)`，并把当前元素绑定到 `epoch` 逐项处理。
    for epoch in range(start_epoch, config.max_epochs + 1):
        # 调用 `time.perf_counter` 读取高精度单调时钟以计算耗时，并将结果保存到 `epoch_started`。
        epoch_started = time.perf_counter()
        # 调用 `set_train_epoch` 同步 epoch 到采样器并按轮次重建可复现顺序。
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            # 执行当前分布式训练步骤并保持既定数据语义。
            train_loader.batch_sampler.set_epoch(epoch)
        # 执行当前分布式训练步骤并保持既定数据语义。
        else:
            # 执行当前分布式训练步骤并保持既定数据语义。
            set_train_epoch(train_loader, epoch)
        # 调用 `train_one_epoch` 遍历训练集完成一个 epoch 的前向、反向传播和参数更新，并将结果保存到 `train_metrics`。
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            bool(args.cuda_prefetch),
            config,
            distributed_context=distributed_context,
        )
        # 调用 `evaluate_model` 在不更新参数的条件下计算验证或样本外评价指标，并将结果保存到 `validation_metrics`。
        validation_metrics = evaluate_model(
            model,
            validation_loader,
            device,
            bool(args.cuda_prefetch),
            config.validation_metric,
            distributed_context=distributed_context,
        )
        # 调用 `float` 统一设备或数据类型，并将结果保存到 `validation_ic`。
        validation_ic = float(validation_metrics["validation_ic"])
        # 根据 `validation_ic > best_validation_ic + config.improvement_tolerance` 计算条件结果，并保存到 `improved`。
        improved = validation_ic > (
            best_validation_ic + config.improvement_tolerance
        )
        # 检查 `improved`；根据条件结果选择对应处理分支。
        if improved:
            # 读取 `epoch` 的当前值，并赋给 `best_epoch`。
            best_epoch = epoch
            # 读取 `validation_ic` 的当前值，并赋给 `best_validation_ic`。
            best_validation_ic = validation_ic
            # 构造键值映射，并保存到 `best_validation_metrics`，供后续按键查询。
            best_validation_metrics = {
                "validation_daily_rank_ic": float(
                    validation_metrics["validation_daily_rank_ic"]
                ),
                "validation_daily_pearson_ic": float(
                    validation_metrics["validation_daily_pearson_ic"]
                ),
                "validation_pooled_ic": float(
                    validation_metrics["validation_pooled_ic"]
                ),
                "validation_temporal_rank_ic": float(
                    validation_metrics["validation_temporal_rank_ic"]
                ),
                "validation_temporal_penalty": float(
                    validation_metrics["validation_temporal_penalty"]
                ),
            }
            # 设置 `epochs_without_improvement` 的固定配置值，供后续流程统一使用。
            epochs_without_improvement = 0
        # 当前置条件不成立时进入备用分支，保证其余合法情况仍被处理。
        else:
            # 用 `1` 对 `epochs_without_improvement` 做 `Add` 累积更新。
            epochs_without_improvement += 1

        # 把本轮结果追加到 `history`，累积后续需要汇总的记录。
        history.append(
            {
                "forecast_year": year,
                "seed": seed,
                "epoch": epoch,
                "train_batch_mode": config.train_batch_mode,
                "loss_name": config.loss_name,
                "validation_metric": config.validation_metric,
                **train_metrics,
                **validation_metrics,
                "best_validation_ic": best_validation_ic,
                "epochs_without_improvement": epochs_without_improvement,
                "epoch_seconds": time.perf_counter() - epoch_started,
            }
        )
        # 调用 `atomic_write_parquet` 先写临时文件再原子替换正式产物，避免留下半成品。
        if primary:
            # 执行当前分布式训练步骤并保持既定数据语义。
            atomic_write_parquet(pd.DataFrame(history), history_path)
        # 执行当前分布式训练步骤并保持既定数据语义。
        rng_state_by_rank = (
            distributed.gather_rng_states(device) if distributed_context else None
        )
        # 构造键值映射，并保存到 `checkpoint`，供后续按键查询。
        checkpoint = {
            "model_state": unwrap_model(model).state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state(device) if device.type == "cuda" else None
            ),
            "train_generator_state": train_loader.generator.get_state(),
            "epoch": epoch,
            "next_trade_date_sampler_epoch": epoch + 1,
            "best_epoch": best_epoch,
            "best_validation_ic": best_validation_ic,
            "best_validation_metrics": best_validation_metrics,
            "epochs_without_improvement": epochs_without_improvement,
            "run_signature": run_signature,
            "run_payload": payload,
            "model_config": asdict(MODEL_CONFIG),
            "training_config": asdict(config),
        }
        # 执行当前分布式训练步骤并保持既定数据语义。
        if rng_state_by_rank is not None:
            # 执行当前分布式训练步骤并保持既定数据语义。
            checkpoint["rng_state_by_rank"] = rng_state_by_rank
        # 调用 `atomic_save_checkpoint` 按约定格式写出当前计算产物。
        if primary:
            # 执行当前分布式训练步骤并保持既定数据语义。
            atomic_save_checkpoint(checkpoint, last_path)
        # 检查 `improved`；根据条件结果选择对应处理分支。
        if primary and improved:
            # 调用 `atomic_save_checkpoint` 按约定格式写出当前计算产物。
            atomic_save_checkpoint(checkpoint, best_path)
        # 执行当前分布式训练步骤并保持既定数据语义。
        if distributed_context is not None:
            # 执行当前分布式训练步骤并保持既定数据语义。
            dist.barrier()
        # 调用 `print` 记录当前流程状态或诊断信息。
        print(
            f"[{year}/seed={seed}] epoch={epoch:03d} "
            f"trainLoss={train_metrics['train_loss']:.6f} "
            f"trainPearson={train_metrics['train_ic']:.6f} "
            f"validRank={validation_metrics['validation_daily_rank_ic']:.6f} "
            f"validPearson={validation_metrics['validation_daily_pearson_ic']:.6f} "
            f"validPooled={validation_metrics['validation_pooled_ic']:.6f} "
            f"selected={validation_ic:.6f} best={best_validation_ic:.6f} "
            f"patience={epochs_without_improvement}/{config.early_stopping_patience}",
            flush=True,
        )
        # 检查 `epochs_without_improvement >= config.early_stopping_patience`；根据条件结果选择对应处理分支。
        if epochs_without_improvement >= config.early_stopping_patience:
            # 设置 `stopped_early` 的固定配置值，供后续流程统一使用。
            stopped_early = True
            # 终止当前循环，避免继续处理已经不满足条件的元素。
            break

    # 检查 `not best_path.exists()`；根据条件结果选择对应处理分支。
    if not best_path.exists():
        # 抛出 `FileNotFoundError(f'未生成最佳 checkpoint：{best_path}')`，阻止不满足数据契约的流程继续执行。
        raise FileNotFoundError(f"未生成最佳 checkpoint：{best_path}")
    # 检查 `prediction_loader is not None`；根据条件结果选择对应处理分支。
    if prediction_loader is not None:
        # 调用 `load_best_model` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `best_model`。
        best_model = load_best_model(
            best_path,
            device,
            bool(args.compile),
            args.compile_mode,
            config.train_batch_mode,
        )
        # 执行当前分布式训练步骤并保持既定数据语义。
        predictions_frame = predict_model(
            best_model,
            prediction_loader,
            device,
            bool(args.cuda_prefetch),
            distributed_context=distributed_context,
        )
        # 执行当前分布式训练步骤并保持既定数据语义。
        if primary:
            # 执行当前分布式训练步骤并保持既定数据语义。
            atomic_write_parquet(predictions_frame, prediction_path)
        # 执行当前分布式训练步骤并保持既定数据语义。
        if distributed_context is not None:
            # 执行当前分布式训练步骤并保持既定数据语义。
            dist.barrier()
    # 构造键值映射，并保存到 `summary`，供后续按键查询。
    summary = {
        "forecast_year": year,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_ic": best_validation_ic,
        "best_validation_rank_ic": best_validation_metrics.get(
            "validation_daily_rank_ic"
        ),
        "best_validation_daily_pearson_ic": best_validation_metrics.get(
            "validation_daily_pearson_ic"
        ),
        "best_validation_pooled_ic": best_validation_metrics.get(
            "validation_pooled_ic"
        ),
        "last_epoch": int(history[-1]["epoch"]),
        "stopped_early": stopped_early,
        "elapsed_seconds": time.perf_counter() - started,
        "physical_gpu": args.worker_physical_gpu,
        "execution_route": getattr(args, "execution_route", "a2"),
        "world_size": int(getattr(args, "ddp_world_size", 1)),
        "run_name": args.run_name,
        "train_batch_mode": config.train_batch_mode,
        "loss_name": config.loss_name,
        "validation_metric": config.validation_metric,
        "prediction_sample_count": len(load_sequence_split(year)["prediction_indices"]),
        "status": "completed",
    }
    # 调用 `atomic_write_json` 先写临时文件再原子替换正式产物，避免留下半成品。
    if primary:
        # 执行当前分布式训练步骤并保持既定数据语义。
        atomic_write_json(
            {"run_signature": run_signature, "summary": summary}, completed_path
        )
    # 执行当前分布式训练步骤并保持既定数据语义。
    if distributed_context is not None:
        # 执行当前分布式训练步骤并保持既定数据语义。
        dist.barrier()
    # 将 `summary` 作为当前函数结果返回给调用方。
    return summary


# 定义 `build_ensemble` 函数，封装这一阶段可独立调用的处理步骤。
def build_ensemble(run_name: str, year: int) -> int:
    """逐日标准化三个 seed 后等权平均；空 prediction 年份返回 0。
    
    输入：
        run_name: str。
        year: int。
    输出：
        int。
        """

    # 调用 `run_paths` 解析当前 run 对应的数据与输出路径，并将结果保存到 `(model_root, prediction_root, _)`。
    model_root, prediction_root, _ = run_paths(run_name)
    # 调用 `load_sequence_split` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `split`。
    split = load_sequence_split(year)
    # 检查 `not len(split['prediction_indices'])`；根据条件结果选择对应处理分支。
    if not len(split["prediction_indices"]):
        # 将 `0` 作为当前函数结果返回给调用方。
        return 0
    # 构造键值映射，并保存到 `seed_frames`，供后续按键查询。
    seed_frames: dict[int, pd.DataFrame] = {}
    # 遍历 `SEEDS`，并把当前元素绑定到 `seed` 逐项处理。
    for seed in SEEDS:
        # 计算 `model_root / f'forecast_{year}' / f'seed_{seed}' / 'prediction.parquet'`，并将数值结果保存到 `path`。
        path = model_root / f"forecast_{year}" / f"seed_{seed}" / "prediction.parquet"
        # 调用 `pd.read_parquet` 读取源数据，并将结果保存到 `seed_frames[seed]`。
        seed_frames[seed] = pd.read_parquet(path)

    # 调用 `seed_frames[SEEDS[0]]['sample_index'].to_numpy` 把表格列转换为指定精度的 NumPy 数组，并将结果保存到 `reference`。
    reference = seed_frames[SEEDS[0]]["sample_index"].to_numpy(dtype=np.int64)
    # 调用 `pd.DataFrame` 转换数据表示，并将标准化对象保存到 `output`。
    output = pd.DataFrame({"sample_index": reference})
    # 构造当前步骤所需的列表，并保存到 `raw_columns`。
    raw_columns: list[str] = []
    # 遍历 `SEEDS`，并把当前元素绑定到 `seed` 逐项处理。
    for seed in SEEDS:
        # 读取 `seed_frames[seed]` 的当前值，并赋给 `frame`。
        frame = seed_frames[seed]
        # 调用 `frame['sample_index'].to_numpy` 把表格列转换为指定精度的 NumPy 数组，并将结果保存到 `indices`。
        indices = frame["sample_index"].to_numpy(dtype=np.int64)
        # 检查 `not np.array_equal(reference, indices)`；根据条件结果选择对应处理分支。
        if not np.array_equal(reference, indices):
            # 抛出 `AssertionError(f'{year} 年三个 seed 的预测行号不一致')`，阻止不满足数据契约的流程继续执行。
            raise AssertionError(f"{year} 年三个 seed 的预测行号不一致")
        # 计算 `f'prediction_seed_{seed}'`，并将结果写入 `column`。
        column = f"prediction_seed_{seed}"
        # 调用 `frame['prediction'].to_numpy` 执行模型推理并生成样本预测分数，并将结果保存到 `output[column]`。
        output[column] = frame["prediction"].to_numpy(dtype=np.float32)
        # 把本轮结果追加到 `raw_columns`，累积后续需要汇总的记录。
        raw_columns.append(column)

    # 调用 `pd.read_parquet(SAMPLE_INDEX_PATH, columns=['sample_index', 'tr…` 读取源数据，并将结果保存到 `metadata`。
    metadata = pd.read_parquet(
        SAMPLE_INDEX_PATH,
        columns=["sample_index", "trade_date", "instrument_id"],
    ).set_index("sample_index")
    # 调用 `metadata.loc[reference].reset_index().merge` 按键合并数据，并将对齐后的结果保存到 `output`。
    output = metadata.loc[reference].reset_index().merge(
        output, on="sample_index", validate="one_to_one"
    )
    # 调用 `output[raw_columns].astype` 统一设备或数据类型，并将结果保存到 `raw`。
    raw = output[raw_columns].astype(np.float64)
    # 调用 `raw.groupby` 按指定字段建立分组对象，并将结果保存到 `grouped`。
    grouped = raw.groupby(output["trade_date"], sort=False)
    # 调用 `grouped.transform` 按约定截面或分组口径汇总统计结果，并将结果保存到 `means`。
    means = grouped.transform("mean")
    # 调用 `grouped.transform` 按约定截面或分组口径汇总统计结果，并将结果保存到 `standard_deviations`。
    standard_deviations = grouped.transform("std", ddof=0)
    # 检查 `(standard_deviations <= 0).any().any()`；根据条件结果选择对应处理分支。
    if (standard_deviations <= 0).any().any():
        # 抛出 `ValueError(f'{year} 年存在常数 seed 预测')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(f"{year} 年存在常数 seed 预测")
    # 构造当前步骤所需的列表，并保存到 `zscore_columns`。
    zscore_columns: list[str] = []
    # 遍历 `raw_columns`，并把当前元素绑定到 `column` 逐项处理。
    for column in raw_columns:
        # 计算 `f'{column}_zscore'`，并将结果写入 `zscore_column`。
        zscore_column = f"{column}_zscore"
        # 调用 `to_numpy` 把表格列转换为指定精度的 NumPy 数组，并将结果保存到 `output[zscore_column]`。
        output[zscore_column] = (
            (raw[column] - means[column]) / standard_deviations[column]
        ).to_numpy(dtype=np.float32)
        # 把本轮结果追加到 `zscore_columns`，累积后续需要汇总的记录。
        zscore_columns.append(zscore_column)
    # 调用 `raw.mean(axis=1).to_numpy` 把表格列转换为指定精度的 NumPy 数组，并将结果保存到 `output['prediction_ensemble_raw']`。
    output["prediction_ensemble_raw"] = raw.mean(axis=1).to_numpy(dtype=np.float32)
    # 调用 `output[zscore_columns].mean(axis=1).astype` 统一设备或数据类型，并将结果保存到 `output['prediction_ensemble']`。
    output["prediction_ensemble"] = output[zscore_columns].mean(axis=1).astype(
        np.float32
    )
    # 读取 `year` 的当前值，并赋给 `output['forecast_year']`。
    output["forecast_year"] = year
    # 设置 `output['factor_role']` 的固定配置值，供后续流程统一使用。
    output["factor_role"] = "full_year_oos"
    # 调用 `atomic_write_parquet` 先写临时文件再原子替换正式产物，避免留下半成品。
    atomic_write_parquet(
        output,
        prediction_root / f"vivit_predictions.{year}.parquet",
    )
    # 将 `len(output)` 作为当前函数结果返回给调用方。
    return len(output)


# 定义 `finalize_run` 函数，封装这一阶段可独立调用的处理步骤。
def finalize_run(args: argparse.Namespace, wall_seconds: float) -> pd.DataFrame:
    """核验年度任务并生成集成预测、诊断、清单和报告。
    
    输入：
        args: argparse.Namespace。
        wall_seconds: float。
    输出：
        pd.DataFrame。
        """
    # 调用 `run_paths` 解析当前 run 对应的数据与输出路径，并将结果保存到 `(model_root, _, quality_root)`。
    model_root, _, quality_root = run_paths(args.run_name)
    # 构造当前步骤所需的列表，并保存到 `records`。
    records: list[dict[str, object]] = []
    # 遍历 `FORECAST_YEARS`，并把当前元素绑定到 `year` 逐项处理。
    for year in FORECAST_YEARS:
        # 遍历 `SEEDS`，并把当前元素绑定到 `seed` 逐项处理。
        for seed in SEEDS:
            # 计算 `model_root / f'forecast_{year}' / f'seed_{seed}' / 'completed.json'`，并将数值结果保存到 `path`。
            path = model_root / f"forecast_{year}" / f"seed_{seed}" / "completed.json"
            # 检查 `not path.exists()`；根据条件结果选择对应处理分支。
            if not path.exists():
                # 抛出 `FileNotFoundError(f'缺少已完成任务：{path}')`，阻止不满足数据契约的流程继续执行。
                raise FileNotFoundError(f"缺少已完成任务：{path}")
            # 把本轮结果追加到 `records`，累积后续需要汇总的记录。
            records.append(json.loads(path.read_text(encoding="utf-8"))["summary"])
    # 调用 `pd.DataFrame(records).sort_values` 根据分数计算截面排序或排序索引，并将结果保存到 `summary`。
    summary = pd.DataFrame(records).sort_values(
        ["forecast_year", "seed"], kind="stable"
    )
    # 调用 `atomic_write_parquet` 先写临时文件再原子替换正式产物，避免留下半成品。
    atomic_write_parquet(summary, quality_root / "vivit_training_summary.parquet")
    # 构造键值映射，并保存到 `factor_counts`，供后续按键查询。
    factor_counts = {year: build_ensemble(args.run_name, year) for year in FORECAST_YEARS}
    # 调用 `require_aligned_sequence_store` 校验输入、配置或产物是否满足当前阶段的数据契约，并将结果保存到 `sequence_manifest`。
    sequence_manifest = require_aligned_sequence_store()
    # 构造键值映射，并保存到 `manifest`，供后续按键查询。
    manifest = {
        "route": (
            "DDP_single_node_tasks"
            if getattr(args, "execution_route", "a2") == "ddp"
            else "A2_independent_tasks"
        ),
        "execution_route": getattr(args, "execution_route", "a2"),
        "world_size": int(getattr(args, "ddp_world_size", 1)),
        "run_name": args.run_name,
        "sequence_signature": sequence_manifest["signature"],
        "source_dataset_signature": sequence_manifest["payload"][
            "source_dataset_signature"
        ],
        "source_rolling_signature": sequence_manifest["payload"][
            "source_rolling_signature"
        ],
        "device_ids": [int(value) for value in args.device_ids.split(",")],
        "tasks_per_gpu": getattr(args, "tasks_per_gpu", None),
        "train_batch_mode": args.train_batch_mode,
        "batch_size_per_task": (
            "full_trade_date" if args.train_batch_mode == "trade_date" else args.batch_size
        ),
        "fixed_batch_size": args.batch_size,
        "inference_batch_size": args.inference_batch_size,
        "workers_per_task": args.workers_per_task,
        "loss_name": args.loss_name,
        "loss_parameters": {
            "rank_temperature": args.rank_temperature,
            "turnover_lambda": args.turnover_lambda,
            "huber_delta": args.huber_delta,
            "smooth_l1_beta": args.smooth_l1_beta,
            "hybrid_weight": args.hybrid_weight,
        },
        "validation_metric": args.validation_metric,
        "validation_audit_metrics": [
            "daily_mean_spearman_ic",
            "daily_mean_pearson_ic",
            "pooled_pearson_ic",
            "temporal_spearman_rank_ic",
            "temporal_rank_penalty",
        ],
        "split_source": "ViT/data/feature_store/rolling_splits",
        "split_policy": "exact_vit_train_validation_prediction_indices",
        "padding_policy": "zero_fill_with_tubelet_coverage_mask",
        "learning_rate": args.learning_rate,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "torch_compile": bool(args.compile),
        "compile_mode": args.compile_mode,
        "fused_adam": bool(args.fused_adam),
        "cuda_prefetch": bool(args.cuda_prefetch),
        "years": list(FORECAST_YEARS),
        "seeds": list(SEEDS),
        "completed_tasks": len(summary),
        "factor_counts": factor_counts,
        "wall_seconds": wall_seconds,
    }
    # 调用 `atomic_write_json` 先写临时文件再原子替换正式产物，避免留下半成品。
    atomic_write_json(manifest, quality_root / "a2_training_manifest.json")
    # 将 `summary` 作为当前函数结果返回给调用方。
    return summary


# 定义 `worker_command` 函数，封装这一阶段可独立调用的处理步骤。
def worker_command(args: argparse.Namespace, year: int, seed: int, gpu: int) -> list[str]:
    """构造单个年度×seed worker 的命令行参数。
    
    输入：
        args: argparse.Namespace。
        year: int。
        seed: int。
        gpu: int。
    输出：
        list[str]。
        """
    # 将 `[sys.executable, str(Path(__file__).resolve()), 'worker', '--run-name', args.run_name, '--train…` 作为当前函数结果返回给调用方。
    return [
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


# 定义 `tail` 函数，封装这一阶段可独立调用的处理步骤。
def ddp_worker_command(
    args: argparse.Namespace,
    year: int,
    seed: int,
    device_ids: tuple[int, ...],
) -> list[str]:
    """执行 ddp_worker_command 对应的分布式训练处理。
    
    输入：
        参数见函数类型标注。
    输出：
        结果见函数返回类型标注。
    """
    # 执行当前分布式训练步骤并保持既定数据语义。
    command = worker_command(args, year, seed, device_ids[0])
    # 执行当前分布式训练步骤并保持既定数据语义。
    command[2] = "ddp-worker"
    # 执行当前分布式训练步骤并保持既定数据语义。
    command.extend(["--device-ids", ",".join(str(value) for value in device_ids)])
    # 执行当前分布式训练步骤并保持既定数据语义。
    return command


# 执行当前分布式训练步骤并保持既定数据语义。
def _baseline_ddp_smoke(
    args: argparse.Namespace,
    context: distributed.DistributedContext,
) -> dict[str, object]:
    """执行 _baseline_ddp_smoke 对应的分布式训练处理。
    
    输入：
        参数见函数类型标注。
    输出：
        结果见函数返回类型标注。
    """
    # 执行当前分布式训练步骤并保持既定数据语义。
    config = training_config_from_args(args)
    # 执行当前分布式训练步骤并保持既定数据语义。
    validate_training_config(config, args.compile_mode)
    # 执行当前分布式训练步骤并保持既定数据语义。
    set_random_seed(args.seed + context.rank)
    # 执行当前分布式训练步骤并保持既定数据语义。
    train_loader, validation_loader, _, batch_summary = make_distributed_data_loaders(
        args.forecast_year,
        args.seed,
        config,
        args.workers_per_task,
        context.device.type == "cuda",
        context,
    )
    # 执行当前分布式训练步骤并保持既定数据语义。
    train_loader.batch_sampler.set_epoch(1)
    # 执行当前分布式训练步骤并保持既定数据语义。
    model = build_model(
        context.device,
        bool(args.compile),
        args.compile_mode,
        config.train_batch_mode,
    )
    # 执行当前分布式训练步骤并保持既定数据语义。
    model = distributed.wrap_model(model, context)
    # 执行当前分布式训练步骤并保持既定数据语义。
    optimizer = build_optimizer(
        model, config, bool(args.fused_adam), context.device
    )
    # 执行当前分布式训练步骤并保持既定数据语义。
    train_metrics = train_one_epoch(
        model,
        train_loader,
        optimizer,
        context.device,
        bool(args.cuda_prefetch),
        config,
        max_batches=args.max_batches,
        distributed_context=context,
    )
    # 执行当前分布式训练步骤并保持既定数据语义。
    validation_metrics = evaluate_model(
        model,
        validation_loader,
        context.device,
        bool(args.cuda_prefetch),
        config.validation_metric,
        max_batches=args.max_batches,
        distributed_context=context,
    )
    # 执行当前分布式训练步骤并保持既定数据语义。
    return {
        "forecast_year": args.forecast_year,
        "seed": args.seed,
        "execution_route": "ddp",
        "world_size": context.world_size,
        "batch_summary": batch_summary,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "persistent_outputs_written": False,
    }


# 执行当前分布式训练步骤并保持既定数据语义。
def _baseline_ddp_rank(
    rank: int,
    world_size: int,
    args: argparse.Namespace,
    smoke: bool,
) -> None:
    """执行 _baseline_ddp_rank 对应的分布式训练处理。
    
    输入：
        参数见函数类型标注。
    输出：
        结果见函数返回类型标注。
    """
    # 执行当前分布式训练步骤并保持既定数据语义。
    context = distributed.initialize_process_group(rank, world_size, backend="nccl")
    # 执行当前分布式训练步骤并保持既定数据语义。
    try:
        # 执行当前分布式训练步骤并保持既定数据语义。
        args.execution_route = "ddp"
        # 执行当前分布式训练步骤并保持既定数据语义。
        args.ddp_world_size = world_size
        # 执行当前分布式训练步骤并保持既定数据语义。
        if smoke:
            # 执行当前分布式训练步骤并保持既定数据语义。
            result = _baseline_ddp_smoke(args, context)
        # 执行当前分布式训练步骤并保持既定数据语义。
        else:
            # 执行当前分布式训练步骤并保持既定数据语义。
            result = train_single_run(args, distributed_context=context)
        # 执行当前分布式训练步骤并保持既定数据语义。
        if context.is_primary:
            # 执行当前分布式训练步骤并保持既定数据语义。
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    # 执行当前分布式训练步骤并保持既定数据语义。
    finally:
        # 执行当前分布式训练步骤并保持既定数据语义。
        distributed.destroy_process_group()


# 执行当前分布式训练步骤并保持既定数据语义。
def run_ddp_worker(args: argparse.Namespace, *, smoke: bool = False) -> None:
    """执行 run_ddp_worker 对应的分布式训练处理。
    
    输入：
        参数见函数类型标注。
    输出：
        结果见函数返回类型标注。
    """
    # 执行当前分布式训练步骤并保持既定数据语义。
    devices = distributed.parse_device_ids(args.device_ids)
    # 执行当前分布式训练步骤并保持既定数据语义。
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(value) for value in devices)
    # 执行当前分布式训练步骤并保持既定数据语义。
    if not torch.cuda.is_available() or torch.cuda.device_count() != len(devices):
        # 执行当前分布式训练步骤并保持既定数据语义。
        raise RuntimeError(
            f"DDP 请求 {len(devices)} 张 GPU，但当前可见 {torch.cuda.device_count()} 张"
        )
    # 执行当前分布式训练步骤并保持既定数据语义。
    args.ddp_world_size = len(devices)
    # 执行当前分布式训练步骤并保持既定数据语义。
    args.execution_route = "ddp"
    # 执行当前分布式训练步骤并保持既定数据语义。
    if smoke:
        # 执行当前分布式训练步骤并保持既定数据语义。
        args.worker_year = args.forecast_year
        # 执行当前分布式训练步骤并保持既定数据语义。
        args.worker_seed = args.seed
    # 执行当前分布式训练步骤并保持既定数据语义。
    args.worker_physical_gpu = devices[0]
    # 执行当前分布式训练步骤并保持既定数据语义。
    distributed.configure_master(distributed.reserve_local_port())
    # 执行当前分布式训练步骤并保持既定数据语义。
    mp.spawn(
        _baseline_ddp_rank,
        args=(len(devices), args, smoke),
        nprocs=len(devices),
        join=True,
    )


# 执行当前分布式训练步骤并保持既定数据语义。
def run_ddp(args: argparse.Namespace) -> pd.DataFrame:
    """执行 run_ddp 对应的分布式训练处理。
    
    输入：
        参数见函数类型标注。
    输出：
        结果见函数返回类型标注。
    """

    # 执行当前分布式训练步骤并保持既定数据语义。
    devices = distributed.parse_device_ids(args.device_ids)
    # 执行当前分布式训练步骤并保持既定数据语义。
    args.execution_route = "ddp"
    # 执行当前分布式训练步骤并保持既定数据语义。
    args.ddp_world_size = len(devices)
    # 执行当前分布式训练步骤并保持既定数据语义。
    _, _, quality_root = run_paths(args.run_name)
    # 执行当前分布式训练步骤并保持既定数据语义。
    log_dir = quality_root / "ddp_task_logs"
    # 执行当前分布式训练步骤并保持既定数据语义。
    log_dir.mkdir(parents=True, exist_ok=True)
    # 执行当前分布式训练步骤并保持既定数据语义。
    started = time.perf_counter()
    # 执行当前分布式训练步骤并保持既定数据语义。
    print(
        f"ViViT DDP start: tasks={len(FORECAST_YEARS) * len(SEEDS)}, "
        f"GPUs={devices}, world_size={len(devices)}",
        flush=True,
    )
    # 执行当前分布式训练步骤并保持既定数据语义。
    for year in FORECAST_YEARS:
        # 执行当前分布式训练步骤并保持既定数据语义。
        for seed in SEEDS:
            # 执行当前分布式训练步骤并保持既定数据语义。
            log_path = log_dir / f"forecast_{year}.seed_{seed}.log"
            # 执行当前分布式训练步骤并保持既定数据语义。
            with log_path.open("a", encoding="utf-8") as handle:
                # 执行当前分布式训练步骤并保持既定数据语义。
                handle.write(
                    f"\n=== launch {year}/seed={seed} at "
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
                )
                # 执行当前分布式训练步骤并保持既定数据语义。
                handle.flush()
                # 执行当前分布式训练步骤并保持既定数据语义。
                environment = os.environ.copy()
                # 执行当前分布式训练步骤并保持既定数据语义。
                environment["CUDA_VISIBLE_DEVICES"] = ",".join(
                    str(value) for value in devices
                )
                # 执行当前分布式训练步骤并保持既定数据语义。
                process = subprocess.Popen(
                    ddp_worker_command(args, year, seed, devices),
                    cwd=VIVIT_DIR,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                # 执行当前分布式训练步骤并保持既定数据语义。
                code = process.wait()
            # 执行当前分布式训练步骤并保持既定数据语义。
            if code:
                # 执行当前分布式训练步骤并保持既定数据语义。
                raise RuntimeError(
                    f"DDP 任务失败：{year}/seed={seed}\n{tail(log_path, 100)}"
                )
            # 执行当前分布式训练步骤并保持既定数据语义。
            print(f"done DDP: {year}/seed={seed}", flush=True)
    # 执行当前分布式训练步骤并保持既定数据语义。
    result = finalize_run(args, time.perf_counter() - started)
    # 执行当前分布式训练步骤并保持既定数据语义。
    print(validate_outputs(args.run_name).to_string(index=False), flush=True)
    # 执行当前分布式训练步骤并保持既定数据语义。
    print(f"实验报告：{write_experiment_report(args.run_name)}", flush=True)
    # 执行当前分布式训练步骤并保持既定数据语义。
    return result


# 执行当前分布式训练步骤并保持既定数据语义。
def tail(path: Path, line_count: int = 80) -> str:
    """读取文本文件末尾若干行，用于展示最新训练日志。
    
    输入：
        path: Path。
        line_count: int。
    输出：
        str。
        """
    # 检查 `not path.exists()`；根据条件结果选择对应处理分支。
    if not path.exists():
        # 将 `''` 作为当前函数结果返回给调用方。
        return ""
    # 将 `'\n'.join(path.read_text(encoding='utf-8', errors='replace').splitlines()[-line_count:])` 作为当前函数结果返回给调用方。
    return "\n".join(
        path.read_text(encoding="utf-8", errors="replace").splitlines()[-line_count:]
    )


# 定义 `run_a2` 函数，封装这一阶段可独立调用的处理步骤。
def run_a2(args: argparse.Namespace) -> pd.DataFrame:
    """在四张 GPU 上并发调度 24 个独立任务。
    
    输入：
        args: argparse.Namespace。
    输出：
        pd.DataFrame。
        """

    # 执行当前分布式训练步骤并保持既定数据语义。
    args.execution_route = "a2"
    # 执行当前分布式训练步骤并保持既定数据语义。
    args.ddp_world_size = 1
    # 检查 `not SEQUENCE_MANIFEST_PATH.exists()`；根据条件结果选择对应处理分支。
    if not SEQUENCE_MANIFEST_PATH.exists():
        # 调用 `build_sequence_store` 根据当前配置和输入构造后续流程所需对象。
        build_sequence_store(overwrite=False)
    # 调用 `require_aligned_sequence_store` 校验输入、配置或产物是否满足当前阶段的数据契约。
    require_aligned_sequence_store()
    # 调用 `validate_training_config` 校验输入、配置或产物是否满足当前阶段的数据契约。
    validate_training_config(training_config_from_args(args), args.compile_mode)
    # 调用 `tuple` 整理数据结构，并将结果保存到 `device_ids`。
    device_ids = tuple(int(value) for value in args.device_ids.split(","))
    # 检查 `len(set(device_ids)) != len(device_ids)`；根据条件结果选择对应处理分支。
    if len(set(device_ids)) != len(device_ids):
        # 抛出 `ValueError('GPU 编号不能重复')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("GPU 编号不能重复")
    # 调用 `deque` 创建支持队首弹出的任务队列，并将结果保存到 `tasks`。
    tasks = deque((year, seed) for year in FORECAST_YEARS for seed in SEEDS)
    # 调用 `run_paths` 解析当前 run 对应的数据与输出路径，并将结果保存到 `(_, _, quality_root)`。
    _, _, quality_root = run_paths(args.run_name)
    # 计算 `quality_root / 'a2_task_logs'`，并将数值结果保存到 `log_dir`。
    log_dir = quality_root / "a2_task_logs"
    # 调用 `log_dir.mkdir` 创建输出目录，确保后续文件可以安全写入。
    log_dir.mkdir(parents=True, exist_ok=True)
    # 构造当前步骤所需的列表，并保存到 `running`。
    running: list[dict[str, object]] = []
    # 调用 `time.perf_counter` 读取高精度单调时钟以计算耗时，并将结果保存到 `started`。
    started = time.perf_counter()
    # 调用 `print` 记录当前流程状态或诊断信息。
    print(
        f"ViViT A2 start: tasks={len(tasks)}, GPUs={device_ids}, "
        f"tasks_per_gpu={args.tasks_per_gpu}",
        flush=True,
    )
    # 尝试执行本段主逻辑；若发生异常，则交由后续异常分支处理。
    try:
        # 只要 `tasks or running` 成立就继续循环，直至满足退出条件。
        while tasks or running:
            # 遍历 `device_ids`，并把当前元素绑定到 `gpu` 逐项处理。
            for gpu in device_ids:
                # 调用 `sum` 汇总当前数据，并将统计量保存到 `active`。
                active = sum(1 for item in running if item["gpu"] == gpu)
                # 只要 `tasks and active < args.tasks_per_gpu` 成立就继续循环，直至满足退出条件。
                while tasks and active < args.tasks_per_gpu:
                    # 调用 `tasks.popleft` 从任务队列头部取出下一项任务，并将结果保存到 `(year, seed)`。
                    year, seed = tasks.popleft()
                    # 计算 `log_dir / f'forecast_{year}.seed_{seed}.log'`，并将数值结果保存到 `log_path`。
                    log_path = log_dir / f"forecast_{year}.seed_{seed}.log"
                    # 调用 `log_path.open` 读取源数据，并将结果保存到 `log_handle`。
                    log_handle = log_path.open("a", encoding="utf-8")
                    # 调用 `log_handle.write` 把本轮任务状态追加到日志文件。
                    log_handle.write(
                        f"\n=== launch {year}/seed={seed} at "
                        f"{time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
                    )
                    # 调用 `log_handle.flush` 把内存映射中的缓存修改刷新到磁盘。
                    log_handle.flush()
                    # 复制 `os.environ.copy`，避免后续更新污染原对象，并保存到 `environment`。
                    environment = os.environ.copy()
                    # 调用 `str` 把当前值转换为统一字符串表示，并将结果保存到 `environment['CUDA_VISIBLE_DEVICES']`。
                    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
                    # 调用 `subprocess.Popen` 启动独立训练子进程并保留进程句柄，并将结果保存到 `process`。
                    process = subprocess.Popen(
                        worker_command(args, year, seed, gpu),
                        cwd=VIVIT_DIR,
                        env=environment,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    # 把本轮结果追加到 `running`，累积后续需要汇总的记录。
                    running.append(
                        {
                            "process": process,
                            "gpu": gpu,
                            "year": year,
                            "seed": seed,
                            "log_path": log_path,
                            "log_handle": log_handle,
                            "started": time.perf_counter(),
                        }
                    )
                    # 用 `1` 对 `active` 做 `Add` 累积更新。
                    active += 1
                    # 调用 `print` 记录当前流程状态或诊断信息。
                    print(
                        f"launch GPU{gpu}: {year}/seed={seed} pid={process.pid}",
                        flush=True,
                    )

            # 构造当前步骤所需的列表，并保存到 `finished`。
            finished = []
            # 遍历 `running`，并把当前元素绑定到 `item` 逐项处理。
            for item in running:
                # 调用 `item['process'].poll` 查询训练子进程是否已经退出，并将结果保存到 `return_code`。
                return_code = item["process"].poll()
                # 检查 `return_code is None`；根据条件结果选择对应处理分支。
                if return_code is None:
                    # 跳过当前元素的剩余步骤，继续处理下一项。
                    continue
                # 调用 `item['log_handle'].close` 释放当前步骤占用的资源。
                item["log_handle"].close()
                # 把本轮结果追加到 `finished`，累积后续需要汇总的记录。
                finished.append(item)
                # 检查 `return_code`；根据条件结果选择对应处理分支。
                if return_code:
                    # 抛出 `RuntimeError(f'任务失败：{item['year']}/seed={item['seed']} GPU{item['gpu']}\n{tail(item['log_path']…`，阻止不满足数据契约的流程继续执行。
                    raise RuntimeError(
                        f"任务失败：{item['year']}/seed={item['seed']} GPU{item['gpu']}\n"
                        f"{tail(item['log_path'])}"
                    )
                # 计算 `(time.perf_counter() - item['started']) / 60`，并将数值结果保存到 `minutes`。
                minutes = (time.perf_counter() - item["started"]) / 60
                # 调用 `print` 记录当前流程状态或诊断信息。
                print(
                    f"done GPU{item['gpu']}: {item['year']}/seed={item['seed']} "
                    f"{minutes:.2f} min",
                    flush=True,
                )
            # 遍历 `finished`，并把当前元素绑定到 `item` 逐项处理。
            for item in finished:
                # 调用 `running.remove` 从运行队列移除已经结束的任务记录。
                running.remove(item)
            # 检查 `running and (not finished)`；根据条件结果选择对应处理分支。
            if running and not finished:
                # 调用 `time.sleep` 短暂等待外部任务状态更新，避免无间隔轮询。
                time.sleep(1)
    # 捕获 `except BaseException` 对应的异常，并执行明确的降级、记录或清理逻辑。
    except BaseException:
        # 遍历 `running`，并把当前元素绑定到 `item` 逐项处理。
        for item in running:
            # 检查 `item['process'].poll() is None`；根据条件结果选择对应处理分支。
            if item["process"].poll() is None:
                # 调用 `item['process'].terminate` 向仍在运行的子进程发送终止信号。
                item["process"].terminate()
            # 调用 `item['log_handle'].close` 释放当前步骤占用的资源。
            item["log_handle"].close()
        # 抛出 `空值`，阻止不满足数据契约的流程继续执行。
        raise

    # 调用 `finalize_run` 汇总运行结果并写出完成标记与最终报告，并将结果保存到 `summary`。
    summary = finalize_run(args, time.perf_counter() - started)
    # 调用 `validate_outputs` 校验输入、配置或产物是否满足当前阶段的数据契约。
    validate_outputs(args.run_name)
    # 调用 `write_experiment_report` 按约定格式写出当前计算产物。
    write_experiment_report(args.run_name)
    # 将 `summary` 作为当前函数结果返回给调用方。
    return summary




# 定义 `markdown_table` 函数，封装这一阶段可独立调用的处理步骤。
def markdown_table(frame: pd.DataFrame) -> str:
    """把数据表转换为适合实验报告的 Markdown 表格。
    
    输入：
        frame: pd.DataFrame。
    输出：
        str。
        """
    # 调用 `list` 整理数据结构，并将结果保存到 `columns`。
    columns = list(frame.columns)
    # 计算 `'| ' + ' | '.join(columns) + ' |'`，并将数值结果保存到 `header`。
    header = "| " + " | ".join(columns) + " |"
    # 计算 `'| ' + ' | '.join(['---'] * len(columns)) + ' |'`，并将数值结果保存到 `separator`。
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    # 构造当前步骤所需的列表，并保存到 `rows`。
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    # 将 `'\n'.join([header, separator, *rows])` 作为当前函数结果返回给调用方。
    return "\n".join([header, separator, *rows])


# 定义 `write_experiment_report` 函数，封装这一阶段可独立调用的处理步骤。
def write_experiment_report(run_name: str) -> Path:
    """根据当前 run 的真实产物生成中文报告，不覆盖历史根目录报告。
    
    输入：
        run_name: str。
    输出：
        Path。
        """

    # 调用 `run_paths` 解析当前 run 对应的数据与输出路径，并将结果保存到 `(_, _, quality_root)`。
    _, _, quality_root = run_paths(run_name)
    # 调用 `pd.read_parquet` 读取源数据，并将结果保存到 `training`。
    training = pd.read_parquet(quality_root / "vivit_training_summary.parquet")
    # 调用 `validate_outputs` 校验输入、配置或产物是否满足当前阶段的数据契约，并将结果保存到 `factor_audit`。
    factor_audit = validate_outputs(run_name)
    # 调用 `pd.read_parquet` 读取源数据，并将结果保存到 `sequence_summary`。
    sequence_summary = pd.read_parquet(SEQUENCE_SUMMARY_PATH)
    # 调用 `json.loads` 读取源数据，并将结果保存到 `run_manifest`。
    run_manifest = json.loads(
        (quality_root / "a2_training_manifest.json").read_text(encoding="utf-8")
    )
    # 调用 `source_manifests` 读取或生成记录数据来源与配置签名的 manifest，并将结果保存到 `(dataset_manifest, _)`。
    dataset_manifest, _ = source_manifests()
    # 调用 `round` 执行 `round` 对应的明确变换，并产生后续步骤使用的结果，并将结果保存到 `yearly`。
    yearly = (
        training.groupby("forecast_year", as_index=False)
        .agg(
            best_validation_ic_mean=("best_validation_ic", "mean"),
            best_validation_ic_min=("best_validation_ic", "min"),
            best_validation_ic_max=("best_validation_ic", "max"),
            best_epoch_mean=("best_epoch", "mean"),
            task_minutes=("elapsed_seconds", lambda values: values.sum() / 60),
        )
        .round(6)
    )
    # 调用 `pd.DataFrame` 转换数据表示，并将标准化对象保存到 `architecture`。
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
        columns=["项目", "设置"],
    )
    # 计算 `f'# ViViT 与 ViT 索引对齐实验报告\n\n生成时间：{pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M:%S')} UTC…`，并将结果写入 `report`。
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
    # 计算 `quality_root / 'EXPERIMENT_REPORT.md'`，并将数值结果保存到 `path`。
    path = quality_root / "EXPERIMENT_REPORT.md"
    # 调用 `atomic_write_text` 先写临时文件再原子替换正式产物，避免留下半成品。
    atomic_write_text(report, path)
    # 将 `path` 作为当前函数结果返回给调用方。
    return path


# 定义 `run_smoke_test` 函数，封装这一阶段可独立调用的处理步骤。
def run_smoke_test(args: argparse.Namespace) -> dict[str, object]:
    """按指定生产配置执行真实数据有限 batch 前向、反向与验证。
    
    输入：
        args: argparse.Namespace。
    输出：
        dict[str, object]。
        """

    # 检查 `not SEQUENCE_MANIFEST_PATH.exists()`；根据条件结果选择对应处理分支。
    if not SEQUENCE_MANIFEST_PATH.exists():
        # 调用 `build_sequence_store` 根据当前配置和输入构造后续流程所需对象。
        build_sequence_store(overwrite=False)
    # 调用 `require_aligned_sequence_store` 校验输入、配置或产物是否满足当前阶段的数据契约。
    require_aligned_sequence_store()
    # 调用 `training_config_from_args` 把命令行参数转换为不可变训练配置，并将结果保存到 `config`。
    config = training_config_from_args(args)
    # 调用 `validate_training_config` 校验输入、配置或产物是否满足当前阶段的数据契约。
    validate_training_config(config, args.compile_mode)
    # 调用 `torch.device` 解析训练或推理所使用的计算设备，并将结果保存到 `device`。
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 调用 `set_random_seed` 同时固定 Python、NumPy 和 PyTorch 随机状态。
    set_random_seed(args.seed)
    # 调用 `make_data_loaders` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `(train_loader, validation_loader, prediction_loader)`。
    train_loader, validation_loader, prediction_loader = make_data_loaders(
        args.forecast_year,
        args.seed,
        config.batch_size,
        config.inference_batch_size,
        config.train_batch_mode,
        config.loss_name,
        args.workers_per_task,
        device.type == "cuda",
    )
    # 调用 `set_train_epoch` 同步 epoch 到采样器并按轮次重建可复现顺序。
    set_train_epoch(train_loader, 1)
    # 调用 `next` 从迭代器取出下一批数据，并将结果保存到 `audit_batch`。
    audit_batch = next(iter(train_loader))
    # 调用 `build_model` 根据模型配置创建网络并部署到目标设备，并将结果保存到 `model`。
    model = build_model(
        device, bool(args.compile), args.compile_mode, config.train_batch_mode
    )
    # 调用 `build_optimizer` 根据训练配置为可学习参数创建优化器，并将结果保存到 `optimizer`。
    optimizer = build_optimizer(model, config, bool(args.fused_adam), device)
    # 调用 `train_one_epoch` 遍历训练集完成一个 epoch 的前向、反向传播和参数更新，并将结果保存到 `train_metrics`。
    train_metrics = train_one_epoch(
        model,
        train_loader,
        optimizer,
        device,
        bool(args.cuda_prefetch),
        config,
        max_batches=args.max_batches,
    )
    # 调用 `evaluate_model` 在不更新参数的条件下计算验证或样本外评价指标，并将结果保存到 `validation_metrics`。
    validation_metrics = evaluate_model(
        model,
        validation_loader,
        device,
        bool(args.cuda_prefetch),
        config.validation_metric,
        max_batches=args.max_batches,
    )
    # 设置 `prediction_rows` 的固定配置值，供后续流程统一使用。
    prediction_rows = 0
    # 检查 `prediction_loader is not None`；根据条件结果选择对应处理分支。
    if prediction_loader is not None:
        # 调用 `len` 汇总当前数据，并将统计量保存到 `prediction_rows`。
        prediction_rows = len(
            predict_model(
                model,
                prediction_loader,
                device,
                bool(args.cuda_prefetch),
                max_batches=1,
            )
        )
    # 读取 `audit_batch['frame_mask']` 的当前值，并赋给 `frame_mask`。
    frame_mask = audit_batch["frame_mask"]
    # 将 `{'forecast_year': args.forecast_year, 'seed': args.seed, 'device': str(device), 'train_batch_mo…` 作为当前函数结果返回给调用方。
    return {
        "forecast_year": args.forecast_year,
        "seed": args.seed,
        "device": str(device),
        "train_batch_mode": config.train_batch_mode,
        "loss_name": config.loss_name,
        "validation_metric": config.validation_metric,
        "batch_summary": loader_batch_summary(train_loader),
        "audit_video_shape": list(audit_batch["videos"].shape),
        "audit_frame_mask_shape": list(frame_mask.shape),
        "audit_valid_frame_ratio": float(frame_mask.float().mean().item()),
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "sample_prediction_rows": prediction_rows,
        "parameter_count": sum(p.numel() for p in unwrap_model(model).parameters()),
        "persistent_outputs_written": False,
    }


# 定义 `run_memory_benchmark` 函数，封装这一阶段可独立调用的处理步骤。
def run_memory_benchmark(args: argparse.Namespace) -> dict[str, object]:
    """用一个指定配置的真实 batch 测量单任务峰值显存和耗时。
    
    输入：
        args: argparse.Namespace。
    输出：
        dict[str, object]。
        """

    # 调用 `torch.device` 解析训练或推理所使用的计算设备，并将结果保存到 `device`。
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 检查 `device.type != 'cuda'`；根据条件结果选择对应处理分支。
    if device.type != "cuda":
        # 抛出 `RuntimeError('显存 benchmark 要求 CUDA')`，阻止不满足数据契约的流程继续执行。
        raise RuntimeError("显存 benchmark 要求 CUDA")
    # 调用 `require_aligned_sequence_store` 校验输入、配置或产物是否满足当前阶段的数据契约。
    require_aligned_sequence_store()
    # 调用 `training_config_from_args` 把命令行参数转换为不可变训练配置，并将结果保存到 `config`。
    config = training_config_from_args(args)
    # 调用 `validate_training_config` 校验输入、配置或产物是否满足当前阶段的数据契约。
    validate_training_config(config, args.compile_mode)
    # 调用 `make_data_loaders` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `(loader, _, _)`。
    loader, _, _ = make_data_loaders(
        args.forecast_year,
        args.seed,
        config.batch_size,
        config.inference_batch_size,
        config.train_batch_mode,
        config.loss_name,
        args.workers_per_task,
        pin_memory=True,
    )
    # 调用 `set_train_epoch` 同步 epoch 到采样器并按轮次重建可复现顺序。
    set_train_epoch(loader, 1)
    # 调用 `build_model` 根据模型配置创建网络并部署到目标设备，并将结果保存到 `model`。
    model = build_model(
        device, bool(args.compile), args.compile_mode, config.train_batch_mode
    )
    # 调用 `build_optimizer` 根据训练配置为可学习参数创建优化器，并将结果保存到 `optimizer`。
    optimizer = build_optimizer(model, config, bool(args.fused_adam), device)
    # 调用 `torch.cuda.reset_peak_memory_stats` 清零 CUDA 峰值显存计数器。
    torch.cuda.reset_peak_memory_stats(device)
    # 调用 `time.perf_counter` 读取高精度单调时钟以计算耗时，并将结果保存到 `started`。
    started = time.perf_counter()
    # 调用 `train_one_epoch` 遍历训练集完成一个 epoch 的前向、反向传播和参数更新，并将结果保存到 `metrics`。
    metrics = train_one_epoch(
        model,
        loader,
        optimizer,
        device,
        bool(args.cuda_prefetch),
        config,
        max_batches=1,
    )
    # 将 `{'forecast_year': args.forecast_year, 'train_batch_mode': config.train_batch_mode, 'loss_name':…` 作为当前函数结果返回给调用方。
    return {
        "forecast_year": args.forecast_year,
        "train_batch_mode": config.train_batch_mode,
        "loss_name": config.loss_name,
        "seconds": time.perf_counter() - started,
        "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        **metrics,
    }


# 定义 `validate_training_config` 函数，封装这一阶段可独立调用的处理步骤。
def validate_training_config(config: TrainingConfig, compile_mode: str) -> None:
    """校验训练模式、损失、早停与资源参数组合。
    
    输入：
        config: TrainingConfig。
        compile_mode: str。
    输出：
        None。
        """
    # 检查 `config.train_batch_mode not in TRAIN_BATCH_MODES`；根据条件结果选择对应处理分支。
    if config.train_batch_mode not in TRAIN_BATCH_MODES:
        # 抛出 `ValueError(f'未知训练 batch 模式：{config.train_batch_mode}')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(f"未知训练 batch 模式：{config.train_batch_mode}")
    # 检查 `config.loss_name not in LOSS_NAMES`；根据条件结果选择对应处理分支。
    if config.loss_name not in LOSS_NAMES:
        # 抛出 `ValueError(f'未知损失函数：{config.loss_name}')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(f"未知损失函数：{config.loss_name}")
    # 检查 `config.validation_metric not in VALIDATION_METRICS`；根据条件结果选择对应处理分支。
    if config.validation_metric not in VALIDATION_METRICS:
        # 抛出 `ValueError(f'未知验证指标：{config.validation_metric}')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(f"未知验证指标：{config.validation_metric}")
    # 检查 `config.loss_name == TEMPORAL_TURNOVER_LOSS and config.train_batch_mode != 'trade_date'`；根据条件结果选择对应处理分支。
    if config.loss_name == TEMPORAL_TURNOVER_LOSS and config.train_batch_mode != "trade_date":
        # 抛出 `ValueError('sigmoid_rank_ic_turnover 必须使用 trade_date batch')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("sigmoid_rank_ic_turnover 必须使用 trade_date batch")
    # 检查 `config.loss_name == TEMPORAL_TURNOVER_LOSS and compile_mode == 'reduce-overhead'`；根据条件结果选择对应处理分支。
    if config.loss_name == TEMPORAL_TURNOVER_LOSS and compile_mode == "reduce-overhead":
        # 抛出 `ValueError('时序换手损失禁止使用 reduce-overhead compile mode')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("时序换手损失禁止使用 reduce-overhead compile mode")
    # 检查 `config.batch_size <= 1 or config.inference_batch_size <= 1 or config.learning_rate <= 0 or (con…`；根据条件结果选择对应处理分支。
    if (
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
        # 抛出 `ValueError('训练、batch 或损失超参数非法')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("训练、batch 或损失超参数非法")

# 定义 `validate_outputs` 函数，封装这一阶段可独立调用的处理步骤。
def validate_outputs(run_name: str) -> pd.DataFrame:
    """核验 24 个任务与 8 个年度因子文件。
    
    输入：
        run_name: str。
    输出：
        pd.DataFrame。
        """

    # 调用 `run_paths` 解析当前 run 对应的数据与输出路径，并将结果保存到 `(model_root, prediction_root, quality_root)`。
    model_root, prediction_root, quality_root = run_paths(run_name)
    # 计算 `quality_root / 'vivit_training_summary.parquet'`，并将数值结果保存到 `summary_path`。
    summary_path = quality_root / "vivit_training_summary.parquet"
    # 检查 `not summary_path.exists()`；根据条件结果选择对应处理分支。
    if not summary_path.exists():
        # 抛出 `FileNotFoundError(summary_path)`，阻止不满足数据契约的流程继续执行。
        raise FileNotFoundError(summary_path)
    # 调用 `pd.read_parquet` 读取源数据，并将结果保存到 `training_summary`。
    training_summary = pd.read_parquet(summary_path)
    # 检查 `len(training_summary) != len(FORECAST_YEARS) * len(SEEDS)`；根据条件结果选择对应处理分支。
    if len(training_summary) != len(FORECAST_YEARS) * len(SEEDS):
        # 抛出 `AssertionError('训练汇总不是 8 年度 × 3 种子')`，阻止不满足数据契约的流程继续执行。
        raise AssertionError("训练汇总不是 8 年度 × 3 种子")

    # 构造当前步骤所需的列表，并保存到 `records`。
    records = []
    # 遍历 `FORECAST_YEARS`，并把当前元素绑定到 `year` 逐项处理。
    for year in FORECAST_YEARS:
        # 调用 `load_sequence_split` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `split`。
        split = load_sequence_split(year)
        # 调用 `len` 汇总当前数据，并将统计量保存到 `expected`。
        expected = len(split["prediction_indices"])
        # 计算 `prediction_root / f'vivit_predictions.{year}.parquet'`，并将数值结果保存到 `path`。
        path = prediction_root / f"vivit_predictions.{year}.parquet"
        # 检查 `expected == 0`；根据条件结果选择对应处理分支。
        if expected == 0:
            # 检查 `path.exists()`；根据条件结果选择对应处理分支。
            if path.exists():
                # 抛出 `AssertionError(f'{year} 年 prediction 为空但存在因子文件')`，阻止不满足数据契约的流程继续执行。
                raise AssertionError(f"{year} 年 prediction 为空但存在因子文件")
            # 把本轮结果追加到 `records`，累积后续需要汇总的记录。
            records.append(
                {
                    "forecast_year": year,
                    "factor_rows": 0,
                    "trade_dates": 0,
                    "start_date": None,
                    "end_date": None,
                    "status": "无 prediction，按 ViT 口径通过",
                }
            )
            # 跳过当前元素的剩余步骤，继续处理下一项。
            continue
        # 检查 `not path.exists()`；根据条件结果选择对应处理分支。
        if not path.exists():
            # 抛出 `FileNotFoundError(path)`，阻止不满足数据契约的流程继续执行。
            raise FileNotFoundError(path)
        # 调用 `pd.read_parquet` 读取源数据，并将结果保存到 `frame`。
        frame = pd.read_parquet(path)
        # 检查 `len(frame) != expected or frame['sample_index'].duplicated().any()`；根据条件结果选择对应处理分支。
        if len(frame) != expected or frame["sample_index"].duplicated().any():
            # 抛出 `AssertionError(f'{year} 年因子行数或主键错误')`，阻止不满足数据契约的流程继续执行。
            raise AssertionError(f"{year} 年因子行数或主键错误")
        # 检查 `not np.array_equal(frame['sample_index'].to_numpy(np.int64), split['prediction_indices'])`；根据条件结果选择对应处理分支。
        if not np.array_equal(
            frame["sample_index"].to_numpy(np.int64), split["prediction_indices"]
        ):
            # 抛出 `AssertionError(f'{year} 年因子索引与 ViT prediction split 不一致')`，阻止不满足数据契约的流程继续执行。
            raise AssertionError(f"{year} 年因子索引与 ViT prediction split 不一致")
        # 检查 `not np.isfinite(frame['prediction_ensemble']).all()`；根据条件结果选择对应处理分支。
        if not np.isfinite(frame["prediction_ensemble"]).all():
            # 抛出 `AssertionError(f'{year} 年因子包含非有限值')`，阻止不满足数据契约的流程继续执行。
            raise AssertionError(f"{year} 年因子包含非有限值")
        # 遍历 `SEEDS`，并把当前元素绑定到 `seed` 逐项处理。
        for seed in SEEDS:
            # 计算 `f'prediction_seed_{seed}_zscore'`，并将结果写入 `column`。
            column = f"prediction_seed_{seed}_zscore"
            # 读取 `frame.groupby('trade_date', sort=False)[column]` 的当前值，并赋给 `grouped`。
            grouped = frame.groupby("trade_date", sort=False)[column]
            # 调用 `float` 统一设备或数据类型，并将结果保存到 `max_mean`。
            max_mean = float(grouped.mean().abs().max())
            # 调用 `float` 统一设备或数据类型，并将结果保存到 `max_std_error`。
            max_std_error = float((grouped.std(ddof=0) - 1).abs().max())
            # 检查 `max_mean > 1e-05 or max_std_error > 1e-05`；根据条件结果选择对应处理分支。
            if max_mean > 1e-5 or max_std_error > 1e-5:
                # 抛出 `AssertionError(f'{year} 年 seed={seed} 日截面标准化失败')`，阻止不满足数据契约的流程继续执行。
                raise AssertionError(f"{year} 年 seed={seed} 日截面标准化失败")
        # 把本轮结果追加到 `records`，累积后续需要汇总的记录。
        records.append(
            {
                "forecast_year": year,
                "factor_rows": len(frame),
                "trade_dates": frame["trade_date"].nunique(),
                "start_date": frame["trade_date"].min(),
                "end_date": frame["trade_date"].max(),
                "status": "通过",
            }
        )

    # 明确确认本任务没有创建用户排除的交付包和回测结果。
    if (VIVIT_DIR / "delivery").exists():
        # 抛出 `AssertionError('本任务不应生成 delivery 目录')`，阻止不满足数据契约的流程继续执行。
        raise AssertionError("本任务不应生成 delivery 目录")
    # 检查 `(DATA_DIR / 'backtest').exists()`；根据条件结果选择对应处理分支。
    if (DATA_DIR / "backtest").exists():
        # 抛出 `AssertionError('本任务不应生成 backtest 目录')`，阻止不满足数据契约的流程继续执行。
        raise AssertionError("本任务不应生成 backtest 目录")
    # 将 `pd.DataFrame(records)` 作为当前函数结果返回给调用方。
    return pd.DataFrame(records)

# 定义 `add_training_arguments` 函数，封装这一阶段可独立调用的处理步骤。
def add_training_arguments(parser: argparse.ArgumentParser) -> None:
    """向命令行解析器注册训练后端参数。
    
    输入：
        parser: argparse.ArgumentParser。
    输出：
        None。
        """
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument(
        "--run-name", default="vivit_aligned_vit_indices_pearson_date_batch"
    )
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument(
        "--train-batch-mode", choices=TRAIN_BATCH_MODES, default="trade_date"
    )
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument("--batch-size", type=int, default=5000)
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument("--inference-batch-size", type=int, default=5000)
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument("--loss-name", choices=LOSS_NAMES, default="pearson_ic")
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument("--rank-temperature", type=float, default=0.1)
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument("--turnover-lambda", type=float, default=0.1)
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument("--huber-delta", type=float, default=1.0)
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument("--smooth-l1-beta", type=float, default=0.1)
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument("--hybrid-weight", type=float, default=0.5)
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument(
        "--validation-metric",
        choices=VALIDATION_METRICS,
        default="daily_mean_pearson_ic",
    )
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument("--workers-per-task", type=int, default=2)
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument("--learning-rate", type=float, default=0.001)
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument("--max-epochs", type=int, default=100)
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument("--patience", type=int, default=10)
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument("--compile", type=int, choices=(0, 1), default=1)
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument("--compile-mode", default="default")
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument("--fused-adam", type=int, choices=(0, 1), default=1)
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument("--cuda-prefetch", type=int, choices=(0, 1), default=1)
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument("--overwrite", type=int, choices=(0, 1), default=0)


# 定义 `add_trial_arguments` 函数，封装这一阶段可独立调用的处理步骤。
def add_trial_arguments(parser: argparse.ArgumentParser) -> None:
    """向命令行解析器注册单任务试运行参数。
    
    输入：
        parser: argparse.ArgumentParser。
    输出：
        None。
        """
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument("--forecast-year", type=int, default=2018)
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument("--seed", type=int, default=42)
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument("--max-batches", type=int, default=1)


# 定义 `parse_args` 函数，封装这一阶段可独立调用的处理步骤。
def parse_args() -> argparse.Namespace:
    """解析并校验命令行参数。
    
    输入：
        无显式输入参数。
    输出：
        argparse.Namespace。
        """
    # 调用 `argparse.ArgumentParser` 创建命令行参数解析器，并将结果保存到 `parser`。
    parser = argparse.ArgumentParser(description=__doc__)
    # 调用 `parser.add_subparsers` 创建子命令分发器以区分不同运行模式，并将结果保存到 `subparsers`。
    subparsers = parser.add_subparsers(dest="command", required=True)
    # 调用 `subparsers.add_parser` 注册一个可独立调用的命令行子命令，并将结果保存到 `smoke_parser`。
    smoke_parser = subparsers.add_parser("smoke")
    # 调用 `add_training_arguments` 向命令行解析器注册当前流程使用的参数。
    add_training_arguments(smoke_parser)
    # 调用 `add_trial_arguments` 向命令行解析器注册该运行模式的参数。
    add_trial_arguments(smoke_parser)
    # 调用 `smoke_parser.set_defaults` 设置当前子命令的默认运行参数。
    smoke_parser.set_defaults(compile=0, workers_per_task=0)

    # 调用 `subparsers.add_parser` 注册一个可独立调用的命令行子命令，并将结果保存到 `benchmark_parser`。
    benchmark_parser = subparsers.add_parser("benchmark")
    # 调用 `add_training_arguments` 向命令行解析器注册当前流程使用的参数。
    add_training_arguments(benchmark_parser)
    # 调用 `add_trial_arguments` 向命令行解析器注册该运行模式的参数。
    add_trial_arguments(benchmark_parser)

    # 调用 `subparsers.add_parser` 注册一个可独立调用的命令行子命令，并将结果保存到 `worker_parser`。
    worker_parser = subparsers.add_parser("worker")
    # 调用 `add_training_arguments` 向命令行解析器注册当前流程使用的参数。
    add_training_arguments(worker_parser)
    # 调用 `worker_parser.add_argument` 向命令行解析器注册当前流程使用的参数。
    worker_parser.add_argument("--worker-year", type=int, required=True)
    # 调用 `worker_parser.add_argument` 向命令行解析器注册当前流程使用的参数。
    worker_parser.add_argument("--worker-seed", type=int, required=True)
    # 调用 `worker_parser.add_argument` 向命令行解析器注册当前流程使用的参数。
    worker_parser.add_argument("--worker-physical-gpu", type=int, required=True)

    # 调用 `subparsers.add_parser` 注册一个可独立调用的命令行子命令，并将结果保存到 `a2_parser`。
    a2_parser = subparsers.add_parser("a2")
    # 调用 `add_training_arguments` 向命令行解析器注册当前流程使用的参数。
    add_training_arguments(a2_parser)
    # 调用 `a2_parser.add_argument` 向命令行解析器注册当前流程使用的参数。
    a2_parser.add_argument("--device-ids", default="0,1,2,3")
    # 调用 `a2_parser.add_argument` 向命令行解析器注册当前流程使用的参数。
    a2_parser.add_argument("--tasks-per-gpu", type=int, default=1)

    # 执行当前分布式训练步骤并保持既定数据语义。
    ddp_parser = subparsers.add_parser("ddp")
    # 执行当前分布式训练步骤并保持既定数据语义。
    add_training_arguments(ddp_parser)
    # 执行当前分布式训练步骤并保持既定数据语义。
    ddp_parser.add_argument("--device-ids", default="0,1,2,3")

    # 执行当前分布式训练步骤并保持既定数据语义。
    ddp_worker_parser = subparsers.add_parser("ddp-worker")
    # 执行当前分布式训练步骤并保持既定数据语义。
    add_training_arguments(ddp_worker_parser)
    # 执行当前分布式训练步骤并保持既定数据语义。
    ddp_worker_parser.add_argument("--worker-year", type=int, required=True)
    # 执行当前分布式训练步骤并保持既定数据语义。
    ddp_worker_parser.add_argument("--worker-seed", type=int, required=True)
    # 执行当前分布式训练步骤并保持既定数据语义。
    ddp_worker_parser.add_argument("--worker-physical-gpu", type=int, required=True)
    # 执行当前分布式训练步骤并保持既定数据语义。
    ddp_worker_parser.add_argument("--device-ids", required=True)

    # 执行当前分布式训练步骤并保持既定数据语义。
    ddp_smoke_parser = subparsers.add_parser("ddp-smoke")
    # 执行当前分布式训练步骤并保持既定数据语义。
    add_training_arguments(ddp_smoke_parser)
    # 执行当前分布式训练步骤并保持既定数据语义。
    add_trial_arguments(ddp_smoke_parser)
    # 执行当前分布式训练步骤并保持既定数据语义。
    ddp_smoke_parser.add_argument("--device-ids", default="0,1,2,3")
    # 执行当前分布式训练步骤并保持既定数据语义。
    ddp_smoke_parser.set_defaults(compile=0, workers_per_task=0)

    # 调用 `subparsers.add_parser` 注册一个可独立调用的命令行子命令，并将结果保存到 `validate_parser`。
    validate_parser = subparsers.add_parser("validate")
    # 调用 `validate_parser.add_argument` 校验输入、配置或产物是否满足当前阶段的数据契约。
    validate_parser.add_argument("--run-name", required=True)
    # 调用 `subparsers.add_parser` 注册一个可独立调用的命令行子命令，并将结果保存到 `report_parser`。
    report_parser = subparsers.add_parser("report")
    # 调用 `report_parser.add_argument` 根据当前结果生成可审计的报告或展示产物。
    report_parser.add_argument("--run-name", required=True)
    # 将 `parser.parse_args()` 作为当前函数结果返回给调用方。
    return parser.parse_args()


# 定义命令行入口，解析参数并按用户选择调度当前阶段的主流程。
def main() -> None:
    """根据命令行子命令分派基线训练、验证或报告流程。
    
    输入：
        无显式输入参数。
    输出：
        None。
        """
    # 调用 `parse_args` 解析命令行文本并生成结构化参数，并将结果保存到 `args`。
    args = parse_args()
    # 检查 `args.command == 'smoke'`；根据条件结果选择对应处理分支。
    if args.command == "smoke":
        # 调用 `print` 记录当前流程状态或诊断信息。
        print(json.dumps(run_smoke_test(args), ensure_ascii=False, indent=2))
    # 检查 `args.command == 'benchmark'`；根据条件结果选择对应处理分支。
    elif args.command == "benchmark":
        # 调用 `print` 记录当前流程状态或诊断信息。
        print(json.dumps(run_memory_benchmark(args), ensure_ascii=False, indent=2))
    # 检查 `args.command == 'worker'`；根据条件结果选择对应处理分支。
    elif args.command == "worker":
        # 执行当前分布式训练步骤并保持既定数据语义。
        args.execution_route = "a2"
        # 执行当前分布式训练步骤并保持既定数据语义。
        args.ddp_world_size = 1
        # 调用 `print` 记录当前流程状态或诊断信息。
        print(json.dumps(train_single_run(args), ensure_ascii=False, indent=2))
    # 执行当前分布式训练步骤并保持既定数据语义。
    elif args.command == "ddp-worker":
        # 执行当前分布式训练步骤并保持既定数据语义。
        run_ddp_worker(args)
    # 执行当前分布式训练步骤并保持既定数据语义。
    elif args.command == "ddp-smoke":
        # 执行当前分布式训练步骤并保持既定数据语义。
        run_ddp_worker(args, smoke=True)
    # 检查 `args.command == 'a2'`；根据条件结果选择对应处理分支。
    elif args.command == "a2":
        # 调用 `print` 记录当前流程状态或诊断信息。
        print(run_a2(args).to_string(index=False))
    # 执行当前分布式训练步骤并保持既定数据语义。
    elif args.command == "ddp":
        # 执行当前分布式训练步骤并保持既定数据语义。
        print(run_ddp(args).to_string(index=False))
    # 检查 `args.command == 'validate'`；根据条件结果选择对应处理分支。
    elif args.command == "validate":
        # 调用 `print` 记录当前流程状态或诊断信息。
        print(validate_outputs(args.run_name).to_string(index=False))
    # 检查 `args.command == 'report'`；根据条件结果选择对应处理分支。
    elif args.command == "report":
        # 调用 `print` 记录当前流程状态或诊断信息。
        print(write_experiment_report(args.run_name))


# 检查 `__name__ == '__main__'`；根据条件结果选择对应处理分支。
if __name__ == "__main__":
    # 调用 `main` 启动当前模块的命令行主流程。
    main()
