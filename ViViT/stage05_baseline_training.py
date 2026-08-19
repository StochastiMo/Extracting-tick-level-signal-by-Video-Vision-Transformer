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

def inference_model(model: nn.Module) -> nn.Module:
    """返回不触发 DDP 集体前向的本地模型。"""

    return getattr(model, "module", model)


def make_distributed_data_loaders(
    forecast_year: int,
    seed: int,
    config: TrainingConfig,
    workers: int,
    pin_memory: bool,
    context: distributed.DistributedContext,
) -> tuple[DataLoader, DataLoader, DataLoader | None, dict[str, object]]:
    """构造无重复的逐 rank loader，并保持全局 batch 语义。"""

    split = load_sequence_split(forecast_year)
    common: dict[str, object] = {
        "num_workers": workers,
        "pin_memory": pin_memory,
        "collate_fn": VideoBatchCollator(),
    }
    if workers > 0:
        common.update({"persistent_workers": True, "prefetch_factor": 1})
    generator = torch.Generator().manual_seed(seed)
    train_indices = split["train_indices"]
    train_dataset = SampleIndexDataset(train_indices)
    if config.train_batch_mode == "trade_date":
        date_ids = np.load(DATE_IDS_PATH, mmap_mode="r")
        temporal_mode = config.loss_name == TEMPORAL_TURNOVER_LOSS
        instruments = None
        if temporal_mode:
            instruments = pd.read_parquet(
                SAMPLE_INDEX_PATH, columns=["instrument_id"]
            )["instrument_id"].to_numpy(copy=False)
        train_sampler = distributed.DistributedTradeDateBatchSampler(
            train_indices,
            date_ids,
            seed=seed,
            shuffle=not temporal_mode,
            instrument_ids=instruments,
            rank=context.rank,
            world_size=context.world_size,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            generator=generator,
            **common,
        )
        batch_summary = train_sampler.summary()
    else:
        train_sampler = distributed.DistributedBatchSampler(
            len(train_dataset),
            config.batch_size,
            rank=context.rank,
            world_size=context.world_size,
            seed=seed,
            shuffle=True,
            drop_last=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            generator=generator,
            **common,
        )
        batch_count = len(train_indices) // config.batch_size
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

    def inference_loader(indices: np.ndarray) -> DataLoader:
        """执行 inference_loader 对应的分布式训练处理。"""
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
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            generator=generator,
            **common,
        )

    validation_loader = inference_loader(split["validation_indices"])
    prediction_loader = None
    if len(split["prediction_indices"]):
        prediction_loader = inference_loader(split["prediction_indices"])
    return train_loader, validation_loader, prediction_loader, batch_summary


def gather_global_training_batch(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    batch: dict[str, torch.Tensor],
    context: distributed.DistributedContext | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """执行 gather_global_training_batch 对应的分布式训练处理。"""

    sample_indices = batch["sample_indices"].to(predictions.device)
    if context is None:
        return predictions, targets, sample_indices
    global_predictions = distributed.differentiable_all_gather(predictions)
    global_targets = distributed.all_gather_tensor(targets)
    global_indices = distributed.all_gather_tensor(sample_indices)
    order = torch.argsort(global_indices)
    return (
        global_predictions[order],
        global_targets[order],
        global_indices[order],
    )


def gather_global_inference_tensor(
    tensor: torch.Tensor,
    context: distributed.DistributedContext | None,
) -> torch.Tensor:
    """执行 gather_global_inference_tensor 对应的分布式训练处理。"""
    return tensor if context is None else distributed.all_gather_tensor(tensor)


VIVIT_DIR = Path(__file__).resolve().parent
VIT_DIR = VIVIT_DIR.parent / "ViT"
if str(VIT_DIR) not in sys.path:
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

DATA_DIR = VIVIT_DIR / "data"
FEATURE_STORE_DIR = DATA_DIR / "feature_store"
SEQUENCE_STORE_DIR = DATA_DIR / "aligned_sequence_store"
HISTORY_INDICES_PATH = SEQUENCE_STORE_DIR / "history_indices.npy"
SEQUENCE_MANIFEST_PATH = SEQUENCE_STORE_DIR / "sequence_manifest.json"
SEQUENCE_SUMMARY_PATH = SEQUENCE_STORE_DIR / "rolling_split_summary.parquet"

FEATURES_PATH = FEATURE_STORE_DIR / "features.npy"
TARGETS_PATH = FEATURE_STORE_DIR / "targets.npy"
DATE_IDS_PATH = FEATURE_STORE_DIR / "date_ids.npy"
SAMPLE_INDEX_PATH = FEATURE_STORE_DIR / "sample_index.parquet"
SOURCE_DATASET_MANIFEST_PATH = FEATURE_STORE_DIR / "dataset_manifest.json"
SOURCE_ROLLING_MANIFEST_PATH = FEATURE_STORE_DIR / "rolling_split_manifest.json"
SOURCE_SPLIT_DIR = FEATURE_STORE_DIR / "rolling_splits"

FORECAST_YEARS = tuple(range(2018, 2026))
SEEDS = (42, 43, 44)
SEQUENCE_LENGTH = 20


# 使用 `dataclass(frozen=True)` 包装 `TrainingConfig`，并定义其可复用的处理逻辑。
@dataclass(frozen=True)
class TrainingConfig:
    """研报训练参数与项目复现参数。
    """

    train_batch_mode: str = "trade_date"
    batch_size: int = 5000
    inference_batch_size: int = 5000
    loss_name: str = "pearson_ic"
    rank_temperature: float = 0.1
    turnover_lambda: float = 0.1
    huber_delta: float = 1.0
    smooth_l1_beta: float = 0.1
    hybrid_weight: float = 0.5
    validation_metric: str = "daily_mean_pearson_ic"
    learning_rate: float = 0.001
    max_epochs: int = 100
    early_stopping_patience: int = 10
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.0
    improvement_tolerance: float = 1e-6


MODEL_CONFIG = VivitModelConfig()


def set_random_seed(seed: int) -> None:
    """同步设置 Python、NumPy、CPU 与 CUDA 随机种子。"""
    random.seed(seed)  # 固定本轮采样种子
    np.random.seed(seed)  # 固定本轮采样种子
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unwrap_model(model: nn.Module) -> nn.Module:
    """从编译或包装模型中取回原始 PyTorch 模型。"""
    wrapped = getattr(model, "module", None)
    if wrapped is not None:
        return unwrap_model(wrapped)
    original = getattr(model, "_orig_mod", None)
    return unwrap_model(original) if original is not None else model


def build_model(
    device: torch.device,
    use_compile: bool,
    compile_mode: str,
    train_batch_mode: str = "trade_date",
) -> nn.Module:
    """按当前配置构建模型，并保持历史 checkpoint 的参数键兼容。"""
    model = OrderFlowViViT(MODEL_CONFIG).to(device)
    if use_compile:
        if device.type != "cuda":
            raise RuntimeError("torch.compile 训练要求 CUDA")
        if train_batch_mode == "trade_date":
            model = torch.compile(model, mode=compile_mode, dynamic=True)  # 编译模型计算图
        else:
            model = torch.compile(model, mode=compile_mode)  # 编译模型计算图
    return model


def autocast_context(device: torch.device) -> contextlib.AbstractContextManager:
    """按设备和配置返回混合精度上下文。"""
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


class CudaBatchPrefetcher:
    """用独立 CUDA stream 预取下一批视频、掩码与标签。
    """

    def __init__(self, loader: DataLoader, device: torch.device, enabled: bool) -> None:
        self.loader = loader
        self.device = device
        self.enabled = enabled and device.type == "cuda"

    def _move(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """把批次张量非阻塞搬到 CUDA，并保留非张量元数据。"""
        non_blocking = self.device.type == "cuda"
        return (
            batch["videos"].to(self.device, non_blocking=non_blocking),
            batch["frame_mask"].to(self.device, non_blocking=non_blocking),
            batch["targets"].to(self.device, non_blocking=non_blocking),
        )

    def __iter__(self) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]]:
        """按既定顺序迭代 CudaBatchPrefetcher 中的训练批次。"""
        if not self.enabled:
            for batch in self.loader:
                videos, frame_mask, targets = self._move(batch)
                # 对 `(yield (batch, videos, frame_mask, targets))` 求值，以触发当前步骤需要的计算。
                yield batch, videos, frame_mask, targets
            return

        iterator = iter(self.loader)
        stream = torch.cuda.Stream(device=self.device)
        next_item = None

        def preload() -> None:
            """把下一批数据异步搬到 GPU，供当前批计算后立即使用。"""
            nonlocal next_item
            try:
                batch = next(iterator)
            except StopIteration:
                next_item = None
                return
            with torch.cuda.stream(stream):
                videos, frame_mask, targets = self._move(batch)
            next_item = (batch, videos, frame_mask, targets)

        preload()  # 异步预取下一批数据
        while next_item is not None:
            torch.cuda.current_stream(self.device).wait_stream(stream)  # 等待预取流完成
            batch, videos, frame_mask, targets = next_item
            for tensor in (videos, frame_mask, targets):
                tensor.record_stream(torch.cuda.current_stream(self.device))  # 绑定当前 CUDA stream 生命周期
            preload()  # 异步预取下一批数据
            # 对 `(yield (batch, videos, frame_mask, targets))` 求值，以触发当前步骤需要的计算。
            yield batch, videos, frame_mask, targets


def loss_kwargs(config: TrainingConfig) -> dict[str, object]:
    """整理基础损失需要的温度、稳健回归和换手超参数。"""
    return {
        "loss_name": config.loss_name,
        "rank_temperature": config.rank_temperature,
        "turnover_lambda": config.turnover_lambda,
        "huber_delta": config.huber_delta,
        "smooth_l1_beta": config.smooth_l1_beta,
        "hybrid_weight": config.hybrid_weight,
    }


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
    """执行该模式下的训练步骤，并返回损失与诊断信息。"""
    model.train()  # 启用训练模式
    date_batch_mode = isinstance(loader.batch_sampler, TradeDateBatchSampler)
    loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    batch_pearson_sum = torch.zeros((), dtype=torch.float64, device=device)
    weighted_pearson = torch.zeros((), dtype=torch.float64, device=device)
    soft_rank_ic_sum = torch.zeros((), dtype=torch.float64, device=device)
    sample_count = 0
    batch_count = 0
    soft_rank_batch_count = 0
    min_batch_size, max_batch_size = None, 0
    started = time.perf_counter()

    for batch_number, (batch, videos, frame_mask, targets) in enumerate(
        CudaBatchPrefetcher(loader, device, use_prefetch), start=1
    ):
        if date_batch_mode:
            date_ids = batch["date_ids"]
            if int(date_ids.min()) != int(date_ids.max()):
                raise AssertionError("日截面训练 batch 混入多个交易日")
        optimizer.zero_grad(set_to_none=True)  # 清空上一批梯度
        with autocast_context(device):
            predictions = model(videos, frame_mask)  # 执行当前 batch 前向传播
        predictions, targets, _ = gather_global_training_batch(
            predictions, targets, batch, distributed_context
        )
        loss = compute_training_loss(predictions, targets, **loss_kwargs(config))  # 按实验口径计算可微损失
        if not torch.isfinite(loss):
            raise FloatingPointError("训练损失出现 NaN 或 Inf")
        loss.backward()  # 反向传播当前损失
        optimizer.step()  # 更新参数或学习率状态

        with torch.no_grad():
            batch_pearson = pearson_correlation(predictions, targets).double()
        current_size = len(targets)
        loss_sum += loss.detach().double()
        batch_pearson_sum += batch_pearson
        weighted_pearson += batch_pearson * current_size
        if config.loss_name in RANK_LOSS_NAMES:
            soft_rank_ic_sum += -loss.detach().double()
            soft_rank_batch_count += 1
        sample_count += current_size
        batch_count += 1
        min_batch_size = current_size if min_batch_size is None else min(
            min_batch_size, current_size
        )
        max_batch_size = max(max_batch_size, current_size)
        if max_batches is not None and batch_number >= max_batches:
            break

    if not batch_count:
        raise ValueError("训练 loader 没有可用 batch")
    if device.type == "cuda":
        torch.cuda.synchronize(device)  # 同步 GPU 后计时
    batch_mean_pearson = float((batch_pearson_sum / batch_count).item())
    sample_weighted_pearson = float((weighted_pearson / sample_count).item())
    selected_train_pearson = (
        batch_mean_pearson if date_batch_mode else sample_weighted_pearson
    )
    soft_rank_ic = (
        float((soft_rank_ic_sum / soft_rank_batch_count).item())
        if soft_rank_batch_count
        else float("nan")
    )
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
    """执行该模式下的训练步骤，并返回损失与诊断信息。"""
    sampler = loader.batch_sampler
    if (
        not isinstance(sampler, TradeDateBatchSampler)
        or sampler.temporal_pairs is None
        or sampler.shuffle
    ):
        raise ValueError("时序换手损失要求按日期升序且关闭 reshuffle")
    available_pairs = len(sampler.temporal_pairs)
    pair_limit = available_pairs if max_batches is None else min(
        int(max_batches), available_pairs
    )
    if pair_limit < 1:
        raise ValueError("时序换手训练至少需要一个相邻日期对")

    model.train()  # 启用训练模式
    optimizer.zero_grad(set_to_none=True)  # 清空上一批梯度
    loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    pearson_sum = torch.zeros((), dtype=torch.float64, device=device)
    weighted_pearson = torch.zeros((), dtype=torch.float64, device=device)
    prediction_rank_sum = torch.zeros((), dtype=torch.float64, device=device)
    temporal_rank_sum = torch.zeros((), dtype=torch.float64, device=device)
    temporal_penalty_sum = torch.zeros((), dtype=torch.float64, device=device)
    sample_count = 0
    pair_count = 0
    min_batch_size, max_batch_size = None, 0
    previous_predictions = None
    previous_date_id = None
    started = time.perf_counter()

    for batch, videos, frame_mask, targets in CudaBatchPrefetcher(
        loader, device, use_prefetch
    ):
        date_ids = batch["date_ids"]
        if int(date_ids.min()) != int(date_ids.max()):
            raise AssertionError("日截面训练 batch 混入多个交易日")
        current_date_id = int(date_ids[0])
        with autocast_context(device):
            current_predictions = model(videos, frame_mask)  # 执行当前 batch 前向传播
        current_predictions, targets, _ = gather_global_training_batch(
            current_predictions, targets, batch, distributed_context
        )
        if previous_predictions is None:
            previous_predictions = current_predictions.detach()
            previous_date_id = current_date_id
            del current_predictions
            continue
        if current_date_id != previous_date_id + 1:
            raise AssertionError(
                f"时序训练日期不连续：{previous_date_id}→{current_date_id}"
            )

        pair = sampler.temporal_pair(pair_count)
        components = temporal_turnover_loss_components(  # 按实验口径计算可微损失
            current_predictions,
            targets,
            previous_predictions,
            pair["previous_positions"],
            pair["current_positions"],
            config.rank_temperature,
            config.turnover_lambda,
        )
        loss = components["loss"]
        values = torch.stack(
            [
                loss,
                components["prediction_soft_rank_ic"],
                components["temporal_soft_rank_ic"],
                components["temporal_penalty"],
            ]
        )
        if not torch.isfinite(values).all():
            raise FloatingPointError("时序换手训练损失或审计指标出现 NaN/Inf")
        (loss / pair_limit).backward()  # 反向传播当前损失

        with torch.no_grad():
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
        min_batch_size = current_size if min_batch_size is None else min(
            min_batch_size, current_size
        )
        max_batch_size = max(max_batch_size, current_size)
        previous_predictions = current_predictions.detach()
        previous_date_id = current_date_id
        del current_predictions, components, loss, values
        if pair_count >= pair_limit:
            break

    if pair_count != pair_limit:
        raise AssertionError(f"时序日期对覆盖不完整：{pair_count}/{pair_limit}")
    optimizer.step()  # 更新参数或学习率状态
    if device.type == "cuda":
        torch.cuda.synchronize(device)  # 同步 GPU 后计时
    mean_pearson = float((pearson_sum / pair_count).item())
    weighted_mean_pearson = float((weighted_pearson / sample_count).item())
    prediction_soft_rank = float((prediction_rank_sum / pair_count).item())
    temporal_soft_rank = float((temporal_rank_sum / pair_count).item())
    temporal_penalty = float((temporal_penalty_sum / pair_count).item())
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
    """训练一个完整 epoch，并返回损失与批次诊断。"""
    if config.loss_name == TEMPORAL_TURNOVER_LOSS:
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


_INSTRUMENT_IDS: np.ndarray | None = None


def instrument_ids() -> np.ndarray:
    """根据样本行号返回对应股票代码。"""
    global _INSTRUMENT_IDS
    if _INSTRUMENT_IDS is None:
        _INSTRUMENT_IDS = pd.read_parquet(
            SAMPLE_INDEX_PATH, columns=["instrument_id"]
        )["instrument_id"].to_numpy(copy=False)
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
    """在验证集上评估模型，并汇总逐日 IC 与损失诊断。"""
    local_model = inference_model(model)  # 关闭 dropout 等训练行为
    local_model.eval()
    prediction_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    date_parts: list[np.ndarray] = []
    sample_index_parts: list[np.ndarray] = []
    started = time.perf_counter()
    for batch_number, (batch, videos, frame_mask, targets) in enumerate(
        CudaBatchPrefetcher(loader, device, use_prefetch), start=1
    ):
        with autocast_context(device):
            predictions = local_model(videos, frame_mask)  # 执行当前 batch 前向传播
        prediction_parts.append(predictions.float().cpu().numpy().astype(np.float64))
        target_parts.append(targets.float().cpu().numpy().astype(np.float64))
        date_parts.append(batch["date_ids"].numpy().astype(np.int64, copy=False))
        sample_index_parts.append(
            batch["sample_indices"].numpy().astype(np.int64, copy=False)
        )
        if max_batches is not None and batch_number >= max_batches:
            break
    if not prediction_parts:
        raise ValueError("验证 loader 没有可用 batch")
    if device.type == "cuda":
        torch.cuda.synchronize(device)  # 同步 GPU 后计时
    predictions_tensor = torch.as_tensor(
        np.concatenate(prediction_parts), device=device, dtype=torch.float64
    )
    targets_tensor = torch.as_tensor(
        np.concatenate(target_parts), device=device, dtype=torch.float64
    )
    dates_tensor = torch.as_tensor(
        np.concatenate(date_parts), device=device, dtype=torch.int64
    )
    indices_tensor = torch.as_tensor(
        np.concatenate(sample_index_parts), device=device, dtype=torch.int64
    )
    predictions_tensor = gather_global_inference_tensor(
        predictions_tensor, distributed_context
    )
    targets_tensor = gather_global_inference_tensor(targets_tensor, distributed_context)
    dates_tensor = gather_global_inference_tensor(dates_tensor, distributed_context)
    indices_tensor = gather_global_inference_tensor(indices_tensor, distributed_context)
    order = torch.argsort(indices_tensor)
    selected_indices = indices_tensor[order].cpu().numpy()
    result = summarize_validation_arrays(
        predictions_tensor[order].cpu().numpy(),
        targets_tensor[order].cpu().numpy(),
        dates_tensor[order].cpu().numpy(),
        validation_metric,
        instrument_ids=instrument_ids()[selected_indices],
    )
    result["validation_seconds"] = time.perf_counter() - started
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
    """在预测集上生成与样本行号严格对齐的模型分数。"""
    local_model = inference_model(model)  # 关闭 dropout 等训练行为
    local_model.eval()
    sample_parts: list[np.ndarray] = []
    prediction_parts: list[np.ndarray] = []
    for batch_number, (batch, videos, frame_mask, _) in enumerate(
        CudaBatchPrefetcher(loader, device, use_prefetch), start=1
    ):
        with autocast_context(device):
            predictions = local_model(videos, frame_mask)  # 执行当前 batch 前向传播
        sample_parts.append(batch["sample_indices"].numpy())
        prediction_parts.append(predictions.float().cpu().numpy())
        if max_batches is not None and batch_number >= max_batches:
            break
    if sample_parts:
        local_indices = torch.as_tensor(
            np.concatenate(sample_parts), device=device, dtype=torch.int64
        )
        local_predictions = torch.as_tensor(
            np.concatenate(prediction_parts), device=device, dtype=torch.float32
        )
    else:
        local_indices = torch.empty(0, device=device, dtype=torch.int64)
        local_predictions = torch.empty(0, device=device, dtype=torch.float32)
    global_indices = gather_global_inference_tensor(local_indices, distributed_context)
    global_predictions = gather_global_inference_tensor(
        local_predictions, distributed_context
    )
    order = torch.argsort(global_indices)
    return pd.DataFrame(
        {
            "sample_index": global_indices[order].cpu().numpy(),
            "prediction": global_predictions[order].cpu().numpy().astype(np.float32),
        }
    )


def run_paths(run_name: str) -> tuple[Path, Path, Path]:
    """返回指定年度和随机种子的模型、预测与日志路径。"""
    model_root = VIVIT_DIR / "models" / "runs" / run_name
    prediction_root = DATA_DIR / "predictions" / "runs" / run_name
    quality_root = DATA_DIR / "quality" / "runs" / run_name
    for path in (model_root, prediction_root, quality_root):
        path.mkdir(parents=True, exist_ok=True)
    return model_root, prediction_root, quality_root


def build_optimizer(
    model: nn.Module,
    config: TrainingConfig,
    fused: bool,
    device: torch.device,
) -> torch.optim.Adam:
    """创建优化器，并按配置启用 fused Adam。"""
    return torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
        weight_decay=config.weight_decay,
        fused=fused and device.type == "cuda",
    )


def training_config_from_args(args: argparse.Namespace) -> TrainingConfig:
    """把命令行参数转换为不可变训练配置。"""
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


def training_payload(
    args: argparse.Namespace,
    year: int,
    seed: int,
    batch_summary: dict[str, object],
) -> dict[str, object]:
    """整理训练配置，供 checkpoint 判断是否可以继续使用。"""
    sequence_manifest = require_aligned_sequence_store()
    config = training_config_from_args(args)
    validate_training_config(config, args.compile_mode)
    temporal_mode = config.loss_name == TEMPORAL_TURNOVER_LOSS
    date_batches = config.train_batch_mode == "trade_date"
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


def load_canonical_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    """把规范化参数字典加载到原始或包装模型。"""
    unwrap_model(model).load_state_dict(state)


def load_best_model(
    path: Path,
    device: torch.device,
    use_compile: bool,
    compile_mode: str,
    train_batch_mode: str,
) -> nn.Module:
    """加载最佳验证 checkpoint，并恢复可用于预测的模型。"""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_model(device, use_compile, compile_mode, train_batch_mode)  # 创建模型并部署到设备
    load_canonical_state(model, checkpoint["model_state"])  # 恢复模型权重
    model.eval()  # 关闭 dropout 等训练行为
    return model


def train_single_run(
    args: argparse.Namespace,
    distributed_context: distributed.DistributedContext | None = None,
) -> dict[str, object]:
    """训练或恢复一个年度、一个随机种子。"""

    year, seed = args.worker_year, args.worker_seed
    if year is None or seed is None:
        raise ValueError("worker 需要 --worker-year 和 --worker-seed")
    require_aligned_sequence_store()
    config = training_config_from_args(args)
    validate_training_config(config, args.compile_mode)
    primary = distributed_context is None or distributed_context.is_primary
    device = (
        distributed_context.device
        if distributed_context is not None
        else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    pin_memory = device.type == "cuda"
    set_random_seed(seed + (distributed_context.rank if distributed_context else 0))
    if distributed_context is None:
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
        batch_summary = loader_batch_summary(train_loader)
    else:
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

    model_root, _, _ = run_paths(args.run_name)
    run_directory = model_root / f"forecast_{year}" / f"seed_{seed}"
    if primary:
        run_directory.mkdir(parents=True, exist_ok=True)
    if distributed_context is not None:
        dist.barrier()
    best_path = run_directory / "best.pt"
    last_path = run_directory / "last.pt"
    history_path = run_directory / "history.parquet"
    prediction_path = run_directory / "prediction.parquet"
    completed_path = run_directory / "completed.json"

    payload = training_payload(args, year, seed, batch_summary)
    run_signature = stable_hash(payload)  # 生成数据复用签名
    if primary:  # 原子落盘，避免半成品
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
    if distributed_context is not None:
        dist.barrier()
    if completed_path.exists() and best_path.exists() and not args.overwrite:
        completed = json.loads(completed_path.read_text(encoding="utf-8"))
        if completed.get("run_signature") == run_signature:
            print(f"[{year}/seed={seed}] 已完成，直接复用。", flush=True)
            if prediction_loader is not None and not prediction_path.exists():
                model = load_best_model(
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
                    distributed_context=distributed_context,
                )
                if primary:
                    atomic_write_parquet(predictions_frame, prediction_path)
                if distributed_context is not None:
                    dist.barrier()
            return completed["summary"]

    model = build_model(  # 创建模型并部署到设备
        device, bool(args.compile), args.compile_mode, config.train_batch_mode
    )
    if distributed_context is not None:
        model = distributed.wrap_model(model, distributed_context)
    optimizer = build_optimizer(model, config, bool(args.fused_adam), device)  # 按训练配置创建优化器
    start_epoch = 1
    best_epoch = 0
    best_validation_ic = -np.inf
    best_validation_metrics: dict[str, float] = {}
    epochs_without_improvement = 0
    history: list[dict[str, object]] = []

    if last_path.exists() and not args.overwrite:
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        if checkpoint.get("run_signature") != run_signature:
            raise ValueError(f"{last_path} 与当前配置不一致；请使用新 run name")
        load_canonical_state(model, checkpoint["model_state"])  # 恢复模型权重
        optimizer.load_state_dict(checkpoint["optimizer_state"])  # 恢复优化器续训状态
        if distributed_context is not None:
            states = checkpoint.get("rng_state_by_rank")
            if states is None or len(states) != distributed_context.world_size:
                raise ValueError("DDP checkpoint 缺少匹配 world size 的逐 rank RNG 状态")
            distributed.restore_rng_state(
                states[distributed_context.rank], device
            )
        else:
            random.setstate(checkpoint["python_random_state"])
            np.random.set_state(checkpoint["numpy_random_state"])
            torch.set_rng_state(checkpoint["torch_rng_state"])
            if device.type == "cuda" and checkpoint["cuda_rng_state"] is not None:
                torch.cuda.set_rng_state(checkpoint["cuda_rng_state"], device)
        train_loader.generator.set_state(checkpoint["train_generator_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_epoch = int(checkpoint["best_epoch"])
        best_validation_ic = float(checkpoint["best_validation_ic"])
        best_validation_metrics = {
            key: float(value)
            for key, value in checkpoint.get("best_validation_metrics", {}).items()
        }
        epochs_without_improvement = int(checkpoint["epochs_without_improvement"])
        if epochs_without_improvement >= config.early_stopping_patience:
            start_epoch = config.max_epochs + 1
        if history_path.exists():
            history = pd.read_parquet(history_path).to_dict("records")
        print(f"[{year}/seed={seed}] 从 epoch {start_epoch} 续训。", flush=True)

    started = time.perf_counter()
    stopped_early = epochs_without_improvement >= config.early_stopping_patience
    for epoch in range(start_epoch, config.max_epochs + 1):
        epoch_started = time.perf_counter()
        if hasattr(train_loader.batch_sampler, "set_epoch"):  # 同步 epoch，保持采样可复现
            train_loader.batch_sampler.set_epoch(epoch)
        else:
            set_train_epoch(train_loader, epoch)
        train_metrics = train_one_epoch(  # 完成一个训练 epoch
            model,
            train_loader,
            optimizer,
            device,
            bool(args.cuda_prefetch),
            config,
            distributed_context=distributed_context,
        )
        validation_metrics = evaluate_model(  # 无梯度计算验证指标
            model,
            validation_loader,
            device,
            bool(args.cuda_prefetch),
            config.validation_metric,
            distributed_context=distributed_context,
        )
        validation_ic = float(validation_metrics["validation_ic"])
        improved = validation_ic > (
            best_validation_ic + config.improvement_tolerance
        )
        if improved:
            best_epoch = epoch
            best_validation_ic = validation_ic
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
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

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
        if primary:  # 原子落盘，避免半成品
            atomic_write_parquet(pd.DataFrame(history), history_path)
        rng_state_by_rank = (
            distributed.gather_rng_states(device) if distributed_context else None
        )
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
        if rng_state_by_rank is not None:
            checkpoint["rng_state_by_rank"] = rng_state_by_rank
        if primary:
            atomic_save_checkpoint(checkpoint, last_path)
        if primary and improved:
            atomic_save_checkpoint(checkpoint, best_path)
        if distributed_context is not None:
            dist.barrier()
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
        if epochs_without_improvement >= config.early_stopping_patience:
            stopped_early = True
            break

    if not best_path.exists():
        raise FileNotFoundError(f"未生成最佳 checkpoint：{best_path}")
    if prediction_loader is not None:
        best_model = load_best_model(
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
            distributed_context=distributed_context,
        )
        if primary:
            atomic_write_parquet(predictions_frame, prediction_path)
        if distributed_context is not None:
            dist.barrier()
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
    if primary:  # 原子落盘，避免半成品
        atomic_write_json(
            {"run_signature": run_signature, "summary": summary}, completed_path
        )
    if distributed_context is not None:
        dist.barrier()
    return summary


def build_ensemble(run_name: str, year: int) -> int:
    """逐日标准化三个 seed 后等权平均；空 prediction 年份返回 0。"""

    model_root, prediction_root, _ = run_paths(run_name)
    split = load_sequence_split(year)
    if not len(split["prediction_indices"]):
        return 0
    seed_frames: dict[int, pd.DataFrame] = {}
    for seed in SEEDS:
        path = model_root / f"forecast_{year}" / f"seed_{seed}" / "prediction.parquet"
        seed_frames[seed] = pd.read_parquet(path)

    reference = seed_frames[SEEDS[0]]["sample_index"].to_numpy(dtype=np.int64)
    output = pd.DataFrame({"sample_index": reference})
    raw_columns: list[str] = []
    for seed in SEEDS:
        frame = seed_frames[seed]
        indices = frame["sample_index"].to_numpy(dtype=np.int64)
        if not np.array_equal(reference, indices):
            raise AssertionError(f"{year} 年三个 seed 的预测行号不一致")
        column = f"prediction_seed_{seed}"
        output[column] = frame["prediction"].to_numpy(dtype=np.float32)
        raw_columns.append(column)

    metadata = pd.read_parquet(
        SAMPLE_INDEX_PATH,
        columns=["sample_index", "trade_date", "instrument_id"],
    ).set_index("sample_index")
    output = metadata.loc[reference].reset_index().merge(  # 按业务键对齐，避免依赖原始行序
        output, on="sample_index", validate="one_to_one"
    )
    raw = output[raw_columns].astype(np.float64)
    grouped = raw.groupby(output["trade_date"], sort=False)
    means = grouped.transform("mean")  # 按约定截面口径汇总
    standard_deviations = grouped.transform("std", ddof=0)  # 按约定截面口径汇总
    if (standard_deviations <= 0).any().any():
        raise ValueError(f"{year} 年存在常数 seed 预测")
    zscore_columns: list[str] = []
    for column in raw_columns:
        zscore_column = f"{column}_zscore"
        output[zscore_column] = (
            (raw[column] - means[column]) / standard_deviations[column]
        ).to_numpy(dtype=np.float32)
        zscore_columns.append(zscore_column)
    output["prediction_ensemble_raw"] = raw.mean(axis=1).to_numpy(dtype=np.float32)
    output["prediction_ensemble"] = output[zscore_columns].mean(axis=1).astype(
        np.float32
    )
    output["forecast_year"] = year
    output["factor_role"] = "full_year_oos"
    atomic_write_parquet(  # 原子落盘，避免半成品
        output,
        prediction_root / f"vivit_predictions.{year}.parquet",
    )
    return len(output)


def finalize_run(args: argparse.Namespace, wall_seconds: float) -> pd.DataFrame:
    """核验年度任务并生成集成预测、诊断、清单和报告。"""
    model_root, _, quality_root = run_paths(args.run_name)
    records: list[dict[str, object]] = []
    for year in FORECAST_YEARS:
        for seed in SEEDS:
            path = model_root / f"forecast_{year}" / f"seed_{seed}" / "completed.json"
            if not path.exists():
                raise FileNotFoundError(f"缺少已完成任务：{path}")
            records.append(json.loads(path.read_text(encoding="utf-8"))["summary"])
    summary = pd.DataFrame(records).sort_values(
        ["forecast_year", "seed"], kind="stable"
    )
    atomic_write_parquet(summary, quality_root / "vivit_training_summary.parquet")  # 原子落盘，避免半成品
    factor_counts = {year: build_ensemble(args.run_name, year) for year in FORECAST_YEARS}
    sequence_manifest = require_aligned_sequence_store()
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
    atomic_write_json(manifest, quality_root / "a2_training_manifest.json")  # 原子落盘，避免半成品
    return summary


def worker_command(args: argparse.Namespace, year: int, seed: int, gpu: int) -> list[str]:
    """构造单个年度×seed worker 的命令行参数。"""
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


def ddp_worker_command(
    args: argparse.Namespace,
    year: int,
    seed: int,
    device_ids: tuple[int, ...],
) -> list[str]:
    """执行 ddp_worker_command 对应的分布式训练处理。"""
    command = worker_command(args, year, seed, device_ids[0])
    command[2] = "ddp-worker"
    command.extend(["--device-ids", ",".join(str(value) for value in device_ids)])
    return command


def _baseline_ddp_smoke(
    args: argparse.Namespace,
    context: distributed.DistributedContext,
) -> dict[str, object]:
    """执行 _baseline_ddp_smoke 对应的分布式训练处理。"""
    config = training_config_from_args(args)
    validate_training_config(config, args.compile_mode)
    set_random_seed(args.seed + context.rank)
    train_loader, validation_loader, _, batch_summary = make_distributed_data_loaders(
        args.forecast_year,
        args.seed,
        config,
        args.workers_per_task,
        context.device.type == "cuda",
        context,
    )
    train_loader.batch_sampler.set_epoch(1)
    model = build_model(
        context.device,
        bool(args.compile),
        args.compile_mode,
        config.train_batch_mode,
    )
    model = distributed.wrap_model(model, context)
    optimizer = build_optimizer(
        model, config, bool(args.fused_adam), context.device
    )
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
    validation_metrics = evaluate_model(
        model,
        validation_loader,
        context.device,
        bool(args.cuda_prefetch),
        config.validation_metric,
        max_batches=args.max_batches,
        distributed_context=context,
    )
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


def _baseline_ddp_rank(
    rank: int,
    world_size: int,
    args: argparse.Namespace,
    smoke: bool,
) -> None:
    """执行 _baseline_ddp_rank 对应的分布式训练处理。"""
    context = distributed.initialize_process_group(rank, world_size, backend="nccl")
    try:
        args.execution_route = "ddp"
        args.ddp_world_size = world_size
        if smoke:
            result = _baseline_ddp_smoke(args, context)
        else:
            result = train_single_run(args, distributed_context=context)
        if context.is_primary:
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    finally:
        distributed.destroy_process_group()


def run_ddp_worker(args: argparse.Namespace, *, smoke: bool = False) -> None:
    """执行 run_ddp_worker 对应的分布式训练处理。"""
    devices = distributed.parse_device_ids(args.device_ids)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(value) for value in devices)
    if not torch.cuda.is_available() or torch.cuda.device_count() != len(devices):
        raise RuntimeError(
            f"DDP 请求 {len(devices)} 张 GPU，但当前可见 {torch.cuda.device_count()} 张"
        )
    args.ddp_world_size = len(devices)
    args.execution_route = "ddp"
    if smoke:
        args.worker_year = args.forecast_year
        args.worker_seed = args.seed
    args.worker_physical_gpu = devices[0]
    distributed.configure_master(distributed.reserve_local_port())
    mp.spawn(
        _baseline_ddp_rank,
        args=(len(devices), args, smoke),
        nprocs=len(devices),
        join=True,
    )


def run_ddp(args: argparse.Namespace) -> pd.DataFrame:
    """执行 run_ddp 对应的分布式训练处理。"""

    devices = distributed.parse_device_ids(args.device_ids)
    args.execution_route = "ddp"
    args.ddp_world_size = len(devices)
    _, _, quality_root = run_paths(args.run_name)
    log_dir = quality_root / "ddp_task_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print(
        f"ViViT DDP start: tasks={len(FORECAST_YEARS) * len(SEEDS)}, "
        f"GPUs={devices}, world_size={len(devices)}",
        flush=True,
    )
    for year in FORECAST_YEARS:
        for seed in SEEDS:
            log_path = log_dir / f"forecast_{year}.seed_{seed}.log"
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    f"\n=== launch {year}/seed={seed} at "
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
                )
                handle.flush()
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = ",".join(
                    str(value) for value in devices
                )
                process = subprocess.Popen(
                    ddp_worker_command(args, year, seed, devices),
                    cwd=VIVIT_DIR,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                code = process.wait()
            if code:
                raise RuntimeError(
                    f"DDP 任务失败：{year}/seed={seed}\n{tail(log_path, 100)}"
                )
            print(f"done DDP: {year}/seed={seed}", flush=True)
    result = finalize_run(args, time.perf_counter() - started)
    print(validate_outputs(args.run_name).to_string(index=False), flush=True)
    print(f"实验报告：{write_experiment_report(args.run_name)}", flush=True)
    return result


def tail(path: Path, line_count: int = 80) -> str:
    """读取文本文件末尾若干行，用于展示最新训练日志。"""
    if not path.exists():
        return ""
    return "\n".join(
        path.read_text(encoding="utf-8", errors="replace").splitlines()[-line_count:]
    )


def run_a2(args: argparse.Namespace) -> pd.DataFrame:
    """在四张 GPU 上并发调度 24 个独立任务。"""

    args.execution_route = "a2"
    args.ddp_world_size = 1
    if not SEQUENCE_MANIFEST_PATH.exists():
        build_sequence_store(overwrite=False)
    require_aligned_sequence_store()
    validate_training_config(training_config_from_args(args), args.compile_mode)
    device_ids = tuple(int(value) for value in args.device_ids.split(","))
    if len(set(device_ids)) != len(device_ids):
        raise ValueError("GPU 编号不能重复")
    tasks = deque((year, seed) for year in FORECAST_YEARS for seed in SEEDS)
    _, _, quality_root = run_paths(args.run_name)
    log_dir = quality_root / "a2_task_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    running: list[dict[str, object]] = []
    started = time.perf_counter()
    print(
        f"ViViT A2 start: tasks={len(tasks)}, GPUs={device_ids}, "
        f"tasks_per_gpu={args.tasks_per_gpu}",
        flush=True,
    )
    try:
        while tasks or running:
            for gpu in device_ids:
                active = sum(1 for item in running if item["gpu"] == gpu)
                while tasks and active < args.tasks_per_gpu:
                    year, seed = tasks.popleft()
                    log_path = log_dir / f"forecast_{year}.seed_{seed}.log"
                    log_handle = log_path.open("a", encoding="utf-8")
                    log_handle.write(  # 追加任务状态日志
                        f"\n=== launch {year}/seed={seed} at "
                        f"{time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
                    )
                    log_handle.flush()  # 将 memmap 修改刷入磁盘
                    environment = os.environ.copy()
                    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
                    process = subprocess.Popen(  # 启动独立训练任务
                        worker_command(args, year, seed, gpu),
                        cwd=VIVIT_DIR,
                        env=environment,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
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
                    active += 1
                    print(
                        f"launch GPU{gpu}: {year}/seed={seed} pid={process.pid}",
                        flush=True,
                    )

            finished = []
            for item in running:
                return_code = item["process"].poll()  # 检查训练任务状态
                if return_code is None:
                    continue
                item["log_handle"].close()
                finished.append(item)
                if return_code:
                    raise RuntimeError(
                        f"任务失败：{item['year']}/seed={item['seed']} GPU{item['gpu']}\n"
                        f"{tail(item['log_path'])}"
                    )
                minutes = (time.perf_counter() - item["started"]) / 60
                print(
                    f"done GPU{item['gpu']}: {item['year']}/seed={item['seed']} "
                    f"{minutes:.2f} min",
                    flush=True,
                )
            for item in finished:
                running.remove(item)  # 移除已结束任务
            if running and not finished:
                time.sleep(1)
    except BaseException:
        for item in running:
            if item["process"].poll() is None:
                item["process"].terminate()  # 终止仍在运行的任务
            item["log_handle"].close()
        raise

    summary = finalize_run(args, time.perf_counter() - started)  # 汇总任务并写出完成标记
    validate_outputs(args.run_name)
    write_experiment_report(args.run_name)
    return summary


def markdown_table(frame: pd.DataFrame) -> str:
    """把数据表转换为适合实验报告的 Markdown 表格。"""
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def write_experiment_report(run_name: str) -> Path:
    """根据当前 run 的真实产物生成中文报告，不覆盖历史根目录报告。"""

    _, _, quality_root = run_paths(run_name)
    training = pd.read_parquet(quality_root / "vivit_training_summary.parquet")
    factor_audit = validate_outputs(run_name)
    sequence_summary = pd.read_parquet(SEQUENCE_SUMMARY_PATH)
    run_manifest = json.loads(
        (quality_root / "a2_training_manifest.json").read_text(encoding="utf-8")
    )
    dataset_manifest, _ = source_manifests()  # 读取或生成来源清单
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
    atomic_write_text(report, path)  # 原子落盘，避免半成品
    return path


def run_smoke_test(args: argparse.Namespace) -> dict[str, object]:
    """按指定生产配置执行真实数据有限 batch 前向、反向与验证。"""

    if not SEQUENCE_MANIFEST_PATH.exists():
        build_sequence_store(overwrite=False)
    require_aligned_sequence_store()
    config = training_config_from_args(args)
    validate_training_config(config, args.compile_mode)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_random_seed(args.seed)  # 固定各随机源
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
    set_train_epoch(train_loader, 1)  # 同步 epoch，保持采样可复现
    audit_batch = next(iter(train_loader))
    model = build_model(  # 创建模型并部署到设备
        device, bool(args.compile), args.compile_mode, config.train_batch_mode
    )
    optimizer = build_optimizer(model, config, bool(args.fused_adam), device)  # 按训练配置创建优化器
    train_metrics = train_one_epoch(  # 完成一个训练 epoch
        model,
        train_loader,
        optimizer,
        device,
        bool(args.cuda_prefetch),
        config,
        max_batches=args.max_batches,
    )
    validation_metrics = evaluate_model(  # 无梯度计算验证指标
        model,
        validation_loader,
        device,
        bool(args.cuda_prefetch),
        config.validation_metric,
        max_batches=args.max_batches,
    )
    prediction_rows = 0
    if prediction_loader is not None:
        prediction_rows = len(
            predict_model(
                model,
                prediction_loader,
                device,
                bool(args.cuda_prefetch),
                max_batches=1,
            )
        )
    frame_mask = audit_batch["frame_mask"]
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


def run_memory_benchmark(args: argparse.Namespace) -> dict[str, object]:
    """用一个指定配置的真实 batch 测量单任务峰值显存和耗时。"""

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("显存 benchmark 要求 CUDA")
    require_aligned_sequence_store()
    config = training_config_from_args(args)
    validate_training_config(config, args.compile_mode)
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
    set_train_epoch(loader, 1)  # 同步 epoch，保持采样可复现
    model = build_model(  # 创建模型并部署到设备
        device, bool(args.compile), args.compile_mode, config.train_batch_mode
    )
    optimizer = build_optimizer(model, config, bool(args.fused_adam), device)  # 按训练配置创建优化器
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    metrics = train_one_epoch(  # 完成一个训练 epoch
        model,
        loader,
        optimizer,
        device,
        bool(args.cuda_prefetch),
        config,
        max_batches=1,
    )
    return {
        "forecast_year": args.forecast_year,
        "train_batch_mode": config.train_batch_mode,
        "loss_name": config.loss_name,
        "seconds": time.perf_counter() - started,
        "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        **metrics,
    }


def validate_training_config(config: TrainingConfig, compile_mode: str) -> None:
    """校验训练模式、损失、早停与资源参数组合。"""
    if config.train_batch_mode not in TRAIN_BATCH_MODES:
        raise ValueError(f"未知训练 batch 模式：{config.train_batch_mode}")
    if config.loss_name not in LOSS_NAMES:
        raise ValueError(f"未知损失函数：{config.loss_name}")
    if config.validation_metric not in VALIDATION_METRICS:
        raise ValueError(f"未知验证指标：{config.validation_metric}")
    if config.loss_name == TEMPORAL_TURNOVER_LOSS and config.train_batch_mode != "trade_date":
        raise ValueError("sigmoid_rank_ic_turnover 必须使用 trade_date batch")
    if config.loss_name == TEMPORAL_TURNOVER_LOSS and compile_mode == "reduce-overhead":
        raise ValueError("时序换手损失禁止使用 reduce-overhead compile mode")
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
        raise ValueError("训练、batch 或损失超参数非法")

def validate_outputs(run_name: str) -> pd.DataFrame:
    """核验 24 个任务与 8 个年度因子文件。"""

    model_root, prediction_root, quality_root = run_paths(run_name)
    summary_path = quality_root / "vivit_training_summary.parquet"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    training_summary = pd.read_parquet(summary_path)
    if len(training_summary) != len(FORECAST_YEARS) * len(SEEDS):
        raise AssertionError("训练汇总不是 8 年度 × 3 种子")

    records = []
    for year in FORECAST_YEARS:
        split = load_sequence_split(year)
        expected = len(split["prediction_indices"])
        path = prediction_root / f"vivit_predictions.{year}.parquet"
        if expected == 0:
            if path.exists():
                raise AssertionError(f"{year} 年 prediction 为空但存在因子文件")
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
            continue
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path)
        if len(frame) != expected or frame["sample_index"].duplicated().any():
            raise AssertionError(f"{year} 年因子行数或主键错误")
        if not np.array_equal(
            frame["sample_index"].to_numpy(np.int64), split["prediction_indices"]
        ):
            raise AssertionError(f"{year} 年因子索引与 ViT prediction split 不一致")
        if not np.isfinite(frame["prediction_ensemble"]).all():
            raise AssertionError(f"{year} 年因子包含非有限值")
        for seed in SEEDS:
            column = f"prediction_seed_{seed}_zscore"
            grouped = frame.groupby("trade_date", sort=False)[column]
            max_mean = float(grouped.mean().abs().max())
            max_std_error = float((grouped.std(ddof=0) - 1).abs().max())
            if max_mean > 1e-5 or max_std_error > 1e-5:
                raise AssertionError(f"{year} 年 seed={seed} 日截面标准化失败")
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
        raise AssertionError("本任务不应生成 delivery 目录")
    if (DATA_DIR / "backtest").exists():
        raise AssertionError("本任务不应生成 backtest 目录")
    return pd.DataFrame(records)

def add_training_arguments(parser: argparse.ArgumentParser) -> None:
    """向命令行解析器注册训练后端参数。"""
    parser.add_argument(
        "--run-name", default="vivit_aligned_vit_indices_pearson_date_batch"
    )
    parser.add_argument(
        "--train-batch-mode", choices=TRAIN_BATCH_MODES, default="trade_date"
    )
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--inference-batch-size", type=int, default=5000)
    parser.add_argument("--loss-name", choices=LOSS_NAMES, default="pearson_ic")
    parser.add_argument("--rank-temperature", type=float, default=0.1)
    parser.add_argument("--turnover-lambda", type=float, default=0.1)
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--smooth-l1-beta", type=float, default=0.1)
    parser.add_argument("--hybrid-weight", type=float, default=0.5)
    parser.add_argument(
        "--validation-metric",
        choices=VALIDATION_METRICS,
        default="daily_mean_pearson_ic",
    )
    parser.add_argument("--workers-per-task", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--compile", type=int, choices=(0, 1), default=1)
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--fused-adam", type=int, choices=(0, 1), default=1)
    parser.add_argument("--cuda-prefetch", type=int, choices=(0, 1), default=1)
    parser.add_argument("--overwrite", type=int, choices=(0, 1), default=0)


def add_trial_arguments(parser: argparse.ArgumentParser) -> None:
    """向命令行解析器注册单任务试运行参数。"""
    parser.add_argument("--forecast-year", type=int, default=2018)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-batches", type=int, default=1)


def parse_args() -> argparse.Namespace:
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
    worker_parser.add_argument("--worker-year", type=int, required=True)
    worker_parser.add_argument("--worker-seed", type=int, required=True)
    worker_parser.add_argument("--worker-physical-gpu", type=int, required=True)

    a2_parser = subparsers.add_parser("a2")
    add_training_arguments(a2_parser)
    a2_parser.add_argument("--device-ids", default="0,1,2,3")
    a2_parser.add_argument("--tasks-per-gpu", type=int, default=1)

    ddp_parser = subparsers.add_parser("ddp")
    add_training_arguments(ddp_parser)
    ddp_parser.add_argument("--device-ids", default="0,1,2,3")

    ddp_worker_parser = subparsers.add_parser("ddp-worker")
    add_training_arguments(ddp_worker_parser)
    ddp_worker_parser.add_argument("--worker-year", type=int, required=True)
    ddp_worker_parser.add_argument("--worker-seed", type=int, required=True)
    ddp_worker_parser.add_argument("--worker-physical-gpu", type=int, required=True)
    ddp_worker_parser.add_argument("--device-ids", required=True)

    ddp_smoke_parser = subparsers.add_parser("ddp-smoke")
    add_training_arguments(ddp_smoke_parser)
    add_trial_arguments(ddp_smoke_parser)
    ddp_smoke_parser.add_argument("--device-ids", default="0,1,2,3")
    ddp_smoke_parser.set_defaults(compile=0, workers_per_task=0)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--run-name", required=True)
    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--run-name", required=True)
    return parser.parse_args()


def main() -> None:
    """根据命令行子命令分派基线训练、验证或报告流程。"""
    args = parse_args()
    if args.command == "smoke":
        print(json.dumps(run_smoke_test(args), ensure_ascii=False, indent=2))
    elif args.command == "benchmark":
        print(json.dumps(run_memory_benchmark(args), ensure_ascii=False, indent=2))
    elif args.command == "worker":
        args.execution_route = "a2"
        args.ddp_world_size = 1
        print(json.dumps(train_single_run(args), ensure_ascii=False, indent=2))
    elif args.command == "ddp-worker":
        run_ddp_worker(args)
    elif args.command == "ddp-smoke":
        run_ddp_worker(args, smoke=True)
    elif args.command == "a2":
        print(run_a2(args).to_string(index=False))
    elif args.command == "ddp":
        print(run_ddp(args).to_string(index=False))
    elif args.command == "validate":
        print(validate_outputs(args.run_name).to_string(index=False))
    elif args.command == "report":
        print(write_experiment_report(args.run_name))


if __name__ == "__main__":
    main()
