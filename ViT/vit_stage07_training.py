"""阶段 07：执行 ViT DataLoader、年度滚动训练、三种子集成与任务调度。"""
from __future__ import annotations

from typing import Any

import argparse
import contextlib
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from IPython.display import display
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

import vit_stage06_ranking as ranking
from vit_stage01_foundation import (
    VIT_DIR, atomic_json, atomic_parquet, get_data_root,
    read_bool_environment, read_json, stable_hash,
)
from vit_stage05_model import OrderFlowVisionTransformer, VitModelConfig


PIPELINE_PATH = VIT_DIR / "vit_stage07_training.py"
BASELINE_A2_RUN_NAME = "baseline"
DEFAULT_A2_RUN_NAME = "loss_tuning"
DEFAULT_TUNING_RUN_NAME = DEFAULT_A2_RUN_NAME
# 构造去重集合，并保存到 `READ_ONLY_TRAINING_RUN_NAMES`。
READ_ONLY_TRAINING_RUN_NAMES = {BASELINE_A2_RUN_NAME, "date_batch"}
DEFAULT_YEARS = "2018,2019,2020,2021,2022,2023,2024,2025"
DEFAULT_SEEDS = "42,43,44"
TRAIN_BATCH_MODES = ("fixed_samples", "trade_date")
TEMPORAL_TURNOVER_LOSS = ranking.TEMPORAL_TURNOVER_LOSS
LOSS_NAMES = ranking.LOSS_NAMES
STANDARD_RANK_LOSS_NAMES = ranking.STANDARD_RANK_LOSS_NAMES
RANK_LOSS_NAMES = (*STANDARD_RANK_LOSS_NAMES, TEMPORAL_TURNOVER_LOSS)
VALIDATION_METRICS = (
    "daily_mean_spearman_ic",
    "daily_mean_pearson_ic",
)

pearson_correlation = ranking.pearson_correlation
negative_pearson_ic = ranking.negative_pearson_ic
sigmoid_soft_rank = ranking.sigmoid_soft_rank
neural_sort_soft_rank = ranking.neural_sort_soft_rank
differentiable_rank_ic = ranking.differentiable_rank_ic
temporal_turnover_loss_components = (
    ranking.temporal_turnover_loss_components
)


# 使用 `dataclass(frozen=True)` 包装 `TrainingConfig`，并定义其可复用的处理逻辑。
@dataclass(frozen=True)
class TrainingConfig:
    train_batch_mode: str
    batch_size: int
    inference_batch_size: int
    loss_name: str
    rank_temperature: float
    turnover_lambda: float
    huber_delta: float
    smooth_l1_beta: float
    hybrid_weight: float
    validation_metric: str
    learning_rate: float
    max_epochs: int
    early_stopping_patience: int
    seeds: tuple[int, ...]
    forecast_years: tuple[int, ...]
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.0
    improvement_tolerance: float = 1e-6


def parse_csv_ints(value: str) -> tuple[int, ...]:
    """解析逗号分隔整数列表并返回去重后的顺序结果。"""
    result = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not result or len(set(result)) != len(result):
        raise ValueError(f"整数列表为空或重复：{value}")
    return result


def bool_int(value: int) -> bool:
    """把命令行中的 0/1 参数转换为布尔值。"""
    return bool(int(value))


def configure_worker_environment(args: Any) -> None:
    """把 CLI 参数发布为单个 A2 worker 的运行参数。"""

    os.environ.update(
        {
            "VIT_DATA_ROOT": str(Path(args.data_root).resolve()),
            "VIT_TRAIN_RUN_NAME": args.run_name,
            "VIT_TRAIN_BATCH_MODE": args.train_batch_mode,
            "VIT_GLOBAL_BATCH_SIZE": str(args.batch_size),
            "VIT_INFERENCE_BATCH_SIZE": str(args.inference_batch_size),
            "VIT_LOSS_NAME": args.loss_name,
            "VIT_RANK_TEMPERATURE": str(args.rank_temperature),
            "VIT_TURNOVER_LAMBDA": str(args.turnover_lambda),
            "VIT_HUBER_DELTA": str(args.huber_delta),
            "VIT_SMOOTH_L1_BETA": str(args.smooth_l1_beta),
            "VIT_HYBRID_WEIGHT": str(args.hybrid_weight),
            "VIT_VALIDATION_METRIC": args.validation_metric,
            "VIT_NUM_WORKERS": str(args.workers_per_task),
            "VIT_LEARNING_RATE": str(args.learning_rate),
            "VIT_MAX_EPOCHS": str(args.max_epochs),
            "VIT_EARLY_STOPPING_PATIENCE": str(args.patience),
            "VIT_FORECAST_YEARS": args.years,
            "VIT_TRAIN_SEEDS": args.seeds,
            "VIT_USE_TORCH_COMPILE": str(args.compile),
            "VIT_TORCH_COMPILE_MODE": args.compile_mode,
            "VIT_USE_FUSED_ADAM": str(args.fused_adam),
            "VIT_USE_CUDA_PREFETCH": str(args.cuda_prefetch),
            "VIT_TRAIN_OVERWRITE": str(args.overwrite),
            "VIT_A2_WORKER_PROCESS": "1",
        }
    )


def initialize_runtime() -> None:
    """初始化一个独立单 GPU 训练任务的全局上下文。"""

    global DATA_ROOT, STORE_DIR, FEATURES_PATH, TARGETS_PATH, LABEL_VALID_PATH
    global DATE_IDS_PATH, SAMPLE_INDEX_PATH, DATASET_MANIFEST_PATH
    global ROLLING_MANIFEST_PATH, ROLLING_SUMMARY_PATH, SPLIT_DIR
    global RUN_NAME, MODEL_ROOT, PREDICTION_DIR, QUALITY_DIR, TRAINING_SUMMARY_PATH
    global MODEL_CONFIG, TRAINING_CONFIG, NUM_WORKERS, DEVICE, PIN_MEMORY
    global USE_BF16, USE_COMPILE, COMPILE_MODE, USE_FUSED_ADAM, USE_CUDA_PREFETCH
    global OVERWRITE_TRAINING, PARALLEL_BACKEND, dataset_manifest, rolling_manifest
    global sample_index, targets_memmap, VISIBLE_GPU_COUNT

    DATA_ROOT = get_data_root().resolve()
    STORE_DIR = DATA_ROOT / "feature_store"
    FEATURES_PATH = STORE_DIR / "features.npy"
    TARGETS_PATH = STORE_DIR / "targets.npy"
    LABEL_VALID_PATH = STORE_DIR / "label_valid.npy"
    DATE_IDS_PATH = STORE_DIR / "date_ids.npy"
    SAMPLE_INDEX_PATH = STORE_DIR / "sample_index.parquet"
    DATASET_MANIFEST_PATH = STORE_DIR / "dataset_manifest.json"
    ROLLING_MANIFEST_PATH = STORE_DIR / "rolling_split_manifest.json"
    ROLLING_SUMMARY_PATH = STORE_DIR / "rolling_split_summary.parquet"
    SPLIT_DIR = STORE_DIR / "rolling_splits"
    RUN_NAME = os.getenv("VIT_TRAIN_RUN_NAME", DEFAULT_A2_RUN_NAME).strip()
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    if not RUN_NAME or RUN_NAME in {".", ".."} or any(x not in allowed for x in RUN_NAME):
        raise ValueError("训练运行名含非法字符")
    MODEL_ROOT = VIT_DIR / "models" / "runs" / RUN_NAME
    PREDICTION_DIR = DATA_ROOT / "predictions" / "runs" / RUN_NAME
    QUALITY_DIR = DATA_ROOT / "quality" / "runs" / RUN_NAME
    TRAINING_SUMMARY_PATH = QUALITY_DIR / "vit_training_summary.parquet"
    for directory in (MODEL_ROOT, PREDICTION_DIR, QUALITY_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    MODEL_CONFIG = VitModelConfig()
    TRAINING_CONFIG = TrainingConfig(
        train_batch_mode=os.getenv(
            "VIT_TRAIN_BATCH_MODE", "trade_date"
        ).strip(),
        batch_size=int(os.getenv("VIT_GLOBAL_BATCH_SIZE", "5000")),
        inference_batch_size=int(
            os.getenv("VIT_INFERENCE_BATCH_SIZE", "5000")
        ),
        loss_name=os.getenv("VIT_LOSS_NAME", "pearson_ic").strip(),
        rank_temperature=float(os.getenv("VIT_RANK_TEMPERATURE", "0.1")),
        turnover_lambda=float(os.getenv("VIT_TURNOVER_LAMBDA", "0.1")),
        huber_delta=float(os.getenv("VIT_HUBER_DELTA", "1.0")),
        smooth_l1_beta=float(os.getenv("VIT_SMOOTH_L1_BETA", "0.1")),
        hybrid_weight=float(os.getenv("VIT_HYBRID_WEIGHT", "0.5")),
        validation_metric=os.getenv(
            "VIT_VALIDATION_METRIC", "daily_mean_spearman_ic"
        ).strip(),
        learning_rate=float(os.getenv("VIT_LEARNING_RATE", "0.001")),
        max_epochs=int(os.getenv("VIT_MAX_EPOCHS", "100")),
        early_stopping_patience=int(
            os.getenv("VIT_EARLY_STOPPING_PATIENCE", "10")
        ),
        seeds=parse_csv_ints(os.getenv("VIT_TRAIN_SEEDS", "42,43,44")),
        forecast_years=parse_csv_ints(
            os.getenv(
                "VIT_FORECAST_YEARS", "2018,2019,2020,2021,2022,2023,2024,2025"
            )
        ),
    )
    if TRAINING_CONFIG.train_batch_mode not in TRAIN_BATCH_MODES:
        raise ValueError(
            f"未知训练 batch 模式：{TRAINING_CONFIG.train_batch_mode}"
        )
    if TRAINING_CONFIG.loss_name not in LOSS_NAMES:
        raise ValueError(f"未知损失函数：{TRAINING_CONFIG.loss_name}")
    if TRAINING_CONFIG.validation_metric not in VALIDATION_METRICS:
        raise ValueError(
            f"未知验证指标：{TRAINING_CONFIG.validation_metric}"
        )
    if (
        TRAINING_CONFIG.rank_temperature <= 0
        or TRAINING_CONFIG.turnover_lambda < 0
        or TRAINING_CONFIG.huber_delta <= 0
        or TRAINING_CONFIG.smooth_l1_beta <= 0
        or not 0 <= TRAINING_CONFIG.hybrid_weight <= 1
    ):
        raise ValueError("损失温度、换手权重、delta、beta 或混合权重非法")
    if (
        TRAINING_CONFIG.loss_name == TEMPORAL_TURNOVER_LOSS
        and TRAINING_CONFIG.train_batch_mode != "trade_date"
    ):
        raise ValueError(
            "sigmoid_rank_ic_turnover 必须使用 trade_date batch"
        )
    if (
        TRAINING_CONFIG.batch_size <= 1
        or TRAINING_CONFIG.inference_batch_size <= 1
        or TRAINING_CONFIG.learning_rate <= 0
    ):
        raise ValueError("batch_size、inference_batch_size 和 learning_rate 必须为正")
    if (
        TRAINING_CONFIG.max_epochs < 1
        or TRAINING_CONFIG.early_stopping_patience < 1
    ):
        raise ValueError("max_epochs 和 patience 必须为正")
    NUM_WORKERS = int(os.getenv("VIT_NUM_WORKERS", "2"))
    if NUM_WORKERS < 0:
        raise ValueError("workers 不能为负数")
    USE_COMPILE = read_bool_environment("VIT_USE_TORCH_COMPILE", True)
    COMPILE_MODE = os.getenv("VIT_TORCH_COMPILE_MODE", "default")
    if (
        TRAINING_CONFIG.loss_name == TEMPORAL_TURNOVER_LOSS
        and USE_COMPILE
        and COMPILE_MODE == "reduce-overhead"
    ):
        raise ValueError(
            "时序换手损失的动态日截面禁止使用 reduce-overhead；"
            "请改用 compile mode=default"
        )
    USE_FUSED_ADAM = read_bool_environment("VIT_USE_FUSED_ADAM", True)
    USE_CUDA_PREFETCH = read_bool_environment("VIT_USE_CUDA_PREFETCH", True)
    OVERWRITE_TRAINING = read_bool_environment("VIT_TRAIN_OVERWRITE", False)
    VISIBLE_GPU_COUNT = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda:0")
    else:
        DEVICE = torch.device("cpu")
    PIN_MEMORY = DEVICE.type == "cuda"
    USE_BF16 = bool(DEVICE.type == "cuda" and torch.cuda.is_bf16_supported())
    PARALLEL_BACKEND = "A2_independent_task_worker"

    required = [
        FEATURES_PATH, TARGETS_PATH, LABEL_VALID_PATH, DATE_IDS_PATH,
        SAMPLE_INDEX_PATH, DATASET_MANIFEST_PATH, ROLLING_MANIFEST_PATH,
        ROLLING_SUMMARY_PATH,
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"缺少训练输入：{missing}")
    dataset_manifest = read_json(DATASET_MANIFEST_PATH)
    rolling_manifest = read_json(ROLLING_MANIFEST_PATH)
    sample_index = pd.read_parquet(SAMPLE_INDEX_PATH)
    sample_index["trade_date"] = sample_index["trade_date"].astype(str)
    sample_index["exit_trade_date"] = sample_index["exit_trade_date"].astype(str)
    targets_memmap = np.load(TARGETS_PATH, mmap_mode="r")

    print(f"Python/PyTorch：{platform.python_version()} / {torch.__version__}")
    print(f"运行名：{RUN_NAME}")
    print(f"设备：{DEVICE}；可见 GPU：{VISIBLE_GPU_COUNT}")
    print(
        f"训练 batch 模式：{TRAINING_CONFIG.train_batch_mode}；"
        f"固定/预测 batch：{TRAINING_CONFIG.batch_size}/"
        f"{TRAINING_CONFIG.inference_batch_size}；workers/lr："
        f"{NUM_WORKERS}/{TRAINING_CONFIG.learning_rate}"
    )
    temporal_mode = TRAINING_CONFIG.loss_name == TEMPORAL_TURNOVER_LOSS
    print(
        f"损失函数：{TRAINING_CONFIG.loss_name}；"
        f"tau/turnover_lambda/delta/beta/hybrid_weight："
        f"{TRAINING_CONFIG.rank_temperature}/"
        f"{TRAINING_CONFIG.turnover_lambda}/{TRAINING_CONFIG.huber_delta}/"
        f"{TRAINING_CONFIG.smooth_l1_beta}/{TRAINING_CONFIG.hybrid_weight}"
    )
    print(
        f"验证选模指标：{TRAINING_CONFIG.validation_metric}；"
        f"reshuffle：{not temporal_mode}；梯度更新："
        f"{'每 epoch 一次' if temporal_mode else '每 batch 一次'}"
    )
    print(
        f"compile/mode/fused Adam/CUDA prefetch："
        f"{USE_COMPILE}/{COMPILE_MODE}/{USE_FUSED_ADAM}/{USE_CUDA_PREFETCH}"
    )


class SampleIndexDataset(Dataset):
    def __init__(self, indices: np.ndarray) -> None:
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        """返回 SampleIndexDataset 管理的样本数或批次数。"""
        return len(self.indices)

    def __getitem__(self, position: int) -> int:
        """按给定索引返回 SampleIndexDataset 中的一个样本。"""
        return int(self.indices[position])


class TradeDateBatchSampler(Sampler[list[int]]):
    """把完整交易日作为 batch；普通模式重排日期，时序模式保持升序。"""

    def __init__(
        self, sample_indices: np.ndarray, date_ids: np.ndarray, seed: int,
        shuffle: bool = True, instrument_ids: np.ndarray | None = None,
    ) -> None:
        indices = np.asarray(sample_indices, dtype=np.int64)
        if not len(indices):
            raise ValueError("日截面 batch 不能使用空训练索引")
        selected_dates = np.asarray(date_ids[indices], dtype=np.int64)
        if np.any(selected_dates[1:] < selected_dates[:-1]):
            raise ValueError("训练索引未按交易日排序，无法构建日截面 batch")
        boundaries = np.flatnonzero(selected_dates[1:] != selected_dates[:-1]) + 1
        self.starts = np.r_[0, boundaries].astype(np.int64)
        self.stops = np.r_[boundaries, len(indices)].astype(np.int64)
        self.date_ids = selected_dates[self.starts]
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0
        self.temporal_pairs: list[dict[str, object]] | None = None
        if instrument_ids is not None:
            if self.shuffle:
                raise ValueError("时序换手损失必须关闭交易日 reshuffle")
            if len(self.starts) < 2 or np.any(np.diff(self.date_ids) != 1):
                raise ValueError("时序换手损失要求至少两个连续训练交易日")
            all_instruments = np.asarray(instrument_ids)
            self.temporal_pairs = []
            for pair_number in range(1, len(self.starts)):
                previous_slice = slice(
                    int(self.starts[pair_number - 1]),
                    int(self.stops[pair_number - 1]),
                )
                current_slice = slice(
                    int(self.starts[pair_number]), int(self.stops[pair_number])
                )
                previous_ids = all_instruments[indices[previous_slice]]
                current_ids = all_instruments[indices[current_slice]]
                if len(np.unique(previous_ids)) != len(previous_ids):
                    raise ValueError(
                        f"date_id={self.date_ids[pair_number - 1]} 股票键重复"
                    )
                if len(np.unique(current_ids)) != len(current_ids):
                    raise ValueError(
                        f"date_id={self.date_ids[pair_number]} 股票键重复"
                    )
                common, previous_positions, current_positions = np.intersect1d(
                    previous_ids, current_ids, assume_unique=True,
                    return_indices=True,
                )
                if len(common) < 2:
                    raise ValueError(
                        f"date_id={self.date_ids[pair_number - 1]}→"
                        f"{self.date_ids[pair_number]} 共同股票少于2只"
                    )
                self.temporal_pairs.append(
                    {
                        "previous_positions": previous_positions.astype(
                            np.int64, copy=False
                        ),
                        "current_positions": current_positions.astype(
                            np.int64, copy=False
                        ),
                        "common_count": int(len(common)),
                        "coverage": float(
                            len(common) / min(len(previous_ids), len(current_ids))
                        ),
                    }
                )

    def __len__(self) -> int:
        """返回 TradeDateBatchSampler 管理的样本数或批次数。"""
        return len(self.starts)

    def set_epoch(self, epoch: int) -> None:
        """记录当前 epoch，用于确定性改变训练日顺序。"""
        self.epoch = int(epoch)

    def __iter__(self) -> Any:
        """按既定顺序迭代 TradeDateBatchSampler 中的训练批次。"""
        if self.shuffle:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)  # 固定 DataLoader 随机源
            order = torch.randperm(len(self), generator=generator).tolist()
        else:
            order = range(len(self))
        for batch_number in order:
            # 对 `(yield range(int(self.starts[batch_number]), int(self.stops[batch_number])))` 求值，以触发当前步骤需要的计算。
            yield range(
                int(self.starts[batch_number]), int(self.stops[batch_number])
            )

    # 使用 `property` 包装 `batch_sizes`，并定义其可复用的处理逻辑。
    @property
    def batch_sizes(self) -> np.ndarray:
        """返回各批次包含的样本数量。"""
        return self.stops - self.starts

    def batch_slice(self, batch_number: int) -> slice:
        """返回指定批次在全局顺序中的切片边界。"""
        return slice(
            int(self.starts[batch_number]), int(self.stops[batch_number])
        )

    def temporal_pair(self, pair_number: int) -> dict[str, object]:
        """返回相邻交易日共同股票的对齐位置。"""
        if self.temporal_pairs is None:
            raise ValueError("当前 sampler 没有时序相邻截面对齐信息")
        return self.temporal_pairs[pair_number]

    def summary(self) -> dict[str, object]:
        """汇总批采样器的日期、样本和批大小信息。"""
        sizes = self.batch_sizes
        result: dict[str, object] = {
            "train_batch_mode": "trade_date",
            "train_batch_count": len(sizes),
            "train_date_count": len(sizes),
            "train_sample_count": int(sizes.sum()),
            "train_batch_size_min": int(sizes.min()),
            "train_batch_size_median": float(np.median(sizes)),
            "train_batch_size_max": int(sizes.max()),
            "dropped_train_samples": 0,
            "reshuffle_each_epoch": self.shuffle,
            "gradient_update_mode": (
                "once_per_epoch" if self.temporal_pairs is not None else "per_batch"
            ),
            "optimizer_steps_per_epoch": (
                1 if self.temporal_pairs is not None else len(sizes)
            ),
        }
        if self.temporal_pairs is not None:
            common_counts = np.array(
                [pair["common_count"] for pair in self.temporal_pairs],
                dtype=np.int64,
            )
            coverages = np.array(
                [pair["coverage"] for pair in self.temporal_pairs],
                dtype=np.float64,
            )
            result.update(
                {
                    "temporal_lag_trade_days": 1,
                    "temporal_anchor_mode": (
                        "previous_prediction_stop_gradient_cached"
                    ),
                    "model_forward_passes_per_train_date": 1,
                    "train_temporal_pair_count": len(self.temporal_pairs),
                    "temporal_common_count_min": int(common_counts.min()),
                    "temporal_common_count_median": float(
                        np.median(common_counts)
                    ),
                    "temporal_common_count_max": int(common_counts.max()),
                    "temporal_common_coverage_min": float(coverages.min()),
                    "temporal_common_coverage_median": float(
                        np.median(coverages)
                    ),
                    "temporal_common_coverage_max": float(coverages.max()),
                }
            )
        return result


class MemmapBatchCollator:
    """每个 worker 首次取 batch 时才打开只读 memmap。"""

    def __init__(self) -> None:
        self.features = None
        self.targets = None
        self.date_ids = None

    def __call__(self, sample_indices: list[int]) -> dict[str, torch.Tensor]:
        """按样本行号批量读取 memmap 特征和标签，并组装设备传输前的批次。"""
        if self.features is None:
            self.features = np.load(FEATURES_PATH, mmap_mode="r")
            self.targets = np.load(TARGETS_PATH, mmap_mode="r")
            self.date_ids = np.load(DATE_IDS_PATH, mmap_mode="r")
        indices = np.asarray(sample_indices, dtype=np.int64)
        return {
            "features": torch.from_numpy(
                np.array(self.features[indices], dtype=np.float32, copy=True, order="C")
            ),
            "targets": torch.from_numpy(
                np.array(self.targets[indices], dtype=np.float32, copy=True)
            ),
            "date_ids": torch.from_numpy(
                np.array(self.date_ids[indices], dtype=np.int32, copy=True)
            ),
            "sample_indices": torch.from_numpy(indices),
        }


def load_split(forecast_year: int) -> dict[str, np.ndarray]:
    """加载指定预测年度的训练、验证和预测样本行号。"""
    path = SPLIT_DIR / f"forecast_{forecast_year}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def make_loaders(
    forecast_year: int,
    seed: int,
    batch_size: int | None = None,
    workers: int | None = None,
    inference_batch_size: int | None = None,
    train_batch_mode: str | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    """根据滚动切分创建训练、验证和预测 DataLoader。"""
    split = load_split(forecast_year)
    effective_train_batch = (
        TRAINING_CONFIG.batch_size if batch_size is None else batch_size
    )
    if inference_batch_size is not None:
        effective_inference_batch = inference_batch_size
    elif batch_size is not None:
        effective_inference_batch = batch_size
    else:
        effective_inference_batch = TRAINING_CONFIG.inference_batch_size
    effective_mode = train_batch_mode or TRAINING_CONFIG.train_batch_mode
    if effective_mode not in TRAIN_BATCH_MODES:
        raise ValueError(f"未知训练 batch 模式：{effective_mode}")
    temporal_mode = TRAINING_CONFIG.loss_name == TEMPORAL_TURNOVER_LOSS
    if temporal_mode and effective_mode != "trade_date":
        raise ValueError("sigmoid_rank_ic_turnover 必须使用 trade_date batch")
    effective_workers = NUM_WORKERS if workers is None else workers
    common = {"num_workers": effective_workers, "pin_memory": PIN_MEMORY}
    if effective_workers > 0:
        common.update({"persistent_workers": True, "prefetch_factor": 2})

    train_dataset = SampleIndexDataset(split["train_indices"])
    generator = torch.Generator().manual_seed(seed)  # 固定 DataLoader 随机源
    if effective_mode == "trade_date":
        date_ids = sample_index["date_id"].to_numpy(np.int32, copy=False)
        batch_sampler = TradeDateBatchSampler(  # 按完整交易日截面组 batch
            split["train_indices"], date_ids, seed=seed,
            shuffle=not temporal_mode,
            instrument_ids=(
                sample_index["instrument_id"].to_numpy(copy=False)
                if temporal_mode else None
            ),
        )
        train_loader = DataLoader(  # 按既定采样口径创建 DataLoader
            train_dataset,
            batch_sampler=batch_sampler,
            generator=generator,
            collate_fn=MemmapBatchCollator(),
            **common,
        )
    else:
        train_loader = DataLoader(  # 按既定采样口径创建 DataLoader
            train_dataset,
            batch_size=effective_train_batch,
            shuffle=True,
            drop_last=True,
            generator=generator,
            collate_fn=MemmapBatchCollator(),
            **common,
        )
    validation_loader = DataLoader(  # 按既定采样口径创建 DataLoader
        SampleIndexDataset(split["validation_indices"]),
        batch_size=effective_inference_batch,
        shuffle=False,
        drop_last=False,
        collate_fn=MemmapBatchCollator(),
        **common,
    )
    prediction_loader = None
    if len(split["prediction_indices"]):
        prediction_loader = DataLoader(  # 按既定采样口径创建 DataLoader
            SampleIndexDataset(split["prediction_indices"]),
            batch_size=effective_inference_batch,
            shuffle=False,
            drop_last=False,
            collate_fn=MemmapBatchCollator(),
            **common,
        )
    return train_loader, validation_loader, prediction_loader


def set_train_epoch(loader: DataLoader, epoch: int) -> None:
    """把当前 epoch 传递给支持确定性重排的批采样器。"""
    if isinstance(loader.batch_sampler, TradeDateBatchSampler):
        loader.batch_sampler.set_epoch(epoch)  # 同步 epoch，保持采样可复现


def build_model() -> nn.Module:
    """按当前配置构建模型，并保持历史 checkpoint 的参数键兼容。"""
    model = OrderFlowVisionTransformer(MODEL_CONFIG).to(DEVICE)
    if USE_COMPILE:
        if DEVICE.type != "cuda":
            raise RuntimeError("torch.compile 正式配置要求 CUDA")
        if TRAINING_CONFIG.train_batch_mode == "trade_date":
            model = torch.compile(model, mode=COMPILE_MODE, dynamic=True)  # 编译模型计算图
        else:
            model = torch.compile(model, mode=COMPILE_MODE)  # 编译模型计算图
    return model


def unwrap_model(model: nn.Module) -> nn.Module:
    """从编译或包装模型中取回原始 PyTorch 模型。"""
    original = getattr(model, "_orig_mod", None)
    return original if original is not None else model


def canonical_state(model: nn.Module) -> Any:
    """提取不含编译包装前缀的规范化模型参数字典。"""
    state = unwrap_model(model).state_dict()
    if any(key.startswith(("module.", "_orig_mod.")) for key in state):
        raise AssertionError("canonical checkpoint 含包装器前缀")
    return state


def load_canonical_state(model: nn.Module, state: Any) -> None:
    """把规范化参数字典加载到原始或包装模型。"""
    unwrap_model(model).load_state_dict(state)


# 根据当前训练配置补齐默认参数；数值计算统一委托给 vit_stage06_ranking。
def compute_training_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    loss_name: str | None = None,
    rank_temperature: float | None = None,
    huber_delta: float | None = None,
    smooth_l1_beta: float | None = None,
    hybrid_weight: float | None = None,
    turnover_lambda: float | None = None,
    previous_predictions: torch.Tensor | None = None,
    previous_positions: np.ndarray | torch.Tensor | None = None,
    current_positions: np.ndarray | torch.Tensor | None = None,
) -> torch.Tensor:
    """解析当前训练配置后，调用无全局状态的共享损失实现。"""

    return ranking.compute_training_loss(
        predictions,
        targets,
        loss_name=loss_name or TRAINING_CONFIG.loss_name,
        rank_temperature=(
            TRAINING_CONFIG.rank_temperature
            if rank_temperature is None else float(rank_temperature)
        ),
        huber_delta=(
            TRAINING_CONFIG.huber_delta
            if huber_delta is None else float(huber_delta)
        ),
        smooth_l1_beta=(
            TRAINING_CONFIG.smooth_l1_beta
            if smooth_l1_beta is None else float(smooth_l1_beta)
        ),
        hybrid_weight=(
            TRAINING_CONFIG.hybrid_weight
            if hybrid_weight is None else float(hybrid_weight)
        ),
        turnover_lambda=(
            TRAINING_CONFIG.turnover_lambda
            if turnover_lambda is None else float(turnover_lambda)
        ),
        previous_predictions=previous_predictions,
        previous_positions=previous_positions,
        current_positions=current_positions,
    )


def set_random_seed(seed: int) -> None:
    """同步设置 Python、NumPy、CPU 与 CUDA 随机种子。"""
    random.seed(seed)  # 固定本轮采样种子
    np.random.seed(seed)  # 固定本轮采样种子
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_context() -> Any:
    """按设备和配置返回混合精度上下文。"""
    if USE_BF16:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def move_batch(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """把一个训练批次移动到指定设备，并保持非阻塞传输策略。"""
    return (
        batch["features"].to(DEVICE, non_blocking=PIN_MEMORY),
        batch["targets"].to(DEVICE, non_blocking=PIN_MEMORY),
    )


class CudaBatchPrefetcher:
    def __init__(self, loader: DataLoader) -> None:
        self.loader = loader

    def __iter__(self) -> Any:
        """按既定顺序迭代 CudaBatchPrefetcher 中的训练批次。"""
        if not USE_CUDA_PREFETCH or DEVICE.type != "cuda":
            for batch in self.loader:
                features, targets = move_batch(batch)  # 将 batch 搬到训练设备
                # 对 `(yield (batch, features, targets))` 求值，以触发当前步骤需要的计算。
                yield batch, features, targets
            return
        iterator = iter(self.loader)
        stream = torch.cuda.Stream(device=DEVICE)
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
                features, targets = move_batch(batch)  # 将 batch 搬到训练设备
            next_item = (batch, features, targets)

        preload()  # 异步预取下一批数据
        while next_item is not None:
            torch.cuda.current_stream(DEVICE).wait_stream(stream)  # 等待预取流完成
            batch, features, targets = next_item
            features.record_stream(torch.cuda.current_stream(DEVICE))  # 绑定当前 CUDA stream 生命周期
            targets.record_stream(torch.cuda.current_stream(DEVICE))  # 绑定当前 CUDA stream 生命周期
            preload()  # 异步预取下一批数据
            # 对 `(yield (batch, features, targets))` 求值，以触发当前步骤需要的计算。
            yield batch, features, targets


def build_optimizer(model: nn.Module) -> torch.optim.Adam:
    """创建优化器，并按配置启用 fused Adam。"""
    return torch.optim.Adam(
        model.parameters(),
        lr=TRAINING_CONFIG.learning_rate,
        betas=(TRAINING_CONFIG.adam_beta1, TRAINING_CONFIG.adam_beta2),
        eps=TRAINING_CONFIG.adam_epsilon,
        weight_decay=TRAINING_CONFIG.weight_decay,
        fused=(USE_FUSED_ADAM and DEVICE.type == "cuda"),
    )


def _train_standard_epoch(model: Any, loader: Any, optimizer: Any, max_batches: int | None = None) -> Any:
    """执行该模式下的训练步骤，并返回损失与诊断信息。"""
    model.train()  # 启用训练模式
    date_batch_mode = isinstance(loader.batch_sampler, TradeDateBatchSampler)
    loss_sum = torch.zeros((), dtype=torch.float64, device=DEVICE)
    batch_pearson_sum = torch.zeros((), dtype=torch.float64, device=DEVICE)
    weighted_pearson = torch.zeros((), dtype=torch.float64, device=DEVICE)
    soft_rank_ic_sum = torch.zeros((), dtype=torch.float64, device=DEVICE)
    sample_count = 0
    batch_count = 0
    soft_rank_batch_count = 0
    min_batch_size, max_batch_size = None, 0
    started = time.perf_counter()
    for batch_number, (batch, features, targets) in enumerate(
        CudaBatchPrefetcher(loader), start=1
    ):
        if date_batch_mode:
            date_ids = batch["date_ids"]
            if int(date_ids.min()) != int(date_ids.max()):
                raise AssertionError("日截面训练 batch 混入多个交易日")
        optimizer.zero_grad(set_to_none=True)  # 清空上一批梯度
        with autocast_context():
            predictions = model(features)  # 执行当前 batch 前向传播
        loss = compute_training_loss(predictions, targets)  # 按实验口径计算可微损失
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
        if TRAINING_CONFIG.loss_name in RANK_LOSS_NAMES:
            soft_rank_ic_sum += -loss.detach().double()
            soft_rank_batch_count += 1
        sample_count += current_size
        batch_count += 1
        min_batch_size = (
            current_size if min_batch_size is None
            else min(min_batch_size, current_size)
        )
        max_batch_size = max(max_batch_size, current_size)
        if max_batches is not None and batch_number >= max_batches:
            break
    if not batch_count:
        raise ValueError("训练 loader 没有可用 batch")
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()  # 同步 GPU 后计时
    batch_mean_pearson = float((batch_pearson_sum / batch_count).item())
    sample_weighted_pearson = float((weighted_pearson / sample_count).item())
    selected_train_pearson = (
        batch_mean_pearson if date_batch_mode else sample_weighted_pearson
    )
    soft_rank_ic = (
        float((soft_rank_ic_sum / soft_rank_batch_count).item())
        if soft_rank_batch_count else float("nan")
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
        # 兼容旧 history 字段；含义仍是 Pearson IC。
        "train_batch_ic_mean": batch_mean_pearson,
        "train_sample_weighted_ic": sample_weighted_pearson,
        "train_samples": sample_count,
        "train_batches": batch_count,
        "train_batch_size_min": int(min_batch_size),
        "train_batch_size_max": int(max_batch_size),
        "train_seconds": time.perf_counter() - started,
    }


def _train_temporal_turnover_epoch(
    model: Any, loader: Any, optimizer: Any, max_batches: int | None = None,
) -> Any:
    """按自然日期累计相邻截面梯度，并在整个 epoch 结束后更新一次。"""

    sampler = loader.batch_sampler
    if (
        not isinstance(sampler, TradeDateBatchSampler)
        or sampler.temporal_pairs is None
        or sampler.shuffle
    ):
        raise ValueError("时序换手损失要求按日期升序且关闭 reshuffle")
    available_pairs = len(sampler.temporal_pairs)
    pair_limit = (
        available_pairs if max_batches is None
        else min(int(max_batches), available_pairs)
    )
    if pair_limit < 1:
        raise ValueError("时序换手训练至少需要一个相邻日期对")

    model.train()  # 启用训练模式
    optimizer.zero_grad(set_to_none=True)  # 清空上一批梯度
    loss_sum = torch.zeros((), dtype=torch.float64, device=DEVICE)
    pearson_sum = torch.zeros((), dtype=torch.float64, device=DEVICE)
    weighted_pearson = torch.zeros((), dtype=torch.float64, device=DEVICE)
    prediction_rank_sum = torch.zeros((), dtype=torch.float64, device=DEVICE)
    temporal_rank_sum = torch.zeros((), dtype=torch.float64, device=DEVICE)
    temporal_penalty_sum = torch.zeros((), dtype=torch.float64, device=DEVICE)
    sample_count = 0
    pair_count = 0
    min_batch_size, max_batch_size = None, 0
    previous_predictions = None
    previous_date_id = None
    started = time.perf_counter()

    for batch, features, targets in CudaBatchPrefetcher(loader):
        date_ids = batch["date_ids"]
        if int(date_ids.min()) != int(date_ids.max()):
            raise AssertionError("日截面训练 batch 混入多个交易日")
        current_date_id = int(date_ids[0])
        with autocast_context():
            current_predictions = model(features)  # 执行当前 batch 前向传播
        if previous_predictions is None:
            # 首日只建立固定锚点，不计算标签项，也不保留计算图。
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
            TRAINING_CONFIG.rank_temperature,
            TRAINING_CONFIG.turnover_lambda,
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
            batch_pearson = pearson_correlation(
                current_predictions, targets
            ).double()
        current_size = len(targets)
        loss_sum += loss.detach().double()
        pearson_sum += batch_pearson
        weighted_pearson += batch_pearson * current_size
        prediction_rank_sum += components[
            "prediction_soft_rank_ic"
        ].detach().double()
        temporal_rank_sum += components["temporal_soft_rank_ic"].detach().double()
        temporal_penalty_sum += components["temporal_penalty"].detach().double()
        sample_count += current_size
        pair_count += 1
        min_batch_size = (
            current_size if min_batch_size is None
            else min(min_batch_size, current_size)
        )
        max_batch_size = max(max_batch_size, current_size)
        # 参数在 epoch 结束前不更新，因此当前预测可直接作为下一日常数锚点。
        previous_predictions = current_predictions.detach()
        previous_date_id = current_date_id
        del current_predictions, components, loss, values
        if pair_count >= pair_limit:
            break

    if pair_count != pair_limit:
        raise AssertionError(
            f"时序日期对覆盖不完整：{pair_count}/{pair_limit}"
        )
    optimizer.step()  # 更新参数或学习率状态
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()  # 同步 GPU 后计时
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
        # 兼容旧 history 字段；含义仍是 Pearson IC。
        "train_batch_ic_mean": mean_pearson,
        "train_sample_weighted_ic": weighted_mean_pearson,
        "train_samples": sample_count,
        "train_batches": pair_count,
        "train_batch_size_min": int(min_batch_size),
        "train_batch_size_max": int(max_batch_size),
        "train_seconds": time.perf_counter() - started,
    }


def train_one_epoch(model: Any, loader: Any, optimizer: Any, max_batches: int | None = None) -> Any:
    """训练一个完整 epoch，并返回损失与批次诊断。"""
    if TRAINING_CONFIG.loss_name == TEMPORAL_TURNOVER_LOSS:
        return _train_temporal_turnover_epoch(
            model, loader, optimizer, max_batches=max_batches
        )
    return _train_standard_epoch(
        model, loader, optimizer, max_batches=max_batches
    )


def average_descending_ranks(values: np.ndarray) -> np.ndarray:
    """精确降序平均秩；相同值共享其占用名次的均值。"""

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(array) < 2 or not np.isfinite(array).all():
        raise ValueError("排名输入必须至少有两个有限值")
    order = np.argsort(-array, kind="stable")
    ordered = array[order]
    starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
    stops = np.r_[starts[1:], len(array)]
    group_ranks = (starts + 1 + stops) / 2
    ranks = np.empty(len(array), dtype=np.float64)
    ranks[order] = np.repeat(group_ranks, stops - starts)  # 将日内秩广播回样本
    return ranks


def numpy_pearson(x: np.ndarray, y: np.ndarray) -> float:
    """用 NumPy 计算两个有效向量的 Pearson 相关系数。"""
    left = np.asarray(x, dtype=np.float64).reshape(-1)
    right = np.asarray(y, dtype=np.float64).reshape(-1)
    centered_left = left - left.mean()
    centered_right = right - right.mean()
    denominator = np.sqrt(
        np.square(centered_left).sum() * np.square(centered_right).sum()
    )
    if not np.isfinite(denominator) or denominator <= 1e-12:
        raise FloatingPointError("相关系数分母为零或非有限数")
    return float(np.dot(centered_left, centered_right) / denominator)


def summarize_validation_arrays(
    predictions: np.ndarray, targets: np.ndarray, date_ids: np.ndarray,
    validation_metric: str, instrument_ids: np.ndarray | None = None,
) -> dict[str, float | int]:
    """先恢复完整日期截面，再计算硬 RankIC、日度 Pearson 和 pooled IC。"""

    x = np.asarray(predictions, dtype=np.float64).reshape(-1)
    y = np.asarray(targets, dtype=np.float64).reshape(-1)
    dates = np.asarray(date_ids, dtype=np.int64).reshape(-1)
    instruments = (
        None if instrument_ids is None
        else np.asarray(instrument_ids).reshape(-1)
    )
    if not (len(x) == len(y) == len(dates)) or len(x) < 2:
        raise ValueError("验证预测、标签和日期必须等长且至少包含两个样本")
    if instruments is not None and len(instruments) != len(x):
        raise ValueError("验证股票键与预测长度不一致")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise FloatingPointError("验证预测或标签包含 NaN/Inf")
    if validation_metric not in VALIDATION_METRICS:
        raise ValueError(f"未知验证指标：{validation_metric}")

    order = np.argsort(dates, kind="stable")
    x, y, dates = x[order], y[order], dates[order]
    if instruments is not None:
        instruments = instruments[order]
    boundaries = np.flatnonzero(dates[1:] != dates[:-1]) + 1
    starts = np.r_[0, boundaries]
    stops = np.r_[boundaries, len(dates)]
    daily_rank_ic, daily_pearson_ic = [], []
    for start, stop in zip(starts, stops):
        if stop - start < 2:
            raise FloatingPointError("验证集存在不足两个样本的交易日")
        daily_x, daily_y = x[start:stop], y[start:stop]
        daily_pearson_ic.append(numpy_pearson(daily_x, daily_y))
        daily_rank_ic.append(
            numpy_pearson(
                average_descending_ranks(daily_x),
                average_descending_ranks(daily_y),
            )
        )
    temporal_rank_ic: list[float] = []
    if instruments is not None:
        unique_dates = dates[starts]
        if len(unique_dates) > 1 and np.any(np.diff(unique_dates) != 1):
            raise ValueError("验证集日期不连续，无法计算相邻预测 RankIC")
        for pair_number in range(1, len(starts)):
            previous_slice = slice(starts[pair_number - 1], stops[pair_number - 1])
            current_slice = slice(starts[pair_number], stops[pair_number])
            previous_ids = instruments[previous_slice]
            current_ids = instruments[current_slice]
            if (
                len(np.unique(previous_ids)) != len(previous_ids)
                or len(np.unique(current_ids)) != len(current_ids)
            ):
                raise ValueError("验证集同一日期股票键重复")
            common, previous_positions, current_positions = np.intersect1d(
                previous_ids, current_ids, assume_unique=True,
                return_indices=True,
            )
            if len(common) < 2:
                raise ValueError("验证集相邻日期共同股票少于2只")
            temporal_rank_ic.append(
                numpy_pearson(
                    average_descending_ranks(
                        x[current_slice][current_positions]
                    ),
                    average_descending_ranks(
                        x[previous_slice][previous_positions]
                    ),
                )
            )
    mean_rank_ic = float(np.mean(daily_rank_ic))
    mean_pearson_ic = float(np.mean(daily_pearson_ic))
    mean_temporal_rank_ic = (
        float(np.mean(temporal_rank_ic)) if temporal_rank_ic else float("nan")
    )
    pooled_ic = numpy_pearson(x, y)  # 计算当前截面相关性
    selected_ic = (
        mean_rank_ic
        if validation_metric == "daily_mean_spearman_ic"
        else mean_pearson_ic
    )
    values = np.array(
        [selected_ic, mean_rank_ic, mean_pearson_ic, pooled_ic], dtype=np.float64
    )
    if not np.isfinite(values).all():
        raise FloatingPointError("验证 IC 不是有限数")
    return {
        "validation_ic": selected_ic,
        "validation_loss": -selected_ic,
        "validation_daily_rank_ic": mean_rank_ic,
        "validation_daily_pearson_ic": mean_pearson_ic,
        # 兼容旧字段，其含义保持为日均 Pearson IC。
        "validation_daily_ic": mean_pearson_ic,
        "validation_pooled_ic": pooled_ic,
        "validation_temporal_rank_ic": mean_temporal_rank_ic,
        "validation_temporal_penalty": 1 - mean_temporal_rank_ic,
        "validation_temporal_pairs": len(temporal_rank_ic),
        "validation_dates": len(starts),
        "validation_samples": len(x),
    }


# 使用 `torch.no_grad()` 包装 `evaluate_model`，并定义其可复用的处理逻辑。
@torch.no_grad()
def evaluate_model(model: Any, loader: Any, max_batches: int | None = None) -> Any:
    """在验证集上评估模型，并汇总逐日 IC 与损失诊断。"""
    model.eval()  # 关闭 dropout 等训练行为
    prediction_parts, target_parts, date_parts, sample_index_parts = [], [], [], []
    started = time.perf_counter()
    for batch_number, (batch, features, targets) in enumerate(
        CudaBatchPrefetcher(loader), start=1
    ):
        with autocast_context():
            predictions = model(features)  # 执行当前 batch 前向传播
        prediction_parts.append(
            predictions.float().cpu().numpy().astype(np.float64, copy=False)
        )
        target_parts.append(
            targets.float().cpu().numpy().astype(np.float64, copy=False)
        )
        date_parts.append(batch["date_ids"].numpy().astype(np.int64, copy=False))
        sample_index_parts.append(
            batch["sample_indices"].numpy().astype(np.int64, copy=False)
        )
        if max_batches is not None and batch_number >= max_batches:
            break
    if not prediction_parts:
        raise ValueError("验证 loader 没有可用 batch")
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()  # 同步 GPU 后计时
    result = summarize_validation_arrays(  # 按约定截面口径汇总
        np.concatenate(prediction_parts),
        np.concatenate(target_parts),
        np.concatenate(date_parts),
        TRAINING_CONFIG.validation_metric,
        instrument_ids=sample_index["instrument_id"].to_numpy(copy=False)[
            np.concatenate(sample_index_parts)
        ],
    )
    result["validation_seconds"] = time.perf_counter() - started
    return result


# 使用 `torch.no_grad()` 包装 `predict_model`，并定义其可复用的处理逻辑。
@torch.no_grad()
def predict_model(model: Any, loader: Any) -> pd.DataFrame:
    """在预测集上生成与样本行号严格对齐的模型分数。"""
    model.eval()  # 关闭 dropout 等训练行为
    indices, predictions = [], []
    for batch, features, _ in CudaBatchPrefetcher(loader):
        with autocast_context():
            output = model(features)  # 执行当前 batch 前向传播
        indices.append(batch["sample_indices"].numpy())
        predictions.append(output.float().cpu().numpy().astype(np.float32, copy=False))
    return pd.DataFrame(
        {"sample_index": np.concatenate(indices), "prediction": np.concatenate(predictions)}
    )


def execution_payload() -> dict[str, object]:
    """整理影响单次训练执行方式的配置载荷。"""
    date_batches = TRAINING_CONFIG.train_batch_mode == "trade_date"
    temporal_mode = TRAINING_CONFIG.loss_name == TEMPORAL_TURNOVER_LOSS
    return {
        "parallel_backend": PARALLEL_BACKEND,
        "gpu_count_per_task": 1,
        "train_batch_mode": TRAINING_CONFIG.train_batch_mode,
        "batch_size_per_task": (
            "full_trade_date" if date_batches else TRAINING_CONFIG.batch_size
        ),
        "fixed_batch_size": TRAINING_CONFIG.batch_size,
        "inference_batch_size": TRAINING_CONFIG.inference_batch_size,
        "reshuffle_each_epoch": not temporal_mode,
        "gradient_update_mode": (
            "once_per_epoch" if temporal_mode else "per_batch"
        ),
        "optimizer_steps_per_epoch": 1 if temporal_mode else "per_batch",
        "temporal_lag_trade_days": 1 if temporal_mode else None,
        "temporal_anchor_mode": (
            "previous_prediction_stop_gradient_cached"
            if temporal_mode else None
        ),
        "model_forward_passes_per_train_date": 1 if temporal_mode else None,
        "loss_name": TRAINING_CONFIG.loss_name,
        "loss_parameters": {
            "rank_temperature": TRAINING_CONFIG.rank_temperature,
            "turnover_lambda": TRAINING_CONFIG.turnover_lambda,
            "huber_delta": TRAINING_CONFIG.huber_delta,
            "smooth_l1_beta": TRAINING_CONFIG.smooth_l1_beta,
            "hybrid_weight": TRAINING_CONFIG.hybrid_weight,
        },
        "validation_metric": TRAINING_CONFIG.validation_metric,
        "data_loader_workers": NUM_WORKERS,
        "torch_compile": USE_COMPILE,
        "compile_mode": COMPILE_MODE,
        "compile_dynamic_batch": bool(USE_COMPILE and date_batches),
        "fused_adam": USE_FUSED_ADAM,
        "cuda_prefetch": USE_CUDA_PREFETCH,
    }


def run_payload(
    year: int, seed: int, batch_summary: dict[str, object]
) -> dict[str, object]:
    """整理影响整个 run 身份和可复用性的签名载荷。"""
    return {
        "schema_version": 6,
        "forecast_year": year,
        "seed": seed,
        "dataset_signature": dataset_manifest["signature"],
        "rolling_signature": rolling_manifest["signature"],
        "model_config": asdict(MODEL_CONFIG),
        "training_config": asdict(TRAINING_CONFIG),
        "execution_config": execution_payload(),
        "training_batch_summary": batch_summary,
        "loss": TRAINING_CONFIG.loss_name,
        "loss_parameters": {
            "rank_temperature": TRAINING_CONFIG.rank_temperature,
            "turnover_lambda": TRAINING_CONFIG.turnover_lambda,
            "huber_delta": TRAINING_CONFIG.huber_delta,
            "smooth_l1_beta": TRAINING_CONFIG.smooth_l1_beta,
            "hybrid_weight": TRAINING_CONFIG.hybrid_weight,
            "prediction_input": "raw_model_output",
            "target_input": "daily_winsorized_1pct_zscore",
            "prediction_preprocessing_inside_loss": "none",
            "target_preprocessing_inside_loss": "none",
            "rank_target_gradient": False,
            "temporal_prediction_gradients": "current_only" if (
                TRAINING_CONFIG.loss_name == TEMPORAL_TURNOVER_LOSS
            ) else None,
            "previous_prediction_gradient": "stop_gradient" if (
                TRAINING_CONFIG.loss_name == TEMPORAL_TURNOVER_LOSS
            ) else None,
            "previous_prediction_source": "cached_previous_date_output" if (
                TRAINING_CONFIG.loss_name == TEMPORAL_TURNOVER_LOSS
            ) else None,
            "turnover_penalty_direction": "1_minus_temporal_rank_ic" if (
                TRAINING_CONFIG.loss_name == TEMPORAL_TURNOVER_LOSS
            ) else None,
        },
        "validation_metric": TRAINING_CONFIG.validation_metric,
        "validation_audit_metrics": [
            "daily_mean_spearman_ic",
            "daily_mean_pearson_ic",
            "pooled_pearson_ic",
            "temporal_spearman_rank_ic",
            "temporal_rank_penalty",
        ],
        "precision": "bf16_autocast_with_fp32_loss" if USE_BF16 else "float32",
    }


def atomic_checkpoint(payload: dict[str, object], path: Path) -> None:
    """先写临时 checkpoint，再原子替换正式模型文件。"""
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def checkpoint_payload(
    model: Any, optimizer: Any, generator_state: Any, epoch: Any, best_epoch: Any, best_ic: Any,
    best_validation_metrics: Any, patience_count: Any, signature: Any, epoch_metrics: Any,
) -> dict[str, object]:
    """构建 checkpoint 中的模型、优化器、进度和签名信息。"""
    return {
        "model_state": canonical_state(model),
        "optimizer_state": optimizer.state_dict(),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "train_generator_state": generator_state,
        "epoch": epoch,
        "next_trade_date_sampler_epoch": epoch + 1,
        "best_epoch": best_epoch,
        "best_validation_ic": best_ic,
        "best_validation_metrics": best_validation_metrics,
        "epoch_metrics": epoch_metrics,
        "epochs_without_improvement": patience_count,
        "run_signature": signature,
        "execution_config": execution_payload(),
        "model_config": asdict(MODEL_CONFIG),
        "training_config": asdict(TRAINING_CONFIG),
    }


def load_best_model(path: Path) -> Any:
    """加载最佳验证 checkpoint，并恢复可用于预测的模型。"""
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    model = build_model()  # 创建模型并部署到设备
    load_canonical_state(model, checkpoint["model_state"])  # 恢复模型权重
    model.eval()  # 关闭 dropout 等训练行为
    return model


def train_single_run(year: int, seed: int) -> Any:
    """完成一个预测年度和一个随机种子的训练、早停与预测。"""
    set_random_seed(seed)  # 固定各随机源
    train_loader, validation_loader, _ = make_loaders(year, seed)
    batch_summary = loader_batch_summary(train_loader)
    payload = run_payload(year, seed, batch_summary)
    signature = stable_hash(payload)  # 生成数据复用签名

    run_dir = MODEL_ROOT / f"forecast_{year}" / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path, last_path = run_dir / "best.pt", run_dir / "last.pt"
    history_path, completed_path = run_dir / "history.parquet", run_dir / "completed.json"
    atomic_json(  # 原子落盘，避免半成品
        {"year_signature": stable_hash({k: v for k, v in payload.items() if k != "seed"}),
         "payload": {k: v for k, v in payload.items() if k != "seed"}},
        MODEL_ROOT / f"forecast_{year}" / "model_config.json",
    )
    if completed_path.exists() and best_path.exists() and not OVERWRITE_TRAINING:
        completed = read_json(completed_path)
        if completed.get("run_signature") == signature:
            print(f"[{year}/seed={seed}] 已完成，直接复用。")
            return load_best_model(best_path), completed["summary"]

    model = build_model()  # 创建模型并部署到设备
    optimizer = build_optimizer(model)  # 按训练配置创建优化器
    start_epoch, best_epoch = 1, 0
    best_ic, patience_count = -np.inf, 0
    best_validation_metrics: dict[str, float] = {}
    history: list[dict[str, object]] = []
    if last_path.exists() and not OVERWRITE_TRAINING:
        checkpoint = torch.load(last_path, map_location=DEVICE, weights_only=False)
        if checkpoint.get("run_signature") != signature:
            raise ValueError(f"{last_path} 配置签名不一致，请显式覆盖重训")
        load_canonical_state(model, checkpoint["model_state"])  # 恢复模型权重
        optimizer.load_state_dict(checkpoint["optimizer_state"])  # 恢复优化器续训状态
        random.setstate(checkpoint["python_random_state"])  # 恢复 Python 随机状态
        np.random.set_state(checkpoint["numpy_random_state"])  # 恢复 NumPy 随机状态
        torch.set_rng_state(checkpoint["torch_rng_state"])  # 恢复 PyTorch CPU 随机状态
        if torch.cuda.is_available() and checkpoint["cuda_rng_state_all"] is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])  # 恢复 CUDA 随机状态
        train_loader.generator.set_state(checkpoint["train_generator_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        expected_sampler_epoch = checkpoint.get("next_trade_date_sampler_epoch")
        if (
            TRAINING_CONFIG.train_batch_mode == "trade_date"
            and expected_sampler_epoch is not None
            and int(expected_sampler_epoch) != start_epoch
        ):
            raise AssertionError("checkpoint 中的下一日截面 sampler epoch 不一致")
        best_epoch = int(checkpoint["best_epoch"])
        best_ic = float(checkpoint["best_validation_ic"])
        best_validation_metrics = {
            key: float(value)
            for key, value in checkpoint.get(
                "best_validation_metrics", {}
            ).items()
        }
        patience_count = int(checkpoint["epochs_without_improvement"])
        if history_path.exists():
            history = pd.read_parquet(history_path).to_dict("records")
        print(f"[{year}/seed={seed}] 从 epoch {start_epoch} 续训。")

    started = time.perf_counter()
    stopped_early = False
    for epoch in range(start_epoch, TRAINING_CONFIG.max_epochs + 1):
        epoch_started = time.perf_counter()
        set_train_epoch(train_loader, epoch)  # 同步 epoch，保持采样可复现
        train_metrics = train_one_epoch(model, train_loader, optimizer)  # 完成一个训练 epoch
        validation_metrics = evaluate_model(model, validation_loader)  # 无梯度计算验证指标
        validation_ic = validation_metrics["validation_ic"]
        improved = validation_ic > best_ic + TRAINING_CONFIG.improvement_tolerance
        if improved:
            best_ic, best_epoch, patience_count = validation_ic, epoch, 0
            best_validation_metrics = {
                "validation_daily_rank_ic": validation_metrics[
                    "validation_daily_rank_ic"
                ],
                "validation_daily_pearson_ic": validation_metrics[
                    "validation_daily_pearson_ic"
                ],
                "validation_pooled_ic": validation_metrics["validation_pooled_ic"],
                "validation_temporal_rank_ic": validation_metrics[
                    "validation_temporal_rank_ic"
                ],
                "validation_temporal_penalty": validation_metrics[
                    "validation_temporal_penalty"
                ],
            }
        else:
            patience_count += 1
        history.append(
            {
                "forecast_year": year,
                "seed": seed,
                "epoch": epoch,
                "train_batch_mode": TRAINING_CONFIG.train_batch_mode,
                "loss_name": TRAINING_CONFIG.loss_name,
                "validation_metric": TRAINING_CONFIG.validation_metric,
                **train_metrics,
                **validation_metrics,
                "best_validation_ic": best_ic,
                "epochs_without_improvement": patience_count,
                "epoch_seconds": time.perf_counter() - epoch_started,
            }
        )
        atomic_parquet(pd.DataFrame(history), history_path)  # 原子落盘，避免半成品
        state = checkpoint_payload(
            model, optimizer, train_loader.generator.get_state(), epoch, best_epoch,
            best_ic, best_validation_metrics, patience_count, signature,
            history[-1],
        )
        atomic_checkpoint(state, last_path)
        if improved:
            atomic_checkpoint(state, best_path)
        print(
            f"[{year}/seed={seed}] epoch={epoch:03d} "
            f"trainLoss={train_metrics['train_loss']:.6f} "
            f"trainPearson={train_metrics['train_ic']:.6f} "
            f"trainSoftRank={train_metrics['train_soft_rank_ic']:.6f} "
            f"trainTemporal={train_metrics['train_temporal_soft_rank_ic']:.6f} "
            f"validRank={validation_metrics['validation_daily_rank_ic']:.6f} "
            f"validPearson={validation_metrics['validation_daily_pearson_ic']:.6f} "
            f"validPooled={validation_metrics['validation_pooled_ic']:.6f} "
            f"validTemporal={validation_metrics['validation_temporal_rank_ic']:.6f} "
            f"selected={validation_ic:.6f} best={best_ic:.6f} "
            f"patience={patience_count}/{TRAINING_CONFIG.early_stopping_patience}",
            flush=True,
        )
        if patience_count >= TRAINING_CONFIG.early_stopping_patience:
            stopped_early = True
            break
    if not best_path.exists():
        raise FileNotFoundError(f"训练后没有最佳 checkpoint：{best_path}")
    best_epoch_history = next(
        row for row in history if int(row["epoch"]) == best_epoch
    )
    summary = {
        "forecast_year": year,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_ic": best_ic,
        "best_validation_rank_ic": best_validation_metrics.get(
            "validation_daily_rank_ic"
        ),
        "best_validation_daily_pearson_ic": best_validation_metrics.get(
            "validation_daily_pearson_ic"
        ),
        # 兼容旧汇总字段，其含义保持为日均 Pearson IC。
        "best_validation_daily_ic": best_validation_metrics.get(
            "validation_daily_pearson_ic"
        ),
        "best_validation_pooled_ic": best_validation_metrics.get(
            "validation_pooled_ic"
        ),
        "best_validation_temporal_rank_ic": best_validation_metrics.get(
            "validation_temporal_rank_ic"
        ),
        "best_validation_temporal_penalty": best_validation_metrics.get(
            "validation_temporal_penalty"
        ),
        "best_epoch_train_loss": best_epoch_history["train_loss"],
        "best_epoch_train_prediction_soft_rank_ic": best_epoch_history[
            "train_prediction_soft_rank_ic"
        ],
        "best_epoch_train_temporal_soft_rank_ic": best_epoch_history[
            "train_temporal_soft_rank_ic"
        ],
        "best_epoch_train_temporal_penalty": best_epoch_history[
            "train_temporal_penalty"
        ],
        "best_epoch_train_forward_passes": best_epoch_history.get(
            "train_forward_passes"
        ),
        "temporal_anchor_gradient": best_epoch_history.get(
            "temporal_anchor_gradient"
        ),
        "validation_metric": TRAINING_CONFIG.validation_metric,
        "loss_name": TRAINING_CONFIG.loss_name,
        "rank_temperature": TRAINING_CONFIG.rank_temperature,
        "turnover_lambda": TRAINING_CONFIG.turnover_lambda,
        "huber_delta": TRAINING_CONFIG.huber_delta,
        "smooth_l1_beta": TRAINING_CONFIG.smooth_l1_beta,
        "hybrid_weight": TRAINING_CONFIG.hybrid_weight,
        "train_batch_mode": TRAINING_CONFIG.train_batch_mode,
        "train_batch_count": batch_summary["train_batch_count"],
        "train_batch_size_min": batch_summary["train_batch_size_min"],
        "train_batch_size_median": batch_summary["train_batch_size_median"],
        "train_batch_size_max": batch_summary["train_batch_size_max"],
        "reshuffle_each_epoch": batch_summary["reshuffle_each_epoch"],
        "gradient_update_mode": batch_summary["gradient_update_mode"],
        "optimizer_steps_per_epoch": batch_summary["optimizer_steps_per_epoch"],
        "temporal_lag_trade_days": batch_summary.get("temporal_lag_trade_days"),
        "temporal_anchor_mode": batch_summary.get("temporal_anchor_mode"),
        "model_forward_passes_per_train_date": batch_summary.get(
            "model_forward_passes_per_train_date"
        ),
        "train_temporal_pair_count": batch_summary.get(
            "train_temporal_pair_count", 0
        ),
        "temporal_common_count_min": batch_summary.get(
            "temporal_common_count_min"
        ),
        "temporal_common_count_median": batch_summary.get(
            "temporal_common_count_median"
        ),
        "temporal_common_count_max": batch_summary.get(
            "temporal_common_count_max"
        ),
        "temporal_common_coverage_min": batch_summary.get(
            "temporal_common_coverage_min"
        ),
        "temporal_common_coverage_median": batch_summary.get(
            "temporal_common_coverage_median"
        ),
        "temporal_common_coverage_max": batch_summary.get(
            "temporal_common_coverage_max"
        ),
        "last_epoch": int(history[-1]["epoch"]),
        "stopped_early": stopped_early,
        "elapsed_seconds": time.perf_counter() - started,
        "epoch_elapsed_seconds": float(sum(x["epoch_seconds"] for x in history)),
        "parallel_backend": PARALLEL_BACKEND,
        "gpu_count": 1,
        "run_name": RUN_NAME,
        "status": "completed",
    }
    atomic_json({"run_signature": signature, "summary": summary}, completed_path)  # 原子落盘，避免半成品
    return load_best_model(best_path), summary


def run_smoke_test() -> pd.DataFrame:
    """运行小批量前向、反向和验证冒烟测试，不生成正式产物。"""
    year, seed = TRAINING_CONFIG.forecast_years[0], TRAINING_CONFIG.seeds[0]
    set_random_seed(seed)  # 固定各随机源
    smoke_batch_mode = (
        "trade_date"
        if TRAINING_CONFIG.loss_name == TEMPORAL_TURNOVER_LOSS
        else "fixed_samples"
    )
    train_loader, validation_loader, _ = make_loaders(
        year, seed, batch_size=64, inference_batch_size=64, workers=0,
        train_batch_mode=smoke_batch_mode,
    )
    model = build_model()  # 创建模型并部署到设备
    optimizer = build_optimizer(model)  # 按训练配置创建优化器
    training = train_one_epoch(model, train_loader, optimizer, 1)  # 完成一个训练 epoch
    validation = evaluate_model(model, validation_loader, 1)  # 无梯度计算验证指标
    state = canonical_state(model)  # 提取标准模型权重
    with tempfile.TemporaryDirectory() as temporary_directory:
        path = Path(temporary_directory) / "smoke.pt"
        torch.save({"model_state": state}, path)
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
        raw_model = OrderFlowVisionTransformer(MODEL_CONFIG).to(DEVICE)
        raw_model.load_state_dict(checkpoint["model_state"])
    del model, raw_model, optimizer
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return pd.DataFrame(
        [{"forecast_year": year, "train_ic": training["train_ic"],
          "validation_ic": validation["validation_ic"], "canonical_checkpoint": True}]
    )


def _trade_date_preflight_batches() -> list[tuple[int, str, dict[str, torch.Tensor]]]:
    """逐日抽取预检批次，核对日批次边界、标签有限性和时序对齐。"""
    date_ids = sample_index["date_id"].to_numpy(np.int32, copy=False)
    seed = TRAINING_CONFIG.seeds[0]
    max_year, max_size = None, -1
    for year in TRAINING_CONFIG.forecast_years:
        split = load_split(year)
        sampler = TradeDateBatchSampler(  # 按完整交易日截面组 batch
            split["train_indices"], date_ids, seed=seed, shuffle=True
        )
        current_max = int(sampler.batch_sizes.max())
        if current_max > max_size:
            max_year, max_size = year, current_max
    if max_year is None:
        raise ValueError("没有可用于日截面显存预检的年度")

    split = load_split(max_year)
    sampler = TradeDateBatchSampler(  # 按完整交易日截面组 batch
        split["train_indices"], date_ids, seed=seed, shuffle=True
    )
    sizes = sampler.batch_sizes
    max_batch = int(np.argmax(sizes))
    median_batch = int(np.argmin(np.abs(sizes - np.median(sizes))))
    selected = [(max_batch, "max"), (median_batch, "median")]
    if sizes[max_batch] == sizes[median_batch]:
        min_batch = int(np.argmin(sizes))
        selected[1] = (min_batch, "min")
    if selected[0][0] == selected[1][0]:
        selected = selected[:1]

    collator = MemmapBatchCollator()  # 整理并搬运 batch
    result = []
    for batch_number, label in selected:
        positions = sampler.batch_slice(batch_number)
        global_indices = split["train_indices"][positions]
        result.append(
            (max_year, label, collator(global_indices.tolist()))
        )
    return result


def _temporal_preflight_pairs() -> list[tuple[
    int, str, dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, object]
]]:
    """选择含最大显存需求在内的十二组不同相邻截面 shape。"""

    date_ids = sample_index["date_id"].to_numpy(np.int32, copy=False)
    instrument_ids = sample_index["instrument_id"].to_numpy(copy=False)
    seed = TRAINING_CONFIG.seeds[0]
    candidates: list[dict[str, object]] = []
    for year in TRAINING_CONFIG.forecast_years:
        split = load_split(year)
        sampler = TradeDateBatchSampler(  # 按完整交易日截面组 batch
            split["train_indices"], date_ids, seed=seed, shuffle=False,
            instrument_ids=instrument_ids,
        )
        for pair_number, pair in enumerate(sampler.temporal_pairs or []):
            previous_size = int(sampler.batch_sizes[pair_number])
            current_size = int(sampler.batch_sizes[pair_number + 1])
            common_count = int(pair["common_count"])
            candidates.append(
                {
                    "year": year,
                    "split": split,
                    "sampler": sampler,
                    "pair_number": pair_number,
                    "pair": pair,
                    "shape": (previous_size, current_size, common_count),
                    # 当前 RankIC 两个软秩、时序项两个共同集软秩均为 O(n²)。
                    "score": 2 * current_size**2 + 2 * common_count**2,
                }
            )
    if not candidates:
        raise ValueError("没有可用于时序换手显存预检的相邻日期对")
    unique_by_shape = {item["shape"]: item for item in candidates}
    ordered = sorted(
        unique_by_shape.values(), key=lambda item: int(item["score"])
    )
    test_count = min(12, len(ordered))
    quantile_positions = np.linspace(
        0, len(ordered) - 1, num=test_count, dtype=np.int64
    )
    selected = [ordered[-1]]
    selected.extend(
        ordered[int(position)]
        for position in quantile_positions
        if ordered[int(position)]["shape"] != selected[0]["shape"]
    )
    selected = selected[:test_count]

    collator = MemmapBatchCollator()  # 整理并搬运 batch
    result = []
    for selection_number, item in enumerate(selected):
        sampler = item["sampler"]
        split = item["split"]
        pair_number = int(item["pair_number"])
        previous_positions = sampler.batch_slice(pair_number)
        current_positions = sampler.batch_slice(pair_number + 1)
        previous_indices = split["train_indices"][previous_positions]
        current_indices = split["train_indices"][current_positions]
        result.append(
            (
                int(item["year"]),
                (
                    "max_temporal_pair" if selection_number == 0
                    else f"dynamic_pair_{selection_number:02d}"
                ),
                collator(previous_indices.tolist()),
                collator(current_indices.tolist()),
                item["pair"],
            )
        )
    return result


def wait_for_preflight_barrier() -> None:
    """让同一 GPU 上的预检 worker 在模型就绪后同时开始。"""

    barrier_value = os.getenv("VIT_PREFLIGHT_BARRIER_DIR")
    if not barrier_value:
        return
    barrier_dir = Path(barrier_value)
    rank = int(os.getenv("VIT_PREFLIGHT_RANK", "0"))
    world_size = int(os.getenv("VIT_PREFLIGHT_WORLD_SIZE", "1"))
    barrier_dir.mkdir(parents=True, exist_ok=True)
    (barrier_dir / f"ready_{rank}").touch()
    deadline = time.monotonic() + 300
    while len(list(barrier_dir.glob("ready_*"))) < world_size:
        if time.monotonic() >= deadline:
            raise TimeoutError("同卡并发预检 worker 屏障等待超时")
        time.sleep(0.1)


def run_full_batch_preflight() -> pd.DataFrame | None:
    """在正式训练前检查全量批次形状、日期与显存相关约束。"""
    if DEVICE.type != "cuda":
        print("当前无 CUDA，跳过正式 batch 显存预检。")
        return None
    year, seed = TRAINING_CONFIG.forecast_years[0], TRAINING_CONFIG.seeds[0]
    temporal_mode = TRAINING_CONFIG.loss_name == TEMPORAL_TURNOVER_LOSS
    if temporal_mode:
        temporal_pairs = _temporal_preflight_pairs()
        batches = []
    elif TRAINING_CONFIG.train_batch_mode == "trade_date":
        temporal_pairs = []
        batches = _trade_date_preflight_batches()
    else:
        temporal_pairs = []
        loader, _, _ = make_loaders(year, seed)
        batches = [(year, "fixed", next(iter(loader)))]

    model = build_model()  # 创建模型并部署到设备
    optimizer = build_optimizer(model)  # 按训练配置创建优化器
    wait_for_preflight_barrier()
    rows = []
    try:
        if temporal_mode:
            optimizer.zero_grad(set_to_none=True)  # 清空上一批梯度
            for batch_year, shape_test, previous_batch, current_batch, pair in temporal_pairs:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                previous_features, _ = move_batch(previous_batch)  # 将 batch 搬到训练设备
                current_features, current_targets = move_batch(current_batch)  # 将 batch 搬到训练设备
                with autocast_context():
                    previous_with_graph = model(previous_features)  # 执行当前 batch 前向传播
                previous_predictions = previous_with_graph.detach()
                del previous_with_graph, previous_features
                with autocast_context():
                    current_predictions = model(current_features)  # 执行当前 batch 前向传播
                components = temporal_turnover_loss_components(  # 按实验口径计算可微损失
                    current_predictions, current_targets, previous_predictions,
                    pair["previous_positions"], pair["current_positions"],
                    TRAINING_CONFIG.rank_temperature,
                    TRAINING_CONFIG.turnover_lambda,
                )
                loss = components["loss"]
                if not torch.isfinite(loss):
                    raise FloatingPointError("时序换手预检损失出现 NaN 或 Inf")
                (loss / len(temporal_pairs)).backward()  # 反向传播当前损失
                torch.cuda.synchronize()  # 同步 GPU 后计时
                common_count = int(pair["common_count"])
                rows.append(
                    {
                        "train_batch_mode": TRAINING_CONFIG.train_batch_mode,
                        "loss_name": TRAINING_CONFIG.loss_name,
                        "rank_temperature": TRAINING_CONFIG.rank_temperature,
                        "turnover_lambda": TRAINING_CONFIG.turnover_lambda,
                        "previous_prediction_gradient": "stop_gradient",
                        "pairwise_relation_count": (
                            2 * len(current_targets) ** 2 + 2 * common_count**2
                        ),
                        "forecast_year": batch_year,
                        "shape_test": shape_test,
                        "previous_batch_size": len(previous_batch["targets"]),
                        "batch_size_per_task": len(current_targets),
                        "common_instrument_count": common_count,
                        "loss": float(loss.item()),
                        "prediction_soft_rank_ic": float(
                            components["prediction_soft_rank_ic"].item()
                        ),
                        "temporal_soft_rank_ic": float(
                            components["temporal_soft_rank_ic"].item()
                        ),
                        "temporal_penalty": float(
                            components["temporal_penalty"].item()
                        ),
                        "elapsed_seconds": time.perf_counter() - started,
                        "peak_allocated_gb": (
                            torch.cuda.max_memory_allocated() / 1024**3
                        ),
                        "peak_reserved_gb": (
                            torch.cuda.max_memory_reserved() / 1024**3
                        ),
                        "compile_dynamic_batch": bool(USE_COMPILE),
                        "compile_mode": COMPILE_MODE,
                        "bf16": USE_BF16,
                    }
                )
                del (
                    previous_batch, current_batch, current_features,
                    current_targets, previous_predictions, current_predictions,
                    components, loss,
                )
            optimizer.step()  # 更新参数或学习率状态
            torch.cuda.synchronize()  # 同步 GPU 后计时
        else:
            for batch_year, shape_test, batch in batches:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                features, targets = move_batch(batch)  # 将 batch 搬到训练设备
                optimizer.zero_grad(set_to_none=True)  # 清空上一批梯度
                with autocast_context():
                    predictions = model(features)  # 执行当前 batch 前向传播
                loss = compute_training_loss(predictions, targets)  # 按实验口径计算可微损失
                if not torch.isfinite(loss):
                    raise FloatingPointError("预检损失出现 NaN 或 Inf")
                loss.backward()  # 反向传播当前损失
                optimizer.step()  # 更新参数或学习率状态
                torch.cuda.synchronize()  # 同步 GPU 后计时
                rows.append(
                    {
                        "train_batch_mode": TRAINING_CONFIG.train_batch_mode,
                        "loss_name": TRAINING_CONFIG.loss_name,
                        "rank_temperature": TRAINING_CONFIG.rank_temperature,
                        "turnover_lambda": TRAINING_CONFIG.turnover_lambda,
                        "pairwise_relation_count": (
                            len(targets) ** 2
                            if TRAINING_CONFIG.loss_name in RANK_LOSS_NAMES else 0
                        ),
                        "forecast_year": batch_year,
                        "shape_test": shape_test,
                        "batch_size_per_task": len(targets),
                        "loss": float(loss.item()),
                        "elapsed_seconds": time.perf_counter() - started,
                        "peak_allocated_gb": (
                            torch.cuda.max_memory_allocated() / 1024**3
                        ),
                        "peak_reserved_gb": (
                            torch.cuda.max_memory_reserved() / 1024**3
                        ),
                        "compile_dynamic_batch": bool(
                            USE_COMPILE
                            and TRAINING_CONFIG.train_batch_mode == "trade_date"
                        ),
                        "bf16": USE_BF16,
                    }
                )
                del batch, features, targets, predictions, loss
    except torch.OutOfMemoryError as error:
        raise RuntimeError(
            f"A2 单任务模式={TRAINING_CONFIG.train_batch_mode}、"
            f"损失={TRAINING_CONFIG.loss_name} 发生 CUDA OOM；"
            "程序不会静默改变训练口径。"
        ) from error
    finally:
        del model, optimizer, batches, temporal_pairs
        torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def load_ensemble_inputs(data_root: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """加载各随机种子的预测，并检查业务键和行序一致。"""
    index = pd.read_parquet(
        data_root / "feature_store" / "sample_index.parquet",
        columns=[
            "sample_index", "trade_date", "instrument_id", "wind_code",
            "can_enter_t1", "label_valid",
        ],
    )
    index["trade_date"] = index["trade_date"].astype(str)
    targets = np.load(data_root / "feature_store" / "targets.npy", mmap_mode="r")
    return index.set_index("sample_index", drop=False), targets


def build_ensemble(year: Any, seeds: Any, model_root: Any, prediction_dir: Any, index: Any, targets: Any) -> int:
    """按既定尺度组合多个随机种子的预测，并保存正式年度信号。"""
    seed_frames = {}
    for seed in seeds:
        path = model_root / f"forecast_{year}" / f"seed_{seed}" / "prediction.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        seed_frames[seed] = pd.read_parquet(path)
    reference = seed_frames[seeds[0]]["sample_index"].to_numpy(np.int64)
    output = index.loc[
        reference,
        ["sample_index", "trade_date", "instrument_id", "wind_code",
         "can_enter_t1", "label_valid"],
    ].reset_index(drop=True)
    output["target_zscore"] = np.asarray(targets[reference], dtype=np.float32)
    raw_columns = []
    for seed in seeds:
        frame = seed_frames[seed]
        if not np.array_equal(reference, frame["sample_index"].to_numpy(np.int64)):
            raise AssertionError(f"{year} 三个 seed 的预测行号不一致")
        column = f"prediction_seed_{seed}"
        output[column] = frame["prediction"].to_numpy(np.float32)
        raw_columns.append(column)
    raw = output[raw_columns].astype(np.float64)
    grouped = raw.groupby(output["trade_date"], sort=False)
    means = grouped.transform("mean")  # 按约定截面口径汇总
    standard_deviations = grouped.transform("std", ddof=0)  # 按约定截面口径汇总
    if (~np.isfinite(standard_deviations) | standard_deviations.le(0)).any().any():
        raise ValueError(f"{year} 存在无法标准化的日度预测")
    zscores = (raw - means) / standard_deviations
    zscore_columns = []
    for column in raw_columns:
        zscore_column = f"{column}_zscore"
        output[zscore_column] = zscores[column].to_numpy(np.float32)
        zscore_columns.append(zscore_column)
    output["prediction_ensemble_raw"] = raw.mean(axis=1).to_numpy(np.float32)
    output["prediction_ensemble"] = output[zscore_columns].mean(axis=1).astype(np.float32)
    audit = output.groupby("trade_date", sort=False)[zscore_columns]
    if (
        audit.mean().abs().to_numpy().max() > 1e-5
        or (audit.std(ddof=0) - 1).abs().to_numpy().max() > 1e-5
    ):
        raise AssertionError(f"{year} seed 预测日度 Z-score 审计失败")
    atomic_parquet(output, prediction_dir / f"vit_predictions.{year}.parquet")  # 原子落盘，避免半成品
    return len(output)


def prediction_counts(data_root: Path, years: tuple[int, ...]) -> dict[int, int]:
    """统计各预测年度的文件、日期和样本数量。"""
    result = {}
    for year in years:
        with np.load(
            data_root / "feature_store" / "rolling_splits" / f"forecast_{year}.npz",
            allow_pickle=False,
        ) as archive:
            result[year] = len(archive["prediction_indices"])
    return result


def split_batch_summary(
    data_root: Path, forecast_year: int, train_batch_mode: str,
    fixed_batch_size: int, loss_name: str,
) -> dict[str, object]:
    """汇总单个滚动切分中的训练、验证和预测批次。"""
    with np.load(
        data_root / "feature_store" / "rolling_splits"
        / f"forecast_{forecast_year}.npz",
        allow_pickle=False,
    ) as archive:
        train_indices = archive["train_indices"].copy()
    if train_batch_mode == "trade_date":
        date_ids = np.load(
            data_root / "feature_store" / "date_ids.npy", mmap_mode="r"
        )
        temporal_mode = loss_name == TEMPORAL_TURNOVER_LOSS
        sampler = TradeDateBatchSampler(  # 按完整交易日截面组 batch
            train_indices, date_ids, seed=0, shuffle=not temporal_mode
        )
        summary = sampler.summary()  # 按约定截面口径汇总
        if temporal_mode:
            summary.update(
                {
                    "reshuffle_each_epoch": False,
                    "gradient_update_mode": "once_per_epoch",
                    "optimizer_steps_per_epoch": 1,
                    "temporal_lag_trade_days": 1,
                    "temporal_anchor_mode": (
                        "previous_prediction_stop_gradient_cached"
                    ),
                    "model_forward_passes_per_train_date": 1,
                    "train_temporal_pair_count": len(sampler) - 1,
                }
            )
        del date_ids
    else:
        batch_count = len(train_indices) // fixed_batch_size
        summary = {
            "train_batch_mode": "fixed_samples",
            "train_batch_count": batch_count,
            "train_date_count": None,
            "train_sample_count": batch_count * fixed_batch_size,
            "train_batch_size_min": fixed_batch_size,
            "train_batch_size_median": float(fixed_batch_size),
            "train_batch_size_max": fixed_batch_size,
            "dropped_train_samples": (
                len(train_indices) - batch_count * fixed_batch_size
            ),
            "reshuffle_each_epoch": True,
            "gradient_update_mode": "per_batch",
            "optimizer_steps_per_epoch": batch_count,
        }
    return {"forecast_year": forecast_year, **summary}


def training_command(args: Any, command: str) -> list[str]:
    """构造启动训练后端的完整命令行参数。"""
    return [
        sys.executable, str(PIPELINE_PATH), command,
        "--data-root", args.data_root,
        "--run-name", args.run_name,
        "--device-ids", args.device_ids,
        "--tasks-per-gpu", str(args.tasks_per_gpu),
        "--years", args.years,
        "--seeds", args.seeds,
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
    ]


def worker_command(args: Any, year: int, seed: int, gpu: int) -> list[str]:
    """构造单个年度×seed worker 的命令行参数。"""
    return [
        *training_command(args, "worker"),
        "--worker-year", str(year),
        "--worker-seed", str(seed),
        "--worker-physical-gpu", str(gpu),
    ]


def preflight_worker_command(args: Any, rank: int) -> list[str]:
    """构造正式训练前单 worker 预检命令。"""
    return [
        *training_command(args, "preflight-worker"),
        "--preflight-rank", str(rank),
    ]


def tail(path: Path, lines: int = 80) -> str:
    """读取文本文件末尾若干行，用于展示最新训练日志。"""
    if not path.exists():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


def run_worker(args: Any) -> None:
    """启动并等待单个训练 worker，传递其退出状态。"""
    configure_worker_environment(args)
    initialize_runtime()
    model, summary = train_single_run(args.worker_year, args.worker_seed)
    split = load_split(args.worker_year)
    prediction_count = len(split["prediction_indices"])
    summary["prediction_sample_count"] = prediction_count
    if prediction_count:
        _, _, loader = make_loaders(args.worker_year, args.worker_seed)
        if loader is None:
            raise AssertionError("预测索引非空但 loader 缺失")
        prediction = predict_model(model, loader)
        atomic_parquet(  # 原子落盘，避免半成品
            prediction,
            MODEL_ROOT / f"forecast_{args.worker_year}" / f"seed_{args.worker_seed}"
            / "prediction.parquet",
        )
    print(json.dumps({"summary": summary}, ensure_ascii=False), flush=True)


def finalize_outputs(args: Any, elapsed_seconds: float) -> pd.DataFrame:
    """聚合已完成任务并生成 run 级预测、清单和报告。"""
    data_root = Path(args.data_root).resolve()
    years, seeds = parse_csv_ints(args.years), parse_csv_ints(args.seeds)
    model_root = VIT_DIR / "models" / "runs" / args.run_name
    prediction_dir = data_root / "predictions" / "runs" / args.run_name
    quality_dir = data_root / "quality" / "runs" / args.run_name
    counts = prediction_counts(data_root, years)
    records = []
    for year in years:
        for seed in seeds:
            completed_path = model_root / f"forecast_{year}" / f"seed_{seed}" / "completed.json"
            completed = read_json(completed_path)
            if completed is None:
                raise FileNotFoundError(completed_path)
            record = dict(completed["summary"])
            record["prediction_sample_count"] = counts[year]
            records.append(record)
    summary = pd.DataFrame(records).sort_values(["forecast_year", "seed"])
    atomic_parquet(summary, quality_dir / "vit_training_summary.parquet")  # 原子落盘，避免半成品
    index, targets = load_ensemble_inputs(data_root)
    for year in years:
        if not counts[year]:
            continue
        count = build_ensemble(year, seeds, model_root, prediction_dir, index, targets)
        if count != counts[year]:
            raise AssertionError(f"{year} 集成预测行数不完整")
        print(f"[{year}] 集成预测 {count:,} 条", flush=True)
    dataset = read_json(data_root / "feature_store" / "dataset_manifest.json")
    rolling = read_json(data_root / "feature_store" / "rolling_split_manifest.json")
    universe = read_json(data_root / "universe" / "universe_manifest.json")
    atomic_json(  # 原子落盘，避免半成品
        {
            "route": "A2",
            "run_name": args.run_name,
            "device_ids": list(parse_csv_ints(args.device_ids)),
            "tasks_per_gpu": args.tasks_per_gpu,
            "years": list(years),
            "seeds": list(seeds),
            "train_batch_mode": args.train_batch_mode,
            "batch_size_per_task": (
                "full_trade_date"
                if args.train_batch_mode == "trade_date"
                else args.batch_size
            ),
            "fixed_batch_size": args.batch_size,
            "inference_batch_size": args.inference_batch_size,
            "reshuffle_each_epoch": args.loss_name != TEMPORAL_TURNOVER_LOSS,
            "gradient_update_mode": (
                "once_per_epoch"
                if args.loss_name == TEMPORAL_TURNOVER_LOSS else "per_batch"
            ),
            "optimizer_steps_per_epoch": (
                1 if args.loss_name == TEMPORAL_TURNOVER_LOSS else "per_batch"
            ),
            "temporal_lag_trade_days": (
                1 if args.loss_name == TEMPORAL_TURNOVER_LOSS else None
            ),
            "temporal_anchor_mode": (
                "previous_prediction_stop_gradient_cached"
                if args.loss_name == TEMPORAL_TURNOVER_LOSS else None
            ),
            "model_forward_passes_per_train_date": (
                1 if args.loss_name == TEMPORAL_TURNOVER_LOSS else None
            ),
            "loss_name": args.loss_name,
            "loss_parameters": {
                "rank_temperature": args.rank_temperature,
                "turnover_lambda": args.turnover_lambda,
                "huber_delta": args.huber_delta,
                "smooth_l1_beta": args.smooth_l1_beta,
                "hybrid_weight": args.hybrid_weight,
                "prediction_input": "raw_model_output",
                "target_input": "daily_winsorized_1pct_zscore",
                "prediction_preprocessing_inside_loss": "none",
                "target_preprocessing_inside_loss": "none",
                "rank_target_gradient": False,
                "temporal_prediction_gradients": (
                    "current_only"
                    if args.loss_name == TEMPORAL_TURNOVER_LOSS else None
                ),
                "previous_prediction_gradient": (
                    "stop_gradient"
                    if args.loss_name == TEMPORAL_TURNOVER_LOSS else None
                ),
                "previous_prediction_source": (
                    "cached_previous_date_output"
                    if args.loss_name == TEMPORAL_TURNOVER_LOSS else None
                ),
                "turnover_penalty_direction": (
                    "1_minus_temporal_rank_ic"
                    if args.loss_name == TEMPORAL_TURNOVER_LOSS else None
                ),
            },
            "validation_metric": args.validation_metric,
            "validation_audit_metrics": [
                "daily_mean_spearman_ic",
                "daily_mean_pearson_ic",
                "pooled_pearson_ic",
                "temporal_spearman_rank_ic",
                "temporal_rank_penalty",
            ],
            "validation_results_by_task": summary[
                [
                    "forecast_year", "seed", "best_validation_ic",
                    "best_validation_rank_ic",
                    "best_validation_daily_pearson_ic",
                    "best_validation_pooled_ic",
                    "best_validation_temporal_rank_ic",
                    "best_validation_temporal_penalty",
                ]
            ].to_dict("records"),
            "temporal_training_audit_by_task": summary[
                [
                    "forecast_year", "seed", "train_temporal_pair_count",
                    "temporal_common_count_min", "temporal_common_count_median",
                    "temporal_common_count_max", "temporal_common_coverage_min",
                    "temporal_common_coverage_median",
                    "temporal_common_coverage_max",
                ]
            ].to_dict("records"),
            "training_batch_size_by_year": [
                split_batch_summary(
                    data_root, year, args.train_batch_mode, args.batch_size,
                    args.loss_name,
                )
                for year in years
            ],
            "workers_per_task": args.workers_per_task,
            "learning_rate": args.learning_rate,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "torch_compile": bool_int(args.compile),
            "compile_mode": args.compile_mode,
            "compile_dynamic_batch": bool(
                bool_int(args.compile) and args.train_batch_mode == "trade_date"
            ),
            "fused_adam": bool_int(args.fused_adam),
            "cuda_prefetch": bool_int(args.cuda_prefetch),
            "universe_signature": universe["signature"],
            "dataset_signature": dataset["signature"],
            "rolling_signature": rolling["signature"],
            "elapsed_seconds": elapsed_seconds,
            "completed_tasks": len(summary),
        },
        quality_dir / "a2_training_manifest.json",
    )
    return summary


def run_parent(args: Any) -> pd.DataFrame:
    """调度多个 GPU worker，并在全部结束后汇总正式产物。"""
    device_ids = parse_csv_ints(args.device_ids)
    years, seeds = parse_csv_ints(args.years), parse_csv_ints(args.seeds)
    tasks = deque((year, seed) for year in sorted(years) for seed in sorted(seeds))
    data_root = Path(args.data_root).resolve()
    log_dir = data_root / "quality" / "runs" / args.run_name / "a2_task_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    running: list[dict[str, object]] = []
    started = time.perf_counter()
    print(
        f"A2 启动：{len(tasks)} 个任务，GPU={device_ids}，"
        f"每卡并发={args.tasks_per_gpu}，损失={args.loss_name}，"
        f"batch={args.train_batch_mode}",
        flush=True,
    )
    try:
        while tasks or running:
            for gpu in device_ids:
                active = sum(item["gpu"] == gpu for item in running)
                while tasks and active < args.tasks_per_gpu:
                    year, seed = tasks.popleft()
                    log_path = log_dir / f"forecast_{year}.seed_{seed}.log"
                    log_handle = log_path.open("w", encoding="utf-8")
                    environment = os.environ.copy()
                    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
                    process = subprocess.Popen(  # 启动独立训练任务
                        worker_command(args, year, seed, gpu),
                        cwd=VIT_DIR,
                        env=environment,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    running.append(
                        {"process": process, "gpu": gpu, "year": year, "seed": seed,
                         "log_path": log_path, "log_handle": log_handle,
                         "started": time.perf_counter()}
                    )
                    active += 1
                    print(f"启动 GPU{gpu}: {year}/seed={seed} pid={process.pid}", flush=True)
            finished = []
            for item in running:
                return_code = item["process"].poll()  # 检查训练任务状态
                if return_code is None:
                    continue
                item["log_handle"].close()
                finished.append(item)
                if return_code:
                    for active_item in running:
                        if active_item["process"].poll() is None:
                            active_item["process"].terminate()
                    raise RuntimeError(
                        f"A2 任务失败：{item['year']}/seed={item['seed']} GPU{item['gpu']} "
                        f"exit={return_code}\n{tail(item['log_path'])}"
                    )
                print(
                    f"完成 GPU{item['gpu']}: {item['year']}/seed={item['seed']} "
                    f"{(time.perf_counter() - item['started']) / 60:.2f} 分钟",
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
    elapsed = time.perf_counter() - started
    summary = finalize_outputs(args, elapsed)  # 汇总任务并写出完成标记
    print(f"A2 完成：{len(summary)} 个任务，墙钟 {elapsed / 60:.2f} 分钟")
    return summary


def run_preflight_worker(args: Any) -> None:
    """在一个独立进程内执行多 shape 显存压力测试。"""

    configure_worker_environment(args)
    os.environ["VIT_PREFLIGHT_RANK"] = str(args.preflight_rank)
    initialize_runtime()
    set_random_seed(  # 固定各随机源
        TRAINING_CONFIG.seeds[args.preflight_rank % len(TRAINING_CONFIG.seeds)]
    )
    result = run_full_batch_preflight()
    if result is None:
        raise RuntimeError("并发显存预检要求 CUDA")
    result.insert(0, "concurrent_worker_rank", args.preflight_rank)
    print(
        "VIT_PREFLIGHT_RESULT="
        + json.dumps(result.to_dict("records"), ensure_ascii=False),
        flush=True,
    )


def run_concurrent_temporal_preflight(args: Any) -> pd.DataFrame:
    """在同一物理 GPU 同时启动与正式训练相同数量的 worker。"""

    physical_gpu = parse_csv_ints(args.device_ids)[0]
    worker_count = int(args.tasks_per_gpu)
    if worker_count < 2:
        raise ValueError("同卡并发预检至少需要两个 worker")
    running: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="vit_temporal_preflight_") as temporary:
        temporary_dir = Path(temporary)
        barrier_dir = temporary_dir / "barrier"
        try:
            for rank in range(worker_count):
                log_path = temporary_dir / f"worker_{rank}.log"
                log_handle = log_path.open("w", encoding="utf-8")
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
                environment["VIT_PREFLIGHT_BARRIER_DIR"] = str(barrier_dir)
                environment["VIT_PREFLIGHT_WORLD_SIZE"] = str(worker_count)
                process = subprocess.Popen(  # 启动独立训练任务
                    preflight_worker_command(args, rank),
                    cwd=VIT_DIR,
                    env=environment,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                running.append(
                    {
                        "rank": rank,
                        "process": process,
                        "log_path": log_path,
                        "log_handle": log_handle,
                    }
                )
            deadline = time.monotonic() + 1200
            while running:
                if time.monotonic() >= deadline:
                    raise TimeoutError("同卡并发显存预检超过20分钟")
                finished = []
                for item in running:
                    return_code = item["process"].poll()  # 检查训练任务状态
                    if return_code is None:
                        continue
                    item["log_handle"].close()
                    finished.append(item)
                    log_text = item["log_path"].read_text(
                        encoding="utf-8", errors="replace"
                    )
                    if return_code:
                        raise RuntimeError(
                            f"并发预检 worker={item['rank']} 失败：\n"
                            + "\n".join(log_text.splitlines()[-100:])
                        )
                    marker_lines = [
                        line for line in log_text.splitlines()
                        if line.startswith("VIT_PREFLIGHT_RESULT=")
                    ]
                    if len(marker_lines) != 1:
                        raise RuntimeError(
                            f"并发预检 worker={item['rank']} 缺少结果标记"
                        )
                    payload = marker_lines[0].split("=", 1)[1]
                    worker_records = json.loads(payload)
                    for record in worker_records:
                        record["physical_gpu"] = physical_gpu
                        record["concurrent_tasks_on_gpu"] = worker_count
                    records.extend(worker_records)
                for item in finished:
                    running.remove(item)  # 移除已结束任务
                if running and not finished:
                    time.sleep(0.5)
        finally:
            for item in running:
                if item["process"].poll() is None:
                    item["process"].terminate()  # 终止仍在运行的任务
                item["log_handle"].close()
    result = pd.DataFrame(records).sort_values(
        ["concurrent_worker_rank", "shape_test"]
    )
    if len(result) != worker_count * 12:
        raise AssertionError(
            f"并发预检结果不完整：{len(result)}/{worker_count * 12}"
        )
    return result.reset_index(drop=True)


def run_preflight(args: Any) -> None:
    """执行数据、切分、模型和训练参数的只读预检。"""
    configure_worker_environment(args)
    initialize_runtime()
    validate_training_inputs()
    display(
        pd.DataFrame(
            [validate_rolling_split(year) for year in TRAINING_CONFIG.forecast_years]
        )
    )
    display(configured_batch_summaries())
    display(run_smoke_test())
    if (
        TRAINING_CONFIG.loss_name == TEMPORAL_TURNOVER_LOSS
        and args.tasks_per_gpu > 1
    ):
        display(run_concurrent_temporal_preflight(args))
    else:
        result = run_full_batch_preflight()
        if result is not None:
            display(result)


# ============================== 训练 CLI ==============================
def validate_training_inputs() -> pd.DataFrame:
    """检查 memmap、label_valid 和 8 个滚动年度。"""

    features = np.load(FEATURES_PATH, mmap_mode="r")
    targets = np.load(TARGETS_PATH, mmap_mode="r")
    label_valid = np.load(LABEL_VALID_PATH, mmap_mode="r")
    date_ids = np.load(DATE_IDS_PATH, mmap_mode="r")
    if features.shape[1:] != (17, 8, 8):
        raise ValueError(f"特征形状异常：{features.shape}")
    if not (len(features) == len(targets) == len(label_valid) == len(date_ids) == len(sample_index)):
        raise ValueError("feature store 各文件行数不一致")
    if not np.array_equal(np.isfinite(targets), label_valid.astype(bool)):
        raise AssertionError("label_valid 与 target NaN 契约不一致")
    rolling = pd.read_parquet(ROLLING_SUMMARY_PATH)
    missing_years = set(TRAINING_CONFIG.forecast_years) - set(
        rolling["forecast_year"].astype(int)
    )
    if missing_years:
        raise ValueError(f"缺少滚动年度：{sorted(missing_years)}")
    if DEVICE.type == "cuda":
        capability = torch.cuda.get_device_capability(0)
        if capability == (12, 0) and "sm_120" not in torch.cuda.get_arch_list():
            raise RuntimeError("当前 PyTorch wheel 未包含 RTX 5090 的 sm_120")
        properties = torch.cuda.get_device_properties(0)
        print(
            f"GPU：{properties.name}，显存 {properties.total_memory / 1024**3:.1f}GB，"
            f"计算能力 sm_{capability[0]}{capability[1]}"
        )
    summary = pd.DataFrame(
        [
            {
                "samples": len(features),
                "label_valid": int(label_valid.sum()),
                "feature_shape": str(features.shape),
                "first_date": sample_index["trade_date"].min(),
                "last_date": sample_index["trade_date"].max(),
                "dataset_signature": dataset_manifest["signature"][:12],
                "rolling_signature": rolling_manifest["signature"][:12],
            }
        ]
    )
    display(summary)
    display(rolling)
    del features, targets, label_valid, date_ids
    return summary

def loader_batch_summary(loader: DataLoader) -> dict[str, object]:
    """汇总 DataLoader 的批次数、样本数和批大小分布。"""
    sampler = loader.batch_sampler
    if isinstance(sampler, TradeDateBatchSampler):
        return sampler.summary()
    sample_count = len(loader.dataset)
    batch_size = int(loader.batch_size)
    batch_count = sample_count // batch_size
    return {
        "train_batch_mode": "fixed_samples",
        "train_batch_count": batch_count,
        "train_date_count": None,
        "train_sample_count": batch_count * batch_size,
        "train_batch_size_min": batch_size,
        "train_batch_size_median": float(batch_size),
        "train_batch_size_max": batch_size,
        "dropped_train_samples": sample_count - batch_count * batch_size,
        "reshuffle_each_epoch": True,
        "gradient_update_mode": "per_batch",
        "optimizer_steps_per_epoch": batch_count,
    }

def validate_rolling_split(year: int) -> dict[str, object]:
    """检查训练、验证与预测切分的行号、日期和防泄漏约束。"""
    split = load_split(year)
    train_rows = sample_index.iloc[split["train_indices"]]
    validation_rows = sample_index.iloc[split["validation_indices"]]
    prediction_rows = sample_index.iloc[split["prediction_indices"]]
    if not train_rows["label_valid"].all() or not validation_rows["label_valid"].all():
        raise AssertionError(f"{year} 训练或验证索引混入无标签样本")
    validation_start = validation_rows["trade_date"].min()
    forecast_start = f"{year}0101"
    if train_rows["exit_trade_date"].max() >= validation_start:
        raise AssertionError(f"{year} 训练标签越过验证边界")
    if validation_rows["exit_trade_date"].max() >= forecast_start:
        raise AssertionError(f"{year} 验证标签越过预测边界")
    if len(prediction_rows) and not prediction_rows["trade_date"].str.startswith(str(year)).all():
        raise AssertionError(f"{year} 预测索引混入其他年度")
    return {
        "forecast_year": year,
        "train_samples": len(train_rows),
        "validation_samples": len(validation_rows),
        "prediction_samples": len(prediction_rows),
        "max_train_exit": train_rows["exit_trade_date"].max(),
        "validation_start": validation_start,
        "max_validation_exit": validation_rows["exit_trade_date"].max(),
        "forecast_start": forecast_start,
    }

def configured_batch_summaries() -> pd.DataFrame:
    """按当前批处理配置生成各切分的批次摘要。"""
    records = []
    seed = TRAINING_CONFIG.seeds[0]
    for year in TRAINING_CONFIG.forecast_years:
        train_loader, _, _ = make_loaders(year, seed, workers=0)
        records.append({"forecast_year": year, **loader_batch_summary(train_loader)})
    return pd.DataFrame(records)

def validate_outputs(args: Any) -> pd.DataFrame:
    """检查 checkpoint、预测、诊断和 manifest 是否完整一致。"""
    data_root = Path(args.data_root).resolve()
    years, seeds = parse_csv_ints(args.years), parse_csv_ints(args.seeds)
    model_root = VIT_DIR / "models" / "runs" / args.run_name
    prediction_dir = data_root / "predictions" / "runs" / args.run_name
    quality_dir = data_root / "quality" / "runs" / args.run_name
    sample_index_frame = pd.read_parquet(
        data_root / "feature_store" / "sample_index.parquet",
        columns=["sample_index", "trade_date"],
    )
    sample_index_frame["trade_date"] = sample_index_frame["trade_date"].astype(str)
    counts = prediction_counts(data_root, years)
    records = []
    for year in years:
        for seed in seeds:
            task_dir = model_root / f"forecast_{year}" / f"seed_{seed}"
            if not (task_dir / "best.pt").exists() or not (task_dir / "completed.json").exists():
                raise FileNotFoundError(f"{year}/seed={seed} checkpoint 不完整")
        prediction_path = prediction_dir / f"vit_predictions.{year}.parquet"
        if counts[year]:
            prediction = pd.read_parquet(prediction_path, columns=["sample_index"])
            expected = sample_index_frame.loc[
                sample_index_frame["trade_date"].str.startswith(str(year)), "sample_index"
            ].to_numpy(np.int64)
            if not np.array_equal(prediction["sample_index"].to_numpy(np.int64), expected):
                raise AssertionError(f"{year} 预测键与 T 日 Universe 不一致")
        elif prediction_path.exists():
            raise AssertionError(f"{year} 无图片却出现预测文件")
        records.append(
            {
                "forecast_year": year,
                "checkpoints": len(seeds),
                "prediction_samples": counts[year],
                "prediction_file": prediction_path.exists(),
                "status": "模型与预测完成" if counts[year] else "模型完成，暂无图片预测",
            }
        )
    summary_path = quality_dir / "vit_training_summary.parquet"
    manifest_path = quality_dir / "a2_training_manifest.json"
    if not summary_path.exists() or not manifest_path.exists():
        raise FileNotFoundError("A2 训练汇总或 manifest 缺失")
    summary = pd.read_parquet(summary_path)
    if len(summary[["forecast_year", "seed"]].drop_duplicates()) != len(years) * len(seeds):
        raise AssertionError("A2 年度×种子任务数不完整")
    result = pd.DataFrame(records)
    display(result)
    return result

def show_status(args: Any) -> pd.DataFrame:
    """显示年度任务、worker、checkpoint 和预测的当前状态。"""
    years, seeds = parse_csv_ints(args.years), parse_csv_ints(args.seeds)
    model_root = VIT_DIR / "models" / "runs" / args.run_name
    records = []
    for year in years:
        for seed in seeds:
            task_dir = model_root / f"forecast_{year}" / f"seed_{seed}"
            completed = read_json(task_dir / "completed.json")
            history_path = task_dir / "history.parquet"
            last_epoch, best_ic = None, None
            if completed:
                last_epoch = completed["summary"].get("last_epoch")
                best_ic = completed["summary"].get("best_validation_ic")
            elif history_path.exists():
                history = pd.read_parquet(history_path)
                if len(history):
                    last_epoch = int(history["epoch"].iloc[-1])
                    best_ic = float(history["best_validation_ic"].iloc[-1])
            records.append(
                {"forecast_year": year, "seed": seed, "completed": bool(completed),
                 "last_epoch": last_epoch, "best_validation_ic": best_ic}
            )
    result = pd.DataFrame(records)
    print(f"完成任务：{int(result['completed'].sum())}/{len(result)}")
    display(result)
    return result

def validate_a2_args(args: argparse.Namespace) -> None:
    """校验 A2 训练任务参数之间的组合约束。"""
    args.run_name = validate_run_name(args.run_name)
    if (
        args.tasks_per_gpu < 1
        or args.batch_size <= 1
        or args.inference_batch_size <= 1
        or args.workers_per_task < 0
    ):
        raise ValueError("A2 并发、batch 或 workers 参数非法")
    if args.learning_rate <= 0 or args.max_epochs < 1 or args.patience < 1:
        raise ValueError("学习率、max_epochs 和 patience 必须为正")
    if args.train_batch_mode not in TRAIN_BATCH_MODES:
        raise ValueError(f"未知训练 batch 模式：{args.train_batch_mode}")
    if args.loss_name not in LOSS_NAMES:
        raise ValueError(f"未知损失函数：{args.loss_name}")
    if args.validation_metric not in VALIDATION_METRICS:
        raise ValueError(f"未知验证指标：{args.validation_metric}")
    if (
        args.rank_temperature <= 0
        or args.turnover_lambda < 0
        or args.huber_delta <= 0
        or args.smooth_l1_beta <= 0
        or not 0 <= args.hybrid_weight <= 1
    ):
        raise ValueError("损失温度、换手权重、delta、beta 或混合权重非法")
    if (
        args.loss_name == TEMPORAL_TURNOVER_LOSS
        and args.train_batch_mode != "trade_date"
    ):
        raise ValueError(
            "sigmoid_rank_ic_turnover 必须使用 trade_date batch"
        )
    if (
        args.loss_name == TEMPORAL_TURNOVER_LOSS
        and bool(args.compile)
        and args.compile_mode == "reduce-overhead"
    ):
        raise ValueError(
            "时序换手损失的动态日截面禁止使用 reduce-overhead；"
            "该模式会启用 CUDAGraph 私有显存池，请使用 compile-mode=default"
        )
    if (
        args.command in {"preflight", "preflight-worker", "a2", "worker"}
        and args.run_name in READ_ONLY_TRAINING_RUN_NAMES
    ):
        raise ValueError(
            f"{args.run_name} 是只读历史实验，请为调参训练设置新的 run name"
        )
    device_ids = parse_csv_ints(args.device_ids)
    if device_ids != (0, 1, 2, 3):
        raise ValueError("正式 A2 固定使用 GPU 0,1,2,3")
    parse_csv_ints(args.years)
    parse_csv_ints(args.seeds)
    if not Path(args.data_root).resolve().is_dir():
        raise FileNotFoundError(args.data_root)
    if args.command == "preflight-worker" and args.preflight_rank < 0:
        raise ValueError("preflight-rank 必须为非负整数")


def add_a2_arguments(parser: argparse.ArgumentParser, worker: bool = False) -> None:
    """向命令行解析器注册 A2 训练参数。"""
    parser.add_argument("--data-root", default=str(VIT_DIR / "data"))
    parser.add_argument("--run-name", default=DEFAULT_TUNING_RUN_NAME)
    parser.add_argument("--device-ids", default="0,1,2,3")
    parser.add_argument("--tasks-per-gpu", type=int, default=2)
    parser.add_argument("--years", default=DEFAULT_YEARS)
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
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
        default="daily_mean_spearman_ic",
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
    if worker:
        parser.add_argument("--worker-year", type=int, required=True)
        parser.add_argument("--worker-seed", type=int, required=True)
        parser.add_argument("--worker-physical-gpu", type=int, required=True)


def parse_args() -> argparse.Namespace:
    """解析阶段 07 的训练、预检、验收和状态命令。"""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "a2", "validate", "status"):
        add_a2_arguments(commands.add_parser(name))
    add_a2_arguments(commands.add_parser("worker"), worker=True)
    preflight_worker = commands.add_parser("preflight-worker")
    add_a2_arguments(preflight_worker)
    preflight_worker.add_argument("--preflight-rank", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    """分派 ViT 训练、预检、验收或状态查询流程。"""
    args = parse_args()
    validate_a2_args(args)
    if args.command == "preflight": run_preflight(args)
    elif args.command == "preflight-worker": run_preflight_worker(args)
    elif args.command == "a2":
        run_parent(args)
        validate_outputs(args)
    elif args.command == "worker": run_worker(args)
    elif args.command == "validate": validate_outputs(args)
    elif args.command == "status": show_status(args)


if __name__ == "__main__":
    main()
