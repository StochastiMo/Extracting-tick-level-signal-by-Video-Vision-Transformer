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


PIPELINE_PATH = VIT_DIR / "vit_stage07_training.py"  # 定位pipeline路径。
BASELINE_A2_RUN_NAME = "baseline"  # 设置baselinea2执行名称。
DEFAULT_A2_RUN_NAME = "loss_tuning"  # 设置默认a2执行名称。
DEFAULT_TUNING_RUN_NAME = DEFAULT_A2_RUN_NAME  # 设置默认tuning执行名称。
# 构造去重集合，并保存到 `READ_ONLY_TRAINING_RUN_NAMES`。
READ_ONLY_TRAINING_RUN_NAMES = {BASELINE_A2_RUN_NAME, "date_batch"}  # 集中声明读取only训练执行名称。
DEFAULT_YEARS = "2018,2019,2020,2021,2022,2023,2024,2025"  # 集中声明默认年份。
DEFAULT_SEEDS = "42,43,44"  # 设置默认随机种子。
TRAIN_BATCH_MODES = ("fixed_samples", "trade_date")  # 设置训练批次modes。
TEMPORAL_TURNOVER_LOSS = ranking.TEMPORAL_TURNOVER_LOSS  # 设置时序turnover损失。
LOSS_NAMES = ranking.LOSS_NAMES  # 集中声明损失名称。
STANDARD_RANK_LOSS_NAMES = ranking.STANDARD_RANK_LOSS_NAMES  # 集中声明standard排名损失名称。
RANK_LOSS_NAMES = (*STANDARD_RANK_LOSS_NAMES, TEMPORAL_TURNOVER_LOSS)  # 集中声明排名损失名称。
VALIDATION_METRICS = (  # 设置验证指标。
    "daily_mean_spearman_ic",
    "daily_mean_pearson_ic",
)

pearson_correlation = ranking.pearson_correlation  # 计算Pearson相关系数。
negative_pearson_ic = ranking.negative_pearson_ic
sigmoid_soft_rank = ranking.sigmoid_soft_rank  # 计算sigmoidsoft排名。
neural_sort_soft_rank = ranking.neural_sort_soft_rank  # 计算neuralsortsoft排名。
differentiable_rank_ic = ranking.differentiable_rank_ic  # 计算differentiable排名ic。
temporal_turnover_loss_components = (  # 计算时序turnover损失components。
    ranking.temporal_turnover_loss_components
)


# 使用 `dataclass(frozen=True)` 包装 `TrainingConfig`，并定义其可复用的处理逻辑。
@dataclass(frozen=True)  # 自动生成配置对象的初始化与比较方法。
class TrainingConfig:  # 集中保存训练超参数与运行口径。
    train_batch_mode: str  # 接收训练批次模式。
    batch_size: int  # 接收批次大小。
    inference_batch_size: int  # 接收inference批次大小。
    loss_name: str  # 接收损失名称。
    rank_temperature: float  # 接收排名temperature。
    turnover_lambda: float  # 接收turnoverlambda。
    huber_delta: float  # 接收huberdelta。
    smooth_l1_beta: float  # 接收smoothl1beta。
    hybrid_weight: float  # 接收hybrid权重。
    validation_metric: str  # 接收验证指标。
    learning_rate: float  # 接收learning比率。
    max_epochs: int  # 接收maxepochs。
    early_stopping_patience: int  # 接收earlystoppingpatience。
    seeds: tuple[int, ...]
    forecast_years: tuple[int, ...]
    adam_beta1: float = 0.9  # 接收adambeta1。
    adam_beta2: float = 0.999  # 接收adambeta2。
    adam_epsilon: float = 1e-8  # 接收adamepsilon。
    weight_decay: float = 0.0  # 接收权重decay。
    improvement_tolerance: float = 1e-6  # 接收improvementtolerance。


def parse_csv_ints(value: str) -> tuple[int, ...]:  # 解析逗号分隔整数列表并返回去重后的顺序结果。
    """解析逗号分隔整数列表并返回去重后的顺序结果。"""
    result = tuple(int(part.strip()) for part in value.split(",") if part.strip())  # 初始化结果容器。
    if not result or len(set(result)) != len(result):  # 检查当前样本数量是否满足计算要求。
        raise ValueError(f"整数列表为空或重复：{value}")  # 输入数据不满足业务或数值约束时立即终止。
    return result  # 返回结果。


def bool_int(value: int) -> bool:  # 把命令行中的 0/1 参数转换为布尔值。
    """把命令行中的 0/1 参数转换为布尔值。"""
    return bool(int(value))  # 返回完成上述格式转换后的结果。


def configure_worker_environment(args: Any) -> None:  # 把 CLI 参数发布为单个 A2 worker 的运行参数。
    """把 CLI 参数发布为单个 A2 worker 的运行参数。"""

    os.environ.update(  # 用当前配置更新参数字典。
        {
            "VIT_DATA_ROOT": str(Path(args.data_root).resolve()),  # 记录vit数据根目录。
            "VIT_TRAIN_RUN_NAME": args.run_name,  # 记录vit训练运行名称。
            "VIT_TRAIN_BATCH_MODE": args.train_batch_mode,  # 记录vit训练批次模式。
            "VIT_GLOBAL_BATCH_SIZE": str(args.batch_size),  # 记录vit全局批次大小。
            "VIT_INFERENCE_BATCH_SIZE": str(args.inference_batch_size),  # 记录vitinference批次大小。
            "VIT_LOSS_NAME": args.loss_name,  # 记录vit损失名称。
            "VIT_RANK_TEMPERATURE": str(args.rank_temperature),  # 记录vit秩temperature。
            "VIT_TURNOVER_LAMBDA": str(args.turnover_lambda),  # 记录vitturnoverlambda。
            "VIT_HUBER_DELTA": str(args.huber_delta),  # 记录vithuberdelta。
            "VIT_SMOOTH_L1_BETA": str(args.smooth_l1_beta),  # 记录vitsmoothl1beta。
            "VIT_HYBRID_WEIGHT": str(args.hybrid_weight),  # 记录vithybrid权重。
            "VIT_VALIDATION_METRIC": args.validation_metric,  # 记录vit验证指标。
            "VIT_NUM_WORKERS": str(args.workers_per_task),  # 记录vitnumworkers。
            "VIT_LEARNING_RATE": str(args.learning_rate),  # 记录vitlearning比例。
            "VIT_MAX_EPOCHS": str(args.max_epochs),  # 记录vitmaxepochs。
            "VIT_EARLY_STOPPING_PATIENCE": str(args.patience),  # 记录vitearlystoppingpatience。
            "VIT_FORECAST_YEARS": args.years,  # 记录vitforecast年份。
            "VIT_TRAIN_SEEDS": args.seeds,  # 记录vit训练随机种子。
            "VIT_USE_TORCH_COMPILE": str(args.compile),  # 记录vitusetorchcompile。
            "VIT_TORCH_COMPILE_MODE": args.compile_mode,  # 记录vittorchcompile模式。
            "VIT_USE_FUSED_ADAM": str(args.fused_adam),  # 记录vitusefusedadam。
            "VIT_USE_CUDA_PREFETCH": str(args.cuda_prefetch),  # 记录vitusecudaprefetch。
            "VIT_TRAIN_OVERWRITE": str(args.overwrite),  # 记录vit训练overwrite。
            "VIT_A2_WORKER_PROCESS": "1",  # 把 VIT_A2_WORKER_PROCESS 映射为"1"。
        }
    )


def initialize_runtime() -> None:  # 初始化一个独立单 GPU 训练任务的全局上下文。
    """初始化一个独立单 GPU 训练任务的全局上下文。"""

    global DATA_ROOT, STORE_DIR, FEATURES_PATH, TARGETS_PATH, LABEL_VALID_PATH
    global DATE_IDS_PATH, SAMPLE_INDEX_PATH, DATASET_MANIFEST_PATH
    global ROLLING_MANIFEST_PATH, ROLLING_SUMMARY_PATH, SPLIT_DIR
    global RUN_NAME, MODEL_ROOT, PREDICTION_DIR, QUALITY_DIR, TRAINING_SUMMARY_PATH
    global MODEL_CONFIG, TRAINING_CONFIG, NUM_WORKERS, DEVICE, PIN_MEMORY
    global USE_BF16, USE_COMPILE, COMPILE_MODE, USE_FUSED_ADAM, USE_CUDA_PREFETCH
    global OVERWRITE_TRAINING, PARALLEL_BACKEND, dataset_manifest, rolling_manifest
    global sample_index, targets_memmap, VISIBLE_GPU_COUNT

    DATA_ROOT = get_data_root().resolve()  # 定位数据根目录。
    STORE_DIR = DATA_ROOT / "feature_store"  # 定位存储目录。
    FEATURES_PATH = STORE_DIR / "features.npy"  # 定位特征路径。
    TARGETS_PATH = STORE_DIR / "targets.npy"  # 定位标签路径。
    LABEL_VALID_PATH = STORE_DIR / "label_valid.npy"  # 定位标签有效路径。
    DATE_IDS_PATH = STORE_DIR / "date_ids.npy"  # 定位交易日ids路径。
    SAMPLE_INDEX_PATH = STORE_DIR / "sample_index.parquet"  # 定位样本索引路径。
    DATASET_MANIFEST_PATH = STORE_DIR / "dataset_manifest.json"  # 定位dataset清单路径。
    ROLLING_MANIFEST_PATH = STORE_DIR / "rolling_split_manifest.json"  # 定位rolling清单路径。
    ROLLING_SUMMARY_PATH = STORE_DIR / "rolling_split_summary.parquet"  # 定位rolling汇总路径。
    SPLIT_DIR = STORE_DIR / "rolling_splits"  # 定位切分目录。
    RUN_NAME = os.getenv("VIT_TRAIN_RUN_NAME", DEFAULT_A2_RUN_NAME).strip()  # 设置执行名称。
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")  # 初始化allowed容器。
    if not RUN_NAME or RUN_NAME in {".", ".."} or any(x not in allowed for x in RUN_NAME):  # 检查当前值是否属于允许集合。
        raise ValueError("训练运行名含非法字符")  # 输入数据不满足业务或数值约束时立即终止。
    MODEL_ROOT = VIT_DIR / "models" / "runs" / RUN_NAME  # 定位模型根目录。
    PREDICTION_DIR = DATA_ROOT / "predictions" / "runs" / RUN_NAME  # 定位预测目录。
    QUALITY_DIR = DATA_ROOT / "quality" / "runs" / RUN_NAME  # 定位质量目录。
    TRAINING_SUMMARY_PATH = QUALITY_DIR / "vit_training_summary.parquet"  # 定位训练汇总路径。
    for directory in (MODEL_ROOT, PREDICTION_DIR, QUALITY_DIR):  # 遍历模型根目录预测目录质量目录，逐项处理目录。
        directory.mkdir(parents=True, exist_ok=True)  # 确保目标目录存在。

    MODEL_CONFIG = VitModelConfig()  # 设置模型配置。
    TRAINING_CONFIG = TrainingConfig(  # 设置训练配置。
        train_batch_mode=os.getenv(
            "VIT_TRAIN_BATCH_MODE", "trade_date"
        ).strip(),
        batch_size=int(os.getenv("VIT_GLOBAL_BATCH_SIZE", "5000")),  # 设置每个优化批次的样本数。
        inference_batch_size=int(
            os.getenv("VIT_INFERENCE_BATCH_SIZE", "5000")
        ),
        loss_name=os.getenv("VIT_LOSS_NAME", "pearson_ic").strip(),  # 配置损失名称。
        rank_temperature=float(os.getenv("VIT_RANK_TEMPERATURE", "0.1")),  # 配置排名temperature。
        turnover_lambda=float(os.getenv("VIT_TURNOVER_LAMBDA", "0.1")),  # 配置turnoverlambda。
        huber_delta=float(os.getenv("VIT_HUBER_DELTA", "1.0")),  # 配置huberdelta。
        smooth_l1_beta=float(os.getenv("VIT_SMOOTH_L1_BETA", "0.1")),  # 配置smoothl1beta。
        hybrid_weight=float(os.getenv("VIT_HYBRID_WEIGHT", "0.5")),  # 配置hybrid权重。
        validation_metric=os.getenv(  # 计算验证指标。
            "VIT_VALIDATION_METRIC", "daily_mean_spearman_ic"
        ).strip(),
        learning_rate=float(os.getenv("VIT_LEARNING_RATE", "0.001")),  # 配置learning比率。
        max_epochs=int(os.getenv("VIT_MAX_EPOCHS", "100")),  # 配置maxepochs。
        early_stopping_patience=int(
            os.getenv("VIT_EARLY_STOPPING_PATIENCE", "10")
        ),
        seeds=parse_csv_ints(os.getenv("VIT_TRAIN_SEEDS", "42,43,44")),  # 配置随机种子。
        forecast_years=parse_csv_ints(
            os.getenv(
                "VIT_FORECAST_YEARS", "2018,2019,2020,2021,2022,2023,2024,2025"
            )
        ),
    )
    if TRAINING_CONFIG.train_batch_mode not in TRAIN_BATCH_MODES:  # 检查当前值是否属于允许集合。
        raise ValueError(  # 输入数据不满足业务或数值约束时立即终止。
            f"未知训练 batch 模式：{TRAINING_CONFIG.train_batch_mode}"
        )
    if TRAINING_CONFIG.loss_name not in LOSS_NAMES:  # 检查当前值是否属于允许集合。
        raise ValueError(f"未知损失函数：{TRAINING_CONFIG.loss_name}")  # 输入数据不满足业务或数值约束时立即终止。
    if TRAINING_CONFIG.validation_metric not in VALIDATION_METRICS:  # 检查当前值是否属于允许集合。
        raise ValueError(  # 输入数据不满足业务或数值约束时立即终止。
            f"未知验证指标：{TRAINING_CONFIG.validation_metric}"
        )
    if (  # 开始组合多项有效性或配置条件。
        TRAINING_CONFIG.rank_temperature <= 0
        or TRAINING_CONFIG.turnover_lambda < 0
        or TRAINING_CONFIG.huber_delta <= 0
        or TRAINING_CONFIG.smooth_l1_beta <= 0
        or not 0 <= TRAINING_CONFIG.hybrid_weight <= 1
    ):
        raise ValueError("损失温度、换手权重、delta、beta 或混合权重非法")  # 输入数据不满足业务或数值约束时立即终止。
    if (  # 开始组合多项有效性或配置条件。
        TRAINING_CONFIG.loss_name == TEMPORAL_TURNOVER_LOSS
        and TRAINING_CONFIG.train_batch_mode != "trade_date"
    ):
        raise ValueError(  # 输入数据不满足业务或数值约束时立即终止。
            "sigmoid_rank_ic_turnover 必须使用 trade_date batch"
        )
    if (  # 开始组合多项有效性或配置条件。
        TRAINING_CONFIG.batch_size <= 1
        or TRAINING_CONFIG.inference_batch_size <= 1
        or TRAINING_CONFIG.learning_rate <= 0
    ):
        raise ValueError("batch_size、inference_batch_size 和 learning_rate 必须为正")  # 输入数据不满足业务或数值约束时立即终止。
    if (  # 开始组合多项有效性或配置条件。
        TRAINING_CONFIG.max_epochs < 1
        or TRAINING_CONFIG.early_stopping_patience < 1
    ):
        raise ValueError("max_epochs 和 patience 必须为正")  # 输入数据不满足业务或数值约束时立即终止。
    NUM_WORKERS = int(os.getenv("VIT_NUM_WORKERS", "2"))  # 设置numworkers。
    if NUM_WORKERS < 0:  # 检查当前数值是否越过业务或数值阈值。
        raise ValueError("workers 不能为负数")  # 输入数据不满足业务或数值约束时立即终止。
    USE_COMPILE = read_bool_environment("VIT_USE_TORCH_COMPILE", True)  # 设置usecompile。
    COMPILE_MODE = os.getenv("VIT_TORCH_COMPILE_MODE", "default")  # 设置compile模式。
    if (  # 开始组合多项有效性或配置条件。
        TRAINING_CONFIG.loss_name == TEMPORAL_TURNOVER_LOSS
        and USE_COMPILE
        and COMPILE_MODE == "reduce-overhead"
    ):
        raise ValueError(  # 输入数据不满足业务或数值约束时立即终止。
            "时序换手损失的动态日截面禁止使用 reduce-overhead；"
            "请改用 compile mode=default"
        )
    USE_FUSED_ADAM = read_bool_environment("VIT_USE_FUSED_ADAM", True)  # 设置usefusedadam。
    USE_CUDA_PREFETCH = read_bool_environment("VIT_USE_CUDA_PREFETCH", True)  # 设置usecudaprefetch。
    OVERWRITE_TRAINING = read_bool_environment("VIT_TRAIN_OVERWRITE", False)  # 设置overwrite训练。
    VISIBLE_GPU_COUNT = torch.cuda.device_count() if torch.cuda.is_available() else 0  # 设置visiblegpu数量。
    if torch.cuda.is_available():  # 根据当前数据状态选择处理分支。
        DEVICE = torch.device("cuda:0")  # 设置设备。
    else:  # 处理上述条件均不成立的情况。
        DEVICE = torch.device("cpu")  # 设置设备。
    PIN_MEMORY = DEVICE.type == "cuda"
    USE_BF16 = bool(DEVICE.type == "cuda" and torch.cuda.is_bf16_supported())
    PARALLEL_BACKEND = "A2_independent_task_worker"  # 设置parallelbackend。

    required = [  # 初始化必需容器。
        FEATURES_PATH, TARGETS_PATH, LABEL_VALID_PATH, DATE_IDS_PATH,
        SAMPLE_INDEX_PATH, DATASET_MANIFEST_PATH, ROLLING_MANIFEST_PATH,
        ROLLING_SUMMARY_PATH,
    ]
    missing = [path for path in required if not path.exists()]  # 初始化missing容器。
    if missing:  # 根据当前数据状态选择处理分支。
        raise FileNotFoundError(f"缺少训练输入：{missing}")  # 目标文件不存在时立即终止。
    dataset_manifest = read_json(DATASET_MANIFEST_PATH)  # 读取dataset清单。
    rolling_manifest = read_json(ROLLING_MANIFEST_PATH)  # 读取rolling清单。
    sample_index = pd.read_parquet(SAMPLE_INDEX_PATH)  # 读取样本索引。
    sample_index["trade_date"] = sample_index["trade_date"].astype(str)  # 转换样本索引的数据类型或存储位置。
    sample_index["exit_trade_date"] = sample_index["exit_trade_date"].astype(str)  # 转换样本索引的数据类型或存储位置。
    targets_memmap = np.load(TARGETS_PATH, mmap_mode="r")  # 读取标签memmap。

    print(f"Python/PyTorch：{platform.python_version()} / {torch.__version__}")  # 输出当前任务的进度或诊断信息。
    print(f"运行名：{RUN_NAME}")  # 输出当前任务的进度或诊断信息。
    print(f"设备：{DEVICE}；可见 GPU：{VISIBLE_GPU_COUNT}")  # 输出当前任务的进度或诊断信息。
    print(  # 输出当前任务的进度或诊断信息。
        f"训练 batch 模式：{TRAINING_CONFIG.train_batch_mode}；"
        f"固定/预测 batch：{TRAINING_CONFIG.batch_size}/"
        f"{TRAINING_CONFIG.inference_batch_size}；workers/lr："
        f"{NUM_WORKERS}/{TRAINING_CONFIG.learning_rate}"
    )
    temporal_mode = TRAINING_CONFIG.loss_name == TEMPORAL_TURNOVER_LOSS
    print(  # 输出当前任务的进度或诊断信息。
        f"损失函数：{TRAINING_CONFIG.loss_name}；"
        f"tau/turnover_lambda/delta/beta/hybrid_weight："
        f"{TRAINING_CONFIG.rank_temperature}/"
        f"{TRAINING_CONFIG.turnover_lambda}/{TRAINING_CONFIG.huber_delta}/"
        f"{TRAINING_CONFIG.smooth_l1_beta}/{TRAINING_CONFIG.hybrid_weight}"
    )
    print(  # 输出当前任务的进度或诊断信息。
        f"验证选模指标：{TRAINING_CONFIG.validation_metric}；"
        f"reshuffle：{not temporal_mode}；梯度更新："
        f"{'每 epoch 一次' if temporal_mode else '每 batch 一次'}"
    )
    print(  # 输出当前任务的进度或诊断信息。
        f"compile/mode/fused Adam/CUDA prefetch："
        f"{USE_COMPILE}/{COMPILE_MODE}/{USE_FUSED_ADAM}/{USE_CUDA_PREFETCH}"
    )


class SampleIndexDataset(Dataset):  # 用样本行号构造轻量 Dataset。
    def __init__(self, indices: np.ndarray) -> None:  # 初始化当前对象的配置、状态与子组件。
        self.indices = np.asarray(indices, dtype=np.int64)  # 保存索引，供该对象后续步骤复用。

    def __len__(self) -> int:  # 返回 SampleIndexDataset 管理的样本数或批次数。
        """返回 SampleIndexDataset 管理的样本数或批次数。"""
        return len(self.indices)  # 返回 SampleIndexDataset 管理的样本数或批次数。

    def __getitem__(self, position: int) -> int:  # 按给定索引返回 SampleIndexDataset 中的一个样本。
        """按给定索引返回 SampleIndexDataset 中的一个样本。"""
        return int(self.indices[position])  # 返回按当前规则组织后的结果。


class TradeDateBatchSampler(Sampler[list[int]]):  # 把完整交易日作为 batch。
    """把完整交易日作为 batch；普通模式重排日期，时序模式保持升序。"""

    def __init__(  # 初始化当前对象的配置、状态与子组件。
        self, sample_indices: np.ndarray, date_ids: np.ndarray, seed: int,
        shuffle: bool = True, instrument_ids: np.ndarray | None = None,  # 接收shuffle。
    ) -> None:
        indices = np.asarray(sample_indices, dtype=np.int64)  # 转换并统一索引的数据表示。
        if not len(indices):  # 检查当前样本数量是否满足计算要求。
            raise ValueError("日截面 batch 不能使用空训练索引")  # 输入数据不满足业务或数值约束时立即终止。
        selected_dates = np.asarray(date_ids[indices], dtype=np.int64)  # 转换并统一筛选结果交易日的数据表示。
        if np.any(selected_dates[1:] < selected_dates[:-1]):  # 检查当前数值是否越过业务或数值阈值。
            raise ValueError("训练索引未按交易日排序，无法构建日截面 batch")  # 输入数据不满足业务或数值约束时立即终止。
        boundaries = np.flatnonzero(selected_dates[1:] != selected_dates[:-1]) + 1
        self.starts = np.r_[0, boundaries].astype(np.int64)  # 保存starts，供该对象后续步骤复用。
        self.stops = np.r_[boundaries, len(indices)].astype(np.int64)  # 保存stops，供该对象后续步骤复用。
        self.date_ids = selected_dates[self.starts]  # 保存交易日ids，供该对象后续步骤复用。
        self.seed = int(seed)  # 保存随机种子，供该对象后续步骤复用。
        self.shuffle = bool(shuffle)  # 保存shuffle，供该对象后续步骤复用。
        self.epoch = 0  # 保存轮次，供该对象后续步骤复用。
        self.temporal_pairs: list[dict[str, object]] | None = None  # 声明可选输入；未提供时沿用默认处理流程。
        if instrument_ids is not None:  # 检查必需数据、配置或校验结果是否缺失。
            if self.shuffle:  # 根据当前数据状态选择处理分支。
                raise ValueError("时序换手损失必须关闭交易日 reshuffle")  # 输入数据不满足业务或数值约束时立即终止。
            if len(self.starts) < 2 or np.any(np.diff(self.date_ids) != 1):  # 检查当前样本数量是否满足计算要求。
                raise ValueError("时序换手损失要求至少两个连续训练交易日")  # 输入数据不满足业务或数值约束时立即终止。
            all_instruments = np.asarray(instrument_ids)  # 转换并统一all股票的数据表示。
            self.temporal_pairs = []  # 保存时序pairs，供该对象后续步骤复用。
            for pair_number in range(1, len(self.starts)):  # 按给定步长遍历样本区间，更新pairnumber。
                previous_slice = slice(
                    int(self.starts[pair_number - 1]),
                    int(self.stops[pair_number - 1]),
                )
                current_slice = slice(
                    int(self.starts[pair_number]), int(self.stops[pair_number])
                )
                previous_ids = all_instruments[indices[previous_slice]]
                current_ids = all_instruments[indices[current_slice]]
                if len(np.unique(previous_ids)) != len(previous_ids):  # 检查当前样本数量是否满足计算要求。
                    raise ValueError(  # 输入数据不满足业务或数值约束时立即终止。
                        f"date_id={self.date_ids[pair_number - 1]} 股票键重复"
                    )
                if len(np.unique(current_ids)) != len(current_ids):  # 检查当前样本数量是否满足计算要求。
                    raise ValueError(  # 输入数据不满足业务或数值约束时立即终止。
                        f"date_id={self.date_ids[pair_number]} 股票键重复"
                    )
                common, previous_positions, current_positions = np.intersect1d(
                    previous_ids, current_ids, assume_unique=True,
                    return_indices=True,  # 返回init。
                )
                if len(common) < 2:  # 检查当前样本数量是否满足计算要求。
                    raise ValueError(  # 输入数据不满足业务或数值约束时立即终止。
                        f"date_id={self.date_ids[pair_number - 1]}→"
                        f"{self.date_ids[pair_number]} 共同股票少于2只"
                    )
                self.temporal_pairs.append(  # 把当前结果追加到汇总列表。
                    {
                        "previous_positions": previous_positions.astype(  # 记录上一期位置。
                            np.int64, copy=False
                        ),
                        "current_positions": current_positions.astype(  # 记录当前位置。
                            np.int64, copy=False
                        ),
                        "common_count": int(len(common)),  # 记录公共数量。
                        "coverage": float(  # 记录覆盖率。
                            len(common) / min(len(previous_ids), len(current_ids))
                        ),
                    }
                )

    def __len__(self) -> int:  # 返回 TradeDateBatchSampler 管理的样本数或批次数。
        """返回 TradeDateBatchSampler 管理的样本数或批次数。"""
        return len(self.starts)  # 返回 TradeDateBatchSampler 管理的样本数或批次数。

    def set_epoch(self, epoch: int) -> None:  # 记录当前 epoch，用于确定性改变训练日顺序。
        """记录当前 epoch，用于确定性改变训练日顺序。"""
        self.epoch = int(epoch)  # 保存轮次，供该对象后续步骤复用。

    def __iter__(self) -> Any:  # 按既定顺序迭代 TradeDateBatchSampler 中的训练批次。
        """按既定顺序迭代 TradeDateBatchSampler 中的训练批次。"""
        if self.shuffle:  # 根据当前数据状态选择处理分支。
            generator = torch.Generator().manual_seed(self.seed + self.epoch)  # 固定 DataLoader 随机源。
            order = torch.randperm(len(self), generator=generator).tolist()
        else:  # 处理上述条件均不成立的情况。
            order = range(len(self))
        for batch_number in order:  # 遍历排序索引，逐项处理批次number。
  # 对 `(yield range(int(self.starts[batch_number]), int(self.stops[batch_number])))` 求值，以触发当前步骤需要的计算。
            yield range(  # 产出当前批次并保留迭代状态。
                int(self.starts[batch_number]), int(self.stops[batch_number])
            )

  # 使用 `property` 包装 `batch_sizes`，并定义其可复用的处理逻辑。
    @property
    def batch_sizes(self) -> np.ndarray:  # 返回各批次包含的样本数量。
        """返回各批次包含的样本数量。"""
        return self.stops - self.starts  # 返回各批次包含的样本数量。

    def batch_slice(self, batch_number: int) -> slice:  # 返回指定批次在全局顺序中的切片边界。
        """返回指定批次在全局顺序中的切片边界。"""
        return slice(  # 返回指定批次在全局顺序中的切片边界。
            int(self.starts[batch_number]), int(self.stops[batch_number])
        )

    def temporal_pair(self, pair_number: int) -> dict[str, object]:  # 返回相邻交易日共同股票的对齐位置。
        """返回相邻交易日共同股票的对齐位置。"""
        if self.temporal_pairs is None:  # 检查所需对象是否尚未初始化。
            raise ValueError("当前 sampler 没有时序相邻截面对齐信息")  # 输入数据不满足业务或数值约束时立即终止。
        return self.temporal_pairs[pair_number]  # 返回相邻交易日共同股票的对齐位置。

    def summary(self) -> dict[str, object]:  # 汇总批采样器的日期、样本和批大小信息。
        """汇总批采样器的日期、样本和批大小信息。"""
        sizes = self.batch_sizes
        result: dict[str, object] = {  # 初始化结果容器。
            "train_batch_mode": "trade_date",  # 把 train_batch_mode 映射为"trade_date"。
            "train_batch_count": len(sizes),  # 记录训练批次数量。
            "train_date_count": len(sizes),  # 记录训练交易日数量。
            "train_sample_count": int(sizes.sum()),  # 记录训练样本数量。
            "train_batch_size_min": int(sizes.min()),  # 记录训练批次大小min。
            "train_batch_size_median": float(np.median(sizes)),  # 记录训练批次大小median。
            "train_batch_size_max": int(sizes.max()),  # 记录训练批次大小max。
            "dropped_train_samples": 0,  # 记录dropped训练样本。
            "reshuffle_each_epoch": self.shuffle,  # 记录reshuffleeach轮次。
            "gradient_update_mode": (  # 记录梯度update模式。
                "once_per_epoch" if self.temporal_pairs is not None else "per_batch"
            ),
            "optimizer_steps_per_epoch": (  # 记录优化器stepsper轮次。
                1 if self.temporal_pairs is not None else len(sizes)
            ),
        }
        if self.temporal_pairs is not None:  # 检查必需数据、配置或校验结果是否缺失。
            common_counts = np.array(  # 计算公共数量。
                [pair["common_count"] for pair in self.temporal_pairs],
                dtype=np.int64,  # 统一结果的数据类型。
            )
            coverages = np.array(
                [pair["coverage"] for pair in self.temporal_pairs],
                dtype=np.float64,  # 统一结果的数据类型。
            )
            result.update(  # 用当前配置更新参数字典。
                {
                    "temporal_lag_trade_days": 1,  # 记录时序lagtradedays。
                    "temporal_anchor_mode": (  # 记录时序anchor模式。
                        "previous_prediction_stop_gradient_cached"
                    ),
                    "model_forward_passes_per_train_date": 1,  # 记录模型forwardpassesper训练交易日。
                    "train_temporal_pair_count": len(self.temporal_pairs),  # 记录训练时序pair数量。
                    "temporal_common_count_min": int(common_counts.min()),  # 记录时序公共数量min。
                    "temporal_common_count_median": float(  # 记录时序公共数量median。
                        np.median(common_counts)
                    ),
                    "temporal_common_count_max": int(common_counts.max()),  # 记录时序公共数量max。
                    "temporal_common_coverage_min": float(coverages.min()),  # 记录时序公共覆盖率min。
                    "temporal_common_coverage_median": float(  # 记录时序公共覆盖率median。
                        np.median(coverages)
                    ),
                    "temporal_common_coverage_max": float(coverages.max()),  # 记录时序公共覆盖率max。
                }
            )
        return result  # 返回结果。


class MemmapBatchCollator:  # 每个 worker 首次取 batch 时才打开只读 memmap。
    """每个 worker 首次取 batch 时才打开只读 memmap。"""

    def __init__(self) -> None:  # 初始化当前对象的配置、状态与子组件。
        self.features = None  # 先把特征初始化为空，后续按条件填充。
        self.targets = None  # 先把标签初始化为空，后续按条件填充。
        self.date_ids = None  # 先把交易日ids初始化为空，后续按条件填充。

    def __call__(self, sample_indices: list[int]) -> dict[str, torch.Tensor]:  # 按样本行号批量读取 memmap 特征和标签，并组装设备传输前的批次。
        """按样本行号批量读取 memmap 特征和标签，并组装设备传输前的批次。"""
        if self.features is None:  # 检查所需对象是否尚未初始化。
            self.features = np.load(FEATURES_PATH, mmap_mode="r")  # 保存特征，供该对象后续步骤复用。
            self.targets = np.load(TARGETS_PATH, mmap_mode="r")  # 保存标签，供该对象后续步骤复用。
            self.date_ids = np.load(DATE_IDS_PATH, mmap_mode="r")  # 保存交易日ids，供该对象后续步骤复用。
        indices = np.asarray(sample_indices, dtype=np.int64)  # 转换并统一索引的数据表示。
        return {  # 返回按当前规则组织后的结果。
            "features": torch.from_numpy(  # 记录特征。
                np.array(self.features[indices], dtype=np.float32, copy=True, order="C")
            ),
            "targets": torch.from_numpy(  # 记录标签。
                np.array(self.targets[indices], dtype=np.float32, copy=True)
            ),
            "date_ids": torch.from_numpy(  # 记录交易日ids。
                np.array(self.date_ids[indices], dtype=np.int32, copy=True)
            ),
            "sample_indices": torch.from_numpy(indices),  # 记录样本索引。
        }


def load_split(forecast_year: int) -> dict[str, np.ndarray]:  # 加载指定预测年度的训练、验证和预测样本行号。
    """加载指定预测年度的训练、验证和预测样本行号。"""
    path = SPLIT_DIR / f"forecast_{forecast_year}.npz"
    if not path.exists():  # 检查目标文件或目录是否满足存在性要求。
        raise FileNotFoundError(path)  # 目标文件不存在时立即终止。
    with np.load(path, allow_pickle=False) as archive:  # 进入受控上下文并在退出时自动释放资源。
        return {name: archive[name].copy() for name in archive.files}  # 返回加载并校验后的结果。


def make_loaders(  # 根据滚动切分创建训练、验证和预测 DataLoader。
    forecast_year: int,  # 接收forecast年份。
    seed: int,  # 接收随机种子。
    batch_size: int | None = None,  # 接收批次大小。
    workers: int | None = None,  # 接收workers。
    inference_batch_size: int | None = None,  # 接收inference批次大小。
    train_batch_mode: str | None = None,  # 接收训练批次模式。
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    """根据滚动切分创建训练、验证和预测 DataLoader。"""
    split = load_split(forecast_year)
    effective_train_batch = (
        TRAINING_CONFIG.batch_size if batch_size is None else batch_size
    )
    if inference_batch_size is not None:  # 检查必需数据、配置或校验结果是否缺失。
        effective_inference_batch = inference_batch_size
    elif batch_size is not None:  # 检查上一条件不成立时的备选分支。
        effective_inference_batch = batch_size
    else:  # 处理上述条件均不成立的情况。
        effective_inference_batch = TRAINING_CONFIG.inference_batch_size
    effective_mode = train_batch_mode or TRAINING_CONFIG.train_batch_mode
    if effective_mode not in TRAIN_BATCH_MODES:  # 检查当前值是否属于允许集合。
        raise ValueError(f"未知训练 batch 模式：{effective_mode}")  # 输入数据不满足业务或数值约束时立即终止。
    temporal_mode = TRAINING_CONFIG.loss_name == TEMPORAL_TURNOVER_LOSS
    if temporal_mode and effective_mode != "trade_date":  # 按当前模式、标识或状态选择处理分支。
        raise ValueError("sigmoid_rank_ic_turnover 必须使用 trade_date batch")  # 输入数据不满足业务或数值约束时立即终止。
    effective_workers = NUM_WORKERS if workers is None else workers  # 计算有效workers判定条件。
    common = {"num_workers": effective_workers, "pin_memory": PIN_MEMORY}  # 初始化公共容器。
    if effective_workers > 0:  # 检查当前数值是否越过业务或数值阈值。
        common.update({"persistent_workers": True, "prefetch_factor": 2})  # 用当前配置更新参数字典。

    train_dataset = SampleIndexDataset(split["train_indices"])  # 创建训练dataset。
    generator = torch.Generator().manual_seed(seed)  # 固定 DataLoader 随机源。
    if effective_mode == "trade_date":  # 按当前模式、标识或状态选择处理分支。
        date_ids = sample_index["date_id"].to_numpy(np.int32, copy=False)  # 转换并统一交易日ids的数据表示。
        batch_sampler = TradeDateBatchSampler(  # 按完整交易日截面组 batch。
            split["train_indices"], date_ids, seed=seed,
            shuffle=not temporal_mode,  # 控制是否打乱样本顺序。
            instrument_ids=(
                sample_index["instrument_id"].to_numpy(copy=False)
                if temporal_mode else None  # 根据当前数据状态选择处理分支。
            ),
        )
        train_loader = DataLoader(  # 按既定采样口径创建 DataLoader。
            train_dataset,
            batch_sampler=batch_sampler,  # 配置批次sampler。
            generator=generator,  # 传入固定随机数生成器以保证复现。
            collate_fn=MemmapBatchCollator(),  # 配置collatefn。
            **common,
        )
    else:  # 处理上述条件均不成立的情况。
        train_loader = DataLoader(  # 按既定采样口径创建 DataLoader。
            train_dataset,
            batch_size=effective_train_batch,  # 设置每个优化批次的样本数。
            shuffle=True,  # 控制是否打乱样本顺序。
            drop_last=True,  # 控制是否丢弃最后一个不完整批次。
            generator=generator,  # 传入固定随机数生成器以保证复现。
            collate_fn=MemmapBatchCollator(),  # 配置collatefn。
            **common,
        )
    validation_loader = DataLoader(  # 按既定采样口径创建 DataLoader。
        SampleIndexDataset(split["validation_indices"]),
        batch_size=effective_inference_batch,  # 设置每个优化批次的样本数。
        shuffle=False,  # 控制是否打乱样本顺序。
        drop_last=False,  # 控制是否丢弃最后一个不完整批次。
        collate_fn=MemmapBatchCollator(),  # 配置collatefn。
        **common,
    )
    prediction_loader = None  # 先把预测数据加载器初始化为空，后续按条件填充。
    if len(split["prediction_indices"]):  # 检查当前样本数量是否满足计算要求。
        prediction_loader = DataLoader(  # 按既定采样口径创建 DataLoader。
            SampleIndexDataset(split["prediction_indices"]),
            batch_size=effective_inference_batch,  # 设置每个优化批次的样本数。
            shuffle=False,  # 控制是否打乱样本顺序。
            drop_last=False,  # 控制是否丢弃最后一个不完整批次。
            collate_fn=MemmapBatchCollator(),  # 配置collatefn。
            **common,
        )
    return train_loader, validation_loader, prediction_loader  # 返回根据滚动切分创建训练、验证和预测 DataLoader。


def set_train_epoch(loader: DataLoader, epoch: int) -> None:  # 把当前 epoch 传递给支持确定性重排的批采样器。
    """把当前 epoch 传递给支持确定性重排的批采样器。"""
    if isinstance(loader.batch_sampler, TradeDateBatchSampler):  # 检查对象类型是否符合接口约定。
        loader.batch_sampler.set_epoch(epoch)  # 同步 epoch，保持采样可复现。


def build_model() -> nn.Module:  # 按当前配置构建模型，并保持历史 checkpoint 的参数键兼容。
    """按当前配置构建模型，并保持历史 checkpoint 的参数键兼容。"""
    model = OrderFlowVisionTransformer(MODEL_CONFIG).to(DEVICE)  # 创建模型。
    if USE_COMPILE:  # 根据当前数据状态选择处理分支。
        if DEVICE.type != "cuda":  # 按当前模式、标识或状态选择处理分支。
            raise RuntimeError("torch.compile 正式配置要求 CUDA")  # 当前运行环境或任务状态不支持继续执行。
        if TRAINING_CONFIG.train_batch_mode == "trade_date":  # 按当前模式、标识或状态选择处理分支。
            model = torch.compile(model, mode=COMPILE_MODE, dynamic=True)  # 编译模型计算图。
        else:  # 处理上述条件均不成立的情况。
            model = torch.compile(model, mode=COMPILE_MODE)  # 编译模型计算图。
    return model  # 返回模型。


def unwrap_model(model: nn.Module) -> nn.Module:  # 从编译或包装模型中取回原始 PyTorch 模型。
    """从编译或包装模型中取回原始 PyTorch 模型。"""
    original = getattr(model, "_orig_mod", None)
    return original if original is not None else model  # 返回完成投影或提取后的结果。


def canonical_state(model: nn.Module) -> Any:  # 提取不含编译包装前缀的规范化模型参数字典。
    """提取不含编译包装前缀的规范化模型参数字典。"""
    state = unwrap_model(model).state_dict()
    if any(key.startswith(("module.", "_orig_mod.")) for key in state):  # 检查当前值是否属于允许集合。
        raise AssertionError("canonical checkpoint 含包装器前缀")  # 关键数据契约被破坏时立即终止。
    return state  # 返回状态。


def load_canonical_state(model: nn.Module, state: Any) -> None:  # 把规范化参数字典加载到原始或包装模型。
    """把规范化参数字典加载到原始或包装模型。"""
    unwrap_model(model).load_state_dict(state)


# 根据当前训练配置补齐默认参数；数值计算统一委托给 vit_stage06_ranking。
def compute_training_loss(  # 解析当前训练配置后，调用无全局状态的共享损失实现。
    predictions: torch.Tensor,  # [B]，预测分数。
    targets: torch.Tensor,  # [B]，训练标签。
    loss_name: str | None = None,  # 接收损失名称。
    rank_temperature: float | None = None,  # 接收排名temperature。
    huber_delta: float | None = None,  # 接收huberdelta。
    smooth_l1_beta: float | None = None,  # 接收smoothl1beta。
    hybrid_weight: float | None = None,  # 接收hybrid权重。
    turnover_lambda: float | None = None,  # 接收turnoverlambda。
    previous_predictions: torch.Tensor | None = None,  # [N_{t-1}]，上一交易日分数。
    previous_positions: np.ndarray | torch.Tensor | None = None,  # [M]，上一日共同股票位置。
    current_positions: np.ndarray | torch.Tensor | None = None,  # [M]，当日共同股票位置。
) -> torch.Tensor:
    """解析当前训练配置后，调用无全局状态的共享损失实现。"""

    return ranking.compute_training_loss(  # 返回解析当前训练配置后，调用无全局状态的共享损失实现。
        predictions,
        targets,
        loss_name=loss_name or TRAINING_CONFIG.loss_name,  # 配置损失名称。
        rank_temperature=(  # 计算排名temperature。
            TRAINING_CONFIG.rank_temperature
            if rank_temperature is None else float(rank_temperature)  # 检查所需对象是否尚未初始化。
        ),
        huber_delta=(
            TRAINING_CONFIG.huber_delta
            if huber_delta is None else float(huber_delta)  # 检查所需对象是否尚未初始化。
        ),
        smooth_l1_beta=(
            TRAINING_CONFIG.smooth_l1_beta
            if smooth_l1_beta is None else float(smooth_l1_beta)  # 检查所需对象是否尚未初始化。
        ),
        hybrid_weight=(
            TRAINING_CONFIG.hybrid_weight
            if hybrid_weight is None else float(hybrid_weight)  # 检查所需对象是否尚未初始化。
        ),
        turnover_lambda=(
            TRAINING_CONFIG.turnover_lambda
            if turnover_lambda is None else float(turnover_lambda)  # 检查所需对象是否尚未初始化。
        ),
        previous_predictions=previous_predictions,  # 配置上一期预测。
        previous_positions=previous_positions,  # 配置上一期位置。
        current_positions=current_positions,  # 配置当前位置。
    )


def set_random_seed(seed: int) -> None:  # 同步设置 Python、NumPy、CPU 与 CUDA 随机种子。
    """同步设置 Python、NumPy、CPU 与 CUDA 随机种子。"""
    random.seed(seed)  # 固定本轮采样种子。
    np.random.seed(seed)  # 固定本轮采样种子。
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # 根据当前数据状态选择处理分支。
        torch.cuda.manual_seed_all(seed)


def autocast_context() -> Any:  # 按设备和配置返回混合精度上下文。
    """按设备和配置返回混合精度上下文。"""
    if USE_BF16:  # 根据当前数据状态选择处理分支。
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)  # 返回按当前规则组织后的结果。
    return contextlib.nullcontext()  # 返回按当前规则组织后的结果。


def move_batch(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:  # 把一个训练批次移动到指定设备，并保持非阻塞传输策略。
    """把一个训练批次移动到指定设备，并保持非阻塞传输策略。"""
    return (  # 返回完成上述格式转换后的结果。
        batch["features"].to(DEVICE, non_blocking=PIN_MEMORY),  # [B,17,8,8]。
        batch["targets"].to(DEVICE, non_blocking=PIN_MEMORY),  # [B]。
    )


class CudaBatchPrefetcher:  # 用独立 CUDA 流预取下一批数据。
    def __init__(self, loader: DataLoader) -> None:  # 初始化当前对象的配置、状态与子组件。
        self.loader = loader  # 保存数据加载器，供该对象后续步骤复用。

    def __iter__(self) -> Any:  # 按既定顺序迭代 CudaBatchPrefetcher 中的训练批次。
        """按既定顺序迭代 CudaBatchPrefetcher 中的训练批次。"""
        if not USE_CUDA_PREFETCH or DEVICE.type != "cuda":  # 按当前模式、标识或状态选择处理分支。
            for batch in self.loader:  # 遍历数据加载器，逐项处理批次。
                features, targets = move_batch(batch)  # 将 batch 搬到训练设备。
  # 对 `(yield (batch, features, targets))` 求值，以触发当前步骤需要的计算。
                yield batch, features, targets  # 产出当前批次并保留迭代状态。
            return  # 返回按当前规则组织后的结果。
        iterator = iter(self.loader)
        stream = torch.cuda.Stream(device=DEVICE)
        next_item = None  # 先把nextitem初始化为空，后续按条件填充。

        def preload() -> None:  # 把下一批数据异步搬到 GPU，供当前批计算后立即使用。
            """把下一批数据异步搬到 GPU，供当前批计算后立即使用。"""
            nonlocal next_item
            try:  # 执行可能失败的外部读取或计算步骤。
                batch = next(iterator)
            except StopIteration:  # 捕获预期异常并执行对应恢复逻辑。
                next_item = None  # 先把nextitem初始化为空，后续按条件填充。
                return  # 返回完成上述格式转换后的结果。
            with torch.cuda.stream(stream):  # 在独立 CUDA 流中预取下一批数据。
                features, targets = move_batch(batch)  # 将 batch 搬到训练设备。
            next_item = (batch, features, targets)

        preload()  # 异步预取下一批数据。
        while next_item is not None:  # 在终止条件出现前持续迭代。
            torch.cuda.current_stream(DEVICE).wait_stream(stream)  # 等待预取流完成。
            batch, features, targets = next_item
            features.record_stream(torch.cuda.current_stream(DEVICE))  # 绑定当前 CUDA stream 生命周期。
            targets.record_stream(torch.cuda.current_stream(DEVICE))  # 绑定当前 CUDA stream 生命周期。
            preload()  # 异步预取下一批数据。
  # 对 `(yield (batch, features, targets))` 求值，以触发当前步骤需要的计算。
            yield batch, features, targets  # 产出当前批次并保留迭代状态。


def build_optimizer(model: nn.Module) -> torch.optim.Adam:  # 创建优化器，并按配置启用 fused Adam。
    """创建优化器，并按配置启用 fused Adam。"""
    return torch.optim.Adam(  # 返回创建优化器，并按配置启用 fused Adam。
        model.parameters(),
        lr=TRAINING_CONFIG.learning_rate,  # 设置优化器学习率。
        betas=(TRAINING_CONFIG.adam_beta1, TRAINING_CONFIG.adam_beta2),  # 配置betas。
        eps=TRAINING_CONFIG.adam_epsilon,  # 设置数值稳定项。
        weight_decay=TRAINING_CONFIG.weight_decay,  # 设置权重衰减强度。
        fused=(USE_FUSED_ADAM and DEVICE.type == "cuda"),  # 配置fused。
    )


def _train_standard_epoch(model: Any, loader: Any, optimizer: Any, max_batches: int | None = None) -> Any:  # 执行该模式下的训练步骤，并返回损失与诊断信息。
    """执行该模式下的训练步骤，并返回损失与诊断信息。"""
    model.train()  # 启用训练模式。
    date_batch_mode = isinstance(loader.batch_sampler, TradeDateBatchSampler)
    loss_sum = torch.zeros((), dtype=torch.float64, device=DEVICE)  # 初始化损失sum。
    batch_pearson_sum = torch.zeros((), dtype=torch.float64, device=DEVICE)  # 初始化批次Pearsonsum。
    weighted_pearson = torch.zeros((), dtype=torch.float64, device=DEVICE)  # 初始化weightedPearson。
    soft_rank_ic_sum = torch.zeros((), dtype=torch.float64, device=DEVICE)  # 初始化soft排名icsum。
    sample_count = 0  # 计算样本数量。
    batch_count = 0  # 计算批次数量。
    soft_rank_batch_count = 0  # 计算soft排名批次数量。
    min_batch_size, max_batch_size = None, 0
    started = time.perf_counter()
    for batch_number, (batch, features, targets) in enumerate(  # 遍历enumerate，逐项处理批次number批次特征标签。
        CudaBatchPrefetcher(loader), start=1
    ):
        if date_batch_mode:  # 根据当前数据状态选择处理分支。
            date_ids = batch["date_ids"]
            if int(date_ids.min()) != int(date_ids.max()):  # 按当前模式、标识或状态选择处理分支。
                raise AssertionError("日截面训练 batch 混入多个交易日")  # 关键数据契约被破坏时立即终止。
        optimizer.zero_grad(set_to_none=True)  # 清空上一批梯度。
        with autocast_context():  # 在自动混合精度上下文中执行前向计算。
            predictions = model(features)  # 执行当前 batch 前向传播。
        loss = compute_training_loss(predictions, targets)  # 按实验口径计算可微损失。
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
        if TRAINING_CONFIG.loss_name in RANK_LOSS_NAMES:  # 检查当前值是否属于允许集合。
            soft_rank_ic_sum += -loss.detach().double()
            soft_rank_batch_count += 1
        sample_count += current_size
        batch_count += 1
        min_batch_size = (
            current_size if min_batch_size is None
            else min(min_batch_size, current_size)
        )
        max_batch_size = max(max_batch_size, current_size)
        if max_batches is not None and batch_number >= max_batches:  # 检查当前数值是否越过业务或数值阈值。
            break  # 满足终止条件后退出当前循环。
    if not batch_count:  # 检查必需数据、配置或校验结果是否缺失。
        raise ValueError("训练 loader 没有可用 batch")  # 输入数据不满足业务或数值约束时立即终止。
    if DEVICE.type == "cuda":  # 按当前模式、标识或状态选择处理分支。
        torch.cuda.synchronize()  # 同步 GPU 后计时。
    batch_mean_pearson = float((batch_pearson_sum / batch_count).item())  # 计算批次均值Pearson。
    sample_weighted_pearson = float((weighted_pearson / sample_count).item())
    selected_train_pearson = (  # 构造筛选结果训练Pearson，限定后续参与计算的样本。
        batch_mean_pearson if date_batch_mode else sample_weighted_pearson
    )
    soft_rank_ic = (  # 计算soft排名ic。
        float((soft_rank_ic_sum / soft_rank_batch_count).item())
        if soft_rank_batch_count else float("nan")  # 根据当前数据状态选择处理分支。
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
  # 兼容旧 history 字段；含义仍是 Pearson IC。
        "train_batch_ic_mean": batch_mean_pearson,  # 记录训练批次ic均值。
        "train_sample_weighted_ic": sample_weighted_pearson,  # 记录训练样本weightedic。
        "train_samples": sample_count,  # 记录训练样本。
        "train_batches": batch_count,  # 记录训练batches。
        "train_batch_size_min": int(min_batch_size),  # 记录训练批次大小min。
        "train_batch_size_max": int(max_batch_size),  # 记录训练批次大小max。
        "train_seconds": time.perf_counter() - started,  # 记录训练seconds。
    }


def _train_temporal_turnover_epoch(  # 按自然日期累计相邻截面梯度，并在整个 epoch 结束后更新一次。
    model: Any, loader: Any, optimizer: Any, max_batches: int | None = None,  # 声明可选输入；未提供时沿用默认处理流程。
) -> Any:
    """按自然日期累计相邻截面梯度，并在整个 epoch 结束后更新一次。"""

    sampler = loader.batch_sampler  # 创建sampler。
    if (  # 开始组合多项有效性或配置条件。
        not isinstance(sampler, TradeDateBatchSampler)
        or sampler.temporal_pairs is None
        or sampler.shuffle
    ):
        raise ValueError("时序换手损失要求按日期升序且关闭 reshuffle")  # 输入数据不满足业务或数值约束时立即终止。
    available_pairs = len(sampler.temporal_pairs)
    pair_limit = (
        available_pairs if max_batches is None
        else min(int(max_batches), available_pairs)
    )
    if pair_limit < 1:  # 检查当前数值是否越过业务或数值阈值。
        raise ValueError("时序换手训练至少需要一个相邻日期对")  # 输入数据不满足业务或数值约束时立即终止。

    model.train()  # 启用训练模式。
    optimizer.zero_grad(set_to_none=True)  # 清空上一批梯度。
    loss_sum = torch.zeros((), dtype=torch.float64, device=DEVICE)  # 初始化损失sum。
    pearson_sum = torch.zeros((), dtype=torch.float64, device=DEVICE)  # 初始化Pearsonsum。
    weighted_pearson = torch.zeros((), dtype=torch.float64, device=DEVICE)  # 初始化weightedPearson。
    prediction_rank_sum = torch.zeros((), dtype=torch.float64, device=DEVICE)  # 初始化预测排名sum。
    temporal_rank_sum = torch.zeros((), dtype=torch.float64, device=DEVICE)  # 初始化时序排名sum。
    temporal_penalty_sum = torch.zeros((), dtype=torch.float64, device=DEVICE)  # 初始化时序penaltysum。
    sample_count = 0  # 计算样本数量。
    pair_count = 0  # 计算pair数量。
    min_batch_size, max_batch_size = None, 0
    previous_predictions = None  # 先把上一期预测初始化为空，后续按条件填充。
    previous_date_id = None  # 先把上一期交易日id初始化为空，后续按条件填充。
    started = time.perf_counter()

    for batch, features, targets in CudaBatchPrefetcher(loader):  # 遍历cudabatchprefetcher数据加载器，逐项处理批次特征标签。
        date_ids = batch["date_ids"]
        if int(date_ids.min()) != int(date_ids.max()):  # 按当前模式、标识或状态选择处理分支。
            raise AssertionError("日截面训练 batch 混入多个交易日")  # 关键数据契约被破坏时立即终止。
        current_date_id = int(date_ids[0])
        with autocast_context():  # 在自动混合精度上下文中执行前向计算。
            current_predictions = model(features)  # 执行当前 batch 前向传播。
        if previous_predictions is None:  # 检查所需对象是否尚未初始化。
  # 首日只建立固定锚点，不计算标签项，也不保留计算图。
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
            current_predictions,
            targets,
            previous_predictions,
            pair["previous_positions"],
            pair["current_positions"],
            TRAINING_CONFIG.rank_temperature,
            TRAINING_CONFIG.turnover_lambda,
        )
        loss = components["loss"]  # 计算损失。
        values = torch.stack(  # 对齐并合并值。
            [
                loss,
                components["prediction_soft_rank_ic"],
                components["temporal_soft_rank_ic"],
                components["temporal_penalty"],
            ]
        )
        if not torch.isfinite(values).all():  # 检查参与计算的数值是否全部有限。
            raise FloatingPointError("时序换手训练损失或审计指标出现 NaN/Inf")  # 检测到 NaN、Inf 或退化数值时立即终止。
        (loss / pair_limit).backward()  # 反向传播当前损失。

        with torch.no_grad():  # 关闭梯度记录以降低验证或推理开销。
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
        del current_predictions, components, loss, values  # 释放不再使用的中间引用。
        if pair_count >= pair_limit:  # 检查当前数值是否越过业务或数值阈值。
            break  # 满足终止条件后退出当前循环。

    if pair_count != pair_limit:  # 按当前模式、标识或状态选择处理分支。
        raise AssertionError(  # 关键数据契约被破坏时立即终止。
            f"时序日期对覆盖不完整：{pair_count}/{pair_limit}"
        )
    optimizer.step()  # 更新参数或学习率状态。
    if DEVICE.type == "cuda":  # 按当前模式、标识或状态选择处理分支。
        torch.cuda.synchronize()  # 同步 GPU 后计时。
    mean_pearson = float((pearson_sum / pair_count).item())  # 计算均值Pearson。
    weighted_mean_pearson = float((weighted_pearson / sample_count).item())  # 计算weighted均值Pearson。
    prediction_soft_rank = float((prediction_rank_sum / pair_count).item())  # 计算预测soft排名。
    temporal_soft_rank = float((temporal_rank_sum / pair_count).item())  # 计算时序soft排名。
    temporal_penalty = float((temporal_penalty_sum / pair_count).item())
    return {  # 返回按当前规则组织后的结果。
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
  # 兼容旧 history 字段；含义仍是 Pearson IC。
        "train_batch_ic_mean": mean_pearson,  # 记录训练批次ic均值。
        "train_sample_weighted_ic": weighted_mean_pearson,  # 记录训练样本weightedic。
        "train_samples": sample_count,  # 记录训练样本。
        "train_batches": pair_count,  # 记录训练batches。
        "train_batch_size_min": int(min_batch_size),  # 记录训练批次大小min。
        "train_batch_size_max": int(max_batch_size),  # 记录训练批次大小max。
        "train_seconds": time.perf_counter() - started,  # 记录训练seconds。
    }


def train_one_epoch(model: Any, loader: Any, optimizer: Any, max_batches: int | None = None) -> Any:  # 训练一个完整 epoch，并返回损失与批次诊断。
    """训练一个完整 epoch，并返回损失与批次诊断。"""
    if TRAINING_CONFIG.loss_name == TEMPORAL_TURNOVER_LOSS:  # 按当前模式、标识或状态选择处理分支。
        return _train_temporal_turnover_epoch(  # 返回训练一个完整 epoch，并返回损失与批次诊断。
            model, loader, optimizer, max_batches=max_batches  # 创建模型。
        )
    return _train_standard_epoch(  # 返回训练一个完整 epoch，并返回损失与批次诊断。
        model, loader, optimizer, max_batches=max_batches  # 创建模型。
    )


def average_descending_ranks(values: np.ndarray) -> np.ndarray:  # 精确降序平均秩。
    """精确降序平均秩；相同值共享其占用名次的均值。"""

    array = np.asarray(values, dtype=np.float64).reshape(-1)  # [N,...] -> [N]。
    if len(array) < 2 or not np.isfinite(array).all():  # 检查参与计算的数值是否全部有限。
        raise ValueError("排名输入必须至少有两个有限值")  # 输入数据不满足业务或数值约束时立即终止。
    order = np.argsort(-array, kind="stable")  # 计算排序索引以固定样本顺序。
    ordered = array[order]
    starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
    stops = np.r_[starts[1:], len(array)]
    group_ranks = (starts + 1 + stops) / 2  # 计算group排名。
    ranks = np.empty(len(array), dtype=np.float64)  # 初始化排名。
    ranks[order] = np.repeat(group_ranks, stops - starts)  # 将日内秩广播回样本。
    return ranks  # 返回排名。


def numpy_pearson(x: np.ndarray, y: np.ndarray) -> float:  # 用 NumPy 计算两个有效向量的 Pearson 相关系数。
    """用 NumPy 计算两个有效向量的 Pearson 相关系数。"""
    left = np.asarray(x, dtype=np.float64).reshape(-1)  # [N,...] -> [N]。
    right = np.asarray(y, dtype=np.float64).reshape(-1)  # [N,...] -> [N]。
    centered_left = left - left.mean()  # 计算中心化left。
    centered_right = right - right.mean()  # 计算中心化right。
    denominator = np.sqrt(
        np.square(centered_left).sum() * np.square(centered_right).sum()
    )
    if not np.isfinite(denominator) or denominator <= 1e-12:  # 检查参与计算的数值是否全部有限。
        raise FloatingPointError("相关系数分母为零或非有限数")  # 检测到 NaN、Inf 或退化数值时立即终止。
    return float(np.dot(centered_left, centered_right) / denominator)  # 返回完成上述回归或变换后的结果。


def summarize_validation_arrays(  # 先恢复完整日期截面，再计算硬 RankIC、日度 Pearson 和 pooled IC。
    predictions: np.ndarray, targets: np.ndarray, date_ids: np.ndarray,
    validation_metric: str, instrument_ids: np.ndarray | None = None,  # 声明可选输入；未提供时沿用默认处理流程。
) -> dict[str, float | int]:
    """先恢复完整日期截面，再计算硬 RankIC、日度 Pearson 和 pooled IC。"""

    x = np.asarray(predictions, dtype=np.float64).reshape(-1)  # [N,...] -> [N]。
    y = np.asarray(targets, dtype=np.float64).reshape(-1)  # [N,...] -> [N]。
    dates = np.asarray(date_ids, dtype=np.int64).reshape(-1)  # [N,...] -> [N]。
    instruments = (
        None if instrument_ids is None
        else np.asarray(instrument_ids).reshape(-1)  # [N,...] -> [N]。
    )
    if not (len(x) == len(y) == len(dates)) or len(x) < 2:  # 检查当前样本数量是否满足计算要求。
        raise ValueError("验证预测、标签和日期必须等长且至少包含两个样本")  # 输入数据不满足业务或数值约束时立即终止。
    if instruments is not None and len(instruments) != len(x):  # 检查当前样本数量是否满足计算要求。
        raise ValueError("验证股票键与预测长度不一致")  # 输入数据不满足业务或数值约束时立即终止。
    if not np.isfinite(x).all() or not np.isfinite(y).all():  # 检查参与计算的数值是否全部有限。
        raise FloatingPointError("验证预测或标签包含 NaN/Inf")  # 检测到 NaN、Inf 或退化数值时立即终止。
    if validation_metric not in VALIDATION_METRICS:  # 检查当前值是否属于允许集合。
        raise ValueError(f"未知验证指标：{validation_metric}")  # 输入数据不满足业务或数值约束时立即终止。

    order = np.argsort(dates, kind="stable")  # 计算排序索引以固定样本顺序。
    x, y, dates = x[order], y[order], dates[order]
    if instruments is not None:  # 检查必需数据、配置或校验结果是否缺失。
        instruments = instruments[order]
    boundaries = np.flatnonzero(dates[1:] != dates[:-1]) + 1
    starts = np.r_[0, boundaries]
    stops = np.r_[boundaries, len(dates)]
    daily_rank_ic, daily_pearson_ic = [], []  # 计算daily排名ic。
    for start, stop in zip(starts, stops):  # 遍历zipstartsstops，逐项处理起点终点。
        if stop - start < 2:  # 检查当前数值是否越过业务或数值阈值。
            raise FloatingPointError("验证集存在不足两个样本的交易日")  # 检测到 NaN、Inf 或退化数值时立即终止。
        daily_x, daily_y = x[start:stop], y[start:stop]
        daily_pearson_ic.append(numpy_pearson(daily_x, daily_y))  # 把当前结果追加到汇总列表。
        daily_rank_ic.append(  # 把当前结果追加到汇总列表。
            numpy_pearson(
                average_descending_ranks(daily_x),
                average_descending_ranks(daily_y),
            )
        )
    temporal_rank_ic: list[float] = []  # 接收时序排名ic。
    if instruments is not None:  # 检查必需数据、配置或校验结果是否缺失。
        unique_dates = dates[starts]
        if len(unique_dates) > 1 and np.any(np.diff(unique_dates) != 1):  # 检查当前样本数量是否满足计算要求。
            raise ValueError("验证集日期不连续，无法计算相邻预测 RankIC")  # 输入数据不满足业务或数值约束时立即终止。
        for pair_number in range(1, len(starts)):  # 按给定步长遍历样本区间，更新pairnumber。
            previous_slice = slice(starts[pair_number - 1], stops[pair_number - 1])
            current_slice = slice(starts[pair_number], stops[pair_number])
            previous_ids = instruments[previous_slice]
            current_ids = instruments[current_slice]
            if (  # 开始组合多项有效性或配置条件。
                len(np.unique(previous_ids)) != len(previous_ids)
                or len(np.unique(current_ids)) != len(current_ids)
            ):
                raise ValueError("验证集同一日期股票键重复")  # 输入数据不满足业务或数值约束时立即终止。
            common, previous_positions, current_positions = np.intersect1d(
                previous_ids, current_ids, assume_unique=True,
                return_indices=True,  # 返回先恢复完整日期截面，再计算硬 RankIC、日度 Pearson 和 pooled IC。
            )
            if len(common) < 2:  # 检查当前样本数量是否满足计算要求。
                raise ValueError("验证集相邻日期共同股票少于2只")  # 输入数据不满足业务或数值约束时立即终止。
            temporal_rank_ic.append(  # 把当前结果追加到汇总列表。
                numpy_pearson(
                    average_descending_ranks(
                        x[current_slice][current_positions]
                    ),
                    average_descending_ranks(
                        x[previous_slice][previous_positions]
                    ),
                )
            )
    mean_rank_ic = float(np.mean(daily_rank_ic))  # 计算均值排名ic。
    mean_pearson_ic = float(np.mean(daily_pearson_ic))  # 计算均值Pearsonic。
    mean_temporal_rank_ic = (  # 计算均值时序排名ic。
        float(np.mean(temporal_rank_ic)) if temporal_rank_ic else float("nan")
    )
    pooled_ic = numpy_pearson(x, y)  # 计算当前截面相关性。
    selected_ic = (  # 构造筛选结果ic，限定后续参与计算的样本。
        mean_rank_ic
        if validation_metric == "daily_mean_spearman_ic"  # 按当前模式、标识或状态选择处理分支。
        else mean_pearson_ic
    )
    values = np.array(
        [selected_ic, mean_rank_ic, mean_pearson_ic, pooled_ic], dtype=np.float64
    )
    if not np.isfinite(values).all():  # 检查参与计算的数值是否全部有限。
        raise FloatingPointError("验证 IC 不是有限数")  # 检测到 NaN、Inf 或退化数值时立即终止。
    return {  # 返回先恢复完整日期截面，再计算硬 RankIC、日度 Pearson 和 pooled IC。
        "validation_ic": selected_ic,  # 记录验证ic。
        "validation_loss": -selected_ic,  # 记录验证损失。
        "validation_daily_rank_ic": mean_rank_ic,  # 记录验证日度秩ic。
        "validation_daily_pearson_ic": mean_pearson_ic,  # 记录验证日度Pearsonic。
  # 兼容旧字段，其含义保持为日均 Pearson IC。
        "validation_daily_ic": mean_pearson_ic,  # 记录验证日度ic。
        "validation_pooled_ic": pooled_ic,  # 记录验证pooledic。
        "validation_temporal_rank_ic": mean_temporal_rank_ic,  # 记录验证时序秩ic。
        "validation_temporal_penalty": 1 - mean_temporal_rank_ic,  # 记录验证时序penalty。
        "validation_temporal_pairs": len(temporal_rank_ic),  # 记录验证时序pairs。
        "validation_dates": len(starts),  # 记录验证交易日。
        "validation_samples": len(x),  # 记录验证样本。
    }


# 使用 `torch.no_grad()` 包装 `evaluate_model`，并定义其可复用的处理逻辑。
@torch.no_grad()  # 验证或推理阶段关闭梯度记录。
def evaluate_model(model: Any, loader: Any, max_batches: int | None = None) -> Any:  # 在验证集上评估模型，并汇总逐日 IC 与损失诊断。
    """在验证集上评估模型，并汇总逐日 IC 与损失诊断。"""
    model.eval()  # 关闭 dropout 等训练行为。
    prediction_parts, target_parts, date_parts, sample_index_parts = [], [], [], []  # 初始化预测parts容器。
    started = time.perf_counter()
    for batch_number, (batch, features, targets) in enumerate(  # 遍历enumerate，逐项处理批次number批次特征标签。
        CudaBatchPrefetcher(loader), start=1
    ):
        with autocast_context():  # 在自动混合精度上下文中执行前向计算。
            predictions = model(features)  # 执行当前 batch 前向传播。
        prediction_parts.append(  # 把当前结果追加到汇总列表。
            predictions.float().cpu().numpy().astype(np.float64, copy=False)
        )
        target_parts.append(  # 把当前结果追加到汇总列表。
            targets.float().cpu().numpy().astype(np.float64, copy=False)
        )
        date_parts.append(batch["date_ids"].numpy().astype(np.int64, copy=False))  # 把当前结果追加到汇总列表。
        sample_index_parts.append(  # 把当前结果追加到汇总列表。
            batch["sample_indices"].numpy().astype(np.int64, copy=False)
        )
        if max_batches is not None and batch_number >= max_batches:  # 检查当前数值是否越过业务或数值阈值。
            break  # 满足终止条件后退出当前循环。
    if not prediction_parts:  # 检查必需数据、配置或校验结果是否缺失。
        raise ValueError("验证 loader 没有可用 batch")  # 输入数据不满足业务或数值约束时立即终止。
    if DEVICE.type == "cuda":  # 按当前模式、标识或状态选择处理分支。
        torch.cuda.synchronize()  # 同步 GPU 后计时。
    result = summarize_validation_arrays(  # 按约定截面口径汇总。
        np.concatenate(prediction_parts),
        np.concatenate(target_parts),
        np.concatenate(date_parts),
        TRAINING_CONFIG.validation_metric,
        instrument_ids=sample_index["instrument_id"].to_numpy(copy=False)[  # 转换并统一股票ids的数据表示。
            np.concatenate(sample_index_parts)
        ],
    )
    result["validation_seconds"] = time.perf_counter() - started
    return result  # 返回结果。


# 使用 `torch.no_grad()` 包装 `predict_model`，并定义其可复用的处理逻辑。
@torch.no_grad()  # 验证或推理阶段关闭梯度记录。
def predict_model(model: Any, loader: Any) -> pd.DataFrame:  # 在预测集上生成与样本行号严格对齐的模型分数。
    """在预测集上生成与样本行号严格对齐的模型分数。"""
    model.eval()  # 关闭 dropout 等训练行为。
    indices, predictions = [], []  # 初始化索引容器。
    for batch, features, _ in CudaBatchPrefetcher(loader):  # 遍历cudabatchprefetcher数据加载器，逐项处理批次特征。
        with autocast_context():  # 在自动混合精度上下文中执行前向计算。
            output = model(features)  # 执行当前 batch 前向传播。
        indices.append(batch["sample_indices"].numpy())  # 把当前结果追加到汇总列表。
        predictions.append(output.float().cpu().numpy().astype(np.float32, copy=False))  # 把当前结果追加到汇总列表。
    return pd.DataFrame(  # 返回在预测集上生成与样本行号严格对齐的模型分数。
        {"sample_index": np.concatenate(indices), "prediction": np.concatenate(predictions)}
    )


def execution_payload() -> dict[str, object]:  # 整理影响单次训练执行方式的配置载荷。
    """整理影响单次训练执行方式的配置载荷。"""
    date_batches = TRAINING_CONFIG.train_batch_mode == "trade_date"
    temporal_mode = TRAINING_CONFIG.loss_name == TEMPORAL_TURNOVER_LOSS
    return {  # 返回整理后的结果。
        "parallel_backend": PARALLEL_BACKEND,  # 记录parallelbackend。
        "gpu_count_per_task": 1,  # 记录gpu数量pertask。
        "train_batch_mode": TRAINING_CONFIG.train_batch_mode,  # 记录训练批次模式。
        "batch_size_per_task": (  # 记录批次大小pertask。
            "full_trade_date" if date_batches else TRAINING_CONFIG.batch_size
        ),
        "fixed_batch_size": TRAINING_CONFIG.batch_size,  # 记录fixed批次大小。
        "inference_batch_size": TRAINING_CONFIG.inference_batch_size,  # 记录inference批次大小。
        "reshuffle_each_epoch": not temporal_mode,  # 记录reshuffleeach轮次。
        "gradient_update_mode": (  # 记录梯度update模式。
            "once_per_epoch" if temporal_mode else "per_batch"
        ),
        "optimizer_steps_per_epoch": 1 if temporal_mode else "per_batch",  # 记录优化器stepsper轮次。
        "temporal_lag_trade_days": 1 if temporal_mode else None,  # 记录时序lagtradedays。
        "temporal_anchor_mode": (  # 记录时序anchor模式。
            "previous_prediction_stop_gradient_cached"
            if temporal_mode else None  # 根据当前数据状态选择处理分支。
        ),
        "model_forward_passes_per_train_date": 1 if temporal_mode else None,  # 记录模型forwardpassesper训练交易日。
        "loss_name": TRAINING_CONFIG.loss_name,  # 记录损失名称。
        "loss_parameters": {  # 记录损失parameters。
            "rank_temperature": TRAINING_CONFIG.rank_temperature,  # 记录秩temperature。
            "turnover_lambda": TRAINING_CONFIG.turnover_lambda,  # 记录turnoverlambda。
            "huber_delta": TRAINING_CONFIG.huber_delta,  # 记录huberdelta。
            "smooth_l1_beta": TRAINING_CONFIG.smooth_l1_beta,  # 记录smoothl1beta。
            "hybrid_weight": TRAINING_CONFIG.hybrid_weight,  # 记录hybrid权重。
        },
        "validation_metric": TRAINING_CONFIG.validation_metric,  # 记录验证指标。
        "data_loader_workers": NUM_WORKERS,  # 记录数据数据加载器workers。
        "torch_compile": USE_COMPILE,  # 记录torchcompile。
        "compile_mode": COMPILE_MODE,  # 记录compile模式。
        "compile_dynamic_batch": bool(USE_COMPILE and date_batches),  # 记录compiledynamic批次。
        "fused_adam": USE_FUSED_ADAM,  # 记录fusedadam。
        "cuda_prefetch": USE_CUDA_PREFETCH,  # 记录cudaprefetch。
    }


def run_payload(  # 整理影响整个 run 身份和可复用性的签名载荷。
    year: int, seed: int, batch_summary: dict[str, object]
) -> dict[str, object]:
    """整理影响整个 run 身份和可复用性的签名载荷。"""
    return {  # 返回整理后的结果。
        "schema_version": 6,  # 记录schema版本。
        "forecast_year": year,  # 记录forecast年份。
        "seed": seed,  # 记录随机种子。
        "dataset_signature": dataset_manifest["signature"],  # 记录dataset签名，用于来源一致性校验。
        "rolling_signature": rolling_manifest["signature"],  # 记录rolling签名，用于来源一致性校验。
        "model_config": asdict(MODEL_CONFIG),  # 记录模型配置。
        "training_config": asdict(TRAINING_CONFIG),  # 记录训练配置。
        "execution_config": execution_payload(),  # 记录执行配置。
        "training_batch_summary": batch_summary,  # 记录训练批次汇总。
        "loss": TRAINING_CONFIG.loss_name,  # 记录损失。
        "loss_parameters": {  # 记录损失parameters。
            "rank_temperature": TRAINING_CONFIG.rank_temperature,  # 记录秩temperature。
            "turnover_lambda": TRAINING_CONFIG.turnover_lambda,  # 记录turnoverlambda。
            "huber_delta": TRAINING_CONFIG.huber_delta,  # 记录huberdelta。
            "smooth_l1_beta": TRAINING_CONFIG.smooth_l1_beta,  # 记录smoothl1beta。
            "hybrid_weight": TRAINING_CONFIG.hybrid_weight,  # 记录hybrid权重。
            "prediction_input": "raw_model_output",  # 把 prediction_input 映射为"raw_model_output"。
            "target_input": "daily_winsorized_1pct_zscore",  # 把 target_input 映射为"daily_winsorized_1pct_zscore"。
            "prediction_preprocessing_inside_loss": "none",  # 把 prediction_preprocessing_inside_loss 映射为"none"。
            "target_preprocessing_inside_loss": "none",  # 把 target_preprocessing_inside_loss 映射为"none"。
            "rank_target_gradient": False,  # 记录秩标签梯度。
            "temporal_prediction_gradients": "current_only" if (  # 记录时序预测gradients。
                TRAINING_CONFIG.loss_name == TEMPORAL_TURNOVER_LOSS
            ) else None,
            "previous_prediction_gradient": "stop_gradient" if (  # 记录上一期预测梯度。
                TRAINING_CONFIG.loss_name == TEMPORAL_TURNOVER_LOSS
            ) else None,
            "previous_prediction_source": "cached_previous_date_output" if (  # 记录上一期预测来源。
                TRAINING_CONFIG.loss_name == TEMPORAL_TURNOVER_LOSS
            ) else None,
            "turnover_penalty_direction": "1_minus_temporal_rank_ic" if (  # 记录turnoverpenaltydirection。
                TRAINING_CONFIG.loss_name == TEMPORAL_TURNOVER_LOSS
            ) else None,
        },
        "validation_metric": TRAINING_CONFIG.validation_metric,  # 记录验证指标。
        "validation_audit_metrics": [  # 记录验证audit指标。
            "daily_mean_spearman_ic",
            "daily_mean_pearson_ic",
            "pooled_pearson_ic",
            "temporal_spearman_rank_ic",
            "temporal_rank_penalty",
        ],
        "precision": "bf16_autocast_with_fp32_loss" if USE_BF16 else "float32",  # 把 precision 映射为"bf16_autocast_with_fp32_loss" if USE_BF16 else "float32"。
    }


def atomic_checkpoint(payload: dict[str, object], path: Path) -> None:  # 先写临时 checkpoint，再原子替换正式模型文件。
    """先写临时 checkpoint，再原子替换正式模型文件。"""
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")  # 构造临时文件。
    torch.save(payload, temporary)  # 保存模型或训练状态。
    temporary.replace(path)  # 用临时文件原子替换正式结果。


def checkpoint_payload(  # 构建 checkpoint 中的模型、优化器、进度和签名信息。
    model: Any, optimizer: Any, generator_state: Any, epoch: Any, best_epoch: Any, best_ic: Any,
    best_validation_metrics: Any, patience_count: Any, signature: Any, epoch_metrics: Any,
) -> dict[str, object]:
    """构建 checkpoint 中的模型、优化器、进度和签名信息。"""
    return {  # 返回构建 checkpoint 中的模型、优化器、进度和签名信息。
        "model_state": canonical_state(model),  # 记录模型状态。
        "optimizer_state": optimizer.state_dict(),  # 记录优化器状态。
        "python_random_state": random.getstate(),  # 记录pythonrandom状态。
        "numpy_random_state": np.random.get_state(),  # 记录numpyrandom状态。
        "torch_rng_state": torch.get_rng_state(),  # 记录torchrng状态。
        "cuda_rng_state_all": (  # 记录cudarng状态all。
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "train_generator_state": generator_state,  # 记录训练generator状态。
        "epoch": epoch,  # 记录轮次。
        "next_trade_date_sampler_epoch": epoch + 1,  # 记录nexttrade交易日sampler轮次。
        "best_epoch": best_epoch,  # 记录最优轮次。
        "best_validation_ic": best_ic,  # 记录最优验证ic。
        "best_validation_metrics": best_validation_metrics,  # 记录最优验证指标。
        "epoch_metrics": epoch_metrics,  # 记录轮次指标。
        "epochs_without_improvement": patience_count,  # 记录epochswithoutimprovement。
        "run_signature": signature,  # 记录运行签名，用于来源一致性校验。
        "execution_config": execution_payload(),  # 记录执行配置。
        "model_config": asdict(MODEL_CONFIG),  # 记录模型配置。
        "training_config": asdict(TRAINING_CONFIG),  # 记录训练配置。
    }


def load_best_model(path: Path) -> Any:  # 加载最佳验证 checkpoint，并恢复可用于预测的模型。
    """加载最佳验证 checkpoint，并恢复可用于预测的模型。"""
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    model = build_model()  # 创建模型并部署到设备。
    load_canonical_state(model, checkpoint["model_state"])  # 恢复模型权重。
    model.eval()  # 关闭 dropout 等训练行为。
    return model  # 返回模型。


def train_single_run(year: int, seed: int) -> Any:  # 完成一个预测年度和一个随机种子的训练、早停与预测。
    """完成一个预测年度和一个随机种子的训练、早停与预测。"""
    set_random_seed(seed)  # 固定各随机源。
    train_loader, validation_loader, _ = make_loaders(year, seed)  # 创建训练数据加载器。
    batch_summary = loader_batch_summary(train_loader)
    payload = run_payload(year, seed, batch_summary)
    signature = stable_hash(payload)  # 生成数据复用签名。

    run_dir = MODEL_ROOT / f"forecast_{year}" / f"seed_{seed}"  # 构造执行目录。
    run_dir.mkdir(parents=True, exist_ok=True)  # 确保目标目录存在。
    best_path, last_path = run_dir / "best.pt", run_dir / "last.pt"  # 构造最优路径。
    history_path, completed_path = run_dir / "history.parquet", run_dir / "completed.json"  # 构造历史记录路径。
    atomic_json(  # 原子落盘，避免半成品。
        {"year_signature": stable_hash({k: v for k, v in payload.items() if k != "seed"}),
         "payload": {k: v for k, v in payload.items() if k != "seed"}},  # 记录载荷。
        MODEL_ROOT / f"forecast_{year}" / "model_config.json",
    )
    if completed_path.exists() and best_path.exists() and not OVERWRITE_TRAINING:  # 检查目标文件或目录是否满足存在性要求。
        completed = read_json(completed_path)  # 读取completed。
        if completed.get("run_signature") == signature:  # 按当前模式、标识或状态选择处理分支。
            print(f"[{year}/seed={seed}] 已完成，直接复用。")  # 输出当前任务的进度或诊断信息。
            return load_best_model(best_path), completed["summary"]  # 返回完成一个预测年度和一个随机种子的训练、早停与预测。

    model = build_model()  # 创建模型并部署到设备。
    optimizer = build_optimizer(model)  # 按训练配置创建优化器。
    start_epoch, best_epoch = 1, 0
    best_ic, patience_count = -np.inf, 0
    best_validation_metrics: dict[str, float] = {}  # 计算最优验证指标。
    history: list[dict[str, object]] = []  # 初始化历史记录容器。
    if last_path.exists() and not OVERWRITE_TRAINING:  # 检查目标文件或目录是否满足存在性要求。
        checkpoint = torch.load(last_path, map_location=DEVICE, weights_only=False)
        if checkpoint.get("run_signature") != signature:  # 按当前模式、标识或状态选择处理分支。
            raise ValueError(f"{last_path} 配置签名不一致，请显式覆盖重训")  # 输入数据不满足业务或数值约束时立即终止。
        load_canonical_state(model, checkpoint["model_state"])  # 恢复模型权重。
        optimizer.load_state_dict(checkpoint["optimizer_state"])  # 恢复优化器续训状态。
        random.setstate(checkpoint["python_random_state"])  # 恢复 Python 随机状态。
        np.random.set_state(checkpoint["numpy_random_state"])  # 恢复 NumPy 随机状态。
        torch.set_rng_state(checkpoint["torch_rng_state"])  # 恢复 PyTorch CPU 随机状态。
        if torch.cuda.is_available() and checkpoint["cuda_rng_state_all"] is not None:  # 检查必需数据、配置或校验结果是否缺失。
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])  # 恢复 CUDA 随机状态。
        train_loader.generator.set_state(checkpoint["train_generator_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        expected_sampler_epoch = checkpoint.get("next_trade_date_sampler_epoch")  # 创建预期sampler轮次。
        if (  # 开始组合多项有效性或配置条件。
            TRAINING_CONFIG.train_batch_mode == "trade_date"
            and expected_sampler_epoch is not None
            and int(expected_sampler_epoch) != start_epoch
        ):
            raise AssertionError("checkpoint 中的下一日截面 sampler epoch 不一致")  # 关键数据契约被破坏时立即终止。
        best_epoch = int(checkpoint["best_epoch"])
        best_ic = float(checkpoint["best_validation_ic"])
        best_validation_metrics = {  # 计算最优验证指标。
            key: float(value)  # 接收键。
            for key, value in checkpoint.get(  # 遍历检查点get，逐项处理键值。
                "best_validation_metrics", {}
            ).items()
        }
        patience_count = int(checkpoint["epochs_without_improvement"])  # 计算patience数量。
        if history_path.exists():  # 检查目标文件或目录是否满足存在性要求。
            history = pd.read_parquet(history_path).to_dict("records")  # 读取历史记录。
        print(f"[{year}/seed={seed}] 从 epoch {start_epoch} 续训。")  # 输出当前任务的进度或诊断信息。

    started = time.perf_counter()
    stopped_early = False
    for epoch in range(start_epoch, TRAINING_CONFIG.max_epochs + 1):  # 按给定步长遍历样本区间，更新轮次。
        epoch_started = time.perf_counter()
        set_train_epoch(train_loader, epoch)  # 同步 epoch，保持采样可复现。
        train_metrics = train_one_epoch(model, train_loader, optimizer)  # 完成一个训练 epoch。
        validation_metrics = evaluate_model(model, validation_loader)  # 无梯度计算验证指标。
        validation_ic = validation_metrics["validation_ic"]
        improved = validation_ic > best_ic + TRAINING_CONFIG.improvement_tolerance  # 计算improved判定条件。
        if improved:  # 根据当前数据状态选择处理分支。
            best_ic, best_epoch, patience_count = validation_ic, epoch, 0
            best_validation_metrics = {  # 计算最优验证指标。
                "validation_daily_rank_ic": validation_metrics[  # 记录验证日度秩ic。
                    "validation_daily_rank_ic"
                ],
                "validation_daily_pearson_ic": validation_metrics[  # 记录验证日度Pearsonic。
                    "validation_daily_pearson_ic"
                ],
                "validation_pooled_ic": validation_metrics["validation_pooled_ic"],  # 记录验证pooledic。
                "validation_temporal_rank_ic": validation_metrics[  # 记录验证时序秩ic。
                    "validation_temporal_rank_ic"
                ],
                "validation_temporal_penalty": validation_metrics[  # 记录验证时序penalty。
                    "validation_temporal_penalty"
                ],
            }
        else:  # 处理上述条件均不成立的情况。
            patience_count += 1
        history.append(  # 把当前结果追加到汇总列表。
            {
                "forecast_year": year,  # 记录forecast年份。
                "seed": seed,  # 记录随机种子。
                "epoch": epoch,  # 记录轮次。
                "train_batch_mode": TRAINING_CONFIG.train_batch_mode,  # 记录训练批次模式。
                "loss_name": TRAINING_CONFIG.loss_name,  # 记录损失名称。
                "validation_metric": TRAINING_CONFIG.validation_metric,  # 记录验证指标。
                **train_metrics,
                **validation_metrics,
                "best_validation_ic": best_ic,  # 记录最优验证ic。
                "epochs_without_improvement": patience_count,  # 记录epochswithoutimprovement。
                "epoch_seconds": time.perf_counter() - epoch_started,  # 记录轮次seconds。
            }
        )
        atomic_parquet(pd.DataFrame(history), history_path)  # 原子落盘，避免半成品。
        state = checkpoint_payload(
            model, optimizer, train_loader.generator.get_state(), epoch, best_epoch,
            best_ic, best_validation_metrics, patience_count, signature,
            history[-1],
        )
        atomic_checkpoint(state, last_path)
        if improved:  # 根据当前数据状态选择处理分支。
            atomic_checkpoint(state, best_path)
        print(  # 输出当前任务的进度或诊断信息。
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
            flush=True,  # 配置flush。
        )
        if patience_count >= TRAINING_CONFIG.early_stopping_patience:  # 检查当前数值是否越过业务或数值阈值。
            stopped_early = True
            break  # 满足终止条件后退出当前循环。
    if not best_path.exists():  # 检查目标文件或目录是否满足存在性要求。
        raise FileNotFoundError(f"训练后没有最佳 checkpoint：{best_path}")  # 目标文件不存在时立即终止。
    best_epoch_history = next(
        row for row in history if int(row["epoch"]) == best_epoch
    )
    summary = {  # 初始化汇总容器。
        "forecast_year": year,  # 记录forecast年份。
        "seed": seed,  # 记录随机种子。
        "best_epoch": best_epoch,  # 记录最优轮次。
        "best_validation_ic": best_ic,  # 记录最优验证ic。
        "best_validation_rank_ic": best_validation_metrics.get(  # 记录最优验证秩ic。
            "validation_daily_rank_ic"
        ),
        "best_validation_daily_pearson_ic": best_validation_metrics.get(  # 记录最优验证日度Pearsonic。
            "validation_daily_pearson_ic"
        ),
  # 兼容旧汇总字段，其含义保持为日均 Pearson IC。
        "best_validation_daily_ic": best_validation_metrics.get(  # 记录最优验证日度ic。
            "validation_daily_pearson_ic"
        ),
        "best_validation_pooled_ic": best_validation_metrics.get(  # 记录最优验证pooledic。
            "validation_pooled_ic"
        ),
        "best_validation_temporal_rank_ic": best_validation_metrics.get(  # 记录最优验证时序秩ic。
            "validation_temporal_rank_ic"
        ),
        "best_validation_temporal_penalty": best_validation_metrics.get(  # 记录最优验证时序penalty。
            "validation_temporal_penalty"
        ),
        "best_epoch_train_loss": best_epoch_history["train_loss"],  # 记录最优轮次训练损失。
        "best_epoch_train_prediction_soft_rank_ic": best_epoch_history[  # 记录最优轮次训练预测soft秩ic。
            "train_prediction_soft_rank_ic"
        ],
        "best_epoch_train_temporal_soft_rank_ic": best_epoch_history[  # 记录最优轮次训练时序soft秩ic。
            "train_temporal_soft_rank_ic"
        ],
        "best_epoch_train_temporal_penalty": best_epoch_history[  # 记录最优轮次训练时序penalty。
            "train_temporal_penalty"
        ],
        "best_epoch_train_forward_passes": best_epoch_history.get(  # 记录最优轮次训练forwardpasses。
            "train_forward_passes"
        ),
        "temporal_anchor_gradient": best_epoch_history.get(  # 记录时序anchor梯度。
            "temporal_anchor_gradient"
        ),
        "validation_metric": TRAINING_CONFIG.validation_metric,  # 记录验证指标。
        "loss_name": TRAINING_CONFIG.loss_name,  # 记录损失名称。
        "rank_temperature": TRAINING_CONFIG.rank_temperature,  # 记录秩temperature。
        "turnover_lambda": TRAINING_CONFIG.turnover_lambda,  # 记录turnoverlambda。
        "huber_delta": TRAINING_CONFIG.huber_delta,  # 记录huberdelta。
        "smooth_l1_beta": TRAINING_CONFIG.smooth_l1_beta,  # 记录smoothl1beta。
        "hybrid_weight": TRAINING_CONFIG.hybrid_weight,  # 记录hybrid权重。
        "train_batch_mode": TRAINING_CONFIG.train_batch_mode,  # 记录训练批次模式。
        "train_batch_count": batch_summary["train_batch_count"],  # 记录训练批次数量。
        "train_batch_size_min": batch_summary["train_batch_size_min"],  # 记录训练批次大小min。
        "train_batch_size_median": batch_summary["train_batch_size_median"],  # 记录训练批次大小median。
        "train_batch_size_max": batch_summary["train_batch_size_max"],  # 记录训练批次大小max。
        "reshuffle_each_epoch": batch_summary["reshuffle_each_epoch"],  # 记录reshuffleeach轮次。
        "gradient_update_mode": batch_summary["gradient_update_mode"],  # 记录梯度update模式。
        "optimizer_steps_per_epoch": batch_summary["optimizer_steps_per_epoch"],  # 记录优化器stepsper轮次。
        "temporal_lag_trade_days": batch_summary.get("temporal_lag_trade_days"),  # 记录时序lagtradedays。
        "temporal_anchor_mode": batch_summary.get("temporal_anchor_mode"),  # 记录时序anchor模式。
        "model_forward_passes_per_train_date": batch_summary.get(  # 记录模型forwardpassesper训练交易日。
            "model_forward_passes_per_train_date"
        ),
        "train_temporal_pair_count": batch_summary.get(  # 记录训练时序pair数量。
            "train_temporal_pair_count", 0
        ),
        "temporal_common_count_min": batch_summary.get(  # 记录时序公共数量min。
            "temporal_common_count_min"
        ),
        "temporal_common_count_median": batch_summary.get(  # 记录时序公共数量median。
            "temporal_common_count_median"
        ),
        "temporal_common_count_max": batch_summary.get(  # 记录时序公共数量max。
            "temporal_common_count_max"
        ),
        "temporal_common_coverage_min": batch_summary.get(  # 记录时序公共覆盖率min。
            "temporal_common_coverage_min"
        ),
        "temporal_common_coverage_median": batch_summary.get(  # 记录时序公共覆盖率median。
            "temporal_common_coverage_median"
        ),
        "temporal_common_coverage_max": batch_summary.get(  # 记录时序公共覆盖率max。
            "temporal_common_coverage_max"
        ),
        "last_epoch": int(history[-1]["epoch"]),  # 记录最后一个轮次。
        "stopped_early": stopped_early,  # 记录stoppedearly。
        "elapsed_seconds": time.perf_counter() - started,  # 记录elapsedseconds。
        "epoch_elapsed_seconds": float(sum(x["epoch_seconds"] for x in history)),  # 记录轮次elapsedseconds。
        "parallel_backend": PARALLEL_BACKEND,  # 记录parallelbackend。
        "gpu_count": 1,  # 记录gpu数量。
        "run_name": RUN_NAME,  # 记录运行名称。
        "status": "completed",  # 把 status 映射为"completed"。
    }
    atomic_json({"run_signature": signature, "summary": summary}, completed_path)  # 原子落盘，避免半成品。
    return load_best_model(best_path), summary  # 返回完成一个预测年度和一个随机种子的训练、早停与预测。


def run_smoke_test() -> pd.DataFrame:  # 运行小批量前向、反向和验证冒烟测试，不生成正式产物。
    """运行小批量前向、反向和验证冒烟测试，不生成正式产物。"""
    year, seed = TRAINING_CONFIG.forecast_years[0], TRAINING_CONFIG.seeds[0]
    set_random_seed(seed)  # 固定各随机源。
    smoke_batch_mode = (
        "trade_date"
        if TRAINING_CONFIG.loss_name == TEMPORAL_TURNOVER_LOSS  # 按当前模式、标识或状态选择处理分支。
        else "fixed_samples"
    )
    train_loader, validation_loader, _ = make_loaders(  # 创建训练数据加载器。
        year, seed, batch_size=64, inference_batch_size=64, workers=0,
        train_batch_mode=smoke_batch_mode,  # 配置训练批次模式。
    )
    model = build_model()  # 创建模型并部署到设备。
    optimizer = build_optimizer(model)  # 按训练配置创建优化器。
    training = train_one_epoch(model, train_loader, optimizer, 1)  # 完成一个训练 epoch。
    validation = evaluate_model(model, validation_loader, 1)  # 无梯度计算验证指标。
    state = canonical_state(model)  # 提取标准模型权重。
    with tempfile.TemporaryDirectory() as temporary_directory:  # 进入受控上下文并在退出时自动释放资源。
        path = Path(temporary_directory) / "smoke.pt"  # 构造路径。
        torch.save({"model_state": state}, path)  # 保存模型或训练状态。
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
        raw_model = OrderFlowVisionTransformer(MODEL_CONFIG).to(DEVICE)  # 创建原始模型。
        raw_model.load_state_dict(checkpoint["model_state"])
    del model, raw_model, optimizer  # 释放不再使用的中间引用。
    if DEVICE.type == "cuda":  # 按当前模式、标识或状态选择处理分支。
        torch.cuda.empty_cache()
    return pd.DataFrame(  # 返回运行小批量前向、反向和验证冒烟测试，不生成正式产物。
        [{"forecast_year": year, "train_ic": training["train_ic"],
          "validation_ic": validation["validation_ic"], "canonical_checkpoint": True}]  # 记录验证ic。
    )


def _trade_date_preflight_batches() -> list[tuple[int, str, dict[str, torch.Tensor]]]:  # 逐日抽取预检批次，核对日批次边界、标签有限性和时序对齐。
    """逐日抽取预检批次，核对日批次边界、标签有限性和时序对齐。"""
    date_ids = sample_index["date_id"].to_numpy(np.int32, copy=False)  # 转换并统一交易日ids的数据表示。
    seed = TRAINING_CONFIG.seeds[0]
    max_year, max_size = None, -1
    for year in TRAINING_CONFIG.forecast_years:  # 遍历训练配置forecast年份，逐项处理年份。
        split = load_split(year)
        sampler = TradeDateBatchSampler(  # 按完整交易日截面组 batch。
            split["train_indices"], date_ids, seed=seed, shuffle=True
        )
        current_max = int(sampler.batch_sizes.max())
        if current_max > max_size:  # 检查当前数值是否越过业务或数值阈值。
            max_year, max_size = year, current_max
    if max_year is None:  # 检查所需对象是否尚未初始化。
        raise ValueError("没有可用于日截面显存预检的年度")  # 输入数据不满足业务或数值约束时立即终止。

    split = load_split(max_year)
    sampler = TradeDateBatchSampler(  # 按完整交易日截面组 batch。
        split["train_indices"], date_ids, seed=seed, shuffle=True
    )
    sizes = sampler.batch_sizes
    max_batch = int(np.argmax(sizes))
    median_batch = int(np.argmin(np.abs(sizes - np.median(sizes))))
    selected = [(max_batch, "max"), (median_batch, "median")]  # 构造筛选结果，限定后续参与计算的样本。
    if sizes[max_batch] == sizes[median_batch]:  # 按当前模式、标识或状态选择处理分支。
        min_batch = int(np.argmin(sizes))
        selected[1] = (min_batch, "min")  # 构造筛选结果，限定后续参与计算的样本。
    if selected[0][0] == selected[1][0]:  # 按当前模式、标识或状态选择处理分支。
        selected = selected[:1]  # 构造筛选结果，限定后续参与计算的样本。

    collator = MemmapBatchCollator()  # 整理并搬运 batch。
    result = []  # 初始化结果容器。
    for batch_number, label in selected:  # 遍历筛选结果，逐项处理批次number标签。
        positions = sampler.batch_slice(batch_number)
        global_indices = split["train_indices"][positions]
        result.append(  # 把当前结果追加到汇总列表。
            (max_year, label, collator(global_indices.tolist()))
        )
    return result  # 返回结果。


def _temporal_preflight_pairs() -> list[tuple[  # 选择含最大显存需求在内的十二组不同相邻截面 shape。
    int, str, dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, object]
]]:
    """选择含最大显存需求在内的十二组不同相邻截面 shape。"""

    date_ids = sample_index["date_id"].to_numpy(np.int32, copy=False)  # 转换并统一交易日ids的数据表示。
    instrument_ids = sample_index["instrument_id"].to_numpy(copy=False)  # 转换并统一股票ids的数据表示。
    seed = TRAINING_CONFIG.seeds[0]
    candidates: list[dict[str, object]] = []  # 初始化candidates容器。
    for year in TRAINING_CONFIG.forecast_years:  # 遍历训练配置forecast年份，逐项处理年份。
        split = load_split(year)
        sampler = TradeDateBatchSampler(  # 按完整交易日截面组 batch。
            split["train_indices"], date_ids, seed=seed, shuffle=False,
            instrument_ids=instrument_ids,  # 配置股票ids。
        )
        for pair_number, pair in enumerate(sampler.temporal_pairs or []):  # 遍历enumeratesampler时序pairsor，逐项处理pairnumberpair。
            previous_size = int(sampler.batch_sizes[pair_number])
            current_size = int(sampler.batch_sizes[pair_number + 1])
            common_count = int(pair["common_count"])  # 计算公共数量。
            candidates.append(  # 把当前结果追加到汇总列表。
                {
                    "year": year,  # 记录年份。
                    "split": split,  # 记录切分。
                    "sampler": sampler,  # 记录sampler。
                    "pair_number": pair_number,  # 记录pairnumber。
                    "pair": pair,  # 记录pair。
                    "shape": (previous_size, current_size, common_count),  # 记录形状。
  # 当前 RankIC 两个软秩、时序项两个共同集软秩均为 O(n²)。
                    "score": 2 * current_size**2 + 2 * common_count**2,  # 记录分数。
                }
            )
    if not candidates:  # 检查必需数据、配置或校验结果是否缺失。
        raise ValueError("没有可用于时序换手显存预检的相邻日期对")  # 输入数据不满足业务或数值约束时立即终止。
    unique_by_shape = {item["shape"]: item for item in candidates}  # 初始化uniqueby形状容器。
    ordered = sorted(  # 计算ordered以固定样本顺序。
        unique_by_shape.values(), key=lambda item: int(item["score"])
    )
    test_count = min(12, len(ordered))  # 计算test数量。
    quantile_positions = np.linspace(
        0, len(ordered) - 1, num=test_count, dtype=np.int64
    )
    selected = [ordered[-1]]  # 构造筛选结果，限定后续参与计算的样本。
    selected.extend(  # 把一组结果并入当前列表。
        ordered[int(position)]
        for position in quantile_positions  # 遍历quantile位置，逐项处理位置。
        if ordered[int(position)]["shape"] != selected[0]["shape"]  # 按当前模式、标识或状态选择处理分支。
    )
    selected = selected[:test_count]  # 构造筛选结果，限定后续参与计算的样本。

    collator = MemmapBatchCollator()  # 整理并搬运 batch。
    result = []  # 初始化结果容器。
    for selection_number, item in enumerate(selected):  # 遍历enumerate筛选结果，逐项处理selectionnumberitem。
        sampler = item["sampler"]  # 创建sampler。
        split = item["split"]
        pair_number = int(item["pair_number"])
        previous_positions = sampler.batch_slice(pair_number)
        current_positions = sampler.batch_slice(pair_number + 1)
        previous_indices = split["train_indices"][previous_positions]
        current_indices = split["train_indices"][current_positions]
        result.append(  # 把当前结果追加到汇总列表。
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
    return result  # 返回结果。


def wait_for_preflight_barrier() -> None:  # 让同一 GPU 上的预检 worker 在模型就绪后同时开始。
    """让同一 GPU 上的预检 worker 在模型就绪后同时开始。"""

    barrier_value = os.getenv("VIT_PREFLIGHT_BARRIER_DIR")
    if not barrier_value:  # 检查必需数据、配置或校验结果是否缺失。
        return  # 返回让同一 GPU 上的预检 worker 在模型就绪后同时开始。
    barrier_dir = Path(barrier_value)  # 构造barrier目录。
    rank = int(os.getenv("VIT_PREFLIGHT_RANK", "0"))  # 计算排名。
    world_size = int(os.getenv("VIT_PREFLIGHT_WORLD_SIZE", "1"))
    barrier_dir.mkdir(parents=True, exist_ok=True)  # 确保目标目录存在。
    (barrier_dir / f"ready_{rank}").touch()
    deadline = time.monotonic() + 300
    while len(list(barrier_dir.glob("ready_*"))) < world_size:  # 在终止条件出现前持续迭代。
        if time.monotonic() >= deadline:  # 检查当前数值是否越过业务或数值阈值。
            raise TimeoutError("同卡并发预检 worker 屏障等待超时")  # 当前数据违反接口约定时立即终止。
        time.sleep(0.1)


def run_full_batch_preflight() -> pd.DataFrame | None:  # 在正式训练前检查全量批次形状、日期与显存相关约束。
    """在正式训练前检查全量批次形状、日期与显存相关约束。"""
    if DEVICE.type != "cuda":  # 按当前模式、标识或状态选择处理分支。
        print("当前无 CUDA，跳过正式 batch 显存预检。")  # 输出当前任务的进度或诊断信息。
        return None  # 当前分支无可用结果，返回空值。
    year, seed = TRAINING_CONFIG.forecast_years[0], TRAINING_CONFIG.seeds[0]
    temporal_mode = TRAINING_CONFIG.loss_name == TEMPORAL_TURNOVER_LOSS
    if temporal_mode:  # 根据当前数据状态选择处理分支。
        temporal_pairs = _temporal_preflight_pairs()
        batches = []  # 初始化batches容器。
    elif TRAINING_CONFIG.train_batch_mode == "trade_date":  # 检查上一条件不成立时的备选分支。
        temporal_pairs = []  # 初始化时序pairs容器。
        batches = _trade_date_preflight_batches()
    else:  # 处理上述条件均不成立的情况。
        temporal_pairs = []  # 初始化时序pairs容器。
        loader, _, _ = make_loaders(year, seed)  # 创建数据加载器。
        batches = [(year, "fixed", next(iter(loader)))]  # 初始化batches容器。

    model = build_model()  # 创建模型并部署到设备。
    optimizer = build_optimizer(model)  # 按训练配置创建优化器。
    wait_for_preflight_barrier()
    rows = []  # 初始化行数容器。
    try:  # 执行可能失败的外部读取或计算步骤。
        if temporal_mode:  # 根据当前数据状态选择处理分支。
            optimizer.zero_grad(set_to_none=True)  # 清空上一批梯度。
            for batch_year, shape_test, previous_batch, current_batch, pair in temporal_pairs:  # 遍历时序pairs，逐项处理批次年份形状test上一期批次当前批次pair。
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                previous_features, _ = move_batch(previous_batch)  # 将 batch 搬到训练设备。
                current_features, current_targets = move_batch(current_batch)  # 将 batch 搬到训练设备。
                with autocast_context():  # 在自动混合精度上下文中执行前向计算。
                    previous_with_graph = model(previous_features)  # 执行当前 batch 前向传播。
                previous_predictions = previous_with_graph.detach()
                del previous_with_graph, previous_features  # 释放不再使用的中间引用。
                with autocast_context():  # 在自动混合精度上下文中执行前向计算。
                    current_predictions = model(current_features)  # 执行当前 batch 前向传播。
                components = temporal_turnover_loss_components(  # 按实验口径计算可微损失。
                    current_predictions, current_targets, previous_predictions,
                    pair["previous_positions"], pair["current_positions"],
                    TRAINING_CONFIG.rank_temperature,
                    TRAINING_CONFIG.turnover_lambda,
                )
                loss = components["loss"]  # 计算损失。
                if not torch.isfinite(loss):  # 检查参与计算的数值是否全部有限。
                    raise FloatingPointError("时序换手预检损失出现 NaN 或 Inf")  # 检测到 NaN、Inf 或退化数值时立即终止。
                (loss / len(temporal_pairs)).backward()  # 反向传播当前损失。
                torch.cuda.synchronize()  # 同步 GPU 后计时。
                common_count = int(pair["common_count"])  # 计算公共数量。
                rows.append(  # 把当前结果追加到汇总列表。
                    {
                        "train_batch_mode": TRAINING_CONFIG.train_batch_mode,  # 记录训练批次模式。
                        "loss_name": TRAINING_CONFIG.loss_name,  # 记录损失名称。
                        "rank_temperature": TRAINING_CONFIG.rank_temperature,  # 记录秩temperature。
                        "turnover_lambda": TRAINING_CONFIG.turnover_lambda,  # 记录turnoverlambda。
                        "previous_prediction_gradient": "stop_gradient",  # 把 previous_prediction_gradient 映射为"stop_gradient"。
                        "pairwise_relation_count": (  # 记录pairwiserelation数量。
                            2 * len(current_targets) ** 2 + 2 * common_count**2
                        ),
                        "forecast_year": batch_year,  # 记录forecast年份。
                        "shape_test": shape_test,  # 记录形状test。
                        "previous_batch_size": len(previous_batch["targets"]),  # 记录上一期批次大小。
                        "batch_size_per_task": len(current_targets),  # 记录批次大小pertask。
                        "common_instrument_count": common_count,  # 记录公共股票数量。
                        "loss": float(loss.item()),  # 记录损失。
                        "prediction_soft_rank_ic": float(  # 记录预测soft秩ic。
                            components["prediction_soft_rank_ic"].item()
                        ),
                        "temporal_soft_rank_ic": float(  # 记录时序soft秩ic。
                            components["temporal_soft_rank_ic"].item()
                        ),
                        "temporal_penalty": float(  # 记录时序penalty。
                            components["temporal_penalty"].item()
                        ),
                        "elapsed_seconds": time.perf_counter() - started,  # 记录elapsedseconds。
                        "peak_allocated_gb": (  # 记录peakallocatedgb。
                            torch.cuda.max_memory_allocated() / 1024**3
                        ),
                        "peak_reserved_gb": (  # 记录peakreservedgb。
                            torch.cuda.max_memory_reserved() / 1024**3
                        ),
                        "compile_dynamic_batch": bool(USE_COMPILE),  # 记录compiledynamic批次。
                        "compile_mode": COMPILE_MODE,  # 记录compile模式。
                        "bf16": USE_BF16,  # 记录bf16。
                    }
                )
                del (  # 释放不再使用的中间引用。
                    previous_batch, current_batch, current_features,
                    current_targets, previous_predictions, current_predictions,
                    components, loss,
                )
            optimizer.step()  # 更新参数或学习率状态。
            torch.cuda.synchronize()  # 同步 GPU 后计时。
        else:  # 处理上述条件均不成立的情况。
            for batch_year, shape_test, batch in batches:  # 遍历batches，逐项处理批次年份形状test批次。
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                features, targets = move_batch(batch)  # 将 batch 搬到训练设备。
                optimizer.zero_grad(set_to_none=True)  # 清空上一批梯度。
                with autocast_context():  # 在自动混合精度上下文中执行前向计算。
                    predictions = model(features)  # 执行当前 batch 前向传播。
                loss = compute_training_loss(predictions, targets)  # 按实验口径计算可微损失。
                if not torch.isfinite(loss):  # 检查参与计算的数值是否全部有限。
                    raise FloatingPointError("预检损失出现 NaN 或 Inf")  # 检测到 NaN、Inf 或退化数值时立即终止。
                loss.backward()  # 反向传播当前损失。
                optimizer.step()  # 更新参数或学习率状态。
                torch.cuda.synchronize()  # 同步 GPU 后计时。
                rows.append(  # 把当前结果追加到汇总列表。
                    {
                        "train_batch_mode": TRAINING_CONFIG.train_batch_mode,  # 记录训练批次模式。
                        "loss_name": TRAINING_CONFIG.loss_name,  # 记录损失名称。
                        "rank_temperature": TRAINING_CONFIG.rank_temperature,  # 记录秩temperature。
                        "turnover_lambda": TRAINING_CONFIG.turnover_lambda,  # 记录turnoverlambda。
                        "pairwise_relation_count": (  # 记录pairwiserelation数量。
                            len(targets) ** 2
                            if TRAINING_CONFIG.loss_name in RANK_LOSS_NAMES else 0  # 检查当前值是否属于允许集合。
                        ),
                        "forecast_year": batch_year,  # 记录forecast年份。
                        "shape_test": shape_test,  # 记录形状test。
                        "batch_size_per_task": len(targets),  # 记录批次大小pertask。
                        "loss": float(loss.item()),  # 记录损失。
                        "elapsed_seconds": time.perf_counter() - started,  # 记录elapsedseconds。
                        "peak_allocated_gb": (  # 记录peakallocatedgb。
                            torch.cuda.max_memory_allocated() / 1024**3
                        ),
                        "peak_reserved_gb": (  # 记录peakreservedgb。
                            torch.cuda.max_memory_reserved() / 1024**3
                        ),
                        "compile_dynamic_batch": bool(  # 记录compiledynamic批次。
                            USE_COMPILE
                            and TRAINING_CONFIG.train_batch_mode == "trade_date"
                        ),
                        "bf16": USE_BF16,  # 记录bf16。
                    }
                )
                del batch, features, targets, predictions, loss  # 释放不再使用的中间引用。
    except torch.OutOfMemoryError as error:  # 捕获预期异常并执行对应恢复逻辑。
        raise RuntimeError(  # 当前运行环境或任务状态不支持继续执行。
            f"A2 单任务模式={TRAINING_CONFIG.train_batch_mode}、"
            f"损失={TRAINING_CONFIG.loss_name} 发生 CUDA OOM；"
            "程序不会静默改变训练口径。"
        ) from error
    finally:  # 无论前序步骤是否成功都执行资源清理。
        del model, optimizer, batches, temporal_pairs  # 释放不再使用的中间引用。
        torch.cuda.empty_cache()
    return pd.DataFrame(rows)  # 返回在正式训练前检查全量批次形状、日期与显存相关约束。


def load_ensemble_inputs(data_root: Path) -> tuple[pd.DataFrame, np.ndarray]:  # 加载各随机种子的预测，并检查业务键和行序一致。
    """加载各随机种子的预测，并检查业务键和行序一致。"""
    index = pd.read_parquet(  # 读取索引。
        data_root / "feature_store" / "sample_index.parquet",
        columns=[  # 初始化列容器。
            "sample_index", "trade_date", "instrument_id", "wind_code",
            "can_enter_t1", "label_valid",
        ],
    )
    index["trade_date"] = index["trade_date"].astype(str)  # 转换索引的数据类型或存储位置。
    targets = np.load(data_root / "feature_store" / "targets.npy", mmap_mode="r")  # 读取标签。
    return index.set_index("sample_index", drop=False), targets  # 返回加载并校验后的结果。


def build_ensemble(year: Any, seeds: Any, model_root: Any, prediction_dir: Any, index: Any, targets: Any) -> int:  # 按既定尺度组合多个随机种子的预测，并保存正式年度信号。
    """按既定尺度组合多个随机种子的预测，并保存正式年度信号。"""
    seed_frames = {}  # 初始化随机种子帧容器。
    for seed in seeds:  # 遍历随机种子，逐项处理随机种子。
        path = model_root / f"forecast_{year}" / f"seed_{seed}" / "prediction.parquet"
        if not path.exists():  # 检查目标文件或目录是否满足存在性要求。
            raise FileNotFoundError(path)  # 目标文件不存在时立即终止。
        seed_frames[seed] = pd.read_parquet(path)  # 读取随机种子帧。
    reference = seed_frames[seeds[0]]["sample_index"].to_numpy(np.int64)  # 转换并统一reference的数据表示。
    output = index.loc[
        reference,
        ["sample_index", "trade_date", "instrument_id", "wind_code",
         "can_enter_t1", "label_valid"],
    ].reset_index(drop=True)
    output["target_zscore"] = np.asarray(targets[reference], dtype=np.float32)  # 转换并统一输出的数据表示。
    raw_columns = []  # 初始化原始列容器。
    for seed in seeds:  # 遍历随机种子，逐项处理随机种子。
        frame = seed_frames[seed]
        if not np.array_equal(reference, frame["sample_index"].to_numpy(np.int64)):  # 检查必需数据、配置或校验结果是否缺失。
            raise AssertionError(f"{year} 三个 seed 的预测行号不一致")  # 关键数据契约被破坏时立即终止。
        column = f"prediction_seed_{seed}"
        output[column] = frame["prediction"].to_numpy(np.float32)  # 转换并统一输出的数据表示。
        raw_columns.append(column)  # 把当前结果追加到汇总列表。
    raw = output[raw_columns].astype(np.float64)  # 转换原始的数据类型或存储位置。
    grouped = raw.groupby(output["trade_date"], sort=False)  # 按业务键分组得到grouped。
    means = grouped.transform("mean")  # 按约定截面口径汇总。
    standard_deviations = grouped.transform("std", ddof=0)  # 按约定截面口径汇总。
    if (~np.isfinite(standard_deviations) | standard_deviations.le(0)).any().any():  # 检查参与计算的数值是否全部有限。
        raise ValueError(f"{year} 存在无法标准化的日度预测")  # 输入数据不满足业务或数值约束时立即终止。
    zscores = (raw - means) / standard_deviations  # 计算zscores。
    zscore_columns = []  # 计算zscore列。
    for column in raw_columns:  # 遍历原始列，逐项处理列。
        zscore_column = f"{column}_zscore"  # 计算zscore列。
        output[zscore_column] = zscores[column].to_numpy(np.float32)  # 转换并统一输出的数据表示。
        zscore_columns.append(zscore_column)  # 把当前结果追加到汇总列表。
    output["prediction_ensemble_raw"] = raw.mean(axis=1).to_numpy(np.float32)  # 转换并统一输出的数据表示。
    output["prediction_ensemble"] = output[zscore_columns].mean(axis=1).astype(np.float32)  # 计算输出。
    audit = output.groupby("trade_date", sort=False)[zscore_columns]  # 按业务键分组得到audit。
    if (  # 开始组合多项有效性或配置条件。
        audit.mean().abs().to_numpy().max() > 1e-5
        or (audit.std(ddof=0) - 1).abs().to_numpy().max() > 1e-5
    ):
        raise AssertionError(f"{year} seed 预测日度 Z-score 审计失败")  # 关键数据契约被破坏时立即终止。
    atomic_parquet(output, prediction_dir / f"vit_predictions.{year}.parquet")  # 原子落盘，避免半成品。
    return len(output)  # 返回按当前规则组织后的结果。


def prediction_counts(data_root: Path, years: tuple[int, ...]) -> dict[int, int]:  # 统计各预测年度的文件、日期和样本数量。
    """统计各预测年度的文件、日期和样本数量。"""
    result = {}  # 初始化结果容器。
    for year in years:  # 遍历年份，逐项处理年份。
        with np.load(  # 进入受控上下文并在退出时自动释放资源。
            data_root / "feature_store" / "rolling_splits" / f"forecast_{year}.npz",
            allow_pickle=False,  # 配置allowpickle。
        ) as archive:
            result[year] = len(archive["prediction_indices"])
    return result  # 返回结果。


def split_batch_summary(  # 汇总单个滚动切分中的训练、验证和预测批次。
    data_root: Path, forecast_year: int, train_batch_mode: str,
    fixed_batch_size: int, loss_name: str,
) -> dict[str, object]:
    """汇总单个滚动切分中的训练、验证和预测批次。"""
    with np.load(  # 进入受控上下文并在退出时自动释放资源。
        data_root / "feature_store" / "rolling_splits"
        / f"forecast_{forecast_year}.npz",
        allow_pickle=False,  # 配置allowpickle。
    ) as archive:
        train_indices = archive["train_indices"].copy()  # 转换训练索引的数据类型或存储位置。
    if train_batch_mode == "trade_date":  # 按当前模式、标识或状态选择处理分支。
        date_ids = np.load(  # 读取交易日ids。
            data_root / "feature_store" / "date_ids.npy", mmap_mode="r"
        )
        temporal_mode = loss_name == TEMPORAL_TURNOVER_LOSS
        sampler = TradeDateBatchSampler(  # 按完整交易日截面组 batch。
            train_indices, date_ids, seed=0, shuffle=not temporal_mode
        )
        summary = sampler.summary()  # 按约定截面口径汇总。
        if temporal_mode:  # 根据当前数据状态选择处理分支。
            summary.update(  # 用当前配置更新参数字典。
                {
                    "reshuffle_each_epoch": False,  # 记录reshuffleeach轮次。
                    "gradient_update_mode": "once_per_epoch",  # 把 gradient_update_mode 映射为"once_per_epoch"。
                    "optimizer_steps_per_epoch": 1,  # 记录优化器stepsper轮次。
                    "temporal_lag_trade_days": 1,  # 记录时序lagtradedays。
                    "temporal_anchor_mode": (  # 记录时序anchor模式。
                        "previous_prediction_stop_gradient_cached"
                    ),
                    "model_forward_passes_per_train_date": 1,  # 记录模型forwardpassesper训练交易日。
                    "train_temporal_pair_count": len(sampler) - 1,  # 记录训练时序pair数量。
                }
            )
        del date_ids  # 释放不再使用的中间引用。
    else:  # 处理上述条件均不成立的情况。
        batch_count = len(train_indices) // fixed_batch_size  # 计算批次数量。
        summary = {  # 初始化汇总容器。
            "train_batch_mode": "fixed_samples",  # 把 train_batch_mode 映射为"fixed_samples"。
            "train_batch_count": batch_count,  # 记录训练批次数量。
            "train_date_count": None,  # 记录训练交易日数量。
            "train_sample_count": batch_count * fixed_batch_size,  # 记录训练样本数量。
            "train_batch_size_min": fixed_batch_size,  # 记录训练批次大小min。
            "train_batch_size_median": float(fixed_batch_size),  # 记录训练批次大小median。
            "train_batch_size_max": fixed_batch_size,  # 记录训练批次大小max。
            "dropped_train_samples": (  # 记录dropped训练样本。
                len(train_indices) - batch_count * fixed_batch_size
            ),
            "reshuffle_each_epoch": True,  # 记录reshuffleeach轮次。
            "gradient_update_mode": "per_batch",  # 把 gradient_update_mode 映射为"per_batch"。
            "optimizer_steps_per_epoch": batch_count,  # 记录优化器stepsper轮次。
        }
    return {"forecast_year": forecast_year, **summary}  # 返回汇总单个滚动切分中的训练、验证和预测批次。


def training_command(args: Any, command: str) -> list[str]:  # 构造启动训练后端的完整命令行参数。
    """构造启动训练后端的完整命令行参数。"""
    return [  # 返回构造启动训练后端的完整命令行参数。
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


def worker_command(args: Any, year: int, seed: int, gpu: int) -> list[str]:  # 构造单个年度×seed worker 的命令行参数。
    """构造单个年度×seed worker 的命令行参数。"""
    return [  # 返回构造单个年度×seed worker 的命令行参数。
        *training_command(args, "worker"),
        "--worker-year", str(year),
        "--worker-seed", str(seed),
        "--worker-physical-gpu", str(gpu),
    ]


def preflight_worker_command(args: Any, rank: int) -> list[str]:  # 构造正式训练前单 worker 预检命令。
    """构造正式训练前单 worker 预检命令。"""
    return [  # 返回构造正式训练前单 worker 预检命令。
        *training_command(args, "preflight-worker"),
        "--preflight-rank", str(rank),
    ]


def tail(path: Path, lines: int = 80) -> str:  # 读取文本文件末尾若干行，用于展示最新训练日志。
    """读取文本文件末尾若干行，用于展示最新训练日志。"""
    if not path.exists():  # 检查目标文件或目录是否满足存在性要求。
        return ""  # 返回读取并校验后的结果。
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])  # 返回读取并校验后的结果。


def run_worker(args: Any) -> None:  # 启动并等待单个训练 worker，传递其退出状态。
    """启动并等待单个训练 worker，传递其退出状态。"""
    configure_worker_environment(args)
    initialize_runtime()
    model, summary = train_single_run(args.worker_year, args.worker_seed)  # 创建模型。
    split = load_split(args.worker_year)
    prediction_count = len(split["prediction_indices"])  # 计算预测数量。
    summary["prediction_sample_count"] = prediction_count  # 计算汇总。
    if prediction_count:  # 根据当前数据状态选择处理分支。
        _, _, loader = make_loaders(args.worker_year, args.worker_seed)
        if loader is None:  # 检查所需对象是否尚未初始化。
            raise AssertionError("预测索引非空但 loader 缺失")  # 关键数据契约被破坏时立即终止。
        prediction = predict_model(model, loader)
        atomic_parquet(  # 原子落盘，避免半成品。
            prediction,
            MODEL_ROOT / f"forecast_{args.worker_year}" / f"seed_{args.worker_seed}"
            / "prediction.parquet",
        )
    print(json.dumps({"summary": summary}, ensure_ascii=False), flush=True)  # 输出当前任务的进度或诊断信息。


def finalize_outputs(args: Any, elapsed_seconds: float) -> pd.DataFrame:  # 聚合已完成任务并生成 run 级预测、清单和报告。
    """聚合已完成任务并生成 run 级预测、清单和报告。"""
    data_root = Path(args.data_root).resolve()  # 构造数据根目录。
    years, seeds = parse_csv_ints(args.years), parse_csv_ints(args.seeds)
    model_root = VIT_DIR / "models" / "runs" / args.run_name  # 构造模型根目录。
    prediction_dir = data_root / "predictions" / "runs" / args.run_name  # 构造预测目录。
    quality_dir = data_root / "quality" / "runs" / args.run_name  # 构造质量目录。
    counts = prediction_counts(data_root, years)  # 计算数量。
    records = []  # 初始化records容器。
    for year in years:  # 遍历年份，逐项处理年份。
        for seed in seeds:  # 遍历随机种子，逐项处理随机种子。
            completed_path = model_root / f"forecast_{year}" / f"seed_{seed}" / "completed.json"  # 构造completed路径。
            completed = read_json(completed_path)  # 读取completed。
            if completed is None:  # 检查所需对象是否尚未初始化。
                raise FileNotFoundError(completed_path)  # 目标文件不存在时立即终止。
            record = dict(completed["summary"])  # 初始化record容器。
            record["prediction_sample_count"] = counts[year]  # 计算record。
            records.append(record)  # 把当前结果追加到汇总列表。
    summary = pd.DataFrame(records).sort_values(["forecast_year", "seed"])  # 计算汇总以固定样本顺序。
    atomic_parquet(summary, quality_dir / "vit_training_summary.parquet")  # 原子落盘，避免半成品。
    index, targets = load_ensemble_inputs(data_root)
    for year in years:  # 遍历年份，逐项处理年份。
        if not counts[year]:  # 检查必需数据、配置或校验结果是否缺失。
            continue  # 跳过当前项并处理下一项。
        count = build_ensemble(year, seeds, model_root, prediction_dir, index, targets)  # 计算数量。
        if count != counts[year]:  # 按当前模式、标识或状态选择处理分支。
            raise AssertionError(f"{year} 集成预测行数不完整")  # 关键数据契约被破坏时立即终止。
        print(f"[{year}] 集成预测 {count:,} 条", flush=True)  # 输出当前任务的进度或诊断信息。
    dataset = read_json(data_root / "feature_store" / "dataset_manifest.json")  # 读取dataset。
    rolling = read_json(data_root / "feature_store" / "rolling_split_manifest.json")  # 读取rolling。
    universe = read_json(data_root / "universe" / "universe_manifest.json")  # 读取股票池。
    atomic_json(  # 原子落盘，避免半成品。
        {
            "route": "A2",  # 把 route 映射为"A2"。
            "run_name": args.run_name,  # 记录运行名称。
            "device_ids": list(parse_csv_ints(args.device_ids)),  # 记录设备ids。
            "tasks_per_gpu": args.tasks_per_gpu,  # 记录taskspergpu。
            "years": list(years),  # 记录年份。
            "seeds": list(seeds),  # 记录随机种子。
            "train_batch_mode": args.train_batch_mode,  # 记录训练批次模式。
            "batch_size_per_task": (  # 记录批次大小pertask。
                "full_trade_date"
                if args.train_batch_mode == "trade_date"  # 按当前模式、标识或状态选择处理分支。
                else args.batch_size
            ),
            "fixed_batch_size": args.batch_size,  # 记录fixed批次大小。
            "inference_batch_size": args.inference_batch_size,  # 记录inference批次大小。
            "reshuffle_each_epoch": args.loss_name != TEMPORAL_TURNOVER_LOSS,  # 记录reshuffleeach轮次。
            "gradient_update_mode": (  # 记录梯度update模式。
                "once_per_epoch"
                if args.loss_name == TEMPORAL_TURNOVER_LOSS else "per_batch"  # 按当前模式、标识或状态选择处理分支。
            ),
            "optimizer_steps_per_epoch": (  # 记录优化器stepsper轮次。
                1 if args.loss_name == TEMPORAL_TURNOVER_LOSS else "per_batch"
            ),
            "temporal_lag_trade_days": (  # 记录时序lagtradedays。
                1 if args.loss_name == TEMPORAL_TURNOVER_LOSS else None
            ),
            "temporal_anchor_mode": (  # 记录时序anchor模式。
                "previous_prediction_stop_gradient_cached"
                if args.loss_name == TEMPORAL_TURNOVER_LOSS else None  # 按当前模式、标识或状态选择处理分支。
            ),
            "model_forward_passes_per_train_date": (  # 记录模型forwardpassesper训练交易日。
                1 if args.loss_name == TEMPORAL_TURNOVER_LOSS else None
            ),
            "loss_name": args.loss_name,  # 记录损失名称。
            "loss_parameters": {  # 记录损失parameters。
                "rank_temperature": args.rank_temperature,  # 记录秩temperature。
                "turnover_lambda": args.turnover_lambda,  # 记录turnoverlambda。
                "huber_delta": args.huber_delta,  # 记录huberdelta。
                "smooth_l1_beta": args.smooth_l1_beta,  # 记录smoothl1beta。
                "hybrid_weight": args.hybrid_weight,  # 记录hybrid权重。
                "prediction_input": "raw_model_output",  # 把 prediction_input 映射为"raw_model_output"。
                "target_input": "daily_winsorized_1pct_zscore",  # 把 target_input 映射为"daily_winsorized_1pct_zscore"。
                "prediction_preprocessing_inside_loss": "none",  # 把 prediction_preprocessing_inside_loss 映射为"none"。
                "target_preprocessing_inside_loss": "none",  # 把 target_preprocessing_inside_loss 映射为"none"。
                "rank_target_gradient": False,  # 记录秩标签梯度。
                "temporal_prediction_gradients": (  # 记录时序预测gradients。
                    "current_only"
                    if args.loss_name == TEMPORAL_TURNOVER_LOSS else None  # 按当前模式、标识或状态选择处理分支。
                ),
                "previous_prediction_gradient": (  # 记录上一期预测梯度。
                    "stop_gradient"
                    if args.loss_name == TEMPORAL_TURNOVER_LOSS else None  # 按当前模式、标识或状态选择处理分支。
                ),
                "previous_prediction_source": (  # 记录上一期预测来源。
                    "cached_previous_date_output"
                    if args.loss_name == TEMPORAL_TURNOVER_LOSS else None  # 按当前模式、标识或状态选择处理分支。
                ),
                "turnover_penalty_direction": (  # 记录turnoverpenaltydirection。
                    "1_minus_temporal_rank_ic"
                    if args.loss_name == TEMPORAL_TURNOVER_LOSS else None  # 按当前模式、标识或状态选择处理分支。
                ),
            },
            "validation_metric": args.validation_metric,  # 记录验证指标。
            "validation_audit_metrics": [  # 记录验证audit指标。
                "daily_mean_spearman_ic",
                "daily_mean_pearson_ic",
                "pooled_pearson_ic",
                "temporal_spearman_rank_ic",
                "temporal_rank_penalty",
            ],
            "validation_results_by_task": summary[  # 记录验证结果bytask。
                [
                    "forecast_year", "seed", "best_validation_ic",
                    "best_validation_rank_ic",
                    "best_validation_daily_pearson_ic",
                    "best_validation_pooled_ic",
                    "best_validation_temporal_rank_ic",
                    "best_validation_temporal_penalty",
                ]
            ].to_dict("records"),
            "temporal_training_audit_by_task": summary[  # 记录时序训练auditbytask。
                [
                    "forecast_year", "seed", "train_temporal_pair_count",
                    "temporal_common_count_min", "temporal_common_count_median",
                    "temporal_common_count_max", "temporal_common_coverage_min",
                    "temporal_common_coverage_median",
                    "temporal_common_coverage_max",
                ]
            ].to_dict("records"),
            "training_batch_size_by_year": [  # 记录训练批次大小by年份。
                split_batch_summary(
                    data_root, year, args.train_batch_mode, args.batch_size,
                    args.loss_name,
                )
                for year in years  # 遍历年份，逐项处理年份。
            ],
            "workers_per_task": args.workers_per_task,  # 记录workerspertask。
            "learning_rate": args.learning_rate,  # 记录learning比例。
            "max_epochs": args.max_epochs,  # 记录maxepochs。
            "patience": args.patience,  # 记录patience。
            "torch_compile": bool_int(args.compile),  # 记录torchcompile。
            "compile_mode": args.compile_mode,  # 记录compile模式。
            "compile_dynamic_batch": bool(  # 记录compiledynamic批次。
                bool_int(args.compile) and args.train_batch_mode == "trade_date"
            ),
            "fused_adam": bool_int(args.fused_adam),  # 记录fusedadam。
            "cuda_prefetch": bool_int(args.cuda_prefetch),  # 记录cudaprefetch。
            "universe_signature": universe["signature"],  # 记录股票池签名，用于来源一致性校验。
            "dataset_signature": dataset["signature"],  # 记录dataset签名，用于来源一致性校验。
            "rolling_signature": rolling["signature"],  # 记录rolling签名，用于来源一致性校验。
            "elapsed_seconds": elapsed_seconds,  # 记录elapsedseconds。
            "completed_tasks": len(summary),  # 记录completedtasks。
        },
        quality_dir / "a2_training_manifest.json",
    )
    return summary  # 返回汇总。


def run_parent(args: Any) -> pd.DataFrame:  # 调度多个 GPU worker，并在全部结束后汇总正式产物。
    """调度多个 GPU worker，并在全部结束后汇总正式产物。"""
    device_ids = parse_csv_ints(args.device_ids)
    years, seeds = parse_csv_ints(args.years), parse_csv_ints(args.seeds)
    tasks = deque((year, seed) for year in sorted(years) for seed in sorted(seeds))  # 计算tasks以固定样本顺序。
    data_root = Path(args.data_root).resolve()  # 构造数据根目录。
    log_dir = data_root / "quality" / "runs" / args.run_name / "a2_task_logs"  # 构造log目录。
    log_dir.mkdir(parents=True, exist_ok=True)  # 确保目标目录存在。
    running: list[dict[str, object]] = []  # 初始化running容器。
    started = time.perf_counter()
    print(  # 输出当前任务的进度或诊断信息。
        f"A2 启动：{len(tasks)} 个任务，GPU={device_ids}，"
        f"每卡并发={args.tasks_per_gpu}，损失={args.loss_name}，"
        f"batch={args.train_batch_mode}",
        flush=True,  # 配置flush。
    )
    try:  # 执行可能失败的外部读取或计算步骤。
        while tasks or running:  # 在终止条件出现前持续迭代。
            for gpu in device_ids:  # 遍历设备ids，逐项处理gpu。
                active = sum(item["gpu"] == gpu for item in running)
                while tasks and active < args.tasks_per_gpu:  # 在终止条件出现前持续迭代。
                    year, seed = tasks.popleft()
                    log_path = log_dir / f"forecast_{year}.seed_{seed}.log"  # 构造log路径。
                    log_handle = log_path.open("w", encoding="utf-8")
                    environment = os.environ.copy()  # 转换environment的数据类型或存储位置。
                    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
                    process = subprocess.Popen(  # 启动独立训练任务。
                        worker_command(args, year, seed, gpu),
                        cwd=VIT_DIR,  # 配置cwd。
                        env=environment,  # 配置env。
                        stdout=log_handle,  # 配置stdout。
                        stderr=subprocess.STDOUT,  # 配置stderr。
                        text=True,  # 配置text。
                    )
                    running.append(  # 把当前结果追加到汇总列表。
                        {"process": process, "gpu": gpu, "year": year, "seed": seed,
                         "log_path": log_path, "log_handle": log_handle,  # 记录log路径。
                         "started": time.perf_counter()}  # 记录started。
                    )
                    active += 1
                    print(f"启动 GPU{gpu}: {year}/seed={seed} pid={process.pid}", flush=True)  # 输出当前任务的进度或诊断信息。
            finished = []  # 初始化finished容器。
            for item in running:  # 遍历running，逐项处理item。
                return_code = item["process"].poll()  # 检查训练任务状态。
                if return_code is None:  # 检查所需对象是否尚未初始化。
                    continue  # 跳过当前项并处理下一项。
                item["log_handle"].close()
                finished.append(item)  # 把当前结果追加到汇总列表。
                if return_code:  # 根据当前数据状态选择处理分支。
                    for active_item in running:  # 遍历running，逐项处理activeitem。
                        if active_item["process"].poll() is None:  # 检查所需对象是否尚未初始化。
                            active_item["process"].terminate()
                    raise RuntimeError(  # 当前运行环境或任务状态不支持继续执行。
                        f"A2 任务失败：{item['year']}/seed={item['seed']} GPU{item['gpu']} "
                        f"exit={return_code}\n{tail(item['log_path'])}"
                    )
                print(  # 输出当前任务的进度或诊断信息。
                    f"完成 GPU{item['gpu']}: {item['year']}/seed={item['seed']} "
                    f"{(time.perf_counter() - item['started']) / 60:.2f} 分钟",
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
    elapsed = time.perf_counter() - started
    summary = finalize_outputs(args, elapsed)  # 汇总任务并写出完成标记。
    print(f"A2 完成：{len(summary)} 个任务，墙钟 {elapsed / 60:.2f} 分钟")  # 输出当前任务的进度或诊断信息。
    return summary  # 返回汇总。


def run_preflight_worker(args: Any) -> None:  # 在一个独立进程内执行多 shape 显存压力测试。
    """在一个独立进程内执行多 shape 显存压力测试。"""

    configure_worker_environment(args)
    os.environ["VIT_PREFLIGHT_RANK"] = str(args.preflight_rank)
    initialize_runtime()
    set_random_seed(  # 固定各随机源。
        TRAINING_CONFIG.seeds[args.preflight_rank % len(TRAINING_CONFIG.seeds)]
    )
    result = run_full_batch_preflight()
    if result is None:  # 检查所需对象是否尚未初始化。
        raise RuntimeError("并发显存预检要求 CUDA")  # 当前运行环境或任务状态不支持继续执行。
    result.insert(0, "concurrent_worker_rank", args.preflight_rank)
    print(  # 输出当前任务的进度或诊断信息。
        "VIT_PREFLIGHT_RESULT="
        + json.dumps(result.to_dict("records"), ensure_ascii=False),
        flush=True,  # 配置flush。
    )


def run_concurrent_temporal_preflight(args: Any) -> pd.DataFrame:  # 在同一物理 GPU 同时启动与正式训练相同数量的 worker。
    """在同一物理 GPU 同时启动与正式训练相同数量的 worker。"""

    physical_gpu = parse_csv_ints(args.device_ids)[0]
    worker_count = int(args.tasks_per_gpu)  # 计算worker数量。
    if worker_count < 2:  # 检查当前数值是否越过业务或数值阈值。
        raise ValueError("同卡并发预检至少需要两个 worker")  # 输入数据不满足业务或数值约束时立即终止。
    running: list[dict[str, object]] = []  # 初始化running容器。
    records: list[dict[str, object]] = []  # 初始化records容器。
    with tempfile.TemporaryDirectory(prefix="vit_temporal_preflight_") as temporary:  # 进入受控上下文并在退出时自动释放资源。
        temporary_dir = Path(temporary)  # 构造临时文件目录。
        barrier_dir = temporary_dir / "barrier"  # 构造barrier目录。
        try:  # 执行可能失败的外部读取或计算步骤。
            for rank in range(worker_count):  # 按给定步长遍历样本区间，更新排名。
                log_path = temporary_dir / f"worker_{rank}.log"  # 构造log路径。
                log_handle = log_path.open("w", encoding="utf-8")
                environment = os.environ.copy()  # 转换environment的数据类型或存储位置。
                environment["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
                environment["VIT_PREFLIGHT_BARRIER_DIR"] = str(barrier_dir)
                environment["VIT_PREFLIGHT_WORLD_SIZE"] = str(worker_count)
                process = subprocess.Popen(  # 启动独立训练任务。
                    preflight_worker_command(args, rank),
                    cwd=VIT_DIR,  # 配置cwd。
                    env=environment,  # 配置env。
                    stdout=log_handle,  # 配置stdout。
                    stderr=subprocess.STDOUT,  # 配置stderr。
                    text=True,  # 配置text。
                )
                running.append(  # 把当前结果追加到汇总列表。
                    {
                        "rank": rank,  # 记录秩。
                        "process": process,  # 记录process。
                        "log_path": log_path,  # 记录log路径。
                        "log_handle": log_handle,  # 记录loghandle。
                    }
                )
            deadline = time.monotonic() + 1200
            while running:  # 在终止条件出现前持续迭代。
                if time.monotonic() >= deadline:  # 检查当前数值是否越过业务或数值阈值。
                    raise TimeoutError("同卡并发显存预检超过20分钟")  # 当前数据违反接口约定时立即终止。
                finished = []  # 初始化finished容器。
                for item in running:  # 遍历running，逐项处理item。
                    return_code = item["process"].poll()  # 检查训练任务状态。
                    if return_code is None:  # 检查所需对象是否尚未初始化。
                        continue  # 跳过当前项并处理下一项。
                    item["log_handle"].close()
                    finished.append(item)  # 把当前结果追加到汇总列表。
                    log_text = item["log_path"].read_text(  # 读取logtext。
                        encoding="utf-8", errors="replace"
                    )
                    if return_code:  # 根据当前数据状态选择处理分支。
                        raise RuntimeError(  # 当前运行环境或任务状态不支持继续执行。
                            f"并发预检 worker={item['rank']} 失败：\n"
                            + "\n".join(log_text.splitlines()[-100:])
                        )
                    marker_lines = [  # 初始化markerlines容器。
                        line for line in log_text.splitlines()
                        if line.startswith("VIT_PREFLIGHT_RESULT=")  # 根据当前数据状态选择处理分支。
                    ]
                    if len(marker_lines) != 1:  # 检查当前样本数量是否满足计算要求。
                        raise RuntimeError(  # 当前运行环境或任务状态不支持继续执行。
                            f"并发预检 worker={item['rank']} 缺少结果标记"
                        )
                    payload = marker_lines[0].split("=", 1)[1]
                    worker_records = json.loads(payload)
                    for record in worker_records:  # 遍历workerrecords，逐项处理record。
                        record["physical_gpu"] = physical_gpu
                        record["concurrent_tasks_on_gpu"] = worker_count
                    records.extend(worker_records)  # 把一组结果并入当前列表。
                for item in finished:  # 遍历finished，逐项处理item。
                    running.remove(item)  # 移除已结束任务。
                if running and not finished:  # 检查必需数据、配置或校验结果是否缺失。
                    time.sleep(0.5)
        finally:  # 无论前序步骤是否成功都执行资源清理。
            for item in running:  # 遍历running，逐项处理item。
                if item["process"].poll() is None:  # 检查所需对象是否尚未初始化。
                    item["process"].terminate()  # 终止仍在运行的任务。
                item["log_handle"].close()
    result = pd.DataFrame(records).sort_values(  # 计算结果以固定样本顺序。
        ["concurrent_worker_rank", "shape_test"]
    )
    if len(result) != worker_count * 12:  # 检查当前样本数量是否满足计算要求。
        raise AssertionError(  # 关键数据契约被破坏时立即终止。
            f"并发预检结果不完整：{len(result)}/{worker_count * 12}"
        )
    return result.reset_index(drop=True)  # 返回在同一物理 GPU 同时启动与正式训练相同数量的 worker。


def run_preflight(args: Any) -> None:  # 执行数据、切分、模型和训练参数的只读预检。
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
    if (  # 开始组合多项有效性或配置条件。
        TRAINING_CONFIG.loss_name == TEMPORAL_TURNOVER_LOSS
        and args.tasks_per_gpu > 1
    ):
        display(run_concurrent_temporal_preflight(args))
    else:  # 处理上述条件均不成立的情况。
        result = run_full_batch_preflight()
        if result is not None:  # 检查必需数据、配置或校验结果是否缺失。
            display(result)


# ============================== 训练 CLI ==============================
def validate_training_inputs() -> pd.DataFrame:  # 检查 memmap、label_valid 和 8 个滚动年度。
    """检查 memmap、label_valid 和 8 个滚动年度。"""

    features = np.load(FEATURES_PATH, mmap_mode="r")  # 读取特征。
    targets = np.load(TARGETS_PATH, mmap_mode="r")  # 读取标签。
    label_valid = np.load(LABEL_VALID_PATH, mmap_mode="r")  # 读取标签有效。
    date_ids = np.load(DATE_IDS_PATH, mmap_mode="r")  # 读取交易日ids。
    if features.shape[1:] != (17, 8, 8):  # 检查张量或数组形状是否符合接口约定。
        raise ValueError(f"特征形状异常：{features.shape}")  # 输入数据不满足业务或数值约束时立即终止。
    if not (len(features) == len(targets) == len(label_valid) == len(date_ids) == len(sample_index)):  # 检查当前样本数量是否满足计算要求。
        raise ValueError("feature store 各文件行数不一致")  # 输入数据不满足业务或数值约束时立即终止。
    if not np.array_equal(np.isfinite(targets), label_valid.astype(bool)):  # 检查参与计算的数值是否全部有限。
        raise AssertionError("label_valid 与 target NaN 契约不一致")  # 关键数据契约被破坏时立即终止。
    rolling = pd.read_parquet(ROLLING_SUMMARY_PATH)  # 读取rolling。
    missing_years = set(TRAINING_CONFIG.forecast_years) - set(  # 初始化missing年份容器。
        rolling["forecast_year"].astype(int)
    )
    if missing_years:  # 根据当前数据状态选择处理分支。
        raise ValueError(f"缺少滚动年度：{sorted(missing_years)}")  # 输入数据不满足业务或数值约束时立即终止。
    if DEVICE.type == "cuda":  # 按当前模式、标识或状态选择处理分支。
        capability = torch.cuda.get_device_capability(0)
        if capability == (12, 0) and "sm_120" not in torch.cuda.get_arch_list():  # 按当前模式、标识或状态选择处理分支。
            raise RuntimeError("当前 PyTorch wheel 未包含 RTX 5090 的 sm_120")  # 当前运行环境或任务状态不支持继续执行。
        properties = torch.cuda.get_device_properties(0)
        print(  # 输出当前任务的进度或诊断信息。
            f"GPU：{properties.name}，显存 {properties.total_memory / 1024**3:.1f}GB，"
            f"计算能力 sm_{capability[0]}{capability[1]}"
        )
    summary = pd.DataFrame(
        [
            {
                "samples": len(features),  # 记录样本。
                "label_valid": int(label_valid.sum()),  # 记录标签有效。
                "feature_shape": str(features.shape),  # 记录特征形状。
                "first_date": sample_index["trade_date"].min(),  # 记录首个交易日。
                "last_date": sample_index["trade_date"].max(),  # 记录最后一个交易日。
                "dataset_signature": dataset_manifest["signature"][:12],  # 记录dataset签名，用于来源一致性校验。
                "rolling_signature": rolling_manifest["signature"][:12],  # 记录rolling签名，用于来源一致性校验。
            }
        ]
    )
    display(summary)
    display(rolling)
    del features, targets, label_valid, date_ids  # 释放不再使用的中间引用。
    return summary  # 返回汇总。

def loader_batch_summary(loader: DataLoader) -> dict[str, object]:  # 汇总 DataLoader 的批次数、样本数和批大小分布。
    """汇总 DataLoader 的批次数、样本数和批大小分布。"""
    sampler = loader.batch_sampler  # 创建sampler。
    if isinstance(sampler, TradeDateBatchSampler):  # 检查对象类型是否符合接口约定。
        return sampler.summary()  # 返回汇总 DataLoader 的批次数、样本数和批大小分布。
    sample_count = len(loader.dataset)  # 计算样本数量。
    batch_size = int(loader.batch_size)
    batch_count = sample_count // batch_size  # 计算批次数量。
    return {  # 返回汇总 DataLoader 的批次数、样本数和批大小分布。
        "train_batch_mode": "fixed_samples",  # 把 train_batch_mode 映射为"fixed_samples"。
        "train_batch_count": batch_count,  # 记录训练批次数量。
        "train_date_count": None,  # 记录训练交易日数量。
        "train_sample_count": batch_count * batch_size,  # 记录训练样本数量。
        "train_batch_size_min": batch_size,  # 记录训练批次大小min。
        "train_batch_size_median": float(batch_size),  # 记录训练批次大小median。
        "train_batch_size_max": batch_size,  # 记录训练批次大小max。
        "dropped_train_samples": sample_count - batch_count * batch_size,  # 记录dropped训练样本。
        "reshuffle_each_epoch": True,  # 记录reshuffleeach轮次。
        "gradient_update_mode": "per_batch",  # 把 gradient_update_mode 映射为"per_batch"。
        "optimizer_steps_per_epoch": batch_count,  # 记录优化器stepsper轮次。
    }

def validate_rolling_split(year: int) -> dict[str, object]:  # 检查训练、验证与预测切分的行号、日期和防泄漏约束。
    """检查训练、验证与预测切分的行号、日期和防泄漏约束。"""
    split = load_split(year)
    train_rows = sample_index.iloc[split["train_indices"]]
    validation_rows = sample_index.iloc[split["validation_indices"]]
    prediction_rows = sample_index.iloc[split["prediction_indices"]]
    if not train_rows["label_valid"].all() or not validation_rows["label_valid"].all():  # 检查当前掩码或数据是否满足有效性要求。
        raise AssertionError(f"{year} 训练或验证索引混入无标签样本")  # 关键数据契约被破坏时立即终止。
    validation_start = validation_rows["trade_date"].min()
    forecast_start = f"{year}0101"
    if train_rows["exit_trade_date"].max() >= validation_start:  # 检查当前数值是否越过业务或数值阈值。
        raise AssertionError(f"{year} 训练标签越过验证边界")  # 关键数据契约被破坏时立即终止。
    if validation_rows["exit_trade_date"].max() >= forecast_start:  # 检查当前数值是否越过业务或数值阈值。
        raise AssertionError(f"{year} 验证标签越过预测边界")  # 关键数据契约被破坏时立即终止。
    if len(prediction_rows) and not prediction_rows["trade_date"].str.startswith(str(year)).all():  # 检查当前样本数量是否满足计算要求。
        raise AssertionError(f"{year} 预测索引混入其他年度")  # 关键数据契约被破坏时立即终止。
    return {  # 返回检查训练、验证与预测切分的行号、日期和防泄漏约束。
        "forecast_year": year,  # 记录forecast年份。
        "train_samples": len(train_rows),  # 记录训练样本。
        "validation_samples": len(validation_rows),  # 记录验证样本。
        "prediction_samples": len(prediction_rows),  # 记录预测样本。
        "max_train_exit": train_rows["exit_trade_date"].max(),  # 记录max训练exit。
        "validation_start": validation_start,  # 记录验证起点。
        "max_validation_exit": validation_rows["exit_trade_date"].max(),  # 记录max验证exit。
        "forecast_start": forecast_start,  # 记录forecast起点。
    }

def configured_batch_summaries() -> pd.DataFrame:  # 按当前批处理配置生成各切分的批次摘要。
    """按当前批处理配置生成各切分的批次摘要。"""
    records = []  # 初始化records容器。
    seed = TRAINING_CONFIG.seeds[0]
    for year in TRAINING_CONFIG.forecast_years:  # 遍历训练配置forecast年份，逐项处理年份。
        train_loader, _, _ = make_loaders(year, seed, workers=0)  # 创建训练数据加载器。
        records.append({"forecast_year": year, **loader_batch_summary(train_loader)})  # 把当前结果追加到汇总列表。
    return pd.DataFrame(records)  # 返回按当前规则组织后的结果。

def validate_outputs(args: Any) -> pd.DataFrame:  # 检查 checkpoint、预测、诊断和 manifest 是否完整一致。
    """检查 checkpoint、预测、诊断和 manifest 是否完整一致。"""
    data_root = Path(args.data_root).resolve()  # 构造数据根目录。
    years, seeds = parse_csv_ints(args.years), parse_csv_ints(args.seeds)
    model_root = VIT_DIR / "models" / "runs" / args.run_name  # 构造模型根目录。
    prediction_dir = data_root / "predictions" / "runs" / args.run_name  # 构造预测目录。
    quality_dir = data_root / "quality" / "runs" / args.run_name  # 构造质量目录。
    sample_index_frame = pd.read_parquet(  # 读取样本索引帧。
        data_root / "feature_store" / "sample_index.parquet",
        columns=["sample_index", "trade_date"],  # 限定本次读取或输出使用的字段。
    )
    sample_index_frame["trade_date"] = sample_index_frame["trade_date"].astype(str)  # 转换样本索引帧的数据类型或存储位置。
    counts = prediction_counts(data_root, years)  # 计算数量。
    records = []  # 初始化records容器。
    for year in years:  # 遍历年份，逐项处理年份。
        for seed in seeds:  # 遍历随机种子，逐项处理随机种子。
            task_dir = model_root / f"forecast_{year}" / f"seed_{seed}"  # 构造task目录。
            if not (task_dir / "best.pt").exists() or not (task_dir / "completed.json").exists():  # 检查目标文件或目录是否满足存在性要求。
                raise FileNotFoundError(f"{year}/seed={seed} checkpoint 不完整")  # 目标文件不存在时立即终止。
        prediction_path = prediction_dir / f"vit_predictions.{year}.parquet"  # 构造预测路径。
        if counts[year]:  # 根据当前数据状态选择处理分支。
            prediction = pd.read_parquet(prediction_path, columns=["sample_index"])  # 读取预测。
            expected = sample_index_frame.loc[
                sample_index_frame["trade_date"].str.startswith(str(year)), "sample_index"
            ].to_numpy(np.int64)
            if not np.array_equal(prediction["sample_index"].to_numpy(np.int64), expected):  # 检查必需数据、配置或校验结果是否缺失。
                raise AssertionError(f"{year} 预测键与 T 日 Universe 不一致")  # 关键数据契约被破坏时立即终止。
        elif prediction_path.exists():  # 检查上一条件不成立时的备选分支。
            raise AssertionError(f"{year} 无图片却出现预测文件")  # 关键数据契约被破坏时立即终止。
        records.append(  # 把当前结果追加到汇总列表。
            {
                "forecast_year": year,  # 记录forecast年份。
                "checkpoints": len(seeds),  # 记录checkpoints。
                "prediction_samples": counts[year],  # 记录预测样本。
                "prediction_file": prediction_path.exists(),  # 记录预测文件。
                "status": "模型与预测完成" if counts[year] else "模型完成，暂无图片预测",  # 把 status 映射为"模型与预测完成" if counts[year] else "模型完成，暂无图片预测"。
            }
        )
    summary_path = quality_dir / "vit_training_summary.parquet"  # 构造汇总路径。
    manifest_path = quality_dir / "a2_training_manifest.json"  # 构造清单路径。
    if not summary_path.exists() or not manifest_path.exists():  # 检查目标文件或目录是否满足存在性要求。
        raise FileNotFoundError("A2 训练汇总或 manifest 缺失")  # 目标文件不存在时立即终止。
    summary = pd.read_parquet(summary_path)  # 读取汇总。
    if len(summary[["forecast_year", "seed"]].drop_duplicates()) != len(years) * len(seeds):  # 检查当前样本数量是否满足计算要求。
        raise AssertionError("A2 年度×种子任务数不完整")  # 关键数据契约被破坏时立即终止。
    result = pd.DataFrame(records)
    display(result)
    return result  # 返回结果。

def show_status(args: Any) -> pd.DataFrame:  # 显示年度任务、worker、checkpoint 和预测的当前状态。
    """显示年度任务、worker、checkpoint 和预测的当前状态。"""
    years, seeds = parse_csv_ints(args.years), parse_csv_ints(args.seeds)
    model_root = VIT_DIR / "models" / "runs" / args.run_name  # 构造模型根目录。
    records = []  # 初始化records容器。
    for year in years:  # 遍历年份，逐项处理年份。
        for seed in seeds:  # 遍历随机种子，逐项处理随机种子。
            task_dir = model_root / f"forecast_{year}" / f"seed_{seed}"  # 构造task目录。
            completed = read_json(task_dir / "completed.json")  # 读取completed。
            history_path = task_dir / "history.parquet"  # 构造历史记录路径。
            last_epoch, best_ic = None, None
            if completed:  # 根据当前数据状态选择处理分支。
                last_epoch = completed["summary"].get("last_epoch")
                best_ic = completed["summary"].get("best_validation_ic")
            elif history_path.exists():  # 检查上一条件不成立时的备选分支。
                history = pd.read_parquet(history_path)  # 读取历史记录。
                if len(history):  # 检查当前样本数量是否满足计算要求。
                    last_epoch = int(history["epoch"].iloc[-1])
                    best_ic = float(history["best_validation_ic"].iloc[-1])
            records.append(  # 把当前结果追加到汇总列表。
                {"forecast_year": year, "seed": seed, "completed": bool(completed),
                 "last_epoch": last_epoch, "best_validation_ic": best_ic}  # 记录最后一个轮次。
            )
    result = pd.DataFrame(records)
    print(f"完成任务：{int(result['completed'].sum())}/{len(result)}")  # 输出当前任务的进度或诊断信息。
    display(result)
    return result  # 返回结果。

def validate_a2_args(args: argparse.Namespace) -> None:  # 校验 A2 训练任务参数之间的组合约束。
    """校验 A2 训练任务参数之间的组合约束。"""
    args.run_name = validate_run_name(args.run_name)
    if (  # 开始组合多项有效性或配置条件。
        args.tasks_per_gpu < 1
        or args.batch_size <= 1
        or args.inference_batch_size <= 1
        or args.workers_per_task < 0
    ):
        raise ValueError("A2 并发、batch 或 workers 参数非法")  # 输入数据不满足业务或数值约束时立即终止。
    if args.learning_rate <= 0 or args.max_epochs < 1 or args.patience < 1:  # 检查当前数值是否越过业务或数值阈值。
        raise ValueError("学习率、max_epochs 和 patience 必须为正")  # 输入数据不满足业务或数值约束时立即终止。
    if args.train_batch_mode not in TRAIN_BATCH_MODES:  # 检查当前值是否属于允许集合。
        raise ValueError(f"未知训练 batch 模式：{args.train_batch_mode}")  # 输入数据不满足业务或数值约束时立即终止。
    if args.loss_name not in LOSS_NAMES:  # 检查当前值是否属于允许集合。
        raise ValueError(f"未知损失函数：{args.loss_name}")  # 输入数据不满足业务或数值约束时立即终止。
    if args.validation_metric not in VALIDATION_METRICS:  # 检查当前值是否属于允许集合。
        raise ValueError(f"未知验证指标：{args.validation_metric}")  # 输入数据不满足业务或数值约束时立即终止。
    if (  # 开始组合多项有效性或配置条件。
        args.rank_temperature <= 0
        or args.turnover_lambda < 0
        or args.huber_delta <= 0
        or args.smooth_l1_beta <= 0
        or not 0 <= args.hybrid_weight <= 1
    ):
        raise ValueError("损失温度、换手权重、delta、beta 或混合权重非法")  # 输入数据不满足业务或数值约束时立即终止。
    if (  # 开始组合多项有效性或配置条件。
        args.loss_name == TEMPORAL_TURNOVER_LOSS
        and args.train_batch_mode != "trade_date"
    ):
        raise ValueError(  # 输入数据不满足业务或数值约束时立即终止。
            "sigmoid_rank_ic_turnover 必须使用 trade_date batch"
        )
    if (  # 开始组合多项有效性或配置条件。
        args.loss_name == TEMPORAL_TURNOVER_LOSS
        and bool(args.compile)
        and args.compile_mode == "reduce-overhead"
    ):
        raise ValueError(  # 输入数据不满足业务或数值约束时立即终止。
            "时序换手损失的动态日截面禁止使用 reduce-overhead；"
            "该模式会启用 CUDAGraph 私有显存池，请使用 compile-mode=default"
        )
    if (  # 开始组合多项有效性或配置条件。
        args.command in {"preflight", "preflight-worker", "a2", "worker"}
        and args.run_name in READ_ONLY_TRAINING_RUN_NAMES
    ):
        raise ValueError(  # 输入数据不满足业务或数值约束时立即终止。
            f"{args.run_name} 是只读历史实验，请为调参训练设置新的 run name"
        )
    device_ids = parse_csv_ints(args.device_ids)
    if device_ids != (0, 1, 2, 3):  # 按当前模式、标识或状态选择处理分支。
        raise ValueError("正式 A2 固定使用 GPU 0,1,2,3")  # 输入数据不满足业务或数值约束时立即终止。
    parse_csv_ints(args.years)
    parse_csv_ints(args.seeds)
    if not Path(args.data_root).resolve().is_dir():  # 检查所需文件或目录是否存在。
        raise FileNotFoundError(args.data_root)  # 目标文件不存在时立即终止。
    if args.command == "preflight-worker" and args.preflight_rank < 0:  # 按命令行子命令选择执行路径。
        raise ValueError("preflight-rank 必须为非负整数")  # 输入数据不满足业务或数值约束时立即终止。


def add_a2_arguments(parser: argparse.ArgumentParser, worker: bool = False) -> None:  # 向命令行解析器注册 A2 训练参数。
    """向命令行解析器注册 A2 训练参数。"""
    parser.add_argument("--data-root", default=str(VIT_DIR / "data"))  # 注册命令行参数及其默认值。
    parser.add_argument("--run-name", default=DEFAULT_TUNING_RUN_NAME)  # 注册命令行参数及其默认值。
    parser.add_argument("--device-ids", default="0,1,2,3")  # 注册命令行参数及其默认值。
    parser.add_argument("--tasks-per-gpu", type=int, default=2)  # 注册命令行参数及其默认值。
    parser.add_argument("--years", default=DEFAULT_YEARS)  # 注册命令行参数及其默认值。
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)  # 注册命令行参数及其默认值。
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
        default="daily_mean_spearman_ic",  # 配置默认。
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
    if worker:  # 根据当前数据状态选择处理分支。
        parser.add_argument("--worker-year", type=int, required=True)  # 注册命令行参数及其默认值。
        parser.add_argument("--worker-seed", type=int, required=True)  # 注册命令行参数及其默认值。
        parser.add_argument("--worker-physical-gpu", type=int, required=True)  # 注册命令行参数及其默认值。


def parse_args() -> argparse.Namespace:  # 解析阶段 07 的训练、预检、验收和状态命令。
    """解析阶段 07 的训练、预检、验收和状态命令。"""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "a2", "validate", "status"):  # 遍历preflighta2校验status，逐项处理名称。
        add_a2_arguments(commands.add_parser(name))
    add_a2_arguments(commands.add_parser("worker"), worker=True)
    preflight_worker = commands.add_parser("preflight-worker")
    add_a2_arguments(preflight_worker)
    preflight_worker.add_argument("--preflight-rank", type=int, required=True)  # 注册命令行参数及其默认值。
    return parser.parse_args()  # 返回解析阶段 07 的训练、预检、验收和状态命令。


def main() -> None:  # 分派 ViT 训练、预检、验收或状态查询流程。
    """分派 ViT 训练、预检、验收或状态查询流程。"""
    args = parse_args()
    validate_a2_args(args)
    if args.command == "preflight": run_preflight(args)  # 按命令行子命令选择执行路径。
    elif args.command == "preflight-worker": run_preflight_worker(args)  # 选择 preflight-worker 子命令的处理路径。
    elif args.command == "a2":  # 选择 a2 子命令的处理路径。
        run_parent(args)
        validate_outputs(args)
    elif args.command == "worker": run_worker(args)  # 选择 worker 子命令的处理路径。
    elif args.command == "validate": validate_outputs(args)  # 选择 validate 子命令的处理路径。
    elif args.command == "status": show_status(args)  # 选择 status 子命令的处理路径。


if __name__ == "__main__":  # 按当前模式、标识或状态选择处理分支。
    main()
