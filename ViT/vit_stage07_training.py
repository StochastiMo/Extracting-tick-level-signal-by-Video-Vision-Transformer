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


# 计算 `VIT_DIR / 'vit_stage07_training.py'`，并将数值结果保存到 `PIPELINE_PATH`。
PIPELINE_PATH = VIT_DIR / "vit_stage07_training.py"
# 设置 `BASELINE_A2_RUN_NAME` 的固定配置值，供后续流程统一使用。
BASELINE_A2_RUN_NAME = "baseline"
# 设置 `DEFAULT_A2_RUN_NAME` 的固定配置值，供后续流程统一使用。
DEFAULT_A2_RUN_NAME = "loss_tuning"
# 读取 `DEFAULT_A2_RUN_NAME` 的当前值，并赋给 `DEFAULT_TUNING_RUN_NAME`。
DEFAULT_TUNING_RUN_NAME = DEFAULT_A2_RUN_NAME
# 构造去重集合，并保存到 `READ_ONLY_TRAINING_RUN_NAMES`。
READ_ONLY_TRAINING_RUN_NAMES = {BASELINE_A2_RUN_NAME, "date_batch"}
# 设置 `DEFAULT_YEARS` 的固定配置值，供后续流程统一使用。
DEFAULT_YEARS = "2018,2019,2020,2021,2022,2023,2024,2025"
# 设置 `DEFAULT_SEEDS` 的固定配置值，供后续流程统一使用。
DEFAULT_SEEDS = "42,43,44"
# 组织一组有序值，并保存到 `TRAIN_BATCH_MODES`。
TRAIN_BATCH_MODES = ("fixed_samples", "trade_date")
# 读取 `ranking.TEMPORAL_TURNOVER_LOSS` 的当前值，并赋给 `TEMPORAL_TURNOVER_LOSS`。
TEMPORAL_TURNOVER_LOSS = ranking.TEMPORAL_TURNOVER_LOSS
# 读取 `ranking.LOSS_NAMES` 的当前值，并赋给 `LOSS_NAMES`。
LOSS_NAMES = ranking.LOSS_NAMES
# 读取 `ranking.STANDARD_RANK_LOSS_NAMES` 的当前值，并赋给 `STANDARD_RANK_LOSS_NAMES`。
STANDARD_RANK_LOSS_NAMES = ranking.STANDARD_RANK_LOSS_NAMES
# 组织一组有序值，并保存到 `RANK_LOSS_NAMES`。
RANK_LOSS_NAMES = (*STANDARD_RANK_LOSS_NAMES, TEMPORAL_TURNOVER_LOSS)
# 组织一组有序值，并保存到 `VALIDATION_METRICS`。
VALIDATION_METRICS = (
    "daily_mean_spearman_ic",
    "daily_mean_pearson_ic",
)

# 读取 `ranking.pearson_correlation` 的当前值，并赋给 `pearson_correlation`。
pearson_correlation = ranking.pearson_correlation
# 读取 `ranking.negative_pearson_ic` 的当前值，并赋给 `negative_pearson_ic`。
negative_pearson_ic = ranking.negative_pearson_ic
# 读取 `ranking.sigmoid_soft_rank` 的当前值，并赋给 `sigmoid_soft_rank`。
sigmoid_soft_rank = ranking.sigmoid_soft_rank
# 读取 `ranking.neural_sort_soft_rank` 的当前值，并赋给 `neural_sort_soft_rank`。
neural_sort_soft_rank = ranking.neural_sort_soft_rank
# 读取 `ranking.differentiable_rank_ic` 的当前值，并赋给 `differentiable_rank_ic`。
differentiable_rank_ic = ranking.differentiable_rank_ic
# 读取 `ranking.temporal_turnover_loss_components` 的当前值，并赋给 `temporal_turnover_loss_components`。
temporal_turnover_loss_components = (
    ranking.temporal_turnover_loss_components
)


# 使用 `dataclass(frozen=True)` 包装 `TrainingConfig`，并定义其可复用的处理逻辑。
@dataclass(frozen=True)
class TrainingConfig:
    # 声明 `train_batch_mode` 的数据类型，明确后续读写契约。
    train_batch_mode: str
    # 声明 `batch_size` 的数据类型，明确后续读写契约。
    batch_size: int
    # 声明 `inference_batch_size` 的数据类型，明确后续读写契约。
    inference_batch_size: int
    # 声明 `loss_name` 的数据类型，明确后续读写契约。
    loss_name: str
    # 声明 `rank_temperature` 的数据类型，明确后续读写契约。
    rank_temperature: float
    # 声明 `turnover_lambda` 的数据类型，明确后续读写契约。
    turnover_lambda: float
    # 声明 `huber_delta` 的数据类型，明确后续读写契约。
    huber_delta: float
    # 声明 `smooth_l1_beta` 的数据类型，明确后续读写契约。
    smooth_l1_beta: float
    # 声明 `hybrid_weight` 的数据类型，明确后续读写契约。
    hybrid_weight: float
    # 声明 `validation_metric` 的数据类型，明确后续读写契约。
    validation_metric: str
    # 声明 `learning_rate` 的数据类型，明确后续读写契约。
    learning_rate: float
    # 声明 `max_epochs` 的数据类型，明确后续读写契约。
    max_epochs: int
    # 声明 `early_stopping_patience` 的数据类型，明确后续读写契约。
    early_stopping_patience: int
    # 声明 `seeds` 的数据类型，明确后续读写契约。
    seeds: tuple[int, ...]
    # 声明 `forecast_years` 的数据类型，明确后续读写契约。
    forecast_years: tuple[int, ...]
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


# 定义 `parse_csv_ints` 函数，封装这一阶段可独立调用的处理步骤。
def parse_csv_ints(value: str) -> tuple[int, ...]:
    """解析逗号分隔整数列表并返回去重后的顺序结果。

    输入：
        value: str。
    输出：
        tuple[int, ...]。
    """
    # 调用 `tuple` 整理数据结构，并将结果保存到 `result`。
    result = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    # 检查 `not result or len(set(result)) != len(result)`；根据条件结果选择对应处理分支。
    if not result or len(set(result)) != len(result):
        # 抛出 `ValueError(f'整数列表为空或重复：{value}')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(f"整数列表为空或重复：{value}")
    # 将 `result` 作为当前函数结果返回给调用方。
    return result


# 定义 `bool_int` 函数，封装这一阶段可独立调用的处理步骤。
def bool_int(value: int) -> bool:
    """把命令行中的 0/1 参数转换为布尔值。

    输入：
        value: int。
    输出：
        bool。
    """
    # 将 `bool(int(value))` 作为当前函数结果返回给调用方。
    return bool(int(value))


# 定义 `configure_worker_environment` 函数，封装这一阶段可独立调用的处理步骤。
def configure_worker_environment(args: Any) -> None:
    """把 CLI 参数发布为单个 A2 worker 的运行参数。

    输入：
        args: Any。
    输出：
        None。
    """

    # 用本轮键值结果更新 `os.environ`，补齐后续查询所需字段。
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


# 定义 `initialize_runtime` 函数，封装这一阶段可独立调用的处理步骤。
def initialize_runtime() -> None:
    """初始化一个独立单 GPU 训练任务的全局上下文。

    输入：
        无显式输入参数。
    输出：
        None。
    """

    # 声明 `DATA_ROOT, STORE_DIR, FEATURES_PATH, TARGETS_PATH, LABEL_VALID_PATH` 使用模块级状态，避免创建同名局部变量。
    global DATA_ROOT, STORE_DIR, FEATURES_PATH, TARGETS_PATH, LABEL_VALID_PATH
    # 声明 `DATE_IDS_PATH, SAMPLE_INDEX_PATH, DATASET_MANIFEST_PATH` 使用模块级状态，避免创建同名局部变量。
    global DATE_IDS_PATH, SAMPLE_INDEX_PATH, DATASET_MANIFEST_PATH
    # 声明 `ROLLING_MANIFEST_PATH, ROLLING_SUMMARY_PATH, SPLIT_DIR` 使用模块级状态，避免创建同名局部变量。
    global ROLLING_MANIFEST_PATH, ROLLING_SUMMARY_PATH, SPLIT_DIR
    # 声明 `RUN_NAME, MODEL_ROOT, PREDICTION_DIR, QUALITY_DIR, TRAINING_SUMMARY_PATH` 使用模块级状态，避免创建同名局部变量。
    global RUN_NAME, MODEL_ROOT, PREDICTION_DIR, QUALITY_DIR, TRAINING_SUMMARY_PATH
    # 声明 `MODEL_CONFIG, TRAINING_CONFIG, NUM_WORKERS, DEVICE, PIN_MEMORY` 使用模块级状态，避免创建同名局部变量。
    global MODEL_CONFIG, TRAINING_CONFIG, NUM_WORKERS, DEVICE, PIN_MEMORY
    # 声明 `USE_BF16, USE_COMPILE, COMPILE_MODE, USE_FUSED_ADAM, USE_CUDA_PREFETCH` 使用模块级状态，避免创建同名局部变量。
    global USE_BF16, USE_COMPILE, COMPILE_MODE, USE_FUSED_ADAM, USE_CUDA_PREFETCH
    # 声明 `OVERWRITE_TRAINING, PARALLEL_BACKEND, dataset_manifest, rolling_manifest` 使用模块级状态，避免创建同名局部变量。
    global OVERWRITE_TRAINING, PARALLEL_BACKEND, dataset_manifest, rolling_manifest
    # 声明 `sample_index, targets_memmap, VISIBLE_GPU_COUNT` 使用模块级状态，避免创建同名局部变量。
    global sample_index, targets_memmap, VISIBLE_GPU_COUNT

    # 调用 `get_data_root().resolve` 规范化文件路径，并将结果保存到 `DATA_ROOT`。
    DATA_ROOT = get_data_root().resolve()
    # 计算 `DATA_ROOT / 'feature_store'`，并将数值结果保存到 `STORE_DIR`。
    STORE_DIR = DATA_ROOT / "feature_store"
    # 计算 `STORE_DIR / 'features.npy'`，并将数值结果保存到 `FEATURES_PATH`。
    FEATURES_PATH = STORE_DIR / "features.npy"
    # 计算 `STORE_DIR / 'targets.npy'`，并将数值结果保存到 `TARGETS_PATH`。
    TARGETS_PATH = STORE_DIR / "targets.npy"
    # 计算 `STORE_DIR / 'label_valid.npy'`，并将数值结果保存到 `LABEL_VALID_PATH`。
    LABEL_VALID_PATH = STORE_DIR / "label_valid.npy"
    # 计算 `STORE_DIR / 'date_ids.npy'`，并将数值结果保存到 `DATE_IDS_PATH`。
    DATE_IDS_PATH = STORE_DIR / "date_ids.npy"
    # 计算 `STORE_DIR / 'sample_index.parquet'`，并将数值结果保存到 `SAMPLE_INDEX_PATH`。
    SAMPLE_INDEX_PATH = STORE_DIR / "sample_index.parquet"
    # 计算 `STORE_DIR / 'dataset_manifest.json'`，并将数值结果保存到 `DATASET_MANIFEST_PATH`。
    DATASET_MANIFEST_PATH = STORE_DIR / "dataset_manifest.json"
    # 计算 `STORE_DIR / 'rolling_split_manifest.json'`，并将数值结果保存到 `ROLLING_MANIFEST_PATH`。
    ROLLING_MANIFEST_PATH = STORE_DIR / "rolling_split_manifest.json"
    # 计算 `STORE_DIR / 'rolling_split_summary.parquet'`，并将数值结果保存到 `ROLLING_SUMMARY_PATH`。
    ROLLING_SUMMARY_PATH = STORE_DIR / "rolling_split_summary.parquet"
    # 计算 `STORE_DIR / 'rolling_splits'`，并将数值结果保存到 `SPLIT_DIR`。
    SPLIT_DIR = STORE_DIR / "rolling_splits"
    # 调用 `os.getenv('VIT_TRAIN_RUN_NAME', DEFAULT_A2_RUN_NAME).strip` 执行模型训练并记录本轮优化指标，并将结果保存到 `RUN_NAME`。
    RUN_NAME = os.getenv("VIT_TRAIN_RUN_NAME", DEFAULT_A2_RUN_NAME).strip()
    # 调用 `set` 整理数据结构，并将结果保存到 `allowed`。
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    # 检查 `not RUN_NAME or RUN_NAME in {'.', '..'} or any((x not in allowed for x in RUN_NAME))`；根据条件结果选择对应处理分支。
    if not RUN_NAME or RUN_NAME in {".", ".."} or any(x not in allowed for x in RUN_NAME):
        # 抛出 `ValueError('训练运行名含非法字符')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("训练运行名含非法字符")
    # 计算 `VIT_DIR / 'models' / 'runs' / RUN_NAME`，并将数值结果保存到 `MODEL_ROOT`。
    MODEL_ROOT = VIT_DIR / "models" / "runs" / RUN_NAME
    # 计算 `DATA_ROOT / 'predictions' / 'runs' / RUN_NAME`，并将数值结果保存到 `PREDICTION_DIR`。
    PREDICTION_DIR = DATA_ROOT / "predictions" / "runs" / RUN_NAME
    # 计算 `DATA_ROOT / 'quality' / 'runs' / RUN_NAME`，并将数值结果保存到 `QUALITY_DIR`。
    QUALITY_DIR = DATA_ROOT / "quality" / "runs" / RUN_NAME
    # 计算 `QUALITY_DIR / 'vit_training_summary.parquet'`，并将数值结果保存到 `TRAINING_SUMMARY_PATH`。
    TRAINING_SUMMARY_PATH = QUALITY_DIR / "vit_training_summary.parquet"
    # 遍历 `(MODEL_ROOT, PREDICTION_DIR, QUALITY_DIR)`，并把当前元素绑定到 `directory` 逐项处理。
    for directory in (MODEL_ROOT, PREDICTION_DIR, QUALITY_DIR):
        # 调用 `directory.mkdir` 创建输出目录，确保后续文件可以安全写入。
        directory.mkdir(parents=True, exist_ok=True)

    # 调用 `VitModelConfig` 整理并返回当前实验的配置对象，并将结果保存到 `MODEL_CONFIG`。
    MODEL_CONFIG = VitModelConfig()
    # 调用 `TrainingConfig` 整理并返回当前实验的配置对象，并将结果保存到 `TRAINING_CONFIG`。
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
    # 检查 `TRAINING_CONFIG.train_batch_mode not in TRAIN_BATCH_MODES`；根据条件结果选择对应处理分支。
    if TRAINING_CONFIG.train_batch_mode not in TRAIN_BATCH_MODES:
        # 抛出 `ValueError(f'未知训练 batch 模式：{TRAINING_CONFIG.train_batch_mode}')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(
            f"未知训练 batch 模式：{TRAINING_CONFIG.train_batch_mode}"
        )
    # 检查 `TRAINING_CONFIG.loss_name not in LOSS_NAMES`；根据条件结果选择对应处理分支。
    if TRAINING_CONFIG.loss_name not in LOSS_NAMES:
        # 抛出 `ValueError(f'未知损失函数：{TRAINING_CONFIG.loss_name}')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(f"未知损失函数：{TRAINING_CONFIG.loss_name}")
    # 检查 `TRAINING_CONFIG.validation_metric not in VALIDATION_METRICS`；根据条件结果选择对应处理分支。
    if TRAINING_CONFIG.validation_metric not in VALIDATION_METRICS:
        # 抛出 `ValueError(f'未知验证指标：{TRAINING_CONFIG.validation_metric}')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(
            f"未知验证指标：{TRAINING_CONFIG.validation_metric}"
        )
    # 检查 `TRAINING_CONFIG.rank_temperature <= 0 or TRAINING_CONFIG.turnover_lambda < 0 or TRAINING_CONFIG…`；根据条件结果选择对应处理分支。
    if (
        TRAINING_CONFIG.rank_temperature <= 0
        or TRAINING_CONFIG.turnover_lambda < 0
        or TRAINING_CONFIG.huber_delta <= 0
        or TRAINING_CONFIG.smooth_l1_beta <= 0
        or not 0 <= TRAINING_CONFIG.hybrid_weight <= 1
    ):
        # 抛出 `ValueError('损失温度、换手权重、delta、beta 或混合权重非法')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("损失温度、换手权重、delta、beta 或混合权重非法")
    # 检查 `TRAINING_CONFIG.loss_name == TEMPORAL_TURNOVER_LOSS and TRAINING_CONFIG.train_batch_mode != 'tr…`；根据条件结果选择对应处理分支。
    if (
        TRAINING_CONFIG.loss_name == TEMPORAL_TURNOVER_LOSS
        and TRAINING_CONFIG.train_batch_mode != "trade_date"
    ):
        # 抛出 `ValueError('sigmoid_rank_ic_turnover 必须使用 trade_date batch')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(
            "sigmoid_rank_ic_turnover 必须使用 trade_date batch"
        )
    # 检查 `TRAINING_CONFIG.batch_size <= 1 or TRAINING_CONFIG.inference_batch_size <= 1 or TRAINING_CONFIG…`；根据条件结果选择对应处理分支。
    if (
        TRAINING_CONFIG.batch_size <= 1
        or TRAINING_CONFIG.inference_batch_size <= 1
        or TRAINING_CONFIG.learning_rate <= 0
    ):
        # 抛出 `ValueError('batch_size、inference_batch_size 和 learning_rate 必须为正')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("batch_size、inference_batch_size 和 learning_rate 必须为正")
    # 检查 `TRAINING_CONFIG.max_epochs < 1 or TRAINING_CONFIG.early_stopping_patience < 1`；根据条件结果选择对应处理分支。
    if (
        TRAINING_CONFIG.max_epochs < 1
        or TRAINING_CONFIG.early_stopping_patience < 1
    ):
        # 抛出 `ValueError('max_epochs 和 patience 必须为正')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("max_epochs 和 patience 必须为正")
    # 调用 `int` 统一设备或数据类型，并将结果保存到 `NUM_WORKERS`。
    NUM_WORKERS = int(os.getenv("VIT_NUM_WORKERS", "2"))
    # 检查 `NUM_WORKERS < 0`；根据条件结果选择对应处理分支。
    if NUM_WORKERS < 0:
        # 抛出 `ValueError('workers 不能为负数')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("workers 不能为负数")
    # 调用 `read_bool_environment` 读取源数据，并将结果保存到 `USE_COMPILE`。
    USE_COMPILE = read_bool_environment("VIT_USE_TORCH_COMPILE", True)
    # 调用 `os.getenv` 读取环境变量中的运行配置，并将结果保存到 `COMPILE_MODE`。
    COMPILE_MODE = os.getenv("VIT_TORCH_COMPILE_MODE", "default")
    # 检查 `TRAINING_CONFIG.loss_name == TEMPORAL_TURNOVER_LOSS and USE_COMPILE and (COMPILE_MODE == 'reduc…`；根据条件结果选择对应处理分支。
    if (
        TRAINING_CONFIG.loss_name == TEMPORAL_TURNOVER_LOSS
        and USE_COMPILE
        and COMPILE_MODE == "reduce-overhead"
    ):
        # 抛出 `ValueError('时序换手损失的动态日截面禁止使用 reduce-overhead；请改用 compile mode=default')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(
            "时序换手损失的动态日截面禁止使用 reduce-overhead；"
            "请改用 compile mode=default"
        )
    # 调用 `read_bool_environment` 读取源数据，并将结果保存到 `USE_FUSED_ADAM`。
    USE_FUSED_ADAM = read_bool_environment("VIT_USE_FUSED_ADAM", True)
    # 调用 `read_bool_environment` 读取源数据，并将结果保存到 `USE_CUDA_PREFETCH`。
    USE_CUDA_PREFETCH = read_bool_environment("VIT_USE_CUDA_PREFETCH", True)
    # 调用 `read_bool_environment` 读取源数据，并将结果保存到 `OVERWRITE_TRAINING`。
    OVERWRITE_TRAINING = read_bool_environment("VIT_TRAIN_OVERWRITE", False)
    # 按 `torch.cuda.is_available()` 在两个候选值间选择，并保存到 `VISIBLE_GPU_COUNT`。
    VISIBLE_GPU_COUNT = torch.cuda.device_count() if torch.cuda.is_available() else 0
    # 检查 `torch.cuda.is_available()`；根据条件结果选择对应处理分支。
    if torch.cuda.is_available():
        # 调用 `torch.device` 解析训练或推理所使用的计算设备，并将结果保存到 `DEVICE`。
        DEVICE = torch.device("cuda:0")
    # 当前置条件不成立时进入备用分支，保证其余合法情况仍被处理。
    else:
        # 调用 `torch.device` 解析训练或推理所使用的计算设备，并将结果保存到 `DEVICE`。
        DEVICE = torch.device("cpu")
    # 根据 `DEVICE.type == 'cuda'` 计算条件结果，并保存到 `PIN_MEMORY`。
    PIN_MEMORY = DEVICE.type == "cuda"
    # 调用 `bool` 统一设备或数据类型，并将结果保存到 `USE_BF16`。
    USE_BF16 = bool(DEVICE.type == "cuda" and torch.cuda.is_bf16_supported())
    # 设置 `PARALLEL_BACKEND` 的固定配置值，供后续流程统一使用。
    PARALLEL_BACKEND = "A2_independent_task_worker"

    # 构造当前步骤所需的列表，并保存到 `required`。
    required = [
        FEATURES_PATH, TARGETS_PATH, LABEL_VALID_PATH, DATE_IDS_PATH,
        SAMPLE_INDEX_PATH, DATASET_MANIFEST_PATH, ROLLING_MANIFEST_PATH,
        ROLLING_SUMMARY_PATH,
    ]
    # 构造当前步骤所需的列表，并保存到 `missing`。
    missing = [path for path in required if not path.exists()]
    # 检查 `missing`；根据条件结果选择对应处理分支。
    if missing:
        # 抛出 `FileNotFoundError(f'缺少训练输入：{missing}')`，阻止不满足数据契约的流程继续执行。
        raise FileNotFoundError(f"缺少训练输入：{missing}")
    # 调用 `read_json` 读取源数据，并将结果保存到 `dataset_manifest`。
    dataset_manifest = read_json(DATASET_MANIFEST_PATH)
    # 调用 `read_json` 读取源数据，并将结果保存到 `rolling_manifest`。
    rolling_manifest = read_json(ROLLING_MANIFEST_PATH)
    # 调用 `pd.read_parquet` 读取源数据，并将结果保存到 `sample_index`。
    sample_index = pd.read_parquet(SAMPLE_INDEX_PATH)
    # 调用 `sample_index['trade_date'].astype` 统一设备或数据类型，并将结果保存到 `sample_index['trade_date']`。
    sample_index["trade_date"] = sample_index["trade_date"].astype(str)
    # 调用 `sample_index['exit_trade_date'].astype` 统一设备或数据类型，并将结果保存到 `sample_index['exit_trade_date']`。
    sample_index["exit_trade_date"] = sample_index["exit_trade_date"].astype(str)
    # 调用 `np.load` 读取源数据，并将结果保存到 `targets_memmap`。
    targets_memmap = np.load(TARGETS_PATH, mmap_mode="r")

    # 调用 `print` 记录当前流程状态或诊断信息。
    print(f"Python/PyTorch：{platform.python_version()} / {torch.__version__}")
    # 调用 `print` 记录当前流程状态或诊断信息。
    print(f"运行名：{RUN_NAME}")
    # 调用 `print` 记录当前流程状态或诊断信息。
    print(f"设备：{DEVICE}；可见 GPU：{VISIBLE_GPU_COUNT}")
    # 调用 `print` 记录当前流程状态或诊断信息。
    print(
        f"训练 batch 模式：{TRAINING_CONFIG.train_batch_mode}；"
        f"固定/预测 batch：{TRAINING_CONFIG.batch_size}/"
        f"{TRAINING_CONFIG.inference_batch_size}；workers/lr："
        f"{NUM_WORKERS}/{TRAINING_CONFIG.learning_rate}"
    )
    # 根据 `TRAINING_CONFIG.loss_name == TEMPORAL_TURNOVER_LOSS` 计算条件结果，并保存到 `temporal_mode`。
    temporal_mode = TRAINING_CONFIG.loss_name == TEMPORAL_TURNOVER_LOSS
    # 调用 `print` 记录当前流程状态或诊断信息。
    print(
        f"损失函数：{TRAINING_CONFIG.loss_name}；"
        f"tau/turnover_lambda/delta/beta/hybrid_weight："
        f"{TRAINING_CONFIG.rank_temperature}/"
        f"{TRAINING_CONFIG.turnover_lambda}/{TRAINING_CONFIG.huber_delta}/"
        f"{TRAINING_CONFIG.smooth_l1_beta}/{TRAINING_CONFIG.hybrid_weight}"
    )
    # 调用 `print` 记录当前流程状态或诊断信息。
    print(
        f"验证选模指标：{TRAINING_CONFIG.validation_metric}；"
        f"reshuffle：{not temporal_mode}；梯度更新："
        f"{'每 epoch 一次' if temporal_mode else '每 batch 一次'}"
    )
    # 调用 `print` 记录当前流程状态或诊断信息。
    print(
        f"compile/mode/fused Adam/CUDA prefetch："
        f"{USE_COMPILE}/{COMPILE_MODE}/{USE_FUSED_ADAM}/{USE_CUDA_PREFETCH}"
    )


# 定义 `SampleIndexDataset` 类，集中封装该阶段的数据结构与相关行为。
class SampleIndexDataset(Dataset):
    # 定义初始化方法，接收配置并建立实例后续计算所需的内部状态。
    def __init__(self, indices: np.ndarray) -> None:
        """初始化 SampleIndexDataset 所需的数据引用、配置和运行状态。

        输入：
            indices: np.ndarray。
        输出：
            None。
        """
        # 调用 `np.asarray` 转换数据表示，并将标准化对象保存到 `self.indices`。
        self.indices = np.asarray(indices, dtype=np.int64)

    # 定义内部辅助函数 `__len__`，封装仅供本模块复用的计算步骤。
    def __len__(self) -> int:
        """返回 SampleIndexDataset 管理的样本数或批次数。

        输入：
            无显式输入参数。
        输出：
            int。
        """
        # 将 `len(self.indices)` 作为当前函数结果返回给调用方。
        return len(self.indices)

    # 定义内部辅助函数 `__getitem__`，封装仅供本模块复用的计算步骤。
    def __getitem__(self, position: int) -> int:
        """按给定索引返回 SampleIndexDataset 中的一个样本。

        输入：
            position: int。
        输出：
            int。
        """
        # 将 `int(self.indices[position])` 作为当前函数结果返回给调用方。
        return int(self.indices[position])


# 定义 `TradeDateBatchSampler` 类，集中封装该阶段的数据结构与相关行为。
class TradeDateBatchSampler(Sampler[list[int]]):
    """把完整交易日作为 batch；普通模式重排日期，时序模式保持升序。"""

    # 定义初始化方法，接收配置并建立实例后续计算所需的内部状态。
    def __init__(
        self, sample_indices: np.ndarray, date_ids: np.ndarray, seed: int,
        shuffle: bool = True, instrument_ids: np.ndarray | None = None,
    ) -> None:
        """初始化 TradeDateBatchSampler 所需的数据引用、配置和运行状态。

        输入：
            sample_indices: np.ndarray。
            date_ids: np.ndarray。
            seed: int。
            shuffle: bool。
            instrument_ids: np.ndarray | None。
        输出：
            None。
        """
        # 调用 `np.asarray` 转换数据表示，并将标准化对象保存到 `indices`。
        indices = np.asarray(sample_indices, dtype=np.int64)
        # 检查 `not len(indices)`；根据条件结果选择对应处理分支。
        if not len(indices):
            # 抛出 `ValueError('日截面 batch 不能使用空训练索引')`，阻止不满足数据契约的流程继续执行。
            raise ValueError("日截面 batch 不能使用空训练索引")
        # 调用 `np.asarray` 转换数据表示，并将标准化对象保存到 `selected_dates`。
        selected_dates = np.asarray(date_ids[indices], dtype=np.int64)
        # 检查 `np.any(selected_dates[1:] < selected_dates[:-1])`；根据条件结果选择对应处理分支。
        if np.any(selected_dates[1:] < selected_dates[:-1]):
            # 抛出 `ValueError('训练索引未按交易日排序，无法构建日截面 batch')`，阻止不满足数据契约的流程继续执行。
            raise ValueError("训练索引未按交易日排序，无法构建日截面 batch")
        # 计算 `np.flatnonzero(selected_dates[1:] != selected_dates[:-1]) + 1`，并将数值结果保存到 `boundaries`。
        boundaries = np.flatnonzero(selected_dates[1:] != selected_dates[:-1]) + 1
        # 调用 `np.r_[0, boundaries].astype` 统一设备或数据类型，并将结果保存到 `self.starts`。
        self.starts = np.r_[0, boundaries].astype(np.int64)
        # 调用 `np.r_[boundaries, len(indices)].astype` 统一设备或数据类型，并将结果保存到 `self.stops`。
        self.stops = np.r_[boundaries, len(indices)].astype(np.int64)
        # 读取 `selected_dates[self.starts]` 的当前值，并赋给 `self.date_ids`。
        self.date_ids = selected_dates[self.starts]
        # 调用 `int` 统一设备或数据类型，并将结果保存到 `self.seed`。
        self.seed = int(seed)
        # 调用 `bool` 统一设备或数据类型，并将结果保存到 `self.shuffle`。
        self.shuffle = bool(shuffle)
        # 设置 `self.epoch` 的固定配置值，供后续流程统一使用。
        self.epoch = 0
        # 设置 `self.temporal_pairs` 的固定配置值，供后续流程统一使用。
        self.temporal_pairs: list[dict[str, object]] | None = None
        # 检查 `instrument_ids is not None`；根据条件结果选择对应处理分支。
        if instrument_ids is not None:
            # 检查 `self.shuffle`；根据条件结果选择对应处理分支。
            if self.shuffle:
                # 抛出 `ValueError('时序换手损失必须关闭交易日 reshuffle')`，阻止不满足数据契约的流程继续执行。
                raise ValueError("时序换手损失必须关闭交易日 reshuffle")
            # 检查 `len(self.starts) < 2 or np.any(np.diff(self.date_ids) != 1)`；根据条件结果选择对应处理分支。
            if len(self.starts) < 2 or np.any(np.diff(self.date_ids) != 1):
                # 抛出 `ValueError('时序换手损失要求至少两个连续训练交易日')`，阻止不满足数据契约的流程继续执行。
                raise ValueError("时序换手损失要求至少两个连续训练交易日")
            # 调用 `np.asarray` 转换数据表示，并将标准化对象保存到 `all_instruments`。
            all_instruments = np.asarray(instrument_ids)
            # 构造当前步骤所需的列表，并保存到 `self.temporal_pairs`。
            self.temporal_pairs = []
            # 遍历 `range(1, len(self.starts))`，并把当前元素绑定到 `pair_number` 逐项处理。
            for pair_number in range(1, len(self.starts)):
                # 调用 `slice` 构造当前数组或张量使用的切片范围，并将结果保存到 `previous_slice`。
                previous_slice = slice(
                    int(self.starts[pair_number - 1]),
                    int(self.stops[pair_number - 1]),
                )
                # 调用 `slice` 构造当前数组或张量使用的切片范围，并将结果保存到 `current_slice`。
                current_slice = slice(
                    int(self.starts[pair_number]), int(self.stops[pair_number])
                )
                # 读取 `all_instruments[indices[previous_slice]]` 的当前值，并赋给 `previous_ids`。
                previous_ids = all_instruments[indices[previous_slice]]
                # 读取 `all_instruments[indices[current_slice]]` 的当前值，并赋给 `current_ids`。
                current_ids = all_instruments[indices[current_slice]]
                # 检查 `len(np.unique(previous_ids)) != len(previous_ids)`；根据条件结果选择对应处理分支。
                if len(np.unique(previous_ids)) != len(previous_ids):
                    # 抛出 `ValueError(f'date_id={self.date_ids[pair_number - 1]} 股票键重复')`，阻止不满足数据契约的流程继续执行。
                    raise ValueError(
                        f"date_id={self.date_ids[pair_number - 1]} 股票键重复"
                    )
                # 检查 `len(np.unique(current_ids)) != len(current_ids)`；根据条件结果选择对应处理分支。
                if len(np.unique(current_ids)) != len(current_ids):
                    # 抛出 `ValueError(f'date_id={self.date_ids[pair_number]} 股票键重复')`，阻止不满足数据契约的流程继续执行。
                    raise ValueError(
                        f"date_id={self.date_ids[pair_number]} 股票键重复"
                    )
                # 调用 `np.intersect1d` 执行 `intersect1d` 对应的明确变换，并产生后续步骤使用的结果，并将结果保存到 `(common, previous_positions, current_positions)`。
                common, previous_positions, current_positions = np.intersect1d(
                    previous_ids, current_ids, assume_unique=True,
                    return_indices=True,
                )
                # 检查 `len(common) < 2`；根据条件结果选择对应处理分支。
                if len(common) < 2:
                    # 抛出 `ValueError(f'date_id={self.date_ids[pair_number - 1]}→{self.date_ids[pair_number]} 共同股票少于2只')`，阻止不满足数据契约的流程继续执行。
                    raise ValueError(
                        f"date_id={self.date_ids[pair_number - 1]}→"
                        f"{self.date_ids[pair_number]} 共同股票少于2只"
                    )
                # 把本轮结果追加到 `self.temporal_pairs`，累积后续需要汇总的记录。
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

    # 定义内部辅助函数 `__len__`，封装仅供本模块复用的计算步骤。
    def __len__(self) -> int:
        """返回 TradeDateBatchSampler 管理的样本数或批次数。

        输入：
            无显式输入参数。
        输出：
            int。
        """
        # 将 `len(self.starts)` 作为当前函数结果返回给调用方。
        return len(self.starts)

    # 定义 `set_epoch` 函数，封装这一阶段可独立调用的处理步骤。
    def set_epoch(self, epoch: int) -> None:
        """记录当前 epoch，用于确定性改变训练日顺序。

        输入：
            epoch: int。
        输出：
            None。
        """
        # 调用 `int` 统一设备或数据类型，并将结果保存到 `self.epoch`。
        self.epoch = int(epoch)

    # 定义内部辅助函数 `__iter__`，封装仅供本模块复用的计算步骤。
    def __iter__(self) -> Any:
        """按既定顺序迭代 TradeDateBatchSampler 中的训练批次。

        输入：
            无显式输入参数。
        输出：
            Any。
        """
        # 检查 `self.shuffle`；根据条件结果选择对应处理分支。
        if self.shuffle:
            # 调用 `torch.Generator().manual_seed` 创建并固定 DataLoader 使用的随机数生成器，并将结果保存到 `generator`。
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            # 调用 `torch.randperm(len(self), generator=generator).tolist` 把张量或数组转换为 Python 列表，并将结果保存到 `order`。
            order = torch.randperm(len(self), generator=generator).tolist()
        # 当前置条件不成立时进入备用分支，保证其余合法情况仍被处理。
        else:
            # 调用 `range` 构造连续整数索引用于稳定遍历，并将结果保存到 `order`。
            order = range(len(self))
        # 遍历 `order`，并把当前元素绑定到 `batch_number` 逐项处理。
        for batch_number in order:
            # 对 `(yield range(int(self.starts[batch_number]), int(self.stops[batch_number])))` 求值，以触发当前步骤需要的计算。
            yield range(
                int(self.starts[batch_number]), int(self.stops[batch_number])
            )

    # 使用 `property` 包装 `batch_sizes`，并定义其可复用的处理逻辑。
    @property
    def batch_sizes(self) -> np.ndarray:
        """返回各批次包含的样本数量。

        输入：
            无显式输入参数。
        输出：
            np.ndarray。
        """
        # 将 `self.stops - self.starts` 作为当前函数结果返回给调用方。
        return self.stops - self.starts

    # 定义 `batch_slice` 函数，封装这一阶段可独立调用的处理步骤。
    def batch_slice(self, batch_number: int) -> slice:
        """返回指定批次在全局顺序中的切片边界。

        输入：
            batch_number: int。
        输出：
            slice。
        """
        # 将 `slice(int(self.starts[batch_number]), int(self.stops[batch_number]))` 作为当前函数结果返回给调用方。
        return slice(
            int(self.starts[batch_number]), int(self.stops[batch_number])
        )

    # 定义 `temporal_pair` 函数，封装这一阶段可独立调用的处理步骤。
    def temporal_pair(self, pair_number: int) -> dict[str, object]:
        """返回相邻交易日共同股票的对齐位置。

        输入：
            pair_number: int。
        输出：
            dict[str, object]。
        """
        # 检查 `self.temporal_pairs is None`；根据条件结果选择对应处理分支。
        if self.temporal_pairs is None:
            # 抛出 `ValueError('当前 sampler 没有时序相邻截面对齐信息')`，阻止不满足数据契约的流程继续执行。
            raise ValueError("当前 sampler 没有时序相邻截面对齐信息")
        # 将 `self.temporal_pairs[pair_number]` 作为当前函数结果返回给调用方。
        return self.temporal_pairs[pair_number]

    # 定义 `summary` 函数，封装这一阶段可独立调用的处理步骤。
    def summary(self) -> dict[str, object]:
        """汇总批采样器的日期、样本和批大小信息。

        输入：
            无显式输入参数。
        输出：
            dict[str, object]。
        """
        # 读取 `self.batch_sizes` 的当前值，并赋给 `sizes`。
        sizes = self.batch_sizes
        # 构造键值映射，并保存到 `result`，供后续按键查询。
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
        # 检查 `self.temporal_pairs is not None`；根据条件结果选择对应处理分支。
        if self.temporal_pairs is not None:
            # 调用 `np.array` 转换数据表示，并将标准化对象保存到 `common_counts`。
            common_counts = np.array(
                [pair["common_count"] for pair in self.temporal_pairs],
                dtype=np.int64,
            )
            # 调用 `np.array` 转换数据表示，并将标准化对象保存到 `coverages`。
            coverages = np.array(
                [pair["coverage"] for pair in self.temporal_pairs],
                dtype=np.float64,
            )
            # 用本轮键值结果更新 `result`，补齐后续查询所需字段。
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
        # 将 `result` 作为当前函数结果返回给调用方。
        return result


# 定义 `MemmapBatchCollator` 类，集中封装该阶段的数据结构与相关行为。
class MemmapBatchCollator:
    """每个 worker 首次取 batch 时才打开只读 memmap。"""

    # 定义初始化方法，接收配置并建立实例后续计算所需的内部状态。
    def __init__(self) -> None:
        """初始化 MemmapBatchCollator 所需的数据引用、配置和运行状态。

        输入：
            无显式输入参数。
        输出：
            None。
        """
        # 设置 `self.features` 的固定配置值，供后续流程统一使用。
        self.features = None
        # 设置 `self.targets` 的固定配置值，供后续流程统一使用。
        self.targets = None
        # 设置 `self.date_ids` 的固定配置值，供后续流程统一使用。
        self.date_ids = None

    # 定义内部辅助函数 `__call__`，封装仅供本模块复用的计算步骤。
    def __call__(self, sample_indices: list[int]) -> dict[str, torch.Tensor]:
        """按样本行号批量读取 memmap 特征和标签，并组装设备传输前的批次。

        输入：
            sample_indices: list[int]。
        输出：
            dict[str, torch.Tensor]。
        """
        # 检查 `self.features is None`；根据条件结果选择对应处理分支。
        if self.features is None:
            # 调用 `np.load` 读取源数据，并将结果保存到 `self.features`。
            self.features = np.load(FEATURES_PATH, mmap_mode="r")
            # 调用 `np.load` 读取源数据，并将结果保存到 `self.targets`。
            self.targets = np.load(TARGETS_PATH, mmap_mode="r")
            # 调用 `np.load` 读取源数据，并将结果保存到 `self.date_ids`。
            self.date_ids = np.load(DATE_IDS_PATH, mmap_mode="r")
        # 调用 `np.asarray` 转换数据表示，并将标准化对象保存到 `indices`。
        indices = np.asarray(sample_indices, dtype=np.int64)
        # 将 `{'features': torch.from_numpy(np.array(self.features[indices], dtype=np.float32, copy=True, ord…` 作为当前函数结果返回给调用方。
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


# 定义 `load_split` 函数，封装这一阶段可独立调用的处理步骤。
def load_split(forecast_year: int) -> dict[str, np.ndarray]:
    """加载指定预测年度的训练、验证和预测样本行号。

    输入：
        forecast_year: int。
    输出：
        dict[str, np.ndarray]。
    """
    # 计算 `SPLIT_DIR / f'forecast_{forecast_year}.npz'`，并将数值结果保存到 `path`。
    path = SPLIT_DIR / f"forecast_{forecast_year}.npz"
    # 检查 `not path.exists()`；根据条件结果选择对应处理分支。
    if not path.exists():
        # 抛出 `FileNotFoundError(path)`，阻止不满足数据契约的流程继续执行。
        raise FileNotFoundError(path)
    # 在 `np.load(path, allow_pickle=False)` 管理的上下文中执行本段逻辑，并保证资源按约定释放。
    with np.load(path, allow_pickle=False) as archive:
        # 将 `{name: archive[name].copy() for name in archive.files}` 作为当前函数结果返回给调用方。
        return {name: archive[name].copy() for name in archive.files}


# 定义 `make_loaders` 函数，封装这一阶段可独立调用的处理步骤。
def make_loaders(
    forecast_year: int,
    seed: int,
    batch_size: int | None = None,
    workers: int | None = None,
    inference_batch_size: int | None = None,
    train_batch_mode: str | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    """根据滚动切分创建训练、验证和预测 DataLoader。

    输入：
        forecast_year: int。
        seed: int。
        batch_size: int | None。
        workers: int | None。
        inference_batch_size: int | None。
        train_batch_mode: str | None。
    输出：
        tuple[DataLoader, DataLoader, DataLoader | None]。
    """
    # 调用 `load_split` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `split`。
    split = load_split(forecast_year)
    # 按 `batch_size is None` 在两个候选值间选择，并保存到 `effective_train_batch`。
    effective_train_batch = (
        TRAINING_CONFIG.batch_size if batch_size is None else batch_size
    )
    # 检查 `inference_batch_size is not None`；根据条件结果选择对应处理分支。
    if inference_batch_size is not None:
        # 读取 `inference_batch_size` 的当前值，并赋给 `effective_inference_batch`。
        effective_inference_batch = inference_batch_size
    # 检查 `batch_size is not None`；根据条件结果选择对应处理分支。
    elif batch_size is not None:
        # 读取 `batch_size` 的当前值，并赋给 `effective_inference_batch`。
        effective_inference_batch = batch_size
    # 当前置条件不成立时进入备用分支，保证其余合法情况仍被处理。
    else:
        # 读取 `TRAINING_CONFIG.inference_batch_size` 的当前值，并赋给 `effective_inference_batch`。
        effective_inference_batch = TRAINING_CONFIG.inference_batch_size
    # 根据 `train_batch_mode or TRAINING_CONFIG.train_batch_mode` 计算条件结果，并保存到 `effective_mode`。
    effective_mode = train_batch_mode or TRAINING_CONFIG.train_batch_mode
    # 检查 `effective_mode not in TRAIN_BATCH_MODES`；根据条件结果选择对应处理分支。
    if effective_mode not in TRAIN_BATCH_MODES:
        # 抛出 `ValueError(f'未知训练 batch 模式：{effective_mode}')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(f"未知训练 batch 模式：{effective_mode}")
    # 根据 `TRAINING_CONFIG.loss_name == TEMPORAL_TURNOVER_LOSS` 计算条件结果，并保存到 `temporal_mode`。
    temporal_mode = TRAINING_CONFIG.loss_name == TEMPORAL_TURNOVER_LOSS
    # 检查 `temporal_mode and effective_mode != 'trade_date'`；根据条件结果选择对应处理分支。
    if temporal_mode and effective_mode != "trade_date":
        # 抛出 `ValueError('sigmoid_rank_ic_turnover 必须使用 trade_date batch')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("sigmoid_rank_ic_turnover 必须使用 trade_date batch")
    # 按 `workers is None` 在两个候选值间选择，并保存到 `effective_workers`。
    effective_workers = NUM_WORKERS if workers is None else workers
    # 构造键值映射，并保存到 `common`，供后续按键查询。
    common = {"num_workers": effective_workers, "pin_memory": PIN_MEMORY}
    # 检查 `effective_workers > 0`；根据条件结果选择对应处理分支。
    if effective_workers > 0:
        # 用本轮键值结果更新 `common`，补齐后续查询所需字段。
        common.update({"persistent_workers": True, "prefetch_factor": 2})

    # 调用 `SampleIndexDataset` 构造或读取当前任务使用的数据集对象，并将结果保存到 `train_dataset`。
    train_dataset = SampleIndexDataset(split["train_indices"])
    # 调用 `torch.Generator().manual_seed` 创建并固定 DataLoader 使用的随机数生成器，并将结果保存到 `generator`。
    generator = torch.Generator().manual_seed(seed)
    # 检查 `effective_mode == 'trade_date'`；根据条件结果选择对应处理分支。
    if effective_mode == "trade_date":
        # 调用 `sample_index['date_id'].to_numpy` 把表格列转换为指定精度的 NumPy 数组，并将结果保存到 `date_ids`。
        date_ids = sample_index["date_id"].to_numpy(np.int32, copy=False)
        # 调用 `TradeDateBatchSampler` 按交易日完整截面构造 batch 采样器，并将结果保存到 `batch_sampler`。
        batch_sampler = TradeDateBatchSampler(
            split["train_indices"], date_ids, seed=seed,
            shuffle=not temporal_mode,
            instrument_ids=(
                sample_index["instrument_id"].to_numpy(copy=False)
                if temporal_mode else None
            ),
        )
        # 调用 `DataLoader` 按采样规则创建可迭代的训练或推理批次，并将结果保存到 `train_loader`。
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            generator=generator,
            collate_fn=MemmapBatchCollator(),
            **common,
        )
    # 当前置条件不成立时进入备用分支，保证其余合法情况仍被处理。
    else:
        # 调用 `DataLoader` 按采样规则创建可迭代的训练或推理批次，并将结果保存到 `train_loader`。
        train_loader = DataLoader(
            train_dataset,
            batch_size=effective_train_batch,
            shuffle=True,
            drop_last=True,
            generator=generator,
            collate_fn=MemmapBatchCollator(),
            **common,
        )
    # 调用 `DataLoader` 按采样规则创建可迭代的训练或推理批次，并将结果保存到 `validation_loader`。
    validation_loader = DataLoader(
        SampleIndexDataset(split["validation_indices"]),
        batch_size=effective_inference_batch,
        shuffle=False,
        drop_last=False,
        collate_fn=MemmapBatchCollator(),
        **common,
    )
    # 设置 `prediction_loader` 的固定配置值，供后续流程统一使用。
    prediction_loader = None
    # 检查 `len(split['prediction_indices'])`；根据条件结果选择对应处理分支。
    if len(split["prediction_indices"]):
        # 调用 `DataLoader` 按采样规则创建可迭代的训练或推理批次，并将结果保存到 `prediction_loader`。
        prediction_loader = DataLoader(
            SampleIndexDataset(split["prediction_indices"]),
            batch_size=effective_inference_batch,
            shuffle=False,
            drop_last=False,
            collate_fn=MemmapBatchCollator(),
            **common,
        )
    # 将 `(train_loader, validation_loader, prediction_loader)` 作为当前函数结果返回给调用方。
    return train_loader, validation_loader, prediction_loader


# 定义 `set_train_epoch` 函数，封装这一阶段可独立调用的处理步骤。
def set_train_epoch(loader: DataLoader, epoch: int) -> None:
    """把当前 epoch 传递给支持确定性重排的批采样器。

    输入：
        loader: DataLoader。
        epoch: int。
    输出：
        None。
    """
    # 检查 `isinstance(loader.batch_sampler, TradeDateBatchSampler)`；根据条件结果选择对应处理分支。
    if isinstance(loader.batch_sampler, TradeDateBatchSampler):
        # 调用 `loader.batch_sampler.set_epoch` 同步当前 epoch，使采样顺序能够按轮次复现。
        loader.batch_sampler.set_epoch(epoch)


# 定义 `build_model` 函数，封装这一阶段可独立调用的处理步骤。
def build_model() -> nn.Module:
    """按当前配置构建模型，并保持历史 checkpoint 的参数键兼容。

    输入：
        无显式输入参数。
    输出：
        nn.Module。
    """
    # 调用 `OrderFlowVisionTransformer(MODEL_CONFIG).to` 统一设备或数据类型，并将结果保存到 `model`。
    model = OrderFlowVisionTransformer(MODEL_CONFIG).to(DEVICE)
    # 检查 `USE_COMPILE`；根据条件结果选择对应处理分支。
    if USE_COMPILE:
        # 检查 `DEVICE.type != 'cuda'`；根据条件结果选择对应处理分支。
        if DEVICE.type != "cuda":
            # 抛出 `RuntimeError('torch.compile 正式配置要求 CUDA')`，阻止不满足数据契约的流程继续执行。
            raise RuntimeError("torch.compile 正式配置要求 CUDA")
        # 检查 `TRAINING_CONFIG.train_batch_mode == 'trade_date'`；根据条件结果选择对应处理分支。
        if TRAINING_CONFIG.train_batch_mode == "trade_date":
            # 调用 `torch.compile` 编译模型计算图以减少重复执行开销，并将结果保存到 `model`。
            model = torch.compile(model, mode=COMPILE_MODE, dynamic=True)
        # 当前置条件不成立时进入备用分支，保证其余合法情况仍被处理。
        else:
            # 调用 `torch.compile` 编译模型计算图以减少重复执行开销，并将结果保存到 `model`。
            model = torch.compile(model, mode=COMPILE_MODE)
    # 将 `model` 作为当前函数结果返回给调用方。
    return model


# 定义 `unwrap_model` 函数，封装这一阶段可独立调用的处理步骤。
def unwrap_model(model: nn.Module) -> nn.Module:
    """从编译或包装模型中取回原始 PyTorch 模型。

    输入：
        model: nn.Module。
    输出：
        nn.Module。
    """
    # 调用 `getattr` 读取对象的可选属性并应用缺省值，并将结果保存到 `original`。
    original = getattr(model, "_orig_mod", None)
    # 将 `original if original is not None else model` 作为当前函数结果返回给调用方。
    return original if original is not None else model


# 定义 `canonical_state` 函数，封装这一阶段可独立调用的处理步骤。
def canonical_state(model: nn.Module) -> Any:
    """提取不含编译包装前缀的规范化模型参数字典。

    输入：
        model: nn.Module。
    输出：
        Any。
    """
    # 调用 `unwrap_model(model).state_dict` 提取模型全部可训练参数和缓冲区，并将结果保存到 `state`。
    state = unwrap_model(model).state_dict()
    # 检查 `any((key.startswith(('module.', '_orig_mod.')) for key in state))`；根据条件结果选择对应处理分支。
    if any(key.startswith(("module.", "_orig_mod.")) for key in state):
        # 抛出 `AssertionError('canonical checkpoint 含包装器前缀')`，阻止不满足数据契约的流程继续执行。
        raise AssertionError("canonical checkpoint 含包装器前缀")
    # 将 `state` 作为当前函数结果返回给调用方。
    return state


# 定义 `load_canonical_state` 函数，封装这一阶段可独立调用的处理步骤。
def load_canonical_state(model: nn.Module, state: Any) -> None:
    """把规范化参数字典加载到原始或包装模型。

    输入：
        model: nn.Module。
        state: Any。
    输出：
        None。
    """
    # 调用 `unwrap_model(model).load_state_dict` 读取并解析后续计算所需的数据或运行状态。
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
    """解析当前训练配置后，调用无全局状态的共享损失实现。

    输入：
        predictions: torch.Tensor。
        targets: torch.Tensor。
        loss_name: str | None。
        rank_temperature: float | None。
        huber_delta: float | None。
        smooth_l1_beta: float | None。
        hybrid_weight: float | None。
        turnover_lambda: float | None。
        previous_predictions: torch.Tensor | None。
        previous_positions: np.ndarray | torch.Tensor | None。
        current_positions: np.ndarray | torch.Tensor | None。
    输出：
        torch.Tensor。
    """

    # 将 `ranking.compute_training_loss(predictions, targets, loss_name=loss_name or TRAINING_CONFIG.loss…` 作为当前函数结果返回给调用方。
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


# 定义 `autocast_context` 函数，封装这一阶段可独立调用的处理步骤。
def autocast_context() -> Any:
    """按设备和配置返回混合精度上下文。

    输入：
        无显式输入参数。
    输出：
        Any。
    """
    # 检查 `USE_BF16`；根据条件结果选择对应处理分支。
    if USE_BF16:
        # 将 `torch.autocast(device_type='cuda', dtype=torch.bfloat16)` 作为当前函数结果返回给调用方。
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    # 将 `contextlib.nullcontext()` 作为当前函数结果返回给调用方。
    return contextlib.nullcontext()


# 定义 `move_batch` 函数，封装这一阶段可独立调用的处理步骤。
def move_batch(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """把一个训练批次移动到指定设备，并保持非阻塞传输策略。

    输入：
        batch: dict[str, torch.Tensor]。
    输出：
        tuple[torch.Tensor, torch.Tensor]。
    """
    # 将 `(batch['features'].to(DEVICE, non_blocking=PIN_MEMORY), batch['targets'].to(DEVICE, non_blockin…` 作为当前函数结果返回给调用方。
    return (
        batch["features"].to(DEVICE, non_blocking=PIN_MEMORY),
        batch["targets"].to(DEVICE, non_blocking=PIN_MEMORY),
    )


# 定义 `CudaBatchPrefetcher` 类，集中封装该阶段的数据结构与相关行为。
class CudaBatchPrefetcher:
    # 定义初始化方法，接收配置并建立实例后续计算所需的内部状态。
    def __init__(self, loader: DataLoader) -> None:
        """初始化 CudaBatchPrefetcher 所需的数据引用、配置和运行状态。

        输入：
            loader: DataLoader。
        输出：
            None。
        """
        # 读取 `loader` 的当前值，并赋给 `self.loader`。
        self.loader = loader

    # 定义内部辅助函数 `__iter__`，封装仅供本模块复用的计算步骤。
    def __iter__(self) -> Any:
        """按既定顺序迭代 CudaBatchPrefetcher 中的训练批次。

        输入：
            无显式输入参数。
        输出：
            Any。
        """
        # 检查 `not USE_CUDA_PREFETCH or DEVICE.type != 'cuda'`；根据条件结果选择对应处理分支。
        if not USE_CUDA_PREFETCH or DEVICE.type != "cuda":
            # 遍历 `self.loader`，并把当前元素绑定到 `batch` 逐项处理。
            for batch in self.loader:
                # 调用 `move_batch` 把当前 batch 的张量移动到目标计算设备，并将结果保存到 `(features, targets)`。
                features, targets = move_batch(batch)
                # 对 `(yield (batch, features, targets))` 求值，以触发当前步骤需要的计算。
                yield batch, features, targets
            # 将 `空值` 作为当前函数结果返回给调用方。
            return
        # 调用 `iter` 为 DataLoader 创建顺序取批迭代器，并将结果保存到 `iterator`。
        iterator = iter(self.loader)
        # 调用 `torch.cuda.Stream` 创建独立 CUDA stream 用于异步数据预取，并将结果保存到 `stream`。
        stream = torch.cuda.Stream(device=DEVICE)
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
                # 调用 `move_batch` 把当前 batch 的张量移动到目标计算设备，并将结果保存到 `(features, targets)`。
                features, targets = move_batch(batch)
            # 组织一组有序值，并保存到 `next_item`。
            next_item = (batch, features, targets)

        # 调用 `preload` 把下一批数据异步预取到 GPU。
        preload()
        # 只要 `next_item is not None` 成立就继续循环，直至满足退出条件。
        while next_item is not None:
            # 调用 `torch.cuda.current_stream(DEVICE).wait_stream` 等待预取 stream 完成后再消费张量。
            torch.cuda.current_stream(DEVICE).wait_stream(stream)
            # 读取 `next_item` 的当前值，并赋给 `(batch, features, targets)`。
            batch, features, targets = next_item
            # 调用 `features.record_stream` 把张量登记到当前 CUDA stream，避免异步使用期间被回收。
            features.record_stream(torch.cuda.current_stream(DEVICE))
            # 调用 `targets.record_stream` 把张量登记到当前 CUDA stream，避免异步使用期间被回收。
            targets.record_stream(torch.cuda.current_stream(DEVICE))
            # 调用 `preload` 把下一批数据异步预取到 GPU。
            preload()
            # 对 `(yield (batch, features, targets))` 求值，以触发当前步骤需要的计算。
            yield batch, features, targets


# 定义 `build_optimizer` 函数，封装这一阶段可独立调用的处理步骤。
def build_optimizer(model: nn.Module) -> torch.optim.Adam:
    """创建优化器，并按配置启用 fused Adam。

    输入：
        model: nn.Module。
    输出：
        torch.optim.Adam。
    """
    # 将 `torch.optim.Adam(model.parameters(), lr=TRAINING_CONFIG.learning_rate, betas=(TRAINING_CONFIG.a…` 作为当前函数结果返回给调用方。
    return torch.optim.Adam(
        model.parameters(),
        lr=TRAINING_CONFIG.learning_rate,
        betas=(TRAINING_CONFIG.adam_beta1, TRAINING_CONFIG.adam_beta2),
        eps=TRAINING_CONFIG.adam_epsilon,
        weight_decay=TRAINING_CONFIG.weight_decay,
        fused=(USE_FUSED_ADAM and DEVICE.type == "cuda"),
    )


# 定义内部辅助函数 `_train_standard_epoch`，封装仅供本模块复用的计算步骤。
def _train_standard_epoch(model: Any, loader: Any, optimizer: Any, max_batches: int | None = None) -> Any:
    """执行该模式下的训练步骤，并返回损失与诊断信息。

    输入：
        model: Any。
        loader: Any。
        optimizer: Any。
        max_batches: int | None。
    输出：
        Any。
    """
    # 调用 `model.train` 将模型切换到训练模式，启用训练期行为。
    model.train()
    # 调用 `isinstance` 执行 `isinstance` 对应的明确变换，并产生后续步骤使用的结果，并将结果保存到 `date_batch_mode`。
    date_batch_mode = isinstance(loader.batch_sampler, TradeDateBatchSampler)
    # 调用 `torch.zeros` 按目标形状初始化数值容器 `loss_sum`。
    loss_sum = torch.zeros((), dtype=torch.float64, device=DEVICE)
    # 调用 `torch.zeros` 按目标形状初始化数值容器 `batch_pearson_sum`。
    batch_pearson_sum = torch.zeros((), dtype=torch.float64, device=DEVICE)
    # 调用 `torch.zeros` 按目标形状初始化数值容器 `weighted_pearson`。
    weighted_pearson = torch.zeros((), dtype=torch.float64, device=DEVICE)
    # 调用 `torch.zeros` 按目标形状初始化数值容器 `soft_rank_ic_sum`。
    soft_rank_ic_sum = torch.zeros((), dtype=torch.float64, device=DEVICE)
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
    # 遍历 `enumerate(CudaBatchPrefetcher(loader), start=1)`，并把当前元素绑定到 `(batch_number, (batch, features, targets))` 逐项处理。
    for batch_number, (batch, features, targets) in enumerate(
        CudaBatchPrefetcher(loader), start=1
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
        # 在 `autocast_context()` 管理的上下文中执行本段逻辑，并保证资源按约定释放。
        with autocast_context():
            # 调用 `model` 对当前输入 batch 执行模型前向传播，并将结果保存到 `predictions`。
            predictions = model(features)
        # 调用 `compute_training_loss` 按当前实验口径计算可反向传播的训练损失，并将结果保存到 `loss`。
        loss = compute_training_loss(predictions, targets)
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
        # 检查 `TRAINING_CONFIG.loss_name in RANK_LOSS_NAMES`；根据条件结果选择对应处理分支。
        if TRAINING_CONFIG.loss_name in RANK_LOSS_NAMES:
            # 用 `-loss.detach().double()` 对 `soft_rank_ic_sum` 做 `Add` 累积更新。
            soft_rank_ic_sum += -loss.detach().double()
            # 用 `1` 对 `soft_rank_batch_count` 做 `Add` 累积更新。
            soft_rank_batch_count += 1
        # 用 `current_size` 对 `sample_count` 做 `Add` 累积更新。
        sample_count += current_size
        # 用 `1` 对 `batch_count` 做 `Add` 累积更新。
        batch_count += 1
        # 按 `min_batch_size is None` 在两个候选值间选择，并保存到 `min_batch_size`。
        min_batch_size = (
            current_size if min_batch_size is None
            else min(min_batch_size, current_size)
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
    # 检查 `DEVICE.type == 'cuda'`；根据条件结果选择对应处理分支。
    if DEVICE.type == "cuda":
        # 调用 `torch.cuda.synchronize` 等待当前 GPU 队列执行完毕后再计时或验收。
        torch.cuda.synchronize()
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
        if soft_rank_batch_count else float("nan")
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
        # 兼容旧 history 字段；含义仍是 Pearson IC。
        "train_batch_ic_mean": batch_mean_pearson,
        "train_sample_weighted_ic": sample_weighted_pearson,
        "train_samples": sample_count,
        "train_batches": batch_count,
        "train_batch_size_min": int(min_batch_size),
        "train_batch_size_max": int(max_batch_size),
        "train_seconds": time.perf_counter() - started,
    }


# 定义内部辅助函数 `_train_temporal_turnover_epoch`，封装仅供本模块复用的计算步骤。
def _train_temporal_turnover_epoch(
    model: Any, loader: Any, optimizer: Any, max_batches: int | None = None,
) -> Any:
    """按自然日期累计相邻截面梯度，并在整个 epoch 结束后更新一次。

    输入：
        model: Any。
        loader: Any。
        optimizer: Any。
        max_batches: int | None。
    输出：
        Any。
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
    pair_limit = (
        available_pairs if max_batches is None
        else min(int(max_batches), available_pairs)
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
    loss_sum = torch.zeros((), dtype=torch.float64, device=DEVICE)
    # 调用 `torch.zeros` 按目标形状初始化数值容器 `pearson_sum`。
    pearson_sum = torch.zeros((), dtype=torch.float64, device=DEVICE)
    # 调用 `torch.zeros` 按目标形状初始化数值容器 `weighted_pearson`。
    weighted_pearson = torch.zeros((), dtype=torch.float64, device=DEVICE)
    # 调用 `torch.zeros` 按目标形状初始化数值容器 `prediction_rank_sum`。
    prediction_rank_sum = torch.zeros((), dtype=torch.float64, device=DEVICE)
    # 调用 `torch.zeros` 按目标形状初始化数值容器 `temporal_rank_sum`。
    temporal_rank_sum = torch.zeros((), dtype=torch.float64, device=DEVICE)
    # 调用 `torch.zeros` 按目标形状初始化数值容器 `temporal_penalty_sum`。
    temporal_penalty_sum = torch.zeros((), dtype=torch.float64, device=DEVICE)
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

    # 遍历 `CudaBatchPrefetcher(loader)`，并把当前元素绑定到 `(batch, features, targets)` 逐项处理。
    for batch, features, targets in CudaBatchPrefetcher(loader):
        # 读取 `batch['date_ids']` 的当前值，并赋给 `date_ids`。
        date_ids = batch["date_ids"]
        # 检查 `int(date_ids.min()) != int(date_ids.max())`；根据条件结果选择对应处理分支。
        if int(date_ids.min()) != int(date_ids.max()):
            # 抛出 `AssertionError('日截面训练 batch 混入多个交易日')`，阻止不满足数据契约的流程继续执行。
            raise AssertionError("日截面训练 batch 混入多个交易日")
        # 调用 `int` 统一设备或数据类型，并将结果保存到 `current_date_id`。
        current_date_id = int(date_ids[0])
        # 在 `autocast_context()` 管理的上下文中执行本段逻辑，并保证资源按约定释放。
        with autocast_context():
            # 调用 `model` 对当前输入 batch 执行模型前向传播，并将结果保存到 `current_predictions`。
            current_predictions = model(features)
        # 检查 `previous_predictions is None`；根据条件结果选择对应处理分支。
        if previous_predictions is None:
            # 首日只建立固定锚点，不计算标签项，也不保留计算图。
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
            TRAINING_CONFIG.rank_temperature,
            TRAINING_CONFIG.turnover_lambda,
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
            batch_pearson = pearson_correlation(
                current_predictions, targets
            ).double()
        # 调用 `len` 汇总当前数据，并将统计量保存到 `current_size`。
        current_size = len(targets)
        # 用 `loss.detach().double()` 对 `loss_sum` 做 `Add` 累积更新。
        loss_sum += loss.detach().double()
        # 用 `batch_pearson` 对 `pearson_sum` 做 `Add` 累积更新。
        pearson_sum += batch_pearson
        # 用 `batch_pearson * current_size` 对 `weighted_pearson` 做 `Add` 累积更新。
        weighted_pearson += batch_pearson * current_size
        # 用 `components['prediction_soft_rank_ic'].detach().double()` 对 `prediction_rank_sum` 做 `Add` 累积更新。
        prediction_rank_sum += components[
            "prediction_soft_rank_ic"
        ].detach().double()
        # 用 `components['temporal_soft_rank_ic'].detach().double()` 对 `temporal_rank_sum` 做 `Add` 累积更新。
        temporal_rank_sum += components["temporal_soft_rank_ic"].detach().double()
        # 用 `components['temporal_penalty'].detach().double()` 对 `temporal_penalty_sum` 做 `Add` 累积更新。
        temporal_penalty_sum += components["temporal_penalty"].detach().double()
        # 用 `current_size` 对 `sample_count` 做 `Add` 累积更新。
        sample_count += current_size
        # 用 `1` 对 `pair_count` 做 `Add` 累积更新。
        pair_count += 1
        # 按 `min_batch_size is None` 在两个候选值间选择，并保存到 `min_batch_size`。
        min_batch_size = (
            current_size if min_batch_size is None
            else min(min_batch_size, current_size)
        )
        # 调用 `max` 汇总当前数据，并将统计量保存到 `max_batch_size`。
        max_batch_size = max(max_batch_size, current_size)
        # 参数在 epoch 结束前不更新，因此当前预测可直接作为下一日常数锚点。
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
        raise AssertionError(
            f"时序日期对覆盖不完整：{pair_count}/{pair_limit}"
        )
    # 调用 `optimizer.step` 推进一次优化器或调度器状态。
    optimizer.step()
    # 检查 `DEVICE.type == 'cuda'`；根据条件结果选择对应处理分支。
    if DEVICE.type == "cuda":
        # 调用 `torch.cuda.synchronize` 等待当前 GPU 队列执行完毕后再计时或验收。
        torch.cuda.synchronize()
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
        # 兼容旧 history 字段；含义仍是 Pearson IC。
        "train_batch_ic_mean": mean_pearson,
        "train_sample_weighted_ic": weighted_mean_pearson,
        "train_samples": sample_count,
        "train_batches": pair_count,
        "train_batch_size_min": int(min_batch_size),
        "train_batch_size_max": int(max_batch_size),
        "train_seconds": time.perf_counter() - started,
    }


# 定义 `train_one_epoch` 函数，封装这一阶段可独立调用的处理步骤。
def train_one_epoch(model: Any, loader: Any, optimizer: Any, max_batches: int | None = None) -> Any:
    """训练一个完整 epoch，并返回损失与批次诊断。

    输入：
        model: Any。
        loader: Any。
        optimizer: Any。
        max_batches: int | None。
    输出：
        Any。
    """
    # 检查 `TRAINING_CONFIG.loss_name == TEMPORAL_TURNOVER_LOSS`；根据条件结果选择对应处理分支。
    if TRAINING_CONFIG.loss_name == TEMPORAL_TURNOVER_LOSS:
        # 将 `_train_temporal_turnover_epoch(model, loader, optimizer, max_batches=max_batches)` 作为当前函数结果返回给调用方。
        return _train_temporal_turnover_epoch(
            model, loader, optimizer, max_batches=max_batches
        )
    # 将 `_train_standard_epoch(model, loader, optimizer, max_batches=max_batches)` 作为当前函数结果返回给调用方。
    return _train_standard_epoch(
        model, loader, optimizer, max_batches=max_batches
    )


# 定义 `average_descending_ranks` 函数，封装这一阶段可独立调用的处理步骤。
def average_descending_ranks(values: np.ndarray) -> np.ndarray:
    """精确降序平均秩；相同值共享其占用名次的均值。

    输入：
        values: np.ndarray。
    输出：
        np.ndarray。
    """

    # 调用 `np.asarray(values, dtype=np.float64).reshape` 调整张量维度，并将新视图保存到 `array`。
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    # 检查 `len(array) < 2 or not np.isfinite(array).all()`；根据条件结果选择对应处理分支。
    if len(array) < 2 or not np.isfinite(array).all():
        # 抛出 `ValueError('排名输入必须至少有两个有限值')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("排名输入必须至少有两个有限值")
    # 调用 `np.argsort` 根据分数计算截面排序或排序索引，并将结果保存到 `order`。
    order = np.argsort(-array, kind="stable")
    # 读取 `array[order]` 的当前值，并赋给 `ordered`。
    ordered = array[order]
    # 读取 `np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]` 的当前值，并赋给 `starts`。
    starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
    # 读取 `np.r_[starts[1:], len(array)]` 的当前值，并赋给 `stops`。
    stops = np.r_[starts[1:], len(array)]
    # 计算 `(starts + 1 + stops) / 2`，并将数值结果保存到 `group_ranks`。
    group_ranks = (starts + 1 + stops) / 2
    # 调用 `np.empty` 按目标形状初始化数值容器 `ranks`。
    ranks = np.empty(len(array), dtype=np.float64)
    # 调用 `np.repeat` 按各交易日样本数展开分组秩值，并将结果保存到 `ranks[order]`。
    ranks[order] = np.repeat(group_ranks, stops - starts)
    # 将 `ranks` 作为当前函数结果返回给调用方。
    return ranks


# 定义 `numpy_pearson` 函数，封装这一阶段可独立调用的处理步骤。
def numpy_pearson(x: np.ndarray, y: np.ndarray) -> float:
    """用 NumPy 计算两个有效向量的 Pearson 相关系数。

    输入：
        x: np.ndarray。
        y: np.ndarray。
    输出：
        float。
    """
    # 调用 `np.asarray(x, dtype=np.float64).reshape` 调整张量维度，并将新视图保存到 `left`。
    left = np.asarray(x, dtype=np.float64).reshape(-1)
    # 调用 `np.asarray(y, dtype=np.float64).reshape` 调整张量维度，并将新视图保存到 `right`。
    right = np.asarray(y, dtype=np.float64).reshape(-1)
    # 计算 `left - left.mean()`，并将数值结果保存到 `centered_left`。
    centered_left = left - left.mean()
    # 计算 `right - right.mean()`，并将数值结果保存到 `centered_right`。
    centered_right = right - right.mean()
    # 调用 `np.sqrt` 开平方得到范数或标准差，并将结果保存到 `denominator`。
    denominator = np.sqrt(
        np.square(centered_left).sum() * np.square(centered_right).sum()
    )
    # 检查 `not np.isfinite(denominator) or denominator <= 1e-12`；根据条件结果选择对应处理分支。
    if not np.isfinite(denominator) or denominator <= 1e-12:
        # 抛出 `FloatingPointError('相关系数分母为零或非有限数')`，阻止不满足数据契约的流程继续执行。
        raise FloatingPointError("相关系数分母为零或非有限数")
    # 将 `float(np.dot(centered_left, centered_right) / denominator)` 作为当前函数结果返回给调用方。
    return float(np.dot(centered_left, centered_right) / denominator)


# 定义 `summarize_validation_arrays` 函数，封装这一阶段可独立调用的处理步骤。
def summarize_validation_arrays(
    predictions: np.ndarray, targets: np.ndarray, date_ids: np.ndarray,
    validation_metric: str, instrument_ids: np.ndarray | None = None,
) -> dict[str, float | int]:
    """先恢复完整日期截面，再计算硬 RankIC、日度 Pearson 和 pooled IC。

    输入：
        predictions: np.ndarray。
        targets: np.ndarray。
        date_ids: np.ndarray。
        validation_metric: str。
        instrument_ids: np.ndarray | None。
    输出：
        dict[str, float | int]。
    """

    # 调用 `np.asarray(predictions, dtype=np.float64).reshape` 调整张量维度，并将新视图保存到 `x`。
    x = np.asarray(predictions, dtype=np.float64).reshape(-1)
    # 调用 `np.asarray(targets, dtype=np.float64).reshape` 调整张量维度，并将新视图保存到 `y`。
    y = np.asarray(targets, dtype=np.float64).reshape(-1)
    # 调用 `np.asarray(date_ids, dtype=np.int64).reshape` 调整张量维度，并将新视图保存到 `dates`。
    dates = np.asarray(date_ids, dtype=np.int64).reshape(-1)
    # 按 `instrument_ids is None` 在两个候选值间选择，并保存到 `instruments`。
    instruments = (
        None if instrument_ids is None
        else np.asarray(instrument_ids).reshape(-1)
    )
    # 检查 `not len(x) == len(y) == len(dates) or len(x) < 2`；根据条件结果选择对应处理分支。
    if not (len(x) == len(y) == len(dates)) or len(x) < 2:
        # 抛出 `ValueError('验证预测、标签和日期必须等长且至少包含两个样本')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("验证预测、标签和日期必须等长且至少包含两个样本")
    # 检查 `instruments is not None and len(instruments) != len(x)`；根据条件结果选择对应处理分支。
    if instruments is not None and len(instruments) != len(x):
        # 抛出 `ValueError('验证股票键与预测长度不一致')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("验证股票键与预测长度不一致")
    # 检查 `not np.isfinite(x).all() or not np.isfinite(y).all()`；根据条件结果选择对应处理分支。
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        # 抛出 `FloatingPointError('验证预测或标签包含 NaN/Inf')`，阻止不满足数据契约的流程继续执行。
        raise FloatingPointError("验证预测或标签包含 NaN/Inf")
    # 检查 `validation_metric not in VALIDATION_METRICS`；根据条件结果选择对应处理分支。
    if validation_metric not in VALIDATION_METRICS:
        # 抛出 `ValueError(f'未知验证指标：{validation_metric}')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(f"未知验证指标：{validation_metric}")

    # 调用 `np.argsort` 根据分数计算截面排序或排序索引，并将结果保存到 `order`。
    order = np.argsort(dates, kind="stable")
    # 组织一组有序值，并保存到 `(x, y, dates)`。
    x, y, dates = x[order], y[order], dates[order]
    # 检查 `instruments is not None`；根据条件结果选择对应处理分支。
    if instruments is not None:
        # 读取 `instruments[order]` 的当前值，并赋给 `instruments`。
        instruments = instruments[order]
    # 计算 `np.flatnonzero(dates[1:] != dates[:-1]) + 1`，并将数值结果保存到 `boundaries`。
    boundaries = np.flatnonzero(dates[1:] != dates[:-1]) + 1
    # 读取 `np.r_[0, boundaries]` 的当前值，并赋给 `starts`。
    starts = np.r_[0, boundaries]
    # 读取 `np.r_[boundaries, len(dates)]` 的当前值，并赋给 `stops`。
    stops = np.r_[boundaries, len(dates)]
    # 组织一组有序值，并保存到 `(daily_rank_ic, daily_pearson_ic)`。
    daily_rank_ic, daily_pearson_ic = [], []
    # 遍历 `zip(starts, stops)`，并把当前元素绑定到 `(start, stop)` 逐项处理。
    for start, stop in zip(starts, stops):
        # 检查 `stop - start < 2`；根据条件结果选择对应处理分支。
        if stop - start < 2:
            # 抛出 `FloatingPointError('验证集存在不足两个样本的交易日')`，阻止不满足数据契约的流程继续执行。
            raise FloatingPointError("验证集存在不足两个样本的交易日")
        # 组织一组有序值，并保存到 `(daily_x, daily_y)`。
        daily_x, daily_y = x[start:stop], y[start:stop]
        # 把本轮结果追加到 `daily_pearson_ic`，累积后续需要汇总的记录。
        daily_pearson_ic.append(numpy_pearson(daily_x, daily_y))
        # 把本轮结果追加到 `daily_rank_ic`，累积后续需要汇总的记录。
        daily_rank_ic.append(
            numpy_pearson(
                average_descending_ranks(daily_x),
                average_descending_ranks(daily_y),
            )
        )
    # 构造当前步骤所需的列表，并保存到 `temporal_rank_ic`。
    temporal_rank_ic: list[float] = []
    # 检查 `instruments is not None`；根据条件结果选择对应处理分支。
    if instruments is not None:
        # 读取 `dates[starts]` 的当前值，并赋给 `unique_dates`。
        unique_dates = dates[starts]
        # 检查 `len(unique_dates) > 1 and np.any(np.diff(unique_dates) != 1)`；根据条件结果选择对应处理分支。
        if len(unique_dates) > 1 and np.any(np.diff(unique_dates) != 1):
            # 抛出 `ValueError('验证集日期不连续，无法计算相邻预测 RankIC')`，阻止不满足数据契约的流程继续执行。
            raise ValueError("验证集日期不连续，无法计算相邻预测 RankIC")
        # 遍历 `range(1, len(starts))`，并把当前元素绑定到 `pair_number` 逐项处理。
        for pair_number in range(1, len(starts)):
            # 调用 `slice` 构造当前数组或张量使用的切片范围，并将结果保存到 `previous_slice`。
            previous_slice = slice(starts[pair_number - 1], stops[pair_number - 1])
            # 调用 `slice` 构造当前数组或张量使用的切片范围，并将结果保存到 `current_slice`。
            current_slice = slice(starts[pair_number], stops[pair_number])
            # 读取 `instruments[previous_slice]` 的当前值，并赋给 `previous_ids`。
            previous_ids = instruments[previous_slice]
            # 读取 `instruments[current_slice]` 的当前值，并赋给 `current_ids`。
            current_ids = instruments[current_slice]
            # 检查 `len(np.unique(previous_ids)) != len(previous_ids) or len(np.unique(current_ids)) != len(current…`；根据条件结果选择对应处理分支。
            if (
                len(np.unique(previous_ids)) != len(previous_ids)
                or len(np.unique(current_ids)) != len(current_ids)
            ):
                # 抛出 `ValueError('验证集同一日期股票键重复')`，阻止不满足数据契约的流程继续执行。
                raise ValueError("验证集同一日期股票键重复")
            # 调用 `np.intersect1d` 执行 `intersect1d` 对应的明确变换，并产生后续步骤使用的结果，并将结果保存到 `(common, previous_positions, current_positions)`。
            common, previous_positions, current_positions = np.intersect1d(
                previous_ids, current_ids, assume_unique=True,
                return_indices=True,
            )
            # 检查 `len(common) < 2`；根据条件结果选择对应处理分支。
            if len(common) < 2:
                # 抛出 `ValueError('验证集相邻日期共同股票少于2只')`，阻止不满足数据契约的流程继续执行。
                raise ValueError("验证集相邻日期共同股票少于2只")
            # 把本轮结果追加到 `temporal_rank_ic`，累积后续需要汇总的记录。
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
    # 调用 `float` 统一设备或数据类型，并将结果保存到 `mean_rank_ic`。
    mean_rank_ic = float(np.mean(daily_rank_ic))
    # 调用 `float` 统一设备或数据类型，并将结果保存到 `mean_pearson_ic`。
    mean_pearson_ic = float(np.mean(daily_pearson_ic))
    # 按 `temporal_rank_ic` 在两个候选值间选择，并保存到 `mean_temporal_rank_ic`。
    mean_temporal_rank_ic = (
        float(np.mean(temporal_rank_ic)) if temporal_rank_ic else float("nan")
    )
    # 调用 `numpy_pearson` 计算预测与标签或风险暴露之间的相关性，并将结果保存到 `pooled_ic`。
    pooled_ic = numpy_pearson(x, y)
    # 按 `validation_metric == 'daily_mean_spearman_ic'` 在两个候选值间选择，并保存到 `selected_ic`。
    selected_ic = (
        mean_rank_ic
        if validation_metric == "daily_mean_spearman_ic"
        else mean_pearson_ic
    )
    # 调用 `np.array` 转换数据表示，并将标准化对象保存到 `values`。
    values = np.array(
        [selected_ic, mean_rank_ic, mean_pearson_ic, pooled_ic], dtype=np.float64
    )
    # 检查 `not np.isfinite(values).all()`；根据条件结果选择对应处理分支。
    if not np.isfinite(values).all():
        # 抛出 `FloatingPointError('验证 IC 不是有限数')`，阻止不满足数据契约的流程继续执行。
        raise FloatingPointError("验证 IC 不是有限数")
    # 将 `{'validation_ic': selected_ic, 'validation_loss': -selected_ic, 'validation_daily_rank_ic': mea…` 作为当前函数结果返回给调用方。
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
    """在验证集上评估模型，并汇总逐日 IC 与损失诊断。

    输入：
        model: Any。
        loader: Any。
        max_batches: int | None。
    输出：
        Any。
    """
    # 调用 `model.eval` 将模型切换到评估模式，关闭训练期随机行为。
    model.eval()
    # 组织一组有序值，并保存到 `(prediction_parts, target_parts, date_parts, sample_index_parts)`。
    prediction_parts, target_parts, date_parts, sample_index_parts = [], [], [], []
    # 调用 `time.perf_counter` 读取高精度单调时钟以计算耗时，并将结果保存到 `started`。
    started = time.perf_counter()
    # 遍历 `enumerate(CudaBatchPrefetcher(loader), start=1)`，并把当前元素绑定到 `(batch_number, (batch, features, targets))` 逐项处理。
    for batch_number, (batch, features, targets) in enumerate(
        CudaBatchPrefetcher(loader), start=1
    ):
        # 在 `autocast_context()` 管理的上下文中执行本段逻辑，并保证资源按约定释放。
        with autocast_context():
            # 调用 `model` 对当前输入 batch 执行模型前向传播，并将结果保存到 `predictions`。
            predictions = model(features)
        # 把本轮结果追加到 `prediction_parts`，累积后续需要汇总的记录。
        prediction_parts.append(
            predictions.float().cpu().numpy().astype(np.float64, copy=False)
        )
        # 把本轮结果追加到 `target_parts`，累积后续需要汇总的记录。
        target_parts.append(
            targets.float().cpu().numpy().astype(np.float64, copy=False)
        )
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
    # 检查 `DEVICE.type == 'cuda'`；根据条件结果选择对应处理分支。
    if DEVICE.type == "cuda":
        # 调用 `torch.cuda.synchronize` 等待当前 GPU 队列执行完毕后再计时或验收。
        torch.cuda.synchronize()
    # 调用 `summarize_validation_arrays` 按约定截面或分组口径汇总统计结果，并将结果保存到 `result`。
    result = summarize_validation_arrays(
        np.concatenate(prediction_parts),
        np.concatenate(target_parts),
        np.concatenate(date_parts),
        TRAINING_CONFIG.validation_metric,
        instrument_ids=sample_index["instrument_id"].to_numpy(copy=False)[
            np.concatenate(sample_index_parts)
        ],
    )
    # 计算 `time.perf_counter() - started`，并将数值结果保存到 `result['validation_seconds']`。
    result["validation_seconds"] = time.perf_counter() - started
    # 将 `result` 作为当前函数结果返回给调用方。
    return result


# 使用 `torch.no_grad()` 包装 `predict_model`，并定义其可复用的处理逻辑。
@torch.no_grad()
def predict_model(model: Any, loader: Any) -> pd.DataFrame:
    """在预测集上生成与样本行号严格对齐的模型分数。

    输入：
        model: Any。
        loader: Any。
    输出：
        pd.DataFrame。
    """
    # 调用 `model.eval` 将模型切换到评估模式，关闭训练期随机行为。
    model.eval()
    # 组织一组有序值，并保存到 `(indices, predictions)`。
    indices, predictions = [], []
    # 遍历 `CudaBatchPrefetcher(loader)`，并把当前元素绑定到 `(batch, features, _)` 逐项处理。
    for batch, features, _ in CudaBatchPrefetcher(loader):
        # 在 `autocast_context()` 管理的上下文中执行本段逻辑，并保证资源按约定释放。
        with autocast_context():
            # 调用 `model` 对当前输入 batch 执行模型前向传播，并将结果保存到 `output`。
            output = model(features)
        # 把本轮结果追加到 `indices`，累积后续需要汇总的记录。
        indices.append(batch["sample_indices"].numpy())
        # 把本轮结果追加到 `predictions`，累积后续需要汇总的记录。
        predictions.append(output.float().cpu().numpy().astype(np.float32, copy=False))
    # 将 `pd.DataFrame({'sample_index': np.concatenate(indices), 'prediction': np.concatenate(predictions…` 作为当前函数结果返回给调用方。
    return pd.DataFrame(
        {"sample_index": np.concatenate(indices), "prediction": np.concatenate(predictions)}
    )


# 定义 `execution_payload` 函数，封装这一阶段可独立调用的处理步骤。
def execution_payload() -> dict[str, object]:
    """整理影响单次训练执行方式的配置载荷。

    输入：
        无显式输入参数。
    输出：
        dict[str, object]。
    """
    # 根据 `TRAINING_CONFIG.train_batch_mode == 'trade_date'` 计算条件结果，并保存到 `date_batches`。
    date_batches = TRAINING_CONFIG.train_batch_mode == "trade_date"
    # 根据 `TRAINING_CONFIG.loss_name == TEMPORAL_TURNOVER_LOSS` 计算条件结果，并保存到 `temporal_mode`。
    temporal_mode = TRAINING_CONFIG.loss_name == TEMPORAL_TURNOVER_LOSS
    # 将 `{'parallel_backend': PARALLEL_BACKEND, 'gpu_count_per_task': 1, 'train_batch_mode': TRAINING_CO…` 作为当前函数结果返回给调用方。
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


# 定义 `run_payload` 函数，封装这一阶段可独立调用的处理步骤。
def run_payload(
    year: int, seed: int, batch_summary: dict[str, object]
) -> dict[str, object]:
    """整理影响整个 run 身份和可复用性的签名载荷。

    输入：
        year: int。
        seed: int。
        batch_summary: dict[str, object]。
    输出：
        dict[str, object]。
    """
    # 将 `{'schema_version': 6, 'forecast_year': year, 'seed': seed, 'dataset_signature': dataset_manifes…` 作为当前函数结果返回给调用方。
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


# 定义 `atomic_checkpoint` 函数，封装这一阶段可独立调用的处理步骤。
def atomic_checkpoint(payload: dict[str, object], path: Path) -> None:
    """先写临时 checkpoint，再原子替换正式模型文件。

    输入：
        payload: dict[str, object]。
        path: Path。
    输出：
        None。
    """
    # 调用 `path.with_name` 规范化文件路径，并将结果保存到 `temporary`。
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    # 调用 `torch.save` 将当前结果持久化到目标文件。
    torch.save(payload, temporary)
    # 调用 `temporary.replace` 用完整临时文件原子替换正式产物。
    temporary.replace(path)


# 定义 `checkpoint_payload` 函数，封装这一阶段可独立调用的处理步骤。
def checkpoint_payload(
    model: Any, optimizer: Any, generator_state: Any, epoch: Any, best_epoch: Any, best_ic: Any,
    best_validation_metrics: Any, patience_count: Any, signature: Any, epoch_metrics: Any,
) -> dict[str, object]:
    """构建 checkpoint 中的模型、优化器、进度和签名信息。

    输入：
        model: Any。
        optimizer: Any。
        generator_state: Any。
        epoch: Any。
        best_epoch: Any。
        best_ic: Any。
        best_validation_metrics: Any。
        patience_count: Any。
        signature: Any。
        epoch_metrics: Any。
    输出：
        dict[str, object]。
    """
    # 将 `{'model_state': canonical_state(model), 'optimizer_state': optimizer.state_dict(), 'python_rand…` 作为当前函数结果返回给调用方。
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


# 定义 `load_best_model` 函数，封装这一阶段可独立调用的处理步骤。
def load_best_model(path: Path) -> Any:
    """加载最佳验证 checkpoint，并恢复可用于预测的模型。

    输入：
        path: Path。
    输出：
        Any。
    """
    # 调用 `torch.load` 读取源数据，并将结果保存到 `checkpoint`。
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    # 调用 `build_model` 根据模型配置创建网络并部署到目标设备，并将结果保存到 `model`。
    model = build_model()
    # 调用 `load_canonical_state` 把标准化 checkpoint 权重加载到当前模型。
    load_canonical_state(model, checkpoint["model_state"])
    # 调用 `model.eval` 将模型切换到评估模式，关闭训练期随机行为。
    model.eval()
    # 将 `model` 作为当前函数结果返回给调用方。
    return model


# 定义 `train_single_run` 函数，封装这一阶段可独立调用的处理步骤。
def train_single_run(year: int, seed: int) -> Any:
    """完成一个预测年度和一个随机种子的训练、早停与预测。

    输入：
        year: int。
        seed: int。
    输出：
        Any。
    """
    # 调用 `set_random_seed` 同时固定 Python、NumPy 和 PyTorch 随机状态。
    set_random_seed(seed)
    # 调用 `make_loaders` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `(train_loader, validation_loader, _)`。
    train_loader, validation_loader, _ = make_loaders(year, seed)
    # 调用 `loader_batch_summary` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `batch_summary`。
    batch_summary = loader_batch_summary(train_loader)
    # 调用 `run_payload` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `payload`。
    payload = run_payload(year, seed, batch_summary)
    # 调用 `stable_hash` 对关键配置生成稳定签名以支持复现和一致性检查，并将结果保存到 `signature`。
    signature = stable_hash(payload)

    # 计算 `MODEL_ROOT / f'forecast_{year}' / f'seed_{seed}'`，并将数值结果保存到 `run_dir`。
    run_dir = MODEL_ROOT / f"forecast_{year}" / f"seed_{seed}"
    # 调用 `run_dir.mkdir` 创建输出目录，确保后续文件可以安全写入。
    run_dir.mkdir(parents=True, exist_ok=True)
    # 组织一组有序值，并保存到 `(best_path, last_path)`。
    best_path, last_path = run_dir / "best.pt", run_dir / "last.pt"
    # 组织一组有序值，并保存到 `(history_path, completed_path)`。
    history_path, completed_path = run_dir / "history.parquet", run_dir / "completed.json"
    # 调用 `atomic_json` 先写临时文件再原子替换正式产物，避免留下半成品。
    atomic_json(
        {"year_signature": stable_hash({k: v for k, v in payload.items() if k != "seed"}),
         "payload": {k: v for k, v in payload.items() if k != "seed"}},
        MODEL_ROOT / f"forecast_{year}" / "model_config.json",
    )
    # 检查 `completed_path.exists() and best_path.exists() and (not OVERWRITE_TRAINING)`；根据条件结果选择对应处理分支。
    if completed_path.exists() and best_path.exists() and not OVERWRITE_TRAINING:
        # 调用 `read_json` 读取源数据，并将结果保存到 `completed`。
        completed = read_json(completed_path)
        # 检查 `completed.get('run_signature') == signature`；根据条件结果选择对应处理分支。
        if completed.get("run_signature") == signature:
            # 调用 `print` 记录当前流程状态或诊断信息。
            print(f"[{year}/seed={seed}] 已完成，直接复用。")
            # 将 `(load_best_model(best_path), completed['summary'])` 作为当前函数结果返回给调用方。
            return load_best_model(best_path), completed["summary"]

    # 调用 `build_model` 根据模型配置创建网络并部署到目标设备，并将结果保存到 `model`。
    model = build_model()
    # 调用 `build_optimizer` 根据训练配置为可学习参数创建优化器，并将结果保存到 `optimizer`。
    optimizer = build_optimizer(model)
    # 组织一组有序值，并保存到 `(start_epoch, best_epoch)`。
    start_epoch, best_epoch = 1, 0
    # 组织一组有序值，并保存到 `(best_ic, patience_count)`。
    best_ic, patience_count = -np.inf, 0
    # 构造键值映射，并保存到 `best_validation_metrics`，供后续按键查询。
    best_validation_metrics: dict[str, float] = {}
    # 构造当前步骤所需的列表，并保存到 `history`。
    history: list[dict[str, object]] = []
    # 检查 `last_path.exists() and (not OVERWRITE_TRAINING)`；根据条件结果选择对应处理分支。
    if last_path.exists() and not OVERWRITE_TRAINING:
        # 调用 `torch.load` 读取源数据，并将结果保存到 `checkpoint`。
        checkpoint = torch.load(last_path, map_location=DEVICE, weights_only=False)
        # 检查 `checkpoint.get('run_signature') != signature`；根据条件结果选择对应处理分支。
        if checkpoint.get("run_signature") != signature:
            # 抛出 `ValueError(f'{last_path} 配置签名不一致，请显式覆盖重训')`，阻止不满足数据契约的流程继续执行。
            raise ValueError(f"{last_path} 配置签名不一致，请显式覆盖重训")
        # 调用 `load_canonical_state` 把标准化 checkpoint 权重加载到当前模型。
        load_canonical_state(model, checkpoint["model_state"])
        # 调用 `optimizer.load_state_dict` 恢复优化器动量等续训状态。
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        # 调用 `random.setstate` 恢复 Python 随机数生成器状态。
        random.setstate(checkpoint["python_random_state"])
        # 调用 `np.random.set_state` 恢复 NumPy 随机数生成器状态。
        np.random.set_state(checkpoint["numpy_random_state"])
        # 调用 `torch.set_rng_state` 恢复 PyTorch CPU 随机数生成器状态。
        torch.set_rng_state(checkpoint["torch_rng_state"])
        # 检查 `torch.cuda.is_available() and checkpoint['cuda_rng_state_all'] is not None`；根据条件结果选择对应处理分支。
        if torch.cuda.is_available() and checkpoint["cuda_rng_state_all"] is not None:
            # 调用 `torch.cuda.set_rng_state_all` 恢复所有 GPU 的随机数生成器状态。
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
        # 调用 `train_loader.generator.set_state` 读取并解析后续计算所需的数据或运行状态。
        train_loader.generator.set_state(checkpoint["train_generator_state"])
        # 计算 `int(checkpoint['epoch']) + 1`，并将数值结果保存到 `start_epoch`。
        start_epoch = int(checkpoint["epoch"]) + 1
        # 调用 `checkpoint.get` 按键读取映射中的可选字段，并在缺失时使用约定默认值，并将结果保存到 `expected_sampler_epoch`。
        expected_sampler_epoch = checkpoint.get("next_trade_date_sampler_epoch")
        # 检查 `TRAINING_CONFIG.train_batch_mode == 'trade_date' and expected_sampler_epoch is not None and (in…`；根据条件结果选择对应处理分支。
        if (
            TRAINING_CONFIG.train_batch_mode == "trade_date"
            and expected_sampler_epoch is not None
            and int(expected_sampler_epoch) != start_epoch
        ):
            # 抛出 `AssertionError('checkpoint 中的下一日截面 sampler epoch 不一致')`，阻止不满足数据契约的流程继续执行。
            raise AssertionError("checkpoint 中的下一日截面 sampler epoch 不一致")
        # 调用 `int` 统一设备或数据类型，并将结果保存到 `best_epoch`。
        best_epoch = int(checkpoint["best_epoch"])
        # 调用 `float` 统一设备或数据类型，并将结果保存到 `best_ic`。
        best_ic = float(checkpoint["best_validation_ic"])
        # 构造键值映射，并保存到 `best_validation_metrics`，供后续按键查询。
        best_validation_metrics = {
            key: float(value)
            for key, value in checkpoint.get(
                "best_validation_metrics", {}
            ).items()
        }
        # 调用 `int` 统一设备或数据类型，并将结果保存到 `patience_count`。
        patience_count = int(checkpoint["epochs_without_improvement"])
        # 检查 `history_path.exists()`；根据条件结果选择对应处理分支。
        if history_path.exists():
            # 调用 `pd.read_parquet(history_path).to_dict` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `history`。
            history = pd.read_parquet(history_path).to_dict("records")
        # 调用 `print` 记录当前流程状态或诊断信息。
        print(f"[{year}/seed={seed}] 从 epoch {start_epoch} 续训。")

    # 调用 `time.perf_counter` 读取高精度单调时钟以计算耗时，并将结果保存到 `started`。
    started = time.perf_counter()
    # 设置 `stopped_early` 的固定配置值，供后续流程统一使用。
    stopped_early = False
    # 遍历 `range(start_epoch, TRAINING_CONFIG.max_epochs + 1)`，并把当前元素绑定到 `epoch` 逐项处理。
    for epoch in range(start_epoch, TRAINING_CONFIG.max_epochs + 1):
        # 调用 `time.perf_counter` 读取高精度单调时钟以计算耗时，并将结果保存到 `epoch_started`。
        epoch_started = time.perf_counter()
        # 调用 `set_train_epoch` 同步 epoch 到采样器并按轮次重建可复现顺序。
        set_train_epoch(train_loader, epoch)
        # 调用 `train_one_epoch` 遍历训练集完成一个 epoch 的前向、反向传播和参数更新，并将结果保存到 `train_metrics`。
        train_metrics = train_one_epoch(model, train_loader, optimizer)
        # 调用 `evaluate_model` 在不更新参数的条件下计算验证或样本外评价指标，并将结果保存到 `validation_metrics`。
        validation_metrics = evaluate_model(model, validation_loader)
        # 读取 `validation_metrics['validation_ic']` 的当前值，并赋给 `validation_ic`。
        validation_ic = validation_metrics["validation_ic"]
        # 根据 `validation_ic > best_ic + TRAINING_CONFIG.improvement_tolerance` 计算条件结果，并保存到 `improved`。
        improved = validation_ic > best_ic + TRAINING_CONFIG.improvement_tolerance
        # 检查 `improved`；根据条件结果选择对应处理分支。
        if improved:
            # 组织一组有序值，并保存到 `(best_ic, best_epoch, patience_count)`。
            best_ic, best_epoch, patience_count = validation_ic, epoch, 0
            # 构造键值映射，并保存到 `best_validation_metrics`，供后续按键查询。
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
        # 当前置条件不成立时进入备用分支，保证其余合法情况仍被处理。
        else:
            # 用 `1` 对 `patience_count` 做 `Add` 累积更新。
            patience_count += 1
        # 把本轮结果追加到 `history`，累积后续需要汇总的记录。
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
        # 调用 `atomic_parquet` 先写临时文件再原子替换正式产物，避免留下半成品。
        atomic_parquet(pd.DataFrame(history), history_path)
        # 调用 `checkpoint_payload` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `state`。
        state = checkpoint_payload(
            model, optimizer, train_loader.generator.get_state(), epoch, best_epoch,
            best_ic, best_validation_metrics, patience_count, signature,
            history[-1],
        )
        # 调用 `atomic_checkpoint` 执行 `atomic_checkpoint` 对应的明确变换，并产生后续步骤使用的结果。
        atomic_checkpoint(state, last_path)
        # 检查 `improved`；根据条件结果选择对应处理分支。
        if improved:
            # 调用 `atomic_checkpoint` 执行 `atomic_checkpoint` 对应的明确变换，并产生后续步骤使用的结果。
            atomic_checkpoint(state, best_path)
        # 调用 `print` 记录当前流程状态或诊断信息。
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
        # 检查 `patience_count >= TRAINING_CONFIG.early_stopping_patience`；根据条件结果选择对应处理分支。
        if patience_count >= TRAINING_CONFIG.early_stopping_patience:
            # 设置 `stopped_early` 的固定配置值，供后续流程统一使用。
            stopped_early = True
            # 终止当前循环，避免继续处理已经不满足条件的元素。
            break
    # 检查 `not best_path.exists()`；根据条件结果选择对应处理分支。
    if not best_path.exists():
        # 抛出 `FileNotFoundError(f'训练后没有最佳 checkpoint：{best_path}')`，阻止不满足数据契约的流程继续执行。
        raise FileNotFoundError(f"训练后没有最佳 checkpoint：{best_path}")
    # 调用 `next` 从迭代器取出下一批数据，并将结果保存到 `best_epoch_history`。
    best_epoch_history = next(
        row for row in history if int(row["epoch"]) == best_epoch
    )
    # 构造键值映射，并保存到 `summary`，供后续按键查询。
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
    # 调用 `atomic_json` 先写临时文件再原子替换正式产物，避免留下半成品。
    atomic_json({"run_signature": signature, "summary": summary}, completed_path)
    # 将 `(load_best_model(best_path), summary)` 作为当前函数结果返回给调用方。
    return load_best_model(best_path), summary


# 定义 `run_smoke_test` 函数，封装这一阶段可独立调用的处理步骤。
def run_smoke_test() -> pd.DataFrame:
    """运行小批量前向、反向和验证冒烟测试，不生成正式产物。

    输入：
        无显式输入参数。
    输出：
        pd.DataFrame。
    """
    # 组织一组有序值，并保存到 `(year, seed)`。
    year, seed = TRAINING_CONFIG.forecast_years[0], TRAINING_CONFIG.seeds[0]
    # 调用 `set_random_seed` 同时固定 Python、NumPy 和 PyTorch 随机状态。
    set_random_seed(seed)
    # 按 `TRAINING_CONFIG.loss_name == TEMPORAL_TURNOVER_LOSS` 在两个候选值间选择，并保存到 `smoke_batch_mode`。
    smoke_batch_mode = (
        "trade_date"
        if TRAINING_CONFIG.loss_name == TEMPORAL_TURNOVER_LOSS
        else "fixed_samples"
    )
    # 调用 `make_loaders` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `(train_loader, validation_loader, _)`。
    train_loader, validation_loader, _ = make_loaders(
        year, seed, batch_size=64, inference_batch_size=64, workers=0,
        train_batch_mode=smoke_batch_mode,
    )
    # 调用 `build_model` 根据模型配置创建网络并部署到目标设备，并将结果保存到 `model`。
    model = build_model()
    # 调用 `build_optimizer` 根据训练配置为可学习参数创建优化器，并将结果保存到 `optimizer`。
    optimizer = build_optimizer(model)
    # 调用 `train_one_epoch` 遍历训练集完成一个 epoch 的前向、反向传播和参数更新，并将结果保存到 `training`。
    training = train_one_epoch(model, train_loader, optimizer, 1)
    # 调用 `evaluate_model` 在不更新参数的条件下计算验证或样本外评价指标，并将结果保存到 `validation`。
    validation = evaluate_model(model, validation_loader, 1)
    # 调用 `canonical_state` 提取去除编译包装后的标准模型权重，并将结果保存到 `state`。
    state = canonical_state(model)
    # 在 `tempfile.TemporaryDirectory()` 管理的上下文中执行本段逻辑，并保证资源按约定释放。
    with tempfile.TemporaryDirectory() as temporary_directory:
        # 计算 `Path(temporary_directory) / 'smoke.pt'`，并将数值结果保存到 `path`。
        path = Path(temporary_directory) / "smoke.pt"
        # 调用 `torch.save` 将当前结果持久化到目标文件。
        torch.save({"model_state": state}, path)
        # 调用 `torch.load` 读取源数据，并将结果保存到 `checkpoint`。
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
        # 调用 `OrderFlowVisionTransformer(MODEL_CONFIG).to` 统一设备或数据类型，并将结果保存到 `raw_model`。
        raw_model = OrderFlowVisionTransformer(MODEL_CONFIG).to(DEVICE)
        # 调用 `raw_model.load_state_dict` 读取并解析后续计算所需的数据或运行状态。
        raw_model.load_state_dict(checkpoint["model_state"])
    # 删除 `model, raw_model, optimizer`，释放不再使用的引用。
    del model, raw_model, optimizer
    # 检查 `DEVICE.type == 'cuda'`；根据条件结果选择对应处理分支。
    if DEVICE.type == "cuda":
        # 调用 `torch.cuda.empty_cache` 释放当前步骤占用的资源。
        torch.cuda.empty_cache()
    # 将 `pd.DataFrame([{'forecast_year': year, 'train_ic': training['train_ic'], 'validation_ic': valida…` 作为当前函数结果返回给调用方。
    return pd.DataFrame(
        [{"forecast_year": year, "train_ic": training["train_ic"],
          "validation_ic": validation["validation_ic"], "canonical_checkpoint": True}]
    )


# 定义内部辅助函数 `_trade_date_preflight_batches`，封装仅供本模块复用的计算步骤。
def _trade_date_preflight_batches() -> list[tuple[int, str, dict[str, torch.Tensor]]]:
    """逐日抽取预检批次，核对日批次边界、标签有限性和时序对齐。

    输入：
        无显式输入参数。
    输出：
        list[tuple[int, str, dict[str, torch.Tensor]]]。
    """
    # 调用 `sample_index['date_id'].to_numpy` 把表格列转换为指定精度的 NumPy 数组，并将结果保存到 `date_ids`。
    date_ids = sample_index["date_id"].to_numpy(np.int32, copy=False)
    # 读取 `TRAINING_CONFIG.seeds[0]` 的当前值，并赋给 `seed`。
    seed = TRAINING_CONFIG.seeds[0]
    # 组织一组有序值，并保存到 `(max_year, max_size)`。
    max_year, max_size = None, -1
    # 遍历 `TRAINING_CONFIG.forecast_years`，并把当前元素绑定到 `year` 逐项处理。
    for year in TRAINING_CONFIG.forecast_years:
        # 调用 `load_split` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `split`。
        split = load_split(year)
        # 调用 `TradeDateBatchSampler` 按交易日完整截面构造 batch 采样器，并将结果保存到 `sampler`。
        sampler = TradeDateBatchSampler(
            split["train_indices"], date_ids, seed=seed, shuffle=True
        )
        # 调用 `int` 统一设备或数据类型，并将结果保存到 `current_max`。
        current_max = int(sampler.batch_sizes.max())
        # 检查 `current_max > max_size`；根据条件结果选择对应处理分支。
        if current_max > max_size:
            # 组织一组有序值，并保存到 `(max_year, max_size)`。
            max_year, max_size = year, current_max
    # 检查 `max_year is None`；根据条件结果选择对应处理分支。
    if max_year is None:
        # 抛出 `ValueError('没有可用于日截面显存预检的年度')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("没有可用于日截面显存预检的年度")

    # 调用 `load_split` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `split`。
    split = load_split(max_year)
    # 调用 `TradeDateBatchSampler` 按交易日完整截面构造 batch 采样器，并将结果保存到 `sampler`。
    sampler = TradeDateBatchSampler(
        split["train_indices"], date_ids, seed=seed, shuffle=True
    )
    # 读取 `sampler.batch_sizes` 的当前值，并赋给 `sizes`。
    sizes = sampler.batch_sizes
    # 调用 `int` 统一设备或数据类型，并将结果保存到 `max_batch`。
    max_batch = int(np.argmax(sizes))
    # 调用 `int` 统一设备或数据类型，并将结果保存到 `median_batch`。
    median_batch = int(np.argmin(np.abs(sizes - np.median(sizes))))
    # 构造当前步骤所需的列表，并保存到 `selected`。
    selected = [(max_batch, "max"), (median_batch, "median")]
    # 检查 `sizes[max_batch] == sizes[median_batch]`；根据条件结果选择对应处理分支。
    if sizes[max_batch] == sizes[median_batch]:
        # 调用 `int` 统一设备或数据类型，并将结果保存到 `min_batch`。
        min_batch = int(np.argmin(sizes))
        # 组织一组有序值，并保存到 `selected[1]`。
        selected[1] = (min_batch, "min")
    # 检查 `selected[0][0] == selected[1][0]`；根据条件结果选择对应处理分支。
    if selected[0][0] == selected[1][0]:
        # 读取 `selected[:1]` 的当前值，并赋给 `selected`。
        selected = selected[:1]

    # 调用 `MemmapBatchCollator` 把样本整理为批张量并移动到目标设备，并将结果保存到 `collator`。
    collator = MemmapBatchCollator()
    # 构造当前步骤所需的列表，并保存到 `result`。
    result = []
    # 遍历 `selected`，并把当前元素绑定到 `(batch_number, label)` 逐项处理。
    for batch_number, label in selected:
        # 调用 `sampler.batch_slice` 取得当前 batch 在样本索引中的切片范围，并将结果保存到 `positions`。
        positions = sampler.batch_slice(batch_number)
        # 读取 `split['train_indices'][positions]` 的当前值，并赋给 `global_indices`。
        global_indices = split["train_indices"][positions]
        # 把本轮结果追加到 `result`，累积后续需要汇总的记录。
        result.append(
            (max_year, label, collator(global_indices.tolist()))
        )
    # 将 `result` 作为当前函数结果返回给调用方。
    return result


# 定义内部辅助函数 `_temporal_preflight_pairs`，封装仅供本模块复用的计算步骤。
def _temporal_preflight_pairs() -> list[tuple[
    int, str, dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, object]
]]:
    """选择含最大显存需求在内的十二组不同相邻截面 shape。

    输入：
        无显式输入参数。
    输出：
        list[tuple[int, str, dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, object]]]。
    """

    # 调用 `sample_index['date_id'].to_numpy` 把表格列转换为指定精度的 NumPy 数组，并将结果保存到 `date_ids`。
    date_ids = sample_index["date_id"].to_numpy(np.int32, copy=False)
    # 调用 `sample_index['instrument_id'].to_numpy` 把表格列转换为指定精度的 NumPy 数组，并将结果保存到 `instrument_ids`。
    instrument_ids = sample_index["instrument_id"].to_numpy(copy=False)
    # 读取 `TRAINING_CONFIG.seeds[0]` 的当前值，并赋给 `seed`。
    seed = TRAINING_CONFIG.seeds[0]
    # 构造当前步骤所需的列表，并保存到 `candidates`。
    candidates: list[dict[str, object]] = []
    # 遍历 `TRAINING_CONFIG.forecast_years`，并把当前元素绑定到 `year` 逐项处理。
    for year in TRAINING_CONFIG.forecast_years:
        # 调用 `load_split` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `split`。
        split = load_split(year)
        # 调用 `TradeDateBatchSampler` 按交易日完整截面构造 batch 采样器，并将结果保存到 `sampler`。
        sampler = TradeDateBatchSampler(
            split["train_indices"], date_ids, seed=seed, shuffle=False,
            instrument_ids=instrument_ids,
        )
        # 遍历 `enumerate(sampler.temporal_pairs or [])`，并把当前元素绑定到 `(pair_number, pair)` 逐项处理。
        for pair_number, pair in enumerate(sampler.temporal_pairs or []):
            # 调用 `int` 统一设备或数据类型，并将结果保存到 `previous_size`。
            previous_size = int(sampler.batch_sizes[pair_number])
            # 调用 `int` 统一设备或数据类型，并将结果保存到 `current_size`。
            current_size = int(sampler.batch_sizes[pair_number + 1])
            # 调用 `int` 统一设备或数据类型，并将结果保存到 `common_count`。
            common_count = int(pair["common_count"])
            # 把本轮结果追加到 `candidates`，累积后续需要汇总的记录。
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
    # 检查 `not candidates`；根据条件结果选择对应处理分支。
    if not candidates:
        # 抛出 `ValueError('没有可用于时序换手显存预检的相邻日期对')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("没有可用于时序换手显存预检的相邻日期对")
    # 构造键值映射，并保存到 `unique_by_shape`，供后续按键查询。
    unique_by_shape = {item["shape"]: item for item in candidates}
    # 调用 `sorted` 整理数据结构，并将结果保存到 `ordered`。
    ordered = sorted(
        unique_by_shape.values(), key=lambda item: int(item["score"])
    )
    # 调用 `min` 汇总当前数据，并将统计量保存到 `test_count`。
    test_count = min(12, len(ordered))
    # 调用 `np.linspace` 按目标形状初始化数值容器 `quantile_positions`。
    quantile_positions = np.linspace(
        0, len(ordered) - 1, num=test_count, dtype=np.int64
    )
    # 构造当前步骤所需的列表，并保存到 `selected`。
    selected = [ordered[-1]]
    # 把一组结果展开追加到 `selected`，保持统一的一维记录序列。
    selected.extend(
        ordered[int(position)]
        for position in quantile_positions
        if ordered[int(position)]["shape"] != selected[0]["shape"]
    )
    # 读取 `selected[:test_count]` 的当前值，并赋给 `selected`。
    selected = selected[:test_count]

    # 调用 `MemmapBatchCollator` 把样本整理为批张量并移动到目标设备，并将结果保存到 `collator`。
    collator = MemmapBatchCollator()
    # 构造当前步骤所需的列表，并保存到 `result`。
    result = []
    # 遍历 `enumerate(selected)`，并把当前元素绑定到 `(selection_number, item)` 逐项处理。
    for selection_number, item in enumerate(selected):
        # 读取 `item['sampler']` 的当前值，并赋给 `sampler`。
        sampler = item["sampler"]
        # 读取 `item['split']` 的当前值，并赋给 `split`。
        split = item["split"]
        # 调用 `int` 统一设备或数据类型，并将结果保存到 `pair_number`。
        pair_number = int(item["pair_number"])
        # 调用 `sampler.batch_slice` 取得当前 batch 在样本索引中的切片范围，并将结果保存到 `previous_positions`。
        previous_positions = sampler.batch_slice(pair_number)
        # 调用 `sampler.batch_slice` 取得当前 batch 在样本索引中的切片范围，并将结果保存到 `current_positions`。
        current_positions = sampler.batch_slice(pair_number + 1)
        # 读取 `split['train_indices'][previous_positions]` 的当前值，并赋给 `previous_indices`。
        previous_indices = split["train_indices"][previous_positions]
        # 读取 `split['train_indices'][current_positions]` 的当前值，并赋给 `current_indices`。
        current_indices = split["train_indices"][current_positions]
        # 把本轮结果追加到 `result`，累积后续需要汇总的记录。
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
    # 将 `result` 作为当前函数结果返回给调用方。
    return result


# 定义 `wait_for_preflight_barrier` 函数，封装这一阶段可独立调用的处理步骤。
def wait_for_preflight_barrier() -> None:
    """让同一 GPU 上的预检 worker 在模型就绪后同时开始。

    输入：
        无显式输入参数。
    输出：
        None。
    """

    # 调用 `os.getenv` 读取环境变量中的运行配置，并将结果保存到 `barrier_value`。
    barrier_value = os.getenv("VIT_PREFLIGHT_BARRIER_DIR")
    # 检查 `not barrier_value`；根据条件结果选择对应处理分支。
    if not barrier_value:
        # 将 `空值` 作为当前函数结果返回给调用方。
        return
    # 调用 `Path` 规范化文件路径，并将结果保存到 `barrier_dir`。
    barrier_dir = Path(barrier_value)
    # 调用 `int` 统一设备或数据类型，并将结果保存到 `rank`。
    rank = int(os.getenv("VIT_PREFLIGHT_RANK", "0"))
    # 调用 `int` 统一设备或数据类型，并将结果保存到 `world_size`。
    world_size = int(os.getenv("VIT_PREFLIGHT_WORLD_SIZE", "1"))
    # 调用 `barrier_dir.mkdir` 创建输出目录，确保后续文件可以安全写入。
    barrier_dir.mkdir(parents=True, exist_ok=True)
    # 调用 `(barrier_dir / f'ready_{rank}').touch` 根据分数计算截面排序或排序索引。
    (barrier_dir / f"ready_{rank}").touch()
    # 计算 `time.monotonic() + 300`，并将数值结果保存到 `deadline`。
    deadline = time.monotonic() + 300
    # 只要 `len(list(barrier_dir.glob('ready_*'))) < world_size` 成立就继续循环，直至满足退出条件。
    while len(list(barrier_dir.glob("ready_*"))) < world_size:
        # 检查 `time.monotonic() >= deadline`；根据条件结果选择对应处理分支。
        if time.monotonic() >= deadline:
            # 抛出 `TimeoutError('同卡并发预检 worker 屏障等待超时')`，阻止不满足数据契约的流程继续执行。
            raise TimeoutError("同卡并发预检 worker 屏障等待超时")
        # 调用 `time.sleep` 短暂等待外部任务状态更新，避免无间隔轮询。
        time.sleep(0.1)


# 定义 `run_full_batch_preflight` 函数，封装这一阶段可独立调用的处理步骤。
def run_full_batch_preflight() -> pd.DataFrame | None:
    """在正式训练前检查全量批次形状、日期与显存相关约束。

    输入：
        无显式输入参数。
    输出：
        pd.DataFrame | None。
    """
    # 检查 `DEVICE.type != 'cuda'`；根据条件结果选择对应处理分支。
    if DEVICE.type != "cuda":
        # 调用 `print` 记录当前流程状态或诊断信息。
        print("当前无 CUDA，跳过正式 batch 显存预检。")
        # 将 `None` 作为当前函数结果返回给调用方。
        return None
    # 组织一组有序值，并保存到 `(year, seed)`。
    year, seed = TRAINING_CONFIG.forecast_years[0], TRAINING_CONFIG.seeds[0]
    # 根据 `TRAINING_CONFIG.loss_name == TEMPORAL_TURNOVER_LOSS` 计算条件结果，并保存到 `temporal_mode`。
    temporal_mode = TRAINING_CONFIG.loss_name == TEMPORAL_TURNOVER_LOSS
    # 检查 `temporal_mode`；根据条件结果选择对应处理分支。
    if temporal_mode:
        # 调用 `_temporal_preflight_pairs` 执行 `_temporal_preflight_pairs` 对应的明确变换，并产生后续步骤使用的结果，并将结果保存到 `temporal_pairs`。
        temporal_pairs = _temporal_preflight_pairs()
        # 构造当前步骤所需的列表，并保存到 `batches`。
        batches = []
    # 检查 `TRAINING_CONFIG.train_batch_mode == 'trade_date'`；根据条件结果选择对应处理分支。
    elif TRAINING_CONFIG.train_batch_mode == "trade_date":
        # 构造当前步骤所需的列表，并保存到 `temporal_pairs`。
        temporal_pairs = []
        # 调用 `_trade_date_preflight_batches` 执行 `_trade_date_preflight_batches` 对应的明确变换，并产生后续步骤使用的结果，并将结果保存到 `batches`。
        batches = _trade_date_preflight_batches()
    # 当前置条件不成立时进入备用分支，保证其余合法情况仍被处理。
    else:
        # 构造当前步骤所需的列表，并保存到 `temporal_pairs`。
        temporal_pairs = []
        # 调用 `make_loaders` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `(loader, _, _)`。
        loader, _, _ = make_loaders(year, seed)
        # 构造当前步骤所需的列表，并保存到 `batches`。
        batches = [(year, "fixed", next(iter(loader)))]

    # 调用 `build_model` 根据模型配置创建网络并部署到目标设备，并将结果保存到 `model`。
    model = build_model()
    # 调用 `build_optimizer` 根据训练配置为可学习参数创建优化器，并将结果保存到 `optimizer`。
    optimizer = build_optimizer(model)
    # 调用 `wait_for_preflight_barrier` 执行 `wait_for_preflight_barrier` 对应的明确变换，并产生后续步骤使用的结果。
    wait_for_preflight_barrier()
    # 构造当前步骤所需的列表，并保存到 `rows`。
    rows = []
    # 尝试执行本段主逻辑；若发生异常，则交由后续异常分支处理。
    try:
        # 检查 `temporal_mode`；根据条件结果选择对应处理分支。
        if temporal_mode:
            # 调用 `optimizer.zero_grad` 清空上一轮梯度，避免跨批次意外累积。
            optimizer.zero_grad(set_to_none=True)
            # 遍历 `temporal_pairs`，并把当前元素绑定到 `(batch_year, shape_test, previous_batch, current_batch, pair)` 逐项处理。
            for batch_year, shape_test, previous_batch, current_batch, pair in temporal_pairs:
                # 调用 `torch.cuda.empty_cache` 释放当前步骤占用的资源。
                torch.cuda.empty_cache()
                # 调用 `torch.cuda.reset_peak_memory_stats` 清零 CUDA 峰值显存计数器。
                torch.cuda.reset_peak_memory_stats()
                # 调用 `time.perf_counter` 读取高精度单调时钟以计算耗时，并将结果保存到 `started`。
                started = time.perf_counter()
                # 调用 `move_batch` 把当前 batch 的张量移动到目标计算设备，并将结果保存到 `(previous_features, _)`。
                previous_features, _ = move_batch(previous_batch)
                # 调用 `move_batch` 把当前 batch 的张量移动到目标计算设备，并将结果保存到 `(current_features, current_targets)`。
                current_features, current_targets = move_batch(current_batch)
                # 在 `autocast_context()` 管理的上下文中执行本段逻辑，并保证资源按约定释放。
                with autocast_context():
                    # 调用 `model` 对当前输入 batch 执行模型前向传播，并将结果保存到 `previous_with_graph`。
                    previous_with_graph = model(previous_features)
                # 复制 `previous_with_graph.detach`，避免后续更新污染原对象，并保存到 `previous_predictions`。
                previous_predictions = previous_with_graph.detach()
                # 删除 `previous_with_graph, previous_features`，释放不再使用的引用。
                del previous_with_graph, previous_features
                # 在 `autocast_context()` 管理的上下文中执行本段逻辑，并保证资源按约定释放。
                with autocast_context():
                    # 调用 `model` 对当前输入 batch 执行模型前向传播，并将结果保存到 `current_predictions`。
                    current_predictions = model(current_features)
                # 调用 `temporal_turnover_loss_components` 按当前实验口径计算可反向传播的训练损失，并将结果保存到 `components`。
                components = temporal_turnover_loss_components(
                    current_predictions, current_targets, previous_predictions,
                    pair["previous_positions"], pair["current_positions"],
                    TRAINING_CONFIG.rank_temperature,
                    TRAINING_CONFIG.turnover_lambda,
                )
                # 读取 `components['loss']` 的当前值，并赋给 `loss`。
                loss = components["loss"]
                # 检查 `not torch.isfinite(loss)`；根据条件结果选择对应处理分支。
                if not torch.isfinite(loss):
                    # 抛出 `FloatingPointError('时序换手预检损失出现 NaN 或 Inf')`，阻止不满足数据契约的流程继续执行。
                    raise FloatingPointError("时序换手预检损失出现 NaN 或 Inf")
                # 调用 `(loss / len(temporal_pairs)).backward` 反向传播当前损失，计算模型参数梯度。
                (loss / len(temporal_pairs)).backward()
                # 调用 `torch.cuda.synchronize` 等待当前 GPU 队列执行完毕后再计时或验收。
                torch.cuda.synchronize()
                # 调用 `int` 统一设备或数据类型，并将结果保存到 `common_count`。
                common_count = int(pair["common_count"])
                # 把本轮结果追加到 `rows`，累积后续需要汇总的记录。
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
                # 删除 `(previous_batch, current_batch, current_features, current_targets, prev…`，释放不再使用的引用。
                del (
                    previous_batch, current_batch, current_features,
                    current_targets, previous_predictions, current_predictions,
                    components, loss,
                )
            # 调用 `optimizer.step` 推进一次优化器或调度器状态。
            optimizer.step()
            # 调用 `torch.cuda.synchronize` 等待当前 GPU 队列执行完毕后再计时或验收。
            torch.cuda.synchronize()
        # 当前置条件不成立时进入备用分支，保证其余合法情况仍被处理。
        else:
            # 遍历 `batches`，并把当前元素绑定到 `(batch_year, shape_test, batch)` 逐项处理。
            for batch_year, shape_test, batch in batches:
                # 调用 `torch.cuda.empty_cache` 释放当前步骤占用的资源。
                torch.cuda.empty_cache()
                # 调用 `torch.cuda.reset_peak_memory_stats` 清零 CUDA 峰值显存计数器。
                torch.cuda.reset_peak_memory_stats()
                # 调用 `time.perf_counter` 读取高精度单调时钟以计算耗时，并将结果保存到 `started`。
                started = time.perf_counter()
                # 调用 `move_batch` 把当前 batch 的张量移动到目标计算设备，并将结果保存到 `(features, targets)`。
                features, targets = move_batch(batch)
                # 调用 `optimizer.zero_grad` 清空上一轮梯度，避免跨批次意外累积。
                optimizer.zero_grad(set_to_none=True)
                # 在 `autocast_context()` 管理的上下文中执行本段逻辑，并保证资源按约定释放。
                with autocast_context():
                    # 调用 `model` 对当前输入 batch 执行模型前向传播，并将结果保存到 `predictions`。
                    predictions = model(features)
                # 调用 `compute_training_loss` 按当前实验口径计算可反向传播的训练损失，并将结果保存到 `loss`。
                loss = compute_training_loss(predictions, targets)
                # 检查 `not torch.isfinite(loss)`；根据条件结果选择对应处理分支。
                if not torch.isfinite(loss):
                    # 抛出 `FloatingPointError('预检损失出现 NaN 或 Inf')`，阻止不满足数据契约的流程继续执行。
                    raise FloatingPointError("预检损失出现 NaN 或 Inf")
                # 调用 `loss.backward` 反向传播当前损失，计算模型参数梯度。
                loss.backward()
                # 调用 `optimizer.step` 推进一次优化器或调度器状态。
                optimizer.step()
                # 调用 `torch.cuda.synchronize` 等待当前 GPU 队列执行完毕后再计时或验收。
                torch.cuda.synchronize()
                # 把本轮结果追加到 `rows`，累积后续需要汇总的记录。
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
                # 删除 `batch, features, targets, predictions, loss`，释放不再使用的引用。
                del batch, features, targets, predictions, loss
    # 捕获 `except torch.OutOfMemoryError as error` 对应的异常，并执行明确的降级、记录或清理逻辑。
    except torch.OutOfMemoryError as error:
        # 抛出 `RuntimeError(f'A2 单任务模式={TRAINING_CONFIG.train_batch_mode}、损失={TRAINING_CONFIG.loss_name} 发生 CU…`，阻止不满足数据契约的流程继续执行。
        raise RuntimeError(
            f"A2 单任务模式={TRAINING_CONFIG.train_batch_mode}、"
            f"损失={TRAINING_CONFIG.loss_name} 发生 CUDA OOM；"
            "程序不会静默改变训练口径。"
        ) from error
    # 无论前序步骤是否抛出异常，都执行本分支完成必要的收尾。
    finally:
        # 删除 `model, optimizer, batches, temporal_pairs`，释放不再使用的引用。
        del model, optimizer, batches, temporal_pairs
        # 调用 `torch.cuda.empty_cache` 释放当前步骤占用的资源。
        torch.cuda.empty_cache()
    # 将 `pd.DataFrame(rows)` 作为当前函数结果返回给调用方。
    return pd.DataFrame(rows)


# 定义 `load_ensemble_inputs` 函数，封装这一阶段可独立调用的处理步骤。
def load_ensemble_inputs(data_root: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """加载各随机种子的预测，并检查业务键和行序一致。

    输入：
        data_root: Path。
    输出：
        tuple[pd.DataFrame, np.ndarray]。
    """
    # 调用 `pd.read_parquet` 读取源数据，并将结果保存到 `index`。
    index = pd.read_parquet(
        data_root / "feature_store" / "sample_index.parquet",
        columns=[
            "sample_index", "trade_date", "instrument_id", "wind_code",
            "can_enter_t1", "label_valid",
        ],
    )
    # 调用 `index['trade_date'].astype` 统一设备或数据类型，并将结果保存到 `index['trade_date']`。
    index["trade_date"] = index["trade_date"].astype(str)
    # 调用 `np.load` 读取源数据，并将结果保存到 `targets`。
    targets = np.load(data_root / "feature_store" / "targets.npy", mmap_mode="r")
    # 将 `(index.set_index('sample_index', drop=False), targets)` 作为当前函数结果返回给调用方。
    return index.set_index("sample_index", drop=False), targets


# 定义 `build_ensemble` 函数，封装这一阶段可独立调用的处理步骤。
def build_ensemble(year: Any, seeds: Any, model_root: Any, prediction_dir: Any, index: Any, targets: Any) -> int:
    """按既定尺度组合多个随机种子的预测，并保存正式年度信号。

    输入：
        year: Any。
        seeds: Any。
        model_root: Any。
        prediction_dir: Any。
        index: Any。
        targets: Any。
    输出：
        int。
    """
    # 构造键值映射，并保存到 `seed_frames`，供后续按键查询。
    seed_frames = {}
    # 遍历 `seeds`，并把当前元素绑定到 `seed` 逐项处理。
    for seed in seeds:
        # 计算 `model_root / f'forecast_{year}' / f'seed_{seed}' / 'prediction.parquet'`，并将数值结果保存到 `path`。
        path = model_root / f"forecast_{year}" / f"seed_{seed}" / "prediction.parquet"
        # 检查 `not path.exists()`；根据条件结果选择对应处理分支。
        if not path.exists():
            # 抛出 `FileNotFoundError(path)`，阻止不满足数据契约的流程继续执行。
            raise FileNotFoundError(path)
        # 调用 `pd.read_parquet` 读取源数据，并将结果保存到 `seed_frames[seed]`。
        seed_frames[seed] = pd.read_parquet(path)
    # 调用 `seed_frames[seeds[0]]['sample_index'].to_numpy` 把表格列转换为指定精度的 NumPy 数组，并将结果保存到 `reference`。
    reference = seed_frames[seeds[0]]["sample_index"].to_numpy(np.int64)
    # 调用 `reset_index` 重建连续行索引并丢弃旧索引，并将结果保存到 `output`。
    output = index.loc[
        reference,
        ["sample_index", "trade_date", "instrument_id", "wind_code",
         "can_enter_t1", "label_valid"],
    ].reset_index(drop=True)
    # 调用 `np.asarray` 转换数据表示，并将标准化对象保存到 `output['target_zscore']`。
    output["target_zscore"] = np.asarray(targets[reference], dtype=np.float32)
    # 构造当前步骤所需的列表，并保存到 `raw_columns`。
    raw_columns = []
    # 遍历 `seeds`，并把当前元素绑定到 `seed` 逐项处理。
    for seed in seeds:
        # 读取 `seed_frames[seed]` 的当前值，并赋给 `frame`。
        frame = seed_frames[seed]
        # 检查 `not np.array_equal(reference, frame['sample_index'].to_numpy(np.int64))`；根据条件结果选择对应处理分支。
        if not np.array_equal(reference, frame["sample_index"].to_numpy(np.int64)):
            # 抛出 `AssertionError(f'{year} 三个 seed 的预测行号不一致')`，阻止不满足数据契约的流程继续执行。
            raise AssertionError(f"{year} 三个 seed 的预测行号不一致")
        # 计算 `f'prediction_seed_{seed}'`，并将结果写入 `column`。
        column = f"prediction_seed_{seed}"
        # 调用 `frame['prediction'].to_numpy` 执行模型推理并生成样本预测分数，并将结果保存到 `output[column]`。
        output[column] = frame["prediction"].to_numpy(np.float32)
        # 把本轮结果追加到 `raw_columns`，累积后续需要汇总的记录。
        raw_columns.append(column)
    # 调用 `output[raw_columns].astype` 统一设备或数据类型，并将结果保存到 `raw`。
    raw = output[raw_columns].astype(np.float64)
    # 调用 `raw.groupby` 按指定字段建立分组对象，并将结果保存到 `grouped`。
    grouped = raw.groupby(output["trade_date"], sort=False)
    # 调用 `grouped.transform` 按约定截面或分组口径汇总统计结果，并将结果保存到 `means`。
    means = grouped.transform("mean")
    # 调用 `grouped.transform` 按约定截面或分组口径汇总统计结果，并将结果保存到 `standard_deviations`。
    standard_deviations = grouped.transform("std", ddof=0)
    # 检查 `(~np.isfinite(standard_deviations) | standard_deviations.le(0)).any().any()`；根据条件结果选择对应处理分支。
    if (~np.isfinite(standard_deviations) | standard_deviations.le(0)).any().any():
        # 抛出 `ValueError(f'{year} 存在无法标准化的日度预测')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(f"{year} 存在无法标准化的日度预测")
    # 计算 `(raw - means) / standard_deviations`，并将数值结果保存到 `zscores`。
    zscores = (raw - means) / standard_deviations
    # 构造当前步骤所需的列表，并保存到 `zscore_columns`。
    zscore_columns = []
    # 遍历 `raw_columns`，并把当前元素绑定到 `column` 逐项处理。
    for column in raw_columns:
        # 计算 `f'{column}_zscore'`，并将结果写入 `zscore_column`。
        zscore_column = f"{column}_zscore"
        # 调用 `zscores[column].to_numpy` 把表格列转换为指定精度的 NumPy 数组，并将结果保存到 `output[zscore_column]`。
        output[zscore_column] = zscores[column].to_numpy(np.float32)
        # 把本轮结果追加到 `zscore_columns`，累积后续需要汇总的记录。
        zscore_columns.append(zscore_column)
    # 调用 `raw.mean(axis=1).to_numpy` 把表格列转换为指定精度的 NumPy 数组，并将结果保存到 `output['prediction_ensemble_raw']`。
    output["prediction_ensemble_raw"] = raw.mean(axis=1).to_numpy(np.float32)
    # 调用 `output[zscore_columns].mean(axis=1).astype` 统一设备或数据类型，并将结果保存到 `output['prediction_ensemble']`。
    output["prediction_ensemble"] = output[zscore_columns].mean(axis=1).astype(np.float32)
    # 读取 `output.groupby('trade_date', sort=False)[zscore_columns]` 的当前值，并赋给 `audit`。
    audit = output.groupby("trade_date", sort=False)[zscore_columns]
    # 检查 `audit.mean().abs().to_numpy().max() > 1e-05 or (audit.std(ddof=0) - 1).abs().to_numpy().max() >…`；根据条件结果选择对应处理分支。
    if (
        audit.mean().abs().to_numpy().max() > 1e-5
        or (audit.std(ddof=0) - 1).abs().to_numpy().max() > 1e-5
    ):
        # 抛出 `AssertionError(f'{year} seed 预测日度 Z-score 审计失败')`，阻止不满足数据契约的流程继续执行。
        raise AssertionError(f"{year} seed 预测日度 Z-score 审计失败")
    # 调用 `atomic_parquet` 先写临时文件再原子替换正式产物，避免留下半成品。
    atomic_parquet(output, prediction_dir / f"vit_predictions.{year}.parquet")
    # 将 `len(output)` 作为当前函数结果返回给调用方。
    return len(output)


# 定义 `prediction_counts` 函数，封装这一阶段可独立调用的处理步骤。
def prediction_counts(data_root: Path, years: tuple[int, ...]) -> dict[int, int]:
    """统计各预测年度的文件、日期和样本数量。

    输入：
        data_root: Path。
        years: tuple[int, ...]。
    输出：
        dict[int, int]。
    """
    # 构造键值映射，并保存到 `result`，供后续按键查询。
    result = {}
    # 遍历 `years`，并把当前元素绑定到 `year` 逐项处理。
    for year in years:
        # 在 `np.load(data_root / 'feature_store' / 'rolling_splits…` 管理的上下文中执行本段逻辑，并保证资源按约定释放。
        with np.load(
            data_root / "feature_store" / "rolling_splits" / f"forecast_{year}.npz",
            allow_pickle=False,
        ) as archive:
            # 调用 `len` 汇总当前数据，并将统计量保存到 `result[year]`。
            result[year] = len(archive["prediction_indices"])
    # 将 `result` 作为当前函数结果返回给调用方。
    return result


# 定义 `split_batch_summary` 函数，封装这一阶段可独立调用的处理步骤。
def split_batch_summary(
    data_root: Path, forecast_year: int, train_batch_mode: str,
    fixed_batch_size: int, loss_name: str,
) -> dict[str, object]:
    """汇总单个滚动切分中的训练、验证和预测批次。

    输入：
        data_root: Path。
        forecast_year: int。
        train_batch_mode: str。
        fixed_batch_size: int。
        loss_name: str。
    输出：
        dict[str, object]。
    """
    # 在 `np.load(data_root / 'feature_store' / 'rolling_splits…` 管理的上下文中执行本段逻辑，并保证资源按约定释放。
    with np.load(
        data_root / "feature_store" / "rolling_splits"
        / f"forecast_{forecast_year}.npz",
        allow_pickle=False,
    ) as archive:
        # 复制 `archive['train_indices'].copy`，避免后续更新污染原对象，并保存到 `train_indices`。
        train_indices = archive["train_indices"].copy()
    # 检查 `train_batch_mode == 'trade_date'`；根据条件结果选择对应处理分支。
    if train_batch_mode == "trade_date":
        # 调用 `np.load` 读取源数据，并将结果保存到 `date_ids`。
        date_ids = np.load(
            data_root / "feature_store" / "date_ids.npy", mmap_mode="r"
        )
        # 根据 `loss_name == TEMPORAL_TURNOVER_LOSS` 计算条件结果，并保存到 `temporal_mode`。
        temporal_mode = loss_name == TEMPORAL_TURNOVER_LOSS
        # 调用 `TradeDateBatchSampler` 按交易日完整截面构造 batch 采样器，并将结果保存到 `sampler`。
        sampler = TradeDateBatchSampler(
            train_indices, date_ids, seed=0, shuffle=not temporal_mode
        )
        # 调用 `sampler.summary` 按约定截面或分组口径汇总统计结果，并将结果保存到 `summary`。
        summary = sampler.summary()
        # 检查 `temporal_mode`；根据条件结果选择对应处理分支。
        if temporal_mode:
            # 用本轮键值结果更新 `summary`，补齐后续查询所需字段。
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
        # 删除 `date_ids`，释放不再使用的引用。
        del date_ids
    # 当前置条件不成立时进入备用分支，保证其余合法情况仍被处理。
    else:
        # 计算 `len(train_indices) // fixed_batch_size`，并将数值结果保存到 `batch_count`。
        batch_count = len(train_indices) // fixed_batch_size
        # 构造键值映射，并保存到 `summary`，供后续按键查询。
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
    # 将 `{'forecast_year': forecast_year, **summary}` 作为当前函数结果返回给调用方。
    return {"forecast_year": forecast_year, **summary}


# 定义 `training_command` 函数，封装这一阶段可独立调用的处理步骤。
def training_command(args: Any, command: str) -> list[str]:
    """构造启动训练后端的完整命令行参数。

    输入：
        args: Any。
        command: str。
    输出：
        list[str]。
    """
    # 将 `[sys.executable, str(PIPELINE_PATH), command, '--data-root', args.data_root, '--run-name', args…` 作为当前函数结果返回给调用方。
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


# 定义 `worker_command` 函数，封装这一阶段可独立调用的处理步骤。
def worker_command(args: Any, year: int, seed: int, gpu: int) -> list[str]:
    """构造单个年度×seed worker 的命令行参数。

    输入：
        args: Any。
        year: int。
        seed: int。
        gpu: int。
    输出：
        list[str]。
    """
    # 将 `[*training_command(args, 'worker'), '--worker-year', str(year), '--worker-seed', str(seed), '--…` 作为当前函数结果返回给调用方。
    return [
        *training_command(args, "worker"),
        "--worker-year", str(year),
        "--worker-seed", str(seed),
        "--worker-physical-gpu", str(gpu),
    ]


# 定义 `preflight_worker_command` 函数，封装这一阶段可独立调用的处理步骤。
def preflight_worker_command(args: Any, rank: int) -> list[str]:
    """构造正式训练前单 worker 预检命令。

    输入：
        args: Any。
        rank: int。
    输出：
        list[str]。
    """
    # 将 `[*training_command(args, 'preflight-worker'), '--preflight-rank', str(rank)]` 作为当前函数结果返回给调用方。
    return [
        *training_command(args, "preflight-worker"),
        "--preflight-rank", str(rank),
    ]


# 定义 `tail` 函数，封装这一阶段可独立调用的处理步骤。
def tail(path: Path, lines: int = 80) -> str:
    """读取文本文件末尾若干行，用于展示最新训练日志。

    输入：
        path: Path。
        lines: int。
    输出：
        str。
    """
    # 检查 `not path.exists()`；根据条件结果选择对应处理分支。
    if not path.exists():
        # 将 `''` 作为当前函数结果返回给调用方。
        return ""
    # 将 `'\n'.join(path.read_text(encoding='utf-8', errors='replace').splitlines()[-lines:])` 作为当前函数结果返回给调用方。
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


# 定义 `run_worker` 函数，封装这一阶段可独立调用的处理步骤。
def run_worker(args: Any) -> None:
    """启动并等待单个训练 worker，传递其退出状态。

    输入：
        args: Any。
    输出：
        None。
    """
    # 调用 `configure_worker_environment` 整理并返回当前实验的配置对象。
    configure_worker_environment(args)
    # 调用 `initialize_runtime` 按配置初始化设备与运行时状态。
    initialize_runtime()
    # 调用 `train_single_run` 执行模型训练并记录本轮优化指标，并将结果保存到 `(model, summary)`。
    model, summary = train_single_run(args.worker_year, args.worker_seed)
    # 调用 `load_split` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `split`。
    split = load_split(args.worker_year)
    # 调用 `len` 汇总当前数据，并将统计量保存到 `prediction_count`。
    prediction_count = len(split["prediction_indices"])
    # 读取 `prediction_count` 的当前值，并赋给 `summary['prediction_sample_count']`。
    summary["prediction_sample_count"] = prediction_count
    # 检查 `prediction_count`；根据条件结果选择对应处理分支。
    if prediction_count:
        # 调用 `make_loaders` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `(_, _, loader)`。
        _, _, loader = make_loaders(args.worker_year, args.worker_seed)
        # 检查 `loader is None`；根据条件结果选择对应处理分支。
        if loader is None:
            # 抛出 `AssertionError('预测索引非空但 loader 缺失')`，阻止不满足数据契约的流程继续执行。
            raise AssertionError("预测索引非空但 loader 缺失")
        # 调用 `predict_model` 执行模型推理并生成样本预测分数，并将结果保存到 `prediction`。
        prediction = predict_model(model, loader)
        # 调用 `atomic_parquet` 先写临时文件再原子替换正式产物，避免留下半成品。
        atomic_parquet(
            prediction,
            MODEL_ROOT / f"forecast_{args.worker_year}" / f"seed_{args.worker_seed}"
            / "prediction.parquet",
        )
    # 调用 `print` 记录当前流程状态或诊断信息。
    print(json.dumps({"summary": summary}, ensure_ascii=False), flush=True)


# 定义 `finalize_outputs` 函数，封装这一阶段可独立调用的处理步骤。
def finalize_outputs(args: Any, elapsed_seconds: float) -> pd.DataFrame:
    """聚合已完成任务并生成 run 级预测、清单和报告。

    输入：
        args: Any。
        elapsed_seconds: float。
    输出：
        pd.DataFrame。
    """
    # 调用 `Path(args.data_root).resolve` 规范化文件路径，并将结果保存到 `data_root`。
    data_root = Path(args.data_root).resolve()
    # 组织一组有序值，并保存到 `(years, seeds)`。
    years, seeds = parse_csv_ints(args.years), parse_csv_ints(args.seeds)
    # 计算 `VIT_DIR / 'models' / 'runs' / args.run_name`，并将数值结果保存到 `model_root`。
    model_root = VIT_DIR / "models" / "runs" / args.run_name
    # 计算 `data_root / 'predictions' / 'runs' / args.run_name`，并将数值结果保存到 `prediction_dir`。
    prediction_dir = data_root / "predictions" / "runs" / args.run_name
    # 计算 `data_root / 'quality' / 'runs' / args.run_name`，并将数值结果保存到 `quality_dir`。
    quality_dir = data_root / "quality" / "runs" / args.run_name
    # 调用 `prediction_counts` 执行模型推理并生成样本预测分数，并将结果保存到 `counts`。
    counts = prediction_counts(data_root, years)
    # 构造当前步骤所需的列表，并保存到 `records`。
    records = []
    # 遍历 `years`，并把当前元素绑定到 `year` 逐项处理。
    for year in years:
        # 遍历 `seeds`，并把当前元素绑定到 `seed` 逐项处理。
        for seed in seeds:
            # 计算 `model_root / f'forecast_{year}' / f'seed_{seed}' / 'completed.json'`，并将数值结果保存到 `completed_path`。
            completed_path = model_root / f"forecast_{year}" / f"seed_{seed}" / "completed.json"
            # 调用 `read_json` 读取源数据，并将结果保存到 `completed`。
            completed = read_json(completed_path)
            # 检查 `completed is None`；根据条件结果选择对应处理分支。
            if completed is None:
                # 抛出 `FileNotFoundError(completed_path)`，阻止不满足数据契约的流程继续执行。
                raise FileNotFoundError(completed_path)
            # 调用 `dict` 整理数据结构，并将结果保存到 `record`。
            record = dict(completed["summary"])
            # 读取 `counts[year]` 的当前值，并赋给 `record['prediction_sample_count']`。
            record["prediction_sample_count"] = counts[year]
            # 把本轮结果追加到 `records`，累积后续需要汇总的记录。
            records.append(record)
    # 调用 `pd.DataFrame(records).sort_values` 根据分数计算截面排序或排序索引，并将结果保存到 `summary`。
    summary = pd.DataFrame(records).sort_values(["forecast_year", "seed"])
    # 调用 `atomic_parquet` 先写临时文件再原子替换正式产物，避免留下半成品。
    atomic_parquet(summary, quality_dir / "vit_training_summary.parquet")
    # 调用 `load_ensemble_inputs` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `(index, targets)`。
    index, targets = load_ensemble_inputs(data_root)
    # 遍历 `years`，并把当前元素绑定到 `year` 逐项处理。
    for year in years:
        # 检查 `not counts[year]`；根据条件结果选择对应处理分支。
        if not counts[year]:
            # 跳过当前元素的剩余步骤，继续处理下一项。
            continue
        # 调用 `build_ensemble` 根据当前配置和输入构造后续流程所需对象，并将结果保存到 `count`。
        count = build_ensemble(year, seeds, model_root, prediction_dir, index, targets)
        # 检查 `count != counts[year]`；根据条件结果选择对应处理分支。
        if count != counts[year]:
            # 抛出 `AssertionError(f'{year} 集成预测行数不完整')`，阻止不满足数据契约的流程继续执行。
            raise AssertionError(f"{year} 集成预测行数不完整")
        # 调用 `print` 记录当前流程状态或诊断信息。
        print(f"[{year}] 集成预测 {count:,} 条", flush=True)
    # 调用 `read_json` 读取源数据，并将结果保存到 `dataset`。
    dataset = read_json(data_root / "feature_store" / "dataset_manifest.json")
    # 调用 `read_json` 读取源数据，并将结果保存到 `rolling`。
    rolling = read_json(data_root / "feature_store" / "rolling_split_manifest.json")
    # 调用 `read_json` 读取源数据，并将结果保存到 `universe`。
    universe = read_json(data_root / "universe" / "universe_manifest.json")
    # 调用 `atomic_json` 先写临时文件再原子替换正式产物，避免留下半成品。
    atomic_json(
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
    # 将 `summary` 作为当前函数结果返回给调用方。
    return summary


# 定义 `run_parent` 函数，封装这一阶段可独立调用的处理步骤。
def run_parent(args: Any) -> pd.DataFrame:
    """调度多个 GPU worker，并在全部结束后汇总正式产物。

    输入：
        args: Any。
    输出：
        pd.DataFrame。
    """
    # 调用 `parse_csv_ints` 解析用户配置并转换为内部统一口径，并将结果保存到 `device_ids`。
    device_ids = parse_csv_ints(args.device_ids)
    # 组织一组有序值，并保存到 `(years, seeds)`。
    years, seeds = parse_csv_ints(args.years), parse_csv_ints(args.seeds)
    # 调用 `deque` 创建支持队首弹出的任务队列，并将结果保存到 `tasks`。
    tasks = deque((year, seed) for year in sorted(years) for seed in sorted(seeds))
    # 调用 `Path(args.data_root).resolve` 规范化文件路径，并将结果保存到 `data_root`。
    data_root = Path(args.data_root).resolve()
    # 计算 `data_root / 'quality' / 'runs' / args.run_name / 'a2_task_logs'`，并将数值结果保存到 `log_dir`。
    log_dir = data_root / "quality" / "runs" / args.run_name / "a2_task_logs"
    # 调用 `log_dir.mkdir` 创建输出目录，确保后续文件可以安全写入。
    log_dir.mkdir(parents=True, exist_ok=True)
    # 构造当前步骤所需的列表，并保存到 `running`。
    running: list[dict[str, object]] = []
    # 调用 `time.perf_counter` 读取高精度单调时钟以计算耗时，并将结果保存到 `started`。
    started = time.perf_counter()
    # 调用 `print` 记录当前流程状态或诊断信息。
    print(
        f"A2 启动：{len(tasks)} 个任务，GPU={device_ids}，"
        f"每卡并发={args.tasks_per_gpu}，损失={args.loss_name}，"
        f"batch={args.train_batch_mode}",
        flush=True,
    )
    # 尝试执行本段主逻辑；若发生异常，则交由后续异常分支处理。
    try:
        # 只要 `tasks or running` 成立就继续循环，直至满足退出条件。
        while tasks or running:
            # 遍历 `device_ids`，并把当前元素绑定到 `gpu` 逐项处理。
            for gpu in device_ids:
                # 调用 `sum` 汇总当前数据，并将统计量保存到 `active`。
                active = sum(item["gpu"] == gpu for item in running)
                # 只要 `tasks and active < args.tasks_per_gpu` 成立就继续循环，直至满足退出条件。
                while tasks and active < args.tasks_per_gpu:
                    # 调用 `tasks.popleft` 从任务队列头部取出下一项任务，并将结果保存到 `(year, seed)`。
                    year, seed = tasks.popleft()
                    # 计算 `log_dir / f'forecast_{year}.seed_{seed}.log'`，并将数值结果保存到 `log_path`。
                    log_path = log_dir / f"forecast_{year}.seed_{seed}.log"
                    # 调用 `log_path.open` 读取源数据，并将结果保存到 `log_handle`。
                    log_handle = log_path.open("w", encoding="utf-8")
                    # 复制 `os.environ.copy`，避免后续更新污染原对象，并保存到 `environment`。
                    environment = os.environ.copy()
                    # 调用 `str` 把当前值转换为统一字符串表示，并将结果保存到 `environment['CUDA_VISIBLE_DEVICES']`。
                    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
                    # 调用 `subprocess.Popen` 启动独立训练子进程并保留进程句柄，并将结果保存到 `process`。
                    process = subprocess.Popen(
                        worker_command(args, year, seed, gpu),
                        cwd=VIT_DIR,
                        env=environment,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    # 把本轮结果追加到 `running`，累积后续需要汇总的记录。
                    running.append(
                        {"process": process, "gpu": gpu, "year": year, "seed": seed,
                         "log_path": log_path, "log_handle": log_handle,
                         "started": time.perf_counter()}
                    )
                    # 用 `1` 对 `active` 做 `Add` 累积更新。
                    active += 1
                    # 调用 `print` 记录当前流程状态或诊断信息。
                    print(f"启动 GPU{gpu}: {year}/seed={seed} pid={process.pid}", flush=True)
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
                    # 遍历 `running`，并把当前元素绑定到 `active_item` 逐项处理。
                    for active_item in running:
                        # 检查 `active_item['process'].poll() is None`；根据条件结果选择对应处理分支。
                        if active_item["process"].poll() is None:
                            # 调用 `active_item['process'].terminate` 终止不再继续等待的外部任务。
                            active_item["process"].terminate()
                    # 抛出 `RuntimeError(f'A2 任务失败：{item['year']}/seed={item['seed']} GPU{item['gpu']} exit={return_code}\n…`，阻止不满足数据契约的流程继续执行。
                    raise RuntimeError(
                        f"A2 任务失败：{item['year']}/seed={item['seed']} GPU{item['gpu']} "
                        f"exit={return_code}\n{tail(item['log_path'])}"
                    )
                # 调用 `print` 记录当前流程状态或诊断信息。
                print(
                    f"完成 GPU{item['gpu']}: {item['year']}/seed={item['seed']} "
                    f"{(time.perf_counter() - item['started']) / 60:.2f} 分钟",
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
    # 计算 `time.perf_counter() - started`，并将数值结果保存到 `elapsed`。
    elapsed = time.perf_counter() - started
    # 调用 `finalize_outputs` 汇总运行结果并写出完成标记与最终报告，并将结果保存到 `summary`。
    summary = finalize_outputs(args, elapsed)
    # 调用 `print` 记录当前流程状态或诊断信息。
    print(f"A2 完成：{len(summary)} 个任务，墙钟 {elapsed / 60:.2f} 分钟")
    # 将 `summary` 作为当前函数结果返回给调用方。
    return summary


# 定义 `run_preflight_worker` 函数，封装这一阶段可独立调用的处理步骤。
def run_preflight_worker(args: Any) -> None:
    """在一个独立进程内执行多 shape 显存压力测试。

    输入：
        args: Any。
    输出：
        None。
    """

    # 调用 `configure_worker_environment` 整理并返回当前实验的配置对象。
    configure_worker_environment(args)
    # 调用 `str` 把当前值转换为统一字符串表示，并将结果保存到 `os.environ['VIT_PREFLIGHT_RANK']`。
    os.environ["VIT_PREFLIGHT_RANK"] = str(args.preflight_rank)
    # 调用 `initialize_runtime` 按配置初始化设备与运行时状态。
    initialize_runtime()
    # 调用 `set_random_seed` 同时固定 Python、NumPy 和 PyTorch 随机状态。
    set_random_seed(
        TRAINING_CONFIG.seeds[args.preflight_rank % len(TRAINING_CONFIG.seeds)]
    )
    # 调用 `run_full_batch_preflight` 按当前配置执行对应的完整子流程，并将结果保存到 `result`。
    result = run_full_batch_preflight()
    # 检查 `result is None`；根据条件结果选择对应处理分支。
    if result is None:
        # 抛出 `RuntimeError('并发显存预检要求 CUDA')`，阻止不满足数据契约的流程继续执行。
        raise RuntimeError("并发显存预检要求 CUDA")
    # 调用 `result.insert` 把当前字段插入指定位置并保持输出列顺序。
    result.insert(0, "concurrent_worker_rank", args.preflight_rank)
    # 调用 `print` 记录当前流程状态或诊断信息。
    print(
        "VIT_PREFLIGHT_RESULT="
        + json.dumps(result.to_dict("records"), ensure_ascii=False),
        flush=True,
    )


# 定义 `run_concurrent_temporal_preflight` 函数，封装这一阶段可独立调用的处理步骤。
def run_concurrent_temporal_preflight(args: Any) -> pd.DataFrame:
    """在同一物理 GPU 同时启动与正式训练相同数量的 worker。

    输入：
        args: Any。
    输出：
        pd.DataFrame。
    """

    # 读取 `parse_csv_ints(args.device_ids)[0]` 的当前值，并赋给 `physical_gpu`。
    physical_gpu = parse_csv_ints(args.device_ids)[0]
    # 调用 `int` 统一设备或数据类型，并将结果保存到 `worker_count`。
    worker_count = int(args.tasks_per_gpu)
    # 检查 `worker_count < 2`；根据条件结果选择对应处理分支。
    if worker_count < 2:
        # 抛出 `ValueError('同卡并发预检至少需要两个 worker')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("同卡并发预检至少需要两个 worker")
    # 构造当前步骤所需的列表，并保存到 `running`。
    running: list[dict[str, object]] = []
    # 构造当前步骤所需的列表，并保存到 `records`。
    records: list[dict[str, object]] = []
    # 在 `tempfile.TemporaryDirectory(prefix='vit_temporal_pref…` 管理的上下文中执行本段逻辑，并保证资源按约定释放。
    with tempfile.TemporaryDirectory(prefix="vit_temporal_preflight_") as temporary:
        # 调用 `Path` 规范化文件路径，并将结果保存到 `temporary_dir`。
        temporary_dir = Path(temporary)
        # 计算 `temporary_dir / 'barrier'`，并将数值结果保存到 `barrier_dir`。
        barrier_dir = temporary_dir / "barrier"
        # 尝试执行本段主逻辑；若发生异常，则交由后续异常分支处理。
        try:
            # 遍历 `range(worker_count)`，并把当前元素绑定到 `rank` 逐项处理。
            for rank in range(worker_count):
                # 计算 `temporary_dir / f'worker_{rank}.log'`，并将数值结果保存到 `log_path`。
                log_path = temporary_dir / f"worker_{rank}.log"
                # 调用 `log_path.open` 读取源数据，并将结果保存到 `log_handle`。
                log_handle = log_path.open("w", encoding="utf-8")
                # 复制 `os.environ.copy`，避免后续更新污染原对象，并保存到 `environment`。
                environment = os.environ.copy()
                # 调用 `str` 把当前值转换为统一字符串表示，并将结果保存到 `environment['CUDA_VISIBLE_DEVICES']`。
                environment["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
                # 调用 `str` 把当前值转换为统一字符串表示，并将结果保存到 `environment['VIT_PREFLIGHT_BARRIER_DIR']`。
                environment["VIT_PREFLIGHT_BARRIER_DIR"] = str(barrier_dir)
                # 调用 `str` 把当前值转换为统一字符串表示，并将结果保存到 `environment['VIT_PREFLIGHT_WORLD_SIZE']`。
                environment["VIT_PREFLIGHT_WORLD_SIZE"] = str(worker_count)
                # 调用 `subprocess.Popen` 启动独立训练子进程并保留进程句柄，并将结果保存到 `process`。
                process = subprocess.Popen(
                    preflight_worker_command(args, rank),
                    cwd=VIT_DIR,
                    env=environment,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                # 把本轮结果追加到 `running`，累积后续需要汇总的记录。
                running.append(
                    {
                        "rank": rank,
                        "process": process,
                        "log_path": log_path,
                        "log_handle": log_handle,
                    }
                )
            # 计算 `time.monotonic() + 1200`，并将数值结果保存到 `deadline`。
            deadline = time.monotonic() + 1200
            # 只要 `running` 成立就继续循环，直至满足退出条件。
            while running:
                # 检查 `time.monotonic() >= deadline`；根据条件结果选择对应处理分支。
                if time.monotonic() >= deadline:
                    # 抛出 `TimeoutError('同卡并发显存预检超过20分钟')`，阻止不满足数据契约的流程继续执行。
                    raise TimeoutError("同卡并发显存预检超过20分钟")
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
                    # 调用 `item['log_path'].read_text` 读取源数据，并将结果保存到 `log_text`。
                    log_text = item["log_path"].read_text(
                        encoding="utf-8", errors="replace"
                    )
                    # 检查 `return_code`；根据条件结果选择对应处理分支。
                    if return_code:
                        # 抛出 `RuntimeError(f'并发预检 worker={item['rank']} 失败：\n' + '\n'.join(log_text.splitlines()[-100:]))`，阻止不满足数据契约的流程继续执行。
                        raise RuntimeError(
                            f"并发预检 worker={item['rank']} 失败：\n"
                            + "\n".join(log_text.splitlines()[-100:])
                        )
                    # 构造当前步骤所需的列表，并保存到 `marker_lines`。
                    marker_lines = [
                        line for line in log_text.splitlines()
                        if line.startswith("VIT_PREFLIGHT_RESULT=")
                    ]
                    # 检查 `len(marker_lines) != 1`；根据条件结果选择对应处理分支。
                    if len(marker_lines) != 1:
                        # 抛出 `RuntimeError(f'并发预检 worker={item['rank']} 缺少结果标记')`，阻止不满足数据契约的流程继续执行。
                        raise RuntimeError(
                            f"并发预检 worker={item['rank']} 缺少结果标记"
                        )
                    # 读取 `marker_lines[0].split('=', 1)[1]` 的当前值，并赋给 `payload`。
                    payload = marker_lines[0].split("=", 1)[1]
                    # 调用 `json.loads` 读取源数据，并将结果保存到 `worker_records`。
                    worker_records = json.loads(payload)
                    # 遍历 `worker_records`，并把当前元素绑定到 `record` 逐项处理。
                    for record in worker_records:
                        # 读取 `physical_gpu` 的当前值，并赋给 `record['physical_gpu']`。
                        record["physical_gpu"] = physical_gpu
                        # 读取 `worker_count` 的当前值，并赋给 `record['concurrent_tasks_on_gpu']`。
                        record["concurrent_tasks_on_gpu"] = worker_count
                    # 把一组结果展开追加到 `records`，保持统一的一维记录序列。
                    records.extend(worker_records)
                # 遍历 `finished`，并把当前元素绑定到 `item` 逐项处理。
                for item in finished:
                    # 调用 `running.remove` 从运行队列移除已经结束的任务记录。
                    running.remove(item)
                # 检查 `running and (not finished)`；根据条件结果选择对应处理分支。
                if running and not finished:
                    # 调用 `time.sleep` 短暂等待外部任务状态更新，避免无间隔轮询。
                    time.sleep(0.5)
        # 无论前序步骤是否抛出异常，都执行本分支完成必要的收尾。
        finally:
            # 遍历 `running`，并把当前元素绑定到 `item` 逐项处理。
            for item in running:
                # 检查 `item['process'].poll() is None`；根据条件结果选择对应处理分支。
                if item["process"].poll() is None:
                    # 调用 `item['process'].terminate` 向仍在运行的子进程发送终止信号。
                    item["process"].terminate()
                # 调用 `item['log_handle'].close` 释放当前步骤占用的资源。
                item["log_handle"].close()
    # 调用 `pd.DataFrame(records).sort_values` 根据分数计算截面排序或排序索引，并将结果保存到 `result`。
    result = pd.DataFrame(records).sort_values(
        ["concurrent_worker_rank", "shape_test"]
    )
    # 检查 `len(result) != worker_count * 12`；根据条件结果选择对应处理分支。
    if len(result) != worker_count * 12:
        # 抛出 `AssertionError(f'并发预检结果不完整：{len(result)}/{worker_count * 12}')`，阻止不满足数据契约的流程继续执行。
        raise AssertionError(
            f"并发预检结果不完整：{len(result)}/{worker_count * 12}"
        )
    # 将 `result.reset_index(drop=True)` 作为当前函数结果返回给调用方。
    return result.reset_index(drop=True)


# 定义 `run_preflight` 函数，封装这一阶段可独立调用的处理步骤。
def run_preflight(args: Any) -> None:
    """执行数据、切分、模型和训练参数的只读预检。

    输入：
        args: Any。
    输出：
        None。
    """
    # 调用 `configure_worker_environment` 整理并返回当前实验的配置对象。
    configure_worker_environment(args)
    # 调用 `initialize_runtime` 按配置初始化设备与运行时状态。
    initialize_runtime()
    # 调用 `validate_training_inputs` 校验输入、配置或产物是否满足当前阶段的数据契约。
    validate_training_inputs()
    # 调用 `display` 在 Notebook 输出区展示当前表格或图形。
    display(
        pd.DataFrame(
            [validate_rolling_split(year) for year in TRAINING_CONFIG.forecast_years]
        )
    )
    # 调用 `display` 在 Notebook 输出区展示当前表格或图形。
    display(configured_batch_summaries())
    # 调用 `display` 在 Notebook 输出区展示当前表格或图形。
    display(run_smoke_test())
    # 检查 `TRAINING_CONFIG.loss_name == TEMPORAL_TURNOVER_LOSS and args.tasks_per_gpu > 1`；根据条件结果选择对应处理分支。
    if (
        TRAINING_CONFIG.loss_name == TEMPORAL_TURNOVER_LOSS
        and args.tasks_per_gpu > 1
    ):
        # 调用 `display` 在 Notebook 输出区展示当前表格或图形。
        display(run_concurrent_temporal_preflight(args))
    # 当前置条件不成立时进入备用分支，保证其余合法情况仍被处理。
    else:
        # 调用 `run_full_batch_preflight` 按当前配置执行对应的完整子流程，并将结果保存到 `result`。
        result = run_full_batch_preflight()
        # 检查 `result is not None`；根据条件结果选择对应处理分支。
        if result is not None:
            # 调用 `display` 在 Notebook 输出区展示当前表格或图形。
            display(result)


# ============================== 训练 CLI ==============================
def validate_training_inputs() -> pd.DataFrame:
    """检查 memmap、label_valid 和 8 个滚动年度。

    输入：
        无显式输入参数。
    输出：
        pd.DataFrame。
    """

    # 调用 `np.load` 读取源数据，并将结果保存到 `features`。
    features = np.load(FEATURES_PATH, mmap_mode="r")
    # 调用 `np.load` 读取源数据，并将结果保存到 `targets`。
    targets = np.load(TARGETS_PATH, mmap_mode="r")
    # 调用 `np.load` 读取源数据，并将结果保存到 `label_valid`。
    label_valid = np.load(LABEL_VALID_PATH, mmap_mode="r")
    # 调用 `np.load` 读取源数据，并将结果保存到 `date_ids`。
    date_ids = np.load(DATE_IDS_PATH, mmap_mode="r")
    # 检查 `features.shape[1:] != (17, 8, 8)`；根据条件结果选择对应处理分支。
    if features.shape[1:] != (17, 8, 8):
        # 抛出 `ValueError(f'特征形状异常：{features.shape}')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(f"特征形状异常：{features.shape}")
    # 检查 `not len(features) == len(targets) == len(label_valid) == len(date_ids) == len(sample_index)`；根据条件结果选择对应处理分支。
    if not (len(features) == len(targets) == len(label_valid) == len(date_ids) == len(sample_index)):
        # 抛出 `ValueError('feature store 各文件行数不一致')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("feature store 各文件行数不一致")
    # 检查 `not np.array_equal(np.isfinite(targets), label_valid.astype(bool))`；根据条件结果选择对应处理分支。
    if not np.array_equal(np.isfinite(targets), label_valid.astype(bool)):
        # 抛出 `AssertionError('label_valid 与 target NaN 契约不一致')`，阻止不满足数据契约的流程继续执行。
        raise AssertionError("label_valid 与 target NaN 契约不一致")
    # 调用 `pd.read_parquet` 读取源数据，并将结果保存到 `rolling`。
    rolling = pd.read_parquet(ROLLING_SUMMARY_PATH)
    # 计算 `set(TRAINING_CONFIG.forecast_years) - set(rolling['forecast_year'].astype(int))`，并将数值结果保存到 `missing_years`。
    missing_years = set(TRAINING_CONFIG.forecast_years) - set(
        rolling["forecast_year"].astype(int)
    )
    # 检查 `missing_years`；根据条件结果选择对应处理分支。
    if missing_years:
        # 抛出 `ValueError(f'缺少滚动年度：{sorted(missing_years)}')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(f"缺少滚动年度：{sorted(missing_years)}")
    # 检查 `DEVICE.type == 'cuda'`；根据条件结果选择对应处理分支。
    if DEVICE.type == "cuda":
        # 调用 `torch.cuda.get_device_capability` 读取 GPU 计算能力版本，并将结果保存到 `capability`。
        capability = torch.cuda.get_device_capability(0)
        # 检查 `capability == (12, 0) and 'sm_120' not in torch.cuda.get_arch_list()`；根据条件结果选择对应处理分支。
        if capability == (12, 0) and "sm_120" not in torch.cuda.get_arch_list():
            # 抛出 `RuntimeError('当前 PyTorch wheel 未包含 RTX 5090 的 sm_120')`，阻止不满足数据契约的流程继续执行。
            raise RuntimeError("当前 PyTorch wheel 未包含 RTX 5090 的 sm_120")
        # 调用 `torch.cuda.get_device_properties` 读取 GPU 型号与显存等硬件属性，并将结果保存到 `properties`。
        properties = torch.cuda.get_device_properties(0)
        # 调用 `print` 记录当前流程状态或诊断信息。
        print(
            f"GPU：{properties.name}，显存 {properties.total_memory / 1024**3:.1f}GB，"
            f"计算能力 sm_{capability[0]}{capability[1]}"
        )
    # 调用 `pd.DataFrame` 转换数据表示，并将标准化对象保存到 `summary`。
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
    # 调用 `display` 在 Notebook 输出区展示当前表格或图形。
    display(summary)
    # 调用 `display` 在 Notebook 输出区展示当前表格或图形。
    display(rolling)
    # 删除 `features, targets, label_valid, date_ids`，释放不再使用的引用。
    del features, targets, label_valid, date_ids
    # 将 `summary` 作为当前函数结果返回给调用方。
    return summary

# 定义 `loader_batch_summary` 函数，封装这一阶段可独立调用的处理步骤。
def loader_batch_summary(loader: DataLoader) -> dict[str, object]:
    """汇总 DataLoader 的批次数、样本数和批大小分布。

    输入：
        loader: DataLoader。
    输出：
        dict[str, object]。
    """
    # 读取 `loader.batch_sampler` 的当前值，并赋给 `sampler`。
    sampler = loader.batch_sampler
    # 检查 `isinstance(sampler, TradeDateBatchSampler)`；根据条件结果选择对应处理分支。
    if isinstance(sampler, TradeDateBatchSampler):
        # 将 `sampler.summary()` 作为当前函数结果返回给调用方。
        return sampler.summary()
    # 调用 `len` 汇总当前数据，并将统计量保存到 `sample_count`。
    sample_count = len(loader.dataset)
    # 调用 `int` 统一设备或数据类型，并将结果保存到 `batch_size`。
    batch_size = int(loader.batch_size)
    # 计算 `sample_count // batch_size`，并将数值结果保存到 `batch_count`。
    batch_count = sample_count // batch_size
    # 将 `{'train_batch_mode': 'fixed_samples', 'train_batch_count': batch_count, 'train_date_count': Non…` 作为当前函数结果返回给调用方。
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

# 定义 `validate_rolling_split` 函数，封装这一阶段可独立调用的处理步骤。
def validate_rolling_split(year: int) -> dict[str, object]:
    """检查训练、验证与预测切分的行号、日期和防泄漏约束。

    输入：
        year: int。
    输出：
        dict[str, object]。
    """
    # 调用 `load_split` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `split`。
    split = load_split(year)
    # 读取 `sample_index.iloc[split['train_indices']]` 的当前值，并赋给 `train_rows`。
    train_rows = sample_index.iloc[split["train_indices"]]
    # 读取 `sample_index.iloc[split['validation_indices']]` 的当前值，并赋给 `validation_rows`。
    validation_rows = sample_index.iloc[split["validation_indices"]]
    # 读取 `sample_index.iloc[split['prediction_indices']]` 的当前值，并赋给 `prediction_rows`。
    prediction_rows = sample_index.iloc[split["prediction_indices"]]
    # 检查 `not train_rows['label_valid'].all() or not validation_rows['label_valid'].all()`；根据条件结果选择对应处理分支。
    if not train_rows["label_valid"].all() or not validation_rows["label_valid"].all():
        # 抛出 `AssertionError(f'{year} 训练或验证索引混入无标签样本')`，阻止不满足数据契约的流程继续执行。
        raise AssertionError(f"{year} 训练或验证索引混入无标签样本")
    # 调用 `validation_rows['trade_date'].min` 汇总当前数据，并将统计量保存到 `validation_start`。
    validation_start = validation_rows["trade_date"].min()
    # 计算 `f'{year}0101'`，并将结果写入 `forecast_start`。
    forecast_start = f"{year}0101"
    # 检查 `train_rows['exit_trade_date'].max() >= validation_start`；根据条件结果选择对应处理分支。
    if train_rows["exit_trade_date"].max() >= validation_start:
        # 抛出 `AssertionError(f'{year} 训练标签越过验证边界')`，阻止不满足数据契约的流程继续执行。
        raise AssertionError(f"{year} 训练标签越过验证边界")
    # 检查 `validation_rows['exit_trade_date'].max() >= forecast_start`；根据条件结果选择对应处理分支。
    if validation_rows["exit_trade_date"].max() >= forecast_start:
        # 抛出 `AssertionError(f'{year} 验证标签越过预测边界')`，阻止不满足数据契约的流程继续执行。
        raise AssertionError(f"{year} 验证标签越过预测边界")
    # 检查 `len(prediction_rows) and (not prediction_rows['trade_date'].str.startswith(str(year)).all())`；根据条件结果选择对应处理分支。
    if len(prediction_rows) and not prediction_rows["trade_date"].str.startswith(str(year)).all():
        # 抛出 `AssertionError(f'{year} 预测索引混入其他年度')`，阻止不满足数据契约的流程继续执行。
        raise AssertionError(f"{year} 预测索引混入其他年度")
    # 将 `{'forecast_year': year, 'train_samples': len(train_rows), 'validation_samples': len(validation_…` 作为当前函数结果返回给调用方。
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

# 定义 `configured_batch_summaries` 函数，封装这一阶段可独立调用的处理步骤。
def configured_batch_summaries() -> pd.DataFrame:
    """按当前批处理配置生成各切分的批次摘要。

    输入：
        无显式输入参数。
    输出：
        pd.DataFrame。
    """
    # 构造当前步骤所需的列表，并保存到 `records`。
    records = []
    # 读取 `TRAINING_CONFIG.seeds[0]` 的当前值，并赋给 `seed`。
    seed = TRAINING_CONFIG.seeds[0]
    # 遍历 `TRAINING_CONFIG.forecast_years`，并把当前元素绑定到 `year` 逐项处理。
    for year in TRAINING_CONFIG.forecast_years:
        # 调用 `make_loaders` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `(train_loader, _, _)`。
        train_loader, _, _ = make_loaders(year, seed, workers=0)
        # 把本轮结果追加到 `records`，累积后续需要汇总的记录。
        records.append({"forecast_year": year, **loader_batch_summary(train_loader)})
    # 将 `pd.DataFrame(records)` 作为当前函数结果返回给调用方。
    return pd.DataFrame(records)

# 定义 `validate_outputs` 函数，封装这一阶段可独立调用的处理步骤。
def validate_outputs(args: Any) -> pd.DataFrame:
    """检查 checkpoint、预测、诊断和 manifest 是否完整一致。

    输入：
        args: Any。
    输出：
        pd.DataFrame。
    """
    # 调用 `Path(args.data_root).resolve` 规范化文件路径，并将结果保存到 `data_root`。
    data_root = Path(args.data_root).resolve()
    # 组织一组有序值，并保存到 `(years, seeds)`。
    years, seeds = parse_csv_ints(args.years), parse_csv_ints(args.seeds)
    # 计算 `VIT_DIR / 'models' / 'runs' / args.run_name`，并将数值结果保存到 `model_root`。
    model_root = VIT_DIR / "models" / "runs" / args.run_name
    # 计算 `data_root / 'predictions' / 'runs' / args.run_name`，并将数值结果保存到 `prediction_dir`。
    prediction_dir = data_root / "predictions" / "runs" / args.run_name
    # 计算 `data_root / 'quality' / 'runs' / args.run_name`，并将数值结果保存到 `quality_dir`。
    quality_dir = data_root / "quality" / "runs" / args.run_name
    # 调用 `pd.read_parquet` 读取源数据，并将结果保存到 `sample_index_frame`。
    sample_index_frame = pd.read_parquet(
        data_root / "feature_store" / "sample_index.parquet",
        columns=["sample_index", "trade_date"],
    )
    # 调用 `sample_index_frame['trade_date'].astype` 统一设备或数据类型，并将结果保存到 `sample_index_frame['trade_date']`。
    sample_index_frame["trade_date"] = sample_index_frame["trade_date"].astype(str)
    # 调用 `prediction_counts` 执行模型推理并生成样本预测分数，并将结果保存到 `counts`。
    counts = prediction_counts(data_root, years)
    # 构造当前步骤所需的列表，并保存到 `records`。
    records = []
    # 遍历 `years`，并把当前元素绑定到 `year` 逐项处理。
    for year in years:
        # 遍历 `seeds`，并把当前元素绑定到 `seed` 逐项处理。
        for seed in seeds:
            # 计算 `model_root / f'forecast_{year}' / f'seed_{seed}'`，并将数值结果保存到 `task_dir`。
            task_dir = model_root / f"forecast_{year}" / f"seed_{seed}"
            # 检查 `not (task_dir / 'best.pt').exists() or not (task_dir / 'completed.json').exists()`；根据条件结果选择对应处理分支。
            if not (task_dir / "best.pt").exists() or not (task_dir / "completed.json").exists():
                # 抛出 `FileNotFoundError(f'{year}/seed={seed} checkpoint 不完整')`，阻止不满足数据契约的流程继续执行。
                raise FileNotFoundError(f"{year}/seed={seed} checkpoint 不完整")
        # 计算 `prediction_dir / f'vit_predictions.{year}.parquet'`，并将数值结果保存到 `prediction_path`。
        prediction_path = prediction_dir / f"vit_predictions.{year}.parquet"
        # 检查 `counts[year]`；根据条件结果选择对应处理分支。
        if counts[year]:
            # 调用 `pd.read_parquet` 读取源数据，并将结果保存到 `prediction`。
            prediction = pd.read_parquet(prediction_path, columns=["sample_index"])
            # 调用 `to_numpy` 把表格列转换为指定精度的 NumPy 数组，并将结果保存到 `expected`。
            expected = sample_index_frame.loc[
                sample_index_frame["trade_date"].str.startswith(str(year)), "sample_index"
            ].to_numpy(np.int64)
            # 检查 `not np.array_equal(prediction['sample_index'].to_numpy(np.int64), expected)`；根据条件结果选择对应处理分支。
            if not np.array_equal(prediction["sample_index"].to_numpy(np.int64), expected):
                # 抛出 `AssertionError(f'{year} 预测键与 T 日 Universe 不一致')`，阻止不满足数据契约的流程继续执行。
                raise AssertionError(f"{year} 预测键与 T 日 Universe 不一致")
        # 检查 `prediction_path.exists()`；根据条件结果选择对应处理分支。
        elif prediction_path.exists():
            # 抛出 `AssertionError(f'{year} 无图片却出现预测文件')`，阻止不满足数据契约的流程继续执行。
            raise AssertionError(f"{year} 无图片却出现预测文件")
        # 把本轮结果追加到 `records`，累积后续需要汇总的记录。
        records.append(
            {
                "forecast_year": year,
                "checkpoints": len(seeds),
                "prediction_samples": counts[year],
                "prediction_file": prediction_path.exists(),
                "status": "模型与预测完成" if counts[year] else "模型完成，暂无图片预测",
            }
        )
    # 计算 `quality_dir / 'vit_training_summary.parquet'`，并将数值结果保存到 `summary_path`。
    summary_path = quality_dir / "vit_training_summary.parquet"
    # 计算 `quality_dir / 'a2_training_manifest.json'`，并将数值结果保存到 `manifest_path`。
    manifest_path = quality_dir / "a2_training_manifest.json"
    # 检查 `not summary_path.exists() or not manifest_path.exists()`；根据条件结果选择对应处理分支。
    if not summary_path.exists() or not manifest_path.exists():
        # 抛出 `FileNotFoundError('A2 训练汇总或 manifest 缺失')`，阻止不满足数据契约的流程继续执行。
        raise FileNotFoundError("A2 训练汇总或 manifest 缺失")
    # 调用 `pd.read_parquet` 读取源数据，并将结果保存到 `summary`。
    summary = pd.read_parquet(summary_path)
    # 检查 `len(summary[['forecast_year', 'seed']].drop_duplicates()) != len(years) * len(seeds)`；根据条件结果选择对应处理分支。
    if len(summary[["forecast_year", "seed"]].drop_duplicates()) != len(years) * len(seeds):
        # 抛出 `AssertionError('A2 年度×种子任务数不完整')`，阻止不满足数据契约的流程继续执行。
        raise AssertionError("A2 年度×种子任务数不完整")
    # 调用 `pd.DataFrame` 转换数据表示，并将标准化对象保存到 `result`。
    result = pd.DataFrame(records)
    # 调用 `display` 在 Notebook 输出区展示当前表格或图形。
    display(result)
    # 将 `result` 作为当前函数结果返回给调用方。
    return result

# 定义 `show_status` 函数，封装这一阶段可独立调用的处理步骤。
def show_status(args: Any) -> pd.DataFrame:
    """显示年度任务、worker、checkpoint 和预测的当前状态。

    输入：
        args: Any。
    输出：
        pd.DataFrame。
    """
    # 组织一组有序值，并保存到 `(years, seeds)`。
    years, seeds = parse_csv_ints(args.years), parse_csv_ints(args.seeds)
    # 计算 `VIT_DIR / 'models' / 'runs' / args.run_name`，并将数值结果保存到 `model_root`。
    model_root = VIT_DIR / "models" / "runs" / args.run_name
    # 构造当前步骤所需的列表，并保存到 `records`。
    records = []
    # 遍历 `years`，并把当前元素绑定到 `year` 逐项处理。
    for year in years:
        # 遍历 `seeds`，并把当前元素绑定到 `seed` 逐项处理。
        for seed in seeds:
            # 计算 `model_root / f'forecast_{year}' / f'seed_{seed}'`，并将数值结果保存到 `task_dir`。
            task_dir = model_root / f"forecast_{year}" / f"seed_{seed}"
            # 调用 `read_json` 读取源数据，并将结果保存到 `completed`。
            completed = read_json(task_dir / "completed.json")
            # 计算 `task_dir / 'history.parquet'`，并将数值结果保存到 `history_path`。
            history_path = task_dir / "history.parquet"
            # 组织一组有序值，并保存到 `(last_epoch, best_ic)`。
            last_epoch, best_ic = None, None
            # 检查 `completed`；根据条件结果选择对应处理分支。
            if completed:
                # 调用 `completed['summary'].get` 按键读取映射中的可选字段，并在缺失时使用约定默认值，并将结果保存到 `last_epoch`。
                last_epoch = completed["summary"].get("last_epoch")
                # 调用 `completed['summary'].get` 按键读取映射中的可选字段，并在缺失时使用约定默认值，并将结果保存到 `best_ic`。
                best_ic = completed["summary"].get("best_validation_ic")
            # 检查 `history_path.exists()`；根据条件结果选择对应处理分支。
            elif history_path.exists():
                # 调用 `pd.read_parquet` 读取源数据，并将结果保存到 `history`。
                history = pd.read_parquet(history_path)
                # 检查 `len(history)`；根据条件结果选择对应处理分支。
                if len(history):
                    # 调用 `int` 统一设备或数据类型，并将结果保存到 `last_epoch`。
                    last_epoch = int(history["epoch"].iloc[-1])
                    # 调用 `float` 统一设备或数据类型，并将结果保存到 `best_ic`。
                    best_ic = float(history["best_validation_ic"].iloc[-1])
            # 把本轮结果追加到 `records`，累积后续需要汇总的记录。
            records.append(
                {"forecast_year": year, "seed": seed, "completed": bool(completed),
                 "last_epoch": last_epoch, "best_validation_ic": best_ic}
            )
    # 调用 `pd.DataFrame` 转换数据表示，并将标准化对象保存到 `result`。
    result = pd.DataFrame(records)
    # 调用 `print` 记录当前流程状态或诊断信息。
    print(f"完成任务：{int(result['completed'].sum())}/{len(result)}")
    # 调用 `display` 在 Notebook 输出区展示当前表格或图形。
    display(result)
    # 将 `result` 作为当前函数结果返回给调用方。
    return result

# 定义 `validate_a2_args` 函数，封装这一阶段可独立调用的处理步骤。
def validate_a2_args(args: argparse.Namespace) -> None:
    """校验 A2 训练任务参数之间的组合约束。

    输入：
        args: argparse.Namespace。
    输出：
        None。
    """
    # 调用 `validate_run_name` 校验输入、配置或产物是否满足当前阶段的数据契约，并将结果保存到 `args.run_name`。
    args.run_name = validate_run_name(args.run_name)
    # 检查 `args.tasks_per_gpu < 1 or args.batch_size <= 1 or args.inference_batch_size <= 1 or (args.worke…`；根据条件结果选择对应处理分支。
    if (
        args.tasks_per_gpu < 1
        or args.batch_size <= 1
        or args.inference_batch_size <= 1
        or args.workers_per_task < 0
    ):
        # 抛出 `ValueError('A2 并发、batch 或 workers 参数非法')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("A2 并发、batch 或 workers 参数非法")
    # 检查 `args.learning_rate <= 0 or args.max_epochs < 1 or args.patience < 1`；根据条件结果选择对应处理分支。
    if args.learning_rate <= 0 or args.max_epochs < 1 or args.patience < 1:
        # 抛出 `ValueError('学习率、max_epochs 和 patience 必须为正')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("学习率、max_epochs 和 patience 必须为正")
    # 检查 `args.train_batch_mode not in TRAIN_BATCH_MODES`；根据条件结果选择对应处理分支。
    if args.train_batch_mode not in TRAIN_BATCH_MODES:
        # 抛出 `ValueError(f'未知训练 batch 模式：{args.train_batch_mode}')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(f"未知训练 batch 模式：{args.train_batch_mode}")
    # 检查 `args.loss_name not in LOSS_NAMES`；根据条件结果选择对应处理分支。
    if args.loss_name not in LOSS_NAMES:
        # 抛出 `ValueError(f'未知损失函数：{args.loss_name}')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(f"未知损失函数：{args.loss_name}")
    # 检查 `args.validation_metric not in VALIDATION_METRICS`；根据条件结果选择对应处理分支。
    if args.validation_metric not in VALIDATION_METRICS:
        # 抛出 `ValueError(f'未知验证指标：{args.validation_metric}')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(f"未知验证指标：{args.validation_metric}")
    # 检查 `args.rank_temperature <= 0 or args.turnover_lambda < 0 or args.huber_delta <= 0 or (args.smooth…`；根据条件结果选择对应处理分支。
    if (
        args.rank_temperature <= 0
        or args.turnover_lambda < 0
        or args.huber_delta <= 0
        or args.smooth_l1_beta <= 0
        or not 0 <= args.hybrid_weight <= 1
    ):
        # 抛出 `ValueError('损失温度、换手权重、delta、beta 或混合权重非法')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("损失温度、换手权重、delta、beta 或混合权重非法")
    # 检查 `args.loss_name == TEMPORAL_TURNOVER_LOSS and args.train_batch_mode != 'trade_date'`；根据条件结果选择对应处理分支。
    if (
        args.loss_name == TEMPORAL_TURNOVER_LOSS
        and args.train_batch_mode != "trade_date"
    ):
        # 抛出 `ValueError('sigmoid_rank_ic_turnover 必须使用 trade_date batch')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(
            "sigmoid_rank_ic_turnover 必须使用 trade_date batch"
        )
    # 检查 `args.loss_name == TEMPORAL_TURNOVER_LOSS and bool(args.compile) and (args.compile_mode == 'redu…`；根据条件结果选择对应处理分支。
    if (
        args.loss_name == TEMPORAL_TURNOVER_LOSS
        and bool(args.compile)
        and args.compile_mode == "reduce-overhead"
    ):
        # 抛出 `ValueError('时序换手损失的动态日截面禁止使用 reduce-overhead；该模式会启用 CUDAGraph 私有显存池，请使用 compile-mode=default')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(
            "时序换手损失的动态日截面禁止使用 reduce-overhead；"
            "该模式会启用 CUDAGraph 私有显存池，请使用 compile-mode=default"
        )
    # 检查 `args.command in {'preflight', 'preflight-worker', 'a2', 'worker'} and args.run_name in READ_ONL…`；根据条件结果选择对应处理分支。
    if (
        args.command in {"preflight", "preflight-worker", "a2", "worker"}
        and args.run_name in READ_ONLY_TRAINING_RUN_NAMES
    ):
        # 抛出 `ValueError(f'{args.run_name} 是只读历史实验，请为调参训练设置新的 run name')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(
            f"{args.run_name} 是只读历史实验，请为调参训练设置新的 run name"
        )
    # 调用 `parse_csv_ints` 解析用户配置并转换为内部统一口径，并将结果保存到 `device_ids`。
    device_ids = parse_csv_ints(args.device_ids)
    # 检查 `device_ids != (0, 1, 2, 3)`；根据条件结果选择对应处理分支。
    if device_ids != (0, 1, 2, 3):
        # 抛出 `ValueError('正式 A2 固定使用 GPU 0,1,2,3')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("正式 A2 固定使用 GPU 0,1,2,3")
    # 调用 `parse_csv_ints` 解析用户配置并转换为内部统一口径。
    parse_csv_ints(args.years)
    # 调用 `parse_csv_ints` 解析用户配置并转换为内部统一口径。
    parse_csv_ints(args.seeds)
    # 检查 `not Path(args.data_root).resolve().is_dir()`；根据条件结果选择对应处理分支。
    if not Path(args.data_root).resolve().is_dir():
        # 抛出 `FileNotFoundError(args.data_root)`，阻止不满足数据契约的流程继续执行。
        raise FileNotFoundError(args.data_root)
    # 检查 `args.command == 'preflight-worker' and args.preflight_rank < 0`；根据条件结果选择对应处理分支。
    if args.command == "preflight-worker" and args.preflight_rank < 0:
        # 抛出 `ValueError('preflight-rank 必须为非负整数')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("preflight-rank 必须为非负整数")


# 定义 `add_a2_arguments` 函数，封装这一阶段可独立调用的处理步骤。
def add_a2_arguments(parser: argparse.ArgumentParser, worker: bool = False) -> None:
    """向命令行解析器注册 A2 训练参数。

    输入：
        parser: argparse.ArgumentParser。
        worker: bool。
    输出：
        None。
    """
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument("--data-root", default=str(VIT_DIR / "data"))
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument("--run-name", default=DEFAULT_TUNING_RUN_NAME)
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument("--device-ids", default="0,1,2,3")
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument("--tasks-per-gpu", type=int, default=2)
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument("--years", default=DEFAULT_YEARS)
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
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
        default="daily_mean_spearman_ic",
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
    # 检查 `worker`；根据条件结果选择对应处理分支。
    if worker:
        # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
        parser.add_argument("--worker-year", type=int, required=True)
        # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
        parser.add_argument("--worker-seed", type=int, required=True)
        # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
        parser.add_argument("--worker-physical-gpu", type=int, required=True)


# 定义 `parse_args` 函数，封装这一阶段可独立调用的处理步骤。
def parse_args() -> argparse.Namespace:
    """解析阶段 07 的训练、预检、验收和状态命令。

    输入：
        无显式输入参数。
    输出：
        argparse.Namespace。
    """
    # 调用 `argparse.ArgumentParser` 创建命令行参数解析器，并将结果保存到 `parser`。
    parser = argparse.ArgumentParser(description=__doc__)
    # 调用 `parser.add_subparsers` 创建子命令分发器以区分不同运行模式，并将结果保存到 `commands`。
    commands = parser.add_subparsers(dest="command", required=True)
    # 遍历 `('preflight', 'a2', 'validate', 'status')`，并把当前元素绑定到 `name` 逐项处理。
    for name in ("preflight", "a2", "validate", "status"):
        # 调用 `add_a2_arguments` 向命令行解析器注册该运行模式的参数。
        add_a2_arguments(commands.add_parser(name))
    # 调用 `add_a2_arguments` 向命令行解析器注册该运行模式的参数。
    add_a2_arguments(commands.add_parser("worker"), worker=True)
    # 调用 `commands.add_parser` 注册命令行子命令解析器，并将结果保存到 `preflight_worker`。
    preflight_worker = commands.add_parser("preflight-worker")
    # 调用 `add_a2_arguments` 向命令行解析器注册该运行模式的参数。
    add_a2_arguments(preflight_worker)
    # 调用 `preflight_worker.add_argument` 向命令行解析器注册当前流程使用的参数。
    preflight_worker.add_argument("--preflight-rank", type=int, required=True)
    # 将 `parser.parse_args()` 作为当前函数结果返回给调用方。
    return parser.parse_args()


# 定义命令行入口，解析参数并按用户选择调度当前阶段的主流程。
def main() -> None:
    """分派 ViT 训练、预检、验收或状态查询流程。

    输入：
        无显式输入参数。
    输出：
        None。
    """
    # 调用 `parse_args` 解析命令行文本并生成结构化参数，并将结果保存到 `args`。
    args = parse_args()
    # 调用 `validate_a2_args` 校验输入、配置或产物是否满足当前阶段的数据契约。
    validate_a2_args(args)
    # 检查 `args.command == 'preflight'`；根据条件结果选择对应处理分支。
    if args.command == "preflight": run_preflight(args)
    # 检查 `args.command == 'preflight-worker'`；根据条件结果选择对应处理分支。
    elif args.command == "preflight-worker": run_preflight_worker(args)
    # 检查 `args.command == 'a2'`；根据条件结果选择对应处理分支。
    elif args.command == "a2":
        # 调用 `run_parent` 按当前配置执行对应的完整子流程。
        run_parent(args)
        # 调用 `validate_outputs` 校验输入、配置或产物是否满足当前阶段的数据契约。
        validate_outputs(args)
    # 检查 `args.command == 'worker'`；根据条件结果选择对应处理分支。
    elif args.command == "worker": run_worker(args)
    # 检查 `args.command == 'validate'`；根据条件结果选择对应处理分支。
    elif args.command == "validate": validate_outputs(args)
    # 检查 `args.command == 'status'`；根据条件结果选择对应处理分支。
    elif args.command == "status": show_status(args)


# 检查 `__name__ == '__main__'`；根据条件结果选择对应处理分支。
if __name__ == "__main__":
    # 调用 `main` 启动当前模块的命令行主流程。
    main()
