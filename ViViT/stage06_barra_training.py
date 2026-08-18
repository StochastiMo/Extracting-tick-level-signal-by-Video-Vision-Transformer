"""ViViT 标签/分数中性化与 raw-score Barra R² 正则训练后端。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import deque
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


# 读取 `Path(__file__).resolve().parent` 的当前值，并赋给 `VIVIT_DIR`。
VIVIT_DIR = Path(__file__).resolve().parent
# 调用 `Path(__file__).resolve` 规范化文件路径，并将结果保存到 `SCRIPT_PATH`。
SCRIPT_PATH = Path(__file__).resolve()
# 检查 `str(VIVIT_DIR) not in sys.path`；根据条件结果选择对应处理分支。
if str(VIVIT_DIR) not in sys.path:
    # 调用 `sys.path.insert` 把项目模块目录加入导入搜索路径。
    sys.path.insert(0, str(VIVIT_DIR))

import stage01_sequence_data as sequence_data
import stage05_baseline_training as base
import stage05_distributed_training as distributed
import stage02_barra_data as style_data
from stage04_no3_adversarial import cosine_adversarial_gamma
from stage04_no4_gate import (
    prepare_conditioning_inputs as prepare_barra_conditioning_inputs,
)
from stage04_no6_dual_tower import prepare_dual_tower_inputs

from vit_stage06_ranking import (
    compute_training_loss,
    pearson_correlation,
    sigmoid_soft_rank,
)
from vit_stage07_training import (
    TradeDateBatchSampler,
    numpy_pearson,
    summarize_validation_arrays,
)


# 设置 `RUN_NAME` 的固定配置值，供后续流程统一使用。
RUN_NAME = "sigmoid_daybatch_rankic_style_residual_label_projected_output"
# 设置 `DEVICE_IDS` 的固定配置值，供后续流程统一使用。
DEVICE_IDS = "0,1,2,3"
# 设置 `TASKS_PER_GPU` 的固定配置值，供后续流程统一使用。
TASKS_PER_GPU = 1
# 设置 `WORKERS_PER_TASK` 的固定配置值，供后续流程统一使用。
WORKERS_PER_TASK = 8
# 设置 `RANK_TEMPERATURE` 的固定配置值，供后续流程统一使用。
RANK_TEMPERATURE = 0.1
# 设置 `TURNOVER_LAMBDA` 的固定配置值，供后续流程统一使用。
TURNOVER_LAMBDA = 0.1
# 设置 `HUBER_DELTA` 的固定配置值，供后续流程统一使用。
HUBER_DELTA = 1.0
# 设置 `SMOOTH_L1_BETA` 的固定配置值，供后续流程统一使用。
SMOOTH_L1_BETA = 0.1
# 设置 `HYBRID_WEIGHT` 的固定配置值，供后续流程统一使用。
HYBRID_WEIGHT = 0.5
# 设置 `LEARNING_RATE` 的固定配置值，供后续流程统一使用。
LEARNING_RATE = 0.001
# 设置 `MAX_EPOCHS` 的固定配置值，供后续流程统一使用。
MAX_EPOCHS = 100
# 设置 `PATIENCE` 的固定配置值，供后续流程统一使用。
PATIENCE = 10
# 设置 `USE_TORCH_COMPILE` 的固定配置值，供后续流程统一使用。
USE_TORCH_COMPILE = 1
# 设置 `COMPILE_MODE` 的固定配置值，供后续流程统一使用。
COMPILE_MODE = "default"
# 设置 `USE_FUSED_ADAM` 的固定配置值，供后续流程统一使用。
USE_FUSED_ADAM = 1
# 设置 `USE_CUDA_PREFETCH` 的固定配置值，供后续流程统一使用。
USE_CUDA_PREFETCH = 1
# 设置 `OVERWRITE` 的固定配置值，供后续流程统一使用。
OVERWRITE = 0
# 设置 `LOSS_NAME` 的固定配置值，供后续流程统一使用。
LOSS_NAME = "sigmoid_rank_ic"
# 设置 `VALIDATION_METRIC` 的固定配置值，供后续流程统一使用。
VALIDATION_METRIC = "daily_mean_spearman_ic"
# 设置 `TRAIN_BATCH_MODE` 的固定配置值，供后续流程统一使用。
TRAIN_BATCH_MODE = "trade_date"
# Execution metadata is deliberately absent from legacy A2 signatures. DDP
# adds it explicitly so checkpoints cannot resume across routes/world sizes.
# 执行当前分布式训练步骤并保持既定数据语义。
EXECUTION_ROUTE = "a2"
# 执行当前分布式训练步骤并保持既定数据语义。
DDP_WORLD_SIZE = 1
# 执行当前分布式训练步骤并保持既定数据语义。
EXECUTION_DEVICE_IDS: tuple[int, ...] = ()
# 设置 `RESIDUALIZE_TARGET` 的固定配置值，供后续流程统一使用。
RESIDUALIZE_TARGET = True
# 设置 `PROJECT_MODEL_OUTPUT` 的固定配置值，供后续流程统一使用。
PROJECT_MODEL_OUTPUT = True
# 设置 `RAW_BARRA_R2_LAMBDA` 的固定配置值，供后续流程统一使用。
RAW_BARRA_R2_LAMBDA = 0.0
# 设置 `ADVERSARIAL_BARRA` 的固定配置值，供后续流程统一使用。
ADVERSARIAL_BARRA = False
# 设置 `BARRA_CONDITIONING_MODE` 的固定配置值，供后续流程统一使用。
BARRA_CONDITIONING_MODE = "off"
# 设置 `BARRA_OUTPUT_NEUTRALIZATION` 的固定配置值，供后续流程统一使用。
BARRA_OUTPUT_NEUTRALIZATION = False
# 设置 `BARRA_DUAL_TOWER` 的固定配置值，供后续流程统一使用。
BARRA_DUAL_TOWER = False
# 设置 `BARRA_DUAL_TOWER_MODE` 的固定配置值，供后续流程统一使用。
BARRA_DUAL_TOWER_MODE = "real"
# 组织一组有序值，并保存到 `BARRA_DUAL_TOWER_MODES`。
BARRA_DUAL_TOWER_MODES = ("real", "alpha_video_only", "shuffled", "zero")
# 设置 `DUAL_TOWER_ALPHA_LOSS_WEIGHT` 的固定配置值，供后续流程统一使用。
DUAL_TOWER_ALPHA_LOSS_WEIGHT = 1.0
# 设置 `DUAL_TOWER_STYLE_LOSS_WEIGHT` 的固定配置值，供后续流程统一使用。
DUAL_TOWER_STYLE_LOSS_WEIGHT = 1.0
# 设置 `DUAL_TOWER_STYLE_HIDDEN_DIM` 的固定配置值，供后续流程统一使用。
DUAL_TOWER_STYLE_HIDDEN_DIM = 64
# 设置 `DUAL_TOWER_STYLE_DROPOUT` 的固定配置值，供后续流程统一使用。
DUAL_TOWER_STYLE_DROPOUT = 0.10
# 组织一组有序值，并保存到 `BARRA_CONDITIONING_MODES`。
BARRA_CONDITIONING_MODES = (
    "off", "video_only", "real", "shuffled", "zero"
)
# 设置 `BARRA_GATE_HIDDEN_DIM` 的固定配置值，供后续流程统一使用。
BARRA_GATE_HIDDEN_DIM = 64
# 设置 `BARRA_INDUSTRY_EMBEDDING_DIM` 的固定配置值，供后续流程统一使用。
BARRA_INDUSTRY_EMBEDDING_DIM = 16
# 设置 `BARRA_GATE_DROPOUT` 的固定配置值，供后续流程统一使用。
BARRA_GATE_DROPOUT = 0.10
# 设置 `BARRA_GATE_SCALE` 的固定配置值，供后续流程统一使用。
BARRA_GATE_SCALE = 0.50
# 设置 `ADVERSARIAL_GAMMA_MAX` 的固定配置值，供后续流程统一使用。
ADVERSARIAL_GAMMA_MAX = 0.10
# 设置 `ADVERSARIAL_WARMUP_EPOCHS` 的固定配置值，供后续流程统一使用。
ADVERSARIAL_WARMUP_EPOCHS = 2
# 设置 `ADVERSARIAL_RAMP_EPOCHS` 的固定配置值，供后续流程统一使用。
ADVERSARIAL_RAMP_EPOCHS = 5
# 设置 `ADVERSARIAL_STYLE_LOSS_WEIGHT` 的固定配置值，供后续流程统一使用。
ADVERSARIAL_STYLE_LOSS_WEIGHT = 1.0
# 设置 `ADVERSARIAL_INDUSTRY_LOSS_WEIGHT` 的固定配置值，供后续流程统一使用。
ADVERSARIAL_INDUSTRY_LOSS_WEIGHT = 1.0
# 设置 `ADVERSARIAL_HEAD_HIDDEN_DIM` 的固定配置值，供后续流程统一使用。
ADVERSARIAL_HEAD_HIDDEN_DIM = 128
# 设置 `ADVERSARIAL_HEAD_DROPOUT` 的固定配置值，供后续流程统一使用。
ADVERSARIAL_HEAD_DROPOUT = 0.10
# 读取 `sequence_data.FORECAST_YEARS` 的当前值，并赋给 `FORECAST_YEARS`。
FORECAST_YEARS = sequence_data.FORECAST_YEARS
# 读取 `sequence_data.SEEDS` 的当前值，并赋给 `SEEDS`。
SEEDS = sequence_data.SEEDS
# 调用 `len` 汇总当前数据，并将统计量保存到 `STYLE_COUNT`。
STYLE_COUNT = len(style_data.STYLE_COLUMNS)
# 设置 `TRAINING_SCHEMA_VERSION` 的固定配置值，供后续流程统一使用。
TRAINING_SCHEMA_VERSION = 4
# 读取 `style_data.EXPECTED_CNE5S_INDUSTRIES` 的当前值，并赋给 `INDUSTRY_COUNT`。
INDUSTRY_COUNT = style_data.EXPECTED_CNE5S_INDUSTRIES

# 设置 `PUBLISHED_FACTOR_POLICY` 的固定配置值，供后续流程统一使用。
PUBLISHED_FACTOR_POLICY = "unprojected_model_score_daily_seed_zscore_mean"
# 构造去重集合，并保存到 `RESUME_COMPATIBLE_TRAINING_CODE_SHA256`。
RESUME_COMPATIBLE_TRAINING_CODE_SHA256 = {
    # No.6 DDP Style Pearson 塌缩诊断阈值修复前的版本；
    # 训练目标和模型参数未变化，允许相同 route/world size 安全恢复。
    "62545fd27e812da5dbe560862377f12688d0d62a4f70b5b94a33bd44630da053",
    # No.6 BF16 Total 后处理修复前的已完成双塔训练版本；
    # 仅代码哈希可不同，其余 run payload 仍须逐字段完全一致。
    "ba9380f4f16b7a94763bbb73a214722a5346667ee22e03bf1ae445d9d5bd071e",
    "07c66a7d7b4d0d1ca3547bcf948273dae4b46237f2e686455211e5f93095cd0e",
    "f57ff0558cf15a1b2752859075c078d78026a8da3d319721df8a96acfe2a9521",
    "93445184d661eefca7facbe78ae41148416d72711872db75df746f07883ce745",
    "f995917d46632377f9451152014f3508b41eb0f5f4d5280ccd126603219cbf1b",
    "d0e8053f55642a6393af2cb9f9a3419345cbbaa1dcb515bf30ff02955e131fe1",
    "e4d3b5b96ddaff1081c55d655da227726044bf20d41300706343465395a085f3",
    "1ecb47d19866b7a63189cf075f41e872319fc7e6ca79828a0dfbf19803f46040",
    "9b48e789a40b4f56af9013d2c86260d2f81d6ad16ff98bcac83ad7405b3db4d6",
}


# 定义 `DegenerateValidationScoreError` 类，集中封装该阶段的数据结构与相关行为。
class DegenerateValidationScoreError(FloatingPointError):
    """验证日的原始预测分数退化为常数或近似常数。
    """

    # 定义初始化方法，接收配置并建立实例后续计算所需的内部状态。
    def __init__(
        self,
        *,
        date_id: int,
        rows: int,
        score_std: float,
        total_sum_squares: float,
    ) -> None:
        """初始化 DegenerateValidationScoreError 所需的数据引用、配置和运行状态。
        
        输入：
            date_id: int。
            rows: int。
            score_std: float。
            total_sum_squares: float。
        输出：
            None。
            """
        # 初始化父类状态，确保继承链中的基础组件正确注册。
        super().__init__(
            "验证集原始预测分数在单日截面内退化："
            f"date_id={date_id}, rows={rows}, score_std={score_std:.6g}, "
            f"total_sum_squares={total_sum_squares:.6g}"
        )
        # 调用 `int` 统一设备或数据类型，并将结果保存到 `self.date_id`。
        self.date_id = int(date_id)
        # 调用 `int` 统一设备或数据类型，并将结果保存到 `self.rows`。
        self.rows = int(rows)
        # 调用 `float` 统一设备或数据类型，并将结果保存到 `self.score_std`。
        self.score_std = float(score_std)
        # 调用 `float` 统一设备或数据类型，并将结果保存到 `self.total_sum_squares`。
        self.total_sum_squares = float(total_sum_squares)


# 定义 `is_resume_compatible_recovery_patch` 函数，封装这一阶段可独立调用的处理步骤。
def is_resume_compatible_recovery_patch(
    checkpoint_payload: dict[str, Any],
    current_payload: dict[str, Any],
) -> bool:
    """仅迁移已确认会因验证分数退化而中断的旧训练代码版本。
    
    输入：
        checkpoint_payload: dict[str, Any]。
        current_payload: dict[str, Any]。
    输出：
        bool。
        """

    # 执行当前分布式训练步骤并保持既定数据语义。
    checkpoint_code_hash = checkpoint_payload.get("training_code_sha256")
    # 执行当前分布式训练步骤并保持既定数据语义。
    checkpoint_route = checkpoint_payload.get("execution_route", "a2")
    # 执行当前分布式训练步骤并保持既定数据语义。
    current_route = current_payload.get("execution_route", "a2")
    # 执行当前分布式训练步骤并保持既定数据语义。
    if checkpoint_route != current_route:
        # 执行当前分布式训练步骤并保持既定数据语义。
        return False
    # 执行当前分布式训练步骤并保持既定数据语义。
    if checkpoint_code_hash not in RESUME_COMPATIBLE_TRAINING_CODE_SHA256:
        # 执行当前分布式训练步骤并保持既定数据语义。
        return False

    # 执行当前分布式训练步骤并保持既定数据语义。
    ignored = {"training_code_sha256"}
    # 执行当前分布式训练步骤并保持既定数据语义。
    if checkpoint_route == "a2":
        # 执行当前分布式训练步骤并保持既定数据语义。
        ignored.update({"execution_route", "world_size"})
    # 执行当前分布式训练步骤并保持既定数据语义。
    checkpoint_normalized = {
        key: value
        for key, value in checkpoint_payload.items()
        if key not in ignored
    }
    # 执行当前分布式训练步骤并保持既定数据语义。
    current_normalized = {
        key: value
        for key, value in current_payload.items()
        if key not in ignored
    }
    # 执行当前分布式训练步骤并保持既定数据语义。
    return checkpoint_normalized == current_normalized


# 定义 `barra_conditioning_enabled` 函数，封装这一阶段可独立调用的处理步骤。
def barra_conditioning_enabled() -> bool:
    """判断当前实验是否启用 Barra 条件输入。
    
    输入：
        无显式输入参数。
    输出：
        bool。
        """
    # 将 `BARRA_CONDITIONING_MODE != 'off'` 作为当前函数结果返回给调用方。
    return BARRA_CONDITIONING_MODE != "off"

# 定义 `strict_barra_output_enabled` 函数，封装这一阶段可独立调用的处理步骤。
def strict_barra_output_enabled() -> bool:
    """判断当前实验是否发布严格中性的投影分数。
    
    输入：
        无显式输入参数。
    输出：
        bool。
        """
    # 将 `bool(BARRA_OUTPUT_NEUTRALIZATION)` 作为当前函数结果返回给调用方。
    return bool(BARRA_OUTPUT_NEUTRALIZATION)


# 定义 `dual_tower_enabled` 函数，封装这一阶段可独立调用的处理步骤。
def dual_tower_enabled() -> bool:
    """判断当前实验是否启用 Alpha/Style 双塔。
    
    输入：
        无显式输入参数。
    输出：
        bool。
        """
    # 将 `bool(BARRA_DUAL_TOWER)` 作为当前函数结果返回给调用方。
    return bool(BARRA_DUAL_TOWER)


# 定义 `target_policy` 函数，封装这一阶段可独立调用的处理步骤。
def target_policy() -> str:
    """返回当前实验使用的原始、连续风格残差或完整风险残差标签策略。
    
    输入：
        无显式输入参数。
    输出：
        str。
        """
    # 检查 `dual_tower_enabled()`；根据条件结果选择对应处理分支。
    if dual_tower_enabled():
        # 将 `'source_plus_full_barra_residual_plus_fitted_targets'` 作为当前函数结果返回给调用方。
        return "source_plus_full_barra_residual_plus_fitted_targets"
    # 检查 `barra_conditioning_enabled()`；根据条件结果选择对应处理分支。
    if barra_conditioning_enabled():
        # 将 `'daily_ten_style_plus_industry_residual_then_zscore_ddof0'` 作为当前函数结果返回给调用方。
        return "daily_ten_style_plus_industry_residual_then_zscore_ddof0"
    # 将 `'daily_ten_style_ols_residual_then_zscore_ddof0' if RESIDUALIZE_TARGET else 'source_target_zsco…` 作为当前函数结果返回给调用方。
    return (
        "daily_ten_style_ols_residual_then_zscore_ddof0"
        if RESIDUALIZE_TARGET
        else "source_target_zscore"
    )


# 定义 `output_policy` 函数，封装这一阶段可独立调用的处理步骤。
def output_policy() -> str:
    """返回训练、验证和正式发布分数的口径策略。
    
    输入：
        无显式输入参数。
    输出：
        str。
        """
    # 检查 `dual_tower_enabled()`；根据条件结果选择对应处理分支。
    if dual_tower_enabled():
        # 将 `'dual_tower_unprojected_alpha_score'` 作为当前函数结果返回给调用方。
        return "dual_tower_unprojected_alpha_score"
    # 检查 `strict_barra_output_enabled()`；根据条件结果选择对应处理分支。
    if strict_barra_output_enabled():
        # 将 `'daily_ten_style_plus_industry_projection_then_zscore_ddof0'` 作为当前函数结果返回给调用方。
        return "daily_ten_style_plus_industry_projection_then_zscore_ddof0"
    # 将 `'daily_ten_style_qr_residual_then_zscore_ddof0' if PROJECT_MODEL_OUTPUT else 'model_raw_score'` 作为当前函数结果返回给调用方。
    return (
        "daily_ten_style_qr_residual_then_zscore_ddof0"
        if PROJECT_MODEL_OUTPUT
        else "model_raw_score"
    )


# 定义 `raw_barra_r2_enabled` 函数，封装这一阶段可独立调用的处理步骤。
def raw_barra_r2_enabled() -> bool:
    """判断是否启用 raw-score 联合 Barra R² 惩罚。
    
    输入：
        无显式输入参数。
    输出：
        bool。
        """
    # 将 `RAW_BARRA_R2_LAMBDA > 0.0` 作为当前函数结果返回给调用方。
    return RAW_BARRA_R2_LAMBDA > 0.0


# 定义 `adversarial_barra_enabled` 函数，封装这一阶段可独立调用的处理步骤。
def adversarial_barra_enabled() -> bool:
    """判断是否启用对抗式 Barra 不变表征。
    
    输入：
        无显式输入参数。
    输出：
        bool。
        """
    # 将 `bool(ADVERSARIAL_BARRA)` 作为当前函数结果返回给调用方。
    return bool(ADVERSARIAL_BARRA)


# 定义 `industry_sidecar_required` 函数，封装这一阶段可独立调用的处理步骤。
def industry_sidecar_required() -> bool:
    """判断当前实验是否必须加载行业 sidecar。
    
    输入：
        无显式输入参数。
    输出：
        bool。
        """
    # 将 `raw_barra_r2_enabled() or adversarial_barra_enabled() or barra_conditioning_enabled() or dual_t…` 作为当前函数结果返回给调用方。
    return (
        raw_barra_r2_enabled()
        or adversarial_barra_enabled()
        or barra_conditioning_enabled()
        or dual_tower_enabled()
    )


# 定义 `factor_role` 函数，封装这一阶段可独立调用的处理步骤。
def factor_role() -> str:
    """说明某个预测列在当前实验中的 Alpha、Style 或总收益角色。
    
    输入：
        无显式输入参数。
    输出：
        str。
        """
    # 检查 `dual_tower_enabled()`；根据条件结果选择对应处理分支。
    if dual_tower_enabled():
        # 将 `f'barra_conditional_dual_tower_alpha_{BARRA_DUAL_TOWER_MODE}'` 作为当前函数结果返回给调用方。
        return f"barra_conditional_dual_tower_alpha_{BARRA_DUAL_TOWER_MODE}"
    # 检查 `strict_barra_output_enabled()`；根据条件结果选择对应处理分支。
    if strict_barra_output_enabled():
        # 将 `f'barra_conditioned_strict_neutral_{BARRA_CONDITIONING_MODE}'` 作为当前函数结果返回给调用方。
        return f"barra_conditioned_strict_neutral_{BARRA_CONDITIONING_MODE}"
    # 检查 `barra_conditioning_enabled()`；根据条件结果选择对应处理分支。
    if barra_conditioning_enabled():
        # 将 `f'barra_conditioned_multiplicative_gate_{BARRA_CONDITIONING_MODE}'` 作为当前函数结果返回给调用方。
        return f"barra_conditioned_multiplicative_gate_{BARRA_CONDITIONING_MODE}"
    # 检查 `adversarial_barra_enabled()`；根据条件结果选择对应处理分支。
    if adversarial_barra_enabled():
        # 将 `'adversarial_barra_invariant_representation_model'` 作为当前函数结果返回给调用方。
        return "adversarial_barra_invariant_representation_model"
    # 检查 `raw_barra_r2_enabled()`；根据条件结果选择对应处理分支。
    if raw_barra_r2_enabled():
        # 将 `'raw_score_barra_r_squared_regularized_model'` 作为当前函数结果返回给调用方。
        return "raw_score_barra_r_squared_regularized_model"
    # 检查 `RESIDUALIZE_TARGET and PROJECT_MODEL_OUTPUT`；根据条件结果选择对应处理分支。
    if RESIDUALIZE_TARGET and PROJECT_MODEL_OUTPUT:
        # 将 `'style_residual_label_projected_output_model'` 作为当前函数结果返回给调用方。
        return "style_residual_label_projected_output_model"
    # 检查 `RESIDUALIZE_TARGET`；根据条件结果选择对应处理分支。
    if RESIDUALIZE_TARGET:
        # 将 `'style_residual_label_model'` 作为当前函数结果返回给调用方。
        return "style_residual_label_model"
    # 检查 `PROJECT_MODEL_OUTPUT`；根据条件结果选择对应处理分支。
    if PROJECT_MODEL_OUTPUT:
        # 将 `'style_projected_output_model'` 作为当前函数结果返回给调用方。
        return "style_projected_output_model"
    # 将 `'style_backend_raw_model'` 作为当前函数结果返回给调用方。
    return "style_backend_raw_model"


# 定义 `configure_from_args` 函数，封装这一阶段可独立调用的处理步骤。
def configure_from_args(args: argparse.Namespace) -> None:
    """将 CLI 配置安装为 worker 进程内只读实验配置。
    
    输入：
        args: argparse.Namespace。
    输出：
        None。
        """

    # 声明 `RUN_NAME, DEVICE_IDS, TASKS_PER_GPU, WORKERS_PER_TASK` 使用模块级状态，避免创建同名局部变量。
    global RUN_NAME, DEVICE_IDS, TASKS_PER_GPU, WORKERS_PER_TASK
    # 执行当前分布式训练步骤并保持既定数据语义。
    global EXECUTION_ROUTE, DDP_WORLD_SIZE, EXECUTION_DEVICE_IDS
    # 声明 `RANK_TEMPERATURE, TURNOVER_LAMBDA, HUBER_DELTA` 使用模块级状态，避免创建同名局部变量。
    global RANK_TEMPERATURE, TURNOVER_LAMBDA, HUBER_DELTA
    # 声明 `SMOOTH_L1_BETA, HYBRID_WEIGHT, LEARNING_RATE` 使用模块级状态，避免创建同名局部变量。
    global SMOOTH_L1_BETA, HYBRID_WEIGHT, LEARNING_RATE
    # 声明 `MAX_EPOCHS, PATIENCE, USE_TORCH_COMPILE, COMPILE_MODE` 使用模块级状态，避免创建同名局部变量。
    global MAX_EPOCHS, PATIENCE, USE_TORCH_COMPILE, COMPILE_MODE
    # 声明 `USE_FUSED_ADAM, USE_CUDA_PREFETCH, OVERWRITE, LOSS_NAME` 使用模块级状态，避免创建同名局部变量。
    global USE_FUSED_ADAM, USE_CUDA_PREFETCH, OVERWRITE, LOSS_NAME
    # 声明 `VALIDATION_METRIC, TRAIN_BATCH_MODE` 使用模块级状态，避免创建同名局部变量。
    global VALIDATION_METRIC, TRAIN_BATCH_MODE
    # 声明 `RESIDUALIZE_TARGET, PROJECT_MODEL_OUTPUT, RAW_BARRA_R2_LAMBDA` 使用模块级状态，避免创建同名局部变量。
    global RESIDUALIZE_TARGET, PROJECT_MODEL_OUTPUT, RAW_BARRA_R2_LAMBDA
    # 声明 `ADVERSARIAL_BARRA, ADVERSARIAL_GAMMA_MAX` 使用模块级状态，避免创建同名局部变量。
    global ADVERSARIAL_BARRA, ADVERSARIAL_GAMMA_MAX
    # 声明 `BARRA_CONDITIONING_MODE, BARRA_GATE_HIDDEN_DIM` 使用模块级状态，避免创建同名局部变量。
    global BARRA_CONDITIONING_MODE, BARRA_GATE_HIDDEN_DIM
    # 声明 `BARRA_INDUSTRY_EMBEDDING_DIM, BARRA_GATE_DROPOUT` 使用模块级状态，避免创建同名局部变量。
    global BARRA_INDUSTRY_EMBEDDING_DIM, BARRA_GATE_DROPOUT
    # 声明 `BARRA_GATE_SCALE, BARRA_OUTPUT_NEUTRALIZATION` 使用模块级状态，避免创建同名局部变量。
    global BARRA_GATE_SCALE, BARRA_OUTPUT_NEUTRALIZATION
    # 声明 `BARRA_DUAL_TOWER, BARRA_DUAL_TOWER_MODE` 使用模块级状态，避免创建同名局部变量。
    global BARRA_DUAL_TOWER, BARRA_DUAL_TOWER_MODE
    # 声明 `DUAL_TOWER_ALPHA_LOSS_WEIGHT, DUAL_TOWER_STYLE_LOSS_WEIGHT` 使用模块级状态，避免创建同名局部变量。
    global DUAL_TOWER_ALPHA_LOSS_WEIGHT, DUAL_TOWER_STYLE_LOSS_WEIGHT
    # 声明 `DUAL_TOWER_STYLE_HIDDEN_DIM, DUAL_TOWER_STYLE_DROPOUT` 使用模块级状态，避免创建同名局部变量。
    global DUAL_TOWER_STYLE_HIDDEN_DIM, DUAL_TOWER_STYLE_DROPOUT
    # 声明 `ADVERSARIAL_WARMUP_EPOCHS, ADVERSARIAL_RAMP_EPOCHS` 使用模块级状态，避免创建同名局部变量。
    global ADVERSARIAL_WARMUP_EPOCHS, ADVERSARIAL_RAMP_EPOCHS
    # 声明 `ADVERSARIAL_STYLE_LOSS_WEIGHT, ADVERSARIAL_INDUSTRY_LOSS_WEIGHT` 使用模块级状态，避免创建同名局部变量。
    global ADVERSARIAL_STYLE_LOSS_WEIGHT, ADVERSARIAL_INDUSTRY_LOSS_WEIGHT
    # 声明 `ADVERSARIAL_HEAD_HIDDEN_DIM, ADVERSARIAL_HEAD_DROPOUT` 使用模块级状态，避免创建同名局部变量。
    global ADVERSARIAL_HEAD_HIDDEN_DIM, ADVERSARIAL_HEAD_DROPOUT

    # 读取 `args.run_name` 的当前值，并赋给 `RUN_NAME`。
    RUN_NAME = args.run_name
    # 调用 `getattr` 读取对象的可选属性并应用缺省值，并将结果保存到 `DEVICE_IDS`。
    DEVICE_IDS = getattr(args, "device_ids", DEVICE_IDS)
    # 执行当前分布式训练步骤并保持既定数据语义。
    EXECUTION_ROUTE = str(getattr(args, "execution_route", "a2"))
    # 执行当前分布式训练步骤并保持既定数据语义。
    DDP_WORLD_SIZE = int(getattr(args, "ddp_world_size", 1))
    # 执行当前分布式训练步骤并保持既定数据语义。
    EXECUTION_DEVICE_IDS = distributed.parse_device_ids(DEVICE_IDS)
    # 调用 `int` 统一设备或数据类型，并将结果保存到 `TASKS_PER_GPU`。
    TASKS_PER_GPU = int(getattr(args, "tasks_per_gpu", TASKS_PER_GPU))
    # 调用 `int` 统一设备或数据类型，并将结果保存到 `WORKERS_PER_TASK`。
    WORKERS_PER_TASK = int(args.workers_per_task)
    # 调用 `float` 统一设备或数据类型，并将结果保存到 `RANK_TEMPERATURE`。
    RANK_TEMPERATURE = float(args.rank_temperature)
    # 调用 `float` 统一设备或数据类型，并将结果保存到 `TURNOVER_LAMBDA`。
    TURNOVER_LAMBDA = float(args.turnover_lambda)
    # 调用 `float` 统一设备或数据类型，并将结果保存到 `HUBER_DELTA`。
    HUBER_DELTA = float(args.huber_delta)
    # 调用 `float` 统一设备或数据类型，并将结果保存到 `SMOOTH_L1_BETA`。
    SMOOTH_L1_BETA = float(args.smooth_l1_beta)
    # 调用 `float` 统一设备或数据类型，并将结果保存到 `HYBRID_WEIGHT`。
    HYBRID_WEIGHT = float(args.hybrid_weight)
    # 调用 `float` 统一设备或数据类型，并将结果保存到 `LEARNING_RATE`。
    LEARNING_RATE = float(args.learning_rate)
    # 调用 `int` 统一设备或数据类型，并将结果保存到 `MAX_EPOCHS`。
    MAX_EPOCHS = int(args.max_epochs)
    # 调用 `int` 统一设备或数据类型，并将结果保存到 `PATIENCE`。
    PATIENCE = int(args.patience)
    # 调用 `int` 统一设备或数据类型，并将结果保存到 `USE_TORCH_COMPILE`。
    USE_TORCH_COMPILE = int(args.compile)
    # 读取 `args.compile_mode` 的当前值，并赋给 `COMPILE_MODE`。
    COMPILE_MODE = args.compile_mode
    # 调用 `int` 统一设备或数据类型，并将结果保存到 `USE_FUSED_ADAM`。
    USE_FUSED_ADAM = int(args.fused_adam)
    # 调用 `int` 统一设备或数据类型，并将结果保存到 `USE_CUDA_PREFETCH`。
    USE_CUDA_PREFETCH = int(args.cuda_prefetch)
    # 调用 `int` 统一设备或数据类型，并将结果保存到 `OVERWRITE`。
    OVERWRITE = int(args.overwrite)
    # 读取 `args.loss_name` 的当前值，并赋给 `LOSS_NAME`。
    LOSS_NAME = args.loss_name
    # 读取 `args.validation_metric` 的当前值，并赋给 `VALIDATION_METRIC`。
    VALIDATION_METRIC = args.validation_metric
    # 读取 `args.train_batch_mode` 的当前值，并赋给 `TRAIN_BATCH_MODE`。
    TRAIN_BATCH_MODE = args.train_batch_mode
    # 调用 `bool` 统一设备或数据类型，并将结果保存到 `RESIDUALIZE_TARGET`。
    RESIDUALIZE_TARGET = bool(args.residualize_target)
    # 调用 `bool` 统一设备或数据类型，并将结果保存到 `PROJECT_MODEL_OUTPUT`。
    PROJECT_MODEL_OUTPUT = bool(args.project_model_output)
    # 调用 `float` 统一设备或数据类型，并将结果保存到 `RAW_BARRA_R2_LAMBDA`。
    RAW_BARRA_R2_LAMBDA = float(args.raw_barra_r2_lambda)
    # 调用 `bool` 统一设备或数据类型，并将结果保存到 `ADVERSARIAL_BARRA`。
    ADVERSARIAL_BARRA = bool(args.adversarial_barra)
    # 调用 `str` 把当前值转换为统一字符串表示，并将结果保存到 `BARRA_CONDITIONING_MODE`。
    BARRA_CONDITIONING_MODE = str(args.barra_conditioning_mode)
    # 调用 `bool` 统一设备或数据类型，并将结果保存到 `BARRA_OUTPUT_NEUTRALIZATION`。
    BARRA_OUTPUT_NEUTRALIZATION = bool(
        args.barra_output_neutralization
    )
    # 调用 `bool` 统一设备或数据类型，并将结果保存到 `BARRA_DUAL_TOWER`。
    BARRA_DUAL_TOWER = bool(args.barra_dual_tower)
    # 调用 `str` 把当前值转换为统一字符串表示，并将结果保存到 `BARRA_DUAL_TOWER_MODE`。
    BARRA_DUAL_TOWER_MODE = str(args.barra_dual_tower_mode)
    # 调用 `float` 统一设备或数据类型，并将结果保存到 `DUAL_TOWER_ALPHA_LOSS_WEIGHT`。
    DUAL_TOWER_ALPHA_LOSS_WEIGHT = float(
        args.dual_tower_alpha_loss_weight
    )
    # 调用 `float` 统一设备或数据类型，并将结果保存到 `DUAL_TOWER_STYLE_LOSS_WEIGHT`。
    DUAL_TOWER_STYLE_LOSS_WEIGHT = float(
        args.dual_tower_style_loss_weight
    )
    # 调用 `int` 统一设备或数据类型，并将结果保存到 `DUAL_TOWER_STYLE_HIDDEN_DIM`。
    DUAL_TOWER_STYLE_HIDDEN_DIM = int(args.dual_tower_style_hidden_dim)
    # 调用 `float` 统一设备或数据类型，并将结果保存到 `DUAL_TOWER_STYLE_DROPOUT`。
    DUAL_TOWER_STYLE_DROPOUT = float(args.dual_tower_style_dropout)
    # 调用 `int` 统一设备或数据类型，并将结果保存到 `BARRA_GATE_HIDDEN_DIM`。
    BARRA_GATE_HIDDEN_DIM = int(args.barra_gate_hidden_dim)
    # 调用 `int` 统一设备或数据类型，并将结果保存到 `BARRA_INDUSTRY_EMBEDDING_DIM`。
    BARRA_INDUSTRY_EMBEDDING_DIM = int(
        args.barra_industry_embedding_dim
    )
    # 调用 `float` 统一设备或数据类型，并将结果保存到 `BARRA_GATE_DROPOUT`。
    BARRA_GATE_DROPOUT = float(args.barra_gate_dropout)
    # 调用 `float` 统一设备或数据类型，并将结果保存到 `BARRA_GATE_SCALE`。
    BARRA_GATE_SCALE = float(args.barra_gate_scale)
    # 调用 `float` 统一设备或数据类型，并将结果保存到 `ADVERSARIAL_GAMMA_MAX`。
    ADVERSARIAL_GAMMA_MAX = float(args.adversarial_gamma_max)
    # 调用 `int` 统一设备或数据类型，并将结果保存到 `ADVERSARIAL_WARMUP_EPOCHS`。
    ADVERSARIAL_WARMUP_EPOCHS = int(args.adversarial_warmup_epochs)
    # 调用 `int` 统一设备或数据类型，并将结果保存到 `ADVERSARIAL_RAMP_EPOCHS`。
    ADVERSARIAL_RAMP_EPOCHS = int(args.adversarial_ramp_epochs)
    # 调用 `float` 统一设备或数据类型，并将结果保存到 `ADVERSARIAL_STYLE_LOSS_WEIGHT`。
    ADVERSARIAL_STYLE_LOSS_WEIGHT = float(
        args.adversarial_style_loss_weight
    )
    # 调用 `float` 统一设备或数据类型，并将结果保存到 `ADVERSARIAL_INDUSTRY_LOSS_WEIGHT`。
    ADVERSARIAL_INDUSTRY_LOSS_WEIGHT = float(
        args.adversarial_industry_loss_weight
    )
    # 调用 `int` 统一设备或数据类型，并将结果保存到 `ADVERSARIAL_HEAD_HIDDEN_DIM`。
    ADVERSARIAL_HEAD_HIDDEN_DIM = int(args.adversarial_head_hidden_dim)
    # 调用 `float` 统一设备或数据类型，并将结果保存到 `ADVERSARIAL_HEAD_DROPOUT`。
    ADVERSARIAL_HEAD_DROPOUT = float(args.adversarial_head_dropout)

    # 调用 `training_config` 整理并返回当前实验的配置对象，并将结果保存到 `config`。
    config = training_config()
    # 调用 `base.validate_training_config` 校验输入、配置或产物是否满足当前阶段的数据契约。
    base.validate_training_config(config, COMPILE_MODE)
    # 检查 `TRAIN_BATCH_MODE != 'trade_date'`；根据条件结果选择对应处理分支。
    if TRAIN_BATCH_MODE != "trade_date":
        # 抛出 `ValueError('风格训练或 Barra R² 正则必须使用 trade_date batch')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("风格训练或 Barra R² 正则必须使用 trade_date batch")
    # 检查 `LOSS_NAME == base.TEMPORAL_TURNOVER_LOSS`；根据条件结果选择对应处理分支。
    if LOSS_NAME == base.TEMPORAL_TURNOVER_LOSS:
        # 抛出 `ValueError('风格训练后端暂不支持 sigmoid_rank_ic_turnover')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("风格训练后端暂不支持 sigmoid_rank_ic_turnover")
    # 检查 `RAW_BARRA_R2_LAMBDA < 0`；根据条件结果选择对应处理分支。
    if RAW_BARRA_R2_LAMBDA < 0:
        # 抛出 `ValueError('raw_barra_r2_lambda 必须 >= 0')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("raw_barra_r2_lambda 必须 >= 0")
    # 检查 `raw_barra_r2_enabled() and (not dual_tower_enabled()) and (RESIDUALIZE_TARGET or PROJECT_MODEL_…`；根据条件结果选择对应处理分支。
    if raw_barra_r2_enabled() and not dual_tower_enabled() and (
        RESIDUALIZE_TARGET or PROJECT_MODEL_OUTPUT
    ):
        # 抛出 `ValueError('No.2 要求 RESIDUALIZE_TARGET=False 且 PROJECT_MODEL_OUTPUT=False')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(
            "No.2 要求 RESIDUALIZE_TARGET=False 且 "
            "PROJECT_MODEL_OUTPUT=False"
        )
    # 检查 `raw_barra_r2_enabled() and (not dual_tower_enabled()) and (LOSS_NAME != 'sigmoid_rank_ic')`；根据条件结果选择对应处理分支。
    if raw_barra_r2_enabled() and not dual_tower_enabled() and LOSS_NAME != "sigmoid_rank_ic":
        # 抛出 `ValueError('No.2 当前只支持 sigmoid_rank_ic')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("No.2 当前只支持 sigmoid_rank_ic")
    # 检查 `raw_barra_r2_enabled() and (not dual_tower_enabled()) and (VALIDATION_METRIC != 'daily_mean_spe…`；根据条件结果选择对应处理分支。
    if (
        raw_barra_r2_enabled() and not dual_tower_enabled()
        and VALIDATION_METRIC != "daily_mean_spearman_ic"
    ):
        # 抛出 `ValueError('No.2 的早停指标必须为 daily_mean_spearman_ic')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(
            "No.2 的早停指标必须为 daily_mean_spearman_ic"
        )
    # 检查 `adversarial_barra_enabled()`；根据条件结果选择对应处理分支。
    if adversarial_barra_enabled():
        # 检查 `RESIDUALIZE_TARGET or PROJECT_MODEL_OUTPUT`；根据条件结果选择对应处理分支。
        if RESIDUALIZE_TARGET or PROJECT_MODEL_OUTPUT:
            # 抛出 `ValueError('No.3 要求 RESIDUALIZE_TARGET=False 且 PROJECT_MODEL_OUTPUT=False')`，阻止不满足数据契约的流程继续执行。
            raise ValueError(
                "No.3 要求 RESIDUALIZE_TARGET=False 且 "
                "PROJECT_MODEL_OUTPUT=False"
            )
        # 检查 `raw_barra_r2_enabled()`；根据条件结果选择对应处理分支。
        if raw_barra_r2_enabled():
            # 抛出 `ValueError('No.3 首轮纯对抗实验禁止叠加 raw-score R²')`，阻止不满足数据契约的流程继续执行。
            raise ValueError("No.3 首轮纯对抗实验禁止叠加 raw-score R²")
        # 检查 `LOSS_NAME != 'sigmoid_rank_ic'`；根据条件结果选择对应处理分支。
        if LOSS_NAME != "sigmoid_rank_ic":
            # 抛出 `ValueError('No.3 当前只支持 sigmoid_rank_ic')`，阻止不满足数据契约的流程继续执行。
            raise ValueError("No.3 当前只支持 sigmoid_rank_ic")
        # 检查 `VALIDATION_METRIC != 'daily_mean_spearman_ic'`；根据条件结果选择对应处理分支。
        if VALIDATION_METRIC != "daily_mean_spearman_ic":
            # 抛出 `ValueError('No.3 的早停指标必须为 daily_mean_spearman_ic')`，阻止不满足数据契约的流程继续执行。
            raise ValueError(
                "No.3 的早停指标必须为 daily_mean_spearman_ic"
            )
        # 检查 `ADVERSARIAL_GAMMA_MAX <= 0 or ADVERSARIAL_WARMUP_EPOCHS < 0 or ADVERSARIAL_RAMP_EPOCHS < 1 or (…`；根据条件结果选择对应处理分支。
        if (
            ADVERSARIAL_GAMMA_MAX <= 0
            or ADVERSARIAL_WARMUP_EPOCHS < 0
            or ADVERSARIAL_RAMP_EPOCHS < 1
            or ADVERSARIAL_STYLE_LOSS_WEIGHT <= 0
            or ADVERSARIAL_INDUSTRY_LOSS_WEIGHT <= 0
            or ADVERSARIAL_HEAD_HIDDEN_DIM < 1
            or not 0 <= ADVERSARIAL_HEAD_DROPOUT < 1
        ):
            # 抛出 `ValueError('No.3 对抗训练超参数非法')`，阻止不满足数据契约的流程继续执行。
            raise ValueError("No.3 对抗训练超参数非法")
    # 检查 `BARRA_CONDITIONING_MODE not in BARRA_CONDITIONING_MODES`；根据条件结果选择对应处理分支。
    if BARRA_CONDITIONING_MODE not in BARRA_CONDITIONING_MODES:
        # 抛出 `ValueError(f'barra_conditioning_mode 必须属于 {BARRA_CONDITIONING_MODES}')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(
            f"barra_conditioning_mode 必须属于 {BARRA_CONDITIONING_MODES}"
        )
    # 检查 `BARRA_DUAL_TOWER_MODE not in BARRA_DUAL_TOWER_MODES`；根据条件结果选择对应处理分支。
    if BARRA_DUAL_TOWER_MODE not in BARRA_DUAL_TOWER_MODES:
        # 抛出 `ValueError(f'barra_dual_tower_mode 必须属于 {BARRA_DUAL_TOWER_MODES}')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(
            f"barra_dual_tower_mode 必须属于 {BARRA_DUAL_TOWER_MODES}"
        )
    # 检查 `dual_tower_enabled()`；根据条件结果选择对应处理分支。
    if dual_tower_enabled():
        # 检查 `RESIDUALIZE_TARGET or PROJECT_MODEL_OUTPUT`；根据条件结果选择对应处理分支。
        if RESIDUALIZE_TARGET or PROJECT_MODEL_OUTPUT:
            # 抛出 `ValueError('No.6 要求关闭 Neutral 标签与输出投影')`，阻止不满足数据契约的流程继续执行。
            raise ValueError("No.6 要求关闭 Neutral 标签与输出投影")
        # 检查 `adversarial_barra_enabled() or barra_conditioning_enabled()`；根据条件结果选择对应处理分支。
        if adversarial_barra_enabled() or barra_conditioning_enabled():
            # 抛出 `ValueError('No.6 不能与 No.3 或 No.4 独立模式同时开启')`，阻止不满足数据契约的流程继续执行。
            raise ValueError("No.6 不能与 No.3 或 No.4 独立模式同时开启")
        # 检查 `strict_barra_output_enabled()`；根据条件结果选择对应处理分支。
        if strict_barra_output_enabled():
            # 抛出 `ValueError('No.6 正式输出为未投影 Alpha，禁止 No.5 严格投影')`，阻止不满足数据契约的流程继续执行。
            raise ValueError("No.6 正式输出为未投影 Alpha，禁止 No.5 严格投影")
        # 检查 `LOSS_NAME != 'sigmoid_rank_ic'`；根据条件结果选择对应处理分支。
        if LOSS_NAME != "sigmoid_rank_ic":
            # 抛出 `ValueError('No.6 只允许 sigmoid_rank_ic')`，阻止不满足数据契约的流程继续执行。
            raise ValueError("No.6 只允许 sigmoid_rank_ic")
        # 检查 `VALIDATION_METRIC != 'daily_mean_spearman_ic'`；根据条件结果选择对应处理分支。
        if VALIDATION_METRIC != "daily_mean_spearman_ic":
            # 抛出 `ValueError('No.6 早停只允许 daily_mean_spearman_ic')`，阻止不满足数据契约的流程继续执行。
            raise ValueError("No.6 早停只允许 daily_mean_spearman_ic")
        # 检查 `DUAL_TOWER_ALPHA_LOSS_WEIGHT <= 0 or DUAL_TOWER_STYLE_LOSS_WEIGHT <= 0 or DUAL_TOWER_STYLE_HIDD…`；根据条件结果选择对应处理分支。
        if (
            DUAL_TOWER_ALPHA_LOSS_WEIGHT <= 0
            or DUAL_TOWER_STYLE_LOSS_WEIGHT <= 0
            or DUAL_TOWER_STYLE_HIDDEN_DIM < 1
            or not 0 <= DUAL_TOWER_STYLE_DROPOUT < 1
        ):
            # 抛出 `ValueError('No.6 双塔损失权重或 Style 塔超参数非法')`，阻止不满足数据契约的流程继续执行。
            raise ValueError("No.6 双塔损失权重或 Style 塔超参数非法")
    # 检查 `strict_barra_output_enabled() and (not barra_conditioning_enabled())`；根据条件结果选择对应处理分支。
    if strict_barra_output_enabled() and not barra_conditioning_enabled():
        # 抛出 `ValueError('严格 Barra 输出中性化要求开启一种条件门控模式')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(
            "严格 Barra 输出中性化要求开启一种条件门控模式"
        )
    # 检查 `barra_conditioning_enabled()`；根据条件结果选择对应处理分支。
    if barra_conditioning_enabled():
        # 检查 `RESIDUALIZE_TARGET or PROJECT_MODEL_OUTPUT`；根据条件结果选择对应处理分支。
        if RESIDUALIZE_TARGET or PROJECT_MODEL_OUTPUT:
            # 抛出 `ValueError('No.4 要求 RESIDUALIZE_TARGET=False 且 PROJECT_MODEL_OUTPUT=False')`，阻止不满足数据契约的流程继续执行。
            raise ValueError(
                "No.4 要求 RESIDUALIZE_TARGET=False 且 "
                "PROJECT_MODEL_OUTPUT=False"
            )
        # 检查 `adversarial_barra_enabled()`；根据条件结果选择对应处理分支。
        if adversarial_barra_enabled():
            # 抛出 `ValueError('No.4 不能与 No.3 对抗训练同时开启')`，阻止不满足数据契约的流程继续执行。
            raise ValueError("No.4 不能与 No.3 对抗训练同时开启")
        # 检查 `LOSS_NAME != 'sigmoid_rank_ic'`；根据条件结果选择对应处理分支。
        if LOSS_NAME != "sigmoid_rank_ic":
            # 抛出 `ValueError('No.4 当前只支持 sigmoid_rank_ic')`，阻止不满足数据契约的流程继续执行。
            raise ValueError("No.4 当前只支持 sigmoid_rank_ic")
        # 检查 `strict_barra_output_enabled() and raw_barra_r2_enabled()`；根据条件结果选择对应处理分支。
        if strict_barra_output_enabled() and raw_barra_r2_enabled():
            # 抛出 `ValueError('严格 Barra 输出中性化已代数正交，RAW_BARRA_R2_LAMBDA 必须为0')`，阻止不满足数据契约的流程继续执行。
            raise ValueError(
                "严格 Barra 输出中性化已代数正交，RAW_BARRA_R2_LAMBDA 必须为0"
            )
        # 检查 `VALIDATION_METRIC != 'daily_mean_spearman_ic'`；根据条件结果选择对应处理分支。
        if VALIDATION_METRIC != "daily_mean_spearman_ic":
            # 抛出 `ValueError('No.4 的早停指标必须为 daily_mean_spearman_ic')`，阻止不满足数据契约的流程继续执行。
            raise ValueError(
                "No.4 的早停指标必须为 daily_mean_spearman_ic"
            )
        # 检查 `BARRA_GATE_HIDDEN_DIM < 1 or BARRA_INDUSTRY_EMBEDDING_DIM < 1 or (not 0 <= BARRA_GATE_DROPOUT <…`；根据条件结果选择对应处理分支。
        if (
            BARRA_GATE_HIDDEN_DIM < 1
            or BARRA_INDUSTRY_EMBEDDING_DIM < 1
            or not 0 <= BARRA_GATE_DROPOUT < 1
            or not 0 < BARRA_GATE_SCALE < 1
        ):
            # 抛出 `ValueError('No.4 条件门控超参数非法')`，阻止不满足数据契约的流程继续执行。
            raise ValueError("No.4 条件门控超参数非法")
    # 检查 `TASKS_PER_GPU < 1 or WORKERS_PER_TASK < 0`；根据条件结果选择对应处理分支。
    if TASKS_PER_GPU < 1 or WORKERS_PER_TASK < 0:
        # 抛出 `ValueError('tasks_per_gpu 必须为正整数且 workers_per_task >= 0')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("tasks_per_gpu 必须为正整数且 workers_per_task >= 0")



# 定义 `build_training_model` 函数，封装这一阶段可独立调用的处理步骤。
def build_training_model(
    device: torch.device, *, use_compile: bool
) -> torch.nn.Module:
    """按显式方案构造模型；新增结构来自短实验模块。
    
    输入：
        device: torch.device。
        use_compile: bool。
    输出：
        torch.nn.Module。
        """

    # 检查 `dual_tower_enabled()`；根据条件结果选择对应处理分支。
    if dual_tower_enabled():
        from stage04_no4_gate import (
            BarraConditionalViViT as CoreConditionalViViT,
        )
        from stage04_no6_dual_tower import (
            BarraDualTowerViViT as CoreDualTowerViViT,
        )

        # 按 `BARRA_DUAL_TOWER_MODE == 'alpha_video_only'` 在两个候选值间选择，并保存到 `alpha_mode`。
        alpha_mode = (
            "video_only"
            if BARRA_DUAL_TOWER_MODE == "alpha_video_only"
            else "real"
        )
        # 调用 `CoreConditionalViViT` 执行 `CoreConditionalViViT` 对应的明确变换，并产生后续步骤使用的结果，并将结果保存到 `alpha_tower`。
        alpha_tower = CoreConditionalViViT(
            base.OrderFlowViViT(base.MODEL_CONFIG),
            conditioning_mode=alpha_mode,
            style_count=STYLE_COUNT,
            industry_count=INDUSTRY_COUNT,
            embedding_dim=base.MODEL_CONFIG.embedding_dim,
            gate_hidden_dim=BARRA_GATE_HIDDEN_DIM,
            industry_embedding_dim=BARRA_INDUSTRY_EMBEDDING_DIM,
            dropout=BARRA_GATE_DROPOUT,
            gate_scale=BARRA_GATE_SCALE,
        )
        # 调用 `to` 把模型或张量移动到指定设备与数据类型，并将结果保存到 `model`。
        model: torch.nn.Module = CoreDualTowerViViT(
            alpha_tower,
            style_count=STYLE_COUNT,
            industry_count=INDUSTRY_COUNT,
            industry_embedding_dim=BARRA_INDUSTRY_EMBEDDING_DIM,
            style_hidden_dim=DUAL_TOWER_STYLE_HIDDEN_DIM,
            style_dropout=DUAL_TOWER_STYLE_DROPOUT,
        ).to(device)
    # 检查 `barra_conditioning_enabled()`；根据条件结果选择对应处理分支。
    elif barra_conditioning_enabled():
        from stage04_no4_gate import (
            BarraConditionalViViT as CoreConditionalViViT,
        )

        # 按 `BARRA_CONDITIONING_MODE == 'video_only'` 在两个候选值间选择，并保存到 `internal_mode`。
        internal_mode = (
            "video_only"
            if BARRA_CONDITIONING_MODE == "video_only"
            else "real"
        )
        # 调用 `to` 把模型或张量移动到指定设备与数据类型，并将结果保存到 `model`。
        model = CoreConditionalViViT(
            base.OrderFlowViViT(base.MODEL_CONFIG),
            conditioning_mode=internal_mode,
            style_count=STYLE_COUNT,
            industry_count=INDUSTRY_COUNT,
            embedding_dim=base.MODEL_CONFIG.embedding_dim,
            gate_hidden_dim=BARRA_GATE_HIDDEN_DIM,
            industry_embedding_dim=BARRA_INDUSTRY_EMBEDDING_DIM,
            dropout=BARRA_GATE_DROPOUT,
            gate_scale=BARRA_GATE_SCALE,
        ).to(device)
    # 检查 `adversarial_barra_enabled()`；根据条件结果选择对应处理分支。
    elif adversarial_barra_enabled():
        from stage04_no3_adversarial import (
            AdversarialBarraViViT as CoreAdversarialViViT,
        )

        # 调用 `to` 把模型或张量移动到指定设备与数据类型，并将结果保存到 `model`。
        model = CoreAdversarialViViT(
            base.OrderFlowViViT(base.MODEL_CONFIG),
            embedding_dim=base.MODEL_CONFIG.embedding_dim,
            style_count=STYLE_COUNT,
            industry_count=INDUSTRY_COUNT,
            hidden_dim=ADVERSARIAL_HEAD_HIDDEN_DIM,
            dropout=ADVERSARIAL_HEAD_DROPOUT,
        ).to(device)
    # 当前置条件不成立时进入备用分支，保证其余合法情况仍被处理。
    else:
        # 将 `base.build_model(device, use_compile, COMPILE_MODE, TRAIN_BATCH_MODE)` 作为当前函数结果返回给调用方。
        return base.build_model(
            device, use_compile, COMPILE_MODE, TRAIN_BATCH_MODE
        )
    # 检查 `use_compile`；根据条件结果选择对应处理分支。
    if use_compile:
        # 检查 `device.type != 'cuda'`；根据条件结果选择对应处理分支。
        if device.type != "cuda":
            # 抛出 `RuntimeError('torch.compile 训练要求 CUDA')`，阻止不满足数据契约的流程继续执行。
            raise RuntimeError("torch.compile 训练要求 CUDA")
        # 调用 `torch.compile` 编译模型计算图以减少重复执行开销，并将结果保存到 `model`。
        model = torch.compile(model, mode=COMPILE_MODE, dynamic=True)
    # 将 `model` 作为当前函数结果返回给调用方。
    return model


# 定义 `adversarial_gamma` 函数，封装这一阶段可独立调用的处理步骤。
def adversarial_gamma(epoch: int) -> float:
    """把当前训练配置传给 No.3 的余弦梯度反转调度器。

    输入：
        epoch: int，当前绝对训练轮次，从 1 开始。
    输出：
        float，当前轮次使用的梯度反转系数。
    """
    # 将 `cosine_adversarial_gamma(epoch, gamma_max=ADVERSARIAL_GAMMA_MAX, warmup_epochs=ADVERSARIAL_WARM…` 作为当前函数结果返回给调用方。
    return cosine_adversarial_gamma(
        epoch,
        gamma_max=ADVERSARIAL_GAMMA_MAX,
        warmup_epochs=ADVERSARIAL_WARMUP_EPOCHS,
        ramp_epochs=ADVERSARIAL_RAMP_EPOCHS,
    )


# 定义 `forward_training_model` 函数，封装这一阶段可独立调用的处理步骤。
def forward_training_model(
    model: torch.nn.Module,
    videos: torch.Tensor,
    frame_mask: torch.Tensor,
    *,
    gamma: float,
    styles: torch.Tensor | None = None,
    style_valid: torch.Tensor | None = None,
    industry_codes: torch.Tensor | None = None,
    sample_indices: torch.Tensor | None = None,
    conditioning_prepared: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """按实验模式执行基线、对抗、门控或双塔模型前向传播。
    
    输入：
        model: torch.nn.Module。
        videos: torch.Tensor，格式 [B,17,20,8,8]。
        frame_mask: torch.Tensor，格式 [B,20]。
        gamma: float。
        styles: torch.Tensor | None，格式 [N,10]。
        style_valid: torch.Tensor | None，格式 [N,10]。
        industry_codes: torch.Tensor | None，格式 [N]。
        sample_indices: torch.Tensor | None，格式 [N]。
    输出：
        tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]。
        """
    # 检查 `dual_tower_enabled()`；根据条件结果选择对应处理分支。
    if dual_tower_enabled():
        # 检查 `styles is None or style_valid is None or industry_codes is None or (sample_indices is None)`；根据条件结果选择对应处理分支。
        if (
            styles is None
            or style_valid is None
            or industry_codes is None
            or sample_indices is None
        ):
            # 抛出 `ValueError('No.6 前向缺少 Barra 条件输入')`，阻止不满足数据契约的流程继续执行。
            raise ValueError("No.6 前向缺少 Barra 条件输入")
        # 调用 `prepare_dual_tower_inputs` 根据当前配置和输入构造后续流程所需对象，并将结果保存到 `(alpha_inputs, style_inputs)`。
        if conditioning_prepared:
            # 执行当前分布式训练步骤并保持既定数据语义。
            alpha_inputs = style_inputs = (styles, style_valid, industry_codes)
        # 执行当前分布式训练步骤并保持既定数据语义。
        else:
            # 执行当前分布式训练步骤并保持既定数据语义。
            alpha_inputs, style_inputs = prepare_dual_tower_inputs(
                styles, style_valid, industry_codes, sample_indices,
                mode=BARRA_DUAL_TOWER_MODE,
            )
        # 调用 `model` 对当前输入 batch 执行模型前向传播，并将结果保存到 `(alpha, style, total, gate, representation)`。
        alpha, style, total, gate, representation = model(
            videos, frame_mask, *alpha_inputs, *style_inputs
        )
        # 将 `(alpha, None, None, representation, gate, style, total)` 作为当前函数结果返回给调用方。
        return alpha, None, None, representation, gate, style, total
    # 检查 `barra_conditioning_enabled()`；根据条件结果选择对应处理分支。
    if barra_conditioning_enabled():
        # 检查 `styles is None or style_valid is None or industry_codes is None or (sample_indices is None)`；根据条件结果选择对应处理分支。
        if (
            styles is None
            or style_valid is None
            or industry_codes is None
            or sample_indices is None
        ):
            # 抛出 `ValueError('No.4 前向缺少 Barra 条件输入')`，阻止不满足数据契约的流程继续执行。
            raise ValueError("No.4 前向缺少 Barra 条件输入")
        # 调用 `prepare_barra_conditioning_inputs` 根据当前配置和输入构造后续流程所需对象，并将结果保存到 `conditional`。
        if conditioning_prepared:
            # 执行当前分布式训练步骤并保持既定数据语义。
            conditional = (styles, style_valid, industry_codes)
        # 执行当前分布式训练步骤并保持既定数据语义。
        else:
            # 执行当前分布式训练步骤并保持既定数据语义。
            conditional = prepare_barra_conditioning_inputs(
                styles, style_valid, industry_codes, sample_indices,
                mode=BARRA_CONDITIONING_MODE,
            )
        # 调用 `model` 对当前输入 batch 执行模型前向传播，并将结果保存到 `(raw_score, gate, representation)`。
        raw_score, gate, representation = model(
            videos, frame_mask, *conditional
        )
        # 将 `(raw_score, None, None, representation, gate, None, None)` 作为当前函数结果返回给调用方。
        return raw_score, None, None, representation, gate, None, None
    # 检查 `adversarial_barra_enabled()`；根据条件结果选择对应处理分支。
    if adversarial_barra_enabled():
        # 调用 `torch.tensor` 转换数据表示，并将标准化对象保存到 `coefficient`。
        coefficient = torch.tensor(
            gamma, dtype=torch.float32, device=videos.device
        )
        # 调用 `model` 对当前输入 batch 执行模型前向传播，并将结果保存到 `(raw, style, industry, representation)`。
        raw, style, industry, representation = model(
            videos, frame_mask, coefficient
        )
        # 将 `(raw, style, industry, representation, None, None, None)` 作为当前函数结果返回给调用方。
        return raw, style, industry, representation, None, None, None
    # 将 `(model(videos, frame_mask), None, None, None, None, None, None)` 作为当前函数结果返回给调用方。
    return model(videos, frame_mask), None, None, None, None, None, None


# %% [markdown]
# ## 日截面数据加载与风格暴露审计
#
# 标签、十风格矩阵与 feature store 共享同一个 `sample_index`。训练、验证、
# 推理都使用完整交易日 batch。缺失暴露不前向填充，而是在当日有效样本标准化
# 后填 0，并保留有效标记。风格矩阵始终用于相关性审计；开启标签残差化时选择
# 预先生成的残差标签，开启模型分数投影时进入可微 QR 投影并由投影分数计算
# 训练损失。


# %%
# 定义 `StyleNeutralBatchCollator` 类，集中封装该阶段的数据结构与相关行为。
class StyleNeutralBatchCollator:
    """读取视频、原标签、风格残差标签和十风格暴露。
    """

    # 定义初始化方法，接收配置并建立实例后续计算所需的内部状态。
    def __init__(self) -> None:
        """初始化 StyleNeutralBatchCollator 所需的数据引用、配置和运行状态。
        
        输入：
            无显式输入参数。
        输出：
            与签名返回类型一致。
            """
        # 调用 `sequence_data.VideoBatchCollator` 把样本整理为批张量并移动到目标设备，并将结果保存到 `self.video_collator`。
        self.video_collator = sequence_data.VideoBatchCollator()
        # 设置 `self.residual_targets` 的固定配置值，供后续流程统一使用。
        self.residual_targets = None
        # 设置 `self.full_barra_targets` 的固定配置值，供后续流程统一使用。
        self.full_barra_targets = None
        # 设置 `self.dual_tower_fitted_targets` 的固定配置值，供后续流程统一使用。
        self.dual_tower_fitted_targets = None
        # 设置 `self.style_exposures` 的固定配置值，供后续流程统一使用。
        self.style_exposures = None
        # 设置 `self.style_valid` 的固定配置值，供后续流程统一使用。
        self.style_valid = None
        # 设置 `self.industry_codes` 的固定配置值，供后续流程统一使用。
        self.industry_codes = None

    # 定义内部辅助函数 `_ensure_open`，封装仅供本模块复用的计算步骤。
    def _ensure_open(self) -> None:
        """在当前 worker 内惰性打开风格、行业和派生标签 memmap。
        
        输入：
            无显式输入参数。
        输出：
            None。
            """
        # 检查 `self.residual_targets is None`；根据条件结果选择对应处理分支。
        if self.residual_targets is None:
            # 读取 `style_data.DEFAULT_STORE_DIR` 的当前值，并赋给 `store`。
            store = style_data.DEFAULT_STORE_DIR
            # 调用 `np.load` 读取源数据，并将结果保存到 `self.residual_targets`。
            self.residual_targets = np.load(
                store / style_data.RESIDUAL_TARGETS_FILE, mmap_mode="r"
            )
            # 调用 `np.load` 读取源数据，并将结果保存到 `self.style_exposures`。
            self.style_exposures = np.load(
                store / style_data.STYLE_EXPOSURES_FILE, mmap_mode="r"
            )
            # 调用 `np.load` 读取源数据，并将结果保存到 `self.style_valid`。
            self.style_valid = np.load(
                store / style_data.STYLE_VALID_FILE, mmap_mode="r"
            )
            # 检查 `barra_conditioning_enabled() or dual_tower_enabled()`；根据条件结果选择对应处理分支。
            if barra_conditioning_enabled() or dual_tower_enabled():
                # 调用 `style_data.require_full_barra_target_sidecar` 校验输入、配置或产物是否满足当前阶段的数据契约。
                style_data.require_full_barra_target_sidecar()
                # 调用 `np.load` 读取源数据，并将结果保存到 `self.full_barra_targets`。
                self.full_barra_targets = np.load(
                    style_data.DEFAULT_FULL_BARRA_TARGET_STORE_DIR
                    / style_data.FULL_BARRA_TARGETS_FILE,
                    mmap_mode="r",
                )
            # 检查 `dual_tower_enabled()`；根据条件结果选择对应处理分支。
            if dual_tower_enabled():
                # 调用 `style_data.require_dual_tower_target_sidecar` 校验输入、配置或产物是否满足当前阶段的数据契约。
                style_data.require_dual_tower_target_sidecar()
                # 调用 `np.load` 读取源数据，并将结果保存到 `self.dual_tower_fitted_targets`。
                self.dual_tower_fitted_targets = np.load(
                    style_data.DEFAULT_DUAL_TOWER_TARGET_STORE_DIR
                    / style_data.DUAL_TOWER_FITTED_TARGETS_FILE,
                    mmap_mode="r",
                )
            # 检查 `industry_sidecar_required()`；根据条件结果选择对应处理分支。
            if industry_sidecar_required():
                # 调用 `style_data.require_industry_sidecar` 校验输入、配置或产物是否满足当前阶段的数据契约。
                style_data.require_industry_sidecar()
                # 调用 `np.load` 读取源数据，并将结果保存到 `self.industry_codes`。
                self.industry_codes = np.load(
                    style_data.DEFAULT_INDUSTRY_STORE_DIR
                    / style_data.INDUSTRY_CODES_FILE, mmap_mode="r"
                )

    # 定义内部辅助函数 `__call__`，封装仅供本模块复用的计算步骤。
    def __call__(self, sample_indices: list[int]) -> dict[str, torch.Tensor]:
        """按样本行号组装视频、Barra 条件、派生标签和真实审计暴露。
        
        输入：
            sample_indices: list[int]，格式 [N]。
        输出：
            dict[str, torch.Tensor]。
            """
        # 调用 `self._ensure_open` 执行 `_ensure_open` 对应的明确变换，并产生后续步骤使用的结果。
        self._ensure_open()
        # 调用 `self.video_collator` 把样本整理为批张量并移动到目标设备，并将结果保存到 `batch`。
        batch = self.video_collator(sample_indices)
        # 调用 `batch['sample_indices'].numpy().astype` 统一设备或数据类型，并将结果保存到 `indices`。
        indices = batch["sample_indices"].numpy().astype(np.int64, copy=False)
        # 读取 `batch['targets']` 的当前值，并赋给 `batch['original_targets']`。
        batch["original_targets"] = batch["targets"]
        # 调用 `torch.from_numpy` 转换数据表示，并将标准化对象保存到 `batch['residual_targets']`。
        batch["residual_targets"] = torch.from_numpy(
            np.array(self.residual_targets[indices], dtype=np.float32, copy=True)
        )
        # 检查 `RESIDUALIZE_TARGET`；根据条件结果选择对应处理分支。
        if RESIDUALIZE_TARGET:
            # 读取 `batch['residual_targets']` 的当前值，并赋给 `batch['targets']`。
            batch["targets"] = batch["residual_targets"]
        # 检查 `barra_conditioning_enabled() or dual_tower_enabled()`；根据条件结果选择对应处理分支。
        if barra_conditioning_enabled() or dual_tower_enabled():
            # 调用 `torch.from_numpy` 转换数据表示，并将标准化对象保存到 `batch['targets']`。
            batch["targets"] = torch.from_numpy(
                np.array(
                    self.full_barra_targets[indices],
                    dtype=np.float32,
                    copy=True,
                )
            )
        # 检查 `dual_tower_enabled()`；根据条件结果选择对应处理分支。
        if dual_tower_enabled():
            # 调用 `torch.from_numpy` 转换数据表示，并将标准化对象保存到 `batch['dual_tower_fitted_targets']`。
            batch["dual_tower_fitted_targets"] = torch.from_numpy(
                np.array(
                    self.dual_tower_fitted_targets[indices],
                    dtype=np.float32,
                    copy=True,
                )
            )
        # 当前置条件不成立时进入备用分支，保证其余合法情况仍被处理。
        else:
            # 调用 `torch.zeros_like` 按目标形状初始化数值容器 `batch['dual_tower_fitted_targets']`。
            batch["dual_tower_fitted_targets"] = torch.zeros_like(
                batch["targets"]
            )
        # 调用 `torch.from_numpy` 转换数据表示，并将标准化对象保存到 `batch['style_exposures']`。
        batch["style_exposures"] = torch.from_numpy(
            np.array(self.style_exposures[indices], dtype=np.float32, copy=True)
        )
        # 调用 `torch.from_numpy` 转换数据表示，并将标准化对象保存到 `batch['style_valid']`。
        batch["style_valid"] = torch.from_numpy(
            np.array(self.style_valid[indices], dtype=np.bool_, copy=True)
        )
        # 检查 `industry_sidecar_required()`；根据条件结果选择对应处理分支。
        if industry_sidecar_required():
            # 调用 `torch.from_numpy` 转换数据表示，并将标准化对象保存到 `batch['industry_codes']`。
            batch["industry_codes"] = torch.from_numpy(
                np.array(self.industry_codes[indices], dtype=np.int16, copy=True)
            )
        # 当前置条件不成立时进入备用分支，保证其余合法情况仍被处理。
        else:
            # 调用 `torch.zeros` 按目标形状初始化数值容器 `batch['industry_codes']`。
            batch["industry_codes"] = torch.zeros(
                len(indices), dtype=torch.int16
            )
        # 将 `batch` 作为当前函数结果返回给调用方。
        return batch


# 定义 `make_style_loaders` 函数，封装这一阶段可独立调用的处理步骤。
def make_style_loaders(
    forecast_year: int,
    seed: int,
    workers: int,
    pin_memory: bool,
    distributed_context: distributed.DistributedContext | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    """训练、验证和预测全部按完整交易日组织 batch。
    
    输入：
        forecast_year: int。
        seed: int。
        workers: int。
        pin_memory: bool。
    输出：
        tuple[DataLoader, DataLoader, DataLoader | None]。
        """

    # 调用 `sequence_data.load_sequence_split` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `split`。
    split = sequence_data.load_sequence_split(forecast_year)
    # 调用 `np.load` 读取源数据，并将结果保存到 `date_ids`。
    date_ids = np.load(sequence_data.DATE_IDS_PATH, mmap_mode="r")
    # 调用 `torch.Generator().manual_seed` 创建并固定 DataLoader 使用的随机数生成器，并将结果保存到 `generator`。
    generator = torch.Generator().manual_seed(seed)
    # 构造键值映射，并保存到 `style_common`，供后续按键查询。
    style_common: dict[str, Any] = {
        "num_workers": workers,
        "pin_memory": pin_memory,
        "collate_fn": StyleNeutralBatchCollator(),
    }
    # 检查 `workers > 0`；根据条件结果选择对应处理分支。
    if workers > 0:
        # 用本轮键值结果更新 `style_common`，补齐后续查询所需字段。
        style_common.update({"persistent_workers": True, "prefetch_factor": 1})

    # 定义 `style_loader` 函数，封装这一阶段可独立调用的处理步骤。
    def style_loader(indices: np.ndarray, *, shuffle: bool) -> DataLoader:
        """处理连续风格数据、有效掩码或相应诊断。
        
        输入：
            indices: np.ndarray。
            shuffle: bool。
        输出：
            DataLoader。
            """
        # 调用 `TradeDateBatchSampler` 按交易日完整截面构造 batch 采样器，并将结果保存到 `sampler`。
        sampler_class = (
            distributed.DistributedTradeDateBatchSampler
            if distributed_context is not None else TradeDateBatchSampler
        )
        # 执行当前分布式训练步骤并保持既定数据语义。
        distributed_kwargs = (
            {"rank": distributed_context.rank, "world_size": distributed_context.world_size}
            if distributed_context is not None else {}
        )
        # 执行当前分布式训练步骤并保持既定数据语义。
        sampler = sampler_class(
            indices, date_ids, seed=seed, shuffle=shuffle, **distributed_kwargs
        )
        # 将 `DataLoader(sequence_data.SampleIndexDataset(indices), batch_sampler=sampler, generator=generato…` 作为当前函数结果返回给调用方。
        return DataLoader(
            sequence_data.SampleIndexDataset(indices),
            batch_sampler=sampler,
            generator=generator,
            **style_common,
        )

    # 定义 `raw_prediction_loader` 函数，封装这一阶段可独立调用的处理步骤。
    def raw_prediction_loader(indices: np.ndarray) -> DataLoader:
        """处理预测文件、预测列或对应的样本数量审计。
        
        输入：
            indices: np.ndarray。
        输出：
            DataLoader。
            """
        # 构造键值映射，并保存到 `raw_common`，供后续按键查询。
        raw_common: dict[str, Any] = {
            "num_workers": workers,
            "pin_memory": pin_memory,
            "collate_fn": sequence_data.VideoBatchCollator(),
        }
        # 检查 `workers > 0`；根据条件结果选择对应处理分支。
        if workers > 0:
            # 用本轮键值结果更新 `raw_common`，补齐后续查询所需字段。
            raw_common.update(
                {"persistent_workers": True, "prefetch_factor": 1}
            )
        # 调用 `TradeDateBatchSampler` 按交易日完整截面构造 batch 采样器，并将结果保存到 `sampler`。
        sampler_class = (
            distributed.DistributedTradeDateBatchSampler
            if distributed_context is not None else TradeDateBatchSampler
        )
        # 执行当前分布式训练步骤并保持既定数据语义。
        distributed_kwargs = (
            {"rank": distributed_context.rank, "world_size": distributed_context.world_size}
            if distributed_context is not None else {}
        )
        # 执行当前分布式训练步骤并保持既定数据语义。
        sampler = sampler_class(
            indices, date_ids, seed=seed, shuffle=False, **distributed_kwargs
        )
        # 将 `DataLoader(sequence_data.SampleIndexDataset(indices), batch_sampler=sampler, generator=generato…` 作为当前函数结果返回给调用方。
        return DataLoader(
            sequence_data.SampleIndexDataset(indices),
            batch_sampler=sampler,
            generator=generator,
            **raw_common,
        )

    # 调用 `style_loader` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `train`。
    train = style_loader(split["train_indices"], shuffle=True)
    # 调用 `style_loader` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `validation`。
    validation = style_loader(split["validation_indices"], shuffle=False)
    # 设置 `prediction` 的固定配置值，供后续流程统一使用。
    prediction = None
    # 检查 `len(split['prediction_indices'])`；根据条件结果选择对应处理分支。
    if len(split["prediction_indices"]):
        # 按 `not barra_conditioning_enabled() and (not dual_tower_enabled()) and (raw_barra_r2_enabled() or …` 在两个候选值间选择，并保存到 `prediction`。
        prediction = (
            raw_prediction_loader(split["prediction_indices"])
            if (
                not barra_conditioning_enabled()
                and not dual_tower_enabled()
                and (raw_barra_r2_enabled() or adversarial_barra_enabled())
            )
            else style_loader(split["prediction_indices"], shuffle=False)
        )
    # 将 `(train, validation, prediction)` 作为当前函数结果返回给调用方。
    return train, validation, prediction


# 定义 `training_industry_class_weights` 函数，封装这一阶段可独立调用的处理步骤。
def training_industry_class_weights(train_indices: np.ndarray) -> torch.Tensor:
    """由当前滚动训练区间生成32行业逆平方根频数权重。
    
    输入：
        train_indices: np.ndarray。
    输出：
        torch.Tensor。
        """

    # 调用 `style_data.require_industry_sidecar` 校验输入、配置或产物是否满足当前阶段的数据契约。
    style_data.require_industry_sidecar()
    # 调用 `np.load` 读取源数据，并将结果保存到 `codes`。
    codes = np.load(
        style_data.DEFAULT_INDUSTRY_STORE_DIR / style_data.INDUSTRY_CODES_FILE,
        mmap_mode="r",
    )
    # 调用 `np.asarray` 转换数据表示，并将标准化对象保存到 `selected`。
    selected = np.asarray(codes[train_indices], dtype=np.int64)
    # 计算 `selected[selected > 0] - 1`，并将数值结果保存到 `known`。
    known = selected[selected > 0] - 1
    # 调用 `np.bincount(known, minlength=INDUSTRY_COUNT).astype` 统一设备或数据类型，并将结果保存到 `counts`。
    counts = np.bincount(known, minlength=INDUSTRY_COUNT).astype(np.float64)
    # 检查 `len(counts) != INDUSTRY_COUNT or (counts <= 0).any()`；根据条件结果选择对应处理分支。
    if len(counts) != INDUSTRY_COUNT or (counts <= 0).any():
        # 抛出 `ValueError('滚动训练区间未覆盖全部32个真实行业')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("滚动训练区间未覆盖全部32个真实行业")
    # 调用 `np.reciprocal` 计算样本数平方根的倒数作为加权系数，并将结果保存到 `weights`。
    weights = np.reciprocal(np.sqrt(counts))
    # 用 `weights.mean()` 对 `weights` 做 `Div` 累积更新。
    weights /= weights.mean()
    # 将 `torch.from_numpy(weights.astype(np.float32))` 作为当前函数结果返回给调用方。
    return torch.from_numpy(weights.astype(np.float32))






# 定义 `StyleCudaPrefetcher` 类，集中封装该阶段的数据结构与相关行为。
class StyleCudaPrefetcher:
    """将视频、双标签和风格矩阵一起异步搬到 GPU。
    """

    # 定义初始化方法，接收配置并建立实例后续计算所需的内部状态。
    def __init__(self, loader: DataLoader, device: torch.device, enabled: bool) -> None:
        """初始化 StyleCudaPrefetcher 所需的数据引用、配置和运行状态。
        
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
    ) -> tuple[torch.Tensor, ...]:
        """把风格训练批次的所有张量非阻塞搬到 CUDA。
        
        输入：
            batch: dict[str, torch.Tensor]。
        输出：
            与签名返回类型一致。
            """
        # 根据 `self.device.type == 'cuda'` 计算条件结果，并保存到 `non_blocking`。
        non_blocking = self.device.type == "cuda"
        # 组织一组有序值，并保存到 `names`。
        names = (
            "videos", "frame_mask", "targets", "original_targets",
            "residual_targets", "dual_tower_fitted_targets",
            "style_exposures", "style_valid",
            "industry_codes",
        )
        # 将 `tuple((batch[name].to(self.device, non_blocking=non_blocking) for name in names))` 作为当前函数结果返回给调用方。
        return tuple(
            batch[name].to(self.device, non_blocking=non_blocking)
            for name in names
        )

    # 定义内部辅助函数 `__iter__`，封装仅供本模块复用的计算步骤。
    def __iter__(self) -> Iterator[tuple[Any, ...]]:
        """按既定顺序迭代 StyleCudaPrefetcher 中的训练批次。
        
        输入：
            无显式输入参数。
        输出：
            与签名返回类型一致。
            """
        # 检查 `not self.enabled`；根据条件结果选择对应处理分支。
        if not self.enabled:
            # 遍历 `self.loader`，并把当前元素绑定到 `batch` 逐项处理。
            for batch in self.loader:
                # 对 `(yield (batch, *self._move(batch)))` 求值，以触发当前步骤需要的计算。
                yield (batch, *self._move(batch))
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
                # 调用 `self._move` 执行 `_move` 对应的明确变换，并产生后续步骤使用的结果，并将结果保存到 `tensors`。
                tensors = self._move(batch)
            # 组织一组有序值，并保存到 `next_item`。
            next_item = (batch, *tensors)

        # 调用 `preload` 把下一批数据异步预取到 GPU。
        preload()
        # 只要 `next_item is not None` 成立就继续循环，直至满足退出条件。
        while next_item is not None:
            # 调用 `torch.cuda.current_stream(self.device).wait_stream` 等待预取 stream 完成后再消费张量。
            torch.cuda.current_stream(self.device).wait_stream(stream)
            # 读取 `next_item` 的当前值，并赋给 `current`。
            current = next_item
            # 遍历 `current[1:]`，并把当前元素绑定到 `tensor` 逐项处理。
            for tensor in current[1:]:
                # 调用 `tensor.record_stream` 把张量登记到当前 CUDA stream，避免异步使用期间被回收。
                tensor.record_stream(torch.cuda.current_stream(self.device))
            # 调用 `preload` 把下一批数据异步预取到 GPU。
            preload()
            # 对 `(yield current)` 求值，以触发当前步骤需要的计算。
            yield current


# 定义 `training_config` 函数，封装这一阶段可独立调用的处理步骤。
def training_config() -> base.TrainingConfig:
    """汇总当前风格中性实验的结构、标签、损失和早停配置。
    
    输入：
        无显式输入参数。
    输出：
        base.TrainingConfig。
        """
    # 将 `base.TrainingConfig(train_batch_mode=TRAIN_BATCH_MODE, loss_name=LOSS_NAME, rank_temperature=RA…` 作为当前函数结果返回给调用方。
    return base.TrainingConfig(
        train_batch_mode=TRAIN_BATCH_MODE,
        loss_name=LOSS_NAME,
        rank_temperature=RANK_TEMPERATURE,
        turnover_lambda=TURNOVER_LAMBDA,
        huber_delta=HUBER_DELTA,
        smooth_l1_beta=SMOOTH_L1_BETA,
        hybrid_weight=HYBRID_WEIGHT,
        validation_metric=VALIDATION_METRIC,
        learning_rate=LEARNING_RATE,
        max_epochs=MAX_EPOCHS,
        early_stopping_patience=PATIENCE,
    )


# 定义 `training_loss` 函数，封装这一阶段可独立调用的处理步骤。
def training_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """现有预测损失；sigmoid soft rank 对预测端保持反向传播。
    
    输入：
        predictions: torch.Tensor，格式 [N]。
        targets: torch.Tensor，格式 [N]。
    输出：
        torch.Tensor。
        """

    # 将 `compute_training_loss(predictions, targets, loss_name=LOSS_NAME, rank_temperature=RANK_TEMPERAT…` 作为当前函数结果返回给调用方。
    return compute_training_loss(
        predictions,
        targets,
        loss_name=LOSS_NAME,
        rank_temperature=RANK_TEMPERATURE,
        turnover_lambda=TURNOVER_LAMBDA,
        huber_delta=HUBER_DELTA,
        smooth_l1_beta=SMOOTH_L1_BETA,
        hybrid_weight=HYBRID_WEIGHT,
    )


# 定义 `rank_loss` 函数，封装这一阶段可独立调用的处理步骤。
def rank_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """兼容旧测试和调用方的训练损失别名。
    
    输入：
        predictions: torch.Tensor，格式 [N]。
        targets: torch.Tensor，格式 [N]。
    输出：
        torch.Tensor。
        """

    # 将 `training_loss(predictions, targets)` 作为当前函数结果返回给调用方。
    return training_loss(predictions, targets)


# 定义 `stable_sigmoid_rank_loss` 函数，封装这一阶段可独立调用的处理步骤。
def stable_sigmoid_rank_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """与现有 Sigmoid SoftRankIC 等价，并稳定支持常数初始分数。
    
    输入：
        predictions: torch.Tensor，格式 [N]。
        targets: torch.Tensor，格式 [N]。
    输出：
        torch.Tensor。
        """

    # 调用 `sigmoid_soft_rank` 根据分数计算截面排序或排序索引，并将结果保存到 `prediction_ranks`。
    prediction_ranks = sigmoid_soft_rank(
        predictions, RANK_TEMPERATURE
    )
    # 在 `torch.no_grad()` 管理的上下文中执行本段逻辑，并保证资源按约定释放。
    with torch.no_grad():
        # 调用 `sigmoid_soft_rank` 根据分数计算截面排序或排序索引，并将结果保存到 `target_ranks`。
        target_ranks = sigmoid_soft_rank(
            targets.detach(), RANK_TEMPERATURE
        )
    # 计算 `prediction_ranks - prediction_ranks.mean()`，并将数值结果保存到 `centered_prediction`。
    centered_prediction = prediction_ranks - prediction_ranks.mean()
    # 计算 `target_ranks - target_ranks.mean()`，并将数值结果保存到 `centered_target`。
    centered_target = target_ranks - target_ranks.mean()
    # 计算 `centered_prediction.square().sum() * centered_target.square().sum()`，并将数值结果保存到 `variance_product`。
    variance_product = (
        centered_prediction.square().sum()
        * centered_target.square().sum()
    )
    # 调用 `torch.sqrt` 对输入开平方得到范数或标准差相关量，并将结果保存到 `denominator`。
    denominator = torch.sqrt(variance_product.clamp_min(1e-8))
    # 计算 `centered_prediction.mul(centered_target).sum() / denominator`，并将数值结果保存到 `correlation`。
    correlation = (
        centered_prediction.mul(centered_target).sum() / denominator
    )
    # 将 `-correlation` 作为当前函数结果返回给调用方。
    return -correlation


# 定义 `masked_style_regression_loss` 函数，封装这一阶段可独立调用的处理步骤。
def masked_style_regression_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """仅在有效暴露位置计算十风格对抗回归损失。
    
    输入：
        predictions: torch.Tensor，格式 [N]。
        targets: torch.Tensor，格式 [N]。
        valid: torch.Tensor。
    输出：
        torch.Tensor。
        """
    # 检查 `predictions.shape != targets.shape or valid.shape != targets.shape`；根据条件结果选择对应处理分支。
    if predictions.shape != targets.shape or valid.shape != targets.shape:
        # 抛出 `ValueError('风格对抗头、标签或有效掩码形状不一致')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("风格对抗头、标签或有效掩码形状不一致")
    # 调用 `valid.bool` 统一设备或数据类型，并将结果保存到 `mask`。
    mask = valid.bool()
    # 检查 `not mask.any()`；根据条件结果选择对应处理分支。
    if not mask.any():
        # 抛出 `ValueError('当前日批次没有有效风格监督')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("当前日批次没有有效风格监督")
    # 读取 `(predictions.float() - targets.float()).square()[mask]` 的当前值，并赋给 `values`。
    values = (predictions.float() - targets.float()).square()[mask]
    # 检查 `not torch.isfinite(values).all()`；根据条件结果选择对应处理分支。
    if not torch.isfinite(values).all():
        # 抛出 `FloatingPointError('风格对抗损失含 NaN/Inf')`，阻止不满足数据契约的流程继续执行。
        raise FloatingPointError("风格对抗损失含 NaN/Inf")
    # 将 `values.mean()` 作为当前函数结果返回给调用方。
    return values.mean()


# 定义 `normalized_industry_classification_loss` 函数，封装这一阶段可独立调用的处理步骤。
def normalized_industry_classification_loss(
    logits: torch.Tensor,
    industry_codes: torch.Tensor,
    class_weights: torch.Tensor,
) -> torch.Tensor:
    """计算忽略 UNKNOWN 且按行业加权归一化的分类损失。
    
    输入：
        logits: torch.Tensor。
        industry_codes: torch.Tensor，格式 [N]。
        class_weights: torch.Tensor。
    输出：
        torch.Tensor。
        """
    # 检查 `logits.ndim != 2 or logits.shape[1] != INDUSTRY_COUNT`；根据条件结果选择对应处理分支。
    if logits.ndim != 2 or logits.shape[1] != INDUSTRY_COUNT:
        # 抛出 `ValueError('行业对抗头必须输出 [N,32]')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("行业对抗头必须输出 [N,32]")
    # 调用 `industry_codes.long().reshape` 调整张量维度，并将新视图保存到 `codes`。
    codes = industry_codes.long().reshape(-1)
    # 检查 `len(codes) != len(logits)`；根据条件结果选择对应处理分支。
    if len(codes) != len(logits):
        # 抛出 `ValueError('行业标签与对抗头行数不一致')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("行业标签与对抗头行数不一致")
    # 根据 `codes > 0` 计算条件结果，并保存到 `known`。
    known = codes > 0
    # 检查 `not known.any()`；根据条件结果选择对应处理分支。
    if not known.any():
        # 抛出 `ValueError('当前日批次没有已知行业监督')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("当前日批次没有已知行业监督")
    # 计算 `codes[known] - 1`，并将数值结果保存到 `targets`。
    targets = codes[known] - 1
    # 检查 `int(targets.min()) < 0 or int(targets.max()) >= INDUSTRY_COUNT`；根据条件结果选择对应处理分支。
    if int(targets.min()) < 0 or int(targets.max()) >= INDUSTRY_COUNT:
        # 抛出 `ValueError('行业标签超出1至32的有效范围')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("行业标签超出1至32的有效范围")
    # 调用 `class_weights.to` 统一设备或数据类型，并将结果保存到 `weights`。
    weights = class_weights.to(device=logits.device, dtype=torch.float32)
    # 检查 `weights.shape != (INDUSTRY_COUNT,) or not torch.isfinite(weights).all()`；根据条件结果选择对应处理分支。
    if weights.shape != (INDUSTRY_COUNT,) or not torch.isfinite(weights).all():
        # 抛出 `ValueError('行业类别权重形状或数值异常')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("行业类别权重形状或数值异常")
    # 计算 `torch.nn.functional.cross_entropy(logits.float()[known], targets, weight=weights) / math.log(IN…`，并将数值结果保存到 `loss`。
    loss = torch.nn.functional.cross_entropy(
        logits.float()[known], targets, weight=weights
    ) / math.log(INDUSTRY_COUNT)
    # 检查 `not torch.isfinite(loss)`；根据条件结果选择对应处理分支。
    if not torch.isfinite(loss):
        # 抛出 `FloatingPointError('行业对抗损失含 NaN/Inf')`，阻止不满足数据契约的流程继续执行。
        raise FloatingPointError("行业对抗损失含 NaN/Inf")
    # 将 `loss` 作为当前函数结果返回给调用方。
    return loss


# 定义 `adversarial_training_loss` 函数，封装这一阶段可独立调用的处理步骤。
def adversarial_training_loss(
    style_predictions: torch.Tensor | None,
    industry_logits: torch.Tensor | None,
    styles: torch.Tensor,
    style_valid: torch.Tensor,
    industry_codes: torch.Tensor,
    class_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """组合收益排序、连续风格对抗和行业对抗损失。
    
    输入：
        style_predictions: torch.Tensor | None，格式 [N,10]。
        industry_logits: torch.Tensor | None，格式 [N,32]。
        styles: torch.Tensor，格式 [N,10]。
        style_valid: torch.Tensor，格式 [N,10]。
        industry_codes: torch.Tensor，格式 [N]。
        class_weights: torch.Tensor。
    输出：
        tuple[torch.Tensor, torch.Tensor]。
        """
    # 调用 `torch.zeros` 按目标形状初始化数值容器 `zero`。
    zero = torch.zeros((), dtype=torch.float32, device=styles.device)
    # 检查 `not adversarial_barra_enabled()`；根据条件结果选择对应处理分支。
    if not adversarial_barra_enabled():
        # 将 `(zero, zero)` 作为当前函数结果返回给调用方。
        return zero, zero
    # 检查 `style_predictions is None or industry_logits is None`；根据条件结果选择对应处理分支。
    if style_predictions is None or industry_logits is None:
        # 抛出 `ValueError('No.3 缺少风格或行业对抗头输出')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("No.3 缺少风格或行业对抗头输出")
    # 将 `(masked_style_regression_loss(style_predictions, styles, style_valid), normalized_industry_clas…` 作为当前函数结果返回给调用方。
    return (
        masked_style_regression_loss(
            style_predictions, styles, style_valid
        ),
        normalized_industry_classification_loss(
            industry_logits, industry_codes, class_weights
        ),
    )


# 定义内部辅助函数 `_joint_r_squared_value`，封装仅供本模块复用的计算步骤。
def _joint_r_squared_value(
    predictions: torch.Tensor,
    styles: torch.Tensor,
    industries: torch.Tensor,
) -> torch.Tensor:
    """把 Barra R² 公共接口收窄为标量，供方案损失组合。
    
    输入：
        predictions: torch.Tensor，格式 [N]。
        styles: torch.Tensor，格式 [N,10]。
        industries: torch.Tensor。
    输出：
        torch.Tensor。
        """

    from stage02_barra_algebra import joint_barra_r_squared

    # 调用 `joint_barra_r_squared` 按股票和交易日键对齐来自不同来源的数据，并将结果保存到 `(value, _)`。
    value, _ = joint_barra_r_squared(
        predictions, styles, industries
    )
    # 将 `value` 作为当前函数结果返回给调用方。
    return value


# 定义 `combined_training_loss` 函数，封装这一阶段可独立调用的处理步骤。
def combined_training_loss(
    raw_predictions: torch.Tensor,
    effective: torch.Tensor,
    targets: torch.Tensor,
    style_exposures: torch.Tensor,
    industry_codes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回总损失、预测损失和 raw-score 联合 Barra R²。
    
    输入：
        raw_predictions: torch.Tensor，格式 [N]。
        effective: torch.Tensor，格式 [N]。
        targets: torch.Tensor，格式 [N]。
        style_exposures: torch.Tensor，格式 [N,10]。
        industry_codes: torch.Tensor，格式 [N]。
    输出：
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]。
        """

    # 调用 `training_loss` 按当前实验口径计算可反向传播的训练损失，并将结果保存到 `predictive_loss`。
    predictive_loss = training_loss(effective, targets)
    # 检查 `not raw_barra_r2_enabled()`；根据条件结果选择对应处理分支。
    if not raw_barra_r2_enabled():
        # 调用 `torch.zeros` 按目标形状初始化数值容器 `barra_r_squared`。
        barra_r_squared = torch.zeros(
            (), dtype=torch.float32, device=raw_predictions.device
        )
        # 将 `(predictive_loss, predictive_loss, barra_r_squared)` 作为当前函数结果返回给调用方。
        return predictive_loss, predictive_loss, barra_r_squared

    from stage04_no2_r2_penalty import (
        r_squared_regularized_loss,
    )

    # 调用 `r_squared_regularized_loss` 按当前实验口径计算可反向传播的训练损失，并将结果保存到 `components`。
    components = r_squared_regularized_loss(
        raw_predictions,
        targets,
        style_exposures,
        industry_codes,
        rank_loss=lambda _score, _target: predictive_loss,
        barra_r_squared=_joint_r_squared_value,
        penalty_weight=RAW_BARRA_R2_LAMBDA,
    )
    # 将 `(components['loss'], components['predictive_loss'], components['barra_r_squared'])` 作为当前函数结果返回给调用方。
    return (
        components["loss"],
        components["predictive_loss"],
        components["barra_r_squared"],
    )



# 定义 `dual_tower_training_loss` 函数，封装这一阶段可独立调用的处理步骤。
def dual_tower_training_loss(
    alpha_score: torch.Tensor,
    style_score: torch.Tensor,
    total_score: torch.Tensor,
    original_targets: torch.Tensor,
    residual_targets: torch.Tensor,
    fitted_targets: torch.Tensor,
    style_exposures: torch.Tensor,
    industry_codes: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """No.6 四项损失；风险惩罚只施加于 Alpha 塔。
    
    输入：
        alpha_score: torch.Tensor。
        style_score: torch.Tensor。
        total_score: torch.Tensor。
        original_targets: torch.Tensor。
        residual_targets: torch.Tensor。
        fitted_targets: torch.Tensor。
        style_exposures: torch.Tensor，格式 [N,10]。
        industry_codes: torch.Tensor，格式 [N]。
    输出：
        dict[str, torch.Tensor]。
        """

    # 检查 `not dual_tower_enabled()`；根据条件结果选择对应处理分支。
    if not dual_tower_enabled():
        # 抛出 `RuntimeError('仅 No.6 可以调用双塔损失')`，阻止不满足数据契约的流程继续执行。
        raise RuntimeError("仅 No.6 可以调用双塔损失")
    from stage04_no6_dual_tower import (
        dual_tower_training_loss as compute_dual_tower_loss,
    )

    # 将 `compute_dual_tower_loss(alpha_score, style_score, total_score, original_targets, residual_targe…` 作为当前函数结果返回给调用方。
    return compute_dual_tower_loss(
        alpha_score,
        style_score,
        total_score,
        original_targets,
        residual_targets,
        fitted_targets,
        style_exposures,
        industry_codes,
        return_rank_loss=training_loss,
        alpha_rank_loss=training_loss,
        style_rank_loss=stable_sigmoid_rank_loss,
        barra_r_squared=_joint_r_squared_value,
        alpha_loss_weight=DUAL_TOWER_ALPHA_LOSS_WEIGHT,
        style_loss_weight=DUAL_TOWER_STYLE_LOSS_WEIGHT,
        r_squared_weight=RAW_BARRA_R2_LAMBDA,
    )



# 定义 `FullBarraOutputNeutralizationLayer` 类，集中封装该阶段的数据结构与相关行为。
class FullBarraOutputNeutralizationLayer(torch.nn.Module):
    """无参数的每日等权完整 Barra 可微中性化层。
    """

    # 定义 `forward` 函数，封装这一阶段可独立调用的处理步骤。
    def forward(
        self,
        raw_predictions: torch.Tensor,
        style_exposures: torch.Tensor,
        industry_codes: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """执行 FullBarraOutputNeutralizationLayer 的前向计算并返回约定张量。
        
        输入：
            raw_predictions: torch.Tensor，格式 [N]。
            style_exposures: torch.Tensor，格式 [N,10]。
            industry_codes: torch.Tensor，格式 [N]。
        输出：
            tuple[torch.Tensor, dict[str, torch.Tensor]]。
            """
        # 将 `style_data.project_out_full_barra_torch(raw_predictions, style_exposures, industry_codes, stand…` 作为当前函数结果返回给调用方。
        return style_data.project_out_full_barra_torch(
            raw_predictions,
            style_exposures,
            industry_codes,
            standardize=True,
        )


# 调用 `FullBarraOutputNeutralizationLayer` 按 Barra 暴露执行截面投影或残差化处理，并将结果保存到 `FULL_BARRA_OUTPUT_NEUTRALIZER`。
FULL_BARRA_OUTPUT_NEUTRALIZER = FullBarraOutputNeutralizationLayer()


# 定义 `effective_predictions` 函数，封装这一阶段可独立调用的处理步骤。
def effective_predictions(
    raw_predictions: torch.Tensor,
    style_exposures: torch.Tensor,
    industry_codes: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
    """按实验开关返回损失实际使用及正式发布的分数口径。
    
    输入：
        raw_predictions: torch.Tensor，格式 [N]。
        style_exposures: torch.Tensor，格式 [N,10]。
        industry_codes: torch.Tensor | None，格式 [N]。
    输出：
        tuple[torch.Tensor, dict[str, torch.Tensor] | None]。
        """
    # 检查 `strict_barra_output_enabled()`；根据条件结果选择对应处理分支。
    if strict_barra_output_enabled():
        # 检查 `industry_codes is None`；根据条件结果选择对应处理分支。
        if industry_codes is None:
            # 抛出 `ValueError('严格 Barra 输出中性化缺少行业代码')`，阻止不满足数据契约的流程继续执行。
            raise ValueError("严格 Barra 输出中性化缺少行业代码")
        # 调用 `FULL_BARRA_OUTPUT_NEUTRALIZER` 按 Barra 暴露执行截面投影或残差化处理，并将结果保存到 `(output, diagnostics)`。
        output, diagnostics = FULL_BARRA_OUTPUT_NEUTRALIZER(
            raw_predictions,
            style_exposures,
            industry_codes,
        )
        # 检查 `float(diagnostics['max_abs_style_pearson']) > 5e-05 or float(diagnostics['max_abs_industry_mean…`；根据条件结果选择对应处理分支。
        if (
            float(diagnostics["max_abs_style_pearson"]) > 5e-5
            or float(diagnostics["max_abs_industry_mean"]) > 5e-5
        ):
            # 抛出 `FloatingPointError('严格 Barra 输出未达到十风格或行业数值正交')`，阻止不满足数据契约的流程继续执行。
            raise FloatingPointError(
                "严格 Barra 输出未达到十风格或行业数值正交"
            )
        # 将 `(output, diagnostics)` 作为当前函数结果返回给调用方。
        return output, diagnostics
    # 检查 `not PROJECT_MODEL_OUTPUT`；根据条件结果选择对应处理分支。
    if not PROJECT_MODEL_OUTPUT:
        # 将 `(raw_predictions.float().reshape(-1), None)` 作为当前函数结果返回给调用方。
        return raw_predictions.float().reshape(-1), None
    # 将 `style_data.project_out_styles_torch(raw_predictions, style_exposures, standardize=True)` 作为当前函数结果返回给调用方。
    return style_data.project_out_styles_torch(
        raw_predictions, style_exposures, standardize=True
    )


# %% [markdown]
# ## 训练、验证和推理
#
# 训练损失和早停指标都使用两个开关选择后的有效标签与有效模型分数。另行记录
# 未投影模型分数对原始标签的 RankIC，作为跨模式一致的审计指标，不参与损失
# 或早停。


# %%
# 定义 `train_one_epoch` 函数，封装这一阶段可独立调用的处理步骤。
def prepare_distributed_conditioning(
    styles: torch.Tensor,
    style_valid: torch.Tensor,
    industry_codes: torch.Tensor,
    sample_indices: torch.Tensor,
    context: distributed.DistributedContext | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """执行 prepare_distributed_conditioning 对应的分布式训练处理。
    
    输入：
        参数见函数类型标注。
    输出：
        结果见函数返回类型标注。
    """

    # 执行当前分布式训练步骤并保持既定数据语义。
    shuffled = (
        dual_tower_enabled() and BARRA_DUAL_TOWER_MODE == "shuffled"
    ) or (
        barra_conditioning_enabled() and BARRA_CONDITIONING_MODE == "shuffled"
    )
    # 执行当前分布式训练步骤并保持既定数据语义。
    if context is None or not shuffled:
        # 执行当前分布式训练步骤并保持既定数据语义。
        return styles, style_valid, industry_codes, False
    # 执行当前分布式训练步骤并保持既定数据语义。
    local_indices = sample_indices.to(styles.device)
    # 执行当前分布式训练步骤并保持既定数据语义。
    global_indices = distributed.all_gather_tensor(local_indices)
    # 执行当前分布式训练步骤并保持既定数据语义。
    global_styles = distributed.all_gather_tensor(styles)
    # 执行当前分布式训练步骤并保持既定数据语义。
    global_valid = distributed.all_gather_tensor(style_valid)
    # 执行当前分布式训练步骤并保持既定数据语义。
    global_industry = distributed.all_gather_tensor(industry_codes)
    # 执行当前分布式训练步骤并保持既定数据语义。
    order = torch.argsort(global_indices)
    # 执行当前分布式训练步骤并保持既定数据语义。
    global_indices = global_indices[order]
    # 执行当前分布式训练步骤并保持既定数据语义。
    global_styles = global_styles[order]
    # 执行当前分布式训练步骤并保持既定数据语义。
    global_valid = global_valid[order]
    # 执行当前分布式训练步骤并保持既定数据语义。
    global_industry = global_industry[order]
    # 执行当前分布式训练步骤并保持既定数据语义。
    mode = (
        BARRA_DUAL_TOWER_MODE
        if dual_tower_enabled() else BARRA_CONDITIONING_MODE
    )
    # 执行当前分布式训练步骤并保持既定数据语义。
    transformed = prepare_barra_conditioning_inputs(
        global_styles,
        global_valid,
        global_industry,
        global_indices,
        mode=mode,
    )
    # 执行当前分布式训练步骤并保持既定数据语义。
    positions = torch.searchsorted(global_indices, local_indices)
    # 执行当前分布式训练步骤并保持既定数据语义。
    if not torch.equal(global_indices[positions], local_indices):
        # 执行当前分布式训练步骤并保持既定数据语义。
        raise AssertionError("DDP Barra 条件输入无法按 sample_index 对齐")
    # 执行当前分布式训练步骤并保持既定数据语义。
    return (
        transformed[0][positions],
        transformed[1][positions],
        transformed[2][positions],
        True,
    )


# 执行当前分布式训练步骤并保持既定数据语义。
def _gather_optional_cross_section(
    value: torch.Tensor | None,
    order: torch.Tensor,
    context: distributed.DistributedContext | None,
    *,
    differentiable: bool,
) -> torch.Tensor | None:
    """执行 _gather_optional_cross_section 对应的分布式训练处理。
    
    输入：
        参数见函数类型标注。
    输出：
        结果见函数返回类型标注。
    """
    # 执行当前分布式训练步骤并保持既定数据语义。
    if value is None:
        # 执行当前分布式训练步骤并保持既定数据语义。
        return None
    # 执行当前分布式训练步骤并保持既定数据语义。
    if context is None:
        # 执行当前分布式训练步骤并保持既定数据语义。
        return value
    # 执行当前分布式训练步骤并保持既定数据语义。
    gathered = (
        distributed.differentiable_all_gather(value)
        if differentiable else distributed.all_gather_tensor(value)
    )
    # 执行当前分布式训练步骤并保持既定数据语义。
    return gathered[order]


# 执行当前分布式训练步骤并保持既定数据语义。
def gather_style_cross_section(
    raw_prediction: torch.Tensor,
    style_prediction: torch.Tensor | None,
    industry_logits: torch.Tensor | None,
    gate: torch.Tensor | None,
    dual_style_score: torch.Tensor | None,
    total_return_score: torch.Tensor | None,
    targets: torch.Tensor,
    original_targets: torch.Tensor,
    residual_targets: torch.Tensor,
    fitted_targets: torch.Tensor,
    styles: torch.Tensor,
    style_valid: torch.Tensor,
    industry_codes: torch.Tensor,
    batch: dict[str, torch.Tensor],
    context: distributed.DistributedContext | None,
    *,
    differentiable_outputs: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """执行 gather_style_cross_section 对应的分布式训练处理。
    
    输入：
        参数见函数类型标注。
    输出：
        结果见函数返回类型标注。
    """
    # 执行当前分布式训练步骤并保持既定数据语义。
    sample_indices = batch["sample_indices"].to(raw_prediction.device)
    # 执行当前分布式训练步骤并保持既定数据语义。
    date_ids = batch["date_ids"].to(raw_prediction.device)
    # 执行当前分布式训练步骤并保持既定数据语义。
    if context is None:
        # 执行当前分布式训练步骤并保持既定数据语义。
        order = torch.arange(len(sample_indices), device=sample_indices.device)
        # 执行当前分布式训练步骤并保持既定数据语义。
        global_indices = sample_indices
        # 执行当前分布式训练步骤并保持既定数据语义。
        global_dates = date_ids
    # 执行当前分布式训练步骤并保持既定数据语义。
    else:
        # 执行当前分布式训练步骤并保持既定数据语义。
        global_indices = distributed.all_gather_tensor(sample_indices)
        # 执行当前分布式训练步骤并保持既定数据语义。
        global_dates = distributed.all_gather_tensor(date_ids)
        # 执行当前分布式训练步骤并保持既定数据语义。
        order = torch.argsort(global_indices)
        # 执行当前分布式训练步骤并保持既定数据语义。
        global_indices = global_indices[order]
        # 执行当前分布式训练步骤并保持既定数据语义。
        global_dates = global_dates[order]

    # 执行当前分布式训练步骤并保持既定数据语义。
    def metadata(value: torch.Tensor) -> torch.Tensor:
        """执行 metadata 对应的分布式训练处理。
        
        输入：
            参数见函数类型标注。
        输出：
            结果见函数返回类型标注。
        """
        # 执行当前分布式训练步骤并保持既定数据语义。
        if context is None:
            # 执行当前分布式训练步骤并保持既定数据语义。
            return value
        # 执行当前分布式训练步骤并保持既定数据语义。
        return distributed.all_gather_tensor(value)[order]

    # 执行当前分布式训练步骤并保持既定数据语义。
    return (
        _gather_optional_cross_section(
            raw_prediction, order, context,
            differentiable=differentiable_outputs,
        ),
        _gather_optional_cross_section(
            style_prediction, order, context,
            differentiable=differentiable_outputs,
        ),
        _gather_optional_cross_section(
            industry_logits, order, context,
            differentiable=differentiable_outputs,
        ),
        _gather_optional_cross_section(
            gate.detach() if gate is not None else None,
            order,
            context,
            differentiable=False,
        ),
        _gather_optional_cross_section(
            dual_style_score, order, context,
            differentiable=differentiable_outputs,
        ),
        _gather_optional_cross_section(
            total_return_score, order, context,
            differentiable=differentiable_outputs,
        ),
        metadata(targets),
        metadata(original_targets),
        metadata(residual_targets),
        metadata(fitted_targets),
        metadata(styles),
        metadata(style_valid),
        metadata(industry_codes),
        global_indices,
        global_dates,
    )


# 执行当前分布式训练步骤并保持既定数据语义。
def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    epoch: int,
    industry_class_weights: torch.Tensor,
    max_batches: int | None = None,
    distributed_context: distributed.DistributedContext | None = None,
) -> dict[str, float | int]:
    """训练一个完整 epoch，并返回损失与批次诊断。
    
    输入：
        model: torch.nn.Module。
        loader: DataLoader。
        optimizer: torch.optim.Optimizer。
        device: torch.device。
        epoch: int。
        industry_class_weights: torch.Tensor。
        max_batches: int | None。
    输出：
        dict[str, float | int]。
        """
    # 调用 `model.train` 将模型切换到训练模式，启用训练期行为。
    model.train()
    # 按 `adversarial_barra_enabled()` 在两个候选值间选择，并保存到 `gamma`。
    gamma = adversarial_gamma(epoch) if adversarial_barra_enabled() else 0.0
    # 构造键值映射，并保存到 `totals`，供后续按键查询。
    totals = {
        "loss": 0.0,
        "rank_loss": 0.0,
        "return_rank_loss": 0.0,
        "alpha_rank_loss": 0.0,
        "dual_style_rank_loss": 0.0,
        "barra_r_squared": 0.0,
        "effective_pearson": 0.0,
        "original_pearson": 0.0,
        "raw_style_max": 0.0,
        "raw_style_mean": 0.0,
        "effective_style_max": 0.0,
        "effective_style_mean": 0.0,
        "adversarial_style_loss": 0.0,
        "adversarial_industry_loss": 0.0,
        "adversarial_industry_accuracy": 0.0,
        "adversarial_unknown_rate": 0.0,
        "gate_mean_abs": 0.0,
        "gate_max_abs": 0.0,
    }
    # 设置 `samples` 的固定配置值，供后续流程统一使用。
    samples = 0
    # 设置 `batches` 的固定配置值，供后续流程统一使用。
    batches = 0
    # 调用 `time.perf_counter` 读取高精度单调时钟以计算耗时，并将结果保存到 `started`。
    started = time.perf_counter()
    # 遍历 `enumerate(StyleCudaPrefetcher(loader, device, bool(USE_CUDA_PREFETCH)), start=1)`，并把当前元素绑定到 `(batch_number, item)` 逐项处理。
    for batch_number, item in enumerate(
        StyleCudaPrefetcher(loader, device, bool(USE_CUDA_PREFETCH)), start=1
    ):
        # 读取 `item` 的当前值，并赋给 `(batch, videos, frame_mask, targets, original_targets, _residual_target…`。
        (
            batch, videos, frame_mask, targets, original_targets,
            _residual_targets, fitted_targets, styles, style_valid,
            industry_codes,
        ) = item
        # 读取 `batch['date_ids']` 的当前值，并赋给 `date_ids`。
        date_ids = batch["date_ids"]
        # 检查 `int(date_ids.min()) != int(date_ids.max())`；根据条件结果选择对应处理分支。
        if int(date_ids.min()) != int(date_ids.max()):
            # 抛出 `AssertionError('训练 batch 混入多个交易日')`，阻止不满足数据契约的流程继续执行。
            raise AssertionError("训练 batch 混入多个交易日")
        # 检查 `not torch.isfinite(targets).all()`；根据条件结果选择对应处理分支。
        if not torch.isfinite(targets).all():
            # 抛出 `FloatingPointError('训练目标含 NaN/Inf')`，阻止不满足数据契约的流程继续执行。
            raise FloatingPointError("训练目标含 NaN/Inf")

        # 调用 `optimizer.zero_grad` 清空上一轮梯度，避免跨批次意外累积。
        (
            conditioning_styles,
            conditioning_valid,
            conditioning_industry,
            conditioning_prepared,
        ) = prepare_distributed_conditioning(
            styles, style_valid, industry_codes, batch["sample_indices"], distributed_context
        )
        # 执行当前分布式训练步骤并保持既定数据语义。
        optimizer.zero_grad(set_to_none=True)
        # 在 `base.autocast_context(device)` 管理的上下文中执行本段逻辑，并保证资源按约定释放。
        with base.autocast_context(device):
            # 调用 `forward_training_model` 对当前 batch 执行模型前向传播，并将结果保存到 `(raw_predictions, style_predictions, industry_logits, _representation, …`。
            (
                raw_predictions,
                style_predictions,
                industry_logits,
                _representation,
                gate,
                dual_style_score,
                total_return_score,
            ) = forward_training_model(
                model,
                videos,
                frame_mask,
                gamma=gamma,
                styles=conditioning_styles,
                style_valid=conditioning_valid,
                industry_codes=conditioning_industry,
                sample_indices=batch["sample_indices"],
                conditioning_prepared=conditioning_prepared,
            )
        # 执行当前分布式训练步骤并保持既定数据语义。
        (
            raw_predictions,
            style_predictions,
            industry_logits,
            gate,
            dual_style_score,
            total_return_score,
            targets,
            original_targets,
            _residual_targets,
            fitted_targets,
            styles,
            style_valid,
            industry_codes,
            _global_sample_indices,
            _global_date_ids,
        ) = gather_style_cross_section(
            raw_predictions,
            style_predictions,
            industry_logits,
            gate,
            dual_style_score,
            total_return_score,
            targets,
            original_targets,
            _residual_targets,
            fitted_targets,
            styles,
            style_valid,
            industry_codes,
            batch,
            distributed_context,
            differentiable_outputs=True,
        )
        # 调用 `effective_predictions` 执行模型推理并生成样本预测分数，并将结果保存到 `(predictions, _)`。
        predictions, _ = effective_predictions(
            raw_predictions, styles, industry_codes
        )
        # 调用 `measure_style_correlations` 计算预测与标签或风险暴露之间的相关性，并将结果保存到 `raw_style`。
        raw_style = measure_style_correlations(raw_predictions, styles)
        # 调用 `measure_style_correlations` 计算预测与标签或风险暴露之间的相关性，并将结果保存到 `effective_style`。
        effective_style = measure_style_correlations(predictions, styles)
        # 检查 `dual_tower_enabled()`；根据条件结果选择对应处理分支。
        if dual_tower_enabled():
            # 检查 `dual_style_score is None or total_return_score is None`；根据条件结果选择对应处理分支。
            if dual_style_score is None or total_return_score is None:
                # 抛出 `ValueError('No.6 缺少 Style 或 Total 分数')`，阻止不满足数据契约的流程继续执行。
                raise ValueError("No.6 缺少 Style 或 Total 分数")
            # 调用 `dual_tower_training_loss` 按当前实验口径计算可反向传播的训练损失，并将结果保存到 `dual_losses`。
            dual_losses = dual_tower_training_loss(
                raw_predictions,
                dual_style_score,
                total_return_score,
                original_targets,
                targets,
                fitted_targets,
                styles,
                industry_codes,
            )
            # 读取 `dual_losses['loss']` 的当前值，并赋给 `loss`。
            loss = dual_losses["loss"]
            # 读取 `dual_losses['alpha_rank_loss']` 的当前值，并赋给 `predictive_loss`。
            predictive_loss = dual_losses["alpha_rank_loss"]
            # 读取 `dual_losses['alpha_barra_r_squared']` 的当前值，并赋给 `barra_r_squared`。
            barra_r_squared = dual_losses["alpha_barra_r_squared"]
            # 读取 `dual_losses['return_rank_loss']` 的当前值，并赋给 `return_rank_loss`。
            return_rank_loss = dual_losses["return_rank_loss"]
            # 读取 `dual_losses['alpha_rank_loss']` 的当前值，并赋给 `alpha_rank_loss`。
            alpha_rank_loss = dual_losses["alpha_rank_loss"]
            # 读取 `dual_losses['style_rank_loss']` 的当前值，并赋给 `dual_style_rank_loss`。
            dual_style_rank_loss = dual_losses["style_rank_loss"]
        # 当前置条件不成立时进入备用分支，保证其余合法情况仍被处理。
        else:
            # 调用 `combined_training_loss` 按当前实验口径计算可反向传播的训练损失，并将结果保存到 `(loss, predictive_loss, barra_r_squared)`。
            loss, predictive_loss, barra_r_squared = combined_training_loss(
                raw_predictions, predictions, targets, styles, industry_codes
            )
            # 调用 `torch.zeros` 按目标形状初始化数值容器 `zero`。
            zero = torch.zeros((), dtype=torch.float32, device=device)
            # 读取 `zero` 的当前值，并赋给 `return_rank_loss`。
            return_rank_loss = zero
            # 读取 `zero` 的当前值，并赋给 `alpha_rank_loss`。
            alpha_rank_loss = zero
            # 读取 `zero` 的当前值，并赋给 `dual_style_rank_loss`。
            dual_style_rank_loss = zero
        # 调用 `adversarial_training_loss` 按当前实验口径计算可反向传播的训练损失，并将结果保存到 `(style_loss, industry_loss)`。
        style_loss, industry_loss = adversarial_training_loss(
            style_predictions,
            industry_logits,
            styles,
            style_valid,
            industry_codes,
            industry_class_weights,
        )
        # 计算 `loss + ADVERSARIAL_STYLE_LOSS_WEIGHT * style_loss + ADVERSARIAL_INDUSTRY_LOSS_WEIGHT * industry…`，并将数值结果保存到 `loss`。
        loss = (
            loss
            + ADVERSARIAL_STYLE_LOSS_WEIGHT * style_loss
            + ADVERSARIAL_INDUSTRY_LOSS_WEIGHT * industry_loss
        )
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
            # 检查 `adversarial_barra_enabled() or barra_conditioning_enabled() or dual_tower_enabled()`；根据条件结果选择对应处理分支。
            if (
                adversarial_barra_enabled()
                or barra_conditioning_enabled()
                or dual_tower_enabled()
            ):
                # 调用 `style_data.joint_barra_r_squared_torch` 按股票和交易日键对齐来自不同来源的数据，并将结果保存到 `(barra_r_squared, _)`。
                barra_r_squared, _ = style_data.joint_barra_r_squared_torch(
                    raw_predictions.detach(), styles, industry_codes
                )
            # 调用 `pearson_correlation` 计算预测与标签或风险暴露之间的相关性，并将结果保存到 `effective_pearson`。
            effective_pearson = pearson_correlation(predictions, targets)
            # 调用 `pearson_correlation` 计算预测与标签或风险暴露之间的相关性，并将结果保存到 `original_pearson`。
            original_pearson = pearson_correlation(
                (
                    total_return_score.float()
                    if dual_tower_enabled() else raw_predictions.float()
                ),
                original_targets,
            )
            # 检查 `industry_logits is not None`；根据条件结果选择对应处理分支。
            if industry_logits is not None:
                # 根据 `industry_codes > 0` 计算条件结果，并保存到 `known`。
                known = industry_codes > 0
                # 调用 `mean` 计算当前张量或数组的算术均值，并将结果保存到 `industry_accuracy`。
                industry_accuracy = (
                    industry_logits[known].argmax(dim=1)
                    == industry_codes[known].long() - 1
                ).float().mean()
            # 当前置条件不成立时进入备用分支，保证其余合法情况仍被处理。
            else:
                # 调用 `torch.zeros` 按目标形状初始化数值容器 `industry_accuracy`。
                industry_accuracy = torch.zeros((), device=device)
        # 用 `float(loss.detach())` 对 `totals['loss']` 做 `Add` 累积更新。
        totals["loss"] += float(loss.detach())
        # 用 `float(predictive_loss.detach())` 对 `totals['rank_loss']` 做 `Add` 累积更新。
        totals["rank_loss"] += float(predictive_loss.detach())
        # 用 `float(return_rank_loss.detach())` 对 `totals['return_rank_loss']` 做 `Add` 累积更新。
        totals["return_rank_loss"] += float(return_rank_loss.detach())
        # 用 `float(alpha_rank_loss.detach())` 对 `totals['alpha_rank_loss']` 做 `Add` 累积更新。
        totals["alpha_rank_loss"] += float(alpha_rank_loss.detach())
        # 用 `float(dual_style_rank_loss.detach())` 对 `totals['dual_style_rank_loss']` 做 `Add` 累积更新。
        totals["dual_style_rank_loss"] += float(
            dual_style_rank_loss.detach()
        )
        # 用 `float(barra_r_squared.detach())` 对 `totals['barra_r_squared']` 做 `Add` 累积更新。
        totals["barra_r_squared"] += float(barra_r_squared.detach())
        # 用 `float(effective_pearson)` 对 `totals['effective_pearson']` 做 `Add` 累积更新。
        totals["effective_pearson"] += float(effective_pearson)
        # 用 `float(original_pearson)` 对 `totals['original_pearson']` 做 `Add` 累积更新。
        totals["original_pearson"] += float(original_pearson)
        # 调用 `max` 汇总当前数据，并将统计量保存到 `totals['raw_style_max']`。
        totals["raw_style_max"] = max(
            totals["raw_style_max"],
            float(raw_style["max_abs_style_pearson"]),
        )
        # 用 `float(raw_style['mean_abs_style_pearson'])` 对 `totals['raw_style_mean']` 做 `Add` 累积更新。
        totals["raw_style_mean"] += float(
            raw_style["mean_abs_style_pearson"]
        )
        # 调用 `max` 汇总当前数据，并将统计量保存到 `totals['effective_style_max']`。
        totals["effective_style_max"] = max(
            totals["effective_style_max"],
            float(effective_style["max_abs_style_pearson"]),
        )
        # 用 `float(effective_style['mean_abs_style_pearson'])` 对 `totals['effective_style_mean']` 做 `Add` 累积更新。
        totals["effective_style_mean"] += float(
            effective_style["mean_abs_style_pearson"]
        )
        # 用 `float(style_loss.detach())` 对 `totals['adversarial_style_loss']` 做 `Add` 累积更新。
        totals["adversarial_style_loss"] += float(style_loss.detach())
        # 用 `float(industry_loss.detach())` 对 `totals['adversarial_industry_loss']` 做 `Add` 累积更新。
        totals["adversarial_industry_loss"] += float(industry_loss.detach())
        # 用 `float(industry_accuracy)` 对 `totals['adversarial_industry_accuracy']` 做 `Add` 累积更新。
        totals["adversarial_industry_accuracy"] += float(industry_accuracy)
        # 用 `float((industry_codes == 0).float().mean())` 对 `totals['adversarial_unknown_rate']` 做 `Add` 累积更新。
        totals["adversarial_unknown_rate"] += float(
            (industry_codes == 0).float().mean()
        )
        # 检查 `gate is not None`；根据条件结果选择对应处理分支。
        if gate is not None:
            # 调用 `gate.detach().float().abs` 逐元素取绝对值，并将结果保存到 `gate_abs`。
            gate_abs = gate.detach().float().abs()
            # 用 `float(gate_abs.mean())` 对 `totals['gate_mean_abs']` 做 `Add` 累积更新。
            totals["gate_mean_abs"] += float(gate_abs.mean())
            # 调用 `max` 汇总当前数据，并将统计量保存到 `totals['gate_max_abs']`。
            totals["gate_max_abs"] = max(
                totals["gate_max_abs"], float(gate_abs.max())
            )
        # 用 `len(targets)` 对 `samples` 做 `Add` 累积更新。
        samples += len(targets)
        # 用 `1` 对 `batches` 做 `Add` 累积更新。
        batches += 1
        # 检查 `max_batches is not None and batch_number >= max_batches`；根据条件结果选择对应处理分支。
        if max_batches is not None and batch_number >= max_batches:
            # 终止当前循环，避免继续处理已经不满足条件的元素。
            break

    # 检查 `not batches`；根据条件结果选择对应处理分支。
    if not batches:
        # 抛出 `ValueError('训练 loader 没有可用 batch')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("训练 loader 没有可用 batch")
    # 检查 `device.type == 'cuda'`；根据条件结果选择对应处理分支。
    if device.type == "cuda":
        # 调用 `torch.cuda.synchronize` 等待当前 GPU 队列执行完毕后再计时或验收。
        torch.cuda.synchronize(device)
    # 将 `{'train_loss': totals['loss'] / batches, 'train_rank_loss': totals['rank_loss'] / batches, 'tra…` 作为当前函数结果返回给调用方。
    return {
        "train_loss": totals["loss"] / batches,
        "train_rank_loss": totals["rank_loss"] / batches,
        "train_return_rank_loss": totals["return_rank_loss"] / batches,
        "train_alpha_rank_loss": totals["alpha_rank_loss"] / batches,
        "train_dual_style_rank_loss": (
            totals["dual_style_rank_loss"] / batches
        ),
        "train_barra_r_squared": totals["barra_r_squared"] / batches,
        "train_effective_target_pearson_ic": (
            totals["effective_pearson"] / batches
        ),
        "train_original_target_pearson_ic": (
            totals["original_pearson"] / batches
        ),
        "train_raw_max_abs_style_pearson": totals["raw_style_max"],
        "train_raw_mean_abs_style_pearson": (
            totals["raw_style_mean"] / batches
        ),
        "train_effective_max_abs_style_pearson": (
            totals["effective_style_max"]
        ),
        "train_effective_mean_abs_style_pearson": (
            totals["effective_style_mean"] / batches
        ),
        "train_adversarial_style_loss": (
            totals["adversarial_style_loss"] / batches
        ),
        "train_adversarial_industry_loss": (
            totals["adversarial_industry_loss"] / batches
        ),
        "train_adversarial_industry_accuracy": (
            totals["adversarial_industry_accuracy"] / batches
        ),
        "train_adversarial_unknown_rate": (
            totals["adversarial_unknown_rate"] / batches
        ),
        "train_adversarial_gamma": gamma,
        "train_gate_mean_abs": totals["gate_mean_abs"] / batches,
        "train_gate_max_abs": totals["gate_max_abs"],
        "train_samples": samples,
        "train_date_batches": batches,
        "train_seconds": time.perf_counter() - started,
    }


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
        # 调用 `pd.read_parquet(sequence_data.SAMPLE_INDEX_PATH, columns=['inst…` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `_INSTRUMENT_IDS`。
        _INSTRUMENT_IDS = pd.read_parquet(
            sequence_data.SAMPLE_INDEX_PATH, columns=["instrument_id"]
        )["instrument_id"].to_numpy(copy=False)
    # 将 `_INSTRUMENT_IDS` 作为当前函数结果返回给调用方。
    return _INSTRUMENT_IDS


# 定义 has_degenerate_daily_pearson 函数，统一复用验证 Pearson 的退化判据。
def has_degenerate_daily_pearson(
    predictions: np.ndarray,
    targets: np.ndarray,
    date_ids: np.ndarray,
) -> bool:
    """判断是否存在无法计算 Pearson 的交易日截面。

    输入：
        predictions: np.ndarray，逐样本预测。
        targets: np.ndarray，逐样本目标。
        date_ids: np.ndarray，逐样本交易日编号。
    输出：
        bool，存在近常数预测或目标时为 True。
    """
    # 调用 np.asarray 统一预测数组的形状和精度。
    x = np.asarray(predictions, dtype=np.float64).reshape(-1)
    # 调用 np.asarray 统一目标数组的形状和精度。
    y = np.asarray(targets, dtype=np.float64).reshape(-1)
    # 调用 np.asarray 统一交易日数组的形状和类型。
    dates = np.asarray(date_ids, dtype=np.int64).reshape(-1)
    # 检查三组输入等长且至少包含两个样本。
    if not (len(x) == len(y) == len(dates)) or len(x) < 2:
        # 抛出长度错误，避免把样本错位误判为分数塌缩。
        raise ValueError("Style Pearson 预测、目标和日期必须等长且至少有两个样本")
    # 检查预测和目标均为有限值。
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        # 保持验证数据的 NaN/Inf 严格失败契约。
        raise FloatingPointError("Style Pearson 预测或目标包含 NaN/Inf")
    # 遍历每个交易日，复用正式验证函数的同一 Pearson 分母阈值。
    for date in np.unique(dates):
        # 计算当前交易日的样本掩码。
        mask = dates == date
        # 检查当前交易日至少包含两个样本。
        if int(mask.sum()) < 2:
            # 抛出截面样本不足错误，不将数据契约问题降级为诊断值。
            raise ValueError("Style Pearson 存在不足两个样本的交易日")
        # 尝试按正式验证口径计算 Pearson。
        try:
            # 调用 numpy_pearson 检查当前截面的统一数值判据。
            numpy_pearson(x[mask], y[mask])
        # 仅捕获分母退化错误，将其标记为 Style 塔塌缩。
        except FloatingPointError:
            # 返回 True，让调用方记录 NaN 审计指标而不中断主训练。
            return True
    # 所有交易日均可计算 Pearson，返回未塌缩。
    return False


# 使用 `torch.no_grad()` 包装 `evaluate_model`，并定义其可复用的处理逻辑。
@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    industry_class_weights: torch.Tensor,
    max_batches: int | None = None,
    distributed_context: distributed.DistributedContext | None = None,
) -> dict[str, float | int]:
    """在验证集上评估模型，并汇总逐日 IC 与损失诊断。
    
    输入：
        model: torch.nn.Module。
        loader: DataLoader。
        device: torch.device。
        industry_class_weights: torch.Tensor。
        max_batches: int | None。
        distributed_context: distributed.DistributedContext | None。
    输出：
        dict[str, float | int]。
        """
    # 调用 `model.eval` 将模型切换到评估模式，关闭训练期随机行为。
    local_model = base.inference_model(model)
    # 执行当前分布式训练步骤并保持既定数据语义。
    local_model.eval()
    # 构造当前步骤所需的列表，并保存到 `effective_predictions_parts`。
    effective_predictions_parts: list[np.ndarray] = []
    # 构造当前步骤所需的列表，并保存到 `raw_predictions_parts`。
    raw_predictions_parts: list[np.ndarray] = []
    # 构造当前步骤所需的列表，并保存到 `effective_targets`。
    effective_targets: list[np.ndarray] = []
    # 构造当前步骤所需的列表，并保存到 `original_targets`。
    original_targets: list[np.ndarray] = []
    # 构造当前步骤所需的列表，并保存到 `dual_style_predictions`。
    dual_style_predictions: list[np.ndarray] = []
    # 构造当前步骤所需的列表，并保存到 `dual_total_predictions`。
    dual_total_predictions: list[np.ndarray] = []
    # 构造当前步骤所需的列表，并保存到 `dual_fitted_targets`。
    dual_fitted_targets: list[np.ndarray] = []
    # 构造当前步骤所需的列表，并保存到 `date_ids`。
    date_ids: list[np.ndarray] = []
    # 构造当前步骤所需的列表，并保存到 `sample_indices`。
    sample_indices: list[np.ndarray] = []
    # 构造当前步骤所需的列表，并保存到 `adversarial_style_predictions`。
    adversarial_style_predictions: list[np.ndarray] = []
    # 构造当前步骤所需的列表，并保存到 `adversarial_style_targets`。
    adversarial_style_targets: list[np.ndarray] = []
    # 构造当前步骤所需的列表，并保存到 `adversarial_style_valid`。
    adversarial_style_valid: list[np.ndarray] = []
    # 构造当前步骤所需的列表，并保存到 `adversarial_industry_logits`。
    adversarial_industry_logits: list[np.ndarray] = []
    # 构造当前步骤所需的列表，并保存到 `adversarial_industry_codes`。
    adversarial_industry_codes: list[np.ndarray] = []
    # 设置 `raw_style_max` 的固定配置值，供后续流程统一使用。
    raw_style_max = 0.0
    # 设置 `raw_style_mean` 的固定配置值，供后续流程统一使用。
    raw_style_mean = 0.0
    # 设置 `effective_style_max` 的固定配置值，供后续流程统一使用。
    effective_style_max = 0.0
    # 设置 `effective_style_mean` 的固定配置值，供后续流程统一使用。
    effective_style_mean = 0.0
    # 设置 `barra_r_squared_sum` 的固定配置值，供后续流程统一使用。
    barra_r_squared_sum = 0.0
    # 设置 `barra_r_squared_max` 的固定配置值，供后续流程统一使用。
    barra_r_squared_max = 0.0
    # 设置 `effective_barra_r_squared_sum` 的固定配置值，供后续流程统一使用。
    effective_barra_r_squared_sum = 0.0
    # 设置 `effective_barra_r_squared_max` 的固定配置值，供后续流程统一使用。
    effective_barra_r_squared_max = 0.0
    # 设置 `effective_industry_mean_max` 的固定配置值，供后续流程统一使用。
    effective_industry_mean_max = 0.0
    # 设置 `gate_mean_abs` 的固定配置值，供后续流程统一使用。
    gate_mean_abs = 0.0
    # 设置 `gate_max_abs` 的固定配置值，供后续流程统一使用。
    gate_max_abs = 0.0
    # 读取 `np.inf` 的当前值，并赋给 `minimum_daily_raw_score_std`。
    minimum_daily_raw_score_std = np.inf
    # 设置 `batches` 的固定配置值，供后续流程统一使用。
    batches = 0
    # 调用 `time.perf_counter` 读取高精度单调时钟以计算耗时，并将结果保存到 `started`。
    started = time.perf_counter()

    # 遍历 `enumerate(StyleCudaPrefetcher(loader, device, bool(USE_CUDA_PREFETCH)), start=1)`，并把当前元素绑定到 `(batch_number, item)` 逐项处理。
    for batch_number, item in enumerate(
        StyleCudaPrefetcher(loader, device, bool(USE_CUDA_PREFETCH)), start=1
    ):
        # 读取 `item` 的当前值，并赋给 `(batch, videos, frame_mask, targets, original, _residual, fitted_target…`。
        (
            batch, videos, frame_mask, targets, original, _residual,
            fitted_targets, styles, style_valid, industry_codes,
        ) = item
        # 检查 `int(batch['date_ids'].min()) != int(batch['date_ids'].max())`；根据条件结果选择对应处理分支。
        if int(batch["date_ids"].min()) != int(batch["date_ids"].max()):
            # 抛出 `AssertionError('验证 batch 混入多个交易日')`，阻止不满足数据契约的流程继续执行。
            raise AssertionError("验证 batch 混入多个交易日")
        # 在 `base.autocast_context(device)` 管理的上下文中执行本段逻辑，并保证资源按约定释放。
        (
            conditioning_styles,
            conditioning_valid,
            conditioning_industry,
            conditioning_prepared,
        ) = prepare_distributed_conditioning(
            styles, style_valid, industry_codes, batch["sample_indices"], distributed_context
        )
        # 执行当前分布式训练步骤并保持既定数据语义。
        with base.autocast_context(device):
            # 调用 `forward_training_model` 对当前 batch 执行模型前向传播，并将结果保存到 `(raw_prediction, style_prediction, industry_logits, _representation, ga…`。
            (
                raw_prediction,
                style_prediction,
                industry_logits,
                _representation,
                gate,
                dual_style_score,
                total_return_score,
            ) = forward_training_model(
                local_model,
                videos,
                frame_mask,
                gamma=0.0,
                styles=conditioning_styles,
                style_valid=conditioning_valid,
                industry_codes=conditioning_industry,
                sample_indices=batch["sample_indices"],
                conditioning_prepared=conditioning_prepared,
            )
        # 执行当前分布式训练步骤并保持既定数据语义。
        (
            raw_prediction,
            style_prediction,
            industry_logits,
            gate,
            dual_style_score,
            total_return_score,
            targets,
            original,
            _residual,
            fitted_targets,
            styles,
            style_valid,
            industry_codes,
            global_sample_indices,
            global_date_ids,
        ) = gather_style_cross_section(
            raw_prediction,
            style_prediction,
            industry_logits,
            gate,
            dual_style_score,
            total_return_score,
            targets,
            original,
            _residual,
            fitted_targets,
            styles,
            style_valid,
            industry_codes,
            batch,
            distributed_context,
            differentiable_outputs=False,
        )
        # 调用 `effective_predictions` 执行模型推理并生成样本预测分数，并将结果保存到 `(prediction, projection_diagnostics)`。
        prediction, projection_diagnostics = effective_predictions(
            raw_prediction, styles, industry_codes
        )
        # 检查 `strict_barra_output_enabled()`；根据条件结果选择对应处理分支。
        if strict_barra_output_enabled():
            # 检查 `projection_diagnostics is None`；根据条件结果选择对应处理分支。
            if projection_diagnostics is None:
                # 抛出 `AssertionError('严格输出缺少投影诊断')`，阻止不满足数据契约的流程继续执行。
                raise AssertionError("严格输出缺少投影诊断")
            # 调用 `max` 汇总当前数据，并将统计量保存到 `effective_industry_mean_max`。
            effective_industry_mean_max = max(
                effective_industry_mean_max,
                float(projection_diagnostics["max_abs_industry_mean"]),
            )
        # 调用 `int` 统一设备或数据类型，并将结果保存到 `date_id`。
        date_id = int(batch["date_ids"][0])
        # 调用 `measure_daily_score_dispersion` 按函数定义的数值口径计算当前统计量，并将结果保存到 `raw_score_dispersion`。
        raw_score_dispersion = measure_daily_score_dispersion(
            raw_prediction,
            date_id=date_id,
        )
        # 调用 `min` 汇总当前数据，并将统计量保存到 `minimum_daily_raw_score_std`。
        minimum_daily_raw_score_std = min(
            minimum_daily_raw_score_std,
            float(raw_score_dispersion["score_std"].item()),
        )
        # 调用 `measure_style_correlations` 计算预测与标签或风险暴露之间的相关性，并将结果保存到 `raw_style`。
        raw_style = measure_style_correlations(raw_prediction, styles)
        # 调用 `measure_style_correlations` 计算预测与标签或风险暴露之间的相关性，并将结果保存到 `effective_style`。
        effective_style = measure_style_correlations(prediction, styles)
        # 检查 `industry_sidecar_required()`；根据条件结果选择对应处理分支。
        if industry_sidecar_required():
            # 调用 `style_data.joint_barra_r_squared_torch` 按股票和交易日键对齐来自不同来源的数据，并将结果保存到 `(barra_r_squared, _)`。
            barra_r_squared, _ = style_data.joint_barra_r_squared_torch(
                raw_prediction, styles, industry_codes
            )
        # 当前置条件不成立时进入备用分支，保证其余合法情况仍被处理。
        else:
            # 调用 `torch.zeros` 按目标形状初始化数值容器 `barra_r_squared`。
            barra_r_squared = torch.zeros(
                (), dtype=torch.float32, device=raw_prediction.device
            )
        # 检查 `strict_barra_output_enabled()`；根据条件结果选择对应处理分支。
        if strict_barra_output_enabled():
            # 调用 `style_data.joint_barra_r_squared_torch` 按股票和交易日键对齐来自不同来源的数据，并将结果保存到 `(effective_barra_r_squared, _)`。
            effective_barra_r_squared, _ = (
                style_data.joint_barra_r_squared_torch(
                    prediction, styles, industry_codes
                )
            )
            # 检查 `float(effective_barra_r_squared) > 1e-06`；根据条件结果选择对应处理分支。
            if float(effective_barra_r_squared) > 1e-6:
                # 抛出 `FloatingPointError('严格 Barra 输出的联合 R² 未接近零')`，阻止不满足数据契约的流程继续执行。
                raise FloatingPointError(
                    "严格 Barra 输出的联合 R² 未接近零"
                )
        # 当前置条件不成立时进入备用分支，保证其余合法情况仍被处理。
        else:
            # 调用 `torch.zeros_like` 按目标形状初始化数值容器 `effective_barra_r_squared`。
            effective_barra_r_squared = torch.zeros_like(barra_r_squared)

        # 把本轮结果追加到 `effective_predictions_parts`，累积后续需要汇总的记录。
        effective_predictions_parts.append(
            prediction.cpu().numpy().astype(np.float64)
        )
        # 把本轮结果追加到 `raw_predictions_parts`，累积后续需要汇总的记录。
        raw_predictions_parts.append(
            raw_prediction.float().cpu().numpy().astype(np.float64)
        )
        # 把本轮结果追加到 `effective_targets`，累积后续需要汇总的记录。
        effective_targets.append(targets.cpu().numpy().astype(np.float64))
        # 把本轮结果追加到 `original_targets`，累积后续需要汇总的记录。
        original_targets.append(original.cpu().numpy().astype(np.float64))
        # 检查 `dual_tower_enabled()`；根据条件结果选择对应处理分支。
        if dual_tower_enabled():
            # 检查 `dual_style_score is None or total_return_score is None`；根据条件结果选择对应处理分支。
            if dual_style_score is None or total_return_score is None:
                # 抛出 `ValueError('No.6 验证缺少 Style 或 Total 分数')`，阻止不满足数据契约的流程继续执行。
                raise ValueError("No.6 验证缺少 Style 或 Total 分数")
            # 把本轮结果追加到 `dual_style_predictions`，累积后续需要汇总的记录。
            dual_style_predictions.append(
                dual_style_score.float().cpu().numpy().astype(np.float64)
            )
            # 把本轮结果追加到 `dual_total_predictions`，累积后续需要汇总的记录。
            dual_total_predictions.append(
                total_return_score.float().cpu().numpy().astype(np.float64)
            )
            # 把本轮结果追加到 `dual_fitted_targets`，累积后续需要汇总的记录。
            dual_fitted_targets.append(
                fitted_targets.cpu().numpy().astype(np.float64)
            )
        # 把本轮结果追加到 `date_ids`，累积后续需要汇总的记录。
        date_ids.append(global_date_ids.cpu().numpy().astype(np.int64, copy=False))
        # 把本轮结果追加到 `sample_indices`，累积后续需要汇总的记录。
        sample_indices.append(global_sample_indices.cpu().numpy().astype(np.int64, copy=False))
        # 检查 `adversarial_barra_enabled()`；根据条件结果选择对应处理分支。
        if adversarial_barra_enabled():
            # 检查 `style_prediction is None or industry_logits is None`；根据条件结果选择对应处理分支。
            if style_prediction is None or industry_logits is None:
                # 抛出 `ValueError('No.3 验证缺少对抗头输出')`，阻止不满足数据契约的流程继续执行。
                raise ValueError("No.3 验证缺少对抗头输出")
            # 把本轮结果追加到 `adversarial_style_predictions`，累积后续需要汇总的记录。
            adversarial_style_predictions.append(
                style_prediction.float().cpu().numpy()
            )
            # 把本轮结果追加到 `adversarial_style_targets`，累积后续需要汇总的记录。
            adversarial_style_targets.append(styles.float().cpu().numpy())
            # 把本轮结果追加到 `adversarial_style_valid`，累积后续需要汇总的记录。
            adversarial_style_valid.append(style_valid.cpu().numpy())
            # 把本轮结果追加到 `adversarial_industry_logits`，累积后续需要汇总的记录。
            adversarial_industry_logits.append(
                industry_logits.float().cpu().numpy()
            )
            # 把本轮结果追加到 `adversarial_industry_codes`，累积后续需要汇总的记录。
            adversarial_industry_codes.append(industry_codes.cpu().numpy())
        # 调用 `max` 汇总当前数据，并将统计量保存到 `raw_style_max`。
        raw_style_max = max(
            raw_style_max, float(raw_style["max_abs_style_pearson"])
        )
        # 用 `float(raw_style['mean_abs_style_pearson'])` 对 `raw_style_mean` 做 `Add` 累积更新。
        raw_style_mean += float(raw_style["mean_abs_style_pearson"])
        # 调用 `max` 汇总当前数据，并将统计量保存到 `effective_style_max`。
        effective_style_max = max(
            effective_style_max,
            float(effective_style["max_abs_style_pearson"]),
        )
        # 用 `float(effective_style['mean_abs_style_pearson'])` 对 `effective_style_mean` 做 `Add` 累积更新。
        effective_style_mean += float(
            effective_style["mean_abs_style_pearson"]
        )
        # 调用 `float` 统一设备或数据类型，并将结果保存到 `barra_value`。
        barra_value = float(barra_r_squared)
        # 用 `barra_value` 对 `barra_r_squared_sum` 做 `Add` 累积更新。
        barra_r_squared_sum += barra_value
        # 调用 `max` 汇总当前数据，并将统计量保存到 `barra_r_squared_max`。
        barra_r_squared_max = max(barra_r_squared_max, barra_value)
        # 调用 `float` 统一设备或数据类型，并将结果保存到 `effective_barra_value`。
        effective_barra_value = float(effective_barra_r_squared)
        # 用 `effective_barra_value` 对 `effective_barra_r_squared_sum` 做 `Add` 累积更新。
        effective_barra_r_squared_sum += effective_barra_value
        # 调用 `max` 汇总当前数据，并将统计量保存到 `effective_barra_r_squared_max`。
        effective_barra_r_squared_max = max(
            effective_barra_r_squared_max, effective_barra_value
        )
        # 检查 `gate is not None`；根据条件结果选择对应处理分支。
        if gate is not None:
            # 调用 `gate.float().abs` 逐元素取绝对值，并将结果保存到 `gate_abs`。
            gate_abs = gate.float().abs()
            # 用 `float(gate_abs.mean())` 对 `gate_mean_abs` 做 `Add` 累积更新。
            gate_mean_abs += float(gate_abs.mean())
            # 调用 `max` 汇总当前数据，并将统计量保存到 `gate_max_abs`。
            gate_max_abs = max(gate_max_abs, float(gate_abs.max()))
        # 用 `1` 对 `batches` 做 `Add` 累积更新。
        batches += 1
        # 检查 `max_batches is not None and batch_number >= max_batches`；根据条件结果选择对应处理分支。
        if max_batches is not None and batch_number >= max_batches:
            # 终止当前循环，避免继续处理已经不满足条件的元素。
            break

    # 检查 `not batches`；根据条件结果选择对应处理分支。
    if not batches:
        # 抛出 `ValueError('验证 loader 没有可用 batch')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("验证 loader 没有可用 batch")
    # 检查 `device.type == 'cuda'`；根据条件结果选择对应处理分支。
    if device.type == "cuda":
        # 调用 `torch.cuda.synchronize` 等待当前 GPU 队列执行完毕后再计时或验收。
        torch.cuda.synchronize(device)
    # 调用 `np.concatenate` 沿指定轴拼接分批产生的数组，并将结果保存到 `effective_x`。
    effective_x = np.concatenate(effective_predictions_parts)
    # 调用 `np.concatenate` 沿指定轴拼接分批产生的数组，并将结果保存到 `raw_x`。
    raw_x = np.concatenate(raw_predictions_parts)
    # 调用 `np.concatenate` 沿指定轴拼接分批产生的数组，并将结果保存到 `effective_y`。
    effective_y = np.concatenate(effective_targets)
    # 调用 `np.concatenate` 沿指定轴拼接分批产生的数组，并将结果保存到 `original_y`。
    original_y = np.concatenate(original_targets)
    # 调用 `np.concatenate` 沿指定轴拼接分批产生的数组，并将结果保存到 `dates`。
    dates = np.concatenate(date_ids)
    # 调用 `np.concatenate` 沿指定轴拼接分批产生的数组，并将结果保存到 `selected`。
    selected = np.concatenate(sample_indices)
    # 调用 `summarize_validation_arrays` 按约定截面或分组口径汇总统计结果，并将结果保存到 `effective_metrics`。
    effective_metrics = summarize_validation_arrays(
        effective_x, effective_y, dates, VALIDATION_METRIC,
        instrument_ids=instrument_ids()[selected],
    )
    # 构造键值映射，并保存到 `dual_metrics`，供后续按键查询。
    dual_metrics: dict[str, float | bool] = {}
    # 读取 `raw_x` 的当前值，并赋给 `original_prediction`。
    original_prediction = raw_x
    # 检查 `dual_tower_enabled()`；根据条件结果选择对应处理分支。
    if dual_tower_enabled():
        # 调用 `np.concatenate` 沿指定轴拼接分批产生的数组，并将结果保存到 `style_x`。
        style_x = np.concatenate(dual_style_predictions)
        # 调用 `np.concatenate` 沿指定轴拼接分批产生的数组，并将结果保存到 `total_x`。
        total_x = np.concatenate(dual_total_predictions)
        # 调用 `np.concatenate` 沿指定轴拼接分批产生的数组，并将结果保存到 `fitted_y`。
        fitted_y = np.concatenate(dual_fitted_targets)
        # 读取 `total_x` 的当前值，并赋给 `original_prediction`。
        original_prediction = total_x
        # 调用 `summarize_validation_arrays` 按约定截面或分组口径汇总统计结果，并将结果保存到 `total_metrics`。
        total_metrics = summarize_validation_arrays(
            total_x,
            original_y,
            dates,
            VALIDATION_METRIC,
            instrument_ids=instrument_ids()[selected],
        )
        # 调用统一 Pearson 判据识别近常数 Style 预测或目标。
        style_collapsed = has_degenerate_daily_pearson(
            style_x, fitted_y, dates
        )
        # 检查 `style_collapsed`；根据条件结果选择对应处理分支。
        if style_collapsed:
            # 调用 `float` 统一设备或数据类型，并将结果保存到 `style_rank_ic`。
            style_rank_ic = float("nan")
            # 调用 `float` 统一设备或数据类型，并将结果保存到 `style_pearson_ic`。
            style_pearson_ic = float("nan")
        # 当前置条件不成立时进入备用分支，保证其余合法情况仍被处理。
        else:
            # 调用 `summarize_validation_arrays` 按约定截面或分组口径汇总统计结果，并将结果保存到 `style_metrics`。
            style_metrics = summarize_validation_arrays(
                style_x,
                fitted_y,
                dates,
                VALIDATION_METRIC,
                instrument_ids=instrument_ids()[selected],
            )
            # 调用 `float` 统一设备或数据类型，并将结果保存到 `style_rank_ic`。
            style_rank_ic = float(
                style_metrics["validation_daily_rank_ic"]
            )
            # 调用 `float` 统一设备或数据类型，并将结果保存到 `style_pearson_ic`。
            style_pearson_ic = float(
                style_metrics["validation_daily_pearson_ic"]
            )
        # 构造键值映射，并保存到 `dual_metrics`，供后续按键查询。
        dual_metrics = {
            "validation_total_original_daily_rank_ic": float(
                total_metrics["validation_daily_rank_ic"]
            ),
            "validation_total_original_daily_pearson_ic": float(
                total_metrics["validation_daily_pearson_ic"]
            ),
            "validation_style_fitted_daily_rank_ic": style_rank_ic,
            "validation_style_fitted_daily_pearson_ic": style_pearson_ic,
            "validation_style_score_collapsed": style_collapsed,
            "validation_dual_additivity_max_abs_error": float(
                np.max(np.abs(total_x - raw_x - style_x))
            ),
        }
    # 调用 `summarize_validation_arrays` 按约定截面或分组口径汇总统计结果，并将结果保存到 `original_metrics`。
    original_metrics = summarize_validation_arrays(
        original_prediction, original_y, dates, VALIDATION_METRIC,
        instrument_ids=instrument_ids()[selected],
    )
    # 构造键值映射，并保存到 `adversarial_metrics`，供后续按键查询。
    adversarial_metrics: dict[str, float] = {}
    # 检查 `adversarial_barra_enabled()`；根据条件结果选择对应处理分支。
    if adversarial_barra_enabled():
        # 构造键值映射，并保存到 `adversarial_metrics`，供后续按键查询。
        adversarial_metrics = {
            f"validation_adversarial_{key}": value
            for key, value in summarize_adversarial_predictions(
                np.concatenate(adversarial_style_predictions),
                np.concatenate(adversarial_style_targets),
                np.concatenate(adversarial_style_valid),
                np.concatenate(adversarial_industry_logits),
                np.concatenate(adversarial_industry_codes),
                industry_class_weights.detach().cpu().numpy(),
            ).items()
        }
    # 将 `{'validation_ic': float(effective_metrics['validation_ic']), 'validation_daily_rank_ic': float(…` 作为当前函数结果返回给调用方。
    return {
        "validation_ic": float(effective_metrics["validation_ic"]),
        "validation_daily_rank_ic": float(
            effective_metrics["validation_daily_rank_ic"]
        ),
        "validation_daily_pearson_ic": float(
            effective_metrics["validation_daily_pearson_ic"]
        ),
        "validation_pooled_ic": float(
            effective_metrics["validation_pooled_ic"]
        ),
        "validation_original_daily_rank_ic": float(
            original_metrics["validation_daily_rank_ic"]
        ),
        "validation_original_daily_pearson_ic": float(
            original_metrics["validation_daily_pearson_ic"]
        ),
        "validation_original_pooled_ic": float(
            original_metrics["validation_pooled_ic"]
        ),
        "validation_raw_max_abs_style_pearson": raw_style_max,
        "validation_raw_mean_abs_style_pearson": raw_style_mean / batches,
        "validation_barra_r_squared_mean": barra_r_squared_sum / batches,
        "validation_barra_r_squared_max": barra_r_squared_max,
        "validation_effective_barra_r_squared_mean": (
            effective_barra_r_squared_sum / batches
        ),
        "validation_effective_barra_r_squared_max": effective_barra_r_squared_max,
        "validation_effective_max_abs_industry_mean": (
            effective_industry_mean_max
        ),
        "validation_gate_mean_abs": gate_mean_abs / batches,
        "validation_gate_max_abs": gate_max_abs,
        "validation_effective_max_abs_style_pearson": effective_style_max,
        "validation_effective_mean_abs_style_pearson": (
            effective_style_mean / batches
        ),
        "validation_dates": int(effective_metrics["validation_dates"]),
        "validation_samples": int(effective_metrics["validation_samples"]),
        "validation_seconds": time.perf_counter() - started,
        "validation_score_collapsed": False,
        "validation_collapsed_date_count": 0,
        "validation_min_daily_raw_score_std": float(
            minimum_daily_raw_score_std
        ),
        **adversarial_metrics,
        **dual_metrics,
    }


# 使用 `torch.no_grad()` 包装 `predict_model`，并定义其可复用的处理逻辑。
@torch.no_grad()
def predict_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    distributed_context: distributed.DistributedContext | None = None,
) -> pd.DataFrame:
    """在预测集上生成与样本行号严格对齐的模型分数。
    
    输入：
        model: torch.nn.Module。
        loader: DataLoader。
        device: torch.device。
    输出：
        pd.DataFrame。
        """
    # 检查 `not barra_conditioning_enabled() and (not dual_tower_enabled()) and (raw_barra_r2_enabled() or …`；根据条件结果选择对应处理分支。
    if (
        not barra_conditioning_enabled()
        and not dual_tower_enabled()
        and (raw_barra_r2_enabled() or adversarial_barra_enabled())
    ):
        # 调用 `base.predict_model` 执行模型推理并生成样本预测分数，并将结果保存到 `raw`。
        raw = base.predict_model(
            model, loader, device, bool(USE_CUDA_PREFETCH),
            distributed_context=distributed_context,
        )
        # 调用 `raw['prediction'].to_numpy` 执行模型推理并生成样本预测分数，并将结果保存到 `values`。
        values = raw["prediction"].to_numpy(np.float32)
        # 将 `pd.DataFrame({'sample_index': raw['sample_index'].to_numpy(np.int64), 'prediction_model_raw': v…` 作为当前函数结果返回给调用方。
        return pd.DataFrame(
            {
                "sample_index": raw["sample_index"].to_numpy(np.int64),
                "prediction_model_raw": values,
                "prediction": values,
            }
        )
    # 调用 `model.eval` 将模型切换到评估模式，关闭训练期随机行为。
    local_model = base.inference_model(model)
    # 执行当前分布式训练步骤并保持既定数据语义。
    local_model.eval()
    # 构造当前步骤所需的列表，并保存到 `sample_parts`。
    sample_parts: list[np.ndarray] = []
    # 构造当前步骤所需的列表，并保存到 `raw_parts`。
    raw_parts: list[np.ndarray] = []
    # 构造当前步骤所需的列表，并保存到 `effective_parts`。
    effective_parts: list[np.ndarray] = []
    # 构造当前步骤所需的列表，并保存到 `style_parts`。
    style_parts: list[np.ndarray] = []
    # 构造当前步骤所需的列表，并保存到 `total_parts`。
    total_parts: list[np.ndarray] = []
    # 遍历 `StyleCudaPrefetcher(loader, device, bool(USE_CUDA_PREFETCH))`，并把当前元素绑定到 `item` 逐项处理。
    for item in StyleCudaPrefetcher(
        loader, device, bool(USE_CUDA_PREFETCH)
    ):
        # 读取 `item` 的当前值，并赋给 `(batch, videos, frame_mask, _targets, _original, _residual, _fitted, st…`。
        (
            batch, videos, frame_mask, _targets, _original, _residual,
            _fitted, styles, style_valid, industry_codes,
        ) = item
        # 检查 `int(batch['date_ids'].min()) != int(batch['date_ids'].max())`；根据条件结果选择对应处理分支。
        if int(batch["date_ids"].min()) != int(batch["date_ids"].max()):
            # 抛出 `AssertionError('推理 batch 混入多个交易日')`，阻止不满足数据契约的流程继续执行。
            raise AssertionError("推理 batch 混入多个交易日")
        # 执行当前分布式训练步骤并保持既定数据语义。
        (
            conditioning_styles,
            conditioning_valid,
            conditioning_industry,
            conditioning_prepared,
        ) = prepare_distributed_conditioning(
            styles, style_valid, industry_codes, batch["sample_indices"], distributed_context
        )
        # 在 `base.autocast_context(device)` 管理的上下文中执行本段逻辑，并保证资源按约定释放。
        with base.autocast_context(device):
            # 调用 `forward_training_model` 对当前 batch 执行模型前向传播，并将结果保存到 `(raw_prediction, _, _, _, _, dual_style_score, total_return_score)`。
            (
                raw_prediction, _, _, _, _, dual_style_score,
                total_return_score,
            ) = forward_training_model(
                local_model,
                videos,
                frame_mask,
                gamma=0.0,
                styles=conditioning_styles,
                style_valid=conditioning_valid,
                industry_codes=conditioning_industry,
                sample_indices=batch["sample_indices"],
                conditioning_prepared=conditioning_prepared,
            )
        # 执行当前分布式训练步骤并保持既定数据语义。
        (
            raw_prediction,
            _style_prediction,
            _industry_logits,
            _gate,
            dual_style_score,
            total_return_score,
            _targets,
            _original,
            _residual,
            _fitted,
            styles,
            style_valid,
            industry_codes,
            global_sample_indices,
            _global_date_ids,
        ) = gather_style_cross_section(
            raw_prediction,
            None,
            None,
            None,
            dual_style_score,
            total_return_score,
            _targets,
            _original,
            _residual,
            _fitted,
            styles,
            style_valid,
            industry_codes,
            batch,
            distributed_context,
            differentiable_outputs=False,
        )
        # 调用 `effective_predictions` 执行模型推理并生成样本预测分数，并将结果保存到 `(prediction, _)`。
        prediction, _ = effective_predictions(
            raw_prediction, styles, industry_codes
        )
        # 把本轮结果追加到 `sample_parts`，累积后续需要汇总的记录。
        sample_parts.append(global_sample_indices.cpu().numpy())
        # 把本轮结果追加到 `raw_parts`，累积后续需要汇总的记录。
        raw_parts.append(raw_prediction.float().cpu().numpy())
        # 把本轮结果追加到 `effective_parts`，累积后续需要汇总的记录。
        effective_parts.append(prediction.cpu().numpy())
        # 检查 `dual_tower_enabled()`；根据条件结果选择对应处理分支。
        if dual_tower_enabled():
            # 检查 `dual_style_score is None or total_return_score is None`；根据条件结果选择对应处理分支。
            if dual_style_score is None or total_return_score is None:
                # 抛出 `ValueError('No.6 推理缺少 Style 或 Total 分数')`，阻止不满足数据契约的流程继续执行。
                raise ValueError("No.6 推理缺少 Style 或 Total 分数")
            # 把本轮结果追加到 `style_parts`，累积后续需要汇总的记录。
            style_parts.append(dual_style_score.float().cpu().numpy())
            # 把本轮结果追加到 `total_parts`，累积后续需要汇总的记录。
            total_parts.append(total_return_score.float().cpu().numpy())
    # 检查 `not sample_parts`；根据条件结果选择对应处理分支。
    if not sample_parts:
        # 将 `pd.DataFrame({'sample_index': pd.Series(dtype=np.int64), 'prediction_model_raw': pd.Series(dtyp…` 作为当前函数结果返回给调用方。
        return pd.DataFrame(
            {
                "sample_index": pd.Series(dtype=np.int64),
                "prediction_model_raw": pd.Series(dtype=np.float32),
                "prediction": pd.Series(dtype=np.float32),
                "prediction_style_component": pd.Series(dtype=np.float32),
                "prediction_total_return": pd.Series(dtype=np.float32),
            }
        )
    # 调用 `pd.DataFrame` 转换数据表示，并将标准化对象保存到 `result`。
    result = pd.DataFrame(
        {
            "sample_index": np.concatenate(sample_parts).astype(np.int64),
            "prediction_model_raw": np.concatenate(raw_parts).astype(np.float32),
            "prediction": np.concatenate(effective_parts).astype(np.float32),
        }
    )
    # 检查 `dual_tower_enabled()`；根据条件结果选择对应处理分支。
    if dual_tower_enabled():
        # 调用 `np.concatenate(style_parts).astype` 统一设备或数据类型，并将结果保存到 `result['prediction_style_component']`。
        result["prediction_style_component"] = np.concatenate(
            style_parts
        ).astype(np.float32)
        # 调用 `np.concatenate(total_parts).astype` 统一设备或数据类型，并将结果保存到 `result['prediction_total_return']`。
        result["prediction_total_return"] = np.concatenate(
            total_parts
        ).astype(np.float32)
    # 将 `result` 作为当前函数结果返回给调用方。
    return result.sort_values("sample_index", kind="stable").reset_index(
        drop=True
    )


# 设置 `PROBE_MAX_SAMPLES_PER_DATE` 的固定配置值，供后续流程统一使用。
PROBE_MAX_SAMPLES_PER_DATE = 512
# 设置 `PROBE_BATCH_SIZE` 的固定配置值，供后续流程统一使用。
PROBE_BATCH_SIZE = 4096
# 设置 `PROBE_LEARNING_RATE` 的固定配置值，供后续流程统一使用。
PROBE_LEARNING_RATE = 1e-3
# 设置 `PROBE_WEIGHT_DECAY` 的固定配置值，供后续流程统一使用。
PROBE_WEIGHT_DECAY = 1e-4
# 设置 `PROBE_MAX_EPOCHS` 的固定配置值，供后续流程统一使用。
PROBE_MAX_EPOCHS = 50
# 设置 `PROBE_PATIENCE` 的固定配置值，供后续流程统一使用。
PROBE_PATIENCE = 5
# 设置 `PROBE_RIDGE_ALPHA` 的固定配置值，供后续流程统一使用。
PROBE_RIDGE_ALPHA = 1e-3




















# %% [markdown]
# ## 年度 × seed 训练和断点续训


# %%
# 定义 `run_payload` 函数，封装这一阶段可独立调用的处理步骤。
def run_payload(
    year: int, seed: int, batch_summary: dict[str, Any]
) -> dict[str, Any]:
    """整理影响整个 run 身份和可复用性的签名载荷。
    
    输入：
        year: int。
        seed: int。
        batch_summary: dict[str, Any]。
    输出：
        dict[str, Any]。
        """
    # 调用 `sequence_data.require_aligned_sequence_store` 校验输入、配置或产物是否满足当前阶段的数据契约，并将结果保存到 `sequence`。
    sequence = sequence_data.require_aligned_sequence_store()
    # 调用 `style_data.require_style_neutral_store` 校验输入、配置或产物是否满足当前阶段的数据契约，并将结果保存到 `style_manifest`。
    style_manifest = style_data.require_style_neutral_store()
    # 按 `industry_sidecar_required()` 在两个候选值间选择，并保存到 `industry_manifest`。
    industry_manifest = (
        style_data.require_industry_sidecar()
        if industry_sidecar_required() else None
    )
    # 按 `barra_conditioning_enabled() or dual_tower_enabled()` 在两个候选值间选择，并保存到 `full_target_manifest`。
    full_target_manifest = (
        style_data.require_full_barra_target_sidecar()
        if barra_conditioning_enabled() or dual_tower_enabled() else None
    )
    # 按 `dual_tower_enabled()` 在两个候选值间选择，并保存到 `dual_target_manifest`。
    dual_target_manifest = (
        style_data.require_dual_tower_target_sidecar()
        if dual_tower_enabled() else None
    )
    # 调用 `training_config` 整理并返回当前实验的配置对象，并将结果保存到 `config`。
    config = training_config()
    # 构造键值映射，并保存到 `payload`，供后续按键查询。
    payload = {
        "schema_version": (
            7 if dual_tower_enabled()
            else 6 if strict_barra_output_enabled()
            else 5 if barra_conditioning_enabled() else TRAINING_SCHEMA_VERSION
        ),
        "training_code_sha256": hashlib.sha256(
            SCRIPT_PATH.read_bytes()
        ).hexdigest(),
        "run_name": RUN_NAME,
        "forecast_year": year,
        "seed": seed,
        "sequence_signature": sequence["signature"],
        "style_store_signature": style_manifest["signature"],
        "industry_sidecar_signature": (
            industry_manifest["signature"]
            if industry_manifest is not None else None
        ),
        "source_dataset_signature": sequence["payload"][
            "source_dataset_signature"
        ],
        "source_rolling_signature": sequence["payload"][
            "source_rolling_signature"
        ],
        "split_policy": "exact_vit_train_validation_prediction_indices",
        "model_config": asdict(base.MODEL_CONFIG),
        "training_config": asdict(config),
        "training_batch_summary": batch_summary,
        "residualize_target": RESIDUALIZE_TARGET,
        "project_model_output": PROJECT_MODEL_OUTPUT,
        "target_policy": target_policy(),
        "output_policy": output_policy(),
        "projection": {
            "basis": "intercept_plus_9_cne5s_styles_plus_reversal20",
            "solver": "torch_fp32_reduced_qr",
            "standardization": "daily_zscore_ddof0",
            "industry_used": False,
        },
        "style_columns": list(style_data.STYLE_COLUMNS),
        "industry_used_by_regularizer": raw_barra_r2_enabled(),
        "industry_used_by_adversary": adversarial_barra_enabled(),
        "inference_requires_barra": (
            dual_tower_enabled()
            or strict_barra_output_enabled()
            or BARRA_CONDITIONING_MODE in {"real", "shuffled"}
        ),
        "early_stop_selection": "validation_ic_only",
        "style_correlation_penalty": raw_barra_r2_enabled(),
        "raw_barra_r2_lambda": RAW_BARRA_R2_LAMBDA,
        "adversarial_barra": adversarial_barra_enabled(),
        "adversarial_config": (
            {
                "representation": "final_norm_cls_128",
                "style_targets": list(style_data.STYLE_COLUMNS),
                "industry_classes": INDUSTRY_COUNT,
                "unknown_industry_policy": "code_zero_masked_from_ce",
                "head": {
                    "hidden_dim": ADVERSARIAL_HEAD_HIDDEN_DIM,
                    "activation": "gelu",
                    "dropout": ADVERSARIAL_HEAD_DROPOUT,
                },
                "gamma_max": ADVERSARIAL_GAMMA_MAX,
                "warmup_epochs": ADVERSARIAL_WARMUP_EPOCHS,
                "cosine_ramp_epochs": ADVERSARIAL_RAMP_EPOCHS,
                "style_loss": "masked_mse",
                "style_loss_weight": ADVERSARIAL_STYLE_LOSS_WEIGHT,
                "industry_loss": "inverse_sqrt_weighted_ce_div_log32",
                "industry_loss_weight": ADVERSARIAL_INDUSTRY_LOSS_WEIGHT,
                "probe": {
                    "sample_cap_per_date": PROBE_MAX_SAMPLES_PER_DATE,
                    "date_split": [0.4, 0.1, 0.5],
                    "batch_size": PROBE_BATCH_SIZE,
                    "learning_rate": PROBE_LEARNING_RATE,
                    "weight_decay": PROBE_WEIGHT_DECAY,
                    "max_epochs": PROBE_MAX_EPOCHS,
                    "patience": PROBE_PATIENCE,
                    "ridge_alpha": PROBE_RIDGE_ALPHA,
                    "mlp_hidden_dim": 128,
                    "mlp_dropout": 0.1,
                },
            }
            if adversarial_barra_enabled() else None
        ),
        "raw_barra_r2_design": {
            "dependent_variable": "raw_model_score",
            "continuous": list(style_data.STYLE_COLUMNS),
            "industry_fixed_effects": True,
            "solver": "industry_within_then_torch_fp32_reduced_qr",
            "missing": "style_zero_industry_unknown_no_forward_fill",
        },
        "loss_formula": (
            f"{LOSS_NAME}(alpha+style,source_target) + "
            f"{DUAL_TOWER_ALPHA_LOSS_WEIGHT}*{LOSS_NAME}(alpha,full_barra_residual) + "
            f"{DUAL_TOWER_STYLE_LOSS_WEIGHT}*{LOSS_NAME}(style,full_barra_fitted) + "
            f"{RAW_BARRA_R2_LAMBDA}*R2(alpha|ten_styles+industry)"
            if dual_tower_enabled()
            else
            f"{LOSS_NAME}(strict_full_barra_projection,full_barra_residual_target)"
            if strict_barra_output_enabled()
            else
            f"{LOSS_NAME}(raw_score,full_barra_residual_target) + "
            f"{RAW_BARRA_R2_LAMBDA}*R2(raw_score|ten_styles+industry)"
            if barra_conditioning_enabled()
            else
            f"{LOSS_NAME}(raw_score,source_target) + masked_style_mse + "
            "weighted_industry_ce/log(32) through GRL"
            if adversarial_barra_enabled()
            else
            f"{LOSS_NAME}(raw_score,source_target) + "
            f"{RAW_BARRA_R2_LAMBDA}*R2(raw_score|ten_styles+industry)"
            if raw_barra_r2_enabled()
            else f"{LOSS_NAME}(effective_prediction,effective_target)"
        ),
        "early_stop_target": (
            "alpha_score_vs_full_barra_residual_target"
            if dual_tower_enabled()
            else "full_barra_residual_target"
            if barra_conditioning_enabled()
            else "style_residual_target"
            if RESIDUALIZE_TARGET
            else "source_target"
        ),
        "precision": "bf16_model_forward_fp32_projection_barra_r2_and_loss",
    }
    # 执行当前分布式训练步骤并保持既定数据语义。
    if EXECUTION_ROUTE == "ddp":
        # 执行当前分布式训练步骤并保持既定数据语义。
        payload.update({
            "execution_route": "ddp",
            "world_size": DDP_WORLD_SIZE,
        })
    # 检查 `dual_tower_enabled()`；根据条件结果选择对应处理分支。
    if dual_tower_enabled():
        # 读取 `full_target_manifest['signature']` 的当前值，并赋给 `payload['full_barra_target_sidecar_signature']`。
        payload["full_barra_target_sidecar_signature"] = (
            full_target_manifest["signature"]
        )
        # 读取 `dual_target_manifest['signature']` 的当前值，并赋给 `payload['dual_tower_target_sidecar_signature']`。
        payload["dual_tower_target_sidecar_signature"] = (
            dual_target_manifest["signature"]
        )
        # 构造键值映射，并保存到 `payload['barra_dual_tower']`，供后续按键查询。
        payload["barra_dual_tower"] = {
            "mode": BARRA_DUAL_TOWER_MODE,
            "alpha_architecture": "vivit_cls_times_bounded_barra_gate",
            "style_architecture": "independent_barra_mlp_scalar_head",
            "style_hidden_dim": DUAL_TOWER_STYLE_HIDDEN_DIM,
            "style_dropout": DUAL_TOWER_STYLE_DROPOUT,
            "industry_embedding_dim": BARRA_INDUSTRY_EMBEDDING_DIM,
            "style_output_initialization": "zeros",
            "return_loss_weight": 1.0,
            "alpha_loss_weight": DUAL_TOWER_ALPHA_LOSS_WEIGHT,
            "style_loss_weight": DUAL_TOWER_STYLE_LOSS_WEIGHT,
            "alpha_r2_lambda": RAW_BARRA_R2_LAMBDA,
            "r2_uses_true_unmodified_barra": True,
            "early_stop": "alpha_vs_full_barra_residual_daily_spearman",
            "published_factor": "unprojected_alpha_score",
            "ensemble_scaling": "center_components_divide_by_alpha_std",
            "shuffle_policy": (
                "within_date_joint_permutation_by_stable_sample_key_v1"
                if BARRA_DUAL_TOWER_MODE == "shuffled" else None
            ),
        }
    # 检查 `barra_conditioning_enabled()`；根据条件结果选择对应处理分支。
    if barra_conditioning_enabled():
        # 读取 `full_target_manifest['signature']` 的当前值，并赋给 `payload['full_barra_target_sidecar_signature']`。
        payload["full_barra_target_sidecar_signature"] = (
            full_target_manifest["signature"]
        )
        # 构造键值映射，并保存到 `payload['barra_conditioning']`，供后续按键查询。
        payload["barra_conditioning"] = {
            "mode": BARRA_CONDITIONING_MODE,
            "continuous_inputs": list(style_data.STYLE_COLUMNS),
            "valid_mask_inputs": list(style_data.STYLE_COLUMNS),
            "industry_classes": INDUSTRY_COUNT,
            "unknown_industry_embedding": "code_zero_fixed_zero",
            "industry_embedding_dim": BARRA_INDUSTRY_EMBEDDING_DIM,
            "hidden_dim": BARRA_GATE_HIDDEN_DIM,
            "activation": "gelu",
            "dropout": BARRA_GATE_DROPOUT,
            "gate": "scale_times_tanh_linear",
            "gate_scale": BARRA_GATE_SCALE,
            "gate_projection_initialization": "zeros",
            "conditioned_representation": "h_times_one_plus_gate",
            "additive_barra_alpha": False,
            "shuffle_policy": (
                "within_date_joint_permutation_by_stable_sample_key_v1"
                if BARRA_CONDITIONING_MODE == "shuffled" else None
            ),
            "r2_uses_true_unmodified_barra": True,
        }
    # 检查 `strict_barra_output_enabled()`；根据条件结果选择对应处理分支。
    if strict_barra_output_enabled():
        # 构造键值映射，并保存到 `payload['strict_barra_output']`，供后续按键查询。
        payload["strict_barra_output"] = {
            "weighting": "equal_weight_W_identity",
            "design": [
                "intercept",
                *style_data.STYLE_COLUMNS,
                "industry_fixed_effects",
            ],
            "solver": "industry_within_then_torch_fp32_reduced_qr",
            "post_projection": "zscore_ddof0",
            "applied_during": ["training_loss", "validation", "inference"],
            "published_factor": "strict_projected_score",
            "pre_neutral_score_is_audit_only": True,
        }
    # 将 `payload` 作为当前函数结果返回给调用方。
    return payload



# 定义 `load_best_model` 函数，封装这一阶段可独立调用的处理步骤。
def load_best_model(
    path: Path, device: torch.device
) -> torch.nn.Module:
    """加载最佳验证 checkpoint，并恢复可用于预测的模型。
    
    输入：
        path: Path。
        device: torch.device。
    输出：
        torch.nn.Module。
        """
    # 调用 `torch.load` 读取源数据，并将结果保存到 `checkpoint`。
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    # 读取 `checkpoint['model_state']` 的当前值，并赋给 `state`。
    state = checkpoint["model_state"]
    # 检查 `dual_tower_enabled() or barra_conditioning_enabled()`；根据条件结果选择对应处理分支。
    if dual_tower_enabled() or barra_conditioning_enabled():
        # 调用 `build_training_model` 根据当前配置和输入构造后续流程所需对象，并将结果保存到 `model`。
        model = build_training_model(device, use_compile=False)
        # 调用 `base.load_canonical_state` 读取并解析后续计算所需的数据或运行状态。
        base.load_canonical_state(model, state)
        # 调用 `model.eval` 将模型切换到评估模式，关闭训练期随机行为。
        model.eval()
        # 将 `model` 作为当前函数结果返回给调用方。
        return model
    # 检查 `adversarial_barra_enabled()`；根据条件结果选择对应处理分支。
    if adversarial_barra_enabled():
        # 设置 `prefix` 的固定配置值，供后续流程统一使用。
        prefix = "backbone."
        # 构造键值映射，并保存到 `state`，供后续按键查询。
        state = {
            key.removeprefix(prefix): value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        # 检查 `not state`；根据条件结果选择对应处理分支。
        if not state:
            # 抛出 `ValueError('No.3 checkpoint 缺少 backbone 参数')`，阻止不满足数据契约的流程继续执行。
            raise ValueError("No.3 checkpoint 缺少 backbone 参数")
    # 调用 `base.build_model` 根据模型配置创建网络并部署到目标设备，并将结果保存到 `model`。
    model = base.build_model(
        device, bool(USE_TORCH_COMPILE), COMPILE_MODE, TRAIN_BATCH_MODE
    )
    # 调用 `base.load_canonical_state` 读取并解析后续计算所需的数据或运行状态。
    base.load_canonical_state(model, state)
    # 调用 `model.eval` 将模型切换到评估模式，关闭训练期随机行为。
    model.eval()
    # 将 `model` 作为当前函数结果返回给调用方。
    return model


# 定义 `load_best_training_model` 函数，封装这一阶段可独立调用的处理步骤。
def load_best_training_model(
    path: Path, device: torch.device
) -> torch.nn.Module:
    """加载包含附加实验分支的最佳训练模型。
    
    输入：
        path: Path。
        device: torch.device。
    输出：
        torch.nn.Module。
        """
    # 调用 `torch.load` 读取源数据，并将结果保存到 `checkpoint`。
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    # 调用 `build_training_model` 根据当前配置和输入构造后续流程所需对象，并将结果保存到 `model`。
    model = build_training_model(device, use_compile=False)
    # 调用 `base.load_canonical_state` 读取并解析后续计算所需的数据或运行状态。
    base.load_canonical_state(model, checkpoint["model_state"])
    # 调用 `model.eval` 将模型切换到评估模式，关闭训练期随机行为。
    model.eval()
    # 将 `model` 作为当前函数结果返回给调用方。
    return model


# 定义 `train_task` 函数，封装这一阶段可独立调用的处理步骤。
def train_task(
    year: int,
    seed: int,
    physical_gpu: int,
    *,
    max_epochs: int | None = None,
    max_batches: int | None = None,
    persist: bool = True,
    distributed_context: distributed.DistributedContext | None = None,
) -> dict[str, Any]:
    """执行一个年度×seed 任务，保存 checkpoint、预测和训练诊断。
    
    输入：
        year: int。
        seed: int。
        physical_gpu: int。
        max_epochs: int | None。
        max_batches: int | None。
        persist: bool。
    输出：
        dict[str, Any]。
        """
    # 调用 `style_data.require_style_neutral_store` 校验输入、配置或产物是否满足当前阶段的数据契约。
    style_data.require_style_neutral_store()
    # 检查 `industry_sidecar_required()`；根据条件结果选择对应处理分支。
    if industry_sidecar_required():
        # 调用 `style_data.require_industry_sidecar` 校验输入、配置或产物是否满足当前阶段的数据契约。
        style_data.require_industry_sidecar()
    # 检查 `barra_conditioning_enabled() or dual_tower_enabled()`；根据条件结果选择对应处理分支。
    if barra_conditioning_enabled() or dual_tower_enabled():
        # 调用 `style_data.require_full_barra_target_sidecar` 校验输入、配置或产物是否满足当前阶段的数据契约。
        style_data.require_full_barra_target_sidecar()
    # 检查 `dual_tower_enabled()`；根据条件结果选择对应处理分支。
    if dual_tower_enabled():
        # 调用 `style_data.require_dual_tower_target_sidecar` 校验输入、配置或产物是否满足当前阶段的数据契约。
        style_data.require_dual_tower_target_sidecar()
    # 执行当前分布式训练步骤并保持既定数据语义。
    primary = distributed_context is None or distributed_context.is_primary
    # 调用 `torch.device` 解析训练或推理所使用的计算设备，并将结果保存到 `device`。
    device = (
        distributed_context.device if distributed_context is not None
        else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    # 检查 `device.type != 'cuda'`；根据条件结果选择对应处理分支。
    if device.type != "cuda":
        # 抛出 `RuntimeError('完整风格中性训练要求 CUDA')`，阻止不满足数据契约的流程继续执行。
        raise RuntimeError("完整风格中性训练要求 CUDA")
    # 调用 `base.set_random_seed` 同时固定 Python、NumPy 和 PyTorch 随机状态。
    base.set_random_seed(
        seed + (distributed_context.rank if distributed_context else 0)
    )
    # 调用 `make_style_loaders` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `(train_loader, validation_loader, prediction_loader)`。
    train_loader, validation_loader, prediction_loader = make_style_loaders(
        year, seed, WORKERS_PER_TASK if persist else 0, pin_memory=True,
        distributed_context=distributed_context,
    )
    # 调用 `sequence_data.load_sequence_split` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `split`。
    split = sequence_data.load_sequence_split(year)
    # 按 `adversarial_barra_enabled()` 在两个候选值间选择，并保存到 `industry_class_weights`。
    industry_class_weights = (
        training_industry_class_weights(split["train_indices"]).to(device)
        if adversarial_barra_enabled()
        else torch.ones(INDUSTRY_COUNT, dtype=torch.float32, device=device)
    )
    # 调用 `sequence_data.loader_batch_summary` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `batch_summary`。
    batch_summary = sequence_data.loader_batch_summary(train_loader)
    # 调用 `run_payload` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `payload`。
    payload = run_payload(year, seed, batch_summary)
    # 调用 `sequence_data.stable_hash` 对关键配置生成稳定签名以支持复现和一致性检查，并将结果保存到 `signature`。
    signature = sequence_data.stable_hash(payload)
    # 调用 `training_config` 整理并返回当前实验的配置对象，并将结果保存到 `config`。
    config = training_config()
    # 调用 `int` 统一设备或数据类型，并将结果保存到 `epoch_limit`。
    epoch_limit = int(max_epochs or MAX_EPOCHS)

    # 调用 `base.run_paths` 解析当前 run 对应的数据与输出路径，并将结果保存到 `(model_root, _, _)`。
    model_root, _, _ = base.run_paths(RUN_NAME)
    # 计算 `model_root / f'forecast_{year}' / f'seed_{seed}'`，并将数值结果保存到 `run_directory`。
    run_directory = model_root / f"forecast_{year}" / f"seed_{seed}"
    # 计算 `run_directory / 'best.pt'`，并将数值结果保存到 `best_path`。
    best_path = run_directory / "best.pt"
    # 计算 `run_directory / 'last.pt'`，并将数值结果保存到 `last_path`。
    last_path = run_directory / "last.pt"
    # 计算 `run_directory / 'history.parquet'`，并将数值结果保存到 `history_path`。
    history_path = run_directory / "history.parquet"
    # 计算 `run_directory / 'prediction.parquet'`，并将数值结果保存到 `prediction_path`。
    prediction_path = run_directory / "prediction.parquet"
    # 计算 `run_directory / 'adversarial_probe_diagnostics.parquet'`，并将数值结果保存到 `probe_path`。
    probe_path = run_directory / "adversarial_probe_diagnostics.parquet"
    # 计算 `run_directory / 'completed.json'`，并将数值结果保存到 `completed_path`。
    completed_path = run_directory / "completed.json"
    # 设置 `resume_checkpoint` 的固定配置值，供后续流程统一使用。
    resume_checkpoint: dict[str, Any] | None = None
    # 检查 `persist`；根据条件结果选择对应处理分支。
    if persist:
        # 调用 `run_directory.mkdir` 创建输出目录，确保后续文件可以安全写入。
        if primary:
            # 执行当前分布式训练步骤并保持既定数据语义。
            run_directory.mkdir(parents=True, exist_ok=True)
        # 执行当前分布式训练步骤并保持既定数据语义。
        if distributed_context is not None:
            # 执行当前分布式训练步骤并保持既定数据语义。
            dist.barrier()
        # 检查 `completed_path.exists() and best_path.exists() and (not OVERWRITE)`；根据条件结果选择对应处理分支。
        if completed_path.exists() and best_path.exists() and not OVERWRITE:
            # 调用 `json.loads` 读取源数据，并将结果保存到 `completed`。
            completed = json.loads(completed_path.read_text(encoding="utf-8"))
            # 检查 `completed.get('run_signature') == signature`；根据条件结果选择对应处理分支。
            if completed.get("run_signature") == signature:
                # 调用 `print` 记录当前流程状态或诊断信息。
                print(f"[{year}/seed={seed}] 已完成，直接复用。", flush=True)
                # 将 `completed['summary']` 作为当前函数结果返回给调用方。
                return completed["summary"]
            # 按 `last_path.exists()` 在两个候选值间选择，并保存到 `compatibility_path`。
            compatibility_path = last_path if last_path.exists() else best_path
            # 调用 `torch.load` 读取源数据，并将结果保存到 `previous`。
            previous = torch.load(
                compatibility_path, map_location="cpu", weights_only=False
            )
            # 检查 `is_resume_compatible_recovery_patch(previous.get('run_payload', {}), payload)`；根据条件结果选择对应处理分支。
            if is_resume_compatible_recovery_patch(
                previous.get("run_payload", {}), payload
            ):
                # 调用 `print` 记录当前流程状态或诊断信息。
                print(
                    f"[{year}/seed={seed}] 已完成产物仅代码哈希不同，"
                    "按白名单直接复用且不改写旧产物。",
                    flush=True,
                )
                # 将 `completed['summary']` 作为当前函数结果返回给调用方。
                return completed["summary"]
            # 抛出 `ValueError(f'{completed_path} 与当前配置签名不一致；请使用新 run name，或显式设置 --overwrite 1')`，阻止不满足数据契约的流程继续执行。
            raise ValueError(
                f"{completed_path} 与当前配置签名不一致；"
                "请使用新 run name，或显式设置 --overwrite 1"
            )
        # 检查 `last_path.exists() and (not OVERWRITE)`；根据条件结果选择对应处理分支。
        if last_path.exists() and not OVERWRITE:
            # 调用 `torch.load` 读取源数据，并将结果保存到 `resume_checkpoint`。
            resume_checkpoint = torch.load(
                last_path, map_location="cpu", weights_only=False
            )
            # 检查 `resume_checkpoint.get('run_signature') != signature`；根据条件结果选择对应处理分支。
            if resume_checkpoint.get("run_signature") != signature:
                # 检查 `not is_resume_compatible_recovery_patch(resume_checkpoint.get('run_payload', {}), payload)`；根据条件结果选择对应处理分支。
                if not is_resume_compatible_recovery_patch(
                    resume_checkpoint.get("run_payload", {}), payload
                ):
                    # 抛出 `ValueError(f'{last_path} 与当前配置签名不一致；请使用新 run name，或显式设置 --overwrite 1')`，阻止不满足数据契约的流程继续执行。
                    raise ValueError(
                        f"{last_path} 与当前配置签名不一致；"
                        "请使用新 run name，或显式设置 --overwrite 1"
                    )
                # 调用 `print` 记录当前流程状态或诊断信息。
                print(
                    f"[{year}/seed={seed}] 旧 checkpoint 仅训练代码哈希不同；"
                    "应用 validation-score-collapse 恢复迁移。",
                    flush=True,
                )
        # 调用 `sequence_data.atomic_write_json` 先写临时文件再原子替换正式产物，避免留下半成品。
        if primary:
            # 执行当前分布式训练步骤并保持既定数据语义。
            sequence_data.atomic_write_json(
                {
                    "signature": sequence_data.stable_hash(
                        {k: v for k, v in payload.items() if k != "seed"}
                    ),
                    "payload": {k: v for k, v in payload.items() if k != "seed"},
                },
                model_root / f"forecast_{year}" / "model_config.json",
            )
        # 执行当前分布式训练步骤并保持既定数据语义。
        if distributed_context is not None:
            # 执行当前分布式训练步骤并保持既定数据语义。
            dist.barrier()

    # 调用 `build_training_model` 根据当前配置和输入构造后续流程所需对象，并将结果保存到 `model`。
    model = build_training_model(
        device, use_compile=bool(USE_TORCH_COMPILE and persist)
    )
    # 执行当前分布式训练步骤并保持既定数据语义。
    if distributed_context is not None:
        # 执行当前分布式训练步骤并保持既定数据语义。
        model = distributed.wrap_model(model, distributed_context)
    # 调用 `base.build_optimizer` 根据训练配置为可学习参数创建优化器，并将结果保存到 `optimizer`。
    optimizer = base.build_optimizer(
        model, config, bool(USE_FUSED_ADAM), device
    )
    # 设置 `start_epoch` 的固定配置值，供后续流程统一使用。
    start_epoch = 1
    # 设置 `best_epoch` 的固定配置值，供后续流程统一使用。
    best_epoch = 0
    # 计算 `-np.inf`，并将数值结果保存到 `best_validation_ic`。
    best_validation_ic = -np.inf
    # 构造键值映射，并保存到 `best_metrics`，供后续按键查询。
    best_metrics: dict[str, float] = {}
    # 设置 `epochs_without_improvement` 的固定配置值，供后续流程统一使用。
    epochs_without_improvement = 0
    # 构造当前步骤所需的列表，并保存到 `history`。
    history: list[dict[str, Any]] = []

    # 检查 `resume_checkpoint is not None`；根据条件结果选择对应处理分支。
    if resume_checkpoint is not None:
        # 读取 `resume_checkpoint` 的当前值，并赋给 `checkpoint`。
        checkpoint = resume_checkpoint
        # 调用 `base.load_canonical_state` 读取并解析后续计算所需的数据或运行状态。
        base.load_canonical_state(model, checkpoint["model_state"])
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
            distributed.restore_rng_state(states[distributed_context.rank], device)
        # 执行当前分布式训练步骤并保持既定数据语义。
        else:
            # 执行当前分布式训练步骤并保持既定数据语义。
            random.setstate(checkpoint["python_random_state"])
            # 执行当前分布式训练步骤并保持既定数据语义。
            np.random.set_state(checkpoint["numpy_random_state"])
            # 执行当前分布式训练步骤并保持既定数据语义。
            torch.set_rng_state(checkpoint["torch_rng_state"])
            # 执行当前分布式训练步骤并保持既定数据语义。
            if checkpoint["cuda_rng_state"] is not None:
                # 执行当前分布式训练步骤并保持既定数据语义。
                torch.cuda.set_rng_state(checkpoint["cuda_rng_state"], device)
        # 计算 `int(checkpoint['epoch']) + 1`，并将数值结果保存到 `start_epoch`。
        start_epoch = int(checkpoint["epoch"]) + 1
        # 调用 `int` 统一设备或数据类型，并将结果保存到 `best_epoch`。
        best_epoch = int(checkpoint["best_epoch"])
        # 调用 `float` 统一设备或数据类型，并将结果保存到 `best_validation_ic`。
        best_validation_ic = float(checkpoint["best_validation_ic"])
        # 构造键值映射，并保存到 `best_metrics`，供后续按键查询。
        best_metrics = {
            k: float(v) for k, v in checkpoint["best_metrics"].items()
        }
        # 调用 `int` 统一设备或数据类型，并将结果保存到 `epochs_without_improvement`。
        epochs_without_improvement = int(
            checkpoint["epochs_without_improvement"]
        )
        # 检查 `history_path.exists()`；根据条件结果选择对应处理分支。
        if history_path.exists():
            # 调用 `pd.read_parquet(history_path).to_dict` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `history`。
            history = pd.read_parquet(history_path).to_dict("records")
            # 调用 `int` 统一设备或数据类型，并将结果保存到 `checkpoint_epoch`。
            checkpoint_epoch = int(checkpoint["epoch"])
            # 构造当前步骤所需的列表，并保存到 `history`。
            history = [
                row for row in history
                if int(row["epoch"]) <= checkpoint_epoch
            ]
        # 调用 `print` 记录当前流程状态或诊断信息。
        print(f"[{year}/seed={seed}] 从 epoch {start_epoch} 续训。", flush=True)

    # 调用 `time.perf_counter` 读取高精度单调时钟以计算耗时，并将结果保存到 `started`。
    started = time.perf_counter()
    # 设置 `stopped_early` 的固定配置值，供后续流程统一使用。
    stopped_early = False
    # 设置 `stop_reason` 的固定配置值，供后续流程统一使用。
    stop_reason = "max_epochs"
    # 设置 `validation_score_collapsed` 的固定配置值，供后续流程统一使用。
    validation_score_collapsed = False
    # 设置 `collapsed_validation_date_id` 的固定配置值，供后续流程统一使用。
    collapsed_validation_date_id: int | None = None
    # 设置 `collapsed_validation_score_std` 的固定配置值，供后续流程统一使用。
    collapsed_validation_score_std: float | None = None
    # 设置 `collapsed_validation_total_sum_squares` 的固定配置值，供后续流程统一使用。
    collapsed_validation_total_sum_squares: float | None = None
    # 计算 `start_epoch - 1`，并将数值结果保存到 `last_successful_epoch`。
    last_successful_epoch = start_epoch - 1
    # 遍历 `range(start_epoch, epoch_limit + 1)`，并把当前元素绑定到 `epoch` 逐项处理。
    for epoch in range(start_epoch, epoch_limit + 1):
        # 调用 `time.perf_counter` 读取高精度单调时钟以计算耗时，并将结果保存到 `epoch_started`。
        epoch_started = time.perf_counter()
        # 调用 `sequence_data.set_train_epoch` 遍历训练集完成一个 epoch 的前向、反向传播和参数更新。
        sequence_data.set_train_epoch(train_loader, epoch)
        # 调用 `train_one_epoch` 遍历训练集完成一个 epoch 的前向、反向传播和参数更新，并将结果保存到 `train_metrics`。
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch=epoch,
            industry_class_weights=industry_class_weights,
            max_batches=max_batches,
            distributed_context=distributed_context,
        )
        # 尝试执行本段主逻辑；若发生异常，则交由后续异常分支处理。
        try:
            # 调用 `evaluate_model` 在不更新参数的条件下计算验证或样本外评价指标，并将结果保存到 `validation_metrics`。
            validation_metrics = evaluate_model(
                model,
                validation_loader,
                device,
                industry_class_weights=industry_class_weights,
                max_batches=max_batches,
                distributed_context=distributed_context,
            )
        # 捕获 `except DegenerateValidationScoreError as exc` 对应的异常，并执行明确的降级、记录或清理逻辑。
        except DegenerateValidationScoreError as exc:
            # 检查 `not persist or best_epoch <= 0 or (not best_path.exists())`；根据条件结果选择对应处理分支。
            if not persist or best_epoch <= 0 or not best_path.exists():
                # 抛出 `空值`，阻止不满足数据契约的流程继续执行。
                raise
            # 设置 `stopped_early` 的固定配置值，供后续流程统一使用。
            stopped_early = True
            # 设置 `stop_reason` 的固定配置值，供后续流程统一使用。
            stop_reason = "validation_score_collapse"
            # 设置 `validation_score_collapsed` 的固定配置值，供后续流程统一使用。
            validation_score_collapsed = True
            # 读取 `exc.date_id` 的当前值，并赋给 `collapsed_validation_date_id`。
            collapsed_validation_date_id = exc.date_id
            # 读取 `exc.score_std` 的当前值，并赋给 `collapsed_validation_score_std`。
            collapsed_validation_score_std = exc.score_std
            # 读取 `exc.total_sum_squares` 的当前值，并赋给 `collapsed_validation_total_sum_squares`。
            collapsed_validation_total_sum_squares = exc.total_sum_squares
            # 把本轮结果追加到 `history`，累积后续需要汇总的记录。
            history.append(
                {
                    "forecast_year": year,
                    "seed": seed,
                    "epoch": epoch,
                    "loss_name": LOSS_NAME,
                    "residualize_target": RESIDUALIZE_TARGET,
                    "project_model_output": PROJECT_MODEL_OUTPUT,
                    "style_correlation_penalty": raw_barra_r2_enabled(),
                    "adversarial_barra": adversarial_barra_enabled(),
                    "raw_barra_r2_lambda": RAW_BARRA_R2_LAMBDA,
                    **train_metrics,
                    "validation_ic": np.nan,
                    "validation_daily_rank_ic": np.nan,
                    "validation_original_daily_rank_ic": np.nan,
                    "validation_effective_max_abs_style_pearson": np.nan,
                    "validation_barra_r_squared_mean": np.nan,
                    "validation_score_collapsed": True,
                    "validation_collapsed_date_count": 1,
                    "validation_collapsed_date_id": exc.date_id,
                    "validation_min_daily_raw_score_std": exc.score_std,
                    "validation_collapse_total_sum_squares": (
                        exc.total_sum_squares
                    ),
                    "best_validation_ic": best_validation_ic,
                    "epochs_without_improvement": epochs_without_improvement,
                    "epoch_seconds": time.perf_counter() - epoch_started,
                }
            )
            # 调用 `sequence_data.atomic_write_parquet` 先写临时文件再原子替换正式产物，避免留下半成品。
            if primary:
                # 执行当前分布式训练步骤并保持既定数据语义。
                sequence_data.atomic_write_parquet(pd.DataFrame(history), history_path)
            # 执行当前分布式训练步骤并保持既定数据语义。
            if distributed_context is not None:
                # 执行当前分布式训练步骤并保持既定数据语义。
                dist.barrier()
            # 调用 `print` 记录当前流程状态或诊断信息。
            print(
                f"[{year}/seed={seed}] epoch={epoch:03d} 验证分数退化；"
                f"date_id={exc.date_id}, std={exc.score_std:.3e}。"
                f"停止当前 seed，并使用 best epoch={best_epoch} 生成预测。",
                flush=True,
            )
            # 终止当前循环，避免继续处理已经不满足条件的元素。
            break
        # 调用 `float` 统一设备或数据类型，并将结果保存到 `selected`。
        selected = float(validation_metrics["validation_ic"])
        # 根据 `selected > best_validation_ic + config.improvement_tolerance` 计算条件结果，并保存到 `improved`。
        improved = selected > best_validation_ic + config.improvement_tolerance
        # 检查 `improved`；根据条件结果选择对应处理分支。
        if improved:
            # 读取 `epoch` 的当前值，并赋给 `best_epoch`。
            best_epoch = epoch
            # 读取 `selected` 的当前值，并赋给 `best_validation_ic`。
            best_validation_ic = selected
            # 构造键值映射，并保存到 `best_metrics`，供后续按键查询。
            best_metrics = {
                key: float(value)
                for key, value in validation_metrics.items()
                if isinstance(value, (int, float))
            }
            # 设置 `epochs_without_improvement` 的固定配置值，供后续流程统一使用。
            epochs_without_improvement = 0
        # 当前置条件不成立时进入备用分支，保证其余合法情况仍被处理。
        else:
            # 用 `1` 对 `epochs_without_improvement` 做 `Add` 累积更新。
            epochs_without_improvement += 1
        # 构造键值映射，并保存到 `record`，供后续按键查询。
        record = {
            "forecast_year": year,
            "seed": seed,
            "epoch": epoch,
            "loss_name": LOSS_NAME,
            "residualize_target": RESIDUALIZE_TARGET,
            "project_model_output": PROJECT_MODEL_OUTPUT,
            "style_correlation_penalty": raw_barra_r2_enabled(),
            "adversarial_barra": adversarial_barra_enabled(),
            "raw_barra_r2_lambda": RAW_BARRA_R2_LAMBDA,
            **train_metrics,
            **validation_metrics,
            "best_validation_ic": best_validation_ic,
            "epochs_without_improvement": epochs_without_improvement,
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        # 把本轮结果追加到 `history`，累积后续需要汇总的记录。
        history.append(record)
        # 检查 `persist`；根据条件结果选择对应处理分支。
        if persist:
            # 调用 `sequence_data.atomic_write_parquet` 先写临时文件再原子替换正式产物，避免留下半成品。
            if primary:
                # 执行当前分布式训练步骤并保持既定数据语义。
                sequence_data.atomic_write_parquet(pd.DataFrame(history), history_path)
            # 执行当前分布式训练步骤并保持既定数据语义。
            rng_state_by_rank = (
                distributed.gather_rng_states(device)
                if distributed_context is not None else None
            )
            # 构造键值映射，并保存到 `checkpoint`，供后续按键查询。
            checkpoint = {
                "model_state": base.unwrap_model(model).state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "python_random_state": random.getstate(),
                "numpy_random_state": np.random.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state(device),
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_validation_ic": best_validation_ic,
                "best_metrics": best_metrics,
                "epochs_without_improvement": epochs_without_improvement,
                "run_signature": signature,
                "run_payload": payload,
            }
            # 执行当前分布式训练步骤并保持既定数据语义。
            if rng_state_by_rank is not None:
                # 执行当前分布式训练步骤并保持既定数据语义。
                checkpoint["rng_state_by_rank"] = rng_state_by_rank
            # 调用 `sequence_data.atomic_save_checkpoint` 按约定格式写出当前计算产物。
            if primary:
                # 执行当前分布式训练步骤并保持既定数据语义。
                sequence_data.atomic_save_checkpoint(checkpoint, last_path)
            # 检查 `improved`；根据条件结果选择对应处理分支。
            if primary and improved:
                # 调用 `sequence_data.atomic_save_checkpoint` 按约定格式写出当前计算产物。
                sequence_data.atomic_save_checkpoint(checkpoint, best_path)
            # 执行当前分布式训练步骤并保持既定数据语义。
            if distributed_context is not None:
                # 执行当前分布式训练步骤并保持既定数据语义。
                dist.barrier()
        # 读取 `epoch` 的当前值，并赋给 `last_successful_epoch`。
        last_successful_epoch = epoch
        # 调用 `print` 记录当前流程状态或诊断信息。
        print(
            f"[{year}/seed={seed}] epoch={epoch:03d} "
            f"loss={train_metrics['train_loss']:.6f} "
            f"rankLoss={train_metrics['train_rank_loss']:.6f} "
            f"trainBarraR2={train_metrics['train_barra_r_squared']:.6f} "
            f"gamma={train_metrics['train_adversarial_gamma']:.4f} "
            f"styleAdv={train_metrics['train_adversarial_style_loss']:.4f} "
            f"industryAdv={train_metrics['train_adversarial_industry_loss']:.4f} "
            f"validEffectiveRank={validation_metrics['validation_daily_rank_ic']:.6f} "
            f"validOriginalRank={validation_metrics['validation_original_daily_rank_ic']:.6f} "
            f"effectiveStylePearsonMax={validation_metrics['validation_effective_max_abs_style_pearson']:.2e} "
            f"validRawBarraR2={validation_metrics['validation_barra_r_squared_mean']:.6f} "
            f"validEffectiveBarraR2="
            f"{validation_metrics['validation_effective_barra_r_squared_mean']:.2e} "
            f"effectiveIndustryMeanMax="
            f"{validation_metrics['validation_effective_max_abs_industry_mean']:.2e} "
            f"best={best_validation_ic:.6f} "
            f"patience={epochs_without_improvement}/{PATIENCE}",
            flush=True,
        )
        # 检查 `epochs_without_improvement >= PATIENCE`；根据条件结果选择对应处理分支。
        if epochs_without_improvement >= PATIENCE:
            # 设置 `stopped_early` 的固定配置值，供后续流程统一使用。
            stopped_early = True
            # 设置 `stop_reason` 的固定配置值，供后续流程统一使用。
            stop_reason = "patience"
            # 终止当前循环，避免继续处理已经不满足条件的元素。
            break

    # 检查 `not persist`；根据条件结果选择对应处理分支。
    if not persist:
        # 将 `{'forecast_year': year, 'seed': seed, 'device': str(device), 'train_metrics': train_metrics, 'v…` 作为当前函数结果返回给调用方。
        return {
            "forecast_year": year,
            "seed": seed,
            "device": str(device),
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
            "persistent_outputs_written": False,
        }
    # 检查 `not best_path.exists()`；根据条件结果选择对应处理分支。
    if not best_path.exists():
        # 抛出 `FileNotFoundError(f'未生成最佳 checkpoint：{best_path}')`，阻止不满足数据契约的流程继续执行。
        raise FileNotFoundError(f"未生成最佳 checkpoint：{best_path}")
    # 检查 `prediction_loader is not None`；根据条件结果选择对应处理分支。
    if prediction_loader is not None:
        # 调用 `load_best_model` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `best_model`。
        best_model = load_best_model(best_path, device)
        # 执行当前分布式训练步骤并保持既定数据语义。
        prediction_frame = predict_model(
            best_model, prediction_loader, device,
            distributed_context=distributed_context,
        )
        # 调用 `sequence_data.atomic_write_parquet` 先写临时文件再原子替换正式产物，避免留下半成品。
        if primary:
            # 执行当前分布式训练步骤并保持既定数据语义。
            sequence_data.atomic_write_parquet(prediction_frame, prediction_path)
        # 删除 `best_model`，释放不再使用的引用。
        del best_model
        # 执行当前分布式训练步骤并保持既定数据语义。
        if distributed_context is not None:
            # 执行当前分布式训练步骤并保持既定数据语义。
            dist.barrier()
    # 构造键值映射，并保存到 `probe_summary`，供后续按键查询。
    probe_summary: dict[str, float] = {}
    # 检查 `adversarial_barra_enabled()`；根据条件结果选择对应处理分支。
    if primary and adversarial_barra_enabled():
        # 执行当前分布式训练步骤并保持既定数据语义。
        probe_validation_loader = validation_loader
        # 执行当前分布式训练步骤并保持既定数据语义。
        if distributed_context is not None:
            # 执行当前分布式训练步骤并保持既定数据语义。
            _, probe_validation_loader, _ = make_style_loaders(
                year, seed, 0, pin_memory=True,
                distributed_context=None,
            )
        # 调用 `load_best_training_model` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `probe_model`。
        probe_model = load_best_training_model(best_path, device)
        # 调用 `run_adversarial_probes` 构建或运行可解释性探针以测量表征中的风险信息，并将结果保存到 `probe_frame`。
        probe_frame = run_adversarial_probes(
            probe_model,
            probe_validation_loader,
            device,
            year=year,
            seed=seed,
        )
        # 调用 `sequence_data.atomic_write_parquet` 先写临时文件再原子替换正式产物，避免留下半成品。
        sequence_data.atomic_write_parquet(probe_frame, probe_path)
        # 调用 `probe_frame[probe_frame['target_kind'].eq('style_summary')].set…` 按约定截面或分组口径汇总统计结果，并将结果保存到 `style_summary`。
        style_summary = probe_frame[
            probe_frame["target_kind"].eq("style_summary")
        ].set_index("probe")
        # 调用 `probe_frame[probe_frame['target_kind'].eq('industry')].set_index` 把指定字段设为索引以便按名称查询，并将结果保存到 `industry_summary`。
        industry_summary = probe_frame[
            probe_frame["target_kind"].eq("industry")
        ].set_index("probe")
        # 构造键值映射，并保存到 `probe_summary`，供后续按键查询。
        probe_summary = {
            "linear_probe_style_r_squared_mean": float(
                style_summary.loc["linear_ridge", "r_squared"]
            ),
            "mlp_probe_style_r_squared_mean": float(
                style_summary.loc["mlp", "r_squared"]
            ),
            "linear_probe_industry_balanced_accuracy": float(
                industry_summary.loc["linear_softmax", "balanced_accuracy"]
            ),
            "mlp_probe_industry_balanced_accuracy": float(
                industry_summary.loc["mlp", "balanced_accuracy"]
            ),
        }
        # 删除 `probe_model`，释放不再使用的引用。
        del probe_model
    # 构造键值映射，并保存到 `summary`，供后续按键查询。
    summary = {
        "forecast_year": year,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_ic": best_validation_ic,
        "best_validation_rank_ic": best_metrics.get(
            "validation_daily_rank_ic"
        ),
        "best_validation_original_rank_ic": best_metrics.get(
            "validation_original_daily_rank_ic"
        ),
        "best_validation_effective_max_abs_style_pearson": best_metrics.get(
            "validation_effective_max_abs_style_pearson"
        ),
        "best_validation_barra_r_squared": best_metrics.get(
            "validation_barra_r_squared_mean"
        ),
        "best_validation_effective_barra_r_squared": best_metrics.get(
            "validation_effective_barra_r_squared_mean"
        ),
        "best_validation_effective_max_abs_industry_mean": best_metrics.get(
            "validation_effective_max_abs_industry_mean"
        ),
        "best_validation_gate_mean_abs": best_metrics.get(
            "validation_gate_mean_abs"
        ),
        "best_validation_gate_max_abs": best_metrics.get(
            "validation_gate_max_abs"
        ),
        "best_validation_total_original_rank_ic": best_metrics.get(
            "validation_total_original_daily_rank_ic"
        ),
        "best_validation_style_fitted_rank_ic": best_metrics.get(
            "validation_style_fitted_daily_rank_ic"
        ),
        "best_validation_dual_additivity_max_abs_error": best_metrics.get(
            "validation_dual_additivity_max_abs_error"
        ),
        "best_validation_adversarial_style_r_squared_mean": best_metrics.get(
            "validation_adversarial_style_r_squared_mean"
        ),
        "best_validation_adversarial_industry_accuracy": best_metrics.get(
            "validation_adversarial_industry_accuracy"
        ),
        "best_validation_adversarial_industry_balanced_accuracy": (
            best_metrics.get(
                "validation_adversarial_industry_balanced_accuracy"
            )
        ),
        "last_epoch": int(last_successful_epoch),
        "last_attempted_epoch": int(history[-1]["epoch"]),
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "validation_score_collapsed": validation_score_collapsed,
        "collapsed_validation_date_count": int(validation_score_collapsed),
        "collapsed_validation_date_id": collapsed_validation_date_id,
        "collapsed_validation_score_std": collapsed_validation_score_std,
        "collapsed_validation_total_sum_squares": (
            collapsed_validation_total_sum_squares
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "physical_gpu": physical_gpu,
        "run_name": RUN_NAME,
        "execution_route": EXECUTION_ROUTE,
        "world_size": DDP_WORLD_SIZE,
        "physical_device_ids": list(EXECUTION_DEVICE_IDS),
        "loss_name": LOSS_NAME,
        "residualize_target": RESIDUALIZE_TARGET,
        "project_model_output": PROJECT_MODEL_OUTPUT,
        "style_correlation_penalty": raw_barra_r2_enabled(),
        "adversarial_barra": adversarial_barra_enabled(),
        "adversarial_gamma_max": ADVERSARIAL_GAMMA_MAX,
        "barra_conditioning_mode": BARRA_CONDITIONING_MODE,
        "barra_output_neutralization": strict_barra_output_enabled(),
        "barra_dual_tower": dual_tower_enabled(),
        "barra_dual_tower_mode": BARRA_DUAL_TOWER_MODE,
        "dual_tower_alpha_loss_weight": DUAL_TOWER_ALPHA_LOSS_WEIGHT,
        "dual_tower_style_loss_weight": DUAL_TOWER_STYLE_LOSS_WEIGHT,
        "barra_gate_scale": BARRA_GATE_SCALE,
        **probe_summary,
        "raw_barra_r2_lambda": RAW_BARRA_R2_LAMBDA,
        "prediction_sample_count": len(
            sequence_data.load_sequence_split(year)["prediction_indices"]
        ),
        "status": "completed",
    }
    # 调用 `sequence_data.atomic_write_json` 先写临时文件再原子替换正式产物，避免留下半成品。
    if distributed_context is not None:
        # 执行当前分布式训练步骤并保持既定数据语义。
        dist.barrier()
    # 执行当前分布式训练步骤并保持既定数据语义。
    if primary:
        # 执行当前分布式训练步骤并保持既定数据语义。
        sequence_data.atomic_write_json(
            {"run_signature": signature, "summary": summary}, completed_path
        )
    # 执行当前分布式训练步骤并保持既定数据语义。
    if distributed_context is not None:
        # 执行当前分布式训练步骤并保持既定数据语义。
        dist.barrier()
    # 将 `summary` 作为当前函数结果返回给调用方。
    return summary


# %% [markdown]
# ## 步骤 5：三 seed 集成、风格相关性审计和 run manifest


# %%
# 定义 `canonical_dual_tower_total_scores` 函数，封装这一阶段可独立调用的处理步骤。
def canonical_dual_tower_total_scores(
    alpha_values: np.ndarray,
    style_values: np.ndarray,
    stored_total_values: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """审计推理精度，并以落盘分量重建严格可加的 Total。
    
    输入：
        alpha_values: np.ndarray。
        style_values: np.ndarray。
        stored_total_values: np.ndarray。
    输出：
        tuple[np.ndarray, dict[str, Any]]。
        """

    # 调用 `np.asarray` 转换数据表示，并将标准化对象保存到 `alpha`。
    alpha = np.asarray(alpha_values, dtype=np.float32)
    # 调用 `np.asarray` 转换数据表示，并将标准化对象保存到 `style`。
    style = np.asarray(style_values, dtype=np.float32)
    # 调用 `np.asarray` 转换数据表示，并将标准化对象保存到 `stored`。
    stored = np.asarray(stored_total_values, dtype=np.float32)
    # 检查 `alpha.shape != style.shape or alpha.shape != stored.shape`；根据条件结果选择对应处理分支。
    if alpha.shape != style.shape or alpha.shape != stored.shape:
        # 抛出 `ValueError('No.6 Alpha、Style、Total 形状不一致')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("No.6 Alpha、Style、Total 形状不一致")
    # 检查 `not np.isfinite(np.column_stack([alpha, style, stored])).all()`；根据条件结果选择对应处理分支。
    if not np.isfinite(np.column_stack([alpha, style, stored])).all():
        # 抛出 `ValueError('No.6 Alpha、Style、Total 含 NaN/Inf')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("No.6 Alpha、Style、Total 含 NaN/Inf")

    # 调用 `(alpha + style).astype` 统一设备或数据类型，并将结果保存到 `canonical`。
    canonical = (alpha + style).astype(np.float32)
    # 调用 `numpy` 把张量转回 NumPy 数组用于 CPU 统计，并将结果保存到 `bfloat16_sum`。
    bfloat16_sum = (
        torch.from_numpy(alpha.copy()).to(torch.bfloat16)
        + torch.from_numpy(style.copy()).to(torch.bfloat16)
    ).float().numpy()
    # 调用 `np.array_equal` 逐元素检查两个数组是否完全一致，并将结果保存到 `matches_float32`。
    matches_float32 = np.array_equal(stored, canonical)
    # 调用 `np.array_equal` 逐元素检查两个数组是否完全一致，并将结果保存到 `matches_bfloat16`。
    matches_bfloat16 = np.array_equal(stored, bfloat16_sum)
    # 检查 `not matches_float32 and (not matches_bfloat16)`；根据条件结果选择对应处理分支。
    if not matches_float32 and not matches_bfloat16:
        # 抛出 `AssertionError('No.6 推理 Total 既不满足 float32 加法，也不满足 bfloat16 加法')`，阻止不满足数据契约的流程继续执行。
        raise AssertionError(
            "No.6 推理 Total 既不满足 float32 加法，也不满足 bfloat16 加法"
        )
    # 调用 `np.abs` 逐元素取绝对值，并将结果保存到 `error`。
    error = np.abs(stored.astype(np.float64) - canonical.astype(np.float64))
    # 将 `(canonical, {'stored_total_precision': 'float32' if matches_float32 else 'bfloat16', 'stored_vs…` 作为当前函数结果返回给调用方。
    return canonical, {
        "stored_total_precision": (
            "float32" if matches_float32 else "bfloat16"
        ),
        "stored_vs_canonical_max_abs_error": float(error.max()),
        "stored_vs_canonical_mean_abs_error": float(error.mean()),
    }


# 定义 `build_ensemble` 函数，封装这一阶段可独立调用的处理步骤。
def build_ensemble(year: int) -> tuple[int, pd.DataFrame]:
    """按既定尺度组合多个随机种子的预测，并保存正式年度信号。
    
    输入：
        year: int。
    输出：
        tuple[int, pd.DataFrame]。
        """
    # 调用 `base.run_paths` 解析当前 run 对应的数据与输出路径，并将结果保存到 `(model_root, prediction_root, _)`。
    model_root, prediction_root, _ = base.run_paths(RUN_NAME)
    # 调用 `sequence_data.load_sequence_split` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `split`。
    split = sequence_data.load_sequence_split(year)
    # 检查 `not len(split['prediction_indices'])`；根据条件结果选择对应处理分支。
    if not len(split["prediction_indices"]):
        # 将 `(0, pd.DataFrame())` 作为当前函数结果返回给调用方。
        return 0, pd.DataFrame()
    # 构造键值映射，并保存到 `frames`，供后续按键查询。
    frames = {
        seed: pd.read_parquet(
            model_root / f"forecast_{year}" / f"seed_{seed}"
            / "prediction.parquet"
        )
        for seed in SEEDS
    }
    # 调用 `frames[SEEDS[0]]['sample_index'].to_numpy` 把表格列转换为指定精度的 NumPy 数组，并将结果保存到 `reference`。
    reference = frames[SEEDS[0]]["sample_index"].to_numpy(np.int64)
    # 调用 `pd.DataFrame` 转换数据表示，并将标准化对象保存到 `output`。
    output = pd.DataFrame({"sample_index": reference})
    # 构造当前步骤所需的列表，并保存到 `raw_columns`。
    raw_columns: list[str] = []
    # 构造当前步骤所需的列表，并保存到 `effective_columns`。
    effective_columns: list[str] = []
    # 构造当前步骤所需的列表，并保存到 `dual_style_columns`。
    dual_style_columns: list[str] = []
    # 构造当前步骤所需的列表，并保存到 `dual_total_columns`。
    dual_total_columns: list[str] = []
    # 构造键值映射，并保存到 `dual_inference_audits`，供后续按键查询。
    dual_inference_audits: dict[int, dict[str, Any]] = {}
    # 遍历 `SEEDS`，并把当前元素绑定到 `seed` 逐项处理。
    for seed in SEEDS:
        # 读取 `frames[seed]` 的当前值，并赋给 `frame`。
        frame = frames[seed]
        # 检查 `not np.array_equal(reference, frame['sample_index'].to_numpy(np.int64))`；根据条件结果选择对应处理分支。
        if not np.array_equal(
            reference, frame["sample_index"].to_numpy(np.int64)
        ):
            # 抛出 `AssertionError(f'{year} 三个 seed 的预测索引不一致')`，阻止不满足数据契约的流程继续执行。
            raise AssertionError(f"{year} 三个 seed 的预测索引不一致")
        # 计算 `f'prediction_seed_{seed}_unprojected'`，并将结果写入 `raw_column`。
        raw_column = f"prediction_seed_{seed}_unprojected"
        # 计算 `f'prediction_seed_{seed}'`，并将结果写入 `effective_column`。
        effective_column = f"prediction_seed_{seed}"
        # 调用 `frame['prediction_model_raw'].to_numpy` 执行模型推理并生成样本预测分数，并将结果保存到 `output[raw_column]`。
        output[raw_column] = frame[
            "prediction_model_raw"
        ].to_numpy(np.float32)
        # 调用 `frame['prediction'].to_numpy` 执行模型推理并生成样本预测分数，并将结果保存到 `output[effective_column]`。
        output[effective_column] = frame["prediction"].to_numpy(np.float32)
        # 把本轮结果追加到 `raw_columns`，累积后续需要汇总的记录。
        raw_columns.append(raw_column)
        # 把本轮结果追加到 `effective_columns`，累积后续需要汇总的记录。
        effective_columns.append(effective_column)
        # 检查 `dual_tower_enabled()`；根据条件结果选择对应处理分支。
        if dual_tower_enabled():
            # 计算 `f'prediction_seed_{seed}_style'`，并将结果写入 `style_column`。
            style_column = f"prediction_seed_{seed}_style"
            # 计算 `f'prediction_seed_{seed}_total_return'`，并将结果写入 `total_column`。
            total_column = f"prediction_seed_{seed}_total_return"
            # 调用 `frame['prediction_style_component'].to_numpy` 执行模型推理并生成样本预测分数，并将结果保存到 `style_values`。
            style_values = frame[
                "prediction_style_component"
            ].to_numpy(np.float32)
            # 调用 `frame['prediction_total_return'].to_numpy` 执行模型推理并生成样本预测分数，并将结果保存到 `stored_total_values`。
            stored_total_values = frame[
                "prediction_total_return"
            ].to_numpy(np.float32)
            # 调用 `canonical_dual_tower_total_scores` 通过当前网络子模块变换输入表征，并将结果保存到 `(total_values, dual_inference_audits[seed])`。
            total_values, dual_inference_audits[seed] = (
                canonical_dual_tower_total_scores(
                    output[raw_column].to_numpy(np.float32),
                    style_values,
                    stored_total_values,
                )
            )
            # 读取 `style_values` 的当前值，并赋给 `output[style_column]`。
            output[style_column] = style_values
            # 读取 `total_values` 的当前值，并赋给 `output[total_column]`。
            output[total_column] = total_values
            # 把本轮结果追加到 `dual_style_columns`，累积后续需要汇总的记录。
            dual_style_columns.append(style_column)
            # 把本轮结果追加到 `dual_total_columns`，累积后续需要汇总的记录。
            dual_total_columns.append(total_column)

    # 调用 `pd.read_parquet(sequence_data.SAMPLE_INDEX_PATH, columns=['samp…` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `metadata`。
    metadata = pd.read_parquet(
        sequence_data.SAMPLE_INDEX_PATH,
        columns=["sample_index", "trade_date", "instrument_id"],
    ).set_index("sample_index")
    # 调用 `metadata.loc[reference].reset_index().merge` 按键合并数据，并将对齐后的结果保存到 `output`。
    output = metadata.loc[reference].reset_index().merge(
        output, on="sample_index", validate="one_to_one"
    )

    # 定义 `daily_zscore` 函数，封装这一阶段可独立调用的处理步骤。
    def daily_zscore(column: str) -> str:
        """按交易日用 Alpha 分数尺度标准化双塔分量，保持加法恒等式。
        
        输入：
            column: str。
        输出：
            str。
            """
        # 调用 `output[column].astype(np.float64).groupby` 按指定字段建立分组对象，并将结果保存到 `grouped`。
        grouped = output[column].astype(np.float64).groupby(
            output["trade_date"], sort=False
        )
        # 调用 `grouped.transform` 按约定截面或分组口径汇总统计结果，并将结果保存到 `mean`。
        mean = grouped.transform("mean")
        # 调用 `grouped.transform` 按约定截面或分组口径汇总统计结果，并将结果保存到 `std`。
        std = grouped.transform("std", ddof=0)
        # 检查 `(std <= 0).any() or not np.isfinite(std).all()`；根据条件结果选择对应处理分支。
        if (std <= 0).any() or not np.isfinite(std).all():
            # 抛出 `ValueError(f'{year} {column} 存在常数或非有限日截面')`，阻止不满足数据契约的流程继续执行。
            raise ValueError(f"{year} {column} 存在常数或非有限日截面")
        # 计算 `f'{column}_zscore'`，并将结果写入 `result_column`。
        result_column = f"{column}_zscore"
        # 调用 `((output[column] - mean) / std).astype` 统一设备或数据类型，并将结果保存到 `output[result_column]`。
        output[result_column] = (
            (output[column] - mean) / std
        ).astype(np.float32)
        # 将 `result_column` 作为当前函数结果返回给调用方。
        return result_column

    # 检查 `dual_tower_enabled()`；根据条件结果选择对应处理分支。
    if dual_tower_enabled():
        # 构造当前步骤所需的列表，并保存到 `alpha_scaled`。
        alpha_scaled: list[str] = []
        # 构造当前步骤所需的列表，并保存到 `style_scaled`。
        style_scaled: list[str] = []
        # 构造当前步骤所需的列表，并保存到 `total_scaled`。
        total_scaled: list[str] = []
        # 遍历 `zip(SEEDS, raw_columns, dual_style_columns)`，并把当前元素绑定到 `(seed, alpha_column, style_column)` 逐项处理。
        for seed, alpha_column, style_column in zip(
            SEEDS, raw_columns, dual_style_columns
        ):
            # 调用 `output[alpha_column].astype(np.float64).groupby` 按指定字段建立分组对象，并将结果保存到 `alpha_grouped`。
            alpha_grouped = output[alpha_column].astype(np.float64).groupby(
                output["trade_date"], sort=False
            )
            # 调用 `alpha_grouped.transform` 按约定截面或分组口径汇总统计结果，并将结果保存到 `alpha_mean`。
            alpha_mean = alpha_grouped.transform("mean")
            # 调用 `alpha_grouped.transform` 按约定截面或分组口径汇总统计结果，并将结果保存到 `alpha_std`。
            alpha_std = alpha_grouped.transform("std", ddof=0)
            # 检查 `(alpha_std <= 0).any() or not np.isfinite(alpha_std).all()`；根据条件结果选择对应处理分支。
            if (alpha_std <= 0).any() or not np.isfinite(alpha_std).all():
                # 抛出 `ValueError(f'{year}/seed={seed} Alpha 日截面为常数')`，阻止不满足数据契约的流程继续执行。
                raise ValueError(f"{year}/seed={seed} Alpha 日截面为常数")
            # 调用 `transform` 把分组统计结果广播回每一条原始记录，并将结果保存到 `style_mean`。
            style_mean = output[style_column].astype(np.float64).groupby(
                output["trade_date"], sort=False
            ).transform("mean")
            # 计算 `f'{alpha_column}_alpha_scale'`，并将结果写入 `alpha_scaled_column`。
            alpha_scaled_column = f"{alpha_column}_alpha_scale"
            # 计算 `f'{style_column}_alpha_scale'`，并将结果写入 `style_scaled_column`。
            style_scaled_column = f"{style_column}_alpha_scale"
            # 计算 `f'prediction_seed_{seed}_total_alpha_scale'`，并将结果写入 `total_scaled_column`。
            total_scaled_column = f"prediction_seed_{seed}_total_alpha_scale"
            # 调用 `((output[alpha_column] - alpha_mean) / alpha_std).astype` 统一设备或数据类型，并将结果保存到 `output[alpha_scaled_column]`。
            output[alpha_scaled_column] = (
                (output[alpha_column] - alpha_mean) / alpha_std
            ).astype(np.float32)
            # 调用 `((output[style_column] - style_mean) / alpha_std).astype` 统一设备或数据类型，并将结果保存到 `output[style_scaled_column]`。
            output[style_scaled_column] = (
                (output[style_column] - style_mean) / alpha_std
            ).astype(np.float32)
            # 调用 `astype` 把数组转换为指定数据类型，并将结果保存到 `output[total_scaled_column]`。
            output[total_scaled_column] = (
                output[alpha_scaled_column] + output[style_scaled_column]
            ).astype(np.float32)
            # 把本轮结果追加到 `alpha_scaled`，累积后续需要汇总的记录。
            alpha_scaled.append(alpha_scaled_column)
            # 把本轮结果追加到 `style_scaled`，累积后续需要汇总的记录。
            style_scaled.append(style_scaled_column)
            # 把本轮结果追加到 `total_scaled`，累积后续需要汇总的记录。
            total_scaled.append(total_scaled_column)
        # 调用 `output[alpha_scaled].mean(axis=1).to_numpy` 把表格列转换为指定精度的 NumPy 数组，并将结果保存到 `final_alpha`。
        final_alpha = output[alpha_scaled].mean(axis=1).to_numpy(np.float64)
        # 调用 `output[style_scaled].mean(axis=1).to_numpy` 把表格列转换为指定精度的 NumPy 数组，并将结果保存到 `final_style`。
        final_style = output[style_scaled].mean(axis=1).to_numpy(np.float64)
        # 调用 `astype` 把数组转换为指定数据类型，并将结果保存到 `final_total`。
        final_total = (
            final_alpha.astype(np.float32) + final_style.astype(np.float32)
        ).astype(np.float32).astype(np.float64)
        # 读取 `final_alpha` 的当前值，并赋给 `final_raw`。
        final_raw = final_alpha
        # 读取 `final_alpha` 的当前值，并赋给 `final_effective`。
        final_effective = final_alpha
    # 当前置条件不成立时进入备用分支，保证其余合法情况仍被处理。
    else:
        # 构造当前步骤所需的列表，并保存到 `raw_zscore_columns`。
        raw_zscore_columns = [daily_zscore(column) for column in raw_columns]
        # 构造当前步骤所需的列表，并保存到 `effective_zscore_columns`。
        effective_zscore_columns = [
            daily_zscore(column) for column in effective_columns
        ]
        # 调用 `output[raw_zscore_columns].mean(axis=1).to_numpy` 把表格列转换为指定精度的 NumPy 数组，并将结果保存到 `final_raw`。
        final_raw = output[raw_zscore_columns].mean(axis=1).to_numpy(np.float64)
        # 调用 `output[effective_zscore_columns].mean(axis=1).to_numpy` 把表格列转换为指定精度的 NumPy 数组，并将结果保存到 `final_effective`。
        final_effective = (
            output[effective_zscore_columns].mean(axis=1).to_numpy(np.float64)
        )

    # 调用 `np.load` 读取源数据，并将结果保存到 `style_matrix`。
    style_matrix = np.load(
        style_data.DEFAULT_STORE_DIR / style_data.STYLE_EXPOSURES_FILE,
        mmap_mode="r",
    )
    # 调用 `np.load` 读取源数据，并将结果保存到 `style_valid`。
    style_valid = np.load(
        style_data.DEFAULT_STORE_DIR / style_data.STYLE_VALID_FILE,
        mmap_mode="r",
    )
    # 设置 `industry_codes` 的固定配置值，供后续流程统一使用。
    industry_codes = None
    # 检查 `industry_sidecar_required()`；根据条件结果选择对应处理分支。
    if industry_sidecar_required():
        # 调用 `style_data.require_industry_sidecar` 校验输入、配置或产物是否满足当前阶段的数据契约。
        style_data.require_industry_sidecar()
        # 调用 `np.load` 读取源数据，并将结果保存到 `industry_codes`。
        industry_codes = np.load(
            style_data.DEFAULT_INDUSTRY_STORE_DIR
            / style_data.INDUSTRY_CODES_FILE,
            mmap_mode="r",
        )
    # 设置 `dual_source_targets` 的固定配置值，供后续流程统一使用。
    dual_source_targets = None
    # 设置 `dual_residual_targets` 的固定配置值，供后续流程统一使用。
    dual_residual_targets = None
    # 设置 `dual_fitted_targets` 的固定配置值，供后续流程统一使用。
    dual_fitted_targets = None
    # 检查 `dual_tower_enabled()`；根据条件结果选择对应处理分支。
    if dual_tower_enabled():
        # 调用 `np.load` 读取源数据，并将结果保存到 `dual_source_targets`。
        dual_source_targets = np.load(sequence_data.TARGETS_PATH, mmap_mode="r")
        # 调用 `np.load` 读取源数据，并将结果保存到 `dual_residual_targets`。
        dual_residual_targets = np.load(
            style_data.DEFAULT_FULL_BARRA_TARGET_STORE_DIR
            / style_data.FULL_BARRA_TARGETS_FILE,
            mmap_mode="r",
        )
        # 调用 `np.load` 读取源数据，并将结果保存到 `dual_fitted_targets`。
        dual_fitted_targets = np.load(
            style_data.DEFAULT_DUAL_TOWER_TARGET_STORE_DIR
            / style_data.DUAL_TOWER_FITTED_TARGETS_FILE,
            mmap_mode="r",
        )
    # 构造当前步骤所需的列表，并保存到 `records`。
    records: list[dict[str, Any]] = []
    # 遍历 `output.groupby('trade_date', sort=False).groups.items()`，并把当前元素绑定到 `(trade_date, positions)` 逐项处理。
    for trade_date, positions in output.groupby(
        "trade_date", sort=False
    ).groups.items():
        # 调用 `np.asarray` 转换数据表示，并将标准化对象保存到 `locations`。
        locations = np.asarray(positions, dtype=np.int64)
        # 调用 `output.iloc[locations]['sample_index'].to_numpy` 把表格列转换为指定精度的 NumPy 数组，并将结果保存到 `indices`。
        indices = output.iloc[locations]["sample_index"].to_numpy(np.int64)
        # 调用 `np.asarray` 转换数据表示，并将标准化对象保存到 `daily_styles`。
        daily_styles = np.asarray(style_matrix[indices], dtype=np.float64)
        # 调用 `pd.DataFrame(daily_styles).rank(method='average').to_numpy` 根据分数计算截面排序或排序索引，并将结果保存到 `style_ranks`。
        style_ranks = pd.DataFrame(daily_styles).rank(
            method="average"
        ).to_numpy(np.float64)
        # 构造键值映射，并保存到 `record`，供后续按键查询。
        record: dict[str, Any] = {
            "year": year,
            "trade_date": str(trade_date),
            "rows": len(locations),
            "complete_style_rate": float(
                np.asarray(style_valid[indices]).all(axis=1).mean()
            ),
        }
        # 构造当前步骤所需的列表，并保存到 `factors`。
        factors = [
            ("raw", final_raw[locations]),
            ("effective", final_effective[locations]),
        ]
        # 检查 `dual_tower_enabled()`；根据条件结果选择对应处理分支。
        if dual_tower_enabled():
            # 构造当前步骤所需的列表，并保存到 `factors`。
            factors = [
                ("alpha", final_alpha[locations]),
                ("style", final_style[locations]),
                ("total", final_total[locations]),
            ]
            # 按 `float(np.std(final_style[locations], ddof=0)) > 0` 在两个候选值间选择，并保存到 `record['alpha_style_pearson']`。
            record["alpha_style_pearson"] = float(
                np.corrcoef(final_alpha[locations], final_style[locations])[0, 1]
            ) if float(np.std(final_style[locations], ddof=0)) > 0 else np.nan
            # 调用 `float` 统一设备或数据类型，并将结果保存到 `record['dual_additivity_max_abs_error']`。
            record["dual_additivity_max_abs_error"] = float(
                np.max(
                    np.abs(
                        final_total[locations]
                        - final_alpha[locations]
                        - final_style[locations]
                    )
                )
            )
            # 调用 `','.join` 按键合并数据，并将对齐后的结果保存到 `record['inference_total_precision']`。
            record["inference_total_precision"] = ",".join(
                sorted({
                    str(audit["stored_total_precision"])
                    for audit in dual_inference_audits.values()
                })
            )
            # 调用 `max` 汇总当前数据，并将统计量保存到 `record['inference_stored_vs_canonical_max_abs_error']`。
            record["inference_stored_vs_canonical_max_abs_error"] = max(
                float(audit["stored_vs_canonical_max_abs_error"])
                for audit in dual_inference_audits.values()
            )
        # 构造键值映射，并保存到 `target_views`，供后续按键查询。
        target_views: dict[str, np.ndarray] = {}
        # 检查 `dual_tower_enabled()`；根据条件结果选择对应处理分支。
        if dual_tower_enabled():
            # 构造键值映射，并保存到 `target_views`，供后续按键查询。
            target_views = {
                "original": np.asarray(
                    dual_source_targets[indices], dtype=np.float64
                ),
                "residual": np.asarray(
                    dual_residual_targets[indices], dtype=np.float64
                ),
                "fitted": np.asarray(
                    dual_fitted_targets[indices], dtype=np.float64
                ),
            }
        # 遍历 `factors`，并把当前元素绑定到 `(prefix, daily_factor)` 逐项处理。
        for prefix, daily_factor in factors:
            # 调用 `float` 统一设备或数据类型，并将结果保存到 `factor_std`。
            factor_std = float(np.std(daily_factor, ddof=0))
            # 读取 `factor_std` 的当前值，并赋给 `record[f'{prefix}_cross_section_std']`。
            record[f"{prefix}_cross_section_std"] = factor_std
            # 检查 `factor_std <= np.finfo(np.float64).eps`；根据条件结果选择对应处理分支。
            if factor_std <= np.finfo(np.float64).eps:
                # 调用 `np.full` 按目标形状初始化数值容器 `pearson`。
                pearson = np.full(STYLE_COUNT, np.nan, dtype=np.float64)
                # 调用 `np.full` 按目标形状初始化数值容器 `spearman`。
                spearman = np.full(STYLE_COUNT, np.nan, dtype=np.float64)
                # 调用 `np.full` 按目标形状初始化数值容器 `factor_rank`。
                factor_rank = np.full(len(daily_factor), np.nan, dtype=np.float64)
            # 当前置条件不成立时进入备用分支，保证其余合法情况仍被处理。
            else:
                # 调用 `pd.Series(daily_factor).rank(method='average').to_numpy` 根据分数计算截面排序或排序索引，并将结果保存到 `factor_rank`。
                factor_rank = pd.Series(daily_factor).rank(
                    method="average"
                ).to_numpy(np.float64)
                # 读取 `np.corrcoef(np.column_stack([daily_factor, daily_styles]), rowvar=False)[0, 1:]` 的当前值，并赋给 `pearson`。
                pearson = np.corrcoef(
                    np.column_stack([daily_factor, daily_styles]), rowvar=False
                )[0, 1:]
                # 读取 `np.corrcoef(np.column_stack([factor_rank, style_ranks]), rowvar=False)[0, 1:]` 的当前值，并赋给 `spearman`。
                spearman = np.corrcoef(
                    np.column_stack([factor_rank, style_ranks]), rowvar=False
                )[0, 1:]
                # 检查 `not np.isfinite(pearson).all() or not np.isfinite(spearman).all()`；根据条件结果选择对应处理分支。
                if (
                    not np.isfinite(pearson).all()
                    or not np.isfinite(spearman).all()
                ):
                    # 抛出 `FloatingPointError(f'{trade_date} {prefix} 预测风格相关性不可计算')`，阻止不满足数据契约的流程继续执行。
                    raise FloatingPointError(
                        f"{trade_date} {prefix} 预测风格相关性不可计算"
                    )
            # 检查 `dual_tower_enabled()`；根据条件结果选择对应处理分支。
            if dual_tower_enabled():
                # 遍历 `target_views.items()`，并把当前元素绑定到 `(target_name, daily_target)` 逐项处理。
                for target_name, daily_target in target_views.items():
                    # 调用 `np.isfinite` 判断数值是否为有限值以排除 NaN 和无穷大，并将结果保存到 `valid_target`。
                    valid_target = np.isfinite(daily_target)
                    # 检查 `factor_std <= np.finfo(np.float64).eps or int(valid_target.sum()) < 2 or float(np.std(daily_tar…`；根据条件结果选择对应处理分支。
                    if (
                        factor_std <= np.finfo(np.float64).eps
                        or int(valid_target.sum()) < 2
                        or float(np.std(daily_target[valid_target], ddof=0))
                        <= np.finfo(np.float64).eps
                    ):
                        # 读取 `np.nan` 的当前值，并赋给 `pearson_ic`。
                        pearson_ic = np.nan
                        # 读取 `np.nan` 的当前值，并赋给 `rank_ic`。
                        rank_ic = np.nan
                    # 当前置条件不成立时进入备用分支，保证其余合法情况仍被处理。
                    else:
                        # 调用 `float` 统一设备或数据类型，并将结果保存到 `pearson_ic`。
                        pearson_ic = float(np.corrcoef(
                            daily_factor[valid_target],
                            daily_target[valid_target],
                        )[0, 1])
                        # 调用 `float` 统一设备或数据类型，并将结果保存到 `rank_ic`。
                        rank_ic = float(np.corrcoef(
                            pd.Series(daily_factor[valid_target]).rank(
                                method="average"
                            ),
                            pd.Series(daily_target[valid_target]).rank(
                                method="average"
                            ),
                        )[0, 1])
                    # 读取 `pearson_ic` 的当前值，并赋给 `record[f'{prefix}_{target_name}_pearson_ic']`。
                    record[f"{prefix}_{target_name}_pearson_ic"] = pearson_ic
                    # 读取 `rank_ic` 的当前值，并赋给 `record[f'{prefix}_{target_name}_rank_ic']`。
                    record[f"{prefix}_{target_name}_rank_ic"] = rank_ic
            # 用本轮键值结果更新 `record`，补齐后续查询所需字段。
            record.update(
                {
                    f"{prefix}_max_abs_style_pearson": float(np.nanmax(np.abs(pearson))) if np.isfinite(pearson).any() else np.nan,
                    f"{prefix}_mean_abs_style_pearson": float(np.nanmean(np.abs(pearson))) if np.isfinite(pearson).any() else np.nan,
                    f"{prefix}_max_abs_style_spearman": float(np.nanmax(np.abs(spearman))) if np.isfinite(spearman).any() else np.nan,
                    f"{prefix}_mean_abs_style_spearman": float(np.nanmean(np.abs(spearman))) if np.isfinite(spearman).any() else np.nan,
                }
            )
            # 检查 `industry_codes is not None`；根据条件结果选择对应处理分支。
            if industry_codes is not None:
                # 调用 `np.asarray` 转换数据表示，并将标准化对象保存到 `daily_industries`。
                daily_industries = np.asarray(
                    industry_codes[indices], dtype=np.int16
                )
                # 检查 `factor_std > np.finfo(np.float64).eps`；根据条件结果选择对应处理分支。
                if factor_std > np.finfo(np.float64).eps:
                    # 调用 `style_data.joint_barra_r_squared_numpy` 按股票和交易日键对齐来自不同来源的数据，并将结果保存到 `(joint_r_squared, _)`。
                    joint_r_squared, _ = style_data.joint_barra_r_squared_numpy(
                        daily_factor, daily_styles, daily_industries
                    )
                # 当前置条件不成立时进入备用分支，保证其余合法情况仍被处理。
                else:
                    # 读取 `np.nan` 的当前值，并赋给 `joint_r_squared`。
                    joint_r_squared = np.nan
                # 调用 `mean` 计算当前张量或数组的算术均值，并将结果保存到 `industry_means`。
                industry_means = pd.Series(daily_factor).groupby(
                    daily_industries, sort=False
                ).mean()
                # 读取 `joint_r_squared` 的当前值，并赋给 `record[f'{prefix}_joint_barra_r_squared']`。
                record[f"{prefix}_joint_barra_r_squared"] = joint_r_squared
                # 调用 `float` 统一设备或数据类型，并将结果保存到 `record[f'{prefix}_max_abs_industry_mean']`。
                record[f"{prefix}_max_abs_industry_mean"] = float(
                    industry_means.abs().max()
                )
            # 遍历 `enumerate(style_data.STYLE_COLUMNS)`，并把当前元素绑定到 `(position, column)` 逐项处理。
            for position, column in enumerate(style_data.STYLE_COLUMNS):
                # 调用 `float` 统一设备或数据类型，并将结果保存到 `record[f'{prefix}_pearson_{column}']`。
                record[f"{prefix}_pearson_{column}"] = float(
                    pearson[position]
                )
                # 调用 `float` 统一设备或数据类型，并将结果保存到 `record[f'{prefix}_spearman_{column}']`。
                record[f"{prefix}_spearman_{column}"] = float(
                    spearman[position]
                )
        # 把本轮结果追加到 `records`，累积后续需要汇总的记录。
        records.append(record)

    # 调用 `final_raw.astype` 统一设备或数据类型，并将结果保存到 `output['prediction_ensemble_unprojected']`。
    output["prediction_ensemble_unprojected"] = final_raw.astype(np.float32)
    # 检查 `dual_tower_enabled()`；根据条件结果选择对应处理分支。
    if dual_tower_enabled():
        # 调用 `final_alpha.astype` 统一设备或数据类型，并将结果保存到 `output['prediction_ensemble_alpha']`。
        output["prediction_ensemble_alpha"] = final_alpha.astype(np.float32)
        # 调用 `final_style.astype` 统一设备或数据类型，并将结果保存到 `output['prediction_ensemble_style']`。
        output["prediction_ensemble_style"] = final_style.astype(np.float32)
        # 调用 `final_total.astype` 统一设备或数据类型，并将结果保存到 `output['prediction_ensemble_total_return']`。
        output["prediction_ensemble_total_return"] = final_total.astype(
            np.float32
        )
    # 检查 `PROJECT_MODEL_OUTPUT or strict_barra_output_enabled()`；根据条件结果选择对应处理分支。
    if PROJECT_MODEL_OUTPUT or strict_barra_output_enabled():
        # 调用 `final_effective.astype` 统一设备或数据类型，并将结果保存到 `output['prediction_ensemble_projected']`。
        output["prediction_ensemble_projected"] = final_effective.astype(
            np.float32
        )
    # 调用 `astype` 把数组转换为指定数据类型，并将结果保存到 `published`。
    published = (
        final_effective if strict_barra_output_enabled() else final_raw
    ).astype(np.float32)
    # 读取 `published` 的当前值，并赋给 `output['prediction_ensemble_raw']`。
    output["prediction_ensemble_raw"] = published
    # 读取 `published` 的当前值，并赋给 `output['prediction_ensemble']`。
    output["prediction_ensemble"] = published
    # 读取 `year` 的当前值，并赋给 `output['forecast_year']`。
    output["forecast_year"] = year
    # 调用 `factor_role` 解析当前实验输出因子的角色与列名口径，并将结果保存到 `output['factor_role']`。
    output["factor_role"] = factor_role()
    # 调用 `sequence_data.atomic_write_parquet` 先写临时文件再原子替换正式产物，避免留下半成品。
    sequence_data.atomic_write_parquet(
        output,
        prediction_root / f"vivit_predictions.{year}.parquet",
    )
    # 将 `(len(output), pd.DataFrame(records))` 作为当前函数结果返回给调用方。
    return len(output), pd.DataFrame(records)




# 定义 `style_diagnostics_root` 函数，封装这一阶段可独立调用的处理步骤。
def style_diagnostics_root(quality_root: Path) -> Path:
    """返回当前 run 的风格暴露诊断目录。
    
    输入：
        quality_root: Path。
    输出：
        Path。
        """
    # 将 `quality_root / 'style_neutral_training'` 作为当前函数结果返回给调用方。
    return quality_root / "style_neutral_training"


# 定义 `finalize_run` 函数，封装这一阶段可独立调用的处理步骤。
def finalize_run(wall_seconds: float) -> pd.DataFrame:
    """核验年度任务并生成集成预测、诊断、清单和报告。
    
    输入：
        wall_seconds: float。
    输出：
        pd.DataFrame。
        """
    # 调用 `base.run_paths` 解析当前 run 对应的数据与输出路径，并将结果保存到 `(model_root, _, quality_root)`。
    model_root, _, quality_root = base.run_paths(RUN_NAME)
    # 构造当前步骤所需的列表，并保存到 `summaries`。
    summaries: list[dict[str, Any]] = []
    # 遍历 `FORECAST_YEARS`，并把当前元素绑定到 `year` 逐项处理。
    for year in FORECAST_YEARS:
        # 遍历 `SEEDS`，并把当前元素绑定到 `seed` 逐项处理。
        for seed in SEEDS:
            # 计算 `model_root / f'forecast_{year}' / f'seed_{seed}' / 'completed.json'`，并将数值结果保存到 `path`。
            path = model_root / f"forecast_{year}" / f"seed_{seed}" / "completed.json"
            # 检查 `not path.exists()`；根据条件结果选择对应处理分支。
            if not path.exists():
                # 抛出 `FileNotFoundError(path)`，阻止不满足数据契约的流程继续执行。
                raise FileNotFoundError(path)
            # 把本轮结果追加到 `summaries`，累积后续需要汇总的记录。
            summaries.append(json.loads(path.read_text(encoding="utf-8"))["summary"])
    # 调用 `pd.DataFrame(summaries).sort_values` 根据分数计算截面排序或排序索引，并将结果保存到 `summary`。
    summary = pd.DataFrame(summaries).sort_values(
        ["forecast_year", "seed"], kind="stable"
    )
    # 按 `'validation_score_collapsed' in summary.columns` 在两个候选值间选择，并保存到 `validation_score_collapse_tasks`。
    validation_score_collapse_tasks = (
        int(summary["validation_score_collapsed"].fillna(False).sum())
        if "validation_score_collapsed" in summary.columns
        else 0
    )
    # 调用 `sequence_data.atomic_write_parquet` 先写临时文件再原子替换正式产物，避免留下半成品。
    sequence_data.atomic_write_parquet(
        summary, quality_root / "vivit_training_summary.parquet"
    )
    # 检查 `adversarial_barra_enabled()`；根据条件结果选择对应处理分支。
    if adversarial_barra_enabled():
        # 构造当前步骤所需的列表，并保存到 `probe_frames`。
        probe_frames: list[pd.DataFrame] = []
        # 遍历 `FORECAST_YEARS`，并把当前元素绑定到 `year` 逐项处理。
        for year in FORECAST_YEARS:
            # 遍历 `SEEDS`，并把当前元素绑定到 `seed` 逐项处理。
            for seed in SEEDS:
                # 计算 `model_root / f'forecast_{year}' / f'seed_{seed}' / 'adversarial_probe_diagnostics.parquet'`，并将数值结果保存到 `probe_path`。
                probe_path = (
                    model_root / f"forecast_{year}" / f"seed_{seed}"
                    / "adversarial_probe_diagnostics.parquet"
                )
                # 检查 `not probe_path.exists()`；根据条件结果选择对应处理分支。
                if not probe_path.exists():
                    # 抛出 `FileNotFoundError(probe_path)`，阻止不满足数据契约的流程继续执行。
                    raise FileNotFoundError(probe_path)
                # 把本轮结果追加到 `probe_frames`，累积后续需要汇总的记录。
                probe_frames.append(pd.read_parquet(probe_path))
        # 调用 `sequence_data.atomic_write_parquet` 先写临时文件再原子替换正式产物，避免留下半成品。
        sequence_data.atomic_write_parquet(
            pd.concat(probe_frames, ignore_index=True),
            quality_root / "adversarial_probe_diagnostics.parquet",
        )

    # 构造键值映射，并保存到 `factor_counts`，供后续按键查询。
    factor_counts: dict[int, int] = {}
    # 构造当前步骤所需的列表，并保存到 `diagnostics`。
    diagnostics: list[pd.DataFrame] = []
    # 遍历 `FORECAST_YEARS`，并把当前元素绑定到 `year` 逐项处理。
    for year in FORECAST_YEARS:
        # 调用 `build_ensemble` 根据当前配置和输入构造后续流程所需对象，并将结果保存到 `(factor_counts[year], yearly)`。
        factor_counts[year], yearly = build_ensemble(year)
        # 检查 `not yearly.empty`；根据条件结果选择对应处理分支。
        if not yearly.empty:
            # 把本轮结果追加到 `diagnostics`，累积后续需要汇总的记录。
            diagnostics.append(yearly)
    # 检查 `not diagnostics`；根据条件结果选择对应处理分支。
    if not diagnostics:
        # 抛出 `ValueError('没有可写入的样本外预测风格诊断')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("没有可写入的样本外预测风格诊断")
    # 调用 `sequence_data.atomic_write_parquet` 先写临时文件再原子替换正式产物，避免留下半成品。
    sequence_data.atomic_write_parquet(
        pd.concat(diagnostics, ignore_index=True),
        style_diagnostics_root(quality_root)
        / "prediction_style_exposure_diagnostics.parquet",
    )
    # 检查 `dual_tower_enabled()`；根据条件结果选择对应处理分支。
    if dual_tower_enabled():
        # 调用 `sequence_data.atomic_write_parquet` 先写临时文件再原子替换正式产物，避免留下半成品。
        sequence_data.atomic_write_parquet(
            pd.concat(diagnostics, ignore_index=True),
            quality_root / "dual_tower_decomposition_diagnostics.parquet",
        )

    # 调用 `sequence_data.require_aligned_sequence_store` 校验输入、配置或产物是否满足当前阶段的数据契约，并将结果保存到 `sequence`。
    sequence = sequence_data.require_aligned_sequence_store()
    # 调用 `style_data.require_style_neutral_store` 校验输入、配置或产物是否满足当前阶段的数据契约，并将结果保存到 `style_manifest`。
    style_manifest = style_data.require_style_neutral_store()
    # 按 `industry_sidecar_required()` 在两个候选值间选择，并保存到 `industry_manifest`。
    industry_manifest = (
        style_data.require_industry_sidecar()
        if industry_sidecar_required() else None
    )
    # 按 `barra_conditioning_enabled() or dual_tower_enabled()` 在两个候选值间选择，并保存到 `full_target_manifest`。
    full_target_manifest = (
        style_data.require_full_barra_target_sidecar()
        if barra_conditioning_enabled() or dual_tower_enabled() else None
    )
    # 按 `dual_tower_enabled()` 在两个候选值间选择，并保存到 `dual_target_manifest`。
    dual_target_manifest = (
        style_data.require_dual_tower_target_sidecar()
        if dual_tower_enabled() else None
    )
    # 构造键值映射，并保存到 `manifest`，供后续按键查询。
    manifest = {
        "schema_version": (
            7 if dual_tower_enabled()
            else 6 if strict_barra_output_enabled()
            else 5 if barra_conditioning_enabled() else TRAINING_SCHEMA_VERSION
        ),
        "route": (
            "style_neutral_ddp" if EXECUTION_ROUTE == "ddp"
            else "style_neutral_independent_tasks"
        ),
        "execution_route": EXECUTION_ROUTE,
        "world_size": DDP_WORLD_SIZE,
        "physical_device_ids": list(EXECUTION_DEVICE_IDS),
        "tasks_per_gpu": TASKS_PER_GPU if EXECUTION_ROUTE == "a2" else None,
        "run_name": RUN_NAME,
        "factor_role": factor_role(),
        "sequence_signature": sequence["signature"],
        "source_dataset_signature": sequence["payload"]["source_dataset_signature"],
        "source_rolling_signature": sequence["payload"]["source_rolling_signature"],
        "style_store_signature": style_manifest["signature"],
        "industry_sidecar_signature": (
            industry_manifest["signature"]
            if industry_manifest is not None else None
        ),
        "style_columns": list(style_data.STYLE_COLUMNS),
        "industry_used": industry_sidecar_required(),
        "residualize_target": RESIDUALIZE_TARGET,
        "project_model_output": PROJECT_MODEL_OUTPUT,
        "target_policy": target_policy(),
        "output_policy": (
            output_policy()
            if strict_barra_output_enabled() or dual_tower_enabled()
            else PUBLISHED_FACTOR_POLICY
        ),
        "validation_score_collapse_tasks": validation_score_collapse_tasks,
        "training_effective_output_policy": output_policy(),
        "published_factor_column": "prediction_ensemble",
        "published_factor_is_projected": strict_barra_output_enabled(),
        "projected_audit_column": (
            "prediction_ensemble_projected"
            if PROJECT_MODEL_OUTPUT or strict_barra_output_enabled()
            else None
        ),
        "output_projection": (
            {
                "method": "daily_equal_weight_full_barra_projection",
                "design": [
                    "intercept",
                    *style_data.STYLE_COLUMNS,
                    "industry_fixed_effects",
                ],
                "solver": "industry_within_then_torch_fp32_reduced_qr",
                "post_projection": "zscore_ddof0",
                "applied_during": [
                    "training_loss",
                    "validation",
                    "inference_and_publication",
                ],
            }
            if strict_barra_output_enabled()
            else {
                "method": "daily_qr_orthogonal_complement",
                "design": ["intercept", *style_data.STYLE_COLUMNS],
                "post_projection": "zscore_ddof0",
                "applied_during": [
                    "training_loss",
                    "validation",
                    "inference_auxiliary_audit",
                ],
            }
            if PROJECT_MODEL_OUTPUT
            else None
        ),
        "style_correlation_penalty": raw_barra_r2_enabled(),
        "raw_barra_r2_lambda": RAW_BARRA_R2_LAMBDA,
        "adversarial_barra": adversarial_barra_enabled(),
        "barra_is_model_input": (
            barra_conditioning_enabled() or dual_tower_enabled()
        ),
        "barra_participates_in_training_loss": (
            RESIDUALIZE_TARGET
            or PROJECT_MODEL_OUTPUT
            or raw_barra_r2_enabled()
            or adversarial_barra_enabled()
            or barra_conditioning_enabled()
            or dual_tower_enabled()
        ),
        "adversarial_config": (
            {
                "gamma_max": ADVERSARIAL_GAMMA_MAX,
                "warmup_epochs": ADVERSARIAL_WARMUP_EPOCHS,
                "cosine_ramp_epochs": ADVERSARIAL_RAMP_EPOCHS,
                "style_loss_weight": ADVERSARIAL_STYLE_LOSS_WEIGHT,
                "industry_loss_weight": ADVERSARIAL_INDUSTRY_LOSS_WEIGHT,
                "head_hidden_dim": ADVERSARIAL_HEAD_HIDDEN_DIM,
                "head_dropout": ADVERSARIAL_HEAD_DROPOUT,
                "industry_classes": INDUSTRY_COUNT,
                "unknown_industry_policy": "masked_from_ce",
                "inference_model": "backbone_and_return_head_only",
                "probe_file": "adversarial_probe_diagnostics.parquet",
                "probe_sample_cap_per_date": PROBE_MAX_SAMPLES_PER_DATE,
                "probe_date_split": [0.4, 0.1, 0.5],
                "probe_batch_size": PROBE_BATCH_SIZE,
                "probe_learning_rate": PROBE_LEARNING_RATE,
                "probe_weight_decay": PROBE_WEIGHT_DECAY,
                "probe_max_epochs": PROBE_MAX_EPOCHS,
                "probe_patience": PROBE_PATIENCE,
                "probe_ridge_alpha": PROBE_RIDGE_ALPHA,
                "probe_mlp_hidden_dim": 128,
                "probe_mlp_dropout": 0.1,
            }
            if adversarial_barra_enabled() else None
        ),
        "loss_name": LOSS_NAME,
        "raw_barra_r2_design": (
            {
                "dependent_variable": "raw_model_score",
                "continuous": list(style_data.STYLE_COLUMNS),
                "industry_fixed_effects": True,
                "solver": "industry_within_then_torch_fp32_reduced_qr",
                "missing": "style_zero_industry_unknown_no_forward_fill",
            }
            if (
                raw_barra_r2_enabled()
                or barra_conditioning_enabled()
                or dual_tower_enabled()
            )
            else None
        ),
        "inference_requires_barra": (
            dual_tower_enabled()
            or strict_barra_output_enabled()
            or BARRA_CONDITIONING_MODE in {"real", "shuffled"}
        ),
        "loss_parameters": {
            "rank_temperature": RANK_TEMPERATURE,
            "turnover_lambda": TURNOVER_LAMBDA,
            "huber_delta": HUBER_DELTA,
            "smooth_l1_beta": SMOOTH_L1_BETA,
            "hybrid_weight": HYBRID_WEIGHT,
        },
        "validation_metric": VALIDATION_METRIC,
        "validation_target": (
            "alpha_score_vs_full_barra_residual_target"
            if dual_tower_enabled()
            else "strict_projected_score_vs_full_barra_residual_target"
            if strict_barra_output_enabled()
            else
            "raw_score_vs_full_barra_residual_target"
            if barra_conditioning_enabled()
            else "raw_score_vs_source_target"
            if raw_barra_r2_enabled() or adversarial_barra_enabled()
            else "effective_prediction_vs_residual_target"
            if RESIDUALIZE_TARGET
            else "effective_prediction_vs_source_target"
        ),
        "train_batch_mode": TRAIN_BATCH_MODE,
        "workers_per_task": WORKERS_PER_TASK,
        "learning_rate": LEARNING_RATE,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "torch_compile": bool(USE_TORCH_COMPILE),
        "compile_mode": COMPILE_MODE,
        "fused_adam": bool(USE_FUSED_ADAM),
        "cuda_prefetch": bool(USE_CUDA_PREFETCH),
        "years": list(FORECAST_YEARS),
        "seeds": list(SEEDS),
        "completed_tasks": len(summary),
        "factor_counts": factor_counts,
        "split_policy": "exact_vit_train_validation_prediction_indices",
        "wall_seconds": wall_seconds,
    }
    # 检查 `dual_tower_enabled()`；根据条件结果选择对应处理分支。
    if dual_tower_enabled():
        # 读取 `full_target_manifest['signature']` 的当前值，并赋给 `manifest['full_barra_target_sidecar_signature']`。
        manifest["full_barra_target_sidecar_signature"] = (
            full_target_manifest["signature"]
        )
        # 读取 `dual_target_manifest['signature']` 的当前值，并赋给 `manifest['dual_tower_target_sidecar_signature']`。
        manifest["dual_tower_target_sidecar_signature"] = (
            dual_target_manifest["signature"]
        )
        # 构造键值映射，并保存到 `manifest['barra_dual_tower']`，供后续按键查询。
        manifest["barra_dual_tower"] = {
            "mode": BARRA_DUAL_TOWER_MODE,
            "alpha_loss_weight": DUAL_TOWER_ALPHA_LOSS_WEIGHT,
            "style_loss_weight": DUAL_TOWER_STYLE_LOSS_WEIGHT,
            "style_hidden_dim": DUAL_TOWER_STYLE_HIDDEN_DIM,
            "style_dropout": DUAL_TOWER_STYLE_DROPOUT,
            "industry_embedding_dim": BARRA_INDUSTRY_EMBEDDING_DIM,
            "alpha_r2_lambda": RAW_BARRA_R2_LAMBDA,
            "style_output_initialization": "zeros",
            "published_factor": "prediction_ensemble_alpha",
            "component_columns": [
                "prediction_ensemble_alpha",
                "prediction_ensemble_style",
                "prediction_ensemble_total_return",
            ],
            "ensemble_scaling": "center_components_divide_by_alpha_std",
            "seed_total_audit_policy": (
                "stored_total_must_match_float32_or_bfloat16_alpha_plus_style"
            ),
            "ensemble_total_policy": (
                "reconstruct_float32_total_from_stored_alpha_plus_style"
            ),
            "r2_uses_true_unmodified_barra": True,
        }
    # 检查 `barra_conditioning_enabled()`；根据条件结果选择对应处理分支。
    if barra_conditioning_enabled():
        # 读取 `full_target_manifest['signature']` 的当前值，并赋给 `manifest['full_barra_target_sidecar_signature']`。
        manifest["full_barra_target_sidecar_signature"] = (
            full_target_manifest["signature"]
        )
        # 构造键值映射，并保存到 `manifest['barra_conditioning']`，供后续按键查询。
        manifest["barra_conditioning"] = {
            "mode": BARRA_CONDITIONING_MODE,
            "hidden_dim": BARRA_GATE_HIDDEN_DIM,
            "industry_embedding_dim": BARRA_INDUSTRY_EMBEDDING_DIM,
            "dropout": BARRA_GATE_DROPOUT,
            "gate_scale": BARRA_GATE_SCALE,
            "gate_projection_initialization": "zeros",
            "multiplicative_only": True,
            "shuffle_policy": (
                "within_date_joint_permutation_by_stable_sample_key_v1"
                if BARRA_CONDITIONING_MODE == "shuffled" else None
            ),
            "r2_uses_true_unmodified_barra": True,
        }
    # 检查 `strict_barra_output_enabled()`；根据条件结果选择对应处理分支。
    if strict_barra_output_enabled():
        # 构造键值映射，并保存到 `manifest['strict_barra_output']`，供后续按键查询。
        manifest["strict_barra_output"] = {
            "weighting": "equal_weight_W_identity",
            "design": [
                "intercept",
                *style_data.STYLE_COLUMNS,
                "industry_fixed_effects",
            ],
            "solver": "industry_within_then_torch_fp32_reduced_qr",
            "post_projection": "zscore_ddof0",
            "applied_during": ["training_loss", "validation", "inference"],
            "published_factor": "prediction_ensemble",
            "pre_neutral_audit_column": "prediction_ensemble_unprojected",
            "orthogonality_requirement": "H_transpose_W_s_equals_zero",
        }
    # 调用 `sequence_data.atomic_write_json` 先写临时文件再原子替换正式产物，避免留下半成品。
    sequence_data.atomic_write_json(manifest, quality_root / "a2_training_manifest.json")
    # 将 `summary` 作为当前函数结果返回给调用方。
    return summary






# %% [markdown]
# ## 四 GPU 调度和 CLI 入口


# %%


# 定义 `measure_daily_score_dispersion` 函数，封装这一阶段可独立调用的处理步骤。
def measure_daily_score_dispersion(
    predictions: torch.Tensor,
    *,
    date_id: int,
) -> dict[str, torch.Tensor]:
    """检查单日原始分数是否仍有可用于排序的截面离散度。
    
    输入：
        predictions: torch.Tensor，格式 [N]。
        date_id: int。
    输出：
        dict[str, torch.Tensor]。
        """

    # 调用 `predictions.detach().float().reshape` 调整张量维度，并将新视图保存到 `raw`。
    raw = predictions.detach().float().reshape(-1)
    # 检查 `len(raw) < 2`；根据条件结果选择对应处理分支。
    if len(raw) < 2:
        # 抛出 `ValueError('单日预测至少需要两个样本')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("单日预测至少需要两个样本")
    # 检查 `not torch.isfinite(raw).all()`；根据条件结果选择对应处理分支。
    if not torch.isfinite(raw).all():
        # 抛出 `FloatingPointError('模型输出包含非有限值')`，阻止不满足数据契约的流程继续执行。
        raise FloatingPointError("模型输出包含非有限值")

    # 计算 `raw - raw.mean()`，并将数值结果保存到 `centered`。
    centered = raw - raw.mean()
    # 调用 `centered.square().sum` 汇总当前数据，并将统计量保存到 `total_sum_squares`。
    total_sum_squares = centered.square().sum()
    # 调用 `torch.sqrt` 对输入开平方得到范数或标准差相关量，并将结果保存到 `score_std`。
    score_std = torch.sqrt(total_sum_squares / len(raw))
    # 检查 `not torch.isfinite(total_sum_squares) or not torch.isfinite(score_std) or total_sum_squares <= …`；根据条件结果选择对应处理分支。
    if (
        not torch.isfinite(total_sum_squares)
        or not torch.isfinite(score_std)
        or total_sum_squares <= torch.finfo(torch.float32).eps
    ):
        # 抛出 `DegenerateValidationScoreError(date_id=date_id, rows=len(raw), score_std=float(score_std.item()…`，阻止不满足数据契约的流程继续执行。
        raise DegenerateValidationScoreError(
            date_id=date_id,
            rows=len(raw),
            score_std=float(score_std.item()),
            total_sum_squares=float(total_sum_squares.item()),
        )
    # 将 `{'score_std': score_std, 'total_sum_squares': total_sum_squares}` 作为当前函数结果返回给调用方。
    return {
        "score_std": score_std,
        "total_sum_squares": total_sum_squares,
    }

# 定义 `measure_style_correlations` 函数，封装这一阶段可独立调用的处理步骤。
def measure_style_correlations(
    predictions: torch.Tensor,
    style_exposures: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """只读衡量单日预测与十个风格的 Pearson 相关性。
    
    输入：
        predictions: torch.Tensor，格式 [N]。
        style_exposures: torch.Tensor，格式 [N,10]。
    输出：
        dict[str, torch.Tensor]。
        """

    # 调用 `predictions.detach().float().reshape` 调整张量维度，并将新视图保存到 `raw`。
    raw = predictions.detach().float().reshape(-1)
    # 调用 `style_exposures.float` 统一设备或数据类型，并将结果保存到 `styles`。
    styles = style_exposures.float()
    # 检查 `styles.shape != (len(raw), STYLE_COUNT)`；根据条件结果选择对应处理分支。
    if styles.shape != (len(raw), STYLE_COUNT):
        # 抛出 `ValueError(f'十风格矩阵形状异常：{tuple(styles.shape)}')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(f"十风格矩阵形状异常：{tuple(styles.shape)}")
    # 检查 `len(raw) < 3`；根据条件结果选择对应处理分支。
    if len(raw) < 3:
        # 抛出 `ValueError('单日股票数不足以计算风格相关性')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("单日股票数不足以计算风格相关性")
    # 检查 `not torch.isfinite(raw).all() or not torch.isfinite(styles).all()`；根据条件结果选择对应处理分支。
    if not torch.isfinite(raw).all() or not torch.isfinite(styles).all():
        # 抛出 `FloatingPointError('预测或十风格暴露含 NaN/Inf')`，阻止不满足数据契约的流程继续执行。
        raise FloatingPointError("预测或十风格暴露含 NaN/Inf")

    # 计算 `raw - raw.mean()`，并将数值结果保存到 `centered_factor`。
    centered_factor = raw - raw.mean()
    # 计算 `styles - styles.mean(dim=0, keepdim=True)`，并将数值结果保存到 `centered_styles`。
    centered_styles = styles - styles.mean(dim=0, keepdim=True)
    # 计算 `centered_styles.transpose(0, 1) @ centered_factor`，并将数值结果保存到 `numerator`。
    numerator = centered_styles.transpose(0, 1) @ centered_factor
    # 调用 `torch.sqrt` 对输入开平方得到范数或标准差相关量，并将结果保存到 `denominator`。
    denominator = torch.sqrt(
        centered_styles.square().sum(dim=0)
        * centered_factor.square().sum()
    )
    # 计算 `numerator / denominator`，并将数值结果保存到 `correlations`。
    correlations = numerator / denominator
    # 检查 `not torch.isfinite(correlations).all()`；根据条件结果选择对应处理分支。
    if not torch.isfinite(correlations).all():
        # 抛出 `FloatingPointError('模型输出的风格相关性不可计算')`，阻止不满足数据契约的流程继续执行。
        raise FloatingPointError("模型输出的风格相关性不可计算")
    # 将 `{'max_abs_style_pearson': correlations.abs().max(), 'mean_abs_style_pearson': correlations.abs(…` 作为当前函数结果返回给调用方。
    return {
        "max_abs_style_pearson": correlations.abs().max(),
        "mean_abs_style_pearson": correlations.abs().mean(),
    }

# 定义 `summarize_adversarial_predictions` 函数，封装这一阶段可独立调用的处理步骤。
def summarize_adversarial_predictions(
    style_predictions: np.ndarray,
    style_targets: np.ndarray,
    style_valid: np.ndarray,
    industry_logits: np.ndarray,
    industry_codes: np.ndarray,
    industry_class_weights: np.ndarray | None = None,
) -> dict[str, float]:
    """汇总对抗头的风格回归和行业分类诊断。
    
    输入：
        style_predictions: np.ndarray，格式 [N,10]。
        style_targets: np.ndarray。
        style_valid: np.ndarray，格式 [N,10]。
        industry_logits: np.ndarray，格式 [N,32]。
        industry_codes: np.ndarray，格式 [N]。
        industry_class_weights: np.ndarray | None。
    输出：
        dict[str, float]。
        """
    # 调用 `np.asarray` 转换数据表示，并将标准化对象保存到 `mask`。
    mask = np.asarray(style_valid, dtype=bool)
    # 调用 `np.asarray` 转换数据表示，并将标准化对象保存到 `predicted_styles`。
    predicted_styles = np.asarray(style_predictions, dtype=np.float64)
    # 调用 `np.asarray` 转换数据表示，并将标准化对象保存到 `target_styles`。
    target_styles = np.asarray(style_targets, dtype=np.float64)
    # 检查 `predicted_styles.shape != target_styles.shape or mask.shape != target_styles.shape`；根据条件结果选择对应处理分支。
    if predicted_styles.shape != target_styles.shape or mask.shape != target_styles.shape:
        # 抛出 `ValueError('对抗风格诊断形状不一致')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("对抗风格诊断形状不一致")
    # 计算 `(predicted_styles - target_styles) ** 2`，并将数值结果保存到 `squared_errors`。
    squared_errors = (predicted_styles - target_styles) ** 2
    # 构造当前步骤所需的列表，并保存到 `style_r_squared`。
    style_r_squared: list[float] = []
    # 遍历 `range(STYLE_COUNT)`，并把当前元素绑定到 `column` 逐项处理。
    for column in range(STYLE_COUNT):
        # 读取 `mask[:, column]` 的当前值，并赋给 `valid`。
        valid = mask[:, column]
        # 读取 `target_styles[valid, column]` 的当前值，并赋给 `values`。
        values = target_styles[valid, column]
        # 调用 `float` 统一设备或数据类型，并将结果保存到 `denominator`。
        denominator = float(((values - values.mean()) ** 2).sum())
        # 检查 `len(values) < 2 or denominator <= np.finfo(np.float64).eps`；根据条件结果选择对应处理分支。
        if len(values) < 2 or denominator <= np.finfo(np.float64).eps:
            # 抛出 `ValueError('验证期风格标签不足或为常数')`，阻止不满足数据契约的流程继续执行。
            raise ValueError("验证期风格标签不足或为常数")
        # 调用 `float` 统一设备或数据类型，并将结果保存到 `numerator`。
        numerator = float(squared_errors[valid, column].sum())
        # 把本轮结果追加到 `style_r_squared`，累积后续需要汇总的记录。
        style_r_squared.append(1.0 - numerator / denominator)

    # 调用 `np.asarray(industry_codes, dtype=np.int64).reshape` 调整张量维度，并将新视图保存到 `codes`。
    codes = np.asarray(industry_codes, dtype=np.int64).reshape(-1)
    # 调用 `np.asarray` 转换数据表示，并将标准化对象保存到 `logits`。
    logits = np.asarray(industry_logits, dtype=np.float64)
    # 根据 `codes > 0` 计算条件结果，并保存到 `known`。
    known = codes > 0
    # 计算 `codes[known] - 1`，并将数值结果保存到 `targets`。
    targets = codes[known] - 1
    # 读取 `logits[known]` 的当前值，并赋给 `known_logits`。
    known_logits = logits[known]
    # 调用 `known_logits.argmax` 取得最大类别分数对应的行业索引，并将结果保存到 `predicted`。
    predicted = known_logits.argmax(axis=1)
    # 计算 `known_logits - known_logits.max(axis=1, keepdims=True)`，并将数值结果保存到 `shifted`。
    shifted = known_logits - known_logits.max(axis=1, keepdims=True)
    # 计算 `np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(targets)), targets]`，并将数值结果保存到 `negative_log_likelihood`。
    negative_log_likelihood = (
        np.log(np.exp(shifted).sum(axis=1))
        - shifted[np.arange(len(targets)), targets]
    )
    # 调用 `np.asarray` 转换数据表示，并将标准化对象保存到 `class_weights`。
    class_weights = np.asarray(
        industry_class_weights
        if industry_class_weights is not None
        else np.ones(INDUSTRY_COUNT),
        dtype=np.float64,
    )
    # 检查 `class_weights.shape != (INDUSTRY_COUNT,) or not np.isfinite(class_weights).all() or (class_weig…`；根据条件结果选择对应处理分支。
    if (
        class_weights.shape != (INDUSTRY_COUNT,)
        or not np.isfinite(class_weights).all()
        or (class_weights <= 0).any()
    ):
        # 抛出 `ValueError('验证期行业类别权重形状或数值异常')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("验证期行业类别权重形状或数值异常")
    # 读取 `class_weights[targets]` 的当前值，并赋给 `sample_weights`。
    sample_weights = class_weights[targets]
    # 调用 `float` 统一设备或数据类型，并将结果保存到 `normalized_ce`。
    normalized_ce = float(
        np.average(negative_log_likelihood, weights=sample_weights)
        / math.log(INDUSTRY_COUNT)
    )
    # 调用 `np.bincount(targets * INDUSTRY_COUNT + predicted, minlength=IND…` 执行模型推理并生成样本预测分数，并将结果保存到 `confusion`。
    confusion = np.bincount(
        targets * INDUSTRY_COUNT + predicted,
        minlength=INDUSTRY_COUNT**2,
    ).reshape(INDUSTRY_COUNT, INDUSTRY_COUNT)
    # 调用 `confusion.sum` 汇总当前数据，并将统计量保存到 `support`。
    support = confusion.sum(axis=1)
    # 根据 `support > 0` 计算条件结果，并保存到 `present`。
    present = support > 0
    # 调用 `np.divide` 按掩码执行安全除法并处理无效分母，并将结果保存到 `recall`。
    recall = np.divide(
        np.diag(confusion), support,
        out=np.zeros(INDUSTRY_COUNT, dtype=np.float64), where=present,
    )
    # 调用 `confusion.sum` 汇总当前数据，并将统计量保存到 `predicted_count`。
    predicted_count = confusion.sum(axis=0)
    # 调用 `np.divide` 按掩码执行安全除法并处理无效分母，并将结果保存到 `precision`。
    precision = np.divide(
        np.diag(confusion), predicted_count,
        out=np.zeros(INDUSTRY_COUNT, dtype=np.float64),
        where=predicted_count > 0,
    )
    # 调用 `np.divide` 按掩码执行安全除法并处理无效分母，并将结果保存到 `f1`。
    f1 = np.divide(
        2 * precision * recall, precision + recall,
        out=np.zeros(INDUSTRY_COUNT, dtype=np.float64),
        where=(precision + recall) > 0,
    )
    # 将 `{'style_mse': float(squared_errors[mask].mean()), 'style_r_squared_mean': float(np.mean(style_r…` 作为当前函数结果返回给调用方。
    return {
        "style_mse": float(squared_errors[mask].mean()),
        "style_r_squared_mean": float(np.mean(style_r_squared)),
        "style_r_squared_max": float(np.max(style_r_squared)),
        "industry_loss": normalized_ce,
        "industry_accuracy": float((predicted == targets).mean()),
        "industry_balanced_accuracy": float(recall[present].mean()),
        "industry_macro_f1": float(f1[present].mean()),
        "industry_unknown_rate": float((codes == 0).mean()),
    }

# 定义 `deterministic_probe_positions` 函数，封装这一阶段可独立调用的处理步骤。
def deterministic_probe_positions(
    rows: int, *, year: int, seed: int, date_id: int
) -> np.ndarray:
    """按稳定样本键确定性抽取每日 probe 样本。
    
    输入：
        rows: int。
        year: int。
        seed: int。
        date_id: int。
    输出：
        np.ndarray。
        """
    # 检查 `rows <= PROBE_MAX_SAMPLES_PER_DATE`；根据条件结果选择对应处理分支。
    if rows <= PROBE_MAX_SAMPLES_PER_DATE:
        # 将 `np.arange(rows, dtype=np.int64)` 作为当前函数结果返回给调用方。
        return np.arange(rows, dtype=np.int64)
    # 调用 `hashlib.sha256(f'{year}:{seed}:{date_id}:adversarial-probe'.enc…` 对关键配置生成稳定签名以支持复现和一致性检查，并将结果保存到 `digest`。
    digest = hashlib.sha256(
        f"{year}:{seed}:{date_id}:adversarial-probe".encode("utf-8")
    ).digest()
    # 调用 `int.from_bytes` 把哈希摘要字节转换为确定性整数种子，并将结果保存到 `random_seed`。
    random_seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
    # 调用 `np.random.default_rng` 创建独立且可复现的 NumPy 随机数生成器，并将结果保存到 `generator`。
    generator = np.random.default_rng(random_seed)
    # 将 `np.sort(generator.choice(rows, size=PROBE_MAX_SAMPLES_PER_DATE, replace=False).astype(np.int64))` 作为当前函数结果返回给调用方。
    return np.sort(
        generator.choice(
            rows, size=PROBE_MAX_SAMPLES_PER_DATE, replace=False
        ).astype(np.int64)
    )

# 定义 `chronological_probe_split` 函数，封装这一阶段可独立调用的处理步骤。
def chronological_probe_split(date_ids: np.ndarray) -> dict[str, np.ndarray]:
    """按日期顺序切分 probe 训练、验证和测试区间。
    
    输入：
        date_ids: np.ndarray，格式 [N]。
    输出：
        dict[str, np.ndarray]。
        """
    # 调用 `np.unique` 提取唯一值并按需要返回反向索引，并将结果保存到 `dates`。
    dates = np.unique(np.asarray(date_ids, dtype=np.int64))
    # 检查 `len(dates) < 10`；根据条件结果选择对应处理分支。
    if len(dates) < 10:
        # 抛出 `ValueError('冻结 probe 至少需要10个验证交易日')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("冻结 probe 至少需要10个验证交易日")
    # 调用 `max` 汇总当前数据，并将统计量保存到 `train_count`。
    train_count = max(1, int(np.floor(0.4 * len(dates))))
    # 调用 `max` 汇总当前数据，并将统计量保存到 `validation_count`。
    validation_count = max(1, int(np.floor(0.1 * len(dates))))
    # 检查 `train_count + validation_count >= len(dates)`；根据条件结果选择对应处理分支。
    if train_count + validation_count >= len(dates):
        # 抛出 `ValueError('冻结 probe 日期不足以划分测试区间')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("冻结 probe 日期不足以划分测试区间")
    # 构造键值映射，并保存到 `boundaries`，供后续按键查询。
    boundaries = {
        "train": dates[:train_count],
        "validation": dates[train_count:train_count + validation_count],
        "test": dates[train_count + validation_count:],
    }
    # 将 `{name: np.flatnonzero(np.isin(date_ids, selected)).astype(np.int64) for name, selected in bound…` 作为当前函数结果返回给调用方。
    return {
        name: np.flatnonzero(np.isin(date_ids, selected)).astype(np.int64)
        for name, selected in boundaries.items()
    }

# 使用 `torch.no_grad()` 包装 `extract_probe_dataset`，并定义其可复用的处理逻辑。
@torch.no_grad()
def extract_probe_dataset(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    year: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """冻结主模型并提取 CLS、风格和行业 probe 数据。
    
    输入：
        model: torch.nn.Module。
        loader: DataLoader。
        device: torch.device。
        year: int。
        seed: int。
    输出：
        dict[str, np.ndarray]。
        """
    # 调用 `model.eval` 将模型切换到评估模式，关闭训练期随机行为。
    model.eval()
    # 构造键值映射，并保存到 `parts`，供后续按键查询。
    parts: dict[str, list[np.ndarray]] = {
        "representations": [],
        "styles": [],
        "style_valid": [],
        "industry_codes": [],
        "date_ids": [],
    }
    # 遍历 `StyleCudaPrefetcher(loader, device, bool(USE_CUDA_PREFETCH))`，并把当前元素绑定到 `item` 逐项处理。
    for item in StyleCudaPrefetcher(
        loader, device, bool(USE_CUDA_PREFETCH)
    ):
        # 读取 `item` 的当前值，并赋给 `(batch, videos, frame_mask, _targets, _original, _residual, _fitted, st…`。
        (
            batch, videos, frame_mask, _targets, _original, _residual,
            _fitted, styles, style_valid, industry_codes,
        ) = item
        # 调用 `int` 统一设备或数据类型，并将结果保存到 `date_id`。
        date_id = int(batch["date_ids"][0])
        # 检查 `int(batch['date_ids'].min()) != int(batch['date_ids'].max())`；根据条件结果选择对应处理分支。
        if int(batch["date_ids"].min()) != int(batch["date_ids"].max()):
            # 抛出 `AssertionError('probe batch 混入多个交易日')`，阻止不满足数据契约的流程继续执行。
            raise AssertionError("probe batch 混入多个交易日")
        # 在 `base.autocast_context(device)` 管理的上下文中执行本段逻辑，并保证资源按约定释放。
        with base.autocast_context(device):
            # 调用 `forward_training_model` 对当前 batch 执行模型前向传播，并将结果保存到 `(_, _, _, representation, _, _, _)`。
            (
                _, _, _, representation, _, _, _
            ) = forward_training_model(
                model,
                videos,
                frame_mask,
                gamma=0.0,
                styles=styles,
                style_valid=style_valid,
                industry_codes=industry_codes,
                sample_indices=batch["sample_indices"],
            )
        # 检查 `representation is None`；根据条件结果选择对应处理分支。
        if representation is None:
            # 抛出 `ValueError('No.3 probe 无法取得 CLS 表征')`，阻止不满足数据契约的流程继续执行。
            raise ValueError("No.3 probe 无法取得 CLS 表征")
        # 调用 `deterministic_probe_positions` 构建或运行可解释性探针以测量表征中的风险信息，并将结果保存到 `positions`。
        positions = deterministic_probe_positions(
            len(videos), year=year, seed=seed, date_id=date_id
        )
        # 调用 `torch.from_numpy(positions).to` 统一设备或数据类型，并将结果保存到 `index`。
        index = torch.from_numpy(positions).to(device)
        # 把本轮结果追加到 `parts['representations']`，累积后续需要汇总的记录。
        parts["representations"].append(
            representation.float().index_select(0, index).cpu().numpy()
        )
        # 把本轮结果追加到 `parts['styles']`，累积后续需要汇总的记录。
        parts["styles"].append(
            styles.float().index_select(0, index).cpu().numpy()
        )
        # 把本轮结果追加到 `parts['style_valid']`，累积后续需要汇总的记录。
        parts["style_valid"].append(
            style_valid.index_select(0, index).cpu().numpy()
        )
        # 把本轮结果追加到 `parts['industry_codes']`，累积后续需要汇总的记录。
        parts["industry_codes"].append(
            industry_codes.index_select(0, index).cpu().numpy()
        )
        # 把本轮结果追加到 `parts['date_ids']`，累积后续需要汇总的记录。
        parts["date_ids"].append(
            np.full(len(positions), date_id, dtype=np.int64)
        )
    # 检查 `not parts['representations']`；根据条件结果选择对应处理分支。
    if not parts["representations"]:
        # 抛出 `ValueError('冻结 probe 没有抽取到验证样本')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("冻结 probe 没有抽取到验证样本")
    # 将 `{name: np.concatenate(values, axis=0) for name, values in parts.items()}` 作为当前函数结果返回给调用方。
    return {
        name: np.concatenate(values, axis=0)
        for name, values in parts.items()
    }

# 定义内部辅助函数 `_style_probe_metrics`，封装仅供本模块复用的计算步骤。
def _style_probe_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    valid: np.ndarray,
) -> tuple[list[float], float]:
    """处理连续风格数据、有效掩码或相应诊断。
    
    输入：
        predictions: np.ndarray，格式 [N]。
        targets: np.ndarray，格式 [N]。
        valid: np.ndarray。
    输出：
        tuple[list[float], float]。
        """
    # 计算 `(np.asarray(predictions, dtype=np.float64) - np.asarray(targets, dtype=np.float64)) ** 2`，并将数值结果保存到 `squared`。
    squared = (
        np.asarray(predictions, dtype=np.float64)
        - np.asarray(targets, dtype=np.float64)
    ) ** 2
    # 调用 `np.asarray` 转换数据表示，并将标准化对象保存到 `mask`。
    mask = np.asarray(valid, dtype=bool)
    # 构造当前步骤所需的列表，并保存到 `r_squared`。
    r_squared: list[float] = []
    # 遍历 `range(STYLE_COUNT)`，并把当前元素绑定到 `column` 逐项处理。
    for column in range(STYLE_COUNT):
        # 读取 `mask[:, column]` 的当前值，并赋给 `selected`。
        selected = mask[:, column]
        # 调用 `targets[selected, column].astype` 统一设备或数据类型，并将结果保存到 `values`。
        values = targets[selected, column].astype(np.float64)
        # 调用 `float` 统一设备或数据类型，并将结果保存到 `denominator`。
        denominator = float(((values - values.mean()) ** 2).sum())
        # 检查 `len(values) < 2 or denominator <= np.finfo(np.float64).eps`；根据条件结果选择对应处理分支。
        if len(values) < 2 or denominator <= np.finfo(np.float64).eps:
            # 抛出 `ValueError('probe 测试集风格标签不足或为常数')`，阻止不满足数据契约的流程继续执行。
            raise ValueError("probe 测试集风格标签不足或为常数")
        # 把本轮结果追加到 `r_squared`，累积后续需要汇总的记录。
        r_squared.append(
            1.0 - float(squared[selected, column].sum()) / denominator
        )
    # 将 `(r_squared, float(squared[mask].mean()))` 作为当前函数结果返回给调用方。
    return r_squared, float(squared[mask].mean())

# 定义内部辅助函数 `_industry_probe_metrics`，封装仅供本模块复用的计算步骤。
def _industry_probe_metrics(
    logits: np.ndarray, industry_codes: np.ndarray
) -> dict[str, float]:
    """处理行业代码或行业诊断，并保留 UNKNOWN 口径。
    
    输入：
        logits: np.ndarray。
        industry_codes: np.ndarray，格式 [N]。
    输出：
        dict[str, float]。
        """
    # 调用 `np.asarray(industry_codes, dtype=np.int64).reshape` 调整张量维度，并将新视图保存到 `codes`。
    codes = np.asarray(industry_codes, dtype=np.int64).reshape(-1)
    # 根据 `codes > 0` 计算条件结果，并保存到 `known`。
    known = codes > 0
    # 计算 `codes[known] - 1`，并将数值结果保存到 `targets`。
    targets = codes[known] - 1
    # 调用 `np.asarray(logits)[known].argmax` 取得最大类别分数对应的行业索引，并将结果保存到 `predicted`。
    predicted = np.asarray(logits)[known].argmax(axis=1)
    # 调用 `np.bincount(targets * INDUSTRY_COUNT + predicted, minlength=IND…` 执行模型推理并生成样本预测分数，并将结果保存到 `confusion`。
    confusion = np.bincount(
        targets * INDUSTRY_COUNT + predicted,
        minlength=INDUSTRY_COUNT**2,
    ).reshape(INDUSTRY_COUNT, INDUSTRY_COUNT)
    # 调用 `confusion.sum` 汇总当前数据，并将统计量保存到 `support`。
    support = confusion.sum(axis=1)
    # 根据 `support > 0` 计算条件结果，并保存到 `present`。
    present = support > 0
    # 调用 `np.diag(confusion).astype` 统一设备或数据类型，并将结果保存到 `true_positive`。
    true_positive = np.diag(confusion).astype(np.float64)
    # 调用 `np.divide` 按掩码执行安全除法并处理无效分母，并将结果保存到 `recall`。
    recall = np.divide(
        true_positive, support,
        out=np.zeros(INDUSTRY_COUNT), where=present,
    )
    # 调用 `confusion.sum` 汇总当前数据，并将统计量保存到 `predicted_count`。
    predicted_count = confusion.sum(axis=0)
    # 调用 `np.divide` 按掩码执行安全除法并处理无效分母，并将结果保存到 `precision`。
    precision = np.divide(
        true_positive, predicted_count,
        out=np.zeros(INDUSTRY_COUNT), where=predicted_count > 0,
    )
    # 调用 `np.divide` 按掩码执行安全除法并处理无效分母，并将结果保存到 `f1`。
    f1 = np.divide(
        2 * precision * recall, precision + recall,
        out=np.zeros(INDUSTRY_COUNT), where=(precision + recall) > 0,
    )
    # 将 `{'accuracy': float((predicted == targets).mean()), 'balanced_accuracy': float(recall[present].m…` 作为当前函数结果返回给调用方。
    return {
        "accuracy": float((predicted == targets).mean()),
        "balanced_accuracy": float(recall[present].mean()),
        "macro_f1": float(f1[present].mean()),
        "unknown_rate": float((codes == 0).mean()),
    }

# 定义 `fit_linear_style_probe` 函数，封装这一阶段可独立调用的处理步骤。
def fit_linear_style_probe(
    representations: np.ndarray,
    styles: np.ndarray,
    style_valid: np.ndarray,
    split: dict[str, np.ndarray],
    device: torch.device,
) -> np.ndarray:
    """拟合线性岭回归 probe，衡量 CLS 中可恢复的风格信息。
    
    输入：
        representations: np.ndarray。
        styles: np.ndarray，格式 [N,10]。
        style_valid: np.ndarray，格式 [N,10]。
        split: dict[str, np.ndarray]。
        device: torch.device。
    输出：
        np.ndarray。
        """
    # 读取 `split['train']` 的当前值，并赋给 `train`。
    train = split["train"]
    # 读取 `split['test']` 的当前值，并赋给 `test`。
    test = split["test"]
    # 调用 `torch.from_numpy(representations[train]).float().to` 统一设备或数据类型，并将结果保存到 `x_train`。
    x_train = torch.from_numpy(representations[train]).float().to(device)
    # 调用 `torch.from_numpy(representations[test]).float().to` 统一设备或数据类型，并将结果保存到 `x_test`。
    x_test = torch.from_numpy(representations[test]).float().to(device)
    # 调用 `torch.cat` 拼接各批结果，并将合并值保存到 `x_train`。
    x_train = torch.cat(
        [x_train, torch.ones((len(x_train), 1), device=device)], dim=1
    )
    # 调用 `torch.cat` 拼接各批结果，并将合并值保存到 `x_test`。
    x_test = torch.cat(
        [x_test, torch.ones((len(x_test), 1), device=device)], dim=1
    )
    # 调用 `torch.eye` 构造单位矩阵作为岭回归正则项，并将结果保存到 `identity`。
    identity = torch.eye(x_train.shape[1], device=device)
    # 设置 `identity[-1, -1]` 的固定配置值，供后续流程统一使用。
    identity[-1, -1] = 0.0
    # 构造当前步骤所需的列表，并保存到 `predictions`。
    predictions: list[np.ndarray] = []
    # 遍历 `range(STYLE_COUNT)`，并把当前元素绑定到 `column` 逐项处理。
    for column in range(STYLE_COUNT):
        # 调用 `torch.from_numpy(style_valid[train, column]).bool().to` 统一设备或数据类型，并将结果保存到 `valid`。
        valid = torch.from_numpy(style_valid[train, column]).bool().to(device)
        # 调用 `torch.from_numpy(styles[train, column]).float().to` 统一设备或数据类型，并将结果保存到 `target`。
        target = torch.from_numpy(styles[train, column]).float().to(device)
        # 读取 `x_train[valid]` 的当前值，并赋给 `design`。
        design = x_train[valid]
        # 调用 `torch.linalg.solve` 求解带岭惩罚的线性方程组，并将结果保存到 `coefficients`。
        coefficients = torch.linalg.solve(
            design.T @ design + PROBE_RIDGE_ALPHA * identity,
            design.T @ target[valid],
        )
        # 把本轮结果追加到 `predictions`，累积后续需要汇总的记录。
        predictions.append((x_test @ coefficients).cpu().numpy())
    # 将 `np.column_stack(predictions).astype(np.float32)` 作为当前函数结果返回给调用方。
    return np.column_stack(predictions).astype(np.float32)

# 定义 `_ProbeNetwork` 类，集中封装该阶段的数据结构与相关行为。
class _ProbeNetwork(torch.nn.Module):
    # 定义初始化方法，接收配置并建立实例后续计算所需的内部状态。
    def __init__(self, output_dim: int, nonlinear: bool) -> None:
        """初始化 _ProbeNetwork 所需的数据引用、配置和运行状态。
        
        输入：
            output_dim: int。
            nonlinear: bool。
        输出：
            与签名返回类型一致。
            """
        # 初始化父类状态，确保继承链中的基础组件正确注册。
        super().__init__()
        # 检查 `nonlinear`；根据条件结果选择对应处理分支。
        if nonlinear:
            # 调用 `torch.nn.Sequential` 按顺序组合探针网络各层，并将结果保存到 `self.layers`。
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(base.MODEL_CONFIG.embedding_dim, 128),
                torch.nn.GELU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(128, output_dim),
            )
        # 当前置条件不成立时进入备用分支，保证其余合法情况仍被处理。
        else:
            # 调用 `torch.nn.Linear` 创建探针使用的线性映射层，并将结果保存到 `self.layers`。
            self.layers = torch.nn.Linear(
                base.MODEL_CONFIG.embedding_dim, output_dim
            )

    # 定义 `forward` 函数，封装这一阶段可独立调用的处理步骤。
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """执行 _ProbeNetwork 的前向计算并返回约定张量。
        
        输入：
            values: torch.Tensor。
        输出：
            torch.Tensor。
            """
        # 将 `self.layers(values)` 作为当前函数结果返回给调用方。
        return self.layers(values)

# 定义 `fit_torch_probe` 函数，封装这一阶段可独立调用的处理步骤。
def fit_torch_probe(
    representations: np.ndarray,
    targets: np.ndarray,
    valid: np.ndarray | None,
    split: dict[str, np.ndarray],
    device: torch.device,
    *,
    output_dim: int,
    nonlinear: bool,
    classification: bool,
    seed: int,
) -> np.ndarray:
    """训练线性或 MLP probe，并在独立日期区间评估。
    
    输入：
        representations: np.ndarray。
        targets: np.ndarray，格式 [N]。
        valid: np.ndarray | None。
        split: dict[str, np.ndarray]。
        device: torch.device。
        output_dim: int。
        nonlinear: bool。
        classification: bool。
        seed: int。
    输出：
        np.ndarray。
        """
    # 调用 `torch.manual_seed` 设置 PyTorch 随机种子以保证结果可复现。
    torch.manual_seed(seed)
    # 调用 `_ProbeNetwork(output_dim, nonlinear).to` 统一设备或数据类型，并将结果保存到 `network`。
    network = _ProbeNetwork(output_dim, nonlinear).to(device)
    # 调用 `torch.optim.AdamW` 创建带解耦权重衰减的 AdamW 优化器，并将结果保存到 `optimizer`。
    optimizer = torch.optim.AdamW(
        network.parameters(), lr=PROBE_LEARNING_RATE,
        weight_decay=PROBE_WEIGHT_DECAY,
    )
    # 设置 `class_weights` 的固定配置值，供后续流程统一使用。
    class_weights = None
    # 检查 `classification`；根据条件结果选择对应处理分支。
    if classification:
        # 调用 `targets[split['train']].astype` 统一设备或数据类型，并将结果保存到 `train_codes`。
        train_codes = targets[split["train"]].astype(np.int64)
        # 调用 `np.bincount(train_codes[train_codes > 0] - 1, minlength=INDUSTR…` 执行模型训练并记录本轮优化指标，并将结果保存到 `counts`。
        counts = np.bincount(
            train_codes[train_codes > 0] - 1, minlength=INDUSTRY_COUNT
        ).astype(np.float64)
        # 检查 `(counts <= 0).any()`；根据条件结果选择对应处理分支。
        if (counts <= 0).any():
            # 抛出 `ValueError('probe 训练集未覆盖全部32行业')`，阻止不满足数据契约的流程继续执行。
            raise ValueError("probe 训练集未覆盖全部32行业")
        # 调用 `np.reciprocal` 计算样本数平方根的倒数作为加权系数，并将结果保存到 `weights`。
        weights = np.reciprocal(np.sqrt(counts))
        # 用 `weights.mean()` 对 `weights` 做 `Div` 累积更新。
        weights /= weights.mean()
        # 调用 `torch.from_numpy(weights.astype(np.float32)).to` 统一设备或数据类型，并将结果保存到 `class_weights`。
        class_weights = torch.from_numpy(weights.astype(np.float32)).to(device)

    # 定义 `loss_for` 函数，封装这一阶段可独立调用的处理步骤。
    def loss_for(indices: np.ndarray, *, training: bool) -> torch.Tensor:
        """按 probe 类型返回连续风格回归或行业分类损失。
        
        输入：
            indices: np.ndarray。
            training: bool。
        输出：
            torch.Tensor。
            """
        # 检查 `not len(indices)`；根据条件结果选择对应处理分支。
        if not len(indices):
            # 抛出 `ValueError('probe split 为空')`，阻止不满足数据契约的流程继续执行。
            raise ValueError("probe split 为空")
        # 调用 `torch.zeros` 按目标形状初始化数值容器 `total`。
        total = torch.zeros((), dtype=torch.float32, device=device)
        # 设置 `rows` 的固定配置值，供后续流程统一使用。
        rows = 0
        # 复制 `indices.copy`，避免后续更新污染原对象，并保存到 `order`。
        order = indices.copy()
        # 检查 `training`；根据条件结果选择对应处理分支。
        if training:
            # 调用 `np.random.default_rng` 创建独立且可复现的 NumPy 随机数生成器，并将结果保存到 `generator`。
            generator = np.random.default_rng(seed + epoch)
            # 调用 `generator.shuffle` 原地打乱训练索引顺序。
            generator.shuffle(order)
        # 遍历 `range(0, len(order), PROBE_BATCH_SIZE)`，并把当前元素绑定到 `start` 逐项处理。
        for start in range(0, len(order), PROBE_BATCH_SIZE):
            # 读取 `order[start:start + PROBE_BATCH_SIZE]` 的当前值，并赋给 `selected`。
            selected = order[start:start + PROBE_BATCH_SIZE]
            # 调用 `torch.from_numpy(representations[selected]).float().to` 统一设备或数据类型，并将结果保存到 `x`。
            x = torch.from_numpy(representations[selected]).float().to(device)
            # 调用 `network(x).float` 统一设备或数据类型，并将结果保存到 `output`。
            output = network(x).float()
            # 检查 `classification`；根据条件结果选择对应处理分支。
            if classification:
                # 调用 `torch.from_numpy(targets[selected]).long().to` 统一设备或数据类型，并将结果保存到 `codes`。
                codes = torch.from_numpy(targets[selected]).long().to(device)
                # 根据 `codes > 0` 计算条件结果，并保存到 `known`。
                known = codes > 0
                # 检查 `not known.any()`；根据条件结果选择对应处理分支。
                if not known.any():
                    # 跳过当前元素的剩余步骤，继续处理下一项。
                    continue
                # 计算 `torch.nn.functional.cross_entropy(output[known], codes[known] - 1, weight=class_weights) / math…`，并将数值结果保存到 `batch_loss`。
                batch_loss = torch.nn.functional.cross_entropy(
                    output[known], codes[known] - 1, weight=class_weights
                ) / math.log(INDUSTRY_COUNT)
                # 调用 `int` 统一设备或数据类型，并将结果保存到 `count`。
                count = int(known.sum())
            # 当前置条件不成立时进入备用分支，保证其余合法情况仍被处理。
            else:
                # 调用 `torch.from_numpy(targets[selected]).float().to` 统一设备或数据类型，并将结果保存到 `y`。
                y = torch.from_numpy(targets[selected]).float().to(device)
                # 调用 `torch.from_numpy(valid[selected]).bool().to` 统一设备或数据类型，并将结果保存到 `mask`。
                mask = torch.from_numpy(valid[selected]).bool().to(device)
                # 调用 `(output - y).square()[mask].mean` 汇总当前数据，并将统计量保存到 `batch_loss`。
                batch_loss = (output - y).square()[mask].mean()
                # 调用 `int` 统一设备或数据类型，并将结果保存到 `count`。
                count = int(mask.sum())
            # 检查 `training`；根据条件结果选择对应处理分支。
            if training:
                # 调用 `optimizer.zero_grad` 清空上一轮梯度，避免跨批次意外累积。
                optimizer.zero_grad(set_to_none=True)
                # 调用 `batch_loss.backward` 反向传播当前损失，计算模型参数梯度。
                batch_loss.backward()
                # 调用 `optimizer.step` 推进一次优化器或调度器状态。
                optimizer.step()
            # 用 `batch_loss.detach() * count` 对 `total` 做 `Add` 累积更新。
            total += batch_loss.detach() * count
            # 用 `count` 对 `rows` 做 `Add` 累积更新。
            rows += count
        # 检查 `not rows`；根据条件结果选择对应处理分支。
        if not rows:
            # 抛出 `ValueError('probe split 没有有效监督')`，阻止不满足数据契约的流程继续执行。
            raise ValueError("probe split 没有有效监督")
        # 将 `total / rows` 作为当前函数结果返回给调用方。
        return total / rows

    # 读取 `np.inf` 的当前值，并赋给 `best_loss`。
    best_loss = np.inf
    # 设置 `best_state` 的固定配置值，供后续流程统一使用。
    best_state: dict[str, torch.Tensor] | None = None
    # 设置 `epochs_without_improvement` 的固定配置值，供后续流程统一使用。
    epochs_without_improvement = 0
    # 遍历 `range(1, PROBE_MAX_EPOCHS + 1)`，并把当前元素绑定到 `epoch` 逐项处理。
    for epoch in range(1, PROBE_MAX_EPOCHS + 1):
        # 调用 `network.train` 将模型切换到训练模式，启用训练期行为。
        network.train()
        # 调用 `loss_for` 按当前实验口径计算可反向传播的训练损失。
        loss_for(split["train"], training=True)
        # 调用 `network.eval` 将模型切换到评估模式，关闭训练期随机行为。
        network.eval()
        # 在 `torch.no_grad()` 管理的上下文中执行本段逻辑，并保证资源按约定释放。
        with torch.no_grad():
            # 调用 `float` 统一设备或数据类型，并将结果保存到 `validation_loss`。
            validation_loss = float(
                loss_for(split["validation"], training=False)
            )
        # 检查 `validation_loss < best_loss - 1e-07`；根据条件结果选择对应处理分支。
        if validation_loss < best_loss - 1e-7:
            # 读取 `validation_loss` 的当前值，并赋给 `best_loss`。
            best_loss = validation_loss
            # 构造键值映射，并保存到 `best_state`，供后续按键查询。
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in network.state_dict().items()
            }
            # 设置 `epochs_without_improvement` 的固定配置值，供后续流程统一使用。
            epochs_without_improvement = 0
        # 当前置条件不成立时进入备用分支，保证其余合法情况仍被处理。
        else:
            # 用 `1` 对 `epochs_without_improvement` 做 `Add` 累积更新。
            epochs_without_improvement += 1
        # 检查 `epochs_without_improvement >= PROBE_PATIENCE`；根据条件结果选择对应处理分支。
        if epochs_without_improvement >= PROBE_PATIENCE:
            # 终止当前循环，避免继续处理已经不满足条件的元素。
            break
    # 检查 `best_state is None`；根据条件结果选择对应处理分支。
    if best_state is None:
        # 抛出 `RuntimeError('probe 未生成最佳参数')`，阻止不满足数据契约的流程继续执行。
        raise RuntimeError("probe 未生成最佳参数")
    # 调用 `network.load_state_dict` 读取并解析后续计算所需的数据或运行状态。
    network.load_state_dict(best_state)
    # 调用 `network.eval` 将模型切换到评估模式，关闭训练期随机行为。
    network.eval()
    # 构造当前步骤所需的列表，并保存到 `outputs`。
    outputs: list[np.ndarray] = []
    # 读取 `split['test']` 的当前值，并赋给 `test`。
    test = split["test"]
    # 在 `torch.no_grad()` 管理的上下文中执行本段逻辑，并保证资源按约定释放。
    with torch.no_grad():
        # 遍历 `range(0, len(test), PROBE_BATCH_SIZE)`，并把当前元素绑定到 `start` 逐项处理。
        for start in range(0, len(test), PROBE_BATCH_SIZE):
            # 读取 `test[start:start + PROBE_BATCH_SIZE]` 的当前值，并赋给 `selected`。
            selected = test[start:start + PROBE_BATCH_SIZE]
            # 调用 `torch.from_numpy(representations[selected]).float().to` 统一设备或数据类型，并将结果保存到 `x`。
            x = torch.from_numpy(representations[selected]).float().to(device)
            # 把本轮结果追加到 `outputs`，累积后续需要汇总的记录。
            outputs.append(network(x).float().cpu().numpy())
    # 将 `np.concatenate(outputs, axis=0)` 作为当前函数结果返回给调用方。
    return np.concatenate(outputs, axis=0)

# 定义 `run_adversarial_probes` 函数，封装这一阶段可独立调用的处理步骤。
def run_adversarial_probes(
    model: torch.nn.Module,
    validation_loader: DataLoader,
    device: torch.device,
    *,
    year: int,
    seed: int,
) -> pd.DataFrame:
    """对最佳 checkpoint 运行冻结 probe 并保存年度诊断。
    
    输入：
        model: torch.nn.Module。
        validation_loader: DataLoader。
        device: torch.device。
        year: int。
        seed: int。
    输出：
        pd.DataFrame。
        """
    # 调用 `extract_probe_dataset` 构建或运行可解释性探针以测量表征中的风险信息，并将结果保存到 `dataset`。
    dataset = extract_probe_dataset(
        model, validation_loader, device, year=year, seed=seed
    )
    # 调用 `chronological_probe_split` 构建或运行可解释性探针以测量表征中的风险信息，并将结果保存到 `split`。
    split = chronological_probe_split(dataset["date_ids"])
    # 读取 `split['test']` 的当前值，并赋给 `test`。
    test = split["test"]
    # 构造键值映射，并保存到 `metadata`，供后续按键查询。
    metadata = {
        "forecast_year": year,
        "seed": seed,
        "sample_count": len(dataset["date_ids"]),
        "train_rows": len(split["train"]),
        "validation_rows": len(split["validation"]),
        "test_rows": len(test),
        "train_date_min": int(dataset["date_ids"][split["train"]].min()),
        "train_date_max": int(dataset["date_ids"][split["train"]].max()),
        "validation_date_min": int(
            dataset["date_ids"][split["validation"]].min()
        ),
        "validation_date_max": int(
            dataset["date_ids"][split["validation"]].max()
        ),
        "test_date_min": int(dataset["date_ids"][test].min()),
        "test_date_max": int(dataset["date_ids"][test].max()),
        "unknown_rate": float((dataset["industry_codes"] == 0).mean()),
    }
    # 构造当前步骤所需的列表，并保存到 `records`。
    records: list[dict[str, Any]] = []
    # 调用 `fit_linear_style_probe` 执行模型训练并记录本轮优化指标，并将结果保存到 `linear_style`。
    linear_style = fit_linear_style_probe(
        dataset["representations"], dataset["styles"],
        dataset["style_valid"], split, device,
    )
    # 调用 `fit_torch_probe` 执行模型训练并记录本轮优化指标，并将结果保存到 `mlp_style`。
    mlp_style = fit_torch_probe(
        dataset["representations"], dataset["styles"],
        dataset["style_valid"], split, device,
        output_dim=STYLE_COUNT, nonlinear=True, classification=False,
        seed=seed + 1000,
    )
    # 遍历 `(('linear_ridge', linear_style), ('mlp', mlp_style))`，并把当前元素绑定到 `(probe_name, predictions)` 逐项处理。
    for probe_name, predictions in (
        ("linear_ridge", linear_style), ("mlp", mlp_style)
    ):
        # 调用 `_style_probe_metrics` 构建或运行可解释性探针以测量表征中的风险信息，并将结果保存到 `(r_squared, mse)`。
        r_squared, mse = _style_probe_metrics(
            predictions, dataset["styles"][test],
            dataset["style_valid"][test],
        )
        # 遍历 `zip(style_data.STYLE_COLUMNS, r_squared)`，并把当前元素绑定到 `(column, value)` 逐项处理。
        for column, value in zip(style_data.STYLE_COLUMNS, r_squared):
            # 把本轮结果追加到 `records`，累积后续需要汇总的记录。
            records.append({
                **metadata, "probe": probe_name, "target_kind": "style",
                "target": column, "r_squared": value, "mse": np.nan,
            })
        # 把本轮结果追加到 `records`，累积后续需要汇总的记录。
        records.append({
            **metadata, "probe": probe_name, "target_kind": "style_summary",
            "target": "all_styles", "r_squared": float(np.mean(r_squared)),
            "max_r_squared": float(np.max(r_squared)), "mse": mse,
        })

    # 遍历 `(('linear_softmax', False, 2000), ('mlp', True, 3000))`，并把当前元素绑定到 `(probe_name, nonlinear, offset)` 逐项处理。
    for probe_name, nonlinear, offset in (
        ("linear_softmax", False, 2000), ("mlp", True, 3000)
    ):
        # 调用 `fit_torch_probe` 执行模型训练并记录本轮优化指标，并将结果保存到 `logits`。
        logits = fit_torch_probe(
            dataset["representations"], dataset["industry_codes"], None,
            split, device, output_dim=INDUSTRY_COUNT, nonlinear=nonlinear,
            classification=True, seed=seed + offset,
        )
        # 调用 `_industry_probe_metrics` 构建或运行可解释性探针以测量表征中的风险信息，并将结果保存到 `metrics`。
        metrics = _industry_probe_metrics(
            logits, dataset["industry_codes"][test]
        )
        # 把本轮结果追加到 `records`，累积后续需要汇总的记录。
        records.append({
            **metadata, "probe": probe_name, "target_kind": "industry",
            "target": "known_industries", **metrics,
        })
    # 将 `pd.DataFrame(records)` 作为当前函数结果返回给调用方。
    return pd.DataFrame(records)

# 定义 `validate_outputs` 函数，封装这一阶段可独立调用的处理步骤。
def validate_outputs() -> pd.DataFrame:
    """检查 checkpoint、预测、诊断和 manifest 是否完整一致。
    
    输入：
        无显式输入参数。
    输出：
        pd.DataFrame。
        """
    # 调用 `base.run_paths` 解析当前 run 对应的数据与输出路径，并将结果保存到 `(_, prediction_root, quality_root)`。
    _, prediction_root, quality_root = base.run_paths(RUN_NAME)
    # 调用 `json.loads` 读取源数据，并将结果保存到 `manifest`。
    manifest = json.loads(
        (quality_root / "a2_training_manifest.json").read_text(encoding="utf-8")
    )
    # 组织一组有序值，并保存到 `expected_mode`。
    expected_mode = (RESIDUALIZE_TARGET, PROJECT_MODEL_OUTPUT)
    # 组织一组有序值，并保存到 `stored_mode`。
    stored_mode = (
        bool(manifest.get("residualize_target")),
        bool(manifest.get("project_model_output")),
    )
    # 检查 `stored_mode != expected_mode`；根据条件结果选择对应处理分支。
    if stored_mode != expected_mode:
        # 抛出 `ValueError(f'run={RUN_NAME!r} 的 manifest 开关为 {stored_mode}，当前 CLI 开关为 {expected_mode}；请使用匹配的 ru…`，阻止不满足数据契约的流程继续执行。
        raise ValueError(
            f"run={RUN_NAME!r} 的 manifest 开关为 {stored_mode}，"
            f"当前 CLI 开关为 {expected_mode}；请使用匹配的 run name/参数"
        )
    # 调用 `float` 统一设备或数据类型，并将结果保存到 `stored_lambda`。
    stored_lambda = float(manifest.get("raw_barra_r2_lambda", 0.0))
    # 检查 `stored_lambda != RAW_BARRA_R2_LAMBDA`；根据条件结果选择对应处理分支。
    if stored_lambda != RAW_BARRA_R2_LAMBDA:
        # 抛出 `ValueError(f'run={RUN_NAME!r} 的 manifest λ={stored_lambda}，当前 CLI λ={RAW_BARRA_R2_LAMBDA}')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(
            f"run={RUN_NAME!r} 的 manifest λ={stored_lambda}，"
            f"当前 CLI λ={RAW_BARRA_R2_LAMBDA}"
        )
    # 检查 `bool(manifest.get('adversarial_barra', False)) != adversarial_barra_enabled()`；根据条件结果选择对应处理分支。
    if bool(manifest.get("adversarial_barra", False)) != adversarial_barra_enabled():
        # 抛出 `ValueError(f'run={RUN_NAME!r} 的 No.3 开关与当前 CLI 不一致')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(
            f"run={RUN_NAME!r} 的 No.3 开关与当前 CLI 不一致"
        )
    # 调用 `manifest.get` 按键读取映射中的可选字段，并在缺失时使用约定默认值，并将结果保存到 `stored_conditioning`。
    stored_conditioning = manifest.get("barra_conditioning")
    # 检查 `barra_conditioning_enabled()`；根据条件结果选择对应处理分支。
    if barra_conditioning_enabled():
        # 构造键值映射，并保存到 `expected_conditioning`，供后续按键查询。
        expected_conditioning = {
            "mode": BARRA_CONDITIONING_MODE,
            "hidden_dim": BARRA_GATE_HIDDEN_DIM,
            "industry_embedding_dim": BARRA_INDUSTRY_EMBEDDING_DIM,
            "dropout": BARRA_GATE_DROPOUT,
            "gate_scale": BARRA_GATE_SCALE,
            "gate_projection_initialization": "zeros",
            "multiplicative_only": True,
            "shuffle_policy": (
                "within_date_joint_permutation_by_stable_sample_key_v1"
                if BARRA_CONDITIONING_MODE == "shuffled" else None
            ),
            "r2_uses_true_unmodified_barra": True,
        }
        # 检查 `stored_conditioning != expected_conditioning`；根据条件结果选择对应处理分支。
        if stored_conditioning != expected_conditioning:
            # 抛出 `ValueError('No.4 manifest 与当前条件门控参数不一致')`，阻止不满足数据契约的流程继续执行。
            raise ValueError("No.4 manifest 与当前条件门控参数不一致")
        # 调用 `style_data.require_full_barra_target_sidecar` 校验输入、配置或产物是否满足当前阶段的数据契约，并将结果保存到 `full_target`。
        full_target = style_data.require_full_barra_target_sidecar()
        # 检查 `manifest.get('full_barra_target_sidecar_signature') != full_target['signature']`；根据条件结果选择对应处理分支。
        if manifest.get("full_barra_target_sidecar_signature") != (
            full_target["signature"]
        ):
            # 抛出 `ValueError('No.4 manifest 的完整标签 sidecar 签名不一致')`，阻止不满足数据契约的流程继续执行。
            raise ValueError("No.4 manifest 的完整标签 sidecar 签名不一致")
    # 检查 `stored_conditioning is not None`；根据条件结果选择对应处理分支。
    elif stored_conditioning is not None:
        # 抛出 `ValueError('当前 CLI 未开启 No.4，但 manifest 含条件门控配置')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("当前 CLI 未开启 No.4，但 manifest 含条件门控配置")
    # 调用 `manifest.get` 按键读取映射中的可选字段，并在缺失时使用约定默认值，并将结果保存到 `stored_dual`。
    stored_dual = manifest.get("barra_dual_tower")
    # 检查 `dual_tower_enabled()`；根据条件结果选择对应处理分支。
    if dual_tower_enabled():
        # 构造键值映射，并保存到 `expected_dual`，供后续按键查询。
        expected_dual = {
            "mode": BARRA_DUAL_TOWER_MODE,
            "alpha_loss_weight": DUAL_TOWER_ALPHA_LOSS_WEIGHT,
            "style_loss_weight": DUAL_TOWER_STYLE_LOSS_WEIGHT,
            "style_hidden_dim": DUAL_TOWER_STYLE_HIDDEN_DIM,
            "style_dropout": DUAL_TOWER_STYLE_DROPOUT,
            "industry_embedding_dim": BARRA_INDUSTRY_EMBEDDING_DIM,
            "alpha_r2_lambda": RAW_BARRA_R2_LAMBDA,
            "style_output_initialization": "zeros",
            "published_factor": "prediction_ensemble_alpha",
            "component_columns": [
                "prediction_ensemble_alpha",
                "prediction_ensemble_style",
                "prediction_ensemble_total_return",
            ],
            "ensemble_scaling": "center_components_divide_by_alpha_std",
            "seed_total_audit_policy": (
                "stored_total_must_match_float32_or_bfloat16_alpha_plus_style"
            ),
            "ensemble_total_policy": (
                "reconstruct_float32_total_from_stored_alpha_plus_style"
            ),
            "r2_uses_true_unmodified_barra": True,
        }
        # 检查 `stored_dual != expected_dual`；根据条件结果选择对应处理分支。
        if stored_dual != expected_dual:
            # 抛出 `ValueError('No.6 manifest 与当前双塔参数不一致')`，阻止不满足数据契约的流程继续执行。
            raise ValueError("No.6 manifest 与当前双塔参数不一致")
        # 调用 `style_data.require_full_barra_target_sidecar` 校验输入、配置或产物是否满足当前阶段的数据契约，并将结果保存到 `full_target`。
        full_target = style_data.require_full_barra_target_sidecar()
        # 调用 `style_data.require_dual_tower_target_sidecar` 校验输入、配置或产物是否满足当前阶段的数据契约，并将结果保存到 `dual_target`。
        dual_target = style_data.require_dual_tower_target_sidecar()
        # 检查 `manifest.get('full_barra_target_sidecar_signature') != full_target['signature']`；根据条件结果选择对应处理分支。
        if manifest.get("full_barra_target_sidecar_signature") != full_target["signature"]:
            # 抛出 `ValueError('No.6 完整残差 sidecar 签名不一致')`，阻止不满足数据契约的流程继续执行。
            raise ValueError("No.6 完整残差 sidecar 签名不一致")
        # 检查 `manifest.get('dual_tower_target_sidecar_signature') != dual_target['signature']`；根据条件结果选择对应处理分支。
        if manifest.get("dual_tower_target_sidecar_signature") != dual_target["signature"]:
            # 抛出 `ValueError('No.6 fitted-target sidecar 签名不一致')`，阻止不满足数据契约的流程继续执行。
            raise ValueError("No.6 fitted-target sidecar 签名不一致")
    # 检查 `stored_dual is not None`；根据条件结果选择对应处理分支。
    elif stored_dual is not None:
        # 抛出 `ValueError('当前 CLI 未开启 No.6，但 manifest 含双塔配置')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("当前 CLI 未开启 No.6，但 manifest 含双塔配置")

    # 调用 `manifest.get` 按键读取映射中的可选字段，并在缺失时使用约定默认值，并将结果保存到 `stored_strict`。
    stored_strict = manifest.get("strict_barra_output")
    # 检查 `strict_barra_output_enabled()`；根据条件结果选择对应处理分支。
    if strict_barra_output_enabled():
        # 构造键值映射，并保存到 `expected_strict`，供后续按键查询。
        expected_strict = {
            "weighting": "equal_weight_W_identity",
            "design": [
                "intercept",
                *style_data.STYLE_COLUMNS,
                "industry_fixed_effects",
            ],
            "solver": "industry_within_then_torch_fp32_reduced_qr",
            "post_projection": "zscore_ddof0",
            "applied_during": ["training_loss", "validation", "inference"],
            "published_factor": "prediction_ensemble",
            "pre_neutral_audit_column": "prediction_ensemble_unprojected",
            "orthogonality_requirement": "H_transpose_W_s_equals_zero",
        }
        # 检查 `stored_strict != expected_strict`；根据条件结果选择对应处理分支。
        if stored_strict != expected_strict:
            # 抛出 `ValueError('严格 Barra 输出 manifest 与当前参数不一致')`，阻止不满足数据契约的流程继续执行。
            raise ValueError("严格 Barra 输出 manifest 与当前参数不一致")
    # 检查 `stored_strict is not None`；根据条件结果选择对应处理分支。
    elif stored_strict is not None:
        # 抛出 `ValueError('当前 CLI 未开启严格输出，但 manifest 含对应配置')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("当前 CLI 未开启严格输出，但 manifest 含对应配置")
    # 检查 `adversarial_barra_enabled()`；根据条件结果选择对应处理分支。
    if adversarial_barra_enabled():
        # 构造键值映射，并保存到 `expected_adversarial_config`，供后续按键查询。
        expected_adversarial_config = {
            "gamma_max": ADVERSARIAL_GAMMA_MAX,
            "warmup_epochs": ADVERSARIAL_WARMUP_EPOCHS,
            "cosine_ramp_epochs": ADVERSARIAL_RAMP_EPOCHS,
            "style_loss_weight": ADVERSARIAL_STYLE_LOSS_WEIGHT,
            "industry_loss_weight": ADVERSARIAL_INDUSTRY_LOSS_WEIGHT,
            "head_hidden_dim": ADVERSARIAL_HEAD_HIDDEN_DIM,
            "head_dropout": ADVERSARIAL_HEAD_DROPOUT,
            "industry_classes": INDUSTRY_COUNT,
            "unknown_industry_policy": "masked_from_ce",
            "inference_model": "backbone_and_return_head_only",
            "probe_file": "adversarial_probe_diagnostics.parquet",
            "probe_sample_cap_per_date": PROBE_MAX_SAMPLES_PER_DATE,
            "probe_date_split": [0.4, 0.1, 0.5],
            "probe_batch_size": PROBE_BATCH_SIZE,
            "probe_learning_rate": PROBE_LEARNING_RATE,
            "probe_weight_decay": PROBE_WEIGHT_DECAY,
            "probe_max_epochs": PROBE_MAX_EPOCHS,
            "probe_patience": PROBE_PATIENCE,
            "probe_ridge_alpha": PROBE_RIDGE_ALPHA,
            "probe_mlp_hidden_dim": 128,
            "probe_mlp_dropout": 0.1,
        }
        # 检查 `manifest.get('adversarial_config') != expected_adversarial_config`；根据条件结果选择对应处理分支。
        if manifest.get("adversarial_config") != expected_adversarial_config:
            # 抛出 `ValueError('No.3 manifest 与当前对抗或 probe 参数不一致')`，阻止不满足数据契约的流程继续执行。
            raise ValueError("No.3 manifest 与当前对抗或 probe 参数不一致")
        # 计算 `quality_root / 'adversarial_probe_diagnostics.parquet'`，并将数值结果保存到 `probe_path`。
        probe_path = quality_root / "adversarial_probe_diagnostics.parquet"
        # 检查 `not probe_path.exists()`；根据条件结果选择对应处理分支。
        if not probe_path.exists():
            # 抛出 `FileNotFoundError(probe_path)`，阻止不满足数据契约的流程继续执行。
            raise FileNotFoundError(probe_path)
        # 调用 `pd.read_parquet` 读取源数据，并将结果保存到 `probes`。
        probes = pd.read_parquet(probe_path)
        # 计算 `len(FORECAST_YEARS) * len(SEEDS)`，并将数值结果保存到 `expected_tasks`。
        expected_tasks = len(FORECAST_YEARS) * len(SEEDS)
        # 调用 `probes[['forecast_year', 'seed']].drop_duplicates` 去除重复值并保留唯一记录，并将结果保存到 `observed_tasks`。
        observed_tasks = probes[["forecast_year", "seed"]].drop_duplicates()
        # 检查 `len(observed_tasks) != expected_tasks`；根据条件结果选择对应处理分支。
        if len(observed_tasks) != expected_tasks:
            # 抛出 `AssertionError('No.3 probe 不是8年度×3 seed')`，阻止不满足数据契约的流程继续执行。
            raise AssertionError("No.3 probe 不是8年度×3 seed")
        # 构造去重集合，并保存到 `required_probe_kinds`。
        required_probe_kinds = {"style", "style_summary", "industry"}
        # 检查 `not required_probe_kinds.issubset(set(probes['target_kind']))`；根据条件结果选择对应处理分支。
        if not required_probe_kinds.issubset(set(probes["target_kind"])):
            # 抛出 `AssertionError('No.3 probe 诊断类型不完整')`，阻止不满足数据契约的流程继续执行。
            raise AssertionError("No.3 probe 诊断类型不完整")
        # 调用 `set` 整理数据结构，并将结果保存到 `style_probes`。
        style_probes = set(
            probes.loc[
                probes["target_kind"].eq("style_summary"), "probe"
            ]
        )
        # 调用 `set` 整理数据结构，并将结果保存到 `industry_probes`。
        industry_probes = set(
            probes.loc[probes["target_kind"].eq("industry"), "probe"]
        )
        # 检查 `style_probes != {'linear_ridge', 'mlp'}`；根据条件结果选择对应处理分支。
        if style_probes != {"linear_ridge", "mlp"}:
            # 抛出 `AssertionError('No.3 风格 probe 类型不完整')`，阻止不满足数据契约的流程继续执行。
            raise AssertionError("No.3 风格 probe 类型不完整")
        # 检查 `industry_probes != {'linear_softmax', 'mlp'}`；根据条件结果选择对应处理分支。
        if industry_probes != {"linear_softmax", "mlp"}:
            # 抛出 `AssertionError('No.3 行业 probe 类型不完整')`，阻止不满足数据契约的流程继续执行。
            raise AssertionError("No.3 行业 probe 类型不完整")


    # 调用 `pd.read_parquet` 读取源数据，并将结果保存到 `summary`。
    summary = pd.read_parquet(quality_root / "vivit_training_summary.parquet")
    # 检查 `len(summary) != len(FORECAST_YEARS) * len(SEEDS)`；根据条件结果选择对应处理分支。
    if len(summary) != len(FORECAST_YEARS) * len(SEEDS):
        # 抛出 `AssertionError('训练汇总不是 8 年×3 seed')`，阻止不满足数据契约的流程继续执行。
        raise AssertionError("训练汇总不是 8 年×3 seed")
    # 构造当前步骤所需的列表，并保存到 `records`。
    records: list[dict[str, Any]] = []
    # 构造当前步骤所需的列表，并保存到 `base_columns`。
    base_columns = [
        "sample_index",
        "prediction_ensemble_unprojected",
        "prediction_ensemble_raw",
        "prediction_ensemble",
    ]
    # 检查 `dual_tower_enabled()`；根据条件结果选择对应处理分支。
    if dual_tower_enabled():
        # 把一组结果展开追加到 `base_columns`，保持统一的一维记录序列。
        base_columns.extend(
            [
                "prediction_ensemble_alpha",
                "prediction_ensemble_style",
                "prediction_ensemble_total_return",
            ]
        )
    # 构造当前步骤所需的列表，并保存到 `columns`。
    columns = [
        *base_columns,
        *(
            ["prediction_ensemble_projected"]
            if PROJECT_MODEL_OUTPUT or strict_barra_output_enabled()
            else []
        ),
    ]
    # 遍历 `FORECAST_YEARS`，并把当前元素绑定到 `year` 逐项处理。
    for year in FORECAST_YEARS:
        # 调用 `sequence_data.load_sequence_split` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `split`。
        split = sequence_data.load_sequence_split(year)
        # 调用 `len` 汇总当前数据，并将统计量保存到 `expected`。
        expected = len(split["prediction_indices"])
        # 计算 `prediction_root / f'vivit_predictions.{year}.parquet'`，并将数值结果保存到 `path`。
        path = prediction_root / f"vivit_predictions.{year}.parquet"
        # 检查 `not expected`；根据条件结果选择对应处理分支。
        if not expected:
            # 检查 `path.exists()`；根据条件结果选择对应处理分支。
            if path.exists():
                # 抛出 `AssertionError(f'{year} prediction 为空但存在预测文件')`，阻止不满足数据契约的流程继续执行。
                raise AssertionError(f"{year} prediction 为空但存在预测文件")
            # 把本轮结果追加到 `records`，累积后续需要汇总的记录。
            records.append({"year": year, "rows": 0, "status": "通过"})
            # 跳过当前元素的剩余步骤，继续处理下一项。
            continue
        # 调用 `pd.read_parquet` 读取源数据，并将结果保存到 `frame`。
        frame = pd.read_parquet(path, columns=columns)
        # 检查 `len(frame) != expected or not np.array_equal(frame['sample_index'].to_numpy(np.int64), split['p…`；根据条件结果选择对应处理分支。
        if len(frame) != expected or not np.array_equal(
            frame["sample_index"].to_numpy(np.int64), split["prediction_indices"]
        ):
            # 抛出 `AssertionError(f'{year} 预测索引不一致')`，阻止不满足数据契约的流程继续执行。
            raise AssertionError(f"{year} 预测索引不一致")
        # 检查 `not np.isfinite(frame[columns[1:]].to_numpy(np.float64)).all()`；根据条件结果选择对应处理分支。
        if not np.isfinite(frame[columns[1:]].to_numpy(np.float64)).all():
            # 抛出 `AssertionError(f'{year} 预测列含 NaN/Inf')`，阻止不满足数据契约的流程继续执行。
            raise AssertionError(f"{year} 预测列含 NaN/Inf")
        # 检查 `dual_tower_enabled()`；根据条件结果选择对应处理分支。
        if dual_tower_enabled():
            # 调用 `frame['prediction_ensemble_alpha'].to_numpy` 执行模型推理并生成样本预测分数，并将结果保存到 `alpha`。
            alpha = frame["prediction_ensemble_alpha"].to_numpy(np.float64)
            # 调用 `frame['prediction_ensemble_style'].to_numpy` 执行模型推理并生成样本预测分数，并将结果保存到 `style`。
            style = frame["prediction_ensemble_style"].to_numpy(np.float64)
            # 调用 `frame['prediction_ensemble_total_return'].to_numpy` 执行模型推理并生成样本预测分数，并将结果保存到 `total`。
            total = frame["prediction_ensemble_total_return"].to_numpy(np.float64)
            # 检查 `not np.array_equal(frame['prediction_ensemble'].to_numpy(), frame['prediction_ensemble_alpha'].…`；根据条件结果选择对应处理分支。
            if not np.array_equal(
                frame["prediction_ensemble"].to_numpy(),
                frame["prediction_ensemble_alpha"].to_numpy(),
            ):
                # 抛出 `AssertionError(f'{year} No.6 正式主列必须等于 Alpha')`，阻止不满足数据契约的流程继续执行。
                raise AssertionError(f"{year} No.6 正式主列必须等于 Alpha")
            # 检查 `not np.allclose(total, alpha + style, rtol=1e-06, atol=1e-06)`；根据条件结果选择对应处理分支。
            if not np.allclose(total, alpha + style, rtol=1e-6, atol=1e-6):
                # 抛出 `AssertionError(f'{year} No.6 集成不满足 Alpha+Style=Total')`，阻止不满足数据契约的流程继续执行。
                raise AssertionError(f"{year} No.6 集成不满足 Alpha+Style=Total")
        # 检查 `strict_barra_output_enabled()`；根据条件结果选择对应处理分支。
        if strict_barra_output_enabled():
            # 检查 `np.array_equal(frame['prediction_ensemble_unprojected'].to_numpy(), frame['prediction_ensemble_…`；根据条件结果选择对应处理分支。
            if np.array_equal(
                frame["prediction_ensemble_unprojected"].to_numpy(),
                frame["prediction_ensemble_raw"].to_numpy(),
            ):
                # 抛出 `AssertionError(f'{year} 严格中性主因子不能等于投影前分数')`，阻止不满足数据契约的流程继续执行。
                raise AssertionError(f"{year} 严格中性主因子不能等于投影前分数")
            # 检查 `not np.array_equal(frame['prediction_ensemble_projected'].to_numpy(), frame['prediction_ensembl…`；根据条件结果选择对应处理分支。
            if not np.array_equal(
                frame["prediction_ensemble_projected"].to_numpy(),
                frame["prediction_ensemble_raw"].to_numpy(),
            ):
                # 抛出 `AssertionError(f'{year} 严格中性主因子必须使用投影后分数')`，阻止不满足数据契约的流程继续执行。
                raise AssertionError(f"{year} 严格中性主因子必须使用投影后分数")
        # 检查 `not np.array_equal(frame['prediction_ensemble_unprojected'].to_numpy(), frame['prediction_ensem…`；根据条件结果选择对应处理分支。
        elif not np.array_equal(
            frame["prediction_ensemble_unprojected"].to_numpy(),
            frame["prediction_ensemble_raw"].to_numpy(),
        ):
            # 抛出 `AssertionError(f'{year} 非严格模式主因子必须使用未投影分数')`，阻止不满足数据契约的流程继续执行。
            raise AssertionError(f"{year} 非严格模式主因子必须使用未投影分数")
        # 检查 `not np.array_equal(frame['prediction_ensemble_raw'].to_numpy(), frame['prediction_ensemble'].to…`；根据条件结果选择对应处理分支。
        if not np.array_equal(
            frame["prediction_ensemble_raw"].to_numpy(),
            frame["prediction_ensemble"].to_numpy(),
        ):
            # 抛出 `AssertionError(f'{year} raw 与 notebook03 输入列必须完全相同')`，阻止不满足数据契约的流程继续执行。
            raise AssertionError(f"{year} raw 与 notebook03 输入列必须完全相同")
        # 检查 `PROJECT_MODEL_OUTPUT and np.array_equal(frame['prediction_ensemble_projected'].to_numpy(), fram…`；根据条件结果选择对应处理分支。
        if PROJECT_MODEL_OUTPUT and np.array_equal(
            frame["prediction_ensemble_projected"].to_numpy(),
            frame["prediction_ensemble"].to_numpy(),
        ):
            # 抛出 `AssertionError(f'{year} No.1 投影审计列不应覆盖未投影主因子')`，阻止不满足数据契约的流程继续执行。
            raise AssertionError(f"{year} No.1 投影审计列不应覆盖未投影主因子")
        # 把本轮结果追加到 `records`，累积后续需要汇总的记录。
        records.append({"year": year, "rows": len(frame), "status": "通过"})

    # 调用 `pd.read_parquet` 读取源数据，并将结果保存到 `diagnostics`。
    diagnostics = pd.read_parquet(
        style_diagnostics_root(quality_root)
        / "prediction_style_exposure_diagnostics.parquet"
    )
    # 按 `dual_tower_enabled()` 在两个候选值间选择，并保存到 `diagnostic_prefixes`。
    diagnostic_prefixes = (
        ("alpha", "style", "total")
        if dual_tower_enabled() else ("raw", "effective")
    )
    # 构造去重集合，并保存到 `required`。
    required = {
        f"{prefix}_{stat}_abs_style_{method}"
        for prefix in diagnostic_prefixes
        for stat in ("max", "mean")
        for method in ("pearson", "spearman")
    }
    # 检查 `not required.issubset(diagnostics.columns)`；根据条件结果选择对应处理分支。
    if not required.issubset(diagnostics.columns):
        # 抛出 `AssertionError(f'最终预测缺少风格相关性审计列：{sorted(required - set(diagnostics.columns))}')`，阻止不满足数据契约的流程继续执行。
        raise AssertionError(
            f"最终预测缺少风格相关性审计列：{sorted(required - set(diagnostics.columns))}"
        )
    # 检查 `not np.isfinite(diagnostics[list(required)].to_numpy()).all()`；根据条件结果选择对应处理分支。
    if not np.isfinite(diagnostics[list(required)].to_numpy()).all():
        # 抛出 `AssertionError('最终预测的风格相关性审计含 NaN/Inf')`，阻止不满足数据契约的流程继续执行。
        raise AssertionError("最终预测的风格相关性审计含 NaN/Inf")
    # 检查 `PROJECT_MODEL_OUTPUT or strict_barra_output_enabled()`；根据条件结果选择对应处理分支。
    if PROJECT_MODEL_OUTPUT or strict_barra_output_enabled():
        # 调用 `float` 统一设备或数据类型，并将结果保存到 `maximum`。
        maximum = float(diagnostics["effective_max_abs_style_pearson"].max())
        # 检查 `maximum > 5e-05`；根据条件结果选择对应处理分支。
        if maximum > 5e-5:
            # 抛出 `AssertionError(f'投影后预测未达到数值正交：max |Pearson|={maximum:.3e}')`，阻止不满足数据契约的流程继续执行。
            raise AssertionError(
                f"投影后预测未达到数值正交：max |Pearson|={maximum:.3e}"
            )
        # 检查 `strict_barra_output_enabled()`；根据条件结果选择对应处理分支。
        if strict_barra_output_enabled():
            # 调用 `float` 统一设备或数据类型，并将结果保存到 `maximum_r_squared`。
            maximum_r_squared = float(
                diagnostics["effective_joint_barra_r_squared"].max()
            )
            # 检查 `maximum_r_squared > 1e-06`；根据条件结果选择对应处理分支。
            if maximum_r_squared > 1e-6:
                # 抛出 `AssertionError('严格中性预测的联合 Barra R² 未接近零')`，阻止不满足数据契约的流程继续执行。
                raise AssertionError("严格中性预测的联合 Barra R² 未接近零")
            # 调用 `float` 统一设备或数据类型，并将结果保存到 `maximum_industry_mean`。
            maximum_industry_mean = float(
                diagnostics["effective_max_abs_industry_mean"].max()
            )
            # 检查 `maximum_industry_mean > 5e-05`；根据条件结果选择对应处理分支。
            if maximum_industry_mean > 5e-5:
                # 抛出 `AssertionError('严格中性预测的行业组均值未接近零')`，阻止不满足数据契约的流程继续执行。
                raise AssertionError(
                    "严格中性预测的行业组均值未接近零"
                )
    # 将 `pd.DataFrame(records)` 作为当前函数结果返回给调用方。
    return pd.DataFrame(records)

# 定义 `write_report` 函数，封装这一阶段可独立调用的处理步骤。
def write_report() -> Path:
    """汇总训练、暴露、probe 和预测结果并写入实验报告。
    
    输入：
        无显式输入参数。
    输出：
        Path。
        """
    # 调用 `base.run_paths` 解析当前 run 对应的数据与输出路径，并将结果保存到 `(_, _, quality_root)`。
    _, _, quality_root = base.run_paths(RUN_NAME)
    # 调用 `pd.read_parquet` 读取源数据，并将结果保存到 `summary`。
    summary = pd.read_parquet(quality_root / "vivit_training_summary.parquet")
    # 调用 `validate_outputs` 校验输入、配置或产物是否满足当前阶段的数据契约，并将结果保存到 `audit`。
    manifest = json.loads(
        (quality_root / "a2_training_manifest.json").read_text(encoding="utf-8")
    )
    # 执行当前分布式训练步骤并保持既定数据语义。
    audit = validate_outputs()
    # 调用 `pd.read_parquet` 读取源数据，并将结果保存到 `diagnostics`。
    diagnostics = pd.read_parquet(
        style_diagnostics_root(quality_root)
        / "prediction_style_exposure_diagnostics.parquet"
    )
    # 调用 `summary.groupby('forecast_year', as_index=False).agg` 按约定截面或分组口径汇总统计结果，并将结果保存到 `yearly`。
    yearly = summary.groupby("forecast_year", as_index=False).agg(
        best_effective_rank_ic=("best_validation_rank_ic", "mean"),
        best_original_rank_ic=("best_validation_original_rank_ic", "mean"),
        max_validation_effective_style_pearson=(
            "best_validation_effective_max_abs_style_pearson", "max"
        ),
        mean_best_epoch=("best_epoch", "mean"),
        mean_validation_barra_r_squared=(
            "best_validation_barra_r_squared", "mean"
        ),
    )
    # 检查 `strict_barra_output_enabled()`；根据条件结果选择对应处理分支。
    if strict_barra_output_enabled():
        # 调用 `summary.groupby('forecast_year')['best_validation_effective_bar…` 按约定截面或分组口径汇总统计结果，并将结果保存到 `yearly['mean_validation_effective_barra_r_squared']`。
        yearly["mean_validation_effective_barra_r_squared"] = (
            summary.groupby("forecast_year")[
                "best_validation_effective_barra_r_squared"
            ].mean().to_numpy()
        )
        # 调用 `summary.groupby('forecast_year')['best_validation_effective_max…` 按约定截面或分组口径汇总统计结果，并将结果保存到 `yearly['max_validation_effective_industry_mean']`。
        yearly["max_validation_effective_industry_mean"] = (
            summary.groupby("forecast_year")[
                "best_validation_effective_max_abs_industry_mean"
            ].max().to_numpy()
        )
    # 检查 `adversarial_barra_enabled()`；根据条件结果选择对应处理分支。
    if adversarial_barra_enabled():
        # 调用 `summary.groupby('forecast_year', as_index=False).agg` 按约定截面或分组口径汇总统计结果，并将结果保存到 `adversarial_yearly`。
        adversarial_yearly = summary.groupby(
            "forecast_year", as_index=False
        ).agg(
            validation_adversarial_style_r_squared_mean=(
                "best_validation_adversarial_style_r_squared_mean", "mean"
            ),
            validation_adversarial_industry_accuracy=(
                "best_validation_adversarial_industry_accuracy", "mean"
            ),
            validation_adversarial_industry_balanced_accuracy=(
                "best_validation_adversarial_industry_balanced_accuracy",
                "mean",
            ),
        )
        # 调用 `yearly.merge` 按键合并数据，并将对齐后的结果保存到 `yearly`。
        yearly = yearly.merge(
            adversarial_yearly, on="forecast_year", validate="one_to_one"
        )
    # 检查 `dual_tower_enabled()`；根据条件结果选择对应处理分支。
    if dual_tower_enabled():
        # 调用 `diagnostics.groupby('year', as_index=False).agg` 按预先声明的聚合规则汇总分组数据，并将结果保存到 `prediction_style`。
        prediction_style = diagnostics.groupby("year", as_index=False).agg(
            alpha_mean_joint_barra_r_squared=(
                "alpha_joint_barra_r_squared", "mean"
            ),
            style_mean_joint_barra_r_squared=(
                "style_joint_barra_r_squared", "mean"
            ),
            total_mean_joint_barra_r_squared=(
                "total_joint_barra_r_squared", "mean"
            ),
            alpha_residual_rank_ic=("alpha_residual_rank_ic", "mean"),
            style_fitted_rank_ic=("style_fitted_rank_ic", "mean"),
            total_original_rank_ic=("total_original_rank_ic", "mean"),
            alpha_style_pearson=("alpha_style_pearson", "mean"),
            max_additivity_error=(
                "dual_additivity_max_abs_error", "max"
            ),
        )
    # 当前置条件不成立时进入备用分支，保证其余合法情况仍被处理。
    else:
        # 调用 `diagnostics.groupby('year', as_index=False).agg` 按预先声明的聚合规则汇总分组数据，并将结果保存到 `prediction_style`。
        prediction_style = diagnostics.groupby("year", as_index=False).agg(
            raw_max_daily_style_pearson=("raw_max_abs_style_pearson", "max"),
            effective_max_daily_style_pearson=(
                "effective_max_abs_style_pearson", "max"
            ),
            raw_mean_daily_style_spearman=(
                "raw_mean_abs_style_spearman", "mean"
            ),
            effective_mean_daily_style_spearman=(
                "effective_mean_abs_style_spearman", "mean"
            ),
        )
        # 检查 `industry_sidecar_required()`；根据条件结果选择对应处理分支。
        if industry_sidecar_required():
            # 调用 `diagnostics.groupby('year')['raw_joint_barra_r_squared'].mean()…` 按股票和交易日键对齐来自不同来源的数据，并将结果保存到 `prediction_style['raw_mean_daily_joint_barra_r_squared']`。
            prediction_style["raw_mean_daily_joint_barra_r_squared"] = (
                diagnostics.groupby("year")[
                    "raw_joint_barra_r_squared"
                ].mean().to_numpy()
            )
            # 检查 `strict_barra_output_enabled()`；根据条件结果选择对应处理分支。
            if strict_barra_output_enabled():
                # 调用 `diagnostics.groupby('year')['effective_joint_barra_r_squared'].…` 按股票和交易日键对齐来自不同来源的数据，并将结果保存到 `prediction_style['effective_max_daily_joint_barra_r_squared']`。
                prediction_style["effective_max_daily_joint_barra_r_squared"] = (
                    diagnostics.groupby("year")[
                        "effective_joint_barra_r_squared"
                    ].max().to_numpy()
                )
                # 调用 `to_numpy` 把表格列转换为指定精度的 NumPy 数组，并将结果保存到 `prediction_style['effective_max_daily_industry_mean']`。
                prediction_style["effective_max_daily_industry_mean"] = (
                    diagnostics.groupby("year")[
                        "effective_max_abs_industry_mean"
                    ].max().to_numpy()
                )
    # 按 `dual_tower_enabled()` 在两个候选值间选择，并保存到 `target_description`。
    target_description = (
        "每日用完整 Barra 风险矩阵把原始标签分为拟合项与残差，"
        "二者分别做 ddof=0 Z-score，监督 Style 与 Alpha；Total 监督原始标签。"
        if dual_tower_enabled()
        else "每日用截距、十个连续风格/反转变量和行业固定效应联合 OLS，"
        "取残差后做 ddof=0 Z-score。"
        if barra_conditioning_enabled()
        else "每日用九个 CNE5S 风格和 20 日反转联合 OLS，"
        "取残差后做 ddof=0 Z-score。"
        if RESIDUALIZE_TARGET
        else "使用 feature store 中的原始标准化标签。"
    )
    # 按 `dual_tower_enabled()` 在两个候选值间选择，并保存到 `output_description`。
    output_description = (
        "双塔同时输出 Alpha、Style 与 Total=Alpha+Style；正式主列保存未经投影的 Alpha，"
        "三个 seed 的组件共同使用 Alpha 日截面标准差缩放。"
        if dual_tower_enabled()
        else "每个完整交易日将条件模型分数投影掉十风格和行业固定效应，"
        "再做 ddof=0 Z-score；投影分数同时用于损失、验证和正式发布，"
        "投影前分数只保留为审计列。"
        if strict_barra_output_enabled()
        else
        "每个完整交易日将模型分数投影到十风格正交补空间，再做 ddof=0 Z-score；"
        "该有效分数用于损失和验证，并在推理时保存为 projected 审计列；"
        "正式发布主因子列保存未投影模型原始分数。"
        if PROJECT_MODEL_OUTPUT
        else "直接使用模型原始分数；正式发布主因子列同样保存原始分数。"
    )
    # 检查 `dual_tower_enabled()`；根据条件结果选择对应处理分支。
    if dual_tower_enabled():
        # 计算 `f'{LOSS_NAME}(alpha+style, source_target) + {DUAL_TOWER_ALPHA_LOSS_WEIGHT} × {LOSS_NAME}(alpha,…`，并将结果写入 `loss_description`。
        loss_description = (
            f"{LOSS_NAME}(alpha+style, source_target) + "
            f"{DUAL_TOWER_ALPHA_LOSS_WEIGHT} × {LOSS_NAME}(alpha, full_barra_residual) + "
            f"{DUAL_TOWER_STYLE_LOSS_WEIGHT} × {LOSS_NAME}(style, full_barra_fitted) + "
            f"{RAW_BARRA_R2_LAMBDA} × R²(alpha | 十风格 + 行业)"
        )
        # 设置 `penalty_description` 的固定配置值，供后续流程统一使用。
        penalty_description = (
            "联合 R² 只约束 Alpha，且始终使用股票真实 Barra；"
            "正式 Alpha 不做训练后投影。"
        )
        # 计算 `f'独立进入 Alpha 条件门控和 Style 塔；当前输入模式为 {BARRA_DUAL_TOWER_MODE}，UNKNOWN embedding 固定为零。'`，并将结果写入 `industry_description`。
        industry_description = (
            f"独立进入 Alpha 条件门控和 Style 塔；当前输入模式为 "
            f"{BARRA_DUAL_TOWER_MODE}，UNKNOWN embedding 固定为零。"
        )
    # 检查 `strict_barra_output_enabled()`；根据条件结果选择对应处理分支。
    elif strict_barra_output_enabled():
        # 计算 `f'{LOSS_NAME}(strict_projected_score, full_barra_residual_target)'`，并将结果写入 `loss_description`。
        loss_description = (
            f"{LOSS_NAME}(strict_projected_score, "
            "full_barra_residual_target)"
        )
        # 设置 `penalty_description` 的固定配置值，供后续流程统一使用。
        penalty_description = (
            "不使用软 R²；输出层通过等权横截面投影严格删除十风格与行业固定效应。"
        )
        # 计算 `f'行业 embedding 进入条件模型，真实行业同时进入输出投影；当前条件模式为 {BARRA_CONDITIONING_MODE}。'`，并将结果写入 `industry_description`。
        industry_description = (
            f"行业 embedding 进入条件模型，真实行业同时进入输出投影；"
            f"当前条件模式为 {BARRA_CONDITIONING_MODE}。"
        )
    # 检查 `barra_conditioning_enabled()`；根据条件结果选择对应处理分支。
    elif barra_conditioning_enabled():
        # 计算 `f'{LOSS_NAME}(raw_score, full_barra_residual_target) + {RAW_BARRA_R2_LAMBDA} × R²(raw_score | 十…`，并将结果写入 `loss_description`。
        loss_description = (
            f"{LOSS_NAME}(raw_score, full_barra_residual_target) + "
            f"{RAW_BARRA_R2_LAMBDA} × R²(raw_score | 十风格 + 行业)"
        )
        # 设置 `penalty_description` 的固定配置值，供后续流程统一使用。
        penalty_description = (
            "Barra 条件乘法门控直接生成原始 Alpha；"
            "联合 R² 始终使用未置乱的真实 Barra。"
        )
        # 计算 `f'行业 embedding 进入模型，UNKNOWN=0 固定为零；当前条件模式为 {BARRA_CONDITIONING_MODE}。'`，并将结果写入 `industry_description`。
        industry_description = (
            f"行业 embedding 进入模型，UNKNOWN=0 固定为零；"
            f"当前条件模式为 {BARRA_CONDITIONING_MODE}。"
        )
    # 检查 `adversarial_barra_enabled()`；根据条件结果选择对应处理分支。
    elif adversarial_barra_enabled():
        # 计算 `f'{LOSS_NAME}(raw_score, source_target) + masked style MSE + inverse-sqrt weighted industry CE/…`，并将结果写入 `loss_description`。
        loss_description = (
            f"{LOSS_NAME}(raw_score, source_target) + masked style MSE + "
            "inverse-sqrt weighted industry CE/log(32)，对抗项经 GRL 回传。"
        )
        # 计算 `f'CLS 对抗不变表征；gamma_max={ADVERSARIAL_GAMMA_MAX}，warmup={ADVERSARIAL_WARMUP_EPOCHS}，cosine ramp={…`，并将结果写入 `penalty_description`。
        penalty_description = (
            f"CLS 对抗不变表征；gamma_max={ADVERSARIAL_GAMMA_MAX}，"
            f"warmup={ADVERSARIAL_WARMUP_EPOCHS}，"
            f"cosine ramp={ADVERSARIAL_RAMP_EPOCHS}。"
        )
        # 设置 `industry_description` 的固定配置值，供后续流程统一使用。
        industry_description = (
            "32个真实行业作为训练期对抗监督；UNKNOWN 不进 CE；"
            "Barra 不进入预测 forward。"
        )
    # 检查 `raw_barra_r2_enabled()`；根据条件结果选择对应处理分支。
    elif raw_barra_r2_enabled():
        # 计算 `f'{LOSS_NAME}(raw_score, source_target) + {RAW_BARRA_R2_LAMBDA} × R²(raw_score | 十风格 + 行业)'`，并将结果写入 `loss_description`。
        loss_description = (
            f"{LOSS_NAME}(raw_score, source_target) + "
            f"{RAW_BARRA_R2_LAMBDA} × R²(raw_score | 十风格 + 行业)"
        )
        # 设置 `penalty_description` 的固定配置值，供后续流程统一使用。
        penalty_description = (
            "直接惩罚原始输出的联合线性可解释度；行业采用固定效应。"
        )
        # 设置 `industry_description` 的固定配置值，供后续流程统一使用。
        industry_description = "进入联合 R² 惩罚；不作为模型预测输入。"
    # 当前置条件不成立时进入备用分支，保证其余合法情况仍被处理。
    else:
        # 计算 `f'{LOSS_NAME}(effective_prediction, effective_target)'`，并将结果写入 `loss_description`。
        loss_description = f"{LOSS_NAME}(effective_prediction, effective_target)"
        # 设置 `penalty_description` 的固定配置值，供后续流程统一使用。
        penalty_description = "未使用 raw-score 联合 Barra R² 正则。"
        # 设置 `industry_description` 的固定配置值，供后续流程统一使用。
        industry_description = "不进入标签回归或分数投影。"

    # 设置 `probe_section` 的固定配置值，供后续流程统一使用。
    probe_section = ""
    # 检查 `adversarial_barra_enabled()`；根据条件结果选择对应处理分支。
    if adversarial_barra_enabled():
        # 调用 `pd.read_parquet` 读取源数据，并将结果保存到 `probes`。
        probes = pd.read_parquet(
            quality_root / "adversarial_probe_diagnostics.parquet"
        )
        # 调用 `probes[probes['target_kind'].isin(['style_summary', 'industry']…` 按约定截面或分组口径汇总统计结果，并将结果保存到 `probe_summary`。
        probe_summary = probes[
            probes["target_kind"].isin(["style_summary", "industry"])
        ].groupby(["probe", "target_kind"], as_index=False).agg(
            mean_style_r_squared=("r_squared", "mean"),
            max_style_r_squared=("max_r_squared", "mean"),
            style_mse=("mse", "mean"),
            industry_accuracy=("accuracy", "mean"),
            industry_balanced_accuracy=("balanced_accuracy", "mean"),
            industry_macro_f1=("macro_f1", "mean"),
            unknown_rate=("unknown_rate", "mean"),
        )
        # 计算 `'## 冻结 CLS Probe\n\n' + base.markdown_table(probe_summary.round(8)) + '\n\n'`，并将数值结果保存到 `probe_section`。
        probe_section = (
            "## 冻结 CLS Probe\n\n"
            + base.markdown_table(probe_summary.round(8))
            + "\n\n"
        )

    # 计算 `f'# ViViT 风格解耦训练报告\n\n实验：`{RUN_NAME}`\n\n## 方法\n\n- 标签：{target_description}\n- 模型分数：{output_des…`，并将结果写入 `report`。
    report = f"""# ViViT 风格解耦训练报告

实验：`{RUN_NAME}`

## 方法

- 训练路线：`{manifest.get("execution_route", "a2")}`（world size={manifest.get("world_size", 1)}）
- 标签：{target_description}
- 模型分数：{output_description}
- 损失：`{loss_description}`。
- Barra 相关性惩罚：{penalty_description}
- 行业：{industry_description}

## 验证结果

{base.markdown_table(yearly.round(8))}

## 预测核验

{base.markdown_table(audit)}

## 样本外预测风格相关性

{base.markdown_table(prediction_style.round(8))}

{probe_section}样本外回测可使用：

```python
FOLDER_NAME = "{RUN_NAME}"
BASELINE_RUN_NAME = "sigmoid_daybatch_rankic"
BACKTEST_MODE = "mode2_t_no_filter"
```
"""
    # 计算 `quality_root / 'EXPERIMENT_REPORT.md'`，并将数值结果保存到 `path`。
    path = quality_root / "EXPERIMENT_REPORT.md"
    # 调用 `sequence_data.atomic_write_text` 先写临时文件再原子替换正式产物，避免留下半成品。
    sequence_data.atomic_write_text(report, path)
    # 将 `path` 作为当前函数结果返回给调用方。
    return path

# 定义 `smoke_test` 函数，封装这一阶段可独立调用的处理步骤。
def smoke_test(
    forecast_year: int = 2018,
    seed: int = 42,
    max_batches: int = 1,
) -> dict[str, Any]:
    """运行小批量前向、反向和验证冒烟测试，不生成正式产物。
    
    输入：
        forecast_year: int。
        seed: int。
        max_batches: int。
    输出：
        dict[str, Any]。
        """
    # 将 `train_task(forecast_year, seed, 0, max_epochs=1, max_batches=max_batches, persist=False)` 作为当前函数结果返回给调用方。
    return train_task(
        forecast_year, seed, 0, max_epochs=1, max_batches=max_batches, persist=False
    )

# 定义 `add_training_arguments` 函数，封装这一阶段可独立调用的处理步骤。
def add_training_arguments(parser: argparse.ArgumentParser) -> None:
    """向命令行解析器注册训练后端参数。
    
    输入：
        parser: argparse.ArgumentParser。
    输出：
        None。
        """
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument("--run-name", default=RUN_NAME)
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument(
        "--train-batch-mode", choices=base.TRAIN_BATCH_MODES, default=TRAIN_BATCH_MODE
    )
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument("--loss-name", choices=base.LOSS_NAMES, default=LOSS_NAME)
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument("--rank-temperature", type=float, default=RANK_TEMPERATURE)
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument("--turnover-lambda", type=float, default=TURNOVER_LAMBDA)
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument("--huber-delta", type=float, default=HUBER_DELTA)
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument("--smooth-l1-beta", type=float, default=SMOOTH_L1_BETA)
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument("--hybrid-weight", type=float, default=HYBRID_WEIGHT)
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument(
        "--validation-metric",
        choices=base.VALIDATION_METRICS,
        default=VALIDATION_METRIC,
    )
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument("--workers-per-task", type=int, default=WORKERS_PER_TASK)
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument("--patience", type=int, default=PATIENCE)
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument("--compile", type=int, choices=(0, 1), default=USE_TORCH_COMPILE)
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument("--compile-mode", default=COMPILE_MODE)
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument("--fused-adam", type=int, choices=(0, 1), default=USE_FUSED_ADAM)
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument(
        "--cuda-prefetch", type=int, choices=(0, 1), default=USE_CUDA_PREFETCH
    )
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument("--overwrite", type=int, choices=(0, 1), default=OVERWRITE)
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument("--residualize-target", type=int, choices=(0, 1), default=1)
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument(
        "--project-model-output", type=int, choices=(0, 1), default=1
    )
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument(
        "--raw-barra-r2-lambda",
        type=float,
        default=RAW_BARRA_R2_LAMBDA,
    )
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument(
        "--barra-conditioning-mode",
        choices=BARRA_CONDITIONING_MODES,
        default=BARRA_CONDITIONING_MODE,
    )
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument(
        "--barra-output-neutralization", type=int, choices=(0, 1),
        default=int(BARRA_OUTPUT_NEUTRALIZATION),
    )
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument(
        "--barra-dual-tower", type=int, choices=(0, 1),
        default=int(BARRA_DUAL_TOWER),
    )
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument(
        "--barra-dual-tower-mode",
        choices=BARRA_DUAL_TOWER_MODES,
        default=BARRA_DUAL_TOWER_MODE,
    )
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument(
        "--dual-tower-alpha-loss-weight", type=float,
        default=DUAL_TOWER_ALPHA_LOSS_WEIGHT,
    )
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument(
        "--dual-tower-style-loss-weight", type=float,
        default=DUAL_TOWER_STYLE_LOSS_WEIGHT,
    )
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument(
        "--dual-tower-style-hidden-dim", type=int,
        default=DUAL_TOWER_STYLE_HIDDEN_DIM,
    )
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument(
        "--dual-tower-style-dropout", type=float,
        default=DUAL_TOWER_STYLE_DROPOUT,
    )
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument(
        "--barra-gate-hidden-dim",
        type=int,
        default=BARRA_GATE_HIDDEN_DIM,
    )
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument(
        "--barra-industry-embedding-dim",
        type=int,
        default=BARRA_INDUSTRY_EMBEDDING_DIM,
    )
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument(
        "--barra-gate-dropout",
        type=float,
        default=BARRA_GATE_DROPOUT,
    )
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument(
        "--barra-gate-scale",
        type=float,
        default=BARRA_GATE_SCALE,
    )
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument(
        "--adversarial-barra", type=int, choices=(0, 1),
        default=int(ADVERSARIAL_BARRA),
    )
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument(
        "--adversarial-gamma-max", type=float,
        default=ADVERSARIAL_GAMMA_MAX,
    )
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument(
        "--adversarial-warmup-epochs", type=int,
        default=ADVERSARIAL_WARMUP_EPOCHS,
    )
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument(
        "--adversarial-ramp-epochs", type=int,
        default=ADVERSARIAL_RAMP_EPOCHS,
    )
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument(
        "--adversarial-style-loss-weight", type=float,
        default=ADVERSARIAL_STYLE_LOSS_WEIGHT,
    )
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument(
        "--adversarial-industry-loss-weight", type=float,
        default=ADVERSARIAL_INDUSTRY_LOSS_WEIGHT,
    )
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument(
        "--adversarial-head-hidden-dim", type=int,
        default=ADVERSARIAL_HEAD_HIDDEN_DIM,
    )
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument(
        "--adversarial-head-dropout", type=float,
        default=ADVERSARIAL_HEAD_DROPOUT,
    )


# 定义 `worker_command` 函数，封装这一阶段可独立调用的处理步骤。
def worker_command(
    args: argparse.Namespace, year: int, seed: int, physical_gpu: int
) -> list[str]:
    """构造单个年度×seed worker 的命令行参数。
    
    输入：
        args: argparse.Namespace。
        year: int。
        seed: int。
        physical_gpu: int。
    输出：
        list[str]。
        """
    # 构造当前步骤所需的列表，并保存到 `command`。
    command = [
        sys.executable,
        str(SCRIPT_PATH),
        "worker",
        "--worker-year", str(year),
        "--worker-seed", str(seed),
        "--worker-physical-gpu", str(physical_gpu),
    ]
    # 遍历 `('run_name', 'train_batch_mode', 'loss_name', 'rank_temperature', 'turnover_lambda', 'huber_del…`，并把当前元素绑定到 `name` 逐项处理。
    for name in (
        "run_name", "train_batch_mode", "loss_name", "rank_temperature",
        "turnover_lambda", "huber_delta", "smooth_l1_beta", "hybrid_weight",
        "validation_metric", "workers_per_task", "learning_rate", "max_epochs",
        "patience", "compile", "compile_mode", "fused_adam", "cuda_prefetch",
        "overwrite", "residualize_target", "project_model_output",
        "raw_barra_r2_lambda",
        "barra_conditioning_mode", "barra_output_neutralization",
        "barra_dual_tower", "barra_dual_tower_mode",
        "dual_tower_alpha_loss_weight", "dual_tower_style_loss_weight",
        "dual_tower_style_hidden_dim", "dual_tower_style_dropout",
        "barra_gate_hidden_dim",
        "barra_industry_embedding_dim", "barra_gate_dropout",
        "barra_gate_scale",
        "adversarial_barra", "adversarial_gamma_max",
        "adversarial_warmup_epochs", "adversarial_ramp_epochs",
        "adversarial_style_loss_weight",
        "adversarial_industry_loss_weight",
        "adversarial_head_hidden_dim", "adversarial_head_dropout",
    ):
        # 把一组结果展开追加到 `command`，保持统一的一维记录序列。
        command.extend([f"--{name.replace('_', '-')}", str(getattr(args, name))])
    # 将 `command` 作为当前函数结果返回给调用方。
    return command


# 执行当前分布式训练步骤并保持既定数据语义。
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
def _barra_ddp_rank(
    rank: int,
    world_size: int,
    args: argparse.Namespace,
    smoke: bool,
) -> None:
    """执行 _barra_ddp_rank 对应的分布式训练处理。
    
    输入：
        参数见函数类型标注。
    输出：
        结果见函数返回类型标注。
    """
    # 执行当前分布式训练步骤并保持既定数据语义。
    args.execution_route = "ddp"
    # 执行当前分布式训练步骤并保持既定数据语义。
    args.ddp_world_size = world_size
    # 执行当前分布式训练步骤并保持既定数据语义。
    configure_from_args(args)
    # 执行当前分布式训练步骤并保持既定数据语义。
    context = distributed.initialize_process_group(rank, world_size, backend="nccl")
    # 执行当前分布式训练步骤并保持既定数据语义。
    try:
        # 执行当前分布式训练步骤并保持既定数据语义。
        result = train_task(
            args.forecast_year if smoke else args.worker_year,
            args.seed if smoke else args.worker_seed,
            int(EXECUTION_DEVICE_IDS[0]),
            max_epochs=1 if smoke else None,
            max_batches=args.max_batches if smoke else None,
            persist=not smoke,
            distributed_context=context,
        )
        # 执行当前分布式训练步骤并保持既定数据语义。
        if context.is_primary:
            # 执行当前分布式训练步骤并保持既定数据语义。
            result.update({
                "execution_route": "ddp",
                "world_size": world_size,
                "persistent_outputs_written": not smoke,
            })
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
    args.execution_route = "ddp"
    # 执行当前分布式训练步骤并保持既定数据语义。
    args.ddp_world_size = len(devices)
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
        _barra_ddp_rank,
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
    configure_from_args(args)
    # 执行当前分布式训练步骤并保持既定数据语义。
    _, _, quality_root = base.run_paths(RUN_NAME)
    # 执行当前分布式训练步骤并保持既定数据语义。
    log_dir = quality_root / "ddp_task_logs"
    # 执行当前分布式训练步骤并保持既定数据语义。
    log_dir.mkdir(parents=True, exist_ok=True)
    # 执行当前分布式训练步骤并保持既定数据语义。
    started = time.perf_counter()
    # 执行当前分布式训练步骤并保持既定数据语义。
    print(
        f"style-neutral DDP start: "
        f"tasks={len(FORECAST_YEARS) * len(SEEDS)}, "
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
                tail_text = "\n".join(
                    log_path.read_text(
                        encoding="utf-8", errors="replace"
                    ).splitlines()[-100:]
                )
                # 执行当前分布式训练步骤并保持既定数据语义。
                raise RuntimeError(
                    f"DDP 任务失败：{year}/seed={seed}\n{tail_text}"
                )
            # 执行当前分布式训练步骤并保持既定数据语义。
            print(f"done DDP: {year}/seed={seed}", flush=True)
    # 执行当前分布式训练步骤并保持既定数据语义。
    result = finalize_run(time.perf_counter() - started)
    # 执行当前分布式训练步骤并保持既定数据语义。
    print(validate_outputs().to_string(index=False), flush=True)
    # 执行当前分布式训练步骤并保持既定数据语义。
    print(f"实验报告：{write_report()}", flush=True)
    # 执行当前分布式训练步骤并保持既定数据语义。
    return result


# 定义 `run_a2` 函数，封装这一阶段可独立调用的处理步骤。
def run_a2(args: argparse.Namespace) -> pd.DataFrame:
    """调度所有年度和随机种子任务，并支持断点续训与失败恢复。
    
    输入：
        args: argparse.Namespace。
    输出：
        pd.DataFrame。
        """
    # 调用 `tuple` 整理数据结构，并将结果保存到 `devices`。
    devices = tuple(int(value) for value in args.device_ids.split(","))
    # 执行当前分布式训练步骤并保持既定数据语义。
    args.execution_route = "a2"
    # 执行当前分布式训练步骤并保持既定数据语义。
    args.ddp_world_size = 1
    # 执行当前分布式训练步骤并保持既定数据语义。
    configure_from_args(args)
    # 检查 `not devices or len(set(devices)) != len(devices)`；根据条件结果选择对应处理分支。
    if not devices or len(set(devices)) != len(devices):
        # 抛出 `ValueError('--device-ids 为空或包含重复 GPU')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("--device-ids 为空或包含重复 GPU")
    # 调用 `deque` 创建支持队首弹出的任务队列，并将结果保存到 `tasks`。
    tasks = deque((year, seed) for year in FORECAST_YEARS for seed in SEEDS)
    # 调用 `base.run_paths` 解析当前 run 对应的数据与输出路径，并将结果保存到 `(_, _, quality_root)`。
    _, _, quality_root = base.run_paths(RUN_NAME)
    # 计算 `quality_root / 'style_task_logs'`，并将数值结果保存到 `log_dir`。
    log_dir = quality_root / "style_task_logs"
    # 调用 `log_dir.mkdir` 创建输出目录，确保后续文件可以安全写入。
    log_dir.mkdir(parents=True, exist_ok=True)
    # 构造当前步骤所需的列表，并保存到 `running`。
    running: list[dict[str, Any]] = []
    # 调用 `time.perf_counter` 读取高精度单调时钟以计算耗时，并将结果保存到 `started`。
    started = time.perf_counter()
    # 调用 `print` 记录当前流程状态或诊断信息。
    print(
        f"style-neutral start: tasks={len(tasks)}, GPUs={devices}, "
        f"tasks_per_gpu={TASKS_PER_GPU}, residualize_target={RESIDUALIZE_TARGET}, "
        f"project_model_output={PROJECT_MODEL_OUTPUT}, "
        f"raw_barra_r2_lambda={RAW_BARRA_R2_LAMBDA}, "
        f"adversarial_barra={ADVERSARIAL_BARRA}, "
        f"adversarial_gamma_max={ADVERSARIAL_GAMMA_MAX}, "
        f"barra_conditioning_mode={BARRA_CONDITIONING_MODE}, "
        f"barra_output_neutralization={BARRA_OUTPUT_NEUTRALIZATION}, "
        f"barra_dual_tower={BARRA_DUAL_TOWER}, "
        f"barra_dual_tower_mode={BARRA_DUAL_TOWER_MODE}",
        flush=True,
    )
    # 尝试执行本段主逻辑；若发生异常，则交由后续异常分支处理。
    try:
        # 只要 `tasks or running` 成立就继续循环，直至满足退出条件。
        while tasks or running:
            # 遍历 `devices`，并把当前元素绑定到 `physical_gpu` 逐项处理。
            for physical_gpu in devices:
                # 调用 `sum` 汇总当前数据，并将统计量保存到 `active`。
                active = sum(
                    item["physical_gpu"] == physical_gpu for item in running
                )
                # 只要 `tasks and active < TASKS_PER_GPU` 成立就继续循环，直至满足退出条件。
                while tasks and active < TASKS_PER_GPU:
                    # 调用 `tasks.popleft` 从任务队列头部取出下一项任务，并将结果保存到 `(year, seed)`。
                    year, seed = tasks.popleft()
                    # 计算 `log_dir / f'forecast_{year}.seed_{seed}.log'`，并将数值结果保存到 `log_path`。
                    log_path = log_dir / f"forecast_{year}.seed_{seed}.log"
                    # 调用 `log_path.open` 读取源数据，并将结果保存到 `handle`。
                    handle = log_path.open("a", encoding="utf-8")
                    # 调用 `handle.write` 把本轮任务状态追加到日志文件。
                    handle.write(
                        f"\n=== launch {year}/seed={seed} at "
                        f"{time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
                    )
                    # 调用 `handle.flush` 把内存映射中的缓存修改刷新到磁盘。
                    handle.flush()
                    # 复制 `os.environ.copy`，避免后续更新污染原对象，并保存到 `environment`。
                    environment = os.environ.copy()
                    # 调用 `str` 把当前值转换为统一字符串表示，并将结果保存到 `environment['CUDA_VISIBLE_DEVICES']`。
                    environment["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
                    # 调用 `subprocess.Popen` 启动独立训练子进程并保留进程句柄，并将结果保存到 `process`。
                    process = subprocess.Popen(
                        worker_command(args, year, seed, physical_gpu),
                        cwd=VIVIT_DIR,
                        env=environment,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    # 把本轮结果追加到 `running`，累积后续需要汇总的记录。
                    running.append(
                        {
                            "process": process,
                            "physical_gpu": physical_gpu,
                            "year": year,
                            "seed": seed,
                            "log": log_path,
                            "handle": handle,
                            "started": time.perf_counter(),
                        }
                    )
                    # 用 `1` 对 `active` 做 `Add` 累积更新。
                    active += 1
                    # 调用 `print` 记录当前流程状态或诊断信息。
                    print(
                        f"launch GPU{physical_gpu}: {year}/seed={seed} "
                        f"pid={process.pid}",
                        flush=True,
                    )
            # 构造当前步骤所需的列表，并保存到 `finished`。
            finished: list[dict[str, Any]] = []
            # 遍历 `running`，并把当前元素绑定到 `item` 逐项处理。
            for item in running:
                # 调用 `item['process'].poll` 查询训练子进程是否已经退出，并将结果保存到 `code`。
                code = item["process"].poll()
                # 检查 `code is None`；根据条件结果选择对应处理分支。
                if code is None:
                    # 跳过当前元素的剩余步骤，继续处理下一项。
                    continue
                # 调用 `item['handle'].close` 释放当前步骤占用的资源。
                item["handle"].close()
                # 把本轮结果追加到 `finished`，累积后续需要汇总的记录。
                finished.append(item)
                # 检查 `code`；根据条件结果选择对应处理分支。
                if code:
                    # 读取 `item['log'].read_text(encoding='utf-8', errors='replace').splitlines()[-100:]` 的当前值，并赋给 `tail`。
                    tail = item["log"].read_text(
                        encoding="utf-8", errors="replace"
                    ).splitlines()[-100:]
                    # 抛出 `RuntimeError(f'任务失败：{item['year']}/seed={item['seed']}\n' + '\n'.join(tail))`，阻止不满足数据契约的流程继续执行。
                    raise RuntimeError(
                        f"任务失败：{item['year']}/seed={item['seed']}\n"
                        + "\n".join(tail)
                    )
                # 调用 `print` 记录当前流程状态或诊断信息。
                print(
                    f"done GPU{item['physical_gpu']}: "
                    f"{item['year']}/seed={item['seed']} "
                    f"{(time.perf_counter()-item['started'])/60:.2f} min",
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
            # 调用 `item['handle'].close` 释放当前步骤占用的资源。
            item["handle"].close()
        # 抛出 `空值`，阻止不满足数据契约的流程继续执行。
        raise

    # 调用 `finalize_run` 汇总运行结果并写出完成标记与最终报告，并将结果保存到 `result`。
    result = finalize_run(time.perf_counter() - started)
    # 调用 `print` 记录当前流程状态或诊断信息。
    print(validate_outputs().to_string(index=False), flush=True)
    # 调用 `print` 记录当前流程状态或诊断信息。
    print(f"实验报告：{write_report()}", flush=True)
    # 将 `result` 作为当前函数结果返回给调用方。
    return result


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

    # 调用 `subparsers.add_parser` 注册一个可独立调用的命令行子命令，并将结果保存到 `prepare`。
    prepare = subparsers.add_parser("prepare-data")
    # 调用 `prepare.add_argument` 向数据准备子命令注册参数及默认口径。
    prepare.add_argument("--overwrite", type=int, choices=(0, 1), default=0)

    # 调用 `prepare.add_argument` 向数据准备子命令注册参数及默认口径。
    prepare.add_argument(
        "--include-industry-sidecar", type=int, choices=(0, 1), default=0
    )
    # 调用 `prepare.add_argument` 向数据准备子命令注册参数及默认口径。
    prepare.add_argument(
        "--include-full-barra-target", type=int, choices=(0, 1), default=0
    )
    # 调用 `prepare.add_argument` 向数据准备子命令注册参数及默认口径。
    prepare.add_argument(
        "--include-dual-tower-target", type=int, choices=(0, 1), default=0
    )
    # 调用 `subparsers.add_parser` 注册一个可独立调用的命令行子命令，并将结果保存到 `smoke`。
    smoke = subparsers.add_parser("smoke")
    # 调用 `add_training_arguments` 向命令行解析器注册当前流程使用的参数。
    add_training_arguments(smoke)
    # 调用 `smoke.add_argument` 向命令行解析器注册当前流程使用的参数。
    smoke.add_argument("--forecast-year", type=int, default=2018)
    # 调用 `smoke.add_argument` 向命令行解析器注册当前流程使用的参数。
    smoke.add_argument("--seed", type=int, default=42)
    # 调用 `smoke.add_argument` 向命令行解析器注册当前流程使用的参数。
    smoke.add_argument("--max-batches", type=int, default=1)
    # 调用 `smoke.set_defaults` 设置当前子命令的默认运行参数。
    smoke.set_defaults(compile=0, workers_per_task=0)

    # 调用 `subparsers.add_parser` 注册一个可独立调用的命令行子命令，并将结果保存到 `worker`。
    worker = subparsers.add_parser("worker")
    # 调用 `add_training_arguments` 向命令行解析器注册当前流程使用的参数。
    add_training_arguments(worker)
    # 调用 `worker.add_argument` 向命令行解析器注册当前流程使用的参数。
    worker.add_argument("--worker-year", type=int, required=True)
    # 调用 `worker.add_argument` 向命令行解析器注册当前流程使用的参数。
    worker.add_argument("--worker-seed", type=int, required=True)
    # 调用 `worker.add_argument` 向命令行解析器注册当前流程使用的参数。
    worker.add_argument("--worker-physical-gpu", type=int, required=True)

    # 调用 `subparsers.add_parser` 注册一个可独立调用的命令行子命令，并将结果保存到 `a2`。
    a2 = subparsers.add_parser("a2")
    # 调用 `add_training_arguments` 向命令行解析器注册当前流程使用的参数。
    add_training_arguments(a2)
    # 调用 `a2.add_argument` 向命令行解析器注册当前流程使用的参数。
    a2.add_argument("--device-ids", default=DEVICE_IDS)
    # 调用 `a2.add_argument` 向命令行解析器注册当前流程使用的参数。
    a2.add_argument("--tasks-per-gpu", type=int, default=TASKS_PER_GPU)
    # 执行当前分布式训练步骤并保持既定数据语义。
    ddp = subparsers.add_parser("ddp")
    # 执行当前分布式训练步骤并保持既定数据语义。
    add_training_arguments(ddp)
    # 执行当前分布式训练步骤并保持既定数据语义。
    ddp.add_argument("--device-ids", default=DEVICE_IDS)

    # 执行当前分布式训练步骤并保持既定数据语义。
    ddp_worker = subparsers.add_parser("ddp-worker")
    # 执行当前分布式训练步骤并保持既定数据语义。
    add_training_arguments(ddp_worker)
    # 执行当前分布式训练步骤并保持既定数据语义。
    ddp_worker.add_argument("--worker-year", type=int, required=True)
    # 执行当前分布式训练步骤并保持既定数据语义。
    ddp_worker.add_argument("--worker-seed", type=int, required=True)
    # 执行当前分布式训练步骤并保持既定数据语义。
    ddp_worker.add_argument("--worker-physical-gpu", type=int, required=True)
    # 执行当前分布式训练步骤并保持既定数据语义。
    ddp_worker.add_argument("--device-ids", required=True)

    # 执行当前分布式训练步骤并保持既定数据语义。
    ddp_smoke = subparsers.add_parser("ddp-smoke")
    # 执行当前分布式训练步骤并保持既定数据语义。
    add_training_arguments(ddp_smoke)
    # 执行当前分布式训练步骤并保持既定数据语义。
    ddp_smoke.add_argument("--forecast-year", type=int, default=2018)
    # 执行当前分布式训练步骤并保持既定数据语义。
    ddp_smoke.add_argument("--seed", type=int, default=42)
    # 执行当前分布式训练步骤并保持既定数据语义。
    ddp_smoke.add_argument("--max-batches", type=int, default=1)
    # 执行当前分布式训练步骤并保持既定数据语义。
    ddp_smoke.add_argument("--device-ids", default=DEVICE_IDS)
    # 执行当前分布式训练步骤并保持既定数据语义。
    ddp_smoke.set_defaults(compile=0, workers_per_task=0)

    # 遍历 `('validate', 'report')`，并把当前元素绑定到 `command` 逐项处理。
    for command in ("validate", "report"):
        # 调用 `subparsers.add_parser` 注册一个可独立调用的命令行子命令，并将结果保存到 `subparser`。
        subparser = subparsers.add_parser(command)
        # 调用 `add_training_arguments` 向命令行解析器注册当前流程使用的参数。
        add_training_arguments(subparser)
    # 将 `parser.parse_args()` 作为当前函数结果返回给调用方。
    return parser.parse_args()


# 定义命令行入口，解析参数并按用户选择调度当前阶段的主流程。
def main() -> None:
    """根据命令行子命令分派数据、训练、验证或评价流程。
    
    输入：
        无显式输入参数。
    输出：
        None。
        """
    # 调用 `parse_args` 解析命令行文本并生成结构化参数，并将结果保存到 `args`。
    args = parse_args()
    # 检查 `args.command == 'prepare-data'`；根据条件结果选择对应处理分支。
    if args.command == "prepare-data":
        # 调用 `style_data.build_style_neutral_store` 根据当前配置和输入构造后续流程所需对象，并将结果保存到 `style_manifest`。
        style_manifest = style_data.build_style_neutral_store(
            overwrite=bool(args.overwrite)
        )
        # 构造键值映射，并保存到 `prepared`，供后续按键查询。
        prepared = {"style_store": style_manifest}
        # 调用 `bool` 统一设备或数据类型，并将结果保存到 `include_dual_target`。
        include_dual_target = bool(args.include_dual_tower_target)
        # 根据 `bool(args.include_full_barra_target) or include_dual_target` 计算条件结果，并保存到 `include_full_target`。
        include_full_target = (
            bool(args.include_full_barra_target) or include_dual_target
        )
        # 检查 `bool(args.include_industry_sidecar) or include_full_target`；根据条件结果选择对应处理分支。
        if bool(args.include_industry_sidecar) or include_full_target:
            # 调用 `style_data.build_industry_sidecar` 根据当前配置和输入构造后续流程所需对象，并将结果保存到 `prepared['industry_sidecar']`。
            prepared["industry_sidecar"] = style_data.build_industry_sidecar(
                overwrite=bool(args.overwrite)
            )
        # 检查 `include_full_target`；根据条件结果选择对应处理分支。
        if include_full_target:
            # 调用 `style_data.build_full_barra_target_sidecar` 根据当前配置和输入构造后续流程所需对象，并将结果保存到 `prepared['full_barra_target']`。
            prepared["full_barra_target"] = (
                style_data.build_full_barra_target_sidecar(
                    overwrite=bool(args.overwrite)
                )
            )
        # 检查 `include_dual_target`；根据条件结果选择对应处理分支。
        if include_dual_target:
            # 调用 `style_data.build_dual_tower_target_sidecar` 根据当前配置和输入构造后续流程所需对象，并将结果保存到 `prepared['dual_tower_target']`。
            prepared["dual_tower_target"] = (
                style_data.build_dual_tower_target_sidecar(
                    overwrite=bool(args.overwrite)
                )
            )
        # 调用 `print` 记录当前流程状态或诊断信息。
        print(
            json.dumps(
                prepared, ensure_ascii=False, indent=2
            )
        )
        # 将 `空值` 作为当前函数结果返回给调用方。
        return

    # 执行当前分布式训练步骤并保持既定数据语义。
    if args.command in {"ddp", "ddp-worker", "ddp-smoke"}:
        # 执行当前分布式训练步骤并保持既定数据语义。
        args.execution_route = "ddp"
        # 执行当前分布式训练步骤并保持既定数据语义。
        args.ddp_world_size = len(distributed.parse_device_ids(args.device_ids))
    # 执行当前分布式训练步骤并保持既定数据语义。
    else:
        # 执行当前分布式训练步骤并保持既定数据语义。
        args.execution_route = "a2"
        # 执行当前分布式训练步骤并保持既定数据语义。
        args.ddp_world_size = 1
    # 调用 `configure_from_args` 整理并返回当前实验的配置对象。
    configure_from_args(args)
    # 检查 `args.command in {'smoke', 'a2'}`；根据条件结果选择对应处理分支。
    if args.command in {"smoke", "a2", "ddp", "ddp-smoke"}:
        # 调用 `style_data.build_style_neutral_store` 根据当前配置和输入构造后续流程所需对象。
        style_data.build_style_neutral_store(overwrite=False)
    # 当前置条件不成立时进入备用分支，保证其余合法情况仍被处理。
    else:
        # 调用 `style_data.require_style_neutral_store` 校验输入、配置或产物是否满足当前阶段的数据契约。
        style_data.require_style_neutral_store()
    # 检查 `industry_sidecar_required()`；根据条件结果选择对应处理分支。
    if industry_sidecar_required():
        # 检查 `args.command in {'smoke', 'a2'}`；根据条件结果选择对应处理分支。
        if args.command in {"smoke", "a2", "ddp", "ddp-smoke"}:
            # 调用 `style_data.build_industry_sidecar` 根据当前配置和输入构造后续流程所需对象。
            style_data.build_industry_sidecar(overwrite=False)
        # 当前置条件不成立时进入备用分支，保证其余合法情况仍被处理。
        else:
            # 调用 `style_data.require_industry_sidecar` 校验输入、配置或产物是否满足当前阶段的数据契约。
            style_data.require_industry_sidecar()
    # 检查 `barra_conditioning_enabled()`；根据条件结果选择对应处理分支。
    if barra_conditioning_enabled() or dual_tower_enabled():
        # 检查 `args.command in {'smoke', 'a2'}`；根据条件结果选择对应处理分支。
        if args.command in {"smoke", "a2", "ddp", "ddp-smoke"}:
            # 调用 `style_data.build_full_barra_target_sidecar` 根据当前配置和输入构造后续流程所需对象。
            style_data.build_full_barra_target_sidecar(overwrite=False)
        # 当前置条件不成立时进入备用分支，保证其余合法情况仍被处理。
        else:
            # 调用 `style_data.require_full_barra_target_sidecar` 校验输入、配置或产物是否满足当前阶段的数据契约。
            style_data.require_full_barra_target_sidecar()

    # 执行当前分布式训练步骤并保持既定数据语义。
    if dual_tower_enabled():
        # 执行当前分布式训练步骤并保持既定数据语义。
        if args.command in {"smoke", "a2", "ddp", "ddp-smoke"}:
            # 执行当前分布式训练步骤并保持既定数据语义。
            style_data.build_dual_tower_target_sidecar(overwrite=False)
        # 执行当前分布式训练步骤并保持既定数据语义。
        else:
            # 执行当前分布式训练步骤并保持既定数据语义。
            style_data.require_dual_tower_target_sidecar()

    # 检查 `args.command == 'smoke'`；根据条件结果选择对应处理分支。
    if args.command == "smoke":
        # 调用 `print` 记录当前流程状态或诊断信息。
        print(
            json.dumps(
                smoke_test(args.forecast_year, args.seed, args.max_batches),
                ensure_ascii=False,
                indent=2,
            )
        )
    # 检查 `args.command == 'worker'`；根据条件结果选择对应处理分支。
    elif args.command == "worker":
        # 调用 `print` 记录当前流程状态或诊断信息。
        print(
            json.dumps(
                train_task(
                    args.worker_year, args.worker_seed, args.worker_physical_gpu
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
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
    # 检查 `args.command == 'validate'`；根据条件结果选择对应处理分支。
    elif args.command == "ddp":
        # 执行当前分布式训练步骤并保持既定数据语义。
        print(run_ddp(args).to_string(index=False))
    # 执行当前分布式训练步骤并保持既定数据语义。
    elif args.command == "validate":
        # 调用 `print` 记录当前流程状态或诊断信息。
        print(validate_outputs().to_string(index=False))
    # 检查 `args.command == 'report'`；根据条件结果选择对应处理分支。
    elif args.command == "report":
        # 调用 `print` 记录当前流程状态或诊断信息。
        print(write_report())


# 检查 `__name__ == '__main__'`；根据条件结果选择对应处理分支。
if __name__ == "__main__":
    # 调用 `main` 启动当前模块的命令行主流程。
    main()
