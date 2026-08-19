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


VIVIT_DIR = Path(__file__).resolve().parent
SCRIPT_PATH = Path(__file__).resolve()
if str(VIVIT_DIR) not in sys.path:
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


RUN_NAME = "sigmoid_daybatch_rankic_style_residual_label_projected_output"
DEVICE_IDS = "0,1,2,3"
TASKS_PER_GPU = 1
WORKERS_PER_TASK = 8
RANK_TEMPERATURE = 0.1
TURNOVER_LAMBDA = 0.1
HUBER_DELTA = 1.0
SMOOTH_L1_BETA = 0.1
HYBRID_WEIGHT = 0.5
LEARNING_RATE = 0.001
MAX_EPOCHS = 100
PATIENCE = 10
USE_TORCH_COMPILE = 1
COMPILE_MODE = "default"
USE_FUSED_ADAM = 1
USE_CUDA_PREFETCH = 1
OVERWRITE = 0
LOSS_NAME = "sigmoid_rank_ic"
VALIDATION_METRIC = "daily_mean_spearman_ic"
TRAIN_BATCH_MODE = "trade_date"
# Execution metadata is deliberately absent from legacy A2 signatures. DDP
# adds it explicitly so checkpoints cannot resume across routes/world sizes.
EXECUTION_ROUTE = "a2"
DDP_WORLD_SIZE = 1
EXECUTION_DEVICE_IDS: tuple[int, ...] = ()
RESIDUALIZE_TARGET = True
PROJECT_MODEL_OUTPUT = True
RAW_BARRA_R2_LAMBDA = 0.0
ADVERSARIAL_BARRA = False
BARRA_CONDITIONING_MODE = "off"
BARRA_OUTPUT_NEUTRALIZATION = False
BARRA_DUAL_TOWER = False
BARRA_DUAL_TOWER_MODE = "real"
BARRA_DUAL_TOWER_MODES = ("real", "alpha_video_only", "shuffled", "zero")
DUAL_TOWER_ALPHA_LOSS_WEIGHT = 1.0
DUAL_TOWER_STYLE_LOSS_WEIGHT = 1.0
DUAL_TOWER_STYLE_HIDDEN_DIM = 64
DUAL_TOWER_STYLE_DROPOUT = 0.10
BARRA_CONDITIONING_MODES = (
    "off", "video_only", "real", "shuffled", "zero"
)
BARRA_GATE_HIDDEN_DIM = 64
BARRA_INDUSTRY_EMBEDDING_DIM = 16
BARRA_GATE_DROPOUT = 0.10
BARRA_GATE_SCALE = 0.50
ADVERSARIAL_GAMMA_MAX = 0.10
ADVERSARIAL_WARMUP_EPOCHS = 2
ADVERSARIAL_RAMP_EPOCHS = 5
ADVERSARIAL_STYLE_LOSS_WEIGHT = 1.0
ADVERSARIAL_INDUSTRY_LOSS_WEIGHT = 1.0
ADVERSARIAL_HEAD_HIDDEN_DIM = 128
ADVERSARIAL_HEAD_DROPOUT = 0.10
FORECAST_YEARS = sequence_data.FORECAST_YEARS
SEEDS = sequence_data.SEEDS
STYLE_COUNT = len(style_data.STYLE_COLUMNS)
TRAINING_SCHEMA_VERSION = 4
INDUSTRY_COUNT = style_data.EXPECTED_CNE5S_INDUSTRIES

PUBLISHED_FACTOR_POLICY = "unprojected_model_score_daily_seed_zscore_mean"
class DegenerateValidationScoreError(FloatingPointError):
    """验证日的原始预测分数退化为常数或近似常数。
    """

    def __init__(
        self,
        *,
        date_id: int,
        rows: int,
        score_std: float,
        total_sum_squares: float,
    ) -> None:
        super().__init__(
            "验证集原始预测分数在单日截面内退化："
            f"date_id={date_id}, rows={rows}, score_std={score_std:.6g}, "
            f"total_sum_squares={total_sum_squares:.6g}"
        )
        self.date_id = int(date_id)
        self.rows = int(rows)
        self.score_std = float(score_std)
        self.total_sum_squares = float(total_sum_squares)


def barra_conditioning_enabled() -> bool:
    """判断当前实验是否启用 Barra 条件输入。"""
    return BARRA_CONDITIONING_MODE != "off"

def strict_barra_output_enabled() -> bool:
    """判断当前实验是否发布严格中性的投影分数。"""
    return bool(BARRA_OUTPUT_NEUTRALIZATION)


def dual_tower_enabled() -> bool:
    """判断当前实验是否启用 Alpha/Style 双塔。"""
    return bool(BARRA_DUAL_TOWER)


def target_policy() -> str:
    """返回当前实验使用的原始、连续风格残差或完整风险残差标签策略。"""
    if dual_tower_enabled():
        return "source_plus_full_barra_residual_plus_fitted_targets"
    if barra_conditioning_enabled():
        return "daily_ten_style_plus_industry_residual_then_zscore_ddof0"
    return (
        "daily_ten_style_ols_residual_then_zscore_ddof0"
        if RESIDUALIZE_TARGET
        else "source_target_zscore"
    )


def output_policy() -> str:
    """返回训练、验证和正式发布分数的口径策略。"""
    if dual_tower_enabled():
        return "dual_tower_unprojected_alpha_score"
    if strict_barra_output_enabled():
        return "daily_ten_style_plus_industry_projection_then_zscore_ddof0"
    return (
        "daily_ten_style_qr_residual_then_zscore_ddof0"
        if PROJECT_MODEL_OUTPUT
        else "model_raw_score"
    )


def raw_barra_r2_enabled() -> bool:
    """判断是否启用 raw-score 联合 Barra R² 惩罚。"""
    return RAW_BARRA_R2_LAMBDA > 0.0


def adversarial_barra_enabled() -> bool:
    """判断是否启用对抗式 Barra 不变表征。"""
    return bool(ADVERSARIAL_BARRA)


def industry_sidecar_required() -> bool:
    """判断当前实验是否必须加载行业 sidecar。"""
    return (
        raw_barra_r2_enabled()
        or adversarial_barra_enabled()
        or barra_conditioning_enabled()
        or dual_tower_enabled()
    )


def factor_role() -> str:
    """说明某个预测列在当前实验中的 Alpha、Style 或总收益角色。"""
    if dual_tower_enabled():
        return f"barra_conditional_dual_tower_alpha_{BARRA_DUAL_TOWER_MODE}"
    if strict_barra_output_enabled():
        return f"barra_conditioned_strict_neutral_{BARRA_CONDITIONING_MODE}"
    if barra_conditioning_enabled():
        return f"barra_conditioned_multiplicative_gate_{BARRA_CONDITIONING_MODE}"
    if adversarial_barra_enabled():
        return "adversarial_barra_invariant_representation_model"
    if raw_barra_r2_enabled():
        return "raw_score_barra_r_squared_regularized_model"
    if RESIDUALIZE_TARGET and PROJECT_MODEL_OUTPUT:
        return "style_residual_label_projected_output_model"
    if RESIDUALIZE_TARGET:
        return "style_residual_label_model"
    if PROJECT_MODEL_OUTPUT:
        return "style_projected_output_model"
    return "style_backend_raw_model"


def configure_from_args(args: argparse.Namespace) -> None:
    """将 CLI 配置安装为 worker 进程内只读实验配置。"""

    global RUN_NAME, DEVICE_IDS, TASKS_PER_GPU, WORKERS_PER_TASK
    global EXECUTION_ROUTE, DDP_WORLD_SIZE, EXECUTION_DEVICE_IDS
    global RANK_TEMPERATURE, TURNOVER_LAMBDA, HUBER_DELTA
    global SMOOTH_L1_BETA, HYBRID_WEIGHT, LEARNING_RATE
    global MAX_EPOCHS, PATIENCE, USE_TORCH_COMPILE, COMPILE_MODE
    global USE_FUSED_ADAM, USE_CUDA_PREFETCH, OVERWRITE, LOSS_NAME
    global VALIDATION_METRIC, TRAIN_BATCH_MODE
    global RESIDUALIZE_TARGET, PROJECT_MODEL_OUTPUT, RAW_BARRA_R2_LAMBDA
    global ADVERSARIAL_BARRA, ADVERSARIAL_GAMMA_MAX
    global BARRA_CONDITIONING_MODE, BARRA_GATE_HIDDEN_DIM
    global BARRA_INDUSTRY_EMBEDDING_DIM, BARRA_GATE_DROPOUT
    global BARRA_GATE_SCALE, BARRA_OUTPUT_NEUTRALIZATION
    global BARRA_DUAL_TOWER, BARRA_DUAL_TOWER_MODE
    global DUAL_TOWER_ALPHA_LOSS_WEIGHT, DUAL_TOWER_STYLE_LOSS_WEIGHT
    global DUAL_TOWER_STYLE_HIDDEN_DIM, DUAL_TOWER_STYLE_DROPOUT
    global ADVERSARIAL_WARMUP_EPOCHS, ADVERSARIAL_RAMP_EPOCHS
    global ADVERSARIAL_STYLE_LOSS_WEIGHT, ADVERSARIAL_INDUSTRY_LOSS_WEIGHT
    global ADVERSARIAL_HEAD_HIDDEN_DIM, ADVERSARIAL_HEAD_DROPOUT

    RUN_NAME = args.run_name
    DEVICE_IDS = getattr(args, "device_ids", DEVICE_IDS)
    EXECUTION_ROUTE = str(getattr(args, "execution_route", "a2"))
    DDP_WORLD_SIZE = int(getattr(args, "ddp_world_size", 1))
    EXECUTION_DEVICE_IDS = distributed.parse_device_ids(DEVICE_IDS)
    TASKS_PER_GPU = int(getattr(args, "tasks_per_gpu", TASKS_PER_GPU))
    WORKERS_PER_TASK = int(args.workers_per_task)
    RANK_TEMPERATURE = float(args.rank_temperature)
    TURNOVER_LAMBDA = float(args.turnover_lambda)
    HUBER_DELTA = float(args.huber_delta)
    SMOOTH_L1_BETA = float(args.smooth_l1_beta)
    HYBRID_WEIGHT = float(args.hybrid_weight)
    LEARNING_RATE = float(args.learning_rate)
    MAX_EPOCHS = int(args.max_epochs)
    PATIENCE = int(args.patience)
    USE_TORCH_COMPILE = int(args.compile)
    COMPILE_MODE = args.compile_mode
    USE_FUSED_ADAM = int(args.fused_adam)
    USE_CUDA_PREFETCH = int(args.cuda_prefetch)
    OVERWRITE = int(args.overwrite)
    LOSS_NAME = args.loss_name
    VALIDATION_METRIC = args.validation_metric
    TRAIN_BATCH_MODE = args.train_batch_mode
    RESIDUALIZE_TARGET = bool(args.residualize_target)
    PROJECT_MODEL_OUTPUT = bool(args.project_model_output)
    RAW_BARRA_R2_LAMBDA = float(args.raw_barra_r2_lambda)
    ADVERSARIAL_BARRA = bool(args.adversarial_barra)
    BARRA_CONDITIONING_MODE = str(args.barra_conditioning_mode)
    BARRA_OUTPUT_NEUTRALIZATION = bool(
        args.barra_output_neutralization
    )
    BARRA_DUAL_TOWER = bool(args.barra_dual_tower)
    BARRA_DUAL_TOWER_MODE = str(args.barra_dual_tower_mode)
    DUAL_TOWER_ALPHA_LOSS_WEIGHT = float(
        args.dual_tower_alpha_loss_weight
    )
    DUAL_TOWER_STYLE_LOSS_WEIGHT = float(
        args.dual_tower_style_loss_weight
    )
    DUAL_TOWER_STYLE_HIDDEN_DIM = int(args.dual_tower_style_hidden_dim)
    DUAL_TOWER_STYLE_DROPOUT = float(args.dual_tower_style_dropout)
    BARRA_GATE_HIDDEN_DIM = int(args.barra_gate_hidden_dim)
    BARRA_INDUSTRY_EMBEDDING_DIM = int(
        args.barra_industry_embedding_dim
    )
    BARRA_GATE_DROPOUT = float(args.barra_gate_dropout)
    BARRA_GATE_SCALE = float(args.barra_gate_scale)
    ADVERSARIAL_GAMMA_MAX = float(args.adversarial_gamma_max)
    ADVERSARIAL_WARMUP_EPOCHS = int(args.adversarial_warmup_epochs)
    ADVERSARIAL_RAMP_EPOCHS = int(args.adversarial_ramp_epochs)
    ADVERSARIAL_STYLE_LOSS_WEIGHT = float(
        args.adversarial_style_loss_weight
    )
    ADVERSARIAL_INDUSTRY_LOSS_WEIGHT = float(
        args.adversarial_industry_loss_weight
    )
    ADVERSARIAL_HEAD_HIDDEN_DIM = int(args.adversarial_head_hidden_dim)
    ADVERSARIAL_HEAD_DROPOUT = float(args.adversarial_head_dropout)

    config = training_config()
    base.validate_training_config(config, COMPILE_MODE)
    if TRAIN_BATCH_MODE != "trade_date":
        raise ValueError("风格训练或 Barra R² 正则必须使用 trade_date batch")
    if LOSS_NAME == base.TEMPORAL_TURNOVER_LOSS:
        raise ValueError("风格训练后端暂不支持 sigmoid_rank_ic_turnover")
    if RAW_BARRA_R2_LAMBDA < 0:
        raise ValueError("raw_barra_r2_lambda 必须 >= 0")
    if raw_barra_r2_enabled() and not dual_tower_enabled() and (
        RESIDUALIZE_TARGET or PROJECT_MODEL_OUTPUT
    ):
        raise ValueError(
            "No.2 要求 RESIDUALIZE_TARGET=False 且 "
            "PROJECT_MODEL_OUTPUT=False"
        )
    if raw_barra_r2_enabled() and not dual_tower_enabled() and LOSS_NAME != "sigmoid_rank_ic":
        raise ValueError("No.2 当前只支持 sigmoid_rank_ic")
    if (
        raw_barra_r2_enabled() and not dual_tower_enabled()
        and VALIDATION_METRIC != "daily_mean_spearman_ic"
    ):
        raise ValueError(
            "No.2 的早停指标必须为 daily_mean_spearman_ic"
        )
    if adversarial_barra_enabled():
        if RESIDUALIZE_TARGET or PROJECT_MODEL_OUTPUT:
            raise ValueError(
                "No.3 要求 RESIDUALIZE_TARGET=False 且 "
                "PROJECT_MODEL_OUTPUT=False"
            )
        if raw_barra_r2_enabled():
            raise ValueError("No.3 首轮纯对抗实验禁止叠加 raw-score R²")
        if LOSS_NAME != "sigmoid_rank_ic":
            raise ValueError("No.3 当前只支持 sigmoid_rank_ic")
        if VALIDATION_METRIC != "daily_mean_spearman_ic":
            raise ValueError(
                "No.3 的早停指标必须为 daily_mean_spearman_ic"
            )
        if (
            ADVERSARIAL_GAMMA_MAX <= 0
            or ADVERSARIAL_WARMUP_EPOCHS < 0
            or ADVERSARIAL_RAMP_EPOCHS < 1
            or ADVERSARIAL_STYLE_LOSS_WEIGHT <= 0
            or ADVERSARIAL_INDUSTRY_LOSS_WEIGHT <= 0
            or ADVERSARIAL_HEAD_HIDDEN_DIM < 1
            or not 0 <= ADVERSARIAL_HEAD_DROPOUT < 1
        ):
            raise ValueError("No.3 对抗训练超参数非法")
    if BARRA_CONDITIONING_MODE not in BARRA_CONDITIONING_MODES:
        raise ValueError(
            f"barra_conditioning_mode 必须属于 {BARRA_CONDITIONING_MODES}"
        )
    if BARRA_DUAL_TOWER_MODE not in BARRA_DUAL_TOWER_MODES:
        raise ValueError(
            f"barra_dual_tower_mode 必须属于 {BARRA_DUAL_TOWER_MODES}"
        )
    if dual_tower_enabled():
        if RESIDUALIZE_TARGET or PROJECT_MODEL_OUTPUT:
            raise ValueError("No.6 要求关闭 Neutral 标签与输出投影")
        if adversarial_barra_enabled() or barra_conditioning_enabled():
            raise ValueError("No.6 不能与 No.3 或 No.4 独立模式同时开启")
        if strict_barra_output_enabled():
            raise ValueError("No.6 正式输出为未投影 Alpha，禁止 No.5 严格投影")
        if LOSS_NAME != "sigmoid_rank_ic":
            raise ValueError("No.6 只允许 sigmoid_rank_ic")
        if VALIDATION_METRIC != "daily_mean_spearman_ic":
            raise ValueError("No.6 早停只允许 daily_mean_spearman_ic")
        if (
            DUAL_TOWER_ALPHA_LOSS_WEIGHT <= 0
            or DUAL_TOWER_STYLE_LOSS_WEIGHT <= 0
            or DUAL_TOWER_STYLE_HIDDEN_DIM < 1
            or not 0 <= DUAL_TOWER_STYLE_DROPOUT < 1
        ):
            raise ValueError("No.6 双塔损失权重或 Style 塔超参数非法")
    if strict_barra_output_enabled() and not barra_conditioning_enabled():
        raise ValueError(
            "严格 Barra 输出中性化要求开启一种条件门控模式"
        )
    if barra_conditioning_enabled():
        if RESIDUALIZE_TARGET or PROJECT_MODEL_OUTPUT:
            raise ValueError(
                "No.4 要求 RESIDUALIZE_TARGET=False 且 "
                "PROJECT_MODEL_OUTPUT=False"
            )
        if adversarial_barra_enabled():
            raise ValueError("No.4 不能与 No.3 对抗训练同时开启")
        if LOSS_NAME != "sigmoid_rank_ic":
            raise ValueError("No.4 当前只支持 sigmoid_rank_ic")
        if strict_barra_output_enabled() and raw_barra_r2_enabled():
            raise ValueError(
                "严格 Barra 输出中性化已代数正交，RAW_BARRA_R2_LAMBDA 必须为0"
            )
        if VALIDATION_METRIC != "daily_mean_spearman_ic":
            raise ValueError(
                "No.4 的早停指标必须为 daily_mean_spearman_ic"
            )
        if (
            BARRA_GATE_HIDDEN_DIM < 1
            or BARRA_INDUSTRY_EMBEDDING_DIM < 1
            or not 0 <= BARRA_GATE_DROPOUT < 1
            or not 0 < BARRA_GATE_SCALE < 1
        ):
            raise ValueError("No.4 条件门控超参数非法")
    if TASKS_PER_GPU < 1 or WORKERS_PER_TASK < 0:
        raise ValueError("tasks_per_gpu 必须为正整数且 workers_per_task >= 0")


def build_training_model(
    device: torch.device, *, use_compile: bool
) -> torch.nn.Module:
    """按显式方案构造模型；新增结构来自短实验模块。"""

    if dual_tower_enabled():
        from stage04_no4_gate import (
            BarraConditionalViViT as CoreConditionalViViT,
        )
        from stage04_no6_dual_tower import (
            BarraDualTowerViViT as CoreDualTowerViViT,
        )

        alpha_mode = (
            "video_only"
            if BARRA_DUAL_TOWER_MODE == "alpha_video_only"
            else "real"
        )
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
        model: torch.nn.Module = CoreDualTowerViViT(
            alpha_tower,
            style_count=STYLE_COUNT,
            industry_count=INDUSTRY_COUNT,
            industry_embedding_dim=BARRA_INDUSTRY_EMBEDDING_DIM,
            style_hidden_dim=DUAL_TOWER_STYLE_HIDDEN_DIM,
            style_dropout=DUAL_TOWER_STYLE_DROPOUT,
        ).to(device)
    elif barra_conditioning_enabled():
        from stage04_no4_gate import (
            BarraConditionalViViT as CoreConditionalViViT,
        )

        internal_mode = (
            "video_only"
            if BARRA_CONDITIONING_MODE == "video_only"
            else "real"
        )
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
    elif adversarial_barra_enabled():
        from stage04_no3_adversarial import (
            AdversarialBarraViViT as CoreAdversarialViViT,
        )

        model = CoreAdversarialViViT(
            base.OrderFlowViViT(base.MODEL_CONFIG),
            embedding_dim=base.MODEL_CONFIG.embedding_dim,
            style_count=STYLE_COUNT,
            industry_count=INDUSTRY_COUNT,
            hidden_dim=ADVERSARIAL_HEAD_HIDDEN_DIM,
            dropout=ADVERSARIAL_HEAD_DROPOUT,
        ).to(device)
    else:
        return base.build_model(
            device, use_compile, COMPILE_MODE, TRAIN_BATCH_MODE
        )
    if use_compile:
        if device.type != "cuda":
            raise RuntimeError("torch.compile 训练要求 CUDA")
        model = torch.compile(model, mode=COMPILE_MODE, dynamic=True)  # 编译模型计算图
    return model


def adversarial_gamma(epoch: int) -> float:
    """把当前训练配置传给 No.3 的余弦梯度反转调度器。"""
    return cosine_adversarial_gamma(
        epoch,
        gamma_max=ADVERSARIAL_GAMMA_MAX,
        warmup_epochs=ADVERSARIAL_WARMUP_EPOCHS,
        ramp_epochs=ADVERSARIAL_RAMP_EPOCHS,
    )


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
    """按实验模式执行基线、对抗、门控或双塔模型前向传播。"""
    if dual_tower_enabled():
        if (
            styles is None
            or style_valid is None
            or industry_codes is None
            or sample_indices is None
        ):
            raise ValueError("No.6 前向缺少 Barra 条件输入")
        if conditioning_prepared:
            alpha_inputs = style_inputs = (styles, style_valid, industry_codes)
        else:
            alpha_inputs, style_inputs = prepare_dual_tower_inputs(
                styles, style_valid, industry_codes, sample_indices,
                mode=BARRA_DUAL_TOWER_MODE,
            )
        alpha, style, total, gate, representation = model(  # 执行当前 batch 前向传播
            videos, frame_mask, *alpha_inputs, *style_inputs
        )
        return alpha, None, None, representation, gate, style, total
    if barra_conditioning_enabled():
        if (
            styles is None
            or style_valid is None
            or industry_codes is None
            or sample_indices is None
        ):
            raise ValueError("No.4 前向缺少 Barra 条件输入")
        if conditioning_prepared:
            conditional = (styles, style_valid, industry_codes)
        else:
            conditional = prepare_barra_conditioning_inputs(
                styles, style_valid, industry_codes, sample_indices,
                mode=BARRA_CONDITIONING_MODE,
            )
        raw_score, gate, representation = model(  # 执行当前 batch 前向传播
            videos, frame_mask, *conditional
        )
        return raw_score, None, None, representation, gate, None, None
    if adversarial_barra_enabled():
        coefficient = torch.tensor(
            gamma, dtype=torch.float32, device=videos.device
        )
        raw, style, industry, representation = model(  # 执行当前 batch 前向传播
            videos, frame_mask, coefficient
        )
        return raw, style, industry, representation, None, None, None
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
class StyleNeutralBatchCollator:
    """读取视频、原标签、风格残差标签和十风格暴露。
    """

    def __init__(self) -> None:
        self.video_collator = sequence_data.VideoBatchCollator()  # 整理并搬运 batch
        self.residual_targets = None
        self.full_barra_targets = None
        self.dual_tower_fitted_targets = None
        self.style_exposures = None
        self.style_valid = None
        self.industry_codes = None

    def _ensure_open(self) -> None:
        """在当前 worker 内惰性打开风格、行业和派生标签 memmap。"""
        if self.residual_targets is None:
            store = style_data.DEFAULT_STORE_DIR
            self.residual_targets = np.load(
                store / style_data.RESIDUAL_TARGETS_FILE, mmap_mode="r"
            )
            self.style_exposures = np.load(
                store / style_data.STYLE_EXPOSURES_FILE, mmap_mode="r"
            )
            self.style_valid = np.load(
                store / style_data.STYLE_VALID_FILE, mmap_mode="r"
            )
            if barra_conditioning_enabled() or dual_tower_enabled():
                style_data.require_full_barra_target_sidecar()
                self.full_barra_targets = np.load(
                    style_data.DEFAULT_FULL_BARRA_TARGET_STORE_DIR
                    / style_data.FULL_BARRA_TARGETS_FILE,
                    mmap_mode="r",
                )
            if dual_tower_enabled():
                style_data.require_dual_tower_target_sidecar()
                self.dual_tower_fitted_targets = np.load(
                    style_data.DEFAULT_DUAL_TOWER_TARGET_STORE_DIR
                    / style_data.DUAL_TOWER_FITTED_TARGETS_FILE,
                    mmap_mode="r",
                )
            if industry_sidecar_required():
                style_data.require_industry_sidecar()
                self.industry_codes = np.load(
                    style_data.DEFAULT_INDUSTRY_STORE_DIR
                    / style_data.INDUSTRY_CODES_FILE, mmap_mode="r"
                )

    def __call__(self, sample_indices: list[int]) -> dict[str, torch.Tensor]:
        """按样本行号组装视频、Barra 条件、派生标签和真实审计暴露。"""
        self._ensure_open()
        batch = self.video_collator(sample_indices)  # 整理并搬运 batch
        indices = batch["sample_indices"].numpy().astype(np.int64, copy=False)
        batch["original_targets"] = batch["targets"]
        batch["residual_targets"] = torch.from_numpy(
            np.array(self.residual_targets[indices], dtype=np.float32, copy=True)
        )
        if RESIDUALIZE_TARGET:
            batch["targets"] = batch["residual_targets"]
        if barra_conditioning_enabled() or dual_tower_enabled():
            batch["targets"] = torch.from_numpy(
                np.array(
                    self.full_barra_targets[indices],
                    dtype=np.float32,
                    copy=True,
                )
            )
        if dual_tower_enabled():
            batch["dual_tower_fitted_targets"] = torch.from_numpy(
                np.array(
                    self.dual_tower_fitted_targets[indices],
                    dtype=np.float32,
                    copy=True,
                )
            )
        else:
            batch["dual_tower_fitted_targets"] = torch.zeros_like(
                batch["targets"]
            )
        batch["style_exposures"] = torch.from_numpy(
            np.array(self.style_exposures[indices], dtype=np.float32, copy=True)
        )
        batch["style_valid"] = torch.from_numpy(
            np.array(self.style_valid[indices], dtype=np.bool_, copy=True)
        )
        if industry_sidecar_required():
            batch["industry_codes"] = torch.from_numpy(
                np.array(self.industry_codes[indices], dtype=np.int16, copy=True)
            )
        else:
            batch["industry_codes"] = torch.zeros(
                len(indices), dtype=torch.int16
            )
        return batch


def make_style_loaders(
    forecast_year: int,
    seed: int,
    workers: int,
    pin_memory: bool,
    distributed_context: distributed.DistributedContext | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    """训练、验证和预测全部按完整交易日组织 batch。"""

    split = sequence_data.load_sequence_split(forecast_year)
    date_ids = np.load(sequence_data.DATE_IDS_PATH, mmap_mode="r")
    generator = torch.Generator().manual_seed(seed)  # 固定 DataLoader 随机源
    style_common: dict[str, Any] = {
        "num_workers": workers,
        "pin_memory": pin_memory,
        "collate_fn": StyleNeutralBatchCollator(),
    }
    if workers > 0:
        style_common.update({"persistent_workers": True, "prefetch_factor": 1})

    def style_loader(indices: np.ndarray, *, shuffle: bool) -> DataLoader:
        """处理连续风格数据、有效掩码或相应诊断。"""
        sampler_class = (  # 按完整交易日截面组 batch
            distributed.DistributedTradeDateBatchSampler
            if distributed_context is not None else TradeDateBatchSampler
        )
        distributed_kwargs = (
            {"rank": distributed_context.rank, "world_size": distributed_context.world_size}
            if distributed_context is not None else {}
        )
        sampler = sampler_class(
            indices, date_ids, seed=seed, shuffle=shuffle, **distributed_kwargs
        )
        return DataLoader(
            sequence_data.SampleIndexDataset(indices),
            batch_sampler=sampler,
            generator=generator,
            **style_common,
        )

    def raw_prediction_loader(indices: np.ndarray) -> DataLoader:
        """处理预测文件、预测列或对应的样本数量审计。"""
        raw_common: dict[str, Any] = {
            "num_workers": workers,
            "pin_memory": pin_memory,
            "collate_fn": sequence_data.VideoBatchCollator(),
        }
        if workers > 0:
            raw_common.update(
                {"persistent_workers": True, "prefetch_factor": 1}
            )
        sampler_class = (  # 按完整交易日截面组 batch
            distributed.DistributedTradeDateBatchSampler
            if distributed_context is not None else TradeDateBatchSampler
        )
        distributed_kwargs = (
            {"rank": distributed_context.rank, "world_size": distributed_context.world_size}
            if distributed_context is not None else {}
        )
        sampler = sampler_class(
            indices, date_ids, seed=seed, shuffle=False, **distributed_kwargs
        )
        return DataLoader(
            sequence_data.SampleIndexDataset(indices),
            batch_sampler=sampler,
            generator=generator,
            **raw_common,
        )

    train = style_loader(split["train_indices"], shuffle=True)
    validation = style_loader(split["validation_indices"], shuffle=False)
    prediction = None
    if len(split["prediction_indices"]):
        prediction = (
            raw_prediction_loader(split["prediction_indices"])
            if (
                not barra_conditioning_enabled()
                and not dual_tower_enabled()
                and (raw_barra_r2_enabled() or adversarial_barra_enabled())
            )
            else style_loader(split["prediction_indices"], shuffle=False)
        )
    return train, validation, prediction


def training_industry_class_weights(train_indices: np.ndarray) -> torch.Tensor:
    """由当前滚动训练区间生成32行业逆平方根频数权重。"""

    style_data.require_industry_sidecar()
    codes = np.load(
        style_data.DEFAULT_INDUSTRY_STORE_DIR / style_data.INDUSTRY_CODES_FILE,
        mmap_mode="r",
    )
    selected = np.asarray(codes[train_indices], dtype=np.int64)
    known = selected[selected > 0] - 1
    counts = np.bincount(known, minlength=INDUSTRY_COUNT).astype(np.float64)
    if len(counts) != INDUSTRY_COUNT or (counts <= 0).any():
        raise ValueError("滚动训练区间未覆盖全部32个真实行业")
    weights = np.reciprocal(np.sqrt(counts))  # 按截面样本数平方根加权
    weights /= weights.mean()
    return torch.from_numpy(weights.astype(np.float32))


class StyleCudaPrefetcher:
    """将视频、双标签和风格矩阵一起异步搬到 GPU。
    """

    def __init__(self, loader: DataLoader, device: torch.device, enabled: bool) -> None:
        self.loader = loader
        self.device = device
        self.enabled = enabled and device.type == "cuda"

    def _move(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, ...]:
        """把风格训练批次的所有张量非阻塞搬到 CUDA。"""
        non_blocking = self.device.type == "cuda"
        names = (
            "videos", "frame_mask", "targets", "original_targets",
            "residual_targets", "dual_tower_fitted_targets",
            "style_exposures", "style_valid",
            "industry_codes",
        )
        return tuple(
            batch[name].to(self.device, non_blocking=non_blocking)
            for name in names
        )

    def __iter__(self) -> Iterator[tuple[Any, ...]]:
        """按既定顺序迭代 StyleCudaPrefetcher 中的训练批次。"""
        if not self.enabled:
            for batch in self.loader:
                # 对 `(yield (batch, *self._move(batch)))` 求值，以触发当前步骤需要的计算。
                yield (batch, *self._move(batch))
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
                tensors = self._move(batch)
            next_item = (batch, *tensors)

        preload()  # 异步预取下一批数据
        while next_item is not None:
            torch.cuda.current_stream(self.device).wait_stream(stream)  # 等待预取流完成
            current = next_item
            for tensor in current[1:]:
                tensor.record_stream(torch.cuda.current_stream(self.device))  # 绑定当前 CUDA stream 生命周期
            preload()  # 异步预取下一批数据
            # 对 `(yield current)` 求值，以触发当前步骤需要的计算。
            yield current


def training_config() -> base.TrainingConfig:
    """汇总当前风格中性实验的结构、标签、损失和早停配置。"""
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


def training_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """现有预测损失；sigmoid soft rank 对预测端保持反向传播。"""

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


def rank_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """兼容旧测试和调用方的训练损失别名。"""

    return training_loss(predictions, targets)


def stable_sigmoid_rank_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """与现有 Sigmoid SoftRankIC 等价，并稳定支持常数初始分数。"""

    prediction_ranks = sigmoid_soft_rank(
        predictions, RANK_TEMPERATURE
    )
    with torch.no_grad():
        target_ranks = sigmoid_soft_rank(
            targets.detach(), RANK_TEMPERATURE
        )
    centered_prediction = prediction_ranks - prediction_ranks.mean()
    centered_target = target_ranks - target_ranks.mean()
    variance_product = (
        centered_prediction.square().sum()
        * centered_target.square().sum()
    )
    denominator = torch.sqrt(variance_product.clamp_min(1e-8))  # 计算截面标准差
    correlation = (
        centered_prediction.mul(centered_target).sum() / denominator
    )
    return -correlation


def masked_style_regression_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """仅在有效暴露位置计算十风格对抗回归损失。"""
    if predictions.shape != targets.shape or valid.shape != targets.shape:
        raise ValueError("风格对抗头、标签或有效掩码形状不一致")
    mask = valid.bool()
    if not mask.any():
        raise ValueError("当前日批次没有有效风格监督")
    values = (predictions.float() - targets.float()).square()[mask]
    if not torch.isfinite(values).all():
        raise FloatingPointError("风格对抗损失含 NaN/Inf")
    return values.mean()


def normalized_industry_classification_loss(
    logits: torch.Tensor,
    industry_codes: torch.Tensor,
    class_weights: torch.Tensor,
) -> torch.Tensor:
    """计算忽略 UNKNOWN 且按行业加权归一化的分类损失。"""
    if logits.ndim != 2 or logits.shape[1] != INDUSTRY_COUNT:
        raise ValueError("行业对抗头必须输出 [N,32]")
    codes = industry_codes.long().reshape(-1)
    if len(codes) != len(logits):
        raise ValueError("行业标签与对抗头行数不一致")
    known = codes > 0
    if not known.any():
        raise ValueError("当前日批次没有已知行业监督")
    targets = codes[known] - 1
    if int(targets.min()) < 0 or int(targets.max()) >= INDUSTRY_COUNT:
        raise ValueError("行业标签超出1至32的有效范围")
    weights = class_weights.to(device=logits.device, dtype=torch.float32)
    if weights.shape != (INDUSTRY_COUNT,) or not torch.isfinite(weights).all():
        raise ValueError("行业类别权重形状或数值异常")
    loss = torch.nn.functional.cross_entropy(
        logits.float()[known], targets, weight=weights
    ) / math.log(INDUSTRY_COUNT)
    if not torch.isfinite(loss):
        raise FloatingPointError("行业对抗损失含 NaN/Inf")
    return loss


def adversarial_training_loss(
    style_predictions: torch.Tensor | None,
    industry_logits: torch.Tensor | None,
    styles: torch.Tensor,
    style_valid: torch.Tensor,
    industry_codes: torch.Tensor,
    class_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """组合收益排序、连续风格对抗和行业对抗损失。"""
    zero = torch.zeros((), dtype=torch.float32, device=styles.device)
    if not adversarial_barra_enabled():
        return zero, zero
    if style_predictions is None or industry_logits is None:
        raise ValueError("No.3 缺少风格或行业对抗头输出")
    return (
        masked_style_regression_loss(
            style_predictions, styles, style_valid
        ),
        normalized_industry_classification_loss(
            industry_logits, industry_codes, class_weights
        ),
    )


def _joint_r_squared_value(
    predictions: torch.Tensor,
    styles: torch.Tensor,
    industries: torch.Tensor,
) -> torch.Tensor:
    """把 Barra R² 公共接口收窄为标量，供方案损失组合。"""

    from stage02_barra_algebra import joint_barra_r_squared

    value, _ = joint_barra_r_squared(  # 按股票与交易日业务键对齐
        predictions, styles, industries
    )
    return value


def combined_training_loss(
    raw_predictions: torch.Tensor,
    effective: torch.Tensor,
    targets: torch.Tensor,
    style_exposures: torch.Tensor,
    industry_codes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回总损失、预测损失和 raw-score 联合 Barra R²。"""

    predictive_loss = training_loss(effective, targets)  # 按实验口径计算可微损失
    if not raw_barra_r2_enabled():
        barra_r_squared = torch.zeros(
            (), dtype=torch.float32, device=raw_predictions.device
        )
        return predictive_loss, predictive_loss, barra_r_squared

    from stage04_no2_r2_penalty import (
        r_squared_regularized_loss,
    )

    components = r_squared_regularized_loss(  # 按实验口径计算可微损失
        raw_predictions,
        targets,
        style_exposures,
        industry_codes,
        rank_loss=lambda _score, _target: predictive_loss,
        barra_r_squared=_joint_r_squared_value,
        penalty_weight=RAW_BARRA_R2_LAMBDA,
    )
    return (
        components["loss"],
        components["predictive_loss"],
        components["barra_r_squared"],
    )


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
    """No.6 四项损失；风险惩罚只施加于 Alpha 塔。"""

    if not dual_tower_enabled():
        raise RuntimeError("仅 No.6 可以调用双塔损失")
    from stage04_no6_dual_tower import (
        dual_tower_training_loss as compute_dual_tower_loss,
    )

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


class FullBarraOutputNeutralizationLayer(torch.nn.Module):
    """无参数的每日等权完整 Barra 可微中性化层。
    """

    def forward(
        self,
        raw_predictions: torch.Tensor,
        style_exposures: torch.Tensor,
        industry_codes: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """执行 FullBarraOutputNeutralizationLayer 的前向计算并返回约定张量。"""
        return style_data.project_out_full_barra_torch(
            raw_predictions,
            style_exposures,
            industry_codes,
            standardize=True,
        )


FULL_BARRA_OUTPUT_NEUTRALIZER = FullBarraOutputNeutralizationLayer()  # 按日投影 Barra 暴露


def effective_predictions(
    raw_predictions: torch.Tensor,
    style_exposures: torch.Tensor,
    industry_codes: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
    """按实验开关返回损失实际使用及正式发布的分数口径。"""
    if strict_barra_output_enabled():
        if industry_codes is None:
            raise ValueError("严格 Barra 输出中性化缺少行业代码")
        output, diagnostics = FULL_BARRA_OUTPUT_NEUTRALIZER(  # 按日投影 Barra 暴露
            raw_predictions,
            style_exposures,
            industry_codes,
        )
        if (
            float(diagnostics["max_abs_style_pearson"]) > 5e-5
            or float(diagnostics["max_abs_industry_mean"]) > 5e-5
        ):
            raise FloatingPointError(
                "严格 Barra 输出未达到十风格或行业数值正交"
            )
        return output, diagnostics
    if not PROJECT_MODEL_OUTPUT:
        return raw_predictions.float().reshape(-1), None
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
def prepare_distributed_conditioning(
    styles: torch.Tensor,
    style_valid: torch.Tensor,
    industry_codes: torch.Tensor,
    sample_indices: torch.Tensor,
    context: distributed.DistributedContext | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """执行 prepare_distributed_conditioning 对应的分布式训练处理。"""

    shuffled = (
        dual_tower_enabled() and BARRA_DUAL_TOWER_MODE == "shuffled"
    ) or (
        barra_conditioning_enabled() and BARRA_CONDITIONING_MODE == "shuffled"
    )
    if context is None or not shuffled:
        return styles, style_valid, industry_codes, False
    local_indices = sample_indices.to(styles.device)
    global_indices = distributed.all_gather_tensor(local_indices)
    global_styles = distributed.all_gather_tensor(styles)
    global_valid = distributed.all_gather_tensor(style_valid)
    global_industry = distributed.all_gather_tensor(industry_codes)
    order = torch.argsort(global_indices)
    global_indices = global_indices[order]
    global_styles = global_styles[order]
    global_valid = global_valid[order]
    global_industry = global_industry[order]
    mode = (
        BARRA_DUAL_TOWER_MODE
        if dual_tower_enabled() else BARRA_CONDITIONING_MODE
    )
    transformed = prepare_barra_conditioning_inputs(
        global_styles,
        global_valid,
        global_industry,
        global_indices,
        mode=mode,
    )
    positions = torch.searchsorted(global_indices, local_indices)
    if not torch.equal(global_indices[positions], local_indices):
        raise AssertionError("DDP Barra 条件输入无法按 sample_index 对齐")
    return (
        transformed[0][positions],
        transformed[1][positions],
        transformed[2][positions],
        True,
    )


def _gather_optional_cross_section(
    value: torch.Tensor | None,
    order: torch.Tensor,
    context: distributed.DistributedContext | None,
    *,
    differentiable: bool,
) -> torch.Tensor | None:
    """执行 _gather_optional_cross_section 对应的分布式训练处理。"""
    if value is None:
        return None
    if context is None:
        return value
    gathered = (
        distributed.differentiable_all_gather(value)
        if differentiable else distributed.all_gather_tensor(value)
    )
    return gathered[order]


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
    """执行 gather_style_cross_section 对应的分布式训练处理。"""
    sample_indices = batch["sample_indices"].to(raw_prediction.device)
    date_ids = batch["date_ids"].to(raw_prediction.device)
    if context is None:
        order = torch.arange(len(sample_indices), device=sample_indices.device)
        global_indices = sample_indices
        global_dates = date_ids
    else:
        global_indices = distributed.all_gather_tensor(sample_indices)
        global_dates = distributed.all_gather_tensor(date_ids)
        order = torch.argsort(global_indices)
        global_indices = global_indices[order]
        global_dates = global_dates[order]

    def metadata(value: torch.Tensor) -> torch.Tensor:
        """执行 metadata 对应的分布式训练处理。"""
        if context is None:
            return value
        return distributed.all_gather_tensor(value)[order]

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
    """训练一个完整 epoch，并返回损失与批次诊断。"""
    model.train()  # 启用训练模式
    gamma = adversarial_gamma(epoch) if adversarial_barra_enabled() else 0.0
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
    samples = 0
    batches = 0
    started = time.perf_counter()
    for batch_number, item in enumerate(
        StyleCudaPrefetcher(loader, device, bool(USE_CUDA_PREFETCH)), start=1
    ):
        (
            batch, videos, frame_mask, targets, original_targets,
            _residual_targets, fitted_targets, styles, style_valid,
            industry_codes,
        ) = item
        date_ids = batch["date_ids"]
        if int(date_ids.min()) != int(date_ids.max()):
            raise AssertionError("训练 batch 混入多个交易日")
        if not torch.isfinite(targets).all():
            raise FloatingPointError("训练目标含 NaN/Inf")

        (  # 清空上一批梯度
            conditioning_styles,
            conditioning_valid,
            conditioning_industry,
            conditioning_prepared,
        ) = prepare_distributed_conditioning(
            styles, style_valid, industry_codes, batch["sample_indices"], distributed_context
        )
        optimizer.zero_grad(set_to_none=True)
        with base.autocast_context(device):
            (  # 执行当前 batch 前向传播
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
        predictions, _ = effective_predictions(
            raw_predictions, styles, industry_codes
        )
        raw_style = measure_style_correlations(raw_predictions, styles)  # 计算当前截面相关性
        effective_style = measure_style_correlations(predictions, styles)  # 计算当前截面相关性
        if dual_tower_enabled():
            if dual_style_score is None or total_return_score is None:
                raise ValueError("No.6 缺少 Style 或 Total 分数")
            dual_losses = dual_tower_training_loss(  # 按实验口径计算可微损失
                raw_predictions,
                dual_style_score,
                total_return_score,
                original_targets,
                targets,
                fitted_targets,
                styles,
                industry_codes,
            )
            loss = dual_losses["loss"]
            predictive_loss = dual_losses["alpha_rank_loss"]
            barra_r_squared = dual_losses["alpha_barra_r_squared"]
            return_rank_loss = dual_losses["return_rank_loss"]
            alpha_rank_loss = dual_losses["alpha_rank_loss"]
            dual_style_rank_loss = dual_losses["style_rank_loss"]
        else:
            loss, predictive_loss, barra_r_squared = combined_training_loss(  # 按实验口径计算可微损失
                raw_predictions, predictions, targets, styles, industry_codes
            )
            zero = torch.zeros((), dtype=torch.float32, device=device)
            return_rank_loss = zero
            alpha_rank_loss = zero
            dual_style_rank_loss = zero
        style_loss, industry_loss = adversarial_training_loss(  # 按实验口径计算可微损失
            style_predictions,
            industry_logits,
            styles,
            style_valid,
            industry_codes,
            industry_class_weights,
        )
        loss = (
            loss
            + ADVERSARIAL_STYLE_LOSS_WEIGHT * style_loss
            + ADVERSARIAL_INDUSTRY_LOSS_WEIGHT * industry_loss
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("训练损失出现 NaN 或 Inf")
        loss.backward()  # 反向传播当前损失
        optimizer.step()  # 更新参数或学习率状态

        with torch.no_grad():
            if (
                adversarial_barra_enabled()
                or barra_conditioning_enabled()
                or dual_tower_enabled()
            ):
                barra_r_squared, _ = style_data.joint_barra_r_squared_torch(  # 按股票与交易日业务键对齐
                    raw_predictions.detach(), styles, industry_codes
                )
            effective_pearson = pearson_correlation(predictions, targets)  # 计算当前截面相关性
            original_pearson = pearson_correlation(  # 计算当前截面相关性
                (
                    total_return_score.float()
                    if dual_tower_enabled() else raw_predictions.float()
                ),
                original_targets,
            )
            if industry_logits is not None:
                known = industry_codes > 0
                industry_accuracy = (
                    industry_logits[known].argmax(dim=1)
                    == industry_codes[known].long() - 1
                ).float().mean()
            else:
                industry_accuracy = torch.zeros((), device=device)
        totals["loss"] += float(loss.detach())
        totals["rank_loss"] += float(predictive_loss.detach())
        totals["return_rank_loss"] += float(return_rank_loss.detach())
        totals["alpha_rank_loss"] += float(alpha_rank_loss.detach())
        totals["dual_style_rank_loss"] += float(
            dual_style_rank_loss.detach()
        )
        totals["barra_r_squared"] += float(barra_r_squared.detach())
        totals["effective_pearson"] += float(effective_pearson)
        totals["original_pearson"] += float(original_pearson)
        totals["raw_style_max"] = max(
            totals["raw_style_max"],
            float(raw_style["max_abs_style_pearson"]),
        )
        totals["raw_style_mean"] += float(
            raw_style["mean_abs_style_pearson"]
        )
        totals["effective_style_max"] = max(
            totals["effective_style_max"],
            float(effective_style["max_abs_style_pearson"]),
        )
        totals["effective_style_mean"] += float(
            effective_style["mean_abs_style_pearson"]
        )
        totals["adversarial_style_loss"] += float(style_loss.detach())
        totals["adversarial_industry_loss"] += float(industry_loss.detach())
        totals["adversarial_industry_accuracy"] += float(industry_accuracy)
        totals["adversarial_unknown_rate"] += float(
            (industry_codes == 0).float().mean()
        )
        if gate is not None:
            gate_abs = gate.detach().float().abs()
            totals["gate_mean_abs"] += float(gate_abs.mean())
            totals["gate_max_abs"] = max(
                totals["gate_max_abs"], float(gate_abs.max())
            )
        samples += len(targets)
        batches += 1
        if max_batches is not None and batch_number >= max_batches:
            break

    if not batches:
        raise ValueError("训练 loader 没有可用 batch")
    if device.type == "cuda":
        torch.cuda.synchronize(device)  # 同步 GPU 后计时
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


_INSTRUMENT_IDS: np.ndarray | None = None


def instrument_ids() -> np.ndarray:
    """根据样本行号返回对应股票代码。"""
    global _INSTRUMENT_IDS
    if _INSTRUMENT_IDS is None:
        _INSTRUMENT_IDS = pd.read_parquet(
            sequence_data.SAMPLE_INDEX_PATH, columns=["instrument_id"]
        )["instrument_id"].to_numpy(copy=False)
    return _INSTRUMENT_IDS


def has_degenerate_daily_pearson(
    predictions: np.ndarray,
    targets: np.ndarray,
    date_ids: np.ndarray,
) -> bool:
    """判断是否存在无法计算 Pearson 的交易日截面。"""
    x = np.asarray(predictions, dtype=np.float64).reshape(-1)
    y = np.asarray(targets, dtype=np.float64).reshape(-1)
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
    """在验证集上评估模型，并汇总逐日 IC 与损失诊断。"""
    local_model = base.inference_model(model)  # 关闭 dropout 等训练行为
    local_model.eval()
    effective_predictions_parts: list[np.ndarray] = []
    raw_predictions_parts: list[np.ndarray] = []
    effective_targets: list[np.ndarray] = []
    original_targets: list[np.ndarray] = []
    dual_style_predictions: list[np.ndarray] = []
    dual_total_predictions: list[np.ndarray] = []
    dual_fitted_targets: list[np.ndarray] = []
    date_ids: list[np.ndarray] = []
    sample_indices: list[np.ndarray] = []
    adversarial_style_predictions: list[np.ndarray] = []
    adversarial_style_targets: list[np.ndarray] = []
    adversarial_style_valid: list[np.ndarray] = []
    adversarial_industry_logits: list[np.ndarray] = []
    adversarial_industry_codes: list[np.ndarray] = []
    raw_style_max = 0.0
    raw_style_mean = 0.0
    effective_style_max = 0.0
    effective_style_mean = 0.0
    barra_r_squared_sum = 0.0
    barra_r_squared_max = 0.0
    effective_barra_r_squared_sum = 0.0
    effective_barra_r_squared_max = 0.0
    effective_industry_mean_max = 0.0
    gate_mean_abs = 0.0
    gate_max_abs = 0.0
    minimum_daily_raw_score_std = np.inf
    batches = 0
    started = time.perf_counter()

    for batch_number, item in enumerate(
        StyleCudaPrefetcher(loader, device, bool(USE_CUDA_PREFETCH)), start=1
    ):
        (
            batch, videos, frame_mask, targets, original, _residual,
            fitted_targets, styles, style_valid, industry_codes,
        ) = item
        if int(batch["date_ids"].min()) != int(batch["date_ids"].max()):
            raise AssertionError("验证 batch 混入多个交易日")
        (
            conditioning_styles,
            conditioning_valid,
            conditioning_industry,
            conditioning_prepared,
        ) = prepare_distributed_conditioning(
            styles, style_valid, industry_codes, batch["sample_indices"], distributed_context
        )
        with base.autocast_context(device):
            (  # 执行当前 batch 前向传播
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
        prediction, projection_diagnostics = effective_predictions(
            raw_prediction, styles, industry_codes
        )
        if strict_barra_output_enabled():
            if projection_diagnostics is None:
                raise AssertionError("严格输出缺少投影诊断")
            effective_industry_mean_max = max(
                effective_industry_mean_max,
                float(projection_diagnostics["max_abs_industry_mean"]),
            )
        date_id = int(batch["date_ids"][0])
        raw_score_dispersion = measure_daily_score_dispersion(
            raw_prediction,
            date_id=date_id,
        )
        minimum_daily_raw_score_std = min(
            minimum_daily_raw_score_std,
            float(raw_score_dispersion["score_std"].item()),
        )
        raw_style = measure_style_correlations(raw_prediction, styles)  # 计算当前截面相关性
        effective_style = measure_style_correlations(prediction, styles)  # 计算当前截面相关性
        if industry_sidecar_required():
            barra_r_squared, _ = style_data.joint_barra_r_squared_torch(  # 按股票与交易日业务键对齐
                raw_prediction, styles, industry_codes
            )
        else:
            barra_r_squared = torch.zeros(
                (), dtype=torch.float32, device=raw_prediction.device
            )
        if strict_barra_output_enabled():
            effective_barra_r_squared, _ = (  # 按股票与交易日业务键对齐
                style_data.joint_barra_r_squared_torch(
                    prediction, styles, industry_codes
                )
            )
            if float(effective_barra_r_squared) > 1e-6:
                raise FloatingPointError(
                    "严格 Barra 输出的联合 R² 未接近零"
                )
        else:
            effective_barra_r_squared = torch.zeros_like(barra_r_squared)

        effective_predictions_parts.append(
            prediction.cpu().numpy().astype(np.float64)
        )
        raw_predictions_parts.append(
            raw_prediction.float().cpu().numpy().astype(np.float64)
        )
        effective_targets.append(targets.cpu().numpy().astype(np.float64))
        original_targets.append(original.cpu().numpy().astype(np.float64))
        if dual_tower_enabled():
            if dual_style_score is None or total_return_score is None:
                raise ValueError("No.6 验证缺少 Style 或 Total 分数")
            dual_style_predictions.append(
                dual_style_score.float().cpu().numpy().astype(np.float64)
            )
            dual_total_predictions.append(
                total_return_score.float().cpu().numpy().astype(np.float64)
            )
            dual_fitted_targets.append(
                fitted_targets.cpu().numpy().astype(np.float64)
            )
        date_ids.append(global_date_ids.cpu().numpy().astype(np.int64, copy=False))
        sample_indices.append(global_sample_indices.cpu().numpy().astype(np.int64, copy=False))
        if adversarial_barra_enabled():
            if style_prediction is None or industry_logits is None:
                raise ValueError("No.3 验证缺少对抗头输出")
            adversarial_style_predictions.append(
                style_prediction.float().cpu().numpy()
            )
            adversarial_style_targets.append(styles.float().cpu().numpy())
            adversarial_style_valid.append(style_valid.cpu().numpy())
            adversarial_industry_logits.append(
                industry_logits.float().cpu().numpy()
            )
            adversarial_industry_codes.append(industry_codes.cpu().numpy())
        raw_style_max = max(
            raw_style_max, float(raw_style["max_abs_style_pearson"])
        )
        raw_style_mean += float(raw_style["mean_abs_style_pearson"])
        effective_style_max = max(
            effective_style_max,
            float(effective_style["max_abs_style_pearson"]),
        )
        effective_style_mean += float(
            effective_style["mean_abs_style_pearson"]
        )
        barra_value = float(barra_r_squared)
        barra_r_squared_sum += barra_value
        barra_r_squared_max = max(barra_r_squared_max, barra_value)
        effective_barra_value = float(effective_barra_r_squared)
        effective_barra_r_squared_sum += effective_barra_value
        effective_barra_r_squared_max = max(
            effective_barra_r_squared_max, effective_barra_value
        )
        if gate is not None:
            gate_abs = gate.float().abs()
            gate_mean_abs += float(gate_abs.mean())
            gate_max_abs = max(gate_max_abs, float(gate_abs.max()))
        batches += 1
        if max_batches is not None and batch_number >= max_batches:
            break

    if not batches:
        raise ValueError("验证 loader 没有可用 batch")
    if device.type == "cuda":
        torch.cuda.synchronize(device)  # 同步 GPU 后计时
    effective_x = np.concatenate(effective_predictions_parts)
    raw_x = np.concatenate(raw_predictions_parts)
    effective_y = np.concatenate(effective_targets)
    original_y = np.concatenate(original_targets)
    dates = np.concatenate(date_ids)
    selected = np.concatenate(sample_indices)
    effective_metrics = summarize_validation_arrays(  # 按约定截面口径汇总
        effective_x, effective_y, dates, VALIDATION_METRIC,
        instrument_ids=instrument_ids()[selected],
    )
    dual_metrics: dict[str, float | bool] = {}
    original_prediction = raw_x
    if dual_tower_enabled():
        style_x = np.concatenate(dual_style_predictions)
        total_x = np.concatenate(dual_total_predictions)
        fitted_y = np.concatenate(dual_fitted_targets)
        original_prediction = total_x
        total_metrics = summarize_validation_arrays(  # 按约定截面口径汇总
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
        if style_collapsed:
            style_rank_ic = float("nan")
            style_pearson_ic = float("nan")
        else:
            style_metrics = summarize_validation_arrays(  # 按约定截面口径汇总
                style_x,
                fitted_y,
                dates,
                VALIDATION_METRIC,
                instrument_ids=instrument_ids()[selected],
            )
            style_rank_ic = float(
                style_metrics["validation_daily_rank_ic"]
            )
            style_pearson_ic = float(
                style_metrics["validation_daily_pearson_ic"]
            )
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
    original_metrics = summarize_validation_arrays(  # 按约定截面口径汇总
        original_prediction, original_y, dates, VALIDATION_METRIC,
        instrument_ids=instrument_ids()[selected],
    )
    adversarial_metrics: dict[str, float] = {}
    if adversarial_barra_enabled():
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
    """在预测集上生成与样本行号严格对齐的模型分数。"""
    if (
        not barra_conditioning_enabled()
        and not dual_tower_enabled()
        and (raw_barra_r2_enabled() or adversarial_barra_enabled())
    ):
        raw = base.predict_model(
            model, loader, device, bool(USE_CUDA_PREFETCH),
            distributed_context=distributed_context,
        )
        values = raw["prediction"].to_numpy(np.float32)
        return pd.DataFrame(
            {
                "sample_index": raw["sample_index"].to_numpy(np.int64),
                "prediction_model_raw": values,
                "prediction": values,
            }
        )
    local_model = base.inference_model(model)  # 关闭 dropout 等训练行为
    local_model.eval()
    sample_parts: list[np.ndarray] = []
    raw_parts: list[np.ndarray] = []
    effective_parts: list[np.ndarray] = []
    style_parts: list[np.ndarray] = []
    total_parts: list[np.ndarray] = []
    for item in StyleCudaPrefetcher(
        loader, device, bool(USE_CUDA_PREFETCH)
    ):
        (
            batch, videos, frame_mask, _targets, _original, _residual,
            _fitted, styles, style_valid, industry_codes,
        ) = item
        if int(batch["date_ids"].min()) != int(batch["date_ids"].max()):
            raise AssertionError("推理 batch 混入多个交易日")
        (
            conditioning_styles,
            conditioning_valid,
            conditioning_industry,
            conditioning_prepared,
        ) = prepare_distributed_conditioning(
            styles, style_valid, industry_codes, batch["sample_indices"], distributed_context
        )
        with base.autocast_context(device):
            (  # 执行当前 batch 前向传播
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
        prediction, _ = effective_predictions(
            raw_prediction, styles, industry_codes
        )
        sample_parts.append(global_sample_indices.cpu().numpy())
        raw_parts.append(raw_prediction.float().cpu().numpy())
        effective_parts.append(prediction.cpu().numpy())
        if dual_tower_enabled():
            if dual_style_score is None or total_return_score is None:
                raise ValueError("No.6 推理缺少 Style 或 Total 分数")
            style_parts.append(dual_style_score.float().cpu().numpy())
            total_parts.append(total_return_score.float().cpu().numpy())
    if not sample_parts:
        return pd.DataFrame(
            {
                "sample_index": pd.Series(dtype=np.int64),
                "prediction_model_raw": pd.Series(dtype=np.float32),
                "prediction": pd.Series(dtype=np.float32),
                "prediction_style_component": pd.Series(dtype=np.float32),
                "prediction_total_return": pd.Series(dtype=np.float32),
            }
        )
    result = pd.DataFrame(
        {
            "sample_index": np.concatenate(sample_parts).astype(np.int64),
            "prediction_model_raw": np.concatenate(raw_parts).astype(np.float32),
            "prediction": np.concatenate(effective_parts).astype(np.float32),
        }
    )
    if dual_tower_enabled():
        result["prediction_style_component"] = np.concatenate(
            style_parts
        ).astype(np.float32)
        result["prediction_total_return"] = np.concatenate(
            total_parts
        ).astype(np.float32)
    return result.sort_values("sample_index", kind="stable").reset_index(
        drop=True
    )


PROBE_MAX_SAMPLES_PER_DATE = 512
PROBE_BATCH_SIZE = 4096
PROBE_LEARNING_RATE = 1e-3
PROBE_WEIGHT_DECAY = 1e-4
PROBE_MAX_EPOCHS = 50
PROBE_PATIENCE = 5
PROBE_RIDGE_ALPHA = 1e-3


# %% [markdown]
# ## 年度 × seed 训练和断点续训


# %%
def run_payload(
    year: int, seed: int, batch_summary: dict[str, Any]
) -> dict[str, Any]:
    """整理影响整个 run 身份和可复用性的签名载荷。"""
    sequence = sequence_data.require_aligned_sequence_store()
    style_manifest = style_data.require_style_neutral_store()
    industry_manifest = (
        style_data.require_industry_sidecar()
        if industry_sidecar_required() else None
    )
    full_target_manifest = (
        style_data.require_full_barra_target_sidecar()
        if barra_conditioning_enabled() or dual_tower_enabled() else None
    )
    dual_target_manifest = (
        style_data.require_dual_tower_target_sidecar()
        if dual_tower_enabled() else None
    )
    config = training_config()
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
    if EXECUTION_ROUTE == "ddp":
        payload.update({
            "execution_route": "ddp",
            "world_size": DDP_WORLD_SIZE,
        })
    if dual_tower_enabled():
        payload["full_barra_target_sidecar_signature"] = (
            full_target_manifest["signature"]
        )
        payload["dual_tower_target_sidecar_signature"] = (
            dual_target_manifest["signature"]
        )
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
    if barra_conditioning_enabled():
        payload["full_barra_target_sidecar_signature"] = (
            full_target_manifest["signature"]
        )
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
    if strict_barra_output_enabled():
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
    return payload


def load_best_model(
    path: Path, device: torch.device
) -> torch.nn.Module:
    """加载最佳验证 checkpoint，并恢复可用于预测的模型。"""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint["model_state"]
    if dual_tower_enabled() or barra_conditioning_enabled():
        model = build_training_model(device, use_compile=False)
        base.load_canonical_state(model, state)
        model.eval()  # 关闭 dropout 等训练行为
        return model
    if adversarial_barra_enabled():
        prefix = "backbone."
        state = {
            key.removeprefix(prefix): value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        if not state:
            raise ValueError("No.3 checkpoint 缺少 backbone 参数")
    model = base.build_model(  # 创建模型并部署到设备
        device, bool(USE_TORCH_COMPILE), COMPILE_MODE, TRAIN_BATCH_MODE
    )
    base.load_canonical_state(model, state)
    model.eval()  # 关闭 dropout 等训练行为
    return model


def load_best_training_model(
    path: Path, device: torch.device
) -> torch.nn.Module:
    """加载包含附加实验分支的最佳训练模型。"""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_training_model(device, use_compile=False)
    base.load_canonical_state(model, checkpoint["model_state"])
    model.eval()  # 关闭 dropout 等训练行为
    return model


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
    """执行一个年度×seed 任务，保存 checkpoint、预测和训练诊断。"""
    style_data.require_style_neutral_store()
    if industry_sidecar_required():
        style_data.require_industry_sidecar()
    if barra_conditioning_enabled() or dual_tower_enabled():
        style_data.require_full_barra_target_sidecar()
    if dual_tower_enabled():
        style_data.require_dual_tower_target_sidecar()
    primary = distributed_context is None or distributed_context.is_primary
    device = (
        distributed_context.device if distributed_context is not None
        else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    if device.type != "cuda":
        raise RuntimeError("完整风格中性训练要求 CUDA")
    base.set_random_seed(  # 固定各随机源
        seed + (distributed_context.rank if distributed_context else 0)
    )
    train_loader, validation_loader, prediction_loader = make_style_loaders(
        year, seed, WORKERS_PER_TASK if persist else 0, pin_memory=True,
        distributed_context=distributed_context,
    )
    split = sequence_data.load_sequence_split(year)
    industry_class_weights = (
        training_industry_class_weights(split["train_indices"]).to(device)
        if adversarial_barra_enabled()
        else torch.ones(INDUSTRY_COUNT, dtype=torch.float32, device=device)
    )
    batch_summary = sequence_data.loader_batch_summary(train_loader)
    payload = run_payload(year, seed, batch_summary)
    signature = sequence_data.stable_hash(payload)  # 生成数据复用签名
    config = training_config()
    epoch_limit = int(max_epochs or MAX_EPOCHS)

    model_root, _, _ = base.run_paths(RUN_NAME)
    run_directory = model_root / f"forecast_{year}" / f"seed_{seed}"
    best_path = run_directory / "best.pt"
    last_path = run_directory / "last.pt"
    history_path = run_directory / "history.parquet"
    prediction_path = run_directory / "prediction.parquet"
    probe_path = run_directory / "adversarial_probe_diagnostics.parquet"
    completed_path = run_directory / "completed.json"
    resume_checkpoint: dict[str, Any] | None = None
    if persist:
        if primary:
            run_directory.mkdir(parents=True, exist_ok=True)
        if distributed_context is not None:
            dist.barrier()
        if completed_path.exists() and best_path.exists() and not OVERWRITE:
            completed = json.loads(completed_path.read_text(encoding="utf-8"))
            if completed.get("run_signature") == signature:
                print(f"[{year}/seed={seed}] 已完成，直接复用。", flush=True)
                return completed["summary"]
            raise ValueError(
                f"{completed_path} 与当前配置签名不一致；"
                "请使用新 run name，或显式设置 --overwrite 1"
            )
        if last_path.exists() and not OVERWRITE:
            resume_checkpoint = torch.load(
                last_path, map_location="cpu", weights_only=False
            )
            if resume_checkpoint.get("run_signature") != signature:
                raise ValueError(
                    f"{last_path} 与当前配置签名不一致；"
                    "请使用新 run name，或显式设置 --overwrite 1"
                )
        if primary:  # 原子落盘，避免半成品
            sequence_data.atomic_write_json(
                {
                    "signature": sequence_data.stable_hash(
                        {k: v for k, v in payload.items() if k != "seed"}
                    ),
                    "payload": {k: v for k, v in payload.items() if k != "seed"},
                },
                model_root / f"forecast_{year}" / "model_config.json",
            )
        if distributed_context is not None:
            dist.barrier()

    model = build_training_model(
        device, use_compile=bool(USE_TORCH_COMPILE and persist)
    )
    if distributed_context is not None:
        model = distributed.wrap_model(model, distributed_context)
    optimizer = base.build_optimizer(  # 按训练配置创建优化器
        model, config, bool(USE_FUSED_ADAM), device
    )
    start_epoch = 1
    best_epoch = 0
    best_validation_ic = -np.inf
    best_metrics: dict[str, float] = {}
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    if resume_checkpoint is not None:
        checkpoint = resume_checkpoint
        base.load_canonical_state(model, checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])  # 恢复优化器续训状态
        if distributed_context is not None:
            states = checkpoint.get("rng_state_by_rank")
            if states is None or len(states) != distributed_context.world_size:
                raise ValueError("DDP checkpoint 缺少匹配 world size 的逐 rank RNG 状态")
            distributed.restore_rng_state(states[distributed_context.rank], device)
        else:
            random.setstate(checkpoint["python_random_state"])
            np.random.set_state(checkpoint["numpy_random_state"])
            torch.set_rng_state(checkpoint["torch_rng_state"])
            if checkpoint["cuda_rng_state"] is not None:
                torch.cuda.set_rng_state(checkpoint["cuda_rng_state"], device)
        start_epoch = int(checkpoint["epoch"]) + 1
        best_epoch = int(checkpoint["best_epoch"])
        best_validation_ic = float(checkpoint["best_validation_ic"])
        best_metrics = {
            k: float(v) for k, v in checkpoint["best_metrics"].items()
        }
        epochs_without_improvement = int(
            checkpoint["epochs_without_improvement"]
        )
        if history_path.exists():
            history = pd.read_parquet(history_path).to_dict("records")
            checkpoint_epoch = int(checkpoint["epoch"])
            history = [
                row for row in history
                if int(row["epoch"]) <= checkpoint_epoch
            ]
        print(f"[{year}/seed={seed}] 从 epoch {start_epoch} 续训。", flush=True)

    started = time.perf_counter()
    stopped_early = False
    stop_reason = "max_epochs"
    validation_score_collapsed = False
    collapsed_validation_date_id: int | None = None
    collapsed_validation_score_std: float | None = None
    collapsed_validation_total_sum_squares: float | None = None
    last_successful_epoch = start_epoch - 1
    for epoch in range(start_epoch, epoch_limit + 1):
        epoch_started = time.perf_counter()
        sequence_data.set_train_epoch(train_loader, epoch)  # 完成一个训练 epoch
        train_metrics = train_one_epoch(  # 完成一个训练 epoch
            model,
            train_loader,
            optimizer,
            device,
            epoch=epoch,
            industry_class_weights=industry_class_weights,
            max_batches=max_batches,
            distributed_context=distributed_context,
        )
        try:
            validation_metrics = evaluate_model(  # 无梯度计算验证指标
                model,
                validation_loader,
                device,
                industry_class_weights=industry_class_weights,
                max_batches=max_batches,
                distributed_context=distributed_context,
            )
        except DegenerateValidationScoreError as exc:
            if not persist or best_epoch <= 0 or not best_path.exists():
                raise
            stopped_early = True
            stop_reason = "validation_score_collapse"
            validation_score_collapsed = True
            collapsed_validation_date_id = exc.date_id
            collapsed_validation_score_std = exc.score_std
            collapsed_validation_total_sum_squares = exc.total_sum_squares
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
            if primary:  # 原子落盘，避免半成品
                sequence_data.atomic_write_parquet(pd.DataFrame(history), history_path)
            if distributed_context is not None:
                dist.barrier()
            print(
                f"[{year}/seed={seed}] epoch={epoch:03d} 验证分数退化；"
                f"date_id={exc.date_id}, std={exc.score_std:.3e}。"
                f"停止当前 seed，并使用 best epoch={best_epoch} 生成预测。",
                flush=True,
            )
            break
        selected = float(validation_metrics["validation_ic"])
        improved = selected > best_validation_ic + config.improvement_tolerance
        if improved:
            best_epoch = epoch
            best_validation_ic = selected
            best_metrics = {
                key: float(value)
                for key, value in validation_metrics.items()
                if isinstance(value, (int, float))
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
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
        history.append(record)
        if persist:
            if primary:  # 原子落盘，避免半成品
                sequence_data.atomic_write_parquet(pd.DataFrame(history), history_path)
            rng_state_by_rank = (
                distributed.gather_rng_states(device)
                if distributed_context is not None else None
            )
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
            if rng_state_by_rank is not None:
                checkpoint["rng_state_by_rank"] = rng_state_by_rank
            if primary:
                sequence_data.atomic_save_checkpoint(checkpoint, last_path)
            if primary and improved:
                sequence_data.atomic_save_checkpoint(checkpoint, best_path)
            if distributed_context is not None:
                dist.barrier()
        last_successful_epoch = epoch
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
        if epochs_without_improvement >= PATIENCE:
            stopped_early = True
            stop_reason = "patience"
            break

    if not persist:
        return {
            "forecast_year": year,
            "seed": seed,
            "device": str(device),
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
            "persistent_outputs_written": False,
        }
    if not best_path.exists():
        raise FileNotFoundError(f"未生成最佳 checkpoint：{best_path}")
    if prediction_loader is not None:
        best_model = load_best_model(best_path, device)
        prediction_frame = predict_model(
            best_model, prediction_loader, device,
            distributed_context=distributed_context,
        )
        if primary:  # 原子落盘，避免半成品
            sequence_data.atomic_write_parquet(prediction_frame, prediction_path)
        del best_model
        if distributed_context is not None:
            dist.barrier()
    probe_summary: dict[str, float] = {}
    if primary and adversarial_barra_enabled():
        probe_validation_loader = validation_loader
        if distributed_context is not None:
            _, probe_validation_loader, _ = make_style_loaders(
                year, seed, 0, pin_memory=True,
                distributed_context=None,
            )
        probe_model = load_best_training_model(best_path, device)
        probe_frame = run_adversarial_probes(  # 用探针测量表征中的风险信息
            probe_model,
            probe_validation_loader,
            device,
            year=year,
            seed=seed,
        )
        sequence_data.atomic_write_parquet(probe_frame, probe_path)  # 原子落盘，避免半成品
        style_summary = probe_frame[  # 按约定截面口径汇总
            probe_frame["target_kind"].eq("style_summary")
        ].set_index("probe")
        industry_summary = probe_frame[
            probe_frame["target_kind"].eq("industry")
        ].set_index("probe")
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
        del probe_model
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
    if distributed_context is not None:  # 原子落盘，避免半成品
        dist.barrier()
    if primary:
        sequence_data.atomic_write_json(
            {"run_signature": signature, "summary": summary}, completed_path
        )
    if distributed_context is not None:
        dist.barrier()
    return summary


# %% [markdown]
# ## 步骤 5：三 seed 集成、风格相关性审计和 run manifest


# %%
def canonical_dual_tower_total_scores(
    alpha_values: np.ndarray,
    style_values: np.ndarray,
    stored_total_values: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """审计推理精度，并以落盘分量重建严格可加的 Total。"""

    alpha = np.asarray(alpha_values, dtype=np.float32)
    style = np.asarray(style_values, dtype=np.float32)
    stored = np.asarray(stored_total_values, dtype=np.float32)
    if alpha.shape != style.shape or alpha.shape != stored.shape:
        raise ValueError("No.6 Alpha、Style、Total 形状不一致")
    if not np.isfinite(np.column_stack([alpha, style, stored])).all():
        raise ValueError("No.6 Alpha、Style、Total 含 NaN/Inf")

    canonical = (alpha + style).astype(np.float32)
    bfloat16_sum = (
        torch.from_numpy(alpha.copy()).to(torch.bfloat16)
        + torch.from_numpy(style.copy()).to(torch.bfloat16)
    ).float().numpy()
    matches_float32 = np.array_equal(stored, canonical)  # 核对样本顺序严格一致
    matches_bfloat16 = np.array_equal(stored, bfloat16_sum)  # 核对样本顺序严格一致
    if not matches_float32 and not matches_bfloat16:
        raise AssertionError(
            "No.6 推理 Total 既不满足 float32 加法，也不满足 bfloat16 加法"
        )
    error = np.abs(stored.astype(np.float64) - canonical.astype(np.float64))
    return canonical, {
        "stored_total_precision": (
            "float32" if matches_float32 else "bfloat16"
        ),
        "stored_vs_canonical_max_abs_error": float(error.max()),
        "stored_vs_canonical_mean_abs_error": float(error.mean()),
    }


def build_ensemble(year: int) -> tuple[int, pd.DataFrame]:
    """按既定尺度组合多个随机种子的预测，并保存正式年度信号。"""
    model_root, prediction_root, _ = base.run_paths(RUN_NAME)
    split = sequence_data.load_sequence_split(year)
    if not len(split["prediction_indices"]):
        return 0, pd.DataFrame()
    frames = {
        seed: pd.read_parquet(
            model_root / f"forecast_{year}" / f"seed_{seed}"
            / "prediction.parquet"
        )
        for seed in SEEDS
    }
    reference = frames[SEEDS[0]]["sample_index"].to_numpy(np.int64)
    output = pd.DataFrame({"sample_index": reference})
    raw_columns: list[str] = []
    effective_columns: list[str] = []
    dual_style_columns: list[str] = []
    dual_total_columns: list[str] = []
    dual_inference_audits: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        frame = frames[seed]
        if not np.array_equal(
            reference, frame["sample_index"].to_numpy(np.int64)
        ):
            raise AssertionError(f"{year} 三个 seed 的预测索引不一致")
        raw_column = f"prediction_seed_{seed}_unprojected"
        effective_column = f"prediction_seed_{seed}"
        output[raw_column] = frame[
            "prediction_model_raw"
        ].to_numpy(np.float32)
        output[effective_column] = frame["prediction"].to_numpy(np.float32)
        raw_columns.append(raw_column)
        effective_columns.append(effective_column)
        if dual_tower_enabled():
            style_column = f"prediction_seed_{seed}_style"
            total_column = f"prediction_seed_{seed}_total_return"
            style_values = frame[
                "prediction_style_component"
            ].to_numpy(np.float32)
            stored_total_values = frame[
                "prediction_total_return"
            ].to_numpy(np.float32)
            total_values, dual_inference_audits[seed] = (  # 通过当前网络子模块
                canonical_dual_tower_total_scores(
                    output[raw_column].to_numpy(np.float32),
                    style_values,
                    stored_total_values,
                )
            )
            output[style_column] = style_values
            output[total_column] = total_values
            dual_style_columns.append(style_column)
            dual_total_columns.append(total_column)

    metadata = pd.read_parquet(
        sequence_data.SAMPLE_INDEX_PATH,
        columns=["sample_index", "trade_date", "instrument_id"],
    ).set_index("sample_index")
    output = metadata.loc[reference].reset_index().merge(  # 按业务键对齐，避免依赖原始行序
        output, on="sample_index", validate="one_to_one"
    )

    def daily_zscore(column: str) -> str:
        """按交易日用 Alpha 分数尺度标准化双塔分量，保持加法恒等式。"""
        grouped = output[column].astype(np.float64).groupby(
            output["trade_date"], sort=False
        )
        mean = grouped.transform("mean")  # 按约定截面口径汇总
        std = grouped.transform("std", ddof=0)  # 按约定截面口径汇总
        if (std <= 0).any() or not np.isfinite(std).all():
            raise ValueError(f"{year} {column} 存在常数或非有限日截面")
        result_column = f"{column}_zscore"
        output[result_column] = (
            (output[column] - mean) / std
        ).astype(np.float32)
        return result_column

    if dual_tower_enabled():
        alpha_scaled: list[str] = []
        style_scaled: list[str] = []
        total_scaled: list[str] = []
        for seed, alpha_column, style_column in zip(
            SEEDS, raw_columns, dual_style_columns
        ):
            alpha_grouped = output[alpha_column].astype(np.float64).groupby(
                output["trade_date"], sort=False
            )
            alpha_mean = alpha_grouped.transform("mean")  # 按约定截面口径汇总
            alpha_std = alpha_grouped.transform("std", ddof=0)  # 按约定截面口径汇总
            if (alpha_std <= 0).any() or not np.isfinite(alpha_std).all():
                raise ValueError(f"{year}/seed={seed} Alpha 日截面为常数")
            style_mean = output[style_column].astype(np.float64).groupby(
                output["trade_date"], sort=False
            ).transform("mean")
            alpha_scaled_column = f"{alpha_column}_alpha_scale"
            style_scaled_column = f"{style_column}_alpha_scale"
            total_scaled_column = f"prediction_seed_{seed}_total_alpha_scale"
            output[alpha_scaled_column] = (
                (output[alpha_column] - alpha_mean) / alpha_std
            ).astype(np.float32)
            output[style_scaled_column] = (
                (output[style_column] - style_mean) / alpha_std
            ).astype(np.float32)
            output[total_scaled_column] = (
                output[alpha_scaled_column] + output[style_scaled_column]
            ).astype(np.float32)
            alpha_scaled.append(alpha_scaled_column)
            style_scaled.append(style_scaled_column)
            total_scaled.append(total_scaled_column)
        final_alpha = output[alpha_scaled].mean(axis=1).to_numpy(np.float64)
        final_style = output[style_scaled].mean(axis=1).to_numpy(np.float64)
        final_total = (
            final_alpha.astype(np.float32) + final_style.astype(np.float32)
        ).astype(np.float32).astype(np.float64)
        final_raw = final_alpha
        final_effective = final_alpha
    else:
        raw_zscore_columns = [daily_zscore(column) for column in raw_columns]
        effective_zscore_columns = [
            daily_zscore(column) for column in effective_columns
        ]
        final_raw = output[raw_zscore_columns].mean(axis=1).to_numpy(np.float64)
        final_effective = (
            output[effective_zscore_columns].mean(axis=1).to_numpy(np.float64)
        )

    style_matrix = np.load(
        style_data.DEFAULT_STORE_DIR / style_data.STYLE_EXPOSURES_FILE,
        mmap_mode="r",
    )
    style_valid = np.load(
        style_data.DEFAULT_STORE_DIR / style_data.STYLE_VALID_FILE,
        mmap_mode="r",
    )
    industry_codes = None
    if industry_sidecar_required():
        style_data.require_industry_sidecar()
        industry_codes = np.load(
            style_data.DEFAULT_INDUSTRY_STORE_DIR
            / style_data.INDUSTRY_CODES_FILE,
            mmap_mode="r",
        )
    dual_source_targets = None
    dual_residual_targets = None
    dual_fitted_targets = None
    if dual_tower_enabled():
        dual_source_targets = np.load(sequence_data.TARGETS_PATH, mmap_mode="r")
        dual_residual_targets = np.load(
            style_data.DEFAULT_FULL_BARRA_TARGET_STORE_DIR
            / style_data.FULL_BARRA_TARGETS_FILE,
            mmap_mode="r",
        )
        dual_fitted_targets = np.load(
            style_data.DEFAULT_DUAL_TOWER_TARGET_STORE_DIR
            / style_data.DUAL_TOWER_FITTED_TARGETS_FILE,
            mmap_mode="r",
        )
    records: list[dict[str, Any]] = []
    for trade_date, positions in output.groupby(
        "trade_date", sort=False
    ).groups.items():
        locations = np.asarray(positions, dtype=np.int64)
        indices = output.iloc[locations]["sample_index"].to_numpy(np.int64)
        daily_styles = np.asarray(style_matrix[indices], dtype=np.float64)
        style_ranks = pd.DataFrame(daily_styles).rank(
            method="average"
        ).to_numpy(np.float64)
        record: dict[str, Any] = {
            "year": year,
            "trade_date": str(trade_date),
            "rows": len(locations),
            "complete_style_rate": float(
                np.asarray(style_valid[indices]).all(axis=1).mean()
            ),
        }
        factors = [
            ("raw", final_raw[locations]),
            ("effective", final_effective[locations]),
        ]
        if dual_tower_enabled():
            factors = [
                ("alpha", final_alpha[locations]),
                ("style", final_style[locations]),
                ("total", final_total[locations]),
            ]
            record["alpha_style_pearson"] = float(
                np.corrcoef(final_alpha[locations], final_style[locations])[0, 1]
            ) if float(np.std(final_style[locations], ddof=0)) > 0 else np.nan
            record["dual_additivity_max_abs_error"] = float(
                np.max(
                    np.abs(
                        final_total[locations]
                        - final_alpha[locations]
                        - final_style[locations]
                    )
                )
            )
            record["inference_total_precision"] = ",".join(  # 按业务键对齐，避免依赖原始行序
                sorted({
                    str(audit["stored_total_precision"])
                    for audit in dual_inference_audits.values()
                })
            )
            record["inference_stored_vs_canonical_max_abs_error"] = max(
                float(audit["stored_vs_canonical_max_abs_error"])
                for audit in dual_inference_audits.values()
            )
        target_views: dict[str, np.ndarray] = {}
        if dual_tower_enabled():
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
        for prefix, daily_factor in factors:
            factor_std = float(np.std(daily_factor, ddof=0))
            record[f"{prefix}_cross_section_std"] = factor_std
            if factor_std <= np.finfo(np.float64).eps:
                pearson = np.full(STYLE_COUNT, np.nan, dtype=np.float64)
                spearman = np.full(STYLE_COUNT, np.nan, dtype=np.float64)
                factor_rank = np.full(len(daily_factor), np.nan, dtype=np.float64)
            else:
                factor_rank = pd.Series(daily_factor).rank(
                    method="average"
                ).to_numpy(np.float64)
                pearson = np.corrcoef(
                    np.column_stack([daily_factor, daily_styles]), rowvar=False
                )[0, 1:]
                spearman = np.corrcoef(
                    np.column_stack([factor_rank, style_ranks]), rowvar=False
                )[0, 1:]
                if (
                    not np.isfinite(pearson).all()
                    or not np.isfinite(spearman).all()
                ):
                    raise FloatingPointError(
                        f"{trade_date} {prefix} 预测风格相关性不可计算"
                    )
            if dual_tower_enabled():
                for target_name, daily_target in target_views.items():
                    valid_target = np.isfinite(daily_target)  # 排除 NaN 与 Inf
                    if (
                        factor_std <= np.finfo(np.float64).eps
                        or int(valid_target.sum()) < 2
                        or float(np.std(daily_target[valid_target], ddof=0))
                        <= np.finfo(np.float64).eps
                    ):
                        pearson_ic = np.nan
                        rank_ic = np.nan
                    else:
                        pearson_ic = float(np.corrcoef(
                            daily_factor[valid_target],
                            daily_target[valid_target],
                        )[0, 1])
                        rank_ic = float(np.corrcoef(
                            pd.Series(daily_factor[valid_target]).rank(
                                method="average"
                            ),
                            pd.Series(daily_target[valid_target]).rank(
                                method="average"
                            ),
                        )[0, 1])
                    record[f"{prefix}_{target_name}_pearson_ic"] = pearson_ic
                    record[f"{prefix}_{target_name}_rank_ic"] = rank_ic
            record.update(
                {
                    f"{prefix}_max_abs_style_pearson": float(np.nanmax(np.abs(pearson))) if np.isfinite(pearson).any() else np.nan,
                    f"{prefix}_mean_abs_style_pearson": float(np.nanmean(np.abs(pearson))) if np.isfinite(pearson).any() else np.nan,
                    f"{prefix}_max_abs_style_spearman": float(np.nanmax(np.abs(spearman))) if np.isfinite(spearman).any() else np.nan,
                    f"{prefix}_mean_abs_style_spearman": float(np.nanmean(np.abs(spearman))) if np.isfinite(spearman).any() else np.nan,
                }
            )
            if industry_codes is not None:
                daily_industries = np.asarray(
                    industry_codes[indices], dtype=np.int16
                )
                if factor_std > np.finfo(np.float64).eps:
                    joint_r_squared, _ = style_data.joint_barra_r_squared_numpy(  # 按股票与交易日业务键对齐
                        daily_factor, daily_styles, daily_industries
                    )
                else:
                    joint_r_squared = np.nan
                industry_means = pd.Series(daily_factor).groupby(
                    daily_industries, sort=False
                ).mean()
                record[f"{prefix}_joint_barra_r_squared"] = joint_r_squared
                record[f"{prefix}_max_abs_industry_mean"] = float(
                    industry_means.abs().max()
                )
            for position, column in enumerate(style_data.STYLE_COLUMNS):
                record[f"{prefix}_pearson_{column}"] = float(
                    pearson[position]
                )
                record[f"{prefix}_spearman_{column}"] = float(
                    spearman[position]
                )
        records.append(record)

    output["prediction_ensemble_unprojected"] = final_raw.astype(np.float32)
    if dual_tower_enabled():
        output["prediction_ensemble_alpha"] = final_alpha.astype(np.float32)
        output["prediction_ensemble_style"] = final_style.astype(np.float32)
        output["prediction_ensemble_total_return"] = final_total.astype(
            np.float32
        )
    if PROJECT_MODEL_OUTPUT or strict_barra_output_enabled():
        output["prediction_ensemble_projected"] = final_effective.astype(
            np.float32
        )
    published = (
        final_effective if strict_barra_output_enabled() else final_raw
    ).astype(np.float32)
    output["prediction_ensemble_raw"] = published
    output["prediction_ensemble"] = published
    output["forecast_year"] = year
    output["factor_role"] = factor_role()  # 确定实验输出因子口径
    sequence_data.atomic_write_parquet(  # 原子落盘，避免半成品
        output,
        prediction_root / f"vivit_predictions.{year}.parquet",
    )
    return len(output), pd.DataFrame(records)


def style_diagnostics_root(quality_root: Path) -> Path:
    """返回当前 run 的风格暴露诊断目录。"""
    return quality_root / "style_neutral_training"


def finalize_run(wall_seconds: float) -> pd.DataFrame:
    """核验年度任务并生成集成预测、诊断、清单和报告。"""
    model_root, _, quality_root = base.run_paths(RUN_NAME)
    summaries: list[dict[str, Any]] = []
    for year in FORECAST_YEARS:
        for seed in SEEDS:
            path = model_root / f"forecast_{year}" / f"seed_{seed}" / "completed.json"
            if not path.exists():
                raise FileNotFoundError(path)
            summaries.append(json.loads(path.read_text(encoding="utf-8"))["summary"])
    summary = pd.DataFrame(summaries).sort_values(
        ["forecast_year", "seed"], kind="stable"
    )
    validation_score_collapse_tasks = (
        int(summary["validation_score_collapsed"].fillna(False).sum())
        if "validation_score_collapsed" in summary.columns
        else 0
    )
    sequence_data.atomic_write_parquet(  # 原子落盘，避免半成品
        summary, quality_root / "vivit_training_summary.parquet"
    )
    if adversarial_barra_enabled():
        probe_frames: list[pd.DataFrame] = []
        for year in FORECAST_YEARS:
            for seed in SEEDS:
                probe_path = (
                    model_root / f"forecast_{year}" / f"seed_{seed}"
                    / "adversarial_probe_diagnostics.parquet"
                )
                if not probe_path.exists():
                    raise FileNotFoundError(probe_path)
                probe_frames.append(pd.read_parquet(probe_path))
        sequence_data.atomic_write_parquet(  # 原子落盘，避免半成品
            pd.concat(probe_frames, ignore_index=True),
            quality_root / "adversarial_probe_diagnostics.parquet",
        )

    factor_counts: dict[int, int] = {}
    diagnostics: list[pd.DataFrame] = []
    for year in FORECAST_YEARS:
        factor_counts[year], yearly = build_ensemble(year)
        if not yearly.empty:
            diagnostics.append(yearly)
    if not diagnostics:
        raise ValueError("没有可写入的样本外预测风格诊断")
    sequence_data.atomic_write_parquet(  # 原子落盘，避免半成品
        pd.concat(diagnostics, ignore_index=True),
        style_diagnostics_root(quality_root)
        / "prediction_style_exposure_diagnostics.parquet",
    )
    if dual_tower_enabled():
        sequence_data.atomic_write_parquet(  # 原子落盘，避免半成品
            pd.concat(diagnostics, ignore_index=True),
            quality_root / "dual_tower_decomposition_diagnostics.parquet",
        )

    sequence = sequence_data.require_aligned_sequence_store()
    style_manifest = style_data.require_style_neutral_store()
    industry_manifest = (
        style_data.require_industry_sidecar()
        if industry_sidecar_required() else None
    )
    full_target_manifest = (
        style_data.require_full_barra_target_sidecar()
        if barra_conditioning_enabled() or dual_tower_enabled() else None
    )
    dual_target_manifest = (
        style_data.require_dual_tower_target_sidecar()
        if dual_tower_enabled() else None
    )
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
    if dual_tower_enabled():
        manifest["full_barra_target_sidecar_signature"] = (
            full_target_manifest["signature"]
        )
        manifest["dual_tower_target_sidecar_signature"] = (
            dual_target_manifest["signature"]
        )
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
    if barra_conditioning_enabled():
        manifest["full_barra_target_sidecar_signature"] = (
            full_target_manifest["signature"]
        )
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
    if strict_barra_output_enabled():
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
    sequence_data.atomic_write_json(manifest, quality_root / "a2_training_manifest.json")  # 原子落盘，避免半成品
    return summary


# %% [markdown]
# ## 四 GPU 调度和 CLI 入口


# %%


def measure_daily_score_dispersion(
    predictions: torch.Tensor,
    *,
    date_id: int,
) -> dict[str, torch.Tensor]:
    """检查单日原始分数是否仍有可用于排序的截面离散度。"""

    raw = predictions.detach().float().reshape(-1)
    if len(raw) < 2:
        raise ValueError("单日预测至少需要两个样本")
    if not torch.isfinite(raw).all():
        raise FloatingPointError("模型输出包含非有限值")

    centered = raw - raw.mean()
    total_sum_squares = centered.square().sum()
    score_std = torch.sqrt(total_sum_squares / len(raw))  # 计算截面标准差
    if (
        not torch.isfinite(total_sum_squares)
        or not torch.isfinite(score_std)
        or total_sum_squares <= torch.finfo(torch.float32).eps
    ):
        raise DegenerateValidationScoreError(
            date_id=date_id,
            rows=len(raw),
            score_std=float(score_std.item()),
            total_sum_squares=float(total_sum_squares.item()),
        )
    return {
        "score_std": score_std,
        "total_sum_squares": total_sum_squares,
    }

def measure_style_correlations(
    predictions: torch.Tensor,
    style_exposures: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """只读衡量单日预测与十个风格的 Pearson 相关性。"""

    raw = predictions.detach().float().reshape(-1)
    styles = style_exposures.float()
    if styles.shape != (len(raw), STYLE_COUNT):
        raise ValueError(f"十风格矩阵形状异常：{tuple(styles.shape)}")
    if len(raw) < 3:
        raise ValueError("单日股票数不足以计算风格相关性")
    if not torch.isfinite(raw).all() or not torch.isfinite(styles).all():
        raise FloatingPointError("预测或十风格暴露含 NaN/Inf")

    centered_factor = raw - raw.mean()
    centered_styles = styles - styles.mean(dim=0, keepdim=True)
    numerator = centered_styles.transpose(0, 1) @ centered_factor
    denominator = torch.sqrt(  # 计算截面标准差
        centered_styles.square().sum(dim=0)
        * centered_factor.square().sum()
    )
    correlations = numerator / denominator
    if not torch.isfinite(correlations).all():
        raise FloatingPointError("模型输出的风格相关性不可计算")
    return {
        "max_abs_style_pearson": correlations.abs().max(),
        "mean_abs_style_pearson": correlations.abs().mean(),
    }

def summarize_adversarial_predictions(
    style_predictions: np.ndarray,
    style_targets: np.ndarray,
    style_valid: np.ndarray,
    industry_logits: np.ndarray,
    industry_codes: np.ndarray,
    industry_class_weights: np.ndarray | None = None,
) -> dict[str, float]:
    """汇总对抗头的风格回归和行业分类诊断。"""
    mask = np.asarray(style_valid, dtype=bool)
    predicted_styles = np.asarray(style_predictions, dtype=np.float64)
    target_styles = np.asarray(style_targets, dtype=np.float64)
    if predicted_styles.shape != target_styles.shape or mask.shape != target_styles.shape:
        raise ValueError("对抗风格诊断形状不一致")
    squared_errors = (predicted_styles - target_styles) ** 2
    style_r_squared: list[float] = []
    for column in range(STYLE_COUNT):
        valid = mask[:, column]
        values = target_styles[valid, column]
        denominator = float(((values - values.mean()) ** 2).sum())
        if len(values) < 2 or denominator <= np.finfo(np.float64).eps:
            raise ValueError("验证期风格标签不足或为常数")
        numerator = float(squared_errors[valid, column].sum())
        style_r_squared.append(1.0 - numerator / denominator)

    codes = np.asarray(industry_codes, dtype=np.int64).reshape(-1)
    logits = np.asarray(industry_logits, dtype=np.float64)
    known = codes > 0
    targets = codes[known] - 1
    known_logits = logits[known]
    predicted = known_logits.argmax(axis=1)  # 输出行业预测类别
    shifted = known_logits - known_logits.max(axis=1, keepdims=True)
    negative_log_likelihood = (
        np.log(np.exp(shifted).sum(axis=1))
        - shifted[np.arange(len(targets)), targets]
    )
    class_weights = np.asarray(
        industry_class_weights
        if industry_class_weights is not None
        else np.ones(INDUSTRY_COUNT),
        dtype=np.float64,
    )
    if (
        class_weights.shape != (INDUSTRY_COUNT,)
        or not np.isfinite(class_weights).all()
        or (class_weights <= 0).any()
    ):
        raise ValueError("验证期行业类别权重形状或数值异常")
    sample_weights = class_weights[targets]
    normalized_ce = float(
        np.average(negative_log_likelihood, weights=sample_weights)
        / math.log(INDUSTRY_COUNT)
    )
    confusion = np.bincount(
        targets * INDUSTRY_COUNT + predicted,
        minlength=INDUSTRY_COUNT**2,
    ).reshape(INDUSTRY_COUNT, INDUSTRY_COUNT)
    support = confusion.sum(axis=1)
    present = support > 0
    recall = np.divide(  # 仅对有效分母执行归一化
        np.diag(confusion), support,
        out=np.zeros(INDUSTRY_COUNT, dtype=np.float64), where=present,
    )
    predicted_count = confusion.sum(axis=0)
    precision = np.divide(  # 仅对有效分母执行归一化
        np.diag(confusion), predicted_count,
        out=np.zeros(INDUSTRY_COUNT, dtype=np.float64),
        where=predicted_count > 0,
    )
    f1 = np.divide(  # 仅对有效分母执行归一化
        2 * precision * recall, precision + recall,
        out=np.zeros(INDUSTRY_COUNT, dtype=np.float64),
        where=(precision + recall) > 0,
    )
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

def deterministic_probe_positions(
    rows: int, *, year: int, seed: int, date_id: int
) -> np.ndarray:
    """按稳定样本键确定性抽取每日 probe 样本。"""
    if rows <= PROBE_MAX_SAMPLES_PER_DATE:
        return np.arange(rows, dtype=np.int64)
    random_seed = np.random.SeedSequence(
        [year, seed, date_id]
    ).generate_state(1)[0]
    generator = np.random.default_rng(random_seed)  # 创建可复现随机源
    return np.sort(
        generator.choice(
            rows, size=PROBE_MAX_SAMPLES_PER_DATE, replace=False
        ).astype(np.int64)
    )

def chronological_probe_split(date_ids: np.ndarray) -> dict[str, np.ndarray]:
    """按日期顺序切分 probe 训练、验证和测试区间。"""
    dates = np.unique(np.asarray(date_ids, dtype=np.int64))  # 压缩当日行业编码
    if len(dates) < 10:
        raise ValueError("冻结 probe 至少需要10个验证交易日")
    train_count = max(1, int(np.floor(0.4 * len(dates))))
    validation_count = max(1, int(np.floor(0.1 * len(dates))))
    if train_count + validation_count >= len(dates):
        raise ValueError("冻结 probe 日期不足以划分测试区间")
    boundaries = {
        "train": dates[:train_count],
        "validation": dates[train_count:train_count + validation_count],
        "test": dates[train_count + validation_count:],
    }
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
    """冻结主模型并提取 CLS、风格和行业 probe 数据。"""
    model.eval()  # 关闭 dropout 等训练行为
    parts: dict[str, list[np.ndarray]] = {
        "representations": [],
        "styles": [],
        "style_valid": [],
        "industry_codes": [],
        "date_ids": [],
    }
    for item in StyleCudaPrefetcher(
        loader, device, bool(USE_CUDA_PREFETCH)
    ):
        (
            batch, videos, frame_mask, _targets, _original, _residual,
            _fitted, styles, style_valid, industry_codes,
        ) = item
        date_id = int(batch["date_ids"][0])
        if int(batch["date_ids"].min()) != int(batch["date_ids"].max()):
            raise AssertionError("probe batch 混入多个交易日")
        with base.autocast_context(device):
            (  # 执行当前 batch 前向传播
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
        if representation is None:
            raise ValueError("No.3 probe 无法取得 CLS 表征")
        positions = deterministic_probe_positions(  # 用探针测量表征中的风险信息
            len(videos), year=year, seed=seed, date_id=date_id
        )
        index = torch.from_numpy(positions).to(device)
        parts["representations"].append(
            representation.float().index_select(0, index).cpu().numpy()
        )
        parts["styles"].append(
            styles.float().index_select(0, index).cpu().numpy()
        )
        parts["style_valid"].append(
            style_valid.index_select(0, index).cpu().numpy()
        )
        parts["industry_codes"].append(
            industry_codes.index_select(0, index).cpu().numpy()
        )
        parts["date_ids"].append(
            np.full(len(positions), date_id, dtype=np.int64)
        )
    if not parts["representations"]:
        raise ValueError("冻结 probe 没有抽取到验证样本")
    return {
        name: np.concatenate(values, axis=0)
        for name, values in parts.items()
    }

def _style_probe_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    valid: np.ndarray,
) -> tuple[list[float], float]:
    """处理连续风格数据、有效掩码或相应诊断。"""
    squared = (
        np.asarray(predictions, dtype=np.float64)
        - np.asarray(targets, dtype=np.float64)
    ) ** 2
    mask = np.asarray(valid, dtype=bool)
    r_squared: list[float] = []
    for column in range(STYLE_COUNT):
        selected = mask[:, column]
        values = targets[selected, column].astype(np.float64)
        denominator = float(((values - values.mean()) ** 2).sum())
        if len(values) < 2 or denominator <= np.finfo(np.float64).eps:
            raise ValueError("probe 测试集风格标签不足或为常数")
        r_squared.append(
            1.0 - float(squared[selected, column].sum()) / denominator
        )
    return r_squared, float(squared[mask].mean())

def _industry_probe_metrics(
    logits: np.ndarray, industry_codes: np.ndarray
) -> dict[str, float]:
    """处理行业代码或行业诊断，并保留 UNKNOWN 口径。"""
    codes = np.asarray(industry_codes, dtype=np.int64).reshape(-1)
    known = codes > 0
    targets = codes[known] - 1
    predicted = np.asarray(logits)[known].argmax(axis=1)  # 输出行业预测类别
    confusion = np.bincount(
        targets * INDUSTRY_COUNT + predicted,
        minlength=INDUSTRY_COUNT**2,
    ).reshape(INDUSTRY_COUNT, INDUSTRY_COUNT)
    support = confusion.sum(axis=1)
    present = support > 0
    true_positive = np.diag(confusion).astype(np.float64)
    recall = np.divide(  # 仅对有效分母执行归一化
        true_positive, support,
        out=np.zeros(INDUSTRY_COUNT), where=present,
    )
    predicted_count = confusion.sum(axis=0)
    precision = np.divide(  # 仅对有效分母执行归一化
        true_positive, predicted_count,
        out=np.zeros(INDUSTRY_COUNT), where=predicted_count > 0,
    )
    f1 = np.divide(  # 仅对有效分母执行归一化
        2 * precision * recall, precision + recall,
        out=np.zeros(INDUSTRY_COUNT), where=(precision + recall) > 0,
    )
    return {
        "accuracy": float((predicted == targets).mean()),
        "balanced_accuracy": float(recall[present].mean()),
        "macro_f1": float(f1[present].mean()),
        "unknown_rate": float((codes == 0).mean()),
    }

def fit_linear_style_probe(
    representations: np.ndarray,
    styles: np.ndarray,
    style_valid: np.ndarray,
    split: dict[str, np.ndarray],
    device: torch.device,
) -> np.ndarray:
    """拟合线性岭回归 probe，衡量 CLS 中可恢复的风格信息。"""
    train = split["train"]
    test = split["test"]
    x_train = torch.from_numpy(representations[train]).float().to(device)
    x_test = torch.from_numpy(representations[test]).float().to(device)
    x_train = torch.cat(
        [x_train, torch.ones((len(x_train), 1), device=device)], dim=1
    )
    x_test = torch.cat(
        [x_test, torch.ones((len(x_test), 1), device=device)], dim=1
    )
    identity = torch.eye(x_train.shape[1], device=device)  # 构造岭回归惩罚项
    identity[-1, -1] = 0.0
    predictions: list[np.ndarray] = []
    for column in range(STYLE_COUNT):
        valid = torch.from_numpy(style_valid[train, column]).bool().to(device)
        target = torch.from_numpy(styles[train, column]).float().to(device)
        design = x_train[valid]
        coefficients = torch.linalg.solve(  # 求解岭回归探针
            design.T @ design + PROBE_RIDGE_ALPHA * identity,
            design.T @ target[valid],
        )
        predictions.append((x_test @ coefficients).cpu().numpy())
    return np.column_stack(predictions).astype(np.float32)

class _ProbeNetwork(torch.nn.Module):
    def __init__(self, output_dim: int, nonlinear: bool) -> None:
        super().__init__()
        if nonlinear:
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(base.MODEL_CONFIG.embedding_dim, 128),
                torch.nn.GELU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(128, output_dim),
            )
        else:
            self.layers = torch.nn.Linear(
                base.MODEL_CONFIG.embedding_dim, output_dim
            )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """执行 _ProbeNetwork 的前向计算并返回约定张量。"""
        return self.layers(values)

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
    """训练线性或 MLP probe，并在独立日期区间评估。"""
    torch.manual_seed(seed)
    network = _ProbeNetwork(output_dim, nonlinear).to(device)
    optimizer = torch.optim.AdamW(
        network.parameters(), lr=PROBE_LEARNING_RATE,
        weight_decay=PROBE_WEIGHT_DECAY,
    )
    class_weights = None
    if classification:
        train_codes = targets[split["train"]].astype(np.int64)
        counts = np.bincount(
            train_codes[train_codes > 0] - 1, minlength=INDUSTRY_COUNT
        ).astype(np.float64)
        if (counts <= 0).any():
            raise ValueError("probe 训练集未覆盖全部32行业")
        weights = np.reciprocal(np.sqrt(counts))  # 按截面样本数平方根加权
        weights /= weights.mean()
        class_weights = torch.from_numpy(weights.astype(np.float32)).to(device)

    def loss_for(indices: np.ndarray, *, training: bool) -> torch.Tensor:
        """按 probe 类型返回连续风格回归或行业分类损失。"""
        if not len(indices):
            raise ValueError("probe split 为空")
        total = torch.zeros((), dtype=torch.float32, device=device)
        rows = 0
        order = indices.copy()
        if training:
            generator = np.random.default_rng(seed + epoch)  # 创建可复现随机源
            generator.shuffle(order)
        for start in range(0, len(order), PROBE_BATCH_SIZE):
            selected = order[start:start + PROBE_BATCH_SIZE]
            x = torch.from_numpy(representations[selected]).float().to(device)
            output = network(x).float()
            if classification:
                codes = torch.from_numpy(targets[selected]).long().to(device)
                known = codes > 0
                if not known.any():
                    continue
                batch_loss = torch.nn.functional.cross_entropy(
                    output[known], codes[known] - 1, weight=class_weights
                ) / math.log(INDUSTRY_COUNT)
                count = int(known.sum())
            else:
                y = torch.from_numpy(targets[selected]).float().to(device)
                mask = torch.from_numpy(valid[selected]).bool().to(device)
                batch_loss = (output - y).square()[mask].mean()
                count = int(mask.sum())
            if training:
                optimizer.zero_grad(set_to_none=True)  # 清空上一批梯度
                batch_loss.backward()  # 反向传播当前损失
                optimizer.step()  # 更新参数或学习率状态
            total += batch_loss.detach() * count
            rows += count
        if not rows:
            raise ValueError("probe split 没有有效监督")
        return total / rows

    best_loss = np.inf
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    for epoch in range(1, PROBE_MAX_EPOCHS + 1):
        network.train()  # 启用训练模式
        loss_for(split["train"], training=True)  # 按实验口径计算可微损失
        network.eval()  # 关闭 dropout 等训练行为
        with torch.no_grad():
            validation_loss = float(
                loss_for(split["validation"], training=False)
            )
        if validation_loss < best_loss - 1e-7:
            best_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in network.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= PROBE_PATIENCE:
            break
    if best_state is None:
        raise RuntimeError("probe 未生成最佳参数")
    network.load_state_dict(best_state)
    network.eval()  # 关闭 dropout 等训练行为
    outputs: list[np.ndarray] = []
    test = split["test"]
    with torch.no_grad():
        for start in range(0, len(test), PROBE_BATCH_SIZE):
            selected = test[start:start + PROBE_BATCH_SIZE]
            x = torch.from_numpy(representations[selected]).float().to(device)
            outputs.append(network(x).float().cpu().numpy())
    return np.concatenate(outputs, axis=0)

def run_adversarial_probes(
    model: torch.nn.Module,
    validation_loader: DataLoader,
    device: torch.device,
    *,
    year: int,
    seed: int,
) -> pd.DataFrame:
    """对最佳 checkpoint 运行冻结 probe 并保存年度诊断。"""
    dataset = extract_probe_dataset(  # 用探针测量表征中的风险信息
        model, validation_loader, device, year=year, seed=seed
    )
    split = chronological_probe_split(dataset["date_ids"])  # 用探针测量表征中的风险信息
    test = split["test"]
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
    records: list[dict[str, Any]] = []
    linear_style = fit_linear_style_probe(
        dataset["representations"], dataset["styles"],
        dataset["style_valid"], split, device,
    )
    mlp_style = fit_torch_probe(
        dataset["representations"], dataset["styles"],
        dataset["style_valid"], split, device,
        output_dim=STYLE_COUNT, nonlinear=True, classification=False,
        seed=seed + 1000,
    )
    for probe_name, predictions in (
        ("linear_ridge", linear_style), ("mlp", mlp_style)
    ):
        r_squared, mse = _style_probe_metrics(  # 用探针测量表征中的风险信息
            predictions, dataset["styles"][test],
            dataset["style_valid"][test],
        )
        for column, value in zip(style_data.STYLE_COLUMNS, r_squared):
            records.append({
                **metadata, "probe": probe_name, "target_kind": "style",
                "target": column, "r_squared": value, "mse": np.nan,
            })
        records.append({
            **metadata, "probe": probe_name, "target_kind": "style_summary",
            "target": "all_styles", "r_squared": float(np.mean(r_squared)),
            "max_r_squared": float(np.max(r_squared)), "mse": mse,
        })

    for probe_name, nonlinear, offset in (
        ("linear_softmax", False, 2000), ("mlp", True, 3000)
    ):
        logits = fit_torch_probe(
            dataset["representations"], dataset["industry_codes"], None,
            split, device, output_dim=INDUSTRY_COUNT, nonlinear=nonlinear,
            classification=True, seed=seed + offset,
        )
        metrics = _industry_probe_metrics(  # 用探针测量表征中的风险信息
            logits, dataset["industry_codes"][test]
        )
        records.append({
            **metadata, "probe": probe_name, "target_kind": "industry",
            "target": "known_industries", **metrics,
        })
    return pd.DataFrame(records)

def validate_outputs() -> pd.DataFrame:
    """检查 checkpoint、预测、诊断和 manifest 是否完整一致。"""
    _, prediction_root, quality_root = base.run_paths(RUN_NAME)
    manifest = json.loads(
        (quality_root / "a2_training_manifest.json").read_text(encoding="utf-8")
    )
    expected_mode = (RESIDUALIZE_TARGET, PROJECT_MODEL_OUTPUT)
    stored_mode = (
        bool(manifest.get("residualize_target")),
        bool(manifest.get("project_model_output")),
    )
    if stored_mode != expected_mode:
        raise ValueError(
            f"run={RUN_NAME!r} 的 manifest 开关为 {stored_mode}，"
            f"当前 CLI 开关为 {expected_mode}；请使用匹配的 run name/参数"
        )
    stored_lambda = float(manifest.get("raw_barra_r2_lambda", 0.0))
    if stored_lambda != RAW_BARRA_R2_LAMBDA:
        raise ValueError(
            f"run={RUN_NAME!r} 的 manifest λ={stored_lambda}，"
            f"当前 CLI λ={RAW_BARRA_R2_LAMBDA}"
        )
    if bool(manifest.get("adversarial_barra", False)) != adversarial_barra_enabled():
        raise ValueError(
            f"run={RUN_NAME!r} 的 No.3 开关与当前 CLI 不一致"
        )
    stored_conditioning = manifest.get("barra_conditioning")
    if barra_conditioning_enabled():
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
        if stored_conditioning != expected_conditioning:
            raise ValueError("No.4 manifest 与当前条件门控参数不一致")
        full_target = style_data.require_full_barra_target_sidecar()
        if manifest.get("full_barra_target_sidecar_signature") != (
            full_target["signature"]
        ):
            raise ValueError("No.4 manifest 的完整标签 sidecar 签名不一致")
    elif stored_conditioning is not None:
        raise ValueError("当前 CLI 未开启 No.4，但 manifest 含条件门控配置")
    stored_dual = manifest.get("barra_dual_tower")
    if dual_tower_enabled():
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
        if stored_dual != expected_dual:
            raise ValueError("No.6 manifest 与当前双塔参数不一致")
        full_target = style_data.require_full_barra_target_sidecar()
        dual_target = style_data.require_dual_tower_target_sidecar()
        if manifest.get("full_barra_target_sidecar_signature") != full_target["signature"]:
            raise ValueError("No.6 完整残差 sidecar 签名不一致")
        if manifest.get("dual_tower_target_sidecar_signature") != dual_target["signature"]:
            raise ValueError("No.6 fitted-target sidecar 签名不一致")
    elif stored_dual is not None:
        raise ValueError("当前 CLI 未开启 No.6，但 manifest 含双塔配置")

    stored_strict = manifest.get("strict_barra_output")
    if strict_barra_output_enabled():
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
        if stored_strict != expected_strict:
            raise ValueError("严格 Barra 输出 manifest 与当前参数不一致")
    elif stored_strict is not None:
        raise ValueError("当前 CLI 未开启严格输出，但 manifest 含对应配置")
    if adversarial_barra_enabled():
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
        if manifest.get("adversarial_config") != expected_adversarial_config:
            raise ValueError("No.3 manifest 与当前对抗或 probe 参数不一致")
        probe_path = quality_root / "adversarial_probe_diagnostics.parquet"
        if not probe_path.exists():
            raise FileNotFoundError(probe_path)
        probes = pd.read_parquet(probe_path)
        expected_tasks = len(FORECAST_YEARS) * len(SEEDS)
        observed_tasks = probes[["forecast_year", "seed"]].drop_duplicates()  # 业务键去重
        if len(observed_tasks) != expected_tasks:
            raise AssertionError("No.3 probe 不是8年度×3 seed")
        # 构造去重集合，并保存到 `required_probe_kinds`。
        required_probe_kinds = {"style", "style_summary", "industry"}
        if not required_probe_kinds.issubset(set(probes["target_kind"])):
            raise AssertionError("No.3 probe 诊断类型不完整")
        style_probes = set(
            probes.loc[
                probes["target_kind"].eq("style_summary"), "probe"
            ]
        )
        industry_probes = set(
            probes.loc[probes["target_kind"].eq("industry"), "probe"]
        )
        if style_probes != {"linear_ridge", "mlp"}:
            raise AssertionError("No.3 风格 probe 类型不完整")
        if industry_probes != {"linear_softmax", "mlp"}:
            raise AssertionError("No.3 行业 probe 类型不完整")


    summary = pd.read_parquet(quality_root / "vivit_training_summary.parquet")
    if len(summary) != len(FORECAST_YEARS) * len(SEEDS):
        raise AssertionError("训练汇总不是 8 年×3 seed")
    records: list[dict[str, Any]] = []
    base_columns = [
        "sample_index",
        "prediction_ensemble_unprojected",
        "prediction_ensemble_raw",
        "prediction_ensemble",
    ]
    if dual_tower_enabled():
        base_columns.extend(
            [
                "prediction_ensemble_alpha",
                "prediction_ensemble_style",
                "prediction_ensemble_total_return",
            ]
        )
    columns = [
        *base_columns,
        *(
            ["prediction_ensemble_projected"]
            if PROJECT_MODEL_OUTPUT or strict_barra_output_enabled()
            else []
        ),
    ]
    for year in FORECAST_YEARS:
        split = sequence_data.load_sequence_split(year)
        expected = len(split["prediction_indices"])
        path = prediction_root / f"vivit_predictions.{year}.parquet"
        if not expected:
            if path.exists():
                raise AssertionError(f"{year} prediction 为空但存在预测文件")
            records.append({"year": year, "rows": 0, "status": "通过"})
            continue
        frame = pd.read_parquet(path, columns=columns)
        if len(frame) != expected or not np.array_equal(
            frame["sample_index"].to_numpy(np.int64), split["prediction_indices"]
        ):
            raise AssertionError(f"{year} 预测索引不一致")
        if not np.isfinite(frame[columns[1:]].to_numpy(np.float64)).all():
            raise AssertionError(f"{year} 预测列含 NaN/Inf")
        if dual_tower_enabled():
            alpha = frame["prediction_ensemble_alpha"].to_numpy(np.float64)
            style = frame["prediction_ensemble_style"].to_numpy(np.float64)
            total = frame["prediction_ensemble_total_return"].to_numpy(np.float64)
            if not np.array_equal(
                frame["prediction_ensemble"].to_numpy(),
                frame["prediction_ensemble_alpha"].to_numpy(),
            ):
                raise AssertionError(f"{year} No.6 正式主列必须等于 Alpha")
            if not np.allclose(total, alpha + style, rtol=1e-6, atol=1e-6):
                raise AssertionError(f"{year} No.6 集成不满足 Alpha+Style=Total")
        if strict_barra_output_enabled():
            if np.array_equal(
                frame["prediction_ensemble_unprojected"].to_numpy(),
                frame["prediction_ensemble_raw"].to_numpy(),
            ):
                raise AssertionError(f"{year} 严格中性主因子不能等于投影前分数")
            if not np.array_equal(
                frame["prediction_ensemble_projected"].to_numpy(),
                frame["prediction_ensemble_raw"].to_numpy(),
            ):
                raise AssertionError(f"{year} 严格中性主因子必须使用投影后分数")
        elif not np.array_equal(
            frame["prediction_ensemble_unprojected"].to_numpy(),
            frame["prediction_ensemble_raw"].to_numpy(),
        ):
            raise AssertionError(f"{year} 非严格模式主因子必须使用未投影分数")
        if not np.array_equal(
            frame["prediction_ensemble_raw"].to_numpy(),
            frame["prediction_ensemble"].to_numpy(),
        ):
            raise AssertionError(f"{year} raw 与 notebook03 输入列必须完全相同")
        if PROJECT_MODEL_OUTPUT and np.array_equal(
            frame["prediction_ensemble_projected"].to_numpy(),
            frame["prediction_ensemble"].to_numpy(),
        ):
            raise AssertionError(f"{year} No.1 投影审计列不应覆盖未投影主因子")
        records.append({"year": year, "rows": len(frame), "status": "通过"})

    diagnostics = pd.read_parquet(
        style_diagnostics_root(quality_root)
        / "prediction_style_exposure_diagnostics.parquet"
    )
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
    if not required.issubset(diagnostics.columns):
        raise AssertionError(
            f"最终预测缺少风格相关性审计列：{sorted(required - set(diagnostics.columns))}"
        )
    if not np.isfinite(diagnostics[list(required)].to_numpy()).all():
        raise AssertionError("最终预测的风格相关性审计含 NaN/Inf")
    if PROJECT_MODEL_OUTPUT or strict_barra_output_enabled():
        maximum = float(diagnostics["effective_max_abs_style_pearson"].max())
        if maximum > 5e-5:
            raise AssertionError(
                f"投影后预测未达到数值正交：max |Pearson|={maximum:.3e}"
            )
        if strict_barra_output_enabled():
            maximum_r_squared = float(
                diagnostics["effective_joint_barra_r_squared"].max()
            )
            if maximum_r_squared > 1e-6:
                raise AssertionError("严格中性预测的联合 Barra R² 未接近零")
            maximum_industry_mean = float(
                diagnostics["effective_max_abs_industry_mean"].max()
            )
            if maximum_industry_mean > 5e-5:
                raise AssertionError(
                    "严格中性预测的行业组均值未接近零"
                )
    return pd.DataFrame(records)

def write_report() -> Path:
    """汇总训练、暴露、probe 和预测结果并写入实验报告。"""
    _, _, quality_root = base.run_paths(RUN_NAME)
    summary = pd.read_parquet(quality_root / "vivit_training_summary.parquet")
    manifest = json.loads(
        (quality_root / "a2_training_manifest.json").read_text(encoding="utf-8")
    )
    audit = validate_outputs()
    diagnostics = pd.read_parquet(
        style_diagnostics_root(quality_root)
        / "prediction_style_exposure_diagnostics.parquet"
    )
    yearly = summary.groupby("forecast_year", as_index=False).agg(  # 按约定截面口径汇总
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
    if strict_barra_output_enabled():
        yearly["mean_validation_effective_barra_r_squared"] = (  # 按约定截面口径汇总
            summary.groupby("forecast_year")[
                "best_validation_effective_barra_r_squared"
            ].mean().to_numpy()
        )
        yearly["max_validation_effective_industry_mean"] = (  # 按约定截面口径汇总
            summary.groupby("forecast_year")[
                "best_validation_effective_max_abs_industry_mean"
            ].max().to_numpy()
        )
    if adversarial_barra_enabled():
        adversarial_yearly = summary.groupby(  # 按约定截面口径汇总
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
        yearly = yearly.merge(  # 按业务键对齐，避免依赖原始行序
            adversarial_yearly, on="forecast_year", validate="one_to_one"
        )
    if dual_tower_enabled():
        prediction_style = diagnostics.groupby("year", as_index=False).agg(  # 按既定口径聚合
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
    else:
        prediction_style = diagnostics.groupby("year", as_index=False).agg(  # 按既定口径聚合
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
        if industry_sidecar_required():
            prediction_style["raw_mean_daily_joint_barra_r_squared"] = (  # 按股票与交易日业务键对齐
                diagnostics.groupby("year")[
                    "raw_joint_barra_r_squared"
                ].mean().to_numpy()
            )
            if strict_barra_output_enabled():
                prediction_style["effective_max_daily_joint_barra_r_squared"] = (  # 按股票与交易日业务键对齐
                    diagnostics.groupby("year")[
                        "effective_joint_barra_r_squared"
                    ].max().to_numpy()
                )
                prediction_style["effective_max_daily_industry_mean"] = (
                    diagnostics.groupby("year")[
                        "effective_max_abs_industry_mean"
                    ].max().to_numpy()
                )
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
    if dual_tower_enabled():
        loss_description = (
            f"{LOSS_NAME}(alpha+style, source_target) + "
            f"{DUAL_TOWER_ALPHA_LOSS_WEIGHT} × {LOSS_NAME}(alpha, full_barra_residual) + "
            f"{DUAL_TOWER_STYLE_LOSS_WEIGHT} × {LOSS_NAME}(style, full_barra_fitted) + "
            f"{RAW_BARRA_R2_LAMBDA} × R²(alpha | 十风格 + 行业)"
        )
        penalty_description = (
            "联合 R² 只约束 Alpha，且始终使用股票真实 Barra；"
            "正式 Alpha 不做训练后投影。"
        )
        industry_description = (
            f"独立进入 Alpha 条件门控和 Style 塔；当前输入模式为 "
            f"{BARRA_DUAL_TOWER_MODE}，UNKNOWN embedding 固定为零。"
        )
    elif strict_barra_output_enabled():
        loss_description = (
            f"{LOSS_NAME}(strict_projected_score, "
            "full_barra_residual_target)"
        )
        penalty_description = (
            "不使用软 R²；输出层通过等权横截面投影严格删除十风格与行业固定效应。"
        )
        industry_description = (
            f"行业 embedding 进入条件模型，真实行业同时进入输出投影；"
            f"当前条件模式为 {BARRA_CONDITIONING_MODE}。"
        )
    elif barra_conditioning_enabled():
        loss_description = (
            f"{LOSS_NAME}(raw_score, full_barra_residual_target) + "
            f"{RAW_BARRA_R2_LAMBDA} × R²(raw_score | 十风格 + 行业)"
        )
        penalty_description = (
            "Barra 条件乘法门控直接生成原始 Alpha；"
            "联合 R² 始终使用未置乱的真实 Barra。"
        )
        industry_description = (
            f"行业 embedding 进入模型，UNKNOWN=0 固定为零；"
            f"当前条件模式为 {BARRA_CONDITIONING_MODE}。"
        )
    elif adversarial_barra_enabled():
        loss_description = (
            f"{LOSS_NAME}(raw_score, source_target) + masked style MSE + "
            "inverse-sqrt weighted industry CE/log(32)，对抗项经 GRL 回传。"
        )
        penalty_description = (
            f"CLS 对抗不变表征；gamma_max={ADVERSARIAL_GAMMA_MAX}，"
            f"warmup={ADVERSARIAL_WARMUP_EPOCHS}，"
            f"cosine ramp={ADVERSARIAL_RAMP_EPOCHS}。"
        )
        industry_description = (
            "32个真实行业作为训练期对抗监督；UNKNOWN 不进 CE；"
            "Barra 不进入预测 forward。"
        )
    elif raw_barra_r2_enabled():
        loss_description = (
            f"{LOSS_NAME}(raw_score, source_target) + "
            f"{RAW_BARRA_R2_LAMBDA} × R²(raw_score | 十风格 + 行业)"
        )
        penalty_description = (
            "直接惩罚原始输出的联合线性可解释度；行业采用固定效应。"
        )
        industry_description = "进入联合 R² 惩罚；不作为模型预测输入。"
    else:
        loss_description = f"{LOSS_NAME}(effective_prediction, effective_target)"
        penalty_description = "未使用 raw-score 联合 Barra R² 正则。"
        industry_description = "不进入标签回归或分数投影。"

    probe_section = ""
    if adversarial_barra_enabled():
        probes = pd.read_parquet(
            quality_root / "adversarial_probe_diagnostics.parquet"
        )
        probe_summary = probes[  # 按约定截面口径汇总
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
        probe_section = (
            "## 冻结 CLS Probe\n\n"
            + base.markdown_table(probe_summary.round(8))
            + "\n\n"
        )

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
    path = quality_root / "EXPERIMENT_REPORT.md"
    sequence_data.atomic_write_text(report, path)  # 原子落盘，避免半成品
    return path

def smoke_test(
    forecast_year: int = 2018,
    seed: int = 42,
    max_batches: int = 1,
) -> dict[str, Any]:
    """运行小批量前向、反向和验证冒烟测试，不生成正式产物。"""
    return train_task(
        forecast_year, seed, 0, max_epochs=1, max_batches=max_batches, persist=False
    )

def add_training_arguments(parser: argparse.ArgumentParser) -> None:
    """向命令行解析器注册训练后端参数。"""
    parser.add_argument("--run-name", default=RUN_NAME)
    parser.add_argument(
        "--train-batch-mode", choices=base.TRAIN_BATCH_MODES, default=TRAIN_BATCH_MODE
    )
    parser.add_argument("--loss-name", choices=base.LOSS_NAMES, default=LOSS_NAME)
    parser.add_argument("--rank-temperature", type=float, default=RANK_TEMPERATURE)
    parser.add_argument("--turnover-lambda", type=float, default=TURNOVER_LAMBDA)
    parser.add_argument("--huber-delta", type=float, default=HUBER_DELTA)
    parser.add_argument("--smooth-l1-beta", type=float, default=SMOOTH_L1_BETA)
    parser.add_argument("--hybrid-weight", type=float, default=HYBRID_WEIGHT)
    parser.add_argument(
        "--validation-metric",
        choices=base.VALIDATION_METRICS,
        default=VALIDATION_METRIC,
    )
    parser.add_argument("--workers-per-task", type=int, default=WORKERS_PER_TASK)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--compile", type=int, choices=(0, 1), default=USE_TORCH_COMPILE)
    parser.add_argument("--compile-mode", default=COMPILE_MODE)
    parser.add_argument("--fused-adam", type=int, choices=(0, 1), default=USE_FUSED_ADAM)
    parser.add_argument(
        "--cuda-prefetch", type=int, choices=(0, 1), default=USE_CUDA_PREFETCH
    )
    parser.add_argument("--overwrite", type=int, choices=(0, 1), default=OVERWRITE)
    parser.add_argument("--residualize-target", type=int, choices=(0, 1), default=1)
    parser.add_argument(
        "--project-model-output", type=int, choices=(0, 1), default=1
    )
    parser.add_argument(
        "--raw-barra-r2-lambda",
        type=float,
        default=RAW_BARRA_R2_LAMBDA,
    )
    parser.add_argument(
        "--barra-conditioning-mode",
        choices=BARRA_CONDITIONING_MODES,
        default=BARRA_CONDITIONING_MODE,
    )
    parser.add_argument(
        "--barra-output-neutralization", type=int, choices=(0, 1),
        default=int(BARRA_OUTPUT_NEUTRALIZATION),
    )
    parser.add_argument(
        "--barra-dual-tower", type=int, choices=(0, 1),
        default=int(BARRA_DUAL_TOWER),
    )
    parser.add_argument(
        "--barra-dual-tower-mode",
        choices=BARRA_DUAL_TOWER_MODES,
        default=BARRA_DUAL_TOWER_MODE,
    )
    parser.add_argument(
        "--dual-tower-alpha-loss-weight", type=float,
        default=DUAL_TOWER_ALPHA_LOSS_WEIGHT,
    )
    parser.add_argument(
        "--dual-tower-style-loss-weight", type=float,
        default=DUAL_TOWER_STYLE_LOSS_WEIGHT,
    )
    parser.add_argument(
        "--dual-tower-style-hidden-dim", type=int,
        default=DUAL_TOWER_STYLE_HIDDEN_DIM,
    )
    parser.add_argument(
        "--dual-tower-style-dropout", type=float,
        default=DUAL_TOWER_STYLE_DROPOUT,
    )
    parser.add_argument(
        "--barra-gate-hidden-dim",
        type=int,
        default=BARRA_GATE_HIDDEN_DIM,
    )
    parser.add_argument(
        "--barra-industry-embedding-dim",
        type=int,
        default=BARRA_INDUSTRY_EMBEDDING_DIM,
    )
    parser.add_argument(
        "--barra-gate-dropout",
        type=float,
        default=BARRA_GATE_DROPOUT,
    )
    parser.add_argument(
        "--barra-gate-scale",
        type=float,
        default=BARRA_GATE_SCALE,
    )
    parser.add_argument(
        "--adversarial-barra", type=int, choices=(0, 1),
        default=int(ADVERSARIAL_BARRA),
    )
    parser.add_argument(
        "--adversarial-gamma-max", type=float,
        default=ADVERSARIAL_GAMMA_MAX,
    )
    parser.add_argument(
        "--adversarial-warmup-epochs", type=int,
        default=ADVERSARIAL_WARMUP_EPOCHS,
    )
    parser.add_argument(
        "--adversarial-ramp-epochs", type=int,
        default=ADVERSARIAL_RAMP_EPOCHS,
    )
    parser.add_argument(
        "--adversarial-style-loss-weight", type=float,
        default=ADVERSARIAL_STYLE_LOSS_WEIGHT,
    )
    parser.add_argument(
        "--adversarial-industry-loss-weight", type=float,
        default=ADVERSARIAL_INDUSTRY_LOSS_WEIGHT,
    )
    parser.add_argument(
        "--adversarial-head-hidden-dim", type=int,
        default=ADVERSARIAL_HEAD_HIDDEN_DIM,
    )
    parser.add_argument(
        "--adversarial-head-dropout", type=float,
        default=ADVERSARIAL_HEAD_DROPOUT,
    )


def worker_command(
    args: argparse.Namespace, year: int, seed: int, physical_gpu: int
) -> list[str]:
    """构造单个年度×seed worker 的命令行参数。"""
    command = [
        sys.executable,
        str(SCRIPT_PATH),
        "worker",
        "--worker-year", str(year),
        "--worker-seed", str(seed),
        "--worker-physical-gpu", str(physical_gpu),
    ]
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
        command.extend([f"--{name.replace('_', '-')}", str(getattr(args, name))])
    return command


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


def _barra_ddp_rank(
    rank: int,
    world_size: int,
    args: argparse.Namespace,
    smoke: bool,
) -> None:
    """执行 _barra_ddp_rank 对应的分布式训练处理。"""
    args.execution_route = "ddp"
    args.ddp_world_size = world_size
    configure_from_args(args)
    context = distributed.initialize_process_group(rank, world_size, backend="nccl")
    try:
        result = train_task(
            args.forecast_year if smoke else args.worker_year,
            args.seed if smoke else args.worker_seed,
            int(EXECUTION_DEVICE_IDS[0]),
            max_epochs=1 if smoke else None,
            max_batches=args.max_batches if smoke else None,
            persist=not smoke,
            distributed_context=context,
        )
        if context.is_primary:
            result.update({
                "execution_route": "ddp",
                "world_size": world_size,
                "persistent_outputs_written": not smoke,
            })
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
    args.execution_route = "ddp"
    args.ddp_world_size = len(devices)
    if smoke:
        args.worker_year = args.forecast_year
        args.worker_seed = args.seed
    args.worker_physical_gpu = devices[0]
    distributed.configure_master(distributed.reserve_local_port())
    mp.spawn(
        _barra_ddp_rank,
        args=(len(devices), args, smoke),
        nprocs=len(devices),
        join=True,
    )


def run_ddp(args: argparse.Namespace) -> pd.DataFrame:
    """执行 run_ddp 对应的分布式训练处理。"""

    devices = distributed.parse_device_ids(args.device_ids)
    args.execution_route = "ddp"
    args.ddp_world_size = len(devices)
    configure_from_args(args)
    _, _, quality_root = base.run_paths(RUN_NAME)
    log_dir = quality_root / "ddp_task_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print(
        f"style-neutral DDP start: "
        f"tasks={len(FORECAST_YEARS) * len(SEEDS)}, "
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
                tail_text = "\n".join(
                    log_path.read_text(
                        encoding="utf-8", errors="replace"
                    ).splitlines()[-100:]
                )
                raise RuntimeError(
                    f"DDP 任务失败：{year}/seed={seed}\n{tail_text}"
                )
            print(f"done DDP: {year}/seed={seed}", flush=True)
    result = finalize_run(time.perf_counter() - started)
    print(validate_outputs().to_string(index=False), flush=True)
    print(f"实验报告：{write_report()}", flush=True)
    return result


def run_a2(args: argparse.Namespace) -> pd.DataFrame:
    """调度所有年度和随机种子任务，并支持断点续训与失败恢复。"""
    devices = tuple(int(value) for value in args.device_ids.split(","))
    args.execution_route = "a2"
    args.ddp_world_size = 1
    configure_from_args(args)
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("--device-ids 为空或包含重复 GPU")
    tasks = deque((year, seed) for year in FORECAST_YEARS for seed in SEEDS)
    _, _, quality_root = base.run_paths(RUN_NAME)
    log_dir = quality_root / "style_task_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    running: list[dict[str, Any]] = []
    started = time.perf_counter()
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
    try:
        while tasks or running:
            for physical_gpu in devices:
                active = sum(
                    item["physical_gpu"] == physical_gpu for item in running
                )
                while tasks and active < TASKS_PER_GPU:
                    year, seed = tasks.popleft()
                    log_path = log_dir / f"forecast_{year}.seed_{seed}.log"
                    handle = log_path.open("a", encoding="utf-8")
                    handle.write(  # 追加任务状态日志
                        f"\n=== launch {year}/seed={seed} at "
                        f"{time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
                    )
                    handle.flush()  # 将 memmap 修改刷入磁盘
                    environment = os.environ.copy()
                    environment["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
                    process = subprocess.Popen(  # 启动独立训练任务
                        worker_command(args, year, seed, physical_gpu),
                        cwd=VIVIT_DIR,
                        env=environment,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
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
                    active += 1
                    print(
                        f"launch GPU{physical_gpu}: {year}/seed={seed} "
                        f"pid={process.pid}",
                        flush=True,
                    )
            finished: list[dict[str, Any]] = []
            for item in running:
                code = item["process"].poll()  # 检查训练任务状态
                if code is None:
                    continue
                item["handle"].close()
                finished.append(item)
                if code:
                    tail = item["log"].read_text(
                        encoding="utf-8", errors="replace"
                    ).splitlines()[-100:]
                    raise RuntimeError(
                        f"任务失败：{item['year']}/seed={item['seed']}\n"
                        + "\n".join(tail)
                    )
                print(
                    f"done GPU{item['physical_gpu']}: "
                    f"{item['year']}/seed={item['seed']} "
                    f"{(time.perf_counter()-item['started'])/60:.2f} min",
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
            item["handle"].close()
        raise

    result = finalize_run(time.perf_counter() - started)  # 汇总任务并写出完成标记
    print(validate_outputs().to_string(index=False), flush=True)
    print(f"实验报告：{write_report()}", flush=True)
    return result


def parse_args() -> argparse.Namespace:
    """解析并校验命令行参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-data")
    prepare.add_argument("--overwrite", type=int, choices=(0, 1), default=0)

    prepare.add_argument(
        "--include-industry-sidecar", type=int, choices=(0, 1), default=0
    )
    prepare.add_argument(
        "--include-full-barra-target", type=int, choices=(0, 1), default=0
    )
    prepare.add_argument(
        "--include-dual-tower-target", type=int, choices=(0, 1), default=0
    )
    smoke = subparsers.add_parser("smoke")
    add_training_arguments(smoke)
    smoke.add_argument("--forecast-year", type=int, default=2018)
    smoke.add_argument("--seed", type=int, default=42)
    smoke.add_argument("--max-batches", type=int, default=1)
    smoke.set_defaults(compile=0, workers_per_task=0)

    worker = subparsers.add_parser("worker")
    add_training_arguments(worker)
    worker.add_argument("--worker-year", type=int, required=True)
    worker.add_argument("--worker-seed", type=int, required=True)
    worker.add_argument("--worker-physical-gpu", type=int, required=True)

    a2 = subparsers.add_parser("a2")
    add_training_arguments(a2)
    a2.add_argument("--device-ids", default=DEVICE_IDS)
    a2.add_argument("--tasks-per-gpu", type=int, default=TASKS_PER_GPU)
    ddp = subparsers.add_parser("ddp")
    add_training_arguments(ddp)
    ddp.add_argument("--device-ids", default=DEVICE_IDS)

    ddp_worker = subparsers.add_parser("ddp-worker")
    add_training_arguments(ddp_worker)
    ddp_worker.add_argument("--worker-year", type=int, required=True)
    ddp_worker.add_argument("--worker-seed", type=int, required=True)
    ddp_worker.add_argument("--worker-physical-gpu", type=int, required=True)
    ddp_worker.add_argument("--device-ids", required=True)

    ddp_smoke = subparsers.add_parser("ddp-smoke")
    add_training_arguments(ddp_smoke)
    ddp_smoke.add_argument("--forecast-year", type=int, default=2018)
    ddp_smoke.add_argument("--seed", type=int, default=42)
    ddp_smoke.add_argument("--max-batches", type=int, default=1)
    ddp_smoke.add_argument("--device-ids", default=DEVICE_IDS)
    ddp_smoke.set_defaults(compile=0, workers_per_task=0)

    for command in ("validate", "report"):
        subparser = subparsers.add_parser(command)
        add_training_arguments(subparser)
    return parser.parse_args()


def main() -> None:
    """根据命令行子命令分派数据、训练、验证或评价流程。"""
    args = parse_args()
    if args.command == "prepare-data":
        style_manifest = style_data.build_style_neutral_store(
            overwrite=bool(args.overwrite)
        )
        prepared = {"style_store": style_manifest}
        include_dual_target = bool(args.include_dual_tower_target)
        include_full_target = (
            bool(args.include_full_barra_target) or include_dual_target
        )
        if bool(args.include_industry_sidecar) or include_full_target:
            prepared["industry_sidecar"] = style_data.build_industry_sidecar(
                overwrite=bool(args.overwrite)
            )
        if include_full_target:
            prepared["full_barra_target"] = (
                style_data.build_full_barra_target_sidecar(
                    overwrite=bool(args.overwrite)
                )
            )
        if include_dual_target:
            prepared["dual_tower_target"] = (
                style_data.build_dual_tower_target_sidecar(
                    overwrite=bool(args.overwrite)
                )
            )
        print(
            json.dumps(
                prepared, ensure_ascii=False, indent=2
            )
        )
        return

    if args.command in {"ddp", "ddp-worker", "ddp-smoke"}:
        args.execution_route = "ddp"
        args.ddp_world_size = len(distributed.parse_device_ids(args.device_ids))
    else:
        args.execution_route = "a2"
        args.ddp_world_size = 1
    configure_from_args(args)
    if args.command in {"smoke", "a2", "ddp", "ddp-smoke"}:
        style_data.build_style_neutral_store(overwrite=False)
    else:
        style_data.require_style_neutral_store()
    if industry_sidecar_required():
        if args.command in {"smoke", "a2", "ddp", "ddp-smoke"}:
            style_data.build_industry_sidecar(overwrite=False)
        else:
            style_data.require_industry_sidecar()
    if barra_conditioning_enabled() or dual_tower_enabled():
        if args.command in {"smoke", "a2", "ddp", "ddp-smoke"}:
            style_data.build_full_barra_target_sidecar(overwrite=False)
        else:
            style_data.require_full_barra_target_sidecar()

    if dual_tower_enabled():
        if args.command in {"smoke", "a2", "ddp", "ddp-smoke"}:
            style_data.build_dual_tower_target_sidecar(overwrite=False)
        else:
            style_data.require_dual_tower_target_sidecar()

    if args.command == "smoke":
        print(
            json.dumps(
                smoke_test(args.forecast_year, args.seed, args.max_batches),
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command == "worker":
        print(
            json.dumps(
                train_task(
                    args.worker_year, args.worker_seed, args.worker_physical_gpu
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command == "ddp-worker":
        run_ddp_worker(args)
    elif args.command == "ddp-smoke":
        run_ddp_worker(args, smoke=True)
    elif args.command == "a2":
        print(run_a2(args).to_string(index=False))
    elif args.command == "ddp":
        print(run_ddp(args).to_string(index=False))
    elif args.command == "validate":
        print(validate_outputs().to_string(index=False))
    elif args.command == "report":
        print(write_report())


if __name__ == "__main__":
    main()
