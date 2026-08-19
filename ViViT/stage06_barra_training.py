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


VIVIT_DIR = Path(__file__).resolve().parent  # 定位vivit目录。
SCRIPT_PATH = Path(__file__).resolve()  # 定位script路径。
if str(VIVIT_DIR) not in sys.path:  # 检查当前值是否属于允许集合。
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


RUN_NAME = "sigmoid_daybatch_rankic_style_residual_label_projected_output"  # 设置执行名称。
DEVICE_IDS = "0,1,2,3"  # 设置设备ids。
TASKS_PER_GPU = 1  # 设置taskspergpu。
WORKERS_PER_TASK = 8  # 设置workerspertask。
RANK_TEMPERATURE = 0.1  # 设置排名temperature。
TURNOVER_LAMBDA = 0.1  # 设置turnoverlambda。
HUBER_DELTA = 1.0  # 设置huberdelta。
SMOOTH_L1_BETA = 0.1  # 设置smoothl1beta。
HYBRID_WEIGHT = 0.5  # 设置hybrid权重。
LEARNING_RATE = 0.001  # 设置learning比率。
MAX_EPOCHS = 100  # 设置maxepochs。
PATIENCE = 10  # 设置patience。
USE_TORCH_COMPILE = 1  # 设置usetorchcompile。
COMPILE_MODE = "default"  # 设置compile模式。
USE_FUSED_ADAM = 1  # 设置usefusedadam。
USE_CUDA_PREFETCH = 1  # 设置usecudaprefetch。
OVERWRITE = 0  # 设置overwrite。
LOSS_NAME = "sigmoid_rank_ic"  # 设置损失名称。
VALIDATION_METRIC = "daily_mean_spearman_ic"  # 设置验证指标。
TRAIN_BATCH_MODE = "trade_date"  # 设置训练批次模式。
# Execution metadata is deliberately absent from legacy A2 signatures. DDP
# adds it explicitly so checkpoints cannot resume across routes/world sizes.
EXECUTION_ROUTE = "a2"  # 设置executionroute。
DDP_WORLD_SIZE = 1  # 设置ddpworld大小。
EXECUTION_DEVICE_IDS: tuple[int, ...] = ()  # 设置execution设备ids。
RESIDUALIZE_TARGET = True  # 设置残差化标签。
PROJECT_MODEL_OUTPUT = True  # 设置投影模型输出。
RAW_BARRA_R2_LAMBDA = 0.0  # 设置原始Barrar2lambda。
ADVERSARIAL_BARRA = False  # 设置adversarialBarra。
BARRA_CONDITIONING_MODE = "off"  # 设置Barraconditioning模式。
BARRA_OUTPUT_NEUTRALIZATION = False  # 设置Barra输出neutralization。
BARRA_DUAL_TOWER = False  # 设置Barradualtower。
BARRA_DUAL_TOWER_MODE = "real"  # 设置Barradualtower模式。
BARRA_DUAL_TOWER_MODES = ("real", "alpha_video_only", "shuffled", "zero")  # 设置Barradualtowermodes。
DUAL_TOWER_ALPHA_LOSS_WEIGHT = 1.0  # 设置dualtowerAlpha损失权重。
DUAL_TOWER_STYLE_LOSS_WEIGHT = 1.0  # 设置dualtower风格损失权重。
DUAL_TOWER_STYLE_HIDDEN_DIM = 64  # 设置dualtower风格隐藏层dim。
DUAL_TOWER_STYLE_DROPOUT = 0.10  # 设置dualtower风格dropout。
BARRA_CONDITIONING_MODES = (  # 设置Barraconditioningmodes。
    "off", "video_only", "real", "shuffled", "zero"
)
BARRA_GATE_HIDDEN_DIM = 64  # 设置Barra门控隐藏层dim。
BARRA_INDUSTRY_EMBEDDING_DIM = 16  # 设置Barra行业嵌入dim。
BARRA_GATE_DROPOUT = 0.10  # 设置Barra门控dropout。
BARRA_GATE_SCALE = 0.50  # 设置Barra门控scale。
ADVERSARIAL_GAMMA_MAX = 0.10  # 设置adversarial对抗系数max。
ADVERSARIAL_WARMUP_EPOCHS = 2  # 设置adversarialwarmupepochs。
ADVERSARIAL_RAMP_EPOCHS = 5  # 设置adversarialrampepochs。
ADVERSARIAL_STYLE_LOSS_WEIGHT = 1.0  # 设置adversarial风格损失权重。
ADVERSARIAL_INDUSTRY_LOSS_WEIGHT = 1.0  # 设置adversarial行业损失权重。
ADVERSARIAL_HEAD_HIDDEN_DIM = 128  # 设置adversarialhead隐藏层dim。
ADVERSARIAL_HEAD_DROPOUT = 0.10  # 设置adversarialheaddropout。
FORECAST_YEARS = sequence_data.FORECAST_YEARS  # 集中声明forecast年份。
SEEDS = sequence_data.SEEDS  # 设置随机种子。
STYLE_COUNT = len(style_data.STYLE_COLUMNS)  # 设置风格数量。
TRAINING_SCHEMA_VERSION = 4  # 固定产物 schema 版本，用于判断缓存与复用兼容性。
INDUSTRY_COUNT = style_data.EXPECTED_CNE5S_INDUSTRIES  # 设置行业数量。

PUBLISHED_FACTOR_POLICY = "unprojected_model_score_daily_seed_zscore_mean"  # 设置published因子policy。
class DegenerateValidationScoreError(FloatingPointError):  # 验证日的原始预测分数退化为常数或近似常数。
    """验证日的原始预测分数退化为常数或近似常数。
    """

    def __init__(  # 初始化当前对象的配置、状态与子组件。
        self,  # 汇入self。
        *,
        date_id: int,  # 接收交易日id。
        rows: int,  # 接收行数。
        score_std: float,  # 接收分数标准差。
        total_sum_squares: float,  # 接收totalsumsquares。
    ) -> None:  # 声明当前函数的返回类型。
        super().__init__(
            "验证集原始预测分数在单日截面内退化："
            f"date_id={date_id}, rows={rows}, score_std={score_std:.6g}, "
            f"total_sum_squares={total_sum_squares:.6g}"
        )
        self.date_id = int(date_id)  # 保存交易日id，供该对象后续步骤复用。
        self.rows = int(rows)  # 保存行数，供该对象后续步骤复用。
        self.score_std = float(score_std)  # 保存分数标准差，供该对象后续步骤复用。
        self.total_sum_squares = float(total_sum_squares)  # 保存totalsumsquares，供该对象后续步骤复用。


def barra_conditioning_enabled() -> bool:  # 判断当前实验是否启用 Barra 条件输入。
    """判断当前实验是否启用 Barra 条件输入。"""
    return BARRA_CONDITIONING_MODE != "off"  # 返回判断当前实验是否启用 Barra 条件输入。

def strict_barra_output_enabled() -> bool:  # 判断当前实验是否发布严格中性的投影分数。
    """判断当前实验是否发布严格中性的投影分数。"""
    return bool(BARRA_OUTPUT_NEUTRALIZATION)  # 返回判断当前实验是否发布严格中性的投影分数。


def dual_tower_enabled() -> bool:  # 判断当前实验是否启用 Alpha/Style 双塔。
    """判断当前实验是否启用 Alpha/Style 双塔。"""
    return bool(BARRA_DUAL_TOWER)  # 返回判断当前实验是否启用 Alpha/Style 双塔。


def target_policy() -> str:  # 返回当前实验使用的原始、连续风格残差或完整风险残差标签策略。
    """返回当前实验使用的原始、连续风格残差或完整风险残差标签策略。"""
    if dual_tower_enabled():  # 根据当前数据状态选择处理分支。
        return "source_plus_full_barra_residual_plus_fitted_targets"  # 返回当前实验使用的原始、连续风格残差或完整风险残差标签策略。
    if barra_conditioning_enabled():  # 根据当前数据状态选择处理分支。
        return "daily_ten_style_plus_industry_residual_then_zscore_ddof0"  # 返回当前实验使用的原始、连续风格残差或完整风险残差标签策略。
    return (  # 返回当前实验使用的原始、连续风格残差或完整风险残差标签策略。
        "daily_ten_style_ols_residual_then_zscore_ddof0"
        if RESIDUALIZE_TARGET  # 根据当前数据状态选择处理分支。
        else "source_target_zscore"
    )


def output_policy() -> str:  # 返回训练、验证和正式发布分数的口径策略。
    """返回训练、验证和正式发布分数的口径策略。"""
    if dual_tower_enabled():  # 根据当前数据状态选择处理分支。
        return "dual_tower_unprojected_alpha_score"  # 返回训练、验证和正式发布分数的口径策略。
    if strict_barra_output_enabled():  # 根据当前数据状态选择处理分支。
        return "daily_ten_style_plus_industry_projection_then_zscore_ddof0"  # 返回训练、验证和正式发布分数的口径策略。
    return (  # 返回训练、验证和正式发布分数的口径策略。
        "daily_ten_style_qr_residual_then_zscore_ddof0"
        if PROJECT_MODEL_OUTPUT  # 根据当前数据状态选择处理分支。
        else "model_raw_score"
    )


def raw_barra_r2_enabled() -> bool:  # 判断是否启用 raw-score 联合 Barra R² 惩罚。
    """判断是否启用 raw-score 联合 Barra R² 惩罚。"""
    return RAW_BARRA_R2_LAMBDA > 0.0  # 返回判断是否启用 raw-score 联合 Barra R² 惩罚。


def adversarial_barra_enabled() -> bool:  # 判断是否启用对抗式 Barra 不变表征。
    """判断是否启用对抗式 Barra 不变表征。"""
    return bool(ADVERSARIAL_BARRA)  # 返回判断是否启用对抗式 Barra 不变表征。


def industry_sidecar_required() -> bool:  # 判断当前实验是否必须加载行业 sidecar。
    """判断当前实验是否必须加载行业 sidecar。"""
    return (  # 返回判断当前实验是否必须加载行业 sidecar。
        raw_barra_r2_enabled()
        or adversarial_barra_enabled()
        or barra_conditioning_enabled()
        or dual_tower_enabled()
    )


def factor_role() -> str:  # 说明某个预测列在当前实验中的 Alpha、Style 或总收益角色。
    """说明某个预测列在当前实验中的 Alpha、Style 或总收益角色。"""
    if dual_tower_enabled():  # 根据当前数据状态选择处理分支。
        return f"barra_conditional_dual_tower_alpha_{BARRA_DUAL_TOWER_MODE}"  # 返回说明某个预测列在当前实验中的 Alpha、Style 或总收益角色。
    if strict_barra_output_enabled():  # 根据当前数据状态选择处理分支。
        return f"barra_conditioned_strict_neutral_{BARRA_CONDITIONING_MODE}"  # 返回说明某个预测列在当前实验中的 Alpha、Style 或总收益角色。
    if barra_conditioning_enabled():  # 根据当前数据状态选择处理分支。
        return f"barra_conditioned_multiplicative_gate_{BARRA_CONDITIONING_MODE}"  # 返回说明某个预测列在当前实验中的 Alpha、Style 或总收益角色。
    if adversarial_barra_enabled():  # 根据当前数据状态选择处理分支。
        return "adversarial_barra_invariant_representation_model"  # 返回说明某个预测列在当前实验中的 Alpha、Style 或总收益角色。
    if raw_barra_r2_enabled():  # 根据当前数据状态选择处理分支。
        return "raw_score_barra_r_squared_regularized_model"  # 返回说明某个预测列在当前实验中的 Alpha、Style 或总收益角色。
    if RESIDUALIZE_TARGET and PROJECT_MODEL_OUTPUT:  # 根据当前数据状态选择处理分支。
        return "style_residual_label_projected_output_model"  # 返回说明某个预测列在当前实验中的 Alpha、Style 或总收益角色。
    if RESIDUALIZE_TARGET:  # 根据当前数据状态选择处理分支。
        return "style_residual_label_model"  # 返回说明某个预测列在当前实验中的 Alpha、Style 或总收益角色。
    if PROJECT_MODEL_OUTPUT:  # 根据当前数据状态选择处理分支。
        return "style_projected_output_model"  # 返回说明某个预测列在当前实验中的 Alpha、Style 或总收益角色。
    return "style_backend_raw_model"  # 返回说明某个预测列在当前实验中的 Alpha、Style 或总收益角色。


def configure_from_args(args: argparse.Namespace) -> None:  # 将 CLI 配置安装为 worker 进程内只读实验配置。
    """将 CLI 配置安装为 worker 进程内只读实验配置。"""

    global RUN_NAME, DEVICE_IDS, TASKS_PER_GPU, WORKERS_PER_TASK  # 声明后续赋值使用外层作用域变量。
    global EXECUTION_ROUTE, DDP_WORLD_SIZE, EXECUTION_DEVICE_IDS  # 声明后续赋值使用外层作用域变量。
    global RANK_TEMPERATURE, TURNOVER_LAMBDA, HUBER_DELTA  # 声明后续赋值使用外层作用域变量。
    global SMOOTH_L1_BETA, HYBRID_WEIGHT, LEARNING_RATE  # 声明后续赋值使用外层作用域变量。
    global MAX_EPOCHS, PATIENCE, USE_TORCH_COMPILE, COMPILE_MODE  # 声明后续赋值使用外层作用域变量。
    global USE_FUSED_ADAM, USE_CUDA_PREFETCH, OVERWRITE, LOSS_NAME  # 声明后续赋值使用外层作用域变量。
    global VALIDATION_METRIC, TRAIN_BATCH_MODE  # 声明后续赋值使用外层作用域变量。
    global RESIDUALIZE_TARGET, PROJECT_MODEL_OUTPUT, RAW_BARRA_R2_LAMBDA  # 声明后续赋值使用外层作用域变量。
    global ADVERSARIAL_BARRA, ADVERSARIAL_GAMMA_MAX  # 声明后续赋值使用外层作用域变量。
    global BARRA_CONDITIONING_MODE, BARRA_GATE_HIDDEN_DIM  # 声明后续赋值使用外层作用域变量。
    global BARRA_INDUSTRY_EMBEDDING_DIM, BARRA_GATE_DROPOUT  # 声明后续赋值使用外层作用域变量。
    global BARRA_GATE_SCALE, BARRA_OUTPUT_NEUTRALIZATION  # 声明后续赋值使用外层作用域变量。
    global BARRA_DUAL_TOWER, BARRA_DUAL_TOWER_MODE  # 声明后续赋值使用外层作用域变量。
    global DUAL_TOWER_ALPHA_LOSS_WEIGHT, DUAL_TOWER_STYLE_LOSS_WEIGHT  # 声明后续赋值使用外层作用域变量。
    global DUAL_TOWER_STYLE_HIDDEN_DIM, DUAL_TOWER_STYLE_DROPOUT  # 声明后续赋值使用外层作用域变量。
    global ADVERSARIAL_WARMUP_EPOCHS, ADVERSARIAL_RAMP_EPOCHS  # 声明后续赋值使用外层作用域变量。
    global ADVERSARIAL_STYLE_LOSS_WEIGHT, ADVERSARIAL_INDUSTRY_LOSS_WEIGHT  # 声明后续赋值使用外层作用域变量。
    global ADVERSARIAL_HEAD_HIDDEN_DIM, ADVERSARIAL_HEAD_DROPOUT  # 声明后续赋值使用外层作用域变量。

    RUN_NAME = args.run_name  # 设置执行名称。
    DEVICE_IDS = getattr(args, "device_ids", DEVICE_IDS)  # 设置设备ids。
    EXECUTION_ROUTE = str(getattr(args, "execution_route", "a2"))  # 设置executionroute。
    DDP_WORLD_SIZE = int(getattr(args, "ddp_world_size", 1))  # 设置ddpworld大小。
    EXECUTION_DEVICE_IDS = distributed.parse_device_ids(DEVICE_IDS)  # 设置execution设备ids。
    TASKS_PER_GPU = int(getattr(args, "tasks_per_gpu", TASKS_PER_GPU))  # 设置taskspergpu。
    WORKERS_PER_TASK = int(args.workers_per_task)  # 设置workerspertask。
    RANK_TEMPERATURE = float(args.rank_temperature)  # 设置排名temperature。
    TURNOVER_LAMBDA = float(args.turnover_lambda)  # 设置turnoverlambda。
    HUBER_DELTA = float(args.huber_delta)  # 设置huberdelta。
    SMOOTH_L1_BETA = float(args.smooth_l1_beta)  # 设置smoothl1beta。
    HYBRID_WEIGHT = float(args.hybrid_weight)  # 设置hybrid权重。
    LEARNING_RATE = float(args.learning_rate)  # 设置learning比率。
    MAX_EPOCHS = int(args.max_epochs)  # 设置maxepochs。
    PATIENCE = int(args.patience)  # 设置patience。
    USE_TORCH_COMPILE = int(args.compile)  # 设置usetorchcompile。
    COMPILE_MODE = args.compile_mode  # 设置compile模式。
    USE_FUSED_ADAM = int(args.fused_adam)  # 设置usefusedadam。
    USE_CUDA_PREFETCH = int(args.cuda_prefetch)  # 设置usecudaprefetch。
    OVERWRITE = int(args.overwrite)  # 设置overwrite。
    LOSS_NAME = args.loss_name  # 设置损失名称。
    VALIDATION_METRIC = args.validation_metric  # 设置验证指标。
    TRAIN_BATCH_MODE = args.train_batch_mode  # 设置训练批次模式。
    RESIDUALIZE_TARGET = bool(args.residualize_target)  # 设置残差化标签。
    PROJECT_MODEL_OUTPUT = bool(args.project_model_output)  # 设置投影模型输出。
    RAW_BARRA_R2_LAMBDA = float(args.raw_barra_r2_lambda)  # 设置原始Barrar2lambda。
    ADVERSARIAL_BARRA = bool(args.adversarial_barra)  # 设置adversarialBarra。
    BARRA_CONDITIONING_MODE = str(args.barra_conditioning_mode)  # 设置Barraconditioning模式。
    BARRA_OUTPUT_NEUTRALIZATION = bool(  # 设置Barra输出neutralization。
        args.barra_output_neutralization  # 汇入Barra输出neutralization。
    )
    BARRA_DUAL_TOWER = bool(args.barra_dual_tower)  # 设置Barradualtower。
    BARRA_DUAL_TOWER_MODE = str(args.barra_dual_tower_mode)  # 设置Barradualtower模式。
    DUAL_TOWER_ALPHA_LOSS_WEIGHT = float(  # 设置dualtowerAlpha损失权重。
        args.dual_tower_alpha_loss_weight  # 汇入dualtowerAlpha损失权重。
    )
    DUAL_TOWER_STYLE_LOSS_WEIGHT = float(  # 设置dualtower风格损失权重。
        args.dual_tower_style_loss_weight  # 汇入dualtower风格损失权重。
    )
    DUAL_TOWER_STYLE_HIDDEN_DIM = int(args.dual_tower_style_hidden_dim)  # 设置dualtower风格隐藏层dim。
    DUAL_TOWER_STYLE_DROPOUT = float(args.dual_tower_style_dropout)  # 设置dualtower风格dropout。
    BARRA_GATE_HIDDEN_DIM = int(args.barra_gate_hidden_dim)  # 设置Barra门控隐藏层dim。
    BARRA_INDUSTRY_EMBEDDING_DIM = int(  # 设置Barra行业嵌入dim。
        args.barra_industry_embedding_dim  # 汇入Barra行业嵌入dim。
    )
    BARRA_GATE_DROPOUT = float(args.barra_gate_dropout)  # 设置Barra门控dropout。
    BARRA_GATE_SCALE = float(args.barra_gate_scale)  # 设置Barra门控scale。
    ADVERSARIAL_GAMMA_MAX = float(args.adversarial_gamma_max)  # 设置adversarial对抗系数max。
    ADVERSARIAL_WARMUP_EPOCHS = int(args.adversarial_warmup_epochs)  # 设置adversarialwarmupepochs。
    ADVERSARIAL_RAMP_EPOCHS = int(args.adversarial_ramp_epochs)  # 设置adversarialrampepochs。
    ADVERSARIAL_STYLE_LOSS_WEIGHT = float(  # 设置adversarial风格损失权重。
        args.adversarial_style_loss_weight  # 汇入adversarial风格损失权重。
    )
    ADVERSARIAL_INDUSTRY_LOSS_WEIGHT = float(  # 设置adversarial行业损失权重。
        args.adversarial_industry_loss_weight  # 汇入adversarial行业损失权重。
    )
    ADVERSARIAL_HEAD_HIDDEN_DIM = int(args.adversarial_head_hidden_dim)  # 设置adversarialhead隐藏层dim。
    ADVERSARIAL_HEAD_DROPOUT = float(args.adversarial_head_dropout)  # 设置adversarialheaddropout。

    config = training_config()
    base.validate_training_config(config, COMPILE_MODE)
    if TRAIN_BATCH_MODE != "trade_date":  # 按当前模式、标识或状态选择处理分支。
        raise ValueError("风格训练或 Barra R² 正则必须使用 trade_date batch")  # 输入数据不满足业务或数值约束时立即终止。
    if LOSS_NAME == base.TEMPORAL_TURNOVER_LOSS:  # 按当前模式、标识或状态选择处理分支。
        raise ValueError("风格训练后端暂不支持 sigmoid_rank_ic_turnover")  # 输入数据不满足业务或数值约束时立即终止。
    if RAW_BARRA_R2_LAMBDA < 0:  # 检查当前数值是否越过业务或数值阈值。
        raise ValueError("raw_barra_r2_lambda 必须 >= 0")  # 输入数据不满足业务或数值约束时立即终止。
    if raw_barra_r2_enabled() and not dual_tower_enabled() and (  # 检查必需数据、配置或校验结果是否缺失。
        RESIDUALIZE_TARGET or PROJECT_MODEL_OUTPUT
    ):
        raise ValueError(  # 输入数据不满足业务或数值约束时立即终止。
            "No.2 要求 RESIDUALIZE_TARGET=False 且 "
            "PROJECT_MODEL_OUTPUT=False"
        )
    if raw_barra_r2_enabled() and not dual_tower_enabled() and LOSS_NAME != "sigmoid_rank_ic":  # 按当前模式、标识或状态选择处理分支。
        raise ValueError("No.2 当前只支持 sigmoid_rank_ic")  # 输入数据不满足业务或数值约束时立即终止。
    if (  # 开始组合多项有效性或配置条件。
        raw_barra_r2_enabled() and not dual_tower_enabled()
        and VALIDATION_METRIC != "daily_mean_spearman_ic"
    ):
        raise ValueError(  # 输入数据不满足业务或数值约束时立即终止。
            "No.2 的早停指标必须为 daily_mean_spearman_ic"
        )
    if adversarial_barra_enabled():  # 根据当前数据状态选择处理分支。
        if RESIDUALIZE_TARGET or PROJECT_MODEL_OUTPUT:  # 根据当前数据状态选择处理分支。
            raise ValueError(  # 输入数据不满足业务或数值约束时立即终止。
                "No.3 要求 RESIDUALIZE_TARGET=False 且 "
                "PROJECT_MODEL_OUTPUT=False"
            )
        if raw_barra_r2_enabled():  # 根据当前数据状态选择处理分支。
            raise ValueError("No.3 首轮纯对抗实验禁止叠加 raw-score R²")  # 输入数据不满足业务或数值约束时立即终止。
        if LOSS_NAME != "sigmoid_rank_ic":  # 按当前模式、标识或状态选择处理分支。
            raise ValueError("No.3 当前只支持 sigmoid_rank_ic")  # 输入数据不满足业务或数值约束时立即终止。
        if VALIDATION_METRIC != "daily_mean_spearman_ic":  # 按当前模式、标识或状态选择处理分支。
            raise ValueError(  # 输入数据不满足业务或数值约束时立即终止。
                "No.3 的早停指标必须为 daily_mean_spearman_ic"
            )
        if (  # 开始组合多项有效性或配置条件。
            ADVERSARIAL_GAMMA_MAX <= 0
            or ADVERSARIAL_WARMUP_EPOCHS < 0
            or ADVERSARIAL_RAMP_EPOCHS < 1
            or ADVERSARIAL_STYLE_LOSS_WEIGHT <= 0
            or ADVERSARIAL_INDUSTRY_LOSS_WEIGHT <= 0
            or ADVERSARIAL_HEAD_HIDDEN_DIM < 1
            or not 0 <= ADVERSARIAL_HEAD_DROPOUT < 1
        ):
            raise ValueError("No.3 对抗训练超参数非法")  # 输入数据不满足业务或数值约束时立即终止。
    if BARRA_CONDITIONING_MODE not in BARRA_CONDITIONING_MODES:  # 检查当前值是否属于允许集合。
        raise ValueError(  # 输入数据不满足业务或数值约束时立即终止。
            f"barra_conditioning_mode 必须属于 {BARRA_CONDITIONING_MODES}"
        )
    if BARRA_DUAL_TOWER_MODE not in BARRA_DUAL_TOWER_MODES:  # 检查当前值是否属于允许集合。
        raise ValueError(  # 输入数据不满足业务或数值约束时立即终止。
            f"barra_dual_tower_mode 必须属于 {BARRA_DUAL_TOWER_MODES}"
        )
    if dual_tower_enabled():  # 根据当前数据状态选择处理分支。
        if RESIDUALIZE_TARGET or PROJECT_MODEL_OUTPUT:  # 根据当前数据状态选择处理分支。
            raise ValueError("No.6 要求关闭 Neutral 标签与输出投影")  # 输入数据不满足业务或数值约束时立即终止。
        if adversarial_barra_enabled() or barra_conditioning_enabled():  # 根据当前数据状态选择处理分支。
            raise ValueError("No.6 不能与 No.3 或 No.4 独立模式同时开启")  # 输入数据不满足业务或数值约束时立即终止。
        if strict_barra_output_enabled():  # 根据当前数据状态选择处理分支。
            raise ValueError("No.6 正式输出为未投影 Alpha，禁止 No.5 严格投影")  # 输入数据不满足业务或数值约束时立即终止。
        if LOSS_NAME != "sigmoid_rank_ic":  # 按当前模式、标识或状态选择处理分支。
            raise ValueError("No.6 只允许 sigmoid_rank_ic")  # 输入数据不满足业务或数值约束时立即终止。
        if VALIDATION_METRIC != "daily_mean_spearman_ic":  # 按当前模式、标识或状态选择处理分支。
            raise ValueError("No.6 早停只允许 daily_mean_spearman_ic")  # 输入数据不满足业务或数值约束时立即终止。
        if (  # 开始组合多项有效性或配置条件。
            DUAL_TOWER_ALPHA_LOSS_WEIGHT <= 0
            or DUAL_TOWER_STYLE_LOSS_WEIGHT <= 0
            or DUAL_TOWER_STYLE_HIDDEN_DIM < 1
            or not 0 <= DUAL_TOWER_STYLE_DROPOUT < 1
        ):
            raise ValueError("No.6 双塔损失权重或 Style 塔超参数非法")  # 输入数据不满足业务或数值约束时立即终止。
    if strict_barra_output_enabled() and not barra_conditioning_enabled():  # 检查必需数据、配置或校验结果是否缺失。
        raise ValueError(  # 输入数据不满足业务或数值约束时立即终止。
            "严格 Barra 输出中性化要求开启一种条件门控模式"
        )
    if barra_conditioning_enabled():  # 根据当前数据状态选择处理分支。
        if RESIDUALIZE_TARGET or PROJECT_MODEL_OUTPUT:  # 根据当前数据状态选择处理分支。
            raise ValueError(  # 输入数据不满足业务或数值约束时立即终止。
                "No.4 要求 RESIDUALIZE_TARGET=False 且 "
                "PROJECT_MODEL_OUTPUT=False"
            )
        if adversarial_barra_enabled():  # 根据当前数据状态选择处理分支。
            raise ValueError("No.4 不能与 No.3 对抗训练同时开启")  # 输入数据不满足业务或数值约束时立即终止。
        if LOSS_NAME != "sigmoid_rank_ic":  # 按当前模式、标识或状态选择处理分支。
            raise ValueError("No.4 当前只支持 sigmoid_rank_ic")  # 输入数据不满足业务或数值约束时立即终止。
        if strict_barra_output_enabled() and raw_barra_r2_enabled():  # 根据当前数据状态选择处理分支。
            raise ValueError(  # 输入数据不满足业务或数值约束时立即终止。
                "严格 Barra 输出中性化已代数正交，RAW_BARRA_R2_LAMBDA 必须为0"
            )
        if VALIDATION_METRIC != "daily_mean_spearman_ic":  # 按当前模式、标识或状态选择处理分支。
            raise ValueError(  # 输入数据不满足业务或数值约束时立即终止。
                "No.4 的早停指标必须为 daily_mean_spearman_ic"
            )
        if (  # 开始组合多项有效性或配置条件。
            BARRA_GATE_HIDDEN_DIM < 1
            or BARRA_INDUSTRY_EMBEDDING_DIM < 1
            or not 0 <= BARRA_GATE_DROPOUT < 1
            or not 0 < BARRA_GATE_SCALE < 1
        ):
            raise ValueError("No.4 条件门控超参数非法")  # 输入数据不满足业务或数值约束时立即终止。
    if TASKS_PER_GPU < 1 or WORKERS_PER_TASK < 0:  # 检查当前数值是否越过业务或数值阈值。
        raise ValueError("tasks_per_gpu 必须为正整数且 workers_per_task >= 0")  # 输入数据不满足业务或数值约束时立即终止。


def build_training_model(  # 按显式方案构造模型。
    device: torch.device, *, use_compile: bool
) -> torch.nn.Module:  # 声明当前函数的返回类型。
    """按显式方案构造模型；新增结构来自短实验模块。"""

    if dual_tower_enabled():  # 根据当前数据状态选择处理分支。
        from stage04_no4_gate import (
            BarraConditionalViViT as CoreConditionalViViT,
        )
        from stage04_no6_dual_tower import (
            BarraDualTowerViViT as CoreDualTowerViViT,
        )

        alpha_mode = (
            "video_only"
            if BARRA_DUAL_TOWER_MODE == "alpha_video_only"  # 按当前模式、标识或状态选择处理分支。
            else "real"
        )
        alpha_tower = CoreConditionalViViT(
            base.OrderFlowViViT(base.MODEL_CONFIG),
            conditioning_mode=alpha_mode,  # 配置conditioning模式。
            style_count=STYLE_COUNT,  # 配置风格数量。
            industry_count=INDUSTRY_COUNT,  # 配置行业数量。
            embedding_dim=base.MODEL_CONFIG.embedding_dim,  # 配置嵌入dim。
            gate_hidden_dim=BARRA_GATE_HIDDEN_DIM,  # 配置门控隐藏层dim。
            industry_embedding_dim=BARRA_INDUSTRY_EMBEDDING_DIM,  # 配置行业嵌入dim。
            dropout=BARRA_GATE_DROPOUT,  # 配置dropout。
            gate_scale=BARRA_GATE_SCALE,  # 配置门控scale。
        )
        model: torch.nn.Module = CoreDualTowerViViT(  # 接收模型。
            alpha_tower,  # 汇入Alphatower。
            style_count=STYLE_COUNT,  # 配置风格数量。
            industry_count=INDUSTRY_COUNT,  # 配置行业数量。
            industry_embedding_dim=BARRA_INDUSTRY_EMBEDDING_DIM,  # 配置行业嵌入dim。
            style_hidden_dim=DUAL_TOWER_STYLE_HIDDEN_DIM,  # 配置风格隐藏层dim。
            style_dropout=DUAL_TOWER_STYLE_DROPOUT,  # 配置风格dropout。
        ).to(device)
    elif barra_conditioning_enabled():  # 检查上一条件不成立时的备选分支。
        from stage04_no4_gate import (
            BarraConditionalViViT as CoreConditionalViViT,
        )

        internal_mode = (
            "video_only"
            if BARRA_CONDITIONING_MODE == "video_only"  # 按当前模式、标识或状态选择处理分支。
            else "real"
        )
        model = CoreConditionalViViT(  # 创建模型。
            base.OrderFlowViViT(base.MODEL_CONFIG),
            conditioning_mode=internal_mode,  # 配置conditioning模式。
            style_count=STYLE_COUNT,  # 配置风格数量。
            industry_count=INDUSTRY_COUNT,  # 配置行业数量。
            embedding_dim=base.MODEL_CONFIG.embedding_dim,  # 配置嵌入dim。
            gate_hidden_dim=BARRA_GATE_HIDDEN_DIM,  # 配置门控隐藏层dim。
            industry_embedding_dim=BARRA_INDUSTRY_EMBEDDING_DIM,  # 配置行业嵌入dim。
            dropout=BARRA_GATE_DROPOUT,  # 配置dropout。
            gate_scale=BARRA_GATE_SCALE,  # 配置门控scale。
        ).to(device)
    elif adversarial_barra_enabled():  # 检查上一条件不成立时的备选分支。
        from stage04_no3_adversarial import (
            AdversarialBarraViViT as CoreAdversarialViViT,
        )

        model = CoreAdversarialViViT(  # 创建模型。
            base.OrderFlowViViT(base.MODEL_CONFIG),
            embedding_dim=base.MODEL_CONFIG.embedding_dim,  # 配置嵌入dim。
            style_count=STYLE_COUNT,  # 配置风格数量。
            industry_count=INDUSTRY_COUNT,  # 配置行业数量。
            hidden_dim=ADVERSARIAL_HEAD_HIDDEN_DIM,  # 配置隐藏层dim。
            dropout=ADVERSARIAL_HEAD_DROPOUT,  # 配置dropout。
        ).to(device)
    else:  # 处理上述条件均不成立的情况。
        return base.build_model(  # 返回按当前规则组织后的结果。
            device, use_compile, COMPILE_MODE, TRAIN_BATCH_MODE  # 传入设备、usecompile、compile模式、训练批次模式。
        )
    if use_compile:  # 根据当前数据状态选择处理分支。
        if device.type != "cuda":  # 按当前模式、标识或状态选择处理分支。
            raise RuntimeError("torch.compile 训练要求 CUDA")  # 当前运行环境或任务状态不支持继续执行。
        model = torch.compile(model, mode=COMPILE_MODE, dynamic=True)  # 编译模型计算图。
    return model  # 返回模型。


def adversarial_gamma(epoch: int) -> float:  # 把当前训练配置传给 No.3 的余弦梯度反转调度器。
    """把当前训练配置传给 No.3 的余弦梯度反转调度器。"""
    return cosine_adversarial_gamma(  # 返回完成上述格式转换后的结果。
        epoch,  # 汇入轮次。
        gamma_max=ADVERSARIAL_GAMMA_MAX,  # 配置对抗系数max。
        warmup_epochs=ADVERSARIAL_WARMUP_EPOCHS,  # 配置warmupepochs。
        ramp_epochs=ADVERSARIAL_RAMP_EPOCHS,  # 配置rampepochs。
    )


def forward_training_model(  # 按实验模式执行基线、对抗、门控或双塔模型前向传播。
    model: torch.nn.Module,  # 接收模型。
    videos: torch.Tensor,  # [B,17,20,8,8]，订单流视频。
    frame_mask: torch.Tensor,  # [B,20]，逐帧有效标记。
    *,
    gamma: float,  # 接收对抗系数。
    styles: torch.Tensor | None = None,  # [B,10]，十个连续 Barra 暴露。
    style_valid: torch.Tensor | None = None,  # [B,10]，风格暴露有效掩码。
    industry_codes: torch.Tensor | None = None,  # [B]，行业整数代码。
    sample_indices: torch.Tensor | None = None,  # [B]，全局样本行号。
    conditioning_prepared: bool = False,  # 接收conditioningprepared。
) -> tuple[  # 声明当前函数的返回类型。
    torch.Tensor,  # 声明当前返回值中的一个数据项。
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """按实验模式执行基线、对抗、门控或双塔模型前向传播。"""
    if dual_tower_enabled():  # 根据当前数据状态选择处理分支。
        if (  # 开始组合多项有效性或配置条件。
            styles is None
            or style_valid is None
            or industry_codes is None
            or sample_indices is None
        ):
            raise ValueError("No.6 前向缺少 Barra 条件输入")  # 输入数据不满足业务或数值约束时立即终止。
        if conditioning_prepared:  # 根据当前数据状态选择处理分支。
            alpha_inputs = style_inputs = (styles, style_valid, industry_codes)
        else:  # 处理上述条件均不成立的情况。
            alpha_inputs, style_inputs = prepare_dual_tower_inputs(
                styles, style_valid, industry_codes, sample_indices,  # 传入风格、风格有效、行业代码、样本索引。
                mode=BARRA_DUAL_TOWER_MODE,  # 选择当前处理模式。
            )
        alpha, style, total, gate, representation = model(  # [B]、[B]、[B]、[B,D]、[B,D]。
            videos, frame_mask, *alpha_inputs, *style_inputs
        )
        return alpha, None, None, representation, gate, style, total  # 返回按当前规则组织后的结果。
    if barra_conditioning_enabled():  # 根据当前数据状态选择处理分支。
        if (  # 开始组合多项有效性或配置条件。
            styles is None
            or style_valid is None
            or industry_codes is None
            or sample_indices is None
        ):
            raise ValueError("No.4 前向缺少 Barra 条件输入")  # 输入数据不满足业务或数值约束时立即终止。
        if conditioning_prepared:  # 根据当前数据状态选择处理分支。
            conditional = (styles, style_valid, industry_codes)
        else:  # 处理上述条件均不成立的情况。
            conditional = prepare_barra_conditioning_inputs(
                styles, style_valid, industry_codes, sample_indices,  # 传入风格、风格有效、行业代码、样本索引。
                mode=BARRA_CONDITIONING_MODE,  # 选择当前处理模式。
            )
        raw_score, gate, representation = model(  # [B]、[B,D]、[B,D]。
            videos, frame_mask, *conditional
        )
        return raw_score, None, None, representation, gate, None, None  # 返回按当前规则组织后的结果。
    if adversarial_barra_enabled():  # 根据当前数据状态选择处理分支。
        coefficient = torch.tensor(  # []，梯度反转强度标量。
            gamma, dtype=torch.float32, device=videos.device
        )
        raw, style, industry, representation = model(  # [B]、[B,10]、[B,32]、[B,D]。
            videos, frame_mask, coefficient  # 传入视频、帧掩码、系数。
        )
        return raw, style, industry, representation, None, None, None  # 返回按当前规则组织后的结果。
    return model(videos, frame_mask), None, None, None, None, None, None  # 返回按当前规则组织后的结果。


# %% [markdown]
# ## 日截面数据加载与风格暴露审计
#
# 标签、十风格矩阵与 feature store 共享同一个 `sample_index`。训练、验证、
# 推理都使用完整交易日 batch。缺失暴露不前向填充，而是在当日有效样本标准化
# 后填 0，并保留有效标记。风格矩阵始终用于相关性审计；开启标签残差化时选择
# 预先生成的残差标签，开启模型分数投影时进入可微 QR 投影并由投影分数计算
# 训练损失。


# %%
class StyleNeutralBatchCollator:  # 读取视频、原标签、风格残差标签和十风格暴露。
    """读取视频、原标签、风格残差标签和十风格暴露。
    """

    def __init__(self) -> None:  # 初始化当前对象的配置、状态与子组件。
        self.video_collator = sequence_data.VideoBatchCollator()  # 整理并搬运 batch。
        self.residual_targets = None  # 先把残差标签初始化为空，后续按条件填充。
        self.full_barra_targets = None  # 先把fullBarra标签初始化为空，后续按条件填充。
        self.dual_tower_fitted_targets = None  # 先把dualtower拟合标签初始化为空，后续按条件填充。
        self.style_exposures = None  # 先把风格暴露初始化为空，后续按条件填充。
        self.style_valid = None  # 先把风格有效初始化为空，后续按条件填充。
        self.industry_codes = None  # 先把行业代码初始化为空，后续按条件填充。

    def _ensure_open(self) -> None:  # 在当前 worker 内惰性打开风格、行业和派生标签 memmap。
        """在当前 worker 内惰性打开风格、行业和派生标签 memmap。"""
        if self.residual_targets is None:  # 检查所需对象是否尚未初始化。
            store = style_data.DEFAULT_STORE_DIR
            self.residual_targets = np.load(  # 保存残差标签，供该对象后续步骤复用。
                store / style_data.RESIDUAL_TARGETS_FILE, mmap_mode="r"
            )
            self.style_exposures = np.load(  # 保存风格暴露，供该对象后续步骤复用。
                store / style_data.STYLE_EXPOSURES_FILE, mmap_mode="r"
            )
            self.style_valid = np.load(  # 保存风格有效，供该对象后续步骤复用。
                store / style_data.STYLE_VALID_FILE, mmap_mode="r"
            )
            if barra_conditioning_enabled() or dual_tower_enabled():  # 根据当前数据状态选择处理分支。
                style_data.require_full_barra_target_sidecar()
                self.full_barra_targets = np.load(  # 保存fullBarra标签，供该对象后续步骤复用。
                    style_data.DEFAULT_FULL_BARRA_TARGET_STORE_DIR  # 汇入风格数据默认fullBarra标签存储目录。
                    / style_data.FULL_BARRA_TARGETS_FILE,
                    mmap_mode="r",  # 使用内存映射避免一次性载入全部数据。
                )
            if dual_tower_enabled():  # 根据当前数据状态选择处理分支。
                style_data.require_dual_tower_target_sidecar()
                self.dual_tower_fitted_targets = np.load(  # 保存dualtower拟合标签，供该对象后续步骤复用。
                    style_data.DEFAULT_DUAL_TOWER_TARGET_STORE_DIR  # 汇入风格数据默认dualtower标签存储目录。
                    / style_data.DUAL_TOWER_FITTED_TARGETS_FILE,
                    mmap_mode="r",  # 使用内存映射避免一次性载入全部数据。
                )
            if industry_sidecar_required():  # 根据当前数据状态选择处理分支。
                style_data.require_industry_sidecar()
                self.industry_codes = np.load(  # 保存行业代码，供该对象后续步骤复用。
                    style_data.DEFAULT_INDUSTRY_STORE_DIR  # 汇入风格数据默认行业存储目录。
                    / style_data.INDUSTRY_CODES_FILE, mmap_mode="r"
                )

    def __call__(self, sample_indices: list[int]) -> dict[str, torch.Tensor]:  # 按样本行号组装视频、Barra 条件、派生标签和真实审计暴露。
        """按样本行号组装视频、Barra 条件、派生标签和真实审计暴露。"""
        self._ensure_open()
        batch = self.video_collator(sample_indices)  # 整理并搬运 batch。
        indices = batch["sample_indices"].numpy().astype(np.int64, copy=False)  # 转换索引的数据类型或存储位置。
        batch["original_targets"] = batch["targets"]
        batch["residual_targets"] = torch.from_numpy(  # [B]，风格残差标签。
            np.array(self.residual_targets[indices], dtype=np.float32, copy=True)
        )
        if RESIDUALIZE_TARGET:  # 根据当前数据状态选择处理分支。
            batch["targets"] = batch["residual_targets"]
        if barra_conditioning_enabled() or dual_tower_enabled():  # 根据当前数据状态选择处理分支。
            batch["targets"] = torch.from_numpy(  # [B]，完整 Barra 残差标签。
                np.array(
                    self.full_barra_targets[indices],  # 汇入fullBarra标签。
                    dtype=np.float32,  # 统一结果的数据类型。
                    copy=True,  # 控制是否复制底层数据。
                )
            )
        if dual_tower_enabled():  # 根据当前数据状态选择处理分支。
            batch["dual_tower_fitted_targets"] = torch.from_numpy(  # [B]，Barra 拟合标签。
                np.array(
                    self.dual_tower_fitted_targets[indices],  # 汇入dualtower拟合标签。
                    dtype=np.float32,  # 统一结果的数据类型。
                    copy=True,  # 控制是否复制底层数据。
                )
            )
        else:  # 处理上述条件均不成立的情况。
            batch["dual_tower_fitted_targets"] = torch.zeros_like(  # [B]，非双塔模式占位。
                batch["targets"]  # 汇入批次。
            )
        batch["style_exposures"] = torch.from_numpy(  # [B,10]，十风格暴露。
            np.array(self.style_exposures[indices], dtype=np.float32, copy=True)
        )
        batch["style_valid"] = torch.from_numpy(  # [B,10]，风格有效掩码。
            np.array(self.style_valid[indices], dtype=np.bool_, copy=True)
        )
        if industry_sidecar_required():  # 根据当前数据状态选择处理分支。
            batch["industry_codes"] = torch.from_numpy(  # [B]，行业代码。
                np.array(self.industry_codes[indices], dtype=np.int16, copy=True)
            )
        else:  # 处理上述条件均不成立的情况。
            batch["industry_codes"] = torch.zeros(  # [B]，无需行业数据时使用 UNKNOWN=0。
                len(indices), dtype=torch.int16
            )
        return batch  # 返回批次。


def make_style_loaders(  # 训练、验证和预测全部按完整交易日组织 batch。
    forecast_year: int,  # 接收forecast年份。
    seed: int,  # 接收随机种子。
    workers: int,  # 接收workers。
    pin_memory: bool,  # 接收pinmemory。
    distributed_context: distributed.DistributedContext | None = None,  # 接收distributed上下文。
) -> tuple[DataLoader, DataLoader, DataLoader | None]:  # 声明当前函数的返回类型。
    """训练、验证和预测全部按完整交易日组织 batch。"""

    split = sequence_data.load_sequence_split(forecast_year)
    date_ids = np.load(sequence_data.DATE_IDS_PATH, mmap_mode="r")  # 读取交易日ids。
    generator = torch.Generator().manual_seed(seed)  # 固定 DataLoader 随机源。
    style_common: dict[str, Any] = {  # 初始化风格公共容器。
        "num_workers": workers,  # 记录numworkers。
        "pin_memory": pin_memory,  # 记录pinmemory。
        "collate_fn": StyleNeutralBatchCollator(),  # 记录collatefn。
    }
    if workers > 0:  # 检查当前数值是否越过业务或数值阈值。
        style_common.update({"persistent_workers": True, "prefetch_factor": 1})  # 用当前配置更新参数字典。

    def style_loader(indices: np.ndarray, *, shuffle: bool) -> DataLoader:  # 处理连续风格数据、有效掩码或相应诊断。
        """处理连续风格数据、有效掩码或相应诊断。"""
        sampler_class = (  # 按完整交易日截面组 batch。
            distributed.DistributedTradeDateBatchSampler  # 汇入distributeddistributedtradedatebatchsampler。
            if distributed_context is not None else TradeDateBatchSampler  # 检查必需数据、配置或校验结果是否缺失。
        )
        distributed_kwargs = (
            {"rank": distributed_context.rank, "world_size": distributed_context.world_size}
            if distributed_context is not None else {}  # 检查必需数据、配置或校验结果是否缺失。
        )
        sampler = sampler_class(  # 创建sampler。
            indices, date_ids, seed=seed, shuffle=shuffle, **distributed_kwargs
        )
        return DataLoader(  # 返回处理连续风格数据、有效掩码或相应诊断。
            sequence_data.SampleIndexDataset(indices),
            batch_sampler=sampler,  # 配置批次sampler。
            generator=generator,  # 传入固定随机数生成器以保证复现。
            **style_common,
        )

    def raw_prediction_loader(indices: np.ndarray) -> DataLoader:  # 处理预测文件、预测列或对应的样本数量审计。
        """处理预测文件、预测列或对应的样本数量审计。"""
        raw_common: dict[str, Any] = {  # 初始化原始公共容器。
            "num_workers": workers,  # 记录numworkers。
            "pin_memory": pin_memory,  # 记录pinmemory。
            "collate_fn": sequence_data.VideoBatchCollator(),  # 记录collatefn。
        }
        if workers > 0:  # 检查当前数值是否越过业务或数值阈值。
            raw_common.update(  # 用当前配置更新参数字典。
                {"persistent_workers": True, "prefetch_factor": 1}
            )
        sampler_class = (  # 按完整交易日截面组 batch。
            distributed.DistributedTradeDateBatchSampler  # 汇入distributeddistributedtradedatebatchsampler。
            if distributed_context is not None else TradeDateBatchSampler  # 检查必需数据、配置或校验结果是否缺失。
        )
        distributed_kwargs = (
            {"rank": distributed_context.rank, "world_size": distributed_context.world_size}
            if distributed_context is not None else {}  # 检查必需数据、配置或校验结果是否缺失。
        )
        sampler = sampler_class(  # 创建sampler。
            indices, date_ids, seed=seed, shuffle=False, **distributed_kwargs
        )
        return DataLoader(  # 返回处理预测文件、预测列或对应的样本数量审计。
            sequence_data.SampleIndexDataset(indices),
            batch_sampler=sampler,  # 配置批次sampler。
            generator=generator,  # 传入固定随机数生成器以保证复现。
            **raw_common,
        )

    train = style_loader(split["train_indices"], shuffle=True)
    validation = style_loader(split["validation_indices"], shuffle=False)
    prediction = None  # 先把预测初始化为空，后续按条件填充。
    if len(split["prediction_indices"]):  # 检查当前样本数量是否满足计算要求。
        prediction = (
            raw_prediction_loader(split["prediction_indices"])
            if (  # 开始组合多项有效性或配置条件。
                not barra_conditioning_enabled()
                and not dual_tower_enabled()
                and (raw_barra_r2_enabled() or adversarial_barra_enabled())
            )
            else style_loader(split["prediction_indices"], shuffle=False)
        )
    return train, validation, prediction  # 返回训练、验证和预测全部按完整交易日组织 batch。


def training_industry_class_weights(train_indices: np.ndarray) -> torch.Tensor:  # 由当前滚动训练区间生成32行业逆平方根频数权重。
    """由当前滚动训练区间生成32行业逆平方根频数权重。"""

    style_data.require_industry_sidecar()
    codes = np.load(  # 读取代码。
        style_data.DEFAULT_INDUSTRY_STORE_DIR / style_data.INDUSTRY_CODES_FILE,
        mmap_mode="r",  # 使用内存映射避免一次性载入全部数据。
    )
    selected = np.asarray(codes[train_indices], dtype=np.int64)  # 转换并统一筛选结果的数据表示。
    known = selected[selected > 0] - 1  # 构造已知行业，限定后续参与计算的样本。
    counts = np.bincount(known, minlength=INDUSTRY_COUNT).astype(np.float64)  # 计算数量。
    if len(counts) != INDUSTRY_COUNT or (counts <= 0).any():  # 检查当前样本数量是否满足计算要求。
        raise ValueError("滚动训练区间未覆盖全部32个真实行业")  # 输入数据不满足业务或数值约束时立即终止。
    weights = np.reciprocal(np.sqrt(counts))  # 按截面样本数平方根加权。
    weights /= weights.mean()
    return torch.from_numpy(weights.astype(np.float32))  # [32]，有效行业类别权重。


class StyleCudaPrefetcher:  # 将视频、双标签和风格矩阵一起异步搬到 GPU。
    """将视频、双标签和风格矩阵一起异步搬到 GPU。
    """

    def __init__(self, loader: DataLoader, device: torch.device, enabled: bool) -> None:  # 初始化当前对象的配置、状态与子组件。
        self.loader = loader  # 保存数据加载器，供该对象后续步骤复用。
        self.device = device  # 保存设备，供该对象后续步骤复用。
        self.enabled = enabled and device.type == "cuda"

    def _move(  # 把风格训练批次的所有张量非阻塞搬到 CUDA。
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, ...]:  # 声明当前函数的返回类型。
        """把风格训练批次的所有张量非阻塞搬到 CUDA。"""
        non_blocking = self.device.type == "cuda"  # 允许设备拷贝与计算异步重叠。
        names = (
            "videos", "frame_mask", "targets", "original_targets",
            "residual_targets", "dual_tower_fitted_targets",
            "style_exposures", "style_valid",
            "industry_codes",
        )
        return tuple(  # 返回完成上述格式转换后的结果。
            batch[name].to(self.device, non_blocking=non_blocking)
            for name in names  # 遍历名称，逐项处理名称。
        )

    def __iter__(self) -> Iterator[tuple[Any, ...]]:  # 按既定顺序迭代 StyleCudaPrefetcher 中的训练批次。
        """按既定顺序迭代 StyleCudaPrefetcher 中的训练批次。"""
        if not self.enabled:  # 检查必需数据、配置或校验结果是否缺失。
            for batch in self.loader:  # 遍历数据加载器，逐项处理批次。
  # 对 `(yield (batch, *self._move(batch)))` 求值，以触发当前步骤需要的计算。
                yield (batch, *self._move(batch))  # 产出当前批次并保留迭代状态。
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
                tensors = self._move(batch)
            next_item = (batch, *tensors)

        preload()  # 异步预取下一批数据。
        while next_item is not None:  # 在终止条件出现前持续迭代。
            torch.cuda.current_stream(self.device).wait_stream(stream)  # 等待预取流完成。
            current = next_item
            for tensor in current[1:]:  # 遍历当前，逐项处理张量。
                tensor.record_stream(torch.cuda.current_stream(self.device))  # 绑定当前 CUDA stream 生命周期。
            preload()  # 异步预取下一批数据。
  # 对 `(yield current)` 求值，以触发当前步骤需要的计算。
            yield current  # 产出当前批次并保留迭代状态。


def training_config() -> base.TrainingConfig:  # 汇总当前风格中性实验的结构、标签、损失和早停配置。
    """汇总当前风格中性实验的结构、标签、损失和早停配置。"""
    return base.TrainingConfig(  # 返回汇总当前风格中性实验的结构、标签、损失和早停配置。
        train_batch_mode=TRAIN_BATCH_MODE,  # 配置训练批次模式。
        loss_name=LOSS_NAME,  # 配置损失名称。
        rank_temperature=RANK_TEMPERATURE,  # 配置排名temperature。
        turnover_lambda=TURNOVER_LAMBDA,  # 配置turnoverlambda。
        huber_delta=HUBER_DELTA,  # 配置huberdelta。
        smooth_l1_beta=SMOOTH_L1_BETA,  # 配置smoothl1beta。
        hybrid_weight=HYBRID_WEIGHT,  # 配置hybrid权重。
        validation_metric=VALIDATION_METRIC,  # 配置验证指标。
        learning_rate=LEARNING_RATE,  # 配置learning比率。
        max_epochs=MAX_EPOCHS,  # 配置maxepochs。
        early_stopping_patience=PATIENCE,  # 配置earlystoppingpatience。
    )


def training_loss(  # 现有预测损失。
    predictions: torch.Tensor,  # [B]，当前 batch 分数。
    targets: torch.Tensor,  # [B]，训练标签。
) -> torch.Tensor:  # 声明当前函数的返回类型。
    """现有预测损失；sigmoid soft rank 对预测端保持反向传播。"""

    return compute_training_loss(  # 返回现有预测损失。
        predictions,  # 汇入预测。
        targets,  # 汇入标签。
        loss_name=LOSS_NAME,  # 配置损失名称。
        rank_temperature=RANK_TEMPERATURE,  # 配置排名temperature。
        turnover_lambda=TURNOVER_LAMBDA,  # 配置turnoverlambda。
        huber_delta=HUBER_DELTA,  # 配置huberdelta。
        smooth_l1_beta=SMOOTH_L1_BETA,  # 配置smoothl1beta。
        hybrid_weight=HYBRID_WEIGHT,  # 配置hybrid权重。
    )


def rank_loss(  # 兼容旧测试和调用方的训练损失别名。
    predictions: torch.Tensor,  # [B]，当前 batch 分数。
    targets: torch.Tensor,  # [B]，训练标签。
) -> torch.Tensor:  # 声明当前函数的返回类型。
    """兼容旧测试和调用方的训练损失别名。"""

    return training_loss(predictions, targets)  # 返回兼容旧测试和调用方的训练损失别名。


def stable_sigmoid_rank_loss(  # 与现有 Sigmoid SoftRankIC 等价，并稳定支持常数初始分数。
    predictions: torch.Tensor,  # [B]，当前 batch 分数。
    targets: torch.Tensor,  # [B]，训练标签。
) -> torch.Tensor:  # 声明当前函数的返回类型。
    """与现有 Sigmoid SoftRankIC 等价，并稳定支持常数初始分数。"""

    prediction_ranks = sigmoid_soft_rank(  # 计算预测排名。
        predictions, RANK_TEMPERATURE  # 传入预测、排名temperature。
    )
    with torch.no_grad():  # 关闭梯度记录以降低验证或推理开销。
        target_ranks = sigmoid_soft_rank(  # 计算标签排名。
            targets.detach(), RANK_TEMPERATURE
        )
    centered_prediction = prediction_ranks - prediction_ranks.mean()  # 计算中心化预测。
    centered_target = target_ranks - target_ranks.mean()  # 计算中心化标签。
    variance_product = (  # 计算方差product。
        centered_prediction.square().sum()
        * centered_target.square().sum()
    )
    denominator = torch.sqrt(variance_product.clamp_min(1e-8))  # 计算截面标准差。
    correlation = (  # 计算相关系数。
        centered_prediction.mul(centered_target).sum() / denominator
    )
    return -correlation  # 返回与现有 Sigmoid SoftRankIC 等价，并稳定支持常数初始分数。


def masked_style_regression_loss(  # 仅在有效暴露位置计算十风格对抗回归损失。
    predictions: torch.Tensor,  # [B]，当前 batch 分数。
    targets: torch.Tensor,  # [B]，训练标签。
    valid: torch.Tensor,  # [B]，样本有效掩码。
) -> torch.Tensor:  # 声明当前函数的返回类型。
    """仅在有效暴露位置计算十风格对抗回归损失。"""
    if predictions.shape != targets.shape or valid.shape != targets.shape:  # 检查张量或数组形状是否符合接口约定。
        raise ValueError("风格对抗头、标签或有效掩码形状不一致")  # 输入数据不满足业务或数值约束时立即终止。
    mask = valid.bool()  # 构造掩码，限定后续参与计算的样本。
    if not mask.any():  # 检查当前掩码或数据是否满足有效性要求。
        raise ValueError("当前日批次没有有效风格监督")  # 输入数据不满足业务或数值约束时立即终止。
    values = (predictions.float() - targets.float()).square()[mask]  # 转换值的数据类型或存储位置。
    if not torch.isfinite(values).all():  # 检查参与计算的数值是否全部有限。
        raise FloatingPointError("风格对抗损失含 NaN/Inf")  # 检测到 NaN、Inf 或退化数值时立即终止。
    return values.mean()  # 返回仅在有效暴露位置计算十风格对抗回归损失。


def normalized_industry_classification_loss(  # 计算忽略 UNKNOWN 且按行业加权归一化的分类损失。
    logits: torch.Tensor,  # [B,32]，有效行业分类 logits。
    industry_codes: torch.Tensor,  # [B]，行业整数代码。
    class_weights: torch.Tensor,  # [32]，有效行业类别权重。
) -> torch.Tensor:  # 声明当前函数的返回类型。
    """计算忽略 UNKNOWN 且按行业加权归一化的分类损失。"""
    if logits.ndim != 2 or logits.shape[1] != INDUSTRY_COUNT:  # 检查张量或数组形状是否符合接口约定。
        raise ValueError("行业对抗头必须输出 [N,32]")  # 输入数据不满足业务或数值约束时立即终止。
    codes = industry_codes.long().reshape(-1)  # [B,...] -> [B]。
    if len(codes) != len(logits):  # 检查当前样本数量是否满足计算要求。
        raise ValueError("行业标签与对抗头行数不一致")  # 输入数据不满足业务或数值约束时立即终止。
    known = codes > 0  # 构造已知行业，限定后续参与计算的样本。
    if not known.any():  # 检查当前掩码或数据是否满足有效性要求。
        raise ValueError("当前日批次没有已知行业监督")  # 输入数据不满足业务或数值约束时立即终止。
    targets = codes[known] - 1  # [B_known]，映射为 0 至 31。
    if int(targets.min()) < 0 or int(targets.max()) >= INDUSTRY_COUNT:  # 检查当前数值是否越过业务或数值阈值。
        raise ValueError("行业标签超出1至32的有效范围")  # 输入数据不满足业务或数值约束时立即终止。
    weights = class_weights.to(device=logits.device, dtype=torch.float32)  # 转换权重的数据类型或存储位置。
    if weights.shape != (INDUSTRY_COUNT,) or not torch.isfinite(weights).all():  # 检查参与计算的数值是否全部有限。
        raise ValueError("行业类别权重形状或数值异常")  # 输入数据不满足业务或数值约束时立即终止。
    loss = torch.nn.functional.cross_entropy(  # 计算损失。
        logits.float()[known], targets, weight=weights  # [B_known,32] 与 [B_known]。
    ) / math.log(INDUSTRY_COUNT)
    if not torch.isfinite(loss):  # 检查参与计算的数值是否全部有限。
        raise FloatingPointError("行业对抗损失含 NaN/Inf")  # 检测到 NaN、Inf 或退化数值时立即终止。
    return loss  # 返回损失。


def adversarial_training_loss(  # 组合收益排序、连续风格对抗和行业对抗损失。
    style_predictions: torch.Tensor | None,  # [B,10]，风格预测。
    industry_logits: torch.Tensor | None,  # [B,32]，有效行业分类 logits。
    styles: torch.Tensor,  # [B,10]，十个连续 Barra 暴露。
    style_valid: torch.Tensor,  # [B,10]，风格暴露有效掩码。
    industry_codes: torch.Tensor,  # [B]，行业整数代码。
    class_weights: torch.Tensor,  # [32]，有效行业类别权重。
) -> tuple[torch.Tensor, torch.Tensor]:  # 声明当前函数的返回类型。
    """组合收益排序、连续风格对抗和行业对抗损失。"""
    zero = torch.zeros((), dtype=torch.float32, device=styles.device)  # 初始化zero。
    if not adversarial_barra_enabled():  # 检查必需数据、配置或校验结果是否缺失。
        return zero, zero  # 返回组合收益排序、连续风格对抗和行业对抗损失。
    if style_predictions is None or industry_logits is None:  # 检查所需对象是否尚未初始化。
        raise ValueError("No.3 缺少风格或行业对抗头输出")  # 输入数据不满足业务或数值约束时立即终止。
    return (  # 返回组合收益排序、连续风格对抗和行业对抗损失。
        masked_style_regression_loss(
            style_predictions, styles, style_valid  # 传入风格预测、风格、风格有效。
        ),
        normalized_industry_classification_loss(
            industry_logits, industry_codes, class_weights  # 传入行业logits、行业代码、class权重。
        ),
    )


def _joint_r_squared_value(  # 把 Barra R² 公共接口收窄为标量，供方案损失组合。
    predictions: torch.Tensor,  # [B]，当前 batch 分数。
    styles: torch.Tensor,  # [B,10]，十个连续 Barra 暴露。
    industries: torch.Tensor,  # [B]，行业整数代码。
) -> torch.Tensor:  # 声明当前函数的返回类型。
    """把 Barra R² 公共接口收窄为标量，供方案损失组合。"""

    from stage02_barra_algebra import joint_barra_r_squared

    value, _ = joint_barra_r_squared(  # 按股票与交易日业务键对齐。
        predictions, styles, industries  # 传入预测、风格、行业。
    )
    return value  # 返回值。


def combined_training_loss(  # 返回总损失、预测损失和 raw-score 联合 Barra R²。
    raw_predictions: torch.Tensor,  # [B]，原始预测分数。
    effective: torch.Tensor,  # [B]，实际用于损失或发布的分数。
    targets: torch.Tensor,  # [B]，训练标签。
    style_exposures: torch.Tensor,  # [B,10]，十个连续 Barra 暴露。
    industry_codes: torch.Tensor,  # [B]，行业整数代码。
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # 声明当前函数的返回类型。
    """返回总损失、预测损失和 raw-score 联合 Barra R²。"""

    predictive_loss = training_loss(effective, targets)  # 按实验口径计算可微损失。
    if not raw_barra_r2_enabled():  # 检查必需数据、配置或校验结果是否缺失。
        barra_r_squared = torch.zeros(  # 初始化Barrarsquared。
            (), dtype=torch.float32, device=raw_predictions.device
        )
        return predictive_loss, predictive_loss, barra_r_squared  # 返回总损失、预测损失和 raw-score 联合 Barra R²。

    from stage04_no2_r2_penalty import (
        r_squared_regularized_loss,
    )

    components = r_squared_regularized_loss(  # 按实验口径计算可微损失。
        raw_predictions,  # 汇入原始预测。
        targets,  # 汇入标签。
        style_exposures,  # 汇入风格暴露。
        industry_codes,  # 汇入行业代码。
        rank_loss=lambda _score, _target: predictive_loss,  # 配置排名损失。
        barra_r_squared=_joint_r_squared_value,  # 配置Barrarsquared。
        penalty_weight=RAW_BARRA_R2_LAMBDA,  # 配置penalty权重。
    )
    return (  # 返回总损失、预测损失和 raw-score 联合 Barra R²。
        components["loss"],  # 汇入components。
        components["predictive_loss"],  # 汇入components。
        components["barra_r_squared"],  # 汇入components。
    )


def dual_tower_training_loss(  # No.6 四项损失。
    alpha_score: torch.Tensor,  # [B]，Alpha 塔分数。
    style_score: torch.Tensor,  # [B]，Style 塔分数。
    total_score: torch.Tensor,  # [B]，Alpha 与 Style 合计分数。
    original_targets: torch.Tensor,  # [B]，原始收益标签。
    residual_targets: torch.Tensor,  # [B]，Barra 残差标签。
    fitted_targets: torch.Tensor,  # [B]，Barra 拟合标签。
    style_exposures: torch.Tensor,  # [B,10]，十个连续 Barra 暴露。
    industry_codes: torch.Tensor,  # [B]，行业整数代码。
) -> dict[str, torch.Tensor]:  # 声明当前函数的返回类型。
    """No.6 四项损失；风险惩罚只施加于 Alpha 塔。"""

    if not dual_tower_enabled():  # 检查必需数据、配置或校验结果是否缺失。
        raise RuntimeError("仅 No.6 可以调用双塔损失")  # 当前运行环境或任务状态不支持继续执行。
    from stage04_no6_dual_tower import (
        dual_tower_training_loss as compute_dual_tower_loss,
    )

    return compute_dual_tower_loss(  # 返回No.6 四项损失。
        alpha_score,  # 汇入Alpha分数。
        style_score,  # 汇入风格分数。
        total_score,  # 汇入total分数。
        original_targets,  # 汇入原始标签。
        residual_targets,  # 汇入残差标签。
        fitted_targets,  # 汇入拟合标签。
        style_exposures,  # 汇入风格暴露。
        industry_codes,  # 汇入行业代码。
        return_rank_loss=training_loss,  # 返回No.6 四项损失。
        alpha_rank_loss=training_loss,  # 配置Alpha排名损失。
        style_rank_loss=stable_sigmoid_rank_loss,  # 配置风格排名损失。
        barra_r_squared=_joint_r_squared_value,  # 配置Barrarsquared。
        alpha_loss_weight=DUAL_TOWER_ALPHA_LOSS_WEIGHT,  # 配置Alpha损失权重。
        style_loss_weight=DUAL_TOWER_STYLE_LOSS_WEIGHT,  # 配置风格损失权重。
        r_squared_weight=RAW_BARRA_R2_LAMBDA,  # 配置rsquared权重。
    )


class FullBarraOutputNeutralizationLayer(torch.nn.Module):  # 无参数的每日等权完整 Barra 可微中性化层。
    """无参数的每日等权完整 Barra 可微中性化层。
    """

    def forward(  # 执行当前模块的前向计算。
        self,  # 汇入self。
        raw_predictions: torch.Tensor,  # [B]，原始预测分数。
        style_exposures: torch.Tensor,  # [B,10]，十个连续 Barra 暴露。
        industry_codes: torch.Tensor,  # [B]，行业整数代码。
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:  # 声明当前函数的返回类型。
        """执行 FullBarraOutputNeutralizationLayer 的前向计算并返回约定张量。"""
        return style_data.project_out_full_barra_torch(  # 返回上述处理结果。
            raw_predictions,  # 汇入原始预测。
            style_exposures,  # 汇入风格暴露。
            industry_codes,  # 汇入行业代码。
            standardize=True,  # 控制投影结果是否截面标准化。
        )


FULL_BARRA_OUTPUT_NEUTRALIZER = FullBarraOutputNeutralizationLayer()  # 按日投影 Barra 暴露。


def effective_predictions(  # 按实验开关返回损失实际使用及正式发布的分数口径。
    raw_predictions: torch.Tensor,  # [B]，原始预测分数。
    style_exposures: torch.Tensor,  # [B,10]，十个连续 Barra 暴露。
    industry_codes: torch.Tensor | None = None,  # [B]，行业整数代码。
) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:  # 声明当前函数的返回类型。
    """按实验开关返回损失实际使用及正式发布的分数口径。"""
    if strict_barra_output_enabled():  # 根据当前数据状态选择处理分支。
        if industry_codes is None:  # 检查所需对象是否尚未初始化。
            raise ValueError("严格 Barra 输出中性化缺少行业代码")  # 输入数据不满足业务或数值约束时立即终止。
        output, diagnostics = FULL_BARRA_OUTPUT_NEUTRALIZER(  # 按日投影 Barra 暴露。
            raw_predictions,  # 汇入原始预测。
            style_exposures,  # 汇入风格暴露。
            industry_codes,  # 汇入行业代码。
        )
        if (  # 开始组合多项有效性或配置条件。
            float(diagnostics["max_abs_style_pearson"]) > 5e-5
            or float(diagnostics["max_abs_industry_mean"]) > 5e-5
        ):
            raise FloatingPointError(  # 检测到 NaN、Inf 或退化数值时立即终止。
                "严格 Barra 输出未达到十风格或行业数值正交"
            )
        return output, diagnostics  # 返回按当前规则组织后的结果。
    if not PROJECT_MODEL_OUTPUT:  # 检查必需数据、配置或校验结果是否缺失。
        return raw_predictions.float().reshape(-1), None  # [B,...] -> [B]。
    return style_data.project_out_styles_torch(  # 返回按当前规则组织后的结果。
        raw_predictions, style_exposures, standardize=True
    )


# %% [markdown]
# ## 训练、验证和推理
#
# 训练损失和早停指标都使用两个开关选择后的有效标签与有效模型分数。另行记录
# 未投影模型分数对原始标签的 RankIC，作为跨模式一致的审计指标，不参与损失
# 或早停。


# %%
def prepare_distributed_conditioning(  # 准备distributedconditioning。
    styles: torch.Tensor,  # [B,10]，十个连续 Barra 暴露。
    style_valid: torch.Tensor,  # [B,10]，风格暴露有效掩码。
    industry_codes: torch.Tensor,  # [B]，行业整数代码。
    sample_indices: torch.Tensor,  # [B]，全局样本行号。
    context: distributed.DistributedContext | None,  # 接收上下文。
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:  # 声明当前函数的返回类型。
    """执行 prepare_distributed_conditioning 对应的分布式训练处理。"""

    shuffled = (
        dual_tower_enabled() and BARRA_DUAL_TOWER_MODE == "shuffled"
    ) or (
        barra_conditioning_enabled() and BARRA_CONDITIONING_MODE == "shuffled"
    )
    if context is None or not shuffled:  # 检查所需对象是否尚未初始化。
        return styles, style_valid, industry_codes, False  # 返回上述处理结果。
    local_indices = sample_indices.to(styles.device)  # [B]。
    global_indices = distributed.all_gather_tensor(local_indices)  # [N_global]。
    global_styles = distributed.all_gather_tensor(styles)  # [N_global,10]。
    global_valid = distributed.all_gather_tensor(style_valid)  # [N_global,10]。
    global_industry = distributed.all_gather_tensor(industry_codes)  # [N_global]。
    order = torch.argsort(global_indices)  # [N_global]，按样本行号恢复全局顺序。
    global_indices = global_indices[order]
    global_styles = global_styles[order]
    global_valid = global_valid[order]  # 构造全局有效，限定后续参与计算的样本。
    global_industry = global_industry[order]
    mode = (
        BARRA_DUAL_TOWER_MODE  # 汇入Barradualtower模式。
        if dual_tower_enabled() else BARRA_CONDITIONING_MODE  # 根据当前数据状态选择处理分支。
    )
    transformed = prepare_barra_conditioning_inputs(
        global_styles,  # 汇入全局风格。
        global_valid,  # 汇入全局有效。
        global_industry,  # 汇入全局行业。
        global_indices,  # 汇入全局索引。
        mode=mode,  # 选择当前处理模式。
    )
    positions = torch.searchsorted(global_indices, local_indices)  # [B]，本 rank 样本的全局位置。
    if not torch.equal(global_indices[positions], local_indices):  # 检查必需数据、配置或校验结果是否缺失。
        raise AssertionError("DDP Barra 条件输入无法按 sample_index 对齐")  # 关键数据契约被破坏时立即终止。
    return (  # 返回上述处理结果。
        transformed[0][positions],  # 汇入transformed。
        transformed[1][positions],  # 汇入transformed。
        transformed[2][positions],  # 汇入transformed。
        True,  # 汇入true。
    )


def _gather_optional_cross_section(  # 汇聚optionalcrosssection。
    value: torch.Tensor | None,  # [N_local,...]，可选的本 rank 输出。
    order: torch.Tensor,  # [N_global]，全局截面重排索引。
    context: distributed.DistributedContext | None,  # 接收上下文。
    *,
    differentiable: bool,  # 接收differentiable。
) -> torch.Tensor | None:  # 声明当前函数的返回类型。
    """执行 _gather_optional_cross_section 对应的分布式训练处理。"""
    if value is None:  # 检查所需对象是否尚未初始化。
        return None  # 当前分支无可用结果，返回空值。
    if context is None:  # 检查所需对象是否尚未初始化。
        return value  # 返回值。
    gathered = (
        distributed.differentiable_all_gather(value)
        if differentiable else distributed.all_gather_tensor(value)  # 根据当前数据状态选择处理分支。
    )
    return gathered[order]  # 返回上述处理结果。


def gather_style_cross_section(  # 汇聚风格crosssection。
    raw_prediction: torch.Tensor,  # [B]，原始预测分数。
    style_prediction: torch.Tensor | None,  # [B,10]，风格预测。
    industry_logits: torch.Tensor | None,  # [B,32]，有效行业分类 logits。
    gate: torch.Tensor | None,  # [B,D]，条件门控向量。
    dual_style_score: torch.Tensor | None,  # [B]，双塔 Style 分数。
    total_return_score: torch.Tensor | None,  # [B]，总收益分数。
    targets: torch.Tensor,  # [B]，训练标签。
    original_targets: torch.Tensor,  # [B]，原始收益标签。
    residual_targets: torch.Tensor,  # [B]，Barra 残差标签。
    fitted_targets: torch.Tensor,  # [B]，Barra 拟合标签。
    styles: torch.Tensor,  # [B,10]，十个连续 Barra 暴露。
    style_valid: torch.Tensor,  # [B,10]，风格暴露有效掩码。
    industry_codes: torch.Tensor,  # [B]，行业整数代码。
    batch: dict[str, torch.Tensor],
    context: distributed.DistributedContext | None,  # 接收上下文。
    *,
    differentiable_outputs: bool,  # 接收differentiableoutputs。
) -> tuple[  # 声明当前函数的返回类型。
    torch.Tensor,  # 声明当前返回值中的一个数据项。
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor,  # 声明当前返回值中的一个数据项。
    torch.Tensor,  # 声明当前返回值中的一个数据项。
    torch.Tensor,  # 声明当前返回值中的一个数据项。
    torch.Tensor,  # 声明当前返回值中的一个数据项。
    torch.Tensor,  # 声明当前返回值中的一个数据项。
    torch.Tensor,  # 声明当前返回值中的一个数据项。
    torch.Tensor,  # 声明当前返回值中的一个数据项。
    torch.Tensor,  # 声明当前返回值中的一个数据项。
    torch.Tensor,  # 声明当前返回值中的一个数据项。
]:
    """执行 gather_style_cross_section 对应的分布式训练处理。"""
    sample_indices = batch["sample_indices"].to(raw_prediction.device)  # [B]。
    date_ids = batch["date_ids"].to(raw_prediction.device)  # [B]。
    if context is None:  # 检查所需对象是否尚未初始化。
        order = torch.arange(len(sample_indices), device=sample_indices.device)  # [B]。
        global_indices = sample_indices
        global_dates = date_ids
    else:  # 处理上述条件均不成立的情况。
        global_indices = distributed.all_gather_tensor(sample_indices)  # [N_global]。
        global_dates = distributed.all_gather_tensor(date_ids)  # [N_global]。
        order = torch.argsort(global_indices)  # [N_global]，按样本行号恢复全局顺序。
        global_indices = global_indices[order]
        global_dates = global_dates[order]

    def metadata(value: torch.Tensor) -> torch.Tensor:  # 元数据。
        """执行 metadata 对应的分布式训练处理。"""
        if context is None:  # 检查所需对象是否尚未初始化。
            return value  # 返回值。
        return distributed.all_gather_tensor(value)[order]  # 返回上述处理结果。

    return (  # 返回上述处理结果。
        _gather_optional_cross_section(
            raw_prediction, order, context,  # 传入原始预测、排序索引、上下文。
            differentiable=differentiable_outputs,  # 配置differentiable。
        ),
        _gather_optional_cross_section(
            style_prediction, order, context,  # 传入风格预测、排序索引、上下文。
            differentiable=differentiable_outputs,  # 配置differentiable。
        ),
        _gather_optional_cross_section(
            industry_logits, order, context,  # 传入行业logits、排序索引、上下文。
            differentiable=differentiable_outputs,  # 配置differentiable。
        ),
        _gather_optional_cross_section(
            gate.detach() if gate is not None else None,
            order,  # 汇入排序索引。
            context,  # 汇入上下文。
            differentiable=False,  # 配置differentiable。
        ),
        _gather_optional_cross_section(
            dual_style_score, order, context,  # 传入dual风格分数、排序索引、上下文。
            differentiable=differentiable_outputs,  # 配置differentiable。
        ),
        _gather_optional_cross_section(
            total_return_score, order, context,  # 传入total收益分数、排序索引、上下文。
            differentiable=differentiable_outputs,  # 配置differentiable。
        ),
        metadata(targets),
        metadata(original_targets),
        metadata(residual_targets),
        metadata(fitted_targets),
        metadata(styles),
        metadata(style_valid),
        metadata(industry_codes),
        global_indices,  # 汇入全局索引。
        global_dates,  # 汇入全局交易日。
    )


def train_one_epoch(  # 训练一个完整 epoch，并返回损失与批次诊断。
    model: torch.nn.Module,  # 接收模型。
    loader: DataLoader,  # 接收数据加载器。
    optimizer: torch.optim.Optimizer,  # 接收优化器。
    device: torch.device,  # 接收设备。
    *,
    epoch: int,  # 接收轮次。
    industry_class_weights: torch.Tensor,  # [32]，有效行业类别权重。
    max_batches: int | None = None,  # 接收maxbatches。
    distributed_context: distributed.DistributedContext | None = None,  # 接收distributed上下文。
) -> dict[str, float | int]:  # 声明当前函数的返回类型。
    """训练一个完整 epoch，并返回损失与批次诊断。"""
    model.train()  # 启用训练模式。
    gamma = adversarial_gamma(epoch) if adversarial_barra_enabled() else 0.0
    totals = {  # 初始化totals容器。
        "loss": 0.0,  # 记录损失。
        "rank_loss": 0.0,  # 记录秩损失。
        "return_rank_loss": 0.0,  # 记录收益秩损失。
        "alpha_rank_loss": 0.0,  # 记录Alpha秩损失。
        "dual_style_rank_loss": 0.0,  # 记录双塔风格秩损失。
        "barra_r_squared": 0.0,  # 记录Barrarsquared。
        "effective_pearson": 0.0,  # 记录有效Pearson。
        "original_pearson": 0.0,  # 记录原始Pearson。
        "raw_style_max": 0.0,  # 记录原始风格max。
        "raw_style_mean": 0.0,  # 记录原始风格均值。
        "effective_style_max": 0.0,  # 记录有效风格max。
        "effective_style_mean": 0.0,  # 记录有效风格均值。
        "adversarial_style_loss": 0.0,  # 记录adversarial风格损失。
        "adversarial_industry_loss": 0.0,  # 记录adversarial行业损失。
        "adversarial_industry_accuracy": 0.0,  # 记录adversarial行业accuracy。
        "adversarial_unknown_rate": 0.0,  # 记录adversarial未知行业比例。
        "gate_mean_abs": 0.0,  # 记录门控均值abs。
        "gate_max_abs": 0.0,  # 记录门控maxabs。
    }
    samples = 0
    batches = 0
    started = time.perf_counter()
    for batch_number, item in enumerate(  # 遍历enumerate，逐项处理批次numberitem。
        StyleCudaPrefetcher(loader, device, bool(USE_CUDA_PREFETCH)), start=1
    ):
        (
            batch, videos, frame_mask, targets, original_targets,  # 传入批次、视频、帧掩码、标签、原始标签。
            _residual_targets, fitted_targets, styles, style_valid,  # 传入残差标签、拟合标签、风格、风格有效。
            industry_codes,  # 汇入行业代码。
        ) = item
        date_ids = batch["date_ids"]
        if int(date_ids.min()) != int(date_ids.max()):  # 按当前模式、标识或状态选择处理分支。
            raise AssertionError("训练 batch 混入多个交易日")  # 关键数据契约被破坏时立即终止。
        if not torch.isfinite(targets).all():  # 检查参与计算的数值是否全部有限。
            raise FloatingPointError("训练目标含 NaN/Inf")  # 检测到 NaN、Inf 或退化数值时立即终止。

        (  # 清空上一批梯度。
            conditioning_styles,  # 汇入conditioning风格。
            conditioning_valid,  # 汇入conditioning有效。
            conditioning_industry,  # 汇入conditioning行业。
            conditioning_prepared,  # 汇入conditioningprepared。
        ) = prepare_distributed_conditioning(
            styles, style_valid, industry_codes, batch["sample_indices"], distributed_context
        )
        optimizer.zero_grad(set_to_none=True)  # 清空上一轮优化残留的梯度。
        with base.autocast_context(device):  # 在自动混合精度上下文中执行前向计算。
            (  # 执行当前 batch 前向传播。
                raw_predictions,  # 汇入原始预测。
                style_predictions,  # 汇入风格预测。
                industry_logits,  # 汇入行业logits。
                _representation,  # 汇入表征。
                gate,  # 汇入门控。
                dual_style_score,  # 汇入dual风格分数。
                total_return_score,  # 汇入total收益分数。
            ) = forward_training_model(
                model,  # 汇入模型。
                videos,  # 汇入视频。
                frame_mask,  # 汇入帧掩码。
                gamma=gamma,  # 配置对抗系数。
                styles=conditioning_styles,  # 配置风格。
                style_valid=conditioning_valid,  # 配置风格有效。
                industry_codes=conditioning_industry,  # 配置行业代码。
                sample_indices=batch["sample_indices"],  # 配置样本索引。
                conditioning_prepared=conditioning_prepared,  # 配置conditioningprepared。
            )
        (
            raw_predictions,  # 汇入原始预测。
            style_predictions,  # 汇入风格预测。
            industry_logits,  # 汇入行业logits。
            gate,  # 汇入门控。
            dual_style_score,  # 汇入dual风格分数。
            total_return_score,  # 汇入total收益分数。
            targets,  # 汇入标签。
            original_targets,  # 汇入原始标签。
            _residual_targets,  # 汇入残差标签。
            fitted_targets,  # 汇入拟合标签。
            styles,  # 汇入风格。
            style_valid,  # 汇入风格有效。
            industry_codes,  # 汇入行业代码。
            _global_sample_indices,  # 汇入全局样本索引。
            _global_date_ids,  # 汇入全局交易日ids。
        ) = gather_style_cross_section(
            raw_predictions,  # 汇入原始预测。
            style_predictions,  # 汇入风格预测。
            industry_logits,  # 汇入行业logits。
            gate,  # 汇入门控。
            dual_style_score,  # 汇入dual风格分数。
            total_return_score,  # 汇入total收益分数。
            targets,  # 汇入标签。
            original_targets,  # 汇入原始标签。
            _residual_targets,  # 汇入残差标签。
            fitted_targets,  # 汇入拟合标签。
            styles,  # 汇入风格。
            style_valid,  # 汇入风格有效。
            industry_codes,  # 汇入行业代码。
            batch,  # 汇入批次。
            distributed_context,  # 汇入distributed上下文。
            differentiable_outputs=True,  # 配置differentiableoutputs。
        )
        predictions, _ = effective_predictions(
            raw_predictions, styles, industry_codes  # 传入原始预测、风格、行业代码。
        )
        raw_style = measure_style_correlations(raw_predictions, styles)  # 计算当前截面相关性。
        effective_style = measure_style_correlations(predictions, styles)  # 计算当前截面相关性。
        if dual_tower_enabled():  # 根据当前数据状态选择处理分支。
            if dual_style_score is None or total_return_score is None:  # 检查所需对象是否尚未初始化。
                raise ValueError("No.6 缺少 Style 或 Total 分数")  # 输入数据不满足业务或数值约束时立即终止。
            dual_losses = dual_tower_training_loss(  # 按实验口径计算可微损失。
                raw_predictions,  # 汇入原始预测。
                dual_style_score,  # 汇入dual风格分数。
                total_return_score,  # 汇入total收益分数。
                original_targets,  # 汇入原始标签。
                targets,  # 汇入标签。
                fitted_targets,  # 汇入拟合标签。
                styles,  # 汇入风格。
                industry_codes,  # 汇入行业代码。
            )
            loss = dual_losses["loss"]  # 计算损失。
            predictive_loss = dual_losses["alpha_rank_loss"]  # 计算predictive损失。
            barra_r_squared = dual_losses["alpha_barra_r_squared"]  # 计算Barrarsquared。
            return_rank_loss = dual_losses["return_rank_loss"]  # 返回训练一个完整 epoch，并返回损失与批次诊断。
            alpha_rank_loss = dual_losses["alpha_rank_loss"]  # 计算Alpha排名损失。
            dual_style_rank_loss = dual_losses["style_rank_loss"]  # 计算dual风格排名损失。
        else:  # 处理上述条件均不成立的情况。
            loss, predictive_loss, barra_r_squared = combined_training_loss(  # 按实验口径计算可微损失。
                raw_predictions, predictions, targets, styles, industry_codes  # 传入原始预测、预测、标签、风格、行业代码。
            )
            zero = torch.zeros((), dtype=torch.float32, device=device)  # 初始化zero。
            return_rank_loss = zero  # 返回训练一个完整 epoch，并返回损失与批次诊断。
            alpha_rank_loss = zero  # 计算Alpha排名损失。
            dual_style_rank_loss = zero  # 计算dual风格排名损失。
        style_loss, industry_loss = adversarial_training_loss(  # 按实验口径计算可微损失。
            style_predictions,  # 汇入风格预测。
            industry_logits,  # 汇入行业logits。
            styles,  # 汇入风格。
            style_valid,  # 汇入风格有效。
            industry_codes,  # 汇入行业代码。
            industry_class_weights,  # 汇入行业class权重。
        )
        loss = (  # 计算损失。
            loss  # 汇入损失。
            + ADVERSARIAL_STYLE_LOSS_WEIGHT * style_loss
            + ADVERSARIAL_INDUSTRY_LOSS_WEIGHT * industry_loss
        )
        if not torch.isfinite(loss):  # 检查参与计算的数值是否全部有限。
            raise FloatingPointError("训练损失出现 NaN 或 Inf")  # 检测到 NaN、Inf 或退化数值时立即终止。
        loss.backward()  # 反向传播当前损失。
        optimizer.step()  # 更新参数或学习率状态。

        with torch.no_grad():  # 关闭梯度记录以降低验证或推理开销。
            if (  # 开始组合多项有效性或配置条件。
                adversarial_barra_enabled()
                or barra_conditioning_enabled()
                or dual_tower_enabled()
            ):
                barra_r_squared, _ = style_data.joint_barra_r_squared_torch(  # 按股票与交易日业务键对齐。
                    raw_predictions.detach(), styles, industry_codes
                )
            effective_pearson = pearson_correlation(predictions, targets)  # 计算当前截面相关性。
            original_pearson = pearson_correlation(  # 计算当前截面相关性。
                (
                    total_return_score.float()
                    if dual_tower_enabled() else raw_predictions.float()  # 根据当前数据状态选择处理分支。
                ),
                original_targets,  # 汇入原始标签。
            )
            if industry_logits is not None:  # 检查必需数据、配置或校验结果是否缺失。
                known = industry_codes > 0  # 构造已知行业，限定后续参与计算的样本。
                industry_accuracy = (
                    industry_logits[known].argmax(dim=1)
                    == industry_codes[known].long() - 1
                ).float().mean()
            else:  # 处理上述条件均不成立的情况。
                industry_accuracy = torch.zeros((), device=device)  # 初始化行业accuracy。
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
            totals["raw_style_max"],  # 汇入totals。
            float(raw_style["max_abs_style_pearson"]),
        )
        totals["raw_style_mean"] += float(
            raw_style["mean_abs_style_pearson"]  # 汇入原始风格。
        )
        totals["effective_style_max"] = max(
            totals["effective_style_max"],  # 汇入totals。
            float(effective_style["max_abs_style_pearson"]),
        )
        totals["effective_style_mean"] += float(
            effective_style["mean_abs_style_pearson"]  # 汇入有效风格。
        )
        totals["adversarial_style_loss"] += float(style_loss.detach())
        totals["adversarial_industry_loss"] += float(industry_loss.detach())
        totals["adversarial_industry_accuracy"] += float(industry_accuracy)
        totals["adversarial_unknown_rate"] += float(
            (industry_codes == 0).float().mean()
        )
        if gate is not None:  # 检查必需数据、配置或校验结果是否缺失。
            gate_abs = gate.detach().float().abs()  # 转换门控abs的数据类型或存储位置。
            totals["gate_mean_abs"] += float(gate_abs.mean())
            totals["gate_max_abs"] = max(
                totals["gate_max_abs"], float(gate_abs.max())
            )
        samples += len(targets)
        batches += 1
        if max_batches is not None and batch_number >= max_batches:  # 检查当前数值是否越过业务或数值阈值。
            break  # 满足终止条件后退出当前循环。

    if not batches:  # 检查必需数据、配置或校验结果是否缺失。
        raise ValueError("训练 loader 没有可用 batch")  # 输入数据不满足业务或数值约束时立即终止。
    if device.type == "cuda":  # 按当前模式、标识或状态选择处理分支。
        torch.cuda.synchronize(device)  # 同步 GPU 后计时。
    return {  # 返回训练一个完整 epoch，并返回损失与批次诊断。
        "train_loss": totals["loss"] / batches,  # 记录训练损失。
        "train_rank_loss": totals["rank_loss"] / batches,  # 记录训练秩损失。
        "train_return_rank_loss": totals["return_rank_loss"] / batches,  # 记录训练收益秩损失。
        "train_alpha_rank_loss": totals["alpha_rank_loss"] / batches,  # 记录训练Alpha秩损失。
        "train_dual_style_rank_loss": (  # 记录训练双塔风格秩损失。
            totals["dual_style_rank_loss"] / batches
        ),
        "train_barra_r_squared": totals["barra_r_squared"] / batches,  # 记录训练Barrarsquared。
        "train_effective_target_pearson_ic": (  # 记录训练有效标签Pearsonic。
            totals["effective_pearson"] / batches
        ),
        "train_original_target_pearson_ic": (  # 记录训练原始标签Pearsonic。
            totals["original_pearson"] / batches
        ),
        "train_raw_max_abs_style_pearson": totals["raw_style_max"],  # 记录训练原始maxabs风格Pearson。
        "train_raw_mean_abs_style_pearson": (  # 记录训练原始均值abs风格Pearson。
            totals["raw_style_mean"] / batches
        ),
        "train_effective_max_abs_style_pearson": (  # 记录训练有效maxabs风格Pearson。
            totals["effective_style_max"]  # 汇入totals。
        ),
        "train_effective_mean_abs_style_pearson": (  # 记录训练有效均值abs风格Pearson。
            totals["effective_style_mean"] / batches
        ),
        "train_adversarial_style_loss": (  # 记录训练adversarial风格损失。
            totals["adversarial_style_loss"] / batches
        ),
        "train_adversarial_industry_loss": (  # 记录训练adversarial行业损失。
            totals["adversarial_industry_loss"] / batches
        ),
        "train_adversarial_industry_accuracy": (  # 记录训练adversarial行业accuracy。
            totals["adversarial_industry_accuracy"] / batches
        ),
        "train_adversarial_unknown_rate": (  # 记录训练adversarial未知行业比例。
            totals["adversarial_unknown_rate"] / batches
        ),
        "train_adversarial_gamma": gamma,  # 记录训练adversarial对抗系数。
        "train_gate_mean_abs": totals["gate_mean_abs"] / batches,  # 记录训练门控均值abs。
        "train_gate_max_abs": totals["gate_max_abs"],  # 记录训练门控maxabs。
        "train_samples": samples,  # 记录训练样本。
        "train_date_batches": batches,  # 记录训练交易日batches。
        "train_seconds": time.perf_counter() - started,  # 记录训练seconds。
    }


_INSTRUMENT_IDS: np.ndarray | None = None  # 接收股票ids。


def instrument_ids() -> np.ndarray:  # 根据样本行号返回对应股票代码。
    """根据样本行号返回对应股票代码。"""
    global _INSTRUMENT_IDS  # 声明后续赋值使用外层作用域变量。
    if _INSTRUMENT_IDS is None:  # 检查所需对象是否尚未初始化。
        _INSTRUMENT_IDS = pd.read_parquet(  # 设置股票ids。
            sequence_data.SAMPLE_INDEX_PATH, columns=["instrument_id"]
        )["instrument_id"].to_numpy(copy=False)
    return _INSTRUMENT_IDS  # 返回股票ids。


def has_degenerate_daily_pearson(  # 判断是否存在无法计算 Pearson 的交易日截面。
    predictions: np.ndarray,  # 接收预测。
    targets: np.ndarray,  # 接收标签。
    date_ids: np.ndarray,  # 接收交易日ids。
) -> bool:  # 声明当前函数的返回类型。
    """判断是否存在无法计算 Pearson 的交易日截面。"""
    x = np.asarray(predictions, dtype=np.float64).reshape(-1)  # [N,...] -> [N]。
    y = np.asarray(targets, dtype=np.float64).reshape(-1)  # [N,...] -> [N]。
    dates = np.asarray(date_ids, dtype=np.int64).reshape(-1)  # [N,...] -> [N]。
  # 检查三组输入等长且至少包含两个样本。
    if not (len(x) == len(y) == len(dates)) or len(x) < 2:  # 检查当前样本数量是否满足计算要求。
  # 抛出长度错误，避免把样本错位误判为分数塌缩。
        raise ValueError("Style Pearson 预测、目标和日期必须等长且至少有两个样本")  # 输入数据不满足业务或数值约束时立即终止。
  # 检查预测和目标均为有限值。
    if not np.isfinite(x).all() or not np.isfinite(y).all():  # 检查参与计算的数值是否全部有限。
  # 保持验证数据的 NaN/Inf 严格失败契约。
        raise FloatingPointError("Style Pearson 预测或目标包含 NaN/Inf")  # 检测到 NaN、Inf 或退化数值时立即终止。
  # 遍历每个交易日，复用正式验证函数的同一 Pearson 分母阈值。
    for date in np.unique(dates):  # 遍历NumPyunique交易日，逐项处理交易日。
  # 计算当前交易日的样本掩码。
        mask = dates == date  # 配置掩码。
  # 检查当前交易日至少包含两个样本。
        if int(mask.sum()) < 2:  # 检查当前数值是否越过业务或数值阈值。
  # 抛出截面样本不足错误，不将数据契约问题降级为诊断值。
            raise ValueError("Style Pearson 存在不足两个样本的交易日")  # 输入数据不满足业务或数值约束时立即终止。
  # 尝试按正式验证口径计算 Pearson。
        try:  # 执行可能失败的外部读取或计算步骤。
            numpy_pearson(x[mask], y[mask])
  # 仅捕获分母退化错误，将其标记为 Style 塔塌缩。
        except FloatingPointError:  # 捕获预期异常并执行对应恢复逻辑。
  # 返回 True，让调用方记录 NaN 审计指标而不中断主训练。
            return True  # 返回true。
  # 所有交易日均可计算 Pearson，返回未塌缩。
    return False  # 返回false。


# 使用 `torch.no_grad()` 包装 `evaluate_model`，并定义其可复用的处理逻辑。
@torch.no_grad()  # 验证或推理阶段关闭梯度记录。
def evaluate_model(  # 在验证集上评估模型，并汇总逐日 IC 与损失诊断。
    model: torch.nn.Module,  # 接收模型。
    loader: DataLoader,  # 接收数据加载器。
    device: torch.device,  # 接收设备。
    *,
    industry_class_weights: torch.Tensor,  # [32]，有效行业类别权重。
    max_batches: int | None = None,  # 接收maxbatches。
    distributed_context: distributed.DistributedContext | None = None,  # 接收distributed上下文。
) -> dict[str, float | int]:  # 声明当前函数的返回类型。
    """在验证集上评估模型，并汇总逐日 IC 与损失诊断。"""
    local_model = base.inference_model(model)  # 关闭 dropout 等训练行为。
    local_model.eval()  # 切换模型到验证或推理模式。
    effective_predictions_parts: list[np.ndarray] = []  # 接收有效预测parts。
    raw_predictions_parts: list[np.ndarray] = []  # 接收原始预测parts。
    effective_targets: list[np.ndarray] = []  # 接收有效标签。
    original_targets: list[np.ndarray] = []  # 接收原始标签。
    dual_style_predictions: list[np.ndarray] = []  # 接收dual风格预测。
    dual_total_predictions: list[np.ndarray] = []  # 接收dualtotal预测。
    dual_fitted_targets: list[np.ndarray] = []  # 接收dual拟合标签。
    date_ids: list[np.ndarray] = []  # 接收交易日ids。
    sample_indices: list[np.ndarray] = []  # 接收样本索引。
    adversarial_style_predictions: list[np.ndarray] = []  # 接收adversarial风格预测。
    adversarial_style_targets: list[np.ndarray] = []  # 接收adversarial风格标签。
    adversarial_style_valid: list[np.ndarray] = []  # 输入adversarial风格有效性掩码。
    adversarial_industry_logits: list[np.ndarray] = []  # 接收adversarial行业logits。
    adversarial_industry_codes: list[np.ndarray] = []  # 接收adversarial行业代码。
    raw_style_max = 0.0
    raw_style_mean = 0.0  # 计算原始风格均值。
    effective_style_max = 0.0
    effective_style_mean = 0.0  # 计算有效风格均值。
    barra_r_squared_sum = 0.0  # 计算Barrarsquaredsum。
    barra_r_squared_max = 0.0  # 计算Barrarsquaredmax。
    effective_barra_r_squared_sum = 0.0  # 计算有效Barrarsquaredsum。
    effective_barra_r_squared_max = 0.0  # 计算有效Barrarsquaredmax。
    effective_industry_mean_max = 0.0  # 计算有效行业均值max。
    gate_mean_abs = 0.0  # 计算门控均值abs。
    gate_max_abs = 0.0
    minimum_daily_raw_score_std = np.inf  # 计算minimumdaily原始分数标准差。
    batches = 0
    started = time.perf_counter()

    for batch_number, item in enumerate(  # 遍历enumerate，逐项处理批次numberitem。
        StyleCudaPrefetcher(loader, device, bool(USE_CUDA_PREFETCH)), start=1
    ):
        (
            batch, videos, frame_mask, targets, original, _residual,  # 传入批次、视频、帧掩码、标签、原始、残差。
            fitted_targets, styles, style_valid, industry_codes,  # 传入拟合标签、风格、风格有效、行业代码。
        ) = item
        if int(batch["date_ids"].min()) != int(batch["date_ids"].max()):  # 按当前模式、标识或状态选择处理分支。
            raise AssertionError("验证 batch 混入多个交易日")  # 关键数据契约被破坏时立即终止。
        (
            conditioning_styles,  # 汇入conditioning风格。
            conditioning_valid,  # 汇入conditioning有效。
            conditioning_industry,  # 汇入conditioning行业。
            conditioning_prepared,  # 汇入conditioningprepared。
        ) = prepare_distributed_conditioning(
            styles, style_valid, industry_codes, batch["sample_indices"], distributed_context
        )
        with base.autocast_context(device):  # 在自动混合精度上下文中执行前向计算。
            (  # 执行当前 batch 前向传播。
                raw_prediction,  # 汇入原始预测。
                style_prediction,  # 汇入风格预测。
                industry_logits,  # 汇入行业logits。
                _representation,  # 汇入表征。
                gate,  # 汇入门控。
                dual_style_score,  # 汇入dual风格分数。
                total_return_score,  # 汇入total收益分数。
            ) = forward_training_model(
                local_model,  # 汇入本地模型。
                videos,  # 汇入视频。
                frame_mask,  # 汇入帧掩码。
                gamma=0.0,  # 配置对抗系数。
                styles=conditioning_styles,  # 配置风格。
                style_valid=conditioning_valid,  # 配置风格有效。
                industry_codes=conditioning_industry,  # 配置行业代码。
                sample_indices=batch["sample_indices"],  # 配置样本索引。
                conditioning_prepared=conditioning_prepared,  # 配置conditioningprepared。
            )
        (
            raw_prediction,  # 汇入原始预测。
            style_prediction,  # 汇入风格预测。
            industry_logits,  # 汇入行业logits。
            gate,  # 汇入门控。
            dual_style_score,  # 汇入dual风格分数。
            total_return_score,  # 汇入total收益分数。
            targets,  # 汇入标签。
            original,  # 汇入原始。
            _residual,  # 汇入残差。
            fitted_targets,  # 汇入拟合标签。
            styles,  # 汇入风格。
            style_valid,  # 汇入风格有效。
            industry_codes,  # 汇入行业代码。
            global_sample_indices,  # 汇入全局样本索引。
            global_date_ids,  # 汇入全局交易日ids。
        ) = gather_style_cross_section(
            raw_prediction,  # 汇入原始预测。
            style_prediction,  # 汇入风格预测。
            industry_logits,  # 汇入行业logits。
            gate,  # 汇入门控。
            dual_style_score,  # 汇入dual风格分数。
            total_return_score,  # 汇入total收益分数。
            targets,  # 汇入标签。
            original,  # 汇入原始。
            _residual,  # 汇入残差。
            fitted_targets,  # 汇入拟合标签。
            styles,  # 汇入风格。
            style_valid,  # 汇入风格有效。
            industry_codes,  # 汇入行业代码。
            batch,  # 汇入批次。
            distributed_context,  # 汇入distributed上下文。
            differentiable_outputs=False,  # 配置differentiableoutputs。
        )
        prediction, projection_diagnostics = effective_predictions(
            raw_prediction, styles, industry_codes  # 传入原始预测、风格、行业代码。
        )
        if strict_barra_output_enabled():  # 根据当前数据状态选择处理分支。
            if projection_diagnostics is None:  # 检查所需对象是否尚未初始化。
                raise AssertionError("严格输出缺少投影诊断")  # 关键数据契约被破坏时立即终止。
            effective_industry_mean_max = max(  # 计算有效行业均值max。
                effective_industry_mean_max,  # 汇入有效行业均值max。
                float(projection_diagnostics["max_abs_industry_mean"]),
            )
        date_id = int(batch["date_ids"][0])
        raw_score_dispersion = measure_daily_score_dispersion(  # 计算原始分数dispersion。
            raw_prediction,  # 汇入原始预测。
            date_id=date_id,  # 配置交易日id。
        )
        minimum_daily_raw_score_std = min(  # 计算minimumdaily原始分数标准差。
            minimum_daily_raw_score_std,  # 汇入minimumdaily原始分数标准差。
            float(raw_score_dispersion["score_std"].item()),
        )
        raw_style = measure_style_correlations(raw_prediction, styles)  # 计算当前截面相关性。
        effective_style = measure_style_correlations(prediction, styles)  # 计算当前截面相关性。
        if industry_sidecar_required():  # 根据当前数据状态选择处理分支。
            barra_r_squared, _ = style_data.joint_barra_r_squared_torch(  # 按股票与交易日业务键对齐。
                raw_prediction, styles, industry_codes  # 传入原始预测、风格、行业代码。
            )
        else:  # 处理上述条件均不成立的情况。
            barra_r_squared = torch.zeros(  # 初始化Barrarsquared。
                (), dtype=torch.float32, device=raw_prediction.device
            )
        if strict_barra_output_enabled():  # 根据当前数据状态选择处理分支。
            effective_barra_r_squared, _ = (  # 按股票与交易日业务键对齐。
                style_data.joint_barra_r_squared_torch(
                    prediction, styles, industry_codes  # 传入预测、风格、行业代码。
                )
            )
            if float(effective_barra_r_squared) > 1e-6:  # 检查当前数值是否越过业务或数值阈值。
                raise FloatingPointError(  # 检测到 NaN、Inf 或退化数值时立即终止。
                    "严格 Barra 输出的联合 R² 未接近零"
                )
        else:  # 处理上述条件均不成立的情况。
            effective_barra_r_squared = torch.zeros_like(barra_r_squared)  # 初始化有效Barrarsquared。

        effective_predictions_parts.append(  # 把当前结果追加到汇总列表。
            prediction.cpu().numpy().astype(np.float64)
        )
        raw_predictions_parts.append(  # 把当前结果追加到汇总列表。
            raw_prediction.float().cpu().numpy().astype(np.float64)
        )
        effective_targets.append(targets.cpu().numpy().astype(np.float64))  # 把当前结果追加到汇总列表。
        original_targets.append(original.cpu().numpy().astype(np.float64))  # 把当前结果追加到汇总列表。
        if dual_tower_enabled():  # 根据当前数据状态选择处理分支。
            if dual_style_score is None or total_return_score is None:  # 检查所需对象是否尚未初始化。
                raise ValueError("No.6 验证缺少 Style 或 Total 分数")  # 输入数据不满足业务或数值约束时立即终止。
            dual_style_predictions.append(  # 把当前结果追加到汇总列表。
                dual_style_score.float().cpu().numpy().astype(np.float64)
            )
            dual_total_predictions.append(  # 把当前结果追加到汇总列表。
                total_return_score.float().cpu().numpy().astype(np.float64)
            )
            dual_fitted_targets.append(  # 把当前结果追加到汇总列表。
                fitted_targets.cpu().numpy().astype(np.float64)
            )
        date_ids.append(global_date_ids.cpu().numpy().astype(np.int64, copy=False))  # 把当前结果追加到汇总列表。
        sample_indices.append(global_sample_indices.cpu().numpy().astype(np.int64, copy=False))  # 把当前结果追加到汇总列表。
        if adversarial_barra_enabled():  # 根据当前数据状态选择处理分支。
            if style_prediction is None or industry_logits is None:  # 检查所需对象是否尚未初始化。
                raise ValueError("No.3 验证缺少对抗头输出")  # 输入数据不满足业务或数值约束时立即终止。
            adversarial_style_predictions.append(  # 把当前结果追加到汇总列表。
                style_prediction.float().cpu().numpy()
            )
            adversarial_style_targets.append(styles.float().cpu().numpy())  # 把当前结果追加到汇总列表。
            adversarial_style_valid.append(style_valid.cpu().numpy())  # 把当前结果追加到汇总列表。
            adversarial_industry_logits.append(  # 把当前结果追加到汇总列表。
                industry_logits.float().cpu().numpy()
            )
            adversarial_industry_codes.append(industry_codes.cpu().numpy())  # 把当前结果追加到汇总列表。
        raw_style_max = max(
            raw_style_max, float(raw_style["max_abs_style_pearson"])
        )
        raw_style_mean += float(raw_style["mean_abs_style_pearson"])
        effective_style_max = max(
            effective_style_max,  # 汇入有效风格max。
            float(effective_style["max_abs_style_pearson"]),
        )
        effective_style_mean += float(
            effective_style["mean_abs_style_pearson"]  # 汇入有效风格。
        )
        barra_value = float(barra_r_squared)
        barra_r_squared_sum += barra_value
        barra_r_squared_max = max(barra_r_squared_max, barra_value)  # 计算Barrarsquaredmax。
        effective_barra_value = float(effective_barra_r_squared)
        effective_barra_r_squared_sum += effective_barra_value
        effective_barra_r_squared_max = max(  # 计算有效Barrarsquaredmax。
            effective_barra_r_squared_max, effective_barra_value  # 传入有效Barrarsquaredmax、有效Barra值。
        )
        if gate is not None:  # 检查必需数据、配置或校验结果是否缺失。
            gate_abs = gate.float().abs()  # 转换门控abs的数据类型或存储位置。
            gate_mean_abs += float(gate_abs.mean())
            gate_max_abs = max(gate_max_abs, float(gate_abs.max()))
        batches += 1
        if max_batches is not None and batch_number >= max_batches:  # 检查当前数值是否越过业务或数值阈值。
            break  # 满足终止条件后退出当前循环。

    if not batches:  # 检查必需数据、配置或校验结果是否缺失。
        raise ValueError("验证 loader 没有可用 batch")  # 输入数据不满足业务或数值约束时立即终止。
    if device.type == "cuda":  # 按当前模式、标识或状态选择处理分支。
        torch.cuda.synchronize(device)  # 同步 GPU 后计时。
    effective_x = np.concatenate(effective_predictions_parts)  # 对齐并合并有效x。
    raw_x = np.concatenate(raw_predictions_parts)  # 对齐并合并原始x。
    effective_y = np.concatenate(effective_targets)  # 对齐并合并有效y。
    original_y = np.concatenate(original_targets)  # 对齐并合并原始y。
    dates = np.concatenate(date_ids)  # 对齐并合并交易日。
    selected = np.concatenate(sample_indices)  # 对齐并合并筛选结果。
    effective_metrics = summarize_validation_arrays(  # 按约定截面口径汇总。
        effective_x, effective_y, dates, VALIDATION_METRIC,  # 传入有效x、有效y、交易日、验证指标。
        instrument_ids=instrument_ids()[selected],  # 配置股票ids。
    )
    dual_metrics: dict[str, float | bool] = {}  # 计算dual指标。
    original_prediction = raw_x
    if dual_tower_enabled():  # 根据当前数据状态选择处理分支。
        style_x = np.concatenate(dual_style_predictions)  # 对齐并合并风格x。
        total_x = np.concatenate(dual_total_predictions)  # 对齐并合并totalx。
        fitted_y = np.concatenate(dual_fitted_targets)  # 对齐并合并拟合y。
        original_prediction = total_x
        total_metrics = summarize_validation_arrays(  # 按约定截面口径汇总。
            total_x,  # 汇入totalx。
            original_y,  # 汇入原始y。
            dates,  # 汇入交易日。
            VALIDATION_METRIC,  # 汇入验证指标。
            instrument_ids=instrument_ids()[selected],  # 配置股票ids。
        )
  # 调用统一 Pearson 判据识别近常数 Style 预测或目标。
        style_collapsed = has_degenerate_daily_pearson(
            style_x, fitted_y, dates  # 传入风格x、拟合y、交易日。
        )
        if style_collapsed:  # 根据当前数据状态选择处理分支。
            style_rank_ic = float("nan")  # 计算风格排名ic。
            style_pearson_ic = float("nan")
        else:  # 处理上述条件均不成立的情况。
            style_metrics = summarize_validation_arrays(  # 按约定截面口径汇总。
                style_x,  # 汇入风格x。
                fitted_y,  # 汇入拟合y。
                dates,  # 汇入交易日。
                VALIDATION_METRIC,  # 汇入验证指标。
                instrument_ids=instrument_ids()[selected],  # 配置股票ids。
            )
            style_rank_ic = float(  # 计算风格排名ic。
                style_metrics["validation_daily_rank_ic"]  # 汇入风格指标。
            )
            style_pearson_ic = float(
                style_metrics["validation_daily_pearson_ic"]  # 汇入风格指标。
            )
        dual_metrics = {  # 计算dual指标。
            "validation_total_original_daily_rank_ic": float(  # 记录验证total原始日度秩ic。
                total_metrics["validation_daily_rank_ic"]  # 汇入total指标。
            ),
            "validation_total_original_daily_pearson_ic": float(  # 记录验证total原始日度Pearsonic。
                total_metrics["validation_daily_pearson_ic"]  # 汇入total指标。
            ),
            "validation_style_fitted_daily_rank_ic": style_rank_ic,  # 记录验证风格拟合日度秩ic。
            "validation_style_fitted_daily_pearson_ic": style_pearson_ic,  # 记录验证风格拟合日度Pearsonic。
            "validation_style_score_collapsed": style_collapsed,  # 记录验证风格分数collapsed。
            "validation_dual_additivity_max_abs_error": float(  # 记录验证双塔additivitymaxabs误差。
                np.max(np.abs(total_x - raw_x - style_x))
            ),
        }
    original_metrics = summarize_validation_arrays(  # 按约定截面口径汇总。
        original_prediction, original_y, dates, VALIDATION_METRIC,  # 传入原始预测、原始y、交易日、验证指标。
        instrument_ids=instrument_ids()[selected],  # 配置股票ids。
    )
    adversarial_metrics: dict[str, float] = {}  # 计算adversarial指标。
    if adversarial_barra_enabled():  # 根据当前数据状态选择处理分支。
        adversarial_metrics = {  # 计算adversarial指标。
            f"validation_adversarial_{key}": value
            for key, value in summarize_adversarial_predictions(  # 遍历汇总adversarial预测，逐项处理键值。
                np.concatenate(adversarial_style_predictions),
                np.concatenate(adversarial_style_targets),
                np.concatenate(adversarial_style_valid),
                np.concatenate(adversarial_industry_logits),
                np.concatenate(adversarial_industry_codes),
                industry_class_weights.detach().cpu().numpy(),
            ).items()
        }
    return {  # 返回在验证集上评估模型，并汇总逐日 IC 与损失诊断。
        "validation_ic": float(effective_metrics["validation_ic"]),  # 记录验证ic。
        "validation_daily_rank_ic": float(  # 记录验证日度秩ic。
            effective_metrics["validation_daily_rank_ic"]  # 汇入有效指标。
        ),
        "validation_daily_pearson_ic": float(  # 记录验证日度Pearsonic。
            effective_metrics["validation_daily_pearson_ic"]
        ),
        "validation_pooled_ic": float(  # 记录验证pooledic。
            effective_metrics["validation_pooled_ic"]
        ),
        "validation_original_daily_rank_ic": float(  # 记录验证原始日度秩ic。
            original_metrics["validation_daily_rank_ic"]
        ),
        "validation_original_daily_pearson_ic": float(  # 记录验证原始日度Pearsonic。
            original_metrics["validation_daily_pearson_ic"]
        ),
        "validation_original_pooled_ic": float(  # 记录验证原始pooledic。
            original_metrics["validation_pooled_ic"]
        ),
        "validation_raw_max_abs_style_pearson": raw_style_max,  # 记录验证原始maxabs风格Pearson。
        "validation_raw_mean_abs_style_pearson": raw_style_mean / batches,  # 记录验证原始均值abs风格Pearson。
        "validation_barra_r_squared_mean": barra_r_squared_sum / batches,  # 记录验证Barrarsquared均值。
        "validation_barra_r_squared_max": barra_r_squared_max,  # 记录验证Barrarsquaredmax。
        "validation_effective_barra_r_squared_mean": (  # 记录验证有效Barrarsquared均值。
            effective_barra_r_squared_sum / batches
        ),
        "validation_effective_barra_r_squared_max": effective_barra_r_squared_max,  # 记录验证有效Barrarsquaredmax。
        "validation_effective_max_abs_industry_mean": (  # 记录验证有效maxabs行业均值。
            effective_industry_mean_max
        ),
        "validation_gate_mean_abs": gate_mean_abs / batches,  # 记录验证门控均值abs。
        "validation_gate_max_abs": gate_max_abs,  # 记录验证门控maxabs。
        "validation_effective_max_abs_style_pearson": effective_style_max,  # 记录验证有效maxabs风格Pearson。
        "validation_effective_mean_abs_style_pearson": (  # 记录验证有效均值abs风格Pearson。
            effective_style_mean / batches
        ),
        "validation_dates": int(effective_metrics["validation_dates"]),  # 记录验证交易日。
        "validation_samples": int(effective_metrics["validation_samples"]),  # 记录验证样本。
        "validation_seconds": time.perf_counter() - started,  # 记录验证seconds。
        "validation_score_collapsed": False,  # 记录验证分数collapsed。
        "validation_collapsed_date_count": 0,  # 记录验证collapsed交易日数量。
        "validation_min_daily_raw_score_std": float(  # 记录验证min日度原始分数标准差。
            minimum_daily_raw_score_std
        ),
        **adversarial_metrics,
        **dual_metrics,
    }


# 使用 `torch.no_grad()` 包装 `predict_model`，并定义其可复用的处理逻辑。
@torch.no_grad()  # 验证或推理阶段关闭梯度记录。
def predict_model(  # 在预测集上生成与样本行号严格对齐的模型分数。
    model: torch.nn.Module,  # 接收模型。
    loader: DataLoader,  # 接收数据加载器。
    device: torch.device,  # 接收设备。
    distributed_context: distributed.DistributedContext | None = None,  # 接收distributed上下文。
) -> pd.DataFrame:  # 声明当前函数的返回类型。
    """在预测集上生成与样本行号严格对齐的模型分数。"""
    if (  # 开始组合多项有效性或配置条件。
        not barra_conditioning_enabled()
        and not dual_tower_enabled()
        and (raw_barra_r2_enabled() or adversarial_barra_enabled())
    ):
        raw = base.predict_model(
            model, loader, device, bool(USE_CUDA_PREFETCH),
            distributed_context=distributed_context,  # 配置distributed上下文。
        )
        values = raw["prediction"].to_numpy(np.float32)  # 转换并统一值的数据表示。
        return pd.DataFrame(  # 返回在预测集上生成与样本行号严格对齐的模型分数。
            {
                "sample_index": raw["sample_index"].to_numpy(np.int64),  # 记录样本索引。
                "prediction_model_raw": values,  # 记录预测模型原始。
                "prediction": values,  # 记录预测。
            }
        )
    local_model = base.inference_model(model)  # 关闭 dropout 等训练行为。
    local_model.eval()  # 切换模型到验证或推理模式。
    sample_parts: list[np.ndarray] = []  # 接收样本parts。
    raw_parts: list[np.ndarray] = []  # 接收原始parts。
    effective_parts: list[np.ndarray] = []  # 接收有效parts。
    style_parts: list[np.ndarray] = []  # 接收风格parts。
    total_parts: list[np.ndarray] = []  # 接收totalparts。
    for item in StyleCudaPrefetcher(  # 遍历stylecudaprefetcher，逐项处理item。
        loader, device, bool(USE_CUDA_PREFETCH)
    ):
        (
            batch, videos, frame_mask, _targets, _original, _residual,  # 传入批次、视频、帧掩码、标签、原始、残差。
            _fitted, styles, style_valid, industry_codes,  # 传入拟合、风格、风格有效、行业代码。
        ) = item
        if int(batch["date_ids"].min()) != int(batch["date_ids"].max()):  # 按当前模式、标识或状态选择处理分支。
            raise AssertionError("推理 batch 混入多个交易日")  # 关键数据契约被破坏时立即终止。
        (
            conditioning_styles,
            conditioning_valid,
            conditioning_industry,
            conditioning_prepared,
        ) = prepare_distributed_conditioning(
            styles, style_valid, industry_codes, batch["sample_indices"], distributed_context
        )
        with base.autocast_context(device):  # 在自动混合精度上下文中执行前向计算。
            (  # 执行当前 batch 前向传播。
                raw_prediction, _, _, _, _, dual_style_score,  # 传入原始预测、、、、、dual风格分数。
                total_return_score,
            ) = forward_training_model(
                local_model,
                videos,
                frame_mask,
                gamma=0.0,  # 配置对抗系数。
                styles=conditioning_styles,  # 配置风格。
                style_valid=conditioning_valid,  # 配置风格有效。
                industry_codes=conditioning_industry,  # 配置行业代码。
                sample_indices=batch["sample_indices"],  # 配置样本索引。
                conditioning_prepared=conditioning_prepared,  # 配置conditioningprepared。
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
            differentiable_outputs=False,  # 配置differentiableoutputs。
        )
        prediction, _ = effective_predictions(
            raw_prediction, styles, industry_codes  # 传入原始预测、风格、行业代码。
        )
        sample_parts.append(global_sample_indices.cpu().numpy())  # 把当前结果追加到汇总列表。
        raw_parts.append(raw_prediction.float().cpu().numpy())  # 把当前结果追加到汇总列表。
        effective_parts.append(prediction.cpu().numpy())  # 把当前结果追加到汇总列表。
        if dual_tower_enabled():  # 根据当前数据状态选择处理分支。
            if dual_style_score is None or total_return_score is None:  # 检查所需对象是否尚未初始化。
                raise ValueError("No.6 推理缺少 Style 或 Total 分数")  # 输入数据不满足业务或数值约束时立即终止。
            style_parts.append(dual_style_score.float().cpu().numpy())  # 把当前结果追加到汇总列表。
            total_parts.append(total_return_score.float().cpu().numpy())  # 把当前结果追加到汇总列表。
    if not sample_parts:  # 检查必需数据、配置或校验结果是否缺失。
        return pd.DataFrame(  # 返回在预测集上生成与样本行号严格对齐的模型分数。
            {
                "sample_index": pd.Series(dtype=np.int64),  # 记录样本索引。
                "prediction_model_raw": pd.Series(dtype=np.float32),  # 记录预测模型原始。
                "prediction": pd.Series(dtype=np.float32),  # 记录预测。
                "prediction_style_component": pd.Series(dtype=np.float32),  # 记录预测风格component。
                "prediction_total_return": pd.Series(dtype=np.float32),  # 记录预测total收益。
            }
        )
    result = pd.DataFrame(
        {
            "sample_index": np.concatenate(sample_parts).astype(np.int64),  # 记录样本索引。
            "prediction_model_raw": np.concatenate(raw_parts).astype(np.float32),  # 记录预测模型原始。
            "prediction": np.concatenate(effective_parts).astype(np.float32),  # 记录预测。
        }
    )
    if dual_tower_enabled():  # 根据当前数据状态选择处理分支。
        result["prediction_style_component"] = np.concatenate(  # 对齐并合并结果。
            style_parts
        ).astype(np.float32)
        result["prediction_total_return"] = np.concatenate(  # 对齐并合并结果。
            total_parts
        ).astype(np.float32)
    return result.sort_values("sample_index", kind="stable").reset_index(  # 返回在预测集上生成与样本行号严格对齐的模型分数。
        drop=True
    )


PROBE_MAX_SAMPLES_PER_DATE = 512  # 设置probemax样本per交易日。
PROBE_BATCH_SIZE = 4096  # 设置probe批次大小。
PROBE_LEARNING_RATE = 1e-3  # 设置probelearning比率。
PROBE_WEIGHT_DECAY = 1e-4  # 设置probe权重decay。
PROBE_MAX_EPOCHS = 50  # 设置probemaxepochs。
PROBE_PATIENCE = 5  # 设置probepatience。
PROBE_RIDGE_ALPHA = 1e-3  # 设置proberidgeAlpha。


# %% [markdown]
# ## 年度 × seed 训练和断点续训


# %%
def run_payload(  # 整理影响整个 run 身份和可复用性的签名载荷。
    year: int, seed: int, batch_summary: dict[str, Any]
) -> dict[str, Any]:  # 声明当前函数的返回类型。
    """整理影响整个 run 身份和可复用性的签名载荷。"""
    sequence = sequence_data.require_aligned_sequence_store()
    style_manifest = style_data.require_style_neutral_store()
    industry_manifest = (
        style_data.require_industry_sidecar()
        if industry_sidecar_required() else None  # 根据当前数据状态选择处理分支。
    )
    full_target_manifest = (
        style_data.require_full_barra_target_sidecar()
        if barra_conditioning_enabled() or dual_tower_enabled() else None  # 根据当前数据状态选择处理分支。
    )
    dual_target_manifest = (
        style_data.require_dual_tower_target_sidecar()
        if dual_tower_enabled() else None  # 根据当前数据状态选择处理分支。
    )
    config = training_config()
    payload = {  # 初始化载荷容器。
        "schema_version": (  # 记录schema版本。
            7 if dual_tower_enabled()
            else 6 if strict_barra_output_enabled()
            else 5 if barra_conditioning_enabled() else TRAINING_SCHEMA_VERSION
        ),
        "training_code_sha256": hashlib.sha256(  # 记录训练代码sha256。
            SCRIPT_PATH.read_bytes()
        ).hexdigest(),
        "run_name": RUN_NAME,  # 记录运行名称。
        "forecast_year": year,  # 记录forecast年份。
        "seed": seed,  # 记录随机种子。
        "sequence_signature": sequence["signature"],  # 记录sequence签名，用于来源一致性校验。
        "style_store_signature": style_manifest["signature"],  # 记录风格存储签名，用于来源一致性校验。
        "industry_sidecar_signature": (  # 记录行业sidecar签名，用于来源一致性校验。
            industry_manifest["signature"]
            if industry_manifest is not None else None  # 检查必需数据、配置或校验结果是否缺失。
        ),
        "source_dataset_signature": sequence["payload"][  # 记录来源dataset签名，用于来源一致性校验。
            "source_dataset_signature"
        ],
        "source_rolling_signature": sequence["payload"][  # 记录来源rolling签名，用于来源一致性校验。
            "source_rolling_signature"
        ],
        "split_policy": "exact_vit_train_validation_prediction_indices",  # 把 split_policy 映射为"exact_vit_train_validation_prediction_indices"。
        "model_config": asdict(base.MODEL_CONFIG),  # 记录模型配置。
        "training_config": asdict(config),  # 记录训练配置。
        "training_batch_summary": batch_summary,  # 记录训练批次汇总。
        "residualize_target": RESIDUALIZE_TARGET,  # 记录残差化标签。
        "project_model_output": PROJECT_MODEL_OUTPUT,  # 记录投影模型输出。
        "target_policy": target_policy(),  # 声明标签处理口径。
        "output_policy": output_policy(),  # 声明输出处理口径。
        "projection": {  # 记录projection。
            "basis": "intercept_plus_9_cne5s_styles_plus_reversal20",  # 把 basis 映射为"intercept_plus_9_cne5s_styles_plus_reversal20"。
            "solver": "torch_fp32_reduced_qr",  # 把 solver 映射为"torch_fp32_reduced_qr"。
            "standardization": "daily_zscore_ddof0",  # 把 standardization 映射为"daily_zscore_ddof0"。
            "industry_used": False,  # 记录行业使用状态。
        },
        "style_columns": list(style_data.STYLE_COLUMNS),  # 记录风格字段列表。
        "industry_used_by_regularizer": raw_barra_r2_enabled(),  # 记录行业使用状态byregularizer。
        "industry_used_by_adversary": adversarial_barra_enabled(),  # 记录行业使用状态byadversary。
        "inference_requires_barra": (  # 记录inferencerequiresBarra。
            dual_tower_enabled()
            or strict_barra_output_enabled()
            or BARRA_CONDITIONING_MODE in {"real", "shuffled"}
        ),
        "early_stop_selection": "validation_ic_only",  # 把 early_stop_selection 映射为"validation_ic_only"。
        "style_correlation_penalty": raw_barra_r2_enabled(),  # 记录风格相关系数penalty。
        "raw_barra_r2_lambda": RAW_BARRA_R2_LAMBDA,  # 记录原始Barrar2lambda。
        "adversarial_barra": adversarial_barra_enabled(),  # 记录adversarialBarra。
        "adversarial_config": (  # 记录adversarial配置。
            {
                "representation": "final_norm_cls_128",  # 把 representation 映射为"final_norm_cls_128"。
                "style_targets": list(style_data.STYLE_COLUMNS),  # 记录风格标签。
                "industry_classes": INDUSTRY_COUNT,  # 记录行业classes。
                "unknown_industry_policy": "code_zero_masked_from_ce",  # 把 unknown_industry_policy 映射为"code_zero_masked_from_ce"。
                "head": {  # 记录head。
                    "hidden_dim": ADVERSARIAL_HEAD_HIDDEN_DIM,  # 记录隐藏层dim。
                    "activation": "gelu",  # 把 activation 映射为"gelu"。
                    "dropout": ADVERSARIAL_HEAD_DROPOUT,  # 记录dropout。
                },
                "gamma_max": ADVERSARIAL_GAMMA_MAX,  # 记录对抗系数max。
                "warmup_epochs": ADVERSARIAL_WARMUP_EPOCHS,  # 记录warmupepochs。
                "cosine_ramp_epochs": ADVERSARIAL_RAMP_EPOCHS,  # 记录cosinerampepochs。
                "style_loss": "masked_mse",  # 把 style_loss 映射为"masked_mse"。
                "style_loss_weight": ADVERSARIAL_STYLE_LOSS_WEIGHT,  # 记录风格损失权重。
                "industry_loss": "inverse_sqrt_weighted_ce_div_log32",  # 把 industry_loss 映射为"inverse_sqrt_weighted_ce_div_log32"。
                "industry_loss_weight": ADVERSARIAL_INDUSTRY_LOSS_WEIGHT,  # 记录行业损失权重。
                "probe": {  # 记录probe。
                    "sample_cap_per_date": PROBE_MAX_SAMPLES_PER_DATE,  # 记录样本capper交易日。
                    "date_split": [0.4, 0.1, 0.5],  # 记录交易日切分。
                    "batch_size": PROBE_BATCH_SIZE,  # 记录批次大小。
                    "learning_rate": PROBE_LEARNING_RATE,  # 记录learning比例。
                    "weight_decay": PROBE_WEIGHT_DECAY,  # 记录权重decay。
                    "max_epochs": PROBE_MAX_EPOCHS,  # 记录maxepochs。
                    "patience": PROBE_PATIENCE,  # 记录patience。
                    "ridge_alpha": PROBE_RIDGE_ALPHA,  # 记录ridgeAlpha。
                    "mlp_hidden_dim": 128,  # 记录mlp隐藏层dim。
                    "mlp_dropout": 0.1,  # 记录mlpdropout。
                },
            }
            if adversarial_barra_enabled() else None  # 根据当前数据状态选择处理分支。
        ),
        "raw_barra_r2_design": {  # 记录原始Barrar2design。
            "dependent_variable": "raw_model_score",  # 把 dependent_variable 映射为"raw_model_score"。
            "continuous": list(style_data.STYLE_COLUMNS),  # 记录continuous。
            "industry_fixed_effects": True,  # 记录行业fixedeffects。
            "solver": "industry_within_then_torch_fp32_reduced_qr",  # 把 solver 映射为"industry_within_then_torch_fp32_reduced_qr"。
            "missing": "style_zero_industry_unknown_no_forward_fill",  # 把 missing 映射为"style_zero_industry_unknown_no_forward_fill"。
        },
        "loss_formula": (  # 记录损失formula。
            f"{LOSS_NAME}(alpha+style,source_target) + "
            f"{DUAL_TOWER_ALPHA_LOSS_WEIGHT}*{LOSS_NAME}(alpha,full_barra_residual) + "
            f"{DUAL_TOWER_STYLE_LOSS_WEIGHT}*{LOSS_NAME}(style,full_barra_fitted) + "
            f"{RAW_BARRA_R2_LAMBDA}*R2(alpha|ten_styles+industry)"
            if dual_tower_enabled()  # 根据当前数据状态选择处理分支。
            else
            f"{LOSS_NAME}(strict_full_barra_projection,full_barra_residual_target)"
            if strict_barra_output_enabled()  # 根据当前数据状态选择处理分支。
            else
            f"{LOSS_NAME}(raw_score,full_barra_residual_target) + "
            f"{RAW_BARRA_R2_LAMBDA}*R2(raw_score|ten_styles+industry)"
            if barra_conditioning_enabled()  # 根据当前数据状态选择处理分支。
            else
            f"{LOSS_NAME}(raw_score,source_target) + masked_style_mse + "
            "weighted_industry_ce/log(32) through GRL"
            if adversarial_barra_enabled()  # 根据当前数据状态选择处理分支。
            else
            f"{LOSS_NAME}(raw_score,source_target) + "
            f"{RAW_BARRA_R2_LAMBDA}*R2(raw_score|ten_styles+industry)"
            if raw_barra_r2_enabled()  # 根据当前数据状态选择处理分支。
            else f"{LOSS_NAME}(effective_prediction,effective_target)"
        ),
        "early_stop_target": (  # 记录early终点标签。
            "alpha_score_vs_full_barra_residual_target"
            if dual_tower_enabled()  # 根据当前数据状态选择处理分支。
            else "full_barra_residual_target"
            if barra_conditioning_enabled()  # 根据当前数据状态选择处理分支。
            else "style_residual_target"
            if RESIDUALIZE_TARGET  # 根据当前数据状态选择处理分支。
            else "source_target"
        ),
        "precision": "bf16_model_forward_fp32_projection_barra_r2_and_loss",  # 把 precision 映射为"bf16_model_forward_fp32_projection_barra_r2_and_loss"。
    }
    if EXECUTION_ROUTE == "ddp":  # 按当前模式、标识或状态选择处理分支。
        payload.update({  # 用当前配置更新参数字典。
            "execution_route": "ddp",  # 把 execution_route 映射为"ddp"。
            "world_size": DDP_WORLD_SIZE,  # 记录world大小。
        })
    if dual_tower_enabled():  # 根据当前数据状态选择处理分支。
        payload["full_barra_target_sidecar_signature"] = (
            full_target_manifest["signature"]
        )
        payload["dual_tower_target_sidecar_signature"] = (
            dual_target_manifest["signature"]
        )
        payload["barra_dual_tower"] = {  # 初始化载荷容器。
            "mode": BARRA_DUAL_TOWER_MODE,  # 记录模式。
            "alpha_architecture": "vivit_cls_times_bounded_barra_gate",  # 把 alpha_architecture 映射为"vivit_cls_times_bounded_barra_gate"。
            "style_architecture": "independent_barra_mlp_scalar_head",  # 把 style_architecture 映射为"independent_barra_mlp_scalar_head"。
            "style_hidden_dim": DUAL_TOWER_STYLE_HIDDEN_DIM,  # 记录风格隐藏层dim。
            "style_dropout": DUAL_TOWER_STYLE_DROPOUT,  # 记录风格dropout。
            "industry_embedding_dim": BARRA_INDUSTRY_EMBEDDING_DIM,  # 记录行业嵌入dim。
            "style_output_initialization": "zeros",  # 把 style_output_initialization 映射为"zeros"。
            "return_loss_weight": 1.0,  # 记录收益损失权重。
            "alpha_loss_weight": DUAL_TOWER_ALPHA_LOSS_WEIGHT,  # 记录Alpha损失权重。
            "style_loss_weight": DUAL_TOWER_STYLE_LOSS_WEIGHT,  # 记录风格损失权重。
            "alpha_r2_lambda": RAW_BARRA_R2_LAMBDA,  # 记录Alphar2lambda。
            "r2_uses_true_unmodified_barra": True,  # 记录r2usestrueunmodifiedBarra。
            "early_stop": "alpha_vs_full_barra_residual_daily_spearman",  # 把 early_stop 映射为"alpha_vs_full_barra_residual_daily_spearman"。
            "published_factor": "unprojected_alpha_score",  # 把 published_factor 映射为"unprojected_alpha_score"。
            "ensemble_scaling": "center_components_divide_by_alpha_std",  # 把 ensemble_scaling 映射为"center_components_divide_by_alpha_std"。
            "shuffle_policy": (  # 声明shuffle处理口径。
                "within_date_joint_permutation_by_stable_sample_key_v1"
                if BARRA_DUAL_TOWER_MODE == "shuffled" else None  # 按当前模式、标识或状态选择处理分支。
            ),
        }
    if barra_conditioning_enabled():  # 根据当前数据状态选择处理分支。
        payload["full_barra_target_sidecar_signature"] = (
            full_target_manifest["signature"]
        )
        payload["barra_conditioning"] = {  # 初始化载荷容器。
            "mode": BARRA_CONDITIONING_MODE,  # 记录模式。
            "continuous_inputs": list(style_data.STYLE_COLUMNS),  # 记录continuousinputs。
            "valid_mask_inputs": list(style_data.STYLE_COLUMNS),  # 记录有效掩码inputs。
            "industry_classes": INDUSTRY_COUNT,  # 记录行业classes。
            "unknown_industry_embedding": "code_zero_fixed_zero",  # 把 unknown_industry_embedding 映射为"code_zero_fixed_zero"。
            "industry_embedding_dim": BARRA_INDUSTRY_EMBEDDING_DIM,  # 记录行业嵌入dim。
            "hidden_dim": BARRA_GATE_HIDDEN_DIM,  # 记录隐藏层dim。
            "activation": "gelu",  # 把 activation 映射为"gelu"。
            "dropout": BARRA_GATE_DROPOUT,  # 记录dropout。
            "gate": "scale_times_tanh_linear",  # 把 gate 映射为"scale_times_tanh_linear"。
            "gate_scale": BARRA_GATE_SCALE,  # 记录门控scale。
            "gate_projection_initialization": "zeros",  # 把 gate_projection_initialization 映射为"zeros"。
            "conditioned_representation": "h_times_one_plus_gate",  # 把 conditioned_representation 映射为"h_times_one_plus_gate"。
            "additive_barra_alpha": False,  # 记录additiveBarraAlpha。
            "shuffle_policy": (  # 声明shuffle处理口径。
                "within_date_joint_permutation_by_stable_sample_key_v1"
                if BARRA_CONDITIONING_MODE == "shuffled" else None  # 按当前模式、标识或状态选择处理分支。
            ),
            "r2_uses_true_unmodified_barra": True,  # 记录r2usestrueunmodifiedBarra。
        }
    if strict_barra_output_enabled():  # 根据当前数据状态选择处理分支。
        payload["strict_barra_output"] = {  # 初始化载荷容器。
            "weighting": "equal_weight_W_identity",  # 把 weighting 映射为"equal_weight_W_identity"。
            "design": [  # 记录design。
                "intercept",
                *style_data.STYLE_COLUMNS,
                "industry_fixed_effects",
            ],
            "solver": "industry_within_then_torch_fp32_reduced_qr",  # 把 solver 映射为"industry_within_then_torch_fp32_reduced_qr"。
            "post_projection": "zscore_ddof0",  # 把 post_projection 映射为"zscore_ddof0"。
            "applied_during": ["training_loss", "validation", "inference"],  # 记录appliedduring。
            "published_factor": "strict_projected_score",  # 把 published_factor 映射为"strict_projected_score"。
            "pre_neutral_score_is_audit_only": True,  # 记录preneutral分数isauditonly。
        }
    return payload  # 返回载荷。


def load_best_model(  # 加载最佳验证 checkpoint，并恢复可用于预测的模型。
    path: Path, device: torch.device
) -> torch.nn.Module:  # 声明当前函数的返回类型。
    """加载最佳验证 checkpoint，并恢复可用于预测的模型。"""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint["model_state"]
    if dual_tower_enabled() or barra_conditioning_enabled():  # 根据当前数据状态选择处理分支。
        model = build_training_model(device, use_compile=False)  # 创建模型。
        base.load_canonical_state(model, state)
        model.eval()  # 关闭 dropout 等训练行为。
        return model  # 返回模型。
    if adversarial_barra_enabled():  # 根据当前数据状态选择处理分支。
        prefix = "backbone."
        state = {  # 初始化状态容器。
            key.removeprefix(prefix): value
            for key, value in state.items()  # 遍历状态items，逐项处理键值。
            if key.startswith(prefix)  # 根据当前数据状态选择处理分支。
        }
        if not state:  # 检查必需数据、配置或校验结果是否缺失。
            raise ValueError("No.3 checkpoint 缺少 backbone 参数")  # 输入数据不满足业务或数值约束时立即终止。
    model = base.build_model(  # 创建模型并部署到设备。
        device, bool(USE_TORCH_COMPILE), COMPILE_MODE, TRAIN_BATCH_MODE
    )
    base.load_canonical_state(model, state)
    model.eval()  # 关闭 dropout 等训练行为。
    return model  # 返回模型。


def load_best_training_model(  # 加载包含附加实验分支的最佳训练模型。
    path: Path, device: torch.device
) -> torch.nn.Module:  # 声明当前函数的返回类型。
    """加载包含附加实验分支的最佳训练模型。"""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_training_model(device, use_compile=False)  # 创建模型。
    base.load_canonical_state(model, checkpoint["model_state"])
    model.eval()  # 关闭 dropout 等训练行为。
    return model  # 返回模型。


def train_task(  # 执行一个年度×seed 任务，保存 checkpoint、预测和训练诊断。
    year: int,  # 接收年份。
    seed: int,  # 接收随机种子。
    physical_gpu: int,  # 接收physicalgpu。
    *,
    max_epochs: int | None = None,  # 接收maxepochs。
    max_batches: int | None = None,  # 接收maxbatches。
    persist: bool = True,  # 接收persist。
    distributed_context: distributed.DistributedContext | None = None,  # 接收distributed上下文。
) -> dict[str, Any]:  # 声明当前函数的返回类型。
    """执行一个年度×seed 任务，保存 checkpoint、预测和训练诊断。"""
    style_data.require_style_neutral_store()
    if industry_sidecar_required():  # 根据当前数据状态选择处理分支。
        style_data.require_industry_sidecar()
    if barra_conditioning_enabled() or dual_tower_enabled():  # 根据当前数据状态选择处理分支。
        style_data.require_full_barra_target_sidecar()
    if dual_tower_enabled():  # 根据当前数据状态选择处理分支。
        style_data.require_dual_tower_target_sidecar()
    primary = distributed_context is None or distributed_context.is_primary  # 计算primary判定条件。
    device = (
        distributed_context.device if distributed_context is not None
        else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    if device.type != "cuda":  # 按当前模式、标识或状态选择处理分支。
        raise RuntimeError("完整风格中性训练要求 CUDA")  # 当前运行环境或任务状态不支持继续执行。
    base.set_random_seed(  # 固定各随机源。
        seed + (distributed_context.rank if distributed_context else 0)
    )
    train_loader, validation_loader, prediction_loader = make_style_loaders(  # 创建训练数据加载器。
        year, seed, WORKERS_PER_TASK if persist else 0, pin_memory=True,
        distributed_context=distributed_context,  # 配置distributed上下文。
    )
    split = sequence_data.load_sequence_split(year)
    industry_class_weights = (
        training_industry_class_weights(split["train_indices"]).to(device)
        if adversarial_barra_enabled()  # 根据当前数据状态选择处理分支。
        else torch.ones(INDUSTRY_COUNT, dtype=torch.float32, device=device)
    )
    batch_summary = sequence_data.loader_batch_summary(train_loader)
    payload = run_payload(year, seed, batch_summary)
    signature = sequence_data.stable_hash(payload)  # 生成数据复用签名。
    config = training_config()
    epoch_limit = int(max_epochs or MAX_EPOCHS)

    model_root, _, _ = base.run_paths(RUN_NAME)  # 构造模型根目录。
    run_directory = model_root / f"forecast_{year}" / f"seed_{seed}"
    best_path = run_directory / "best.pt"  # 构造最优路径。
    last_path = run_directory / "last.pt"  # 构造last路径。
    history_path = run_directory / "history.parquet"  # 构造历史记录路径。
    prediction_path = run_directory / "prediction.parquet"  # 构造预测路径。
    probe_path = run_directory / "adversarial_probe_diagnostics.parquet"  # 构造probe路径。
    completed_path = run_directory / "completed.json"  # 构造completed路径。
    resume_checkpoint: dict[str, Any] | None = None  # 声明可选输入；未提供时沿用默认处理流程。
    if persist:  # 根据当前数据状态选择处理分支。
        if primary:  # 根据当前数据状态选择处理分支。
            run_directory.mkdir(parents=True, exist_ok=True)  # 确保目标目录存在。
        if distributed_context is not None:  # 检查必需数据、配置或校验结果是否缺失。
            dist.barrier()
        if completed_path.exists() and best_path.exists() and not OVERWRITE:  # 检查目标文件或目录是否满足存在性要求。
            completed = json.loads(completed_path.read_text(encoding="utf-8"))  # 读取completed。
            if completed.get("run_signature") == signature:  # 按当前模式、标识或状态选择处理分支。
                print(f"[{year}/seed={seed}] 已完成，直接复用。", flush=True)  # 输出当前任务的进度或诊断信息。
                return completed["summary"]  # 返回上述处理结果。
            raise ValueError(  # 输入数据不满足业务或数值约束时立即终止。
                f"{completed_path} 与当前配置签名不一致；"
                "请使用新 run name，或显式设置 --overwrite 1"
            )
        if last_path.exists() and not OVERWRITE:  # 检查目标文件或目录是否满足存在性要求。
            resume_checkpoint = torch.load(
                last_path, map_location="cpu", weights_only=False  # 构造last路径。
            )
            if resume_checkpoint.get("run_signature") != signature:  # 按当前模式、标识或状态选择处理分支。
                raise ValueError(  # 输入数据不满足业务或数值约束时立即终止。
                    f"{last_path} 与当前配置签名不一致；"
                    "请使用新 run name，或显式设置 --overwrite 1"
                )
        if primary:  # 原子落盘，避免半成品。
            sequence_data.atomic_write_json(
                {
                    "signature": sequence_data.stable_hash(  # 记录签名。
                        {k: v for k, v in payload.items() if k != "seed"}
                    ),
                    "payload": {k: v for k, v in payload.items() if k != "seed"},  # 记录载荷。
                },
                model_root / f"forecast_{year}" / "model_config.json",
            )
        if distributed_context is not None:  # 检查必需数据、配置或校验结果是否缺失。
            dist.barrier()

    model = build_training_model(  # 创建模型。
        device, use_compile=bool(USE_TORCH_COMPILE and persist)
    )
    if distributed_context is not None:  # 检查必需数据、配置或校验结果是否缺失。
        model = distributed.wrap_model(model, distributed_context)  # 创建模型。
    optimizer = base.build_optimizer(  # 按训练配置创建优化器。
        model, config, bool(USE_FUSED_ADAM), device
    )
    start_epoch = 1
    best_epoch = 0
    best_validation_ic = -np.inf
    best_metrics: dict[str, float] = {}  # 计算最优指标。
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []  # 初始化历史记录容器。

    if resume_checkpoint is not None:  # 检查必需数据、配置或校验结果是否缺失。
        checkpoint = resume_checkpoint
        base.load_canonical_state(model, checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])  # 恢复优化器续训状态。
        if distributed_context is not None:  # 检查必需数据、配置或校验结果是否缺失。
            states = checkpoint.get("rng_state_by_rank")
            if states is None or len(states) != distributed_context.world_size:  # 检查所需对象是否尚未初始化。
                raise ValueError("DDP checkpoint 缺少匹配 world size 的逐 rank RNG 状态")  # 输入数据不满足业务或数值约束时立即终止。
            distributed.restore_rng_state(states[distributed_context.rank], device)
        else:  # 处理上述条件均不成立的情况。
            random.setstate(checkpoint["python_random_state"])
            np.random.set_state(checkpoint["numpy_random_state"])
            torch.set_rng_state(checkpoint["torch_rng_state"])
            if checkpoint["cuda_rng_state"] is not None:  # 检查必需数据、配置或校验结果是否缺失。
                torch.cuda.set_rng_state(checkpoint["cuda_rng_state"], device)
        start_epoch = int(checkpoint["epoch"]) + 1
        best_epoch = int(checkpoint["best_epoch"])
        best_validation_ic = float(checkpoint["best_validation_ic"])
        best_metrics = {  # 计算最优指标。
            k: float(v) for k, v in checkpoint["best_metrics"].items()
        }
        epochs_without_improvement = int(
            checkpoint["epochs_without_improvement"]
        )
        if history_path.exists():  # 检查目标文件或目录是否满足存在性要求。
            history = pd.read_parquet(history_path).to_dict("records")  # 读取历史记录。
            checkpoint_epoch = int(checkpoint["epoch"])
            history = [  # 初始化历史记录容器。
                row for row in history
                if int(row["epoch"]) <= checkpoint_epoch  # 检查当前数值是否越过业务或数值阈值。
            ]
        print(f"[{year}/seed={seed}] 从 epoch {start_epoch} 续训。", flush=True)  # 输出当前任务的进度或诊断信息。

    started = time.perf_counter()
    stopped_early = False
    stop_reason = "max_epochs"
    validation_score_collapsed = False  # 计算验证分数collapsed。
    collapsed_validation_date_id: int | None = None  # 接收collapsed验证交易日id。
    collapsed_validation_score_std: float | None = None  # 接收collapsed验证分数标准差。
    collapsed_validation_total_sum_squares: float | None = None  # 接收collapsed验证totalsumsquares。
    last_successful_epoch = start_epoch - 1
    for epoch in range(start_epoch, epoch_limit + 1):  # 按给定步长遍历样本区间，更新轮次。
        epoch_started = time.perf_counter()
        sequence_data.set_train_epoch(train_loader, epoch)  # 完成一个训练 epoch。
        train_metrics = train_one_epoch(  # 完成一个训练 epoch。
            model,
            train_loader,
            optimizer,
            device,
            epoch=epoch,  # 配置轮次。
            industry_class_weights=industry_class_weights,  # 配置行业class权重。
            max_batches=max_batches,  # 限制调试运行处理的最大批次数。
            distributed_context=distributed_context,  # 配置distributed上下文。
        )
        try:  # 执行可能失败的外部读取或计算步骤。
            validation_metrics = evaluate_model(  # 无梯度计算验证指标。
                model,
                validation_loader,
                device,
                industry_class_weights=industry_class_weights,  # 配置行业class权重。
                max_batches=max_batches,  # 限制调试运行处理的最大批次数。
                distributed_context=distributed_context,  # 配置distributed上下文。
            )
        except DegenerateValidationScoreError as exc:  # 捕获预期异常并执行对应恢复逻辑。
            if not persist or best_epoch <= 0 or not best_path.exists():  # 检查目标文件或目录是否满足存在性要求。
                raise
            stopped_early = True
            stop_reason = "validation_score_collapse"
            validation_score_collapsed = True  # 计算验证分数collapsed。
            collapsed_validation_date_id = exc.date_id
            collapsed_validation_score_std = exc.score_std  # 计算collapsed验证分数标准差。
            collapsed_validation_total_sum_squares = exc.total_sum_squares
            history.append(  # 把当前结果追加到汇总列表。
                {
                    "forecast_year": year,  # 记录forecast年份。
                    "seed": seed,  # 记录随机种子。
                    "epoch": epoch,  # 记录轮次。
                    "loss_name": LOSS_NAME,  # 记录损失名称。
                    "residualize_target": RESIDUALIZE_TARGET,  # 记录残差化标签。
                    "project_model_output": PROJECT_MODEL_OUTPUT,  # 记录投影模型输出。
                    "style_correlation_penalty": raw_barra_r2_enabled(),  # 记录风格相关系数penalty。
                    "adversarial_barra": adversarial_barra_enabled(),  # 记录adversarialBarra。
                    "raw_barra_r2_lambda": RAW_BARRA_R2_LAMBDA,  # 记录原始Barrar2lambda。
                    **train_metrics,
                    "validation_ic": np.nan,  # 记录验证ic。
                    "validation_daily_rank_ic": np.nan,  # 记录验证日度秩ic。
                    "validation_original_daily_rank_ic": np.nan,  # 记录验证原始日度秩ic。
                    "validation_effective_max_abs_style_pearson": np.nan,  # 记录验证有效maxabs风格Pearson。
                    "validation_barra_r_squared_mean": np.nan,  # 记录验证Barrarsquared均值。
                    "validation_score_collapsed": True,  # 记录验证分数collapsed。
                    "validation_collapsed_date_count": 1,  # 记录验证collapsed交易日数量。
                    "validation_collapsed_date_id": exc.date_id,  # 记录验证collapsed交易日id。
                    "validation_min_daily_raw_score_std": exc.score_std,  # 记录验证min日度原始分数标准差。
                    "validation_collapse_total_sum_squares": (  # 记录验证collapsetotalsumsquares。
                        exc.total_sum_squares
                    ),
                    "best_validation_ic": best_validation_ic,  # 记录最优验证ic。
                    "epochs_without_improvement": epochs_without_improvement,  # 记录epochswithoutimprovement。
                    "epoch_seconds": time.perf_counter() - epoch_started,  # 记录轮次seconds。
                }
            )
            if primary:  # 原子落盘，避免半成品。
                sequence_data.atomic_write_parquet(pd.DataFrame(history), history_path)
            if distributed_context is not None:  # 检查必需数据、配置或校验结果是否缺失。
                dist.barrier()
            print(  # 输出当前任务的进度或诊断信息。
                f"[{year}/seed={seed}] epoch={epoch:03d} 验证分数退化；"
                f"date_id={exc.date_id}, std={exc.score_std:.3e}。"
                f"停止当前 seed，并使用 best epoch={best_epoch} 生成预测。",
                flush=True,  # 配置flush。
            )
            break  # 满足终止条件后退出当前循环。
        selected = float(validation_metrics["validation_ic"])  # 构造筛选结果，限定后续参与计算的样本。
        improved = selected > best_validation_ic + config.improvement_tolerance  # 计算improved判定条件。
        if improved:  # 根据当前数据状态选择处理分支。
            best_epoch = epoch
            best_validation_ic = selected
            best_metrics = {  # 计算最优指标。
                key: float(value)  # 接收键。
                for key, value in validation_metrics.items()  # 遍历验证指标items，逐项处理键值。
                if isinstance(value, (int, float))  # 检查对象类型是否符合接口约定。
            }
            epochs_without_improvement = 0
        else:  # 处理上述条件均不成立的情况。
            epochs_without_improvement += 1
        record = {  # 初始化record容器。
            "forecast_year": year,  # 记录forecast年份。
            "seed": seed,  # 记录随机种子。
            "epoch": epoch,  # 记录轮次。
            "loss_name": LOSS_NAME,  # 记录损失名称。
            "residualize_target": RESIDUALIZE_TARGET,  # 记录残差化标签。
            "project_model_output": PROJECT_MODEL_OUTPUT,  # 记录投影模型输出。
            "style_correlation_penalty": raw_barra_r2_enabled(),  # 记录风格相关系数penalty。
            "adversarial_barra": adversarial_barra_enabled(),  # 记录adversarialBarra。
            "raw_barra_r2_lambda": RAW_BARRA_R2_LAMBDA,  # 记录原始Barrar2lambda。
            **train_metrics,
            **validation_metrics,
            "best_validation_ic": best_validation_ic,  # 记录最优验证ic。
            "epochs_without_improvement": epochs_without_improvement,  # 记录epochswithoutimprovement。
            "epoch_seconds": time.perf_counter() - epoch_started,  # 记录轮次seconds。
        }
        history.append(record)  # 把当前结果追加到汇总列表。
        if persist:  # 根据当前数据状态选择处理分支。
            if primary:  # 原子落盘，避免半成品。
                sequence_data.atomic_write_parquet(pd.DataFrame(history), history_path)
            rng_state_by_rank = (  # 计算rng状态by排名。
                distributed.gather_rng_states(device)
                if distributed_context is not None else None  # 检查必需数据、配置或校验结果是否缺失。
            )
            checkpoint = {  # 初始化检查点容器。
                "model_state": base.unwrap_model(model).state_dict(),  # 记录模型状态。
                "optimizer_state": optimizer.state_dict(),  # 记录优化器状态。
                "python_random_state": random.getstate(),  # 记录pythonrandom状态。
                "numpy_random_state": np.random.get_state(),  # 记录numpyrandom状态。
                "torch_rng_state": torch.get_rng_state(),  # 记录torchrng状态。
                "cuda_rng_state": torch.cuda.get_rng_state(device),  # 记录cudarng状态。
                "epoch": epoch,  # 记录轮次。
                "best_epoch": best_epoch,  # 记录最优轮次。
                "best_validation_ic": best_validation_ic,  # 记录最优验证ic。
                "best_metrics": best_metrics,  # 记录最优指标。
                "epochs_without_improvement": epochs_without_improvement,  # 记录epochswithoutimprovement。
                "run_signature": signature,  # 记录运行签名，用于来源一致性校验。
                "run_payload": payload,  # 记录运行载荷。
            }
            if rng_state_by_rank is not None:  # 检查必需数据、配置或校验结果是否缺失。
                checkpoint["rng_state_by_rank"] = rng_state_by_rank  # 计算检查点。
            if primary:  # 根据当前数据状态选择处理分支。
                sequence_data.atomic_save_checkpoint(checkpoint, last_path)
            if primary and improved:  # 根据当前数据状态选择处理分支。
                sequence_data.atomic_save_checkpoint(checkpoint, best_path)
            if distributed_context is not None:  # 检查必需数据、配置或校验结果是否缺失。
                dist.barrier()
        last_successful_epoch = epoch
        print(  # 输出当前任务的进度或诊断信息。
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
            flush=True,  # 配置flush。
        )
        if epochs_without_improvement >= PATIENCE:  # 检查当前数值是否越过业务或数值阈值。
            stopped_early = True
            stop_reason = "patience"
            break  # 满足终止条件后退出当前循环。

    if not persist:  # 检查必需数据、配置或校验结果是否缺失。
        return {  # 返回上述处理结果。
            "forecast_year": year,  # 记录forecast年份。
            "seed": seed,  # 记录随机种子。
            "device": str(device),  # 记录设备。
            "train_metrics": train_metrics,  # 记录训练指标。
            "validation_metrics": validation_metrics,  # 记录验证指标。
            "persistent_outputs_written": False,  # 记录persistentoutputswritten。
        }
    if not best_path.exists():  # 检查目标文件或目录是否满足存在性要求。
        raise FileNotFoundError(f"未生成最佳 checkpoint：{best_path}")  # 目标文件不存在时立即终止。
    if prediction_loader is not None:  # 检查必需数据、配置或校验结果是否缺失。
        best_model = load_best_model(best_path, device)  # 创建最优模型。
        prediction_frame = predict_model(
            best_model, prediction_loader, device,  # 传入最优模型、预测数据加载器、设备。
            distributed_context=distributed_context,  # 配置distributed上下文。
        )
        if primary:  # 原子落盘，避免半成品。
            sequence_data.atomic_write_parquet(prediction_frame, prediction_path)
        del best_model  # 释放不再使用的中间引用。
        if distributed_context is not None:  # 检查必需数据、配置或校验结果是否缺失。
            dist.barrier()
    probe_summary: dict[str, float] = {}  # 初始化probe汇总容器。
    if primary and adversarial_barra_enabled():  # 根据当前数据状态选择处理分支。
        probe_validation_loader = validation_loader  # 创建probe验证数据加载器。
        if distributed_context is not None:  # 检查必需数据、配置或校验结果是否缺失。
            _, probe_validation_loader, _ = make_style_loaders(
                year, seed, 0, pin_memory=True,
                distributed_context=None,  # 配置distributed上下文。
            )
        probe_model = load_best_training_model(best_path, device)  # 创建probe模型。
        probe_frame = run_adversarial_probes(  # 用探针测量表征中的风险信息。
            probe_model,
            probe_validation_loader,
            device,
            year=year,  # 配置年份。
            seed=seed,  # 固定本轮随机种子。
        )
        sequence_data.atomic_write_parquet(probe_frame, probe_path)  # 原子落盘，避免半成品。
        style_summary = probe_frame[  # 按约定截面口径汇总。
            probe_frame["target_kind"].eq("style_summary")
        ].set_index("probe")
        industry_summary = probe_frame[
            probe_frame["target_kind"].eq("industry")
        ].set_index("probe")
        probe_summary = {  # 初始化probe汇总容器。
            "linear_probe_style_r_squared_mean": float(  # 记录linearprobe风格rsquared均值。
                style_summary.loc["linear_ridge", "r_squared"]
            ),
            "mlp_probe_style_r_squared_mean": float(  # 记录mlpprobe风格rsquared均值。
                style_summary.loc["mlp", "r_squared"]
            ),
            "linear_probe_industry_balanced_accuracy": float(  # 记录linearprobe行业balancedaccuracy。
                industry_summary.loc["linear_softmax", "balanced_accuracy"]
            ),
            "mlp_probe_industry_balanced_accuracy": float(  # 记录mlpprobe行业balancedaccuracy。
                industry_summary.loc["mlp", "balanced_accuracy"]
            ),
        }
        del probe_model  # 释放不再使用的中间引用。
    summary = {  # 初始化汇总容器。
        "forecast_year": year,  # 记录forecast年份。
        "seed": seed,  # 记录随机种子。
        "best_epoch": best_epoch,  # 记录最优轮次。
        "best_validation_ic": best_validation_ic,  # 记录最优验证ic。
        "best_validation_rank_ic": best_metrics.get(  # 记录最优验证秩ic。
            "validation_daily_rank_ic"
        ),
        "best_validation_original_rank_ic": best_metrics.get(  # 记录最优验证原始秩ic。
            "validation_original_daily_rank_ic"
        ),
        "best_validation_effective_max_abs_style_pearson": best_metrics.get(  # 记录最优验证有效maxabs风格Pearson。
            "validation_effective_max_abs_style_pearson"
        ),
        "best_validation_barra_r_squared": best_metrics.get(  # 记录最优验证Barrarsquared。
            "validation_barra_r_squared_mean"
        ),
        "best_validation_effective_barra_r_squared": best_metrics.get(  # 记录最优验证有效Barrarsquared。
            "validation_effective_barra_r_squared_mean"
        ),
        "best_validation_effective_max_abs_industry_mean": best_metrics.get(  # 记录最优验证有效maxabs行业均值。
            "validation_effective_max_abs_industry_mean"
        ),
        "best_validation_gate_mean_abs": best_metrics.get(  # 记录最优验证门控均值abs。
            "validation_gate_mean_abs"
        ),
        "best_validation_gate_max_abs": best_metrics.get(  # 记录最优验证门控maxabs。
            "validation_gate_max_abs"
        ),
        "best_validation_total_original_rank_ic": best_metrics.get(  # 记录最优验证total原始秩ic。
            "validation_total_original_daily_rank_ic"
        ),
        "best_validation_style_fitted_rank_ic": best_metrics.get(  # 记录最优验证风格拟合秩ic。
            "validation_style_fitted_daily_rank_ic"
        ),
        "best_validation_dual_additivity_max_abs_error": best_metrics.get(  # 记录最优验证双塔additivitymaxabs误差。
            "validation_dual_additivity_max_abs_error"
        ),
        "best_validation_adversarial_style_r_squared_mean": best_metrics.get(  # 记录最优验证adversarial风格rsquared均值。
            "validation_adversarial_style_r_squared_mean"
        ),
        "best_validation_adversarial_industry_accuracy": best_metrics.get(  # 记录最优验证adversarial行业accuracy。
            "validation_adversarial_industry_accuracy"
        ),
        "best_validation_adversarial_industry_balanced_accuracy": (  # 记录最优验证adversarial行业balancedaccuracy。
            best_metrics.get(
                "validation_adversarial_industry_balanced_accuracy"
            )
        ),
        "last_epoch": int(last_successful_epoch),  # 记录最后一个轮次。
        "last_attempted_epoch": int(history[-1]["epoch"]),  # 记录最后一个attempted轮次。
        "stopped_early": stopped_early,  # 记录stoppedearly。
        "stop_reason": stop_reason,  # 记录终点reason。
        "validation_score_collapsed": validation_score_collapsed,  # 记录验证分数collapsed。
        "collapsed_validation_date_count": int(validation_score_collapsed),  # 记录collapsed验证交易日数量。
        "collapsed_validation_date_id": collapsed_validation_date_id,  # 记录collapsed验证交易日id。
        "collapsed_validation_score_std": collapsed_validation_score_std,  # 记录collapsed验证分数标准差。
        "collapsed_validation_total_sum_squares": (  # 记录collapsed验证totalsumsquares。
            collapsed_validation_total_sum_squares
        ),
        "elapsed_seconds": time.perf_counter() - started,  # 记录elapsedseconds。
        "physical_gpu": physical_gpu,  # 记录physicalgpu。
        "run_name": RUN_NAME,  # 记录运行名称。
        "execution_route": EXECUTION_ROUTE,  # 记录执行route。
        "world_size": DDP_WORLD_SIZE,  # 记录world大小。
        "physical_device_ids": list(EXECUTION_DEVICE_IDS),  # 记录physical设备ids。
        "loss_name": LOSS_NAME,  # 记录损失名称。
        "residualize_target": RESIDUALIZE_TARGET,  # 记录残差化标签。
        "project_model_output": PROJECT_MODEL_OUTPUT,  # 记录投影模型输出。
        "style_correlation_penalty": raw_barra_r2_enabled(),  # 记录风格相关系数penalty。
        "adversarial_barra": adversarial_barra_enabled(),  # 记录adversarialBarra。
        "adversarial_gamma_max": ADVERSARIAL_GAMMA_MAX,  # 记录adversarial对抗系数max。
        "barra_conditioning_mode": BARRA_CONDITIONING_MODE,  # 记录Barraconditioning模式。
        "barra_output_neutralization": strict_barra_output_enabled(),  # 记录Barra输出neutralization。
        "barra_dual_tower": dual_tower_enabled(),  # 记录Barra双塔塔。
        "barra_dual_tower_mode": BARRA_DUAL_TOWER_MODE,  # 记录Barra双塔塔模式。
        "dual_tower_alpha_loss_weight": DUAL_TOWER_ALPHA_LOSS_WEIGHT,  # 记录双塔塔Alpha损失权重。
        "dual_tower_style_loss_weight": DUAL_TOWER_STYLE_LOSS_WEIGHT,  # 记录双塔塔风格损失权重。
        "barra_gate_scale": BARRA_GATE_SCALE,  # 记录Barra门控scale。
        **probe_summary,
        "raw_barra_r2_lambda": RAW_BARRA_R2_LAMBDA,  # 记录原始Barrar2lambda。
        "prediction_sample_count": len(  # 记录预测样本数量。
            sequence_data.load_sequence_split(year)["prediction_indices"]
        ),
        "status": "completed",  # 把 status 映射为"completed"。
    }
    if distributed_context is not None:  # 原子落盘，避免半成品。
        dist.barrier()
    if primary:  # 根据当前数据状态选择处理分支。
        sequence_data.atomic_write_json(
            {"run_signature": signature, "summary": summary}, completed_path
        )
    if distributed_context is not None:  # 检查必需数据、配置或校验结果是否缺失。
        dist.barrier()
    return summary  # 返回汇总。


# %% [markdown]
# ## 步骤 5：三 seed 集成、风格相关性审计和 run manifest


# %%
def canonical_dual_tower_total_scores(  # 审计推理精度，并以落盘分量重建严格可加的 Total。
    alpha_values: np.ndarray,  # 接收Alpha值。
    style_values: np.ndarray,  # 接收风格值。
    stored_total_values: np.ndarray,  # 接收storedtotal值。
) -> tuple[np.ndarray, dict[str, Any]]:  # 声明当前函数的返回类型。
    """审计推理精度，并以落盘分量重建严格可加的 Total。"""

    alpha = np.asarray(alpha_values, dtype=np.float32)  # 转换并统一Alpha的数据表示。
    style = np.asarray(style_values, dtype=np.float32)  # 转换并统一风格的数据表示。
    stored = np.asarray(stored_total_values, dtype=np.float32)  # 转换并统一stored的数据表示。
    if alpha.shape != style.shape or alpha.shape != stored.shape:  # 检查张量或数组形状是否符合接口约定。
        raise ValueError("No.6 Alpha、Style、Total 形状不一致")  # 输入数据不满足业务或数值约束时立即终止。
    if not np.isfinite(np.column_stack([alpha, style, stored])).all():  # 检查参与计算的数值是否全部有限。
        raise ValueError("No.6 Alpha、Style、Total 含 NaN/Inf")  # 输入数据不满足业务或数值约束时立即终止。

    canonical = (alpha + style).astype(np.float32)  # 转换canonical的数据类型或存储位置。
    bfloat16_sum = (
        torch.from_numpy(alpha.copy()).to(torch.bfloat16)
        + torch.from_numpy(style.copy()).to(torch.bfloat16)
    ).float().numpy()
    matches_float32 = np.array_equal(stored, canonical)  # 核对样本顺序严格一致。
    matches_bfloat16 = np.array_equal(stored, bfloat16_sum)  # 核对样本顺序严格一致。
    if not matches_float32 and not matches_bfloat16:  # 检查必需数据、配置或校验结果是否缺失。
        raise AssertionError(  # 关键数据契约被破坏时立即终止。
            "No.6 推理 Total 既不满足 float32 加法，也不满足 bfloat16 加法"
        )
    error = np.abs(stored.astype(np.float64) - canonical.astype(np.float64))  # 转换error的数据类型或存储位置。
    return canonical, {  # 返回审计推理精度，并以落盘分量重建严格可加的 Total。
        "stored_total_precision": (  # 记录存储值totalprecision。
            "float32" if matches_float32 else "bfloat16"
        ),
        "stored_vs_canonical_max_abs_error": float(error.max()),  # 记录存储值vs规范值maxabs误差。
        "stored_vs_canonical_mean_abs_error": float(error.mean()),  # 记录存储值vs规范值均值abs误差。
    }


def build_ensemble(year: int) -> tuple[int, pd.DataFrame]:  # 按既定尺度组合多个随机种子的预测，并保存正式年度信号。
    """按既定尺度组合多个随机种子的预测，并保存正式年度信号。"""
    model_root, prediction_root, _ = base.run_paths(RUN_NAME)  # 构造模型根目录。
    split = sequence_data.load_sequence_split(year)
    if not len(split["prediction_indices"]):  # 检查当前样本数量是否满足计算要求。
        return 0, pd.DataFrame()  # 返回按当前规则组织后的结果。
    frames = {  # 初始化帧容器。
        seed: pd.read_parquet(  # 接收随机种子。
            model_root / f"forecast_{year}" / f"seed_{seed}"
            / "prediction.parquet"
        )
        for seed in SEEDS  # 遍历随机种子，逐项处理随机种子。
    }
    reference = frames[SEEDS[0]]["sample_index"].to_numpy(np.int64)  # 转换并统一reference的数据表示。
    output = pd.DataFrame({"sample_index": reference})
    raw_columns: list[str] = []  # 接收原始列。
    effective_columns: list[str] = []  # 接收有效列。
    dual_style_columns: list[str] = []  # 接收dual风格列。
    dual_total_columns: list[str] = []  # 接收dualtotal列。
    dual_inference_audits: dict[int, dict[str, Any]] = {}  # 初始化dualinferenceaudits容器。
    for seed in SEEDS:  # 遍历随机种子，逐项处理随机种子。
        frame = frames[seed]
        if not np.array_equal(  # 检查必需数据、配置或校验结果是否缺失。
            reference, frame["sample_index"].to_numpy(np.int64)
        ):
            raise AssertionError(f"{year} 三个 seed 的预测索引不一致")  # 关键数据契约被破坏时立即终止。
        raw_column = f"prediction_seed_{seed}_unprojected"
        effective_column = f"prediction_seed_{seed}"
        output[raw_column] = frame[
            "prediction_model_raw"
        ].to_numpy(np.float32)
        output[effective_column] = frame["prediction"].to_numpy(np.float32)  # 转换并统一输出的数据表示。
        raw_columns.append(raw_column)  # 把当前结果追加到汇总列表。
        effective_columns.append(effective_column)  # 把当前结果追加到汇总列表。
        if dual_tower_enabled():  # 根据当前数据状态选择处理分支。
            style_column = f"prediction_seed_{seed}_style"
            total_column = f"prediction_seed_{seed}_total_return"
            style_values = frame[
                "prediction_style_component"
            ].to_numpy(np.float32)
            stored_total_values = frame[
                "prediction_total_return"
            ].to_numpy(np.float32)
            total_values, dual_inference_audits[seed] = (  # 通过当前网络子模块。
                canonical_dual_tower_total_scores(
                    output[raw_column].to_numpy(np.float32),
                    style_values,
                    stored_total_values,
                )
            )
            output[style_column] = style_values
            output[total_column] = total_values
            dual_style_columns.append(style_column)  # 把当前结果追加到汇总列表。
            dual_total_columns.append(total_column)  # 把当前结果追加到汇总列表。

    metadata = pd.read_parquet(  # 读取元数据。
        sequence_data.SAMPLE_INDEX_PATH,
        columns=["sample_index", "trade_date", "instrument_id"],  # 限定本次读取或输出使用的字段。
    ).set_index("sample_index")
    output = metadata.loc[reference].reset_index().merge(  # 按业务键对齐，避免依赖原始行序。
        output, on="sample_index", validate="one_to_one"
    )

    def daily_zscore(column: str) -> str:  # 按交易日用 Alpha 分数尺度标准化双塔分量，保持加法恒等式。
        """按交易日用 Alpha 分数尺度标准化双塔分量，保持加法恒等式。"""
        grouped = output[column].astype(np.float64).groupby(  # 按业务键分组得到grouped。
            output["trade_date"], sort=False
        )
        mean = grouped.transform("mean")  # 按约定截面口径汇总。
        std = grouped.transform("std", ddof=0)  # 按约定截面口径汇总。
        if (std <= 0).any() or not np.isfinite(std).all():  # 检查参与计算的数值是否全部有限。
            raise ValueError(f"{year} {column} 存在常数或非有限日截面")  # 输入数据不满足业务或数值约束时立即终止。
        result_column = f"{column}_zscore"
        output[result_column] = (
            (output[column] - mean) / std
        ).astype(np.float32)
        return result_column  # 返回结果列。

    if dual_tower_enabled():  # 根据当前数据状态选择处理分支。
        alpha_scaled: list[str] = []  # 接收Alphascaled。
        style_scaled: list[str] = []  # 接收风格scaled。
        total_scaled: list[str] = []  # 接收totalscaled。
        for seed, alpha_column, style_column in zip(  # 遍历zip，逐项处理随机种子Alpha列风格列。
            SEEDS, raw_columns, dual_style_columns  # 传入随机种子、原始列、dual风格列。
        ):
            alpha_grouped = output[alpha_column].astype(np.float64).groupby(  # 按业务键分组得到Alphagrouped。
                output["trade_date"], sort=False
            )
            alpha_mean = alpha_grouped.transform("mean")  # 按约定截面口径汇总。
            alpha_std = alpha_grouped.transform("std", ddof=0)  # 按约定截面口径汇总。
            if (alpha_std <= 0).any() or not np.isfinite(alpha_std).all():  # 检查参与计算的数值是否全部有限。
                raise ValueError(f"{year}/seed={seed} Alpha 日截面为常数")  # 输入数据不满足业务或数值约束时立即终止。
            style_mean = output[style_column].astype(np.float64).groupby(  # 按业务键分组得到风格均值。
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
            alpha_scaled.append(alpha_scaled_column)  # 把当前结果追加到汇总列表。
            style_scaled.append(style_scaled_column)  # 把当前结果追加到汇总列表。
            total_scaled.append(total_scaled_column)  # 把当前结果追加到汇总列表。
        final_alpha = output[alpha_scaled].mean(axis=1).to_numpy(np.float64)  # 转换并统一finalAlpha的数据表示。
        final_style = output[style_scaled].mean(axis=1).to_numpy(np.float64)  # 转换并统一final风格的数据表示。
        final_total = (
            final_alpha.astype(np.float32) + final_style.astype(np.float32)
        ).astype(np.float32).astype(np.float64)
        final_raw = final_alpha
        final_effective = final_alpha
    else:  # 处理上述条件均不成立的情况。
        raw_zscore_columns = [daily_zscore(column) for column in raw_columns]  # 计算原始zscore列。
        effective_zscore_columns = [  # 计算有效zscore列。
            daily_zscore(column) for column in effective_columns
        ]
        final_raw = output[raw_zscore_columns].mean(axis=1).to_numpy(np.float64)  # 转换并统一final原始的数据表示。
        final_effective = (
            output[effective_zscore_columns].mean(axis=1).to_numpy(np.float64)
        )

    style_matrix = np.load(  # 读取风格矩阵。
        style_data.DEFAULT_STORE_DIR / style_data.STYLE_EXPOSURES_FILE,
        mmap_mode="r",  # 使用内存映射避免一次性载入全部数据。
    )
    style_valid = np.load(  # 读取风格有效。
        style_data.DEFAULT_STORE_DIR / style_data.STYLE_VALID_FILE,
        mmap_mode="r",  # 使用内存映射避免一次性载入全部数据。
    )
    industry_codes = None  # 先把行业代码初始化为空，后续按条件填充。
    if industry_sidecar_required():  # 根据当前数据状态选择处理分支。
        style_data.require_industry_sidecar()
        industry_codes = np.load(  # 读取行业代码。
            style_data.DEFAULT_INDUSTRY_STORE_DIR
            / style_data.INDUSTRY_CODES_FILE,
            mmap_mode="r",  # 使用内存映射避免一次性载入全部数据。
        )
    dual_source_targets = None  # 先把dual来源标签初始化为空，后续按条件填充。
    dual_residual_targets = None  # 先把dual残差标签初始化为空，后续按条件填充。
    dual_fitted_targets = None  # 先把dual拟合标签初始化为空，后续按条件填充。
    if dual_tower_enabled():  # 根据当前数据状态选择处理分支。
        dual_source_targets = np.load(sequence_data.TARGETS_PATH, mmap_mode="r")  # 读取dual来源标签。
        dual_residual_targets = np.load(  # 读取dual残差标签。
            style_data.DEFAULT_FULL_BARRA_TARGET_STORE_DIR
            / style_data.FULL_BARRA_TARGETS_FILE,
            mmap_mode="r",  # 使用内存映射避免一次性载入全部数据。
        )
        dual_fitted_targets = np.load(  # 读取dual拟合标签。
            style_data.DEFAULT_DUAL_TOWER_TARGET_STORE_DIR
            / style_data.DUAL_TOWER_FITTED_TARGETS_FILE,
            mmap_mode="r",  # 使用内存映射避免一次性载入全部数据。
        )
    records: list[dict[str, Any]] = []  # 初始化records容器。
    for trade_date, positions in output.groupby(  # 遍历输出groupby，逐项处理trade交易日位置。
        "trade_date", sort=False
    ).groups.items():
        locations = np.asarray(positions, dtype=np.int64)  # 转换并统一locations的数据表示。
        indices = output.iloc[locations]["sample_index"].to_numpy(np.int64)  # 转换并统一索引的数据表示。
        daily_styles = np.asarray(style_matrix[indices], dtype=np.float64)  # 转换并统一daily风格的数据表示。
        style_ranks = pd.DataFrame(daily_styles).rank(  # 计算风格排名。
            method="average"
        ).to_numpy(np.float64)
        record: dict[str, Any] = {  # 初始化record容器。
            "year": year,  # 记录年份。
            "trade_date": str(trade_date),  # 记录trade交易日。
            "rows": len(locations),  # 记录行数。
            "complete_style_rate": float(  # 记录complete风格比例。
                np.asarray(style_valid[indices]).all(axis=1).mean()
            ),
        }
        factors = [  # 初始化factors容器。
            ("raw", final_raw[locations]),
            ("effective", final_effective[locations]),
        ]
        if dual_tower_enabled():  # 根据当前数据状态选择处理分支。
            factors = [  # 初始化factors容器。
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
            record["inference_total_precision"] = ",".join(  # 按业务键对齐，避免依赖原始行序。
                sorted({
                    str(audit["stored_total_precision"])
                    for audit in dual_inference_audits.values()  # 遍历dualinferenceaudits值，逐项处理audit。
                })
            )
            record["inference_stored_vs_canonical_max_abs_error"] = max(
                float(audit["stored_vs_canonical_max_abs_error"])
                for audit in dual_inference_audits.values()  # 遍历dualinferenceaudits值，逐项处理audit。
            )
        target_views: dict[str, np.ndarray] = {}  # 初始化标签views容器。
        if dual_tower_enabled():  # 根据当前数据状态选择处理分支。
            target_views = {  # 初始化标签views容器。
                "original": np.asarray(  # 记录原始。
                    dual_source_targets[indices], dtype=np.float64
                ),
                "residual": np.asarray(  # 记录残差。
                    dual_residual_targets[indices], dtype=np.float64
                ),
                "fitted": np.asarray(  # 记录拟合。
                    dual_fitted_targets[indices], dtype=np.float64
                ),
            }
        for prefix, daily_factor in factors:  # 遍历factors，逐项处理prefixdaily因子。
            factor_std = float(np.std(daily_factor, ddof=0))  # 计算因子标准差。
            record[f"{prefix}_cross_section_std"] = factor_std  # 计算record。
            if factor_std <= np.finfo(np.float64).eps:  # 检查当前数值是否越过业务或数值阈值。
                pearson = np.full(STYLE_COUNT, np.nan, dtype=np.float64)
                spearman = np.full(STYLE_COUNT, np.nan, dtype=np.float64)
                factor_rank = np.full(len(daily_factor), np.nan, dtype=np.float64)  # 计算因子排名。
            else:  # 处理上述条件均不成立的情况。
                factor_rank = pd.Series(daily_factor).rank(  # 计算因子排名。
                    method="average"
                ).to_numpy(np.float64)
                pearson = np.corrcoef(
                    np.column_stack([daily_factor, daily_styles]), rowvar=False
                )[0, 1:]
                spearman = np.corrcoef(
                    np.column_stack([factor_rank, style_ranks]), rowvar=False
                )[0, 1:]
                if (  # 开始组合多项有效性或配置条件。
                    not np.isfinite(pearson).all()
                    or not np.isfinite(spearman).all()
                ):
                    raise FloatingPointError(  # 检测到 NaN、Inf 或退化数值时立即终止。
                        f"{trade_date} {prefix} 预测风格相关性不可计算"
                    )
            if dual_tower_enabled():  # 根据当前数据状态选择处理分支。
                for target_name, daily_target in target_views.items():  # 遍历标签viewsitems，逐项处理标签名称daily标签。
                    valid_target = np.isfinite(daily_target)  # 排除 NaN 与 Inf。
                    if (  # 开始组合多项有效性或配置条件。
                        factor_std <= np.finfo(np.float64).eps
                        or int(valid_target.sum()) < 2
                        or float(np.std(daily_target[valid_target], ddof=0))
                        <= np.finfo(np.float64).eps
                    ):
                        pearson_ic = np.nan
                        rank_ic = np.nan  # 计算排名ic。
                    else:  # 处理上述条件均不成立的情况。
                        pearson_ic = float(np.corrcoef(
                            daily_factor[valid_target],
                            daily_target[valid_target],
                        )[0, 1])
                        rank_ic = float(np.corrcoef(  # 计算排名ic。
                            pd.Series(daily_factor[valid_target]).rank(
                                method="average"
                            ),
                            pd.Series(daily_target[valid_target]).rank(
                                method="average"
                            ),
                        )[0, 1])
                    record[f"{prefix}_{target_name}_pearson_ic"] = pearson_ic
                    record[f"{prefix}_{target_name}_rank_ic"] = rank_ic  # 计算record。
            record.update(  # 用当前配置更新参数字典。
                {
                    f"{prefix}_max_abs_style_pearson": float(np.nanmax(np.abs(pearson))) if np.isfinite(pearson).any() else np.nan,
                    f"{prefix}_mean_abs_style_pearson": float(np.nanmean(np.abs(pearson))) if np.isfinite(pearson).any() else np.nan,
                    f"{prefix}_max_abs_style_spearman": float(np.nanmax(np.abs(spearman))) if np.isfinite(spearman).any() else np.nan,
                    f"{prefix}_mean_abs_style_spearman": float(np.nanmean(np.abs(spearman))) if np.isfinite(spearman).any() else np.nan,
                }
            )
            if industry_codes is not None:  # 检查必需数据、配置或校验结果是否缺失。
                daily_industries = np.asarray(  # 转换并统一daily行业的数据表示。
                    industry_codes[indices], dtype=np.int16
                )
                if factor_std > np.finfo(np.float64).eps:  # 检查当前数值是否越过业务或数值阈值。
                    joint_r_squared, _ = style_data.joint_barra_r_squared_numpy(  # 按股票与交易日业务键对齐。
                        daily_factor, daily_styles, daily_industries  # 传入daily因子、daily风格、daily行业。
                    )
                else:  # 处理上述条件均不成立的情况。
                    joint_r_squared = np.nan  # 计算jointrsquared。
                industry_means = pd.Series(daily_factor).groupby(  # 按业务键分组得到行业means。
                    daily_industries, sort=False
                ).mean()
                record[f"{prefix}_joint_barra_r_squared"] = joint_r_squared  # 计算record。
                record[f"{prefix}_max_abs_industry_mean"] = float(  # 计算record。
                    industry_means.abs().max()
                )
            for position, column in enumerate(style_data.STYLE_COLUMNS):  # 遍历enumerate风格数据风格列，逐项处理位置列。
                record[f"{prefix}_pearson_{column}"] = float(
                    pearson[position]
                )
                record[f"{prefix}_spearman_{column}"] = float(
                    spearman[position]
                )
        records.append(record)  # 把当前结果追加到汇总列表。

    output["prediction_ensemble_unprojected"] = final_raw.astype(np.float32)  # 转换输出的数据类型或存储位置。
    if dual_tower_enabled():  # 根据当前数据状态选择处理分支。
        output["prediction_ensemble_alpha"] = final_alpha.astype(np.float32)  # 转换输出的数据类型或存储位置。
        output["prediction_ensemble_style"] = final_style.astype(np.float32)  # 转换输出的数据类型或存储位置。
        output["prediction_ensemble_total_return"] = final_total.astype(  # 转换输出的数据类型或存储位置。
            np.float32
        )
    if PROJECT_MODEL_OUTPUT or strict_barra_output_enabled():  # 根据当前数据状态选择处理分支。
        output["prediction_ensemble_projected"] = final_effective.astype(  # 转换输出的数据类型或存储位置。
            np.float32
        )
    published = (
        final_effective if strict_barra_output_enabled() else final_raw
    ).astype(np.float32)
    output["prediction_ensemble_raw"] = published
    output["prediction_ensemble"] = published
    output["forecast_year"] = year
    output["factor_role"] = factor_role()  # 确定实验输出因子口径。
    sequence_data.atomic_write_parquet(  # 原子落盘，避免半成品。
        output,
        prediction_root / f"vivit_predictions.{year}.parquet",
    )
    return len(output), pd.DataFrame(records)  # 返回按当前规则组织后的结果。


def style_diagnostics_root(quality_root: Path) -> Path:  # 返回当前 run 的风格暴露诊断目录。
    """返回当前 run 的风格暴露诊断目录。"""
    return quality_root / "style_neutral_training"  # 返回当前 run 的风格暴露诊断目录。


def finalize_run(wall_seconds: float) -> pd.DataFrame:  # 核验年度任务并生成集成预测、诊断、清单和报告。
    """核验年度任务并生成集成预测、诊断、清单和报告。"""
    model_root, _, quality_root = base.run_paths(RUN_NAME)  # 构造模型根目录。
    summaries: list[dict[str, Any]] = []  # 初始化summaries容器。
    for year in FORECAST_YEARS:  # 遍历forecast年份，逐项处理年份。
        for seed in SEEDS:  # 遍历随机种子，逐项处理随机种子。
            path = model_root / f"forecast_{year}" / f"seed_{seed}" / "completed.json"
            if not path.exists():  # 检查目标文件或目录是否满足存在性要求。
                raise FileNotFoundError(path)  # 目标文件不存在时立即终止。
            summaries.append(json.loads(path.read_text(encoding="utf-8"))["summary"])  # 把当前结果追加到汇总列表。
    summary = pd.DataFrame(summaries).sort_values(  # 计算汇总以固定样本顺序。
        ["forecast_year", "seed"], kind="stable"
    )
    validation_score_collapse_tasks = (  # 计算验证分数collapsetasks。
        int(summary["validation_score_collapsed"].fillna(False).sum())
        if "validation_score_collapsed" in summary.columns  # 检查当前值是否属于允许集合。
        else 0
    )
    sequence_data.atomic_write_parquet(  # 原子落盘，避免半成品。
        summary, quality_root / "vivit_training_summary.parquet"
    )
    if adversarial_barra_enabled():  # 根据当前数据状态选择处理分支。
        probe_frames: list[pd.DataFrame] = []  # 接收probe帧。
        for year in FORECAST_YEARS:  # 遍历forecast年份，逐项处理年份。
            for seed in SEEDS:  # 遍历随机种子，逐项处理随机种子。
                probe_path = (  # 构造probe路径。
                    model_root / f"forecast_{year}" / f"seed_{seed}"
                    / "adversarial_probe_diagnostics.parquet"
                )
                if not probe_path.exists():  # 检查目标文件或目录是否满足存在性要求。
                    raise FileNotFoundError(probe_path)  # 目标文件不存在时立即终止。
                probe_frames.append(pd.read_parquet(probe_path))  # 把当前结果追加到汇总列表。
        sequence_data.atomic_write_parquet(  # 原子落盘，避免半成品。
            pd.concat(probe_frames, ignore_index=True),
            quality_root / "adversarial_probe_diagnostics.parquet",
        )

    factor_counts: dict[int, int] = {}  # 计算因子数量。
    diagnostics: list[pd.DataFrame] = []  # 接收诊断。
    for year in FORECAST_YEARS:  # 遍历forecast年份，逐项处理年份。
        factor_counts[year], yearly = build_ensemble(year)  # 计算因子数量。
        if not yearly.empty:  # 检查结果表是否为空。
            diagnostics.append(yearly)  # 把当前结果追加到汇总列表。
    if not diagnostics:  # 检查必需数据、配置或校验结果是否缺失。
        raise ValueError("没有可写入的样本外预测风格诊断")  # 输入数据不满足业务或数值约束时立即终止。
    sequence_data.atomic_write_parquet(  # 原子落盘，避免半成品。
        pd.concat(diagnostics, ignore_index=True),
        style_diagnostics_root(quality_root)
        / "prediction_style_exposure_diagnostics.parquet",
    )
    if dual_tower_enabled():  # 根据当前数据状态选择处理分支。
        sequence_data.atomic_write_parquet(  # 原子落盘，避免半成品。
            pd.concat(diagnostics, ignore_index=True),
            quality_root / "dual_tower_decomposition_diagnostics.parquet",
        )

    sequence = sequence_data.require_aligned_sequence_store()
    style_manifest = style_data.require_style_neutral_store()
    industry_manifest = (
        style_data.require_industry_sidecar()
        if industry_sidecar_required() else None  # 根据当前数据状态选择处理分支。
    )
    full_target_manifest = (
        style_data.require_full_barra_target_sidecar()
        if barra_conditioning_enabled() or dual_tower_enabled() else None  # 根据当前数据状态选择处理分支。
    )
    dual_target_manifest = (
        style_data.require_dual_tower_target_sidecar()
        if dual_tower_enabled() else None  # 根据当前数据状态选择处理分支。
    )
    manifest = {  # 初始化清单容器。
        "schema_version": (  # 记录schema版本。
            7 if dual_tower_enabled()
            else 6 if strict_barra_output_enabled()
            else 5 if barra_conditioning_enabled() else TRAINING_SCHEMA_VERSION
        ),
        "route": (  # 记录route。
            "style_neutral_ddp" if EXECUTION_ROUTE == "ddp"
            else "style_neutral_independent_tasks"
        ),
        "execution_route": EXECUTION_ROUTE,  # 记录执行route。
        "world_size": DDP_WORLD_SIZE,  # 记录world大小。
        "physical_device_ids": list(EXECUTION_DEVICE_IDS),  # 记录physical设备ids。
        "tasks_per_gpu": TASKS_PER_GPU if EXECUTION_ROUTE == "a2" else None,  # 记录taskspergpu。
        "run_name": RUN_NAME,  # 记录运行名称。
        "factor_role": factor_role(),  # 记录因子role。
        "sequence_signature": sequence["signature"],  # 记录sequence签名，用于来源一致性校验。
        "source_dataset_signature": sequence["payload"]["source_dataset_signature"],  # 记录来源dataset签名，用于来源一致性校验。
        "source_rolling_signature": sequence["payload"]["source_rolling_signature"],  # 记录来源rolling签名，用于来源一致性校验。
        "style_store_signature": style_manifest["signature"],  # 记录风格存储签名，用于来源一致性校验。
        "industry_sidecar_signature": (  # 记录行业sidecar签名，用于来源一致性校验。
            industry_manifest["signature"]
            if industry_manifest is not None else None  # 检查必需数据、配置或校验结果是否缺失。
        ),
        "style_columns": list(style_data.STYLE_COLUMNS),  # 记录风格字段列表。
        "industry_used": industry_sidecar_required(),  # 记录行业使用状态。
        "residualize_target": RESIDUALIZE_TARGET,  # 记录残差化标签。
        "project_model_output": PROJECT_MODEL_OUTPUT,  # 记录投影模型输出。
        "target_policy": target_policy(),  # 声明标签处理口径。
        "output_policy": (  # 声明输出处理口径。
            output_policy()
            if strict_barra_output_enabled() or dual_tower_enabled()  # 根据当前数据状态选择处理分支。
            else PUBLISHED_FACTOR_POLICY
        ),
        "validation_score_collapse_tasks": validation_score_collapse_tasks,  # 记录验证分数collapsetasks。
        "training_effective_output_policy": output_policy(),  # 声明训练有效输出处理口径。
        "published_factor_column": "prediction_ensemble",  # 把 published_factor_column 映射为"prediction_ensemble"。
        "published_factor_is_projected": strict_barra_output_enabled(),  # 记录published因子isprojected。
        "projected_audit_column": (  # 记录projectedaudit字段。
            "prediction_ensemble_projected"
            if PROJECT_MODEL_OUTPUT or strict_barra_output_enabled()  # 根据当前数据状态选择处理分支。
            else None
        ),
        "output_projection": (  # 记录输出projection。
            {
                "method": "daily_equal_weight_full_barra_projection",  # 把 method 映射为"daily_equal_weight_full_barra_projection"。
                "design": [  # 记录design。
                    "intercept",
                    *style_data.STYLE_COLUMNS,
                    "industry_fixed_effects",
                ],
                "solver": "industry_within_then_torch_fp32_reduced_qr",  # 把 solver 映射为"industry_within_then_torch_fp32_reduced_qr"。
                "post_projection": "zscore_ddof0",  # 把 post_projection 映射为"zscore_ddof0"。
                "applied_during": [  # 记录appliedduring。
                    "training_loss",
                    "validation",
                    "inference_and_publication",
                ],
            }
            if strict_barra_output_enabled()  # 根据当前数据状态选择处理分支。
            else {
                "method": "daily_qr_orthogonal_complement",  # 把 method 映射为"daily_qr_orthogonal_complement"。
                "design": ["intercept", *style_data.STYLE_COLUMNS],  # 记录design。
                "post_projection": "zscore_ddof0",  # 把 post_projection 映射为"zscore_ddof0"。
                "applied_during": [  # 记录appliedduring。
                    "training_loss",
                    "validation",
                    "inference_auxiliary_audit",
                ],
            }
            if PROJECT_MODEL_OUTPUT  # 根据当前数据状态选择处理分支。
            else None
        ),
        "style_correlation_penalty": raw_barra_r2_enabled(),  # 记录风格相关系数penalty。
        "raw_barra_r2_lambda": RAW_BARRA_R2_LAMBDA,  # 记录原始Barrar2lambda。
        "adversarial_barra": adversarial_barra_enabled(),  # 记录adversarialBarra。
        "barra_is_model_input": (  # 记录Barrais模型输入。
            barra_conditioning_enabled() or dual_tower_enabled()
        ),
        "barra_participates_in_training_loss": (  # 记录Barraparticipatesin训练损失。
            RESIDUALIZE_TARGET
            or PROJECT_MODEL_OUTPUT
            or raw_barra_r2_enabled()
            or adversarial_barra_enabled()
            or barra_conditioning_enabled()
            or dual_tower_enabled()
        ),
        "adversarial_config": (  # 记录adversarial配置。
            {
                "gamma_max": ADVERSARIAL_GAMMA_MAX,  # 记录对抗系数max。
                "warmup_epochs": ADVERSARIAL_WARMUP_EPOCHS,  # 记录warmupepochs。
                "cosine_ramp_epochs": ADVERSARIAL_RAMP_EPOCHS,  # 记录cosinerampepochs。
                "style_loss_weight": ADVERSARIAL_STYLE_LOSS_WEIGHT,  # 记录风格损失权重。
                "industry_loss_weight": ADVERSARIAL_INDUSTRY_LOSS_WEIGHT,  # 记录行业损失权重。
                "head_hidden_dim": ADVERSARIAL_HEAD_HIDDEN_DIM,  # 记录head隐藏层dim。
                "head_dropout": ADVERSARIAL_HEAD_DROPOUT,  # 记录headdropout。
                "industry_classes": INDUSTRY_COUNT,  # 记录行业classes。
                "unknown_industry_policy": "masked_from_ce",  # 把 unknown_industry_policy 映射为"masked_from_ce"。
                "inference_model": "backbone_and_return_head_only",  # 把 inference_model 映射为"backbone_and_return_head_only"。
                "probe_file": "adversarial_probe_diagnostics.parquet",  # 把 probe_file 映射为"adversarial_probe_diagnostics.parquet"。
                "probe_sample_cap_per_date": PROBE_MAX_SAMPLES_PER_DATE,  # 记录probe样本capper交易日。
                "probe_date_split": [0.4, 0.1, 0.5],  # 记录probe交易日切分。
                "probe_batch_size": PROBE_BATCH_SIZE,  # 记录probe批次大小。
                "probe_learning_rate": PROBE_LEARNING_RATE,  # 记录probelearning比例。
                "probe_weight_decay": PROBE_WEIGHT_DECAY,  # 记录probe权重decay。
                "probe_max_epochs": PROBE_MAX_EPOCHS,  # 记录probemaxepochs。
                "probe_patience": PROBE_PATIENCE,  # 记录probepatience。
                "probe_ridge_alpha": PROBE_RIDGE_ALPHA,  # 记录proberidgeAlpha。
                "probe_mlp_hidden_dim": 128,  # 记录probemlp隐藏层dim。
                "probe_mlp_dropout": 0.1,  # 记录probemlpdropout。
            }
            if adversarial_barra_enabled() else None  # 根据当前数据状态选择处理分支。
        ),
        "loss_name": LOSS_NAME,  # 记录损失名称。
        "raw_barra_r2_design": (  # 记录原始Barrar2design。
            {
                "dependent_variable": "raw_model_score",  # 把 dependent_variable 映射为"raw_model_score"。
                "continuous": list(style_data.STYLE_COLUMNS),  # 记录continuous。
                "industry_fixed_effects": True,  # 记录行业fixedeffects。
                "solver": "industry_within_then_torch_fp32_reduced_qr",  # 把 solver 映射为"industry_within_then_torch_fp32_reduced_qr"。
                "missing": "style_zero_industry_unknown_no_forward_fill",  # 把 missing 映射为"style_zero_industry_unknown_no_forward_fill"。
            }
            if (  # 开始组合多项有效性或配置条件。
                raw_barra_r2_enabled()
                or barra_conditioning_enabled()
                or dual_tower_enabled()
            )
            else None
        ),
        "inference_requires_barra": (  # 记录inferencerequiresBarra。
            dual_tower_enabled()
            or strict_barra_output_enabled()
            or BARRA_CONDITIONING_MODE in {"real", "shuffled"}
        ),
        "loss_parameters": {  # 记录损失parameters。
            "rank_temperature": RANK_TEMPERATURE,  # 记录秩temperature。
            "turnover_lambda": TURNOVER_LAMBDA,  # 记录turnoverlambda。
            "huber_delta": HUBER_DELTA,  # 记录huberdelta。
            "smooth_l1_beta": SMOOTH_L1_BETA,  # 记录smoothl1beta。
            "hybrid_weight": HYBRID_WEIGHT,  # 记录hybrid权重。
        },
        "validation_metric": VALIDATION_METRIC,  # 记录验证指标。
        "validation_target": (  # 记录验证标签。
            "alpha_score_vs_full_barra_residual_target"
            if dual_tower_enabled()  # 根据当前数据状态选择处理分支。
            else "strict_projected_score_vs_full_barra_residual_target"
            if strict_barra_output_enabled()  # 根据当前数据状态选择处理分支。
            else
            "raw_score_vs_full_barra_residual_target"
            if barra_conditioning_enabled()  # 根据当前数据状态选择处理分支。
            else "raw_score_vs_source_target"
            if raw_barra_r2_enabled() or adversarial_barra_enabled()  # 根据当前数据状态选择处理分支。
            else "effective_prediction_vs_residual_target"
            if RESIDUALIZE_TARGET  # 根据当前数据状态选择处理分支。
            else "effective_prediction_vs_source_target"
        ),
        "train_batch_mode": TRAIN_BATCH_MODE,  # 记录训练批次模式。
        "workers_per_task": WORKERS_PER_TASK,  # 记录workerspertask。
        "learning_rate": LEARNING_RATE,  # 记录learning比例。
        "max_epochs": MAX_EPOCHS,  # 记录maxepochs。
        "patience": PATIENCE,  # 记录patience。
        "torch_compile": bool(USE_TORCH_COMPILE),  # 记录torchcompile。
        "compile_mode": COMPILE_MODE,  # 记录compile模式。
        "fused_adam": bool(USE_FUSED_ADAM),  # 记录fusedadam。
        "cuda_prefetch": bool(USE_CUDA_PREFETCH),  # 记录cudaprefetch。
        "years": list(FORECAST_YEARS),  # 记录年份。
        "seeds": list(SEEDS),  # 记录随机种子。
        "completed_tasks": len(summary),  # 记录completedtasks。
        "factor_counts": factor_counts,  # 记录因子数量。
        "split_policy": "exact_vit_train_validation_prediction_indices",  # 把 split_policy 映射为"exact_vit_train_validation_prediction_indices"。
        "wall_seconds": wall_seconds,  # 记录wallseconds。
    }
    if dual_tower_enabled():  # 根据当前数据状态选择处理分支。
        manifest["full_barra_target_sidecar_signature"] = (
            full_target_manifest["signature"]
        )
        manifest["dual_tower_target_sidecar_signature"] = (
            dual_target_manifest["signature"]
        )
        manifest["barra_dual_tower"] = {  # 初始化清单容器。
            "mode": BARRA_DUAL_TOWER_MODE,  # 记录模式。
            "alpha_loss_weight": DUAL_TOWER_ALPHA_LOSS_WEIGHT,  # 记录Alpha损失权重。
            "style_loss_weight": DUAL_TOWER_STYLE_LOSS_WEIGHT,  # 记录风格损失权重。
            "style_hidden_dim": DUAL_TOWER_STYLE_HIDDEN_DIM,  # 记录风格隐藏层dim。
            "style_dropout": DUAL_TOWER_STYLE_DROPOUT,  # 记录风格dropout。
            "industry_embedding_dim": BARRA_INDUSTRY_EMBEDDING_DIM,  # 记录行业嵌入dim。
            "alpha_r2_lambda": RAW_BARRA_R2_LAMBDA,  # 记录Alphar2lambda。
            "style_output_initialization": "zeros",  # 把 style_output_initialization 映射为"zeros"。
            "published_factor": "prediction_ensemble_alpha",  # 把 published_factor 映射为"prediction_ensemble_alpha"。
            "component_columns": [  # 记录component字段列表。
                "prediction_ensemble_alpha",
                "prediction_ensemble_style",
                "prediction_ensemble_total_return",
            ],
            "ensemble_scaling": "center_components_divide_by_alpha_std",  # 把 ensemble_scaling 映射为"center_components_divide_by_alpha_std"。
            "seed_total_audit_policy": (  # 声明随机种子totalaudit处理口径。
                "stored_total_must_match_float32_or_bfloat16_alpha_plus_style"
            ),
            "ensemble_total_policy": (  # 声明ensembletotal处理口径。
                "reconstruct_float32_total_from_stored_alpha_plus_style"
            ),
            "r2_uses_true_unmodified_barra": True,  # 记录r2usestrueunmodifiedBarra。
        }
    if barra_conditioning_enabled():  # 根据当前数据状态选择处理分支。
        manifest["full_barra_target_sidecar_signature"] = (
            full_target_manifest["signature"]
        )
        manifest["barra_conditioning"] = {  # 初始化清单容器。
            "mode": BARRA_CONDITIONING_MODE,  # 记录模式。
            "hidden_dim": BARRA_GATE_HIDDEN_DIM,  # 记录隐藏层dim。
            "industry_embedding_dim": BARRA_INDUSTRY_EMBEDDING_DIM,  # 记录行业嵌入dim。
            "dropout": BARRA_GATE_DROPOUT,  # 记录dropout。
            "gate_scale": BARRA_GATE_SCALE,  # 记录门控scale。
            "gate_projection_initialization": "zeros",  # 把 gate_projection_initialization 映射为"zeros"。
            "multiplicative_only": True,  # 记录multiplicativeonly。
            "shuffle_policy": (  # 声明shuffle处理口径。
                "within_date_joint_permutation_by_stable_sample_key_v1"
                if BARRA_CONDITIONING_MODE == "shuffled" else None  # 按当前模式、标识或状态选择处理分支。
            ),
            "r2_uses_true_unmodified_barra": True,  # 记录r2usestrueunmodifiedBarra。
        }
    if strict_barra_output_enabled():  # 根据当前数据状态选择处理分支。
        manifest["strict_barra_output"] = {  # 初始化清单容器。
            "weighting": "equal_weight_W_identity",  # 把 weighting 映射为"equal_weight_W_identity"。
            "design": [  # 记录design。
                "intercept",
                *style_data.STYLE_COLUMNS,
                "industry_fixed_effects",
            ],
            "solver": "industry_within_then_torch_fp32_reduced_qr",  # 把 solver 映射为"industry_within_then_torch_fp32_reduced_qr"。
            "post_projection": "zscore_ddof0",  # 把 post_projection 映射为"zscore_ddof0"。
            "applied_during": ["training_loss", "validation", "inference"],  # 记录appliedduring。
            "published_factor": "prediction_ensemble",  # 把 published_factor 映射为"prediction_ensemble"。
            "pre_neutral_audit_column": "prediction_ensemble_unprojected",  # 把 pre_neutral_audit_column 映射为"prediction_ensemble_unprojected"。
            "orthogonality_requirement": "H_transpose_W_s_equals_zero",  # 把 orthogonality_requirement 映射为"H_transpose_W_s_equals_zero"。
        }
    sequence_data.atomic_write_json(manifest, quality_root / "a2_training_manifest.json")  # 原子落盘，避免半成品。
    return summary  # 返回汇总。


# %% [markdown]
# ## 四 GPU 调度和 CLI 入口


# %%


def measure_daily_score_dispersion(  # 检查单日原始分数是否仍有可用于排序的截面离散度。
    predictions: torch.Tensor,  # [B]，当前 batch 分数。
    *,
    date_id: int,  # 接收交易日id。
) -> dict[str, torch.Tensor]:  # 声明当前函数的返回类型。
    """检查单日原始分数是否仍有可用于排序的截面离散度。"""

    raw = predictions.detach().float().reshape(-1)  # [B,...] -> [B]。
    if len(raw) < 2:  # 检查当前样本数量是否满足计算要求。
        raise ValueError("单日预测至少需要两个样本")  # 输入数据不满足业务或数值约束时立即终止。
    if not torch.isfinite(raw).all():  # 检查参与计算的数值是否全部有限。
        raise FloatingPointError("模型输出包含非有限值")  # 检测到 NaN、Inf 或退化数值时立即终止。

    centered = raw - raw.mean()  # 计算中心化。
    total_sum_squares = centered.square().sum()  # 计算totalsumsquares。
    score_std = torch.sqrt(total_sum_squares / len(raw))  # 计算截面标准差。
    if (  # 开始组合多项有效性或配置条件。
        not torch.isfinite(total_sum_squares)
        or not torch.isfinite(score_std)
        or total_sum_squares <= torch.finfo(torch.float32).eps
    ):
        raise DegenerateValidationScoreError(  # 验证分数截面退化时停止当前任务。
            date_id=date_id,  # 配置交易日id。
            rows=len(raw),  # 配置行数。
            score_std=float(score_std.item()),  # 配置分数标准差。
            total_sum_squares=float(total_sum_squares.item()),  # 配置totalsumsquares。
        )
    return {  # 返回检查单日原始分数是否仍有可用于排序的截面离散度。
        "score_std": score_std,  # 记录分数标准差。
        "total_sum_squares": total_sum_squares,  # 记录totalsumsquares。
    }

def measure_style_correlations(  # 只读衡量单日预测与十个风格的 Pearson 相关性。
    predictions: torch.Tensor,  # [B]，当前 batch 分数。
    style_exposures: torch.Tensor,  # [B,10]，十个连续 Barra 暴露。
) -> dict[str, torch.Tensor]:  # 声明当前函数的返回类型。
    """只读衡量单日预测与十个风格的 Pearson 相关性。"""

    raw = predictions.detach().float().reshape(-1)  # [B,...] -> [B]。
    styles = style_exposures.float()  # 转换风格的数据类型或存储位置。
    if styles.shape != (len(raw), STYLE_COUNT):  # 检查张量或数组形状是否符合接口约定。
        raise ValueError(f"十风格矩阵形状异常：{tuple(styles.shape)}")  # 输入数据不满足业务或数值约束时立即终止。
    if len(raw) < 3:  # 检查当前样本数量是否满足计算要求。
        raise ValueError("单日股票数不足以计算风格相关性")  # 输入数据不满足业务或数值约束时立即终止。
    if not torch.isfinite(raw).all() or not torch.isfinite(styles).all():  # 检查参与计算的数值是否全部有限。
        raise FloatingPointError("预测或十风格暴露含 NaN/Inf")  # 检测到 NaN、Inf 或退化数值时立即终止。

    centered_factor = raw - raw.mean()  # 计算中心化因子。
    centered_styles = styles - styles.mean(dim=0, keepdim=True)  # 计算中心化风格。
    numerator = centered_styles.transpose(0, 1) @ centered_factor  # [10,B]@[B] -> [10]。
    denominator = torch.sqrt(  # 计算截面标准差。
        centered_styles.square().sum(dim=0)
        * centered_factor.square().sum()
    )
    correlations = numerator / denominator  # 计算correlations。
    if not torch.isfinite(correlations).all():  # 检查参与计算的数值是否全部有限。
        raise FloatingPointError("模型输出的风格相关性不可计算")  # 检测到 NaN、Inf 或退化数值时立即终止。
    return {  # 返回只读衡量单日预测与十个风格的 Pearson 相关性。
        "max_abs_style_pearson": correlations.abs().max(),  # 记录maxabs风格Pearson。
        "mean_abs_style_pearson": correlations.abs().mean(),  # 记录均值abs风格Pearson。
    }

def summarize_adversarial_predictions(  # 汇总对抗头的风格回归和行业分类诊断。
    style_predictions: np.ndarray,  # 接收风格预测。
    style_targets: np.ndarray,  # 接收风格标签。
    style_valid: np.ndarray,  # 输入风格有效性掩码。
    industry_logits: np.ndarray,  # 接收行业logits。
    industry_codes: np.ndarray,  # 接收行业代码。
    industry_class_weights: np.ndarray | None = None,  # 接收行业class权重。
) -> dict[str, float]:  # 声明当前函数的返回类型。
    """汇总对抗头的风格回归和行业分类诊断。"""
    mask = np.asarray(style_valid, dtype=bool)  # 转换并统一掩码的数据表示。
    predicted_styles = np.asarray(style_predictions, dtype=np.float64)  # 转换并统一predicted风格的数据表示。
    target_styles = np.asarray(style_targets, dtype=np.float64)  # 转换并统一标签风格的数据表示。
    if predicted_styles.shape != target_styles.shape or mask.shape != target_styles.shape:  # 检查张量或数组形状是否符合接口约定。
        raise ValueError("对抗风格诊断形状不一致")  # 输入数据不满足业务或数值约束时立即终止。
    squared_errors = (predicted_styles - target_styles) ** 2
    style_r_squared: list[float] = []  # 接收风格rsquared。
    for column in range(STYLE_COUNT):  # 按给定步长遍历样本区间，更新列。
        valid = mask[:, column]  # 构造有效，限定后续参与计算的样本。
        values = target_styles[valid, column]
        denominator = float(((values - values.mean()) ** 2).sum())  # 计算分母。
        if len(values) < 2 or denominator <= np.finfo(np.float64).eps:  # 检查当前样本数量是否满足计算要求。
            raise ValueError("验证期风格标签不足或为常数")  # 输入数据不满足业务或数值约束时立即终止。
        numerator = float(squared_errors[valid, column].sum())  # 计算分子。
        style_r_squared.append(1.0 - numerator / denominator)  # 把当前结果追加到汇总列表。

    codes = np.asarray(industry_codes, dtype=np.int64).reshape(-1)  # [N,...] -> [N]。
    logits = np.asarray(industry_logits, dtype=np.float64)  # 转换并统一logits的数据表示。
    known = codes > 0  # 构造已知行业，限定后续参与计算的样本。
    targets = codes[known] - 1  # [B_known]，映射为 0 至 31。
    known_logits = logits[known]  # 构造已知行业logits，限定后续参与计算的样本。
    predicted = known_logits.argmax(axis=1)  # 输出行业预测类别。
    shifted = known_logits - known_logits.max(axis=1, keepdims=True)
    negative_log_likelihood = (
        np.log(np.exp(shifted).sum(axis=1))
        - shifted[np.arange(len(targets)), targets]
    )
    class_weights = np.asarray(  # 转换并统一class权重的数据表示。
        industry_class_weights
        if industry_class_weights is not None  # 检查必需数据、配置或校验结果是否缺失。
        else np.ones(INDUSTRY_COUNT),
        dtype=np.float64,  # 统一结果的数据类型。
    )
    if (  # 开始组合多项有效性或配置条件。
        class_weights.shape != (INDUSTRY_COUNT,)
        or not np.isfinite(class_weights).all()
        or (class_weights <= 0).any()
    ):
        raise ValueError("验证期行业类别权重形状或数值异常")  # 输入数据不满足业务或数值约束时立即终止。
    sample_weights = class_weights[targets]
    normalized_ce = float(
        np.average(negative_log_likelihood, weights=sample_weights)
        / math.log(INDUSTRY_COUNT)
    )
    confusion = np.bincount(  # 计算confusion。
        targets * INDUSTRY_COUNT + predicted,
        minlength=INDUSTRY_COUNT**2,  # 保证计数结果覆盖全部类别。
    ).reshape(INDUSTRY_COUNT, INDUSTRY_COUNT)  # [32×32] -> [32,32]。
    support = confusion.sum(axis=1)  # 计算support。
    present = support > 0  # 计算present判定条件。
    recall = np.divide(  # 仅对有效分母执行归一化。
        np.diag(confusion), support,
        out=np.zeros(INDUSTRY_COUNT, dtype=np.float64), where=present,  # 配置out。
    )
    predicted_count = confusion.sum(axis=0)  # 计算predicted数量。
    precision = np.divide(  # 仅对有效分母执行归一化。
        np.diag(confusion), predicted_count,
        out=np.zeros(INDUSTRY_COUNT, dtype=np.float64),  # 配置out。
        where=predicted_count > 0,  # 配置where。
    )
    f1 = np.divide(  # 仅对有效分母执行归一化。
        2 * precision * recall, precision + recall,
        out=np.zeros(INDUSTRY_COUNT, dtype=np.float64),  # 配置out。
        where=(precision + recall) > 0,  # 配置where。
    )
    return {  # 返回汇总对抗头的风格回归和行业分类诊断。
        "style_mse": float(squared_errors[mask].mean()),  # 记录风格mse。
        "style_r_squared_mean": float(np.mean(style_r_squared)),  # 记录风格rsquared均值。
        "style_r_squared_max": float(np.max(style_r_squared)),  # 记录风格rsquaredmax。
        "industry_loss": normalized_ce,  # 记录行业损失。
        "industry_accuracy": float((predicted == targets).mean()),  # 记录行业accuracy。
        "industry_balanced_accuracy": float(recall[present].mean()),  # 记录行业balancedaccuracy。
        "industry_macro_f1": float(f1[present].mean()),  # 记录行业macrof1。
        "industry_unknown_rate": float((codes == 0).mean()),  # 记录行业未知行业比例。
    }

def deterministic_probe_positions(  # 按稳定样本键确定性抽取每日 probe 样本。
    rows: int, *, year: int, seed: int, date_id: int
) -> np.ndarray:  # 声明当前函数的返回类型。
    """按稳定样本键确定性抽取每日 probe 样本。"""
    if rows <= PROBE_MAX_SAMPLES_PER_DATE:  # 检查当前数值是否越过业务或数值阈值。
        return np.arange(rows, dtype=np.int64)  # 返回按当前规则组织后的结果。
    random_seed = np.random.SeedSequence(
        [year, seed, date_id]
    ).generate_state(1)[0]
    generator = np.random.default_rng(random_seed)  # 创建可复现随机源。
    return np.sort(  # 返回按当前规则组织后的结果。
        generator.choice(
            rows, size=PROBE_MAX_SAMPLES_PER_DATE, replace=False
        ).astype(np.int64)
    )

def chronological_probe_split(date_ids: np.ndarray) -> dict[str, np.ndarray]:  # 按日期顺序切分 probe 训练、验证和测试区间。
    """按日期顺序切分 probe 训练、验证和测试区间。"""
    dates = np.unique(np.asarray(date_ids, dtype=np.int64))  # 压缩当日行业编码。
    if len(dates) < 10:  # 检查当前样本数量是否满足计算要求。
        raise ValueError("冻结 probe 至少需要10个验证交易日")  # 输入数据不满足业务或数值约束时立即终止。
    train_count = max(1, int(np.floor(0.4 * len(dates))))  # 计算训练数量。
    validation_count = max(1, int(np.floor(0.1 * len(dates))))  # 计算验证数量。
    if train_count + validation_count >= len(dates):  # 检查当前样本数量是否满足计算要求。
        raise ValueError("冻结 probe 日期不足以划分测试区间")  # 输入数据不满足业务或数值约束时立即终止。
    boundaries = {  # 初始化boundaries容器。
        "train": dates[:train_count],  # 记录训练。
        "validation": dates[train_count:train_count + validation_count],  # 记录验证。
        "test": dates[train_count + validation_count:],  # 记录test。
    }
    return {  # 返回按当前规则组织后的结果。
        name: np.flatnonzero(np.isin(date_ids, selected)).astype(np.int64)
        for name, selected in boundaries.items()  # 遍历boundariesitems，逐项处理名称筛选结果。
    }

# 使用 `torch.no_grad()` 包装 `extract_probe_dataset`，并定义其可复用的处理逻辑。
@torch.no_grad()  # 验证或推理阶段关闭梯度记录。
def extract_probe_dataset(  # 冻结主模型并提取 CLS、风格和行业 probe 数据。
    model: torch.nn.Module,  # 接收模型。
    loader: DataLoader,  # 接收数据加载器。
    device: torch.device,  # 接收设备。
    *,
    year: int,  # 接收年份。
    seed: int,  # 接收随机种子。
) -> dict[str, np.ndarray]:  # 声明当前函数的返回类型。
    """冻结主模型并提取 CLS、风格和行业 probe 数据。"""
    model.eval()  # 关闭 dropout 等训练行为。
    parts: dict[str, list[np.ndarray]] = {  # 初始化parts容器。
        "representations": [],  # 记录representations。
        "styles": [],  # 记录风格。
        "style_valid": [],  # 记录风格有效。
        "industry_codes": [],  # 记录行业代码。
        "date_ids": [],  # 记录交易日ids。
    }
    for item in StyleCudaPrefetcher(  # 遍历stylecudaprefetcher，逐项处理item。
        loader, device, bool(USE_CUDA_PREFETCH)
    ):
        (
            batch, videos, frame_mask, _targets, _original, _residual,  # 传入批次、视频、帧掩码、标签、原始、残差。
            _fitted, styles, style_valid, industry_codes,  # 传入拟合、风格、风格有效、行业代码。
        ) = item
        date_id = int(batch["date_ids"][0])
        if int(batch["date_ids"].min()) != int(batch["date_ids"].max()):  # 按当前模式、标识或状态选择处理分支。
            raise AssertionError("probe batch 混入多个交易日")  # 关键数据契约被破坏时立即终止。
        with base.autocast_context(device):  # 在自动混合精度上下文中执行前向计算。
            (  # 执行当前 batch 前向传播。
                _, _, _, representation, _, _, _  # 传入、、、表征、、、。
            ) = forward_training_model(
                model,
                videos,
                frame_mask,
                gamma=0.0,  # 配置对抗系数。
                styles=styles,  # 配置风格。
                style_valid=style_valid,  # 配置风格有效。
                industry_codes=industry_codes,  # 配置行业代码。
                sample_indices=batch["sample_indices"],  # 配置样本索引。
            )
        if representation is None:  # 检查所需对象是否尚未初始化。
            raise ValueError("No.3 probe 无法取得 CLS 表征")  # 输入数据不满足业务或数值约束时立即终止。
        positions = deterministic_probe_positions(  # 用探针测量表征中的风险信息。
            len(videos), year=year, seed=seed, date_id=date_id
        )
        index = torch.from_numpy(positions).to(device)  # 转换并统一索引的数据表示。
        parts["representations"].append(  # 把当前结果追加到汇总列表。
            representation.float().index_select(0, index).cpu().numpy()
        )
        parts["styles"].append(  # 把当前结果追加到汇总列表。
            styles.float().index_select(0, index).cpu().numpy()
        )
        parts["style_valid"].append(  # 把当前结果追加到汇总列表。
            style_valid.index_select(0, index).cpu().numpy()
        )
        parts["industry_codes"].append(  # 把当前结果追加到汇总列表。
            industry_codes.index_select(0, index).cpu().numpy()
        )
        parts["date_ids"].append(  # 把当前结果追加到汇总列表。
            np.full(len(positions), date_id, dtype=np.int64)
        )
    if not parts["representations"]:  # 检查必需数据、配置或校验结果是否缺失。
        raise ValueError("冻结 probe 没有抽取到验证样本")  # 输入数据不满足业务或数值约束时立即终止。
    return {  # 返回冻结主模型并提取 CLS、风格和行业 probe 数据。
        name: np.concatenate(values, axis=0)
        for name, values in parts.items()  # 遍历partsitems，逐项处理名称值。
    }

def _style_probe_metrics(  # 处理连续风格数据、有效掩码或相应诊断。
    predictions: np.ndarray,  # 接收预测。
    targets: np.ndarray,  # 接收标签。
    valid: np.ndarray,  # 输入与数值等长的有效性掩码。
) -> tuple[list[float], float]:  # 声明当前函数的返回类型。
    """处理连续风格数据、有效掩码或相应诊断。"""
    squared = (
        np.asarray(predictions, dtype=np.float64)
        - np.asarray(targets, dtype=np.float64)
    ) ** 2
    mask = np.asarray(valid, dtype=bool)  # 转换并统一掩码的数据表示。
    r_squared: list[float] = []  # 接收rsquared。
    for column in range(STYLE_COUNT):  # 按给定步长遍历样本区间，更新列。
        selected = mask[:, column]  # 构造筛选结果，限定后续参与计算的样本。
        values = targets[selected, column].astype(np.float64)  # 转换值的数据类型或存储位置。
        denominator = float(((values - values.mean()) ** 2).sum())  # 计算分母。
        if len(values) < 2 or denominator <= np.finfo(np.float64).eps:  # 检查当前样本数量是否满足计算要求。
            raise ValueError("probe 测试集风格标签不足或为常数")  # 输入数据不满足业务或数值约束时立即终止。
        r_squared.append(  # 把当前结果追加到汇总列表。
            1.0 - float(squared[selected, column].sum()) / denominator
        )
    return r_squared, float(squared[mask].mean())  # 返回处理连续风格数据、有效掩码或相应诊断。

def _industry_probe_metrics(  # 处理行业代码或行业诊断，并保留 UNKNOWN 口径。
    logits: np.ndarray, industry_codes: np.ndarray
) -> dict[str, float]:  # 声明当前函数的返回类型。
    """处理行业代码或行业诊断，并保留 UNKNOWN 口径。"""
    codes = np.asarray(industry_codes, dtype=np.int64).reshape(-1)  # [N,...] -> [N]。
    known = codes > 0  # 构造已知行业，限定后续参与计算的样本。
    targets = codes[known] - 1  # [B_known]，映射为 0 至 31。
    predicted = np.asarray(logits)[known].argmax(axis=1)  # 输出行业预测类别。
    confusion = np.bincount(  # 计算confusion。
        targets * INDUSTRY_COUNT + predicted,
        minlength=INDUSTRY_COUNT**2,  # 保证计数结果覆盖全部类别。
    ).reshape(INDUSTRY_COUNT, INDUSTRY_COUNT)  # [32×32] -> [32,32]。
    support = confusion.sum(axis=1)  # 计算support。
    present = support > 0  # 计算present判定条件。
    true_positive = np.diag(confusion).astype(np.float64)  # 转换truepositive的数据类型或存储位置。
    recall = np.divide(  # 仅对有效分母执行归一化。
        true_positive, support,  # 传入truepositive、support。
        out=np.zeros(INDUSTRY_COUNT), where=present,  # 配置out。
    )
    predicted_count = confusion.sum(axis=0)  # 计算predicted数量。
    precision = np.divide(  # 仅对有效分母执行归一化。
        true_positive, predicted_count,  # 传入truepositive、predicted数量。
        out=np.zeros(INDUSTRY_COUNT), where=predicted_count > 0,  # 配置out。
    )
    f1 = np.divide(  # 仅对有效分母执行归一化。
        2 * precision * recall, precision + recall,
        out=np.zeros(INDUSTRY_COUNT), where=(precision + recall) > 0,  # 配置out。
    )
    return {  # 返回处理行业代码或行业诊断，并保留 UNKNOWN 口径。
        "accuracy": float((predicted == targets).mean()),  # 记录accuracy。
        "balanced_accuracy": float(recall[present].mean()),  # 记录balancedaccuracy。
        "macro_f1": float(f1[present].mean()),  # 记录macrof1。
        "unknown_rate": float((codes == 0).mean()),  # 记录未知行业比例。
    }

def fit_linear_style_probe(  # 拟合线性岭回归 probe，衡量 CLS 中可恢复的风格信息。
    representations: np.ndarray,  # 接收representations。
    styles: np.ndarray,  # 接收风格。
    style_valid: np.ndarray,  # 输入风格有效性掩码。
    split: dict[str, np.ndarray],
    device: torch.device,  # 接收设备。
) -> np.ndarray:  # 声明当前函数的返回类型。
    """拟合线性岭回归 probe，衡量 CLS 中可恢复的风格信息。"""
    train = split["train"]
    test = split["test"]
    x_train = torch.from_numpy(representations[train]).float().to(device)  # 转换并统一x训练的数据表示。
    x_test = torch.from_numpy(representations[test]).float().to(device)  # 转换并统一xtest的数据表示。
    x_train = torch.cat(  # 对齐并合并x训练。
        [x_train, torch.ones((len(x_train), 1), device=device)], dim=1
    )
    x_test = torch.cat(  # 对齐并合并xtest。
        [x_test, torch.ones((len(x_test), 1), device=device)], dim=1
    )
    identity = torch.eye(x_train.shape[1], device=device)  # 构造岭回归惩罚项。
    identity[-1, -1] = 0.0
    predictions: list[np.ndarray] = []  # 接收预测。
    for column in range(STYLE_COUNT):  # 按给定步长遍历样本区间，更新列。
        valid = torch.from_numpy(style_valid[train, column]).bool().to(device)  # 转换并统一有效的数据表示。
        target = torch.from_numpy(styles[train, column]).float().to(device)  # 转换并统一标签的数据表示。
        design = x_train[valid]
        coefficients = torch.linalg.solve(  # 求解岭回归探针。
            design.T @ design + PROBE_RIDGE_ALPHA * identity,
            design.T @ target[valid],
        )
        predictions.append((x_test @ coefficients).cpu().numpy())  # 把当前结果追加到汇总列表。
    return np.column_stack(predictions).astype(np.float32)  # 返回拟合线性岭回归 probe，衡量 CLS 中可恢复的风格信息。

class _ProbeNetwork(torch.nn.Module):  # 构造用于表征泄漏诊断的轻量 probe 网络。
    def __init__(self, output_dim: int, nonlinear: bool) -> None:  # 初始化当前对象的配置、状态与子组件。
        super().__init__()
        if nonlinear:  # 根据当前数据状态选择处理分支。
            self.layers = torch.nn.Sequential(  # 保存layers，供该对象后续步骤复用。
                torch.nn.Linear(base.MODEL_CONFIG.embedding_dim, 128),
                torch.nn.GELU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(128, output_dim),
            )
        else:  # 处理上述条件均不成立的情况。
            self.layers = torch.nn.Linear(  # 保存layers，供该对象后续步骤复用。
                base.MODEL_CONFIG.embedding_dim, output_dim  # 传入base模型配置嵌入dim、输出dim。
            )

    def forward(self, values: torch.Tensor) -> torch.Tensor:  # 执行当前模块的前向计算。
        """执行 _ProbeNetwork 的前向计算并返回约定张量。"""
        return self.layers(values)  # 返回上述处理结果。

def fit_torch_probe(  # 训练线性或 MLP probe，并在独立日期区间评估。
    representations: np.ndarray,  # 接收representations。
    targets: np.ndarray,  # 接收标签。
    valid: np.ndarray | None,  # 输入与数值等长的有效性掩码。
    split: dict[str, np.ndarray],
    device: torch.device,  # 接收设备。
    *,
    output_dim: int,  # 接收输出dim。
    nonlinear: bool,  # 接收nonlinear。
    classification: bool,  # 接收classification。
    seed: int,  # 接收随机种子。
) -> np.ndarray:  # 声明当前函数的返回类型。
    """训练线性或 MLP probe，并在独立日期区间评估。"""
    torch.manual_seed(seed)
    network = _ProbeNetwork(output_dim, nonlinear).to(device)  # 转换network的数据类型或存储位置。
    optimizer = torch.optim.AdamW(  # 创建优化器。
        network.parameters(), lr=PROBE_LEARNING_RATE,
        weight_decay=PROBE_WEIGHT_DECAY,  # 设置权重衰减强度。
    )
    class_weights = None  # 先把class权重初始化为空，后续按条件填充。
    if classification:  # 根据当前数据状态选择处理分支。
        train_codes = targets[split["train"]].astype(np.int64)  # 转换训练代码的数据类型或存储位置。
        counts = np.bincount(  # 计算数量。
            train_codes[train_codes > 0] - 1, minlength=INDUSTRY_COUNT
        ).astype(np.float64)
        if (counts <= 0).any():  # 检查当前掩码或数据是否满足有效性要求。
            raise ValueError("probe 训练集未覆盖全部32行业")  # 输入数据不满足业务或数值约束时立即终止。
        weights = np.reciprocal(np.sqrt(counts))  # 按截面样本数平方根加权。
        weights /= weights.mean()
        class_weights = torch.from_numpy(weights.astype(np.float32)).to(device)  # 转换并统一class权重的数据表示。

    def loss_for(indices: np.ndarray, *, training: bool) -> torch.Tensor:  # 按 probe 类型返回连续风格回归或行业分类损失。
        """按 probe 类型返回连续风格回归或行业分类损失。"""
        if not len(indices):  # 检查当前样本数量是否满足计算要求。
            raise ValueError("probe split 为空")  # 输入数据不满足业务或数值约束时立即终止。
        total = torch.zeros((), dtype=torch.float32, device=device)  # 初始化total。
        rows = 0
        order = indices.copy()  # 转换排序索引的数据类型或存储位置。
        if training:  # 根据当前数据状态选择处理分支。
            generator = np.random.default_rng(seed + epoch)  # 创建可复现随机源。
            generator.shuffle(order)
        for start in range(0, len(order), PROBE_BATCH_SIZE):  # 按给定步长遍历样本区间，更新起点。
            selected = order[start:start + PROBE_BATCH_SIZE]  # 构造筛选结果，限定后续参与计算的样本。
            x = torch.from_numpy(representations[selected]).float().to(device)  # 转换并统一x的数据表示。
            output = network(x).float()  # 转换输出的数据类型或存储位置。
            if classification:  # 根据当前数据状态选择处理分支。
                codes = torch.from_numpy(targets[selected]).long().to(device)  # 转换并统一代码的数据表示。
                known = codes > 0  # 构造已知行业，限定后续参与计算的样本。
                if not known.any():  # 检查当前掩码或数据是否满足有效性要求。
                    continue  # 跳过当前项并处理下一项。
                batch_loss = torch.nn.functional.cross_entropy(  # 计算批次损失。
                    output[known], codes[known] - 1, weight=class_weights
                ) / math.log(INDUSTRY_COUNT)
                count = int(known.sum())  # 计算数量。
            else:  # 处理上述条件均不成立的情况。
                y = torch.from_numpy(targets[selected]).float().to(device)  # 转换并统一y的数据表示。
                mask = torch.from_numpy(valid[selected]).bool().to(device)  # 转换并统一掩码的数据表示。
                batch_loss = (output - y).square()[mask].mean()  # 计算批次损失。
                count = int(mask.sum())  # 计算数量。
            if training:  # 根据当前数据状态选择处理分支。
                optimizer.zero_grad(set_to_none=True)  # 清空上一批梯度。
                batch_loss.backward()  # 反向传播当前损失。
                optimizer.step()  # 更新参数或学习率状态。
            total += batch_loss.detach() * count
            rows += count
        if not rows:  # 检查必需数据、配置或校验结果是否缺失。
            raise ValueError("probe split 没有有效监督")  # 输入数据不满足业务或数值约束时立即终止。
        return total / rows  # 返回按当前规则组织后的结果。

    best_loss = np.inf  # 计算最优损失。
    best_state: dict[str, torch.Tensor] | None = None  # 声明可选输入；未提供时沿用默认处理流程。
    epochs_without_improvement = 0
    for epoch in range(1, PROBE_MAX_EPOCHS + 1):  # 按给定步长遍历样本区间，更新轮次。
        network.train()  # 启用训练模式。
        loss_for(split["train"], training=True)  # 按实验口径计算可微损失。
        network.eval()  # 关闭 dropout 等训练行为。
        with torch.no_grad():  # 关闭梯度记录以降低验证或推理开销。
            validation_loss = float(  # 计算验证损失。
                loss_for(split["validation"], training=False)
            )
        if validation_loss < best_loss - 1e-7:  # 检查当前数值是否越过业务或数值阈值。
            best_loss = validation_loss  # 计算最优损失。
            best_state = {  # 初始化最优状态容器。
                key: value.detach().cpu().clone()  # 接收键。
                for key, value in network.state_dict().items()  # 遍历network状态dictitems，逐项处理键值。
            }
            epochs_without_improvement = 0
        else:  # 处理上述条件均不成立的情况。
            epochs_without_improvement += 1
        if epochs_without_improvement >= PROBE_PATIENCE:  # 检查当前数值是否越过业务或数值阈值。
            break  # 满足终止条件后退出当前循环。
    if best_state is None:  # 检查所需对象是否尚未初始化。
        raise RuntimeError("probe 未生成最佳参数")  # 当前运行环境或任务状态不支持继续执行。
    network.load_state_dict(best_state)
    network.eval()  # 关闭 dropout 等训练行为。
    outputs: list[np.ndarray] = []  # 接收outputs。
    test = split["test"]
    with torch.no_grad():  # 关闭梯度记录以降低验证或推理开销。
        for start in range(0, len(test), PROBE_BATCH_SIZE):  # 按给定步长遍历样本区间，更新起点。
            selected = test[start:start + PROBE_BATCH_SIZE]  # 构造筛选结果，限定后续参与计算的样本。
            x = torch.from_numpy(representations[selected]).float().to(device)  # 转换并统一x的数据表示。
            outputs.append(network(x).float().cpu().numpy())  # 把当前结果追加到汇总列表。
    return np.concatenate(outputs, axis=0)  # 拼接各分块并返回完整结果。

def run_adversarial_probes(  # 对最佳 checkpoint 运行冻结 probe 并保存年度诊断。
    model: torch.nn.Module,  # 接收模型。
    validation_loader: DataLoader,  # 接收验证数据加载器。
    device: torch.device,  # 接收设备。
    *,
    year: int,  # 接收年份。
    seed: int,  # 接收随机种子。
) -> pd.DataFrame:  # 声明当前函数的返回类型。
    """对最佳 checkpoint 运行冻结 probe 并保存年度诊断。"""
    dataset = extract_probe_dataset(  # 用探针测量表征中的风险信息。
        model, validation_loader, device, year=year, seed=seed  # 创建模型。
    )
    split = chronological_probe_split(dataset["date_ids"])  # 用探针测量表征中的风险信息。
    test = split["test"]
    metadata = {  # 初始化元数据容器。
        "forecast_year": year,  # 记录forecast年份。
        "seed": seed,  # 记录随机种子。
        "sample_count": len(dataset["date_ids"]),  # 记录样本数量。
        "train_rows": len(split["train"]),  # 记录训练行数。
        "validation_rows": len(split["validation"]),  # 记录验证行数。
        "test_rows": len(test),  # 记录test行数。
        "train_date_min": int(dataset["date_ids"][split["train"]].min()),  # 记录训练交易日min。
        "train_date_max": int(dataset["date_ids"][split["train"]].max()),  # 记录训练交易日max。
        "validation_date_min": int(  # 记录验证交易日min。
            dataset["date_ids"][split["validation"]].min()
        ),
        "validation_date_max": int(  # 记录验证交易日max。
            dataset["date_ids"][split["validation"]].max()
        ),
        "test_date_min": int(dataset["date_ids"][test].min()),  # 记录test交易日min。
        "test_date_max": int(dataset["date_ids"][test].max()),  # 记录test交易日max。
        "unknown_rate": float((dataset["industry_codes"] == 0).mean()),  # 记录未知行业比例。
    }
    records: list[dict[str, Any]] = []  # 初始化records容器。
    linear_style = fit_linear_style_probe(
        dataset["representations"], dataset["styles"],
        dataset["style_valid"], split, device,
    )
    mlp_style = fit_torch_probe(
        dataset["representations"], dataset["styles"],
        dataset["style_valid"], split, device,
        output_dim=STYLE_COUNT, nonlinear=True, classification=False,  # 配置输出dim。
        seed=seed + 1000,  # 固定本轮随机种子。
    )
    for probe_name, predictions in (  # 遍历(，逐项处理probe名称预测。
        ("linear_ridge", linear_style), ("mlp", mlp_style)
    ):
        r_squared, mse = _style_probe_metrics(  # 用探针测量表征中的风险信息。
            predictions, dataset["styles"][test],
            dataset["style_valid"][test],
        )
        for column, value in zip(style_data.STYLE_COLUMNS, r_squared):  # 遍历zip风格数据风格列rsquared，逐项处理列值。
            records.append({  # 把当前结果追加到汇总列表。
                **metadata, "probe": probe_name, "target_kind": "style",
                "target": column, "r_squared": value, "mse": np.nan,  # 记录标签。
            })
        records.append({  # 把当前结果追加到汇总列表。
            **metadata, "probe": probe_name, "target_kind": "style_summary",
            "target": "all_styles", "r_squared": float(np.mean(r_squared)),  # 记录标签。
            "max_r_squared": float(np.max(r_squared)), "mse": mse,  # 记录maxrsquared。
        })

    for probe_name, nonlinear, offset in (  # 遍历(，逐项处理probe名称nonlinearoffset。
        ("linear_softmax", False, 2000), ("mlp", True, 3000)
    ):
        logits = fit_torch_probe(
            dataset["representations"], dataset["industry_codes"], None,
            split, device, output_dim=INDUSTRY_COUNT, nonlinear=nonlinear,
            classification=True, seed=seed + offset,  # 配置classification。
        )
        metrics = _industry_probe_metrics(  # 用探针测量表征中的风险信息。
            logits, dataset["industry_codes"][test]
        )
        records.append({  # 把当前结果追加到汇总列表。
            **metadata, "probe": probe_name, "target_kind": "industry",
            "target": "known_industries", **metrics,  # 记录标签。
        })
    return pd.DataFrame(records)  # 返回完成上述处理后的结果。

def validate_outputs() -> pd.DataFrame:  # 检查 checkpoint、预测、诊断和 manifest 是否完整一致。
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
    if stored_mode != expected_mode:  # 按当前模式、标识或状态选择处理分支。
        raise ValueError(  # 输入数据不满足业务或数值约束时立即终止。
            f"run={RUN_NAME!r} 的 manifest 开关为 {stored_mode}，"
            f"当前 CLI 开关为 {expected_mode}；请使用匹配的 run name/参数"
        )
    stored_lambda = float(manifest.get("raw_barra_r2_lambda", 0.0))
    if stored_lambda != RAW_BARRA_R2_LAMBDA:  # 按当前模式、标识或状态选择处理分支。
        raise ValueError(  # 输入数据不满足业务或数值约束时立即终止。
            f"run={RUN_NAME!r} 的 manifest λ={stored_lambda}，"
            f"当前 CLI λ={RAW_BARRA_R2_LAMBDA}"
        )
    if bool(manifest.get("adversarial_barra", False)) != adversarial_barra_enabled():  # 按当前模式、标识或状态选择处理分支。
        raise ValueError(  # 输入数据不满足业务或数值约束时立即终止。
            f"run={RUN_NAME!r} 的 No.3 开关与当前 CLI 不一致"
        )
    stored_conditioning = manifest.get("barra_conditioning")
    if barra_conditioning_enabled():  # 根据当前数据状态选择处理分支。
        expected_conditioning = {  # 初始化预期conditioning容器。
            "mode": BARRA_CONDITIONING_MODE,  # 记录模式。
            "hidden_dim": BARRA_GATE_HIDDEN_DIM,  # 记录隐藏层dim。
            "industry_embedding_dim": BARRA_INDUSTRY_EMBEDDING_DIM,  # 记录行业嵌入dim。
            "dropout": BARRA_GATE_DROPOUT,  # 记录dropout。
            "gate_scale": BARRA_GATE_SCALE,  # 记录门控scale。
            "gate_projection_initialization": "zeros",  # 把 gate_projection_initialization 映射为"zeros"。
            "multiplicative_only": True,  # 记录multiplicativeonly。
            "shuffle_policy": (  # 声明shuffle处理口径。
                "within_date_joint_permutation_by_stable_sample_key_v1"
                if BARRA_CONDITIONING_MODE == "shuffled" else None  # 按当前模式、标识或状态选择处理分支。
            ),
            "r2_uses_true_unmodified_barra": True,  # 记录r2usestrueunmodifiedBarra。
        }
        if stored_conditioning != expected_conditioning:  # 按当前模式、标识或状态选择处理分支。
            raise ValueError("No.4 manifest 与当前条件门控参数不一致")  # 输入数据不满足业务或数值约束时立即终止。
        full_target = style_data.require_full_barra_target_sidecar()
        if manifest.get("full_barra_target_sidecar_signature") != (  # 按当前模式、标识或状态选择处理分支。
            full_target["signature"]
        ):
            raise ValueError("No.4 manifest 的完整标签 sidecar 签名不一致")  # 输入数据不满足业务或数值约束时立即终止。
    elif stored_conditioning is not None:  # 检查上一条件不成立时的备选分支。
        raise ValueError("当前 CLI 未开启 No.4，但 manifest 含条件门控配置")  # 输入数据不满足业务或数值约束时立即终止。
    stored_dual = manifest.get("barra_dual_tower")
    if dual_tower_enabled():  # 根据当前数据状态选择处理分支。
        expected_dual = {  # 初始化预期dual容器。
            "mode": BARRA_DUAL_TOWER_MODE,  # 记录模式。
            "alpha_loss_weight": DUAL_TOWER_ALPHA_LOSS_WEIGHT,  # 记录Alpha损失权重。
            "style_loss_weight": DUAL_TOWER_STYLE_LOSS_WEIGHT,  # 记录风格损失权重。
            "style_hidden_dim": DUAL_TOWER_STYLE_HIDDEN_DIM,  # 记录风格隐藏层dim。
            "style_dropout": DUAL_TOWER_STYLE_DROPOUT,  # 记录风格dropout。
            "industry_embedding_dim": BARRA_INDUSTRY_EMBEDDING_DIM,  # 记录行业嵌入dim。
            "alpha_r2_lambda": RAW_BARRA_R2_LAMBDA,  # 记录Alphar2lambda。
            "style_output_initialization": "zeros",  # 把 style_output_initialization 映射为"zeros"。
            "published_factor": "prediction_ensemble_alpha",  # 把 published_factor 映射为"prediction_ensemble_alpha"。
            "component_columns": [  # 记录component字段列表。
                "prediction_ensemble_alpha",
                "prediction_ensemble_style",
                "prediction_ensemble_total_return",
            ],
            "ensemble_scaling": "center_components_divide_by_alpha_std",  # 把 ensemble_scaling 映射为"center_components_divide_by_alpha_std"。
            "seed_total_audit_policy": (  # 声明随机种子totalaudit处理口径。
                "stored_total_must_match_float32_or_bfloat16_alpha_plus_style"
            ),
            "ensemble_total_policy": (  # 声明ensembletotal处理口径。
                "reconstruct_float32_total_from_stored_alpha_plus_style"
            ),
            "r2_uses_true_unmodified_barra": True,  # 记录r2usestrueunmodifiedBarra。
        }
        if stored_dual != expected_dual:  # 按当前模式、标识或状态选择处理分支。
            raise ValueError("No.6 manifest 与当前双塔参数不一致")  # 输入数据不满足业务或数值约束时立即终止。
        full_target = style_data.require_full_barra_target_sidecar()
        dual_target = style_data.require_dual_tower_target_sidecar()
        if manifest.get("full_barra_target_sidecar_signature") != full_target["signature"]:  # 按当前模式、标识或状态选择处理分支。
            raise ValueError("No.6 完整残差 sidecar 签名不一致")  # 输入数据不满足业务或数值约束时立即终止。
        if manifest.get("dual_tower_target_sidecar_signature") != dual_target["signature"]:  # 按当前模式、标识或状态选择处理分支。
            raise ValueError("No.6 fitted-target sidecar 签名不一致")  # 输入数据不满足业务或数值约束时立即终止。
    elif stored_dual is not None:  # 检查上一条件不成立时的备选分支。
        raise ValueError("当前 CLI 未开启 No.6，但 manifest 含双塔配置")  # 输入数据不满足业务或数值约束时立即终止。

    stored_strict = manifest.get("strict_barra_output")
    if strict_barra_output_enabled():  # 根据当前数据状态选择处理分支。
        expected_strict = {  # 初始化预期strict容器。
            "weighting": "equal_weight_W_identity",  # 把 weighting 映射为"equal_weight_W_identity"。
            "design": [  # 记录design。
                "intercept",
                *style_data.STYLE_COLUMNS,
                "industry_fixed_effects",
            ],
            "solver": "industry_within_then_torch_fp32_reduced_qr",  # 把 solver 映射为"industry_within_then_torch_fp32_reduced_qr"。
            "post_projection": "zscore_ddof0",  # 把 post_projection 映射为"zscore_ddof0"。
            "applied_during": ["training_loss", "validation", "inference"],  # 记录appliedduring。
            "published_factor": "prediction_ensemble",  # 把 published_factor 映射为"prediction_ensemble"。
            "pre_neutral_audit_column": "prediction_ensemble_unprojected",  # 把 pre_neutral_audit_column 映射为"prediction_ensemble_unprojected"。
            "orthogonality_requirement": "H_transpose_W_s_equals_zero",  # 把 orthogonality_requirement 映射为"H_transpose_W_s_equals_zero"。
        }
        if stored_strict != expected_strict:  # 按当前模式、标识或状态选择处理分支。
            raise ValueError("严格 Barra 输出 manifest 与当前参数不一致")  # 输入数据不满足业务或数值约束时立即终止。
    elif stored_strict is not None:  # 检查上一条件不成立时的备选分支。
        raise ValueError("当前 CLI 未开启严格输出，但 manifest 含对应配置")  # 输入数据不满足业务或数值约束时立即终止。
    if adversarial_barra_enabled():  # 根据当前数据状态选择处理分支。
        expected_adversarial_config = {  # 初始化预期adversarial配置容器。
            "gamma_max": ADVERSARIAL_GAMMA_MAX,  # 记录对抗系数max。
            "warmup_epochs": ADVERSARIAL_WARMUP_EPOCHS,  # 记录warmupepochs。
            "cosine_ramp_epochs": ADVERSARIAL_RAMP_EPOCHS,  # 记录cosinerampepochs。
            "style_loss_weight": ADVERSARIAL_STYLE_LOSS_WEIGHT,  # 记录风格损失权重。
            "industry_loss_weight": ADVERSARIAL_INDUSTRY_LOSS_WEIGHT,  # 记录行业损失权重。
            "head_hidden_dim": ADVERSARIAL_HEAD_HIDDEN_DIM,  # 记录head隐藏层dim。
            "head_dropout": ADVERSARIAL_HEAD_DROPOUT,  # 记录headdropout。
            "industry_classes": INDUSTRY_COUNT,  # 记录行业classes。
            "unknown_industry_policy": "masked_from_ce",  # 把 unknown_industry_policy 映射为"masked_from_ce"。
            "inference_model": "backbone_and_return_head_only",  # 把 inference_model 映射为"backbone_and_return_head_only"。
            "probe_file": "adversarial_probe_diagnostics.parquet",  # 把 probe_file 映射为"adversarial_probe_diagnostics.parquet"。
            "probe_sample_cap_per_date": PROBE_MAX_SAMPLES_PER_DATE,  # 记录probe样本capper交易日。
            "probe_date_split": [0.4, 0.1, 0.5],  # 记录probe交易日切分。
            "probe_batch_size": PROBE_BATCH_SIZE,  # 记录probe批次大小。
            "probe_learning_rate": PROBE_LEARNING_RATE,  # 记录probelearning比例。
            "probe_weight_decay": PROBE_WEIGHT_DECAY,  # 记录probe权重decay。
            "probe_max_epochs": PROBE_MAX_EPOCHS,  # 记录probemaxepochs。
            "probe_patience": PROBE_PATIENCE,  # 记录probepatience。
            "probe_ridge_alpha": PROBE_RIDGE_ALPHA,  # 记录proberidgeAlpha。
            "probe_mlp_hidden_dim": 128,  # 记录probemlp隐藏层dim。
            "probe_mlp_dropout": 0.1,  # 记录probemlpdropout。
        }
        if manifest.get("adversarial_config") != expected_adversarial_config:  # 按当前模式、标识或状态选择处理分支。
            raise ValueError("No.3 manifest 与当前对抗或 probe 参数不一致")  # 输入数据不满足业务或数值约束时立即终止。
        probe_path = quality_root / "adversarial_probe_diagnostics.parquet"  # 构造probe路径。
        if not probe_path.exists():  # 检查目标文件或目录是否满足存在性要求。
            raise FileNotFoundError(probe_path)  # 目标文件不存在时立即终止。
        probes = pd.read_parquet(probe_path)  # 读取probes。
        expected_tasks = len(FORECAST_YEARS) * len(SEEDS)
        observed_tasks = probes[["forecast_year", "seed"]].drop_duplicates()  # 业务键去重。
        if len(observed_tasks) != expected_tasks:  # 检查当前样本数量是否满足计算要求。
            raise AssertionError("No.3 probe 不是8年度×3 seed")  # 关键数据契约被破坏时立即终止。
  # 构造去重集合，并保存到 `required_probe_kinds`。
        required_probe_kinds = {"style", "style_summary", "industry"}  # 初始化必需probekinds容器。
        if not required_probe_kinds.issubset(set(probes["target_kind"])):  # 检查必需数据、配置或校验结果是否缺失。
            raise AssertionError("No.3 probe 诊断类型不完整")  # 关键数据契约被破坏时立即终止。
        style_probes = set(  # 初始化风格probes容器。
            probes.loc[
                probes["target_kind"].eq("style_summary"), "probe"
            ]
        )
        industry_probes = set(  # 初始化行业probes容器。
            probes.loc[probes["target_kind"].eq("industry"), "probe"]
        )
        if style_probes != {"linear_ridge", "mlp"}:  # 按当前模式、标识或状态选择处理分支。
            raise AssertionError("No.3 风格 probe 类型不完整")  # 关键数据契约被破坏时立即终止。
        if industry_probes != {"linear_softmax", "mlp"}:  # 按当前模式、标识或状态选择处理分支。
            raise AssertionError("No.3 行业 probe 类型不完整")  # 关键数据契约被破坏时立即终止。


    summary = pd.read_parquet(quality_root / "vivit_training_summary.parquet")  # 读取汇总。
    if len(summary) != len(FORECAST_YEARS) * len(SEEDS):  # 检查当前样本数量是否满足计算要求。
        raise AssertionError("训练汇总不是 8 年×3 seed")  # 关键数据契约被破坏时立即终止。
    records: list[dict[str, Any]] = []  # 初始化records容器。
    base_columns = [  # 初始化base列容器。
        "sample_index",
        "prediction_ensemble_unprojected",
        "prediction_ensemble_raw",
        "prediction_ensemble",
    ]
    if dual_tower_enabled():  # 根据当前数据状态选择处理分支。
        base_columns.extend(  # 把一组结果并入当前列表。
            [
                "prediction_ensemble_alpha",
                "prediction_ensemble_style",
                "prediction_ensemble_total_return",
            ]
        )
    columns = [  # 初始化列容器。
        *base_columns,
        *(
            ["prediction_ensemble_projected"]
            if PROJECT_MODEL_OUTPUT or strict_barra_output_enabled()  # 根据当前数据状态选择处理分支。
            else []
        ),
    ]
    for year in FORECAST_YEARS:  # 遍历forecast年份，逐项处理年份。
        split = sequence_data.load_sequence_split(year)
        expected = len(split["prediction_indices"])
        path = prediction_root / f"vivit_predictions.{year}.parquet"
        if not expected:  # 检查必需数据、配置或校验结果是否缺失。
            if path.exists():  # 检查目标文件或目录是否满足存在性要求。
                raise AssertionError(f"{year} prediction 为空但存在预测文件")  # 关键数据契约被破坏时立即终止。
            records.append({"year": year, "rows": 0, "status": "通过"})  # 把当前结果追加到汇总列表。
            continue  # 跳过当前项并处理下一项。
        frame = pd.read_parquet(path, columns=columns)  # 读取帧。
        if len(frame) != expected or not np.array_equal(  # 检查当前样本数量是否满足计算要求。
            frame["sample_index"].to_numpy(np.int64), split["prediction_indices"]
        ):
            raise AssertionError(f"{year} 预测索引不一致")  # 关键数据契约被破坏时立即终止。
        if not np.isfinite(frame[columns[1:]].to_numpy(np.float64)).all():  # 检查参与计算的数值是否全部有限。
            raise AssertionError(f"{year} 预测列含 NaN/Inf")  # 关键数据契约被破坏时立即终止。
        if dual_tower_enabled():  # 根据当前数据状态选择处理分支。
            alpha = frame["prediction_ensemble_alpha"].to_numpy(np.float64)  # 转换并统一Alpha的数据表示。
            style = frame["prediction_ensemble_style"].to_numpy(np.float64)  # 转换并统一风格的数据表示。
            total = frame["prediction_ensemble_total_return"].to_numpy(np.float64)  # 转换并统一total的数据表示。
            if not np.array_equal(  # 检查必需数据、配置或校验结果是否缺失。
                frame["prediction_ensemble"].to_numpy(),
                frame["prediction_ensemble_alpha"].to_numpy(),
            ):
                raise AssertionError(f"{year} No.6 正式主列必须等于 Alpha")  # 关键数据契约被破坏时立即终止。
            if not np.allclose(total, alpha + style, rtol=1e-6, atol=1e-6):  # 检查必需数据、配置或校验结果是否缺失。
                raise AssertionError(f"{year} No.6 集成不满足 Alpha+Style=Total")  # 关键数据契约被破坏时立即终止。
        if strict_barra_output_enabled():  # 根据当前数据状态选择处理分支。
            if np.array_equal(  # 根据当前数据状态选择处理分支。
                frame["prediction_ensemble_unprojected"].to_numpy(),
                frame["prediction_ensemble_raw"].to_numpy(),
            ):
                raise AssertionError(f"{year} 严格中性主因子不能等于投影前分数")  # 关键数据契约被破坏时立即终止。
            if not np.array_equal(  # 检查必需数据、配置或校验结果是否缺失。
                frame["prediction_ensemble_projected"].to_numpy(),
                frame["prediction_ensemble_raw"].to_numpy(),
            ):
                raise AssertionError(f"{year} 严格中性主因子必须使用投影后分数")  # 关键数据契约被破坏时立即终止。
        elif not np.array_equal(  # 检查上一条件不成立时的备选分支。
            frame["prediction_ensemble_unprojected"].to_numpy(),
            frame["prediction_ensemble_raw"].to_numpy(),
        ):
            raise AssertionError(f"{year} 非严格模式主因子必须使用未投影分数")  # 关键数据契约被破坏时立即终止。
        if not np.array_equal(  # 检查必需数据、配置或校验结果是否缺失。
            frame["prediction_ensemble_raw"].to_numpy(),
            frame["prediction_ensemble"].to_numpy(),
        ):
            raise AssertionError(f"{year} raw 与 notebook03 输入列必须完全相同")  # 关键数据契约被破坏时立即终止。
        if PROJECT_MODEL_OUTPUT and np.array_equal(  # 根据当前数据状态选择处理分支。
            frame["prediction_ensemble_projected"].to_numpy(),
            frame["prediction_ensemble"].to_numpy(),
        ):
            raise AssertionError(f"{year} No.1 投影审计列不应覆盖未投影主因子")  # 关键数据契约被破坏时立即终止。
        records.append({"year": year, "rows": len(frame), "status": "通过"})  # 把当前结果追加到汇总列表。

    diagnostics = pd.read_parquet(  # 读取诊断。
        style_diagnostics_root(quality_root)
        / "prediction_style_exposure_diagnostics.parquet"
    )
    diagnostic_prefixes = (
        ("alpha", "style", "total")
        if dual_tower_enabled() else ("raw", "effective")  # 根据当前数据状态选择处理分支。
    )
  # 构造去重集合，并保存到 `required`。
    required = {  # 初始化必需容器。
        f"{prefix}_{stat}_abs_style_{method}"
        for prefix in diagnostic_prefixes  # 遍历diagnosticprefixes，逐项处理prefix。
        for stat in ("max", "mean")  # 遍历max均值，逐项处理stat。
        for method in ("pearson", "spearman")  # 遍历Pearsonspearman，逐项处理method。
    }
    if not required.issubset(diagnostics.columns):  # 检查必需数据、配置或校验结果是否缺失。
        raise AssertionError(  # 关键数据契约被破坏时立即终止。
            f"最终预测缺少风格相关性审计列：{sorted(required - set(diagnostics.columns))}"
        )
    if not np.isfinite(diagnostics[list(required)].to_numpy()).all():  # 检查参与计算的数值是否全部有限。
        raise AssertionError("最终预测的风格相关性审计含 NaN/Inf")  # 关键数据契约被破坏时立即终止。
    if PROJECT_MODEL_OUTPUT or strict_barra_output_enabled():  # 根据当前数据状态选择处理分支。
        maximum = float(diagnostics["effective_max_abs_style_pearson"].max())
        if maximum > 5e-5:  # 检查当前数值是否越过业务或数值阈值。
            raise AssertionError(  # 关键数据契约被破坏时立即终止。
                f"投影后预测未达到数值正交：max |Pearson|={maximum:.3e}"
            )
        if strict_barra_output_enabled():  # 根据当前数据状态选择处理分支。
            maximum_r_squared = float(  # 计算maximumrsquared。
                diagnostics["effective_joint_barra_r_squared"].max()
            )
            if maximum_r_squared > 1e-6:  # 检查当前数值是否越过业务或数值阈值。
                raise AssertionError("严格中性预测的联合 Barra R² 未接近零")  # 关键数据契约被破坏时立即终止。
            maximum_industry_mean = float(  # 计算maximum行业均值。
                diagnostics["effective_max_abs_industry_mean"].max()
            )
            if maximum_industry_mean > 5e-5:  # 检查当前数值是否越过业务或数值阈值。
                raise AssertionError(  # 关键数据契约被破坏时立即终止。
                    "严格中性预测的行业组均值未接近零"
                )
    return pd.DataFrame(records)  # 返回检查 checkpoint、预测、诊断和 manifest 是否完整一致。

def write_report() -> Path:  # 汇总训练、暴露、probe 和预测结果并写入实验报告。
    """汇总训练、暴露、probe 和预测结果并写入实验报告。"""
    _, _, quality_root = base.run_paths(RUN_NAME)
    summary = pd.read_parquet(quality_root / "vivit_training_summary.parquet")  # 读取汇总。
    manifest = json.loads(
        (quality_root / "a2_training_manifest.json").read_text(encoding="utf-8")
    )
    audit = validate_outputs()
    diagnostics = pd.read_parquet(  # 读取诊断。
        style_diagnostics_root(quality_root)
        / "prediction_style_exposure_diagnostics.parquet"
    )
    yearly = summary.groupby("forecast_year", as_index=False).agg(  # 按约定截面口径汇总。
        best_effective_rank_ic=("best_validation_rank_ic", "mean"),  # 配置最优有效排名ic。
        best_original_rank_ic=("best_validation_original_rank_ic", "mean"),  # 配置最优原始排名ic。
        max_validation_effective_style_pearson=(
            "best_validation_effective_max_abs_style_pearson", "max"
        ),
        mean_best_epoch=("best_epoch", "mean"),  # 配置均值最优轮次。
        mean_validation_barra_r_squared=(  # 计算均值验证Barrarsquared。
            "best_validation_barra_r_squared", "mean"
        ),
    )
    if strict_barra_output_enabled():  # 根据当前数据状态选择处理分支。
        yearly["mean_validation_effective_barra_r_squared"] = (  # 按约定截面口径汇总。
            summary.groupby("forecast_year")[
                "best_validation_effective_barra_r_squared"
            ].mean().to_numpy()
        )
        yearly["max_validation_effective_industry_mean"] = (  # 按约定截面口径汇总。
            summary.groupby("forecast_year")[
                "best_validation_effective_max_abs_industry_mean"
            ].max().to_numpy()
        )
    if adversarial_barra_enabled():  # 根据当前数据状态选择处理分支。
        adversarial_yearly = summary.groupby(  # 按约定截面口径汇总。
            "forecast_year", as_index=False
        ).agg(
            validation_adversarial_style_r_squared_mean=(  # 计算验证adversarial风格rsquared均值。
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
        yearly = yearly.merge(  # 按业务键对齐，避免依赖原始行序。
            adversarial_yearly, on="forecast_year", validate="one_to_one"
        )
    if dual_tower_enabled():  # 根据当前数据状态选择处理分支。
        prediction_style = diagnostics.groupby("year", as_index=False).agg(  # 按既定口径聚合。
            alpha_mean_joint_barra_r_squared=(  # 计算Alpha均值jointBarrarsquared。
                "alpha_joint_barra_r_squared", "mean"
            ),
            style_mean_joint_barra_r_squared=(  # 计算风格均值jointBarrarsquared。
                "style_joint_barra_r_squared", "mean"
            ),
            total_mean_joint_barra_r_squared=(  # 计算total均值jointBarrarsquared。
                "total_joint_barra_r_squared", "mean"
            ),
            alpha_residual_rank_ic=("alpha_residual_rank_ic", "mean"),  # 配置Alpha残差排名ic。
            style_fitted_rank_ic=("style_fitted_rank_ic", "mean"),  # 配置风格拟合排名ic。
            total_original_rank_ic=("total_original_rank_ic", "mean"),  # 配置total原始排名ic。
            alpha_style_pearson=("alpha_style_pearson", "mean"),  # 配置Alpha风格Pearson。
            max_additivity_error=(
                "dual_additivity_max_abs_error", "max"
            ),
        )
    else:  # 处理上述条件均不成立的情况。
        prediction_style = diagnostics.groupby("year", as_index=False).agg(  # 按既定口径聚合。
            raw_max_daily_style_pearson=("raw_max_abs_style_pearson", "max"),  # 配置原始maxdaily风格Pearson。
            effective_max_daily_style_pearson=(
                "effective_max_abs_style_pearson", "max"
            ),
            raw_mean_daily_style_spearman=(  # 计算原始均值daily风格spearman。
                "raw_mean_abs_style_spearman", "mean"
            ),
            effective_mean_daily_style_spearman=(  # 计算有效均值daily风格spearman。
                "effective_mean_abs_style_spearman", "mean"
            ),
        )
        if industry_sidecar_required():  # 根据当前数据状态选择处理分支。
            prediction_style["raw_mean_daily_joint_barra_r_squared"] = (  # 按股票与交易日业务键对齐。
                diagnostics.groupby("year")[
                    "raw_joint_barra_r_squared"
                ].mean().to_numpy()
            )
            if strict_barra_output_enabled():  # 根据当前数据状态选择处理分支。
                prediction_style["effective_max_daily_joint_barra_r_squared"] = (  # 按股票与交易日业务键对齐。
                    diagnostics.groupby("year")[
                        "effective_joint_barra_r_squared"
                    ].max().to_numpy()
                )
                prediction_style["effective_max_daily_industry_mean"] = (  # 计算预测风格。
                    diagnostics.groupby("year")[
                        "effective_max_abs_industry_mean"
                    ].max().to_numpy()
                )
    target_description = (
        "每日用完整 Barra 风险矩阵把原始标签分为拟合项与残差，"
        "二者分别做 ddof=0 Z-score，监督 Style 与 Alpha；Total 监督原始标签。"
        if dual_tower_enabled()  # 根据当前数据状态选择处理分支。
        else "每日用截距、十个连续风格/反转变量和行业固定效应联合 OLS，"
        "取残差后做 ddof=0 Z-score。"
        if barra_conditioning_enabled()  # 根据当前数据状态选择处理分支。
        else "每日用九个 CNE5S 风格和 20 日反转联合 OLS，"
        "取残差后做 ddof=0 Z-score。"
        if RESIDUALIZE_TARGET  # 根据当前数据状态选择处理分支。
        else "使用 feature store 中的原始标准化标签。"
    )
    output_description = (
        "双塔同时输出 Alpha、Style 与 Total=Alpha+Style；正式主列保存未经投影的 Alpha，"
        "三个 seed 的组件共同使用 Alpha 日截面标准差缩放。"
        if dual_tower_enabled()  # 根据当前数据状态选择处理分支。
        else "每个完整交易日将条件模型分数投影掉十风格和行业固定效应，"
        "再做 ddof=0 Z-score；投影分数同时用于损失、验证和正式发布，"
        "投影前分数只保留为审计列。"
        if strict_barra_output_enabled()  # 根据当前数据状态选择处理分支。
        else
        "每个完整交易日将模型分数投影到十风格正交补空间，再做 ddof=0 Z-score；"
        "该有效分数用于损失和验证，并在推理时保存为 projected 审计列；"
        "正式发布主因子列保存未投影模型原始分数。"
        if PROJECT_MODEL_OUTPUT  # 根据当前数据状态选择处理分支。
        else "直接使用模型原始分数；正式发布主因子列同样保存原始分数。"
    )
    if dual_tower_enabled():  # 根据当前数据状态选择处理分支。
        loss_description = (  # 计算损失description。
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
    elif strict_barra_output_enabled():  # 检查上一条件不成立时的备选分支。
        loss_description = (  # 计算损失description。
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
    elif barra_conditioning_enabled():  # 检查上一条件不成立时的备选分支。
        loss_description = (  # 计算损失description。
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
    elif adversarial_barra_enabled():  # 检查上一条件不成立时的备选分支。
        loss_description = (  # 计算损失description。
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
    elif raw_barra_r2_enabled():  # 检查上一条件不成立时的备选分支。
        loss_description = (  # 计算损失description。
            f"{LOSS_NAME}(raw_score, source_target) + "
            f"{RAW_BARRA_R2_LAMBDA} × R²(raw_score | 十风格 + 行业)"
        )
        penalty_description = (
            "直接惩罚原始输出的联合线性可解释度；行业采用固定效应。"
        )
        industry_description = "进入联合 R² 惩罚；不作为模型预测输入。"
    else:  # 处理上述条件均不成立的情况。
        loss_description = f"{LOSS_NAME}(effective_prediction, effective_target)"  # 计算损失description。
        penalty_description = "未使用 raw-score 联合 Barra R² 正则。"
        industry_description = "不进入标签回归或分数投影。"

    probe_section = ""
    if adversarial_barra_enabled():  # 根据当前数据状态选择处理分支。
        probes = pd.read_parquet(  # 读取probes。
            quality_root / "adversarial_probe_diagnostics.parquet"
        )
        probe_summary = probes[  # 按约定截面口径汇总。
            probes["target_kind"].isin(["style_summary", "industry"])
        ].groupby(["probe", "target_kind"], as_index=False).agg(
            mean_style_r_squared=("r_squared", "mean"),  # 配置均值风格rsquared。
            max_style_r_squared=("max_r_squared", "mean"),  # 配置max风格rsquared。
            style_mse=("mse", "mean"),  # 配置风格mse。
            industry_accuracy=("accuracy", "mean"),  # 配置行业accuracy。
            industry_balanced_accuracy=("balanced_accuracy", "mean"),  # 配置行业balancedaccuracy。
            industry_macro_f1=("macro_f1", "mean"),  # 配置行业macrof1。
            unknown_rate=("unknown_rate", "mean"),  # 配置未知行业比率。
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
    sequence_data.atomic_write_text(report, path)  # 原子落盘，避免半成品。
    return path  # 返回路径。

def smoke_test(  # 运行小批量前向、反向和验证冒烟测试，不生成正式产物。
    forecast_year: int = 2018,  # 接收forecast年份。
    seed: int = 42,  # 接收随机种子。
    max_batches: int = 1,  # 接收maxbatches。
) -> dict[str, Any]:  # 声明当前函数的返回类型。
    """运行小批量前向、反向和验证冒烟测试，不生成正式产物。"""
    return train_task(  # 返回运行小批量前向、反向和验证冒烟测试，不生成正式产物。
        forecast_year, seed, 0, max_epochs=1, max_batches=max_batches, persist=False
    )

def add_training_arguments(parser: argparse.ArgumentParser) -> None:  # 向命令行解析器注册训练后端参数。
    """向命令行解析器注册训练后端参数。"""
    parser.add_argument("--run-name", default=RUN_NAME)  # 注册命令行参数及其默认值。
    parser.add_argument(  # 注册命令行参数及其默认值。
        "--train-batch-mode", choices=base.TRAIN_BATCH_MODES, default=TRAIN_BATCH_MODE
    )
    parser.add_argument("--loss-name", choices=base.LOSS_NAMES, default=LOSS_NAME)  # 注册命令行参数及其默认值。
    parser.add_argument("--rank-temperature", type=float, default=RANK_TEMPERATURE)  # 注册命令行参数及其默认值。
    parser.add_argument("--turnover-lambda", type=float, default=TURNOVER_LAMBDA)  # 注册命令行参数及其默认值。
    parser.add_argument("--huber-delta", type=float, default=HUBER_DELTA)  # 注册命令行参数及其默认值。
    parser.add_argument("--smooth-l1-beta", type=float, default=SMOOTH_L1_BETA)  # 注册命令行参数及其默认值。
    parser.add_argument("--hybrid-weight", type=float, default=HYBRID_WEIGHT)  # 注册命令行参数及其默认值。
    parser.add_argument(  # 注册命令行参数及其默认值。
        "--validation-metric",
        choices=base.VALIDATION_METRICS,  # 配置choices。
        default=VALIDATION_METRIC,  # 配置默认。
    )
    parser.add_argument("--workers-per-task", type=int, default=WORKERS_PER_TASK)  # 注册命令行参数及其默认值。
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)  # 注册命令行参数及其默认值。
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)  # 注册命令行参数及其默认值。
    parser.add_argument("--patience", type=int, default=PATIENCE)  # 注册命令行参数及其默认值。
    parser.add_argument("--compile", type=int, choices=(0, 1), default=USE_TORCH_COMPILE)  # 注册命令行参数及其默认值。
    parser.add_argument("--compile-mode", default=COMPILE_MODE)  # 注册命令行参数及其默认值。
    parser.add_argument("--fused-adam", type=int, choices=(0, 1), default=USE_FUSED_ADAM)  # 注册命令行参数及其默认值。
    parser.add_argument(  # 注册命令行参数及其默认值。
        "--cuda-prefetch", type=int, choices=(0, 1), default=USE_CUDA_PREFETCH
    )
    parser.add_argument("--overwrite", type=int, choices=(0, 1), default=OVERWRITE)  # 注册命令行参数及其默认值。
    parser.add_argument("--residualize-target", type=int, choices=(0, 1), default=1)  # 注册命令行参数及其默认值。
    parser.add_argument(  # 注册命令行参数及其默认值。
        "--project-model-output", type=int, choices=(0, 1), default=1
    )
    parser.add_argument(  # 注册命令行参数及其默认值。
        "--raw-barra-r2-lambda",
        type=float,  # 配置type。
        default=RAW_BARRA_R2_LAMBDA,  # 配置默认。
    )
    parser.add_argument(  # 注册命令行参数及其默认值。
        "--barra-conditioning-mode",
        choices=BARRA_CONDITIONING_MODES,  # 配置choices。
        default=BARRA_CONDITIONING_MODE,  # 配置默认。
    )
    parser.add_argument(  # 注册命令行参数及其默认值。
        "--barra-output-neutralization", type=int, choices=(0, 1),
        default=int(BARRA_OUTPUT_NEUTRALIZATION),  # 配置默认。
    )
    parser.add_argument(  # 注册命令行参数及其默认值。
        "--barra-dual-tower", type=int, choices=(0, 1),
        default=int(BARRA_DUAL_TOWER),  # 配置默认。
    )
    parser.add_argument(  # 注册命令行参数及其默认值。
        "--barra-dual-tower-mode",
        choices=BARRA_DUAL_TOWER_MODES,  # 配置choices。
        default=BARRA_DUAL_TOWER_MODE,  # 配置默认。
    )
    parser.add_argument(  # 注册命令行参数及其默认值。
        "--dual-tower-alpha-loss-weight", type=float,
        default=DUAL_TOWER_ALPHA_LOSS_WEIGHT,  # 配置默认。
    )
    parser.add_argument(  # 注册命令行参数及其默认值。
        "--dual-tower-style-loss-weight", type=float,
        default=DUAL_TOWER_STYLE_LOSS_WEIGHT,  # 配置默认。
    )
    parser.add_argument(  # 注册命令行参数及其默认值。
        "--dual-tower-style-hidden-dim", type=int,
        default=DUAL_TOWER_STYLE_HIDDEN_DIM,  # 配置默认。
    )
    parser.add_argument(  # 注册命令行参数及其默认值。
        "--dual-tower-style-dropout", type=float,
        default=DUAL_TOWER_STYLE_DROPOUT,  # 配置默认。
    )
    parser.add_argument(  # 注册命令行参数及其默认值。
        "--barra-gate-hidden-dim",
        type=int,  # 配置type。
        default=BARRA_GATE_HIDDEN_DIM,  # 配置默认。
    )
    parser.add_argument(  # 注册命令行参数及其默认值。
        "--barra-industry-embedding-dim",
        type=int,  # 配置type。
        default=BARRA_INDUSTRY_EMBEDDING_DIM,  # 配置默认。
    )
    parser.add_argument(  # 注册命令行参数及其默认值。
        "--barra-gate-dropout",
        type=float,  # 配置type。
        default=BARRA_GATE_DROPOUT,  # 配置默认。
    )
    parser.add_argument(  # 注册命令行参数及其默认值。
        "--barra-gate-scale",
        type=float,  # 配置type。
        default=BARRA_GATE_SCALE,  # 配置默认。
    )
    parser.add_argument(  # 注册命令行参数及其默认值。
        "--adversarial-barra", type=int, choices=(0, 1),
        default=int(ADVERSARIAL_BARRA),  # 配置默认。
    )
    parser.add_argument(  # 注册命令行参数及其默认值。
        "--adversarial-gamma-max", type=float,
        default=ADVERSARIAL_GAMMA_MAX,  # 配置默认。
    )
    parser.add_argument(  # 注册命令行参数及其默认值。
        "--adversarial-warmup-epochs", type=int,
        default=ADVERSARIAL_WARMUP_EPOCHS,  # 配置默认。
    )
    parser.add_argument(  # 注册命令行参数及其默认值。
        "--adversarial-ramp-epochs", type=int,
        default=ADVERSARIAL_RAMP_EPOCHS,  # 配置默认。
    )
    parser.add_argument(  # 注册命令行参数及其默认值。
        "--adversarial-style-loss-weight", type=float,
        default=ADVERSARIAL_STYLE_LOSS_WEIGHT,  # 配置默认。
    )
    parser.add_argument(  # 注册命令行参数及其默认值。
        "--adversarial-industry-loss-weight", type=float,
        default=ADVERSARIAL_INDUSTRY_LOSS_WEIGHT,  # 配置默认。
    )
    parser.add_argument(  # 注册命令行参数及其默认值。
        "--adversarial-head-hidden-dim", type=int,
        default=ADVERSARIAL_HEAD_HIDDEN_DIM,  # 配置默认。
    )
    parser.add_argument(  # 注册命令行参数及其默认值。
        "--adversarial-head-dropout", type=float,
        default=ADVERSARIAL_HEAD_DROPOUT,  # 配置默认。
    )


def worker_command(  # 构造单个年度×seed worker 的命令行参数。
    args: argparse.Namespace, year: int, seed: int, physical_gpu: int
) -> list[str]:  # 声明当前函数的返回类型。
    """构造单个年度×seed worker 的命令行参数。"""
    command = [  # 初始化command容器。
        sys.executable,
        str(SCRIPT_PATH),
        "worker",
        "--worker-year", str(year),
        "--worker-seed", str(seed),
        "--worker-physical-gpu", str(physical_gpu),
    ]
    for name in (  # 遍历(，逐项处理名称。
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
        command.extend([f"--{name.replace('_', '-')}", str(getattr(args, name))])  # 把一组结果并入当前列表。
    return command  # 返回command。


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


def _barra_ddp_rank(  # Barraddp排名。
    rank: int,  # 接收排名。
    world_size: int,  # 接收world大小。
    args: argparse.Namespace,  # 接收args。
    smoke: bool,  # 接收smoke。
) -> None:  # 声明当前函数的返回类型。
    """执行 _barra_ddp_rank 对应的分布式训练处理。"""
    args.execution_route = "ddp"
    args.ddp_world_size = world_size
    configure_from_args(args)
    context = distributed.initialize_process_group(rank, world_size, backend="nccl")
    try:  # 执行可能失败的外部读取或计算步骤。
        result = train_task(
            args.forecast_year if smoke else args.worker_year,
            args.seed if smoke else args.worker_seed,
            int(EXECUTION_DEVICE_IDS[0]),
            max_epochs=1 if smoke else None,  # 配置maxepochs。
            max_batches=args.max_batches if smoke else None,  # 限制调试运行处理的最大批次数。
            persist=not smoke,  # 配置persist。
            distributed_context=context,  # 配置distributed上下文。
        )
        if context.is_primary:  # 根据当前数据状态选择处理分支。
            result.update({  # 用当前配置更新参数字典。
                "execution_route": "ddp",  # 把 execution_route 映射为"ddp"。
                "world_size": world_size,  # 记录world大小。
                "persistent_outputs_written": not smoke,  # 记录persistentoutputswritten。
            })
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
    args.execution_route = "ddp"
    args.ddp_world_size = len(devices)
    if smoke:  # 根据当前数据状态选择处理分支。
        args.worker_year = args.forecast_year
        args.worker_seed = args.seed
    args.worker_physical_gpu = devices[0]
    distributed.configure_master(distributed.reserve_local_port())
    mp.spawn(
        _barra_ddp_rank,
        args=(len(devices), args, smoke),  # 配置args。
        nprocs=len(devices),  # 配置nprocs。
        join=True,  # 配置join。
    )


def run_ddp(args: argparse.Namespace) -> pd.DataFrame:  # 执行ddp。
    """执行 run_ddp 对应的分布式训练处理。"""

    devices = distributed.parse_device_ids(args.device_ids)
    args.execution_route = "ddp"
    args.ddp_world_size = len(devices)
    configure_from_args(args)
    _, _, quality_root = base.run_paths(RUN_NAME)
    log_dir = quality_root / "ddp_task_logs"  # 构造log目录。
    log_dir.mkdir(parents=True, exist_ok=True)  # 确保目标目录存在。
    started = time.perf_counter()
    print(  # 输出当前任务的进度或诊断信息。
        f"style-neutral DDP start: "
        f"tasks={len(FORECAST_YEARS) * len(SEEDS)}, "
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
                tail_text = "\n".join(  # 对齐并合并tailtext。
                    log_path.read_text(
                        encoding="utf-8", errors="replace"
                    ).splitlines()[-100:]
                )
                raise RuntimeError(  # 当前运行环境或任务状态不支持继续执行。
                    f"DDP 任务失败：{year}/seed={seed}\n{tail_text}"
                )
            print(f"done DDP: {year}/seed={seed}", flush=True)  # 输出当前任务的进度或诊断信息。
    result = finalize_run(time.perf_counter() - started)
    print(validate_outputs().to_string(index=False), flush=True)  # 输出当前任务的进度或诊断信息。
    print(f"实验报告：{write_report()}", flush=True)  # 输出当前任务的进度或诊断信息。
    return result  # 返回结果。


def run_a2(args: argparse.Namespace) -> pd.DataFrame:  # 调度所有年度和随机种子任务，并支持断点续训与失败恢复。
    """调度所有年度和随机种子任务，并支持断点续训与失败恢复。"""
    devices = tuple(int(value) for value in args.device_ids.split(","))  # 初始化devices容器。
    args.execution_route = "a2"
    args.ddp_world_size = 1
    configure_from_args(args)
    if not devices or len(set(devices)) != len(devices):  # 检查当前样本数量是否满足计算要求。
        raise ValueError("--device-ids 为空或包含重复 GPU")  # 输入数据不满足业务或数值约束时立即终止。
    tasks = deque((year, seed) for year in FORECAST_YEARS for seed in SEEDS)  # 计算tasks判定条件。
    _, _, quality_root = base.run_paths(RUN_NAME)
    log_dir = quality_root / "style_task_logs"  # 构造log目录。
    log_dir.mkdir(parents=True, exist_ok=True)  # 确保目标目录存在。
    running: list[dict[str, Any]] = []  # 初始化running容器。
    started = time.perf_counter()
    print(  # 输出当前任务的进度或诊断信息。
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
        flush=True,  # 配置flush。
    )
    try:  # 执行可能失败的外部读取或计算步骤。
        while tasks or running:  # 在终止条件出现前持续迭代。
            for physical_gpu in devices:  # 遍历devices，逐项处理physicalgpu。
                active = sum(
                    item["physical_gpu"] == physical_gpu for item in running
                )
                while tasks and active < TASKS_PER_GPU:  # 在终止条件出现前持续迭代。
                    year, seed = tasks.popleft()
                    log_path = log_dir / f"forecast_{year}.seed_{seed}.log"  # 构造log路径。
                    handle = log_path.open("a", encoding="utf-8")
                    handle.write(  # 追加任务状态日志。
                        f"\n=== launch {year}/seed={seed} at "
                        f"{time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
                    )
                    handle.flush()  # 将 memmap 修改刷入磁盘。
                    environment = os.environ.copy()  # 转换environment的数据类型或存储位置。
                    environment["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
                    process = subprocess.Popen(  # 启动独立训练任务。
                        worker_command(args, year, seed, physical_gpu),
                        cwd=VIVIT_DIR,  # 配置cwd。
                        env=environment,  # 配置env。
                        stdout=handle,  # 配置stdout。
                        stderr=subprocess.STDOUT,  # 配置stderr。
                        text=True,  # 配置text。
                    )
                    running.append(  # 把当前结果追加到汇总列表。
                        {
                            "process": process,  # 记录process。
                            "physical_gpu": physical_gpu,  # 记录physicalgpu。
                            "year": year,  # 记录年份。
                            "seed": seed,  # 记录随机种子。
                            "log": log_path,  # 记录log。
                            "handle": handle,  # 记录handle。
                            "started": time.perf_counter(),  # 记录started。
                        }
                    )
                    active += 1
                    print(  # 输出当前任务的进度或诊断信息。
                        f"launch GPU{physical_gpu}: {year}/seed={seed} "
                        f"pid={process.pid}",
                        flush=True,  # 配置flush。
                    )
            finished: list[dict[str, Any]] = []  # 初始化finished容器。
            for item in running:  # 遍历running，逐项处理item。
                code = item["process"].poll()  # 检查训练任务状态。
                if code is None:  # 检查所需对象是否尚未初始化。
                    continue  # 跳过当前项并处理下一项。
                item["handle"].close()
                finished.append(item)  # 把当前结果追加到汇总列表。
                if code:  # 根据当前数据状态选择处理分支。
                    tail = item["log"].read_text(  # 读取tail。
                        encoding="utf-8", errors="replace"
                    ).splitlines()[-100:]
                    raise RuntimeError(  # 当前运行环境或任务状态不支持继续执行。
                        f"任务失败：{item['year']}/seed={item['seed']}\n"
                        + "\n".join(tail)
                    )
                print(  # 输出当前任务的进度或诊断信息。
                    f"done GPU{item['physical_gpu']}: "
                    f"{item['year']}/seed={item['seed']} "
                    f"{(time.perf_counter()-item['started'])/60:.2f} min",
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
            item["handle"].close()
        raise

    result = finalize_run(time.perf_counter() - started)  # 汇总任务并写出完成标记。
    print(validate_outputs().to_string(index=False), flush=True)  # 输出当前任务的进度或诊断信息。
    print(f"实验报告：{write_report()}", flush=True)  # 输出当前任务的进度或诊断信息。
    return result  # 返回结果。


def parse_args() -> argparse.Namespace:  # 解析并校验命令行参数。
    """解析并校验命令行参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-data")
    prepare.add_argument("--overwrite", type=int, choices=(0, 1), default=0)  # 注册命令行参数及其默认值。

    prepare.add_argument(  # 注册命令行参数及其默认值。
        "--include-industry-sidecar", type=int, choices=(0, 1), default=0
    )
    prepare.add_argument(  # 注册命令行参数及其默认值。
        "--include-full-barra-target", type=int, choices=(0, 1), default=0
    )
    prepare.add_argument(  # 注册命令行参数及其默认值。
        "--include-dual-tower-target", type=int, choices=(0, 1), default=0
    )
    smoke = subparsers.add_parser("smoke")
    add_training_arguments(smoke)
    smoke.add_argument("--forecast-year", type=int, default=2018)  # 注册命令行参数及其默认值。
    smoke.add_argument("--seed", type=int, default=42)  # 注册命令行参数及其默认值。
    smoke.add_argument("--max-batches", type=int, default=1)  # 注册命令行参数及其默认值。
    smoke.set_defaults(compile=0, workers_per_task=0)

    worker = subparsers.add_parser("worker")
    add_training_arguments(worker)
    worker.add_argument("--worker-year", type=int, required=True)  # 注册命令行参数及其默认值。
    worker.add_argument("--worker-seed", type=int, required=True)  # 注册命令行参数及其默认值。
    worker.add_argument("--worker-physical-gpu", type=int, required=True)  # 注册命令行参数及其默认值。

    a2 = subparsers.add_parser("a2")
    add_training_arguments(a2)
    a2.add_argument("--device-ids", default=DEVICE_IDS)  # 注册命令行参数及其默认值。
    a2.add_argument("--tasks-per-gpu", type=int, default=TASKS_PER_GPU)  # 注册命令行参数及其默认值。
    ddp = subparsers.add_parser("ddp")
    add_training_arguments(ddp)
    ddp.add_argument("--device-ids", default=DEVICE_IDS)  # 注册命令行参数及其默认值。

    ddp_worker = subparsers.add_parser("ddp-worker")
    add_training_arguments(ddp_worker)
    ddp_worker.add_argument("--worker-year", type=int, required=True)  # 注册命令行参数及其默认值。
    ddp_worker.add_argument("--worker-seed", type=int, required=True)  # 注册命令行参数及其默认值。
    ddp_worker.add_argument("--worker-physical-gpu", type=int, required=True)  # 注册命令行参数及其默认值。
    ddp_worker.add_argument("--device-ids", required=True)  # 注册命令行参数及其默认值。

    ddp_smoke = subparsers.add_parser("ddp-smoke")
    add_training_arguments(ddp_smoke)
    ddp_smoke.add_argument("--forecast-year", type=int, default=2018)  # 注册命令行参数及其默认值。
    ddp_smoke.add_argument("--seed", type=int, default=42)  # 注册命令行参数及其默认值。
    ddp_smoke.add_argument("--max-batches", type=int, default=1)  # 注册命令行参数及其默认值。
    ddp_smoke.add_argument("--device-ids", default=DEVICE_IDS)  # 注册命令行参数及其默认值。
    ddp_smoke.set_defaults(compile=0, workers_per_task=0)

    for command in ("validate", "report"):  # 遍历校验报告，逐项处理command。
        subparser = subparsers.add_parser(command)
        add_training_arguments(subparser)
    return parser.parse_args()  # 返回解析并校验命令行参数。


def main() -> None:  # 根据命令行子命令分派数据、训练、验证或评价流程。
    """根据命令行子命令分派数据、训练、验证或评价流程。"""
    args = parse_args()
    if args.command == "prepare-data":  # 按命令行子命令选择执行路径。
        style_manifest = style_data.build_style_neutral_store(
            overwrite=bool(args.overwrite)
        )
        prepared = {"style_store": style_manifest}  # 初始化prepared容器。
        include_dual_target = bool(args.include_dual_tower_target)
        include_full_target = (
            bool(args.include_full_barra_target) or include_dual_target
        )
        if bool(args.include_industry_sidecar) or include_full_target:  # 根据当前数据状态选择处理分支。
            prepared["industry_sidecar"] = style_data.build_industry_sidecar(
                overwrite=bool(args.overwrite)
            )
        if include_full_target:  # 根据当前数据状态选择处理分支。
            prepared["full_barra_target"] = (
                style_data.build_full_barra_target_sidecar(
                    overwrite=bool(args.overwrite)
                )
            )
        if include_dual_target:  # 根据当前数据状态选择处理分支。
            prepared["dual_tower_target"] = (
                style_data.build_dual_tower_target_sidecar(
                    overwrite=bool(args.overwrite)
                )
            )
        print(  # 输出当前任务的进度或诊断信息。
            json.dumps(
                prepared, ensure_ascii=False, indent=2
            )
        )
        return  # 返回根据命令行子命令分派数据、训练、验证或评价流程。

    if args.command in {"ddp", "ddp-worker", "ddp-smoke"}:  # 按命令行子命令选择执行路径。
        args.execution_route = "ddp"
        args.ddp_world_size = len(distributed.parse_device_ids(args.device_ids))
    else:  # 处理上述条件均不成立的情况。
        args.execution_route = "a2"
        args.ddp_world_size = 1
    configure_from_args(args)
    if args.command in {"smoke", "a2", "ddp", "ddp-smoke"}:  # 按命令行子命令选择执行路径。
        style_data.build_style_neutral_store(overwrite=False)
    else:  # 处理上述条件均不成立的情况。
        style_data.require_style_neutral_store()
    if industry_sidecar_required():  # 根据当前数据状态选择处理分支。
        if args.command in {"smoke", "a2", "ddp", "ddp-smoke"}:  # 按命令行子命令选择执行路径。
            style_data.build_industry_sidecar(overwrite=False)
        else:  # 处理上述条件均不成立的情况。
            style_data.require_industry_sidecar()
    if barra_conditioning_enabled() or dual_tower_enabled():  # 根据当前数据状态选择处理分支。
        if args.command in {"smoke", "a2", "ddp", "ddp-smoke"}:  # 按命令行子命令选择执行路径。
            style_data.build_full_barra_target_sidecar(overwrite=False)
        else:  # 处理上述条件均不成立的情况。
            style_data.require_full_barra_target_sidecar()

    if dual_tower_enabled():  # 根据当前数据状态选择处理分支。
        if args.command in {"smoke", "a2", "ddp", "ddp-smoke"}:  # 按命令行子命令选择执行路径。
            style_data.build_dual_tower_target_sidecar(overwrite=False)
        else:  # 处理上述条件均不成立的情况。
            style_data.require_dual_tower_target_sidecar()

    if args.command == "smoke":  # 按命令行子命令选择执行路径。
        print(  # 输出当前任务的进度或诊断信息。
            json.dumps(
                smoke_test(args.forecast_year, args.seed, args.max_batches),
                ensure_ascii=False,  # 配置确保ascii。
                indent=2,  # 配置indent。
            )
        )
    elif args.command == "worker":  # 选择 worker 子命令的处理路径。
        print(  # 输出当前任务的进度或诊断信息。
            json.dumps(
                train_task(
                    args.worker_year, args.worker_seed, args.worker_physical_gpu  # 传入worker年份、worker随机种子、workerphysicalgpu。
                ),
                ensure_ascii=False,  # 配置确保ascii。
                indent=2,  # 配置indent。
            )
        )
    elif args.command == "ddp-worker":  # 选择 ddp-worker 子命令的处理路径。
        run_ddp_worker(args)
    elif args.command == "ddp-smoke":  # 选择 ddp-smoke 子命令的处理路径。
        run_ddp_worker(args, smoke=True)
    elif args.command == "a2":  # 选择 a2 子命令的处理路径。
        print(run_a2(args).to_string(index=False))  # 输出当前任务的进度或诊断信息。
    elif args.command == "ddp":  # 选择 ddp 子命令的处理路径。
        print(run_ddp(args).to_string(index=False))  # 输出当前任务的进度或诊断信息。
    elif args.command == "validate":  # 选择 validate 子命令的处理路径。
        print(validate_outputs().to_string(index=False))  # 输出当前任务的进度或诊断信息。
    elif args.command == "report":  # 选择 report 子命令的处理路径。
        print(write_report())  # 输出当前任务的进度或诊断信息。


if __name__ == "__main__":  # 按当前模式、标识或状态选择处理分支。
    main()
