"""基线模型训练、独立 GPU 调度与样本外因子生成。"""
from __future__ import annotations

import contextlib
import json
import os
import random
import subprocess
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch
from torch import nn

from .data import (
    FORECAST_YEARS,
    SEEDS,
    PipelinePaths,
    atomic_json,
    atomic_parquet,
    load_split,
    make_data_loaders,
    stable_hash,
)
from .model import OrderFlowViViT, ViViTConfig, pearson_correlation, sigmoid_rank_ic_loss


@dataclass(frozen=True)
class TrainingConfig:
    # 仅保留基础 baseline 使用的训练参数，不引入额外模型或损失配置。
    learning_rate: float = 0.001
    max_epochs: int = 100
    patience: int = 10
    rank_temperature: float = 0.1
    inference_batch_size: int = 5000
    workers: int = 8
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.0
    improvement_tolerance: float = 1e-6
    use_compile: bool = True
    compile_mode: str = "default"
    use_fused_adam: bool = True
    use_cuda_prefetch: bool = True


MODEL_CONFIG = ViViTConfig()  # 全部任务共享唯一基础模型结构。


def set_random_seed(seed: int) -> None:
    # 同时固定 Python、NumPy 和 Torch，保证各 year×seed 任务可复现。
    random.seed(seed)  # 固定 Python 标准库随机序列。
    np.random.seed(seed)  # 固定 NumPy 数据处理随机序列。
    torch.manual_seed(seed)  # 固定 Torch CPU 与当前设备随机序列。
    if torch.cuda.is_available():  # 存在 CUDA 时同步固定所有可见设备。
        torch.cuda.manual_seed_all(seed)  # 保证多卡环境下初始化一致可复现。


def unwrap_model(model: nn.Module) -> nn.Module:
    # torch.compile 会包装原模型；保存权重时统一取内部原始模块。
    original = getattr(model, "_orig_mod", None)  # torch.compile 包装器把原模型保存在该属性。
    return unwrap_model(original) if original is not None else model  # 递归剥离包装直到真实模块。


def build_model(device: torch.device, config: TrainingConfig) -> nn.Module:
    model: nn.Module = OrderFlowViViT(MODEL_CONFIG).to(device)  # 实例化基础模型并移动到目标设备。
    if config.use_compile and device.type == "cuda":  # 仅 GPU 训练启用编译优化。
        # 动态模式允许不同交易日拥有不同横截面股票数。
        model = torch.compile(model, mode=config.compile_mode, dynamic=True)  # 允许每日横截面批大小变化。
    return model  # 返回普通或已编译模型，调用接口一致。


def build_optimizer(
    model: nn.Module, config: TrainingConfig, device: torch.device
) -> torch.optim.Adam:
    kwargs = {  # 显式固定 README baseline 的 Adam 参数。
        "lr": config.learning_rate,
        "betas": (config.adam_beta1, config.adam_beta2),
        "eps": config.adam_epsilon,
        "weight_decay": config.weight_decay,
    }
    if config.use_fused_adam and device.type == "cuda":  # fused 实现只在 CUDA 路径启用。
        # GPU 上使用 PyTorch 原生 fused Adam，CPU 自动回退普通实现。
        kwargs["fused"] = True  # 请求 PyTorch 合并优化器内核。
    return torch.optim.Adam(model.parameters(), **kwargs)  # 优化全部可训练参数。


def autocast_context(device: torch.device) -> contextlib.AbstractContextManager:
    # 仅在硬件明确支持时启用 BF16，避免额外的 GradScaler 状态。
    if device.type == "cuda" and torch.cuda.is_bf16_supported():  # 同时检查设备类型和硬件能力。
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)  # 前向使用 BF16、关键归约由框架保持精度。
    return contextlib.nullcontext()  # CPU 或不支持 BF16 时保持原精度。


def _move_batch(
    batch: dict[str, torch.Tensor], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # pinned memory 配合 non_blocking，把主机到 GPU 的拷贝与计算重叠。
    non_blocking = device.type == "cuda"  # 仅 CUDA 配合 pinned memory 支持异步拷贝。
    return (
        batch["videos"].to(device, non_blocking=non_blocking),
        batch["frame_mask"].to(device, non_blocking=non_blocking),
        batch["targets"].to(device, non_blocking=non_blocking),
    )


def iter_device_batches(
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    prefetch: bool,
) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]]:
    if not prefetch or device.type != "cuda":  # 预取关闭或非 GPU 时走简单同步路径。
        # CPU 或显式关闭预取时沿用普通同步迭代路径。
        for batch in loader:  # 逐批从 DataLoader 取 CPU 数据。
            yield batch, *_move_batch(batch, device)  # 同步移动训练张量并保留 CPU 元数据。
        return  # 避免继续创建 CUDA stream。

    iterator = iter(loader)  # 手动控制迭代器以提前拉取下一批。
    # 独立 CUDA stream 预加载下一批，实现最小的双缓冲。
    stream = torch.cuda.Stream(device=device)  # 专用流负责主机到设备传输。
    next_item = None  # 保存已经发起拷贝的下一批。

    def preload() -> None:
        nonlocal next_item
        try:  # 尝试从 DataLoader 取得下一 CPU 批次。
            batch = next(iterator)  # 推进底层 worker 队列。
        except StopIteration:  # 数据耗尽后清空预取槽。
            next_item = None  # 通知外层循环结束。
            return  # 不再提交 CUDA 拷贝。
        with torch.cuda.stream(stream):  # 在预取流异步移动张量。
            # batch 元数据留在 CPU，仅视频、掩码和标签进入设备。
            next_item = (batch, *_move_batch(batch, device))  # CPU 元数据与设备张量保持同一批对应。

    preload()  # 循环开始前先填充第一个预取槽。
    while next_item is not None:  # 直到 DataLoader 完全耗尽。
        # 当前计算流必须等待预取流完成本批数据拷贝。
        torch.cuda.current_stream(device).wait_stream(stream)  # 保证当前批拷贝完成后再计算。
        current = next_item  # 固定本轮要返回的批次。
        for tensor in current[1:]:  # 三个设备张量都登记当前计算流。
            # 告知缓存分配器张量仍被当前流使用，防止显存过早复用。
            tensor.record_stream(torch.cuda.current_stream(device))  # 防止缓存分配器过早回收显存。
        preload()  # 当前批计算前立即发起下一批拷贝。
        yield current  # 向训练循环交付已经就绪的批次。


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: TrainingConfig,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.train()  # 启用 dropout 等训练态行为。
    losses = []  # 累计每个交易日批次损失。
    correlations = []  # 累计用于监控的原始 Pearson IC。
    for batch_number, (_, videos, frame_mask, targets) in enumerate(
        iter_device_batches(loader, device, config.use_cuda_prefetch), start=1
    ):  # 每次迭代对应一个完整交易日横截面。
        # 每批是一个完整交易日横截面，排序损失不会混合不同日期。
        if not torch.isfinite(targets).all():  # 训练切分应只含有效 z-score 标签。
            raise ValueError("training batch contains invalid targets")  # 避免 NaN 污染全部梯度。
        optimizer.zero_grad(set_to_none=True)  # 释放旧梯度张量并降低显存写入。
        with autocast_context(device):  # 在支持设备上对前向启用 BF16。
            predictions = model(videos, frame_mask)  # 生成当日所有股票原始分数。
            loss = sigmoid_rank_ic_loss(predictions, targets, config.rank_temperature)  # 优化横截面软 RankIC。
        if not torch.isfinite(loss):  # 反向前检查数值稳定性。
            raise FloatingPointError("non-finite training loss")  # 防止无效梯度破坏模型。
        loss.backward()  # 计算模型参数梯度。
        optimizer.step()  # 用当前日梯度更新一次参数。
        # Pearson IC 仅作训练监控；优化目标仍是 sigmoid RankIC。
        losses.append(float(loss.detach().cpu()))  # 将标量移到 CPU 避免保留计算图。
        correlations.append(float(pearson_correlation(predictions.detach(), targets).cpu()))  # 记录非优化监控指标。
        if max_batches is not None and batch_number >= max_batches:  # 冒烟测试可限制处理批数。
            break  # 达到上限立即结束本轮。
    if not losses:  # 空训练集无法产生有效模型更新。
        raise ValueError("training loader produced no batches")  # 将数据切分问题提前暴露。
    return {"train_loss": float(np.mean(losses)), "train_pearson_ic": float(np.mean(correlations))}  # 返回日批次等权均值。


def _descending_ranks(values: np.ndarray) -> np.ndarray:
    # average 处理并列值，与标准 Spearman 的秩定义一致。
    return pd.Series(values).rank(method="average", ascending=False).to_numpy(dtype=np.float64)  # 分数越高秩越靠前。


def _numpy_pearson(left: np.ndarray, right: np.ndarray) -> float:
    # 小样本或常数向量没有有效相关系数，交给日均统计忽略。
    left = np.asarray(left, dtype=np.float64)  # 用双精度降低验证统计误差。
    right = np.asarray(right, dtype=np.float64)  # 标签转换为同精度数组。
    if len(left) < 2 or left.std(ddof=0) <= 0 or right.std(ddof=0) <= 0:  # 相关系数要求两侧有变异。
        return np.nan  # 无定义日期交由日均聚合忽略。
    return float(np.corrcoef(left, right)[0, 1])  # 取标准相关矩阵的非对角元素。


def summarize_validation(
    predictions: np.ndarray, targets: np.ndarray, date_ids: np.ndarray
) -> dict[str, float]:
    daily_rank = []  # 保存逐日 Spearman IC。
    daily_pearson = []  # 保存逐日原始分数 Pearson IC。
    for date_id in np.unique(date_ids):  # 每个交易日独立评价横截面。
        # 先逐日计算，再对日期等权平均，避免大横截面日期权重过高。
        mask = date_ids == date_id  # 选出当前日全部股票。
        daily_rank.append(_numpy_pearson(  # Spearman 等价于两侧秩的 Pearson。
            _descending_ranks(predictions[mask]), _descending_ranks(targets[mask])
        ))
        daily_pearson.append(_numpy_pearson(predictions[mask], targets[mask]))  # 同时保留未排序相关作诊断。
    rank_ic = float(np.nanmean(daily_rank))  # 对交易日等权平均主指标。
    pearson_ic = float(np.nanmean(daily_pearson))  # 对交易日等权平均原始 IC。
    pooled_ic = _numpy_pearson(predictions, targets)  # 计算跨日混合相关仅供对照。
    if not np.isfinite(rank_ic):  # 主早停指标必须有效。
        raise FloatingPointError("validation daily Spearman IC is not finite")  # 阻止错误指标参与模型选择。
    return {
        # validation_ic 是早停使用的唯一主指标，即日均 Spearman IC。
        "validation_ic": rank_ic,
        "validation_daily_spearman_ic": rank_ic,
        "validation_daily_pearson_ic": pearson_ic,
        "validation_pooled_pearson_ic": pooled_ic,
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    config: TrainingConfig,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()  # 关闭 dropout 并采用推理行为。
    # 分批推理后汇总到 CPU，保证指标在完整验证集上统一计算。
    predictions = []  # 累计各批预测到 CPU。
    targets = []  # 累计对应验证标签。
    date_ids = []  # 累计日期编号供逐日统计。
    for batch_number, (batch, videos, frame_mask, target) in enumerate(
        iter_device_batches(loader, device, config.use_cuda_prefetch), start=1
    ):  # 验证可以按固定样本数拆批，不要求完整日期同批。
        with autocast_context(device):  # 使用与训练一致的推理精度。
            prediction = model(videos, frame_mask)  # 计算当前验证批分数。
        predictions.append(prediction.float().cpu().numpy())  # 转 FP32 并释放 GPU 张量引用。
        targets.append(target.float().cpu().numpy())  # 保存对应标签。
        date_ids.append(batch["date_ids"].numpy())  # 日期编号始终保留在 CPU。
        if max_batches is not None and batch_number >= max_batches:  # 冒烟测试限制验证批数。
            break  # 达到上限后停止。
    if not predictions:  # 空验证集无法用于早停。
        raise ValueError("validation loader produced no batches")  # 报告切分或加载异常。
    return summarize_validation(  # 拼接完整验证集后统一计算逐日指标。
        np.concatenate(predictions), np.concatenate(targets), np.concatenate(date_ids)
    )


@torch.no_grad()
def predict_model(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    config: TrainingConfig,
) -> pd.DataFrame:
    model.eval()  # 固定推理模式。
    indices = []  # 保存预测对应的全局 sample_index。
    predictions = []  # 保存模型原始分数。
    for batch, videos, frame_mask, _ in iter_device_batches(  # 顺序遍历 OOS 样本。
        loader, device, config.use_cuda_prefetch
    ):
        with autocast_context(device):  # 使用硬件支持的 BF16 推理。
            prediction = model(videos, frame_mask)  # 生成当前批 OOS 原始分数。
        indices.append(batch["sample_indices"].numpy())  # 记录结果回连键。
        predictions.append(prediction.float().cpu().numpy())  # 统一输出为 FP32 CPU 数组。
    # 重新按 sample_index 排序，使输出与预测切分的原始顺序严格一致。
    frame = pd.DataFrame({  # 将分批结果恢复为单一预测表。
        "sample_index": np.concatenate(indices).astype(np.int64),
        "raw_score": np.concatenate(predictions).astype(np.float32),
    }).sort_values("sample_index", kind="stable")
    if frame["sample_index"].duplicated().any() or not np.isfinite(frame["raw_score"]).all():  # 主键唯一且分数必须有限。
        raise ValueError("invalid prediction output")  # 拒绝不可交付预测。
    return frame  # 返回与预测切分顺序一致的结果。


def _atomic_checkpoint(payload: dict[str, object], path: Path) -> None:
    # checkpoint 可能很大，仍先写临时文件再原子替换。
    path.parent.mkdir(parents=True, exist_ok=True)  # 创建模型运行目录。
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")  # 隔离未写完 checkpoint。
    torch.save(payload, temporary)  # 序列化模型、优化器和随机状态。
    temporary.replace(path)  # 原子替换正式 checkpoint。


def _run_signature(
    paths: PipelinePaths, year: int, seed: int, config: TrainingConfig
) -> tuple[str, dict[str, object]]:
    # 数据、年份、随机种子、模型和训练参数共同定义一次可恢复运行。
    data_manifest = json.loads((paths.feature_dir / "manifest.json").read_text(encoding="utf-8"))  # 读取训练输入版本。
    payload = {  # 将全部影响结果的配置纳入运行身份。
        "data_signature": data_manifest["signature"],
        "forecast_year": year,
        "seed": seed,
        "model": asdict(MODEL_CONFIG),
        "training": asdict(config),
        "loss": "sigmoid_rank_ic",
        "validation": "daily_mean_spearman_ic",
    }
    return stable_hash(payload), payload  # 同时返回摘要与可审计原始配置。


def train_task(
    paths: PipelinePaths,
    forecast_year: int,
    seed: int,
    physical_gpu: int = 0,
    config: TrainingConfig = TrainingConfig(),
    overwrite: bool = False,
) -> dict[str, object]:
    if forecast_year not in FORECAST_YEARS or seed not in SEEDS:  # 只允许预定义滚动年份和三个基础种子。
        raise ValueError("unsupported forecast year or seed")  # 拒绝无对应数据切分的任务。
    set_random_seed(seed)  # 在模型和 DataLoader 创建前固定全部随机源。
    # 父进程通过 CUDA_VISIBLE_DEVICES 隔离物理卡，子进程内始终使用 cuda:0。
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 子进程内选择唯一可见 GPU 或 CPU。
    train_loader, validation_loader, prediction_loader = make_data_loaders(  # 创建当前年份和种子的三阶段加载器。
        paths, forecast_year, seed, workers=config.workers,
        inference_batch_size=config.inference_batch_size,
        pin_memory=device.type == "cuda",
    )
    run_dir = paths.models_dir / "baseline" / f"forecast_{forecast_year}" / f"seed_{seed}"  # 每项任务使用独立目录。
    run_dir.mkdir(parents=True, exist_ok=True)  # 允许断点文件和日志持久化。
    best_path = run_dir / "best.pt"  # 保存验证 IC 最优权重。
    last_path = run_dir / "last.pt"  # 保存最近完整 epoch 供恢复。
    history_path = run_dir / "history.parquet"  # 保存逐轮训练与验证指标。
    prediction_path = run_dir / "prediction.parquet"  # 保存该 seed 的 OOS 原始分数。
    completed_path = run_dir / "completed.json"  # 标记整个任务已完成且可复用。
    signature, payload = _run_signature(paths, forecast_year, seed, config)  # 计算本任务唯一运行签名。

    if completed_path.exists() and best_path.exists() and not overwrite:  # 已完成任务默认不重复训练。
        # 已完成且签名一致时直接复用；缺预测文件则只补一次推理。
        completed = json.loads(completed_path.read_text(encoding="utf-8"))  # 读取完成标记及旧签名。
        if completed.get("signature") != signature:  # 数据或配置变化时不能复用旧模型。
            raise ValueError(f"configuration changed for {run_dir}; use --overwrite")  # 要求显式确认重训。
        if prediction_loader is not None and not prediction_path.exists():  # 训练完成但预测文件丢失时只补推理。
            checkpoint = torch.load(best_path, map_location=device, weights_only=False)  # 加载最优模型状态。
            model = build_model(device, config)  # 按相同配置重建模型结构。
            unwrap_model(model).load_state_dict(checkpoint["model_state"])  # 将权重载入未编译原模块。
            atomic_parquet(predict_model(model, prediction_loader, device, config), prediction_path)  # 原子补写 OOS 分数。
        return completed["summary"]  # 返回已完成任务摘要供调度器汇总。

    model = build_model(device, config)  # 创建新训练模型。
    optimizer = build_optimizer(model, config, device)  # 创建与 baseline 一致的 Adam。
    start_epoch = 1  # 无 checkpoint 时从第一轮开始。
    best_epoch = 0  # 尚未产生最优轮次。
    best_validation_ic = -np.inf  # 首个有效验证结果必然成为最优。
    epochs_without_improvement = 0  # 初始化早停计数器。
    history: list[dict[str, object]] = []  # 保存当前任务所有已完成 epoch 指标。
    if last_path.exists() and not overwrite:  # 存在最近 checkpoint 时尝试精确续训。
        # last.pt 用于断点续训，签名不同则拒绝静默混用旧状态。
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)  # 先在 CPU 反序列化降低设备峰值。
        if checkpoint.get("signature") != signature:  # 断点必须属于同一数据和配置。
            raise ValueError(f"checkpoint configuration changed for {run_dir}; use --overwrite")  # 禁止混用不兼容状态。
        unwrap_model(model).load_state_dict(checkpoint["model_state"])  # 恢复模型参数。
        optimizer.load_state_dict(checkpoint["optimizer_state"])  # 恢复 Adam 动量和步数。
        # 恢复所有随机状态，保证中断续训与连续训练的批次顺序一致。
        random.setstate(checkpoint["python_random_state"])  # 恢复 Python 随机序列位置。
        np.random.set_state(checkpoint["numpy_random_state"])  # 恢复 NumPy 随机序列位置。
        torch.set_rng_state(checkpoint["torch_rng_state"])  # 恢复 Torch CPU 随机状态。
        if device.type == "cuda" and checkpoint["cuda_rng_state"] is not None:  # CUDA 续训还需恢复设备随机状态。
            torch.cuda.set_rng_state(checkpoint["cuda_rng_state"], device)  # 保持 dropout 等随机行为连续。
        start_epoch = int(checkpoint["epoch"]) + 1  # 从最近完整轮次的下一轮继续。
        best_epoch = int(checkpoint["best_epoch"])  # 恢复历史最优轮次。
        best_validation_ic = float(checkpoint["best_validation_ic"])  # 恢复早停比较基准。
        epochs_without_improvement = int(checkpoint["epochs_without_improvement"])  # 恢复耐心计数。
        if history_path.exists():  # 历史表存在时接续而非覆盖旧记录。
            history = pd.read_parquet(history_path).to_dict("records")  # 转回可追加记录列表。
        if epochs_without_improvement >= config.patience:  # 中断前已经满足早停条件。
            # 已触发早停的 checkpoint 不再重复训练，直接进入最佳模型推理。
            start_epoch = config.max_epochs + 1  # 跳过训练循环并直接生成预测。

    started = time.perf_counter()  # 记录本次进程实际运行耗时。
    for epoch in range(start_epoch, config.max_epochs + 1):  # 最多训练配置指定轮数。
        # sampler 以 seed+epoch 打乱“日期”，不打乱日内股票。
        train_loader.batch_sampler.set_epoch(epoch)  # 用 seed+epoch 生成本轮日期顺序。
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, config)  # 完成一轮日横截面训练。
        validation_metrics = evaluate_model(model, validation_loader, device, config)  # 在完整验证集计算逐日指标。
        validation_ic = validation_metrics["validation_ic"]  # 提取日均 Spearman 作为模型选择指标。
        improved = validation_ic > best_validation_ic + config.improvement_tolerance  # 只有超过容差才视为真实提升。
        if improved:  # 更新最优模型和重置早停计数。
            best_epoch = epoch  # 记录当前最优轮次。
            best_validation_ic = validation_ic  # 更新最佳验证 IC。
            epochs_without_improvement = 0  # 提升后重新计算耐心窗口。
        else:  # 未达到提升阈值。
            epochs_without_improvement += 1  # 累计连续未提升轮数。
        history.append({  # 记录本轮全部训练与验证指标。
            "forecast_year": forecast_year, "seed": seed, "epoch": epoch,
            **train_metrics, **validation_metrics,
            "best_validation_ic": best_validation_ic,
            "epochs_without_improvement": epochs_without_improvement,
        })
        atomic_parquet(pd.DataFrame(history), history_path)  # 每轮原子更新可审计训练历史。
        # last 每轮覆盖用于恢复，best 只在验证 IC 真正提升时更新。
        checkpoint = {  # 组合恢复训练所需的全部状态。
            "signature": signature, "payload": payload,
            "model_state": unwrap_model(model).state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
            "epoch": epoch, "best_epoch": best_epoch,
            "best_validation_ic": best_validation_ic,
            "epochs_without_improvement": epochs_without_improvement,
        }
        _atomic_checkpoint(checkpoint, last_path)  # 无论是否提升都保存最近完整轮次。
        if improved:  # 仅验证指标提升时更新最优权重。
            _atomic_checkpoint(checkpoint, best_path)  # 保存供最终 OOS 推理使用的 checkpoint。
        print(
            f"[{forecast_year}/seed={seed}] epoch={epoch:03d} "
            f"loss={train_metrics['train_loss']:.6f} "
            f"valid_spearman={validation_ic:.6f} "
            f"patience={epochs_without_improvement}/{config.patience}",
            flush=True,
        )
        if epochs_without_improvement >= config.patience:  # 连续未提升达到耐心阈值。
            break  # 提前停止后续训练轮次。

    if not best_path.exists():  # 至少一个 epoch 应产生最优 checkpoint。
        raise FileNotFoundError(f"best checkpoint not created: {best_path}")  # 防止用未训练模型生成因子。
    if prediction_loader is not None:  # 目标年存在 OOS 样本时执行推理。
        # OOS 分数必须来自验证集最优 checkpoint，而不是最后一轮权重。
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)  # 加载验证最优状态。
        best_model = build_model(device, config)  # 重建干净推理模型。
        unwrap_model(best_model).load_state_dict(checkpoint["model_state"])  # 写入最优权重。
        atomic_parquet(predict_model(best_model, prediction_loader, device, config), prediction_path)  # 保存该 seed OOS 分数。
    summary = {  # 汇总调度与审计所需任务结果。
        "forecast_year": forecast_year,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_ic": best_validation_ic,
        "last_epoch": int(history[-1]["epoch"]),
        "physical_gpu": physical_gpu,
        "execution_route": "independent_gpu_tasks",
        "prediction_samples": len(load_split(paths, forecast_year)["prediction_indices"]),
        "elapsed_seconds": time.perf_counter() - started,
        "status": "completed",
    }
    atomic_json({"signature": signature, "summary": summary}, completed_path)  # 最后写完成标记，保证预测已落盘。
    return summary  # 返回父调度器汇总。


def build_oos_factor(paths: PipelinePaths, forecast_year: int) -> int:
    split = load_split(paths, forecast_year)  # 读取该年度期望预测样本顺序。
    if not len(split["prediction_indices"]):  # 无 OOS 样本的年度无需创建空因子文件。
        return 0  # 向调度器报告零行。
    seed_frames = {}  # 保存三个 seed 的预测表。
    model_root = paths.models_dir / "baseline" / f"forecast_{forecast_year}"  # 定位年度模型目录。
    for seed in SEEDS:  # 三个基础 seed 必须全部完成后才可集成。
        seed_frames[seed] = pd.read_parquet(model_root / f"seed_{seed}" / "prediction.parquet")  # 读取单 seed 原始分数。
    reference = seed_frames[SEEDS[0]]["sample_index"].to_numpy(np.int64)  # 以首个 seed 的样本顺序为基准。
    # 三个种子的 sample_index 必须逐行一致，禁止按位置误合并不同股票。
    output = pd.DataFrame({"sample_index": reference})  # 先建立唯一主键列。
    raw_columns = []  # 记录三个原始分数列名供统一标准化。
    for seed in SEEDS:  # 逐 seed 校验并横向拼接。
        frame = seed_frames[seed]  # 取得当前 seed 预测。
        if not np.array_equal(reference, frame["sample_index"].to_numpy(np.int64)):  # 样本键和顺序必须完全相同。
            raise ValueError(f"seed prediction keys differ in {forecast_year}")  # 禁止错位集成不同股票。
        column = f"seed_{seed}_raw_score"  # 为原始分数生成可审计列名。
        output[column] = frame["raw_score"].to_numpy(np.float32)  # 按已验证顺序写入分数。
        raw_columns.append(column)  # 加入后续逐日统计列集合。
    metadata = pd.read_parquet(  # 读取因子交付所需日期与股票代码。
        paths.sample_index_path, columns=["sample_index", "trade_date", "instrument_id"]
    ).set_index("sample_index")
    output = metadata.loc[reference].reset_index().merge(output, on="sample_index", validate="one_to_one")  # 按主键一对一补齐元数据。
    raw = output[raw_columns].astype(np.float64)  # 用双精度计算每日标准化统计量。
    # 每个种子先按日独立 z-score，再等权平均，避免种子尺度差异。
    grouped = raw.groupby(output["trade_date"], sort=False)  # 每日横截面独立处理三个 seed。
    means = grouped.transform("mean")  # 将每日列均值广播回每只股票。
    standard_deviations = grouped.transform("std", ddof=0)  # 使用总体标准差匹配标签口径。
    if (standard_deviations <= 0).any().any():  # 常数分数无法形成有效截面排序。
        raise ValueError(f"constant daily seed score in {forecast_year}")  # 拒绝除零或无区分度因子。
    zscore_columns = []  # 保存标准化 seed 列名供最终等权平均。
    for column in raw_columns:  # 每个 seed 使用自己的日均值和标准差。
        zscore_column = column.replace("_raw_score", "_zscore")  # 生成对应标准化列名。
        output[zscore_column] = ((raw[column] - means[column]) / standard_deviations[column]).astype(np.float32)  # 计算逐日 z-score。
        zscore_columns.append(zscore_column)  # 纳入最终集成列集合。
    output["raw_score_mean"] = raw.mean(axis=1).astype(np.float32)  # 保留未标准化均值用于审计。
    # factor_value 是最终交付因子；raw_score_mean 仅保留作审计。
    output["factor_value"] = output[zscore_columns].mean(axis=1).astype(np.float32)  # 三个 seed 标准化分数等权集成。
    output["forecast_year"] = forecast_year  # 显式记录模型对应 OOS 年份。
    output["factor_role"] = "full_year_oos"  # 标注该文件只含全年样本外分数。
    atomic_parquet(output, paths.factors_dir / f"vivit_oos_factor.{forecast_year}.parquet")  # 原子输出年度交付因子。
    return len(output)  # 返回实际因子行数供训练清单记录。


def validate_outputs(paths: PipelinePaths) -> pd.DataFrame:
    records = []  # 收集每个 OOS 年的交付校验结果。
    for year in FORECAST_YEARS:  # 所有配置年份必须逐一检查。
        split = load_split(paths, year)  # 读取预测索引作为覆盖率真值。
        expected = len(split["prediction_indices"])  # 计算应交付因子行数。
        for seed in SEEDS:  # 因子依赖的三个训练任务都必须完成。
            completed = paths.models_dir / "baseline" / f"forecast_{year}" / f"seed_{seed}" / "completed.json"  # 定位完成标记。
            if not completed.exists():  # 缺标记说明该 seed 结果不可交付。
                raise FileNotFoundError(completed)  # 报出具体缺失任务。
        factor_path = paths.factors_dir / f"vivit_oos_factor.{year}.parquet"  # 定位年度因子文件。
        if expected == 0:  # 目标年没有图片样本时允许不生成文件。
            records.append({"forecast_year": year, "factor_rows": 0, "status": "no prediction rows"})  # 记录合法空年度。
            continue  # 跳过文件和标准化检查。
        factor = pd.read_parquet(factor_path)  # 读取非空年度交付因子。
        # 同时校验覆盖率、唯一键、顺序和数值有限性。
        if len(factor) != expected or factor.duplicated(["trade_date", "instrument_id"]).any():  # 行数和股票日期主键必须正确。
            raise AssertionError(f"factor coverage failed for {year}")  # 报告覆盖或重复问题。
        if not np.array_equal(factor["sample_index"].to_numpy(np.int64), split["prediction_indices"]):  # 顺序必须与预测切分一致。
            raise AssertionError(f"factor order failed for {year}")  # 防止因子错配到其他股票。
        if not np.isfinite(factor["factor_value"]).all():  # 最终因子不允许 NaN 或无穷值。
            raise AssertionError(f"non-finite factor values for {year}")  # 拒绝不可用交付结果。
        for seed in SEEDS:  # 分别复核三个 seed 的每日 z-score。
            # 每个种子的日截面均值应为 0、总体标准差应为 1。
            grouped = factor.groupby("trade_date")[f"seed_{seed}_zscore"]  # 按日提取当前 seed 标准化分数。
            if grouped.mean().abs().max() > 1e-5 or (grouped.std(ddof=0) - 1).abs().max() > 1e-5:  # 检查均值零和总体标准差一。
                raise AssertionError(f"daily seed standardization failed for {year}/{seed}")  # 报告具体失败年份和种子。
        records.append({"forecast_year": year, "factor_rows": len(factor), "status": "passed"})  # 记录通过年度。
    return pd.DataFrame(records)  # 返回 Notebook 可直接展示的校验表。


def _tail(path: Path, lines: int = 80) -> str:
    # 子任务失败时只回显日志尾部，避免把超长训练日志塞入异常。
    if not path.exists():  # 日志尚未创建时没有可展示内容。
        return ""  # 返回空字符串供异常拼接。
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])  # 只取末尾指定行数。


def run_parallel_training(
    paths: PipelinePaths,
    device_ids: tuple[int, ...] = (0, 1, 2, 3),
    tasks_per_gpu: int = 1,
    workers: int = 8,
    overwrite: bool = False,
) -> pd.DataFrame:
    if not device_ids or len(set(device_ids)) != len(device_ids) or tasks_per_gpu < 1:  # GPU 列表须非空唯一且并发数为正。
        raise ValueError("GPU IDs must be non-empty/unique and tasks_per_gpu must be positive")  # 拒绝歧义调度配置。
    if torch.cuda.device_count() <= max(device_ids):  # 最大物理编号必须落在可见设备范围内。
        raise RuntimeError(f"requested GPUs {device_ids}, available count={torch.cuda.device_count()}")  # 启动任务前报告设备不足。
    tasks = deque((year, seed) for year in FORECAST_YEARS for seed in SEEDS)  # 按年份优先、种子次序建立 FIFO 队列。
    # 任务是独立的 forecast_year×seed 组合，不共享模型或优化器状态。
    log_dir = paths.logs_dir / "parallel_task_logs"  # 每个子任务使用独立日志文件。
    log_dir.mkdir(parents=True, exist_ok=True)  # 创建调度日志目录。
    running: list[dict[str, object]] = []  # 保存活跃子进程及其资源句柄。
    completed_seeds = {year: set() for year in FORECAST_YEARS}  # 跟踪年度三 seed 聚合条件。
    factor_counts: dict[str, int] = {}  # 记录已即时输出的年度因子行数。
    started = time.perf_counter()  # 记录整个并行阶段墙钟时间。
    try:  # 统一回收异常情况下的所有活跃子进程。
        while tasks or running:  # 队列和活跃列表都为空时训练才结束。
            for gpu in device_ids:  # 逐张物理 GPU 填充空闲槽位。
                # 每张物理卡最多同时运行 tasks_per_gpu 个子进程。
                active = sum(item["gpu"] == gpu for item in running)  # 统计该卡当前活跃任务数。
                while tasks and active < tasks_per_gpu:  # 有排队任务且 GPU 尚有容量时继续启动。
                    year, seed = tasks.popleft()  # 从 FIFO 取下一个年度种子组合。
                    log_path = log_dir / f"forecast_{year}.seed_{seed}.log"  # 为任务生成唯一日志路径。
                    log_handle = log_path.open("a", encoding="utf-8")  # 追加模式保留断点前历史日志。
                    command = [  # 子进程复用同一 CLI train-task 入口。
                        sys.executable, "-m", "vivit_factor", "train-task",
                        "--forecast-year", str(year), "--seed", str(seed),
                        "--physical-gpu", str(gpu), "--workers", str(workers),
                    ]
                    if overwrite:  # 用户明确要求重训时向子任务透传覆盖标记。
                        command.append("--overwrite")  # 禁用完成缓存和断点复用。
                    environment = os.environ.copy()  # 继承父进程依赖和路径环境。
                    # 子进程只看见被分配的一张卡，内部设备编号因此固定为 0。
                    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)  # 将物理卡隔离为子进程 cuda:0。
                    process = subprocess.Popen(  # 非阻塞启动独立训练进程。
                        command, cwd=paths.project_dir, env=environment,
                        stdout=log_handle, stderr=subprocess.STDOUT, text=True,
                    )
                    running.append({  # 保存轮询、错误报告和资源回收所需状态。
                        "process": process, "gpu": gpu, "year": year, "seed": seed,
                        "log_path": log_path, "log_handle": log_handle,
                    })
                    active += 1  # 更新当前 GPU 已占用槽位数。
                    print(f"launch GPU{gpu}: {year}/seed={seed} pid={process.pid}", flush=True)  # 实时输出启动信息。
            finished = []  # 收集本轮检测到的结束任务。
            for item in running:  # 非阻塞检查所有活跃进程。
                # 非阻塞轮询让空闲 GPU 能立即领取下一项任务。
                return_code = item["process"].poll()  # None 表示进程仍在运行。
                if return_code is None:  # 活跃任务留待下一轮检查。
                    continue  # 不占用父进程等待。
                item["log_handle"].close()  # 子进程结束后关闭父进程日志句柄。
                finished.append(item)  # 标记稍后从 running 移除。
                if return_code:  # 非零退出码表示该训练任务失败。
                    # 首个失败立即终止调度，并携带该任务日志尾部。
                    raise RuntimeError(  # 携带任务身份和日志尾部终止整体调度。
                        f"task failed {item['year']}/seed={item['seed']} on GPU{item['gpu']}\n"
                        f"{_tail(item['log_path'])}"
                    )
                year = int(item["year"])  # 读取完成任务所属预测年。
                completed_seeds[year].add(int(item["seed"]))  # 登记该年度已完成 seed。
                print(f"done GPU{item['gpu']}: {item['year']}/seed={item['seed']}", flush=True)  # 实时输出完成信息。
                # 某年度三个 seed 全部完成后立即聚合，其他年度继续训练。
                if completed_seeds[year] == set(SEEDS):  # 三个指定 seed 集齐时立即触发年度集成。
                    factor_counts[str(year)] = build_oos_factor(paths, year)  # 输出该年度 OOS 因子。
                    print(f"factor {year}: {factor_counts[str(year)]:,} rows", flush=True)  # 报告即时产物行数。
            for item in finished:  # 释放所有本轮已结束任务的调度槽位。
                running.remove(item)  # 下一循环可在相应 GPU 启动新任务。
            if running and not finished:  # 有任务运行但本轮没有状态变化。
                time.sleep(1)  # 限制轮询频率，避免父进程空转。
    except BaseException:  # KeyboardInterrupt 和任务异常都进入统一清理。
        # 主进程异常时回收仍运行的子进程，避免后台继续占用 GPU。
        for item in running:  # 遍历仍登记的所有子任务。
            if item["process"].poll() is None:  # 只终止尚未退出的进程。
                item["process"].terminate()  # 发送可处理的终止信号而非强制杀死。
            item["log_handle"].close()  # 无论进程状态都释放日志句柄。
        raise  # 保留原始异常和堆栈给调用方。

    summaries = []  # 收集全部成功子任务摘要。
    for year in FORECAST_YEARS:  # 按固定年度顺序汇总。
        for seed in SEEDS:  # 每年按固定 seed 顺序读取。
            completed_path = paths.models_dir / "baseline" / f"forecast_{year}" / f"seed_{seed}" / "completed.json"  # 定位任务完成清单。
            summaries.append(json.loads(completed_path.read_text(encoding="utf-8"))["summary"])  # 提取任务级统计。
    # 因子已随年度完成而生成；全部任务结束后只汇总清单并执行交付校验。
    summary = pd.DataFrame(summaries).sort_values(["forecast_year", "seed"], kind="stable")  # 形成确定顺序训练汇总表。
    atomic_parquet(summary, paths.artifacts_dir / "training_summary.parquet")  # 原子保存所有任务指标。
    atomic_json({  # 保存调度配置、模型参数和年度因子行数。
        "route": "independent_gpu_tasks",
        "device_ids": list(device_ids),
        "tasks_per_gpu": tasks_per_gpu,
        "model_config": asdict(MODEL_CONFIG),
        "training_config": asdict(TrainingConfig(workers=workers)),
        "years": list(FORECAST_YEARS),
        "seeds": list(SEEDS),
        "factor_counts": factor_counts,
        "wall_seconds": time.perf_counter() - started,
    }, paths.artifacts_dir / "training_manifest.json")
    validate_outputs(paths)  # 最终统一复核所有年度交付因子。
    return summary  # 返回 Notebook 展示用训练表。


def smoke_test(
    paths: PipelinePaths, forecast_year: int = 2018, seed: int = 42
) -> dict[str, float]:
    # 只跑一个训练批和一个验证批，用最低成本检查端到端张量与梯度。
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 优先检查真实 GPU 路径。
    config = TrainingConfig(workers=0, use_compile=False, use_cuda_prefetch=False)  # 关闭并行优化以缩短诊断路径。
    set_random_seed(seed)  # 固定冒烟测试结果。
    train_loader, validation_loader, _ = make_data_loaders(  # 创建最小训练和验证输入。
        paths, forecast_year, seed, workers=0, inference_batch_size=512,
        pin_memory=device.type == "cuda",
    )
    model = build_model(device, config)  # 实例化未经编译的 baseline 模型。
    optimizer = build_optimizer(model, config, device)  # 创建与正式训练相同优化器。
    train_metrics = train_one_epoch(model, train_loader, optimizer, device, config, max_batches=1)  # 验证一批前向和反向。
    validation_metrics = evaluate_model(model, validation_loader, device, config, max_batches=1)  # 验证一批推理和指标。
    return {**train_metrics, **validation_metrics}  # 合并结果供命令行或 Notebook 展示。
