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


MODEL_CONFIG = ViViTConfig()


def set_random_seed(seed: int) -> None:
    # 同时固定 Python、NumPy 和 Torch，保证各 year×seed 任务可复现。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unwrap_model(model: nn.Module) -> nn.Module:
    # torch.compile 会包装原模型；保存权重时统一取内部原始模块。
    original = getattr(model, "_orig_mod", None)
    return unwrap_model(original) if original is not None else model


def build_model(device: torch.device, config: TrainingConfig) -> nn.Module:
    model: nn.Module = OrderFlowViViT(MODEL_CONFIG).to(device)
    if config.use_compile and device.type == "cuda":
        # 动态模式允许不同交易日拥有不同横截面股票数。
        model = torch.compile(model, mode=config.compile_mode, dynamic=True)
    return model


def build_optimizer(
    model: nn.Module, config: TrainingConfig, device: torch.device
) -> torch.optim.Adam:
    kwargs = {
        "lr": config.learning_rate,
        "betas": (config.adam_beta1, config.adam_beta2),
        "eps": config.adam_epsilon,
        "weight_decay": config.weight_decay,
    }
    if config.use_fused_adam and device.type == "cuda":
        # GPU 上使用 PyTorch 原生 fused Adam，CPU 自动回退普通实现。
        kwargs["fused"] = True
    return torch.optim.Adam(model.parameters(), **kwargs)


def autocast_context(device: torch.device) -> contextlib.AbstractContextManager:
    # 仅在硬件明确支持时启用 BF16，避免额外的 GradScaler 状态。
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _move_batch(
    batch: dict[str, torch.Tensor], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # pinned memory 配合 non_blocking，把主机到 GPU 的拷贝与计算重叠。
    non_blocking = device.type == "cuda"
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
    if not prefetch or device.type != "cuda":
        # CPU 或显式关闭预取时沿用普通同步迭代路径。
        for batch in loader:
            yield batch, *_move_batch(batch, device)
        return

    iterator = iter(loader)
    # 独立 CUDA stream 预加载下一批，实现最小的双缓冲。
    stream = torch.cuda.Stream(device=device)
    next_item = None

    def preload() -> None:
        nonlocal next_item
        try:
            batch = next(iterator)
        except StopIteration:
            next_item = None
            return
        with torch.cuda.stream(stream):
            # batch 元数据留在 CPU，仅视频、掩码和标签进入设备。
            next_item = (batch, *_move_batch(batch, device))

    preload()
    while next_item is not None:
        # 当前计算流必须等待预取流完成本批数据拷贝。
        torch.cuda.current_stream(device).wait_stream(stream)
        current = next_item
        for tensor in current[1:]:
            # 告知缓存分配器张量仍被当前流使用，防止显存过早复用。
            tensor.record_stream(torch.cuda.current_stream(device))
        preload()
        yield current


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: TrainingConfig,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.train()
    losses = []
    correlations = []
    for batch_number, (_, videos, frame_mask, targets) in enumerate(
        iter_device_batches(loader, device, config.use_cuda_prefetch), start=1
    ):
        # 每批是一个完整交易日横截面，排序损失不会混合不同日期。
        if not torch.isfinite(targets).all():
            raise ValueError("training batch contains invalid targets")
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device):
            predictions = model(videos, frame_mask)
            loss = sigmoid_rank_ic_loss(predictions, targets, config.rank_temperature)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite training loss")
        loss.backward()
        optimizer.step()
        # Pearson IC 仅作训练监控；优化目标仍是 sigmoid RankIC。
        losses.append(float(loss.detach().cpu()))
        correlations.append(float(pearson_correlation(predictions.detach(), targets).cpu()))
        if max_batches is not None and batch_number >= max_batches:
            break
    if not losses:
        raise ValueError("training loader produced no batches")
    return {"train_loss": float(np.mean(losses)), "train_pearson_ic": float(np.mean(correlations))}


def _descending_ranks(values: np.ndarray) -> np.ndarray:
    # average 处理并列值，与标准 Spearman 的秩定义一致。
    return pd.Series(values).rank(method="average", ascending=False).to_numpy(dtype=np.float64)


def _numpy_pearson(left: np.ndarray, right: np.ndarray) -> float:
    # 小样本或常数向量没有有效相关系数，交给日均统计忽略。
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) < 2 or left.std(ddof=0) <= 0 or right.std(ddof=0) <= 0:
        return np.nan
    return float(np.corrcoef(left, right)[0, 1])


def summarize_validation(
    predictions: np.ndarray, targets: np.ndarray, date_ids: np.ndarray
) -> dict[str, float]:
    daily_rank = []
    daily_pearson = []
    for date_id in np.unique(date_ids):
        # 先逐日计算，再对日期等权平均，避免大横截面日期权重过高。
        mask = date_ids == date_id
        daily_rank.append(_numpy_pearson(
            _descending_ranks(predictions[mask]), _descending_ranks(targets[mask])
        ))
        daily_pearson.append(_numpy_pearson(predictions[mask], targets[mask]))
    rank_ic = float(np.nanmean(daily_rank))
    pearson_ic = float(np.nanmean(daily_pearson))
    pooled_ic = _numpy_pearson(predictions, targets)
    if not np.isfinite(rank_ic):
        raise FloatingPointError("validation daily Spearman IC is not finite")
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
    model.eval()
    # 分批推理后汇总到 CPU，保证指标在完整验证集上统一计算。
    predictions = []
    targets = []
    date_ids = []
    for batch_number, (batch, videos, frame_mask, target) in enumerate(
        iter_device_batches(loader, device, config.use_cuda_prefetch), start=1
    ):
        with autocast_context(device):
            prediction = model(videos, frame_mask)
        predictions.append(prediction.float().cpu().numpy())
        targets.append(target.float().cpu().numpy())
        date_ids.append(batch["date_ids"].numpy())
        if max_batches is not None and batch_number >= max_batches:
            break
    if not predictions:
        raise ValueError("validation loader produced no batches")
    return summarize_validation(
        np.concatenate(predictions), np.concatenate(targets), np.concatenate(date_ids)
    )


@torch.no_grad()
def predict_model(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    config: TrainingConfig,
) -> pd.DataFrame:
    model.eval()
    indices = []
    predictions = []
    for batch, videos, frame_mask, _ in iter_device_batches(
        loader, device, config.use_cuda_prefetch
    ):
        with autocast_context(device):
            prediction = model(videos, frame_mask)
        indices.append(batch["sample_indices"].numpy())
        predictions.append(prediction.float().cpu().numpy())
    # 重新按 sample_index 排序，使输出与预测切分的原始顺序严格一致。
    frame = pd.DataFrame({
        "sample_index": np.concatenate(indices).astype(np.int64),
        "raw_score": np.concatenate(predictions).astype(np.float32),
    }).sort_values("sample_index", kind="stable")
    if frame["sample_index"].duplicated().any() or not np.isfinite(frame["raw_score"]).all():
        raise ValueError("invalid prediction output")
    return frame


def _atomic_checkpoint(payload: dict[str, object], path: Path) -> None:
    # checkpoint 可能很大，仍先写临时文件再原子替换。
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _run_signature(
    paths: PipelinePaths, year: int, seed: int, config: TrainingConfig
) -> tuple[str, dict[str, object]]:
    # 数据、年份、随机种子、模型和训练参数共同定义一次可恢复运行。
    data_manifest = json.loads((paths.feature_dir / "manifest.json").read_text(encoding="utf-8"))
    payload = {
        "data_signature": data_manifest["signature"],
        "forecast_year": year,
        "seed": seed,
        "model": asdict(MODEL_CONFIG),
        "training": asdict(config),
        "loss": "sigmoid_rank_ic",
        "validation": "daily_mean_spearman_ic",
    }
    return stable_hash(payload), payload


def train_task(
    paths: PipelinePaths,
    forecast_year: int,
    seed: int,
    physical_gpu: int = 0,
    config: TrainingConfig = TrainingConfig(),
    overwrite: bool = False,
) -> dict[str, object]:
    if forecast_year not in FORECAST_YEARS or seed not in SEEDS:
        raise ValueError("unsupported forecast year or seed")
    set_random_seed(seed)
    # 父进程通过 CUDA_VISIBLE_DEVICES 隔离物理卡，子进程内始终使用 cuda:0。
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader, validation_loader, prediction_loader = make_data_loaders(
        paths, forecast_year, seed, workers=config.workers,
        inference_batch_size=config.inference_batch_size,
        pin_memory=device.type == "cuda",
    )
    run_dir = paths.models_dir / "baseline" / f"forecast_{forecast_year}" / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    history_path = run_dir / "history.parquet"
    prediction_path = run_dir / "prediction.parquet"
    completed_path = run_dir / "completed.json"
    signature, payload = _run_signature(paths, forecast_year, seed, config)

    if completed_path.exists() and best_path.exists() and not overwrite:
        # 已完成且签名一致时直接复用；缺预测文件则只补一次推理。
        completed = json.loads(completed_path.read_text(encoding="utf-8"))
        if completed.get("signature") != signature:
            raise ValueError(f"configuration changed for {run_dir}; use --overwrite")
        if prediction_loader is not None and not prediction_path.exists():
            checkpoint = torch.load(best_path, map_location=device, weights_only=False)
            model = build_model(device, config)
            unwrap_model(model).load_state_dict(checkpoint["model_state"])
            atomic_parquet(predict_model(model, prediction_loader, device, config), prediction_path)
        return completed["summary"]

    model = build_model(device, config)
    optimizer = build_optimizer(model, config, device)
    start_epoch = 1
    best_epoch = 0
    best_validation_ic = -np.inf
    epochs_without_improvement = 0
    history: list[dict[str, object]] = []
    if last_path.exists() and not overwrite:
        # last.pt 用于断点续训，签名不同则拒绝静默混用旧状态。
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        if checkpoint.get("signature") != signature:
            raise ValueError(f"checkpoint configuration changed for {run_dir}; use --overwrite")
        unwrap_model(model).load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        # 恢复所有随机状态，保证中断续训与连续训练的批次顺序一致。
        random.setstate(checkpoint["python_random_state"])
        np.random.set_state(checkpoint["numpy_random_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if device.type == "cuda" and checkpoint["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state(checkpoint["cuda_rng_state"], device)
        start_epoch = int(checkpoint["epoch"]) + 1
        best_epoch = int(checkpoint["best_epoch"])
        best_validation_ic = float(checkpoint["best_validation_ic"])
        epochs_without_improvement = int(checkpoint["epochs_without_improvement"])
        if history_path.exists():
            history = pd.read_parquet(history_path).to_dict("records")
        if epochs_without_improvement >= config.patience:
            # 已触发早停的 checkpoint 不再重复训练，直接进入最佳模型推理。
            start_epoch = config.max_epochs + 1

    started = time.perf_counter()
    for epoch in range(start_epoch, config.max_epochs + 1):
        # sampler 以 seed+epoch 打乱“日期”，不打乱日内股票。
        train_loader.batch_sampler.set_epoch(epoch)
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, config)
        validation_metrics = evaluate_model(model, validation_loader, device, config)
        validation_ic = validation_metrics["validation_ic"]
        improved = validation_ic > best_validation_ic + config.improvement_tolerance
        if improved:
            best_epoch = epoch
            best_validation_ic = validation_ic
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        history.append({
            "forecast_year": forecast_year, "seed": seed, "epoch": epoch,
            **train_metrics, **validation_metrics,
            "best_validation_ic": best_validation_ic,
            "epochs_without_improvement": epochs_without_improvement,
        })
        atomic_parquet(pd.DataFrame(history), history_path)
        # last 每轮覆盖用于恢复，best 只在验证 IC 真正提升时更新。
        checkpoint = {
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
        _atomic_checkpoint(checkpoint, last_path)
        if improved:
            _atomic_checkpoint(checkpoint, best_path)
        print(
            f"[{forecast_year}/seed={seed}] epoch={epoch:03d} "
            f"loss={train_metrics['train_loss']:.6f} "
            f"valid_spearman={validation_ic:.6f} "
            f"patience={epochs_without_improvement}/{config.patience}",
            flush=True,
        )
        if epochs_without_improvement >= config.patience:
            break

    if not best_path.exists():
        raise FileNotFoundError(f"best checkpoint not created: {best_path}")
    if prediction_loader is not None:
        # OOS 分数必须来自验证集最优 checkpoint，而不是最后一轮权重。
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        best_model = build_model(device, config)
        unwrap_model(best_model).load_state_dict(checkpoint["model_state"])
        atomic_parquet(predict_model(best_model, prediction_loader, device, config), prediction_path)
    summary = {
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
    atomic_json({"signature": signature, "summary": summary}, completed_path)
    return summary


def build_oos_factor(paths: PipelinePaths, forecast_year: int) -> int:
    split = load_split(paths, forecast_year)
    if not len(split["prediction_indices"]):
        return 0
    seed_frames = {}
    model_root = paths.models_dir / "baseline" / f"forecast_{forecast_year}"
    for seed in SEEDS:
        seed_frames[seed] = pd.read_parquet(model_root / f"seed_{seed}" / "prediction.parquet")
    reference = seed_frames[SEEDS[0]]["sample_index"].to_numpy(np.int64)
    # 三个种子的 sample_index 必须逐行一致，禁止按位置误合并不同股票。
    output = pd.DataFrame({"sample_index": reference})
    raw_columns = []
    for seed in SEEDS:
        frame = seed_frames[seed]
        if not np.array_equal(reference, frame["sample_index"].to_numpy(np.int64)):
            raise ValueError(f"seed prediction keys differ in {forecast_year}")
        column = f"seed_{seed}_raw_score"
        output[column] = frame["raw_score"].to_numpy(np.float32)
        raw_columns.append(column)
    metadata = pd.read_parquet(
        paths.sample_index_path, columns=["sample_index", "trade_date", "instrument_id"]
    ).set_index("sample_index")
    output = metadata.loc[reference].reset_index().merge(output, on="sample_index", validate="one_to_one")
    raw = output[raw_columns].astype(np.float64)
    # 每个种子先按日独立 z-score，再等权平均，避免种子尺度差异。
    grouped = raw.groupby(output["trade_date"], sort=False)
    means = grouped.transform("mean")
    standard_deviations = grouped.transform("std", ddof=0)
    if (standard_deviations <= 0).any().any():
        raise ValueError(f"constant daily seed score in {forecast_year}")
    zscore_columns = []
    for column in raw_columns:
        zscore_column = column.replace("_raw_score", "_zscore")
        output[zscore_column] = ((raw[column] - means[column]) / standard_deviations[column]).astype(np.float32)
        zscore_columns.append(zscore_column)
    output["raw_score_mean"] = raw.mean(axis=1).astype(np.float32)
    # factor_value 是最终交付因子；raw_score_mean 仅保留作审计。
    output["factor_value"] = output[zscore_columns].mean(axis=1).astype(np.float32)
    output["forecast_year"] = forecast_year
    output["factor_role"] = "full_year_oos"
    atomic_parquet(output, paths.factors_dir / f"vivit_oos_factor.{forecast_year}.parquet")
    return len(output)


def validate_outputs(paths: PipelinePaths) -> pd.DataFrame:
    records = []
    for year in FORECAST_YEARS:
        split = load_split(paths, year)
        expected = len(split["prediction_indices"])
        for seed in SEEDS:
            completed = paths.models_dir / "baseline" / f"forecast_{year}" / f"seed_{seed}" / "completed.json"
            if not completed.exists():
                raise FileNotFoundError(completed)
        factor_path = paths.factors_dir / f"vivit_oos_factor.{year}.parquet"
        if expected == 0:
            records.append({"forecast_year": year, "factor_rows": 0, "status": "no prediction rows"})
            continue
        factor = pd.read_parquet(factor_path)
        # 同时校验覆盖率、唯一键、顺序和数值有限性。
        if len(factor) != expected or factor.duplicated(["trade_date", "instrument_id"]).any():
            raise AssertionError(f"factor coverage failed for {year}")
        if not np.array_equal(factor["sample_index"].to_numpy(np.int64), split["prediction_indices"]):
            raise AssertionError(f"factor order failed for {year}")
        if not np.isfinite(factor["factor_value"]).all():
            raise AssertionError(f"non-finite factor values for {year}")
        for seed in SEEDS:
            # 每个种子的日截面均值应为 0、总体标准差应为 1。
            grouped = factor.groupby("trade_date")[f"seed_{seed}_zscore"]
            if grouped.mean().abs().max() > 1e-5 or (grouped.std(ddof=0) - 1).abs().max() > 1e-5:
                raise AssertionError(f"daily seed standardization failed for {year}/{seed}")
        records.append({"forecast_year": year, "factor_rows": len(factor), "status": "passed"})
    return pd.DataFrame(records)


def _tail(path: Path, lines: int = 80) -> str:
    # 子任务失败时只回显日志尾部，避免把超长训练日志塞入异常。
    if not path.exists():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


def run_parallel_training(
    paths: PipelinePaths,
    device_ids: tuple[int, ...] = (0, 1, 2, 3),
    tasks_per_gpu: int = 1,
    workers: int = 8,
    overwrite: bool = False,
) -> pd.DataFrame:
    if not device_ids or len(set(device_ids)) != len(device_ids) or tasks_per_gpu < 1:
        raise ValueError("GPU IDs must be non-empty/unique and tasks_per_gpu must be positive")
    if torch.cuda.device_count() <= max(device_ids):
        raise RuntimeError(f"requested GPUs {device_ids}, available count={torch.cuda.device_count()}")
    tasks = deque((year, seed) for year in FORECAST_YEARS for seed in SEEDS)
    # 任务是独立的 forecast_year×seed 组合，不共享模型或优化器状态。
    log_dir = paths.logs_dir / "parallel_task_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    running: list[dict[str, object]] = []
    started = time.perf_counter()
    try:
        while tasks or running:
            for gpu in device_ids:
                # 每张物理卡最多同时运行 tasks_per_gpu 个子进程。
                active = sum(item["gpu"] == gpu for item in running)
                while tasks and active < tasks_per_gpu:
                    year, seed = tasks.popleft()
                    log_path = log_dir / f"forecast_{year}.seed_{seed}.log"
                    log_handle = log_path.open("a", encoding="utf-8")
                    command = [
                        sys.executable, "-m", "vivit_factor", "train-task",
                        "--forecast-year", str(year), "--seed", str(seed),
                        "--physical-gpu", str(gpu), "--workers", str(workers),
                    ]
                    if overwrite:
                        command.append("--overwrite")
                    environment = os.environ.copy()
                    # 子进程只看见被分配的一张卡，内部设备编号因此固定为 0。
                    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
                    process = subprocess.Popen(
                        command, cwd=paths.project_dir, env=environment,
                        stdout=log_handle, stderr=subprocess.STDOUT, text=True,
                    )
                    running.append({
                        "process": process, "gpu": gpu, "year": year, "seed": seed,
                        "log_path": log_path, "log_handle": log_handle,
                    })
                    active += 1
                    print(f"launch GPU{gpu}: {year}/seed={seed} pid={process.pid}", flush=True)
            finished = []
            for item in running:
                # 非阻塞轮询让空闲 GPU 能立即领取下一项任务。
                return_code = item["process"].poll()
                if return_code is None:
                    continue
                item["log_handle"].close()
                finished.append(item)
                if return_code:
                    # 首个失败立即终止调度，并携带该任务日志尾部。
                    raise RuntimeError(
                        f"task failed {item['year']}/seed={item['seed']} on GPU{item['gpu']}\n"
                        f"{_tail(item['log_path'])}"
                    )
                print(f"done GPU{item['gpu']}: {item['year']}/seed={item['seed']}", flush=True)
            for item in finished:
                running.remove(item)
            if running and not finished:
                time.sleep(1)
    except BaseException:
        # 主进程异常时回收仍运行的子进程，避免后台继续占用 GPU。
        for item in running:
            if item["process"].poll() is None:
                item["process"].terminate()
            item["log_handle"].close()
        raise

    summaries = []
    factor_counts = {}
    for year in FORECAST_YEARS:
        for seed in SEEDS:
            completed_path = paths.models_dir / "baseline" / f"forecast_{year}" / f"seed_{seed}" / "completed.json"
            summaries.append(json.loads(completed_path.read_text(encoding="utf-8"))["summary"])
        factor_counts[str(year)] = build_oos_factor(paths, year)
    # 所有任务完成后统一生成因子、清单并执行交付校验。
    summary = pd.DataFrame(summaries).sort_values(["forecast_year", "seed"], kind="stable")
    atomic_parquet(summary, paths.artifacts_dir / "training_summary.parquet")
    atomic_json({
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
    validate_outputs(paths)
    return summary


def smoke_test(
    paths: PipelinePaths, forecast_year: int = 2018, seed: int = 42
) -> dict[str, float]:
    # 只跑一个训练批和一个验证批，用最低成本检查端到端张量与梯度。
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = TrainingConfig(workers=0, use_compile=False, use_cuda_prefetch=False)
    set_random_seed(seed)
    train_loader, validation_loader, _ = make_data_loaders(
        paths, forecast_year, seed, workers=0, inference_batch_size=512,
        pin_memory=device.type == "cuda",
    )
    model = build_model(device, config)
    optimizer = build_optimizer(model, config, device)
    train_metrics = train_one_epoch(model, train_loader, optimizer, device, config, max_batches=1)
    validation_metrics = evaluate_model(model, validation_loader, device, config, max_batches=1)
    return {**train_metrics, **validation_metrics}
