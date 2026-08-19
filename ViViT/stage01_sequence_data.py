"""阶段 01：构建 20 日精确历史索引，并组装 ViViT 的 Dataset 与 DataLoader。

本阶段只负责从 ViT feature store 派生视频输入；缺失历史帧补零并通过
frame_mask 显式传递，不包含模型定义、训练循环或因子评价。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


VIVIT_DIR = Path(__file__).resolve().parent
VIT_DIR = VIVIT_DIR.parent / "ViT"
if str(VIT_DIR) not in sys.path:
    sys.path.insert(0, str(VIT_DIR))

from vit_stage06_ranking import TEMPORAL_TURNOVER_LOSS  # noqa: E402
from vit_stage07_training import (  # noqa: E402
    TRAIN_BATCH_MODES,
    TradeDateBatchSampler,
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


def stable_hash(payload: object) -> str:
    """对配置生成稳定 SHA256，防止错误续训。"""

    text = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_write_text(text: str, path: Path) -> None:
    """原子写入文本文件，避免中断留下半成品。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_write_json(payload: dict[str, object], path: Path) -> None:
    """原子写入 JSON 文件，避免中断留下半成品。"""
    atomic_write_text(  # 原子落盘，避免半成品
        json.dumps(payload, ensure_ascii=False, indent=2), path
    )


def atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    """原子写入 Parquet 文件，避免中断留下半成品。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def atomic_save_numpy(array: np.ndarray, path: Path) -> None:
    """原子保存单个 NumPy 数组。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with temporary.open("wb") as file:
        np.save(file, array, allow_pickle=False)
    temporary.replace(path)


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    """原子保存多个具名 NumPy 数组。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with temporary.open("wb") as file:
        np.savez_compressed(file, **arrays)
    temporary.replace(path)


def atomic_save_checkpoint(payload: dict[str, object], path: Path) -> None:
    """原子保存 PyTorch checkpoint。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def source_manifests() -> tuple[dict[str, object], dict[str, object]]:
    """读取 ViT feature store 与滚动切分的上游清单。"""
    require_source_data()
    dataset = json.loads(SOURCE_DATASET_MANIFEST_PATH.read_text(encoding="utf-8"))
    rolling = json.loads(SOURCE_ROLLING_MANIFEST_PATH.read_text(encoding="utf-8"))
    return dataset, rolling


def aligned_store_payload() -> dict[str, object]:
    """整理 ViViT 序列仓库的来源、数组形状和清洗规则。"""
    dataset, rolling = source_manifests()  # 读取或生成来源清单
    return {
        "schema_version": 2,
        "source_dataset_signature": dataset["signature"],
        "source_rolling_signature": rolling["signature"],
        "source_sample_count": int(dataset["sample_count"]),
        "sequence_length": SEQUENCE_LENGTH,
        "history_shape": [int(dataset["sample_count"]), SEQUENCE_LENGTH],
        "history_dtype": "int32",
        "missing_history_sentinel": -1,
        "history_rule": "same instrument at exact target date_id minus lag; no carry-forward and no future data",
        "padding_rule": "zero-fill missing frames and expose a boolean frame mask",
        "split_rule": "load ViT train, validation and prediction indices without filtering or copying",
        "forecast_years": list(FORECAST_YEARS),
    }


def build_history_indices(
    sample_index: pd.DataFrame, output_path: Path = HISTORY_INDICES_PATH
) -> None:
    """为每个目标行构建固定 20 日槽位；精确日期缺失时保留 -1。"""

    sample_numbers = sample_index["sample_index"].to_numpy(dtype=np.int64)
    sample_count = len(sample_index)
    if sample_count >= np.iinfo(np.int32).max:
        raise OverflowError("feature store 行数超过 int32 history index 上限")
    if not np.array_equal(sample_numbers, np.arange(sample_count, dtype=np.int64)):
        raise AssertionError("sample_index 必须与 feature store 行号连续一致")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(
        f"{output_path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    if temporary.exists():
        temporary.unlink()
    history = np.lib.format.open_memmap(  # 使用 memmap，避免整库载入内存
        temporary,
        mode="w+",
        dtype=np.int32,
        shape=(sample_count, SEQUENCE_LENGTH),
    )
    history[:] = -1  # -1 明确表示该历史槽位缺帧，不做前向填充。
    history[:, -1] = sample_numbers.astype(np.int32)  # 最后一槽固定为目标日 T。
    date_ids = sample_index["date_id"].to_numpy(dtype=np.int32)

    for positions in sample_index.groupby("instrument_id", sort=False).indices.values():  # 每只股票独立找历史，禁止跨股票取帧。
        group_rows = np.asarray(positions, dtype=np.int64)
        group_dates = date_ids[group_rows]
        group_samples = sample_numbers[group_rows]
        if len(group_dates) > 1 and np.any(np.diff(group_dates) <= 0):
            raise AssertionError("同一股票的 date_id 必须严格递增")
        for lag in range(1, SEQUENCE_LENGTH):
            desired_dates = group_dates - lag  # 只接受精确的 T-lag 日期，不用邻近观测代替。
            locations = np.searchsorted(group_dates, desired_dates)  # 定位精确历史交易日
            in_bounds = locations < len(group_dates)
            matches = np.zeros(len(group_dates), dtype=bool)
            matches[in_bounds] = (  # searchsorted 落点仍须等于目标日期才算有效。
                group_dates[locations[in_bounds]] == desired_dates[in_bounds]
            )
            if matches.any():
                history[group_samples[matches], SEQUENCE_LENGTH - 1 - lag] = (
                    group_samples[locations[matches]].astype(np.int32)
                )

    history.flush()  # 将 memmap 修改刷入磁盘
    del history
    temporary.replace(output_path)


def history_coverage_stats(
    history: np.ndarray, indices: np.ndarray, chunk_size: int = 200_000
) -> dict[str, object]:
    """统计序列历史帧覆盖率和缺失分布。"""
    indices = np.asarray(indices, dtype=np.int64)
    if not len(indices):
        return {
            "sample_count": 0,
            "complete_history_count": 0,
            "incomplete_history_count": 0,
            "valid_frame_ratio": None,
        }
    complete_count = 0
    valid_frame_count = 0
    for start in range(0, len(indices), chunk_size):
        selected = np.asarray(history[indices[start : start + chunk_size]])
        valid = selected >= 0  # history 中非负值才是可读取的 feature-store 行号。
        complete_count += int(valid.all(axis=1).sum())
        valid_frame_count += int(valid.sum())
    return {
        "sample_count": len(indices),
        "complete_history_count": complete_count,
        "incomplete_history_count": len(indices) - complete_count,
        "valid_frame_ratio": valid_frame_count / (len(indices) * SEQUENCE_LENGTH),
    }


def build_sequence_store(overwrite: bool = False) -> dict[str, object]:
    """构建与 ViT 索引严格对齐、允许缺帧的 20 日历史矩阵。"""

    expected_payload = aligned_store_payload()
    expected_signature = stable_hash(expected_payload)  # 生成数据复用签名
    expected_outputs = (
        HISTORY_INDICES_PATH,
        SEQUENCE_MANIFEST_PATH,
        SEQUENCE_SUMMARY_PATH,
    )
    if not overwrite and SEQUENCE_MANIFEST_PATH.exists():
        existing = json.loads(SEQUENCE_MANIFEST_PATH.read_text(encoding="utf-8"))
        if existing.get("signature") != expected_signature:
            raise ValueError(
                "aligned sequence store 与当前 ViT 数据签名不一致；"
                "请显式使用 --overwrite 1 重建"
            )
        if all(path.exists() for path in expected_outputs):
            history = np.load(HISTORY_INDICES_PATH, mmap_mode="r")
            validate_history_indices(
                history, int(expected_payload["source_sample_count"])
            )
            return existing

    sample_index = pd.read_parquet(  # sample_index 是序列仓库与 ViT feature store 的公共主键表。
        SAMPLE_INDEX_PATH,
        columns=["sample_index", "trade_date", "instrument_id", "date_id"],
    )
    build_history_indices(sample_index)
    history = np.load(HISTORY_INDICES_PATH, mmap_mode="r")
    validate_history_indices(history, len(sample_index))

    summary_records: list[dict[str, object]] = []
    for year in FORECAST_YEARS:
        source_path = SOURCE_SPLIT_DIR / f"forecast_{year}.npz"
        with np.load(source_path, allow_pickle=False) as source:  # 直接沿用 ViT 滚动切分，避免重新划分造成口径漂移。
            split = {name: source[name].copy() for name in source.files}
        record: dict[str, object] = {"forecast_year": year}
        for split_name in ("train", "validation", "prediction"):
            stats = history_coverage_stats(history, split[f"{split_name}_indices"])  # 读取精确历史窗口索引
            for name, value in stats.items():
                record[f"{split_name}_{name}"] = value
        summary_records.append(record)

    summary = pd.DataFrame(summary_records)
    atomic_write_parquet(summary, SEQUENCE_SUMMARY_PATH)  # 原子落盘，避免半成品
    total_valid_frame_count = 0
    for start in range(0, len(history), 200_000):
        total_valid_frame_count += int(
            (np.asarray(history[start : start + 200_000]) >= 0).sum()
        )
    payload = {
        "signature": expected_signature,
        "payload": expected_payload,
        "total_valid_frame_count": total_valid_frame_count,
        "total_frame_count": int(history.size),
        "missing_frame_count": int(history.size) - total_valid_frame_count,
    }
    atomic_write_json(payload, SEQUENCE_MANIFEST_PATH)  # 原子落盘，避免半成品
    return payload


def load_sequence_split(forecast_year: int) -> dict[str, np.ndarray]:
    """直接加载 ViT split，不复制、不过滤任何目标索引。"""

    require_aligned_sequence_store()
    path = SOURCE_SPLIT_DIR / f"forecast_{forecast_year}.npz"
    if not path.exists():
        raise FileNotFoundError(f"缺少 ViT 滚动切分：{path}")
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


class SampleIndexDataset(Dataset):
    """Dataset 只保存目标日的 feature store 行号。
    """

    def __init__(self, sample_indices: np.ndarray) -> None:
        self.sample_indices = np.asarray(sample_indices, dtype=np.int64)  # Dataset 仅保存目标行号，实际数组由 collator 批量读取。

    def __len__(self) -> int:
        """返回 SampleIndexDataset 管理的样本数或批次数。"""
        return len(self.sample_indices)

    def __getitem__(self, position: int) -> int:
        """按给定索引返回 SampleIndexDataset 中的一个样本。"""
        return int(self.sample_indices[position])


class VideoBatchCollator:
    """按 aligned history matrix 读取视频，并对缺失历史帧补零。
    """

    def __init__(self) -> None:
        self.features = None
        self.targets = None
        self.date_ids = None
        self.history_indices = None

    def _ensure_open(self) -> None:
        """在当前 worker 首次取数时惰性打开共享 memmap，避免跨进程复制句柄。"""
        if self.features is None:
            self.features = np.load(FEATURES_PATH, mmap_mode="r")  # 每个 worker 独立惰性打开只读 memmap。
            self.targets = np.load(TARGETS_PATH, mmap_mode="r")
            self.date_ids = np.load(DATE_IDS_PATH, mmap_mode="r")
            self.history_indices = np.load(HISTORY_INDICES_PATH, mmap_mode="r")

    def __call__(self, sample_indices: list[int]) -> dict[str, torch.Tensor]:
        """根据历史行号组装视频、有效帧 mask、标签和样本元数据。"""
        self._ensure_open()
        target_indices = np.asarray(sample_indices, dtype=np.int64)  # 保留 DataLoader 给出的目标样本顺序。
        sequence_indices = np.array(
            self.history_indices[target_indices], dtype=np.int64, copy=True
        )
        frame_mask = sequence_indices >= 0  # True 表示该槽位有真实历史图像。
        if not frame_mask[:, -1].all():
            raise AssertionError("目标日自身不能是缺失帧")

        frame_array = np.zeros(  # 缺失历史帧保持全零，有效性由 frame_mask 单独表达。
            (len(target_indices), SEQUENCE_LENGTH, 17, 8, 8), dtype=np.float32
        )
        valid_indices = sequence_indices[frame_mask]  # 只从 memmap 读取真实存在的帧。
        if len(valid_indices):
            frame_array[frame_mask] = np.array(
                self.features[valid_indices], dtype=np.float32, copy=True, order="C"
            )
        video_array = np.ascontiguousarray(frame_array.transpose(0, 2, 1, 3, 4))  # [B,T,C,H,W] -> [B,C,T,H,W]。
        target_array = np.array(
            self.targets[target_indices], dtype=np.float32, copy=True
        )
        date_id_array = np.array(
            self.date_ids[target_indices], dtype=np.int32, copy=True
        )
        return {
            "videos": torch.from_numpy(video_array),
            "frame_mask": torch.from_numpy(frame_mask),
            "targets": torch.from_numpy(target_array),
            "date_ids": torch.from_numpy(date_id_array),
            "sample_indices": torch.from_numpy(target_indices),
        }


def make_data_loaders(
    forecast_year: int,
    seed: int,
    batch_size: int,
    inference_batch_size: int,
    train_batch_mode: str,
    loss_name: str,
    num_workers: int,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    """创建与 ViT batch 语义一致的 train/validation/prediction loaders。"""

    if train_batch_mode not in TRAIN_BATCH_MODES:
        raise ValueError(f"未知训练 batch 模式：{train_batch_mode}")
    temporal_mode = loss_name == TEMPORAL_TURNOVER_LOSS  # 换手损失需要相邻交易日的共同股票。
    if temporal_mode and train_batch_mode != "trade_date":
        raise ValueError("sigmoid_rank_ic_turnover 必须使用 trade_date batch")

    split = load_sequence_split(forecast_year)
    common: dict[str, object] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": VideoBatchCollator(),
    }
    if num_workers > 0:
        common.update({"persistent_workers": True, "prefetch_factor": 1})

    train_dataset = SampleIndexDataset(split["train_indices"])
    generator = torch.Generator().manual_seed(seed)  # 固定 DataLoader 随机源
    if train_batch_mode == "trade_date":
        date_ids = np.load(DATE_IDS_PATH, mmap_mode="r")
        instrument_ids = None
        if temporal_mode:
            instrument_ids = pd.read_parquet(  # 仅换手实验需要股票代码来配对相邻截面。
                SAMPLE_INDEX_PATH, columns=["instrument_id"]
            )["instrument_id"].to_numpy(copy=False)
        batch_sampler = TradeDateBatchSampler(  # 按完整交易日截面组 batch
            split["train_indices"],
            date_ids,
            seed=seed,
            shuffle=not temporal_mode,
            instrument_ids=instrument_ids,
        )
        train_loader = DataLoader(  # 按既定采样口径创建 DataLoader
            train_dataset,
            batch_sampler=batch_sampler,
            generator=generator,
            **common,
        )
    else:
        train_loader = DataLoader(  # 按既定采样口径创建 DataLoader
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            generator=generator,
            **common,
        )

    validation_loader = DataLoader(  # 按既定采样口径创建 DataLoader
        SampleIndexDataset(split["validation_indices"]),
        batch_size=inference_batch_size,
        shuffle=False,  # 验证集保持 split 内的确定性顺序。
        drop_last=False,
        **common,
    )
    prediction_loader = None
    if len(split["prediction_indices"]):
        prediction_loader = DataLoader(  # 按既定采样口径创建 DataLoader
            SampleIndexDataset(split["prediction_indices"]),
            batch_size=inference_batch_size,
            shuffle=False,  # 预测结果按 sample_indices 回连业务键。
            drop_last=False,
            **common,
        )
    return train_loader, validation_loader, prediction_loader


def set_train_epoch(loader: DataLoader, epoch: int) -> None:
    """把当前 epoch 传递给支持确定性重排的批采样器。"""
    if isinstance(loader.batch_sampler, TradeDateBatchSampler):
        loader.batch_sampler.set_epoch(epoch)  # 同步 epoch，保持采样可复现


def parse_args() -> argparse.Namespace:
    """解析阶段 01 的序列构建与切分审计命令。"""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build-data")
    build_parser.add_argument(
        "--overwrite", type=int, choices=(0, 1), default=0
    )
    subparsers.add_parser("audit-splits")
    return parser.parse_args()


def main() -> None:
    """分派序列数据构建或 rolling split 对齐审计。"""
    args = parse_args()
    if args.command == "build-data":
        print(
            json.dumps(
                build_sequence_store(bool(args.overwrite)),
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command == "audit-splits":
        print(audit_split_alignment().to_string(index=False))


def validate_history_indices(
    history: np.ndarray, sample_count: int, chunk_size: int = 200_000
) -> None:
    """检查 20 帧历史索引只引用目标日及过去日期。"""
    if history.shape != (sample_count, SEQUENCE_LENGTH):
        raise ValueError(f"history_indices 形状异常：{history.shape}")
    if history.dtype != np.int32:
        raise ValueError(f"history_indices dtype 异常：{history.dtype}")
    for start in range(0, sample_count, chunk_size):
        stop = min(start + chunk_size, sample_count)
        block = np.asarray(history[start:stop])
        expected = np.arange(start, stop, dtype=np.int32)
        if not np.array_equal(block[:, -1], expected):
            raise AssertionError("history_indices 最后一帧必须是目标样本自身")
        if block.min(initial=-1) < -1 or block.max(initial=-1) >= sample_count:  # 只允许缺帧哨兵 -1 或合法历史行号。
            raise IndexError("history_indices 包含越界行号")


def audit_split_alignment() -> pd.DataFrame:
    """逐年逐元素审计运行时 split 与 ViT 源 split 完全相同。"""

    records: list[dict[str, object]] = []
    for year in FORECAST_YEARS:
        runtime = load_sequence_split(year)
        with np.load(
            SOURCE_SPLIT_DIR / f"forecast_{year}.npz", allow_pickle=False
        ) as source:
            for split_name in ("train", "validation", "prediction"):
                key = f"{split_name}_indices"
                equal = np.array_equal(runtime[key], source[key])  # 核对样本顺序严格一致
                records.append(
                    {
                        "forecast_year": year,
                        "split": split_name,
                        "sample_count": len(runtime[key]),
                        "array_equal": bool(equal),
                    }
                )
                if not equal:
                    raise AssertionError(f"{year} {split_name} 与 ViT 索引不一致")
    return pd.DataFrame(records)


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


def require_source_data() -> None:
    """确认链接后的 ViT 数据与全部年度 split 完整存在。"""

    required = (
        FEATURES_PATH,
        TARGETS_PATH,
        DATE_IDS_PATH,
        SAMPLE_INDEX_PATH,
        SOURCE_DATASET_MANIFEST_PATH,
        SOURCE_ROLLING_MANIFEST_PATH,
        *(SOURCE_SPLIT_DIR / f"forecast_{year}.npz" for year in FORECAST_YEARS),
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"缺少 ViT 源数据：{missing}")


def require_aligned_sequence_store() -> dict[str, object]:
    """校验或按需构建与 ViT feature store 对齐的 ViViT 序列仓库。"""
    if not all(
        path.exists()
        for path in (HISTORY_INDICES_PATH, SEQUENCE_MANIFEST_PATH, SEQUENCE_SUMMARY_PATH)
    ):
        raise FileNotFoundError("缺少 aligned sequence store；请先运行 build-data")
    expected = aligned_store_payload()
    manifest = json.loads(SEQUENCE_MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("signature") != stable_hash(expected):
        raise ValueError(
            "aligned sequence store 与当前 ViT 数据签名不一致；"
            "请显式使用 build-data --overwrite 1 重建"
        )
    history = np.load(HISTORY_INDICES_PATH, mmap_mode="r")
    expected_shape = tuple(expected["history_shape"])
    if history.shape != expected_shape or history.dtype != np.int32:
        raise ValueError(
            f"history_indices 元数据异常：shape={history.shape}, dtype={history.dtype}"
        )
    return manifest


if __name__ == "__main__":
    main()
