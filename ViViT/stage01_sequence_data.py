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


VIVIT_DIR = Path(__file__).resolve().parent  # 定位vivit目录。
VIT_DIR = VIVIT_DIR.parent / "ViT"  # 定位vit目录。
if str(VIT_DIR) not in sys.path:  # 检查当前值是否属于允许集合。
    sys.path.insert(0, str(VIT_DIR))

from vit_stage06_ranking import TEMPORAL_TURNOVER_LOSS  # noqa: E402。
from vit_stage07_training import (  # noqa: E402。
    TRAIN_BATCH_MODES,
    TradeDateBatchSampler,
)

DATA_DIR = VIVIT_DIR / "data"  # 定位数据目录。
FEATURE_STORE_DIR = DATA_DIR / "feature_store"  # 定位特征存储目录。
SEQUENCE_STORE_DIR = DATA_DIR / "aligned_sequence_store"  # 定位sequence存储目录。
HISTORY_INDICES_PATH = SEQUENCE_STORE_DIR / "history_indices.npy"  # 定位历史记录索引路径。
SEQUENCE_MANIFEST_PATH = SEQUENCE_STORE_DIR / "sequence_manifest.json"  # 定位sequence清单路径。
SEQUENCE_SUMMARY_PATH = SEQUENCE_STORE_DIR / "rolling_split_summary.parquet"  # 定位sequence汇总路径。

FEATURES_PATH = FEATURE_STORE_DIR / "features.npy"  # 定位特征路径。
TARGETS_PATH = FEATURE_STORE_DIR / "targets.npy"  # 定位标签路径。
DATE_IDS_PATH = FEATURE_STORE_DIR / "date_ids.npy"  # 定位交易日ids路径。
SAMPLE_INDEX_PATH = FEATURE_STORE_DIR / "sample_index.parquet"  # 定位样本索引路径。
SOURCE_DATASET_MANIFEST_PATH = FEATURE_STORE_DIR / "dataset_manifest.json"  # 定位来源dataset清单路径。
SOURCE_ROLLING_MANIFEST_PATH = FEATURE_STORE_DIR / "rolling_split_manifest.json"  # 定位来源rolling清单路径。
SOURCE_SPLIT_DIR = FEATURE_STORE_DIR / "rolling_splits"  # 定位来源切分目录。

FORECAST_YEARS = tuple(range(2018, 2026))  # 集中声明forecast年份。
SEEDS = (42, 43, 44)  # 设置随机种子。
SEQUENCE_LENGTH = 20  # 设置sequencelength。


def stable_hash(payload: object) -> str:  # 对配置生成稳定 SHA256，防止错误续训。
    """对配置生成稳定 SHA256，防止错误续训。"""

    text = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()  # 返回稳定的 SHA-256 十六进制摘要。


def atomic_write_text(text: str, path: Path) -> None:  # 原子写入文本文件，避免中断留下半成品。
    """原子写入文本文件，避免中断留下半成品。"""
    path.parent.mkdir(parents=True, exist_ok=True)  # 确保目标目录存在。
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")  # 构造临时文件。
    temporary.write_text(text, encoding="utf-8")  # 把文本结果写入临时或目标文件。
    temporary.replace(path)  # 用临时文件原子替换正式结果。


def atomic_write_json(payload: dict[str, object], path: Path) -> None:  # 原子写入 JSON 文件，避免中断留下半成品。
    """原子写入 JSON 文件，避免中断留下半成品。"""
    atomic_write_text(  # 原子落盘，避免半成品。
        json.dumps(payload, ensure_ascii=False, indent=2), path
    )


def atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:  # 原子写入 Parquet 文件，避免中断留下半成品。
    """原子写入 Parquet 文件，避免中断留下半成品。"""
    path.parent.mkdir(parents=True, exist_ok=True)  # 确保目标目录存在。
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")  # 构造临时文件。
    frame.to_parquet(temporary, index=False, compression="zstd")  # 把表格按约定格式写入 Parquet。
    temporary.replace(path)  # 用临时文件原子替换正式结果。


def atomic_save_numpy(array: np.ndarray, path: Path) -> None:  # 原子保存单个 NumPy 数组。
    """原子保存单个 NumPy 数组。"""
    path.parent.mkdir(parents=True, exist_ok=True)  # 确保目标目录存在。
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")  # 构造临时文件。
    with temporary.open("wb") as file:  # 按指定模式打开文件并确保退出时关闭句柄。
        np.save(file, array, allow_pickle=False)
    temporary.replace(path)  # 用临时文件原子替换正式结果。


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:  # 原子保存多个具名 NumPy 数组。
    """原子保存多个具名 NumPy 数组。"""
    path.parent.mkdir(parents=True, exist_ok=True)  # 确保目标目录存在。
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")  # 构造临时文件。
    with temporary.open("wb") as file:  # 按指定模式打开文件并确保退出时关闭句柄。
        np.savez_compressed(file, **arrays)
    temporary.replace(path)  # 用临时文件原子替换正式结果。


def atomic_save_checkpoint(payload: dict[str, object], path: Path) -> None:  # 原子保存 PyTorch checkpoint。
    """原子保存 PyTorch checkpoint。"""
    path.parent.mkdir(parents=True, exist_ok=True)  # 确保目标目录存在。
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")  # 构造临时文件。
    torch.save(payload, temporary)  # 保存模型或训练状态。
    temporary.replace(path)  # 用临时文件原子替换正式结果。


def source_manifests() -> tuple[dict[str, object], dict[str, object]]:  # 读取 ViT feature store 与滚动切分的上游清单。
    """读取 ViT feature store 与滚动切分的上游清单。"""
    require_source_data()
    dataset = json.loads(SOURCE_DATASET_MANIFEST_PATH.read_text(encoding="utf-8"))  # 读取上游数据清单。
    rolling = json.loads(SOURCE_ROLLING_MANIFEST_PATH.read_text(encoding="utf-8"))  # 读取rolling。
    return dataset, rolling  # 返回读取并校验后的结果。


def aligned_store_payload() -> dict[str, object]:  # 整理 ViViT 序列仓库的来源、数组形状和清洗规则。
    """整理 ViViT 序列仓库的来源、数组形状和清洗规则。"""
    dataset, rolling = source_manifests()  # 读取或生成来源清单。
    return {  # 返回整理后的结果。
        "schema_version": 2,  # 记录schema版本。
        "source_dataset_signature": dataset["signature"],  # 记录来源dataset签名，用于来源一致性校验。
        "source_rolling_signature": rolling["signature"],  # 记录来源rolling签名，用于来源一致性校验。
        "source_sample_count": int(dataset["sample_count"]),  # 记录来源样本数量。
        "sequence_length": SEQUENCE_LENGTH,  # 记录sequencelength。
        "history_shape": [int(dataset["sample_count"]), SEQUENCE_LENGTH],  # 记录历史记录形状。
        "history_dtype": "int32",  # 把 history_dtype 映射为"int32"。
        "missing_history_sentinel": -1,  # 记录缺失值历史记录sentinel。
        "history_rule": "same instrument at exact target date_id minus lag; no carry-forward and no future data",  # 把 history_rule 映射为"same instrument at exact target date_id minus lag; no carry-forward and no future data"。
        "padding_rule": "zero-fill missing frames and expose a boolean frame mask",  # 把 padding_rule 映射为"zero-fill missing frames and expose a boolean frame mask"。
        "split_rule": "load ViT train, validation and prediction indices without filtering or copying",  # 把 split_rule 映射为"load ViT train, validation and prediction indices without filtering or copying"。
        "forecast_years": list(FORECAST_YEARS),  # 记录forecast年份。
    }


def build_history_indices(  # 为每个目标行构建固定 20 日槽位。
    sample_index: pd.DataFrame, output_path: Path = HISTORY_INDICES_PATH
) -> None:
    """为每个目标行构建固定 20 日槽位；精确日期缺失时保留 -1。"""

    sample_numbers = sample_index["sample_index"].to_numpy(dtype=np.int64)  # 转换并统一样本numbers的数据表示。
    sample_count = len(sample_index)  # 计算样本数量。
    if sample_count >= np.iinfo(np.int32).max:  # 检查当前数值是否越过业务或数值阈值。
        raise OverflowError("feature store 行数超过 int32 history index 上限")  # 数据规模超出存储类型上限时立即终止。
    if not np.array_equal(sample_numbers, np.arange(sample_count, dtype=np.int64)):  # 检查必需数据、配置或校验结果是否缺失。
        raise AssertionError("sample_index 必须与 feature store 行号连续一致")  # 关键数据契约被破坏时立即终止。

    output_path.parent.mkdir(parents=True, exist_ok=True)  # 确保目标目录存在。
    temporary = output_path.with_name(  # 构造临时文件。
        f"{output_path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    if temporary.exists():  # 检查目标文件或目录是否满足存在性要求。
        temporary.unlink()
    history = np.lib.format.open_memmap(  # 使用 memmap，避免整库载入内存。
        temporary,
        mode="w+",  # 选择当前处理模式。
        dtype=np.int32,  # 统一结果的数据类型。
        shape=(sample_count, SEQUENCE_LENGTH),  # 配置形状。
    )
    history[:] = -1  # -1 明确表示该历史槽位缺帧，不做前向填充。
    history[:, -1] = sample_numbers.astype(np.int32)  # 最后一槽固定为目标日 T。
    date_ids = sample_index["date_id"].to_numpy(dtype=np.int32)  # 转换并统一交易日ids的数据表示。

    for positions in sample_index.groupby("instrument_id", sort=False).indices.values():  # 每只股票独立找历史，禁止跨股票取帧。
        group_rows = np.asarray(positions, dtype=np.int64)  # 转换并统一group行数的数据表示。
        group_dates = date_ids[group_rows]
        group_samples = sample_numbers[group_rows]
        if len(group_dates) > 1 and np.any(np.diff(group_dates) <= 0):  # 检查当前样本数量是否满足计算要求。
            raise AssertionError("同一股票的 date_id 必须严格递增")  # 关键数据契约被破坏时立即终止。
        for lag in range(1, SEQUENCE_LENGTH):  # 按给定步长遍历样本区间，更新lag。
            desired_dates = group_dates - lag  # 只接受精确的 T-lag 日期，不用邻近观测代替。
            locations = np.searchsorted(group_dates, desired_dates)  # 定位精确历史交易日。
            in_bounds = locations < len(group_dates)  # 计算inbounds判定条件。
            matches = np.zeros(len(group_dates), dtype=bool)  # 初始化matches。
            matches[in_bounds] = (  # searchsorted 落点仍须等于目标日期才算有效。
                group_dates[locations[in_bounds]] == desired_dates[in_bounds]
            )
            if matches.any():  # 检查当前掩码或数据是否满足有效性要求。
                history[group_samples[matches], SEQUENCE_LENGTH - 1 - lag] = (
                    group_samples[locations[matches]].astype(np.int32)
                )

    history.flush()  # 将 memmap 修改刷入磁盘。
    del history  # 释放不再使用的中间引用。
    temporary.replace(output_path)  # 用临时文件原子替换正式结果。


def history_coverage_stats(  # 统计序列历史帧覆盖率和缺失分布。
    history: np.ndarray, indices: np.ndarray, chunk_size: int = 200_000
) -> dict[str, object]:
    """统计序列历史帧覆盖率和缺失分布。"""
    indices = np.asarray(indices, dtype=np.int64)  # 转换并统一索引的数据表示。
    if not len(indices):  # 检查当前样本数量是否满足计算要求。
        return {  # 返回统计序列历史帧覆盖率和缺失分布。
            "sample_count": 0,  # 记录样本数量。
            "complete_history_count": 0,  # 记录complete历史记录数量。
            "incomplete_history_count": 0,  # 记录incomplete历史记录数量。
            "valid_frame_ratio": None,  # 记录有效帧比例。
        }
    complete_count = 0  # 计算complete数量。
    valid_frame_count = 0  # 构造有效帧数量，限定后续参与计算的样本。
    for start in range(0, len(indices), chunk_size):  # 按给定步长遍历样本区间，更新起点。
        selected = np.asarray(history[indices[start : start + chunk_size]])  # 转换并统一筛选结果的数据表示。
        valid = selected >= 0  # history 中非负值才是可读取的 feature-store 行号。
        complete_count += int(valid.all(axis=1).sum())
        valid_frame_count += int(valid.sum())
    return {  # 返回统计序列历史帧覆盖率和缺失分布。
        "sample_count": len(indices),  # 记录样本数量。
        "complete_history_count": complete_count,  # 记录complete历史记录数量。
        "incomplete_history_count": len(indices) - complete_count,
        "valid_frame_ratio": valid_frame_count / (len(indices) * SEQUENCE_LENGTH),
    }


def build_sequence_store(overwrite: bool = False) -> dict[str, object]:  # 构建与 ViT 索引严格对齐、允许缺帧的 20 日历史矩阵。
    """构建与 ViT 索引严格对齐、允许缺帧的 20 日历史矩阵。"""

    expected_payload = aligned_store_payload()
    expected_signature = stable_hash(expected_payload)  # 生成数据复用签名。
    expected_outputs = (
        HISTORY_INDICES_PATH,
        SEQUENCE_MANIFEST_PATH,
        SEQUENCE_SUMMARY_PATH,
    )
    if not overwrite and SEQUENCE_MANIFEST_PATH.exists():  # 检查目标文件或目录是否满足存在性要求。
        existing = json.loads(SEQUENCE_MANIFEST_PATH.read_text(encoding="utf-8"))  # 读取existing。
        if existing.get("signature") != expected_signature:  # 按当前模式、标识或状态选择处理分支。
            raise ValueError(  # 输入数据不满足业务或数值约束时立即终止。
                "aligned sequence store 与当前 ViT 数据签名不一致；"
                "请显式使用 --overwrite 1 重建"
            )
        if all(path.exists() for path in expected_outputs):  # 检查目标文件或目录是否满足存在性要求。
            history = np.load(HISTORY_INDICES_PATH, mmap_mode="r")  # 读取历史记录。
            validate_history_indices(
                history, int(expected_payload["source_sample_count"])
            )
            return existing  # 返回existing。

    sample_index = pd.read_parquet(  # sample_index 是序列仓库与 ViT feature store 的公共主键表。
        SAMPLE_INDEX_PATH,
        columns=["sample_index", "trade_date", "instrument_id", "date_id"],  # 限定本次读取或输出使用的字段。
    )
    build_history_indices(sample_index)
    history = np.load(HISTORY_INDICES_PATH, mmap_mode="r")  # 读取历史记录。
    validate_history_indices(history, len(sample_index))

    summary_records: list[dict[str, object]] = []  # 初始化汇总records容器。
    for year in FORECAST_YEARS:  # 遍历forecast年份，逐项处理年份。
        source_path = SOURCE_SPLIT_DIR / f"forecast_{year}.npz"  # 构造来源路径。
        with np.load(source_path, allow_pickle=False) as source:  # 直接沿用 ViT 滚动切分，避免重新划分造成口径漂移。
            split = {name: source[name].copy() for name in source.files}  # 转换切分的数据类型或存储位置。
        record: dict[str, object] = {"forecast_year": year}  # 初始化record容器。
        for split_name in ("train", "validation", "prediction"):  # 遍历训练验证预测，逐项处理切分名称。
            stats = history_coverage_stats(history, split[f"{split_name}_indices"])  # 读取精确历史窗口索引。
            for name, value in stats.items():  # 遍历statsitems，逐项处理名称值。
                record[f"{split_name}_{name}"] = value
        summary_records.append(record)  # 把当前结果追加到汇总列表。

    summary = pd.DataFrame(summary_records)
    atomic_write_parquet(summary, SEQUENCE_SUMMARY_PATH)  # 原子落盘，避免半成品。
    total_valid_frame_count = 0  # 构造total有效帧数量，限定后续参与计算的样本。
    for start in range(0, len(history), 200_000):  # 按给定步长遍历样本区间，更新起点。
        total_valid_frame_count += int(
            (np.asarray(history[start : start + 200_000]) >= 0).sum()
        )
    payload = {  # 初始化载荷容器。
        "signature": expected_signature,
        "payload": expected_payload,
        "total_valid_frame_count": total_valid_frame_count,
        "total_frame_count": int(history.size),
        "missing_frame_count": int(history.size) - total_valid_frame_count,
    }
    atomic_write_json(payload, SEQUENCE_MANIFEST_PATH)  # 原子落盘，避免半成品。
    return payload  # 返回载荷。


def load_sequence_split(forecast_year: int) -> dict[str, np.ndarray]:  # 直接加载 ViT split，不复制、不过滤任何目标索引。
    """直接加载 ViT split，不复制、不过滤任何目标索引。"""

    require_aligned_sequence_store()
    path = SOURCE_SPLIT_DIR / f"forecast_{forecast_year}.npz"
    if not path.exists():  # 检查目标文件或目录是否满足存在性要求。
        raise FileNotFoundError(f"缺少 ViT 滚动切分：{path}")  # 目标文件不存在时立即终止。
    with np.load(path, allow_pickle=False) as archive:  # 进入受控上下文并在退出时自动释放资源。
        return {name: archive[name].copy() for name in archive.files}  # 返回直接加载 ViT split，不复制、不过滤任何目标索引。


class SampleIndexDataset(Dataset):  # Dataset 只保存目标日的 feature store 行号。
    """Dataset 只保存目标日的 feature store 行号。
    """

    def __init__(self, sample_indices: np.ndarray) -> None:  # 初始化当前对象的配置、状态与子组件。
        self.sample_indices = np.asarray(sample_indices, dtype=np.int64)  # Dataset 仅保存目标行号，实际数组由 collator 批量读取。

    def __len__(self) -> int:  # 返回 SampleIndexDataset 管理的样本数或批次数。
        """返回 SampleIndexDataset 管理的样本数或批次数。"""
        return len(self.sample_indices)  # 返回 SampleIndexDataset 管理的样本数或批次数。

    def __getitem__(self, position: int) -> int:  # 按给定索引返回 SampleIndexDataset 中的一个样本。
        """按给定索引返回 SampleIndexDataset 中的一个样本。"""
        return int(self.sample_indices[position])  # 返回按当前规则组织后的结果。


class VideoBatchCollator:  # 按 aligned history matrix 读取视频，并对缺失历史帧补零。
    """按 aligned history matrix 读取视频，并对缺失历史帧补零。
    """

    def __init__(self) -> None:  # 初始化当前对象的配置、状态与子组件。
        self.features = None  # 先把特征初始化为空，后续按条件填充。
        self.targets = None  # 先把标签初始化为空，后续按条件填充。
        self.date_ids = None  # 先把交易日ids初始化为空，后续按条件填充。
        self.history_indices = None  # 先把历史记录索引初始化为空，后续按条件填充。

    def _ensure_open(self) -> None:  # 在当前 worker 首次取数时惰性打开共享 memmap，避免跨进程复制句柄。
        """在当前 worker 首次取数时惰性打开共享 memmap，避免跨进程复制句柄。"""
        if self.features is None:  # 检查所需对象是否尚未初始化。
            self.features = np.load(FEATURES_PATH, mmap_mode="r")  # 每个 worker 独立惰性打开只读 memmap。
            self.targets = np.load(TARGETS_PATH, mmap_mode="r")  # 保存标签，供该对象后续步骤复用。
            self.date_ids = np.load(DATE_IDS_PATH, mmap_mode="r")  # 保存交易日ids，供该对象后续步骤复用。
            self.history_indices = np.load(HISTORY_INDICES_PATH, mmap_mode="r")  # 保存历史记录索引，供该对象后续步骤复用。

    def __call__(self, sample_indices: list[int]) -> dict[str, torch.Tensor]:  # 根据历史行号组装视频、有效帧 mask、标签和样本元数据。
        """根据历史行号组装视频、有效帧 mask、标签和样本元数据。"""
        self._ensure_open()
        target_indices = np.asarray(sample_indices, dtype=np.int64)  # 保留 DataLoader 给出的目标样本顺序。
        sequence_indices = np.array(
            self.history_indices[target_indices], dtype=np.int64, copy=True  # 保存历史记录索引，供该对象后续步骤复用。
        )
        frame_mask = sequence_indices >= 0  # True 表示该槽位有真实历史图像。
        if not frame_mask[:, -1].all():  # 检查当前掩码或数据是否满足有效性要求。
            raise AssertionError("目标日自身不能是缺失帧")  # 关键数据契约被破坏时立即终止。

        frame_array = np.zeros(  # 缺失历史帧保持全零，有效性由 frame_mask 单独表达。
            (len(target_indices), SEQUENCE_LENGTH, 17, 8, 8), dtype=np.float32
        )
        valid_indices = sequence_indices[frame_mask]  # 只从 memmap 读取真实存在的帧。
        if len(valid_indices):  # 检查当前样本数量是否满足计算要求。
            frame_array[frame_mask] = np.array(  # 构造帧数组，限定后续参与计算的样本。
                self.features[valid_indices], dtype=np.float32, copy=True, order="C"  # 保存特征，供该对象后续步骤复用。
            )
        video_array = np.ascontiguousarray(frame_array.transpose(0, 2, 1, 3, 4))  # [B,T,C,H,W] -> [B,C,T,H,W]。
        target_array = np.array(
            self.targets[target_indices], dtype=np.float32, copy=True  # 保存标签，供该对象后续步骤复用。
        )
        date_id_array = np.array(
            self.date_ids[target_indices], dtype=np.int32, copy=True  # 保存交易日ids，供该对象后续步骤复用。
        )
        return {  # 返回根据历史行号组装视频、有效帧 mask、标签和样本元数据。
            "videos": torch.from_numpy(video_array),
            "frame_mask": torch.from_numpy(frame_mask),
            "targets": torch.from_numpy(target_array),
            "date_ids": torch.from_numpy(date_id_array),
            "sample_indices": torch.from_numpy(target_indices),
        }


def make_data_loaders(  # 创建与 ViT batch 语义一致的 train/validation/prediction loaders。
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

    if train_batch_mode not in TRAIN_BATCH_MODES:  # 检查当前值是否属于允许集合。
        raise ValueError(f"未知训练 batch 模式：{train_batch_mode}")  # 输入数据不满足业务或数值约束时立即终止。
    temporal_mode = loss_name == TEMPORAL_TURNOVER_LOSS  # 换手损失需要相邻交易日的共同股票。
    if temporal_mode and train_batch_mode != "trade_date":  # 按当前模式、标识或状态选择处理分支。
        raise ValueError("sigmoid_rank_ic_turnover 必须使用 trade_date batch")  # 输入数据不满足业务或数值约束时立即终止。

    split = load_sequence_split(forecast_year)
    common: dict[str, object] = {  # 初始化公共容器。
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": VideoBatchCollator(),
    }
    if num_workers > 0:  # 检查当前数值是否越过业务或数值阈值。
        common.update({"persistent_workers": True, "prefetch_factor": 1})  # 用当前配置更新参数字典。

    train_dataset = SampleIndexDataset(split["train_indices"])  # 创建训练dataset。
    generator = torch.Generator().manual_seed(seed)  # 固定 DataLoader 随机源。
    if train_batch_mode == "trade_date":  # 按当前模式、标识或状态选择处理分支。
        date_ids = np.load(DATE_IDS_PATH, mmap_mode="r")  # 读取交易日ids。
        instrument_ids = None  # 先把股票ids初始化为空，后续按条件填充。
        if temporal_mode:  # 根据当前数据状态选择处理分支。
            instrument_ids = pd.read_parquet(  # 仅换手实验需要股票代码来配对相邻截面。
                SAMPLE_INDEX_PATH, columns=["instrument_id"]  # 定位样本索引路径。
            )["instrument_id"].to_numpy(copy=False)
        batch_sampler = TradeDateBatchSampler(  # 按完整交易日截面组 batch。
            split["train_indices"],
            date_ids,
            seed=seed,  # 固定本轮随机种子。
            shuffle=not temporal_mode,  # 控制是否打乱样本顺序。
            instrument_ids=instrument_ids,  # 配置股票ids。
        )
        train_loader = DataLoader(  # 按既定采样口径创建 DataLoader。
            train_dataset,
            batch_sampler=batch_sampler,  # 配置批次sampler。
            generator=generator,  # 传入固定随机数生成器以保证复现。
            **common,
        )
    else:  # 处理上述条件均不成立的情况。
        train_loader = DataLoader(  # 按既定采样口径创建 DataLoader。
            train_dataset,
            batch_size=batch_size,  # 设置每个优化批次的样本数。
            shuffle=True,  # 控制是否打乱样本顺序。
            drop_last=True,  # 控制是否丢弃最后一个不完整批次。
            generator=generator,  # 传入固定随机数生成器以保证复现。
            **common,
        )

    validation_loader = DataLoader(  # 按既定采样口径创建 DataLoader。
        SampleIndexDataset(split["validation_indices"]),
        batch_size=inference_batch_size,  # 设置每个优化批次的样本数。
        shuffle=False,  # 验证集保持 split 内的确定性顺序。
        drop_last=False,  # 控制是否丢弃最后一个不完整批次。
        **common,
    )
    prediction_loader = None  # 先把预测数据加载器初始化为空，后续按条件填充。
    if len(split["prediction_indices"]):  # 检查当前样本数量是否满足计算要求。
        prediction_loader = DataLoader(  # 按既定采样口径创建 DataLoader。
            SampleIndexDataset(split["prediction_indices"]),
            batch_size=inference_batch_size,  # 设置每个优化批次的样本数。
            shuffle=False,  # 预测结果按 sample_indices 回连业务键。
            drop_last=False,  # 控制是否丢弃最后一个不完整批次。
            **common,
        )
    return train_loader, validation_loader, prediction_loader  # 返回创建与 ViT batch 语义一致的 train/validation/prediction 。


def set_train_epoch(loader: DataLoader, epoch: int) -> None:  # 把当前 epoch 传递给支持确定性重排的批采样器。
    """把当前 epoch 传递给支持确定性重排的批采样器。"""
    if isinstance(loader.batch_sampler, TradeDateBatchSampler):  # 检查对象类型是否符合接口约定。
        loader.batch_sampler.set_epoch(epoch)  # 同步 epoch，保持采样可复现。


def parse_args() -> argparse.Namespace:  # 解析阶段 01 的序列构建与切分审计命令。
    """解析阶段 01 的序列构建与切分审计命令。"""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build-data")
    build_parser.add_argument(  # 注册命令行参数及其默认值。
        "--overwrite", type=int, choices=(0, 1), default=0
    )
    subparsers.add_parser("audit-splits")
    return parser.parse_args()  # 返回解析阶段 01 的序列构建与切分审计命令。


def main() -> None:  # 分派序列数据构建或 rolling split 对齐审计。
    """分派序列数据构建或 rolling split 对齐审计。"""
    args = parse_args()
    if args.command == "build-data":  # 按命令行子命令选择执行路径。
        print(  # 输出当前任务的进度或诊断信息。
            json.dumps(
                build_sequence_store(bool(args.overwrite)),
                ensure_ascii=False,  # 配置确保ascii。
                indent=2,  # 配置indent。
            )
        )
    elif args.command == "audit-splits":  # 选择 audit-splits 子命令的处理路径。
        print(audit_split_alignment().to_string(index=False))  # 输出当前任务的进度或诊断信息。


def validate_history_indices(  # 检查 20 帧历史索引只引用目标日及过去日期。
    history: np.ndarray, sample_count: int, chunk_size: int = 200_000
) -> None:
    """检查 20 帧历史索引只引用目标日及过去日期。"""
    if history.shape != (sample_count, SEQUENCE_LENGTH):  # 检查张量或数组形状是否符合接口约定。
        raise ValueError(f"history_indices 形状异常：{history.shape}")  # 输入数据不满足业务或数值约束时立即终止。
    if history.dtype != np.int32:  # 按当前模式、标识或状态选择处理分支。
        raise ValueError(f"history_indices dtype 异常：{history.dtype}")  # 输入数据不满足业务或数值约束时立即终止。
    for start in range(0, sample_count, chunk_size):  # 按给定步长遍历样本区间，更新起点。
        stop = min(start + chunk_size, sample_count)
        block = np.asarray(history[start:stop])  # 转换并统一block的数据表示。
        expected = np.arange(start, stop, dtype=np.int32)
        if not np.array_equal(block[:, -1], expected):  # 检查必需数据、配置或校验结果是否缺失。
            raise AssertionError("history_indices 最后一帧必须是目标样本自身")  # 关键数据契约被破坏时立即终止。
        if block.min(initial=-1) < -1 or block.max(initial=-1) >= sample_count:  # 只允许缺帧哨兵 -1 或合法历史行号。
            raise IndexError("history_indices 包含越界行号")  # 当前数据违反接口约定时立即终止。


def audit_split_alignment() -> pd.DataFrame:  # 逐年逐元素审计运行时 split 与 ViT 源 split 完全相同。
    """逐年逐元素审计运行时 split 与 ViT 源 split 完全相同。"""

    records: list[dict[str, object]] = []  # 初始化records容器。
    for year in FORECAST_YEARS:  # 遍历forecast年份，逐项处理年份。
        runtime = load_sequence_split(year)
        with np.load(  # 进入受控上下文并在退出时自动释放资源。
            SOURCE_SPLIT_DIR / f"forecast_{year}.npz", allow_pickle=False
        ) as source:
            for split_name in ("train", "validation", "prediction"):  # 遍历训练验证预测，逐项处理切分名称。
                key = f"{split_name}_indices"
                equal = np.array_equal(runtime[key], source[key])  # 核对样本顺序严格一致。
                records.append(  # 把当前结果追加到汇总列表。
                    {
                        "forecast_year": year,
                        "split": split_name,
                        "sample_count": len(runtime[key]),
                        "array_equal": bool(equal),
                    }
                )
                if not equal:  # 检查必需数据、配置或校验结果是否缺失。
                    raise AssertionError(f"{year} {split_name} 与 ViT 索引不一致")  # 关键数据契约被破坏时立即终止。
    return pd.DataFrame(records)  # 返回逐年逐元素审计运行时 split 与 ViT 源 split 完全相同。


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
        "train_batch_count": batch_count,
        "train_date_count": None,
        "train_sample_count": batch_count * batch_size,
        "train_batch_size_min": batch_size,
        "train_batch_size_median": float(batch_size),
        "train_batch_size_max": batch_size,
        "dropped_train_samples": sample_count - batch_count * batch_size,
        "reshuffle_each_epoch": True,
        "gradient_update_mode": "per_batch",  # 把 gradient_update_mode 映射为"per_batch"。
        "optimizer_steps_per_epoch": batch_count,
    }


def require_source_data() -> None:  # 确认链接后的 ViT 数据与全部年度 split 完整存在。
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
    missing = [str(path) for path in required if not path.exists()]  # 初始化missing容器。
    if missing:  # 根据当前数据状态选择处理分支。
        raise FileNotFoundError(f"缺少 ViT 源数据：{missing}")  # 目标文件不存在时立即终止。


def require_aligned_sequence_store() -> dict[str, object]:  # 校验或按需构建与 ViT feature store 对齐的 ViViT 序列仓库。
    """校验或按需构建与 ViT feature store 对齐的 ViViT 序列仓库。"""
    if not all(  # 检查必需数据、配置或校验结果是否缺失。
        path.exists()
        for path in (HISTORY_INDICES_PATH, SEQUENCE_MANIFEST_PATH, SEQUENCE_SUMMARY_PATH)  # 遍历历史记录索引路径sequence清单路径sequence汇总路径，逐项处理路径。
    ):
        raise FileNotFoundError("缺少 aligned sequence store；请先运行 build-data")  # 目标文件不存在时立即终止。
    expected = aligned_store_payload()
    manifest = json.loads(SEQUENCE_MANIFEST_PATH.read_text(encoding="utf-8"))  # 读取清单。
    if manifest.get("signature") != stable_hash(expected):  # 按当前模式、标识或状态选择处理分支。
        raise ValueError(  # 输入数据不满足业务或数值约束时立即终止。
            "aligned sequence store 与当前 ViT 数据签名不一致；"
            "请显式使用 build-data --overwrite 1 重建"
        )
    history = np.load(HISTORY_INDICES_PATH, mmap_mode="r")  # 读取历史记录。
    expected_shape = tuple(expected["history_shape"])  # 初始化预期形状容器。
    if history.shape != expected_shape or history.dtype != np.int32:  # 检查张量或数组形状是否符合接口约定。
        raise ValueError(  # 输入数据不满足业务或数值约束时立即终止。
            f"history_indices 元数据异常：shape={history.shape}, dtype={history.dtype}"
        )
    return manifest  # 返回清单。


if __name__ == "__main__":  # 按当前模式、标识或状态选择处理分支。
    main()
