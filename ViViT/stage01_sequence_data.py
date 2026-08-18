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


# 读取 `Path(__file__).resolve().parent` 的当前值，并赋给 `VIVIT_DIR`。
VIVIT_DIR = Path(__file__).resolve().parent
# 计算 `VIVIT_DIR.parent / 'ViT'`，并将数值结果保存到 `VIT_DIR`。
VIT_DIR = VIVIT_DIR.parent / "ViT"
# 检查 `str(VIT_DIR) not in sys.path`；根据条件结果选择对应处理分支。
if str(VIT_DIR) not in sys.path:
    # 调用 `sys.path.insert` 把项目模块目录加入导入搜索路径。
    sys.path.insert(0, str(VIT_DIR))

from vit_stage06_ranking import TEMPORAL_TURNOVER_LOSS  # noqa: E402
from vit_stage07_training import (  # noqa: E402
    TRAIN_BATCH_MODES,
    TradeDateBatchSampler,
)

# 计算 `VIVIT_DIR / 'data'`，并将数值结果保存到 `DATA_DIR`。
DATA_DIR = VIVIT_DIR / "data"
# 计算 `DATA_DIR / 'feature_store'`，并将数值结果保存到 `FEATURE_STORE_DIR`。
FEATURE_STORE_DIR = DATA_DIR / "feature_store"
# 计算 `DATA_DIR / 'aligned_sequence_store'`，并将数值结果保存到 `SEQUENCE_STORE_DIR`。
SEQUENCE_STORE_DIR = DATA_DIR / "aligned_sequence_store"
# 计算 `SEQUENCE_STORE_DIR / 'history_indices.npy'`，并将数值结果保存到 `HISTORY_INDICES_PATH`。
HISTORY_INDICES_PATH = SEQUENCE_STORE_DIR / "history_indices.npy"
# 计算 `SEQUENCE_STORE_DIR / 'sequence_manifest.json'`，并将数值结果保存到 `SEQUENCE_MANIFEST_PATH`。
SEQUENCE_MANIFEST_PATH = SEQUENCE_STORE_DIR / "sequence_manifest.json"
# 计算 `SEQUENCE_STORE_DIR / 'rolling_split_summary.parquet'`，并将数值结果保存到 `SEQUENCE_SUMMARY_PATH`。
SEQUENCE_SUMMARY_PATH = SEQUENCE_STORE_DIR / "rolling_split_summary.parquet"

# 计算 `FEATURE_STORE_DIR / 'features.npy'`，并将数值结果保存到 `FEATURES_PATH`。
FEATURES_PATH = FEATURE_STORE_DIR / "features.npy"
# 计算 `FEATURE_STORE_DIR / 'targets.npy'`，并将数值结果保存到 `TARGETS_PATH`。
TARGETS_PATH = FEATURE_STORE_DIR / "targets.npy"
# 计算 `FEATURE_STORE_DIR / 'date_ids.npy'`，并将数值结果保存到 `DATE_IDS_PATH`。
DATE_IDS_PATH = FEATURE_STORE_DIR / "date_ids.npy"
# 计算 `FEATURE_STORE_DIR / 'sample_index.parquet'`，并将数值结果保存到 `SAMPLE_INDEX_PATH`。
SAMPLE_INDEX_PATH = FEATURE_STORE_DIR / "sample_index.parquet"
# 计算 `FEATURE_STORE_DIR / 'dataset_manifest.json'`，并将数值结果保存到 `SOURCE_DATASET_MANIFEST_PATH`。
SOURCE_DATASET_MANIFEST_PATH = FEATURE_STORE_DIR / "dataset_manifest.json"
# 计算 `FEATURE_STORE_DIR / 'rolling_split_manifest.json'`，并将数值结果保存到 `SOURCE_ROLLING_MANIFEST_PATH`。
SOURCE_ROLLING_MANIFEST_PATH = FEATURE_STORE_DIR / "rolling_split_manifest.json"
# 计算 `FEATURE_STORE_DIR / 'rolling_splits'`，并将数值结果保存到 `SOURCE_SPLIT_DIR`。
SOURCE_SPLIT_DIR = FEATURE_STORE_DIR / "rolling_splits"

# 调用 `tuple` 整理数据结构，并将结果保存到 `FORECAST_YEARS`。
FORECAST_YEARS = tuple(range(2018, 2026))
# 组织一组有序值，并保存到 `SEEDS`。
SEEDS = (42, 43, 44)
# 设置 `SEQUENCE_LENGTH` 的固定配置值，供后续流程统一使用。
SEQUENCE_LENGTH = 20



# 定义 `stable_hash` 函数，封装这一阶段可独立调用的处理步骤。
def stable_hash(payload: object) -> str:
    """对配置生成稳定 SHA256，防止错误续训。
    
    输入：
        payload: object。
    输出：
        str。
        """

    # 调用 `json.dumps` 把结构化对象序列化为 JSON 文本，并将结果保存到 `text`。
    text = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    # 将 `hashlib.sha256(text.encode('utf-8')).hexdigest()` 作为当前函数结果返回给调用方。
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# 定义 `atomic_write_text` 函数，封装这一阶段可独立调用的处理步骤。
def atomic_write_text(text: str, path: Path) -> None:
    """原子写入文本文件，避免中断留下半成品。
    
    输入：
        text: str。
        path: Path。
    输出：
        None。
        """
    # 调用 `path.parent.mkdir` 创建输出目录，确保后续文件可以安全写入。
    path.parent.mkdir(parents=True, exist_ok=True)
    # 调用 `path.with_name` 规范化文件路径，并将结果保存到 `temporary`。
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    # 调用 `temporary.write_text` 将当前结果持久化到目标文件。
    temporary.write_text(text, encoding="utf-8")
    # 调用 `temporary.replace` 用完整临时文件原子替换正式产物。
    temporary.replace(path)


# 定义 `atomic_write_json` 函数，封装这一阶段可独立调用的处理步骤。
def atomic_write_json(payload: dict[str, object], path: Path) -> None:
    """原子写入 JSON 文件，避免中断留下半成品。
    
    输入：
        payload: dict[str, object]。
        path: Path。
    输出：
        None。
        """
    # 调用 `atomic_write_text` 先写临时文件再原子替换正式产物，避免留下半成品。
    atomic_write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), path
    )


# 定义 `atomic_write_parquet` 函数，封装这一阶段可独立调用的处理步骤。
def atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    """原子写入 Parquet 文件，避免中断留下半成品。
    
    输入：
        frame: pd.DataFrame。
        path: Path。
    输出：
        None。
        """
    # 调用 `path.parent.mkdir` 创建输出目录，确保后续文件可以安全写入。
    path.parent.mkdir(parents=True, exist_ok=True)
    # 调用 `path.with_name` 规范化文件路径，并将结果保存到 `temporary`。
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    # 调用 `frame.to_parquet` 将当前结果持久化到目标文件。
    frame.to_parquet(temporary, index=False, compression="zstd")
    # 调用 `temporary.replace` 用完整临时文件原子替换正式产物。
    temporary.replace(path)


# 定义 `atomic_save_numpy` 函数，封装这一阶段可独立调用的处理步骤。
def atomic_save_numpy(array: np.ndarray, path: Path) -> None:
    """原子保存单个 NumPy 数组。
    
    输入：
        array: np.ndarray。
        path: Path。
    输出：
        None。
        """
    # 调用 `path.parent.mkdir` 创建输出目录，确保后续文件可以安全写入。
    path.parent.mkdir(parents=True, exist_ok=True)
    # 调用 `path.with_name` 规范化文件路径，并将结果保存到 `temporary`。
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    # 在 `temporary.open('wb')` 管理的上下文中执行本段逻辑，并保证资源按约定释放。
    with temporary.open("wb") as file:
        # 调用 `np.save` 将当前结果持久化到目标文件。
        np.save(file, array, allow_pickle=False)
    # 调用 `temporary.replace` 用完整临时文件原子替换正式产物。
    temporary.replace(path)


# 定义 `atomic_save_npz` 函数，封装这一阶段可独立调用的处理步骤。
def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    """原子保存多个具名 NumPy 数组。
    
    输入：
        path: Path。
        arrays: np.ndarray。
    输出：
        None。
        """
    # 调用 `path.parent.mkdir` 创建输出目录，确保后续文件可以安全写入。
    path.parent.mkdir(parents=True, exist_ok=True)
    # 调用 `path.with_name` 规范化文件路径，并将结果保存到 `temporary`。
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    # 在 `temporary.open('wb')` 管理的上下文中执行本段逻辑，并保证资源按约定释放。
    with temporary.open("wb") as file:
        # 调用 `np.savez_compressed` 压缩保存多个数组及其字段名。
        np.savez_compressed(file, **arrays)
    # 调用 `temporary.replace` 用完整临时文件原子替换正式产物。
    temporary.replace(path)


# 定义 `atomic_save_checkpoint` 函数，封装这一阶段可独立调用的处理步骤。
def atomic_save_checkpoint(payload: dict[str, object], path: Path) -> None:
    """原子保存 PyTorch checkpoint。
    
    输入：
        payload: dict[str, object]。
        path: Path。
    输出：
        None。
        """
    # 调用 `path.parent.mkdir` 创建输出目录，确保后续文件可以安全写入。
    path.parent.mkdir(parents=True, exist_ok=True)
    # 调用 `path.with_name` 规范化文件路径，并将结果保存到 `temporary`。
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    # 调用 `torch.save` 将当前结果持久化到目标文件。
    torch.save(payload, temporary)
    # 调用 `temporary.replace` 用完整临时文件原子替换正式产物。
    temporary.replace(path)


# 定义 `source_manifests` 函数，封装这一阶段可独立调用的处理步骤。
def source_manifests() -> tuple[dict[str, object], dict[str, object]]:
    """读取 ViT feature store 与滚动切分的上游清单。
    
    输入：
        无显式输入参数。
    输出：
        tuple[dict[str, object], dict[str, object]]。
        """
    # 调用 `require_source_data` 校验输入、配置或产物是否满足当前阶段的数据契约。
    require_source_data()
    # 调用 `json.loads` 读取源数据，并将结果保存到 `dataset`。
    dataset = json.loads(SOURCE_DATASET_MANIFEST_PATH.read_text(encoding="utf-8"))
    # 调用 `json.loads` 读取源数据，并将结果保存到 `rolling`。
    rolling = json.loads(SOURCE_ROLLING_MANIFEST_PATH.read_text(encoding="utf-8"))
    # 将 `(dataset, rolling)` 作为当前函数结果返回给调用方。
    return dataset, rolling


# 定义 `aligned_store_payload` 函数，封装这一阶段可独立调用的处理步骤。
def aligned_store_payload() -> dict[str, object]:
    """整理 ViViT 序列仓库的来源、数组形状和清洗规则。
    
    输入：
        无显式输入参数。
    输出：
        dict[str, object]。
        """
    # 调用 `source_manifests` 读取或生成记录数据来源与配置签名的 manifest，并将结果保存到 `(dataset, rolling)`。
    dataset, rolling = source_manifests()
    # 将 `{'schema_version': 2, 'source_dataset_signature': dataset['signature'], 'source_rolling_signatu…` 作为当前函数结果返回给调用方。
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




# 定义 `build_history_indices` 函数，封装这一阶段可独立调用的处理步骤。
def build_history_indices(
    sample_index: pd.DataFrame, output_path: Path = HISTORY_INDICES_PATH
) -> None:
    """为每个目标行构建固定 20 日槽位；精确日期缺失时保留 -1。
    
    输入：
        sample_index: pd.DataFrame。
        output_path: Path。
    输出：
        None。
        """

    # 调用 `sample_index['sample_index'].to_numpy` 把表格列转换为指定精度的 NumPy 数组，并将结果保存到 `sample_numbers`。
    sample_numbers = sample_index["sample_index"].to_numpy(dtype=np.int64)
    # 调用 `len` 汇总当前数据，并将统计量保存到 `sample_count`。
    sample_count = len(sample_index)
    # 检查 `sample_count >= np.iinfo(np.int32).max`；根据条件结果选择对应处理分支。
    if sample_count >= np.iinfo(np.int32).max:
        # 抛出 `OverflowError('feature store 行数超过 int32 history index 上限')`，阻止不满足数据契约的流程继续执行。
        raise OverflowError("feature store 行数超过 int32 history index 上限")
    # 检查 `not np.array_equal(sample_numbers, np.arange(sample_count, dtype=np.int64))`；根据条件结果选择对应处理分支。
    if not np.array_equal(sample_numbers, np.arange(sample_count, dtype=np.int64)):
        # 抛出 `AssertionError('sample_index 必须与 feature store 行号连续一致')`，阻止不满足数据契约的流程继续执行。
        raise AssertionError("sample_index 必须与 feature store 行号连续一致")

    # 调用 `output_path.parent.mkdir` 创建输出目录，确保后续文件可以安全写入。
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # 调用 `output_path.with_name` 规范化文件路径，并将结果保存到 `temporary`。
    temporary = output_path.with_name(
        f"{output_path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    # 检查 `temporary.exists()`；根据条件结果选择对应处理分支。
    if temporary.exists():
        # 调用 `temporary.unlink` 删除已确认不再需要的临时文件。
        temporary.unlink()
    # 调用 `np.lib.format.open_memmap` 以 NPY 格式打开内存映射数组，避免一次载入全部样本，并将结果保存到 `history`。
    history = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.int32,
        shape=(sample_count, SEQUENCE_LENGTH),
    )
    # 计算 `-1`，并将数值结果保存到 `history[:]`。
    history[:] = -1
    # 调用 `sample_numbers.astype` 统一设备或数据类型，并将结果保存到 `history[:, -1]`。
    history[:, -1] = sample_numbers.astype(np.int32)
    # 调用 `sample_index['date_id'].to_numpy` 把表格列转换为指定精度的 NumPy 数组，并将结果保存到 `date_ids`。
    date_ids = sample_index["date_id"].to_numpy(dtype=np.int32)

    # 遍历 `sample_index.groupby('instrument_id', sort=False).indices.values()`，并把当前元素绑定到 `positions` 逐项处理。
    for positions in sample_index.groupby("instrument_id", sort=False).indices.values():
        # 调用 `np.asarray` 转换数据表示，并将标准化对象保存到 `group_rows`。
        group_rows = np.asarray(positions, dtype=np.int64)
        # 读取 `date_ids[group_rows]` 的当前值，并赋给 `group_dates`。
        group_dates = date_ids[group_rows]
        # 读取 `sample_numbers[group_rows]` 的当前值，并赋给 `group_samples`。
        group_samples = sample_numbers[group_rows]
        # 检查 `len(group_dates) > 1 and np.any(np.diff(group_dates) <= 0)`；根据条件结果选择对应处理分支。
        if len(group_dates) > 1 and np.any(np.diff(group_dates) <= 0):
            # 抛出 `AssertionError('同一股票的 date_id 必须严格递增')`，阻止不满足数据契约的流程继续执行。
            raise AssertionError("同一股票的 date_id 必须严格递增")
        # 遍历 `range(1, SEQUENCE_LENGTH)`，并把当前元素绑定到 `lag` 逐项处理。
        for lag in range(1, SEQUENCE_LENGTH):
            # 计算 `group_dates - lag`，并将数值结果保存到 `desired_dates`。
            desired_dates = group_dates - lag
            # 调用 `np.searchsorted` 在有序交易日中定位历史窗口日期，并将结果保存到 `locations`。
            locations = np.searchsorted(group_dates, desired_dates)
            # 根据 `locations < len(group_dates)` 计算条件结果，并保存到 `in_bounds`。
            in_bounds = locations < len(group_dates)
            # 调用 `np.zeros` 按目标形状初始化数值容器 `matches`。
            matches = np.zeros(len(group_dates), dtype=bool)
            # 根据 `group_dates[locations[in_bounds]] == desired_dates[in_bounds]` 计算条件结果，并保存到 `matches[in_bounds]`。
            matches[in_bounds] = (
                group_dates[locations[in_bounds]] == desired_dates[in_bounds]
            )
            # 检查 `matches.any()`；根据条件结果选择对应处理分支。
            if matches.any():
                # 调用 `group_samples[locations[matches]].astype` 统一设备或数据类型，并将结果保存到 `history[group_samples[matches], SEQUENCE_LENGTH - 1 - lag]`。
                history[group_samples[matches], SEQUENCE_LENGTH - 1 - lag] = (
                    group_samples[locations[matches]].astype(np.int32)
                )

    # 调用 `history.flush` 把内存映射中的缓存修改刷新到磁盘。
    history.flush()
    # 删除 `history`，释放不再使用的引用。
    del history
    # 调用 `temporary.replace` 用完整临时文件原子替换正式产物。
    temporary.replace(output_path)


# 定义 `history_coverage_stats` 函数，封装这一阶段可独立调用的处理步骤。
def history_coverage_stats(
    history: np.ndarray, indices: np.ndarray, chunk_size: int = 200_000
) -> dict[str, object]:
    """统计序列历史帧覆盖率和缺失分布。
    
    输入：
        history: np.ndarray。
        indices: np.ndarray。
        chunk_size: int。
    输出：
        dict[str, object]。
        """
    # 调用 `np.asarray` 转换数据表示，并将标准化对象保存到 `indices`。
    indices = np.asarray(indices, dtype=np.int64)
    # 检查 `not len(indices)`；根据条件结果选择对应处理分支。
    if not len(indices):
        # 将 `{'sample_count': 0, 'complete_history_count': 0, 'incomplete_history_count': 0, 'valid_frame_ra…` 作为当前函数结果返回给调用方。
        return {
            "sample_count": 0,
            "complete_history_count": 0,
            "incomplete_history_count": 0,
            "valid_frame_ratio": None,
        }
    # 设置 `complete_count` 的固定配置值，供后续流程统一使用。
    complete_count = 0
    # 设置 `valid_frame_count` 的固定配置值，供后续流程统一使用。
    valid_frame_count = 0
    # 遍历 `range(0, len(indices), chunk_size)`，并把当前元素绑定到 `start` 逐项处理。
    for start in range(0, len(indices), chunk_size):
        # 调用 `np.asarray` 转换数据表示，并将标准化对象保存到 `selected`。
        selected = np.asarray(history[indices[start : start + chunk_size]])
        # 根据 `selected >= 0` 计算条件结果，并保存到 `valid`。
        valid = selected >= 0
        # 用 `int(valid.all(axis=1).sum())` 对 `complete_count` 做 `Add` 累积更新。
        complete_count += int(valid.all(axis=1).sum())
        # 用 `int(valid.sum())` 对 `valid_frame_count` 做 `Add` 累积更新。
        valid_frame_count += int(valid.sum())
    # 将 `{'sample_count': len(indices), 'complete_history_count': complete_count, 'incomplete_history_co…` 作为当前函数结果返回给调用方。
    return {
        "sample_count": len(indices),
        "complete_history_count": complete_count,
        "incomplete_history_count": len(indices) - complete_count,
        "valid_frame_ratio": valid_frame_count / (len(indices) * SEQUENCE_LENGTH),
    }


# 定义 `build_sequence_store` 函数，封装这一阶段可独立调用的处理步骤。
def build_sequence_store(overwrite: bool = False) -> dict[str, object]:
    """构建与 ViT 索引严格对齐、允许缺帧的 20 日历史矩阵。
    
    输入：
        overwrite: bool。
    输出：
        dict[str, object]。
        """

    # 调用 `aligned_store_payload` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `expected_payload`。
    expected_payload = aligned_store_payload()
    # 调用 `stable_hash` 对关键配置生成稳定签名以支持复现和一致性检查，并将结果保存到 `expected_signature`。
    expected_signature = stable_hash(expected_payload)
    # 组织一组有序值，并保存到 `expected_outputs`。
    expected_outputs = (
        HISTORY_INDICES_PATH,
        SEQUENCE_MANIFEST_PATH,
        SEQUENCE_SUMMARY_PATH,
    )
    # 检查 `not overwrite and SEQUENCE_MANIFEST_PATH.exists()`；根据条件结果选择对应处理分支。
    if not overwrite and SEQUENCE_MANIFEST_PATH.exists():
        # 调用 `json.loads` 读取源数据，并将结果保存到 `existing`。
        existing = json.loads(SEQUENCE_MANIFEST_PATH.read_text(encoding="utf-8"))
        # 检查 `existing.get('signature') != expected_signature`；根据条件结果选择对应处理分支。
        if existing.get("signature") != expected_signature:
            # 抛出 `ValueError('aligned sequence store 与当前 ViT 数据签名不一致；请显式使用 --overwrite 1 重建')`，阻止不满足数据契约的流程继续执行。
            raise ValueError(
                "aligned sequence store 与当前 ViT 数据签名不一致；"
                "请显式使用 --overwrite 1 重建"
            )
        # 检查 `all((path.exists() for path in expected_outputs))`；根据条件结果选择对应处理分支。
        if all(path.exists() for path in expected_outputs):
            # 调用 `np.load` 读取源数据，并将结果保存到 `history`。
            history = np.load(HISTORY_INDICES_PATH, mmap_mode="r")
            # 调用 `validate_history_indices` 校验输入、配置或产物是否满足当前阶段的数据契约。
            validate_history_indices(
                history, int(expected_payload["source_sample_count"])
            )
            # 将 `existing` 作为当前函数结果返回给调用方。
            return existing

    # 调用 `pd.read_parquet` 读取源数据，并将结果保存到 `sample_index`。
    sample_index = pd.read_parquet(
        SAMPLE_INDEX_PATH,
        columns=["sample_index", "trade_date", "instrument_id", "date_id"],
    )
    # 调用 `build_history_indices` 根据当前配置和输入构造后续流程所需对象。
    build_history_indices(sample_index)
    # 调用 `np.load` 读取源数据，并将结果保存到 `history`。
    history = np.load(HISTORY_INDICES_PATH, mmap_mode="r")
    # 调用 `validate_history_indices` 校验输入、配置或产物是否满足当前阶段的数据契约。
    validate_history_indices(history, len(sample_index))

    # 构造当前步骤所需的列表，并保存到 `summary_records`。
    summary_records: list[dict[str, object]] = []
    # 遍历 `FORECAST_YEARS`，并把当前元素绑定到 `year` 逐项处理。
    for year in FORECAST_YEARS:
        # 计算 `SOURCE_SPLIT_DIR / f'forecast_{year}.npz'`，并将数值结果保存到 `source_path`。
        source_path = SOURCE_SPLIT_DIR / f"forecast_{year}.npz"
        # 在 `np.load(source_path, allow_pickle=False)` 管理的上下文中执行本段逻辑，并保证资源按约定释放。
        with np.load(source_path, allow_pickle=False) as source:
            # 构造键值映射，并保存到 `split`，供后续按键查询。
            split = {name: source[name].copy() for name in source.files}
        # 构造键值映射，并保存到 `record`，供后续按键查询。
        record: dict[str, object] = {"forecast_year": year}
        # 遍历 `('train', 'validation', 'prediction')`，并把当前元素绑定到 `split_name` 逐项处理。
        for split_name in ("train", "validation", "prediction"):
            # 调用 `history_coverage_stats` 计算历史窗口覆盖或读取对应索引，并将结果保存到 `stats`。
            stats = history_coverage_stats(history, split[f"{split_name}_indices"])
            # 遍历 `stats.items()`，并把当前元素绑定到 `(name, value)` 逐项处理。
            for name, value in stats.items():
                # 读取 `value` 的当前值，并赋给 `record[f'{split_name}_{name}']`。
                record[f"{split_name}_{name}"] = value
        # 把本轮结果追加到 `summary_records`，累积后续需要汇总的记录。
        summary_records.append(record)

    # 调用 `pd.DataFrame` 转换数据表示，并将标准化对象保存到 `summary`。
    summary = pd.DataFrame(summary_records)
    # 调用 `atomic_write_parquet` 先写临时文件再原子替换正式产物，避免留下半成品。
    atomic_write_parquet(summary, SEQUENCE_SUMMARY_PATH)
    # 设置 `total_valid_frame_count` 的固定配置值，供后续流程统一使用。
    total_valid_frame_count = 0
    # 遍历 `range(0, len(history), 200000)`，并把当前元素绑定到 `start` 逐项处理。
    for start in range(0, len(history), 200_000):
        # 用 `int((np.asarray(history[start:start + 200000]) >= 0).sum())` 对 `total_valid_frame_count` 做 `Add` 累积更新。
        total_valid_frame_count += int(
            (np.asarray(history[start : start + 200_000]) >= 0).sum()
        )
    # 构造键值映射，并保存到 `payload`，供后续按键查询。
    payload = {
        "signature": expected_signature,
        "payload": expected_payload,
        "total_valid_frame_count": total_valid_frame_count,
        "total_frame_count": int(history.size),
        "missing_frame_count": int(history.size) - total_valid_frame_count,
    }
    # 调用 `atomic_write_json` 先写临时文件再原子替换正式产物，避免留下半成品。
    atomic_write_json(payload, SEQUENCE_MANIFEST_PATH)
    # 将 `payload` 作为当前函数结果返回给调用方。
    return payload


# 定义 `load_sequence_split` 函数，封装这一阶段可独立调用的处理步骤。
def load_sequence_split(forecast_year: int) -> dict[str, np.ndarray]:
    """直接加载 ViT split，不复制、不过滤任何目标索引。
    
    输入：
        forecast_year: int。
    输出：
        dict[str, np.ndarray]。
        """

    # 调用 `require_aligned_sequence_store` 校验输入、配置或产物是否满足当前阶段的数据契约。
    require_aligned_sequence_store()
    # 计算 `SOURCE_SPLIT_DIR / f'forecast_{forecast_year}.npz'`，并将数值结果保存到 `path`。
    path = SOURCE_SPLIT_DIR / f"forecast_{forecast_year}.npz"
    # 检查 `not path.exists()`；根据条件结果选择对应处理分支。
    if not path.exists():
        # 抛出 `FileNotFoundError(f'缺少 ViT 滚动切分：{path}')`，阻止不满足数据契约的流程继续执行。
        raise FileNotFoundError(f"缺少 ViT 滚动切分：{path}")
    # 在 `np.load(path, allow_pickle=False)` 管理的上下文中执行本段逻辑，并保证资源按约定释放。
    with np.load(path, allow_pickle=False) as archive:
        # 将 `{name: archive[name].copy() for name in archive.files}` 作为当前函数结果返回给调用方。
        return {name: archive[name].copy() for name in archive.files}




# 定义 `SampleIndexDataset` 类，集中封装该阶段的数据结构与相关行为。
class SampleIndexDataset(Dataset):
    """Dataset 只保存目标日的 feature store 行号。
    """

    # 定义初始化方法，接收配置并建立实例后续计算所需的内部状态。
    def __init__(self, sample_indices: np.ndarray) -> None:
        """初始化 SampleIndexDataset 所需的数据引用、配置和运行状态。
        
        输入：
            sample_indices: np.ndarray，格式 [N]。
        输出：
            None。
            """
        # 调用 `np.asarray` 转换数据表示，并将标准化对象保存到 `self.sample_indices`。
        self.sample_indices = np.asarray(sample_indices, dtype=np.int64)

    # 定义内部辅助函数 `__len__`，封装仅供本模块复用的计算步骤。
    def __len__(self) -> int:
        """返回 SampleIndexDataset 管理的样本数或批次数。
        
        输入：
            无显式输入参数。
        输出：
            int。
            """
        # 将 `len(self.sample_indices)` 作为当前函数结果返回给调用方。
        return len(self.sample_indices)

    # 定义内部辅助函数 `__getitem__`，封装仅供本模块复用的计算步骤。
    def __getitem__(self, position: int) -> int:
        """按给定索引返回 SampleIndexDataset 中的一个样本。
        
        输入：
            position: int。
        输出：
            int。
            """
        # 将 `int(self.sample_indices[position])` 作为当前函数结果返回给调用方。
        return int(self.sample_indices[position])


# 定义 `VideoBatchCollator` 类，集中封装该阶段的数据结构与相关行为。
class VideoBatchCollator:
    """按 aligned history matrix 读取视频，并对缺失历史帧补零。
    """

    # 定义初始化方法，接收配置并建立实例后续计算所需的内部状态。
    def __init__(self) -> None:
        """初始化 VideoBatchCollator 所需的数据引用、配置和运行状态。
        
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
        # 设置 `self.history_indices` 的固定配置值，供后续流程统一使用。
        self.history_indices = None

    # 定义内部辅助函数 `_ensure_open`，封装仅供本模块复用的计算步骤。
    def _ensure_open(self) -> None:
        """在当前 worker 首次取数时惰性打开共享 memmap，避免跨进程复制句柄。
        
        输入：
            无显式输入参数。
        输出：
            None。
            """
        # 检查 `self.features is None`；根据条件结果选择对应处理分支。
        if self.features is None:
            # 调用 `np.load` 读取源数据，并将结果保存到 `self.features`。
            self.features = np.load(FEATURES_PATH, mmap_mode="r")
            # 调用 `np.load` 读取源数据，并将结果保存到 `self.targets`。
            self.targets = np.load(TARGETS_PATH, mmap_mode="r")
            # 调用 `np.load` 读取源数据，并将结果保存到 `self.date_ids`。
            self.date_ids = np.load(DATE_IDS_PATH, mmap_mode="r")
            # 调用 `np.load` 读取源数据，并将结果保存到 `self.history_indices`。
            self.history_indices = np.load(HISTORY_INDICES_PATH, mmap_mode="r")

    # 定义内部辅助函数 `__call__`，封装仅供本模块复用的计算步骤。
    def __call__(self, sample_indices: list[int]) -> dict[str, torch.Tensor]:
        """根据历史行号组装视频、有效帧 mask、标签和样本元数据。
        
        输入：
            sample_indices: list[int]，格式 [N]。
        输出：
            dict[str, torch.Tensor]。
            """
        # 调用 `self._ensure_open` 执行 `_ensure_open` 对应的明确变换，并产生后续步骤使用的结果。
        self._ensure_open()
        # 调用 `np.asarray` 转换数据表示，并将标准化对象保存到 `target_indices`。
        target_indices = np.asarray(sample_indices, dtype=np.int64)
        # 调用 `np.array` 转换数据表示，并将标准化对象保存到 `sequence_indices`。
        sequence_indices = np.array(
            self.history_indices[target_indices], dtype=np.int64, copy=True
        )
        # 根据 `sequence_indices >= 0` 计算条件结果，并保存到 `frame_mask`。
        frame_mask = sequence_indices >= 0
        # 检查 `not frame_mask[:, -1].all()`；根据条件结果选择对应处理分支。
        if not frame_mask[:, -1].all():
            # 抛出 `AssertionError('目标日自身不能是缺失帧')`，阻止不满足数据契约的流程继续执行。
            raise AssertionError("目标日自身不能是缺失帧")

        # 调用 `np.zeros` 按目标形状初始化数值容器 `frame_array`。
        frame_array = np.zeros(
            (len(target_indices), SEQUENCE_LENGTH, 17, 8, 8), dtype=np.float32
        )
        # 读取 `sequence_indices[frame_mask]` 的当前值，并赋给 `valid_indices`。
        valid_indices = sequence_indices[frame_mask]
        # 检查 `len(valid_indices)`；根据条件结果选择对应处理分支。
        if len(valid_indices):
            # 调用 `np.array` 转换数据表示，并将标准化对象保存到 `frame_array[frame_mask]`。
            frame_array[frame_mask] = np.array(
                self.features[valid_indices], dtype=np.float32, copy=True, order="C"
            )
        # 调用 `np.ascontiguousarray` 把转置结果整理为连续内存布局，并将结果保存到 `video_array`。
        video_array = np.ascontiguousarray(frame_array.transpose(0, 2, 1, 3, 4))
        # 调用 `np.array` 转换数据表示，并将标准化对象保存到 `target_array`。
        target_array = np.array(
            self.targets[target_indices], dtype=np.float32, copy=True
        )
        # 调用 `np.array` 转换数据表示，并将标准化对象保存到 `date_id_array`。
        date_id_array = np.array(
            self.date_ids[target_indices], dtype=np.int32, copy=True
        )
        # 将 `{'videos': torch.from_numpy(video_array), 'frame_mask': torch.from_numpy(frame_mask), 'targets'…` 作为当前函数结果返回给调用方。
        return {
            "videos": torch.from_numpy(video_array),
            "frame_mask": torch.from_numpy(frame_mask),
            "targets": torch.from_numpy(target_array),
            "date_ids": torch.from_numpy(date_id_array),
            "sample_indices": torch.from_numpy(target_indices),
        }


# 定义 `make_data_loaders` 函数，封装这一阶段可独立调用的处理步骤。
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
    """创建与 ViT batch 语义一致的 train/validation/prediction loaders。
    
    输入：
        forecast_year: int。
        seed: int。
        batch_size: int。
        inference_batch_size: int。
        train_batch_mode: str。
        loss_name: str。
        num_workers: int。
        pin_memory: bool。
    输出：
        tuple[DataLoader, DataLoader, DataLoader | None]。
        """

    # 检查 `train_batch_mode not in TRAIN_BATCH_MODES`；根据条件结果选择对应处理分支。
    if train_batch_mode not in TRAIN_BATCH_MODES:
        # 抛出 `ValueError(f'未知训练 batch 模式：{train_batch_mode}')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(f"未知训练 batch 模式：{train_batch_mode}")
    # 根据 `loss_name == TEMPORAL_TURNOVER_LOSS` 计算条件结果，并保存到 `temporal_mode`。
    temporal_mode = loss_name == TEMPORAL_TURNOVER_LOSS
    # 检查 `temporal_mode and train_batch_mode != 'trade_date'`；根据条件结果选择对应处理分支。
    if temporal_mode and train_batch_mode != "trade_date":
        # 抛出 `ValueError('sigmoid_rank_ic_turnover 必须使用 trade_date batch')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("sigmoid_rank_ic_turnover 必须使用 trade_date batch")

    # 调用 `load_sequence_split` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `split`。
    split = load_sequence_split(forecast_year)
    # 构造键值映射，并保存到 `common`，供后续按键查询。
    common: dict[str, object] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": VideoBatchCollator(),
    }
    # 检查 `num_workers > 0`；根据条件结果选择对应处理分支。
    if num_workers > 0:
        # 用本轮键值结果更新 `common`，补齐后续查询所需字段。
        common.update({"persistent_workers": True, "prefetch_factor": 1})

    # 调用 `SampleIndexDataset` 构造或读取当前任务使用的数据集对象，并将结果保存到 `train_dataset`。
    train_dataset = SampleIndexDataset(split["train_indices"])
    # 调用 `torch.Generator().manual_seed` 创建并固定 DataLoader 使用的随机数生成器，并将结果保存到 `generator`。
    generator = torch.Generator().manual_seed(seed)
    # 检查 `train_batch_mode == 'trade_date'`；根据条件结果选择对应处理分支。
    if train_batch_mode == "trade_date":
        # 调用 `np.load` 读取源数据，并将结果保存到 `date_ids`。
        date_ids = np.load(DATE_IDS_PATH, mmap_mode="r")
        # 设置 `instrument_ids` 的固定配置值，供后续流程统一使用。
        instrument_ids = None
        # 检查 `temporal_mode`；根据条件结果选择对应处理分支。
        if temporal_mode:
            # 调用 `pd.read_parquet(SAMPLE_INDEX_PATH, columns=['instrument_id'])['…` 读取源数据，并将结果保存到 `instrument_ids`。
            instrument_ids = pd.read_parquet(
                SAMPLE_INDEX_PATH, columns=["instrument_id"]
            )["instrument_id"].to_numpy(copy=False)
        # 调用 `TradeDateBatchSampler` 按交易日完整截面构造 batch 采样器，并将结果保存到 `batch_sampler`。
        batch_sampler = TradeDateBatchSampler(
            split["train_indices"],
            date_ids,
            seed=seed,
            shuffle=not temporal_mode,
            instrument_ids=instrument_ids,
        )
        # 调用 `DataLoader` 按采样规则创建可迭代的训练或推理批次，并将结果保存到 `train_loader`。
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            generator=generator,
            **common,
        )
    # 当前置条件不成立时进入备用分支，保证其余合法情况仍被处理。
    else:
        # 调用 `DataLoader` 按采样规则创建可迭代的训练或推理批次，并将结果保存到 `train_loader`。
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            generator=generator,
            **common,
        )

    # 调用 `DataLoader` 按采样规则创建可迭代的训练或推理批次，并将结果保存到 `validation_loader`。
    validation_loader = DataLoader(
        SampleIndexDataset(split["validation_indices"]),
        batch_size=inference_batch_size,
        shuffle=False,
        drop_last=False,
        **common,
    )
    # 设置 `prediction_loader` 的固定配置值，供后续流程统一使用。
    prediction_loader = None
    # 检查 `len(split['prediction_indices'])`；根据条件结果选择对应处理分支。
    if len(split["prediction_indices"]):
        # 调用 `DataLoader` 按采样规则创建可迭代的训练或推理批次，并将结果保存到 `prediction_loader`。
        prediction_loader = DataLoader(
            SampleIndexDataset(split["prediction_indices"]),
            batch_size=inference_batch_size,
            shuffle=False,
            drop_last=False,
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


# 定义 `parse_args` 函数，封装这一阶段可独立调用的处理步骤。
def parse_args() -> argparse.Namespace:
    """解析阶段 01 的序列构建与切分审计命令。

    输入：
        无显式输入参数。
    输出：
        argparse.Namespace，包含命令名和覆盖开关。
    """
    # 调用 `argparse.ArgumentParser` 创建命令行参数解析器，并将结果保存到 `parser`。
    parser = argparse.ArgumentParser(description=__doc__)
    # 调用 `parser.add_subparsers` 创建子命令分发器以区分不同运行模式，并将结果保存到 `subparsers`。
    subparsers = parser.add_subparsers(dest="command", required=True)
    # 调用 `subparsers.add_parser` 注册一个可独立调用的命令行子命令，并将结果保存到 `build_parser`。
    build_parser = subparsers.add_parser("build-data")
    # 调用 `build_parser.add_argument` 根据当前配置和输入构造后续流程所需对象。
    build_parser.add_argument(
        "--overwrite", type=int, choices=(0, 1), default=0
    )
    # 调用 `subparsers.add_parser` 注册一个可独立调用的命令行子命令。
    subparsers.add_parser("audit-splits")
    # 将 `parser.parse_args()` 作为当前函数结果返回给调用方。
    return parser.parse_args()


# 定义命令行入口，解析参数并按用户选择调度当前阶段的主流程。
def main() -> None:
    """分派序列数据构建或 rolling split 对齐审计。

    输入：
        无显式输入参数。
    输出：
        None。
    """
    # 调用 `parse_args` 解析命令行文本并生成结构化参数，并将结果保存到 `args`。
    args = parse_args()
    # 检查 `args.command == 'build-data'`；根据条件结果选择对应处理分支。
    if args.command == "build-data":
        # 调用 `print` 记录当前流程状态或诊断信息。
        print(
            json.dumps(
                build_sequence_store(bool(args.overwrite)),
                ensure_ascii=False,
                indent=2,
            )
        )
    # 检查 `args.command == 'audit-splits'`；根据条件结果选择对应处理分支。
    elif args.command == "audit-splits":
        # 调用 `print` 记录当前流程状态或诊断信息。
        print(audit_split_alignment().to_string(index=False))


# 定义 `validate_history_indices` 函数，封装这一阶段可独立调用的处理步骤。
def validate_history_indices(
    history: np.ndarray, sample_count: int, chunk_size: int = 200_000
) -> None:
    """检查 20 帧历史索引只引用目标日及过去日期。
    
    输入：
        history: np.ndarray。
        sample_count: int。
        chunk_size: int。
    输出：
        None。
        """
    # 检查 `history.shape != (sample_count, SEQUENCE_LENGTH)`；根据条件结果选择对应处理分支。
    if history.shape != (sample_count, SEQUENCE_LENGTH):
        # 抛出 `ValueError(f'history_indices 形状异常：{history.shape}')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(f"history_indices 形状异常：{history.shape}")
    # 检查 `history.dtype != np.int32`；根据条件结果选择对应处理分支。
    if history.dtype != np.int32:
        # 抛出 `ValueError(f'history_indices dtype 异常：{history.dtype}')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(f"history_indices dtype 异常：{history.dtype}")
    # 遍历 `range(0, sample_count, chunk_size)`，并把当前元素绑定到 `start` 逐项处理。
    for start in range(0, sample_count, chunk_size):
        # 调用 `min` 汇总当前数据，并将统计量保存到 `stop`。
        stop = min(start + chunk_size, sample_count)
        # 调用 `np.asarray` 转换数据表示，并将标准化对象保存到 `block`。
        block = np.asarray(history[start:stop])
        # 调用 `np.arange` 按目标形状初始化数值容器 `expected`。
        expected = np.arange(start, stop, dtype=np.int32)
        # 检查 `not np.array_equal(block[:, -1], expected)`；根据条件结果选择对应处理分支。
        if not np.array_equal(block[:, -1], expected):
            # 抛出 `AssertionError('history_indices 最后一帧必须是目标样本自身')`，阻止不满足数据契约的流程继续执行。
            raise AssertionError("history_indices 最后一帧必须是目标样本自身")
        # 检查 `block.min(initial=-1) < -1 or block.max(initial=-1) >= sample_count`；根据条件结果选择对应处理分支。
        if block.min(initial=-1) < -1 or block.max(initial=-1) >= sample_count:
            # 抛出 `IndexError('history_indices 包含越界行号')`，阻止不满足数据契约的流程继续执行。
            raise IndexError("history_indices 包含越界行号")


# 定义 `audit_split_alignment` 函数，封装这一阶段可独立调用的处理步骤。
def audit_split_alignment() -> pd.DataFrame:
    """逐年逐元素审计运行时 split 与 ViT 源 split 完全相同。
    
    输入：
        无显式输入参数。
    输出：
        pd.DataFrame。
        """

    # 构造当前步骤所需的列表，并保存到 `records`。
    records: list[dict[str, object]] = []
    # 遍历 `FORECAST_YEARS`，并把当前元素绑定到 `year` 逐项处理。
    for year in FORECAST_YEARS:
        # 调用 `load_sequence_split` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `runtime`。
        runtime = load_sequence_split(year)
        # 在 `np.load(SOURCE_SPLIT_DIR / f'forecast_{year}.npz', al…` 管理的上下文中执行本段逻辑，并保证资源按约定释放。
        with np.load(
            SOURCE_SPLIT_DIR / f"forecast_{year}.npz", allow_pickle=False
        ) as source:
            # 遍历 `('train', 'validation', 'prediction')`，并把当前元素绑定到 `split_name` 逐项处理。
            for split_name in ("train", "validation", "prediction"):
                # 计算 `f'{split_name}_indices'`，并将结果写入 `key`。
                key = f"{split_name}_indices"
                # 调用 `np.array_equal` 逐元素检查两个数组是否完全一致，并将结果保存到 `equal`。
                equal = np.array_equal(runtime[key], source[key])
                # 把本轮结果追加到 `records`，累积后续需要汇总的记录。
                records.append(
                    {
                        "forecast_year": year,
                        "split": split_name,
                        "sample_count": len(runtime[key]),
                        "array_equal": bool(equal),
                    }
                )
                # 检查 `not equal`；根据条件结果选择对应处理分支。
                if not equal:
                    # 抛出 `AssertionError(f'{year} {split_name} 与 ViT 索引不一致')`，阻止不满足数据契约的流程继续执行。
                    raise AssertionError(f"{year} {split_name} 与 ViT 索引不一致")
    # 将 `pd.DataFrame(records)` 作为当前函数结果返回给调用方。
    return pd.DataFrame(records)


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


# 定义 `require_source_data` 函数，封装这一阶段可独立调用的处理步骤。
def require_source_data() -> None:
    """确认链接后的 ViT 数据与全部年度 split 完整存在。
    
    输入：
        无显式输入参数。
    输出：
        None。
        """

    # 组织一组有序值，并保存到 `required`。
    required = (
        FEATURES_PATH,
        TARGETS_PATH,
        DATE_IDS_PATH,
        SAMPLE_INDEX_PATH,
        SOURCE_DATASET_MANIFEST_PATH,
        SOURCE_ROLLING_MANIFEST_PATH,
        *(SOURCE_SPLIT_DIR / f"forecast_{year}.npz" for year in FORECAST_YEARS),
    )
    # 构造当前步骤所需的列表，并保存到 `missing`。
    missing = [str(path) for path in required if not path.exists()]
    # 检查 `missing`；根据条件结果选择对应处理分支。
    if missing:
        # 抛出 `FileNotFoundError(f'缺少 ViT 源数据：{missing}')`，阻止不满足数据契约的流程继续执行。
        raise FileNotFoundError(f"缺少 ViT 源数据：{missing}")


# 定义 `require_aligned_sequence_store` 函数，封装这一阶段可独立调用的处理步骤。
def require_aligned_sequence_store() -> dict[str, object]:
    """校验或按需构建与 ViT feature store 对齐的 ViViT 序列仓库。
    
    输入：
        无显式输入参数。
    输出：
        dict[str, object]。
        """
    # 检查 `not all((path.exists() for path in (HISTORY_INDICES_PATH, SEQUENCE_MANIFEST_PATH, SEQUENCE_SUMM…`；根据条件结果选择对应处理分支。
    if not all(
        path.exists()
        for path in (HISTORY_INDICES_PATH, SEQUENCE_MANIFEST_PATH, SEQUENCE_SUMMARY_PATH)
    ):
        # 抛出 `FileNotFoundError('缺少 aligned sequence store；请先运行 build-data')`，阻止不满足数据契约的流程继续执行。
        raise FileNotFoundError("缺少 aligned sequence store；请先运行 build-data")
    # 调用 `aligned_store_payload` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `expected`。
    expected = aligned_store_payload()
    # 调用 `json.loads` 读取源数据，并将结果保存到 `manifest`。
    manifest = json.loads(SEQUENCE_MANIFEST_PATH.read_text(encoding="utf-8"))
    # 检查 `manifest.get('signature') != stable_hash(expected)`；根据条件结果选择对应处理分支。
    if manifest.get("signature") != stable_hash(expected):
        # 抛出 `ValueError('aligned sequence store 与当前 ViT 数据签名不一致；请显式使用 build-data --overwrite 1 重建')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(
            "aligned sequence store 与当前 ViT 数据签名不一致；"
            "请显式使用 build-data --overwrite 1 重建"
        )
    # 调用 `np.load` 读取源数据，并将结果保存到 `history`。
    history = np.load(HISTORY_INDICES_PATH, mmap_mode="r")
    # 调用 `tuple` 整理数据结构，并将结果保存到 `expected_shape`。
    expected_shape = tuple(expected["history_shape"])
    # 检查 `history.shape != expected_shape or history.dtype != np.int32`；根据条件结果选择对应处理分支。
    if history.shape != expected_shape or history.dtype != np.int32:
        # 抛出 `ValueError(f'history_indices 元数据异常：shape={history.shape}, dtype={history.dtype}')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(
            f"history_indices 元数据异常：shape={history.shape}, dtype={history.dtype}"
        )
    # 将 `manifest` 作为当前函数结果返回给调用方。
    return manifest


# 检查 `__name__ == '__main__'`；根据条件结果选择对应处理分支。
if __name__ == "__main__":
    # 调用 `main` 启动当前模块的命令行主流程。
    main()
