"""阶段 04：构建单日图像 feature store、样本索引与无泄漏滚动切分。"""
from __future__ import annotations


import json
import os

import numpy as np
import pandas as pd
from IPython.display import display

from vit_stage01_foundation import (
    LEVEL2_DIR, atomic_json, atomic_npz, atomic_parquet, file_signature,
    get_data_root, read_bool_environment, read_json, stable_hash,
)
from vit_stage02_universe import list_image_paths


# ===================== 04｜构建特征与滚动切分 =====================
def normalize_pixels(pixel_batch: np.ndarray) -> np.ndarray:
    """逐股票、逐通道对 8x8 像素做 Min-Max 归一化。

    输入：
        pixel_batch: np.ndarray。
    输出：
        np.ndarray。
    """

    # 调用 `pixel_batch.astype` 统一设备或数据类型，并将结果保存到 `values`。
    values = pixel_batch.astype(np.float32, copy=False)
    # 调用 `values.min` 汇总当前数据，并将统计量保存到 `minimum`。
    minimum = values.min(axis=(-2, -1), keepdims=True)
    # 计算 `values.max(axis=(-2, -1), keepdims=True) - minimum`，并将数值结果保存到 `value_range`。
    value_range = values.max(axis=(-2, -1), keepdims=True) - minimum
    # 计算 `values - minimum`，并将数值结果保存到 `normalized`。
    normalized = values - minimum
    # 调用 `np.divide` 按掩码执行安全除法并处理无效分母。
    np.divide(normalized, value_range, out=normalized, where=value_range > 0)
    # 调用 `np.where` 根据条件在标准化值与零之间逐元素选择，并将结果保存到 `normalized`。
    normalized = np.where(value_range > 0, normalized, 0.0)
    # 将 `np.ascontiguousarray(normalized, dtype=np.float32)` 作为当前函数结果返回给调用方。
    return np.ascontiguousarray(normalized, dtype=np.float32)

# 定义 `build_feature_store` 函数，封装这一阶段可独立调用的处理步骤。
def build_feature_store() -> dict[str, object]:
    """保存 T 日有效 Universe 特征；标签缺失保留 NaN 并由 label_valid 标记。

    输入：
        无显式输入参数。
    输出：
        dict[str, object]。
    """

    # 调用 `get_data_root` 解析当前 run 对应的数据与输出路径，并将结果保存到 `root`。
    root = get_data_root()
    # 计算 `root / 'universe'`，并将数值结果保存到 `universe_dir`。
    universe_dir = root / "universe"
    # 计算 `root / 'labels_t10'`，并将数值结果保存到 `label_dir`。
    label_dir = root / "labels_t10"
    # 计算 `root / 'feature_store'`，并将数值结果保存到 `store_dir`。
    store_dir = root / "feature_store"
    # 调用 `store_dir.mkdir` 创建输出目录，确保后续文件可以安全写入。
    store_dir.mkdir(parents=True, exist_ok=True)
    # 计算 `store_dir / 'rolling_splits'`，并将数值结果保存到 `split_dir`。
    split_dir = store_dir / "rolling_splits"
    # 调用 `split_dir.mkdir` 创建输出目录，确保后续文件可以安全写入。
    split_dir.mkdir(parents=True, exist_ok=True)
    # 调用 `read_json` 读取源数据，并将结果保存到 `universe_manifest`。
    universe_manifest = read_json(universe_dir / "universe_manifest.json")
    # 调用 `read_json` 读取源数据，并将结果保存到 `label_manifest`。
    label_manifest = read_json(label_dir / "labels_manifest.json")
    # 检查 `universe_manifest is None or label_manifest is None`；根据条件结果选择对应处理分支。
    if universe_manifest is None or label_manifest is None:
        # 抛出 `FileNotFoundError('请先依次运行 universe 和 labels')`，阻止不满足数据契约的流程继续执行。
        raise FileNotFoundError("请先依次运行 universe 和 labels")

    # 调用 `sorted` 整理数据结构，并将结果保存到 `universe_paths`。
    universe_paths = sorted(universe_dir.glob("universe.*.parquet"))
    # 调用 `list_image_paths` 解析当前 run 对应的数据与输出路径，并将结果保存到 `image_paths`。
    image_paths = list_image_paths("VIT_MAX_FEATURE_DATES")
    # 检查 `os.getenv('VIT_MAX_FEATURE_DATES')`；根据条件结果选择对应处理分支。
    if os.getenv("VIT_MAX_FEATURE_DATES"):
        # 调用 `set` 整理数据结构，并将结果保存到 `selected_dates`。
        selected_dates = set(image_paths)
    # 当前置条件不成立时进入备用分支，保证其余合法情况仍被处理。
    else:
        # 设置 `selected_dates` 的固定配置值，供后续流程统一使用。
        selected_dates = None
    # 构造当前步骤所需的列表，并保存到 `yearly_frames`。
    yearly_frames: list[pd.DataFrame] = []
    # 遍历 `universe_paths`，并把当前元素绑定到 `universe_path` 逐项处理。
    for universe_path in universe_paths:
        # 读取 `universe_path.stem.split('.')[-1]` 的当前值，并赋给 `year`。
        year = universe_path.stem.split(".")[-1]
        # 调用 `pd.read_parquet` 读取源数据，并将结果保存到 `universe`。
        universe = pd.read_parquet(
            universe_path,
            columns=[
                "trade_date", "instrument_id", "wind_code", "stock_index",
                "entry_trade_date", "exit_trade_date", "feature_valid",
                "in_universe_t", "can_enter_t1",
            ],
        )
        # 检查 `selected_dates is not None`；根据条件结果选择对应处理分支。
        if selected_dates is not None:
            # 读取 `universe.loc[universe['trade_date'].isin(selected_dates)]` 的当前值，并赋给 `universe`。
            universe = universe.loc[universe["trade_date"].isin(selected_dates)]
        # 调用 `copy` 复制当前对象以避免后续原地修改污染源数据，并将结果保存到 `selected`。
        selected = universe.loc[
            universe["feature_valid"] & universe["in_universe_t"]
        ].copy()
        # 计算 `label_dir / f'labels_t10.{year}.parquet'`，并将数值结果保存到 `label_path`。
        label_path = label_dir / f"labels_t10.{year}.parquet"
        # 检查 `label_path.exists()`；根据条件结果选择对应处理分支。
        if label_path.exists():
            # 调用 `pd.read_parquet` 读取源数据，并将结果保存到 `labels`。
            labels = pd.read_parquet(
                label_path,
                columns=[
                    "trade_date", "instrument_id", "stock_index",
                    "forward_return_10d", "target_zscore",
                ],
            )
            # 调用 `selected.merge` 按键合并数据，并将对齐后的结果保存到 `selected`。
            selected = selected.merge(
                labels,
                on=["trade_date", "instrument_id", "stock_index"],
                how="left",
                validate="one_to_one",
            )
        # 当前置条件不成立时进入备用分支，保证其余合法情况仍被处理。
        else:
            # 读取 `np.nan` 的当前值，并赋给 `selected['forward_return_10d']`。
            selected["forward_return_10d"] = np.nan
            # 读取 `np.nan` 的当前值，并赋给 `selected['target_zscore']`。
            selected["target_zscore"] = np.nan
        # 调用 `np.isfinite` 判断数值是否为有限值以排除 NaN 和无穷大，并将结果保存到 `selected['label_valid']`。
        selected["label_valid"] = np.isfinite(
            selected["target_zscore"].to_numpy(dtype=np.float64)
        )
        # 把本轮结果追加到 `yearly_frames`，累积后续需要汇总的记录。
        yearly_frames.append(selected)

    # 调用 `pd.concat(yearly_frames, ignore_index=True).sort_values(['trade…` 根据分数计算截面排序或排序索引，并将结果保存到 `samples`。
    samples = (
        pd.concat(yearly_frames, ignore_index=True)
        .sort_values(["trade_date", "stock_index"], kind="stable")
        .reset_index(drop=True)
    )
    # 检查 `samples.empty`；根据条件结果选择对应处理分支。
    if samples.empty:
        # 抛出 `ValueError('T 日 Universe 中没有可用图片特征')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("T 日 Universe 中没有可用图片特征")
    # 检查 `samples.duplicated(['trade_date', 'instrument_id']).any()`；根据条件结果选择对应处理分支。
    if samples.duplicated(["trade_date", "instrument_id"]).any():
        # 抛出 `ValueError('feature store 候选业务键重复')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("feature store 候选业务键重复")
    # 调用 `np.sort` 执行 `sort` 对应的明确变换，并产生后续步骤使用的结果，并将结果保存到 `feature_dates`。
    feature_dates = np.sort(samples["trade_date"].astype(str).unique())
    # 构造键值映射，并保存到 `date_to_id`，供后续按键查询。
    date_to_id = {date: index for index, date in enumerate(feature_dates)}
    # 调用 `samples['trade_date'].map(date_to_id).astype` 统一设备或数据类型，并将结果保存到 `samples['date_id']`。
    samples["date_id"] = samples["trade_date"].map(date_to_id).astype(np.int32)

    # 计算 `store_dir / 'features.npy'`，并将数值结果保存到 `features_path`。
    features_path = store_dir / "features.npy"
    # 计算 `store_dir / 'targets.npy'`，并将数值结果保存到 `targets_path`。
    targets_path = store_dir / "targets.npy"
    # 计算 `store_dir / 'label_valid.npy'`，并将数值结果保存到 `label_valid_path`。
    label_valid_path = store_dir / "label_valid.npy"
    # 计算 `store_dir / 'date_ids.npy'`，并将数值结果保存到 `date_ids_path`。
    date_ids_path = store_dir / "date_ids.npy"
    # 计算 `store_dir / 'sample_index.parquet'`，并将数值结果保存到 `sample_index_path`。
    sample_index_path = store_dir / "sample_index.parquet"
    # 计算 `store_dir / 'dataset_manifest.json'`，并将数值结果保存到 `dataset_manifest_path`。
    dataset_manifest_path = store_dir / "dataset_manifest.json"
    # 计算 `store_dir / 'rolling_split_manifest.json'`，并将数值结果保存到 `rolling_manifest_path`。
    rolling_manifest_path = store_dir / "rolling_split_manifest.json"
    # 计算 `store_dir / 'rolling_split_summary.parquet'`，并将数值结果保存到 `rolling_summary_path`。
    rolling_summary_path = store_dir / "rolling_split_summary.parquet"
    # 构造当前步骤所需的列表，并保存到 `selected_image_paths`。
    selected_image_paths = [
        LEVEL2_DIR / f"stock_pixels_cn.{date}.npz" for date in feature_dates
    ]
    # 构造键值映射，并保存到 `payload`，供后续按键查询。
    payload = {
        "schema_version": 4,
        "universe_signature": universe_manifest["signature"],
        "label_signature": label_manifest["signature"],
        "image_files_hash": stable_hash(
            [file_signature(path) for path in selected_image_paths]
        ),
        "feature_dates": feature_dates.tolist(),
        "sample_rule": "feature_valid and in_universe_t",
        "target_rule": "NaN unless a valid winsorized label exists",
        "normalization": "per_stock_per_channel_minmax_on_8x8",
        "feature_dtype": "float32",
        "feature_shape_tail": [17, 8, 8],
    }
    # 调用 `stable_hash` 对关键配置生成稳定签名以支持复现和一致性检查，并将结果保存到 `store_signature`。
    store_signature = stable_hash(payload)
    # 调用 `read_json` 读取源数据，并将结果保存到 `existing`。
    existing = read_json(dataset_manifest_path)
    # 构造当前步骤所需的列表，并保存到 `required_paths`。
    required_paths = [
        features_path, targets_path, label_valid_path, date_ids_path,
        sample_index_path,
    ]
    # 根据 `not read_bool_environment('VIT_FEATURE_STORE_OVERWRITE') and existing is not None and (existing…` 计算条件结果，并保存到 `reusable`。
    reusable = (
        not read_bool_environment("VIT_FEATURE_STORE_OVERWRITE")
        and existing is not None
        and existing.get("signature") == store_signature
        and all(path.exists() for path in required_paths)
    )

    # 检查 `not reusable`；根据条件结果选择对应处理分支。
    if not reusable:
        # 调用 `print` 记录当前流程状态或诊断信息。
        print("开始构建与标签解耦的 float32 feature store。")
        # 构造键值映射，并保存到 `temporary_paths`，供后续按键查询。
        temporary_paths = {
            path: path.with_name(f"{path.name}.{os.getpid()}.tmp")
            for path in [features_path, targets_path, label_valid_path, date_ids_path]
        }
        # 遍历 `temporary_paths.values()`，并把当前元素绑定到 `path` 逐项处理。
        for path in temporary_paths.values():
            # 检查 `path.exists()`；根据条件结果选择对应处理分支。
            if path.exists():
                # 调用 `path.unlink` 解析当前 run 对应的数据与输出路径。
                path.unlink()
        # 调用 `len` 汇总当前数据，并将统计量保存到 `sample_count`。
        sample_count = len(samples)
        # 调用 `np.lib.format.open_memmap` 以 NPY 格式打开内存映射数组，避免一次载入全部样本，并将结果保存到 `features`。
        features = np.lib.format.open_memmap(
            temporary_paths[features_path], "w+", np.float32,
            shape=(sample_count, 17, 8, 8),
        )
        # 调用 `np.lib.format.open_memmap` 以 NPY 格式打开内存映射数组，避免一次载入全部样本，并将结果保存到 `targets`。
        targets = np.lib.format.open_memmap(
            temporary_paths[targets_path], "w+", np.float32, shape=(sample_count,)
        )
        # 调用 `np.lib.format.open_memmap` 以 NPY 格式打开内存映射数组，避免一次载入全部样本，并将结果保存到 `label_valid`。
        label_valid = np.lib.format.open_memmap(
            temporary_paths[label_valid_path], "w+", np.bool_, shape=(sample_count,)
        )
        # 调用 `np.lib.format.open_memmap` 以 NPY 格式打开内存映射数组，避免一次载入全部样本，并将结果保存到 `date_ids`。
        date_ids = np.lib.format.open_memmap(
            temporary_paths[date_ids_path], "w+", np.int32, shape=(sample_count,)
        )
        # 读取 `np.nan` 的当前值，并赋给 `targets[:]`。
        targets[:] = np.nan
        # 设置 `cursor` 的固定配置值，供后续流程统一使用。
        cursor = 0
        # 构造当前步骤所需的列表，并保存到 `index_frames`。
        index_frames: list[pd.DataFrame] = []
        # 遍历 `enumerate(feature_dates, start=1)`，并把当前元素绑定到 `(date_number, trade_date)` 逐项处理。
        for date_number, trade_date in enumerate(feature_dates, start=1):
            # 调用 `samples.loc[samples['trade_date'].eq(trade_date)].sort_values` 根据分数计算截面排序或排序索引，并将结果保存到 `daily`。
            daily = samples.loc[samples["trade_date"].eq(trade_date)].sort_values(
                "stock_index", kind="stable"
            )
            # 调用 `daily['stock_index'].to_numpy` 把表格列转换为指定精度的 NumPy 数组，并将结果保存到 `stock_indices`。
            stock_indices = daily["stock_index"].to_numpy(dtype=np.int64)
            # 在 `np.load(LEVEL2_DIR / f'stock_pixels_cn.{trade_date}.n…` 管理的上下文中执行本段逻辑，并保证资源按约定释放。
            with np.load(
                LEVEL2_DIR / f"stock_pixels_cn.{trade_date}.npz", allow_pickle=False
            ) as archive:
                # 检查 `stock_indices.min() < 0 or stock_indices.max() >= len(archive['instrument_ids'])`；根据条件结果选择对应处理分支。
                if stock_indices.min() < 0 or stock_indices.max() >= len(
                    archive["instrument_ids"]
                ):
                    # 抛出 `IndexError(f'{trade_date} stock_index 越界')`，阻止不满足数据契约的流程继续执行。
                    raise IndexError(f"{trade_date} stock_index 越界")
                # 调用 `archive['instrument_ids'][stock_indices].astype` 统一设备或数据类型，并将结果保存到 `image_ids`。
                image_ids = archive["instrument_ids"][stock_indices].astype(str)
                # 检查 `not np.array_equal(image_ids, daily['instrument_id'].to_numpy(dtype=str))`；根据条件结果选择对应处理分支。
                if not np.array_equal(
                    image_ids, daily["instrument_id"].to_numpy(dtype=str)
                ):
                    # 抛出 `ValueError(f'{trade_date} Universe 与 NPZ 股票行错位')`，阻止不满足数据契约的流程继续执行。
                    raise ValueError(f"{trade_date} Universe 与 NPZ 股票行错位")
                # 检查 `not archive['valid_mask'][stock_indices].astype(bool).all()`；根据条件结果选择对应处理分支。
                if not archive["valid_mask"][stock_indices].astype(bool).all():
                    # 抛出 `ValueError(f'{trade_date} feature store 混入无效图片')`，阻止不满足数据契约的流程继续执行。
                    raise ValueError(f"{trade_date} feature store 混入无效图片")
                # 调用 `normalize_pixels` 按当前截面统计量完成缩尾或标准化处理，并将结果保存到 `pixels`。
                pixels = normalize_pixels(archive["pixel_matrices"][stock_indices])
            # 计算 `cursor + len(daily)`，并将数值结果保存到 `next_cursor`。
            next_cursor = cursor + len(daily)
            # 读取 `pixels` 的当前值，并赋给 `features[cursor:next_cursor]`。
            features[cursor:next_cursor] = pixels
            # 调用 `daily['target_zscore'].to_numpy` 把表格列转换为指定精度的 NumPy 数组，并将结果保存到 `targets[cursor:next_cursor]`。
            targets[cursor:next_cursor] = daily["target_zscore"].to_numpy(
                dtype=np.float32
            )
            # 调用 `daily['label_valid'].to_numpy` 把表格列转换为指定精度的 NumPy 数组，并将结果保存到 `label_valid[cursor:next_cursor]`。
            label_valid[cursor:next_cursor] = daily["label_valid"].to_numpy(bool)
            # 调用 `daily['date_id'].to_numpy` 把表格列转换为指定精度的 NumPy 数组，并将结果保存到 `date_ids[cursor:next_cursor]`。
            date_ids[cursor:next_cursor] = daily["date_id"].to_numpy(np.int32)
            # 调用 `copy` 复制当前对象以避免后续原地修改污染源数据，并将结果保存到 `daily_index`。
            daily_index = daily[
                [
                    "trade_date", "instrument_id", "wind_code", "stock_index",
                    "entry_trade_date", "exit_trade_date", "date_id",
                    "can_enter_t1", "label_valid",
                ]
            ].copy()
            # 调用 `daily_index.insert` 把当前字段插入指定位置并保持输出列顺序。
            daily_index.insert(
                0, "sample_index", np.arange(cursor, next_cursor, dtype=np.int64)
            )
            # 把本轮结果追加到 `index_frames`，累积后续需要汇总的记录。
            index_frames.append(daily_index)
            # 读取 `next_cursor` 的当前值，并赋给 `cursor`。
            cursor = next_cursor
            # 检查 `date_number % 50 == 0 or date_number == len(feature_dates)`；根据条件结果选择对应处理分支。
            if date_number % 50 == 0 or date_number == len(feature_dates):
                # 调用 `print` 记录当前流程状态或诊断信息。
                print(
                    f"特征 {date_number}/{len(feature_dates)} 日，累计 {cursor:,} 条"
                )
        # 检查 `cursor != sample_count`；根据条件结果选择对应处理分支。
        if cursor != sample_count:
            # 抛出 `AssertionError('feature store 写入行数不一致')`，阻止不满足数据契约的流程继续执行。
            raise AssertionError("feature store 写入行数不一致")
        # 遍历 `(features, targets, label_valid, date_ids)`，并把当前元素绑定到 `array` 逐项处理。
        for array in (features, targets, label_valid, date_ids):
            # 调用 `array.flush` 把内存映射中的缓存修改刷新到磁盘。
            array.flush()
        # 删除 `features, targets, label_valid, date_ids`，释放不再使用的引用。
        del features, targets, label_valid, date_ids
        # 遍历 `temporary_paths.items()`，并把当前元素绑定到 `(final_path, temporary_path)` 逐项处理。
        for final_path, temporary_path in temporary_paths.items():
            # 调用 `temporary_path.replace` 解析当前 run 对应的数据与输出路径。
            temporary_path.replace(final_path)
        # 调用 `pd.concat` 拼接各批结果，并将合并值保存到 `sample_index`。
        sample_index = pd.concat(index_frames, ignore_index=True)
        # 调用 `atomic_parquet` 先写临时文件再原子替换正式产物，避免留下半成品。
        atomic_parquet(sample_index, sample_index_path)
        # 调用 `atomic_json` 先写临时文件再原子替换正式产物，避免留下半成品。
        atomic_json(
            {
                "signature": store_signature,
                "payload": payload,
                "sample_count": sample_count,
                "label_valid_count": int(sample_index["label_valid"].sum()),
                "first_sample_date": str(feature_dates[0]),
                "last_sample_date": str(feature_dates[-1]),
            },
            dataset_manifest_path,
        )
    # 当前置条件不成立时进入备用分支，保证其余合法情况仍被处理。
    else:
        # 调用 `print` 记录当前流程状态或诊断信息。
        print("feature store 签名一致，直接复用。")
        # 调用 `pd.read_parquet` 读取源数据，并将结果保存到 `sample_index`。
        sample_index = pd.read_parquet(sample_index_path)

    # 调用 `sample_index['trade_date'].astype` 统一设备或数据类型，并将结果保存到 `sample_index['trade_date']`。
    sample_index["trade_date"] = sample_index["trade_date"].astype(str)
    # 调用 `sample_index['exit_trade_date'].astype` 统一设备或数据类型，并将结果保存到 `sample_index['exit_trade_date']`。
    sample_index["exit_trade_date"] = sample_index["exit_trade_date"].astype(str)
    # 检查 `not np.array_equal(sample_index['sample_index'].to_numpy(np.int64), np.arange(len(sample_index)…`；根据条件结果选择对应处理分支。
    if not np.array_equal(
        sample_index["sample_index"].to_numpy(np.int64),
        np.arange(len(sample_index), dtype=np.int64),
    ):
        # 抛出 `ValueError('sample_index 不是连续全局行号')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("sample_index 不是连续全局行号")
    # 调用 `np.load` 读取源数据，并将结果保存到 `targets_check`。
    targets_check = np.load(targets_path, mmap_mode="r")
    # 调用 `np.load` 读取源数据，并将结果保存到 `label_valid_check`。
    label_valid_check = np.load(label_valid_path, mmap_mode="r")
    # 检查 `not np.array_equal(np.isfinite(targets_check), label_valid_check.astype(bool))`；根据条件结果选择对应处理分支。
    if not np.array_equal(
        np.isfinite(targets_check), label_valid_check.astype(bool)
    ):
        # 抛出 `AssertionError('label_valid 与 targets 的有限性不一致')`，阻止不满足数据契约的流程继续执行。
        raise AssertionError("label_valid 与 targets 的有限性不一致")
    # 删除 `targets_check, label_valid_check`，释放不再使用的引用。
    del targets_check, label_valid_check

    # 训练/验证只含有效标签；预测包含该年度全部 T 日有效 Universe 特征。
    labeled = sample_index.loc[sample_index["label_valid"]].copy()
    # 调用 `labeled.groupby('trade_date', sort=True)['exit_trade_date'].max` 汇总当前数据，并将统计量保存到 `date_max_exit`。
    date_max_exit = labeled.groupby("trade_date", sort=True)["exit_trade_date"].max()
    # 调用 `int` 统一设备或数据类型，并将结果保存到 `first_year`。
    first_year = int(sample_index["trade_date"].min()[:4])
    # 调用 `int` 统一设备或数据类型，并将结果保存到 `last_year`。
    last_year = int(sample_index["trade_date"].max()[:4])
    # 调用 `list` 整理数据结构，并将结果保存到 `forecast_years`。
    forecast_years = list(range(first_year + 1, last_year + 2))
    # 构造键值映射，并保存到 `rolling_payload`，供后续按键查询。
    rolling_payload = {
        "schema_version": 3,
        "feature_store_signature": store_signature,
        "forecast_years": forecast_years,
        "train_date_ratio": 0.8,
        "train_and_validation": "label_valid only",
        "prediction": "all feature_valid and in_universe_t rows in forecast year",
        "purge": "max T+11 must precede next partition boundary",
    }
    # 调用 `stable_hash` 对关键配置生成稳定签名以支持复现和一致性检查，并将结果保存到 `rolling_signature`。
    rolling_signature = stable_hash(rolling_payload)
    # 调用 `read_json` 读取源数据，并将结果保存到 `rolling_existing`。
    rolling_existing = read_json(rolling_manifest_path)
    # 构造键值映射，并保存到 `split_paths`，供后续按键查询。
    split_paths = {
        year: split_dir / f"forecast_{year}.npz" for year in forecast_years
    }
    # 根据 `not read_bool_environment('VIT_FEATURE_STORE_OVERWRITE') and rolling_existing is not None and (…` 计算条件结果，并保存到 `split_reusable`。
    split_reusable = (
        not read_bool_environment("VIT_FEATURE_STORE_OVERWRITE")
        and rolling_existing is not None
        and rolling_existing.get("signature") == rolling_signature
        and rolling_summary_path.exists()
        and all(path.exists() for path in split_paths.values())
    )
    # 检查 `not split_reusable`；根据条件结果选择对应处理分支。
    if not split_reusable:
        # 构造当前步骤所需的列表，并保存到 `rolling_records`。
        rolling_records = []
        # 遍历 `forecast_years`，并把当前元素绑定到 `forecast_year` 逐项处理。
        for forecast_year in forecast_years:
            # 计算 `f'{forecast_year}0101'`，并将结果写入 `forecast_start`。
            forecast_start = f"{forecast_year}0101"
            # 调用 `to_numpy` 把表格列转换为指定精度的 NumPy 数组，并将结果保存到 `eligible_dates`。
            eligible_dates = date_max_exit.loc[
                (date_max_exit.index < forecast_start)
                & date_max_exit.lt(forecast_start)
            ].index.to_numpy(dtype=str)
            # 检查 `len(eligible_dates) < 2`；根据条件结果选择对应处理分支。
            if len(eligible_dates) < 2:
                # 抛出 `ValueError(f'{forecast_year} 历史标签日期不足')`，阻止不满足数据契约的流程继续执行。
                raise ValueError(f"{forecast_year} 历史标签日期不足")
            # 调用 `min` 汇总当前数据，并将统计量保存到 `split_position`。
            split_position = min(
                max(int(np.floor(len(eligible_dates) * 0.8)), 1),
                len(eligible_dates) - 1,
            )
            # 读取 `eligible_dates[:split_position]` 的当前值，并赋给 `raw_train_dates`。
            raw_train_dates = eligible_dates[:split_position]
            # 读取 `eligible_dates[split_position:]` 的当前值，并赋给 `validation_dates`。
            validation_dates = eligible_dates[split_position:]
            # 调用 `str` 把当前值转换为统一字符串表示，并将结果保存到 `validation_start`。
            validation_start = str(validation_dates[0])
            # 读取 `date_max_exit.loc[raw_train_dates]` 的当前值，并赋给 `train_dates`。
            train_dates = date_max_exit.loc[raw_train_dates]
            # 调用 `train_dates.loc[train_dates.lt(validation_start)].index.to_numpy` 执行模型训练并记录本轮优化指标，并将结果保存到 `train_dates`。
            train_dates = train_dates.loc[
                train_dates.lt(validation_start)
            ].index.to_numpy(dtype=str)
            # 计算 `sample_index['label_valid'] & sample_index['trade_date'].isin(train_dates)`，并将数值结果保存到 `train_mask`。
            train_mask = sample_index["label_valid"] & sample_index[
                "trade_date"
            ].isin(train_dates)
            # 计算 `sample_index['label_valid'] & sample_index['trade_date'].isin(validation_dates)`，并将数值结果保存到 `validation_mask`。
            validation_mask = sample_index["label_valid"] & sample_index[
                "trade_date"
            ].isin(validation_dates)
            # 调用 `sample_index['trade_date'].str.startswith` 判断日期文本是否属于目标预测年份，并将结果保存到 `prediction_mask`。
            prediction_mask = sample_index["trade_date"].str.startswith(
                str(forecast_year)
            )
            # 调用 `sample_index.loc[train_mask, 'sample_index'].to_numpy` 执行模型训练并记录本轮优化指标，并将结果保存到 `train_indices`。
            train_indices = sample_index.loc[train_mask, "sample_index"].to_numpy(
                np.int64
            )
            # 调用 `sample_index.loc[validation_mask, 'sample_index'].to_numpy` 根据有效性条件筛选当前样本或构造掩码，并将结果保存到 `validation_indices`。
            validation_indices = sample_index.loc[
                validation_mask, "sample_index"
            ].to_numpy(np.int64)
            # 调用 `sample_index.loc[prediction_mask, 'sample_index'].to_numpy` 执行模型推理并生成样本预测分数，并将结果保存到 `prediction_indices`。
            prediction_indices = sample_index.loc[
                prediction_mask, "sample_index"
            ].to_numpy(np.int64)
            # 检查 `not sample_index.iloc[train_indices]['label_valid'].all()`；根据条件结果选择对应处理分支。
            if not sample_index.iloc[train_indices]["label_valid"].all():
                # 抛出 `AssertionError('训练索引混入无标签样本')`，阻止不满足数据契约的流程继续执行。
                raise AssertionError("训练索引混入无标签样本")
            # 检查 `not sample_index.iloc[validation_indices]['label_valid'].all()`；根据条件结果选择对应处理分支。
            if not sample_index.iloc[validation_indices]["label_valid"].all():
                # 抛出 `AssertionError('验证索引混入无标签样本')`，阻止不满足数据契约的流程继续执行。
                raise AssertionError("验证索引混入无标签样本")
            # 检查 `np.intersect1d(train_indices, validation_indices).size`；根据条件结果选择对应处理分支。
            if np.intersect1d(train_indices, validation_indices).size:
                # 抛出 `AssertionError('训练和验证索引重叠')`，阻止不满足数据契约的流程继续执行。
                raise AssertionError("训练和验证索引重叠")
            # 调用 `sample_index.loc[train_mask, 'exit_trade_date'].max` 汇总当前数据，并将统计量保存到 `max_train_exit`。
            max_train_exit = sample_index.loc[train_mask, "exit_trade_date"].max()
            # 调用 `sample_index.loc[validation_mask, 'exit_trade_date'].max` 汇总当前数据，并将统计量保存到 `max_validation_exit`。
            max_validation_exit = sample_index.loc[
                validation_mask, "exit_trade_date"
            ].max()
            # 检查 `max_train_exit >= validation_start`；根据条件结果选择对应处理分支。
            if max_train_exit >= validation_start:
                # 抛出 `AssertionError('训练标签越过验证边界')`，阻止不满足数据契约的流程继续执行。
                raise AssertionError("训练标签越过验证边界")
            # 检查 `max_validation_exit >= forecast_start`；根据条件结果选择对应处理分支。
            if max_validation_exit >= forecast_start:
                # 抛出 `AssertionError('验证标签越过预测边界')`，阻止不满足数据契约的流程继续执行。
                raise AssertionError("验证标签越过预测边界")
            # 调用 `atomic_npz` 执行 `atomic_npz` 对应的明确变换，并产生后续步骤使用的结果。
            atomic_npz(
                split_paths[forecast_year],
                train_indices=train_indices,
                validation_indices=validation_indices,
                prediction_indices=prediction_indices,
            )
            # 调用 `np.sort` 执行 `sort` 对应的明确变换，并产生后续步骤使用的结果，并将结果保存到 `prediction_dates`。
            prediction_dates = np.sort(
                sample_index.loc[prediction_mask, "trade_date"].unique()
            )
            # 把本轮结果追加到 `rolling_records`，累积后续需要汇总的记录。
            rolling_records.append(
                {
                    "forecast_year": forecast_year,
                    "history_start_date": str(eligible_dates[0]),
                    "history_end_date": str(eligible_dates[-1]),
                    "history_date_count": len(eligible_dates),
                    "raw_train_date_count": len(raw_train_dates),
                    "purged_train_date_count": len(raw_train_dates) - len(train_dates),
                    "train_start_date": str(train_dates[0]),
                    "train_end_date": str(train_dates[-1]),
                    "train_date_count": len(train_dates),
                    "train_sample_count": len(train_indices),
                    "max_train_exit_date": str(max_train_exit),
                    "validation_start_date": validation_start,
                    "validation_end_date": str(validation_dates[-1]),
                    "validation_date_count": len(validation_dates),
                    "validation_sample_count": len(validation_indices),
                    "max_validation_exit_date": str(max_validation_exit),
                    "forecast_start_date": forecast_start,
                    "prediction_start_date": (
                        str(prediction_dates[0]) if len(prediction_dates) else None
                    ),
                    "prediction_end_date": (
                        str(prediction_dates[-1]) if len(prediction_dates) else None
                    ),
                    "prediction_date_count": len(prediction_dates),
                    "prediction_sample_count": len(prediction_indices),
                }
            )
        # 调用 `pd.DataFrame` 转换数据表示，并将标准化对象保存到 `rolling_summary`。
        rolling_summary = pd.DataFrame(rolling_records)
        # 调用 `atomic_parquet` 先写临时文件再原子替换正式产物，避免留下半成品。
        atomic_parquet(rolling_summary, rolling_summary_path)
        # 调用 `atomic_json` 先写临时文件再原子替换正式产物，避免留下半成品。
        atomic_json(
            {"signature": rolling_signature, "payload": rolling_payload},
            rolling_manifest_path,
        )
    # 当前置条件不成立时进入备用分支，保证其余合法情况仍被处理。
    else:
        # 调用 `print` 记录当前流程状态或诊断信息。
        print("年度滚动切分签名一致，直接复用。")
        # 调用 `pd.read_parquet` 读取源数据，并将结果保存到 `rolling_summary`。
        rolling_summary = pd.read_parquet(rolling_summary_path)

    # 调用 `display` 在 Notebook 输出区展示当前表格或图形。
    display(
        rolling_summary[
            [
                "forecast_year", "train_sample_count", "validation_sample_count",
                "prediction_sample_count", "max_train_exit_date",
                "validation_start_date", "max_validation_exit_date",
                "forecast_start_date",
            ]
        ]
    )
    # 将 `{'feature_store_directory': str(store_dir), 'dataset_manifest_path': str(dataset_manifest_path)…` 作为当前函数结果返回给调用方。
    return {
        "feature_store_directory": str(store_dir),
        "dataset_manifest_path": str(dataset_manifest_path),
        "rolling_manifest_path": str(rolling_manifest_path),
        "rolling_summary_path": str(rolling_summary_path),
        "signature": store_signature,
    }


# 定义命令行入口，解析参数并按用户选择调度当前阶段的主流程。
def main() -> None:
    """构建 feature store 并打印当前产物清单。

    输入：
        无显式输入参数。
    输出：
        None。
    """
    # 调用 `print` 记录当前流程状态或诊断信息。
    print(json.dumps(build_feature_store(), ensure_ascii=False, indent=2))


# 检查 `__name__ == '__main__'`；根据条件结果选择对应处理分支。
if __name__ == "__main__":
    # 调用 `main` 启动当前模块的命令行主流程。
    main()
