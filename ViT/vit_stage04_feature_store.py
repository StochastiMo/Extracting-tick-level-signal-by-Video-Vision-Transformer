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
    """逐股票、逐通道对 8x8 像素做 Min-Max 归一化。"""

    values = pixel_batch.astype(np.float32, copy=False)
    minimum = values.min(axis=(-2, -1), keepdims=True)
    value_range = values.max(axis=(-2, -1), keepdims=True) - minimum
    normalized = values - minimum
    np.divide(normalized, value_range, out=normalized, where=value_range > 0)  # 仅对有效分母执行归一化
    normalized = np.where(value_range > 0, normalized, 0.0)  # 常数通道归零
    return np.ascontiguousarray(normalized, dtype=np.float32)

def build_feature_store() -> dict[str, object]:
    """保存 T 日有效 Universe 特征；标签缺失保留 NaN 并由 label_valid 标记。"""

    root = get_data_root()
    universe_dir = root / "universe"
    label_dir = root / "labels_t10"
    store_dir = root / "feature_store"
    store_dir.mkdir(parents=True, exist_ok=True)
    split_dir = store_dir / "rolling_splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    universe_manifest = read_json(universe_dir / "universe_manifest.json")
    label_manifest = read_json(label_dir / "labels_manifest.json")
    if universe_manifest is None or label_manifest is None:
        raise FileNotFoundError("请先依次运行 universe 和 labels")

    universe_paths = sorted(universe_dir.glob("universe.*.parquet"))
    image_paths = list_image_paths("VIT_MAX_FEATURE_DATES")
    if os.getenv("VIT_MAX_FEATURE_DATES"):
        selected_dates = set(image_paths)
    else:
        selected_dates = None
    yearly_frames: list[pd.DataFrame] = []
    for universe_path in universe_paths:
        year = universe_path.stem.split(".")[-1]
        universe = pd.read_parquet(
            universe_path,
            columns=[
                "trade_date", "instrument_id", "wind_code", "stock_index",
                "entry_trade_date", "exit_trade_date", "feature_valid",
                "in_universe_t", "can_enter_t1",
            ],
        )
        if selected_dates is not None:
            universe = universe.loc[universe["trade_date"].isin(selected_dates)]
        selected = universe.loc[
            universe["feature_valid"] & universe["in_universe_t"]
        ].copy()
        label_path = label_dir / f"labels_t10.{year}.parquet"
        if label_path.exists():
            labels = pd.read_parquet(
                label_path,
                columns=[
                    "trade_date", "instrument_id", "stock_index",
                    "forward_return_10d", "target_zscore",
                ],
            )
            selected = selected.merge(  # 按业务键对齐，避免依赖原始行序
                labels,
                on=["trade_date", "instrument_id", "stock_index"],
                how="left",
                validate="one_to_one",
            )
        else:
            selected["forward_return_10d"] = np.nan
            selected["target_zscore"] = np.nan
        selected["label_valid"] = np.isfinite(  # 排除 NaN 与 Inf
            selected["target_zscore"].to_numpy(dtype=np.float64)
        )
        yearly_frames.append(selected)

    samples = (
        pd.concat(yearly_frames, ignore_index=True)
        .sort_values(["trade_date", "stock_index"], kind="stable")
        .reset_index(drop=True)
    )
    if samples.empty:
        raise ValueError("T 日 Universe 中没有可用图片特征")
    if samples.duplicated(["trade_date", "instrument_id"]).any():
        raise ValueError("feature store 候选业务键重复")
    feature_dates = np.sort(samples["trade_date"].astype(str).unique())
    date_to_id = {date: index for index, date in enumerate(feature_dates)}
    samples["date_id"] = samples["trade_date"].map(date_to_id).astype(np.int32)

    features_path = store_dir / "features.npy"
    targets_path = store_dir / "targets.npy"
    label_valid_path = store_dir / "label_valid.npy"
    date_ids_path = store_dir / "date_ids.npy"
    sample_index_path = store_dir / "sample_index.parquet"
    dataset_manifest_path = store_dir / "dataset_manifest.json"
    rolling_manifest_path = store_dir / "rolling_split_manifest.json"
    rolling_summary_path = store_dir / "rolling_split_summary.parquet"
    selected_image_paths = [
        LEVEL2_DIR / f"stock_pixels_cn.{date}.npz" for date in feature_dates
    ]
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
    store_signature = stable_hash(payload)  # 生成数据复用签名
    existing = read_json(dataset_manifest_path)
    required_paths = [
        features_path, targets_path, label_valid_path, date_ids_path,
        sample_index_path,
    ]
    reusable = (
        not read_bool_environment("VIT_FEATURE_STORE_OVERWRITE")
        and existing is not None
        and existing.get("signature") == store_signature
        and all(path.exists() for path in required_paths)
    )

    if not reusable:
        print("开始构建与标签解耦的 float32 feature store。")
        temporary_paths = {
            path: path.with_name(f"{path.name}.{os.getpid()}.tmp")
            for path in [features_path, targets_path, label_valid_path, date_ids_path]
        }
        for path in temporary_paths.values():
            if path.exists():
                path.unlink()
        sample_count = len(samples)
        features = np.lib.format.open_memmap(  # 使用 memmap，避免整库载入内存
            temporary_paths[features_path], "w+", np.float32,
            shape=(sample_count, 17, 8, 8),
        )
        targets = np.lib.format.open_memmap(  # 使用 memmap，避免整库载入内存
            temporary_paths[targets_path], "w+", np.float32, shape=(sample_count,)
        )
        label_valid = np.lib.format.open_memmap(  # 使用 memmap，避免整库载入内存
            temporary_paths[label_valid_path], "w+", np.bool_, shape=(sample_count,)
        )
        date_ids = np.lib.format.open_memmap(  # 使用 memmap，避免整库载入内存
            temporary_paths[date_ids_path], "w+", np.int32, shape=(sample_count,)
        )
        targets[:] = np.nan
        cursor = 0
        index_frames: list[pd.DataFrame] = []
        for date_number, trade_date in enumerate(feature_dates, start=1):
            daily = samples.loc[samples["trade_date"].eq(trade_date)].sort_values(
                "stock_index", kind="stable"
            )
            stock_indices = daily["stock_index"].to_numpy(dtype=np.int64)
            with np.load(
                LEVEL2_DIR / f"stock_pixels_cn.{trade_date}.npz", allow_pickle=False
            ) as archive:
                if stock_indices.min() < 0 or stock_indices.max() >= len(
                    archive["instrument_ids"]
                ):
                    raise IndexError(f"{trade_date} stock_index 越界")
                image_ids = archive["instrument_ids"][stock_indices].astype(str)
                if not np.array_equal(
                    image_ids, daily["instrument_id"].to_numpy(dtype=str)
                ):
                    raise ValueError(f"{trade_date} Universe 与 NPZ 股票行错位")
                if not archive["valid_mask"][stock_indices].astype(bool).all():
                    raise ValueError(f"{trade_date} feature store 混入无效图片")
                pixels = normalize_pixels(archive["pixel_matrices"][stock_indices])  # 仅使用当日截面统计量
            next_cursor = cursor + len(daily)
            features[cursor:next_cursor] = pixels
            targets[cursor:next_cursor] = daily["target_zscore"].to_numpy(
                dtype=np.float32
            )
            label_valid[cursor:next_cursor] = daily["label_valid"].to_numpy(bool)
            date_ids[cursor:next_cursor] = daily["date_id"].to_numpy(np.int32)
            daily_index = daily[
                [
                    "trade_date", "instrument_id", "wind_code", "stock_index",
                    "entry_trade_date", "exit_trade_date", "date_id",
                    "can_enter_t1", "label_valid",
                ]
            ].copy()
            daily_index.insert(
                0, "sample_index", np.arange(cursor, next_cursor, dtype=np.int64)
            )
            index_frames.append(daily_index)
            cursor = next_cursor
            if date_number % 50 == 0 or date_number == len(feature_dates):
                print(
                    f"特征 {date_number}/{len(feature_dates)} 日，累计 {cursor:,} 条"
                )
        if cursor != sample_count:
            raise AssertionError("feature store 写入行数不一致")
        for array in (features, targets, label_valid, date_ids):
            array.flush()  # 将 memmap 修改刷入磁盘
        del features, targets, label_valid, date_ids
        for final_path, temporary_path in temporary_paths.items():
            temporary_path.replace(final_path)
        sample_index = pd.concat(index_frames, ignore_index=True)
        atomic_parquet(sample_index, sample_index_path)  # 原子落盘，避免半成品
        atomic_json(  # 原子落盘，避免半成品
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
    else:
        print("feature store 签名一致，直接复用。")
        sample_index = pd.read_parquet(sample_index_path)

    sample_index["trade_date"] = sample_index["trade_date"].astype(str)
    sample_index["exit_trade_date"] = sample_index["exit_trade_date"].astype(str)
    if not np.array_equal(
        sample_index["sample_index"].to_numpy(np.int64),
        np.arange(len(sample_index), dtype=np.int64),
    ):
        raise ValueError("sample_index 不是连续全局行号")
    targets_check = np.load(targets_path, mmap_mode="r")
    label_valid_check = np.load(label_valid_path, mmap_mode="r")
    if not np.array_equal(
        np.isfinite(targets_check), label_valid_check.astype(bool)
    ):
        raise AssertionError("label_valid 与 targets 的有限性不一致")
    del targets_check, label_valid_check

    # 训练/验证只含有效标签；预测包含该年度全部 T 日有效 Universe 特征。
    labeled = sample_index.loc[sample_index["label_valid"]].copy()
    date_max_exit = labeled.groupby("trade_date", sort=True)["exit_trade_date"].max()
    first_year = int(sample_index["trade_date"].min()[:4])
    last_year = int(sample_index["trade_date"].max()[:4])
    forecast_years = list(range(first_year + 1, last_year + 2))
    rolling_payload = {
        "schema_version": 3,
        "feature_store_signature": store_signature,
        "forecast_years": forecast_years,
        "train_date_ratio": 0.8,
        "train_and_validation": "label_valid only",
        "prediction": "all feature_valid and in_universe_t rows in forecast year",
        "purge": "max T+11 must precede next partition boundary",
    }
    rolling_signature = stable_hash(rolling_payload)  # 生成数据复用签名
    rolling_existing = read_json(rolling_manifest_path)
    split_paths = {
        year: split_dir / f"forecast_{year}.npz" for year in forecast_years
    }
    split_reusable = (
        not read_bool_environment("VIT_FEATURE_STORE_OVERWRITE")
        and rolling_existing is not None
        and rolling_existing.get("signature") == rolling_signature
        and rolling_summary_path.exists()
        and all(path.exists() for path in split_paths.values())
    )
    if not split_reusable:
        rolling_records = []
        for forecast_year in forecast_years:
            forecast_start = f"{forecast_year}0101"
            eligible_dates = date_max_exit.loc[
                (date_max_exit.index < forecast_start)
                & date_max_exit.lt(forecast_start)
            ].index.to_numpy(dtype=str)
            if len(eligible_dates) < 2:
                raise ValueError(f"{forecast_year} 历史标签日期不足")
            split_position = min(
                max(int(np.floor(len(eligible_dates) * 0.8)), 1),
                len(eligible_dates) - 1,
            )
            raw_train_dates = eligible_dates[:split_position]
            validation_dates = eligible_dates[split_position:]
            validation_start = str(validation_dates[0])
            train_dates = date_max_exit.loc[raw_train_dates]
            train_dates = train_dates.loc[
                train_dates.lt(validation_start)
            ].index.to_numpy(dtype=str)
            train_mask = sample_index["label_valid"] & sample_index[
                "trade_date"
            ].isin(train_dates)
            validation_mask = sample_index["label_valid"] & sample_index[
                "trade_date"
            ].isin(validation_dates)
            prediction_mask = sample_index["trade_date"].str.startswith(
                str(forecast_year)
            )
            train_indices = sample_index.loc[train_mask, "sample_index"].to_numpy(
                np.int64
            )
            validation_indices = sample_index.loc[
                validation_mask, "sample_index"
            ].to_numpy(np.int64)
            prediction_indices = sample_index.loc[
                prediction_mask, "sample_index"
            ].to_numpy(np.int64)
            if not sample_index.iloc[train_indices]["label_valid"].all():
                raise AssertionError("训练索引混入无标签样本")
            if not sample_index.iloc[validation_indices]["label_valid"].all():
                raise AssertionError("验证索引混入无标签样本")
            if np.intersect1d(train_indices, validation_indices).size:
                raise AssertionError("训练和验证索引重叠")
            max_train_exit = sample_index.loc[train_mask, "exit_trade_date"].max()
            max_validation_exit = sample_index.loc[
                validation_mask, "exit_trade_date"
            ].max()
            if max_train_exit >= validation_start:
                raise AssertionError("训练标签越过验证边界")
            if max_validation_exit >= forecast_start:
                raise AssertionError("验证标签越过预测边界")
            atomic_npz(
                split_paths[forecast_year],
                train_indices=train_indices,
                validation_indices=validation_indices,
                prediction_indices=prediction_indices,
            )
            prediction_dates = np.sort(
                sample_index.loc[prediction_mask, "trade_date"].unique()
            )
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
        rolling_summary = pd.DataFrame(rolling_records)
        atomic_parquet(rolling_summary, rolling_summary_path)  # 原子落盘，避免半成品
        atomic_json(  # 原子落盘，避免半成品
            {"signature": rolling_signature, "payload": rolling_payload},
            rolling_manifest_path,
        )
    else:
        print("年度滚动切分签名一致，直接复用。")
        rolling_summary = pd.read_parquet(rolling_summary_path)

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
    return {
        "feature_store_directory": str(store_dir),
        "dataset_manifest_path": str(dataset_manifest_path),
        "rolling_manifest_path": str(rolling_manifest_path),
        "rolling_summary_path": str(rolling_summary_path),
        "signature": store_signature,
    }


def main() -> None:
    """构建 feature store 并打印当前产物清单。"""
    print(json.dumps(build_feature_store(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
