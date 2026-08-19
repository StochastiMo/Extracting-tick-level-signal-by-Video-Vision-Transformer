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
def normalize_pixels(pixel_batch: np.ndarray) -> np.ndarray:  # 逐股票、逐通道对 8x8 像素做 Min-Max 归一化。
    """逐股票、逐通道对 8x8 像素做 Min-Max 归一化。"""

    values = pixel_batch.astype(np.float32, copy=False)  # 转换值的数据类型或存储位置。
    minimum = values.min(axis=(-2, -1), keepdims=True)  # 计算minimum。
    value_range = values.max(axis=(-2, -1), keepdims=True) - minimum  # 计算值range。
    normalized = values - minimum  # 计算标准化。
    np.divide(normalized, value_range, out=normalized, where=value_range > 0)  # 仅对有效分母执行归一化。
    normalized = np.where(value_range > 0, normalized, 0.0)  # 常数通道归零。
    return np.ascontiguousarray(normalized, dtype=np.float32)  # 返回逐股票、逐通道对 8x8 像素做 Min-Max 归一化。

def build_feature_store() -> dict[str, object]:  # 保存 T 日有效 Universe 特征。
    """保存 T 日有效 Universe 特征；标签缺失保留 NaN 并由 label_valid 标记。"""

    root = get_data_root()  # 解析可由环境变量覆盖的数据根目录。
    universe_dir = root / "universe"  # 构造股票池目录。
    label_dir = root / "labels_t10"  # 构造标签目录。
    store_dir = root / "feature_store"  # 构造存储目录。
    store_dir.mkdir(parents=True, exist_ok=True)  # 确保目标目录存在。
    split_dir = store_dir / "rolling_splits"  # 构造切分目录。
    split_dir.mkdir(parents=True, exist_ok=True)  # 确保目标目录存在。
    universe_manifest = read_json(universe_dir / "universe_manifest.json")  # 读取股票池清单。
    label_manifest = read_json(label_dir / "labels_manifest.json")  # 读取标签清单。
    if universe_manifest is None or label_manifest is None:  # 检查所需对象是否尚未初始化。
        raise FileNotFoundError("请先依次运行 universe 和 labels")  # 目标文件不存在时立即终止。

    universe_paths = sorted(universe_dir.glob("universe.*.parquet"))  # 计算股票池paths以固定样本顺序。
    image_paths = list_image_paths("VIT_MAX_FEATURE_DATES")  # 计算图像paths。
    if os.getenv("VIT_MAX_FEATURE_DATES"):  # 根据当前数据状态选择处理分支。
        selected_dates = set(image_paths)  # 构造筛选结果交易日，限定后续参与计算的样本。
    else:  # 处理上述条件均不成立的情况。
        selected_dates = None  # 先把筛选结果交易日初始化为空，后续按条件填充。
    yearly_frames: list[pd.DataFrame] = []  # 接收yearly帧。
    for universe_path in universe_paths:  # 遍历股票池paths，逐项处理股票池路径。
        year = universe_path.stem.split(".")[-1]  # 计算年份。
        universe = pd.read_parquet(  # 读取股票池。
            universe_path,  # 汇入股票池路径。
            columns=[  # 初始化列容器。
                "trade_date", "instrument_id", "wind_code", "stock_index",
                "entry_trade_date", "exit_trade_date", "feature_valid",
                "in_universe_t", "can_enter_t1",
            ],
        )
        if selected_dates is not None:  # 检查必需数据、配置或校验结果是否缺失。
            universe = universe.loc[universe["trade_date"].isin(selected_dates)]  # 从上游结果中读取股票池清单。
        selected = universe.loc[  # 构造筛选结果，限定后续参与计算的样本。
            universe["feature_valid"] & universe["in_universe_t"]
        ].copy()
        label_path = label_dir / f"labels_t10.{year}.parquet"  # 构造标签路径。
        if label_path.exists():  # 检查目标文件或目录是否满足存在性要求。
            labels = pd.read_parquet(  # 读取标签。
                label_path,  # 汇入标签路径。
                columns=[  # 初始化列容器。
                    "trade_date", "instrument_id", "stock_index",
                    "forward_return_10d", "target_zscore",
                ],
            )
            selected = selected.merge(  # 按业务键对齐，避免依赖原始行序。
                labels,  # 汇入标签。
                on=["trade_date", "instrument_id", "stock_index"],  # 指定数据连接使用的业务键。
                how="left",  # 指定数据连接方式。
                validate="one_to_one",  # 校验数据连接的基数关系。
            )
        else:  # 处理上述条件均不成立的情况。
            selected["forward_return_10d"] = np.nan  # 构造筛选结果，限定后续参与计算的样本。
            selected["target_zscore"] = np.nan  # 构造筛选结果，限定后续参与计算的样本。
        selected["label_valid"] = np.isfinite(  # 排除 NaN 与 Inf。
            selected["target_zscore"].to_numpy(dtype=np.float64)
        )
        yearly_frames.append(selected)  # 把当前结果追加到汇总列表。

    samples = (
        pd.concat(yearly_frames, ignore_index=True)
        .sort_values(["trade_date", "stock_index"], kind="stable")
        .reset_index(drop=True)
    )
    if samples.empty:  # 检查结果表是否为空。
        raise ValueError("T 日 Universe 中没有可用图片特征")  # 输入数据不满足业务或数值约束时立即终止。
    if samples.duplicated(["trade_date", "instrument_id"]).any():  # 检查当前掩码或数据是否满足有效性要求。
        raise ValueError("feature store 候选业务键重复")  # 输入数据不满足业务或数值约束时立即终止。
    feature_dates = np.sort(samples["trade_date"].astype(str).unique())  # 转换特征交易日的数据类型或存储位置。
    date_to_id = {date: index for index, date in enumerate(feature_dates)}  # 初始化交易日toid容器。
    samples["date_id"] = samples["trade_date"].map(date_to_id).astype(np.int32)  # 转换样本的数据类型或存储位置。

    features_path = store_dir / "features.npy"  # 构造特征路径。
    targets_path = store_dir / "targets.npy"  # 构造标签路径。
    label_valid_path = store_dir / "label_valid.npy"  # 构造标签有效路径。
    date_ids_path = store_dir / "date_ids.npy"  # 构造交易日ids路径。
    sample_index_path = store_dir / "sample_index.parquet"  # 构造样本索引路径。
    dataset_manifest_path = store_dir / "dataset_manifest.json"  # 构造dataset清单路径。
    rolling_manifest_path = store_dir / "rolling_split_manifest.json"  # 构造rolling清单路径。
    rolling_summary_path = store_dir / "rolling_split_summary.parquet"  # 构造rolling汇总路径。
    selected_image_paths = [  # 构造筛选结果图像paths，限定后续参与计算的样本。
        LEVEL2_DIR / f"stock_pixels_cn.{date}.npz" for date in feature_dates
    ]
    payload = {  # 初始化载荷容器。
        "schema_version": 4,  # 记录schema版本。
        "universe_signature": universe_manifest["signature"],  # 记录股票池签名，用于来源一致性校验。
        "label_signature": label_manifest["signature"],  # 记录标签签名，用于来源一致性校验。
        "image_files_hash": stable_hash(  # 记录图像files哈希。
            [file_signature(path) for path in selected_image_paths]
        ),
        "feature_dates": feature_dates.tolist(),  # 记录特征交易日。
        "sample_rule": "feature_valid and in_universe_t",  # 把 sample_rule 映射为"feature_valid and in_universe_t"。
        "target_rule": "NaN unless a valid winsorized label exists",  # 把 target_rule 映射为"NaN unless a valid winsorized label exists"。
        "normalization": "per_stock_per_channel_minmax_on_8x8",  # 把 normalization 映射为"per_stock_per_channel_minmax_on_8x8"。
        "feature_dtype": "float32",  # 把 feature_dtype 映射为"float32"。
        "feature_shape_tail": [17, 8, 8],  # 记录特征形状tail。
    }
    store_signature = stable_hash(payload)  # 生成数据复用签名。
    existing = read_json(dataset_manifest_path)  # 读取existing。
    required_paths = [  # 初始化必需paths容器。
        features_path, targets_path, label_valid_path, date_ids_path,  # 传入特征路径、标签路径、标签有效路径、交易日ids路径。
        sample_index_path,  # 汇入样本索引路径。
    ]
    reusable = (
        not read_bool_environment("VIT_FEATURE_STORE_OVERWRITE")
        and existing is not None
        and existing.get("signature") == store_signature
        and all(path.exists() for path in required_paths)
    )

    if not reusable:  # 检查必需数据、配置或校验结果是否缺失。
        print("开始构建与标签解耦的 float32 feature store。")  # 输出当前任务的进度或诊断信息。
        temporary_paths = {  # 初始化临时文件paths容器。
            path: path.with_name(f"{path.name}.{os.getpid()}.tmp")  # 接收路径。
            for path in [features_path, targets_path, label_valid_path, date_ids_path]  # 遍历，逐项处理路径。
        }
        for path in temporary_paths.values():  # 遍历临时文件paths值，逐项处理路径。
            if path.exists():  # 检查目标文件或目录是否满足存在性要求。
                path.unlink()
        sample_count = len(samples)  # 计算样本数量。
        features = np.lib.format.open_memmap(  # 使用 memmap，避免整库载入内存。
            temporary_paths[features_path], "w+", np.float32,
            shape=(sample_count, 17, 8, 8),  # 配置形状。
        )
        targets = np.lib.format.open_memmap(  # 使用 memmap，避免整库载入内存。
            temporary_paths[targets_path], "w+", np.float32, shape=(sample_count,)
        )
        label_valid = np.lib.format.open_memmap(  # 使用 memmap，避免整库载入内存。
            temporary_paths[label_valid_path], "w+", np.bool_, shape=(sample_count,)
        )
        date_ids = np.lib.format.open_memmap(  # 使用 memmap，避免整库载入内存。
            temporary_paths[date_ids_path], "w+", np.int32, shape=(sample_count,)
        )
        targets[:] = np.nan
        cursor = 0
        index_frames: list[pd.DataFrame] = []  # 接收索引帧。
        for date_number, trade_date in enumerate(feature_dates, start=1):  # 遍历enumerate特征交易日起点1，逐项处理交易日numbertrade交易日。
            daily = samples.loc[samples["trade_date"].eq(trade_date)].sort_values(  # 计算daily以固定样本顺序。
                "stock_index", kind="stable"
            )
            stock_indices = daily["stock_index"].to_numpy(dtype=np.int64)  # 转换并统一stock索引的数据表示。
            with np.load(  # 进入受控上下文并在退出时自动释放资源。
                LEVEL2_DIR / f"stock_pixels_cn.{trade_date}.npz", allow_pickle=False
            ) as archive:
                if stock_indices.min() < 0 or stock_indices.max() >= len(  # 检查当前样本数量是否满足计算要求。
                    archive["instrument_ids"]  # 汇入archive。
                ):
                    raise IndexError(f"{trade_date} stock_index 越界")  # 当前数据违反接口约定时立即终止。
                image_ids = archive["instrument_ids"][stock_indices].astype(str)  # 转换图像ids的数据类型或存储位置。
                if not np.array_equal(  # 检查必需数据、配置或校验结果是否缺失。
                    image_ids, daily["instrument_id"].to_numpy(dtype=str)
                ):
                    raise ValueError(f"{trade_date} Universe 与 NPZ 股票行错位")  # 输入数据不满足业务或数值约束时立即终止。
                if not archive["valid_mask"][stock_indices].astype(bool).all():  # 检查当前掩码或数据是否满足有效性要求。
                    raise ValueError(f"{trade_date} feature store 混入无效图片")  # 输入数据不满足业务或数值约束时立即终止。
                pixels = normalize_pixels(archive["pixel_matrices"][stock_indices])  # 仅使用当日截面统计量。
            next_cursor = cursor + len(daily)
            features[cursor:next_cursor] = pixels
            targets[cursor:next_cursor] = daily["target_zscore"].to_numpy(  # 转换并统一标签的数据表示。
                dtype=np.float32
            )
            label_valid[cursor:next_cursor] = daily["label_valid"].to_numpy(bool)  # 转换并统一标签有效的数据表示。
            date_ids[cursor:next_cursor] = daily["date_id"].to_numpy(np.int32)  # 转换并统一交易日ids的数据表示。
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
            index_frames.append(daily_index)  # 把当前结果追加到汇总列表。
            cursor = next_cursor
            if date_number % 50 == 0 or date_number == len(feature_dates):  # 检查当前样本数量是否满足计算要求。
                print(  # 输出当前任务的进度或诊断信息。
                    f"特征 {date_number}/{len(feature_dates)} 日，累计 {cursor:,} 条"
                )
        if cursor != sample_count:  # 按当前模式、标识或状态选择处理分支。
            raise AssertionError("feature store 写入行数不一致")  # 关键数据契约被破坏时立即终止。
        for array in (features, targets, label_valid, date_ids):  # 遍历特征标签标签有效交易日ids，逐项处理数组。
            array.flush()  # 将 memmap 修改刷入磁盘。
        del features, targets, label_valid, date_ids  # 释放不再使用的中间引用。
        for final_path, temporary_path in temporary_paths.items():  # 遍历临时文件pathsitems，逐项处理final路径临时文件路径。
            temporary_path.replace(final_path)  # 用临时文件原子替换正式结果。
        sample_index = pd.concat(index_frames, ignore_index=True)  # 对齐并合并样本索引。
        atomic_parquet(sample_index, sample_index_path)  # 原子落盘，避免半成品。
        atomic_json(  # 原子落盘，避免半成品。
            {
                "signature": store_signature,  # 记录签名。
                "payload": payload,  # 记录载荷。
                "sample_count": sample_count,  # 记录样本数量。
                "label_valid_count": int(sample_index["label_valid"].sum()),  # 记录标签有效数量。
                "first_sample_date": str(feature_dates[0]),  # 记录首个样本交易日。
                "last_sample_date": str(feature_dates[-1]),  # 记录最后一个样本交易日。
            },
            dataset_manifest_path,  # 汇入dataset清单路径。
        )
    else:  # 处理上述条件均不成立的情况。
        print("feature store 签名一致，直接复用。")  # 输出当前任务的进度或诊断信息。
        sample_index = pd.read_parquet(sample_index_path)  # 读取样本索引。

    sample_index["trade_date"] = sample_index["trade_date"].astype(str)  # 转换样本索引的数据类型或存储位置。
    sample_index["exit_trade_date"] = sample_index["exit_trade_date"].astype(str)  # 转换样本索引的数据类型或存储位置。
    if not np.array_equal(  # 检查必需数据、配置或校验结果是否缺失。
        sample_index["sample_index"].to_numpy(np.int64),
        np.arange(len(sample_index), dtype=np.int64),
    ):
        raise ValueError("sample_index 不是连续全局行号")  # 输入数据不满足业务或数值约束时立即终止。
    targets_check = np.load(targets_path, mmap_mode="r")  # 读取标签检查。
    label_valid_check = np.load(label_valid_path, mmap_mode="r")  # 读取标签有效检查。
    if not np.array_equal(  # 检查必需数据、配置或校验结果是否缺失。
        np.isfinite(targets_check), label_valid_check.astype(bool)
    ):
        raise AssertionError("label_valid 与 targets 的有限性不一致")  # 关键数据契约被破坏时立即终止。
    del targets_check, label_valid_check  # 释放不再使用的中间引用。

  # 训练/验证只含有效标签；预测包含该年度全部 T 日有效 Universe 特征。
    labeled = sample_index.loc[sample_index["label_valid"]].copy()  # 转换labeled的数据类型或存储位置。
    date_max_exit = labeled.groupby("trade_date", sort=True)["exit_trade_date"].max()  # 按业务键分组得到交易日maxexit。
    first_year = int(sample_index["trade_date"].min()[:4])
    last_year = int(sample_index["trade_date"].max()[:4])
    forecast_years = list(range(first_year + 1, last_year + 2))  # 初始化forecast年份容器。
    rolling_payload = {  # 初始化rolling载荷容器。
        "schema_version": 3,  # 记录schema版本。
        "feature_store_signature": store_signature,  # 记录特征存储签名，用于来源一致性校验。
        "forecast_years": forecast_years,  # 记录forecast年份。
        "train_date_ratio": 0.8,  # 记录训练交易日比例。
        "train_and_validation": "label_valid only",  # 把 train_and_validation 映射为"label_valid only"。
        "prediction": "all feature_valid and in_universe_t rows in forecast year",  # 把 prediction 映射为"all feature_valid and in_universe_t rows in forecast year"。
        "purge": "max T+11 must precede next partition boundary",  # 把 purge 映射为"max T+11 must precede next partition boundary"。
    }
    rolling_signature = stable_hash(rolling_payload)  # 生成数据复用签名。
    rolling_existing = read_json(rolling_manifest_path)  # 读取rollingexisting。
    split_paths = {  # 初始化切分paths容器。
        year: split_dir / f"forecast_{year}.npz" for year in forecast_years  # 接收年份。
    }
    split_reusable = (
        not read_bool_environment("VIT_FEATURE_STORE_OVERWRITE")
        and rolling_existing is not None
        and rolling_existing.get("signature") == rolling_signature
        and rolling_summary_path.exists()
        and all(path.exists() for path in split_paths.values())
    )
    if not split_reusable:  # 检查必需数据、配置或校验结果是否缺失。
        rolling_records = []  # 初始化rollingrecords容器。
        for forecast_year in forecast_years:  # 遍历forecast年份，逐项处理forecast年份。
            forecast_start = f"{forecast_year}0101"
            eligible_dates = date_max_exit.loc[
                (date_max_exit.index < forecast_start)
                & date_max_exit.lt(forecast_start)
            ].index.to_numpy(dtype=str)
            if len(eligible_dates) < 2:  # 检查当前样本数量是否满足计算要求。
                raise ValueError(f"{forecast_year} 历史标签日期不足")  # 输入数据不满足业务或数值约束时立即终止。
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
            train_mask = sample_index["label_valid"] & sample_index[  # 构造训练掩码，限定后续参与计算的样本。
                "trade_date"
            ].isin(train_dates)
            validation_mask = sample_index["label_valid"] & sample_index[  # 构造验证掩码，限定后续参与计算的样本。
                "trade_date"
            ].isin(validation_dates)
            prediction_mask = sample_index["trade_date"].str.startswith(  # 构造预测掩码，限定后续参与计算的样本。
                str(forecast_year)
            )
            train_indices = sample_index.loc[train_mask, "sample_index"].to_numpy(  # 转换并统一训练索引的数据表示。
                np.int64  # 汇入NumPyint64。
            )
            validation_indices = sample_index.loc[
                validation_mask, "sample_index"
            ].to_numpy(np.int64)
            prediction_indices = sample_index.loc[
                prediction_mask, "sample_index"
            ].to_numpy(np.int64)
            if not sample_index.iloc[train_indices]["label_valid"].all():  # 检查当前掩码或数据是否满足有效性要求。
                raise AssertionError("训练索引混入无标签样本")  # 关键数据契约被破坏时立即终止。
            if not sample_index.iloc[validation_indices]["label_valid"].all():  # 检查当前掩码或数据是否满足有效性要求。
                raise AssertionError("验证索引混入无标签样本")  # 关键数据契约被破坏时立即终止。
            if np.intersect1d(train_indices, validation_indices).size:  # 检查当前样本数量是否满足计算要求。
                raise AssertionError("训练和验证索引重叠")  # 关键数据契约被破坏时立即终止。
            max_train_exit = sample_index.loc[train_mask, "exit_trade_date"].max()
            max_validation_exit = sample_index.loc[
                validation_mask, "exit_trade_date"
            ].max()
            if max_train_exit >= validation_start:  # 检查当前数值是否越过业务或数值阈值。
                raise AssertionError("训练标签越过验证边界")  # 关键数据契约被破坏时立即终止。
            if max_validation_exit >= forecast_start:  # 检查当前数值是否越过业务或数值阈值。
                raise AssertionError("验证标签越过预测边界")  # 关键数据契约被破坏时立即终止。
            atomic_npz(
                split_paths[forecast_year],  # 汇入切分paths。
                train_indices=train_indices,  # 配置训练索引。
                validation_indices=validation_indices,  # 配置验证索引。
                prediction_indices=prediction_indices,  # 配置预测索引。
            )
            prediction_dates = np.sort(
                sample_index.loc[prediction_mask, "trade_date"].unique()
            )
            rolling_records.append(  # 把当前结果追加到汇总列表。
                {
                    "forecast_year": forecast_year,  # 记录forecast年份。
                    "history_start_date": str(eligible_dates[0]),  # 记录历史记录起点交易日。
                    "history_end_date": str(eligible_dates[-1]),  # 记录历史记录end交易日。
                    "history_date_count": len(eligible_dates),  # 记录历史记录交易日数量。
                    "raw_train_date_count": len(raw_train_dates),  # 记录原始训练交易日数量。
                    "purged_train_date_count": len(raw_train_dates) - len(train_dates),  # 记录purged训练交易日数量。
                    "train_start_date": str(train_dates[0]),  # 记录训练起点交易日。
                    "train_end_date": str(train_dates[-1]),  # 记录训练end交易日。
                    "train_date_count": len(train_dates),  # 记录训练交易日数量。
                    "train_sample_count": len(train_indices),  # 记录训练样本数量。
                    "max_train_exit_date": str(max_train_exit),  # 记录max训练exit交易日。
                    "validation_start_date": validation_start,  # 记录验证起点交易日。
                    "validation_end_date": str(validation_dates[-1]),  # 记录验证end交易日。
                    "validation_date_count": len(validation_dates),  # 记录验证交易日数量。
                    "validation_sample_count": len(validation_indices),  # 记录验证样本数量。
                    "max_validation_exit_date": str(max_validation_exit),  # 记录max验证exit交易日。
                    "forecast_start_date": forecast_start,  # 记录forecast起点交易日。
                    "prediction_start_date": (  # 记录预测起点交易日。
                        str(prediction_dates[0]) if len(prediction_dates) else None
                    ),
                    "prediction_end_date": (  # 记录预测end交易日。
                        str(prediction_dates[-1]) if len(prediction_dates) else None
                    ),
                    "prediction_date_count": len(prediction_dates),  # 记录预测交易日数量。
                    "prediction_sample_count": len(prediction_indices),  # 记录预测样本数量。
                }
            )
        rolling_summary = pd.DataFrame(rolling_records)
        atomic_parquet(rolling_summary, rolling_summary_path)  # 原子落盘，避免半成品。
        atomic_json(  # 原子落盘，避免半成品。
            {"signature": rolling_signature, "payload": rolling_payload},
            rolling_manifest_path,  # 汇入rolling清单路径。
        )
    else:  # 处理上述条件均不成立的情况。
        print("年度滚动切分签名一致，直接复用。")  # 输出当前任务的进度或诊断信息。
        rolling_summary = pd.read_parquet(rolling_summary_path)  # 读取rolling汇总。

    display(
        rolling_summary[  # 汇入rolling汇总。
            [
                "forecast_year", "train_sample_count", "validation_sample_count",
                "prediction_sample_count", "max_train_exit_date",
                "validation_start_date", "max_validation_exit_date",
                "forecast_start_date",
            ]
        ]
    )
    return {  # 返回保存 T 日有效 Universe 特征。
        "feature_store_directory": str(store_dir),  # 记录特征存储目录。
        "dataset_manifest_path": str(dataset_manifest_path),  # 记录dataset清单路径。
        "rolling_manifest_path": str(rolling_manifest_path),  # 记录rolling清单路径。
        "rolling_summary_path": str(rolling_summary_path),  # 记录rolling汇总路径。
        "signature": store_signature,  # 记录签名。
    }


def main() -> None:  # 构建 feature store 并打印当前产物清单。
    """构建 feature store 并打印当前产物清单。"""
    print(json.dumps(build_feature_store(), ensure_ascii=False, indent=2))  # 输出当前任务的进度或诊断信息。


if __name__ == "__main__":  # 按当前模式、标识或状态选择处理分支。
    main()
