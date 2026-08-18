"""阶段 03：构建 T+1 入场、T+11 离场的截面收益标签。"""
from __future__ import annotations


import gc
import json

import numpy as np
import pandas as pd

from vit_stage01_foundation import (
    EOD_PATH, atomic_json, atomic_parquet, file_signature, get_data_root,
    normalize_dates, read_bool_environment, read_json, stable_hash,
)


# ========================== 03｜构建标签 ==========================
def build_t10_labels() -> dict[str, object]:
    """计算 T+1 到 T+11 复权收益，并做每日 1% Winsorize 和 Z-score。

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
    # 计算 `root / 'quality'`，并将数值结果保存到 `quality_dir`。
    quality_dir = root / "quality"
    # 调用 `label_dir.mkdir` 创建输出目录，确保后续文件可以安全写入。
    label_dir.mkdir(parents=True, exist_ok=True)
    # 调用 `quality_dir.mkdir` 创建输出目录，确保后续文件可以安全写入。
    quality_dir.mkdir(parents=True, exist_ok=True)
    # 调用 `read_json` 读取源数据，并将结果保存到 `universe_manifest`。
    universe_manifest = read_json(universe_dir / "universe_manifest.json")
    # 检查 `universe_manifest is None`；根据条件结果选择对应处理分支。
    if universe_manifest is None:
        # 抛出 `FileNotFoundError('请先运行 universe 阶段')`，阻止不满足数据契约的流程继续执行。
        raise FileNotFoundError("请先运行 universe 阶段")
    # 调用 `sorted` 整理数据结构，并将结果保存到 `universe_paths`。
    universe_paths = sorted(universe_dir.glob("universe.*.parquet"))
    # 检查 `not universe_paths`；根据条件结果选择对应处理分支。
    if not universe_paths:
        # 抛出 `FileNotFoundError('Universe 年度文件不存在')`，阻止不满足数据契约的流程继续执行。
        raise FileNotFoundError("Universe 年度文件不存在")
    # 构造当前步骤所需的列表，并保存到 `years`。
    years = [path.stem.split(".")[-1] for path in universe_paths]
    # 构造键值映射，并保存到 `output_paths`，供后续按键查询。
    output_paths = {
        year: label_dir / f"labels_t10.{year}.parquet" for year in years
    }
    # 计算 `label_dir / 'labels_manifest.json'`，并将数值结果保存到 `manifest_path`。
    manifest_path = label_dir / "labels_manifest.json"
    # 计算 `quality_dir / 'label_quality_by_date.parquet'`，并将数值结果保存到 `quality_path`。
    quality_path = quality_dir / "label_quality_by_date.parquet"
    # 构造键值映射，并保存到 `payload`，供后续按键查询。
    payload = {
        "schema_version": 3,
        "universe_signature": universe_manifest["signature"],
        "eod_source": file_signature(EOD_PATH),
        "formula": "s_dq_adjclose(T+11) / s_dq_adjclose(T+1) - 1",
        "candidate_rule": "in_universe_t and can_enter_t1",
        "winsorization": "daily 1%/99%, linear interpolation, clip",
        "zscore_ddof": 0,
        "feature_valid": "not_used",
    }
    # 调用 `stable_hash` 对关键配置生成稳定签名以支持复现和一致性检查，并将结果保存到 `signature`。
    signature = stable_hash(payload)
    # 调用 `read_json` 读取源数据，并将结果保存到 `existing`。
    existing = read_json(manifest_path)
    # 根据 `not read_bool_environment('VIT_LABEL_OVERWRITE') and existing is not None and (existing.get('si…` 计算条件结果，并保存到 `reusable`。
    reusable = (
        not read_bool_environment("VIT_LABEL_OVERWRITE")
        and existing is not None
        and existing.get("signature") == signature
        and quality_path.exists()
        and all(path.exists() for path in output_paths.values())
    )
    # 检查 `reusable`；根据条件结果选择对应处理分支。
    if reusable:
        # 调用 `print` 记录当前流程状态或诊断信息。
        print("标签签名一致，直接复用。")
        # 将 `{'label_directory': str(label_dir), 'manifest_path': str(manifest_path), 'quality_path': str(qu…` 作为当前函数结果返回给调用方。
        return {
            "label_directory": str(label_dir),
            "manifest_path": str(manifest_path),
            "quality_path": str(quality_path),
            "signature": signature,
        }

    # 构造当前步骤所需的列表，并保存到 `quality_records`。
    quality_records: list[dict[str, object]] = []
    # 设置 `total_labels` 的固定配置值，供后续流程统一使用。
    total_labels = 0
    # 遍历 `zip(universe_paths, years)`，并把当前元素绑定到 `(universe_path, year)` 逐项处理。
    for universe_path, year in zip(universe_paths, years):
        # 调用 `pd.read_parquet` 读取源数据，并将结果保存到 `universe`。
        universe = pd.read_parquet(
            universe_path,
            columns=[
                "trade_date", "instrument_id", "wind_code", "stock_index",
                "entry_trade_date", "exit_trade_date", "in_universe_t",
                "can_enter_t1",
            ],
        )
        # 调用 `copy` 复制当前对象以避免后续原地修改污染源数据，并将结果保存到 `candidates`。
        candidates = universe.loc[
            universe["in_universe_t"] & universe["can_enter_t1"]
        ].copy()
        # 计算 `set(candidates['entry_trade_date']) | set(candidates['exit_trade_date'])`，并将数值结果保存到 `price_dates`。
        price_dates = set(candidates["entry_trade_date"]) | set(
            candidates["exit_trade_date"]
        )
        # 调用 `pd.read_parquet` 读取源数据，并将结果保存到 `prices`。
        prices = pd.read_parquet(
            EOD_PATH,
            columns=["s_info_windcode", "trade_dt", "s_dq_adjclose"],
            filters=[
                ("trade_dt", ">=", min(price_dates)),
                ("trade_dt", "<=", max(price_dates)),
            ],
        )
        # 调用 `normalize_dates` 按当前截面统计量完成缩尾或标准化处理，并将结果保存到 `prices['trade_dt']`。
        prices["trade_dt"] = normalize_dates(prices["trade_dt"])
        # 调用 `prices['s_info_windcode'].astype` 统一设备或数据类型，并将结果保存到 `prices['s_info_windcode']`。
        prices["s_info_windcode"] = prices["s_info_windcode"].astype("string")
        # 调用 `copy` 复制当前对象以避免后续原地修改污染源数据，并将结果保存到 `prices`。
        prices = prices.loc[
            prices["trade_dt"].isin(price_dates)
            & prices["s_info_windcode"].isin(set(candidates["wind_code"]))
        ].copy()
        # 检查 `prices.duplicated(['s_info_windcode', 'trade_dt']).any()`；根据条件结果选择对应处理分支。
        if prices.duplicated(["s_info_windcode", "trade_dt"]).any():
            # 抛出 `ValueError(f'{year} 价格业务键重复')`，阻止不满足数据契约的流程继续执行。
            raise ValueError(f"{year} 价格业务键重复")

        # 调用 `prices.rename` 把完整临时产物替换为正式输出，并将结果保存到 `entry`。
        entry = prices.rename(
            columns={
                "s_info_windcode": "wind_code", "trade_dt": "entry_trade_date",
                "s_dq_adjclose": "entry_adjclose",
            }
        )
        # 调用 `prices.rename` 把完整临时产物替换为正式输出，并将结果保存到 `exits`。
        exits = prices.rename(
            columns={
                "s_info_windcode": "wind_code", "trade_dt": "exit_trade_date",
                "s_dq_adjclose": "exit_adjclose",
            }
        )
        # 调用 `candidates.merge(entry, on=['wind_code', 'entry_trade_date'], h…` 按股票和交易日键对齐来自不同来源的数据，并将结果保存到 `candidates`。
        candidates = (
            candidates.merge(
                entry,
                on=["wind_code", "entry_trade_date"],
                how="left",
                validate="many_to_one",
            )
            .merge(
                exits,
                on=["wind_code", "exit_trade_date"],
                how="left",
                validate="many_to_one",
            )
            .reset_index(drop=True)
        )
        # 调用 `candidates['entry_adjclose'].to_numpy` 把表格列转换为指定精度的 NumPy 数组，并将结果保存到 `entry_values`。
        entry_values = candidates["entry_adjclose"].to_numpy(dtype=np.float64)
        # 调用 `candidates['exit_adjclose'].to_numpy` 把表格列转换为指定精度的 NumPy 数组，并将结果保存到 `exit_values`。
        exit_values = candidates["exit_adjclose"].to_numpy(dtype=np.float64)
        # 计算 `np.isfinite(entry_values) & np.isfinite(exit_values) & (entry_values > 0) & (exit_values > 0)`，并将数值结果保存到 `valid_price`。
        valid_price = (
            np.isfinite(entry_values) & np.isfinite(exit_values)
            & (entry_values > 0) & (exit_values > 0)
        )
        # 读取 `np.nan` 的当前值，并赋给 `candidates['forward_return_10d']`。
        candidates["forward_return_10d"] = np.nan
        # 计算 `exit_values[valid_price] / entry_values[valid_price] - 1.0`，并将数值结果保存到 `candidates.loc[valid_price, 'forward_return_10d']`。
        candidates.loc[valid_price, "forward_return_10d"] = (
            exit_values[valid_price] / entry_values[valid_price] - 1.0
        )
        # 读取 `np.nan` 的当前值，并赋给 `candidates['forward_return_10d_winsorized']`。
        candidates["forward_return_10d_winsorized"] = np.nan
        # 读取 `np.nan` 的当前值，并赋给 `candidates['winsor_lower_1pct']`。
        candidates["winsor_lower_1pct"] = np.nan
        # 读取 `np.nan` 的当前值，并赋给 `candidates['winsor_upper_99pct']`。
        candidates["winsor_upper_99pct"] = np.nan
        # 读取 `np.nan` 的当前值，并赋给 `candidates['target_zscore']`。
        candidates["target_zscore"] = np.nan

        # 遍历 `candidates.groupby('trade_date', sort=True).groups.items()`，并把当前元素绑定到 `(_, group_index)` 逐项处理。
        for _, group_index in candidates.groupby("trade_date", sort=True).groups.items():
            # 调用 `candidates.loc[group_index, 'forward_return_10d'].dropna` 剔除缺失观测后再计算统计量，并将结果保存到 `values`。
            values = candidates.loc[group_index, "forward_return_10d"].dropna()
            # 读取 `values.loc[np.isfinite(values.to_numpy(dtype=np.float64))]` 的当前值，并赋给 `values`。
            values = values.loc[np.isfinite(values.to_numpy(dtype=np.float64))]
            # 检查 `len(values) < 2`；根据条件结果选择对应处理分支。
            if len(values) < 2:
                # 跳过当前元素的剩余步骤，继续处理下一项。
                continue
            # 调用 `float` 统一设备或数据类型，并将结果保存到 `lower`。
            lower = float(values.quantile(0.01, interpolation="linear"))
            # 调用 `float` 统一设备或数据类型，并将结果保存到 `upper`。
            upper = float(values.quantile(0.99, interpolation="linear"))
            # 调用 `values.clip` 把极端值截断在截面分位数边界内，并将结果保存到 `winsorized`。
            winsorized = values.clip(lower=lower, upper=upper)
            # 调用 `float` 统一设备或数据类型，并将结果保存到 `standard_deviation`。
            standard_deviation = float(winsorized.std(ddof=0))
            # 检查 `not np.isfinite(standard_deviation) or standard_deviation <= 0`；根据条件结果选择对应处理分支。
            if not np.isfinite(standard_deviation) or standard_deviation <= 0:
                # 跳过当前元素的剩余步骤，继续处理下一项。
                continue
            # 读取 `values.index` 的当前值，并赋给 `valid_index`。
            valid_index = values.index
            # 读取 `winsorized` 的当前值，并赋给 `candidates.loc[valid_index, 'forward_return_10d_winsorized']`。
            candidates.loc[valid_index, "forward_return_10d_winsorized"] = winsorized
            # 读取 `lower` 的当前值，并赋给 `candidates.loc[valid_index, 'winsor_lower_1pct']`。
            candidates.loc[valid_index, "winsor_lower_1pct"] = lower
            # 读取 `upper` 的当前值，并赋给 `candidates.loc[valid_index, 'winsor_upper_99pct']`。
            candidates.loc[valid_index, "winsor_upper_99pct"] = upper
            # 计算 `(winsorized - float(winsorized.mean())) / standard_deviation`，并将数值结果保存到 `candidates.loc[valid_index, 'target_zscore']`。
            candidates.loc[valid_index, "target_zscore"] = (
                winsorized - float(winsorized.mean())
            ) / standard_deviation

        # 调用 `np.isfinite` 判断数值是否为有限值以排除 NaN 和无穷大，并将结果保存到 `label_valid`。
        label_valid = np.isfinite(
            candidates["target_zscore"].to_numpy(dtype=np.float64)
        )
        # 调用 `copy` 复制当前对象以避免后续原地修改污染源数据，并将结果保存到 `labels`。
        labels = candidates.loc[
            label_valid,
            [
                "trade_date", "instrument_id", "wind_code", "stock_index",
                "entry_trade_date", "exit_trade_date", "entry_adjclose",
                "exit_adjclose", "forward_return_10d",
                "forward_return_10d_winsorized", "winsor_lower_1pct",
                "winsor_upper_99pct", "target_zscore",
            ],
        ].copy()
        # 构造当前步骤所需的列表，并保存到 `float_columns`。
        float_columns = [
            "entry_adjclose", "exit_adjclose", "forward_return_10d",
            "forward_return_10d_winsorized", "winsor_lower_1pct",
            "winsor_upper_99pct", "target_zscore",
        ]
        # 调用 `labels[float_columns].astype` 统一设备或数据类型，并将结果保存到 `labels[float_columns]`。
        labels[float_columns] = labels[float_columns].astype(np.float32)
        # 检查 `labels.duplicated(['trade_date', 'instrument_id']).any()`；根据条件结果选择对应处理分支。
        if labels.duplicated(["trade_date", "instrument_id"]).any():
            # 抛出 `ValueError(f'{year} 标签业务键重复')`，阻止不满足数据契约的流程继续执行。
            raise ValueError(f"{year} 标签业务键重复")

        # 调用 `universe.groupby('trade_date')['in_universe_t'].sum` 汇总当前数据，并将统计量保存到 `universe_daily`。
        universe_daily = universe.groupby("trade_date")["in_universe_t"].sum()
        # 遍历 `candidates.groupby('trade_date', sort=True)`，并把当前元素绑定到 `(trade_date, daily)` 逐项处理。
        for trade_date, daily in candidates.groupby("trade_date", sort=True):
            # 读取 `labels.loc[labels['trade_date'].eq(trade_date)]` 的当前值，并赋给 `daily_labels`。
            daily_labels = labels.loc[labels["trade_date"].eq(trade_date)]
            # 调用 `daily[['entry_adjclose', 'exit_adjclose']].to_numpy` 把表格列转换为指定精度的 NumPy 数组，并将结果保存到 `price_array`。
            price_array = daily[["entry_adjclose", "exit_adjclose"]].to_numpy(
                dtype=np.float64
            )
            # 计算 `~np.isfinite(price_array).all(axis=1) | (price_array <= 0).any(axis=1)`，并将数值结果保存到 `invalid_price`。
            invalid_price = (
                ~np.isfinite(price_array).all(axis=1) | (price_array <= 0).any(axis=1)
            )
            # 把本轮结果追加到 `quality_records`，累积后续需要汇总的记录。
            quality_records.append(
                {
                    "trade_date": trade_date,
                    "universe_rows": int(universe_daily.loc[trade_date]),
                    "label_candidate_rows": len(daily),
                    "missing_or_nonpositive_price_rows": int(invalid_price.sum()),
                    "label_valid_rows": len(daily_labels),
                    "winsor_lower_1pct": (
                        float(daily_labels["winsor_lower_1pct"].iloc[0])
                        if len(daily_labels) else np.nan
                    ),
                    "winsor_upper_99pct": (
                        float(daily_labels["winsor_upper_99pct"].iloc[0])
                        if len(daily_labels) else np.nan
                    ),
                    "target_mean": (
                        float(daily_labels["target_zscore"].mean())
                        if len(daily_labels) else np.nan
                    ),
                    "target_std_ddof0": (
                        float(daily_labels["target_zscore"].std(ddof=0))
                        if len(daily_labels) else np.nan
                    ),
                }
            )
        # 调用 `atomic_parquet` 先写临时文件再原子替换正式产物，避免留下半成品。
        atomic_parquet(labels, output_paths[year])
        # 用 `len(labels)` 对 `total_labels` 做 `Add` 累积更新。
        total_labels += len(labels)
        # 调用 `print` 记录当前流程状态或诊断信息。
        print(f"{year}：候选 {len(candidates):,}，有效标签 {len(labels):,}")
        # 删除 `universe, candidates, prices, entry, exits, labels`，释放不再使用的引用。
        del universe, candidates, prices, entry, exits, labels
        # 调用 `gc.collect` 触发 Python 垃圾回收以释放失去引用的对象。
        gc.collect()

    # 调用 `pd.DataFrame(quality_records).sort_values` 根据分数计算截面排序或排序索引，并将结果保存到 `quality`。
    quality = pd.DataFrame(quality_records).sort_values("trade_date")
    # 读取 `quality.loc[quality['label_valid_rows'].ge(2)]` 的当前值，并赋给 `audit`。
    audit = quality.loc[quality["label_valid_rows"].ge(2)]
    # 检查 `audit['target_mean'].abs().max() > 2e-05 or (audit['target_std_ddof0'] - 1.0).abs().max() > 2e-…`；根据条件结果选择对应处理分支。
    if (
        audit["target_mean"].abs().max() > 2e-5
        or (audit["target_std_ddof0"] - 1.0).abs().max() > 2e-5
    ):
        # 抛出 `AssertionError('每日标签标准化审计失败')`，阻止不满足数据契约的流程继续执行。
        raise AssertionError("每日标签标准化审计失败")
    # 调用 `atomic_parquet` 先写临时文件再原子替换正式产物，避免留下半成品。
    atomic_parquet(quality, quality_path)
    # 调用 `atomic_json` 先写临时文件再原子替换正式产物，避免留下半成品。
    atomic_json(
        {
            "signature": signature,
            "payload": payload,
            "years": years,
            "label_rows": total_labels,
        },
        manifest_path,
    )
    # 将 `{'label_directory': str(label_dir), 'manifest_path': str(manifest_path), 'quality_path': str(qu…` 作为当前函数结果返回给调用方。
    return {
        "label_directory": str(label_dir),
        "manifest_path": str(manifest_path),
        "quality_path": str(quality_path),
        "signature": signature,
        "label_rows": total_labels,
    }


# 定义命令行入口，解析参数并按用户选择调度当前阶段的主流程。
def main() -> None:
    """构建 标签 并打印当前产物清单。

    输入：
        无显式输入参数。
    输出：
        None。
    """
    # 调用 `print` 记录当前流程状态或诊断信息。
    print(json.dumps(build_t10_labels(), ensure_ascii=False, indent=2))


# 检查 `__name__ == '__main__'`；根据条件结果选择对应处理分支。
if __name__ == "__main__":
    # 调用 `main` 启动当前模块的命令行主流程。
    main()
