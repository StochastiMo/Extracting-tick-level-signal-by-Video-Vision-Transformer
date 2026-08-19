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
    """计算 T+1 到 T+11 复权收益，并做每日 1% Winsorize 和 Z-score。"""

    root = get_data_root()
    universe_dir = root / "universe"
    label_dir = root / "labels_t10"
    quality_dir = root / "quality"
    label_dir.mkdir(parents=True, exist_ok=True)
    quality_dir.mkdir(parents=True, exist_ok=True)
    universe_manifest = read_json(universe_dir / "universe_manifest.json")
    if universe_manifest is None:
        raise FileNotFoundError("请先运行 universe 阶段")
    universe_paths = sorted(universe_dir.glob("universe.*.parquet"))
    if not universe_paths:
        raise FileNotFoundError("Universe 年度文件不存在")
    years = [path.stem.split(".")[-1] for path in universe_paths]
    output_paths = {
        year: label_dir / f"labels_t10.{year}.parquet" for year in years
    }
    manifest_path = label_dir / "labels_manifest.json"
    quality_path = quality_dir / "label_quality_by_date.parquet"
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
    signature = stable_hash(payload)  # 生成数据复用签名
    existing = read_json(manifest_path)
    reusable = (
        not read_bool_environment("VIT_LABEL_OVERWRITE")
        and existing is not None
        and existing.get("signature") == signature
        and quality_path.exists()
        and all(path.exists() for path in output_paths.values())
    )
    if reusable:
        print("标签签名一致，直接复用。")
        return {
            "label_directory": str(label_dir),
            "manifest_path": str(manifest_path),
            "quality_path": str(quality_path),
            "signature": signature,
        }

    quality_records: list[dict[str, object]] = []
    total_labels = 0
    for universe_path, year in zip(universe_paths, years):
        universe = pd.read_parquet(
            universe_path,
            columns=[
                "trade_date", "instrument_id", "wind_code", "stock_index",
                "entry_trade_date", "exit_trade_date", "in_universe_t",
                "can_enter_t1",
            ],
        )
        candidates = universe.loc[
            universe["in_universe_t"] & universe["can_enter_t1"]
        ].copy()
        price_dates = set(candidates["entry_trade_date"]) | set(
            candidates["exit_trade_date"]
        )
        prices = pd.read_parquet(
            EOD_PATH,
            columns=["s_info_windcode", "trade_dt", "s_dq_adjclose"],
            filters=[
                ("trade_dt", ">=", min(price_dates)),
                ("trade_dt", "<=", max(price_dates)),
            ],
        )
        prices["trade_dt"] = normalize_dates(prices["trade_dt"])  # 仅使用当日截面统计量
        prices["s_info_windcode"] = prices["s_info_windcode"].astype("string")
        prices = prices.loc[
            prices["trade_dt"].isin(price_dates)
            & prices["s_info_windcode"].isin(set(candidates["wind_code"]))
        ].copy()
        if prices.duplicated(["s_info_windcode", "trade_dt"]).any():
            raise ValueError(f"{year} 价格业务键重复")

        entry = prices.rename(
            columns={
                "s_info_windcode": "wind_code", "trade_dt": "entry_trade_date",
                "s_dq_adjclose": "entry_adjclose",
            }
        )
        exits = prices.rename(
            columns={
                "s_info_windcode": "wind_code", "trade_dt": "exit_trade_date",
                "s_dq_adjclose": "exit_adjclose",
            }
        )
        candidates = (  # 按股票与交易日业务键对齐
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
        entry_values = candidates["entry_adjclose"].to_numpy(dtype=np.float64)
        exit_values = candidates["exit_adjclose"].to_numpy(dtype=np.float64)
        valid_price = (
            np.isfinite(entry_values) & np.isfinite(exit_values)
            & (entry_values > 0) & (exit_values > 0)
        )
        candidates["forward_return_10d"] = np.nan
        candidates.loc[valid_price, "forward_return_10d"] = (
            exit_values[valid_price] / entry_values[valid_price] - 1.0
        )
        candidates["forward_return_10d_winsorized"] = np.nan
        candidates["winsor_lower_1pct"] = np.nan
        candidates["winsor_upper_99pct"] = np.nan
        candidates["target_zscore"] = np.nan

        for _, group_index in candidates.groupby("trade_date", sort=True).groups.items():
            values = candidates.loc[group_index, "forward_return_10d"].dropna()  # 剔除缺失值后计算
            values = values.loc[np.isfinite(values.to_numpy(dtype=np.float64))]
            if len(values) < 2:
                continue
            lower = float(values.quantile(0.01, interpolation="linear"))
            upper = float(values.quantile(0.99, interpolation="linear"))
            winsorized = values.clip(lower=lower, upper=upper)  # 按日截面分位数缩尾
            standard_deviation = float(winsorized.std(ddof=0))
            if not np.isfinite(standard_deviation) or standard_deviation <= 0:
                continue
            valid_index = values.index
            candidates.loc[valid_index, "forward_return_10d_winsorized"] = winsorized
            candidates.loc[valid_index, "winsor_lower_1pct"] = lower
            candidates.loc[valid_index, "winsor_upper_99pct"] = upper
            candidates.loc[valid_index, "target_zscore"] = (
                winsorized - float(winsorized.mean())
            ) / standard_deviation

        label_valid = np.isfinite(  # 排除 NaN 与 Inf
            candidates["target_zscore"].to_numpy(dtype=np.float64)
        )
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
        float_columns = [
            "entry_adjclose", "exit_adjclose", "forward_return_10d",
            "forward_return_10d_winsorized", "winsor_lower_1pct",
            "winsor_upper_99pct", "target_zscore",
        ]
        labels[float_columns] = labels[float_columns].astype(np.float32)
        if labels.duplicated(["trade_date", "instrument_id"]).any():
            raise ValueError(f"{year} 标签业务键重复")

        universe_daily = universe.groupby("trade_date")["in_universe_t"].sum()
        for trade_date, daily in candidates.groupby("trade_date", sort=True):
            daily_labels = labels.loc[labels["trade_date"].eq(trade_date)]
            price_array = daily[["entry_adjclose", "exit_adjclose"]].to_numpy(
                dtype=np.float64
            )
            invalid_price = (
                ~np.isfinite(price_array).all(axis=1) | (price_array <= 0).any(axis=1)
            )
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
        atomic_parquet(labels, output_paths[year])  # 原子落盘，避免半成品
        total_labels += len(labels)
        print(f"{year}：候选 {len(candidates):,}，有效标签 {len(labels):,}")
        del universe, candidates, prices, entry, exits, labels
        gc.collect()

    quality = pd.DataFrame(quality_records).sort_values("trade_date")
    audit = quality.loc[quality["label_valid_rows"].ge(2)]
    if (
        audit["target_mean"].abs().max() > 2e-5
        or (audit["target_std_ddof0"] - 1.0).abs().max() > 2e-5
    ):
        raise AssertionError("每日标签标准化审计失败")
    atomic_parquet(quality, quality_path)  # 原子落盘，避免半成品
    atomic_json(  # 原子落盘，避免半成品
        {
            "signature": signature,
            "payload": payload,
            "years": years,
            "label_rows": total_labels,
        },
        manifest_path,
    )
    return {
        "label_directory": str(label_dir),
        "manifest_path": str(manifest_path),
        "quality_path": str(quality_path),
        "signature": signature,
        "label_rows": total_labels,
    }


def main() -> None:
    """构建 标签 并打印当前产物清单。"""
    print(json.dumps(build_t10_labels(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
