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
def build_t10_labels() -> dict[str, object]:  # 计算 T+1 到 T+11 复权收益，并做每日 1% Winsorize 和 Z-score。
    """计算 T+1 到 T+11 复权收益，并做每日 1% Winsorize 和 Z-score。"""

    root = get_data_root()  # 解析可由环境变量覆盖的数据根目录。
    universe_dir = root / "universe"  # 构造股票池目录。
    label_dir = root / "labels_t10"  # 构造标签目录。
    quality_dir = root / "quality"  # 构造质量目录。
    label_dir.mkdir(parents=True, exist_ok=True)  # 确保目标目录存在。
    quality_dir.mkdir(parents=True, exist_ok=True)  # 确保目标目录存在。
    universe_manifest = read_json(universe_dir / "universe_manifest.json")  # 读取股票池清单。
    if universe_manifest is None:  # 检查所需对象是否尚未初始化。
        raise FileNotFoundError("请先运行 universe 阶段")  # 目标文件不存在时立即终止。
    universe_paths = sorted(universe_dir.glob("universe.*.parquet"))  # 计算股票池paths以固定样本顺序。
    if not universe_paths:  # 检查必需数据、配置或校验结果是否缺失。
        raise FileNotFoundError("Universe 年度文件不存在")  # 目标文件不存在时立即终止。
    years = [path.stem.split(".")[-1] for path in universe_paths]  # 初始化年份容器。
    output_paths = {  # 初始化输出paths容器。
        year: label_dir / f"labels_t10.{year}.parquet" for year in years  # 接收年份。
    }
    manifest_path = label_dir / "labels_manifest.json"  # 构造清单路径。
    quality_path = quality_dir / "label_quality_by_date.parquet"  # 构造质量路径。
    payload = {  # 初始化载荷容器。
        "schema_version": 3,  # 记录schema版本。
        "universe_signature": universe_manifest["signature"],  # 记录股票池签名，用于来源一致性校验。
        "eod_source": file_signature(EOD_PATH),  # 记录日行情来源。
        "formula": "s_dq_adjclose(T+11) / s_dq_adjclose(T+1) - 1",  # 把 formula 映射为"s_dq_adjclose(T+11) / s_dq_adjclose(T+1) - 1"。
        "candidate_rule": "in_universe_t and can_enter_t1",  # 把 candidate_rule 映射为"in_universe_t and can_enter_t1"。
        "winsorization": "daily 1%/99%, linear interpolation, clip",  # 把 winsorization 映射为"daily 1%/99%, linear interpolation, clip"。
        "zscore_ddof": 0,  # 记录zscoreddof。
        "feature_valid": "not_used",  # 把 feature_valid 映射为"not_used"。
    }
    signature = stable_hash(payload)  # 生成数据复用签名。
    existing = read_json(manifest_path)  # 读取existing。
    reusable = (  # 计算reusable。
        not read_bool_environment("VIT_LABEL_OVERWRITE")
        and existing is not None
        and existing.get("signature") == signature
        and quality_path.exists()
        and all(path.exists() for path in output_paths.values())
    )
    if reusable:  # 根据当前数据状态选择处理分支。
        print("标签签名一致，直接复用。")  # 输出当前任务的进度或诊断信息。
        return {  # 返回上述计算结果。
            "label_directory": str(label_dir),  # 记录标签目录。
            "manifest_path": str(manifest_path),  # 记录清单路径。
            "quality_path": str(quality_path),  # 记录质量路径。
            "signature": signature,  # 记录签名。
        }

    quality_records: list[dict[str, object]] = []  # 初始化质量records容器。
    total_labels = 0  # 计算total标签。
    for universe_path, year in zip(universe_paths, years):  # 遍历zip股票池paths年份，逐项处理股票池路径年份。
        universe = pd.read_parquet(  # 读取股票池。
            universe_path,  # 汇入股票池路径。
            columns=[  # 初始化列容器。
                "trade_date", "instrument_id", "wind_code", "stock_index",
                "entry_trade_date", "exit_trade_date", "in_universe_t",
                "can_enter_t1",
            ],
        )
        candidates = universe.loc[  # 计算candidates。
            universe["in_universe_t"] & universe["can_enter_t1"]
        ].copy()
        price_dates = set(candidates["entry_trade_date"]) | set(  # 初始化price交易日容器。
            candidates["exit_trade_date"]  # 汇入candidates。
        )
        prices = pd.read_parquet(  # 读取prices。
            EOD_PATH,  # 汇入日行情路径。
            columns=["s_info_windcode", "trade_dt", "s_dq_adjclose"],  # 限定本次读取或输出使用的字段。
            filters=[  # 初始化filters容器。
                ("trade_dt", ">=", min(price_dates)),
                ("trade_dt", "<=", max(price_dates)),
            ],
        )
        prices["trade_dt"] = normalize_dates(prices["trade_dt"])  # 仅使用当日截面统计量。
        prices["s_info_windcode"] = prices["s_info_windcode"].astype("string")  # 转换prices的数据类型或存储位置。
        prices = prices.loc[  # 计算复权价矩阵。
            prices["trade_dt"].isin(price_dates)
            & prices["s_info_windcode"].isin(set(candidates["wind_code"]))
        ].copy()
        if prices.duplicated(["s_info_windcode", "trade_dt"]).any():  # 检查当前掩码或数据是否满足有效性要求。
            raise ValueError(f"{year} 价格业务键重复")  # 输入数据不满足业务或数值约束时立即终止。

        entry = prices.rename(  # 提取并重命名entry。
            columns={  # 初始化列容器。
                "s_info_windcode": "wind_code", "trade_dt": "entry_trade_date",  # 把 s_info_windcode 映射为"wind_code", "trade_dt": "entry_trade_date"。
                "s_dq_adjclose": "entry_adjclose",  # 把 s_dq_adjclose 映射为"entry_adjclose"。
            }
        )
        exits = prices.rename(  # 提取并重命名exits。
            columns={  # 初始化列容器。
                "s_info_windcode": "wind_code", "trade_dt": "exit_trade_date",  # 把 s_info_windcode 映射为"wind_code", "trade_dt": "exit_trade_date"。
                "s_dq_adjclose": "exit_adjclose",  # 把 s_dq_adjclose 映射为"exit_adjclose"。
            }
        )
        candidates = (  # 按股票与交易日业务键对齐。
            candidates.merge(
                entry,  # 汇入entry。
                on=["wind_code", "entry_trade_date"],  # 指定数据连接使用的业务键。
                how="left",  # 指定数据连接方式。
                validate="many_to_one",  # 校验数据连接的基数关系。
            )
            .merge(
                exits,  # 汇入exits。
                on=["wind_code", "exit_trade_date"],  # 指定数据连接使用的业务键。
                how="left",  # 指定数据连接方式。
                validate="many_to_one",  # 校验数据连接的基数关系。
            )
            .reset_index(drop=True)
        )
        entry_values = candidates["entry_adjclose"].to_numpy(dtype=np.float64)  # 转换并统一entry值的数据表示。
        exit_values = candidates["exit_adjclose"].to_numpy(dtype=np.float64)  # 转换并统一exit值的数据表示。
        valid_price = (  # 构造有效price，限定后续参与计算的样本。
            np.isfinite(entry_values) & np.isfinite(exit_values)
            & (entry_values > 0) & (exit_values > 0)
        )
        candidates["forward_return_10d"] = np.nan  # 计算candidates。
        candidates.loc[valid_price, "forward_return_10d"] = (
            exit_values[valid_price] / entry_values[valid_price] - 1.0
        )
        candidates["forward_return_10d_winsorized"] = np.nan
        candidates["winsor_lower_1pct"] = np.nan
        candidates["winsor_upper_99pct"] = np.nan
        candidates["target_zscore"] = np.nan  # 计算candidates。

        for _, group_index in candidates.groupby("trade_date", sort=True).groups.items():  # 遍历candidatesgroupbytrade交易日sorttrue分组items，逐项处理group索引。
            values = candidates.loc[group_index, "forward_return_10d"].dropna()  # 剔除缺失值后计算。
            values = values.loc[np.isfinite(values.to_numpy(dtype=np.float64))]  # 转换并统一值的数据表示。
            if len(values) < 2:  # 检查当前样本数量是否满足计算要求。
                continue  # 跳过当前项并处理下一项。
            lower = float(values.quantile(0.01, interpolation="linear"))
            upper = float(values.quantile(0.99, interpolation="linear"))
            winsorized = values.clip(lower=lower, upper=upper)  # 按日截面分位数缩尾。
            standard_deviation = float(winsorized.std(ddof=0))  # 计算standarddeviation。
            if not np.isfinite(standard_deviation) or standard_deviation <= 0:  # 检查参与计算的数值是否全部有限。
                continue  # 跳过当前项并处理下一项。
            valid_index = values.index  # 构造有效索引，限定后续参与计算的样本。
            candidates.loc[valid_index, "forward_return_10d_winsorized"] = winsorized
            candidates.loc[valid_index, "winsor_lower_1pct"] = lower
            candidates.loc[valid_index, "winsor_upper_99pct"] = upper
            candidates.loc[valid_index, "target_zscore"] = (
                winsorized - float(winsorized.mean())
            ) / standard_deviation

        label_valid = np.isfinite(  # 排除 NaN 与 Inf。
            candidates["target_zscore"].to_numpy(dtype=np.float64)
        )
        labels = candidates.loc[
            label_valid,  # 汇入标签有效。
            [
                "trade_date", "instrument_id", "wind_code", "stock_index",
                "entry_trade_date", "exit_trade_date", "entry_adjclose",
                "exit_adjclose", "forward_return_10d",
                "forward_return_10d_winsorized", "winsor_lower_1pct",
                "winsor_upper_99pct", "target_zscore",
            ],
        ].copy()
        float_columns = [  # 初始化float列容器。
            "entry_adjclose", "exit_adjclose", "forward_return_10d",
            "forward_return_10d_winsorized", "winsor_lower_1pct",
            "winsor_upper_99pct", "target_zscore",
        ]
        labels[float_columns] = labels[float_columns].astype(np.float32)  # 转换标签的数据类型或存储位置。
        if labels.duplicated(["trade_date", "instrument_id"]).any():  # 检查当前掩码或数据是否满足有效性要求。
            raise ValueError(f"{year} 标签业务键重复")  # 输入数据不满足业务或数值约束时立即终止。

        universe_daily = universe.groupby("trade_date")["in_universe_t"].sum()  # 按业务键分组得到股票池daily。
        for trade_date, daily in candidates.groupby("trade_date", sort=True):  # 遍历candidatesgroupbytrade交易日sorttrue，逐项处理trade交易日daily。
            daily_labels = labels.loc[labels["trade_date"].eq(trade_date)]
            price_array = daily[["entry_adjclose", "exit_adjclose"]].to_numpy(  # 转换并统一price数组的数据表示。
                dtype=np.float64
            )
            invalid_price = (  # 构造invalidprice，限定后续参与计算的样本。
                ~np.isfinite(price_array).all(axis=1) | (price_array <= 0).any(axis=1)
            )
            quality_records.append(  # 把当前结果追加到汇总列表。
                {
                    "trade_date": trade_date,  # 记录trade交易日。
                    "universe_rows": int(universe_daily.loc[trade_date]),  # 记录股票池行数。
                    "label_candidate_rows": len(daily),  # 记录标签candidate行数。
                    "missing_or_nonpositive_price_rows": int(invalid_price.sum()),  # 记录缺失值ornonpositive价格行数。
                    "label_valid_rows": len(daily_labels),  # 记录标签有效行数。
                    "winsor_lower_1pct": (  # 记录winsorlower1pct。
                        float(daily_labels["winsor_lower_1pct"].iloc[0])
                        if len(daily_labels) else np.nan  # 检查当前样本数量是否满足计算要求。
                    ),
                    "winsor_upper_99pct": (  # 记录winsorupper99pct。
                        float(daily_labels["winsor_upper_99pct"].iloc[0])
                        if len(daily_labels) else np.nan  # 检查当前样本数量是否满足计算要求。
                    ),
                    "target_mean": (  # 记录标签均值。
                        float(daily_labels["target_zscore"].mean())
                        if len(daily_labels) else np.nan  # 检查当前样本数量是否满足计算要求。
                    ),
                    "target_std_ddof0": (  # 记录标签标准差ddof0。
                        float(daily_labels["target_zscore"].std(ddof=0))
                        if len(daily_labels) else np.nan  # 检查当前样本数量是否满足计算要求。
                    ),
                }
            )
        atomic_parquet(labels, output_paths[year])  # 原子落盘，避免半成品。
        total_labels += len(labels)
        print(f"{year}：候选 {len(candidates):,}，有效标签 {len(labels):,}")  # 输出当前任务的进度或诊断信息。
        del universe, candidates, prices, entry, exits, labels  # 释放不再使用的中间引用。
        gc.collect()

    quality = pd.DataFrame(quality_records).sort_values("trade_date")  # 计算质量以固定样本顺序。
    audit = quality.loc[quality["label_valid_rows"].ge(2)]
    if (  # 开始组合多项有效性或配置条件。
        audit["target_mean"].abs().max() > 2e-5
        or (audit["target_std_ddof0"] - 1.0).abs().max() > 2e-5
    ):
        raise AssertionError("每日标签标准化审计失败")  # 关键数据契约被破坏时立即终止。
    atomic_parquet(quality, quality_path)  # 原子落盘，避免半成品。
    atomic_json(  # 原子落盘，避免半成品。
        {
            "signature": signature,  # 记录签名。
            "payload": payload,  # 记录载荷。
            "years": years,  # 记录年份。
            "label_rows": total_labels,  # 记录标签行数。
        },
        manifest_path,  # 汇入清单路径。
    )
    return {  # 返回上述计算结果。
        "label_directory": str(label_dir),  # 记录标签目录。
        "manifest_path": str(manifest_path),  # 记录清单路径。
        "quality_path": str(quality_path),  # 记录质量路径。
        "signature": signature,  # 记录签名。
        "label_rows": total_labels,  # 记录标签行数。
    }


def main() -> None:  # 构建 标签 并打印当前产物清单。
    """构建 标签 并打印当前产物清单。"""
    print(json.dumps(build_t10_labels(), ensure_ascii=False, indent=2))  # 输出当前任务的进度或诊断信息。


if __name__ == "__main__":  # 按当前模式、标识或状态选择处理分支。
    main()
