"""阶段 02：构建全市场有效 Universe 并清洗交易状态。

Universe 是后续特征、标签和滚动训练的共同业务键基座。本模块明确区分：

* ``in_universe_t``：股票在 T 日是否属于可研究市场池；
* ``can_enter_t1``：股票在所属交易所 T+1 是否具备可成交状态；
* ``feature_valid``：T 日图像特征本身是否有效。

状态缺失采用 fail-closed：未知状态不会被误判为有效样本。
"""
from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd

from vit_stage01_foundation import (
    CALENDAR_PATH, DERIVATIVE_PATH, EOD_PATH, IMAGE_PATTERN, LEVEL2_DIR,
    REQUIRED_IMAGE_KEYS, atomic_json, atomic_parquet, file_signature,
    get_data_root, normalize_dates, read_bool_environment, read_json, stable_hash,
)


class UniverseBuildResult(TypedDict):  # Universe 构建函数的稳定返回结构。
    """Universe 构建函数的稳定返回结构。"""

    universe_directory: str  # 接收股票池目录。
    manifest_path: str  # 接收清单路径。
    quality_path: str  # 接收质量路径。
    signature: str  # 接收签名。



def list_image_paths(max_dates_variable: str | None = None) -> dict[str, Path]:  # 发现日度图像文件并返回 ``交易日 -> 文件路径`` 的有序映射。
    """发现日度图像文件并返回 ``交易日 -> 文件路径`` 的有序映射。"""

    paths: dict[str, Path] = {}  # 初始化paths容器。
    for path in LEVEL2_DIR.glob("stock_pixels_cn.*.npz"):  # 遍历level2目录globstockpixelscnnpz，逐项处理路径。
        match = IMAGE_PATTERN.match(path.name)
        if match is None:  # 检查所需对象是否尚未初始化。
            continue  # 跳过当前项并处理下一项。
        trade_date = match.group(1)
        if trade_date in paths:  # 检查当前值是否属于允许集合。
            raise ValueError(f"同一日期出现多个图片：{trade_date}")  # 输入数据不满足业务或数值约束时立即终止。
        paths[trade_date] = path
    paths = dict(sorted(paths.items()))  # 计算paths以固定样本顺序。
    if max_dates_variable and os.getenv(max_dates_variable):  # 根据当前数据状态选择处理分支。
        paths = dict(list(paths.items())[: int(os.environ[max_dates_variable])])  # 初始化paths容器。
    if not paths:  # 检查必需数据、配置或校验结果是否缺失。
        raise FileNotFoundError(f"{LEVEL2_DIR} 下没有日度 NPZ")  # 目标文件不存在时立即终止。
    return paths  # 返回paths。


def load_calendars() -> dict[str, np.ndarray]:  # 读取上交所和深交所交易日历，返回八位日期数组。
    """读取上交所和深交所交易日历，返回八位日期数组。"""

    frame = pd.read_parquet(  # 读取帧。
        CALENDAR_PATH, columns=["trade_days", "s_info_exchmarket"]  # 定位交易日历路径。
    )
    frame["trade_days"] = normalize_dates(frame["trade_days"])  # 仅使用当日截面统计量。
    calendars: dict[str, np.ndarray] = {}  # 初始化calendars容器。
    for exchange, market in {"SH": "SSE", "SZ": "SZSE"}.items():  # 遍历shSSEszszseitems，逐项处理exchangemarket。
        dates = (
            frame.loc[  # 汇入帧loc。
                frame["s_info_exchmarket"].astype(str).eq(market),
                "trade_days",
            ]
            .drop_duplicates()
            .sort_values()
            .to_numpy(dtype=str)
        )
        if not len(dates):  # 检查当前样本数量是否满足计算要求。
            raise ValueError(f"交易日历缺少 {market}")  # 输入数据不满足业务或数值约束时立即终止。
        calendars[exchange] = dates
    return calendars  # 返回calendars。


def horizon_dates(  # 返回股票所属交易所的 T+1 与 T+11 日期。
    trade_date: str,  # 接收trade交易日。
    exchange: str,  # 接收exchange。
    calendars: dict[str, np.ndarray],
) -> tuple[str, str]:  # 声明当前函数的返回类型。
    """返回股票所属交易所的 T+1 与 T+11 日期。"""

    dates = calendars[exchange]
    position = int(np.searchsorted(dates, trade_date))  # 定位位置。
    if position >= len(dates) or dates[position] != trade_date:  # 检查当前样本数量是否满足计算要求。
        raise ValueError(f"{trade_date} 不在 {exchange} 交易日历")  # 输入数据不满足业务或数值约束时立即终止。
    if position + 11 >= len(dates):  # 检查当前样本数量是否满足计算要求。
        raise ValueError(f"{trade_date} 之后不足 11 个交易日")  # 输入数据不满足业务或数值约束时立即终止。
    return str(dates[position + 1]), str(dates[position + 11])  # 返回股票所属交易所的 T+1 与 T+11 日期。


def instrument_to_wind(instrument_id: str) -> str:  # 把 ``SH.600000`` 转换为 Wind 使用的 ``600000.SH``。
    """把 ``SH.600000`` 转换为 Wind 使用的 ``600000.SH``。"""

    exchange, security_id = str(instrument_id).split(".", maxsplit=1)
    if exchange not in {"SH", "SZ"}:  # 检查当前值是否属于允许集合。
        raise ValueError(f"不支持的股票代码：{instrument_id}")  # 输入数据不满足业务或数值约束时立即终止。
    return f"{security_id}.{exchange}"  # 返回完成上述格式转换后的结果。


def read_image_metadata(  # 校验图像 NPZ 契约，返回股票代码和独立特征有效标记。
    trade_date: str, path: Path
) -> tuple[np.ndarray, np.ndarray]:  # 声明当前函数的返回类型。
    """校验图像 NPZ 契约，返回股票代码和独立特征有效标记。"""

    with np.load(path, allow_pickle=False) as archive:  # 进入受控上下文并在退出时自动释放资源。
        missing = REQUIRED_IMAGE_KEYS.difference(archive.files)  # 检查输入缺失字段。
        if missing:  # 根据当前数据状态选择处理分支。
            raise KeyError(f"{path.name} 缺少字段：{sorted(missing)}")  # 缺少必需字段或配置键时立即终止。
        instruments = archive["instrument_ids"].astype(str)  # 转换股票的数据类型或存储位置。
        valid_mask = archive["valid_mask"].astype(bool)  # 构造有效掩码，限定后续参与计算的样本。
        if str(archive["trade_date"].item()) != trade_date:  # 按当前模式、标识或状态选择处理分支。
            raise ValueError(f"{path.name} 文件名日期与内部日期不一致")  # 输入数据不满足业务或数值约束时立即终止。
        if archive["pixel_matrices"].shape != (  # 检查张量或数组形状是否符合接口约定。
            len(instruments), 17, 8, 8
        ):
            raise ValueError(f"{path.name} pixel_matrices 形状异常")  # 输入数据不满足业务或数值约束时立即终止。
        if valid_mask.shape != (len(instruments),):  # 检查张量或数组形状是否符合接口约定。
            raise ValueError(f"{path.name} valid_mask 未与股票行对齐")  # 输入数据不满足业务或数值约束时立即终止。
        if len(np.unique(instruments)) != len(instruments):  # 检查当前样本数量是否满足计算要求。
            raise ValueError(f"{path.name} instrument_ids 重复")  # 输入数据不满足业务或数值约束时立即终止。
    return instruments, valid_mask  # 返回校验图像 NPZ 契约，返回股票代码和独立特征有效标记。




def read_market_status(  # 读取所需停牌和涨跌停状态，并验证 ``股票×日期`` 唯一。
    required_dates: set[str], required_codes: set[str]
) -> pd.DataFrame:  # 声明当前函数的返回类型。
    """读取所需停牌和涨跌停状态，并验证 ``股票×日期`` 唯一。"""

    filters = [  # 初始化filters容器。
        ("trade_dt", ">=", min(required_dates)),
        ("trade_dt", "<=", max(required_dates)),
    ]
    eod = pd.read_parquet(  # 读取日行情。
        EOD_PATH,  # 汇入日行情路径。
        columns=["s_info_windcode", "trade_dt", "s_dq_tradestatus"],  # 限定本次读取或输出使用的字段。
        filters=filters,  # 下推数据筛选条件，减少无关数据读取。
    )
    derivative = pd.read_parquet(  # 读取衍生指标。
        DERIVATIVE_PATH,  # 汇入衍生指标路径。
        columns=["s_info_windcode", "trade_dt", "up_down_limit_status"],  # 限定本次读取或输出使用的字段。
        filters=filters,  # 下推数据筛选条件，减少无关数据读取。
    )
    filtered_frames: list[pd.DataFrame] = []  # 接收筛选结果帧。
    for frame in (eod, derivative):  # 遍历日行情衍生指标，逐项处理帧。
        frame["trade_dt"] = normalize_dates(frame["trade_dt"])  # 仅使用当日截面统计量。
        frame["s_info_windcode"] = frame["s_info_windcode"].astype("string")  # 转换帧的数据类型或存储位置。
        frame = frame.loc[
            frame["trade_dt"].isin(required_dates)
            & frame["s_info_windcode"].isin(required_codes)
        ].copy()
        duplicated = frame.duplicated(  # 检查股票×日期业务键唯一性。
            ["s_info_windcode", "trade_dt"], keep=False
        )
        if duplicated.any():  # 检查当前掩码或数据是否满足有效性要求。
            raise ValueError(  # 输入数据不满足业务或数值约束时立即终止。
                "Wind 状态业务键重复：\n"
                + frame.loc[duplicated].head().to_string(index=False)
            )
        filtered_frames.append(frame)  # 把当前结果追加到汇总列表。
    return filtered_frames[0].merge(  # 返回读取并校验后的结果。
        filtered_frames[1],  # 汇入筛选结果帧。
        on=["s_info_windcode", "trade_dt"],  # 指定数据连接使用的业务键。
        how="outer",  # 指定数据连接方式。
        validate="one_to_one",  # 校验数据连接的基数关系。
    )


def classify_market_eligibility(candidates: pd.DataFrame) -> pd.DataFrame:  # 根据 T/T+1 状态生成两个市场有效性标记。
    """根据 T/T+1 状态生成两个市场有效性标记。 输入必须已经包含 T 日和 T+1 的交易、涨跌停状态。函数不删除行， 只增加布尔列；缺失状态通过 ``known_*`` 逻辑自然归为 ``False``。"""

    output = candidates.copy()  # 转换输出的数据类型或存储位置。
    known_t = output["tradestatus_t"].notna() & output[  # 构造已知行业t，限定后续参与计算的样本。
        "limit_status_t"
    ].notna()
    known_t1 = output["tradestatus_t1"].notna() & output[  # 构造已知行业t1，限定后续参与计算的样本。
        "limit_status_t1"
    ].notna()
    output["in_universe_t"] = (
        known_t  # 汇入已知行业t。
        & ~output["tradestatus_t"].astype("string").eq("停牌")
        & ~output["limit_status_t"].isin([-1, 1])
    )
    output["can_enter_t1"] = (
        known_t1  # 汇入已知行业t1。
        & ~output["tradestatus_t1"].astype("string").eq("停牌")
        & ~output["limit_status_t1"].isin([-1, 1])
    )
    return output  # 返回输出。




def build_universe() -> UniverseBuildResult:  # 构建全市场有效 Universe、逐日质量表和 lineage manifest。
    """构建全市场有效 Universe、逐日质量表和 lineage manifest。"""

    root = get_data_root()
    universe_dir = root / "universe"  # 构造股票池目录。
    quality_dir = root / "quality"  # 构造质量目录。
    universe_dir.mkdir(parents=True, exist_ok=True)  # 确保目标目录存在。
    quality_dir.mkdir(parents=True, exist_ok=True)  # 确保目标目录存在。
    manifest_path = universe_dir / "universe_manifest.json"  # 构造清单路径。
    quality_path = quality_dir / "universe_quality_by_date.parquet"  # 构造质量路径。
    image_paths = list_image_paths("VIT_MAX_IMAGE_DATES")
    calendars = load_calendars()
    years = sorted({date[:4] for date in image_paths})  # 计算年份以固定样本顺序。
    output_paths = {  # 初始化输出paths容器。
        year: universe_dir / f"universe.{year}.parquet" for year in years  # 接收年份。
    }

  # 所有上游文件共同进入签名，防止悄悄复用过时 Universe。
    image_signatures = [file_signature(path) for path in image_paths.values()]  # 初始化图像signatures容器。
    payload: dict[str, object] = {  # 初始化载荷容器。
  # 保持既有 schema/signature 口径，纯代码迁移不触发全量重建。
        "schema_version": 1,  # 记录schema版本。
        "image_files_hash": stable_hash(image_signatures),  # 记录图像files哈希。
        "image_file_count": len(image_signatures),  # 记录图像文件数量。
        "first_image_date": min(image_paths),  # 记录首个图像交易日。
        "last_image_date": max(image_paths),  # 记录最后一个图像交易日。
        "eod_source": file_signature(EOD_PATH),  # 记录日行情来源。
        "derivative_source": file_signature(DERIVATIVE_PATH),  # 记录衍生指标来源。
        "calendar_source": file_signature(CALENDAR_PATH),  # 记录交易日历来源。
        "in_universe_t": (  # 记录in股票池t。
            "T status known; s_dq_tradestatus != 停牌; "
            "up_down_limit_status not in {-1,1}"
        ),
        "can_enter_t1": "same rule on exchange-calendar T+1",  # 把 can_enter_t1 映射为"same rule on exchange-calendar T+1"。
        "missing_status": "fail_closed",  # 把 missing_status 映射为"fail_closed"。
        "st_and_listing_age": "not_used",  # 把 st_and_listing_age 映射为"not_used"。
    }
    signature = stable_hash(payload)  # 生成数据复用签名。
    existing = read_json(manifest_path)  # 读取existing。
    reusable = (
        not read_bool_environment("VIT_UNIVERSE_OVERWRITE")
        and existing is not None
        and existing.get("signature") == signature
        and quality_path.exists()
        and all(path.exists() for path in output_paths.values())
    )
    if reusable:  # 根据当前数据状态选择处理分支。
        print("Universe 签名一致，直接复用。")  # 输出当前任务的进度或诊断信息。
        return {  # 返回构建全市场有效 Universe、逐日质量表和 lineage manifest。
            "universe_directory": str(universe_dir),  # 记录股票池目录。
            "manifest_path": str(manifest_path),  # 记录清单路径。
            "quality_path": str(quality_path),  # 记录质量路径。
            "signature": signature,  # 记录签名。
        }

    quality_records: list[dict[str, object]] = []  # 初始化质量records容器。
    total_candidates = 0
    print("开始构建 Universe：不使用 ST 或上市日期。")  # 输出当前任务的进度或诊断信息。
    for year in years:  # 遍历年份，逐项处理年份。
        daily_frames: list[pd.DataFrame] = []  # 接收daily帧。
        year_dates = [date for date in image_paths if date.startswith(year)]  # 初始化年份交易日容器。
        for trade_date in year_dates:  # 遍历年份交易日，逐项处理trade交易日。
            instruments, feature_valid = read_image_metadata(
                trade_date, image_paths[trade_date]
            )
            exchanges = np.char.partition(instruments, ".")[:, 0]
            entry_dates = np.full(len(instruments), "", dtype="<U8")  # 计算entry交易日判定条件。
            exit_dates = np.full(len(instruments), "", dtype="<U8")  # 计算exit交易日判定条件。
            for exchange in ("SH", "SZ"):  # 遍历shsz，逐项处理exchange。
                mask = exchanges == exchange  # 配置掩码。
                entry_date, exit_date = horizon_dates(  # 按交易日历定位收益起止日。
                    trade_date, exchange, calendars  # 传入trade交易日、exchange、calendars。
                )
                entry_dates[mask] = entry_date  # 构造entry交易日，限定后续参与计算的样本。
                exit_dates[mask] = exit_date  # 构造exit交易日，限定后续参与计算的样本。
            if (entry_dates == "").any() or (exit_dates == "").any():  # 检查当前掩码或数据是否满足有效性要求。
                raise ValueError(f"{trade_date} 含不支持的交易所股票")  # 输入数据不满足业务或数值约束时立即终止。
            daily_frames.append(  # 把当前结果追加到汇总列表。
                pd.DataFrame(
                    {
                        "wind_code": [  # 记录wind代码。
                            instrument_to_wind(value) for value in instruments
                        ],
                        "trade_date": trade_date,  # 记录trade交易日。
                        "entry_trade_date": entry_dates,  # 记录entrytrade交易日。
                        "exit_trade_date": exit_dates,  # 记录exittrade交易日。
                        "instrument_id": instruments,  # 记录股票id。
                        "stock_index": np.arange(  # 记录stock索引。
                            len(instruments), dtype=np.int32
                        ),
                        "feature_valid": feature_valid,  # 记录特征有效。
                    }
                )
            )

        candidates = pd.concat(daily_frames, ignore_index=True)  # 对齐并合并candidates。
        required_dates = set(candidates["trade_date"]) | set(  # 初始化必需交易日容器。
            candidates["entry_trade_date"]  # 汇入candidates。
        )
        status = read_market_status(
            required_dates, set(candidates["wind_code"])
        )
        t_status = status.rename(
            columns={  # 初始化列容器。
                "s_info_windcode": "wind_code",  # 把 s_info_windcode 映射为"wind_code"。
                "trade_dt": "trade_date",  # 把 trade_dt 映射为"trade_date"。
                "s_dq_tradestatus": "tradestatus_t",  # 把 s_dq_tradestatus 映射为"tradestatus_t"。
                "up_down_limit_status": "limit_status_t",  # 把 up_down_limit_status 映射为"limit_status_t"。
            }
        )
        t1_status = status.rename(
            columns={  # 初始化列容器。
                "s_info_windcode": "wind_code",  # 把 s_info_windcode 映射为"wind_code"。
                "trade_dt": "entry_trade_date",  # 把 trade_dt 映射为"entry_trade_date"。
                "s_dq_tradestatus": "tradestatus_t1",  # 把 s_dq_tradestatus 映射为"tradestatus_t1"。
                "up_down_limit_status": "limit_status_t1",  # 把 up_down_limit_status 映射为"limit_status_t1"。
            }
        )
        candidates = (  # 按股票与交易日业务键对齐。
            candidates.merge(
                t_status,  # 汇入tstatus。
                on=["wind_code", "trade_date"],  # 指定数据连接使用的业务键。
                how="left",  # 指定数据连接方式。
                validate="many_to_one",  # 校验数据连接的基数关系。
            )
            .merge(
                t1_status,  # 汇入t1status。
                on=["wind_code", "entry_trade_date"],  # 指定数据连接使用的业务键。
                how="left",  # 指定数据连接方式。
                validate="many_to_one",  # 校验数据连接的基数关系。
            )
            .reset_index(drop=True)
        )
        candidates = classify_market_eligibility(candidates)

        for trade_date, daily in candidates.groupby(  # 遍历candidatesgroupby，逐项处理trade交易日daily。
            "trade_date", sort=True
        ):
            quality_records.append(  # 把当前结果追加到汇总列表。
                _daily_quality_record(str(trade_date), daily)
            )

        output_columns = [  # 初始化输出列容器。
            "wind_code",
            "trade_date",
            "entry_trade_date",
            "exit_trade_date",
            "instrument_id",
            "stock_index",
            "feature_valid",
            "in_universe_t",
            "can_enter_t1",
            "tradestatus_t",
            "limit_status_t",
            "tradestatus_t1",
            "limit_status_t1",
        ]
        output = candidates[output_columns].sort_values(  # 计算输出以固定样本顺序。
            ["trade_date", "stock_index"], kind="stable"
        )
        if output.duplicated(["trade_date", "instrument_id"]).any():  # 检查当前掩码或数据是否满足有效性要求。
            raise ValueError(f"{year} Universe 业务键重复")  # 输入数据不满足业务或数值约束时立即终止。
        atomic_parquet(output, output_paths[year])  # 原子落盘，避免半成品。
        total_candidates += len(output)
        print(  # 输出当前任务的进度或诊断信息。
            f"{year}：候选 {len(output):,}，T 日池 "
            f"{int(output['in_universe_t'].sum()):,}，可预测 "
            f"{int((output['feature_valid'] & output['in_universe_t']).sum()):,}"
        )
        del daily_frames, candidates, output, status, t_status, t1_status  # 释放不再使用的中间引用。
        gc.collect()

    quality = pd.DataFrame(quality_records).sort_values("trade_date")  # 计算质量以固定样本顺序。
    if not (  # 检查必需数据、配置或校验结果是否缺失。
        quality["candidate_rows"]  # 汇入质量。
        == quality["in_universe_t_rows"] + quality["excluded_t_rows"]
    ).all():
        raise AssertionError("Universe 每日候选数无法守恒")  # 关键数据契约被破坏时立即终止。
    atomic_parquet(quality, quality_path)  # 原子落盘，避免半成品。
    atomic_json(  # 原子落盘，避免半成品。
        {
            "signature": signature,  # 记录签名。
            "payload": payload,  # 记录载荷。
            "years": years,  # 记录年份。
            "candidate_rows": total_candidates,  # 记录candidate行数。
            "in_universe_t_rows": int(quality["in_universe_t_rows"].sum()),  # 记录in股票池t行数。
            "prediction_eligible_rows": int(  # 记录预测eligible行数。
                quality["prediction_eligible_rows"].sum()
            ),
            "label_candidate_rows": int(  # 记录标签candidate行数。
                quality["label_candidate_rows"].sum()
            ),
        },
        manifest_path,  # 汇入清单路径。
    )
    return {  # 返回构建全市场有效 Universe、逐日质量表和 lineage manifest。
        "universe_directory": str(universe_dir),  # 记录股票池目录。
        "manifest_path": str(manifest_path),  # 记录清单路径。
        "quality_path": str(quality_path),  # 记录质量路径。
        "signature": signature,  # 记录签名。
    }



def _daily_quality_record(  # 汇总一日 Universe 的逐步过滤数量守恒信息。
    trade_date: str, daily: pd.DataFrame
) -> dict[str, object]:  # 声明当前函数的返回类型。
    """汇总一日 Universe 的逐步过滤数量守恒信息。"""

    return {  # 返回汇总一日 Universe 的逐步过滤数量守恒信息。
        "trade_date": trade_date,  # 记录trade交易日。
        "candidate_rows": len(daily),  # 记录candidate行数。
        "feature_valid_rows": int(daily["feature_valid"].sum()),  # 记录特征有效行数。
        "missing_status_t_rows": int(  # 记录缺失值状态t行数。
            (
                daily["tradestatus_t"].isna()
                | daily["limit_status_t"].isna()
            ).sum()
        ),
        "suspended_t_rows": int(  # 记录suspendedt行数。
            daily["tradestatus_t"].astype("string").eq("停牌").sum()
        ),
        "limit_up_t_rows": int(daily["limit_status_t"].eq(1).sum()),  # 记录limitupt行数。
        "limit_down_t_rows": int(daily["limit_status_t"].eq(-1).sum()),  # 记录limitdownt行数。
        "in_universe_t_rows": int(daily["in_universe_t"].sum()),  # 记录in股票池t行数。
        "excluded_t_rows": int((~daily["in_universe_t"]).sum()),  # 记录excludedt行数。
        "missing_status_t1_rows": int(  # 记录缺失值状态t1行数。
            (
                daily["tradestatus_t1"].isna()
                | daily["limit_status_t1"].isna()
            ).sum()
        ),
        "suspended_t1_rows": int(  # 记录suspendedt1行数。
            daily["tradestatus_t1"].astype("string").eq("停牌").sum()
        ),
        "limit_up_t1_rows": int(daily["limit_status_t1"].eq(1).sum()),  # 记录limitupt1行数。
        "limit_down_t1_rows": int(daily["limit_status_t1"].eq(-1).sum()),  # 记录limitdownt1行数。
        "can_enter_t1_rows": int(daily["can_enter_t1"].sum()),  # 记录canentert1行数。
        "prediction_eligible_rows": int(  # 记录预测eligible行数。
            (daily["feature_valid"] & daily["in_universe_t"]).sum()
        ),
        "label_candidate_rows": int(  # 记录标签candidate行数。
            (daily["in_universe_t"] & daily["can_enter_t1"]).sum()
        ),
    }


def main() -> None:  # 构建 Universe 并打印当前产物清单。
    """构建 Universe 并打印当前产物清单。"""
    print(json.dumps(build_universe(), ensure_ascii=False, indent=2))  # 输出当前任务的进度或诊断信息。


if __name__ == "__main__":  # 按当前模式、标识或状态选择处理分支。
    main()
