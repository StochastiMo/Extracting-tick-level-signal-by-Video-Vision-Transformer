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


class UniverseBuildResult(TypedDict):
    """Universe 构建函数的稳定返回结构。"""

    universe_directory: str
    manifest_path: str
    quality_path: str
    signature: str


# region TR-DATA-UNIVERSE-DISCOVERY
def list_image_paths(max_dates_variable: str | None = None) -> dict[str, Path]:
    """发现日度图像文件并返回 ``交易日 -> 文件路径`` 的有序映射。"""

    paths: dict[str, Path] = {}
    for path in LEVEL2_DIR.glob("stock_pixels_cn.*.npz"):
        match = IMAGE_PATTERN.match(path.name)
        if match is None:
            continue
        trade_date = match.group(1)
        if trade_date in paths:
            raise ValueError(f"同一日期出现多个图片：{trade_date}")
        paths[trade_date] = path
    paths = dict(sorted(paths.items()))
    if max_dates_variable and os.getenv(max_dates_variable):
        paths = dict(list(paths.items())[: int(os.environ[max_dates_variable])])
    if not paths:
        raise FileNotFoundError(f"{LEVEL2_DIR} 下没有日度 NPZ")
    return paths


def load_calendars() -> dict[str, np.ndarray]:
    """读取上交所和深交所交易日历，返回八位日期数组。"""

    frame = pd.read_parquet(
        CALENDAR_PATH, columns=["trade_days", "s_info_exchmarket"]
    )
    frame["trade_days"] = normalize_dates(frame["trade_days"])  # 仅使用当日截面统计量
    calendars: dict[str, np.ndarray] = {}
    for exchange, market in {"SH": "SSE", "SZ": "SZSE"}.items():
        dates = (
            frame.loc[
                frame["s_info_exchmarket"].astype(str).eq(market),
                "trade_days",
            ]
            .drop_duplicates()
            .sort_values()
            .to_numpy(dtype=str)
        )
        if not len(dates):
            raise ValueError(f"交易日历缺少 {market}")
        calendars[exchange] = dates
    return calendars


def horizon_dates(
    trade_date: str,
    exchange: str,
    calendars: dict[str, np.ndarray],
) -> tuple[str, str]:
    """返回股票所属交易所的 T+1 与 T+11 日期。"""

    dates = calendars[exchange]
    position = int(np.searchsorted(dates, trade_date))
    if position >= len(dates) or dates[position] != trade_date:
        raise ValueError(f"{trade_date} 不在 {exchange} 交易日历")
    if position + 11 >= len(dates):
        raise ValueError(f"{trade_date} 之后不足 11 个交易日")
    return str(dates[position + 1]), str(dates[position + 11])


def instrument_to_wind(instrument_id: str) -> str:
    """把 ``SH.600000`` 转换为 Wind 使用的 ``600000.SH``。"""

    exchange, security_id = str(instrument_id).split(".", maxsplit=1)
    if exchange not in {"SH", "SZ"}:
        raise ValueError(f"不支持的股票代码：{instrument_id}")
    return f"{security_id}.{exchange}"


def read_image_metadata(
    trade_date: str, path: Path
) -> tuple[np.ndarray, np.ndarray]:
    """校验图像 NPZ 契约，返回股票代码和独立特征有效标记。"""

    with np.load(path, allow_pickle=False) as archive:
        missing = REQUIRED_IMAGE_KEYS.difference(archive.files)  # 检查输入缺失字段
        if missing:
            raise KeyError(f"{path.name} 缺少字段：{sorted(missing)}")
        instruments = archive["instrument_ids"].astype(str)
        valid_mask = archive["valid_mask"].astype(bool)
        if str(archive["trade_date"].item()) != trade_date:
            raise ValueError(f"{path.name} 文件名日期与内部日期不一致")
        if archive["pixel_matrices"].shape != (
            len(instruments), 17, 8, 8
        ):
            raise ValueError(f"{path.name} pixel_matrices 形状异常")
        if valid_mask.shape != (len(instruments),):
            raise ValueError(f"{path.name} valid_mask 未与股票行对齐")
        if len(np.unique(instruments)) != len(instruments):
            raise ValueError(f"{path.name} instrument_ids 重复")
    return instruments, valid_mask
# endregion


# region TR-DATA-UNIVERSE-STATUS
def read_market_status(
    required_dates: set[str], required_codes: set[str]
) -> pd.DataFrame:
    """读取所需停牌和涨跌停状态，并验证 ``股票×日期`` 唯一。"""

    filters = [
        ("trade_dt", ">=", min(required_dates)),
        ("trade_dt", "<=", max(required_dates)),
    ]
    eod = pd.read_parquet(
        EOD_PATH,
        columns=["s_info_windcode", "trade_dt", "s_dq_tradestatus"],
        filters=filters,
    )
    derivative = pd.read_parquet(
        DERIVATIVE_PATH,
        columns=["s_info_windcode", "trade_dt", "up_down_limit_status"],
        filters=filters,
    )
    filtered_frames: list[pd.DataFrame] = []
    for frame in (eod, derivative):
        frame["trade_dt"] = normalize_dates(frame["trade_dt"])  # 仅使用当日截面统计量
        frame["s_info_windcode"] = frame["s_info_windcode"].astype("string")
        frame = frame.loc[
            frame["trade_dt"].isin(required_dates)
            & frame["s_info_windcode"].isin(required_codes)
        ].copy()
        duplicated = frame.duplicated(  # 检查股票×日期业务键唯一性
            ["s_info_windcode", "trade_dt"], keep=False
        )
        if duplicated.any():
            raise ValueError(
                "Wind 状态业务键重复：\n"
                + frame.loc[duplicated].head().to_string(index=False)
            )
        filtered_frames.append(frame)
    return filtered_frames[0].merge(
        filtered_frames[1],
        on=["s_info_windcode", "trade_dt"],
        how="outer",
        validate="one_to_one",
    )


def classify_market_eligibility(candidates: pd.DataFrame) -> pd.DataFrame:
    """根据 T/T+1 状态生成两个市场有效性标记。 输入必须已经包含 T 日和 T+1 的交易、涨跌停状态。函数不删除行， 只增加布尔列；缺失状态通过 ``known_*`` 逻辑自然归为 ``False``。"""

    output = candidates.copy()
    known_t = output["tradestatus_t"].notna() & output[
        "limit_status_t"
    ].notna()
    known_t1 = output["tradestatus_t1"].notna() & output[
        "limit_status_t1"
    ].notna()
    output["in_universe_t"] = (
        known_t
        & ~output["tradestatus_t"].astype("string").eq("停牌")
        & ~output["limit_status_t"].isin([-1, 1])
    )
    output["can_enter_t1"] = (
        known_t1
        & ~output["tradestatus_t1"].astype("string").eq("停牌")
        & ~output["limit_status_t1"].isin([-1, 1])
    )
    return output
# endregion


# region TR-DATA-UNIVERSE-BUILD
def build_universe() -> UniverseBuildResult:
    """构建全市场有效 Universe、逐日质量表和 lineage manifest。"""

    root = get_data_root()
    universe_dir = root / "universe"
    quality_dir = root / "quality"
    universe_dir.mkdir(parents=True, exist_ok=True)
    quality_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = universe_dir / "universe_manifest.json"
    quality_path = quality_dir / "universe_quality_by_date.parquet"
    image_paths = list_image_paths("VIT_MAX_IMAGE_DATES")
    calendars = load_calendars()
    years = sorted({date[:4] for date in image_paths})
    output_paths = {
        year: universe_dir / f"universe.{year}.parquet" for year in years
    }

    # 所有上游文件共同进入签名，防止悄悄复用过时 Universe。
    image_signatures = [file_signature(path) for path in image_paths.values()]
    payload: dict[str, object] = {
        # 保持既有 schema/signature 口径，纯代码迁移不触发全量重建。
        "schema_version": 1,
        "image_files_hash": stable_hash(image_signatures),
        "image_file_count": len(image_signatures),
        "first_image_date": min(image_paths),
        "last_image_date": max(image_paths),
        "eod_source": file_signature(EOD_PATH),
        "derivative_source": file_signature(DERIVATIVE_PATH),
        "calendar_source": file_signature(CALENDAR_PATH),
        "in_universe_t": (
            "T status known; s_dq_tradestatus != 停牌; "
            "up_down_limit_status not in {-1,1}"
        ),
        "can_enter_t1": "same rule on exchange-calendar T+1",
        "missing_status": "fail_closed",
        "st_and_listing_age": "not_used",
    }
    signature = stable_hash(payload)  # 生成数据复用签名
    existing = read_json(manifest_path)
    reusable = (
        not read_bool_environment("VIT_UNIVERSE_OVERWRITE")
        and existing is not None
        and existing.get("signature") == signature
        and quality_path.exists()
        and all(path.exists() for path in output_paths.values())
    )
    if reusable:
        print("Universe 签名一致，直接复用。")
        return {
            "universe_directory": str(universe_dir),
            "manifest_path": str(manifest_path),
            "quality_path": str(quality_path),
            "signature": signature,
        }

    quality_records: list[dict[str, object]] = []
    total_candidates = 0
    print("开始构建 Universe：不使用 ST 或上市日期。")
    for year in years:
        daily_frames: list[pd.DataFrame] = []
        year_dates = [date for date in image_paths if date.startswith(year)]
        for trade_date in year_dates:
            instruments, feature_valid = read_image_metadata(
                trade_date, image_paths[trade_date]
            )
            exchanges = np.char.partition(instruments, ".")[:, 0]
            entry_dates = np.full(len(instruments), "", dtype="<U8")
            exit_dates = np.full(len(instruments), "", dtype="<U8")
            for exchange in ("SH", "SZ"):
                mask = exchanges == exchange
                entry_date, exit_date = horizon_dates(  # 按交易日历定位收益起止日
                    trade_date, exchange, calendars
                )
                entry_dates[mask] = entry_date
                exit_dates[mask] = exit_date
            if (entry_dates == "").any() or (exit_dates == "").any():
                raise ValueError(f"{trade_date} 含不支持的交易所股票")
            daily_frames.append(
                pd.DataFrame(
                    {
                        "wind_code": [
                            instrument_to_wind(value) for value in instruments
                        ],
                        "trade_date": trade_date,
                        "entry_trade_date": entry_dates,
                        "exit_trade_date": exit_dates,
                        "instrument_id": instruments,
                        "stock_index": np.arange(
                            len(instruments), dtype=np.int32
                        ),
                        "feature_valid": feature_valid,
                    }
                )
            )

        candidates = pd.concat(daily_frames, ignore_index=True)
        required_dates = set(candidates["trade_date"]) | set(
            candidates["entry_trade_date"]
        )
        status = read_market_status(
            required_dates, set(candidates["wind_code"])
        )
        t_status = status.rename(
            columns={
                "s_info_windcode": "wind_code",
                "trade_dt": "trade_date",
                "s_dq_tradestatus": "tradestatus_t",
                "up_down_limit_status": "limit_status_t",
            }
        )
        t1_status = status.rename(
            columns={
                "s_info_windcode": "wind_code",
                "trade_dt": "entry_trade_date",
                "s_dq_tradestatus": "tradestatus_t1",
                "up_down_limit_status": "limit_status_t1",
            }
        )
        candidates = (  # 按股票与交易日业务键对齐
            candidates.merge(
                t_status,
                on=["wind_code", "trade_date"],
                how="left",
                validate="many_to_one",
            )
            .merge(
                t1_status,
                on=["wind_code", "entry_trade_date"],
                how="left",
                validate="many_to_one",
            )
            .reset_index(drop=True)
        )
        candidates = classify_market_eligibility(candidates)

        for trade_date, daily in candidates.groupby(
            "trade_date", sort=True
        ):
            quality_records.append(
                _daily_quality_record(str(trade_date), daily)
            )

        output_columns = [
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
        output = candidates[output_columns].sort_values(
            ["trade_date", "stock_index"], kind="stable"
        )
        if output.duplicated(["trade_date", "instrument_id"]).any():
            raise ValueError(f"{year} Universe 业务键重复")
        atomic_parquet(output, output_paths[year])  # 原子落盘，避免半成品
        total_candidates += len(output)
        print(
            f"{year}：候选 {len(output):,}，T 日池 "
            f"{int(output['in_universe_t'].sum()):,}，可预测 "
            f"{int((output['feature_valid'] & output['in_universe_t']).sum()):,}"
        )
        del daily_frames, candidates, output, status, t_status, t1_status
        gc.collect()

    quality = pd.DataFrame(quality_records).sort_values("trade_date")
    if not (
        quality["candidate_rows"]
        == quality["in_universe_t_rows"] + quality["excluded_t_rows"]
    ).all():
        raise AssertionError("Universe 每日候选数无法守恒")
    atomic_parquet(quality, quality_path)  # 原子落盘，避免半成品
    atomic_json(  # 原子落盘，避免半成品
        {
            "signature": signature,
            "payload": payload,
            "years": years,
            "candidate_rows": total_candidates,
            "in_universe_t_rows": int(quality["in_universe_t_rows"].sum()),
            "prediction_eligible_rows": int(
                quality["prediction_eligible_rows"].sum()
            ),
            "label_candidate_rows": int(
                quality["label_candidate_rows"].sum()
            ),
        },
        manifest_path,
    )
    return {
        "universe_directory": str(universe_dir),
        "manifest_path": str(manifest_path),
        "quality_path": str(quality_path),
        "signature": signature,
    }
# endregion


def _daily_quality_record(
    trade_date: str, daily: pd.DataFrame
) -> dict[str, object]:
    """汇总一日 Universe 的逐步过滤数量守恒信息。"""

    return {
        "trade_date": trade_date,
        "candidate_rows": len(daily),
        "feature_valid_rows": int(daily["feature_valid"].sum()),
        "missing_status_t_rows": int(
            (
                daily["tradestatus_t"].isna()
                | daily["limit_status_t"].isna()
            ).sum()
        ),
        "suspended_t_rows": int(
            daily["tradestatus_t"].astype("string").eq("停牌").sum()
        ),
        "limit_up_t_rows": int(daily["limit_status_t"].eq(1).sum()),
        "limit_down_t_rows": int(daily["limit_status_t"].eq(-1).sum()),
        "in_universe_t_rows": int(daily["in_universe_t"].sum()),
        "excluded_t_rows": int((~daily["in_universe_t"]).sum()),
        "missing_status_t1_rows": int(
            (
                daily["tradestatus_t1"].isna()
                | daily["limit_status_t1"].isna()
            ).sum()
        ),
        "suspended_t1_rows": int(
            daily["tradestatus_t1"].astype("string").eq("停牌").sum()
        ),
        "limit_up_t1_rows": int(daily["limit_status_t1"].eq(1).sum()),
        "limit_down_t1_rows": int(daily["limit_status_t1"].eq(-1).sum()),
        "can_enter_t1_rows": int(daily["can_enter_t1"].sum()),
        "prediction_eligible_rows": int(
            (daily["feature_valid"] & daily["in_universe_t"]).sum()
        ),
        "label_candidate_rows": int(
            (daily["in_universe_t"] & daily["can_enter_t1"]).sum()
        ),
    }


def main() -> None:
    """构建 Universe 并打印当前产物清单。"""
    print(json.dumps(build_universe(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
