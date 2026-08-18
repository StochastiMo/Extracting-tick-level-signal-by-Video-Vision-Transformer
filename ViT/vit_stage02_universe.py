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


# 定义 `UniverseBuildResult` 类，集中封装该阶段的数据结构与相关行为。
class UniverseBuildResult(TypedDict):
    """Universe 构建函数的稳定返回结构。"""

    # 声明 `universe_directory` 的数据类型，明确后续读写契约。
    universe_directory: str
    # 声明 `manifest_path` 的数据类型，明确后续读写契约。
    manifest_path: str
    # 声明 `quality_path` 的数据类型，明确后续读写契约。
    quality_path: str
    # 声明 `signature` 的数据类型，明确后续读写契约。
    signature: str


# region TR-DATA-UNIVERSE-DISCOVERY
# 定义 `list_image_paths` 函数，封装这一阶段可独立调用的处理步骤。
def list_image_paths(max_dates_variable: str | None = None) -> dict[str, Path]:
    """发现日度图像文件并返回 ``交易日 -> 文件路径`` 的有序映射。

    输入：
        max_dates_variable: str | None。
    输出：
        dict[str, Path]。
    """

    # 构造键值映射，并保存到 `paths`，供后续按键查询。
    paths: dict[str, Path] = {}
    # 遍历 `LEVEL2_DIR.glob('stock_pixels_cn.*.npz')`，并把当前元素绑定到 `path` 逐项处理。
    for path in LEVEL2_DIR.glob("stock_pixels_cn.*.npz"):
        # 调用 `IMAGE_PATTERN.match` 用预编译正则匹配完整图片文件名，并将结果保存到 `match`。
        match = IMAGE_PATTERN.match(path.name)
        # 检查 `match is None`；根据条件结果选择对应处理分支。
        if match is None:
            # 跳过当前元素的剩余步骤，继续处理下一项。
            continue
        # 调用 `match.group` 读取正则匹配结果中的指定捕获组，并将结果保存到 `trade_date`。
        trade_date = match.group(1)
        # 检查 `trade_date in paths`；根据条件结果选择对应处理分支。
        if trade_date in paths:
            # 抛出 `ValueError(f'同一日期出现多个图片：{trade_date}')`，阻止不满足数据契约的流程继续执行。
            raise ValueError(f"同一日期出现多个图片：{trade_date}")
        # 读取 `path` 的当前值，并赋给 `paths[trade_date]`。
        paths[trade_date] = path
    # 调用 `dict` 整理数据结构，并将结果保存到 `paths`。
    paths = dict(sorted(paths.items()))
    # 检查 `max_dates_variable and os.getenv(max_dates_variable)`；根据条件结果选择对应处理分支。
    if max_dates_variable and os.getenv(max_dates_variable):
        # 调用 `dict` 整理数据结构，并将结果保存到 `paths`。
        paths = dict(list(paths.items())[: int(os.environ[max_dates_variable])])
    # 检查 `not paths`；根据条件结果选择对应处理分支。
    if not paths:
        # 抛出 `FileNotFoundError(f'{LEVEL2_DIR} 下没有日度 NPZ')`，阻止不满足数据契约的流程继续执行。
        raise FileNotFoundError(f"{LEVEL2_DIR} 下没有日度 NPZ")
    # 将 `paths` 作为当前函数结果返回给调用方。
    return paths


# 定义 `load_calendars` 函数，封装这一阶段可独立调用的处理步骤。
def load_calendars() -> dict[str, np.ndarray]:
    """读取上交所和深交所交易日历，返回八位日期数组。

    输入：
        无显式输入参数。
    输出：
        dict[str, np.ndarray]。
    """

    # 调用 `pd.read_parquet` 读取源数据，并将结果保存到 `frame`。
    frame = pd.read_parquet(
        CALENDAR_PATH, columns=["trade_days", "s_info_exchmarket"]
    )
    # 调用 `normalize_dates` 按当前截面统计量完成缩尾或标准化处理，并将结果保存到 `frame['trade_days']`。
    frame["trade_days"] = normalize_dates(frame["trade_days"])
    # 构造键值映射，并保存到 `calendars`，供后续按键查询。
    calendars: dict[str, np.ndarray] = {}
    # 遍历 `{'SH': 'SSE', 'SZ': 'SZSE'}.items()`，并把当前元素绑定到 `(exchange, market)` 逐项处理。
    for exchange, market in {"SH": "SSE", "SZ": "SZSE"}.items():
        # 调用 `to_numpy` 把表格列转换为指定精度的 NumPy 数组，并将结果保存到 `dates`。
        dates = (
            frame.loc[
                frame["s_info_exchmarket"].astype(str).eq(market),
                "trade_days",
            ]
            .drop_duplicates()
            .sort_values()
            .to_numpy(dtype=str)
        )
        # 检查 `not len(dates)`；根据条件结果选择对应处理分支。
        if not len(dates):
            # 抛出 `ValueError(f'交易日历缺少 {market}')`，阻止不满足数据契约的流程继续执行。
            raise ValueError(f"交易日历缺少 {market}")
        # 读取 `dates` 的当前值，并赋给 `calendars[exchange]`。
        calendars[exchange] = dates
    # 将 `calendars` 作为当前函数结果返回给调用方。
    return calendars


# 定义 `horizon_dates` 函数，封装这一阶段可独立调用的处理步骤。
def horizon_dates(
    trade_date: str,
    exchange: str,
    calendars: dict[str, np.ndarray],
) -> tuple[str, str]:
    """返回股票所属交易所的 T+1 与 T+11 日期。

    输入：
        trade_date: str。
        exchange: str。
        calendars: dict[str, np.ndarray]。
    输出：
        tuple[str, str]。
    """

    # 读取 `calendars[exchange]` 的当前值，并赋给 `dates`。
    dates = calendars[exchange]
    # 调用 `int` 统一设备或数据类型，并将结果保存到 `position`。
    position = int(np.searchsorted(dates, trade_date))
    # 检查 `position >= len(dates) or dates[position] != trade_date`；根据条件结果选择对应处理分支。
    if position >= len(dates) or dates[position] != trade_date:
        # 抛出 `ValueError(f'{trade_date} 不在 {exchange} 交易日历')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(f"{trade_date} 不在 {exchange} 交易日历")
    # 检查 `position + 11 >= len(dates)`；根据条件结果选择对应处理分支。
    if position + 11 >= len(dates):
        # 抛出 `ValueError(f'{trade_date} 之后不足 11 个交易日')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(f"{trade_date} 之后不足 11 个交易日")
    # 将 `(str(dates[position + 1]), str(dates[position + 11]))` 作为当前函数结果返回给调用方。
    return str(dates[position + 1]), str(dates[position + 11])


# 定义 `instrument_to_wind` 函数，封装这一阶段可独立调用的处理步骤。
def instrument_to_wind(instrument_id: str) -> str:
    """把 ``SH.600000`` 转换为 Wind 使用的 ``600000.SH``。

    输入：
        instrument_id: str。
    输出：
        str。
    """

    # 调用 `str(instrument_id).split` 按分隔符拆分证券代码组成部分，并将结果保存到 `(exchange, security_id)`。
    exchange, security_id = str(instrument_id).split(".", maxsplit=1)
    # 检查 `exchange not in {'SH', 'SZ'}`；根据条件结果选择对应处理分支。
    if exchange not in {"SH", "SZ"}:
        # 抛出 `ValueError(f'不支持的股票代码：{instrument_id}')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(f"不支持的股票代码：{instrument_id}")
    # 将 `f'{security_id}.{exchange}'` 作为当前函数结果返回给调用方。
    return f"{security_id}.{exchange}"


# 定义 `read_image_metadata` 函数，封装这一阶段可独立调用的处理步骤。
def read_image_metadata(
    trade_date: str, path: Path
) -> tuple[np.ndarray, np.ndarray]:
    """校验图像 NPZ 契约，返回股票代码和独立特征有效标记。

    输入：
        trade_date: str。
        path: Path。
    输出：
        tuple[np.ndarray, np.ndarray]。
    """

    # 在 `np.load(path, allow_pickle=False)` 管理的上下文中执行本段逻辑，并保证资源按约定释放。
    with np.load(path, allow_pickle=False) as archive:
        # 调用 `REQUIRED_IMAGE_KEYS.difference` 计算两个集合的差集以找出缺失字段，并将结果保存到 `missing`。
        missing = REQUIRED_IMAGE_KEYS.difference(archive.files)
        # 检查 `missing`；根据条件结果选择对应处理分支。
        if missing:
            # 抛出 `KeyError(f'{path.name} 缺少字段：{sorted(missing)}')`，阻止不满足数据契约的流程继续执行。
            raise KeyError(f"{path.name} 缺少字段：{sorted(missing)}")
        # 调用 `archive['instrument_ids'].astype` 统一设备或数据类型，并将结果保存到 `instruments`。
        instruments = archive["instrument_ids"].astype(str)
        # 调用 `archive['valid_mask'].astype` 统一设备或数据类型，并将结果保存到 `valid_mask`。
        valid_mask = archive["valid_mask"].astype(bool)
        # 检查 `str(archive['trade_date'].item()) != trade_date`；根据条件结果选择对应处理分支。
        if str(archive["trade_date"].item()) != trade_date:
            # 抛出 `ValueError(f'{path.name} 文件名日期与内部日期不一致')`，阻止不满足数据契约的流程继续执行。
            raise ValueError(f"{path.name} 文件名日期与内部日期不一致")
        # 检查 `archive['pixel_matrices'].shape != (len(instruments), 17, 8, 8)`；根据条件结果选择对应处理分支。
        if archive["pixel_matrices"].shape != (
            len(instruments), 17, 8, 8
        ):
            # 抛出 `ValueError(f'{path.name} pixel_matrices 形状异常')`，阻止不满足数据契约的流程继续执行。
            raise ValueError(f"{path.name} pixel_matrices 形状异常")
        # 检查 `valid_mask.shape != (len(instruments),)`；根据条件结果选择对应处理分支。
        if valid_mask.shape != (len(instruments),):
            # 抛出 `ValueError(f'{path.name} valid_mask 未与股票行对齐')`，阻止不满足数据契约的流程继续执行。
            raise ValueError(f"{path.name} valid_mask 未与股票行对齐")
        # 检查 `len(np.unique(instruments)) != len(instruments)`；根据条件结果选择对应处理分支。
        if len(np.unique(instruments)) != len(instruments):
            # 抛出 `ValueError(f'{path.name} instrument_ids 重复')`，阻止不满足数据契约的流程继续执行。
            raise ValueError(f"{path.name} instrument_ids 重复")
    # 将 `(instruments, valid_mask)` 作为当前函数结果返回给调用方。
    return instruments, valid_mask
# endregion


# region TR-DATA-UNIVERSE-STATUS
# 定义 `read_market_status` 函数，封装这一阶段可独立调用的处理步骤。
def read_market_status(
    required_dates: set[str], required_codes: set[str]
) -> pd.DataFrame:
    """读取所需停牌和涨跌停状态，并验证 ``股票×日期`` 唯一。

    输入：
        required_dates: set[str]。
        required_codes: set[str]。
    输出：
        pd.DataFrame。
    """

    # 构造当前步骤所需的列表，并保存到 `filters`。
    filters = [
        ("trade_dt", ">=", min(required_dates)),
        ("trade_dt", "<=", max(required_dates)),
    ]
    # 调用 `pd.read_parquet` 读取源数据，并将结果保存到 `eod`。
    eod = pd.read_parquet(
        EOD_PATH,
        columns=["s_info_windcode", "trade_dt", "s_dq_tradestatus"],
        filters=filters,
    )
    # 调用 `pd.read_parquet` 读取源数据，并将结果保存到 `derivative`。
    derivative = pd.read_parquet(
        DERIVATIVE_PATH,
        columns=["s_info_windcode", "trade_dt", "up_down_limit_status"],
        filters=filters,
    )
    # 构造当前步骤所需的列表，并保存到 `filtered_frames`。
    filtered_frames: list[pd.DataFrame] = []
    # 遍历 `(eod, derivative)`，并把当前元素绑定到 `frame` 逐项处理。
    for frame in (eod, derivative):
        # 调用 `normalize_dates` 按当前截面统计量完成缩尾或标准化处理，并将结果保存到 `frame['trade_dt']`。
        frame["trade_dt"] = normalize_dates(frame["trade_dt"])
        # 调用 `frame['s_info_windcode'].astype` 统一设备或数据类型，并将结果保存到 `frame['s_info_windcode']`。
        frame["s_info_windcode"] = frame["s_info_windcode"].astype("string")
        # 调用 `copy` 复制当前对象以避免后续原地修改污染源数据，并将结果保存到 `frame`。
        frame = frame.loc[
            frame["trade_dt"].isin(required_dates)
            & frame["s_info_windcode"].isin(required_codes)
        ].copy()
        # 调用 `frame.duplicated` 标记股票与日期键重复的记录，并将结果保存到 `duplicated`。
        duplicated = frame.duplicated(
            ["s_info_windcode", "trade_dt"], keep=False
        )
        # 检查 `duplicated.any()`；根据条件结果选择对应处理分支。
        if duplicated.any():
            # 抛出 `ValueError('Wind 状态业务键重复：\n' + frame.loc[duplicated].head().to_string(index=False))`，阻止不满足数据契约的流程继续执行。
            raise ValueError(
                "Wind 状态业务键重复：\n"
                + frame.loc[duplicated].head().to_string(index=False)
            )
        # 把本轮结果追加到 `filtered_frames`，累积后续需要汇总的记录。
        filtered_frames.append(frame)
    # 将 `filtered_frames[0].merge(filtered_frames[1], on=['s_info_windcode', 'trade_dt'], how='outer', v…` 作为当前函数结果返回给调用方。
    return filtered_frames[0].merge(
        filtered_frames[1],
        on=["s_info_windcode", "trade_dt"],
        how="outer",
        validate="one_to_one",
    )


# 定义 `classify_market_eligibility` 函数，封装这一阶段可独立调用的处理步骤。
def classify_market_eligibility(candidates: pd.DataFrame) -> pd.DataFrame:
    """根据 T/T+1 状态生成两个市场有效性标记。

输入必须已经包含 T 日和 T+1 的交易、涨跌停状态。函数不删除行，
只增加布尔列；缺失状态通过 ``known_*`` 逻辑自然归为 ``False``。

    输入：
        candidates: pd.DataFrame。
    输出：
        pd.DataFrame。
    """

    # 复制 `candidates.copy`，避免后续更新污染原对象，并保存到 `output`。
    output = candidates.copy()
    # 计算 `output['tradestatus_t'].notna() & output['limit_status_t'].notna()`，并将数值结果保存到 `known_t`。
    known_t = output["tradestatus_t"].notna() & output[
        "limit_status_t"
    ].notna()
    # 计算 `output['tradestatus_t1'].notna() & output['limit_status_t1'].notna()`，并将数值结果保存到 `known_t1`。
    known_t1 = output["tradestatus_t1"].notna() & output[
        "limit_status_t1"
    ].notna()
    # 计算 `known_t & ~output['tradestatus_t'].astype('string').eq('停牌') & ~output['limit_status_t'].isin([…`，并将数值结果保存到 `output['in_universe_t']`。
    output["in_universe_t"] = (
        known_t
        & ~output["tradestatus_t"].astype("string").eq("停牌")
        & ~output["limit_status_t"].isin([-1, 1])
    )
    # 计算 `known_t1 & ~output['tradestatus_t1'].astype('string').eq('停牌') & ~output['limit_status_t1'].isi…`，并将数值结果保存到 `output['can_enter_t1']`。
    output["can_enter_t1"] = (
        known_t1
        & ~output["tradestatus_t1"].astype("string").eq("停牌")
        & ~output["limit_status_t1"].isin([-1, 1])
    )
    # 将 `output` 作为当前函数结果返回给调用方。
    return output
# endregion


# region TR-DATA-UNIVERSE-BUILD
# 定义 `build_universe` 函数，封装这一阶段可独立调用的处理步骤。
def build_universe() -> UniverseBuildResult:
    """构建全市场有效 Universe、逐日质量表和 lineage manifest。

    输入：
        无显式输入参数。
    输出：
        UniverseBuildResult。
    """

    # 调用 `get_data_root` 解析当前 run 对应的数据与输出路径，并将结果保存到 `root`。
    root = get_data_root()
    # 计算 `root / 'universe'`，并将数值结果保存到 `universe_dir`。
    universe_dir = root / "universe"
    # 计算 `root / 'quality'`，并将数值结果保存到 `quality_dir`。
    quality_dir = root / "quality"
    # 调用 `universe_dir.mkdir` 创建输出目录，确保后续文件可以安全写入。
    universe_dir.mkdir(parents=True, exist_ok=True)
    # 调用 `quality_dir.mkdir` 创建输出目录，确保后续文件可以安全写入。
    quality_dir.mkdir(parents=True, exist_ok=True)
    # 计算 `universe_dir / 'universe_manifest.json'`，并将数值结果保存到 `manifest_path`。
    manifest_path = universe_dir / "universe_manifest.json"
    # 计算 `quality_dir / 'universe_quality_by_date.parquet'`，并将数值结果保存到 `quality_path`。
    quality_path = quality_dir / "universe_quality_by_date.parquet"
    # 调用 `list_image_paths` 解析当前 run 对应的数据与输出路径，并将结果保存到 `image_paths`。
    image_paths = list_image_paths("VIT_MAX_IMAGE_DATES")
    # 调用 `load_calendars` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `calendars`。
    calendars = load_calendars()
    # 调用 `sorted` 整理数据结构，并将结果保存到 `years`。
    years = sorted({date[:4] for date in image_paths})
    # 构造键值映射，并保存到 `output_paths`，供后续按键查询。
    output_paths = {
        year: universe_dir / f"universe.{year}.parquet" for year in years
    }

    # 所有上游文件共同进入签名，防止悄悄复用过时 Universe。
    image_signatures = [file_signature(path) for path in image_paths.values()]
    # 构造键值映射，并保存到 `payload`，供后续按键查询。
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
    # 调用 `stable_hash` 对关键配置生成稳定签名以支持复现和一致性检查，并将结果保存到 `signature`。
    signature = stable_hash(payload)
    # 调用 `read_json` 读取源数据，并将结果保存到 `existing`。
    existing = read_json(manifest_path)
    # 根据 `not read_bool_environment('VIT_UNIVERSE_OVERWRITE') and existing is not None and (existing.get(…` 计算条件结果，并保存到 `reusable`。
    reusable = (
        not read_bool_environment("VIT_UNIVERSE_OVERWRITE")
        and existing is not None
        and existing.get("signature") == signature
        and quality_path.exists()
        and all(path.exists() for path in output_paths.values())
    )
    # 检查 `reusable`；根据条件结果选择对应处理分支。
    if reusable:
        # 调用 `print` 记录当前流程状态或诊断信息。
        print("Universe 签名一致，直接复用。")
        # 将 `{'universe_directory': str(universe_dir), 'manifest_path': str(manifest_path), 'quality_path': …` 作为当前函数结果返回给调用方。
        return {
            "universe_directory": str(universe_dir),
            "manifest_path": str(manifest_path),
            "quality_path": str(quality_path),
            "signature": signature,
        }

    # 构造当前步骤所需的列表，并保存到 `quality_records`。
    quality_records: list[dict[str, object]] = []
    # 设置 `total_candidates` 的固定配置值，供后续流程统一使用。
    total_candidates = 0
    # 调用 `print` 记录当前流程状态或诊断信息。
    print("开始构建 Universe：不使用 ST 或上市日期。")
    # 遍历 `years`，并把当前元素绑定到 `year` 逐项处理。
    for year in years:
        # 构造当前步骤所需的列表，并保存到 `daily_frames`。
        daily_frames: list[pd.DataFrame] = []
        # 构造当前步骤所需的列表，并保存到 `year_dates`。
        year_dates = [date for date in image_paths if date.startswith(year)]
        # 遍历 `year_dates`，并把当前元素绑定到 `trade_date` 逐项处理。
        for trade_date in year_dates:
            # 调用 `read_image_metadata` 读取源数据，并将结果保存到 `(instruments, feature_valid)`。
            instruments, feature_valid = read_image_metadata(
                trade_date, image_paths[trade_date]
            )
            # 读取 `np.char.partition(instruments, '.')[:, 0]` 的当前值，并赋给 `exchanges`。
            exchanges = np.char.partition(instruments, ".")[:, 0]
            # 调用 `np.full` 按目标形状初始化数值容器 `entry_dates`。
            entry_dates = np.full(len(instruments), "", dtype="<U8")
            # 调用 `np.full` 按目标形状初始化数值容器 `exit_dates`。
            exit_dates = np.full(len(instruments), "", dtype="<U8")
            # 遍历 `('SH', 'SZ')`，并把当前元素绑定到 `exchange` 逐项处理。
            for exchange in ("SH", "SZ"):
                # 根据 `exchanges == exchange` 计算条件结果，并保存到 `mask`。
                mask = exchanges == exchange
                # 调用 `horizon_dates` 按交易日历定位收益窗口对应日期，并将结果保存到 `(entry_date, exit_date)`。
                entry_date, exit_date = horizon_dates(
                    trade_date, exchange, calendars
                )
                # 读取 `entry_date` 的当前值，并赋给 `entry_dates[mask]`。
                entry_dates[mask] = entry_date
                # 读取 `exit_date` 的当前值，并赋给 `exit_dates[mask]`。
                exit_dates[mask] = exit_date
            # 检查 `(entry_dates == '').any() or (exit_dates == '').any()`；根据条件结果选择对应处理分支。
            if (entry_dates == "").any() or (exit_dates == "").any():
                # 抛出 `ValueError(f'{trade_date} 含不支持的交易所股票')`，阻止不满足数据契约的流程继续执行。
                raise ValueError(f"{trade_date} 含不支持的交易所股票")
            # 把本轮结果追加到 `daily_frames`，累积后续需要汇总的记录。
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

        # 调用 `pd.concat` 拼接各批结果，并将合并值保存到 `candidates`。
        candidates = pd.concat(daily_frames, ignore_index=True)
        # 计算 `set(candidates['trade_date']) | set(candidates['entry_trade_date'])`，并将数值结果保存到 `required_dates`。
        required_dates = set(candidates["trade_date"]) | set(
            candidates["entry_trade_date"]
        )
        # 调用 `read_market_status` 读取源数据，并将结果保存到 `status`。
        status = read_market_status(
            required_dates, set(candidates["wind_code"])
        )
        # 调用 `status.rename` 把完整临时产物替换为正式输出，并将结果保存到 `t_status`。
        t_status = status.rename(
            columns={
                "s_info_windcode": "wind_code",
                "trade_dt": "trade_date",
                "s_dq_tradestatus": "tradestatus_t",
                "up_down_limit_status": "limit_status_t",
            }
        )
        # 调用 `status.rename` 把完整临时产物替换为正式输出，并将结果保存到 `t1_status`。
        t1_status = status.rename(
            columns={
                "s_info_windcode": "wind_code",
                "trade_dt": "entry_trade_date",
                "s_dq_tradestatus": "tradestatus_t1",
                "up_down_limit_status": "limit_status_t1",
            }
        )
        # 调用 `candidates.merge(t_status, on=['wind_code', 'trade_date'], how=…` 按股票和交易日键对齐来自不同来源的数据，并将结果保存到 `candidates`。
        candidates = (
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
        # 调用 `classify_market_eligibility` 根据停牌与涨跌停状态计算可交易性标记，并将结果保存到 `candidates`。
        candidates = classify_market_eligibility(candidates)

        # 遍历 `candidates.groupby('trade_date', sort=True)`，并把当前元素绑定到 `(trade_date, daily)` 逐项处理。
        for trade_date, daily in candidates.groupby(
            "trade_date", sort=True
        ):
            # 把本轮结果追加到 `quality_records`，累积后续需要汇总的记录。
            quality_records.append(
                _daily_quality_record(str(trade_date), daily)
            )

        # 构造当前步骤所需的列表，并保存到 `output_columns`。
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
        # 调用 `candidates[output_columns].sort_values` 根据分数计算截面排序或排序索引，并将结果保存到 `output`。
        output = candidates[output_columns].sort_values(
            ["trade_date", "stock_index"], kind="stable"
        )
        # 检查 `output.duplicated(['trade_date', 'instrument_id']).any()`；根据条件结果选择对应处理分支。
        if output.duplicated(["trade_date", "instrument_id"]).any():
            # 抛出 `ValueError(f'{year} Universe 业务键重复')`，阻止不满足数据契约的流程继续执行。
            raise ValueError(f"{year} Universe 业务键重复")
        # 调用 `atomic_parquet` 先写临时文件再原子替换正式产物，避免留下半成品。
        atomic_parquet(output, output_paths[year])
        # 用 `len(output)` 对 `total_candidates` 做 `Add` 累积更新。
        total_candidates += len(output)
        # 调用 `print` 记录当前流程状态或诊断信息。
        print(
            f"{year}：候选 {len(output):,}，T 日池 "
            f"{int(output['in_universe_t'].sum()):,}，可预测 "
            f"{int((output['feature_valid'] & output['in_universe_t']).sum()):,}"
        )
        # 删除 `daily_frames, candidates, output, status, t_status, t1_status`，释放不再使用的引用。
        del daily_frames, candidates, output, status, t_status, t1_status
        # 调用 `gc.collect` 触发 Python 垃圾回收以释放失去引用的对象。
        gc.collect()

    # 调用 `pd.DataFrame(quality_records).sort_values` 根据分数计算截面排序或排序索引，并将结果保存到 `quality`。
    quality = pd.DataFrame(quality_records).sort_values("trade_date")
    # 检查 `not (quality['candidate_rows'] == quality['in_universe_t_rows'] + quality['excluded_t_rows']).a…`；根据条件结果选择对应处理分支。
    if not (
        quality["candidate_rows"]
        == quality["in_universe_t_rows"] + quality["excluded_t_rows"]
    ).all():
        # 抛出 `AssertionError('Universe 每日候选数无法守恒')`，阻止不满足数据契约的流程继续执行。
        raise AssertionError("Universe 每日候选数无法守恒")
    # 调用 `atomic_parquet` 先写临时文件再原子替换正式产物，避免留下半成品。
    atomic_parquet(quality, quality_path)
    # 调用 `atomic_json` 先写临时文件再原子替换正式产物，避免留下半成品。
    atomic_json(
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
    # 将 `{'universe_directory': str(universe_dir), 'manifest_path': str(manifest_path), 'quality_path': …` 作为当前函数结果返回给调用方。
    return {
        "universe_directory": str(universe_dir),
        "manifest_path": str(manifest_path),
        "quality_path": str(quality_path),
        "signature": signature,
    }
# endregion


# 定义内部辅助函数 `_daily_quality_record`，封装仅供本模块复用的计算步骤。
def _daily_quality_record(
    trade_date: str, daily: pd.DataFrame
) -> dict[str, object]:
    """汇总一日 Universe 的逐步过滤数量守恒信息。

    输入：
        trade_date: str。
        daily: pd.DataFrame。
    输出：
        dict[str, object]。
    """

    # 将 `{'trade_date': trade_date, 'candidate_rows': len(daily), 'feature_valid_rows': int(daily['featu…` 作为当前函数结果返回给调用方。
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


# 定义命令行入口，解析参数并按用户选择调度当前阶段的主流程。
def main() -> None:
    """构建 Universe 并打印当前产物清单。

    输入：
        无显式输入参数。
    输出：
        None。
    """
    # 调用 `print` 记录当前流程状态或诊断信息。
    print(json.dumps(build_universe(), ensure_ascii=False, indent=2))


# 检查 `__name__ == '__main__'`；根据条件结果选择对应处理分支。
if __name__ == "__main__":
    # 调用 `main` 启动当前模块的命令行主流程。
    main()
