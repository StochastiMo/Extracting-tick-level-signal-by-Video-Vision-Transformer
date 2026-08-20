"""ViViT 基线因子的输入校验与数据准备。"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


IMAGE_PATTERN = re.compile(r"^stock_pixels_cn\.(\d{8})\.npz$")
# 每日图片必须同时提供这些键；下游依赖固定的股票顺序和有效性掩码。
IMAGE_KEYS = {
    "trade_date", "instrument_ids", "pixel_matrices", "valid_mask",
    "channel_names", "axis_order",
}
FORECAST_YEARS = tuple(range(2018, 2026))
SEEDS = (42, 43, 44)
SEQUENCE_LENGTH = 20


@dataclass(frozen=True)
class PipelinePaths:
    """集中管理流水线使用的输入路径和产物路径。"""

    project_dir: Path
    level2_dir: Path
    wind_dir: Path
    artifacts_dir: Path

    @classmethod
    def default(cls) -> "PipelinePaths":
        # 交付目录与 level2_pic、Wind 平行，避免依赖运行命令所在位置。
        project_dir = Path(__file__).resolve().parents[1]  # 以包文件定位项目根目录，不依赖当前工作目录。
        input_root = project_dir.parent  # 输入数据约定放在项目目录的同级目录。
        wind_dir = input_root / "wind"  # 优先采用 README 中约定的小写目录名。
        # Linux 区分大小写；同时兼容实际交付目录 Wind/ 和约定写法 wind/。
        if not wind_dir.exists() and (input_root / "Wind").exists():
            wind_dir = input_root / "Wind"  # 实际目录为大写时切换到兼容路径。
        return cls(
            project_dir=project_dir,
            level2_dir=input_root / "level2_pic",
            wind_dir=wind_dir,
            artifacts_dir=project_dir / "artifacts",
        )

    @property
    def eod_path(self) -> Path:
        return self.wind_dir / "ASHAREEODPRICES.parquet"

    @property
    def derivative_path(self) -> Path:
        return self.wind_dir / "ASHAREEODDERIVATIVEINDICATOR.parquet"

    @property
    def calendar_path(self) -> Path:
        return self.wind_dir / "ASHARECALENDAR.parquet"

    @property
    def data_dir(self) -> Path:
        return self.artifacts_dir / "data"

    @property
    def universe_dir(self) -> Path:
        return self.data_dir / "universe"

    @property
    def labels_dir(self) -> Path:
        return self.data_dir / "labels"

    @property
    def feature_dir(self) -> Path:
        return self.data_dir / "feature_store"

    @property
    def split_dir(self) -> Path:
        return self.feature_dir / "rolling_splits"

    @property
    def features_path(self) -> Path:
        return self.feature_dir / "features.npy"

    @property
    def targets_path(self) -> Path:
        return self.feature_dir / "targets.npy"

    @property
    def label_valid_path(self) -> Path:
        return self.feature_dir / "label_valid.npy"

    @property
    def date_ids_path(self) -> Path:
        return self.feature_dir / "date_ids.npy"

    @property
    def history_path(self) -> Path:
        return self.feature_dir / "history_indices.npy"

    @property
    def sample_index_path(self) -> Path:
        return self.feature_dir / "sample_index.parquet"

    @property
    def factors_dir(self) -> Path:
        return self.artifacts_dir / "factors"

    @property
    def models_dir(self) -> Path:
        return self.artifacts_dir / "models"

    @property
    def logs_dir(self) -> Path:
        return self.artifacts_dir / "logs"


def stable_hash(payload: object) -> str:
    # 固定 JSON 字段顺序，使相同配置跨进程得到相同签名。
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))  # 序列化为稳定字节表示。
    return hashlib.sha256(text.encode("utf-8")).hexdigest()  # 用摘要标识输入和配置是否发生变化。


def file_signature(path: Path) -> dict[str, object]:
    stat = path.stat()  # 读取文件大小和纳秒级修改时间。
    return {"name": path.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}  # 形成轻量缓存指纹。


def atomic_json(payload: dict[str, object], path: Path) -> None:
    # 先写同目录临时文件，再原子替换，防止中断后留下半成品。
    path.parent.mkdir(parents=True, exist_ok=True)  # 确保产物目录存在。
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")  # 临时名避免并发进程冲突。
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")  # 完整写入临时清单。
    temporary.replace(path)  # 同文件系统原子替换正式清单。


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    # Parquet 同样采用原子替换，训练进程不会读到未写完的文件。
    path.parent.mkdir(parents=True, exist_ok=True)  # 创建目标目录。
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")  # 隔离未完成的 Parquet。
    frame.to_parquet(temporary, index=False, compression="zstd")  # 不保存 DataFrame 行号并启用无损压缩。
    temporary.replace(path)  # 写完后一次性暴露正式文件。


def atomic_numpy(array: np.ndarray, path: Path) -> None:
    # 显式关闭 pickle，保证数组产物只包含数值数据。
    path.parent.mkdir(parents=True, exist_ok=True)  # 创建数组产物目录。
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")  # 使用进程唯一临时路径。
    with temporary.open("wb") as handle:  # 以二进制方式写入标准 NPY。
        np.save(handle, array, allow_pickle=False)  # 禁止对象反序列化，保持纯数值契约。
    temporary.replace(path)  # 原子发布完整数组。


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    # 滚动切分体积较小，使用压缩 NPZ 一次性保存全部索引。
    path.parent.mkdir(parents=True, exist_ok=True)  # 创建切分索引目录。
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")  # 避免覆盖可用旧版本。
    with temporary.open("wb") as handle:  # 打开压缩归档的临时输出流。
        np.savez_compressed(handle, **arrays)  # 将同一切分的多个索引数组一起保存。
    temporary.replace(path)  # 原子替换旧归档。


def normalize_dates(values: pd.Series) -> pd.Series:
    # Wind 日期可能被读取成浮点字符串，统一还原为 YYYYMMDD。
    return values.astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(8)  # 消除浮点尾缀并补齐八位日期。


def list_image_paths(paths: PipelinePaths) -> dict[str, Path]:
    # 文件名中的日期是图片与行情数据对齐的唯一日期键。
    result: dict[str, Path] = {}  # 建立交易日到图片文件的一对一映射。
    for path in paths.level2_dir.glob("stock_pixels_cn.*.npz"):  # 只遍历约定前缀的 NPZ 文件。
        match = IMAGE_PATTERN.match(path.name)  # 严格提取八位交易日。
        if match:  # 忽略前缀相同但命名不合规的文件。
            trade_date = match.group(1)  # 文件名日期作为全流程日期键。
            if trade_date in result:  # 同一日期只能存在一份图片。
                raise ValueError(f"duplicate Level2 image date: {trade_date}")  # 防止后写文件静默覆盖。
            result[trade_date] = path  # 保存该日图片路径。
    if not result:  # 没有图片时不能继续构建样本。
        raise FileNotFoundError(f"no stock_pixels_cn.YYYYMMDD.npz under {paths.level2_dir}")  # 明确报告输入目录。
    return dict(sorted(result.items()))  # 固定日期顺序，保证后续样本可复现。


def validate_inputs(paths: PipelinePaths) -> dict[str, object]:
    # 先校验三个必需的 Wind Parquet，失败时在耗时处理前立即退出。
    required = (paths.eod_path, paths.derivative_path, paths.calendar_path)  # 汇总三个硬性依赖的 Wind 文件。
    missing = [str(path) for path in required if not path.exists()]  # 一次收集全部缺失项。
    if missing:  # 任一行情文件缺失都无法形成 Universe。
        raise FileNotFoundError(f"missing Wind parquet files: {missing}")  # 同时报出全部缺失路径。
    image_paths = list_image_paths(paths)  # 校验图片目录并取得有序文件表。
    first_path = next(iter(image_paths.values()))  # 用首日文件快速核验图片结构。
    with np.load(first_path, allow_pickle=False) as archive:  # 禁止 pickle，按只读数值归档打开。
        # 抽查首个文件的数据契约；逐日读取时仍会复核日期和股票索引。
        missing_keys = IMAGE_KEYS.difference(archive.files)  # 比较实际键与固定数据契约。
        if missing_keys:  # 缺任一键都会造成后续索引或张量错位。
            raise KeyError(f"{first_path.name} missing keys: {sorted(missing_keys)}")  # 列出缺失键便于修复数据。
        shape = archive["pixel_matrices"].shape  # 读取股票数和通道空间维度。
        if shape[1:] != (17, 8, 8):  # 模型固定接收 17×8×8 单日图片。
            raise ValueError(f"unexpected Level2 shape: {shape}")  # 在建库前拒绝错误图片形状。
    return {
        "image_count": len(image_paths),
        "first_image_date": min(image_paths),
        "last_image_date": max(image_paths),
        "wind_dir": str(paths.wind_dir),
    }


def load_calendars(paths: PipelinePaths) -> dict[str, np.ndarray]:
    # 沪深交易日分别推进，不能用自然日或统一日历替代 T+1。
    frame = pd.read_parquet(paths.calendar_path, columns=["trade_days", "s_info_exchmarket"])  # 仅读取交易日与市场列。
    frame["trade_days"] = normalize_dates(frame["trade_days"])  # 将日期统一为可排序字符串。
    calendars: dict[str, np.ndarray] = {}  # 分别保存沪深交易日序列。
    for exchange, market in {"SH": "SSE", "SZ": "SZSE"}.items():  # 图片代码与 Wind 市场代码逐一映射。
        dates = (
            frame.loc[frame["s_info_exchmarket"].astype(str).eq(market), "trade_days"]
            .drop_duplicates().sort_values().to_numpy(dtype=str)
        )
        if not len(dates):  # 任一交易所日历为空都无法计算对应股票 T+1。
            raise ValueError(f"calendar missing {market}")  # 明确指出缺失市场。
        calendars[exchange] = dates  # 保存去重且升序的交易日数组。
    return calendars  # 返回供日期位移复用的交易所日历。


def calendar_shift(trade_date: str, exchange: str, calendars: dict[str, np.ndarray], steps: int) -> str:
    # searchsorted 同时验证 T 本身在对应交易所日历中。
    dates = calendars[exchange]  # 只在股票所属交易所日历内移动。
    position = int(np.searchsorted(dates, trade_date))  # 二分定位 T 日位置。
    if position >= len(dates) or dates[position] != trade_date or position + steps >= len(dates):  # 同时检查起点和目标是否存在。
        raise ValueError(f"cannot shift {trade_date} by {steps} on {exchange}")  # 禁止用错误日期继续对齐行情。
    return str(dates[position + steps])  # 返回精确移动后的交易日。


def instrument_to_wind(instrument_id: str) -> str:
    # 图片代码为 SH.600000，Wind 行情键为 600000.SH。
    exchange, security_id = str(instrument_id).split(".", maxsplit=1)  # 拆分图片中的交易所前缀和证券代码。
    if exchange not in {"SH", "SZ"}:  # 当前行情映射只支持沪深股票。
        raise ValueError(f"unsupported instrument: {instrument_id}")  # 避免生成无效 Wind 代码。
    return f"{security_id}.{exchange}"  # 调换顺序得到 Wind 标准代码。


def read_image_metadata(trade_date: str, path: Path) -> tuple[np.ndarray, np.ndarray]:
    # 此处只读元数据，不加载整批像素，降低构建 Universe 时的内存占用。
    with np.load(path, allow_pickle=False) as archive:  # 只读取可信数值数组，避免加载对象。
        missing = IMAGE_KEYS.difference(archive.files)  # 逐日复核必需键，不能只依赖首日抽查。
        if missing:  # 数据契约不完整时立即停止。
            raise KeyError(f"{path.name} missing keys: {sorted(missing)}")  # 报出具体文件和缺失键。
        instruments = archive["instrument_ids"].astype(str)  # 固化图片行对应的股票代码。
        valid = archive["valid_mask"].astype(bool)  # 转为布尔掩码供 Universe 直接使用。
        if str(archive["trade_date"].item()) != trade_date:  # 文件内部日期必须与文件名一致。
            raise ValueError(f"date mismatch in {path.name}")  # 防止图片挂到错误交易日。
        if archive["pixel_matrices"].shape != (len(instruments), 17, 8, 8):  # 股票行数和模型输入维度必须同时匹配。
            raise ValueError(f"invalid pixel shape in {path.name}")  # 拒绝形状错配。
        if valid.shape != (len(instruments),) or len(np.unique(instruments)) != len(instruments):  # 掩码长度和股票唯一性共同约束行索引。
            raise ValueError(f"invalid stock index contract in {path.name}")  # 防止同一像素行映射歧义。
    return instruments, valid  # 返回股票顺序及其 T 日图片有效性。


def classify_universe(candidates: pd.DataFrame) -> pd.DataFrame:
    """按 T/T+1 可交易性和 T 日图片规则标记样本，但保留全部候选行。"""
    output = candidates.copy()  # 保留候选表供审计，不在此处直接删行。
    # 缺失状态一律视为不可用；这里没有额外过滤 T 日涨跌停。
    output["is_eligible"] = (
        output["feature_valid"].fillna(False).astype(bool)
        & output["tradestatus_t"].notna()
        & ~output["tradestatus_t"].astype("string").eq("停牌")
        & output["tradestatus_t1"].notna()
        & ~output["tradestatus_t1"].astype("string").eq("停牌")
        & output["limit_status_t1"].notna()
        & ~output["limit_status_t1"].isin([-1, 1])
    )
    return output  # 仅新增入选标记，交由下游筛选。


def read_market_status(
    paths: PipelinePaths, required_dates: set[str], required_codes: set[str]
) -> pd.DataFrame:
    # 仅按本年度候选股票和 T/T+1 日期下推读取，避免扫描完整行情表。
    filters = [("trade_dt", ">=", min(required_dates)), ("trade_dt", "<=", max(required_dates))]  # 用日期上下界下推 Parquet 过滤。
    eod = pd.read_parquet(  # 读取 T/T+1 停复牌状态。
        paths.eod_path,
        columns=["s_info_windcode", "trade_dt", "s_dq_tradestatus"],
        filters=filters,
    )
    derivative = pd.read_parquet(  # 读取 T+1 涨跌停状态。
        paths.derivative_path,
        columns=["s_info_windcode", "trade_dt", "up_down_limit_status"],
        filters=filters,
    )
    frames = []  # 保存清洗后的两张状态表。
    for frame in (eod, derivative):  # 两类状态执行完全相同的键清洗。
        # 两张表先各自去重校验，再以股票和日期做一对一合并。
        frame["trade_dt"] = normalize_dates(frame["trade_dt"])  # 统一连接键日期格式。
        frame["s_info_windcode"] = frame["s_info_windcode"].astype("string")  # 统一证券代码类型。
        frame = frame.loc[  # 只保留当前年度实际需要的日期和股票。
            frame["trade_dt"].isin(required_dates)
            & frame["s_info_windcode"].isin(required_codes)
        ].copy()
        if frame.duplicated(["s_info_windcode", "trade_dt"]).any():  # 状态表连接键必须唯一。
            raise ValueError("duplicate Wind stock/date status key")  # 禁止重复行情放大候选行。
        frames.append(frame)  # 保留清洗结果供一对一合并。
    return frames[0].merge(  # 外连接保留任一来源存在的状态，缺失值随后按不可用处理。
        frames[1], on=["s_info_windcode", "trade_dt"], how="outer", validate="one_to_one"
    )


def build_universe(paths: PipelinePaths, overwrite: bool = False) -> dict[str, object]:
    # 输入文件签名和筛选口径共同决定缓存是否仍然可复用。
    validate_inputs(paths)  # 在大规模扫描前验证所有输入契约。
    image_paths = list_image_paths(paths)  # 建立全量日期到图片路径的有序映射。
    calendars = load_calendars(paths)  # 加载沪深独立交易日历供 T+1 对齐。
    years = sorted({date[:4] for date in image_paths})  # 从实际图片日期确定需要处理的年度。
    output_paths = {year: paths.universe_dir / f"universe.{year}.parquet" for year in years}  # 每年独立落盘便于增量读取。
    manifest_path = paths.universe_dir / "manifest.json"  # 总清单控制所有年度缓存有效性。
    payload = {  # 将输入指纹和筛选口径写入缓存签名。
        "schema_version": 1,
        "image_hash": stable_hash([file_signature(path) for path in image_paths.values()]),
        "eod": file_signature(paths.eod_path),
        "derivative": file_signature(paths.derivative_path),
        "calendar": file_signature(paths.calendar_path),
        "rule": "valid T image; T and T+1 not suspended; T+1 not limit up/down",
    }
    signature = stable_hash(payload)  # 任一图片、行情或规则变化都会得到新签名。
    if not overwrite and manifest_path.exists() and all(path.exists() for path in output_paths.values()):  # 仅在产物完整时尝试复用。
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))  # 读取上次构建清单。
        if existing.get("signature") == signature:  # 签名相同表示无需重新清洗。
            return existing  # 直接返回缓存统计。

    total_rows = eligible_rows = 0  # 累计候选数与最终入选数。
    for year in years:  # 按年处理以控制行情连接的内存占用。
        daily_frames = []  # 暂存本年度逐日图片元数据。
        for trade_date in (date for date in image_paths if date.startswith(year)):  # 按日期顺序构建候选横截面。
            # 图片中的 stock_index 必须原样保留，后续据此从像素矩阵取行。
            instruments, feature_valid = read_image_metadata(trade_date, image_paths[trade_date])  # 取得图片股票顺序和有效掩码。
            exchanges = np.char.partition(instruments, ".")[:, 0]  # 从图片代码提取所属交易所。
            t1_dates = np.full(len(instruments), "", dtype="<U8")  # 预分配每只股票对应的下一交易日。
            for exchange in ("SH", "SZ"):  # 分交易所推进日历。
                mask = exchanges == exchange  # 选出当前交易所股票。
                if mask.any():  # 无该交易所股票时跳过日历查询。
                    # 每只股票按所属交易所推进一个交易日，不硬编码具体日期。
                    t1_dates[mask] = calendar_shift(trade_date, exchange, calendars, 1)  # 为该组股票写入精确 T+1。
            if (t1_dates == "").any():  # 空值说明出现未支持交易所。
                raise ValueError(f"unsupported exchange on {trade_date}")  # 禁止缺失 T+1 的股票进入连接。
            daily_frames.append(pd.DataFrame({  # 保存该日全部候选及其图片行号。
                "wind_code": [instrument_to_wind(value) for value in instruments],
                "trade_date": trade_date,
                "t1_trade_date": t1_dates,
                "instrument_id": instruments,
                "stock_index": np.arange(len(instruments), dtype=np.int32),
                "feature_valid": feature_valid,
            }))
        candidates = pd.concat(daily_frames, ignore_index=True)  # 合并年度候选并重建连续行号。
        # 一次读取本年所有 T 和 T+1 状态，再拆成两份回连候选表。
        required_dates = set(candidates["trade_date"]) | set(candidates["t1_trade_date"])  # 行情读取同时覆盖 T 和 T+1。
        status = read_market_status(paths, required_dates, set(candidates["wind_code"]))  # 一次取得所需停牌和涨跌停状态。
        t_status = status[["s_info_windcode", "trade_dt", "s_dq_tradestatus"]].rename(columns={  # 整理 T 日停牌连接键。
            "s_info_windcode": "wind_code", "trade_dt": "trade_date",
            "s_dq_tradestatus": "tradestatus_t",
        })
        t1_status = status.rename(columns={  # 将同一状态表重命名为 T+1 语义。
            "s_info_windcode": "wind_code", "trade_dt": "t1_trade_date",
            "s_dq_tradestatus": "tradestatus_t1",
            "up_down_limit_status": "limit_status_t1",
        })
        candidates = candidates.merge(  # 先连接 T 日停牌状态。
            t_status, on=["wind_code", "trade_date"], how="left", validate="many_to_one"
        ).merge(
            t1_status[["wind_code", "t1_trade_date", "tradestatus_t1", "limit_status_t1"]],
            on=["wind_code", "t1_trade_date"], how="left", validate="many_to_one",
        )
        # 保留未入选行便于审计，只通过 is_eligible 控制后续样本。
        candidates = classify_universe(candidates).sort_values(  # 计算入选标记并恢复确定性顺序。
            ["trade_date", "stock_index"], kind="stable"
        )
        if candidates.duplicated(["trade_date", "instrument_id"]).any():  # 每日每只股票只能有一条候选记录。
            raise ValueError(f"duplicate universe key in {year}")  # 防止重复样本进入特征库。
        atomic_parquet(candidates, output_paths[year])  # 原子保存完整可审计 Universe。
        total_rows += len(candidates)  # 累加年度候选行数。
        eligible_rows += int(candidates["is_eligible"].sum())  # 累加满足交易条件的样本数。
        print(f"universe {year}: {len(candidates):,} rows, {int(candidates['is_eligible'].sum()):,} eligible")  # 输出年度进度。

    result = {
        "signature": signature, "payload": payload, "years": years,
        "candidate_rows": total_rows, "eligible_rows": eligible_rows,
    }
    atomic_json(result, manifest_path)  # 最后发布与所有年度文件匹配的清单。
    return result  # 返回构建签名和汇总计数。


def calculate_forward_vwap_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """按股票直接位移，计算 T+1 到 T+11 的复权 VWAP 收益率。"""
    required = {"s_info_windcode", "trade_dt", "s_dq_avgprice", "s_dq_adjfactor"}  # 声明收益计算所需最小字段集。
    missing = required.difference(prices.columns)  # 检查调用方是否提供全部字段。
    if missing:  # 缺列时不能可靠计算复权 VWAP。
        raise KeyError(f"price data missing columns: {sorted(missing)}")  # 明确列出缺失字段。
    frame = prices[list(required)].copy()  # 隔离所需列，避免修改调用方数据。
    frame["trade_dt"] = normalize_dates(frame["trade_dt"])  # 统一组内时间排序格式。
    frame["s_info_windcode"] = frame["s_info_windcode"].astype("string")  # 固化分组键类型。
    frame = frame.sort_values(["s_info_windcode", "trade_dt"], kind="stable")  # 先按股票再按日期稳定排序。
    if frame.duplicated(["s_info_windcode", "trade_dt"]).any():  # shift 要求股票日期键唯一。
        raise ValueError("duplicate Wind price key")  # 防止未来第 N 条记录含义歧义。
    # 复权 VWAP 用均价乘复权因子；非正数和无穷值不能参与收益率。
    adjusted_vwap = (
        pd.to_numeric(frame["s_dq_avgprice"], errors="coerce")
        * pd.to_numeric(frame["s_dq_adjfactor"], errors="coerce")
    )
    adjusted_vwap = adjusted_vwap.where(np.isfinite(adjusted_vwap) & adjusted_vwap.gt(0))  # 非法价格统一置为缺失。
    grouped_price = adjusted_vwap.groupby(frame["s_info_windcode"], sort=False)  # 后续位移严格限制在单只股票内。
    grouped_date = frame["trade_dt"].groupby(frame["s_info_windcode"], sort=False)  # 同步位移标签结束日期。
    # 用户口径是不显式对齐日期，直接取每只股票未来第 1 和第 11 条交易记录。
    entry = grouped_price.shift(-1)  # 取每只股票未来第一个 VWAP 作为买入价。
    exit_ = grouped_price.shift(-11)  # 取未来第十一个 VWAP 作为卖出价。
    output = pd.DataFrame({  # 保留入口、出口和结束日便于审计标签。
        "wind_code": frame["s_info_windcode"].to_numpy(),
        "trade_date": frame["trade_dt"].to_numpy(),
        "entry_adjusted_vwap": entry.to_numpy(dtype=np.float64),
        "exit_adjusted_vwap": exit_.to_numpy(dtype=np.float64),
        "label_end_date": grouped_date.shift(-11).to_numpy(),
    })
    output["forward_vwap_return_10d"] = (  # 按用户公式计算持有区间收益。
        output["exit_adjusted_vwap"] / output["entry_adjusted_vwap"] - 1.0
    )
    return output  # 返回未去极值的原始未来收益。


def mad_winsorize_zscore(values: pd.Series) -> pd.DataFrame:
    """先按中位数 ± 3×1.4826×MAD 去极值，再做 ddof=0 标准化。"""
    raw = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)  # 非数值标签转为 NaN。
    valid = np.isfinite(raw)  # 仅有限值参与横截面统计。
    clipped = np.full(len(raw), np.nan, dtype=np.float64)  # 预分配去极值结果并保留无效位置。
    zscore = np.full(len(raw), np.nan, dtype=np.float64)  # 预分配标准化结果。
    lower = upper = np.nan  # MAD 无法计算时边界保持缺失。
    # 少于两个有效样本时无法形成可靠横截面，保持结果为缺失。
    if valid.sum() >= 2:  # 至少两个样本才尝试形成横截面分布。
        selected = raw[valid]  # 取出有效标签参与稳健统计。
        median = float(np.median(selected))  # 中位数作为稳健中心。
        scaled_mad = 1.4826 * float(np.median(np.abs(selected - median)))  # 将 MAD 缩放到正态标准差量纲。
        if np.isfinite(scaled_mad) and scaled_mad > 0:  # 只有正的有限尺度才能定义裁剪区间。
            # MAD 为零时不裁剪，随后标准差检查会阻止无效 z-score。
            lower, upper = median - 3.0 * scaled_mad, median + 3.0 * scaled_mad  # 构造三倍稳健标准差边界。
            selected = np.clip(selected, lower, upper)  # 将极端收益压到边界而非删除。
        clipped[valid] = selected  # 写回原始行位置，维持索引对齐。
        standard_deviation = float(selected.std(ddof=0))  # 使用总体标准差匹配指定口径。
        if np.isfinite(standard_deviation) and standard_deviation > 0:  # 常数横截面不产生伪 z-score。
            zscore[valid] = (selected - selected.mean()) / standard_deviation  # 生成每日均值零、标准差一标签。
    return pd.DataFrame({  # 同时返回中间结果和边界便于审计。
        "winsorized_forward_return": clipped,
        "target_zscore": zscore,
        "mad_lower": lower,
        "mad_upper": upper,
    }, index=values.index)


def _price_read_end(last_date: str, calendars: dict[str, np.ndarray]) -> str:
    # 多读 15 个交易日，为年末样本留出 T+11 标签窗口。
    ends = []  # 收集沪深各自可读取的结束日期。
    for dates in calendars.values():  # 两个交易所分别向后留出缓冲。
        position = int(np.searchsorted(dates, last_date, side="left"))  # 定位年度最后候选日。
        ends.append(str(dates[min(position + 15, len(dates) - 1)]))  # 最多推进十五个交易日且不越界。
    return max(ends)  # 用较晚日期覆盖两市标签窗口。


def build_labels(paths: PipelinePaths, overwrite: bool = False) -> dict[str, object]:
    universe_paths = sorted(paths.universe_dir.glob("universe.*.parquet"))  # 按年度顺序收集 Universe 产物。
    if not universe_paths:  # 标签构建依赖已经筛选的股票池。
        raise FileNotFoundError("run build_universe first")  # 明确提示缺少上游步骤。
    paths.labels_dir.mkdir(parents=True, exist_ok=True)  # 创建年度标签输出目录。
    calendars = load_calendars(paths)  # 用交易日历扩展年末价格读取范围。
    years = [path.stem.split(".")[-1] for path in universe_paths]  # 从文件名提取对应年度。
    outputs = {year: paths.labels_dir / f"labels.{year}.parquet" for year in years}  # 建立年度标签路径映射。
    manifest_path = paths.labels_dir / "manifest.json"  # 标签清单记录输入和口径。
    universe_manifest = json.loads((paths.universe_dir / "manifest.json").read_text(encoding="utf-8"))  # 继承上游数据签名。
    payload = {  # 固化收益、去极值和标准化定义。
        "schema_version": 1,
        "universe_signature": universe_manifest["signature"],
        "eod": file_signature(paths.eod_path),
        "return": "adjusted_vwap.shift(-11) / adjusted_vwap.shift(-1) - 1",
        "winsorization": "median +/- 3 * 1.4826 * MAD",
        "zscore_ddof": 0,
    }
    signature = stable_hash(payload)  # 计算标签缓存签名。
    if not overwrite and manifest_path.exists() and all(path.exists() for path in outputs.values()):  # 产物齐全才允许复用。
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))  # 读取旧标签清单。
        if existing.get("signature") == signature:  # 口径和输入未变化时跳过重算。
            return existing  # 返回旧统计信息。

    total = 0  # 累计所有年度有效标签数。
    for universe_path, year in zip(universe_paths, years):  # Universe 文件与年度一一配对处理。
        # 只有 Universe 入选样本生成标签，未入选行仍保留在 Universe 产物中。
        universe = pd.read_parquet(universe_path)  # 读取含完整审计字段的年度 Universe。
        candidates = universe.loc[universe["is_eligible"]].copy()  # 仅为可交易样本计算标签。
        if candidates.empty:  # 空年度仍需生成合法空产物。
            atomic_parquet(candidates, outputs[year])  # 保持年度文件集合完整。
            continue  # 跳过无意义的行情读取。
        start_date = str(candidates["trade_date"].min())  # 行情读取从首个候选 T 日开始。
        end_date = _price_read_end(str(candidates["trade_date"].max()), calendars)  # 年末额外覆盖 T+11。
        prices = pd.read_parquet(  # 仅读取收益计算需要的价格字段和日期区间。
            paths.eod_path,
            columns=["s_info_windcode", "trade_dt", "s_dq_avgprice", "s_dq_adjfactor"],
            filters=[("trade_dt", ">=", start_date), ("trade_dt", "<=", end_date)],
        )
        returns = calculate_forward_vwap_returns(prices)  # 按股票位移得到所有 T 日未来收益。
        # 以股票和 T 日一对一回连，避免依赖 T+1/T+11 的外部日期映射。
        labels = candidates[["trade_date", "instrument_id", "wind_code", "stock_index"]].merge(  # 将收益精确回连图片样本。
            returns, on=["wind_code", "trade_date"], how="left", validate="one_to_one"
        )
        normalized = []  # 收集各交易日独立标准化结果。
        for _, indices in labels.groupby("trade_date", sort=True).groups.items():  # 按日处理横截面，禁止跨日统计。
            # MAD 和 z-score 都按每日横截面独立计算，杜绝跨日信息混用。
            normalized.append(mad_winsorize_zscore(labels.loc[indices, "forward_vwap_return_10d"]))  # 执行 MAD 去极值和 z-score。
        normalized_frame = pd.concat(normalized).sort_index()  # 按原始行索引恢复全年顺序。
        labels[normalized_frame.columns] = normalized_frame  # 写回标准化标签及审计边界。
        labels = labels.loc[np.isfinite(labels["target_zscore"])].copy()  # 删除无法形成有效横截面标签的行。
        float_columns = [
            "entry_adjusted_vwap", "exit_adjusted_vwap", "forward_vwap_return_10d",
            "winsorized_forward_return", "target_zscore", "mad_lower", "mad_upper",
        ]
        labels[float_columns] = labels[float_columns].astype(np.float32)  # 统一精度并缩小标签文件体积。
        if labels.duplicated(["trade_date", "instrument_id"]).any():  # 标签主键必须与 Universe 一致唯一。
            raise ValueError(f"duplicate label key in {year}")  # 阻止重复标签进入训练。
        atomic_parquet(labels, outputs[year])  # 原子保存年度有效标签。
        total += len(labels)  # 累计有效样本数。
        print(f"labels {year}: {len(labels):,} valid")  # 输出年度标签进度。

    result = {"signature": signature, "payload": payload, "years": years, "label_rows": total}
    atomic_json(result, manifest_path)  # 标签文件全部成功后发布清单。
    return result  # 返回缓存签名和有效标签数。


def normalize_pixels(pixel_batch: np.ndarray) -> np.ndarray:
    # 每只股票、每个通道在 8×8 平面内独立做 min-max；常数通道置零。
    values = pixel_batch.astype(np.float32, copy=False)  # 模型输入统一为单精度浮点。
    minimum = values.min(axis=(-2, -1), keepdims=True)  # 每只股票每通道计算空间最小值。
    value_range = values.max(axis=(-2, -1), keepdims=True) - minimum  # 计算对应 8×8 极差。
    normalized = values - minimum  # 将各通道平移到零起点。
    np.divide(normalized, value_range, out=normalized, where=value_range > 0)  # 仅对非恒定通道缩放。
    return np.ascontiguousarray(np.where(value_range > 0, normalized, 0.0), dtype=np.float32)  # 常数通道置零并整理连续内存。


def build_feature_store(paths: PipelinePaths, overwrite: bool = False) -> dict[str, object]:
    universe_paths = sorted(paths.universe_dir.glob("universe.*.parquet"))  # 收集所有年度可交易样本来源。
    if not universe_paths or not (paths.labels_dir / "manifest.json").exists():  # 特征库要求 Universe 与标签均已完成。
        raise FileNotFoundError("run universe and labels first")  # 阻止跳过上游依赖。
    frames = []  # 暂存各年度统一结构的样本表。
    for universe_path in universe_paths:  # 逐年连接标签以限制峰值内存。
        year = universe_path.stem.split(".")[-1]  # 提取当前年度。
        universe = pd.read_parquet(universe_path)  # 读取股票池和图片行号。
        selected = universe.loc[universe["is_eligible"]].copy()  # 特征库只保留可预测 Universe。
        label_path = paths.labels_dir / f"labels.{year}.parquet"  # 定位同年度标签文件。
        labels = pd.read_parquet(label_path, columns=[  # 仅读取训练和防泄漏所需标签字段。
            "trade_date", "instrument_id", "stock_index", "label_end_date", "target_zscore",
        ])
        # 左连接保留可预测但暂时无标签的样本，供对应 OOS 年生成因子。
        selected = selected.merge(  # 左连接确保 OOS 无标签样本仍保留。
            labels, on=["trade_date", "instrument_id", "stock_index"],
            how="left", validate="one_to_one",
        )
        selected["label_valid"] = np.isfinite(selected["target_zscore"])  # 显式标记能否用于训练和验证。
        frames.append(selected)  # 加入全量样本集合。
    samples = pd.concat(frames, ignore_index=True).sort_values(  # 合并后按日期和图片行号固定顺序。
        ["trade_date", "stock_index"], kind="stable"
    ).reset_index(drop=True)
    if samples.empty or samples.duplicated(["trade_date", "instrument_id"]).any():  # 特征主键必须非空且唯一。
        raise ValueError("invalid feature-store sample keys")  # 防止数组位置与元数据产生歧义。
    feature_dates = np.sort(samples["trade_date"].astype(str).unique())  # 建立全局有序特征日期轴。
    # date_id 是连续交易日位置，历史窗口据此判断是否真正连续。
    samples["date_id"] = samples["trade_date"].map(  # 将日期映射为连续整数以构造精确历史。
        {date: index for index, date in enumerate(feature_dates)}
    ).astype(np.int32)
    universe_manifest = json.loads((paths.universe_dir / "manifest.json").read_text(encoding="utf-8"))  # 读取股票池签名。
    label_manifest = json.loads((paths.labels_dir / "manifest.json").read_text(encoding="utf-8"))  # 读取标签签名。
    manifest_path = paths.feature_dir / "manifest.json"  # 特征库缓存清单路径。
    payload = {  # 特征签名绑定两个上游产物和归一化口径。
        "schema_version": 1,
        "universe_signature": universe_manifest["signature"],
        "label_signature": label_manifest["signature"],
        "sample_rule": "is_eligible",
        "normalization": "per-stock per-channel 8x8 min-max",
        "sample_count": len(samples),
    }
    signature = stable_hash(payload)  # 计算特征库版本签名。
    required = (
        paths.features_path, paths.targets_path, paths.label_valid_path,
        paths.date_ids_path, paths.sample_index_path,
    )
    if not overwrite and manifest_path.exists() and all(path.exists() for path in required):  # 数组和索引齐全时才尝试缓存。
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))  # 读取旧特征清单。
        if existing.get("signature") == signature:  # 上游数据与样本数均未改变。
            return existing  # 跳过耗时图片归一化。

    paths.feature_dir.mkdir(parents=True, exist_ok=True)  # 创建 memmap 和索引目录。
    temporary_paths = {  # 为四个数组生成进程唯一临时路径。
        path: path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
        for path in required[:4]
    }
    arrays = {
        # 使用 .npy memmap 顺序写入，避免把全量图片一次装入内存。
        paths.features_path: np.lib.format.open_memmap(
            temporary_paths[paths.features_path], "w+", np.float32,
            shape=(len(samples), 17, 8, 8),
        ),
        paths.targets_path: np.lib.format.open_memmap(
            temporary_paths[paths.targets_path], "w+", np.float32, shape=(len(samples),),
        ),
        paths.label_valid_path: np.lib.format.open_memmap(
            temporary_paths[paths.label_valid_path], "w+", np.bool_, shape=(len(samples),),
        ),
        paths.date_ids_path: np.lib.format.open_memmap(
            temporary_paths[paths.date_ids_path], "w+", np.int32, shape=(len(samples),),
        ),
    }
    arrays[paths.targets_path][:] = np.nan  # 无标签 OOS 样本默认保留 NaN。
    cursor = 0  # 指向下一个待写全局样本位置。
    index_frames = []  # 收集与数组位置一致的元数据。
    for trade_date in feature_dates:  # 按日期顺序流式读取图片。
        daily = samples.loc[samples["trade_date"].eq(trade_date)].sort_values("stock_index", kind="stable")  # 恢复 NPZ 内股票顺序。
        stock_indices = daily["stock_index"].to_numpy(dtype=np.int64)  # 提取需要读取的图片行号。
        with np.load(paths.level2_dir / f"stock_pixels_cn.{trade_date}.npz", allow_pickle=False) as archive:  # 打开当前日图片归档。
            # stock_index 和 instrument_id 双重校验，防止像素错配到其他股票。
            image_ids = archive["instrument_ids"][stock_indices].astype(str)  # 读取这些行实际对应的股票。
            if not np.array_equal(image_ids, daily["instrument_id"].to_numpy(dtype=str)):  # Universe 顺序必须与图片完全一致。
                raise ValueError(f"image/universe row mismatch on {trade_date}")  # 拒绝任何像素错配。
            if not archive["valid_mask"][stock_indices].astype(bool).all():  # 入库样本必须全部通过 T 日图片有效性。
                raise ValueError(f"invalid image entered feature store on {trade_date}")  # 防止筛选逻辑失效。
            pixels = normalize_pixels(archive["pixel_matrices"][stock_indices])  # 读取并逐股票逐通道归一化。
        stop = cursor + len(daily)  # 计算当前日期在全局数组中的右边界。
        # 四个数组使用完全相同的样本顺序，sample_index 即其共同位置。
        arrays[paths.features_path][cursor:stop] = pixels  # 写入与样本位置对应的视频帧。
        arrays[paths.targets_path][cursor:stop] = daily["target_zscore"].to_numpy(np.float32)  # 写入标签或 NaN。
        arrays[paths.label_valid_path][cursor:stop] = daily["label_valid"].to_numpy(bool)  # 写入训练可用掩码。
        arrays[paths.date_ids_path][cursor:stop] = daily["date_id"].to_numpy(np.int32)  # 写入连续日期编号。
        index = daily[[  # 选取因子回连和数据审计所需元数据。
            "trade_date", "instrument_id", "wind_code", "stock_index", "date_id",
            "t1_trade_date", "label_end_date", "label_valid",
        ]].copy()
        index.insert(0, "sample_index", np.arange(cursor, stop, dtype=np.int64))  # 将 memmap 行号作为全局样本主键。
        index_frames.append(index)  # 暂存当前日索引。
        cursor = stop  # 推进下一日写入起点。
    for array in arrays.values():  # 确保四个 memmap 均写回磁盘。
        # flush 后释放映射，再替换正式文件，保证其他进程只看到完整版本。
        array.flush()  # 强制提交尚在操作系统缓存中的数据。
    arrays.clear()  # 释放映射句柄后再替换文件。
    for final_path, temporary_path in temporary_paths.items():  # 逐个发布完成的数组。
        temporary_path.replace(final_path)  # 原子替换正式 memmap。
    atomic_parquet(pd.concat(index_frames, ignore_index=True), paths.sample_index_path)  # 保存同顺序样本索引。
    result = {"signature": signature, "payload": payload, "sample_count": len(samples)}
    atomic_json(result, manifest_path)  # 数组与索引全部成功后发布清单。
    return result  # 返回特征签名和样本数。


def build_history_indices(sample_index: pd.DataFrame, output_path: Path) -> None:
    sample_numbers = sample_index["sample_index"].to_numpy(dtype=np.int64)  # 全局样本号同时是特征数组行号。
    if not np.array_equal(sample_numbers, np.arange(len(sample_index), dtype=np.int64)):  # 历史矩阵要求样本号从零连续。
        raise ValueError("sample_index must be contiguous")  # 防止用稀疏编号直接索引 memmap。
    history = np.full((len(sample_index), SEQUENCE_LENGTH), -1, dtype=np.int32)  # -1 显式表示缺失历史帧。
    # 序列最后一格永远是 T 日自身；缺失历史保持 -1，不向前填充。
    history[:, -1] = sample_numbers.astype(np.int32)  # 每条序列末端固定指向自身 T 日。
    date_ids = sample_index["date_id"].to_numpy(dtype=np.int32)  # 取连续日期轴位置用于精确回溯。
    for positions in sample_index.groupby("instrument_id", sort=False).indices.values():  # 每只股票独立构建历史。
        rows = np.asarray(positions, dtype=np.int64)  # 转为可用于 NumPy 高级索引的位置。
        group_dates = date_ids[rows]  # 取得该股票实际有图的日期位置。
        group_samples = sample_numbers[rows]  # 取得对应全局特征行号。
        if len(group_dates) > 1 and np.any(np.diff(group_dates) <= 0):  # 股票内日期必须严格递增且不重复。
            raise ValueError("instrument dates must be strictly increasing")  # 避免 searchsorted 返回错误历史。
        for lag in range(1, SEQUENCE_LENGTH):  # 依次寻找 T-1 至 T-19。
            # 只有 date_id 恰好等于 T-lag 才写入，停牌或缺图形成显式空槽。
            desired = group_dates - lag  # 计算每个样本期望的精确历史 date_id。
            locations = np.searchsorted(group_dates, desired)  # 二分定位该股票可能的历史位置。
            in_bounds = locations < len(group_dates)  # 排除落在数组右侧的查找结果。
            matches = np.zeros(len(group_dates), dtype=bool)  # 默认所有历史槽位缺失。
            matches[in_bounds] = group_dates[locations[in_bounds]] == desired[in_bounds]  # 只接受日期恰好相等的记录。
            history[group_samples[matches], SEQUENCE_LENGTH - 1 - lag] = (
                group_samples[locations[matches]].astype(np.int32)
            )
    atomic_numpy(history, output_path)  # 原子保存固定 20 槽历史索引矩阵。


def build_rolling_splits(paths: PipelinePaths) -> pd.DataFrame:
    sample_index = pd.read_parquet(paths.sample_index_path)  # 读取全量样本键、日期和标签有效性。
    sample_index["trade_date"] = sample_index["trade_date"].astype(str)  # 固化字符串比较语义。
    sample_index["label_end_date"] = sample_index["label_end_date"].astype("string")  # 统一标签结束日类型。
    labeled = sample_index.loc[sample_index["label_valid"]].copy()  # 训练和验证只使用有效标签。
    date_max_end = labeled.groupby("trade_date", sort=True)["label_end_date"].max()  # 每日取最晚标签结束日作为防泄漏边界。
    records = []  # 收集各预测年的切分审计摘要。
    for forecast_year in FORECAST_YEARS:  # 为每个 OOS 年建立独立滚动窗口。
        forecast_start = f"{forecast_year}0101"  # 用年度首日作为预测边界。
        # 先剔除标签结束日越过预测年起点的日期，阻断未来收益泄漏。
        eligible_dates = date_max_end.loc[  # 只保留 T 日和标签结束日均早于预测年的日期。
            (date_max_end.index < forecast_start) & date_max_end.lt(forecast_start)
        ].index.to_numpy(dtype=str)
        if len(eligible_dates) < 2:  # 至少需要一个训练日和一个验证日。
            raise ValueError(f"insufficient history for {forecast_year}")  # 明确报告无法切分的预测年。
        split_position = min(max(int(np.floor(len(eligible_dates) * 0.8)), 1), len(eligible_dates) - 1)  # 按日期做 80/20 且保证两侧非空。
        raw_train_dates = eligible_dates[:split_position]  # 取较早 80% 日期作为初始训练段。
        validation_dates = eligible_dates[split_position:]  # 取较晚 20% 日期作为验证段。
        validation_start = str(validation_dates[0])  # 首个验证日定义第二道防泄漏边界。
        # 再剔除训练标签越过验证起点的日期，形成训练/验证 purge。
        train_dates = date_max_end.loc[raw_train_dates]  # 查看初始训练日各自标签结束位置。
        train_dates = train_dates.loc[train_dates.lt(validation_start)].index.to_numpy(dtype=str)  # 剔除标签跨入验证期的日期。
        train = sample_index.loc[  # 展开为有效训练样本的全局索引。
            sample_index["label_valid"] & sample_index["trade_date"].isin(train_dates), "sample_index"
        ].to_numpy(np.int64)
        validation = sample_index.loc[  # 展开为有效验证样本的全局索引。
            sample_index["label_valid"] & sample_index["trade_date"].isin(validation_dates), "sample_index"
        ].to_numpy(np.int64)
        prediction = sample_index.loc[  # 预测集保留目标年所有可交易样本。
            # 预测集取目标年份全部 Universe 样本，不要求标签有效。
            sample_index["trade_date"].str.startswith(str(forecast_year)), "sample_index"
        ].to_numpy(np.int64)
        if len(train) == 0 or len(validation) == 0:  # purge 后两侧仍必须含样本。
            raise ValueError(f"empty train/validation split for {forecast_year}")  # 防止空 DataLoader 延迟报错。
        atomic_npz(  # 保存该预测年的三组全局索引。
            paths.split_dir / f"forecast_{forecast_year}.npz",
            train_indices=train, validation_indices=validation, prediction_indices=prediction,
        )
        records.append({  # 记录样本量和日期边界供人工审计。
            "forecast_year": forecast_year,
            "train_samples": len(train), "validation_samples": len(validation),
            "prediction_samples": len(prediction),
            "max_train_label_end": str(sample_index.iloc[train]["label_end_date"].max()),
            "validation_start": validation_start,
            "max_validation_label_end": str(sample_index.iloc[validation]["label_end_date"].max()),
            "forecast_start": forecast_start,
        })
    summary = pd.DataFrame(records)  # 汇总所有滚动窗口。
    if not (summary["max_train_label_end"] < summary["validation_start"]).all():  # 全局复核训练标签未跨验证边界。
        raise AssertionError("train labels cross validation boundary")  # 将潜在未来信息泄漏视为硬错误。
    if not (summary["max_validation_label_end"] < summary["forecast_start"]).all():  # 全局复核验证标签未跨预测边界。
        raise AssertionError("validation labels cross forecast boundary")  # 禁止 OOS 信息进入模型选择。
    atomic_parquet(summary, paths.feature_dir / "rolling_split_summary.parquet")  # 保存切分审计表。
    return summary  # 返回供 Notebook 展示。


def prepare_data(paths: PipelinePaths, overwrite: bool = False) -> dict[str, object]:
    # 依赖顺序固定：Universe → 标签 → 特征 → 历史索引 → 滚动切分。
    universe = build_universe(paths, overwrite)  # 第一步：按图片和可交易状态构建股票池。
    labels = build_labels(paths, overwrite)  # 第二步：计算未来 VWAP 收益并标准化。
    features = build_feature_store(paths, overwrite)  # 第三步：归一化图片并建立样本主索引。
    sample_index = pd.read_parquet(paths.sample_index_path)  # 读取刚生成的统一样本顺序。
    build_history_indices(sample_index, paths.history_path)  # 第四步：构造不填充的 20 日历史槽位。
    splits = build_rolling_splits(paths)  # 第五步：构造防泄漏滚动训练、验证和预测集。
    return {
        "universe_rows": universe["eligible_rows"],
        "label_rows": labels["label_rows"],
        "sample_count": features["sample_count"],
        "forecast_years": splits["forecast_year"].tolist(),
    }


def load_split(paths: PipelinePaths, forecast_year: int) -> dict[str, np.ndarray]:
    split_path = paths.split_dir / f"forecast_{forecast_year}.npz"  # 定位目标预测年切分文件。
    if not split_path.exists():  # 切分缺失说明数据准备未完成。
        raise FileNotFoundError(split_path)  # 报出调用方需要补建的具体文件。
    with np.load(split_path, allow_pickle=False) as archive:  # 安全读取纯数值索引归档。
        return {name: archive[name].copy() for name in archive.files}  # 复制数组以脱离关闭后的 NPZ 句柄。


class SampleIndexDataset(Dataset):
    def __init__(self, indices: np.ndarray) -> None:
        self.indices = np.asarray(indices, dtype=np.int64)  # 保存全局 sample_index 序列。

    def __len__(self) -> int:
        return len(self.indices)  # DataLoader 长度等于当前切分样本数。

    def __getitem__(self, position: int) -> int:
        return int(self.indices[position])  # 将局部位置映射回全局样本号。


class TradeDateBatchSampler(Sampler[list[int]]):
    """每个批次返回一个完整交易日横截面的数据集位置。"""

    def __init__(self, indices: np.ndarray, date_ids: np.ndarray, seed: int) -> None:
        self.indices = np.asarray(indices, dtype=np.int64)  # 保存日期有序的训练全局索引。
        selected_dates = np.asarray(date_ids[self.indices], dtype=np.int64)  # 查出每个训练样本所属日期。
        if not len(self.indices) or np.any(selected_dates[1:] < selected_dates[:-1]):  # 完整横截面分组依赖非空升序日期。
            raise ValueError("training indices must be non-empty and date-sorted")  # 在采样前阻止错误切分。
        boundaries = np.flatnonzero(selected_dates[1:] != selected_dates[:-1]) + 1  # 找到相邻交易日的切换位置。
        # starts/stops 是 Dataset 的局部位置，不是全局 sample_index。
        self.starts = np.r_[0, boundaries]  # 每个日期批次的局部起点。
        self.stops = np.r_[boundaries, len(self.indices)]  # 每个日期批次的右开终点。
        self.seed = int(seed)  # 基础种子保证日期打乱可复现。
        self.epoch = 0  # 初始轮次用于派生每轮随机序列。

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)  # 更新轮次以改变日期批次顺序。

    def __iter__(self) -> Iterator[list[int]]:
        # 每轮只打乱交易日顺序，日内股票仍保持固定顺序并完整入批。
        order = np.random.default_rng(self.seed + self.epoch).permutation(len(self.starts))  # 只随机排列日期编号。
        for position in order:  # 依次返回该轮交易日批次。
            yield np.arange(self.starts[position], self.stops[position]).tolist()  # 输出 Dataset 局部位置而非全局索引。

    def __len__(self) -> int:
        return len(self.starts)  # 每轮批次数等于训练日期数。


class VideoBatchCollator:
    def __init__(self, paths: PipelinePaths) -> None:
        self.paths = paths  # 保存 memmap 路径配置。
        self.features = self.targets = self.date_ids = self.history = None  # 延迟到 worker 内首次使用时打开。

    def _open(self) -> None:
        if self.features is None:  # 每个 DataLoader worker 仅初始化一次映射。
            # DataLoader worker 首次使用时再各自打开只读 memmap，避免序列化大数组。
            self.features = np.load(self.paths.features_path, mmap_mode="r")  # 映射图片特征而不复制全量数组。
            self.targets = np.load(self.paths.targets_path, mmap_mode="r")  # 映射标准化标签。
            self.date_ids = np.load(self.paths.date_ids_path, mmap_mode="r")  # 映射日期编号。
            self.history = np.load(self.paths.history_path, mmap_mode="r")  # 映射 20 日历史样本号。

    def __call__(self, sample_indices: list[int]) -> dict[str, torch.Tensor]:
        self._open()  # 确保当前 worker 已打开全部只读数组。
        indices = np.asarray(sample_indices, dtype=np.int64)  # 当前批次的全局样本号。
        sequence = np.array(self.history[indices], dtype=np.int64, copy=True)  # 复制历史行以安全处理缺失值。
        frame_mask = sequence >= 0  # -1 槽位转为模型可用的有效帧掩码。
        if not frame_mask[:, -1].all():  # Universe 样本的 T 日图片必须有效。
            raise AssertionError("target frame must be valid")  # 防止训练目标缺少当前帧。
        # 缺失历史帧以零像素占位，frame_mask 交给模型屏蔽对应 tubelet。
        frames = np.zeros((len(indices), SEQUENCE_LENGTH, 17, 8, 8), dtype=np.float32)  # 为缺失历史预留零帧。
        frames[frame_mask] = np.array(self.features[sequence[frame_mask]], dtype=np.float32, copy=True)  # 只读取真实历史帧。
        videos = np.ascontiguousarray(frames.transpose(0, 2, 1, 3, 4))  # 调整为模型要求的 B×C×T×H×W。
        return {
            "videos": torch.from_numpy(videos),
            "frame_mask": torch.from_numpy(frame_mask),
            "targets": torch.from_numpy(np.array(self.targets[indices], dtype=np.float32, copy=True)),
            "date_ids": torch.from_numpy(np.array(self.date_ids[indices], dtype=np.int32, copy=True)),
            "sample_indices": torch.from_numpy(indices),
        }


def make_data_loaders(
    paths: PipelinePaths,
    forecast_year: int,
    seed: int,
    workers: int = 8,
    inference_batch_size: int = 5000,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    split = load_split(paths, forecast_year)  # 读取当前预测年的训练、验证和预测索引。
    common: dict[str, object] = {  # 三个 DataLoader 共享 worker、内存和组批逻辑。
        "num_workers": workers, "pin_memory": pin_memory,
        "collate_fn": VideoBatchCollator(paths),
    }
    if workers:  # 多进程加载时开启 worker 复用。
        # worker 常驻并仅预取一个批次，降低大横截面视频占用的主存峰值。
        common.update({"persistent_workers": True, "prefetch_factor": 1})  # 限制预取以控制大批次内存。
    date_ids = np.load(paths.date_ids_path, mmap_mode="r")  # 只读映射日期编号供训练分组。
    train_sampler = TradeDateBatchSampler(split["train_indices"], date_ids, seed)  # 创建完整日截面采样器。
    # 训练按完整日期批处理；验证和预测按固定样本数顺序读取。
    generator = torch.Generator().manual_seed(seed)  # 固定 DataLoader 内部随机状态。
    train = DataLoader(  # 训练批次由日期采样器控制，不设置固定 batch_size。
        SampleIndexDataset(split["train_indices"]), batch_sampler=train_sampler,
        generator=generator, **common,
    )
    validation = DataLoader(  # 验证保持样本顺序并按大批次推理。
        SampleIndexDataset(split["validation_indices"]), batch_size=inference_batch_size,
        shuffle=False, drop_last=False, **common,
    )
    prediction = None  # 无目标年样本时明确返回空预测加载器。
    if len(split["prediction_indices"]):  # 仅为非空 OOS 集创建加载器。
        prediction = DataLoader(  # 预测按 sample_index 顺序读取以便结果回连。
            SampleIndexDataset(split["prediction_indices"]), batch_size=inference_batch_size,
            shuffle=False, drop_last=False, **common,
        )
    return train, validation, prediction  # 返回三个阶段共用同一数据契约的加载器。
