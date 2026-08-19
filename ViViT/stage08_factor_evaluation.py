"""ViViT 样本外因子评价；只读复用 ViT 评价引擎且不依赖 PyTorch。"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

VIVIT_DIR = Path(__file__).resolve().parent  # 定位vivit目录。
PROJECT_ROOT = VIVIT_DIR.parent  # 定位项目根目录。
VIT_DIR = PROJECT_ROOT / "ViT"  # 定位vit目录。
DEFAULT_VIVIT_DATA_ROOT = VIVIT_DIR / "data"  # 定位默认vivit数据根目录。
DEFAULT_REFERENCE_DATA_ROOT = VIT_DIR / "data"  # 定位默认reference数据根目录。
DEFAULT_RUN_NAME = "sigmoid_daybatch_rankic"  # 设置默认执行名称。
PREDICTION_PATTERN = "vivit_predictions.{year}.parquet"  # 声明逐年预测文件名模板。
FACTOR_COLUMN = "prediction_ensemble"  # 设置因子列。
PREDICTION_COLUMNS = (  # 集中声明预测列。
    "prediction_ensemble",
    "prediction_ensemble_unprojected",
    "prediction_ensemble_projected",
    "prediction_ensemble_raw",
    "prediction_ensemble_alpha",
    "prediction_ensemble_style",
    "prediction_ensemble_total_return",
)
EVALUATION_DIRECTORY_NAMES = {  # 集中声明evaluation目录名称。
    "prediction_ensemble": "factor_evaluation",  # 把 prediction_ensemble 映射为"factor_evaluation"。
    "prediction_ensemble_unprojected": "factor_evaluation_unprojected",  # 把 prediction_ensemble_unprojected 映射为"factor_evaluation_unprojected"。
    "prediction_ensemble_projected": "factor_evaluation_projected",  # 把 prediction_ensemble_projected 映射为"factor_evaluation_projected"。
    "prediction_ensemble_raw": "factor_evaluation_raw",  # 把 prediction_ensemble_raw 映射为"factor_evaluation_raw"。
    "prediction_ensemble_alpha": "factor_evaluation_alpha",  # 把 prediction_ensemble_alpha 映射为"factor_evaluation_alpha"。
    "prediction_ensemble_style": "factor_evaluation_style",  # 把 prediction_ensemble_style 映射为"factor_evaluation_style"。
    "prediction_ensemble_total_return": "factor_evaluation_total_return",  # 把 prediction_ensemble_total_return 映射为"factor_evaluation_total_return"。
}
EXPERIMENT_REPORT_NAMES = {  # 集中声明experiment报告名称。
    "prediction_ensemble": "EXPERIMENT_REPORT.md",  # 把 prediction_ensemble 映射为"EXPERIMENT_REPORT.md"。
    "prediction_ensemble_unprojected": "EXPERIMENT_REPORT_unprojected.md",  # 把 prediction_ensemble_unprojected 映射为"EXPERIMENT_REPORT_unprojected.md"。
    "prediction_ensemble_projected": "EXPERIMENT_REPORT_projected.md",  # 把 prediction_ensemble_projected 映射为"EXPERIMENT_REPORT_projected.md"。
    "prediction_ensemble_raw": "EXPERIMENT_REPORT_raw.md",  # 把 prediction_ensemble_raw 映射为"EXPERIMENT_REPORT_raw.md"。
    "prediction_ensemble_alpha": "EXPERIMENT_REPORT_alpha.md",  # 把 prediction_ensemble_alpha 映射为"EXPERIMENT_REPORT_alpha.md"。
    "prediction_ensemble_style": "EXPERIMENT_REPORT_style.md",  # 把 prediction_ensemble_style 映射为"EXPERIMENT_REPORT_style.md"。
    "prediction_ensemble_total_return": "EXPERIMENT_REPORT_total_return.md",  # 把 prediction_ensemble_total_return 映射为"EXPERIMENT_REPORT_total_return.md"。
}
PERIODS = (1, 5, 10)  # 设置periods。
QUANTILES = 5  # 设置quantiles。
MAX_LOSS = 0.001  # 设置max损失。
BACKTEST_MODES = (  # 设置backtestmodes。
    "mode1_t_with_t1_filter",
    "mode2_t_no_filter",
    "mode3_t1_no_filter",
)
DEFAULT_BACKTEST_MODE = "mode2_t_no_filter"  # 设置默认backtest模式。
GROUPING_POLICY = "T_full_cross_section_before_execution_filter"  # 设置groupingpolicy。
EQUAL_WEIGHT_MARKET_COLUMN = "Equal-weight market"  # 设置equal权重market列。
BENCHMARK_SCOPE = (  # 设置benchmarkscope。
    "feature_valid_and_in_universe_t_full_factor_pool_after_required_price_filter"
)
BACKTEST_MODE_CONFIGS = {  # 设置backtest模式configs。
    "mode1_t_with_t1_filter": {  # 记录mode1twitht1filter。
        "price_start_offset": 0,  # 记录价格起点偏移。
        "price_start": "T_no_shift",  # 把 price_start 映射为"T_no_shift"。
        "portfolio_scope": "preassigned_quantile_then_can_enter_t1",  # 把 portfolio_scope 映射为"preassigned_quantile_then_can_enter_t1"。
        "execution_mask": "can_enter_t1",  # 把 execution_mask 映射为"can_enter_t1"。
    },
    "mode2_t_no_filter": {  # 记录mode2tnofilter。
        "price_start_offset": 0,  # 记录价格起点偏移。
        "price_start": "T_no_shift",  # 把 price_start 映射为"T_no_shift"。
        "portfolio_scope": "full_t_cross_section_no_t1_filter",  # 把 portfolio_scope 映射为"full_t_cross_section_no_t1_filter"。
        "execution_mask": "none",  # 把 execution_mask 映射为"none"。
    },
    "mode3_t1_no_filter": {  # 记录mode3t1nofilter。
        "price_start_offset": 1,  # 记录价格起点偏移。
        "price_start": "T_plus_1",  # 把 price_start 映射为"T_plus_1"。
        "portfolio_scope": "full_t_cross_section_no_t1_filter",  # 把 portfolio_scope 映射为"full_t_cross_section_no_t1_filter"。
        "execution_mask": "none",  # 把 execution_mask 映射为"none"。
    },
}

if str(VIT_DIR) not in sys.path:  # 检查当前值是否属于允许集合。
    sys.path.insert(0, str(VIT_DIR))

from vit_stage01_foundation import CALENDAR_PATH, atomic_json, normalize_dates  # noqa: E402。


def validate_run_name(value: str) -> str:  # 校验 run 名称，避免路径穿越和非法目录字符。
    """校验 run 名称，避免路径穿越和非法目录字符。"""
    run_name = str(value).strip()  # 去除首尾空白并规范化运行名称。
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")  # 初始化allowed容器。
    if (  # 开始组合多项有效性或配置条件。
        not run_name
        or run_name in {".", ".."}
        or any(character not in allowed for character in run_name)
    ):
        raise ValueError("运行名含非法字符")  # 输入数据不满足业务或数值约束时立即终止。
    return run_name  # 返回运行名称。


def validate_prediction_column(value: str) -> str:  # 校验待评价预测列是否属于允许的信号口径。
    """校验待评价预测列是否属于允许的信号口径。"""
    prediction_column = str(value).strip()  # 去除首尾空白并规范化预测列。
    if prediction_column not in PREDICTION_COLUMNS:  # 检查当前值是否属于允许集合。
        raise ValueError(  # 输入数据不满足业务或数值约束时立即终止。
            f"未知预测列：{value!r}；可选 {PREDICTION_COLUMNS}"
        )
    return prediction_column  # 返回预测列。


def evaluation_directory_name(prediction_column: str = FACTOR_COLUMN) -> str:  # 根据预测列生成互不覆盖的评价目录名。
    """根据预测列生成互不覆盖的评价目录名。"""
    return EVALUATION_DIRECTORY_NAMES[  # 返回根据预测列生成互不覆盖的评价目录名。
        validate_prediction_column(prediction_column)
    ]


def experiment_report_name(prediction_column: str = FACTOR_COLUMN) -> str:  # 根据预测列生成对应实验报告文件名。
    """根据预测列生成对应实验报告文件名。"""
    return EXPERIMENT_REPORT_NAMES[  # 返回根据预测列生成对应实验报告文件名。
        validate_prediction_column(prediction_column)
    ]


def resolve_backtest_mode(backtest_mode: str) -> dict[str, object]:  # 解析并校验统一样本外回测模式。
    """解析并校验统一样本外回测模式。"""
    mode = str(backtest_mode).strip()  # 去除首尾空白并规范化运行模式。
    if mode not in BACKTEST_MODE_CONFIGS:  # 检查当前值是否属于允许集合。
        raise ValueError(  # 输入数据不满足业务或数值约束时立即终止。
            f"未知回测模式：{backtest_mode!r}；可选 {BACKTEST_MODES}"
        )
    return {"name": mode, **BACKTEST_MODE_CONFIGS[mode]}  # 返回解析并校验统一样本外回测模式。


def read_json(path: Path) -> dict[str, object]:  # 读取 JSON 对象。
    """读取 JSON 对象；目标不存在时按接口约定返回空结果。"""
    if not path.exists():  # 检查目标文件或目录是否满足存在性要求。
        raise FileNotFoundError(path)  # 目标文件不存在时立即终止。
    return json.loads(path.read_text(encoding="utf-8"))  # 解析并返回 JSON 文件内容。


def _roots(  # 解析当前流程的数据、模型或质量目录。
    vivit_data_root: str | Path | None,  # 接收vivit数据根目录。
    reference_data_root: str | Path | None,  # 接收reference数据根目录。
) -> tuple[Path, Path]:  # 声明当前函数的返回类型。
    """解析当前流程的数据、模型或质量目录。"""
    vivit_root = Path(vivit_data_root or DEFAULT_VIVIT_DATA_ROOT).resolve()  # 构造vivit根目录。
    reference_root = Path(  # 构造reference根目录。
        reference_data_root or DEFAULT_REFERENCE_DATA_ROOT
    ).resolve()
    if not vivit_root.is_dir():  # 检查所需文件或目录是否存在。
        raise FileNotFoundError(vivit_root)  # 目标文件不存在时立即终止。
    if not reference_root.is_dir():  # 检查所需文件或目录是否存在。
        raise FileNotFoundError(reference_root)  # 目标文件不存在时立即终止。
    return vivit_root, reference_root  # 返回解析当前流程的数据、模型或质量目录。


def _manifests(  # 读取并校验上游与当前产物的 manifest 签名链。
    run_name: str, vivit_root: Path, reference_root: Path
) -> dict[str, dict[str, object]]:  # 声明当前函数的返回类型。
    """读取并校验上游与当前产物的 manifest 签名链。"""
    quality_run = vivit_root / "quality" / "runs" / run_name  # 拼接得到质量执行路径。
    result = {  # 初始化结果容器。
        "training": read_json(quality_run / "a2_training_manifest.json"),  # 记录训练。
        "sequence": read_json(  # 记录sequence。
            vivit_root / "aligned_sequence_store" / "sequence_manifest.json"
        ),
        "dataset": read_json(  # 记录dataset。
            reference_root / "feature_store" / "dataset_manifest.json"
        ),
        "rolling": read_json(  # 记录rolling。
            reference_root / "feature_store" / "rolling_split_manifest.json"
        ),
        "universe": read_json(  # 记录股票池。
            reference_root / "universe" / "universe_manifest.json"
        ),
        "labels": read_json(  # 记录标签。
            reference_root / "labels_t10" / "labels_manifest.json"
        ),
    }
    training = result["training"]  # 从上游结果中读取训练清单。
    sequence = result["sequence"]  # 从上游结果中读取序列清单。
    dataset = result["dataset"]  # 创建dataset。
    rolling = result["rolling"]  # 从上游结果中读取滚动切分清单。
    universe = result["universe"]  # 从上游结果中读取股票池清单。
    sequence_payload = sequence.get("payload", {})  # 读取sequence载荷配置，缺失时使用约定默认值。
    dataset_payload = dataset.get("payload", {})  # 创建dataset载荷。
    expected = {  # 初始化预期容器。
        "training source dataset": (  # 记录training source dataset。
            training.get("source_dataset_signature"), dataset.get("signature")
        ),
        "training source rolling": (  # 记录training source rolling。
            training.get("source_rolling_signature"), rolling.get("signature")
        ),
        "training aligned sequence": (  # 记录training aligned sequence。
            training.get("sequence_signature"), sequence.get("signature")
        ),
        "sequence source dataset": (  # 记录sequence source dataset。
            sequence_payload.get("source_dataset_signature"), dataset.get("signature")
        ),
        "sequence source rolling": (  # 记录sequence source rolling。
            sequence_payload.get("source_rolling_signature"), rolling.get("signature")
        ),
        "dataset Universe": (  # 记录dataset universe。
            dataset_payload.get("universe_signature"), universe.get("signature")
        ),
    }
    mismatches = [  # 初始化mismatches容器。
        f"{name}: {actual} != {wanted}"
        for name, (actual, wanted) in expected.items()  # 遍历预期items，逐项处理名称actualwanted。
        if actual != wanted  # 按当前模式、标识或状态选择处理分支。
    ]
    if mismatches:  # 根据当前数据状态选择处理分支。
        raise AssertionError("签名链不一致：" + "; ".join(mismatches))  # 关键数据契约被破坏时立即终止。
    return result  # 返回结果。


def _prediction_counts(training_manifest: dict[str, object]) -> dict[int, int]:  # 处理预测文件、预测列或对应的样本数量审计。
    """处理预测文件、预测列或对应的样本数量审计。"""
    years = tuple(int(year) for year in training_manifest.get("years", ()))  # 初始化年份容器。
    if not years:  # 检查必需数据、配置或校验结果是否缺失。
        raise ValueError("训练 manifest 缺少 years")  # 输入数据不满足业务或数值约束时立即终止。
    raw_counts = training_manifest.get("factor_counts")  # 计算原始数量。
    if not isinstance(raw_counts, dict):  # 检查对象类型是否符合接口约定。
        raise ValueError("训练 manifest 缺少 factor_counts")  # 输入数据不满足业务或数值约束时立即终止。
    counts = {  # 计算数量。
        year: int(raw_counts.get(str(year), raw_counts.get(year, -1)))
        for year in years  # 遍历年份，逐项处理年份。
    }
    if any(value < 0 for value in counts.values()):  # 检查当前值是否属于允许集合。
        raise ValueError("训练 manifest 的 factor_counts 不完整")  # 输入数据不满足业务或数值约束时立即终止。
    return counts  # 返回数量。


def _calendar_last_date(  # 返回指定交易所日历中不晚于给定日期的最后交易日。
    last_factor_date: str, backtest_mode: str = DEFAULT_BACKTEST_MODE  # 计算last因子交易日。
) -> str:  # 声明当前函数的返回类型。
    """返回指定交易所日历中不晚于给定日期的最后交易日。"""
    calendar = pd.read_parquet(  # 读取交易日历。
        CALENDAR_PATH,  # 汇入交易日历路径。
        columns=["trade_days", "s_info_exchmarket"],  # 限定本次读取或输出使用的字段。
        filters=[("s_info_exchmarket", "==", "SSE")],  # 下推数据筛选条件，减少无关数据读取。
    )
    dates = (  # 计算交易日。
        normalize_dates(calendar["trade_days"])
        .drop_duplicates()
        .sort_values()
        .to_numpy(dtype=str)
    )
    position = int(np.searchsorted(dates, last_factor_date))  # 定位位置。
    if position >= len(dates) or dates[position] != last_factor_date:  # 检查当前样本数量是否满足计算要求。
        raise AssertionError(f"{last_factor_date} 不在 SSE 交易日历")  # 关键数据契约被破坏时立即终止。
    mode = resolve_backtest_mode(backtest_mode)  # 解析运行模式对应的运行配置。
    final_offset = int(mode["price_start_offset"]) + max(PERIODS)  # 从上游结果中读取finaloffset。
    if position + final_offset >= len(dates):  # 检查当前样本数量是否满足计算要求。
        raise ValueError(  # 输入数据不满足业务或数值约束时立即终止。
            f"交易日历不足以覆盖最大 {final_offset} 个交易日收益期限"
        )
    return str(dates[position + final_offset])  # 返回指定交易所日历中不晚于给定日期的最后交易日。


def assign_factor_quantiles(  # 使用稳定排序在每个完整 T 日截面分配近似等量分位组。
    factor: pd.Series, quantiles: int = QUANTILES  # 计算因子序列。
) -> pd.Series:  # 声明当前函数的返回类型。
    """使用稳定排序在每个完整 T 日截面分配近似等量分位组。"""

    if quantiles < 2:  # 检查当前数值是否越过业务或数值阈值。
        raise ValueError("quantiles 必须至少为 2")  # 输入数据不满足业务或数值约束时立即终止。
    if not isinstance(factor.index, pd.MultiIndex) or "date" not in factor.index.names:  # 检查对象类型是否符合接口约定。
        raise ValueError("factor 必须使用含 date level 的 MultiIndex")  # 输入数据不满足业务或数值约束时立即终止。
    groups = factor.groupby(level="date", sort=False)  # 按业务键分组得到分组。
    ordinal_rank = groups.rank(method="first", ascending=True)  # 计算ordinal排名。
    group_size = groups.transform("size")  # 计算group大小。
    if (group_size < quantiles).any():  # 检查当前掩码或数据是否满足有效性要求。
        raise ValueError("存在样本数少于分位组数的日期")  # 输入数据不满足业务或数值约束时立即终止。
    result = pd.Series(  # 构造带业务索引的结果。
        np.floor(
            (ordinal_rank.to_numpy(np.float64) - 1)
            * quantiles
            / group_size.to_numpy(np.float64)
        ).astype(np.int64) + 1,
        index=factor.index,  # 控制是否把 DataFrame 索引写入结果。
        name="factor_quantile",  # 设置结果对象名称。
    )
    counts = result.groupby(  # 按业务键分组得到数量。
        [pd.Grouper(level="date"), result], observed=True
    ).size().unstack("factor_quantile")
    if counts.isna().any(axis=None) or (  # 根据当前数据状态选择处理分支。
        counts.max(axis=1) - counts.min(axis=1)
    ).gt(1).any():
        raise AssertionError("T 日完整截面分位组不均衡")  # 关键数据契约被破坏时立即终止。
    return result  # 返回结果。


def _normalized_vit_manifest(  # 读取并校验上游与当前产物的 manifest 签名链。
    manifests: dict[str, dict[str, object]],
) -> dict[str, object]:  # 声明当前函数的返回类型。
    """读取并校验上游与当前产物的 manifest 签名链。"""
    training = dict(manifests["training"])  # 初始化训练容器。
    training["dataset_signature"] = manifests["dataset"]["signature"]  # 创建训练。
    training["universe_signature"] = manifests["universe"]["signature"]  # 从上游结果中读取训练清单。
    return training  # 返回训练清单。


def _create_vit_evaluation_view(  # 在临时目录建立 ViT evaluator 所需的只读文件名视图。
    temporary_root: Path,  # 接收临时文件根目录。
    run_name: str,  # 接收执行名称。
    vivit_root: Path,  # 接收vivit根目录。
    reference_root: Path,  # 接收reference根目录。
    manifests: dict[str, dict[str, object]],
    prediction_column: str = FACTOR_COLUMN,  # 接收预测列。
) -> Path:  # 声明当前函数的返回类型。
    """在临时目录建立 ViT evaluator 所需的只读文件名视图。"""

    prediction_column = validate_prediction_column(prediction_column)  # 校验并规范化预测列。
    for name in ("feature_store", "universe", "labels_t10"):  # 遍历特征存储股票池标签t10，逐项处理名称。
        (temporary_root / name).symlink_to(
            reference_root / name, target_is_directory=True
        )
    source_predictions = vivit_root / "predictions" / "runs" / run_name  # 拼接得到来源预测路径。
    view_predictions = temporary_root / "predictions" / "runs" / run_name  # 拼接得到view预测路径。
    view_predictions.mkdir(parents=True)  # 确保目标目录存在。
    counts = _prediction_counts(manifests["training"])  # 计算数量。
    for year, count in counts.items():  # 遍历数量items，逐项处理年份数量。
        if count:  # 根据当前数据状态选择处理分支。
            source = source_predictions / PREDICTION_PATTERN.format(year=year)  # 拼接得到源文件路径。
            target = view_predictions / f"vit_predictions.{year}.parquet"  # 拼接得到目标路径路径。
            if prediction_column == FACTOR_COLUMN:  # 按当前模式、标识或状态选择处理分支。
                target.symlink_to(source)
            else:  # 处理上述条件均不成立的情况。
                prediction = pd.read_parquet(  # 读取预测。
                    source,  # 汇入来源。
                    columns=["trade_date", "instrument_id", prediction_column],  # 限定本次读取或输出使用的字段。
                ).rename(columns={prediction_column: FACTOR_COLUMN})
                prediction.to_parquet(target, index=False)  # 把表格按约定格式写入 Parquet。
                del prediction  # 释放不再使用的中间引用。

    view_quality_run = temporary_root / "quality" / "runs" / run_name  # 拼接得到view质量执行路径。
    view_quality_run.mkdir(parents=True)  # 确保目标目录存在。
    atomic_json(  # 原子落盘，避免半成品。
        _normalized_vit_manifest(manifests),
        view_quality_run / "a2_training_manifest.json",
    )
    evaluation_target = vivit_root / "quality" / "runs" / run_name / (  # 拼接得到evaluation标签路径。
        evaluation_directory_name(prediction_column)
    )
    evaluation_target.mkdir(parents=True, exist_ok=True)  # 确保目标目录存在。
    (view_quality_run / "factor_evaluation").symlink_to(  # 无梯度计算验证指标。
        evaluation_target, target_is_directory=True  # 计算当前中间结果。
    )
    return evaluation_target  # 返回evaluation标签。


def _markdown_table(frame: pd.DataFrame) -> str:  # 把结构化结果转换为 Markdown 表格。
    """把结构化结果转换为 Markdown 表格。"""
    if frame.empty:  # 检查结果表是否为空。
        return "（暂无记录）"  # 返回完成上述格式转换后的结果。
    clean = frame.copy().fillna("")  # 转换clean的数据类型或存储位置。
    headers = [str(column) for column in clean.columns]  # 初始化headers容器。
    lines = [  # 初始化lines容器。
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in clean.astype(str).itertuples(index=False, name=None):  # 遍历cleanastypestritertuples索引false名称none，逐项处理行。
        lines.append(  # 把当前结果追加到汇总列表。
            "| " + " | ".join(value.replace("|", "\\|") for value in row) + " |"  # 按规则替换或转义字符串内容。
        )
    return "\n".join(lines)  # 返回完成上述格式转换后的结果。


def _read_optional_parquet(path: Path) -> pd.DataFrame:  # 读取对应产物，并按接口约定处理缺失文件。
    """读取对应产物，并按接口约定处理缺失文件。"""
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()  # 返回读取并校验后的结果。


def generate_experiment_report(  # 生成同时包含 ViViT 训练与因子评价结果的 run 专属报告。
    run_name: str,  # 接收执行名称。
    vivit_data_root: str | Path | None = None,  # 接收vivit数据根目录。
    reference_data_root: str | Path | None = None,  # 接收reference数据根目录。
    prediction_column: str = FACTOR_COLUMN,  # 接收预测列。
) -> Path:  # 声明当前函数的返回类型。
    """生成同时包含 ViViT 训练与因子评价结果的 run 专属报告。"""

    run_name = validate_run_name(run_name)  # 校验并规范化运行名称。
    prediction_column = validate_prediction_column(prediction_column)  # 校验并规范化预测列。
    evaluation_name = evaluation_directory_name(prediction_column)  # 无梯度计算验证指标。
    vivit_root, reference_root = _roots(vivit_data_root, reference_data_root)  # 构造vivit根目录。
    manifests = _manifests(run_name, vivit_root, reference_root)  # 读取或生成来源清单。
    quality_run = vivit_root / "quality" / "runs" / run_name  # 拼接得到质量执行路径。
    evaluation_dir = quality_run / evaluation_name  # 构造evaluation目录。
    training = _read_optional_parquet(  # 按需读取训练清单，文件缺失时保留空结果。
        quality_run / "vivit_training_summary.parquet"
    )
    if training.empty:  # 检查结果表是否为空。
        training_yearly = pd.DataFrame()  # 把累计记录整理为训练yearly。
    else:  # 处理上述条件均不成立的情况。
        aggregations: dict[str, tuple[str, str]] = {  # 初始化aggregations容器。
            "completed_seeds": ("seed", "nunique"),  # 记录completed随机种子。
            "mean_best_validation_ic": ("best_validation_ic", "mean"),  # 记录均值最优验证ic。
            "min_best_validation_ic": ("best_validation_ic", "min"),  # 记录min最优验证ic。
            "mean_best_epoch": ("best_epoch", "mean"),  # 记录均值最优轮次。
        }
        for output, source in (  # 遍历(，逐项处理输出来源。
            ("mean_best_validation_rank_ic", "best_validation_rank_ic"),
            ("mean_best_validation_daily_pearson_ic", "best_validation_daily_pearson_ic"),
            ("mean_best_validation_pooled_ic", "best_validation_pooled_ic"),
        ):
            if source in training.columns:  # 检查当前值是否属于允许集合。
                aggregations[output] = (source, "mean")  # 计算aggregations。
        training_yearly = training.groupby(  # 按既定口径聚合。
            "forecast_year", as_index=False
        ).agg(**aggregations)

    sequence_summary = _read_optional_parquet(  # 按需读取sequence汇总，文件缺失时保留空结果。
        vivit_root / "aligned_sequence_store" / "rolling_split_summary.parquet"
    )
    final_audit = _read_optional_parquet(evaluation_dir / "final_audit.parquet")  # 按需读取finalaudit，文件缺失时保留空结果。
    ic_summary = _read_optional_parquet(evaluation_dir / "ic_summary.parquet")  # 按需读取ic汇总，文件缺失时保留空结果。
    spread_summary = _read_optional_parquet(evaluation_dir / "spread_summary.parquet")  # 按需读取spread汇总，文件缺失时保留空结果。
    pyfolio = _read_optional_parquet(evaluation_dir / "pyfolio_performance.parquet")  # 按需读取pyfolio，文件缺失时保留空结果。
    relative_performance = _read_optional_parquet(  # 按需读取relativeperformance，文件缺失时保留空结果。
        evaluation_dir / "relative_performance.parquet"
    )
    evaluation_manifest_path = evaluation_dir / "factor_evaluation_manifest.json"  # 构造evaluation清单路径。
    evaluation_manifest = (  # 计算evaluation清单。
        read_json(evaluation_manifest_path)
        if evaluation_manifest_path.exists()  # 检查目标文件或目录是否满足存在性要求。
        else None
    )
    if evaluation_manifest:  # 根据当前数据状态选择处理分支。
        start_offset = int(evaluation_manifest.get("price_start_offset", 0))  # 读取起点offset配置，缺失时使用约定默认值。
        period_description = (  # 计算perioddescription。
            "1/5/10D 对应 T→T+1/T+5/T+10"
            if start_offset == 0  # 按当前模式、标识或状态选择处理分支。
            else "1/5/10D 对应 T+1→T+2/T+6/T+11"
        )
        execution_description = (  # 计算executiondescription。
            "组合收益和换手应用 `can_enter_t1`"
            if evaluation_manifest.get("execution_mask") == "can_enter_t1"  # 按当前模式、标识或状态选择处理分支。
            else "组合收益和换手不应用 `can_enter_t1` 未来状态过滤"
        )
        evaluation_policy_lines = [  # 初始化evaluationpolicylines容器。
            f"- 回测模式：`{evaluation_manifest.get('backtest_mode')}`；{execution_description}。",
            f"- 收益公式：`{evaluation_manifest.get('forward_return_formula')}`；{period_description}。",
            "- 全市场等权基准使用完整因子覆盖池和相同收益时点，不应用 `can_enter_t1`。",
            "- Q5 与 Q5−Q1 均报告相对市场的超额年化、跟踪误差和信息比。",
        ]
    else:  # 处理上述条件均不成立的情况。
        evaluation_policy_lines = ["- 因子评价尚未按新版回测口径生成。"]  # 初始化evaluationpolicylines容器。
    training_manifest = manifests["training"]  # 从上游结果中读取训练清单。
    missing_batch_value = (  # 计算missing批次值。
        "full_trade_date"
        if training_manifest.get("train_batch_mode") == "trade_date"  # 按当前模式、标识或状态选择处理分支。
        else "not_recorded"
    )
    fixed_batch_size = training_manifest.get(  # 读取fixed批次大小配置，缺失时使用约定默认值。
        "fixed_batch_size", missing_batch_value
    )
    inference_batch_size = training_manifest.get(  # 读取inference批次大小配置，缺失时使用约定默认值。
        "inference_batch_size", missing_batch_value
    )
    derivation = training_manifest.get("factor_derivation")  # 读取derivation配置，缺失时使用约定默认值。
    if isinstance(derivation, dict):  # 检查对象类型是否符合接口约定。
        derivation_lines = [  # 初始化derivationlines容器。
            "## Barra 残差因子派生",
            "",
            f"- 源 run：`{training_manifest.get('source_run_name')}`",
            f"- 因子角色：`{training_manifest.get('factor_role')}`",
            f"- 派生签名：`{training_manifest.get('factor_derivation_signature')}`",
            f"- CNE5S 风格暴露：`{', '.join(derivation.get('style_columns', []))}`",
            f"- 回归连续变量：`{', '.join(derivation.get('regression_columns', derivation.get('style_columns', [])))}`",
            f"- 20 日反转：`{derivation.get('reversal', {}).get('formula')}`；缩放：`{derivation.get('reversal', {}).get('regression_scaling')}`。",
            f"- 行业字段：`{derivation.get('industry_column')}`；不使用 `cne5s_resvol`。",
            f"- 匹配规则：{derivation.get('match_policy')}",
            f"- 缺失处理：CNE5S=`{derivation.get('missing_barra_policy')}`；反转=`{derivation.get('missing_reversal_policy')}`。",
            f"- 残差尺度：{derivation.get('residual_scaling')}",
            "- 相关性、回归诊断和 lineage 位于 `barra_neutralization/`。",
            "",
        ]
    else:  # 处理上述条件均不成立的情况。
        derivation_lines = []  # 初始化derivationlines容器。
    lines = [  # 初始化lines容器。
        f"# ViViT 对齐训练与样本外因子评价｜{run_name}",
        "",
        f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')} UTC",
        "",
        "## 数据与模型口径",
        "",
        "- train、validation、prediction 直接使用 ViT 年度索引，不做每 5 日降采样或完整历史过滤。",
        "- 20 个历史槽位只匹配同股票精确日期；缺失帧补零，并以 frame mask 和 0/1/2 Tubelet 覆盖状态入模。",
        "- T 日完整预测截面先固定五分位；RankIC 始终使用完整价格有效截面。",
        *evaluation_policy_lines,
        f"- 本报告评价预测列：`{prediction_column}`。",
        "",
        "## 对齐历史覆盖",
        "",
        _markdown_table(sequence_summary),
        "",
        "## 训练配置",
        "",
        f"- loss / batch mode / validation metric：`{training_manifest['loss_name']}` / `{training_manifest['train_batch_mode']}` / `{training_manifest['validation_metric']}`",
        f"- fixed / inference batch：{fixed_batch_size} / {inference_batch_size}",
        f"- compile / mode / fused Adam / CUDA prefetch：{training_manifest['torch_compile']} / `{training_manifest.get('compile_mode', 'default')}` / {training_manifest['fused_adam']} / {training_manifest['cuda_prefetch']}",
        "",
        _markdown_table(training_yearly),
        "",
        "## 样本外因子评价",
        "",
        _markdown_table(final_audit),
        "",
        "### RankIC",
        "",
        _markdown_table(ic_summary),
        "",
        "### Q5−Q1 收益差",
        "",
        _markdown_table(spread_summary),
        "",
        "### Pyfolio 绝对绩效",
        "",
        _markdown_table(pyfolio),
        "",
        "### 相对全市场等权绩效",
        "",
        _markdown_table(relative_performance),
        "",
        "## 签名与产物",
        "",
        f"- Universe：`{manifests['universe']['signature']}`",
        f"- feature store：`{manifests['dataset']['signature']}`",
        f"- aligned sequence store：`{manifests['sequence']['signature']}`",
        f"- 累计收益图：`{evaluation_name}/portfolio_cumulative_returns.png`。",
        f"- 市场基准与相对绩效：`{evaluation_name}/portfolio_returns.parquet`、`relative_performance.parquet`。",
        f"- 最大回撤及 Alphalens returns/information/turnover tear sheets：`{evaluation_name}/pic/`。",
        "- 2025 prediction 与 ViT 一致为空，不生成或评价 2025 因子。",
        "- ViT 数据与评价产物保持只读。",
        "",
    ]
    if derivation_lines:  # 根据当前数据状态选择处理分支。
        insert_at = lines.index("## 对齐历史覆盖")
        lines[insert_at:insert_at] = derivation_lines  # 计算lines。
    path = quality_run / experiment_report_name(prediction_column)  # 拼接得到路径路径。
    path.parent.mkdir(parents=True, exist_ok=True)  # 确保目标目录存在。
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")  # 构造临时文件。
    temporary.write_text("\n".join(lines), encoding="utf-8")  # 把文本结果写入临时或目标文件。
    temporary.replace(path)  # 用临时文件原子替换正式结果。
    print(f"实验报告：{path}")  # 输出当前任务的进度或诊断信息。
    return path  # 返回路径。


def run_oos_backtest(  # 按指定共享口径执行 ViViT Alphalens/Pyfolio 评价。
    run_name: str = DEFAULT_RUN_NAME,  # 接收执行名称。
    vivit_data_root: str | Path | None = None,  # 接收vivit数据根目录。
    reference_data_root: str | Path | None = None,  # 接收reference数据根目录。
    backtest_mode: str = DEFAULT_BACKTEST_MODE,  # 接收backtest模式。
    prediction_column: str = FACTOR_COLUMN,  # 接收预测列。
) -> pd.DataFrame:  # 声明当前函数的返回类型。
    """按指定共享口径执行 ViViT Alphalens/Pyfolio 评价。"""

    run_name = validate_run_name(run_name)  # 校验并规范化运行名称。
    prediction_column = validate_prediction_column(prediction_column)  # 校验并规范化预测列。
    mode = resolve_backtest_mode(backtest_mode)  # 解析运行模式对应的运行配置。
    vivit_root, reference_root = _roots(vivit_data_root, reference_data_root)  # 构造vivit根目录。
    input_audit = audit_evaluation_inputs(  # 计算输入audit。
        run_name, vivit_root, reference_root, str(mode["name"]),
        prediction_column,  # 汇入预测列。
    )
    manifests = _manifests(run_name, vivit_root, reference_root)  # 读取或生成来源清单。
    try:  # 执行可能失败的外部读取或计算步骤。
        import vit_stage08_factor_evaluation as vit_evaluation
    except ModuleNotFoundError as error:  # 捕获预期异常并执行对应恢复逻辑。
        raise RuntimeError(  # 当前运行环境或任务状态不支持继续执行。
            "请使用 Jupyter 内核 `ViT Factor Eval (Alphalens/Pyfolio)`。"
        ) from error

    with tempfile.TemporaryDirectory(  # 进入受控上下文并在退出时自动释放资源。
        prefix=f"vivit_factor_eval_{run_name}_"  # 计算prefix。
    ) as directory:
        temporary_root = Path(directory)  # 构造临时文件根目录。
        evaluation_dir = _create_vit_evaluation_view(  # 构造evaluation目录。
            temporary_root, run_name, vivit_root, reference_root, manifests,  # 传入临时文件根目录、执行名称、vivit根目录、reference根目录、manifests。
            prediction_column,  # 汇入预测列。
        )
        original_report = vit_evaluation.generate_experiment_report  # 计算原始报告。

        def vivit_report_adapter(**_kwargs: object) -> Path:  # 把 ViT 报告回调重定向到当前 ViViT run 和预测列。
            """把 ViT 报告回调重定向到当前 ViViT run 和预测列。"""
            return generate_experiment_report(  # 返回完成上述格式转换后的结果。
                run_name, vivit_root, reference_root, prediction_column  # 传入执行名称、vivit根目录、reference根目录、预测列。
            )

        vit_evaluation.generate_experiment_report = vivit_report_adapter
        try:  # 执行可能失败的外部读取或计算步骤。
            result = vit_evaluation.run_oos_backtest(  # 无梯度计算验证指标。
                run_name=run_name,  # 配置执行名称。
                data_root=temporary_root,  # 配置数据根目录。
                backtest_mode=str(mode["name"]),  # 配置backtest模式。
            )
        finally:  # 无论前序步骤是否成功都执行资源清理。
            vit_evaluation.generate_experiment_report = original_report

    manifest_path = evaluation_dir / "factor_evaluation_manifest.json"  # 构造清单路径。
    evaluation_manifest = read_json(manifest_path)  # 读取evaluation清单。
    training_manifest = manifests["training"]  # 从上游结果中读取训练清单。
    evaluation_manifest.update(  # 用当前配置更新参数字典。
        {
            "model_family": "ViViT",  # 把 model_family 映射为"ViViT"。
            "factor_role": training_manifest.get(  # 记录因子role。
                "factor_role", "model_prediction"
            ),
            "source_run_name": training_manifest.get("source_run_name"),  # 记录来源运行名称。
            "factor_derivation_signature": training_manifest.get(  # 记录因子derivation签名，用于来源一致性校验。
                "factor_derivation_signature"
            ),
            "prediction_pattern": PREDICTION_PATTERN,  # 记录预测pattern。
            "prediction_column": prediction_column,  # 记录预测字段。
            "dataset_signature": manifests["dataset"]["signature"],  # 记录dataset签名，用于来源一致性校验。
            "universe_signature": manifests["universe"]["signature"],  # 记录股票池签名，用于来源一致性校验。
            "sequence_signature": manifests["sequence"]["signature"],  # 记录sequence签名，用于来源一致性校验。
            "input_audit_rows": int(input_audit["prediction_rows"].sum()),  # 记录输入audit行数。
            "quantiles": QUANTILES,  # 记录quantiles。
            "max_loss": MAX_LOSS,  # 记录max损失。
        }
    )
    atomic_json(evaluation_manifest, manifest_path)  # 原子落盘，避免半成品。
    generate_experiment_report(
        run_name, vivit_root, reference_root, prediction_column  # 传入执行名称、vivit根目录、reference根目录、预测列。
    )
    return result  # 返回结果。


def compare_factor_evaluations(  # 比较两个已完成评价的 ViViT run。
    run_name: str,  # 接收执行名称。
    baseline_run_name: str | None,  # 接收baseline执行名称。
    vivit_data_root: str | Path | None = None,  # 接收vivit数据根目录。
    prediction_column: str = FACTOR_COLUMN,  # 接收预测列。
    baseline_prediction_column: str = FACTOR_COLUMN,  # 接收baseline预测列。
) -> dict[str, pd.DataFrame]:  # 声明当前函数的返回类型。
    """比较两个已完成评价的 ViViT run；基准为空时返回空字典。"""

    run_name = validate_run_name(run_name)  # 校验并规范化运行名称。
    prediction_column = validate_prediction_column(prediction_column)  # 校验并规范化预测列。
    if baseline_run_name is None:  # 检查所需对象是否尚未初始化。
        return {}  # 返回比较两个已完成评价的 ViViT run。
    baseline_run_name = validate_run_name(baseline_run_name)  # 校验并规范化baseline执行名称。
    baseline_prediction_column = validate_prediction_column(  # 校验并规范化baseline预测列。
        baseline_prediction_column  # 汇入baseline预测列。
    )
    if (  # 开始组合多项有效性或配置条件。
        baseline_run_name == run_name  # 配置baseline执行名称。
        and baseline_prediction_column == prediction_column
    ):
        raise ValueError("基准 run 与预测列不能与待评价 run 相同")  # 输入数据不满足业务或数值约束时立即终止。
    vivit_root = Path(  # 构造vivit根目录。
        vivit_data_root or DEFAULT_VIVIT_DATA_ROOT
    ).resolve()
    if not vivit_root.is_dir():  # 检查所需文件或目录是否存在。
        raise FileNotFoundError(vivit_root)  # 目标文件不存在时立即终止。
    run_manifest = _validated_evaluation_manifest(  # 计算执行清单。
        run_name, vivit_root, prediction_column  # 传入执行名称、vivit根目录、预测列。
    )
    baseline_manifest = _validated_evaluation_manifest(  # 计算baseline清单。
        baseline_run_name, vivit_root,  # 传入baseline执行名称、vivit根目录。
        baseline_prediction_column,  # 汇入baseline预测列。
    )
    for key in (  # 遍历(，逐项处理键。
        "dataset_signature", "universe_signature", "sequence_signature",
        "grouping_policy", "periods", "quantiles", "backtest_mode",
        "price_start", "price_start_offset", "forward_return_formula",
        "portfolio_scope", "execution_mask", "benchmark_column",
        "benchmark_scope", "benchmark_weighting", "benchmark_execution_mask",
        "annual_excess_method", "information_ratio_method",
        "annual_tracking_error_method", "annualization_days",
        "long_short_gross_exposure",
    ):
        if run_manifest.get(key) != baseline_manifest.get(key):  # 按当前模式、标识或状态选择处理分支。
            raise RuntimeError(f"两个 run 的 {key} 不一致，禁止直接比较")  # 当前运行环境或任务状态不支持继续执行。

    def read_table(run: str, column: str, file_name: str) -> pd.DataFrame:  # 读取对应产物，并按接口约定处理缺失文件。
        """读取对应产物，并按接口约定处理缺失文件。"""
        path = (  # 计算路径。
            vivit_root / "quality" / "runs" / run
            / evaluation_directory_name(column) / file_name
        )
        if not path.exists():  # 检查目标文件或目录是否满足存在性要求。
            raise FileNotFoundError(path)  # 目标文件不存在时立即终止。
        frame = pd.read_parquet(path).copy()  # 读取帧。
        frame.insert(0, "run_name", run)
        frame.insert(1, "prediction_column", column)
        return frame  # 返回帧。

    runs = (  # 计算runs。
        (baseline_run_name, baseline_prediction_column),
        (run_name, prediction_column),
    )
    return {  # 返回比较两个已完成评价的 ViViT run。
        "ic": pd.concat(  # 记录ic。
            [read_table(name, column, "ic_summary.parquet") for name, column in runs],
            ignore_index=True,  # 拼接后重建连续行索引。
        ),
        "spread": pd.concat(  # 记录spread。
            [read_table(name, column, "spread_summary.parquet") for name, column in runs],
            ignore_index=True,  # 拼接后重建连续行索引。
        ),
        "pyfolio": pd.concat(  # 记录pyfolio。
            [read_table(name, column, "pyfolio_performance.parquet") for name, column in runs],
            ignore_index=True,  # 拼接后重建连续行索引。
        ),
        "relative": pd.concat(  # 记录relative。
            [read_table(name, column, "relative_performance.parquet") for name, column in runs],
            ignore_index=True,  # 拼接后重建连续行索引。
        ),
    }


def compare_prediction_signal_pearson(  # 按完整业务键配对两个 run，并计算逐日横截面 Pearson。
    run_name: str,  # 接收执行名称。
    baseline_run_name: str | None,  # 接收baseline执行名称。
    vivit_data_root: str | Path | None = None,  # 接收vivit数据根目录。
    prediction_column: str = FACTOR_COLUMN,  # 接收预测列。
    baseline_prediction_column: str = FACTOR_COLUMN,  # 接收baseline预测列。
) -> dict[str, pd.DataFrame]:  # 声明当前函数的返回类型。
    """按完整业务键配对两个 run，并计算逐日横截面 Pearson。"""

    run_name = validate_run_name(run_name)
    prediction_column = validate_prediction_column(prediction_column)
    if baseline_run_name is None:  # 检查所需对象是否尚未初始化。
        return {}  # 返回按当前规则组织后的结果。
    baseline_run_name = validate_run_name(baseline_run_name)
    baseline_prediction_column = validate_prediction_column(
        baseline_prediction_column  # 汇入baseline预测列。
    )
    if baseline_run_name == run_name and baseline_prediction_column == prediction_column:  # 按当前模式、标识或状态选择处理分支。
        raise ValueError("基准 run 与预测列不能与待评价 run 相同")  # 输入数据不满足业务或数值约束时立即终止。
    vivit_root = Path(  # 构造vivit根目录。
        vivit_data_root or DEFAULT_VIVIT_DATA_ROOT
    ).resolve()
    if not vivit_root.is_dir():  # 检查所需文件或目录是否存在。
        raise FileNotFoundError(vivit_root)  # 目标文件不存在时立即终止。

    def training_manifest(name: str) -> dict[str, object]:  # 读取并校验上游与当前产物的 manifest 签名链。
        """读取并校验上游与当前产物的 manifest 签名链。"""
        return read_json(  # 返回读取并校验后的结果。
            vivit_root / "quality" / "runs" / name
            / "a2_training_manifest.json"
        )

    run_manifest = training_manifest(run_name)  # 读取或生成来源清单。
    baseline_manifest = training_manifest(baseline_run_name)  # 读取或生成来源清单。
    for key in (  # 遍历(，逐项处理键。
        "source_dataset_signature",
        "source_rolling_signature",
        "sequence_signature",
    ):
        if run_manifest.get(key) != baseline_manifest.get(key):  # 按当前模式、标识或状态选择处理分支。
            raise RuntimeError(f"两个 run 的 {key} 不一致，禁止直接比较")  # 当前运行环境或任务状态不支持继续执行。
    run_counts = _prediction_counts(run_manifest)  # 计算执行数量。
    baseline_counts = _prediction_counts(baseline_manifest)  # 计算baseline数量。
    if run_counts != baseline_counts:  # 按当前模式、标识或状态选择处理分支。
        raise RuntimeError("两个 run 的预测年度或行数不一致，禁止直接比较")  # 当前运行环境或任务状态不支持继续执行。

    keys = ["trade_date", "instrument_id"]  # 初始化键容器。

    def read_prediction(  # 读取对应产物，并按接口约定处理缺失文件。
        name: str, year: int, expected_count: int, column: str
    ) -> pd.DataFrame:  # 声明当前函数的返回类型。
        """读取对应产物，并按接口约定处理缺失文件。"""
        path = (
            vivit_root / "predictions" / "runs" / name
            / PREDICTION_PATTERN.format(year=year)
        )
        if not path.exists():  # 检查目标文件或目录是否满足存在性要求。
            raise FileNotFoundError(path)  # 目标文件不存在时立即终止。
        frame = pd.read_parquet(path, columns=[*keys, column])  # 读取帧。
        if len(frame) != expected_count:  # 检查当前样本数量是否满足计算要求。
            raise AssertionError(  # 关键数据契约被破坏时立即终止。
                f"{name} {year} 预测行数 {len(frame):,} != {expected_count:,}"
            )
        frame["trade_date"] = frame["trade_date"].astype(str)  # 转换帧的数据类型或存储位置。
        frame["instrument_id"] = frame["instrument_id"].astype(str)  # 转换帧的数据类型或存储位置。
        if frame.duplicated(keys).any():  # 检查当前掩码或数据是否满足有效性要求。
            raise ValueError(f"{name} {year} prediction 业务键重复")  # 输入数据不满足业务或数值约束时立即终止。
        values = frame[column].to_numpy(np.float64)  # 转换并统一值的数据表示。
        if not np.isfinite(values).all():  # 检查参与计算的数值是否全部有限。
            raise ValueError(f"{name} {year} 因子含 NaN 或 Inf")  # 输入数据不满足业务或数值约束时立即终止。
        return frame  # 返回帧。

    daily_frames: list[pd.DataFrame] = []  # 接收daily帧。
    for year, expected_count in run_counts.items():  # 遍历执行数量items，逐项处理年份预期数量。
        if expected_count == 0:  # 按当前模式、标识或状态选择处理分支。
            continue  # 跳过当前项并处理下一项。
        baseline = read_prediction(
            baseline_run_name, year, expected_count, baseline_prediction_column  # 传入baseline执行名称、年份、预期数量、baseline预测列。
        ).rename(columns={baseline_prediction_column: "baseline_signal"})
        target = read_prediction(
            run_name, year, expected_count, prediction_column  # 传入执行名称、年份、预期数量、预测列。
        ).rename(columns={prediction_column: "target_signal"})
        paired = baseline.merge(  # 按业务键对齐，避免依赖原始行序。
            target,  # 汇入标签。
            on=keys,  # 指定数据连接使用的业务键。
            how="outer",  # 指定数据连接方式。
            validate="one_to_one",  # 校验数据连接的基数关系。
            indicator=True,  # 保留连接来源标记以审计匹配率。
        )
        if not paired["_merge"].eq("both").all():  # 检查当前掩码或数据是否满足有效性要求。
            mismatch = paired["_merge"].value_counts().to_dict()  # 按股票与交易日业务键对齐。
            raise AssertionError(  # 关键数据契约被破坏时立即终止。
                f"{year} 两个 prediction 的业务键不一致：{mismatch}"
            )
        paired = paired.drop(columns="_merge")
        paired["baseline_square"] = paired["baseline_signal"].pow(2)
        paired["target_square"] = paired["target_signal"].pow(2)
        paired["cross_product"] = (
            paired["baseline_signal"] * paired["target_signal"]
        )
        stats = paired.groupby("trade_date", sort=True).agg(  # 按既定口径聚合。
            observations=("target_signal", "size"),  # 配置observations。
            baseline_sum=("baseline_signal", "sum"),  # 配置baselinesum。
            target_sum=("target_signal", "sum"),  # 配置标签sum。
            baseline_square_sum=("baseline_square", "sum"),  # 配置baselinesquaresum。
            target_square_sum=("target_square", "sum"),  # 配置标签squaresum。
            cross_product_sum=("cross_product", "sum"),  # 配置crossproductsum。
        )
        count = stats["observations"].to_numpy(np.float64)  # 转换并统一数量的数据表示。
        baseline_variation = (
            count * stats["baseline_square_sum"].to_numpy(np.float64)
            - stats["baseline_sum"].to_numpy(np.float64) ** 2
        )
        target_variation = (
            count * stats["target_square_sum"].to_numpy(np.float64)
            - stats["target_sum"].to_numpy(np.float64) ** 2
        )
        invalid = (  # 构造invalid，限定后续参与计算的样本。
            (count < 2)
            | (baseline_variation <= 0)
            | (target_variation <= 0)
        )
        if invalid.any():  # 检查当前掩码或数据是否满足有效性要求。
            dates = stats.index[invalid].astype(str).tolist()[:5]  # 转换交易日的数据类型或存储位置。
            raise ValueError(  # 输入数据不满足业务或数值约束时立即终止。
                f"{year} 存在样本不足或常数信号截面：{dates}"
            )
        covariance = (  # 计算covariance。
            count * stats["cross_product_sum"].to_numpy(np.float64)
            - stats["baseline_sum"].to_numpy(np.float64)
            * stats["target_sum"].to_numpy(np.float64)
        )
        pearson = covariance / np.sqrt(
            baseline_variation * target_variation
        )
        daily_frames.append(  # 把当前结果追加到汇总列表。
            pd.DataFrame(
                {
                    "run_name": run_name,  # 记录运行名称。
                    "baseline_run_name": baseline_run_name,  # 记录baseline运行名称。
                    "prediction_column": prediction_column,  # 记录预测字段。
                    "baseline_prediction_column": baseline_prediction_column,  # 记录baseline预测字段。
                    "year": year,  # 记录年份。
                    "trade_date": stats.index.astype(str),  # 记录trade交易日。
                    "observations": stats["observations"].to_numpy(np.int64),  # 记录observations。
                    "pearson": np.clip(pearson, -1.0, 1.0),  # 记录Pearson。
                }
            )
        )

    if not daily_frames:  # 检查必需数据、配置或校验结果是否缺失。
        raise FileNotFoundError("两个 run 均没有可比较的非空年度预测")  # 目标文件不存在时立即终止。
    daily = pd.concat(daily_frames, ignore_index=True)  # 对齐并合并daily。
    annual = daily.groupby("year", as_index=False).agg(  # 按既定口径聚合。
        dates=("trade_date", "size"),  # 配置交易日。
        matched_rows=("observations", "sum"),  # 配置匹配数量行数。
        mean_daily_pearson=("pearson", "mean"),  # 配置均值dailyPearson。
        std_daily_pearson=("pearson", "std"),  # 配置标准差dailyPearson。
        median_daily_pearson=("pearson", "median"),  # 配置mediandailyPearson。
        min_daily_pearson=("pearson", "min"),  # 配置mindailyPearson。
        max_daily_pearson=("pearson", "max"),  # 配置maxdailyPearson。
    )
    annual.insert(0, "baseline_run_name", baseline_run_name)
    annual.insert(0, "baseline_prediction_column", baseline_prediction_column)
    annual.insert(0, "prediction_column", prediction_column)
    annual.insert(0, "run_name", run_name)
    overall = pd.DataFrame(
        {
            "run_name": [run_name],  # 记录运行名称。
            "baseline_run_name": [baseline_run_name],  # 记录baseline运行名称。
            "years": [daily["year"].nunique()],  # 记录年份。
            "prediction_column": [prediction_column],  # 记录预测字段。
            "baseline_prediction_column": [baseline_prediction_column],  # 记录baseline预测字段。
            "dates": [len(daily)],  # 记录交易日。
            "matched_rows": [int(daily["observations"].sum())],  # 记录匹配行数。
            "mean_daily_pearson": [daily["pearson"].mean()],  # 记录均值日度Pearson。
            "std_daily_pearson": [daily["pearson"].std(ddof=1)],  # 记录标准差日度Pearson。
            "median_daily_pearson": [daily["pearson"].median()],  # 记录median日度Pearson。
            "min_daily_pearson": [daily["pearson"].min()],  # 记录min日度Pearson。
            "max_daily_pearson": [daily["pearson"].max()],  # 记录max日度Pearson。
        }
    )
    return {"daily": daily, "annual": annual, "overall": overall}  # 返回按当前规则组织后的结果。
def audit_evaluation_inputs(  # 只读审计 ViViT 预测、签名链、Universe 键和 T+10 日历覆盖。
    run_name: str = DEFAULT_RUN_NAME,  # 接收执行名称。
    vivit_data_root: str | Path | None = None,  # 接收vivit数据根目录。
    reference_data_root: str | Path | None = None,  # 接收reference数据根目录。
    backtest_mode: str = DEFAULT_BACKTEST_MODE,  # 接收backtest模式。
    prediction_column: str = FACTOR_COLUMN,  # 接收预测列。
) -> pd.DataFrame:  # 声明当前函数的返回类型。
    """只读审计 ViViT 预测、签名链、Universe 键和 T+10 日历覆盖。"""

    run_name = validate_run_name(run_name)
    prediction_column = validate_prediction_column(prediction_column)
    mode = resolve_backtest_mode(backtest_mode)
    vivit_root, reference_root = _roots(vivit_data_root, reference_data_root)  # 构造vivit根目录。
    manifests = _manifests(run_name, vivit_root, reference_root)  # 读取或生成来源清单。
    counts = _prediction_counts(manifests["training"])  # 计算数量。
    prediction_dir = vivit_root / "predictions" / "runs" / run_name  # 构造预测目录。
    universe_dir = reference_root / "universe"  # 构造股票池目录。
    records: list[dict[str, object]] = []  # 初始化records容器。
    last_factor_date: str | None = None  # 接收last因子交易日。

    for year, expected_count in counts.items():  # 遍历数量items，逐项处理年份预期数量。
        prediction_path = prediction_dir / PREDICTION_PATTERN.format(year=year)  # 构造预测路径。
        if expected_count == 0:  # 按当前模式、标识或状态选择处理分支。
            if prediction_path.exists():  # 检查目标文件或目录是否满足存在性要求。
                raise AssertionError(  # 关键数据契约被破坏时立即终止。
                    f"{year} factor_counts=0 但存在预测文件：{prediction_path}"
                )
            records.append(  # 把当前结果追加到汇总列表。
                {
                    "year": year,  # 记录年份。
                    "prediction_rows": 0,  # 记录预测行数。
                    "universe_rows": 0,  # 记录股票池行数。
                    "dates": 0,  # 记录交易日。
                    "first_date": None,  # 记录首个交易日。
                    "last_date": None,  # 记录最后一个交易日。
                    "can_enter_t1_rows": 0,  # 记录canentert1行数。
                    "status": "empty_prediction_expected",  # 把 status 映射为"empty_prediction_expected"。
                }
            )
            continue  # 跳过当前项并处理下一项。
        if not prediction_path.exists():  # 检查目标文件或目录是否满足存在性要求。
            raise FileNotFoundError(prediction_path)  # 目标文件不存在时立即终止。
        prediction = pd.read_parquet(  # 读取预测。
            prediction_path,  # 汇入预测路径。
            columns=["trade_date", "instrument_id", prediction_column],  # 限定本次读取或输出使用的字段。
        )
        prediction["trade_date"] = prediction["trade_date"].astype(str)  # 转换预测的数据类型或存储位置。
        prediction["instrument_id"] = prediction["instrument_id"].astype(str)  # 转换预测的数据类型或存储位置。
        if len(prediction) != expected_count:  # 检查当前样本数量是否满足计算要求。
            raise AssertionError(  # 关键数据契约被破坏时立即终止。
                f"{year} 预测行数 {len(prediction):,} != {expected_count:,}"
            )
        if prediction.duplicated(["trade_date", "instrument_id"]).any():  # 检查当前掩码或数据是否满足有效性要求。
            raise ValueError(f"{year} prediction 业务键重复")  # 输入数据不满足业务或数值约束时立即终止。
        if not prediction["trade_date"].str.startswith(str(year)).all():  # 检查当前掩码或数据是否满足有效性要求。
            raise ValueError(f"{year} prediction 含跨年度日期")  # 输入数据不满足业务或数值约束时立即终止。
        if not np.isfinite(  # 检查参与计算的数值是否全部有限。
            prediction[prediction_column].to_numpy(np.float64)
        ).all():
            raise ValueError(f"{year} 因子含 NaN 或 Inf")  # 输入数据不满足业务或数值约束时立即终止。

        universe = pd.read_parquet(  # 读取股票池。
            universe_dir / f"universe.{year}.parquet",
            columns=[  # 初始化列容器。
                "trade_date", "instrument_id", "feature_valid",
                "in_universe_t", "can_enter_t1",
            ],
        )
        expected = universe.loc[
            universe["feature_valid"] & universe["in_universe_t"],
            ["trade_date", "instrument_id", "can_enter_t1"],
        ].copy()
        expected["trade_date"] = expected["trade_date"].astype(str)  # 转换预期的数据类型或存储位置。
        expected["instrument_id"] = expected["instrument_id"].astype(str)  # 转换预期的数据类型或存储位置。
        joined = prediction.merge(  # 按业务键对齐，避免依赖原始行序。
            expected,  # 汇入预期。
            on=["trade_date", "instrument_id"],  # 指定数据连接使用的业务键。
            how="outer",  # 指定数据连接方式。
            validate="one_to_one",  # 校验数据连接的基数关系。
            indicator=True,  # 保留连接来源标记以审计匹配率。
        )
        if not joined["_merge"].eq("both").all():  # 检查当前掩码或数据是否满足有效性要求。
            mismatch = joined["_merge"].value_counts().to_dict()  # 按股票与交易日业务键对齐。
            raise AssertionError(  # 关键数据契约被破坏时立即终止。
                f"{year} prediction 与 T 日 Universe 键不一致：{mismatch}"
            )
        if joined["can_enter_t1"].isna().any():  # 检查当前掩码或数据是否满足有效性要求。
            raise AssertionError(f"{year} can_enter_t1 含缺失值")  # 关键数据契约被破坏时立即终止。
        first_date = str(prediction["trade_date"].min())
        year_last_date = str(prediction["trade_date"].max())
        last_factor_date = max(last_factor_date or year_last_date, year_last_date)
        records.append(  # 把当前结果追加到汇总列表。
            {
                "year": year,  # 记录年份。
                "prediction_rows": len(prediction),  # 记录预测行数。
                "universe_rows": len(expected),  # 记录股票池行数。
                "dates": prediction["trade_date"].nunique(),  # 记录交易日。
                "first_date": first_date,  # 记录首个交易日。
                "last_date": year_last_date,  # 记录最后一个交易日。
                "can_enter_t1_rows": int(joined["can_enter_t1"].astype(bool).sum()),  # 记录canentert1行数。
                "status": "passed",  # 把 status 映射为"passed"。
            }
        )

    if last_factor_date is None:  # 检查所需对象是否尚未初始化。
        raise FileNotFoundError("没有可评价的非空年度预测")  # 目标文件不存在时立即终止。
    final_price_date = _calendar_last_date(last_factor_date, str(mode["name"]))  # 按交易日历定位收益起止日。
    audit = pd.DataFrame(records)
    audit.attrs.update(  # 用当前配置更新参数字典。
        {
            "run_name": run_name,  # 记录运行名称。
            "model_family": "ViViT",  # 把 model_family 映射为"ViViT"。
            "factor_role": manifests["training"].get(  # 记录因子role。
                "factor_role", "model_prediction"
            ),
            "source_run_name": manifests["training"].get("source_run_name"),  # 记录来源运行名称。
            "prediction_pattern": PREDICTION_PATTERN,  # 记录预测pattern。
            "prediction_column": prediction_column,  # 记录预测字段。
            "last_factor_date": last_factor_date,  # 记录最后一个因子交易日。
            "final_price_date": final_price_date,  # 记录final价格交易日。
            "backtest_mode": str(mode["name"]),  # 记录backtest模式。
            "price_start_offset": int(mode["price_start_offset"]),  # 记录价格起点偏移。
            "dataset_signature": manifests["dataset"]["signature"],  # 记录dataset签名，用于来源一致性校验。
            "universe_signature": manifests["universe"]["signature"],  # 记录股票池签名，用于来源一致性校验。
            "sequence_signature": manifests["sequence"]["signature"],  # 记录sequence签名，用于来源一致性校验。
        }
    )
    return audit  # 返回audit。

def _validated_evaluation_manifest(  # 校验输入形状、业务键、有限性和签名约束。
    run_name: str, vivit_root: Path,
    prediction_column: str = FACTOR_COLUMN,  # 接收预测列。
) -> dict[str, object]:  # 声明当前函数的返回类型。
    """校验输入形状、业务键、有限性和签名约束；异常时明确失败。"""
    prediction_column = validate_prediction_column(prediction_column)
    path = (
        vivit_root / "quality" / "runs" / run_name
        / evaluation_directory_name(prediction_column) / "factor_evaluation_manifest.json"
    )
    manifest = read_json(path)  # 读取清单。
    if manifest.get("schema_version") != 2 or "backtest_mode" not in manifest:  # 按当前模式、标识或状态选择处理分支。
        raise RuntimeError(  # 当前运行环境或任务状态不支持继续执行。
            f"{run_name} 是旧版评价产物，请用新版 Notebook 03 重新运行评价"
        )
    mode = resolve_backtest_mode(str(manifest["backtest_mode"]))
    expected = {  # 初始化预期容器。
        "model_family": "ViViT",  # 把 model_family 映射为"ViViT"。
        "grouping_policy": GROUPING_POLICY,  # 声明分组处理口径。
        "periods": list(PERIODS),  # 记录periods。
        "quantiles": QUANTILES,  # 记录quantiles。
        "price_start": mode["price_start"],  # 记录价格起点。
        "price_start_offset": mode["price_start_offset"],  # 记录价格起点偏移。
        "portfolio_scope": mode["portfolio_scope"],  # 记录portfolioscope。
        "execution_mask": mode["execution_mask"],  # 记录执行掩码。
        "benchmark_column": EQUAL_WEIGHT_MARKET_COLUMN,  # 记录benchmark字段。
        "benchmark_scope": BENCHMARK_SCOPE,  # 记录benchmarkscope。
        "benchmark_weighting": "daily_equal_weight",  # 把 benchmark_weighting 映射为"daily_equal_weight"。
        "benchmark_execution_mask": "none",  # 把 benchmark_execution_mask 映射为"none"。
        "prediction_column": prediction_column,  # 记录预测字段。
    }
    for key, value in expected.items():  # 遍历预期items，逐项处理键值。
        actual = manifest.get(key, FACTOR_COLUMN if key == "prediction_column" else None)  # 配置actual。
        if actual != value:  # 按当前模式、标识或状态选择处理分支。
            raise RuntimeError(  # 当前运行环境或任务状态不支持继续执行。
                f"{run_name} 评价口径不一致：{key}={actual!r}，预期 {value!r}"
            )
    return manifest  # 返回清单。
