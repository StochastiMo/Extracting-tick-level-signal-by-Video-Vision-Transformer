"""显式实验编号与统一行为契约。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


# 读取 `Literal['no0', 'no1_y', 'no1_x', 'no1_xy', 'no2', 'no3', 'no4', 'no5', 'no6']` 的当前值，并赋给 `ExperimentId`。
ExperimentId = Literal[
    "no0",
    "no1_y",
    "no1_x",
    "no1_xy",
    "no2",
    "no3",
    "no4",
    "no5",
    "no6",
]


# 使用 `dataclass(frozen=True)` 包装 `ExperimentSpec`，并定义其可复用的处理逻辑。
@dataclass(frozen=True)
class ExperimentSpec:
    """Notebook、CLI 和训练器共享的方案元数据。
    """

    # 声明 `experiment_id` 的数据类型，明确后续读写契约。
    experiment_id: ExperimentId
    # 声明 `title` 的数据类型，明确后续读写契约。
    title: str
    # 声明 `uses_barra_as_model_input` 的数据类型，明确后续读写契约。
    uses_barra_as_model_input: bool
    # 声明 `uses_barra_in_loss` 的数据类型，明确后续读写契约。
    uses_barra_in_loss: bool
    # 声明 `publishes_strict_neutral_score` 的数据类型，明确后续读写契约。
    publishes_strict_neutral_score: bool


# 构造键值映射，并保存到 `EXPERIMENTS`，供后续按键查询。
EXPERIMENTS: dict[ExperimentId, ExperimentSpec] = {
    "no0": ExperimentSpec("no0", "Baseline", False, False, False),
    "no1_y": ExperimentSpec("no1_y", "Y neutral", False, True, False),
    "no1_x": ExperimentSpec("no1_x", "X neutral", False, True, False),
    "no1_xy": ExperimentSpec("no1_xy", "XY neutral", False, True, False),
    "no2": ExperimentSpec("no2", "raw-score R² penalty", False, True, False),
    "no3": ExperimentSpec("no3", "adversarial Barra", False, True, False),
    "no4": ExperimentSpec("no4", "Barra gate", True, True, False),
    "no5": ExperimentSpec("no5", "strict neutral output", True, True, True),
    "no6": ExperimentSpec("no6", "Alpha/Style dual tower", True, True, False),
}


# 定义 `resolve_experiment` 函数，封装这一阶段可独立调用的处理步骤。
def resolve_experiment(experiment_id: str) -> ExperimentSpec:
    """校验实验编号并返回不可变方案元数据。
    
    输入：
        experiment_id: str。
    输出：
        ExperimentSpec。
        """

    # 尝试执行本段主逻辑；若发生异常，则交由后续异常分支处理。
    try:
        # 将 `EXPERIMENTS[experiment_id]` 作为当前函数结果返回给调用方。
        return EXPERIMENTS[experiment_id]  # type: ignore[index]
    # 捕获 `except KeyError as error` 对应的异常，并执行明确的降级、记录或清理逻辑。
    except KeyError as error:
        # 抛出 `ValueError(f'未知实验编号 {experiment_id!r}；可选 {tuple(EXPERIMENTS)}')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(
            f"未知实验编号 {experiment_id!r}；可选 {tuple(EXPERIMENTS)}"
        ) from error
