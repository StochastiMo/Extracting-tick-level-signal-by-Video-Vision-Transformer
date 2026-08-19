"""显式实验编号与统一行为契约。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


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

    experiment_id: ExperimentId
    title: str
    uses_barra_as_model_input: bool
    uses_barra_in_loss: bool
    publishes_strict_neutral_score: bool


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


def resolve_experiment(experiment_id: str) -> ExperimentSpec:
    """校验实验编号并返回不可变方案元数据。"""

    try:
        return EXPERIMENTS[experiment_id]  # type: ignore[index]
    except KeyError as error:
        raise ValueError(
            f"未知实验编号 {experiment_id!r}；可选 {tuple(EXPERIMENTS)}"
        ) from error
