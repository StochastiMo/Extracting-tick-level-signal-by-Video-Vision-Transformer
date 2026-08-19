"""显式实验编号与统一行为契约。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ExperimentId = Literal[  # 计算experimentid。
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
@dataclass(frozen=True)  # 自动生成配置对象的初始化与比较方法。
class ExperimentSpec:  # Notebook、CLI 和训练器共享的方案元数据。
    """Notebook、CLI 和训练器共享的方案元数据。
    """

    experiment_id: ExperimentId  # 接收experimentid。
    title: str  # 接收title。
    uses_barra_as_model_input: bool  # 接收usesBarraas模型输入。
    uses_barra_in_loss: bool  # 接收usesBarrain损失。
    publishes_strict_neutral_score: bool  # 接收publishesstrictneutral分数。


EXPERIMENTS: dict[ExperimentId, ExperimentSpec] = {  # 设置experiments。
    "no0": ExperimentSpec("no0", "Baseline", False, False, False),  # 记录no0。
    "no1_y": ExperimentSpec("no1_y", "Y neutral", False, True, False),  # 记录no1y。
    "no1_x": ExperimentSpec("no1_x", "X neutral", False, True, False),  # 记录no1x。
    "no1_xy": ExperimentSpec("no1_xy", "XY neutral", False, True, False),  # 记录no1xy。
    "no2": ExperimentSpec("no2", "raw-score R² penalty", False, True, False),  # 记录no2。
    "no3": ExperimentSpec("no3", "adversarial Barra", False, True, False),  # 记录no3。
    "no4": ExperimentSpec("no4", "Barra gate", True, True, False),  # 记录no4。
    "no5": ExperimentSpec("no5", "strict neutral output", True, True, True),  # 记录no5。
    "no6": ExperimentSpec("no6", "Alpha/Style dual tower", True, True, False),  # 记录no6。
}


def resolve_experiment(experiment_id: str) -> ExperimentSpec:  # 校验实验编号并返回不可变方案元数据。
    """校验实验编号并返回不可变方案元数据。"""

    try:  # 执行可能失败的外部读取或计算步骤。
        return EXPERIMENTS[experiment_id]  # type: ignore[index]。
    except KeyError as error:  # 捕获预期异常并执行对应恢复逻辑。
        raise ValueError(  # 输入数据不满足业务或数值约束时立即终止。
            f"未知实验编号 {experiment_id!r}；可选 {tuple(EXPERIMENTS)}"  # 执行tuple对应的处理步骤。
        ) from error
