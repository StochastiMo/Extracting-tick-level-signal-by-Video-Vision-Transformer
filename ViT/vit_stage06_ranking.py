"""阶段 06：定义可微 RankIC、相关性与基础回归损失。

本模块不读取数据或全局训练状态；所有影响数值的超参数都由调用者显式传入。
"""
from __future__ import annotations


import numpy as np
import torch


TEMPORAL_TURNOVER_LOSS = "sigmoid_rank_ic_turnover"
LOSS_NAMES = (
    "pearson_ic",
    "sigmoid_rank_ic",
    "neural_sort_rank_ic",
    "mse",
    "huber",
    "smooth_l1",
    "pearson_huber",
    TEMPORAL_TURNOVER_LOSS,
)
STANDARD_RANK_LOSS_NAMES = ("sigmoid_rank_ic", "neural_sort_rank_ic")


def pearson_correlation(
    predictions: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """以 FP32 返回两个等长向量的 Pearson 相关系数。"""

    x = predictions.reshape(-1).float()
    y = targets.reshape(-1).float()
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("相关系数要求预测与标签等长且至少包含两个样本")
    centered_x = x - x.mean()
    centered_y = y - y.mean()
    denominator = torch.sqrt(  # 下界截断以避免除零
        centered_x.square().sum() * centered_y.square().sum()
    ).clamp_min(1e-8)
    return centered_x.mul(centered_y).sum() / denominator


def negative_pearson_ic(
    predictions: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """返回可直接最小化的负 Pearson IC。"""

    return -pearson_correlation(predictions, targets)


# region TR-COMMON-SOFT-RANK
def sigmoid_soft_rank(
    values: torch.Tensor, temperature: float
) -> torch.Tensor:
    """通过所有两两 Sigmoid 比较近似降序名次。"""

    vector = values.reshape(-1).float()
    if len(vector) < 2 or temperature <= 0:
        raise ValueError("Sigmoid 软排序要求至少两个样本且 temperature > 0")
    pairwise = vector.unsqueeze(0) - vector.unsqueeze(1)
    # 对角线为 0.5；末尾补 0.5 后严格对应 1 + sum(j != i)。
    return torch.sigmoid(pairwise / temperature).sum(dim=1) + 0.5


def neural_sort_soft_rank(
    values: torch.Tensor, temperature: float
) -> torch.Tensor:
    """用 NeuralSort 软排列矩阵计算原始元素的降序软名次。"""

    vector = values.reshape(-1).float()
    size = len(vector)
    if size < 2 or temperature <= 0:
        raise ValueError("NeuralSort 要求至少两个样本且 temperature > 0")
    absolute_differences = (
        vector.unsqueeze(1) - vector.unsqueeze(0)
    ).abs()
    distance_sums = absolute_differences.sum(dim=1)
    positions = torch.arange(
        1, size + 1, device=vector.device, dtype=vector.dtype
    )
    scaling = size + 1 - 2 * positions
    logits = (
        scaling.unsqueeze(1) * vector.unsqueeze(0)
        - distance_sums.unsqueeze(0)
    ) / temperature
    soft_permutation = torch.softmax(logits, dim=1)  # 归一化为软排序权重
    return soft_permutation.transpose(0, 1).matmul(positions)


def differentiable_rank_ic(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    method: str,
    temperature: float,
) -> torch.Tensor:
    """对预测和标签使用同一种软排序，再返回可微相关系数。"""

    rank_function = {  # 计算当前截面相关性
        "sigmoid_rank_ic": sigmoid_soft_rank,
        "neural_sort_rank_ic": neural_sort_soft_rank,
    }.get(method)
    if rank_function is None:
        raise ValueError(f"未知可微 RankIC 方法：{method}")
    prediction_ranks = rank_function(predictions, temperature)
    # 标签不是模型输出，因此不需要为标签软名次保留计算图。
    with torch.no_grad():
        target_ranks = rank_function(targets.detach(), temperature)
    return pearson_correlation(prediction_ranks, target_ranks)
# endregion


def temporal_turnover_loss_components(
    current_predictions: torch.Tensor,
    current_targets: torch.Tensor,
    previous_predictions: torch.Tensor,
    previous_positions: np.ndarray | torch.Tensor,
    current_positions: np.ndarray | torch.Tensor,
    temperature: float,
    turnover_lambda: float,
) -> dict[str, torch.Tensor]:
    """返回预测 RankIC、相邻日秩相关、换手惩罚及总损失。"""

    if temperature <= 0 or turnover_lambda < 0:
        raise ValueError("时序换手损失要求 temperature > 0 且 lambda >= 0")
    current = current_predictions.reshape(-1).float()
    targets = current_targets.reshape(-1).float()
    previous = previous_predictions.detach().reshape(-1).float()
    previous_index = torch.as_tensor(
        previous_positions, dtype=torch.long, device=previous.device
    )
    current_index = torch.as_tensor(
        current_positions, dtype=torch.long, device=current.device
    )
    if len(previous_index) != len(current_index) or len(current_index) < 2:
        raise ValueError("时序换手损失要求至少两只对齐的共同股票")
    prediction_rank_ic = differentiable_rank_ic(  # 计算当前截面相关性
        current, targets, "sigmoid_rank_ic", temperature
    )
    current_common_ranks = sigmoid_soft_rank(
        current.index_select(0, current_index), temperature
    )
    with torch.no_grad():
        previous_common_ranks = sigmoid_soft_rank(
            previous.index_select(0, previous_index), temperature
        )
    temporal_rank_ic = pearson_correlation(  # 计算当前截面相关性
        current_common_ranks, previous_common_ranks
    )
    temporal_penalty = 1 - temporal_rank_ic
    loss = -prediction_rank_ic + turnover_lambda * temporal_penalty
    return {
        "loss": loss,
        "prediction_soft_rank_ic": prediction_rank_ic,
        "temporal_soft_rank_ic": temporal_rank_ic,
        "temporal_penalty": temporal_penalty,
    }


# region TR-COMMON-TRAINING-LOSS
def compute_training_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    loss_name: str,
    rank_temperature: float,
    huber_delta: float,
    smooth_l1_beta: float,
    hybrid_weight: float,
    turnover_lambda: float,
    previous_predictions: torch.Tensor | None = None,
    previous_positions: np.ndarray | torch.Tensor | None = None,
    current_positions: np.ndarray | torch.Tensor | None = None,
) -> torch.Tensor:
    """根据显式配置返回一个标量训练损失。"""

    if loss_name not in LOSS_NAMES:
        raise ValueError(f"未知损失函数：{loss_name}")
    if (
        rank_temperature <= 0
        or turnover_lambda < 0
        or huber_delta <= 0
        or smooth_l1_beta <= 0
        or not 0 <= hybrid_weight <= 1
    ):
        raise ValueError("损失温度、换手权重、delta、beta 或混合权重非法")

    x = predictions.reshape(-1).float()
    y = targets.reshape(-1).float()
    if loss_name == "pearson_ic":
        return negative_pearson_ic(x, y)
    if loss_name == TEMPORAL_TURNOVER_LOSS:
        if (
            previous_predictions is None
            or previous_positions is None
            or current_positions is None
        ):
            raise ValueError(
                "sigmoid_rank_ic_turnover 缺少上一日预测或股票对齐位置"
            )
        return temporal_turnover_loss_components(
            x,
            y,
            previous_predictions,
            previous_positions,
            current_positions,
            rank_temperature,
            turnover_lambda,
        )["loss"]
    if loss_name in STANDARD_RANK_LOSS_NAMES:
        return -differentiable_rank_ic(
            x, y, loss_name, rank_temperature
        )
    if loss_name == "mse":
        return torch.nn.functional.mse_loss(x, y)
    if loss_name == "huber":
        return torch.nn.functional.huber_loss(
            x, y, delta=huber_delta
        )
    if loss_name == "smooth_l1":
        return torch.nn.functional.smooth_l1_loss(
            x, y, beta=smooth_l1_beta
        )
    pearson_term = 1 - pearson_correlation(x, y)
    huber_term = torch.nn.functional.huber_loss(  # 按实验口径计算可微损失
        x, y, delta=huber_delta
    )
    return hybrid_weight * pearson_term + (
        1 - hybrid_weight
    ) * huber_term
# endregion
