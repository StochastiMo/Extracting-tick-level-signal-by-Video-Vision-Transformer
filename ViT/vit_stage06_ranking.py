"""阶段 06：定义可微 RankIC、相关性与基础回归损失。

本模块不读取数据或全局训练状态；所有影响数值的超参数都由调用者显式传入。
"""
from __future__ import annotations


import numpy as np
import torch


# 设置 `TEMPORAL_TURNOVER_LOSS` 的固定配置值，供后续流程统一使用。
TEMPORAL_TURNOVER_LOSS = "sigmoid_rank_ic_turnover"
# 组织一组有序值，并保存到 `LOSS_NAMES`。
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
# 组织一组有序值，并保存到 `STANDARD_RANK_LOSS_NAMES`。
STANDARD_RANK_LOSS_NAMES = ("sigmoid_rank_ic", "neural_sort_rank_ic")


# 定义 `pearson_correlation` 函数，封装这一阶段可独立调用的处理步骤。
def pearson_correlation(
    predictions: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """以 FP32 返回两个等长向量的 Pearson 相关系数。

    输入：
        predictions: torch.Tensor。
        targets: torch.Tensor。
    输出：
        torch.Tensor。
    """

    # 调用 `predictions.reshape(-1).float` 统一设备或数据类型，并将结果保存到 `x`。
    x = predictions.reshape(-1).float()
    # 调用 `targets.reshape(-1).float` 统一设备或数据类型，并将结果保存到 `y`。
    y = targets.reshape(-1).float()
    # 检查 `len(x) != len(y) or len(x) < 2`；根据条件结果选择对应处理分支。
    if len(x) != len(y) or len(x) < 2:
        # 抛出 `ValueError('相关系数要求预测与标签等长且至少包含两个样本')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("相关系数要求预测与标签等长且至少包含两个样本")
    # 计算 `x - x.mean()`，并将数值结果保存到 `centered_x`。
    centered_x = x - x.mean()
    # 计算 `y - y.mean()`，并将数值结果保存到 `centered_y`。
    centered_y = y - y.mean()
    # 调用 `clamp_min` 把数值下界截为 epsilon 以避免除零，并将结果保存到 `denominator`。
    denominator = torch.sqrt(
        centered_x.square().sum() * centered_y.square().sum()
    ).clamp_min(1e-8)
    # 将 `centered_x.mul(centered_y).sum() / denominator` 作为当前函数结果返回给调用方。
    return centered_x.mul(centered_y).sum() / denominator


# 定义 `negative_pearson_ic` 函数，封装这一阶段可独立调用的处理步骤。
def negative_pearson_ic(
    predictions: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """返回可直接最小化的负 Pearson IC。

    输入：
        predictions: torch.Tensor。
        targets: torch.Tensor。
    输出：
        torch.Tensor。
    """

    # 将 `-pearson_correlation(predictions, targets)` 作为当前函数结果返回给调用方。
    return -pearson_correlation(predictions, targets)


# region TR-COMMON-SOFT-RANK
# 定义 `sigmoid_soft_rank` 函数，封装这一阶段可独立调用的处理步骤。
def sigmoid_soft_rank(
    values: torch.Tensor, temperature: float
) -> torch.Tensor:
    """通过所有两两 Sigmoid 比较近似降序名次。

    输入：
        values: torch.Tensor。
        temperature: float。
    输出：
        torch.Tensor。
    """

    # 调用 `values.reshape(-1).float` 统一设备或数据类型，并将结果保存到 `vector`。
    vector = values.reshape(-1).float()
    # 检查 `len(vector) < 2 or temperature <= 0`；根据条件结果选择对应处理分支。
    if len(vector) < 2 or temperature <= 0:
        # 抛出 `ValueError('Sigmoid 软排序要求至少两个样本且 temperature > 0')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("Sigmoid 软排序要求至少两个样本且 temperature > 0")
    # 计算 `vector.unsqueeze(0) - vector.unsqueeze(1)`，并将数值结果保存到 `pairwise`。
    pairwise = vector.unsqueeze(0) - vector.unsqueeze(1)
    # 对角线为 0.5；末尾补 0.5 后严格对应 1 + sum(j != i)。
    return torch.sigmoid(pairwise / temperature).sum(dim=1) + 0.5


# 定义 `neural_sort_soft_rank` 函数，封装这一阶段可独立调用的处理步骤。
def neural_sort_soft_rank(
    values: torch.Tensor, temperature: float
) -> torch.Tensor:
    """用 NeuralSort 软排列矩阵计算原始元素的降序软名次。

    输入：
        values: torch.Tensor。
        temperature: float。
    输出：
        torch.Tensor。
    """

    # 调用 `values.reshape(-1).float` 统一设备或数据类型，并将结果保存到 `vector`。
    vector = values.reshape(-1).float()
    # 调用 `len` 汇总当前数据，并将统计量保存到 `size`。
    size = len(vector)
    # 检查 `size < 2 or temperature <= 0`；根据条件结果选择对应处理分支。
    if size < 2 or temperature <= 0:
        # 抛出 `ValueError('NeuralSort 要求至少两个样本且 temperature > 0')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("NeuralSort 要求至少两个样本且 temperature > 0")
    # 调用 `(vector.unsqueeze(1) - vector.unsqueeze(0)).abs` 逐元素取绝对值，并将结果保存到 `absolute_differences`。
    absolute_differences = (
        vector.unsqueeze(1) - vector.unsqueeze(0)
    ).abs()
    # 调用 `absolute_differences.sum` 汇总当前数据，并将统计量保存到 `distance_sums`。
    distance_sums = absolute_differences.sum(dim=1)
    # 调用 `torch.arange` 按目标形状初始化数值容器 `positions`。
    positions = torch.arange(
        1, size + 1, device=vector.device, dtype=vector.dtype
    )
    # 计算 `size + 1 - 2 * positions`，并将数值结果保存到 `scaling`。
    scaling = size + 1 - 2 * positions
    # 计算 `(scaling.unsqueeze(1) * vector.unsqueeze(0) - distance_sums.unsqueeze(0)) / temperature`，并将数值结果保存到 `logits`。
    logits = (
        scaling.unsqueeze(1) * vector.unsqueeze(0)
        - distance_sums.unsqueeze(0)
    ) / temperature
    # 调用 `torch.softmax` 沿指定维度归一化为概率权重，并将结果保存到 `soft_permutation`。
    soft_permutation = torch.softmax(logits, dim=1)
    # 将 `soft_permutation.transpose(0, 1).matmul(positions)` 作为当前函数结果返回给调用方。
    return soft_permutation.transpose(0, 1).matmul(positions)


# 定义 `differentiable_rank_ic` 函数，封装这一阶段可独立调用的处理步骤。
def differentiable_rank_ic(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    method: str,
    temperature: float,
) -> torch.Tensor:
    """对预测和标签使用同一种软排序，再返回可微相关系数。

    输入：
        predictions: torch.Tensor。
        targets: torch.Tensor。
        method: str。
        temperature: float。
    输出：
        torch.Tensor。
    """

    # 调用 `{'sigmoid_rank_ic': sigmoid_soft_rank, 'neural_sort_rank_ic': n…` 计算预测与标签或风险暴露之间的相关性，并将结果保存到 `rank_function`。
    rank_function = {
        "sigmoid_rank_ic": sigmoid_soft_rank,
        "neural_sort_rank_ic": neural_sort_soft_rank,
    }.get(method)
    # 检查 `rank_function is None`；根据条件结果选择对应处理分支。
    if rank_function is None:
        # 抛出 `ValueError(f'未知可微 RankIC 方法：{method}')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(f"未知可微 RankIC 方法：{method}")
    # 调用 `rank_function` 根据分数计算截面排序或排序索引，并将结果保存到 `prediction_ranks`。
    prediction_ranks = rank_function(predictions, temperature)
    # 标签不是模型输出，因此不需要为标签软名次保留计算图。
    with torch.no_grad():
        # 调用 `rank_function` 根据分数计算截面排序或排序索引，并将结果保存到 `target_ranks`。
        target_ranks = rank_function(targets.detach(), temperature)
    # 将 `pearson_correlation(prediction_ranks, target_ranks)` 作为当前函数结果返回给调用方。
    return pearson_correlation(prediction_ranks, target_ranks)
# endregion


# 定义 `temporal_turnover_loss_components` 函数，封装这一阶段可独立调用的处理步骤。
def temporal_turnover_loss_components(
    current_predictions: torch.Tensor,
    current_targets: torch.Tensor,
    previous_predictions: torch.Tensor,
    previous_positions: np.ndarray | torch.Tensor,
    current_positions: np.ndarray | torch.Tensor,
    temperature: float,
    turnover_lambda: float,
) -> dict[str, torch.Tensor]:
    """返回预测 RankIC、相邻日秩相关、换手惩罚及总损失。

    输入：
        current_predictions: torch.Tensor。
        current_targets: torch.Tensor。
        previous_predictions: torch.Tensor。
        previous_positions: np.ndarray | torch.Tensor。
        current_positions: np.ndarray | torch.Tensor。
        temperature: float。
        turnover_lambda: float。
    输出：
        dict[str, torch.Tensor]。
    """

    # 检查 `temperature <= 0 or turnover_lambda < 0`；根据条件结果选择对应处理分支。
    if temperature <= 0 or turnover_lambda < 0:
        # 抛出 `ValueError('时序换手损失要求 temperature > 0 且 lambda >= 0')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("时序换手损失要求 temperature > 0 且 lambda >= 0")
    # 调用 `current_predictions.reshape(-1).float` 统一设备或数据类型，并将结果保存到 `current`。
    current = current_predictions.reshape(-1).float()
    # 调用 `current_targets.reshape(-1).float` 统一设备或数据类型，并将结果保存到 `targets`。
    targets = current_targets.reshape(-1).float()
    # 调用 `previous_predictions.detach().reshape(-1).float` 统一设备或数据类型，并将结果保存到 `previous`。
    previous = previous_predictions.detach().reshape(-1).float()
    # 调用 `torch.as_tensor` 转换数据表示，并将标准化对象保存到 `previous_index`。
    previous_index = torch.as_tensor(
        previous_positions, dtype=torch.long, device=previous.device
    )
    # 调用 `torch.as_tensor` 转换数据表示，并将标准化对象保存到 `current_index`。
    current_index = torch.as_tensor(
        current_positions, dtype=torch.long, device=current.device
    )
    # 检查 `len(previous_index) != len(current_index) or len(current_index) < 2`；根据条件结果选择对应处理分支。
    if len(previous_index) != len(current_index) or len(current_index) < 2:
        # 抛出 `ValueError('时序换手损失要求至少两只对齐的共同股票')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("时序换手损失要求至少两只对齐的共同股票")
    # 调用 `differentiable_rank_ic` 计算预测与标签或风险暴露之间的相关性，并将结果保存到 `prediction_rank_ic`。
    prediction_rank_ic = differentiable_rank_ic(
        current, targets, "sigmoid_rank_ic", temperature
    )
    # 调用 `sigmoid_soft_rank` 根据分数计算截面排序或排序索引，并将结果保存到 `current_common_ranks`。
    current_common_ranks = sigmoid_soft_rank(
        current.index_select(0, current_index), temperature
    )
    # 在 `torch.no_grad()` 管理的上下文中执行本段逻辑，并保证资源按约定释放。
    with torch.no_grad():
        # 调用 `sigmoid_soft_rank` 根据分数计算截面排序或排序索引，并将结果保存到 `previous_common_ranks`。
        previous_common_ranks = sigmoid_soft_rank(
            previous.index_select(0, previous_index), temperature
        )
    # 调用 `pearson_correlation` 计算预测与标签或风险暴露之间的相关性，并将结果保存到 `temporal_rank_ic`。
    temporal_rank_ic = pearson_correlation(
        current_common_ranks, previous_common_ranks
    )
    # 计算 `1 - temporal_rank_ic`，并将数值结果保存到 `temporal_penalty`。
    temporal_penalty = 1 - temporal_rank_ic
    # 计算 `-prediction_rank_ic + turnover_lambda * temporal_penalty`，并将数值结果保存到 `loss`。
    loss = -prediction_rank_ic + turnover_lambda * temporal_penalty
    # 将 `{'loss': loss, 'prediction_soft_rank_ic': prediction_rank_ic, 'temporal_soft_rank_ic': temporal…` 作为当前函数结果返回给调用方。
    return {
        "loss": loss,
        "prediction_soft_rank_ic": prediction_rank_ic,
        "temporal_soft_rank_ic": temporal_rank_ic,
        "temporal_penalty": temporal_penalty,
    }


# region TR-COMMON-TRAINING-LOSS
# 定义 `compute_training_loss` 函数，封装这一阶段可独立调用的处理步骤。
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
    """根据显式配置返回一个标量训练损失。

    输入：
        predictions: torch.Tensor。
        targets: torch.Tensor。
        loss_name: str。
        rank_temperature: float。
        huber_delta: float。
        smooth_l1_beta: float。
        hybrid_weight: float。
        turnover_lambda: float。
        previous_predictions: torch.Tensor | None。
        previous_positions: np.ndarray | torch.Tensor | None。
        current_positions: np.ndarray | torch.Tensor | None。
    输出：
        torch.Tensor。
    """

    # 检查 `loss_name not in LOSS_NAMES`；根据条件结果选择对应处理分支。
    if loss_name not in LOSS_NAMES:
        # 抛出 `ValueError(f'未知损失函数：{loss_name}')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(f"未知损失函数：{loss_name}")
    # 检查 `rank_temperature <= 0 or turnover_lambda < 0 or huber_delta <= 0 or (smooth_l1_beta <= 0) or (n…`；根据条件结果选择对应处理分支。
    if (
        rank_temperature <= 0
        or turnover_lambda < 0
        or huber_delta <= 0
        or smooth_l1_beta <= 0
        or not 0 <= hybrid_weight <= 1
    ):
        # 抛出 `ValueError('损失温度、换手权重、delta、beta 或混合权重非法')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("损失温度、换手权重、delta、beta 或混合权重非法")

    # 调用 `predictions.reshape(-1).float` 统一设备或数据类型，并将结果保存到 `x`。
    x = predictions.reshape(-1).float()
    # 调用 `targets.reshape(-1).float` 统一设备或数据类型，并将结果保存到 `y`。
    y = targets.reshape(-1).float()
    # 检查 `loss_name == 'pearson_ic'`；根据条件结果选择对应处理分支。
    if loss_name == "pearson_ic":
        # 将 `negative_pearson_ic(x, y)` 作为当前函数结果返回给调用方。
        return negative_pearson_ic(x, y)
    # 检查 `loss_name == TEMPORAL_TURNOVER_LOSS`；根据条件结果选择对应处理分支。
    if loss_name == TEMPORAL_TURNOVER_LOSS:
        # 检查 `previous_predictions is None or previous_positions is None or current_positions is None`；根据条件结果选择对应处理分支。
        if (
            previous_predictions is None
            or previous_positions is None
            or current_positions is None
        ):
            # 抛出 `ValueError('sigmoid_rank_ic_turnover 缺少上一日预测或股票对齐位置')`，阻止不满足数据契约的流程继续执行。
            raise ValueError(
                "sigmoid_rank_ic_turnover 缺少上一日预测或股票对齐位置"
            )
        # 将 `temporal_turnover_loss_components(x, y, previous_predictions, previous_positions, current_posit…` 作为当前函数结果返回给调用方。
        return temporal_turnover_loss_components(
            x,
            y,
            previous_predictions,
            previous_positions,
            current_positions,
            rank_temperature,
            turnover_lambda,
        )["loss"]
    # 检查 `loss_name in STANDARD_RANK_LOSS_NAMES`；根据条件结果选择对应处理分支。
    if loss_name in STANDARD_RANK_LOSS_NAMES:
        # 将 `-differentiable_rank_ic(x, y, loss_name, rank_temperature)` 作为当前函数结果返回给调用方。
        return -differentiable_rank_ic(
            x, y, loss_name, rank_temperature
        )
    # 检查 `loss_name == 'mse'`；根据条件结果选择对应处理分支。
    if loss_name == "mse":
        # 将 `torch.nn.functional.mse_loss(x, y)` 作为当前函数结果返回给调用方。
        return torch.nn.functional.mse_loss(x, y)
    # 检查 `loss_name == 'huber'`；根据条件结果选择对应处理分支。
    if loss_name == "huber":
        # 将 `torch.nn.functional.huber_loss(x, y, delta=huber_delta)` 作为当前函数结果返回给调用方。
        return torch.nn.functional.huber_loss(
            x, y, delta=huber_delta
        )
    # 检查 `loss_name == 'smooth_l1'`；根据条件结果选择对应处理分支。
    if loss_name == "smooth_l1":
        # 将 `torch.nn.functional.smooth_l1_loss(x, y, beta=smooth_l1_beta)` 作为当前函数结果返回给调用方。
        return torch.nn.functional.smooth_l1_loss(
            x, y, beta=smooth_l1_beta
        )
    # 计算 `1 - pearson_correlation(x, y)`，并将数值结果保存到 `pearson_term`。
    pearson_term = 1 - pearson_correlation(x, y)
    # 调用 `torch.nn.functional.huber_loss` 按当前实验口径计算可反向传播的训练损失，并将结果保存到 `huber_term`。
    huber_term = torch.nn.functional.huber_loss(
        x, y, delta=huber_delta
    )
    # 将 `hybrid_weight * pearson_term + (1 - hybrid_weight) * huber_term` 作为当前函数结果返回给调用方。
    return hybrid_weight * pearson_term + (
        1 - hybrid_weight
    ) * huber_term
# endregion
