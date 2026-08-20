"""阶段 06：定义可微 RankIC、相关性与基础回归损失。

本模块不读取数据或全局训练状态；所有影响数值的超参数都由调用者显式传入。
"""
from __future__ import annotations


import numpy as np
import torch


TEMPORAL_TURNOVER_LOSS = "sigmoid_rank_ic_turnover"  # 设置时序turnover损失。
LOSS_NAMES = (  # 集中声明损失名称。
    "pearson_ic",
    "sigmoid_rank_ic",
    "neural_sort_rank_ic",
    "mse",
    "huber",
    "smooth_l1",
    "pearson_huber",
    TEMPORAL_TURNOVER_LOSS,
)
STANDARD_RANK_LOSS_NAMES = ("sigmoid_rank_ic", "neural_sort_rank_ic")  # 集中声明standard排名损失名称。


def pearson_correlation(  # 以 FP32 返回两个等长向量的 Pearson 相关系数。
    predictions: torch.Tensor, targets: torch.Tensor  # 均为 [N]。
) -> torch.Tensor:  # 声明当前函数的返回类型。
    """以 FP32 返回两个等长向量的 Pearson 相关系数。"""

    x = predictions.reshape(-1).float()  # [N,...] -> [N]。
    y = targets.reshape(-1).float()  # [N,...] -> [N]。
    if len(x) != len(y) or len(x) < 2:  # 检查当前样本数量是否满足计算要求。
        raise ValueError("相关系数要求预测与标签等长且至少包含两个样本")  # 输入数据不满足业务或数值约束时立即终止。
    centered_x = x - x.mean()  # 计算中心化x。
    centered_y = y - y.mean()  # 计算中心化y。
    denominator = torch.sqrt(  # 下界截断以避免除零。
        centered_x.square().sum() * centered_y.square().sum()
    ).clamp_min(1e-8)
    return centered_x.mul(centered_y).sum() / denominator  # 返回以 FP32 返回两个等长向量的 Pearson 相关系数。


def negative_pearson_ic(  # 返回可直接最小化的负 Pearson IC。
    predictions: torch.Tensor, targets: torch.Tensor  # 均为 [N]。
) -> torch.Tensor:  # 声明当前函数的返回类型。
    """返回可直接最小化的负 Pearson IC。"""

    return -pearson_correlation(predictions, targets)  # 返回可直接最小化的负 Pearson IC。



def sigmoid_soft_rank(  # 通过所有两两 Sigmoid 比较近似降序名次。
    values: torch.Tensor, temperature: float  # values:[N]。
) -> torch.Tensor:  # 声明当前函数的返回类型。
    """通过所有两两 Sigmoid 比较近似降序名次。"""

    vector = values.reshape(-1).float()  # [N,...] -> [N]。
    if len(vector) < 2 or temperature <= 0:  # 检查当前样本数量是否满足计算要求。
        raise ValueError("Sigmoid 软排序要求至少两个样本且 temperature > 0")  # 输入数据不满足业务或数值约束时立即终止。
    pairwise = vector.unsqueeze(0) - vector.unsqueeze(1)  # [1,N]-[N,1] -> [N,N]。
  # 对角线为 0.5；末尾补 0.5 后严格对应 1 + sum(j != i)。
    return torch.sigmoid(pairwise / temperature).sum(dim=1) + 0.5  # 返回通过所有两两 Sigmoid 比较近似降序名次。


def neural_sort_soft_rank(  # 用 NeuralSort 软排列矩阵计算原始元素的降序软名次。
    values: torch.Tensor, temperature: float  # values:[N]。
) -> torch.Tensor:  # 声明当前函数的返回类型。
    """用 NeuralSort 软排列矩阵计算原始元素的降序软名次。"""

    vector = values.reshape(-1).float()  # [N,...] -> [N]。
    size = len(vector)
    if size < 2 or temperature <= 0:  # 检查当前数值是否越过业务或数值阈值。
        raise ValueError("NeuralSort 要求至少两个样本且 temperature > 0")  # 输入数据不满足业务或数值约束时立即终止。
    absolute_differences = (
        vector.unsqueeze(1) - vector.unsqueeze(0)  # [N,1]-[1,N] -> [N,N]。
    ).abs()
    distance_sums = absolute_differences.sum(dim=1)  # 计算distancesums。
    positions = torch.arange(
        1, size + 1, device=vector.device, dtype=vector.dtype
    )
    scaling = size + 1 - 2 * positions
    logits = (
        scaling.unsqueeze(1) * vector.unsqueeze(0)  # [N,1]*[1,N] -> [N,N]。
        - distance_sums.unsqueeze(0)  # [N] -> [1,N]，广播后 logits 保持 [N,N]。
    ) / temperature
    soft_permutation = torch.softmax(logits, dim=1)  # [N,N]，每个排序位置对原元素的软权重。
    return soft_permutation.transpose(0, 1).matmul(positions)  # [N,N]@[N] -> [N] 软名次。


def differentiable_rank_ic(  # 对预测和标签使用同一种软排序，再返回可微相关系数。
    predictions: torch.Tensor,  # [N]，截面预测分数。
    targets: torch.Tensor,  # [N]，截面标签。
    method: str,  # 接收method。
    temperature: float,  # 接收temperature。
) -> torch.Tensor:  # 声明当前函数的返回类型。
    """对预测和标签使用同一种软排序，再返回可微相关系数。"""

    rank_function = {  # 计算当前截面相关性。
        "sigmoid_rank_ic": sigmoid_soft_rank,  # 记录sigmoid秩ic。
        "neural_sort_rank_ic": neural_sort_soft_rank,  # 记录neuralsort秩ic。
    }.get(method)
    if rank_function is None:  # 检查所需对象是否尚未初始化。
        raise ValueError(f"未知可微 RankIC 方法：{method}")  # 输入数据不满足业务或数值约束时立即终止。
    prediction_ranks = rank_function(predictions, temperature)  # 计算预测排名。
  # 标签不是模型输出，因此不需要为标签软名次保留计算图。
    with torch.no_grad():  # 关闭梯度记录以降低验证或推理开销。
        target_ranks = rank_function(targets.detach(), temperature)  # 计算标签排名。
    return pearson_correlation(prediction_ranks, target_ranks)  # 返回完成上述处理后的结果。



def temporal_turnover_loss_components(  # 返回预测 RankIC、相邻日秩相关、换手惩罚及总损失。
    current_predictions: torch.Tensor,  # [N_t]，当前交易日分数。
    current_targets: torch.Tensor,  # [N_t]，当前交易日标签。
    previous_predictions: torch.Tensor,  # [N_{t-1}]，上一交易日分数。
    previous_positions: np.ndarray | torch.Tensor,  # [M]，共同股票在上一日的位置。
    current_positions: np.ndarray | torch.Tensor,  # [M]，共同股票在当日的位置。
    temperature: float,  # 接收temperature。
    turnover_lambda: float,  # 接收turnoverlambda。
) -> dict[str, torch.Tensor]:
    """返回预测 RankIC、相邻日秩相关、换手惩罚及总损失。"""

    if temperature <= 0 or turnover_lambda < 0:  # 检查当前数值是否越过业务或数值阈值。
        raise ValueError("时序换手损失要求 temperature > 0 且 lambda >= 0")  # 输入数据不满足业务或数值约束时立即终止。
    current = current_predictions.reshape(-1).float()  # [N_t,...] -> [N_t]。
    targets = current_targets.reshape(-1).float()  # [N_t,...] -> [N_t]。
    previous = previous_predictions.detach().reshape(-1).float()  # [N_{t-1},...] -> [N_{t-1}]。
    previous_index = torch.as_tensor(  # 转换并统一上一期索引的数据表示。
        previous_positions, dtype=torch.long, device=previous.device
    )
    current_index = torch.as_tensor(  # 转换并统一当前索引的数据表示。
        current_positions, dtype=torch.long, device=current.device
    )
    if len(previous_index) != len(current_index) or len(current_index) < 2:  # 检查当前样本数量是否满足计算要求。
        raise ValueError("时序换手损失要求至少两只对齐的共同股票")  # 输入数据不满足业务或数值约束时立即终止。
    prediction_rank_ic = differentiable_rank_ic(  # 计算当前截面相关性。
        current, targets, "sigmoid_rank_ic", temperature
    )
    current_common_ranks = sigmoid_soft_rank(  # 计算当前公共排名。
        current.index_select(0, current_index), temperature
    )
    with torch.no_grad():  # 关闭梯度记录以降低验证或推理开销。
        previous_common_ranks = sigmoid_soft_rank(  # 计算上一期公共排名。
            previous.index_select(0, previous_index), temperature
        )
    temporal_rank_ic = pearson_correlation(  # 计算当前截面相关性。
        current_common_ranks, previous_common_ranks  # 传入当前公共排名、上一期公共排名。
    )
    temporal_penalty = 1 - temporal_rank_ic
    loss = -prediction_rank_ic + turnover_lambda * temporal_penalty  # 计算损失。
    return {  # 返回预测 RankIC、相邻日秩相关、换手惩罚及总损失。
        "loss": loss,  # 记录损失。
        "prediction_soft_rank_ic": prediction_rank_ic,  # 记录预测soft秩ic。
        "temporal_soft_rank_ic": temporal_rank_ic,  # 记录时序soft秩ic。
        "temporal_penalty": temporal_penalty,  # 记录时序penalty。
    }



def compute_training_loss(  # 根据显式配置返回一个标量训练损失。
    predictions: torch.Tensor,  # [N]，截面预测分数。
    targets: torch.Tensor,  # [N]，截面标签。
    *,
    loss_name: str,  # 接收损失名称。
    rank_temperature: float,  # 接收排名temperature。
    huber_delta: float,  # 接收huberdelta。
    smooth_l1_beta: float,  # 接收smoothl1beta。
    hybrid_weight: float,  # 接收hybrid权重。
    turnover_lambda: float,  # 接收turnoverlambda。
    previous_predictions: torch.Tensor | None = None,  # [N_{t-1}]。
    previous_positions: np.ndarray | torch.Tensor | None = None,  # [M]。
    current_positions: np.ndarray | torch.Tensor | None = None,  # [M]。
) -> torch.Tensor:
    """根据显式配置返回一个标量训练损失。"""

    if loss_name not in LOSS_NAMES:  # 检查当前值是否属于允许集合。
        raise ValueError(f"未知损失函数：{loss_name}")  # 输入数据不满足业务或数值约束时立即终止。
    if (  # 开始组合多项有效性或配置条件。
        rank_temperature <= 0
        or turnover_lambda < 0
        or huber_delta <= 0
        or smooth_l1_beta <= 0
        or not 0 <= hybrid_weight <= 1
    ):
        raise ValueError("损失温度、换手权重、delta、beta 或混合权重非法")  # 输入数据不满足业务或数值约束时立即终止。

    x = predictions.reshape(-1).float()  # [N,...] -> [N]。
    y = targets.reshape(-1).float()  # [N,...] -> [N]。
    if loss_name == "pearson_ic":  # 按当前模式、标识或状态选择处理分支。
        return negative_pearson_ic(x, y)  # 返回根据显式配置返回一个标量训练损失。
    if loss_name == TEMPORAL_TURNOVER_LOSS:  # 按当前模式、标识或状态选择处理分支。
        if (  # 开始组合多项有效性或配置条件。
            previous_predictions is None
            or previous_positions is None
            or current_positions is None
        ):
            raise ValueError(  # 输入数据不满足业务或数值约束时立即终止。
                "sigmoid_rank_ic_turnover 缺少上一日预测或股票对齐位置"
            )
        return temporal_turnover_loss_components(  # 返回根据显式配置返回一个标量训练损失。
            x,
            y,
            previous_predictions,
            previous_positions,
            current_positions,
            rank_temperature,
            turnover_lambda,
        )["loss"]
    if loss_name in STANDARD_RANK_LOSS_NAMES:  # 检查当前值是否属于允许集合。
        return -differentiable_rank_ic(  # 返回根据显式配置返回一个标量训练损失。
            x, y, loss_name, rank_temperature  # 传入x、y、损失名称、排名temperature。
        )
    if loss_name == "mse":  # 按当前模式、标识或状态选择处理分支。
        return torch.nn.functional.mse_loss(x, y)  # 返回根据显式配置返回一个标量训练损失。
    if loss_name == "huber":  # 按当前模式、标识或状态选择处理分支。
        return torch.nn.functional.huber_loss(  # 返回根据显式配置返回一个标量训练损失。
            x, y, delta=huber_delta
        )
    if loss_name == "smooth_l1":  # 按当前模式、标识或状态选择处理分支。
        return torch.nn.functional.smooth_l1_loss(  # 返回根据显式配置返回一个标量训练损失。
            x, y, beta=smooth_l1_beta
        )
    pearson_term = 1 - pearson_correlation(x, y)
    huber_term = torch.nn.functional.huber_loss(  # 按实验口径计算可微损失。
        x, y, delta=huber_delta
    )
    return hybrid_weight * pearson_term + (  # 返回根据显式配置返回一个标量训练损失。
        1 - hybrid_weight
    ) * huber_term
