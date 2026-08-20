"""可独立运行的 ViViT 因子流水线命令行入口。"""
from __future__ import annotations

import argparse
import json

from .data import (
    PipelinePaths,
    build_feature_store,
    build_history_indices,
    build_labels,
    build_rolling_splits,
    build_universe,
    prepare_data,
    validate_inputs,
)
from .training import (
    TrainingConfig,
    run_parallel_training,
    smoke_test,
    train_task,
    validate_outputs,
)


def _device_ids(value: str) -> tuple[int, ...]:
    # 将 0,1,2,3 形式的参数解析为物理 GPU 编号，并拒绝重复值。
    result = tuple(int(part.strip()) for part in value.split(",") if part.strip())  # 拆分逗号列表并忽略空片段。
    if not result or len(set(result)) != len(result):  # 至少指定一张卡且物理编号不能重复。
        raise argparse.ArgumentTypeError("device IDs must be a non-empty unique CSV list")  # 在启动调度前拒绝无效参数。
    return result  # 返回调度器使用的不可变 GPU 元组。


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)  # 用模块说明作为命令行总描述。
    commands = parser.add_subparsers(dest="command", required=True)  # 强制用户明确选择流程步骤。

    # 各子命令与 Notebook 中的步骤一一对应，便于交互和批处理共用函数。
    commands.add_parser("check-inputs")  # 只验证 Level2 与 Wind 输入契约。
    prepare = commands.add_parser("prepare-data")  # 注册完整数据准备子命令。
    prepare.add_argument("--overwrite", action="store_true")  # 可选强制重建所有数据产物。

    smoke = commands.add_parser("smoke-test")  # 注册单批前向反向检查。
    smoke.add_argument("--forecast-year", type=int, default=2018)  # 默认使用首个 OOS 年。
    smoke.add_argument("--seed", type=int, default=42)  # 默认使用首个基础随机种子。

    task = commands.add_parser("train-task")  # 注册调度器内部调用的单任务入口。
    task.add_argument("--forecast-year", type=int, required=True)  # 每个子进程只负责一个预测年。
    task.add_argument("--seed", type=int, required=True)  # 每个子进程只负责一个随机种子。
    task.add_argument("--physical-gpu", type=int, default=0)  # 记录父进程分配的物理卡号。
    task.add_argument("--workers", type=int, default=8)  # 控制该任务 DataLoader worker 数。
    task.add_argument("--overwrite", action="store_true")  # 显式忽略完成标记和断点。

    parallel = commands.add_parser("train-parallel")  # 注册多 GPU 年度种子调度入口。
    parallel.add_argument("--device-ids", type=_device_ids, default=(0, 1, 2, 3))  # 默认使用四张物理卡。
    parallel.add_argument("--tasks-per-gpu", type=int, default=1)  # 默认每卡只运行一个训练进程。
    parallel.add_argument("--workers", type=int, default=8)  # 每个训练进程的加载 worker 数。
    parallel.add_argument("--overwrite", action="store_true")  # 向所有子任务透传强制重训。

    run_all = commands.add_parser("run-all")  # 注册从数据到因子的端到端入口。
    run_all.add_argument("--device-ids", type=_device_ids, default=(0, 1, 2, 3))  # 指定端到端训练使用的物理卡。
    run_all.add_argument("--tasks-per-gpu", type=int, default=1)  # 控制单卡并发训练数。
    run_all.add_argument("--workers", type=int, default=8)  # 控制每项任务的数据加载并发。
    run_all.add_argument("--overwrite-data", action="store_true")  # 单独控制数据产物重建。
    run_all.add_argument("--overwrite-training", action="store_true")  # 单独控制模型任务重训。

    commands.add_parser("validate")  # 注册只读交付校验入口。
    return parser.parse_args()  # 解析当前进程参数并返回命名空间。


def main() -> None:
    args = parse_args()  # 取得用户选择的子命令及参数。
    paths = PipelinePaths.default()  # 从包位置解析平行输入目录和产物目录。
    # 每个分支只负责参数转交和展示结果，业务逻辑全部留在包函数中。
    if args.command == "check-inputs":  # 仅执行输入存在性和结构检查。
        print(json.dumps(validate_inputs(paths), ensure_ascii=False, indent=2))  # 以中文安全 JSON 展示检查结果。
    elif args.command == "prepare-data":  # 执行数据清洗和滚动切分。
        print(json.dumps(prepare_data(paths, args.overwrite), ensure_ascii=False, indent=2))  # 展示数据规模摘要。
    elif args.command == "smoke-test":  # 只验证一批训练与推理。
        print(json.dumps(smoke_test(paths, args.forecast_year, args.seed), indent=2))  # 输出最小训练指标。
    elif args.command == "train-task":  # 执行一个年度种子子任务。
        config = TrainingConfig(workers=args.workers)  # 仅用命令行值覆盖 worker 参数。
        print(json.dumps(train_task(  # 调用可恢复训练并输出任务摘要。
            paths, args.forecast_year, args.seed, args.physical_gpu,
            config=config, overwrite=args.overwrite,
        ), indent=2))
    elif args.command == "train-parallel":  # 调度全部年度种子训练。
        print(run_parallel_training(  # 输出任务级训练汇总表。
            paths, args.device_ids, args.tasks_per_gpu, args.workers, args.overwrite
        ).to_string(index=False))
    elif args.command == "run-all":  # 按固定依赖顺序执行端到端流程。
        # 完整流程先准备数据并冒烟验证，再启动耗时的独立 GPU 任务。
        print(json.dumps(prepare_data(paths, args.overwrite_data), ensure_ascii=False, indent=2))  # 先准备并汇报数据。
        print(json.dumps(smoke_test(paths), indent=2))  # 正式训练前执行单批健康检查。
        print(run_parallel_training(  # 健康检查通过后启动多 GPU 训练与因子生成。
            paths, args.device_ids, args.tasks_per_gpu,
            args.workers, args.overwrite_training,
        ).to_string(index=False))
    elif args.command == "validate":  # 只读取已有产物执行最终一致性检查。
        print(validate_outputs(paths).to_string(index=False))  # 以表格显示逐年度结果。


if __name__ == "__main__":  # 仅通过 python -m vivit_factor 启动时执行入口。
    main()  # 解析命令并分发到对应流程。
