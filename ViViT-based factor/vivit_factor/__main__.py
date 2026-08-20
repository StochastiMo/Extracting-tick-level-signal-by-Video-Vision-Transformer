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
    result = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("device IDs must be a non-empty unique CSV list")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    # 各子命令与 Notebook 中的步骤一一对应，便于交互和批处理共用函数。
    commands.add_parser("check-inputs")
    prepare = commands.add_parser("prepare-data")
    prepare.add_argument("--overwrite", action="store_true")

    smoke = commands.add_parser("smoke-test")
    smoke.add_argument("--forecast-year", type=int, default=2018)
    smoke.add_argument("--seed", type=int, default=42)

    task = commands.add_parser("train-task")
    task.add_argument("--forecast-year", type=int, required=True)
    task.add_argument("--seed", type=int, required=True)
    task.add_argument("--physical-gpu", type=int, default=0)
    task.add_argument("--workers", type=int, default=8)
    task.add_argument("--overwrite", action="store_true")

    parallel = commands.add_parser("train-parallel")
    parallel.add_argument("--device-ids", type=_device_ids, default=(0, 1, 2, 3))
    parallel.add_argument("--tasks-per-gpu", type=int, default=1)
    parallel.add_argument("--workers", type=int, default=8)
    parallel.add_argument("--overwrite", action="store_true")

    run_all = commands.add_parser("run-all")
    run_all.add_argument("--device-ids", type=_device_ids, default=(0, 1, 2, 3))
    run_all.add_argument("--tasks-per-gpu", type=int, default=1)
    run_all.add_argument("--workers", type=int, default=8)
    run_all.add_argument("--overwrite-data", action="store_true")
    run_all.add_argument("--overwrite-training", action="store_true")

    commands.add_parser("validate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = PipelinePaths.default()
    # 每个分支只负责参数转交和展示结果，业务逻辑全部留在包函数中。
    if args.command == "check-inputs":
        print(json.dumps(validate_inputs(paths), ensure_ascii=False, indent=2))
    elif args.command == "prepare-data":
        print(json.dumps(prepare_data(paths, args.overwrite), ensure_ascii=False, indent=2))
    elif args.command == "smoke-test":
        print(json.dumps(smoke_test(paths, args.forecast_year, args.seed), indent=2))
    elif args.command == "train-task":
        config = TrainingConfig(workers=args.workers)
        print(json.dumps(train_task(
            paths, args.forecast_year, args.seed, args.physical_gpu,
            config=config, overwrite=args.overwrite,
        ), indent=2))
    elif args.command == "train-parallel":
        print(run_parallel_training(
            paths, args.device_ids, args.tasks_per_gpu, args.workers, args.overwrite
        ).to_string(index=False))
    elif args.command == "run-all":
        # 完整流程先准备数据并冒烟验证，再启动耗时的独立 GPU 任务。
        print(json.dumps(prepare_data(paths, args.overwrite_data), ensure_ascii=False, indent=2))
        print(json.dumps(smoke_test(paths), indent=2))
        print(run_parallel_training(
            paths, args.device_ids, args.tasks_per_gpu,
            args.workers, args.overwrite_training,
        ).to_string(index=False))
    elif args.command == "validate":
        print(validate_outputs(paths).to_string(index=False))


if __name__ == "__main__":
    main()
