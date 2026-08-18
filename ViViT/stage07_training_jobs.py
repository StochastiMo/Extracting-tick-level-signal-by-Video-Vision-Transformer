"""阶段 07：启动、查询和恢复长期运行的 ViViT 后台训练任务。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


# 读取 `Path(__file__).resolve().parent` 的当前值，并赋给 `VIVIT_DIR`。
VIVIT_DIR = Path(__file__).resolve().parent
# 调用 `Path(__file__).resolve` 规范化文件路径，并将结果保存到 `SCRIPT_PATH`。
SCRIPT_PATH = Path(__file__).resolve()
# 调用 `tuple` 整理数据结构，并将结果保存到 `FORECAST_YEARS`。
FORECAST_YEARS = tuple(range(2018, 2026))
# 组织一组有序值，并保存到 `SEEDS`。
SEEDS = (42, 43, 44)
# 计算 `len(FORECAST_YEARS) * len(SEEDS)`，并将数值结果保存到 `EXPECTED_TASKS`。
EXPECTED_TASKS = len(FORECAST_YEARS) * len(SEEDS)
# 计算 `len(FORECAST_YEARS) - 1`，并将数值结果保存到 `EXPECTED_PREDICTION_FILES`。
EXPECTED_PREDICTION_FILES = len(FORECAST_YEARS) - 1
# 设置 `HEARTBEAT_SECONDS` 的固定配置值，供后续流程统一使用。
HEARTBEAT_SECONDS = 30.0
# 调用 `re.compile` 编译正则表达式供后续重复匹配，并将结果保存到 `_SAFE_NAME`。
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")
# 调用 `re.compile` 编译正则表达式供后续重复匹配，并将结果保存到 `_EPOCH_PATTERN`。
_EPOCH_PATTERN = re.compile(r"epoch=(\d+)")


# 定义内部辅助函数 `_utc_now`，封装仅供本模块复用的计算步骤。
def _utc_now() -> str:
    """返回带 UTC 时区的当前时间字符串。
    
    输入：
        无显式输入参数。
    输出：
        str。
        """
    # 将 `datetime.now(timezone.utc).isoformat()` 作为当前函数结果返回给调用方。
    return datetime.now(timezone.utc).isoformat()


# 定义内部辅助函数 `_atomic_write_json`，封装仅供本模块复用的计算步骤。
def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """通过临时文件和原子替换写入产物，避免产生半成品。
    
    输入：
        path: Path。
        payload: dict[str, Any]。
    输出：
        None。
        """
    # 调用 `path.parent.mkdir` 创建输出目录，确保后续文件可以安全写入。
    path.parent.mkdir(parents=True, exist_ok=True)
    # 调用 `path.with_name` 规范化文件路径，并将结果保存到 `temporary`。
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    # 调用 `temporary.write_text` 将当前结果持久化到目标文件。
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # 调用 `os.replace` 把完整临时产物替换为正式输出。
    os.replace(temporary, path)


# 定义内部辅助函数 `_read_json`，封装仅供本模块复用的计算步骤。
def _read_json(path: Path) -> dict[str, Any] | None:
    """读取对应产物，并按接口约定处理缺失文件。
    
    输入：
        path: Path。
    输出：
        dict[str, Any] | None。
        """
    # 检查 `not path.exists()`；根据条件结果选择对应处理分支。
    if not path.exists():
        # 将 `None` 作为当前函数结果返回给调用方。
        return None
    # 将 `json.loads(path.read_text(encoding='utf-8'))` 作为当前函数结果返回给调用方。
    return json.loads(path.read_text(encoding="utf-8"))


# 定义 `default_screen_session_name` 函数，封装这一阶段可独立调用的处理步骤。
def default_screen_session_name(run_name: str) -> str:
    """为一个 run 生成确定、合法且长度受限的 screen 会话名。
    
    输入：
        run_name: str。
    输出：
        str。
        """
    # 调用 `_SAFE_NAME.sub('_', str(run_name).strip()).strip` 去除字符串首尾空白以统一配置值，并将结果保存到 `cleaned`。
    cleaned = _SAFE_NAME.sub("_", str(run_name).strip()).strip("._-")
    # 检查 `not cleaned`；根据条件结果选择对应处理分支。
    if not cleaned:
        # 抛出 `ValueError('run_name 无法生成有效的 screen 会话名')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("run_name 无法生成有效的 screen 会话名")
    # 计算 `f'vivit_{cleaned}'`，并将结果写入 `value`。
    value = f"vivit_{cleaned}"
    # 检查 `len(value) <= 80`；根据条件结果选择对应处理分支。
    if len(value) <= 80:
        # 将 `value` 作为当前函数结果返回给调用方。
        return value
    # 读取 `hashlib.sha256(value.encode('utf-8')).hexdigest()[:8]` 的当前值，并赋给 `suffix`。
    suffix = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
    # 将 `f'{value[:71]}_{suffix}'` 作为当前函数结果返回给调用方。
    return f"{value[:71]}_{suffix}"


# 定义内部辅助函数 `_validate_session_name`，封装仅供本模块复用的计算步骤。
def _validate_session_name(session_name: str) -> str:
    """校验输入形状、业务键、有限性和签名约束；异常时明确失败。
    
    输入：
        session_name: str。
    输出：
        str。
        """
    # 调用 `str(session_name).strip` 去除字符串首尾空白以统一配置值，并将结果保存到 `value`。
    value = str(session_name).strip()
    # 检查 `not value or _SAFE_NAME.search(value)`；根据条件结果选择对应处理分支。
    if not value or _SAFE_NAME.search(value):
        # 抛出 `ValueError('SCREEN_SESSION_NAME 只能包含字母、数字、下划线、连字符和句点')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(
            "SCREEN_SESSION_NAME 只能包含字母、数字、下划线、连字符和句点"
        )
    # 将 `value` 作为当前函数结果返回给调用方。
    return value


# 定义内部辅助函数 `_job_paths`，封装仅供本模块复用的计算步骤。
def _job_paths(run_name: str, vivit_dir: Path) -> tuple[Path, Path]:
    """返回指定 run 的后台任务清单和总控日志路径。
    
    输入：
        run_name: str。
        vivit_dir: Path。
    输出：
        tuple[Path, Path]。
        """
    # 计算 `vivit_dir / 'data' / 'quality' / 'runs' / run_name / 'background_job'`，并将数值结果保存到 `root`。
    root = vivit_dir / "data" / "quality" / "runs" / run_name / "background_job"
    # 将 `(root / 'job.json', root / 'orchestrator.log')` 作为当前函数结果返回给调用方。
    return root / "job.json", root / "orchestrator.log"


# 定义内部辅助函数 `_command_signature`，封装仅供本模块复用的计算步骤。
def _command_signature(command: Sequence[str]) -> str:
    """构造或签名后台任务使用的命令行。
    
    输入：
        command: Sequence[str]。
    输出：
        str。
        """
    # 调用 `json.dumps` 把结构化对象序列化为 JSON 文本，并将结果保存到 `encoded`。
    encoded = json.dumps(list(command), ensure_ascii=False, separators=(",", ":"))
    # 将 `hashlib.sha256(encoded.encode('utf-8')).hexdigest()` 作为当前函数结果返回给调用方。
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


# 定义内部辅助函数 `_screen_sessions`，封装仅供本模块复用的计算步骤。
def _screen_sessions() -> dict[str, str]:
    """读取 screen 会话并解析其运行状态。
    
    输入：
        无显式输入参数。
    输出：
        dict[str, str]。
        """
    # 调用 `shutil.which` 查找外部命令在系统中的可执行路径，并将结果保存到 `executable`。
    executable = shutil.which("screen")
    # 检查 `executable is None`；根据条件结果选择对应处理分支。
    if executable is None:
        # 将 `{}` 作为当前函数结果返回给调用方。
        return {}
    # 调用 `subprocess.run` 同步执行外部命令并收集退出状态与输出，并将结果保存到 `result`。
    result = subprocess.run(
        [executable, "-ls"], text=True, capture_output=True, check=False
    )
    # 构造键值映射，并保存到 `sessions`，供后续按键查询。
    sessions: dict[str, str] = {}
    # 遍历 `(result.stdout + result.stderr).splitlines()`，并把当前元素绑定到 `line` 逐项处理。
    for line in (result.stdout + result.stderr).splitlines():
        # 调用 `re.search` 在目标文本中查找满足规则的片段，并将结果保存到 `match`。
        match = re.search(r"\s*(\d+\.\S+)\s+\(", line)
        # 检查 `match`；根据条件结果选择对应处理分支。
        if match:
            # 调用 `match.group` 读取正则匹配结果中的指定捕获组，并将结果保存到 `full_name`。
            full_name = match.group(1)
            # 读取 `full_name` 的当前值，并赋给 `sessions[full_name.split('.', 1)[1]]`。
            sessions[full_name.split(".", 1)[1]] = full_name
    # 将 `sessions` 作为当前函数结果返回给调用方。
    return sessions


# 定义内部辅助函数 `_training_processes`，封装仅供本模块复用的计算步骤。
def _training_processes(run_name: str) -> list[dict[str, Any]]:
    """执行该模式下的训练步骤，并返回损失与诊断信息。
    
    输入：
        run_name: str。
    输出：
        list[dict[str, Any]]。
        """
    # 调用 `subprocess.run` 同步执行外部命令并收集退出状态与输出，并将结果保存到 `result`。
    result = subprocess.run(
        ["ps", "-eo", "pid=,args="], text=True, capture_output=True, check=True
    )
    # 构造当前步骤所需的列表，并保存到 `processes`。
    processes: list[dict[str, Any]] = []
    # 遍历 `result.stdout.splitlines()`，并把当前元素绑定到 `line` 逐项处理。
    for line in result.stdout.splitlines():
        # 调用 `line.strip` 去除字符串首尾空白以统一配置值，并将结果保存到 `stripped`。
        stripped = line.strip()
        # 检查 `not stripped`；根据条件结果选择对应处理分支。
        if not stripped:
            # 跳过当前元素的剩余步骤，继续处理下一项。
            continue
        # 调用 `stripped.partition` 按首个分隔符拆成前缀、分隔符和剩余文本，并将结果保存到 `(pid_text, _, command)`。
        pid_text, _, command = stripped.partition(" ")
        # 检查 `not pid_text.isdigit()`；根据条件结果选择对应处理分支。
        if not pid_text.isdigit():
            # 跳过当前元素的剩余步骤，继续处理下一项。
            continue
        # 检查 `not ('stage05_baseline_training.py' in command or 'stage06_barra_training.py' in command)`；根据条件结果选择对应处理分支。
        if not (
            "stage05_baseline_training.py" in command
            or "stage06_barra_training.py" in command
        ):
            # 跳过当前元素的剩余步骤，继续处理下一项。
            continue
        # 调用 `re.search` 在目标文本中查找满足规则的片段，并将结果保存到 `run_match`。
        run_match = re.search(r"--run-name\s+(\S+)", command)
        # 检查 `run_match is None or run_match.group(1) != run_name`；根据条件结果选择对应处理分支。
        if run_match is None or run_match.group(1) != run_name:
            # 跳过当前元素的剩余步骤，继续处理下一项。
            continue
        # 调用 `re.search` 在目标文本中查找满足规则的片段，并将结果保存到 `year_match`。
        year_match = re.search(r"--worker-year\s+(\d+)", command)
        # 调用 `re.search` 在目标文本中查找满足规则的片段，并将结果保存到 `seed_match`。
        seed_match = re.search(r"--worker-seed\s+(\d+)", command)
        # 把本轮结果追加到 `processes`，累积后续需要汇总的记录。
        processes.append(
            {
                "pid": int(pid_text),
                "role": (
                    "worker"
                    if re.search(r"\s(?:worker|ddp-worker)\s", f" {command} ")
                    else "orchestrator"
                ),
                "forecast_year": int(year_match.group(1)) if year_match else None,
                "seed": int(seed_match.group(1)) if seed_match else None,
            }
        )
    # 将 `processes` 作为当前函数结果返回给调用方。
    return processes


# 定义内部辅助函数 `_last_logged_epoch`，封装仅供本模块复用的计算步骤。
def _last_logged_epoch(path: Path) -> int | None:
    """从 checkpoint 或日志提取训练任务进度。
    
    输入：
        path: Path。
    输出：
        int | None。
        """
    # 检查 `not path.exists() or path.stat().st_size == 0`；根据条件结果选择对应处理分支。
    if not path.exists() or path.stat().st_size == 0:
        # 将 `None` 作为当前函数结果返回给调用方。
        return None
    # 在 `path.open('rb')` 管理的上下文中执行本段逻辑，并保证资源按约定释放。
    with path.open("rb") as handle:
        # 调用 `handle.seek` 把文件指针移动到日志尾部附近。
        handle.seek(max(0, path.stat().st_size - 16384))
        # 调用 `handle.read().decode` 把日志字节按 UTF-8 解码为文本，并将结果保存到 `tail`。
        tail = handle.read().decode("utf-8", errors="replace")
    # 调用 `_EPOCH_PATTERN.findall` 提取日志中全部 epoch 进度记录，并将结果保存到 `matches`。
    matches = _EPOCH_PATTERN.findall(tail)
    # 将 `int(matches[-1]) if matches else None` 作为当前函数结果返回给调用方。
    return int(matches[-1]) if matches else None


# 定义内部辅助函数 `_task_progress`，封装仅供本模块复用的计算步骤。
def _task_progress(run_name: str, vivit_dir: Path) -> list[dict[str, Any]]:
    """从 checkpoint 或日志提取训练任务进度。
    
    输入：
        run_name: str。
        vivit_dir: Path。
    输出：
        list[dict[str, Any]]。
        """
    # 计算 `vivit_dir / 'models' / 'runs' / run_name`，并将数值结果保存到 `model_root`。
    model_root = vivit_dir / "models" / "runs" / run_name
    # 计算 `vivit_dir / 'data' / 'quality' / 'runs' / run_name`，并将数值结果保存到 `quality_root`。
    quality_root = vivit_dir / "data" / "quality" / "runs" / run_name
    # 构造键值映射，并保存到 `processes`，供后续按键查询。
    processes = {
        (item["forecast_year"], item["seed"]): item["pid"]
        for item in _training_processes(run_name)
        if item["role"] == "worker"
    }
    # 构造当前步骤所需的列表，并保存到 `records`。
    records: list[dict[str, Any]] = []
    # 遍历 `FORECAST_YEARS`，并把当前元素绑定到 `year` 逐项处理。
    for year in FORECAST_YEARS:
        # 遍历 `SEEDS`，并把当前元素绑定到 `seed` 逐项处理。
        for seed in SEEDS:
            # 计算 `model_root / f'forecast_{year}' / f'seed_{seed}'`，并将数值结果保存到 `task_root`。
            task_root = model_root / f"forecast_{year}" / f"seed_{seed}"
            # 调用 `_read_json` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `completed`。
            completed = _read_json(task_root / "completed.json")
            # 设置 `last_epoch` 的固定配置值，供后续流程统一使用。
            last_epoch = None
            # 检查 `completed is not None`；根据条件结果选择对应处理分支。
            if completed is not None:
                # 调用 `completed.get('summary', {}).get` 按键读取映射中的可选字段，并在缺失时使用约定默认值，并将结果保存到 `last_epoch`。
                last_epoch = completed.get("summary", {}).get("last_epoch")
                # 设置 `state` 的固定配置值，供后续流程统一使用。
                state = "completed"
            # 当前置条件不成立时进入备用分支，保证其余合法情况仍被处理。
            else:
                # 计算 `f'forecast_{year}.seed_{seed}.log'`，并将结果写入 `log_name`。
                log_name = f"forecast_{year}.seed_{seed}.log"
                # 调用 `_last_logged_epoch` 执行 `_last_logged_epoch` 对应的明确变换，并产生后续步骤使用的结果，并将结果保存到 `last_epoch`。
                last_epoch = _last_logged_epoch(
                    quality_root / "style_task_logs" / log_name
                )
                # 检查 `last_epoch is None`；根据条件结果选择对应处理分支。
                if last_epoch is None:
                    # 调用 `_last_logged_epoch` 执行 `_last_logged_epoch` 对应的明确变换，并产生后续步骤使用的结果，并将结果保存到 `last_epoch`。
                    last_epoch = _last_logged_epoch(
                        quality_root / "a2_task_logs" / log_name
                    )
                # 检查 `(year, seed) in processes`；根据条件结果选择对应处理分支。
                if last_epoch is None:
                    # 执行当前分布式训练步骤并保持既定数据语义。
                    last_epoch = _last_logged_epoch(
                        quality_root / "ddp_task_logs" / log_name
                    )
                # 执行当前分布式训练步骤并保持既定数据语义。
                if (year, seed) in processes:
                    # 设置 `state` 的固定配置值，供后续流程统一使用。
                    state = "running"
                # 检查 `(task_root / 'last.pt').exists()`；根据条件结果选择对应处理分支。
                elif (task_root / "last.pt").exists():
                    # 设置 `state` 的固定配置值，供后续流程统一使用。
                    state = "resumable"
                # 当前置条件不成立时进入备用分支，保证其余合法情况仍被处理。
                else:
                    # 设置 `state` 的固定配置值，供后续流程统一使用。
                    state = "queued"
            # 把本轮结果追加到 `records`，累积后续需要汇总的记录。
            records.append(
                {
                    "forecast_year": year,
                    "seed": seed,
                    "state": state,
                    "last_epoch": last_epoch,
                    "pid": processes.get((year, seed)),
                }
            )
    # 将 `records` 作为当前函数结果返回给调用方。
    return records


# 定义 `training_job_status` 函数，封装这一阶段可独立调用的处理步骤。
def training_job_status(
    run_name: str,
    *,
    screen_session_name: str | None = None,
    vivit_dir: Path | str = VIVIT_DIR,
) -> dict[str, Any]:
    """汇总持久化任务状态、screen 存活情况及各年度 checkpoint 进度。
    
    输入：
        run_name: str。
        screen_session_name: str | None。
        vivit_dir: Path | str。
    输出：
        dict[str, Any]。
        """
    # 调用 `Path(vivit_dir).resolve` 规范化文件路径，并将结果保存到 `root`。
    root = Path(vivit_dir).resolve()
    # 调用 `_job_paths` 解析当前 run 对应的数据与输出路径，并将结果保存到 `(job_path, log_path)`。
    job_path, log_path = _job_paths(run_name, root)
    # 根据 `_read_json(job_path) or {}` 计算条件结果，并保存到 `job`。
    job = _read_json(job_path) or {}
    # 根据 `screen_session_name or job.get('screen_session')` 计算条件结果，并保存到 `session_name`。
    session_name = screen_session_name or job.get("screen_session")
    # 检查 `session_name is None`；根据条件结果选择对应处理分支。
    if session_name is None:
        # 调用 `default_screen_session_name` 查询或管理外部训练会话与进程，并将结果保存到 `session_name`。
        session_name = default_screen_session_name(run_name)
    # 调用 `_validate_session_name` 校验输入、配置或产物是否满足当前阶段的数据契约，并将结果保存到 `session_name`。
    session_name = _validate_session_name(session_name)
    # 调用 `_screen_sessions` 查询或管理外部训练会话与进程，并将结果保存到 `sessions`。
    sessions = _screen_sessions()
    # 调用 `_training_processes` 读取或组织模型训练所需的配置与状态，并将结果保存到 `processes`。
    processes = _training_processes(run_name)
    # 调用 `_task_progress` 汇总当前任务状态与已完成进度，并将结果保存到 `tasks`。
    tasks = _task_progress(run_name, root)
    # 调用 `sum` 汇总当前数据，并将统计量保存到 `completed_tasks`。
    completed_tasks = sum(item["state"] == "completed" for item in tasks)
    # 调用 `sum` 汇总当前数据，并将统计量保存到 `resumable_tasks`。
    resumable_tasks = sum(item["state"] == "resumable" for item in tasks)
    # 调用 `sum` 汇总当前数据，并将统计量保存到 `running_tasks`。
    running_tasks = sum(item["state"] == "running" for item in tasks)
    # 根据 `session_name in sessions` 计算条件结果，并保存到 `screen_alive`。
    screen_alive = session_name in sessions

    # 调用 `job.get` 按键读取映射中的可选字段，并在缺失时使用约定默认值，并将结果保存到 `recorded_state`。
    recorded_state = job.get("status")
    # 调用 `len` 汇总当前数据，并将统计量保存到 `prediction_files`。
    prediction_files = len(
        list(
            (
                root / "data" / "predictions" / "runs" / run_name
            ).glob("vivit_predictions.*.parquet")
        )
    )
    # 检查 `screen_alive or processes`；根据条件结果选择对应处理分支。
    if screen_alive or processes:
        # 按 `job` 在两个候选值间选择，并保存到 `state`。
        state = "running" if job else "running_unmanaged"
    # 检查 `recorded_state == 'failed'`；根据条件结果选择对应处理分支。
    elif recorded_state == "failed":
        # 设置 `state` 的固定配置值，供后续流程统一使用。
        state = "failed"
    # 检查 `recorded_state == 'completed'`；根据条件结果选择对应处理分支。
    elif recorded_state == "completed":
        # 按 `completed_tasks == EXPECTED_TASKS and prediction_files == EXPECTED_PREDICTION_FILES` 在两个候选值间选择，并保存到 `state`。
        state = (
            "completed"
            if completed_tasks == EXPECTED_TASKS
            and prediction_files == EXPECTED_PREDICTION_FILES
            else "incomplete_outputs"
        )
    # 检查 `recorded_state in {'launching', 'running'}`；根据条件结果选择对应处理分支。
    elif recorded_state in {"launching", "running"}:
        # 设置 `state` 的固定配置值，供后续流程统一使用。
        state = "interrupted"
    # 检查 `completed_tasks == EXPECTED_TASKS and prediction_files == EXPECTED_PREDICTION_FILES`；根据条件结果选择对应处理分支。
    elif (
        completed_tasks == EXPECTED_TASKS
        and prediction_files == EXPECTED_PREDICTION_FILES
    ):
        # 设置 `state` 的固定配置值，供后续流程统一使用。
        state = "completed_existing"
    # 检查 `completed_tasks or resumable_tasks`；根据条件结果选择对应处理分支。
    elif completed_tasks or resumable_tasks:
        # 设置 `state` 的固定配置值，供后续流程统一使用。
        state = "resumable"
    # 当前置条件不成立时进入备用分支，保证其余合法情况仍被处理。
    else:
        # 设置 `state` 的固定配置值，供后续流程统一使用。
        state = "not_started"

    # 将 `{'run_name': run_name, 'status': state, 'screen_session': session_name, 'screen_alive': screen_…` 作为当前函数结果返回给调用方。
    return {
        "run_name": run_name,
        "status": state,
        "screen_session": session_name,
        "screen_alive": screen_alive,
        "completed_tasks": completed_tasks,
        "expected_tasks": EXPECTED_TASKS,
        "running_tasks": running_tasks,
        "resumable_tasks": resumable_tasks,
        "prediction_files": prediction_files,
        "job_path": str(job_path),
        "log_path": str(log_path),
        "last_heartbeat": job.get("last_heartbeat"),
        "exit_code": job.get("exit_code"),
        "error": job.get("error"),
        "processes": processes,
        "tasks": tasks,
    }


# 定义 `format_training_job_status` 函数，封装这一阶段可独立调用的处理步骤。
def format_training_job_status(status: dict[str, Any]) -> str:
    """把后台任务状态格式化为 Notebook02 展示的紧凑文本。
    
    输入：
        status: dict[str, Any]。
    输出：
        str。
        """
    # 构造当前步骤所需的列表，并保存到 `lines`。
    lines = [
        f"run_name: {status['run_name']}",
        f"status: {status['status']}",
        f"screen_session: {status['screen_session']}",
        f"screen_alive: {status['screen_alive']}",
        (
            "tasks: "
            f"completed={status['completed_tasks']}/{status['expected_tasks']}, "
            f"running={status['running_tasks']}, "
            f"resumable={status['resumable_tasks']}"
        ),
        f"prediction_files: {status['prediction_files']}",
        f"log: {status['log_path']}",
    ]
    # 检查 `status.get('last_heartbeat')`；根据条件结果选择对应处理分支。
    if status.get("last_heartbeat"):
        # 把本轮结果追加到 `lines`，累积后续需要汇总的记录。
        lines.append(f"last_heartbeat: {status['last_heartbeat']}")
    # 检查 `status.get('exit_code') is not None`；根据条件结果选择对应处理分支。
    if status.get("exit_code") is not None:
        # 把本轮结果追加到 `lines`，累积后续需要汇总的记录。
        lines.append(f"exit_code: {status['exit_code']}")
    # 检查 `status.get('error')`；根据条件结果选择对应处理分支。
    if status.get("error"):
        # 把本轮结果追加到 `lines`，累积后续需要汇总的记录。
        lines.append(f"error: {status['error']}")
    # 构造当前步骤所需的列表，并保存到 `active`。
    active = [item for item in status["tasks"] if item["state"] == "running"]
    # 遍历 `active`，并把当前元素绑定到 `item` 逐项处理。
    for item in active:
        # 把本轮结果追加到 `lines`，累积后续需要汇总的记录。
        lines.append(
            "active: "
            f"{item['forecast_year']}/seed={item['seed']} "
            f"epoch={item['last_epoch']} pid={item['pid']}"
        )
    # 将 `'\n'.join(lines)` 作为当前函数结果返回给调用方。
    return "\n".join(lines)


# 定义 `tail_training_job_log` 函数，封装这一阶段可独立调用的处理步骤。
def tail_training_job_log(
    run_name: str,
    *,
    lines: int = 30,
    vivit_dir: Path | str = VIVIT_DIR,
) -> str:
    """读取后台总控日志末尾若干行。
    
    输入：
        run_name: str。
        lines: int。
        vivit_dir: Path | str。
    输出：
        str。
        """
    # 检查 `lines < 0`；根据条件结果选择对应处理分支。
    if lines < 0:
        # 抛出 `ValueError('lines 必须 >= 0')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("lines 必须 >= 0")
    # 调用 `_job_paths` 解析当前 run 对应的数据与输出路径，并将结果保存到 `(_, log_path)`。
    _, log_path = _job_paths(run_name, Path(vivit_dir).resolve())
    # 检查 `lines == 0 or not log_path.exists()`；根据条件结果选择对应处理分支。
    if lines == 0 or not log_path.exists():
        # 将 `''` 作为当前函数结果返回给调用方。
        return ""
    # 将 `'\n'.join(log_path.read_text(encoding='utf-8', errors='replace').splitlines()[-lines:])` 作为当前函数结果返回给调用方。
    return "\n".join(
        log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    )


# 定义 `launch_training_job` 函数，封装这一阶段可独立调用的处理步骤。
def launch_training_job(
    command: Sequence[str],
    run_name: str,
    *,
    screen_session_name: str | None = None,
    allow_command_change: bool = False,
    restart_completed: bool = False,
    vivit_dir: Path | str = VIVIT_DIR,
) -> dict[str, Any]:
    """在独立 screen 中启动 A2 总控，或复用已有任务状态。
    
    输入：
        command: Sequence[str]。
        run_name: str。
        screen_session_name: str | None。
        allow_command_change: bool。
        restart_completed: bool。
        vivit_dir: Path | str。
    输出：
        dict[str, Any]。
        """
    # 调用 `Path(vivit_dir).resolve` 规范化文件路径，并将结果保存到 `root`。
    root = Path(vivit_dir).resolve()
    # 构造当前步骤所需的列表，并保存到 `normalized_command`。
    normalized_command = [str(value) for value in command]
    # 检查 `not normalized_command`；根据条件结果选择对应处理分支。
    if not normalized_command:
        # 抛出 `ValueError('训练命令不能为空')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("训练命令不能为空")
    # 检查 `'--run-name' not in normalized_command`；根据条件结果选择对应处理分支。
    if "--run-name" not in normalized_command:
        # 抛出 `ValueError('训练命令缺少 --run-name')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("训练命令缺少 --run-name")
    # 调用 `normalized_command.index` 按当前截面统计量完成缩尾或标准化处理，并将结果保存到 `index`。
    index = normalized_command.index("--run-name")
    # 检查 `index + 1 >= len(normalized_command) or normalized_command[index + 1] != run_name`；根据条件结果选择对应处理分支。
    if index + 1 >= len(normalized_command) or normalized_command[index + 1] != run_name:
        # 抛出 `ValueError('训练命令中的 --run-name 与 run_name 不一致')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("训练命令中的 --run-name 与 run_name 不一致")
    # 调用 `_validate_session_name` 校验输入、配置或产物是否满足当前阶段的数据契约，并将结果保存到 `session_name`。
    session_name = _validate_session_name(
        screen_session_name or default_screen_session_name(run_name)
    )
    # 调用 `shutil.which` 查找外部命令在系统中的可执行路径，并将结果保存到 `screen_executable`。
    screen_executable = shutil.which("screen")
    # 检查 `screen_executable is None`；根据条件结果选择对应处理分支。
    if screen_executable is None:
        # 抛出 `FileNotFoundError('未找到 screen；无法启动持久化后台训练')`，阻止不满足数据契约的流程继续执行。
        raise FileNotFoundError("未找到 screen；无法启动持久化后台训练")

    # 调用 `_job_paths` 解析当前 run 对应的数据与输出路径，并将结果保存到 `(job_path, log_path)`。
    job_path, log_path = _job_paths(run_name, root)
    # 调用 `_read_json` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `current_job`。
    current_job = _read_json(job_path)
    # 调用 `_command_signature` 对关键配置生成稳定签名以支持复现和一致性检查，并将结果保存到 `signature`。
    signature = _command_signature(normalized_command)
    # 调用 `_screen_sessions` 查询或管理外部训练会话与进程，并将结果保存到 `sessions`。
    sessions = _screen_sessions()
    # 调用 `_training_processes` 读取或组织模型训练所需的配置与状态，并将结果保存到 `processes`。
    processes = _training_processes(run_name)
    # 检查 `session_name in sessions`；根据条件结果选择对应处理分支。
    if session_name in sessions:
        # 检查 `current_job and current_job.get('command_signature') == signature`；根据条件结果选择对应处理分支。
        if current_job and current_job.get("command_signature") == signature:
            # 调用 `training_job_status` 汇总当前任务状态与已完成进度，并将结果保存到 `result`。
            result = training_job_status(
                run_name, screen_session_name=session_name, vivit_dir=root
            )
            # 设置 `result['launch_action']` 的固定配置值，供后续流程统一使用。
            result["launch_action"] = "already_running"
            # 将 `result` 作为当前函数结果返回给调用方。
            return result
        # 抛出 `RuntimeError(f'screen 会话 {session_name!r} 已存在，但不属于当前同签名任务')`，阻止不满足数据契约的流程继续执行。
        raise RuntimeError(
            f"screen 会话 {session_name!r} 已存在，但不属于当前同签名任务"
        )
    # 检查 `processes`；根据条件结果选择对应处理分支。
    if processes:
        # 抛出 `RuntimeError(f'run {run_name!r} 已有训练进程，但不在受管 screen 会话中；为避免重复占用 GPU，本次不启动')`，阻止不满足数据契约的流程继续执行。
        raise RuntimeError(
            f"run {run_name!r} 已有训练进程，但不在受管 screen 会话中；"
            "为避免重复占用 GPU，本次不启动"
        )
    # 检查 `current_job and current_job.get('command_signature') != signature`；根据条件结果选择对应处理分支。
    if current_job and current_job.get("command_signature") != signature:
        # 检查 `not allow_command_change`；根据条件结果选择对应处理分支。
        if not allow_command_change:
            # 抛出 `ValueError('同名后台任务的训练命令签名已变化；请使用新 run name，或显式设置 OVERWRITE=1')`，阻止不满足数据契约的流程继续执行。
            raise ValueError(
                "同名后台任务的训练命令签名已变化；请使用新 run name，"
                "或显式设置 OVERWRITE=1"
            )
    # 调用 `training_job_status` 汇总当前任务状态与已完成进度，并将结果保存到 `existing`。
    existing = training_job_status(
        run_name, screen_session_name=session_name, vivit_dir=root
    )
    # 检查 `existing['completed_tasks'] == EXPECTED_TASKS and existing['prediction_files'] == EXPECTED_PRED…`；根据条件结果选择对应处理分支。
    if (
        existing["completed_tasks"] == EXPECTED_TASKS
        and existing["prediction_files"] == EXPECTED_PREDICTION_FILES
        and not restart_completed
    ):
        # 设置 `existing['launch_action']` 的固定配置值，供后续流程统一使用。
        existing["launch_action"] = "reuse_completed"
        # 将 `existing` 作为当前函数结果返回给调用方。
        return existing

    # 计算 `int((current_job or {}).get('attempt', 0)) + 1`，并将数值结果保存到 `attempt`。
    attempt = int((current_job or {}).get("attempt", 0)) + 1
    # 构造键值映射，并保存到 `payload`，供后续按键查询。
    payload = {
        "schema_version": 1,
        "run_name": run_name,
        "screen_session": session_name,
        "status": "launching",
        "command": normalized_command,
        "command_signature": signature,
        "cwd": str(root),
        "job_path": str(job_path),
        "log_path": str(log_path),
        "attempt": attempt,
        "created_at": (current_job or {}).get("created_at", _utc_now()),
        "launched_at": _utc_now(),
        "last_heartbeat": None,
        "exit_code": None,
        "error": None,
    }
    # 调用 `_atomic_write_json` 先写临时文件再原子替换正式产物，避免留下半成品。
    _atomic_write_json(job_path, payload)
    # 调用 `log_path.parent.mkdir` 创建输出目录，确保后续文件可以安全写入。
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # 尝试执行本段主逻辑；若发生异常，则交由后续异常分支处理。
    try:
        # 调用 `subprocess.run` 同步执行外部命令并收集退出状态与输出。
        subprocess.run(
            [
                screen_executable,
                "-dmS",
                session_name,
                "-L",
                "-Logfile",
                str(log_path),
                sys.executable,
                str(SCRIPT_PATH),
                "run-job",
                "--job-file",
                str(job_path),
            ],
            cwd=root,
            check=True,
            text=True,
            capture_output=True,
        )
    # 捕获 `except BaseException as error` 对应的异常，并执行明确的降级、记录或清理逻辑。
    except BaseException as error:
        # 用本轮键值结果更新 `payload`，补齐后续查询所需字段。
        payload.update(
            {
                "status": "failed",
                "error": f"screen launch failed: {type(error).__name__}: {error}",
                "finished_at": _utc_now(),
            }
        )
        # 调用 `_atomic_write_json` 先写临时文件再原子替换正式产物，避免留下半成品。
        _atomic_write_json(job_path, payload)
        # 抛出 `空值`，阻止不满足数据契约的流程继续执行。
        raise
    # 调用 `time.sleep` 短暂等待外部任务状态更新，避免无间隔轮询。
    time.sleep(0.2)
    # 调用 `training_job_status` 汇总当前任务状态与已完成进度，并将结果保存到 `result`。
    result = training_job_status(
        run_name, screen_session_name=session_name, vivit_dir=root
    )
    # 设置 `result['launch_action']` 的固定配置值，供后续流程统一使用。
    result["launch_action"] = "launched"
    # 将 `result` 作为当前函数结果返回给调用方。
    return result


# 定义 `run_job` 函数，封装这一阶段可独立调用的处理步骤。
def run_job(job_file: Path | str) -> int:
    """在 screen 内执行持久化任务清单，并记录心跳、退出码和异常。
    
    输入：
        job_file: Path | str。
    输出：
        int。
        """
    # 调用 `Path(job_file).resolve` 规范化文件路径，并将结果保存到 `path`。
    path = Path(job_file).resolve()
    # 调用 `_read_json` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `payload`。
    payload = _read_json(path)
    # 检查 `payload is None`；根据条件结果选择对应处理分支。
    if payload is None:
        # 抛出 `FileNotFoundError(path)`，阻止不满足数据契约的流程继续执行。
        raise FileNotFoundError(path)
    # 用本轮键值结果更新 `payload`，补齐后续查询所需字段。
    payload.update(
        {
            "status": "running",
            "runner_pid": os.getpid(),
            "started_at": _utc_now(),
            "last_heartbeat": _utc_now(),
            "exit_code": None,
            "error": None,
        }
    )
    # 调用 `_atomic_write_json` 先写临时文件再原子替换正式产物，避免留下半成品。
    _atomic_write_json(path, payload)
    # 设置 `process` 的固定配置值，供后续流程统一使用。
    process: subprocess.Popen[str] | None = None
    # 尝试执行本段主逻辑；若发生异常，则交由后续异常分支处理。
    try:
        # 复制 `os.environ.copy`，避免后续更新污染原对象，并保存到 `environment`。
        environment = os.environ.copy()
        # 设置 `environment['PYTHONUNBUFFERED']` 的固定配置值，供后续流程统一使用。
        environment["PYTHONUNBUFFERED"] = "1"
        # 调用 `subprocess.Popen` 启动独立训练子进程并保留进程句柄，并将结果保存到 `process`。
        process = subprocess.Popen(
            payload["command"],
            cwd=payload["cwd"],
            env=environment,
            stdin=subprocess.DEVNULL,
            text=True,
        )
        # 读取 `process.pid` 的当前值，并赋给 `payload['orchestrator_pid']`。
        payload["orchestrator_pid"] = process.pid
        # 调用 `_atomic_write_json` 先写临时文件再原子替换正式产物，避免留下半成品。
        _atomic_write_json(path, payload)
        # 只要 `True` 成立就继续循环，直至满足退出条件。
        while True:
            # 尝试执行本段主逻辑；若发生异常，则交由后续异常分支处理。
            try:
                # 调用 `process.wait` 等待子进程退出或达到超时阈值，并将结果保存到 `code`。
                code = process.wait(timeout=HEARTBEAT_SECONDS)
                # 终止当前循环，避免继续处理已经不满足条件的元素。
                break
            # 捕获 `except subprocess.TimeoutExpired` 对应的异常，并执行明确的降级、记录或清理逻辑。
            except subprocess.TimeoutExpired:
                # 调用 `_utc_now` 执行 `_utc_now` 对应的明确变换，并产生后续步骤使用的结果，并将结果保存到 `payload['last_heartbeat']`。
                payload["last_heartbeat"] = _utc_now()
                # 调用 `_atomic_write_json` 先写临时文件再原子替换正式产物，避免留下半成品。
                _atomic_write_json(path, payload)
        # 用本轮键值结果更新 `payload`，补齐后续查询所需字段。
        payload.update(
            {
                "status": "completed" if code == 0 else "failed",
                "exit_code": int(code),
                "finished_at": _utc_now(),
                "last_heartbeat": _utc_now(),
            }
        )
        # 调用 `_atomic_write_json` 先写临时文件再原子替换正式产物，避免留下半成品。
        _atomic_write_json(path, payload)
        # 将 `int(code)` 作为当前函数结果返回给调用方。
        return int(code)
    # 捕获 `except BaseException as error` 对应的异常，并执行明确的降级、记录或清理逻辑。
    except BaseException as error:
        # 检查 `process is not None and process.poll() is None`；根据条件结果选择对应处理分支。
        if process is not None and process.poll() is None:
            # 调用 `process.terminate` 终止不再继续等待的外部任务。
            process.terminate()
        # 用本轮键值结果更新 `payload`，补齐后续查询所需字段。
        payload.update(
            {
                "status": "failed",
                "exit_code": None,
                "error": f"{type(error).__name__}: {error}",
                "finished_at": _utc_now(),
                "last_heartbeat": _utc_now(),
            }
        )
        # 调用 `_atomic_write_json` 先写临时文件再原子替换正式产物，避免留下半成品。
        _atomic_write_json(path, payload)
        # 抛出 `空值`，阻止不满足数据契约的流程继续执行。
        raise


# 定义 `parse_args` 函数，封装这一阶段可独立调用的处理步骤。
def parse_args() -> argparse.Namespace:
    """解析并校验命令行参数。
    
    输入：
        无显式输入参数。
    输出：
        argparse.Namespace。
        """
    # 调用 `argparse.ArgumentParser` 创建命令行参数解析器，并将结果保存到 `parser`。
    parser = argparse.ArgumentParser(description=__doc__)
    # 调用 `parser.add_subparsers` 创建子命令分发器以区分不同运行模式，并将结果保存到 `subparsers`。
    subparsers = parser.add_subparsers(dest="command", required=True)
    # 调用 `subparsers.add_parser` 注册一个可独立调用的命令行子命令，并将结果保存到 `runner`。
    runner = subparsers.add_parser("run-job")
    # 调用 `runner.add_argument` 向命令行解析器注册当前流程使用的参数。
    runner.add_argument("--job-file", type=Path, required=True)
    # 将 `parser.parse_args()` 作为当前函数结果返回给调用方。
    return parser.parse_args()


# 定义命令行入口，解析参数并按用户选择调度当前阶段的主流程。
def main() -> int:
    """根据命令行子命令分派数据、训练、验证或评价流程。
    
    输入：
        无显式输入参数。
    输出：
        int。
        """
    # 调用 `parse_args` 解析命令行文本并生成结构化参数，并将结果保存到 `args`。
    args = parse_args()
    # 检查 `args.command == 'run-job'`；根据条件结果选择对应处理分支。
    if args.command == "run-job":
        # 将 `run_job(args.job_file)` 作为当前函数结果返回给调用方。
        return run_job(args.job_file)
    # 抛出 `ValueError(args.command)`，阻止不满足数据契约的流程继续执行。
    raise ValueError(args.command)


# 检查 `__name__ == '__main__'`；根据条件结果选择对应处理分支。
if __name__ == "__main__":
    # 抛出 `SystemExit(main())`，阻止不满足数据契约的流程继续执行。
    raise SystemExit(main())
