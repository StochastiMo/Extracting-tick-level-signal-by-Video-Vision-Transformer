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


VIVIT_DIR = Path(__file__).resolve().parent  # 定位vivit目录。
SCRIPT_PATH = Path(__file__).resolve()  # 定位script路径。
FORECAST_YEARS = tuple(range(2018, 2026))  # 集中声明forecast年份。
SEEDS = (42, 43, 44)  # 设置随机种子。
EXPECTED_TASKS = len(FORECAST_YEARS) * len(SEEDS)  # 设置预期tasks。
EXPECTED_PREDICTION_FILES = len(FORECAST_YEARS) - 1  # 设置预期预测files。
HEARTBEAT_SECONDS = 30.0  # 设置heartbeatseconds。
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")  # 编译safe名称匹配规则。
_EPOCH_PATTERN = re.compile(r"epoch=(\d+)")  # 编译轮次pattern匹配规则。


def _utc_now() -> str:  # 返回带 UTC 时区的当前时间字符串。
    """返回带 UTC 时区的当前时间字符串。"""
    return datetime.now(timezone.utc).isoformat()  # 返回带 UTC 时区的当前时间字符串。


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:  # 通过临时文件和原子替换写入产物，避免产生半成品。
    """通过临时文件和原子替换写入产物，避免产生半成品。"""
    path.parent.mkdir(parents=True, exist_ok=True)  # 确保目标目录存在。
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")  # 构造临时文件。
    temporary.write_text(  # 把文本结果写入临时或目标文件。
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)  # 用临时文件原子替换正式结果。


def _read_json(path: Path) -> dict[str, Any] | None:  # 读取对应产物，并按接口约定处理缺失文件。
    """读取对应产物，并按接口约定处理缺失文件。"""
    if not path.exists():  # 检查目标文件或目录是否满足存在性要求。
        return None  # 当前分支无可用结果，返回空值。
    return json.loads(path.read_text(encoding="utf-8"))  # 解析并返回 JSON 文件内容。


def default_screen_session_name(run_name: str) -> str:  # 为一个 run 生成确定、合法且长度受限的 screen 会话名。
    """为一个 run 生成确定、合法且长度受限的 screen 会话名。"""
    cleaned = _SAFE_NAME.sub("_", str(run_name).strip()).strip("._-")
    if not cleaned:  # 检查必需数据、配置或校验结果是否缺失。
        raise ValueError("run_name 无法生成有效的 screen 会话名")  # 输入数据不满足业务或数值约束时立即终止。
    value = f"vivit_{cleaned}"
    return value[:80]  # 返回为一个 run 生成确定、合法且长度受限的 screen 会话名。


def _validate_session_name(session_name: str) -> str:  # 校验输入形状、业务键、有限性和签名约束。
    """校验输入形状、业务键、有限性和签名约束；异常时明确失败。"""
    value = str(session_name).strip()
    if not value or _SAFE_NAME.search(value):  # 检查必需数据、配置或校验结果是否缺失。
        raise ValueError(  # 输入数据不满足业务或数值约束时立即终止。
            "SCREEN_SESSION_NAME 只能包含字母、数字、下划线、连字符和句点"
        )
    return value  # 返回值。


def _job_paths(run_name: str, vivit_dir: Path) -> tuple[Path, Path]:  # 返回指定 run 的后台任务清单和总控日志路径。
    """返回指定 run 的后台任务清单和总控日志路径。"""
    root = vivit_dir / "data" / "quality" / "runs" / run_name / "background_job"
    return root / "job.json", root / "orchestrator.log"  # 返回指定 run 的后台任务清单和总控日志路径。


def _command_signature(command: Sequence[str]) -> str:  # 构造或签名后台任务使用的命令行。
    """构造或签名后台任务使用的命令行。"""
    encoded = json.dumps(list(command), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()  # 返回稳定的 SHA-256 十六进制摘要。


def _screen_sessions() -> dict[str, str]:  # 读取 screen 会话并解析其运行状态。
    """读取 screen 会话并解析其运行状态。"""
    executable = shutil.which("screen")
    if executable is None:  # 检查所需对象是否尚未初始化。
        return {}  # 返回读取并校验后的结果。
    result = subprocess.run(  # 执行命令并收集退出状态。
        [executable, "-ls"], text=True, capture_output=True, check=False
    )
    sessions: dict[str, str] = {}  # 初始化sessions容器。
    for line in (result.stdout + result.stderr).splitlines():  # 遍历结果stdout结果stderrsplitlines，逐项处理line。
        match = re.search(r"\s*(\d+\.\S+)\s+\(", line)
        if match:  # 根据当前数据状态选择处理分支。
            full_name = match.group(1)
            sessions[full_name.split(".", 1)[1]] = full_name
    return sessions  # 返回sessions。


def _training_processes(run_name: str) -> list[dict[str, Any]]:  # 执行该模式下的训练步骤，并返回损失与诊断信息。
    """执行该模式下的训练步骤，并返回损失与诊断信息。"""
    result = subprocess.run(  # 执行命令并收集退出状态。
        ["ps", "-eo", "pid=,args="], text=True, capture_output=True, check=True
    )
    processes: list[dict[str, Any]] = []  # 初始化processes容器。
    for line in result.stdout.splitlines():  # 遍历结果stdoutsplitlines，逐项处理line。
        stripped = line.strip()
        if not stripped:  # 检查必需数据、配置或校验结果是否缺失。
            continue  # 跳过当前项并处理下一项。
        pid_text, _, command = stripped.partition(" ")
        if not pid_text.isdigit():  # 检查必需数据、配置或校验结果是否缺失。
            continue  # 跳过当前项并处理下一项。
        if not (  # 检查必需数据、配置或校验结果是否缺失。
            "stage05_baseline_training.py" in command
            or "stage06_barra_training.py" in command
        ):
            continue  # 跳过当前项并处理下一项。
        run_match = re.search(r"--run-name\s+(\S+)", command)
        if run_match is None or run_match.group(1) != run_name:  # 检查所需对象是否尚未初始化。
            continue  # 跳过当前项并处理下一项。
        year_match = re.search(r"--worker-year\s+(\d+)", command)
        seed_match = re.search(r"--worker-seed\s+(\d+)", command)
        processes.append(  # 把当前结果追加到汇总列表。
            {
                "pid": int(pid_text),  # 记录pid。
                "role": (  # 记录role。
                    "worker"
                    if re.search(r"\s(?:worker|ddp-worker)\s", f" {command} ")  # 根据当前数据状态选择处理分支。
                    else "orchestrator"
                ),
                "forecast_year": int(year_match.group(1)) if year_match else None,  # 记录forecast年份。
                "seed": int(seed_match.group(1)) if seed_match else None,  # 记录随机种子。
            }
        )
    return processes  # 返回processes。


def _last_logged_epoch(path: Path) -> int | None:  # 从 checkpoint 或日志提取训练任务进度。
    """从 checkpoint 或日志提取训练任务进度。"""
    if not path.exists() or path.stat().st_size == 0:  # 检查目标文件或目录是否满足存在性要求。
        return None  # 当前分支无可用结果，返回空值。
    with path.open("rb") as handle:  # 按指定模式打开文件并确保退出时关闭句柄。
        handle.seek(max(0, path.stat().st_size - 16384))
        tail = handle.read().decode("utf-8", errors="replace")
    matches = _EPOCH_PATTERN.findall(tail)
    return int(matches[-1]) if matches else None  # 返回完成投影或提取后的结果。


def _task_progress(run_name: str, vivit_dir: Path) -> list[dict[str, Any]]:  # 从 checkpoint 或日志提取训练任务进度。
    """从 checkpoint 或日志提取训练任务进度。"""
    model_root = vivit_dir / "models" / "runs" / run_name  # 构造模型根目录。
    quality_root = vivit_dir / "data" / "quality" / "runs" / run_name  # 构造质量根目录。
    processes = {  # 初始化processes容器。
        (item["forecast_year"], item["seed"]): item["pid"]
        for item in _training_processes(run_name)  # 遍历训练processes执行名称，逐项处理item。
        if item["role"] == "worker"  # 按当前模式、标识或状态选择处理分支。
    }
    records: list[dict[str, Any]] = []  # 初始化records容器。
    for year in FORECAST_YEARS:  # 遍历forecast年份，逐项处理年份。
        for seed in SEEDS:  # 遍历随机种子，逐项处理随机种子。
            task_root = model_root / f"forecast_{year}" / f"seed_{seed}"  # 构造task根目录。
            completed = _read_json(task_root / "completed.json")  # 读取completed。
            last_epoch = None  # 先把last轮次初始化为空，后续按条件填充。
            if completed is not None:  # 检查必需数据、配置或校验结果是否缺失。
                last_epoch = completed.get("summary", {}).get("last_epoch")
                state = "completed"
            else:  # 处理上述条件均不成立的情况。
                log_name = f"forecast_{year}.seed_{seed}.log"
                last_epoch = _last_logged_epoch(
                    quality_root / "style_task_logs" / log_name
                )
                if last_epoch is None:  # 检查所需对象是否尚未初始化。
                    last_epoch = _last_logged_epoch(
                        quality_root / "a2_task_logs" / log_name
                    )
                if last_epoch is None:  # 检查所需对象是否尚未初始化。
                    last_epoch = _last_logged_epoch(
                        quality_root / "ddp_task_logs" / log_name
                    )
                if (year, seed) in processes:  # 检查当前值是否属于允许集合。
                    state = "running"
                elif (task_root / "last.pt").exists():  # 检查上一条件不成立时的备选分支。
                    state = "resumable"
                else:  # 处理上述条件均不成立的情况。
                    state = "queued"
            records.append(  # 把当前结果追加到汇总列表。
                {
                    "forecast_year": year,  # 记录forecast年份。
                    "seed": seed,  # 记录随机种子。
                    "state": state,  # 记录状态。
                    "last_epoch": last_epoch,  # 记录最后一个轮次。
                    "pid": processes.get((year, seed)),  # 记录pid。
                }
            )
    return records  # 返回记录。


def training_job_status(  # 汇总持久化任务状态、screen 存活情况及各年度 checkpoint 进度。
    run_name: str,  # 接收执行名称。
    *,
    screen_session_name: str | None = None,  # 接收screensession名称。
    vivit_dir: Path | str = VIVIT_DIR,  # 接收vivit目录。
) -> dict[str, Any]:  # 声明当前函数的返回类型。
    """汇总持久化任务状态、screen 存活情况及各年度 checkpoint 进度。"""
    root = Path(vivit_dir).resolve()  # 构造根目录。
    job_path, log_path = _job_paths(run_name, root)  # 构造job路径。
    job = _read_json(job_path) or {}  # 读取job。
    session_name = screen_session_name or job.get("screen_session")
    if session_name is None:  # 检查所需对象是否尚未初始化。
        session_name = default_screen_session_name(run_name)
    session_name = _validate_session_name(session_name)
    sessions = _screen_sessions()
    processes = _training_processes(run_name)
    tasks = _task_progress(run_name, root)
    completed_tasks = sum(item["state"] == "completed" for item in tasks)  # 配置completedtasks。
    resumable_tasks = sum(item["state"] == "resumable" for item in tasks)  # 配置resumabletasks。
    running_tasks = sum(item["state"] == "running" for item in tasks)  # 配置runningtasks。
    screen_alive = session_name in sessions  # 计算screenalive判定条件。

    recorded_state = job.get("status")
    prediction_files = len(
        list(
            (
                root / "data" / "predictions" / "runs" / run_name
            ).glob("vivit_predictions.*.parquet")
        )
    )
    if screen_alive or processes:  # 根据当前数据状态选择处理分支。
        state = "running" if job else "running_unmanaged"
    elif recorded_state == "failed":  # 检查上一条件不成立时的备选分支。
        state = "failed"
    elif recorded_state == "completed":  # 检查上一条件不成立时的备选分支。
        state = (
            "completed"
            if completed_tasks == EXPECTED_TASKS  # 按当前模式、标识或状态选择处理分支。
            and prediction_files == EXPECTED_PREDICTION_FILES
            else "incomplete_outputs"
        )
    elif recorded_state in {"launching", "running"}:  # 检查上一条件不成立时的备选分支。
        state = "interrupted"
    elif (  # 检查上一条件不成立时的备选分支。
        completed_tasks == EXPECTED_TASKS  # 配置completedtasks。
        and prediction_files == EXPECTED_PREDICTION_FILES
    ):
        state = "completed_existing"
    elif completed_tasks or resumable_tasks:  # 检查上一条件不成立时的备选分支。
        state = "resumable"
    else:  # 处理上述条件均不成立的情况。
        state = "not_started"

    return {  # 返回汇总持久化任务状态、screen 存活情况及各年度 checkpoint 进度。
        "run_name": run_name,  # 记录运行名称。
        "status": state,  # 记录状态。
        "screen_session": session_name,  # 记录screensession。
        "screen_alive": screen_alive,  # 记录screenalive。
        "completed_tasks": completed_tasks,  # 记录completedtasks。
        "expected_tasks": EXPECTED_TASKS,  # 记录预期tasks。
        "running_tasks": running_tasks,  # 记录runningtasks。
        "resumable_tasks": resumable_tasks,  # 记录resumabletasks。
        "prediction_files": prediction_files,  # 记录预测files列表。
        "job_path": str(job_path),  # 记录job路径。
        "log_path": str(log_path),  # 记录log路径。
        "last_heartbeat": job.get("last_heartbeat"),  # 记录最后一个heartbeat。
        "exit_code": job.get("exit_code"),  # 记录exit代码。
        "error": job.get("error"),  # 记录误差。
        "processes": processes,  # 记录processes。
        "tasks": tasks,  # 记录tasks。
    }


def format_training_job_status(status: dict[str, Any]) -> str:  # 把后台任务状态格式化为 Notebook02 展示的紧凑文本。
    """把后台任务状态格式化为 Notebook02 展示的紧凑文本。"""
    lines = [  # 初始化lines容器。
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
    if status.get("last_heartbeat"):  # 根据当前数据状态选择处理分支。
        lines.append(f"last_heartbeat: {status['last_heartbeat']}")  # 把当前结果追加到汇总列表。
    if status.get("exit_code") is not None:  # 检查必需数据、配置或校验结果是否缺失。
        lines.append(f"exit_code: {status['exit_code']}")  # 把当前结果追加到汇总列表。
    if status.get("error"):  # 根据当前数据状态选择处理分支。
        lines.append(f"error: {status['error']}")  # 把当前结果追加到汇总列表。
    active = [item for item in status["tasks"] if item["state"] == "running"]  # 配置active。
    for item in active:  # 遍历active，逐项处理item。
        lines.append(  # 把当前结果追加到汇总列表。
            "active: "
            f"{item['forecast_year']}/seed={item['seed']} "
            f"epoch={item['last_epoch']} pid={item['pid']}"
        )
    return "\n".join(lines)  # 返回完成上述格式转换后的结果。


def tail_training_job_log(  # 读取后台总控日志末尾若干行。
    run_name: str,  # 接收执行名称。
    *,
    lines: int = 30,  # 接收lines。
    vivit_dir: Path | str = VIVIT_DIR,  # 接收vivit目录。
) -> str:  # 声明当前函数的返回类型。
    """读取后台总控日志末尾若干行。"""
    if lines < 0:  # 检查当前数值是否越过业务或数值阈值。
        raise ValueError("lines 必须 >= 0")  # 输入数据不满足业务或数值约束时立即终止。
    _, log_path = _job_paths(run_name, Path(vivit_dir).resolve())  # 构造。
    if lines == 0 or not log_path.exists():  # 检查目标文件或目录是否满足存在性要求。
        return ""  # 返回读取并校验后的结果。
    return "\n".join(  # 返回读取并校验后的结果。
        log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    )


def launch_training_job(  # 在独立 screen 中启动 A2 总控，或复用已有任务状态。
    command: Sequence[str],  # 接收command。
    run_name: str,  # 接收执行名称。
    *,
    screen_session_name: str | None = None,  # 接收screensession名称。
    allow_command_change: bool = False,  # 接收allowcommandchange。
    restart_completed: bool = False,  # 接收restartcompleted。
    vivit_dir: Path | str = VIVIT_DIR,  # 接收vivit目录。
) -> dict[str, Any]:  # 声明当前函数的返回类型。
    """在独立 screen 中启动 A2 总控，或复用已有任务状态。"""
    root = Path(vivit_dir).resolve()  # 构造根目录。
    normalized_command = [str(value) for value in command]  # 初始化标准化command容器。
    if not normalized_command:  # 检查必需数据、配置或校验结果是否缺失。
        raise ValueError("训练命令不能为空")  # 输入数据不满足业务或数值约束时立即终止。
    if "--run-name" not in normalized_command:  # 检查当前值是否属于允许集合。
        raise ValueError("训练命令缺少 --run-name")  # 输入数据不满足业务或数值约束时立即终止。
    index = normalized_command.index("--run-name")  # 仅使用当日截面统计量。
    if index + 1 >= len(normalized_command) or normalized_command[index + 1] != run_name:  # 检查当前样本数量是否满足计算要求。
        raise ValueError("训练命令中的 --run-name 与 run_name 不一致")  # 输入数据不满足业务或数值约束时立即终止。
    session_name = _validate_session_name(
        screen_session_name or default_screen_session_name(run_name)
    )
    screen_executable = shutil.which("screen")
    if screen_executable is None:  # 检查所需对象是否尚未初始化。
        raise FileNotFoundError("未找到 screen；无法启动持久化后台训练")  # 目标文件不存在时立即终止。

    job_path, log_path = _job_paths(run_name, root)  # 构造job路径。
    current_job = _read_json(job_path)  # 读取当前job。
    signature = _command_signature(normalized_command)  # 生成数据复用签名。
    sessions = _screen_sessions()
    processes = _training_processes(run_name)
    if session_name in sessions:  # 检查当前值是否属于允许集合。
        if current_job and current_job.get("command_signature") == signature:  # 按当前模式、标识或状态选择处理分支。
            result = training_job_status(
                run_name, screen_session_name=session_name, vivit_dir=root
            )
            result["launch_action"] = "already_running"
            return result  # 返回结果。
        raise RuntimeError(  # 当前运行环境或任务状态不支持继续执行。
            f"screen 会话 {session_name!r} 已存在，但不属于当前同签名任务"
        )
    if processes:  # 根据当前数据状态选择处理分支。
        raise RuntimeError(  # 当前运行环境或任务状态不支持继续执行。
            f"run {run_name!r} 已有训练进程，但不在受管 screen 会话中；"
            "为避免重复占用 GPU，本次不启动"
        )
    if current_job and current_job.get("command_signature") != signature:  # 按当前模式、标识或状态选择处理分支。
        if not allow_command_change:  # 检查必需数据、配置或校验结果是否缺失。
            raise ValueError(  # 输入数据不满足业务或数值约束时立即终止。
                "同名后台任务的训练命令签名已变化；请使用新 run name，"
                "或显式设置 OVERWRITE=1"
            )
    existing = training_job_status(
        run_name, screen_session_name=session_name, vivit_dir=root
    )
    if (  # 开始组合多项有效性或配置条件。
        existing["completed_tasks"] == EXPECTED_TASKS
        and existing["prediction_files"] == EXPECTED_PREDICTION_FILES
        and not restart_completed
    ):
        existing["launch_action"] = "reuse_completed"
        return existing  # 返回existing。

    attempt = int((current_job or {}).get("attempt", 0)) + 1
    payload = {  # 初始化载荷容器。
        "schema_version": 1,  # 记录schema版本。
        "run_name": run_name,  # 记录运行名称。
        "screen_session": session_name,  # 记录screensession。
        "status": "launching",  # 把 status 映射为"launching"。
        "command": normalized_command,  # 记录command。
        "command_signature": signature,  # 记录command签名，用于来源一致性校验。
        "cwd": str(root),  # 记录cwd。
        "job_path": str(job_path),  # 记录job路径。
        "log_path": str(log_path),  # 记录log路径。
        "attempt": attempt,  # 记录attempt。
        "created_at": (current_job or {}).get("created_at", _utc_now()),  # 记录createdat。
        "launched_at": _utc_now(),  # 记录launchedat。
        "last_heartbeat": None,  # 记录最后一个heartbeat。
        "exit_code": None,  # 记录exit代码。
        "error": None,  # 记录误差。
    }
    _atomic_write_json(job_path, payload)  # 原子落盘，避免半成品。
    log_path.parent.mkdir(parents=True, exist_ok=True)  # 确保目标目录存在。
    try:  # 执行可能失败的外部读取或计算步骤。
        subprocess.run(  # 执行命令并收集退出状态。
            [
                screen_executable,  # 汇入screenexecutable。
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
            cwd=root,  # 配置cwd。
            check=True,  # 配置检查。
            text=True,  # 配置text。
            capture_output=True,  # 配置capture输出。
        )
    except BaseException as error:  # 捕获预期异常并执行对应恢复逻辑。
        payload.update(  # 用当前配置更新参数字典。
            {
                "status": "failed",  # 把 status 映射为"failed"。
                "error": f"screen launch failed: {type(error).__name__}: {error}",  # 记录误差。
                "finished_at": _utc_now(),  # 记录finishedat。
            }
        )
        _atomic_write_json(job_path, payload)  # 原子落盘，避免半成品。
        raise
    time.sleep(0.2)
    result = training_job_status(
        run_name, screen_session_name=session_name, vivit_dir=root
    )
    result["launch_action"] = "launched"
    return result  # 返回结果。


def run_job(job_file: Path | str) -> int:  # 在 screen 内执行持久化任务清单，并记录心跳、退出码和异常。
    """在 screen 内执行持久化任务清单，并记录心跳、退出码和异常。"""
    path = Path(job_file).resolve()  # 构造路径。
    payload = _read_json(path)  # 读取载荷。
    if payload is None:  # 检查所需对象是否尚未初始化。
        raise FileNotFoundError(path)  # 目标文件不存在时立即终止。
    payload.update(  # 用当前配置更新参数字典。
        {
            "status": "running",  # 把 status 映射为"running"。
            "runner_pid": os.getpid(),  # 记录runnerpid。
            "started_at": _utc_now(),  # 记录startedat。
            "last_heartbeat": _utc_now(),  # 记录最后一个heartbeat。
            "exit_code": None,  # 记录exit代码。
            "error": None,  # 记录误差。
        }
    )
    _atomic_write_json(path, payload)  # 原子落盘，避免半成品。
    process: subprocess.Popen[str] | None = None  # 接收process。
    try:  # 执行可能失败的外部读取或计算步骤。
        environment = os.environ.copy()  # 转换environment的数据类型或存储位置。
        environment["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(  # 启动独立训练任务。
            payload["command"],
            cwd=payload["cwd"],  # 配置cwd。
            env=environment,  # 配置env。
            stdin=subprocess.DEVNULL,  # 配置stdin。
            text=True,  # 配置text。
        )
        payload["orchestrator_pid"] = process.pid
        _atomic_write_json(path, payload)  # 原子落盘，避免半成品。
        while True:  # 在终止条件出现前持续迭代。
            try:  # 执行可能失败的外部读取或计算步骤。
                code = process.wait(timeout=HEARTBEAT_SECONDS)
                break  # 满足终止条件后退出当前循环。
            except subprocess.TimeoutExpired:  # 捕获预期异常并执行对应恢复逻辑。
                payload["last_heartbeat"] = _utc_now()
                _atomic_write_json(path, payload)  # 原子落盘，避免半成品。
        payload.update(  # 用当前配置更新参数字典。
            {
                "status": "completed" if code == 0 else "failed",  # 把 status 映射为"completed" if code == 0 else "failed"。
                "exit_code": int(code),  # 记录exit代码。
                "finished_at": _utc_now(),  # 记录finishedat。
                "last_heartbeat": _utc_now(),  # 记录最后一个heartbeat。
            }
        )
        _atomic_write_json(path, payload)  # 原子落盘，避免半成品。
        return int(code)  # 返回在 screen 内执行持久化任务清单，并记录心跳、退出码和异常。
    except BaseException as error:  # 捕获预期异常并执行对应恢复逻辑。
        if process is not None and process.poll() is None:  # 检查所需对象是否尚未初始化。
            process.terminate()
        payload.update(  # 用当前配置更新参数字典。
            {
                "status": "failed",  # 把 status 映射为"failed"。
                "exit_code": None,  # 记录exit代码。
                "error": f"{type(error).__name__}: {error}",  # 记录误差。
                "finished_at": _utc_now(),  # 记录finishedat。
                "last_heartbeat": _utc_now(),  # 记录最后一个heartbeat。
            }
        )
        _atomic_write_json(path, payload)  # 原子落盘，避免半成品。
        raise


def parse_args() -> argparse.Namespace:  # 解析并校验命令行参数。
    """解析并校验命令行参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    runner = subparsers.add_parser("run-job")
    runner.add_argument("--job-file", type=Path, required=True)  # 注册命令行参数及其默认值。
    return parser.parse_args()  # 返回解析并校验命令行参数。


def main() -> int:  # 根据命令行子命令分派数据、训练、验证或评价流程。
    """根据命令行子命令分派数据、训练、验证或评价流程。"""
    args = parse_args()
    if args.command == "run-job":  # 按命令行子命令选择执行路径。
        return run_job(args.job_file)  # 返回根据命令行子命令分派数据、训练、验证或评价流程。
    raise ValueError(args.command)  # 输入数据不满足业务或数值约束时立即终止。


if __name__ == "__main__":  # 按当前模式、标识或状态选择处理分支。
    raise SystemExit(main())  # 当前数据违反接口约定时立即终止。
